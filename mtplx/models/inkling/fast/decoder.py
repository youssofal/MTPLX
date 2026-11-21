"""The product's entry to the fast decode step: verify panels, single-token decode, commits.

``InklingRuntime`` asks ``FastDecoder`` first for every verify panel, every commit and every
single-token forward (the pending anchor, explicit AR, a depth-0 step). The decoder serves
the call through the fast step or declines it with a reason; a declined call runs the
reference code unchanged. Every served call and every decline is counted per process and
per request (``stats.inkling_fast`` in the request log), so coverage is read from the log:

    verify.w<W>          verify panels of W rows served
    decode.w1            single-token forwards served
    committed_rows       rows committed into the cache by the fast path
    stages.<spec>        steps served with that set of lane kernels (``reference`` = none)
    fallback.<reason>    calls the reference code served, and why

    engine.native.w<W>   steps replayed by the native engine (one command buffer)
    engine.mlx.<reason>  steps the MLX graph served while a program records, checks or
                         is refused; engine.decline.<reason> steps the engine never saw

MTPLX_INKLING_FAST=off returns every call to the reference code (a tool, not a default).
MTPLX_INKLING_FAST_STAGES (a comma list of a, bc, d, e) overrides the proven default set.
MTPLX_NATIVE_ENGINE=0 serves the fast step from the MLX graph (a diagnostic rollback).
"""

from __future__ import annotations

import os

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache

from ..cache import WindowCache
# First: the engine's compiled module loads before the weights and before any Metal kernel.
from . import engine_site
from .routes import count, fast_route_counts, request_receipt
from .state import MAX_PANEL, DecodeState
from .step import REFERENCE, stages_for

ENV = "MTPLX_INKLING_FAST"
#: The lane kernels the product uses, each proven bit for bit against the reference step
#: (lanes/L0.md): stage A and the expert input (L1), attention core and output (L2), experts (L3).
DEFAULT_STAGES = "a,bc,d,e"
#: forward_one's answer when the reference forward must serve the call.
DECLINED = object()


def fast_mode():
    return os.environ.get(ENV, "on").strip().lower()


def fast_enabled():
    return fast_mode() not in ("off", "0", "stock", "reference")


def stage_spec():
    return os.environ.get("MTPLX_INKLING_FAST_STAGES", DEFAULT_STAGES).strip().lower()


class FastPending:
    """What a fast verify hands to commit in place of the reference tapes."""

    def __init__(self, state, deltas, rows):
        self.state, self.deltas, self.rows = state, deltas, rows

    def __iter__(self):  # route-tape bookkeeping counts four convolution slots a visit
        return (d.taps for d in self.deltas)

    def __len__(self):
        return len(self.deltas)


class FastDecoder:
    def __init__(self, rt):
        self.rt = rt
        self.model = rt.model
        self.trunk = rt.trunk_entries
        self._stages = {}

    def step(self, ids, state):
        """The fast step with the proven lane kernels (reference stages off the GPU): replayed
        by the native engine where a program is live, the MLX graph otherwise (same bits)."""
        spec = stage_spec() if mx.default_device() == mx.gpu and mx.metal.is_available() else ""
        stages = self._stages.get(spec)
        if stages is None:
            stages = self._stages[spec] = stages_for(spec) if spec else REFERENCE
        count(f"stages.{spec or 'reference'}")
        return engine_site.decode(self, ids, state, stages, spec)

    # ------------------------------------------------------------------ eligibility
    def decline(self, rows, cache, embeddings=None):
        if not fast_enabled():
            return "disabled"
        if embeddings is not None:
            return "embeddings"
        if not 1 <= rows <= MAX_PANEL:
            return "width"
        if cache is None or len(cache) < self.trunk:
            return "cache"
        for visit, layer in enumerate(self.model.visit_order):
            kv, conv = cache[2 * visit], cache[2 * visit + 1]
            want = WindowCache if self.model.layers[layer].sliding else KVCache
            if type(kv) is not want or not isinstance(conv, ArraysCache) or kv.keys is None:
                return "layout"
            if getattr(kv, "keys", None) is not None and kv.keys.shape[0] != 1:
                return "batch"
        return None

    def _state(self, cache):
        state = DecodeState(self.model, cache[: self.trunk])
        state.prepare()
        return state

    # ------------------------------------------------------------------ verify and commit
    def verify(self, block, cache):
        ids = block if isinstance(block, mx.array) else mx.array(block)
        rows = ids.shape[-1]
        reason = self.decline(rows, cache)
        if reason is not None:
            count(f"fallback.verify.{reason}")
            return None
        state = self._state(cache)
        logits, captures, deltas = self.step(ids.reshape(-1).astype(mx.int32), state)
        count(f"verify.w{rows}")
        return logits[0], captures, FastPending(state, deltas, rows)

    def commit(self, cache, features, pending, kept, total):
        if not 1 <= kept <= total or total != pending.rows:
            raise ValueError("Inkling commits the anchor and a prefix of the verified block")
        # The rows were evaluated with the logits; nothing reads the old arrays any more,
        # so the slice updates below stay in place.
        pending.state.commit(pending.deltas, kept)
        count("committed_rows", kept)
        if self.rt.drafter is not None:
            self.rt.drafter.observe([h[:, :kept] for h in features], cache[self.trunk:])

    # ------------------------------------------------------------------ single token
    def forward_one(self, input_ids, cache, return_hidden, emit_logits, input_embeddings):
        """A one-token forward through the fast step, committed; or DECLINED."""
        if not isinstance(input_ids, mx.array) or input_ids.ndim != 2 or input_ids.shape != (1, 1):
            return DECLINED  # prompts and chunks keep the reference forward
        reason = self.decline(1, cache, input_embeddings)
        if reason is not None:
            count(f"fallback.decode.{reason}")
            return DECLINED
        state = self._state(cache)
        logits, captures, deltas = self.step(input_ids.reshape(-1).astype(mx.int32), state)
        # Evaluate before the commit so the commit's writes find the arrays unshared.
        mx.eval(logits, captures, [d.arrays() for d in deltas])
        state.commit(deltas, 1)
        count("decode.w1")
        count("committed_rows", 1)
        if self.rt.drafter is not None:
            self.rt.drafter.observe(captures, cache[self.trunk:])
        logits = logits if emit_logits else None
        if return_hidden:
            return logits, mx.concatenate(captures, axis=-1)
        return logits


def measure_round_costs(rt, *, length=1100, reps=3, seed=20261121):
    """The depth controller's table for this Mac and pack, timed at load on a synthetic state.

    ``cost_ms[k]`` is the ms of a round with k proposals (k = 0 .. drafter width - 1): the
    fast step at width k + 1 as the product serves it (replayed where its program is live),
    the drafter's k proposals on the GPU, the GPU decision and its read, the commit and the
    drafter's observe. Every piece is synchronised, so the table is what a round costs on
    the dependent chain here, including host time between launches. No request is touched.
    Returns ``(cost_ms, detail)``, or ``(None, reason)`` where the fast round does not serve.
    """
    import statistics
    import time

    from . import draft as fast_draft
    from . import round as fast_round
    from .engine_site import _synthetic_cache

    drafter, model = rt.drafter, rt.model
    if drafter is None:
        return None, "no drafter"
    if not (fast_enabled() and mx.default_device() == mx.gpu and mx.metal.is_available()):
        return None, "fast path off"

    def timed(fn):
        mx.synchronize()
        started = time.perf_counter()
        out = fn()
        mx.synchronize()
        return out, (time.perf_counter() - started) * 1e3

    def median_of(fn):
        fn()  # first call: recording, checks, kernel builds
        return statistics.median(timed(fn)[1] for _ in range(reps))

    cache = _synthetic_cache(rt, length, seed)
    state = DecodeState(model, cache[: rt.trunk_entries])
    state.prepare()
    draft_cache = drafter.make_cache()
    vocab = model.embed_tokens.weight.shape[0]
    ids = {w: mx.array([(seed * 7919 + 104729 * i) % vocab for i in range(w)], mx.int32)
           for w in range(1, drafter.width + 1)}
    outputs = {}

    def step(width):
        logits, captures, deltas = rt.fast.step(ids[width], state)
        mx.eval(logits, captures, [d.arrays() for d in deltas])
        outputs[width] = (logits, captures, deltas)

    step_ms = {w: median_of(lambda w=w: step(w)) for w in range(1, drafter.width + 1)}
    drafter.observe([h[:, :1] for h in outputs[1][1]], draft_cache)
    draft_ms, decide_ms, laws = {0: 0.0}, {0: 0.0}, {}

    def propose(k):
        proposed, law = fast_draft.propose(drafter, model, int(ids[1][0]), draft_cache, k,
                                           position=length, seed=seed, temperature=1.0)
        mx.eval(proposed, *[v for v in vars(law).values() if isinstance(v, mx.array)])
        laws[k] = (proposed, law)

    def decide(k):
        proposed, law = laws[k]
        packet = fast_round.decide(outputs[k + 1][0][0], proposed, position=length, seed=seed,
                                   temperature=1.0, draft=law)
        fast_round.read_packet(packet, k + 1)

    for k in range(1, drafter.width):
        draft_ms[k] = median_of(lambda k=k: propose(k))
        decide_ms[k] = median_of(lambda k=k: decide(k))

    def commit(kept):
        logits, captures, deltas = outputs[4]
        state.commit(deltas, kept)
        mx.eval(state.arrays())
        drafter.observe([h[:, :kept] for h in captures], draft_cache)
        mx.eval([c.state for c in draft_cache if c is not None and hasattr(c, "state")])

    commit_ms = statistics.mean(
        statistics.median(timed(lambda kept=kept: (step(4), commit(kept)))[1] - step_ms[4]
                          for _ in range(reps))
        for kept in (1, 4))
    commit_ms = max(0.0, commit_ms)
    cost = tuple(round(step_ms[k + 1] + draft_ms[k] + decide_ms[k] + commit_ms, 2)
                 for k in range(drafter.width))
    detail = {"step_ms": {w: round(v, 2) for w, v in step_ms.items()},
              "draft_ms": {k: round(v, 2) for k, v in draft_ms.items()},
              "decide_ms": {k: round(v, 2) for k, v in decide_ms.items()},
              "commit_observe_ms": round(commit_ms, 2), "length": length, "reps": reps}
    return cost, detail


def observe_fast_request(fn, rt, *args, **kwargs):
    """Run one request with its own fast-path counters in ``stats.inkling_fast``."""
    if request_receipt.get() is not None:
        return fn(rt, *args, **kwargs)
    receipt = {"mode": fast_mode(), "stages": stage_spec() or "reference"}
    token = request_receipt.set(receipt)
    try:
        output = fn(rt, *args, **kwargs)
        output.stats.inkling_fast = receipt
        return output
    finally:
        request_receipt.reset(token)


def fast_health(rt=None):
    report = {"mode": fast_mode(), "stages": stage_spec() or "reference", "process": fast_route_counts(),
              "engine": engine_site.health()}
    if rt is not None:
        report["round_cost_ms"] = getattr(rt, "round_cost_ms", None)
        report["round_cost_detail"] = getattr(rt, "round_cost_detail", None)
    return report


__all__ = ["DECLINED", "FastDecoder", "FastPending", "fast_enabled", "fast_health", "fast_mode",
           "measure_round_costs", "observe_fast_request"]
