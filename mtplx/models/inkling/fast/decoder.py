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

MTPLX_INKLING_FAST=off returns every call to the reference code (a tool, not a default).
MTPLX_INKLING_FAST_STAGES (a comma list of a, bc, d, e) overrides the proven default set.
"""

from __future__ import annotations

import os

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache

from ..cache import WindowCache
from .routes import count, fast_route_counts, request_receipt
from .state import MAX_PANEL, DecodeState
from .step import REFERENCE, decode_panel, stages_for

ENV = "MTPLX_INKLING_FAST"
#: The lane kernels the product uses, each proven against the reference step (lanes/L0.md).
DEFAULT_STAGES = "bc,e"
#: forward_one's answer when the reference forward must serve the call.
DECLINED = object()


def fast_mode():
    return os.environ.get(ENV, "on").strip().lower()


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
        """The eager fast step with the proven lane kernels (reference stages off the GPU)."""
        spec = stage_spec() if mx.default_device() == mx.gpu and mx.metal.is_available() else ""
        stages = self._stages.get(spec)
        if stages is None:
            stages = self._stages[spec] = stages_for(spec) if spec else REFERENCE
        count(f"stages.{spec or 'reference'}")
        return decode_panel(self.model, ids, state, stages=stages)

    # ------------------------------------------------------------------ eligibility
    def decline(self, rows, cache, embeddings=None):
        mode = fast_mode()
        if mode in ("off", "0", "stock", "reference"):
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


def fast_health():
    return {"mode": fast_mode(), "stages": stage_spec() or "reference", "process": fast_route_counts()}


__all__ = ["DECLINED", "FastDecoder", "FastPending", "fast_health", "fast_mode", "observe_fast_request"]
