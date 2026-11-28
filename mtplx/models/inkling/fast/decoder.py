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
from mlx_lm.models.cache import ArraysCache

from ..cache import WindowCache, is_global_bank
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


def commit_async():
    """The stock commit's graph handed to the GPU as soon as it is built (``mx.async_eval``), so it
    runs under the host's work before the next round; ``MTPLX_INKLING_COMMIT_ASYNC=0`` leaves it to
    the next round's first evaluation, as before."""
    return os.environ.get("MTPLX_INKLING_COMMIT_ASYNC", "1").strip().lower() not in ("0", "off", "false", "no", "")


def _arrays(tree):
    """The arrays in a nest of lists and tuples (cache states hold None for empty slots)."""
    out = []
    stack = [tree]
    while stack:
        item = stack.pop()
        if isinstance(item, mx.array):
            out.append(item)
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return out


def round_commit_enabled():
    """The committing round (lane L7: the one-program round also writes the kept rows):
    off unless MTPLX_INKLING_ROUND_COMMIT=1. By default the program decides the round and
    the stock commit writes the kept rows (candidate 7's served A/B, same seeds and rounds:
    the committing round was not faster warm and slower cold). The served loop and the
    load's warm-up read the same switch, so the load warms the variant that serves."""
    from ..cache import q8_enabled

    if q8_enabled():
        # 8-bit global banks: the program decides and the host commit packs the kept rows.
        return False
    return os.environ.get("MTPLX_INKLING_ROUND_COMMIT", "off").strip().lower() not in ("0", "off", "false", "no")


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
        # A replayed step runs no stage function: the site counts again the stage paths its
        # record call took (the verdicts its stages reached, not format eligibility).
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
            sliding = self.model.layers[layer].sliding
            if (type(kv) is not WindowCache if sliding else not is_global_bank(kv)) or not isinstance(conv, ArraysCache) or kv.keys is None:
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
        if commit_async():
            # Lane F1: the commit's launches start now, under the host's work before the next
            # round (the walk's bookkeeping, the stream, the next round's inputs), not when the
            # next round first evaluates its inputs. The same graph, so the same bits.
            mx.async_eval(pending.state.arrays(), _arrays([c.state for c in cache[self.trunk:]]))

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


def measure_round_costs(rt, *, length=1100, reps=5, seed=20261121, proposals=None):
    """Round costs from the round's pieces, timed at load on a synthetic state.

    ``cost_ms[k]`` is the ms of a round with k proposals (k = 0 .. drafter width - 1): the
    fast step at width k + 1 as the product serves it (replayed where its program is live),
    the drafter's k proposals on the GPU, the GPU decision and its read, the commit and the
    drafter's observe. Every piece is synchronised, so the table is what a round costs on
    the dependent chain here, including host time between launches. No request is touched.
    ``proposals`` names the counts to time (default all); k = 0, the one-row round, is always
    timed, and a count not timed is None. ``detail["commit_observe_ms"]`` is the commit and
    the drafter's observe alone. Returns ``(cost_ms, detail)``, or ``(None, reason)`` where
    the fast round does not serve.
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

    routes = {}

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
        route = engine_site.last_route()
        routes.setdefault(width, []).append(route.get("route") if route.get("reason") is None
                                            else f"{route.get('route')}:{route.get('reason')}")

    # Widths interleaved within every repetition: the GPU clock drifts between blocks of one
    # width, which made a smooth cost look like a step at one width.
    counts = tuple(range(1, drafter.width)) if proposals is None else tuple(sorted(set(proposals)))
    widths = sorted({1, *(k + 1 for k in counts)})
    for w in widths:
        step(w)  # first call: recording, checks, kernel builds
    samples = {w: [] for w in widths}
    for _ in range(reps):
        for w in widths:
            samples[w].append(timed(lambda w=w: step(w))[1])
    # The least of the samples: contention (another process, a hot GPU) only adds time, and the
    # round program's warm-up keeps its least native call too, so the two compare like for like.
    step_ms = {w: min(samples[w]) for w in widths}
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

    for k in counts:
        draft_ms[k] = median_of(lambda k=k: propose(k))
        decide_ms[k] = median_of(lambda k=k: decide(k))

    def commit(kept):
        logits, captures, deltas = outputs[4]
        state.commit(deltas, kept)
        mx.eval(state.arrays())
        drafter.observe([h[:, :kept] for h in captures], draft_cache)
        mx.eval([c.state for c in draft_cache if c is not None and hasattr(c, "state")])

    def commit_once(kept):
        step(4)  # fresh deltas, untimed
        return timed(lambda: commit(kept))[1]

    commit_ms = statistics.mean(statistics.median(commit_once(kept) for _ in range(reps)) for kept in (1, 4))
    cost = tuple(round(step_ms[k + 1] + draft_ms[k] + decide_ms[k] + commit_ms, 2) if k in draft_ms else None
                 for k in range(drafter.width))
    detail = {"step_ms": {w: round(v, 2) for w, v in step_ms.items()},
              "draft_ms": {k: round(v, 2) for k, v in draft_ms.items()},
              "decide_ms": {k: round(v, 2) for k, v in decide_ms.items()},
              "commit_observe_ms": round(commit_ms, 2), "length": length, "reps": reps,
              "step_routes": routes}
    return cost, detail


#: A table is plausible against the one-row round timed beside it when a round with a proposal
#: costs between one and two one-row rounds and each further proposal adds between
#: PER_PROPOSAL_BAND of the one-row round (tonight's fresh loads: 0.25 on Bare Speed, 0.17 on the
#: native pack; the busy load's Optimized Speed fit, 0.33, served mostly 1 and 2 proposals).
#: Outside that the timing caught something other than these rounds (another process, a hot GPU,
#: a round on MLX) and the controller would choose depths from noise: on 21 November a
#: 1-proposal round priced below the one-row round made every round 1 proposal on all three
#: packs (L5, gate item 1).
PLAUSIBLE_FIRST_MAX = 2.0
PER_PROPOSAL_BAND = (0.1, 0.3)
#: The shape served when neither timing is plausible: the one-row round plus a fifth of it a
#: proposal (lead, 11:04), said in /health and in every request.
FIXED_SHAPE_PER_PROPOSAL = 0.2


def fit_round_costs(raw):
    """``fit_round_points`` over a table indexed by proposal count (k = 0 .. len - 1)."""
    return fit_round_points(list(enumerate(raw)), len(raw))


def fit_round_points(points, size):
    """A smooth table of ``size`` entries from noisy ``(k, ms)`` timings: least squares
    ``base + per_row * k``, and a second slope past ``k = 4`` only when the timings carry a real
    step there (it must at least halve the residual and add 2 ms a row). Coefficients are
    rounded coarsely (base to 1 ms, slopes to 0.5 ms), so the next load's re-timing lands on the
    same choices and a seeded request makes the same choices after a restart."""
    import numpy as np

    k = np.asarray([p[0] for p in points], dtype=np.float64)
    y = np.asarray([p[1] for p in points], dtype=np.float64)
    hinge = 4.0
    linear = np.stack([np.ones_like(k), k], axis=1)
    coef, *_ = np.linalg.lstsq(linear, y, rcond=None)
    rms = float(np.sqrt(np.mean((linear @ coef - y) ** 2)))
    step, model = 0.0, "linear"
    if len(y) > hinge + 2 and int((k > hinge).sum()) >= 2:
        bent = np.stack([np.ones_like(k), k, np.maximum(0.0, k - hinge)], axis=1)
        coef3, *_ = np.linalg.lstsq(bent, y, rcond=None)
        rms3 = float(np.sqrt(np.mean((bent @ coef3 - y) ** 2)))
        if coef3[2] >= 2.0 and rms3 <= 0.5 * rms:
            coef, step, rms, model = coef3[:2], float(coef3[2]), rms3, "linear_with_step"
    base = float(round(float(coef[0])))
    per_row = max(0.5, round(float(coef[1]) * 2) / 2)
    step = round(step * 2) / 2
    table = tuple(round(base + per_row * i + step * max(0.0, i - hinge), 2) for i in range(size))
    return {"model": model, "base_ms": base, "per_row_ms": per_row, "step_ms_past_4": step,
            "residual_ms": round(rms, 2), "table": table}


def implausible(table):
    """Why a round-cost table cannot be this Mac's rounds, judged against its own one-row
    round ``table[0]``, or None."""
    one = table[0] if table else None
    if not one or one <= 0:
        return "no one-row round"
    if len(table) > 1 and table[1] < one:
        return f"one proposal costs {table[1]} ms, less than the one-row round's {one} ms"
    if len(table) > 1 and table[1] > PLAUSIBLE_FIRST_MAX * one:
        return f"one proposal costs {table[1]} ms, over {PLAUSIBLE_FIRST_MAX} times the one-row round's {one} ms"
    low, high = PER_PROPOSAL_BAND
    for k in range(2, len(table)):
        added = table[k] - table[k - 1]
        if not low * one <= added <= high * one:
            return (f"proposal {k} adds {round(added, 2)} ms, outside {low} to {high} of the one-row "
                    f"round's {one} ms")
    return None


def fixed_shape(one_row_ms, size):
    """The one-row round plus FIXED_SHAPE_PER_PROPOSAL of it a proposal, ``size`` entries."""
    return tuple(_half_ms(one_row_ms * (1 + FIXED_SHAPE_PER_PROPOSAL * k)) for k in range(size))


def _half_ms(value):
    return round(float(value) * 2) / 2


def round_costs(rt):
    """The depth controller's table for this load: what a round with k proposals costs here.

    Where the round program serves: its own whole rounds from this load's warm-up
    (``round_program.warm``: the median of the native calls for each proposal count, the
    variant that serves), fitted to a line, plus the stock commit and the drafter's observe
    when the program only decides. k = 0 (the one-row round, which no program serves) and any
    count the program did not serve are timed from the pieces (``measure_round_costs``), as
    those rounds are served. Nothing is kept across loads: one timing taken while the machine
    was busy must not price every later load, and the warm-up times its rounds at every load
    anyway. A table that is not plausible against the one-row round timed beside it
    (``implausible``) is refused and said in ``detail`` (so in /health and in the request
    record); the pieces' line from this load then serves, and if that is refused too, the fixed
    shape from the one-row round (``fixed_shape``). Every load also checks its table against the
    rounds it serves (``DepthTableCheck``). Returns ``(cost_ms, detail)``; ``(None, reason)`` where
    the fast round does not serve.""" 
    width = getattr(getattr(rt, "drafter", None), "width", None)
    if not width:
        return None, "no drafter"
    warm = getattr(rt, "round_warm", None) or {}
    program = {int(k): float(v) for k, v in (warm.get("round_ms") or {}).items()
               if v is not None and 1 <= int(k) < width}
    missing = [k for k in range(1, width) if k not in program]
    raw, timing = measure_round_costs(rt, proposals=missing)
    if raw is None:
        return None, timing
    commit_in_program = round_commit_enabled()
    detail = {"one_row_ms": raw[0], "program_round_ms": program, "commit_in_program": commit_in_program,
              "pieces_ms": {k: v for k, v in enumerate(raw) if v is not None}, "timing": timing, "refused": []}
    if len(program) >= 2:
        added = 0.0 if commit_in_program else float(timing.get("commit_observe_ms") or 0.0)
        fit = fit_round_points(sorted(program.items()), width)
        line = tuple(_half_ms(raw[0]) if k == 0 else _half_ms(fit["table"][k] + added) for k in range(width))
        why = implausible(line)
        if why is None:
            # A count the program did not serve runs on the pieces: priced as it runs.
            table = tuple(line[k] if k == 0 or k in program else _half_ms(raw[k]) for k in range(width))
            return table, {**detail, "source": "program_warm", "fit": fit, "commit_added_ms": round(added, 2),
                           "pieces_for": missing}
        detail["refused"].append({"source": "program_warm", "table": line, "reason": why})
        print(f"[mtplx] inkling: round cost table from the warm-up refused ({why}); timing the pieces", flush=True)
    if any(v is None for v in raw):
        raw, timing = measure_round_costs(rt)
        if raw is None:
            return None, {**detail, "source": "rc6", "reason": timing}
        detail.update(pieces_ms=dict(enumerate(raw)), timing=timing)
    fit = fit_round_costs(raw)
    table = tuple(_half_ms(v) for v in fit["table"])
    why = implausible(table)
    if why is None:
        return table, {**detail, "source": "pieces", "fit": fit}
    detail["refused"].append({"source": "pieces", "table": table, "reason": why})
    shape = fixed_shape(detail["one_row_ms"], width)
    print(f"[mtplx] inkling: round cost table from the pieces refused ({why}); the fixed shape {shape} "
          f"prices rounds", flush=True)
    return shape, {**detail, "source": "fixed_shape"}


class DepthTableCheck:
    """The depth table against the rounds it prices, whatever made the table: over the first
    ROUNDS served rounds the controller chose, the mean ms a round at each proposal count; a
    count with at least MIN_ROUNDS rounds more than OFF_BY off its table entry is said once
    (logged, counted ``depth_table.off_by`` in the process and the request that saw it, kept for
    /health), so a wrong table shows in the first session, not in a review of receipts."""

    ROUNDS = 300
    MIN_ROUNDS = 20
    OFF_BY = 0.25

    def __init__(self, table):
        self.table = tuple(table) if table else None
        self.ms, self.rounds, self.seen, self.verdict = {}, {}, 0, None

    def observe(self, proposals, seconds):
        if self.table is None or self.verdict is not None or not 0 <= proposals < len(self.table):
            return
        self.ms[proposals] = self.ms.get(proposals, 0.0) + seconds * 1e3
        self.rounds[proposals] = self.rounds.get(proposals, 0) + 1
        self.seen += 1
        if self.seen >= self.ROUNDS:
            self.judge()

    def judge(self):
        by_count = {k: {"served_ms": round(self.ms[k] / n, 2), "table_ms": self.table[k], "rounds": n,
                        "ratio": round(self.ms[k] / n / self.table[k], 3)}
                    for k, n in sorted(self.rounds.items()) if n >= self.MIN_ROUNDS and self.table[k] > 0}
        off = {k: row for k, row in by_count.items() if abs(row["ratio"] - 1) > self.OFF_BY}
        self.verdict = {"rounds": self.seen, "by_count": by_count, "off": off}
        if off:
            count("depth_table.off_by")
            print(f"[mtplx] inkling: served rounds are more than {int(self.OFF_BY * 100)}% off the depth "
                  f"table: {off}", flush=True)
        return self.verdict

    def report(self):
        return {"rounds": self.seen, "verdict": self.verdict, "table": self.table}


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
        check = getattr(rt, "depth_table_check", None)
        report["depth_table_check"] = check.report() if check is not None else None
    return report


__all__ = ["DECLINED", "DepthTableCheck", "FastDecoder", "FastPending", "fast_enabled", "fast_health", "fast_mode",
           "fit_round_costs", "measure_round_costs", "observe_fast_request", "round_commit_enabled",
           "round_costs"]
