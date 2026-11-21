"""Inkling's decode step as a replay site of the native engine.

The fast step (``step.decode_panel``) is a pure function of arrays: the panel's token ids,
three integers of the round (committed tokens, the newest slot of the window rings, the
rows a window holds), and per layer visit the committed keys and values and the four
convolution tails. It reads them and writes nothing; it returns the logits, the drafter's
captures and each visit's own rows (deltas), which the decoder commits afterwards for the
accepted rows only.

That is what the native engine replays. The first calls of a geometry run on MLX; one is
recorded (every Metal launch of the evaluation); the next is replayed beside the MLX path
and every output is compared bit for bit; from then on the program serves: the launches
are encoded from compiled code into a command buffer of the engine's own queue, Python
builds no graph and the MLX function is not called. The MLX path stays the reference and
the fallback for any geometry whose program is not live, and every call that does not
replay is counted with its reason (``stats.inkling_fast``: ``engine.native.w<W>``,
``engine.mlx.<reason>``, ``engine.decline.<reason>``).

A geometry is the panel width and the lane set; the engine adds the inputs' layout, which
carries the global banks' capacity (it grows in steps of ``state.BANK_STEP``), so each
width has one program per capacity. ``warm(rt)`` records and checks every width from 1 to
``MAX_WARM_WIDTH`` on a synthetic state while the model loads, so a request's rounds at the
first capacity replay from the first one, and a new capacity costs one call on MLX, one
recorded call and one checked call per width (the widths' family is live by then).

``decode(decoder, ids, state, stages, spec)`` is the one call ``FastDecoder.step`` makes in
place of ``decode_panel``; it returns exactly what ``decode_panel`` returns.
"""

from __future__ import annotations

import time
from typing import Any

import mlx.core as mx

from mtplx.engine import runtime as engine_runtime
from mtplx.engine import sites
from mtplx.engine.site import load_early

from .routes import capture_counts, count, stage_paths
from .step import attention_reason, decode_panel

# The engine learns each Metal kernel when MLX compiles it; a kernel compiled before the
# compiled module is in the process is unknown to it and refuses every program that
# launches it. This module is imported with the decoder, before Inkling's weights load.
load_early()

NAME = "inkling.step"
_ATTRIBUTE = "_mtplx_inkling_step_site"
#: Widths warmed at load: one-row decode and every verify panel of the drafter's 8-slot block.
MAX_WARM_WIDTH = 8
#: Calls a warm-up gives one width to become live (warm calls, recordings, the check).
_WARM_TRIES = 10


def _weights(rt: Any) -> list:
    return sites.module_weights(getattr(rt, "model", None))


def _named_weights(rt: Any) -> list:
    return sites.named_model_arrays(rt)


def _output_name(index: int) -> str:
    if index == 0:
        return "logits"
    if index <= 4:
        return f"capture[{index - 1}]"
    visit, part = divmod(index - 5, 6)
    return f"visit[{visit}].{('keys', 'values', 'k_taps', 'v_taps', 'attn_taps', 'mlp_taps')[part]}"


def weight_tables() -> list:
    """Arrays made once from the weights and kept for the process, which a recorded launch reads
    as it reads a weight: lane L1's joined scales and biases (one bf16 buffer a layer of a 4-bit
    pack, 1.2 MB each, ``attn_input.joined_scales``). They are not module parameters, so the sites
    pin them before every recording. Unpinned, every program held all 24 as constants, snapshot
    and kept twice (about 60 MB a program on the 4-bit packs against 2 MB on the native pack):
    the 256 MB budget then held four programs a site, and the other widths were evicted and
    recorded again on use (21 November: Bare Speed's round site 16 programs built, 12 evicted
    in one served pair; the load's timing passes found their widths evicted)."""
    from .attn_input import _joined_sb

    tables = []
    for _ref, (table, _offsets) in list(_joined_sb.values()):
        if table is not None:
            tables.append(table)
    return tables


def site(rt: Any):
    """The process's step site for this runtime (made on first use)."""
    replay = getattr(rt, _ATTRIBUTE, None)
    if replay is None:
        replay = sites.site_for(
            rt, _ATTRIBUTE, NAME,
            weights=_weights, pin_operands=True, names=_named_weights,
            output_names=_output_name,
            # The step only reads its inputs, so a view (a convolution tail cut from a longer
            # array, a window right after prefill) is replayed as bound, not fenced.
            fence_strided=False,
        )
        replay.pin_before_each_recording(weight_tables)
    return replay


class RoundState:
    """The decode state as one round reads it.

    Every window ring has the same newest slot and row count and every global bank's count
    is the committed length, so the round's integers are three uint32 arrays that every
    visit shares (``DecodeState`` makes a fresh array per visit and stage). They are inputs
    of the replayed program; nothing of the round is a Python value frozen into it.
    """

    __slots__ = ("_state", "pos", "newest", "count")

    def __init__(self, state, pos, newest, count_rows):
        self._state = state
        self.pos = pos
        self.newest = newest
        self.count = count_rows

    def window_view(self, visit):
        kv = self._state.cache[2 * visit]
        return kv.keys, kv.values, self.newest, self.count

    def global_view(self, visit):
        kv = self._state.cache[2 * visit]
        return kv.keys, kv.values, self.pos

    def __getattr__(self, name):
        return getattr(self._state, name)


def round_inputs(state, ids):
    """``(RoundState, inputs)`` for one call, or ``(None, reason)``.

    Inputs in the program's order: ``ids`` (W,) int32; ``pos``, ``newest``, ``count``
    uint32 scalars; then per visit its keys, values and four convolution tails.
    """
    round_state, inputs = state_inputs(state)
    if round_state is None:
        return None, inputs
    return round_state, [ids, *inputs]


def state_inputs(state):
    """``(RoundState, inputs)`` without the ids (``pos``, ``newest``, ``count``, then per
    visit keys, values and four tails), or ``(None, reason)``."""
    window = state.window
    newest = None
    for visit, local in enumerate(state.local):
        if local:
            kv = state.cache[2 * visit]
            slot = (kv._idx - 1) % window
            if newest is None:
                newest = slot
            elif slot != newest:
                return None, "ring_skew"
    length = state.length
    pos = mx.array(length, mx.uint32)
    newest_a = mx.array(0 if newest is None else newest, mx.uint32)
    count_a = mx.array(min(length, window), mx.uint32)
    inputs = [pos, newest_a, count_a]
    for visit in range(len(state.local)):
        kv, conv = state.cache[2 * visit], state.cache[2 * visit + 1]
        tails = conv.cache if hasattr(conv, "cache") else [conv[s] for s in range(4)]
        if kv.keys is None or any(t is None for t in tails):
            return None, "state_missing"
        inputs += [kv.keys, kv.values, *tails]
    return RoundState(state, pos, newest_a, count_a), inputs


def flatten(logits, captures, deltas) -> list:
    out = [logits, *captures]
    for delta in deltas:
        out += delta.arrays()
    return out


def unflatten(outputs: list, n_captures: int):
    from .state import VisitDelta

    logits = outputs[0]
    captures = list(outputs[1 : 1 + n_captures])
    deltas = []
    rest = outputs[1 + n_captures :]
    for start in range(0, len(rest), 6):
        k, v, kt, vt, at, mt = rest[start : start + 6]
        deltas.append(VisitDelta(k, v, (kt, vt, at, mt)))
    return logits, captures, deltas


def identity(rt: Any) -> tuple:
    """The pack a program was recorded on: its path and a fingerprint of its weights.

    A site belongs to one loaded runtime, so its programs never meet other weights; the
    identity is in every key all the same, so a program is tied by name to the pack and
    the lane set it was recorded with and a report says which pack served.
    """
    replay = site(rt)
    known = getattr(replay, "_inkling_identity", None)
    if known is None:
        weights = _weights(rt)
        dtypes = sorted({str(a.dtype) for a in weights})
        known = (str(getattr(rt, "model_path", "")), len(weights), int(sum(a.nbytes for a in weights)),
                 tuple(dtypes))
        replay._inkling_identity = known
    return known


#: How the last call of this process was served (per-round attribution for the route tape).
_LAST: dict[str, Any] = {"route": None, "reason": None, "rows": 0}
#: geometry -> the stage-path counts its record call took (``routes.stage_paths``), counted
#: again on each native replay: a replayed step runs no stage function (review 2, finding 11).
_PATHS: dict = {}


def last_route() -> dict[str, Any]:
    """``{"route": native | recording | verifying | mlx | declined, "reason": ..., "rows": W}``
    for the last step served in this process: what the round's tape entry records."""
    return dict(_LAST)


def _note(route: str, reason: str | None, rows: int) -> None:
    _LAST["route"], _LAST["reason"], _LAST["rows"] = route, reason, rows


def reset_route() -> None:
    """A round that serves no step through this site records no step route (not the last one)."""
    _note(None, None, 0)


def _unfit(spec: str) -> str | None:
    lanes = {part.strip() for part in spec.split(",") if part.strip()}
    if "bc" not in lanes:
        # The reference attention reads the history back with host-side shapes that change
        # every token: not one program.
        return "reference_attention"
    declined = attention_reason()
    if declined is not None:
        # The fused attention's qualification declined on this Mac, so the step's attention is
        # the chronological one-row reference, whose shapes follow the live history: not one
        # program either (the step and the round program both decline, counted).
        return "attention_" + declined
    return None


def decode(decoder, ids, state, stages, spec):
    """``decode_panel(decoder.model, ids, state, stages=stages)``, replayed when it can be."""
    model = decoder.model
    rows = int(ids.shape[-1])
    reason = _unfit(spec)
    if reason is None and engine_runtime.mode() == "0":
        reason = "disabled"
    if reason is not None:
        count(f"engine.decline.{reason}")
        _note("declined", reason, rows)
        return decode_panel(model, ids, state, stages=stages)
    replay = site(decoder.rt)
    if replay.declined():
        count("engine.decline.unavailable")
        _note("declined", "unavailable", rows)
        return decode_panel(model, ids, state, stages=stages)
    round_state, inputs = round_inputs(state, ids)
    if round_state is None:
        reason = inputs
        replay.refuse(reason)
        count(f"engine.decline.{reason}")
        _note("declined", reason, rows)
        return decode_panel(model, ids, state, stages=stages)
    # The routes a trace bakes in (width, lane set, captures) and the pack it reads; the
    # site adds the inputs' layout, which carries the global banks' capacity.
    geometry = (rows, spec, tuple(model.capture_ids), identity(decoder.rt))
    started = time.perf_counter()
    if not replay.plan(geometry, family=geometry):
        reason = replay.last_decline or "declined"
        count(f"engine.mlx.{reason}")
        _note("mlx", reason, rows)
        return decode_panel(model, ids, round_state, stages=stages)
    ticket = replay.begin(geometry, inputs, planned=True, started=started)
    del inputs
    if ticket.native:
        outputs = ticket.outputs
        ticket.finish(outputs)
        count(f"engine.native.w{rows}")
        for name, n in _PATHS.get(geometry, {}).items():
            count(name, n)
        _note("native", None, rows)
        return unflatten(outputs, len(model.capture_ids))
    with capture_counts() as box:
        logits, captures, deltas = decode_panel(model, ids, round_state, stages=stages)
    if ticket.kind == "record":
        _PATHS[geometry] = stage_paths(box)
    # A recorded or checked call evaluates the MLX path here and closes; others do nothing.
    ticket.finish(flatten(logits, captures, deltas))
    reason = ticket.reason or ticket.kind
    count(f"engine.mlx.{reason}")
    _note({"record": "recording", "verify": "verifying"}.get(ticket.kind, "mlx"), reason, rows)
    return logits, captures, deltas


# ------------------------------------------------------------------------------- warm-up
def _synthetic_cache(rt, length: int, seed: int):
    """A trunk cache at ``length`` committed tokens with random contents, in the decode
    layout (window rings of the window size, global banks of one capacity step)."""
    from .state import BANK_STEP, round_up

    model = rt.model
    cache = model.make_cache()
    cfg = model.config
    window = cfg["sliding_window_size"]
    heads, head_dim = cfg["num_key_value_heads"], cfg["head_dim"]
    dtype = model.embed_tokens.weight.dtype
    keep = cfg["sconv_kernel_size"] - 1
    key = mx.random.key(seed)
    arrays = []
    for visit, index in enumerate(model.visit_order):
        layer = model.layers[index]
        kv, conv = cache[2 * visit], cache[2 * visit + 1]
        rows = window if layer.sliding else round_up(max(length, 1), BANK_STEP)
        key, k1, k2 = mx.random.split(key, 3)
        kv.keys = (mx.random.normal((1, heads, rows, head_dim), key=k1) * 0.5).astype(dtype)
        kv.values = (mx.random.normal((1, heads, rows, head_dim), key=k2) * 0.5).astype(dtype)
        kv.offset = length
        if layer.sliding:
            # The ring form DecodeState.prepare leaves: the slot after the newest row.
            kv._idx = (length - 1) % window + 1 if length >= window else length
        for slot, conv_layer in enumerate(layer.convs):
            key, k3 = mx.random.split(key)
            conv[slot] = mx.random.normal((1, keep, conv_layer.weight.shape[1]), key=k3) * 0.1
            arrays.append(conv[slot])
        arrays += [kv.keys, kv.values]
    mx.eval(arrays)
    return cache


def warm(rt, widths=None, *, length: int = 1100, seed: int = 20261121, stages=None, spec=None) -> dict:
    """Record and check the step's program for each width before the first request.

    Runs on a synthetic state (random contents, ``length`` committed tokens: past the
    window, inside the first capacity step). Nothing of a request is touched. Returns a
    report: per width the calls it took and the state it reached.
    """
    from .decoder import FastDecoder, stage_spec
    from .state import DecodeState
    from .step import set_activation_dtype, stages_for

    # Before the load's first decline check: the attention's qualification is asked for the
    # dtype the trunk computes in (the FP16 siblings ask float16).
    set_activation_dtype(rt.model.embed_tokens.weight.dtype)
    report: dict[str, Any] = {"widths": {}, "mode": engine_runtime.mode()}
    if engine_runtime.mode() == "0" or not engine_runtime.available():
        report["skipped"] = engine_runtime.unavailable() or "disabled"
        return report
    decoder = getattr(rt, "fast", None)
    if decoder is None:
        decoder = FastDecoder(rt)
    spec = stage_spec() if spec is None else spec
    if _unfit(spec):
        report["skipped"] = _unfit(spec)
        return report
    stages = stages_for(spec) if stages is None else stages
    widths = tuple(range(1, MAX_WARM_WIDTH + 1)) if widths is None else tuple(widths)
    started = time.perf_counter()
    cache = _synthetic_cache(rt, length, seed)
    state = DecodeState(rt.model, cache[: rt.trunk_entries])
    state.prepare()
    replay = site(rt)
    vocab = rt.model.embed_tokens.weight.shape[0]
    key = mx.random.key(seed + 1)
    for width in widths:
        entry_started = time.perf_counter()
        native = 0
        tries = 0
        while tries < _WARM_TRIES and native < 1:
            tries += 1
            key, sub = mx.random.split(key)
            ids = mx.random.randint(0, vocab, (width,), key=sub).astype(mx.int32)
            before = replay.counters["native"]
            logits, captures, deltas = decode(decoder, ids, state, stages, spec)
            mx.eval(logits, captures, [d.arrays() for d in deltas])
            native += replay.counters["native"] - before
            del logits, captures, deltas
        # The depth controller's cost table is timed once at load by
        # ``decoder.measure_round_costs`` (whole rounds through this site): not again here.
        report["widths"][width] = {"calls": tries, "live": native > 0,
                                   "seconds": round(time.perf_counter() - entry_started, 3)}
    del state, cache
    report["context_tokens"] = length
    report["seconds"] = round(time.perf_counter() - started, 3)
    report["site"] = replay.report()
    return report


def health(rt=None) -> dict:
    """The engine's state and this process's sites (for /health and the request log)."""
    return sites.process_report()


__all__ = ["MAX_WARM_WIDTH", "NAME", "RoundState", "decode", "flatten", "health", "identity", "last_route",
           "round_inputs", "site", "unflatten", "warm"]
