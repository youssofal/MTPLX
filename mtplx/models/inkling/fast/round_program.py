"""Inkling's DSpark round as one program on the native engine: one command buffer, one read.

A DSpark round of the fast path was three MLX evaluations with a host wait after each: the
drafter's ``k`` proposals (evaluated before the verify could start), the target's verify of
the ``W = k + 1`` rows, and the decision on the GPU (lane L4's ratio law), which the host
read. Here the three are one function of arrays. The native engine records it once per
geometry, checks the recording bit for bit against this same function on MLX, and from
then on replays it as one command buffer on its own queue; the host builds no graph and
reads one small packet a round. The MLX path is this same function, so a round reads the
same law whether it was replayed or not.

The drafter's arithmetic is RC6's (three blocks at the trained width of 8 slots, the
vocabulary head over the proposal rows, the low-rank Markov walk on the folded table, the
draws of ``round.sample_rows``), with one change that makes it one program: each block
attends to its cache **in place** with the row count as data, a window block to its ring
of ``window`` slots in physical order and the global block to its bank of a fixed capacity
(a multiple of ``state.BANK_STEP``), each with a mask of the valid slots, and the rope
offset is an array. RC6 instead reads the history back in temporal order at its exact
length, a shape that changes every token. The keys and values are the same; the order of
the softmax sums is not, so the drafter's bits differ from the stock drafter's in the last
place. That changes which tokens a seed draws under ratio acceptance (class B already: the
output law is the target's whatever the drafter proposes, ``lanes/L4.md``), and changes
nothing at temperature 0 (greedy tokens are the target's argmax) or under the seed-match
rule (which keeps the host path).

What is an input (all arrays, in this order): the anchor (1,) int32; the absolute position
of the first proposal (uint32); the request's seed words (2,) uint32; the law parameters of
the target rows and of the draft rows when that side samples (``round.law_params``: the
temperatures are data, so one program serves every temperature); the drafter's rope offset
(int32), its rings' newest slot and row count (uint32) and its last mixed feature; each
drafter block's keys and values; then the step's inputs (``engine_site.state_inputs``).
The proposal count, whether each side is greedy, the lane set and the pack are in the key.
Nothing is written in place: the outputs are the packet, the block of ids, the logits, the captures
and every visit's delta, and the caller commits the rows the host accepts (``rt.commit``)
exactly as after a stock round.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mtplx.engine import runtime as engine_runtime
from mtplx.engine import sites
from mtplx.engine.site import room_for_check_copies

from . import engine_site
from . import round as fast_round
from .routes import count
from .state import BANK_STEP, DecodeState, round_up
from .step import decode_panel

NAME = "inkling.round"
_ATTRIBUTE = "_mtplx_inkling_round_site"
#: Proposals a round can carry: the drafter's 8-slot block holds the anchor and 7.
MAX_PROPOSALS = 7
_WARM_TRIES = 10


# ------------------------------------------------------------------------------- the site
def _weights(rt: Any) -> list:
    arrays = sites.module_weights(getattr(rt, "model", None))
    drafter = getattr(rt, "drafter", None)
    if drafter is not None:
        arrays += sites.module_weights(drafter)
        # The folded Markov table is built once per process; the program reads it in place.
        arrays.append(drafter.fold(rt.model.unembed))
    return arrays


def _named_weights(rt: Any) -> list:
    from mlx.utils import tree_flatten

    named = sites.named_model_arrays(rt)
    drafter = getattr(rt, "drafter", None)
    if drafter is not None:
        named += [(f"drafter.{n}", a) for n, a in tree_flatten(drafter.parameters()) if hasattr(a, "nbytes")]
    return named


def _output_name(index: int) -> str:
    if index == 0:
        return "packet"
    if index == 1:
        return "ids"
    return engine_site._output_name(index - 2)


def site(rt: Any):
    replay = getattr(rt, _ATTRIBUTE, None)
    if replay is None:
        replay = sites.site_for(
            rt, _ATTRIBUTE, NAME,
            weights=_weights, pin_operands=True, names=_named_weights,
            output_names=_output_name, fence_strided=False,
        )
        # The round's kept constants (sizes, per-row parameters, modes) are bound in place.
        replay.pin_before_each_recording(fast_round.constants)
    return replay


# ------------------------------------------------------------------------- drafter state
class DraftView:
    """The drafter's caches as one round reads them: per block its keys and values, the
    round's integers as arrays, and the last mixed feature."""

    __slots__ = ("keys", "values", "offset", "newest", "count", "length", "last", "windows")

    def __init__(self, keys, values, offset, newest, count_rows, length, last, windows):
        self.keys, self.values = keys, values
        self.offset, self.newest, self.count, self.length = offset, newest, count_rows, length
        self.last, self.windows = last, windows


def _is_ring(entry, window: int) -> bool:
    if entry.keys is None or entry.keys.shape[2] != window or getattr(entry, "keep", 0):
        return False
    if entry.offset >= window:
        return 1 <= entry._idx <= window
    return entry._idx == entry.offset


def _make_ring(entry, window: int) -> None:
    """The same chronological rows as a ``window``-slot ring (RC6's window cache in the
    form the decode state keeps the target's rings: ``_idx`` is the slot after the newest)."""
    keys, values = entry._temporal_order(entry.keys), entry._temporal_order(entry.values)
    rows = min(int(entry.offset), window)
    if keys.shape[2] < rows:
        raise ValueError("a draft window cache holds fewer rows than its position needs")
    keys, values = keys[:, :, keys.shape[2] - rows:], values[:, :, values.shape[2] - rows:]
    if rows == window:
        entry.keys, entry.values = mx.contiguous(keys), mx.contiguous(values)
    else:
        shape = (keys.shape[0], keys.shape[1], window, keys.shape[3])
        ring_k, ring_v = mx.zeros(shape, keys.dtype), mx.zeros(shape, values.dtype)
        ring_k[:, :, :rows, :] = keys
        ring_v[:, :, :rows, :] = values
        entry.keys, entry.values = ring_k, ring_v
    entry._idx = rows


def _bucket(entry, length: int) -> None:
    """A global draft bank at a capacity that is a multiple of ``BANK_STEP`` (one copy),
    growing in those steps from now on (``KVCache.step``)."""
    cap = round_up(max(length, 1), BANK_STEP)
    keys, values = entry.keys, entry.values
    if keys.shape[2] != cap:
        shape = (keys.shape[0], keys.shape[1], cap, keys.shape[3])
        new_k, new_v = mx.zeros(shape, keys.dtype), mx.zeros(shape, values.dtype)
        new_k[:, :, :length, :] = keys[:, :, :length, :]
        new_v[:, :, :length, :] = values[:, :, :length, :]
        entry.keys, entry.values = new_k, new_v
    entry.step = BANK_STEP


def draft_inputs(drafter, dcache, length: int):
    """``(DraftView, inputs)`` for one round, or ``(None, reason)``. Puts the drafter's
    caches in the fixed layout first (cheap when they already are: every round but the
    first after a prefill, a multi-row commit of a window block, or a bank growth)."""
    blocks = drafter.blocks
    if dcache is None or len(dcache) != len(blocks) + 1:
        return None, "draft_cache"
    last = dcache[len(blocks)][0]
    if last is None:
        return None, "draft_empty"
    changed = []
    windows = []
    for i, block in enumerate(blocks):
        entry = dcache[i]
        if entry.keys is None:
            return None, "draft_empty"
        if int(entry.offset) != length:
            return None, "draft_offset"
        if block.window:
            if not _is_ring(entry, block.window):
                _make_ring(entry, block.window)
                changed += [entry.keys, entry.values]
        else:
            if entry.keys.shape[2] % BANK_STEP or entry.keys.shape[2] < length:
                _bucket(entry, length)
                changed += [entry.keys, entry.values]
            entry.step = BANK_STEP
        windows.append(block.window)
    if changed:
        mx.eval(changed)
    newest, rows = 0, 0
    for i, block in enumerate(blocks):
        if block.window:
            entry = dcache[i]
            slot = (entry._idx - 1) % block.window
            if rows and (slot != newest):
                return None, "draft_ring_skew"
            newest, rows = slot, min(length, block.window)
    offset = mx.array(length, mx.int32)
    newest_a = mx.array(newest, mx.uint32)
    count_a = mx.array(rows, mx.uint32)
    length_a = mx.array(length, mx.uint32)
    keys = [dcache[i].keys for i in range(len(blocks))]
    values = [dcache[i].values for i in range(len(blocks))]
    view = DraftView(keys, values, offset, newest_a, count_a, length_a, last, windows)
    inputs = [offset, newest_a, count_a, length_a, last]
    for k, v in zip(keys, values):
        inputs += [k, v]
    return view, inputs


# --------------------------------------------------------------------- the fixed drafter
def _valid(view: DraftView, index: int, rows: int) -> mx.array:
    """Which history slots of block ``index`` a proposal row reads, then the block's own rows."""
    window = view.windows[index]
    slots = view.keys[index].shape[2]
    j = mx.arange(slots, dtype=mx.uint32)
    if window:
        size = fast_round._const(slots, mx.uint32)
        age = (view.newest + size - j) % size
        history = age < view.count
    else:
        history = j < view.length
    own = fast_round._const([True] * rows, mx.bool_)
    return mx.concatenate([history, own]).reshape(1, 1, 1, slots + rows)


def _block(block, hidden, residual, view: DraftView, index: int):
    """``DraftBlock.__call__`` with its attention read in place (see the module docstring)."""
    summed = hidden if residual is None else hidden.astype(mx.float32) + residual.astype(mx.float32)
    x = block.normalize(summed, "input_layernorm").astype(hidden.dtype)
    q = block.rotate(block.normalize(block.heads(x, "q"), "self_attn.q_norm"), view.offset)
    k = block.rotate(block.normalize(block.heads(x, "k"), "self_attn.k_norm"), view.offset)
    out = mx.fast.scaled_dot_product_attention(
        q,
        mx.concatenate([view.keys[index], k], axis=2),
        mx.concatenate([view.values[index], block.heads(x, "v")], axis=2),
        scale=block.cfg["head_dim"] ** -0.5,
        mask=_valid(view, index, x.shape[1]),
    )
    from ..model import linear

    attn = linear(out.transpose(0, 2, 1, 3).reshape(*x.shape[:2], -1), block.w["self_attn.o_proj.weight"])
    summed = summed.astype(hidden.dtype).astype(mx.float32) + attn.astype(mx.float32)
    y = block.normalize(summed, "post_attention_layernorm").astype(hidden.dtype)
    gate = linear(y, block.w["mlp.gate_proj.weight"]).astype(mx.float32)
    up = linear(y, block.w["mlp.up_proj.weight"]).astype(mx.float32)
    return linear((nn.silu(gate) * up).astype(y.dtype), block.w["mlp.down_proj.weight"]), summed.astype(hidden.dtype)


def propose(drafter, target, anchor, view: DraftView, count_: int, *, position, seed, temperature=1.0,
            sample=True, cap=None, params=None):
    """``fast.draft.propose`` on the fixed-layout drafter: (ids (count,) int32, DraftLaw).
    ``params`` (``round.law_params(1, temperature, cap)``) gives a sampled law's temperature
    as data."""
    from ..model import linear

    if not 1 <= count_ < drafter.width:
        raise ValueError("DSpark proposal count exceeds its trained block")
    greedy = (not sample) or (not temperature or temperature <= 0)
    if cap is None:
        cap = target.config.get("final_logit_softcapping") or 0.0
    anchor_ids = anchor.astype(mx.int32).reshape(1, 1)
    embed = target.embed_tokens(anchor_ids)
    dtype = embed.dtype
    masks = mx.broadcast_to(drafter.mask, (1, drafter.width - 1, embed.shape[-1]))
    h, residual = mx.concatenate([embed, masks], axis=1), None
    for i, block in enumerate(drafter.blocks):
        h, residual = _block(block, h, residual, view, i)
    h = mx.fast.rms_norm(
        h.astype(mx.float32) + residual.astype(mx.float32),
        drafter.output_norm,
        drafter.cfg["rms_norm_eps"],
    ).astype(dtype)[:, 1 : count_ + 1]
    base = linear(h, target.unembed)[..., : drafter.vocab]
    rank = drafter.cfg["markov_rank"]
    weights = drafter.markov["rnn_proj.weight"].astype(mx.float32)
    inputs = mx.concatenate([h, mx.broadcast_to(view.last, h.shape)], axis=-1)
    conditioned = linear(inputs.astype(mx.float32), weights[:, 2 * rank :]) + drafter.markov[
        "rnn_proj.bias"
    ].astype(mx.float32)
    folded = drafter.fold(target.unembed)
    start = position.astype(mx.uint32).reshape(())
    state = mx.zeros((1, rank), dtype)
    prev = anchor_ids.reshape(1)
    picks, laws, stats, confidence = [], [], [], []
    for i in range(count_):
        step_in = mx.concatenate([state, drafter.markov["markov_w1.weight"][prev]], axis=-1)
        update = (linear(step_in.astype(mx.float32), weights[:, : 2 * rank]) + conditioned[:, i]).astype(dtype)
        gate, candidate, output = mx.split(update, 3, axis=-1)
        gate = mx.sigmoid(gate.astype(mx.float32)).astype(dtype)
        state = gate * state + (1 - gate) * mx.tanh(candidate.astype(mx.float32)).astype(dtype)
        signal = mx.tanh(output.astype(mx.float32)).astype(dtype)
        bias = mx.addmm(folded[:, -1], signal, folded[:, :-1].T)
        scores = base[:, i] + bias
        if greedy:
            law, st = fast_round.law_rows(scores, temperature=None, cap=0.0, write=True)
        else:
            law, st = fast_round.law_rows(scores, temperature=temperature, cap=cap, params=params)
        token, lse = fast_round.sample_rows(law, st, positions=(start + fast_round._const(i, mx.uint32)).reshape(1), seed=seed,
                                            purpose=fast_round.PURPOSE_DRAFT, greedy=greedy)
        prev = token
        picks.append(token)
        laws.append(law)
        stats.append(st)
        if not greedy:
            confidence.append((mx.take_along_axis(law, token[:, None], axis=1)[:, 0] - lse[:, 0]) - lse[:, 1])
    ids = mx.concatenate(picks)
    law = fast_round.DraftLaw(mx.concatenate(laws), mx.concatenate(stats), point_mass=greedy,
                              log_q=mx.concatenate(confidence) if confidence else None)
    return ids, law


# ------------------------------------------------------------------------------- a round
def round_function(rt, anchor, position, seed, view: DraftView, round_state, proposals: int, *,
                   temperature: float, draft_temperature: float, stages, params=None, draft_params=None,
                   stops=None, remaining=None):
    """The round as MLX builds it (lazy): [packet, ids (W,), logits, captures..., deltas...].
    What the program records and what serves when no program is live. ``params`` and
    ``draft_params`` carry the sampled temperatures as data (``round.law_params``).

    With ``stops`` ((S,) int32, padded with -2) and ``remaining`` ((1,) int32) the round also
    commits: [packet, ids, [kept, emitted] (2,) uint32, logits, captures..., deltas..., then
    the state the commit wrote (per visit keys, values and four tails; per drafter block keys
    and values; the drafter's last feature), then the old rows of every ring slot it wrote]."""
    model = rt.model
    ids, law = propose(rt.drafter, model, anchor, view, proposals, position=position, seed=seed,
                       temperature=draft_temperature, params=draft_params)
    block = mx.concatenate([anchor.astype(mx.int32).reshape(1), ids])
    logits, captures, deltas = decode_panel(model, block, round_state, stages=stages)
    packet = fast_round.decide(logits[0], ids, position=position, seed=seed,
                               temperature=temperature, draft=law, params=params)
    if stops is None:
        return [packet, block, *engine_site.flatten(logits, captures, deltas)]
    # The commit, inside the round: the loop's own walk over the packet (stops, the request's
    # remaining budget), then the kept rows written into the state and observed by the drafter.
    rows = proposals + 1
    kept, emitted = fast_round.commit_count(packet, rows, stops=stops, remaining=remaining)
    kept = mx.where(packet[1] == 0, kept, mx.zeros_like(kept))  # a row that is no law: nothing kept
    target_new, target_undo = _commit_target(round_state, deltas, kept, rows)
    draft_new, last, draft_undo = _observe_draft(rt.drafter, view, captures, kept, rows)
    return [packet, block, mx.stack([kept, emitted]), *engine_site.flatten(logits, captures, deltas),
            *target_new, *draft_new, last, *target_undo, *draft_undo]


# --------------------------------------------------------------------- the commit inside
_TAILS = """
    uint gid = thread_position_in_grid.x;
    const uint n0 = KEEP * W0, n1 = KEEP * W1, n2 = KEEP * W2, n3 = KEEP * W3;
    if (gid >= n0 + n1 + n2 + n3) return;
    const uint k = kept[0];
    if (gid < n0) { uint r = gid / W0, c = gid % W0, s = r + k;
        o0[gid] = s < KEEP ? t0[s * W0 + c] : p0[(s - KEEP) * W0 + c]; return; }
    gid -= n0;
    if (gid < n1) { uint r = gid / W1, c = gid % W1, s = r + k;
        o1[gid] = s < KEEP ? t1[s * W1 + c] : p1[(s - KEEP) * W1 + c]; return; }
    gid -= n1;
    if (gid < n2) { uint r = gid / W2, c = gid % W2, s = r + k;
        o2[gid] = s < KEEP ? t2[s * W2 + c] : p2[(s - KEEP) * W2 + c]; return; }
    gid -= n2;
    { uint r = gid / W3, c = gid % W3, s = r + k;
        o3[gid] = s < KEEP ? t3[s * W3 + c] : p3[(s - KEEP) * W3 + c]; }
"""

# Per ring slot the round writes: the panel's row where it is kept, the slot's own row where
# it is not (so a rejected row never touches state), and the slot's old row for an undo.
_RING_ROWS = """
    uint gid = thread_position_in_grid.x;
    const uint per = HEADS * ROWS * DIM;
    if (gid >= per) return;
    uint d = gid % DIM, i = (gid / DIM) % ROWS, h = gid / (DIM * ROWS);
    uint slot = (newest[0] + 1 + i) % SLOTS;
    uint src = (h * SLOTS + slot) * DIM + d;
    auto ok = old_k_in[src]; auto ov = old_v_in[src];
    bool keep = i < kept[0];
    upd_k[gid] = keep ? new_k[gid] : ok;
    upd_v[gid] = keep ? new_v[gid] : ov;
    old_k[gid] = ok;
    old_v[gid] = ov;
"""


@lru_cache(maxsize=None)
def _tails_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_inkling_l7_tails",
        input_names=["t0", "t1", "t2", "t3", "p0", "p1", "p2", "p3", "kept"],
        output_names=["o0", "o1", "o2", "o3"],
        source=_TAILS,
    )


@lru_cache(maxsize=None)
def _ring_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_inkling_l7_ring_rows",
        input_names=["old_k_in", "old_v_in", "new_k", "new_v", "newest", "kept"],
        output_names=["upd_k", "upd_v", "old_k", "old_v"],
        source=_RING_ROWS,
    )


def _ring_slots(newest, slots: int, rows: int):
    """Ring slots of the round's rows, (1, 1, rows, 1) int32 (index layout of put_along_axis)."""
    i = fast_round._const(list(range(rows)), mx.uint32)
    at = (newest.astype(mx.uint32).reshape(()) + fast_round._const(1, mx.uint32) + i) % fast_round._const(slots, mx.uint32)
    return at.astype(mx.int32).reshape(1, 1, rows, 1)


def _write_ring(ring_k, ring_v, new_k, new_v, newest, kept, at):
    """The kept rows into a ring in place (``put_along_axis`` takes the ring's buffer once
    nothing else holds it): (ring_k', ring_v', old_k, old_v)."""
    heads, slots, dim = ring_k.shape[1], ring_k.shape[2], ring_k.shape[3]
    rows = new_k.shape[2]
    upd_k, upd_v, old_k, old_v = _ring_kernel()(
        inputs=[ring_k, ring_v, new_k, new_v, newest.astype(mx.uint32).reshape(1), kept.reshape(1)],
        template=[("HEADS", heads), ("ROWS", rows), ("DIM", dim), ("SLOTS", slots)],
        grid=(heads * rows * dim, 1, 1),
        threadgroup=(min(256, heads * rows * dim), 1, 1),
        output_shapes=[new_k.shape, new_v.shape, new_k.shape, new_v.shape],
        output_dtypes=[ring_k.dtype, ring_v.dtype, ring_k.dtype, ring_v.dtype],
    )
    index = mx.broadcast_to(at, new_k.shape)
    return (mx.put_along_axis(ring_k, index, upd_k, axis=2), mx.put_along_axis(ring_v, index, upd_v, axis=2),
            old_k, old_v)


def _commit_target(st, deltas, kept, rows: int):
    """``DecodeState.commit`` of ``kept`` rows (an array), as a function of the round's arrays.

    Window rings: the kept rows at their slots, other slots unchanged (old rows kept for an
    undo). Global banks: every row at the committed length (rows past ``kept`` lie past the
    new length, where nothing reads them; the bank has room, ``serve`` grows it first).
    Convolution tails: the last three rows of tail + kept rows. Same bytes as the stock commit.
    Returns (per visit [keys, values, tail0..3], per window visit [old keys, old values])."""
    at = _ring_slots(st.newest, st.window, rows)
    start = st.pos.astype(mx.int32).reshape(1)
    kept_u = kept.astype(mx.uint32)
    new, undo = [], []
    for visit, local in enumerate(st.local):
        delta = deltas[visit]
        if local:
            keys, values, newest, _count = st.window_view(visit)
            keys, values, old_k, old_v = _write_ring(keys, values, delta.keys, delta.values, newest, kept_u, at)
            undo += [old_k, old_v]
        else:
            keys, values, _count = st.global_view(visit)
            keys = mx.slice_update(keys, delta.keys, start, axes=(2,))
            values = mx.slice_update(values, delta.values, start, axes=(2,))
        tails = [st.tail(visit, slot) for slot in range(4)]
        widths = [int(t.shape[-1]) for t in tails]
        new_tails = _tails_kernel()(
            inputs=[*tails, *delta.taps, kept_u.reshape(1)],
            template=[("KEEP", int(tails[0].shape[1])), ("W0", widths[0]), ("W1", widths[1]), ("W2", widths[2]),
                      ("W3", widths[3])],
            grid=(int(tails[0].shape[1]) * sum(widths), 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[t.shape for t in tails],
            output_dtypes=[t.dtype for t in tails],
        )
        new += [keys, values, *new_tails]
    return new, undo


def _observe_draft(drafter, view, captures, kept, rows: int):
    """``Drafter.observe`` of the first ``kept`` capture rows, as a function of arrays.

    Every row's context keys and values are projected (one product of ``rows`` rows) and the
    kept ones written: a window block's ring at its slots, the global block's bank at the
    committed length. The projection's row count can change the product's last bits against
    a ``kept``-row product, so the drafter's history may differ from the stock observe in the
    last place: it changes which tokens are proposed, never the output law (class B, as the
    drafter of this program already is). The last mixed feature is the kept rows' last one,
    the same bytes as the stock observe. Returns ([keys, values] per block, last, undo rows)."""
    kept_u = kept.astype(mx.uint32)
    at = _ring_slots(view.newest, max((w for w in view.windows if w), default=1), rows)
    start = view.length.astype(mx.int32).reshape(1)
    new, undo = [], []
    for index, block in enumerate(drafter.blocks):
        features = drafter.mix(captures, index, block.w["dflare_norm.weight"])
        keys = block.normalize(block.heads(features, "k_ctx"), "self_attn.k_norm_ctx")
        keys = block.rotate(keys, view.offset)
        values = block.heads(features, "v_ctx")
        ring_k, ring_v = view.keys[index], view.values[index]
        if block.window:
            ring_k, ring_v, old_k, old_v = _write_ring(ring_k, ring_v, keys.astype(ring_k.dtype),
                                                      values.astype(ring_v.dtype), view.newest, kept_u, at)
            undo += [old_k, old_v]
        else:
            ring_k = mx.slice_update(ring_k, keys.astype(ring_k.dtype), start, axes=(2,))
            ring_v = mx.slice_update(ring_v, values.astype(ring_v.dtype), start, axes=(2,))
        new += [ring_k, ring_v]
    mixed = drafter.mix(captures, -1, drafter.markov["dflare_norm.weight"])
    last_row = mx.maximum(kept.astype(mx.int32), fast_round._const(1, mx.int32)) - fast_round._const(1, mx.int32)
    last = mx.take(mixed, last_row.reshape(1), axis=1)
    last = mx.where(kept > 0, last, view.last)
    return new, last, undo


class Round:
    """What one served round hands the generation loop (the same objects a stock round has).

    A committing round (``serve(..., stops=..., remaining=...)``) has already written its
    kept rows: ``committed`` is True, ``kept`` and ``emitted`` are the device's walk over the
    packet, and the loop calls ``rewind(kept)`` only when its own walk keeps another count
    (a cancellation, the opt-in repetition guard): the round's writes are undone and the
    stock commit writes that count instead."""

    __slots__ = ("block", "rows", "features", "pending", "packet", "native", "route", "committed", "kept",
                 "emitted", "_rewind")

    def __init__(self, block, rows, features, pending, packet, native, route, committed=False, kept=None,
                 emitted=None, rewind=None):
        self.block, self.rows, self.features, self.pending = block, rows, features, pending
        self.packet, self.native, self.route = packet, native, route
        self.committed, self.kept, self.emitted, self._rewind = committed, kept, emitted, rewind

    def rewind(self, kept: int) -> None:
        """Undo this round's commit and commit ``kept`` rows the stock way (rare, counted)."""
        if not self.committed:
            raise ValueError("only a committing round can be rewound")
        if kept != self.kept:
            self._rewind(kept)
            self.kept = kept


_LAST: dict[str, Any] = {"route": None, "reason": None, "proposals": 0}
#: Stop ids a committing round takes as data (padded with -2).
MAX_STOPS = 16


def last_route() -> dict[str, Any]:
    return dict(_LAST)


def _note(route: str, reason: str | None, proposals: int) -> None:
    _LAST["route"], _LAST["reason"], _LAST["proposals"] = route, reason, proposals


def _decline(reason: str, proposals: int):
    count(f"round.decline.{reason}")
    _note("declined", reason, proposals)
    return None


def _view_from(drafter, arrays) -> DraftView:
    """The drafter view of a round from its inputs (``draft_inputs``' order)."""
    offset, newest, count_rows, length, last = arrays[:5]
    rest = arrays[5:]
    return DraftView(list(rest[0::2]), list(rest[1::2]), offset, newest, count_rows, length, last,
                     [b.window for b in drafter.blocks])


def _grow_bank(entry, need: int) -> None:
    """A global drafter bank with room for ``need`` rows (whole ``BANK_STEP`` steps)."""
    cap = round_up(need, BANK_STEP)
    length = int(entry.offset)
    shape = (entry.keys.shape[0], entry.keys.shape[1], cap, entry.keys.shape[3])
    keys, values = mx.zeros(shape, entry.keys.dtype), mx.zeros(shape, entry.values.dtype)
    keys[:, :, :length, :] = entry.keys[:, :, :length, :]
    values[:, :, :length, :] = entry.values[:, :, :length, :]
    mx.eval(keys, values)
    entry.keys, entry.values = keys, values


def serve(rt, anchor, cache, proposals: int, *, position, seed, temperature, draft_temperature,
          stages=None, spec=None, state=None, stops=None, remaining=None):
    """One DSpark round (proposals, verify, decision): a ``Round``, or None (counted) when
    the stock round must serve it, with nothing changed.

    ``anchor`` is the last emitted token, ``position`` the absolute position of the first
    proposal (``len(prompt) + len(tokens)``), ``seed`` the request's 64-bit round seed,
    ``temperature`` the target's and ``draft_temperature`` the drafter's (0: greedy).
    The caller reads ``Round.packet`` (``fast.round.read_packet``), walks it as for a stock
    round and commits with ``rt.commit(cache, Round.features, Round.pending, kept, W)``.

    With ``stops`` (the request's stop ids) and ``remaining`` (tokens it may still emit) the
    round also commits, inside the same program: the loop then skips ``rt.commit`` and calls
    ``Round.rewind(kept)`` only if its own walk keeps a count other than ``Round.kept``.
    """
    from .decoder import FastPending, fast_mode, stage_spec
    from .step import stages_for

    k = int(proposals)
    commit = stops is not None
    if fast_mode() in ("off", "0", "stock", "reference"):
        return _decline("fast_off", k)
    if engine_runtime.mode() == "0":
        return _decline("disabled", k)
    if rt.drafter is None:
        return _decline("no_drafter", k)
    if not 1 <= k <= MAX_PROPOSALS:
        return _decline("proposals", k)
    if commit:
        stop_ids = [int(t) for t in stops]
        if len(stop_ids) > MAX_STOPS:
            return _decline("stops", k)
        if remaining is None or int(remaining) < 1:
            return _decline("remaining", k)
    decoder = rt.fast
    if spec is None:
        spec = stage_spec() if mx.default_device() == mx.gpu and mx.metal.is_available() else ""
    unfit = engine_site._unfit(spec)
    if unfit:
        return _decline(unfit, k)
    if stages is None:
        stages = decoder._stages.get(spec) or stages_for(spec)
    reason = decoder.decline(k + 1, cache)
    if reason is not None:
        return _decline(f"step_{reason}", k)
    replay = site(rt)
    if replay.declined():
        return _decline("unavailable", k)
    if state is None:
        state = DecodeState(rt.model, cache[: rt.trunk_entries])
        state.prepare()
    model, drafter = rt.model, rt.drafter
    dcache = cache[rt.trunk_entries :]
    rows = k + 1
    if commit:
        # Room for the round's rows in every global bank before the round (the stock commit
        # grows after it): a growth is a new capacity step, so a new program, as for the step.
        for visit, local in enumerate(state.local):
            kv = state.cache[2 * visit]
            if not local and state.length + rows > kv.keys.shape[2]:
                state._grow(kv, state.length + rows)
    round_state, target_inputs = engine_site.state_inputs(state)
    if round_state is None:
        return _decline(target_inputs, k)
    view, draft_in = draft_inputs(drafter, dcache, state.length)
    if view is None:
        return _decline(draft_in, k)
    if commit:
        grown = False
        for i, block in enumerate(drafter.blocks):
            entry = dcache[i]
            if not block.window and state.length + rows > entry.keys.shape[2]:
                _grow_bank(entry, state.length + rows)
                grown = True
        if grown:
            view, draft_in = draft_inputs(drafter, dcache, state.length)
    anchor_a = mx.array([int(anchor)], mx.int32)
    position_a = mx.array(int(position), mx.uint32)
    seed_a = fast_round._seed_words(seed)
    t_target, t_draft = float(temperature or 0.0), float(draft_temperature or 0.0)
    greedy, draft_greedy = t_target <= 0, t_draft <= 0
    cap = model.config.get("final_logit_softcapping") or 0.0
    # The temperatures are data: one program per greedy/sampled pair serves every temperature.
    params_a = None if greedy else fast_round.law_params(k + 1, t_target)
    draft_params_a = None if draft_greedy else fast_round.law_params(1, t_draft, cap)
    head = [anchor_a, position_a, seed_a, *(a for a in (params_a, draft_params_a) if a is not None)]
    stops_a = remaining_a = None
    if commit:
        stops_a = mx.array(stop_ids + [-2] * (MAX_STOPS - len(stop_ids)), mx.int32)
        remaining_a = mx.array([int(remaining)], mx.int32)
        head += [stops_a, remaining_a]
    first_draft = len(head)
    inputs = [*head, *draft_in, *target_inputs]
    first_target = first_draft + len(draft_in)
    n_draft = len(draft_in)
    del target_inputs, draft_in, head
    geometry = ("round", k, "greedy" if greedy else "sampled", "greedy" if draft_greedy else "sampled",
                "commit" if commit else "decide", spec, tuple(model.capture_ids), engine_site.identity(rt))
    n_cap = len(model.capture_ids)
    visits = len(state.local)
    windows = None
    before = None
    if commit:
        # Nothing but the cache containers and ``inputs`` may hold an array the program writes:
        # the engine writes it in place, and MLX takes its buffer when recording.
        del view
        # The banks the round writes only its own rows of, checked in place (the rings are
        # read before they are written, so their checks run on private copies).
        out_state = 3 + 1 + n_cap + 6 * visits
        out_draft = out_state + 6 * visits
        windows = []
        for visit, local in enumerate(state.local):
            if not local:
                at = first_target + 3 + 6 * visit
                windows += [(at, 2, first_target, rows, out_state + 6 * visit),
                            (at + 1, 2, first_target, rows, out_state + 6 * visit + 1)]
        for i, block in enumerate(drafter.blocks):
            if not block.window:
                at = first_draft + 5 + 2 * i
                windows += [(at, 2, first_draft + 3, rows, out_draft + 2 * i),
                            (at + 1, 2, first_draft + 3, rows, out_draft + 2 * i + 1)]
        # What a rewind needs: the counters and tails the round starts from.
        before = {
            "length": state.length,
            "target_idx": [state.cache[2 * v]._idx if local else None for v, local in enumerate(state.local)],
            "tails": [list(state.cache[2 * v + 1].cache) for v in range(visits)],
            "draft_idx": [dcache[i]._idx if b.window else None for i, b in enumerate(drafter.blocks)],
            "draft_last": dcache[len(drafter.blocks)][0],
            "newest": int((state.cache[2 * state.local.index(True)]._idx - 1) % state.window)
            if True in state.local else 0,
            "draft_newest": None,
        }
        for i, b in enumerate(drafter.blocks):
            if b.window:
                before["draft_newest"] = int((dcache[i]._idx - 1) % b.window)
                break

    def run_mlx(arrays):
        view_ = _view_from(drafter, arrays[first_draft:first_draft + n_draft])
        return round_function(rt, anchor_a, position_a, seed_a, view_, round_state, k,
                              temperature=t_target, draft_temperature=t_draft, stages=stages,
                              params=params_a, draft_params=draft_params_a, stops=stops_a, remaining=remaining_a)

    def install(outputs):
        base = 3 + 1 + n_cap + 6 * visits
        for visit in range(visits):
            kv, conv = state.cache[2 * visit], state.cache[2 * visit + 1]
            kv.keys, kv.values = outputs[base + 6 * visit], outputs[base + 6 * visit + 1]
            for slot in range(4):
                conv[slot] = outputs[base + 6 * visit + 2 + slot]
        base += 6 * visits
        for i in range(len(drafter.blocks)):
            dcache[i].keys, dcache[i].values = outputs[base + 2 * i], outputs[base + 2 * i + 1]
        dcache[len(drafter.blocks)][0] = outputs[base + 2 * len(drafter.blocks)]

    started = time.perf_counter()
    if not replay.plan(geometry, family=geometry):
        why = replay.last_decline or "declined"
        outputs = run_mlx(inputs)
        if commit:
            install(outputs)
            inputs = None
            mx.eval(outputs)
        native, route = False, ("mlx", why)
    else:
        extra = {"extra_refs": 1, "admit": room_for_check_copies, "windows": windows} if commit else {}
        ticket = replay.begin(geometry, inputs, planned=True, started=started, **extra)
        if ticket.native:
            outputs = ticket.outputs
            if commit:
                install(outputs)
                inputs = None
            ticket.finish(outputs)
            native, route = True, ("native", None)
        else:
            outputs = run_mlx(inputs)
            if commit:
                # Every container holds the round's outputs before anything is evaluated, so
                # the graph is the last holder of each written input and MLX writes in place.
                install(outputs)
                inputs = None
                mx.eval(outputs)
            ticket.finish(outputs)
            why = ticket.reason or ticket.kind
            native = False
            route = ({"record": "recording", "verify": "verifying"}.get(ticket.kind, "mlx"), why)
    inputs = None
    packet, block = outputs[0], outputs[1]
    offset = 3 if commit else 2
    logits, captures, deltas = engine_site.unflatten(outputs[offset : offset + 1 + n_cap + 6 * visits], n_cap)
    count(f"round.native.k{k}" if native else f"round.mlx.{route[1]}")
    _note(route[0], route[1], k)
    pending = FastPending(state, deltas, rows)
    if not commit:
        return Round(block, logits[0], captures, pending, packet, native, route[0])
    import numpy as np

    kept, emitted = (int(v) for v in np.asarray(outputs[2]).tolist())
    length = before["length"]
    for visit, local in enumerate(state.local):
        kv = state.cache[2 * visit]
        if local:
            kv._idx = (before["target_idx"][visit] + kept - 1) % state.window + 1 if kept else kv._idx
        kv.offset = length + kept
    for i, b in enumerate(drafter.blocks):
        entry = dcache[i]
        if b.window and kept:
            entry._idx = (before["draft_idx"][i] + kept - 1) % b.window + 1
        entry.offset = length + kept
    undo = outputs[3 + 1 + n_cap + 6 * visits + 6 * visits + 2 * len(drafter.blocks) + 1:]
    count("round.commit")

    def rewind(kept_host):
        count("round.rewind")
        devices = kept
        windows_t = [v for v, local in enumerate(state.local) if local]
        for j, visit in enumerate(windows_t):
            kv = state.cache[2 * visit]
            old_k, old_v = undo[2 * j], undo[2 * j + 1]
            for i in range(devices):
                slot = (before["newest"] + 1 + i) % state.window
                kv.keys[:, :, slot : slot + 1, :] = old_k[:, :, i : i + 1, :]
                kv.values[:, :, slot : slot + 1, :] = old_v[:, :, i : i + 1, :]
            kv._idx = before["target_idx"][visit]
        for visit in range(visits):
            state.cache[2 * visit].offset = length
            conv = state.cache[2 * visit + 1]
            for slot in range(4):
                conv[slot] = before["tails"][visit][slot]
        draft_undo = undo[2 * len(windows_t):]
        j = 0
        for i, b in enumerate(drafter.blocks):
            entry = dcache[i]
            if b.window:
                old_k, old_v = draft_undo[2 * j], draft_undo[2 * j + 1]
                for r in range(devices):
                    slot = (before["draft_newest"] + 1 + r) % b.window
                    entry.keys[:, :, slot : slot + 1, :] = old_k[:, :, r : r + 1, :]
                    entry.values[:, :, slot : slot + 1, :] = old_v[:, :, r : r + 1, :]
                entry._idx = before["draft_idx"][i]
                j += 1
            entry.offset = length
        dcache[len(drafter.blocks)][0] = before["draft_last"]
        mx.eval([c.state for c in cache])
        rt.commit(cache, captures, FastPending(state, deltas, rows), kept_host, rows)

    return Round(block, logits[0], captures, pending, packet, native, route[0], committed=True, kept=kept,
                 emitted=emitted, rewind=rewind)


# ------------------------------------------------------------------------------- warm-up
def _synthetic_draft_cache(rt, length: int, seed: int):
    drafter = rt.drafter
    cfg = drafter.cfg
    cache = drafter.make_cache()
    heads, head_dim = cfg["num_key_value_heads"], cfg["head_dim"]
    dtype = rt.model.embed_tokens.weight.dtype
    key = mx.random.key(seed)
    arrays = []
    for i, block in enumerate(drafter.blocks):
        entry = cache[i]
        rows = block.window if block.window else round_up(max(length, 1), BANK_STEP)
        key, k1, k2 = mx.random.split(key, 3)
        entry.keys = (mx.random.normal((1, heads, rows, head_dim), key=k1) * 0.5).astype(dtype)
        entry.values = (mx.random.normal((1, heads, rows, head_dim), key=k2) * 0.5).astype(dtype)
        entry.offset = length
        if block.window:
            entry._idx = (length - 1) % block.window + 1 if length >= block.window else length
        else:
            entry.step = BANK_STEP
        arrays += [entry.keys, entry.values]
    key, k3 = mx.random.split(key)
    cache[-1][0] = (mx.random.normal((1, 1, cfg["hidden_size"]), key=k3)).astype(dtype)
    arrays.append(cache[-1][0])
    mx.eval(arrays)
    return cache


def warm(rt, proposals=None, *, length: int = 1100, seed: int = 20261121, temperature: float = 1.0,
         draft_temperature: float | None = None) -> dict:
    """Record and check the round program for each proposal count before the first request
    (synthetic state, nothing of a request touched)."""
    report: dict[str, Any] = {"counts": {}, "mode": engine_runtime.mode()}
    if engine_runtime.mode() == "0" or not engine_runtime.available() or rt.drafter is None:
        report["skipped"] = engine_runtime.unavailable() or ("no_drafter" if rt.drafter is None else "disabled")
        return report
    proposals = tuple(range(1, MAX_PROPOSALS + 1)) if proposals is None else tuple(proposals)
    draft_temperature = temperature if draft_temperature is None else draft_temperature
    started = time.perf_counter()
    target = engine_site._synthetic_cache(rt, length, seed)
    draft = _synthetic_draft_cache(rt, length, seed + 7)
    cache = target + draft
    state = DecodeState(rt.model, cache[: rt.trunk_entries])
    state.prepare()
    replay = site(rt)
    vocab = rt.model.embed_tokens.weight.shape[0]
    round_ms: dict[int, float] = {}
    for k in proposals:
        entry_started = time.perf_counter()
        native = tries = 0
        while tries < _WARM_TRIES and native < 1:
            tries += 1
            served = serve(rt, (seed + tries * 7919 + k) % vocab, cache, k, position=length + 1, seed=seed + k,
                           temperature=temperature, draft_temperature=draft_temperature, state=state)
            if served is None:
                break
            mx.eval(served.packet, served.rows, served.features, [d.arrays() for d in served.pending.deltas])
            native += int(served.native)
            del served
        seconds = time.perf_counter() - entry_started
        if native:
            samples = []
            for t in range(3):
                mx.synchronize()
                t0 = time.perf_counter()
                served = serve(rt, (seed + 31 * t + k) % vocab, cache, k, position=length + 1, seed=seed + k,
                               temperature=temperature, draft_temperature=draft_temperature, state=state)
                fast_round.read_packet(served.packet, k + 1)
                mx.eval(served.rows, served.features, [d.arrays() for d in served.pending.deltas])
                samples.append((time.perf_counter() - t0) * 1e3)
                del served
            round_ms[k] = _seconds(sorted(samples)[1] / 1e3) * 1e3
        report["counts"][k] = {"calls": tries, "live": native > 0, "seconds": _seconds(seconds),
                               "round_ms": round_ms.get(k)}
    del state, cache, target, draft
    report["round_ms"] = round_ms
    report["context_tokens"] = length
    report["seconds"] = _seconds(time.perf_counter() - started)
    report["site"] = replay.report()
    return report


def _seconds(value: float) -> float:
    return float(f"{value:.3f}")


__all__ = ["DraftView", "MAX_PROPOSALS", "NAME", "Round", "draft_inputs", "last_route", "propose",
           "round_function", "serve", "site", "warm"]
