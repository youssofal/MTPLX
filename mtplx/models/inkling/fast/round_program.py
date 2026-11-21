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
(a multiple of ``state.BANK_STEP``), reading only the valid slots (``draft_attention``: each
key and value read once for all the query rows that share it, nothing concatenated), and the
rope offset is an array. RC6 instead reads the history back in temporal order at its exact
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
A deciding round writes nothing in place: its outputs are the packet, the block of ids, the
logits, the captures and every visit's delta, and the caller commits the rows the host accepts
(``rt.commit``) exactly as after a stock round. A committing round (``stops`` and ``remaining``
given) also walks the packet as the loop does and writes the kept rows into the target's rings,
banks and tails and the drafter's history in place (``Round.rewind`` undoes them for a cut the
device cannot see).
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mtplx.engine import runtime as engine_runtime
from mtplx.engine import sites
from mtplx.engine.site import KEY_NEW, WARMING, room_for_check_copies

from . import engine_site
from . import round as fast_round
from .routes import capture_counts, count, stage_paths
from .state import BANK_STEP, DecodeState, round_up
from .step import decode_panel
from .step import embed as fast_embed

NAME = "inkling.round"
_ATTRIBUTE = "_mtplx_inkling_round_site"
#: Proposals a round can carry: the drafter's 8-slot block holds the anchor and 7.
MAX_PROPOSALS = 7
_WARM_TRIES = 10
#: Timed calls per width at load (interleaved across widths; the fastest native one is kept).
_TIMING_PASSES = 5


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


def _commit_enabled() -> bool:
    from .decoder import round_commit_enabled

    return bool(round_commit_enabled())


def site(rt: Any):
    replay = getattr(rt, _ATTRIBUTE, None)
    if replay is None:
        replay = sites.site_for(
            rt, _ATTRIBUTE, NAME,
            weights=_weights, pin_operands=True, names=_named_weights,
            output_names=_output_name, fence_strided=False,
            # The committing round writes its state in place, so a recording made while the
            # loop's previous work still holds a state array carries MLX's copy of it, and the
            # allocator can hand a constant's buffer to an array the program later writes: two
            # recordings of one round then differ by copies, or by a constant against the same
            # bytes. Both are one program (the engine's round-program mode): with the copies
            # removed the rest must be the same program structurally, then it is bit-checked
            # twice. Only the committing round needs it (the deciding round writes nothing in
            # place), so the site settles copies only when the committing round is on.
            settle_copies=_commit_enabled(),
        )
        # The round's kept constants (sizes, per-row parameters, modes) are bound in place, and
        # so are the tables made once from the weights (see ``engine_site.weight_tables``).
        replay.pin_before_each_recording(fast_round.constants)
        replay.pin_before_each_recording(engine_site.weight_tables)
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
def draft_kernel_on() -> bool:
    """The drafter's key-sharing attention below, on unless ``MTPLX_INKLING_DRAFT_ATTENTION=0``
    (MLX's masked attention over the concatenated history: the A/B base and a rollback). The
    choice is in the program key, so a process can hold both."""
    return os.environ.get("MTPLX_INKLING_DRAFT_ATTENTION", "on").strip().lower() not in ("0", "off", "false", "no")


def _valid(view: DraftView, index: int, rows: int) -> mx.array:
    """MLX path: which history slots of block ``index`` a proposal row reads, then its own rows."""
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


# The drafter's attention: every query row of a block (3 query heads of each key head times 8
# rows) against the history and the block's own rows, reading each key and value once for all
# 24 of them (MLX's vector kernel reads them once per query: 24 times, which at a long context
# made the drafter's global block cost more than the rest of the drafter). Two launches: chunk
# partials (online softmax in float32), then their merge. The history is a ring (a slot is valid
# when its age is under the count) or a bank (a row is valid below the count); the block's own
# rows are a chunk of their own, so nothing is concatenated and no bank is copied.
_ATTEND = """
    uint c = threadgroup_position_in_grid.x;
    uint h = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32, lane = tid % 32;
    const uint QV = G * R;
    const uint NHIST = (S + CHUNK - 1) / CHUNK;
    const uint C = NHIST + 1;
    threadgroup float qs[QV * D];
    threadgroup T kt[TK * D];
    threadgroup T vt[TK * D];
    const float scale = metal::rsqrt(float(D));
    for (uint i = tid; i < QV * D; i += 256) {
        uint qv = i / D, d = i % D;
        qs[i] = float(q[((h * G + qv / R) * R + qv % R) * D + d]) * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint count = meta[0], newest = meta[1];
    const bool own = c == NHIST;
    const uint start = own ? 0 : c * CHUNK;
    // a bank's rows past the count are not history: a chunk wholly past it reads nothing
    const uint end = own ? R : metal::min(metal::min(start + CHUNK, (uint)S), RING ? (uint)S : metal::max(count, start));
    float m[QPS], l[QPS], acc[QPS][D / 32];
    for (uint a = 0; a < QPS; ++a) {
        m[a] = -INFINITY; l[a] = 0.0f;
        for (uint e = 0; e < D / 32; ++e) acc[a][e] = 0.0f;
    }
    for (uint base = start; base < end; base += TK) {
        uint n = metal::min((uint)TK, end - base);
        for (uint i = tid; i < n * D; i += 256) {
            uint j = base + i / D, d = i % D;
            kt[i] = own ? ok[(h * R + j) * D + d] : hk[(h * S + j) * D + d];
            vt[i] = own ? ov[(h * R + j) * D + d] : hv[(h * S + j) * D + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = 0; t < n; ++t) {
            uint j = base + t;
            bool valid = own || (RING ? (((newest + S - j) % S) < count) : (j < count));
            if (!valid) continue;
            float kd[D / 32], vd[D / 32];
            for (uint e = 0; e < D / 32; ++e) {
                kd[e] = float(kt[t * D + lane * (D / 32) + e]);
                vd[e] = float(vt[t * D + lane * (D / 32) + e]);
            }
            for (uint a = 0; a < QPS; ++a) {
                uint qv = sg * QPS + a;
                float part = 0.0f;
                for (uint e = 0; e < D / 32; ++e) part += qs[qv * D + lane * (D / 32) + e] * kd[e];
                float sc = simd_sum(part);
                float mn = metal::max(m[a], sc);
                float corr = metal::exp(m[a] - mn);
                float p = metal::exp(sc - mn);
                l[a] = l[a] * corr + p;
                for (uint e = 0; e < D / 32; ++e) acc[a][e] = acc[a][e] * corr + p * vd[e];
                m[a] = mn;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint a = 0; a < QPS; ++a) {
        uint qv = sg * QPS + a;
        uint pi = (h * C + c) * QV + qv;
        if (lane == 0) { pm[pi] = m[a]; pl[pi] = l[a]; }
        for (uint e = 0; e < D / 32; ++e) pacc[pi * D + lane * (D / 32) + e] = acc[a][e];
    }
"""

_MERGE = """
    uint qv = threadgroup_position_in_grid.x;
    uint h = threadgroup_position_in_grid.y;
    uint d = thread_position_in_threadgroup.x;
    const uint QV = G * R;
    const uint C = (S + CHUNK - 1) / CHUNK + 1;
    float mx_ = -INFINITY;
    for (uint c = 0; c < C; ++c) mx_ = metal::max(mx_, pm[(h * C + c) * QV + qv]);
    float lsum = 0.0f, o = 0.0f;
    for (uint c = 0; c < C; ++c) {
        uint pi = (h * C + c) * QV + qv;
        float w = metal::exp(pm[pi] - mx_);
        lsum += pl[pi] * w;
        o += pacc[pi * D + d] * w;
    }
    out[((h * G + qv / R) * R + qv % R) * D + d] = static_cast<T>(o / lsum);
"""
_TILE = 32


def _chunk(slots: int, ring: bool) -> int:
    """History rows a threadgroup reads: 64 for a 1,024-slot ring (more threadgroups for a short
    history), 128 for a bank up to 32K rows, then doubling so a bank never has more than 257
    chunks (their float32 partials, 12 KiB a chunk and key head, stay under 26 MB at any context).
    Engine profile, per block (`l7/receipts/draft-chunks-1.json`): ring 372 -> 223 us at 64 (MLX
    334), bank 8K 728 -> 641 at 128 (MLX 1,303), bank 64K 1,692 at 256, 1,696 at 128 (MLX 3,076)."""
    if ring:
        return 64
    return max(128, 1 << (max(int(slots) - 1, 1).bit_length() - 8))


@lru_cache(maxsize=None)
def _attend_kernels():
    partials = mx.fast.metal_kernel(
        name="mtplx_inkling_l7_draft_attend",
        input_names=["q", "hk", "hv", "ok", "ov", "meta"],
        output_names=["pm", "pl", "pacc"],
        source=_ATTEND,
    )
    merge = mx.fast.metal_kernel(
        name="mtplx_inkling_l7_draft_merge",
        input_names=["pm", "pl", "pacc"],
        output_names=["out"],
        source=_MERGE,
    )
    return partials, merge


def draft_attention(q, keys, values, own_k, own_v, count, newest, *, ring: bool, chunk: int | None = None):
    """Softmax attention of a drafter block: ``q`` (1, Hq, R, D) against the history ``keys``,
    ``values`` (1, Hk, S, D) (a ring when ``ring``, else a bank) and the block's own rows
    ``own_k``, ``own_v`` (1, Hk, R, D), scale ``D ** -0.5``. Float32 inside; bfloat16 or float16
    in and out (the drafter's own type: the FP16 siblings for M1 and M2 keep float16)."""
    _, hq, rows, dim = q.shape
    hk, slots = keys.shape[1], keys.shape[2]
    group = hq // hk
    qps = group * rows // 8
    if (group * rows % 8 or dim % 32 or q.dtype not in (mx.bfloat16, mx.float16) or keys.dtype != q.dtype
            or values.dtype != q.dtype):
        raise ValueError("draft_attention: 8 simdgroups share the query vectors (bf16 or f16, D a multiple of 32)")
    chunk = int(chunk or _chunk(slots, ring))
    if chunk % _TILE:
        raise ValueError("draft_attention: the chunk is a whole number of tiles")
    chunks = -(-slots // chunk) + 1
    template = [("T", q.dtype), ("G", group), ("R", rows), ("D", dim), ("S", slots), ("CHUNK", chunk),
                ("TK", _TILE), ("QPS", qps), ("RING", bool(ring))]
    meta = mx.stack([count.astype(mx.uint32).reshape(()), newest.astype(mx.uint32).reshape(())])
    partials, merge = _attend_kernels()
    pm, pl, pacc = partials(
        inputs=[q, keys, values, own_k.astype(keys.dtype), own_v.astype(values.dtype), meta],
        template=template,
        grid=(256 * chunks, hk, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(hk * chunks * group * rows,), (hk * chunks * group * rows,),
                       (hk * chunks * group * rows * dim,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    (out,) = merge(
        inputs=[pm, pl, pacc],
        template=template,
        grid=(group * rows * dim, hk, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )
    return out


def _block(block, hidden, residual, view: DraftView, index: int):
    """``DraftBlock.__call__`` with its attention read in place (see the module docstring)."""
    summed = hidden if residual is None else hidden.astype(mx.float32) + residual.astype(mx.float32)
    x = block.normalize(summed, "input_layernorm").astype(hidden.dtype)
    q = block.rotate(block.normalize(block.heads(x, "q"), "self_attn.q_norm"), view.offset)
    k = block.rotate(block.normalize(block.heads(x, "k"), "self_attn.k_norm"), view.offset)
    v = block.heads(x, "v")
    window = view.windows[index]
    if draft_kernel_on():
        out = draft_attention(q, view.keys[index], view.values[index], k, v,
                              view.count if window else view.length, view.newest, ring=bool(window))
    else:
        out = mx.fast.scaled_dot_product_attention(
            q,
            mx.concatenate([view.keys[index], k], axis=2),
            mx.concatenate([view.values[index], v], axis=2),
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
    embed = fast_embed(target, anchor_ids)
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
    from ..precision import AffineWeight

    from . import draft_table

    table = draft_table.bind(drafter, target)       # class B, MTPLX_INKLING_DRAFT_TABLE (off by default)
    head = target.unembed if table is None else table.head
    base = None
    if isinstance(head, AffineWeight):
        # A packed head (Bare Speed, Optimized Speed): lane L1's launch, one read of the codes for
        # all the proposal rows (MLX's 2-to-24-row product reads them per row block, 2.7 against
        # 1.9 ms at 8 rows); fed float32, as the verify's head is (a bf16 row widened inside the
        # launch: the same values, no cast launch). Where it refuses (a stored form it does not take,
        # a failed first-use proof on this Mac), MLX's product, counted.
        from .qproduct import STREAM_TYPES, affine_product

        width = drafter.vocab if table is None else None
        try:
            if h.dtype in STREAM_TYPES:
                base = affine_product(h, head, widen=True)[..., :width]
            else:
                base = affine_product(h.astype(mx.float32), head)[..., :width]
        except RuntimeError:
            count("fallback.draft_head.affine_launch_refused")
    if base is None:
        base = linear(h, head)[..., : (drafter.vocab if table is None else None)]
    rank = drafter.cfg["markov_rank"]
    weights = drafter.markov["rnn_proj.weight"].astype(mx.float32)
    inputs = mx.concatenate([h, mx.broadcast_to(view.last, h.shape)], axis=-1)
    conditioned = linear(inputs.astype(mx.float32), weights[:, 2 * rank :]) + drafter.markov[
        "rnn_proj.bias"
    ].astype(mx.float32)
    folded = drafter.fold(target.unembed) if table is None else table.folded
    from .matvec import matvec

    start = position.astype(mx.uint32).reshape(())
    # Kept constants, not fresh host scalars, anywhere in the program (see round.commit_count):
    # the same values, so the same bits.
    state = fast_round._const([[0.0] * rank], dtype)
    one = fast_round._const(1.0, dtype)
    prev = anchor_ids.reshape(1)
    picks, laws, stats, confidence = [], [], [], []
    for i in range(count_):
        step_in = mx.concatenate([state, drafter.markov["markov_w1.weight"][prev]], axis=-1)
        # The drafter's products as the library's mat-vec in launches of ours (fast/matvec.py), bit for bit.
        update = (matvec(step_in.astype(mx.float32), weights, cols=2 * rank) + conditioned[:, i]).astype(dtype)
        gate, candidate, output = mx.split(update, 3, axis=-1)
        gate = mx.sigmoid(gate.astype(mx.float32)).astype(dtype)
        state = gate * state + (one - gate) * mx.tanh(candidate.astype(mx.float32)).astype(dtype)
        signal = mx.tanh(output.astype(mx.float32)).astype(dtype)
        bias = matvec(signal, folded, cols=folded.shape[1] - 1, bias_col=folded.shape[1] - 1)
        scores = base[:, i] + bias
        if table is not None:
            scores = draft_table.expand(table, scores)
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
    # a row that is no law: nothing kept (kept constants, not fresh host scalars: see commit_count)
    kept = mx.where(packet[1] == fast_round._const(0, packet.dtype), kept, mx.zeros_like(kept))
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


def _mix(drafter, features, row, weight):
    """``Drafter.mix`` with its sum started from a kept 0.0 (``sum()`` starts from a Python 0, a
    fresh host scalar each call); the same operations in the same order, so the same bits."""
    dtype = features[0].dtype
    mixed = fast_round._const(0.0, mx.float32)
    for i, h in enumerate(features):
        mixed = mixed + h.astype(mx.float32) * drafter.blend[row, i].astype(mx.float32)
    return mx.fast.rms_norm(mixed, weight, drafter.cfg["rms_norm_eps"]).astype(dtype)


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
        features = _mix(drafter, captures, index, block.w["dflare_norm.weight"])
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
    mixed = _mix(drafter, captures, -1, drafter.markov["dflare_norm.weight"])
    last_row = mx.maximum(kept.astype(mx.int32), fast_round._const(1, mx.int32)) - fast_round._const(1, mx.int32)
    last = mx.take(mixed, last_row.reshape(1), axis=1)
    last = mx.where(kept > fast_round._const(0, kept.dtype), last, view.last)
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
#: geometry -> the stage-path counts its record call took, counted again on each native replay
#: (a replayed round runs no stage function; review 2, finding 11).
_PATHS: dict = {}
#: Stop ids a committing round takes as data (padded with -2).
MAX_STOPS = 32


def last_route() -> dict[str, Any]:
    return dict(_LAST)


def _note(route: str, reason: str | None, proposals: int) -> None:
    _LAST["route"], _LAST["reason"], _LAST["proposals"] = route, reason, proposals
    _LAST.pop("refusal", None)


def reset_route() -> None:
    """A round this program did not serve records no program route (not the last one)."""
    _note(None, None, 0)


def _decline_program(replay, reason: str, proposals: int):
    """A program that will not serve or record this round: the loop's pieces serve it, counted
    ``round.decline.program_<reason>``; the route carries the program's last refusal (its code
    and the launch where its recordings differed)."""
    out = _decline("program_" + reason.replace(":", "_"), proposals)
    if replay.last_refusal is not None:
        _LAST["refusal"] = dict(replay.last_refusal)
    return out


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
        stop_ids = sorted({int(t) for t in stops})
        if len(stop_ids) > MAX_STOPS or remaining is None or int(remaining) < 1:
            # More stop ids than the program takes: the round is served without its commit
            # (the caller commits, as for a deciding round), counted.
            count("round.commit_declined.stops" if len(stop_ids) > MAX_STOPS else "round.commit_declined.remaining")
            commit = False
            stops = remaining = None
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
                "commit" if commit else "decide", "dattn" if draft_kernel_on() else "sdpa", spec,
                tuple(model.capture_ids), engine_site.identity(rt))
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
    # A diagnostic reference arm: the same round function served on MLX every round, never
    # replayed (equal seeds then give equal token ids to the replayed program's).
    reference = os.environ.get("MTPLX_INKLING_ROUND_REFERENCE", "").strip() == "1"
    if reference or not replay.plan(geometry, family=geometry):
        why = "reference" if reference else (replay.last_decline or "declined")
        if not reference and why not in (WARMING, KEY_NEW):
            # Refused, record-only or strided: the round function on MLX would run the step on
            # MLX too; the loop's pieces run the replayed step (counted).
            return _decline_program(replay, why, k)
        outputs = run_mlx(inputs)
        if commit:
            install(outputs)
            inputs = None
            mx.eval(outputs)
        native, route = False, ("mlx", why)
    else:
        extra = {"extra_refs": 1, "admit": room_for_check_copies, "windows": windows} if commit else {}
        ticket = replay.begin(geometry, inputs, planned=True, started=started, **extra)
        if ticket.kind == "fallback":
            # Resting after recordings that disagreed, or a call the executor would not take:
            # the pieces serve it with the replayed step, not the round function on MLX.
            return _decline_program(replay, ticket.reason or "fallback", k)
        if ticket.native:
            outputs = ticket.outputs
            if commit:
                install(outputs)
                inputs = None
            ticket.finish(outputs)
            native, route = True, ("native", None)
            for name, n in _PATHS.get(geometry, {}).items():
                count(name, n)
        else:
            with capture_counts() as box:
                outputs = run_mlx(inputs)
            if ticket.kind == "record":
                _PATHS[geometry] = stage_paths(box)
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
         draft_temperature: float | None = None, commit: bool = False) -> dict:
    """Record and check the round program for each proposal count before the first request
    (synthetic state, nothing of a request touched). ``commit`` warms the committing round
    (each warm round then commits its rows into the synthetic state, which only grows)."""
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
    extra = {"stops": [], "remaining": 1 << 30} if commit else {}
    round_ms: dict[int, float] = {}
    live: list[int] = []
    for k in proposals:
        entry_started = time.perf_counter()
        native = tries = 0
        while tries < _WARM_TRIES and native < 1:
            tries += 1
            # Every call differs in what is an input (anchor, position, seed), so the check call
            # after the recordings fails at load on any value a recording froze as a constant.
            served = serve(rt, (seed + tries * 7919 + k) % vocab, cache, k, position=length + 1 + 5 * tries,
                           seed=seed + k + 1009 * tries, temperature=temperature,
                           draft_temperature=draft_temperature, state=None if commit else state, **extra)
            if served is None:
                break
            mx.eval(served.packet, served.rows, served.features, [d.arrays() for d in served.pending.deltas])
            native += int(served.native)
            del served
        seconds = time.perf_counter() - entry_started
        if native:
            live.append(k)
        report["counts"][k] = {"calls": tries, "live": native > 0, "seconds": _seconds(seconds),
                               "round_ms": None}
    # The timings the depth table is fitted from: every live width timed in turn, one call per
    # width a pass, and the fastest native call of each width kept. Timed width by width (a median
    # of three), a busy GPU near the end of the warm-up raised only the widest rounds (83 and 86 ms
    # at 6 and 7 proposals on the native pack in the release candidate's gate), and the table fitted
    # through them served every round at one proposal; interleaved, a burst reaches every width
    # alike, and the minimum is the round's cost when the GPU is the model's.
    samples: dict[int, list[float]] = {k: [] for k in live}
    declined: set[int] = set()
    for t in range(_TIMING_PASSES):
        for k in live:
            if k in declined:
                continue
            mx.synchronize()
            t0 = time.perf_counter()
            served = serve(rt, (seed + 31 * t + k) % vocab, cache, k, position=length + 2 + 3 * t,
                           seed=seed + k + 17 * t, temperature=temperature, draft_temperature=draft_temperature,
                           state=None if commit else state, **extra)
            if served is None:
                # A call the program declined (busy, shared state, a layout it does not take):
                # no more timing for this width, and the load carries on (the loop's pieces serve
                # such rounds).
                report.setdefault("timing_declined", {})[k] = last_route().get("reason")
                declined.add(k)
                continue
            fast_round.read_packet(served.packet, k + 1)
            mx.eval(served.rows, served.features, [d.arrays() for d in served.pending.deltas])
            if served.native:
                samples[k].append((time.perf_counter() - t0) * 1e3)
            del served
    for k in live:
        if samples[k]:
            round_ms[k] = _seconds(min(samples[k]) / 1e3) * 1e3
            report["counts"][k]["round_ms"] = round_ms[k]
            report["counts"][k]["timed_ms"] = [_seconds(v / 1e3) * 1e3 for v in samples[k]]
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
