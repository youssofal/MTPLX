"""Inkling 2 Mini attention core for a decode panel: in-place reads, every row and head in one pass.

Stage B of the fast decode step (SIGNATURES.md, lane L2). A panel of ``W`` rows (1 to 32)
at absolute positions ``pos .. pos + W - 1`` attends to two key/value sources, both read
in place and never copied, reordered or concatenated:

* the committed history: a window visit's ring ``(1, 8, R, 128)`` in the cache's physical
  order (the row of age ``a`` sits at slot ``(newest - a) mod R``), or a global visit's
  bank ``(1, 8, CAP, 128)`` (position ``p`` at slot ``p``);
* the panel's own rows ``k_rows``, ``v_rows`` ``(1, 8, W, 128)`` from stage A.

Row ``r`` at position ``P = pos + r`` sees committed positions and panel rows ``0 .. r``;
on window visits only keys with ``0 <= P - key < 1024``. The logit is ``q.k / sqrt(128)``
plus the relative bias ``profile[P - key]`` (zero beyond distance 1,023), and one extra
sink logit per (row, head) adds softmax mass with a zero value.

Exactness: each row reproduces, bit for bit, the arithmetic of the one-row call RC6 makes
for that token (``attention.stock_attention``: the chronological keys plus an appended zero
row, an additive bf16 mask, MLX 0.32.2's vector attention). That kernel is single-pass
(key ``i`` folded by simdgroup ``i mod 32``) below a key count and two-pass above it (key
``i`` folded by block ``i mod BLOCKS``, partials rounded to the query type, then merged),
with the threshold and block count set by the GPU family (MLX v0.32.2
``scaled_dot_product_attention.cpp``). This module follows the same rule per row, so the
order changes exactly where MLX's does, and a row's bits never depend on how many rows
the panel holds (row stability by construction).

Two launches: ``partials`` (one threadgroup per key/value head and physical position
residue; each streams its keys once for all rows and the three query heads that share
them) and ``merge`` (one threadgroup per row and head: MLX's reduction, then the gate).
"""

from __future__ import annotations

import math
import struct
from functools import lru_cache

import mlx.core as mx

HEADS, KV_HEADS, HEAD_DIM, EXTENT, WINDOW = 24, 8, 128, 1024, 1024
GROUP = HEADS // KV_HEADS
MAX_ROWS = 32


def _float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


#: MLX rounds the Python scale ``128 ** -0.5`` to float32 once.
SCALE_BITS = _float_bits(HEAD_DIM ** -0.5)


@lru_cache(maxsize=1)
def reduction_plan():
    """``(threshold, blocks)`` of MLX 0.32.2's vector attention for RC6's one-row call.

    RC6 calls it with one query row, 24 query heads over 8 key/value heads and a float
    mask. Two-pass when (architecture ends in ``d`` or ``s``) and N >= 1024, or (grouped
    heads and N >= 4096); blocks for 3 simdgroups: 64 on ``s``, 128 on ``d``, 32 elsewhere.
    """
    arch = str(mx.device_info().get("architecture", ""))
    tail = arch[-1:] if arch else ""
    if tail == "s":
        return 1024, 64
    if tail == "d":
        return 1024, 128
    return 4096, 32


_HEADER = """
// One key folded into one chain: MLX 0.32.2 vector attention's statements, in its order.
inline void fold_key(thread const float (&qs)[4], thread const float (&kv)[4],
                     thread const float (&vv)[4], float bias, thread float& max_score,
                     thread float& sum_exp_score, thread float (&o)[4]) {
    float score = 0;
    for (int j = 0; j < 4; j++) {
        score += qs[j] * kv[j];
    }
    score = simd_sum(score);
    score += bias;
    float new_max = max(max_score, score);
    float factor = metal::fast::exp(max_score - new_max);
    float exp_score = metal::fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    for (int j = 0; j < 4; j++) {
        o[j] = o[j] * factor + exp_score * vv[j];
    }
}
"""


_PARTIALS = """
    const int rho = int(threadgroup_position_in_grid.x);   // physical position residue
    const int g = int(threadgroup_position_in_grid.y);     // key/value head
    const int r = int(simdgroup_index_in_threadgroup);     // panel row
    const int lane = int(thread_index_in_simdgroup);
    const int cap = int(kc_shape[2]);
    const int pos0 = int(pos[0]);
    const int cnt = int(count[0]);
    const int nw = int(newest[0]);
    const int P = pos0 + r;
    // Committed keys this row sees, oldest first: on window visits the last 1,023 - r.
    int nc = WINDOW ? min(cnt, 1023 - r) : cnt;
    nc = max(nc, 0);
    const int n = nc + r + 1;                  // visible keys, the row itself included
    const bool two = (n + 1) >= THRESH;        // + 1: RC6's appended zero row
    const int span = two ? BLOCKS : 32;        // chains of this row's reduction
    if (!two && rho >= 32) {
        return;                                // single-pass rows have 32 chains
    }
    const int first = pos0 - nc;               // position of logical key 0
    const int chain = ((rho - first) % span + span) % span;
    const float scale = as_type<float>(uint(SCALE));

    float qs[3][4], o[3][4], max_score[3], sum_exp_score[3];
    for (int h3 = 0; h3 < 3; ++h3) {
        for (int j = 0; j < 4; ++j) {
            qs[h3][j] = static_cast<float>(scale) * static_cast<float>(
                q[r * q_strides[1] + (3 * g + h3) * q_strides[2] + 4 * lane + j]);
            o[h3][j] = 0;
        }
        max_score[h3] = -FLT_MAX;
        sum_exp_score[h3] = 0;
    }
    const device T* pr = prof + (size_t(r) * 24 + size_t(3 * g)) * 1024;
    const device T* kg = kc + size_t(g) * kc_strides[1];
    const device T* vg = vc + size_t(g) * vc_strides[1];
    const device T* krg = kr + size_t(g) * kr_strides[1];
    const device T* vrg = vr + size_t(g) * vr_strides[1];

    // Keys of this residue in [first, P], ascending (key p has logical index p - first).
    for (int p = first + chain; p <= P; p += span) {
        const device T* kp;
        const device T* vp;
        if (p < pos0) {
            int slot = p;
            if (WINDOW) {
                slot = nw - (pos0 - 1 - p);    // age pos - 1 - p < count <= R
                if (slot < 0) {
                    slot += cap;
                }
            }
            kp = kg + size_t(slot) * kc_strides[2];
            vp = vg + size_t(slot) * vc_strides[2];
        } else {
            kp = krg + size_t(p - pos0) * kr_strides[2];
            vp = vrg + size_t(p - pos0) * vr_strides[2];
        }
        float kv[4], vv[4];
        for (int j = 0; j < 4; ++j) {
            kv[j] = static_cast<float>(kp[4 * lane + j]);
            vv[j] = static_cast<float>(vp[4 * lane + j]);
        }
        const int dist = P - p;
        for (int h3 = 0; h3 < 3; ++h3) {
            const float bias = dist < 1024 ? static_cast<float>(pr[h3 * 1024 + dist]) : 0.0f;
            fold_key(qs[h3], kv, vv, bias, max_score[h3], sum_exp_score[h3], o[h3]);
        }
    }
    // RC6's appended zero row (logical index n, virtual position P + 1) carries the sink
    // logit; it is the last key of its chain.
    if (((P + 1 - rho) % span + span) % span == 0) {
        float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int h3 = 0; h3 < 3; ++h3) {
            const float bias = static_cast<float>(sink[r * sink_strides[1] + (3 * g + h3) * sink_strides[2]]);
            fold_key(qs[h3], zero, zero, bias, max_score[h3], sum_exp_score[h3], o[h3]);
        }
    }
    for (int h3 = 0; h3 < 3; ++h3) {
        const size_t c = (size_t(r) * 24 + size_t(3 * g + h3)) * BLOCKS + size_t(chain);
        for (int j = 0; j < 4; ++j) {
            // Two-pass partials are stored in the query type, as MLX stores them.
            po[c * 128 + 4 * lane + j] = two ? static_cast<float>(static_cast<T>(o[h3][j])) : o[h3][j];
        }
        if (lane == 0) {
            pm[c] = max_score[h3];
            ps[c] = sum_exp_score[h3];
        }
    }
"""


_MERGE = """
    const int h = int(threadgroup_position_in_grid.x);
    const int r = int(threadgroup_position_in_grid.y);
    const int sg = int(simdgroup_index_in_threadgroup);
    const int lane = int(thread_index_in_simdgroup);
    const int pos0 = int(pos[0]);
    const int cnt = int(count[0]);
    int nc = WINDOW ? min(cnt, 1023 - r) : cnt;
    nc = max(nc, 0);
    const int n = nc + r + 1;
    const bool two = (n + 1) >= THRESH;
    const size_t base = (size_t(r) * 24 + size_t(h)) * BLOCKS;
    threadgroup float outputs[32 * 32];
    float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sum_exp_score;
    if (two) {
        // MLX sdpa_vector_2pass_2: blocks / 32 partials a lane.
        float max_score = -FLT_MAX;
        for (int b = 0; b < BLOCKS / 32; ++b) {
            max_score = max(max_score, pm[base + lane + 32 * b]);
        }
        max_score = simd_max(max_score);
        sum_exp_score = 0.0f;
        for (int b = 0; b < BLOCKS / 32; ++b) {
            float factor = metal::fast::exp(pm[base + lane + 32 * b] - max_score);
            sum_exp_score += factor * ps[base + lane + 32 * b];
        }
        sum_exp_score = simd_sum(sum_exp_score);
        for (int b = 0; b < BLOCKS / 32; ++b) {
            float factor = metal::fast::exp(pm[base + sg + 32 * b] - max_score);
            for (int i = 0; i < 4; ++i) {
                o[i] += factor * po[(base + sg + 32 * b) * 128 + 4 * lane + i];
            }
        }
        for (int i = 0; i < 4; ++i) {
            outputs[lane * 32 + sg] = o[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            o[i] = simd_sum(outputs[sg * 32 + lane]);
            o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        // MLX sdpa_vector's combine: simdgroup sg held chain sg; lane l reads chain l.
        for (int i = 0; i < 4; ++i) {
            o[i] = po[(base + sg) * 128 + 4 * lane + i];
        }
        float max_score = pm[base + lane];
        float new_max = simd_max(max_score);
        float factor = metal::fast::exp(max_score - new_max);
        sum_exp_score = simd_sum(ps[base + lane] * factor);
        for (int i = 0; i < 4; ++i) {
            outputs[lane * 32 + sg] = o[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            o[i] = simd_sum(outputs[sg * 32 + lane] * factor);
            o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (lane == 0) {
        for (int i = 0; i < 4; ++i) {
            const size_t at = size_t(r) * 3072 + size_t(h) * 128 + size_t(4 * sg + i);
            // The attention result in the query type, then the gate as RC6 rounds it:
            // MLX's precise sigmoid of the float32 logit, a float32 product, one cast.
            const float hv = static_cast<float>(static_cast<T>(o[i]));
            const float x = static_cast<float>(
                gate[r * gate_strides[1] + (h * 128 + 4 * sg + i) * gate_strides[2]]);
            const float y = 1 / (1 + metal::precise::exp(metal::abs(x)));
            const float gv = (x < 0) ? y : 1 - y;
            hg[at] = static_cast<T>(hv * gv);
        }
    }
"""


@lru_cache(maxsize=None)
def _kernel(name, source, header=""):
    return mx.fast.metal_kernel(
        name=name,
        input_names=_INPUTS[name],
        output_names=_OUTPUTS[name],
        source=source,
        header=header,
        ensure_row_contiguous=False,
    )


_INPUTS = {
    "mtplx_inkling_fast_attend_partials": ["q", "kc", "vc", "kr", "vr", "prof", "sink", "pos", "count", "newest"],
    "mtplx_inkling_fast_attend_merge": ["po", "pm", "ps", "gate", "pos", "count"],
}
_OUTPUTS = {
    "mtplx_inkling_fast_attend_partials": ["po", "pm", "ps"],
    "mtplx_inkling_fast_attend_merge": ["hg"],
}


def _u32(value):
    if isinstance(value, mx.array):
        return value.astype(mx.uint32).reshape(1) if value.dtype != mx.uint32 or value.ndim != 1 else value
    return mx.array([int(value)], mx.uint32)


def profiles(rel, rel_basis):
    """RC6's relative profile ``rel @ rel_basis`` (bf16), one MLX product for the panel."""
    return rel @ rel_basis.astype(rel.dtype)


def attend(q, rel, sink, gate_logits, k_rows, v_rows, keys, values, count, newest, pos,
           rel_basis, *, window, plan=None):
    """The gated attention result of a panel, ``(1, W, 3072)`` in the query dtype.

    ``q (1, W, 24, 128)``, ``rel (1, W, 24, 16)``, ``sink (1, W, 24)``,
    ``gate_logits (1, W, 3072)`` and ``k_rows``, ``v_rows (1, 8, W, 128)`` come from stage A.
    ``keys``/``values`` are the committed history, read in place: ``window=True`` for a
    ring in physical order with ``newest``/``count``, ``window=False`` for a bank whose
    first ``count`` slots are positions ``0 .. count - 1``. ``count``, ``newest`` and
    ``pos`` are uint32 arrays (or ints, for tests); they never become shapes.
    """
    rows = q.shape[1]
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"A panel holds 1 to {MAX_ROWS} rows, got {rows}")
    threshold, blocks = plan or reduction_plan()
    dtype = q.dtype
    prof = profiles(rel, rel_basis)
    pos_a, count_a, newest_a = _u32(pos), _u32(count), _u32(newest)
    template = [("T", dtype), ("WINDOW", bool(window)), ("BLOCKS", blocks),
                ("THRESH", threshold), ("SCALE", SCALE_BITS)]
    chains = rows * HEADS * blocks
    po, pm, ps = _kernel("mtplx_inkling_fast_attend_partials", _PARTIALS, _HEADER)(
        inputs=[q, keys, values, k_rows, v_rows, prof, sink, pos_a, count_a, newest_a],
        template=template,
        grid=(32 * rows * blocks, KV_HEADS, 1),
        threadgroup=(32 * rows, 1, 1),
        output_shapes=[(chains * HEAD_DIM,), (chains,), (chains,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    (hg,) = _kernel("mtplx_inkling_fast_attend_merge", _MERGE)(
        inputs=[po, pm, ps, gate_logits, pos_a, count_a],
        template=[("T", dtype), ("WINDOW", bool(window)), ("BLOCKS", blocks), ("THRESH", threshold)],
        grid=(1024 * HEADS, rows, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(1, rows, HEADS * HEAD_DIM)],
        output_dtypes=[dtype],
    )
    return hg
