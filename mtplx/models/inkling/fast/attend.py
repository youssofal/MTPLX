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
import os
import struct
from functools import lru_cache

import mlx.core as mx

HEADS, KV_HEADS, HEAD_DIM, EXTENT, WINDOW = 24, 8, 128, 1024, 1024
GROUP = HEADS // KV_HEADS
MAX_ROWS = 32
#: The long-history tile order a global visit takes by panel width (measured, l2/longab-*.json):
#: order 1 (scores by lane, exp_score by shuffle) is fastest for 1 and 2 rows, where the visit is
#: bound by its reads; order 2 (float4 queries, 16-input key pieces, exp_score through threadgroup
#: memory, value rows four keys ahead) is 14 to 25% faster from 3 rows, where it is bound by its
#: arithmetic. Both give MLX's per-key order bit for bit.
def long_order(rows):
    return 1 if rows <= 2 else 2


def _float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


#: MLX rounds the Python scale ``128 ** -0.5`` to float32 once.
SCALE_BITS = _float_bits(HEAD_DIM ** -0.5)


_TREE_PROOF = """
    const uint lane = thread_index_in_simdgroup;
    const uint base = threadgroup_position_in_grid.x * 32;
    const float v = x[base + lane];
    const float s = simd_sum(v);
    if (lane == 0) {
        float t[32];
        for (int i = 0; i < 32; ++i) t[i] = x[base + i];
        for (int w = 1; w < 32; w <<= 1) {
            for (int i = 0; i + w < 32; i += 2 * w) t[i] = t[i] + t[i + w];
        }
        ok[threadgroup_position_in_grid.x] = as_type<uint>(s) == as_type<uint>(t[0]) ? 1u : 0u;
    }
"""


@lru_cache(maxsize=1)
def lane_tree_ok():
    """Does this GPU's simd_sum equal the lane-order pairwise tree, bit for bit?

    The exact long-history order (``LONG``) computes each key's score in one lane by that tree;
    where the proof fails, the kernel keeps MLX's own cross-lane sum (``LONG`` off).
    """
    if not mx.metal.is_available():
        return False
    groups = 4096
    key = mx.random.key(20261121)
    a, b = mx.random.split(key)
    x = (mx.random.normal((groups * 32,), key=a) * mx.exp(mx.random.uniform(-8, 8, (groups * 32,), key=b)))
    kern = mx.fast.metal_kernel(name="mtplx_inkling_fast_lane_tree_proof", input_names=["x"],
                                output_names=["ok"], source=_TREE_PROOF)
    ok = kern(inputs=[x.astype(mx.float32)], grid=(32 * groups, 1, 1), threadgroup=(32, 1, 1),
              output_shapes=[(groups,)], output_dtypes=[mx.uint32])[0]
    return bool(mx.all(ok == 1).item())


def _atoi(text):
    """C ``atoi``: leading blanks, an optional sign, then digits; 0 when there are none."""
    t = (text or "").lstrip(" \t\n\r\f\v")
    sign = -1 if t[:1] == "-" else 1
    t = t[1:] if t[:1] in ("+", "-") else t
    digits = len(t) - len(t.lstrip("0123456789"))
    return sign * int(t[:digits]) if digits else 0


@lru_cache(maxsize=1)
def _family():
    arch = str(mx.device_info().get("architecture", ""))
    return arch[-1:] if arch else ""


def reduction_plan():
    """``(threshold, blocks)`` MLX 0.32.2's vector attention uses for RC6's one-row call here.

    RC6 calls it with one query row, 24 query heads over 8 key/value heads (3 simdgroups a
    threadgroup) and a float mask. Two-pass when (architecture ends in ``d`` or ``s``) and
    N >= 1024, or (grouped heads and N >= 4096); blocks for 3 simdgroups: 64 on ``s``, 128 on
    ``d``, 32 elsewhere, and ``MLX_SDPA_BLOCKS`` > 0 replaces them, rounded up to a multiple of
    32 (MLX reads it on every call with ``atoi``, so this does too).
    """
    tail = _family()
    threshold, blocks = (1024, 64) if tail == "s" else (1024, 128) if tail == "d" else (4096, 32)
    override = _atoi(os.environ.get("MLX_SDPA_BLOCKS"))
    if override > 0:
        blocks = ((override + 31) // 32) * 32
    return threshold, blocks


#: Largest reduction the fused route serves (MLX's own choices for RC6's call are 32 to 128).
MAX_BLOCKS = 256


class FusedAttentionDeclined(RuntimeError):
    """The fused attention would not give MLX's bits here; the caller runs RC6's stage."""


#: (dtype, plan) -> (reason, detail) of every declined qualification in this process.
DECLINED = {}


def decline_reason(dtype=mx.bfloat16, plan=None):
    """``None`` when the fused attention gives MLX's own bits on this Mac and setting, else a
    short reason for the caller's fallback counter. Qualified once per (dtype, plan), at first
    use (warm-up): the reduction must be one this module serves, and the fused kernels must
    equal RC6's attention (MLX's kernel, called as RC6 calls it) bit for bit on synthetic
    windows and banks covering both reductions, the long-history order and the widest panel.
    """
    return _qualify(str(dtype), plan or reduction_plan())


@lru_cache(maxsize=16)
def _qualify(dtype_name, plan):
    threshold, blocks = plan
    if blocks > MAX_BLOCKS or blocks % 32:
        return _decline(dtype_name, plan, f"blocks_{blocks}", "reduction outside the fused route")
    dtype = getattr(mx, dtype_name.split(".")[-1])
    try:
        for window, history, rows, rot in _qualification_cases(plan):
            if not _same_as_mlx(dtype, plan, window, history, rows, rot):
                return _decline(dtype_name, plan, "bits_" + ("window" if window else "global"),
                                f"history {history}, {rows} rows")
    except (RuntimeError, ValueError) as exc:          # a pipeline this Mac refuses
        return _decline(dtype_name, plan, "kernel_refused", str(exc)[:300])
    return None


def _decline(dtype_name, plan, reason, detail):
    DECLINED[(dtype_name, plan)] = (reason, detail)
    return reason


def _qualification_cases(plan):
    threshold, blocks = plan
    tiles = max(threshold, 32 * blocks) + 160          # two-pass rows with full key tiles
    return [
        # (window, committed history, rows, ring rotation)
        (True, 300, 1, 0),        # window before fill: single pass
        (True, 1500, 2, 517),     # full, wrapped ring: two-pass on d and s families
        (False, 300, 32, 0),      # widest panel: 1,024-thread threadgroups
        (False, tiles, 2, 0),     # long-history tile order 1
        (False, tiles, 3, 0),     # long-history tile order 2
    ]


def _same_as_mlx(dtype, plan, window, history, rows, rot):
    from ..attention import stock_attention

    length = history + rows
    keys = mx.random.split(mx.random.key(20261121 + 7 * history + rows), 8)

    def clipped(shape, limit, key):
        v = mx.random.normal(shape, key=key) * 2.2
        n = mx.linalg.norm(v, axis=-1, keepdims=True)
        return (v * mx.minimum(1.0, limit / (n + 1e-6))).astype(dtype)

    k = clipped((1, KV_HEADS, length, HEAD_DIM), 22.6, keys[0])
    v = mx.fast.rms_norm(mx.random.normal((1, KV_HEADS, length, HEAD_DIM), key=keys[1]), None, 1e-6).astype(dtype)
    q = clipped((1, rows, HEADS, HEAD_DIM), 22.6, keys[2])
    rel = clipped((1, rows, HEADS, 16), 8.0, keys[3])
    sink = (mx.random.normal((1, rows, HEADS), key=keys[4]) * 3.0).astype(dtype)
    gate = (mx.random.normal((1, rows, HEADS * HEAD_DIM), key=keys[5]) * 2.0).astype(dtype)
    basis = (mx.random.normal((16, EXTENT), key=keys[6]) * 0.3).astype(dtype)
    if window:
        count = min(history, WINDOW)
        kc = mx.zeros((1, KV_HEADS, WINDOW, HEAD_DIM), dtype)
        vc = mx.zeros((1, KV_HEADS, WINDOW, HEAD_DIM), dtype)
        if count:
            slots = mx.array([(p + rot) % WINDOW for p in range(history - count, history)], mx.int32)
            kc[:, :, slots, :] = k[:, :, history - count:history]
            vc[:, :, slots, :] = v[:, :, history - count:history]
        newest = (history - 1 + rot) % WINDOW if count else 0
    else:
        count, newest = history, 0
        kc, vc = k[:, :, :history], v[:, :, :history]
    fast = _attend(q, rel, sink, gate, k[:, :, history:], v[:, :, history:], kc, vc, count, newest,
                   history, basis, window=window, plan=plan)
    prof = rel @ basis.astype(rel.dtype)
    heads = []
    for r in range(rows):
        P = history + r
        begin = max(0, P - (WINDOW - 1)) if window else 0
        h = stock_attention(q[:, r:r + 1].transpose(0, 2, 1, 3), k[:, :, begin:P + 1], v[:, :, begin:P + 1],
                            prof[:, r:r + 1], sink[:, r:r + 1], WINDOW if window else 0)
        heads.append(h.transpose(0, 2, 1, 3).reshape(1, 1, -1))
    h = mx.concatenate(heads, axis=1)
    ref = (h.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))).astype(h.dtype)
    word = mx.uint16 if dtype.size == 2 else mx.uint32
    return bool(mx.array_equal(fast.view(word), ref.view(word)).item())


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
    threadgroup float qsh[LONG ? ROWS * 3 * 128 : 1];
    threadgroup float esh[LONG == 2 ? ROWS * 32 * 4 : 1];

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

    int p_from = first + chain;
    if (LONG == 1 && two && p_from < pos0) {
        // Exact long-history order for global visits (committed keys sit at slot = position):
        // full tiles of 32 keys of this chain. Lane t computes key t's score itself: MLX's 32
        // per-lane partial sums, each the same four multiply-adds, combined in simd_sum's
        // lane-order pairwise tree (proven on this GPU at first use, `lane_tree_ok`). The
        // running max of a tile is a prefix max (max is exact), so every key gets MLX's factor
        // and exp_score. When no key of the tile raises a head's running max, every factor is
        // fast::exp(0) == 1 and MLX's updates reduce exactly to sum + e and o + round(e * v)
        // (its multiply-add form is fma(o, factor, e * v), l2/probe-contract.json); only the
        // output update stays sequential, as MLX's.
        threadgroup float* qr = qsh + r * 3 * 128;
        for (int h3 = 0; h3 < 3; ++h3) {
            for (int j = 0; j < 4; ++j) {
                qr[h3 * 128 + 4 * lane + j] = qs[h3][j];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const int ncom = (pos0 - 1 - p_from) / span + 1;   // committed keys of this chain
        const int tiles = ncom >> 5;
        const size_t kst = size_t(kc_strides[2]);
        const size_t vst = size_t(vc_strides[2]);
        const size_t vstep = size_t(span) * vst;
        for (int tix = 0; tix < tiles; ++tix) {
            const int base = p_from + tix * 32 * span;
            const int pt = base + lane * span;               // this lane's key, < pos0
            const device uint4* kp = (const device uint4*)(kg + size_t(pt) * kst);
            float sub[3][4];
            for (int cc = 0; cc < 4; ++cc) {                 // MLX lanes 8 cc .. 8 cc + 7
                float kf[32];
                for (int u = 0; u < 4; ++u) {
                    const uint4 w = kp[4 * cc + u];
                    kf[8 * u + 0] = as_type<float>(w.x << 16); kf[8 * u + 1] = as_type<float>(w.x & 0xffff0000u);
                    kf[8 * u + 2] = as_type<float>(w.y << 16); kf[8 * u + 3] = as_type<float>(w.y & 0xffff0000u);
                    kf[8 * u + 4] = as_type<float>(w.z << 16); kf[8 * u + 5] = as_type<float>(w.z & 0xffff0000u);
                    kf[8 * u + 6] = as_type<float>(w.w << 16); kf[8 * u + 7] = as_type<float>(w.w & 0xffff0000u);
                }
                for (int h3 = 0; h3 < 3; ++h3) {
                    const threadgroup float* qv = qr + h3 * 128 + 32 * cc;
                    float part[8];
                    for (int m8 = 0; m8 < 8; ++m8) {
                        float score = 0;
                        for (int j = 0; j < 4; j++) {
                            score += qv[4 * m8 + j] * kf[4 * m8 + j];
                        }
                        part[m8] = score;
                    }
                    sub[h3][cc] = ((part[0] + part[1]) + (part[2] + part[3])) + ((part[4] + part[5]) + (part[6] + part[7]));
                }
            }
            const int dist = P - pt;
            float ex[3], fac[3];
            bool all_fast = true;
            for (int h3 = 0; h3 < 3; ++h3) {
                float score = (sub[h3][0] + sub[h3][1]) + (sub[h3][2] + sub[h3][3]);
                score += dist < 1024 ? static_cast<float>(pr[h3 * 1024 + dist]) : 0.0f;
                const float tmax = simd_max(score);
                if (tmax <= max_score[h3]) {
                    fac[h3] = 1.0f;                           // no key raises the max
                    ex[h3] = metal::fast::exp(score - max_score[h3]);
                } else {
                    all_fast = false;
                    float run = score;                        // inclusive prefix max over the tile
                    for (ushort d2 = 1; d2 < 32; d2 <<= 1) {
                        const float other = simd_shuffle_up(run, d2);
                        if (lane >= int(d2)) run = max(run, other);
                    }
                    const float new_max = max(max_score[h3], run);
                    float before = simd_shuffle_up(run, ushort(1));
                    before = lane == 0 ? max_score[h3] : max(max_score[h3], before);
                    fac[h3] = metal::fast::exp(before - new_max);
                    ex[h3] = metal::fast::exp(score - new_max);
                    max_score[h3] = simd_shuffle(new_max, ushort(31));
                }
            }
            const device T* vrow = vg + size_t(base) * vst + 4 * lane;
            if (all_fast) {
                for (int t = 0; t < 32; ++t) {
                    const uint2 vb = *(const device uint2*)(vrow + size_t(t) * vstep);
                    const float vv[4] = {as_type<float>(vb.x << 16), as_type<float>(vb.x & 0xffff0000u),
                                         as_type<float>(vb.y << 16), as_type<float>(vb.y & 0xffff0000u)};
                    for (int h3 = 0; h3 < 3; ++h3) {
                        const float exp_score = simd_shuffle(ex[h3], ushort(t));
                        sum_exp_score[h3] = sum_exp_score[h3] + exp_score;
                        for (int j = 0; j < 4; j++) {
                            #pragma clang fp contract(off)
                            const float ev = exp_score * vv[j];
                            o[h3][j] = o[h3][j] + ev;
                        }
                    }
                }
            } else {
                for (int t = 0; t < 32; ++t) {
                    const uint2 vb = *(const device uint2*)(vrow + size_t(t) * vstep);
                    const float vv[4] = {as_type<float>(vb.x << 16), as_type<float>(vb.x & 0xffff0000u),
                                         as_type<float>(vb.y << 16), as_type<float>(vb.y & 0xffff0000u)};
                    for (int h3 = 0; h3 < 3; ++h3) {
                        const float factor = simd_shuffle(fac[h3], ushort(t));
                        const float exp_score = simd_shuffle(ex[h3], ushort(t));
                        sum_exp_score[h3] = sum_exp_score[h3] * factor + exp_score;
                        for (int j = 0; j < 4; j++) {
                            o[h3][j] = o[h3][j] * factor + exp_score * vv[j];
                        }
                    }
                }
            }
        }
        p_from += tiles * 32 * span;                          // the rest goes through MLX's loop
    }
    if (LONG == 2 && two && p_from < pos0) {
        // The same order as LONG == 1 with less work a key (3 rows or more): q read as float4,
        // keys in 16-input pieces with the lane tree kept as running partial sums (fewer live
        // registers),
        // exp_score handed to the lanes through threadgroup memory (one load for three heads),
        // value rows loaded four keys ahead.
        threadgroup float4* qr4 = (threadgroup float4*)(qsh + r * 3 * 128);
        threadgroup float4* es = (threadgroup float4*)(esh + r * 32 * 4);
        for (int h3 = 0; h3 < 3; ++h3) {
            qr4[h3 * 32 + lane] = float4(qs[h3][0], qs[h3][1], qs[h3][2], qs[h3][3]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const int ncom = (pos0 - 1 - p_from) / span + 1;
        const int tiles = ncom >> 5;
        const size_t kst = size_t(kc_strides[2]);
        const size_t vst2 = size_t(vc_strides[2]) / 4;
        const size_t vstep2 = size_t(span) * vst2;
        for (int tix = 0; tix < tiles; ++tix) {
            const int base = p_from + tix * 32 * span;
            const int pt = base + lane * span;
            const device uint4* kp = (const device uint4*)(kg + size_t(pt) * kst);
            float t01[3], t2[3], sc[3];
            for (int cc = 0; cc < 4; ++cc) {
                float s8[3];
                for (int hh = 0; hh < 2; ++hh) {
                    const uint4 w0 = kp[4 * cc + 2 * hh];
                    const uint4 w1 = kp[4 * cc + 2 * hh + 1];
                    const float kf[16] = {
                        as_type<float>(w0.x << 16), as_type<float>(w0.x & 0xffff0000u),
                        as_type<float>(w0.y << 16), as_type<float>(w0.y & 0xffff0000u),
                        as_type<float>(w0.z << 16), as_type<float>(w0.z & 0xffff0000u),
                        as_type<float>(w0.w << 16), as_type<float>(w0.w & 0xffff0000u),
                        as_type<float>(w1.x << 16), as_type<float>(w1.x & 0xffff0000u),
                        as_type<float>(w1.y << 16), as_type<float>(w1.y & 0xffff0000u),
                        as_type<float>(w1.z << 16), as_type<float>(w1.z & 0xffff0000u),
                        as_type<float>(w1.w << 16), as_type<float>(w1.w & 0xffff0000u)};
                    for (int h3 = 0; h3 < 3; ++h3) {
                        float part[4];
                        for (int m4 = 0; m4 < 4; ++m4) {
                            const float4 qq = qr4[h3 * 32 + 8 * cc + 4 * hh + m4];
                            float score = 0;
                            score += qq.x * kf[4 * m4 + 0];
                            score += qq.y * kf[4 * m4 + 1];
                            score += qq.z * kf[4 * m4 + 2];
                            score += qq.w * kf[4 * m4 + 3];
                            part[m4] = score;
                        }
                        const float s4 = (part[0] + part[1]) + (part[2] + part[3]);
                        s8[h3] = hh == 0 ? s4 : (s8[h3] + s4);
                    }
                }
                for (int h3 = 0; h3 < 3; ++h3) {
                    if (cc == 0) {
                        t01[h3] = s8[h3];
                    } else if (cc == 1) {
                        t01[h3] = t01[h3] + s8[h3];
                    } else if (cc == 2) {
                        t2[h3] = s8[h3];
                    } else {
                        sc[h3] = t01[h3] + (t2[h3] + s8[h3]);
                    }
                }
            }
            const int dist = P - pt;
            float ex[3], fac[3];
            bool all_fast = true;
            for (int h3 = 0; h3 < 3; ++h3) {
                float score = sc[h3];
                score += dist < 1024 ? static_cast<float>(pr[h3 * 1024 + dist]) : 0.0f;
                const float tmax = simd_max(score);
                if (tmax <= max_score[h3]) {
                    fac[h3] = 1.0f;
                    ex[h3] = metal::fast::exp(score - max_score[h3]);
                } else {
                    all_fast = false;
                    float run = score;
                    for (ushort d2 = 1; d2 < 32; d2 <<= 1) {
                        const float other = simd_shuffle_up(run, d2);
                        if (lane >= int(d2)) run = max(run, other);
                    }
                    const float new_max = max(max_score[h3], run);
                    float before = simd_shuffle_up(run, ushort(1));
                    before = lane == 0 ? max_score[h3] : max(max_score[h3], before);
                    fac[h3] = metal::fast::exp(before - new_max);
                    ex[h3] = metal::fast::exp(score - new_max);
                    max_score[h3] = simd_shuffle(new_max, ushort(31));
                }
            }
            es[lane] = float4(ex[0], ex[1], ex[2], 0.0f);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const device uint2* vrow = (const device uint2*)(vg + size_t(base) * size_t(vc_strides[2]) + 4 * lane);
            if (all_fast) {
                for (int t0 = 0; t0 < 32; t0 += 4) {
                    uint2 vb[4];
                    for (int u = 0; u < 4; ++u) {
                        vb[u] = vrow[size_t(t0 + u) * vstep2];
                    }
                    for (int u = 0; u < 4; ++u) {
                        const float4 e4 = es[t0 + u];
                        const float vv[4] = {as_type<float>(vb[u].x << 16), as_type<float>(vb[u].x & 0xffff0000u),
                                             as_type<float>(vb[u].y << 16), as_type<float>(vb[u].y & 0xffff0000u)};
                        sum_exp_score[0] = sum_exp_score[0] + e4.x;
                        sum_exp_score[1] = sum_exp_score[1] + e4.y;
                        sum_exp_score[2] = sum_exp_score[2] + e4.z;
                        for (int j = 0; j < 4; j++) {
                            #pragma clang fp contract(off)
                            const float ev0 = e4.x * vv[j];
                            const float ev1 = e4.y * vv[j];
                            const float ev2 = e4.z * vv[j];
                            o[0][j] = o[0][j] + ev0;
                            o[1][j] = o[1][j] + ev1;
                            o[2][j] = o[2][j] + ev2;
                        }
                    }
                }
            } else {
                for (int t = 0; t < 32; ++t) {
                    const uint2 vb = vrow[size_t(t) * vstep2];
                    const float vv[4] = {as_type<float>(vb.x << 16), as_type<float>(vb.x & 0xffff0000u),
                                         as_type<float>(vb.y << 16), as_type<float>(vb.y & 0xffff0000u)};
                    const float4 e4 = es[t];
                    const float e3[3] = {e4.x, e4.y, e4.z};
                    for (int h3 = 0; h3 < 3; ++h3) {
                        const float factor = simd_shuffle(fac[h3], ushort(t));
                        const float exp_score = e3[h3];
                        sum_exp_score[h3] = sum_exp_score[h3] * factor + exp_score;
                        for (int j = 0; j < 4; j++) {
                            o[h3][j] = o[h3][j] * factor + exp_score * vv[j];
                        }
                    }
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        p_from += tiles * 32 * span;
        for (int h3 = 0; h3 < 3; ++h3) {                      // q again for MLX's loop below
            const float4 qq = qr4[h3 * 32 + lane];
            qs[h3][0] = qq.x; qs[h3][1] = qq.y; qs[h3][2] = qq.z; qs[h3][3] = qq.w;
        }
    }
    // Keys of this residue in [first, P], ascending (key p has logical index p - first).
    for (int p = p_from; p <= P; p += span) {
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


_WINDOW_ONE = """
    // One threadgroup per (row, HPT heads): all of the row's MLX blocks for those heads,
    // folded in registers, then MLX's own merge in threadgroup memory, then RC6's gate.
    const int hgrp = int(threadgroup_position_in_grid.x);
    const int r = int(threadgroup_position_in_grid.y);
    const int sg = int(simdgroup_index_in_threadgroup);       // 32 simdgroups
    const int lane = int(thread_index_in_simdgroup);
    const int g = HPT == 3 ? hgrp : hgrp / 3;                 // key/value head
    const int h0 = HPT == 3 ? 3 * hgrp : hgrp;                // first query head
    const int cap = int(kc_shape[2]);
    const int pos0 = int(pos[0]);
    const int cnt = int(count[0]);
    const int nw = int(newest[0]);
    const int P = pos0 + r;
    int nc = min(cnt, 1023 - r);
    nc = max(nc, 0);
    const int n = nc + r + 1;
    const bool two = (n + 1) >= THRESH;
    const int span = two ? BLOCKS : 32;
    const int kps = two ? BLOCKS / 32 : 1;                    // chains of this simdgroup
    const int first = pos0 - nc;
    const float scale = as_type<float>(uint(SCALE));

    constexpr int KMAX = BLOCKS / 32;                         // chains a simdgroup at most
    float qs[HPT][4];
    float o[HPT][KMAX][4], mx_[HPT][KMAX], sm[HPT][KMAX];
    for (int j = 0; j < HPT; ++j) {
        for (int i = 0; i < 4; ++i) {
            qs[j][i] = static_cast<float>(scale) * static_cast<float>(
                q[r * q_strides[1] + (h0 + j) * q_strides[2] + 4 * lane + i]);
        }
        for (int k = 0; k < KMAX; ++k) {
            mx_[j][k] = -FLT_MAX;
            sm[j][k] = 0;
            for (int i = 0; i < 4; ++i) o[j][k][i] = 0;
        }
    }
    const device T* kg = kc + size_t(g) * kc_strides[1];
    const device T* vg = vc + size_t(g) * vc_strides[1];
    const device T* krg = kr + size_t(g) * kr_strides[1];
    const device T* vrg = vr + size_t(g) * vr_strides[1];
    const device T* pr = prof + (size_t(r) * 24 + size_t(h0)) * 1024;
    // Keys of logical index i = sg + 32 m, i.e. chains sg, sg + 32, ... in turn.
    for (int i = sg; i < n; i += 32) {
        const int k = two ? (i / 32) % kps : 0;
        const int p = first + i;
        const device T* kp;
        const device T* vp;
        if (p < pos0) {
            int slot = nw - (pos0 - 1 - p);
            if (slot < 0) slot += cap;
            kp = kg + size_t(slot) * kc_strides[2];
            vp = vg + size_t(slot) * vc_strides[2];
        } else {
            kp = krg + size_t(p - pos0) * kr_strides[2];
            vp = vrg + size_t(p - pos0) * vr_strides[2];
        }
        float kv[4], vv[4];
        for (int t = 0; t < 4; ++t) {
            kv[t] = static_cast<float>(kp[4 * lane + t]);
            vv[t] = static_cast<float>(vp[4 * lane + t]);
        }
        const int dist = P - p;                               // < 1024 on a window visit
        for (int j = 0; j < HPT; ++j) {
            const float bias = static_cast<float>(pr[j * 1024 + dist]);
            for (int kk = 0; kk < KMAX; ++kk) {
                if (kk == k) {
                    fold_key(qs[j], kv, vv, bias, mx_[j][kk], sm[j][kk], o[j][kk]);
                }
            }
        }
    }
    // RC6's appended zero row: logical index n, last in its chain.
    if ((n % 32) == sg) {
        const int k = two ? (n / 32) % kps : 0;
        float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int j = 0; j < HPT; ++j) {
            const float bias = static_cast<float>(sink[r * sink_strides[1] + (h0 + j) * sink_strides[2]]);
            for (int kk = 0; kk < KMAX; ++kk) {
                if (kk == k) {
                    fold_key(qs[j], zero, zero, bias, mx_[j][kk], sm[j][kk], o[j][kk]);
                }
            }
        }
    }
    threadgroup float tmax[HPT][128];
    threadgroup float tsum[HPT][128];
    threadgroup float outputs[32 * 32];
    if (lane == 0) {
        for (int j = 0; j < HPT; ++j) {
            for (int k = 0; k < kps; ++k) {
                tmax[j][sg + 32 * k] = mx_[j][k];
                tsum[j][sg + 32 * k] = sm[j][k];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = 0; j < HPT; ++j) {
        float out4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float sum_exp_score;
        if (two) {
            // MLX sdpa_vector_2pass_2; this simdgroup's partials are blocks sg + 32 b.
            float max_score = -FLT_MAX;
            for (int b = 0; b < BLOCKS / 32; ++b) {
                max_score = max(max_score, tmax[j][lane + 32 * b]);
            }
            max_score = simd_max(max_score);
            sum_exp_score = 0;
            for (int b = 0; b < BLOCKS / 32; ++b) {
                float factor = metal::fast::exp(tmax[j][lane + 32 * b] - max_score);
                sum_exp_score += factor * tsum[j][lane + 32 * b];
            }
            sum_exp_score = simd_sum(sum_exp_score);
            for (int b = 0; b < BLOCKS / 32; ++b) {
                float factor = metal::fast::exp(tmax[j][sg + 32 * b] - max_score);
                for (int i = 0; i < 4; ++i) {
                    // Pass one writes its partials in the query type.
                    out4[i] += factor * static_cast<float>(static_cast<T>(o[j][b][i]));
                }
            }
            for (int i = 0; i < 4; ++i) {
                outputs[lane * 32 + sg] = out4[i];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                out4[i] = simd_sum(outputs[sg * 32 + lane]);
                out4[i] = sum_exp_score == 0 ? out4[i] : (out4[i] / sum_exp_score);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        } else {
            // MLX sdpa_vector's combine: simdgroup sg holds chain sg.
            float max_score = tmax[j][lane];
            float new_max = simd_max(max_score);
            float factor = metal::fast::exp(max_score - new_max);
            sum_exp_score = simd_sum(tsum[j][lane] * factor);
            for (int i = 0; i < 4; ++i) {
                outputs[lane * 32 + sg] = o[j][0][i];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                out4[i] = simd_sum(outputs[sg * 32 + lane] * factor);
                out4[i] = sum_exp_score == 0 ? out4[i] : (out4[i] / sum_exp_score);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        if (lane == 0) {
            for (int i = 0; i < 4; ++i) {
                const int d = 4 * sg + i;
                const size_t at = size_t(r) * 3072 + size_t(h0 + j) * 128 + size_t(d);
                const float hv = static_cast<float>(static_cast<T>(out4[i]));
                const float x = static_cast<float>(gate[r * gate_strides[1] + ((h0 + j) * 128 + d) * gate_strides[2]]);
                const float y = 1 / (1 + metal::precise::exp(metal::abs(x)));
                const float gv = (x < 0) ? y : 1 - y;
                hg[at] = static_cast<T>(hv * gv);
            }
        }
    }
"""



def attend_window(q, rel, sink, gate_logits, k_rows, v_rows, keys, values, count, newest, pos,
                  rel_basis, *, plan=None):
    """Exact window attention in one launch (plus RC6's profile product): same output as
    ``attend(..., window=True)``, bit for bit."""
    rows = q.shape[1]
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"A panel holds 1 to {MAX_ROWS} rows, got {rows}")
    threshold, blocks = plan or reduction_plan()
    if blocks > 128:
        raise ValueError("window attention keeps at most 4 MLX blocks a simdgroup")
    dtype = q.dtype
    prof = profiles(rel, rel_basis)
    hpt = 1 if rows <= 2 else 3                 # heads a threadgroup: parallelism vs key reuse
    (hg,) = _kernel("mtplx_inkling_fast_attend_window", _WINDOW_ONE, _HEADER)(
        inputs=[q, keys, values, k_rows, v_rows, prof, sink, gate_logits, _u32(pos), _u32(count), _u32(newest)],
        template=[("T", dtype), ("HPT", hpt), ("BLOCKS", blocks), ("THRESH", threshold), ("SCALE", SCALE_BITS)],
        grid=(1024 * (HEADS // hpt), rows, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(1, rows, HEADS * HEAD_DIM)],
        output_dtypes=[dtype],
    )
    return hg


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
_INPUTS["mtplx_inkling_fast_attend_window"] = ["q", "kc", "vc", "kr", "vr", "prof", "sink", "gate", "pos", "count", "newest"]
_OUTPUTS["mtplx_inkling_fast_attend_window"] = ["hg"]


def _u32(value):
    if isinstance(value, mx.array):
        return value.astype(mx.uint32).reshape(1) if value.dtype != mx.uint32 or value.ndim != 1 else value
    return mx.array([int(value)], mx.uint32)


def profiles(rel, rel_basis):
    """RC6's relative profile ``rel @ rel_basis`` (bf16), one MLX product for the panel."""
    return rel @ rel_basis.astype(rel.dtype)


def attend(q, rel, sink, gate_logits, k_rows, v_rows, keys, values, count, newest, pos,
           rel_basis, *, window, plan=None, one_launch_window=False, long=None):
    """The gated attention result of a panel, ``(1, W, 3072)`` in the query dtype.

    ``q (1, W, 24, 128)``, ``rel (1, W, 24, 16)``, ``sink (1, W, 24)``,
    ``gate_logits (1, W, 3072)`` and ``k_rows``, ``v_rows (1, 8, W, 128)`` come from stage A.
    ``keys``/``values`` are the committed history, read in place: ``window=True`` for a
    ring in physical order with ``newest``/``count``, ``window=False`` for a bank whose
    first ``count`` slots are positions ``0 .. count - 1``. ``count``, ``newest`` and
    ``pos`` are uint32 arrays (or ints, for tests); they never become shapes.

    ``long`` (default: on for global visits of bf16 panels up to 8 rows, the order by
    ``long_order``; 1 or 2 to choose) takes the exact long-history tile order for the committed
    keys; ``long=False`` keeps MLX's per-key order everywhere. All give RC6's bits.

    Raises ``FusedAttentionDeclined`` where ``decline_reason`` is not ``None`` (a reduction
    this route does not serve, or bits that differ from MLX's on this Mac): call
    ``decline_reason`` at warm-up and run RC6's stage, counted, instead.
    """
    plan = plan or reduction_plan()
    reason = decline_reason(q.dtype, plan)
    if reason is not None:
        raise FusedAttentionDeclined(f"fused attention declined on this Mac: {reason} ({DECLINED[(str(q.dtype), plan)][1]})")
    return _attend(q, rel, sink, gate_logits, k_rows, v_rows, keys, values, count, newest, pos,
                   rel_basis, window=window, plan=plan, one_launch_window=one_launch_window, long=long)


def _attend(q, rel, sink, gate_logits, k_rows, v_rows, keys, values, count, newest, pos,
            rel_basis, *, window, plan, one_launch_window=False, long=None):
    rows = q.shape[1]
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"A panel holds 1 to {MAX_ROWS} rows, got {rows}")
    threshold, blocks = plan or reduction_plan()
    # One launch for a window visit is exact too, but measured slower than two at 1 to 8 rows
    # (lanes/l2/bench-window-1launch.json): too little parallelism per threadgroup.
    if window and blocks <= 128 and one_launch_window:
        return attend_window(q, rel, sink, gate_logits, k_rows, v_rows, keys, values, count, newest,
                             pos, rel_basis, plan=(threshold, blocks))
    dtype = q.dtype
    if long is None or long is True:
        long = long_order(rows)
    prof = profiles(rel, rel_basis)
    pos_a, count_a, newest_a = _u32(pos), _u32(count), _u32(newest)
    template = [("T", dtype), ("WINDOW", bool(window)), ("BLOCKS", blocks),
                ("THRESH", threshold), ("SCALE", SCALE_BITS),
                ("ROWS", rows), ("LONG", int(long) if (long and not window and rows <= 8 and dtype == mx.bfloat16
                                                    and lane_tree_ok()) else 0)]
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


# ----------------------------------------------------------------------------------------
# Class C variant for long histories (bounded rounding against RC6, row-stable by design).
#
# The exact path repeats MLX's per-key arithmetic: a cross-lane sum per key and query and a
# rescale of the running output after every key. With 4 to 8 rows that order is compute
# bound (about 50 instructions per key and query). This variant reads the same bytes once
# per panel and changes only the order of additions: the history is cut into fixed chunks
# of CHUNK absolute positions (so the order depends neither on the panel width nor on the
# bank capacity); one threadgroup per (chunk, key/value head, group of 8 rows) stages 32-key
# tiles of keys and values in threadgroup memory once for its rows and the three query heads
# that share them; each simdgroup owns three query vectors from scores to weighted sum, so
# only the tile loads need a threadgroup barrier; the running output is rescaled once per
# tile. A second launch merges the chunks in ascending order, adds the sink, and applies
# RC6's gate. Every per-(row, head) order is fixed by absolute positions, so a row's bits do
# not depend on how many rows share the launch.
# ----------------------------------------------------------------------------------------

CHUNK = 1024

# One simdgroup per block of 8 query vectors (the panel's rows x 3 heads, packed: query
# qq = 3 * row + head). Matrix products on 8x8 simdgroup matrices: scores S^T = K . Q^T over
# 16 blocks of 8 dims, weighted sum O += P . V over 16 blocks of 8 dims. This GPU's matrix
# product gives each element the same arithmetic wherever its row and column sit
# (`l2/probe-mma.json`), so packing rows does not break row stability. No fragment layout is
# assumed: per-element work (bias, mask, softmax) goes through a small threadgroup tile, and
# the per-row rescale is loaded as a matrix with constant rows, so it lands in the
# accumulator's own lane layout.
_PARTIALS_C = """
    const int c = int(threadgroup_position_in_grid.x);     // chunk of CHUNK positions
    const int g = int(threadgroup_position_in_grid.y);     // key/value head
    const int sgi = int(simdgroup_index_in_threadgroup);
    const int qb = QB * int(threadgroup_position_in_grid.z) + sgi;   // block of 8 query vectors
    const int lane = int(thread_index_in_simdgroup);
    const int tid = int(thread_index_in_threadgroup);
    const int nthreads = 32 * QB;
    const int rows = int(q_shape[1]);
    const int nq = 3 * rows;
    const int pos0 = int(pos[0]);
    const int cnt = int(count[0]);
    const int nw = int(newest[0]);
    const int cap = int(kc_shape[2]);
    const int last = pos0 + rows - 1;
    const int lo_pos = c * CHUNK;
    if (lo_pos > last) {
        return;                                // the whole threadgroup leaves together
    }
    const int hi_pos = min(lo_pos + CHUNK - 1, last);
    const int lo_any = WINDOW ? pos0 - max(min(cnt, 1023), 0) : 0;
    const float scale = as_type<float>(uint(SCALE));
    const bool block_live = 8 * qb < nq;       // padding simdgroups still join the barriers

    threadgroup float ks[8 * 128];             // the tile's keys, shared by the query blocks
    threadgroup float vs[8 * 128];             // the tile's values
    threadgroup float tile_all[QB][64];
    threadgroup float prob_all[QB][64];
    threadgroup float fac_all[QB][64];
    threadgroup float* tl = tile_all[sgi];
    threadgroup float* pt = prob_all[sgi];
    threadgroup float* fc = fac_all[sgi];

    // Softmax lanes: query (lane / 4) of the block, keys 2 * (lane % 4) and the next.
    const int qi = lane / 4, kp2 = 2 * (lane % 4);
    const int qq = 8 * qb + qi;
    const bool live = block_live && qq < nq;
    const int qrow = qq / 3, qhead = 3 * g + qq % 3;
    int first_q = 0, last_q = -1;
    if (live) {
        int nc = WINDOW ? min(cnt, 1023 - qrow) : cnt;
        nc = max(nc, 0);
        first_q = pos0 - nc;
        last_q = pos0 + qrow;
    }
    const device T* pr = prof + (size_t(min(qrow, rows - 1)) * 24 + size_t(qhead)) * 1024;
    float m_run = -INFINITY, l_run = 0.0f;

    // Q^T blocks (dims x queries), scaled as MLX scales the query; staged through ks.
    simdgroup_float8x8 qt[16];
    for (int s2 = 0; s2 < QB; ++s2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < 8 * 128; i += nthreads) {
            const int qi2 = i / 128, d = i % 128;
            const int q2 = 8 * (QB * int(threadgroup_position_in_grid.z) + s2) + qi2;
            ks[i] = q2 < nq ? scale * static_cast<float>(
                q[(q2 / 3) * q_strides[1] + (3 * g + q2 % 3) * q_strides[2] + d]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (s2 == sgi) {
            for (int b = 0; b < 16; ++b) {
                simdgroup_load(qt[b], ks + 8 * b, 128, ulong2(0, 0), true);
            }
        }
    }
    simdgroup_float8x8 acc_o[16];
    for (int b = 0; b < 16; ++b) {
        acc_o[b] = simdgroup_float8x8(0.0f);
    }
    const device T* kg = kc + size_t(g) * kc_strides[1];
    const device T* vg = vc + size_t(g) * vc_strides[1];
    const device T* krg = kr + size_t(g) * kr_strides[1];
    const device T* vrg = vr + size_t(g) * vr_strides[1];

    // A tile is 256 sixteen-byte pieces (8 keys x 16 of keys, then of values), spread over
    // the threadgroup; each thread keeps its next pieces in flight while the tile computes.
    constexpr int PIECES = 256;
    constexpr int PER = (PIECES + 32 * QB - 1) / (32 * QB);
    uint4 pre[PER];
    #define FETCH(T0)                                                                     \\
    for (int u = 0; u < PER; ++u) {                                                       \\
        const int piece = tid + u * nthreads;                                             \\
        pre[u] = uint4(0);                                                                \\
        if (piece < PIECES) {                                                             \\
            const int is_v = piece >> 7, key = (piece >> 4) & 7, part = piece & 15;       \\
            const int p = (T0) + key;                                                     \\
            if (p <= last && p >= lo_any && p >= 0) {                                     \\
                const device T* src;                                                      \\
                if (p < pos0) {                                                           \\
                    int slot = p;                                                         \\
                    if (WINDOW) { slot = nw - (pos0 - 1 - p); if (slot < 0) slot += cap; } \\
                    src = (is_v ? vg : kg) + size_t(slot) * (is_v ? vc_strides[2] : kc_strides[2]); \\
                } else {                                                                  \\
                    src = (is_v ? vrg : krg) + size_t(p - pos0) * (is_v ? vr_strides[2] : kr_strides[2]); \\
                }                                                                         \\
                pre[u] = *(const device uint4*)(src + 8 * part);                          \\
            }                                                                             \\
        }                                                                                 \\
    }

    FETCH(lo_pos)
    for (int t0 = lo_pos; t0 <= hi_pos; t0 += 8) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int u = 0; u < PER; ++u) {
            const int piece = tid + u * nthreads;
            if (piece < PIECES) {
                const int is_v = piece >> 7, key = (piece >> 4) & 7, part = piece & 15;
                threadgroup float* dst = (is_v ? vs : ks) + key * 128 + 8 * part;
                const uint4 w = pre[u];
                dst[0] = as_type<float>(w.x << 16); dst[1] = as_type<float>(w.x & 0xffff0000u);
                dst[2] = as_type<float>(w.y << 16); dst[3] = as_type<float>(w.y & 0xffff0000u);
                dst[4] = as_type<float>(w.z << 16); dst[5] = as_type<float>(w.z & 0xffff0000u);
                dst[6] = as_type<float>(w.w << 16); dst[7] = as_type<float>(w.w & 0xffff0000u);
            }
        }
        if (t0 + 8 <= hi_pos) {
            FETCH(t0 + 8)                     // next tile in flight while this one computes
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!block_live) {
            continue;
        }
        // Scores S^T (8 keys x 8 queries) on the matrix units, 16 blocks of 8 dims in order.
        simdgroup_float8x8 s_t = simdgroup_float8x8(0.0f);
        for (int b = 0; b < 16; ++b) {
            simdgroup_float8x8 kb;
            simdgroup_load(kb, ks + 8 * b, 128);
            simdgroup_multiply_accumulate(s_t, kb, qt[b], s_t);
        }
        simdgroup_store(s_t, tl, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // Tile softmax: 4 lanes a query, 2 keys a lane, lanes combined pairwise (xor 1, 2).
        float e0 = -INFINITY, e1 = -INFINITY;
        {
            const int p0 = t0 + kp2, p1 = p0 + 1;
            if (live && p0 >= first_q && p0 <= last_q) {
                const int dist = last_q - p0;
                e0 = tl[kp2 * 8 + qi] + (dist < 1024 ? static_cast<float>(pr[dist]) : 0.0f);
            }
            if (live && p1 >= first_q && p1 <= last_q) {
                const int dist = last_q - p1;
                e1 = tl[(kp2 + 1) * 8 + qi] + (dist < 1024 ? static_cast<float>(pr[dist]) : 0.0f);
            }
        }
        float tile_max = max(e0, e1);
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, ushort(1)));
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, ushort(2)));
        const float m_new = max(m_run, tile_max);
        const float alpha = m_new == -INFINITY ? 1.0f : metal::fast::exp(m_run - m_new);
        e0 = e0 == -INFINITY ? 0.0f : metal::fast::exp(e0 - m_new);
        e1 = e1 == -INFINITY ? 0.0f : metal::fast::exp(e1 - m_new);
        float part_sum = e0 + e1;
        part_sum = part_sum + simd_shuffle_xor(part_sum, ushort(1));
        part_sum = part_sum + simd_shuffle_xor(part_sum, ushort(2));
        l_run = l_run * alpha + part_sum;
        m_run = m_new;
        pt[qi * 8 + kp2] = e0;                 // P (queries x keys), row-major
        pt[qi * 8 + kp2 + 1] = e1;
        fc[qi * 8 + kp2] = alpha;              // constant rows: the accumulator's own layout
        fc[qi * 8 + kp2 + 1] = alpha;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 p8, f8;
        simdgroup_load(p8, pt, 8);
        simdgroup_load(f8, fc, 8);
        // O = O * alpha + P . V on the matrix units, 16 blocks of 8 dims.
        for (int b = 0; b < 16; ++b) {
            simdgroup_float8x8 vb;
            simdgroup_load(vb, vs + 8 * b, 128);
            acc_o[b].thread_elements()[0] *= f8.thread_elements()[0];
            acc_o[b].thread_elements()[1] *= f8.thread_elements()[1];
            simdgroup_multiply_accumulate(acc_o[b], p8, vb, acc_o[b]);
        }
    }
    #undef FETCH
    if (!block_live) {
        return;
    }
    // Partials, layout [chunk][key/value head][query of the head][dim]: the block's 8 query
    // rows are contiguous, so each output block is one strided store to device memory.
    const int nqpad = 8 * ((nq + 7) / 8);
    const size_t row0 = (size_t(c) * 8 + size_t(g)) * size_t(nqpad) + size_t(8 * qb);
    for (int b = 0; b < 16; ++b) {
        simdgroup_store(acc_o[b], po + row0 * 128 + 8 * b, 128);
    }
    if (live && (lane % 4) == 0) {
        pm[row0 + qi] = m_run;
        pl[row0 + qi] = l_run;
    }
"""


_MERGE_C = """
    const int h = int(threadgroup_position_in_grid.x);
    const int r = int(threadgroup_position_in_grid.y);
    const int d = int(thread_position_in_threadgroup.x);   // 128 threads: one dim each
    const int rows = int(gate_shape[1]);
    const int nqpad = 8 * ((3 * rows + 7) / 8);
    const int g = h / 3;
    const int q2 = 3 * r + h % 3;
    const int pos0 = int(pos[0]);
    const int cnt = int(count[0]);
    int nc = WINDOW ? min(cnt, 1023 - r) : cnt;
    nc = max(nc, 0);
    const int first = pos0 - nc;
    const int P = pos0 + r;
    const int c_lo = first / CHUNK, c_hi = P / CHUNK;
    const float sink_logit = static_cast<float>(sink[r * sink_strides[1] + h * sink_strides[2]]);
    float big = sink_logit;
    for (int c = c_lo; c <= c_hi; ++c) {
        big = max(big, pm[(size_t(c) * 8 + size_t(g)) * size_t(nqpad) + size_t(q2)]);
    }
    float denom = 0.0f, acc = 0.0f;
    for (int c = c_lo; c <= c_hi; ++c) {
        const size_t at = (size_t(c) * 8 + size_t(g)) * size_t(nqpad) + size_t(q2);
        const float mc = pm[at];
        const float w = mc == -INFINITY ? 0.0f : metal::fast::exp(mc - big);
        denom = metal::fma(pl[at], w, denom);
        acc = metal::fma(po[at * 128 + d], w, acc);
    }
    denom += metal::fast::exp(sink_logit - big);   // the sink: mass, no value
    const float hv = static_cast<float>(static_cast<T>(acc / denom));
    const size_t at = size_t(r) * 3072 + size_t(h) * 128 + size_t(d);
    const float x = static_cast<float>(gate[r * gate_strides[1] + (h * 128 + d) * gate_strides[2]]);
    const float y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    const float gv = (x < 0) ? y : 1 - y;
    hg[at] = static_cast<T>(hv * gv);
"""

_INPUTS["mtplx_inkling_fast_attend_partials_c"] = ["q", "kc", "vc", "kr", "vr", "prof", "pos", "count", "newest"]
_OUTPUTS["mtplx_inkling_fast_attend_partials_c"] = ["po", "pm", "pl"]
_INPUTS["mtplx_inkling_fast_attend_merge_c"] = ["po", "pm", "pl", "sink", "gate", "pos", "count"]
_OUTPUTS["mtplx_inkling_fast_attend_merge_c"] = ["hg"]


def attend_c(q, rel, sink, gate_logits, k_rows, v_rows, keys, values, count, newest, pos,
             rel_basis, *, window):
    """Class C sibling of ``attend`` for global visits: same inputs and output.

    Bounded rounding against RC6 (a different, fixed order of additions); row-stable by
    construction (every per-(row, head) order is set by absolute key positions: chunks of
    ``CHUNK``, tiles of 32, dims in order, butterfly lane sums), independent of the panel
    width and of the bank capacity.
    """
    if window:
        raise NotImplementedError("attend_c serves global visits; window visits use attend")
    rows = q.shape[1]
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"A panel holds 1 to {MAX_ROWS} rows, got {rows}")
    dtype = q.dtype
    if dtype != mx.bfloat16:
        raise ValueError("attend_c reads bf16 keys and values")
    prof = profiles(rel, rel_basis)
    pos_a, count_a, newest_a = _u32(pos), _u32(count), _u32(newest)
    nch = -(-(keys.shape[2] + 32) // CHUNK)     # every position the bank can hold
    qblocks = -(-(GROUP * rows) // 8)          # blocks of 8 query vectors
    per_group = min(qblocks, 3)                 # simdgroups a threadgroup (threadgroup memory)
    groups = -(-qblocks // per_group)
    chains = nch * KV_HEADS * 8 * qblocks       # [chunk][kv head][query of the head, padded]
    po, pm, pl = _kernel("mtplx_inkling_fast_attend_partials_c", _PARTIALS_C, "#include <metal_simdgroup_matrix>\n")(
        inputs=[q, keys, values, k_rows, v_rows, prof, pos_a, count_a, newest_a],
        template=[("T", dtype), ("WINDOW", False), ("CHUNK", CHUNK), ("SCALE", SCALE_BITS), ("QB", per_group)],
        grid=(32 * per_group * nch, KV_HEADS, groups),
        threadgroup=(32 * per_group, 1, 1),
        output_shapes=[(chains * HEAD_DIM,), (chains,), (chains,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    (hg,) = _kernel("mtplx_inkling_fast_attend_merge_c", _MERGE_C)(
        inputs=[po, pm, pl, sink, gate_logits, pos_a, count_a],
        template=[("T", dtype), ("WINDOW", False), ("CHUNK", CHUNK)],
        grid=(128 * HEADS, rows, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, rows, HEADS * HEAD_DIM)],
        output_dtypes=[dtype],
    )
    return hg
