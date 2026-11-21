"""Inkling 2 Mini expert block (stages D and E of a layer visit) in four launches for a panel.

A visit's tail in RC6 (`model.py` 533-536) is: the MLP norm, the router product, top 8,
the expert input rounding, two gathered quantised products with glue around them, the
combine, the MLP convolution and the residual add: 13 launches, and each verify row
re-reads all eight of its experts. Here the same arithmetic runs in four launches for
1 to 32 rows, and each distinct expert of the panel is read once:

    inputs   RMS norm, bf16 round, router logits, expert input rounding (FP4 / FP8)
    gate_up  top 8 per row, the panel's distinct experts, gate/up product on the native
             codes read in place, capped SwiGLU, per-tile peaks for the down rounding
    down     down-input rounding per (row, expert), down product on the native codes
    combine  outer scale, round, route weight, ordered 8-way sum, MLP convolution with
             the committed tail, 32-bit add with `partial`, cast

Exactness (class A, proven per specialisation by bits against RC6's own statements):
every operation repeats the order of the MLX 0.32.2 kernel RC6 runs there.
  * norm: `rms_single_row` (576 threads of 4 values, fused square-accumulate from zero,
    two levels of `simd_sum`, `precise::rsqrt`, `w * (x * inv)`), then the bf16 cast;
  * router: the one-row `gemv` (bm1 bn8 sm1 sn32 tm4 tn4): blocks of 1024 plus a padded
    tail, per-thread fused multiply-add, shuffle-down tree, then the 8 simdgroup partials
    in order. RC6's 2-8 row projection proves itself against exactly these bits, so the
    logits, and therefore the routing, are identical at every width;
  * top 8, probabilities, input rounding, SwiGLU, combine: RC6's own fused statements
    (`expert_fusion.py`), whose bits RC6 proves against its stock body;
  * products: each (row, expert, output) dot product keeps `gather_qmv`'s lane slices
    (8 values a lane in 256-value blocks for FP4 and FP8 gate-up, 4 a lane in 128-value
    blocks for FP8 down, the half block of FP4 down on lanes 0-15), its statement forms
    (FP4 quads, FP8 one product per value), `scale * accum` then a plain add into the
    lane's sum, and `simd_sum`. Every product of a rounded activation and a code is exact
    in float32, so sharing one decoded weight between the rows that chose its expert
    changes nothing;
  * convolution and residual: `short_conv`'s static form, then one float32 add and a cast.
Row stability holds by construction: no row's arithmetic depends on how many rows a
launch carries.

Kernels are pure: state is read, never written. The caller passes the MLP convolution's
committed tail and gets back the panel's convolution input rows beside the visit output.

Which products serve a width (`products_for`): one row the four launches above; 2 to 8 rows the
panel products (`PANEL`): a route launch writes RC6's routing and a plan of the panel's distinct
experts, then each product threadgroup takes one distinct expert and a tile of output rows and
computes every member pair from the same decoded weights, so each distinct expert is read once
per panel instead of once per pair (a 4-row decode panel names about 19 distinct experts for its
32 pairs). Per pair the arithmetic is MLX's gathered product exactly, so each pair's raw product
is MLX's bit for bit; RC6's SwiGLU and down-input rounding and the combine above are unchanged.
9 to 32 rows take MLX's gathered product with the same glue.
"""

from __future__ import annotations

from functools import lru_cache
import math
import os
import struct
from threading import Lock

import mlx.core as mx

from mtplx.models.inkling import expert_fusion

HIDDEN = 2304
MIDDLE = 1152
EXPERTS = 128
TOP_K = 8
MAX_ROWS = 32
TILE_GATE_UP = 128    # w13 rows per gate-up threadgroup (64 mid outputs), 8 simdgroups
TILE_DOWN = 128       # w2 rows per down threadgroup, 8 simdgroups

_lock = Lock()
_routes: dict[str, int] = {}


def _count(name):
    with _lock:
        _routes[name] = _routes.get(name, 0) + 1


def route_counts():
    with _lock:
        return dict(_routes)


# ----------------------------------------------------------------------------------------
# Metal text
# ----------------------------------------------------------------------------------------

_HELPERS = """
// MLX fp4.h: e2m1 magnitude through half precision, then the sign (code 8 is -0).
inline float l3_e2m1(uint c) {
    return float(as_type<half>(ushort(((c & 7u) << 9) | ((c & 8u) << 12)))) * 16384.0f;
}
// MLX fp8.h: e4m3 through half precision (0x7F decodes to 480, 0x80 to -0).
inline float l3_e4m3(uint b) {
    return float(as_type<half>(ushort(((b & 127u) << 7) | ((b & 128u) << 8)))) * 256.0f;
}
// MLX fp8.h: e8m0 (code 0 is 2^-127).
inline float l3_e8m0(uint b) {
    return as_type<float>(b == 0u ? 0x400000u : (b << 23));
}
// The same codes without MLX's final power-of-two multiply (x 2^14 for e2m1, x 256 for e4m3).
// Products of rounded activations and codes are exact in float32, so a sum of such products
// is exactly 2^-k times MLX's sum; multiplying the sum by 2^k restores MLX's value bit for bit.
inline float l3_e2m1_raw(uint c) {
    return float(as_type<half>(ushort(((c & 7u) << 9) | ((c & 8u) << 12))));
}
inline float l3_e4m3_raw(uint b) {
    return float(as_type<half>(ushort(((b & 127u) << 7) | ((b & 128u) << 8))));
}

// Eight e2m1 codes of a word (code v in bits 4v..4v+3) as the raw table values, two a half2:
// magnitude bits moved to half bits 9-11, the sign to bit 15 (MLX's fp4.h before its x 2^14).
inline void l3_e2m1x8(uint w, thread float* v) {
    const float2 f04 = float2(as_type<half2>(((w & 0x00070007u) << 9) | ((w & 0x00080008u) << 12)));
    const float2 f15 = float2(as_type<half2>(((w & 0x00700070u) << 5) | ((w & 0x00800080u) << 8)));
    const float2 f26 = float2(as_type<half2>(((w & 0x07000700u) << 1) | ((w & 0x08000800u) << 4)));
    const float2 f37 = float2(as_type<half2>(((w & 0x70007000u) >> 3) | (w & 0x80008000u)));
    v[0] = f04.x; v[1] = f15.x; v[2] = f26.x; v[3] = f37.x;
    v[4] = f04.y; v[5] = f15.y; v[6] = f26.y; v[7] = f37.y;
}
// Four e4m3 codes of a word as the raw table values (MLX's fp8.h before its x 256).
inline void l3_e4m3x4(uint w, thread float* v) {
    const float2 f02 = float2(as_type<half2>(((w & 0x007F007Fu) << 7) | ((w & 0x00800080u) << 8)));
    const float2 f13 = float2(as_type<half2>(((w & 0x7F007F00u) >> 1) | (w & 0x80008000u)));
    v[0] = f02.x; v[1] = f13.x; v[2] = f02.y; v[3] = f13.y;
}

// Gate-up dot products of NM member rows with R consecutive w13 rows (gather_qmv lane slices).
template <int NM, int R, bool NV4>
inline void l3_gate_up_dots(const device uint* wexp, const device uchar* sexp, const device float* values,
                            threadgroup const uint* rows_of, uint m0, uint nb, uint lane,
                            thread float (&out)[4][R]) {
    const device float* xm[NM];
    for (int m = 0; m < NM; ++m) xm[m] = values + rows_of[m0 + m] * 2304 + 8 * lane;
    float res[NM][R];
    for (int m = 0; m < NM; ++m)
        for (int q = 0; q < R; ++q) res[m][q] = 0.0f;
    for (uint b = 0; b < 9; ++b) {
        float xv[NM][8];
        for (int m = 0; m < NM; ++m) {
            const float4 a = *(const device float4*)(xm[m] + 256 * b);
            const float4 c = *(const device float4*)(xm[m] + 256 * b + 4);
            xv[m][0] = a.x; xv[m][1] = a.y; xv[m][2] = a.z; xv[m][3] = a.w;
            xv[m][4] = c.x; xv[m][5] = c.y; xv[m][6] = c.z; xv[m][7] = c.w;
        }
        for (int q = 0; q < R; ++q) {
            const uint n = nb + q;
            if (NV4) {
                const uint word = wexp[n * 288 + 32 * b + lane];
                const float s = l3_e4m3(sexp[n * 144 + 16 * b + lane / 2]);
                float w8[8];
                l3_e2m1x8(word, w8);
                for (int m = 0; m < NM; ++m) {
                    float accum = 0;
                    accum += (xv[m][0] * w8[0] + xv[m][1] * w8[1] + xv[m][2] * w8[2] + xv[m][3] * w8[3]);
                    accum += (xv[m][4] * w8[4] + xv[m][5] * w8[5] + xv[m][6] * w8[6] + xv[m][7] * w8[7]);
                    const float full = accum * 16384.0f;
                    const float tt = s * full;
                    res[m][q] = res[m][q] + tt;
                }
            } else {
                const uint w0 = wexp[n * 576 + 64 * b + 2 * lane];
                const uint w1 = wexp[n * 576 + 64 * b + 2 * lane + 1];
                const float s = l3_e8m0(sexp[n * 72 + 8 * b + lane / 4]);
                float wv[8];
                l3_e4m3x4(w0, wv);
                l3_e4m3x4(w1, wv + 4);
                for (int m = 0; m < NM; ++m) {
                    float accum = 0;
                    for (int i = 0; i < 8; ++i) accum += xv[m][i] * wv[i];
                    const float full = accum * 256.0f;
                    const float tt = s * full;
                    res[m][q] = res[m][q] + tt;
                }
            }
        }
    }
    for (int m = 0; m < NM; ++m)
        for (int q = 0; q < R; ++q) out[m][q] = simd_sum(res[m][q]);
}

// Down dot products of NM member inputs (lane-major in threadgroup memory) with R w2 rows.
template <int NM, int R, bool NV4>
inline void l3_down_dots(const device uint* wexp, const device uchar* sexp, threadgroup const float* xs,
                         uint nb, uint lane, thread float (&out)[4][R]) {
    float res[NM][R];
    for (int m = 0; m < NM; ++m)
        for (int q = 0; q < R; ++q) res[m][q] = 0.0f;
    if (NV4) {
        for (uint b = 0; b < 5; ++b) {
            if (b == 4 && lane >= 16) break;
            float xv[NM][8];
            for (int m = 0; m < NM; ++m)
                for (int v = 0; v < 8; ++v) xv[m][v] = xs[m * 1280 + (b * 8 + v) * 32 + lane];
            for (int q = 0; q < R; ++q) {
                const uint n = nb + q;
                const uint word = wexp[n * 144 + 32 * b + lane];
                const float s = l3_e4m3(sexp[n * 72 + 16 * b + lane / 2]);
                float w8[8];
                l3_e2m1x8(word, w8);
                for (int m = 0; m < NM; ++m) {
                    float accum = 0;
                    accum += (xv[m][0] * w8[0] + xv[m][1] * w8[1] + xv[m][2] * w8[2] + xv[m][3] * w8[3]);
                    accum += (xv[m][4] * w8[4] + xv[m][5] * w8[5] + xv[m][6] * w8[6] + xv[m][7] * w8[7]);
                    const float full = accum * 16384.0f;
                    const float tt = s * full;
                    res[m][q] = res[m][q] + tt;
                }
            }
        }
    } else {
        for (uint b = 0; b < 9; ++b) {
            float xv[NM][4];
            for (int m = 0; m < NM; ++m)
                for (int v = 0; v < 4; ++v) xv[m][v] = xs[m * 1280 + (b * 4 + v) * 32 + lane];
            for (int q = 0; q < R; ++q) {
                const uint n = nb + q;
                const uint word = wexp[n * 288 + 32 * b + lane];
                const float s = l3_e8m0(sexp[n * 36 + 4 * b + lane / 8]);
                float wv[4];
                l3_e4m3x4(word, wv);
                for (int m = 0; m < NM; ++m) {
                    float accum = 0;
                    for (int i = 0; i < 4; ++i) accum += xv[m][i] * wv[i];
                    const float full = accum * 256.0f;
                    const float tt = s * full;
                    res[m][q] = res[m][q] + tt;
                }
            }
        }
    }
    for (int m = 0; m < NM; ++m)
        for (int q = 0; q < R; ++q) out[m][q] = simd_sum(res[m][q]);
}

// RC6's fused route (expert_fusion ROUTE, fused variant) for one row on one simdgroup.
// Lanes 0..K-1 return the k-th chosen expert and its scaled probability.
inline void l3_route_row(const device float* z, const device float* bias, float scale,
                         float tiny, uint lane, thread uint& e_out, thread float& p_out) {
    float s[4], key[4];
    bool taken[4];
    for (int i = 0; i < 4; ++i) {
        uint e = i * 32 + lane;
        float zz = z[e];
        float y = 1 / (1 + metal::precise::exp(metal::abs(zz)));
        s[i] = (zz < 0) ? y : 1 - y;
        float sb = s[i] + float(bias[e]);
        key[i] = -sb;
        taken[i] = false;
    }
    float chosen = 0, total = 0;
    uint expert = 0;
    for (int k = 0; k < 8; ++k) {
        float best = NAN;
        uint at = 0xffffffffu;
        for (int i = 0; i < 4; ++i)
            if (!taken[i] && fused_before(key[i], i * 32 + lane, best, at)) {
                best = key[i];
                at = i * 32 + lane;
            }
        for (uint d = 16; d > 0; d /= 2) {
            float ob = simd_shuffle_xor(best, d);
            uint oa = simd_shuffle_xor(at, d);
            if (fused_before(ob, oa, best, at)) {
                best = ob;
                at = oa;
            }
        }
        uint slot = at / 32;
        float p = 0;
        for (int i = 0; i < 4; ++i) {
            if (i == int(slot)) p = s[i];
            if (i == int(slot) && lane == at % 32) taken[i] = true;
        }
        p = simd_shuffle(p, at % 32);
        total = p + total;
        if (lane == uint(k)) {
            chosen = p;
            expert = at;
        }
    }
    float denominator = total + tiny;
    float share = chosen / denominator;
    float scaled = share * scale;
    e_out = expert;
    p_out = scaled;
}
"""

# One shared plan: the panel's distinct experts in ascending id, the slot's expert and its
# member rows in ascending row order. Every threadgroup computes it from the same data.
_PLAN = """
    threadgroup uint pl_used[128];
    threadgroup uint pl_info[4];
    threadgroup uint pl_row[W];
    threadgroup uint pl_pair[W];
    // One writer per expert flag (review 2, F5, as in ROUTE_PLAN): thread e scans the W x 8 choices.
    for (uint e = tid; e < 128u; e += NT) {
        uint f = 0u;
        for (uint i = 0; i < uint(W) * 8u; ++i) f |= PAIR_EXPERT(i) == e ? 1u : 0u;
        pl_used[e] = f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint c = pl_used[4 * lane] + pl_used[4 * lane + 1] + pl_used[4 * lane + 2] + pl_used[4 * lane + 3];
        uint base = simd_prefix_exclusive_sum(c);
        uint total = simd_sum(c);
        if (base <= slot && slot < base + c) {
            uint want = slot - base;
            uint found = 0;
            for (uint q = 0; q < 4; ++q) {
                if (pl_used[4 * lane + q] != 0u) {
                    if (want == 0u) found = 4 * lane + q;
                    want -= 1u;
                }
            }
            pl_info[0] = found;
        }
        if (lane == 0) pl_info[1] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slot >= pl_info[1]) return;
    const uint e = pl_info[0];
    if (sg == 0) {
        uint k_hit = 8u;
        if (lane < uint(W)) {
            for (uint k = 0; k < 8u; ++k) if (PAIR_EXPERT(lane * 8u + k) == e) k_hit = k;
        }
        uint has = k_hit < 8u ? 1u : 0u;
        uint at = simd_prefix_exclusive_sum(has);
        if (has != 0u) {
            pl_row[at] = lane;
            pl_pair[at] = lane * 8u + k_hit;
        }
        uint members = simd_sum(has);
        if (lane == 0) pl_info[2] = members;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint M = pl_info[2];
"""

INPUTS = """
    const uint j = threadgroup_position_in_grid.x;
    const uint t = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup float normed[2304];
    threadgroup float sums[32];
    threadgroup float red[64];
    threadgroup float scratch[32];
    threadgroup float inv_mean[1];
    const float eps = consts[0];
    // This threadgroup's four router rows. A thread owns four columns in each 1,024-wide span of
    // the 2,304; the last span holds 256, so only the first 64 threads reach into it.
    const int col = int(32 * sg + lane) * 4;
    const bool in_last = 2048 + col + 4 <= 2304;
    float wt[3][4][4];
    for (int e = 0; e < 4; ++e) {
        const device float* wrow = router + (4 * j + e) * 2304;
        for (int c = 0; c < 4; ++c) {
            wt[0][e][c] = wrow[col + c];
            wt[1][e][c] = wrow[1024 + col + c];
            wt[2][e][c] = in_last ? wrow[2048 + col + c] : 0.0f;
        }
    }
    if (t < 32) sums[t] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        const uint r = threadgroup_position_in_grid.y;
        const device float* xr = source + r * 2304;
        // The row's sum of squares in the reference norm's order: 576 slices of four values, each
        // summed in index order from zero, 18 sums of 32 slices, then one sum of those (zeros past
        // the 18). This threadgroup has 8 simdgroups, so each takes the partials 8 apart.
        for (uint part = sg; part < 18; part += 8) {
            const device float* four = xr + 4 * (32 * part + lane);
            float sq = 0;
            for (int i = 0; i < 4; i++) {
                const float v = four[i];
                sq += v * v;
            }
            sq = simd_sum(sq);
            if (lane == 0) sums[part] = sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            const float row_sq = simd_sum(sums[lane]);
            if (lane == 0) inv_mean[0] = metal::precise::rsqrt(row_sq / 2304.0f + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv = inv_mean[0];
        float local = 0;
        bool nan_seen = false;
        for (uint i = t; i < 2304; i += 256) {
            const float xi = xr[i];
            const float wi = float(norm_w[i]);
            const float y = wi * static_cast<float>(xi * inv);
            const float v = float(static_cast<T>(y));     // the stream's 16-bit type, as RC6 casts
            normed[i] = v;
            const float a = abs(v);
            nan_seen = nan_seen || isnan(a);
            local = max(local, a);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Router logits, bit for bit the one-row product's: a thread's sum starts at zero and takes
        // its three spans in order, one product a statement; the 32 lanes meet by halving (16, 8, 4,
        // 2, 1); the eight simdgroups' sums are added in order onto the first.
        float xs[3][4];
        for (int c = 0; c < 4; ++c) {
            xs[0][c] = normed[col + c];
            xs[1][c] = normed[1024 + col + c];
            xs[2][c] = in_last ? normed[2048 + col + c] : 0.0f;
        }
        float dot[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int span = 0; span < 3; ++span)
            for (int e = 0; e < 4; ++e)
                for (int c = 0; c < 4; ++c)
                    dot[e] += wt[span][e][c] * xs[span][c];
        for (int e = 0; e < 4; ++e)
            for (ushort hop = 16; hop != 0; hop /= 2)
                dot[e] += simd_shuffle_down(dot[e], hop);
        if (lane == 0)
            for (int e = 0; e < 4; ++e) red[sg * 8 + e] = dot[e];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            for (int e = 0; e < 4; ++e) {
                float total = dot[e];
                for (int s = 1; s < 8; ++s) total += red[s * 8 + e];
                logits[r * 128 + 4 * j + e] = total;
            }
        }
        // Expert input rounding: RC6's per-lane quantiser on 32 consecutive values a simdgroup.
#if FP4
        const float peak = fused_row_peak(local, nan_seen, scratch, sg, lane, 8);
        if (j == 0 && t == 0) outer[r] = peak / 1536.0f;
#else
        const float peak = 0.0f;
#endif
        if (sg < 3) {
            const uint g = j + 32 * sg;
            if (g < 72) {
                const uint i = 32 * g + lane;
                values[r * 2304 + i] = fused_quantize<bool(FP4)>(normed[i], peak, lane);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

GATE_UP = """
    const uint tile = threadgroup_position_in_grid.x;
    const uint slot = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup uint pe[W * 8];
    threadgroup float pp[W * 8];
    for (uint r = sg; r < uint(W); r += S) {
        uint ex; float pr;
        l3_route_row(logits + r * 128, bias, rconsts[0], rconsts[1], lane, ex, pr);
        if (lane < 8) {
            pe[r * 8 + lane] = ex;
            pp[r * 8 + lane] = pr;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#define PAIR_EXPERT(i) pe[(i)]
""" + _PLAN + """
#undef PAIR_EXPERT
    if (tile == 0 && slot == 0) {
        for (uint i = tid; i < uint(W) * 8u; i += NT) {
            experts_out[i] = pe[i];
            prob_out[i] = pp[i];
        }
    }
    const device uint* wexp = codes + size_t(e) * 2304 * CW;
    const device uchar* sexp = scales + size_t(e) * 2304 * SW;
#if FP4
    const float eo = expert_outer[e];
#endif
    threadgroup float tpk[S * 32];
    threadgroup uint tnan[S * 32];
    for (uint i = tid; i < S * 32u; i += NT) { tpk[i] = 0.0f; tnan[i] = 0u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr uint NB = TILE / R;
    for (uint it = sg; it < M * NB; it += S) {
        const uint m = it % M;
        const uint nb = tile * TILE + (it / M) * R;
        float res[4][R];
        for (uint q = 0; q < R; ++q) res[0][q] = 0.0f;
        l3_gate_up_dots<1, R, bool(FP4)>(wexp, sexp, values, pl_row, m, nb, lane, res);
        // Capped SwiGLU (RC6 SWIGLU statements); lane jj takes mid nb/2 + jj of member m.
        float graw = 0, uraw = 0;
        for (uint jj = 0; jj < R / 2; ++jj)
            if (lane == jj) { graw = res[0][2 * jj]; uraw = res[0][2 * jj + 1]; }
        const bool active = lane < R / 2;
        float hv = 0.0f;
        if (active) {
            const uint pair = pl_pair[m];
#if FP4
            const float scale = row_outer[pl_row[m]] * eo;
            const float gs = graw * scale;
            const float us = uraw * scale;
            const T gq = static_cast<T>(gs);
            const T uq = static_cast<T>(us);
#else
            const T gq = static_cast<T>(graw);
            const T uq = static_cast<T>(uraw);
#endif
            const float gate = float(gq);
            const float up = float(uq);
            const float gcap = static_cast<float>(GATE_CAP);
            const float ucap = static_cast<float>(UP_CAP);
            const float t1 = gate / gcap;
            const float t2 = metal::precise::tanh(t1);
            const float t3 = gcap * t2;
            const float y = 1 / (1 + metal::exp(metal::abs(gate)));
            const float t4 = (gate < 0) ? y : 1 - y;
            const float t5 = t3 * t4;
            const float t6 = up / ucap;
            const float t7 = metal::precise::tanh(t6);
            const float t8 = ucap * t7;
            const float t9 = t5 * t8;
            const T mid = static_cast<T>(t9);
            h[pair * 1152 + nb / 2 + lane] = mid;
            hv = float(mid);
        }
        const float a = active ? abs(hv) : 0.0f;
        const float top = simd_max(a);
        const bool bad = simd_any(active && isnan(a));
        if (lane == 0) {
            tpk[sg * 32 + m] = max(tpk[sg * 32 + m], top);
            tnan[sg * 32 + m] = tnan[sg * 32 + m] | (bad ? 1u : 0u);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint m = tid; m < M; m += NT) {
        float best = 0.0f;
        bool any = false;
        for (uint s2 = 0; s2 < S; ++s2) {
            best = max(best, tpk[s2 * 32 + m]);
            any = any || tnan[s2 * 32 + m] != 0u;
        }
        peaks[pl_pair[m] * NTILE + tile] = any ? NAN : best;
    }
"""

DOWN = """
    const uint tile = threadgroup_position_in_grid.x;
    const uint slot = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
#define PAIR_EXPERT(i) experts_in[(i)]
""" + _PLAN + """
#undef PAIR_EXPERT
    const device uint* wexp = codes + size_t(e) * 2304 * CW;
    const device uchar* sexp = scales + size_t(e) * 2304 * SW;
    threadgroup float xs[MC * 1280];
    threadgroup float tpeak[MC];
    for (uint m0 = 0; m0 < M; m0 += MC) {
        const uint mc = min(uint(MC), M - m0);
        if (tid < mc) {
            const uint pair = pl_pair[m0 + tid];
            float best = 0.0f;
            bool any = false;
            for (uint i = 0; i < uint(NTILE13); ++i) {
                const float v = peaks[pair * NTILE13 + i];
                any = any || isnan(v);
                best = max(best, v);
            }
            const float peak = any ? NAN : best;
            tpeak[tid] = peak;
#if FP4
            if (tile == 0) mid_outer[pair] = peak / 1536.0f;
#endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // The down input of each member, rounded as RC6 rounds it, laid out lane-major.
        for (uint m = 0; m < mc; ++m) {
            const uint pair = pl_pair[m0 + m];
#if FP4
            const float peak = tpeak[m];
#else
            const float peak = 0.0f;
#endif
            for (uint g = sg; g < 36; g += S) {
                const uint i = 32 * g + lane;
                const float hv = float(h[pair * 1152 + i]);
                const float qv = fused_quantize<bool(FP4)>(hv, peak, lane);
#if FP4
                const uint b = i / 256, l = (i % 256) / 8, v = i % 8;
                xs[m * 1280 + (b * 8 + v) * 32 + l] = qv;
#else
                const uint b = i / 128, l = (i % 128) / 4, v = i % 4;
                xs[m * 1280 + (b * 4 + v) * 32 + l] = qv;
#endif
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        constexpr uint NB = TILE / R;
        for (uint it = sg; it < mc * NB; it += S) {
            const uint ml = it % mc;
            const uint nb = tile * TILE + (it / mc) * R;
            float res[4][R];
            for (uint q = 0; q < R; ++q) res[0][q] = 0.0f;
            l3_down_dots<1, R, bool(FP4)>(wexp, sexp, xs + ml * 1280, nb, lane, res);
            for (uint q = 0; q < R; ++q)
                if (lane == q) raw[pl_pair[m0 + ml] * 2304 + nb + q] = res[0][q];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

COMBINE = """
    const uint c = thread_position_in_grid.x;
    if (c >= 2304) return;
    float h0 = tail[0 * tail_strides[0] + c * tail_strides[1]];
    float h1 = tail[1 * tail_strides[0] + c * tail_strides[1]];
    float h2 = tail[2 * tail_strides[0] + c * tail_strides[1]];
    const float w0 = conv_w[0 * conv_w_strides[0] + c * conv_w_strides[1]];
    const float w1 = conv_w[1 * conv_w_strides[0] + c * conv_w_strides[1]];
    const float w2 = conv_w[2 * conv_w_strides[0] + c * conv_w_strides[1]];
    const float w3 = conv_w[3 * conv_w_strides[0] + c * conv_w_strides[1]];
    for (uint r = 0; r < uint(W); ++r) {
        float acc = 0;
        for (int k = 0; k < 8; ++k) {
            const uint p = r * 8 + k;
            const float v = raw[p * 2304 + c];
#if FP4
            const float scale = mid_outer[p] * expert_outer[experts[p]];
            const float vs = v * scale;
            const T q = static_cast<T>(vs);
#else
            const T q = static_cast<T>(v);
#endif
            const float w = float(q) * prob[p];
            const float u = w + 0.0f;
            acc = u + acc;
        }
        const T routed = static_cast<T>(acc);
        const float x3 = float(routed);
        rows[r * 2304 + c] = x3;
        float cacc = 0.0f;
#if CONV_FORM == 0
        cacc += h0 * w0;
        cacc += h1 * w1;
        cacc += h2 * w2;
        cacc += x3 * w3;
#else
        cacc = metal::fma(h0, w0, cacc);
        cacc = metal::fma(h1, w1, cacc);
        cacc = metal::fma(h2, w2, cacc);
        cacc = metal::fma(x3, w3, cacc);
#endif
        const float total = x3 + cacc;
        const float value = total / as_type<float>(uint(DIVISOR));
        const T conv = static_cast<T>(value);
        const float sum = partial[r * partial_strides[0] + c * partial_strides[1]] + float(conv);
        out[r * 2304 + c] = static_cast<T>(sum);
        h0 = h1;
        h1 = h2;
        h2 = x3;
    }
"""


# Panels of 2 to 8 rows: each distinct expert of the panel read once. A route launch writes
# RC6's routing and a plan (distinct experts ascending; each one's member rows and pairs in row
# order); each product threadgroup takes one (distinct expert, tile of output rows) and computes
# every member from the same decoded weights. Per (pair, output) the arithmetic is MLX 0.32.2's
# gathered product exactly (`fp_qmv_impl` / `fp_qmv_fast_impl` lane slices, statement forms,
# simd_sum), so each pair's raw product equals the gathered product's bit for bit; codes are
# decoded straight into half bit patterns (two codes a half2), the values the byte tables hold.
PLAN_STRIDE = 18   # per slot: expert, members, member rows[8], member pairs[8]
PANEL_ROWS = 8     # widest panel the plan describes (an expert has at most one member a row)

ROUTE_PLAN = """
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup uint pe[W * 8];
    threadgroup uint used[128];
    threadgroup uint slot_of[128];
    for (uint r = sg; r < uint(W); r += 8u) {
        uint ex;
        float pr;
        l3_route_row(logits + r * 128, bias, rconsts[0], rconsts[1], lane, ex, pr);
        if (lane < 8u) {
            pe[r * 8 + lane] = ex;
            experts[r * 8 + lane] = ex;
            prob[r * 8 + lane] = pr;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // One writer per expert flag (thread e scans the panel's W x 8 choices for e), so no
    // two threads store to one flag, even when every row picks the same experts.
    for (uint e = tid; e < 128u; e += 256u) {
        uint f = 0u;
        for (uint i = 0; i < uint(W) * 8u; ++i) f |= pe[i] == e ? 1u : 0u;
        used[e] = f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint c[4];
        uint mine = 0;
        for (int q = 0; q < 4; ++q) {
            c[q] = used[4 * lane + q];
            mine += c[q];
        }
        uint at = simd_prefix_exclusive_sum(mine);
        const uint total = simd_sum(mine);
        for (int q = 0; q < 4; ++q) {
            if (c[q] != 0u) {
                slot_of[4 * lane + q] = at;
                plan[1 + at * PLAN_STRIDE] = 4 * lane + q;
                at += 1u;
            }
        }
        if (lane == 0) plan[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < uint(W) * 8u; p += 256u) {
        const uint e = pe[p];
        uint before = 0, after = 0;
        for (uint q = 0; q < uint(W) * 8u; ++q) {
            const uint same = pe[q] == e ? 1u : 0u;
            before += q < p ? same : 0u;
            after += q > p ? same : 0u;
        }
        const uint base = 1 + slot_of[e] * PLAN_STRIDE;
        plan[base + 2 + before] = p / 8u;
        plan[base + 10 + before] = p;
        if (after == 0u) plan[base + 1] = before + 1u;
    }
"""

PANEL = """
    const uint tile = threadgroup_position_in_grid.x;
    const uint slot = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    if (slot >= plan[0]) return;
    const device uint* ps = plan + 1 + slot * PLAN_STRIDE;
    const uint e = ps[0];
    const uint M = ps[1];
    const device uint* wexp = codes + size_t(e) * 2304 * CW;
    const device uchar* sexp = scales + size_t(e) * 2304 * SW;
    for (uint st = 0; st < RPS / R; ++st) {
        const uint n0 = tile * (S * RPS) + sg * RPS + st * R;
        for (uint m0 = 0; m0 < M; m0 += MC) {
            const uint mc = min(uint(MC), M - m0);
            float res[MC][R];
            for (int m = 0; m < MC; ++m)
                for (int q = 0; q < R; ++q) res[m][q] = 0.0f;
            for (uint b = 0; b < NBLK; ++b) {
#if HALF_LAST
                if (b == NBLK - 1 && lane >= 16) break;
#endif
                float wv[R][NV];
                float sv[R];
                for (int q = 0; q < R; ++q) {
                    const uint n = n0 + q;
                    const uint sb = sexp[n * SW + (32 / SPL) * b + lane / SPL];
#if FP4
                    sv[q] = l3_e4m3(sb) * SFOLD;
                    l3_e2m1x8(wexp[n * CW + 32 * b + lane], wv[q]);
#else
                    sv[q] = l3_e8m0(sb) * SFOLD;
                    for (int j = 0; j < WPL; ++j)
                        l3_e4m3x4(wexp[n * CW + (32 * WPL) * b + WPL * lane + j], wv[q] + 4 * j);
#endif
                }
                for (uint m = 0; m < uint(MC); ++m) {
                    if (m < mc) {
                        const device float* xr = xin + ps[XSEL + m0 + m] * K + (32 * NV) * b + NV * lane;
                        float xv[NV];
                        const float4 a = *(const device float4*)xr;
                        xv[0] = a.x; xv[1] = a.y; xv[2] = a.z; xv[3] = a.w;
#if NV == 8
                        const float4 c = *(const device float4*)(xr + 4);
                        xv[4] = c.x; xv[5] = c.y; xv[6] = c.z; xv[7] = c.w;
#endif
                        for (int q = 0; q < R; ++q) {
                            float accum = 0;
#if FP4
                            accum += (xv[0] * wv[q][0] + xv[1] * wv[q][1] + xv[2] * wv[q][2] + xv[3] * wv[q][3]);
                            accum += (xv[4] * wv[q][4] + xv[5] * wv[q][5] + xv[6] * wv[q][6] + xv[7] * wv[q][7]);
#else
                            for (int i = 0; i < NV; ++i) accum += xv[i] * wv[q][i];
#endif
#if FOLD
                            const float tt = sv[q] * accum;
#else
                            const float full = accum * POW;
                            const float tt = sv[q] * full;
#endif
                            res[m][q] = res[m][q] + tt;
                        }
                    }
                }
            }
            for (uint m = 0; m < uint(MC); ++m) {
                if (m < mc) {
                    const uint pair = ps[10 + m0 + m];
                    for (int q = 0; q < R; ++q) {
                        const float v = simd_sum(res[m][q]);
                        if (lane == uint(q)) raw[pair * 2304 + n0 + q] = v;
                    }
                }
            }
        }
    }
"""

# Prompt chunks: the same products for pairs sorted by expert (the stock body's sort). The
# threadgroup of expert e finds its members, a contiguous run of the sorted pairs, by binary
# search, and writes each member's raw product at its sorted position.
PANEL_PROMPT = (PANEL
                .replace("""    if (slot >= plan[0]) return;
    const device uint* ps = plan + 1 + slot * PLAN_STRIDE;
    const uint e = ps[0];
    const uint M = ps[1];""", """    const uint e = slot;
    uint lo = 0;
    for (uint a = 0, z = PAIRS; a < z;) {
        const uint mid = (a + z) / 2;
        if (sorted_experts[mid] < e) a = mid + 1; else z = mid;
        lo = a;
    }
    uint hi = lo;
    for (uint a = lo, z = PAIRS; a < z;) {
        const uint mid = (a + z) / 2;
        if (sorted_experts[mid] <= e) a = mid + 1; else z = mid;
        hi = a;
    }
    if (hi == lo) return;
    const uint M = hi - lo;""")
                .replace("ps[XSEL + m0 + m] * K", "sorted_rows[lo + m0 + m] * K")
                .replace("const uint pair = ps[10 + m0 + m];", "const uint pair = lo + m0 + m;"))
assert PANEL_PROMPT.count("sorted_rows[lo + m0 + m]") == 1 and "ps[" not in PANEL_PROMPT

# Prompt chunks name about 32 pairs an expert. Here the loop runs blocks outside rows: each
# block's inputs of a chunk of MC members sit in registers while every row of the simdgroup is
# decoded once and applied to all of them, so an input slice is loaded once for RPS rows.
PANEL_PROMPT2 = """
    const uint tile = threadgroup_position_in_grid.x;
    const uint e = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    uint lo = 0;
    for (uint a = 0, z = PAIRS; a < z;) {
        const uint mid = (a + z) / 2;
        if (sorted_experts[mid] < e) a = mid + 1; else z = mid;
        lo = a;
    }
    uint hi = lo;
    for (uint a = lo, z = PAIRS; a < z;) {
        const uint mid = (a + z) / 2;
        if (sorted_experts[mid] <= e) a = mid + 1; else z = mid;
        hi = a;
    }
    if (hi == lo) return;
    const uint M = hi - lo;
    const device uint* wexp = codes + size_t(e) * 2304 * CW;
    const device uchar* sexp = scales + size_t(e) * 2304 * SW;
    const uint n0 = tile * (S * RPS) + sg * RPS;
    for (uint m0 = 0; m0 < M; m0 += MC) {
        const uint mc = min(uint(MC), M - m0);
        float res[MC][RPS];
        for (int m = 0; m < MC; ++m)
            for (int q = 0; q < RPS; ++q) res[m][q] = 0.0f;
        for (uint b = 0; b < NBLK; ++b) {
            float xv[MC][NV];
            for (uint m = 0; m < uint(MC); ++m) {
                if (m < mc) {
                    const device float* xr = xin + sorted_rows[lo + m0 + m] * K + (32 * NV) * b + NV * lane;
                    const float4 a = *(const device float4*)xr;
                    xv[m][0] = a.x; xv[m][1] = a.y; xv[m][2] = a.z; xv[m][3] = a.w;
#if NV == 8
                    const float4 c = *(const device float4*)(xr + 4);
                    xv[m][4] = c.x; xv[m][5] = c.y; xv[m][6] = c.z; xv[m][7] = c.w;
#endif
                }
            }
            for (int q = 0; q < RPS; ++q) {
                const uint n = n0 + q;
                const uint sb = sexp[n * SW + (32 / SPL) * b + lane / SPL];
                float wv[NV];
#if FP4
                const float sv = l3_e4m3(sb) * SFOLD;
                l3_e2m1x8(wexp[n * CW + 32 * b + lane], wv);
#else
                const float sv = l3_e8m0(sb) * SFOLD;
                for (int j = 0; j < WPL; ++j)
                    l3_e4m3x4(wexp[n * CW + (32 * WPL) * b + WPL * lane + j], wv + 4 * j);
#endif
                for (uint m = 0; m < uint(MC); ++m) {
                    if (m < mc) {
                        float accum = 0;
#if FP4
                        accum += (xv[m][0] * wv[0] + xv[m][1] * wv[1] + xv[m][2] * wv[2] + xv[m][3] * wv[3]);
                        accum += (xv[m][4] * wv[4] + xv[m][5] * wv[5] + xv[m][6] * wv[6] + xv[m][7] * wv[7]);
#else
                        for (int i = 0; i < NV; ++i) accum += xv[m][i] * wv[i];
#endif
#if FOLD
                        const float tt = sv * accum;
#else
                        const float full = accum * POW;
                        const float tt = sv * full;
#endif
                        res[m][q] = res[m][q] + tt;
                    }
                }
            }
        }
        for (uint m = 0; m < uint(MC); ++m) {
            if (m < mc) {
                const uint pair = lo + m0 + m;
                for (int q = 0; q < RPS; ++q) {
                    const float v = simd_sum(res[m][q]);
                    if (lane == uint(q)) raw[pair * 2304 + n0 + q] = v;
                }
            }
        }
    }
"""

# Per (product, fp4): K, words a row, scale bytes a row, values a lane, words a lane, blocks,
# lanes per scale byte, and the power of two the decoded codes leave out (MLX's dispatch:
# gate-up NVFP4 and both MXFP8 down use `fp_qmv_impl`, gate-up MXFP8 `fp_qmv_fast_impl`).
PANEL_GEOMETRY = {
    ("gu", True): dict(K=2304, CW=288, SW=144, NV=8, WPL=1, NBLK=9, SPL=2, POW="16384.0f"),
    ("gu", False): dict(K=2304, CW=576, SW=72, NV=8, WPL=2, NBLK=9, SPL=4, POW="256.0f"),
    ("dn", True): dict(K=1152, CW=144, SW=72, NV=8, WPL=1, NBLK=5, SPL=2, POW="16384.0f"),
    ("dn", False): dict(K=1152, CW=288, SW=36, NV=4, WPL=1, NBLK=9, SPL=8, POW="256.0f"),
}

def _header(extra=""):
    return expert_fusion.HEADER + "constexpr constant int THREADS = 384;\n" + _HELPERS + extra


@lru_cache(maxsize=None)
def _divisor_bits(size=4):
    return struct.unpack("<I", struct.pack("<f", math.sqrt(size + 1)))[0]


@lru_cache(maxsize=None)
def _kernel(stage, fp4, defines=()):
    """One kernel object per (stage, format, compile-time switches)."""
    fp4_only = lambda *names: list(names) if fp4 else []
    inputs, outputs, body, contiguous = {
        "inputs": (["source", "norm_w", "router", "consts"], ["logits", "values", *fp4_only("outer")], INPUTS, True),
        "gate_up": (
            ["logits", "bias", "rconsts", "values", *fp4_only("row_outer"), "codes", "scales", *fp4_only("expert_outer")],
            ["h", "peaks", "experts_out", "prob_out"], GATE_UP, True,
        ),
        "down": (
            ["experts_in", "h", "peaks", "codes", "scales"],
            ["raw", *fp4_only("mid_outer")], DOWN, True,
        ),
        "combine": (
            ["raw", *fp4_only("mid_outer"), "experts", "prob", *fp4_only("expert_outer"), "conv_w", "tail", "partial"],
            ["out", "rows"], COMBINE, False,
        ),
    }[stage]
    extra = f"#define FP4 {int(fp4)}\n" + "".join(f"#define {k} {v}\n" for k, v in defines)
    tag = "_".join(f"{k.lower()}{v}" for k, v in defines)
    name = f"mtplx_inkling_l3_{stage}_{'nvfp4' if fp4 else 'mxfp8'}" + (f"_{tag}" if tag else "")
    return mx.fast.metal_kernel(
        name=name, input_names=inputs, output_names=outputs, source=body,
        header=_header(extra), ensure_row_contiguous=contiguous,
    )


@lru_cache(maxsize=None)
def _caps(gate_cap, up_cap):
    """The SwiGLU caps as MLX's compiler writes them, computed once per value (one host read each)."""
    return (("GATE_CAP", expert_fusion.mlx_literal(gate_cap)), ("UP_CAP", expert_fusion.mlx_literal(up_cap)))


@lru_cache(maxsize=None)
def _consts(eps):
    return mx.array([eps], mx.float32)


@lru_cache(maxsize=None)
def _route_consts(scale):
    return mx.array([scale, 1e-20], mx.float32)


#: The stream types the block serves: bf16, and float16 on the FP16 packs (M1 and M2 Macs). Its kernels
#: round to the stream's type wherever RC6 casts to it (`.astype(x.dtype)`), in no other place.
STREAM_TYPES = (mx.bfloat16, mx.float16)


def unserved(block, rows, dtype=mx.bfloat16):
    """Why the fast path cannot serve this visit, or None. `dtype` is the visit's stream type."""
    if os.environ.get("MTPLX_INKLING_FAST_EXPERTS", "on").lower() in {"0", "off", "stock"}:
        return "disabled"
    if dtype not in STREAM_TYPES:
        return "dtype"
    if not (mx.default_device() == mx.gpu and mx.metal.is_available()):
        return "cpu"
    if not 1 <= rows <= MAX_ROWS:
        return "rows"
    gu, dn = block.gate_up, block.down
    if gu.mode not in {"nvfp4", "mxfp8"} or gu.mode != dn.mode:
        return "format"
    if gu.affine is not None or dn.affine is not None:
        return "affine_experts"
    if block.router.shape != (EXPERTS, HIDDEN) or block.router.dtype != mx.float32 or block.count != TOP_K:
        return "geometry"
    fp4 = gu.mode == "nvfp4"
    cw13, cw2 = (HIDDEN // 8, MIDDLE // 8) if fp4 else (HIDDEN // 4, MIDDLE // 4)
    if gu.codes.shape != (EXPERTS, 2 * MIDDLE, cw13) or dn.codes.shape != (EXPERTS, HIDDEN, cw2):
        return "geometry"
    return None


def _member_chunk(rows):
    return 1 if rows == 1 else 2 if rows <= 2 else 4


DEFAULT_TILES = {"gu_tile": TILE_GATE_UP, "gu_s": 8, "gu_r": 8, "dn_tile": TILE_DOWN, "dn_s": 8, "dn_r": 8}


def _mlx_products(block, fp4, rows, logits, values, outer, dtype=mx.bfloat16):
    """RC6's route kernel, MLX's gathered quantised products and RC6's SwiGLU kernel (4 launches)."""
    pairs = rows * block.count
    bits, group = (4, 16) if fp4 else (8, 32)
    experts, row_ids, prob = expert_fusion.route(logits, block.bias, block.count, block.scale, "fused")
    raw13 = mx.gather_qmm(values.reshape(rows, 1, HIDDEN), block.gate_up.codes, scales=block.gate_up.scales,
                          lhs_indices=row_ids, rhs_indices=experts, group_size=group, bits=bits,
                          mode=block.gate_up.mode)
    mid, *mid_outer = expert_fusion._kernel("swiglu", fp4, _caps(float(block.gate_cap), float(block.up_cap)))(
        inputs=[raw13.reshape(pairs, 2 * MIDDLE), *([outer, block.gate_up.outer] if fp4 else []), experts, row_ids],
        template=[("T", dtype), ("W", MIDDLE), ("E", MIDDLE // expert_fusion.THREADS)],
        grid=(pairs * expert_fusion.THREADS, 1, 1), threadgroup=(expert_fusion.THREADS, 1, 1),
        output_shapes=[(pairs, 1, MIDDLE), *([(pairs,)] if fp4 else [])],
        output_dtypes=[mx.float32] * (2 if fp4 else 1),
    )
    raw2 = mx.gather_qmm(mid, block.down.codes, scales=block.down.scales, lhs_indices=expert_fusion._pair_ids(pairs),
                         rhs_indices=experts, group_size=group, bits=bits, mode=block.down.mode)
    return raw2.reshape(pairs, HIDDEN), (mid_outer[0] if fp4 else None), experts, prob


# Threadgroup layout of the panel products per (product, fp4): S simdgroups, RPS output rows a
# simdgroup, R rows a step, MC members a pass, FOLD the scale times the codes' left-out power of
# two (one multiply fewer a product; the same real product, rounded once). Measured on real
# routing, cold weights (lanes/l2/xbench-*.json).
PANEL_LAYOUT = {
    ("gu", True): dict(S=4, RPS=2, R=2, MC=4, FOLD=1),
    ("dn", True): dict(S=4, RPS=2, R=2, MC=4, FOLD=1),
    ("gu", False): dict(S=4, RPS=2, R=2, MC=4, FOLD=1),
    ("dn", False): dict(S=4, RPS=2, R=2, MC=4, FOLD=1),
}


@lru_cache(maxsize=None)
def _panel_kernel(product, fp4, fold):
    g = PANEL_GEOMETRY[(product, fp4)]
    defines = {"FP4": int(fp4), "NV": g["NV"], "WPL": g["WPL"], "NBLK": g["NBLK"],
               "HALF_LAST": int(product == "dn" and fp4), "CW": g["CW"], "SW": g["SW"], "SPL": g["SPL"],
               "K": g["K"], "POW": g["POW"], "XSEL": 2 if product == "gu" else 10, "PLAN_STRIDE": PLAN_STRIDE,
               "FOLD": int(fold), "SFOLD": (g["POW"] if fold else "1.0f")}
    extra = "".join(f"#define {k} {v}\n" for k, v in defines.items())
    name = f"mtplx_inkling_panel_{product}_{'nvfp4' if fp4 else 'mxfp8'}_f{int(fold)}"
    return mx.fast.metal_kernel(name=name, input_names=["plan", "xin", "codes", "scales"], output_names=["raw"],
                                source=PANEL, header=_header(extra), ensure_row_contiguous=True)


@lru_cache(maxsize=None)
def _panel_prompt_kernel(fp4, fold, body=1):
    g = PANEL_GEOMETRY[("gu", fp4)]
    defines = {"FP4": int(fp4), "NV": g["NV"], "WPL": g["WPL"], "NBLK": g["NBLK"], "HALF_LAST": 0,
               "CW": g["CW"], "SW": g["SW"], "SPL": g["SPL"], "K": g["K"], "POW": g["POW"],
               "FOLD": int(fold), "SFOLD": (g["POW"] if fold else "1.0f")}
    extra = "".join(f"#define {k} {v}\n" for k, v in defines.items())
    name = f"mtplx_inkling_panel_prompt{body}_gu_{'nvfp4' if fp4 else 'mxfp8'}_f{int(fold)}"
    return mx.fast.metal_kernel(name=name, input_names=["sorted_experts", "sorted_rows", "xin", "codes", "scales"],
                                output_names=["raw"], source=PANEL_PROMPT if body == 1 else PANEL_PROMPT2,
                                header=_header(extra), ensure_row_contiguous=True)


# Measured on real layer weights at 512 rows (lanes/l6/xprefill-5.json): MXFP8 gate-up 12.9 ms
# against 17.9 for the row-step body and 19.4 for the stock body's indexed gather.
PROMPT_LAYOUT = dict(BODY=2, S=4, RPS=4, MC=4, FOLD=1)


def prompt_gate_up(block, values, sorted_experts, sorted_rows, layout=None):
    """Gate-up raw products of a prompt chunk's pairs sorted by expert, (pairs, 2304) float32 at
    their sorted positions: each pair MLX's gathered (one-row) product bit for bit, each expert
    read once for all its members. `values` (rows, 2304) or (rows, 1, 2304) float32."""
    fp4 = block.gate_up.mode == "nvfp4"
    lay = layout or PROMPT_LAYOUT
    pairs = sorted_experts.shape[0]
    tgr = lay["S"] * lay["RPS"]
    body = lay.get("BODY", 1)
    template = [("S", lay["S"]), ("RPS", lay["RPS"]), ("MC", lay["MC"]), ("PAIRS", pairs)]
    if body == 1:
        template.insert(2, ("R", lay["R"]))
    return _panel_prompt_kernel(fp4, lay.get("FOLD", 1), body)(
        inputs=[sorted_experts, sorted_rows, values.reshape(-1, HIDDEN), block.gate_up.codes, block.gate_up.scales],
        template=template,
        grid=((HIDDEN // tgr) * 32 * lay["S"], block.gate_up.codes.shape[0], 1), threadgroup=(32 * lay["S"], 1, 1),
        output_shapes=[(pairs, HIDDEN)], output_dtypes=[mx.float32],
    )[0]


@lru_cache(maxsize=None)
def _route_plan_kernel():
    return mx.fast.metal_kernel(name="mtplx_inkling_panel_route_plan", input_names=["logits", "bias", "rconsts"],
                                output_names=["experts", "prob", "plan"], source=ROUTE_PLAN,
                                header=_header(f"#define PLAN_STRIDE {PLAN_STRIDE}\n"), ensure_row_contiguous=True)


def _panel_raw(product, fp4, plan, xin, codes, scales, rows, layout=None):
    lay = layout or PANEL_LAYOUT[(product, fp4)]
    tgr = lay["S"] * lay["RPS"]
    return _panel_kernel(product, fp4, lay.get("FOLD", 1))(
        inputs=[plan, xin, codes, scales],
        template=[("S", lay["S"]), ("RPS", lay["RPS"]), ("R", lay["R"]), ("MC", lay["MC"])],
        grid=((HIDDEN // tgr) * 32 * lay["S"], min(TOP_K * rows, EXPERTS), 1), threadgroup=(32 * lay["S"], 1, 1),
        output_shapes=[(rows * TOP_K, HIDDEN)], output_dtypes=[mx.float32],
    )[0]


@lru_cache(maxsize=None)
def _pair_rows(rows):
    return mx.repeat(mx.arange(rows, dtype=mx.uint32), TOP_K)


def _panel_products(block, fp4, rows, logits, values, outer, layouts=None, dtype=mx.bfloat16):
    """Route + plan, gate-up products, RC6's SwiGLU and down-input rounding, down products
    (4 launches), each distinct expert of the panel read once."""
    pairs = rows * block.count
    experts, prob, plan = _route_plan_kernel()(
        inputs=[logits, block.bias, _route_consts(float(block.scale))],
        template=[("W", rows)], grid=(256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(pairs,), (pairs,), (1 + EXPERTS * PLAN_STRIDE,)],
        output_dtypes=[mx.uint32, mx.float32, mx.uint32],
    )
    lay = layouts or {}
    raw13 = _panel_raw("gu", fp4, plan, values, block.gate_up.codes, block.gate_up.scales, rows, lay.get("gu"))
    mid, *mid_outer = expert_fusion._kernel("swiglu", fp4, _caps(float(block.gate_cap), float(block.up_cap)))(
        inputs=[raw13, *([outer, block.gate_up.outer] if fp4 else []), experts, _pair_rows(rows)],
        template=[("T", dtype), ("W", MIDDLE), ("E", MIDDLE // expert_fusion.THREADS)],
        grid=(pairs * expert_fusion.THREADS, 1, 1), threadgroup=(expert_fusion.THREADS, 1, 1),
        output_shapes=[(pairs, 1, MIDDLE), *([(pairs,)] if fp4 else [])],
        output_dtypes=[mx.float32] * (2 if fp4 else 1),
    )
    raw2 = _panel_raw("dn", fp4, plan, mid.reshape(pairs, MIDDLE), block.down.codes, block.down.scales, rows,
                      lay.get("dn"))
    return raw2, (mid_outer[0] if fp4 else None), experts, prob


#: Which expert products serve a panel (all exact, class A):
#:   "auto" (default): one row takes the block's own products (4 launches); 2 to 8 rows the panel
#:   products, each distinct expert read once with every member computed from the same decoded
#:   weights (6 launches); 9 to 32 rows MLX's gathered product with fused glue (6 launches).
#:   "mlx", "own" or "panel" pin one ("panel" for 1 to 8 rows; wider panels take "mlx").
PRODUCTS = os.environ.get("MTPLX_INKLING_FAST_EXPERT_PRODUCTS", "auto").lower()


def products_for(rows):
    if PRODUCTS in ("mlx", "own"):
        return PRODUCTS
    if PRODUCTS == "panel":
        return "panel" if rows <= PANEL_ROWS else "mlx"
    return "own" if rows == 1 else "panel" if rows <= PANEL_ROWS else "mlx"


def expert_visit_parts(block, norm_weight, conv_weight, source, partial, tail, *, eps, conv_form=0, trace=None,
                       tiles=None, stop_after=None, products=None, panel_layouts=None, dtype=mx.bfloat16):
    """Shapes: source/partial (1, W, 2304) float32, tail (1, 3, 2304) float32; ``dtype`` the stream's
    16-bit type (bf16, float16 on the FP16 packs).

    Returns (out (1, W, 2304) in ``dtype``, conv_rows float32 (1, W, 2304)).
    """
    products = products or products_for(source.shape[-2])
    rows = source.shape[-2]
    if products == "panel" and rows > PANEL_ROWS:
        products = "mlx"
    fp4 = block.gate_up.mode == "nvfp4"
    tl = DEFAULT_TILES if tiles is None else {**DEFAULT_TILES, **tiles}
    src = source.reshape(rows, HIDDEN)
    part = partial.reshape(rows, HIDDEN)
    mc = _member_chunk(rows)
    common = (("W", rows),)

    logits, values, *outer = _kernel("inputs", fp4)(
        inputs=[src, norm_weight, block.router, _consts(float(eps))],
        template=[*common, ("T", dtype)],
        grid=(32 * 256, rows, 1), threadgroup=(256, 1, 1),
        output_shapes=[(rows, EXPERTS), (rows, HIDDEN), *([(rows,)] if fp4 else [])],
        output_dtypes=[mx.float32] * (3 if fp4 else 2),
    )

    if stop_after == "inputs":
        return logits, values
    dn = block.down
    if products in ("mlx", "panel"):
        if products == "panel":
            raw, mid_outer_m, experts, prob = _panel_products(block, fp4, rows, logits, values,
                                                              outer[0] if fp4 else None, panel_layouts, dtype)
        else:
            raw, mid_outer_m, experts, prob = _mlx_products(block, fp4, rows, logits, values,
                                                            outer[0] if fp4 else None, dtype)
        mid_outer = [mid_outer_m] if fp4 else []
        h = peaks = None
        return _combine(fp4, rows, raw, mid_outer, experts, prob, dn, conv_weight, tail, part, conv_form, source,
                        trace, logits, values, outer, h, peaks, dtype)
    gu = block.gate_up
    cw13, sw13 = (HIDDEN // 8, HIDDEN // 16) if fp4 else (HIDDEN // 4, HIDDEN // 32)
    s_gu, r_gu, tile_gu = tl["gu_s"], tl["gu_r"], tl["gu_tile"]
    slots = min(TOP_K * rows, EXPERTS)
    ntile13 = 2 * MIDDLE // tile_gu
    caps = _caps(float(block.gate_cap), float(block.up_cap))
    h, peaks, experts, prob = _kernel("gate_up", fp4, caps)(
        inputs=[logits, block.bias, _route_consts(float(block.scale)), values,
                *([outer[0]] if fp4 else []), gu.codes, gu.scales, *([gu.outer] if fp4 else [])],
        template=[*common, ("T", dtype), ("S", s_gu), ("NT", 32 * s_gu), ("TILE", tile_gu),
                  ("R", r_gu), ("MC", mc), ("CW", cw13), ("SW", sw13), ("NTILE", ntile13)],
        grid=(ntile13 * 32 * s_gu, slots, 1), threadgroup=(32 * s_gu, 1, 1),
        output_shapes=[(rows * TOP_K, MIDDLE), (rows * TOP_K, ntile13), (rows * TOP_K,), (rows * TOP_K,)],
        output_dtypes=[dtype, mx.float32, mx.uint32, mx.float32],
    )

    if stop_after == "gate_up":
        return h, peaks
    dn = block.down
    cw2, sw2 = (MIDDLE // 8, MIDDLE // 16) if fp4 else (MIDDLE // 4, MIDDLE // 32)
    s_dn, r_dn, tile_dn = tl["dn_s"], tl["dn_r"], tl["dn_tile"]
    ntile2 = HIDDEN // tile_dn
    raw, *mid_outer = _kernel("down", fp4)(
        inputs=[experts, h, peaks, dn.codes, dn.scales],
        template=[*common, ("S", s_dn), ("NT", 32 * s_dn), ("TILE", tile_dn), ("R", r_dn), ("MC", mc),
                  ("CW", cw2), ("SW", sw2), ("NTILE13", ntile13)],
        grid=(ntile2 * 32 * s_dn, slots, 1), threadgroup=(32 * s_dn, 1, 1),
        output_shapes=[(rows * TOP_K, HIDDEN), *([(rows * TOP_K,)] if fp4 else [])],
        output_dtypes=[mx.float32] * (2 if fp4 else 1),
    )

    if stop_after == "down":
        return raw, experts
    return _combine(fp4, rows, raw, mid_outer, experts, prob, dn, conv_weight, tail, part, conv_form, source,
                    trace, logits, values, outer, h, peaks, dtype)


def _combine(fp4, rows, raw, mid_outer, experts, prob, dn, conv_weight, tail, part, conv_form, source,
             trace, logits, values, outer, h, peaks, dtype=mx.bfloat16):
    out, conv_rows = _kernel("combine", fp4, (("CONV_FORM", conv_form), ("DIVISOR", _divisor_bits())))(
        inputs=[raw, *([mid_outer[0]] if fp4 else []), experts, prob, *([dn.outer] if fp4 else []),
                conv_weight, tail.reshape(3, HIDDEN), part],
        template=[("W", rows), ("T", dtype)],
        grid=(HIDDEN, 1, 1), threadgroup=(64, 1, 1),
        output_shapes=[(rows, HIDDEN), (rows, HIDDEN)],
        output_dtypes=[dtype, mx.float32],
    )
    if trace is not None:
        trace.update(logits=logits, values=values, outer=outer[0] if fp4 else None, h=h, peaks=peaks,
                     experts=experts, prob=prob, raw=raw, mid_outer=mid_outer[0] if fp4 else None)
    return out.reshape(source.shape), conv_rows.reshape(source.shape)


@lru_cache(maxsize=4)
def _zero_tail(width=HIDDEN):
    return mx.zeros((1, 3, width), mx.float32)


def reference_visit(layer, source, partial, tail, dtype=mx.bfloat16):
    """RC6's own statements for the same edge (model.py 533-536), for fallback and proofs."""
    from mtplx.models.inkling import model as rc6

    hidden = rc6.norm(source, layer.ffn_norm, layer.eps).astype(dtype)
    routed = layer.experts(hidden)
    cache = [None, None, None, tail]
    conv = layer.convs[3](routed, cache, 3, tape={} if source.shape[-2] > 1 else None)
    return (partial + conv.astype(mx.float32)).astype(dtype), routed.astype(mx.float32)


def prove(layer, source, partial, tail, dtype=mx.bfloat16):
    """(True, None) when the fast block gives RC6's bits on these operands, else (False, why).

    Host reads: call it outside a compiled step (at load or first use per format and width).
    """
    want = reference_visit(layer, source, partial, tail, dtype)
    got = expert_visit(layer, source, partial, tail, dtype)
    mx.eval(want, got)
    for name, a, b in zip(("output", "convolution rows"), got, want):
        view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
        if a.shape != b.shape or a.dtype != b.dtype or not bool(mx.array_equal(a.view(view), b.view(view)).item()):
            return False, f"{name} differs from RC6 at {source.shape[-2]} rows"
    return True, None


def expert_visit(layer, source, partial, tail, dtype=mx.bfloat16):
    """Stages D and E of one layer visit for a 1-32 row panel (SIGNATURES.md section 3).

    `layer` is the RC6 `model.Layer`; `source` the float32 MLP input after the depth read
    and controller, before the norm; `partial` the float32 attention-side partial sum; `tail`
    the MLP convolution's committed (1, 3, 2304) float32 tail, oldest first (zeros before
    position 3); `dtype` the stream's 16-bit type. Returns (visit output in `dtype`, the panel's
    MLP convolution input rows float32).
    """
    tail = tail if tail is not None else _zero_tail()
    _count("fast." + products_for(source.shape[-2]))
    return expert_visit_parts(layer.experts, layer.ffn_norm, layer.convs[3].weight, source, partial, tail,
                              eps=layer.eps, dtype=dtype)
