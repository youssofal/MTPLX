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
TILE_GATE_UP = 64     # w13 rows per gate-up threadgroup (32 mid outputs)
TILE_DOWN = 64        # w2 rows per down threadgroup

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
    for (uint i = tid; i < 128u; i += NT) pl_used[i] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < uint(W) * 8u; i += NT) pl_used[PAIR_EXPERT(i)] = 1u;
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
    // This threadgroup's four router rows, in the one-row gemv's per-thread slices.
    const int bn = int(32 * sg + lane) * 4;
    const bool tail_ok = 2048 + bn + 4 <= 2304;
    float wr0[4][4], wr1[4][4], wr2[4][4];
    for (int tm = 0; tm < 4; ++tm) {
        const device float* row = router + (4 * j + tm) * 2304;
        for (int tn = 0; tn < 4; ++tn) {
            wr0[tm][tn] = row[bn + tn];
            wr1[tm][tn] = row[1024 + bn + tn];
            wr2[tm][tn] = tail_ok ? row[2048 + bn + tn] : 0.0f;
        }
    }
    if (t < 32) sums[t] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = 0; r < uint(W); ++r) {
        const device float* xr = source + r * 2304;
        // MLX rms_single_row: 576 threads of 4 values; this threadgroup plays its 18 simdgroups.
        for (uint s = sg; s < 18; s += 8) {
            float thread_x[4];
            float acc = 0;
            const uint base = 4 * (32 * s + lane);
            for (int i = 0; i < 4; i++) {
                thread_x[i] = xr[base + i];
                acc += thread_x[i] * thread_x[i];
            }
            acc = simd_sum(acc);
            if (lane == 0) sums[s] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float acc = simd_sum(sums[lane]);
            if (lane == 0) inv_mean[0] = metal::precise::rsqrt(acc / 2304.0f + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv = inv_mean[0];
        float local = 0;
        bool nan_seen = false;
        for (uint i = t; i < 2304; i += 256) {
            const float xi = xr[i];
            const float wi = float(norm_w[i]);
            const float y = wi * static_cast<float>(xi * inv);
            const float v = float(static_cast<bfloat16_t>(y));
            normed[i] = v;
            const float a = abs(v);
            nan_seen = nan_seen || isnan(a);
            local = max(local, a);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Router logits: MLX one-row gemv, bm1 bn8 sm1 sn32 tm4 tn4.
        float v0[4], v1[4], v2[4];
        for (int tn = 0; tn < 4; ++tn) {
            v0[tn] = normed[bn + tn];
            v1[tn] = normed[1024 + bn + tn];
            v2[tn] = tail_ok ? normed[2048 + bn + tn] : 0.0f;
        }
        float res[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int tm = 0; tm < 4; ++tm)
            for (int tn = 0; tn < 4; ++tn)
                res[tm] += wr0[tm][tn] * v0[tn];
        for (int tm = 0; tm < 4; ++tm)
            for (int tn = 0; tn < 4; ++tn)
                res[tm] += wr1[tm][tn] * v1[tn];
        for (int tm = 0; tm < 4; ++tm)
            for (int tn = 0; tn < 4; ++tn)
                res[tm] += wr2[tm][tn] * v2[tn];
        for (int tm = 0; tm < 4; ++tm)
            for (ushort sn = 16; sn >= 1; sn >>= 1)
                res[tm] += simd_shuffle_down(res[tm], sn);
        if (lane == 0)
            for (int tm = 0; tm < 4; ++tm) red[sg * 8 + tm] = res[tm];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            for (int tm = 0; tm < 4; ++tm) {
                float acc = res[tm];
                for (int s = 1; s < 8; ++s) acc += red[s * 8 + tm];
                logits[r * 128 + 4 * j + tm] = acc;
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
    constexpr uint RS = TILE / S;
    const uint n_sg = tile * TILE + sg * RS;
    threadgroup float tpk[S * MC];
    threadgroup uint tnan[S * MC];
    for (uint m0 = 0; m0 < M; m0 += MC) {
        const uint mc = min(uint(MC), M - m0);
        const device float* xm[MC];
        for (uint m = 0; m < MC; ++m) xm[m] = values + pl_row[m0 + min(m, mc - 1)] * 2304 + 8 * lane;
        float pk[MC];
        bool pn[MC];
        for (uint m = 0; m < MC; ++m) { pk[m] = 0.0f; pn[m] = false; }
        for (uint rb = 0; rb < RS; rb += R) {
            const uint nb = n_sg + rb;
            float res[MC][R];
            for (uint m = 0; m < MC; ++m)
                for (uint q = 0; q < R; ++q) res[m][q] = 0.0f;
            for (uint b = 0; b < 9; ++b) {
                float xv[MC][8];
                for (uint m = 0; m < MC; ++m) {
                    const float4 a = *(const device float4*)(xm[m] + 256 * b);
                    const float4 c = *(const device float4*)(xm[m] + 256 * b + 4);
                    xv[m][0] = a.x; xv[m][1] = a.y; xv[m][2] = a.z; xv[m][3] = a.w;
                    xv[m][4] = c.x; xv[m][5] = c.y; xv[m][6] = c.z; xv[m][7] = c.w;
                }
                for (uint q = 0; q < R; ++q) {
                    const uint n = nb + q;
#if FP4
                    const uint word = wexp[n * CW + 32 * b + lane];
                    const float s = l3_e4m3(sexp[n * SW + 16 * b + lane / 2]);
                    float wv[8];
                    for (uint i = 0; i < 8; ++i) wv[i] = l3_e2m1(word >> (4 * i));
                    for (uint m = 0; m < MC; ++m) {
                        float accum = 0;
                        accum += (xv[m][0] * wv[0] + xv[m][1] * wv[1] + xv[m][2] * wv[2] + xv[m][3] * wv[3]);
                        accum += (xv[m][4] * wv[4] + xv[m][5] * wv[5] + xv[m][6] * wv[6] + xv[m][7] * wv[7]);
                        const float tt = s * accum;
                        res[m][q] = res[m][q] + tt;
                    }
#else
                    const uint w0 = wexp[n * CW + 64 * b + 2 * lane];
                    const uint w1 = wexp[n * CW + 64 * b + 2 * lane + 1];
                    const float s = l3_e8m0(sexp[n * SW + 8 * b + lane / 4]);
                    float wv[8];
                    for (uint i = 0; i < 4; ++i) {
                        wv[i] = l3_e4m3((w0 >> (8 * i)) & 255u);
                        wv[4 + i] = l3_e4m3((w1 >> (8 * i)) & 255u);
                    }
                    for (uint m = 0; m < MC; ++m) {
                        float accum = 0;
                        for (uint i = 0; i < 8; ++i) accum += xv[m][i] * wv[i];
                        const float tt = s * accum;
                        res[m][q] = res[m][q] + tt;
                    }
#endif
                }
            }
            for (uint m = 0; m < MC; ++m)
                for (uint q = 0; q < R; ++q) res[m][q] = simd_sum(res[m][q]);
            // Capped SwiGLU (RC6 SWIGLU statements); lane m * R/2 + jj takes member m, mid nb/2 + jj.
            float graw = 0, uraw = 0;
            uint mine = 0;
            for (uint m = 0; m < MC; ++m)
                for (uint jj = 0; jj < R / 2; ++jj)
                    if (lane == m * (R / 2) + jj) { graw = res[m][2 * jj]; uraw = res[m][2 * jj + 1]; mine = m; }
            const bool active = lane < MC * (R / 2) && mine < mc;
            float hv = 0.0f;
            if (active) {
                const uint jj = lane % (R / 2);
                const uint pair = pl_pair[m0 + mine];
#if FP4
                const float scale = row_outer[pl_row[m0 + mine]] * eo;
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
                h[pair * 1152 + nb / 2 + jj] = mid;
                hv = float(mid);
            }
            for (uint m = 0; m < MC; ++m) {
                const bool mm = active && mine == m;
                const float a = mm ? abs(hv) : 0.0f;
                pk[m] = max(pk[m], simd_max(a));
                pn[m] = pn[m] || simd_any(mm && isnan(a));
            }
        }
        if (lane == 0)
            for (uint m = 0; m < MC; ++m) { tpk[sg * MC + m] = pk[m]; tnan[sg * MC + m] = pn[m] ? 1u : 0u; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < mc) {
            float best = 0.0f;
            bool any = false;
            for (uint s2 = 0; s2 < S; ++s2) {
                best = max(best, tpk[s2 * MC + tid]);
                any = any || tnan[s2 * MC + tid] != 0u;
            }
            peaks[pl_pair[m0 + tid] * NTILE + tile] = any ? NAN : best;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
    constexpr uint RS = TILE / S;
    const uint n_sg = tile * TILE + sg * RS;
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
        for (uint rb = 0; rb < RS; rb += R) {
            const uint nb = n_sg + rb;
            float res[MC][R];
            for (uint m = 0; m < MC; ++m)
                for (uint q = 0; q < R; ++q) res[m][q] = 0.0f;
#if FP4
            for (uint b = 0; b < 5; ++b) {
                if (b == 4 && lane >= 16) break;
                float xv[MC][8];
                for (uint m = 0; m < MC; ++m)
                    for (uint v = 0; v < 8; ++v) xv[m][v] = xs[m * 1280 + (b * 8 + v) * 32 + lane];
                for (uint q = 0; q < R; ++q) {
                    const uint n = nb + q;
                    const uint word = wexp[n * CW + 32 * b + lane];
                    const float s = l3_e4m3(sexp[n * SW + 16 * b + lane / 2]);
                    float wv[8];
                    for (uint i = 0; i < 8; ++i) wv[i] = l3_e2m1(word >> (4 * i));
                    for (uint m = 0; m < MC; ++m) {
                        float accum = 0;
                        accum += (xv[m][0] * wv[0] + xv[m][1] * wv[1] + xv[m][2] * wv[2] + xv[m][3] * wv[3]);
                        accum += (xv[m][4] * wv[4] + xv[m][5] * wv[5] + xv[m][6] * wv[6] + xv[m][7] * wv[7]);
                        const float tt = s * accum;
                        res[m][q] = res[m][q] + tt;
                    }
                }
            }
#else
            for (uint b = 0; b < 9; ++b) {
                float xv[MC][4];
                for (uint m = 0; m < MC; ++m)
                    for (uint v = 0; v < 4; ++v) xv[m][v] = xs[m * 1280 + (b * 4 + v) * 32 + lane];
                for (uint q = 0; q < R; ++q) {
                    const uint n = nb + q;
                    const uint word = wexp[n * CW + 32 * b + lane];
                    const float s = l3_e8m0(sexp[n * SW + 4 * b + lane / 8]);
                    float wv[4];
                    for (uint i = 0; i < 4; ++i) wv[i] = l3_e4m3((word >> (8 * i)) & 255u);
                    for (uint m = 0; m < MC; ++m) {
                        float accum = 0;
                        for (uint i = 0; i < 4; ++i) accum += xv[m][i] * wv[i];
                        const float tt = s * accum;
                        res[m][q] = res[m][q] + tt;
                    }
                }
            }
#endif
            for (uint m = 0; m < MC; ++m)
                for (uint q = 0; q < R; ++q) {
                    const float total = simd_sum(res[m][q]);
                    if (lane == q && m < mc) raw[pl_pair[m0 + m] * 2304 + nb + q] = total;
                }
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
def _consts(eps):
    return mx.array([eps], mx.float32)


@lru_cache(maxsize=None)
def _route_consts(scale):
    return mx.array([scale, 1e-20], mx.float32)


def unserved(block, rows, dtype=mx.bfloat16):
    """Why the fast path cannot serve this visit, or None."""
    if os.environ.get("MTPLX_INKLING_FAST_EXPERTS", "on").lower() in {"0", "off", "stock"}:
        return "disabled"
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


def expert_visit_parts(block, norm_weight, conv_weight, source, partial, tail, *, eps, conv_form=0, trace=None):
    """Four launches. Shapes: source/partial (1, W, 2304) float32, tail (1, 3, 2304) float32.

    Returns (out bf16 (1, W, 2304), conv_rows float32 (1, W, 2304)).
    """
    rows = source.shape[-2]
    fp4 = block.gate_up.mode == "nvfp4"
    src = source.reshape(rows, HIDDEN)
    part = partial.reshape(rows, HIDDEN)
    mc = _member_chunk(rows)
    common = (("W", rows),)

    logits, values, *outer = _kernel("inputs", fp4)(
        inputs=[src, norm_weight, block.router, _consts(float(eps))],
        template=[*common],
        grid=(32 * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(rows, EXPERTS), (rows, HIDDEN), *([(rows,)] if fp4 else [])],
        output_dtypes=[mx.float32] * (3 if fp4 else 2),
    )

    gu = block.gate_up
    cw13, sw13 = (HIDDEN // 8, HIDDEN // 16) if fp4 else (HIDDEN // 4, HIDDEN // 32)
    s_gu, r_gu = 4, 8
    slots = min(TOP_K * rows, EXPERTS)
    ntile13 = 2 * MIDDLE // TILE_GATE_UP
    caps = (("GATE_CAP", expert_fusion.mlx_literal(block.gate_cap)), ("UP_CAP", expert_fusion.mlx_literal(block.up_cap)))
    h, peaks, experts, prob = _kernel("gate_up", fp4, caps)(
        inputs=[logits, block.bias, _route_consts(float(block.scale)), values,
                *([outer[0]] if fp4 else []), gu.codes, gu.scales, *([gu.outer] if fp4 else [])],
        template=[*common, ("T", mx.bfloat16), ("S", s_gu), ("NT", 32 * s_gu), ("TILE", TILE_GATE_UP),
                  ("R", r_gu), ("MC", mc), ("CW", cw13), ("SW", sw13), ("NTILE", ntile13)],
        grid=(ntile13 * 32 * s_gu, slots, 1), threadgroup=(32 * s_gu, 1, 1),
        output_shapes=[(rows * TOP_K, MIDDLE), (rows * TOP_K, ntile13), (rows * TOP_K,), (rows * TOP_K,)],
        output_dtypes=[mx.bfloat16, mx.float32, mx.uint32, mx.float32],
    )

    dn = block.down
    cw2, sw2 = (MIDDLE // 8, MIDDLE // 16) if fp4 else (MIDDLE // 4, MIDDLE // 32)
    s_dn, r_dn = 4, 8
    ntile2 = HIDDEN // TILE_DOWN
    raw, *mid_outer = _kernel("down", fp4)(
        inputs=[experts, h, peaks, dn.codes, dn.scales],
        template=[*common, ("S", s_dn), ("NT", 32 * s_dn), ("TILE", TILE_DOWN), ("R", r_dn), ("MC", mc),
                  ("CW", cw2), ("SW", sw2), ("NTILE13", ntile13)],
        grid=(ntile2 * 32 * s_dn, slots, 1), threadgroup=(32 * s_dn, 1, 1),
        output_shapes=[(rows * TOP_K, HIDDEN), *([(rows * TOP_K,)] if fp4 else [])],
        output_dtypes=[mx.float32] * (2 if fp4 else 1),
    )

    out, conv_rows = _kernel("combine", fp4, (("CONV_FORM", conv_form), ("DIVISOR", _divisor_bits())))(
        inputs=[raw, *([mid_outer[0]] if fp4 else []), experts, prob, *([dn.outer] if fp4 else []),
                conv_weight, tail.reshape(3, HIDDEN), part],
        template=[*common, ("T", mx.bfloat16)],
        grid=(HIDDEN, 1, 1), threadgroup=(64, 1, 1),
        output_shapes=[(rows, HIDDEN), (rows, HIDDEN)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )
    if trace is not None:
        trace.update(logits=logits, values=values, outer=outer[0] if fp4 else None, h=h, peaks=peaks,
                     experts=experts, prob=prob, raw=raw, mid_outer=mid_outer[0] if fp4 else None)
    return out.reshape(source.shape), conv_rows.reshape(source.shape)


def expert_visit(layer, source, partial, tail):
    """Stages D and E of one layer visit for a 1-32 row panel (SIGNATURES.md section 3).

    `layer` is the RC6 `model.Layer`; `source` the float32 MLP input after the depth read
    and controller, before the norm; `partial` the float32 attention-side partial sum; `tail`
    the MLP convolution's committed (1, 3, 2304) float32 tail, oldest first (zeros before
    position 3). Returns (visit output bf16, the panel's MLP convolution input rows float32).
    """
    tail = tail if tail is not None else mx.zeros((1, 3, HIDDEN), mx.float32)
    _count("fast")
    return expert_visit_parts(layer.experts, layer.ffn_norm, layer.convs[3].weight, source, partial, tail,
                              eps=layer.eps)
