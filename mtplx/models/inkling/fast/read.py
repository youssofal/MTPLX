"""The depth read of an Inkling 2 Mini layer visit, its cross-pass controllers and norms.

A visit reads its input twice from the closed residual blocks and the running partial: once
with the layer's attention query (one query, three on global layers) and once with its MLP
query. On loop passes two and three each source then goes through a low-rank controller
driven by the visit's cross-pass message. RC6 writes this as long chains of small MLX
operations (``model.py`` ``read_blocks``, ``serial_depth_dots``, ``Controller``, ``norm`` and
``projection.py`` ``stock_messages``): about 25 launches a read on the first pass and about
70 more on passes two and three.

Two kernels here:

* ``read``: one threadgroup per panel row (and, with controllers, per source and slice of
  the controller's down product). The depth read, the message read, the controller norms,
  a slice of the down product, or the input norm(s).
* ``mix``: the controllers' forget and gain products, the tanh mix, and the per-128 partial
  sums the following input norm needs; one threadgroup per source and 128-column slice,
  serving every panel row.

Exactness. Each step repeats MLX 0.32.2's own kernel for that step, thread for thread:
``rms_single_row`` (4 values a thread fused-multiply-added from zero, ``simd_sum``, then
``simd_sum`` over the zero-padded simdgroup totals, ``precise::rsqrt``), the one-query
``gemv`` (``bn8``: eight slices of 1,024-column blocks with a zero-padded tail, a tree per
slice, slices added in order), ``block_softmax`` (``fast::exp``, ``1 / sum``), ``gemv_t``
(``bn16 sm8 sn4``: sequential over sources, then three shuffle steps), the controller's
``gemv`` (down: ``bn8`` over 4,608; forget/gain: ``sm8 sn4`` over 64) and ``precise::tanh``.
One-query reads, every norm and the controllers are therefore class A. RC6's three-query
depth read (global visits) and four-query message read go through MLX's matrix-unit GEMM,
whose order no kernel can repeat and which differs between Macs; here each query is computed
exactly as MLX's one-query path would (class C, fixed and row-stable, measured).
"""

from __future__ import annotations

from functools import lru_cache
import math
import struct

import mlx.core as mx

#: Down-product rows one controller threadgroup computes (two blocks of MLX's four).
_DOWN_ROWS = 8
#: Columns of the controller output one ``mix`` threadgroup serves.
_MIX_COLS = 128


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


_READ_HEADER = """
// One closed block or the partial, element c of panel row r, as float.
#define SOURCE(s, r, c)                                                                    \\
    ((s) == S - 1 ? static_cast<float>(partial[size_t(r) * D + (c)])                      \\
     : (s) == 0 ? static_cast<float>(c0[size_t(r) * D + (c)])                              \\
     : (s) == 1 ? static_cast<float>(c1[size_t(r) * D + (c)])                              \\
     : (s) == 2 ? static_cast<float>(c2[size_t(r) * D + (c)])                              \\
     : static_cast<float>(c3[size_t(r) * D + (c)]))
"""

_READ = """
    constexpr uint NSG = TPG / 32;
    constexpr uint FULL = D / 1024;
    constexpr uint LEFT = D - FULL * 1024;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint t = thread_index_in_threadgroup;
    const uint g = threadgroup_position_in_grid.x;
    // CTRL: g = (row * NC + ctrl) * CHUNKS + chunk; otherwise g = row.
    const uint chunk = MODE == 2 ? g % CHUNKS : 0;
    const uint ctrl = MODE == 2 ? (g / CHUNKS) % NC : 0;
    const uint r = MODE == 2 ? g / (CHUNKS * NC) : g;
    // Which read query feeds this threadgroup's output(s); which message query its controller uses.
    const uint q_first = MODE == 2 ? (NQ == 3 ? Q_BASE + ctrl : 0) : 0;
    const uint q_count = MODE == 2 ? 1 : NQ;

    threadgroup float sums[(S > 2 ? S : 2) + 2][32];   // per-source rms partials (+2 for later norms)
    threadgroup float inv[(S > 2 ? S : 2) + 2];
    threadgroup float dots[4 * 5][8];                  // (query, source) x bn8 slice
    threadgroup float prob[4][8];
    threadgroup bfloat16_t mixed[MODE == 2 ? 2 * D : 1];
    threadgroup float down_part[2][8][4];

    const bool in_row = t * 4 < D;                     // threads past D/4 hold no columns
    const uint c4 = t * 4;

    float out[3][4];
    if (NOREAD) {
        // Layer 0 has no depth read: RC6 takes x itself (read = x[..., None, :]).
        for (uint i = 0; i < 4; ++i) out[0][i] = in_row ? SOURCE(S - 1, r, c4 + i) : 0.0f;
    } else {
    // ---- 1. MLX rms_single_row over each source (closed blocks, then the partial): its inverse RMS.
    for (uint s = 0; s < S; ++s) {
        float acc = 0;
        for (uint i = 0; i < 4; ++i) {
            const float xi = in_row ? SOURCE(s, r, c4 + i) : 0.0f;
            acc += xi * xi;
        }
        acc = simd_sum(acc);
        if (lane == 0) sums[s][sg] = acc;
    }
    if (sg == 0) {
        for (uint s = 0; s < S; ++s) {
            if (lane >= NSG) sums[s][lane] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        for (uint s = 0; s < S; ++s) {
            const float total = simd_sum(sums[s][lane]);
            if (lane == 0) inv[s] = metal::precise::rsqrt(total / float(D) + as_type<float>(uint(EPS)));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- 2. One-query gemv (bn8) of each query with each normalized source: q . (x * inv).
    if (sg < 8) {
        for (uint qi = q_first; qi < q_first + q_count; ++qi) {
            for (uint s = 0; s < S; ++s) {
                const float is = inv[s];
                float p = 0.0f;
                for (uint b = 0; b < FULL; ++b) {
                    for (uint tn = 0; tn < 4; ++tn) {
                        const uint c = b * 1024 + sg * 128 + lane * 4 + tn;
                        const float key = 1.0f * (SOURCE(s, r, c) * is);
                        p += key * queries[qi * D + c];
                    }
                }
                if (LEFT > 0) {
                    const uint c0_ = FULL * 1024 + sg * 128 + lane * 4;
                    for (uint tn = 0; tn < 4; ++tn) {
                        const uint c = c0_ + tn;
                        const float key = (c < D) ? 1.0f * (SOURCE(s, r, c) * is) : 0.0f;
                        const float qv = (c < D) ? queries[qi * D + c] : 0.0f;
                        p += key * qv;
                    }
                }
                for (ushort sn = 16; sn >= 1; sn >>= 1) p += simd_shuffle_down(p, sn);
                if (lane == 0) dots[(qi - q_first) * S + s][sg] = p;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- 3. Scale and MLX block_softmax over the sources, one simdgroup per query.
    if (sg < q_count) {
        const uint qn = sg;
        float ld[4];
        for (uint i = 0; i < 4; ++i) {
            const uint s = lane * 4 + i;
            if (s < S) {
                float score = dots[qn * S + s][0];
                for (uint v = 1; v < 8; ++v) score += dots[qn * S + s][v];
                ld[i] = score * as_type<float>(uint(SCALE));
            } else {
                ld[i] = -INFINITY;
            }
        }
        float maxval = -FLT_MAX;
        for (uint i = 0; i < 4; ++i) maxval = (maxval < ld[i]) ? ld[i] : maxval;
        maxval = simd_max(maxval);
        float normalizer = 0;
        for (uint i = 0; i < 4; ++i) {
            const float e = metal::fast::exp(ld[i] - maxval);
            ld[i] = e;
            normalizer += e;
        }
        normalizer = simd_sum(normalizer);
        normalizer = 1 / normalizer;
        for (uint i = 0; i < 4; ++i) {
            const uint s = lane * 4 + i;
            if (s < S) prob[qn][s] = ld[i] * normalizer;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- 4. gemv_t mix (bn16 sm8 sn4): sources 0..3 in one lane, 4..7 in the next row group,
    // then MLX's three shuffle steps (16, 8, 4) that add the empty row groups.
    for (uint qn = 0; qn < q_count; ++qn) {
        for (uint i = 0; i < 4; ++i) {
            const uint c = c4 + i;
            float v0 = 0.0f, v1 = 0.0f;
            if (in_row) {
                for (uint s = 0; s < S && s < 4; ++s) {
                    const float key = 1.0f * (SOURCE(s, r, c) * inv[s]);
                    v0 += prob[qn][s] * key;
                }
                for (uint s = 4; s < S; ++s) {
                    const float key = 1.0f * (SOURCE(s, r, c) * inv[s]);
                    v1 += prob[qn][s] * key;
                }
            }
            const float w0 = v0 + 0.0f, w1 = v1 + 0.0f;
            const float x0 = w0 + 0.0f, x1 = w1 + 0.0f;
            out[qn][i] = x0 + x1;
        }
    }
    }   // NOREAD

    if (MODE == 1) {
        // The MLP source: the read itself, float32.
        if (in_row)
            for (uint i = 0; i < 4; ++i) o0[size_t(r) * D + c4 + i] = static_cast<OUT0>(out[0][i]);
        return;
    }

    if (MODE == 0) {
        // ---- 5. Input norm(s): MLX rms_single_row with the layer's gain, then the bf16 cast.
        for (uint qn = 0; qn < NQ; ++qn) {
            float acc = 0;
            for (uint i = 0; i < 4; ++i) acc += out[qn][i] * out[qn][i];
            acc = simd_sum(acc);
            if (lane == 0) sums[S + (qn & 1)][sg] = acc;
            if (sg == 0 && lane >= NSG) sums[S + (qn & 1)][lane] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
                const float total = simd_sum(sums[S + (qn & 1)][lane]);
                if (lane == 0) inv[S + (qn & 1)] = metal::precise::rsqrt(total / float(D) + as_type<float>(uint(EPS)));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float iv = inv[S + (qn & 1)];
            const device bfloat16_t* gain = qn == 0 ? g0 : (qn == 1 ? g1 : g2);
            if (in_row) {
                for (uint i = 0; i < 4; ++i) {
                    const float w = static_cast<float>(gain[c4 + i]);
                    o0[(size_t(r) * NQ + qn) * D + c4 + i] = static_cast<OUT0>(static_cast<bfloat16_t>(w * (out[qn][i] * iv)));
                }
            }
        }
        return;
    }

    // ---- MODE 2: one controller (index ctrl) for this row.
    // 5a. Its cross-pass message: one-query read of the normalized message states (class C
    // against RC6's four-query matrix-unit product), states used as they are.
    const uint mq = MSG_BASE + ctrl;
    if (sg < 8) {
        for (uint p_ = 0; p_ < P; ++p_) {
            float p = 0.0f;
            for (uint b = 0; b < FULL; ++b) {
                for (uint tn = 0; tn < 4; ++tn) {
                    const uint c = b * 1024 + sg * 128 + lane * 4 + tn;
                    p += mstates[(size_t(r) * P + p_) * D + c] * mqueries[mq * D + c];
                }
            }
            if (LEFT > 0) {
                const uint c0_ = FULL * 1024 + sg * 128 + lane * 4;
                for (uint tn = 0; tn < 4; ++tn) {
                    const uint c = c0_ + tn;
                    const float key = (c < D) ? mstates[(size_t(r) * P + p_) * D + c] : 0.0f;
                    const float qv = (c < D) ? mqueries[mq * D + c] : 0.0f;
                    p += key * qv;
                }
            }
            for (ushort sn = 16; sn >= 1; sn >>= 1) p += simd_shuffle_down(p, sn);
            if (lane == 0) dots[p_][sg] = p;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float ld[4];
        for (uint i = 0; i < 4; ++i) {
            const uint s = lane * 4 + i;
            if (s < P) {
                float score = dots[s][0];
                for (uint v = 1; v < 8; ++v) score += dots[s][v];
                ld[i] = score * as_type<float>(uint(SCALE));
            } else {
                ld[i] = -INFINITY;
            }
        }
        float maxval = -FLT_MAX;
        for (uint i = 0; i < 4; ++i) maxval = (maxval < ld[i]) ? ld[i] : maxval;
        maxval = simd_max(maxval);
        float normalizer = 0;
        for (uint i = 0; i < 4; ++i) {
            const float e = metal::fast::exp(ld[i] - maxval);
            ld[i] = e;
            normalizer += e;
        }
        normalizer = simd_sum(normalizer);
        normalizer = 1 / normalizer;
        for (uint i = 0; i < 4; ++i) {
            const uint s = lane * 4 + i;
            if (s < P) prob[3][s] = ld[i] * normalizer;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float msg[4];
    for (uint i = 0; i < 4; ++i) {
        const uint c = c4 + i;
        float v0 = 0.0f, v1 = 0.0f;
        if (in_row) {
            for (uint s = 0; s < P && s < 4; ++s) v0 += prob[3][s] * mstates[(size_t(r) * P + s) * D + c];
            for (uint s = 4; s < P; ++s) v1 += prob[3][s] * mstates[(size_t(r) * P + s) * D + c];
        }
        const float w0 = v0 + 0.0f, w1 = v1 + 0.0f;
        const float x0 = w0 + 0.0f, x1 = w1 + 0.0f;
        msg[i] = x0 + x1;
    }

    // 5b. The controller's two gain-free norms (MLX rms_single_row), concatenated, cast to bf16.
    {
        float ax = 0, am = 0;
        for (uint i = 0; i < 4; ++i) ax += out[0][i] * out[0][i];
        for (uint i = 0; i < 4; ++i) am += msg[i] * msg[i];
        ax = simd_sum(ax);
        am = simd_sum(am);
        if (lane == 0) { sums[S][sg] = ax; sums[S + 1][sg] = am; }
        if (sg == 0 && lane >= NSG) { sums[S][lane] = 0.0f; sums[S + 1][lane] = 0.0f; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            const float tx = simd_sum(sums[S][lane]);
            const float tm = simd_sum(sums[S + 1][lane]);
            if (lane == 0) {
                inv[S] = metal::precise::rsqrt(tx / float(D) + as_type<float>(uint(EPS)));
                inv[S + 1] = metal::precise::rsqrt(tm / float(D) + as_type<float>(uint(EPS)));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (in_row) {
            for (uint i = 0; i < 4; ++i) {
                mixed[c4 + i] = static_cast<bfloat16_t>(1.0f * (out[0][i] * inv[S]));
                mixed[D + c4 + i] = static_cast<bfloat16_t>(1.0f * (msg[i] * inv[S + 1]));
            }
            if (chunk == 0) {
                for (uint i = 0; i < 4; ++i) {
                    o1[(size_t(r) * NC + ctrl) * D + c4 + i] = out[0][i];
                    o2[(size_t(r) * NC + ctrl) * D + c4 + i] = msg[i];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 5c. Rows chunk*8 .. +7 of the controller's down product (64 x 2D, bf16), MLX gemv bn8.
    if (sg < 16) {
        const uint blk = sg / 8;           // which 4-row block of the slice
        const uint v = sg % 8;             // bn8 slice
        const device bfloat16_t* dw = ctrl == 0 ? d0 : (ctrl == 1 ? d1 : d2);
        const uint row0 = chunk * DROWS + blk * 4;
        constexpr uint K2 = 2 * D;
        constexpr uint FULL2 = K2 / 1024;
        constexpr uint LEFT2 = K2 - FULL2 * 1024;
        float p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint b = 0; b < FULL2; ++b) {
            const uint c = b * 1024 + v * 128 + lane * 4;
            float xv[4];
            for (uint tn = 0; tn < 4; ++tn) xv[tn] = static_cast<float>(mixed[c + tn]);
            for (uint tm = 0; tm < 4; ++tm)
                for (uint tn = 0; tn < 4; ++tn)
                    p[tm] += static_cast<float>(dw[size_t(row0 + tm) * K2 + c + tn]) * xv[tn];
        }
        if (LEFT2 > 0) {
            const uint c = FULL2 * 1024 + v * 128 + lane * 4;
            float xv[4];
            for (uint tn = 0; tn < 4; ++tn) xv[tn] = (c + tn < K2) ? static_cast<float>(mixed[c + tn]) : 0.0f;
            for (uint tm = 0; tm < 4; ++tm)
                for (uint tn = 0; tn < 4; ++tn) {
                    const float w = (c + tn < K2) ? static_cast<float>(dw[size_t(row0 + tm) * K2 + c + tn]) : 0.0f;
                    p[tm] += w * xv[tn];
                }
        }
        for (uint tm = 0; tm < 4; ++tm)
            for (ushort sn = 16; sn >= 1; sn >>= 1) p[tm] += simd_shuffle_down(p[tm], sn);
        if (lane == 0)
            for (uint tm = 0; tm < 4; ++tm) down_part[blk][v][tm] = p[tm];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < DROWS) {
        const uint blk = t / 4, tm = t % 4;
        float total = down_part[blk][0][tm];
        for (uint v = 1; v < 8; ++v) total += down_part[blk][v][tm];
        o0[(size_t(r) * NC + ctrl) * 64 + chunk * DROWS + t] =
            static_cast<OUT0>(static_cast<float>(static_cast<bfloat16_t>(total)));
    }
"""

_MIX = """
    // One threadgroup: controller ctrl, output columns col0 .. col0+127, every panel row.
    constexpr uint NSG = 8;
    constexpr uint RC = W < 8 ? W : 8;       // panel rows per pass
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint t = thread_index_in_threadgroup;
    const uint g = threadgroup_position_in_grid.x;
    constexpr uint SLICES = D / 128;
    const uint ctrl = g / SLICES;
    const uint col0 = (g % SLICES) * 128;
    threadgroup float fg[2][RC][128];

    const uint thrM = lane / 4, thrN = lane % 4;
    const bool gain_rows = sg >= 4;
    const uint row0 = col0 + (sg % 4) * 32 + thrM * 4;
    const device float* fw32 = ctrl == 0 ? f0 : (ctrl == 1 ? f1 : f2);
    const device float* gw32 = ctrl == 0 ? h0 : (ctrl == 1 ? h1 : h2);
    const device float* mat = gain_rows ? gw32 : fw32;

    for (uint rc = 0; rc < W; rc += RC) {
        // MLX gemv sm8 sn4 (K = 64): a simdgroup covers 32 rows, lane (thrM, thrN) rows 4 thrM..+3,
        // columns 16 i + 4 thrN + tn; then shuffles by 2 and 1 across thrN.
        float acc[4][RC];
        for (uint tm = 0; tm < 4; ++tm)
            for (uint rr = 0; rr < RC; ++rr) acc[tm][rr] = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            const uint c = 16 * i + 4 * thrN;
            float wt[4][4];
            for (uint tm = 0; tm < 4; ++tm)
                for (uint tn = 0; tn < 4; ++tn) wt[tm][tn] = mat[size_t(row0 + tm) * 64 + c + tn];
            for (uint rr = 0; rr < RC; ++rr) {
                if (rc + rr < W) {
                    float sv[4];
                    for (uint tn = 0; tn < 4; ++tn) sv[tn] = state[(size_t(rc + rr) * NC + ctrl) * 64 + c + tn];
                    for (uint tm = 0; tm < 4; ++tm)
                        for (uint tn = 0; tn < 4; ++tn) acc[tm][rr] += wt[tm][tn] * sv[tn];
                }
            }
        }
        for (uint tm = 0; tm < 4; ++tm)
            for (uint rr = 0; rr < RC; ++rr)
                for (ushort sn = 2; sn >= 1; sn >>= 1) acc[tm][rr] += simd_shuffle_down(acc[tm][rr], sn);
        if (thrN == 0)
            for (uint tm = 0; tm < 4; ++tm)
                for (uint rr = 0; rr < RC; ++rr)
                    fg[gain_rows ? 1 : 0][rr][(sg % 4) * 32 + thrM * 4 + tm] = acc[tm][rr];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // (1 - tanh(f)) * x + tanh(g) * message, each operation rounded on its own as RC6's launches are.
        for (uint idx = t; idx < RC * 128; idx += 256) {
            const uint rr = idx / 128, c = idx % 128;
            if (rc + rr < W) {
                const size_t at = (size_t(rc + rr) * NC + ctrl) * D + col0 + c;
                const float t1 = metal::precise::tanh(fg[0][rr][c]);
                const float a = 1.0f - t1;
                const float b = a * xs[at];
                const float t2 = metal::precise::tanh(fg[1][rr][c]);
                const float m = t2 * ms[at];
                const float hval = b + m;
                hout[at] = hval;
                fg[0][rr][c] = hval;
            }
        }
        if (PARTS_OUT) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // The following norm's partial for these 128 columns: one simdgroup of MLX rms_single_row.
            for (uint rr = sg; rr < RC; rr += NSG) {
                if (rc + rr < W) {
                    float acc2 = 0;
                    for (uint i = 0; i < 4; ++i) {
                        const float x = fg[0][rr][lane * 4 + i];
                        acc2 += x * x;
                    }
                    acc2 = simd_sum(acc2);
                    if (lane == 0) parts[(size_t(rc + rr) * NC + ctrl) * SLICES + col0 / 128] = acc2;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""


@lru_cache(maxsize=1)
def _read_kernel():
    # Outputs: mode 0 o0 = normed inputs (bf16); mode 1 o0 = the read (f32); mode 2 o0 = the
    # controllers' down products, o1 = their sources, o2 = their messages. Unused ones are tiny.
    return mx.fast.metal_kernel(
        name="mtplx_inkling_fast_read",
        input_names=["c0", "c1", "c2", "c3", "partial", "queries", "g0", "g1", "g2",
                     "mstates", "mqueries", "d0", "d1", "d2"],
        output_names=["o0", "o1", "o2"],
        header=_READ_HEADER,
        source=_READ,
    )


@lru_cache(maxsize=1)
def _mix_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_inkling_fast_mix",
        input_names=["state", "xs", "ms", "f0", "f1", "f2", "h0", "h1", "h2"],
        output_names=["hout", "parts"],
        source=_MIX,
    )


def _threads(width):
    return -(-(width // 4) // 32) * 32


def read(partial, closed, queries, eps, *, mode, gains=(), messages=None, message_queries=None,
         downs=(), msg_base=0, controllers=0, q_base=0, f32_stand_in=None):
    """One launch of the read kernel for a panel.

    ``partial``: ``(W, D)`` bf16 or f32; ``closed``: up to four ``(W, D)`` bf16 blocks;
    ``queries``: ``(Q, D)`` f32, or ``None`` for layer 0 (no depth read: the partial itself is
    normed; then ``f32_stand_in``, any float32 array, fills the absent query buffer).
    ``mode`` 0: input norms with ``gains`` -> ``(W, Q, D)`` bf16; 1: the read itself ->
    ``(W, 1, D)`` f32; 2: ``controllers`` controllers (down weights ``downs``, message states
    ``messages`` ``(W, P, D)`` f32 and queries ``message_queries`` from row ``msg_base``) ->
    ``state (W, C, 64)``, ``xs``, ``ms`` ``(W, C, D)`` f32.

    Absent operands are filled with existing arrays of the same dtype (never read, no launch).
    """
    noread = queries is None
    rows, width = partial.shape[-2], partial.shape[-1]
    partial = mx.contiguous(partial.reshape(rows, width))
    closed = [mx.contiguous(c.reshape(rows, width)) for c in closed]
    if len(closed) > 4:
        raise ValueError("the read kernel takes at most four closed blocks")
    if width % 128 or width > 4096:
        raise ValueError("the read kernel serves widths that are multiples of 128 up to 4,096")
    if noread:
        if f32_stand_in is None:
            raise ValueError("layer 0 needs a float32 stand-in for the absent query buffer")
        queries = f32_stand_in
    sources = len(closed) + 1
    nq = 1 if noread else (queries.shape[0] if queries.ndim == 2 else 1)
    queries = queries.reshape(-1)[: nq * width].reshape(nq, width) if noread else queries.reshape(nq, width)
    bf16 = [a for a in (*gains, *downs, partial, *closed) if a.dtype == mx.bfloat16]
    stand_bf16 = bf16[0]
    stand_block = partial if partial.dtype == mx.bfloat16 else closed[0] if closed else partial
    pad = closed + [stand_block] * (4 - len(closed))
    g = list(gains) + [stand_bf16] * (3 - len(gains))
    if mode == 2:
        p_states = messages.shape[-2]
        mstates = mx.contiguous(messages.reshape(rows, p_states, width))
        mq = message_queries
        d = list(downs) + [downs[-1]] * (3 - len(downs))
        chunks = 64 // _DOWN_ROWS
        groups = rows * controllers * chunks
        shapes = [(rows, controllers, 64), (rows, controllers, width), (rows, controllers, width)]
        dtypes = [mx.float32] * 3
    else:
        p_states, mstates, mq, d = 1, queries, queries, [stand_bf16] * 3
        chunks, groups = 1, rows
        shapes = [(rows, nq, width) if mode == 0 else (rows, 1, width), (1,), (1,)]
        dtypes = [mx.bfloat16 if mode == 0 else mx.float32, mx.float32, mx.float32]
    tpg = _threads(width)
    outs = _read_kernel()(
        inputs=[*pad, partial, queries, *g, mstates, mq, *d],
        template=[("MODE", mode), ("D", width), ("S", sources), ("NQ", nq), ("TPG", tpg),
                  ("NOREAD", bool(noread)), ("OUT0", dtypes[0]),
                  ("NC", max(controllers, 1)), ("CHUNKS", chunks), ("DROWS", _DOWN_ROWS),
                  ("P", p_states), ("MSG_BASE", msg_base), ("Q_BASE", q_base),
                  ("EPS", _f32_bits(eps)), ("SCALE", _f32_bits(width ** -0.5))],
        grid=(groups * tpg, 1, 1),
        threadgroup=(tpg, 1, 1),
        output_shapes=shapes,
        output_dtypes=dtypes,
    )
    return outs if mode == 2 else outs[:1]


def mix(state, xs, ms, forgets, gains, *, parts):
    """Controller forget/gain products and the tanh mix for every source and panel row."""
    rows, controllers, width = xs.shape
    f = list(forgets) + [forgets[-1]] * (3 - len(forgets))
    h = list(gains) + [gains[-1]] * (3 - len(gains))
    shapes = [(rows, controllers, width), (rows, controllers, width // 128) if parts else (1,)]
    dtypes = [mx.float32, mx.float32]
    outs = _mix_kernel()(
        inputs=[state, xs, ms, *f, *h],
        template=[("W", rows), ("NC", controllers), ("D", width), ("PARTS_OUT", bool(parts))],
        grid=(controllers * (width // _MIX_COLS) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=shapes,
        output_dtypes=dtypes,
    )
    return outs if parts else outs[:1]


def mlp_source(layer, partial, closed, messages=None, message_queries=None):
    """RC6's ``ffn`` (``model.py`` 524-532): MLP depth read, then the MLP controller, before the norm.

    ``partial`` ``(1, W, D)`` f32 from stage B/C; ``closed`` including this layer's boundary
    append. Returns ``(1, W, D)`` float32. One launch on the first pass, two on passes 2 and 3.
    """
    rows, width = partial.shape[-2], partial.shape[-1]
    if messages is None:
        out = read(partial, closed, layer.mlp_query, layer.eps, mode=1)
        return out[0].reshape(1, rows, width)
    ctrl = layer.controllers[3]
    state, xs, ms = read(partial, closed, layer.mlp_query, layer.eps, mode=2,
                         messages=messages, message_queries=message_queries,
                         downs=[ctrl.down], msg_base=3, controllers=1)
    (h,) = mix(state, xs, ms, [ctrl.forget], [ctrl.gain], parts=False)
    return h.reshape(1, rows, width)


def stage_expert_input(layer, partial, closed, message, panel):
    """Drop-in for ``fast.step.Stages.expert_input`` (L0's frame): RC6's ``ffn`` before the MLP norm."""
    return mlp_source(layer, partial, closed,
                      None if message is None else message.stack,
                      None if message is None else message.queries)
