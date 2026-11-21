"""The depth read of an Inkling 2 Mini layer visit, its cross-pass controllers and norms.

A visit reads its input twice from the closed residual blocks and the running partial: once
with the layer's attention query (one query, three on global layers) and once with its MLP
query. On loop passes two and three each source then goes through a low-rank controller
driven by the visit's cross-pass message. RC6 writes this as long chains of small MLX
operations (``model.py`` ``read_blocks``, ``serial_depth_dots``, ``Controller``, ``norm`` and
``projection.py`` ``stock_messages``): about 25 launches a read on the first pass and about
70 more on passes two and three.

Three kernels here:

* ``read``: one threadgroup per panel row (and, with controllers, per source and slice of
  the controller's down product). The one-query depth read, the controller norms, a slice of
  the down product, the input norm(s), or the keys of a three-query read.
* ``probs``: RC6's scale and softmax between the two products of a three- or four-query read.
* ``mix``: the controllers' forget and gain products, the tanh mix, and the per-128 partial
  sums the following input norm needs; one threadgroup per source and 128-column slice,
  serving every panel row.

Exactness (class A against RC6 on every Mac). Each kernel step repeats MLX 0.32.2's own kernel
for that step, thread for thread: ``rms_single_row`` (4 values a thread fused-multiply-added
from zero, ``simd_sum``, then ``simd_sum`` over the zero-padded simdgroup totals,
``precise::rsqrt``), the one-query ``gemv`` (``bn8``: eight slices of 1,024-column blocks with a
zero-padded tail, a tree per slice, slices added in order), its softmax for rows of up to 128
values (``share_four``: ``fast::exp``, one reciprocal of the sum), ``gemv_t`` (``bn16 sm8 sn4``:
sequential over sources, then three shuffle steps), the controller's ``gemv`` (down: ``bn8`` over
4,608; forget/gain: ``sm8 sn4`` over 64) and
``precise::tanh``. RC6's three-query depth read (global visits) and four-query message read
multiply on MLX's matrix units, whose summation order is the hardware's and differs between
Macs. On a Mac with matrix units they run in one launch (``exact_read``): the scores, RC6's
softmax and the mix, the two products as the installed MLX's own product code compiled into the
launch (``library_gemm``: its split-K tile loop or mat-vec, and its tile loop for the mix),
called as MLX's launches for those shapes call them, proven on each shape's first use against
the library's launches. Elsewhere (or refused) they run as the very MLX calls RC6 makes, on the
same shapes and types (scores one row at a time, as RC6's verify and one-row step do; the depth
read's mix batched over the panel and the message mix per row, as RC6 calls them).
"""

from __future__ import annotations

from functools import lru_cache
import math
import os
import struct
import weakref

import mlx.core as mx

#: Down-product rows one controller threadgroup computes (two blocks of MLX's four).
_DOWN_ROWS = 8
#: Columns of the controller output one ``mix`` threadgroup serves.
_MIX_COLS = 128


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


_SHARE = """
// Softmax over a row of up to 128 scores held four to a lane across one simdgroup (slots past the
// row's end hold -inf), with MLX's arithmetic for rows that short: the largest score by comparison
// from -FLT_MAX, then the simdgroup's largest; each slot's weight fast::exp(score - top), added up
// in slot order and summed across the simdgroup; every weight times one reciprocal of that sum.
// The weights replace the scores.
inline void share_four(thread float (&slot)[4]) {
    float top = -FLT_MAX;
    for (uint k = 0; k < 4; ++k) top = slot[k] > top ? slot[k] : top;
    top = simd_max(top);
    float mass = 0.0f;
    for (uint k = 0; k < 4; ++k) {
        slot[k] = metal::fast::exp(slot[k] - top);
        mass = mass + slot[k];
    }
    const float inv = 1.0f / simd_sum(mass);
    for (uint k = 0; k < 4; ++k) slot[k] = slot[k] * inv;
}
"""

_READ_HEADER = _SHARE + """
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
    // The in-kernel read is one query (window and MLP reads); mode 0 also norms NQ sources.
    const uint q_first = 0;
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
    } else if (READ_IN) {
        // Global visits on passes 2 and 3: RC6's three-query read, computed by MLX's own products
        // outside this launch; controller ctrl takes source ctrl.
        for (uint i = 0; i < 4; ++i) out[0][i] = in_row ? rin[(size_t(r) * 3 + ctrl) * D + c4 + i] : 0.0f;
    } else {
    // ---- 1. MLX rms_single_row over each source (closed blocks, then the partial): its inverse RMS.
    for (uint s = 0; s < S; ++s) {
        float acc = 0;
        for (uint i = 0; i < 4; ++i) {
            const float xval = in_row ? SOURCE(s, r, c4 + i) : 0.0f;
            acc += xval * xval;
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

    if (MODE == 3) {
        // The keys of a three-query read: mx.fast.rms_norm of each source without a weight
        // (RC6 read_blocks), float32, (W, S, D) as RC6 stacks them.
        if (in_row)
            for (uint s = 0; s < S; ++s)
                for (uint i = 0; i < 4; ++i)
                    o0[(size_t(r) * S + s) * D + c4 + i] = static_cast<OUT0>(1.0f * (SOURCE(s, r, c4 + i) * inv[s]));
        return;
    }

    // ---- 2. One-query gemv (bn8) of each query with each normalized source: q . (x * inv).
    if (sg < 8) {
        for (uint qi = q_first; qi < q_first + q_count; ++qi) {
            for (uint s = 0; s < S; ++s) {
                const float is = inv[s];
                float p = 0.0f;
                for (uint b = 0; b < FULL; ++b) {
                    for (uint co = 0; co < 4; ++co) {
                        const uint c = b * 1024 + sg * 128 + lane * 4 + co;
                        const float key = 1.0f * (SOURCE(s, r, c) * is);
                        p += key * queries[qi * D + c];
                    }
                }
                if (LEFT > 0) {
                    const uint c0_ = FULL * 1024 + sg * 128 + lane * 4;
                    for (uint co = 0; co < 4; ++co) {
                        const uint c = c0_ + co;
                        const float key = (c < D) ? 1.0f * (SOURCE(s, r, c) * is) : 0.0f;
                        const float qv = (c < D) ? queries[qi * D + c] : 0.0f;
                        p += key * qv;
                    }
                }
                for (ushort hop = 16; hop != 0; hop /= 2) p += simd_shuffle_down(p, hop);
                if (lane == 0) dots[(qi - q_first) * S + s][sg] = p;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- 3. Scale and softmax over the sources (share_four), one simdgroup per query.
    if (sg < q_count) {
        const uint qn = sg;
        float slot[4];
        for (uint i = 0; i < 4; ++i) {
            const uint s = lane * 4 + i;
            if (s < S) {
                float score = dots[qn * S + s][0];
                for (uint v = 1; v < 8; ++v) score += dots[qn * S + s][v];
                slot[i] = score * as_type<float>(uint(SCALE));
            } else {
                slot[i] = -INFINITY;
            }
        }
        share_four(slot);
        for (uint i = 0; i < 4; ++i) {
            const uint s = lane * 4 + i;
            if (s < S) prob[qn][s] = slot[i];
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
    // 5a. Its cross-pass message: RC6's four-query message read of this row (MLX's own products,
    // computed outside this launch), query MSG_BASE + ctrl. MSEP: one (1, 1, 4, D) array a row.
    const device float* mrow = MSEP
        ? (r == 0 ? m0 : r == 1 ? m1 : r == 2 ? m2 : r == 3 ? m3 : r == 4 ? m4 : r == 5 ? m5 : r == 6 ? m6 : m7)
        : m0 + size_t(r) * 4 * D;
    const uint mq = MSG_BASE + ctrl;
    float msg[4];
    for (uint i = 0; i < 4; ++i) msg[i] = in_row ? mrow[size_t(mq) * D + c4 + i] : 0.0f;

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
            const float sum_x = simd_sum(sums[S][lane]);
            const float sum_m = simd_sum(sums[S + 1][lane]);
            if (lane == 0) {
                inv[S] = metal::precise::rsqrt(sum_x / float(D) + as_type<float>(uint(EPS)));
                inv[S + 1] = metal::precise::rsqrt(sum_m / float(D) + as_type<float>(uint(EPS)));
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
            for (uint co = 0; co < 4; ++co) xv[co] = static_cast<float>(mixed[c + co]);
            for (uint ro = 0; ro < 4; ++ro)
                for (uint co = 0; co < 4; ++co)
                    p[ro] += static_cast<float>(dw[size_t(row0 + ro) * K2 + c + co]) * xv[co];
        }
        if (LEFT2 > 0) {
            const uint c = FULL2 * 1024 + v * 128 + lane * 4;
            float xv[4];
            for (uint co = 0; co < 4; ++co) xv[co] = (c + co < K2) ? static_cast<float>(mixed[c + co]) : 0.0f;
            for (uint ro = 0; ro < 4; ++ro)
                for (uint co = 0; co < 4; ++co) {
                    const float w = (c + co < K2) ? static_cast<float>(dw[size_t(row0 + ro) * K2 + c + co]) : 0.0f;
                    p[ro] += w * xv[co];
                }
        }
        for (uint ro = 0; ro < 4; ++ro)
            for (ushort hop = 16; hop != 0; hop /= 2) p[ro] += simd_shuffle_down(p[ro], hop);
        if (lane == 0)
            for (uint ro = 0; ro < 4; ++ro) down_part[blk][v][ro] = p[ro];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < DROWS) {
        const uint blk = t / 4, ro = t % 4;
        float total = down_part[blk][0][ro];
        for (uint v = 1; v < 8; ++v) total += down_part[blk][v][ro];
        o0[(size_t(r) * NC + ctrl) * 64 + chunk * DROWS + t] =
            static_cast<OUT0>(static_cast<float>(static_cast<bfloat16_t>(total)));
    }
"""

_PROBS = """
    // One simdgroup per (panel row, query): RC6's multiply by the scale (rounded on its own), then
    // the softmax over the S scores (share_four; MLX takes one 32-thread threadgroup for S <= 128).
    // A score array under 8 values arrives in the constant address space and a larger one in device
    // memory, so rows are read by value, never through one pointer.
    const uint lane = thread_index_in_simdgroup;
    const uint g = threadgroup_position_in_grid.x;
    const uint r = g / NQ, qn = g % NQ;
    const uint which = SEP ? r : 0;
    float slot[4];
    for (uint i = 0; i < 4; ++i) {
        const uint s = lane * 4 + i;
        if (s < S) {
            // SEP: one (1, 1, Q, S) array a row; JOINED: one (Q, W * S) array (rows' keys as columns);
            // otherwise one (1, W, Q, S) array.
            const uint at = SEP ? qn * S + s : (JOINED ? qn * (W * S) + r * S + s : (r * NQ + qn) * S + s);
            float v;
            switch (which) {
                case 0: v = s0[at]; break;
                case 1: v = s1[at]; break;
                case 2: v = s2[at]; break;
                case 3: v = s3[at]; break;
                case 4: v = s4[at]; break;
                case 5: v = s5[at]; break;
                case 6: v = s6[at]; break;
                default: v = s7[at]; break;
            }
            slot[i] = v * as_type<float>(uint(SCALE));
        } else {
            slot[i] = -INFINITY;
        }
    }
    share_four(slot);
    for (uint i = 0; i < 4; ++i) {
        const uint s = lane * 4 + i;
        if (s < S) p[(size_t(r) * NQ + qn) * S + s] = slot[i];
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

    const uint lane_row = lane / 4, lane_col = lane % 4;
    const bool gain_rows = sg >= 4;
    const uint row0 = col0 + (sg % 4) * 32 + lane_row * 4;
    const device float* fw32 = ctrl == 0 ? f0 : (ctrl == 1 ? f1 : f2);
    const device float* gw32 = ctrl == 0 ? h0 : (ctrl == 1 ? h1 : h2);
    const device float* mat = gain_rows ? gw32 : fw32;

    for (uint rc = 0; rc < W; rc += RC) {
        // MLX gemv sm8 sn4 (K = 64): a simdgroup covers 32 rows, lane (lane_row, lane_col) rows 4 lane_row..+3,
        // columns 16 i + 4 lane_col + co; then shuffles by 2 and 1 across lane_col.
        float acc[4][RC];
        for (uint ro = 0; ro < 4; ++ro)
            for (uint rr = 0; rr < RC; ++rr) acc[ro][rr] = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            const uint c = 16 * i + 4 * lane_col;
            float wt[4][4];
            for (uint ro = 0; ro < 4; ++ro)
                for (uint co = 0; co < 4; ++co) wt[ro][co] = mat[size_t(row0 + ro) * 64 + c + co];
            for (uint rr = 0; rr < RC; ++rr) {
                if (rc + rr < W) {
                    float sv[4];
                    for (uint co = 0; co < 4; ++co) sv[co] = state[(size_t(rc + rr) * NC + ctrl) * 64 + c + co];
                    for (uint ro = 0; ro < 4; ++ro)
                        for (uint co = 0; co < 4; ++co) acc[ro][rr] += wt[ro][co] * sv[co];
                }
            }
        }
        for (uint ro = 0; ro < 4; ++ro)
            for (uint rr = 0; rr < RC; ++rr)
                for (ushort hop = 2; hop != 0; hop /= 2) acc[ro][rr] += simd_shuffle_down(acc[ro][rr], hop);
        if (lane_col == 0)
            for (uint ro = 0; ro < 4; ++ro)
                for (uint rr = 0; rr < RC; ++rr)
                    fg[gain_rows ? 1 : 0][rr][(sg % 4) * 32 + lane_row * 4 + ro] = acc[ro][rr];
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
    # controllers' down products, o1 = their sources, o2 = their messages; mode 3 o0 = the keys
    # of a three-query read (f32). Unused ones are tiny.
    return mx.fast.metal_kernel(
        name="mtplx_inkling_fast_read",
        input_names=["c0", "c1", "c2", "c3", "partial", "queries", "g0", "g1", "g2", "rin",
                     "m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "d0", "d1", "d2"],
        output_names=["o0", "o1", "o2"],
        header=_READ_HEADER,
        source=_READ,
    )


@lru_cache(maxsize=1)
def _probs_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_inkling_fast_probs",
        input_names=[f"s{i}" for i in range(8)],
        output_names=["p"],
        header=_SHARE,
        source=_PROBS,
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


def read(partial, closed, queries, eps, *, mode, gains=(), messages=None, read_in=None, downs=(),
         msg_base=0, controllers=0, f32_stand_in=None):
    """One launch of the read kernel for a panel.

    ``partial``: ``(W, D)`` bf16 or f32; ``closed``: up to four ``(W, D)`` bf16 blocks;
    ``queries``: ``(Q, D)`` f32, or ``None`` for layer 0 (no depth read: the partial itself is
    normed; then ``f32_stand_in``, any float32 array, fills the absent query buffer).
    ``mode`` 0: input norms with ``gains`` -> ``(W, Q, D)`` bf16; 1: the read itself ->
    ``(W, 1, D)`` f32; 2: ``controllers`` controllers (down weights ``downs``) on the panel's
    exact message rows ``messages`` (``message_rows``) from query ``msg_base``, their sources the
    one-query read or, with ``read_in`` ``(W, 3, D)`` f32, RC6's three-query read ->
    ``state (W, C, 64)``, ``xs``, ``ms`` ``(W, C, D)`` f32; 3: the keys of a three-query read ->
    ``(W, S, D)`` f32.

    Absent operands are filled with existing arrays of the same dtype (never read, no launch).
    """
    noread = queries is None
    rows, width = partial.shape[-2], partial.shape[-1]
    # No mx.contiguous here: the kernel copies only an input that is not row-contiguous, while
    # mx.contiguous also copies a row-contiguous one whose buffer is 16 KB larger than the array
    # (a slice of a larger state, or a recycled buffer): a hidden launch per input per visit.
    partial = partial.reshape(rows, width)
    closed = [c.reshape(rows, width) for c in closed]
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
    # The kernel rounds where RC6 rounds to the stream's dtype, and it rounds to bf16: a float16
    # stream (the FP16 sibling packs) would get the wrong bits, so it is refused, counted, and the
    # caller serves the reference (fast/step.py declines stage A and D by the activation dtype).
    if any(getattr(a, "dtype", mx.bfloat16) not in (mx.bfloat16, mx.float32) for a in (*gains, *downs, partial, *closed)):
        from .attn_input import count_route

        count_route("refused:read_stream_dtype")
        raise RuntimeError("the read kernel serves bf16 streams only")
    bf16 = [a for a in (*gains, *downs, partial, *closed) if a.dtype == mx.bfloat16]
    if not bf16:
        from .attn_input import count_route

        count_route("refused:read_stream_dtype")
        raise RuntimeError("the read kernel needs the bf16 stream's operands")
    stand_bf16 = bf16[0]
    stand_block = partial if partial.dtype == mx.bfloat16 else closed[0] if closed else partial
    pad = closed + [stand_block] * (4 - len(closed))
    g = list(gains) + [stand_bf16] * (3 - len(gains))
    rin = queries if read_in is None else read_in.reshape(rows, 3, width)
    msep, m = False, [queries] * 8
    if mode == 2:
        if isinstance(messages, (list, tuple)):
            if len(messages) > 8:
                messages = [mx.concatenate(list(messages), axis=1)]
            msep = len(messages) > 1
            m = list(messages) + [messages[0]] * (8 - len(messages))
        else:
            m = [messages] * 8
        d = list(downs) + [downs[-1]] * (3 - len(downs))
        chunks = 64 // _DOWN_ROWS
        groups = rows * controllers * chunks
        shapes = [(rows, controllers, 64), (rows, controllers, width), (rows, controllers, width)]
        dtypes = [mx.float32] * 3
    else:
        d = [stand_bf16] * 3
        chunks, groups = 1, rows
        out0 = {0: (rows, nq, width), 1: (rows, 1, width), 3: (rows, sources, width)}[mode]
        shapes = [out0, (1,), (1,)]
        dtypes = [mx.bfloat16 if mode == 0 else mx.float32, mx.float32, mx.float32]
    tpg = _threads(width)
    template = [("MODE", mode), ("D", width), ("S", sources), ("NQ", nq), ("TPG", tpg),
                ("NOREAD", bool(noread)), ("READ_IN", read_in is not None), ("MSEP", msep),
                ("OUT0", dtypes[0]), ("NC", max(controllers, 1)), ("CHUNKS", chunks),
                ("DROWS", _DOWN_ROWS), ("MSG_BASE", msg_base),
                ("EPS", _f32_bits(eps)), ("SCALE", _f32_bits(width ** -0.5))]
    inputs = [*pad, partial, queries, *g, rin, *m, *d]

    def launch(kernel):
        return kernel(inputs=inputs, template=template, grid=(groups * tpg, 1, 1), threadgroup=(tpg, 1, 1),
                      output_shapes=shapes, output_dtypes=dtypes)

    # Lane F1: the same arithmetic in fewer dependent steps where proven (read_lean.py).
    from .read_lean import launch_read

    key = (mode, sources, nq, rows, width, bool(noread), read_in is not None, msep, controllers, msg_base,
           str(partial.dtype), str(queries.dtype), str(dtypes[0]))
    outs = launch_read(key, launch, _read_kernel, mode)
    return outs if mode == 2 else outs[:1]


def probs(scores, scale, joined=None):
    """``mx.softmax(scores * scale, axis=-1)`` for each row and query, as RC6 computes it.

    ``scores``: per-row ``(1, 1, Q, S)`` float32 arrays, or one ``(1, W, Q, S)`` array; or
    ``joined = (array (Q, W * S), W, S)``, the stacked product's layout. Returns ``(1, W, Q, S)``
    float32. One launch (more than eight separate rows are joined first).
    """
    if joined is not None:
        array, rows, count = joined
        scores, nq, sep, flat = [array], array.shape[0], False, True
    else:
        scores = list(scores)
        if len(scores) > 8:
            scores = [mx.concatenate(scores, axis=1)]
        rows = sum(a.shape[1] for a in scores)
        nq, count = scores[0].shape[2], scores[0].shape[3]
        sep, flat = len(scores) > 1, False
    if count > 128:
        raise ValueError("probs serves at most 128 scores a row (one simdgroup)")
    return _probs_kernel()(
        inputs=scores + [scores[0]] * (8 - len(scores)),
        template=[("SEP", sep), ("JOINED", flat), ("W", rows), ("NQ", nq), ("S", count),
                  ("SCALE", _f32_bits(scale))],
        grid=(rows * nq * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, rows, nq, count)],
        output_dtypes=[mx.float32],
    )[0]


def depth_keys(partial, closed, queries, eps):
    """RC6 ``read_blocks``' keys, ``rms_norm(stack([*closed, partial], -2).astype(f32), None, eps)``:
    ``(1, W, S, D)`` float32 in one launch (``queries``, the read's ``(Q, D)`` float32 queries,
    only fill the kernel's query slot)."""
    rows, width = partial.shape[-2], partial.shape[-1]
    (keys,) = read(partial, closed, queries, eps, mode=3)
    return keys.reshape(1, rows, len(closed) + 1, width)


# First-use verdicts on this Mac, per shape: does one MLX product give RC6's per-row bits?
_one_launch: dict = {}


def _same(a, b):
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return a.shape == b.shape and bool(mx.array_equal(a.view(view), b.view(view)).item())


def _proven(key, per_row, joined, slices):
    """``slices(joined)`` must equal ``per_row`` bit for bit; decided once per shape on real
    operands (a host sync on that first call only) and counted."""
    from .attn_input import count_route

    verdict = _one_launch.get(key)
    if verdict is None:
        rows = per_row()
        whole = joined()
        mx.eval(rows, whole)
        verdict = all(_same(mx.contiguous(got), want) for got, want in zip(slices(whole), rows))
        _one_launch[key] = verdict
        count_route(("proved:" if verdict else "refused:") + "one_launch_" + "_".join(map(str, key)))
    return verdict


def scores(q, keys):
    """RC6's score products for every row (``q @ keys[:, r:r+1].swapaxes(-1, -2)``, one MLX call
    a row, as RC6 takes them): as one MLX product with the rows' keys as columns where a first-use
    proof on this Mac shows the same bits (an M3 to M5 measurement: always, for 3 and 4 queries,
    2 to 5 sources, 1 to 32 rows; receipts/probe-batch-1.json), else the per-row calls.
    Returns ``probs`` keyword arguments."""
    rows, count, width = keys.shape[1], keys.shape[2], keys.shape[3]
    nq = q.shape[0]
    per_row = lambda: [q @ keys[:, r:r + 1].swapaxes(-1, -2) for r in range(rows)]
    joined = lambda: q @ keys.reshape(rows * count, width).T
    key = ("scores", nq, count, rows, width)
    if _proven(key, per_row, joined,
               lambda whole: [whole[:, r * count:(r + 1) * count].reshape(1, 1, nq, count) for r in range(rows)]):
        return {"joined": (joined(), rows, count)}
    return {"scores": per_row()}


_ATTEND = """
    // softmax(q . keys^T * SCALE) @ keys for one panel row and one query, in this kernel's own
    // fixed order (the same on every Mac): per-thread float4 partial dots, simd_sum, the simdgroup
    // totals added in order; max-subtracted softmax in one simdgroup; the mix per column.
    constexpr uint NSG = TPG / 32;
    const uint g = threadgroup_position_in_grid.x;
    const uint r = g / NQ, qn = g % NQ;
    const uint t = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    threadgroup float red[NSG * S];
    threadgroup float p[S];
    const device float* krow = keys + size_t(r) * S * D;
    const uint c = t * 4;
    float part[S];
    for (uint s = 0; s < S; ++s) part[s] = 0.0f;
    if (c < D) {
        const float4 qv = *reinterpret_cast<const device float4*>(q + size_t(qn) * D + c);
        for (uint s = 0; s < S; ++s) {
            const float4 kv = *reinterpret_cast<const device float4*>(krow + s * D + c);
            part[s] = ((qv[0] * kv[0] + qv[1] * kv[1]) + qv[2] * kv[2]) + qv[3] * kv[3];
        }
    }
    for (uint s = 0; s < S; ++s) {
        const float v = simd_sum(part[s]);
        if (lane == 0) red[sg * S + s] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = -INFINITY;
        if (lane < S) {
            float total = 0.0f;
            for (uint k = 0; k < NSG; ++k) total += red[k * S + lane];
            v = total * as_type<float>(uint(SCALE));
        }
        const float m = simd_max(v);
        const float e = lane < S ? metal::precise::exp(v - m) : 0.0f;
        const float z = simd_sum(e);
        if (lane < S) p[lane] = e / z;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (c < D) {
        float4 acc = float4(0.0f);
        for (uint s = 0; s < S; ++s) acc += p[s] * *reinterpret_cast<const device float4*>(krow + s * D + c);
        *reinterpret_cast<device float4*>(out + (size_t(r) * NQ + qn) * D + c) = acc;
    }
"""


@lru_cache(maxsize=1)
def _attend_kernel():
    return mx.fast.metal_kernel(name="mtplx_inkling_fast_attend_rows", input_names=["q", "keys"],
                                output_names=["out"], source=_ATTEND)


def fixed_order():
    """``MTPLX_INKLING_DEPTH_READ=fixed``: the three- and four-query reads in one launch each, in this
    lane's own order on every Mac (class C against RC6, whose scores and mix go to MLX's matrix
    products: another order per GPU family). Off by default: the founder decides."""
    return os.environ.get("MTPLX_INKLING_DEPTH_READ", "exact") == "fixed"


def attend_rows(q, keys):
    """``softmax(q @ keys[r].T / sqrt(D)) @ keys[r]`` for every row ``r``, one launch, fixed order.
    ``q`` ``(Q, D)`` float32, ``keys`` ``(1, W, S, D)`` float32 -> ``(1, W, Q, D)`` float32."""
    from .attn_input import count_route

    nq, width = q.shape
    rows, count = keys.shape[1], keys.shape[2]
    if width % 4 or count > 32:
        raise ValueError("attend_rows serves widths that are multiples of 4 and up to 32 keys")
    tpg = _threads(width)
    count_route(f"fixed:attend_rows_q{nq}_s{count}")
    return _attend_kernel()(
        inputs=[q.astype(mx.float32), keys],
        template=[("NQ", nq), ("S", count), ("D", width), ("TPG", tpg), ("SCALE", _f32_bits(width ** -0.5))],
        grid=(rows * nq * tpg, 1, 1),
        threadgroup=(tpg, 1, 1),
        output_shapes=[(1, rows, nq, width)],
        output_dtypes=[mx.float32],
    )[0]


_EXACT = """
    // One threadgroup per (visit v, panel row r, column group g); a depth read is one visit (V = 1), a
    // loop pass's message reads are its V visits' queries against the same stack. The three products of a
    // read run here as
    // the library's own code (``fast.library_gemm``), each called the way the library's
    // launch for that product's shape calls it, so every value is the library's:
    //   scores  = queries . keys_r^T: the split-K product on the matrix units (each K partition one
    //             tile loop from zero; the partitions then added in order from zero, as the library's
    //             accumulate launch adds them), or, for a single key, the library's mat-vec;
    //   weights = RC6's multiply by the scale, then its softmax (share_four), as the probs kernel;
    //   mix     = weights . keys_r, for this group's 32-column chunks, one tile loop each.
    // Results pass between the steps through device scratch inside this threadgroup.
    using namespace mlx::steel;
    const uint grp = threadgroup_position_in_grid.x;
    const uint v = grp / (ROWS * G), r = (grp / G) % ROWS, g = grp % G;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const device float* kr = keys + size_t(r) * N * D;
    const device float* qs = queries + size_t(v) * Q * D;          // this visit's Q queries
    device float* sk = scratch + size_t(grp) * (PARTS + 1) * Q * N;   // PARTS partials, then weights
    device float* wt = sk + PARTS * Q * N;

    if constexpr (N == 1) {
        // The library's mat-vec, launched by it as one threadgroup of 8 x 32 threads (this one); VT
        // query rows a thread (4, or 1 for fewer than 4 queries).
        threadgroup float reduce[8 * (VT + VT)];
        using MatVec = GEMVKernel<float, 1, 8, 1, 32, VT, 4, false, float>;
        const constant int& k_len = width[0];       // the keys' width: the mat-vec's input length
        const constant int& q_len = nq[0];          // the queries: its output length
        const constant float& none = zero[0];       // no bias term (alpha, beta unused)
        const uint3 first_group = uint3(0, 0, 0);
        const uint3 thread_at = uint3(lane, sg, 0);
        MatVec::run(qs, kr, qs /* no bias */, sk, k_len, q_len, k_len /* row stride */, none, none,
                    q_len, &reduce[0], first_group, thread_at, uint(sg), uint(lane));
    } else {
        // Split-K tiles of 64 x 64 over simdgroups 0-3 (2 x 2, 32 x 32 each); simdgroups 4-7 hold a tile
        // past the last query, so they own no values but meet every barrier of the tile loop.
        const short row0 = sg < 4 ? short(32 * (sg >> 1)) : short(64);
        const short col0 = sg < 4 ? short(32 * (sg & 1)) : short(0);
        const short rows_here = short(min(32, Q - int(row0)));
        const short cols_here = short(min(32, N - int(col0)));
        for (int part = 0; part < PARTS; ++part) {
            const int kbeg = PART * part;
            const int klen = min(kbeg + PART, D) - kbeg;
            NAXTile<float, 2, 2> tile;
            dispatch_bool(klen % KB == 0, [&](auto whole_k) {
              dispatch_bool(rows_here == 32, [&](auto whole_m) {
                dispatch_bool(cols_here == 32, [&](auto whole_n) {
                  tile = gemm_loop<float, 32, 32, 32, KB, false, true, whole_m.value, whole_n.value,
                                   whole_k.value, float>(qs + kbeg + int(row0) * D, kr + kbeg + int(col0) * D,
                                                         D, D, klen, klen / KB, rows_here, cols_here);
                });
              });
            });
            if (rows_here > 0 && cols_here > 0)
                tile.store_safe(sk + part * Q * N + int(row0) * N + int(col0), N, short2(cols_here, rows_here));
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (sg < uint(Q)) {
        float slot[4];
        for (uint i = 0; i < 4; ++i) {
            const uint s = lane * 4 + i;
            float v = -INFINITY;
            if (s < uint(N)) {
                float total = 0.0f;                       // the accumulate launch starts from zero
                if constexpr (N == 1) {
                    total = sk[sg];
                } else {
                    for (int part = 0; part < PARTS; ++part) total += sk[part * Q * N + sg * N + s];
                }
                v = total * as_type<float>(uint(SCALE));
            }
            slot[i] = v;
        }
        share_four(slot);
        for (uint i = 0; i < 4; ++i)
            if (lane * 4 + i < uint(N)) wt[sg * N + lane * 4 + i] = slot[i];
    }
    threadgroup_barrier(mem_flags::mem_device);
    constexpr uint CHUNKS = D / 32;
    constexpr uint PER_GROUP = CHUNKS / G;
    for (uint c = g * PER_GROUP + sg; c < (g + 1) * PER_GROUP; c += 8) {
        const int col = int(c) * 32;
        NAXTile<float, 2, 2> tile = gemm_loop<float, 32, 32, 32, 256, false, false, false, true, false, float>(
            wt, kr + col, N, D, N, 0, short(Q), short(32));
        tile.store_safe(out + (size_t(v) * ROWS + r) * Q * D + col, D, short2(32, Q));
    }
"""

#: Threadgroups a panel row in the one-launch read (each recomputes the scores; the 72 mix chunks
#: of 32 columns are shared): 18.4 against 19.8 us a read at one (engine, d1/receipts/proto-read-t1).
_EXACT_GROUPS = 9
#: The one-launch read (MTPLX_INKLING_EXACT_READ=0 turns it off: the reads then take the library's
#: own launches, three or four a read, and the probs kernel).
_EXACT_ON = os.environ.get("MTPLX_INKLING_EXACT_READ", "1") != "0"
_exact_verdicts: dict = {}
_exact_scalars: dict = {}


@lru_cache(maxsize=1)
def _exact_kernel():
    from . import library_gemm

    text, why = library_gemm.library("matrix_units", "matvec")
    if text is None:
        return None, why
    return mx.fast.metal_kernel(name="mtplx_inkling_fast_exact_read",
                                input_names=["queries", "keys", "width", "nq", "zero"],
                                output_names=["out", "scratch"], header=text + _SHARE, source=_EXACT), None


def _exact_refusal(key, why, tag):
    from .attn_input import count_route

    _exact_verdicts[key] = False
    count_route(f"refused:exact_read_{tag}_" + "_".join(map(str, key)))
    if os.environ.get("MTPLX_INKLING_TRACE_REFUSALS"):
        print(f"[inkling] one-launch read {key} refused: {why}")


def exact_read(q, keys, reference, visits=1):
    """The depth read ``softmax(q . keys_r^T * D^-1/2) . keys_r`` of every panel row in one launch, bit
    for bit the read ``reference()`` gives (the library's products and RC6's softmax): proven on
    this Mac's first use of each shape (counted), else ``None`` and the caller keeps its launches.
    ``q`` ``(Q, D)`` float32, ``keys`` ``(1, W, N, D)`` float32 -> ``(1, W, Q, D)`` float32.

    ``visits`` > 1: ``q`` is ``(visits * Q, D)``, ``visits`` groups of ``Q`` queries each read the same
    ``keys`` (a loop pass's message reads: every visit's four queries against the pass's stack), in
    the same launch, -> ``(visits, W, Q, D)``; each visit's rows are the read of its own ``Q`` queries."""
    from . import library_gemm

    from .attn_input import count_route

    width = q.shape[1]
    nq = q.shape[0] // visits
    rows, count = keys.shape[1], keys.shape[2]
    key = (nq, count, rows, width) if visits == 1 else (nq, count, rows, width, visits)
    verdict = _exact_verdicts.get(key)
    if verdict is False or not _EXACT_ON:
        return None
    if library_gemm.gpu_generation() < 17:
        # The mix runs on the matrix units on every Mac this launch serves; without them the library's
        # products are its simdgroup-matrix forms, not twinned here yet.
        _exact_refusal(key, "no matrix units on this GPU", "gpu")
        return None
    tiles, why = library_gemm.product_tiles(nq, count, width)
    if tiles is None:
        _exact_refusal(key, why, "tiles")
        return None
    if (nq > 4 or count > 32 or width % (32 * _EXACT_GROUPS) or nq * visits != q.shape[0]
            or q.dtype != mx.float32 or keys.dtype != mx.float32):
        _exact_refusal(key, "shape or type outside the one-launch read", "shape")
        return None
    kernel, why = _exact_kernel()
    if kernel is None:
        _exact_refusal(key, why, "library")
        return None
    scalars = _exact_scalars.get((width, nq))
    if scalars is None:
        scalars = _exact_scalars[(width, nq)] = (mx.array([width], mx.int32), mx.array([nq], mx.int32),
                                                 mx.array([0.0], mx.float32))
    parts = 1 if tiles.kind == "matvec" else tiles.partitions
    groups = visits * rows * _EXACT_GROUPS
    out = kernel(
        inputs=[q, keys, *scalars],
        template=[("Q", nq), ("N", count), ("D", width), ("G", _EXACT_GROUPS), ("ROWS", rows), ("PARTS", parts),
                  ("PART", tiles.partition or width), ("KB", tiles.block_k or 256), ("VT", 4 if nq >= 4 else 1),
                  ("SCALE", _f32_bits(width ** -0.5))],
        grid=(groups * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(visits, rows, nq, width), (groups * (parts + 1) * nq * count,)],
        output_dtypes=[mx.float32, mx.float32],
    )[0]
    if verdict is None:
        try:
            want = reference()
            mx.eval(out, want)
            verdict = _same(out, want)
        except RuntimeError as exc:     # the library text did not build on this Mac
            _exact_refusal(key, exc, "build")
            return None
        _exact_verdicts[key] = verdict
        count_route(("proved:" if verdict else "refused:") + "exact_read_" + "_".join(map(str, key)))
        if not verdict:
            return None
    return out


def depth_read(queries, keys):
    """RC6's three-query depth read from its keys (``model.py`` ``serial_depth_dots``): the
    scores as RC6's per-row products give them (``scores``), RC6's scale and softmax here, the
    mix as RC6's batched MLX product. ``(1, W, Q, D)`` float32. With ``fixed_order()``, one launch
    in this lane's own order instead (``attend_rows``)."""
    width = keys.shape[-1]
    q = queries.reshape(-1, width).astype(mx.float32)
    if fixed_order():
        return attend_rows(q, keys)

    def library():
        kw = scores(q, keys)
        return probs(kw.get("scores"), width ** -0.5, kw.get("joined")) @ keys

    # The one-launch read built from the library's product code (exact_read), else its launches.
    # (Lane F1's other one-launch read was timed against it in one job per pack, f1/receipts/c2-*,
    # and removed: about 60 against 40 us a launch.)
    one = exact_read(q, keys, library)
    return one if one is not None else library()


def message_rows(stack, queries):
    """RC6's cross-pass message read (``projection.py`` ``serial_messages``): scores as RC6's
    per-row products give them (``scores``), RC6's scale and softmax here, then the mix: one
    batched MLX product where a first-use proof shows RC6's per-row bits (always on the M5 Max,
    receipts/probe-rowstable-1.json), else one product a row as RC6 calls it.

    ``stack`` ``(1, W, P, D)`` float32 (normalised prior states), ``queries`` ``(4, D)`` float32.
    Returns one ``(1, W, 4, D)`` float32 array, or one ``(1, 1, 4, D)`` array a row.
    """
    rows, count, width = stack.shape[1], stack.shape[2], stack.shape[-1]
    if fixed_order():
        return attend_rows(queries, stack)

    def library_launches():
        kw = scores(queries, stack)
        weights = probs(kw.get("scores"), width ** -0.5, kw.get("joined"))
        per_row = lambda: [weights[:, r:r + 1] @ stack[:, r:r + 1] for r in range(rows)]
        batched = lambda: weights @ stack
        key = ("mix", queries.shape[0], count, rows, width)
        if _proven(key, per_row, batched, lambda whole: [whole[:, r:r + 1] for r in range(rows)]):
            return batched()
        return per_row()

    def joined():
        got = library_launches()
        return got if not isinstance(got, list) else mx.concatenate(got, axis=1)

    one = exact_read(queries.astype(mx.float32), stack, joined)
    return one if one is not None else library_launches()


_message_rows = weakref.WeakKeyDictionary()


def message_rows_of(message):
    """``message_rows`` of L0's per-visit ``Message`` (``fast/step.py``), computed once and shared
    by the visit's attention input (controllers 0-2) and expert input (controller 3)."""
    rows = _message_rows.get(message)
    if rows is None:
        rows = message_rows(message.stack, message.queries)
        _message_rows[message] = rows
    return rows


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


def mlp_source(layer, partial, closed, messages=None, message_queries=None, *, rows=None):
    """RC6's ``ffn`` (``model.py`` 524-532): MLP depth read, then the MLP controller, before the norm.

    ``partial`` ``(1, W, D)`` f32 from stage B/C; ``closed`` including this layer's boundary
    append; on passes 2 and 3 the message stack and queries, or the visit's exact message
    ``rows`` (``message_rows``) when stage A already read them. Returns ``(1, W, D)`` float32.
    One launch on the first pass, two on passes 2 and 3.
    """
    from .attn_input import count_route

    if any(c.dtype != mx.bfloat16 for c in closed) or partial.dtype not in (mx.bfloat16, mx.float32):
        # RC6's read and controller round to the stream's dtype; these kernels round to bf16.
        count_route("refused:mlp_stream_dtype")
        raise RuntimeError("the fused MLP read serves bf16 streams only")
    width = partial.shape[-1]
    count = partial.shape[-2]
    loop = messages is not None or rows is not None
    count_route("fused:mlp_" + ("loop" if loop else "first"))
    if not loop:
        out = read(partial, closed, layer.mlp_query, layer.eps, mode=1)
        return out[0].reshape(1, count, width)
    if rows is None:
        rows = message_rows(messages, message_queries)
    ctrl = layer.controllers[3]
    state, xs, ms = read(partial, closed, layer.mlp_query, layer.eps, mode=2, messages=rows,
                         downs=[ctrl.down], msg_base=3, controllers=1)
    (h,) = mix(state, xs, ms, [ctrl.forget], [ctrl.gain], parts=False)
    return h.reshape(1, count, width)


def stage_expert_input(layer, partial, closed, message, panel):
    """Drop-in for ``fast.step.Stages.expert_input`` (L0's frame): RC6's ``ffn`` before the MLP norm."""
    return mlp_source(layer, partial, closed, rows=None if message is None else message_rows_of(message))
