"""Stage A of an Inkling 2 Mini layer visit: everything between the visit input and attention.

RC6 spends about 55 Metal launches here on a first-pass visit and about 100 on loop passes
two and three (`model.py` 487-510 and 441-466): the depth read, the cross-pass controllers,
the input norms, six or eight separate projections, the per-head clips, the value norm,
the position scale, both short convolutions and the cache update. This module runs the
same arithmetic in these launches (W panel rows):

1. ``read`` (``fast/read.py``): one-query depth read, controller norms and down product, input
   norms; on global visits the keys of the three-query read.
2. RC6's three-query depth read (global visits, ``W + 2`` launches: per-row scores and the mix
   on MLX's own products, softmax between) and four-query message read (passes 2 and 3,
   ``2 W + 1``: per-row scores and mixes, softmax between, shared with the expert input):
   their order is the matrix unit's, so they run as the very MLX calls RC6 makes.
3. passes two and three only: controller forget/gain products and the tanh mix.
4. ``project`` (here): every projection of the visit for the whole panel in one launch,
   each stored bf16 weight row read once per panel and in place, followed by the per-head
   epilogues: scaled clips of query and relative, the position scale, both convolutions
   with their committed tails, the key clip and the value norm.

Launches a visit: window first pass 2; global first pass ``W + 4``; window loop pass ``2 W + 4``;
global loop pass ``3 W + 7``.

Exactness (class A, proven on first use): every output row of a product is a twin of
MLX 0.32.2's one-row ``gemv`` for that matrix shape (``matmul.cpp`` ``gemv_axbpy``):

* query, relative, gate, key, value: ``bm4 bn1 sm1 sn32 tm4 tn4``: lane ``l`` sums columns
  ``128 i + 4 l .. + 3`` for ``i = 0 .. 17`` in order, then the shuffle-down tree 16..1;
* sink and the dynamic taps (``K >= 16 N``): ``bm1 bn8``: eight slices of 1,024-column blocks
  plus a zero-padded tail, a tree per slice, then the eight slice totals in order.

A panel row's arithmetic never depends on the panel width, so every row of a W-row panel
gets the bits of a one-row step (rule 2). Products of two bf16 values are exact in float32,
so the compiler's choice between a fused and a separate multiply-add cannot move a bit.
Epilogues repeat RC6's statements one by one, each rounded where RC6 rounds:
``clip_heads`` (square, the library's row-reduce order, precise sqrt, ``+ 1e-6``, divide,
the library's ``minimum``, multiply, cast), ``short_conv``'s proven forms, the value norm as
``rms_single_row`` over 128, and the position scale with precise ``log``.

No kernel writes state (``lanes/SIGNATURES.md``): the committed convolution tails are read
only; the panel's key/value rows and convolution inputs are returned for L0 to commit.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import math
import struct
from threading import Lock
from typing import NamedTuple

import mlx.core as mx

_routes: Counter = Counter()
_lock = Lock()


def count_route(name):
    with _lock:
        _routes[name] += 1


def route_counts():
    """Stage-A and MLP-source selections since the last reset: ``fused:<kind>`` or ``refused:<why>``."""
    with _lock:
        return dict(_routes)


def reset_route_counts():
    with _lock:
        _routes.clear()

#: Threads of one projection threadgroup (eight simdgroups).
_THREADS = 256
#: Gate rows a gate threadgroup projects.
_GATE_ROWS = 32
#: Panel rows one simdgroup accumulates at once (wider panels loop, re-reading cached weights).
_ROW_CHUNK = 8
#: Stage each threadgroup's input in threadgroup memory (whole panel when it fits, else in chunks).
_STAGE = True


def f32_bits(value: float) -> int:
    """The float32 MLX rounds a Python float to, as raw bits for a template constant."""
    return struct.unpack("<I", struct.pack("<f", value))[0]


_HEADER = """
// Four consecutive bf16 as float from one 8-byte load (float(bf16) is the bf16 bits << 16).
#define LOAD_BF16X4(ptr, out)                                                           \\
    {                                                                                   \\
        const uint2 b_ = *reinterpret_cast<const device uint2*>(ptr);                   \\
        out[0] = as_type<float>(b_.x << 16);                                            \\
        out[1] = as_type<float>(b_.x & 0xffff0000u);                                    \\
        out[2] = as_type<float>(b_.y << 16);                                            \\
        out[3] = as_type<float>(b_.y & 0xffff0000u);                                    \\
    }

// Panel row r, columns col..col+3 of this threadgroup's input source, as float.
// NORMED: the bf16 normed input. RAW: the norm applied here, as MLX's rms_single_row writes it
// (gv holds the four gains of these columns).
#define FETCH4(r, col, xv, gv)                                                          \\
    if (STAGE) {                                                                        \\
        for (uint t_ = 0; t_ < 4; ++t_) xv[t_] = static_cast<float>(stage_in[(r) * D + (col) + t_]); \\
    } else if (RAW) {                                                                   \\
        const size_t base_ = (size_t(r) * NS + src) * D + (col);                        \\
        const float inv_ = inv_rows[r];                                                 \\
        const float4 h_ = *reinterpret_cast<const device float4*>(hraw + base_);        \\
        for (uint t_ = 0; t_ < 4; ++t_) {                                               \\
            const float y_ = gv[t_] * (h_[t_] * inv_);                                  \\
            xv[t_] = static_cast<float>(static_cast<bfloat16_t>(y_));                   \\
        }                                                                               \\
    } else {                                                                            \\
        LOAD_BF16X4(normed + (size_t(r) * NS + src) * D + (col), xv);                   \\
    }

// Four consecutive columns of panel row r of this threadgroup's source, staged as bf16 at dst.
#define STAGE4(r, c, dst)                                                               \\
    if (RAW) {                                                                          \\
        const device bfloat16_t* gs_ = (src == 0 ? g0 : (src == 1 ? g1 : g2)) + (c);    \\
        const float4 h4_ = *reinterpret_cast<const device float4*>(                     \\
            hraw + (size_t(r) * NS + src) * D + (c));                                   \\
        const float iv_ = inv_rows[r];                                                  \\
        for (uint j_ = 0; j_ < 4; ++j_)                                                 \\
            (dst)[j_] = static_cast<bfloat16_t>(static_cast<float>(gs_[j_]) * (h4_[j_] * iv_)); \\
    } else {                                                                            \\
        *reinterpret_cast<threadgroup uint2*>(dst) =                                    \\
            *reinterpret_cast<const device uint2*>(normed + (size_t(r) * NS + src) * D + (c)); \\
    }

#define GAINS4(col, gv)                                                                 \\
    if (RAW && !STAGE) {                                                                \\
        const device bfloat16_t* gp_ = (src == 0 ? g0 : (src == 1 ? g1 : g2)) + (col);  \\
        for (uint t_ = 0; t_ < 4; ++t_) gv[t_] = static_cast<float>(gp_[t_]);          \\
    }
"""

_PROJECT = """
    const uint tg = threadgroup_position_in_grid.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    constexpr uint NSG = THREADS / 32;
    constexpr uint STEPS = D / 128;
    constexpr uint FULL = D / 1024;            // bn8: whole 1,024-column blocks
    constexpr uint OQ = HD + RD + 1;           // per panel row: q head, r head, sink
    constexpr uint OK = HD + TAPS;             // per panel row: k (or v) head, its taps
    constexpr uint OUTS = (OQ > OK ? OQ : OK) * W;

    threadgroup float outs[OUTS];
    threadgroup float inv_rows[W];
    // Staged inputs, declared as uint2 so their 8-byte stores are aligned; read as bf16.
    threadgroup uint2 stage_u2[STAGE ? W * D / 4 : 1];
    threadgroup uint2 chunk_u2[STAGE2 ? RCH * KC / 4 : 1];
    threadgroup bfloat16_t* stage_in = reinterpret_cast<threadgroup bfloat16_t*>(stage_u2);
    threadgroup bfloat16_t* chunk_in = reinterpret_cast<threadgroup bfloat16_t*>(chunk_u2);

    // Units, heaviest first: query heads (with their relative head and sink row), key heads,
    // value heads, then gate slices.
    uint kind, h;
    if (tg < QH) { kind = 0; h = tg; }
    else if (tg < QH + KH) { kind = 1; h = tg - QH; }
    else if (tg < QH + 2 * KH) { kind = 2; h = tg - QH - KH; }
    else { kind = 3; h = tg - QH - 2 * KH; }
    const uint src = (NS == 3) ? (kind == 1 ? 1u : (kind == 2 ? 2u : 0u)) : 0u;

    if (RAW) {
        // MLX rms_single_row, second level: simd_sum of the per-128 partials, zero padded. PARTS_IN:
        // the partials come with the input; otherwise the first level runs here, one simdgroup a
        // row: 4 values a lane fused-multiply-added from zero, simd_sum per 128 columns.
        for (uint r = sg; r < W; r += NSG) {
            float part = 0.0f;
            if (PARTS_IN) {
                part = lane < (D / 128) ? parts[(r * NS + src) * (D / 128) + lane] : 0.0f;
            } else {
                for (uint j = 0; j < D / 128; ++j) {
                    const float4 h4 = *reinterpret_cast<const device float4*>(
                        hraw + (size_t(r) * NS + src) * D + j * 128 + lane * 4);
                    float acc = 0;
                    for (uint i = 0; i < 4; ++i) acc += h4[i] * h4[i];
                    acc = simd_sum(acc);
                    if (lane == j) part = acc;
                }
            }
            const float total = simd_sum(part);
            if (lane == 0) {
                inv_rows[r] = metal::precise::rsqrt(total / float(D) + as_type<float>(uint(NORM_EPS)));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (STAGE) {
        // The whole panel's input in threadgroup memory: every simdgroup then reads it from there.
        for (uint idx = thread_index_in_threadgroup; idx < W * (D / 4); idx += THREADS) {
            const uint rr = idx / (D / 4), c = (idx % (D / 4)) * 4;
            STAGE4(rr, c, &stage_in[rr * D + c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ------------------------------------------------------------------ products
    const device bfloat16_t* mat = wq;
    uint nblocks = 0;
    if (kind == 0) nblocks = HD / 4 + RD / 4;
    else if (kind == 3) nblocks = GROWS / 4;
    else nblocks = HD / 4;

    if (STAGE2) {
        // Panels too wide to stage whole: all simdgroups advance block by block in step and the
        // input is staged KC columns at a time (each value written once per round instead of read
        // or normed once per simdgroup). Same values, same per-row order as below.
        const uint rounds = (nblocks + NSG - 1) / NSG;
        for (uint rc = 0; rc < W; rc += RCH) {
            for (uint round = 0; round < rounds; ++round) {
                const uint blk = round * NSG + sg;
                const bool active = blk < nblocks;
                uint m0 = 0, slot = 0;
                if (active) {
                    if (kind == 0) {
                        if (blk < HD / 4) { mat = wq; m0 = h * HD + 4 * blk; slot = 4 * blk; }
                        else { mat = wr; m0 = h * RD + 4 * (blk - HD / 4); slot = HD + 4 * (blk - HD / 4); }
                    } else if (kind == 1) { mat = wk; m0 = h * HD + 4 * blk; slot = 4 * blk; }
                    else if (kind == 2) { mat = wv; m0 = h * HD + 4 * blk; slot = 4 * blk; }
                    else { mat = wg; m0 = h * GROWS + 4 * blk; slot = 0; }
                }
                const device bfloat16_t* wrow = mat + size_t(m0) * D + lane * 4;
                float acc[4][RCH];
                for (uint tm = 0; tm < 4; ++tm)
                    for (uint r = 0; r < RCH; ++r) acc[tm][r] = 0.0f;
                for (uint c0 = 0; c0 < D; c0 += KC) {
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint idx = thread_index_in_threadgroup; idx < RCH * (KC / 4); idx += THREADS) {
                        const uint rr = idx / (KC / 4), c = (idx % (KC / 4)) * 4;
                        if (rc + rr < W && c0 + c < D) {
                            STAGE4(rc + rr, c0 + c, &chunk_in[rr * KC + c]);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (active) {
                        const uint i_end = (c0 + KC < D ? c0 + KC : D) / 128;
                        for (uint i = c0 / 128; i < i_end; ++i) {
                            float wt[4][4];
                            for (uint tm = 0; tm < 4; ++tm) LOAD_BF16X4(wrow + size_t(tm) * D + i * 128, wt[tm]);
                            for (uint r = 0; r < RCH; ++r) {
                                if (rc + r < W) {
                                    float xv[4];
                                    for (uint t_ = 0; t_ < 4; ++t_)
                                        xv[t_] = static_cast<float>(chunk_in[r * KC + (i * 128 - c0) + lane * 4 + t_]);
                                    for (uint tm = 0; tm < 4; ++tm)
                                        for (uint tn = 0; tn < 4; ++tn)
                                            acc[tm][r] += wt[tm][tn] * xv[tn];
                                }
                            }
                        }
                    }
                }
                if (active) {
                    for (uint tm = 0; tm < 4; ++tm)
                        for (uint r = 0; r < RCH; ++r)
                            for (ushort sn = 16; sn >= 1; sn >>= 1)
                                acc[tm][r] += simd_shuffle_down(acc[tm][r], sn);
                    if (lane == 0) {
                        for (uint r = 0; r < RCH && rc + r < W; ++r) {
                            for (uint tm = 0; tm < 4; ++tm) {
                                const bfloat16_t y = static_cast<bfloat16_t>(acc[tm][r]);
                                if (kind == 3) {
                                    gate[size_t(rc + r) * GN + m0 + tm] = y;
                                } else {
                                    const uint stride = kind == 0 ? OQ : OK;
                                    outs[(rc + r) * stride + slot + tm] = static_cast<float>(y);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for (uint blk = sg; !STAGE2 && blk < nblocks; blk += NSG) {
        uint m0;        // first matrix row of this block
        uint slot;      // where row tm of the block lands in outs, per panel row
        if (kind == 0) {
            if (blk < HD / 4) { mat = wq; m0 = h * HD + 4 * blk; slot = 4 * blk; }
            else { mat = wr; m0 = h * RD + 4 * (blk - HD / 4); slot = HD + 4 * (blk - HD / 4); }
        } else if (kind == 1) { mat = wk; m0 = h * HD + 4 * blk; slot = 4 * blk; }
        else if (kind == 2) { mat = wv; m0 = h * HD + 4 * blk; slot = 4 * blk; }
        else { mat = wg; m0 = h * GROWS + 4 * blk; slot = 0; }
        const device bfloat16_t* wrow = mat + size_t(m0) * D + lane * 4;
        for (uint rc = 0; rc < W; rc += RCH) {
            float acc[4][RCH];
            for (uint tm = 0; tm < 4; ++tm)
                for (uint r = 0; r < RCH; ++r) acc[tm][r] = 0.0f;
            for (uint i = 0; i < STEPS; ++i) {
                const uint col = i * 128 + lane * 4;
                float wt[4][4];
                for (uint tm = 0; tm < 4; ++tm) LOAD_BF16X4(wrow + size_t(tm) * D + i * 128, wt[tm]);
                float gv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                GAINS4(col, gv);
                for (uint r = 0; r < RCH; ++r) {
                    if (rc + r < W) {
                        float xv[4];
                        FETCH4(rc + r, col, xv, gv);
                        for (uint tm = 0; tm < 4; ++tm)
                            for (uint tn = 0; tn < 4; ++tn)
                                acc[tm][r] += wt[tm][tn] * xv[tn];
                    }
                }
            }
            for (uint tm = 0; tm < 4; ++tm)
                for (uint r = 0; r < RCH; ++r)
                    for (ushort sn = 16; sn >= 1; sn >>= 1)
                        acc[tm][r] += simd_shuffle_down(acc[tm][r], sn);
            if (lane == 0) {
                for (uint r = 0; r < RCH && rc + r < W; ++r) {
                    for (uint tm = 0; tm < 4; ++tm) {
                        const bfloat16_t y = static_cast<bfloat16_t>(acc[tm][r]);
                        if (kind == 3) {
                            gate[size_t(rc + r) * GN + m0 + tm] = y;
                        } else {
                            const uint stride = kind == 0 ? OQ : OK;
                            outs[(rc + r) * stride + slot + tm] = static_cast<float>(y);
                        }
                    }
                }
            }
        }
    }

    // bn8 rows: the sink row of a query unit, the four dynamic taps of a key or value unit.
    const bool bn8_sink = kind == 0 && sg == NSG - 1;
    const bool bn8_taps = (kind == 1 || kind == 2) && TAPS > 0 && sg == 0;
    if (bn8_sink || bn8_taps) {
        const device bfloat16_t* bm = bn8_sink ? wsink : (kind == 1 ? wkt : wvt);
        const uint nr = bn8_sink ? 1u : uint(TAPS);
        const uint b0 = bn8_sink ? h : h * uint(TAPS);
        for (uint rc = 0; rc < W; rc += RCH) {
            float total[4][RCH];
            for (uint v = 0; v < 8; ++v) {
                float p[4][RCH];
                for (uint tm = 0; tm < 4; ++tm)
                    for (uint r = 0; r < RCH; ++r) p[tm][r] = 0.0f;
                for (uint b = 0; b <= FULL; ++b) {
                    const uint col = b * 1024 + v * 128 + lane * 4;
                    if (b == FULL && D == FULL * 1024) break;      // no leftover block
                    for (uint tm = 0; tm < 4; ++tm) {
                        if (tm >= nr) break;
                        float wt[4];
                        for (uint tn = 0; tn < 4; ++tn)
                            wt[tn] = (col + tn < D) ? static_cast<float>(bm[size_t(b0 + tm) * D + col + tn]) : 0.0f;
                        for (uint r = 0; r < RCH; ++r) {
                            if (rc + r < W) {
                                float xv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                                if (col + 4 <= D) {
                                    float gv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                                    GAINS4(col, gv);
                                    FETCH4(rc + r, col, xv, gv);
                                }
                                for (uint tn = 0; tn < 4; ++tn) p[tm][r] += wt[tn] * xv[tn];
                            }
                        }
                    }
                }
                for (uint tm = 0; tm < 4; ++tm)
                    for (uint r = 0; r < RCH; ++r)
                        for (ushort sn = 16; sn >= 1; sn >>= 1)
                            p[tm][r] += simd_shuffle_down(p[tm][r], sn);
                for (uint tm = 0; tm < 4; ++tm)
                    for (uint r = 0; r < RCH; ++r)
                        total[tm][r] = (v == 0) ? p[tm][r] : total[tm][r] + p[tm][r];
            }
            if (lane == 0) {
                for (uint r = 0; r < RCH && rc + r < W; ++r) {
                    for (uint tm = 0; tm < nr; ++tm) {
                        const float y = static_cast<float>(static_cast<bfloat16_t>(total[tm][r]));
                        if (bn8_sink) outs[(rc + r) * OQ + HD + RD] = y;
                        else outs[(rc + r) * OK + HD + tm] = y;
                    }
                }
            }
        }
    }

    if (kind == 3) {
        return;        // gate slices have no epilogue (uniform per threadgroup)
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ------------------------------------------------------------------ epilogues
    for (uint r = sg; r < W; r += NSG) {
        float tau = 1.0f;
        if (GLOBAL) {
            // RC6: pos = arange(rows) + offset + 1 (int32); tau = 1 + alpha * log(max(pos / floor, 1)).
            const float p = static_cast<float>(int(pos[0]) + int(r) + 1);
            const float d = p / as_type<float>(uint(LOG_FLOOR));
            const float m = metal::isnan(d) ? d : (d > 1.0f ? d : 1.0f);
            const float l = metal::precise::log(m);
            const float a = as_type<float>(uint(LOG_ALPHA)) * l;
            tau = 1.0f + a;
        }
        if (kind == 0) {
            // Query head: clip_heads(x, threshold, scale), then the position scale.
            float value[4];
            float square[4];
            for (uint j = 0; j < 4; ++j) {
                const uint d = lane * 4 + j;
                value[j] = outs[r * OQ + d] * static_cast<float>(q_scale[h * HD + d]);
                square[j] = value[j] * value[j];
            }
            float part = 0.0f;
            for (uint j = 0; j < 4; ++j) part = square[j] + part;
            const float norm = metal::precise::sqrt(simd_sum(part));
            const float denom = norm + as_type<float>(uint(CLIP_EPS));
            const float ratio = as_type<float>(uint(QK_CLIP)) / denom;
            const float factor = (1.0f < ratio) ? 1.0f : ratio;
            for (uint j = 0; j < 4; ++j) {
                const uint d = lane * 4 + j;
                bfloat16_t y = static_cast<bfloat16_t>(value[j] * factor);
                if (GLOBAL) y = static_cast<bfloat16_t>(static_cast<float>(y) * tau);
                q[(size_t(r) * QH + h) * HD + d] = y;
            }
            if (lane == 0) {
                // Relative head (16 wide): one thread, the library's sequential small-row sum.
                float rv[RD];
                float total = 0.0f;
                for (uint j = 0; j < RD; ++j) {
                    rv[j] = outs[r * OQ + HD + j] * static_cast<float>(r_scale[h * RD + j]);
                    const float sq = rv[j] * rv[j];
                    total = sq + total;
                }
                const float rnorm = metal::precise::sqrt(total);
                const float rden = rnorm + as_type<float>(uint(CLIP_EPS));
                const float rratio = as_type<float>(uint(REL_CLIP)) / rden;
                const float rfactor = (1.0f < rratio) ? 1.0f : rratio;
                for (uint j = 0; j < RD; ++j) {
                    bfloat16_t y = static_cast<bfloat16_t>(rv[j] * rfactor);
                    if (GLOBAL) y = static_cast<bfloat16_t>(static_cast<float>(y) * tau);
                    rel[(size_t(r) * QH + h) * RD + j] = y;
                }
                sink[size_t(r) * QH + h] = static_cast<bfloat16_t>(outs[r * OQ + HD + RD]);
            }
        } else {
            // Key or value head: 4-tap convolution with the committed tail, then the key clip or
            // the value norm. Lane l owns channels 4l .. 4l+3 of head h.
            const bool keys = kind == 1;
            const device float* tail = keys ? k_tail : v_tail;
            const device float* cw = keys ? k_conv : v_conv;
            // Tap t of channel ch: the weight arrives as stored, (KN, 4) channel-major for the
            // window layers' static weight (CONV_CM), (4, KN) for the global layers'.
            const uint cw0 = CONV_CM ? 1 : KN;
            const uint cw1 = CONV_CM ? 4 : 1;
            device float* conv_in = keys ? k_conv_in : v_conv_in;
            device bfloat16_t* rows_out = keys ? k_rows : v_rows;
            float y[4];
            for (uint j = 0; j < 4; ++j) {
                const uint c = lane * 4 + j;
                const uint ch = h * HD + c;
                conv_in[size_t(r) * KN + ch] = outs[r * OK + c];
                float hh[4];
                for (uint t = 0; t < 4; ++t) {
                    const uint i = r + t;
                    if (i < 3) hh[t] = HAS_TAIL ? tail[i * KN + ch] : 0.0f;
                    else hh[t] = outs[(i - 3) * OK + c];
                }
                float acc = 0.0f;
                if (TAPS > 0) {
                    // Dynamic: newest tap first; offset, product and sum each rounded alone.
                    for (uint lag = 0; lag < 4; ++lag) {
                        const float offset = outs[r * OK + HD + lag];
                        const float coefficient = cw[lag * cw0 + ch * cw1] + offset;
                        const float product = hh[3 - lag] * coefficient;
                        acc = acc + product;
                    }
                } else if (FORM == 0) {
                    for (uint t = 0; t < 4; ++t) {
                        acc += hh[t] * cw[t * cw0 + ch * cw1];
                    }
                } else {
                    for (uint t = 0; t < 4; ++t) {
                        acc = metal::fma(hh[t], cw[t * cw0 + ch * cw1], acc);
                    }
                }
                const float total = hh[3] + acc;
                const float value = total / as_type<float>(uint(DIVISOR));
                y[j] = static_cast<float>(static_cast<bfloat16_t>(value));
            }
            if (keys) {
                float square[4];
                for (uint j = 0; j < 4; ++j) square[j] = y[j] * y[j];
                float part = 0.0f;
                for (uint j = 0; j < 4; ++j) part = square[j] + part;
                const float norm = metal::precise::sqrt(simd_sum(part));
                const float denom = norm + as_type<float>(uint(CLIP_EPS));
                const float ratio = as_type<float>(uint(QK_CLIP)) / denom;
                const float factor = (1.0f < ratio) ? 1.0f : ratio;
                for (uint j = 0; j < 4; ++j)
                    rows_out[(size_t(h) * W + r) * HD + lane * 4 + j] = static_cast<bfloat16_t>(y[j] * factor);
            } else if (VNORM) {
                float acc = 0;
                for (uint j = 0; j < 4; ++j) acc += y[j] * y[j];
                acc = simd_sum(acc);
                const float inverse = metal::precise::rsqrt(acc / float(HD) + as_type<float>(uint(NORM_EPS)));
                for (uint j = 0; j < 4; ++j)
                    rows_out[(size_t(h) * W + r) * HD + lane * 4 + j] = static_cast<bfloat16_t>(y[j] * inverse);
            } else {
                for (uint j = 0; j < 4; ++j)
                    rows_out[(size_t(h) * W + r) * HD + lane * 4 + j] = static_cast<bfloat16_t>(y[j]);
            }
        }
    }
"""


@lru_cache(maxsize=1)
def _project_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_inkling_fast_project",
        input_names=[
            "normed", "hraw", "parts", "g0", "g1", "g2",
            "wq", "wr", "wsink", "wg", "wk", "wv", "wkt", "wvt",
            "q_scale", "r_scale", "k_conv", "v_conv", "k_tail", "v_tail", "pos",
        ],
        output_names=["q", "rel", "sink", "gate", "k_rows", "v_rows", "k_conv_in", "v_conv_in"],
        header=_HEADER,
        source=_PROJECT,
    )


class Projected(NamedTuple):
    q: mx.array
    rel: mx.array
    sink: mx.array
    gate_logits: mx.array
    k_rows: mx.array
    v_rows: mx.array
    k_conv_in: mx.array
    v_conv_in: mx.array


def project(layer, sources, k_tail, v_tail, pos, *, raw=None, form="float", threads=None, gate_rows=None,
            stage=None):
    """Every projection of one visit and its epilogues, for a panel, in one launch.

    ``sources``: ``(W, NS, D)`` bf16 normed inputs (``NS`` 1: q = k = v input; 3: q, k, v
    inputs), or ``None`` with ``raw = (h, parts, gains)``: ``h`` ``(W, 3, D)`` float32
    un-normed sources, ``parts`` ``(W, 3, D // 128)`` the norm's per-128 partial sums (or
    ``None``: computed here), ``gains`` the three ``(D,)`` bf16 norm weights (the norm then runs
    here, as MLX writes it).
    ``k_tail``/``v_tail``: ``(1, 3, KN)`` float32 committed convolution inputs or ``None``.
    ``pos``: uint32 array, committed tokens. ``form``: short_conv's proven static form.
    """
    cfg = layer.cfg
    threads = threads or _THREADS
    gate_rows_unit = gate_rows or _GATE_ROWS
    global_visit = not layer.sliding
    q_heads, head_dim = layer.q_scale.shape
    rel_dim = layer.r_scale.shape[1]
    kv_width = layer.k.shape[0]
    kv_heads = kv_width // head_dim
    gate_rows = layer.gate.shape[0]
    width = layer.q.shape[1]
    if width % 128 or head_dim % 4 or rel_dim % 4 or gate_rows % gate_rows_unit:
        raise ValueError("project serves widths that are multiples of 128 (Inkling: 2,304)")
    taps = 0
    if global_visit:
        taps = layer.convs[0].dynamic.shape[0] // kv_heads
    # Every input must reach the launch row-contiguous without a copy (the kernel copies any
    # that is not, and only those). The window layers' static convolution weight is a transposed
    # view of its stored (KN, 4) array (model.Convolution: weight[:, 0, :].T), so it is passed
    # back in that stored layout; the global layers' static_w is stored (4, KN).
    conv_cm = layer.convs[0].dynamic is None
    k_conv = layer.convs[0].weight.T if conv_cm else layer.convs[0].weight
    v_conv = layer.convs[1].weight.T if conv_cm else layer.convs[1].weight
    # Stand-ins for absent operands are existing arrays (never read), so no fill launches.
    stand_f32, stand_bf16 = k_conv, layer.q_scale
    parts_in = True
    if raw is None:
        rows, ns = sources.shape[0], sources.shape[1]
        normed = sources
        hraw = parts = stand_f32
        gain_list = [stand_bf16] * 3
    else:
        hraw, parts, gain_list = raw
        rows, ns = hraw.shape[0], hraw.shape[1]
        parts_in = parts is not None
        if not parts_in:
            parts = stand_f32
        normed = stand_bf16
        gain_list = list(gain_list) + [gain_list[-1]] * (3 - len(gain_list))
    has_tail = k_tail is not None and v_tail is not None
    stand_in = stand_f32
    wkt = layer.convs[0].dynamic if taps else layer.k
    wvt = layer.convs[1].dynamic if taps else layer.v
    size = layer.convs[0].size
    divisor = math.sqrt((2 * size + 1) if taps else (size + 1))
    eps = layer.eps
    pos = pos.reshape(1) if pos.ndim == 0 else pos
    if pos.dtype != mx.uint32:
        pos = pos.astype(mx.uint32)
    groups = q_heads + 2 * kv_heads + gate_rows // gate_rows_unit
    out_bytes = (head_dim + rel_dim + 1) * rows * 4
    staging = _STAGE if stage is None else bool(stage)
    stage = bool(staging and rows * width * 2 + out_bytes <= 31 * 1024)
    # A wider loop-pass panel stages KC columns at a time (all simdgroups in step) rather than norm
    # its input in every simdgroup; a wider first-pass panel reads its bf16 input from device memory,
    # which measured faster than chunked staging (receipts/probe-mode-1.json: 86 vs 104 us at 8 rows).
    stage2 = bool(raw is not None and not stage)
    chunk = min(rows, _ROW_CHUNK)
    kc = max(128, min(width, ((31 * 1024 - out_bytes) // (chunk * 2)) // 128 * 128))
    outputs = _project_kernel()(
        inputs=[
            normed, hraw, parts, *gain_list,
            layer.q, layer.r, layer.sink, layer.gate, layer.k, layer.v, wkt, wvt,
            layer.q_scale, layer.r_scale, k_conv, v_conv,
            k_tail if has_tail else stand_in, v_tail if has_tail else stand_in, pos,
        ],
        template=[
            ("W", rows), ("NS", ns), ("RAW", raw is not None), ("PARTS_IN", parts_in),
            ("STAGE", stage), ("STAGE2", stage2),
            ("KC", kc), ("GLOBAL", global_visit),
            ("D", width), ("QH", q_heads), ("HD", head_dim), ("RD", rel_dim),
            ("KH", kv_heads), ("KN", kv_width), ("GN", gate_rows), ("GROWS", gate_rows_unit),
            ("TAPS", taps), ("RCH", min(rows, _ROW_CHUNK)), ("THREADS", threads),
            ("HAS_TAIL", has_tail), ("FORM", 0 if form == "float" else 1), ("CONV_CM", conv_cm),
            ("VNORM", bool(cfg["use_value_norm_no_gamma"])),
            ("QK_CLIP", f32_bits(cfg["qk_l2_clip_threshold"])),
            ("REL_CLIP", f32_bits(cfg["rel_l2_clip_threshold"])),
            ("CLIP_EPS", f32_bits(1e-6)), ("NORM_EPS", f32_bits(eps)),
            ("DIVISOR", f32_bits(divisor)),
            ("LOG_FLOOR", f32_bits(float(cfg["log_scaling_n_floor"]))),
            ("LOG_ALPHA", f32_bits(cfg["log_scaling_alpha"])),
        ],
        grid=(groups * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[
            (1, rows, q_heads, head_dim), (1, rows, q_heads, rel_dim), (1, rows, q_heads),
            (1, rows, gate_rows), (1, kv_heads, rows, head_dim), (1, kv_heads, rows, head_dim),
            (1, rows, kv_width), (1, rows, kv_width),
        ],
        output_dtypes=[mx.bfloat16] * 6 + [mx.float32] * 2,
    )
    return Projected(*outputs)


@lru_cache(maxsize=8)
def _static_form(width, size):
    """short_conv's first-use verdict for a static bf16 convolution of this width ("float"/"fused")."""
    from .. import short_conv

    return short_conv._proven(mx.bfloat16, None, width, 1, size)


def attention_input(layer, x, closed, k_tail, v_tail, pos, messages=None, message_queries=None, *,
                    rows=None):
    """Stage A of one visit for a panel of ``W`` rows (``lanes/SIGNATURES.md`` section 3A).

    ``x`` ``(1, W, D)`` bf16; ``closed`` RC6's closed blocks before this layer's append;
    ``k_tail``/``v_tail`` ``(1, 3, 1024)`` f32 (or ``None``: zeros); ``pos`` uint32 committed
    tokens; on loop passes 2 and 3 ``messages`` (RC6's normalised message stack
    ``(1, W, P, D)`` f32) and ``message_queries`` (the visit's ``(4, D)`` rows), or ``rows``, the
    visit's exact message rows (``read.message_rows``) when they are already read. Returns
    ``Projected`` (q, rel, sink, gate_logits, k_rows, v_rows, k_conv_in, v_conv_in), bit-identical
    to RC6's one-row step for every row. Raises ``RuntimeError`` when this Mac's convolution
    proof refused the static form (the caller serves RC6's path).
    """
    from .read import depth_keys, depth_read, message_rows, mix, read

    form = "float"
    if layer.sliding:
        form = _static_form(layer.k.shape[0], layer.convs[0].size)
        if form not in ("float", "fused"):
            count_route("refused:static_conv_unproven")
            raise RuntimeError(f"static convolution unproven on this GPU: {form}")
    if not layer.cfg["attn_res_normalize_values"]:
        count_route("refused:unnormalized_values")
        raise RuntimeError("the fused stage serves depth reads that mix normalized sources")
    loop = messages is not None or rows is not None
    count_route("fused:" + ("window" if layer.sliding else "global") + ("_loop" if loop else "_first"))
    count, width = x.shape[-2], x.shape[-1]
    x2 = x.reshape(count, width)
    closed2 = [c.reshape(count, width) for c in closed]
    if layer.sliding and not loop:
        (normed,) = read(x2, closed2, layer.attn_query, layer.eps, mode=0, gains=[layer.qkv_norm[0]],
                         f32_stand_in=layer.mlp_query)
        return project(layer, normed, k_tail, v_tail, pos, form=form)
    exact = None
    if not layer.sliding:
        # RC6's three-query read: keys here, products on MLX's own kernels (see read.depth_read).
        keys = depth_keys(x2, closed2, layer.attn_query, layer.eps)
        exact = depth_read(layer.attn_query, keys).reshape(count, 3, width)
        if not loop:
            return project(layer, None, k_tail, v_tail, pos, raw=(exact, None, list(layer.qkv_norm)),
                           form=form)
    if rows is None:
        rows = message_rows(messages, message_queries)
    controllers = layer.controllers[:3]
    state, xs, ms = read(x2, closed2, layer.attn_query, layer.eps, mode=2, messages=rows, read_in=exact,
                         downs=[c.down for c in controllers], msg_base=0, controllers=3)
    h, parts = mix(state, xs, ms, [c.forget for c in controllers], [c.gain for c in controllers],
                   parts=True)
    gains = [layer.qkv_norm[0]] * 3 if layer.sliding else list(layer.qkv_norm)
    return project(layer, None, k_tail, v_tail, pos, raw=(h, parts, gains), form=form)


def stage_attention_input(layer, x, closed, message, state, visit, panel):
    """Drop-in for ``fast.step.Stages.attention_input`` (L0's frame): same arguments, same result type."""
    from .read import message_rows_of
    from .step import AttentionInput

    p = attention_input(layer, x, closed, state.tail(visit, 0), state.tail(visit, 1), panel.pos,
                        rows=None if message is None else message_rows_of(message))
    return AttentionInput(p.q, p.rel, p.sink, p.gate_logits, p.k_rows, p.v_rows, p.k_conv_in, p.v_conv_in)
