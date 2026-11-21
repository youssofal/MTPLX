"""Stage A of an Inkling 2 Mini layer visit: everything between the visit input and attention.

RC6 spends about 55 Metal launches here on a first-pass visit and about 100 on loop passes
two and three (`model.py` 487-510 and 441-466): the depth read, the cross-pass controllers,
the input norms, six or eight separate projections, the per-head clips, the value norm,
the position scale, both short convolutions and the cache update. This module runs the
same arithmetic in these launches (W panel rows):

1. ``read`` (``fast/read.py``): one-query depth read, controller norms and down product, input
   norms; on global visits the keys of the three-query read.
2. RC6's three-query depth read (global visits) and four-query message read (passes 2 and 3,
   shared with the expert input): their products sum in the matrix unit's order, so they run on
   MLX's own products, the scores of every row as one product (the rows' keys as columns) and
   the mix batched, each admitted per shape by a first-use proof that it gives RC6's per-row
   bits on this Mac (else one MLX call a row, as RC6 makes them), with RC6's scale and softmax in
   one launch between: 5 launches for a depth read (keys included), 4 for a message read.
3. passes two and three only: controller forget/gain products and the tanh mix.
4. ``project`` (here): every projection of the visit for the whole panel in one launch,
   each stored bf16 weight row read once per panel and in place, followed by the per-head
   epilogues: scaled clips of query and relative, the position scale, both convolutions
   with their committed tails, the key clip and the value norm.

Launches a visit, any panel width: window first pass 2; global first pass 6; window loop pass 7;
global loop pass 12 (where a first-use proof refused the one-launch products: ``W + 4``,
``2 W + 4``, ``3 W + 7``).

Exactness (class A, proven on first use): every output row of a product is a twin of
MLX 0.32.2's one-row ``gemv`` for that matrix shape (``matmul.cpp`` ``gemv_axbpy``):

* query, relative, gate, key, value: ``bm4 bn1 sm1 sn32 tm4 tn4``: lane ``l`` sums columns
  ``128 i + 4 l .. + 3`` for ``i = 0 .. 17`` in order, then the shuffle-down tree 16..1;
* sink and the dynamic taps (``K >= 16 N``): ``bm1 bn8``: eight slices of 1,024-column blocks
  plus a zero-padded tail, a tree per slice, then the eight slice totals in order.

* affine-coded matrices (the Bare Speed pack: 4-bit codes, bf16 scales and biases in groups of
  64, MLX's layout, read in place): MLX's one-row quantized product, ``affine_qmv`` at 4 bits and
  ``affine_qmv_fast`` at 8 bits for K = 2,304 (lane ``l`` takes 8 values of each 256-column block;
  ``load_vector``'s bf16 quartet sums and pre-scaled inputs, once per panel row and block;
  ``qdot``'s masked products in its expression tree; ``scale * accum + sum * bias``; ``simd_sum``).
  A matrix whose stored form this launch cannot twin is refused, counted, never approximated.

A panel row's arithmetic never depends on the panel width, so every row of a W-row panel
gets the bits of a one-row step (rule 2). (MLX's own product does not have that property for
affine matrices: from 2 rows on M3 and newer it switches to ``affine_qmv_wide``, another order.) Products of two bf16 values are exact in float32,
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
import weakref

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

#: Threads of one projection threadgroup (sixteen simdgroups): with gate slices of threads / 4 rows,
#: every unit (query head 36 four-row blocks, key or value head 32, gate slice 32) takes about two
#: rounds of blocks, so no unit is the launch's long pole (receipts/projvisit-3.json).
_THREADS = 512
#: Gate rows a gate threadgroup projects.
_GATE_ROWS = 32
#: Panel rows one simdgroup accumulates at once (wider panels loop, re-reading cached weights).
#: Panels of 5 to 7 rows take the full chunk too, the rows past the panel skipped: an odd or short
#: chunk compiled to a much slower loop (7 rows: window first pass 131.6 -> 67.1 µs, loop pass
#: 139.1 -> 85.4; 6 rows: 78.8 -> 63.8 and 89.5 -> 64.2; receipts/rch-1.json), bits unchanged.
_ROW_CHUNK = 8
#: Stage each threadgroup's input in threadgroup memory (whole panel when it fits, else in chunks).
_STAGE = True


def f32_bits(value: float) -> int:
    """The float32 MLX rounds a Python float to, as raw bits for a template constant."""
    return struct.unpack("<I", struct.pack("<f", value))[0]


class Matrix(NamedTuple):
    """One projection as the launch reads it: bf16 rows, or affine codes with scales and biases."""

    data: mx.array      # (rows, cols) bf16, or (rows, cols * bits / 32) uint32 codes
    scales: object      # (rows, cols / group) bf16, or None
    biases: object
    bits: int           # 16, 8 or 4
    group: int
    rows: int
    cols: int


def matrix(weight):
    """``weight``: a bf16 ``mx.array`` or an affine weight (``precision.AffineWeight``: codes,
    scales, biases, bits, group_size)."""
    if hasattr(weight, "codes") and hasattr(weight, "group_size"):
        bits, group = int(weight.bits), int(weight.group_size)
        rows, words = weight.codes.shape
        return Matrix(weight.codes, weight.scales, weight.biases, bits, group, rows, words * 32 // bits)
    return Matrix(weight, None, None, 16, 64, weight.shape[0], weight.shape[1])


# Layers are MLX modules (dicts, unhashable): keyed by id, dropped when the layer is collected.
_joined_sb: dict = {}


def joined_scales(layer, matrices):
    """Every affine matrix's scales then biases (q, r, gate, k, v order) in one bf16 buffer, built
    once per layer and kept (1.2 MB a layer at 4 bits, group 64): Metal's 31-buffer limit leaves
    the launch one slot for them. Returns (buffer or None, element offset of each matrix's scales)."""
    key = id(layer)
    hit = _joined_sb.get(key)
    if hit is not None and hit[0]() is layer:
        return hit[1]
    pieces, offsets, at = [], [], 0
    for m in matrices:
        offsets.append(at)
        if m.bits != 16:
            pieces += [m.scales.reshape(-1), m.biases.reshape(-1)]
            at += m.scales.size + m.biases.size
    entry = (mx.concatenate(pieces) if pieces else None, offsets)
    _joined_sb[key] = (weakref.ref(layer, lambda _ref, k=key: _joined_sb.pop(k, None)), entry)
    return entry


def affine_route(m):
    """Whether this launch twins MLX 0.32.2's one-row product for an affine matrix: the lane map
    here is 8 values a lane per 256 columns, which is ``qmv`` at 4 bits when the row is not a
    multiple of 512 wide (or the rows are not a multiple of 8), and ``qmv_fast`` at 8 bits when the
    rows are a multiple of 8 and the row a multiple of 256. ``''`` when served, else why not."""
    if m.bits == 16:
        return ""
    if m.bits not in (4, 8):
        return f"bits_{m.bits}"
    if m.group not in (32, 64, 128) or m.cols % m.group or m.cols % 256:
        return f"group_{m.group}_cols_{m.cols}"
    if m.scales is None or m.biases is None or m.scales.dtype != mx.bfloat16 or m.biases.dtype != mx.bfloat16:
        return "scales_not_bf16"
    if m.bits == 4 and m.cols % 512 == 0 and m.rows % 8 == 0:
        return "qmv_fast_4bit"          # MLX would take 16 values a lane: another kernel
    if m.bits == 8 and (m.rows % 8 or m.cols % 256):
        return "qmv_8bit"
    return ""


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

// An affine-coded matrix (MLX layout: uint32 codes, the first in the lowest bits; one bf16 scale
// and bias per group), as MLX 0.32.2's one-row product takes it when K = 2,304: qmv at 4 bits,
// qmv_fast at 8 bits, both 8 values a lane per 256-column block. affine_prep is its load_vector
// (once per panel row, block and lane, shared by every output row, as MLX shares it by 4); affine_dot
// is its qdot for one output row; the caller adds each block's term to the row's running total and
// ends with simd_sum. xv: the lane's 8 input values (bf16 values held as float).
inline float affine_prep(const float xv[8], thread float xt[8], uint bits) {
    float sum = 0;
    if (bits == 4) {
        for (uint i = 0; i < 8; i += 4) {
            const bfloat16_t a0 = static_cast<bfloat16_t>(xv[i]);
            const bfloat16_t a1 = static_cast<bfloat16_t>(xv[i + 1]);
            const bfloat16_t a2 = static_cast<bfloat16_t>(xv[i + 2]);
            const bfloat16_t a3 = static_cast<bfloat16_t>(xv[i + 3]);
            sum += a0 + a1 + a2 + a3;
            xt[i] = a0;
            xt[i + 1] = a1 / 16.0f;
            xt[i + 2] = a2 / 256.0f;
            xt[i + 3] = a3 / 4096.0f;
        }
    } else {
        for (uint i = 0; i < 8; i++) {
            const bfloat16_t a = static_cast<bfloat16_t>(xv[i]);
            sum += a;
            xt[i] = a;
        }
    }
    return sum;
}

inline float affine_dot(const thread float xt[8], float sum, const device uint32_t* code, float scale, float bias,
                        uint bits) {
    float accum = 0;
    if (bits == 4) {
        const uint word = code[0];
        const ushort w16[2] = {ushort(word & 0xffffu), ushort(word >> 16)};
        for (uint i = 0; i < 2; ++i) {
            accum +=
                (xt[4 * i] * (w16[i] & 0x000f) +
                 xt[4 * i + 1] * (w16[i] & 0x00f0) +
                 xt[4 * i + 2] * (w16[i] & 0x0f00) +
                 xt[4 * i + 3] * (w16[i] & 0xf000));
        }
    } else {
        const uint words[2] = {code[0], code[1]};
        for (uint i = 0; i < 8; i++) {
            const uchar w = uchar((words[i / 4] >> (8 * (i % 4))) & 0xffu);
            accum += xt[i] * w;
        }
    }
    return scale * accum + sum * bias;
}

// The lane's 8 codes of one output row as the floats MLX's qdot multiplies by: at 4 bits the masked
// 16-bit words (q, q << 4, q << 8, q << 12, each exact in float), at 8 bits the bytes. Decoded once a
// block and shared by every panel row.
inline void affine_decode(const device uint32_t* code, thread float wf[8], uint bits) {
    if (bits == 4) {
        const uint word = code[0];
        const ushort w16[2] = {ushort(word & 0xffffu), ushort(word >> 16)};
        for (uint i = 0; i < 2; ++i) {
            wf[4 * i] = float(w16[i] & 0x000f);
            wf[4 * i + 1] = float(w16[i] & 0x00f0);
            wf[4 * i + 2] = float(w16[i] & 0x0f00);
            wf[4 * i + 3] = float(w16[i] & 0xf000);
        }
    } else {
        const uint words[2] = {code[0], code[1]};
        for (uint i = 0; i < 8; i++) wf[i] = float(uchar((words[i / 4] >> (8 * (i % 4))) & 0xffu));
    }
}

// MLX's pre-scaled inputs alone (load_vector without its sum), 4 bits.
inline void affine_xt4(const float xv[8], thread float xt[8]) {
    for (uint i = 0; i < 8; i += 4) {
        xt[i] = static_cast<float>(static_cast<bfloat16_t>(xv[i]));
        xt[i + 1] = static_cast<bfloat16_t>(xv[i + 1]) / 16.0f;
        xt[i + 2] = static_cast<bfloat16_t>(xv[i + 2]) / 256.0f;
        xt[i + 3] = static_cast<bfloat16_t>(xv[i + 3]) / 4096.0f;
    }
}

// qdot on decoded codes: the same products (exact) in the same expression tree.
inline float affine_dotf(const thread float xt[8], float sum, const thread float wf[8], float scale, float bias,
                         uint bits) {
    float accum = 0;
    if (bits == 4) {
        for (uint i = 0; i < 2; ++i) {
            accum +=
                (xt[4 * i] * wf[4 * i] +
                 xt[4 * i + 1] * wf[4 * i + 1] +
                 xt[4 * i + 2] * wf[4 * i + 2] +
                 xt[4 * i + 3] * wf[4 * i + 3]);
        }
    } else {
        for (uint i = 0; i < 8; i++) accum += xt[i] * wf[i];
    }
    return scale * accum + sum * bias;
}

// One block term in one call (the bench kernels' form): affine_prep then affine_dot.
inline float affine_term(const float xv[8], const device uint32_t* code, float scale, float bias, uint bits) {
    float xt[8];
    const float sum = affine_prep(xv, xt, bits);
    return affine_dot(xt, sum, code, scale, bias, bits);
}
"""

_PROJECT = """
    // One input slot holds the bf16 normed sources (NORMED) or the float32 raw ones (RAW).
    const device bfloat16_t* normed = reinterpret_cast<const device bfloat16_t*>(xin);
    const device float* hraw = reinterpret_cast<const device float*>(xin);
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
    // AFFSUM: the 4-bit load_vector sum (bf16 quartets, MLX's order) of every panel row, 256-column
    // block and lane, once per threadgroup instead of once per output row.
    threadgroup float asum[AFFSUM ? W * (D / 256) * 32 : 1];
    if (AFFSUM) {
        for (uint idx = thread_index_in_threadgroup; idx < W * (D / 256) * 32; idx += THREADS) {
            const uint rr = idx / ((D / 256) * 32), cb = (idx / 32) % (D / 256), ll = idx % 32;
            const uint col = cb * 256 + ll * 8;
            float gv0[4] = {0.0f, 0.0f, 0.0f, 0.0f}, gv1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            GAINS4(col, gv0);
            GAINS4(col + 4, gv1);
            float xv[8], xh[4];
            FETCH4(rr, col, xv, gv0);
            FETCH4(rr, col + 4, xh, gv1);
            for (uint t_ = 0; t_ < 4; ++t_) xv[4 + t_] = xh[t_];
            float xt[8];
            asum[idx] = affine_prep(xv, xt, 4);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ------------------------------------------------------------------ products
    // Each 4-row block belongs to one matrix: its bf16 rows or its affine codes (bits 4 or 8, one
    // scale and bias per group of gsize), per the matrix's compile-time format.
    const device uchar* mat = reinterpret_cast<const device uchar*>(wq);
    uint bits = 16, gsize = 64;
    // sb: every affine matrix's scales then biases in one buffer; XSO, the element offset of X's scales.
    const device bfloat16_t* scl = sb;
    const device bfloat16_t* bia = sb;
#define AS_BF16(p) reinterpret_cast<const device bfloat16_t*>(p)
#define AS_CODES(p) reinterpret_cast<const device uint32_t*>(p)
#define SB(OFF, ROWS, G) scl = sb + (OFF); bia = sb + (OFF) + (ROWS) * (D / (G));
#define SELECT_BLOCK(blk)                                                                         \\
    if (kind == 0) {                                                                              \\
        if ((blk) < HD / 4) {                                                                     \\
            mat = reinterpret_cast<const device uchar*>(wq); bits = QB; gsize = QG;              \\
            SB(QSO, QH * HD, QG); m0 = h * HD + 4 * (blk); slot = 4 * (blk);                      \\
        } else {                                                                                  \\
            mat = reinterpret_cast<const device uchar*>(wr); bits = RB; gsize = RG;              \\
            SB(RSO, QH * RD, RG); m0 = h * RD + 4 * ((blk) - HD / 4);                             \\
            slot = HD + 4 * ((blk) - HD / 4);                                                     \\
        }                                                                                         \\
    } else if (kind == 1) {                                                                       \\
        mat = reinterpret_cast<const device uchar*>(wk); bits = KB; gsize = KG;                  \\
        SB(KSO, KN, KG); m0 = h * HD + 4 * (blk); slot = 4 * (blk);                               \\
    } else if (kind == 2) {                                                                       \\
        mat = reinterpret_cast<const device uchar*>(wv); bits = VB; gsize = VG;                  \\
        SB(VSO, KN, VG); m0 = h * HD + 4 * (blk); slot = 4 * (blk);                               \\
    } else {                                                                                      \\
        mat = reinterpret_cast<const device uchar*>(wg); bits = GB; gsize = GG;                  \\
        SB(GSO, GN, GG); m0 = h * GROWS + 4 * (blk); slot = 0;                                    \\
    }
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
                    SELECT_BLOCK(blk);
                }
                const device bfloat16_t* wrow = AS_BF16(mat) + size_t(m0) * D + lane * 4;
                const device uint32_t* cw = AS_CODES(mat);
                const uint wpr = D * bits / 32;
                const uint gpr = D / gsize;
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
                    if (AFFINE && active && bits != 16) {
                        // Affine-coded matrix: blocks of 256 columns (KC is a multiple of 256 then).
                        const uint b_end = (c0 + KC < D ? c0 + KC : D) / 256;
                        for (uint b = c0 / 256; b < b_end; ++b) {
                            const uint col = b * 256 + lane * 8;
                            float sc[4], bi[4], wf[4][8];
                            for (uint tm = 0; tm < 4; ++tm) {
                                sc[tm] = static_cast<float>(scl[size_t(m0 + tm) * gpr + col / gsize]);
                                bi[tm] = static_cast<float>(bia[size_t(m0 + tm) * gpr + col / gsize]);
                                affine_decode(cw + size_t(m0 + tm) * wpr + col * bits / 32, wf[tm], bits);
                            }
                            for (uint r = 0; r < RCH; ++r) {
                                if (rc + r < W) {
                                    float xv[8];
                                    for (uint t_ = 0; t_ < 8; ++t_)
                                        xv[t_] = static_cast<float>(chunk_in[r * KC + (col - c0) + t_]);
                                    float xt[8];
                                    float sum;
                                    if (AFFSUM && bits == 4) {
                                        affine_xt4(xv, xt);
                                        sum = asum[((rc + r) * (D / 256) + b) * 32 + lane];
                                    } else {
                                        sum = affine_prep(xv, xt, bits);
                                    }
                                    for (uint tm = 0; tm < 4; ++tm)
                                        acc[tm][r] += affine_dotf(xt, sum, wf[tm], sc[tm], bi[tm], bits);
                                }
                            }
                        }
                    } else if (active) {
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
                        for (uint r = 0; r < RCH; ++r) {
                            if (AFFINE && bits != 16) {
                                acc[tm][r] = simd_sum(acc[tm][r]);
                            } else {
                                for (ushort sn = 16; sn >= 1; sn >>= 1)
                                    acc[tm][r] += simd_shuffle_down(acc[tm][r], sn);
                            }
                        }
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
        SELECT_BLOCK(blk);
        if (AFFINE && bits != 16) {
            // Affine-coded matrix: MLX's one-row product per panel row (see affine_prep, affine_dot).
            const device uint32_t* cw = AS_CODES(mat);
            const uint wpr = D * bits / 32;          // code words a row
            const uint gpr = D / gsize;              // groups a row
            for (uint rc = 0; rc < W; rc += RCH) {
                float acc[4][RCH];
                for (uint tm = 0; tm < 4; ++tm)
                    for (uint r = 0; r < RCH; ++r) acc[tm][r] = 0.0f;
                for (uint b = 0; b < D / 256; ++b) {
                    const uint col = b * 256 + lane * 8;
                    float sc[4], bi[4], wf[4][8];
                    for (uint tm = 0; tm < 4; ++tm) {
                        sc[tm] = static_cast<float>(scl[size_t(m0 + tm) * gpr + col / gsize]);
                        bi[tm] = static_cast<float>(bia[size_t(m0 + tm) * gpr + col / gsize]);
                        affine_decode(cw + size_t(m0 + tm) * wpr + col * bits / 32, wf[tm], bits);
                    }
                    float gv0[4] = {0.0f, 0.0f, 0.0f, 0.0f}, gv1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    GAINS4(col, gv0);
                    GAINS4(col + 4, gv1);
                    for (uint r = 0; r < RCH; ++r) {
                        if (rc + r < W) {
                            float xv[8];
                            FETCH4(rc + r, col, xv, gv0);
                            float xh[4];
                            FETCH4(rc + r, col + 4, xh, gv1);
                            for (uint t_ = 0; t_ < 4; ++t_) xv[4 + t_] = xh[t_];
                            float xt[8];
                            float sum;
                            if (AFFSUM && bits == 4) {
                                affine_xt4(xv, xt);
                                sum = asum[((rc + r) * (D / 256) + b) * 32 + lane];
                            } else {
                                sum = affine_prep(xv, xt, bits);
                            }
                            for (uint tm = 0; tm < 4; ++tm)
                                acc[tm][r] += affine_dotf(xt, sum, wf[tm], sc[tm], bi[tm], bits);
                        }
                    }
                }
                for (uint tm = 0; tm < 4; ++tm)
                    for (uint r = 0; r < RCH; ++r) acc[tm][r] = simd_sum(acc[tm][r]);
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
            continue;
        }
        const device bfloat16_t* wrow = AS_BF16(mat) + size_t(m0) * D + lane * 4;
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
        # Metal allows 31 buffers a launch: 21 inputs and 8 outputs.
        input_names=[
            "xin", "parts", "g0", "g1", "g2",
            "wq", "wr", "wsink", "wg", "wk", "wv", "wkt", "wvt", "sb",
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
            stage=None, rch=None, affsum=None, stage2=None):
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
    # A gate slice gives every simdgroup two 4-row blocks, as a query head gives it about 2.25.
    gate_rows_unit = gate_rows or max(_GATE_ROWS, threads // 4)
    global_visit = not layer.sliding
    q_heads, head_dim = layer.q_scale.shape
    rel_dim = layer.r_scale.shape[1]
    mq, mr, mg, mk, mv = (matrix(w) for w in (layer.q, layer.r, layer.gate, layer.k, layer.v))
    for name, m in (("q", mq), ("r", mr), ("gate", mg), ("k", mk), ("v", mv)):
        why = affine_route(m)
        if why:
            count_route(f"refused:{name}_{why}")
            raise RuntimeError(f"projection launch does not serve {name} as stored: {why}")
    if any(hasattr(w, "codes") for w in (layer.sink, layer.convs[0].dynamic, layer.convs[1].dynamic)
           if w is not None):
        count_route("refused:affine_sink_or_taps")
        raise RuntimeError("projection launch reads the sink and dynamic taps as bf16")
    kv_width = mk.rows
    kv_heads = kv_width // head_dim
    gate_rows = mg.rows
    width = mq.cols
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
        xin = sources
        parts = stand_f32
        gain_list = [stand_bf16] * 3
    else:
        xin, parts, gain_list = raw
        rows, ns = xin.shape[0], xin.shape[1]
        parts_in = parts is not None
        if not parts_in:
            parts = stand_f32
        gain_list = list(gain_list) + [gain_list[-1]] * (3 - len(gain_list))
    sb, (qso, rso, gso, kso, vso) = joined_scales(layer, (mq, mr, mg, mk, mv))
    has_tail = k_tail is not None and v_tail is not None
    stand_in = stand_f32
    wkt = layer.convs[0].dynamic if taps else layer.q_scale
    wvt = layer.convs[1].dynamic if taps else layer.q_scale
    size = layer.convs[0].size
    divisor = math.sqrt((2 * size + 1) if taps else (size + 1))
    eps = layer.eps
    pos = pos.reshape(1) if pos.ndim == 0 else pos
    if pos.dtype != mx.uint32:
        pos = pos.astype(mx.uint32)
    groups = q_heads + 2 * kv_heads + gate_rows // gate_rows_unit
    out_bytes = (head_dim + rel_dim + 1) * rows * 4
    staging = _STAGE if stage is None else bool(stage)
    affine = any(m.bits != 16 for m in (mq, mr, mg, mk, mv))
    # 4-bit sums once per threadgroup (MLX's load_vector per row, block and lane) when they fit, and
    # not when they alone keep the panel from staging whole: Bare Speed at 6 rows, loop pass 251 ->
    # 61 µs, global passes 355 -> 95 (receipts/q4forms-1.json).
    asum_bytes = rows * (width // 256) * 32 * 4
    if affsum is None:
        affsum = any(m.bits == 4 for m in (mq, mr, mg, mk, mv)) and rows <= 16
        panel_bytes = rows * width * 2 + out_bytes
        if affsum and staging and panel_bytes + asum_bytes > 31 * 1024 >= panel_bytes:
            affsum = False
    out_bytes += asum_bytes if affsum else 0
    stage = bool(staging and rows * width * 2 + out_bytes <= 31 * 1024)
    # A wider loop-pass panel stages KC columns at a time (all simdgroups in step) rather than norm
    # its input in every simdgroup; a wider first-pass panel reads its bf16 input from device memory,
    # which measured faster than chunked staging (receipts/probe-mode-1.json: 86 vs 104 us at 8 rows).
    # Affine panels too wide to stage read their sources from device memory in every simdgroup:
    # chunked staging measured 2.2 times slower there (7 rows: loop pass 286 -> 108 µs, global
    # passes 404 -> 188; 8 rows 329 -> 145 and 472 -> 220; receipts/q4forms-1.json).
    if stage2 is None:
        stage2 = not affine
    stage2 = bool(raw is not None and not stage and stage2)
    chunk = rch or (_ROW_CHUNK if rows > 4 else rows)
    # Chunks hold whole 256-column blocks when an affine matrix reads them.
    unit = 256 if affine else 128
    kc = max(unit, min(width, ((31 * 1024 - out_bytes) // (chunk * 2)) // unit * unit))
    outputs = _project_kernel()(
        inputs=[
            xin, parts, *gain_list,
            mq.data, mr.data, layer.sink, mg.data, mk.data, mv.data, wkt, wvt,
            sb if sb is not None else stand_bf16,
            layer.q_scale, layer.r_scale, k_conv, v_conv,
            k_tail if has_tail else stand_in, v_tail if has_tail else stand_in, pos,
        ],
        template=[
            ("W", rows), ("NS", ns), ("RAW", raw is not None), ("PARTS_IN", parts_in),
            ("STAGE", stage), ("STAGE2", stage2),
            ("KC", kc), ("GLOBAL", global_visit),
            ("D", width), ("QH", q_heads), ("HD", head_dim), ("RD", rel_dim),
            ("KH", kv_heads), ("KN", kv_width), ("GN", gate_rows), ("GROWS", gate_rows_unit),
            ("TAPS", taps), ("RCH", chunk), ("THREADS", threads),
            ("HAS_TAIL", has_tail), ("FORM", 0 if form == "float" else 1), ("CONV_CM", conv_cm),
            ("QB", mq.bits), ("QG", mq.group), ("RB", mr.bits), ("RG", mr.group),
            ("GB", mg.bits), ("GG", mg.group), ("KB", mk.bits), ("KG", mk.group),
            ("VB", mv.bits), ("VG", mv.group),
            ("QSO", qso), ("RSO", rso), ("GSO", gso), ("KSO", kso), ("VSO", vso),
            ("AFFINE", any(m.bits != 16 for m in (mq, mr, mg, mk, mv))), ("AFFSUM", affsum),
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
        form = _static_form(matrix(layer.k).rows, layer.convs[0].size)
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
