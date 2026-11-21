"""Inkling prompt attention in one launch per visit, on the GPU's tensor units.

What one prompt call computes, for query row r of the chunk (absolute position p), query head h:
the logits ``q.k_c / sqrt(128) + bias(p - c)`` over the keys c the row may see (``c <= p``, and
on a window visit ``p - c < 1024``), where ``bias(d)`` is the row's profile entry d for d below
the profile's extent and 0 beyond it, plus one more logit, the sink ``s[r, h]``, whose value is
zero; the output is the softmax of those logits applied to the values.

The default prompt route builds that from the library's attention: a bias tensor written per
call, the bank copied with a zero row appended for the sink, every key of the call computed
whether a row may see it or not, and past about 19,400 keys a decode-shaped kernel that has no
matrix units. Here a launch computes it for a whole visit:

* Fragment rows. The tensor units multiply 16-row blocks. By default a block is 16 consecutive
  query rows of one head and a threadgroup holds the three heads that share a key/value head,
  one simdgroup each, so every key and value fragment the threadgroup loads serves all three
  (``FOLD``); the other layout puts ``RPB = 16 // G`` rows times the ``G`` heads in one block.
  A row gets the same bits in either layout. All simdgroups of a threadgroup walk the same
  32-key tiles.
* Scores and products. ``q k^T`` over 32 keys and 32 dimensions at a time and ``P V`` over 32
  keys and 32 output dimensions, both on 16 x 32 x 32 tensor products with float sums. The
  scores, the running maximum and sum, and the output stay in float.
* The softmax is online and opened by the sink: before the first key a row's maximum is its
  sink logit, its sum 1 and its output 0, which is the sink's own term (value zero).
* Tiles. A threadgroup visits only the tiles one of its rows may see: from the first key of
  its earliest row's window (or key 0) to its latest row's own position. Tiles at a distance of
  at least the profile's extent from every row of the threadgroup need no bias and no mask
  (on a global visit at long context, nearly all of them); the rest are masked per element and
  read their bias from the profile.
* Keys, values, queries, profiles and sinks are read in place through their strides: nothing
  is concatenated, no bias or mask tensor exists, the launch writes only its output
  ``(1, rows, heads, 128)`` in the query type.

Class C against the default prompt route: the same sums in another order, all in float (the
probabilities enter the value product as float). On the shapes a prompt read makes, 99.3 to 99.9%
of output bits equal the default route's where it uses the library's attention (below about
19,400 keys), and past that the result is closer to the float32 definition than the default's
(lanes/p2/fcheck-3, fcheck-4). The simdgroups of a threadgroup are held on one tile by a barrier
each tile so they share its keys and values in the cache (2.4x at 2,048 rows and 64K keys,
lanes/p2/fsweep-1). Off by default (``MTPLX_INKLING_PROMPT_FLASH``); :func:`decline` names why a
call is not served.
"""

from __future__ import annotations

import os
import struct
from functools import lru_cache

import mlx.core as mx

HEAD_DIM = 128
#: Simdgroups a threadgroup: with FOLD (the default) one per head of a key/value head, each 16
#: rows of that head (3 for Inkling); without, TG_M blocks of 16 // G rows x G heads. Measured
#: fastest at 512-row chunks and level at 2,048 (lanes/p2/fsweep-2).
TG_M = 3
FOLD = True

_HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

#define INK_UNROLL _Pragma("clang loop unroll(full)")

// A lane's place in the tensor units' register layout of a 16-row by 16-column piece: it holds
// rows ink_row and ink_row + 8, columns ink_col .. ink_col + 3 (a 32-column operand is two such
// pieces side by side). Row from lane bits 4, 2, 1; column from lane bits 3, 0.
inline short ink_row(ushort lane) {
    return short(((lane >> 2) & 4) | ((lane >> 1) & 3));
}
inline short ink_col(ushort lane) {
    return short((((lane >> 2) & 2) | (lane & 1)) << 2);
}

// Eight consecutive elements starting at p: two 8-byte loads when the stride is one element
// (every address formed here is a multiple of four elements from an aligned first element),
// element by element otherwise.
template <typename E>
struct InkEight {
    vec<E, 4> lo;
    vec<E, 4> hi;
};

template <typename E>
inline InkEight<E> ink_load8(const device E* p, uint stride) {
    InkEight<E> r;
    if (stride == 1u) {
        r.lo = *reinterpret_cast<const device vec<E, 4>*>(p);
        r.hi = *reinterpret_cast<const device vec<E, 4>*>(p + 4);
    } else {
        INK_UNROLL for (short i = 0; i < 4; ++i) {
            r.lo[i] = p[uint(i) * stride];
            r.hi[i] = p[uint(i + 4) * stride];
        }
    }
    return r;
}
"""

# Element e of a lane's 16 x 32 destination (and of a left operand of that shape) is row
# ink_row + 8 * ((e >> 2) & 1), column ink_col + 16 * (e >> 3) + (e & 3). The 32-wide reduction
# of the score product is taken in a permuted order (each lane holds 8 consecutive dimensions
# 2 * ink_col .. + 7 of every 32), the same for queries and keys, so the score is unchanged; the
# output columns carry the same permutation, undone when the output is stored.
_SOURCE = """
    constexpr int D = 128;
    constexpr int STEPS = D / 32;
    const float scale_log2 = as_type<float>(uint(SCALE)) * 1.4426950408889634f;
    constexpr float LOG2E = 1.4426950408889634f;

    const uint tid = thread_position_in_threadgroup.x;
    const int sg = int(tid >> 5);
    const ushort lane = ushort(tid & 31u);
    const short fr = ink_row(lane);
    const short fc = ink_col(lane);
    const int kvh = int(threadgroup_position_in_grid.y);
    const int rows = int(params[0]);
    const int total = int(params[1]);
    const int hist = total - rows;                              // position of chunk row 0
    // A block is RPB rows x the G heads of the KV head (FOLD 0), or 16 rows of one head (FOLD 1:
    // the threadgroup's simdgroups are its G heads x TG_M / G row blocks, every row live).
    constexpr int TG_ROWS = FOLD ? (TG_M / G) * 16 : TG_M * RPB;
    const int tg_row0 = int(threadgroup_position_in_grid.x) * TG_ROWS;
    const int tg_row_last = min(tg_row0 + TG_ROWS, rows) - 1;
    const int blk_row0 = tg_row0 + (FOLD ? (sg / G) * 16 : sg * RPB);

    // The lane's two fragment rows: query row and head, position, whether it is a real row.
    int qrow[2];
    int qhead[2];
    int pos[2];
    bool live[2];
    INK_UNROLL for (short rp = 0; rp < 2; ++rp) {
        const int m = int(fr) + 8 * rp;
        qrow[rp] = FOLD ? blk_row0 + m : blk_row0 + m / G;
        qhead[rp] = FOLD ? kvh * G + sg % G : kvh * G + m % G;
        live[rp] = (FOLD || m < RPB * G) && qrow[rp] < rows;
        pos[rp] = hist + qrow[rp];
    }

    // The two products, each on the tensor units through one simdgroup: scores (keys transposed)
    // and the value sum. One destination of each kind is made and copied for the others.
    namespace tops = mpp::tensor_ops;
    constexpr auto mac = tops::matmul2d_descriptor::mode::multiply_accumulate;
    tops::matmul2d<tops::matmul2d_descriptor(16, 32, 32, false, true, RELAX, mac), metal::execution_simdgroup> qk_op;
    tops::matmul2d<tops::matmul2d_descriptor(16, 32, 32, false, false, RELAX, mac), metal::execution_simdgroup> pv_op;
    auto qk_a = qk_op.template get_left_input_cooperative_tensor<T, T, float>();
    auto qk_b = qk_op.template get_right_input_cooperative_tensor<T, T, float>();
    auto pv_a = pv_op.template get_left_input_cooperative_tensor<PT, T, float>();
    auto pv_b = pv_op.template get_right_input_cooperative_tensor<PT, T, float>();
    auto qk_c = qk_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(qk_a)>, metal::remove_addrspace_t<decltype(qk_b)>, float>();
    auto acc0 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pv_a)>, metal::remove_addrspace_t<decltype(pv_b)>, float>();
    auto acc1 = acc0;
    auto acc2 = acc0;
    auto acc3 = acc0;
    INK_UNROLL for (short e = 0; e < 16; ++e) {
        acc0[e] = 0.0f;
        acc1[e] = 0.0f;
        acc2[e] = 0.0f;
        acc3[e] = 0.0f;
    }

    // Query fragments: held for the whole walk (QREG) or re-read each tile (from L1).
    const uint q_dim = uint(q_strides[3]);
    const device T* q_lane[2];
    INK_UNROLL for (short rp = 0; rp < 2; ++rp) {
        q_lane[rp] = q + (size_t(qhead[rp]) * size_t(q_strides[1]) +
                          size_t(live[rp] ? qrow[rp] : 0) * size_t(q_strides[2]) +
                          size_t(2 * fc) * size_t(q_dim));
    }
    T qf[QREG ? STEPS : 1][16];
    if (QREG) {
        INK_UNROLL for (short rp = 0; rp < 2; ++rp) {
            INK_UNROLL for (short s = 0; s < (QREG ? STEPS : 1); ++s) {
                InkEight<T> x = ink_load8(q_lane[rp] + uint(32 * s) * q_dim, q_dim);
                INK_UNROLL for (short i = 0; i < 4; ++i) {
                    qf[s][rp * 4 + i] = live[rp] ? x.lo[i] : T(0);
                    qf[s][8 + rp * 4 + i] = live[rp] ? x.hi[i] : T(0);
                }
            }
        }
    }

    // The sink opens each row: maximum its logit, sum 1, output 0.
    float run_max[2];
    float run_sum[2] = {1.0f, 1.0f};
    INK_UNROLL for (short rp = 0; rp < 2; ++rp) {
        run_max[rp] = live[rp]
            ? float(sink[size_t(qrow[rp]) * size_t(sink_strides[1]) + size_t(qhead[rp]) * size_t(sink_strides[2])]) * LOG2E
            : 0.0f;
    }
    const device T* prof_row[2];
    INK_UNROLL for (short rp = 0; rp < 2; ++rp) {
        prof_row[rp] = prof + (size_t(live[rp] ? qrow[rp] : 0) * size_t(prof_strides[1]) +
                               size_t(qhead[rp]) * size_t(prof_strides[2]));
    }
    const uint prof_step = uint(prof_strides[3]);

    // Key and value rows of this lane in a tile: 16 kh + fr + 8 rh, part = 2 kh + rh. One 64-bit
    // base a lane, 32-bit element offsets from it (a head of 4 G elements is far beyond any bank).
    const uint k_dim = uint(k_strides[3]);
    const uint v_dim = uint(v_strides[3]);
    const device T* k_lane = k + (size_t(kvh) * size_t(k_strides[1]) + size_t(2 * fc) * size_t(k_dim));
    const device T* v_lane = v + (size_t(kvh) * size_t(v_strides[1]) + size_t(2 * fc) * size_t(v_dim));
    const uint k_row = uint(k_strides[2]);
    const uint v_row = uint(v_strides[2]);

    // Tiles some row of this threadgroup may see; tiles before t_near need no bias and no mask.
    const int pos_lo = hist + tg_row0;
    const int pos_hi = hist + tg_row_last;
    const int t_end = pos_hi / 32 + 1;
    int t_begin = 0;
    int t_near = 0;
    if (WINDOW > 0) {
        t_begin = max(0, pos_lo - (WINDOW - 1)) / 32;
        t_near = t_begin;
    } else {
        const int clear = pos_lo - EXTENT + 1;          // keys below this are far from every row
        t_near = clear > 0 ? min(clear / 32, t_end) : 0;
    }

    for (int t = t_begin; t < t_end; ++t) {
        if (TGBAR) {
            threadgroup_barrier(mem_flags::mem_none);      // keep the blocks on one tile: shared in L1
        }
        const int base = t * 32;
        const bool near = t >= t_near;
        uint koff[4];
        uint voff[4];
        INK_UNROLL for (short part = 0; part < 4; ++part) {
            int key = base + 16 * (part >> 1) + int(fr) + 8 * (part & 1);
            key = key < total ? key : 0;                   // past the bank: masked below
            koff[part] = uint(key) * k_row;
            voff[part] = uint(key) * v_row;
        }

        INK_UNROLL for (short e = 0; e < 16; ++e) {
            qk_c[e] = 0.0f;
        }
        INK_UNROLL for (short s = 0; s < STEPS; ++s) {
            if (QREG) {
                INK_UNROLL for (short e = 0; e < 16; ++e) {
                    qk_a[e] = qf[QREG ? s : 0][e];
                }
            } else {
                INK_UNROLL for (short rp = 0; rp < 2; ++rp) {
                    InkEight<T> x = ink_load8(q_lane[rp] + uint(32 * s) * q_dim, q_dim);
                    INK_UNROLL for (short i = 0; i < 4; ++i) {
                        qk_a[rp * 4 + i] = live[rp] ? x.lo[i] : T(0);
                        qk_a[8 + rp * 4 + i] = live[rp] ? x.hi[i] : T(0);
                    }
                }
            }
            INK_UNROLL for (short part = 0; part < 4; ++part) {
                InkEight<T> x = ink_load8(k_lane + (koff[part] + uint(32 * s) * k_dim), k_dim);
                INK_UNROLL for (short i = 0; i < 4; ++i) {
                    qk_b[part * 4 + i] = x.lo[i];
                    qk_b[16 + part * 4 + i] = x.hi[i];
                }
            }
            qk_op.run(qk_a, qk_b, qk_c);
        }

        // Logits in base-2 units.
        float lg[16];
        INK_UNROLL for (short e = 0; e < 16; ++e) {
            const short rp = (e >> 2) & 1;
            float x = qk_c[e] * scale_log2;
            if (near) {
                const int key = base + int(fc) + 16 * (e >> 3) + (e & 3);
                const int d = pos[rp] - key;
                const bool seen = live[rp] && d >= 0 && (WINDOW == 0 || d < WINDOW);
                if (!seen) {
                    x = -INFINITY;
                } else if (d < EXTENT) {
                    x = fma(float(prof_row[rp][uint(d) * prof_step]), LOG2E, x);
                }
            }
            lg[e] = x;
        }

        float corr[2];
        INK_UNROLL for (short rp = 0; rp < 2; ++rp) {
            float m = -INFINITY;
            INK_UNROLL for (short h = 0; h < 2; ++h) {
                INK_UNROLL for (short i = 0; i < 4; ++i) {
                    m = max(m, lg[h * 8 + rp * 4 + i]);
                }
            }
            m = max(m, simd_shuffle_xor(m, ushort(1)));
            m = max(m, simd_shuffle_xor(m, ushort(8)));
            const float nm = max(run_max[rp], m);
            corr[rp] = fast::exp2(run_max[rp] - nm);
            float sum = 0.0f;
            INK_UNROLL for (short h = 0; h < 2; ++h) {
                INK_UNROLL for (short i = 0; i < 4; ++i) {
                    const short e = h * 8 + rp * 4 + i;
                    const float w = fast::exp2(lg[e] - nm);
                    lg[e] = w;
                    sum += w;
                }
            }
            sum += simd_shuffle_xor(sum, ushort(1));
            sum += simd_shuffle_xor(sum, ushort(8));
            run_sum[rp] = run_sum[rp] * corr[rp] + sum;
            run_max[rp] = nm;
        }
        if (simd_any(corr[0] != 1.0f || corr[1] != 1.0f)) {
            INK_UNROLL for (short e = 0; e < 16; ++e) {
                const float c = corr[(e >> 2) & 1];
                acc0[e] *= c;
                acc1[e] *= c;
                acc2[e] *= c;
                acc3[e] *= c;
            }
        }
        if (!PSPLIT) {
            INK_UNROLL for (short e = 0; e < 16; ++e) {
                pv_a[e] = PT(lg[e]);
            }
        }
        INK_PV(0, acc0)
        INK_PV(1, acc1)
        INK_PV(2, acc2)
        INK_PV(3, acc3)
    }

    // Store: element (j >> 2) * 8 + rp * 4 + (j & 3) of group g is dimension 32 g + 2 fc + j.
    INK_UNROLL for (short rp = 0; rp < 2; ++rp) {
        if (!live[rp]) {
            continue;
        }
        const float inv = 1.0f / run_sum[rp];
        device T* o = out + (size_t(qrow[rp]) * size_t(HQ) + size_t(qhead[rp])) * size_t(D) + size_t(2 * fc);
        INK_GATE_ROW
        INK_STORE(0, acc0)
        INK_STORE(1, acc1)
        INK_STORE(2, acc2)
        INK_STORE(3, acc3)
    }
"""

# P V for output dimensions 32 g .. 32 g + 31. One piece: the probabilities of the 32 keys in
# half precision against the values of the same keys. Two pieces: 16 keys at a time, each
# probability as a high and a low half whose sum is the float value, against the values of
# those 16 keys entered twice.
_PV_ONE = """
#define INK_PV(G_, ACC)                                                                  \\
    {                                                                                    \\
        INK_UNROLL for (short part = 0; part < 4; ++part) {                              \\
            InkEight<T> x = ink_load8(v_lane + (voff[part] + uint(32 * (G_)) * v_dim), v_dim); \\
            const short at = (part >> 1) * 16 + (part & 1) * 4;                          \\
            INK_UNROLL for (short i = 0; i < 4; ++i) {                                   \\
                pv_b[at + i] = x.lo[i];                                                  \\
                pv_b[at + 8 + i] = x.hi[i];                                              \\
            }                                                                            \\
        }                                                                                \\
        pv_op.run(pv_a, pv_b, ACC);                                                      \\
    }
"""

_PV_TWO = """
#define INK_PV(G_, ACC)                                                                  \\
    INK_UNROLL for (short kh = 0; kh < 2; ++kh) {                                        \\
        INK_UNROLL for (short rh = 0; rh < 2; ++rh) {                                    \\
            const short part = kh * 2 + rh;                                              \\
            InkEight<T> x = ink_load8(v_lane + (voff[part] + uint(32 * (G_)) * v_dim), v_dim); \\
            INK_UNROLL for (short i = 0; i < 4; ++i) {                                   \\
                pv_b[rh * 4 + i] = x.lo[i];                                              \\
                pv_b[8 + rh * 4 + i] = x.hi[i];                                          \\
                pv_b[16 + rh * 4 + i] = x.lo[i];                                         \\
                pv_b[24 + rh * 4 + i] = x.hi[i];                                         \\
            }                                                                            \\
        }                                                                                \\
        INK_UNROLL for (short i = 0; i < 8; ++i) {                                       \\
            const float w = lg[kh * 8 + i];                                              \\
            const PT high = PT(w);                                                       \\
            pv_a[i] = high;                                                              \\
            pv_a[8 + i] = PT(w - float(high));                                           \\
        }                                                                                \\
        pv_op.run(pv_a, pv_b, ACC);                                                      \\
    }
"""

_STORE = """
#define INK_STORE(G_, ACC)                                                               \\
    INK_UNROLL for (short j = 0; j < 8; ++j) {                                           \\
        o[32 * (G_) + j] = T(ACC[(j >> 2) * 8 + rp * 4 + (j & 3)] * inv);                \\
    }
"""

# The gated store: the attention result in the query type, then times the sigmoid of the row's
# gate logit, as the layer applies it (``h.astype(float32) * sigmoid(gate.astype(float32))``,
# rounded to the query type); the sigmoid in its stable form, 1 / (1 + e^|x|) mirrored at 0.
_STORE_GATED = """
#define INK_STORE(G_, ACC)                                                               \\
    INK_UNROLL for (short j = 0; j < 8; ++j) {                                           \\
        const T hv = T(ACC[(j >> 2) * 8 + rp * 4 + (j & 3)] * inv);                      \\
        const float gx = float(gp[32 * (G_) + j]);                                       \\
        const float tail = 1.0f / (1.0f + metal::exp(metal::abs(gx)));                   \\
        o[32 * (G_) + j] = T(float(hv) * (gx < 0.0f ? tail : 1.0f - tail));              \\
    }
"""


def _float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


SCALE_BITS = _float_bits(HEAD_DIM ** -0.5)


@lru_cache(maxsize=None)
def _kernel(psplit, gated=False):
    # The gate is the gate product's own output, row-contiguous: a row is HQ * D logits.
    gate_row = ("#define INK_GATE_ROW const device T* gp = gate + (size_t(qrow[rp]) * size_t(HQ) + "
                "size_t(qhead[rp])) * size_t(D) + size_t(2 * fc);\n") if gated else "#define INK_GATE_ROW\n"
    return mx.fast.metal_kernel(
        name="mtplx_inkling_prompt_flash" + ("_split" if psplit else "") + ("_gated" if gated else ""),
        input_names=["q", "k", "v", "prof", "sink", "params"] + (["gate"] if gated else []),
        output_names=["out"],
        header=_HEADER + (_PV_TWO if psplit else _PV_ONE) + (_STORE_GATED if gated else _STORE) + gate_row,
        source=_SOURCE,
        ensure_row_contiguous=False,
    )


@lru_cache(maxsize=1)
def tensor_units():
    """Whether this Mac has the tensor units (and macOS) the launch is written for."""
    from mtplx.nax_detect import nax_available

    return bool(nax_available())


def settings():
    """The launch's measured choices, each overridable for measurement:
    ``(tg_m, psplit, ptype, qreg, tgbar, relax, fold)``."""
    env = os.environ.get
    tg_m = int(env("MTPLX_INKLING_PROMPT_FLASH_TGM", TG_M))
    psplit = env("MTPLX_INKLING_PROMPT_FLASH_PSPLIT", "0") in ("1", "on", "true")
    ptype = mx.float32 if env("MTPLX_INKLING_PROMPT_FLASH_PTYPE", "float") == "float" else mx.float16
    qreg = env("MTPLX_INKLING_PROMPT_FLASH_QREG", "1") not in ("0", "off", "false")
    tgbar = env("MTPLX_INKLING_PROMPT_FLASH_TGBAR", "1") not in ("0", "off", "false")
    relax = env("MTPLX_INKLING_PROMPT_FLASH_RELAX", "1") not in ("0", "off", "false")
    fold = env("MTPLX_INKLING_PROMPT_FLASH_FOLD", "1" if FOLD else "0") in ("1", "on", "true")
    return tg_m, psplit, ptype, qreg, tgbar, relax, fold


def decline(query, key, value, profiles, sink, window):
    """None when one launch can serve this prompt call, else a short reason."""
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return "device"
    if not tensor_units():
        return "no_tensor_units"
    if query.dtype not in (mx.bfloat16, mx.float16) or any(
            x.dtype != query.dtype for x in (key, value, profiles, sink)):
        return "dtype"
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4 or profiles.ndim != 4 or sink.ndim != 3:
        return "geometry"
    batch, heads, rows, width = query.shape
    kv_heads, keys = key.shape[1], key.shape[2]
    if (batch != 1 or width != HEAD_DIM or key.shape[0] != 1 or key.shape[3] != HEAD_DIM
            or tuple(value.shape) != tuple(key.shape) or not kv_heads or heads % kv_heads
            or not 0 < rows <= keys or profiles.shape[:3] != (1, rows, heads)
            or tuple(sink.shape) != (1, rows, heads) or window < 0):
        return "geometry"
    group = heads // kv_heads
    if 16 // group < 1:
        return "geometry"
    return None


def flash_attention(query, key, value, profiles, sink, window, *, tg_m=None, psplit=None, ptype=None,
                    qreg=None, tgbar=None, relax=None, fold=None, gate=None):
    """The prompt call's attention, ``(1, heads, rows, 128)`` (a view of a ``(1, rows, heads,
    128)`` array), in one launch. With ``gate`` (the gate projection's ``(1, rows, heads * 128)``
    logits) it returns the gated result ``(1, rows, heads * 128)`` the output projection reads.
    Call only where :func:`decline` returned None."""
    d = settings()
    tg_m = d[0] if tg_m is None else tg_m
    psplit = d[1] if psplit is None else psplit
    ptype = d[2] if ptype is None else ptype
    qreg = d[3] if qreg is None else qreg
    tgbar = d[4] if tgbar is None else tgbar
    relax = d[5] if relax is None else relax
    fold = d[6] if fold is None else fold
    if psplit:
        ptype = mx.float16
    if gate is not None:
        gate = mx.contiguous(gate)
    _, heads, rows, _ = query.shape
    kv_heads, keys = key.shape[1], key.shape[2]
    group = heads // kv_heads
    rpb = 16 // group
    if fold and tg_m % group:
        raise ValueError("one head a block takes a multiple of the heads a KV head serves")
    tg_rows = (tg_m // group) * 16 if fold else tg_m * rpb
    (out,) = _kernel(bool(psplit), gate is not None)(
        inputs=[query, key, value, profiles, sink, mx.array([rows, keys], mx.int32)] + ([gate] if gate is not None else []),
        template=[("T", query.dtype), ("PT", ptype), ("G", group), ("RPB", rpb), ("TG_M", tg_m),
                  ("HQ", heads), ("WINDOW", int(window)), ("EXTENT", int(profiles.shape[3])),
                  ("SCALE", SCALE_BITS), ("RELAX", bool(relax)), ("PSPLIT", bool(psplit)),
                  ("QREG", bool(qreg)), ("TGBAR", bool(tgbar)), ("FOLD", bool(fold))],
        grid=(-(-rows // tg_rows) * tg_m * 32, kv_heads, 1),
        threadgroup=(tg_m * 32, 1, 1),
        output_shapes=[(1, rows, heads, HEAD_DIM)],
        output_dtypes=[query.dtype],
    )
    if gate is not None:
        return out.reshape(1, rows, heads * HEAD_DIM)
    return out.transpose(0, 2, 1, 3)


# ----------------------------------------------------------------------------------------
# The prompt chunk's attention inputs in one launch: what the layer computes between its
# projections and the attention (``model.Layer.attention``), for every row of the chunk:
#   q   = clip(q_proj * q_scale, 22.6), then on a global visit times the row's log-position
#         factor tau = 1 + alpha * log(max(position / n_floor, 1)), rounded after each step;
#   rel = clip(r_proj * r_scale, 8.0), the same factor on a global visit;
#   k   = clip(k after its convolution, 22.6);  v = v after its convolution / its RMS (no gain);
#   the bias profile rel @ rel_basis (1,024 distances a head), rounded to the query type.
# clip(x, L) = x * min(1, L / (|x| + 1e-6)) with |x| the vector's Euclidean length, in float.
# The length and the RMS are sums of squares over a head (128 or 16 values) in this launch's own
# order; everything else is the layer's arithmetic in the layer's order with the library's
# precise square root, reciprocal square root and logarithm.
# ----------------------------------------------------------------------------------------

_PREP_SOURCE = """
    constexpr int D = 128;
    const int row = int(threadgroup_position_in_grid.x);
    const uint tid = thread_position_in_threadgroup.x;
    const int sg = int(tid >> 5);
    const int lane = int(tid & 31u);
    const float qlim = as_type<float>(uint(QLIM));
    const float rlim = as_type<float>(uint(RLIM));
    const float eps = as_type<float>(uint(EPS));
    const float alpha = as_type<float>(uint(ALPHA));
    float tau = 1.0f;
    if (GLOBAL) {
        const float where = float(int(params[0]) + row + 1) / float(NFLOOR);
        tau = 1.0f + alpha * metal::precise::log(metal::max(where, 1.0f));
    }
    threadgroup float relf[HQ * 16];

    const device T* qrow = qp + size_t(row) * size_t(qp_strides[1]);
    const device T* rrow = rp + size_t(row) * size_t(rp_strides[1]);
    const device T* krow = kp + size_t(row) * size_t(kp_strides[1]);
    const device T* vrow = vp + size_t(row) * size_t(vp_strides[1]);
    const size_t fq = size_t(qp_strides[2]);
    const size_t fr = size_t(rp_strides[2]);
    const size_t fk = size_t(kp_strides[2]);
    const size_t fv = size_t(vp_strides[2]);

    // Queries: heads sg, sg + 8, sg + 16; a lane holds dimensions 4 lane .. 4 lane + 3.
    for (int h = sg; h < HQ; h += 8) {
        float x[4];
        float sq = 0.0f;
        for (int i = 0; i < 4; ++i) {
            const int at = h * D + 4 * lane + i;
            x[i] = float(qrow[size_t(at) * fq]) * float(qs[at]);
            const float s2 = x[i] * x[i];
            sq += s2;
        }
        sq = simd_sum(sq);
        const float keep = metal::min(1.0f, qlim / (metal::precise::sqrt(sq) + 1e-6f));
        for (int i = 0; i < 4; ++i) {
            T t = T(x[i] * keep);
            if (GLOBAL) {
                t = T(float(t) * tau);
            }
            qo[(size_t(row) * HQ + h) * D + 4 * lane + i] = t;
        }
    }
    // Keys and values: head sg of the KV_H key/value heads.
    for (int h = sg; h < KV_H; h += 8) {
        float x[4];
        float y[4];
        float sk = 0.0f;
        float sv = 0.0f;
        for (int i = 0; i < 4; ++i) {
            const int at = h * D + 4 * lane + i;
            x[i] = float(krow[size_t(at) * fk]);
            y[i] = float(vrow[size_t(at) * fv]);
            const float k2 = x[i] * x[i];
            const float v2 = y[i] * y[i];
            sk += k2;
            sv += v2;
        }
        sk = simd_sum(sk);
        sv = simd_sum(sv);
        const float keep = metal::min(1.0f, qlim / (metal::precise::sqrt(sk) + 1e-6f));
        const float unit = metal::precise::rsqrt(sv / float(D) + eps);
        for (int i = 0; i < 4; ++i) {
            ko[(size_t(row) * KV_H + h) * D + 4 * lane + i] = T(x[i] * keep);
            vo[(size_t(row) * KV_H + h) * D + 4 * lane + i] = VNORM ? T(y[i] * unit) : T(y[i]);
        }
    }
    // Relative heads: 16 values a head on lanes 0 .. 15, heads sg, sg + 8, sg + 16.
    for (int h = sg; h < HQ; h += 8) {
        const bool mine = lane < 16;
        const float x = mine ? float(rrow[size_t(h * 16 + lane) * fr]) * float(rs[h * 16 + lane]) : 0.0f;
        const float x2 = x * x;
        const float sq = simd_sum(x2);
        const float keep = metal::min(1.0f, rlim / (metal::precise::sqrt(sq) + 1e-6f));
        if (mine) {
            T t = T(x * keep);
            if (GLOBAL) {
                t = T(float(t) * tau);
            }
            relf[h * 16 + lane] = float(t);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The profile: entry (head, distance) = sum over the 16 relative values of rel * basis.
    for (int d = int(tid); d < EXT; d += int(threads_per_threadgroup.x)) {
        float b[16];
        for (int j = 0; j < 16; ++j) {
            b[j] = float(basis[size_t(j) * size_t(basis_strides[0]) + size_t(d) * size_t(basis_strides[1])]);
        }
        for (int h = 0; h < HQ; ++h) {
            float acc = 0.0f;
            for (int j = 0; j < 16; ++j) {
                acc += relf[h * 16 + j] * b[j];
            }
            po[(size_t(row) * HQ + h) * EXT + d] = T(acc);
        }
    }
"""


def _bits(value):
    return _float_bits(float(value))


@lru_cache(maxsize=None)
def _prep_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_inkling_prompt_prep",
        input_names=["qp", "rp", "kp", "vp", "qs", "rs", "basis", "params"],
        output_names=["qo", "ko", "vo", "po"],
        source=_PREP_SOURCE,
        ensure_row_contiguous=False,
    )


def prep(q_proj, r_proj, k_conv, v_conv, q_scale, r_scale, rel_basis, offset, *, glob, cfg):
    """``(q (1, R, 24, 128), k (1, R, 8, 128), v (1, R, 8, 128), profiles (1, R, 24, 1024))`` from the
    projections of a prompt chunk (``q_proj (1, R, 3072)``, ``r_proj (1, R, 384)``, and the key and
    value rows after their convolutions, ``(1, R, 1024)``); ``offset`` is the visit's cache length
    before the chunk (it sets the log-position factor of a global visit)."""
    rows = q_proj.shape[1]
    heads, extent = q_scale.shape[0], rel_basis.shape[1]
    kv_heads = k_conv.shape[-1] // HEAD_DIM
    dtype = q_proj.dtype
    qo, ko, vo, po = _prep_kernel()(
        inputs=[q_proj, r_proj, k_conv, v_conv, q_scale, r_scale, rel_basis, mx.array([int(offset)], mx.int32)],
        template=[("T", dtype), ("HQ", heads), ("KV_H", kv_heads), ("EXT", extent),
                  ("QLIM", _bits(cfg["qk_l2_clip_threshold"])), ("RLIM", _bits(cfg["rel_l2_clip_threshold"])),
                  ("EPS", _bits(cfg["rms_norm_eps"])), ("ALPHA", _bits(cfg["log_scaling_alpha"])),
                  ("NFLOOR", int(cfg["log_scaling_n_floor"])), ("GLOBAL", bool(glob)),
                  ("VNORM", bool(cfg["use_value_norm_no_gamma"]))],
        grid=(rows * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, rows, heads, HEAD_DIM), (1, rows, kv_heads, HEAD_DIM), (1, rows, kv_heads, HEAD_DIM),
                       (1, rows, heads, extent)],
        output_dtypes=[dtype, dtype, dtype, dtype],
    )
    return qo, ko, vo, po


# ----------------------------------------------------------------------------------------
# The caches a prompt chunk writes.
#
# A global visit's bank grows by the chunk's rows every chunk: the library's cache copies the
# whole bank into a larger one each time (a 512-row chunk at history H copies 2 x H rows a visit;
# over a 1M-token read that is tens of terabytes moved). The prompt's length is known before the
# read (``MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS``, set by the prompt read), so the bank is sized for
# it once, the history it already holds copied once, and every chunk then writes its rows in
# place. The same rows at the same positions: what the attention reads is unchanged.
#
# A window visit's cache keeps the last 1,023 rows and appends the chunk (the library's
# ``RotatingKVCache._update_concat``: a slice and a concatenation a tensor). One launch here
# moves the same values into the same new arrays.
# ----------------------------------------------------------------------------------------

#: A prompt length beyond this is not believed (a stale or wrong value sizes nothing).
MAX_PROMPT = 2 * 1024 * 1024


def _prompt_tokens():
    try:
        return int(os.environ.get("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "0"))
    except ValueError:
        return 0


def reserve_bank(kv, rows_k):
    """Size a global visit's cache for the whole prompt before a chunk would make it grow; returns
    whether it did. ``rows_k`` is the chunk's keys ``(1, heads, rows, dim)``."""
    keys = kv.keys
    need = kv.offset + rows_k.shape[2]
    if keys is not None and need <= keys.shape[2]:
        return False
    total = _prompt_tokens()
    if not need <= total <= MAX_PROMPT:
        return False
    step = getattr(kv, "step", 256)
    cap = -(-total // step) * step
    shape = (1, rows_k.shape[1], cap, rows_k.shape[3])
    new_k, new_v = mx.zeros(shape, rows_k.dtype), mx.zeros(shape, rows_k.dtype)
    if keys is not None and kv.offset:
        new_k[..., : kv.offset, :] = keys[..., : kv.offset, :]
        new_v[..., : kv.offset, :] = kv.values[..., : kv.offset, :]
    kv.keys, kv.values = new_k, new_v
    return True


_WINDOW_SOURCE = """
    const uint d = thread_position_in_grid.x;
    const int r = int(thread_position_in_grid.y);
    const int h = int(thread_position_in_grid.z);
    const int start = int(params[0]);
    const int kept = int(params[1]) - start;
    const int n_out = kept + int(params[2]);
    const size_t at = (size_t(h) * size_t(n_out) + size_t(r)) * size_t(D) + size_t(d);
    if (r < kept) {
        const int s = start + r;
        ko[at] = kold[size_t(h) * size_t(kold_strides[1]) + size_t(s) * size_t(kold_strides[2]) + size_t(d) * size_t(kold_strides[3])];
        vo[at] = vold[size_t(h) * size_t(vold_strides[1]) + size_t(s) * size_t(vold_strides[2]) + size_t(d) * size_t(vold_strides[3])];
    } else {
        const int n = r - kept;
        ko[at] = knew[size_t(h) * size_t(knew_strides[1]) + size_t(n) * size_t(knew_strides[2]) + size_t(d) * size_t(knew_strides[3])];
        vo[at] = vnew[size_t(h) * size_t(vnew_strides[1]) + size_t(n) * size_t(vnew_strides[2]) + size_t(d) * size_t(vnew_strides[3])];
    }
"""


@lru_cache(maxsize=1)
def _window_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_inkling_window_rows",
        input_names=["kold", "vold", "knew", "vnew", "params"],
        output_names=["ko", "vo"],
        source=_WINDOW_SOURCE,
        ensure_row_contiguous=False,
    )


def window_update(kv, keys, values):
    """``kv.update_and_fetch(keys, values)`` for a multi-row chunk on a window cache, with the new
    window built by one launch; the library's own update where its window is not in time order
    (a ring rotated by decode) or it keeps leading rows."""
    old = kv.keys
    if (old is None or keys.shape[2] < 2 or getattr(kv, "keep", 0) != 0 or kv._idx != old.shape[2]
            or kv.values.shape != old.shape or old.dtype != keys.dtype):
        return kv.update_and_fetch(keys, values)
    start = max(0, kv._idx - kv.max_size + 1)
    rows = keys.shape[2]
    n_out = old.shape[2] - start + rows
    heads, dim = keys.shape[1], keys.shape[3]
    new_k, new_v = _window_kernel()(
        inputs=[old, kv.values, keys, values, mx.array([start, old.shape[2], rows], mx.int32)],
        template=[("D", dim)],
        grid=(dim, n_out, heads),
        threadgroup=(dim, 1, 1),
        output_shapes=[(1, heads, n_out, dim), (1, heads, n_out, dim)],
        output_dtypes=[keys.dtype, keys.dtype],
    )
    kv.keys, kv.values = new_k, new_v
    kv.offset += rows
    kv._idx = n_out
    return new_k, new_v


_prep_checks = {}


def _prep_first_use(layer, dtype, glob):
    """Once per dtype and visit kind: the launch against the layer's own expressions (``model``'s
    ``clip_heads``, ``norm``, the log-position factor and the profile product) on synthetic
    projections with this layer's scales and basis: relative RMS within the query type's rounding."""
    from .. import model as M

    spec = (str(dtype), bool(glob))
    if spec in _prep_checks:
        return _prep_checks[spec]
    cfg = layer.cfg
    rows, offset = 96, (9000 if glob else 3)
    ks = mx.random.split(mx.random.key(20261122), 4)
    qp = (mx.random.normal((1, rows, 3072), key=ks[0]) * 0.8).astype(dtype)
    rp = (mx.random.normal((1, rows, 384), key=ks[1]) * 1.5).astype(dtype)
    kc = (mx.random.normal((1, rows, 1024), key=ks[2]) * 1.2).astype(dtype)
    vc = (mx.random.normal((1, rows, 1024), key=ks[3]) * 1.2).astype(dtype)
    width = layer.q_scale.shape[-1]
    q = M.clip_heads(qp.reshape(1, rows, -1, width), cfg["qk_l2_clip_threshold"], layer.q_scale)
    rel = M.clip_heads(rp.reshape(1, rows, layer.q_scale.shape[0], -1), cfg["rel_l2_clip_threshold"], layer.r_scale)
    k = M.clip_heads(kc.reshape(1, rows, -1, width), cfg["qk_l2_clip_threshold"])
    v = vc.reshape(1, rows, -1, width)
    if cfg["use_value_norm_no_gamma"]:
        v = M.norm(v, None, cfg["rms_norm_eps"]).astype(dtype)
    if glob:
        pos = mx.arange(rows) + offset + 1
        tau = 1 + cfg["log_scaling_alpha"] * mx.log(mx.maximum(pos / cfg["log_scaling_n_floor"], 1))
        q = (q.astype(mx.float32) * tau[:, None, None]).astype(q.dtype)
        rel = (rel.astype(mx.float32) * tau[:, None, None]).astype(rel.dtype)
    prof = rel @ layer.rel_basis.astype(rel.dtype)
    bound = 4e-3 if dtype == mx.bfloat16 else 5e-4
    try:
        got = prep(qp, rp, kc, vc, layer.q_scale, layer.r_scale, layer.rel_basis, offset, glob=glob, cfg=cfg)
        worst = 0.0
        for a, b in zip(got, (q, k, v, prof)):
            d = a.astype(mx.float32) - b.astype(mx.float32)
            r = mx.sqrt(mx.mean(d * d)) / mx.maximum(mx.sqrt(mx.mean(b.astype(mx.float32) ** 2)), 1e-30)
            ok = mx.all(mx.isfinite(a.astype(mx.float32)))
            mx.eval(r, ok)
            worst = max(worst, float(r.item()) if bool(ok.item()) else float("inf"))
        reason = None if worst <= bound else "prep_first_use_numbers"
    except (RuntimeError, ValueError, TypeError):
        reason = "prep_first_use_error"
    _prep_checks[spec] = reason
    return reason


def stage(layer, q_input, k_input, v_input, kv, conv, linear):
    """``Layer.attention`` for a prompt chunk (more than 32 rows, not a verify panel) on our
    launches around the layer's own projections and convolutions: the attention inputs and the
    bias profile in one launch (:func:`prep`), the cache written as the layer writes it, the
    attention and its gate in one launch, the output projection. ``None`` (counted) where the
    switch is off or the stage declines; the caller then runs the layer's own path."""
    from .. import attention as A

    if os.environ.get(A._FLASH_ENV, "off").strip().lower() not in ("1", "on", "true", "yes"):
        return None
    visit = "window" if layer.sliding else "full"
    dtype = q_input.dtype
    reason = None
    if os.environ.get("MTPLX_INKLING_ATTENTION", "stock") != "stock":
        reason = "attention_mode"
    elif os.environ.get("MTPLX_INKLING_PROMPT_ATTENTION", "fused") == "stock":
        reason = "forced_stock"
    elif mx.default_device() != mx.gpu or not mx.metal.is_available() or not tensor_units():
        reason = "no_tensor_units"
    elif dtype not in (mx.bfloat16, mx.float16):
        reason = "dtype"
    elif (q_input.ndim != 3 or q_input.shape[0] != 1 or layer.q_scale.shape[-1] != HEAD_DIM
            or layer.rel_basis.shape[0] != 16 or layer.r_scale.shape != (layer.q_scale.shape[0], 16)):
        reason = "geometry"
    if reason is None:
        reason = _prep_first_use(layer, dtype, not layer.sliding)
    if reason is None:
        reason = A._flash_first_use(dtype, layer.sliding)
    A._note_route(f"prompt_stage.{visit}." + ("selected" if reason is None else "stock." + reason))
    if reason is not None:
        return None
    cfg = layer.cfg
    offset = kv.offset
    qp = linear(q_input, layer.q)
    rp = linear(q_input, layer.r)
    kc = layer.convs[0](linear(k_input, layer.k), conv, 0, k_input, None)
    vc = layer.convs[1](linear(v_input, layer.v), conv, 1, v_input, None)
    q, k, v, prof = prep(qp, rp, kc, vc, layer.q_scale, layer.r_scale, layer.rel_basis, offset,
                         glob=not layer.sliding, cfg=cfg)
    k_rows, v_rows = k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)
    if layer.sliding:
        keys, values = window_update(kv, k_rows, v_rows)
        keys, values = kv._temporal_order(keys), kv._temporal_order(values)
    else:
        reserve_bank(kv, k_rows)
        keys, values = kv.update_and_fetch(k_rows, v_rows)
    window = cfg["sliding_window_size"] if layer.sliding else 0
    sink = linear(q_input, layer.sink)
    gate = linear(q_input, layer.gate)
    h = flash_attention(q.transpose(0, 2, 1, 3), keys, values, prof, sink, window, gate=gate)
    return linear(h, layer.out)


def reference(query, key, value, profiles, sink, window):
    """The definition in float32 with MLX's own ops (for tests and checks; small calls only)."""
    batch, heads, rows, width = query.shape
    kv_heads, keys = key.shape[1], key.shape[2]
    group = heads // kv_heads
    extent = profiles.shape[3]
    q = query.astype(mx.float32)
    k = mx.repeat(key.astype(mx.float32), group, axis=1)
    v = mx.repeat(value.astype(mx.float32), group, axis=1)
    scores = (q @ k.transpose(0, 1, 3, 2)) * (width ** -0.5)
    p = (keys - rows) + mx.arange(rows)[:, None]
    d = p - mx.arange(keys)[None, :]
    prof = profiles.astype(mx.float32).transpose(0, 2, 1, 3)          # (1, heads, rows, extent)
    bias = mx.take_along_axis(prof, mx.broadcast_to(mx.clip(d, 0, extent - 1)[None, None],
                                                    (batch, heads, rows, keys)), axis=-1)
    bias = mx.where(d >= extent, 0.0, bias)
    seen = (d >= 0) if not window else ((d >= 0) & (d < window))
    logits = mx.where(seen, scores + bias, -mx.inf)
    s = sink.astype(mx.float32).transpose(0, 2, 1)[..., None]           # (1, heads, rows, 1)
    top = mx.maximum(mx.max(logits, axis=-1, keepdims=True), s)
    w = mx.exp(logits - top)
    denom = w.sum(axis=-1, keepdims=True) + mx.exp(s - top)
    return (w @ v) / denom
