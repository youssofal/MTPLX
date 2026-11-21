"""Inkling 2 Mini attention output for a decode panel in one launch: projection, convolution, residual.

Stage C of the fast decode step (SIGNATURES.md, lane L2). From the gated attention result
``hg (1, W, 3072)`` of ``attend``:

    attn    = hg @ out.T                         (bf16, 2304 outputs)
    conv    = (attn_P + sum_j taps_j * w_j) / sqrt(5)   over positions P-3 .. P (bf16)
    partial = f32(conv) + f32(x)                 (an ordinary layer)
            = f32(conv)                          (a boundary layer: L0 closes the block)

Exactness, class A by construction:

* the projection repeats MLX 0.32.2's one-row GEMV for this shape
  (``gemv_bfloat16_bm4_bn1_sm1_sn32_tm4_tn4``): a simdgroup owns four output rows, lane
  ``l`` owns inputs ``128 i + 4 l .. + 3`` for steps ``i = 0 .. 23`` and accumulates them in
  that order, then a shuffle-down tree (16, 8, 4, 2, 1) sums the lanes. Every panel row
  gets that exact sequence while each weight is read once per panel, so a verify row
  equals its one-row decode and both equal RC6's projection;
* the convolution repeats ``short_conv``'s statements: taps oldest first from the
  committed float32 tail ``(1, 3, 2304)`` and the panel's own rows, ``acc += h * w``,
  ``(h_P + acc) / sqrt(5)`` with the float32 divisor, one cast; the products of two
  bf16-valued floats are exact, so the fused and unfused forms agree;
* the residual add is one float32 add, as RC6's.

Returns ``partial (1, W, 2304) f32`` and the convolution's input rows of the panel
``(1, W, 2304) f32`` (the projection, as RC6 would push it into the tail). Nothing is
written into state: L0 commits the accepted rows of the second output into the tail.
"""

from __future__ import annotations

import math
import struct
from functools import lru_cache

import mlx.core as mx

from .attend import MAX_ROWS

_SOURCE = """
    const int group = int(threadgroup_position_in_grid.x);
    const int sg = int(simdgroup_index_in_threadgroup);
    const int lane = int(thread_index_in_simdgroup);
    const int hidden = int(wo_shape[1]);           // 3072 gated values a row
    const int width = int(wo_shape[0]);            // 2304 outputs
    const int first = group * 16 + sg * 4;         // this simdgroup's four outputs
    const size_t pitch = wo_strides[0];
    const device T* weights = wo + size_t(first) * pitch;

    // The projection of every panel row at once. A lane holds four consecutive gated values of
    // each 128-wide span and walks the spans in order; a sum starts at zero and takes one product
    // a statement, so each row repeats the one-row product's sequence while every weight is read
    // once for the whole panel.
    float sum[W][4];
    for (int w = 0; w < W; ++w) {
        for (int o = 0; o < 4; ++o) {
            sum[w][o] = 0;
        }
    }
    for (int span = 0; span < hidden / 128; ++span) {
        const int col = span * 128 + 4 * lane;
        float gated[W][4];
        for (int w = 0; w < W; ++w) {
            for (int k = 0; k < 4; ++k) {
                gated[w][k] = static_cast<float>(hg[w * hg_strides[1] + col + k]);
            }
        }
        for (int o = 0; o < 4; ++o) {
            T wt[4];
            for (int k = 0; k < 4; ++k) {
                wt[k] = weights[o * pitch + col + k];
            }
            for (int w = 0; w < W; ++w) {
                for (int k = 0; k < 4; ++k) {
                    sum[w][o] += wt[k] * gated[w][k];
                }
            }
        }
    }
    // The 32 lanes' sums meet in lane 0 by halving: a lane adds the lane 16 above it, then 8, 4,
    // 2 and 1.
    for (int w = 0; w < W; ++w) {
        for (int o = 0; o < 4; ++o) {
            for (ushort hop = 16; hop != 0; hop /= 2) {
                sum[w][o] += simd_shuffle_down(sum[w][o], hop);
            }
        }
    }
    if (lane != 0) {
        return;
    }
    const float divisor = as_type<float>(uint(DIVISOR));
    for (int o = 0; o < 4; ++o) {
        const int c = first + o;
        float hist[W + 3];
        for (int t = 0; t < 3; ++t) {
            hist[t] = tail[t * tail_strides[1] + c * tail_strides[2]];
        }
        for (int w = 0; w < W; ++w) {
            hist[3 + w] = static_cast<float>(static_cast<T>(sum[w][o]));
        }
        const float w0 = cw[0 * cw_strides[0] + c * cw_strides[1]];
        const float w1 = cw[1 * cw_strides[0] + c * cw_strides[1]];
        const float w2 = cw[2 * cw_strides[0] + c * cw_strides[1]];
        const float w3 = cw[3 * cw_strides[0] + c * cw_strides[1]];
        for (int w = 0; w < W; ++w) {
            const size_t at = size_t(w) * size_t(width) + size_t(c);
            rows[at] = hist[3 + w];
            float acc = 0.0f;
            acc += hist[w + 0] * w0;
            acc += hist[w + 1] * w1;
            acc += hist[w + 2] * w2;
            acc += hist[w + 3] * w3;
            const float total = hist[w + 3] + acc;
            const float value = total / divisor;
            const float conv = static_cast<float>(static_cast<T>(value));
            if (BOUNDARY) {
                partial[at] = conv;
            } else {
                partial[at] = conv + static_cast<float>(x[w * x_strides[1] + c * x_strides[2]]);
            }
        }
    }
"""


def _float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


#: The float32 RC6 divides a static four-tap convolution by.
DIVISOR_BITS = _float_bits(math.sqrt(5))


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="mtplx_inkling_fast_attn_output",
        input_names=["hg", "wo", "cw", "tail", "x"],
        output_names=["partial", "rows"],
        source=_SOURCE,
        ensure_row_contiguous=False,
    )


def attention_output(hg, out_weight, conv_weight, tail, x, *, boundary):
    """``(partial, conv_rows)``, both ``(1, W, 2304)`` float32. See the module docstring.

    ``out_weight (2304, 3072)`` bf16 is ``layer.out``; ``conv_weight (4, 2304)`` float32 is
    ``layer.convs[2].weight`` (read through its strides, no copy); ``tail (1, 3, 2304)``
    float32 is the committed attention-convolution tail, oldest first; ``x (1, W, 2304)``
    is the visit input.
    """
    rows = hg.shape[1]
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"A panel holds 1 to {MAX_ROWS} rows, got {rows}")
    out_size, in_size = out_weight.shape
    if out_size % 16 or in_size % 128 or conv_weight.shape != (4, out_size):
        raise ValueError("attention_output serves the 2304 x 3072 projection with four taps")
    partial, conv_rows = _kernel()(
        inputs=[hg, out_weight, conv_weight, tail, x],
        template=[("T", hg.dtype), ("W", rows), ("BOUNDARY", bool(boundary)), ("DIVISOR", DIVISOR_BITS)],
        grid=((out_size // 16) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, rows, out_size), (1, rows, out_size)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return partial, conv_rows
