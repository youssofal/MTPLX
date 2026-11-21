"""Round routed activations to the formats declared in the native training checkpoint.

NVFP4 has a second, per-token scale. Keep it outside the matrix product
so rounding its product into a BF16 activation does not change the expert.
"""

from functools import lru_cache

import mlx.core as mx


def _e4m3(x):
    magnitude = mx.minimum(mx.abs(x), 448.0)
    unit = mx.power(
        2.0, mx.maximum(-9, mx.floor(mx.log2(mx.maximum(magnitude, 2**-6))) - 3)
    )
    return mx.sign(x) * mx.round(magnitude / unit) * unit


def _e2m1(x):
    # Midpoints alternate closed/open because the low mantissa wins ties-to-even.
    a = mx.abs(x)
    code = (a > 0.25).astype(mx.int32)
    for midpoint, closed in (
        (0.75, True),
        (1.25, False),
        (1.75, True),
        (2.5, False),
        (3.5, True),
        (5.0, False),
    ):
        code = code + ((a >= midpoint) if closed else (a > midpoint))
    return mx.sign(x) * mx.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])[code]


@lru_cache(maxsize=1)
def _quantizer():
    return mx.fast.metal_kernel(
        name="mtplx_inkling_expert_inputs",
        input_names=["input", "peaks"],
        output_names=["values"],
        header="""
            inline float inkling_e4m3(float x) {
                float a = min(abs(x), 448.0f);
                float unit = ldexp(1.0f, max(-9, ilogb(max(a, 0.015625f)) - 3));
                return copysign(rint(a / unit) * unit, x);
            }
            inline float inkling_e2m1(float x) {
                float a = abs(x);
                float y = a > 5 ? 6 : a >= 3.5f ? 4 : a > 2.5f ? 3 : a >= 1.75f ? 2 :
                          a > 1.25f ? 1.5f : a >= .75f ? 1 : a > .25f ? .5f : 0;
                return copysign(y, x);
            }
        """,
        source="""
            uint i = thread_position_in_grid.x;
            uint lane = thread_index_in_simdgroup;
            float x = float(input[i]);
            if (FP4) {
                float peak = peaks[i / WIDTH];
                float outer = peak / 1536.0f;
                uint first = lane & 16u;
                float top = abs(x);
                for (uint d = 1; d < 16; d *= 2)
                    top = max(top, simd_shuffle_xor(top, d));
                float encoder = peak > 0 ? 1536.0f / peak : 0;
                float s6 = inkling_e4m3(top * (encoder / 6.0f));
                float s4 = inkling_e4m3(top * (encoder / 4.0f));
                float a = s6 == 0 ? 0 : inkling_e2m1(x / (s6 * outer)) * s6;
                float b = s4 == 0 ? 0 : inkling_e2m1(x / (s4 * outer)) * s4;
                float ae = abs(a * outer - x), be = abs(b * outer - x);
                for (uint d = 1; d < 16; d *= 2) {
                    ae += simd_shuffle_xor(ae, d);
                    be += simd_shuffle_xor(be, d);
                }
                values[i] = simd_shuffle(be, first) < simd_shuffle(ae, first) ? b : a;
            } else {
                float top = simd_max(abs(x));
                float scale = top > 0 ? exp2(ceil(log2(top / 448.0f))) : 1.0f;
                values[i] = inkling_e4m3(x / scale) * scale;
            }
        """,
    )


def expert_inputs(x, mode, *, metal=True):
    """Return FP32 accumulation inputs and the outer scale for each input row."""
    fp4 = mode == "nvfp4"
    width = x.shape[-1]
    if mode not in {"nvfp4", "mxfp8"} or width % 32:
        raise ValueError(f"Unsupported Inkling expert format {mode} at width {width}")
    peaks = (
        mx.max(mx.abs(x), axis=-1, keepdims=True).astype(mx.float32)
        if fp4
        else mx.zeros((1,), mx.float32)
    )
    outer = peaks / 1536 if fp4 else None
    if metal and mx.default_device() == mx.gpu and mx.metal.is_available():
        values = _quantizer()(
            inputs=[x, peaks],
            template=[("WIDTH", width), ("FP4", fp4)],
            grid=(x.size, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[mx.float32],
        )[0]
        return values, outer
    blocks = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 16 if fp4 else 32)
    high = mx.max(mx.abs(blocks), axis=-1, keepdims=True)
    if not fp4:
        scale = mx.where(high > 0, mx.power(2.0, mx.ceil(mx.log2(high / 448))), 1)
        return (_e4m3(blocks / scale) * scale).reshape(x.shape), None
    encoder = mx.where(peaks > 0, 1536 / peaks, 0)[..., None]
    outer_block = outer[..., None]

    def candidate(limit):
        scale = _e4m3(high * (encoder / limit))
        divisor = scale * outer_block
        rounded = _e2m1(blocks / mx.where(divisor != 0, divisor, 1)) * scale
        error = mx.sum(mx.abs(rounded * outer_block - blocks), axis=-1, keepdims=True)
        return rounded, error

    four, error4 = candidate(4)
    six, error6 = candidate(6)
    return mx.where(error4 < error6, four, six).reshape(x.shape), outer
