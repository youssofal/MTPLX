"""Project a short panel with one weight read and the same reduction for each input row."""

from functools import lru_cache
import mlx.core as mx

from .precision import AffineWeight


# Four consecutive values of a weight or state row as floats, from one load (8 bytes for a 16-bit
# array, 16 for float32). The same floats one-element loads give: a 16-bit value widens exactly.
_LOAD4 = """
inline void rp_load4(const device float* p, thread float* o) {
    const float4 v = *(const device float4*)p;
    o[0] = v.x; o[1] = v.y; o[2] = v.z; o[3] = v.w;
}
inline void rp_load4(const device half* p, thread float* o) {
    const half4 v = *(const device half4*)p;
    o[0] = float(v.x); o[1] = float(v.y); o[2] = float(v.z); o[3] = float(v.w);
}
inline void rp_load4(const device bfloat16_t* p, thread float* o) {
    const ushort4 v = *(const device ushort4*)p;
    o[0] = as_type<float>(uint(v.x) << 16); o[1] = as_type<float>(uint(v.y) << 16);
    o[2] = as_type<float>(uint(v.z) << 16); o[3] = as_type<float>(uint(v.w) << 16);
}
"""


@lru_cache(maxsize=1)
def _panel_kernel():
    # A lane's four consecutive columns come from one load per row (its weight row and every state
    # row); the multiply-adds keep their order: column by column, each row's sum in turn.
    return mx.fast.metal_kernel(
        name="mtplx_inkling_row_projection",
        input_names=["states", "weight"],
        output_names=["result"],
        header=_LOAD4,
        source="""
            const uint worker = thread_position_in_grid.x % (LANES * PARTS);
            const uint out = thread_position_in_grid.x / (LANES * PARTS);
            threadgroup float pieces[PARTS * ROWS];
            float sums[ROWS] = {};
            for (uint first = 4 * worker; first < WIDTH; first += 4 * LANES * PARTS) {
                if (WIDTH % 4 == 0) {
                    float w4[4];
                    float x4[ROWS][4];
                    rp_load4(weight + out * WIDTH + first, w4);
                    for (uint r = 0; r < ROWS; ++r)
                        rp_load4(states + r * WIDTH + first, x4[r]);
                    for (uint c = 0; c < 4; ++c)
                        for (uint r = 0; r < ROWS; ++r)
                            sums[r] = fma(w4[c], x4[r][c], sums[r]);
                } else {
                    for (uint c = first; c < min(first + 4, uint(WIDTH)); ++c) {
                        const float w = float(weight[out * WIDTH + c]);
                        for (uint r = 0; r < ROWS; ++r)
                            sums[r] = fma(w, float(states[r * WIDTH + c]), sums[r]);
                    }
                }
            }
            for (uint r = 0; r < ROWS; ++r) {
                for (ushort gap = LANES / 2; gap; gap >>= 1)
                    sums[r] += simd_shuffle_down(sums[r], gap);
            }
            if constexpr (PARTS == 1) {
                if (worker == 0)
                    for (uint r = 0; r < ROWS; ++r) result[r * OUTPUTS + out] = OUT(sums[r]);
            } else {
                if (worker % LANES == 0)
                    for (uint r = 0; r < ROWS; ++r) pieces[r * PARTS + worker / LANES] = sums[r];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (worker == 0) {
                    for (uint r = 0; r < ROWS; ++r) {
                        float total = pieces[r * PARTS];
                        for (uint part = 1; part < PARTS; ++part) total += pieces[r * PARTS + part];
                        result[r * OUTPUTS + out] = OUT(total);
                    }
                }
            }
        """,
    )


def panel_linear(x, weight):
    if isinstance(weight, AffineWeight):
        return weight.product(x)
    if (
        weight.ndim != 2
        or not x.ndim
        or not weight.size
        or x.shape[-1] != weight.shape[1]
    ):
        return x @ weight.T
    outputs, width = weight.shape
    rows = x.size // width
    if (
        not 1 <= rows <= 8
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or x.dtype not in (mx.bfloat16, mx.float16, mx.float32)
        or weight.dtype not in (mx.bfloat16, mx.float16, mx.float32)
    ):
        return x @ weight.T
    # Choose how a dot product is split using just the matrix dimensions. Adding a
    # speculative token changes how many sums each thread holds, never their order.
    lanes = 4 if width <= 64 else 32
    parts = 8 if width > 64 and width >= 16 * outputs else 1
    dtype = x.dtype if x.dtype == weight.dtype else mx.float32
    return _panel_kernel()(
        inputs=[x, weight],
        template=[
            ("ROWS", rows),
            ("WIDTH", width),
            ("OUTPUTS", outputs),
            ("LANES", lanes),
            ("PARTS", parts),
            ("OUT", dtype),
        ],
        grid=(outputs * lanes * parts, 1, 1),
        threadgroup=(lanes * parts if parts > 1 else 128, 1, 1),
        output_shapes=[(*x.shape[:-1], outputs)],
        output_dtypes=[dtype],
    )[0]


def stock_messages(queries, states, scale):
    weights = mx.softmax((queries @ states.swapaxes(-1, -2)) * scale, axis=-1)
    return weights @ states


def serial_messages(queries, states, scale):
    """Retain the single-token reduction inside a batched target verification."""
    return mx.concatenate(
        [stock_messages(queries, states[:, row : row + 1], scale) for row in range(states.shape[1])],
        axis=1,
    )
