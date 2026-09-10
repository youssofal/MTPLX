"""Exact fixed-M4 q4 routed-down reduction for Qwen4 Flash-Next."""

from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx


ROWS = 4
TOP_K = 10
K = 640
HIDDEN = 2560
THREADS = 64
OUTPUTS_PER_THREADGROUP = 8

#: One routed-reduce kernel per affine group size (32 Optimized-Speed, 64
#: Bare-Speed); GROUP_SIZE is the only constexpr that changes. The shared-add
#: tail kernels below operate on already-computed activations and carry no
#: group size, so they are compiled once.
_ROUTED_KERNEL: dict[int, Any] = {}
_TAIL_KERNEL: Any | None = None
_RESIDUAL_TAIL_KERNEL: Any | None = None


def _header(group_size: int = 32) -> str:
    return f"""
    #include <metal_simdgroup>
    #include <metal_stdlib>
    using namespace metal;

    constant constexpr uint ROWS = {ROWS};
    constant constexpr uint TOP_K = {TOP_K};
    constant constexpr uint K = {K};
    constant constexpr uint HIDDEN = {HIDDEN};
    constant constexpr uint GROUP_SIZE = {group_size};
    constant constexpr uint VALUES_PER_THREAD = 8;
    constant constexpr uint BLOCK_SIZE = VALUES_PER_THREAD * 32;
    constant constexpr uint OUTPUTS_PER_SIMD = 4;
    constant constexpr uint OUTPUTS_PER_THREADGROUP = 8;
    constant constexpr uint WEIGHT_BYTES_PER_ROW = K / 2;
    constant constexpr uint GROUPS_PER_ROW = K / GROUP_SIZE;
    constant constexpr ushort SLOT_ORDER[TOP_K] = {{0, 8, 1, 9, 2, 3, 4, 5, 6, 7}};

    inline float load_q4_vector(
        const device bfloat* x,
        thread float* x_thread) {{
        float sum = 0.0f;
        for (int i = 0; i < VALUES_PER_THREAD; i += 4) {{
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            x_thread[i] = x[i];
            x_thread[i + 1] = x[i + 1] / 16.0f;
            x_thread[i + 2] = x[i + 2] / 256.0f;
            x_thread[i + 3] = x[i + 3] / 4096.0f;
        }}
        return sum;
    }}

    inline float load_q4_vector_safe(
        const device bfloat* x,
        thread float* x_thread,
        int count) {{
        float sum = 0.0f;
        for (int i = 0; i < count; i += 4) {{
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            x_thread[i] = x[i];
            x_thread[i + 1] = x[i + 1] / 16.0f;
            x_thread[i + 2] = x[i + 2] / 256.0f;
            x_thread[i + 3] = x[i + 3] / 4096.0f;
        }}
        for (int i = count; i < VALUES_PER_THREAD; ++i) {{
            x_thread[i] = 0.0f;
        }}
        return sum;
    }}

    inline float qdot_q4(
        const device uchar* w,
        const thread float* x_thread,
        float scale,
        float bias,
        float sum) {{
        float accum = 0.0f;
        const device ushort* ws = (const device ushort*)w;
        for (int i = 0; i < VALUES_PER_THREAD / 4; ++i) {{
            accum +=
                (x_thread[4 * i] * (ws[i] & 0x000f) +
                 x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
                 x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
                 x_thread[4 * i + 3] * (ws[i] & 0xf000));
        }}
        return scale * accum + sum * bias;
    }}

    inline float qdot_q4_safe(
        const device uchar* w,
        const thread float* x_thread,
        float scale,
        float bias,
        float sum,
        int count) {{
        float accum = 0.0f;
        const device ushort* ws = (const device ushort*)w;
        for (int i = 0; i < count / 4; ++i) {{
            accum +=
                (x_thread[4 * i] * (ws[i] & 0x000f) +
                 x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
                 x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
                 x_thread[4 * i + 3] * (ws[i] & 0xf000));
        }}
        return scale * accum + sum * bias;
    }}
"""


_ROUTED_SOURCE = """
    const uint output_tile = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint simd_group = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint output_base =
        output_tile * OUTPUTS_PER_THREADGROUP
        + simd_group * OUTPUTS_PER_SIMD;

    bfloat pending[OUTPUTS_PER_SIMD];
    bfloat routed_value[OUTPUTS_PER_SIMD];

    for (uint order_index = 0; order_index < TOP_K; ++order_index) {
        const uint slot = SLOT_ORDER[order_index];
        const uint expert = expert_ids[row * TOP_K + slot];
        const device bfloat* x =
            routed_h + (row * TOP_K + slot) * K + lane * VALUES_PER_THREAD;
        const device uchar* w = (const device uchar*)weights
            + ((size_t)expert * HIDDEN + output_base) * WEIGHT_BYTES_PER_ROW
            + lane * sizeof(uint);
        const device bfloat* scale = scales
            + ((size_t)expert * HIDDEN + output_base) * GROUPS_PER_ROW
            + lane / (GROUP_SIZE / VALUES_PER_THREAD);
        const device bfloat* bias = biases
            + ((size_t)expert * HIDDEN + output_base) * GROUPS_PER_ROW
            + lane / (GROUP_SIZE / VALUES_PER_THREAD);

        float result[OUTPUTS_PER_SIMD] = {0.0f};
        int k = 0;
        for (; k < int(K - BLOCK_SIZE); k += BLOCK_SIZE) {
            float x_thread[VALUES_PER_THREAD];
            float sum = load_q4_vector(x, x_thread);
            for (uint out = 0; out < OUTPUTS_PER_SIMD; ++out) {
                const device uchar* output_w =
                    w + out * WEIGHT_BYTES_PER_ROW;
                const device bfloat* output_scale =
                    scale + out * GROUPS_PER_ROW;
                const device bfloat* output_bias =
                    bias + out * GROUPS_PER_ROW;
                result[out] += qdot_q4(
                    output_w,
                    x_thread,
                    float(output_scale[0]),
                    float(output_bias[0]),
                    sum);
            }
            w += BLOCK_SIZE / 2;
            scale += BLOCK_SIZE / GROUP_SIZE;
            bias += BLOCK_SIZE / GROUP_SIZE;
            x += BLOCK_SIZE;
        }

        const int remaining = clamp(
            int(K) - k - int(lane * VALUES_PER_THREAD),
            0,
            int(VALUES_PER_THREAD));
        if (remaining > 0) {
            float x_thread[VALUES_PER_THREAD];
            float sum = load_q4_vector_safe(x, x_thread, remaining);
            for (uint out = 0; out < OUTPUTS_PER_SIMD; ++out) {
                const device uchar* output_w =
                    w + out * WEIGHT_BYTES_PER_ROW;
                const device bfloat* output_scale =
                    scale + out * GROUPS_PER_ROW;
                const device bfloat* output_bias =
                    bias + out * GROUPS_PER_ROW;
                result[out] += qdot_q4_safe(
                    output_w,
                    x_thread,
                    float(output_scale[0]),
                    float(output_bias[0]),
                    sum,
                    remaining);
            }
        }

        for (uint out = 0; out < OUTPUTS_PER_SIMD; ++out) {
            result[out] = simd_sum(result[out]);
        }
        if (lane == 0) {
            for (uint out = 0; out < OUTPUTS_PER_SIMD; ++out) {
                bfloat down_value = bfloat(result[out]);
                bfloat product = bfloat(
                    float(down_value) * float(route_scores[row * TOP_K + slot]));
                if (order_index == 0 || order_index == 2) {
                    pending[out] = product;
                } else if (order_index == 1) {
                    routed_value[out] = bfloat(float(pending[out]) + float(product));
                } else if (order_index == 3) {
                    bfloat second = bfloat(float(pending[out]) + float(product));
                    routed_value[out] = bfloat(float(second) + float(routed_value[out]));
                } else {
                    routed_value[out] = bfloat(float(product) + float(routed_value[out]));
                }
            }
        }
    }

    if (lane == 0) {
        for (uint out = 0; out < OUTPUTS_PER_SIMD; ++out) {
            routed_down[row * HIDDEN + output_base + out] = routed_value[out];
        }
    }
"""


_TAIL_SOURCE = f"""
    using namespace metal;
    constexpr uint ROWS = {ROWS};
    constexpr uint HIDDEN = {HIDDEN};

    uint index = thread_position_in_grid.x;
    if (index >= ROWS * HIDDEN) {{
        return;
    }}
    uint row = index / HIDDEN;
    bfloat gated_shared = bfloat(
        float(shared_factor[row]) * float(shared_down[index]));
    output[index] =
        bfloat(float(routed_down[index]) + float(gated_shared));
"""


_RESIDUAL_TAIL_SOURCE = f"""
    using namespace metal;
    constexpr uint ROWS = {ROWS};
    constexpr uint HIDDEN = {HIDDEN};
    constexpr uint HYPER_STREAMS = 4;

    uint index = thread_position_in_grid.x;
    if (index >= ROWS * HIDDEN) {{
        return;
    }}
    uint row = index / HIDDEN;
    uint column = index % HIDDEN;
    bfloat gated_shared = bfloat(
        float(shared_factor[row]) * float(shared_down[index]));
    bfloat block_out = bfloat(
        float(routed_down[index]) + float(gated_shared));
    for (uint stream = 0; stream < HYPER_STREAMS; ++stream) {{
        uint hidden_index =
            row * HYPER_STREAMS * HIDDEN + stream * HIDDEN + column;
        bfloat inject_value = inject[row * HYPER_STREAMS + stream];
        bfloat product = bfloat(
            float(block_out) * float(inject_value));
        output[hidden_index] = bfloat(
            float(hyper[hidden_index]) + float(product));
    }}
"""


def source(group_size: int = 32) -> str:
    """Return the exact fixed-shape q4 QMV plus routed-reduction source."""

    return _header(group_size) + _ROUTED_SOURCE


def tail_source() -> str:
    """Return the separate exact shared-branch addition tail."""

    return _TAIL_SOURCE


def residual_tail_source() -> str:
    """Return the exact shared-add plus hyper-residual tail source."""

    return _RESIDUAL_TAIL_SOURCE


def launch_geometry() -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    return (
        (HIDDEN // OUTPUTS_PER_THREADGROUP * THREADS, ROWS, 1),
        (THREADS, 1, 1),
    )


def _routed_reduce_kernel(group_size: int) -> Any:
    """Compile-and-cache the routed-down reduce kernel for a group size."""

    kernel = _ROUTED_KERNEL.get(group_size)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"mtplx_qwen4_m4_routed_down_reduce_gs{group_size}",
            input_names=[
                "routed_h",
                "weights",
                "scales",
                "biases",
                "expert_ids",
                "route_scores",
            ],
            output_names=["routed_down"],
            header=_header(group_size),
            source=_ROUTED_SOURCE,
            ensure_row_contiguous=True,
        )
        _ROUTED_KERNEL[group_size] = kernel
    return kernel


def bind(group_size: int = 32) -> Callable[..., mx.array]:
    """Bind the routed-only reduction and separate shared-add tail.

    ``group_size`` selects the affine group the pack quantized the routed
    down projection to (32 Optimized-Speed, 64 Bare-Speed).
    """

    global _TAIL_KERNEL
    routed_kernel = _routed_reduce_kernel(group_size)
    if _TAIL_KERNEL is None:
        _TAIL_KERNEL = mx.fast.metal_kernel(
            name="mtplx_qwen4_m4_routed_shared_tail",
            input_names=[
                "routed_down",
                "shared_down",
                "shared_factor",
            ],
            output_names=["output"],
            source=_TAIL_SOURCE,
            ensure_row_contiguous=True,
        )
    tail_kernel = _TAIL_KERNEL
    grid, threadgroup = launch_geometry()

    def routed_down_reduce(
        routed_h,
        weights,
        scales,
        biases,
        expert_ids,
        route_scores,
        shared_down,
        shared_factor,
    ):
        (routed_down,) = routed_kernel(
            inputs=[
                routed_h,
                weights,
                scales,
                biases,
                expert_ids,
                route_scores,
            ],
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=[(ROWS, HIDDEN)],
            output_dtypes=[mx.bfloat16],
        )
        (output,) = tail_kernel(
            inputs=[routed_down, shared_down, shared_factor],
            grid=(ROWS * HIDDEN, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(ROWS, HIDDEN)],
            output_dtypes=[mx.bfloat16],
        )
        return output

    return routed_down_reduce


def bind_residual_tail(group_size: int = 32) -> Callable[..., mx.array]:
    """Bind routed reduction plus the combined shared and residual tail.

    ``group_size`` selects the routed down projection's affine group (32
    Optimized-Speed, 64 Bare-Speed).
    """

    global _RESIDUAL_TAIL_KERNEL
    routed_kernel = _routed_reduce_kernel(group_size)
    if _RESIDUAL_TAIL_KERNEL is None:
        _RESIDUAL_TAIL_KERNEL = mx.fast.metal_kernel(
            name="mtplx_qwen4_m4_routed_shared_residual_tail",
            input_names=[
                "routed_down",
                "shared_down",
                "shared_factor",
                "hyper",
                "inject",
            ],
            output_names=["output"],
            source=_RESIDUAL_TAIL_SOURCE,
            ensure_row_contiguous=True,
        )
    residual_tail_kernel = _RESIDUAL_TAIL_KERNEL
    grid, threadgroup = launch_geometry()

    def routed_down_residual_tail(
        routed_h,
        weights,
        scales,
        biases,
        expert_ids,
        route_scores,
        shared_down,
        shared_factor,
        hyper,
        inject,
    ):
        (routed_down,) = routed_kernel(
            inputs=[
                routed_h,
                weights,
                scales,
                biases,
                expert_ids,
                route_scores,
            ],
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=[(ROWS, HIDDEN)],
            output_dtypes=[mx.bfloat16],
        )
        (output,) = residual_tail_kernel(
            inputs=[routed_down, shared_down, shared_factor, hyper, inject],
            grid=(ROWS * HIDDEN, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(ROWS, 4 * HIDDEN)],
            output_dtypes=[mx.bfloat16],
        )
        return output.reshape(*hyper.shape)

    return routed_down_residual_tail


__all__ = [
    "bind",
    "bind_residual_tail",
    "launch_geometry",
    "residual_tail_source",
    "source",
    "tail_source",
]
