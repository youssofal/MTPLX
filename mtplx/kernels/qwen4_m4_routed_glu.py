"""Exact fixed-M4 paired routed gate/up producer for Qwen4 Flash-Next.

The kernel specializes MLX 0.32.2's affine q4/group-32
``gather_qmv_fast`` arithmetic for the installed physical-M4 pack.  Each
threadgroup pairs fused-pack rows ``j`` and ``640 + j`` so the hidden-input
tile is loaded once, then applies the stock BF16 sigmoid/SILU/product
boundaries and emits only the existing routed activation tensor.
"""

from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx


ROWS = 4
TOP_K = 10
HIDDEN = 2560
INTERMEDIATE = 640
THREADS = 64
OUTPUTS_PER_SIMD = 4
OUTPUTS_PER_THREADGROUP = 8

#: One compiled kernel per affine group size (32 for Optimized-Speed, 64 for
#: Bare-Speed). GROUP_SIZE is the only constexpr that changes: the 4-bit
#: packing is group-size-independent (WEIGHT_BYTES_PER_ROW = HIDDEN / 2) and
#: every scale/bias stride derives from GROUP_SIZE.
_KERNEL: dict[int, Any] = {}


def _header(group_size: int = 32) -> str:
    return f"""
    #include <metal_simdgroup>
    #include <metal_stdlib>
    using namespace metal;

    constant constexpr uint ROWS = {ROWS};
    constant constexpr uint TOP_K = {TOP_K};
    constant constexpr uint HIDDEN = {HIDDEN};
    constant constexpr uint INTERMEDIATE = {INTERMEDIATE};
    constant constexpr uint FUSED_OUTPUTS = 2 * INTERMEDIATE;
    constant constexpr uint GROUP_SIZE = {group_size};
    constant constexpr uint VALUES_PER_THREAD = 16;
    constant constexpr uint BLOCK_SIZE = VALUES_PER_THREAD * 32;
    constant constexpr uint OUTPUTS_PER_SIMD = 4;
    constant constexpr uint OUTPUTS_PER_THREADGROUP = 8;
    constant constexpr uint WEIGHT_BYTES_PER_ROW = HIDDEN / 2;
    constant constexpr uint GROUPS_PER_ROW = HIDDEN / GROUP_SIZE;

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
"""


_SOURCE = """
    const uint output_tile = threadgroup_position_in_grid.x;
    const uint selected = threadgroup_position_in_grid.y;
    const uint simd_group = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint output_base =
        output_tile * OUTPUTS_PER_THREADGROUP
        + simd_group * OUTPUTS_PER_SIMD;
    const uint row = selected / TOP_K;
    const uint expert = expert_ids[selected];

    const device bfloat* x = value + row * HIDDEN
        + lane * VALUES_PER_THREAD;
    const size_t expert_weight_base =
        (size_t)expert * FUSED_OUTPUTS * WEIGHT_BYTES_PER_ROW;
    const size_t expert_metadata_base =
        (size_t)expert * FUSED_OUTPUTS * GROUPS_PER_ROW;
    const device uchar* gate_weights = (const device uchar*)weights
        + expert_weight_base
        + (size_t)output_base * WEIGHT_BYTES_PER_ROW
        + lane * (VALUES_PER_THREAD / 2);
    const device uchar* up_weights = (const device uchar*)weights
        + expert_weight_base
        + (size_t)(INTERMEDIATE + output_base) * WEIGHT_BYTES_PER_ROW
        + lane * (VALUES_PER_THREAD / 2);
    const device bfloat* gate_scales = scales
        + expert_metadata_base
        + (size_t)output_base * GROUPS_PER_ROW
        + lane / (GROUP_SIZE / VALUES_PER_THREAD);
    const device bfloat* gate_biases = biases
        + expert_metadata_base
        + (size_t)output_base * GROUPS_PER_ROW
        + lane / (GROUP_SIZE / VALUES_PER_THREAD);
    const device bfloat* up_scales = scales
        + expert_metadata_base
        + (size_t)(INTERMEDIATE + output_base) * GROUPS_PER_ROW
        + lane / (GROUP_SIZE / VALUES_PER_THREAD);
    const device bfloat* up_biases = biases
        + expert_metadata_base
        + (size_t)(INTERMEDIATE + output_base) * GROUPS_PER_ROW
        + lane / (GROUP_SIZE / VALUES_PER_THREAD);

    float gate_result[OUTPUTS_PER_SIMD] = {0.0f};
    float up_result[OUTPUTS_PER_SIMD] = {0.0f};
    for (uint k = 0; k < HIDDEN; k += BLOCK_SIZE) {
        float x_thread[VALUES_PER_THREAD];
        const float sum = load_q4_vector(x, x_thread);
        for (uint out = 0; out < OUTPUTS_PER_SIMD; ++out) {
            gate_result[out] += qdot_q4(
                gate_weights + out * WEIGHT_BYTES_PER_ROW,
                x_thread,
                float(gate_scales[out * GROUPS_PER_ROW]),
                float(gate_biases[out * GROUPS_PER_ROW]),
                sum);
            up_result[out] += qdot_q4(
                up_weights + out * WEIGHT_BYTES_PER_ROW,
                x_thread,
                float(up_scales[out * GROUPS_PER_ROW]),
                float(up_biases[out * GROUPS_PER_ROW]),
                sum);
        }
        x += BLOCK_SIZE;
        gate_weights += BLOCK_SIZE / 2;
        up_weights += BLOCK_SIZE / 2;
        gate_scales += BLOCK_SIZE / GROUP_SIZE;
        gate_biases += BLOCK_SIZE / GROUP_SIZE;
        up_scales += BLOCK_SIZE / GROUP_SIZE;
        up_biases += BLOCK_SIZE / GROUP_SIZE;
    }

    for (uint out = 0; out < OUTPUTS_PER_SIMD; ++out) {
        gate_result[out] = simd_sum(gate_result[out]);
        up_result[out] = simd_sum(up_result[out]);
    }
    if (lane == 0) {
        for (uint out = 0; out < OUTPUTS_PER_SIMD; ++out) {
            bfloat gate_value = bfloat(gate_result[out]);
            bfloat up_value = bfloat(up_result[out]);
            auto sigmoid_y = 1 / (1 + metal::exp(metal::abs(gate_value)));
            bfloat sigmoid_value = gate_value < bfloat(0.0f)
                ? bfloat(sigmoid_y) : bfloat(1 - sigmoid_y);
            bfloat silu = bfloat(gate_value * sigmoid_value);
            routed_h[selected * INTERMEDIATE + output_base + out] =
                bfloat(silu * up_value);
        }
    }
"""


def source(group_size: int = 32) -> str:
    """Return the exact paired q4 QMV and BF16 GLU source for a group size."""

    return _header(group_size) + _SOURCE


def launch_geometry() -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return the fixed 3,200-threadgroup physical-M4 geometry."""

    return (
        (INTERMEDIATE // OUTPUTS_PER_THREADGROUP * THREADS, ROWS * TOP_K, 1),
        (THREADS, 1, 1),
    )


def bind(group_size: int = 32) -> Callable[..., mx.array]:
    """Bind the construction-validated paired routed-GU producer.

    ``group_size`` selects the affine group the pack quantized the routed
    experts to (32 for Optimized-Speed, 64 for Bare-Speed); one kernel is
    compiled and cached per group size.
    """

    kernel = _KERNEL.get(group_size)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"mtplx_qwen4_m4_paired_routed_glu_gs{group_size}",
            input_names=[
                "value",
                "weights",
                "scales",
                "biases",
                "expert_ids",
            ],
            output_names=["routed_h"],
            header=_header(group_size),
            source=_SOURCE,
            ensure_row_contiguous=True,
        )
        _KERNEL[group_size] = kernel
    grid, threadgroup = launch_geometry()

    def routed_glu(value, weights, scales, biases, expert_ids):
        (routed_h,) = kernel(
            inputs=[value, weights, scales, biases, expert_ids],
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=[(ROWS, TOP_K, INTERMEDIATE)],
            output_dtypes=[mx.bfloat16],
        )
        return routed_h

    return routed_glu


__all__ = ["bind", "launch_geometry", "source"]
