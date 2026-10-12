"""Exact fixed-M4 paired routed gate/up producer, affine q2/group-64 experts.

The q2/g64 twin of ``qwen4_m4_routed_glu`` (q4/g32): the same threadgroup
geometry, pairing and BF16 GLU epilogue, with the weight decode of MLX
0.32.2's affine bits=2 ``gather_qmv_fast`` (16 values = 4 bytes per lane,
512-value blocks, byte masks 0x03/0x0c/0x30/0xc0, x scaled by 1/4/16/64),
which is the kernel MLX dispatches for this pack at K=2560.  Each
threadgroup pairs fused-pack rows ``j`` and ``640 + j`` so the hidden-input
tile is loaded once, then applies the stock BF16 sigmoid/SILU/product
boundaries and emits only the existing routed activation tensor.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mlx.core as mx

ROWS = 4
TOP_K = 10
HIDDEN = 2560
INTERMEDIATE = 640
THREADS = 64
OUTPUTS_PER_SIMD = 4
OUTPUTS_PER_THREADGROUP = 8

_KERNEL: Any | None = None


_HEADER = f"""
    #include <metal_simdgroup>
    #include <metal_stdlib>
    using namespace metal;

    constant constexpr uint ROWS = {ROWS};
    constant constexpr uint TOP_K = {TOP_K};
    constant constexpr uint HIDDEN = {HIDDEN};
    constant constexpr uint INTERMEDIATE = {INTERMEDIATE};
    constant constexpr uint FUSED_OUTPUTS = 2 * INTERMEDIATE;
    constant constexpr uint GROUP_SIZE = 64;
    constant constexpr uint VALUES_PER_THREAD = 16;
    constant constexpr uint BLOCK_SIZE = VALUES_PER_THREAD * 32;
    constant constexpr uint OUTPUTS_PER_SIMD = 4;
    constant constexpr uint OUTPUTS_PER_THREADGROUP = 8;
    constant constexpr uint WEIGHT_BYTES_PER_ROW = HIDDEN / 4;
    constant constexpr uint GROUPS_PER_ROW = HIDDEN / GROUP_SIZE;

    inline float load_q2_vector(
        const device bfloat* x,
        thread float* x_thread) {{
        float sum = 0.0f;
        for (int i = 0; i < VALUES_PER_THREAD; i += 4) {{
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            x_thread[i] = x[i];
            x_thread[i + 1] = x[i + 1] / 4.0f;
            x_thread[i + 2] = x[i + 2] / 16.0f;
            x_thread[i + 3] = x[i + 3] / 64.0f;
        }}
        return sum;
    }}

    inline float qdot_q2(
        const device uchar* w,
        const thread float* x_thread,
        float scale,
        float bias,
        float sum) {{
        float accum = 0.0f;
        const device uchar* ws = w;
        for (int i = 0; i < VALUES_PER_THREAD / 4; ++i) {{
            accum +=
                (x_thread[4 * i] * (ws[i] & 0x03) +
                 x_thread[4 * i + 1] * (ws[i] & 0x0c) +
                 x_thread[4 * i + 2] * (ws[i] & 0x30) +
                 x_thread[4 * i + 3] * (ws[i] & 0xc0));
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
        + lane * (VALUES_PER_THREAD / 4);
    const device uchar* up_weights = (const device uchar*)weights
        + expert_weight_base
        + (size_t)(INTERMEDIATE + output_base) * WEIGHT_BYTES_PER_ROW
        + lane * (VALUES_PER_THREAD / 4);
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
        const float sum = load_q2_vector(x, x_thread);
        for (uint out = 0; out < OUTPUTS_PER_SIMD; ++out) {
            gate_result[out] += qdot_q2(
                gate_weights + out * WEIGHT_BYTES_PER_ROW,
                x_thread,
                float(gate_scales[out * GROUPS_PER_ROW]),
                float(gate_biases[out * GROUPS_PER_ROW]),
                sum);
            up_result[out] += qdot_q2(
                up_weights + out * WEIGHT_BYTES_PER_ROW,
                x_thread,
                float(up_scales[out * GROUPS_PER_ROW]),
                float(up_biases[out * GROUPS_PER_ROW]),
                sum);
        }
        x += BLOCK_SIZE;
        gate_weights += BLOCK_SIZE / 4;
        up_weights += BLOCK_SIZE / 4;
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


def source() -> str:
    """Return the exact paired q2/g64 QMV and BF16 GLU source."""

    return _HEADER + _SOURCE


def launch_geometry() -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return the fixed 3,200-threadgroup physical-M4 geometry."""

    return (
        (INTERMEDIATE // OUTPUTS_PER_THREADGROUP * THREADS, ROWS * TOP_K, 1),
        (THREADS, 1, 1),
    )


def bind() -> Callable[..., mx.array]:
    """Bind the construction-validated q2/g64 paired routed-GU producer."""

    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="mtplx_qwen4_m4_paired_routed_glu_q2g64",
            input_names=[
                "value",
                "weights",
                "scales",
                "biases",
                "expert_ids",
            ],
            output_names=["routed_h"],
            header=_HEADER,
            source=_SOURCE,
            ensure_row_contiguous=True,
        )
    kernel = _KERNEL
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
