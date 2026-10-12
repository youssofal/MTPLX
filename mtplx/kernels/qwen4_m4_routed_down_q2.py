"""Exact fixed-M4 routed-down reduction, affine q2/group-64 experts.

The q2/g64 twin of ``qwen4_m4_routed_down`` (q4/g32): the same grids, slot
order, BF16 reduction boundaries and residual tail, with the weight decode of
MLX 0.32.2's affine bits=2 ``gather_qmv`` -- the kernel MLX dispatches for
this pack at K=640 (640 is not a multiple of the 512-value fast block): 16
values = 4 bytes per lane, one full 512-value block, then a 16-value ``_safe``
tail for lanes 0..7, byte masks 0x03/0x0c/0x30/0xc0, x scaled by 1/4/16/64.
The shared-add and residual tails carry no weights and are the q4 module's.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mlx.core as mx

from .qwen4_m4_routed_down import _RESIDUAL_TAIL_SOURCE, _TAIL_SOURCE

ROWS = 4
TOP_K = 10
K = 640
HIDDEN = 2560
THREADS = 64
OUTPUTS_PER_THREADGROUP = 8

_ROUTED_KERNEL: Any | None = None
_TAIL_KERNEL: Any | None = None
_RESIDUAL_TAIL_KERNEL: Any | None = None
_PARALLEL_RESIDUAL_KERNEL: Any | None = None


_HEADER = f"""
    #include <metal_simdgroup>
    #include <metal_stdlib>
    using namespace metal;

    constant constexpr uint ROWS = {ROWS};
    constant constexpr uint TOP_K = {TOP_K};
    constant constexpr uint K = {K};
    constant constexpr uint HIDDEN = {HIDDEN};
    constant constexpr uint GROUP_SIZE = 64;
    constant constexpr uint VALUES_PER_THREAD = 16;
    constant constexpr uint BLOCK_SIZE = VALUES_PER_THREAD * 32;
    constant constexpr uint OUTPUTS_PER_SIMD = 4;
    constant constexpr uint OUTPUTS_PER_THREADGROUP = 8;
    constant constexpr uint WEIGHT_BYTES_PER_ROW = K / 4;
    constant constexpr uint GROUPS_PER_ROW = K / GROUP_SIZE;
    constant constexpr ushort SLOT_ORDER[TOP_K] = {{0, 8, 1, 9, 2, 3, 4, 5, 6, 7}};

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

    inline float load_q2_vector_safe(
        const device bfloat* x,
        thread float* x_thread,
        int count) {{
        float sum = 0.0f;
        for (int i = 0; i < count; i += 4) {{
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            x_thread[i] = x[i];
            x_thread[i + 1] = x[i + 1] / 4.0f;
            x_thread[i + 2] = x[i + 2] / 16.0f;
            x_thread[i + 3] = x[i + 3] / 64.0f;
        }}
        for (int i = count; i < VALUES_PER_THREAD; ++i) {{
            x_thread[i] = 0.0f;
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

    inline float qdot_q2_safe(
        const device uchar* w,
        const thread float* x_thread,
        float scale,
        float bias,
        float sum,
        int count) {{
        float accum = 0.0f;
        const device uchar* ws = w;
        for (int i = 0; i < count / 4; ++i) {{
            accum +=
                (x_thread[4 * i] * (ws[i] & 0x03) +
                 x_thread[4 * i + 1] * (ws[i] & 0x0c) +
                 x_thread[4 * i + 2] * (ws[i] & 0x30) +
                 x_thread[4 * i + 3] * (ws[i] & 0xc0));
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
            float sum = load_q2_vector(x, x_thread);
            for (uint out = 0; out < OUTPUTS_PER_SIMD; ++out) {
                const device uchar* output_w =
                    w + out * WEIGHT_BYTES_PER_ROW;
                const device bfloat* output_scale =
                    scale + out * GROUPS_PER_ROW;
                const device bfloat* output_bias =
                    bias + out * GROUPS_PER_ROW;
                result[out] += qdot_q2(
                    output_w,
                    x_thread,
                    float(output_scale[0]),
                    float(output_bias[0]),
                    sum);
            }
            w += BLOCK_SIZE / 4;
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
            float sum = load_q2_vector_safe(x, x_thread, remaining);
            for (uint out = 0; out < OUTPUTS_PER_SIMD; ++out) {
                const device uchar* output_w =
                    w + out * WEIGHT_BYTES_PER_ROW;
                const device bfloat* output_scale =
                    scale + out * GROUPS_PER_ROW;
                const device bfloat* output_bias =
                    bias + out * GROUPS_PER_ROW;
                result[out] += qdot_q2_safe(
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




#: The routed reduction and the residual tail in one dispatch, with the ten
#: slots' dot products in parallel simdgroups: each (slot, output) dot is
#: _ROUTED_SOURCE's code, lane for lane, so it keeps its bits; the combine
#: then runs in SLOT_ORDER with the same bfloat16 boundaries, and the tail is
#: _RESIDUAL_TAIL_SOURCE's arithmetic. Five simdgroups own two slots each.
PARALLEL_SIMDGROUPS = 5
PARALLEL_THREADS = 32 * PARALLEL_SIMDGROUPS

_PARALLEL_RESIDUAL_SOURCE = """
    const uint output_tile = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint simd_group = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint output_base = output_tile * OUTPUTS_PER_SIMD;
    threadgroup float dots[TOP_K][OUTPUTS_PER_SIMD];

    for (uint pass = 0; pass < TOP_K / 5; ++pass) {
        const uint slot = simd_group + pass * 5;
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
            float sum = load_q2_vector(x, x_thread);
            for (uint out = 0; out < OUTPUTS_PER_SIMD; ++out) {
                result[out] += qdot_q2(
                    w + out * WEIGHT_BYTES_PER_ROW,
                    x_thread,
                    float(scale[out * GROUPS_PER_ROW]),
                    float(bias[out * GROUPS_PER_ROW]),
                    sum);
            }
            w += BLOCK_SIZE / 4;
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
            float sum = load_q2_vector_safe(x, x_thread, remaining);
            for (uint out = 0; out < OUTPUTS_PER_SIMD; ++out) {
                result[out] += qdot_q2_safe(
                    w + out * WEIGHT_BYTES_PER_ROW,
                    x_thread,
                    float(scale[out * GROUPS_PER_ROW]),
                    float(bias[out * GROUPS_PER_ROW]),
                    sum,
                    remaining);
            }
        }
        for (uint out = 0; out < OUTPUTS_PER_SIMD; ++out) {
            result[out] = simd_sum(result[out]);
        }
        if (lane == 0) {
            for (uint out = 0; out < OUTPUTS_PER_SIMD; ++out) {
                dots[slot][out] = result[out];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint out = simd_group * 32 + lane;
    if (out >= OUTPUTS_PER_SIMD) {
        return;
    }
    bfloat pending = bfloat(0.0f);
    bfloat routed_value = bfloat(0.0f);
    for (uint order_index = 0; order_index < TOP_K; ++order_index) {
        const uint slot = SLOT_ORDER[order_index];
        bfloat down_value = bfloat(dots[slot][out]);
        bfloat product = bfloat(
            float(down_value) * float(route_scores[row * TOP_K + slot]));
        if (order_index == 0 || order_index == 2) {
            pending = product;
        } else if (order_index == 1) {
            routed_value = bfloat(float(pending) + float(product));
        } else if (order_index == 3) {
            bfloat second = bfloat(float(pending) + float(product));
            routed_value = bfloat(float(second) + float(routed_value));
        } else {
            routed_value = bfloat(float(product) + float(routed_value));
        }
    }

    const uint column = output_base + out;
    const uint index = row * HIDDEN + column;
    bfloat gated_shared = bfloat(
        float(shared_factor[row]) * float(shared_down[index]));
    bfloat block_out = bfloat(
        float(routed_value) + float(gated_shared));
    for (uint stream = 0; stream < 4; ++stream) {
        uint hidden_index = row * 4 * HIDDEN + stream * HIDDEN + column;
        bfloat inject_value = inject[row * 4 + stream];
        bfloat product = bfloat(
            float(block_out) * float(inject_value));
        output[hidden_index] = bfloat(
            float(hyper[hidden_index]) + float(product));
    }
"""


def parallel_residual_source() -> str:
    """The one-dispatch routed reduction plus residual tail."""

    return _HEADER + _PARALLEL_RESIDUAL_SOURCE


def parallel_launch_geometry() -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """640 threadgroups of four outputs per row, five simdgroups each."""

    return (
        (HIDDEN // 4 * PARALLEL_THREADS, ROWS, 1),
        (PARALLEL_THREADS, 1, 1),
    )


def source() -> str:
    """Return the exact fixed-shape q2/g64 QMV plus routed-reduction source."""

    return _HEADER + _ROUTED_SOURCE


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


def bind() -> Callable[..., mx.array]:
    """Bind the routed-only reduction and separate shared-add tail."""

    global _ROUTED_KERNEL, _TAIL_KERNEL
    if _ROUTED_KERNEL is None:
        _ROUTED_KERNEL = mx.fast.metal_kernel(
            name="mtplx_qwen4_m4_routed_down_reduce_q2g64",
            input_names=[
                "routed_h",
                "weights",
                "scales",
                "biases",
                "expert_ids",
                "route_scores",
            ],
            output_names=["routed_down"],
            header=_HEADER,
            source=_ROUTED_SOURCE,
            ensure_row_contiguous=True,
        )
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
    routed_kernel = _ROUTED_KERNEL
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


def bind_residual_tail() -> Callable[..., mx.array]:
    """Bind the routed reduction and the combined shared and residual tail.

    One dispatch (_PARALLEL_RESIDUAL_SOURCE): the two-kernel spelling below it
    in the file computes the same bits with a second launch and ten serial
    slot passes per simdgroup.
    """

    global _PARALLEL_RESIDUAL_KERNEL
    if _PARALLEL_RESIDUAL_KERNEL is None:
        _PARALLEL_RESIDUAL_KERNEL = mx.fast.metal_kernel(
            name="mtplx_qwen4_m4_routed_down_parallel_residual_tail_q2g64",
            input_names=[
                "routed_h",
                "weights",
                "scales",
                "biases",
                "expert_ids",
                "route_scores",
                "shared_down",
                "shared_factor",
                "hyper",
                "inject",
            ],
            output_names=["output"],
            header=_HEADER,
            source=_PARALLEL_RESIDUAL_SOURCE,
            ensure_row_contiguous=True,
        )
    kernel = _PARALLEL_RESIDUAL_KERNEL
    grid, threadgroup = parallel_launch_geometry()

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
        (output,) = kernel(
            inputs=[
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
            ],
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=[(ROWS, 4 * HIDDEN)],
            output_dtypes=[mx.bfloat16],
        )
        return output.reshape(*hyper.shape)

    return routed_down_residual_tail


def bind_residual_tail_two_dispatch() -> Callable[..., mx.array]:
    """The two-dispatch spelling of ``bind_residual_tail`` (its parent)."""

    global _ROUTED_KERNEL, _RESIDUAL_TAIL_KERNEL
    if _ROUTED_KERNEL is None:
        _ROUTED_KERNEL = mx.fast.metal_kernel(
            name="mtplx_qwen4_m4_routed_down_reduce_q2g64",
            input_names=[
                "routed_h",
                "weights",
                "scales",
                "biases",
                "expert_ids",
                "route_scores",
            ],
            output_names=["routed_down"],
            header=_HEADER,
            source=_ROUTED_SOURCE,
            ensure_row_contiguous=True,
        )
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
    routed_kernel = _ROUTED_KERNEL
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
    "bind_residual_tail_two_dispatch",
    "launch_geometry",
    "parallel_launch_geometry",
    "parallel_residual_source",
    "residual_tail_source",
    "source",
    "tail_source",
]
