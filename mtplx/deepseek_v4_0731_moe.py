"""Receipt-backed routed-MoE primitives for DeepSeek-V4-Flash-0731.

This module owns only the measured target lane: affine Q2/group-128 routed
gate/up packing and the fixed M1 top-6 row-owned reduction. Construction
validates storage once; installed callables perform no environment reads,
eligibility checks, counters, or fallback routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

from .moe_packed_projections import PackedSwitchGLU, _pack_pair


@dataclass(frozen=True, slots=True)
class RoutedQ2Contract:
    bits: int
    group_size: int
    hidden_size: int
    width: int
    experts: int


class DeepseekV40731PackedQ2SwitchGLU(PackedSwitchGLU):
    """Fixed M1/M3 unsorted gather path for the pinned top-6 lane."""

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        packed = self.gate_up_proj.gather(x, indices, False)
        gate, up = mx.split(packed, [self._split_at], axis=-1)
        routed = self.down_proj(
            self.activation(up, gate),
            indices,
            sorted_indices=False,
        )
        return routed.squeeze(-2)


def validate_routed_q2_pair(
    gate: nn.Module,
    up: nn.Module,
    *,
    hidden_size: int,
    width: int,
    experts: int,
) -> RoutedQ2Contract:
    """Validate the exact affine-Q2 expert-bank layout before packing."""

    hidden = int(hidden_size)
    intermediate = int(width)
    expert_count = int(experts)
    expected_weight = (expert_count, intermediate, hidden // 16)
    expected_meta = (expert_count, intermediate, hidden // 128)
    valid = all(
        (
            hidden > 0,
            intermediate > 0,
            expert_count > 0,
            hidden % 128 == 0,
            isinstance(gate, QuantizedSwitchLinear),
            isinstance(up, QuantizedSwitchLinear),
            int(getattr(gate, "bits", 0) or 0) == 2,
            int(getattr(up, "bits", 0) or 0) == 2,
            int(getattr(gate, "group_size", 0) or 0) == 128,
            int(getattr(up, "group_size", 0) or 0) == 128,
            str(getattr(gate, "mode", "")) == "affine",
            str(getattr(up, "mode", "")) == "affine",
            getattr(getattr(gate, "weight", None), "dtype", None) == mx.uint32,
            getattr(getattr(up, "weight", None), "dtype", None) == mx.uint32,
            tuple(getattr(getattr(gate, "weight", None), "shape", ()))
            == expected_weight,
            tuple(getattr(getattr(up, "weight", None), "shape", ())) == expected_weight,
            tuple(getattr(getattr(gate, "scales", None), "shape", ())) == expected_meta,
            tuple(getattr(getattr(up, "scales", None), "shape", ())) == expected_meta,
            tuple(getattr(getattr(gate, "biases", None), "shape", ())) == expected_meta,
            tuple(getattr(getattr(up, "biases", None), "shape", ())) == expected_meta,
            getattr(getattr(gate, "scales", None), "dtype", None) == mx.bfloat16,
            getattr(getattr(up, "scales", None), "dtype", None) == mx.bfloat16,
            getattr(getattr(gate, "biases", None), "dtype", None) == mx.bfloat16,
            getattr(getattr(up, "biases", None), "dtype", None) == mx.bfloat16,
            "bias" not in gate,
            "bias" not in up,
        )
    )
    if not valid:
        raise ValueError("DeepSeek-V4-0731 routed affine Q2/group-128 contract failed")
    return RoutedQ2Contract(
        bits=2,
        group_size=128,
        hidden_size=hidden,
        width=intermediate,
        experts=expert_count,
    )


def build_routed_q2_pair(
    switch_mlp: nn.Module,
    *,
    hidden_size: int,
    width: int,
    experts: int,
) -> DeepseekV40731PackedQ2SwitchGLU:
    """Pack gate/up output rows without changing any affine input group."""

    validate_routed_q2_pair(
        switch_mlp.gate_proj,
        switch_mlp.up_proj,
        hidden_size=hidden_size,
        width=width,
        experts=experts,
    )
    packed = _pack_pair(switch_mlp.gate_proj, switch_mlp.up_proj, axis=1)
    if isinstance(packed, str):
        raise ValueError(f"DeepSeek-V4-0731 routed Q2 pair failed: {packed}")
    gate_up, split_at = packed
    if int(split_at) != int(width):
        raise ValueError("DeepSeek-V4-0731 routed Q2 pair split is invalid")
    return DeepseekV40731PackedQ2SwitchGLU(
        gate_up,
        switch_mlp.down_proj,
        switch_mlp.activation,
        split_at,
    )


@lru_cache(maxsize=None)
def _row_owned_combine_m1_kernel():
    source = r"""
        uint column = thread_position_in_grid.x;
        int hidden = int(HIDDEN_size);
        if (int(column) >= hidden) { return; }

        bfloat accumulator = bfloat(0.0f);
        _Pragma("unroll")
        for (int expert = 0; expert < 6; ++expert) {
          uint routed_index = uint(expert * hidden) + column;
          bfloat score = bfloat(route_weights[expert]);
          bfloat product = bfloat(
            float(routed[routed_index]) * float(score));
          accumulator = bfloat(float(accumulator) + float(product));
        }
        combined[column] = accumulator;
    """
    return mx.fast.metal_kernel(
        name="mtplx_dsv4_0731_row_owned_combine_top6_bf16",
        input_names=["routed", "route_weights", "HIDDEN_size"],
        output_names=["combined"],
        source=source,
        ensure_row_contiguous=True,
    )


def build_row_owned_combine_m1(
    *,
    hidden_size: int,
    top_k: int,
) -> Callable[[mx.array, mx.array], mx.array]:
    """Build the measured one-output-owner BF16 top-6 reduction."""

    hidden = int(hidden_size)
    if hidden != 4096 or int(top_k) != 6:
        raise ValueError("DeepSeek-V4-0731 row-owned combine geometry is invalid")
    if not mx.metal.is_available():
        raise ValueError("DeepSeek-V4-0731 row-owned combine requires Metal")
    kernel = _row_owned_combine_m1_kernel()

    def combine(routed: mx.array, route_weights: mx.array) -> mx.array:
        (output,) = kernel(
            inputs=[
                routed.reshape(6, hidden),
                route_weights.reshape(6),
                hidden,
            ],
            grid=(hidden, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(1, hidden)],
            output_dtypes=[mx.bfloat16],
        )
        return output

    return combine


def exact_selfcheck_row_owned_combine_m1(
    combine: Callable[[mx.array, mx.array], mx.array],
) -> None:
    """Execute the fixed BF16 reduction oracle once before publication."""

    if not callable(combine):
        raise ValueError("DeepSeek-V4-0731 row-owned combine is not callable")
    routed = (
        ((mx.arange(6 * 4096, dtype=mx.float32) % 29 - 14) / 16.0)
        .astype(mx.bfloat16)
        .reshape(1, 6, 4096)
    )
    route_weights = mx.array(
        [[0.03125, 0.09375, 0.15625, 0.21875, 0.28125, 0.34375]],
        dtype=mx.float32,
    )
    expected = mx.zeros((1, 4096), dtype=mx.bfloat16)
    weights_bf16 = route_weights.astype(mx.bfloat16)
    for expert in range(6):
        product = (routed[:, expert] * weights_bf16[:, expert : expert + 1]).astype(
            mx.bfloat16
        )
        expected = (expected + product).astype(mx.bfloat16)
    actual = combine(routed, route_weights)
    mx.eval(actual, expected)
    if not bool(mx.array_equal(actual, expected)):
        raise ValueError("DeepSeek-V4-0731 row-owned combine exact self-check failed")
