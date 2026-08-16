"""Retained native M=5 DSpark Q3 gate/up packing for Flash-0731.

The three installed DSpark stages use affine Q3/group-128 routed projections
with the fixed physical layout recorded by the 0731 receipt.  This module
validates loaded storage once, then replaces each gate/up pair with the native
MLX packed projection.  Activation and the Q3 down projection stay unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

from .models.deepseek_v4 import (
    DeepseekV4DSpark,
    DeepseekV4DSparkStage,
    DeepseekV4MoE,
    Model,
    MoEGate,
)
from .moe_packed_projections import PackedSwitchGLU, _pack_pair


DSPARK_Q3_GATE_UP_GEOMETRY = {
    "rows": 5,
    "hidden_size": 4096,
    "width": 2048,
    "experts": 256,
    "top_k": 6,
    "bits": 3,
    "group_size": 128,
    "weight_shape": (256, 2048, 384),
    "metadata_shape": (256, 2048, 32),
}


@dataclass(frozen=True)
class DSparkQ3GateUpContract:
    rows: int
    hidden_size: int
    width: int
    experts: int
    top_k: int
    bits: int
    group_size: int
    weight_shape: tuple[int, int, int]
    metadata_shape: tuple[int, int, int]


class DeepseekV40731DSparkM5PackedSwitchGLU(PackedSwitchGLU):
    """Construction-qualified fixed-M5 route with one unsorted packed gather."""

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        expanded = mx.expand_dims(x, (-2, -3))
        packed = self.gate_up_proj.gather(expanded, indices, False)
        gate, up = mx.split(packed, [self._split_at], axis=-1)
        routed = self.down_proj(
            self.activation(up, gate),
            indices,
            sorted_indices=False,
        )
        return routed.squeeze(-2)


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        getter = getattr(value, "get_shape", None)
        shape = () if getter is None else getter()
    return tuple(int(item) for item in shape)


def _q3_contract(
    *, hidden_size: int, width: int, experts: int, top_k: int, rows: int
) -> DSparkQ3GateUpContract:
    hidden_size = int(hidden_size)
    width = int(width)
    experts = int(experts)
    top_k = int(top_k)
    rows = int(rows)
    if (
        hidden_size <= 0
        or hidden_size % 128
        or hidden_size * 3 % 32
        or width <= 0
        or experts <= 0
        or top_k != 6
        or rows != 5
    ):
        raise ValueError("DSpark Q3 M=5 geometry is invalid")
    return DSparkQ3GateUpContract(
        rows=rows,
        hidden_size=hidden_size,
        width=width,
        experts=experts,
        top_k=top_k,
        bits=3,
        group_size=128,
        weight_shape=(experts, width, hidden_size * 3 // 32),
        metadata_shape=(experts, width, hidden_size // 128),
    )


def validate_dspark_q3_gate_up(
    gate: nn.Module,
    up: nn.Module,
    *,
    hidden_size: int,
    width: int,
    experts: int,
    top_k: int,
    rows: int,
) -> DSparkQ3GateUpContract:
    """Validate physical Q3 storage once before binding the candidate route."""

    contract = _q3_contract(
        hidden_size=hidden_size,
        width=width,
        experts=experts,
        top_k=top_k,
        rows=rows,
    )
    for label, projection in (("gate", gate), ("up", up)):
        if not isinstance(projection, QuantizedSwitchLinear):
            raise ValueError(
                f"DSpark Q3 {label} projection is not QuantizedSwitchLinear"
            )
        if int(getattr(projection, "bits", 0)) != contract.bits:
            raise ValueError(f"DSpark Q3 {label} projection must use Q3")
        if int(getattr(projection, "group_size", 0)) != contract.group_size:
            raise ValueError(f"DSpark Q3 {label} projection must use group-128")
        if str(getattr(projection, "mode", "")) != "affine":
            raise ValueError(
                f"DSpark Q3 {label} projection must use affine quantization"
            )
        if getattr(projection.weight, "dtype", None) != mx.uint32:
            raise ValueError(f"DSpark Q3 {label} packed weight must be U32")
        if _shape(projection.weight) != contract.weight_shape:
            raise ValueError(f"DSpark Q3 {label} packed weight shape is invalid")
        if _shape(projection.scales) != contract.metadata_shape:
            raise ValueError(f"DSpark Q3 {label} scales shape is invalid")
        if _shape(getattr(projection, "biases", None)) != contract.metadata_shape:
            raise ValueError(f"DSpark Q3 {label} biases shape is invalid")
        if (
            getattr(projection.scales, "dtype", None) != mx.bfloat16
            or getattr(projection.biases, "dtype", None) != mx.bfloat16
        ):
            raise ValueError(f"DSpark Q3 {label} affine metadata must be BF16")
        if "bias" in projection:
            raise ValueError(f"DSpark Q3 {label} projection must not have output bias")
    return contract


def _validate_dspark_q3_down(
    projection: nn.Module, contract: DSparkQ3GateUpContract
) -> None:
    """Prove the unchanged down projection is the paired native Q3 bank."""

    expected_weight = (
        contract.experts,
        contract.hidden_size,
        contract.width * contract.bits // 32,
    )
    expected_metadata = (
        contract.experts,
        contract.hidden_size,
        contract.width // contract.group_size,
    )
    if not isinstance(projection, QuantizedSwitchLinear):
        raise ValueError("DSpark Q3 down projection is not QuantizedSwitchLinear")
    if (
        int(getattr(projection, "bits", 0)) != contract.bits
        or int(getattr(projection, "group_size", 0)) != contract.group_size
        or str(getattr(projection, "mode", "")) != "affine"
        or getattr(projection.weight, "dtype", None) != mx.uint32
        or _shape(projection.weight) != expected_weight
        or _shape(projection.scales) != expected_metadata
        or _shape(getattr(projection, "biases", None)) != expected_metadata
        or getattr(projection.scales, "dtype", None) != mx.bfloat16
        or getattr(projection.biases, "dtype", None) != mx.bfloat16
        or "bias" in projection
    ):
        raise ValueError("DSpark Q3 down projection storage is invalid")


def _validate_dspark_q3_switch(
    switch: nn.Module,
    *,
    hidden_size: int,
    width: int,
    experts: int,
    top_k: int,
    rows: int,
) -> DSparkQ3GateUpContract:
    contract = validate_dspark_q3_gate_up(
        switch.gate_proj,
        switch.up_proj,
        hidden_size=hidden_size,
        width=width,
        experts=experts,
        top_k=top_k,
        rows=rows,
    )
    _validate_dspark_q3_down(switch.down_proj, contract)
    if not callable(getattr(switch, "activation", None)):
        raise ValueError("DSpark Q3 switch activation is absent")
    return contract


def build_dspark_q3_packed_gate_up(
    switch: nn.Module,
    *,
    hidden_size: int,
    width: int,
    experts: int,
    top_k: int,
    rows: int,
) -> DeepseekV40731DSparkM5PackedSwitchGLU:
    """Pack native Q3 gate/up rows without changing activation or down math."""

    _validate_dspark_q3_switch(
        switch,
        hidden_size=hidden_size,
        width=width,
        experts=experts,
        top_k=top_k,
        rows=rows,
    )
    result = _pack_pair(switch.gate_proj, switch.up_proj, axis=1)
    if isinstance(result, str):
        raise ValueError(f"DSpark native Q3 gate/up packing failed: {result}")
    packed, split_at = result
    if int(split_at) != int(width):
        raise ValueError("DSpark native Q3 gate/up split is invalid")
    return DeepseekV40731DSparkM5PackedSwitchGLU(
        packed,
        switch.down_proj,
        switch.activation,
        split_at,
    )


def _validate_dspark_m5_owner(model: Any) -> tuple[DeepseekV4DSparkStage, ...]:
    if type(model) is not Model:
        raise ValueError("DSpark native Q3 packing requires the exact model owner")
    dspark = getattr(model, "_dspark", None)
    if type(dspark) is not DeepseekV4DSpark:
        raise ValueError("DSpark native Q3 packing requires the exact model owner")
    stages = tuple(getattr(dspark, "stages", ()))
    if len(stages) != 3:
        raise ValueError("DSpark native Q3 packing requires exactly three stages")
    if any(type(stage) is not DeepseekV4DSparkStage for stage in stages):
        raise ValueError("DSpark native Q3 packing stage identity is invalid")
    if tuple(int(stage.stage_id) for stage in stages) != (0, 1, 2):
        raise ValueError("DSpark native Q3 packing stage order must be 0, 1, 2")
    if any(type(stage.ffn) is not DeepseekV4MoE for stage in stages):
        raise ValueError("DSpark native Q3 packing FFN identity is invalid")
    if any(type(stage.ffn.gate) is not MoEGate for stage in stages):
        raise ValueError("DSpark native Q3 packing router identity is invalid")
    published = tuple(getattr(model.mtp, "layers", model.mtp))
    if len(published) != 3 or any(
        registered is not owned
        for registered, owned in zip(published, stages, strict=True)
    ):
        raise ValueError("DSpark native Q3 packing model.mtp ownership is invalid")
    if int(getattr(dspark, "block_size", 0)) != 5 or any(
        int(getattr(stage, "block_size", 0)) != 5 for stage in stages
    ):
        raise ValueError("DSpark native Q3 packing requires exact M=5 ownership")
    if any(int(getattr(stage.ffn.gate, "topk", 0)) != 6 for stage in stages):
        raise ValueError("DSpark native Q3 packing requires router top-k=6")
    return stages


def _receipt() -> dict[str, Any]:
    return {
        "candidate": "dspark-native-packed-q3-gate-up-m5",
        "stages": 3,
        "geometry": dict(DSPARK_Q3_GATE_UP_GEOMETRY),
        "gate_up_dispatches_per_stage": 1,
        "stock_gate_up_dispatches_per_stage": 2,
        "explicit_dequantize": False,
        "resident_weight_bytes_added": 0,
    }


@dataclass(frozen=True, slots=True)
class PreparedDSparkQ3PackedGateUpM5:
    """Fully constructed three-stage FFN replacements awaiting publication."""

    stages: tuple[DeepseekV4DSparkStage, ...]
    originals: tuple[Any, ...]
    replacements: tuple[DeepseekV40731DSparkM5PackedSwitchGLU, ...]
    receipt: dict[str, Any]

    def publish(self) -> None:
        try:
            for stage, replacement in zip(self.stages, self.replacements, strict=True):
                stage.ffn.switch_mlp = replacement
        except Exception as publication_error:
            try:
                self.restore()
            except ExceptionGroup as restoration_errors:
                publication_error.add_note(
                    f"DSpark FFN publication rollback also failed: {restoration_errors}"
                )
            raise

    def restore(self) -> None:
        errors = []
        for stage, original in zip(self.stages, self.originals, strict=True):
            try:
                stage.ffn.switch_mlp = original
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("DSpark FFN restoration failed", errors)


def prepare_dspark_q3_packed_gate_up_m5(
    model: Any,
) -> PreparedDSparkQ3PackedGateUpM5:
    """Validate and build the exact model-owned M5 stages without publishing."""

    stages = _validate_dspark_m5_owner(model)
    geometry = {
        key: DSPARK_Q3_GATE_UP_GEOMETRY[key]
        for key in ("hidden_size", "width", "experts", "top_k", "rows")
    }
    switches = tuple(stage.ffn.switch_mlp for stage in stages)
    for switch in switches:
        _validate_dspark_q3_switch(switch, **geometry)
    replacements = tuple(
        build_dspark_q3_packed_gate_up(switch, **geometry) for switch in switches
    )
    return PreparedDSparkQ3PackedGateUpM5(
        stages=stages,
        originals=switches,
        replacements=replacements,
        receipt=_receipt(),
    )


def install_dspark_q3_packed_gate_up_m5(model: Any) -> dict[str, Any]:
    """Convenience wrapper that prepares, then publishes all three stages."""

    prepared = prepare_dspark_q3_packed_gate_up_m5(model)
    prepared.publish()
    return prepared.receipt
