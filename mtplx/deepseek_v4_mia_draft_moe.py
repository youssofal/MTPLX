"""Fixed-M5/K64 packed gate-up route for the packaged Mia DSpark draft."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class MiaStackedMXFP4Experts:
    """One native MXFP4 gathered projection over adjacent gate/up rows."""

    __slots__ = ("weight", "scales", "split")

    @staticmethod
    def _validate_projection(module, label: str) -> None:
        observed = (
            int(getattr(module, "group_size", -1)),
            int(getattr(module, "bits", -1)),
            str(getattr(module, "mode", "")),
            getattr(getattr(module, "weight", None), "dtype", None),
            getattr(getattr(module, "scales", None), "dtype", None),
        )
        expected = (32, 4, "mxfp4", mx.uint32, mx.uint8)
        if observed != expected:
            raise ValueError(
                f"Mia K64 {label} projection changed: {observed!r} != {expected!r}"
            )
        if getattr(module, "bias", None) is not None or getattr(
            module, "biases", None
        ) is not None:
            raise ValueError("Mia K64 routed projections must be bias-free")

    @classmethod
    def validate_pair(cls, gate, up) -> None:
        cls._validate_projection(gate, "gate")
        cls._validate_projection(up, "up")
        if (
            gate.weight.ndim != 3
            or up.weight.ndim != 3
            or tuple(gate.weight.shape) != tuple(up.weight.shape)
            or gate.scales.ndim != 3
            or tuple(gate.scales.shape) != tuple(up.scales.shape)
            or int(gate.weight.shape[0]) != int(gate.scales.shape[0])
            or int(gate.weight.shape[1]) != int(gate.scales.shape[1])
        ):
            raise ValueError("Mia K64 gate/up expert geometry changed")

    def __init__(self, gate, up) -> None:
        self.validate_pair(gate, up)
        self.split = int(gate.weight.shape[1])
        self.weight = mx.contiguous(
            mx.concatenate((gate.weight, up.weight), axis=1)
        )
        self.scales = mx.contiguous(
            mx.concatenate((gate.scales, up.scales), axis=1)
        )
        mx.eval(self.weight, self.scales)

        gate.weight = self.weight[:, : self.split]
        gate.scales = self.scales[:, : self.split]
        up.weight = self.weight[:, self.split :]
        up.scales = self.scales[:, self.split :]

    def gather(self, values: mx.array, indices: mx.array) -> mx.array:
        return mx.gather_qmm(
            values,
            self.weight,
            self.scales,
            None,
            rhs_indices=indices,
            transpose=True,
            group_size=32,
            bits=4,
            mode="mxfp4",
            sorted_indices=False,
        )


class MiaPhysicalM5K64SwitchGLU(nn.Module):
    """Inference-only K64 SwitchGLU with one gate/up gathered projection."""

    def __init__(self, switch, packed: MiaStackedMXFP4Experts) -> None:
        super().__init__()
        self.gate_proj = switch.gate_proj
        self.up_proj = switch.up_proj
        self.down_proj = switch.down_proj
        self.activation = switch.activation
        self._packed_gate_up = packed

    def __call__(self, values: mx.array, indices: mx.array) -> mx.array:
        values = mx.expand_dims(values, (-2, -3))
        gate_up = self._packed_gate_up.gather(values, indices)
        gate, up = mx.split(
            gate_up,
            [self._packed_gate_up.split],
            axis=-1,
        )
        hidden = self.activation(up, gate)
        output = self.down_proj(hidden, indices, sorted_indices=False)
        return output.squeeze(-2)


def install_mia_k64_packed_switch(switch) -> MiaPhysicalM5K64SwitchGLU:
    """Pack a qualified native-MXFP4 gate/up pair at construction."""

    packed = MiaStackedMXFP4Experts(switch.gate_proj, switch.up_proj)
    return MiaPhysicalM5K64SwitchGLU(switch, packed)

