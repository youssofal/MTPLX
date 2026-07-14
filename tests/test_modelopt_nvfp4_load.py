from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mtplx.modelopt_nvfp4_load import (
    attach_modelopt_nvfp4_aux,
    is_modelopt_nvfp4_native_config,
    partition_modelopt_nvfp4_weights,
    resolve_module_path,
)


def test_is_modelopt_nvfp4_native_config():
    assert is_modelopt_nvfp4_native_config(
        {"mtplx_policy": {"source_format": "modelopt-nvfp4-native-w4a16"}}
    )
    assert not is_modelopt_nvfp4_native_config({"mtplx_policy": {"source_format": "affine"}})


def test_partition_modelopt_nvfp4_weights_splits_aux_tensors():
    weights = {
        "language_model.model.layers.0.mlp.gate_proj.weight": mx.zeros((2, 2)),
        "language_model.model.layers.0.mlp.gate_proj.scales": mx.zeros((2, 2)),
        "language_model.model.layers.0.mlp.gate_proj.global_scale_w": mx.array([1.0]),
        "language_model.model.layers.0.mlp.gate_proj.input_scale": mx.array([0.5]),
    }
    core, aux = partition_modelopt_nvfp4_weights(weights)
    assert set(core) == {
        "language_model.model.layers.0.mlp.gate_proj.weight",
        "language_model.model.layers.0.mlp.gate_proj.scales",
    }
    assert set(aux) == {
        "language_model.model.layers.0.mlp.gate_proj.global_scale_w",
        "language_model.model.layers.0.mlp.gate_proj.input_scale",
    }


def test_attach_modelopt_nvfp4_aux_sets_module_parameters():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.QuantizedLinear(16, 8, group_size=16, bits=4, mode="nvfp4")

    class Layers(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Block()]

        def __getitem__(self, index: int) -> Block:
            return self.layers[index]

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = Layers()

    class LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = LanguageModel()

    model = Root()
    module = resolve_module_path(model, "language_model.model.layers.0.gate_proj")
    assert module is model.language_model.model.layers[0].gate_proj
    attached = attach_modelopt_nvfp4_aux(
        model,
        {
            "language_model.model.layers.0.gate_proj.global_scale_w": mx.array([2.0]),
            "language_model.model.layers.0.gate_proj.input_scale": mx.array([0.25]),
        },
    )
    assert attached == 2
    assert float(module["global_scale_w"].item()) == 2.0
    assert float(module["input_scale"].item()) == 0.25
