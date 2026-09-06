"""Regression coverage for Qwen3.8 Flash-Next FP8 scale layouts."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "convert_qwen4_exp_fp8.py"


@pytest.fixture
def converter(monkeypatch):
    """Load the converter with a small NumPy-backed MLX stand-in.

    The converter itself only needs these array operations for dequantization,
    which keeps this focused regression runnable on non-macOS CI.
    """
    fake_mx = types.SimpleNamespace(
        array=np.asarray,
        repeat=np.repeat,
        bfloat16=np.float32,
        float16=np.float16,
        float32=np.float32,
    )
    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

    spec = importlib.util.spec_from_file_location("qwen4_exp_fp8_converter_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reader(module, tensors: dict[str, tuple[np.ndarray, str]]):
    reader = object.__new__(module.SourceReader)
    reader.weight_map = {name: "synthetic.safetensors" for name in tensors}
    reader.raw = lambda name: tensors[name]
    return reader


def test_bf16_uses_per_block_weight_scale_inv_for_ordinary_fp8_weight(converter) -> None:
    weight_name = "model.layers.0.mlp.gate_proj.weight"
    scale_name = "model.layers.0.mlp.gate_proj.weight_scale_inv"
    reader = _reader(
        converter,
        {
            weight_name: (np.full((2, 3), 0x38, dtype=np.uint8), "F8_E4M3"),
            scale_name: (np.array([[2.0]], dtype=np.float32), "F32"),
        },
    )

    np.testing.assert_allclose(reader.bf16(weight_name), np.full((2, 3), 2.0))


def test_bf16_uses_shared_scalar_weight_scale_for_ngram_fp8_shard(converter) -> None:
    weight_name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
    scale_name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
    reader = _reader(
        converter,
        {
            weight_name: (np.full((2, 3), 0x38, dtype=np.uint8), "F8_E4M3"),
            scale_name: (np.array([0.5], dtype=np.float32), "F32"),
        },
    )

    np.testing.assert_allclose(reader.bf16(weight_name), np.full((2, 3), 0.5))
