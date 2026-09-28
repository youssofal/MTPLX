"""A tiny quantized Qwen3.5-MoE with the A3B layout, for bit-identity tests.

Three GDN layers and one full-attention layer, routed plus shared experts,
bf16 with 4-bit affine projections, optionally with a one-layer MTP draft
head injected the product way (``mtp_patch.inject_mtp_support``). 32 experts
(not 8): with 8 experts, top-2, MLX picks another expert kernel below 33
tokens even on the invariant lane, which A3B (256 experts) never does.
No pack is loaded; the weights are random.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten

TEXT_CONFIG = {
    "model_type": "qwen3_5_moe_text",
    "hidden_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 64,
    "vocab_size": 256,
    "linear_num_value_heads": 2,
    "linear_num_key_heads": 1,
    "linear_key_head_dim": 64,
    "linear_value_head_dim": 64,
    "num_experts": 32,
    "num_experts_per_tok": 4,
    "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 64,
    "intermediate_size": 64,
}


def tiny_model(seed: int = 0, **overrides):
    from mlx_lm.models import qwen3_5_moe

    text = dict(TEXT_CONFIG, **overrides)
    args = qwen3_5_moe.ModelArgs(model_type="qwen3_5_moe", text_config=text)
    mx.random.seed(seed)
    model = qwen3_5_moe.Model(args)
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())
    return model


def tiny_model_with_draft_head(tmp_path: Path, seed: int = 0, **overrides):
    """``tiny_model`` plus a one-layer MoE MTP head, injected the product way."""

    from mlx_lm.models.qwen3_5 import DecoderLayer

    from mtplx.mtp_patch import inject_mtp_support

    model = tiny_model(seed, **overrides)
    args = model.language_model.args
    mx.random.seed(seed + 1)
    donor = DecoderLayer(args, layer_idx=args.full_attention_interval - 1)
    donor.set_dtype(mx.bfloat16)
    hidden = args.hidden_size
    tensors = {
        "mtp.fc.weight": (mx.random.normal((hidden, hidden * 2)) * 0.05).astype(mx.bfloat16),
        "mtp.norm.weight": mx.ones((hidden,), dtype=mx.bfloat16),
        "mtp.pre_fc_norm_hidden.weight": mx.ones((hidden,), dtype=mx.bfloat16),
        "mtp.pre_fc_norm_embedding.weight": mx.ones((hidden,), dtype=mx.bfloat16),
    }
    for path, value in tree_flatten(donor.parameters()):
        tensors[f"mtp.layers.0.{path}"] = value
    mx.save_safetensors(str(tmp_path / "mtp.safetensors"), tensors)
    text = dict(TEXT_CONFIG, **overrides, mtp_num_hidden_layers=1)
    config = {
        "model_type": "qwen3_5_moe",
        "text_config": text,
        "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert inject_mtp_support(model, tmp_path, config) is True
    return model


def prompt(tokens: int, seed: int = 7, vocab: int = 256) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(token) for token in rng.integers(0, vocab, size=tokens)]


def as_numpy(value: mx.array) -> np.ndarray:
    if value.dtype == mx.bfloat16:
        value = value.view(mx.uint16)
    return np.array(value)


def assert_bit_equal(left, right) -> None:
    left, right = list(left), list(right)
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype
        assert np.array_equal(as_numpy(a), as_numpy(b))
