"""Depth-N MTP head contract: multi-layer draft heads load, gate, and run.

The named expected-key sets in ``mtplx.constants`` are canonical depth-1
templates. Checkpoints may declare ``mtp_num_hidden_layers > 1`` (the config
key is N-generic in the vLLM reference contract); the weight layout then
replicates the per-layer template at each index. These tests pin that the
tensor gate, the contract detector, and the live injector/forward all honor
the declared layer count — and that the depth-1 behavior is unchanged.
"""

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from safetensors.numpy import save_file

from mtplx.artifacts import inspect_mtp_tensors
from mtplx.constants import (
    EXPECTED_MTP_KEYS,
    EXPECTED_PREQUANTIZED_MTP_KEYS,
    expand_mtp_layer_keys,
)
from mtplx.mtp_patch import MTPContract, _mtp_contract_for_weight_keys, inject_mtp_support


def test_expand_mtp_layer_keys_identity_at_depth_one() -> None:
    assert expand_mtp_layer_keys(EXPECTED_MTP_KEYS, 1) == set(EXPECTED_MTP_KEYS)
    # Degenerate inputs clamp to depth 1 rather than emptying the gate.
    assert expand_mtp_layer_keys(EXPECTED_MTP_KEYS, 0) == set(EXPECTED_MTP_KEYS)


def test_expand_mtp_layer_keys_replicates_only_layer_template() -> None:
    expanded = expand_mtp_layer_keys(EXPECTED_MTP_KEYS, 3)

    shared = {key for key in EXPECTED_MTP_KEYS if "mtp.layers.0." not in key}
    per_layer = set(EXPECTED_MTP_KEYS) - shared
    assert shared <= expanded
    for index in range(3):
        assert {
            key.replace("mtp.layers.0.", f"mtp.layers.{index}.") for key in per_layer
        } <= expanded
    assert len(expanded) == len(shared) + 3 * len(per_layer)


def _dense_config(n_layers: int) -> dict:
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "mtp_num_hidden_layers": n_layers,
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "vocab_size": 248320,
        "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
    }


def test_two_layer_mtp_sidecar_passes_tensor_gate(tmp_path) -> None:
    config = _dense_config(2)
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    expanded = expand_mtp_layer_keys(EXPECTED_MTP_KEYS, 2)
    save_file(
        {key: np.ones((1,), dtype=np.float32) for key in expanded},
        tmp_path / "mtp.safetensors",
    )

    result = inspect_mtp_tensors(tmp_path, config)

    assert result.expected_tensor_count == len(expanded)
    assert result.tensor_count == len(expanded)
    assert result.missing_expected_keys == ()
    assert result.extra_keys == ()
    assert result.passes_tensor_gate is True


def test_two_layer_config_with_single_layer_sidecar_fails_gate(tmp_path) -> None:
    config = _dense_config(2)
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(
        {key: np.ones((1,), dtype=np.float32) for key in EXPECTED_MTP_KEYS},
        tmp_path / "mtp.safetensors",
    )

    result = inspect_mtp_tensors(tmp_path, config)

    assert result.passes_tensor_gate is False
    assert any("mtp.layers.1." in key for key in result.missing_expected_keys)


def test_mtp_contract_detects_prequantized_sidecar_at_depth_two() -> None:
    keys = tuple(sorted(expand_mtp_layer_keys(EXPECTED_PREQUANTIZED_MTP_KEYS, 2)))
    contract = _mtp_contract_for_weight_keys(
        MTPContract(),
        keys,
        {
            "text_config": {
                "model_type": "qwen3_5",
                "mtp_num_hidden_layers": 2,
                "quantization": {"bits": 4, "group_size": 32, "mode": "affine"},
            }
        },
    )

    assert contract.mtp_prequantized is True
    assert contract.mtp_quant_policy == "cyankiwi"
    assert contract.mtp_quant_bits == 4
    assert contract.mtp_quant_group_size == 32


def _tiny_text_model_args():
    from mlx_lm.models.qwen3_5 import TextModelArgs

    return TextModelArgs(
        model_type="qwen3_5",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        tie_word_embeddings=True,
        full_attention_interval=4,
    )


def _write_tiny_mtp_sidecar(tmp_path, args, *, n_layers: int = 2) -> dict:
    """Harvest correctly-shaped weights from real DecoderLayer donors."""
    from mlx_lm.models.qwen3_5 import DecoderLayer

    fa_idx = args.full_attention_interval - 1
    tensors: dict[str, mx.array] = {
        "mtp.fc.weight": mx.random.normal((args.hidden_size, args.hidden_size * 2)) * 0.02,
        "mtp.norm.weight": mx.ones((args.hidden_size,)),
        "mtp.pre_fc_norm_hidden.weight": mx.ones((args.hidden_size,)),
        "mtp.pre_fc_norm_embedding.weight": mx.ones((args.hidden_size,)),
    }
    for index in range(n_layers):
        donor = DecoderLayer(args, layer_idx=fa_idx)
        for path, value in tree_flatten(donor.parameters()):
            tensors[f"mtp.layers.{index}.{path}"] = value
    mx.save_safetensors(str(tmp_path / "mtp.safetensors"), tensors)

    config = _dense_config(n_layers)
    config.update(
        {
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "num_hidden_layers": args.num_hidden_layers,
            "num_attention_heads": args.num_attention_heads,
            "num_key_value_heads": args.num_key_value_heads,
            "head_dim": args.head_dim,
            "vocab_size": args.vocab_size,
            "linear_num_value_heads": args.linear_num_value_heads,
            "linear_num_key_heads": args.linear_num_key_heads,
            "linear_key_head_dim": args.linear_key_head_dim,
            "linear_value_head_dim": args.linear_value_head_dim,
            "linear_conv_kernel_dim": args.linear_conv_kernel_dim,
            "tie_word_embeddings": True,
            "full_attention_interval": args.full_attention_interval,
        }
    )
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return config


def _write_tiny_two_layer_sidecar(tmp_path, args) -> dict:
    return _write_tiny_mtp_sidecar(tmp_path, args, n_layers=2)


def test_inject_and_forward_two_layer_mtp_head(tmp_path) -> None:
    from mlx_lm.models.qwen3_5 import TextModel

    args = _tiny_text_model_args()
    config = _write_tiny_two_layer_sidecar(tmp_path, args)
    model = TextModel(args)

    assert inject_mtp_support(model, tmp_path, config) is True
    assert len(model.mtp.layers) == 2

    mtp_cache = model.make_mtp_cache()
    assert len(mtp_cache) == 2

    hidden = mx.random.normal((1, 3, args.hidden_size))
    tokens = mx.array([[1, 2, 3]])
    logits = model.mtp_forward(hidden, tokens, mtp_cache=mtp_cache)
    mx.eval(logits)

    assert logits.shape == (1, 3, args.vocab_size)
    assert all(int(cache.offset) == 3 for cache in mtp_cache)

    # A second step must keep every draft-layer cache advancing in lockstep.
    step_hidden = mx.random.normal((1, 1, args.hidden_size))
    step_tokens = mx.array([[4]])
    step_logits = model.mtp_forward(step_hidden, step_tokens, mtp_cache=mtp_cache)
    mx.eval(step_logits)
    assert step_logits.shape == (1, 1, args.vocab_size)
    assert all(int(cache.offset) == 4 for cache in mtp_cache)


def test_mtp_forward_uses_qwen38_dual_norm_concat_route(
    tmp_path,
    monkeypatch,
) -> None:
    from mlx_lm.models.qwen3_5 import TextModel

    import mtplx.qwen38_challenge_kernels as kernels

    args = _tiny_text_model_args()
    config = _write_tiny_two_layer_sidecar(tmp_path, args)
    model = TextModel(args)
    assert inject_mtp_support(model, tmp_path, config) is True
    calls: list[tuple[int, int]] = []

    def fused(a, b, a_weight, b_weight, eps):
        calls.append((int(a.shape[-1]), int(b.shape[-1])))
        return mx.concatenate((a, b), axis=-1)

    monkeypatch.setattr(kernels, "qwen38_dual_rms_norm_concat", fused)
    model._mtplx_prepare_mtp_inputs = model._mtplx_prepare_mtp_inputs_dual
    hidden = mx.random.normal((1, 1, args.hidden_size))
    tokens = mx.array([[1]])
    logits = model.mtp_forward(hidden, tokens, mtp_cache=model.make_mtp_cache())
    mx.eval(logits)

    assert calls == [(args.hidden_size, args.hidden_size)]


def test_row24_target_eval_ladder_uses_decode_rungs(tmp_path, monkeypatch) -> None:
    from mlx_lm.models.qwen3_5 import TextModel

    import mtplx.qwen38_challenge_kernels as kernels

    args = _tiny_text_model_args()
    config = _write_tiny_mtp_sidecar(tmp_path, args, n_layers=1)
    model = TextModel(args)
    assert inject_mtp_support(model, tmp_path, config) is True
    calls: list[tuple[int, ...]] = []
    monkeypatch.setattr(
        kernels,
        "qwen38_row24_async_eval",
        lambda value, **_kwargs: calls.append(tuple(value.shape)),
    )
    model._mtplx_forward_layers = model._mtplx_forward_layers_row24

    logits = model(mx.array([[1, 2, 3, 4]]))
    mx.eval(logits)

    assert calls == [(1, 4, args.hidden_size), (1, 4, args.hidden_size)]


def test_mtp_cache_length_mismatch_fails_loud(tmp_path) -> None:
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.qwen3_5 import TextModel

    args = _tiny_text_model_args()
    config = _write_tiny_two_layer_sidecar(tmp_path, args)
    model = TextModel(args)
    assert inject_mtp_support(model, tmp_path, config) is True

    hidden = mx.random.normal((1, 1, args.hidden_size))
    tokens = mx.array([[1]])
    with pytest.raises(ValueError, match="draft"):
        model.mtp_forward(hidden, tokens, mtp_cache=[KVCache()])


def test_one_layer_kv_only_history_append_matches_control_cache(tmp_path) -> None:
    from mlx_lm.models.qwen3_5 import TextModel

    args = _tiny_text_model_args()
    config = _write_tiny_mtp_sidecar(tmp_path, args, n_layers=1)
    model = TextModel(args)
    assert inject_mtp_support(model, tmp_path, config) is True

    hidden = mx.random.normal((1, 4, args.hidden_size))
    tokens = mx.array([[1, 2, 3, 4]])
    control_cache = model.make_mtp_cache()
    control_root = model.mtp_update_cache(hidden, tokens, mtp_cache=control_cache)
    mx.eval(control_root, *(control_cache[0].state))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dead full-layer work ran during K/V-only append")

    layer = model.mtp.layers[0]
    layer.self_attn.q_proj = forbidden
    layer.self_attn.o_proj = forbidden
    layer.mlp = forbidden
    model.mtp.norm = forbidden
    candidate_cache = model.make_mtp_cache()
    candidate_root = model.mtp_update_cache_kv_only_history(
        hidden,
        tokens,
        mtp_cache=candidate_cache,
    )
    mx.eval(candidate_root)

    assert candidate_cache[0].offset == control_cache[0].offset == 4
    for candidate, control in zip(candidate_cache[0].state, control_cache[0].state):
        assert mx.array_equal(candidate, control).item()
    assert getattr(layer.self_attn, "_mtplx_qwen38_packed_kv", None) is not None


def test_qwen38_kv_only_history_packs_quantized_kv_projection(tmp_path) -> None:
    from mlx_lm.models.qwen3_5 import TextModel

    args = _tiny_text_model_args()
    config = _write_tiny_mtp_sidecar(tmp_path, args, n_layers=1)
    model = TextModel(args)
    contract = MTPContract(
        mtp_quant_bits=4,
        mtp_quant_group_size=32,
        mtp_quant_policy="all",
    )
    assert inject_mtp_support(model, tmp_path, config, contract=contract) is True

    hidden = mx.random.normal((1, 4, args.hidden_size))
    tokens = mx.array([[1, 2, 3, 4]])
    cache = model.make_mtp_cache()
    root = model.mtp_update_cache_kv_only_history(hidden, tokens, mtp_cache=cache)
    mx.eval(root)

    assert getattr(
        model.mtp.layers[0].self_attn,
        "_mtplx_qwen38_packed_kv",
        None,
    ) is not None
