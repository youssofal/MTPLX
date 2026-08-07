"""Hermetic object-level contract for the qwen3_5_mtp injector.

The full-checkpoint acceptance sweep runs during hardware bring-up, but the
*object level* at which ``inject_qwen3_5_mtp_support`` attaches the MTP surface
is CPU-testable and load-bearing: mlx-lm's outer ``qwen3_5_moe.Model`` has
``.language_model`` (no ``.model``/``.lm_head``), while its ``.language_model``
(the qwen3_5 ``TextModel``) has ``.model`` + ``.lm_head`` (no ``.language_model``).
``validate_mtp_support`` inspects ``_text_model(model).mtp`` — i.e. the TextModel
— so the injector must attach ``.mtp`` there and re-expose it on the outer wrapper.

This builds a tiny outer model + a synthetic 1-layer MTP head (weights taken from
the head module itself, so the strict load trivially matches) and asserts the
runtime-facing surface resolves on the outer object the runtime actually holds:
``validate_mtp_support`` passes, ``model(..., return_hidden=True)`` returns
``(logits, hidden)``, and ``model.mtp_forward`` drafts against a fresh
``model.make_mtp_cache()``.
"""
import json

import mlx.core as mx
from mlx.utils import tree_flatten

import mlx_lm.models.qwen3_5_moe as qm
from mtplx.mtp_patch import _text_model, validate_mtp_support
from mtplx.qwen3_5_mtp_patch import (
    _make_qwen3_5_mtp_module,
    inject_qwen3_5_mtp_support,
)


def _tiny_text_config():
    return {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 128,
        "head_dim": 64,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_hidden_layers": 4,          # idx 3 is full-attention (interval 4)
        "full_attention_interval": 4,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
        "shared_expert_intermediate_size": 64,
        "vocab_size": 256,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "attention_bias": False,
        "attn_output_gate": True,
        "partial_rotary_factor": 0.25,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_conv_kernel_dim": 4,
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 4096,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "tie_word_embeddings": False,
        "mtp_num_hidden_layers": 1,
        "num_nextn_predict_layers": 1,
    }


def _build_tiny_mtp_checkpoint(tmp_path):
    """A qwen3_5_mtp config + a model-mtp-head.safetensors whose tensors are
    exactly the head module's own parameters (so the strict load matches)."""
    tcfg = _tiny_text_config()
    config = {
        "model_type": "qwen3_5_mtp",
        "text_config": tcfg,
        "num_nextn_predict_layers": 1,
        "tie_word_embeddings": False,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    from mlx_lm.models.qwen3_5 import TextModelArgs

    args = TextModelArgs.from_dict(tcfg)
    head = _make_qwen3_5_mtp_module(args)
    mx.eval(head.parameters())
    flat = dict(tree_flatten(head.parameters()))
    payload = {f"mtp.{k}": v for k, v in flat.items()}
    mx.save_safetensors(str(tmp_path / "model-mtp-head.safetensors"), payload)
    return config, args


def test_inject_attaches_mtp_at_text_model_level_and_validates(tmp_path):
    config, args = _build_tiny_mtp_checkpoint(tmp_path)

    # Outer mlx-lm model, exactly what _load_base_model returns for this arch.
    model = qm.Model(qm.ModelArgs.from_dict(config))
    mx.eval(model.parameters())

    text_model = _text_model(model)
    assert text_model is model.language_model  # outer wrapper, not a bare TextModel

    ok = inject_qwen3_5_mtp_support(model, tmp_path, config)
    assert ok is True

    # .mtp must land on the TextModel (where validate looks), not only the outer.
    assert getattr(text_model, "mtp", None) is not None
    assert validate_mtp_support(model) is True

    # The runtime holds the outer model and calls this surface on it.
    assert callable(getattr(model, "mtp_forward", None))
    assert callable(getattr(model, "make_mtp_cache", None))


def test_outer_model_forward_and_draft_surface(tmp_path):
    config, args = _build_tiny_mtp_checkpoint(tmp_path)
    model = qm.Model(qm.ModelArgs.from_dict(config))
    mx.eval(model.parameters())
    assert inject_qwen3_5_mtp_support(model, tmp_path, config) is True

    H = args.hidden_size
    inputs = mx.array([[1, 2, 3, 4]])  # [B=1, T=4]

    # forward_ar path: MTPLXRuntime calls model(inputs, cache=cache, return_hidden=True)
    logits, hidden = model(inputs, cache=model.make_cache(), return_hidden=True)
    assert logits.shape == (1, 4, args.vocab_size)
    assert hidden.shape == (1, 4, H)

    # draft path: model.mtp_forward(hidden, next_ids, mtp_cache=model.make_mtp_cache())
    next_ids = mx.array([[5]])
    last_hidden = hidden[:, -1:, :]
    draft_logits = model.mtp_forward(
        last_hidden, next_ids, mtp_cache=model.make_mtp_cache()
    )
    mx.eval(draft_logits)
    assert draft_logits.shape[-1] == args.vocab_size
