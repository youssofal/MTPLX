"""hy_v3 MTP graft: build the draft head from a standard sharded checkpoint.

The released mlx-lm hy_v3 model class (PR#1211 line) loads the trunk and
sanitizes away the appended NextN layer; the MTPLX injection must therefore
graft the draft head itself from the checkpoint's canonical
``model.layers.{num_hidden_layers}.*`` tensors (tencent-native appended-layer
layout, the form real exports ship in) rather than requiring a native
``model.mtp`` submodule.
"""
import json

import pytest

hy_v3 = pytest.importorskip(
    "mlx_lm.models.hy_v3",
    reason="mlx-lm does not ship models/hy_v3 yet (unreleased upstream)",
)

import mlx.core as mx
from mlx.utils import tree_flatten

from mtplx.hy_v3_mtp_patch import inject_hy_v3_mtp_support, is_hy_v3_mtp_config
from mtplx.mtp_patch import validate_mtp_support

VOCAB, HIDDEN, LAYERS = 128, 64, 2
SPEC_IDX = LAYERS  # appended NextN layer index


def _args(nextn=1):
    return hy_v3.ModelArgs(
        model_type="hy_v3", vocab_size=VOCAB, hidden_size=HIDDEN,
        intermediate_size=128, num_hidden_layers=LAYERS, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, num_experts=4, num_experts_per_tok=2,
        num_shared_experts=1, expert_hidden_dim=64, first_k_dense_replace=1,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        num_nextn_predict_layers=nextn)


def _config(nextn=1, quantization=None):
    cfg = {
        "model_type": "hy_v3", "vocab_size": VOCAB, "hidden_size": HIDDEN,
        "intermediate_size": 128, "num_hidden_layers": LAYERS,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        "num_experts": 4, "num_experts_per_tok": 2, "num_shared_experts": 1,
        "expert_hidden_dim": 64, "first_k_dense_replace": 1,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "num_nextn_predict_layers": nextn,
    }
    if quantization is not None:
        cfg["quantization"] = quantization
    return cfg


def _mtp_tensors():
    """Canonical appended-layer tensors, named via the donor layer itself."""
    donor = hy_v3.DecoderLayer(_args(), layer_idx=SPEC_IDX)
    tensors = {
        f"model.layers.{SPEC_IDX}.{k}": v for k, v in tree_flatten(donor.parameters())
    }
    p = f"model.layers.{SPEC_IDX}"
    tensors[f"{p}.enorm.weight"] = mx.ones((HIDDEN,))
    tensors[f"{p}.hnorm.weight"] = mx.ones((HIDDEN,))
    tensors[f"{p}.eh_proj.weight"] = 0.02 * mx.random.normal((HIDDEN, 2 * HIDDEN))
    tensors[f"{p}.final_layernorm.weight"] = mx.ones((HIDDEN,))
    return tensors


def _write_checkpoint(tmp_path, tensors, config):
    shard = "model-mtp.safetensors"
    mx.save_safetensors(str(tmp_path / shard), tensors)
    json.dump(
        {"metadata": {}, "weight_map": {k: shard for k in tensors}},
        open(tmp_path / "model.safetensors.index.json", "w"),
    )
    json.dump(config, open(tmp_path / "config.json", "w"))


def _grafted(tmp_path, config=None):
    cfg = config or _config()
    _write_checkpoint(tmp_path, _mtp_tensors(), cfg)
    model = hy_v3.Model(_args())
    assert is_hy_v3_mtp_config(cfg)
    assert inject_hy_v3_mtp_support(model, tmp_path, cfg, None)
    return model


def test_graft_injects_and_validates(tmp_path):
    model = _grafted(tmp_path)
    assert validate_mtp_support(model)


def test_return_hidden_is_post_final_norm(tmp_path):
    model = _grafted(tmp_path)
    x = mx.array([[1, 2, 3, 4]])
    logits, hidden = model(x, return_hidden=True)
    assert logits.shape == (1, 4, VOCAB) and hidden.shape == (1, 4, HIDDEN)
    # hidden must be the POST-final-norm trunk state (the measured draft
    # contract — teacher-forced agreement 0.773 post vs 0.387 pre on real
    # code): the head applied directly must reproduce the returned logits.
    again = model.lm_head(hidden)
    assert mx.allclose(again, logits, atol=1e-5).item()


def test_mtp_forward_shapes_and_cache(tmp_path):
    model = _grafted(tmp_path)
    x = mx.array([[1, 2, 3, 4]])
    _logits, hidden = model(x, return_hidden=True)
    mtp_cache = model.make_mtp_cache()
    d = model.mtp_forward(
        hidden[:, -1:, :], mx.array([[5]]), mtp_cache=mtp_cache,
        concat_order="embedding_hidden",
    )
    assert d.shape == (1, 1, VOCAB)
    d2, h2 = model.mtp_forward(
        hidden[:, -1:, :], mx.array([[5]]), mtp_cache=mtp_cache,
        concat_order="embedding_hidden", return_hidden=True,
    )
    assert h2.shape == (1, 1, HIDDEN)
    hidden_upd = model.mtp_update_cache(
        hidden[:, -1:, :], mx.array([[5]]), mtp_cache=mtp_cache)
    assert hidden_upd.shape == (1, 1, HIDDEN)


def test_mtp_logits_use_final_layernorm_and_shared_head(tmp_path):
    model = _grafted(tmp_path)
    x = mx.array([[1, 2, 3, 4]])
    _logits, hidden = model(x, return_hidden=True)
    logits, h = model.mtp_forward(
        hidden[:, -1:, :], mx.array([[5]]), return_hidden=True,
        concat_order="embedding_hidden",
    )
    again = model.lm_head(model.mtp.final_layernorm(h))
    assert mx.allclose(again, logits, atol=1e-5).item()


def test_quantized_overrides_are_honored(tmp_path):
    import mlx.nn as nn

    tensors = _mtp_tensors()
    p = f"model.layers.{SPEC_IDX}"
    quant = {"group_size": 32, "bits": 4, "mode": "affine"}
    overrides = {}
    for mod in (f"{p}.self_attn.q_proj", f"{p}.eh_proj"):
        w = tensors.pop(f"{mod}.weight")
        wq, sc, bs = mx.quantize(w, group_size=32, bits=8)
        tensors[f"{mod}.weight"] = wq
        tensors[f"{mod}.scales"] = sc
        tensors[f"{mod}.biases"] = bs
        overrides[mod] = {"group_size": 32, "bits": 8, "mode": "affine"}
    cfg = _config(quantization={**quant, **overrides})
    _write_checkpoint(tmp_path, tensors, cfg)
    model = hy_v3.Model(_args())
    # This test pins the GRAFT lane's quantize contract. The vendored model
    # class constructs a native MTPBlock unconditionally (its real flow loads
    # + quantizes via mlx_lm.load_model), which would bypass the graft path;
    # drop it so the injector builds the head from the checkpoint like the
    # released (sanitizing) class forces it to.
    if getattr(model, "mtp", None) is not None:
        model.mtp = None
    assert inject_hy_v3_mtp_support(model, tmp_path, cfg, None)
    assert isinstance(model.mtp.layer.self_attn.q_proj, nn.QuantizedLinear)
    assert model.mtp.layer.self_attn.q_proj.bits == 8
    assert isinstance(model.mtp.eh_proj, nn.QuantizedLinear)
    # un-quantized tensors stay plain even with a global quantization block
    assert not isinstance(model.mtp.layer.self_attn.k_proj, nn.QuantizedLinear)
    x = mx.array([[1, 2, 3]])
    _logits, hidden = model(x, return_hidden=True)
    d = model.mtp_forward(hidden[:, -1:, :], mx.array([[5]]),
                          concat_order="embedding_hidden")
    assert d.shape == (1, 1, VOCAB)


def test_ar_only_checkpoint_raises_clearly(tmp_path):
    cfg = _config()
    _write_checkpoint(
        tmp_path, {"model.embed_tokens.weight": mx.zeros((VOCAB, HIDDEN))}, cfg)
    model = hy_v3.Model(_args())
    with pytest.raises(RuntimeError, match="AR-only|no MTP"):
        inject_hy_v3_mtp_support(model, tmp_path, cfg, None)


def test_registry_gate_accepts_tencent_native_layout():
    from mtplx.backends.registry import _passes_hy_v3_gate

    class _Inspection:
        num_hidden_layers = 80
        mtp_num_hidden_layers = 1
        weight_keys = (
            "model.layers.80.enorm.weight",
            "model.layers.80.hnorm.weight",
            "model.layers.80.eh_proj.weight",
            "model.layers.80.final_layernorm.weight",
            "model.layers.80.self_attn.q_proj.weight",
            "model.layers.80.mlp.switch_mlp.gate_proj.weight",
        )

    assert _passes_hy_v3_gate(_Inspection())


def test_registry_gate_still_accepts_mtp_prefix_layout():
    from mtplx.backends.registry import _passes_hy_v3_gate

    class _Inspection:
        num_hidden_layers = 80
        mtp_num_hidden_layers = 1
        weight_keys = (
            "mtp.enorm.weight",
            "mtp.hnorm.weight",
            "mtp.eh_proj.weight",
            "mtp.final_layernorm.weight",
            "mtp.layer.self_attn.q_proj.weight",
        )

    assert _passes_hy_v3_gate(_Inspection())
