"""The Inkling forge's affine recipes: which matrices change, the stored layout, the rounding and the audit."""

import json
from pathlib import Path

import numpy as np
import pytest

from mtplx.commands.forge_inkling_quant import (
    HEAD,
    _write_shard,
    affine_quantize,
    audit_pack,
    bf16_to_f32,
    f32_to_bf16,
    matrix_layout,
)

PROJECTIONS = ("wq_du", "wk_dv", "wv_dv", "wr_du", "gate_attn", "wo_ud")


def test_bare_speed_converts_every_projection_and_the_head_and_nothing_that_steers():
    for layer in range(24):
        for name in PROJECTIONS:
            assert matrix_layout(f"model.llm.layers.{layer}.attn.{name}.weight", "bare-speed") == (4, 64)
        for kept in (
            "attn.attn_sink_proj.weight",
            "attn.k_sconv.weight_proj",
            "attn.rel_logits_proj.proj",
            "attn.q_l2_clip.scale",
            "mlp.gate.weight",
            "mlp.experts.w13_weight",
            "mlp.experts.w2_weight",
            "controllers.k.down_weight",
            "attn_norm.weight",
        ):
            assert matrix_layout(f"model.llm.layers.{layer}.{kept}", "bare-speed") is None
    assert matrix_layout(HEAD, "bare-speed") == (4, 64)
    assert matrix_layout("model.llm.embed.weight", "bare-speed") is None
    assert matrix_layout("model.visual.proj_in.weight", "bare-speed") is None


def test_the_e4_comparator_moves_only_the_trained_mxfp8_experts():
    for layer in range(24):
        for name in ("w13_weight", "w2_weight"):
            want = (4, 64) if layer >= 20 else None
            assert matrix_layout(f"model.llm.layers.{layer}.mlp.experts.{name}", "bare-speed-e4") == want
    assert matrix_layout("model.llm.layers.21.mlp.experts.w13_weight_scale", "bare-speed-e4") is None


def test_unknown_recipe_is_refused():
    with pytest.raises(ValueError):
        matrix_layout(HEAD, "bare")


def _unpack(codes, bits, columns):
    per = 32 // bits
    mask = (1 << bits) - 1
    return np.stack([(codes >> (bits * i)) & mask for i in range(per)], -1).reshape(*codes.shape[:-1], columns)


@pytest.mark.parametrize("bits,group", [(4, 64), (4, 32), (8, 32)])
def test_codes_scales_and_biases_are_mlx_affine_layout(bits, group):
    mx = pytest.importorskip("mlx.core")
    rng = np.random.default_rng(bits * group)
    w = bf16_to_f32(f32_to_bf16((rng.standard_normal((48, 512)) * 0.05).astype(np.float32))).reshape(48, 512)
    codes, scales, biases, _ = affine_quantize(w, bits, group, "mse")
    assert codes.dtype == np.uint32 and codes.shape == (48, 512 * bits // 32)
    assert scales.shape == biases.shape == (48, 512 // group)
    s, b = bf16_to_f32(scales).reshape(scales.shape), bf16_to_f32(biases).reshape(biases.shape)
    ours = np.repeat(s, group, 1) * _unpack(codes, bits, 512) + np.repeat(b, group, 1)
    with mx.stream(mx.cpu):
        theirs = mx.dequantize(mx.array(codes), mx.array(s), mx.array(b), group_size=group, bits=bits)
        assert np.array_equal(np.array(theirs), ours)


def test_mlx_rounding_reproduces_mx_quantize_codes():
    mx = pytest.importorskip("mlx.core")
    rng = np.random.default_rng(7)
    w = bf16_to_f32(f32_to_bf16((rng.standard_normal((32, 2304)) * 0.02).astype(np.float32))).reshape(32, 2304)
    codes, _, _, _ = affine_quantize(w, 4, 64, "mlx")
    with mx.stream(mx.cpu):
        reference, _, _ = mx.quantize(mx.array(w).astype(mx.bfloat16), group_size=64, bits=4)
        assert np.array_equal(np.array(reference), codes)


def test_refined_rounding_never_loses_to_plain_rounding():
    rng = np.random.default_rng(3)
    w = rng.standard_normal((64, 2304)).astype(np.float32) * 0.02
    w[5, 17] = 0.4
    w[9, :64] = 0.0
    w = bf16_to_f32(f32_to_bf16(w)).reshape(w.shape)
    plain = affine_quantize(w, 4, 64, "mlx")[3]["relative_rms_error"]
    refined = affine_quantize(w, 4, 64, "mse")[3]["relative_rms_error"]
    assert refined <= plain


def _safetensors(path: Path, tensors: dict):
    entries = [(name, dtype, list(arr.shape), (lambda a=arr: iter([a.tobytes()]))) for name, (dtype, arr) in tensors.items()]
    _write_shard(path, entries, {"format": "pt"})


def test_audit_fails_closed_when_a_vision_tensor_changes(tmp_path, monkeypatch):
    import mtplx.commands.forge_inkling_quant as forge

    source, pack = tmp_path / "source", tmp_path / "pack"
    source.mkdir(), pack.mkdir()
    vision = {f"model.visual.block{i}.weight": ("BF16", np.full((4, 4), i, np.uint16)) for i in range(3)}
    _safetensors(source / "target.safetensors", vision)
    _safetensors(source / "draft.safetensors", {f"d{i}": ("BF16", np.zeros(2, np.uint16)) for i in range(54)})
    tampered = dict(vision)
    tampered["model.visual.block1.weight"] = ("BF16", np.full((4, 4), 9, np.uint16))
    _safetensors(pack / "model-00001-of-00001.safetensors", tampered)
    (pack / "draft.safetensors").write_bytes((source / "draft.safetensors").read_bytes())
    for directory in (source, pack):
        (directory / "config.json").write_text(json.dumps({"vision_config": {"depth": 3}}))
    (pack / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(vision, "model-00001-of-00001.safetensors")})
    )
    import hashlib

    digest = hashlib.sha256((source / "draft.safetensors").read_bytes()).hexdigest()
    monkeypatch.setattr(forge, "WEIGHTS", {"target.safetensors": "", "draft.safetensors": digest})
    with pytest.raises(ValueError, match="bytes differ from upstream"):
        audit_pack(pack, source)
    tampered["model.visual.block1.weight"] = vision["model.visual.block1.weight"]
    _safetensors(pack / "model-00001-of-00001.safetensors", tampered)
    # Identical bytes still fail closed here: a real pack carries 422 vision tensors.
    with pytest.raises(ValueError):
        audit_pack(pack, source)
