from __future__ import annotations

import json

from mtplx.artifacts import inspect_model
from mtplx.backends.descriptors import DFLASH2_DESCRIPTOR, descriptor_from_inspection
from mtplx.dflash2_bundle import (
    DFLASH2_ARCH_ID,
    DFLASH2_BACKEND,
    resolve_dflash2_bundle_paths,
)


def _write_bundle(
    tmp_path,
    *,
    target_hidden=5120,
    draft_hidden=5120,
    target_vocab=151936,
    draft_vocab=None,
    target_architecture="Qwen3_5ForCausalLM",
    target_model_type="qwen3",
    base_model="Qwen/Qwen3.8-27B",
):
    target = tmp_path / "target"
    draft = tmp_path / "dflash2"
    target.mkdir()
    draft.mkdir()
    draft_vocab = target_vocab if draft_vocab is None else draft_vocab
    target_section = {
        "repo": "converted/qwen3.8-27b-mlx",
        "revision": "target-revision-abc",
    }
    if base_model is not None:
        target_section["base_model"] = base_model
    (tmp_path / "mtplx_dflash2.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "backend": DFLASH2_BACKEND,
                "layout": {"target": "target", "draft": "dflash2"},
                "target": target_section,
                "draft": {
                    "repo": "z-lab/Qwen3.8-27B-DFlash2",
                    "revision": "draft-revision-def",
                    "precision": "4bit",
                },
                "algorithm": {
                    "repo": "z-lab/dflash",
                    "revision": "algorithm-revision-ghi",
                },
                "checksums": {
                    "target_config": {
                        "path": "target/config.json",
                        "sha256": "a" * 64,
                    },
                    "draft_config": {
                        "path": "dflash2/config.json",
                        "sha256": "b" * 64,
                    },
                    "draft_weights": {
                        "path": "dflash2/model.safetensors",
                        "sha256": "c" * 64,
                    },
                },
                "sampler": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
            }
        ),
        encoding="utf-8",
    )
    (target / "config.json").write_text(
        json.dumps(
            {
                "model_type": target_model_type,
                "architectures": [target_architecture],
                "hidden_size": target_hidden,
                "num_hidden_layers": 64,
                "vocab_size": target_vocab,
            }
        ),
        encoding="utf-8",
    )
    (draft / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["DFlash2DraftModel"],
                "hidden_size": draft_hidden,
                "vocab_size": draft_vocab,
                "num_hidden_layers": 5,
                "num_target_layers": 64,
                # Attention heads intentionally differ from the target. They
                # are DFlash2 draft geometry, not a compatibility gate.
                "num_attention_heads": 3,
                "num_key_value_heads": 1,
                "dflash_config": {"target_layer_ids": [1, 15, 30, 45, 60]},
            }
        ),
        encoding="utf-8",
    )
    (target / "model.safetensors").write_bytes(b"target")
    (draft / "model.safetensors").write_bytes(b"draft")
    return tmp_path


def test_dflash2_bundle_resolves_and_selects_backend(tmp_path):
    bundle = _write_bundle(tmp_path)

    paths = resolve_dflash2_bundle_paths(bundle)
    result = inspect_model(bundle)

    assert paths is not None
    assert paths["bundle_root"] == str(bundle)
    assert paths["target_model"] == str(bundle / "target")
    assert paths["draft_model"] == str(bundle / "dflash2")
    assert result.passes_primary_gate is True
    assert result.compatibility["recommended_backend"] == DFLASH2_BACKEND
    assert result.compatibility["runtime_compatibility"] == "dflash2-bundle-native"
    assert result.dflash2_bundle["target_revision"] == "target-revision-abc"
    assert result.dflash2_bundle["draft_revision"] == "draft-revision-def"
    assert result.dflash2_bundle["algorithm_revision"] == "algorithm-revision-ghi"
    assert result.dflash2_bundle["draft_precision"] == "4bit"
    assert result.recommended_sampler == {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
    assert descriptor_from_inspection(result.to_dict()) is DFLASH2_DESCRIPTOR
    assert result.compatibility["arch_id"] == DFLASH2_ARCH_ID


def test_dflash2_bundle_does_not_require_attention_geometry_match(tmp_path):
    bundle = _write_bundle(tmp_path)
    result = inspect_model(bundle)
    assert result.passes_primary_gate is True


def test_dflash2_bundle_rejects_hidden_size_mismatch(tmp_path):
    result = inspect_model(_write_bundle(tmp_path, draft_hidden=4096))

    assert result.passes_primary_gate is False
    assert "hidden_size mismatch" in result.compatibility["message"]


def test_dflash2_bundle_rejects_missing_weights_fail_closed(tmp_path):
    bundle = _write_bundle(tmp_path)
    (bundle / "dflash2" / "model.safetensors").unlink()

    result = inspect_model(bundle)

    assert result.passes_primary_gate is False
    assert "no safetensors weights" in result.compatibility["message"]
    assert "checksum references missing" in result.compatibility["message"]


def test_dflash2_bundle_rejects_layer_and_vocab_mismatch(tmp_path):
    bundle = _write_bundle(tmp_path, draft_vocab=32000)
    config_path = bundle / "dflash2" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["num_target_layers"] = 63
    config["dflash_config"]["target_layer_ids"] = [1, 15, 30, 45, 64]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = inspect_model(bundle)

    assert result.passes_primary_gate is False
    assert "vocab_size mismatch" in result.compatibility["message"]
    assert "num_target_layers mismatch" in result.compatibility["message"]
    assert "target_layer_ids must be within target layer range" in result.compatibility["message"]


def test_dflash2_bundle_rejects_invalid_checksum_and_path_traversal(tmp_path):
    bundle = _write_bundle(tmp_path)
    manifest_path = bundle / "mtplx_dflash2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checksums"]["draft_config"] = {
        "path": "../outside.json",
        "sha256": "not-a-sha256",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = inspect_model(bundle)

    assert result.passes_primary_gate is False
    assert "draft_config checksum must be a 64-character SHA-256 hex digest" in result.compatibility["message"]


def test_dflash2_bundle_rejects_non_qwen_or_non_dflash_configs(tmp_path):
    bundle = _write_bundle(tmp_path)
    target_config = json.loads((bundle / "target/config.json").read_text())
    target_config["model_type"] = "llama"
    target_config["architectures"] = ["LlamaForCausalLM"]
    (bundle / "target/config.json").write_text(json.dumps(target_config))
    draft_config = json.loads((bundle / "dflash2/config.json").read_text())
    draft_config["architectures"] = ["Qwen3ForCausalLM"]
    draft_config["model_type"] = "qwen3"
    (bundle / "dflash2/config.json").write_text(json.dumps(draft_config))

    result = inspect_model(bundle)

    assert result.passes_primary_gate is False
    assert "not the Qwen3.8 target" in result.compatibility["message"]
    assert "not a DFlash2 draft" in result.compatibility["message"]
