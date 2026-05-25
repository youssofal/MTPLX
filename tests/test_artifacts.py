from __future__ import annotations

import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from mtplx.artifacts import expected_mtp_file, inspect_model
from mtplx.backends.registry import (
    UnverifiedArchitectureError,
    architecture_catalog,
    require_verified_or_raise,
)
from mtplx.constants import (
    EXPECTED_ALL_PREQUANTIZED_MTP_KEYS,
    EXPECTED_MTP_KEYS,
    EXPECTED_PREQUANTIZED_MTP_KEYS,
)


def _write_runtime_contract(
    path,
    *,
    arch_id="qwen3-next-mtp",
    profile="stable",
    recommended_draft_lm_head=None,
    recommended_draft_sampler=None,
):
    contract = {
        "mtplx_version": "0.1.4",
        "arch_id": arch_id,
        "mtp_depth_max": 3,
        "recommended_profile": profile,
        "exactness_baseline": {"phase0h": "smoke", "max_abs_diff": 0.0},
        "verified_on": {
            "timestamp": "2026-05-02T00:00:00Z",
            "hardware": "test",
            "macos": "test",
        },
    }
    if recommended_draft_lm_head is not None:
        contract["recommended_draft_lm_head"] = recommended_draft_lm_head
    if recommended_draft_sampler is not None:
        contract["recommended_draft_sampler"] = recommended_draft_sampler
    (path / "mtplx_runtime.json").write_text(
        json.dumps(contract),
        encoding="utf-8",
    )


def test_expected_mtp_file_uses_extra_tensor_metadata(tmp_path):
    config = {"mlx_lm_extra_tensors": {"mtp_file": "extra-mtp.safetensors"}}
    assert expected_mtp_file(tmp_path, config) == tmp_path / "extra-mtp.safetensors"


def test_inspect_model_reports_missing_config(tmp_path):
    result = inspect_model(tmp_path)
    assert result.config_exists is False
    assert result.passes_primary_gate is False


def test_inspect_model_reads_qwen_mtp_config_without_weights(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "mtp_num_hidden_layers": 1,
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "vocab_size": 248320,
                "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
            }
        )
    )
    result = inspect_model(tmp_path)
    assert result.model_type == "qwen3_5"
    assert result.mtp_num_hidden_layers == 1
    assert result.mtp is not None
    assert result.mtp.exists is False
    assert result.compatibility["tier"] == "architecture-compatible-but-unverified"
    assert result.compatibility["exit_code"] == 3
    assert result.compatibility["runtime_compatibility"] == "missing-mtp-weights"
    assert result.compatibility["unsafe_force_required"] is False
    assert "mtplx_runtime.json is optional metadata" in result.compatibility["message"]
    assert "missing MTP weights" in result.compatibility["message"]


def test_qwen3_5_text_subtype_can_pass_primary_gate_when_mtp_is_valid(monkeypatch, tmp_path):
    from mtplx import artifacts
    from mtplx.artifacts import MTPInspection

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "mtp_num_hidden_layers": 1,
                    "hidden_size": 5120,
                    "num_hidden_layers": 64,
                    "vocab_size": 248320,
                },
                "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
            }
        )
    )

    monkeypatch.setattr(
        artifacts,
        "inspect_mtp_tensors",
        lambda *_args, **_kwargs: MTPInspection(
            mtp_file=str(tmp_path / "mtp.safetensors"),
            exists=True,
            tensor_count=15,
            missing_expected_keys=(),
        ),
    )
    _write_runtime_contract(tmp_path)
    assert inspect_model(tmp_path).passes_primary_gate is True


def test_runtime_contract_preserves_recommended_draft_metadata(monkeypatch, tmp_path):
    from mtplx import artifacts
    from mtplx.artifacts import MTPInspection

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "mtp_num_hidden_layers": 1,
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "vocab_size": 248320,
                "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        artifacts,
        "inspect_mtp_tensors",
        lambda *_args, **_kwargs: MTPInspection(
            mtp_file=str(tmp_path / "mtp.safetensors"),
            exists=True,
            tensor_count=15,
            missing_expected_keys=(),
        ),
    )
    _write_runtime_contract(
        tmp_path,
        profile="performance-cold",
        recommended_draft_lm_head={"bits": "3", "group_size": "64", "mode": "affine"},
        recommended_draft_sampler={"temperature": "0.7", "top_p": "0.95", "top_k": "20"},
    )

    result = inspect_model(tmp_path)

    assert result.compatibility["runtime_contract"]["recommended_draft_lm_head"] == {
        "bits": 3,
        "group_size": 64,
        "mode": "affine",
    }
    assert result.compatibility["runtime_contract"]["recommended_draft_sampler"] == {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
    }
    assert result.compatibility["tier"] == "verified"


def test_prequantized_mtp_sidecar_accepts_mlx_affine_scale_bias_tensors(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "mtp_num_hidden_layers": 1,
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "vocab_size": 248320,
                "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
                "mtplx_mtp_quantization": {
                    "policy": "cyankiwi",
                    "bits": 4,
                    "group_size": 32,
                    "mode": "affine",
                    "prequantized": True,
                },
            }
        )
    )
    save_file(
        {key: np.ones((1,), dtype=np.float32) for key in EXPECTED_PREQUANTIZED_MTP_KEYS},
        tmp_path / "mtp.safetensors",
    )
    _write_runtime_contract(tmp_path, profile="performance-cold")

    result = inspect_model(tmp_path)
    assert result.mtp is not None
    assert result.mtp.sidecar_format == "prequantized-mlx-affine"
    assert result.mtp.tensor_count == 29
    assert result.mtp.missing_expected_keys == ()
    assert result.mtp.extra_keys == ()
    assert result.passes_primary_gate is True


def test_all_prequantized_mtp_sidecar_accepts_quantized_fc_tensors(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "mtp_num_hidden_layers": 1,
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "vocab_size": 248320,
                "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
                "mtplx_mtp_quantization": {
                    "policy": "all",
                    "bits": 4,
                    "group_size": 32,
                    "mode": "affine",
                    "prequantized": True,
                },
            }
        )
    )
    save_file(
        {key: np.ones((1,), dtype=np.float32) for key in EXPECTED_ALL_PREQUANTIZED_MTP_KEYS},
        tmp_path / "mtp.safetensors",
    )
    _write_runtime_contract(tmp_path, profile="performance-cold")

    result = inspect_model(tmp_path)
    assert result.mtp is not None
    assert result.mtp.sidecar_format == "prequantized-mlx-affine"
    assert result.mtp.tensor_count == 31
    assert result.mtp.missing_expected_keys == ()
    assert result.mtp.extra_keys == ()
    assert result.passes_primary_gate is True
    assert result.compatibility["tier"] == "verified"
    assert result.compatibility["recommended_profile"] == "performance-cold"


def test_qwen3_5_moe_mtp_sidecar_passes_moe_tensor_gate(tmp_path):
    num_experts = 4
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "mtp_num_hidden_layers": 1,
                    "num_experts": num_experts,
                    "hidden_size": 256,
                    "num_hidden_layers": 8,
                    "vocab_size": 1000,
                },
                "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
            }
        )
    )
    base = "mtp.layers.0"
    keys = {
        "mtp.fc.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        f"{base}.input_layernorm.weight",
        f"{base}.post_attention_layernorm.weight",
        f"{base}.self_attn.q_proj.weight",
        f"{base}.self_attn.k_proj.weight",
        f"{base}.self_attn.v_proj.weight",
        f"{base}.self_attn.o_proj.weight",
        f"{base}.self_attn.q_norm.weight",
        f"{base}.self_attn.k_norm.weight",
        f"{base}.mlp.gate.weight",
        f"{base}.mlp.shared_expert.gate_proj.weight",
        f"{base}.mlp.shared_expert.up_proj.weight",
        f"{base}.mlp.shared_expert.down_proj.weight",
        f"{base}.mlp.shared_expert_gate.weight",
    }
    for e in range(num_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            keys.add(f"{base}.mlp.experts.{e}.{proj}.weight")
    save_file(
        {key: np.ones((1,), dtype=np.float32) for key in keys},
        tmp_path / "mtp.safetensors",
    )

    result = inspect_model(tmp_path)
    assert result.mtp is not None
    assert result.mtp.sidecar_format == "bf16-moe"
    # 4 global + 13 per-layer structural + 3 projections per expert.
    assert result.mtp.tensor_count == 4 + 13 + 3 * num_experts
    assert result.mtp.missing_expected_keys == ()
    assert result.mtp.extra_keys == ()
    assert result.mtp.passes_tensor_gate is True
    assert result.passes_primary_gate is True
    assert result.compatibility["can_run"] is True
    assert result.compatibility["arch_id"] == "qwen3-next-mtp"


def test_qwen_mtp_without_runtime_contract_is_family_runnable(monkeypatch, tmp_path):
    from mtplx import artifacts
    from mtplx.artifacts import MTPInspection

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "mtp_num_hidden_layers": 1,
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "vocab_size": 248320,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        artifacts,
        "inspect_mtp_tensors",
        lambda *_args, **_kwargs: MTPInspection(
            mtp_file=str(tmp_path / "mtp.safetensors"),
            exists=True,
            tensor_count=15,
            missing_expected_keys=(),
        ),
    )

    result = inspect_model(tmp_path)

    assert result.passes_primary_gate is True
    assert result.compatibility["tier"] == "family-compatible-unverified"
    assert result.compatibility["can_run"] is True
    assert result.compatibility["exit_code"] == 0
    assert result.compatibility["unsafe_force_required"] is False
    assert result.compatibility["runtime_compatibility"] == "native-family-gated"


def test_qwen_embedded_mtp_index_without_runtime_contract_is_family_runnable(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "mtp_num_hidden_layers": 1,
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "vocab_size": 248320,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    **{key: "model-00001-of-00001.safetensors" for key in EXPECTED_MTP_KEYS},
                    "model.layers.0.mlp.down_proj.weight": "model-00001-of-00001.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    result = inspect_model(tmp_path)

    assert result.mtp is not None
    assert result.mtp.exists is True
    assert result.mtp.metadata_only is True
    assert result.mtp.passes_tensor_gate is True
    assert result.compatibility["tier"] == "family-compatible-unverified"
    assert result.compatibility["can_run"] is True


def test_qwen3_next_architecture_without_mtp_sidecar_is_unverified(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3NextForCausalLM"],
                "model_type": "qwen3_next",
            }
        ),
        encoding="utf-8",
    )

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "architecture-compatible-but-unverified"
    assert result.compatibility["exit_code"] == 3
    assert result.compatibility["runtime_compatibility"] == "missing-mtp-weights"


def test_architecture_catalog_tracks_main_mtp_families():
    ids = {row["arch_id"] for row in architecture_catalog()}
    assert "qwen3-next-mtp" in ids
    assert "deepseek-v3-mtp" in ids
    assert "glm4-moe-mtp" in ids
    assert "minimax-m2-mtp" in ids
    assert "gemma-mtp" in ids


def test_deepseek_mtp_without_runtime_contract_needs_contract(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV3ForCausalLM"],
                "model_type": "deepseek_v3",
                "num_nextn_predict_layers": 2,
            }
        ),
        encoding="utf-8",
    )

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "architecture-compatible-but-unverified"
    assert result.compatibility["exit_code"] == 3
    assert result.compatibility["arch_id"] == "deepseek-v3-mtp"
    assert result.compatibility["recognized"] is True
    assert result.compatibility["can_run"] is False
    assert result.compatibility["unsafe_force_required"] is False
    assert result.compatibility["runtime_compatibility"] == "needs-contract"
    assert result.compatibility["mtp_supported"] == "recognized"

    with pytest.raises(UnverifiedArchitectureError):
        require_verified_or_raise(result, unsafe_force_unverified=True, yes=True)


def test_deepseek_mtp_with_runtime_contract_and_model_file_is_verified(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV3ForCausalLM"],
                "model_type": "deepseek_v3",
                "num_nextn_predict_layers": 2,
                "num_hidden_layers": 61,
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "model.layers.61.enorm.weight": np.ones((1,), dtype=np.float32),
            "model.layers.62.enorm.weight": np.ones((1,), dtype=np.float32),
        },
        tmp_path / "model.safetensors",
    )
    _write_runtime_contract(tmp_path, arch_id="deepseek-v3-mtp")

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "verified"
    assert result.compatibility["can_run"] is True
    assert result.compatibility["recommended_backend"] == "deepseek_mtp"
    assert result.compatibility["runtime_compatibility"] == "native-contract-gated"


def test_deepseek_mtp_with_family_layer_weights_is_runnable_without_contract(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV3ForCausalLM"],
                "model_type": "deepseek_v3",
                "num_nextn_predict_layers": 2,
                "num_hidden_layers": 61,
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "model.layers.61.enorm.weight": np.ones((1,), dtype=np.float32),
            "model.layers.62.hnorm.weight": np.ones((1,), dtype=np.float32),
        },
        tmp_path / "model.safetensors",
    )

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "family-compatible-unverified"
    assert result.compatibility["arch_id"] == "deepseek-v3-mtp"
    assert result.compatibility["can_run"] is True
    assert result.compatibility["exit_code"] == 0
    assert result.compatibility["runtime_compatibility"] == "native-family-gated"


def test_glm4_moe_mtp_without_runtime_contract_needs_contract(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Glm4MoeForCausalLM"],
                "model_type": "glm4_moe",
                "num_nextn_predict_layers": 1,
            }
        ),
        encoding="utf-8",
    )

    result = inspect_model(tmp_path)

    assert result.compatibility["arch_id"] == "glm4-moe-mtp"
    assert result.compatibility["runtime_compatibility"] == "needs-contract"
    assert result.compatibility["recognized"] is True
    assert result.compatibility["recommended_backend"] == "glm_mtp"
    assert result.compatibility["can_run"] is False


def test_glm4_moe_mtp_with_runtime_contract_and_model_file_is_verified(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Glm4MoeForCausalLM"],
                "model_type": "glm4_moe",
                "num_nextn_predict_layers": 1,
                "num_hidden_layers": 47,
            }
        ),
        encoding="utf-8",
    )
    save_file({"model.layers.47.enorm.weight": np.ones((1,), dtype=np.float32)}, tmp_path / "model.safetensors")
    _write_runtime_contract(tmp_path, arch_id="glm4-moe-mtp")

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "verified"
    assert result.compatibility["can_run"] is True
    assert result.compatibility["recommended_backend"] == "glm_mtp"
    assert result.compatibility["runtime_compatibility"] == "native-contract-gated"


def test_glm4_moe_mtp_with_family_layer_weights_is_runnable_without_contract(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Glm4MoeForCausalLM"],
                "model_type": "glm4_moe",
                "num_nextn_predict_layers": 1,
                "num_hidden_layers": 47,
            }
        ),
        encoding="utf-8",
    )
    save_file({"model.layers.47.hnorm.weight": np.ones((1,), dtype=np.float32)}, tmp_path / "model.safetensors")

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "family-compatible-unverified"
    assert result.compatibility["arch_id"] == "glm4-moe-mtp"
    assert result.compatibility["can_run"] is True
    assert result.compatibility["exit_code"] == 0
    assert result.compatibility["runtime_compatibility"] == "native-family-gated"


def test_glm4_moe_mtp_with_unrelated_model_file_still_needs_contract(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Glm4MoeForCausalLM"],
                "model_type": "glm4_moe",
                "num_nextn_predict_layers": 1,
                "num_hidden_layers": 47,
            }
        ),
        encoding="utf-8",
    )
    save_file({"model.layers.0.self_attn.q_proj.weight": np.ones((1,), dtype=np.float32)}, tmp_path / "model.safetensors")

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "architecture-compatible-but-unverified"
    assert result.compatibility["can_run"] is False
    assert result.compatibility["runtime_compatibility"] == "needs-contract"


def test_glm4_moe_mtp_sidecar_layer_keys_are_family_runnable(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Glm4MoeForCausalLM"],
                "model_type": "glm4_moe",
                "num_nextn_predict_layers": 1,
                "num_hidden_layers": 47,
                "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    save_file({"layers.0.enorm.weight": np.ones((1,), dtype=np.float32)}, tmp_path / "mtp.safetensors")

    result = inspect_model(tmp_path)

    assert result.mtp is not None
    assert result.mtp.exists is True
    assert result.compatibility["tier"] == "family-compatible-unverified"
    assert result.compatibility["can_run"] is True
    assert result.compatibility["runtime_compatibility"] == "native-family-gated"


def test_glm4_moe_lite_mtp_with_runtime_contract_and_model_file_is_verified(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Glm4MoeLiteForCausalLM"],
                "model_type": "glm4_moe_lite",
                "num_nextn_predict_layers": 1,
                "num_hidden_layers": 47,
            }
        ),
        encoding="utf-8",
    )
    save_file({"model.layers.47.enorm.weight": np.ones((1,), dtype=np.float32)}, tmp_path / "model.safetensors")
    _write_runtime_contract(tmp_path, arch_id="glm4-moe-lite-mtp")

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "verified"
    assert result.compatibility["can_run"] is True
    assert result.compatibility["recommended_backend"] == "glm_mtp"
    assert result.compatibility["runtime_compatibility"] == "native-contract-gated"


def test_glm_moe_dsa_mtp_without_runtime_contract_needs_contract(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["GlmMoeDsaForCausalLM"],
                "model_type": "glm_moe_dsa",
                "num_nextn_predict_layers": 2,
            }
        ),
        encoding="utf-8",
    )

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "architecture-compatible-but-unverified"
    assert result.compatibility["arch_id"] == "glm-moe-dsa-mtp"
    assert result.compatibility["recommended_backend"] == "deepseek_mtp"
    assert result.compatibility["runtime_compatibility"] == "needs-contract"
    assert result.compatibility["mtp_supported"] == "recognized"
    assert result.compatibility["can_run"] is False


def test_glm_moe_dsa_mtp_with_runtime_contract_and_model_file_is_verified(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["GlmMoeDsaForCausalLM"],
                "model_type": "glm_moe_dsa",
                "num_nextn_predict_layers": 2,
                "num_hidden_layers": 61,
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "model.layers.61.enorm.weight": np.ones((1,), dtype=np.float32),
            "model.layers.62.enorm.weight": np.ones((1,), dtype=np.float32),
        },
        tmp_path / "model.safetensors",
    )
    _write_runtime_contract(tmp_path, arch_id="glm-moe-dsa-mtp")

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "verified"
    assert result.compatibility["can_run"] is True
    assert result.compatibility["recommended_backend"] == "deepseek_mtp"
    assert result.compatibility["runtime_compatibility"] == "native-contract-gated"


def test_mimo_mtp_with_runtime_contract_and_model_file_is_verified(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["MiMoForCausalLM"],
                "model_type": "mimo",
                "num_nextn_predict_layers": 2,
                "num_hidden_layers": 46,
            }
        ),
        encoding="utf-8",
    )
    save_file({"model.mtp_layers.0.token_layernorm.weight": np.ones((1,), dtype=np.float32)}, tmp_path / "model.safetensors")
    _write_runtime_contract(tmp_path, arch_id="mimo-mtp")

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "verified"
    assert result.compatibility["can_run"] is True
    assert result.compatibility["recommended_backend"] == "mimo_mtp"
    assert result.compatibility["runtime_compatibility"] == "native-contract-gated"


def test_mimo_mtp_with_family_layer_weights_is_runnable_without_contract(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["MiMoForCausalLM"],
                "model_type": "mimo",
                "num_nextn_predict_layers": 2,
                "num_hidden_layers": 46,
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {"model.mtp_layers.0.hidden_layernorm.weight": np.ones((1,), dtype=np.float32)},
        tmp_path / "model.safetensors",
    )

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "family-compatible-unverified"
    assert result.compatibility["arch_id"] == "mimo-mtp"
    assert result.compatibility["can_run"] is True
    assert result.compatibility["runtime_compatibility"] == "native-family-gated"


def test_mimo_mtp_with_unrelated_model_file_still_needs_contract(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["MiMoForCausalLM"],
                "model_type": "mimo",
                "num_nextn_predict_layers": 2,
                "num_hidden_layers": 46,
            }
        ),
        encoding="utf-8",
    )
    save_file({"model.layers.0.self_attn.q_proj.weight": np.ones((1,), dtype=np.float32)}, tmp_path / "model.safetensors")

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "architecture-compatible-but-unverified"
    assert result.compatibility["can_run"] is False
    assert result.compatibility["runtime_compatibility"] == "needs-contract"


def test_minimax_m2_with_nextn_marker_is_recognized_backend_pending(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["MiniMaxM2ForCausalLM"],
                "model_type": "minimax_m2",
                "num_nextn_predict_layers": 1,
            }
        ),
        encoding="utf-8",
    )

    result = inspect_model(tmp_path)

    assert result.compatibility["arch_id"] == "minimax-m2-mtp"
    assert result.compatibility["runtime_compatibility"] == "recognized-backend-pending"
    assert result.compatibility["can_run"] is False


def test_minimax_m2_with_num_mtp_modules_marker_is_recognized_backend_pending(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["MiniMaxM2ForCausalLM"],
                "model_type": "minimax_m2",
                "num_mtp_modules": 2,
            }
        ),
        encoding="utf-8",
    )

    result = inspect_model(tmp_path)

    assert result.mtp_num_hidden_layers == 2
    assert result.compatibility["arch_id"] == "minimax-m2-mtp"
    assert result.compatibility["runtime_compatibility"] == "recognized-backend-pending"
    assert result.compatibility["can_run"] is False


def test_nemotron_h_mtp_without_weights_needs_contract(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["NemotronHForCausalLM"],
                "model_type": "nemotron_h",
                "num_nextn_predict_layers": 1,
                "num_hidden_layers": 52,
                "mtp_hybrid_override_pattern": "*E",
            }
        ),
        encoding="utf-8",
    )

    result = inspect_model(tmp_path)

    assert result.mtp_pattern == "*E"
    assert result.compatibility["arch_id"] == "nemotron-h-mtp"
    assert result.compatibility["runtime_compatibility"] == "needs-contract"
    assert result.compatibility["can_run"] is False


def test_nemotron_h_mtp_with_runtime_contract_and_sidecar_is_verified(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["NemotronHForCausalLM"],
                "model_type": "nemotron_h",
                "num_nextn_predict_layers": 1,
                "num_hidden_layers": 52,
                "mtp_hybrid_override_pattern": "*E",
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "mtp.layers.0.enorm.weight": np.ones((1,), dtype=np.float32),
            "mtp.layers.0.hnorm.weight": np.ones((1,), dtype=np.float32),
            "mtp.layers.0.eh_proj.weight": np.ones((1,), dtype=np.float32),
            "mtp.layers.0.norm.weight": np.ones((1,), dtype=np.float32),
            "mtp.layers.0.mixer.q_proj.weight": np.ones((1,), dtype=np.float32),
            "mtp.layers.1.norm.weight": np.ones((1,), dtype=np.float32),
            "mtp.layers.1.mixer.gate.weight": np.ones((1,), dtype=np.float32),
            "mtp.layers.1.final_layernorm.weight": np.ones((1,), dtype=np.float32),
        },
        tmp_path / "mtp.safetensors",
    )
    _write_runtime_contract(tmp_path, arch_id="nemotron-h-mtp")

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "verified"
    assert result.compatibility["can_run"] is True
    assert result.compatibility["recommended_backend"] == "nemotron_h_mtp"
    assert result.compatibility["runtime_compatibility"] == "native-contract-gated"


def test_nemotron_h_mtp_with_family_sidecar_is_runnable_without_contract(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["NemotronHForCausalLM"],
                "model_type": "nemotron_h",
                "num_nextn_predict_layers": 1,
                "num_hidden_layers": 52,
                "mtp_hybrid_override_pattern": "*E",
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "mtp.layers.0.enorm.weight": np.ones((1,), dtype=np.float32),
            "mtp.layers.0.hnorm.weight": np.ones((1,), dtype=np.float32),
            "mtp.layers.0.eh_proj.weight": np.ones((1,), dtype=np.float32),
            "mtp.layers.0.norm.weight": np.ones((1,), dtype=np.float32),
            "mtp.layers.0.mixer.q_proj.weight": np.ones((1,), dtype=np.float32),
            "mtp.layers.1.norm.weight": np.ones((1,), dtype=np.float32),
            "mtp.layers.1.mixer.gate.weight": np.ones((1,), dtype=np.float32),
            "mtp.layers.1.final_layernorm.weight": np.ones((1,), dtype=np.float32),
        },
        tmp_path / "mtp.safetensors",
    )

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "family-compatible-unverified"
    assert result.compatibility["can_run"] is True
    assert result.compatibility["runtime_compatibility"] == "native-family-gated"


def test_nemotron_h_mtp_rejects_unsupported_mtp_pattern(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["NemotronHForCausalLM"],
                "model_type": "nemotron_h",
                "num_nextn_predict_layers": 1,
                "num_hidden_layers": 52,
                "mtp_hybrid_override_pattern": "M*",
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "mtp.layers.0.enorm.weight": np.ones((1,), dtype=np.float32),
            "mtp.layers.0.hnorm.weight": np.ones((1,), dtype=np.float32),
            "mtp.layers.0.eh_proj.weight": np.ones((1,), dtype=np.float32),
            "mtp.layers.0.norm.weight": np.ones((1,), dtype=np.float32),
            "mtp.layers.0.mixer.q_proj.weight": np.ones((1,), dtype=np.float32),
            "mtp.layers.1.norm.weight": np.ones((1,), dtype=np.float32),
            "mtp.layers.1.mixer.q_proj.weight": np.ones((1,), dtype=np.float32),
            "mtp.layers.1.final_layernorm.weight": np.ones((1,), dtype=np.float32),
        },
        tmp_path / "mtp.safetensors",
    )

    result = inspect_model(tmp_path)

    assert result.compatibility["arch_id"] == "nemotron-h-mtp"
    assert result.compatibility["can_run"] is False
    assert result.compatibility["runtime_compatibility"] == "needs-contract"


def test_gemma4_without_mtp_marker_stays_no_mtp(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}),
        encoding="utf-8",
    )

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "no-MTP"
    assert result.compatibility["arch_id"] is None
    assert result.compatibility["recognized"] is False


def test_gemma4_with_mtp_marker_is_recognized_backend_pending(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma4ForCausalLM"],
                "model_type": "gemma4",
                "num_nextn_predict_layers": 1,
            }
        ),
        encoding="utf-8",
    )

    result = inspect_model(tmp_path)

    assert result.compatibility["arch_id"] == "gemma-mtp"
    assert result.compatibility["runtime_compatibility"] == "recognized-backend-pending"
    assert result.compatibility["mtp_supported"] == "recognized"


@pytest.mark.parametrize(
    ("architecture", "model_type", "expected_arch_id"),
    [
        ("DeepseekV32ForCausalLM", "deepseek_v32", "deepseek-v3-mtp"),
        ("GlmMoeDsaForCausalLM", "glm_moe_dsa", "glm-moe-dsa-mtp"),
        ("DeepseekV4ForCausalLM", "deepseek_v4", "deepseek-v4-mtp"),
        ("Glm4MoeLiteForCausalLM", "glm4_moe_lite", "glm4-moe-lite-mtp"),
        ("GlmOcrForCausalLM", "glm_ocr", "glm-ocr-mtp"),
        ("MiniMaxM2ForCausalLM", "minimax_m2", "minimax-m2-mtp"),
        ("MiniMaxM25ForCausalLM", "minimax_m2_5", "minimax-m2-mtp"),
        ("MiniMaxM26ForCausalLM", "minimax_m2_6", "minimax-m2-mtp"),
        ("MiMoForCausalLM", "mimo", "mimo-mtp"),
        ("Ernie45MoeForCausalLM", "ernie4_5_moe", "ernie-mtp"),
        ("NemotronHForCausalLM", "nemotron_h", "nemotron-h-mtp"),
        ("ExaoneMoeForCausalLM", "exaone_moe", "exaone-moe-mtp"),
        ("Exaone45ForCausalLM", "exaone4_5", "exaone4-5-mtp"),
        ("LongCatFlashForCausalLM", "longcat_flash", "longcat-flash-mtp"),
        ("OpenPanguForCausalLM", "openpangu", "pangu-ultra-moe-mtp"),
        ("Step3P5ForCausalLM", "step3p5", "step3p5-mtp"),
        ("HyV3ForCausalLM", "hy_v3", "hy-v3-mtp"),
    ],
)
def test_big_mtp_architecture_markers_are_recognized_backend_pending(
    tmp_path,
    architecture,
    model_type,
    expected_arch_id,
):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": [architecture],
                "model_type": model_type,
                "num_nextn_predict_layers": 1,
            }
        ),
        encoding="utf-8",
    )

    result = inspect_model(tmp_path)

    assert result.compatibility["arch_id"] == expected_arch_id
    assert result.compatibility["tier"] == "architecture-compatible-but-unverified"
    assert result.compatibility["recognized"] is True
    assert result.compatibility["can_run"] is False
    assert result.compatibility["unsafe_force_required"] is False
    expected_runtime = (
        "needs-contract"
        if expected_arch_id
        in {
            "deepseek-v3-mtp",
            "glm-moe-dsa-mtp",
            "glm4-moe-mtp",
            "glm4-moe-lite-mtp",
            "mimo-mtp",
            "nemotron-h-mtp",
        }
        else "recognized-backend-pending"
    )
    assert result.compatibility["runtime_compatibility"] == expected_runtime
    assert result.compatibility["mtp_supported"] == "recognized"


def test_recognized_non_qwen_runtime_contract_stays_backend_pending(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV3ForCausalLM"],
                "model_type": "deepseek_v3",
                "num_nextn_predict_layers": 2,
            }
        ),
        encoding="utf-8",
    )
    _write_runtime_contract(tmp_path, arch_id="deepseek-v3-mtp")

    result = inspect_model(tmp_path)

    assert result.compatibility["arch_id"] == "deepseek-v3-mtp"
    assert result.compatibility["tier"] == "architecture-compatible-but-unverified"
    assert result.compatibility["recognized"] is True
    assert result.compatibility["can_run"] is False
    assert result.compatibility["runtime_contract"]["arch_id"] == "deepseek-v3-mtp"
    assert result.compatibility["runtime_compatibility"] == "needs-grafting"


def test_llama_without_mtp_is_no_mtp(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["LlamaForCausalLM"], "model_type": "llama"}),
        encoding="utf-8",
    )

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "no-MTP"
    assert result.compatibility["exit_code"] == 2


def test_hf_qwen_mtp_without_runtime_contract_is_family_runnable(monkeypatch):
    from mtplx import artifacts

    calls = []

    def fake_files(repo_id):
        assert repo_id == "Qwen/Qwen3-Next-80B-A3B-Instruct"
        return {"config.json", "mtp.safetensors", "model.safetensors.index.json"}, None

    def fake_json(repo_id, filename):
        if filename == "config.json":
            return (
                {
                    "architectures": ["Qwen3_5ForConditionalGeneration"],
                    "model_type": "qwen3_5",
                    "mtp_num_hidden_layers": 1,
                },
                "/tmp/config.json",
                None,
            )
        if filename == "mtplx_runtime.json":
            return None, None, "404 Client Error: entry not found"
        raise AssertionError(filename)

    def fake_keys(repo_id, filename):
        calls.append((repo_id, filename))
        return tuple(sorted(EXPECTED_MTP_KEYS)), None

    monkeypatch.setattr(artifacts, "_hf_list_repo_files", fake_files)
    monkeypatch.setattr(artifacts, "_hf_download_json", fake_json)
    monkeypatch.setattr(artifacts, "_remote_safetensors_keys", fake_keys)

    result = inspect_model("Qwen/Qwen3-Next-80B-A3B-Instruct")

    assert result.source == "hf"
    assert result.mtp is not None
    assert result.mtp.metadata_only is True
    assert result.compatibility["tier"] == "family-compatible-unverified"
    assert result.compatibility["can_run"] is True
    assert result.compatibility["exit_code"] == 0
    assert calls == [("Qwen/Qwen3-Next-80B-A3B-Instruct", "mtp.safetensors")]


def test_hf_qwen_embedded_mtp_index_is_family_runnable(monkeypatch):
    from mtplx import artifacts

    def fake_files(repo_id):
        assert repo_id == "Qwen/Qwen3.5-4B"
        return {
            "config.json",
            "model.safetensors.index.json",
            "model.safetensors-00001-of-00002.safetensors",
            "model.safetensors-00002-of-00002.safetensors",
        }, None

    def fake_json(repo_id, filename):
        if filename == "config.json":
            return (
                {
                    "architectures": ["Qwen3_5ForConditionalGeneration"],
                    "model_type": "qwen3_5",
                    "text_config": {
                        "model_type": "qwen3_5_text",
                        "mtp_num_hidden_layers": 1,
                    },
                },
                "/tmp/config.json",
                None,
            )
        if filename == "model.safetensors.index.json":
            return (
                {
                    "metadata": {},
                    "weight_map": {
                        **{key: "model.safetensors-00001-of-00002.safetensors" for key in EXPECTED_MTP_KEYS},
                        "model.language_model.layers.0.mlp.down_proj.weight": "model.safetensors-00001-of-00002.safetensors",
                    },
                },
                "/tmp/model.safetensors.index.json",
                None,
            )
        if filename == "mtplx_runtime.json":
            return None, None, "404 Client Error: entry not found"
        raise AssertionError(filename)

    monkeypatch.setattr(artifacts, "_hf_list_repo_files", fake_files)
    monkeypatch.setattr(artifacts, "_hf_download_json", fake_json)

    result = inspect_model("Qwen/Qwen3.5-4B")

    assert result.source == "hf"
    assert result.mtp is not None
    assert result.mtp.exists is True
    assert result.mtp.metadata_only is True
    assert result.mtp.mtp_file == "model.safetensors.index.json::embedded"
    assert result.mtp.passes_tensor_gate is True
    assert result.compatibility["tier"] == "family-compatible-unverified"
    assert result.compatibility["can_run"] is True
    assert result.compatibility["exit_code"] == 0


def test_qwen_runtime_loader_reads_embedded_mtp_weights(tmp_path):
    from mtplx.mtp_patch import _load_embedded_mtp_weights

    save_file(
        {
            **{key: np.ones((1,), dtype=np.float32) for key in EXPECTED_MTP_KEYS},
            "model.language_model.layers.0.mlp.down_proj.weight": np.ones((1,), dtype=np.float32),
        },
        tmp_path / "model.safetensors",
    )

    weights = _load_embedded_mtp_weights(
        tmp_path,
        {"model_type": "qwen3_5", "mtp_num_hidden_layers": 1},
    )

    assert sorted(weights) == sorted(key.removeprefix("mtp.") for key in EXPECTED_MTP_KEYS)
    assert "fc.weight" in weights


def test_qwen_runtime_loader_reads_bf16_embedded_mtp_weights(tmp_path):
    import mlx.core as mx
    from mtplx.mtp_patch import _load_embedded_mtp_weights

    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        {
            **{key: mx.ones((1,), dtype=mx.bfloat16) for key in EXPECTED_MTP_KEYS},
            "model.language_model.layers.0.mlp.down_proj.weight": mx.ones(
                (1,), dtype=mx.bfloat16
            ),
        },
    )

    weights = _load_embedded_mtp_weights(
        tmp_path,
        {"model_type": "qwen3_5", "mtp_num_hidden_layers": 1},
    )

    assert sorted(weights) == sorted(key.removeprefix("mtp.") for key in EXPECTED_MTP_KEYS)
    assert weights["fc.weight"].dtype == mx.bfloat16


def test_hf_verified_contract_passes_metadata_gate(monkeypatch):
    from mtplx import artifacts

    def fake_files(_repo_id):
        return {"config.json", "mtplx_runtime.json", "mtp.safetensors"}, None

    def fake_json(_repo_id, filename):
        if filename == "config.json":
            return (
                {
                    "architectures": ["Qwen3_5ForConditionalGeneration"],
                    "model_type": "qwen3_5",
                    "mtp_num_hidden_layers": 1,
                    "mtplx_mtp_quantization": {
                        "prequantized": True,
                        "bits": 4,
                        "group_size": 32,
                        "mode": "affine",
                    },
                },
                "/tmp/config.json",
                None,
            )
        if filename == "mtplx_runtime.json":
            return (
                {
                    "mtplx_version": "0.1.4",
                    "arch_id": "qwen3-next-mtp",
                    "mtp_depth_max": 3,
                    "recommended_profile": "stable",
                    "exactness_baseline": {"phase0h": "smoke", "max_abs_diff": 0.0},
                    "verified_on": {"timestamp": "2026-05-02T00:00:00Z"},
                },
                "/tmp/mtplx_runtime.json",
                None,
            )
        raise AssertionError(filename)

    monkeypatch.setattr(artifacts, "_hf_list_repo_files", fake_files)
    monkeypatch.setattr(artifacts, "_hf_download_json", fake_json)
    monkeypatch.setattr(
        artifacts,
        "_remote_safetensors_keys",
        lambda _repo_id, _filename: (tuple(sorted(EXPECTED_PREQUANTIZED_MTP_KEYS)), None),
    )

    result = inspect_model("https://huggingface.co/mtplx/example/tree/main")

    assert result.source == "hf"
    assert result.mtp is not None
    assert result.mtp.metadata_only is True
    assert result.mtp.passes_tensor_gate is True
    assert result.compatibility["tier"] == "verified"
    assert result.compatibility["can_run"] is True
    assert result.runtime_contract_path == "/tmp/mtplx_runtime.json"


def test_hf_llama_without_mtp_is_no_mtp(monkeypatch):
    from mtplx import artifacts

    monkeypatch.setattr(
        artifacts,
        "_hf_list_repo_files",
        lambda _repo_id: ({"config.json", "model.safetensors.index.json"}, None),
    )

    def fake_json(_repo_id, filename):
        if filename == "config.json":
            return (
                {"architectures": ["LlamaForCausalLM"], "model_type": "llama"},
                "/tmp/config.json",
                None,
            )
        if filename == "mtplx_runtime.json":
            return None, None, "404 Client Error: not found"
        raise AssertionError(filename)

    monkeypatch.setattr(artifacts, "_hf_download_json", fake_json)

    result = inspect_model("https://huggingface.co/meta-llama/Llama-3.2-1B")

    assert result.source == "hf"
    assert result.compatibility["tier"] == "no-MTP"
    assert result.compatibility["exit_code"] == 2
