from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from mtplx.artifacts import expected_mtp_file, inspect_model, inspect_mtp_tensors
from mtplx.backends.registry import (
    UnverifiedArchitectureError,
    architecture_catalog,
    require_verified_or_raise,
)
from mtplx.constants import (
    EXPECTED_ALL_PREQUANTIZED_MTP_KEYS,
    EXPECTED_MTP_KEYS,
    EXPECTED_PREQUANTIZED_MTP_KEYS,
    EXPECTED_QWEN_MOE_MTP_KEYS,
    EXPECTED_QWEN_MOE_PREQUANTIZED_MTP_KEYS,
    EXPECTED_QWEN_MOE_SWITCH_MLP_MTP_KEYS,
    EXPECTED_QWEN_MOE_SWITCH_MLP_PREQUANTIZED_MTP_KEYS,
)


def _write_runtime_contract(
    path,
    *,
    arch_id="qwen3-next-mtp",
    profile="stable",
    exactness_baseline=None,
    speed_evidence=None,
    recommended_draft_lm_head=None,
    recommended_draft_sampler=None,
    runtime_env_overrides=None,
):
    contract = {
        "mtplx_version": "0.1.4",
        "arch_id": arch_id,
        "mtp_depth_max": 3,
        "recommended_profile": profile,
        "exactness_baseline": exactness_baseline
        if exactness_baseline is not None
        else {"phase0h": "smoke", "max_abs_diff": 0.0},
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
    if runtime_env_overrides is not None:
        contract["runtime_env_overrides"] = runtime_env_overrides
    if speed_evidence is not None:
        contract["speed_evidence"] = speed_evidence
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
    assert "complete model" in result.compatibility["message"]
    assert "original source with Forge" in result.compatibility["message"]
    assert "cannot safely attach an arbitrary sidecar" in result.compatibility["message"]
    assert "graft an MTP sidecar" not in result.compatibility["message"]


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
        runtime_env_overrides={
            "MTPLX_LAZY_VERIFY_LOGITS": False,
            "MTPLX_BATCH_TARGET_ARRAYS": "0",
        },
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
    assert result.compatibility["runtime_contract"]["runtime_env_overrides"] == {
        "MTPLX_LAZY_VERIFY_LOGITS": "0",
        "MTPLX_BATCH_TARGET_ARRAYS": "0",
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


def test_dense_fp16_mtp_sidecar_reports_fp16_format(tmp_path):
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "mtp_num_hidden_layers": 1,
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "vocab_size": 248320,
        "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(
        {key: np.ones((1,), dtype=np.float16) for key in EXPECTED_MTP_KEYS},
        tmp_path / "mtp.safetensors",
    )

    result = inspect_mtp_tensors(tmp_path, config)

    assert result.sidecar_format == "fp16"
    assert result.tensor_count == 15
    assert result.missing_expected_keys == ()
    assert result.extra_keys == ()


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


def test_qwen_moe_mtp_sidecar_accepts_native_expert_layout(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe_text",
                "mtp_num_hidden_layers": 1,
                "hidden_size": 2048,
                "num_hidden_layers": 40,
                "vocab_size": 248320,
                "num_experts": 256,
                "num_experts_per_tok": 8,
                "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {key: np.ones((1,), dtype=np.float32) for key in EXPECTED_QWEN_MOE_MTP_KEYS},
        tmp_path / "mtp.safetensors",
    )
    _write_runtime_contract(tmp_path, profile="performance-cold")

    result = inspect_model(tmp_path)

    assert result.mtp is not None
    assert result.mtp.sidecar_format == "bf16-qwen-moe"
    assert result.mtp.tensor_count == 19
    assert result.mtp.missing_expected_keys == ()
    assert result.mtp.extra_keys == ()
    assert result.passes_primary_gate is True
    assert result.compatibility["tier"] == "verified"


def test_qwen_moe_prequantized_mtp_sidecar_accepts_mlx_affine_layout(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe_text",
                "mtp_num_hidden_layers": 1,
                "hidden_size": 2048,
                "num_hidden_layers": 40,
                "vocab_size": 248320,
                "num_experts": 256,
                "num_experts_per_tok": 8,
                "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
                "mtplx_mtp_quantization": {
                    "policy": "cyankiwi",
                    "bits": 4,
                    "group_size": 64,
                    "mode": "affine",
                    "prequantized": True,
                },
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            key: np.ones((1,), dtype=np.float32)
            for key in EXPECTED_QWEN_MOE_PREQUANTIZED_MTP_KEYS
        },
        tmp_path / "mtp.safetensors",
    )
    _write_runtime_contract(tmp_path, profile="performance-cold")

    result = inspect_model(tmp_path)

    assert result.mtp is not None
    assert result.mtp.sidecar_format == "prequantized-mlx-affine-qwen-moe"
    assert result.mtp.tensor_count == 37
    assert result.mtp.missing_expected_keys == ()
    assert result.mtp.extra_keys == ()
    assert result.passes_primary_gate is True
    assert result.compatibility["tier"] == "verified"


def test_qwen_moe_prequantized_mtp_sidecar_accepts_numbered_expert_layout(tmp_path):
    num_experts = 4
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "mtp_num_hidden_layers": 1,
                    "hidden_size": 2048,
                    "num_hidden_layers": 40,
                    "vocab_size": 248320,
                    "num_experts": num_experts,
                    "num_experts_per_tok": 8,
                },
                "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
                "mtplx_mtp_quantization": {
                    "policy": "cyankiwi",
                    "bits": 4,
                    "group_size": 32,
                    "mode": "affine",
                    "prequantized": True,
                },
            }
        ),
        encoding="utf-8",
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
    for expert_index in range(num_experts):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"{base}.mlp.experts.{expert_index}.{proj}"
            keys.update({f"{prefix}.weight", f"{prefix}.scales", f"{prefix}.biases"})
    save_file(
        {key: np.ones((1,), dtype=np.float32) for key in keys},
        tmp_path / "mtp.safetensors",
    )
    _write_runtime_contract(tmp_path, profile="performance-cold")

    result = inspect_model(tmp_path)

    assert result.mtp is not None
    assert result.mtp.sidecar_format == "prequantized-mlx-affine-qwen-moe-experts"
    assert result.mtp.tensor_count == 4 + 13 + (num_experts * 3 * 3)
    assert result.mtp.missing_expected_keys == ()
    assert result.mtp.extra_keys == ()
    assert result.passes_primary_gate is True


def test_qwen_moe_mtp_sidecar_accepts_stacked_switch_mlp_layout(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "mtp_num_hidden_layers": 1,
                    "hidden_size": 2048,
                    "num_hidden_layers": 40,
                    "vocab_size": 248320,
                    "num_experts": 256,
                    "num_experts_per_tok": 8,
                },
                "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            key: np.ones((1,), dtype=np.float32)
            for key in EXPECTED_QWEN_MOE_SWITCH_MLP_MTP_KEYS
        },
        tmp_path / "mtp.safetensors",
    )
    _write_runtime_contract(tmp_path, profile="performance-cold")

    result = inspect_model(tmp_path)

    assert result.mtp is not None
    assert result.mtp.sidecar_format == "bf16-qwen-moe-switch-mlx"
    assert result.mtp.tensor_count == len(EXPECTED_QWEN_MOE_SWITCH_MLP_MTP_KEYS)
    assert result.mtp.missing_expected_keys == ()
    assert result.mtp.extra_keys == ()
    assert result.passes_primary_gate is True


def test_qwen_moe_mtp_sidecar_accepts_prequantized_stacked_switch_mlp_layout(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "mtp_num_hidden_layers": 1,
                    "hidden_size": 2048,
                    "num_hidden_layers": 40,
                    "vocab_size": 248320,
                    "num_experts": 256,
                    "num_experts_per_tok": 8,
                },
                "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
                "mtplx_mtp_quantization": {
                    "policy": "all",
                    "bits": 4,
                    "group_size": 64,
                    "mode": "affine",
                    "prequantized": True,
                },
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            key: np.ones((1,), dtype=np.float32)
            for key in EXPECTED_QWEN_MOE_SWITCH_MLP_PREQUANTIZED_MTP_KEYS
        },
        tmp_path / "mtp.safetensors",
    )
    _write_runtime_contract(tmp_path, profile="performance-cold")

    result = inspect_model(tmp_path)

    assert result.mtp is not None
    assert result.mtp.sidecar_format == "prequantized-mlx-affine-qwen-moe-switch-mlx"
    assert result.mtp.tensor_count == len(
        EXPECTED_QWEN_MOE_SWITCH_MLP_PREQUANTIZED_MTP_KEYS
    )
    assert result.mtp.missing_expected_keys == ()
    assert result.mtp.extra_keys == ()
    assert result.passes_primary_gate is True


def test_qwen_moe_mtp_sidecar_accepts_mixed_switch_mlp_prequantization(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "mtp_num_hidden_layers": 1,
                    "hidden_size": 2048,
                    "num_hidden_layers": 40,
                    "vocab_size": 248320,
                    "num_experts": 256,
                    "num_experts_per_tok": 8,
                },
                "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
                "mtplx_mtp_quantization": {
                    "policy": "cyankiwi",
                    "bits": 4,
                    "group_size": 32,
                    "mode": "affine",
                    "prequantized": True,
                },
            }
        ),
        encoding="utf-8",
    )
    switch_weight_keys = {
        "mtp.layers.0.mlp.switch_mlp.down_proj.weight",
        "mtp.layers.0.mlp.switch_mlp.gate_proj.weight",
        "mtp.layers.0.mlp.switch_mlp.up_proj.weight",
    }
    mixed_keys = set(EXPECTED_QWEN_MOE_SWITCH_MLP_MTP_KEYS)
    mixed_keys.update(
        key.rsplit(".", 1)[0] + suffix
        for key in switch_weight_keys
        for suffix in (".scales", ".biases")
    )
    save_file(
        {key: np.ones((1,), dtype=np.float32) for key in mixed_keys},
        tmp_path / "mtp.safetensors",
    )
    _write_runtime_contract(tmp_path, profile="performance-cold")

    result = inspect_model(tmp_path)

    assert result.mtp is not None
    assert result.mtp.sidecar_format == "prequantized-mlx-affine-qwen-moe-switch-mlx"
    assert result.mtp.tensor_count == len(mixed_keys)
    assert result.mtp.expected_tensor_count == len(mixed_keys)
    assert result.mtp.missing_expected_keys == ()
    assert result.mtp.extra_keys == ()
    assert result.passes_primary_gate is True


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


def test_runtime_contract_public_release_blocker_is_not_verified(monkeypatch, tmp_path):
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
    _write_runtime_contract(
        tmp_path,
        exactness_baseline={
            "public_release_blocker": True,
            "status": "candidate-build-only-benchmark-pending",
        },
    )

    result = inspect_model(tmp_path)

    assert result.passes_primary_gate is True
    assert result.compatibility["tier"] == "architecture-compatible-but-unverified"
    assert result.compatibility["can_run"] is True
    assert result.compatibility["runtime_compatibility"] == "runtime-contract-unverified"
    assert result.compatibility["unsafe_force_required"] is False
    assert "public_release_blocker" in result.compatibility["message"]


def test_runtime_contract_failed_speed_evidence_is_not_verified(monkeypatch, tmp_path):
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
    _write_runtime_contract(
        tmp_path,
        speed_evidence={
            "depth": 0,
            "verdict": "no_mtp_depth_beat_ar",
            "failure_reasons": ["no_mtp_depth_beat_ar"],
        },
    )

    result = inspect_model(tmp_path)

    assert result.passes_primary_gate is True
    assert result.compatibility["runtime_compatibility"] == "runtime-contract-unverified"
    assert "no_mtp_depth_beat_ar" in result.compatibility["message"]


def test_runtime_contract_pending_status_prefix_is_not_verified(monkeypatch, tmp_path):
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
    _write_runtime_contract(
        tmp_path,
        exactness_baseline={"status": "pending-cyankiwi-35b-moe-benchmark"},
    )

    result = inspect_model(tmp_path)

    assert result.passes_primary_gate is True
    assert result.compatibility["runtime_compatibility"] == "runtime-contract-unverified"
    assert "pending-cyankiwi-35b-moe-benchmark" in result.compatibility["message"]


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


def test_qwen_moe_numbered_body_layout_is_not_marked_runnable(monkeypatch, tmp_path):
    from mtplx import artifacts
    from mtplx.artifacts import MTPInspection

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "mtp_num_hidden_layers": 1,
                    "num_experts": 256,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    (
                        "language_model.model.layers.0.mlp.experts.0."
                        "down_proj.weight"
                    ): "model-00001-of-00001.safetensors",
                    (
                        "language_model.model.layers.0.mlp.gate.weight"
                    ): "model-00001-of-00001.safetensors",
                },
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
            expected_tensor_count=15,
        ),
    )

    result = inspect_model(tmp_path)

    assert result.passes_primary_gate is False
    assert result.compatibility["can_run"] is False
    assert result.compatibility["runtime_compatibility"] == "invalid-base-tensor-layout"
    assert "switch_mlp" in result.compatibility["message"]


def test_qwen_moe_switch_body_layout_remains_family_runnable(monkeypatch, tmp_path):
    from mtplx import artifacts
    from mtplx.artifacts import MTPInspection

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "mtp_num_hidden_layers": 1,
                    "num_experts": 256,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    (
                        "language_model.model.layers.0.mlp.switch_mlp."
                        "down_proj.weight"
                    ): "model-00001-of-00001.safetensors",
                    (
                        "language_model.model.layers.0.mlp.gate.weight"
                    ): "model-00001-of-00001.safetensors",
                },
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
            expected_tensor_count=15,
        ),
    )

    result = inspect_model(tmp_path)

    assert result.passes_primary_gate is True
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
    assert "gemma4-assistant-mtp" in ids


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


def _write_gemma4_pair_bundle(path):
    target = path / "target"
    assistant = path / "assistant"
    target.mkdir()
    assistant.mkdir()
    (path / "mtplx_pair.json").write_text(
        json.dumps(
            {
                "variant": "optimized-speed",
                "layout": {"target": "target", "assistant": "assistant"},
                "target": {"repo": "google/gemma-4-31B-it", "quantization": "q4/g64"},
                "assistant": {
                    "repo": "google/gemma-4-31B-it-assistant",
                    "quantization": "q6/g64",
                },
                "benchmark": {
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 64,
                    "best_block_size": 6,
                },
            }
        ),
        encoding="utf-8",
    )
    (target / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4",
                "architectures": ["Gemma4ForConditionalGeneration"],
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 5376,
                    "num_hidden_layers": 60,
                    "hidden_size_per_layer_input": 0,
                    "enable_moe_block": False,
                    "vocab_size": 262144,
                },
                "quantization_config": {"bits": 4, "group_size": 64},
            }
        ),
        encoding="utf-8",
    )
    (target / "generation_config.json").write_text(
        json.dumps({"temperature": 1.0, "top_p": 0.95, "top_k": 64}),
        encoding="utf-8",
    )
    (assistant / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4_assistant",
                "architectures": ["Gemma4AssistantForCausalLM"],
                "backbone_hidden_size": 5376,
                "use_ordered_embeddings": False,
                "text_config": {
                    "hidden_size": 1024,
                    "num_hidden_layers": 4,
                    "num_kv_shared_layers": 4,
                    "layer_types": [
                        "sliding_attention",
                        "sliding_attention",
                        "sliding_attention",
                        "full_attention",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_gemma4_pair_bundle_inspects_as_assistant_runtime(tmp_path):
    bundle = _write_gemma4_pair_bundle(tmp_path)

    result = inspect_model(bundle)

    assert result.model_type == "gemma4_pair"
    assert result.architecture == "Gemma4AssistantPair"
    assert result.compatibility["can_run"] is True
    assert result.compatibility["recommended_backend"] == "gemma4_assistant"
    assert result.compatibility["runtime_compatibility"] == "assistant-pair-native"
    assert result.recommended_sampler == {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
    assert result.gemma4_pair["target_model"].endswith("/target")
    assert result.gemma4_pair["assistant_model"].endswith("/assistant")


def test_dflash_pair_bundle_inspects_as_native_runtime(tmp_path):
    target = tmp_path / "target"
    drafter = tmp_path / "drafter"
    target.mkdir()
    drafter.mkdir()
    (tmp_path / "dflash_pair.json").write_text(
        json.dumps(
            {
                "backend": "dflash",
                "layout": {"target": "target", "drafter": "drafter"},
                "benchmark": {"best_block_size": 8},
            }
        ),
        encoding="utf-8",
    )
    (target / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["NemotronHForCausalLM"],
                "model_type": "nemotron_h",
                "hidden_size": 2688,
                "num_hidden_layers": 52,
                "vocab_size": 131072,
                "n_routed_experts": 128,
                "num_experts_per_tok": 6,
                "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
            }
        ),
        encoding="utf-8",
    )
    (drafter / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DFlashDraftModel"],
                "model_type": "qwen3",
                "target_layer_ids": [1, 5, 19, 29, 41, 51],
            }
        ),
        encoding="utf-8",
    )

    result = inspect_model(tmp_path)

    assert result.model_type == "dflash_pair"
    assert result.architecture == "DFlashDrafterPair"
    assert result.compatibility["can_run"] is True
    assert result.compatibility["recommended_backend"] == "dflash"
    assert result.compatibility["runtime_compatibility"] == "drafter-pair-native"
    assert result.dflash_pair["target_model"].endswith("/target")
    assert result.dflash_pair["drafter_model"].endswith("/drafter")


def test_gemma4_pair_subfolder_reports_bundle_required(tmp_path):
    bundle = _write_gemma4_pair_bundle(tmp_path)

    result = inspect_model(bundle / "target")

    assert result.compatibility["arch_id"] == "gemma4-assistant-mtp"
    assert result.compatibility["can_run"] is False
    assert result.compatibility["runtime_compatibility"] == "incomplete-assistant-pair"
    assert "bundle root" in result.compatibility["message"]


@pytest.mark.parametrize(
    ("architecture", "model_type", "expected_arch_id"),
    [
        ("DeepseekV32ForCausalLM", "deepseek_v32", "deepseek-v3-mtp"),
        ("GlmMoeDsaForCausalLM", "glm_moe_dsa", "glm-moe-dsa-mtp"),
        # deepseek_v4 now has a native MLX AR backend (arch_id "deepseek-v4"),
        # so it is no longer a backend-pending arch — covered by
        # test_deepseek_v4_routes_to_supported_ar_backend below.
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
            "step3p5-mtp",
        }
        else "recognized-backend-pending"
    )
    assert result.compatibility["runtime_compatibility"] == expected_runtime
    assert result.compatibility["mtp_supported"] == "recognized"


def test_deepseek_v4_routes_to_supported_ar_backend(tmp_path):
    # The mlx-community DeepSeek-V4-Flash conversion drops the MTP block, so the
    # artifact is a target-only AR model handled by the native deepseek_v4 MLX
    # loader (arch_id "deepseek-v4").
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV4ForCausalLM"],
                "model_type": "deepseek_v4",
                "num_hidden_layers": 43,
                "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
            }
        ),
        encoding="utf-8",
    )

    result = inspect_model(tmp_path)

    assert result.compatibility["arch_id"] == "deepseek-v4"
    assert result.compatibility["can_run"] is True
    assert result.compatibility["tier"] == "AR-only"
    assert result.compatibility["recommended_backend"] == "deepseek_v4"
    assert result.compatibility["mtp_supported"] == "no"


def test_deepseek_v4_mtp_split_stays_backend_pending(tmp_path):
    # An MTP-split V4 checkpoint (vLLM layout) still has no runnable MTP backend;
    # it must keep detecting as the pending deepseek-v4-mtp arch, not get captured
    # by the AR "deepseek-v4" entry (the substring-alias hazard).
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV4MTPForCausalLM"],
                "model_type": "deepseek_v4_mtp",
                "num_nextn_predict_layers": 1,
            }
        ),
        encoding="utf-8",
    )

    result = inspect_model(tmp_path)

    assert result.compatibility["arch_id"] == "deepseek-v4-mtp"
    assert result.compatibility["can_run"] is False


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


def _laguna_s_2_1_config(**updates):
    layer_types = [
        layer_type
        for _ in range(12)
        for layer_type in (
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
        )
    ]
    from mtplx.models.laguna_config import LAGUNA_S_2_1_QUANTIZATION

    quantization = copy.deepcopy(LAGUNA_S_2_1_QUANTIZATION)
    config = {
        "architectures": ["LagunaForCausalLM"],
        "model_type": "laguna",
        "hidden_size": 3072,
        "num_hidden_layers": 48,
        "intermediate_size": 12288,
        "num_attention_heads": 48,
        "num_attention_heads_per_layer": [
            48 if layer_type == "full_attention" else 72
            for layer_type in layer_types
        ],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 100352,
        "bos_token_id": 2,
        "eos_token_id": [2, 24],
        "pad_token_id": 9,
        "rms_norm_eps": 1e-6,
        "num_experts": 256,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "decoder_sparse_step": 1,
        "norm_topk_prob": True,
        "moe_routed_scaling_factor": 2.5,
        "moe_router_logit_softcapping": 0.0,
        "moe_apply_router_weight_on_input": False,
        "router_aux_loss_coef": 0.0,
        "mlp_only_layers": [0],
        "gating": "per-head",
        "gating_types": ["per_head"] * 48,
        "sliding_window": 512,
        "layer_types": layer_types,
        "mlp_layer_types": ["dense", *("sparse" for _ in range(47))],
        "rope_parameters": {
            "full_attention": {
                "rope_type": "yarn",
                "rope_theta": 500_000.0,
                "factor": 128.0,
                "original_max_position_embeddings": 8192,
                "beta_slow": 1.0,
                "beta_fast": 32.0,
                "attention_factor": 1.4852030263919618,
                "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "partial_rotary_factor": 1.0,
            },
        },
        "max_position_embeddings": 1_048_576,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "quantization": copy.deepcopy(quantization),
        "quantization_config": copy.deepcopy(quantization),
    }
    config.update(updates)
    return config


def _pipenetwork_laguna_config(**updates):
    """The superseded uniform-4bit build; kept only as rejection coverage."""

    quantization = {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
        **{
            f"model.layers.{layer}.mlp.gate": {"bits": 8, "group_size": 64}
            for layer in range(1, 48)
        },
    }
    return _laguna_s_2_1_config(
        quantization=copy.deepcopy(quantization),
        quantization_config=copy.deepcopy(quantization),
        **updates,
    )


def _write_laguna_s_2_1_artifacts(path, monkeypatch):
    from mtplx.models import laguna_config
    from mtplx.models.laguna_config import (
        LAGUNA_S_2_1_REPO_ID,
        LAGUNA_S_2_1_REVISION,
        LAGUNA_S_2_1_SHARD_SIZES,
    )

    (path / ".mtplx-source.json").write_text(
        json.dumps(
            {
                "repo_id": LAGUNA_S_2_1_REPO_ID,
                "revision": LAGUNA_S_2_1_REVISION,
            }
        ),
        encoding="utf-8",
    )

    shards = [
        f"model-{index:05d}-of-00013.safetensors"
        for index in range(1, 14)
    ]
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    f"model.layer.{index}": shard
                    for index, shard in enumerate(shards)
                }
            }
        ),
        encoding="utf-8",
    )
    for shard in shards:
        with (path / shard).open("wb") as handle:
            handle.truncate(LAGUNA_S_2_1_SHARD_SIZES[shard])
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "generation_config.json").write_text("{}", encoding="utf-8")
    (path / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    (path / "chat_template.jinja").write_text(
        "{{ messages }}",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        laguna_config,
        "LAGUNA_S_2_1_SIDECAR_SHA256",
        {
            name: hashlib.sha256((path / name).read_bytes()).hexdigest()
            for name in laguna_config.LAGUNA_S_2_1_SIDECAR_SHA256
        },
    )


def test_laguna_config_only_directory_is_not_runnable(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(_laguna_s_2_1_config()),
        encoding="utf-8",
    )

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "no-MTP"
    assert result.compatibility["can_run"] is False


def test_complete_laguna_s_2_1_mlx_4bit_is_runnable_ar_only(
    tmp_path, monkeypatch
):
    (tmp_path / "config.json").write_text(
        json.dumps(_laguna_s_2_1_config()),
        encoding="utf-8",
    )
    _write_laguna_s_2_1_artifacts(tmp_path, monkeypatch)

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "AR-only"
    assert result.compatibility["arch_id"] == "laguna-s-2.1-ar"
    assert result.compatibility["recognized"] is True
    assert result.compatibility["supported"] is True
    assert result.compatibility["can_run"] is True
    assert result.compatibility["mtp_supported"] == "no"
    assert result.compatibility["runtime_compatibility"] == "native-ar-only"


def test_laguna_shard_with_wrong_size_is_not_runnable(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps(_laguna_s_2_1_config()),
        encoding="utf-8",
    )
    _write_laguna_s_2_1_artifacts(tmp_path, monkeypatch)
    (tmp_path / "model-00013-of-00013.safetensors").write_bytes(b"tampered")

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "no-MTP"
    assert result.compatibility["can_run"] is False


def test_laguna_without_poolside_chat_template_is_not_runnable(
    tmp_path, monkeypatch
):
    (tmp_path / "config.json").write_text(
        json.dumps(_laguna_s_2_1_config()),
        encoding="utf-8",
    )
    _write_laguna_s_2_1_artifacts(tmp_path, monkeypatch)
    (tmp_path / "chat_template.jinja").unlink()

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "no-MTP"
    assert result.compatibility["can_run"] is False


@pytest.mark.parametrize(
    ("filename", "contents"),
    (
        ("tokenizer.json", "not-json"),
        ("chat_template.jinja", "{% if broken %}"),
    ),
)
def test_laguna_with_mutated_pinned_sidecar_is_not_runnable(
    tmp_path, monkeypatch, filename, contents
):
    (tmp_path / "config.json").write_text(
        json.dumps(_laguna_s_2_1_config()),
        encoding="utf-8",
    )
    _write_laguna_s_2_1_artifacts(tmp_path, monkeypatch)
    (tmp_path / filename).write_text(contents, encoding="utf-8")

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "no-MTP"
    assert result.compatibility["can_run"] is False


def test_non_4bit_laguna_is_not_admitted_by_target_only_gate(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            _laguna_s_2_1_config(
                quantization={"bits": 8, "group_size": 64, "mode": "affine"},
                quantization_config={"bits": 8, "group_size": 64, "mode": "affine"},
            )
        ),
        encoding="utf-8",
    )

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "no-MTP"
    assert result.compatibility["can_run"] is False


def test_superseded_pipenetwork_laguna_is_not_admitted(tmp_path, monkeypatch):
    # The pin moved to mlx-community/Laguna-S-2.1-oQ4e; the older uniform-4bit
    # pipenetwork build shares the geometry but not the quantization map, so it
    # is now blocked like any other non-pinned variant even with a fully written
    # (but wrong-hash) artifact set.
    (tmp_path / "config.json").write_text(
        json.dumps(_pipenetwork_laguna_config()),
        encoding="utf-8",
    )
    _write_laguna_s_2_1_artifacts(tmp_path, monkeypatch)

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "no-MTP"
    assert result.compatibility["can_run"] is False


def test_laguna_config_with_remote_model_file_is_blocked(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(_laguna_s_2_1_config(model_file="evil.py")),
        encoding="utf-8",
    )

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "no-MTP"
    assert result.compatibility["can_run"] is False


def test_wrong_laguna_expert_geometry_is_not_admitted_by_target_only_gate(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(_laguna_s_2_1_config(num_experts=128)),
        encoding="utf-8",
    )

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "no-MTP"
    assert result.compatibility["can_run"] is False


def test_wrong_laguna_attention_geometry_is_not_admitted_by_target_only_gate(
    tmp_path,
):
    (tmp_path / "config.json").write_text(
        json.dumps(
            _laguna_s_2_1_config(num_attention_heads_per_layer=[48] * 48)
        ),
        encoding="utf-8",
    )

    result = inspect_model(tmp_path)

    assert result.compatibility["tier"] == "no-MTP"
    assert result.compatibility["can_run"] is False


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


def test_gemma4_pair_bundle_root_is_runnable_from_remote_file_listing():
    """Issue #16: the HF preflight refused the official Gemma 4 repos.

    A complete assistant-pair bundle has no MTP tensors of its own and
    no root config.json; the bundle root is recognizable by weights
    under both target/ and assistant/. The compatibility gate must
    treat that as the runnable artifact, exactly like the app does.
    """
    from mtplx.artifacts import ModelInspection
    from mtplx.backends.registry import compatibility_for_inspection

    inspection = ModelInspection(
        model_dir="hf://Youssofal/Gemma4-MTPLX-Optimized-Speed",
        config_exists=True,
        architecture="Gemma4ForConditionalGeneration",
        model_type="gemma4",
        mtp_num_hidden_layers=0,
        hidden_size=None,
        num_hidden_layers=None,
        vocab_size=None,
        source="huggingface",
        model_files=(
            "assistant/model.safetensors",
            "target/model-00001-of-00004.safetensors",
        ),
    )

    verdict = compatibility_for_inspection(inspection)

    assert verdict.can_run is True
    assert verdict.exit_code == 0
    assert verdict.arch_id == "gemma4-assistant-mtp"
    assert verdict.runtime_compatibility == "assistant-pair-native"


def test_gemma4_target_subfolder_alone_still_refuses():
    from mtplx.artifacts import ModelInspection
    from mtplx.backends.registry import compatibility_for_inspection

    inspection = ModelInspection(
        model_dir="hf://someone/gemma4-target-only",
        config_exists=True,
        architecture="Gemma4ForConditionalGeneration",
        model_type="gemma4",
        mtp_num_hidden_layers=0,
        hidden_size=None,
        num_hidden_layers=None,
        vocab_size=None,
        source="huggingface",
        model_files=("model-00001-of-00004.safetensors",),
    )

    verdict = compatibility_for_inspection(inspection)

    assert verdict.can_run is False
    assert verdict.runtime_compatibility == "incomplete-assistant-pair"


def test_user_promoted_contract_runs_as_unverified_label(monkeypatch, tmp_path):
    # The published Optimized Quality shape (#98): a human promoted the
    # artifact at publish time. Verification is a label, never a load
    # gate, so the model must stay runnable.
    import json as _json

    from mtplx import artifacts
    from mtplx.artifacts import MTPInspection

    (tmp_path / "config.json").write_text(
        _json.dumps(
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
    _write_runtime_contract(
        tmp_path,
        exactness_baseline={"status": "candidate-promoted-by-user-decision"},
    )

    result = inspect_model(tmp_path)

    compatibility = result.compatibility
    assert compatibility["tier"] == "architecture-compatible-but-unverified"
    assert compatibility["can_run"] is True
    assert compatibility["unsafe_force_required"] is False
    assert compatibility["runtime_compatibility"] == "runtime-contract-unverified"
    assert "runs as unverified" in compatibility["message"]


def test_pull_refreshes_when_remote_index_changed(monkeypatch, tmp_path):
    # An explicit pull must sync repo deltas (restored vision towers)
    # even when the local copy passes the completeness check.
    import json as _json

    from mtplx import hf_loader

    local = tmp_path / "model"
    local.mkdir()
    (local / "model.safetensors.index.json").write_text(
        _json.dumps({"weight_map": {"a": "model.safetensors"}}), encoding="utf-8"
    )

    remote_same = tmp_path / "remote_same.json"
    remote_same.write_bytes((local / "model.safetensors.index.json").read_bytes())
    remote_changed = tmp_path / "remote_changed.json"
    remote_changed.write_text(
        _json.dumps({"weight_map": {"a": "model.safetensors", "vision_tower.x": "model-vision.safetensors"}}),
        encoding="utf-8",
    )

    def fake_download(repo_id, filename, revision=None):
        return str(fake_download.target)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    fake_download.target = remote_same
    assert hf_loader._local_matches_remote_index(local, "org/repo", None) is True

    fake_download.target = remote_changed
    assert hf_loader._local_matches_remote_index(local, "org/repo", None) is False

    def broken_download(repo_id, filename, revision=None):
        raise RuntimeError("offline")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", broken_download)
    assert hf_loader._local_matches_remote_index(local, "org/repo", None) is True


def test_served_public_ids_resolve_to_first_party_repos():
    """The exact ids /v1/models advertises must resolve for serve/run/pull.

    Explicit first-party ids only (contract-match-only stance, #57):
    anything that is not a known public id or two-part repo stays None.
    """
    from mtplx.artifacts import _hf_repo_id_from_ref
    from mtplx.profiles import (
        OPTIMIZED_SPEED_V1_HF_MODEL_ID,
        OPTIMIZED_SPEED_V2_HF_MODEL_ID,
        QUALITY_HF_MODEL_ID,
        QWEN35_9B_OPTIMIZED_SPEED_HF_MODEL_ID,
        QWEN36_35B_OPTIMIZED_SPEED_HF_MODEL_ID,
    )

    assert (
        _hf_repo_id_from_ref("mtplx-qwen36-27b-optimized-speed-v2")
        == OPTIMIZED_SPEED_V2_HF_MODEL_ID
    )
    assert (
        _hf_repo_id_from_ref("mtplx-qwen36-27b-optimized-speed")
        == OPTIMIZED_SPEED_V1_HF_MODEL_ID
    )
    assert (
        _hf_repo_id_from_ref("MTPLX-Qwen36-27B-Optimized-Speed")
        == OPTIMIZED_SPEED_V1_HF_MODEL_ID
    )
    assert _hf_repo_id_from_ref("mtplx-qwen36-27b-optimized-quality") == QUALITY_HF_MODEL_ID
    assert (
        _hf_repo_id_from_ref("mtplx-qwen35-9b-optimized-speed")
        == QWEN35_9B_OPTIMIZED_SPEED_HF_MODEL_ID
    )
    assert (
        _hf_repo_id_from_ref("mtplx-qwen36-35b-a3b-optimized-speed")
        == QWEN36_35B_OPTIMIZED_SPEED_HF_MODEL_ID
    )
    assert _hf_repo_id_from_ref("mtplx-qwopus-madeup-id") is None
    assert _hf_repo_id_from_ref("some-random-model-name") is None
