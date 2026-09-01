from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.backends.deepseek_v4_mlxserve import (
    BACKEND_ID,
    DeepSeekV4MlxServeError,
    build_command,
    child_environment,
    resolve_binary,
)
from mtplx.backends.descriptors import descriptor_for_backend_id
from mtplx.backends.registry import compatibility_for_inspection
from mtplx.cli import build_parser
from mtplx.commands import public
from mtplx import artifacts, hf_loader
from mtplx.models import deepseek_v4_target_only_config as contract
from mtplx.reasoning_effort import REASONING_EFFORT_CHOICES


def _quantization() -> dict[str, object]:
    quantization: dict[str, object] = {
        "bits": 8,
        "group_size": 64,
        "mode": "affine",
        "embed": {"bits": 8, "group_size": 64, "mode": "affine"},
        "head": {"bits": 8, "group_size": 64, "mode": "affine"},
    }
    for layer in range(43):
        bits = (2, 3, 2) if layer < 39 else (4, 4, 4)
        group_size = 128 if layer < 39 else 64
        for projection, projection_bits in zip(("w1", "w2", "w3"), bits, strict=True):
            quantization[f"layers.{layer}.ffn.experts.{projection}"] = {
                "bits": projection_bits,
                "group_size": group_size,
                "mode": "affine",
            }
    return quantization


def _config() -> dict[str, object]:
    return {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43,
        "hidden_size": 4096,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "vocab_size": 129_280,
        "num_nextn_predict_layers": 0,
        "dspark_block_size": 0,
        "num_experts_per_tok": 6,
        "n_routed_experts": 256,
        "quantization": _quantization(),
    }


def _inspection(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "model_dir": "/tmp/deepseek-v4-target-only",
        "architecture": "DeepseekV4ForCausalLM",
        "model_type": "deepseek_v4",
        "mtp_num_hidden_layers": 0,
        "deepseek_v4_target_only_match": True,
        "deepseek_v4_target_only_artifacts_complete": True,
        "mtp": None,
        "runtime_contract_data": None,
        "runtime_contract_error": None,
        "runtime_contract_path": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_public_identity_is_immutable_and_complete() -> None:
    assert contract.DEEPSEEK_V4_TARGET_ONLY_REVISION == "ac33e4f3ca3546e6cec104558d42161e15814e33"
    assert len(contract.DEEPSEEK_V4_TARGET_ONLY_WEIGHT_SHARDS) == 44
    assert set(contract.DEEPSEEK_V4_TARGET_ONLY_WEIGHT_SHARDS) == set(
        contract.DEEPSEEK_V4_TARGET_ONLY_SHARD_SIZES
    )
    assert sum(contract.DEEPSEEK_V4_TARGET_ONLY_SHARD_SIZES.values()) == 103_849_215_724
    assert contract.is_deepseek_v4_target_only_config(_config()) is True


def test_pull_identity_rejects_a_noncanonical_deepseek_revision() -> None:
    assert hf_loader._effective_model_revision(
        contract.DEEPSEEK_V4_TARGET_ONLY_REPO_ID, None
    ) == contract.DEEPSEEK_V4_TARGET_ONLY_REVISION
    with pytest.raises(ValueError, match="pinned to revision"):
        hf_loader._effective_model_revision(
            contract.DEEPSEEK_V4_TARGET_ONLY_REPO_ID, "not-the-public-revision"
        )


def test_target_only_config_rejects_expert_recipe_drift() -> None:
    config = _config()
    quantization = config["quantization"]
    assert isinstance(quantization, dict)
    quantization["layers.8.ffn.experts.w2"] = {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
    }
    assert contract.is_deepseek_v4_target_only_config(config) is False


def test_integrity_rejects_same_size_shard_content_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shard = tmp_path / "model-layer-0.safetensors"
    shard.write_bytes(b"good")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": shard.name}}), encoding="utf-8")
    monkeypatch.setattr(contract, "DEEPSEEK_V4_TARGET_ONLY_WEIGHT_SHARDS", (shard.name,))
    monkeypatch.setattr(contract, "DEEPSEEK_V4_TARGET_ONLY_SHARD_SIZES", {shard.name: 4})
    monkeypatch.setattr(
        contract,
        "DEEPSEEK_V4_TARGET_ONLY_SHARD_SHA256",
        {shard.name: hashlib.sha256(b"good").hexdigest()},
    )
    monkeypatch.setattr(
        contract,
        "DEEPSEEK_V4_TARGET_ONLY_SIDECAR_SHA256",
        {index.name: hashlib.sha256(index.read_bytes()).hexdigest()},
    )
    assert contract.deepseek_v4_target_only_artifact_integrity_errors(tmp_path) == ()
    shard.write_bytes(b"evil")
    assert contract.deepseek_v4_target_only_artifact_integrity_errors(tmp_path) == (shard.name,)


def test_exact_admission_is_external_ar_not_mtp_or_dspark() -> None:
    verdict = compatibility_for_inspection(_inspection())
    assert verdict.tier == "AR-only"
    assert verdict.arch_id == "deepseek-v4-mlxserve-ar"
    assert verdict.recommended_backend == BACKEND_ID
    assert verdict.mtp_supported == "no"
    assert verdict.runtime_compatibility == "external-mem-preflight-required"
    descriptor = descriptor_for_backend_id(BACKEND_ID)
    assert descriptor.uses_draft_lm_head is False
    assert set(descriptor.reasoning_codec.effort_levels) <= set(
        REASONING_EFFORT_CHOICES
    )


def test_external_launch_is_closed_and_keeps_memory_preflight(tmp_path: Path) -> None:
    binary = tmp_path / "mlx-serve"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    admitted = resolve_binary({"MTPLX_MLX_SERVE_BIN": str(binary)})
    command = build_command(
        binary=admitted,
        model="/models/dsv4",
        host="127.0.0.1",
        port=8123,
        context_window=None,
        api_key=None,
    )
    environment = child_environment(
        {"PATH": "/usr/bin", "MLX_SERVE_WIRED": "off", "MLXSERVE_DEVICE": "foreign"}
    )
    assert environment == {
        "PATH": "/usr/bin",
        "MLX_SERVE_WIRED": "fit",
        "MLX_SERVE_CACHE_LIMIT": "268435456",
    }
    assert "--skip-mem-preflight" not in command
    assert {"--no-pld", "--no-decode-attn-quant", "--no-vision"}.issubset(command)


def test_external_binary_resolution_uses_the_admitted_path(tmp_path: Path) -> None:
    binary = tmp_path / "mlx-serve"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)

    assert resolve_binary({"PATH": str(tmp_path)}) == binary.resolve()


def _patch_external_route(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    binary = tmp_path / "mlx-serve"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    inspection = {
        "model_dir": str(tmp_path),
        "recommended_backend": BACKEND_ID,
        "compatibility": {
            "tier": "AR-only",
            "can_run": True,
            "exit_code": 0,
            "runtime_compatibility": "external-mem-preflight-required",
            "recommended_backend": BACKEND_ID,
        },
    }
    monkeypatch.setattr(public, "_serve_should_onboard", lambda _args: False)
    monkeypatch.setattr(public, "_resolve_runtime_model_path", lambda *_args, **_kwargs: (str(tmp_path), None))
    monkeypatch.setattr(public, "_model_gate", lambda *_args, **_kwargs: (inspection, None))
    monkeypatch.setattr(public, "resolve_deepseek_v4_mlxserve_binary", lambda: binary.resolve())
    monkeypatch.setattr(public, "resolve_deepseek_v4_mlxserve_working_directory", lambda _binary: tmp_path)
    monkeypatch.setattr(os, "environ", {"PATH": "/usr/bin"})
    return binary


def test_cli_dry_run_reports_external_admission_without_loading_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_external_route(monkeypatch, tmp_path)
    args = build_parser().parse_args(["serve", "--model", str(tmp_path), "--yes"])
    args.dry_run = True
    args.json = True
    assert public.cmd_serve_public(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend_id"] == BACKEND_ID
    assert payload["external_runtime"] == "mlx-serve"
    assert payload["generation_mode"] == "ar"
    assert payload["mtp_available"] is False
    assert payload["dspark_available"] is False
    assert payload["memory_preflight"] == "required"
    assert "no MTP or DSpark" in payload["runtime_compatibility_note"]
    assert "--skip-mem-preflight" not in payload["argv"]


def test_pull_marker_shape_is_admitted_by_inspect_and_serve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_config()), encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {}}),
        encoding="utf-8",
    )
    hf_loader._write_source_marker(
        tmp_path,
        repo_id=contract.DEEPSEEK_V4_TARGET_ONLY_REPO_ID,
        revision=contract.DEEPSEEK_V4_TARGET_ONLY_REVISION,
        resolved_sha="a" * 40,
        files={"config.json": {"size": 1, "blob_id": "b" * 40}},
    )
    marker = json.loads((tmp_path / hf_loader.SOURCE_MARKER_FILE).read_text())
    assert {
        "resolved_sha",
        "pulled_at",
        "engine_version",
        "files",
    }.issubset(marker)

    monkeypatch.setattr(
        artifacts,
        "deepseek_v4_target_only_artifact_integrity_errors",
        lambda _path: (),
    )
    inspection = artifacts.inspect_model(tmp_path).to_dict()
    assert inspection["deepseek_v4_target_only_artifacts_complete"] is True
    assert inspection["compatibility"]["can_run"] is True
    assert inspection["compatibility"]["recommended_backend"] == BACKEND_ID

    binary = tmp_path / "mlx-serve"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setattr(public, "_serve_should_onboard", lambda _args: False)
    monkeypatch.setattr(
        public, "resolve_deepseek_v4_mlxserve_binary", lambda: binary.resolve()
    )
    monkeypatch.setattr(
        public,
        "resolve_deepseek_v4_mlxserve_working_directory",
        lambda _binary: tmp_path,
    )
    monkeypatch.setattr(os, "environ", {"PATH": "/usr/bin"})

    args = build_parser().parse_args(["serve", "--model", str(tmp_path), "--yes"])
    args.dry_run = True
    args.json = True
    assert public.cmd_serve_public(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend_id"] == BACKEND_ID
    assert payload["external_runtime"] == "mlx-serve"


def test_cli_dry_run_reports_missing_external_binary_as_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_external_route(monkeypatch, tmp_path)
    monkeypatch.setattr(
        public,
        "resolve_deepseek_v4_mlxserve_binary",
        lambda: (_ for _ in ()).throw(DeepSeekV4MlxServeError("mlx-serve is required")),
    )
    args = build_parser().parse_args(["serve", "--model", str(tmp_path), "--yes"])
    args.dry_run = True
    args.json = True
    assert public.cmd_serve_public(args) == 2
    assert json.loads(capsys.readouterr().out) == {
        "ok": False,
        "dry_run": True,
        "target": "server",
        "error": "external_runtime_admission_failed",
        "backend_id": BACKEND_ID,
        "external_runtime": "mlx-serve",
        "detail": "mlx-serve is required",
    }
