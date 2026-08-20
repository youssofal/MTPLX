from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.cli import build_parser, main
from mtplx.commands import forge
from mtplx.constants import EXPECTED_MTP_KEYS


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_raw_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header: dict[str, dict[str, object]] = {}
    chunks: list[bytes] = []
    offset = 0
    for key, (dtype, shape, payload) in tensors.items():
        end = offset + len(payload)
        header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, end]}
        chunks.append(payload)
        offset = end
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(len(encoded).to_bytes(8, "little"))
        handle.write(encoded)
        for chunk in chunks:
            handle.write(chunk)


def _mtp_config() -> dict:
    return {
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "mtp_num_hidden_layers": 1,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "vocab_size": 128,
        },
        "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
    }


def _speed_win_rows() -> list[dict]:
    return [
        {
            "depth": 0,
            "tok_s": 22.0,
            "multiplier_vs_ar": 1.0,
            "acceptance_by_position": [],
            "verify_time_s": 0.1,
        },
        {
            "depth": 1,
            "tok_s": 34.0,
            "multiplier_vs_ar": 1.5454545455,
            "acceptance_by_position": [0.9],
            "verify_time_s": 0.2,
        },
        {
            "depth": 2,
            "tok_s": 40.0,
            "multiplier_vs_ar": 1.8181818182,
            "acceptance_by_position": [0.88, 0.55],
            "verify_time_s": 0.3,
        },
        {
            "depth": 3,
            "tok_s": 44.0,
            "multiplier_vs_ar": 2.0,
            "acceptance_by_position": [0.9, 0.7, 0.5],
            "verify_time_s": 0.4,
        },
    ]


def _runtime(depth: int = 3) -> dict:
    rows = _speed_win_rows()
    selected = next((row for row in rows if row["depth"] == depth), rows[-1])
    return {
        "mtplx_version": "0.1.test",
        "arch_id": "qwen3-next-mtp",
        "mtp_depth_max": depth,
        "recommended_profile": "sustained",
        "sampler": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        "verified_on": {
            "timestamp": "2026-05-26T00:00:00+01:00",
            "hardware": "test",
            "machine_arch": "arm64",
            "macos": "26.0",
            "model": "Fixture-MTPLX-Speed",
        },
        "exactness_baseline": {},
        "speed_evidence": {
            "depth": depth,
            "tok_s": [selected["tok_s"]],
            "acceptance_by_depth": selected["acceptance_by_position"],
            "greedy_diagnostic": {"tok_s": 22.0},
            "forge_verify_rows": rows,
            "verdict": "mtp_depth_wins",
            "failure_reasons": [],
        },
        "mtp_contract": {
            "base_hidden_variant": "post_norm",
            "hidden_variant": "post_norm",
            "concat_order": "embedding_hidden",
            "mtp_quant_group_size": 64,
            "mtp_quant_mode": "affine",
        },
        "mtp_sidecar": "bf16",
        "base_trunk": "fixture/source",
        "artifact_role": "fixture",
    }


def _write_qwen_sidecar_fixture(path: Path) -> None:
    _write_json(path / "config.json", _mtp_config())
    mx.save_safetensors(
        str(path / "mtp.safetensors"),
        {key: mx.zeros((1, 1), dtype=mx.float16) for key in EXPECTED_MTP_KEYS},
    )


def _write_appended_mtp_fixture(
    path: Path,
    *,
    architecture: str,
    model_type: str,
    num_hidden_layers: int = 47,
) -> None:
    _write_json(
        path / "config.json",
        {
            "architectures": [architecture],
            "model_type": model_type,
            "num_hidden_layers": num_hidden_layers,
            "num_nextn_predict_layers": 1,
        },
    )
    _write_json(
        path / "model.safetensors.index.json",
        {"weight_map": {f"model.layers.{num_hidden_layers}.enorm.weight": "model.safetensors"}},
    )


def _write_legacy_speed_grid(path: Path) -> None:
    rows = _speed_win_rows()
    _write_json(
        path,
        {
            "ar_rows": [
                {"tok_s": rows[0]["tok_s"], "elapsed_s": rows[0]["verify_time_s"]}
            ],
            "depths": [
                {
                    "depth": row["depth"],
                    "rows": [
                        {
                            "tok_s": row["tok_s"],
                            "elapsed_s": row["verify_time_s"],
                            "acceptance_by_depth": row["acceptance_by_position"],
                        }
                    ],
                }
                for row in rows
                if row["depth"] > 0
            ],
        },
    )


def _write_gemma4_pair_bundle(path: Path) -> Path:
    _write_json(
        path / "mtplx_pair.json",
        {
            "variant": "optimized-speed",
            "layout": {"target": "target", "assistant": "assistant"},
            "target": {"quantization": "mlx-6bit"},
            "assistant": {"quantization": "mlx-4bit"},
        },
    )
    _write_json(
        path / "target" / "config.json",
        {
            "architectures": ["Gemma4ForCausalLM"],
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 1024,
                "num_hidden_layers": 4,
                "vocab_size": 262144,
            },
        },
    )
    _write_json(
        path / "assistant" / "config.json",
        {
            "architectures": ["Gemma4AssistantForCausalLM"],
            "model_type": "gemma4_assistant",
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 1024,
                "num_hidden_layers": 4,
                "vocab_size": 262144,
            },
        },
    )
    return path


def test_forge_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["forge", "--help"])

    assert exc.value.code == 0


def test_forge_parser_recognizes_app_subcommands():
    parser = build_parser()

    cases = [
        ["forge", "probe", "owner/model", "--json"],
        [
            "forge",
            "build",
            "--repo",
            "owner/model",
            "--out",
            "/tmp/out",
            "--run-id",
            "r1",
            "--recipe",
            "{}",
            "--branded-name",
            "Model-MTPLX-Speed",
        ],
        ["forge", "discover", "--json", "--limit", "5", "--offset", "10"],
        [
            "forge",
            "publish",
            "--path",
            "/tmp/model",
            "--repo",
            "owner/model",
            "--visibility",
            "private",
            "--license",
            "apache-2.0",
            "--out",
            "/tmp/out",
            "--run-id",
            "p1",
            "--token",
            "stdin",
        ],
        ["forge", "inspect", "/tmp/model", "--json"],
        ["forge", "verify", "/tmp/model", "--json", "--max-tokens", "1536", "--suite", "long-code-uncapped"],
        ["forge", "cancel", "run-123"],
    ]

    for argv in cases:
        args = parser.parse_args(argv)
        assert args.command == "forge"
        assert args.forge_action == argv[1]
        assert args.func.__name__ == "cmd_forge_public"
        if argv[1] == "verify":
            assert args.max_tokens == 1536
            assert args.suite == "long-code-uncapped"


def test_atomic_phase_writer_never_leaves_malformed_json(tmp_path):
    target = tmp_path / "run" / "download.json"

    for index in range(25):
        forge.atomic_write_json(target, {"bytes_on_disk": index, "finished": index == 24})
        loaded = json.loads(target.read_text(encoding="utf-8"))
        assert loaded["bytes_on_disk"] == index

    assert loaded["finished"] is True


def test_tune_failure_detail_prefers_candidate_errors(tmp_path):
    run = tmp_path / "run"
    output_root = run / "tune" / "forge-verify"
    output_root.mkdir(parents=True)
    ar_log = output_root / "ar.log"
    ar_log.write_text(
        "thermal prelude\nTraceback (most recent call last):\nValueError: bad mtp policy\n",
        encoding="utf-8",
    )
    stdout = run / "verify.stdout.log"
    stderr = run / "verify.stderr.log"
    stderr.write_text("[max] fans ramped\n", encoding="utf-8")
    stdout.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "candidate": "ar",
                        "error": "candidate did not write an artifact",
                        "stdout": str(ar_log),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    detail = forge._tune_failure_detail(
        stdout_path=stdout,
        stderr_path=stderr,
        output_root=output_root,
    )

    assert "ValueError: bad mtp policy" in detail
    assert "[max] fans ramped" not in detail


def test_probe_classifies_already_mtplx_local_artifact(tmp_path):
    _write_json(tmp_path / "config.json", _mtp_config())
    _write_json(tmp_path / "mtplx_runtime.json", _runtime())

    payload = forge.probe_source(str(tmp_path))

    assert payload["verdict"] == "already_mtplx"
    assert payload["forgeable"] is True
    assert payload["source_format"] == "mlx_affine_with_mtp"
    assert payload["has_mtp_weights"] is True


def test_probe_classifies_gemma4_pair_bundle_as_runnable_not_convertible(tmp_path):
    bundle = _write_gemma4_pair_bundle(tmp_path)

    payload = forge.probe_source(str(bundle))

    assert payload["verdict"] == "already_mtplx"
    assert payload["already_mtplx"] is True
    assert payload["forgeable"] is False
    assert payload["supported"] is True
    assert payload["source_format"] == "gemma4_assistant_pair"
    assert payload["has_mtp_weights"] is True
    assert "runnable MTPLX Gemma 4 assistant-pair bundle" in payload["message"]
    assert "conversion is not needed" in payload["message"]


def test_probe_detects_gguf_as_unsupported_v1(tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"GGUF")

    payload = forge.probe_source(str(tmp_path))

    assert payload["verdict"] == "unsupported_source"
    assert payload["forgeable"] is False
    assert payload["source_format"] == "unknown"
    assert payload["diagnostic"] == "gguf_unsupported_v1"


def test_probe_cli_exits_zero_for_clear_unsupported_verdict(tmp_path, capsys):
    (tmp_path / "model.gguf").write_bytes(b"GGUF")

    code = main(["forge", "probe", str(tmp_path), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["verdict"] == "unsupported_source"


def test_probe_refuses_no_mtp_sources(tmp_path):
    _write_json(
        tmp_path / "config.json",
        {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "text_config": {"model_type": "qwen3_text"},
        },
    )

    payload = forge.probe_source(str(tmp_path))

    assert payload["verdict"] == "no_mtp_heads"
    assert payload["forgeable"] is False
    assert payload["has_mtp_weights"] is False


def test_probe_refuses_config_only_mtp_sources(tmp_path):
    _write_json(
        tmp_path / "config.json",
        {
            "architectures": ["Glm4MoeLiteForCausalLM"],
            "model_type": "glm4_moe_lite",
            "num_hidden_layers": 47,
            "num_nextn_predict_layers": 1,
        },
    )

    payload = forge.probe_source(str(tmp_path))

    assert payload["verdict"] == "no_mtp_heads"
    assert payload["forgeable"] is False
    assert payload["has_mtp_weights"] is False
    assert "config-only" in payload["message"]


def test_probe_refuses_gemma_subfolder_with_bundle_root_hint(tmp_path):
    _write_json(
        tmp_path / "config.json",
        {
            "architectures": ["Gemma4AssistantForCausalLM"],
            "model_type": "gemma4_text",
            "hidden_size": 1024,
            "num_hidden_layers": 4,
            "vocab_size": 262144,
            "quantization_config": {"bits": 6, "group_size": 64},
        },
    )

    payload = forge.probe_source(str(tmp_path))

    assert payload["verdict"] == "no_mtp_heads"
    assert payload["diagnostic"] == "incomplete-assistant-pair"
    assert "bundle root" in payload["message"]
    assert "mtplx_pair.json" in payload["message"]


def test_probe_architecture_matrix_routes_supported_and_pending_mtp(tmp_path):
    qwen = tmp_path / "qwen"
    glm = tmp_path / "glm"
    deepseek = tmp_path / "deepseek"
    step = tmp_path / "step"
    pending = tmp_path / "pending"
    no_mtp = tmp_path / "plain"
    gemma = _write_gemma4_pair_bundle(tmp_path / "gemma_pair")

    _write_qwen_sidecar_fixture(qwen)
    _write_appended_mtp_fixture(
        glm,
        architecture="Glm4MoeLiteForCausalLM",
        model_type="glm4_moe_lite",
    )
    _write_appended_mtp_fixture(
        deepseek,
        architecture="DeepseekV3ForCausalLM",
        model_type="deepseek_v3",
        num_hidden_layers=61,
    )
    _write_appended_mtp_fixture(
        step,
        architecture="Step3P5MTPForCausalLM",
        model_type="step3p5_mtp",
        num_hidden_layers=48,
    )
    _write_appended_mtp_fixture(
        pending,
        architecture="DeepseekV4MTPForCausalLM",
        model_type="deepseek_v4_mtp",
        num_hidden_layers=48,
    )
    _write_json(
        no_mtp / "config.json",
        {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"},
    )

    qwen_payload = forge.probe_source(str(qwen))
    glm_payload = forge.probe_source(str(glm))
    deepseek_payload = forge.probe_source(str(deepseek))
    step_payload = forge.probe_source(str(step))
    gemma_payload = forge.probe_source(str(gemma))
    no_mtp_payload = forge.probe_source(str(no_mtp))
    pending_payload = forge.probe_source(str(pending))

    assert qwen_payload["forgeable"] is True
    assert qwen_payload["architecture_id"] == "qwen3-next-mtp"
    assert glm_payload["forgeable"] is True
    assert glm_payload["architecture_id"] == "glm4-moe-lite-mtp"
    assert deepseek_payload["forgeable"] is True
    assert deepseek_payload["architecture_id"] == "deepseek-v3-mtp"
    assert step_payload["forgeable"] is True
    assert step_payload["architecture_id"] == "step3p5-mtp"
    assert gemma_payload["source_format"] == "gemma4_assistant_pair"
    assert gemma_payload["forgeable"] is False
    assert gemma_payload["supported"] is True
    assert no_mtp_payload["verdict"] == "no_mtp_heads"
    assert no_mtp_payload["forgeable"] is False
    assert pending_payload["verdict"] == "unsupported_source"
    assert pending_payload["forgeable"] is False
    assert pending_payload["architecture_id"] == "deepseek-v4-mtp"
    assert pending_payload["diagnostic"] == "backend_pending_mtp:deepseek-v4-mtp"
    assert pending_payload["message"] == "MTP detected, backend not ready."


def test_saved_verify_rows_are_stale_when_mtp_quant_contract_changed(tmp_path):
    _write_json(
        tmp_path / "config.json",
        {
            "mtplx_mtp_contract": {
                "base_hidden_variant": "post_norm",
                "hidden_variant": "post_norm",
                "concat_order": "embedding_hidden",
                "mtp_quant_group_size": 64,
                "mtp_quant_mode": "affine",
            },
            "mtplx_mtp_quantization": {
                "policy": "cyankiwi",
                "bits": 4,
                "group_size": 32,
                "mode": "affine",
                "prequantized": True,
            },
        },
    )
    runtime = _runtime()

    assert forge._runtime_has_mtp_contract(runtime, tmp_path) is False

    normalized = forge._runtime_or_default_mtp_contract(runtime, tmp_path)
    assert normalized["mtp_quant_bits"] == 4
    assert normalized["mtp_quant_group_size"] == 32
    assert normalized["mtp_quant_policy"] == "cyankiwi"


def test_probe_classifies_compressed_tensors_awq_with_mtp(tmp_path):
    config = {
        "architectures": ["Glm4MoeLiteForCausalLM"],
        "model_type": "glm4_moe_lite",
        "num_hidden_layers": 47,
        "num_nextn_predict_layers": 1,
    }
    config["quantization_config"] = {"quant_method": "compressed-tensors", "format": "awq"}
    _write_json(tmp_path / "config.json", config)
    _write_json(
        tmp_path / "model.safetensors.index.json",
        {"weight_map": {"model.layers.47.enorm.weight": "model.safetensors"}},
    )

    payload = forge.probe_source(str(tmp_path))

    assert payload["verdict"] == "forgeable"
    assert payload["source_format"] == "compressed_tensors_awq"
    assert payload["has_mtp_weights"] is True


def test_probe_classifies_autoawq_with_mtp(tmp_path):
    config = {
        "architectures": ["Glm4MoeLiteForCausalLM"],
        "model_type": "glm4_moe_lite",
        "num_hidden_layers": 47,
        "num_nextn_predict_layers": 1,
        "quantization_config": {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
        },
    }
    _write_json(tmp_path / "config.json", config)
    _write_json(
        tmp_path / "model.safetensors.index.json",
        {"weight_map": {"model.layers.47.enorm.weight": "model.safetensors"}},
    )

    payload = forge.probe_source(str(tmp_path))

    assert payload["verdict"] == "forgeable"
    assert payload["source_format"] == "autoawq"
    assert payload["has_mtp_weights"] is True
    assert "AutoAWQ" in payload["message"]


def test_probe_classifies_compressed_tensors_nvfp4_with_mtp(tmp_path):
    config = {
        "architectures": ["Glm4MoeLiteForCausalLM"],
        "model_type": "glm4_moe_lite",
        "num_hidden_layers": 47,
        "num_nextn_predict_layers": 1,
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                        "strategy": "tensor_group",
                        "group_size": 16,
                    }
                }
            },
        },
    }
    _write_json(tmp_path / "config.json", config)
    _write_json(
        tmp_path / "model.safetensors.index.json",
        {"weight_map": {"model.layers.47.enorm.weight": "model.safetensors"}},
    )

    payload = forge.probe_source(str(tmp_path))

    assert payload["verdict"] == "forgeable"
    assert payload["forgeable"] is True
    assert payload["source_format"] == "compressed_tensors_nvfp4"
    assert payload["has_mtp_weights"] is True
    assert "convert FP4 weights" in payload["message"]


def test_probe_hf_source_uses_current_model_info_signature(monkeypatch):
    config = _mtp_config()
    config.pop("mlx_lm_extra_tensors")
    config["quantization_config"] = {"quant_method": "compressed-tensors", "format": "awq"}

    class FakeApi:
        def model_info(self, repo_id, *, files_metadata=False, token=None):
            assert repo_id == "owner/Fixture-AWQ"
            assert files_metadata is True
            return SimpleNamespace(
                sha="abc123",
                siblings=[
                    SimpleNamespace(rfilename="config.json", size=123),
                    SimpleNamespace(rfilename="model.safetensors", size=456),
                ],
            )

        def list_repo_files(self, **kwargs):
            raise AssertionError("siblings should provide file names")

    monkeypatch.setattr(forge, "_make_hf_api", lambda: FakeApi())
    monkeypatch.setattr(forge, "_hf_json", lambda repo_id, filename: config)
    monkeypatch.setattr(
        forge,
        "inspect_model",
        lambda source: SimpleNamespace(
            compatibility={"can_run": True},
            mtp=SimpleNamespace(tensor_count=15, sidecar_format="bf16"),
        ),
    )

    payload = forge.probe_source("https://huggingface.co/owner/Fixture-AWQ")

    assert payload["verdict"] == "forgeable"
    assert payload["hf_repo"] == "owner/Fixture-AWQ"
    assert payload["source_format"] == "compressed_tensors_awq"
    assert payload["estimated_size_bytes"] == 579
    assert payload["source_sha"] == "abc123"


def test_requantize_mtp_policy_is_refused_without_explicit_allow(tmp_path, capsys):
    code = main(
        [
            "forge",
            "build",
            "--repo",
            str(tmp_path / "source"),
            "--out",
            str(tmp_path / "out"),
            "--run-id",
            "r1",
            "--recipe",
            '{"mtp_policy":"requantize"}',
            "--branded-name",
            "Fixture-MTPLX-Speed",
        ]
    )

    assert code == 2
    assert "pass --allow-degraded-mtp to confirm" in capsys.readouterr().err


def test_build_probe_failure_stays_on_download_phase(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        forge,
        "probe_source",
        lambda source: {
            "forgeable": False,
            "message": "Could not inspect Hugging Face source.",
            "diagnostic": "model_info signature mismatch",
        },
    )

    code = main(
        [
            "forge",
            "build",
            "--repo",
            "owner/model",
            "--out",
            str(tmp_path / "out"),
            "--run-id",
            "r1",
            "--recipe",
            '{"body_bits":4,"body_group_size":64,"body_mode":"affine","mtp_policy":"keep_bf16"}',
            "--branded-name",
            "Fixture-MTPLX-Speed",
        ]
    )

    run = tmp_path / "out" / "r1"

    assert code == 1
    assert (run / "download.json").exists()
    assert not (run / "convert.json").exists()
    assert not (run / "calibrate.json").exists()
    assert not (run / "verify.json").exists()
    assert "model_info signature mismatch" in capsys.readouterr().err


def test_build_refuses_gemma_subfolder_with_bundle_root_hint(tmp_path, capsys):
    source = tmp_path / "assistant"
    _write_json(
        source / "config.json",
        {
            "architectures": ["Gemma4AssistantForCausalLM"],
            "model_type": "gemma4_text",
            "hidden_size": 1024,
            "num_hidden_layers": 4,
            "vocab_size": 262144,
            "quantization_config": {"bits": 6, "group_size": 64},
        },
    )

    code = main(
        [
            "forge",
            "build",
            "--repo",
            str(source),
            "--out",
            str(tmp_path / "out"),
            "--run-id",
            "r1",
            "--recipe",
            '{"mtp_policy":"keep_bf16"}',
            "--branded-name",
            "Gemma-Assistant-MTPLX-Speed",
        ]
    )

    err = capsys.readouterr().err
    assert code == 1
    assert "bundle root" in err
    assert "mtplx_pair.json" in err
    assert (tmp_path / "out" / "r1" / "download.json").exists()
    assert not (tmp_path / "out" / "r1" / "convert.json").exists()


def test_inspect_reads_runtime_metadata(tmp_path, capsys):
    runtime = _runtime(depth=2)
    _write_json(tmp_path / "mtplx_runtime.json", runtime)

    code = main(["forge", "inspect", str(tmp_path), "--json"])

    assert code == 0
    assert json.loads(capsys.readouterr().out)["mtp_depth_max"] == 2


def test_inspect_falls_back_to_model_inspection_without_runtime_metadata(tmp_path, capsys):
    _write_json(
        tmp_path / "config.json",
        {
            "architectures": ["Gemma4AssistantForCausalLM"],
            "model_type": "gemma4_text",
            "hidden_size": 1024,
            "num_hidden_layers": 4,
            "vocab_size": 262144,
            "quantization_config": {"bits": 6, "group_size": 64},
        },
    )

    code = main(["forge", "inspect", str(tmp_path), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["model_type"] == "gemma4_text"
    assert payload["runtime_compatibility"] == "incomplete-assistant-pair"
    assert "bundle root" in payload["compatibility"]["message"]


def test_runtime_stamp_keeps_depth_max_when_ar_wins(tmp_path):
    _write_json(
        tmp_path / "config.json",
        {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "mtp_num_hidden_layers": 1,
            },
        },
    )
    rows = forge._annotate_verify_rows(
        [
            {"depth": 0, "tok_s": 70.0, "acceptance_by_position": []},
            {"depth": 1, "tok_s": 40.0, "acceptance_by_position": [0.3]},
            {"depth": 2, "tok_s": 41.0, "acceptance_by_position": [0.3, 0.1]},
            {"depth": 3, "tok_s": 39.0, "acceptance_by_position": [0.3, 0.1, 0.0]},
        ]
    )

    runtime = forge._stamp_runtime_metadata(
        tmp_path,
        branded_name="Fixture-MTPLX-Stable",
        source_repo="owner/source",
        source_sha="abc123",
        source_format=forge.SOURCE_COMPRESSED_TENSORS_AWQ,
        recipe={"mtp_policy": "keep_bf16"},
        forge_inputs={"trunk_path": str(tmp_path), "mtp_source_path": str(tmp_path)},
        rows=rows,
        mtp_contract={
            "base_hidden_variant": "post_norm",
            "hidden_variant": "post_norm",
            "concat_order": "embedding_hidden",
        },
        existing=None,
    )

    assert runtime["mtp_depth_max"] == 3
    assert runtime["recommended_profile"] == "stable"
    assert runtime["speed_evidence"]["depth"] == 0
    assert runtime["speed_evidence"]["verdict"] == "no_mtp_depth_beat_ar"
    assert runtime["mtp_contract"]["hidden_variant"] == "post_norm"
    forge._require_verify_rows(rows, require_all_depths=True)


def test_runtime_stamp_rejects_quality_failed_speed_winner(tmp_path):
    _write_json(
        tmp_path / "config.json",
        {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "mtp_num_hidden_layers": 1,
            },
        },
    )
    rows = forge._annotate_verify_rows(
        [
            {"depth": 0, "tok_s": 65.0, "acceptance_by_position": []},
            {
                "depth": 1,
                "tok_s": 59.0,
                "quality_passed": True,
                "acceptance_by_position": [0.84],
            },
            {
                "depth": 2,
                "tok_s": 62.0,
                "quality_passed": True,
                "acceptance_by_position": [0.74, 0.39],
            },
            {
                "depth": 3,
                "tok_s": 68.0,
                "quality_passed": False,
                "acceptance_by_position": [0.83, 0.46, 0.21],
            },
        ]
    )

    runtime = forge._stamp_runtime_metadata(
        tmp_path,
        branded_name="Fixture-MTPLX-Stable",
        source_repo="owner/source",
        source_sha="abc123",
        source_format=forge.SOURCE_COMPRESSED_TENSORS_AWQ,
        recipe={"mtp_policy": "keep_bf16"},
        forge_inputs={"trunk_path": str(tmp_path), "mtp_source_path": str(tmp_path)},
        rows=rows,
        mtp_contract={
            "base_hidden_variant": "post_norm",
            "hidden_variant": "post_norm",
            "concat_order": "embedding_hidden",
        },
        existing=None,
    )

    assert runtime["recommended_profile"] == "stable"
    assert runtime["speed_evidence"]["depth"] == 0
    assert runtime["speed_evidence"]["verdict"] == "no_quality_passed_mtp_depth_beat_ar"
    assert runtime["speed_evidence"]["quality_rejected"][0]["depth"] == 3


def test_runtime_stamp_labels_collapsed_mtp_acceptance(tmp_path):
    _write_json(
        tmp_path / "config.json",
        {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "mtp_num_hidden_layers": 1,
            },
        },
    )
    rows = forge._annotate_verify_rows(
        [
            {"depth": 0, "tok_s": 96.0, "acceptance_by_position": []},
            {"depth": 1, "tok_s": 68.0, "acceptance_by_position": [0.0]},
            {"depth": 2, "tok_s": 54.0, "acceptance_by_position": [0.0, 0.0]},
            {"depth": 3, "tok_s": 45.0, "acceptance_by_position": [0.0, 0.0, 0.0]},
        ]
    )

    runtime = forge._stamp_runtime_metadata(
        tmp_path,
        branded_name="Fixture-MTPLX-Stable",
        source_repo="owner/source",
        source_sha="abc123",
        source_format=forge.SOURCE_COMPRESSED_TENSORS_AWQ,
        recipe={"mtp_policy": "keep_bf16"},
        forge_inputs={"trunk_path": str(tmp_path), "mtp_source_path": str(tmp_path)},
        rows=rows,
        mtp_contract={
            "base_hidden_variant": "post_norm",
            "hidden_variant": "post_norm",
            "concat_order": "embedding_hidden",
        },
        existing=None,
    )

    assert runtime["recommended_profile"] == "stable"
    assert runtime["speed_evidence"]["depth"] == 0
    assert runtime["speed_evidence"]["verdict"] == "mtp_acceptance_collapsed"
    assert runtime["speed_evidence"]["failure_reasons"] == [
        "mtp_acceptance_collapsed",
        "no_mtp_depth_beat_ar",
    ]
    assert [row["depth"] for row in runtime["speed_evidence"]["acceptance_collapsed"]] == [
        1,
        2,
        3,
    ]


def test_speed_win_gate_writes_no_failure_outcome(tmp_path):
    _write_json(tmp_path / "model" / "config.json", _mtp_config())
    rows = forge._annotate_verify_rows(
        [
            {"depth": 0, "tok_s": 108.4, "acceptance_by_position": []},
            {"depth": 1, "tok_s": 114.2, "acceptance_by_position": [0.66]},
            {"depth": 2, "tok_s": 120.1, "acceptance_by_position": [0.66, 0.22]},
            {"depth": 3, "tok_s": 116.8, "acceptance_by_position": [0.62, 0.20, 0.08]},
        ]
    )

    evidence = forge._require_speed_win_or_write_outcome(
        tmp_path / "model",
        tmp_path / "run",
        rows,
    )

    assert evidence["verdict"] == "mtp_depth_wins"
    assert evidence["depth"] == 2
    assert not (tmp_path / "run" / "build_outcome.json").exists()


def test_zero_acceptance_speed_gate_writes_build_outcome_not_runtime(tmp_path):
    model = tmp_path / "Qwen-Qwen3.5-9B-MTPLX-Speed"
    _write_json(model / "config.json", _mtp_config())
    rows = forge._annotate_verify_rows(
        [
            {"depth": 0, "tok_s": 94.65, "acceptance_by_position": []},
            {"depth": 1, "tok_s": 67.23, "acceptance_by_position": [0.0]},
            {"depth": 2, "tok_s": 54.87, "acceptance_by_position": [0.0, 0.0]},
            {"depth": 3, "tok_s": 45.48, "acceptance_by_position": [0.0, 0.0, 0.0]},
        ]
    )

    with pytest.raises(forge.ForgeError, match="MTP did not accelerate this model"):
        forge._require_speed_win_or_write_outcome(model, tmp_path / "run", rows)

    outcome = json.loads((tmp_path / "run" / "build_outcome.json").read_text(encoding="utf-8"))
    assert outcome["verdict"] == "mtp_acceptance_collapsed"
    assert outcome["failure_reasons"] == [
        "mtp_acceptance_collapsed",
        "no_mtp_depth_beat_ar",
    ]
    assert outcome["ar_tok_s"] == 94.65
    assert outcome["best_mtp_depth"] == 1
    assert outcome["best_mtp_tok_s"] == 67.23
    assert [row["depth"] for row in outcome["verify_rows"]] == [0, 1, 2, 3]
    assert not (model / "mtplx_runtime.json").exists()
    assert not (tmp_path / "run" / "forge.json").exists()


def test_build_zero_agreement_contract_still_measures_speed_rows(tmp_path, monkeypatch):
    source = tmp_path / "source"
    _write_json(source / "config.json", _mtp_config())
    model_root = tmp_path / "models"
    monkeypatch.setenv("MTPLX_FORGE_MODEL_ROOT", str(model_root))
    rows = [
        {"depth": 0, "tok_s": 94.65, "acceptance_by_position": []},
        {"depth": 1, "tok_s": 67.23, "acceptance_by_position": [0.0]},
        {"depth": 2, "tok_s": 54.87, "acceptance_by_position": [0.0, 0.0]},
        {"depth": 3, "tok_s": 45.48, "acceptance_by_position": [0.0, 0.0, 0.0]},
    ]
    calls = {"verify": 0}

    monkeypatch.setattr(
        forge,
        "probe_source",
        lambda repo: {
            "forgeable": True,
            "source_format": forge.SOURCE_MLX_AFFINE_WITH_MTP,
            "source_sha": "abc123",
        },
    )
    monkeypatch.setattr(
        forge,
        "_prepare_source",
        lambda repo, run, probe: (source, "Qwen/Qwen3.5-9B", "abc123"),
    )

    def fake_mirror(source_path, destination):
        destination.mkdir(parents=True)
        _write_json(destination / "config.json", _mtp_config())

    def fake_calibrate_sidecar(source_path, destination, *, recipe, run):
        forge._write_progress(
            run,
            "calibrate",
            progress=1.0,
            label="pack_sidecar",
            finished=True,
        )

    def fake_calibrate_contract(destination, run, *, recipe, existing, max_fans=False):
        return {
            "base_hidden_variant": "post_norm",
            "hidden_variant": "post_norm",
            "concat_order": "embedding_hidden",
            "calibration": {
                "status": "no_agreement_signal",
                "diagnostic": "MTP contract calibration found no agreement signal",
            },
        }

    def fake_run_verify(
        destination,
        run,
        *,
        max_fans,
        mtp_contract=None,
        max_tokens=None,
        prompt_suite=None,
    ):
        calls["verify"] += 1
        assert mtp_contract["calibration"]["status"] == "no_agreement_signal"
        return rows

    monkeypatch.setattr(forge, "_mirror_model_tree", fake_mirror)
    monkeypatch.setattr(forge, "_calibrate_sidecar", fake_calibrate_sidecar)
    monkeypatch.setattr(forge, "_calibrate_mtp_contract", fake_calibrate_contract)
    monkeypatch.setattr(forge, "_run_verify", fake_run_verify)

    code = main(
        [
            "forge",
            "build",
            "--repo",
            "Qwen/Qwen3.5-9B",
            "--out",
            str(tmp_path / "out"),
            "--run-id",
            "r1",
            "--recipe",
            '{"body_bits":4,"body_group_size":64,"body_mode":"affine","mtp_policy":"keep_bf16"}',
            "--branded-name",
            "Qwen-Qwen3.5-9B-MTPLX-Speed",
        ]
    )

    run = tmp_path / "out" / "r1"
    forged = model_root / "Qwen-Qwen3.5-9B-MTPLX-Speed"
    outcome = json.loads((run / "build_outcome.json").read_text(encoding="utf-8"))
    config = json.loads((forged / "config.json").read_text(encoding="utf-8"))

    assert code == 1
    assert calls == {"verify": 1}
    assert "calibration" not in config["mtplx_mtp_contract"]
    assert outcome["phase"] == "verify"
    assert outcome["diagnostic"] == "MTP contract calibration found no agreement signal"
    assert outcome["verdict"] == "mtp_acceptance_collapsed"
    assert outcome["ar_tok_s"] == 94.65
    assert outcome["best_mtp_depth"] == 1
    assert [row["depth"] for row in outcome["verify_rows"]] == [0, 1, 2, 3]
    assert not (forged / "mtplx_runtime.json").exists()
    assert not (run / "forge.json").exists()


def test_runtime_stamp_clears_stale_launch_blockers(tmp_path):
    _write_json(
        tmp_path / "config.json",
        {
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "mtp_num_hidden_layers": 1,
            },
        },
    )
    rows = forge._annotate_verify_rows(
        [
            {"depth": 0, "tok_s": 90.0, "acceptance_by_position": []},
            {"depth": 1, "tok_s": 120.0, "acceptance_by_position": [0.8]},
            {"depth": 2, "tok_s": 130.0, "acceptance_by_position": [0.8, 0.4]},
            {"depth": 3, "tok_s": 110.0, "acceptance_by_position": [0.7, 0.4, 0.2]},
        ]
    )
    existing = _runtime(depth=3)
    existing["mtplx_version"] = "0.1.0-preview"
    existing["exactness_baseline"] = {
        "status": "pending-cyankiwi-35b-moe-benchmark"
    }
    existing["verified_on"] = {"model": "old-candidate"}

    runtime = forge._stamp_runtime_metadata(
        tmp_path,
        branded_name="Fixture-MTPLX-Repaired",
        source_repo="owner/source",
        source_sha="abc123",
        source_format=forge.SOURCE_MLX_AFFINE_WITH_MTP,
        recipe={"mtp_policy": "keep_bf16"},
        forge_inputs={"trunk_path": str(tmp_path), "mtp_source_path": str(tmp_path)},
        rows=rows,
        mtp_contract={
            "base_hidden_variant": "pre_norm",
            "hidden_variant": "post_norm",
            "concat_order": "embedding_hidden",
        },
        existing=existing,
    )

    assert runtime["exactness_baseline"] == {}
    assert runtime["mtplx_version"] == forge.__version__
    assert runtime["verified_on"]["model"] == "Fixture-MTPLX-Repaired"
    assert runtime["speed_evidence"]["verdict"] == "mtp_depth_wins"


def test_build_verify_rows_must_include_ar_and_d1_d2_d3():
    with pytest.raises(forge.ForgeError, match="D2, D3"):
        forge._require_verify_rows(
            [
                {"depth": 0, "tok_s": 70.0},
                {"depth": 1, "tok_s": 40.0},
            ],
            require_all_depths=True,
        )


def test_contract_calibration_default_prompt_path_is_absolute():
    path = forge._contract_calibration_prompts_path(None)

    assert path.is_absolute()
    assert path.name == "calibration_coding.jsonl"
    assert path.exists()


def test_requested_max_session_fails_closed_without_verified_ramp(monkeypatch):
    class FailedSession:
        def __init__(self, *, log):
            self.log = log
            self.thermal = {"verified": {"message": "actual RPM never ramped"}}

        def start(self):
            return False

    import mtplx.thermal as thermal

    monkeypatch.setattr(thermal, "MaxSession", FailedSession)

    with pytest.raises(forge.ForgeError, match="refusing Forge model load"):
        forge._start_max_session_if_requested(True)


@pytest.mark.parametrize("max_fans", [False, True])
def test_forge_verify_only_requires_verified_ramp_when_requested(
    tmp_path, monkeypatch, max_fans
):
    captured: dict[str, list[str]] = {}

    class FinishedProcess:
        returncode = 1

        def poll(self):
            return self.returncode

    def fake_popen(command, **_kwargs):
        captured["command"] = command
        return FinishedProcess()

    monkeypatch.setattr(forge.subprocess, "Popen", fake_popen)

    with pytest.raises(forge.ForgeError, match="mtplx tune failed"):
        forge._run_verify(
            tmp_path / "model",
            tmp_path / "run",
            max_fans=max_fans,
        )

    assert ("--require-max-fans" in captured["command"]) is max_fans


def test_contract_calibration_fails_closed_on_probe_failure(tmp_path, monkeypatch):
    class FailedRun:
        returncode = 2

    def fake_run(*args, **kwargs):
        return FailedRun()

    monkeypatch.setattr(forge.subprocess, "run", fake_run)

    with pytest.raises(forge.ForgeError, match="MTP contract calibration failed"):
        forge._calibrate_mtp_contract(
            tmp_path / "model",
            tmp_path / "run",
            recipe={},
            existing=None,
            max_fans=False,
        )

    progress = json.loads((tmp_path / "run" / "calibrate.json").read_text(encoding="utf-8"))
    assert progress["label"] == "contract_calibration_failed"
    assert progress["contract_candidates_tested"] == 0
    assert progress["best_agreement"] == 0.0
    assert progress["best_topk_hint"] == 0.0
    outcome = json.loads((tmp_path / "run" / "build_outcome.json").read_text(encoding="utf-8"))
    assert outcome["verdict"] == "contract_calibration_failed"
    assert outcome["phase"] == "calibrate"
    assert outcome["failure_reasons"] == ["contract_calibration_failed"]


def test_contract_chain_probe_keeps_zero_agreement_as_diagnostic():
    contract = forge._contract_from_chain_probe(
        {
            "variants": [
                {
                    "base_hidden_variant": "post_norm",
                    "hidden_variant": "post_norm",
                    "concat_order": "embedding_hidden",
                    "agreement_by_depth": [0.0, 0.0, 0.0],
                    "topk_rates_by_depth": {"4": [0.0, 0.0, 0.0], "8": [0.0, 0.0, 0.0]},
                }
            ]
        },
        fallback={
            "base_hidden_variant": "post_norm",
            "hidden_variant": "post_norm",
            "concat_order": "embedding_hidden",
        },
    )

    assert contract["hidden_variant"] == "post_norm"
    assert contract["calibration"]["status"] == "no_agreement_signal"
    assert "no agreement signal" in contract["calibration"]["diagnostic"]


def test_contract_chain_probe_keeps_topk_only_signal_as_diagnostic():
    contract = forge._contract_from_chain_probe(
        {
            "variants": [
                {
                    "base_hidden_variant": "post_norm",
                    "hidden_variant": "fc",
                    "concat_order": "hidden_embedding",
                    "agreement_by_depth": [0.0, 0.0, 0.0],
                    "topk_rates_by_depth": {"4": [0.0, 0.0, 1.0], "8": [0.0, 0.0, 1.0]},
                }
            ]
        },
        fallback={
            "base_hidden_variant": "post_norm",
            "hidden_variant": "post_norm",
            "concat_order": "embedding_hidden",
        },
    )

    assert contract["hidden_variant"] == "fc"
    assert contract["concat_order"] == "hidden_embedding"
    assert contract["calibration"]["status"] == "topk_only_no_exact_agreement"
    assert "top-k hints" in contract["calibration"]["diagnostic"]


def test_discover_maps_hf_rows(monkeypatch):
    class FakeApi:
        def list_models(self, **kwargs):
            assert kwargs["search"] == "MTPLX"
            return [
                SimpleNamespace(
                    modelId="owner/Plain-Model",
                    downloads=99,
                    tags=[],
                    siblings=[],
                ),
                SimpleNamespace(
                    modelId="owner/Qwen-MTPLX-Speed",
                    downloads=42,
                    tags=["license:apache-2.0"],
                    siblings=[SimpleNamespace(size=123)],
                    lastModified="2026-05-26T00:00:00Z",
                ),
            ]

    monkeypatch.setattr(forge, "_make_hf_api", lambda: FakeApi())

    cards = forge.discover_models(query="MTPLX", limit=5, offset=0)

    assert cards == [
        {
            "repo": "owner/Qwen-MTPLX-Speed",
            "owner": "owner",
            "branded_name": "Qwen-MTPLX-Speed",
            "downloads": 42,
            "size_bytes": 123,
            "license": "apache-2.0",
            "last_updated": "2026-05-26T00:00:00Z",
        }
    ]


def test_discover_network_failure_mentions_hf_unreachable(monkeypatch, capsys):
    def fail():
        raise RuntimeError("name resolution failed")

    monkeypatch.setattr(forge, "_make_hf_api", fail)

    code = main(["forge", "discover", "--json", "--limit", "5"])

    assert code == 1
    assert "hf_unreachable" in capsys.readouterr().err


def test_build_local_already_mtplx_writes_phase_files_and_runtime(tmp_path, monkeypatch):
    source = tmp_path / "source"
    _write_json(source / "config.json", _mtp_config())
    _write_json(source / "mtplx_runtime.json", _runtime(depth=3))
    model_root = tmp_path / "models"
    monkeypatch.setenv("MTPLX_FORGE_MODEL_ROOT", str(model_root))

    code = main(
        [
            "forge",
            "build",
            "--repo",
            str(source),
            "--out",
            str(tmp_path / "out"),
            "--run-id",
            "r1",
            "--recipe",
            '{"body_bits":4,"body_group_size":64,"body_mode":"affine","mtp_policy":"keep_bf16"}',
            "--branded-name",
            "Fixture-MTPLX-Speed",
        ]
    )

    run = tmp_path / "out" / "r1"
    forged = model_root / "Fixture-MTPLX-Speed"
    runtime = json.loads((forged / "mtplx_runtime.json").read_text(encoding="utf-8"))
    forge_done = json.loads((run / "forge.json").read_text(encoding="utf-8"))

    assert code == 0
    assert forge_done["local_path"] == str(forged)
    assert runtime["forge_provenance"]["forged_locally"] is True
    assert runtime["forge_provenance"]["source_format"] == "mlx_affine_with_mtp"
    assert runtime["speed_evidence"]["verdict"] == "mtp_depth_wins"
    assert runtime["speed_evidence"]["depth"] == 3
    assert {row["depth"] for row in runtime["speed_evidence"]["forge_verify_rows"]} == {0, 1, 2, 3}
    assert json.loads((run / "download.json").read_text(encoding="utf-8"))["finished"] is True
    assert json.loads((run / "convert.json").read_text(encoding="utf-8"))["finished"] is True
    assert json.loads((run / "calibrate.json").read_text(encoding="utf-8"))["finished"] is True
    verify = json.loads((run / "verify.json").read_text(encoding="utf-8"))
    assert verify["verdict"] == "mtp_depth_wins"
    assert {row["depth"] for row in verify["rows"]} == {0, 1, 2, 3}
    assert not (run / "build_outcome.json").exists()


def test_build_reuses_legacy_speed_grid_positive_control(tmp_path, monkeypatch):
    source = tmp_path / "source"
    _write_json(source / "config.json", _mtp_config())
    grid_path = tmp_path / "legacy_speed_grid.json"
    _write_legacy_speed_grid(grid_path)
    runtime = _runtime(depth=2)
    runtime.pop("mtp_contract")
    runtime["speed_evidence"] = {
        "artifact": str(grid_path),
        "ar_tok_s": 22.0,
        "mtp_depth": 2,
        "mtp_tok_s": 40.0,
        "acceptance_by_depth": [0.88, 0.55],
        "speedup_vs_ar": 1.8181818182,
    }
    _write_json(source / "mtplx_runtime.json", runtime)
    model_root = tmp_path / "models"
    monkeypatch.setenv("MTPLX_FORGE_MODEL_ROOT", str(model_root))
    monkeypatch.setattr(
        forge,
        "_calibrate_mtp_contract",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should reuse legacy grid")),
    )
    monkeypatch.setattr(
        forge,
        "_run_verify",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should reuse legacy grid")),
    )

    code = main(
        [
            "forge",
            "build",
            "--repo",
            str(source),
            "--out",
            str(tmp_path / "out"),
            "--run-id",
            "r1",
            "--recipe",
            '{"body_bits":4,"body_group_size":64,"body_mode":"affine","mtp_policy":"keep_bf16"}',
            "--branded-name",
            "Fixture-MTPLX-Speed",
        ]
    )

    run = tmp_path / "out" / "r1"
    runtime = json.loads(
        (model_root / "Fixture-MTPLX-Speed" / "mtplx_runtime.json").read_text(
            encoding="utf-8"
        )
    )
    verify = json.loads((run / "verify.json").read_text(encoding="utf-8"))

    assert code == 0
    assert runtime["speed_evidence"]["verdict"] == "mtp_depth_wins"
    assert runtime["speed_evidence"]["depth"] == 3
    assert runtime["mtp_contract"]["hidden_variant"] == "post_norm"
    assert {row["depth"] for row in verify["rows"]} == {0, 1, 2, 3}
    assert not (run / "build_outcome.json").exists()


def test_saved_verify_rows_are_bound_to_the_forged_artifact(tmp_path):
    model = tmp_path / "model"
    _write_qwen_sidecar_fixture(model)
    runtime = _runtime(depth=3)
    runtime["speed_evidence"]["artifact_fingerprint"] = (
        forge._verification_artifact_fingerprint(model)
    )

    assert (
        forge._saved_verify_rows_reuse_blocker(
            _speed_win_rows(),
            runtime,
            model_path=model,
            source_path=model,
            require_all_depths=True,
        )
        is None
    )

    changed_config = _mtp_config()
    changed_config["mtplx_mtp_quantization"] = {
        "policy": "requantize",
        "bits": 4,
        "group_size": 64,
    }
    _write_json(model / "config.json", changed_config)
    assert forge._saved_verify_rows_reuse_blocker(
        _speed_win_rows(),
        runtime,
        model_path=model,
        source_path=model,
        require_all_depths=True,
    ) == "saved verification belongs to different artifact bytes"

    runtime["speed_evidence"].pop("artifact_fingerprint")
    runtime["forge_provenance"] = {"forged_at": "2026-05-27T00:00:00+01:00"}
    assert forge._saved_verify_rows_reuse_blocker(
        _speed_win_rows(),
        runtime,
        model_path=model,
        source_path=model,
        require_all_depths=True,
    ) == "saved verification predates the forged artifact"


def test_build_reverifies_old_runtime_without_mtp_contract(tmp_path, monkeypatch):
    source = tmp_path / "source"
    _write_json(source / "config.json", _mtp_config())
    old_runtime = _runtime(depth=3)
    old_runtime.pop("mtp_contract")
    _write_json(source / "mtplx_runtime.json", old_runtime)
    model_root = tmp_path / "models"
    monkeypatch.setenv("MTPLX_FORGE_MODEL_ROOT", str(model_root))
    calls = {"calibrate_contract": 0, "verify": 0}

    def fake_calibrate_contract(destination, run, *, recipe, existing, max_fans=False):
        calls["calibrate_contract"] += 1
        return {
            "base_hidden_variant": "post_norm",
            "hidden_variant": "fc",
            "concat_order": "hidden_embedding",
        }

    def fake_run_verify(
        destination,
        run,
        *,
        max_fans,
        mtp_contract=None,
        max_tokens=None,
        prompt_suite=None,
    ):
        calls["verify"] += 1
        assert mtp_contract["hidden_variant"] == "fc"
        assert max_tokens == forge.FORGE_VERIFY_DEFAULT_MAX_TOKENS
        assert prompt_suite == forge.FORGE_VERIFY_DEFAULT_PROMPT_SUITE
        return [
            {
                "depth": 0,
                "tok_s": 20.0,
                "multiplier_vs_ar": 1.0,
                "acceptance_by_position": [],
                "verify_time_s": 0.1,
            },
            {
                "depth": 1,
                "tok_s": 30.0,
                "multiplier_vs_ar": 1.5,
                "acceptance_by_position": [0.8],
                "verify_time_s": 0.2,
            },
            {
                "depth": 2,
                "tok_s": 32.0,
                "multiplier_vs_ar": 1.6,
                "acceptance_by_position": [0.8, 0.6],
                "verify_time_s": 0.3,
            },
            {
                "depth": 3,
                "tok_s": 31.0,
                "multiplier_vs_ar": 1.55,
                "acceptance_by_position": [0.8, 0.6, 0.4],
                "verify_time_s": 0.4,
            },
        ]

    monkeypatch.setattr(forge, "_calibrate_mtp_contract", fake_calibrate_contract)
    monkeypatch.setattr(forge, "_run_verify", fake_run_verify)

    code = main(
        [
            "forge",
            "build",
            "--repo",
            str(source),
            "--out",
            str(tmp_path / "out"),
            "--run-id",
            "r1",
            "--recipe",
            '{"body_bits":4,"body_group_size":64,"body_mode":"affine","mtp_policy":"keep_bf16"}',
            "--branded-name",
            "Fixture-MTPLX-Speed",
        ]
    )

    runtime = json.loads(
        (model_root / "Fixture-MTPLX-Speed" / "mtplx_runtime.json").read_text(
            encoding="utf-8"
        )
    )

    assert code == 0
    assert calls == {"calibrate_contract": 1, "verify": 1}
    assert runtime["mtp_contract"]["hidden_variant"] == "fc"
    assert runtime["speed_evidence"]["depth"] == 2


def test_build_reverifies_saved_losing_runtime_with_contract(tmp_path, monkeypatch):
    source = tmp_path / "source"
    _write_json(source / "config.json", _mtp_config())
    stale_runtime = _runtime(depth=3)
    stale_runtime["speed_evidence"] = {
        "depth": 0,
        "tok_s": [70.0],
        "greedy_diagnostic": {"tok_s": 70.0},
        "verdict": "no_mtp_depth_beat_ar",
        "failure_reasons": ["no_mtp_depth_beat_ar"],
        "forge_verify_rows": [
            {
                "depth": 0,
                "tok_s": 70.0,
                "multiplier_vs_ar": 1.0,
                "acceptance_by_position": [],
                "verify_time_s": 0.1,
            },
            {
                "depth": 1,
                "tok_s": 64.0,
                "multiplier_vs_ar": 0.91,
                "acceptance_by_position": [0.8],
                "verify_time_s": 0.2,
            },
            {
                "depth": 2,
                "tok_s": 65.0,
                "multiplier_vs_ar": 0.93,
                "acceptance_by_position": [0.8, 0.4],
                "verify_time_s": 0.3,
            },
            {
                "depth": 3,
                "tok_s": 63.0,
                "multiplier_vs_ar": 0.9,
                "acceptance_by_position": [0.8, 0.4, 0.2],
                "verify_time_s": 0.4,
            },
        ],
    }
    _write_json(source / "mtplx_runtime.json", stale_runtime)
    model_root = tmp_path / "models"
    monkeypatch.setenv("MTPLX_FORGE_MODEL_ROOT", str(model_root))
    calls = {"calibrate_contract": 0, "verify": 0}

    def fake_calibrate_contract(destination, run, *, recipe, existing, max_fans=False):
        calls["calibrate_contract"] += 1
        assert existing["speed_evidence"]["verdict"] == "no_mtp_depth_beat_ar"
        return {
            "base_hidden_variant": "post_norm",
            "hidden_variant": "post_norm",
            "concat_order": "embedding_hidden",
        }

    def fake_run_verify(
        destination,
        run,
        *,
        max_fans,
        mtp_contract=None,
        max_tokens=None,
        prompt_suite=None,
    ):
        calls["verify"] += 1
        return [
            {
                "depth": 0,
                "tok_s": 64.0,
                "multiplier_vs_ar": 1.0,
                "acceptance_by_position": [],
                "verify_time_s": 0.1,
            },
            {
                "depth": 1,
                "tok_s": 66.0,
                "multiplier_vs_ar": 1.03125,
                "acceptance_by_position": [0.84],
                "verify_time_s": 0.2,
                "quality_passed": True,
            },
            {
                "depth": 2,
                "tok_s": 70.0,
                "multiplier_vs_ar": 1.09375,
                "acceptance_by_position": [0.82, 0.45],
                "verify_time_s": 0.3,
                "quality_passed": True,
            },
            {
                "depth": 3,
                "tok_s": 68.0,
                "multiplier_vs_ar": 1.0625,
                "acceptance_by_position": [0.8, 0.42, 0.18],
                "verify_time_s": 0.4,
                "quality_passed": True,
            },
        ]

    monkeypatch.setattr(forge, "_calibrate_mtp_contract", fake_calibrate_contract)
    monkeypatch.setattr(forge, "_run_verify", fake_run_verify)

    code = main(
        [
            "forge",
            "build",
            "--repo",
            str(source),
            "--out",
            str(tmp_path / "out"),
            "--run-id",
            "r1",
            "--recipe",
            '{"body_bits":4,"body_group_size":64,"body_mode":"affine","mtp_policy":"keep_bf16"}',
            "--branded-name",
            "Fixture-MTPLX-Speed",
        ]
    )

    runtime = json.loads(
        (model_root / "Fixture-MTPLX-Speed" / "mtplx_runtime.json").read_text(
            encoding="utf-8"
        )
    )

    assert code == 0
    assert calls == {"calibrate_contract": 1, "verify": 1}
    assert runtime["speed_evidence"]["verdict"] == "mtp_depth_wins"
    assert runtime["speed_evidence"]["depth"] == 2
    assert {row["depth"] for row in runtime["speed_evidence"]["forge_verify_rows"]} == {0, 1, 2, 3}


def test_build_reverifies_saved_candidate_runtime_with_contract(tmp_path, monkeypatch):
    source = tmp_path / "source"
    _write_json(source / "config.json", _mtp_config())
    stale_runtime = _runtime(depth=3)
    stale_runtime["exactness_baseline"] = {
        "status": "candidate_build_only_benchmark_pending",
        "public_release_blocker": True,
    }
    _write_json(source / "mtplx_runtime.json", stale_runtime)
    model_root = tmp_path / "models"
    monkeypatch.setenv("MTPLX_FORGE_MODEL_ROOT", str(model_root))
    calls = {"calibrate_contract": 0, "verify": 0}

    def fake_calibrate_contract(destination, run, *, recipe, existing, max_fans=False):
        calls["calibrate_contract"] += 1
        assert existing["exactness_baseline"]["public_release_blocker"] is True
        return {
            "base_hidden_variant": "post_norm",
            "hidden_variant": "post_norm",
            "concat_order": "embedding_hidden",
        }

    def fake_run_verify(
        destination,
        run,
        *,
        max_fans,
        mtp_contract=None,
        max_tokens=None,
        prompt_suite=None,
    ):
        calls["verify"] += 1
        return _speed_win_rows()

    monkeypatch.setattr(forge, "_calibrate_mtp_contract", fake_calibrate_contract)
    monkeypatch.setattr(forge, "_run_verify", fake_run_verify)

    code = main(
        [
            "forge",
            "build",
            "--repo",
            str(source),
            "--out",
            str(tmp_path / "out"),
            "--run-id",
            "r1",
            "--recipe",
            '{"body_bits":4,"body_group_size":64,"body_mode":"affine","mtp_policy":"keep_bf16"}',
            "--branded-name",
            "Fixture-MTPLX-Speed",
        ]
    )

    runtime = json.loads(
        (model_root / "Fixture-MTPLX-Speed" / "mtplx_runtime.json").read_text(
            encoding="utf-8"
        )
    )

    assert code == 0
    assert calls == {"calibrate_contract": 1, "verify": 1}
    assert runtime["exactness_baseline"] == {}
    assert runtime["speed_evidence"]["verdict"] == "mtp_depth_wins"


def test_build_conversion_receives_nonexistent_destination(tmp_path, monkeypatch):
    source = tmp_path / "source"
    _write_json(source / "config.json", _mtp_config())
    model_root = tmp_path / "models"
    monkeypatch.setenv("MTPLX_FORGE_MODEL_ROOT", str(model_root))
    monkeypatch.setattr(
        forge,
        "inspect_model",
        lambda source: SimpleNamespace(
            compatibility={"can_run": True},
            mtp=SimpleNamespace(tensor_count=15, sidecar_format="bf16"),
        ),
    )

    def fake_convert(source_path, destination, *, recipe, source_format, run):
        assert source_path == source
        assert source_format == "bf16_native"
        assert not destination.exists()
        destination.mkdir(parents=True)
        _write_json(destination / "config.json", _mtp_config())
        forge._write_progress(run, "convert", progress=1.0, label="quantize_body", finished=True)

    def fake_calibrate(source_path, destination, *, recipe, run):
        forge._write_progress(run, "calibrate", progress=1.0, label="pack_sidecar", finished=True)

    monkeypatch.setattr(forge, "_convert_with_mlx_lm", fake_convert)
    monkeypatch.setattr(forge, "_calibrate_sidecar", fake_calibrate)
    monkeypatch.setattr(
        forge,
        "_calibrate_mtp_contract",
        lambda destination, run, *, recipe, existing, max_fans=False: forge._runtime_or_default_mtp_contract(existing),
    )
    monkeypatch.setattr(
        forge,
        "_run_verify",
        lambda destination, run, *, max_fans, mtp_contract=None, max_tokens=None, prompt_suite=None: _speed_win_rows(),
    )

    code = main(
        [
            "forge",
            "build",
            "--repo",
            str(source),
            "--out",
            str(tmp_path / "out"),
            "--run-id",
            "r1",
            "--recipe",
            '{"body_bits":4,"body_group_size":64,"body_mode":"affine","mtp_policy":"keep_bf16"}',
            "--branded-name",
            "Fixture-MTPLX-Speed",
        ]
    )

    forged = model_root / "Fixture-MTPLX-Speed"
    runtime = json.loads((forged / "mtplx_runtime.json").read_text(encoding="utf-8"))

    assert code == 0
    assert forged.is_dir()
    assert runtime["forge_provenance"]["source_format"] == "bf16_native"


def test_build_compressed_tensors_uses_packed_awq_converter(tmp_path, monkeypatch):
    source = tmp_path / "source"
    config = _mtp_config()
    config.pop("mlx_lm_extra_tensors")
    config["quantization_config"] = {"quant_method": "compressed-tensors", "format": "awq"}
    _write_json(source / "config.json", config)
    model_root = tmp_path / "models"
    monkeypatch.setenv("MTPLX_FORGE_MODEL_ROOT", str(model_root))
    monkeypatch.setattr(
        forge,
        "inspect_model",
        lambda source: SimpleNamespace(
            compatibility={"can_run": True},
            mtp=SimpleNamespace(tensor_count=15, sidecar_format="bf16"),
        ),
    )

    def fake_convert(source_path, destination, *, run, source_repo, source_sha):
        assert source_path == source
        assert source_repo == str(source)
        assert not destination.exists()
        destination.mkdir(parents=True)
        _write_json(destination / "config.json", config)
        mx.save_safetensors(
            str(destination / "mtp.safetensors"),
            {
                "mtp.layers.0.self_attn.q_proj.weight": mx.array([[1], [2]], dtype=mx.uint32),
                "mtp.layers.0.self_attn.q_proj.scales": mx.ones((2, 1), dtype=mx.float16),
                "mtp.layers.0.self_attn.q_proj.biases": mx.zeros((2, 1), dtype=mx.float16),
            },
        )
        forge._write_progress(run, "convert", progress=1.0, label="convert_packed_awq", finished=True)

    monkeypatch.setattr(forge, "_convert_compressed_tensors_awq", fake_convert)
    monkeypatch.setattr(
        forge,
        "_calibrate_mtp_contract",
        lambda destination, run, *, recipe, existing, max_fans=False: forge._runtime_or_default_mtp_contract(existing),
    )
    monkeypatch.setattr(
        forge,
        "_run_verify",
        lambda destination, run, *, max_fans, mtp_contract=None, max_tokens=None, prompt_suite=None: _speed_win_rows(),
    )

    code = main(
        [
            "forge",
            "build",
            "--repo",
            str(source),
            "--out",
            str(tmp_path / "out"),
            "--run-id",
            "r1",
            "--recipe",
            '{"body_bits":4,"body_group_size":64,"body_mode":"affine","mtp_policy":"keep_bf16"}',
            "--branded-name",
            "Fixture-MTPLX-Speed",
        ]
    )

    forged = model_root / "Fixture-MTPLX-Speed"
    runtime = json.loads((forged / "mtplx_runtime.json").read_text(encoding="utf-8"))

    assert code == 0
    assert forge._audit_mtp_sidecar_payload(forged / "mtp.safetensors")["passed"] is True
    assert runtime["forge_provenance"]["source_format"] == "compressed_tensors_awq"


def test_mlx_lm_convert_command_uses_supported_entrypoint(tmp_path):
    command = forge._mlx_lm_convert_command(
        tmp_path / "source",
        tmp_path / "dest",
        recipe={"body_bits": 4, "body_group_size": 64, "body_mode": "affine"},
        source_format="compressed_tensors_awq",
    )

    assert command[1:5] == ["-P", "-m", "mlx_lm", "convert"]
    assert "mlx_lm.convert" not in command
    assert "--dequantize" in command
    assert command[-6:] == ["--q-bits", "4", "--q-group-size", "64", "--q-mode", "affine"]


def test_mlx_lm_convert_command_default_dtype_passes_no_dtype_flag(tmp_path):
    command = forge._mlx_lm_convert_command(
        tmp_path / "source",
        tmp_path / "dest",
        recipe={"body_bits": 4},
        source_format="bf16_native",
    )

    assert "--dtype" not in command


def test_mlx_lm_convert_command_fp16_recipe_sets_float16_dtype(tmp_path):
    for spelling in ("fp16", "float16", "f16"):
        command = forge._mlx_lm_convert_command(
            tmp_path / "source",
            tmp_path / "dest",
            recipe={"body_bits": 4, "body_dtype": spelling},
            source_format="bf16_native",
        )

        assert command[-2:] == ["--dtype", "float16"]


def test_body_dtype_rejects_unknown_value():
    with pytest.raises(forge.ForgeError):
        forge._body_dtype({"body_dtype": "int8"})


def test_body_dtype_auto_resolves_by_host_chip(monkeypatch):
    monkeypatch.setattr(forge, "_host_chip_brand", lambda: "Apple M2 Max")
    assert forge._body_dtype({"body_dtype": "auto"}) == "fp16"
    monkeypatch.setattr(forge, "_host_chip_brand", lambda: "Apple M1")
    assert forge._body_dtype({"body_dtype": "auto"}) == "fp16"
    monkeypatch.setattr(forge, "_host_chip_brand", lambda: "Apple M5 Max")
    assert forge._body_dtype({"body_dtype": "auto"}) == "bf16"
    monkeypatch.setattr(forge, "_host_chip_brand", lambda: "")
    assert forge._body_dtype({"body_dtype": "auto"}) == "bf16"


def test_cast_sidecar_float_tensors_fp16(tmp_path):
    sidecar = tmp_path / "mtp.safetensors"
    mx.save_safetensors(
        str(sidecar),
        {
            "mtp.weight_bf16": mx.ones((2, 2), dtype=mx.bfloat16),
            "mtp.weight_f32": mx.ones((2, 2), dtype=mx.float32),
            "mtp.weight_f16": mx.ones((2, 2), dtype=mx.float16),
            "mtp.qidx": mx.zeros((2, 2), dtype=mx.uint32),
        },
    )

    forge._cast_sidecar_float_tensors_fp16(sidecar)

    reloaded = mx.load(str(sidecar))
    assert reloaded["mtp.weight_bf16"].dtype == mx.float16
    assert reloaded["mtp.weight_f32"].dtype == mx.float16
    assert reloaded["mtp.weight_f16"].dtype == mx.float16
    assert reloaded["mtp.qidx"].dtype == mx.uint32


def _healthy_sidecar_tensors() -> dict[str, mx.array]:
    return {
        "mtp.layers.0.input_layernorm.weight": mx.full((4,), 1.10, dtype=mx.bfloat16),
        "mtp.layers.0.post_attention_layernorm.weight": mx.full((4,), 1.24, dtype=mx.bfloat16),
        "mtp.layers.0.self_attn.q_norm.weight": mx.full((4,), 1.75, dtype=mx.bfloat16),
        "mtp.layers.0.self_attn.k_norm.weight": mx.full((4,), 1.74, dtype=mx.bfloat16),
        "mtp.norm.weight": mx.full((4,), 2.43, dtype=mx.bfloat16),
        "mtp.pre_fc_norm_embedding.weight": mx.full((4,), 0.52, dtype=mx.bfloat16),
        "mtp.pre_fc_norm_hidden.weight": mx.full((4,), 0.77, dtype=mx.bfloat16),
        "mtp.fc.weight": mx.ones((4, 8), dtype=mx.bfloat16),
    }


def test_existing_healthy_mtp_sidecar_is_not_overwritten(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    destination.mkdir()
    mx.save_safetensors(str(destination / "mtp.safetensors"), _healthy_sidecar_tensors())
    before = (destination / "mtp.safetensors").read_bytes()

    assert forge._ensure_mtp_sidecar(source, destination) is True
    assert (destination / "mtp.safetensors").read_bytes() == before


def test_existing_degenerate_mtp_sidecar_is_quarantined_and_rebuilt(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "destination"
    destination.mkdir()
    good = _healthy_sidecar_tensors()
    mx.save_safetensors(str(source / "mtp.safetensors"), good)
    corrupt = dict(good)
    corrupt["mtp.norm.weight"] = mx.zeros((4,), dtype=mx.bfloat16)
    mx.save_safetensors(str(destination / "mtp.safetensors"), corrupt)

    assert forge._ensure_mtp_sidecar(source, destination) is True

    quarantined = sorted(destination.glob("mtp.safetensors.corrupt-*"))
    assert quarantined, "corrupt sidecar was not quarantined"
    rebuilt = mx.load(str(destination / "mtp.safetensors"))
    assert float(rebuilt["mtp.norm.weight"].astype(mx.float32).mean().item()) == pytest.approx(
        2.43, abs=0.02
    )


def test_cast_sidecar_fp16_is_value_faithful_per_name(tmp_path):
    # Regression for the #176 corruption: same-shape 1-D norms with distinct
    # values must survive the fp16 cast under their own names. The old cast
    # saved lazily over the file it was reading and clobbered these payloads.
    sidecar = tmp_path / "mtp.safetensors"
    expected = {
        "mtp.layers.0.input_layernorm.weight": 1.30,
        "mtp.layers.0.post_attention_layernorm.weight": 1.39,
        "mtp.norm.weight": 3.58,
        "mtp.pre_fc_norm_embedding.weight": 0.59,
        "mtp.pre_fc_norm_hidden.weight": 0.75,
    }
    tensors = {key: mx.full((2560,), value, dtype=mx.bfloat16) for key, value in expected.items()}
    tensors["mtp.fc.weight"] = mx.full((64, 128), 0.007, dtype=mx.bfloat16)
    mx.save_safetensors(str(sidecar), tensors)

    forge._cast_sidecar_float_tensors_fp16(sidecar)

    reloaded = mx.load(str(sidecar))
    for key, value in expected.items():
        assert reloaded[key].dtype == mx.float16
        assert float(reloaded[key].astype(mx.float32).mean().item()) == pytest.approx(
            value, abs=0.01
        ), key
    assert float(reloaded["mtp.fc.weight"].astype(mx.float32).mean().item()) == pytest.approx(
        0.007, abs=0.001
    )
    assert not list(tmp_path.glob("*.fp16-tmp.safetensors"))


def test_embedded_bf16_mtp_extraction_does_not_require_torch(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    destination.mkdir()
    _write_json(source / "config.json", _mtp_config())
    _write_json(
        source / "model.safetensors.index.json",
        {
            "weight_map": {
                "mtp.fc.weight": "model.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model.safetensors",
            }
        },
    )
    mx.save_safetensors(
        str(source / "model.safetensors"),
        {
            "mtp.fc.weight": mx.ones((2, 2), dtype=mx.bfloat16),
            "model.layers.0.self_attn.q_proj.weight": mx.zeros((2, 2), dtype=mx.bfloat16),
        },
    )

    assert forge._ensure_mtp_sidecar(source, destination) is True

    tensors = mx.load(str(destination / "mtp.safetensors"))
    assert sorted(tensors) == ["mtp.fc.weight"]
    assert bool(mx.allclose(tensors["mtp.fc.weight"], mx.ones((2, 2), dtype=mx.bfloat16)).item())


def test_embedded_qwen_mtp_extraction_sanitizes_norm_weights(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    destination.mkdir()
    _write_json(source / "config.json", _mtp_config())
    _write_json(
        source / "model.safetensors.index.json",
        {
            "weight_map": {
                "mtp.fc.weight": "model.safetensors",
                "mtp.layers.0.input_layernorm.weight": "model.safetensors",
                "mtp.layers.0.self_attn.q_norm.weight": "model.safetensors",
                "mtp.norm.weight": "model.safetensors",
            }
        },
    )
    mx.save_safetensors(
        str(source / "model.safetensors"),
        {
            "mtp.fc.weight": mx.ones((2, 2), dtype=mx.bfloat16),
            "mtp.layers.0.input_layernorm.weight": mx.array([0.1, 0.2], dtype=mx.bfloat16),
            "mtp.layers.0.self_attn.q_norm.weight": mx.array([0.3, 0.4], dtype=mx.bfloat16),
            "mtp.norm.weight": mx.array([0.5, 0.6], dtype=mx.bfloat16),
        },
    )

    assert forge._ensure_mtp_sidecar(source, destination) is True

    tensors = mx.load(str(destination / "mtp.safetensors"))
    assert bool(mx.allclose(tensors["mtp.fc.weight"], mx.ones((2, 2), dtype=mx.bfloat16)).item())
    assert bool(
        mx.allclose(
            tensors["mtp.layers.0.input_layernorm.weight"].astype(mx.float32),
            mx.array([1.1, 1.2], dtype=mx.float32),
            atol=0.01,
        ).item()
    )
    assert bool(
        mx.allclose(
            tensors["mtp.layers.0.self_attn.q_norm.weight"].astype(mx.float32),
            mx.array([1.3, 1.4], dtype=mx.float32),
            atol=0.01,
        ).item()
    )
    assert bool(
        mx.allclose(
            tensors["mtp.norm.weight"].astype(mx.float32),
            mx.array([1.5, 1.6], dtype=mx.float32),
            atol=0.01,
        ).item()
    )


def test_mtp_sidecar_payload_audit_rejects_zero_projection_weights(tmp_path):
    mtp_path = tmp_path / "mtp.safetensors"
    mx.save_safetensors(
        str(mtp_path),
        {
            "mtp.fc.weight": mx.zeros((4, 4), dtype=mx.bfloat16),
            "mtp.layers.0.self_attn.q_proj.weight": mx.zeros((4, 4), dtype=mx.bfloat16),
            "mtp.layers.0.post_attention_layernorm.weight": mx.ones((4,), dtype=mx.bfloat16),
        },
    )

    audit = forge._audit_mtp_sidecar_payload(mtp_path)

    assert audit["passed"] is False
    assert "MTP sidecar has no nonzero projection or MLP payload tensors" in audit["problems"]


def test_mtp_sidecar_payload_audit_accepts_quantized_mtp_payload(tmp_path):
    mtp_path = tmp_path / "mtp.safetensors"
    mx.save_safetensors(
        str(mtp_path),
        {
            "mtp.layers.0.self_attn.q_proj.weight": mx.array([[1], [2], [3], [4]], dtype=mx.uint32),
            "mtp.layers.0.self_attn.q_proj.scales": mx.ones((4, 1), dtype=mx.float16),
            "mtp.layers.0.self_attn.q_proj.biases": mx.zeros((4, 1), dtype=mx.float16),
        },
    )

    audit = forge._audit_mtp_sidecar_payload(mtp_path)

    assert audit["passed"] is True
    assert audit["nonzero_payload_tensor_count"] == 1


def test_calibrate_sidecar_rejects_existing_zero_mtp_payload(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    run = tmp_path / "run"
    source.mkdir()
    destination.mkdir()
    run.mkdir()
    _write_json(destination / "config.json", _mtp_config())
    mx.save_safetensors(
        str(destination / "mtp.safetensors"),
        {
            "mtp.fc.weight": mx.zeros((4, 4), dtype=mx.bfloat16),
            "mtp.layers.0.self_attn.q_proj.weight": mx.zeros((4, 4), dtype=mx.bfloat16),
        },
    )

    with pytest.raises(forge.ForgeError, match="no nonzero projection"):
        forge._calibrate_sidecar(source, destination, recipe={"mtp_policy": "keep_bf16"}, run=run)


def test_publish_reads_one_token_line_and_keeps_artifacts_secret_free(tmp_path, monkeypatch):
    local = tmp_path / "model"
    _write_json(local / "mtplx_runtime.json", {"forge_provenance": {"forged_locally": True}})
    (local / "config.json").write_text("{}", encoding="utf-8")
    calls: list[tuple[str, str | None]] = []

    class FakeApi:
        def create_repo(self, **kwargs):
            calls.append(("create", kwargs.get("token")))

        def upload_folder(self, **kwargs):
            calls.append(("folder", kwargs.get("token")))
            return SimpleNamespace(oid="rev-folder")

        def upload_file(self, **kwargs):
            calls.append(("file", kwargs.get("token")))
            return SimpleNamespace(oid="rev-file")

        def model_info(self, repo_id, *, token=None):
            calls.append(("info", token))
            return SimpleNamespace(sha="rev-final")

    monkeypatch.setattr(forge, "_make_hf_api", lambda: FakeApi())
    monkeypatch.setattr("sys.stdin", io.StringIO("hf_secret\nsecond-line-ignored\n"))

    code = main(
        [
            "forge",
            "publish",
            "--path",
            str(local),
            "--repo",
            "owner/Fixture-MTPLX-Speed",
            "--visibility",
            "private",
            "--license",
            "apache-2.0",
            "--out",
            str(tmp_path / "publish"),
            "--run-id",
            "p1",
            "--token",
            "stdin",
        ]
    )

    publish_json = (tmp_path / "publish" / "p1" / "publish.json").read_text(
        encoding="utf-8"
    )
    runtime_json = (local / "mtplx_runtime.json").read_text(encoding="utf-8")

    assert code == 0
    assert calls and all(token == "hf_secret" for _, token in calls)
    assert "hf_secret" not in publish_json
    assert "hf_secret" not in runtime_json
    assert json.loads(runtime_json)["forge_provenance"]["published_to_hf"]["repo"] == "owner/Fixture-MTPLX-Speed"


def _tiny_vision_config() -> dict:
    return {
        "model_type": "qwen3_5",
        "depth": 1,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_heads": 2,
        "out_hidden_size": 4,
        "patch_size": 2,
        "spatial_merge_size": 2,
        "temporal_patch_size": 1,
        "in_channels": 3,
        "num_position_embeddings": 4,
        "deepstack_visual_indexes": [],
    }


def _write_multimodal_source(source: Path) -> dict[str, object]:
    """Synthetic multimodal checkpoint: language + model.visual.* weights.

    The vision weights are dumped from a real (miniature) tower so the
    graft's strict verification load has an exact key/shape match.
    """
    from mlx.utils import tree_flatten

    from mtplx.vision.qwen3_vl_tower import Qwen3VLVisionConfig, Qwen3VLVisionTower

    vision_config = _tiny_vision_config()
    tower = Qwen3VLVisionTower(Qwen3VLVisionConfig.from_dict(vision_config))
    tensors: dict[str, object] = {
        "model.visual." + name: value for name, value in tree_flatten(tower.parameters())
    }
    tensors["model.language_model.layers.0.self_attn.q_proj.weight"] = mx.zeros(
        (2, 2), dtype=mx.bfloat16
    )
    _write_json(
        source / "config.json",
        {
            "model_type": "qwen3_5",
            "vision_config": vision_config,
            "image_token_id": 248056,
            "video_token_id": 248057,
            "vision_start_token_id": 248053,
            "vision_end_token_id": 248054,
        },
    )
    _write_json(
        source / "model.safetensors.index.json",
        {"weight_map": {key: "model.safetensors" for key in tensors}},
    )
    mx.save_safetensors(str(source / "model.safetensors"), tensors)
    _write_json(source / "preprocessor_config.json", {"patch_size": 2, "merge_size": 2})
    _write_json(source / "video_preprocessor_config.json", {"patch_size": 2})
    return tensors


def _write_text_only_destination(destination: Path) -> None:
    _write_json(destination / "config.json", {"model_type": "qwen3_5"})
    _write_json(
        destination / "model.safetensors.index.json",
        {
            "metadata": {"total_size": 8},
            "weight_map": {
                "language_model.model.layers.0.self_attn.q_proj.weight": "model-00001-of-00001.safetensors"
            },
        },
    )
    mx.save_safetensors(
        str(destination / "model-00001-of-00001.safetensors"),
        {
            "language_model.model.layers.0.self_attn.q_proj.weight": mx.zeros(
                (2, 2), dtype=mx.bfloat16
            )
        },
    )


def test_forge_preserves_vision_tower(tmp_path):
    from mtplx.vision_graft import graft_vision_tower

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    destination.mkdir()
    source_tensors = _write_multimodal_source(source)
    _write_text_only_destination(destination)

    forge._ensure_vision_tower(source, destination)

    index = json.loads(
        (destination / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    vision_keys = [
        key for key in index["weight_map"] if key.startswith("vision_tower.")
    ]
    expected = sum(1 for key in source_tensors if key.startswith("model.visual."))
    assert len(vision_keys) == expected
    assert all(
        index["weight_map"][key] == "model-vision.safetensors" for key in vision_keys
    )
    assert index["metadata"]["total_size"] > 8

    config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
    assert isinstance(config.get("vision_config"), dict)
    assert config["image_token_id"] == 248056
    assert (destination / "model-vision.safetensors").exists()
    assert (destination / "preprocessor_config.json").exists()
    assert (destination / "video_preprocessor_config.json").exists()

    grafted = mx.load(str(destination / "model-vision.safetensors"))
    assert sorted(grafted) == sorted(
        "vision_tower." + key[len("model.visual.") :]
        for key in source_tensors
        if key.startswith("model.visual.")
    )

    # The fail-closed validator must accept the repaired artifact.
    forge._validate_vision_payload(source, destination)

    # Provenance stamp resolves the grafted tower.
    stamp = forge._vision_metadata_stamp(destination)
    assert stamp == {
        "tensor_count": expected,
        "prefix": "vision_tower.",
        "shards": ["model-vision.safetensors"],
    }

    # Idempotent: a second pass must be a no-op.
    report = graft_vision_tower(source, destination)
    assert report["status"] == "already-present"


def test_forge_vision_noop_for_text_only_source(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    destination.mkdir()
    _write_json(source / "config.json", {"model_type": "qwen3_5"})
    _write_json(
        source / "model.safetensors.index.json",
        {"weight_map": {"model.layers.0.self_attn.q_proj.weight": "model.safetensors"}},
    )
    _write_text_only_destination(destination)
    index_before = (destination / "model.safetensors.index.json").read_text(
        encoding="utf-8"
    )
    config_before = (destination / "config.json").read_text(encoding="utf-8")

    forge._ensure_vision_tower(source, destination)
    forge._validate_vision_payload(source, destination)

    assert not (destination / "model-vision.safetensors").exists()
    assert (destination / "model.safetensors.index.json").read_text(
        encoding="utf-8"
    ) == index_before
    assert (destination / "config.json").read_text(encoding="utf-8") == config_before
    assert forge._vision_metadata_stamp(destination) is None


def test_forge_vision_validation_fails_closed_on_blind_artifact(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    destination.mkdir()
    # Source declares vision_config, but its index carries no vision tensors,
    # so the graft cannot restore anything: the build must fail, not ship blind.
    _write_json(
        source / "config.json",
        {"model_type": "qwen3_5", "vision_config": _tiny_vision_config()},
    )
    _write_json(
        source / "model.safetensors.index.json",
        {"weight_map": {"model.layers.0.self_attn.q_proj.weight": "model.safetensors"}},
    )
    _write_text_only_destination(destination)

    forge._ensure_vision_tower(source, destination)
    with pytest.raises(forge.ForgeError, match="vision"):
        forge._validate_vision_payload(source, destination)
