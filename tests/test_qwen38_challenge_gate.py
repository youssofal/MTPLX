from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.qwen38_challenge import (
    QWEN38_LOW_ADAPTIVE_ROUTE,
    QWEN38_LOW_FIXED_ROUTE,
    QWEN38_XHIGH_ADAPTIVE_ROUTE,
    QWEN38_XHIGH_FIXED_ROUTE,
)

SCRIPT = Path(__file__).parents[1] / "scripts/qwen38_challenge_port_gate.py"
ISOLATED_SCRIPT = (
    Path(__file__).parents[1] / "scripts/qwen38_challenge_port_isolated_gate.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("qwen38_challenge_port_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _isolated_module():
    spec = importlib.util.spec_from_file_location(
        "qwen38_challenge_port_isolated_gate", ISOLATED_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_native_route_contains_only_retained_features() -> None:
    gate = _module()
    assert gate.FIXED_NATIVE_ROUTE == "control"
    assert gate.ADAPTIVE_NATIVE_ROUTE == "r11_position_ema"
    assert "r70_qmv_sumtable" not in gate.ALLOWED_ROUTE_FEATURES
    assert "source_proposal" not in gate.ALLOWED_ROUTE_FEATURES


def test_low_and_xhigh_benchmark_routes_use_independently_promoted_stacks() -> None:
    gate = _module()

    expected_low_shared = (
        "r20_kv_only_history+r53_command_buffers+r08_device_draft+"
        "r10_compact_vocab+r21_qk_rms_rope+r24_eval_ladder+"
        "r26_prefill_ladder_3"
    )
    expected_xhigh_shared = (
        "r20_kv_only_history+r24_eval_ladder+r26_prefill_ladder_3+"
        "r50_wired_residency+r53_command_buffers"
    )
    assert gate.LOW_ADAPTIVE_SHARED_ROUTE == expected_low_shared
    assert gate.LOW_FIXED_NATIVE_ROUTE == expected_low_shared
    assert gate.LOW_ADAPTIVE_NATIVE_ROUTE == (
        expected_low_shared + "+r11_position_ema"
    )
    assert gate.LOW_Q4_ADAPTIVE_NATIVE_ROUTE == (
        expected_low_shared + "+r11_position_ema+r17_q4_mtp_block"
    )
    assert gate.XHIGH_ADAPTIVE_SHARED_ROUTE == expected_xhigh_shared
    assert gate.XHIGH_FIXED_NATIVE_ROUTE == expected_xhigh_shared
    assert gate.XHIGH_ADAPTIVE_NATIVE_ROUTE == (
        expected_xhigh_shared + "+r11_position_ema"
    )
    assert gate.XHIGH_Q4_ADAPTIVE_NATIVE_ROUTE == (
        expected_xhigh_shared + "+r11_position_ema+r17_q4_mtp_block"
    )
    assert QWEN38_LOW_FIXED_ROUTE == gate.LOW_FIXED_NATIVE_ROUTE
    assert QWEN38_LOW_ADAPTIVE_ROUTE == gate.LOW_Q4_ADAPTIVE_NATIVE_ROUTE
    assert QWEN38_XHIGH_FIXED_ROUTE == gate.XHIGH_FIXED_NATIVE_ROUTE
    assert QWEN38_XHIGH_ADAPTIVE_ROUTE == gate.XHIGH_ADAPTIVE_NATIVE_ROUTE

    low_bf16 = gate._route_execution_options(gate.LOW_ADAPTIVE_NATIVE_ROUTE)
    low_q4 = gate._route_execution_options(gate.LOW_Q4_ADAPTIVE_NATIVE_ROUTE)
    xhigh_bf16 = gate._route_execution_options(gate.XHIGH_ADAPTIVE_NATIVE_ROUTE)
    xhigh_q4 = gate._route_execution_options(gate.XHIGH_Q4_ADAPTIVE_NATIVE_ROUTE)
    assert low_bf16["mtp_block_variant"] is None
    assert low_q4["mtp_block_variant"] == "r17"
    assert low_bf16["adaptive_policy"] == "position_ema"
    assert low_q4["adaptive_policy"] == "position_ema"
    assert low_bf16["source_rows"] == (8, 10, 20, 21, 24, 26, 53, 11)
    assert low_q4["source_rows"] == (8, 10, 17, 20, 21, 24, 26, 53, 11)
    assert low_bf16["draft_core"] == low_q4["draft_core"] == "device"
    assert xhigh_bf16["mtp_block_variant"] is None
    assert xhigh_q4["mtp_block_variant"] == "r17"
    assert xhigh_bf16["source_rows"] == (20, 24, 26, 50, 53, 11)
    assert xhigh_q4["source_rows"] == (17, 20, 24, 26, 50, 53, 11)
    assert xhigh_bf16["draft_core"] == xhigh_q4["draft_core"] == "stock"


def test_full_adaptive_benchmark_routes_include_measured_command_buffer_profile() -> None:
    gate = _module()

    for route in (
        gate.LOW_ADAPTIVE_NATIVE_ROUTE,
        gate.LOW_Q4_ADAPTIVE_NATIVE_ROUTE,
        gate.XHIGH_ADAPTIVE_NATIVE_ROUTE,
        gate.XHIGH_Q4_ADAPTIVE_NATIVE_ROUTE,
    ):
        assert "r53_command_buffers" in route.split("+")
        assert gate._route_execution_options(route)["row53_command_buffers"] is True


def test_row53_route_rejects_missing_process_latched_environment() -> None:
    gate = _module()

    with pytest.raises(RuntimeError, match="row 53 process contract"):
        gate._validate_process_latched_route(
            gate._route_execution_options("r53_command_buffers"),
            environment={},
        )

    gate._validate_process_latched_route(
        gate._route_execution_options("r53_command_buffers"),
        environment={
            "MLX_MAX_MB_PER_BUFFER": "512",
            "MLX_MAX_OPS_PER_BUFFER": "50",
        },
    )


def test_promotion_accepts_an_already_validated_atomic_bundle() -> None:
    gate = _module()
    control = "r20_kv_only_history+r53_command_buffers"
    candidate = control + "+r08_device_draft+r10_compact_vocab"
    validated = SimpleNamespace(
        control_features=gate._validate_route_id(control),
        candidate_features_set=gate._validate_route_id(candidate),
    )

    result = gate._promotion_decision(
        order=[control, candidate, candidate, control],
        control_id=control,
        candidate_id=candidate,
        improvement_pct=1.2,
        correctness={"passed": True},
        source_status=[],
        validated_route_delta=validated,
    )

    assert result == {
        "passed": True,
        "threshold_pct": gate.PROMOTION_THRESHOLD_PCT,
        "errors": [],
    }


def test_native_gate_has_no_rejected_source_artifact_flag() -> None:
    source = ISOLATED_SCRIPT.read_text(encoding="utf-8")
    assert "--source-artifact" not in source
    assert "source_artifact" not in source
    assert "--comparison-kind" not in source


def _receipt_run(
    route: str,
    *,
    overrides: dict[str, dict] | None = None,
    prompt_tokens: int = 16_384,
    policy_events: list[dict] | None = None,
    adaptive_policy_receipt: dict | None = None,
) -> dict:
    receipt_keys = {
        "r10_compact_vocab": "r10_compact_vocab",
        "r17_q4_mtp_block": "r17_q4_mtp_block",
        "r18_gdn_decay_memo": "r18_gdn_decay_memo",
        "r21_qk_rms_rope": "r21_qk_rms_rope",
        "r24_eval_ladder": "r24_eval_ladder",
        "r26_prefill_ladder_3": "r26_prefill_ladder_3",
        "r28_q4_mtp_block": "r28_q4_mtp_block",
        "r36_qkv_islands": "r36_qkv_islands",
        "r48_boundary_fused": "r48_boundary_fused",
        "r50_wired_residency": "r50_wired_residency",
        "r53_command_buffers": "r53_command_buffers",
        "r61_dual_norm_concat": "dual_norm",
        "r63_q8_embedding_dual_norm": "r63_q8_embedding_dual_norm",
    }
    receipts = {
        receipt_key: {"installed": True, "active": True, "active_modules": 1}
        for feature, receipt_key in receipt_keys.items()
        if feature in route.split("+")
    }
    if "r17_q4_mtp_block" in route:
        receipts["r17_q4_mtp_block"].update(
            {"variant": "r17", "bits": 4, "group_size": 64}
        )
    if "r50_wired_residency" in route:
        receipts["r50_wired_residency"].update(
            {"target_limit_bytes": 1024, "active_memory_bytes": 512}
        )
    if "r53_command_buffers" in route:
        receipts["r53_command_buffers"].update(
            {"max_mb_per_buffer": 512, "max_ops_per_buffer": 50}
        )
    receipts.update(overrides or {})
    return {
        "route_id": route,
        "draft_core": "device",
        "drafted_by_depth": [1, 1, 1],
        "adaptive_policy_events": policy_events or [],
        "adaptive_policy_receipt": adaptive_policy_receipt,
        "feature_receipt": receipts,
        "history_route_receipt": {
            "route_id": (
                "kv_only_history" if prompt_tokens >= 16_384 else "stock_history"
            ),
            "prompt_tokens": prompt_tokens,
            "row20_engaged": prompt_tokens >= 16_384,
            "reason": (
                "min_context_satisfied"
                if prompt_tokens >= 16_384
                else "below_min_context"
            ),
        },
    }


def test_generation_metrics_include_prefill_decode_and_peak_memory() -> None:
    gate = _module()
    stats = SimpleNamespace(
        new_prefill_tokens=512,
        generated_tokens=1024,
        prompt_target_prefill_time_s=0.25,
        prompt_target_prefill_tok_s=2048.0,
        prompt_mtp_history_time_s=0.125,
        prompt_mtp_history_tok_s=4096.0,
        decode_elapsed_s=25.6,
        decode_tok_s=40.0,
        draft_time_s=2.0,
        drafted_tokens=300,
        peak_memory_bytes=24 * 2**30,
        capture_commit_time_s=0.125,
        speculative_depth=3,
        requested_speculative_depth=3,
        verify_calls=80,
        bonus_tokens=60,
        correction_tokens=20,
        context_copy_active=False,
        context_copy_rounds=0,
        context_copy_drafted_tokens=0,
        context_copy_accepted_tokens=0,
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
        events=[
            {"capture_repair": "captured_prefix_commit"},
            {"capture_repair": "captured_prefix_pending_correction"},
            {"capture_repair": "standard_reforward"},
        ],
    )

    metrics = gate._generation_metrics(
        stats,
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
    )

    assert metrics == {
        "prefill_tokens": 512,
        "prefill_time_s": 0.25,
        "prefill_tok_s": 2048.0,
        "target_prefill_time_s": 0.25,
        "target_prefill_tok_s": 2048.0,
        "mtp_history_tokens": 512,
        "mtp_history_time_s": 0.125,
        "mtp_history_tok_s": 4096.0,
        "mtp_decode_tokens": 300,
        "mtp_decode_time_s": 2.0,
        "mtp_decode_tok_s": 150.0,
        "decode_elapsed_s": 25.6,
        "decode_tok_s": 40.0,
        "peak_memory_bytes": 24 * 2**30,
        "peak_memory_gib": 24.0,
        "capture_commit_time_s": 0.125,
        "capture_commit_events": 2,
        "verify_strategy": "capture_commit",
        "verify_core": "linear-gdn-from-conv-tape",
        "speculative_depth": 3,
        "requested_speculative_depth": 3,
        "verify_calls": 80,
        "bonus_tokens": 60,
        "correction_tokens": 20,
        "context_copy_active": False,
        "context_copy_rounds": 0,
        "context_copy_drafted_tokens": 0,
        "context_copy_accepted_tokens": 0,
    }


def test_model_artifact_hashes_cover_mtp_and_every_indexed_shard(tmp_path) -> None:
    gate = _module()
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "mtplx_runtime.json").write_bytes(b"runtime-manifest")
    (tmp_path / "mtp.safetensors").write_bytes(b"mtp-tensors")
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"shard-one")
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"shard-two")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                    "c": "model-00001-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    hashes = gate._model_artifact_hashes(tmp_path)

    assert hashes["mtp.safetensors"] == hashlib.sha256(b"mtp-tensors").hexdigest()
    assert hashes["mtplx_runtime.json"] == hashlib.sha256(
        b"runtime-manifest"
    ).hexdigest()
    assert hashes["model-00001-of-00002.safetensors"] == hashlib.sha256(
        b"shard-one"
    ).hexdigest()
    assert hashes["model-00002-of-00002.safetensors"] == hashlib.sha256(
        b"shard-two"
    ).hexdigest()


def test_model_artifact_hashes_reject_a_missing_indexed_shard(tmp_path) -> None:
    gate = _module()
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "mtp.safetensors").write_bytes(b"mtp-tensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "missing.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="referenced model shard is missing"):
        gate._model_artifact_hashes(tmp_path)


def test_attested_artifact_hashes_require_authoritative_tensor_keys(tmp_path) -> None:
    gate = _module()
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "mtp.safetensors").write_bytes(b"mtp")
    (tmp_path / "model.safetensors").write_bytes(b"model")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model.safetensors"}}),
        encoding="utf-8",
    )
    hashes = gate._model_artifact_hashes(tmp_path)
    hashes.pop("mtp.safetensors")

    with pytest.raises(RuntimeError, match="missing mtp.safetensors"):
        gate._attested_model_artifact_hashes(
            tmp_path,
            guarded_by_parent=True,
            environment={
                gate.MODEL_ARTIFACT_HASHES_ENV: json.dumps(hashes),
            },
        )


def test_attested_children_reuse_parent_tensor_hashes_without_rehashing(
    tmp_path, monkeypatch
) -> None:
    gate = _module()
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "mtp.safetensors").write_bytes(b"mtp")
    (tmp_path / "model.safetensors").write_bytes(b"model")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model.safetensors"}}),
        encoding="utf-8",
    )
    hashes = gate._model_artifact_hashes(tmp_path)
    monkeypatch.setattr(
        gate,
        "_sha256_file",
        lambda _path: pytest.fail("attested child rehashed tensor bytes"),
    )

    observed = gate._attested_model_artifact_hashes(
        tmp_path,
        guarded_by_parent=True,
        environment={gate.MODEL_ARTIFACT_HASHES_ENV: json.dumps(hashes)},
    )

    assert observed == hashes


def test_exact_campaign_software_requires_mlx_and_metal_0322() -> None:
    gate = _module()

    versions = gate._validated_software_versions(
        version_fn=lambda name: {"mlx": "0.32.2", "mlx-metal": "0.32.2"}[name]
    )

    assert versions == {"mlx": "0.32.2", "mlx_metal": "0.32.2"}
    with pytest.raises(RuntimeError, match="mlx-metal==0.32.2"):
        gate._validated_software_versions(
            version_fn=lambda name: {"mlx": "0.32.2", "mlx-metal": "0.32.1"}[
                name
            ]
        )


def test_phase_summary_keeps_prefill_history_and_decode_sortable() -> None:
    gate = _module()
    arms = []
    for route, scale in (("control", 1.0), ("candidate", 0.9)) * 2:
        arms.append(
            {
                "route_id": route,
                "wall_s": 40.0 * scale,
                "target_prefill_time_s": 4.0 * scale,
                "target_prefill_tok_s": 256.0 / scale,
                "mtp_history_tokens": 1024,
                "mtp_history_time_s": 2.0 * scale,
                "mtp_history_tok_s": 512.0 / scale,
                "mtp_decode_tokens": 1024,
                "mtp_decode_time_s": 34.0 * scale,
                "mtp_decode_tok_s": 30.0 / scale,
                "decode_elapsed_s": 34.0 * scale,
                "decode_tok_s": 30.0 / scale,
            }
        )

    summary = gate._phase_summary(
        arms, control_id="control", candidate_id="candidate"
    )

    assert list(summary["time_improvement_pct"]) == [
        "wall",
        "target_prefill",
        "mtp_history",
        "mtp_decode",
        "decode",
    ]
    assert list(summary["throughput_improvement_pct"]) == [
        "target_prefill",
        "mtp_history",
        "mtp_decode",
        "decode",
    ]
    assert summary["mean_time_s"]["candidate"]["mtp_history"] == pytest.approx(
        1.8
    )
    assert summary["mean_throughput_tok_s"]["control"]["mtp_decode"] == 30.0
    assert summary["time_improvement_pct"]["mtp_decode"] > 0.05
    assert summary["throughput_improvement_pct"]["mtp_decode"] > 0.05


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf"), 0.0])
def test_phase_summary_rejects_invalid_elapsed_values(invalid: float) -> None:
    gate = _module()
    arms = [
        {
            "route_id": route,
            "wall_s": invalid if index == 0 else 10.0,
            "target_prefill_time_s": 1.0,
            "target_prefill_tok_s": 1024.0,
            "mtp_history_time_s": 1.0,
            "mtp_history_tok_s": 1024.0,
            "mtp_decode_time_s": 1.0,
            "mtp_decode_tok_s": 100.0,
            "decode_elapsed_s": 1.0,
            "decode_tok_s": 100.0,
        }
        for index, route in enumerate(("control", "candidate"))
    ]

    with pytest.raises(ValueError, match="finite and positive"):
        gate._phase_summary(arms, control_id="control", candidate_id="candidate")


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_phase_summary_rejects_nonfinite_throughput(invalid: float) -> None:
    gate = _module()
    arms = [
        {
            "route_id": route,
            "wall_s": 10.0,
            "target_prefill_time_s": 1.0,
            "target_prefill_tok_s": 1024.0,
            "mtp_history_time_s": 1.0,
            "mtp_history_tok_s": 1024.0,
            "mtp_decode_time_s": 1.0,
            "mtp_decode_tok_s": invalid if index == 0 else 100.0,
            "decode_elapsed_s": 1.0,
            "decode_tok_s": 100.0,
        }
        for index, route in enumerate(("control", "candidate"))
    ]

    with pytest.raises(ValueError, match="finite and nonnegative"):
        gate._phase_summary(arms, control_id="control", candidate_id="candidate")


def test_optimized_speed_stack_applies_turbo_before_load_and_installs_q4_head() -> None:
    gate = _module()
    calls: list[tuple[object, ...]] = []
    runtime = SimpleNamespace(model=object())
    contract = {
        "recommended_draft_sampler": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
        }
    }

    def apply_profile(name, *, runtime_env_overrides):
        calls.append(("profile", name, runtime_env_overrides))

    def load_runtime(path, *, mtp):
        calls.append(("load", path, mtp))
        return runtime

    def install_draft_head(loaded, *, bits, group_size, mode):
        calls.append(("draft_head", loaded, bits, group_size, mode))
        return {"installed": True}

    loaded, stack = gate._load_optimized_speed_stack(
        Path("/model"),
        contract,
        apply_profile_env_fn=apply_profile,
        load_runtime_fn=load_runtime,
        install_draft_head_fn=install_draft_head,
    )

    assert loaded is runtime
    assert calls == [
        ("profile", "turbo", {}),
        ("load", Path("/model"), True),
        ("draft_head", runtime, 4, 64, "affine"),
    ]
    assert stack["profile"] == "turbo"
    assert stack["runtime_profile"] == "native_mtp_turbo"
    assert stack["draft_lm_head"] == {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
    }
    assert stack["draft_sampler"] == contract["recommended_draft_sampler"]
    assert stack["verify_strategy"] == "capture_commit"
    assert stack["verify_core"] == "linear-gdn-from-conv-tape"
    assert stack["base_stack"] == {
        "id": "upstream_main_qwen38_optimized_speed",
        "commit": "bd4421567f9e16ce957c6ef97708b072dcd73937",
        "internal_control_route": "control",
    }


def test_expand_prompt_hits_exact_token_budget() -> None:
    gate = _module()

    class CharacterTokenizer:
        @staticmethod
        def encode(text):
            return [ord(character) for character in text]

        @staticmethod
        def decode(tokens):
            return "".join(chr(token) for token in tokens)

    prompt, token_ids = gate._expand_prompt_to_token_count(
        CharacterTokenizer(),
        "ab",
        9,
    )

    assert prompt == "ab\nab\nab\n"
    assert token_ids == [ord(character) for character in prompt]


def test_context_prompt_preserves_one_tail_instruction_at_exact_budget() -> None:
    gate = _module()

    class CharacterTokenizer:
        @staticmethod
        def encode(text):
            return [ord(character) for character in text]

        @staticmethod
        def decode(tokens):
            return "".join(chr(token) for token in tokens)

    prompt, token_ids = gate._context_prompt_to_token_count(
        CharacterTokenizer(),
        context="0123456789",
        instruction="WRITE-LONG",
        target_tokens=32,
    )

    assert len(token_ids) == 32
    assert prompt.endswith("WRITE-LONG")
    assert prompt.count("WRITE-LONG") == 1


def test_gate_defaults_to_the_16k_generation_python_prompt(monkeypatch) -> None:
    gate = _module()
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output", "/tmp/out.json"])

    args = gate._parse_args()

    assert args.prompt_tokens == 16_384
    assert args.context_file == gate.ROOT / "mtplx/generation.py"
    assert args.max_tokens == 1_024


def test_correctness_requires_exact_cross_route_replay() -> None:
    gate = _module()
    arms = [
        {
            "route_id": "control",
            "generated_tokens": 1024,
            "token_hash": "control-hash",
            "attempted_depth_schedule": [3, 3],
            "accepted_depth_schedule": [2, 1],
        },
        {
            "route_id": "kv_only_history",
            "generated_tokens": 1024,
            "token_hash": "control-hash",
            "attempted_depth_schedule": [3, 3],
            "accepted_depth_schedule": [2, 1],
        },
        {
            "route_id": "kv_only_history",
            "generated_tokens": 1024,
            "token_hash": "control-hash",
            "attempted_depth_schedule": [3, 3],
            "accepted_depth_schedule": [2, 1],
        },
        {
            "route_id": "control",
            "generated_tokens": 1024,
            "token_hash": "control-hash",
            "attempted_depth_schedule": [3, 3],
            "accepted_depth_schedule": [2, 1],
        },
    ]

    correctness = gate._correctness_summary(
        arms,
        route_ids=["control", "kv_only_history"],
        max_tokens=1024,
    )

    assert correctness["passed"] is True
    assert correctness["full_output"] is True
    assert correctness["cross_route_token_exact"] is True
    assert correctness["per_route_deterministic"] == {
        "control": True,
        "kv_only_history": True,
    }


def test_deterministic_cross_route_drift_is_recorded_without_rejection() -> None:
    gate = _module()
    arms = [
        {
            "route_id": "control",
            "generated_tokens": 1024,
            "token_hash": "control-hash",
            "attempted_depth_schedule": [3, 3],
            "accepted_depth_schedule": [2, 1],
        },
        {
            "route_id": "kv_only_history",
            "generated_tokens": 1024,
            "token_hash": "candidate-hash",
            "attempted_depth_schedule": [3, 2],
            "accepted_depth_schedule": [1, 2],
        },
        {
            "route_id": "kv_only_history",
            "generated_tokens": 1024,
            "token_hash": "candidate-hash",
            "attempted_depth_schedule": [3, 2],
            "accepted_depth_schedule": [1, 2],
        },
        {
            "route_id": "control",
            "generated_tokens": 1024,
            "token_hash": "control-hash",
            "attempted_depth_schedule": [3, 3],
            "accepted_depth_schedule": [2, 1],
        },
    ]
    correctness = gate._correctness_summary(
        arms,
        route_ids=["control", "kv_only_history"],
        max_tokens=1024,
    )

    assert correctness["passed"] is True
    assert correctness["mode"] == "deterministic_drift"
    assert correctness["cross_route_token_exact"] is False
    assert correctness["cross_route_schedule_exact"] is False


def test_empty_event_schedules_do_not_create_vacuous_exactness() -> None:
    gate = _module()
    arms = [
        {
            "route_id": "control",
            "generated_tokens": 1024,
            "token_hash": "same",
            "attempted_depth_schedule": [],
            "accepted_depth_schedule": [],
            "drafted_by_depth": [10, 9, 8],
            "accepted_by_depth": [9, 8, 7],
        },
        {
            "route_id": "candidate",
            "generated_tokens": 1024,
            "token_hash": "same",
            "attempted_depth_schedule": [],
            "accepted_depth_schedule": [],
            "drafted_by_depth": [10, 9, 8],
            "accepted_by_depth": [9, 8, 6],
        },
    ]

    correctness = gate._correctness_summary(
        arms,
        route_ids=["control", "candidate"],
        max_tokens=1024,
    )

    assert correctness["cross_route_schedule_exact"] is False
    assert correctness["schedule_capture"] == "depth_histograms"


def test_route_validation_accepts_the_single_cumulative_winner_stack() -> None:
    gate = _module()

    assert gate._validate_route_id("control") == {"control"}
    assert gate._validate_route_id("kv_only_history") == {"r20_kv_only_history"}
    assert gate._validate_route_id("dual_norm") == {"r61_dual_norm_concat"}
    assert gate._validate_route_id("r08_device_draft") == {"r08_device_draft"}
    assert gate._validate_route_id(
        "r08_device_draft+r10_compact_vocab"
    ) == {"r08_device_draft", "r10_compact_vocab"}
    assert gate._validate_route_id(
        "r08_device_draft+r10_compact_vocab+r17_q4_mtp_block"
    ) == {"r08_device_draft", "r10_compact_vocab", "r17_q4_mtp_block"}
    with pytest.raises(ValueError, match="incompatible native-MTP alternatives"):
        gate._validate_route_id(
            "r08_device_draft+r10_compact_vocab+r17_q4_mtp_block+"
            "r28_q4_mtp_block"
        )
    with pytest.raises(ValueError, match="incompatible native-MTP alternatives"):
        gate._validate_route_id(
            "r08_device_draft+r10_compact_vocab+r17_q4_mtp_block+"
            "r36_qkv_islands"
        )
    assert gate._validate_route_id(
        "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo"
    ) == {"r08_device_draft", "r10_compact_vocab", "r18_gdn_decay_memo"}
    assert gate._validate_route_id(
        "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo+"
        "r20_kv_only_history"
    ) == {
        "r08_device_draft",
        "r10_compact_vocab",
        "r18_gdn_decay_memo",
        "r20_kv_only_history",
    }
    assert gate._validate_route_id(
        "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo+"
        "r20_kv_only_history+r24_eval_ladder"
    ) == {
        "r08_device_draft",
        "r10_compact_vocab",
        "r18_gdn_decay_memo",
        "r20_kv_only_history",
        "r24_eval_ladder",
    }
    assert gate._validate_route_id(
        "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo+"
        "r20_kv_only_history+r24_eval_ladder+r26_prefill_ladder_3"
    ) == {
        "r08_device_draft",
        "r10_compact_vocab",
        "r18_gdn_decay_memo",
        "r20_kv_only_history",
        "r24_eval_ladder",
        "r26_prefill_ladder_3",
    }
    assert gate._validate_route_id(
        "r08_device_draft+r61_dual_norm_concat"
    ) == {"r08_device_draft", "r61_dual_norm_concat"}
    assert gate._validate_route_id(
        "r08_device_draft+r48_boundary_fused"
    ) == {"r08_device_draft", "r48_boundary_fused"}
    assert gate._validate_route_id(
        "r08_device_draft+r50_wired_residency"
    ) == {"r08_device_draft", "r50_wired_residency"}
    assert gate._validate_route_id(
        "r08_device_draft+r63_q8_embedding_dual_norm"
    ) == {"r08_device_draft", "r63_q8_embedding_dual_norm"}
    with pytest.raises(ValueError, match="removed native-MTP family"):
        gate._validate_route_id("kv_only_history+dual_norm+source_proposal")
    with pytest.raises(ValueError, match="removed native-MTP family"):
        gate._validate_route_id("r08_device_draft+r70_qmv_sumtable")
    with pytest.raises(ValueError, match="unknown route feature"):
        gate._validate_route_id("kv_only_history+dual_norm+qmv_final")
    with pytest.raises(ValueError, match="unknown route feature"):
        gate._validate_route_id("packed_qkv")
    with pytest.raises(ValueError, match="unknown route feature"):
        gate._validate_route_id("gdn_projection_pairs")


def test_route_validation_rejects_control_combinations() -> None:
    gate = _module()

    with pytest.raises(ValueError, match="control cannot be combined"):
        gate._validate_route_id("control+dual_norm")

    with pytest.raises(ValueError, match="incompatible native-MTP alternatives"):
        gate._validate_route_id(
            "r61_dual_norm_concat+r63_q8_embedding_dual_norm"
        )

    with pytest.raises(ValueError, match="duplicate raw route feature"):
        gate._validate_route_id("r08_device_draft+r08_device_draft")
    with pytest.raises(ValueError, match="duplicate canonical route feature"):
        gate._validate_route_id("dual_norm+r61_dual_norm_concat")
    with pytest.raises(ValueError, match="incompatible native-MTP alternatives"):
        gate._validate_route_id("dual_norm+r63_q8_embedding_dual_norm")


def test_row_8_adapts_device_resident_draft_chaining_to_the_fixed_d3_route() -> None:
    gate = _module()

    control = gate._route_execution_options("control")
    row_8 = gate._route_execution_options("r08_device_draft")

    assert control["draft_core"] == "stock"
    assert row_8 == {
        "cache_route": "control",
        "adaptive_policy": "none",
        "speculative_depth": 3,
        "adaptive_depth_cap": 0,
        "dual_norm": False,
        "row10_compact_vocab": False,
        "mtp_block_variant": None,
        "row18_gdn_decay_memo": False,
        "row21_qk_rms_rope": False,
        "row24_eval_ladder": False,
        "row26_prefill_ladder_3": False,
        "row48_boundary_fused": False,
        "row50_wired_residency": False,
        "row53_command_buffers": False,
        "row63_q8_embedding_dual_norm": False,
        "draft_core": "device",
        "source_rows": (8,),
    }


def test_row_10_extends_retained_row_8_with_compact_proposal_vocabulary() -> None:
    gate = _module()

    row_10 = gate._route_execution_options(
        "r08_device_draft+r10_compact_vocab"
    )

    assert row_10["draft_core"] == "device"
    assert row_10["row10_compact_vocab"] is True
    assert row_10["source_rows"] == (8, 10)


def test_row11_clamps_position_ema_to_real_native_depth_three() -> None:
    gate = _module()

    options = gate._route_execution_options(gate.ADAPTIVE_NATIVE_ROUTE)

    assert options["adaptive_policy"] == "position_ema"
    assert options["speculative_depth"] == 3
    assert options["adaptive_depth_cap"] == 3
    assert options["source_rows"][-1] == 11


def test_row11_promotion_requires_position_ema_policy_events() -> None:
    gate = _module()
    route = "r08_device_draft+r10_compact_vocab+r11_position_ema"
    zero = _receipt_run(route)
    engaged = _receipt_run(
        route,
        policy_events=[
            {"kind": "position_ema", "attempted_depth": 3, "next_depth": 3}
        ],
    )

    assert gate._candidate_engagement_errors(route, [], [zero]) == [
        "row 11 position-EMA adaptive policy did not execute"
    ]
    assert gate._candidate_engagement_errors(route, [], [engaged, zero]) == [
        "row 11 position-EMA adaptive policy did not execute"
    ]
    assert gate._candidate_engagement_errors(route, [zero], [engaged, engaged]) == []


def test_row11_promotion_accepts_postrun_policy_state_when_events_are_dropped() -> None:
    gate = _module()
    route = "r08_device_draft+r10_compact_vocab+r11_position_ema"
    engaged = _receipt_run(
        route,
        adaptive_policy_receipt={
            "kind": "position_ema",
            "executed": True,
            "initial_accept_ema": [0.85, 0.833, 0.81634],
            "final_accept_ema": [0.9, 0.8, 0.75],
            "final_depth": 2,
        },
    )

    assert gate._candidate_engagement_errors(route, [], [engaged, engaged]) == []


def test_row_18_decay_memo_extends_rows_8_and_10() -> None:
    gate = _module()
    row_18 = gate._route_execution_options(
        "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo"
    )

    assert row_18["draft_core"] == "device"
    assert row_18["row18_gdn_decay_memo"] is True
    assert row_18["source_rows"] == (8, 10, 18)


def test_row18_promotion_requires_memoized_decay_engagement() -> None:
    gate = _module()
    route = "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo"
    zero = _receipt_run(
        route, overrides={"r18_gdn_decay_memo": {"active_modules": 0}}
    )
    engaged = _receipt_run(route)

    assert gate._candidate_engagement_errors(route, [], [zero]) == [
        "r18_gdn_decay_memo construction receipt is not active"
    ]
    assert gate._candidate_engagement_errors(route, [zero], [engaged, engaged]) == []


def test_candidate_engagement_rejects_one_inactive_timed_arm() -> None:
    gate = _module()
    route = "r08_device_draft+r10_compact_vocab"
    engaged = _receipt_run(route)
    inactive = _receipt_run(
        route, overrides={"r10_compact_vocab": {"active": False}}
    )

    assert gate._candidate_engagement_errors(route, [], [engaged, inactive]) == [
        "r10_compact_vocab construction receipt is not active"
    ]


def test_row_20_kv_only_history_extends_rows_8_10_and_18() -> None:
    gate = _module()
    route = (
        "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo+"
        "r20_kv_only_history"
    )

    options = gate._route_execution_options(route)

    assert options["cache_route"] == "kv_only_history"
    assert options["source_rows"] == (8, 10, 18, 20)


def test_row20_promotion_requires_kv_only_history_engagement() -> None:
    gate = _module()
    route = (
        "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo+"
        "r20_kv_only_history"
    )
    zero = _receipt_run(route)
    zero["history_route_receipt"]["row20_engaged"] = False
    engaged = _receipt_run(route)

    assert gate._candidate_engagement_errors(route, [], [zero]) == [
        "row 20 request route receipt contradicts prompt phase",
    ]
    assert gate._candidate_engagement_errors(route, [], [engaged, engaged]) == []
    assert gate._candidate_engagement_errors(route, [], [engaged, zero]) == [
        "row 20 request route receipt contradicts prompt phase"
    ]


def test_row20_1k_receipt_is_explicit_stock_without_false_engagement() -> None:
    gate = _module()
    route = "r08_device_draft+r20_kv_only_history"

    run = _receipt_run(route, prompt_tokens=1_024)

    assert run["history_route_receipt"]["route_id"] == "stock_history"
    assert run["history_route_receipt"]["row20_engaged"] is False
    assert gate._candidate_engagement_errors(route, [], [run, run]) == []


def test_row50_conditions_cache_clearing_candidate_before_control() -> None:
    gate = _module()
    control = "r08_device_draft"
    candidate = control + "+r50_wired_residency"

    assert gate._conditioning_order(
        [control, candidate, candidate, control],
        candidate_id=candidate,
    ) == [candidate, control]
    assert gate._conditioning_order(
        [control, control + "+r48_boundary_fused"],
        candidate_id=control + "+r48_boundary_fused",
    ) == [control, control + "+r48_boundary_fused"]


def test_row53_extends_the_retained_stack_with_process_latched_buffers() -> None:
    gate = _module()
    control = "r08_device_draft+r50_wired_residency"
    candidate = control + "+r53_command_buffers"

    options = gate._route_execution_options(candidate)

    assert options["row53_command_buffers"] is True
    assert options["source_rows"] == (8, 50, 53)
    engaged = _receipt_run(candidate)
    assert gate._candidate_engagement_errors(candidate, [], [engaged, engaged]) == []


def test_row53_requires_exact_process_latched_buffer_values() -> None:
    gate = _module()
    route = "r08_device_draft+r53_command_buffers"
    wrong = _receipt_run(
        route,
        overrides={
            "r53_command_buffers": {
                "installed": True,
                "active": False,
                "max_mb_per_buffer": 128,
                "max_ops_per_buffer": 50,
            }
        },
    )

    assert gate._candidate_engagement_errors(route, [], [wrong]) == [
        "row 53 process-latched command-buffer profile was not active"
    ]


def test_row53_isolated_children_set_candidate_and_unset_control_env() -> None:
    isolated = _isolated_module()
    inherited = {
        "KEEP": "yes",
        "MLX_MAX_MB_PER_BUFFER": "128",
        "MLX_MAX_OPS_PER_BUFFER": "99",
    }

    control = isolated._environment_for_route(
        "r08_device_draft+r50_wired_residency",
        inherited,
    )
    candidate = isolated._environment_for_route(
        "r08_device_draft+r50_wired_residency+r53_command_buffers",
        inherited,
    )

    assert control == {"KEEP": "yes"}
    assert candidate["KEEP"] == "yes"
    assert candidate["MLX_MAX_MB_PER_BUFFER"] == "512"
    assert candidate["MLX_MAX_OPS_PER_BUFFER"] == "50"


def test_row53_isolated_gate_runs_as_a_direct_script_outside_repo() -> None:
    result = subprocess.run(
        [sys.executable, str(ISOLATED_SCRIPT), "--help"],
        cwd="/tmp",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_row21_promotion_requires_qk_fusion_engagement() -> None:
    gate = _module()
    route = (
        "r08_device_draft+r10_compact_vocab+r17_q4_mtp_block+"
        "r18_gdn_decay_memo+r20_kv_only_history+r21_qk_rms_rope"
    )
    zero = _receipt_run(
        route, overrides={"r21_qk_rms_rope": {"active_modules": 0}}
    )
    engaged = _receipt_run(route)

    assert gate._candidate_engagement_errors(route, [], [zero]) == [
        "r21_qk_rms_rope construction receipt is not active"
    ]
    assert gate._candidate_engagement_errors(route, [], [engaged, engaged]) == []


def test_row48_promotion_requires_boundary_fusion_engagement() -> None:
    gate = _module()
    route = "r08_device_draft+r48_boundary_fused"
    zero = _receipt_run(
        route, overrides={"r48_boundary_fused": {"active": False}}
    )
    engaged = _receipt_run(route)

    assert gate._candidate_engagement_errors(route, [], [zero]) == [
        "r48_boundary_fused construction receipt is not active",
    ]
    assert gate._candidate_engagement_errors(route, [], [engaged, engaged]) == []


def test_row63_promotion_requires_q8_embedding_fusion_engagement() -> None:
    gate = _module()
    route = "r08_device_draft+r63_q8_embedding_dual_norm"
    zero = _receipt_run(
        route, overrides={"r63_q8_embedding_dual_norm": {"active": False}}
    )
    engaged = _receipt_run(route)

    assert gate._candidate_engagement_errors(route, [], [zero]) == [
        "r63_q8_embedding_dual_norm construction receipt is not active"
    ]
    assert gate._candidate_engagement_errors(route, [], [engaged, engaged]) == []


def test_row17_promotion_requires_the_pinned_q4_mtp_block() -> None:
    gate = _module()
    route = "r08_device_draft+r10_compact_vocab+r17_q4_mtp_block"
    wrong = _receipt_run(
        route,
        overrides={
            "r17_q4_mtp_block": {
                "installed": True,
                "active": True,
                "variant": "r28",
                "bits": 4,
                "group_size": 64,
            }
        },
    )
    engaged = _receipt_run(
        route,
        overrides={
            "r17_q4_mtp_block": {
                "installed": True,
                "active": True,
                "variant": "r17",
                "bits": 4,
                "group_size": 64,
            }
        },
    )

    assert gate._candidate_engagement_errors(route, [], [wrong]) == [
        "row 17 pinned Q4/group-64 MTP block was not active"
    ]
    assert gate._candidate_engagement_errors(route, [], [engaged, engaged]) == []


def test_row_24_eval_ladder_extends_the_retained_stack() -> None:
    gate = _module()
    route = (
        "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo+"
        "r20_kv_only_history+r24_eval_ladder"
    )

    options = gate._route_execution_options(route)

    assert options["row24_eval_ladder"] is True
    assert options["source_rows"] == (8, 10, 18, 20, 24)


def test_row24_promotion_requires_eval_ladder_engagement() -> None:
    gate = _module()
    route = (
        "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo+"
        "r20_kv_only_history+r24_eval_ladder"
    )
    zero = _receipt_run(
        route, overrides={"r24_eval_ladder": {"active": False}}
    )
    engaged = _receipt_run(route)

    assert gate._candidate_engagement_errors(route, [], [zero]) == [
        "r24_eval_ladder construction receipt is not active"
    ]
    assert gate._candidate_engagement_errors(route, [], [engaged, engaged]) == []


def test_row_26_prefill_cadence_extends_retained_row_24() -> None:
    gate = _module()
    route = (
        "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo+"
        "r20_kv_only_history+r24_eval_ladder+r26_prefill_ladder_3"
    )

    options = gate._route_execution_options(route)

    assert options["row26_prefill_ladder_3"] is True
    assert options["source_rows"] == (8, 10, 18, 20, 24, 26)


def test_row26_promotion_requires_prefill_cadence_engagement() -> None:
    gate = _module()
    route = (
        "r08_device_draft+r10_compact_vocab+r18_gdn_decay_memo+"
        "r20_kv_only_history+r24_eval_ladder+r26_prefill_ladder_3"
    )
    zero = _receipt_run(
        route, overrides={"r26_prefill_ladder_3": {"active": False}}
    )
    engaged = _receipt_run(route)

    assert gate._candidate_engagement_errors(route, [], [zero]) == [
        "r26_prefill_ladder_3 construction receipt is not active"
    ]
    assert gate._candidate_engagement_errors(route, [], [engaged, engaged]) == []


def test_promotion_gate_is_strictly_above_point_zero_five_and_clean() -> None:
    gate = _module()
    order = ["kv_only_history", "kv_only_history+dual_norm"] * 2
    order[2:] = ["kv_only_history+dual_norm", "kv_only_history"]
    kwargs = {
        "order": order,
        "control_id": "kv_only_history",
        "candidate_id": "kv_only_history+dual_norm",
        "correctness": {"passed": True},
        "source_status": [],
    }

    passed = gate._promotion_decision(improvement_pct=0.050453818, **kwargs)
    tied = gate._promotion_decision(improvement_pct=0.05, **kwargs)
    dirty = gate._promotion_decision(
        improvement_pct=0.050453818,
        **{**kwargs, "source_status": [" M mtplx/runtime.py"]},
    )

    assert passed == {"passed": True, "threshold_pct": 0.05, "errors": []}
    assert tied["passed"] is False
    assert any("strictly greater" in error for error in tied["errors"])
    assert dirty["passed"] is False
    assert any("source tree" in error for error in dirty["errors"])

    for invalid in (float("nan"), float("inf"), float("-inf")):
        rejected = gate._promotion_decision(improvement_pct=invalid, **kwargs)
        assert rejected["passed"] is False
        assert any("finite" in error for error in rejected["errors"])


def test_promotion_gate_allows_row28_to_replace_retained_row17_artifact() -> None:
    gate = _module()
    prefix = (
        "r08_device_draft+r10_compact_vocab+r17_q4_mtp_block+"
        "r18_gdn_decay_memo+r20_kv_only_history+r21_qk_rms_rope+"
        "r24_eval_ladder+r26_prefill_ladder_3"
    )
    candidate = prefix.replace("r17_q4_mtp_block", "r28_q4_mtp_block")

    result = gate._promotion_decision(
        order=[prefix, candidate, candidate, prefix],
        control_id=prefix,
        candidate_id=candidate,
        improvement_pct=0.1,
        correctness={"passed": True},
        source_status=[],
    )

    assert result == {"passed": True, "threshold_pct": 0.05, "errors": []}


def test_promotion_gate_uses_the_native_mtp_registry_for_one_delta() -> None:
    gate = _module()
    control = "r08_device_draft+r18_gdn_decay_memo"
    candidate = control + "+r10_compact_vocab"

    result = gate._promotion_decision(
        order=[control, candidate, candidate, control],
        control_id=control,
        candidate_id=candidate,
        improvement_pct=0.1,
        correctness={"passed": True},
        source_status=[],
    )

    assert result == {"passed": True, "threshold_pct": 0.05, "errors": []}


def test_rebench_promotion_gate_can_isolate_one_frozen_feature() -> None:
    gate = _module()
    control = "r08_device_draft+r10_compact_vocab"
    candidate = control + "+r18_gdn_decay_memo"

    result = gate._promotion_decision(
        order=[control, candidate, candidate, control],
        control_id=control,
        candidate_id=candidate,
        improvement_pct=0.1,
        correctness={"passed": True},
        source_status=[],
        allow_frozen_candidate=True,
    )

    assert result == {"passed": True, "threshold_pct": 0.05, "errors": []}


@pytest.mark.parametrize(
    ("control", "candidate", "message"),
    [
        ("control", "r08_device_draft+r10_compact_vocab", "exactly one"),
        ("r08_device_draft", "r08_device_draft+r48_boundary_fused", "frozen substrate"),
        ("r08_device_draft", "r08_device_draft+r17_q4_mtp_block+r28_q4_mtp_block", "incompatible"),
        ("r61_dual_norm_concat", "r61_dual_norm_concat+r63_q8_embedding_dual_norm", "incompatible"),
        ("r08_device_draft", "r08_device_draft+dflash_m5", "unreachable"),
        ("r08_device_draft", "r08_device_draft+r42_argmax_shortlist", "correctness-ineligible"),
        ("r08_device_draft", "r08_device_draft+r70_qmv_sumtable", "removed"),
        ("r08_device_draft+r10_compact_vocab", "r08_device_draft+r20_kv_only_history", "arbitrary route"),
    ],
)
def test_promotion_gate_rejects_invalid_native_mtp_route_deltas(
    control: str,
    candidate: str,
    message: str,
) -> None:
    gate = _module()

    result = gate._promotion_decision(
        order=[control, candidate, candidate, control],
        control_id=control,
        candidate_id=candidate,
        improvement_pct=0.1,
        correctness={"passed": True},
        source_status=[],
    )

    assert result["passed"] is False
    assert any(message in error for error in result["errors"])


def test_promotion_gate_allows_row63_to_replace_row61_input_fusion() -> None:
    gate = _module()
    control = "r08_device_draft+r61_dual_norm_concat"
    candidate = "r08_device_draft+r63_q8_embedding_dual_norm"

    result = gate._promotion_decision(
        order=[control, candidate, candidate, control],
        control_id=control,
        candidate_id=candidate,
        improvement_pct=0.1,
        correctness={"passed": True},
        source_status=[],
    )

    assert result == {"passed": True, "threshold_pct": 0.05, "errors": []}


def test_promotion_gate_allows_block_candidate_to_replace_implicit_stock() -> None:
    gate = _module()
    control = "r08_device_draft"
    candidate = control + "+r28_q4_mtp_block"

    result = gate._promotion_decision(
        order=[control, candidate, candidate, control],
        control_id=control,
        candidate_id=candidate,
        improvement_pct=0.1,
        correctness={"passed": True},
        source_status=[],
    )

    assert result == {"passed": True, "threshold_pct": 0.05, "errors": []}
