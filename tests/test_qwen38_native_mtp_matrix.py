from __future__ import annotations

import importlib.util
from pathlib import Path
import json
import pytest
import subprocess
import sys
from typing import Any


SCRIPT = Path(__file__).parents[1] / "scripts/qwen38_native_mtp_matrix.py"
ARM_SCRIPT = Path(__file__).parents[1] / "scripts/qwen38_native_mtp_matrix_arm.py"


def _module():
    spec = importlib.util.spec_from_file_location("qwen38_native_mtp_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arm_module():
    spec = importlib.util.spec_from_file_location(
        "qwen38_native_mtp_matrix_arm", ARM_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _optimized_stack_receipt() -> dict[str, object]:
    return {
        "profile": "turbo",
        "runtime_profile": "native_mtp_turbo",
        "draft_lm_head": {"bits": 4, "group_size": 64, "mode": "affine"},
        "draft_sampler": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "mtp_hidden_variant": "post_norm",
        "mtp_cache_policy": "persistent",
        "mtp_history_policy": "committed",
        "verify_strategy": "capture_commit",
        "verify_core": "linear-gdn-from-conv-tape",
        "runtime_env": {
            "MTPLX_COMPILED_VERIFY": "1",
            "MTPLX_DROP_EVENTS": "1",
            "MTPLX_LAZY_MTP_HISTORY_APPEND": "1",
            "MTPLX_MTP_HISTORY_POLICY": "committed",
        },
    }


def test_matrix_has_four_fresh_lanes_and_no_historical_pr335_lane() -> None:
    matrix = _module()

    assert matrix.LANE_IDS == (
        "v2.9.2-mlx0322",
        "full-fixed-k3",
        "full-adaptive",
        "full-q4-adaptive",
    )
    assert matrix.PAIRED_ORDER == (
        "v2.9.2-mlx0322",
        "full-fixed-k3",
        "full-adaptive",
        "full-q4-adaptive",
        "full-q4-adaptive",
        "full-adaptive",
        "full-fixed-k3",
        "v2.9.2-mlx0322",
    )
    assert matrix.ONE_PASS_ORDER == matrix.LANE_IDS
    assert "pr335" not in " ".join(matrix.LANE_IDS).lower()


def test_matrix_workload_contract_redoes_every_requested_context() -> None:
    matrix = _module()

    assert matrix.CONTEXT_TOKENS == (1_024, 16_384, 65_536, 131_072)
    assert matrix.CONDITIONER_OUTPUT_TOKENS == 1_024
    assert matrix.LOW_OUTPUT_TOKENS == 1_024
    assert matrix.XHIGH_OUTPUT_TOKENS == 1_024
    assert matrix.VANITY_PROMPT_TOKENS == 100
    assert matrix.VANITY_TEMPERATURE == 0.0
    assert matrix.VANITY_PROMPT_FILE.name == "qwen38_palindrome_vanity.jsonl"
    assert matrix.PYTHON_PROMPT_FILE.name == "python_modules_long.jsonl"
    assert matrix.PYTHON_CONTEXT_MANIFEST.name == "qwen38-pr335-python-context.json"


def test_matrix_can_pair_only_the_selected_optimized_bf16_lane() -> None:
    matrix = _module()

    assert matrix.order_for_context(16_384, ("full-adaptive",)) == (
        "full-adaptive",
        "full-adaptive",
    )
    assert matrix.order_for_context(131_072, ("full-adaptive",)) == (
        "full-adaptive",
    )


def test_matrix_rejects_duplicate_and_unknown_lane_selections() -> None:
    matrix = _module()

    with pytest.raises(ValueError, match="unique"):
        matrix.order_for_context(
            16_384,
            ("full-adaptive", "full-adaptive"),
        )
    with pytest.raises(ValueError, match="unknown benchmark lanes"):
        matrix.order_for_context(16_384, ("unknown-lane",))


def test_matrix_cli_exposes_an_explicit_lane_selector() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--lanes" in result.stdout
    assert "full-adaptive" in result.stdout


def test_frozen_input_artifact_hashes_match_repository_bytes() -> None:
    matrix = _module()

    assert matrix._sha256(matrix.VANITY_PROMPT_FILE) == (
        matrix.PROMPT_ARTIFACT_SHA256["vanity"]
    )
    assert matrix._sha256(matrix.PYTHON_PROMPT_FILE) == (
        matrix.PROMPT_ARTIFACT_SHA256["python"]
    )
    manifest = json.loads(matrix.PYTHON_CONTEXT_MANIFEST.read_text(encoding="utf-8"))
    assert manifest == {
        "path": "mtplx/generation.py",
        "sha256": matrix.PYTHON_CONTEXT_SHA256,
        "source_commit": "9a6f48e69f9c8c6932d0f005c364844b2bf33e9c",
        "source_pr": 335,
    }


def test_campaign_accepts_the_external_pr335_context_by_hash(
    monkeypatch, tmp_path: Path
) -> None:
    matrix = _module()
    context = tmp_path / "pr335-generation.py"
    row17 = tmp_path / "row17.safetensors"
    args = type(
        "Args",
        (),
        {
            "baseline_root": tmp_path / "baseline",
            "candidate_root": tmp_path / "candidate",
            "output_root": tmp_path / "receipts",
            "workload": "low",
            "prompt_file": matrix.PYTHON_PROMPT_FILE,
            "context_file": context,
            "row17_artifact": row17,
        },
    )()
    monkeypatch.setattr(matrix, "_git_status", lambda root: [])
    hashes = {
        matrix.PYTHON_PROMPT_FILE: matrix.PROMPT_ARTIFACT_SHA256["python"],
        context: matrix.PYTHON_CONTEXT_SHA256,
        row17: matrix.ROW17_ARTIFACT_SHA256,
    }
    monkeypatch.setattr(matrix, "_sha256", hashes.__getitem__)

    matrix._assert_campaign_inputs(args)


def test_lane_specs_keep_source_and_head_changes_separate(tmp_path: Path) -> None:
    matrix = _module()
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    specs = matrix.lane_specs(
        baseline_root=baseline,
        baseline_commit=matrix.V292_COMMIT,
        candidate_root=candidate,
        candidate_commit="c" * 40,
        workload="low",
    )

    assert specs["v2.9.2-mlx0322"].source_root == baseline
    assert specs["v2.9.2-mlx0322"].source_commit == matrix.V292_COMMIT
    assert specs["v2.9.2-mlx0322"].route_id == "control"
    assert specs["full-fixed-k3"].source_root == candidate
    assert specs["full-fixed-k3"].route_id == matrix.LOW_FIXED_NATIVE_ROUTE
    assert specs["full-adaptive"].route_id == matrix.LOW_ADAPTIVE_NATIVE_ROUTE
    assert specs["full-q4-adaptive"].route_id == (
        matrix.LOW_Q4_ADAPTIVE_NATIVE_ROUTE
    )


def test_lane_specs_select_stock_draft_once_for_vanity_and_workload_profiles_once(
    tmp_path: Path,
) -> None:
    matrix = _module()
    common = {
        "baseline_root": tmp_path / "baseline",
        "baseline_commit": matrix.V292_COMMIT,
        "candidate_root": tmp_path / "candidate",
        "candidate_commit": "c" * 40,
    }

    specs = matrix.lane_specs(workload="vanity", **common)
    assert specs["full-fixed-k3"].route_id == matrix.XHIGH_FIXED_NATIVE_ROUTE
    assert specs["full-adaptive"].route_id == matrix.XHIGH_ADAPTIVE_NATIVE_ROUTE
    assert (
        specs["full-q4-adaptive"].route_id
        == matrix.XHIGH_Q4_ADAPTIVE_NATIVE_ROUTE
    )

    specs = matrix.lane_specs(workload="low", **common)
    assert specs["full-fixed-k3"].route_id == matrix.LOW_FIXED_NATIVE_ROUTE
    assert specs["full-adaptive"].route_id == matrix.LOW_ADAPTIVE_NATIVE_ROUTE
    assert specs["full-q4-adaptive"].route_id == matrix.LOW_Q4_ADAPTIVE_NATIVE_ROUTE

    specs = matrix.lane_specs(workload="xhigh", **common)
    assert specs["full-fixed-k3"].route_id == matrix.XHIGH_FIXED_NATIVE_ROUTE
    assert specs["full-adaptive"].route_id == matrix.XHIGH_ADAPTIVE_NATIVE_ROUTE
    assert (
        specs["full-q4-adaptive"].route_id
        == matrix.XHIGH_Q4_ADAPTIVE_NATIVE_ROUTE
    )


def test_full_fixed_k3_uses_the_complete_bf16_stack_without_adaptive_depth(
    tmp_path: Path,
) -> None:
    matrix = _module()

    low_features = matrix.gate._validate_route_id(matrix.LOW_FIXED_NATIVE_ROUTE)
    assert low_features == {
        "r08_device_draft",
        "r10_compact_vocab",
        "r20_kv_only_history",
        "r21_qk_rms_rope",
        "r24_eval_ladder",
        "r26_prefill_ladder_3",
        "r53_command_buffers",
    }
    xhigh_features = matrix.gate._validate_route_id(matrix.XHIGH_FIXED_NATIVE_ROUTE)
    assert xhigh_features == {
        "r20_kv_only_history",
        "r24_eval_ladder",
        "r26_prefill_ladder_3",
        "r50_wired_residency",
        "r53_command_buffers",
    }
    for route, draft_core in (
        (matrix.LOW_FIXED_NATIVE_ROUTE, "device"),
        (matrix.XHIGH_FIXED_NATIVE_ROUTE, "stock"),
    ):
        options = matrix.gate._route_execution_options(route)
        assert options["draft_core"] == draft_core
        assert options["speculative_depth"] == 3
        assert options["adaptive_policy"] == "none"
        assert options["adaptive_depth_cap"] == 0
        assert options["mtp_block_variant"] is None


def test_only_v292_is_unoptimized(tmp_path: Path) -> None:
    matrix = _module()
    specs = matrix.lane_specs(
        baseline_root=tmp_path / "baseline",
        baseline_commit=matrix.V292_COMMIT,
        candidate_root=tmp_path / "candidate",
        candidate_commit="c" * 40,
        workload="low",
    )

    assert specs["v2.9.2-mlx0322"].route_id == "control"
    assert all(
        specs[lane_id].route_id != "control"
        for lane_id in ("full-fixed-k3", "full-adaptive", "full-q4-adaptive")
    )


def test_full_fixed_receipt_requires_bf16_kernels_features_and_no_policy() -> None:
    matrix = _module()
    route = matrix.XHIGH_FIXED_NATIVE_ROUTE
    receipt = {
        "route_id": route,
        "installed_route_id": matrix.XHIGH_BF16_OPTIMIZED_INSTALLED_ROUTE_ID,
        "performance_profile": "xhigh",
        "speculative_depth": 3,
        "requested_speculative_depth": 3,
        "adaptive_policy_receipt": None,
        "adaptive_policy_events": [],
        "kernel_ids": list(matrix.XHIGH_BF16_OPTIMIZED_KERNEL_IDS),
        "feature_receipt": {
            "r20_kv_only_history": {"installed": True},
            "r24_eval_ladder": {"active": 1},
            "r26_prefill_ladder_3": {
                "active": 1,
                "phase_scope": "prefill",
                "decode_route": "stock",
            },
            "r50_wired_residency": {"active": True, "installed": True},
            "r53_command_buffers": {"active": True, "installed": True},
        },
        "draft_core": "stock",
        "device_core_receipt": {
            "requested": "stock",
            "device_calls": 0,
            "device_fallbacks": 0,
        },
        "compiled_verify_receipt": {
            "mode": "on",
            "compiled_calls": 8,
            "fallback_reasons": {},
            "permanent_eager": False,
            "permanent_eager_reason": None,
        },
    }

    assert matrix.full_fixed_receipt_errors(receipt, expected_route=route) == []

    missing_prefill = receipt["feature_receipt"].pop("r26_prefill_ladder_3")
    assert "optimized fixed K3 feature receipt mismatch" in (
        matrix.full_fixed_receipt_errors(receipt, expected_route=route)
    )
    receipt["feature_receipt"]["r26_prefill_ladder_3"] = missing_prefill
    receipt["feature_receipt"]["r18_gdn_decay_memo"] = {"active": True}
    assert "optimized fixed K3 feature receipt mismatch" in (
        matrix.full_fixed_receipt_errors(receipt, expected_route=route)
    )
    del receipt["feature_receipt"]["r18_gdn_decay_memo"]

    receipt["adaptive_policy_receipt"] = {"kind": "position_ema", "executed": True}
    assert "optimized fixed K3 executed an adaptive policy" in (
        matrix.full_fixed_receipt_errors(receipt, expected_route=route)
    )
    receipt["adaptive_policy_receipt"] = None
    receipt["feature_receipt"]["r28_q4_mtp_block"] = {"active": True}
    assert "optimized fixed K3 installed a Q4 MTP block" in (
        matrix.full_fixed_receipt_errors(receipt, expected_route=route)
    )
    del receipt["feature_receipt"]["r28_q4_mtp_block"]
    receipt["device_core_receipt"]["device_fallbacks"] = 1
    assert "optimized fixed K3 stock draft fallback occurred" in (
        matrix.full_fixed_receipt_errors(receipt, expected_route=route)
    )


def test_adaptive_receipts_require_the_exact_shared_stack_and_q4_delta() -> None:
    matrix = _module()
    shared_features = {
        key: {"active": True}
        for key in matrix.XHIGH_BF16_OPTIMIZED_FEATURE_KEYS
    }
    shared_features["r26_prefill_ladder_3"].update(
        {"phase_scope": "prefill", "decode_route": "stock"}
    )
    bf16 = {
        "route_id": matrix.XHIGH_ADAPTIVE_NATIVE_ROUTE,
        "installed_route_id": matrix.XHIGH_BF16_OPTIMIZED_INSTALLED_ROUTE_ID,
        "performance_profile": "xhigh",
        "kernel_ids": list(matrix.XHIGH_BF16_OPTIMIZED_KERNEL_IDS),
        "feature_receipt": shared_features,
    }

    assert matrix.adaptive_optimized_receipt_errors(
        bf16, expected_route=matrix.XHIGH_ADAPTIVE_NATIVE_ROUTE
    ) == []
    bf16["feature_receipt"]["r48_boundary_fused"] = {"active": True}
    assert "adaptive feature receipt mismatch" in (
        matrix.adaptive_optimized_receipt_errors(
            bf16, expected_route=matrix.XHIGH_ADAPTIVE_NATIVE_ROUTE
        )
    )
    del bf16["feature_receipt"]["r48_boundary_fused"]
    bf16["kernel_ids"] = list(matrix.XHIGH_BF16_OPTIMIZED_KERNEL_IDS[:-1])
    assert "adaptive BF16 kernel stack mismatch" in matrix.adaptive_optimized_receipt_errors(
        bf16, expected_route=matrix.XHIGH_ADAPTIVE_NATIVE_ROUTE
    )

    q4 = {
        "route_id": matrix.XHIGH_Q4_ADAPTIVE_NATIVE_ROUTE,
        "installed_route_id": matrix.XHIGH_Q4_OPTIMIZED_INSTALLED_ROUTE_ID,
        "performance_profile": "xhigh",
        "kernel_ids": list(matrix.XHIGH_Q4_OPTIMIZED_KERNEL_IDS),
        "feature_receipt": {
            **shared_features,
            "r17_q4_mtp_block": {"active": True},
        },
    }
    assert matrix.adaptive_optimized_receipt_errors(
        q4, expected_route=matrix.XHIGH_Q4_ADAPTIVE_NATIVE_ROUTE
    ) == []
    del q4["feature_receipt"]["r17_q4_mtp_block"]
    assert "adaptive Q4 MTP block is inactive" in matrix.adaptive_optimized_receipt_errors(
        q4, expected_route=matrix.XHIGH_Q4_ADAPTIVE_NATIVE_ROUTE
    )


def test_arm_policy_contract_distinguishes_fixed_from_adaptive() -> None:
    matrix = _module()
    arm = _arm_module()

    arm._assert_route_policy_contract(
        matrix.LOW_FIXED_NATIVE_ROUTE,
        {"adaptive_policy_receipt": None, "adaptive_policy_events": []},
    )
    try:
        arm._assert_route_policy_contract(
            matrix.LOW_FIXED_NATIVE_ROUTE,
            {
                "adaptive_policy_receipt": {
                    "kind": "position_ema",
                    "executed": True,
                }
            },
        )
    except RuntimeError as exc:
        assert "fixed optimized route executed an adaptive policy" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("fixed optimized route accepted an adaptive policy")

    arm._assert_route_policy_contract(
        matrix.LOW_ADAPTIVE_NATIVE_ROUTE,
        {
            "adaptive_policy_receipt": {
                "kind": "position_ema",
                "executed": True,
            }
        },
    )


def test_vanity_lane_specs_keep_the_stock_draft_optimized_native_mtp_stack(
    tmp_path: Path,
) -> None:
    matrix = _module()
    specs = matrix.lane_specs(
        baseline_root=tmp_path / "baseline",
        baseline_commit=matrix.V292_COMMIT,
        candidate_root=tmp_path / "candidate",
        candidate_commit="c" * 40,
        workload="vanity",
    )

    bf16 = specs["full-adaptive"].route_id
    q4 = specs["full-q4-adaptive"].route_id
    assert bf16 == matrix.XHIGH_ADAPTIVE_NATIVE_ROUTE
    assert q4 == matrix.XHIGH_Q4_ADAPTIVE_NATIVE_ROUTE
    assert matrix.gate._route_execution_options(bf16)["draft_core"] == "stock"
    assert matrix.gate._route_execution_options(q4)["draft_core"] == "stock"
    for optimized in (
        "r20_kv_only_history",
        "r24_eval_ladder",
        "r26_prefill_ladder_3",
        "r50_wired_residency",
        "r53_command_buffers",
        "r11_position_ema",
    ):
        assert optimized in bf16
    assert "r08_device_draft" not in bf16
    assert "r10_compact_vocab" not in bf16
    assert "r17_q4_mtp_block" in q4


def test_stochastic_lane_specs_keep_the_phase_split_stock_draft_stack(
    tmp_path: Path,
) -> None:
    matrix = _module()

    expected = {
        "low": matrix.LOW_ADAPTIVE_NATIVE_ROUTE,
        "xhigh": matrix.XHIGH_ADAPTIVE_NATIVE_ROUTE,
    }
    expected_q4 = {
        "low": matrix.LOW_Q4_ADAPTIVE_NATIVE_ROUTE,
        "xhigh": matrix.XHIGH_Q4_ADAPTIVE_NATIVE_ROUTE,
    }
    for workload in ("low", "xhigh"):
        specs = matrix.lane_specs(
            baseline_root=tmp_path / "baseline",
            baseline_commit=matrix.V292_COMMIT,
            candidate_root=tmp_path / "candidate",
            candidate_commit="c" * 40,
            workload=workload,
        )
        assert specs["full-adaptive"].route_id == expected[workload]
        assert (
            specs["full-q4-adaptive"].route_id
            == expected_q4[workload]
        )
        assert (
            matrix.gate._route_execution_options(
                specs["full-adaptive"].route_id
            )["draft_core"]
            == ("device" if workload == "low" else "stock")
        )


def test_128k_uses_one_pass_but_shorter_contexts_use_symmetric_pairs() -> None:
    matrix = _module()

    assert matrix.order_for_context(1_024) == matrix.PAIRED_ORDER
    assert matrix.order_for_context(16_384) == matrix.PAIRED_ORDER
    assert matrix.order_for_context(65_536) == matrix.PAIRED_ORDER
    assert matrix.order_for_context(131_072) == matrix.ONE_PASS_ORDER


def test_aggregate_uses_the_renamed_full_fixed_lane_as_wall_baseline(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    matrix = _module()
    specs = matrix.lane_specs(
        baseline_root=tmp_path / "baseline",
        baseline_commit=matrix.V292_COMMIT,
        candidate_root=tmp_path / "candidate",
        candidate_commit="c" * 40,
        workload="low",
    )
    wall_by_lane = {
        "v2.9.2-mlx0322": 12.0,
        "full-fixed-k3": 10.0,
        "full-adaptive": 8.0,
        "full-q4-adaptive": 9.0,
    }
    receipts = [
        {
            "lane_id": lane_id,
            "wall_s": wall_by_lane[lane_id],
            "prefill_tok_s": 800.0,
            "decode_tok_s": 20.0,
            "peak_memory_gib": 40.0,
            "token_hash": lane_id,
        }
        for lane_id in matrix.ONE_PASS_ORDER
    ]
    monkeypatch.setattr(matrix, "receipt_errors", lambda *args, **kwargs: [])

    result = matrix.aggregate(
        workload="low",
        context_tokens=131_072,
        order=matrix.ONE_PASS_ORDER,
        receipts=receipts,
        specs=specs,
    )

    assert result["summary"]["full-fixed-k3"]["wall_faster_vs_fixed_k3_pct"] == 0.0
    assert result["summary"]["full-adaptive"]["wall_faster_vs_fixed_k3_pct"] == 25.0


def test_128k_aggregate_derives_depth_usage_from_recorded_schedules(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    matrix = _module()
    specs = matrix.lane_specs(
        baseline_root=tmp_path / "baseline",
        baseline_commit=matrix.V292_COMMIT,
        candidate_root=tmp_path / "candidate",
        candidate_commit="c" * 40,
        workload="xhigh",
    )
    receipt = {
        "lane_id": "full-adaptive",
        "wall_s": 10.0,
        "prefill_tok_s": 500.0,
        "decode_tok_s": 25.0,
        "peak_memory_gib": 40.0,
        "token_hash": "single-pass",
        "verify_calls": 4,
        "attempted_depth_schedule": [1, 2, 2, 3],
        "accepted_depth_schedule": [0, 1, 2, 3],
        "drafted_by_depth": [4, 3, 1],
        "accepted_by_depth": [3, 2, 1],
    }
    monkeypatch.setattr(matrix, "receipt_errors", lambda *args, **kwargs: [])

    result = matrix.aggregate(
        workload="xhigh",
        context_tokens=131_072,
        order=("full-adaptive",),
        receipts=[receipt],
        specs=specs,
    )

    assert result["invariant_errors"] == []
    assert result["summary"]["full-adaptive"]["depth_usage"] == (
        matrix.depth_usage_from_schedules(
            attempted_depth_schedule=[1, 2, 2, 3],
            accepted_depth_schedule=[0, 1, 2, 3],
            verify_calls=4,
            drafted_by_depth=[4, 3, 1],
            accepted_by_depth=[3, 2, 1],
        )
    )


def test_single_bf16_lane_aggregate_does_not_require_an_unmeasured_fixed_lane(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    matrix = _module()
    specs = matrix.lane_specs(
        baseline_root=tmp_path / "baseline",
        baseline_commit=matrix.V292_COMMIT,
        candidate_root=tmp_path / "candidate",
        candidate_commit="c" * 40,
        workload="low",
    )
    order = ("full-adaptive", "full-adaptive")
    receipts = [
        {
            "lane_id": "full-adaptive",
            "wall_s": wall,
            "prefill_tok_s": 800.0,
            "decode_tok_s": 60.0,
            "peak_memory_gib": 22.0,
            "token_hash": "same",
        }
        for wall in (20.0, 22.0)
    ]
    monkeypatch.setattr(matrix, "receipt_errors", lambda *args, **kwargs: [])

    result = matrix.aggregate(
        workload="low",
        context_tokens=16_384,
        order=order,
        receipts=receipts,
        specs=specs,
    )

    assert result["invariant_errors"] == []
    assert set(result["summary"]) == {"full-adaptive"}
    assert result["summary"]["full-adaptive"][
        "wall_faster_vs_fixed_k3_pct"
    ] is None


@pytest.mark.parametrize(
    "order",
    (
        ("full-adaptive",),
        ("full-adaptive", "full-adaptive", "full-adaptive"),
        (
            "full-adaptive",
            "full-fixed-k3",
            "full-adaptive",
            "full-fixed-k3",
        ),
        ("unknown-lane", "unknown-lane"),
    ),
)
def test_aggregate_rejects_noncanonical_16k_lane_orders(
    order: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    matrix = _module()
    specs = matrix.lane_specs(
        baseline_root=tmp_path / "baseline",
        baseline_commit=matrix.V292_COMMIT,
        candidate_root=tmp_path / "candidate",
        candidate_commit="c" * 40,
        workload="low",
    )
    receipts = [
        {
            "lane_id": lane_id,
            "wall_s": 20.0,
            "prefill_tok_s": 800.0,
            "decode_tok_s": 60.0,
            "peak_memory_gib": 22.0,
            "token_hash": "same",
        }
        for lane_id in order
    ]
    monkeypatch.setattr(matrix, "receipt_errors", lambda *args, **kwargs: [])

    result = matrix.aggregate(
        workload="low",
        context_tokens=16_384,
        order=order,
        receipts=receipts,
        specs=specs,
    )

    assert result["invariant_errors"]


def test_child_command_attests_source_workload_and_custom_head(tmp_path: Path) -> None:
    matrix = _module()
    lane = matrix.LaneSpec(
        lane_id="full-q4-adaptive",
        source_root=tmp_path / "source",
        source_commit="d" * 40,
        route_id=matrix.XHIGH_Q4_ADAPTIVE_NATIVE_ROUTE,
    )
    command = matrix.child_command(
        lane=lane,
        workload="xhigh",
        context_tokens=16_384,
        output=tmp_path / "arm.json",
        model=tmp_path / "model",
        prompt_file=tmp_path / "prompt.jsonl",
        context_file=tmp_path / "context.py",
        row17_artifact=tmp_path / "mtp.safetensors",
        python=tmp_path / "python",
        lock=tmp_path / "gpu.lock",
    )
    joined = " ".join(map(str, command))

    for expected in (
        f"--source-root {lane.source_root}",
        f"--source-commit {lane.source_commit}",
        "--lane-id full-q4-adaptive",
        f"--route {matrix.XHIGH_Q4_ADAPTIVE_NATIVE_ROUTE}",
        f"--row17-artifact {tmp_path / 'mtp.safetensors'}",
        "--workload xhigh",
        "--prompt-tokens 16384",
        "--max-tokens 1024",
        "--warmup-tokens 1024",
        "--target-temperature 1.0",
        "--top-p 0.95",
        "--top-k 20",
        "--force-exact-output",
    ):
        assert expected in joined
    assert "--record-depth-usage" not in command

    command_128k = matrix.child_command(
        lane=lane,
        workload="xhigh",
        context_tokens=131_072,
        output=tmp_path / "arm-128k.json",
        model=tmp_path / "model",
        prompt_file=tmp_path / "prompt.jsonl",
        context_file=tmp_path / "context.py",
        row17_artifact=tmp_path / "mtp.safetensors",
        python=tmp_path / "python",
        lock=tmp_path / "gpu.lock",
    )
    assert "--record-depth-usage" in command_128k

    fixed_command_128k = matrix.child_command(
        lane=matrix.LaneSpec(
            lane_id="full-fixed-k3",
            source_root=tmp_path / "source",
            source_commit="d" * 40,
            route_id=matrix.XHIGH_FIXED_NATIVE_ROUTE,
        ),
        workload="xhigh",
        context_tokens=131_072,
        output=tmp_path / "fixed-128k.json",
        model=tmp_path / "model",
        prompt_file=tmp_path / "prompt.jsonl",
        context_file=tmp_path / "context.py",
        row17_artifact=tmp_path / "mtp.safetensors",
        python=tmp_path / "python",
        lock=tmp_path / "gpu.lock",
    )
    assert "--record-depth-usage" not in fixed_command_128k

    vanity_command = matrix.child_command(
        lane=lane,
        workload="vanity",
        context_tokens=100,
        output=tmp_path / "vanity.json",
        model=tmp_path / "model",
        prompt_file=tmp_path / "prompt.jsonl",
        context_file=tmp_path / "context.py",
        row17_artifact=tmp_path / "mtp.safetensors",
        python=tmp_path / "python",
        lock=tmp_path / "gpu.lock",
    )
    vanity_joined = " ".join(map(str, vanity_command))
    assert "--warmup-tokens 0" in vanity_joined
    assert "--force-exact-output" not in vanity_command


def test_matrix_arm_disables_stop_tokens_for_exact_output(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    arm = _arm_module()
    from scripts import qwen38_challenge_port_gate as gate

    observed: dict[str, object] = {}

    def fake_run_arm(*args: Any, **kwargs: Any) -> dict[str, object]:
        del args
        observed.update(kwargs)
        return {"draft_core": "device"}

    monkeypatch.setattr(gate, "_run_arm", fake_run_arm)

    arm._run_one(
        object(),
        {},
        tmp_path / "model",
        [1, 2, 3],
        route=gate.XHIGH_ADAPTIVE_NATIVE_ROUTE,
        max_tokens=16_384,
        seed=42,
        target_temperature=1.0,
        draft_temperature=1.0,
        top_p=0.95,
        top_k=20,
        row17_artifact=tmp_path / "row17.safetensors",
        record_depth_usage=False,
        force_exact_output=True,
    )

    assert observed["stop_token_ids"] == set()


def test_matrix_arm_requires_explicit_opt_in_for_fixed_diagnostic_route(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    arm = _arm_module()
    from scripts import qwen38_challenge_port_gate as gate

    observed: dict[str, object] = {}

    def fake_run_arm(*args: Any, **kwargs: Any) -> dict[str, object]:
        del args
        observed.update(kwargs)
        return {"draft_core": "device"}

    monkeypatch.setattr(gate, "_run_arm", fake_run_arm)

    common = {
        "runtime": object(),
        "config": {},
        "model": tmp_path / "model",
        "prompt_ids": [1, 2, 3],
        "route": "r08_device_draft",
        "max_tokens": 16_384,
        "seed": 42,
        "target_temperature": 1.0,
        "draft_temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "row17_artifact": tmp_path / "row17.safetensors",
        "record_depth_usage": False,
        "force_exact_output": True,
    }

    with pytest.raises(RuntimeError, match="non-final optimized route"):
        arm._run_one(**common)

    arm._run_one(**common, allow_fixed_diagnostic_route=True)

    assert observed["route_id"] == "r08_device_draft"
    assert observed["stop_token_ids"] == set()


@pytest.mark.parametrize(
    "route",
    (
        "r11_position_ema",
        "r28_q4_mtp_block",
    ),
)
def test_matrix_arm_fixed_diagnostic_route_rejects_adaptive_and_q4(
    route: str,
    tmp_path: Path,
) -> None:
    arm = _arm_module()

    with pytest.raises(RuntimeError, match="fixed BF16"):
        arm._run_one(
            object(),
            {},
            tmp_path / "model",
            [1, 2, 3],
            route=route,
            max_tokens=16_384,
            seed=42,
            target_temperature=1.0,
            draft_temperature=1.0,
            top_p=0.95,
            top_k=20,
            row17_artifact=tmp_path / "row17.safetensors",
            record_depth_usage=False,
            force_exact_output=True,
            allow_fixed_diagnostic_route=True,
        )


def test_matrix_arm_cli_requires_explicit_fixed_diagnostic_flag(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    arm = _arm_module()
    argv = [
        "qwen38_native_mtp_matrix_arm.py",
        "--source-root", str(tmp_path / "source"),
        "--source-commit", "a" * 40,
        "--lane-id", "candidate-r08",
        "--route", "r08_device_draft",
        "--workload", "xhigh",
        "--model", str(tmp_path / "model"),
        "--prompt-file", str(tmp_path / "prompt.jsonl"),
        "--context-file", str(tmp_path / "context.py"),
        "--prompt-tokens", "16384",
        "--max-tokens", "16384",
        "--warmup-tokens", "1024",
        "--seed", "42",
        "--target-temperature", "1.0",
        "--draft-temperature", "1.0",
        "--top-p", "0.95",
        "--top-k", "20",
        "--row17-artifact", str(tmp_path / "row17.safetensors"),
        "--force-exact-output",
        "--allow-fixed-diagnostic-route",
        "--lock", str(tmp_path / "gpu.lock"),
        "--output", str(tmp_path / "receipt.json"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    args = arm._parse_args()

    assert args.allow_fixed_diagnostic_route is True


def test_output_contract_accepts_natural_vanity_and_rejects_length_cap() -> None:
    matrix = _module()
    receipt = {
        "workload": "vanity",
        "generated_tokens": 120,
        "finish_reason": "stop",
        "stop_token_policy": "tokenizer_default",
        "conditioner_generated_tokens": 0,
        "conditioner_finish_reason": None,
    }

    assert matrix._output_contract_errors(receipt, output_tokens=1_024) == []
    receipt["finish_reason"] = "length"
    assert "vanity arm did not stop naturally" in matrix._output_contract_errors(
        receipt,
        output_tokens=1_024,
    )


def test_output_contract_rejects_short_conditioner() -> None:
    matrix = _module()
    receipt = {
        "workload": "low",
        "generated_tokens": 1_024,
        "finish_reason": "length",
        "stop_token_policy": "disabled_for_exact_output",
        "conditioner_generated_tokens": 700,
        "conditioner_finish_reason": "stop",
    }

    errors = matrix._output_contract_errors(receipt, output_tokens=1_024)

    assert "conditioner output token count is not exact" in errors
    assert "conditioner did not finish at the requested length" in errors


def test_campaign_rejects_dirty_sources_before_entering_gpu_window(
    monkeypatch, tmp_path: Path
) -> None:
    matrix = _module()
    args = type(
        "Args",
        (),
        {
            "baseline_root": tmp_path / "baseline",
            "candidate_root": tmp_path / "candidate",
            "output_root": tmp_path / "receipts",
            "workload": "low",
            "prompt_file": matrix.PYTHON_PROMPT_FILE,
            "context_file": tmp_path / "pr335-generation.py",
        },
    )()
    monkeypatch.setattr(
        matrix,
        "_git_status",
        lambda root: [" M mtplx/generation.py"] if root == args.candidate_root else [],
    )

    try:
        matrix._assert_campaign_inputs(args)
    except RuntimeError as exc:
        assert "candidate source tree must be clean" in str(exc)
    else:  # pragma: no cover - assertion message is more useful than pytest.raises here
        raise AssertionError("dirty candidate source was accepted")


def test_campaign_rejects_receipt_output_inside_either_source(
    monkeypatch, tmp_path: Path
) -> None:
    matrix = _module()
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    args = type(
        "Args",
        (),
        {
            "baseline_root": baseline,
            "candidate_root": candidate,
            "output_root": candidate / "bench" / "results",
            "workload": "low",
            "prompt_file": matrix.PYTHON_PROMPT_FILE,
            "context_file": tmp_path / "pr335-generation.py",
        },
    )()
    monkeypatch.setattr(matrix, "_git_status", lambda root: [])

    try:
        matrix._assert_campaign_inputs(args)
    except RuntimeError as exc:
        assert "outside the candidate source tree" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("in-tree receipt output was accepted")


def test_receipt_validation_requires_exact_source_and_route_engagement() -> None:
    matrix = _module()
    lane = matrix.LaneSpec(
        lane_id="full-adaptive",
        source_root=Path("candidate"),
        source_commit="e" * 40,
        route_id=matrix.LOW_ADAPTIVE_NATIVE_ROUTE,
    )
    receipt = {
        "lane_id": lane.lane_id,
        "source_commit": lane.source_commit,
        "source_status": [],
        "route_id": lane.route_id,
        "performance_profile": "low",
        "prompt_tokens": 16_384,
        "conditioner_output_tokens": 1_024,
        "conditioner_generated_tokens": 1_024,
        "conditioner_finish_reason": "length",
        "max_tokens": 1_024,
        "generated_tokens": 1_024,
        "finish_reason": "length",
        "stop_token_policy": "disabled_for_exact_output",
        "mlx_version": "0.32.2",
        "mlx_metal_version": "0.32.2",
        "gpu_lock_scope": "attested_parent",
        "source_import_attested": True,
        "model_id": matrix.MODEL_ID,
        "model_artifact_hashes": {
            "config.json": "a" * 64,
            "mtp.safetensors": "b" * 64,
        },
        "verify_strategy": "capture_commit",
        "verify_core": "linear-gdn-from-conv-tape",
        "speculative_depth": 3,
        "requested_speculative_depth": 3,
        "optimized_stack": _optimized_stack_receipt(),
        "workload": "low",
        "sampler": {
            "target_temperature": 1.0,
            "draft_temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
        },
        "prompt_token_sha256": matrix.PROMPT_TOKEN_SHA256["low"][16_384],
        "prompt_artifact_sha256": matrix.PROMPT_ARTIFACT_SHA256["python"],
        "context_artifact_sha256": matrix.PYTHON_CONTEXT_SHA256,
        "row17_artifact_sha256": matrix.ROW17_ARTIFACT_SHA256,
        "source_rows": [8, 10, 20, 21, 24, 26, 53, 11],
        "installed_route_id": matrix.LOW_BF16_OPTIMIZED_INSTALLED_ROUTE_ID,
        "kernel_ids": list(matrix.LOW_BF16_OPTIMIZED_KERNEL_IDS),
        "feature_receipt": {
            key: {"active": True}
            for key in matrix.LOW_BF16_OPTIMIZED_FEATURE_KEYS
        },
        "adaptive_policy_receipt": {"kind": "position_ema", "executed": True},
        "device_core_receipt": {
            "requested": "device",
            "device_calls": 400,
            "device_fallbacks": 0,
        },
        "compiled_verify_receipt": {
            "mode": "on",
            "compiled_calls": 400,
            "fallback_reasons": {},
            "permanent_eager": False,
            "permanent_eager_reason": None,
        },
        "history_route_receipt": {
            "route_id": "kv_only_history",
            "prompt_tokens": 16_384,
            "row20_engaged": True,
        },
        "draft_core": "device",
        "drafted_by_depth": [1, 1, 1],
        "accepted_by_depth": [1, 1, 1],
        "verify_calls": 1,
        "attempted_depth_schedule": [0] * 1_020 + [3],
        "depth_usage": matrix.depth_usage(
            decode_cycles=1_021,
            verify_calls=1,
            drafted_by_depth=[1, 1, 1],
            accepted_by_depth=[1, 1, 1],
        ),
    }
    receipt["adaptive_policy_receipt"].update(
        {
            "initial_accept_ema": [0.5, 0.5, 0.5],
            "final_accept_ema": [0.7, 0.6, 0.5],
            "initial_depth": 3,
            "final_depth": 2,
            "max_depth": 3,
            "depth_cap": 3,
        }
    )
    receipt["feature_receipt"]["r53_command_buffers"].update(
        {"max_mb_per_buffer": 512, "max_ops_per_buffer": 50}
    )

    assert matrix.receipt_errors(
        receipt,
        lane=lane,
        context_tokens=16_384,
        output_tokens=1_024,
    ) == []
    receipt["kernel_ids"] = []
    assert "optimized route reported no installed kernels" in matrix.receipt_errors(
        receipt,
        lane=lane,
        context_tokens=16_384,
        output_tokens=1_024,
    )
    receipt["kernel_ids"] = list(matrix.LOW_BF16_OPTIMIZED_KERNEL_IDS)
    receipt["model_artifact_hashes"] = {}
    assert "model artifact attestation is missing" in matrix.receipt_errors(
        receipt,
        lane=lane,
        context_tokens=16_384,
        output_tokens=1_024,
    )


class _Tokenizer:
    def __init__(self) -> None:
        self.template_calls: list[dict] = []

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(value) for value in ids)

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append(kwargs)
        content = messages[0]["content"]
        rendered = "<chat>" + content + "</chat>"
        if kwargs.get("tokenize"):
            return self.encode(rendered)
        return rendered


def test_arm_builds_exact_low_and_xhigh_prompt_budgets() -> None:
    arm = _arm_module()
    tokenizer = _Tokenizer()

    _, low_ids = arm.build_prompt(
        tokenizer,
        workload="low",
        instruction="solve",
        context="abc",
        target_tokens=1_024,
    )
    _, xhigh_ids = arm.build_prompt(
        tokenizer,
        workload="xhigh",
        instruction="solve",
        context="abc",
        target_tokens=16_384,
    )

    assert len(low_ids) == 1_024
    assert len(xhigh_ids) == 16_384
    assert tokenizer.template_calls[0]["enable_thinking"] is True
    assert tokenizer.template_calls[0]["reasoning_effort"] == "low"
    assert tokenizer.template_calls[-1]["enable_thinking"] is True
    assert tokenizer.template_calls[-1]["reasoning_effort"] == "xhigh"


def test_arm_requires_exact_100_token_non_thinking_vanity_prompt() -> None:
    arm = _arm_module()
    tokenizer = _Tokenizer()
    prompt = "x" * (100 - len("<chat></chat>"))

    _, token_ids = arm.build_prompt(
        tokenizer,
        workload="vanity",
        instruction=prompt,
        context="ignored",
        target_tokens=100,
    )

    assert len(token_ids) == 100
    assert tokenizer.template_calls[-1]["enable_thinking"] is False


@pytest.mark.parametrize(
    ("workload", "expected_profile"),
    (("vanity", "xhigh"), ("low", "low"), ("xhigh", "xhigh")),
)
def test_arm_maps_vanity_to_the_installed_stock_draft_performance_profile(
    workload: str,
    expected_profile: str,
) -> None:
    arm = _arm_module()

    assert arm._performance_profile_for_workload(workload) == expected_profile


def test_arm_module_imports_no_mlx_or_mtplx_runtime() -> None:
    source = ARM_SCRIPT.read_text(encoding="utf-8")
    prefix = source.split("def _activate_source_root", maxsplit=1)[0]

    assert "import mlx" not in prefix
    assert "from mtplx" not in prefix


def test_arm_uses_v292_native_internal_history_when_construction_binding_is_absent() -> None:
    arm = _arm_module()

    receipt = arm._history_route_receipt(object(), 16_384)

    assert receipt == {
        "route_id": "native_internal_committed_history",
        "prompt_tokens": 16_384,
        "row20_engaged": False,
        "construction_binding_available": False,
    }


def test_depth_usage_derives_attempted_and_accepted_d0_through_d3() -> None:
    matrix = _module()

    usage = matrix.depth_usage(
        decode_cycles=100,
        verify_calls=80,
        drafted_by_depth=[80, 50, 20],
        accepted_by_depth=[60, 30, 10],
    )

    assert usage["decode_cycles"] == 100
    assert usage["attempted_counts"] == {"D0": 20, "D1": 30, "D2": 30, "D3": 20}
    assert usage["accepted_counts"] == {"D0": 40, "D1": 30, "D2": 20, "D3": 10}
    assert usage["attempted_tokens_by_position"] == {"D1": 80, "D2": 50, "D3": 20}
    assert usage["accepted_tokens_by_position"] == {"D1": 60, "D2": 30, "D3": 10}
    assert usage["acceptance_rate_pct_by_position"] == {
        "D1": 75.0,
        "D2": 60.0,
        "D3": 50.0,
    }
    assert sum(usage["attempted_share_pct"].values()) == 100.0
    assert sum(usage["accepted_share_pct"].values()) == 100.0
    assert usage["mean_attempted_depth"] == 1.5
    assert usage["mean_accepted_depth"] == 1.0


def test_depth_usage_from_schedules_uses_recorded_cycle_depths() -> None:
    matrix = _module()

    usage = matrix.depth_usage_from_schedules(
        attempted_depth_schedule=[1, 2, 2, 3],
        accepted_depth_schedule=[0, 1, 2, 3],
        verify_calls=4,
        drafted_by_depth=[4, 3, 1],
        accepted_by_depth=[3, 2, 1],
    )

    assert usage["decode_cycles"] == 4
    assert usage["attempted_counts"] == {"D0": 0, "D1": 1, "D2": 2, "D3": 1}
    assert usage["accepted_counts"] == {"D0": 1, "D1": 1, "D2": 1, "D3": 1}
    assert usage["attempted_share_pct"] == {
        "D0": 0.0,
        "D1": 25.0,
        "D2": 50.0,
        "D3": 25.0,
    }
    assert usage["accepted_share_pct"] == {
        "D0": 25.0,
        "D1": 25.0,
        "D2": 25.0,
        "D3": 25.0,
    }
    assert usage["mean_attempted_depth"] == 2.0
    assert usage["mean_accepted_depth"] == 1.5


def test_adaptive_receipt_rejects_missing_or_mismatched_depth_telemetry() -> None:
    matrix = _module()
    lane = matrix.LaneSpec(
        lane_id="full-adaptive",
        source_root=Path("candidate"),
        source_commit="e" * 40,
        route_id=matrix.LOW_ADAPTIVE_NATIVE_ROUTE,
    )
    base = {
        "lane_id": lane.lane_id,
        "source_commit": lane.source_commit,
        "source_status": [],
        "route_id": lane.route_id,
        "performance_profile": "low",
        "workload": "low",
        "prompt_tokens": 1_024,
        "conditioner_output_tokens": 1_024,
        "conditioner_generated_tokens": 1_024,
        "conditioner_finish_reason": "length",
        "max_tokens": 1_024,
        "generated_tokens": 1_024,
        "finish_reason": "length",
        "stop_token_policy": "disabled_for_exact_output",
        "mlx_version": "0.32.2",
        "mlx_metal_version": "0.32.2",
        "gpu_lock_scope": "attested_parent",
        "source_import_attested": True,
        "model_id": matrix.MODEL_ID,
        "model_artifact_hashes": {
            "config.json": "a" * 64,
            "mtp.safetensors": "b" * 64,
        },
        "verify_strategy": "capture_commit",
        "verify_core": "linear-gdn-from-conv-tape",
        "speculative_depth": 3,
        "requested_speculative_depth": 3,
        "optimized_stack": _optimized_stack_receipt(),
        "sampler": {
            "target_temperature": 1.0,
            "draft_temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
        },
        "prompt_token_sha256": matrix.PROMPT_TOKEN_SHA256["low"][1_024],
        "prompt_artifact_sha256": matrix.PROMPT_ARTIFACT_SHA256["python"],
        "context_artifact_sha256": matrix.PYTHON_CONTEXT_SHA256,
        "row17_artifact_sha256": matrix.ROW17_ARTIFACT_SHA256,
        "source_rows": [8, 10, 20, 21, 24, 26, 53, 11],
        "installed_route_id": matrix.LOW_BF16_OPTIMIZED_INSTALLED_ROUTE_ID,
        "kernel_ids": list(matrix.LOW_BF16_OPTIMIZED_KERNEL_IDS),
        "feature_receipt": {
            key: {"active": True}
            for key in matrix.LOW_BF16_OPTIMIZED_FEATURE_KEYS
        },
        "adaptive_policy_receipt": {
            "kind": "position_ema",
            "executed": True,
            "initial_accept_ema": [0.5, 0.5, 0.5],
            "final_accept_ema": [0.6, 0.5, 0.4],
            "initial_depth": 3,
            "final_depth": 2,
            "max_depth": 3,
            "depth_cap": 3,
        },
        "device_core_receipt": {
            "requested": "device",
            "device_calls": 400,
            "device_fallbacks": 0,
        },
        "compiled_verify_receipt": {
            "mode": "on",
            "compiled_calls": 400,
            "fallback_calls": 0,
            "fallback_reasons": {},
            "permanent_eager": False,
            "permanent_eager_reason": None,
        },
        "history_route_receipt": {
            "route_id": "stock_history",
            "prompt_tokens": 1_024,
            "row20_engaged": False,
        },
        "draft_core": "device",
        "drafted_by_depth": [500, 300, 100],
        "accepted_by_depth": [300, 150, 50],
        "verify_calls": 500,
        "attempted_depth_schedule": [0] * 24
        + [1] * 200
        + [2] * 200
        + [3] * 100,
    }
    base["depth_usage"] = matrix.depth_usage(
        decode_cycles=len(base["attempted_depth_schedule"]),
        verify_calls=base["verify_calls"],
        drafted_by_depth=base["drafted_by_depth"],
        accepted_by_depth=base["accepted_by_depth"],
    )
    base["feature_receipt"]["r53_command_buffers"].update(
        {"max_mb_per_buffer": 512, "max_ops_per_buffer": 50}
    )
    assert matrix.receipt_errors(
        base, lane=lane, context_tokens=1_024, output_tokens=1_024
    ) == []

    broken = json.loads(json.dumps(base))
    broken["depth_usage"]["attempted_counts"]["D3"] += 1
    errors = matrix.receipt_errors(
        broken, lane=lane, context_tokens=1_024, output_tokens=1_024
    )
    assert "adaptive depth usage does not match raw histograms" in errors

    historical = json.loads(json.dumps(base))
    historical["prompt_tokens"] = 131_072
    historical["prompt_token_sha256"] = matrix.PROMPT_TOKEN_SHA256["low"][131_072]
    historical["optimized_stack"]["runtime_env"]["MTPLX_DROP_EVENTS"] = "0"
    historical["attempted_depth_schedule"] = [3] * 500
    historical["accepted_depth_schedule"] = (
        [0] * 200 + [1] * 150 + [2] * 100 + [3] * 50
    )
    errors = matrix.receipt_errors(
        historical, lane=lane, context_tokens=131_072, output_tokens=1_024
    )
    assert "adaptive depth usage does not match raw histograms" not in errors
    assert not any(error.startswith("adaptive depth usage is invalid") for error in errors)

    broken = json.loads(json.dumps(base))
    del broken["adaptive_policy_receipt"]["final_accept_ema"]
    errors = matrix.receipt_errors(
        broken, lane=lane, context_tokens=1_024, output_tokens=1_024
    )
    assert "adaptive policy state receipt is incomplete" in errors

    broken = json.loads(json.dumps(base))
    broken["device_core_receipt"]["device_calls"] = 0
    errors = matrix.receipt_errors(
        broken, lane=lane, context_tokens=1_024, output_tokens=1_024
    )
    assert "adaptive device draft core did not engage without fallback" in errors

    broken = json.loads(json.dumps(base))
    broken["compiled_verify_receipt"]["fallback_reasons"] = {
        "exception:ValueError": 1
    }
    errors = matrix.receipt_errors(
        broken, lane=lane, context_tokens=1_024, output_tokens=1_024
    )
    assert "adaptive compiled verification did not engage cleanly" in errors


def test_parent_preflight_reads_both_distribution_versions_from_selected_python(
    monkeypatch, tmp_path: Path
) -> None:
    matrix = _module()
    observed: dict[str, object] = {}

    def fake_check_output(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return '{"mlx": "0.32.2", "mlx_metal": "0.32.2"}\n'

    monkeypatch.setattr(matrix.subprocess, "check_output", fake_check_output)
    versions = matrix._interpreter_versions(tmp_path / "python")

    assert versions == {"mlx": "0.32.2", "mlx_metal": "0.32.2"}
    assert observed["command"][0] == str((tmp_path / "python").resolve())
    assert "mlx-metal" in observed["command"][2]


def test_parent_preflight_does_not_dereference_virtualenv_python_symlink(
    monkeypatch, tmp_path: Path
) -> None:
    matrix = _module()
    selected_python = tmp_path / "venv" / "bin" / "python"
    selected_python.parent.mkdir(parents=True)
    selected_python.symlink_to(sys.executable)
    observed: dict[str, object] = {}

    def fake_check_output(command, **kwargs):
        observed["command"] = command
        return '{"mlx": "0.32.2", "mlx_metal": "0.32.2"}\n'

    monkeypatch.setattr(matrix.subprocess, "check_output", fake_check_output)
    matrix._interpreter_versions(selected_python)

    assert observed["command"][0] == str(selected_python.absolute())


def test_matrix_parent_accepts_direct_or_delegated_guard_ownership() -> None:
    matrix = _module()

    assert matrix._validated_parent_guard_scope("direct") == "direct"
    assert matrix._validated_parent_guard_scope("attested_parent") == "attested_parent"


def test_matrix_entrypoint_imports_from_outside_repository(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
