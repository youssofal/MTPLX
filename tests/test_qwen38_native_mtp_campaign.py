from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/qwen38_native_mtp_campaign.py"


def _module():
    spec = importlib.util.spec_from_file_location("qwen38_native_mtp_campaign", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _exact_args(**overrides):
    campaign = _module()
    values = {
        "prompt_file": campaign.DEFAULT_PROMPT,
        "context_file": campaign.DEFAULT_CONTEXT,
        "max_tokens": 1024,
        "warmup_tokens": 1024,
        "seed": 42,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "lock": Path("/tmp/mtplx-gpu-exclusive.lock"),
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _bracket(
    *,
    context: int,
    feature: str,
    improvements: dict[str, float],
    throughput_improvements: dict[str, float] | None = None,
    row20_engaged: bool | None = None,
):
    candidate_runs = []
    for _ in range(2):
        run = {"route_id": feature, "feature_receipt": {}}
        if row20_engaged is not None:
            run["history_route_receipt"] = {
                "prompt_tokens": context,
                "route_id": "kv_only_history" if row20_engaged else "stock_history",
                "row20_engaged": row20_engaged,
            }
        candidate_runs.append(run)
    return {
        "prompt_token_target": context,
        "candidate_feature": feature,
        "phase_summary": {
            "time_improvement_pct": dict(improvements),
            "throughput_improvement_pct": (
                dict(improvements)
                if throughput_improvements is None
                else dict(throughput_improvements)
            ),
        },
        "candidate_engagement_errors": [],
        "correctness": {"passed": True},
        "receipt_invariant_errors": [],
        "arms": [
            {"route_id": "control"},
            candidate_runs[0],
            candidate_runs[1],
            {"route_id": "control"},
        ],
    }


def test_exact_workload_contract_has_two_contexts_and_first_python_row() -> None:
    campaign = _module()

    assert campaign.EXACT_CONTEXT_TOKENS == (1024, 16_384)
    assert campaign.EXACT_OUTPUT_TOKENS == 1024
    assert campaign.EXACT_CONDITIONER_TOKENS == 1024
    assert campaign.DEFAULT_PROMPT.name == "python_modules_long.jsonl"
    assert campaign.DEFAULT_CONTEXT.name == "generation.py"
    assert campaign._exact_workload_errors(_exact_args()) == []
    assert campaign._read_first_prompt_id(campaign.DEFAULT_PROMPT)

    errors = campaign._exact_workload_errors(_exact_args(top_p=0.9))
    assert any("top-p" in error for error in errors)


@pytest.mark.parametrize("feature", ["r08_device_draft", "r10_compact_vocab"])
def test_pure_decode_candidates_require_decode_and_wall_wins_at_each_context(
    feature: str,
) -> None:
    campaign = _module()
    winner = _bracket(
        context=1024,
        feature=feature,
        improvements={"wall": 0.06, "mtp_decode": 0.06, "mtp_history": -2.0},
    )
    neutral = _bracket(
        context=1024,
        feature=feature,
        improvements={"wall": 0.06, "mtp_decode": 0.05, "mtp_history": 9.0},
    )

    assert campaign._context_decision(feature, winner)["passed"] is True
    assert campaign._context_decision(feature, neutral)["passed"] is False


def test_adaptive_depth_gates_output_decode_not_variable_draft_work_rate() -> None:
    campaign = _module()
    receipt = _bracket(
        context=1024,
        feature="r11_position_ema",
        improvements={"wall": 2.0, "decode": 3.0, "mtp_decode": -1.0},
    )

    assert campaign._context_decision("r11_position_ema", receipt)["passed"]
    receipt["phase_summary"]["throughput_improvement_pct"]["decode"] = 0.05
    assert not campaign._context_decision("r11_position_ema", receipt)["passed"]


@pytest.mark.parametrize("feature", ["r17_q4_mtp_block", "r28_q4_mtp_block", "r36_qkv_islands", "r61_dual_norm_concat", "r63_q8_embedding_dual_norm"])
def test_shared_candidates_gate_only_wall_and_decode_at_1k(feature: str) -> None:
    campaign = _module()
    receipt = _bracket(
        context=1024,
        feature=feature,
        improvements={"wall": 0.08, "mtp_history": 0.07, "mtp_decode": 0.06},
    )
    assert campaign._context_decision(feature, receipt)["passed"] is True
    receipt["phase_summary"]["throughput_improvement_pct"][
        "mtp_history"
    ] = -50.0
    decision = campaign._context_decision(feature, receipt)
    assert decision["passed"] is True
    assert decision["phase_contract"]["mtp_history"] == "full"


def test_row20_is_stock_at_1k_then_wins_history_and_wall_at_16k() -> None:
    campaign = _module()
    short = _bracket(
        context=1024,
        feature="r20_kv_only_history",
        improvements={"wall": 0.0, "mtp_history": 0.0, "mtp_decode": 0.0},
        row20_engaged=False,
    )
    long = _bracket(
        context=16_384,
        feature="r20_kv_only_history",
        improvements={"wall": 0.06, "mtp_history": 0.07, "mtp_decode": 0.0},
        row20_engaged=True,
    )

    assert campaign._context_decision("r20_kv_only_history", short)["passed"]
    assert campaign._context_decision("r20_kv_only_history", long)["passed"]
    long["phase_summary"]["throughput_improvement_pct"][
        "mtp_decode"
    ] = -0.000001
    assert not campaign._context_decision("r20_kv_only_history", long)["passed"]


def test_row20_long_context_marks_block_partial_and_input_history_bypass() -> None:
    campaign = _module()
    parent = "r20_kv_only_history"

    assert campaign._phase_contract(
        "r36_qkv_islands", 16_384, control_features={parent}
    )["mtp_history"] == "partial"
    assert campaign._phase_contract(
        "r61_dual_norm_concat", 16_384, control_features={parent}
    )["mtp_history"] == "bypass"
    assert campaign._phase_contract(
        "r61_dual_norm_concat", 1024, control_features={parent}
    )["mtp_history"] == "full"

    block = _bracket(
        context=16_384,
        feature="r36_qkv_islands",
        improvements={"wall": 0.08, "mtp_history": 0.05, "mtp_decode": 0.08},
    )
    input_route = _bracket(
        context=16_384,
        feature="r61_dual_norm_concat",
        improvements={"wall": 0.08, "mtp_history": -5.0, "mtp_decode": 0.08},
    )
    assert campaign._context_decision(
        "r36_qkv_islands", block, control_features={parent}
    )["passed"]
    assert campaign._context_decision(
        "r61_dual_norm_concat", input_route, control_features={parent}
    )["passed"]


def test_campaign_aborts_after_first_completed_neutral_context() -> None:
    campaign = _module()
    losing = _bracket(
        context=1024,
        feature="r08_device_draft",
        improvements={"wall": 0.05, "mtp_decode": 0.08, "mtp_history": 0.0},
    )
    called = []

    result = campaign._run_contexts(
        "r08_device_draft",
        lambda context: called.append(context) or losing,
    )

    assert called == [1024]
    assert result["aborted_after_context"] == 1024
    assert result["contexts"][0]["decision"]["passed"] is False


def test_audit_mode_completes_both_contexts_after_a_short_context_loss() -> None:
    campaign = _module()
    losing = _bracket(
        context=1024,
        feature="r08_device_draft",
        improvements={"wall": -1.0, "mtp_decode": -1.0, "mtp_history": 0.0},
    )
    winning = _bracket(
        context=16_384,
        feature="r08_device_draft",
        improvements={"wall": 1.0, "mtp_decode": 1.0, "mtp_history": 0.0},
    )
    called = []

    result = campaign._run_contexts(
        "r08_device_draft",
        lambda context: called.append(context)
        or (losing if context == 1024 else winning),
        stop_on_failure=False,
    )

    assert called == [1024, 16_384]
    assert result["aborted_after_context"] is None
    assert [item["decision"]["passed"] for item in result["contexts"]] == [
        False,
        True,
    ]


def test_lower_proposer_time_with_worse_throughput_is_rejected() -> None:
    campaign = _module()
    arms = []
    for route, draft_time, draft_tokens in (
        ("control", 2.0, 400),
        ("candidate", 1.0, 100),
        ("candidate", 1.0, 100),
        ("control", 2.0, 400),
    ):
        arms.append(
            {
                "route_id": route,
                "wall_s": 10.0 if route == "control" else 9.0,
                "target_prefill_time_s": 1.0,
                "target_prefill_tok_s": 1024.0,
                "mtp_history_time_s": 1.0,
                    "mtp_history_tok_s": 1024.0,
                    "mtp_decode_time_s": draft_time,
                    "mtp_decode_tok_s": draft_tokens / draft_time,
                    "decode_elapsed_s": 8.0,
                    "decode_tok_s": 128.0,
                }
        )
    summary = campaign.gate._phase_summary(
        arms, control_id="control", candidate_id="candidate"
    )
    receipt = _bracket(
        context=1024,
        feature="r08_device_draft",
        improvements=summary["time_improvement_pct"],
        throughput_improvements=summary["throughput_improvement_pct"],
    )

    assert summary["time_improvement_pct"]["mtp_decode"] > 0.05
    assert summary["throughput_improvement_pct"]["mtp_decode"] < 0.0
    assert not campaign._context_decision("r08_device_draft", receipt)["passed"]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_context_decision_rejects_nonfinite_phase_improvements(invalid: float) -> None:
    campaign = _module()
    receipt = _bracket(
        context=1024,
        feature="r08_device_draft",
        improvements={"wall": 0.1, "mtp_decode": 0.1, "mtp_history": 0.1},
        throughput_improvements={
            "wall": 0.1,
            "mtp_decode": invalid,
            "mtp_history": 0.1,
        },
    )

    decision = campaign._context_decision("r08_device_draft", receipt)

    assert decision["passed"] is False
    assert any("finite" in error for error in decision["errors"])


def test_context_decision_rejects_dirty_aggregate_source_status() -> None:
    campaign = _module()
    receipt = _bracket(
        context=1024,
        feature="r20_kv_only_history",
        improvements={"wall": 0.0, "mtp_decode": 0.0, "mtp_history": 0.0},
        row20_engaged=False,
    )
    receipt["source_status"] = [" M mtplx/runtime.py"]

    decision = campaign._context_decision("r20_kv_only_history", receipt)

    assert decision["passed"] is False
    assert any("source tree" in error for error in decision["errors"])
