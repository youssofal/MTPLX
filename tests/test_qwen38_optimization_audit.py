from __future__ import annotations

import argparse

from scripts import qwen38_challenge_inventory as inventory
from scripts import qwen38_native_mtp_candidates as candidates
from scripts import qwen38_optimization_audit as audit


def test_audit_covers_all_54_qualifying_rows_once() -> None:
    source = inventory.load_inventory(
        inventory.DEFAULT_RECEIPT, inventory.DEFAULT_DESIGN
    )
    qualifying = inventory.validate_inventory(source).qualifying_rows

    rows = audit.build_audit_rows(qualifying)

    assert len(rows) == 54
    assert [row.ordinal for row in rows] == [row.ordinal for row in qualifying]
    assert len({row.ordinal for row in rows}) == 54
    assert all(row.disposition_kind for row in rows)
    assert all(row.reason for row in rows)


def test_direct_cases_cover_every_current_optimized_switch() -> None:
    expected_features = set(candidates.NATIVE_MTP_CANDIDATES) | {
        item.feature for item in candidates.FROZEN_TARGET_SUBSTRATE.values()
    }

    assert len(audit.DIRECT_CASES) == 16
    assert {case.feature for case in audit.DIRECT_CASES} == expected_features
    assert len({case.case_id for case in audit.DIRECT_CASES}) == 16

    for case in audit.DIRECT_CASES:
        delta = candidates.validate_native_mtp_route_delta(
            case.control_route,
            case.candidate_route,
            allow_frozen_candidate=case.allow_frozen_candidate,
        )
        assert delta.candidate_feature == case.feature


def test_every_direct_case_runs_both_exact_contexts_without_early_abort() -> None:
    plan = audit.build_execution_plan(audit.DIRECT_CASES)

    assert len(plan) == 32
    assert {item.context_tokens for item in plan} == {1024, 16_384}
    for case in audit.DIRECT_CASES:
        assert [
            item.context_tokens for item in plan if item.case_id == case.case_id
        ] == [1024, 16_384]


def test_prompt_hashes_are_pinned_for_this_gate_prompt_constructor() -> None:
    assert audit.EXPECTED_PROMPT_TOKEN_SHA256 == {
        1_024: "3015401ec3e421502b1a23f18d0a6e5d53004b189fdbab0e2e3ba27802fcd7e6",
        16_384: "af141694261c1d3c4d8aa6e36e903fa55fae08e2fc3ad21ad78ebcde213f6954",
    }


def test_historical_rows_point_only_to_real_direct_cases() -> None:
    source = inventory.load_inventory(
        inventory.DEFAULT_RECEIPT, inventory.DEFAULT_DESIGN
    )
    rows = audit.build_audit_rows(
        inventory.validate_inventory(source).qualifying_rows
    )
    case_ids = {case.case_id for case in audit.DIRECT_CASES}

    assert {row.direct_case_id for row in rows if row.direct_case_id} <= case_ids
    assert {row.ordinal for row in rows if row.disposition_kind == "direct-abba"} == {
        8,
        10,
        11,
        17,
        18,
        20,
        21,
        24,
        26,
        28,
        36,
        48,
        50,
        53,
        61,
        63,
    }
    consolidated = {
        row.ordinal: row.direct_case_id
        for row in rows
        if row.disposition_kind == "consolidated-to"
    }
    assert consolidated == {
        33: "r36-qkv-islands",
        45: "r48-boundary-fused",
        60: "r61-dual-norm-concat",
    }


def test_row26_runs_only_on_its_required_row24_parent() -> None:
    row24 = next(
        item for item in audit.DIRECT_CASES if item.feature == "r24_eval_ladder"
    )
    case = next(
        item for item in audit.DIRECT_CASES if item.feature == "r26_prefill_ladder_3"
    )

    assert row24.control_route == "r21_qk_rms_rope"
    assert row24.candidate_route == "r21_qk_rms_rope+r24_eval_ladder"
    assert case.control_route == "r21_qk_rms_rope+r24_eval_ladder"
    assert case.candidate_route == (
        "r21_qk_rms_rope+r24_eval_ladder+r26_prefill_ladder_3"
    )


def test_compact_vocab_runs_only_on_its_required_device_draft_parent() -> None:
    case = next(
        item for item in audit.DIRECT_CASES if item.feature == "r10_compact_vocab"
    )

    assert case.control_route == "r08_device_draft"
    assert case.candidate_route == "r08_device_draft+r10_compact_vocab"


def test_route_contract_proves_exact_installation_and_negative_feature_state() -> None:
    for item in audit.build_execution_plan(audit.DIRECT_CASES):
        control = audit.expected_route_contract(item.control_route)
        candidate = audit.expected_route_contract(item.candidate_route)

        assert item.feature not in control.requested_features
        assert item.feature in candidate.requested_features
        assert control.installed_route_id
        assert candidate.installed_route_id
        assert isinstance(control.kernel_ids, tuple)
        assert isinstance(candidate.kernel_ids, tuple)


def test_bf16_first_order_defers_custom_q4_block_artifacts() -> None:
    artifact_features = {
        "r17_q4_mtp_block",
        "r28_q4_mtp_block",
        "r36_qkv_islands",
    }
    first_artifact = next(
        index
        for index, case in enumerate(audit.DIRECT_CASES)
        if case.feature in artifact_features
    )

    assert all(
        case.feature not in artifact_features
        for case in audit.DIRECT_CASES[:first_artifact]
    )
    assert {
        case.feature for case in audit.DIRECT_CASES[first_artifact:]
    } == artifact_features


def test_isolated_command_carries_exact_protocol_and_frozen_opt_in(tmp_path) -> None:
    case = next(
        item for item in audit.DIRECT_CASES if item.feature == "r18_gdn_decay_memo"
    )
    args = argparse.Namespace(
        model=tmp_path / "model",
        prompt_file=audit.DEFAULT_PROMPT,
        context_file=audit.DEFAULT_CONTEXT,
        max_tokens=1024,
        warmup_tokens=1024,
        seed=42,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        lock=audit.DEFAULT_LOCK,
        row17_artifact=None,
        row28_artifact=None,
        row36_artifact=None,
    )

    command = audit._isolated_command(
        args,
        audit.ExecutionItem(
            case_id=case.case_id,
            feature=case.feature,
            control_route=case.control_route,
            candidate_route=case.candidate_route,
            context_tokens=16_384,
            allow_frozen_candidate=True,
        ),
        tmp_path / "receipt.json",
    )

    assert command[0] == audit.sys.executable
    assert "--allow-frozen-candidate" in command
    assert command[command.index("--prompt-tokens") + 1] == "16384"
    assert command[command.index("--max-tokens") + 1] == "1024"
    assert command[command.index("--warmup-tokens") + 1] == "1024"
    assert command[command.index("--order") + 1] == (
        "control,r18_gdn_decay_memo,r18_gdn_decay_memo,control"
    )


def test_receipt_validation_accepts_a_slow_but_exact_candidate() -> None:
    item = audit.build_execution_plan(audit.DIRECT_CASES)[0]
    order = [
        item.control_route,
        item.candidate_route,
        item.candidate_route,
        item.control_route,
    ]
    arms = []
    for route in order:
        contract = audit.expected_route_contract(route)
        arms.append(
            {
                "generated_tokens": 1024,
                "route_id": route,
                "installed_route_id": contract.installed_route_id,
                "kernel_ids": list(contract.kernel_ids),
                "feature_receipt": {
                    key: {"active": True}
                    for key in contract.feature_receipt_keys
                },
                "draft_core": contract.draft_core,
                "adaptive_policy_receipt": (
                    {"kind": "position_ema", "executed": True}
                    if contract.adaptive
                    else None
                ),
            }
        )
    receipt = {
        "prompt_token_target": item.context_tokens,
        "max_tokens": 1024,
        "seed": 42,
        "target_temperature": 1.0,
        "draft_temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "control_route_id": item.control_route,
        "candidate_route_id": item.candidate_route,
        "candidate_feature": item.feature,
        "timed_arm_count": 4,
        "order": order,
        "source_status": [],
        "source_commit": "abc",
        "receipt_invariant_errors": [],
        "candidate_engagement_errors": [],
        "correctness": {"passed": True},
        "mlx_version": "0.32.2",
        "mlx_metal_version": "0.32.2",
        "model_artifact_hashes": {"config.json": "a" * 64},
        "prompt_token_sha256": audit.EXPECTED_PROMPT_TOKEN_SHA256[
            item.context_tokens
        ],
        "phase_summary": {
            "time_improvement_pct": {"wall": -2.0},
            "throughput_improvement_pct": {"decode": -2.0},
        },
        "arms": arms,
    }

    assert audit._receipt_errors(item, receipt) == []
    receipt["arms"][0] = {**receipt["arms"][0], "generated_tokens": 1000}
    assert "all four arms must generate exactly 1024 tokens" in audit._receipt_errors(
        item, receipt
    )

    receipt["arms"][0] = {**receipt["arms"][0], "generated_tokens": 1024}
    receipt["prompt_token_sha256"] = "0" * 64
    assert "prompt token hash does not match the exact workload" in audit._receipt_errors(
        item, receipt
    )

    receipt["prompt_token_sha256"] = audit.EXPECTED_PROMPT_TOKEN_SHA256[
        item.context_tokens
    ]
    assert audit._receipt_errors(
        item,
        receipt,
        expected_source_commit="abc",
        expected_model_artifact_hashes={"config.json": "a" * 64},
    ) == []
    assert "source commit does not match the audit parent" in audit._receipt_errors(
        item,
        receipt,
        expected_source_commit="def",
        expected_model_artifact_hashes={"config.json": "a" * 64},
    )

    receipt["arms"][1] = {
        **receipt["arms"][1],
        "installed_route_id": "wrong",
    }
    assert any(
        "installed route" in error for error in audit._receipt_errors(item, receipt)
    )


def test_parent_hashes_model_once_even_under_outer_guard(monkeypatch, tmp_path) -> None:
    model = tmp_path / "model"
    observed = []
    monkeypatch.setattr(
        audit.gate,
        "_model_artifact_hashes",
        lambda path: observed.append(path) or {"config.json": "a" * 64},
    )
    monkeypatch.setattr(
        audit.gate,
        "_attested_model_artifact_hashes",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("outer guard does not provide model hash attestation")
        ),
    )

    hashes = audit._parent_model_artifact_hashes(model)

    assert hashes == {"config.json": "a" * 64}
    assert observed == [model]
