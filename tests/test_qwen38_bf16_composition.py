from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts import qwen38_bf16_composition as composition


def _step() -> composition.CompositionStep:
    return composition.CompositionStep(
        step_id="decode-add-r61",
        feature="r61_dual_norm_concat",
        phase="decode",
        control_route="r53_command_buffers",
        candidate_route="r53_command_buffers+r61_dual_norm_concat",
        allow_frozen_candidate=False,
    )


def test_composition_step_stacks_one_feature_at_16k_only() -> None:
    step = _step()

    delta = composition.validate_step(step)
    plan = composition.build_execution_plan(step)

    assert delta.candidate_feature == "r61_dual_norm_concat"
    assert [item.context_tokens for item in plan] == [16_384]
    assert all(item.control_route == step.control_route for item in plan)
    assert all(item.candidate_route == step.candidate_route for item in plan)
    assert all(item.feature == step.feature for item in plan)


def test_composition_step_rejects_a_feature_label_that_is_not_the_delta() -> None:
    step = composition.CompositionStep(
        step_id="bad-label",
        feature="r20_kv_only_history",
        phase="decode",
        control_route="r53_command_buffers",
        candidate_route="r53_command_buffers+r61_dual_norm_concat",
        allow_frozen_candidate=False,
    )

    with pytest.raises(ValueError, match="feature label"):
        composition.validate_step(step)


def test_composition_step_requires_explicit_opt_in_for_frozen_addition() -> None:
    step = composition.CompositionStep(
        step_id="prefill-add-r50",
        feature="r50_wired_residency",
        phase="prefill",
        control_route="r53_command_buffers+r61_dual_norm_concat",
        candidate_route=(
            "r53_command_buffers+r61_dual_norm_concat+r50_wired_residency"
        ),
        allow_frozen_candidate=False,
    )

    with pytest.raises(composition.candidates.NativeMTPRouteError):
        composition.validate_step(step)

    opted_in = composition.CompositionStep(
        **{
            **step.__dict__,
            "allow_frozen_candidate": True,
        }
    )
    assert composition.validate_step(opted_in).candidate_feature == (
        "r50_wired_residency"
    )


@pytest.mark.parametrize(
    ("control_route", "candidate_route", "feature"),
    [
        ("control", "r17_q4_mtp_block", "r17_q4_mtp_block"),
        (
            "r17_q4_mtp_block",
            "r17_q4_mtp_block+r20_kv_only_history",
            "r20_kv_only_history",
        ),
    ],
)
def test_composition_step_rejects_q4_in_candidate_or_parent_route(
    control_route: str,
    candidate_route: str,
    feature: str,
) -> None:
    step = composition.CompositionStep(
        step_id="q4-is-not-bf16",
        feature=feature,
        phase="decode",
        control_route=control_route,
        candidate_route=candidate_route,
        allow_frozen_candidate=False,
    )

    with pytest.raises(ValueError, match="BF16-only"):
        composition.validate_step(step)


def test_step_from_args_preserves_the_phase_and_routes() -> None:
    args = argparse.Namespace(
        step_id="prefill-add-r20",
        feature="r20_kv_only_history",
        phase="prefill",
        control_route="r53_command_buffers+r61_dual_norm_concat",
        candidate_route=(
            "r53_command_buffers+r61_dual_norm_concat+r20_kv_only_history"
        ),
        allow_frozen_candidate=False,
    )

    step = composition.step_from_args(args)

    assert step == composition.CompositionStep(
        step_id="prefill-add-r20",
        feature="r20_kv_only_history",
        phase="prefill",
        control_route="r53_command_buffers+r61_dual_norm_concat",
        candidate_route=(
            "r53_command_buffers+r61_dual_norm_concat+r20_kv_only_history"
        ),
        allow_frozen_candidate=False,
    )


def test_campaign_payload_is_complete_after_the_16k_result() -> None:
    step = _step()
    plan = composition.build_execution_plan(step)
    common = {
        "step": step,
        "plan": plan,
        "source_commit": "a" * 40,
        "model_artifact_hashes": {"config.json": "b" * 64},
        "lock_scope": "attested_parent",
    }

    complete = composition.campaign_payload(
        results=[{"context_tokens": 16_384}],
        **common,
    )

    assert complete["complete"] is True
    assert complete["phase"] == "decode"
    assert complete["protocol"]["contexts"] == [16_384]


def test_output_paths_require_a_filename_safe_step_id(tmp_path: Path) -> None:
    step = composition.CompositionStep(
        **{**_step().__dict__, "step_id": "../escape"}
    )

    with pytest.raises(ValueError, match="filename-safe"):
        composition.output_paths(
            step=step,
            plan=composition.build_execution_plan(_step()),
            output_dir=tmp_path / "receipts",
            output=tmp_path / "receipts" / "campaign.json",
        )


def test_output_paths_reject_campaign_raw_and_temporary_collisions(
    tmp_path: Path,
) -> None:
    step = _step()
    plan = composition.build_execution_plan(step)
    output_dir = tmp_path / "receipts"
    first_raw = output_dir / "01-decode-add-r61-16384.json"

    with pytest.raises(ValueError, match="distinct"):
        composition.output_paths(
            step=step,
            plan=plan,
            output_dir=output_dir,
            output=first_raw,
        )


def test_output_paths_reject_any_existing_nested_content(tmp_path: Path) -> None:
    step = _step()
    plan = composition.build_execution_plan(step)
    output_dir = tmp_path / "receipts"
    nested = output_dir / "old" / "receipt.tmp"
    nested.parent.mkdir(parents=True)
    nested.write_text("stale", encoding="utf-8")

    with pytest.raises(RuntimeError, match="empty"):
        composition.output_paths(
            step=step,
            plan=plan,
            output_dir=output_dir,
            output=output_dir / "campaign.json",
        )


def test_output_paths_reject_a_file_valued_output_directory(
    tmp_path: Path,
) -> None:
    step = _step()
    plan = composition.build_execution_plan(step)
    output_dir = tmp_path / "receipts"
    output_dir.write_text("not a directory", encoding="utf-8")

    with pytest.raises(RuntimeError, match="must be a directory"):
        composition.output_paths(
            step=step,
            plan=plan,
            output_dir=output_dir,
            output=tmp_path / "campaign.json",
        )


def test_output_paths_reject_a_file_in_the_summary_parent_chain(
    tmp_path: Path,
) -> None:
    step = _step()
    plan = composition.build_execution_plan(step)
    blocked_parent = tmp_path / "blocked"
    blocked_parent.write_text("not a directory", encoding="utf-8")

    with pytest.raises(RuntimeError, match="existing parent must be a directory"):
        composition.output_paths(
            step=step,
            plan=plan,
            output_dir=tmp_path / "receipts",
            output=blocked_parent / "campaign.json",
        )
