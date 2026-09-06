from __future__ import annotations

import json
from pathlib import Path

import pytest

from mtplx.replay_orchestrator import (
    CaptureFilter,
    ReplayOrchestrator,
    ReplayPlanConfig,
    ReplayPlanError,
    StaleReplayPlanError,
)


def write_capture(root: Path, name: str, payload: dict) -> Path:
    path = root / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_plan_selection_is_deterministic_deduplicated_and_bounded(tmp_path):
    write_capture(
        tmp_path,
        "a",
        {
            "request_id": "a",
            "model": "m",
            "request": {"payload": {"prompt": "one"}},
            "request_fingerprint": "same",
            "prompt_tokens": 4,
        },
    )
    write_capture(
        tmp_path,
        "b",
        {
            "request_id": "b",
            "model": "m",
            "request": {"payload": {"prompt": "duplicate"}},
            "request_fingerprint": "same",
            "prompt_tokens": 5,
        },
    )
    write_capture(
        tmp_path,
        "c",
        {
            "request_id": "c",
            "model": "m",
            "request": {"payload": {"prompt": "two"}},
            "request_fingerprint": "different",
            "prompt_tokens": 6,
        },
    )
    orchestrator = ReplayOrchestrator(
        tmp_path,
        config=ReplayPlanConfig(maximum_cases=2, deterministic_seed="fixed"),
    )
    first = orchestrator.build_plan(CaptureFilter(model="m"))
    second = orchestrator.build_plan(CaptureFilter(model="m"))
    assert [item.capture_id for item in first.cases] == [
        item.capture_id for item in second.cases
    ]
    assert len(first.cases) == 2
    assert first.duplicate_count == 1
    assert first.source_digest == second.source_digest


def test_redacted_count_only_capture_is_not_treated_as_replayable(tmp_path):
    write_capture(
        tmp_path,
        "redacted",
        {
            "request_id": "redacted",
            "prompt_tokens": 42,
            "prompt": {"content_redacted": True, "content_sha256": "x" * 64},
        },
    )
    plan = ReplayOrchestrator(tmp_path).build_plan()
    assert len(plan.cases) == 1
    assert plan.cases[0].replayable is False
    assert plan.cases[0].unavailable_reason == "request_content_not_captured"


def test_stale_capture_is_rejected_before_receipt_write(tmp_path):
    capture = write_capture(
        tmp_path,
        "case",
        {"request_id": "case", "request": {"payload": {"prompt": "one"}}},
    )
    receipts = tmp_path / "receipts"
    orchestrator = ReplayOrchestrator(tmp_path, receipt_directory=receipts)
    plan = orchestrator.build_plan()
    capture.write_text(
        json.dumps(
            {"request_id": "case", "request": {"payload": {"prompt": "changed"}}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(StaleReplayPlanError):
        orchestrator.record_receipt(
            plan,
            candidate_name="candidate",
            report={"passed": True},
            decision={"promote": True},
        )
    assert not receipts.exists()


def test_new_capture_that_changes_selection_makes_plan_stale(tmp_path):
    write_capture(tmp_path, "first", {"request_id": "first"})
    orchestrator = ReplayOrchestrator(tmp_path)
    plan = orchestrator.build_plan()
    write_capture(tmp_path, "second", {"request_id": "second"})
    with pytest.raises(StaleReplayPlanError):
        orchestrator.assert_fresh(plan)


def test_receipt_is_atomic_private_by_default_and_listable(tmp_path):
    write_capture(
        tmp_path,
        "case",
        {
            "request_id": "case",
            "request": {"payload": {"prompt": "private"}},
            "outcome": {"response": {"answer": "baseline"}},
        },
    )
    receipts = tmp_path / "receipts"
    orchestrator = ReplayOrchestrator(tmp_path, receipt_directory=receipts)
    plan = orchestrator.build_plan()
    receipt = orchestrator.record_receipt(
        plan,
        candidate_name="candidate",
        report={"pass_rate": 1.0},
        decision={"promote": True, "reasons": []},
    )
    files = list(receipts.glob("*.json"))
    assert len(files) == 1
    assert list(receipts.glob("*.tmp")) == []
    raw = files[0].read_text(encoding="utf-8")
    assert "private" not in raw
    assert receipt.receipt_id in raw
    assert receipt.promotion_applied is False
    assert orchestrator.list_receipts(limit=1)[0]["receipt_id"] == receipt.receipt_id


def test_empty_candidate_name_is_rejected(tmp_path):
    write_capture(tmp_path, "case", {"request_id": "case"})
    orchestrator = ReplayOrchestrator(tmp_path)
    plan = orchestrator.build_plan()
    with pytest.raises(ReplayPlanError, match="candidate_name"):
        orchestrator.record_receipt(
            plan,
            candidate_name=" ",
            report={},
            decision={},
        )


def test_plan_from_another_capture_root_is_rejected(tmp_path):
    source = tmp_path / "source"
    other = tmp_path / "other"
    source.mkdir()
    other.mkdir()
    write_capture(source, "case", {"request_id": "case"})
    plan = ReplayOrchestrator(source).build_plan()
    with pytest.raises(ReplayPlanError, match="capture root"):
        ReplayOrchestrator(other).assert_fresh(plan)


def test_filters_cover_error_and_prompt_size(tmp_path):
    write_capture(
        tmp_path,
        "small",
        {
            "request_id": "small",
            "prompt_tokens": 2,
            "request": {"payload": {"prompt": "a"}},
        },
    )
    write_capture(
        tmp_path,
        "large-error",
        {
            "request_id": "large-error",
            "prompt_tokens": 100,
            "error_type": "RuntimeError",
            "request": {"payload": {"prompt": "b"}},
        },
    )
    plan = ReplayOrchestrator(tmp_path).build_plan(
        CaptureFilter(require_error=True, minimum_prompt_tokens=50)
    )
    assert [item.capture_id for item in plan.cases] == ["large-error"]


def test_iso_timestamp_filter_and_external_symlink_are_handled_safely(tmp_path):
    write_capture(
        tmp_path,
        "iso",
        {
            "request_id": "iso",
            "created_at": "2026-08-24T00:00:00Z",
            "request": {"payload": {"prompt": "safe"}},
        },
    )
    outside = tmp_path.parent / "outside-capture.json"
    outside.write_text(
        json.dumps(
            {"request_id": "outside", "request": {"payload": {"prompt": "outside"}}}
        ),
        encoding="utf-8",
    )
    link = tmp_path / "outside.json"
    try:
        link.symlink_to(outside)
    except OSError:
        pass
    orchestrator = ReplayOrchestrator(tmp_path)
    plan = orchestrator.build_plan(CaptureFilter(created_after_s=1_700_000_000))
    assert [item.capture_id for item in plan.cases] == ["iso"]
