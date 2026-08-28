from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import qwen38_challenge_inventory as inventory_tool
from scripts.qwen38_challenge_inventory import (
    CHALLENGE_COMMIT,
    QUALIFYING_RELATIVE_PERCENT,
    build_source_diff_manifest,
    load_inventory,
    validate_inventory,
)

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "docs"
    / "perf"
    / "receipts"
    / "qwen38-challenge-port"
    / "yukon-accepted-2026-08-23.json"
)
DESIGN = ROOT / "docs" / "specs" / "2026-08-23-qwen38-challenge-port-design.md"


def test_pinned_inventory_covers_every_accepted_submission() -> None:
    inventory = load_inventory(RECEIPT, DESIGN)

    assert CHALLENGE_COMMIT == "eb5eadc7a165047d4321ce883b9ff30894d8bd19"
    assert QUALIFYING_RELATIVE_PERCENT == 0.10
    assert len(inventory.rows) == 82
    assert [row.ordinal for row in inventory.rows] == list(range(1, 83))
    assert len({row.submission_id for row in inventory.rows}) == 82
    assert all(row.score > 0 for row in inventory.rows)
    assert all(row.submission_commit for row in inventory.rows)


def test_relative_threshold_and_dispositions_are_reproducible() -> None:
    inventory = load_inventory(RECEIPT, DESIGN)
    report = validate_inventory(inventory)

    assert report.errors == ()
    assert len(report.qualifying_rows) == 54
    assert inventory.rows[0].relative_percent is None
    assert all(
        row.relative_percent > QUALIFYING_RELATIVE_PERCENT
        for row in report.qualifying_rows
    )
    assert all(row.disposition for row in inventory.rows)


def test_every_port_or_dependency_has_pr_and_source_commit() -> None:
    inventory = load_inventory(RECEIPT, DESIGN)

    selected = [
        row
        for row in inventory.rows
        if row.disposition.startswith(("PORT", "DEPENDENCY"))
    ]
    assert selected
    assert all(row.pr_number > 0 for row in selected)
    assert all(len(row.source_commit) == 40 for row in selected)


def test_source_receipt_pr_must_match_the_approved_ledger(tmp_path: Path) -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    payload["rows"][4]["pr_number"] = 9999
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_inventory(load_inventory(altered, DESIGN))

    assert "row 5: source PR 9999 does not match approved PR 29" in report.errors


def test_regeneration_requires_explicit_checkout_and_threshold() -> None:
    with pytest.raises(
        SystemExit,
        match="--emit requires --yukon-html, --github-pulls, --challenge-repo, "
        "and --threshold",
    ):
        inventory_tool.main(
            [
                "--emit",
                "--yukon-html",
                "/does/not/matter.html",
                "--github-pulls",
                "/does/not/matter.json",
            ]
        )


def test_source_diff_manifest_uses_configured_objects_not_checkout_head(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "challenge"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=repo, check=True
    )
    source = repo / "candidate.swift"
    source.write_text("control\n", encoding="utf-8")
    subprocess.run(["git", "add", "candidate.swift"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "control"], cwd=repo, check=True)
    source.write_text("control\ncandidate\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-qam", "candidate"], cwd=repo, check=True)
    candidate = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(["git", "checkout", "-q", "HEAD^"], cwd=repo, check=True)
    row = inventory_tool.InventoryRow(
        ordinal=1,
        submission_id="submission",
        score=1.0,
        submission_commit=candidate,
        source_commit=candidate,
        pr_number=1,
        source_pr_number=1,
    )

    (diff,) = build_source_diff_manifest((row,), repo, pinned_commit=candidate)

    assert diff.source_commit == candidate
    assert diff.files == ("candidate.swift",)
    assert diff.insertions == 1
    assert diff.deletions == 0
