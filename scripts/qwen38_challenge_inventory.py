#!/usr/bin/env python3
"""Reproduce and validate the pinned Qwen 3.8 challenge submission ledger.

The validator is deliberately offline.  A changing leaderboard is never fetched
implicitly; regeneration requires explicit saved Yukon HTML and GitHub pull JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT = (
    ROOT
    / "docs"
    / "perf"
    / "receipts"
    / "qwen38-challenge-port"
    / "yukon-accepted-2026-08-23.json"
)
DEFAULT_DESIGN = ROOT / "docs" / "specs" / "2026-08-23-qwen38-challenge-port-design.md"
CHALLENGE_COMMIT = "eb5eadc7a165047d4321ce883b9ff30894d8bd19"
QUALIFYING_RELATIVE_PERCENT = 0.10
EXPECTED_ROWS = 82
EXPECTED_QUALIFYING_ROWS = 54

_NEXT_CHUNK = re.compile(
    r'<script>self\.__next_f\.push\(\[1,("(?:[^"\\]|\\.)*")\]\)</script>'
)
_LEDGER_ROW = re.compile(
    r"^\|\s*(?P<ordinal>\d+)\s*\|\s*(?P<pr>\d+)\s*\|"
    r"\s*(?P<delta>[^|]+?)\s*\|\s*(?P<mechanism>[^|]+?)\s*\|"
    r"\s*(?P<disposition>[^|]+?)\s*\|$"
)


@dataclass(frozen=True)
class InventoryRow:
    ordinal: int
    submission_id: str
    score: float
    submission_commit: str
    source_commit: str
    pr_number: int
    source_pr_number: int
    relative_percent: float | None = None
    mechanism: str = ""
    disposition: str = ""


@dataclass(frozen=True)
class Inventory:
    metadata: dict[str, Any]
    rows: tuple[InventoryRow, ...]


@dataclass(frozen=True)
class ValidationReport:
    qualifying_rows: tuple[InventoryRow, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class SourceDiffReceipt:
    ordinal: int
    pr_number: int
    source_commit: str
    parent_commit: str
    patch_sha256: str
    files: tuple[str, ...]
    insertions: int
    deletions: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _design_rows(path: Path) -> dict[int, tuple[int, str, str]]:
    rows: dict[int, tuple[int, str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _LEDGER_ROW.match(line)
        if match is None:
            continue
        ordinal = int(match.group("ordinal"))
        rows[ordinal] = (
            int(match.group("pr")),
            match.group("mechanism").strip(),
            match.group("disposition").strip(),
        )
    return rows


def load_inventory(receipt_path: Path, design_path: Path) -> Inventory:
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    design = _design_rows(design_path)
    previous_score: float | None = None
    rows: list[InventoryRow] = []
    for raw in payload["rows"]:
        score = float(raw["score"])
        relative = (
            None
            if previous_score is None
            else (score / previous_score - 1.0) * 100.0
        )
        ordinal = int(raw["ordinal"])
        pr_number, mechanism, disposition = design.get(ordinal, (0, "", ""))
        rows.append(
            InventoryRow(
                ordinal=ordinal,
                submission_id=str(raw["submission_id"]),
                score=score,
                submission_commit=str(raw["submission_commit"]),
                source_commit=str(raw["source_commit"]),
                pr_number=pr_number,
                source_pr_number=int(raw["pr_number"]),
                relative_percent=relative,
                mechanism=mechanism,
                disposition=disposition,
            )
        )
        previous_score = score
    return Inventory(metadata=dict(payload["metadata"]), rows=tuple(rows))


def validate_inventory(inventory: Inventory) -> ValidationReport:
    errors: list[str] = []
    rows = inventory.rows
    if len(rows) != EXPECTED_ROWS:
        errors.append(f"expected {EXPECTED_ROWS} rows, found {len(rows)}")
    if [row.ordinal for row in rows] != list(range(1, EXPECTED_ROWS + 1)):
        errors.append("row ordinals are not the complete chronological sequence 1..82")
    ids = [row.submission_id for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("submission IDs are not unique")
    if inventory.metadata.get("challenge_commit") != CHALLENGE_COMMIT:
        errors.append("challenge commit does not match the approved pin")
    if float(inventory.metadata.get("threshold_relative_percent", -1)) != (
        QUALIFYING_RELATIVE_PERCENT
    ):
        errors.append("relative threshold does not match the approved 0.10 percent")
    for row in rows:
        if row.score <= 0:
            errors.append(f"row {row.ordinal}: official score must be positive")
        if not row.submission_commit or not row.source_commit:
            errors.append(f"row {row.ordinal}: missing source commit identity")
        if row.pr_number <= 0 or not row.disposition:
            errors.append(f"row {row.ordinal}: missing approved ledger disposition")
        if row.source_pr_number != row.pr_number:
            errors.append(
                f"row {row.ordinal}: source PR {row.source_pr_number} does not match "
                f"approved PR {row.pr_number}"
            )
    qualifying = tuple(
        row
        for row in rows
        if row.relative_percent is not None
        and row.relative_percent > QUALIFYING_RELATIVE_PERCENT
    )
    if len(qualifying) != EXPECTED_QUALIFYING_ROWS:
        errors.append(
            f"expected {EXPECTED_QUALIFYING_ROWS} qualifying rows, "
            f"found {len(qualifying)}"
        )
    return ValidationReport(qualifying_rows=qualifying, errors=tuple(errors))


def build_source_diff_manifest(
    rows: Sequence[InventoryRow],
    challenge_repo: Path,
    *,
    pinned_commit: str = CHALLENGE_COMMIT,
) -> tuple[SourceDiffReceipt, ...]:
    """Bind every qualifying row to its exact parent patch in the source repo."""

    commits = [row.source_commit for row in rows]
    _validate_challenge_checkout(challenge_repo, [pinned_commit, *commits])
    receipts: list[SourceDiffReceipt] = []
    for row in rows:
        parent = subprocess.run(
            ["git", "rev-parse", f"{row.source_commit}^"],
            cwd=challenge_repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        patch = subprocess.run(
            [
                "git",
                "diff",
                "--no-ext-diff",
                "--binary",
                parent,
                row.source_commit,
                "--",
            ],
            cwd=challenge_repo,
            check=True,
            capture_output=True,
        ).stdout
        numstat = subprocess.run(
            [
                "git",
                "diff",
                "--no-ext-diff",
                "--numstat",
                parent,
                row.source_commit,
                "--",
            ],
            cwd=challenge_repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        files: list[str] = []
        insertions = 0
        deletions = 0
        for line in numstat:
            added, removed, path = line.split("\t", 2)
            if added == "-" or removed == "-":
                raise ValueError(
                    f"row {row.ordinal}: binary numstat cannot prove line coverage"
                )
            insertions += int(added)
            deletions += int(removed)
            files.append(path)
        if not patch or not files:
            raise ValueError(f"row {row.ordinal}: source parent diff is empty")
        receipts.append(
            SourceDiffReceipt(
                ordinal=row.ordinal,
                pr_number=row.pr_number,
                source_commit=row.source_commit,
                parent_commit=parent,
                patch_sha256=hashlib.sha256(patch).hexdigest(),
                files=tuple(files),
                insertions=insertions,
                deletions=deletions,
            )
        )
    return tuple(receipts)


def _extract_yukon_rows(path: Path) -> list[dict[str, Any]]:
    html = path.read_text(encoding="utf-8")
    chunks = [json.loads(match.group(1)) for match in _NEXT_CHUNK.finditer(html)]
    flight_data = "".join(chunks)
    marker = '"submissions"'
    start = flight_data.find(marker)
    if start < 0:
        raise ValueError("saved Yukon payload does not contain submissions")
    start = flight_data.find(":", start) + 1
    submissions, _ = json.JSONDecoder().raw_decode(flight_data[start:])
    accepted = [
        row
        for row in submissions
        if row.get("status") == "accepted"
        and row.get("promotionStatus") == "promoted"
    ]
    accepted.sort(key=lambda row: (row["createdAt"], row["id"]))
    return accepted


def _pull_numbers(path: Path) -> dict[str, int]:
    pulls = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, int] = {}
    for pull in pulls:
        for commit in (pull.get("merge_commit_sha"), pull.get("head", {}).get("sha")):
            if commit:
                mapping[commit] = int(pull["number"])
    return mapping


def _validate_challenge_checkout(path: Path, commits: Sequence[str]) -> None:
    unique = tuple(dict.fromkeys(commits))
    checked = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype)"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
        input="".join(f"{commit}\n" for commit in unique),
    ).stdout.splitlines()
    missing = [
        commit
        for commit, result in zip(unique, checked, strict=True)
        if result != f"{commit} commit"
    ]
    if missing:
        raise ValueError(f"challenge checkout is missing source commits: {missing}")


def generate_receipt(
    yukon_html: Path,
    github_pulls: Path,
    *,
    challenge_repo: Path,
    threshold: float,
) -> dict[str, Any]:
    if float(threshold) != QUALIFYING_RELATIVE_PERCENT:
        raise ValueError(
            f"threshold must equal approved {QUALIFYING_RELATIVE_PERCENT:.2f} percent"
        )
    pulls = _pull_numbers(github_pulls)
    source_rows = _extract_yukon_rows(yukon_html)
    rows: list[dict[str, Any]] = []
    for ordinal, source in enumerate(source_rows, start=1):
        source_commit = str(source["promotedSourceRef"])
        if source_commit not in pulls:
            raise ValueError(f"no challenge PR maps to source commit {source_commit}")
        rows.append(
            {
                "ordinal": ordinal,
                "submission_id": source["id"],
                "score": source["officialScore"],
                "submission_commit": source.get("submissionCommitSha") or source_commit,
                "source_commit": source_commit,
                "pr_number": pulls[source_commit],
            }
        )
    _validate_challenge_checkout(
        challenge_repo,
        [CHALLENGE_COMMIT, *[str(row["source_commit"]) for row in rows]],
    )
    return {
        "metadata": {
            "schema_version": 1,
            "source_url": "https://www.yukon.org/mlxfast",
            "snapshot_rendered_date": "2026-08-23",
            "yukon_html_sha256": _sha256(yukon_html),
            "github_pulls_sha256": _sha256(github_pulls),
            "challenge_commit": CHALLENGE_COMMIT,
            "threshold_relative_percent": float(threshold),
        },
        "rows": rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--yukon-html", type=Path)
    parser.add_argument("--github-pulls", type=Path)
    parser.add_argument("--challenge-repo", type=Path)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--emit", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.emit:
        if any(
            value is None
            for value in (
                args.yukon_html,
                args.github_pulls,
                args.challenge_repo,
                args.threshold,
            )
        ):
            raise SystemExit(
                "--emit requires --yukon-html, --github-pulls, --challenge-repo, "
                "and --threshold"
            )
        print(
            json.dumps(
                generate_receipt(
                    args.yukon_html,
                    args.github_pulls,
                    challenge_repo=args.challenge_repo,
                    threshold=args.threshold,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.check:
        raise SystemExit("choose --check or --emit with explicit saved inputs")
    inventory = load_inventory(args.receipt, args.design)
    report = validate_inventory(inventory)
    if report.errors:
        for error in report.errors:
            print(f"ERROR: {error}")
        return 1
    print(
        f"OK: {len(inventory.rows)} promoted submissions; "
        f"{len(report.qualifying_rows)} above {QUALIFYING_RELATIVE_PERCENT:.2f}%"
    )
    print(f"receipt_sha256={_sha256(args.receipt)}")
    print(f"design_sha256={_sha256(args.design)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
