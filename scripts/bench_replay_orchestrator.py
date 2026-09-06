#!/usr/bin/env python3
"""Measure capture plan selection, freshness checks, and receipt persistence."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from mtplx.replay_orchestrator import (
    ReplayOrchestrator,
    ReplayPlanConfig,
    StaleReplayPlanError,
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


def summary(samples_ns: list[int]) -> dict[str, float]:
    samples_ms = [value / 1_000_000 for value in samples_ns]
    return {
        "mean_ms": round(statistics.fmean(samples_ms), 3),
        "p50_ms": round(percentile(samples_ms, 0.50), 3),
        "p95_ms": round(percentile(samples_ms, 0.95), 3),
        "p99_ms": round(percentile(samples_ms, 0.99), 3),
    }


def write_captures(root: Path, count: int) -> None:
    for index in range(count):
        payload = {
            "request_id": f"case-{index:05d}",
            "model": "benchmark-model",
            "request_fingerprint": f"fingerprint-{index:05d}",
            "prompt_tokens": 64 + index % 2048,
            "request": {"payload": {"messages": index % 32}},
        }
        (root / f"capture-{index:05d}.json").write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="mtplx-replay-bench-") as temporary:
        root = Path(temporary)
        capture_root = root / "captures"
        receipt_root = root / "receipts"
        capture_root.mkdir()
        write_captures(capture_root, args.captures)
        orchestrator = ReplayOrchestrator(
            capture_root,
            config=ReplayPlanConfig(
                maximum_scan_files=args.captures,
                maximum_cases=args.maximum_cases,
                deterministic_seed="benchmark",
            ),
            receipt_directory=receipt_root,
        )

        plan_samples: list[int] = []
        source_digests: set[str] = set()
        plan = orchestrator.build_plan()
        for _ in range(args.plan_iterations):
            started = time.perf_counter_ns()
            plan = orchestrator.build_plan()
            plan_samples.append(time.perf_counter_ns() - started)
            source_digests.add(plan.source_digest)

        freshness_samples: list[int] = []
        for _ in range(args.freshness_iterations):
            started = time.perf_counter_ns()
            orchestrator.assert_fresh(plan)
            freshness_samples.append(time.perf_counter_ns() - started)

        receipt_samples: list[int] = []
        for index in range(args.receipt_iterations):
            started = time.perf_counter_ns()
            orchestrator.record_receipt(
                plan,
                candidate_name=f"candidate-{index}",
                report={"pass_rate": 1.0, "case_count": len(plan.cases)},
                decision={"promote": True, "reasons": []},
            )
            receipt_samples.append(time.perf_counter_ns() - started)

        receipt_files = tuple(receipt_root.glob("*.json"))
        temporary_files = tuple(receipt_root.glob("*.tmp"))
        selected_capture = capture_root / plan.cases[0].path
        selected_capture.write_text("{}", encoding="utf-8")
        stale_started = time.perf_counter_ns()
        stale_mutation_rejected = False
        try:
            orchestrator.assert_fresh(plan)
        except StaleReplayPlanError:
            stale_mutation_rejected = True
        stale_rejection_ms = (time.perf_counter_ns() - stale_started) / 1_000_000
        return {
            "captures": args.captures,
            "selected_cases": len(plan.cases),
            "plan_iterations": args.plan_iterations,
            "freshness_iterations": args.freshness_iterations,
            "receipt_iterations": args.receipt_iterations,
            "plan_selection": summary(plan_samples),
            "stale_plan_validation": summary(freshness_samples),
            "atomic_receipt_write": summary(receipt_samples),
            "stable_source_digest_count": len(source_digests),
            "stale_mutation_rejected": stale_mutation_rejected,
            "stale_rejection_ms": round(stale_rejection_ms, 3),
            "receipt_files": len(receipt_files),
            "temporary_files_remaining": len(temporary_files),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures", type=int, default=1000)
    parser.add_argument("--maximum-cases", type=int, default=128)
    parser.add_argument("--plan-iterations", type=int, default=30)
    parser.add_argument("--freshness-iterations", type=int, default=100)
    parser.add_argument("--receipt-iterations", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
