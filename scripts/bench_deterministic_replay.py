#!/usr/bin/env python3
"""Measure deterministic replay throughput, deduplication, and timeouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtplx.deterministic_replay import (  # noqa: E402
    CounterfactualReplay,
    Evaluation,
    RegressionPolicy,
    ReplayCase,
)


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _evaluator(name: str):
    def evaluate(case: ReplayCase, output: Any, baseline: Any) -> Evaluation:
        score = float(output["score"])
        baseline_score = float(baseline["score"])
        return Evaluation(
            name=name,
            score=score,
            passed=score >= baseline_score,
            baseline_score=baseline_score,
            details={"case_id": case.case_id},
        )

    return evaluate


def _report_digest(report: Any, decision: Any) -> str:
    payload = {
        "report": report.to_dict(),
        "decision": decision.to_dict(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _timeout_measurement(timeout_s: float) -> dict[str, Any]:
    def slow_candidate(_request: Any) -> dict[str, float]:
        time.sleep(timeout_s * 10)
        return {"score": 1.0}

    started = time.perf_counter()
    report = CounterfactualReplay(candidate_timeout_s=timeout_s).run(
        [ReplayCase("timeout", {"value": 1}, baseline_output={"score": 1.0})],
        candidate=slow_candidate,
        evaluators={"score": _evaluator("score")},
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "configured_timeout_ms": timeout_s * 1000.0,
        "caller_elapsed_ms": elapsed_ms,
        "error_type": report.results[0].errors[0].error_type,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=128)
    parser.add_argument("--evaluators", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    unique_requests = max(1, args.cases // 2)
    cases = [
        ReplayCase(
            case_id=f"case-{index:04d}",
            request={"value": index % unique_requests},
            baseline_output={"score": float(index % unique_requests)},
        )
        for index in range(args.cases)
    ]
    evaluators = {
        f"score-{index}": _evaluator(f"score-{index}")
        for index in range(args.evaluators)
    }
    replay = CounterfactualReplay(max_concurrency=4, deduplicate_requests=True)
    policy = RegressionPolicy(minimum_cases=args.cases)

    elapsed_ms: list[float] = []
    digests: list[str] = []
    reused = 0
    for _ in range(args.repeats):
        started = time.perf_counter()
        report = replay.run(
            cases,
            candidate=lambda request: {"score": float(request["value"])},
            evaluators=evaluators,
            candidate_name="measurement-candidate",
        )
        decision = policy.evaluate(report)
        elapsed_ms.append((time.perf_counter() - started) * 1000.0)
        digests.append(_report_digest(report, decision))
        reused = sum(result.candidate_reused for result in report.results)

    mean_ms = statistics.fmean(elapsed_ms)
    receipt = {
        "measurement": "deterministic_counterfactual_replay",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cases": args.cases,
        "evaluators": args.evaluators,
        "repeats": args.repeats,
        "candidate_executions": args.cases - reused,
        "deduplicated_cases": reused,
        "suite_ms": {
            "mean": mean_ms,
            "p50": _percentile(elapsed_ms, 0.50),
            "p95": _percentile(elapsed_ms, 0.95),
            "p99": _percentile(elapsed_ms, 0.99),
        },
        "mean_cases_per_second": args.cases / (mean_ms / 1000.0),
        "unique_report_digests": len(set(digests)),
        "report_digest": digests[0],
        "timeout": _timeout_measurement(0.005),
    }
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
