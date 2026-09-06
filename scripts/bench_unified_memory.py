#!/usr/bin/env python3
"""Measure unified-memory planning, application, and rollback cost."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtplx.unified_memory import (  # noqa: E402
    GIB,
    UnifiedMemoryConfig,
    UnifiedMemoryCoordinator,
    UnifiedMemorySample,
)


@dataclass
class MeasurementConsumer:
    name: str
    budget: int
    fail: bool = False

    def current_budget_bytes(self) -> int:
        return self.budget

    def apply_budget_bytes(self, value: int, *, reason: str) -> int:
        if not reason:
            raise ValueError("reason is required")
        if self.fail:
            raise RuntimeError("injected apply failure")
        self.budget = int(value)
        return self.budget


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _timing(samples: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(samples),
        "p50": _percentile(samples, 0.50),
        "p95": _percentile(samples, 0.95),
        "p99": _percentile(samples, 0.99),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=int, default=100_000)
    parser.add_argument("--apply-iterations", type=int, default=10_000)
    parser.add_argument("--failure-every", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = UnifiedMemoryConfig(
        enabled=True,
        reserve_bytes=4 * GIB,
        minimum_available_bytes=512 * 1024 * 1024,
        target_utilization=0.88,
        warning_utilization=0.92,
        critical_utilization=0.96,
        hysteresis_ratio=0.0,
        minimum_apply_interval_s=0.0,
        minimum_session_bank_bytes=1 * GIB,
        minimum_expert_bytes=1 * GIB,
        minimum_kv_headroom_bytes=1 * GIB,
    )
    coordinator = UnifiedMemoryCoordinator(config)
    samples = (
        UnifiedMemorySample(128 * GIB, 80 * GIB, 48 * GIB, 8 * GIB, 6 * GIB, 8 * GIB),
        UnifiedMemorySample(128 * GIB, 119 * GIB, 48 * GIB, 8 * GIB, 6 * GIB, 8 * GIB),
        UnifiedMemorySample(128 * GIB, 125 * GIB, 48 * GIB, 8 * GIB, 6 * GIB, 8 * GIB),
    )

    plan_us: list[float] = []
    over_budget = 0
    final_plan = None
    for index in range(args.plans):
        started = time.perf_counter_ns()
        final_plan = coordinator.plan(
            samples[index % len(samples)],
            safe=True,
            now_s=float(index + 1),
        )
        plan_us.append((time.perf_counter_ns() - started) / 1000.0)
        allocated = (
            final_plan.session_bank_budget_bytes
            + final_plan.expert_budget_bytes
            + final_plan.kv_headroom_bytes
        )
        over_budget = max(over_budget, allocated - final_plan.managed_budget_bytes)

    assert final_plan is not None
    apply_us: list[float] = []
    rollback_us: list[float] = []
    rollback_count = 0
    rollback_failures = 0
    for index in range(args.apply_iterations):
        inject_failure = args.failure_every > 0 and index % args.failure_every == 0
        session = MeasurementConsumer("session_bank", 8 * GIB)
        expert = MeasurementConsumer("expert_residency", 6 * GIB, fail=inject_failure)
        kv = MeasurementConsumer("kv_headroom", 8 * GIB)
        started = time.perf_counter_ns()
        receipt = coordinator.apply(final_plan, [session, expert, kv], safe=True)
        elapsed_us = (time.perf_counter_ns() - started) / 1000.0
        if inject_failure:
            rollback_count += 1
            rollback_us.append(elapsed_us)
            if not receipt.rolled_back or session.budget != 8 * GIB:
                rollback_failures += 1
        else:
            apply_us.append(elapsed_us)

    result = {
        "measurement": "unified_memory_coordinator",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "plans": args.plans,
        "apply_iterations": args.apply_iterations,
        "failure_every": args.failure_every,
        "plan_us": _timing(plan_us),
        "apply_us": _timing(apply_us),
        "rollback_us": _timing(rollback_us),
        "rollback_count": rollback_count,
        "rollback_failures": rollback_failures,
        "maximum_budget_overage_bytes": over_budget,
        "final_pressure": final_plan.pressure,
        "final_managed_budget_bytes": final_plan.managed_budget_bytes,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
