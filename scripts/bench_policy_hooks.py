#!/usr/bin/env python3
"""Measure PolicyBus dispatch cost, outcomes, and executor bounds."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mtplx.policy_hooks import (  # noqa: E402
    HookPhase,
    HookResult,
    PolicyBus,
    PolicyHookConfig,
)


def _percentile(values: list[int], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * percentile))
    return ordered[index] / 1_000


def _summary(samples_ns: list[int], wall_ns: int) -> dict[str, float]:
    elapsed_s = wall_ns / 1_000_000_000
    return {
        "p50_us": _percentile(samples_ns, 0.50),
        "p95_us": _percentile(samples_ns, 0.95),
        "p99_us": _percentile(samples_ns, 0.99),
        "mean_us": statistics.fmean(samples_ns) / 1_000,
        "throughput_per_s": len(samples_ns) / elapsed_s,
    }


def _measure_dispatches(requests: int, hook_count: int) -> dict[str, Any]:
    bus = PolicyBus(
        PolicyHookConfig(maximum_hooks=max(1, hook_count), maximum_workers=4)
    )
    for index in range(hook_count):
        bus.register(
            f"allow-{index}",
            lambda _value, _context: HookResult.allow(),
            phases=[HookPhase.REQUEST],
        )

    samples: list[int] = []
    wall_started = time.perf_counter_ns()
    for index in range(requests):
        started = time.perf_counter_ns()
        outcome = bus.execute(HookPhase.REQUEST, {"request": index})
        samples.append(time.perf_counter_ns() - started)
        if not outcome.allowed or len(outcome.executed_hooks) != hook_count:
            raise AssertionError("allow path returned an invalid outcome")

    wall_ns = time.perf_counter_ns() - wall_started
    snapshot = bus.snapshot()
    bus.shutdown()
    return {
        "hook_count": hook_count,
        "requests": requests,
        "executor_started": snapshot["executor_started"],
        **_summary(samples, wall_ns),
    }


def _measure_outcomes(requests: int) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    cases = {
        "rewrite": lambda _value, _context: HookResult.rewrite({"rewritten": True}),
        "reject": lambda _value, _context: HookResult.reject(
            "rate_limited", status_code=429
        ),
    }
    for name, callback in cases.items():
        bus = PolicyBus()
        bus.register(name, callback, phases=[HookPhase.REQUEST])
        samples: list[int] = []
        correct = 0
        wall_started = time.perf_counter_ns()
        for _ in range(requests):
            started = time.perf_counter_ns()
            outcome = bus.execute(HookPhase.REQUEST, {"input": True})
            samples.append(time.perf_counter_ns() - started)
            if name == "rewrite":
                correct += int(outcome.allowed and outcome.value == {"rewritten": True})
            else:
                correct += int(not outcome.allowed and outcome.status_code == 429)
        wall_ns = time.perf_counter_ns() - wall_started
        bus.shutdown()
        arms[name] = {
            "requests": requests,
            "correct": correct,
            **_summary(samples, wall_ns),
        }
    return arms


def _measure_timeout_bound() -> dict[str, Any]:
    state_lock = threading.Lock()
    active = 0
    maximum_active = 0

    def slow(_value: Any, _context: Any) -> HookResult:
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.05)
        finally:
            with state_lock:
                active -= 1
        return HookResult.allow()

    bus = PolicyBus(
        PolicyHookConfig(
            default_timeout_s=0.005,
            maximum_workers=2,
            maximum_pending_tasks=2,
        )
    )
    bus.register("slow", slow, phases=[HookPhase.REQUEST])

    def invoke(index: int) -> tuple[int, bool, bool]:
        started = time.perf_counter_ns()
        outcome = bus.execute(HookPhase.REQUEST, {"request": index})
        return (
            time.perf_counter_ns() - started,
            bool(outcome.timed_out_hooks),
            bool(outcome.failed_hooks),
        )

    wall_started = time.perf_counter_ns()
    with ThreadPoolExecutor(max_workers=8) as callers:
        results = list(callers.map(invoke, range(32)))
    wall_ns = time.perf_counter_ns() - wall_started
    snapshot = bus.snapshot()
    bus.shutdown()
    samples = [row[0] for row in results]
    return {
        "callers": len(results),
        "configured_workers": 2,
        "configured_pending_tasks": 2,
        "maximum_active_callbacks": maximum_active,
        "timed_out_calls": sum(row[1] for row in results),
        "saturated_calls": sum(row[2] for row in results),
        "snapshot_timeouts": snapshot["timeouts"],
        "snapshot_failures": snapshot["failures"],
        **_summary(samples, wall_ns),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=5_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.requests < 1:
        raise SystemExit("--requests must be positive")

    receipt = {
        "benchmark": "policy_hooks",
        "clock": "time.perf_counter_ns",
        "dispatch": [
            _measure_dispatches(args.requests, hook_count)
            for hook_count in (0, 1, 4, 16)
        ],
        "outcomes": _measure_outcomes(args.requests),
        "timeout_bound": _measure_timeout_bound(),
    }
    rendered = json.dumps(receipt, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
