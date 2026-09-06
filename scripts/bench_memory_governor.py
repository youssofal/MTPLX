#!/usr/bin/env python3
"""Measure memory-governor decision and safe-point application overhead."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtplx.memory_governor import (  # noqa: E402
    GIB,
    MemoryGovernorConfig,
    MemorySafePoint,
    MemorySample,
    RuntimeMemoryGovernor,
)


class BenchmarkBank:
    def __init__(self) -> None:
        self.max_bytes = 20 * GIB
        self.per_session_max_bytes = 8 * GIB
        self.total_nbytes = 12 * GIB
        self._entries = {}
        self.rebalance_calls = 0

    def rebalance_limits(
        self,
        *,
        max_bytes: int,
        per_session_max_bytes: int,
        reason: str,
    ) -> None:
        if reason != "runtime_memory_governor":
            raise AssertionError(reason)
        self.max_bytes = int(max_bytes)
        self.per_session_max_bytes = int(per_session_max_bytes)
        self.rebalance_calls += 1


def _governor() -> RuntimeMemoryGovernor:
    return RuntimeMemoryGovernor(
        initial_bank_max_bytes=20 * GIB,
        initial_per_session_max_bytes=8 * GIB,
        config=MemoryGovernorConfig(
            high_observations=1,
            recovery_observations=1,
            minimum_apply_interval_s=0.0,
        ),
    )


def _sample(rss_gib: int, *, safe: bool = True) -> MemorySample:
    return MemorySample(
        total_bytes=100 * GIB,
        rss_bytes=rss_gib * GIB,
        session_bank_bytes=12 * GIB,
        model_bytes=60 * GIB,
        timestamp_s=10.0,
        safe_point=(
            MemorySafePoint()
            if safe
            else MemorySafePoint(foreground_active=1)
        ),
    )


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * percentile))
    return ordered[index]


def _measure(
    name: str,
    iterations: int,
    operation: Callable[[int], None],
    *,
    batches: int,
) -> dict[str, float | int | str]:
    batch_count = min(max(1, batches), iterations)
    base_size, remainder = divmod(iterations, batch_count)
    durations_ns: list[int] = []
    operations_per_batch: list[int] = []
    offset = 0
    for batch_index in range(batch_count):
        batch_size = base_size + (1 if batch_index < remainder else 0)
        start_ns = time.perf_counter_ns()
        for local_index in range(batch_size):
            operation(offset + local_index)
        durations_ns.append(time.perf_counter_ns() - start_ns)
        operations_per_batch.append(batch_size)
        offset += batch_size

    total_ns = sum(durations_ns)
    per_operation_ns = [
        duration / count
        for duration, count in zip(durations_ns, operations_per_batch)
    ]
    return {
        "name": name,
        "operations": iterations,
        "batches": batch_count,
        "total_seconds": total_ns / 1_000_000_000,
        "operations_per_second": iterations / (total_ns / 1_000_000_000),
        "mean_ns_per_operation": statistics.fmean(per_operation_ns),
        "p50_ns_per_operation": _percentile(per_operation_ns, 0.50),
        "p95_ns_per_operation": _percentile(per_operation_ns, 0.95),
        "p99_ns_per_operation": _percentile(per_operation_ns, 0.99),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observe-iterations", type=int, default=200_000)
    parser.add_argument("--apply-iterations", type=int, default=50_000)
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.observe_iterations < 1 or args.apply_iterations < 1:
        parser.error("iteration counts must be positive")
    if args.batches < 1:
        parser.error("batches must be positive")

    observe_governor = _governor()
    samples = tuple(_sample(rss) for rss in (65, 78, 86, 93))

    def observe(index: int) -> None:
        observe_governor.observe(samples[index % len(samples)])

    safe_governor = _governor()
    safe_bank = BenchmarkBank()
    safe_decision = safe_governor.observe(_sample(93))
    safe_applied = 0

    def apply_safe(_index: int) -> None:
        nonlocal safe_applied
        safe_applied += int(
            safe_governor.apply(safe_decision, bank=safe_bank).applied
        )

    blocked_governor = _governor()
    blocked_bank = BenchmarkBank()
    blocked_decision = blocked_governor.observe(_sample(93, safe=False))
    blocked_applied = 0

    def apply_blocked(_index: int) -> None:
        nonlocal blocked_applied
        blocked_applied += int(
            blocked_governor.apply(blocked_decision, bank=blocked_bank).applied
        )

    measurements = [
        _measure(
            "observe_synthetic_pressure",
            args.observe_iterations,
            observe,
            batches=args.batches,
        ),
        _measure(
            "apply_safe_budget",
            args.apply_iterations,
            apply_safe,
            batches=args.batches,
        ),
        _measure(
            "reject_unsafe_budget",
            args.apply_iterations,
            apply_blocked,
            batches=args.batches,
        ),
    ]
    if safe_applied != args.apply_iterations:
        raise RuntimeError("safe apply path did not apply every decision")
    if safe_bank.rebalance_calls != args.apply_iterations:
        raise RuntimeError("safe apply path did not rebalance every decision")
    if blocked_applied != 0 or blocked_bank.rebalance_calls != 0:
        raise RuntimeError("unsafe apply path mutated the SessionBank budget")
    receipt = {
        "schema_version": 1,
        "benchmark": "sessionbank_memory_governor_no_model",
        "command": (
            "python3 scripts/bench_memory_governor.py "
            f"--observe-iterations {args.observe_iterations} "
            f"--apply-iterations {args.apply_iterations} "
            f"--batches {args.batches} "
            "--output docs/benchmarks/"
            "sessionbank-memory-governor-no-model-20260826.json"
        ),
        "scope": (
            "Synthetic controller overhead only. No model, MLX runtime, "
            "or live memory-pressure event is used."
        ),
        "host": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "inputs_sha256": {
            "mtplx/memory_governor.py": _sha256(
                ROOT / "mtplx" / "memory_governor.py"
            ),
            "scripts/bench_memory_governor.py": _sha256(Path(__file__)),
        },
        "configuration": {
            "observe_iterations": args.observe_iterations,
            "apply_iterations": args.apply_iterations,
            "batches": args.batches,
        },
        "measurements": measurements,
        "correctness": {
            "safe_apply_receipts": safe_applied,
            "safe_rebalance_calls": safe_bank.rebalance_calls,
            "blocked_apply_receipts": blocked_applied,
            "blocked_rebalance_calls": blocked_bank.rebalance_calls,
        },
    }
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
