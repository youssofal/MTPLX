#!/usr/bin/env python3
"""Measure expert warm-set planning without loading MLX or a model."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtplx.expert_residency import (  # noqa: E402
    ExpertRef,
    ExpertResidencyConfig,
    ExpertResidencyController,
)


class MeasurementBackend:
    mode = "synthetic_residency"

    def __init__(self, refs: Iterable[ExpertRef], expert_bytes: int) -> None:
        self._sizes = {ref: expert_bytes for ref in refs}
        self._resident: set[ExpertRef] = set()

    def resident_experts(self) -> tuple[ExpertRef, ...]:
        return tuple(sorted(self._resident))

    def expert_nbytes(self, expert: ExpertRef) -> int:
        return self._sizes[ExpertRef.coerce(expert)]

    def prefetch_experts(self, experts: Sequence[ExpertRef]) -> tuple[ExpertRef, ...]:
        values = tuple(ExpertRef.coerce(item) for item in experts)
        self._resident.update(values)
        return values

    def evict_experts(self, experts: Sequence[ExpertRef]) -> tuple[ExpertRef, ...]:
        values = tuple(ExpertRef.coerce(item) for item in experts)
        completed = tuple(item for item in values if item in self._resident)
        self._resident.difference_update(completed)
        return completed


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile)))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--events", type=int, default=1_000_000)
    parser.add_argument("--plans", type=int, default=100)
    parser.add_argument("--expert-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--budget-bytes", type=int, default=1024 * 1024 * 1024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    refs = [
        ExpertRef(layer, expert)
        for layer in range(args.layers)
        for expert in range(args.experts)
    ]
    controller = ExpertResidencyController(
        ExpertResidencyConfig(
            enabled=True,
            budget_bytes=args.budget_bytes,
            minimum_observations=1,
            minimum_tick_interval_s=0.0,
            maximum_tracked_experts=len(refs),
            maximum_prefetch_per_tick=8,
            maximum_evict_per_tick=8,
        )
    )
    for index, ref in enumerate(refs):
        weight = 1.0 + ((args.events - index) % max(1, args.experts))
        controller.observe(ref.layer, [ref.expert], weights=[weight], now_s=1.0)

    backend = MeasurementBackend(refs, args.expert_bytes)
    samples_ms: list[float] = []
    final_plan = None
    for index in range(args.plans):
        started = time.perf_counter_ns()
        final_plan = controller.plan(backend, now_s=2.0 + index)
        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)

    assert final_plan is not None
    receipt = controller.apply(final_plan, backend, safe=True)
    mean_ms = statistics.fmean(samples_ms)
    result = {
        "measurement": "expert_residency_planner",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "backend_mode": backend.mode,
        "layers": args.layers,
        "experts_per_layer": args.experts,
        "tracked_experts": len(refs),
        "routing_weight_fixture": args.events,
        "plans": args.plans,
        "plan_ms": {
            "mean": mean_ms,
            "p50": _percentile(samples_ms, 0.50),
            "p95": _percentile(samples_ms, 0.95),
            "p99": _percentile(samples_ms, 0.99),
        },
        "target_experts": len(final_plan.target),
        "target_bytes": final_plan.target_bytes,
        "budget_bytes": final_plan.budget_bytes,
        "prefetched_experts": len(receipt.prefetched),
        "apply_ms": receipt.duration_ms,
        "router_mutation": False,
        "physical_residency_claim": False,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
