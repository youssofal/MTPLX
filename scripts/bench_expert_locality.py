#!/usr/bin/env python3
"""Measure expert-locality instrumentation overhead without loading a model."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtplx.expert_locality import (
    expert_locality_metrics,
    install_expert_locality_instrumentation,
    reset_expert_locality_tracker,
)


class FakeSwitch:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self, hidden: int, indices: list[list[int]]
    ) -> tuple[int, list[list[int]]]:
        self.calls += 1
        return hidden, indices


class FakeBlock:
    num_experts = 64

    def __init__(self) -> None:
        self.switch_mlp = FakeSwitch()


class FakeModel:
    def __init__(self) -> None:
        self.block = FakeBlock()

    def named_modules(self) -> list[tuple[str, Any]]:
        return [
            ("", self),
            ("layers.0", self.block),
            ("layers.0.switch_mlp", self.block.switch_mlp),
        ]


@dataclass(frozen=True)
class Timing:
    samples_ns: tuple[int, ...]

    def to_dict(self, iterations: int) -> dict[str, float | list[float]]:
        per_call_us = [value / iterations / 1_000 for value in self.samples_ns]
        return {
            "median_us_per_call": statistics.median(per_call_us),
            "min_us_per_call": min(per_call_us),
            "max_us_per_call": max(per_call_us),
            "samples_us_per_call": per_call_us,
        }


def measure(
    operation: Callable[[int, list[list[int]]], Any],
    routes: list[list[int]],
    repeats: int,
) -> tuple[Timing, Any]:
    samples: list[int] = []
    result: Any = None
    for _ in range(repeats):
        started = time.perf_counter_ns()
        for index, route in enumerate(routes):
            result = operation(index, [route])
        samples.append(time.perf_counter_ns() - started)
    return Timing(tuple(samples)), result


def instrumented_arm(
    routes: list[list[int]],
    repeats: int,
    sample_every: int,
) -> tuple[Timing, Any, dict[str, Any], dict[str, Any]]:
    os.environ["MTPLX_EXPERT_LOCALITY"] = "1"
    os.environ["MTPLX_EXPERT_LOCALITY_SAMPLE_EVERY"] = str(sample_every)
    os.environ["MTPLX_EXPERT_LOCALITY_MAX_EVENTS"] = str(len(routes) * repeats + 128)
    os.environ["MTPLX_EXPERT_LOCALITY_CACHE_SIZES"] = "16,32,64"
    reset_expert_locality_tracker()
    model = FakeModel()
    install = install_expert_locality_instrumentation(model)
    for index in range(64):
        model.block.switch_mlp(index, [routes[index % len(routes)]])
    reset_expert_locality_tracker()
    timing, result = measure(model.block.switch_mlp, routes, repeats)
    return timing, result, install, expert_locality_metrics()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50_000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--sample-every", type=int, default=16)
    args = parser.parse_args()
    iterations = max(1, args.iterations)
    repeats = max(1, args.repeats)
    sample_every = max(1, args.sample_every)
    routes = [[(index % 63) % 8, ((index % 63) + 1) % 8] for index in range(iterations)]

    baseline_model = FakeModel()
    baseline, baseline_result = measure(
        baseline_model.block.switch_mlp,
        routes,
        repeats,
    )
    sampled, sampled_result, sampled_install, sampled_metrics = instrumented_arm(
        routes,
        repeats,
        sample_every,
    )
    full, full_result, full_install, full_metrics = instrumented_arm(
        routes,
        repeats,
        1,
    )

    baseline_data = baseline.to_dict(iterations)
    sampled_data = sampled.to_dict(iterations)
    full_data = full.to_dict(iterations)
    baseline_us = float(baseline_data["median_us_per_call"])
    sampled_us = float(sampled_data["median_us_per_call"])
    full_us = float(full_data["median_us_per_call"])
    sampled_layer = sampled_metrics["layers"][0]
    full_layer = full_metrics["layers"][0]
    output_parity = (
        baseline_result == sampled_result == full_result
        and baseline_model.block.switch_mlp.calls == iterations * repeats
    )

    report = {
        "schema_version": 1,
        "measurement": "cpu_python_list_router_tap_overhead",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "iterations_per_repeat": iterations,
        "repeats": repeats,
        "route_pattern": "rotating_8_experts_top_2_period_63",
        "baseline": baseline_data,
        "sampled": {
            **sampled_data,
            "sample_every": sample_every,
            "overhead_us_per_call": sampled_us - baseline_us,
            "slowdown_ratio": sampled_us / baseline_us,
            "accepted_calls": sampled_metrics["accepted_calls"],
            "working_set_90": sampled_layer["working_set_90"],
            "lru_16_hit_rate": sampled_layer["lru_simulation"]["16"]["hit_rate"],
            "installed_modules": sampled_install["instrumented_modules"],
        },
        "full_sampling": {
            **full_data,
            "sample_every": 1,
            "overhead_us_per_call": full_us - baseline_us,
            "slowdown_ratio": full_us / baseline_us,
            "accepted_calls": full_metrics["accepted_calls"],
            "working_set_90": full_layer["working_set_90"],
            "lru_16_hit_rate": full_layer["lru_simulation"]["16"]["hit_rate"],
            "installed_modules": full_install["instrumented_modules"],
        },
        "output_parity": output_parity,
        "scope": (
            "CPU-only Python-list measurement of the router tap boundary. "
            "It does not claim model throughput or MLX materialization cost."
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if output_parity else 1


if __name__ == "__main__":
    raise SystemExit(main())
