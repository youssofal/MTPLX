#!/usr/bin/env python3
"""Measure runtime status update, snapshot, and HTTP endpoint overhead."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mtplx.runtime_systems import (
    RuntimeSystemsRegistry,
    install_runtime_systems_endpoint,
)


def percentile(samples_ns: list[int], percentile_value: float) -> float:
    ordered = sorted(samples_ns)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * percentile_value))
    return ordered[index] / 1_000.0


def summarize(samples_ns: list[int]) -> dict[str, float]:
    elapsed_s = sum(samples_ns) / 1_000_000_000.0
    return {
        "p50_us": percentile(samples_ns, 0.50),
        "p95_us": percentile(samples_ns, 0.95),
        "p99_us": percentile(samples_ns, 0.99),
        "mean_us": statistics.fmean(samples_ns) / 1_000.0,
        "operations_per_s": len(samples_ns) / elapsed_s,
    }


def measure_call(callback: Any, iterations: int) -> list[int]:
    samples: list[int] = []
    for index in range(iterations):
        started = time.perf_counter_ns()
        callback(index)
        samples.append(time.perf_counter_ns() - started)
    return samples


def run(iterations: int, endpoint_requests: int, system_count: int) -> dict[str, Any]:
    registry = RuntimeSystemsRegistry(max_systems=max(128, system_count))
    for index in range(system_count):
        registry.update(
            f"system.{index}",
            {
                "available": True,
                "enabled": index % 2 == 0,
                "metrics": {"requests": index, "queue_depth": index % 4},
            },
        )

    update_samples = measure_call(
        lambda index: registry.update(
            "system.0",
            {
                "available": True,
                "enabled": True,
                "metrics": {"requests": index, "queue_depth": index % 4},
            },
        ),
        iterations,
    )
    snapshot_samples = measure_call(lambda _index: registry.snapshot(), iterations)

    app = FastAPI()
    install_runtime_systems_endpoint(app, SimpleNamespace(runtime_systems=registry))
    with TestClient(app) as client:
        endpoint_samples = measure_call(
            lambda _index: client.get("/v1/mtplx/systems").raise_for_status(),
            endpoint_requests,
        )
        response = client.get("/v1/mtplx/systems")

    payload = response.json()
    return {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "parameters": {
            "iterations": iterations,
            "endpoint_requests": endpoint_requests,
            "system_count": system_count,
        },
        "update": summarize(update_samples),
        "snapshot": summarize(snapshot_samples),
        "http_get": summarize(endpoint_samples),
        "correctness": {
            "http_status": response.status_code,
            "reported_system_count": payload["system_count"],
            "reported_revision": payload["revision"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--endpoint-requests", type=int, default=1_000)
    parser.add_argument("--system-count", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.endpoint_requests < 1 or args.system_count < 1:
        parser.error("all counts must be positive")

    result = run(args.iterations, args.endpoint_requests, args.system_count)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
