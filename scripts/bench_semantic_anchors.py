#!/usr/bin/env python3
"""Measure semantic-anchor planning cost without loading MLX or a model."""

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

from mtplx.semantic_anchors import (  # noqa: E402
    plan_semantic_anchors,
    render_message_prefixes,
)


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _fixture(
    message_count: int,
    tokens_per_message: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    messages: list[dict[str, Any]] = []
    tokens: list[int] = []
    roles = ("system", "user", "assistant", "tool")
    for index in range(message_count):
        role = roles[index % len(roles)]
        message: dict[str, Any] = {"role": role, "content": f"message-{index}"}
        if index and index % 31 == 0:
            message["metadata"] = {"compaction_end": True}
        messages.append(message)
        tokens.extend(((index + 1) * 1000 + offset) for offset in range(tokens_per_message))
    return messages, tokens


def _measure(message_count: int, iterations: int, tokens_per_message: int) -> dict[str, Any]:
    messages, final_tokens = _fixture(message_count, tokens_per_message)
    candidate_indexes = tuple(
        sorted({0, *range(max(0, message_count - 15), message_count)})
    )

    def render(prefix: list[dict[str, Any]]) -> list[int]:
        return final_tokens[: len(prefix) * tokens_per_message]

    samples_us: list[float] = []
    final_plan = None
    for _ in range(iterations):
        started = time.perf_counter_ns()
        rendered = render_message_prefixes(
            messages,
            render,
            message_indexes=candidate_indexes,
            estimated_checkpoint_bytes=4096,
        )
        final_plan = plan_semantic_anchors(
            final_tokens,
            rendered,
            template_hash="measurement-fixture-v1",
            max_anchors=8,
            max_checkpoint_bytes=8 * 4096,
            default_checkpoint_bytes=4096,
        )
        samples_us.append((time.perf_counter_ns() - started) / 1000.0)

    assert final_plan is not None
    digest = hashlib.sha256(
        json.dumps(final_plan.to_metrics(), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "messages": message_count,
        "tokens": len(final_tokens),
        "iterations": iterations,
        "planner_us": {
            "mean": statistics.fmean(samples_us),
            "p50": _percentile(samples_us, 0.50),
            "p95": _percentile(samples_us, 0.95),
            "p99": _percentile(samples_us, 0.99),
        },
        "selected_edges": len(final_plan.anchors),
        "rejected_edges": len(final_plan.rejected),
        "estimated_checkpoint_bytes": final_plan.estimated_checkpoint_bytes,
        "plan_digest": digest,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages", default="32,128,512")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--tokens-per-message", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    message_counts = [int(value) for value in args.messages.split(",")]
    receipt = {
        "measurement": "semantic_anchor_planner",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "fixture": "synthetic complete-message token prefixes",
        "results": [
            _measure(count, args.iterations, args.tokens_per_message)
            for count in message_counts
        ],
    }
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
