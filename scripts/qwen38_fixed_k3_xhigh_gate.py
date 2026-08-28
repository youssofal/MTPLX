#!/usr/bin/env python3
"""Matched 16K-context/xhigh fixed-K3 route attribution gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import qwen38_challenge_port_gate as route_gate  # noqa: E402
from scripts import qwen38_native_mtp_matrix as matrix  # noqa: E402


WORKLOAD = "xhigh"
CONTEXT_TOKENS = 16_384
FORBIDDEN_FIXED_FEATURES = frozenset(
    {
        "r11_position_ema",
        "r17_q4_mtp_block",
        "r28_q4_mtp_block",
        "r36_qkv_islands",
    }
)


def _lane_id(route: str) -> str:
    return "candidate-" + route.replace("_", "-").replace("+", "--")


def _validate_fixed_route(route: str) -> None:
    if route == "control":
        return
    features = route_gate._validate_route_id(route)
    forbidden = features & FORBIDDEN_FIXED_FEATURES
    if forbidden:
        raise ValueError(
            "fixed BF16 diagnostic route contains adaptive or Q4 features: "
            + ", ".join(sorted(forbidden))
        )


def build_lane_specs(
    *,
    baseline_root: Path,
    baseline_commit: str,
    candidate_root: Path,
    candidate_commit: str,
    routes: Sequence[str],
) -> dict[str, matrix.LaneSpec]:
    if baseline_commit != matrix.V292_COMMIT:
        raise ValueError("baseline source is not exact v2.9.2")
    if not routes:
        raise ValueError("at least one current-source fixed route is required")
    if len(set(routes)) != len(routes):
        raise ValueError("current-source fixed routes must be unique")
    for route in routes:
        _validate_fixed_route(route)
    specs = {
        "v2.9.2-mlx0322": matrix.LaneSpec(
            "v2.9.2-mlx0322",
            baseline_root,
            baseline_commit,
            "control",
        )
    }
    for route in routes:
        lane_id = _lane_id(route)
        if lane_id in specs:
            raise ValueError(f"diagnostic lane ID collision: {lane_id}")
        specs[lane_id] = matrix.LaneSpec(
            lane_id,
            candidate_root,
            candidate_commit,
            route,
        )
    return specs


def paired_order(specs: Mapping[str, matrix.LaneSpec]) -> tuple[str, ...]:
    lane_ids = tuple(specs)
    return (*lane_ids, *reversed(lane_ids))


def arm_command(
    *,
    lane: matrix.LaneSpec,
    output: Path,
    model: Path,
    prompt_file: Path,
    context_file: Path,
    row17_artifact: Path,
    python: Path,
    lock: Path,
) -> list[str]:
    command = matrix.child_command(
        lane=lane,
        workload=WORKLOAD,
        context_tokens=CONTEXT_TOKENS,
        output=output,
        model=model,
        prompt_file=prompt_file,
        context_file=context_file,
        row17_artifact=row17_artifact,
        python=python,
        lock=lock,
    )
    if lane.lane_id != "v2.9.2-mlx0322":
        command.append("--allow-fixed-diagnostic-route")
    return command


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def aggregate(
    *,
    order: tuple[str, ...],
    receipts: list[dict[str, Any]],
    specs: Mapping[str, matrix.LaneSpec],
) -> dict[str, Any]:
    errors: list[str] = []
    if order != paired_order(specs):
        errors.append("route order is not the exact forward/reverse pair")
    if len(receipts) != len(order):
        errors.append(f"expected {len(order)} arms, found {len(receipts)}")
    for index, (lane_id, receipt) in enumerate(zip(order, receipts, strict=False)):
        lane = specs.get(lane_id)
        if lane is None:
            errors.append(f"arm {index}: unknown lane {lane_id}")
            continue
        errors.extend(
            f"arm {index}: {error}"
            for error in matrix.receipt_errors(
                receipt,
                lane=lane,
                context_tokens=CONTEXT_TOKENS,
                output_tokens=matrix.XHIGH_OUTPUT_TOKENS,
            )
        )
    for key in (
        "prompt_token_sha256",
        "prompt_artifact_sha256",
        "context_artifact_sha256",
        "model_artifact_hashes",
        "row17_artifact_sha256",
    ):
        values = {json.dumps(row.get(key), sort_keys=True) for row in receipts}
        if len(values) != 1:
            errors.append(f"{key} changed across arms")

    summary: dict[str, dict[str, Any]] = {}
    for lane_id, spec in specs.items():
        rows = [row for row in receipts if row.get("lane_id") == lane_id]
        if len(rows) != 2:
            errors.append(f"{lane_id} has {len(rows)} arms, expected 2")
            continue
        hashes = {str(row["token_hash"]) for row in rows}
        deterministic = len(hashes) == 1
        if not deterministic:
            errors.append(f"{lane_id} token nondeterminism across paired arms")
        summary[lane_id] = {
            "source_commit": spec.source_commit,
            "route_id": spec.route_id,
            "prefill_tok_s_mean": _mean(rows, "prefill_tok_s"),
            "decode_tok_s_mean": _mean(rows, "decode_tok_s"),
            "wall_s_mean": _mean(rows, "wall_s"),
            "peak_memory_gib_max": max(float(row["peak_memory_gib"]) for row in rows),
            "verify_calls_mean": _mean(rows, "verify_calls"),
            "bonus_tokens_mean": _mean(rows, "bonus_tokens"),
            "correction_tokens_mean": _mean(rows, "correction_tokens"),
            "draft_time_s_mean": _mean(rows, "draft_time_s"),
            "per_lane_token_deterministic": deterministic,
            "token_hashes": sorted(hashes),
        }
    return {
        "schema_version": 1,
        "kind": "qwen38_fixed_k3_xhigh_route_attribution",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workload": WORKLOAD,
        "context_tokens": CONTEXT_TOKENS,
        "conditioner_output_tokens": matrix.CONDITIONER_OUTPUT_TOKENS,
        "timed_output_tokens": matrix.XHIGH_OUTPUT_TOKENS,
        "order": list(order),
        "invariant_errors": errors,
        "summary": summary,
        "arms": receipts,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, default=ROOT)
    parser.add_argument("--routes", nargs="+", required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, default=matrix.PYTHON_PROMPT_FILE)
    parser.add_argument("--context-file", type=Path, required=True)
    parser.add_argument("--row17-artifact", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path("/tmp/mtplx-gpu-exclusive.lock"))
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    args.workload = WORKLOAD
    matrix._assert_campaign_inputs(args)
    if args.output_root.exists() and any(args.output_root.rglob("*")):
        raise RuntimeError("diagnostic output root must be empty")
    versions = matrix._interpreter_versions(args.python)
    if versions != {
        "mlx": matrix.REQUIRED_MLX_VERSION,
        "mlx_metal": matrix.REQUIRED_MLX_METAL_VERSION,
    }:
        raise RuntimeError(f"benchmark interpreter versions mismatch: {versions}")
    baseline_commit = matrix._git_commit(args.baseline_root)
    candidate_commit = matrix._git_commit(args.candidate_root)
    specs = build_lane_specs(
        baseline_root=args.baseline_root,
        baseline_commit=baseline_commit,
        candidate_root=args.candidate_root,
        candidate_commit=candidate_commit,
        routes=tuple(args.routes),
    )
    order = paired_order(specs)
    isolated = matrix._load_isolated()
    args.output_root.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []
    with isolated._gpu_lock_scope(args.lock) as lock_scope:
        matrix._validated_parent_guard_scope(lock_scope)
        model_hashes = route_gate._model_artifact_hashes(args.model)
        for index, lane_id in enumerate(order):
            lane = specs[lane_id]
            output = args.output_root / f"arm-{index}-{lane_id}.json"
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(lane.source_root)
            environment[matrix.MODEL_HASHES_ENV] = json.dumps(
                model_hashes, sort_keys=True, separators=(",", ":")
            )
            command = arm_command(
                lane=lane,
                output=output,
                model=args.model,
                prompt_file=args.prompt_file,
                context_file=args.context_file,
                row17_artifact=args.row17_artifact,
                python=args.python,
                lock=args.lock,
            )
            result = isolated._run_attested_child(
                command,
                environment=isolated._environment_for_route(
                    lane.route_id, environment
                ),
                lock_path=args.lock,
                owns_process_group=True,
            )
            output.with_suffix(".log").write_text(
                result.stdout or "", encoding="utf-8"
            )
            if result.returncode != 0 or not output.is_file():
                raise RuntimeError(f"diagnostic arm {index} failed")
            receipt = json.loads(output.read_text(encoding="utf-8"))
            receipts.append(receipt)
            print(
                json.dumps(
                    {
                        "event": "arm_complete",
                        "arm": index + 1,
                        "lane": lane_id,
                        "wall_s": receipt["wall_s"],
                    }
                ),
                flush=True,
            )
    combined = aggregate(order=order, receipts=receipts, specs=specs)
    _write_json(args.output_root / "combined.json", combined)
    return 0 if not combined["invariant_errors"] else 2


def main(argv: Sequence[str] | None = None) -> int:
    return run(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
