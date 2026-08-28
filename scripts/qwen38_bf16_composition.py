#!/usr/bin/env python3
"""Stack one qualified Qwen 3.8 BF16 optimization onto a candidate.

Each invocation owns one cumulative control/candidate step and runs the exact
1K and 16K ABBA brackets.  The control route is the previously retained
candidate; the candidate route adds exactly one construction-time feature (or
one explicitly opted-in frozen feature).  Bundle comparisons are deliberately
separate because they require a different receipt contract.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import qwen38_challenge_port_isolated_gate as isolated  # noqa: E402
from scripts import qwen38_native_mtp_campaign as campaign  # noqa: E402
from scripts import qwen38_native_mtp_candidates as candidates  # noqa: E402
from scripts import qwen38_optimization_audit as audit  # noqa: E402
from scripts.qwen38_native_mtp_campaign import (  # noqa: E402
    DEFAULT_CONTEXT,
    DEFAULT_LOCK,
    DEFAULT_PROMPT,
    EXACT_CONDITIONER_TOKENS,
    EXACT_OUTPUT_TOKENS,
    EXACT_SEED,
    EXACT_TEMPERATURE,
    EXACT_TOP_K,
    EXACT_TOP_P,
)

COMPOSITION_CONTEXT_TOKENS = (16_384,)


@dataclass(frozen=True)
class CompositionStep:
    step_id: str
    feature: str
    phase: str
    control_route: str
    candidate_route: str
    allow_frozen_candidate: bool


def step_from_args(args: argparse.Namespace) -> CompositionStep:
    return CompositionStep(
        step_id=str(args.step_id),
        feature=str(args.feature),
        phase=str(args.phase),
        control_route=str(args.control_route),
        candidate_route=str(args.candidate_route),
        allow_frozen_candidate=bool(args.allow_frozen_candidate),
    )


def validate_step(step: CompositionStep) -> candidates.NativeMTPRouteDelta:
    if not step.step_id or not step.step_id.strip():
        raise ValueError("step ID must be non-empty")
    if step.phase not in {"prefill", "decode", "mixed"}:
        raise ValueError("phase must be prefill, decode, or mixed")
    route_features = (
        candidates.canonicalize_native_mtp_route(step.control_route)
        | candidates.canonicalize_native_mtp_route(step.candidate_route)
    ) - {"control"}
    artifact_features = {
        feature
        for feature in route_features
        if feature in candidates.NATIVE_MTP_CANDIDATES
        and candidates.NATIVE_MTP_CANDIDATES[feature].ownership == "artifact"
    }
    if artifact_features:
        raise ValueError(
            "BF16-only composition cannot contain custom Q4 artifact features: "
            + ", ".join(sorted(artifact_features))
        )
    delta = candidates.validate_native_mtp_route_delta(
        step.control_route,
        step.candidate_route,
        allow_frozen_candidate=step.allow_frozen_candidate,
    )
    if delta.candidate_feature != step.feature:
        raise ValueError(
            "feature label does not match the construction-time route delta"
        )
    return delta


def build_execution_plan(
    step: CompositionStep,
) -> tuple[audit.ExecutionItem, ...]:
    validate_step(step)
    return tuple(
        audit.ExecutionItem(
            case_id=step.step_id,
            feature=step.feature,
            control_route=step.control_route,
            candidate_route=step.candidate_route,
            context_tokens=context_tokens,
            allow_frozen_candidate=step.allow_frozen_candidate,
        )
        for context_tokens in COMPOSITION_CONTEXT_TOKENS
    )


_SAFE_STEP_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def output_paths(
    *,
    step: CompositionStep,
    plan: Sequence[audit.ExecutionItem],
    output_dir: Path,
    output: Path,
) -> tuple[Path, ...]:
    """Precompute collision-free fresh receipt paths before GPU ownership."""

    if _SAFE_STEP_ID.fullmatch(step.step_id) is None:
        raise ValueError(
            "step ID must be filename-safe lowercase words separated by hyphens"
        )
    if output_dir.exists() and not output_dir.is_dir():
        raise RuntimeError("composition output directory must be a directory")
    raw_paths = tuple(
        output_dir / f"{index:02d}-{step.step_id}-{item.context_tokens}.json"
        for index, item in enumerate(plan, start=1)
    )
    final_paths = (output, *raw_paths)
    temporary_paths = tuple(
        path.with_suffix(path.suffix + ".tmp") for path in final_paths
    )
    resolved = [path.resolve() for path in (*final_paths, *temporary_paths)]
    if len(resolved) != len(set(resolved)):
        raise ValueError(
            "campaign, raw receipt, and temporary paths must be distinct"
        )
    for path in (*final_paths, *temporary_paths):
        parent = path.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        if not parent.is_dir():
            raise RuntimeError(
                f"destination existing parent must be a directory: {parent}"
            )
    if output_dir.exists() and any(output_dir.rglob("*")):
        raise RuntimeError("composition output directory must be recursively empty")
    existing = [path for path in (*final_paths, *temporary_paths) if path.exists()]
    if existing:
        raise RuntimeError(
            "composition output paths must not exist: "
            + ", ".join(str(path) for path in existing)
        )
    return raw_paths


def campaign_payload(
    *,
    step: CompositionStep,
    plan: Sequence[audit.ExecutionItem],
    results: Sequence[dict[str, object]],
    source_commit: str,
    model_artifact_hashes: dict[str, str],
    lock_scope: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "qwen38_bf16_cumulative_composition",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "complete": len(results) == len(plan),
        "step": asdict(step),
        "phase": step.phase,
        "source_commit": source_commit,
        "mlx_version": "0.32.2",
        "mlx_metal_version": "0.32.2",
        "model_artifact_hashes": model_artifact_hashes,
        "gpu_lock_scope": lock_scope,
        "protocol": {
            "contexts": list(COMPOSITION_CONTEXT_TOKENS),
            "conditioner_tokens": EXACT_CONDITIONER_TOKENS,
            "output_tokens": EXACT_OUTPUT_TOKENS,
            "temperature": EXACT_TEMPERATURE,
            "top_p": EXACT_TOP_P,
            "top_k": EXACT_TOP_K,
            "seed": EXACT_SEED,
            "order": "ABBA",
            "cold_timed_prompt": True,
            "prefix_or_session_restore": False,
        },
        "execution_plan": [asdict(item) for item in plan],
        "results": list(results),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-id", required=True)
    parser.add_argument("--feature", required=True)
    parser.add_argument("--phase", choices=("prefill", "decode", "mixed"), required=True)
    parser.add_argument("--control-route", required=True)
    parser.add_argument("--candidate-route", required=True)
    parser.add_argument("--allow-frozen-candidate", action="store_true")
    parser.add_argument("--model", type=Path, default=audit.gate.DEFAULT_MODEL)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--context-file", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--max-tokens", type=int, default=EXACT_OUTPUT_TOKENS)
    parser.add_argument(
        "--warmup-tokens", type=int, default=EXACT_CONDITIONER_TOKENS
    )
    parser.add_argument("--seed", type=int, default=EXACT_SEED)
    parser.add_argument("--temperature", type=float, default=EXACT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=EXACT_TOP_P)
    parser.add_argument("--top-k", type=int, default=EXACT_TOP_K)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _result_entry(
    *,
    step: CompositionStep,
    item: audit.ExecutionItem,
    receipt: dict[str, Any],
    raw_path: Path,
) -> dict[str, object]:
    return {
        **asdict(item),
        "phase": step.phase,
        "raw_receipt": str(raw_path.resolve()),
        "candidate_improvement_pct": receipt["candidate_improvement_pct"],
        "phase_summary": receipt["phase_summary"],
        "promotion": receipt["promotion"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # The shared command builder accepts artifact options, but this driver is
    # intentionally BF16-only and never installs a custom Q4 MTP block.
    args.row17_artifact = None
    args.row28_artifact = None
    args.row36_artifact = None
    step = step_from_args(args)
    plan = build_execution_plan(step)
    raw_paths = output_paths(
        step=step,
        plan=plan,
        output_dir=args.output_dir,
        output=args.output,
    )
    workload_errors = campaign._exact_workload_errors(args)
    if workload_errors:
        raise ValueError("; ".join(workload_errors))
    source_status = subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    ).splitlines()
    if source_status:
        raise RuntimeError("exact composition step requires a clean source tree")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    results: list[dict[str, object]] = []

    with isolated._gpu_lock_scope(args.lock) as lock_scope:
        model_path = args.model.expanduser().resolve()
        model_artifact_hashes = audit._parent_model_artifact_hashes(model_path)
        environment = dict(os.environ)
        environment[audit.gate.MODEL_ARTIFACT_HASHES_ENV] = json.dumps(
            model_artifact_hashes,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for index, (item, raw_path) in enumerate(
            zip(plan, raw_paths, strict=True), start=1
        ):
            print(
                f"START {index}/{len(plan)} {step.step_id} "
                f"context={item.context_tokens}",
                flush=True,
            )
            process = isolated._run_attested_child(
                audit._isolated_command(args, item, raw_path),
                environment=environment,
                lock_path=args.lock,
                owns_process_group=lock_scope == "direct",
            )
            if process.returncode not in (0, 2) or not raw_path.is_file():
                raise RuntimeError(
                    f"{step.step_id} context {item.context_tokens} failed "
                    f"({process.returncode}):\n{process.stdout}"
                )
            receipt = json.loads(raw_path.read_text(encoding="utf-8"))
            receipt_errors = audit._receipt_errors(
                item,
                receipt,
                expected_source_commit=source_commit,
                expected_model_artifact_hashes=model_artifact_hashes,
            )
            if receipt_errors:
                raise RuntimeError(
                    f"{step.step_id} context {item.context_tokens} receipt "
                    f"rejected: {'; '.join(receipt_errors)}"
                )
            results.append(
                _result_entry(
                    step=step,
                    item=item,
                    receipt=receipt,
                    raw_path=raw_path,
                )
            )
            _write_json(
                args.output,
                campaign_payload(
                    step=step,
                    plan=plan,
                    results=results,
                    source_commit=source_commit,
                    model_artifact_hashes=model_artifact_hashes,
                    lock_scope=lock_scope,
                ),
            )
            print(
                f"DONE {index}/{len(plan)} {step.step_id} "
                f"context={item.context_tokens} "
                f"wall_delta={float(receipt['candidate_improvement_pct']):+.4f}%",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
