#!/usr/bin/env python3
"""Exact dual-context, phase-aware native-MTP campaign driver.

This parent process never imports MLX.  It owns the exclusive GPU lock and
delegates it through the existing attested four-process ABBA gate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load script module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_script(
    "qwen38_challenge_port_gate_campaign",
    ROOT / "scripts/qwen38_challenge_port_gate.py",
)
isolated = _load_script(
    "qwen38_challenge_port_isolated_gate_campaign",
    ROOT / "scripts/qwen38_challenge_port_isolated_gate.py",
)

EXACT_CONTEXT_TOKENS = (1_024, 16_384)
EXACT_OUTPUT_TOKENS = 1_024
EXACT_CONDITIONER_TOKENS = 1_024
EXACT_TEMPERATURE = 1.0
EXACT_TOP_P = 0.95
EXACT_TOP_K = 20
EXACT_SEED = 42
DEFAULT_PROMPT = gate.DEFAULT_PROMPT
DEFAULT_CONTEXT = gate.DEFAULT_CONTEXT
DEFAULT_LOCK = Path("/tmp/mtplx-gpu-exclusive.lock")
PROMOTION_THRESHOLD_PCT = gate.PROMOTION_THRESHOLD_PCT

PURE_DECODE_FEATURES = frozenset(
    {"r08_device_draft", "r10_compact_vocab", "r11_position_ema"}
)
SHARED_BLOCK_FEATURES = frozenset(
    {"r17_q4_mtp_block", "r28_q4_mtp_block", "r36_qkv_islands"}
)
SHARED_INPUT_FEATURES = frozenset(
    {"r61_dual_norm_concat", "r63_q8_embedding_dual_norm"}
)
ROW20_FEATURE = "r20_kv_only_history"


def _read_first_prompt_id(path: Path) -> str:
    return gate._read_prompt(path)[0]


def _exact_workload_errors(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    checks = (
        (Path(args.prompt_file).resolve() == DEFAULT_PROMPT.resolve(), "prompt file must be the exact python_modules_long.jsonl corpus"),
        (Path(args.context_file).resolve() == DEFAULT_CONTEXT.resolve(), "context file must be mtplx/generation.py"),
        (int(args.max_tokens) == EXACT_OUTPUT_TOKENS, "output must be exactly 1024 tokens"),
        (int(args.warmup_tokens) == EXACT_CONDITIONER_TOKENS, "conditioner must be exactly 1024 tokens"),
        (int(args.seed) == EXACT_SEED, "seed must be exactly 42"),
        (float(args.temperature) == EXACT_TEMPERATURE, "temperature must be exactly 1.0"),
        (float(args.top_p) == EXACT_TOP_P, "top-p must be exactly 0.95"),
        (int(args.top_k) == EXACT_TOP_K, "top-k must be exactly 20"),
        (Path(args.lock).resolve() == DEFAULT_LOCK.resolve(), "GPU lock must be /tmp/mtplx-gpu-exclusive.lock"),
    )
    errors.extend(message for passed, message in checks if not passed)
    try:
        _read_first_prompt_id(Path(args.prompt_file))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"first Python prompt row is unreadable: {exc}")
    return errors


def _phase_contract(
    candidate_feature: str,
    context_tokens: int,
    *,
    control_features: set[str] | frozenset[str],
) -> dict[str, str]:
    if context_tokens not in EXACT_CONTEXT_TOKENS:
        raise ValueError(f"unsupported exact context: {context_tokens}")
    if candidate_feature in PURE_DECODE_FEATURES:
        return {"target_prefill": "frozen", "mtp_history": "unaffected", "mtp_decode": "full"}
    if candidate_feature == ROW20_FEATURE:
        return {
            "target_prefill": "frozen",
            "mtp_history": "stock" if context_tokens == 1_024 else "full",
            "mtp_decode": "unaffected",
        }
    row20_active = ROW20_FEATURE in control_features and context_tokens >= 16_384
    if candidate_feature in SHARED_BLOCK_FEATURES:
        return {
            "target_prefill": "frozen",
            "mtp_history": "partial" if row20_active else "full",
            "mtp_decode": "full",
        }
    if candidate_feature in SHARED_INPUT_FEATURES:
        return {
            "target_prefill": "frozen",
            "mtp_history": "bypass" if row20_active else "full",
            "mtp_decode": "full",
        }
    raise ValueError(f"unsupported native-MTP campaign feature: {candidate_feature}")


def _candidate_row20_states(receipt: dict[str, Any]) -> list[bool]:
    candidate_id = receipt.get("candidate_route_id") or receipt.get(
        "candidate_feature"
    )
    return [
        bool((arm.get("history_route_receipt") or {}).get("row20_engaged"))
        for arm in receipt.get("arms") or ()
        if arm.get("route_id") == candidate_id
        or receipt.get("candidate_route_id") is None
        and arm.get("route_id") == receipt.get("candidate_feature")
    ]


def _context_decision(
    candidate_feature: str,
    receipt: dict[str, Any],
    *,
    control_features: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    context_tokens = int(receipt.get("prompt_token_target", 0))
    contract = _phase_contract(
        candidate_feature,
        context_tokens,
        control_features=control_features,
    )
    errors = list(receipt.get("receipt_invariant_errors") or ())
    errors.extend(receipt.get("candidate_engagement_errors") or ())
    if receipt.get("source_status"):
        errors.append("aggregate receipt requires a clean source tree")
    if not bool((receipt.get("correctness") or {}).get("passed")):
        errors.append("correctness/determinism gate did not pass")
    phase_summary = receipt.get("phase_summary") or {}
    time_improvements = phase_summary.get("time_improvement_pct") or {}
    throughput_improvements = (
        phase_summary.get("throughput_improvement_pct") or {}
    )
    invalid_improvements: set[tuple[str, str]] = set()
    for namespace, values in (
        ("elapsed-time", time_improvements),
        ("throughput", throughput_improvements),
    ):
        for phase, raw_value in values.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                value = float("nan")
            if not math.isfinite(value):
                errors.append(f"{phase} {namespace} improvement must be finite")
                invalid_improvements.add((namespace, phase))

    def require_time_win(phase: str) -> None:
        if ("elapsed-time", phase) in invalid_improvements:
            return
        value = float(time_improvements.get(phase, float("-inf")))
        if not math.isfinite(value):
            errors.append(f"{phase} elapsed-time improvement must be finite")
            return
        if value <= PROMOTION_THRESHOLD_PCT:
            errors.append(
                f"{phase} elapsed-time improvement must be strictly greater than "
                f"{PROMOTION_THRESHOLD_PCT:.2f}%"
            )

    def require_throughput_win(phase: str) -> None:
        if ("throughput", phase) in invalid_improvements:
            return
        value = float(throughput_improvements.get(phase, float("-inf")))
        if not math.isfinite(value):
            errors.append(f"{phase} throughput improvement must be finite")
            return
        if value <= PROMOTION_THRESHOLD_PCT:
            errors.append(
                f"{phase} throughput improvement must be strictly greater than "
                f"{PROMOTION_THRESHOLD_PCT:.2f}%"
            )

    if candidate_feature == "r11_position_ema":
        require_time_win("wall")
        require_throughput_win("decode")
    elif candidate_feature in PURE_DECODE_FEATURES:
        require_time_win("wall")
        require_throughput_win("mtp_decode")
    elif candidate_feature in SHARED_BLOCK_FEATURES | SHARED_INPUT_FEATURES:
        require_time_win("wall")
        require_throughput_win("mtp_decode")
    elif candidate_feature == ROW20_FEATURE:
        states = _candidate_row20_states(receipt)
        if context_tokens == 1_024:
            if len(states) != 2 or any(states):
                errors.append("row 20 must remain stock and unengaged at 1K")
        else:
            require_time_win("wall")
            if len(states) != 2 or not all(states):
                errors.append("row 20 must engage in both candidate arms at 16K")
            require_throughput_win("mtp_history")
            proposer_improvement = float(
                throughput_improvements.get("mtp_decode", float("-inf"))
            )
            if not math.isfinite(proposer_improvement):
                if ("throughput", "mtp_decode") not in invalid_improvements:
                    errors.append("mtp_decode throughput improvement must be finite")
            elif proposer_improvement < 0.0:
                errors.append("row 20 must not regress MTP decode")
    return {
        "passed": not errors,
        "candidate_feature": candidate_feature,
        "context_tokens": context_tokens,
        "phase_contract": contract,
        "threshold_pct": PROMOTION_THRESHOLD_PCT,
        "errors": errors,
    }


def _run_contexts(
    candidate_feature: str,
    run_context: Callable[[int], dict[str, Any]],
    *,
    control_features: set[str] | frozenset[str] = frozenset(),
    stop_on_failure: bool = True,
) -> dict[str, Any]:
    contexts: list[dict[str, Any]] = []
    aborted_after: int | None = None
    for context_tokens in EXACT_CONTEXT_TOKENS:
        receipt = run_context(context_tokens)
        decision = _context_decision(
            candidate_feature,
            receipt,
            control_features=control_features,
        )
        contexts.append({"receipt": receipt, "decision": decision})
        if not decision["passed"] and stop_on_failure:
            aborted_after = context_tokens
            break
    return {
        "candidate_feature": candidate_feature,
        "passed": len(contexts) == len(EXACT_CONTEXT_TOKENS)
        and all(item["decision"]["passed"] for item in contexts),
        "aborted_after_context": aborted_after,
        "contexts": contexts,
    }


def _isolated_command(
    args: argparse.Namespace,
    *,
    context_tokens: int,
    output: Path,
) -> list[str]:
    order = ",".join(
        (
            args.control_route,
            args.candidate_route,
            args.candidate_route,
            args.control_route,
        )
    )
    command = [
        sys.executable,
        str(ROOT / "scripts/qwen38_challenge_port_isolated_gate.py"),
        "--model",
        str(args.model),
        "--prompt-file",
        str(args.prompt_file),
        "--prompt-tokens",
        str(context_tokens),
        "--context-file",
        str(args.context_file),
        "--max-tokens",
        str(args.max_tokens),
        "--warmup-tokens",
        str(args.warmup_tokens),
        "--seed",
        str(args.seed),
        "--target-temperature",
        str(args.temperature),
        "--draft-temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--order",
        order,
        "--control-route",
        args.control_route,
        "--candidate-route",
        args.candidate_route,
        "--lock",
        str(args.lock),
        "--output",
        str(output),
    ]
    for flag, value in (
        ("--row17-artifact", args.row17_artifact),
        ("--row28-artifact", args.row28_artifact),
        ("--row36-artifact", args.row36_artifact),
    ):
        if value is not None:
            command.extend((flag, str(value)))
    return command


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=gate.DEFAULT_MODEL)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--context-file", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--max-tokens", type=int, default=EXACT_OUTPUT_TOKENS)
    parser.add_argument("--warmup-tokens", type=int, default=EXACT_CONDITIONER_TOKENS)
    parser.add_argument("--seed", type=int, default=EXACT_SEED)
    parser.add_argument("--temperature", type=float, default=EXACT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=EXACT_TOP_P)
    parser.add_argument("--top-k", type=int, default=EXACT_TOP_K)
    parser.add_argument("--control-route", required=True)
    parser.add_argument("--candidate-route", required=True)
    parser.add_argument("--row17-artifact", type=Path)
    parser.add_argument("--row28-artifact", type=Path)
    parser.add_argument("--row36-artifact", type=Path)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    workload_errors = _exact_workload_errors(args)
    if workload_errors:
        raise ValueError("; ".join(workload_errors))
    delta = gate.validate_native_mtp_route_delta(
        args.control_route, args.candidate_route
    )
    source_status = subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    ).splitlines()
    if source_status:
        raise RuntimeError("exact campaign requires a clean source tree")
    with tempfile.TemporaryDirectory(prefix="qwen38-native-mtp-") as temp_dir:
        temp_root = Path(temp_dir)
        campaign_environment = dict(os.environ)

        def run_context(context_tokens: int) -> dict[str, Any]:
            child_output = temp_root / f"{context_tokens}.json"
            process = isolated._run_attested_child(
                _isolated_command(
                    args,
                    context_tokens=context_tokens,
                    output=child_output,
                ),
                environment=campaign_environment,
                lock_path=args.lock,
                owns_process_group=lock_scope == "direct",
            )
            if process.returncode not in (0, 2) or not child_output.is_file():
                raise RuntimeError(
                    f"isolated {context_tokens} bracket failed "
                    f"({process.returncode}):\n{process.stdout}"
                )
            return json.loads(child_output.read_text(encoding="utf-8"))

        with isolated._gpu_lock_scope(args.lock) as lock_scope:
            if lock_scope == "direct":
                model_artifact_hashes = gate._model_artifact_hashes(
                    args.model.expanduser().resolve()
                )
            else:
                model_artifact_hashes = gate._attested_model_artifact_hashes(
                    args.model.expanduser().resolve(),
                    guarded_by_parent=True,
                )
            campaign_environment[gate.MODEL_ARTIFACT_HASHES_ENV] = json.dumps(
                model_artifact_hashes,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            result = _run_contexts(
                delta.candidate_feature,
                run_context,
                control_features=delta.control_features,
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for item in result["contexts"]:
        context_tokens = item["decision"]["context_tokens"]
        context_output = args.output_dir / (
            f"{delta.candidate_feature}-{context_tokens}-mlx0322.json"
        )
        context_output.write_text(
            json.dumps(
                item["receipt"], indent=2, sort_keys=True, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
    result.update(
        {
            "kind": "qwen38_native_mtp_campaign",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "control_route_id": args.control_route,
            "candidate_route_id": args.candidate_route,
            "gpu_lock_scope": lock_scope,
            "gpu_lock_path": str(args.lock.resolve()),
            "model_artifact_hashes": model_artifact_hashes,
            "exact_context_tokens": list(EXACT_CONTEXT_TOKENS),
            "exact_output_tokens": EXACT_OUTPUT_TOKENS,
            "exact_conditioner_tokens_per_arm": EXACT_CONDITIONER_TOKENS,
            "prompt_id": _read_first_prompt_id(args.prompt_file),
            "phase_results": [
                {
                    "context_tokens": item["decision"]["context_tokens"],
                    "phase_contract": item["decision"]["phase_contract"],
                    "time_improvement_pct": item["receipt"]["phase_summary"][
                        "time_improvement_pct"
                    ],
                    "throughput_improvement_pct": item["receipt"][
                        "phase_summary"
                    ]["throughput_improvement_pct"],
                }
                for item in result["contexts"]
            ],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "candidate_feature": delta.candidate_feature,
                "passed": result["passed"],
                "aborted_after_context": result["aborted_after_context"],
                "output": str(args.output),
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
