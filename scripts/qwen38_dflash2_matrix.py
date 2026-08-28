#!/usr/bin/env python3
"""Run the final PR335 DFlash2 comparator matrix under one GPU lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, NamedTuple

try:
    from scripts import qwen38_dflash2_comparator_arm as comparator
    from scripts import qwen38_native_mtp_matrix as native_matrix
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    import qwen38_dflash2_comparator_arm as comparator  # type: ignore[no-redef]
    import qwen38_native_mtp_matrix as native_matrix  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
ARM_SCRIPT = ROOT / "scripts/qwen38_dflash2_comparator_arm.py"
ISOLATED_SCRIPT = ROOT / "scripts/qwen38_challenge_port_isolated_gate.py"
PR335_SOURCE_COMMIT = comparator.PR335_SOURCE_COMMIT
REQUIRED_MLX_VERSION = "0.32.2"
PROMPT_ARTIFACT_SHA256 = native_matrix.PROMPT_ARTIFACT_SHA256
PROMPT_TOKEN_SHA256 = native_matrix.PROMPT_TOKEN_SHA256
PYTHON_CONTEXT_SHA256 = native_matrix.PYTHON_CONTEXT_SHA256
LOW_CONTEXTS = (1_024, 16_384, 65_536, 131_072)


class Scenario(NamedTuple):
    workload: str
    context_tokens: int
    max_tokens: int
    conditioner_tokens: int
    temperature: float
    top_p: float
    top_k: int
    prompt_kind: str
    enable_thinking: bool
    reasoning_effort: str | None


def scenario(workload: str, context_tokens: int) -> Scenario:
    if workload == "vanity":
        if context_tokens != 100:
            raise ValueError("DFlash2 vanity requires the exact 100-token prompt")
        return Scenario(
            workload, 100, 1_024, 0, 0.0, 1.0, 0,
            "is_palindrome", False, None,
        )
    if workload == "low":
        if context_tokens not in LOW_CONTEXTS:
            raise ValueError(f"unsupported DFlash2 low context: {context_tokens}")
        return Scenario(
            workload, context_tokens, 1_024, 1_024, 1.0, 0.95, 20,
            "coding", True, "low",
        )
    if workload == "xhigh":
        if context_tokens != 1_024:
            raise ValueError("DFlash2 xhigh is restricted to 1024 input tokens")
        return Scenario(
            workload, context_tokens, 1_024, 1_024, 1.0, 0.95, 20,
            "coding", True, "xhigh",
        )
    raise ValueError(f"unknown DFlash2 workload: {workload}")


def order_for_context(workload: str, context_tokens: int) -> tuple[str, ...]:
    scenario(workload, context_tokens)
    return ("dflash2",) if context_tokens == 131_072 else ("dflash2", "dflash2")


def child_command(
    *,
    python: Path,
    source_root: Path,
    source_commit: str,
    model: Path,
    draft: Path,
    prompt_file: Path,
    context_file: Path,
    lock: Path,
    output: Path,
    scenario: Scenario,
) -> list[str]:
    command = [
        str(python.absolute()),
        str(ARM_SCRIPT),
        "--engine", "pr_dflash2",
        "--source-root", str(source_root.resolve()),
        "--source-commit", source_commit,
        "--model", str(model.resolve()),
        "--draft", str(draft.resolve()),
        "--prompt-file", str(prompt_file.resolve()),
        "--context-file", str(context_file.resolve()),
        "--prompt-kind", scenario.prompt_kind,
        "--prompt-tokens", str(scenario.context_tokens),
        "--max-tokens", str(scenario.max_tokens),
        "--temperature", str(scenario.temperature),
        "--top-p", str(scenario.top_p),
        "--top-k", str(scenario.top_k),
        "--enable-thinking" if scenario.enable_thinking else "--no-enable-thinking",
        "--conditioner-tokens", str(scenario.conditioner_tokens),
        "--conditioner-mode", "same_prompt",
        "--dflash2-adaptive",
        "--draft-block-size", "8",
        "--seed", "42",
        "--lock", str(lock.resolve()),
        "--output", str(output.resolve()),
    ]
    if scenario.reasoning_effort is not None:
        command.extend(("--reasoning-effort", scenario.reasoning_effort))
    return command


def _finite_positive(value: Any, label: str, errors: list[str]) -> None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} is not numeric")
        return
    if not math.isfinite(number) or number <= 0:
        errors.append(f"{label} is not finite and positive")


def receipt_errors(
    receipt: dict[str, Any],
    *,
    scenario: Scenario,
    expected_harness_commit: str,
    lock: Path,
    model: Path,
    draft: Path,
) -> list[str]:
    errors = comparator._optimized_stack_errors(receipt)
    exact = {
        "kind": "qwen38_dflash2_frozen_matrix_arm",
        "engine": "pr_dflash2",
        "source_commit": PR335_SOURCE_COMMIT,
        "harness_commit": expected_harness_commit,
        "mlx_version": REQUIRED_MLX_VERSION,
        "mlx_metal_version": REQUIRED_MLX_VERSION,
        "gpu_lock_scope": str(lock.resolve()),
        "model": str(model.resolve()),
        "draft": str(draft.resolve()),
    }
    for key, expected in exact.items():
        if receipt.get(key) != expected:
            errors.append(f"DFlash2 {key} mismatch")

    workload = receipt.get("workload") or {}
    expected_prompt_format = {
        "vanity": "qwen_chat_template_non_thinking",
        "low": "qwen_chat_template_thinking_low",
        "xhigh": "qwen_chat_template_thinking_xhigh",
    }[scenario.workload]
    expected_workload = {
        "workload": scenario.workload,
        "prompt_kind": scenario.prompt_kind,
        "prompt_format": expected_prompt_format,
        "enable_thinking": scenario.enable_thinking,
        "reasoning_effort": scenario.reasoning_effort,
        "prompt_tokens": scenario.context_tokens,
        "prompt_token_sha256": PROMPT_TOKEN_SHA256[scenario.workload][
            scenario.context_tokens
        ],
        "prompt_artifact_sha256": PROMPT_ARTIFACT_SHA256[
            "vanity" if scenario.workload == "vanity" else "python"
        ],
        "context_artifact_sha256": PYTHON_CONTEXT_SHA256,
        "output_limit": scenario.max_tokens,
        "temperature": scenario.temperature,
        "top_p": scenario.top_p,
        "top_k": scenario.top_k,
        "seed": 42,
        "conditioner_prompt_tokens": scenario.context_tokens,
        "conditioner_output_tokens": scenario.conditioner_tokens,
        "conditioner_mode": "same_prompt",
        "conditioner_reuses_timed_prompt": True,
        "cold_prefill": True,
        "prefix_cache_used": False,
        "requested_adaptive": True,
        "requested_draft_m": 8,
    }
    for key, expected in expected_workload.items():
        if workload.get(key) != expected:
            errors.append(f"DFlash2 workload {key} mismatch")

    arm = receipt.get("arm") or {}
    arm_exact = {
        "engine": "pr_dflash2",
        "prompt_tokens": scenario.context_tokens,
        "prefix_cache_used": False,
        "cached_tokens": 0,
        "session_cache_hit": False,
        "session_restore_mode": "cold",
        "requested_adaptive": True,
        "effective_adaptive": True,
        "fallback_ar": False,
    }
    for key, expected in arm_exact.items():
        if arm.get(key) != expected:
            errors.append(f"DFlash2 arm {key} mismatch")
    generated = int(arm.get("generated_tokens", -1))
    if scenario.workload == "vanity":
        if not (0 < generated < scenario.max_tokens) or arm.get("finish_reason") != "stop":
            errors.append("DFlash2 vanity did not stop naturally")
    elif generated != scenario.max_tokens:
        errors.append("DFlash2 timed output token count is not exact")
    if not arm.get("effective_widths"):
        errors.append("DFlash2 adaptive widths did not execute")
    for key in ("prefill_tps", "decode_tps", "wall_s", "peak_memory_gib"):
        _finite_positive(arm.get(key), f"DFlash2 {key}", errors)
    token_hash = arm.get("token_sha256")
    if not isinstance(token_hash, str) or len(token_hash) != 64:
        errors.append("DFlash2 token hash is missing")
    return errors


def _mean(receipts: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(receipt["arm"][key]) for receipt in receipts)


def aggregate(
    *,
    scenario: Scenario,
    order: tuple[str, ...],
    receipts: list[dict[str, Any]],
    expected_harness_commit: str,
    lock: Path,
    model: Path,
    draft: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    if len(receipts) != len(order):
        errors.append(f"expected {len(order)} DFlash2 arms, found {len(receipts)}")
    for index, receipt in enumerate(receipts):
        errors.extend(
            f"arm {index}: {error}"
            for error in receipt_errors(
                receipt,
                scenario=scenario,
                expected_harness_commit=expected_harness_commit,
                lock=lock,
                model=model,
                draft=draft,
            )
        )
    hashes = {
        (receipt.get("arm") or {}).get("token_sha256") for receipt in receipts
    }
    if len(receipts) > 1 and len(hashes) != 1:
        errors.append("paired DFlash2 tokens are not deterministic")
    summary: dict[str, Any] = {}
    if receipts:
        summary = {
            "arms": len(receipts),
            "source_commit": PR335_SOURCE_COMMIT,
            "prefill_tok_s_mean": _mean(receipts, "prefill_tps"),
            "decode_tok_s_mean": _mean(receipts, "decode_tps"),
            "wall_s_mean": _mean(receipts, "wall_s"),
            "peak_memory_gib_max": max(
                float(receipt["arm"]["peak_memory_gib"])
                for receipt in receipts
            ),
            "generated_tokens": [
                int(receipt["arm"]["generated_tokens"])
                for receipt in receipts
            ],
            "paired_token_deterministic": len(hashes) == 1,
        }
    return {
        "schema_version": 1,
        "kind": "qwen38_dflash2_final_matrix",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workload": scenario.workload,
        "context_tokens": scenario.context_tokens,
        "conditioner_output_tokens": scenario.conditioner_tokens,
        "timed_output_tokens": scenario.max_tokens,
        "order": list(order),
        "invariant_errors": errors,
        "summary": summary,
        "arms": receipts,
    }


def _git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _git_status(root: Path) -> list[str]:
    return subprocess.check_output(
        ["git", "status", "--short"], cwd=root, text=True
    ).splitlines()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _interpreter_versions(python: Path) -> dict[str, str]:
    program = (
        "import importlib.metadata,json;"
        "print(json.dumps({'mlx':importlib.metadata.version('mlx'),"
        "'mlx_metal':importlib.metadata.version('mlx-metal')},sort_keys=True))"
    )
    return json.loads(
        subprocess.check_output([str(python.absolute()), "-c", program], text=True)
    )


def _assert_inputs(args: argparse.Namespace) -> tuple[str, str]:
    source_commit = _git_commit(args.source_root)
    harness_commit = _git_commit(ROOT)
    if source_commit != PR335_SOURCE_COMMIT:
        raise RuntimeError("DFlash2 source is not the exact PR335 comparator commit")
    if _git_status(args.source_root) or _git_status(ROOT):
        raise RuntimeError("DFlash2 campaign requires clean source trees")
    versions = _interpreter_versions(args.python)
    if versions != {"mlx": REQUIRED_MLX_VERSION, "mlx_metal": REQUIRED_MLX_VERSION}:
        raise RuntimeError(f"DFlash2 interpreter version mismatch: {versions}")
    expected_prompt = (
        native_matrix.VANITY_PROMPT_FILE
        if args.workload == "vanity"
        else native_matrix.PYTHON_PROMPT_FILE
    )
    if args.prompt_file.resolve() != expected_prompt.resolve():
        raise RuntimeError("DFlash2 campaign prompt is not the frozen artifact")
    expected_hashes = {
        args.prompt_file: PROMPT_ARTIFACT_SHA256[
            "vanity" if args.workload == "vanity" else "python"
        ],
        args.context_file: PYTHON_CONTEXT_SHA256,
    }
    for path, expected in expected_hashes.items():
        if _sha256(path) != expected:
            raise RuntimeError(f"DFlash2 frozen artifact hash mismatch: {path}")
    return source_commit, harness_commit


def run(args: argparse.Namespace) -> int:
    source_commit, harness_commit = _assert_inputs(args)
    contexts = (100,) if args.workload == "vanity" else tuple(args.contexts)
    scenarios = [scenario(args.workload, context) for context in contexts]
    isolated = native_matrix._load_isolated()
    args.output_root.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    with isolated._gpu_lock_scope(args.lock) as lock_scope:
        native_matrix._validated_parent_guard_scope(lock_scope)
        for current in scenarios:
            order = order_for_context(current.workload, current.context_tokens)
            context_root = args.output_root / f"{current.workload}-{current.context_tokens}"
            context_root.mkdir(parents=True, exist_ok=True)
            receipts: list[dict[str, Any]] = []
            for index, _lane in enumerate(order):
                output = context_root / f"arm-{index}-dflash2.json"
                command = child_command(
                    python=args.python,
                    source_root=args.source_root,
                    source_commit=source_commit,
                    model=args.model,
                    draft=args.draft,
                    prompt_file=args.prompt_file,
                    context_file=args.context_file,
                    lock=args.lock,
                    output=output,
                    scenario=current,
                )
                environment = dict(os.environ)
                environment["PYTHONPATH"] = str(args.source_root.resolve())
                environment["MLX_MAX_MB_PER_BUFFER"] = "512"
                environment["MLX_MAX_OPS_PER_BUFFER"] = "50"
                result = isolated._run_attested_child(
                    command,
                    environment=environment,
                    lock_path=args.lock,
                    owns_process_group=True,
                )
                output.with_suffix(".log").write_text(
                    result.stdout or "", encoding="utf-8"
                )
                if result.returncode != 0 or not output.is_file():
                    raise RuntimeError(
                        f"DFlash2 {current.workload} {current.context_tokens} "
                        f"arm {index} failed"
                    )
                receipt = json.loads(output.read_text(encoding="utf-8"))
                receipts.append(receipt)
                print(json.dumps({
                    "event": "dflash_arm_complete",
                    "workload": current.workload,
                    "context_tokens": current.context_tokens,
                    "arm": index + 1,
                    "wall_s": receipt["arm"]["wall_s"],
                }), flush=True)
            combined = aggregate(
                scenario=current,
                order=order,
                receipts=receipts,
                expected_harness_commit=harness_commit,
                lock=args.lock,
                model=args.model,
                draft=args.draft,
            )
            combined_path = context_root / "combined.json"
            combined_path.write_text(
                json.dumps(combined, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            if combined["invariant_errors"]:
                raise RuntimeError(
                    f"DFlash2 invariant errors: {combined['invariant_errors']}"
                )
            completed.append({
                "workload": current.workload,
                "context_tokens": current.context_tokens,
                "receipt_sha256": hashlib.sha256(combined_path.read_bytes()).hexdigest(),
                "summary": combined["summary"],
            })
            (args.output_root / "index.json").write_text(
                json.dumps({
                    "kind": "qwen38_dflash2_final_campaign",
                    "completed": completed,
                }, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("vanity", "low", "xhigh"), required=True)
    parser.add_argument("--contexts", nargs="+", type=int, default=list(LOW_CONTEXTS))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--context-file", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    return run(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
