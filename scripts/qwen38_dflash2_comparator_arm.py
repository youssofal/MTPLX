#!/usr/bin/env python3
"""Run PR335 DFlash2 with the current matrix's frozen prompt contract."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROMPT_ARM = ROOT / "scripts/qwen38_native_mtp_matrix_arm.py"
PR335_SOURCE_COMMIT = "9a6f48e69f9c8c6932d0f005c364844b2bf33e9c"
REQUIRED_MLX_VERSION = "0.32.2"
REQUIRED_DFLASH_FEATURES = (
    "adaptive_policy",
    "context_route",
    "dflash_gqa_widths",
    "dflash_m6_barrier_free_kp1",
    "dflash_m8_nax_island",
    "r21_qk_rms_rope",
    "r24_eval_ladder",
    "r24_qk_length_limit",
    "r26_prefill_ladder_3",
    "r48_boundary_fused",
    "r50_wired_residency",
    "r53_command_buffers",
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workload(args: Any) -> str:
    if args.prompt_kind == "is_palindrome":
        return "vanity"
    if args.reasoning_effort in {"low", "xhigh"}:
        return str(args.reasoning_effort)
    raise ValueError("coding comparator requires low or xhigh reasoning effort")


def _generate_or_skip(generate: Any, runtime: Any, prompt_ids: list[int], args: Any) -> Any:
    if int(args.max_tokens) == 0:
        return None
    return generate(runtime, prompt_ids, args)


def _dflash_feature_active(name: str, value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if name == "context_route":
        return all(
            value.get(key) is True
            for key in (
                "effective_adaptive",
                "row21_active",
                "row24_decode_active",
                "row24_prefill_active",
                "row48_decode_active",
                "row48_prefill_active",
                "row50_active",
            )
        )
    if name == "dflash_m8_nax_island":
        return value.get("active") is True and int(
            value.get("validated_projections", 0)
        ) > 0
    if name in {"r21_qk_rms_rope", "r24_qk_length_limit"}:
        return int(value.get("active_modules", 0)) > 0
    if name == "r24_eval_ladder":
        return all(int(value.get(key, 0)) > 0 for key in ("active", "decode_active", "prefill_active"))
    if name == "r48_boundary_fused":
        return (
            int(value.get("active_modules", 0)) > 0
            and int(value.get("decode_active", 0)) > 0
            and int(value.get("prefill_active", 0)) > 0
        )
    if name in {"r50_wired_residency", "r53_command_buffers"}:
        return value.get("active") is True and value.get("installed") is True
    return value.get("active") is True or int(value.get("active", 0)) > 0


def _optimized_stack_errors(receipt: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    exact = {
        "engine": "pr_dflash2",
        "source_commit": PR335_SOURCE_COMMIT,
        "mlx_version": REQUIRED_MLX_VERSION,
        "mlx_metal_version": REQUIRED_MLX_VERSION,
    }
    for key, expected in exact.items():
        if receipt.get(key) != expected:
            errors.append(f"DFlash2 {key} mismatch")
    stack = receipt.get("stack") or {}
    if stack.get("profile") != "turbo" or int(stack.get("dflash_block_size", 0)) != 8:
        errors.append("DFlash2 turbo stack is not installed")
    if stack.get("native_mtp_loaded") is not False:
        errors.append("DFlash2 comparator retained the native MTP head")
    features = stack.get("feature_receipt") or {}
    for name in REQUIRED_DFLASH_FEATURES:
        if not _dflash_feature_active(name, features.get(name)):
            errors.append(f"DFlash2 optimized feature {name} is inactive")
    if (receipt.get("arm") or {}).get("fallback_ar") is not False:
        errors.append("DFlash2 fell back to autoregressive decode")
    return errors


def main() -> int:
    prompt_arm = _load_module("qwen38_matrix_prompt_contract", PROMPT_ARM)
    source_root = Path(sys.argv[sys.argv.index("--source-root") + 1]).resolve(strict=True)
    expected_commit = sys.argv[sys.argv.index("--source-commit") + 1]
    observed_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--short"], cwd=source_root, text=True
    ).splitlines()
    if observed_commit != expected_commit or status:
        raise RuntimeError(
            f"DFlash2 source must be clean at {expected_commit}; "
            f"found commit={observed_commit} status={status}"
        )

    upstream = _load_module(
        "qwen38_pr335_final_benchmark_arm",
        source_root / "scripts/qwen38_final_benchmark_arm.py",
    )

    def load_frozen_prompt(args: Any, tokenizer: Any) -> tuple[str, list[int]]:
        _prompt_id, instruction = prompt_arm._read_prompt(args.prompt_file)
        return prompt_arm.build_prompt(
            tokenizer,
            workload=_workload(args),
            instruction=instruction,
            context=args.context_file.read_text(encoding="utf-8"),
            target_tokens=int(args.prompt_tokens),
        )

    original_generate = upstream._generate_dflash
    upstream._load_prompt = load_frozen_prompt
    upstream._sha256_tokens = prompt_arm._token_hash
    upstream._generate_dflash = (
        lambda runtime, prompt_ids, args: _generate_or_skip(
            original_generate, runtime, prompt_ids, args
        )
    )
    exit_code = int(upstream.main())

    output = Path(sys.argv[sys.argv.index("--output") + 1])
    receipt = json.loads(output.read_text(encoding="utf-8"))
    workload = _workload(upstream._parse_args())
    receipt["kind"] = "qwen38_dflash2_frozen_matrix_arm"
    receipt["harness_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    receipt["workload"]["workload"] = workload
    receipt["workload"]["enable_thinking"] = workload in {"low", "xhigh"}
    receipt["workload"]["prompt_format"] = {
        "vanity": "qwen_chat_template_non_thinking",
        "low": "qwen_chat_template_thinking_low",
        "xhigh": "qwen_chat_template_thinking_xhigh",
    }[workload]
    receipt["workload"]["prompt_artifact_sha256"] = prompt_arm._sha256(
        Path(receipt["workload"].get("prompt_file", ""))
    ) if receipt["workload"].get("prompt_file") else prompt_arm._sha256(
        Path(sys.argv[sys.argv.index("--prompt-file") + 1])
    )
    receipt["workload"]["context_artifact_sha256"] = prompt_arm._sha256(
        Path(sys.argv[sys.argv.index("--context-file") + 1])
    )
    receipt["mlx_metal_version"] = importlib.metadata.version("mlx-metal")
    stack_errors = _optimized_stack_errors(receipt)
    if stack_errors:
        raise RuntimeError(f"DFlash2 optimized receipt invariant errors: {stack_errors}")
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
