#!/usr/bin/env python3
"""Run the pinned Qwen3.8 Quality DFlash2 cold-prefix matrix."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path


QUALITY_MODEL = "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality"
QUALITY_REVISION = "09f71b39a75c416be3c974840b53f9fbe9aa1841"
DRAFT_MODEL = "z-lab/Qwen3.8-27B-DFlash2"
PHYSICAL_WIDTHS = tuple(range(2, 9))  # DFlash K=1..7 means physical M=2..8.
COLD_PREFIX_TOKENS = (1024, 16384, 65536)
TEST_PROMPT_TOKENS = 1024
OUTPUT_TOKENS = 1024
NATURAL_PYTHON_PROMPT = (
    "Implement a Python function `merge_intervals(intervals)` that accepts an "
    "iterable of integer `(start, end)` pairs. Validate malformed or reversed "
    "intervals, do not mutate the input, merge overlapping and directly adjacent "
    "ranges, and return a sorted list of tuples. Include type hints, a concise "
    "docstring, time and space complexity, and five pytest tests covering empty "
    "input, duplicates, negative values, nesting, and invalid ranges. Return the "
    "implementation first, followed by an explanation."
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, choices=(3,), default=3)
    return parser.parse_args(argv)


def _ids_hash(ids) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def _natural_prompt(tokenizer):
    from mtplx.benchmarks.dflash2_contract import ExactPrompt

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": NATURAL_PYTHON_PROMPT}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = rendered["input_ids"] if isinstance(rendered, Mapping) else rendered
    token_ids = tuple(int(token) for token in encoded)
    if len(token_ids) != 105:
        raise ValueError(
            f"natural Python test input encoded to {len(token_ids)} tokens, expected 105"
        )
    return ExactPrompt(
        token_ids=token_ids,
        token_count=len(token_ids),
        token_sha256=_ids_hash(token_ids),
        enable_thinking=False,
    )


def _prompt_receipt(prompt, *, kind: str) -> dict[str, object]:
    if kind == "natural":
        return {
            "kind": "natural_test_input",
            "cold_prefix_tokens": 0,
            "test_prompt_tokens": prompt.token_count,
            "total_prompt_tokens": prompt.token_count,
            "test_prompt_sha256": prompt.token_sha256,
            "total_prompt_sha256": prompt.token_sha256,
            "enable_thinking": prompt.enable_thinking,
        }
    return {
        "kind": "cold_coding_prefix_plus_fixed_test_input",
        "cold_prefix_tokens": prompt.cold_prefix_tokens,
        "test_prompt_tokens": prompt.test_prompt_tokens,
        "total_prompt_tokens": prompt.total_prompt_tokens,
        "cold_prefix_sha256": prompt.cold_prefix_sha256,
        "test_prompt_sha256": prompt.test_prompt_sha256,
        "total_prompt_sha256": prompt.token_sha256,
        "enable_thinking": prompt.enable_thinking,
    }


def main() -> int:
    args = parse_args()

    from deepseek_v4_guard_window import issue_guard_window

    guard_path, guard_sha256 = issue_guard_window(
        expected_lock=Path("/tmp/mtplx-gpu-exclusive.lock")
    )

    from mtplx.benchmarks.dflash2_contract import (
        build_cold_prefill_python_prompt,
    )
    from mtplx.benchmarks.runners import dflash2_depth_sweep as sweep

    resolved_model = str(sweep._resolve_model_path(QUALITY_MODEL))
    if Path(resolved_model).name not in sweep.QWEN38_OPTIMIZED_QUALITY_DIRNAMES:
        raise ValueError("Quality model did not resolve to its verified local artifact")
    inspection = sweep._inspect_model(resolved_model).to_dict()
    runtime_contract = sweep._validated_runtime_contract(
        inspection,
        model_id=QUALITY_MODEL,
    )
    pinned_draft = sweep._resolve_draft_snapshot(
        DRAFT_MODEL,
        sweep.QWEN38_DFLASH2_REVISION,
    )
    runtime_overrides = sweep._runtime_env_overrides_from_contract(runtime_contract)
    sweep._apply_profile_env("turbo", runtime_env_overrides=runtime_overrides)
    overridden = sweep._profile_env_overridden()
    if overridden:
        names = ", ".join(str(row.get("var")) for row in overridden)
        raise ValueError("operator environment overrides turbo control: " + names)

    bundle = sweep._load_mtplx_dflash2_bundle(resolved_model, pinned_draft)
    draft_quant = sweep._validated_draft_meta(bundle, pinned_draft)
    draft_head_report = sweep._install_draft_lm_head(
        bundle.runtime,
        bits=4,
        group_size=64,
        mode="affine",
    )

    natural = _natural_prompt(bundle.tokenizer)
    cold_prompts = tuple(
        build_cold_prefill_python_prompt(
            bundle.tokenizer,
            cold_prefix_tokens=prefix_tokens,
            test_prompt_tokens=TEST_PROMPT_TOKENS,
        )
        for prefix_tokens in COLD_PREFIX_TOKENS
    )
    test_prompt_hashes = {prompt.test_prompt_sha256 for prompt in cold_prompts}
    if len(test_prompt_hashes) != 1:
        raise RuntimeError("cold-prefix rows do not share one exact 1024-token test input")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    guard_receipt = {"path": str(guard_path), "sha256": guard_sha256}
    model_receipt = {
        "requested": QUALITY_MODEL,
        "hf_revision": QUALITY_REVISION,
        "resolved": resolved_model,
        "draft": {
            "requested": DRAFT_MODEL,
            "revision": sweep.QWEN38_DFLASH2_REVISION,
            "resolved": pinned_draft,
            "quant": draft_quant,
        },
        "profile": "turbo",
        "mtp_depth": sweep.MTP_DEPTH,
        "draft_lm_head": {"bits": 4, "group_size": 64, "mode": "affine"},
        "draft_lm_head_install_report": draft_head_report,
    }
    workloads = (("natural-105", natural, "natural"),) + tuple(
        (f"cold-prefix-{prompt.cold_prefix_tokens}", prompt, "cold")
        for prompt in cold_prompts
    )
    manifest: dict[str, object] = {
        "model": model_receipt,
        "guard_window": guard_receipt,
        "sampling": {
            "mode": "greedy",
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 1,
            "seed": 0,
            "thinking": False,
        },
        "width_semantics": {
            "requested_K": list(range(1, 8)),
            "physical_M": list(PHYSICAL_WIDTHS),
        },
        "output_tokens": OUTPUT_TOKENS,
        "repetitions": args.repetitions,
        "runs": [],
    }

    for label, prompt, kind in workloads:
        prompt_receipt = _prompt_receipt(prompt, kind=kind)
        receipt = sweep.run_dflash2_depth_sweep(
            bundle=bundle,
            prompt_ids=prompt.token_ids,
            widths=PHYSICAL_WIDTHS,
            repetitions=args.repetitions,
            max_tokens=OUTPUT_TOKENS,
        )
        receipt["model"] = model_receipt
        receipt["guard_window"] = guard_receipt
        receipt["prompt"] = prompt_receipt
        receipt["width_semantics"] = manifest["width_semantics"]
        output = args.output_dir / f"qwen38-quality-dflash2-{label}-k1-7.json"
        sweep.write_depth_sweep_result(output, receipt)
        manifest["runs"].append(
            {
                "label": label,
                "output": str(output),
                "prompt": prompt_receipt,
                "selection": receipt["selection"],
                "determinism": receipt["determinism"],
            }
        )
        sweep.write_depth_sweep_result(
            args.output_dir / "qwen38-quality-dflash2-cold-matrix-manifest.json",
            manifest,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
