"""Sequential policy grid for native MTP fixed-depth runs."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mtplx.benchmarks.schema import encode_prompt_case, load_prompt_suite
from mtplx.benchmarks.validators.basic import (
    validate_json_text,
    validate_no_degenerate_loop,
)
from mtplx.correctors import load_runtime_corrector
from mtplx.generation import generate_ar, generate_mtpk
from mtplx.runtime import load
from mtplx.sampling import SamplerConfig


def _parse_float_grid(values: str | list[float | None]) -> list[float | None]:
    if not isinstance(values, str):
        return list(values)
    parsed: list[float | None] = []
    for raw in values.split(","):
        item = raw.strip()
        if not item:
            continue
        if item.lower() in {"none", "off", "null"}:
            parsed.append(None)
        else:
            parsed.append(float(item))
    if not parsed:
        raise ValueError("float grid must contain at least one value")
    return parsed


def _parse_int_grid(values: str | list[int]) -> list[int]:
    if not isinstance(values, str):
        parsed = [int(value) for value in values]
    else:
        parsed = [int(item.strip()) for item in values.split(",") if item.strip()]
    if not parsed:
        raise ValueError("integer grid must contain at least one value")
    return parsed


def _mean_present(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.mean(present) if present else None


def _rate_by_depth(accepted: list[int], drafted: list[int]) -> list[float | None]:
    return [(a / d if d else None) for a, d in zip(accepted, drafted)]


def _sum_lists(values: list[list[int]], length: int) -> list[int]:
    totals = [0 for _ in range(length)]
    for value in values:
        for index, item in enumerate(value[:length]):
            totals[index] += int(item)
    return totals


def _cycle_count(events: list[dict], verify_calls: int) -> int:
    return len(events) or max(0, int(verify_calls))


def run_mtp_depth_policy_grid(
    model_path: Path | str,
    prompt_suite: Path | str,
    *,
    depth: int = 5,
    thresholds: str | list[float | None] = "0.5,0.75,1.0,1.25,1.5,2.0",
    min_depths: str | list[int] = "0,1,2,3",
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    draft_temperature: float | None = 0.0,
    draft_top_p: float | None = None,
    draft_top_k: int | None = None,
    max_tokens: int = 96,
    seed: int = 0,
    limit: int | None = None,
    enable_thinking: bool | None = None,
    compare_ar: bool = False,
    mtp_hidden_variant: str = "pre_norm",
    mtp_cache_policy: str = "fresh",
    mtp_history_policy: str = "cycle",
    verify_strategy: str = "batched",
    mtp_corrector_path: Path | str | None = None,
    mtp_corrector_blend: float | None = None,
    store_events: bool = False,
) -> dict[str, Any]:
    if depth < 1:
        raise ValueError("depth must be >= 1")

    threshold_values = _parse_float_grid(thresholds)
    min_depth_values = _parse_int_grid(min_depths)
    for min_depth in min_depth_values:
        if min_depth < 0 or min_depth > depth:
            raise ValueError("min_depth values must be in [0, depth]")

    rt = load(model_path, mtp=True)
    prompt_cases = load_prompt_suite(prompt_suite)
    if limit is not None:
        prompt_cases = prompt_cases[:limit]

    sampler = SamplerConfig(temperature=temperature, top_p=top_p, top_k=top_k)
    draft_sampler = SamplerConfig(
        temperature=temperature if draft_temperature is None else draft_temperature,
        top_p=top_p if draft_top_p is None else draft_top_p,
        top_k=top_k if draft_top_k is None else draft_top_k,
    )
    mtp_corrector = load_runtime_corrector(
        mtp_corrector_path,
        blend=mtp_corrector_blend,
    )

    encoded = [
        (
            case,
            encode_prompt_case(
                rt.tokenizer,
                case,
                chat_template=True,
                enable_thinking=enable_thinking,
            ),
        )
        for case in prompt_cases
    ]

    ar_rows: list[dict[str, Any]] = []
    if compare_ar:
        for index, (case, ids) in enumerate(encoded):
            ar = generate_ar(
                rt,
                ids,
                max_tokens=min(max_tokens, case.max_tokens),
                sampler=sampler,
                seed=seed + index,
            )
            ar_rows.append(
                {
                    "prompt_id": case.id,
                    "category": case.category,
                    "generated_tokens": ar.stats.generated_tokens,
                    "tok_s": ar.stats.tok_s,
                    "tokens": ar.tokens,
                    "text": ar.text,
                    "elapsed_s": ar.stats.elapsed_s,
                    "target_forward_time_s": ar.stats.target_forward_time_s,
                }
            )

    results: list[dict[str, Any]] = []
    for threshold in threshold_values:
        for min_depth in min_depth_values:
            rows: list[dict[str, Any]] = []
            for index, (case, ids) in enumerate(encoded):
                out = generate_mtpk(
                    rt,
                    ids,
                    max_tokens=min(max_tokens, case.max_tokens),
                    sampler=sampler,
                    speculative_depth=depth,
                    seed=seed + index,
                    stop_token_ids=None,
                    mtp_hidden_variant=mtp_hidden_variant,
                    mtp_cache_policy=mtp_cache_policy,
                    mtp_history_policy=mtp_history_policy,
                    draft_sampler=draft_sampler,
                    draft_margin_threshold=threshold,
                    min_speculative_depth=min_depth,
                    verify_strategy=verify_strategy,
                    mtp_corrector=mtp_corrector,
                )
                validations = [asdict(validate_no_degenerate_loop(out.text))]
                if case.category == "json_tool":
                    validations.append(asdict(validate_json_text(out.text.strip())))
                cycles = _cycle_count(out.stats.events, out.stats.verify_calls)
                ar_row = ar_rows[index] if compare_ar else None
                row: dict[str, Any] = {
                    "prompt_id": case.id,
                    "category": case.category,
                    "prompt_sha256": case.prompt_sha256,
                    "generated_tokens": out.stats.generated_tokens,
                    "elapsed_s": out.stats.elapsed_s,
                    "tok_s": out.stats.tok_s,
                    "ar_tok_s": ar_row["tok_s"] if ar_row is not None else None,
                    "speedup_vs_ar": (
                        out.stats.tok_s / ar_row["tok_s"]
                        if ar_row is not None and ar_row["tok_s"]
                        else None
                    ),
                    "cycles": cycles,
                    "accepted_drafts": out.stats.accepted_drafts,
                    "rejected_drafts": out.stats.rejected_drafts,
                    "drafted_tokens": out.stats.drafted_tokens,
                    "accepted_by_depth": out.stats.accepted_by_depth,
                    "drafted_by_depth": out.stats.drafted_by_depth,
                    "acceptance_by_depth": _rate_by_depth(
                        out.stats.accepted_by_depth,
                        out.stats.drafted_by_depth,
                    ),
                    "accepted_drafts_per_cycle": (
                        out.stats.accepted_drafts / cycles if cycles else None
                    ),
                    "drafted_tokens_per_cycle": (
                        out.stats.drafted_tokens / cycles if cycles else None
                    ),
                    "acceptance_rate": (
                        out.stats.accepted_drafts / out.stats.drafted_tokens
                        if out.stats.drafted_tokens
                        else None
                    ),
                    "verify_time_s": out.stats.verify_time_s,
                    "draft_time_s": out.stats.draft_time_s,
                    "target_forward_time_s": out.stats.target_forward_time_s,
                    "model_path_tok_s": (
                        out.stats.generated_tokens
                        / (out.stats.target_forward_time_s + out.stats.draft_time_s)
                        if out.stats.target_forward_time_s + out.stats.draft_time_s
                        else None
                    ),
                    "peak_memory_bytes": out.stats.peak_memory_bytes,
                    "graphbank": out.stats.graphbank,
                    "validations": validations,
                    "text": out.text,
                }
                if store_events:
                    row["events"] = out.stats.events
                rows.append(row)

            validations = [v for row in rows for v in row["validations"]]
            accepted_by_depth = _sum_lists(
                [row["accepted_by_depth"] for row in rows],
                depth,
            )
            drafted_by_depth = _sum_lists(
                [row["drafted_by_depth"] for row in rows],
                depth,
            )
            results.append(
                {
                    "depth": depth,
                    "draft_margin_threshold": threshold,
                    "min_speculative_depth": min_depth,
                    "rows": rows,
                    "summary": {
                        "prompts": len(rows),
                        "generated_tokens": sum(
                            row["generated_tokens"] for row in rows
                        ),
                        "mean_tok_s": statistics.mean([row["tok_s"] for row in rows])
                        if rows
                        else 0.0,
                        "mean_ar_tok_s": _mean_present(
                            [row["ar_tok_s"] for row in rows]
                        ),
                        "mean_speedup_vs_ar": _mean_present(
                            [row["speedup_vs_ar"] for row in rows]
                        ),
                        "mean_model_path_tok_s": _mean_present(
                            [row["model_path_tok_s"] for row in rows]
                        ),
                        "cycles": sum(row["cycles"] for row in rows),
                        "accepted_drafts": sum(row["accepted_drafts"] for row in rows),
                        "rejected_drafts": sum(row["rejected_drafts"] for row in rows),
                        "drafted_tokens": sum(row["drafted_tokens"] for row in rows),
                        "accepted_by_depth": accepted_by_depth,
                        "drafted_by_depth": drafted_by_depth,
                        "acceptance_by_depth": _rate_by_depth(
                            accepted_by_depth, drafted_by_depth
                        ),
                        "accepted_drafts_per_cycle": (
                            sum(row["accepted_drafts"] for row in rows)
                            / max(1, sum(row["cycles"] for row in rows))
                        ),
                        "drafted_tokens_per_cycle": (
                            sum(row["drafted_tokens"] for row in rows)
                            / max(1, sum(row["cycles"] for row in rows))
                        ),
                        "acceptance_rate": (
                            sum(row["accepted_drafts"] for row in rows)
                            / sum(row["drafted_tokens"] for row in rows)
                            if sum(row["drafted_tokens"] for row in rows)
                            else None
                        ),
                        "verify_time_s": sum(row["verify_time_s"] for row in rows),
                        "draft_time_s": sum(row["draft_time_s"] for row in rows),
                        "target_forward_time_s": sum(
                            row["target_forward_time_s"] for row in rows
                        ),
                        "validations_passed": sum(
                            1 for v in validations if v["passed"]
                        ),
                        "validations_total": len(validations),
                        "peak_memory_bytes": max(
                            [row["peak_memory_bytes"] for row in rows] or [0]
                        ),
                    },
                }
            )

    results.sort(
        key=lambda item: (
            item["summary"]["validations_passed"]
            == item["summary"]["validations_total"],
            item["summary"]["mean_tok_s"],
        ),
        reverse=True,
    )

    return {
        "model_path": str(model_path),
        "prompt_suite": str(prompt_suite),
        "sampler": asdict(sampler),
        "draft_sampler": asdict(draft_sampler),
        "max_tokens": max_tokens,
        "seed": seed,
        "enable_thinking": enable_thinking,
        "compare_ar": compare_ar,
        "mtp_hidden_variant": mtp_hidden_variant,
        "mtp_cache_policy": mtp_cache_policy,
        "mtp_history_policy": mtp_history_policy,
        "verify_strategy": verify_strategy,
        "mtp_corrector_path": str(mtp_corrector_path)
        if mtp_corrector_path is not None
        else None,
        "mtp_corrector_blend": mtp_corrector_blend,
        "mtp_corrector_kind": getattr(mtp_corrector, "kind", None)
        if mtp_corrector is not None
        else None,
        "thresholds": threshold_values,
        "min_depths": min_depth_values,
        "ar_rows": ar_rows,
        "grid": results,
        "best": results[0] if results else None,
    }


def write_depth_grid(path: Path | str, result: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
