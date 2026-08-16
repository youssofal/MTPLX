"""Fixed-depth native MTP sweep runner."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mtplx.benchmarks.schema import encode_prompt_case, load_prompt_suite
from mtplx.benchmarks.validators.basic import validate_benchmark_output
from mtplx.correctors import load_runtime_corrector
from mtplx.generation import generate_ar, generate_mtpk
from mtplx.mtp_patch import MTPContract
from mtplx.proposal_reranker import TopKProposalReranker
from mtplx.runtime import load
from mtplx.sampling import SamplerConfig


def _parse_depths(depths: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(depths, str):
        out = [int(part.strip()) for part in depths.split(",") if part.strip()]
    else:
        out = [int(value) for value in depths]
    if not out or any(value < 1 for value in out):
        raise ValueError("depths must contain one or more positive integers")
    return out


def _rate_by_depth(accepted: list[int], drafted: list[int]) -> list[float | None]:
    return [(a / d if d else None) for a, d in zip(accepted, drafted)]


def _cycle_count(events: list[dict], verify_calls: int) -> int:
    return len(events) or max(0, int(verify_calls))


def _active_memory_bytes() -> int:
    import mlx.core as mx

    return int(mx.get_active_memory())


def _token_budget(max_tokens: int, case_max_tokens: int) -> int:
    return min(int(max_tokens), int(case_max_tokens))


def _hit_token_budget(
    generated_tokens: int, token_budget: int, finish_reason: str | None
) -> bool:
    if finish_reason == "length":
        return True
    return int(generated_tokens) >= int(token_budget)


def _finish_reason_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = row.get("finish_reason")
        if not isinstance(reason, str) or not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def run_mtp_depth_sweep(
    model_path: Path | str,
    prompt_suite: Path | str,
    *,
    depths: str | list[int] | tuple[int, ...] = (1, 2, 3),
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    draft_temperature: float | None = None,
    draft_top_p: float | None = None,
    draft_top_k: int | None = None,
    max_tokens: int = 96,
    seed: int = 0,
    limit: int | None = None,
    enable_thinking: bool | None = None,
    compare_ar: bool = False,
    ar_only: bool = False,
    gemma4_draft_block_size: int | None = None,
    base_hidden_variant: str | None = None,
    mtp_hidden_variant: str | None = None,
    concat_order: str | None = None,
    mtp_cache_policy: str = "persistent",
    mtp_history_policy: str = "cycle",
    draft_margin_threshold: float | None = None,
    min_speculative_depth: int = 1,
    verify_strategy: str = "batched",
    verify_core: str = "stock",
    draft_core: str = "stock",
    mtp_quant_bits: int | None = None,
    mtp_quant_group_size: int = 64,
    mtp_quant_mode: str = "affine",
    mtp_adapter_path: Path | str | None = None,
    merge_mtp_adapter: bool = False,
    mtp_corrector_path: Path | str | None = None,
    mtp_corrector_blend: float | None = None,
    online_hidden_corrector_alpha: float = 0.0,
    online_hidden_corrector_decay: float = 0.8,
    online_hidden_corrector_warmup: int = 1,
    online_hidden_corrector_max_feed_depth: int | None = None,
    online_hidden_corrector_key: str = "global",
    online_correction_cache: bool = False,
    online_correction_cache_min_depth: int = 1,
    online_correction_cache_key: str = "local_prefix",
    prompt_correction_cache: bool = False,
    prompt_correction_cache_min_depth: int = 2,
    adapter_ensemble_q: bool = False,
    adapter_ensemble_epsilon: float = 0.5,
    adapter_ensemble_min_depth: int = 2,
    mtp_topk_reranker_calib: Path | str | None = None,
    mtp_topk_reranker_depths: str | list[int] | tuple[int, ...] = (4,),
    mtp_topk_reranker_topk: int = 32,
    mtp_topk_reranker_q_weight: float = 0.5,
    mtp_topk_reranker_token_weight: float = 1.0,
    mtp_topk_reranker_rank_weight: float = 0.0,
    mtp_topk_reranker_prefix_active_only: bool = True,
    draft_lm_head_bits: int | None = None,
    draft_lm_head_group_size: int = 64,
    draft_lm_head_mode: str = "affine",
    deepseek_v4_0731_k2: bool = False,
) -> dict[str, Any]:
    contract_kwargs: dict[str, Any] = {
        "mtp_quant_bits": mtp_quant_bits,
        "mtp_quant_group_size": mtp_quant_group_size,
        "mtp_quant_mode": mtp_quant_mode,
    }
    if base_hidden_variant not in {None, "auto", "contract"}:
        contract_kwargs["base_hidden_variant"] = str(base_hidden_variant)
    if mtp_hidden_variant not in {None, "auto", "contract"}:
        contract_kwargs["hidden_variant"] = str(mtp_hidden_variant)
    if concat_order not in {None, "auto", "contract"}:
        contract_kwargs["concat_order"] = str(concat_order)
    rt = load(
        model_path,
        mtp=True,
        contract=MTPContract(**contract_kwargs),
        mtp_adapter=mtp_adapter_path,
        merge_mtp_adapter=merge_mtp_adapter,
        gemma4_draft_block_size=gemma4_draft_block_size,
        deepseek_v4_0731_k2=deepseek_v4_0731_k2,
    )
    load_active_memory_bytes = _active_memory_bytes()
    contract = getattr(rt, "contract", None)
    is_gemma4_assistant = getattr(rt, "backend_id", None) == "gemma4_assistant"
    is_native_block_backend = getattr(rt, "block_speculative_backend", None) is not None
    resolved_base_hidden_variant = str(
        getattr(contract, "base_hidden_variant", "gemma4_assistant")
    )
    resolved_mtp_hidden_variant = str(
        getattr(contract, "hidden_variant", "gemma4_assistant")
    )
    resolved_concat_order = str(getattr(contract, "concat_order", "assistant_pair"))
    draft_lm_head_report: dict[str, Any] | None = None
    if draft_lm_head_bits is not None and not is_gemma4_assistant:
        from mtplx.draft_lm_head import _install_draft_lm_head

        draft_lm_head_report = _install_draft_lm_head(
            rt,
            bits=int(draft_lm_head_bits),
            group_size=int(draft_lm_head_group_size),
            mode=str(draft_lm_head_mode),
        )
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
    if ar_only and not compare_ar:
        raise ValueError("ar_only requires compare_ar=True")
    depth_values = [] if ar_only else _parse_depths(depths)
    mtp_topk_reranker = (
        TopKProposalReranker.from_calibration(
            mtp_topk_reranker_calib,
            depths=set(_parse_depths(mtp_topk_reranker_depths)),
            topk=mtp_topk_reranker_topk,
            q_weight=mtp_topk_reranker_q_weight,
            token_weight=mtp_topk_reranker_token_weight,
            rank_weight=mtp_topk_reranker_rank_weight,
            prefix_active_only=mtp_topk_reranker_prefix_active_only,
        )
        if mtp_topk_reranker_calib is not None
        else None
    )

    encoded = []
    for case in prompt_cases:
        encoded.append(
            (
                case,
                encode_prompt_case(
                    rt.tokenizer,
                    case,
                    chat_template=True,
                    enable_thinking=enable_thinking,
                ),
            )
        )

    ar_rows: list[dict[str, Any]] = []
    if compare_ar:
        for index, (case, ids) in enumerate(encoded):
            token_budget = _token_budget(max_tokens, case.max_tokens)
            generation_started_at = time.time()
            ar = generate_ar(
                rt,
                ids,
                max_tokens=token_budget,
                sampler=sampler,
                seed=seed + index,
            )
            generation_ended_at = time.time()
            active_memory_bytes = _active_memory_bytes()
            validations = [
                asdict(validation)
                for validation in validate_benchmark_output(
                    ar.text,
                    category=case.category,
                    prompt_id=case.id,
                )
            ]
            ar_rows.append(
                {
                    "prompt_id": case.id,
                    "category": case.category,
                    "generation_started_at": generation_started_at,
                    "generation_ended_at": generation_ended_at,
                    "generation_window_s": generation_ended_at - generation_started_at,
                    "token_budget": token_budget,
                    "finish_reason": ar.finish_reason,
                    "hit_token_budget": _hit_token_budget(
                        ar.stats.generated_tokens,
                        token_budget,
                        ar.finish_reason,
                    ),
                    "generated_tokens": ar.stats.generated_tokens,
                    "prompt_tokens": len(ids),
                    "elapsed_s": ar.stats.elapsed_s,
                    "tok_s": ar.stats.tok_s,
                    "decode_tok_s": ar.stats.decode_tok_s,
                    "decode_elapsed_s": ar.stats.decode_elapsed_s,
                    "end_to_end_tok_s": ar.stats.end_to_end_tok_s,
                    "prompt_eval_time_s": ar.stats.prompt_eval_time_s,
                    "prompt_tps": ar.stats.prompt_tps,
                    "prompt_target_prefill_tok_s": (
                        ar.stats.prompt_target_prefill_tok_s
                    ),
                    "active_memory_bytes": active_memory_bytes,
                    "active_memory_growth_bytes": (
                        active_memory_bytes - load_active_memory_bytes
                    ),
                    "peak_memory_bytes": ar.stats.peak_memory_bytes,
                    "tokens": ar.tokens,
                    "text": ar.text,
                    "validations": validations,
                }
            )

    depth_results = []
    for depth in depth_values:
        rows = []
        for index, (case, ids) in enumerate(encoded):
            token_budget = _token_budget(max_tokens, case.max_tokens)
            generation_started_at = time.time()
            out = generate_mtpk(
                rt,
                ids,
                max_tokens=token_budget,
                sampler=sampler,
                speculative_depth=depth,
                seed=seed + index,
                base_hidden_variant=(
                    None if is_native_block_backend else resolved_base_hidden_variant
                ),
                mtp_hidden_variant=(
                    None if is_native_block_backend else resolved_mtp_hidden_variant
                ),
                mtp_cache_policy=mtp_cache_policy,
                mtp_history_policy=mtp_history_policy,
                draft_sampler=draft_sampler,
                draft_margin_threshold=draft_margin_threshold,
                min_speculative_depth=min_speculative_depth,
                verify_strategy=verify_strategy,
                verify_core=verify_core,
                draft_core=draft_core,
                mtp_corrector=mtp_corrector,
                online_hidden_corrector_alpha=online_hidden_corrector_alpha,
                online_hidden_corrector_decay=online_hidden_corrector_decay,
                online_hidden_corrector_warmup=online_hidden_corrector_warmup,
                online_hidden_corrector_max_feed_depth=online_hidden_corrector_max_feed_depth,
                online_hidden_corrector_key=online_hidden_corrector_key,
                online_correction_cache=online_correction_cache,
                online_correction_cache_min_depth=online_correction_cache_min_depth,
                online_correction_cache_key=online_correction_cache_key,
                prompt_correction_cache=prompt_correction_cache,
                prompt_correction_cache_min_depth=prompt_correction_cache_min_depth,
                adapter_ensemble_q=adapter_ensemble_q,
                adapter_ensemble_epsilon=adapter_ensemble_epsilon,
                adapter_ensemble_min_depth=adapter_ensemble_min_depth,
                mtp_topk_reranker=mtp_topk_reranker,
            )
            generation_ended_at = time.time()
            active_memory_bytes = _active_memory_bytes()
            validations = [
                asdict(validation)
                for validation in validate_benchmark_output(
                    out.text,
                    category=case.category,
                    prompt_id=case.id,
                )
            ]
            ar_row = ar_rows[index] if compare_ar else None
            cycles = _cycle_count(out.stats.events, out.stats.verify_calls)
            rows.append(
                {
                    "prompt_id": case.id,
                    "category": case.category,
                    "prompt_sha256": case.prompt_sha256,
                    "generation_started_at": generation_started_at,
                    "generation_ended_at": generation_ended_at,
                    "generation_window_s": generation_ended_at - generation_started_at,
                    "token_budget": token_budget,
                    "finish_reason": out.finish_reason,
                    "hit_token_budget": _hit_token_budget(
                        out.stats.generated_tokens,
                        token_budget,
                        out.finish_reason,
                    ),
                    "generated_tokens": out.stats.generated_tokens,
                    "prompt_tokens": len(ids),
                    "elapsed_s": out.stats.elapsed_s,
                    "tok_s": out.stats.tok_s,
                    "decode_tok_s": out.stats.decode_tok_s,
                    "decode_elapsed_s": out.stats.decode_elapsed_s,
                    "end_to_end_tok_s": out.stats.end_to_end_tok_s,
                    "prompt_eval_time_s": out.stats.prompt_eval_time_s,
                    "prompt_tps": out.stats.prompt_tps,
                    "prompt_target_prefill_tok_s": (
                        out.stats.prompt_target_prefill_tok_s
                    ),
                    "active_memory_bytes": active_memory_bytes,
                    "active_memory_growth_bytes": (
                        active_memory_bytes - load_active_memory_bytes
                    ),
                    "ar_tok_s": ar_row["tok_s"] if ar_row is not None else None,
                    "ar_decode_tok_s": ar_row["decode_tok_s"]
                    if ar_row is not None
                    else None,
                    "ar_end_to_end_tok_s": ar_row["end_to_end_tok_s"]
                    if ar_row is not None
                    else None,
                    "exact_match": (
                        out.tokens == ar_row["tokens"]
                        if ar_row is not None and temperature <= 0
                        else None
                    ),
                    "tokens": out.tokens,
                    "speedup_vs_ar": (
                        out.stats.tok_s / ar_row["tok_s"]
                        if ar_row is not None and ar_row["tok_s"]
                        else None
                    ),
                    "end_to_end_speedup_vs_ar": (
                        out.stats.end_to_end_tok_s / ar_row["end_to_end_tok_s"]
                        if ar_row is not None and ar_row.get("end_to_end_tok_s")
                        else None
                    ),
                    "accepted_drafts": out.stats.accepted_drafts,
                    "rejected_drafts": out.stats.rejected_drafts,
                    "drafted_tokens": out.stats.drafted_tokens,
                    "accepted_by_depth": out.stats.accepted_by_depth,
                    "drafted_by_depth": out.stats.drafted_by_depth,
                    "accept_probability_sum_by_depth": out.stats.accept_probability_sum_by_depth,
                    "mean_accept_probability_by_depth": out.stats.mean_accept_probability_by_depth,
                    "acceptance_by_depth": _rate_by_depth(
                        out.stats.accepted_by_depth,
                        out.stats.drafted_by_depth,
                    ),
                    "mean_accepted_drafts_per_cycle": (
                        out.stats.accepted_drafts / max(1, cycles)
                    ),
                    "acceptance_rate": (
                        out.stats.accepted_drafts / out.stats.drafted_tokens
                        if out.stats.drafted_tokens
                        else None
                    ),
                    "verify_time_s": out.stats.verify_time_s,
                    "verify_forward_time_s": out.stats.verify_forward_time_s,
                    "verify_eval_time_s": out.stats.verify_eval_time_s,
                    "verify_hidden_eval_time_s": out.stats.verify_hidden_eval_time_s,
                    "verify_joint_eval_time_s": out.stats.verify_joint_eval_time_s,
                    "verify_target_distribution_time_s": out.stats.verify_target_distribution_time_s,
                    "target_distribution_materialized_rows": out.stats.target_distribution_materialized_rows,
                    "target_distribution_materialized_windows": out.stats.target_distribution_materialized_windows,
                    "target_distribution_share": out.stats.target_distribution_share,
                    "lazy_bonus_verify_calls": out.stats.lazy_bonus_verify_calls,
                    "lazy_bonus_commit_time_s": out.stats.lazy_bonus_commit_time_s,
                    "draft_time_s": out.stats.draft_time_s,
                    "target_forward_time_s": out.stats.target_forward_time_s,
                    "snapshot_time_s": out.stats.snapshot_time_s,
                    "accept_time_s": out.stats.accept_time_s,
                    "rollback_time_s": out.stats.rollback_time_s,
                    "repair_time_s": out.stats.repair_time_s,
                    "commit_time_s": out.stats.commit_time_s,
                    "capture_commit_time_s": out.stats.capture_commit_time_s,
                    "bonus_time_s": out.stats.bonus_time_s,
                    "online_hidden_corrector_time_s": out.stats.online_hidden_corrector_time_s,
                    "online_correction_cache": out.stats.online_correction_cache,
                    "adapter_ensemble_q": out.stats.adapter_ensemble_q,
                    "mtp_topk_reranker": out.stats.mtp_topk_reranker,
                    "draft_core": out.stats.draft_core,
                    "bonus_tokens": out.stats.bonus_tokens,
                    "correction_tokens": out.stats.correction_tokens,
                    "deferred_correction_repairs": out.stats.deferred_correction_repairs,
                    "verify_calls": out.stats.verify_calls,
                    "reject_path_counts": out.stats.reject_path_counts,
                    "repair_time_by_reject_depth_s": out.stats.repair_time_by_reject_depth_s,
                    "graphbank": out.stats.graphbank,
                    "peak_memory_bytes": out.stats.peak_memory_bytes,
                    "validations": validations,
                    "text": out.text,
                    "events": out.stats.events,
                }
            )

        validations = [v for row in rows for v in row["validations"]]
        finish_reasons = _finish_reason_counts(rows)
        accepted_by_depth = _sum_lists(
            [row["accepted_by_depth"] for row in rows], depth
        )
        drafted_by_depth = _sum_lists([row["drafted_by_depth"] for row in rows], depth)
        accept_probability_sum_by_depth = _sum_float_lists(
            [row["accept_probability_sum_by_depth"] for row in rows],
            depth,
        )
        verify_time_s = sum(row["verify_time_s"] for row in rows)
        verify_target_distribution_time_s = sum(
            row["verify_target_distribution_time_s"] for row in rows
        )
        target_distribution_rows = sum(
            row["target_distribution_materialized_rows"] for row in rows
        )
        target_distribution_windows = sum(
            row["target_distribution_materialized_windows"] for row in rows
        )
        lazy_bonus_verify_calls = sum(row["lazy_bonus_verify_calls"] for row in rows)
        lazy_bonus_commit_time_s = sum(row["lazy_bonus_commit_time_s"] for row in rows)
        depth_results.append(
            {
                "depth": depth,
                "rows": rows,
                "summary": {
                    "prompts": len(rows),
                    "generated_tokens": sum(row["generated_tokens"] for row in rows),
                    "hit_token_budget_count": sum(
                        1 for row in rows if row.get("hit_token_budget")
                    ),
                    "finish_reasons": finish_reasons,
                    "elapsed_s": sum(row["elapsed_s"] for row in rows),
                    "accepted_drafts": sum(row["accepted_drafts"] for row in rows),
                    "rejected_drafts": sum(row["rejected_drafts"] for row in rows),
                    "drafted_tokens": sum(row["drafted_tokens"] for row in rows),
                    "accepted_by_depth": accepted_by_depth,
                    "drafted_by_depth": drafted_by_depth,
                    "accept_probability_sum_by_depth": accept_probability_sum_by_depth,
                    "mean_accept_probability_by_depth": _rate_by_depth(
                        accept_probability_sum_by_depth,
                        drafted_by_depth,
                    ),
                    "acceptance_by_depth": _rate_by_depth(
                        accepted_by_depth, drafted_by_depth
                    ),
                    "mean_tok_s": statistics.mean([row["tok_s"] for row in rows])
                    if rows
                    else 0.0,
                    "mean_decode_tok_s": statistics.mean(
                        [row["decode_tok_s"] for row in rows]
                    )
                    if rows
                    else 0.0,
                    "mean_end_to_end_tok_s": (
                        statistics.mean([row["end_to_end_tok_s"] for row in rows])
                        if rows
                        else 0.0
                    ),
                    "mean_ar_tok_s": (
                        statistics.mean(
                            [
                                row["ar_tok_s"]
                                for row in rows
                                if row["ar_tok_s"] is not None
                            ]
                        )
                        if compare_ar and rows
                        else None
                    ),
                    "mean_ar_decode_tok_s": (
                        statistics.mean(
                            [
                                row["ar_decode_tok_s"]
                                for row in rows
                                if row["ar_decode_tok_s"] is not None
                            ]
                        )
                        if compare_ar and rows
                        else None
                    ),
                    "mean_ar_end_to_end_tok_s": (
                        statistics.mean(
                            [
                                row["ar_end_to_end_tok_s"]
                                for row in rows
                                if row["ar_end_to_end_tok_s"] is not None
                            ]
                        )
                        if compare_ar and rows
                        else None
                    ),
                    "mean_speedup_vs_ar": (
                        statistics.mean(
                            [
                                row["speedup_vs_ar"]
                                for row in rows
                                if row["speedup_vs_ar"] is not None
                            ]
                        )
                        if compare_ar and rows
                        else None
                    ),
                    "verify_time_s": verify_time_s,
                    "verify_forward_time_s": sum(
                        row["verify_forward_time_s"] for row in rows
                    ),
                    "verify_eval_time_s": sum(
                        row["verify_eval_time_s"] for row in rows
                    ),
                    "verify_hidden_eval_time_s": sum(
                        row["verify_hidden_eval_time_s"] for row in rows
                    ),
                    "verify_joint_eval_time_s": sum(
                        row["verify_joint_eval_time_s"] for row in rows
                    ),
                    "verify_target_distribution_time_s": (
                        verify_target_distribution_time_s
                    ),
                    "target_distribution_materialized_rows": (target_distribution_rows),
                    "target_distribution_materialized_windows": (
                        target_distribution_windows
                    ),
                    "target_distribution_rows_per_window": (
                        target_distribution_rows / target_distribution_windows
                        if target_distribution_windows
                        else None
                    ),
                    "verify_target_distribution_ms_per_row": (
                        1000.0
                        * verify_target_distribution_time_s
                        / target_distribution_rows
                        if target_distribution_rows
                        else None
                    ),
                    "target_distribution_share": (
                        verify_target_distribution_time_s / verify_time_s
                        if verify_time_s
                        else 0.0
                    ),
                    "lazy_bonus_verify_calls": lazy_bonus_verify_calls,
                    "lazy_bonus_commit_time_s": lazy_bonus_commit_time_s,
                    "lazy_bonus_commit_ms_per_call": (
                        1000.0 * lazy_bonus_commit_time_s / lazy_bonus_verify_calls
                        if lazy_bonus_verify_calls
                        else None
                    ),
                    "draft_time_s": sum(row["draft_time_s"] for row in rows),
                    "target_forward_time_s": sum(
                        row["target_forward_time_s"] for row in rows
                    ),
                    "snapshot_time_s": sum(row["snapshot_time_s"] for row in rows),
                    "accept_time_s": sum(row["accept_time_s"] for row in rows),
                    "rollback_time_s": sum(row["rollback_time_s"] for row in rows),
                    "repair_time_s": sum(row["repair_time_s"] for row in rows),
                    "commit_time_s": sum(row["commit_time_s"] for row in rows),
                    "capture_commit_time_s": sum(
                        row["capture_commit_time_s"] for row in rows
                    ),
                    "bonus_time_s": sum(row["bonus_time_s"] for row in rows),
                    "online_hidden_corrector_time_s": sum(
                        row["online_hidden_corrector_time_s"] for row in rows
                    ),
                    "online_correction_cache": _sum_correction_cache(
                        [row["online_correction_cache"] for row in rows]
                    ),
                    "adapter_ensemble_q": _sum_adapter_ensemble_q(
                        [row["adapter_ensemble_q"] for row in rows]
                    ),
                    "mtp_topk_reranker": _sum_topk_reranker(
                        [row["mtp_topk_reranker"] for row in rows]
                    ),
                    "draft_core": _sum_draft_core([row["draft_core"] for row in rows]),
                    "bonus_tokens": sum(row["bonus_tokens"] for row in rows),
                    "correction_tokens": sum(row["correction_tokens"] for row in rows),
                    "deferred_correction_repairs": sum(
                        row["deferred_correction_repairs"] for row in rows
                    ),
                    "verify_calls": sum(row["verify_calls"] for row in rows),
                    "reject_path_counts": _sum_dicts(
                        [row["reject_path_counts"] for row in rows]
                    ),
                    "repair_time_by_reject_depth_s": _sum_dicts(
                        [row["repair_time_by_reject_depth_s"] for row in rows]
                    ),
                    "validations_passed": sum(1 for v in validations if v["passed"]),
                    "validations_total": len(validations),
                    "exact_matches": (
                        sum(1 for row in rows if row["exact_match"])
                        if compare_ar and temperature <= 0
                        else None
                    ),
                    "exact_total": (
                        len(rows) if compare_ar and temperature <= 0 else None
                    ),
                    "peak_memory_bytes": max(
                        [row["peak_memory_bytes"] for row in rows] or [0]
                    ),
                    "active_memory_bytes": max(
                        [row["active_memory_bytes"] for row in rows] or [0]
                    ),
                    "active_memory_growth_bytes": max(
                        [row["active_memory_growth_bytes"] for row in rows] or [0]
                    ),
                    "speed_model": _speed_model_summary(rows),
                },
            }
        )

    return {
        "model_path": str(model_path),
        "load_active_memory_bytes": load_active_memory_bytes,
        "deepseek_v4_0731_k2": deepseek_v4_0731_k2,
        "prompt_suite": str(prompt_suite),
        "sampler": asdict(sampler),
        "draft_sampler": asdict(draft_sampler),
        "max_tokens": max_tokens,
        "seed": seed,
        "enable_thinking": enable_thinking,
        "compare_ar": compare_ar,
        "ar_only": ar_only,
        "base_hidden_variant": resolved_base_hidden_variant,
        "mtp_hidden_variant": resolved_mtp_hidden_variant,
        "concat_order": resolved_concat_order,
        "mtp_cache_policy": mtp_cache_policy,
        "mtp_history_policy": mtp_history_policy,
        "draft_margin_threshold": draft_margin_threshold,
        "min_speculative_depth": min_speculative_depth,
        "verify_strategy": verify_strategy,
        "verify_core": verify_core,
        "draft_core": draft_core,
        "draft_lm_head": draft_lm_head_report,
        "mtp_quant_bits": getattr(contract, "mtp_quant_bits", None),
        "mtp_quant_group_size": getattr(contract, "mtp_quant_group_size", None),
        "mtp_quant_mode": getattr(contract, "mtp_quant_mode", None),
        "mtp_quant_policy": getattr(contract, "mtp_quant_policy", None),
        "mtp_adapter_path": str(mtp_adapter_path)
        if mtp_adapter_path is not None
        else None,
        "mtp_adapter_kind": (
            (getattr(rt, "mtp_adapter_metadata", None) or {}).get("kind")
            if getattr(rt, "mtp_adapter_metadata", None) is not None
            else None
        ),
        "mtp_adapter_merged": bool(getattr(rt, "mtp_adapter_merge_report", None)),
        "mtp_adapter_merge_report": getattr(rt, "mtp_adapter_merge_report", None),
        "mtp_corrector_path": str(mtp_corrector_path)
        if mtp_corrector_path is not None
        else None,
        "mtp_corrector_blend": mtp_corrector_blend,
        "mtp_corrector_kind": getattr(mtp_corrector, "kind", None)
        if mtp_corrector is not None
        else None,
        "online_hidden_corrector": {
            "alpha": online_hidden_corrector_alpha,
            "decay": online_hidden_corrector_decay,
            "warmup": online_hidden_corrector_warmup,
            "max_feed_depth": online_hidden_corrector_max_feed_depth,
            "key": online_hidden_corrector_key,
        },
        "online_correction_cache": {
            "enabled": online_correction_cache,
            "min_depth": online_correction_cache_min_depth,
            "key_policy": online_correction_cache_key,
            "prompt_enabled": prompt_correction_cache,
            "prompt_min_depth": prompt_correction_cache_min_depth,
        },
        "adapter_ensemble_q": {
            "enabled": adapter_ensemble_q,
            "epsilon": adapter_ensemble_epsilon,
            "min_depth": adapter_ensemble_min_depth,
        },
        "mtp_topk_reranker": (
            mtp_topk_reranker.to_dict()
            if mtp_topk_reranker is not None
            else {"enabled": False}
        ),
        "ar_rows": ar_rows,
        "depths": depth_results,
    }


def _sum_lists(values: list[list[int]], length: int) -> list[int]:
    totals = [0 for _ in range(length)]
    for value in values:
        for index, item in enumerate(value[:length]):
            totals[index] += int(item)
    return totals


def _sum_float_lists(values: list[list[float]], length: int) -> list[float]:
    totals = [0.0 for _ in range(length)]
    for value in values:
        for index, item in enumerate(value[:length]):
            totals[index] += float(item)
    return totals


def _sum_dicts(values: list[dict[str, int | float]]) -> dict[str, int | float]:
    totals: dict[str, int | float] = {}
    for value in values:
        for key, item in value.items():
            totals[key] = totals.get(key, 0) + item
    return totals


def _sum_correction_cache(values: list[dict[str, object]]) -> dict[str, int | bool]:
    key_policies = sorted(
        {
            str(value.get("key_policy"))
            for value in values
            if value.get("key_policy") is not None
        }
    )
    return {
        "enabled": any(bool(value.get("enabled")) for value in values),
        "hits": sum(int(value.get("hits", 0) or 0) for value in values),
        "stores": sum(int(value.get("stores", 0) or 0) for value in values),
        "entries": sum(int(value.get("entries", 0) or 0) for value in values),
        "prompt_enabled": any(bool(value.get("prompt_enabled")) for value in values),
        "prompt_hits": sum(int(value.get("prompt_hits", 0) or 0) for value in values),
        "prompt_stores": sum(
            int(value.get("prompt_stores", 0) or 0) for value in values
        ),
        "prompt_collisions": sum(
            int(value.get("prompt_collisions", 0) or 0) for value in values
        ),
        "prompt_skipped": sum(
            int(value.get("prompt_skipped", 0) or 0) for value in values
        ),
        "key_policy": key_policies[0]
        if len(key_policies) == 1
        else ",".join(key_policies),
    }


def _sum_adapter_ensemble_q(values: list[dict[str, object]]) -> dict[str, object]:
    epsilons = sorted(
        {
            str(value.get("epsilon"))
            for value in values
            if value.get("epsilon") is not None
        }
    )
    min_depths = sorted(
        {
            str(value.get("min_depth"))
            for value in values
            if value.get("min_depth") is not None
        }
    )
    return {
        "enabled": any(bool(value.get("enabled")) for value in values),
        "epsilon": epsilons[0] if len(epsilons) == 1 else ",".join(epsilons),
        "min_depth": min_depths[0] if len(min_depths) == 1 else ",".join(min_depths),
        "calls": sum(int(value.get("calls", 0) or 0) for value in values),
        "changed": sum(int(value.get("changed", 0) or 0) for value in values),
        "base_selected": sum(
            int(value.get("base_selected", 0) or 0) for value in values
        ),
        "adapter_selected": sum(
            int(value.get("adapter_selected", 0) or 0) for value in values
        ),
        "shared_selected": sum(
            int(value.get("shared_selected", 0) or 0) for value in values
        ),
        "fallbacks": sum(int(value.get("fallbacks", 0) or 0) for value in values),
    }


def _sum_topk_reranker(values: list[dict[str, object]]) -> dict[str, object]:
    calls = sum(int(value.get("calls", 0) or 0) for value in values)
    selected_rank_sum = sum(
        int(value.get("selected_rank_sum", 0) or 0) for value in values
    )
    depths = sorted(
        {
            str(depth)
            for value in values
            for depth in value.get("depths", [])  # type: ignore[union-attr]
        }
    )
    topks = sorted(
        {str(value.get("topk")) for value in values if value.get("topk") is not None}
    )
    return {
        "enabled": any(bool(value.get("enabled")) for value in values),
        "calls": calls,
        "changed": sum(int(value.get("changed", 0) or 0) for value in values),
        "fallbacks": sum(int(value.get("fallbacks", 0) or 0) for value in values),
        "selected_rank_sum": selected_rank_sum,
        "mean_selected_rank": selected_rank_sum / calls if calls else None,
        "topk": topks[0] if len(topks) == 1 else ",".join(topks),
        "depths": ",".join(depths),
    }


def _sum_draft_core(values: list[dict[str, object]]) -> dict[str, object]:
    requested = sorted(
        {
            str(value.get("requested"))
            for value in values
            if value.get("requested") is not None
        }
    )
    return {
        "requested": requested[0] if len(requested) == 1 else ",".join(requested),
        "device_d2_calls": sum(
            int(value.get("device_d2_calls", 0) or 0) for value in values
        ),
        "device_d2_fallbacks": sum(
            int(value.get("device_d2_fallbacks", 0) or 0) for value in values
        ),
        "device_d2_compile_time_s": sum(
            float(value.get("device_d2_compile_time_s", 0.0) or 0.0) for value in values
        ),
    }


def _speed_model_summary(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    generated = sum(float(row["generated_tokens"]) for row in rows)
    elapsed = sum(float(row["elapsed_s"]) for row in rows)
    repair = sum(float(row["repair_time_s"]) for row in rows)
    verify = sum(float(row["verify_time_s"]) for row in rows)
    draft = sum(float(row["draft_time_s"]) for row in rows)
    online_hidden = sum(
        float(row.get("online_hidden_corrector_time_s", 0.0)) for row in rows
    )
    if generated <= 0 or elapsed <= 0:
        return {
            "observed_tok_s": None,
            "ideal_no_repair_tok_s": None,
            "ideal_no_repair_verify25_tok_s": None,
            "ideal_no_repair_verify50_tok_s": None,
            "ideal_no_repair_verify50_draft30_tok_s": None,
            "ideal_no_repair_verify50_draft30_no_online_hidden_tok_s": None,
        }

    def rate(seconds_removed: float) -> float | None:
        remaining = elapsed - seconds_removed
        if remaining <= 0:
            return None
        return generated / remaining

    return {
        "observed_tok_s": generated / elapsed,
        "ideal_no_repair_tok_s": rate(repair),
        "ideal_no_repair_verify25_tok_s": rate(repair + 0.25 * verify),
        "ideal_no_repair_verify50_tok_s": rate(repair + 0.50 * verify),
        "ideal_no_repair_verify50_draft30_tok_s": rate(
            repair + 0.50 * verify + 0.30 * draft
        ),
        "ideal_no_repair_verify50_draft30_no_online_hidden_tok_s": rate(
            repair + 0.50 * verify + 0.30 * draft + online_hidden
        ),
    }


def write_depth_sweep(path: Path | str, result: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
