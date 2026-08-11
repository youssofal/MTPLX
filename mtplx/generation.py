"""Reference AR and native-MTP generation loops.

These loops intentionally favor correctness and observability over speed. The
optimized runtime can tighten the same contracts after the MTP-1 gates pass.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
import inspect
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal

import mlx.core as mx
import numpy as np

from .a3b_compiled_target_prefix import (
    ensure_a3b_whole_moe_request_preflight as _ensure_a3b_whole_moe_request_preflight,
    install_a3b_k1_target_prefix_route,
    validate_a3b_k1_device_draft_request,
    validate_a3b_k1_target_prefix_sampler,
)
from .a3b_whole_moe import validate_a3b_whole_moe_request
from .adaptive import AdaptiveDepthPolicy, ExpectedValueDepthPolicy
from .attention_context import attention_phase, model_forward_kind
from .deepseek_v4_adaptive_width import (
    validate_installed_deepseek_v4_adaptive_width_policy,
)
from .progress_heartbeat import tick as _owner_progress_tick
from .cache_state import (
    detach_array_leaf,
    detach_cache_state,
    owned_recurrent_state_stats,
    restore_cache,
    rollback_after_verify,
    trim_verified_window_without_snapshot,
    snapshot_cache,
    snapshot_untrimmable_cache,
    tail_owned_attention_kv_stats,
    trim_verified_window_to_prefix,
)
from .fast_sampling import (
    BatchedSparseDistributions,
    apply_penalties_mlx,
    batched_sparse_distributions_from_mlx_logits,
    sample_token_ids_from_mlx_logits,
    sparse_distribution_from_mlx_logits,
    sparse_distributions_from_mlx_logits,
)
from .gdn_capture import resolve_gdn_capture_backend
from .graphbank import (
    CompiledVerifyBank,
    SpecDecodeGraphBank,
    cache_array_tree,
    compiled_verify_mode,
    promote_kv_cache_offsets,
)
from .native_mlp import set_native_mlp_context
from .loop_guard import LoopGuard, loop_guard_config_from_env
from .thinking_guard import ThinkingGuard, ThinkingGuardConfig
from .profiles import resolve_long_context_mtp_depth
from .runtime import MTPLXRuntime
from .sampling import (
    SamplerConfig,
    SparseDistribution,
    acceptance_probability as compute_acceptance_probability,
    distribution_from_logits as dense_distribution_from_logits,
    residual_distribution,
    sample_from_distribution,
)
from .session_bank import _boundary_true_restore_enabled
from .runtime_options import block_prefix_restore_enabled, env_bool

Mode = Literal["ar", "mtp1", "mtpk", "mtpa"]
VerifyStrategy = Literal[
    "batched",
    "sequential",
    "capture",
    "capture_commit",
    "graphbank",
    "graphbank_capture_commit",
    "target_prefix",
    "trim_commit",
]

_PREFILL_CHUNK_SIZE_OVERRIDE: ContextVar[int | None] = ContextVar(
    "mtplx_prefill_chunk_size_override",
    default=None,
)


def reject_non_k1_a3b_whole_moe_request(rt: MTPLXRuntime, *, entrypoint: str) -> None:
    """Reject unsupported generation modes once, before they construct a prompt.

    generate_ar is supported: every one of its decode forwards is a single
    row, which the installed M1 route serves with per-row arithmetic that
    bit-matches the M2 verify route (enforced at install by the
    a3b_whole_moe_target_m1_m2_row_parity selfcheck lane).  Pure AR under
    whole-MoE is the ground-truth arm of the K1 AR-exactness gate.
    """

    if entrypoint == "generate_ar":
        return
    if bool(getattr(rt, "a3b_whole_moe_installed", False)):
        raise RuntimeError(
            f"installed A3B whole-MoE is owned by exact K1 generate_mtpk, not {entrypoint}"
        )


def ensure_a3b_whole_moe_request_preflight(
    rt: MTPLXRuntime,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    base_hidden_variant: str,
    prefill_layout: str | None = None,
) -> dict[str, Any]:
    """Prime the installed exact request geometry before prompt generation."""

    if not bool(getattr(rt, "a3b_whole_moe_installed", False)):
        return {"status": "disabled"}
    os.environ["MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS"] = str(len(prompt_ids))
    layout = _sustained_prefill_layout() if prefill_layout is None else prefill_layout
    return _ensure_a3b_whole_moe_request_preflight(
        rt,
        rt.a3b_compiled_target_prefix_factory,
        prompt_tokens=len(prompt_ids),
        max_tokens=max_tokens,
        hidden_variant=base_hidden_variant,
        cache_factory=lambda: _make_target_prefill_cache(rt),
        prefill_layout=layout,
    )


def _resolve_runtime_mtp_hidden_variant(
    rt: MTPLXRuntime,
    requested: str | None,
) -> str:
    if requested in {None, "auto", "contract"}:
        return str(getattr(rt.contract, "hidden_variant", "post_norm") or "post_norm")
    return str(requested)


def _resolve_runtime_base_hidden_variant(
    rt: MTPLXRuntime,
    requested: str | None,
) -> str:
    if requested in {None, "auto", "contract"}:
        return str(getattr(rt.contract, "base_hidden_variant", "post_norm") or "post_norm")
    return str(requested)


def _resolve_runtime_mtp_position_mode(rt: MTPLXRuntime) -> str:
    raw = os.environ.get("MTPLX_MTP_POSITION_MODE")
    if raw is None:
        raw = getattr(rt.contract, "mtp_position_mode", "cache")
    normalized = str(raw or "cache").strip().lower().replace("-", "_")
    if normalized in {"", "0", "off", "false", "default", "cache", "local"}:
        return "cache"
    return normalized


def _eval_value_summary(value: Any) -> dict[str, Any]:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "type": "array",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": [str(key) for key in value.keys()],
            "items": {
                str(key): _eval_value_summary(item) for key, item in value.items()
            },
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "items": [_eval_value_summary(item) for item in value],
        }
    return {"type": type(value).__name__}


def _eval(*values: Any, _caller_depth: int = 1) -> None:
    audit_path = os.environ.get("MTPLX_EVAL_AUDIT")
    if not audit_path:
        mx.eval(*values)
        # Every settled engine forward (prefill chunk, verify, AR step) proves
        # the model owner is alive; the stream stall watchdog compares readings.
        _owner_progress_tick()
        return

    try:
        caller = sys._getframe(_caller_depth)
    except ValueError:
        caller = None
    started = time.perf_counter()
    mx.eval(*values)
    _owner_progress_tick()
    elapsed_s = time.perf_counter() - started
    entry = {
        "elapsed_s": elapsed_s,
        "function": caller.f_code.co_name if caller is not None else None,
        "line": caller.f_lineno if caller is not None else None,
        "values": [_eval_value_summary(value) for value in values],
    }
    out = Path(audit_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    if os.environ.get("MTPLX_EVAL_AUDIT_STDERR"):
        print(json.dumps(entry, sort_keys=True), file=sys.stderr)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_falsey(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }


def _skip_verify_snapshot() -> bool:
    """The single parse of ``MTPLX_SKIP_VERIFY_SNAPSHOT`` (default OFF).

    The serve fast path force-sets this to "1"; whether that is safe is
    decided by the verify strategy, and the server now answers that from an
    explicit list of strategies known to survive without the snapshot
    rather than from a two-element list of the ones that need it.
    """

    return env_bool("MTPLX_SKIP_VERIFY_SNAPSHOT", default=False)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _generation_rate_fields(
    *,
    generated_tokens: int,
    elapsed_s: float,
    prompt_eval_time_s: float,
    cache_restore_time_s: float = 0.0,
) -> dict[str, float]:
    end_to_end_tok_s = generated_tokens / elapsed_s if elapsed_s > 0.0 else 0.0
    non_decode_elapsed_s = min(
        max(0.0, prompt_eval_time_s) + max(0.0, cache_restore_time_s),
        max(0.0, elapsed_s),
    )
    decode_elapsed_s = max(0.0, elapsed_s - non_decode_elapsed_s)
    decode_tok_s = (
        generated_tokens / decode_elapsed_s if decode_elapsed_s > 0.0 else 0.0
    )
    return {
        "tok_s": decode_tok_s,
        "decode_elapsed_s": decode_elapsed_s,
        "decode_tok_s": decode_tok_s,
        "end_to_end_tok_s": end_to_end_tok_s,
    }


def _normalize_mtp_history_policy(policy: str | None) -> str:
    normalized = (policy or "cycle").strip().lower().replace("-", "_")
    aliases = {
        "full": "committed",
        "lastwindow": "last_window",
        "window": "last_window",
        "none": "cycle",
        "off": "cycle",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"auto", "cycle", "committed", "last_window"}:
        raise ValueError(
            "mtp_history_policy must be 'auto', 'cycle', 'committed', "
            "'full', 'last_window', or 'none'"
        )
    return normalized


def _mtp_history_uses_committed_cache(policy: str) -> bool:
    return _normalize_mtp_history_policy(policy) in {"committed", "last_window"}


def _mtp_history_last_window_tokens() -> int:
    return max(1, _env_int("MTPLX_MTP_HISTORY_LAST_WINDOW", 8192))


def _resolve_mtp_history_policy(requested_policy: str, prompt_tokens: int) -> str:
    requested = _normalize_mtp_history_policy(requested_policy)
    env_policy = os.environ.get("MTPLX_MTP_HISTORY_POLICY")
    # Honor the env-var override whenever the caller requested either the
    # product default "committed" or the auto-resolution path. This keeps
    # diagnostic history-policy overrides reachable from the server hot path.
    if env_policy and requested in ("committed", "auto"):
        requested = _normalize_mtp_history_policy(env_policy)
    if requested != "auto":
        return requested
    threshold = max(
        1,
        _env_int("MTPLX_MTP_HISTORY_LAST_WINDOW_THRESHOLD", 16384),
    )
    return "last_window" if int(prompt_tokens) >= threshold else "committed"


def _runtime_count(rt: MTPLXRuntime, key: str, amount: int = 1) -> None:
    counters = getattr(rt, "diagnostic_counters", None)
    if counters is None:
        return
    counters[key] = int(counters.get(key, 0)) + int(amount)


def _runtime_counter_snapshot(rt: MTPLXRuntime) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in getattr(rt, "diagnostic_counters", {}).items()
    }


def _runtime_counter_delta(
    rt: MTPLXRuntime,
    before: dict[str, int],
) -> dict[str, int]:
    current = getattr(rt, "diagnostic_counters", {})
    keys = set(before) | set(current)
    return {
        str(key): int(current.get(key, 0)) - int(before.get(key, 0)) for key in keys
    }


def _attach_runtime_diagnostics(
    stats: "GenerationStats",
    rt: MTPLXRuntime,
    before: dict[str, int],
    *,
    ar_return_hidden: bool | None = None,
) -> None:
    counters = _runtime_counter_delta(rt, before)
    stats.runtime_mtp_enabled = bool(getattr(rt, "mtp_enabled", False))
    if ar_return_hidden is not None:
        stats.ar_return_hidden = bool(ar_return_hidden)
    stats.forward_ar_hidden_calls = int(counters.get("forward_ar_hidden_calls", 0))
    stats.forward_ar_plain_calls = int(counters.get("forward_ar_plain_calls", 0))
    stats.mtp_forward_calls = int(counters.get("draft_mtp_calls", 0))
    stats.make_mtp_cache_calls = int(counters.get("make_mtp_cache_calls", 0))
    stats.update_mtp_cache_calls = int(counters.get("update_mtp_cache_calls", 0))
    stats.mtp_history_append_calls = int(counters.get("mtp_history_append_calls", 0))
    stats.full_logits_tokens_emitted = int(
        counters.get("full_logits_tokens_emitted", 0)
    )
    stats.final_logits_tokens_emitted = int(
        counters.get("final_logits_tokens_emitted", 0)
    )
    stats.logits_tokens_emitted = int(counters.get("logits_tokens_emitted", 0))
    stats.prefill_chunks = int(counters.get("prefill_chunks", 0))
    stats.prefill_chunk_size = _prefill_chunk_size()
    stats.prefill_chunk_cache_cleanup_enabled = _prefill_chunk_cache_cleanup_enabled()
    stats.prefill_chunk_cache_cleanup_every = _prefill_chunk_cache_cleanup_every()
    stats.prefill_chunk_cache_cleanup_events = int(
        counters.get("prefill_chunk_cache_cleanup_events", 0)
    )
    stats.prefill_stock_cache_only_enabled = _prefill_stock_cache_only_enabled()
    stats.prefill_stock_cache_only_calls = int(
        counters.get("prefill_stock_cache_only_calls", 0)
    )
    stats.prefill_omlx_external_enabled = _prefill_omlx_external_enabled()
    stats.prefill_omlx_external_calls = int(
        counters.get("prefill_omlx_external_calls", 0)
    )
    stats.prefill_external_emit_logits_enabled = _prefill_external_emit_logits_enabled()
    stats.prefill_external_cache_only_calls = int(
        counters.get("prefill_external_cache_only_calls", 0)
    )
    owned_attn = stats.owned_attn_kv if isinstance(stats.owned_attn_kv, dict) else {}
    stats.paged_kv_capacity_tokens = int(owned_attn.get("capacity") or 0)
    stats.paged_kv_num_blocks = int(owned_attn.get("num_blocks") or 0)
    stats.paged_active_array_calls = int(owned_attn.get("active_array_calls") or 0)
    stats.paged_active_array_time_s = float(
        owned_attn.get("active_array_time_s") or 0.0
    )
    stats.paged_turboquant = bool(owned_attn.get("turboquant") or False)
    stats.paged_turboquant_k_quant = str(owned_attn.get("turboquant_k_quant") or "")
    stats.paged_turboquant_v_quant = str(owned_attn.get("turboquant_v_quant") or "")
    stats.paged_turboquant_attention_calls = int(
        owned_attn.get("turboquant_attention_calls") or 0
    )
    stats.paged_kv_quant = bool(owned_attn.get("kv_quant") or False)
    stats.paged_kv_quant_mode = str(owned_attn.get("kv_quant_mode") or "")
    stats.paged_kv_quant_attention_calls = int(
        owned_attn.get("kv_quant_attention_calls") or 0
    )
    stats.paged_kv_quant_dequant_calls = int(
        owned_attn.get("kv_quant_dequant_calls") or 0
    )
    stats.paged_kv_quant_dequant_time_s = float(
        owned_attn.get("kv_quant_dequant_time_s") or 0.0
    )
    stats.paged_kv_quant_dequant_tokens = int(
        owned_attn.get("kv_quant_dequant_tokens") or 0
    )
    stats.paged_gqa_sdpa_calls = int(owned_attn.get("gqa_sdpa_calls") or 0)
    gqa_by_route = owned_attn.get("gqa_sdpa_calls_by_route") or {}
    stats.paged_gqa_sdpa_calls_by_route = (
        dict(gqa_by_route) if isinstance(gqa_by_route, dict) else {}
    )
    gqa_by_phase = owned_attn.get("gqa_sdpa_calls_by_phase") or {}
    stats.paged_gqa_sdpa_calls_by_phase = (
        dict(gqa_by_phase) if isinstance(gqa_by_phase, dict) else {}
    )
    gqa_misses = owned_attn.get("gqa_sdpa_route_misses_by_phase_reason") or {}
    stats.paged_gqa_sdpa_route_misses_by_phase_reason = (
        dict(gqa_misses) if isinstance(gqa_misses, dict) else {}
    )
    gqa_misses_by_q = owned_attn.get("gqa_sdpa_route_misses_by_q_len") or {}
    stats.paged_gqa_sdpa_route_misses_by_q_len = (
        dict(gqa_misses_by_q) if isinstance(gqa_misses_by_q, dict) else {}
    )
    gqa_last_miss = owned_attn.get("gqa_sdpa_last_route_miss") or {}
    stats.paged_gqa_sdpa_last_route_miss = (
        dict(gqa_last_miss) if isinstance(gqa_last_miss, dict) else {}
    )
    stats.attention_dense_fallback_calls = int(
        owned_attn.get("dense_fallback_calls") or 0
    )
    stats.prefill_dense_fallback_calls = int(
        owned_attn.get("prefill_dense_fallback_calls") or 0
    )
    stats.decode_dense_fallback_calls = int(
        owned_attn.get("decode_dense_fallback_calls") or 0
    )
    stats.ar_dense_fallback_calls = int(owned_attn.get("ar_dense_fallback_calls") or 0)
    stats.postcommit_dense_fallback_calls = int(
        owned_attn.get("postcommit_dense_fallback_calls") or 0
    )
    bailouts = owned_attn.get("paged_attention_bailouts_by_phase_reason") or {}
    stats.paged_attention_bailouts_by_phase_reason = (
        dict(bailouts) if isinstance(bailouts, dict) else {}
    )
    stats.paged_attention_large_q_path = str(
        owned_attn.get("paged_attention_large_q_path") or ""
    )
    stats.prefill_route = (
        _sustained_prefill_layout()
        if _contiguous_prefill_cache_layout_enabled()
        else stats.paged_attention_large_q_path
    )
    stats.large_q_split_sdpa_fallback_calls = int(
        owned_attn.get("large_q_split_sdpa_fallback_calls") or 0
    )
    large_q_by_phase = (
        owned_attn.get("large_q_split_sdpa_fallback_calls_by_phase") or {}
    )
    stats.large_q_split_sdpa_fallback_calls_by_phase = (
        dict(large_q_by_phase) if isinstance(large_q_by_phase, dict) else {}
    )
    stats.prefill_large_q_split_sdpa_fallback_calls = int(
        owned_attn.get("prefill_large_q_split_sdpa_fallback_calls") or 0
    )
    stats.decode_large_q_split_sdpa_fallback_calls = int(
        owned_attn.get("decode_large_q_split_sdpa_fallback_calls") or 0
    )
    stats.partitioned_paged_calls = int(owned_attn.get("partitioned_paged_calls") or 0)
    partitioned_by_phase = owned_attn.get("partitioned_paged_calls_by_phase") or {}
    stats.partitioned_paged_calls_by_phase = (
        dict(partitioned_by_phase) if isinstance(partitioned_by_phase, dict) else {}
    )
    stats.prefill_partitioned_paged_calls = int(
        owned_attn.get("prefill_partitioned_paged_calls") or 0
    )
    stats.decode_partitioned_paged_calls = int(
        owned_attn.get("decode_partitioned_paged_calls") or 0
    )


def _sustained_prefill_enabled() -> bool:
    return _env_truthy("MTPLX_SUSTAINED_PREFILL")


def _final_logits_prefill_enabled() -> bool:
    return _sustained_prefill_enabled() or _env_falsey(
        "MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS"
    )


def _prefill_chunk_cache_cleanup_enabled() -> bool:
    return _env_truthy("MTPLX_PREFILL_CHUNK_CACHE_CLEANUP")


def _prefill_chunk_cache_cleanup_every() -> int:
    raw = os.environ.get("MTPLX_PREFILL_CHUNK_CACHE_CLEANUP_EVERY")
    if raw is None or not str(raw).strip():
        return 1
    raw_text = str(raw).strip().lower()
    if raw_text == "auto":
        # Dense layout: cleanup every 4 chunks. The per-chunk
        # synchronize+clear_cache was costing 5-21% prefill throughput with
        # zero memory benefit (A/B 2026-07-05, fresh daemon per arm, max
        # fans, 2048-token chunks: 16k 565->682 pp, 32k 521->547, 64k
        # 423->464, 128k 294->315 tok/s; peak memory byte-identical at
        # 20.7/25.4/30.6/41.5 GB). The repage layout keeps its measured
        # every-2 cadence: its chunk intermediates feed the repage copy and
        # accumulate differently.
        return 2 if _sustained_prefill_layout() == "contiguous_then_repage" else 4
    try:
        return max(1, int(raw_text))
    except ValueError:
        return 1


def _prefill_chunk_cache_cleanup(rt: MTPLXRuntime) -> float:
    if not _prefill_chunk_cache_cleanup_enabled():
        return 0.0
    every = _prefill_chunk_cache_cleanup_every()
    pending = (
        int(rt.diagnostic_counters.get("_prefill_chunks_since_cache_cleanup", 0)) + 1
    )
    rt.diagnostic_counters["_prefill_chunks_since_cache_cleanup"] = pending
    if pending < every:
        return 0.0
    rt.diagnostic_counters["_prefill_chunks_since_cache_cleanup"] = 0
    started = time.perf_counter()
    try:
        mx.synchronize()
    except RuntimeError:
        pass
    mx.clear_cache()
    _runtime_count(rt, "prefill_chunk_cache_cleanup_events")
    return time.perf_counter() - started


def _prefill_stock_cache_only_enabled() -> bool:
    return _env_truthy("MTPLX_PREFILL_STOCK_CACHE_ONLY") and _env_truthy(
        "MTPLX_ALLOW_UNSAFE_PREFILL_STOCK_CACHE_ONLY"
    )


def _unsafe_long_context_prefill_guard_tokens() -> int:
    raw = os.environ.get("MTPLX_UNSAFE_LONG_CONTEXT_PREFILL_GUARD_TOKENS")
    if raw is None or not str(raw).strip():
        return 16384
    try:
        return max(0, int(str(raw).strip()))
    except ValueError:
        return 16384


def _unsafe_long_context_prefill_allowed() -> bool:
    return _env_truthy("MTPLX_ALLOW_UNSAFE_LONG_CONTEXT_PREFILL")


def _assert_safe_long_context_prefill(prompt_tokens: int) -> None:
    if _sustained_prefill_enabled() or _unsafe_long_context_prefill_allowed():
        return
    threshold = _unsafe_long_context_prefill_guard_tokens()
    if threshold <= 0 or int(prompt_tokens) < threshold:
        return
    raise RuntimeError(
        "Blocked unsafe long-context MTP prefill path: "
        f"{int(prompt_tokens)} prompt tokens would use the non-Sustained full "
        "hidden/logits prefill route. Start MTPLX with `--profile sustained` "
        "or run `mtplx config set profile sustained`. To intentionally run "
        "this diagnostic path, set MTPLX_ALLOW_UNSAFE_LONG_CONTEXT_PREFILL=1."
    )


def _prefill_omlx_external_enabled() -> bool:
    return _env_truthy("MTPLX_PREFILL_OMLX_EXTERNAL")


def _prefill_external_cache_only_enabled() -> bool:
    return _prefill_omlx_external_enabled() or _prefill_stock_cache_only_enabled()


def _prefill_external_emit_logits_enabled() -> bool:
    return not _env_falsey("MTPLX_PREFILL_EXTERNAL_EMIT_LOGITS")


def _batched_token_array(token_ids: Any) -> mx.array:
    if hasattr(token_ids, "shape") and hasattr(token_ids, "dtype"):
        if len(token_ids.shape) == 1:
            return token_ids[None]
        return token_ids
    return mx.array([token_ids])


def _prefill_cache_only_forward(
    rt: MTPLXRuntime,
    token_ids: Any,
    cache: Any,
    input_embeddings: Any | None = None,
) -> Any:
    token_array = _batched_token_array(token_ids)
    if not _prefill_external_cache_only_enabled():
        return rt.forward_ar(
            token_array,
            cache=cache,
            return_hidden=False,
            emit_logits=not _final_logits_prefill_enabled(),
            input_embeddings=input_embeddings,
        )
    _runtime_count(rt, "prefill_external_cache_only_calls")
    if _prefill_stock_cache_only_enabled():
        _runtime_count(rt, "prefill_stock_cache_only_calls")
    if _prefill_omlx_external_enabled():
        _runtime_count(rt, "prefill_omlx_external_calls")
    if not _prefill_external_emit_logits_enabled():
        return rt.forward_ar(
            token_array,
            cache=cache,
            return_hidden=False,
            emit_logits=False,
            input_embeddings=input_embeddings,
        )
    if input_embeddings is not None:
        unused_logits = rt.model(
            token_array, cache=cache, input_embeddings=input_embeddings
        )
    else:
        unused_logits = rt.model(token_array, cache=cache)
    del unused_logits
    return None


def _forward_ar_optional_hidden(
    rt: MTPLXRuntime,
    token_array: Any,
    *,
    cache: Any,
    hidden_variant: str | None,
    emit_logits: bool = True,
    logits_keep: int | None = None,
    input_embeddings: Any | None = None,
) -> tuple[Any, Any]:
    """`forward_ar` as (logits, hidden), with hidden None on target-only runtimes.

    Only request hidden states from a runtime that can produce them. Target-only
    AR runtimes (laguna_ar) have no draft head: their forward_ar returns logits
    alone, so an ungated ``return_hidden=True`` unpacks a lone logits array as
    ``(logits, hidden)`` and raises "not enough values to unpack (expected 2,
    got 1)" — the live serving crash in the warm session-restore suffix prefill.
    `hidden_variant` travels only on the hidden branch for the same reason: the
    generic runtime forwards it to the model as a kwarg a stock target does not
    accept. This mirrors the cold prefill path and generate_ar, which both gate
    return_hidden on rt.mtp_enabled. Callers must treat hidden as optional.
    """

    if not rt.mtp_enabled:
        logits = rt.forward_ar(
            token_array,
            cache=cache,
            return_hidden=False,
            emit_logits=emit_logits,
            logits_keep=logits_keep,
            input_embeddings=input_embeddings,
        )
        return logits, None
    return rt.forward_ar(
        token_array,
        cache=cache,
        return_hidden=True,
        hidden_variant=hidden_variant,
        emit_logits=emit_logits,
        logits_keep=logits_keep,
        input_embeddings=input_embeddings,
    )


def _prefill_chunk_size() -> int:
    override = _PREFILL_CHUNK_SIZE_OVERRIDE.get()
    if override is not None:
        return max(1, int(override))
    raw = (os.environ.get("MTPLX_PREFILL_CHUNK_SIZE") or "2048").strip().lower()
    if raw == "auto":
        layout = _sustained_prefill_layout()
        if layout == "contiguous_dense_decode":
            return max(1, _env_int("MTPLX_PREFILL_CHUNK_SIZE_DENSE", 2048))
        return max(1, _env_int("MTPLX_PREFILL_CHUNK_SIZE_REPAGE", 2048))
    try:
        return max(1, int(raw))
    except ValueError:
        return 2048


@contextmanager
def prefill_chunk_size_override(chunk_size: int | None):
    """Apply a request-local prefill chunk override.

    The legacy env knob remains supported for profiles and CLI diagnostics, but
    the native app needs a live next-request setting. A ContextVar keeps that
    override off process-global environment state.
    """

    token = _PREFILL_CHUNK_SIZE_OVERRIDE.set(
        None if chunk_size is None else max(1, int(chunk_size))
    )
    try:
        yield
    finally:
        _PREFILL_CHUNK_SIZE_OVERRIDE.reset(token)


def _iter_prefill_chunks(token_ids: list[int]) -> list[list[int]]:
    if not token_ids:
        return []
    if not _sustained_prefill_enabled():
        return [token_ids]
    chunk_size = _prefill_chunk_size()
    return [
        token_ids[start : start + chunk_size]
        for start in range(0, len(token_ids), chunk_size)
    ]


def _split_spans_at(
    spans: list[tuple[int, int]], edges: tuple[int, ...]
) -> list[tuple[int, int]]:
    """Split contiguous spans so every in-range edge is an exact span end.

    Used to align a prefill chunk boundary with a stable prompt-prefix
    position (the pre-injection boundary of the transient trailing tool
    hint), so the existing gdn-boundary capture records recurrent state
    exactly there. Chunked prefill is mathematically split-invariant; only
    the chunk layout changes. Edges outside (0, total) or already on a
    span end are no-ops.
    """
    if not spans or not edges:
        return spans
    out = spans
    for edge in sorted(set(int(e) for e in edges)):
        split: list[tuple[int, int]] = []
        for start, end in out:
            if start < edge < end:
                split.append((start, edge))
                split.append((edge, end))
            else:
                split.append((start, end))
        out = split
    return out


def _iter_prefill_chunk_spans(
    token_count: int, *, mandatory_edges: tuple[int, ...] = ()
) -> list[tuple[int, int]]:
    if token_count <= 0:
        return []
    if not _sustained_prefill_enabled():
        return _split_spans_at([(0, token_count)], mandatory_edges)
    chunk_size = _prefill_chunk_size()
    return _split_spans_at(
        [
            (start, min(token_count, start + chunk_size))
            for start in range(0, token_count, chunk_size)
        ],
        mandatory_edges,
    )


def _sustained_prefill_layout() -> str:
    layout = (
        os.environ.get("MTPLX_SUSTAINED_PREFILL_LAYOUT", "")
        .strip()
        .lower()
        .replace("-", "_")
    )
    if layout != "auto":
        return layout
    # Canonicalize through the one parser: a raw membership test here missed
    # documented spellings ("8", "8bit", "uint8") that the rest of the stack
    # honours as q8, and silently picked the dense-decode layout for a
    # quantized cache.
    from .kv_quant import paged_kv_quant_mode_from_env

    if paged_kv_quant_mode_from_env() != "off":
        return "contiguous_then_repage"
    context_tokens = _env_int("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", 0)
    dense_max = _env_int("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", 131072)
    if context_tokens > 0 and context_tokens <= dense_max:
        return "contiguous_dense_decode"
    return "contiguous_then_repage"


def _defer_verify_hidden_eval_enabled() -> bool:
    raw = (os.environ.get("MTPLX_DEFER_VERIFY_HIDDEN_EVAL") or "").strip().lower()
    if raw == "auto":
        context_tokens = _env_int("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", 0)
        dense_max = _env_int("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", 131072)
        return context_tokens > 0 and context_tokens <= dense_max
    return _env_truthy("MTPLX_DEFER_VERIFY_HIDDEN_EVAL")


def _verify_hidden_mode() -> str:
    raw = (
        (os.environ.get("MTPLX_VERIFY_HIDDEN_MODE") or "default")
        .strip()
        .lower()
        .replace("-", "_")
    )
    return raw or "default"


def _clear_cache_every() -> int:
    raw = (os.environ.get("MTPLX_CLEAR_CACHE_EVERY") or "auto").strip().lower()
    if raw == "auto":
        context_tokens = _env_int("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", 0)
        # Lowered default 98304 -> 16384 so clear_cache fires for the typical
        # opencode subagent context regime (16-40K) where wired-memory pressure
        # has been observed in practice. The previous threshold only kicked in
        # past 96K, well above the crash zone.
        threshold = _env_int("MTPLX_CLEAR_CACHE_EVERY_CONTEXT_THRESHOLD", 16384)
        if context_tokens >= threshold and _contiguous_dense_decode_prefill_enabled():
            # Default 16 tokens was per-step aggressive (sync barrier every
            # tick). 256 amortized it; 1024 (2026-07-16) removes the remaining
            # -3.8% decode tax on 512-token generations at 33k ctx while
            # marathon responses still get periodic allocator bounding.
            return max(0, _env_int("MTPLX_CLEAR_CACHE_EVERY_LONG_CONTEXT", 1024))
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _contiguous_then_repage_prefill_enabled() -> bool:
    return _sustained_prefill_layout() == "contiguous_then_repage"


def _contiguous_dense_decode_prefill_enabled() -> bool:
    return _sustained_prefill_layout() == "contiguous_dense_decode"


def _contiguous_prefill_cache_layout_enabled() -> bool:
    return (
        _contiguous_then_repage_prefill_enabled()
        or _contiguous_dense_decode_prefill_enabled()
    )


@contextmanager
def _target_prefill_cache_layout_scope():
    if not _contiguous_prefill_cache_layout_enabled():
        yield
        return
    keys = (
        "MTPLX_VLLM_METAL_PAGED_ATTN",
        "MTPLX_OWNED_ATTN_KV",
        "MTPLX_BLOCK_OWNED_ATTN_KV",
    )
    saved = {key: os.environ.get(key) for key in keys}
    os.environ["MTPLX_VLLM_METAL_PAGED_ATTN"] = "0"
    os.environ["MTPLX_OWNED_ATTN_KV"] = "0"
    os.environ["MTPLX_BLOCK_OWNED_ATTN_KV"] = "0"
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _make_target_prefill_cache(rt: MTPLXRuntime):
    with _target_prefill_cache_layout_scope():
        return rt.make_cache()


def _maybe_repage_target_prefill_cache(rt: MTPLXRuntime, cache: Any) -> float:
    if not _contiguous_then_repage_prefill_enabled():
        return 0.0

    started = time.perf_counter()
    if not rt.repage_target_prefill_cache(cache):
        return 0.0
    _eval_cache_roots(cache)
    return time.perf_counter() - started


def _session_restore_cache_factory(rt: MTPLXRuntime) -> Callable[[], Any] | None:
    if not _contiguous_prefill_cache_layout_enabled():
        return None
    return lambda: _make_target_prefill_cache(rt)


def _session_live_frontier_reference_restore_enabled() -> bool:
    name = "MTPLX_SESSION_LIVE_FRONTIER_REFERENCE_RESTORE"
    if name not in os.environ:
        name = "MTPLX_OPENCODE_TOOL_HISTORY_LIVE_FRONTIER"
    if name not in os.environ:
        return False
    return _env_truthy(name)


def _eval_cache_roots(cache: Any) -> None:
    arrays = _tree_mx_arrays(cache)
    if not arrays:
        return
    deduped: list[mx.array] = []
    seen: set[int] = set()
    for array in arrays:
        ident = id(array)
        if ident in seen:
            continue
        seen.add(ident)
        deduped.append(array)
    if deduped:
        _eval(*deduped, _caller_depth=2)


def _eval_verify_outputs(
    verify_logits: mx.array, verify_hidden: mx.array, captures: Any | None = None
) -> dict[str, float]:
    # Keep capture tensors lazy; commit_captured_prefix materializes only the selected prefix slice.
    timings = {
        "verify_logits_eval_time_s": 0.0,
        "verify_hidden_eval_time_s": 0.0,
        "verify_joint_eval_time_s": 0.0,
    }
    if _env_truthy("MTPLX_LAZY_VERIFY_LOGITS"):
        started = time.perf_counter()
        _eval(verify_hidden, _caller_depth=2)
        timings["verify_hidden_eval_time_s"] += time.perf_counter() - started
        return timings
    if _env_truthy("MTPLX_SPLIT_VERIFY_EVAL"):
        started = time.perf_counter()
        _eval(verify_logits, _caller_depth=2)
        timings["verify_logits_eval_time_s"] += time.perf_counter() - started
        started = time.perf_counter()
        _eval(verify_hidden, _caller_depth=2)
        timings["verify_hidden_eval_time_s"] += time.perf_counter() - started
        return timings
    started = time.perf_counter()
    _eval(verify_logits, verify_hidden, _caller_depth=2)
    timings["verify_joint_eval_time_s"] += time.perf_counter() - started
    return timings


def _tree_nbytes(value: Any, seen: set[int] | None = None) -> int:
    """Best-effort recursive byte count for MLX/NumPy array trees."""
    if value is None:
        return 0
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return 0
    seen.add(value_id)
    if isinstance(value, mx.array):
        return int(value.nbytes)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (str, bytes, bytearray, int, float, bool)):
        return 0
    if isinstance(value, dict):
        return sum(_tree_nbytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(_tree_nbytes(item, seen) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return sum(
            _tree_nbytes(getattr(value, item.name), seen) for item in fields(value)
        )
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        return sum(_tree_nbytes(item, seen) for item in attrs.values())
    return 0


def _tree_mx_arrays(value: Any, seen: set[int] | None = None) -> list[mx.array]:
    if value is None:
        return []
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return []
    seen.add(value_id)
    if isinstance(value, mx.array):
        return [value]
    if isinstance(value, np.ndarray):
        return []
    if isinstance(value, (str, bytes, bytearray, int, float, bool)):
        return []
    if isinstance(value, dict):
        arrays: list[mx.array] = []
        for item in value.values():
            arrays.extend(_tree_mx_arrays(item, seen))
        return arrays
    if isinstance(value, (list, tuple, set)):
        arrays = []
        for item in value:
            arrays.extend(_tree_mx_arrays(item, seen))
        return arrays
    if is_dataclass(value) and not isinstance(value, type):
        arrays = []
        for item in fields(value):
            arrays.extend(_tree_mx_arrays(getattr(value, item.name), seen))
        return arrays
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        arrays = []
        for item in attrs.values():
            arrays.extend(_tree_mx_arrays(item, seen))
        return arrays
    return []


def _mlx_memory_stats() -> dict[str, int]:
    return {
        "active_memory_bytes": int(mx.get_active_memory()),
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "cache_memory_bytes": int(mx.get_cache_memory()),
    }


class _DecodeTrace:
    def __init__(
        self,
        *,
        prompt_tokens: int,
        max_tokens: int,
        speculative_depth: int,
        sampler: SamplerConfig,
        verify_strategy: str,
        verify_core: str,
        mtp_history_policy: str,
        mtp_cache_policy: str,
        trace_label: str | None,
        trace_metadata: dict[str, Any] | None,
    ) -> None:
        trace_path = os.environ.get("MTPLX_DECODE_TRACE_JSONL")
        self.enabled = bool(trace_path)
        self.path = Path(trace_path).expanduser() if trace_path else None
        self.interval_s = max(
            0.1,
            float(os.environ.get("MTPLX_DECODE_TRACE_INTERVAL_S") or 1.0),
        )
        self.run_id = f"{int(time.time() * 1000)}-{os.getpid()}-{id(self):x}"
        self.label = trace_label or os.environ.get("MTPLX_DECODE_TRACE_LABEL") or None
        self.metadata = dict(trace_metadata or {})
        self.prompt_tokens = int(prompt_tokens)
        self.max_tokens = int(max_tokens)
        self.speculative_depth = int(speculative_depth)
        self.sampler = sampler
        self.verify_strategy = verify_strategy
        self.verify_core = verify_core
        self.mtp_history_policy = mtp_history_policy
        self.mtp_cache_policy = mtp_cache_policy
        self.started_s = time.perf_counter()
        self.last_emit_s = self.started_s
        self.bucket_index = 0
        self.last_totals: dict[str, Any] = {
            "generated_tokens": 0,
            "accepted_drafts": 0,
            "rejected_drafts": 0,
            "drafted_tokens": 0,
            "verify_calls": 0,
            "correction_tokens": 0,
            "bonus_tokens": 0,
            "verify_time_s": 0.0,
            "verify_forward_time_s": 0.0,
            "verify_eval_time_s": 0.0,
            "verify_logits_eval_time_s": 0.0,
            "verify_hidden_eval_time_s": 0.0,
            "verify_joint_eval_time_s": 0.0,
            "verify_target_distribution_time_s": 0.0,
            "target_distribution_materialized_rows": 0,
            "target_distribution_materialized_windows": 0,
            "lazy_bonus_verify_calls": 0,
            "lazy_bonus_commit_time_s": 0.0,
            "verify_eval_unattributed_time_s": 0.0,
            "draft_time_s": 0.0,
            "accept_time_s": 0.0,
            "repair_time_s": 0.0,
            "commit_time_s": 0.0,
            "capture_commit_time_s": 0.0,
            "snapshot_time_s": 0.0,
            "bonus_time_s": 0.0,
            "verify_output_nbytes": 0,
            "draft_output_nbytes": 0,
            "mtp_history_append_nbytes": 0,
            "clear_cache_events": 0,
            "clear_cache_time_s": 0.0,
            "trunk_cache_materialize_events": 0,
            "trunk_cache_materialize_time_s": 0.0,
            "dirty_detach_events": 0,
            "dirty_detach_time_s": 0.0,
            "dirty_detach_arrays": 0,
            "dirty_detach_bytes": 0,
            "live_output_detach_events": 0,
            "live_output_detach_time_s": 0.0,
            "live_output_detach_arrays": 0,
            "live_output_detach_bytes": 0,
            "state_rebase_events": 0,
            "state_rebase_time_s": 0.0,
            "state_root_eval_events": 0,
            "state_root_eval_time_s": 0.0,
            "state_root_eval_arrays": 0,
            "trace_accounting_time_s": 0.0,
            "accepted_by_depth": [0 for _ in range(speculative_depth)],
            "drafted_by_depth": [0 for _ in range(speculative_depth)],
            "accept_probability_sum_by_depth": [0.0 for _ in range(speculative_depth)],
        }
        if self.enabled and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def _delta(self, totals: dict[str, Any], key: str) -> Any:
        # Lanes maintain different counter sets (AR omits MTP-only keys);
        # a counter absent on either side is a zero delta, not an error.
        value = totals.get(key)
        previous = self.last_totals.get(key)
        if value is None:
            value = previous if previous is not None else 0
        if previous is None:
            previous = [0.0] * len(value) if isinstance(value, list) else 0
        if isinstance(value, list):
            return [(float(item) - float(prev)) for item, prev in zip(value, previous)]
        return value - previous

    def maybe_emit(
        self,
        *,
        force: bool,
        final: bool,
        totals: dict[str, Any],
        cache: Any,
        mtp_cache: Any,
        mtp_history_materialize_every: int,
        mtp_history_materialize_events: int,
    ) -> None:
        if not self.enabled or self.path is None:
            return
        now = time.perf_counter()
        if not force and now - self.last_emit_s < self.interval_s:
            return
        elapsed_s = max(0.0, now - self.last_emit_s)
        generated_delta = int(self._delta(totals, "generated_tokens"))
        drafted_by_depth_delta = [
            int(item) for item in self._delta(totals, "drafted_by_depth")
        ]
        accepted_by_depth_delta = [
            int(item) for item in self._delta(totals, "accepted_by_depth")
        ]
        accept_probability_sum_delta = [
            float(item)
            for item in self._delta(totals, "accept_probability_sum_by_depth")
        ]
        acceptance_rate_by_depth_delta = [
            (float(accepted) / int(drafted) if drafted else None)
            for accepted, drafted in zip(
                accepted_by_depth_delta, drafted_by_depth_delta
            )
        ]
        mean_accept_probability_by_depth_delta = [
            (float(total) / int(drafted) if drafted else None)
            for total, drafted in zip(
                accept_probability_sum_delta, drafted_by_depth_delta
            )
        ]
        verify_calls_delta = int(self._delta(totals, "verify_calls"))
        accepted_drafts_delta = int(self._delta(totals, "accepted_drafts"))
        drafted_tokens_delta = int(self._delta(totals, "drafted_tokens"))
        verify_time_delta = float(self._delta(totals, "verify_time_s"))
        verify_forward_time_delta = float(self._delta(totals, "verify_forward_time_s"))
        verify_eval_time_delta = float(self._delta(totals, "verify_eval_time_s"))
        verify_logits_eval_time_delta = float(
            self._delta(totals, "verify_logits_eval_time_s")
        )
        verify_hidden_eval_time_delta = float(
            self._delta(totals, "verify_hidden_eval_time_s")
        )
        verify_joint_eval_time_delta = float(
            self._delta(totals, "verify_joint_eval_time_s")
        )
        verify_target_distribution_time_delta = float(
            self._delta(totals, "verify_target_distribution_time_s")
        )
        target_distribution_rows_delta = int(
            self._delta(totals, "target_distribution_materialized_rows")
        )
        target_distribution_windows_delta = int(
            self._delta(totals, "target_distribution_materialized_windows")
        )
        lazy_bonus_verify_calls_delta = int(
            self._delta(totals, "lazy_bonus_verify_calls")
        )
        lazy_bonus_commit_time_delta = float(
            self._delta(totals, "lazy_bonus_commit_time_s")
        )
        verify_eval_unattributed_time_delta = float(
            self._delta(totals, "verify_eval_unattributed_time_s")
        )
        draft_time_delta = float(self._delta(totals, "draft_time_s"))
        clear_cache_events_delta = int(self._delta(totals, "clear_cache_events"))
        clear_cache_time_delta = float(self._delta(totals, "clear_cache_time_s"))
        trunk_cache_materialize_events_delta = int(
            self._delta(totals, "trunk_cache_materialize_events")
        )
        trunk_cache_materialize_time_delta = float(
            self._delta(totals, "trunk_cache_materialize_time_s")
        )
        dirty_detach_events_delta = int(self._delta(totals, "dirty_detach_events"))
        dirty_detach_time_delta = float(self._delta(totals, "dirty_detach_time_s"))
        dirty_detach_arrays_delta = int(self._delta(totals, "dirty_detach_arrays"))
        dirty_detach_bytes_delta = int(self._delta(totals, "dirty_detach_bytes"))
        live_output_detach_events_delta = int(
            self._delta(totals, "live_output_detach_events")
        )
        live_output_detach_time_delta = float(
            self._delta(totals, "live_output_detach_time_s")
        )
        live_output_detach_arrays_delta = int(
            self._delta(totals, "live_output_detach_arrays")
        )
        live_output_detach_bytes_delta = int(
            self._delta(totals, "live_output_detach_bytes")
        )
        state_rebase_events_delta = int(self._delta(totals, "state_rebase_events"))
        state_rebase_time_delta = float(self._delta(totals, "state_rebase_time_s"))
        state_root_eval_events_delta = int(
            self._delta(totals, "state_root_eval_events")
        )
        state_root_eval_time_delta = float(
            self._delta(totals, "state_root_eval_time_s")
        )
        state_root_eval_arrays_delta = int(
            self._delta(totals, "state_root_eval_arrays")
        )
        trace_accounting_time_delta = float(
            self._delta(totals, "trace_accounting_time_s")
        )
        bytes_delta = {
            "verify_output_nbytes_delta": int(
                self._delta(totals, "verify_output_nbytes")
            ),
            "draft_output_nbytes_delta": int(
                self._delta(totals, "draft_output_nbytes")
            ),
            "mtp_history_append_nbytes_delta": int(
                self._delta(totals, "mtp_history_append_nbytes")
            ),
        }
        materialized_nbytes = sum(bytes_delta.values())
        row = {
            "event": "decode_trace_bucket",
            "run_id": self.run_id,
            "label": self.label,
            "bucket_index": self.bucket_index,
            "final": bool(final),
            "t_start_s": self.last_emit_s - self.started_s,
            "t_end_s": now - self.started_s,
            "elapsed_s": elapsed_s,
            "prompt_tokens": self.prompt_tokens,
            "max_tokens": self.max_tokens,
            "generated_tokens_total": int(totals["generated_tokens"]),
            "generated_tokens_delta": generated_delta,
            "tok_s_delta": generated_delta / elapsed_s if elapsed_s > 0 else None,
            "context_len": self.prompt_tokens + int(totals["generated_tokens"]),
            "speculative_depth": self.speculative_depth,
            "verify_calls_total": int(totals["verify_calls"]),
            "verify_calls_delta": verify_calls_delta,
            "accepted_drafts_total": int(totals["accepted_drafts"]),
            "accepted_drafts_delta": accepted_drafts_delta,
            "drafted_tokens_total": int(totals["drafted_tokens"]),
            "drafted_tokens_delta": drafted_tokens_delta,
            "accepted_per_verify_delta": (
                accepted_drafts_delta / verify_calls_delta
                if verify_calls_delta
                else None
            ),
            "draft_acceptance_rate_delta": (
                accepted_drafts_delta / drafted_tokens_delta
                if drafted_tokens_delta
                else None
            ),
            "accepted_by_depth_total": [
                int(item) for item in totals["accepted_by_depth"]
            ],
            "accepted_by_depth_delta": accepted_by_depth_delta,
            "drafted_by_depth_total": [
                int(item) for item in totals["drafted_by_depth"]
            ],
            "drafted_by_depth_delta": drafted_by_depth_delta,
            "acceptance_rate_by_depth_delta": acceptance_rate_by_depth_delta,
            "mean_accept_probability_by_depth_delta": mean_accept_probability_by_depth_delta,
            "rejected_drafts_delta": int(self._delta(totals, "rejected_drafts")),
            "correction_tokens_delta": int(self._delta(totals, "correction_tokens")),
            "bonus_tokens_delta": int(self._delta(totals, "bonus_tokens")),
            "verify_time_s_delta": verify_time_delta,
            "verify_forward_time_s_delta": verify_forward_time_delta,
            "verify_eval_time_s_delta": verify_eval_time_delta,
            "verify_logits_eval_time_s_delta": verify_logits_eval_time_delta,
            "verify_hidden_eval_time_s_delta": verify_hidden_eval_time_delta,
            "verify_joint_eval_time_s_delta": verify_joint_eval_time_delta,
            "verify_target_distribution_time_s_delta": verify_target_distribution_time_delta,
            "target_distribution_materialized_rows_delta": target_distribution_rows_delta,
            "target_distribution_materialized_windows_delta": target_distribution_windows_delta,
            "target_distribution_rows_per_window_delta": (
                target_distribution_rows_delta / target_distribution_windows_delta
                if target_distribution_windows_delta
                else None
            ),
            "verify_target_distribution_ms_per_row_delta": (
                1000.0
                * verify_target_distribution_time_delta
                / target_distribution_rows_delta
                if target_distribution_rows_delta
                else None
            ),
            "lazy_bonus_verify_calls_delta": lazy_bonus_verify_calls_delta,
            "lazy_bonus_commit_time_s_delta": lazy_bonus_commit_time_delta,
            "lazy_bonus_commit_ms_per_call_delta": (
                1000.0 * lazy_bonus_commit_time_delta / lazy_bonus_verify_calls_delta
                if lazy_bonus_verify_calls_delta
                else None
            ),
            "verify_eval_unattributed_time_s_delta": verify_eval_unattributed_time_delta,
            "draft_time_s_delta": draft_time_delta,
            "accept_time_s_delta": float(self._delta(totals, "accept_time_s")),
            "repair_time_s_delta": float(self._delta(totals, "repair_time_s")),
            "commit_time_s_delta": float(self._delta(totals, "commit_time_s")),
            "capture_commit_time_s_delta": float(
                self._delta(totals, "capture_commit_time_s")
            ),
            "snapshot_time_s_delta": float(self._delta(totals, "snapshot_time_s")),
            "bonus_time_s_delta": float(self._delta(totals, "bonus_time_s")),
            "verify_ms_per_call_delta": (
                1000.0 * verify_time_delta / verify_calls_delta
                if verify_calls_delta
                else None
            ),
            "verify_forward_ms_per_call_delta": (
                1000.0 * verify_forward_time_delta / verify_calls_delta
                if verify_calls_delta
                else None
            ),
            "verify_eval_ms_per_call_delta": (
                1000.0 * verify_eval_time_delta / verify_calls_delta
                if verify_calls_delta
                else None
            ),
            "verify_logits_eval_ms_per_call_delta": (
                1000.0 * verify_logits_eval_time_delta / verify_calls_delta
                if verify_calls_delta
                else None
            ),
            "verify_hidden_eval_ms_per_call_delta": (
                1000.0 * verify_hidden_eval_time_delta / verify_calls_delta
                if verify_calls_delta
                else None
            ),
            "verify_joint_eval_ms_per_call_delta": (
                1000.0 * verify_joint_eval_time_delta / verify_calls_delta
                if verify_calls_delta
                else None
            ),
            "verify_target_distribution_ms_per_call_delta": (
                1000.0 * verify_target_distribution_time_delta / verify_calls_delta
                if verify_calls_delta
                else None
            ),
            "verify_eval_unattributed_ms_per_call_delta": (
                1000.0 * verify_eval_unattributed_time_delta / verify_calls_delta
                if verify_calls_delta
                else None
            ),
            "draft_ms_per_token_delta": (
                1000.0 * draft_time_delta / drafted_tokens_delta
                if drafted_tokens_delta
                else None
            ),
            **bytes_delta,
            "estimated_materialized_nbytes_delta": materialized_nbytes,
            "estimated_materialized_gib_s": (
                (materialized_nbytes / (1024**3)) / elapsed_s if elapsed_s > 0 else None
            ),
            "cache_state_nbytes": _tree_nbytes(cache),
            "mtp_cache_state_nbytes": _tree_nbytes(mtp_cache),
            "mlx_memory": _mlx_memory_stats(),
            "lazy_verify_logits": _env_truthy("MTPLX_LAZY_VERIFY_LOGITS"),
            "defer_verify_hidden_eval": _defer_verify_hidden_eval_enabled(),
            "verify_hidden_mode": _verify_hidden_mode(),
            "split_verify_eval": _env_truthy("MTPLX_SPLIT_VERIFY_EVAL"),
            "lazy_mtp_history_append": _env_truthy("MTPLX_LAZY_MTP_HISTORY_APPEND"),
            "batch_target_arrays": _batch_target_arrays_enabled(),
            "drop_events": _env_truthy("MTPLX_DROP_EVENTS"),
            "skip_verify_snapshot": _skip_verify_snapshot(),
            "mtp_history_materialize_every": int(mtp_history_materialize_every),
            "mtp_history_materialize_events": int(mtp_history_materialize_events),
            "clear_cache_every": int(_clear_cache_every()),
            "clear_cache_events_total": int(totals["clear_cache_events"]),
            "clear_cache_events_delta": clear_cache_events_delta,
            "clear_cache_time_s_total": float(totals["clear_cache_time_s"]),
            "clear_cache_time_s_delta": clear_cache_time_delta,
            "trunk_cache_materialize_every": int(
                os.environ.get("MTPLX_TRUNK_CACHE_MATERIALIZE_EVERY") or 0
            ),
            "trunk_cache_materialize_events_total": int(
                totals["trunk_cache_materialize_events"]
            ),
            "trunk_cache_materialize_events_delta": trunk_cache_materialize_events_delta,
            "trunk_cache_materialize_time_s_total": float(
                totals["trunk_cache_materialize_time_s"]
            ),
            "trunk_cache_materialize_time_s_delta": trunk_cache_materialize_time_delta,
            "dirty_detach_components": os.environ.get("MTPLX_DETACH_COMPONENTS"),
            "dirty_detach_mode": os.environ.get("MTPLX_DETACH_MODE"),
            "dirty_detach_gdn_every": int(
                os.environ.get("MTPLX_DETACH_GDN_EVERY") or 0
            ),
            "dirty_detach_conv_every": int(
                os.environ.get("MTPLX_DETACH_CONV_EVERY") or 0
            ),
            "dirty_detach_attn_every": int(
                os.environ.get("MTPLX_DETACH_ATTN_EVERY") or 0
            ),
            "dirty_detach_events_total": int(totals["dirty_detach_events"]),
            "dirty_detach_events_delta": dirty_detach_events_delta,
            "dirty_detach_time_s_total": float(totals["dirty_detach_time_s"]),
            "dirty_detach_time_s_delta": dirty_detach_time_delta,
            "dirty_detach_arrays_total": int(totals["dirty_detach_arrays"]),
            "dirty_detach_arrays_delta": dirty_detach_arrays_delta,
            "dirty_detach_bytes_total": int(totals["dirty_detach_bytes"]),
            "dirty_detach_bytes_delta": dirty_detach_bytes_delta,
            "live_output_detach_enabled": bool(
                os.environ.get("MTPLX_DETACH_LIVE_OUTPUTS")
            ),
            "live_output_detach_mode": os.environ.get("MTPLX_DETACH_LIVE_OUTPUTS_MODE"),
            "live_output_detach_events_total": int(totals["live_output_detach_events"]),
            "live_output_detach_events_delta": live_output_detach_events_delta,
            "live_output_detach_time_s_total": float(
                totals["live_output_detach_time_s"]
            ),
            "live_output_detach_time_s_delta": live_output_detach_time_delta,
            "live_output_detach_arrays_total": int(totals["live_output_detach_arrays"]),
            "live_output_detach_arrays_delta": live_output_detach_arrays_delta,
            "live_output_detach_bytes_total": int(totals["live_output_detach_bytes"]),
            "live_output_detach_bytes_delta": live_output_detach_bytes_delta,
            "state_rebase_every": int(os.environ.get("MTPLX_STATE_REBASE_EVERY") or 0),
            "state_rebase_events_total": int(totals["state_rebase_events"]),
            "state_rebase_events_delta": state_rebase_events_delta,
            "state_rebase_time_s_total": float(totals["state_rebase_time_s"]),
            "state_rebase_time_s_delta": state_rebase_time_delta,
            "state_root_eval_enabled": bool(
                os.environ.get("MTPLX_EVAL_STATE_ROOTS_ON_COMMIT")
            ),
            "state_root_eval_include_mtp": bool(
                os.environ.get("MTPLX_EVAL_STATE_ROOTS_INCLUDE_MTP", "1")
                .strip()
                .lower()
                not in {"0", "false", "no", "off"}
            ),
            "state_root_eval_events_total": int(totals["state_root_eval_events"]),
            "state_root_eval_events_delta": state_root_eval_events_delta,
            "state_root_eval_time_s_total": float(totals["state_root_eval_time_s"]),
            "state_root_eval_time_s_delta": state_root_eval_time_delta,
            "state_root_eval_arrays_total": int(totals["state_root_eval_arrays"]),
            "state_root_eval_arrays_delta": state_root_eval_arrays_delta,
            "trace_accounting_time_s_total": float(totals["trace_accounting_time_s"]),
            "trace_accounting_time_s_delta": trace_accounting_time_delta,
            "verify_strategy": self.verify_strategy,
            "verify_core": self.verify_core,
            "mtp_history_policy": self.mtp_history_policy,
            "mtp_cache_policy": self.mtp_cache_policy,
            "sampler": {
                "temperature": float(self.sampler.temperature),
                "top_p": float(self.sampler.top_p),
                "top_k": int(self.sampler.top_k)
                if self.sampler.top_k is not None
                else None,
            },
            "metadata": self.metadata,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        self.bucket_index += 1
        self.last_emit_s = now
        self.last_totals = {
            key: (list(value) if isinstance(value, list) else value)
            for key, value in totals.items()
        }


_AR_FORWARD_PROFILE: Any = None


def _ar_forward_profiler(step: int) -> Any:
    """Diagnostic lane: MTPLX_AR_PROFILE_TOKENS=N cProfiles decode forwards
    for steps [8, 8+N) and dumps pstats to MTPLX_AR_PROFILE_PATH at the
    last profiled step. Off (None) unless the env is set; throughput
    measured with this enabled is not promotion evidence."""

    global _AR_FORWARD_PROFILE
    raw = os.environ.get("MTPLX_AR_PROFILE_TOKENS")
    if not raw:
        return None
    try:
        budget = int(raw)
    except ValueError:
        return None
    first, last = 8, 8 + budget
    if not first <= step < last:
        if step == last and _AR_FORWARD_PROFILE is not None:
            import pstats

            path = os.environ.get(
                "MTPLX_AR_PROFILE_PATH", "/tmp/mtplx-ar-forward.pstats"
            )
            pstats.Stats(_AR_FORWARD_PROFILE).dump_stats(path)
            _AR_FORWARD_PROFILE = None
        return None
    if _AR_FORWARD_PROFILE is None:
        import cProfile

        _AR_FORWARD_PROFILE = cProfile.Profile()
    return _AR_FORWARD_PROFILE


def _batch_target_distributions_enabled() -> bool:
    return os.environ.get("MTPLX_BATCH_TARGET_DISTS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _batch_target_arrays_enabled() -> bool:
    return os.environ.get("MTPLX_BATCH_TARGET_ARRAYS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _lazy_target_distributions_enabled() -> bool:
    return _env_truthy("MTPLX_LAZY_TARGET_DISTRIBUTIONS")


def _lazy_bonus_verify_enabled() -> bool:
    return _env_truthy("MTPLX_LAZY_BONUS_VERIFY")


def _lazy_bonus_verify_min_depth() -> int:
    raw = os.environ.get("MTPLX_LAZY_BONUS_VERIFY_MIN_DEPTH")
    if raw is None or raw.strip() == "":
        return 2
    try:
        return max(1, int(raw))
    except ValueError:
        return 2


def _omit_speculative_bonus_enabled() -> bool:
    return _env_truthy("MTPLX_OMIT_SPECULATIVE_BONUS")


@dataclass
class GenerationStats:
    mode: Mode
    generated_tokens: int
    elapsed_s: float
    tok_s: float
    decode_elapsed_s: float = 0.0
    decode_tok_s: float = 0.0
    end_to_end_tok_s: float = 0.0
    benchmark_mode: str | None = None
    load_mtp: bool | None = None
    runtime_mtp_enabled: bool = False
    draft_head_installed: bool | None = None
    ar_return_hidden: bool = False
    forward_ar_hidden_calls: int = 0
    forward_ar_plain_calls: int = 0
    mtp_forward_calls: int = 0
    make_mtp_cache_calls: int = 0
    update_mtp_cache_calls: int = 0
    mtp_history_append_calls: int = 0
    full_logits_tokens_emitted: int = 0
    final_logits_tokens_emitted: int = 0
    logits_tokens_emitted: int = 0
    prefill_chunk_size: int = 0
    prefill_chunks: int = 0
    prefill_chunk_cache_cleanup_enabled: bool = False
    prefill_chunk_cache_cleanup_every: int = 1
    prefill_chunk_cache_cleanup_events: int = 0
    prefill_stock_cache_only_enabled: bool = False
    prefill_stock_cache_only_calls: int = 0
    prefill_omlx_external_enabled: bool = False
    prefill_omlx_external_calls: int = 0
    prefill_external_emit_logits_enabled: bool = True
    prefill_external_cache_only_calls: int = 0
    paged_kv_capacity_tokens: int = 0
    paged_kv_num_blocks: int = 0
    paged_active_array_calls: int = 0
    paged_active_array_time_s: float = 0.0
    paged_turboquant: bool = False
    paged_turboquant_k_quant: str = ""
    paged_turboquant_v_quant: str = ""
    paged_turboquant_attention_calls: int = 0
    paged_kv_quant: bool = False
    paged_kv_quant_mode: str = ""
    paged_kv_quant_attention_calls: int = 0
    paged_kv_quant_dequant_calls: int = 0
    paged_kv_quant_dequant_time_s: float = 0.0
    paged_kv_quant_dequant_tokens: int = 0
    paged_gqa_sdpa_calls: int = 0
    paged_gqa_sdpa_calls_by_route: dict[str, int] = field(default_factory=dict)
    paged_gqa_sdpa_calls_by_phase: dict[str, int] = field(default_factory=dict)
    paged_gqa_sdpa_route_misses_by_phase_reason: dict[str, int] = field(
        default_factory=dict
    )
    paged_gqa_sdpa_route_misses_by_q_len: dict[str, int] = field(default_factory=dict)
    paged_gqa_sdpa_last_route_miss: dict[str, object] = field(default_factory=dict)
    attention_dense_fallback_calls: int = 0
    prefill_dense_fallback_calls: int = 0
    decode_dense_fallback_calls: int = 0
    ar_dense_fallback_calls: int = 0
    postcommit_dense_fallback_calls: int = 0
    paged_attention_bailouts_by_phase_reason: dict[str, int] = field(
        default_factory=dict
    )
    paged_attention_large_q_path: str = ""
    prefill_route: str = ""
    large_q_split_sdpa_fallback_calls: int = 0
    large_q_split_sdpa_fallback_calls_by_phase: dict[str, int] = field(
        default_factory=dict
    )
    prefill_large_q_split_sdpa_fallback_calls: int = 0
    decode_large_q_split_sdpa_fallback_calls: int = 0
    partitioned_paged_calls: int = 0
    partitioned_paged_calls_by_phase: dict[str, int] = field(default_factory=dict)
    prefill_partitioned_paged_calls: int = 0
    decode_partitioned_paged_calls: int = 0
    sessionbank_snapshot_bytes: int = 0
    sessionbank_skipped_oversized_snapshot: bool = False
    session_prompt_prefix_bank_commit: dict[str, object] = field(default_factory=dict)
    # Store-on-prefill telemetry ({} when the store did not run) and the
    # restore-return -> first-decode-iteration span. The span includes the
    # prompt-prefix bank commit plus graph/policy construction — it is setup
    # wall time that decode_elapsed_s already contains, NOT pure decode.
    session_prefill_store: dict[str, object] = field(default_factory=dict)
    pre_first_token_setup_s: float = 0.0
    # Passive probe (2026-08-06): served-entry truth, prompt-state wall
    # decomposition, first-primary-sample latency, and round-1 snapshots of
    # the existing cumulative timers. Observational only — no metric above
    # is redefined and no evaluation point moves.
    session_restore_served: dict[str, object] = field(default_factory=dict)
    prompt_state_total_time_s: float = 0.0
    prompt_state_unattributed_time_s: float = 0.0
    first_primary_sample_time_s: float = 0.0
    first_round: dict[str, object] = field(default_factory=dict)
    accepted_drafts: int = 0
    rejected_drafts: int = 0
    drafted_tokens: int = 0
    verify_time_s: float = 0.0
    verify_forward_time_s: float = 0.0
    verify_eval_time_s: float = 0.0
    verify_logits_eval_time_s: float = 0.0
    verify_hidden_eval_time_s: float = 0.0
    verify_joint_eval_time_s: float = 0.0
    verify_target_distribution_time_s: float = 0.0
    target_distribution_materialized_rows: int = 0
    target_distribution_materialized_windows: int = 0
    target_distribution_share: float = 0.0
    lazy_bonus_verify_calls: int = 0
    lazy_bonus_commit_time_s: float = 0.0
    verify_eval_unattributed_time_s: float = 0.0
    verify_hidden_mode: str = "default"
    draft_time_s: float = 0.0
    target_forward_time_s: float = 0.0
    prompt_eval_time_s: float = 0.0
    prompt_tps: float = 0.0
    prompt_target_prefill_time_s: float = 0.0
    prompt_mtp_history_time_s: float = 0.0
    prompt_target_prefill_tok_s: float = 0.0
    prompt_mtp_history_tok_s: float = 0.0
    cache_restore_time_s: float = 0.0
    mtp_history_policy: str = "cycle"
    mtp_history_window_tokens: int = 0
    mtp_history_position_base: int = 0
    cached_tokens: int = 0
    new_prefill_tokens: int = 0
    session_cache_hit: bool = False
    cache_source: str = "none"
    ssd_cache_hit: bool = False
    ssd_cached_tokens: int = 0
    ssd_restore_s: float = 0.0
    ssd_suffix_tokens: int = 0
    cache_miss_reason: str | None = None
    session_restore_mode: str = "cold"
    snapshot_time_s: float = 0.0
    accept_time_s: float = 0.0
    rollback_time_s: float = 0.0
    repair_time_s: float = 0.0
    commit_time_s: float = 0.0
    capture_commit_time_s: float = 0.0
    mtp_history_materialize_every: int = 0
    mtp_history_materialize_events: int = 0
    clear_cache_every: int = 0
    clear_cache_events: int = 0
    clear_cache_time_s: float = 0.0
    trunk_cache_materialize_every: int = 0
    trunk_cache_materialize_events: int = 0
    trunk_cache_materialize_time_s: float = 0.0
    dirty_detach_components: list[str] = field(default_factory=list)
    dirty_detach_mode: str = "selected_slice_contiguous_eval"
    dirty_detach_gdn_every: int = 0
    dirty_detach_conv_every: int = 0
    dirty_detach_attn_every: int = 0
    dirty_detach_events: int = 0
    dirty_detach_time_s: float = 0.0
    dirty_detach_arrays: int = 0
    dirty_detach_bytes: int = 0
    live_output_detach_enabled: bool = False
    live_output_detach_mode: str = "contiguous_eval"
    live_output_detach_events: int = 0
    live_output_detach_time_s: float = 0.0
    live_output_detach_arrays: int = 0
    live_output_detach_bytes: int = 0
    state_rebase_every: int = 0
    state_rebase_events: int = 0
    state_rebase_time_s: float = 0.0
    state_root_eval_enabled: bool = False
    state_root_eval_include_mtp: bool = True
    state_root_eval_events: int = 0
    state_root_eval_time_s: float = 0.0
    state_root_eval_arrays: int = 0
    capture_commit_detach_components: list[str] = field(default_factory=list)
    capture_commit_detach_mode: str = "selected_slice_contiguous_eval"
    capture_commit_detach_gdn_every: int = 0
    capture_commit_detach_conv_every: int = 0
    capture_commit_detach_events: int = 0
    capture_commit_detach_time_s: float = 0.0
    capture_commit_detach_arrays: int = 0
    capture_commit_detach_bytes: int = 0
    trace_accounting_time_s: float = 0.0
    decode_trace_path: str | None = None
    decode_trace_run_id: str | None = None
    bonus_time_s: float = 0.0
    online_hidden_corrector_time_s: float = 0.0
    peak_memory_bytes: int = 0
    speculative_depth: int = 0
    requested_speculative_depth: int = 0
    long_context_mtp_depth_policy: dict[str, object] = field(default_factory=dict)
    accepted_by_depth: list[int] = field(default_factory=list)
    drafted_by_depth: list[int] = field(default_factory=list)
    accept_probability_sum_by_depth: list[float] = field(default_factory=list)
    mean_accept_probability_by_depth: list[float | None] = field(default_factory=list)
    skipped_drafts: int = 0
    bonus_tokens: int = 0
    correction_tokens: int = 0
    verify_calls: int = 0
    # Context-copy (prompt-lookup) drafting. Counters are cumulative per
    # generation; the per-round detail stays in events. accepted_tokens counts
    # verified matches (_cc_nacc), which can exceed emitted tokens when a stop
    # token truncates the accepted block. suspended/backoff_tokens are gauges
    # of the end-of-generation state, suspensions counts entries into backoff.
    context_copy_active: bool = False
    context_copy_probes: int = 0
    context_copy_rounds: int = 0
    context_copy_drafted_tokens: int = 0
    context_copy_accepted_blocks: int = 0
    context_copy_accepted_tokens: int = 0
    context_copy_suspensions: int = 0
    context_copy_suspended: bool = False
    context_copy_backoff_tokens: int = 0
    context_copy_disabled_reason: str | None = None
    # Grammar-constrained decoding (response_format). constraint_completed is
    # None when no constraint was active, False when generation ended before
    # the grammar reached a complete document (truncation is never passed off
    # as valid output).
    constraint_active: bool = False
    constraint_completed: bool | None = None
    constraint_masked_steps: int = 0
    constraint_mask_time_s: float = 0.0
    graphbank: dict[str, object] = field(default_factory=dict)
    reject_path_counts: dict[str, int] = field(default_factory=dict)
    repair_time_by_reject_depth_s: dict[str, float] = field(default_factory=dict)
    deferred_correction_repairs: int = 0
    online_correction_cache: dict[str, object] = field(default_factory=dict)
    adapter_ensemble_q: dict[str, object] = field(default_factory=dict)
    mtp_topk_reranker: dict[str, object] = field(default_factory=dict)
    draft_core: dict[str, object] = field(default_factory=dict)
    owned_recurrent_state: dict[str, object] = field(default_factory=dict)
    owned_attn_kv: dict[str, object] = field(default_factory=dict)
    repetition_stop_triggered: bool = False
    repetition_stop_reason: str | None = None
    repetition_stop_block_tokens: int = 0
    repetition_stop_repeats: int = 0
    repetition_stop_trimmed_tokens: int = 0
    repetition_stop_raw_tokens: int = 0
    loop_guard: dict[str, object] = field(default_factory=dict)
    thinking_guard: dict[str, object] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenerationOutput:
    tokens: list[int]
    text: str
    stats: GenerationStats
    final_state: GenerationFinalState | None = None
    finish_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "tokens": self.tokens,
            "text": self.text,
            "stats": self.stats.to_dict(),
            "finish_reason": self.finish_reason,
        }


@dataclass(frozen=True)
class RepetitionStopConfig:
    enabled: bool = False
    min_tokens: int = 768
    min_repeated_tokens: int = 192
    min_repeats: int = 4
    min_block_tokens: int = 1
    max_block_tokens: int = 96


@dataclass(frozen=True)
class RepetitionStopResult:
    trim_start: int
    block_tokens: int
    repeats: int
    repeated_tokens: int


def _repetition_stop_config(enabled: bool) -> RepetitionStopConfig:
    if not enabled:
        return RepetitionStopConfig(enabled=False)
    min_tokens = max(1, _env_int("MTPLX_REPETITION_STOP_MIN_TOKENS", 768))
    min_repeated_tokens = max(
        1,
        _env_int("MTPLX_REPETITION_STOP_MIN_REPEATED_TOKENS", 192),
    )
    min_repeats = max(2, _env_int("MTPLX_REPETITION_STOP_MIN_REPEATS", 4))
    min_block_tokens = max(1, _env_int("MTPLX_REPETITION_STOP_MIN_BLOCK_TOKENS", 1))
    max_block_tokens = max(
        min_block_tokens,
        _env_int("MTPLX_REPETITION_STOP_MAX_BLOCK_TOKENS", 96),
    )
    return RepetitionStopConfig(
        enabled=True,
        min_tokens=min_tokens,
        min_repeated_tokens=min_repeated_tokens,
        min_repeats=min_repeats,
        min_block_tokens=min_block_tokens,
        max_block_tokens=max_block_tokens,
    )


@dataclass
class PromptState:
    trunk_cache: list[Any]
    logits: Any
    hidden: Any | None
    committed_mtp_cache: Any | None
    token_prefix: tuple[int, ...]
    prompt_eval_time_s: float
    prompt_mtp_history_time_s: float = 0.0
    cache_restore_time_s: float = 0.0
    mtp_history_policy: str = "cycle"
    mtp_history_window_tokens: int = 0
    mtp_history_position_base: int = 0
    cached_tokens: int = 0
    suffix_tokens: int = 0
    cache_hit: bool = False
    cache_source: str = "none"
    ssd_cache_hit: bool = False
    ssd_cached_tokens: int = 0
    ssd_restore_s: float = 0.0
    cache_miss_reason: str | None = None
    restore_mode: str = "cold"
    # kvcache-v2: recurrent-only interior snapshots captured during cold
    # chunked prefill — (token_count, CacheSnapshot) ascending. Threaded into
    # SessionBank.put so sub-prefix restores can land on a recurrent-true
    # boundary instead of reusing recurrent state from the stored end.
    gdn_boundaries: list = field(default_factory=list)
    # Telemetry only: elapsed/split timings when the store-on-prefill
    # snapshot ran for this prompt state ({} when it did not run). This
    # store executes outside the prompt_eval_time_s window, so without a
    # timer its wall time is unattributable in per-request telemetry.
    prefill_store_snapshot: dict = field(default_factory=dict)
    # Passive probe: the entry actually SERVED by a bank restore for this
    # prompt state ({} on cold paths). Resolution diagnostics record
    # matches[0] before generation may skip it on achievable-boundary
    # checks, so served truth is recorded where the restore succeeds.
    restore_served: dict = field(default_factory=dict)


class PostcommitAbort(RuntimeError):
    """Raised when best-effort postcommit prefill yields to foreground work."""


def _check_postcommit_abort(abort_check: Callable[[], bool] | None) -> None:
    if abort_check is not None and bool(abort_check()):
        raise PostcommitAbort("foreground_preempted_postcommit")


@dataclass
class GenerationFinalState:
    final_trunk_cache: list[Any]
    final_logits: Any
    final_hidden: Any | None
    final_committed_mtp_cache: Any | None
    generated_token_ids: tuple[int, ...]
    safe_to_commit: bool
    finish_reason: str
    mtp_history_policy: str = "cycle"
    mtp_history_window_tokens: int = 0
    mtp_history_position_base: int = 0
    extra_state: dict[str, Any] | None = None


def _finish_reason_from_tokens(
    tokens: list[int],
    *,
    stop_token_ids: set[int],
    max_tokens: int,
) -> str:
    if any(_is_stop(token, stop_token_ids) for token in tokens):
        return "stop"
    return "length" if len(tokens) >= max_tokens else "stop"


def _detect_repeated_token_suffix(
    tokens: list[int],
    config: RepetitionStopConfig,
) -> RepetitionStopResult | None:
    if not config.enabled:
        return None
    token_count = len(tokens)
    if token_count < max(1, int(config.min_tokens)):
        return None
    max_block = min(
        max(1, int(config.max_block_tokens)),
        token_count // max(1, int(config.min_repeats)),
    )
    min_block = max(1, int(config.min_block_tokens))
    if max_block < min_block:
        return None
    best: RepetitionStopResult | None = None
    for block_tokens in range(min_block, max_block + 1):
        block = tokens[token_count - block_tokens : token_count]
        repeats = 1
        cursor = token_count - block_tokens
        while (
            cursor >= block_tokens
            and tokens[cursor - block_tokens : cursor] == block
        ):
            repeats += 1
            cursor -= block_tokens
        repeated_tokens = repeats * block_tokens
        if repeats < int(config.min_repeats):
            continue
        if repeated_tokens < int(config.min_repeated_tokens):
            continue
        candidate = RepetitionStopResult(
            trim_start=token_count - repeated_tokens,
            block_tokens=block_tokens,
            repeats=repeats,
            repeated_tokens=repeated_tokens,
        )
        if best is None or candidate.repeated_tokens > best.repeated_tokens:
            best = candidate
    return best


def _trim_repeated_suffix(
    tokens: list[int],
    config: RepetitionStopConfig,
) -> RepetitionStopResult | None:
    result = _detect_repeated_token_suffix(tokens, config)
    if result is None:
        return None
    del tokens[result.trim_start :]
    return result


def _prefill_restored_prompt_suffix(
    rt: MTPLXRuntime,
    restored: Any,
    suffix: list[int],
    *,
    base_hidden_variant: str,
    mtp_hidden_variant: str,
    mtp_history_policy: str,
    abort_check: Callable[[], bool] | None = None,
    chunk_callback: Callable[[dict[str, Any]], None] | None = None,
    tokens_total: int | None = None,
    cached_tokens: int = 0,
    chunk_started_s: float | None = None,
    gdn_boundary_sink: list[tuple[int, Any, Any]] | None = None,
    vision_splice: Any | None = None,
    stable_prefix_len: int | None = None,
) -> tuple[Any, Any, float, float]:
    """Extend a restored SessionBank prefix without one giant suffix forward.

    The old warm-prefix path sent the entire suffix through `forward_ar` with
    hidden capture and logits enabled. In OpenCode sessions that made a stale
    postcommit suffix behave like a full long-context prefill and, worse, it
    could not observe abort requests until the huge Metal graph completed. This
    mirrors the oMLX-style prefill shape: cache-only or hidden-only chunks for
    the body, then a single final-token logits/hidden pass for decode startup.
    """

    if not suffix:
        raise ValueError("suffix must not be empty")
    _check_postcommit_abort(abort_check)
    target_forward_time = 0.0
    mtp_history_time = 0.0
    final_logits_only = _final_logits_prefill_enabled()
    tokens_total = int(tokens_total if tokens_total is not None else len(suffix))
    cached_tokens = max(0, int(cached_tokens))
    suffix_total = int(len(suffix))
    suffix_done = 0
    # A committed history needs a draft head to append to. Requiring
    # rt.mtp_enabled here is the chokepoint that keeps the hidden-only chunk
    # branch (and every append_history call) off target-only AR runtimes, whose
    # forward_ar returns logits alone — restore_or_prefill_prompt_state already
    # downgrades those to the cycle policy, and _append_mtp_history could not
    # run against them regardless.
    use_committed_mtp = (
        rt.mtp_enabled
        and _mtp_history_uses_committed_cache(mtp_history_policy)
        and restored.mtp_history_cache is not None
    )
    # Vision suffixes: the caller pre-advanced the cursor past pads inside
    # the restored prefix; trunk chunks consume the remaining rows
    # sequentially, history windows read the same rows cursor-free.
    splice_initial_cursor = (
        int(vision_splice.cursor) if vision_splice is not None else 0
    )

    def _suffix_chunk_embeddings(chunk_array: Any) -> Any | None:
        if vision_splice is None:
            return None
        from mtplx.vision.splice import spliced_chunk_embeddings

        return spliced_chunk_embeddings(rt.embed_tokens, chunk_array, vision_splice)

    def _history_window_embeddings(
        token_ids: list[int], window_start: int
    ) -> Any | None:
        if vision_splice is None or not token_ids:
            return None
        pad_id = vision_splice.image_pad_token_id
        if not any(token == pad_id for token in token_ids):
            return None
        from mtplx.vision.splice import spliced_embeddings_for_window

        rows_before = splice_initial_cursor + sum(
            1 for token in suffix[:window_start] if token == pad_id
        )
        return spliced_embeddings_for_window(
            rt.embed_tokens,
            mx.array([token_ids]),
            vision_splice,
            rows_before=rows_before,
        )

    def _check_splice_consumed() -> None:
        if vision_splice is not None and vision_splice.remaining() > 0:
            raise ValueError(
                "vision splice overflow: restored-suffix prefill left "
                f"{vision_splice.remaining()} unconsumed vision rows"
            )

    def emit_chunk(chunk_len: int, chunk_elapsed: float, started: float) -> None:
        if chunk_callback is None:
            return
        try:
            phase_start = chunk_started_s if chunk_started_s is not None else started
            elapsed = max(0.0, time.perf_counter() - phase_start)
            new_done = max(0, min(suffix_total, suffix_done))
            tokens_done = max(0, min(tokens_total, cached_tokens + new_done))
            chunk_tok_s = (
                float(chunk_len) / chunk_elapsed
                if chunk_elapsed > 0.0 and chunk_len > 0
                else None
            )
            cumulative_tok_s = (
                float(new_done) / elapsed
                if elapsed > 0.0 and new_done > 0
                else None
            )
            chunk_callback(
                {
                    "phase": "chunk",
                    "tokens_done": tokens_done,
                    "tokens_total": tokens_total,
                    "cached_tokens": cached_tokens,
                    "new_prefill_tokens": suffix_total,
                    "elapsed_s": elapsed,
                    "prefill_tok_s": cumulative_tok_s,
                    "cumulative_prefill_tok_s": cumulative_tok_s,
                    "prefill_wall_tok_s": cumulative_tok_s,
                    "live_prefill_tok_s": (
                        chunk_tok_s if chunk_tok_s is not None else cumulative_tok_s
                    ),
                    "chunk_size": int(chunk_len),
                    "chunk_elapsed_s": chunk_elapsed,
                    "chunk_prefill_tok_s": chunk_tok_s,
                }
            )
        except Exception:
            pass

    def append_history(
        hidden_states: Any, token_ids: list[int], window_start: int = 0
    ) -> None:
        nonlocal mtp_history_time
        if not use_committed_mtp or not token_ids:
            return
        mtp_history_time += _append_mtp_history(
            rt,
            restored.mtp_history_cache,
            hidden_states,
            token_ids,
            phase="prefill",
            mtp_hidden_variant=mtp_hidden_variant,
            force_eval=True,
            input_embeddings=_history_window_embeddings(token_ids, window_start),
        )
        _check_postcommit_abort(abort_check)

    if use_committed_mtp and restored.hidden is not None:
        append_history(restored.hidden, [int(suffix[0])], window_start=0)

    # kvcache-v2 small-suffix fast path: warm restores usually leave a tail of
    # tens-to-hundreds of tokens, and the chunked body/final split plus its
    # per-chunk forced evals costs more than the math (measured 154-524 tok/s
    # on 33-199-token suffixes at 4k-48k). One fused forward with final-only
    # logits does the same work with two eval barriers total. Large suffixes
    # keep the chunked path for abort responsiveness.
    # A stable prompt-prefix edge inside the suffix must become a chunk
    # boundary so the gdn capture records recurrent state exactly there —
    # the fused single-forward cannot capture interior boundaries, so it
    # defers to the chunked path in that case (same tokens, one extra
    # launch; no re-evaluation).
    _stable_edge_rel: int | None = None
    if (
        stable_prefix_len is not None
        and gdn_boundary_sink is not None
        and 0 < int(stable_prefix_len) - int(cached_tokens) < max(0, len(suffix) - 1)
    ):
        _stable_edge_rel = int(stable_prefix_len) - int(cached_tokens)
    fused_max = _small_suffix_fused_max()
    if 0 < len(suffix) <= fused_max and _stable_edge_rel is None:
        fused_array = mx.array([suffix])
        fused_embeddings = _suffix_chunk_embeddings(fused_array)
        started = time.perf_counter()
        with attention_phase("prefill"):
            suffix_logits, suffix_hidden = _forward_ar_optional_hidden(
                rt,
                fused_array,
                cache=restored.cache,
                hidden_variant=base_hidden_variant,
                emit_logits=True,
                logits_keep=1 if final_logits_only else None,
                input_embeddings=fused_embeddings,
            )
        if suffix_hidden is None:
            _eval(suffix_logits)
        else:
            _eval(suffix_logits, suffix_hidden)
        chunk_elapsed = time.perf_counter() - started
        target_forward_time += chunk_elapsed
        _runtime_count(rt, "restored_suffix_prefill_fused")
        _runtime_count(rt, "prefill_chunks")
        suffix_done = suffix_total
        emit_chunk(suffix_total, chunk_elapsed, started)
        _check_postcommit_abort(abort_check)
        if len(suffix) > 1 and suffix_hidden is not None:
            append_history(
                suffix_hidden[:, :-1, :],
                [int(token) for token in suffix[1:]],
                window_start=1,
            )
        target_forward_time += _maybe_repage_target_prefill_cache(
            rt, restored.cache
        )
        _check_splice_consumed()
        return (
            suffix_logits[:, -1, :],
            suffix_hidden[:, -1:, :] if suffix_hidden is not None else None,
            target_forward_time,
            mtp_history_time,
        )

    capture_boundaries = (
        gdn_boundary_sink is not None
        and _cache_has_recurrent_entries(restored.cache)
    )
    if len(suffix) > 1:
        body = suffix[:-1]
        body_array = mx.array([body])
        spans = (
            _prefill_spans_with_tail_grid(
                len(body),
                tail_interval=_gdn_boundary_tail_interval(),
                mandatory_edges=(
                    (_stable_edge_rel,) if _stable_edge_rel is not None else ()
                ),
            )
            if capture_boundaries
            else _iter_prefill_chunk_spans(len(body))
        )
        for start, end in spans:
            _check_postcommit_abort(abort_check)
            chunk_array = body_array[:, start:end]
            chunk_embeddings = _suffix_chunk_embeddings(chunk_array)
            started = time.perf_counter()
            with attention_phase("prefill"):
                if use_committed_mtp:
                    logits_chunk, hidden_chunk = rt.forward_ar(
                        chunk_array,
                        cache=restored.cache,
                        return_hidden=True,
                        hidden_variant=base_hidden_variant,
                        emit_logits=False,
                        input_embeddings=chunk_embeddings,
                    )
                else:
                    hidden_chunk = None
                    logits_chunk = _prefill_cache_only_forward(
                        rt,
                        chunk_array,
                        restored.cache,
                        input_embeddings=chunk_embeddings,
                    )
            if hidden_chunk is None:
                if logits_chunk is None:
                    _eval_cache_roots(restored.cache)
                else:
                    _eval(logits_chunk)
            elif logits_chunk is None:
                _eval(hidden_chunk)
            else:
                _eval(logits_chunk, hidden_chunk)
            chunk_elapsed = time.perf_counter() - started
            target_forward_time += chunk_elapsed
            _runtime_count(rt, "restored_suffix_prefill_chunks")
            _runtime_count(rt, "prefill_chunks")
            suffix_done = min(suffix_total, end)
            emit_chunk(end - start, chunk_elapsed, started)
            _check_postcommit_abort(abort_check)

            if capture_boundaries:
                # Warm prefills must capture boundaries exactly like cold
                # ones (absolute positions; hidden of the chunk's last token
                # when committed-MTP hidden is available) — otherwise every
                # entry banked after a warm restore is boundary-free and the
                # boundary-true block restore fails closed on it.
                _capture_gdn_boundary(
                    gdn_boundary_sink,
                    cached_tokens + end,
                    restored.cache,
                    hidden_last=(
                        hidden_chunk[:, -1:, :]
                        if hidden_chunk is not None
                        else None
                    ),
                )
            if hidden_chunk is not None:
                append_history(
                    hidden_chunk,
                    [int(token) for token in suffix[start + 1 : end + 1]],
                    window_start=start + 1,
                )
            del hidden_chunk
            del logits_chunk
            target_forward_time += _prefill_chunk_cache_cleanup(rt)
            _check_postcommit_abort(abort_check)

    started = time.perf_counter()
    _check_postcommit_abort(abort_check)
    final_array = mx.array([[suffix[-1]]])
    final_embeddings = _suffix_chunk_embeddings(final_array)
    with attention_phase("prefill"):
        suffix_logits, suffix_hidden = _forward_ar_optional_hidden(
            rt,
            final_array,
            cache=restored.cache,
            hidden_variant=base_hidden_variant,
            emit_logits=True,
            logits_keep=1 if final_logits_only else None,
            input_embeddings=final_embeddings,
        )
    if suffix_hidden is None:
        _eval(suffix_logits)
    else:
        _eval(suffix_logits, suffix_hidden)
    chunk_elapsed = time.perf_counter() - started
    target_forward_time += chunk_elapsed
    target_forward_time += _maybe_repage_target_prefill_cache(rt, restored.cache)
    suffix_done = suffix_total
    emit_chunk(1, chunk_elapsed, started)
    _check_postcommit_abort(abort_check)
    _check_splice_consumed()
    return (
        suffix_logits[:, -1, :],
        suffix_hidden[:, -1:, :] if suffix_hidden is not None else None,
        target_forward_time,
        mtp_history_time,
    )


def _emit_prefill_restore_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    *,
    tokens_total: int,
    cached_tokens: int,
    new_prefill_tokens: int,
    started_s: float,
    cache_source: str | None = None,
    ssd_cache_hit: bool = False,
    ssd_cached_tokens: int = 0,
    ssd_restore_s: float = 0.0,
    ssd_suffix_tokens: int | None = None,
) -> None:
    """Publish useful prefill state immediately after a prefix restore."""

    if callback is None:
        return
    cached_tokens = max(0, int(cached_tokens))
    tokens_total = max(0, int(tokens_total))
    new_prefill_tokens = max(0, int(new_prefill_tokens))
    try:
        payload: dict[str, Any] = {
            "phase": "chunk",
            "tokens_done": min(tokens_total, cached_tokens),
            "tokens_total": tokens_total,
            "cached_tokens": cached_tokens,
            "new_prefill_tokens": new_prefill_tokens,
            "elapsed_s": max(0.0, time.perf_counter() - float(started_s)),
            "cache_hit": cached_tokens > 0,
        }
        if cache_source:
            payload["cache_source"] = str(cache_source)
        if ssd_cache_hit:
            payload.update(
                {
                    "ssd_cache_hit": True,
                    "ssd_cached_tokens": int(ssd_cached_tokens),
                    "ssd_restore_s": float(ssd_restore_s),
                    "ssd_suffix_tokens": (
                        int(ssd_suffix_tokens)
                        if ssd_suffix_tokens is not None
                        else new_prefill_tokens
                    ),
                }
            )
        callback(payload)
    except Exception:
        pass


def _cache_offset(cache: Any) -> int:
    if not cache:
        return 0
    try:
        return int(getattr(cache[0], "offset", 0) or 0)
    except Exception:
        return 0


def _trim_cache_to_offset(cache: Any, offset: int) -> bool:
    target = max(0, int(offset))
    if not cache:
        return target == 0
    trims: list[tuple[Callable[[int], Any], int]] = []
    for entry in cache:
        current = int(getattr(entry, "offset", target) or 0)
        if current < target:
            return False
        delta = current - target
        trim = getattr(entry, "trim", None)
        if not callable(trim):
            return False
        if delta <= 0:
            continue
        max_rollback = getattr(entry, "max_rollback", None)
        if max_rollback is not None and delta > int(max_rollback):
            return False
        trims.append((trim, delta))
    for trim, delta in trims:
        trimmed = int(trim(delta))
        if trimmed != delta:
            return False
    return True


def _entry_matches_restore_lookup(
    entry: Any,
    rt: MTPLXRuntime,
    *,
    hidden_variant: str | None,
    template_hash: str | None,
    mtp_history_policy: str | None,
    draft_head_identity: str | None,
    policy_fingerprint: str | None,
) -> bool:
    if getattr(entry, "model_path", None) != str(rt.model_path):
        return False
    if (
        hidden_variant is not None
        and getattr(entry, "hidden_variant", None) != hidden_variant
    ):
        return False
    if (
        template_hash is not None
        and getattr(entry, "template_hash", None) != template_hash
    ):
        return False
    entry_policy = getattr(entry, "mtp_history_policy", None)
    if mtp_history_policy is not None:
        if entry_policy != mtp_history_policy:
            committed = {"committed", "last_window"}
            if entry_policy not in committed or mtp_history_policy not in committed:
                return False
    if (
        draft_head_identity is not None
        and getattr(entry, "draft_head_identity", None) != draft_head_identity
    ):
        return False
    if (
        policy_fingerprint is not None
        and getattr(entry, "policy_fingerprint", None) != policy_fingerprint
    ):
        return False
    if getattr(entry, "mtp_snapshot_epoch", None) is not None and int(
        getattr(entry, "mtp_snapshot_epoch")
    ) != int(getattr(entry, "snapshot_epoch", 0)):
        return False
    return True


def _near_prefix_restore_enabled() -> bool:
    return not _env_falsey("MTPLX_SESSION_NEAR_PREFIX_RESTORE")


def _opencode_compact_tool_history_policy(policy_fingerprint: str | None) -> bool:
    fingerprint = str(policy_fingerprint or "")
    return (
        "tool_prompt_mode=compact" in fingerprint
        and "tool_contract=compact_tool_contract:schema_free:v1" in fingerprint
        and "opencode_prompt_contract=opencode_agent" in fingerprint
    )


def _restore_near_prefix_prompt_state(
    rt: MTPLXRuntime,
    prompt_ids: list[int],
    *,
    base_hidden_variant: str,
    mtp_hidden_variant: str,
    mtp_history_policy: str,
    session_bank: Any,
    template_hash: str | None,
    draft_head_identity: str | None,
    policy_fingerprint: str | None,
    min_restore_tokens: int = 0,
    allow_block_prefix: bool | None = None,
    abort_check: Callable[[], bool] | None = None,
    chunk_callback: Callable[[dict[str, Any]], None] | None = None,
    chunk_started_s: float | None = None,
    cache_factory: Callable[[], Any] | None = None,
    stable_prefix_len: int | None = None,
    matched_ceiling: int | None = None,
) -> PromptState | None:
    """matched_ceiling: hard cap on any candidate's matched length.

    Vision requests pass the FIRST image-pad position: this lane matches on
    raw token ids, where every pad equals every pad, so an uncapped match
    can run INTO an image span and a boundary restore there resurrects KV
    whose embeddings came from different pixels (2026-08-07 pillar
    alias-leg regression — served restore_point 14704 vs content divergence
    at the span start). Capping at the first pad keeps this lane text-only;
    full-image warm reuse stays with the exact restore path, which matches
    on content-keyed surrogate ids.
    """
    if not _near_prefix_restore_enabled() or len(prompt_ids) < 2:
        return None
    if matched_ceiling is not None and int(matched_ceiling) < 2:
        return None
    candidates = getattr(session_bank, "near_prefix_candidates", None)
    if not callable(candidates):
        return None
    max_gap = max(0, _env_int("MTPLX_SESSION_NEAR_PREFIX_MAX_TOKEN_GAP", 8))
    min_match = max(1, _env_int("MTPLX_SESSION_NEAR_PREFIX_MIN_MATCH_TOKENS", 64))
    block_prefix_enabled = (
        block_prefix_restore_enabled()
        if allow_block_prefix is None
        else bool(allow_block_prefix)
    )
    block_size = max(1, _env_int("MTPLX_SESSION_PREFIX_BLOCK_SIZE", 256))
    block_min_match = max(
        block_size,
        _env_int("MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS", 512),
    )
    candidates_seen = 0
    _prefix_restore_fn = getattr(session_bank, "restore_entry_prefix_cache", None)
    _prefix_restore_supports_served = callable(
        _prefix_restore_fn
    ) and _accepts_served_out(_prefix_restore_fn)
    # Pass the serve floor so the bank's resident-duplicate shadow gate can
    # mirror THIS caller's eligibility exactly (explicit capability
    # attribute; duck-typed banks get the legacy call shape).
    _candidates_kwargs: dict[str, Any] = {}
    if getattr(session_bank, "SUPPORTS_NEAR_PREFIX_MIN_RESTORE", False):
        _candidates_kwargs["min_restore_tokens"] = int(min_restore_tokens)
    for entry, matched in candidates(
        prompt_ids,
        max_token_gap=max_gap,
        min_matched_tokens=min_match,
        **_candidates_kwargs,
        block_size=block_size,
        block_min_matched_tokens=block_min_match,
        allow_block_prefix=block_prefix_enabled,
        model_path=str(rt.model_path),
        mtp_enabled=bool(rt.mtp_enabled),
        hidden_variant=base_hidden_variant,
        template_hash=template_hash,
        mtp_history_policy=mtp_history_policy,
        draft_head_identity=draft_head_identity,
        policy_fingerprint=policy_fingerprint,
    ):
        _check_postcommit_abort(abort_check)
        candidates_seen += 1
        matched = int(matched)
        if matched_ceiling is not None and matched > int(matched_ceiling):
            matched = int(matched_ceiling)

        def _near_debug(reason: str) -> None:
            if os.environ.get("MTPLX_DEBUG_PREFIX_DIVERGENCE"):
                print(
                    f"[mtplx] near-prefix reject: entry_len="
                    f"{int(getattr(entry, 'prefix_len', 0) or 0)} "
                    f"matched={matched} min_restore={int(min_restore_tokens)} "
                    f"reason={reason}",
                    file=sys.stderr,
                    flush=True,
                )

        if matched <= int(min_restore_tokens):
            _near_debug("matched_below_min_restore")
            continue
        if matched < 2 or matched >= int(getattr(entry, "prefix_len", 0) or 0):
            _near_debug("matched_out_of_range")
            continue
        if not _entry_matches_restore_lookup(
            entry,
            rt,
            hidden_variant=base_hidden_variant,
            template_hash=template_hash,
            mtp_history_policy=mtp_history_policy,
            draft_head_identity=draft_head_identity,
            policy_fingerprint=policy_fingerprint,
        ):
            _near_debug("identity_mismatch")
            continue
        committed_history_required = _mtp_history_uses_committed_cache(
            mtp_history_policy
        )
        has_committed_history = (
            entry.mtp_history_snapshot is not None
            or getattr(entry, "mtp_history_cache_ref", None) is not None
        )
        if committed_history_required and not has_committed_history:
            _near_debug("missing_committed_mtp_history")
            continue
        if getattr(entry, "has_recurrent", False):
            gap_from_entry = int(getattr(entry, "prefix_len", 0) or 0) - matched
            if gap_from_entry > max_gap:
                # Boundary-true restores land at the newest recurrent boundary
                # at/below `matched`, not at `matched` itself. A candidate is
                # only worth taking when that achievable point still beats the
                # exact-prefix alternative — otherwise a boundary-quantized
                # restore silently LOSES tokens vs the plain exact restore
                # (observed: block candidate matched=2560 restoring at 2354
                # while an exact 2383-entry existed).
                probe = getattr(entry, "recurrent_boundary_at_or_below", None)
                achievable = 0
                if callable(probe):
                    boundary_probe = probe(matched)
                    if boundary_probe is not None:
                        achievable = int(boundary_probe[0])
                if achievable <= int(min_restore_tokens):
                    _near_debug(f"boundary_not_better:{achievable}")
                    continue

        prefix_restore = None
        cache_restore_time_s = 0.0
        restore_entry_prefix_cache = getattr(
            session_bank, "restore_entry_prefix_cache", None
        )
        if callable(restore_entry_prefix_cache):
            # kvcache-v2: prefer the zero-copy reference lease whenever the
            # entry still owns live buffers; clone is the fallback for
            # consumed leases and snapshot-only entries. (Pre-v2 this was
            # clone-first, paying O(bytes) even when a free lease existed.)
            restore_modes = (
                ["reference"]
                if getattr(entry, "live_ref_only", False)
                else ["reference", "clone"]
                if getattr(entry, "cache_ref", None) is not None
                else ["clone"]
            )
            bank_served: dict[str, Any] = {}
            for restore_mode in restore_modes:
                # Fresh dict per attempt: a failed reference attempt must not
                # pollute the successful clone attempt's telemetry. Only the
                # winning attempt's dict is retained.
                attempt_served: dict[str, Any] = {}
                restore_kwargs: dict[str, Any] = {"served_out": attempt_served}
                if not _prefix_restore_supports_served:
                    restore_kwargs = {}
                restore_started = time.perf_counter()
                prefix_restore = restore_entry_prefix_cache(
                    rt,
                    entry,
                    matched,
                    mode=restore_mode,
                    cache_factory=cache_factory,
                    **restore_kwargs,
                )
                cache_restore_time_s += time.perf_counter() - restore_started
                if prefix_restore is not None:
                    bank_served = attempt_served
                    break
        else:
            restore_started = time.perf_counter()
            cache = cache_factory() if cache_factory is not None else rt.make_cache()
            restore_cache(
                cache,
                entry.cache_snapshot,
                restore_meta_state=cache_factory is None,
            )
            if _trim_cache_to_offset(cache, matched - 1):
                mtp_history_cache = None
                if entry.mtp_history_snapshot is not None:
                    mtp_history_cache = rt.make_mtp_cache()
                    restore_cache(mtp_history_cache, entry.mtp_history_snapshot)
                    if not _trim_cache_to_offset(mtp_history_cache, matched - 1):
                        mtp_history_cache = None
                        cache = None
                if cache is not None:
                    prefix_restore = (cache, mtp_history_cache, "clone")
            cache_restore_time_s = time.perf_counter() - restore_started
        if prefix_restore is None:
            _near_debug(
                "restore_failed:"
                + str(getattr(session_bank, "last_miss_reason", None))
            )
            continue
        _near_debug("served")
        boundary_hidden = None
        if len(prefix_restore) == 5:
            (
                cache,
                mtp_history_cache,
                storage_restore_mode,
                restore_point,
                boundary_hidden,
            ) = prefix_restore
        elif len(prefix_restore) == 4:
            cache, mtp_history_cache, storage_restore_mode, restore_point = (
                prefix_restore
            )
        else:  # legacy 3-tuple session banks (duck-typed callers)
            cache, mtp_history_cache, storage_restore_mode = prefix_restore
            restore_point = matched
        restore_point = int(restore_point)
        boundary_restore = boundary_hidden is not None or restore_point < matched
        served_truth: dict[str, Any] = {
            "entry_prefix_len": int(getattr(entry, "prefix_len", 0) or 0),
            "entry_token_hash": str(getattr(entry, "token_hash", "") or ""),
            "requested_matched": int(matched),
            "actual_restore_point": int(restore_point),
            "boundary_restore": bool(boundary_restore),
            "storage_restore_mode": str(storage_restore_mode),
            "lazy_kv": bool(getattr(entry, "lazy_kv", False)),
            "candidate_index": int(candidates_seen),
            "bank": bank_served,
        }
        _done_at = getattr(entry, "cold_encode_completed_at", None)
        served_truth["encode_completed"] = _done_at is not None
        if _done_at is not None:
            served_truth["encode_completed_age_s"] = round(
                max(0.0, time.monotonic() - float(_done_at)), 3
            )
        if committed_history_required and mtp_history_cache is None:
            continue
        if (
            boundary_restore
            and committed_history_required
            and boundary_hidden is None
        ):
            # Without the boundary's hidden state the committed MTP history
            # cannot resume exactly at b; running a seed forward instead would
            # advance the recurrent state twice. Fail closed to the next
            # candidate (or cold).
            continue
        cache_source = str(getattr(entry, "cache_source", "ram") or "ram")
        ssd_cache_hit = bool(getattr(entry, "ssd_cache_hit", False)) or cache_source == "ssd"
        ssd_restore_s = float(getattr(entry, "ssd_restore_s", 0.0) or 0.0)
        ssd_cached_tokens = restore_point if ssd_cache_hit else 0
        total_cache_restore_time_s = (
            cache_restore_time_s + ssd_restore_s if ssd_cache_hit else cache_restore_time_s
        )

        _check_postcommit_abort(abort_check)
        if boundary_restore:
            # Boundary-true restore: KV and recurrent state both sit exactly
            # at restore_point; token restore_point-1 must NOT be re-run (the
            # recurrent state already consumed it). The suffix prefill below
            # regenerates logits; MTP history resumes from boundary_hidden.
            logits = None
            hidden = boundary_hidden
            repair_time = 0.0
        else:
            started = time.perf_counter()
            with attention_phase("prefill"):
                logits, hidden = _forward_ar_optional_hidden(
                    rt,
                    mx.array([[int(prompt_ids[restore_point - 1])]]),
                    cache=cache,
                    hidden_variant=base_hidden_variant,
                    emit_logits=True,
                    logits_keep=1 if _final_logits_prefill_enabled() else None,
                )
            if hidden is None:
                _eval(logits)
            else:
                _eval(logits, hidden)
            repair_time = time.perf_counter() - started
        _check_postcommit_abort(abort_check)
        restore_kind_base = (
            "block_prefix" if int(entry.prefix_len) - matched > max_gap else "near_prefix"
        )
        if restore_point < matched:
            restore_kind_base = f"{restore_kind_base}_boundary"
        restore_kind_suffix = (
            "reference_lease"
            if str(storage_restore_mode) == "reference_lease"
            else "clone"
        )
        restore_kind = f"{restore_kind_base}_{restore_kind_suffix}"
        if ssd_cache_hit:
            restore_kind = f"ssd_{restore_kind}"
        try:
            diagnostic = getattr(session_bank, "last_prefix_diagnostic", None)
            if isinstance(diagnostic, dict):
                diagnostic["restore_kind"] = restore_kind
                diagnostic["cache_source"] = cache_source
                diagnostic["ssd_cache_hit"] = bool(ssd_cache_hit)
                diagnostic["ssd_cached_tokens"] = int(ssd_cached_tokens)
                diagnostic["ssd_restore_s"] = float(ssd_restore_s)
        except Exception:
            pass
        suffix = list(prompt_ids[restore_point:])
        if boundary_restore and not suffix:
            # A boundary restore has no seed logits; decode cannot start from
            # an empty suffix. (Only reachable when a boundary coincides with
            # a fully-contained prompt — fall through to other candidates.)
            continue
        inherited_boundaries = _inherited_gdn_boundaries(entry, restore_point)
        restored = SimpleNamespace(
            entry=SimpleNamespace(prefix_len=restore_point),
            cache=cache,
            logits=logits[:, -1, :] if logits is not None else None,
            hidden=hidden[:, -1:, :] if hidden is not None else None,
            mtp_history_cache=mtp_history_cache,
            restore_mode=restore_kind,
        )
        _emit_prefill_restore_progress(
            chunk_callback,
            tokens_total=len(prompt_ids),
            cached_tokens=restore_point,
            new_prefill_tokens=len(suffix),
            started_s=chunk_started_s if chunk_started_s is not None else started,
            cache_source=cache_source,
            ssd_cache_hit=ssd_cache_hit,
            ssd_cached_tokens=ssd_cached_tokens,
            ssd_restore_s=ssd_restore_s,
            ssd_suffix_tokens=len(suffix),
        )
        if not suffix:
            entry.hits += 1
            entry.last_access_s = time.time()
            repage_time = _maybe_repage_target_prefill_cache(rt, cache)
            return PromptState(
                trunk_cache=cache,
                logits=logits[:, -1, :],
                hidden=hidden[:, -1:, :] if hidden is not None else None,
                committed_mtp_cache=mtp_history_cache,
                token_prefix=tuple(int(token) for token in prompt_ids),
                prompt_eval_time_s=repair_time + repage_time,
                cache_restore_time_s=total_cache_restore_time_s,
                mtp_history_policy=mtp_history_policy,
                cached_tokens=restore_point,
                suffix_tokens=0,
                cache_hit=True,
                cache_source=cache_source,
                ssd_cache_hit=ssd_cache_hit,
                ssd_cached_tokens=ssd_cached_tokens,
                ssd_restore_s=ssd_restore_s,
                restore_mode=restore_kind,
                gdn_boundaries=inherited_boundaries,
                restore_served=served_truth,
            )
        suffix_boundary_sink: list[tuple[int, Any, Any]] | None = (
            list(inherited_boundaries)
            if _gdn_boundary_capture_enabled()
            else None
        )
        suffix_logits, suffix_hidden, suffix_time, mtp_history_time = (
            _prefill_restored_prompt_suffix(
                rt,
                restored,
                suffix,
                base_hidden_variant=base_hidden_variant,
                mtp_hidden_variant=mtp_hidden_variant,
                mtp_history_policy=mtp_history_policy,
                abort_check=abort_check,
                chunk_callback=chunk_callback,
                tokens_total=len(prompt_ids),
                cached_tokens=restore_point,
                chunk_started_s=chunk_started_s,
                gdn_boundary_sink=suffix_boundary_sink,
                stable_prefix_len=stable_prefix_len,
            )
        )
        entry.hits += 1
        entry.last_access_s = time.time()
        return PromptState(
            trunk_cache=cache,
            logits=suffix_logits,
            hidden=suffix_hidden,
            committed_mtp_cache=mtp_history_cache,
            token_prefix=tuple(int(token) for token in prompt_ids),
            prompt_eval_time_s=repair_time + suffix_time + mtp_history_time,
            prompt_mtp_history_time_s=mtp_history_time,
            cache_restore_time_s=total_cache_restore_time_s,
            mtp_history_policy=mtp_history_policy,
            cached_tokens=restore_point,
            suffix_tokens=len(suffix),
            cache_hit=True,
            cache_source=cache_source,
            ssd_cache_hit=ssd_cache_hit,
            ssd_cached_tokens=ssd_cached_tokens,
            ssd_restore_s=ssd_restore_s,
            restore_mode=restore_kind,
            gdn_boundaries=(
                suffix_boundary_sink
                if suffix_boundary_sink is not None
                else inherited_boundaries
            ),
            restore_served=served_truth,
        )
    return None


def _small_suffix_fused_max() -> int:
    """Suffix length at or below which restored-prefix prefill fuses into one
    forward (kvcache-v2). 0 disables the fast path."""
    raw = os.environ.get("MTPLX_SMALL_SUFFIX_FUSED_MAX", "512")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 512


def _gdn_boundary_capture_enabled() -> bool:
    """Interior recurrent boundary capture during cold prefill (kvcache-v2)."""
    raw = str(os.environ.get("MTPLX_GDN_BOUNDARY_CAPTURE", "1")).strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _gdn_boundary_max_count() -> int:
    raw = os.environ.get("MTPLX_GDN_BOUNDARY_MAX", "8")
    try:
        return max(2, int(raw))
    except (TypeError, ValueError):
        return 8


def _gdn_boundary_tail_interval() -> int:
    """Sub-chunk capture grid for the final prefill chunk (0 disables).

    Agent/RAG divergence concentrates near the prompt tail, so the last chunk
    gets a finer boundary grid than the chunk-edge default; 256 keeps the
    worst-case boundary→match re-prefill to a few hundred ms.
    """
    raw = os.environ.get("MTPLX_GDN_BOUNDARY_TAIL_INTERVAL", "256")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 256


def _cache_has_recurrent_entries(cache: list[Any] | None) -> bool:
    from .cache_state import _is_trimmable

    return any(not _is_trimmable(entry) for entry in (cache or []))


def _thin_gdn_boundary_records(
    records: list[tuple[int, Any, Any]], cap: int
) -> list[tuple[int, Any, Any]]:
    """Thin boundary records to `cap` with geometric distance-from-tail coverage.

    The retired policy ("keep the oldest + a dense tail", implemented as
    pop(1) on every over-cap append) had no coverage guarantee in the middle:
    any append churn after the cap — postcommit re-forward chunk edges,
    inheritance re-thinning on clone/lease chains — ate the mid-prefix records
    one by one, leaving [oldest, <recent tail>]. A near-miss prefix match that
    landed in the hole then restored at the OLDEST boundary: measured
    2026-07-17 on the Hermes lane, matched≈22.5k restored at 2,048 → 20k+
    re-prefill and 35-50s TTFT (MEASUREMENTS 01:05 §B).

    This policy keeps, in one pass over the records sorted by position:
      - the newest record (divergence distance ~0),
      - one record per power-of-two bucket of distance-from-newest
        (256..512, 512..1024, ... tokens), preferring the record CLOSEST to
        the newest inside each bucket,
      - the oldest record (deep-divergence anchor).
    Coverage invariant (unit-tested): for any matched position covered by the
    original records, restoring at the nearest kept boundary at or below it
    re-prefills at most ~3x the true divergence distance from the tail (plus
    one capture-grid interval) — cost stays proportional to how far the
    request actually diverged, never a cliff.
    """
    if len(records) <= max(2, cap):
        return list(records)
    ordered = sorted(records, key=lambda record: int(record[0]))
    newest = ordered[-1]
    oldest = ordered[0]
    newest_pos = int(newest[0])
    kept: dict[int, tuple[int, Any, Any]] = {int(newest[0]): newest, int(oldest[0]): oldest}
    # Walk from the tail toward the head (distance from newest increasing).
    # Keep the first record past each doubling floor — one keeper per
    # distance scale, geometric spacing by construction regardless of how
    # dense or lumpy the input grid is. The first floor adapts to the record
    # span so the doubling chain always reaches the oldest record within the
    # cap budget (cap-2 scales after newest+oldest): full-span coverage with
    # worst-case slack span/2^(cap-2) instead of an uncovered deep-middle.
    span = max(1, newest_pos - int(oldest[0]))
    base = 256
    scales = max(1, cap - 2)
    while base * (1 << (scales - 1)) < span and base < span:
        base *= 2
    floor = 0
    next_floor = base
    idx = len(ordered) - 2
    while idx > 0 and len(kept) < cap:
        pos = int(ordered[idx][0])
        distance = newest_pos - pos
        if distance > floor:
            kept.setdefault(pos, ordered[idx])
            while next_floor < distance:
                next_floor *= 2
            floor = next_floor
            next_floor *= 2
        idx -= 1
    return sorted(kept.values(), key=lambda record: int(record[0]))


def _capture_gdn_boundary(
    sink: list[tuple[int, Any, Any]] | None,
    tokens_done: int,
    cache: list[Any],
    hidden_last: Any | None = None,
) -> None:
    """Append a recurrent-only snapshot at `tokens_done`, geometric retention.

    `hidden_last` is the base hidden state of token `tokens_done - 1` when the
    producing chunk computed hidden (MTP streaming prefill). Restores use it to
    resume committed MTP history at the boundary WITHOUT a seed re-forward —
    re-running token b-1 through the model would advance the recurrent state a
    second time and break exactness (temp-0 divergence, found 2026-07-03).

    Snapshot cost is MB-scale per boundary (conv tail + GDN matrix state), so
    the count is capped; over cap the list is re-thinned to geometric
    distance-from-tail coverage (see _thin_gdn_boundary_records — the previous
    oldest+dense-tail pop(1) policy left a mid-prefix coverage hole that
    near-miss restores fell into).
    """
    if sink is None or tokens_done < 1:
        return
    try:
        hidden_leaf = None
        if hidden_last is not None:
            hidden_leaf = detach_array_leaf(hidden_last, mode="contiguous_eval")
        sink.append(
            (int(tokens_done), snapshot_untrimmable_cache(cache), hidden_leaf)
        )
        cap = _gdn_boundary_max_count()
        if len(sink) > cap:
            sink[:] = _thin_gdn_boundary_records(sink, cap)
    except Exception:
        # Boundary capture is an accelerator for future restores; never let it
        # break the cold prefill that is running right now.
        pass


def _prefill_spans_with_tail_grid(
    token_count: int,
    *,
    tail_interval: int,
    mandatory_edges: tuple[int, ...] = (),
) -> list[tuple[int, int]]:
    spans = list(_iter_prefill_chunk_spans(token_count))
    if not spans or tail_interval <= 0:
        return _split_spans_at(spans, mandatory_edges)
    start, end = spans[-1]
    if end - start <= tail_interval:
        return _split_spans_at(spans, mandatory_edges)
    refined = spans[:-1]
    cursor = start
    while cursor < end:
        refined.append((cursor, min(end, cursor + tail_interval)))
        cursor += tail_interval
    return _split_spans_at(refined, mandatory_edges)


def _inherited_gdn_boundaries(entry: Any, restore_point: int) -> list:
    """Boundaries carried over from a restored SessionBank entry.

    A boundary record (position, recurrent snapshot[, hidden]) describes the
    token PREFIX up to `position`. After a restore at `restore_point`, every
    record at or before that point still describes the identical prefix of
    the new request, so the entry banked for this request can reuse them
    verbatim. Without inheritance, entries produced by warm (restored-suffix)
    prefills carried NO boundaries — the boundary-true block restore then
    failed closed on them and every mid-loop agent round fell back to the
    last completed postcommit (measured 2026-07-04: rounds pinned at a stale
    6.5k prefix while prompts grew to 12.5k, and the follow-up turn went
    fully cold with `no_snapshot_coverage`).
    """
    records = list(getattr(entry, "gdn_boundaries", None) or [])
    kept = [record for record in records if int(record[0]) <= int(restore_point)]
    cap = _gdn_boundary_max_count()
    if len(kept) > cap:
        # Geometric retention, mirroring _capture_gdn_boundary — the old
        # oldest+dense-tail pop(1) here was the second churn site that
        # hollowed out mid-prefix coverage on clone/lease chains.
        kept = _thin_gdn_boundary_records(kept, cap)
    return kept


def _accepts_served_out(fn: Any) -> bool:
    """Feature-detect the passive-probe ``served_out`` kwarg.

    Detection happens ONCE, before any call — never a blanket
    TypeError-retry around the restore itself, which could re-execute a
    partially completed restore (for example after a consumed live lease)
    and would mask internal TypeErrors.
    """
    try:
        return "served_out" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _prefill_store_result(
    entry: Any,
    *,
    suffix_tokens: int,
    elapsed_s: float,
    mtp_snapshot_elapsed_s: float,
    put_elapsed_s: float,
    put_timing: dict[str, object],
) -> dict[str, object]:
    # SessionBank.put legitimately returns None (oversized/skipped snapshot);
    # "stored" must reflect that return, never assume success.
    return {
        "stored": entry is not None,
        "reason": (
            "committed_prefill_prefix"
            if entry is not None
            else "sessionbank_snapshot_skipped"
        ),
        "suffix_tokens": int(suffix_tokens),
        "elapsed_s": float(elapsed_s),
        "mtp_snapshot_elapsed_s": float(mtp_snapshot_elapsed_s),
        "put_elapsed_s": float(put_elapsed_s),
        "put_timing": put_timing,
    }


def _store_on_prefill_env_enabled() -> bool:
    """Default ON (2026-07-02 A/B: agent turn-2 TTFT 40s -> 1.3s, e2e 1.7 ->
    33.6 tok/s at 25k ctx; cost is one bank snapshot copy on large cold
    prefills). MTPLX_SESSION_STORE_ON_PREFILL=0 is the kill switch."""
    raw = str(os.environ.get("MTPLX_SESSION_STORE_ON_PREFILL", "1")).strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _store_on_prefill_min_suffix() -> int:
    raw = os.environ.get("MTPLX_SESSION_STORE_ON_PREFILL_MIN_SUFFIX", "1024")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1024


def _debug_prefix_divergence(rt: MTPLXRuntime, prompt_ids: list[int], session_bank: Any) -> None:
    """Env-gated diagnostic: report where the prompt diverges from each bank entry.

    For every bank entry that shares a non-trivial prefix with the incoming
    prompt but is not a clean prefix of it, print the first divergent token
    index plus decoded context on both sides. Debug-only (MTPLX_DEBUG_PREFIX_DIVERGENCE).
    """
    try:
        entries = list(getattr(session_bank, "_entries", {}).values())
        tokenizer = getattr(rt, "tokenizer", None)
        prompt = list(int(t) for t in prompt_ids)

        def _decode(ids: list[int]) -> str:
            if tokenizer is None:
                return str(ids)
            try:
                return tokenizer.decode(ids)
            except Exception:
                return str(ids)

        rows = []
        for entry in entries:
            toks = list(entry.token_ids)
            n = min(len(toks), len(prompt))
            i = 0
            while i < n and toks[i] == prompt[i]:
                i += 1
            rows.append((i, len(toks), toks))
        rows.sort(key=lambda r: -r[0])
        for matched, entry_len, toks in rows[:3]:
            if matched >= min(entry_len, len(prompt)):
                print(
                    f"[mtplx] prefix-diverge: entry_len={entry_len} clean prefix (matched={matched})",
                    file=sys.stderr,
                )
                continue
            lo = max(0, matched - 24)
            print(
                f"[mtplx] prefix-diverge: entry_len={entry_len} matched={matched} "
                f"prompt_len={len(prompt)}\n"
                f"  entry [{lo}:{matched + 40}]: "
                f"{_decode(toks[lo:matched + 40])!r}\n"
                f"  prompt[{lo}:{matched + 40}]: "
                f"{_decode(prompt[lo:matched + 40])!r}",
                file=sys.stderr,
            )
    except Exception as exc:  # diagnostic only - never break the request
        print(f"[mtplx] prefix-diverge diagnostic failed: {exc}", file=sys.stderr)


def restore_or_prefill_prompt_state(
    rt: MTPLXRuntime,
    prompt_ids: list[int],
    *,
    base_hidden_variant: str | None = None,
    mtp_hidden_variant: str | None = None,
    mtp_history_policy: str = "cycle",
    session_bank: Any | None = None,
    restore_mode: str = "clone",
    session_id: str | None = None,
    template_hash: str | None = None,
    draft_head_identity: str | None = None,
    policy_fingerprint: str | None = None,
    abort_check: Callable[[], bool] | None = None,
    prefill_callback: Callable[[dict[str, Any]], None] | None = None,
    vision_splice: Any | None = None,
    store_prefix_snapshot: bool | None = None,
    stable_prefix_len: int | None = None,
) -> PromptState:
    """Build the initial prompt state used by MTP-k decode.

    This is the first mechanical split point for the serving engine. It keeps
    today's cold path behavior intact while giving EngineSession a concrete
    target for future warm SessionBank restores.

    store_prefix_snapshot: store the completed prompt-boundary state into the
    session bank before decode starts (None = follow the
    MTPLX_SESSION_STORE_ON_PREFILL env gate). This makes warm turns
    cadence-independent for agent loops: the idle async postcommit aborts
    whenever foreground work is pending and never retries, so back-to-back
    tool-calling turns otherwise starve the cache and full-prefill every
    turn (2026-07-02 diagnosis). The KV for the prompt is already computed
    here — the only cost is the bank's snapshot copy, taken only when the
    new-prefill suffix is large enough to have been a real miss.
    """
    bank_key_ids: list[int] | None = None
    vision_restore_spans: list[tuple[int, int]] | None = None
    if vision_splice is not None and session_bank is not None:
        # Image content is not represented in token ids, so raw prefix reuse
        # would alias different images. The bank may only participate through
        # the content-keyed view: image pad positions remapped to surrogates
        # derived from each image's byte digest, making the key sequence a
        # pure function of text + pixels. Without that identity the server
        # bypasses the bank; enforce the invariant here as well.
        from mtplx.vision.splice import vision_bank_key_ids, vision_image_spans

        bank_key_ids = vision_bank_key_ids(prompt_ids, vision_splice)
        if bank_key_ids is None:
            raise ValueError(
                "vision requests must not use the session bank without "
                "content-keyed ids"
            )
        # Restore-safety spans: a prefix match may not END inside an image's
        # pad run — id-equality there is not input-equality (embeddings ride
        # out-of-band), so a partial-span restore resurrects another image's
        # KV. Full-span matches (same pixels -> same surrogates through the
        # span) stay fully warm. 2026-08-07 pillar alias-leg regression.
        vision_restore_spans = vision_image_spans(bank_key_ids, vision_splice)
    base_hidden_variant = _resolve_runtime_base_hidden_variant(rt, base_hidden_variant)
    mtp_hidden_variant = _resolve_runtime_mtp_hidden_variant(rt, mtp_hidden_variant)
    mtp_position_mode = _resolve_runtime_mtp_position_mode(rt)
    os.environ["MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS"] = str(len(prompt_ids))
    mtp_history_policy = _resolve_mtp_history_policy(
        mtp_history_policy,
        len(prompt_ids),
    )
    if not rt.mtp_enabled and _mtp_history_uses_committed_cache(mtp_history_policy):
        # Target-only AR runtimes (e.g. laguna_ar) carry no MTP head, so a
        # committed/last_window history policy would enter the
        # _prefill_committed_mtp_history_streaming branch and call
        # rt.make_mtp_cache(), which raises "MTP is not enabled for this
        # runtime". Degrade to the cycle (AR) prefill path, which banks only
        # the trunk cache — the prefix-reuse benefit AR turns actually use.
        # MTP-enabled runtimes keep their requested committed policy.
        mtp_history_policy = "cycle"
    mtp_history_window_tokens = (
        _mtp_history_last_window_tokens() if mtp_history_policy == "last_window" else 0
    )
    # Dashboard prefill instrumentation. We fire `phase: "started"` before
    # any restore/prefill work runs (so the UI can flip into prefill mode
    # immediately), `phase: "chunk"` from inside the chunked path after
    # each chunk completes, and `phase: "completed"` just before this
    # function returns. The callback must be cheap and exception-safe —
    # it runs on the generation hot path.
    prefill_started_s = time.perf_counter()
    if prefill_callback is not None:
        try:
            prefill_callback(
                {
                    "phase": "started",
                    "tokens_done": 0,
                    "tokens_total": int(len(prompt_ids)),
                    "cached_tokens": 0,
                    "new_prefill_tokens": int(len(prompt_ids)),
                    "elapsed_s": 0.0,
                    "started_s": prefill_started_s,
                }
            )
        except Exception:
            pass

    def _maybe_store_prefix_snapshot(state: PromptState) -> None:
        enabled = (
            _store_on_prefill_env_enabled()
            if store_prefix_snapshot is None
            else bool(store_prefix_snapshot)
        )
        if not enabled or session_bank is None:
            state.prefill_store_snapshot = {
                "stored": False,
                "skip_reason": "disabled" if session_bank is not None else "no_bank",
            }
            return
        if vision_splice is not None and bank_key_ids is None:
            state.prefill_store_snapshot = {
                "stored": False,
                "skip_reason": "vision_no_bank_key",
            }
            return
        if int(state.suffix_tokens or 0) < _store_on_prefill_min_suffix():
            # Warm restore or trivial extension: the existing postcommit
            # machinery owns those; storing again would just churn the bank.
            state.prefill_store_snapshot = {
                "stored": False,
                "skip_reason": "min_suffix",
                "suffix_tokens": int(state.suffix_tokens or 0),
            }
            return
        if os.environ.get("MTPLX_DEBUG_PREFIX_DIVERGENCE"):
            print(
                f"[mtplx] store-on-prefill: len={len(prompt_ids)} "
                f"boundaries={len(list(getattr(state, 'gdn_boundaries', None) or []))} "
                f"cached={int(state.cached_tokens)} restore={state.restore_mode}",
                file=sys.stderr,
                flush=True,
            )
        store_started = time.perf_counter()
        snapshot_done = store_started
        try:
            mtp_snapshot = (
                snapshot_cache(state.committed_mtp_cache)
                if state.committed_mtp_cache is not None
                else None
            )
            snapshot_done = time.perf_counter()
            put_timing: dict[str, object] = {}
            entry = session_bank.put(
                runtime=rt,
                token_ids=list(bank_key_ids if bank_key_ids is not None else prompt_ids),
                cache=state.trunk_cache,
                logits=state.logits,
                hidden=state.hidden,
                hidden_variant=base_hidden_variant,
                keep_live_ref=False,
                session_id=session_id,
                template_hash=template_hash,
                mtp_history_policy=mtp_history_policy,
                draft_head_identity=draft_head_identity,
                policy_fingerprint=policy_fingerprint,
                mtp_history_snapshot=mtp_snapshot,
                snapshot_epoch=len(prompt_ids),
                mtp_snapshot_epoch=len(prompt_ids) if mtp_snapshot is not None else None,
                gdn_boundaries=list(getattr(state, "gdn_boundaries", None) or []),
                timing_out=put_timing,
            )
            put_done = time.perf_counter()
            state.prefill_store_snapshot = _prefill_store_result(
                entry,
                suffix_tokens=int(state.suffix_tokens),
                elapsed_s=put_done - store_started,
                mtp_snapshot_elapsed_s=snapshot_done - store_started,
                put_elapsed_s=put_done - snapshot_done,
                put_timing=put_timing,
            )
        except Exception as exc:
            # Cache priming must never break or slow the request path in a
            # user-visible way; a failed store just means a cold next turn.
            state.prefill_store_snapshot = {
                "stored": False,
                "reason": f"prefill_store_error:{type(exc).__name__}",
                "suffix_tokens": int(state.suffix_tokens),
                "elapsed_s": time.perf_counter() - store_started,
                "mtp_snapshot_elapsed_s": max(0.0, snapshot_done - store_started),
            }

    def _emit_prefill_complete(state: PromptState) -> PromptState:
        _maybe_store_prefix_snapshot(state)
        if prefill_callback is None:
            return state
        try:
            elapsed = max(0.0, time.perf_counter() - prefill_started_s)
            new_tokens = int(state.suffix_tokens)
            wall_tok_s = (
                (new_tokens / elapsed) if elapsed > 0 and new_tokens > 0 else None
            )
            compute_elapsed = max(0.0, float(state.prompt_eval_time_s or 0.0))
            compute_tok_s = (
                (new_tokens / compute_elapsed)
                if compute_elapsed > 0 and new_tokens > 0
                else None
            )
            prefill_callback(
                {
                    "phase": "completed",
                    "tokens_total": int(len(prompt_ids)),
                    "cached_tokens": int(state.cached_tokens),
                    "new_prefill_tokens": new_tokens,
                    "elapsed_s": elapsed,
                    "prompt_eval_time_s": compute_elapsed,
                    "prefill_tok_s": compute_tok_s if compute_tok_s is not None else wall_tok_s,
                    "prefill_compute_tok_s": compute_tok_s,
                    "prefill_wall_tok_s": wall_tok_s,
                    "cache_hit": bool(state.cache_hit),
                }
            )
        except Exception:
            pass
        return state

    _check_postcommit_abort(abort_check)
    if session_bank is not None and _env_truthy("MTPLX_DEBUG_PREFIX_DIVERGENCE"):
        _debug_prefix_divergence(rt, prompt_ids, session_bank)
    if session_bank is not None:
        restore_cache_factory = _session_restore_cache_factory(rt)
        normalized_restore_mode = str(restore_mode).replace("-", "_")
        allow_live_frontier_reference = (
            normalized_restore_mode in {"reference", "reference_lease"}
            and _session_live_frontier_reference_restore_enabled()
        )
        effective_restore_mode = (
            "clone"
            if restore_cache_factory is not None
            and normalized_restore_mode in {"reference", "reference_lease"}
            and not allow_live_frontier_reference
            else restore_mode
        )
        # The bank only ever sees the content-keyed view of a vision prompt;
        # for text prompts the two views are the same list.
        bank_match_ids = bank_key_ids if bank_key_ids is not None else prompt_ids
        exact_prefix_len = 0
        try:
            longest_prefix = getattr(session_bank, "longest_prefix", None)
            if callable(longest_prefix):
                exact_entry = longest_prefix(bank_match_ids)
                if exact_entry is not None:
                    exact_prefix_len = int(
                        getattr(exact_entry, "prefix_len", 0) or 0
                    )
        except Exception:
            exact_prefix_len = 0

        tried_larger_near_prefix = False
        # Vision prompts stay off the near/block-prefix lane for now: that
        # path interleaves bank matching with model forwards and would need
        # the keyed/real id split threaded through it. Exact-prefix restores
        # cover the strictly-extending agent flow; divergent vision histories
        # fall back to a full prefill exactly like the pre-keying behavior.
        # Fires on RAM exact-miss too (exact_prefix_len == 0): supersede
        # removes short same-lineage RAM entries as turns extend, so a
        # divergent agent turn often has rich near-prefix candidates in RAM
        # but NO RAM exact match — gating this lane on an exact hit sent
        # those turns to session_bank.restore(), whose SSD fallback served a
        # stale short clean-prefix entry instead (#121, turns 6+ in the
        # 2026-07-16 replay: ssd_clone at 2361 while RAM candidates matched
        # 2770+ with boundaries).
        if exact_prefix_len < len(prompt_ids) and vision_splice is None:
            # A short exact-prefix entry must not shadow a longer entry that
            # shares a bigger prompt prefix. Pre-v2 the block-prefix lane was
            # OpenCode-compact-only because broad block reuse could restore KV
            # to `matched` while recurrent state stayed at the stored end,
            # visibly degrading answers. kvcache-v2's boundary-true restore
            # closed that class fail-safe (hybrid entries without a stored
            # recurrent boundary at/below the match point are skipped), so the
            # lane is safe for every client. Without it, agent harnesses whose
            # transcripts diverge >8 tokens before the stored end (Pi,
            # little-coder, Hermes, ...) froze on the oldest exact prefix and
            # re-prefilled a growing suffix every turn (issue #138).
            #   - OpenCode-compact keeps its unconditional True (unchanged).
            #   - Other clients: env-decided default (block restore on, the
            #     MTPLX_SESSION_BLOCK_PREFIX_RESTORE=0 kill-switch honored)
            #     while boundary-true restore is enabled; tiny-gap-only when
            #     the boundary-true off-switch restores pre-v2 semantics.
            if _opencode_compact_tool_history_policy(policy_fingerprint):
                allow_block_prefix: bool | None = True
            elif _boundary_true_restore_enabled():
                allow_block_prefix = None
            else:
                allow_block_prefix = False
            tried_larger_near_prefix = True
            near_prompt_state = _restore_near_prefix_prompt_state(
                rt,
                prompt_ids,
                base_hidden_variant=base_hidden_variant,
                mtp_hidden_variant=mtp_hidden_variant,
                mtp_history_policy=mtp_history_policy,
                session_bank=session_bank,
                template_hash=template_hash,
                draft_head_identity=draft_head_identity,
                policy_fingerprint=policy_fingerprint,
                min_restore_tokens=exact_prefix_len,
                allow_block_prefix=allow_block_prefix,
                abort_check=abort_check,
                chunk_callback=prefill_callback,
                chunk_started_s=prefill_started_s,
                matched_ceiling=(
                    vision_restore_spans[0][0]
                    if vision_restore_spans
                    else None
                ),
                cache_factory=restore_cache_factory,
            )
            if near_prompt_state is not None:
                return _emit_prefill_complete(near_prompt_state)

        restore_started = time.perf_counter()
        restored = session_bank.restore(
            rt,
            bank_match_ids,
            mode=effective_restore_mode,
            session_id=session_id,
            hidden_variant=base_hidden_variant,
            template_hash=template_hash,
            mtp_history_policy=mtp_history_policy,
            draft_head_identity=draft_head_identity,
            policy_fingerprint=policy_fingerprint,
            cache_factory=restore_cache_factory,
        )
        restore_elapsed_s = time.perf_counter() - restore_started
        if restored is not None and (
            not _mtp_history_uses_committed_cache(mtp_history_policy)
            or restored.mtp_history_cache is not None
        ):
            _check_postcommit_abort(abort_check)
            suffix = list(prompt_ids[restored.entry.prefix_len :])
            inherited_boundaries = _inherited_gdn_boundaries(
                restored.entry, restored.entry.prefix_len
            )
            exact_served: dict[str, Any] = {
                "entry_prefix_len": int(restored.entry.prefix_len),
                "entry_token_hash": str(
                    getattr(restored.entry, "token_hash", "") or ""
                ),
                "requested_matched": int(restored.entry.prefix_len),
                "actual_restore_point": int(restored.entry.prefix_len),
                "boundary_restore": False,
                "storage_restore_mode": str(restored.restore_mode),
                "lazy_kv": bool(getattr(restored.entry, "lazy_kv", False)),
                "candidate_index": 0,
            }
            _done_at = getattr(restored.entry, "cold_encode_completed_at", None)
            exact_served["encode_completed"] = _done_at is not None
            if _done_at is not None:
                exact_served["encode_completed_age_s"] = round(
                    max(0.0, time.monotonic() - float(_done_at)), 3
                )
            if os.environ.get("MTPLX_DEBUG_PREFIX_DIVERGENCE"):
                print(
                    f"[mtplx] exact-restore: entry_len={restored.entry.prefix_len} "
                    f"entry_boundaries={len(list(getattr(restored.entry, 'gdn_boundaries', None) or []))} "
                    f"inherited={len(inherited_boundaries)} suffix={len(suffix)}",
                    file=sys.stderr,
                    flush=True,
                )
            if not suffix:
                repage_time = _maybe_repage_target_prefill_cache(
                    rt, restored.cache
                )
                return _emit_prefill_complete(PromptState(
                    trunk_cache=restored.cache,
                    logits=restored.logits,
                    hidden=restored.hidden,
                    committed_mtp_cache=restored.mtp_history_cache,
                    token_prefix=tuple(int(token) for token in prompt_ids),
                    prompt_eval_time_s=repage_time,
                    cache_restore_time_s=restore_elapsed_s,
                    mtp_history_policy=mtp_history_policy,
                    mtp_history_window_tokens=mtp_history_window_tokens,
                    cached_tokens=restored.entry.prefix_len,
                    suffix_tokens=0,
                    cache_hit=True,
                    cache_source=getattr(restored, "cache_source", "ram"),
                    ssd_cache_hit=bool(getattr(restored, "ssd_cache_hit", False)),
                    ssd_cached_tokens=int(getattr(restored, "ssd_cached_tokens", 0) or 0),
                    ssd_restore_s=float(getattr(restored, "ssd_restore_s", 0.0) or 0.0),
                    restore_mode=restored.restore_mode,
                    gdn_boundaries=inherited_boundaries,
                    restore_served=exact_served,
                ))

            _check_postcommit_abort(abort_check)
            _emit_prefill_restore_progress(
                prefill_callback,
                tokens_total=len(prompt_ids),
                cached_tokens=restored.entry.prefix_len,
                new_prefill_tokens=len(suffix),
                started_s=prefill_started_s,
                cache_source=getattr(restored, "cache_source", "ram"),
                ssd_cache_hit=bool(getattr(restored, "ssd_cache_hit", False)),
                ssd_cached_tokens=int(
                    getattr(restored, "ssd_cached_tokens", 0) or 0
                ),
                ssd_restore_s=float(getattr(restored, "ssd_restore_s", 0.0) or 0.0),
                ssd_suffix_tokens=len(suffix),
            )
            suffix_boundary_sink: list[tuple[int, Any, Any]] | None = (
                list(inherited_boundaries)
                if session_bank is not None
                and vision_splice is None
                and _gdn_boundary_capture_enabled()
                else None
            )
            if vision_splice is not None:
                # Rows for pads inside the restored prefix are already baked
                # into the restored KV; the suffix consumes strictly after
                # them.
                pad_id = int(vision_splice.image_pad_token_id)
                vision_splice.cursor = sum(
                    1
                    for token in prompt_ids[: restored.entry.prefix_len]
                    if token == pad_id
                )
            suffix_logits, suffix_hidden, suffix_time, mtp_history_time = (
                _prefill_restored_prompt_suffix(
                    rt,
                    restored,
                    suffix,
                    base_hidden_variant=base_hidden_variant,
                    mtp_hidden_variant=mtp_hidden_variant,
                    mtp_history_policy=mtp_history_policy,
                    abort_check=abort_check,
                    chunk_callback=prefill_callback,
                    tokens_total=len(prompt_ids),
                    cached_tokens=restored.entry.prefix_len,
                    chunk_started_s=prefill_started_s,
                    gdn_boundary_sink=suffix_boundary_sink,
                    vision_splice=vision_splice,
                    stable_prefix_len=stable_prefix_len,
                )
            )
            return _emit_prefill_complete(PromptState(
                trunk_cache=restored.cache,
                logits=suffix_logits,
                hidden=suffix_hidden,
                committed_mtp_cache=restored.mtp_history_cache,
                token_prefix=tuple(int(token) for token in prompt_ids),
                prompt_eval_time_s=suffix_time + mtp_history_time,
                prompt_mtp_history_time_s=mtp_history_time,
                cache_restore_time_s=restore_elapsed_s,
                mtp_history_policy=mtp_history_policy,
                mtp_history_window_tokens=mtp_history_window_tokens,
                cached_tokens=restored.entry.prefix_len,
                suffix_tokens=len(suffix),
                cache_hit=True,
                cache_source=getattr(restored, "cache_source", "ram"),
                ssd_cache_hit=bool(getattr(restored, "ssd_cache_hit", False)),
                ssd_cached_tokens=int(getattr(restored, "ssd_cached_tokens", 0) or 0),
                ssd_restore_s=float(getattr(restored, "ssd_restore_s", 0.0) or 0.0),
                restore_mode=restored.restore_mode,
                gdn_boundaries=(
                    suffix_boundary_sink
                    if suffix_boundary_sink is not None
                    else inherited_boundaries
                ),
                restore_served=exact_served,
            ))

        near_prompt_state = _restore_near_prefix_prompt_state(
            rt,
            prompt_ids,
            base_hidden_variant=base_hidden_variant,
            mtp_hidden_variant=mtp_hidden_variant,
            mtp_history_policy=mtp_history_policy,
            session_bank=session_bank,
            template_hash=template_hash,
            draft_head_identity=draft_head_identity,
            policy_fingerprint=policy_fingerprint,
            min_restore_tokens=0 if tried_larger_near_prefix else exact_prefix_len,
            abort_check=abort_check,
            chunk_callback=prefill_callback,
            chunk_started_s=prefill_started_s,
            cache_factory=restore_cache_factory,
            stable_prefix_len=stable_prefix_len,
            matched_ceiling=(
                vision_restore_spans[0][0] if vision_restore_spans else None
            ),
        )
        if near_prompt_state is not None:
            return _emit_prefill_complete(near_prompt_state)

    mtp_history_cache = None
    prompt_history_time = 0.0
    mtp_history_position_base = 1 if mtp_position_mode == "absolute" else 0
    # kvcache-v2: capture interior recurrent boundaries during the cold prefill
    # whenever the result will be banked — they are what make sub-prefix
    # restores on hybrid models exact instead of approximate.
    gdn_boundary_sink: list[tuple[int, Any]] | None = (
        []
        if session_bank is not None
        and vision_splice is None
        and _gdn_boundary_capture_enabled()
        else None
    )
    if _mtp_history_uses_committed_cache(mtp_history_policy):
        if _sustained_prefill_enabled():
            (
                cache,
                logits,
                hidden,
                mtp_history_cache,
                target_time,
                prompt_history_time,
                mtp_history_position_base,
            ) = _prefill_committed_mtp_history_streaming(
                rt,
                prompt_ids,
                base_hidden_variant=base_hidden_variant,
                mtp_hidden_variant=mtp_hidden_variant,
                mtp_position_mode=mtp_position_mode,
                history_window_tokens=(
                    mtp_history_window_tokens
                    if mtp_history_policy == "last_window"
                    else None
                ),
                abort_check=abort_check,
                chunk_callback=prefill_callback,
                cached_tokens=0,
                chunk_started_s=prefill_started_s,
                vision_splice=vision_splice,
                stable_prefix_len=stable_prefix_len,
                gdn_boundary_sink=gdn_boundary_sink,
            )
            prompt_eval_time = target_time + prompt_history_time
        else:
            _assert_safe_long_context_prefill(len(prompt_ids))
            _check_postcommit_abort(abort_check)
            cache, logits, hidden, prompt_hidden, target_time = (
                _prefill_with_hidden_sequence(
                    rt,
                    prompt_ids,
                    hidden_variant=base_hidden_variant,
                    vision_splice=vision_splice,
                )
            )
            _check_postcommit_abort(abort_check)
            prompt_eval_time = target_time
            mtp_history_cache = rt.make_mtp_cache()
            if len(prompt_ids) > 1:
                history_token_ids = prompt_ids[1:]
                history_hidden = prompt_hidden[:, :-1, :]
                history_window_start = 1
                if mtp_history_policy == "last_window":
                    keep = min(len(history_token_ids), mtp_history_window_tokens)
                    dropped = len(history_token_ids) - keep
                    mtp_history_position_base = (
                        dropped + 1 if mtp_position_mode == "absolute" else max(0, dropped)
                    )
                    history_token_ids = history_token_ids[-keep:]
                    history_hidden = history_hidden[:, -keep:, :]
                    history_window_start = 1 + dropped
                history_embeddings = None
                if vision_splice is not None:
                    pad_id = vision_splice.image_pad_token_id
                    rows_before = sum(
                        1 for token in prompt_ids[:history_window_start] if token == pad_id
                    )
                    if any(token == pad_id for token in history_token_ids):
                        from mtplx.vision.splice import spliced_embeddings_for_window

                        history_embeddings = spliced_embeddings_for_window(
                            rt.embed_tokens,
                            mx.array([history_token_ids]),
                            vision_splice,
                            rows_before=rows_before,
                        )
                prompt_history_time = _append_mtp_history(
                    rt,
                    mtp_history_cache,
                    history_hidden,
                    history_token_ids,
                    phase="prefill",
                    mtp_hidden_variant=mtp_hidden_variant,
                    position_offset=(
                        mtp_history_position_base
                        if mtp_position_mode == "absolute" or mtp_history_policy == "last_window"
                        else None
                    ),
                    input_embeddings=history_embeddings,
                )
                prompt_eval_time += prompt_history_time
    else:
        # Only request hidden states from a runtime that can produce them.
        # Target-only AR runtimes (laguna_ar) have no draft head: their
        # forward_ar returns logits alone, so _prefill(return_hidden=True)
        # would unpack a lone logits array as (logits, hidden) and raise
        # "not enough values to unpack (expected 2, got 1)" (the cycle-policy
        # AR snapshot path exposed this once the committed-branch crash was
        # fixed). MTP runtimes still get hidden — the draft head needs it —
        # and this mirrors generate_ar, which gates return_hidden on
        # rt.mtp_enabled. hidden stays None for AR; nothing downstream in the
        # AR path consumes it (the bank stores trunk cache only).
        cache, logits, hidden, target_time = _prefill(
            rt,
            prompt_ids,
            return_hidden=rt.mtp_enabled,
            hidden_variant=base_hidden_variant,
            abort_check=abort_check,
            vision_splice=vision_splice,
            gdn_boundary_sink=gdn_boundary_sink,
            stable_prefix_len=stable_prefix_len,
        )
        prompt_eval_time = target_time
    return _emit_prefill_complete(PromptState(
        trunk_cache=cache,
        logits=logits,
        hidden=hidden,
        committed_mtp_cache=mtp_history_cache,
        token_prefix=tuple(int(token) for token in prompt_ids),
        prompt_eval_time_s=prompt_eval_time,
        prompt_mtp_history_time_s=prompt_history_time,
        mtp_history_policy=mtp_history_policy,
        mtp_history_window_tokens=mtp_history_window_tokens,
        mtp_history_position_base=mtp_history_position_base,
        suffix_tokens=len(prompt_ids),
        cache_miss_reason=getattr(session_bank, "last_miss_reason", None)
        if session_bank is not None
        else None,
        gdn_boundaries=list(gdn_boundary_sink or []),
    ))


def _decode(tokenizer, tokens: list[int]) -> str:
    return tokenizer.decode(tokens)


def _default_stop_tokens(tokenizer) -> set[int]:
    ids: set[int] = set()
    for attr in ("eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int):
            ids.add(value)
    value = getattr(tokenizer, "eos_token_ids", None)
    if isinstance(value, (list, tuple, set)):
        ids.update(int(x) for x in value if isinstance(x, int))
    return ids


def _is_stop(token: int, stop_token_ids: set[int]) -> bool:
    return int(token) in stop_token_ids


def _strip_terminal_stop(tokens: list[int], stop_token_ids: set[int]) -> list[int]:
    stripped = list(tokens)
    while stripped and _is_stop(stripped[-1], stop_token_ids):
        stripped.pop()
    return stripped


def _truncate_after_first_stop(
    tokens: list[int], stop_token_ids: set[int]
) -> list[int]:
    for index, token in enumerate(tokens):
        if _is_stop(token, stop_token_ids):
            return list(tokens[: index + 1])
    return list(tokens)


def _logits_to_numpy(logits: mx.array) -> np.ndarray:
    logits = logits.astype(mx.float32)
    _eval(logits)
    arr = np.asarray(logits, dtype=np.float32).astype(np.float64)
    return arr.reshape(-1)


def _distribution_from_mlx_logits(
    logits: mx.array,
    config: SamplerConfig,
    *,
    token_counts: Mapping[int, int] | None = None,
    penalty_overlay: Mapping[int, float] | None = None,
) -> np.ndarray | SparseDistribution:
    sparse = sparse_distribution_from_mlx_logits(
        logits, config, token_counts=token_counts, penalty_overlay=penalty_overlay
    )
    if sparse is not None:
        return sparse
    return dense_distribution_from_logits(
        _logits_to_numpy(logits),
        config,
        token_counts=token_counts,
        penalty_overlay=penalty_overlay,
    )


def _distributions_from_mlx_logits(
    logits: mx.array,
    config: SamplerConfig,
) -> list[np.ndarray | SparseDistribution] | None:
    sparse = sparse_distributions_from_mlx_logits(logits, config)
    if sparse is not None:
        return list(sparse)
    return None


def _batched_distributions_from_mlx_logits(
    logits: mx.array,
    config: SamplerConfig,
) -> BatchedSparseDistributions | None:
    return batched_sparse_distributions_from_mlx_logits(logits, config)


def _validate_target_prefix_sampler_request(config: SamplerConfig) -> None:
    """Reject an unsupported external target-prefix sampler before prompt work."""
    if (
        config.temperature > 0
        and int(config.top_k or 0) <= 0
        and 0 < config.top_p < 1.0
    ):
        raise RuntimeError(
            "target_prefix verification requires top-k sampling or top_p=1"
        )


def _sample_from_logits(
    logits: mx.array,
    config: SamplerConfig,
    rng: np.random.Generator,
    *,
    token_counts: Mapping[int, int] | None = None,
    penalty_overlay: Mapping[int, float] | None = None,
) -> tuple[int, np.ndarray | SparseDistribution | None]:
    if config.temperature <= 0:
        if penalty_overlay or (
            token_counts and (config.presence_penalty or config.frequency_penalty)
        ):
            logits = apply_penalties_mlx(
                logits.reshape(-1),
                token_counts,
                config.presence_penalty,
                config.frequency_penalty,
                penalty_overlay=penalty_overlay,
            )
        _eval(logits)
        return int(mx.argmax(logits, axis=-1).item()), None
    probs = _distribution_from_mlx_logits(
        logits, config, token_counts=token_counts, penalty_overlay=penalty_overlay
    )
    return sample_from_distribution(probs, rng), probs


def _greedy_draft_token_and_top2(logits: mx.array) -> tuple[int, float, float]:
    """Materialize one greedy token and its FP32 top-two values together."""

    row = (
        logits[:, -1, :][0]
        if logits.ndim == 3
        else logits.reshape(-1)
    ).astype(mx.float32)
    token_id = mx.argmax(row, axis=-1)
    top2_values = mx.topk(row, k=2)
    _eval(token_id, top2_values)
    token = int(np.asarray(token_id).reshape(-1)[0])
    top2 = np.asarray(top2_values, dtype=np.float32).reshape(-1)
    return token, float(top2[-1]), float(top2[-2])


def _sample_draft_from_logits(
    logits: mx.array,
    config: SamplerConfig,
    rng: np.random.Generator,
    *,
    need_distribution: bool,
) -> tuple[int, np.ndarray | SparseDistribution | None]:
    if config.temperature > 0:
        return _sample_from_logits(logits, config, rng)
    _eval(logits)
    token = int(mx.argmax(logits, axis=-1).item())
    if not need_distribution:
        return token, None
    return token, SparseDistribution.one_hot(token, int(logits.shape[-1]))


def _fixed_width_draft_reader(
    draft_logits: mx.array,
    config: SamplerConfig,
    rng: np.random.Generator,
    *,
    need_distribution: bool,
) -> tuple[int, np.ndarray | SparseDistribution | None, bool]:
    token, distribution = _sample_draft_from_logits(
        draft_logits[:, -1, :][0],
        config,
        rng,
        need_distribution=need_distribution,
    )
    return token, distribution, False


def _adaptive_tail_k1_draft_reader(
    draft_logits: mx.array,
    config: SamplerConfig,
    rng: np.random.Generator,
    *,
    need_distribution: bool,
) -> tuple[int, np.ndarray | SparseDistribution | None, bool]:
    token, distribution = _sample_draft_from_logits(
        draft_logits[:, -1, :][0],
        config,
        rng,
        need_distribution=need_distribution,
    )
    return token, distribution, False


def _adaptive_tail_k2_draft_reader(
    draft_logits: mx.array,
    config: SamplerConfig,
    rng: np.random.Generator,
    *,
    need_distribution: bool,
) -> tuple[int, np.ndarray | SparseDistribution | None, bool]:
    token, distribution = _sample_draft_from_logits(
        draft_logits[:, -1, :][0],
        config,
        rng,
        need_distribution=need_distribution,
    )
    return token, distribution, False


def _adaptive_full_k3_draft_reader(
    draft_logits: mx.array,
    config: SamplerConfig,
    rng: np.random.Generator,
    *,
    depth_index: int,
    need_distribution: bool,
    decision_margins: list[float],
    margin_stops: tuple[Callable[[float], bool], Callable[[float], bool]],
) -> tuple[int, np.ndarray | SparseDistribution | None, bool]:
    if depth_index < 2:
        token, top1, top2 = _greedy_draft_token_and_top2(draft_logits)
        margin = float(top1 - top2)
        decision_margins.append(margin)
        distribution = (
            SparseDistribution.one_hot(token, int(draft_logits.shape[-1]))
            if need_distribution
            else None
        )
        return token, distribution, margin_stops[depth_index](margin)
    token, distribution = _sample_draft_from_logits(
        draft_logits[:, -1, :][0],
        config,
        rng,
        need_distribution=need_distribution,
    )
    return token, distribution, False


def _env_scaled_draft_sampler(
    sampler: SamplerConfig,
    draft_sampler: SamplerConfig | None,
) -> SamplerConfig:
    base = draft_sampler or sampler
    raw = os.environ.get("MTPLX_DRAFT_TEMPERATURE_SCALE")
    if raw is None or raw.strip() == "":
        return base
    try:
        scale = float(raw)
    except ValueError:
        return base
    if scale <= 0 or base.temperature <= 0:
        return base
    return SamplerConfig(
        temperature=float(base.temperature) * scale,
        top_p=float(base.top_p),
        top_k=int(base.top_k),
    )


def _sample_adapter_ensemble_q(
    base_logits: mx.array,
    adapter_logits: mx.array,
    *,
    epsilon: float,
    rng: np.random.Generator,
) -> tuple[int, SparseDistribution, dict[str, Any]]:
    """Sample a two-candidate exact proposal q from base and adapter argmaxes."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("adapter ensemble epsilon must be in [0, 1]")
    _eval(base_logits, adapter_logits)
    base_token = int(mx.argmax(base_logits, axis=-1).item())
    adapter_token = int(mx.argmax(adapter_logits, axis=-1).item())
    vocab_size = int(adapter_logits.shape[-1])
    if adapter_token == base_token or epsilon <= 0.0:
        q = SparseDistribution.one_hot(base_token, vocab_size)
        token = base_token
        selected = "shared"
    elif epsilon >= 1.0:
        q = SparseDistribution.one_hot(adapter_token, vocab_size)
        token = adapter_token
        selected = "adapter"
    else:
        q = SparseDistribution(
            np.array([base_token, adapter_token], dtype=np.int64),
            np.array([1.0 - float(epsilon), float(epsilon)], dtype=np.float64),
            vocab_size,
        )
        token = sample_from_distribution(q, rng)
        selected = "adapter" if token == adapter_token else "base"
    return (
        token,
        q,
        {
            "base_token": base_token,
            "adapter_token": adapter_token,
            "epsilon": float(epsilon),
            "changed": bool(adapter_token != base_token),
            "selected": selected,
            "q_token_ids": [int(token_id) for token_id in q.token_ids],
            "q_probs": [float(prob) for prob in q.probs],
        },
    )


def _distribution_argmax(distribution: np.ndarray | SparseDistribution) -> int:
    if isinstance(distribution, SparseDistribution):
        return int(distribution.token_ids[int(np.argmax(distribution.probs))])
    return int(np.argmax(np.asarray(distribution, dtype=np.float64)))


def _online_correction_cache_key(
    policy: str,
    *,
    depth: int,
    primary: int,
    source_token: int,
    draft_prefix: list[int],
) -> tuple[int, ...]:
    if policy == "local_prefix":
        return tuple(
            [int(depth), int(primary), *[int(token) for token in draft_prefix]]
        )
    if policy == "source_token":
        return (int(depth), int(source_token))
    if policy == "primary_source":
        return (int(depth), int(primary), int(source_token))
    raise ValueError(
        "online_correction_cache_key must be one of "
        "'local_prefix', 'source_token', or 'primary_source'"
    )


def _seed_prompt_correction_cache(
    prompt_ids: list[int],
    *,
    max_depth: int,
    min_depth: int,
    key_policy: str,
) -> tuple[dict[tuple[int, ...], int], dict[str, int]]:
    """Seed exact proposal overrides from prompt-local n-gram continuations."""
    if key_policy != "local_prefix":
        return {}, {"stores": 0, "collisions": 0, "skipped": 1}
    seeded: dict[tuple[int, ...], int] = {}
    collisions = 0
    lower = max(1, int(min_depth))
    upper = max(lower, int(max_depth))
    for depth in range(lower, upper + 1):
        if len(prompt_ids) <= depth:
            continue
        for start in range(0, len(prompt_ids) - depth):
            key = tuple(
                [int(depth), int(prompt_ids[start])]
                + [int(token) for token in prompt_ids[start + 1 : start + depth]]
            )
            if key in seeded and seeded[key] != int(prompt_ids[start + depth]):
                collisions += 1
            seeded[key] = int(prompt_ids[start + depth])
    return seeded, {
        "stores": len(seeded),
        "collisions": collisions,
        "skipped": 0,
    }


def _reset_tensor_offset_cache(cache: Any) -> None:
    for entry in cache or []:
        if hasattr(entry, "offset"):
            entry.offset = 0
        if hasattr(entry, "rollback_state"):
            entry.rollback_state = [None, None, None]


def _make_device_d2_draft_core(
    rt: MTPLXRuntime,
    hidden: mx.array,
    token_ids: mx.array,
    *,
    mtp_hidden_variant: str,
) -> dict[str, Any]:
    mtp_cache = rt.make_mtp_cache()
    logits, draft_hidden = rt.draft_mtp(
        hidden,
        token_ids,
        mtp_cache=mtp_cache,
        return_hidden=True,
        mtp_hidden_variant=mtp_hidden_variant,
        mtp_depth=1,
    )
    _eval(logits, draft_hidden)
    promoted, failures = promote_kv_cache_offsets(mtp_cache, reserve_tokens=4)
    _reset_tensor_offset_cache(mtp_cache)

    def draft2_fn(hidden_states, first_token_ids):
        logits1, hidden1 = rt.draft_mtp(
            hidden_states,
            first_token_ids,
            mtp_cache=mtp_cache,
            return_hidden=True,
            mtp_hidden_variant=mtp_hidden_variant,
            mtp_depth=1,
        )
        token1 = mx.argmax(logits1[:, -1, :], axis=-1).reshape(1, 1)
        logits2, _hidden2 = rt.draft_mtp(
            hidden1[:, -1:, :],
            token1,
            mtp_cache=mtp_cache,
            return_hidden=True,
            mtp_hidden_variant=mtp_hidden_variant,
            mtp_depth=2,
        )
        token2 = mx.argmax(logits2[:, -1, :], axis=-1).reshape(1, 1)
        return token1, token2

    compiled = mx.compile(
        draft2_fn,
        inputs=cache_array_tree(mtp_cache),
        outputs=cache_array_tree(mtp_cache),
    )
    smoke = compiled(hidden, token_ids)
    _eval(smoke)
    _reset_tensor_offset_cache(mtp_cache)
    return {
        "fn": compiled,
        "cache": mtp_cache,
        "promoted": promoted,
        "promotion_failures": failures,
    }


def _run_device_d2_draft_core(
    core: dict[str, Any],
    hidden: mx.array,
    primary: int,
) -> list[int]:
    _reset_tensor_offset_cache(core["cache"])
    result = core["fn"](hidden, mx.array([[primary]]))
    _eval(result)
    token1, token2 = result
    return [
        int(token1.reshape(-1)[0].item()),
        int(token2.reshape(-1)[0].item()),
    ]


# Depth-N device draft core ("device"): the whole draft chain — block forward,
# head logits, and the draft sampler itself — compiled as one graph over the
# LIVE cycle cache (the committed history cache under committed policy),
# evaluated with a single sync per cycle. Unlike device-d2 it supports sampled
# drafts: each level reproduces the host sampler's q construction on device
# (temp divide, full-vocab logsumexp, top-k sort, cumulative-before top-p with
# first-token keep, renormalize, inverse-CDF draw) and emits its truncated q
# support, so the acceptance ratio and rejection residual downstream use
# exactly the distribution that proposed. Output exactness therefore holds for
# any q; the rng stream simply lives on device (per-cycle split keys).
_DEVICE_CORE_MAX_TOP_K = 32
_DEVICE_CORE_HISTORY_RESERVE = 4096


def _device_draft_q_arrays(
    row: mx.array,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> tuple[mx.array, mx.array]:
    """On-device mirror of ``sparse_distribution_from_mlx_logits``.

    Returns (sorted top-k token ids, renormalized q probs over that support;
    top-p-dropped entries hold exact zeros). All ops trace under mx.compile.
    """
    flat = row.astype(mx.float32) * (1.0 / float(temperature))
    top_idx = mx.argpartition(-flat, kth=top_k - 1, axis=-1)[:top_k]
    top_vals = flat[top_idx]
    order = mx.argsort(-top_vals, axis=-1)
    top_idx = top_idx[order]
    top_vals = top_vals[order]
    if top_p >= 1.0:
        probs_full = mx.softmax(top_vals, axis=-1)
    else:
        probs_full = mx.exp(top_vals - mx.logsumexp(flat, axis=-1))
    if 0.0 < top_p < 1.0:
        before = mx.cumsum(probs_full, axis=-1) - probs_full
        keep = mx.logical_or(before < top_p, mx.arange(top_k) == 0)
        kept = mx.where(keep, probs_full, mx.zeros_like(probs_full))
    else:
        kept = probs_full
    return top_idx, kept / kept.sum()


def _device_core_state_tree(cache: Any) -> list[Any]:
    """State arrays the compiled draft chain reads and writes.

    Deliberately narrower than ``cache_array_tree``: the rollback_state
    snapshots that eager trims stash between cycles change shape every cycle
    and are never touched inside the chain, so including them would force a
    rebuild per cycle (and did).
    """
    tree: list[Any] = []
    for entry in cache or []:
        if entry is None:
            tree.append(None)
        elif hasattr(entry, "cache"):
            tree.append(entry.cache)
        else:
            tree.append(cache_array_tree([entry]))
    return tree


def _device_core_state_signature(cache: Any) -> tuple[Any, ...]:
    """Structural signature of the cache state a compiled chain depends on.

    mx arrays are immutable, so eager appends between cycles swap the leaf
    array OBJECTS while mx.compile re-reads them through the captured list
    containers — identity changes are routine and harmless. Only the leaf
    SHAPES/dtypes matter: a capacity growth (ensure_capacity swap to a larger
    buffer) changes the traced shapes and requires a rebuild.
    """
    signature: list[Any] = []

    def visit(node: Any) -> None:
        if node is None:
            return
        if hasattr(node, "shape"):
            signature.append((tuple(node.shape), str(node.dtype)))
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                visit(child)
        elif isinstance(node, dict):
            for child in node.values():
                visit(child)

    visit(_device_core_state_tree(cache))
    return tuple(signature)


def _make_device_draft_core(
    rt: MTPLXRuntime,
    hidden: mx.array,
    token_ids: mx.array,
    *,
    mtp_hidden_variant: str,
    depth: int,
    mtp_cache: Any,
    draft_sampler: SamplerConfig,
    seed: int,
) -> dict[str, Any]:
    temperature = float(draft_sampler.temperature)
    top_k = int(draft_sampler.top_k)
    top_p = float(draft_sampler.top_p)
    greedy = temperature <= 0

    base_offset = _mtp_cache_offset(mtp_cache)
    promoted, failures = promote_kv_cache_offsets(
        mtp_cache,
        reserve_tokens=depth + 2,
        initial_reserve_tokens=_DEVICE_CORE_HISTORY_RESERVE,
    )
    # Warm one forward per level so every module and cache view is built
    # before tracing, then trim the warm entries back off the live history.
    warm_hidden, warm_tok = hidden, token_ids
    for level in range(1, depth + 1):
        warm_logits, warm_h = rt.draft_mtp(
            warm_hidden,
            warm_tok,
            mtp_cache=mtp_cache,
            return_hidden=True,
            mtp_hidden_variant=mtp_hidden_variant,
            mtp_depth=level,
        )
        warm_tok = mx.argmax(warm_logits[:, -1, :], axis=-1).reshape(1, 1)
        warm_hidden = warm_h[:, -1:, :]
    _eval(warm_tok, warm_hidden)
    vocab_size = int(warm_logits.shape[-1])
    _rollback_mtp_cache(mtp_cache, base_offset)

    def chain_fn(hidden_states, first_token_ids, level_keys):
        h, tok = hidden_states, first_token_ids
        tokens: list[mx.array] = []
        q_ids: list[mx.array] = []
        q_probs: list[mx.array] = []
        for level in range(1, depth + 1):
            logits_level, hidden_level = rt.draft_mtp(
                h,
                tok,
                mtp_cache=mtp_cache,
                return_hidden=True,
                mtp_hidden_variant=mtp_hidden_variant,
                mtp_depth=level,
            )
            row = logits_level[:, -1, :].reshape(-1)
            if greedy:
                next_tok = mx.argmax(row, axis=-1).reshape(1, 1)
            else:
                top_idx, q_norm = _device_draft_q_arrays(
                    row,
                    temperature=temperature,
                    top_k=min(top_k, vocab_size),
                    top_p=top_p,
                )
                cdf = mx.cumsum(q_norm, axis=-1)
                u = mx.random.uniform(key=level_keys[level - 1])
                pick = mx.minimum(
                    (cdf <= u).sum(), int(top_idx.shape[0]) - 1
                ).astype(mx.int32)
                next_tok = top_idx[pick].reshape(1, 1)
                q_ids.append(top_idx)
                q_probs.append(q_norm)
            tokens.append(next_tok)
            h = hidden_level[:, -1:, :]
            tok = next_tok
        return tuple(tokens + q_ids + q_probs)

    compiled = mx.compile(
        chain_fn,
        inputs=_device_core_state_tree(mtp_cache),
        outputs=_device_core_state_tree(mtp_cache),
    )
    smoke_keys = mx.random.split(mx.random.key(int(seed) & 0x7FFFFFFF), depth)
    smoke = compiled(hidden, token_ids, smoke_keys)
    _eval(smoke)
    _rollback_mtp_cache(mtp_cache, base_offset)
    return {
        "fn": compiled,
        "depth": depth,
        "greedy": greedy,
        "vocab_size": vocab_size,
        "promoted": promoted,
        "promotion_failures": failures,
        "state_signature": _device_core_state_signature(mtp_cache),
    }


def _run_device_draft_core(
    core: dict[str, Any],
    hidden: mx.array,
    primary: int,
    *,
    seed: int,
) -> tuple[list[int], list[SparseDistribution | None]]:
    depth = int(core["depth"])
    level_keys = mx.random.split(mx.random.key(int(seed) & 0x7FFFFFFF), depth)
    result = core["fn"](hidden, mx.array([[primary]]), level_keys)
    _eval(result)
    tokens = [int(t.reshape(-1)[0].item()) for t in result[:depth]]
    if core["greedy"]:
        return tokens, [
            SparseDistribution.one_hot(token, core["vocab_size"]) for token in tokens
        ]
    dists: list[SparseDistribution | None] = []
    for ids, probs in zip(result[depth : 2 * depth], result[2 * depth : 3 * depth]):
        ids_np = np.asarray(ids, dtype=np.int64).reshape(-1)
        probs_np = np.asarray(probs, dtype=np.float64).reshape(-1)
        keep = probs_np > 0
        kept_probs = probs_np[keep]
        dists.append(
            SparseDistribution(
                ids_np[keep],
                kept_probs / kept_probs.sum(),
                core["vocab_size"],
            )
        )
    return tokens, dists


def _draft_confidence_metrics(logits: mx.array, *, topk: int = 8) -> dict[str, float]:
    k = max(2, min(int(topk), int(logits.shape[-1])))
    top_values = mx.topk(logits.astype(mx.float32), k)
    _eval(top_values)
    values = np.sort(np.asarray(top_values, dtype=np.float32).reshape(-1))
    if values.size < 2:
        return {"top2_margin": 0.0, "top1_prob_topk": 1.0, "entropy_topk": 0.0}
    descending = values[::-1].astype(np.float64)
    shifted = descending - float(descending[0])
    exp_values = np.exp(shifted)
    probs = exp_values / float(np.sum(exp_values))
    entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-30))))
    return {
        "top2_margin": float(values[-1] - values[-2]),
        "top1_prob_topk": float(probs[0]),
        "entropy_topk": entropy,
    }


def _top2_margin(logits: mx.array) -> float:
    return _draft_confidence_metrics(logits, topk=2)["top2_margin"]


def _prefill(
    rt: MTPLXRuntime,
    prompt_ids: list[int],
    *,
    return_hidden: bool,
    hidden_variant: str | None = None,
    abort_check: Callable[[], bool] | None = None,
    vision_splice: Any | None = None,
    gdn_boundary_sink: list[tuple[int, Any]] | None = None,
    stable_prefix_len: int | None = None,
):
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")

    _check_postcommit_abort(abort_check)
    cache = _make_target_prefill_cache(rt)
    target_forward_time = 0.0
    final_logits_only = _final_logits_prefill_enabled()
    capture_boundaries = (
        gdn_boundary_sink is not None and _cache_has_recurrent_entries(cache)
    )

    if len(prompt_ids) > 1:
        body = prompt_ids[:-1]
        body_array = mx.array([body])
        _cold_edges: tuple[int, ...] = ()
        if (
            stable_prefix_len is not None
            and capture_boundaries
            and 0 < int(stable_prefix_len) < len(body)
        ):
            _cold_edges = (int(stable_prefix_len),)
        spans = (
            _prefill_spans_with_tail_grid(
                len(body),
                tail_interval=_gdn_boundary_tail_interval(),
                mandatory_edges=_cold_edges,
            )
            if capture_boundaries
            else _iter_prefill_chunk_spans(len(body))
        )
        for start, end in spans:
            _check_postcommit_abort(abort_check)
            chunk_array = body_array[:, start:end]
            chunk_embeddings = None
            if vision_splice is not None:
                from mtplx.vision.splice import spliced_chunk_embeddings

                chunk_embeddings = spliced_chunk_embeddings(
                    rt.embed_tokens, chunk_array, vision_splice
                )
            started = time.perf_counter()
            with attention_phase("prefill"):
                prefill = _prefill_cache_only_forward(
                    rt, chunk_array, cache, input_embeddings=chunk_embeddings
                )
            if prefill is None:
                _eval_cache_roots(cache)
            else:
                _eval(prefill)
            _runtime_count(rt, "prefill_chunks")
            target_forward_time += time.perf_counter() - started
            target_forward_time += _prefill_chunk_cache_cleanup(rt)
            if capture_boundaries:
                _capture_gdn_boundary(gdn_boundary_sink, end, cache)
            _check_postcommit_abort(abort_check)
        if vision_splice is not None and vision_splice.remaining() > 0:
            raise ValueError(
                "vision splice overflow: request supplied more vision rows "
                f"({vision_splice.total_rows}) than image pad tokens in the prompt"
            )

    started = time.perf_counter()
    _check_postcommit_abort(abort_check)
    with attention_phase("prefill"):
        result = rt.forward_ar(
            mx.array([[prompt_ids[-1]]]),
            cache=cache,
            return_hidden=return_hidden,
            hidden_variant=hidden_variant if return_hidden else None,
            emit_logits=True,
            logits_keep=1 if final_logits_only else None,
        )
    if return_hidden:
        logits, hidden = result
        _eval(logits, hidden)
        hidden = hidden[:, -1:, :]
    else:
        logits = result
        hidden = None
        _eval(logits)
    target_forward_time += time.perf_counter() - started
    target_forward_time += _maybe_repage_target_prefill_cache(rt, cache)
    _check_postcommit_abort(abort_check)
    return cache, logits[:, -1, :], hidden, target_forward_time


def _prefill_committed_mtp_history_streaming(
    rt: MTPLXRuntime,
    prompt_ids: list[int],
    *,
    base_hidden_variant: str = "post_norm",
    mtp_hidden_variant: str = "post_norm",
    mtp_position_mode: str = "cache",
    history_window_tokens: int | None = None,
    abort_check: Callable[[], bool] | None = None,
    chunk_callback: Callable[[dict[str, Any]], None] | None = None,
    cached_tokens: int = 0,
    chunk_started_s: float | None = None,
    vision_splice: Any | None = None,
    gdn_boundary_sink: list[tuple[int, Any]] | None = None,
    stable_prefix_len: int | None = None,
):
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")

    _check_postcommit_abort(abort_check)
    cache = _make_target_prefill_cache(rt)
    mtp_history_cache = rt.make_mtp_cache()
    target_forward_time = 0.0
    prompt_history_time = 0.0
    final_logits_only = _final_logits_prefill_enabled()
    capture_boundaries = (
        gdn_boundary_sink is not None and _cache_has_recurrent_entries(cache)
    )
    body = prompt_ids[:-1]
    history_start_token_index = 1
    use_absolute_positions = mtp_position_mode == "absolute"
    mtp_history_position_base = 1 if use_absolute_positions else 0
    if history_window_tokens is not None:
        window = max(1, int(history_window_tokens))
        history_start_token_index = max(1, len(prompt_ids) - window)
        mtp_history_position_base = (
            history_start_token_index if use_absolute_positions else max(0, history_start_token_index - 1)
        )

    cursor = 0
    body_array = mx.array([body]) if body else None
    prompt_array = None
    pad_prefix_counts: list[int] | None = None
    if vision_splice is not None:
        # Prefix counts of image-pad tokens let the (one-token-shifted) MTP
        # history windows read their vision rows at an explicit offset
        # without disturbing the trunk's sequential splice cursor.
        prompt_array = mx.array([prompt_ids])
        pad_prefix_counts = [0]
        pad_id = vision_splice.image_pad_token_id
        for token in prompt_ids:
            pad_prefix_counts.append(
                pad_prefix_counts[-1] + (1 if token == pad_id else 0)
            )
    _cold_edges: tuple[int, ...] = ()
    if (
        stable_prefix_len is not None
        and capture_boundaries
        and 0 < int(stable_prefix_len) < len(body)
    ):
        _cold_edges = (int(stable_prefix_len),)
    mtp_streaming_spans = (
        _prefill_spans_with_tail_grid(
            len(body),
            tail_interval=_gdn_boundary_tail_interval(),
            mandatory_edges=_cold_edges,
        )
        if capture_boundaries
        else _iter_prefill_chunk_spans(len(body))
    )
    for start, end in mtp_streaming_spans:
        _check_postcommit_abort(abort_check)
        chunk_array = body_array[:, start:end]
        chunk_len = end - start
        token_start_index = cursor + 1
        token_end_index = token_start_index + chunk_len
        needs_history_hidden = (
            history_window_tokens is None or token_end_index > history_start_token_index
        )
        chunk_embeddings = None
        if vision_splice is not None:
            from mtplx.vision.splice import spliced_chunk_embeddings

            chunk_embeddings = spliced_chunk_embeddings(
                rt.embed_tokens, chunk_array, vision_splice
            )
        started = time.perf_counter()
        with attention_phase("prefill"):
            if needs_history_hidden:
                logits_chunk, hidden_chunk = rt.forward_ar(
                    chunk_array,
                    cache=cache,
                    return_hidden=True,
                    hidden_variant=base_hidden_variant,
                    emit_logits=not final_logits_only,
                    input_embeddings=chunk_embeddings,
                )
            else:
                hidden_chunk = None
                logits_chunk = _prefill_cache_only_forward(
                    rt, chunk_array, cache, input_embeddings=chunk_embeddings
                )
        if hidden_chunk is None:
            if logits_chunk is None:
                _eval_cache_roots(cache)
            else:
                _eval(logits_chunk)
        elif logits_chunk is None:
            _eval(hidden_chunk)
        else:
            _eval(logits_chunk, hidden_chunk)
        target_forward_time += time.perf_counter() - started
        _runtime_count(rt, "prefill_chunks")
        if chunk_callback is not None:
            try:
                now = time.perf_counter()
                phase_start = chunk_started_s if chunk_started_s is not None else started
                chunk_elapsed = max(0.0, now - started)
                elapsed = max(0.0, now - phase_start)
                tokens_done = int(cursor + chunk_len)
                chunk_tok_s = (
                    float(chunk_len) / chunk_elapsed
                    if chunk_elapsed > 0.0
                    else None
                )
                cumulative_tok_s = (
                    float(tokens_done) / elapsed
                    if elapsed > 0.0 and tokens_done > 0
                    else None
                )
                chunk_callback(
                    {
                        "phase": "chunk",
                        "tokens_done": tokens_done,
                        "tokens_total": int(len(prompt_ids)),
                        "cached_tokens": int(cached_tokens),
                        "elapsed_s": elapsed,
                        "prefill_tok_s": cumulative_tok_s,
                        "cumulative_prefill_tok_s": cumulative_tok_s,
                        "prefill_wall_tok_s": cumulative_tok_s,
                        "live_prefill_tok_s": (
                            chunk_tok_s if chunk_tok_s is not None else cumulative_tok_s
                        ),
                        "chunk_size": int(chunk_len),
                        "chunk_elapsed_s": chunk_elapsed,
                        "chunk_prefill_tok_s": chunk_tok_s,
                    }
                )
            except Exception:
                pass
        _check_postcommit_abort(abort_check)

        if hidden_chunk is not None:
            token_ids = prompt_ids[token_start_index : token_start_index + chunk_len]
            slice_start = max(0, history_start_token_index - token_start_index)
            if slice_start < len(token_ids):
                sliced_token_ids = token_ids[slice_start:]
                sliced_hidden = hidden_chunk[
                    :,
                    slice_start : slice_start + len(sliced_token_ids),
                    :,
                ]
                history_embeddings = None
                if vision_splice is not None and pad_prefix_counts is not None:
                    window_start = token_start_index + slice_start
                    window_end = window_start + len(sliced_token_ids)
                    if (
                        pad_prefix_counts[window_end]
                        > pad_prefix_counts[window_start]
                    ):
                        from mtplx.vision.splice import (
                            spliced_embeddings_for_window,
                        )

                        history_embeddings = spliced_embeddings_for_window(
                            rt.embed_tokens,
                            prompt_array[:, window_start:window_end],
                            vision_splice,
                            rows_before=pad_prefix_counts[window_start],
                        )
                prompt_history_time += _append_mtp_history(
                    rt,
                    mtp_history_cache,
                    sliced_hidden,
                    sliced_token_ids,
                    phase="prefill",
                    mtp_hidden_variant=mtp_hidden_variant,
                    position_offset=(
                        token_start_index + slice_start
                        if use_absolute_positions
                        else token_start_index + slice_start - 1
                        if history_window_tokens is not None
                        else None
                    ),
                    force_eval=True,
                    input_embeddings=history_embeddings,
                )
                _check_postcommit_abort(abort_check)
        cursor += chunk_len
        boundary_hidden = (
            hidden_chunk[:, -1:, :] if hidden_chunk is not None else None
        )
        del hidden_chunk
        del logits_chunk
        target_forward_time += _prefill_chunk_cache_cleanup(rt)
        if capture_boundaries:
            _capture_gdn_boundary(
                gdn_boundary_sink, cursor, cache, hidden_last=boundary_hidden
            )
        del boundary_hidden
        _check_postcommit_abort(abort_check)

    started = time.perf_counter()
    _check_postcommit_abort(abort_check)
    with attention_phase("prefill"):
        logits, hidden = rt.forward_ar(
            mx.array([[prompt_ids[-1]]]),
            cache=cache,
            return_hidden=True,
            hidden_variant=base_hidden_variant,
            emit_logits=True,
            logits_keep=1 if final_logits_only else None,
        )
    _eval(logits, hidden)
    target_forward_time += time.perf_counter() - started
    target_forward_time += _maybe_repage_target_prefill_cache(rt, cache)
    _check_postcommit_abort(abort_check)
    return (
        cache,
        logits[:, -1, :],
        hidden[:, -1:, :],
        mtp_history_cache,
        target_forward_time,
        prompt_history_time,
        mtp_history_position_base,
    )


def _prefill_with_hidden_sequence(
    rt: MTPLXRuntime,
    prompt_ids: list[int],
    *,
    hidden_variant: str,
    vision_splice: Any | None = None,
):
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")

    cache = _make_target_prefill_cache(rt)
    prompt_array = mx.array([prompt_ids])
    prompt_embeddings = None
    if vision_splice is not None:
        from mtplx.vision.splice import spliced_chunk_embeddings

        prompt_embeddings = spliced_chunk_embeddings(
            rt.embed_tokens, prompt_array, vision_splice
        )
    started = time.perf_counter()
    with attention_phase("prefill"):
        logits, hidden = rt.forward_ar(
            prompt_array,
            cache=cache,
            return_hidden=True,
            hidden_variant=hidden_variant,
            emit_logits=True,
            logits_keep=1 if _final_logits_prefill_enabled() else None,
            input_embeddings=prompt_embeddings,
        )
    _eval(logits, hidden)
    target_forward_time = time.perf_counter() - started
    target_forward_time += _maybe_repage_target_prefill_cache(rt, cache)
    return cache, logits[:, -1, :], hidden[:, -1:, :], hidden, target_forward_time


def _mtp_cache_offset(mtp_cache) -> int:
    if not mtp_cache:
        return 0
    return int(getattr(mtp_cache[0], "offset", 0))


def _mtp_position_offset(
    cache_offset: int,
    *,
    mode: str,
    cap: int,
    period: int,
    base: int = 0,
) -> int | None:
    """Map an MTP-cache offset to an explicit draft-side RoPE offset.

    ``None`` preserves the stock MLX behavior where the KV cache owns the
    offset. Non-default modes are proposal-only diagnostics: target verify and
    residual correction remain authoritative.
    """

    normalized = (mode or "default").strip().lower().replace("-", "_")
    offset = max(0, int(cache_offset))
    if normalized in {"", "0", "off", "false", "default", "cache"}:
        return None
    if normalized == "absolute":
        return max(0, int(base)) + offset
    if normalized in {"cap", "capped", "clamp", "clamped"}:
        if cap <= 0:
            return None
        return min(offset, int(cap))
    if normalized in {"mod", "modulo", "wrap", "wrapped"}:
        if period <= 0:
            return None
        anchor = max(0, int(base))
        if offset < anchor:
            return offset
        return anchor + ((offset - anchor) % int(period))
    raise ValueError(f"unknown MTPLX_MTP_POSITION_MODE: {mode!r}")


def _rollback_mtp_cache(mtp_cache, offset: int) -> None:
    if not mtp_cache:
        return
    for cache in mtp_cache:
        current = int(getattr(cache, "offset", 0))
        trim = max(0, current - offset)
        if trim and hasattr(cache, "trim"):
            cache.trim(trim)


def _add_timing(event: dict, key: str, elapsed_s: float) -> None:
    timings = event.setdefault("timing_s", {})
    timings[key] = timings.get(key, 0.0) + elapsed_s


def _reject_repair_breakdown(
    events: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, float]]:
    counts: dict[str, int] = {}
    repair_times: dict[str, float] = {}
    for event in events:
        rejected_at_depth = event.get("rejected_at_depth")
        if rejected_at_depth is None:
            continue
        key = f"reject_depth_{int(rejected_at_depth)}"
        counts[key] = counts.get(key, 0) + 1
        timing = event.get("timing_s", {})
        repair_time = float(timing.get("repair_forward", 0.0))
        if repair_time:
            repair_times[key] = repair_times.get(key, 0.0) + repair_time
    return counts, repair_times


def _mean_accept_probability_by_depth(
    sums: list[float],
    drafted: list[int],
) -> list[float | None]:
    return [
        (float(total) / int(count) if count else None)
        for total, count in zip(sums, drafted)
    ]


def _append_mtp_history(
    rt: MTPLXRuntime,
    mtp_cache,
    hidden_states: mx.array,
    token_ids: list[int],
    *,
    phase: Literal["prefill", "ar_decode"],
    mtp_hidden_variant: str,
    position_offset: int | None = None,
    force_eval: bool = False,
    input_embeddings: mx.array | None = None,
) -> float:
    if not token_ids:
        return 0.0
    if hidden_states.shape[1] != len(token_ids):
        raise ValueError("hidden_states length must match token_ids length")
    if input_embeddings is not None and input_embeddings.shape[1] != len(token_ids):
        raise ValueError("input_embeddings length must match token_ids length")
    _runtime_count(rt, "mtp_history_append_calls")
    started = time.perf_counter()
    with attention_phase(phase):
        hidden = rt.update_mtp_cache(
            hidden_states,
            mx.array([token_ids]),
            mtp_cache=mtp_cache,
            mtp_hidden_variant=mtp_hidden_variant,
            position_offset=position_offset,
            input_embeddings=input_embeddings,
        )
    if _env_truthy("MTPLX_LAZY_MTP_HISTORY_APPEND") and not force_eval:
        return time.perf_counter() - started
    _eval(hidden)
    return time.perf_counter() - started


def generate_ar(
    rt: MTPLXRuntime,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    sampler: SamplerConfig,
    seed: int = 0,
    stop_token_ids: set[int] | None = None,
    token_callback: Callable[[list[int]], None] | None = None,
    trace_label: str | None = None,
    trace_metadata: dict[str, Any] | None = None,
    prefill_callback: Callable[[dict[str, Any]], None] | None = None,
    repetition_stop: bool = False,
    loop_guard: bool = False,
    thinking_guard: ThinkingGuardConfig | None = None,
    constraint: Any | None = None,
) -> GenerationOutput:
    reject_non_k1_a3b_whole_moe_request(rt, entrypoint="generate_ar")
    if getattr(rt, "backend_id", None) == "gemma4_assistant":
        if constraint is not None:
            raise ValueError(
                "constrained decoding is not supported on the gemma4_assistant backend"
            )
        from .backends.gemma4_assistant import generate_gemma4_ar

        return generate_gemma4_ar(
            rt,
            prompt_ids,
            max_tokens=max_tokens,
            sampler=sampler,
            seed=seed,
            stop_token_ids=stop_token_ids,
            token_callback=token_callback,
            trace_label=trace_label,
            trace_metadata=trace_metadata,
            prefill_callback=prefill_callback,
            repetition_stop=repetition_stop,
        )
    counter_start = _runtime_counter_snapshot(rt)
    rng = np.random.default_rng(seed)
    stop_token_ids = (
        _default_stop_tokens(rt.tokenizer) if stop_token_ids is None else stop_token_ids
    )
    started_all = time.perf_counter()
    ar_return_hidden = bool(
        rt.mtp_enabled
        and (
            _env_truthy("MTPLX_AR_RETURN_HIDDEN")
            or _env_truthy("MTPLX_DIAGNOSTIC_AR_RETURN_HIDDEN")
        )
    )
    # Dashboard prefill instrumentation for AR. `_prefill` is unchunked,
    # so we only fire started/completed (no chunk progress).
    prefill_started_s = time.perf_counter()
    if prefill_callback is not None:
        try:
            prefill_callback(
                {
                    "phase": "started",
                    "tokens_done": 0,
                    "tokens_total": int(len(prompt_ids)),
                    "cached_tokens": 0,
                    "new_prefill_tokens": int(len(prompt_ids)),
                    "elapsed_s": 0.0,
                    "started_s": prefill_started_s,
                }
            )
        except Exception:
            pass
    cache, logits, hidden, prompt_eval_time = _prefill(
        rt,
        prompt_ids,
        return_hidden=ar_return_hidden,
    )
    if prefill_callback is not None:
        try:
            elapsed = max(0.0, time.perf_counter() - prefill_started_s)
            tok_s = (
                (len(prompt_ids) / elapsed)
                if elapsed > 0 and prompt_ids
                else None
            )
            prefill_callback(
                {
                    "phase": "completed",
                    "tokens_total": int(len(prompt_ids)),
                    "new_prefill_tokens": int(len(prompt_ids)),
                    "cached_tokens": 0,
                    "elapsed_s": elapsed,
                    "prompt_eval_time_s": elapsed,
                    "prefill_tok_s": tok_s,
                    "prefill_compute_tok_s": tok_s,
                    "prefill_wall_tok_s": tok_s,
                    "cache_hit": False,
                }
            )
        except Exception:
            pass
    tokens: list[int] = []
    events: list[dict] = []
    if constraint is not None:
        # The repetition trimmer retracts committed tokens, which would
        # desync the grammar matcher; constrained output is schema-shaped.
        repetition_stop = False
    repetition_config = _repetition_stop_config(bool(repetition_stop))
    repetition_result: RepetitionStopResult | None = None
    _loop_guard_config = loop_guard_config_from_env(
        bool(loop_guard), tokenizer=getattr(rt, "tokenizer", None)
    )
    _loop_guard = LoopGuard(_loop_guard_config) if _loop_guard_config.enabled else None
    _thinking_guard = (
        ThinkingGuard(thinking_guard)
        if thinking_guard is not None and thinking_guard.enabled
        else None
    )

    def _ar_steer_overlay(working: Sequence[int]) -> dict[int, float] | None:
        merged = (
            _loop_guard.penalties_for(working)
            if _loop_guard is not None and _loop_guard.armed
            else None
        )
        forced = (
            _thinking_guard.overlay_for(working)
            if _thinking_guard is not None and _thinking_guard.steering_active
            else None
        )
        if not forced:
            return merged
        if not merged:
            return forced
        combined = dict(merged)
        for token_id, value in forced.items():
            combined[token_id] = combined.get(token_id, 0.0) + value
        return combined

    target_decode_time = 0.0
    target_forward_graph_time = 0.0
    target_eval_time = 0.0
    verify_calls = 0
    trace = _DecodeTrace(
        prompt_tokens=len(prompt_ids),
        max_tokens=max_tokens,
        speculative_depth=0,
        sampler=sampler,
        verify_strategy="ar",
        verify_core="stock",
        mtp_history_policy="none",
        mtp_cache_policy="none",
        trace_label=trace_label,
        trace_metadata={**(trace_metadata or {}), "generation_mode": "ar"},
    )

    def trace_totals() -> dict[str, Any]:
        return {
            "generated_tokens": len(tokens),
            "accepted_drafts": 0,
            "rejected_drafts": 0,
            "drafted_tokens": 0,
            "verify_calls": verify_calls,
            "correction_tokens": 0,
            "bonus_tokens": 0,
            "verify_time_s": target_decode_time,
            "verify_forward_time_s": target_forward_graph_time,
            "verify_eval_time_s": target_eval_time,
            "verify_logits_eval_time_s": 0.0,
            "verify_hidden_eval_time_s": 0.0,
            "verify_joint_eval_time_s": target_eval_time,
            "verify_target_distribution_time_s": 0.0,
            "target_distribution_materialized_rows": 0,
            "target_distribution_materialized_windows": 0,
            "lazy_bonus_verify_calls": 0,
            "lazy_bonus_commit_time_s": 0.0,
            "verify_eval_unattributed_time_s": 0.0,
            "draft_time_s": 0.0,
            "accept_time_s": 0.0,
            "repair_time_s": 0.0,
            "commit_time_s": 0.0,
            "capture_commit_time_s": 0.0,
            "snapshot_time_s": 0.0,
            "bonus_time_s": 0.0,
            "verify_output_nbytes": 0,
            "draft_output_nbytes": 0,
            "mtp_history_append_nbytes": 0,
            "clear_cache_events": 0,
            "clear_cache_time_s": 0.0,
            "trunk_cache_materialize_events": 0,
            "trunk_cache_materialize_time_s": 0.0,
            "dirty_detach_events": 0,
            "dirty_detach_time_s": 0.0,
            "dirty_detach_arrays": 0,
            "dirty_detach_bytes": 0,
            "live_output_detach_events": 0,
            "live_output_detach_time_s": 0.0,
            "live_output_detach_arrays": 0,
            "live_output_detach_bytes": 0,
            "state_rebase_events": 0,
            "state_rebase_time_s": 0.0,
            "state_root_eval_events": 0,
            "state_root_eval_time_s": 0.0,
            "state_root_eval_arrays": 0,
            "trace_accounting_time_s": 0.0,
            "accepted_by_depth": [],
            "drafted_by_depth": [],
            "accept_probability_sum_by_depth": [],
        }

    def emit_trace(*, force: bool = False, final: bool = False) -> None:
        trace.maybe_emit(
            force=force,
            final=final,
            totals=trace_totals(),
            cache=cache,
            mtp_cache=None,
            mtp_history_materialize_every=0,
            mtp_history_materialize_events=0,
        )

    def emit_token(token: int) -> None:
        if token_callback is not None and not _is_stop(int(token), stop_token_ids):
            token_callback([int(token)])
        emit_trace()

    for step in range(max_tokens):
        if _loop_guard is not None:
            _guard_transition = _loop_guard.observe(tokens)
            if _guard_transition is not None:
                events.append(
                    {
                        "step": step,
                        "loop_guard": {
                            "transition": _guard_transition,
                            "completion_tokens": len(tokens),
                            **_loop_guard.summary(),
                        },
                    }
                )
        if _thinking_guard is not None:
            _tg_transition = _thinking_guard.observe(tokens)
            if _tg_transition is not None:
                events.append(
                    {
                        "step": step,
                        "thinking_guard": {
                            "transition": _tg_transition,
                            "completion_tokens": len(tokens),
                            **_thinking_guard.summary(),
                        },
                    }
                )
        _steer_active = (
            (_loop_guard is not None and _loop_guard.armed)
            or (_thinking_guard is not None and _thinking_guard.steering_active)
        )
        logits_row = logits[0]
        if constraint is not None:
            # Masking precedes every shaping step in _sample_from_logits, so
            # both the greedy and sampled branches draw from the constrained
            # distribution (-inf survives temperature/top-p/penalties).
            logits_row = constraint.mask_logits_row(logits_row)
        token, _ = _sample_from_logits(
            logits_row,
            sampler,
            rng,
            token_counts=Counter(tokens)
            if (sampler.presence_penalty or sampler.frequency_penalty)
            else None,
            penalty_overlay=(_ar_steer_overlay(tokens) if _steer_active else None),
        )
        tokens.append(token)
        emit_token(token)
        events.append({"step": step, "token": token})
        if constraint is not None:
            constraint.advance(token)
            if constraint.stopped and not _is_stop(token, stop_token_ids):
                # The grammar reached its terminal without the model emitting
                # a stop token; end here rather than decode past the document.
                events.append({"step": step, "constraint_stop": True})
                break
        repetition_result = _trim_repeated_suffix(tokens, repetition_config)
        if repetition_result is not None:
            events.append(
                {
                    "step": step,
                    "repetition_stop": {
                        "reason": "exact_repeated_token_suffix",
                        "block_tokens": repetition_result.block_tokens,
                        "repeats": repetition_result.repeats,
                        "trimmed_tokens": repetition_result.repeated_tokens,
                    },
                }
            )
            break
        if step + 1 >= max_tokens or _is_stop(token, stop_token_ids):
            break

        started = time.perf_counter()
        profiler = _ar_forward_profiler(step)
        with attention_phase("ar_decode"):
            if profiler is not None:
                profiler.enable()
            result_next = rt.forward_ar(
                mx.array([[token]]),
                cache=cache,
                return_hidden=ar_return_hidden,
            )
            if profiler is not None:
                profiler.disable()
        if ar_return_hidden:
            logits_next, hidden_next = result_next
        else:
            logits_next = result_next
            hidden_next = None
        forward_graph_elapsed = time.perf_counter() - started
        eval_started = time.perf_counter()
        if hidden_next is None:
            _eval(logits_next)
        else:
            _eval(logits_next, hidden_next)
        eval_elapsed = time.perf_counter() - eval_started
        elapsed_decode = time.perf_counter() - started
        target_decode_time += elapsed_decode
        target_forward_graph_time += forward_graph_elapsed
        target_eval_time += eval_elapsed
        verify_calls += 1
        logits = logits_next[:, -1, :]

    elapsed = time.perf_counter() - started_all
    emit_trace(force=True, final=True)
    stats = GenerationStats(
        mode="ar",
        generated_tokens=len(tokens),
        elapsed_s=elapsed,
        **_generation_rate_fields(
            generated_tokens=len(tokens),
            elapsed_s=elapsed,
            prompt_eval_time_s=prompt_eval_time,
        ),
        target_forward_time_s=prompt_eval_time + target_decode_time,
        prompt_eval_time_s=prompt_eval_time,
        prompt_tps=(
            len(prompt_ids) / prompt_eval_time if prompt_eval_time > 0 else 0.0
        ),
        prompt_target_prefill_time_s=prompt_eval_time,
        prompt_target_prefill_tok_s=(
            len(prompt_ids) / prompt_eval_time if prompt_eval_time > 0 else 0.0
        ),
        verify_time_s=target_decode_time,
        verify_forward_time_s=target_forward_graph_time,
        verify_eval_time_s=target_eval_time,
        verify_joint_eval_time_s=target_eval_time,
        verify_calls=verify_calls,
        peak_memory_bytes=mx.get_peak_memory(),
        repetition_stop_triggered=repetition_result is not None,
        repetition_stop_reason=(
            "exact_repeated_token_suffix" if repetition_result is not None else None
        ),
        repetition_stop_block_tokens=(
            0 if repetition_result is None else repetition_result.block_tokens
        ),
        repetition_stop_repeats=(
            0 if repetition_result is None else repetition_result.repeats
        ),
        repetition_stop_trimmed_tokens=(
            0 if repetition_result is None else repetition_result.repeated_tokens
        ),
        repetition_stop_raw_tokens=(
            0
            if repetition_result is None
            else len(tokens) + repetition_result.repeated_tokens
        ),
        loop_guard=(_loop_guard.summary() if _loop_guard is not None else {}),
        thinking_guard=(
            _thinking_guard.summary() if _thinking_guard is not None else {}
        ),
        decode_trace_path=str(trace.path) if trace.path is not None else None,
        decode_trace_run_id=trace.run_id if trace.enabled else None,
        constraint_active=constraint is not None,
        constraint_completed=(constraint.completed if constraint is not None else None),
        constraint_masked_steps=(
            constraint.masked_steps if constraint is not None else 0
        ),
        constraint_mask_time_s=(
            constraint.mask_time_s if constraint is not None else 0.0
        ),
        events=events,
    )
    _attach_runtime_diagnostics(
        stats,
        rt,
        counter_start,
        ar_return_hidden=ar_return_hidden,
    )
    finish_reason = _finish_reason_from_tokens(
        tokens,
        stop_token_ids=stop_token_ids,
        max_tokens=max_tokens,
    )
    return GenerationOutput(
        tokens=tokens,
        text=_decode(rt.tokenizer, _strip_terminal_stop(tokens, stop_token_ids)),
        stats=stats,
        finish_reason=finish_reason,
    )


def generate_mtp1(
    rt: MTPLXRuntime,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    sampler: SamplerConfig,
    seed: int = 0,
    stop_token_ids: set[int] | None = None,
    draft_sampler: SamplerConfig | None = None,
    verify_strategy: VerifyStrategy = "batched",
    verify_core: str = "stock",
    draft_margin_threshold: float | None = None,
    repetition_stop: bool = False,
) -> GenerationOutput:
    reject_non_k1_a3b_whole_moe_request(rt, entrypoint="generate_mtp1")
    if not rt.mtp_enabled:
        raise RuntimeError("generate_mtp1 requires an MTP-enabled runtime")
    if verify_strategy not in {
        "batched",
        "sequential",
        "capture",
        "capture_commit",
        "graphbank",
        "graphbank_capture_commit",
    }:
        raise ValueError(
            "verify_strategy must be 'batched', 'sequential', 'capture', "
            "'capture_commit', 'graphbank', or 'graphbank_capture_commit'"
        )
    counter_start = _runtime_counter_snapshot(rt)
    verify_core_backend = resolve_gdn_capture_backend(verify_core)

    rng = np.random.default_rng(seed)
    draft_sampler = _env_scaled_draft_sampler(sampler, draft_sampler)
    stop_token_ids = (
        _default_stop_tokens(rt.tokenizer) if stop_token_ids is None else stop_token_ids
    )
    started_all = time.perf_counter()
    cache, logits, hidden, target_time = _prefill(rt, prompt_ids, return_hidden=True)
    prompt_eval_time = target_time
    graphbank = (
        SpecDecodeGraphBank(rt, capture_backend=verify_core_backend)
        if verify_strategy in {"graphbank", "graphbank_capture_commit"}
        else None
    )
    tokens: list[int] = []
    events: list[dict] = []
    repetition_config = _repetition_stop_config(bool(repetition_stop))
    repetition_result: RepetitionStopResult | None = None
    accepted = rejected = drafted = 0
    skipped = 0
    draft_time = verify_time = 0.0
    verify_forward_time = 0.0
    verify_eval_time = 0.0
    snapshot_time = accept_time = rollback_time = repair_time = 0.0
    commit_time = capture_commit_time = 0.0
    bonus_time = 0.0
    bonus_tokens = correction_tokens = verify_calls = 0
    accept_probability_sum_by_depth = [0.0]
    deferred_correction_repairs = 0
    pending_primary: int | None = None

    step = 0
    while len(tokens) < max_tokens:
        repetition_result = _trim_repeated_suffix(tokens, repetition_config)
        if repetition_result is not None:
            events.append(
                {
                    "step": step,
                    "repetition_stop": {
                        "reason": "exact_repeated_token_suffix",
                        "block_tokens": repetition_result.block_tokens,
                        "repeats": repetition_result.repeats,
                        "trimmed_tokens": repetition_result.repeated_tokens,
                    },
                }
            )
            break
        primary_already_emitted = pending_primary is not None
        if pending_primary is None:
            primary, _ = _sample_from_logits(logits[0], sampler, rng)
            tokens.append(primary)
        else:
            primary = pending_primary
            pending_primary = None
        event = {
            "step": step,
            "primary": primary,
            "accepted": None,
            "primary_already_emitted": primary_already_emitted,
            "verify_core": verify_core_backend.replace("_", "-"),
        }
        step += 1
        if len(tokens) >= max_tokens or _is_stop(primary, stop_token_ids):
            events.append(event)
            break

        started = time.perf_counter()
        draft_logits = rt.draft_mtp(
            hidden,
            mx.array([[primary]]),
            mtp_cache=rt.make_mtp_cache(),
        )
        draft_timed = False
        elapsed_draft = 0.0
        if draft_margin_threshold is not None or draft_sampler.temperature <= 0:
            _eval(draft_logits)
            elapsed_draft = time.perf_counter() - started
            draft_time += elapsed_draft
            draft_timed = True
        if draft_margin_threshold is not None:
            margin = _top2_margin(draft_logits[:, -1, :][0])
            event["top2_margin"] = margin
            if margin < draft_margin_threshold:
                _add_timing(event, "draft", elapsed_draft)
                skipped += 1
                event["accepted"] = None
                event["speculation_skipped"] = True
                event["verify_strategy"] = verify_strategy
                started = time.perf_counter()
                with attention_phase("decode_verify"):
                    logits_next, hidden_next = rt.forward_ar(
                        mx.array([[primary]]),
                        cache=cache,
                        return_hidden=True,
                    )
                _eval(logits_next, hidden_next)
                elapsed_commit = time.perf_counter() - started
                target_time += elapsed_commit
                commit_time += elapsed_commit
                _add_timing(event, "skip_forward", elapsed_commit)
                logits = logits_next[:, -1, :]
                hidden = hidden_next[:, -1:, :]
                events.append(event)
                continue
        draft_token, draft_q = _sample_draft_from_logits(
            draft_logits[:, -1, :][0],
            draft_sampler,
            rng,
            need_distribution=sampler.temperature > 0,
        )
        if not draft_timed:
            elapsed_draft = time.perf_counter() - started
            draft_time += elapsed_draft
            _add_timing(event, "draft", elapsed_draft)
        else:
            _add_timing(event, "draft", elapsed_draft)
        drafted += 1
        event["draft"] = draft_token

        if verify_strategy == "sequential":
            started = time.perf_counter()
            with attention_phase("decode_verify"):
                verify_logits, verify_hidden = rt.forward_ar(
                    mx.array([[primary]]),
                    cache=cache,
                    return_hidden=True,
                )
            _eval(verify_logits, verify_hidden)
            elapsed_verify = time.perf_counter() - started
            verify_time += elapsed_verify
            target_time += elapsed_verify
            verify_calls += 1

            target_logits_for_draft = verify_logits[:, -1, :]
            started_accept = time.perf_counter()
            if sampler.temperature <= 0:
                target_token = int(
                    mx.argmax(target_logits_for_draft[0], axis=-1).item()
                )
                accepted_now = draft_token == target_token
                correction = target_token
                accept_probability = 1.0 if accepted_now else 0.0
            else:
                target_p = _distribution_from_mlx_logits(
                    target_logits_for_draft[0], sampler
                )
                if draft_q is None:
                    raise RuntimeError("non-greedy MTP requires a draft distribution")
                accept_prob = compute_acceptance_probability(
                    target_p,
                    draft_q,
                    draft_token,
                )
                accepted_now = float(rng.random()) <= accept_prob
                correction = (
                    draft_token
                    if accepted_now
                    else sample_from_distribution(
                        residual_distribution(target_p, draft_q), rng
                    )
                )
            elapsed_accept = time.perf_counter() - started_accept
            accept_time += elapsed_accept
            _add_timing(event, "accept", elapsed_accept)

            event["accepted"] = accepted_now
            event["accept_probability"] = float(
                accept_prob if sampler.temperature > 0 else accept_probability
            )
            event["correction"] = int(correction)
            event["verify_strategy"] = verify_strategy
            accept_probability_sum_by_depth[0] += float(event["accept_probability"])

            if accepted_now:
                accepted += 1
                tokens.append(draft_token)
                started = time.perf_counter()
                with attention_phase("decode_verify"):
                    logits_next, hidden_next = rt.forward_ar(
                        mx.array([[draft_token]]),
                        cache=cache,
                        return_hidden=True,
                    )
                _eval(logits_next, hidden_next)
                elapsed_commit = time.perf_counter() - started
                verify_time += elapsed_commit
                target_time += elapsed_commit
                commit_time += elapsed_commit
                _add_timing(event, "commit_forward", elapsed_commit)
                logits = logits_next[:, -1, :]
                hidden = hidden_next[:, -1:, :]
                if _is_stop(draft_token, stop_token_ids):
                    events.append(event)
                    break
            elif sampler.temperature <= 0:
                rejected += 1
                logits = verify_logits[:, -1, :]
                hidden = verify_hidden[:, -1:, :]
            else:
                rejected += 1
                correction_tokens += 1
                tokens.append(int(correction))
                started = time.perf_counter()
                with attention_phase("decode_verify"):
                    logits_next, hidden_next = rt.forward_ar(
                        mx.array([[int(correction)]]),
                        cache=cache,
                        return_hidden=True,
                    )
                _eval(logits_next, hidden_next)
                elapsed_repair = time.perf_counter() - started
                target_time += elapsed_repair
                repair_time += elapsed_repair
                _add_timing(event, "repair_forward", elapsed_repair)
                logits = logits_next[:, -1, :]
                hidden = hidden_next[:, -1:, :]
                if _is_stop(int(correction), stop_token_ids):
                    events.append(event)
                    break

            events.append(event)
            continue

        started = time.perf_counter()
        before_verify = snapshot_untrimmable_cache(cache)
        elapsed_snapshot = time.perf_counter() - started
        snapshot_time += elapsed_snapshot
        _add_timing(event, "snapshot", elapsed_snapshot)
        captures = None
        if verify_strategy in {"capture", "capture_commit", "graphbank_capture_commit"}:
            started = time.perf_counter()
            with attention_phase("decode_verify"):
                if graphbank is not None:
                    verify_logits, verify_hidden, captures = (
                        graphbank.forward_ar_capture(
                            mx.array([[primary, draft_token]]),
                            cache=cache,
                            return_hidden=True,
                        )
                    )
                else:
                    verify_logits, verify_hidden, captures = rt.forward_ar_capture(
                        mx.array([[primary, draft_token]]),
                        cache=cache,
                        return_hidden=True,
                        capture_backend=verify_core_backend,
                    )
            _eval_verify_outputs(verify_logits, verify_hidden, captures)
            elapsed_verify = time.perf_counter() - started
            verify_time += elapsed_verify
            target_time += elapsed_verify
            verify_calls += 1
            if graphbank is not None:
                event["graphbank"] = graphbank.to_dict()

            target_logits_for_draft = verify_logits[:, 0, :]
            started_accept = time.perf_counter()
            if sampler.temperature <= 0:
                target_token = int(
                    mx.argmax(target_logits_for_draft[0], axis=-1).item()
                )
                accepted_now = draft_token == target_token
                correction = target_token
                accept_probability = 1.0 if accepted_now else 0.0
            else:
                target_p = _distribution_from_mlx_logits(
                    target_logits_for_draft[0], sampler
                )
                if draft_q is None:
                    raise RuntimeError("non-greedy MTP requires a draft distribution")
                accept_prob = compute_acceptance_probability(
                    target_p,
                    draft_q,
                    draft_token,
                )
                accepted_now = float(rng.random()) <= accept_prob
                correction = (
                    draft_token
                    if accepted_now
                    else sample_from_distribution(
                        residual_distribution(target_p, draft_q), rng
                    )
                )
            elapsed_accept = time.perf_counter() - started_accept
            accept_time += elapsed_accept
            _add_timing(event, "accept", elapsed_accept)

            event["accepted"] = accepted_now
            event["accept_probability"] = float(
                accept_prob if sampler.temperature > 0 else accept_probability
            )
            event["correction"] = int(correction)
            event["verify_strategy"] = verify_strategy
            accept_probability_sum_by_depth[0] += float(event["accept_probability"])

            if accepted_now:
                accepted += 1
                tokens.append(draft_token)
                logits = verify_logits[:, 1, :]
                hidden = verify_hidden[:, -1:, :]
                if _is_stop(draft_token, stop_token_ids):
                    events.append(event)
                    break
                if len(tokens) < max_tokens:
                    started_bonus = time.perf_counter()
                    bonus, _ = _sample_from_logits(logits[0], sampler, rng)
                    elapsed_bonus = time.perf_counter() - started_bonus
                    bonus_time += elapsed_bonus
                    _add_timing(event, "bonus_sample", elapsed_bonus)
                    tokens.append(bonus)
                    pending_primary = bonus
                    bonus_tokens += 1
                    event["bonus_token"] = int(bonus)
                    if _is_stop(bonus, stop_token_ids):
                        events.append(event)
                        break
            else:
                rejected += 1
                if sampler.temperature <= 0:
                    committed = False
                    if verify_strategy in {
                        "capture_commit",
                        "graphbank_capture_commit",
                    }:
                        from .gdn_capture import commit_captured_prefix

                        started_commit = time.perf_counter()
                        committed = commit_captured_prefix(
                            cache,
                            captures,
                            keep_tokens=1,
                            verified_tokens=2,
                        )
                        elapsed_commit = time.perf_counter() - started_commit
                        capture_commit_time += elapsed_commit
                        _add_timing(event, "capture_commit", elapsed_commit)
                    if committed:
                        logits = verify_logits[:, 0, :]
                        hidden = verify_hidden[:, 0:1, :]
                        event["capture_repair"] = "captured_primary_commit"
                    else:
                        started_rollback = time.perf_counter()
                        rollback_after_verify(cache, before_verify, verified_tokens=2)
                        elapsed_rollback = time.perf_counter() - started_rollback
                        rollback_time += elapsed_rollback
                        _add_timing(event, "rollback", elapsed_rollback)
                        started = time.perf_counter()
                        with attention_phase("decode_verify"):
                            logits_next, hidden_next = rt.forward_ar(
                                mx.array([[primary]]),
                                cache=cache,
                                return_hidden=True,
                            )
                        _eval(logits_next, hidden_next)
                        elapsed_repair = time.perf_counter() - started
                        target_time += elapsed_repair
                        repair_time += elapsed_repair
                        _add_timing(event, "repair_forward", elapsed_repair)
                        logits = logits_next[:, -1, :]
                        hidden = hidden_next[:, -1:, :]
                        event["capture_repair"] = "standard_primary_reforward"
                else:
                    correction_tokens += 1
                    tokens.append(int(correction))
                    committed = False
                    if verify_strategy in {
                        "capture_commit",
                        "graphbank_capture_commit",
                    }:
                        from .gdn_capture import commit_captured_prefix

                        started_commit = time.perf_counter()
                        committed = commit_captured_prefix(
                            cache,
                            captures,
                            keep_tokens=1,
                            verified_tokens=2,
                        )
                        elapsed_commit = time.perf_counter() - started_commit
                        capture_commit_time += elapsed_commit
                        _add_timing(event, "capture_commit", elapsed_commit)
                    if not committed:
                        started_rollback = time.perf_counter()
                        rollback_after_verify(cache, before_verify, verified_tokens=2)
                        elapsed_rollback = time.perf_counter() - started_rollback
                        rollback_time += elapsed_rollback
                        _add_timing(event, "rollback", elapsed_rollback)
                    if committed:
                        logits = verify_logits[:, 0, :]
                        hidden = verify_hidden[:, 0:1, :]
                        pending_primary = int(correction)
                        deferred_correction_repairs += 1
                        event["capture_repair"] = "captured_primary_pending_correction"
                        event["pending_primary"] = int(correction)
                    else:
                        started = time.perf_counter()
                        with attention_phase("decode_verify"):
                            logits_next, hidden_next = rt.forward_ar(
                                mx.array([[primary, int(correction)]]),
                                cache=cache,
                                return_hidden=True,
                            )
                        event["capture_repair"] = (
                            "standard_primary_correction_reforward"
                        )
                        _eval(logits_next, hidden_next)
                        elapsed_repair = time.perf_counter() - started
                        target_time += elapsed_repair
                        repair_time += elapsed_repair
                        _add_timing(event, "repair_forward", elapsed_repair)
                        logits = logits_next[:, -1, :]
                        hidden = hidden_next[:, -1:, :]
                    if _is_stop(int(correction), stop_token_ids):
                        events.append(event)
                        break

            events.append(event)
            continue

        started = time.perf_counter()
        with (
            attention_phase("decode_verify"),
            model_forward_kind("target_verify"),
        ):
            if graphbank is not None:
                verify_logits, verify_hidden = graphbank.forward_ar(
                    mx.array([[primary, draft_token]]),
                    cache=cache,
                    return_hidden=True,
                )
            else:
                verify_logits, verify_hidden = rt.forward_ar(
                    mx.array([[primary, draft_token]]),
                    cache=cache,
                    return_hidden=True,
                )
        if captures is not None:
            _eval_verify_outputs(verify_logits, verify_hidden, captures)
        else:
            _eval_verify_outputs(verify_logits, verify_hidden)
        elapsed_verify = time.perf_counter() - started
        verify_time += elapsed_verify
        target_time += elapsed_verify
        verify_calls += 1
        if graphbank is not None:
            event["graphbank"] = graphbank.to_dict()

        target_logits_for_draft = verify_logits[:, 0, :]
        started_accept = time.perf_counter()
        if sampler.temperature <= 0:
            target_token = int(mx.argmax(target_logits_for_draft[0], axis=-1).item())
            accepted_now = draft_token == target_token
            correction = target_token
            accept_probability = 1.0 if accepted_now else 0.0
        else:
            target_p = _distribution_from_mlx_logits(
                target_logits_for_draft[0], sampler
            )
            if draft_q is None:
                raise RuntimeError("non-greedy MTP requires a draft distribution")
            accept_prob = compute_acceptance_probability(
                target_p,
                draft_q,
                draft_token,
            )
            accepted_now = float(rng.random()) <= accept_prob
            correction = (
                draft_token
                if accepted_now
                else sample_from_distribution(
                    residual_distribution(target_p, draft_q), rng
                )
            )
        elapsed_accept = time.perf_counter() - started_accept
        accept_time += elapsed_accept
        _add_timing(event, "accept", elapsed_accept)

        event["accepted"] = accepted_now
        event["accept_probability"] = float(
            accept_prob if sampler.temperature > 0 else accept_probability
        )
        event["correction"] = int(correction)
        event["verify_strategy"] = verify_strategy
        accept_probability_sum_by_depth[0] += float(event["accept_probability"])

        if accepted_now:
            accepted += 1
            tokens.append(draft_token)
            logits = verify_logits[:, 1, :]
            hidden = verify_hidden[:, -1:, :]
            if _is_stop(draft_token, stop_token_ids):
                events.append(event)
                break
            if len(tokens) < max_tokens:
                started_bonus = time.perf_counter()
                bonus, _ = _sample_from_logits(logits[0], sampler, rng)
                elapsed_bonus = time.perf_counter() - started_bonus
                bonus_time += elapsed_bonus
                _add_timing(event, "bonus_sample", elapsed_bonus)
                tokens.append(bonus)
                pending_primary = bonus
                bonus_tokens += 1
                event["bonus_token"] = int(bonus)
                if _is_stop(bonus, stop_token_ids):
                    events.append(event)
                    break
        elif sampler.temperature <= 0:
            rejected += 1
            started_rollback = time.perf_counter()
            rollback_after_verify(cache, before_verify, verified_tokens=2)
            elapsed_rollback = time.perf_counter() - started_rollback
            rollback_time += elapsed_rollback
            _add_timing(event, "rollback", elapsed_rollback)
            started = time.perf_counter()
            with (
                attention_phase("decode_verify"),
                model_forward_kind("repair"),
            ):
                logits_next, hidden_next = rt.forward_ar(
                    mx.array([[primary]]),
                    cache=cache,
                    return_hidden=True,
                )
            _eval(logits_next, hidden_next)
            elapsed_repair = time.perf_counter() - started
            target_time += elapsed_repair
            repair_time += elapsed_repair
            _add_timing(event, "repair_forward", elapsed_repair)
            logits = logits_next[:, -1, :]
            hidden = hidden_next[:, -1:, :]
        else:
            rejected += 1
            correction_tokens += 1
            tokens.append(int(correction))
            started_rollback = time.perf_counter()
            rollback_after_verify(cache, before_verify, verified_tokens=2)
            elapsed_rollback = time.perf_counter() - started_rollback
            rollback_time += elapsed_rollback
            _add_timing(event, "rollback", elapsed_rollback)
            started = time.perf_counter()
            with attention_phase("decode_verify"):
                logits_next, hidden_next = rt.forward_ar(
                    mx.array([[primary, int(correction)]]),
                    cache=cache,
                    return_hidden=True,
                )
            _eval(logits_next, hidden_next)
            elapsed_repair = time.perf_counter() - started
            target_time += elapsed_repair
            repair_time += elapsed_repair
            _add_timing(event, "repair_forward", elapsed_repair)
            logits = logits_next[:, -1, :]
            hidden = hidden_next[:, -1:, :]
            if _is_stop(int(correction), stop_token_ids):
                events.append(event)
                break

        events.append(event)

    elapsed = time.perf_counter() - started_all
    reject_path_counts, repair_time_by_reject_depth = _reject_repair_breakdown(events)
    stats = GenerationStats(
        mode="mtp1",
        generated_tokens=len(tokens),
        elapsed_s=elapsed,
        **_generation_rate_fields(
            generated_tokens=len(tokens),
            elapsed_s=elapsed,
            prompt_eval_time_s=prompt_eval_time,
        ),
        accepted_drafts=accepted,
        rejected_drafts=rejected,
        drafted_tokens=drafted,
        skipped_drafts=skipped,
        verify_time_s=verify_time,
        verify_forward_time_s=verify_forward_time,
        verify_eval_time_s=verify_eval_time,
        draft_time_s=draft_time,
        target_forward_time_s=target_time,
        prompt_eval_time_s=prompt_eval_time,
        prompt_tps=(
            len(prompt_ids) / prompt_eval_time if prompt_eval_time > 0 else 0.0
        ),
        prompt_target_prefill_time_s=prompt_eval_time,
        prompt_target_prefill_tok_s=(
            len(prompt_ids) / prompt_eval_time if prompt_eval_time > 0 else 0.0
        ),
        snapshot_time_s=snapshot_time,
        accept_time_s=accept_time,
        rollback_time_s=rollback_time,
        repair_time_s=repair_time,
        commit_time_s=commit_time,
        capture_commit_time_s=capture_commit_time,
        bonus_time_s=bonus_time,
        peak_memory_bytes=mx.get_peak_memory(),
        bonus_tokens=bonus_tokens,
        correction_tokens=correction_tokens,
        verify_calls=verify_calls,
        accepted_by_depth=[accepted],
        drafted_by_depth=[drafted],
        accept_probability_sum_by_depth=accept_probability_sum_by_depth,
        mean_accept_probability_by_depth=_mean_accept_probability_by_depth(
            accept_probability_sum_by_depth,
            [drafted],
        ),
        graphbank=graphbank.to_dict() if graphbank is not None else {},
        reject_path_counts=reject_path_counts,
        repair_time_by_reject_depth_s=repair_time_by_reject_depth,
        deferred_correction_repairs=deferred_correction_repairs,
        repetition_stop_triggered=repetition_result is not None,
        repetition_stop_reason=(
            "exact_repeated_token_suffix" if repetition_result is not None else None
        ),
        repetition_stop_block_tokens=(
            0 if repetition_result is None else repetition_result.block_tokens
        ),
        repetition_stop_repeats=(
            0 if repetition_result is None else repetition_result.repeats
        ),
        repetition_stop_trimmed_tokens=(
            0 if repetition_result is None else repetition_result.repeated_tokens
        ),
        repetition_stop_raw_tokens=(
            0
            if repetition_result is None
            else len(tokens) + repetition_result.repeated_tokens
        ),
        events=events,
    )
    _attach_runtime_diagnostics(stats, rt, counter_start)
    finish_reason = (
        "stop"
        if repetition_result is not None
        else _finish_reason_from_tokens(
            tokens,
            stop_token_ids=stop_token_ids,
            max_tokens=max_tokens,
        )
    )
    return GenerationOutput(
        tokens=tokens,
        text=_decode(rt.tokenizer, _strip_terminal_stop(tokens, stop_token_ids)),
        stats=stats,
        finish_reason=finish_reason,
    )


def generate_mtpk(
    rt: MTPLXRuntime,
    prompt_ids: list[int],
    *,
    abort_check: Callable[[], bool] | None = None,
    max_tokens: int,
    sampler: SamplerConfig,
    speculative_depth: int,
    seed: int = 0,
    stop_token_ids: set[int] | None = None,
    base_hidden_variant: str | None = None,
    mtp_hidden_variant: str | None = None,
    mtp_cache_policy: str = "persistent",
    mtp_history_policy: str = "cycle",
    draft_sampler: SamplerConfig | None = None,
    draft_margin_threshold: float | None = None,
    min_speculative_depth: int = 1,
    verify_strategy: VerifyStrategy = "batched",
    verify_core: str = "stock",
    draft_core: str = "stock",
    mtp_corrector: Any | None = None,
    adaptive_policy: AdaptiveDepthPolicy | ExpectedValueDepthPolicy | None = None,
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
    mtp_topk_reranker: Any | None = None,
    token_callback: Callable[[list[int]], None] | None = None,
    session_bank: Any | None = None,
    session_id: str | None = None,
    session_restore_mode: str = "clone",
    session_template_hash: str | None = None,
    session_draft_head_identity: str | None = None,
    session_policy_fingerprint: str | None = None,
    capture_final_state: bool = False,
    commit_prompt_state_to_bank: bool = False,
    commit_prompt_state_keep_live_ref: bool = False,
    trace_label: str | None = None,
    trace_metadata: dict[str, Any] | None = None,
    prefill_callback: Callable[[dict[str, Any]], None] | None = None,
    repetition_stop: bool = False,
    loop_guard: bool = False,
    thinking_guard: ThinkingGuardConfig | None = None,
    vision_splice: Any | None = None,
    constraint: Any | None = None,
    adaptive_width_policy: Any | None = None,
) -> GenerationOutput:
    """Generate with a fixed native-MTP depth.

    The implementation is deliberately conservative: every reject restores the
    target cache snapshot and re-forwards only the committed prefix. This keeps
    the hybrid GDN/attention cache contract exact while we measure depth.
    """
    if getattr(rt, "backend_id", None) == "gemma4_assistant":
        from .backends.gemma4_assistant import generate_gemma4_assistant

        runtime_block_size = int(getattr(getattr(rt, "config", None), "draft_block_size", 0) or 0)
        requested_block_size = int(speculative_depth or 0)
        effective_block_size = (
            runtime_block_size
            if requested_block_size in {0, 3} and runtime_block_size > 0
            else requested_block_size
        )
        return generate_gemma4_assistant(
            rt,
            prompt_ids,
            max_tokens=max_tokens,
            sampler=sampler,
            draft_sampler=draft_sampler,
            speculative_depth=effective_block_size,
            seed=seed,
            stop_token_ids=stop_token_ids,
            token_callback=token_callback,
            session_bank=session_bank,
            session_restore_mode=session_restore_mode,
            session_template_hash=session_template_hash,
            session_draft_head_identity=session_draft_head_identity,
            session_policy_fingerprint=session_policy_fingerprint,
            capture_final_state=capture_final_state,
            trace_label=trace_label,
            trace_metadata=trace_metadata,
            prefill_callback=prefill_callback,
            repetition_stop=repetition_stop,
            requested_speculative_depth=requested_block_size,
        )
    if getattr(rt, "backend_id", None) == "dflash":
        from .backends.dflash import generate_dflash

        return generate_dflash(
            rt,
            prompt_ids,
            max_tokens=max_tokens,
            sampler=sampler,
            speculative_depth=int(speculative_depth or 0),
            stop_token_ids=stop_token_ids,
            token_callback=token_callback,
            seed=seed,
        )
    if not rt.mtp_enabled:
        raise RuntimeError("generate_mtpk requires an MTP-enabled runtime")
    base_hidden_variant = _resolve_runtime_base_hidden_variant(rt, base_hidden_variant)
    mtp_hidden_variant = _resolve_runtime_mtp_hidden_variant(rt, mtp_hidden_variant)
    requested_speculative_depth = int(speculative_depth)
    if requested_speculative_depth < 1:
        raise ValueError("speculative_depth must be >= 1")
    if min_speculative_depth < 0:
        raise ValueError("min_speculative_depth must be >= 0")
    if min_speculative_depth > requested_speculative_depth:
        raise ValueError("min_speculative_depth cannot exceed speculative_depth")
    speculative_depth, long_context_depth_policy = resolve_long_context_mtp_depth(
        prompt_tokens=len(prompt_ids),
        requested_depth=requested_speculative_depth,
        min_depth=min_speculative_depth,
    )
    if min_speculative_depth > speculative_depth:
        raise ValueError("min_speculative_depth cannot exceed speculative_depth")
    if mtp_cache_policy not in {"persistent", "fresh"}:
        raise ValueError("mtp_cache_policy must be 'persistent' or 'fresh'")
    mtp_history_policy = _normalize_mtp_history_policy(mtp_history_policy)
    if online_hidden_corrector_alpha < 0:
        raise ValueError("online_hidden_corrector_alpha must be >= 0")
    if not 0 <= online_hidden_corrector_decay < 1:
        raise ValueError("online_hidden_corrector_decay must be in [0, 1)")
    if online_hidden_corrector_warmup < 0:
        raise ValueError("online_hidden_corrector_warmup must be >= 0")
    if (
        online_hidden_corrector_max_feed_depth is not None
        and online_hidden_corrector_max_feed_depth < 1
    ):
        raise ValueError("online_hidden_corrector_max_feed_depth must be >= 1")
    if online_hidden_corrector_key not in {"global", "token"}:
        raise ValueError("online_hidden_corrector_key must be 'global' or 'token'")
    if online_correction_cache_min_depth < 1:
        raise ValueError("online_correction_cache_min_depth must be >= 1")
    if prompt_correction_cache_min_depth < 1:
        raise ValueError("prompt_correction_cache_min_depth must be >= 1")
    if online_correction_cache_key not in {
        "local_prefix",
        "source_token",
        "primary_source",
    }:
        raise ValueError(
            "online_correction_cache_key must be 'local_prefix', "
            "'source_token', or 'primary_source'"
        )
    if draft_core not in {"stock", "device-d2", "device"}:
        raise ValueError("draft_core must be 'stock', 'device-d2', or 'device'")
    if not 0.0 <= adapter_ensemble_epsilon <= 1.0:
        raise ValueError("adapter_ensemble_epsilon must be in [0, 1]")
    if adapter_ensemble_min_depth < 1:
        raise ValueError("adapter_ensemble_min_depth must be >= 1")
    if verify_strategy not in {
        "batched",
        "capture_commit",
        "graphbank",
        "graphbank_capture_commit",
        "target_prefix",
        "trim_commit",
    }:
        raise ValueError(
            "verify_strategy must be 'batched', 'capture_commit', "
            "'graphbank', 'graphbank_capture_commit', 'target_prefix', "
            "or 'trim_commit'"
        )
    target_prefix_verify = verify_strategy == "target_prefix"
    # Constrained requests never engage the exact A3B route: the route
    # pre-commits its rejection correction (no None-guard on the append),
    # while the #186 phase-3 grammar clamp expects a grammar-illegal
    # correction to be dropped so the next masked primary resamples it.
    # The stock target_prefix lane below carries that contract.
    #
    # Context-copy on this lane is a DRAFT SOURCE (a prompt match feeds the
    # depth-1 draft; see context_copy_target_prefix_enabled), which conflicts
    # with the compiled K1 route's device-draft (R1) contract.  So the
    # compiled route stays STRICTLY K1/device-drafted: when the opt-in flag
    # takes over the lane, the route steps aside (exactly like the constraint
    # case) and the whole request runs the non-compiled target_prefix lane,
    # whose 2-row verify cycles are byte-exact to AR for any draft source.
    # The two improvement families never share a cycle: the compiled route
    # wins on pure-K1 requests, prompt-lookup drafting wins on the
    # non-compiled lane.  Keyed on the FLAG (not on whether streaks fire) so
    # ccopy-off on this lane is a clean byte-exactness baseline.  Whole-MoE
    # (needs the compiled route) and penalties (disable ccopy) both keep the
    # compiled route -- mirrors the ccopy_active gate below.
    from .context_copy import (
        context_copy_target_prefix_enabled as _cc_tp_enabled_early,
    )
    _ccopy_takes_over_lane = (
        target_prefix_verify
        and _cc_tp_enabled_early()
        and not (bool(sampler.presence_penalty) or bool(sampler.frequency_penalty))
        and not bool(getattr(rt, "a3b_whole_moe_installed", False))
    )
    exact_a3b_target_prefix_factory = (
        rt.a3b_compiled_target_prefix_factory
        if target_prefix_verify and constraint is None and not _ccopy_takes_over_lane
        else None
    )
    exact_a3b_target_prefix = exact_a3b_target_prefix_factory is not None
    draft_sampler = _env_scaled_draft_sampler(sampler, draft_sampler)
    _loop_guard_config = loop_guard_config_from_env(
        bool(loop_guard), tokenizer=getattr(rt, "tokenizer", None)
    )
    if adaptive_width_policy is not None:
        if adaptive_policy is not None:
            raise ValueError(
                "adaptive width policy cannot be combined with another adaptive policy"
            )
        if draft_margin_threshold is not None:
            raise ValueError(
                "adaptive width policy cannot be combined with draft_margin_threshold"
            )
        incompatible_features = {
            "draft_core": draft_core != "stock",
            "mtp_corrector": mtp_corrector is not None,
            "online_hidden_corrector": online_hidden_corrector_alpha != 0.0,
            "online_correction_cache": online_correction_cache,
            "prompt_correction_cache": prompt_correction_cache,
            "adapter_ensemble_q": adapter_ensemble_q,
            "mtp_topk_reranker": mtp_topk_reranker is not None,
            "session_bank": session_bank is not None,
            "vision_splice": vision_splice is not None,
            "constraint": constraint is not None,
            "loop_guard": _loop_guard_config.enabled,
            "thinking_guard": thinking_guard is not None,
            "compiled_verify": compiled_verify_mode() != "off",
        }
        selected_features = [
            name for name, selected in incompatible_features.items() if selected
        ]
        if selected_features:
            raise ValueError(
                "adaptive width policy requires its fixed canonical lane; "
                f"incompatible features: {selected_features}"
            )
        validate_installed_deepseek_v4_adaptive_width_policy(
            adaptive_width_policy,
            rt,
            sampler=sampler,
            draft_sampler=draft_sampler,
            speculative_depth=speculative_depth,
            verify_strategy=verify_strategy,
            verify_core=verify_core,
            mtp_history_policy=mtp_history_policy,
        )
    if bool(getattr(rt, "a3b_whole_moe_installed", False)):
        os.environ["MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS"] = str(len(prompt_ids))
        whole_moe_prefill_layout = _sustained_prefill_layout()
        validate_a3b_whole_moe_request(
            verify_strategy=verify_strategy,
            requested_speculative_depth=requested_speculative_depth,
            speculative_depth=speculative_depth,
            verify_core=verify_core,
            draft_core=draft_core,
            compiled_target_prefix=exact_a3b_target_prefix,
            session_bank_present=session_bank is not None,
            vision_splice_present=vision_splice is not None,
            prefill_layout=whole_moe_prefill_layout,
        )
        ensure_a3b_whole_moe_request_preflight(
            rt,
            prompt_ids,
            max_tokens=max_tokens,
            base_hidden_variant=base_hidden_variant,
            prefill_layout=whole_moe_prefill_layout,
        )
    if target_prefix_verify:
        if exact_a3b_target_prefix:
            validate_a3b_k1_target_prefix_sampler(sampler)
            validate_a3b_k1_device_draft_request(
                draft_sampler,
                draft_margin_threshold=draft_margin_threshold,
                adaptive_policy=adaptive_policy,
                draft_core=draft_core,
                online_correction_cache=online_correction_cache,
                prompt_correction_cache=prompt_correction_cache,
                adapter_ensemble_q=adapter_ensemble_q,
                mtp_topk_reranker=mtp_topk_reranker,
                loop_guard=_loop_guard_config.enabled,
                presence_penalty=float(sampler.presence_penalty),
                frequency_penalty=float(sampler.frequency_penalty),
            )
        else:
            _validate_target_prefix_sampler_request(sampler)
    counter_start = _runtime_counter_snapshot(rt)
    verify_core_backend = resolve_gdn_capture_backend(verify_core)
    online_hidden_enabled = online_hidden_corrector_alpha > 0.0
    online_hidden_max_feed_depth = (
        max(0, speculative_depth - 1)
        if online_hidden_corrector_max_feed_depth is None
        else int(online_hidden_corrector_max_feed_depth)
    )

    rng = np.random.default_rng(seed)

    def _default_cycle_draft_reader(
        draft_logits: mx.array,
        *,
        depth_index: int,
        need_distribution: bool,
        decision_margins: list[float],
    ) -> tuple[int, np.ndarray | SparseDistribution | None, bool]:
        del depth_index, decision_margins
        return _fixed_width_draft_reader(
            draft_logits,
            draft_sampler,
            rng,
            need_distribution=need_distribution,
        )

    if adaptive_width_policy is None:
        adaptive_width_cycle_readers = (_default_cycle_draft_reader,) * max(
            1, int(speculative_depth)
        )
        capture_forward_routes = (rt.forward_ar_capture,) * max(
            1, int(speculative_depth)
        )

        def record_adaptive_width_event(
            event: dict[str, Any],
            *,
            cycle_depth: int,
            decision_margins: list[float],
            selected_draft_depth: int,
        ) -> None:
            del event, cycle_depth, decision_margins, selected_draft_depth

    else:
        adaptive_width_margin_stops = (
            adaptive_width_policy.stop_after_d1,
            adaptive_width_policy.stop_after_d2,
        )
        adaptive_width_d1_threshold = float(
            adaptive_width_policy.d1_margin_threshold
        )
        adaptive_width_d2_threshold = float(
            adaptive_width_policy.d2_margin_threshold
        )
        adaptive_width_max_depth = int(adaptive_width_policy.max_speculative_depth)
        capture_forward_routes = adaptive_width_policy.target_routes

        def adaptive_tail_k1_reader(
            draft_logits: mx.array,
            *,
            depth_index: int,
            need_distribution: bool,
            decision_margins: list[float],
        ) -> tuple[int, np.ndarray | SparseDistribution | None, bool]:
            del depth_index, decision_margins
            return _adaptive_tail_k1_draft_reader(
                draft_logits,
                draft_sampler,
                rng,
                need_distribution=need_distribution,
            )

        def adaptive_tail_k2_reader(
            draft_logits: mx.array,
            *,
            depth_index: int,
            need_distribution: bool,
            decision_margins: list[float],
        ) -> tuple[int, np.ndarray | SparseDistribution | None, bool]:
            del depth_index, decision_margins
            return _adaptive_tail_k2_draft_reader(
                draft_logits,
                draft_sampler,
                rng,
                need_distribution=need_distribution,
            )

        def adaptive_full_k3_reader(
            draft_logits: mx.array,
            *,
            depth_index: int,
            need_distribution: bool,
            decision_margins: list[float],
        ) -> tuple[int, np.ndarray | SparseDistribution | None, bool]:
            return _adaptive_full_k3_draft_reader(
                draft_logits,
                draft_sampler,
                rng,
                depth_index=depth_index,
                need_distribution=need_distribution,
                decision_margins=decision_margins,
                margin_stops=adaptive_width_margin_stops,
            )

        adaptive_width_cycle_readers = (
            adaptive_tail_k1_reader,
            adaptive_tail_k2_reader,
            adaptive_full_k3_reader,
        )

        def record_adaptive_width_event(
            event: dict[str, Any],
            *,
            cycle_depth: int,
            decision_margins: list[float],
            selected_draft_depth: int,
        ) -> None:
            event["adaptive_width_policy"] = {
                "kind": "deepseek_v4_preregistered_max_k3",
                "eligible_full_k3": cycle_depth == adaptive_width_max_depth,
                "d1_margin_threshold": adaptive_width_d1_threshold,
                "d2_margin_threshold": adaptive_width_d2_threshold,
                "decision_margins": list(decision_margins),
                "selected_draft_depth": selected_draft_depth,
                "target_rows": selected_draft_depth + 1,
            }

    if mtp_corrector is not None:
        corrector_variant = getattr(mtp_corrector, "hidden_variant", mtp_hidden_variant)
        if corrector_variant != mtp_hidden_variant:
            raise ValueError(
                f"MTP corrector expects hidden variant {corrector_variant!r}, "
                f"but mtp_hidden_variant is {mtp_hidden_variant!r}"
            )
    stop_token_ids = (
        _default_stop_tokens(rt.tokenizer) if stop_token_ids is None else stop_token_ids
    )
    started_all = time.perf_counter()
    if constraint is not None:
        # The repetition trimmer retracts committed tokens, which would
        # desync the grammar matcher; constrained output is schema-shaped.
        repetition_stop = False
    repetition_config = _repetition_stop_config(bool(repetition_stop))
    repetition_result: RepetitionStopResult | None = None
    draft_time = verify_time = 0.0
    verify_forward_time = 0.0
    verify_eval_time = 0.0
    verify_logits_eval_time = 0.0
    verify_hidden_eval_time = 0.0
    verify_joint_eval_time = 0.0
    verify_target_distribution_time = 0.0
    target_distribution_materialized_rows = 0
    target_distribution_materialized_windows = 0
    lazy_bonus_verify_calls = 0
    lazy_bonus_commit_time = 0.0
    verify_eval_unattributed_time = 0.0
    # Stable prompt-prefix boundary (aligned-boundary design, 2026-08-06):
    # the encoder reports where the transient trailing tool-continuation
    # hint begins; prefill span planning makes that position a chunk edge so
    # the existing gdn-boundary capture records recurrent state exactly
    # there. Absent metadata leaves every span byte-identical to today.
    _stable_prefix_len: int | None = None
    try:
        _raw_stable = (trace_metadata or {}).get("stable_prefix_len")
        if _raw_stable is not None:
            _stable_prefix_len = max(0, int(_raw_stable)) or None
    except (TypeError, ValueError):
        _stable_prefix_len = None
    _prompt_state_started = time.perf_counter()
    prompt_state = restore_or_prefill_prompt_state(
        rt,
        prompt_ids,
        vision_splice=vision_splice,
        base_hidden_variant=base_hidden_variant,
        mtp_hidden_variant=mtp_hidden_variant,
        mtp_history_policy=mtp_history_policy,
        session_bank=session_bank,
        restore_mode=session_restore_mode,
        session_id=session_id,
        template_hash=session_template_hash,
        draft_head_identity=session_draft_head_identity,
        policy_fingerprint=session_policy_fingerprint,
        prefill_callback=prefill_callback,
        # kvcache-v2: client disconnect aborts the prefill through the same
        # chunk-granular check the postcommit path uses — an abandoned agent
        # request must not pin the GPU for a full long-context prefill
        # (measured: an orphaned ~200k prefill blocked all sessions for
        # 10+ minutes, 2026-07-03).
        abort_check=abort_check,
        stable_prefix_len=_stable_prefix_len,
    )
    prompt_state_total_time_s = time.perf_counter() - _prompt_state_started
    pre_first_token_setup_started = time.perf_counter()
    pre_first_token_setup_s = 0.0
    prompt_prefix_bank_commit: dict[str, object] = {}
    bank_commit_ids = prompt_ids
    if vision_splice is not None and session_bank is not None:
        # Same content-keyed view restore_or_prefill_prompt_state used; a
        # None here is impossible past its guard.
        from mtplx.vision.splice import vision_bank_key_ids

        bank_commit_ids = vision_bank_key_ids(prompt_ids, vision_splice) or prompt_ids
    if (
        commit_prompt_state_to_bank
        and session_bank is not None
        and session_id is not None
        and prompt_ids
        and int(prompt_state.suffix_tokens) > 0
    ):
        commit_started = time.perf_counter()
        commit_snapshot_done = commit_started
        try:
            mtp_snapshot = (
                snapshot_cache(prompt_state.committed_mtp_cache)
                if prompt_state.committed_mtp_cache is not None
                else None
            )
            commit_snapshot_done = time.perf_counter()
            commit_put_timing: dict[str, object] = {}
            entry = session_bank.put(
                runtime=rt,
                token_ids=list(bank_commit_ids),
                cache=prompt_state.trunk_cache,
                logits=prompt_state.logits,
                hidden=prompt_state.hidden,
                hidden_variant=base_hidden_variant,
                # This prompt cache is committed before decode continues and
                # mutates the same KV/MTP objects. A live reference lease here
                # can restore a post-decode cache under a pre-decode token
                # prefix on the next OpenCode turn. Store a real snapshot or
                # skip; generation-final/postcommit snapshots are the safe
                # places for live leases.
                keep_live_ref=False,
                session_id=session_id,
                template_hash=session_template_hash,
                mtp_history_policy=prompt_state.mtp_history_policy,
                draft_head_identity=session_draft_head_identity,
                policy_fingerprint=session_policy_fingerprint,
                mtp_history_snapshot=mtp_snapshot,
                mtp_history_cache_ref=None,
                snapshot_epoch=len(prompt_ids),
                mtp_snapshot_epoch=len(prompt_ids)
                if mtp_snapshot is not None
                or prompt_state.committed_mtp_cache is not None
                else None,
                # Issue #121 root cause (measured 2026-07-16): this commit is
                # the PRIMARY store for tool-session turns and it dropped the
                # recurrent boundaries the prefill captured/inherited. Every
                # descendant entry was boundary-less, so hybrid-model
                # near-prefix restores fail-closed (no_snapshot_coverage) and
                # agent turns pinned on the oldest clean-prefix entry while
                # skip% decayed (78%->52% over 12 turns in the replay).
                gdn_boundaries=list(
                    getattr(prompt_state, "gdn_boundaries", None) or []
                ),
                timing_out=commit_put_timing,
            )
            put_done = time.perf_counter()
            prompt_prefix_bank_commit = {
                "stored": entry is not None,
                "mode": "prompt_prefix",
                "reason": (
                    "committed_prompt_prefix"
                    if entry is not None
                    else "sessionbank_snapshot_skipped"
                ),
                "prefix_len": int(
                    entry.prefix_len if entry is not None else len(prompt_ids)
                ),
                "nbytes": int(entry.nbytes if entry is not None else 0),
                "elapsed_s": put_done - commit_started,
                "mtp_snapshot_elapsed_s": commit_snapshot_done - commit_started,
                "put_elapsed_s": put_done - commit_snapshot_done,
                "put_timing": commit_put_timing,
                "cached_tokens": int(prompt_state.cached_tokens),
                "suffix_tokens": int(prompt_state.suffix_tokens),
            }
        except BaseException as exc:
            prompt_prefix_bank_commit = {
                "stored": False,
                "mode": "prompt_prefix",
                "reason": f"prompt_prefix_commit_error:{type(exc).__name__}",
                "elapsed_s": time.perf_counter() - commit_started,
                "mtp_snapshot_elapsed_s": max(
                    0.0, commit_snapshot_done - commit_started
                ),
            }
    cache = prompt_state.trunk_cache
    logits = prompt_state.logits
    hidden = prompt_state.hidden
    mtp_history_cache = prompt_state.committed_mtp_cache
    mtp_history_policy = prompt_state.mtp_history_policy
    mtp_history_position_base = int(prompt_state.mtp_history_position_base)
    prompt_eval_time = prompt_state.prompt_eval_time_s
    prompt_target_prefill_time = max(
        0.0, prompt_eval_time - prompt_state.prompt_mtp_history_time_s
    )
    target_time = prompt_target_prefill_time
    draft_time += prompt_state.prompt_mtp_history_time_s
    graphbank = (
        SpecDecodeGraphBank(rt, capture_backend=verify_core_backend)
        if verify_strategy in {"graphbank", "graphbank_capture_commit"}
        else None
    )
    _compiled_verify_mode = compiled_verify_mode()
    generic_compiled_target_prefix = (
        target_prefix_verify
        and not exact_a3b_target_prefix
        and _env_truthy("MTPLX_COMPILED_TARGET_PREFIX")
    )
    compiled_verify_bank = (
        CompiledVerifyBank(
            rt,
            request_max_tokens=max_tokens,
            capture_backend=verify_core_backend,
            parity=_compiled_verify_mode == "parity",
            parity2=_compiled_verify_mode == "parity2",
            # Warm restores hand this generation exact-size KV buffers; the
            # bank defers its first round(s) to eager so the O(context)
            # promotion copy lands after TTFT, not inside it. cached_tokens
            # is 0 on cold prompts and the restored prefix length on hits.
            restored_tokens=int(getattr(prompt_state, "cached_tokens", 0) or 0),
        )
        if _compiled_verify_mode != "off"
        and (
            verify_strategy in {"capture_commit", "graphbank_capture_commit"}
            or generic_compiled_target_prefix
        )
        else None
    )
    a3b_target_prefix_route = None
    a3b_rebase_state = None  # stashed post-primary state for a deferred correction
    snapshot_time = accept_time = rollback_time = repair_time = 0.0
    commit_time = capture_commit_time = 0.0
    bonus_time = 0.0
    online_hidden_corrector_time = 0.0
    tokens: list[int] = []
    # Grammar-constrained decoding (#186 phase 3): the matcher advances only
    # through committed tokens (synced once per cycle after the primary);
    # speculative windows are clamped to the matcher's legal prefix before
    # commit, so drafts stay unmasked and correctness is target-side only.
    constraint_synced_tokens = 0
    # OpenAI-style presence/frequency penalties. When active, each token is
    # penalized by the counts of the completion-so-far (prompt excluded), and
    # every verified MTP position by its growing in-block prefix (per-position /
    # vLLM-exact). Counts are rebuilt from `tokens` at each sample point — simple
    # and drift-proof; an incremental counter is a documented perf follow-up.
    _penalties_active = bool(sampler.presence_penalty) or bool(sampler.frequency_penalty)
    # Loop Guard: loop-armed DRY-style steering (see mtplx/loop_guard.py).
    # Disarmed = zero distribution impact (identity transform, fast paths kept).
    # Armed = target distributions get sparse anti-cycle penalties per position;
    # the draft proposal q stays untouched (proposal mismatch only costs
    # acceptance, never correctness).
    _loop_guard = LoopGuard(_loop_guard_config) if _loop_guard_config.enabled else None
    # Thinking Guard: surfaced reasoning-token budget (mtplx/thinking_guard.py).
    # Below budget = zero distribution impact; at budget the guard force-closes
    # the reasoning segment through the same target-side overlay slot the Loop
    # Guard uses (drafts stay untouched; rejections correct exactly).
    _thinking_guard = (
        ThinkingGuard(thinking_guard)
        if thinking_guard is not None and thinking_guard.enabled
        else None
    )

    def _steer_overlay(working: Sequence[int]) -> dict[int, float] | None:
        merged = (
            _loop_guard.penalties_for(working)
            if _loop_guard is not None and _loop_guard.armed
            else None
        )
        forced = (
            _thinking_guard.overlay_for(working)
            if _thinking_guard is not None and _thinking_guard.steering_active
            else None
        )
        if not forced:
            return merged
        if not merged:
            return forced
        combined = dict(merged)
        for token, value in forced.items():
            combined[token] = combined.get(token, 0.0) + value
        return combined

    events: list[dict] = []
    record_events = not _env_truthy("MTPLX_DROP_EVENTS")
    append_event = events.append if record_events else (lambda _event: None)
    accepted = rejected = drafted = 0
    bonus_tokens = correction_tokens = verify_calls = 0
    accepted_by_depth = [0 for _ in range(speculative_depth)]
    drafted_by_depth = [0 for _ in range(speculative_depth)]
    accept_probability_sum_by_depth = [0.0 for _ in range(speculative_depth)]
    deferred_correction_repairs = 0
    pending_primary: int | None = None
    online_hidden_deltas: dict[object, mx.array] = {}
    online_hidden_update_counts: dict[object, int] = {}
    online_hidden_apply_counts: dict[object, int] = {}
    correction_cache: dict[tuple[int, ...], int] = {}
    prompt_seeded_cache_keys: set[tuple[int, ...]] = set()
    prompt_correction_cache_hits = 0
    prompt_seed_stats = {"stores": 0, "collisions": 0, "skipped": 0}
    if prompt_correction_cache:
        seeded_cache, prompt_seed_stats = _seed_prompt_correction_cache(
            prompt_ids,
            max_depth=speculative_depth,
            min_depth=prompt_correction_cache_min_depth,
            key_policy=online_correction_cache_key,
        )
        correction_cache.update(seeded_cache)
        prompt_seeded_cache_keys = set(seeded_cache)
    correction_cache_hits = 0
    correction_cache_stores = 0
    adapter_ensemble_calls = 0
    adapter_ensemble_changed = 0
    adapter_ensemble_base_selected = 0
    adapter_ensemble_adapter_selected = 0
    adapter_ensemble_shared_selected = 0
    adapter_ensemble_fallbacks = 0
    topk_reranker_calls = 0
    topk_reranker_changed = 0
    topk_reranker_fallbacks = 0
    topk_reranker_selected_rank_sum = 0
    device_d2_core: dict[str, Any] | None = None
    device_d2_compile_time = 0.0
    device_d2_calls = 0
    device_d2_fallbacks = 0
    # k=2 (depth-2) compiled target-prefix: a chained 2-draft producer for the
    # [primary, d1, d2] verify, plus the two mid-window rebase states the last
    # verify_m3 returned (post-row-0, post-row-1).  Dormant for K1.
    compiled_k2_d2_core: dict[str, Any] | None = None
    a3b_m3_rebase0_state = None
    a3b_m3_rebase1_state = None
    device_core: dict[str, Any] | None = None
    device_core_compile_time = 0.0
    device_core_calls = 0
    device_core_fallbacks = 0
    streamed_token_count = 0
    mtp_history_materialize_every = max(
        0,
        int(os.environ.get("MTPLX_MTP_HISTORY_MATERIALIZE_EVERY") or 0),
    )
    late_depth_switch_after = max(
        0,
        int(os.environ.get("MTPLX_LATE_DEPTH_SWITCH_AFTER_TOKENS") or 0),
    )
    late_depth_before = int(
        os.environ.get("MTPLX_LATE_DEPTH_BEFORE") or speculative_depth
    )
    late_depth_after = int(
        os.environ.get("MTPLX_LATE_DEPTH_AFTER") or speculative_depth
    )
    mtp_position_mode = _resolve_runtime_mtp_position_mode(rt)
    mtp_position_cap = max(
        0,
        int(os.environ.get("MTPLX_MTP_POSITION_CAP") or 4096),
    )
    mtp_position_period = max(
        0,
        int(os.environ.get("MTPLX_MTP_POSITION_PERIOD") or 4096),
    )
    position_base_env = os.environ.get("MTPLX_MTP_POSITION_BASE")
    mtp_position_base = max(
        0,
        int(
            position_base_env
            if position_base_env is not None
            else (len(prompt_state.token_prefix) if mtp_position_mode == "absolute" else 0)
        ),
    )
    # Validate env spelling before a long generation starts.
    _mtp_position_offset(
        0,
        mode=mtp_position_mode,
        cap=mtp_position_cap,
        period=mtp_position_period,
        base=mtp_position_base,
    )

    def mtp_position_offset_for_cache(mtp_cache) -> int | None:
        position_base = mtp_position_base
        if mtp_cache is mtp_history_cache and _mtp_history_uses_committed_cache(mtp_history_policy):
            position_base = mtp_history_position_base
        return _mtp_position_offset(
            _mtp_cache_offset(mtp_cache),
            mode=mtp_position_mode,
            cap=mtp_position_cap,
            period=mtp_position_period,
            base=position_base,
        )

    mtp_history_tokens_since_materialize = 0
    mtp_history_materialize_events = 0
    clear_cache_every = _clear_cache_every()
    clear_cache_tokens_since = 0
    clear_cache_observed_tokens = 0
    clear_cache_events = 0
    clear_cache_time_s = 0.0
    trunk_cache_materialize_every = max(
        0,
        int(os.environ.get("MTPLX_TRUNK_CACHE_MATERIALIZE_EVERY") or 0),
    )
    trunk_cache_materialize_tokens_since = 0
    trunk_cache_materialize_observed_tokens = 0
    trunk_cache_materialize_events = 0
    trunk_cache_materialize_time_s = 0.0
    state_rebase_every = max(
        0,
        int(os.environ.get("MTPLX_STATE_REBASE_EVERY") or 0),
    )
    state_rebase_tokens_since = 0
    state_rebase_observed_tokens = 0
    state_rebase_events = 0
    state_rebase_time_s = 0.0
    state_root_eval_enabled = _env_truthy("MTPLX_EVAL_STATE_ROOTS_ON_COMMIT")
    state_root_eval_include_mtp = os.environ.get(
        "MTPLX_EVAL_STATE_ROOTS_INCLUDE_MTP", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}
    state_root_eval_include_live = os.environ.get(
        "MTPLX_EVAL_STATE_ROOTS_INCLUDE_LIVE", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}
    defer_verify_hidden_eval = _defer_verify_hidden_eval_enabled()
    verify_hidden_mode = _verify_hidden_mode()
    lazy_target_distributions = _lazy_target_distributions_enabled()
    state_root_eval_events = 0
    state_root_eval_time_s = 0.0
    state_root_eval_arrays = 0
    dirty_detach_mode = (
        (os.environ.get("MTPLX_DETACH_MODE") or "selected_slice_contiguous_eval")
        .strip()
        .lower()
        .replace("-", "_")
    )
    dirty_detach_components_env = os.environ.get("MTPLX_DETACH_COMPONENTS") or ""
    dirty_detach_component_filter = {
        item.strip().lower().replace("-", "_")
        for item in dirty_detach_components_env.split(",")
        if item.strip()
    }
    dirty_detach_supported_components = {"gdn", "conv", "attn"}
    dirty_detach_global_every = max(
        0,
        int(os.environ.get("MTPLX_DETACH_EVERY") or 0),
    )
    dirty_detach_cadences = {
        "gdn": max(
            0,
            int(os.environ.get("MTPLX_DETACH_GDN_EVERY") or dirty_detach_global_every),
        ),
        "conv": max(
            0,
            int(os.environ.get("MTPLX_DETACH_CONV_EVERY") or dirty_detach_global_every),
        ),
        "attn": max(
            0,
            int(os.environ.get("MTPLX_DETACH_ATTN_EVERY") or dirty_detach_global_every),
        ),
    }
    if dirty_detach_component_filter:
        dirty_detach_cadences = {
            key: value if key in dirty_detach_component_filter else 0
            for key, value in dirty_detach_cadences.items()
        }
    dirty_detach_enabled_components = sorted(
        component
        for component, cadence in dirty_detach_cadences.items()
        if component in dirty_detach_supported_components and cadence > 0
    )
    dirty_detach_tokens_since = {
        component: 0 for component in dirty_detach_supported_components
    }
    dirty_detach_observed_tokens = 0
    dirty_detach_events = 0
    dirty_detach_time_s = 0.0
    dirty_detach_arrays = 0
    dirty_detach_bytes = 0
    live_output_detach_enabled = _env_truthy("MTPLX_DETACH_LIVE_OUTPUTS")
    live_output_detach_mode = (
        (
            os.environ.get("MTPLX_DETACH_LIVE_OUTPUTS_MODE")
            or os.environ.get("MTPLX_DETACH_MODE")
            or "contiguous_eval"
        )
        .strip()
        .lower()
        .replace("-", "_")
    )
    live_output_detach_events = 0
    live_output_detach_time_s = 0.0
    live_output_detach_arrays = 0
    live_output_detach_bytes = 0
    capture_commit_detach_mode = (
        (os.environ.get("MTPLX_CAPTURE_COMMIT_DETACH_MODE") or dirty_detach_mode)
        .strip()
        .lower()
        .replace("-", "_")
    )
    capture_commit_detach_components_env = (
        os.environ.get("MTPLX_CAPTURE_COMMIT_DETACH_COMPONENTS") or ""
    )
    capture_commit_detach_component_filter = {
        item.strip().lower().replace("-", "_")
        for item in capture_commit_detach_components_env.split(",")
        if item.strip()
    }
    capture_commit_detach_global_every = max(
        0,
        int(os.environ.get("MTPLX_CAPTURE_COMMIT_DETACH_EVERY") or 0),
    )
    capture_commit_detach_cadences = {
        "gdn": max(
            0,
            int(
                os.environ.get("MTPLX_CAPTURE_COMMIT_DETACH_GDN_EVERY")
                or capture_commit_detach_global_every
            ),
        ),
        "conv": max(
            0,
            int(
                os.environ.get("MTPLX_CAPTURE_COMMIT_DETACH_CONV_EVERY")
                or capture_commit_detach_global_every
            ),
        ),
    }
    if capture_commit_detach_component_filter:
        capture_commit_detach_cadences = {
            key: value if key in capture_commit_detach_component_filter else 0
            for key, value in capture_commit_detach_cadences.items()
        }
    capture_commit_detach_enabled_components = sorted(
        component
        for component, cadence in capture_commit_detach_cadences.items()
        if component in dirty_detach_supported_components and cadence > 0
    )
    capture_commit_detach_tokens_since = {
        component: 0 for component in dirty_detach_supported_components
    }
    capture_commit_detach_observed_tokens = 0
    capture_commit_detach_events = 0
    capture_commit_detach_time_s = 0.0
    capture_commit_detach_arrays = 0
    capture_commit_detach_bytes = 0
    trace_verify_output_nbytes = 0
    trace_draft_output_nbytes = 0
    trace_mtp_history_append_nbytes = 0
    trace_accounting_time_s = 0.0
    trace_extra_metadata = dict(trace_metadata or {})
    if mtp_position_mode not in {"", "0", "off", "false", "default", "cache"}:
        trace_extra_metadata["mtp_position"] = {
            "mode": mtp_position_mode,
            "cap": int(mtp_position_cap),
            "period": int(mtp_position_period),
            "base": int(mtp_position_base),
        }
    if mtp_history_policy == "last_window":
        trace_extra_metadata["mtp_history_last_window"] = {
            "tokens": int(prompt_state.mtp_history_window_tokens),
            "position_base": int(prompt_state.mtp_history_position_base),
        }
    trace = _DecodeTrace(
        prompt_tokens=len(prompt_ids),
        max_tokens=max_tokens,
        speculative_depth=speculative_depth,
        sampler=sampler,
        verify_strategy=verify_strategy,
        verify_core=verify_core_backend.replace("_", "-"),
        mtp_history_policy=mtp_history_policy,
        mtp_cache_policy=mtp_cache_policy,
        trace_label=trace_label,
        trace_metadata=trace_extra_metadata,
    )
    trace_current_mtp_cache = mtp_history_cache

    def own_live_output_leaf(value: Any) -> Any:
        nonlocal live_output_detach_events
        nonlocal live_output_detach_time_s
        nonlocal live_output_detach_arrays
        nonlocal live_output_detach_bytes
        if not live_output_detach_enabled or value is None:
            return value
        started_detach = time.perf_counter()
        detached = detach_array_leaf(value, mode=live_output_detach_mode)
        live_output_detach_time_s += time.perf_counter() - started_detach
        if isinstance(detached, mx.array):
            live_output_detach_events += 1
            live_output_detach_arrays += 1
            live_output_detach_bytes += int(detached.nbytes)
        return detached

    def own_live_logits_hidden(logit_leaf: Any, hidden_leaf: Any) -> tuple[Any, Any]:
        return own_live_output_leaf(logit_leaf), own_live_output_leaf(hidden_leaf)

    if live_output_detach_enabled:
        logits, hidden = own_live_logits_hidden(logits, hidden)

    def append_mtp_history(
        mtp_cache,
        hidden_states: mx.array,
        token_ids: list[int],
    ) -> float:
        nonlocal mtp_history_tokens_since_materialize, mtp_history_materialize_events
        nonlocal trace_mtp_history_append_nbytes, trace_accounting_time_s
        if not token_ids:
            return 0.0
        if trace.enabled:
            trace_accounting_started = time.perf_counter()
            trace_mtp_history_append_nbytes += _tree_nbytes(hidden_states) + (
                8 * len(token_ids)
            )
            trace_accounting_time_s += time.perf_counter() - trace_accounting_started
        hidden_states = own_live_output_leaf(hidden_states)
        mtp_history_tokens_since_materialize += len(token_ids)
        force_eval = (
            mtp_history_materialize_every > 0
            and mtp_history_tokens_since_materialize >= mtp_history_materialize_every
        )
        elapsed = _append_mtp_history(
            rt,
            mtp_cache,
            hidden_states,
            token_ids,
            phase="ar_decode",
            mtp_hidden_variant=mtp_hidden_variant,
            position_offset=mtp_position_offset_for_cache(mtp_cache),
            force_eval=force_eval,
        )
        if force_eval:
            mtp_history_materialize_events += 1
            mtp_history_tokens_since_materialize = 0
        return elapsed

    def maybe_eval_state_roots(event: dict[str, Any], current_tokens: int) -> None:
        nonlocal state_root_eval_events, state_root_eval_time_s
        nonlocal state_root_eval_arrays
        if not state_root_eval_enabled:
            return
        arrays = _tree_mx_arrays(cache)
        if state_root_eval_include_mtp:
            arrays.extend(_tree_mx_arrays(trace_current_mtp_cache))
        if state_root_eval_include_live:
            arrays.extend(_tree_mx_arrays(logits))
            arrays.extend(_tree_mx_arrays(hidden))
        deduped: list[mx.array] = []
        seen_arrays: set[int] = set()
        for array in arrays:
            array_id = id(array)
            if array_id in seen_arrays:
                continue
            seen_arrays.add(array_id)
            deduped.append(array)
        if not deduped:
            return
        started_eval = time.perf_counter()
        _eval(*deduped)
        elapsed_eval = time.perf_counter() - started_eval
        state_root_eval_events += 1
        state_root_eval_time_s += elapsed_eval
        state_root_eval_arrays += len(deduped)
        event["state_root_eval"] = {
            "current_tokens": int(current_tokens),
            "arrays": int(len(deduped)),
            "elapsed_s": float(elapsed_eval),
            "include_mtp": bool(state_root_eval_include_mtp),
            "include_live": bool(state_root_eval_include_live),
        }
        _add_timing(event, "state_root_eval", elapsed_eval)

    def maybe_rebase_decode_state(current_tokens: int) -> None:
        nonlocal cache, logits, hidden, mtp_history_cache, trace_current_mtp_cache
        nonlocal target_time, draft_time
        nonlocal state_rebase_tokens_since, state_rebase_observed_tokens
        nonlocal state_rebase_events, state_rebase_time_s
        if state_rebase_every <= 0 or current_tokens <= 0:
            return
        if current_tokens < state_rebase_observed_tokens:
            state_rebase_observed_tokens = current_tokens
            return
        delta_tokens = current_tokens - state_rebase_observed_tokens
        if delta_tokens <= 0:
            return
        state_rebase_observed_tokens = current_tokens
        state_rebase_tokens_since += delta_tokens
        if state_rebase_tokens_since < state_rebase_every:
            return
        prefix_tokens = list(prompt_ids) + [
            int(token) for token in tokens[:current_tokens]
        ]
        started_rebase = time.perf_counter()
        if vision_splice is not None:
            # The rebase replays the full prompt, so the splice queue
            # must rewind to serve the image rows again.
            vision_splice.reset()
        rebased = restore_or_prefill_prompt_state(
            rt,
            prefix_tokens,
            vision_splice=vision_splice,
            base_hidden_variant=base_hidden_variant,
            mtp_hidden_variant=mtp_hidden_variant,
            mtp_history_policy=mtp_history_policy,
            session_bank=None,
        )
        state_rebase_time_s += time.perf_counter() - started_rebase
        state_rebase_events += 1
        state_rebase_tokens_since = 0
        cache = rebased.trunk_cache
        logits = rebased.logits
        hidden = rebased.hidden
        mtp_history_cache = rebased.committed_mtp_cache
        trace_current_mtp_cache = mtp_history_cache
        target_time += max(
            0.0, rebased.prompt_eval_time_s - rebased.prompt_mtp_history_time_s
        )
        draft_time += rebased.prompt_mtp_history_time_s

    def maybe_clear_mlx_cache() -> None:
        nonlocal clear_cache_tokens_since, clear_cache_observed_tokens
        nonlocal clear_cache_events, clear_cache_time_s
        if clear_cache_every <= 0:
            return
        current_tokens = len(tokens)
        if current_tokens < clear_cache_observed_tokens:
            clear_cache_observed_tokens = current_tokens
            return
        delta_tokens = current_tokens - clear_cache_observed_tokens
        if delta_tokens <= 0:
            return
        clear_cache_observed_tokens = current_tokens
        clear_cache_tokens_since += delta_tokens
        if clear_cache_tokens_since < clear_cache_every:
            return
        started_clear = time.perf_counter()
        try:
            mx.synchronize()
        except RuntimeError:
            pass
        mx.clear_cache()
        clear_cache_time_s += time.perf_counter() - started_clear
        clear_cache_events += 1
        clear_cache_tokens_since = 0

    def maybe_materialize_trunk_cache() -> None:
        nonlocal trunk_cache_materialize_tokens_since
        nonlocal trunk_cache_materialize_observed_tokens
        nonlocal trunk_cache_materialize_events
        nonlocal trunk_cache_materialize_time_s
        if trunk_cache_materialize_every <= 0:
            return
        current_tokens = len(tokens)
        if current_tokens < trunk_cache_materialize_observed_tokens:
            trunk_cache_materialize_observed_tokens = current_tokens
            return
        delta_tokens = current_tokens - trunk_cache_materialize_observed_tokens
        if delta_tokens <= 0:
            return
        trunk_cache_materialize_observed_tokens = current_tokens
        trunk_cache_materialize_tokens_since += delta_tokens
        if trunk_cache_materialize_tokens_since < trunk_cache_materialize_every:
            return
        arrays = _tree_mx_arrays(cache)
        started_materialize = time.perf_counter()
        if arrays:
            mx.eval(*arrays)
        trunk_cache_materialize_time_s += time.perf_counter() - started_materialize
        trunk_cache_materialize_events += 1
        trunk_cache_materialize_tokens_since = 0

    def maybe_detach_dirty_state(current_tokens: int | None = None) -> None:
        nonlocal dirty_detach_observed_tokens
        nonlocal dirty_detach_events, dirty_detach_time_s
        nonlocal dirty_detach_arrays, dirty_detach_bytes
        if not dirty_detach_enabled_components:
            return
        if current_tokens is None:
            current_tokens = len(tokens)
        if current_tokens < dirty_detach_observed_tokens:
            dirty_detach_observed_tokens = current_tokens
            return
        delta_tokens = current_tokens - dirty_detach_observed_tokens
        if delta_tokens <= 0:
            return
        dirty_detach_observed_tokens = current_tokens
        due_components: set[str] = set()
        for component in dirty_detach_enabled_components:
            dirty_detach_tokens_since[component] += delta_tokens
            cadence = dirty_detach_cadences[component]
            if cadence > 0 and dirty_detach_tokens_since[component] >= cadence:
                due_components.add(component)
                dirty_detach_tokens_since[component] = 0
        if not due_components:
            return
        started_detach = time.perf_counter()
        stats = detach_cache_state(
            cache,
            components=due_components,
            mode=dirty_detach_mode,
        )
        dirty_detach_time_s += time.perf_counter() - started_detach
        if int(stats.get("arrays", 0)) <= 0:
            return
        dirty_detach_events += 1
        dirty_detach_arrays += int(stats.get("arrays", 0))
        dirty_detach_bytes += int(stats.get("bytes", 0))

    def capture_commit_detach_due(current_tokens: int) -> set[str]:
        nonlocal capture_commit_detach_observed_tokens
        if not capture_commit_detach_enabled_components:
            return set()
        if current_tokens < capture_commit_detach_observed_tokens:
            capture_commit_detach_observed_tokens = current_tokens
            return set()
        delta_tokens = current_tokens - capture_commit_detach_observed_tokens
        if delta_tokens <= 0:
            return set()
        capture_commit_detach_observed_tokens = current_tokens
        due_components: set[str] = set()
        for component in capture_commit_detach_enabled_components:
            capture_commit_detach_tokens_since[component] += delta_tokens
            cadence = capture_commit_detach_cadences[component]
            if cadence > 0 and capture_commit_detach_tokens_since[component] >= cadence:
                due_components.add(component)
                capture_commit_detach_tokens_since[component] = 0
        return due_components

    def detach_capture_committed_state(current_tokens: int) -> None:
        nonlocal capture_commit_detach_events, capture_commit_detach_time_s
        nonlocal capture_commit_detach_arrays, capture_commit_detach_bytes
        due_components = capture_commit_detach_due(current_tokens)
        if not due_components:
            return
        started_detach = time.perf_counter()
        stats = detach_cache_state(
            cache,
            components=due_components,
            mode=capture_commit_detach_mode,
        )
        capture_commit_detach_time_s += time.perf_counter() - started_detach
        if int(stats.get("arrays", 0)) <= 0:
            return
        capture_commit_detach_events += 1
        capture_commit_detach_arrays += int(stats.get("arrays", 0))
        capture_commit_detach_bytes += int(stats.get("bytes", 0))

    def trace_totals() -> dict[str, Any]:
        return {
            "generated_tokens": len(tokens),
            "accepted_drafts": accepted,
            "rejected_drafts": rejected,
            "drafted_tokens": drafted,
            "verify_calls": verify_calls,
            "correction_tokens": correction_tokens,
            "bonus_tokens": bonus_tokens,
            "verify_time_s": verify_time,
            "verify_forward_time_s": verify_forward_time,
            "verify_eval_time_s": verify_eval_time,
            "verify_logits_eval_time_s": verify_logits_eval_time,
            "verify_hidden_eval_time_s": verify_hidden_eval_time,
            "verify_joint_eval_time_s": verify_joint_eval_time,
            "verify_target_distribution_time_s": verify_target_distribution_time,
            "target_distribution_materialized_rows": target_distribution_materialized_rows,
            "target_distribution_materialized_windows": target_distribution_materialized_windows,
            "lazy_bonus_verify_calls": lazy_bonus_verify_calls,
            "lazy_bonus_commit_time_s": lazy_bonus_commit_time,
            "verify_eval_unattributed_time_s": verify_eval_unattributed_time,
            "draft_time_s": draft_time,
            "accept_time_s": accept_time,
            "repair_time_s": repair_time,
            "commit_time_s": commit_time,
            "capture_commit_time_s": capture_commit_time,
            "snapshot_time_s": snapshot_time,
            "bonus_time_s": bonus_time,
            "verify_output_nbytes": trace_verify_output_nbytes,
            "draft_output_nbytes": trace_draft_output_nbytes,
            "mtp_history_append_nbytes": trace_mtp_history_append_nbytes,
            "clear_cache_events": clear_cache_events,
            "clear_cache_time_s": clear_cache_time_s,
            "trunk_cache_materialize_events": trunk_cache_materialize_events,
            "trunk_cache_materialize_time_s": trunk_cache_materialize_time_s,
            "dirty_detach_events": dirty_detach_events,
            "dirty_detach_time_s": dirty_detach_time_s,
            "dirty_detach_arrays": dirty_detach_arrays,
            "dirty_detach_bytes": dirty_detach_bytes,
            "live_output_detach_events": live_output_detach_events,
            "live_output_detach_time_s": live_output_detach_time_s,
            "live_output_detach_arrays": live_output_detach_arrays,
            "live_output_detach_bytes": live_output_detach_bytes,
            "state_rebase_events": state_rebase_events,
            "state_rebase_time_s": state_rebase_time_s,
            "state_root_eval_events": state_root_eval_events,
            "state_root_eval_time_s": state_root_eval_time_s,
            "state_root_eval_arrays": state_root_eval_arrays,
            "capture_commit_detach_events": capture_commit_detach_events,
            "capture_commit_detach_time_s": capture_commit_detach_time_s,
            "capture_commit_detach_arrays": capture_commit_detach_arrays,
            "capture_commit_detach_bytes": capture_commit_detach_bytes,
            "trace_accounting_time_s": trace_accounting_time_s,
            "accepted_by_depth": list(accepted_by_depth),
            "drafted_by_depth": list(drafted_by_depth),
            "accept_probability_sum_by_depth": list(accept_probability_sum_by_depth),
        }

    def emit_trace(*, force: bool = False, final: bool = False) -> None:
        trace.maybe_emit(
            force=force,
            final=final,
            totals=trace_totals(),
            cache=cache,
            mtp_cache=trace_current_mtp_cache,
            mtp_history_materialize_every=mtp_history_materialize_every,
            mtp_history_materialize_events=mtp_history_materialize_events,
        )

    def emit_new_tokens() -> None:
        nonlocal streamed_token_count
        maybe_materialize_trunk_cache()
        maybe_clear_mlx_cache()
        if token_callback is None or streamed_token_count >= len(tokens):
            return
        new_tokens = [
            int(token)
            for token in tokens[streamed_token_count:]
            if not _is_stop(int(token), stop_token_ids)
        ]
        streamed_token_count = len(tokens)
        if new_tokens:
            token_callback(new_tokens)

    if exact_a3b_target_prefix:
        if _compiled_verify_mode != "on":
            raise RuntimeError(
                "exact A3B compiled target-prefix requires compiled verify mode 'on'"
            )
        a3b_target_prefix_route = install_a3b_k1_target_prefix_route(
            rt,
            cache,
            factory=exact_a3b_target_prefix_factory,
            max_tokens=max_tokens,
            prompt_tokens=len(prompt_ids),
            verify_strategy=verify_strategy,
            speculative_depth=speculative_depth,
            requested_speculative_depth=requested_speculative_depth,
            verify_core=verify_core_backend,
            hidden_variant=base_hidden_variant,
            state_rebase_every=state_rebase_every,
            require_request_preflight=bool(
                getattr(rt, "a3b_whole_moe_installed", False)
            ),
        )

    step = 0
    # ---- context-copy (prompt-lookup) drafting: always on (kill switch
    # MTPLX_CONTEXT_COPY=0); any temperature, no repetition penalties, on
    # capture-commit verify strategies ----
    from .context_copy import (NgramIndex, block_for_ext, context_copy_block_k,
                               context_copy_enabled, context_copy_min_ext,
                               context_copy_ng_max, context_copy_ng_min,
                               context_copy_target_prefix_enabled)
    # Temperature is supported through the same probability-ratio acceptance
    # as the MTP path: the copy block is a point-mass proposal, so a copied
    # token is accepted with the target's own shaped probability and a
    # rejection samples the residual — the output law is exactly the target
    # sampling distribution at any temperature (no greedy shortcut).
    #
    # Copy rounds normally require a capture-commit verify strategy.  The opt-in
    # MTPLX_CONTEXT_COPY_TARGET_PREFIX flag also enables the target_prefix
    # lane-takeover, where context-copy is a DRAFT SOURCE (streaks feed the
    # depth-1 draft; block rounds stay capture_commit-only -- their T+1-row
    # forwards are not AR-exact).  With whole-MoE installed the compiled
    # route is kept and the flag is inert, recorded via disabled_reason.
    _ccopy_capture_lane = verify_strategy in {"capture_commit", "graphbank_capture_commit"}
    _ccopy_tp_requested = (
        context_copy_target_prefix_enabled() and verify_strategy == "target_prefix"
    )
    _ccopy_whole_moe_conflict = _ccopy_tp_requested and bool(
        getattr(rt, "a3b_whole_moe_installed", False)
    )
    ccopy_active = (
        context_copy_enabled()
        and not _penalties_active
        and (_ccopy_capture_lane or (_ccopy_tp_requested and not _ccopy_whole_moe_conflict))
    )
    ccopy_rounds = ccopy_drafted = ccopy_accepted = 0
    ccopy_probes = ccopy_blocks_accepted = ccopy_suspensions = 0
    ccopy_disabled_reason = None
    if _ccopy_whole_moe_conflict:
        # Requested the target_prefix takeover but whole-MoE is installed:
        # whole-MoE requires the compiled route, whose device-draft contract
        # excludes draft substitution.  The compiled route is kept.
        ccopy_disabled_reason = "whole_moe_keeps_compiled_route"
    ccopy_ema, ccopy_seen, ccopy_suspend_until = 0.5, 0, 0
    ccopy_backoff = 64   # doubles on each suspension (self-repetitive novel text would
                         # otherwise re-trigger copy rounds after every backoff and pay
                         # the probe cost recurrently); a paying round resets it.
    # Draft-source streak state (target_prefix takeover lane): the copy match
    # feeds the depth-1 DRAFT instead of a block round, so every forward stays
    # on the lane's proven 2-row verify geometry -- bit-exact by construction.
    # _cc_src_idx = next prompt index the streak proposes; the streak advances
    # by diffing committed tokens against the prompt continuation and breaks on
    # the first mismatch (covers accept, bonus, and correction paths without
    # touching the accept machinery).
    _cc_src_idx: int | None = None
    _cc_src_check_from = 0
    _cc_streak_drafted = 0
    _cc_streak_accepted = 0
    _cc_streak_outstanding = 0  # substituted drafts not yet seen by the sync
    ccopy_index = None
    ccopy_k = context_copy_block_k()
    ccopy_min_ext = context_copy_min_ext()
    if ccopy_active:
        ccopy_index = NgramIndex(context_copy_ng_min(), context_copy_ng_max())
        # Prompt-lookup semantics: the index covers the PROMPT only. Matches into
        # the model's own generated text (self-repetition) tend to have weak
        # continuation predictiveness and can cost more to verify than they commit,
        # while grounded re-emission matches into the prompt (see the PR benchmarks).
        ccopy_index.sync(prompt_ids)
    # Close the pre-first-token setup span here: everything from the
    # restore/prefill return to this point (prompt-prefix bank commit,
    # graphbank/policy/sampler construction) is setup wall time that
    # decode_elapsed_s contains but the per-round timers never see.
    pre_first_token_setup_s = time.perf_counter() - pre_first_token_setup_started
    decode_loop_entered_s = time.perf_counter()
    first_primary_sample_time_s = 0.0
    first_round_snapshot: dict[str, object] | None = None
    # Cost-model depth policy: cycle wall-time measured by the loop itself
    # (first observe gets the span since loop entry, later ones the span
    # since the previous observe) — real cycle cost, not inter-request gaps.
    _policy_cycle_started = time.perf_counter()
    while len(tokens) < max_tokens:
        if first_round_snapshot is None and step >= 1:
            # Top of iteration 2: the cumulative timers now hold exactly
            # round 1's totals. Pure bookkeeping — no evaluation forced.
            first_round_snapshot = {
                "wall_s": time.perf_counter() - decode_loop_entered_s,
                "draft_time_s": float(draft_time),
                "verify_time_s": float(verify_time),
                "verify_forward_time_s": float(verify_forward_time),
                "accept_time_s": float(accept_time),
                "verify_calls": int(verify_calls),
                "committed_tokens": len(tokens),
            }
        repetition_result = _trim_repeated_suffix(tokens, repetition_config)
        if repetition_result is not None:
            events.append(
                {
                    "step": step,
                    "repetition_stop": {
                        "reason": "exact_repeated_token_suffix",
                        "block_tokens": repetition_result.block_tokens,
                        "repeats": repetition_result.repeats,
                        "trimmed_tokens": repetition_result.repeated_tokens,
                    },
                }
            )
            streamed_token_count = min(streamed_token_count, len(tokens))
            emit_trace(force=True)
            break
        if _loop_guard is not None:
            _guard_transition = _loop_guard.observe(tokens)
            if _guard_transition is not None:
                append_event(
                    {
                        "step": step,
                        "loop_guard": {
                            "transition": _guard_transition,
                            "completion_tokens": len(tokens),
                            **_loop_guard.summary(),
                        },
                    }
                )
        if _thinking_guard is not None:
            _tg_transition = _thinking_guard.observe(tokens)
            if _tg_transition is not None:
                append_event(
                    {
                        "step": step,
                        "thinking_guard": {
                            "transition": _tg_transition,
                            "completion_tokens": len(tokens),
                            **_thinking_guard.summary(),
                        },
                    }
                )
        _guard_armed = _loop_guard is not None and _loop_guard.armed
        _steer_active = _guard_armed or (
            _thinking_guard is not None and _thinking_guard.steering_active
        )
        if constraint is not None:
            # Sync the matcher through the previous cycle's committed window
            # BEFORE masking this cycle's primary — a stale matcher would
            # compute the mask at the wrong grammar position.
            constraint.advance_many(tokens[constraint_synced_tokens:])
            constraint_synced_tokens = len(tokens)
            if (
                constraint.stopped
                and tokens
                and not _is_stop(tokens[-1], stop_token_ids)
            ):
                append_event({"step": len(tokens), "constraint_stop": True})
                break
        primary_already_emitted = pending_primary is not None
        if pending_primary is None:
            primary_row = logits[0]
            if constraint is not None:
                # The one guaranteed-progress mask site: every cycle's fresh
                # position samples from the constrained target distribution.
                # Speculative windows are handled by the legality clamp below
                # instead of per-row masks (see #186 phase 3).
                primary_row = constraint.mask_logits_row(primary_row)
            primary, _ = _sample_from_logits(
                primary_row,
                sampler,
                rng,
                token_counts=Counter(tokens) if _penalties_active else None,
                penalty_overlay=(
                    _steer_overlay(tokens) if _steer_active else None
                ),
            )
            if first_primary_sample_time_s == 0.0:
                # First primary token sampled: any lazy tail forced by
                # touching the seed logits has just been paid. Passive read.
                first_primary_sample_time_s = (
                    time.perf_counter() - decode_loop_entered_s
                )
            tokens.append(primary)
            emit_new_tokens()
            if constraint is not None:
                # Everything later this cycle (copy-block truncation, the
                # accept-loop clamp, bonus checks) validates windows that
                # FOLLOW the primary, so consume it now.
                constraint.advance_many(tokens[constraint_synced_tokens:])
                constraint_synced_tokens = len(tokens)
        else:
            primary = pending_primary
            pending_primary = None
        planned_depth = (
            adaptive_policy.current_depth
            if adaptive_policy is not None
            else speculative_depth
        )
        if adaptive_policy is None and late_depth_switch_after > 0:
            planned_depth = (
                late_depth_after
                if len(tokens) >= late_depth_switch_after
                else late_depth_before
            )
        planned_depth = max(1, min(int(planned_depth), int(speculative_depth)))
        event = {
            "step": step,
            "primary": primary,
            "primary_already_emitted": primary_already_emitted,
            "depth": planned_depth,
            "requested_depth": requested_speculative_depth,
            "drafts": [],
            "accepted_depths": 0,
            "rejected_at_depth": None,
            "gated_stop_depth": None,
            "mtp_history_policy": mtp_history_policy,
            "verify_strategy": verify_strategy,
            "verify_core": verify_core_backend.replace("_", "-"),
            "draft_core": draft_core,
        }
        if late_depth_switch_after > 0:
            event["late_depth_switch"] = {
                "after_tokens": int(late_depth_switch_after),
                "before": int(late_depth_before),
                "after": int(late_depth_after),
            }
        if long_context_depth_policy.get("active"):
            event["long_context_mtp_depth_policy"] = long_context_depth_policy
        if mtp_position_mode not in {"", "0", "off", "false", "default", "cache"}:
            event["mtp_position"] = {
                "mode": mtp_position_mode,
                "cap": int(mtp_position_cap),
                "period": int(mtp_position_period),
                "base": int(mtp_position_base),
                "history_offset": _mtp_cache_offset(mtp_history_cache),
            }
        if online_hidden_enabled:
            event["online_hidden_corrector"] = {
                "alpha": float(online_hidden_corrector_alpha),
                "decay": float(online_hidden_corrector_decay),
                "warmup": int(online_hidden_corrector_warmup),
                "max_feed_depth": int(online_hidden_max_feed_depth),
                "key": online_hidden_corrector_key,
            }
        correction_cache_enabled = online_correction_cache or prompt_correction_cache
        if correction_cache_enabled:
            event["online_correction_cache"] = {
                "enabled": bool(online_correction_cache),
                "prompt_enabled": bool(prompt_correction_cache),
                "min_depth": int(online_correction_cache_min_depth),
                "prompt_min_depth": int(prompt_correction_cache_min_depth),
                "key_policy": online_correction_cache_key,
            }
        if adapter_ensemble_q:
            event["adapter_ensemble_q"] = {
                "enabled": True,
                "epsilon": float(adapter_ensemble_epsilon),
                "min_depth": int(adapter_ensemble_min_depth),
            }
        if mtp_topk_reranker is not None:
            event["mtp_topk_reranker"] = mtp_topk_reranker.to_dict()
        step += 1
        if len(tokens) >= max_tokens or _is_stop(primary, stop_token_ids):
            append_event(event)
            emit_trace()
            break

        cycle_depth = min(planned_depth, max_tokens - len(tokens))
        cycle_draft_reader = adaptive_width_cycle_readers[cycle_depth - 1]
        adaptive_width_decision_margins: list[float] = []
        draft_tokens: list[int | None] = []
        draft_probs: list[np.ndarray | None] = []
        draft_cache_keys: list[tuple[int, ...]] = []
        draft_hidden_for_update: list[mx.array] = []
        draft_hidden_update_keys: list[object] = []
        if _mtp_history_uses_committed_cache(mtp_history_policy):
            mtp_cache = mtp_history_cache
            cycle_mtp_offset = _mtp_cache_offset(mtp_cache)
        else:
            mtp_cache = (
                rt.make_mtp_cache() if mtp_cache_policy == "persistent" else None
            )
            cycle_mtp_offset = None
        trace_current_mtp_cache = (
            mtp_cache if mtp_cache is not None else mtp_history_cache
        )
        # ---- context-copy as DRAFT SOURCE (target_prefix takeover lane) ----
        # The block-round machinery is NOT AR-exact on this lane: its T+1-row
        # block forward runs M>2 kernel paths (stock gather_qmm fallbacks)
        # whose retained rows differ at ulp scale from the M<=2 decode path,
        # surfacing as delayed argmax flips (windows 083910/085411).  Feeding
        # the copy match as the depth-1 draft keeps every forward on the
        # proven 2-row cycle: the accepted token is always the pre-sampled
        # target id, so the emitted stream is bit-exact for ANY draft source,
        # at any temperature.  MTP head compute is skipped during a streak.
        if ccopy_active and _ccopy_takes_over_lane:
            if _cc_src_idx is not None:
                for _cc_committed in tokens[_cc_src_check_from:]:
                    if _cc_src_idx < len(prompt_ids) and int(_cc_committed) == int(
                        prompt_ids[_cc_src_idx]
                    ):
                        _cc_src_idx += 1
                        # Acceptance stats count only tokens WE drafted; a
                        # bonus/primary token that happens to continue the
                        # prompt match advances the streak but is the
                        # verify's own win, not copy acceptance.
                        if _cc_streak_outstanding > 0:
                            _cc_streak_outstanding -= 1
                            _cc_streak_accepted += 1
                            ccopy_accepted += 1
                    else:
                        _cc_src_idx = None
                        _cc_streak_outstanding = 0
                        break
                if _cc_src_idx is not None and _cc_src_idx >= len(prompt_ids):
                    _cc_src_idx = None
                if _cc_src_idx is None:
                    # Streak over: same acceptance-EMA suspend/backoff contract
                    # as the round path, per streak.
                    _cc_ratio = (
                        _cc_streak_accepted / _cc_streak_drafted
                        if _cc_streak_drafted
                        else 0.0
                    )
                    ccopy_ema = 0.7 * ccopy_ema + 0.3 * min(1.0, _cc_ratio)
                    ccopy_seen += 1
                    if _cc_ratio >= 0.5:
                        ccopy_backoff = 64
                    if ccopy_seen >= 4 and ccopy_ema < 0.35:
                        ccopy_suspend_until = len(tokens) + ccopy_backoff
                        ccopy_backoff = min(ccopy_backoff * 2, 4096)
                        ccopy_ema, ccopy_seen = 0.5, 0
                        ccopy_suspensions += 1
            _cc_src_check_from = len(tokens)
            if _cc_src_idx is None and len(tokens) >= ccopy_suspend_until:
                ccopy_probes += 1
                _cc_pos, _cc_ext = ccopy_index.find(
                    prompt_ids + tokens, max_pos=len(prompt_ids)
                )
                if (
                    _cc_pos is not None
                    and _cc_ext >= ccopy_min_ext
                    and int(_cc_pos) < len(prompt_ids)
                ):
                    _cc_src_idx = int(_cc_pos)
                    _cc_streak_drafted = 0
                    _cc_streak_accepted = 0
                    _cc_streak_outstanding = 0
                    ccopy_rounds += 1
                    event["context_copy"] = {
                        "mode": "draft_source",
                        "extension": int(_cc_ext),
                        "at_tokens": len(tokens),
                        "block": 0,
                        "accepted": 0,
                        "correction": None,
                    }
        # ---- context-copy round: verbatim block from context, no MTP compute this cycle ----
        if ccopy_active and _ccopy_capture_lane and cycle_depth >= 1 and len(tokens) >= ccopy_suspend_until:
            _cc_hist = prompt_ids + tokens
            ccopy_probes += 1
            # Prompt-only contract: candidates whose continuation starts at the
            # prompt edge are dropped inside find() (the best VALID match wins),
            # and the block is sliced from the prompt and capped at its
            # boundary — never from already-generated output (self-repetition).
            _cc_pos, _cc_ext = ccopy_index.find(_cc_hist, max_pos=len(prompt_ids))
            _cc_block: list[int] = []
            if _cc_pos is not None and _cc_ext >= ccopy_min_ext:
                _cc_klen = block_for_ext(_cc_ext, ccopy_k)
                _cc_block = [int(t) for t in prompt_ids[_cc_pos:_cc_pos + _cc_klen]]
                _cc_block = _cc_block[: max(1, max_tokens - len(tokens))]
                if constraint is not None:
                    # Truncate the copy proposal at the first grammar-illegal
                    # token so masked rejections stay rare instead of
                    # systematic; an empty result falls through to the normal
                    # MTP round (#186 phase 3).
                    _cc_block = _cc_block[: constraint.validate_prefix(_cc_block)]
            if _cc_block:
                _cc_T = 1 + len(_cc_block)
                _cc_before = None
                if not _env_truthy("MTPLX_SKIP_VERIFY_SNAPSHOT"):
                    started = time.perf_counter()
                    _cc_before = snapshot_untrimmable_cache(cache)
                    snapshot_time += time.perf_counter() - started
                started_forward = time.perf_counter()
                with attention_phase("decode_verify"):
                    _cc_logits, _cc_hidden, _cc_captures = rt.forward_ar_capture(
                        mx.array([[primary] + _cc_block]),
                        cache=cache,
                        return_hidden=True,
                        hidden_variant=base_hidden_variant,
                        capture_backend=verify_core_backend,
                    )
                if sampler.temperature <= 0:
                    _cc_g = [int(x) for x in mx.argmax(_cc_logits[0], axis=-1).tolist()]
                else:
                    mx.eval(_cc_logits)
                elapsed_verify = time.perf_counter() - started_forward
                verify_forward_time += elapsed_verify
                verify_time += elapsed_verify
                target_time += elapsed_verify
                verify_calls += 1
                _cc_correction: int | None = None
                if sampler.temperature <= 0:
                    _cc_nacc = 0
                    for _cc_d, _cc_t in zip(_cc_block, _cc_g):
                        if _cc_d == _cc_t:
                            _cc_nacc += 1
                        else:
                            break
                else:
                    # The copy block is a point-mass proposal, so each copied
                    # token is accepted with the target's own shaped
                    # probability of that token, and a rejection samples the
                    # residual (target with the copied token's mass removed,
                    # renormalized). Identical probability-ratio contract to
                    # the MTP verify path: the emitted stream follows the
                    # target sampling distribution exactly at any temperature.
                    _cc_nacc = 0
                    _cc_vocab = int(_cc_logits.shape[-1])
                    for _cc_i, _cc_d in enumerate(_cc_block):
                        _cc_target_p = _distribution_from_mlx_logits(
                            _cc_logits[0, _cc_i],
                            sampler,
                            token_counts=None,
                        )
                        _cc_draft_q = SparseDistribution(
                            np.array([int(_cc_d)], dtype=np.int64),
                            np.array([1.0], dtype=np.float64),
                            _cc_vocab,
                        )
                        _cc_accept_prob = compute_acceptance_probability(
                            _cc_target_p, _cc_draft_q, int(_cc_d)
                        )
                        if float(rng.random()) <= _cc_accept_prob:
                            _cc_nacc += 1
                            continue
                        _cc_correction = int(
                            sample_from_distribution(
                                residual_distribution(_cc_target_p, _cc_draft_q),
                                rng,
                            )
                        )
                        break
                # An accepted stop token ends the response: never accept, commit,
                # or select state past it (mirrors the MTP acceptance loop's stop
                # break). Every downstream boundary — capture-commit trim, the
                # logits/hidden row, MTP history, and the emitted tokens — derives
                # from _cc_nacc, so capping it here keeps them all at the stop. A
                # rejection past the stop is void: the response is already over.
                for _cc_i in range(_cc_nacc):
                    if _is_stop(int(_cc_block[_cc_i]), stop_token_ids):
                        _cc_nacc = _cc_i + 1
                        _cc_correction = None
                        break
                _cc_m = _cc_nacc + 1
                _cc_ok = True
                if _cc_nacc < len(_cc_block):
                    from .gdn_capture import commit_captured_prefix
                    started_commit = time.perf_counter()
                    _cc_ok = commit_captured_prefix(
                        cache, _cc_captures, keep_tokens=_cc_m, verified_tokens=_cc_T,
                    )
                    capture_commit_time += time.perf_counter() - started_commit
                if not _cc_ok:
                    # This capture core cannot commit a per-position prefix (for
                    # example final-state-only cores). Roll the whole block back,
                    # restore the primary's logits, and stop proposing copies.
                    if _cc_before is None:
                        raise RuntimeError(
                            "context-copy: capture commit failed and the verify "
                            "snapshot was skipped (MTPLX_SKIP_VERIFY_SNAPSHOT=1)"
                        )
                    started_rollback = time.perf_counter()
                    rollback_after_verify(cache, _cc_before, verified_tokens=_cc_T)
                    rollback_time += time.perf_counter() - started_rollback
                    started = time.perf_counter()
                    with attention_phase("decode_verify"):
                        _cc_l2, _cc_h2 = rt.forward_ar(
                            mx.array([[primary]]),
                            cache=cache,
                            return_hidden=True,
                            hidden_variant=base_hidden_variant,
                        )
                    _eval(_cc_l2, _cc_h2)
                    repair_time += time.perf_counter() - started
                    logits = _cc_l2[:, -1, :]
                    hidden = _cc_h2[:, -1:, :]
                    ccopy_active = False
                    ccopy_disabled_reason = "no_per_position_commit"
                    event["context_copy"] = {"disabled": "no_per_position_commit"}
                    append_event(event)
                    continue
                _cc_round_pos = len(tokens)
                _cc_acc = _cc_block[:_cc_nacc]
                _cc_stop_idx = next((i for i, t in enumerate(_cc_acc)
                                     if _is_stop(int(t), stop_token_ids)), None)
                if _cc_stop_idx is not None:
                    _cc_acc = _cc_acc[:_cc_stop_idx + 1]
                tokens.extend(_cc_acc)
                _cc_finished = _cc_stop_idx is not None
                if constraint is not None and _cc_correction is not None and (
                    constraint.validate_prefix([*_cc_acc, int(_cc_correction)])
                    != len(_cc_acc) + 1
                ):
                    # Grammar-illegal residual: drop it; the next cycle's
                    # masked primary resamples the position, which preserves
                    # the masked target law exactly.
                    _cc_correction = None
                if _cc_correction is not None and not _cc_finished:
                    # Exactness requires the rejected position's token to be
                    # the residual sample drawn above, not a fresh draw from
                    # the full distribution next cycle. Emit it now and defer
                    # its forward exactly like an MTP rejection: the pending
                    # primary's KV is computed by whichever forward runs next.
                    tokens.append(int(_cc_correction))
                    correction_tokens += 1
                    pending_primary = int(_cc_correction)
                    if _is_stop(int(_cc_correction), stop_token_ids):
                        _cc_finished = True
                ccopy_rounds += 1
                ccopy_drafted += len(_cc_block)
                ccopy_accepted += _cc_nacc
                if _cc_nacc:
                    ccopy_blocks_accepted += 1
                ccopy_ema = 0.7 * ccopy_ema + 0.3 * (_cc_nacc / len(_cc_block))
                ccopy_seen += 1
                if _cc_nacc / len(_cc_block) >= 0.5:
                    ccopy_backoff = 64          # copy is paying again: full retry rate
                if ccopy_seen >= 4 and ccopy_ema < 0.35:
                    # acceptance collapsed (novel region with incidental repeats):
                    # suspend copy rounds and let the MTP head work; retry with
                    # exponential backoff so recurring probes stay cheap
                    ccopy_suspend_until = len(tokens) + ccopy_backoff
                    ccopy_backoff = min(ccopy_backoff * 2, 4096)
                    ccopy_ema, ccopy_seen = 0.5, 0
                    ccopy_suspensions += 1
                event["context_copy"] = {
                    "block": len(_cc_block),
                    "accepted": _cc_nacc,
                    "extension": int(_cc_ext),
                    "time_s": float(elapsed_verify),
                    "correction": (
                        int(_cc_correction) if _cc_correction is not None else None
                    ),
                    # Completion-stream position of the round (tokens emitted
                    # BEFORE this round's block landed): byte-exactness gates
                    # correlate a divergence index with round windows to tell
                    # an accept/continuation fault from post-commit state
                    # corruption.
                    "at_tokens": int(_cc_round_pos),
                }
                # Committed-history MTP caches pair every committed token with the
                # hidden state of the token before it, including (previous hidden,
                # primary), which the drafting path would normally have added.
                if _mtp_history_uses_committed_cache(mtp_history_policy) and mtp_cache is not None:
                    _cc_committed_toks = [primary] + _cc_acc
                    _cc_hiddens = mx.concatenate(
                        [hidden, _cc_hidden[:, : len(_cc_acc), :]], axis=1
                    )
                    draft_time += append_mtp_history(
                        mtp_cache, _cc_hiddens, _cc_committed_toks
                    )
                logits = _cc_logits[:, _cc_m - 1, :]
                hidden = _cc_hidden[:, _cc_m - 1:_cc_m, :]
                append_event(event)
                emit_new_tokens()
                if _cc_finished:
                    break
                continue
        draft_hidden = hidden
        next_token = primary
        device_draft_token = None

        # Copy-streak draft substitution: propose the prompt continuation as
        # this cycle's depth-1 draft and skip MTP head compute entirely.  The
        # compiled route keeps its device-draft contract (no substitution).
        _cc_draft_source_token: int | None = None
        if (
            _cc_src_idx is not None
            and a3b_target_prefix_route is None
            and cycle_depth == 1
        ):
            _cc_draft_source_token = int(prompt_ids[_cc_src_idx])
            _cc_streak_drafted += 1
            _cc_streak_outstanding += 1
            ccopy_drafted += 1

        used_device_d2_core = False
        device_d2_eligible = (
            _cc_draft_source_token is None
            and draft_core == "device-d2"
            and cycle_depth == 2
            and speculative_depth == 2
            and mtp_cache_policy == "persistent"
            and mtp_history_policy == "cycle"
            and draft_sampler.temperature <= 0
            and draft_margin_threshold is None
            and adaptive_policy is None
            and mtp_corrector is None
            and not online_hidden_enabled
            and not online_correction_cache
        )
        if device_d2_eligible:
            try:
                if device_d2_core is None:
                    compile_started = time.perf_counter()
                    device_d2_core = _make_device_d2_draft_core(
                        rt,
                        draft_hidden,
                        mx.array([[primary]]),
                        mtp_hidden_variant=mtp_hidden_variant,
                    )
                    elapsed_compile = time.perf_counter() - compile_started
                    device_d2_compile_time += elapsed_compile
                    draft_time += elapsed_compile
                    _add_timing(event, "draft_core_compile", elapsed_compile)
                    event["draft_core_compile"] = {
                        "kind": "device-d2",
                        "mtp_cache_promoted": int(device_d2_core["promoted"]),
                        "promotion_failures": dict(
                            device_d2_core["promotion_failures"]
                        ),
                    }
                started = time.perf_counter()
                draft_tokens = _run_device_d2_draft_core(
                    device_d2_core,
                    draft_hidden,
                    int(primary),
                )
                elapsed_draft = time.perf_counter() - started
                draft_time += elapsed_draft
                device_d2_calls += 1
                used_device_d2_core = True
                for depth_index, draft_token in enumerate(draft_tokens):
                    draft_probs.append(
                        SparseDistribution.one_hot(
                            draft_token,
                            int(logits.shape[-1]),
                        )
                        if sampler.temperature > 0
                        else None
                    )
                    drafted += 1
                    drafted_by_depth[depth_index] += 1
                    event["drafts"].append(
                        {
                            "depth": depth_index + 1,
                            "token": int(draft_token),
                            "timing_s": {
                                "draft": elapsed_draft
                                if depth_index == len(draft_tokens) - 1
                                else 0.0,
                            },
                            "mtp_corrector": None,
                            "draft_core": "device-d2",
                        }
                    )
                next_token = draft_tokens[-1]
            except Exception as exc:
                device_d2_fallbacks += 1
                event["draft_core_error"] = repr(exc)
                used_device_d2_core = False

        if not used_device_d2_core:
            if draft_core == "device-d2" and not device_d2_eligible:
                device_d2_fallbacks += 1
                event["draft_core_fallback"] = {
                    "requested": "device-d2",
                    "reason": "ineligible_contract",
                }

        used_device_core = used_device_d2_core
        a3b_k2 = (
            a3b_target_prefix_route is not None
            and int(getattr(a3b_target_prefix_route, "speculative_depth", 1)) == 2
        )
        if a3b_k2 and not used_device_core:
            # k=2 compiled path: produce the two chained greedy MTP drafts
            # [d1, d2] on-device (one host sync) BEFORE the draft loop, then
            # skip the loop (used_device_core).  d2 chains from d1's hidden --
            # the same single-module recurrence characterized for a2~0.45; we
            # measure the 3-row verify cost, and commits stay target-argmax so
            # the greedy stream is byte-exact vs generate_ar regardless of a2.
            k2_started = time.perf_counter()
            if compiled_k2_d2_core is None:
                compiled_k2_d2_core = _make_device_d2_draft_core(
                    rt,
                    draft_hidden,
                    mx.array([[primary]]),
                    mtp_hidden_variant=mtp_hidden_variant,
                )
            _k2_drafts = _run_device_d2_draft_core(
                compiled_k2_d2_core, draft_hidden, int(primary)
            )
            draft_time += time.perf_counter() - k2_started
            draft_tokens = [int(_k2_drafts[0]), int(_k2_drafts[1])]
            draft_probs = [None, None]
            for _k2_depth, _k2_tok in enumerate(draft_tokens):
                drafted += 1
                drafted_by_depth[_k2_depth] += 1
                event["drafts"].append(
                    {
                        "depth": _k2_depth + 1,
                        "token": _k2_tok,
                        "timing_s": {"draft": 0.0},
                        "mtp_corrector": None,
                        "draft_core": "compiled-k2-d2",
                    }
                )
            next_token = draft_tokens[-1]
            used_device_core = True
        if _cc_draft_source_token is not None:
            # Copy streak owns this cycle's draft: one host token, no MTP
            # forward.  The accept path is draft-source-agnostic (the
            # accepted token is always the pre-sampled target id).
            draft_tokens = [int(_cc_draft_source_token)]
            draft_probs = [None]
            next_token = int(_cc_draft_source_token)
            used_device_core = True  # skip the host MTP drafting loop below
            event["drafts"].append(
                {
                    "depth": 1,
                    "token": int(_cc_draft_source_token),
                    "timing_s": {"draft": 0.0},
                    "mtp_corrector": None,
                    "draft_core": "context_copy",
                }
            )
        elif not used_device_core and draft_core == "device":
            device_core_eligible = (
                2 <= cycle_depth <= 5
                and cycle_depth == speculative_depth
                and mtp_cache is not None
                and _mtp_history_uses_committed_cache(mtp_history_policy)
                and draft_margin_threshold is None
                and adaptive_policy is None
                and mtp_corrector is None
                and mtp_topk_reranker is None
                and not adapter_ensemble_q
                and not online_hidden_enabled
                and not correction_cache_enabled
                and not target_prefix_verify
                and (
                    draft_sampler.temperature <= 0
                    or 0 < draft_sampler.top_k <= _DEVICE_CORE_MAX_TOP_K
                )
            )
            if device_core_eligible:
                try:
                    live_signature = _device_core_state_signature(mtp_cache)
                    core_current = (
                        device_core is not None
                        and int(device_core["depth"]) == int(cycle_depth)
                        and device_core["state_signature"] == live_signature
                    )
                    if (
                        not core_current
                        and device_core is not None
                        and os.environ.get("MTPLX_DEVICE_CORE_DEBUG")
                    ):
                        stored = device_core["state_signature"]
                        diffs = [
                            (i, stored[i] if i < len(stored) else None,
                             live_signature[i] if i < len(live_signature) else None)
                            for i in range(max(len(stored), len(live_signature)))
                            if (stored[i] if i < len(stored) else None)
                            != (live_signature[i] if i < len(live_signature) else None)
                        ]
                        print(
                            f"[device-core] signature diff ({len(diffs)} leaves): "
                            f"{diffs[:4]}",
                            file=sys.stderr,
                            flush=True,
                        )
                    if not core_current:
                        compile_started = time.perf_counter()
                        device_core = _make_device_draft_core(
                            rt,
                            draft_hidden,
                            mx.array([[primary]]),
                            mtp_hidden_variant=mtp_hidden_variant,
                            depth=cycle_depth,
                            mtp_cache=mtp_cache,
                            draft_sampler=draft_sampler,
                            seed=int(rng.integers(0, 2**31 - 1)),
                        )
                        elapsed_compile = time.perf_counter() - compile_started
                        device_core_compile_time += elapsed_compile
                        draft_time += elapsed_compile
                        _add_timing(event, "draft_core_compile", elapsed_compile)
                        event["draft_core_compile"] = {
                            "kind": "device",
                            "depth": int(cycle_depth),
                            "mtp_cache_promoted": int(device_core["promoted"]),
                            "promotion_failures": dict(
                                device_core["promotion_failures"]
                            ),
                        }
                    started = time.perf_counter()
                    core_tokens, core_qs = _run_device_draft_core(
                        device_core,
                        draft_hidden,
                        int(primary),
                        seed=int(rng.integers(0, 2**31 - 1)),
                    )
                    elapsed_draft = time.perf_counter() - started
                    draft_time += elapsed_draft
                    device_core_calls += 1
                    draft_tokens = list(core_tokens)
                    for depth_index, (draft_token, draft_q) in enumerate(
                        zip(core_tokens, core_qs)
                    ):
                        draft_probs.append(
                            draft_q if sampler.temperature > 0 else None
                        )
                        drafted += 1
                        drafted_by_depth[depth_index] += 1
                        event["drafts"].append(
                            {
                                "depth": depth_index + 1,
                                "token": int(draft_token),
                                "timing_s": {
                                    "draft": elapsed_draft
                                    if depth_index == len(core_tokens) - 1
                                    else 0.0,
                                },
                                "mtp_corrector": None,
                                "draft_core": "device",
                            }
                        )
                    next_token = draft_tokens[-1]
                    used_device_core = True
                except Exception as exc:
                    device_core_fallbacks += 1
                    event["draft_core_error"] = repr(exc)
                    used_device_core = False
            else:
                device_core_fallbacks += 1
                event["draft_core_fallback"] = {
                    "requested": "device",
                    "reason": "ineligible_contract",
                }
        for depth_index in range(0 if used_device_core else cycle_depth):
            source_token = int(next_token)
            step_mtp_cache = (
                mtp_cache if mtp_cache_policy == "persistent" else rt.make_mtp_cache()
            )
            draft_position_offset = mtp_position_offset_for_cache(step_mtp_cache)
            started = time.perf_counter()
            cache_depth = depth_index + 1
            ensemble_info: dict[str, Any] | None = None
            ensemble_base_logits = None
            ensemble_adapter_logits = None
            ensemble_base_hidden = None
            ensemble_adapter_hidden = None
            ensemble_eligible = (
                adapter_ensemble_q
                and rt.mtp_adapter_path is not None
                and sampler.temperature > 0
                and draft_sampler.temperature <= 0
                and cache_depth >= adapter_ensemble_min_depth
                and cache_depth == cycle_depth
                and mtp_cache_policy == "persistent"
                and mtp_history_policy == "cycle"
                and step_mtp_cache is not None
            )
            if ensemble_eligible:
                cache_offset = _mtp_cache_offset(step_mtp_cache)
                base_result = rt.draft_mtp(
                    draft_hidden,
                    mx.array([[next_token]]),
                    mtp_cache=step_mtp_cache,
                    return_hidden=True,
                    mtp_hidden_variant=mtp_hidden_variant,
                    mtp_depth=0,
                    position_offset=draft_position_offset,
                )
                ensemble_base_logits, ensemble_base_hidden = base_result
                _eval(ensemble_base_logits, ensemble_base_hidden)
                _rollback_mtp_cache(step_mtp_cache, cache_offset)
                adapter_result = rt.draft_mtp(
                    draft_hidden,
                    mx.array([[next_token]]),
                    mtp_cache=step_mtp_cache,
                    return_hidden=True,
                    mtp_hidden_variant=mtp_hidden_variant,
                    mtp_depth=cache_depth,
                    position_offset=draft_position_offset,
                )
                ensemble_adapter_logits, ensemble_adapter_hidden = adapter_result
                draft_logits, draft_hidden_next = adapter_result
            else:
                if adapter_ensemble_q and cache_depth >= adapter_ensemble_min_depth:
                    adapter_ensemble_fallbacks += 1
                draft_result = rt.draft_mtp(
                    draft_hidden,
                    mx.array([[next_token]]),
                    mtp_cache=step_mtp_cache,
                    return_hidden=True,
                    mtp_hidden_variant=mtp_hidden_variant,
                    mtp_depth=cache_depth,
                    position_offset=draft_position_offset,
                )
                draft_logits, draft_hidden_next = draft_result
            wants_policy_metrics = bool(
                getattr(adaptive_policy, "wants_draft_metrics", False)
            )
            draft_metrics = (
                _draft_confidence_metrics(draft_logits[:, -1, :][0])
                if draft_margin_threshold is not None or wants_policy_metrics
                else {}
            )
            margin = draft_metrics.get("top2_margin")
            if (
                draft_margin_threshold is not None
                and margin is not None
                and margin < draft_margin_threshold
                and depth_index >= min_speculative_depth
            ):
                event["gated_stop_depth"] = depth_index + 1
                event["drafts"].append(
                    {
                        "depth": depth_index + 1,
                        "top2_margin": margin,
                        "speculation_skipped": True,
                    }
                )
                draft_time += time.perf_counter() - started
                break
            cache_key = _online_correction_cache_key(
                online_correction_cache_key,
                depth=cache_depth,
                primary=int(primary),
                source_token=source_token,
                draft_prefix=draft_tokens,
            )
            cache_enabled_for_depth = correction_cache_enabled and (
                (
                    online_correction_cache
                    and cache_depth >= online_correction_cache_min_depth
                )
                or (
                    prompt_correction_cache
                    and cache_depth >= prompt_correction_cache_min_depth
                )
            )
            reranker_info = None
            adaptive_width_stop = False
            cached_token = (
                correction_cache.get(cache_key) if cache_enabled_for_depth else None
            )
            if a3b_target_prefix_route is not None:
                device_draft_token = sample_token_ids_from_mlx_logits(
                    draft_logits[:, -1, :],
                    draft_sampler,
                )
                draft_token = None
                draft_q = None
            elif cached_token is not None:
                draft_token = int(cached_token)
                draft_q = (
                    SparseDistribution.one_hot(draft_token, int(draft_logits.shape[-1]))
                    if sampler.temperature > 0
                    else None
                )
                correction_cache_hits += 1
                if cache_key in prompt_seeded_cache_keys:
                    prompt_correction_cache_hits += 1
            elif (
                ensemble_eligible
                and ensemble_base_logits is not None
                and ensemble_adapter_logits is not None
            ):
                draft_token, draft_q, ensemble_info = _sample_adapter_ensemble_q(
                    ensemble_base_logits[:, -1, :][0],
                    ensemble_adapter_logits[:, -1, :][0],
                    epsilon=adapter_ensemble_epsilon,
                    rng=rng,
                )
                adapter_ensemble_calls += 1
                if bool(ensemble_info["changed"]):
                    adapter_ensemble_changed += 1
                selected = str(ensemble_info["selected"])
                if selected == "adapter":
                    adapter_ensemble_adapter_selected += 1
                    draft_hidden_next = ensemble_adapter_hidden
                    draft_logits = ensemble_adapter_logits
                elif selected == "base":
                    adapter_ensemble_base_selected += 1
                    draft_hidden_next = ensemble_base_hidden
                    draft_logits = ensemble_base_logits
                else:
                    adapter_ensemble_shared_selected += 1
                    draft_hidden_next = ensemble_adapter_hidden
                    draft_logits = ensemble_adapter_logits
            else:
                if (
                    mtp_topk_reranker is not None
                    and sampler.temperature > 0
                    and cache_depth in mtp_topk_reranker.depth_priors
                ):
                    reranked = mtp_topk_reranker.select(
                        draft_logits[:, -1, :][0],
                        depth=cache_depth,
                    )
                    if reranked is not None:
                        draft_token, reranker_info = reranked
                        draft_q = SparseDistribution.one_hot(
                            draft_token,
                            int(draft_logits.shape[-1]),
                        )
                        topk_reranker_calls += 1
                        if bool(reranker_info["changed"]):
                            topk_reranker_changed += 1
                        topk_reranker_selected_rank_sum += int(
                            reranker_info["selected_rank"]
                        )
                    else:
                        topk_reranker_fallbacks += 1
                        draft_token, draft_q = _sample_draft_from_logits(
                            draft_logits[:, -1, :][0],
                            draft_sampler,
                            rng,
                            need_distribution=(
                                sampler.temperature > 0 and not target_prefix_verify
                            ),
                        )
                else:
                    need_draft_distribution = (
                        sampler.temperature > 0 and not target_prefix_verify
                    )
                    draft_token, draft_q, adaptive_width_stop = (
                        cycle_draft_reader(
                            draft_logits,
                            depth_index=depth_index,
                            need_distribution=need_draft_distribution,
                            decision_margins=adaptive_width_decision_margins,
                        )
                    )
            elapsed_draft = time.perf_counter() - started
            draft_time += elapsed_draft
            if trace.enabled:
                trace_accounting_started = time.perf_counter()
                trace_draft_output_nbytes += _tree_nbytes(draft_logits) + _tree_nbytes(
                    draft_hidden_next
                )
                if (
                    ensemble_base_logits is not None
                    and ensemble_base_logits is not draft_logits
                ):
                    trace_draft_output_nbytes += _tree_nbytes(ensemble_base_logits)
                if (
                    ensemble_base_hidden is not None
                    and ensemble_base_hidden is not draft_hidden_next
                ):
                    trace_draft_output_nbytes += _tree_nbytes(ensemble_base_hidden)
                trace_accounting_time_s += (
                    time.perf_counter() - trace_accounting_started
                )
            draft_tokens.append(draft_token)
            draft_probs.append(draft_q)
            draft_cache_keys.append(cache_key)
            draft_hidden_base = draft_hidden_next[:, -1:, :]
            if mtp_corrector is not None:
                draft_hidden_base = mtp_corrector.apply_mlx(
                    draft_hidden_base,
                    depth=depth_index + 1,
                )
            feed_depth = depth_index + 1
            draft_hidden_for_update.append(draft_hidden_base)
            online_key: object = (
                (feed_depth, source_token)
                if online_hidden_corrector_key == "token"
                else feed_depth
            )
            draft_hidden_update_keys.append(online_key)
            draft_hidden = draft_hidden_base
            online_draft_event: dict[str, object] | None = None
            if (
                online_hidden_enabled
                and feed_depth <= online_hidden_max_feed_depth
                and feed_depth < cycle_depth
            ):
                started_online = time.perf_counter()
                update_count = online_hidden_update_counts.get(online_key, 0)
                delta = online_hidden_deltas.get(online_key)
                if delta is not None and update_count >= online_hidden_corrector_warmup:
                    draft_hidden = draft_hidden + (
                        float(online_hidden_corrector_alpha)
                        * delta.astype(draft_hidden.dtype)
                    )
                    online_hidden_apply_counts[online_key] = (
                        online_hidden_apply_counts.get(online_key, 0) + 1
                    )
                    online_draft_event = {
                        "feed_depth": feed_depth,
                        "key": online_hidden_corrector_key,
                        "source_token": source_token
                        if online_hidden_corrector_key == "token"
                        else None,
                        "applied": True,
                        "updates": update_count,
                        "apply_count": online_hidden_apply_counts[online_key],
                    }
                else:
                    online_draft_event = {
                        "feed_depth": feed_depth,
                        "key": online_hidden_corrector_key,
                        "source_token": source_token
                        if online_hidden_corrector_key == "token"
                        else None,
                        "applied": False,
                        "updates": update_count,
                    }
                online_hidden_corrector_time += time.perf_counter() - started_online
            next_token = draft_token
            drafted += 1
            drafted_by_depth[depth_index] += 1
            draft_event = {
                "depth": depth_index + 1,
                "token": draft_token,
                "timing_s": {"draft": elapsed_draft},
                "mtp_corrector": getattr(mtp_corrector, "kind", None)
                if mtp_corrector is not None
                else None,
                **draft_metrics,
            }
            if draft_position_offset is not None:
                draft_event["position_offset"] = int(draft_position_offset)
            if correction_cache_enabled:
                draft_event["online_correction_cache"] = {
                    "hit": cached_token is not None,
                    "enabled_for_depth": cache_enabled_for_depth,
                    "key_policy": online_correction_cache_key,
                    "key": list(cache_key),
                    "cached_token": int(cached_token)
                    if cached_token is not None
                    else None,
                    "prompt_seeded": cache_key in prompt_seeded_cache_keys,
                }
            if ensemble_info is not None:
                draft_event["adapter_ensemble_q"] = ensemble_info
            if reranker_info is not None:
                draft_event["mtp_topk_reranker"] = reranker_info
            if online_draft_event is not None:
                draft_event["online_hidden_corrector"] = online_draft_event
            event["drafts"].append(draft_event)
            if adaptive_width_stop:
                event["gated_stop_depth"] = depth_index + 1
                break
            if adaptive_policy is not None and hasattr(
                adaptive_policy, "should_continue_after_draft"
            ):
                policy_continue = adaptive_policy.should_continue_after_draft(
                    drafted_depth=depth_index + 1,
                    max_depth=cycle_depth,
                    draft_metrics=event["drafts"][-1],
                )
                event["drafts"][-1]["policy_continue"] = policy_continue
                if not bool(policy_continue.get("continue", True)):
                    event["gated_stop_depth"] = depth_index + 1
                    event["policy_stop"] = policy_continue
                    break

        record_adaptive_width_event(
            event,
            cycle_depth=cycle_depth,
            decision_margins=adaptive_width_decision_margins,
            selected_draft_depth=len(draft_tokens),
        )

        before_verify = None
        if a3b_target_prefix_route is None:
            if _skip_verify_snapshot():
                event["snapshot"] = "skipped_capture_commit_required"
            else:
                started = time.perf_counter()
                before_verify = snapshot_untrimmable_cache(cache)
                elapsed_snapshot = time.perf_counter() - started
                snapshot_time += elapsed_snapshot
                _add_timing(event, "snapshot", elapsed_snapshot)
        lazy_bonus_verify_min_depth = _lazy_bonus_verify_min_depth()
        lazy_bonus_verify_requested = _lazy_bonus_verify_enabled()
        omit_speculative_bonus = _omit_speculative_bonus_enabled()
        if a3b_target_prefix_route is not None and a3b_k2:
            # 3-row verify [primary, d1, d2]; greedy needs all 3 target rows so
            # the accept loop can commit a1/a2/a3 and pick the rebase point.
            lazy_bonus_verify = False
            bonus_distribution_row_needed = (
                not omit_speculative_bonus and len(tokens) + 1 < max_tokens
            )
            target_distribution_rows_needed = 3
            verified_token_count = 3
            verify_input_array = mx.array([[int(primary), *draft_tokens]])
        elif a3b_target_prefix_route is not None:
            lazy_bonus_verify = False
            bonus_distribution_row_needed = (
                not omit_speculative_bonus and len(tokens) + 1 < max_tokens
            )
            target_distribution_rows_needed = 1 + int(
                bonus_distribution_row_needed
            )
            verified_token_count = 2
            verify_input_array = mx.concatenate(
                (mx.array([[primary]]), device_draft_token.reshape(1, 1)),
                axis=1,
            )
        else:
            lazy_bonus_verify = (
                lazy_bonus_verify_requested
                and not lazy_target_distributions
                and not target_prefix_verify
                and len(draft_tokens) > 0
                and len(draft_tokens) >= lazy_bonus_verify_min_depth
                and not any(
                    _is_stop(token, stop_token_ids) for token in draft_tokens[:-1]
                )
            )
            bonus_distribution_row_needed = (
                not omit_speculative_bonus
                and not lazy_bonus_verify
                and len(draft_tokens) > 0
                and len(tokens) + len(draft_tokens) < max_tokens
                and not any(_is_stop(token, stop_token_ids) for token in draft_tokens)
            )
            target_distribution_rows_needed = len(draft_tokens) + (
                1 if bonus_distribution_row_needed else 0
            )
            verify_input = [primary] + (
                draft_tokens[:-1] if lazy_bonus_verify else draft_tokens
            )
            verified_token_count = len(verify_input)
            verify_input_array = mx.array([verify_input])
        if lazy_bonus_verify:
            lazy_bonus_verify_calls += 1
        event["lazy_bonus_verify"] = {
            "enabled": bool(lazy_bonus_verify),
            "requested": bool(lazy_bonus_verify_requested),
            "disabled_by": "lazy_target_distributions"
            if lazy_bonus_verify_requested
            and lazy_target_distributions
            and not target_prefix_verify
            else None,
            "min_depth": int(lazy_bonus_verify_min_depth),
            "verify_input_tokens": int(verified_token_count),
            "draft_tokens": int(len(draft_tokens)),
        }
        event["speculative_bonus"] = {
            "omitted": bool(omit_speculative_bonus),
            "distribution_row_needed": bool(bonus_distribution_row_needed),
        }
        set_native_mlp_context(len(tokens))
        started_forward = time.perf_counter()
        captures = None
        with (
            attention_phase("decode_verify"),
            model_forward_kind("target_verify"),
        ):
            if verify_strategy in {"capture_commit", "graphbank_capture_commit"}:
                if compiled_verify_bank is not None:
                    verify_logits, verify_hidden, captures = (
                        compiled_verify_bank.forward_ar_capture(
                            verify_input_array,
                            cache=cache,
                            return_hidden=True,
                            hidden_variant=base_hidden_variant,
                        )
                    )
                elif graphbank is not None:
                    verify_logits, verify_hidden, captures = (
                        graphbank.forward_ar_capture(
                            verify_input_array,
                            cache=cache,
                            return_hidden=True,
                            hidden_variant=base_hidden_variant,
                        )
                    )
                else:
                    capture_forward = capture_forward_routes[len(draft_tokens) - 1]
                    verify_logits, verify_hidden, captures = capture_forward(
                        verify_input_array,
                        cache=cache,
                        return_hidden=True,
                        hidden_variant=base_hidden_variant,
                        capture_backend=verify_core_backend,
                    )
            elif a3b_target_prefix_route is not None and a3b_k2:
                # k=2 3-row verify.  Returns the two mid-window rebase states
                # (post-row-0, post-row-1); the accept loop picks which one the
                # next cycle rebases from after a d1 or d2 reject.
                if a3b_rebase_state is not None:
                    (
                        verify_logits,
                        verify_hidden,
                        a3b_m3_rebase0_state,
                        a3b_m3_rebase1_state,
                    ) = a3b_target_prefix_route.verify_m3_rebased(
                        verify_input_array, a3b_rebase_state
                    )
                    a3b_rebase_state = None
                else:
                    (
                        verify_logits,
                        verify_hidden,
                        a3b_m3_rebase0_state,
                        a3b_m3_rebase1_state,
                    ) = a3b_target_prefix_route.verify_m3(verify_input_array)
                # a3b_primary_state (the K1 single-rebase leaf) is unused on the
                # k=2 path: the reject rebase selects m3 rebase0/rebase1 instead.
            elif a3b_target_prefix_route is not None:
                if a3b_rebase_state is not None:
                    # Deferred-correction fold: the pending correction is
                    # this cycle's primary and the verify runs from the
                    # stashed post-primary state of the cycle that rejected
                    # it -- the repair_m1 forward never happens.
                    verify_logits, verify_hidden, a3b_primary_state = (
                        a3b_target_prefix_route.verify_m2_rebased(
                            verify_input_array, a3b_rebase_state
                        )
                    )
                    a3b_rebase_state = None
                else:
                    verify_logits, verify_hidden, a3b_primary_state = (
                        a3b_target_prefix_route.verify_m2(verify_input_array)
                    )
            elif compiled_verify_bank is not None:
                # Replace only the target forward. target_prefix keeps its
                # authoritative snapshot/trim, pre-sampling, and correction
                # forward; captures here must not change its commit semantics.
                verify_logits, verify_hidden, _compiled_captures = (
                    compiled_verify_bank.forward_ar_capture(
                        verify_input_array,
                        cache=cache,
                        return_hidden=True,
                        hidden_variant=base_hidden_variant,
                    )
                )
            elif graphbank is not None:
                verify_logits, verify_hidden = graphbank.forward_ar(
                    verify_input_array,
                    cache=cache,
                    return_hidden=True,
                    hidden_variant=base_hidden_variant,
                )
            else:
                verify_logits, verify_hidden = rt.forward_ar(
                    verify_input_array,
                    cache=cache,
                    return_hidden=True,
                    hidden_variant=base_hidden_variant,
                )
        elapsed_verify_forward = time.perf_counter() - started_forward
        verify_forward_time += elapsed_verify_forward
        _add_timing(event, "verify_forward", elapsed_verify_forward)
        target_distribution_batch = None
        target_distributions = None
        target_prefix_tokens: list[int] | None = None
        target_distribution_precomputed = False
        elapsed_target_distribution_eval = 0.0
        started_eval = time.perf_counter()
        if target_prefix_verify:
            target_distribution_rows = min(
                int(verify_logits.shape[1]),
                target_distribution_rows_needed,
            )
            target_distribution_logits = verify_logits[:, :target_distribution_rows, :]
            started_distribution = time.perf_counter()
            if _steer_active:
                # Steering on the target_prefix lane (Loop Guard + Thinking
                # Guard): the accepted token is always the pre-sampled target
                # id, so overlays must land on the pre-sample logits. Row r
                # conditions on the committed tokens plus the in-block draft
                # prefix before position r.
                _guarded_rows = []
                for _row_index in range(int(target_distribution_rows)):
                    _row = target_distribution_logits[:, _row_index, :].reshape(-1)
                    _row_overlay = _steer_overlay(
                        [*tokens, *draft_tokens[:_row_index]]
                    )
                    if _row_overlay:
                        _row = apply_penalties_mlx(
                            _row, None, penalty_overlay=_row_overlay
                        )
                    _guarded_rows.append(_row)
                target_distribution_logits = mx.stack(_guarded_rows, axis=0)[
                    None, ...
                ]
            sampled_target_ids = sample_token_ids_from_mlx_logits(
                target_distribution_logits,
                sampler,
            )
            if a3b_target_prefix_route is not None and not a3b_k2:
                _eval(sampled_target_ids, device_draft_token)
                draft_token = int(np.asarray(device_draft_token).reshape(-1)[0])
                draft_tokens[0] = draft_token
                event["drafts"][0]["token"] = draft_token
            else:
                # k=2 (and non-compiled) already hold host-int draft tokens.
                _eval(sampled_target_ids)
            target_prefix_tokens = [
                int(token) for token in np.asarray(sampled_target_ids).reshape(-1)
            ]
            elapsed_target_distribution_eval = (
                time.perf_counter() - started_distribution
            )
            target_distribution_materialized_rows += int(target_distribution_rows)
            target_distribution_materialized_windows += 1
            verify_target_distribution_time += elapsed_target_distribution_eval
            target_distribution_precomputed = True
            event["target_distribution_materialized"] = {
                "mode": "target_prefix",
                "exact": True,
                "p_q_residual": False,
                "rows": int(target_distribution_rows),
                "time_s": float(elapsed_target_distribution_eval),
                "top_k": int(sampler.top_k),
            }
            if defer_verify_hidden_eval:
                verify_eval_timings = {
                    "verify_logits_eval_time_s": elapsed_target_distribution_eval,
                    "verify_hidden_eval_time_s": 0.0,
                    "verify_joint_eval_time_s": 0.0,
                }
                event["defer_verify_hidden_eval"] = {
                    "mode": "target_prefix",
                    "verify_hidden_mode": verify_hidden_mode,
                    "rows": int(target_distribution_rows),
                    "time_s": float(elapsed_target_distribution_eval),
                }
            elif captures is not None:
                verify_eval_timings = _eval_verify_outputs(
                    verify_logits, verify_hidden, captures
                )
            else:
                verify_eval_timings = _eval_verify_outputs(verify_logits, verify_hidden)
        elif (
            defer_verify_hidden_eval
            and sampler.temperature > 0
            and not lazy_target_distributions
            and not _steer_active
            and (
                _batch_target_arrays_enabled() or _batch_target_distributions_enabled()
            )
        ):
            target_distribution_rows = min(
                int(verify_logits.shape[1]),
                target_distribution_rows_needed,
            )
            target_distribution_logits = verify_logits[:, :target_distribution_rows, :]
            started_distribution = time.perf_counter()
            if _batch_target_arrays_enabled():
                target_distribution_batch = _batched_distributions_from_mlx_logits(
                    target_distribution_logits,
                    sampler,
                )
            else:
                target_distributions = _distributions_from_mlx_logits(
                    target_distribution_logits,
                    sampler,
                )
            elapsed_target_distribution_eval = (
                time.perf_counter() - started_distribution
            )
            target_distribution_materialized_rows += int(target_distribution_rows)
            target_distribution_materialized_windows += 1
            verify_eval_timings = {
                "verify_logits_eval_time_s": elapsed_target_distribution_eval,
                "verify_hidden_eval_time_s": 0.0,
                "verify_joint_eval_time_s": 0.0,
            }
            verify_target_distribution_time += elapsed_target_distribution_eval
            target_distribution_precomputed = True
            event["defer_verify_hidden_eval"] = {
                "mode": "target_distribution_first",
                "verify_hidden_mode": verify_hidden_mode,
                "batch_target_arrays": bool(_batch_target_arrays_enabled()),
                "batch_target_distributions": bool(
                    _batch_target_distributions_enabled()
                ),
                "rows": int(target_distribution_rows),
                "time_s": float(elapsed_target_distribution_eval),
            }
        elif captures is not None:
            verify_eval_timings = _eval_verify_outputs(
                verify_logits, verify_hidden, captures
            )
        else:
            verify_eval_timings = _eval_verify_outputs(verify_logits, verify_hidden)
        elapsed_verify_eval = time.perf_counter() - started_eval
        eval_attributed = sum(float(value) for value in verify_eval_timings.values())
        elapsed_verify_eval_unattributed = max(
            0.0, elapsed_verify_eval - eval_attributed
        )
        verify_logits_eval_time += float(
            verify_eval_timings["verify_logits_eval_time_s"]
        )
        verify_hidden_eval_time += float(
            verify_eval_timings["verify_hidden_eval_time_s"]
        )
        verify_joint_eval_time += float(verify_eval_timings["verify_joint_eval_time_s"])
        verify_eval_unattributed_time += elapsed_verify_eval_unattributed
        verify_eval_time += elapsed_verify_eval
        _add_timing(event, "verify_eval", elapsed_verify_eval)
        _add_timing(
            event,
            "verify_eval_logits",
            float(verify_eval_timings["verify_logits_eval_time_s"]),
        )
        _add_timing(
            event,
            "verify_eval_hidden",
            float(verify_eval_timings["verify_hidden_eval_time_s"]),
        )
        _add_timing(
            event,
            "verify_eval_joint",
            float(verify_eval_timings["verify_joint_eval_time_s"]),
        )
        _add_timing(
            event, "verify_target_distribution", elapsed_target_distribution_eval
        )
        _add_timing(event, "verify_eval_unattributed", elapsed_verify_eval_unattributed)
        elapsed_verify = elapsed_verify_forward + elapsed_verify_eval
        verify_time += elapsed_verify
        target_time += elapsed_verify
        verify_calls += 1
        if trace.enabled:
            trace_accounting_started = time.perf_counter()
            trace_verify_output_nbytes += (
                _tree_nbytes(verify_logits)
                + _tree_nbytes(verify_hidden)
                + _tree_nbytes(captures)
            )
            trace_accounting_time_s += time.perf_counter() - trace_accounting_started
        if graphbank is not None:
            event["graphbank"] = graphbank.to_dict()

        accepted_count = 0
        rejection_correction: int | None = None
        started_accept = time.perf_counter()
        if (
            sampler.temperature > 0
            and not target_distribution_precomputed
            and not lazy_target_distributions
            and not _steer_active
        ):
            target_distribution_rows = min(
                int(verify_logits.shape[1]),
                target_distribution_rows_needed,
            )
            target_distribution_logits = verify_logits[:, :target_distribution_rows, :]
            if _batch_target_arrays_enabled():
                target_distribution_batch = _batched_distributions_from_mlx_logits(
                    target_distribution_logits,
                    sampler,
                )
            elif _batch_target_distributions_enabled():
                target_distributions = _distributions_from_mlx_logits(
                    target_distribution_logits,
                    sampler,
                )
            if target_distribution_batch is not None or target_distributions is not None:
                target_distribution_materialized_rows += int(target_distribution_rows)
                target_distribution_materialized_windows += 1
                event["target_distribution_materialized"] = {
                    "mode": "accept_path",
                    "rows": int(target_distribution_rows),
                    "batch_target_arrays": bool(_batch_target_arrays_enabled()),
                    "batch_target_distributions": bool(
                        _batch_target_distributions_enabled()
                    ),
                }
        lazy_target_distribution_rows = 0
        lazy_target_distribution_time = 0.0
        lazy_target_distribution_window_counted = False
        if _penalties_active:
            # Force the lazy per-row verify path: the batched/prefix precomputes
            # build one un-penalized distribution per block and cannot carry the
            # per-row counts.
            target_prefix_tokens = None
            target_distribution_batch = None
        elif _steer_active:
            # Steering active (Loop Guard armed and/or Thinking Guard forcing):
            # null only the batch so p/q rows rebuild per position with the
            # merged overlay. target_prefix_tokens stays — the target_prefix
            # pre-sample above already carried the overlay (and its lane has
            # no draft distributions to fall back on).
            target_distribution_batch = None
        # Grammar clamp (#186 phase 3): drafts are proposed unmasked, so the
        # committed window must stop at the grammar's legal prefix. One
        # stateless validate call per cycle; the matcher itself only advances
        # through committed tokens at the top-of-cycle sync.
        constraint_legal_prefix = (
            constraint.validate_prefix(list(draft_tokens))
            if constraint is not None
            else None
        )
        for depth_index, draft_token in enumerate(draft_tokens):
            target_logits_for_draft = verify_logits[:, depth_index, :]
            if _steer_active:
                _row_guard_overlay = _steer_overlay(
                    [*tokens, *draft_tokens[:depth_index]]
                )
            else:
                _row_guard_overlay = None
            if _penalties_active:
                # Per-position (vLLM-exact) prefix counts: committed completion
                # (incl. this step's primary, already in `tokens`) + the in-block
                # draft tokens *before* this position. Rebuilt per position so it
                # cannot drift; an incremental counter is a perf follow-up.
                _working_counts: Counter[int] = Counter(tokens)
                _working_counts.update(draft_tokens[:depth_index])
            target_p_for_cache = None
            if sampler.temperature <= 0:
                _greedy_row = target_logits_for_draft[0]
                if _penalties_active or _row_guard_overlay:
                    _greedy_row = apply_penalties_mlx(
                        _greedy_row,
                        _working_counts if _penalties_active else None,
                        sampler.presence_penalty,
                        sampler.frequency_penalty,
                        penalty_overlay=_row_guard_overlay,
                    )
                target_token = int(mx.argmax(_greedy_row, axis=-1).item())
                accepted_now = draft_token == target_token
                accept_prob = 1.0 if accepted_now else 0.0
                correction = target_token
            elif target_prefix_tokens is not None:
                target_token = int(target_prefix_tokens[depth_index])
                accepted_now = int(draft_token) == target_token
                accept_prob = 1.0 if accepted_now else 0.0
                correction = target_token
            elif target_distribution_batch is not None:
                draft_q = draft_probs[depth_index]
                if draft_q is None:
                    raise RuntimeError("non-greedy MTP requires draft distributions")
                p = target_distribution_batch.probability(depth_index, draft_token)
                q = (
                    draft_q.probability(draft_token)
                    if isinstance(draft_q, SparseDistribution)
                    else float(draft_q[draft_token])
                )
                accept_prob = (
                    1.0 if q <= 0 and p > 0 else (0.0 if q <= 0 else min(1.0, p / q))
                )
                accepted_now = float(rng.random()) <= accept_prob
                target_p_for_cache = (
                    target_distribution_batch.to_distribution(depth_index)
                    if online_correction_cache
                    and depth_index + 1 >= online_correction_cache_min_depth
                    else None
                )
                correction = (
                    draft_token
                    if accepted_now
                    else sample_from_distribution(
                        residual_distribution(
                            target_p_for_cache
                            if target_p_for_cache is not None
                            else target_distribution_batch.to_distribution(depth_index),
                            draft_q,
                        ),
                        rng,
                    )
                )
            else:
                target_p = (
                    target_distributions[depth_index]
                    if target_distributions is not None
                    and not _penalties_active
                    and not _steer_active
                    else None
                )
                if target_p is None:
                    started_target_distribution = time.perf_counter()
                    target_p = _distribution_from_mlx_logits(
                        target_logits_for_draft[0],
                        sampler,
                        token_counts=_working_counts if _penalties_active else None,
                        penalty_overlay=_row_guard_overlay,
                    )
                    elapsed_target_distribution = (
                        time.perf_counter() - started_target_distribution
                    )
                    lazy_target_distribution_time += elapsed_target_distribution
                    lazy_target_distribution_rows += 1
                    if not lazy_target_distribution_window_counted:
                        target_distribution_materialized_windows += 1
                        lazy_target_distribution_window_counted = True
                    target_distribution_materialized_rows += 1
                    verify_target_distribution_time += elapsed_target_distribution
                    verify_logits_eval_time += elapsed_target_distribution
                    verify_eval_time += elapsed_target_distribution
                    verify_time += elapsed_target_distribution
                    target_time += elapsed_target_distribution
                draft_q = draft_probs[depth_index]
                if draft_q is None:
                    raise RuntimeError("non-greedy MTP requires draft distributions")
                accept_prob = compute_acceptance_probability(
                    target_p, draft_q, draft_token
                )
                accepted_now = float(rng.random()) <= accept_prob
                target_p_for_cache = target_p
                correction = (
                    draft_token
                    if accepted_now
                    else sample_from_distribution(
                        residual_distribution(target_p, draft_q), rng
                    )
                )

            if (
                constraint_legal_prefix is not None
                and accepted_now
                and depth_index >= constraint_legal_prefix
            ):
                # The model accepted a draft the grammar forbids here; reject
                # it and let the next cycle's masked primary resample the
                # position from the constrained distribution. Under pure
                # temperature sampling the committed law is exactly the
                # masked target law (Leviathan-Chen telescopes through the
                # drop-and-resample). Under top-k/top-p the two coincide
                # except in sub-top-k tail mass: draft-path positions commit
                # from restrict-then-renormalize of the SHAPED unmasked law,
                # masked-primary positions from shaping of the MASKED row.
                # Every committed token is grammar-legal either way; a
                # verify-row-masked variant would close the tail gap.
                accepted_now = False
                accept_prob = 0.0
                event["drafts"][depth_index]["constraint_clamped"] = True

            event["drafts"][depth_index]["accepted"] = accepted_now
            event["drafts"][depth_index]["accept_probability"] = float(accept_prob)
            event["drafts"][depth_index]["correction"] = int(correction)
            accept_probability_sum_by_depth[depth_index] += float(accept_prob)

            if accepted_now:
                accepted += 1
                accepted_count += 1
                accepted_by_depth[depth_index] += 1
                if _is_stop(draft_token, stop_token_ids):
                    break
                continue

            rejected += 1
            event["rejected_at_depth"] = depth_index + 1
            if (
                online_correction_cache
                and target_prefix_tokens is None
                and depth_index + 1 >= online_correction_cache_min_depth
                and depth_index < len(draft_cache_keys)
            ):
                cached_target = int(
                    correction
                    if sampler.temperature <= 0
                    else _distribution_argmax(
                        target_p_for_cache
                        if target_p_for_cache is not None
                        else target_distribution_batch.to_distribution(depth_index)
                    )
                )
                correction_cache[draft_cache_keys[depth_index]] = cached_target
                prompt_seeded_cache_keys.discard(draft_cache_keys[depth_index])
                correction_cache_stores += 1
                event["drafts"][depth_index]["online_correction_cache"][
                    "stored_token"
                ] = cached_target
            if (
                sampler.temperature > 0
                or a3b_target_prefix_route is not None
            ) and (
                constraint is None
                or constraint.validate_prefix(
                    [*draft_tokens[:depth_index], int(correction)]
                )
                == depth_index + 1
            ):
                # A grammar-illegal residual correction is dropped, not
                # committed; the masked primary resamples the position.
                # Greedy normally defers the correction to the next cycle's
                # argmax over the retained rejection row, but the compiled
                # K1 route's fixed cycle geometry commits + repair-forwards
                # the correction in-cycle, so it must be recorded at any
                # temperature -- under greedy `correction` IS the
                # pre-sampled argmax target id (the AR token).
                rejection_correction = int(correction)
            break
        elapsed_accept = max(
            0.0,
            time.perf_counter() - started_accept - lazy_target_distribution_time,
        )
        accept_time += elapsed_accept
        _add_timing(event, "accept", elapsed_accept)
        if lazy_target_distribution_rows:
            event["target_distribution_materialized"] = {
                "mode": "lazy_accept_path",
                "rows": int(lazy_target_distribution_rows),
                "time_s": float(lazy_target_distribution_time),
                "batch_target_arrays": False,
                "batch_target_distributions": False,
            }
            _add_timing(
                event,
                "verify_target_distribution_lazy_accept",
                lazy_target_distribution_time,
            )

        event["accepted_depths"] = accepted_count
        if adaptive_policy is not None:
            _policy_now = time.perf_counter()
            _policy_kwargs: dict[str, float] = {}
            if getattr(adaptive_policy, "accepts_cycle_ms", False):
                _policy_kwargs["cycle_ms"] = (
                    _policy_now - _policy_cycle_started
                ) * 1000.0
            _policy_cycle_started = _policy_now
            event["policy"] = adaptive_policy.observe(
                attempted_depth=max(1, len(draft_tokens)),
                accepted_depths=accepted_count,
                **_policy_kwargs,
            )

        if online_hidden_enabled and draft_hidden_for_update:
            started_online = time.perf_counter()
            update_events = []
            for feed_depth, predicted_hidden in enumerate(
                draft_hidden_for_update, start=1
            ):
                if feed_depth > online_hidden_max_feed_depth:
                    continue
                if feed_depth > int(verify_hidden.shape[1]):
                    continue
                if accepted_count < feed_depth - 1:
                    continue
                online_key = draft_hidden_update_keys[feed_depth - 1]
                target_hidden = verify_hidden[:, feed_depth - 1 : feed_depth, :].astype(
                    mx.float32
                )
                residual = target_hidden - predicted_hidden.astype(mx.float32)
                previous = online_hidden_deltas.get(online_key)
                if previous is None:
                    updated = residual
                else:
                    updated = (
                        float(online_hidden_corrector_decay) * previous
                        + (1.0 - float(online_hidden_corrector_decay)) * residual
                    )
                _eval(updated)
                online_hidden_deltas[online_key] = updated
                online_hidden_update_counts[online_key] = (
                    online_hidden_update_counts.get(online_key, 0) + 1
                )
                update_events.append(
                    {
                        "feed_depth": feed_depth,
                        "key": online_hidden_corrector_key,
                        "source_token": (
                            online_key[1]
                            if online_hidden_corrector_key == "token"
                            and isinstance(online_key, tuple)
                            else None
                        ),
                        "updates": online_hidden_update_counts[online_key],
                        "accepted_prefix_required": feed_depth - 1,
                    }
                )
            elapsed_online = time.perf_counter() - started_online
            online_hidden_corrector_time += elapsed_online
            if update_events:
                event["online_hidden_corrector_updates"] = update_events
                _add_timing(event, "online_hidden_corrector_update", elapsed_online)

        if accepted_count == len(draft_tokens):
            committed = [primary] + draft_tokens
            tokens.extend(draft_tokens)
            if _mtp_history_uses_committed_cache(mtp_history_policy):
                assert mtp_cache is not None and cycle_mtp_offset is not None
                _rollback_mtp_cache(mtp_cache, cycle_mtp_offset + 1)
                draft_time += append_mtp_history(
                    mtp_cache,
                    verify_hidden[:, : max(0, len(committed) - 1), :],
                    committed[1:],
                )
            if lazy_bonus_verify:
                started_bonus_commit_forward = time.perf_counter()
                with attention_phase("decode_verify"):
                    bonus_commit_logits, bonus_commit_hidden = rt.forward_ar(
                        mx.array([[int(draft_tokens[-1])]]),
                        cache=cache,
                        return_hidden=True,
                        hidden_variant=base_hidden_variant,
                    )
                elapsed_bonus_commit_forward = (
                    time.perf_counter() - started_bonus_commit_forward
                )
                started_bonus_commit_eval = time.perf_counter()
                _eval(bonus_commit_logits, bonus_commit_hidden)
                elapsed_bonus_commit_eval = (
                    time.perf_counter() - started_bonus_commit_eval
                )
                elapsed_bonus_commit = (
                    elapsed_bonus_commit_forward + elapsed_bonus_commit_eval
                )
                lazy_bonus_commit_time += elapsed_bonus_commit
                verify_forward_time += elapsed_bonus_commit_forward
                verify_eval_time += elapsed_bonus_commit_eval
                verify_joint_eval_time += elapsed_bonus_commit_eval
                verify_time += elapsed_bonus_commit
                target_time += elapsed_bonus_commit
                commit_time += elapsed_bonus_commit
                event["lazy_bonus_verify"]["bonus_commit_forward_s"] = float(
                    elapsed_bonus_commit_forward
                )
                event["lazy_bonus_verify"]["bonus_commit_eval_s"] = float(
                    elapsed_bonus_commit_eval
                )
                _add_timing(
                    event,
                    "lazy_bonus_commit_forward",
                    elapsed_bonus_commit_forward,
                )
                _add_timing(
                    event,
                    "lazy_bonus_commit_eval",
                    elapsed_bonus_commit_eval,
                )
                logits, hidden = own_live_logits_hidden(
                    bonus_commit_logits[:, -1, :],
                    bonus_commit_hidden[:, -1:, :],
                )
            else:
                logits, hidden = own_live_logits_hidden(
                    verify_logits[:, len(draft_tokens), :],
                    verify_hidden[:, -1:, :],
                )
            if any(_is_stop(token, stop_token_ids) for token in draft_tokens):
                tokens = _truncate_after_first_stop(tokens, stop_token_ids)
                detach_capture_committed_state(len(tokens))
                maybe_detach_dirty_state(len(tokens))
                maybe_eval_state_roots(event, len(tokens))
                emit_new_tokens()
                append_event(event)
                break
            detach_capture_committed_state(len(tokens))
            maybe_detach_dirty_state(len(tokens))
            maybe_rebase_decode_state(len(tokens))
            emit_new_tokens()
            if len(tokens) < max_tokens:
                if omit_speculative_bonus:
                    event["bonus_token_omitted"] = True
                    maybe_eval_state_roots(event, len(tokens))
                    append_event(event)
                    emit_trace()
                    continue
                started_bonus = time.perf_counter()
                bonus_target_distribution_time = 0.0
                if (
                    target_prefix_tokens is not None
                    and len(target_prefix_tokens) > len(draft_tokens)
                ):
                    bonus = int(target_prefix_tokens[len(draft_tokens)])
                elif target_distribution_batch is not None and not lazy_bonus_verify:
                    bonus = target_distribution_batch.sample(len(draft_tokens), rng)
                elif (
                    target_distributions is not None
                    and not lazy_bonus_verify
                    and not _penalties_active
                    and not _steer_active
                    and len(target_distributions) > len(draft_tokens)
                ):
                    bonus = sample_from_distribution(
                        target_distributions[len(draft_tokens)],
                        rng,
                    )
                else:
                    started_bonus_distribution = time.perf_counter()
                    # all-accept bonus: tokens already includes the committed block,
                    # so Counter(tokens) is the correct prefix for this next token.
                    bonus, _ = _sample_from_logits(
                        logits[0],
                        sampler,
                        rng,
                        token_counts=Counter(tokens) if _penalties_active else None,
                        penalty_overlay=(
                            _steer_overlay(tokens)
                            if _steer_active
                            else None
                        ),
                    )
                    if sampler.temperature > 0:
                        bonus_target_distribution_time = (
                            time.perf_counter() - started_bonus_distribution
                        )
                        if not lazy_target_distribution_window_counted:
                            target_distribution_materialized_windows += 1
                            lazy_target_distribution_window_counted = True
                        target_distribution_materialized_rows += 1
                        verify_target_distribution_time += bonus_target_distribution_time
                        verify_logits_eval_time += bonus_target_distribution_time
                        verify_eval_time += bonus_target_distribution_time
                        verify_time += bonus_target_distribution_time
                        target_time += bonus_target_distribution_time
                        materialized = event.get("target_distribution_materialized")
                        if isinstance(materialized, dict) and str(
                            materialized.get("mode", "")
                        ).startswith("lazy"):
                            materialized["mode"] = "lazy_accept_bonus_path"
                            materialized["rows"] = int(materialized.get("rows") or 0) + 1
                            materialized["time_s"] = float(
                                materialized.get("time_s") or 0.0
                            ) + float(bonus_target_distribution_time)
                        else:
                            event["target_distribution_materialized"] = {
                                "mode": "lazy_bonus_path",
                                "rows": 1,
                                "time_s": float(bonus_target_distribution_time),
                                "batch_target_arrays": False,
                                "batch_target_distributions": False,
                            }
                        _add_timing(
                            event,
                            "verify_target_distribution_lazy_bonus",
                            bonus_target_distribution_time,
                        )
                elapsed_bonus = max(
                    0.0,
                    time.perf_counter() - started_bonus - bonus_target_distribution_time,
                )
                bonus_time += elapsed_bonus
                _add_timing(event, "bonus_sample", elapsed_bonus)
                if constraint is not None and (
                    constraint.validate_prefix([*draft_tokens, int(bonus)])
                    != len(draft_tokens) + 1
                ):
                    # Grammar-illegal bonus: skip it (same control path as
                    # omit_speculative_bonus). `logits` already holds the
                    # bonus-position row, so the next cycle's masked primary
                    # resamples this exact position from the constrained
                    # distribution.
                    event["bonus_token_constraint_skipped"] = True
                    maybe_eval_state_roots(event, len(tokens))
                    append_event(event)
                    emit_trace()
                    continue
                tokens.append(bonus)
                pending_primary = bonus
                bonus_tokens += 1
                event["bonus_token"] = int(bonus)
                emit_new_tokens()
                if _is_stop(bonus, stop_token_ids):
                    maybe_eval_state_roots(event, len(tokens))
                    append_event(event)
                    emit_trace()
                    break
            maybe_eval_state_roots(event, len(tokens))
            append_event(event)
            emit_trace()
            continue

        committed = [primary] + draft_tokens[:accepted_count]
        if a3b_target_prefix_route is not None:
            committed.append(rejection_correction)
            correction_tokens += 1
            tokens.extend(committed[1:])
            # Deferred-correction fold: no repair_m1 forward.  The correction
            # is emitted as the pending primary; the next verify runs the M2
            # graph FROM the stashed post-primary state and computes the
            # correction's row itself.  Byte-neutral vs repair: M2 row-0
            # arithmetic is install-enforced bit-identical to the fused M1
            # route.  Drafting for the folded cycle consumes the rejection
            # boundary row (the primary's verify row), the same hidden the
            # committed-history append pairs with the correction.
            pending_primary = int(rejection_correction)
            if a3b_k2:
                # Rebase to the state matching the accepted prefix: 0 accepted
                # -> post-row-0, 1 accepted -> post-row-1, 2 accepted -> the
                # live post-row-2 state already written by verify_m3 (no
                # rebase).  The next verify_m3 starts from here.
                if accepted_count >= 2:
                    a3b_rebase_state = None
                elif accepted_count == 1:
                    a3b_rebase_state = a3b_m3_rebase1_state
                else:
                    a3b_rebase_state = a3b_m3_rebase0_state
            else:
                a3b_rebase_state = a3b_primary_state
            deferred_correction_repairs += 1
            event["capture_repair"] = "route_pending_correction"
            event["pending_primary"] = int(rejection_correction)
            if _mtp_history_uses_committed_cache(mtp_history_policy):
                _rollback_mtp_cache(mtp_cache, cycle_mtp_offset + 1)
                draft_time += append_mtp_history(
                    mtp_cache,
                    verify_hidden[:, 0:1, :],
                    [rejection_correction],
                )
            cache_committed_token_count = max(0, len(tokens) - 1)
            maybe_detach_dirty_state(cache_committed_token_count)
            logits, hidden = own_live_logits_hidden(
                verify_logits[:, 0:1, :].reshape(1, -1),
                verify_hidden[:, 0:1, :],
            )
            maybe_rebase_decode_state(cache_committed_token_count)
            maybe_eval_state_roots(event, cache_committed_token_count)
            append_event(event)

            if any(_is_stop(token, stop_token_ids) for token in committed):
                stop_index = next(
                    i
                    for i, token in enumerate(tokens)
                    if _is_stop(token, stop_token_ids)
                )
                tokens = tokens[: stop_index + 1]
                emit_new_tokens()
                emit_trace()
                break
            emit_new_tokens()
            emit_trace()
            continue

        if rejection_correction is not None:
            committed.append(rejection_correction)
            correction_tokens += 1
        tokens.extend(committed[1:])

        committed_prefix_len = 1 + accepted_count
        committed_from_capture = False
        committed_from_trim = False
        cache_committed_token_count = len(tokens)
        if rejection_correction is not None:
            cache_committed_token_count = max(0, cache_committed_token_count - 1)
        capture_commit_detach_components = capture_commit_detach_due(
            cache_committed_token_count
        )
        if (
            verify_strategy in {"capture_commit", "graphbank_capture_commit"}
            and captures is not None
        ):
            from .gdn_capture import commit_captured_prefix

            started_commit = time.perf_counter()
            commit_detach_stats = {"arrays": 0, "bytes": 0}
            committed_from_capture = commit_captured_prefix(
                cache,
                captures,
                keep_tokens=committed_prefix_len,
                verified_tokens=verified_token_count,
                detach_components=capture_commit_detach_components,
                detach_mode=capture_commit_detach_mode,
                detach_stats=commit_detach_stats,
            )
            elapsed_commit = time.perf_counter() - started_commit
            capture_commit_time += elapsed_commit
            if int(commit_detach_stats.get("arrays", 0)) > 0:
                capture_commit_detach_events += 1
                capture_commit_detach_time_s += elapsed_commit
                capture_commit_detach_arrays += int(commit_detach_stats["arrays"])
                capture_commit_detach_bytes += int(commit_detach_stats["bytes"])
            _add_timing(event, "capture_commit", elapsed_commit)

        if (
            not committed_from_capture
            and verify_strategy in {"trim_commit", "target_prefix"}
            and before_verify is not None
        ):
            started_trim_commit = time.perf_counter()
            committed_from_trim = trim_verified_window_to_prefix(
                cache,
                before_verify,
                verified_tokens=verified_token_count,
                keep_tokens=committed_prefix_len,
            )
            elapsed_trim_commit = time.perf_counter() - started_trim_commit
            if committed_from_trim:
                commit_time += elapsed_trim_commit
                _add_timing(event, "trim_commit", elapsed_trim_commit)
        if (
            not committed_from_capture
            and not committed_from_trim
            and before_verify is None
        ):
            # The verify snapshot was skipped (MTPLX_SKIP_VERIFY_SNAPSHOT=1,
            # the product-profile default) and no capture/trim lane committed.
            # All-trimmable caches can still repair exactly by trimming the
            # uncommitted verify tail — without this, the first rejection on
            # such a lane raised and killed the request.
            started_trim_commit = time.perf_counter()
            committed_from_trim = trim_verified_window_without_snapshot(
                cache,
                verified_tokens=len(verify_input),
                keep_tokens=committed_prefix_len,
            )
            elapsed_trim_commit = time.perf_counter() - started_trim_commit
            if committed_from_trim:
                commit_time += elapsed_trim_commit
                _add_timing(event, "trim_commit", elapsed_trim_commit)
            else:
                _add_timing(event, "trim_commit_failed", elapsed_trim_commit)

        if committed_from_capture:
            event["capture_repair"] = "captured_prefix_commit"
            if rejection_correction is None:
                repair_logits, repair_hidden = own_live_logits_hidden(
                    verify_logits[
                        :, committed_prefix_len - 1 : committed_prefix_len, :
                    ],
                    verify_hidden[
                        :, committed_prefix_len - 1 : committed_prefix_len, :
                    ],
                )
            else:
                repair_logits, repair_hidden = own_live_logits_hidden(
                    verify_logits[
                        :, committed_prefix_len - 1 : committed_prefix_len, :
                    ],
                    verify_hidden[
                        :, committed_prefix_len - 1 : committed_prefix_len, :
                    ],
                )
                pending_primary = int(rejection_correction)
                deferred_correction_repairs += 1
                event["capture_repair"] = "captured_prefix_pending_correction"
                event["pending_primary"] = int(rejection_correction)
        elif committed_from_trim:
            if rejection_correction is None:
                repair_logits, repair_hidden = own_live_logits_hidden(
                    verify_logits[
                        :, committed_prefix_len - 1 : committed_prefix_len, :
                    ],
                    verify_hidden[
                        :, committed_prefix_len - 1 : committed_prefix_len, :
                    ],
                )
                event["capture_repair"] = "trimmed_prefix_commit"
            else:
                # Deferred correction repair (the 2.3.0 capture-commit fix,
                # ported to the trim lane): the correction is emitted now and
                # becomes the pending primary, whose KV is computed by
                # whichever forward runs next -- no dedicated one-row
                # correction forward.  Drafting needs the hidden of the token
                # BEFORE the pending primary, which is exactly the retained
                # verify row at the rejection boundary; the trim commit
                # already restored the cache to the committed prefix, the
                # same state the old correction forward ran on.
                repair_logits, repair_hidden = own_live_logits_hidden(
                    verify_logits[
                        :, committed_prefix_len - 1 : committed_prefix_len, :
                    ],
                    verify_hidden[
                        :, committed_prefix_len - 1 : committed_prefix_len, :
                    ],
                )
                pending_primary = int(rejection_correction)
                deferred_correction_repairs += 1
                event["capture_repair"] = "trimmed_prefix_pending_correction"
                event["pending_primary"] = int(rejection_correction)
        else:
            if before_verify is None:
                raise RuntimeError(
                    "capture commit failed after MTPLX_SKIP_VERIFY_SNAPSHOT=1"
                )
            event["capture_repair"] = (
                "standard_reforward" if verify_strategy == "capture_commit" else None
            )
            started_rollback = time.perf_counter()
            rollback_after_verify(
                cache, before_verify, verified_tokens=verified_token_count
            )
            elapsed_rollback = time.perf_counter() - started_rollback
            rollback_time += elapsed_rollback
            _add_timing(event, "rollback", elapsed_rollback)
            started = time.perf_counter()
            with (
                attention_phase("decode_verify"),
                model_forward_kind("repair"),
            ):
                if generic_compiled_target_prefix and compiled_verify_bank is not None:
                    repair_logits, repair_hidden, _repair_captures = (
                        compiled_verify_bank.forward_ar_capture(
                            mx.array([committed]),
                            cache=cache,
                            return_hidden=True,
                            hidden_variant=base_hidden_variant,
                        )
                    )
                else:
                    repair_logits, repair_hidden = rt.forward_ar(
                        mx.array([committed]),
                        cache=cache,
                        return_hidden=True,
                        hidden_variant=base_hidden_variant,
                    )
            _eval(repair_logits, repair_hidden)
            elapsed_repair = time.perf_counter() - started
            target_time += elapsed_repair
            repair_time += elapsed_repair
            _add_timing(event, "repair_forward", elapsed_repair)
        if _mtp_history_uses_committed_cache(mtp_history_policy):
            assert mtp_cache is not None and cycle_mtp_offset is not None
            _rollback_mtp_cache(mtp_cache, cycle_mtp_offset + 1)
            history_tokens = committed[1:]
            if committed_from_capture or committed_from_trim:
                history_hidden = verify_hidden[:, : max(0, len(committed) - 1), :]
            else:
                history_hidden = repair_hidden[:, : max(0, len(committed) - 1), :]
            if committed_from_capture and rejection_correction is not None:
                history_tokens = history_tokens[:-1]
                history_hidden = verify_hidden[:, : max(0, committed_prefix_len - 1), :]
            draft_time += append_mtp_history(
                mtp_cache,
                history_hidden,
                history_tokens,
            )
        maybe_detach_dirty_state(cache_committed_token_count)
        logits, hidden = own_live_logits_hidden(
            repair_logits[:, -1, :],
            repair_hidden[:, -1:, :],
        )
        maybe_rebase_decode_state(cache_committed_token_count)
        maybe_eval_state_roots(event, cache_committed_token_count)
        append_event(event)

        if any(_is_stop(token, stop_token_ids) for token in committed):
            stop_index = next(
                i for i, token in enumerate(tokens) if _is_stop(token, stop_token_ids)
            )
            tokens = tokens[: stop_index + 1]
            emit_new_tokens()
            emit_trace()
            break
        emit_new_tokens()
        emit_trace()

    if first_round_snapshot is None and int(verify_calls) >= 1:
        # Single-cycle generation: the loop never reached iteration 2, so the
        # cumulative timers ARE round 1's totals. Product telemetry stays
        # complete; single_cycle marks the provenance.
        first_round_snapshot = {
            "wall_s": time.perf_counter() - decode_loop_entered_s,
            "draft_time_s": float(draft_time),
            "verify_time_s": float(verify_time),
            "verify_forward_time_s": float(verify_forward_time),
            "accept_time_s": float(accept_time),
            "verify_calls": int(verify_calls),
            "committed_tokens": len(tokens),
            "single_cycle": True,
        }
    final_state: GenerationFinalState | None = None
    if (
        capture_final_state
        and pending_primary is not None
        and tokens
        and repetition_result is None
    ):
        pending_token = int(pending_primary)
        if (
            _mtp_history_uses_committed_cache(mtp_history_policy)
            and mtp_history_cache is not None
            and hidden is not None
        ):
            commit_started = time.perf_counter()
            draft_time += append_mtp_history(
                mtp_history_cache,
                hidden,
                [pending_token],
            )
            commit_time += time.perf_counter() - commit_started
        commit_started = time.perf_counter()
        with attention_phase("decode_verify"):
            commit_logits, commit_hidden = rt.forward_ar(
                mx.array([[pending_token]]),
                cache=cache,
                return_hidden=True,
                hidden_variant=base_hidden_variant,
            )
        _eval(commit_logits, commit_hidden)
        elapsed_commit_forward = time.perf_counter() - commit_started
        target_time += elapsed_commit_forward
        commit_time += elapsed_commit_forward
        logits, hidden = own_live_logits_hidden(
            commit_logits[:, -1, :],
            commit_hidden[:, -1:, :],
        )
        pending_primary = None
        detach_capture_committed_state(len(tokens))
        maybe_detach_dirty_state(len(tokens))
        maybe_rebase_decode_state(len(tokens))
        maybe_eval_state_roots({"final_pending_commit": True}, len(tokens))

    emit_trace(force=True, final=True)
    elapsed = time.perf_counter() - started_all
    compiled_verify_report: dict[str, Any] | None = None
    if a3b_target_prefix_route is not None:
        compiled_verify_report = a3b_target_prefix_route.final_report(
            # Corrections are deferred into rebased M2 verifies; repair_m1 is
            # never dispatched, so m1_calls reports the truth: zero.
            verify_calls=verify_calls,
            repair_calls=correction_tokens - deferred_correction_repairs,
        )
        a3b_target_prefix_route.demote()
    elif compiled_verify_bank is not None:
        compiled_verify_report = compiled_verify_bank.to_dict()
        if _env_truthy("MTPLX_COMPILED_VERIFY_STATS"):
            try:
                print(
                    "[mtplx] compiled-verify stats "
                    + json.dumps(compiled_verify_report),
                    file=sys.stderr,
                    flush=True,
                )
            except Exception:
                pass
        # Mandatory before the final-state capture: postcommit and every
        # other downstream cache consumer must never see promoted
        # tensor-offset adapters.
        compiled_verify_bank.demote(cache)
    if constraint is not None:
        # Final sync so `completed` reflects every committed token (the loop
        # may exit between the per-cycle sync and the last commit).
        constraint.advance_many(tokens[constraint_synced_tokens:])
        constraint_synced_tokens = len(tokens)
    finish_reason = (
        "stop"
        if repetition_result is not None
        or any(_is_stop(token, stop_token_ids) for token in tokens)
        # A grammar-terminal exit is a completed document, not truncation.
        or (constraint is not None and constraint.stopped)
        else "length"
    )
    if capture_final_state:
        final_state = GenerationFinalState(
            final_trunk_cache=cache,
            final_logits=logits,
            final_hidden=hidden,
            final_committed_mtp_cache=mtp_history_cache,
            generated_token_ids=tuple(int(token) for token in tokens),
            safe_to_commit=pending_primary is None and repetition_result is None,
            finish_reason=finish_reason,
            mtp_history_policy=mtp_history_policy,
            mtp_history_window_tokens=int(prompt_state.mtp_history_window_tokens),
            mtp_history_position_base=int(prompt_state.mtp_history_position_base),
        )
    reject_path_counts, repair_time_by_reject_depth = _reject_repair_breakdown(events)
    stats = GenerationStats(
        mode="mtpk",
        constraint_active=constraint is not None,
        constraint_completed=(
            constraint.completed if constraint is not None else None
        ),
        constraint_masked_steps=(
            constraint.masked_steps if constraint is not None else 0
        ),
        constraint_mask_time_s=(
            constraint.mask_time_s if constraint is not None else 0.0
        ),
        generated_tokens=len(tokens),
        elapsed_s=elapsed,
        **_generation_rate_fields(
            generated_tokens=len(tokens),
            elapsed_s=elapsed,
            prompt_eval_time_s=prompt_eval_time,
            cache_restore_time_s=prompt_state.cache_restore_time_s,
        ),
        accepted_drafts=accepted,
        rejected_drafts=rejected,
        drafted_tokens=drafted,
        verify_time_s=verify_time,
        verify_forward_time_s=verify_forward_time,
        verify_eval_time_s=verify_eval_time,
        verify_logits_eval_time_s=verify_logits_eval_time,
        verify_hidden_eval_time_s=verify_hidden_eval_time,
        verify_joint_eval_time_s=verify_joint_eval_time,
        verify_target_distribution_time_s=verify_target_distribution_time,
        target_distribution_materialized_rows=target_distribution_materialized_rows,
        target_distribution_materialized_windows=target_distribution_materialized_windows,
        target_distribution_share=(
            verify_target_distribution_time / verify_time if verify_time > 0 else 0.0
        ),
        lazy_bonus_verify_calls=lazy_bonus_verify_calls,
        lazy_bonus_commit_time_s=lazy_bonus_commit_time,
        verify_eval_unattributed_time_s=verify_eval_unattributed_time,
        verify_hidden_mode=verify_hidden_mode,
        draft_time_s=draft_time,
        target_forward_time_s=target_time,
        prompt_eval_time_s=prompt_eval_time,
        prompt_tps=(
            prompt_state.suffix_tokens / prompt_eval_time
            if prompt_eval_time > 0
            else 0.0
        ),
        prompt_target_prefill_time_s=prompt_target_prefill_time,
        prompt_mtp_history_time_s=prompt_state.prompt_mtp_history_time_s,
        cache_restore_time_s=prompt_state.cache_restore_time_s,
        prompt_target_prefill_tok_s=(
            prompt_state.suffix_tokens / prompt_target_prefill_time
            if prompt_target_prefill_time > 0
            else 0.0
        ),
        prompt_mtp_history_tok_s=(
            prompt_state.suffix_tokens / prompt_state.prompt_mtp_history_time_s
            if prompt_state.prompt_mtp_history_time_s > 0
            else 0.0
        ),
        mtp_history_policy=mtp_history_policy,
        mtp_history_window_tokens=int(prompt_state.mtp_history_window_tokens),
        mtp_history_position_base=int(prompt_state.mtp_history_position_base),
        cached_tokens=prompt_state.cached_tokens,
        new_prefill_tokens=prompt_state.suffix_tokens,
        session_cache_hit=prompt_state.cache_hit,
        cache_source=prompt_state.cache_source,
        ssd_cache_hit=prompt_state.ssd_cache_hit,
        ssd_cached_tokens=prompt_state.ssd_cached_tokens,
        ssd_restore_s=prompt_state.ssd_restore_s,
        ssd_suffix_tokens=prompt_state.suffix_tokens if prompt_state.ssd_cache_hit else 0,
        cache_miss_reason=prompt_state.cache_miss_reason,
        session_restore_mode=prompt_state.restore_mode,
        session_prompt_prefix_bank_commit=prompt_prefix_bank_commit,
        session_prefill_store=dict(
            getattr(prompt_state, "prefill_store_snapshot", None) or {}
        ),
        pre_first_token_setup_s=float(pre_first_token_setup_s),
        session_restore_served=dict(
            getattr(prompt_state, "restore_served", None) or {}
        ),
        prompt_state_total_time_s=float(prompt_state_total_time_s),
        prompt_state_unattributed_time_s=float(
            max(
                0.0,
                prompt_state_total_time_s
                - float(prompt_state.prompt_eval_time_s or 0.0)
                - float(prompt_state.cache_restore_time_s or 0.0)
                - float(
                    (getattr(prompt_state, "prefill_store_snapshot", None) or {}).get(
                        "elapsed_s"
                    )
                    or 0.0
                ),
            )
        ),
        first_primary_sample_time_s=float(first_primary_sample_time_s),
        first_round=dict(first_round_snapshot or {}),
        snapshot_time_s=snapshot_time,
        accept_time_s=accept_time,
        rollback_time_s=rollback_time,
        repair_time_s=repair_time,
        commit_time_s=commit_time,
        capture_commit_time_s=capture_commit_time,
        mtp_history_materialize_every=mtp_history_materialize_every,
        mtp_history_materialize_events=mtp_history_materialize_events,
        clear_cache_every=clear_cache_every,
        clear_cache_events=clear_cache_events,
        clear_cache_time_s=clear_cache_time_s,
        trunk_cache_materialize_every=trunk_cache_materialize_every,
        trunk_cache_materialize_events=trunk_cache_materialize_events,
        trunk_cache_materialize_time_s=trunk_cache_materialize_time_s,
        dirty_detach_components=dirty_detach_enabled_components,
        dirty_detach_mode=dirty_detach_mode,
        dirty_detach_gdn_every=dirty_detach_cadences["gdn"],
        dirty_detach_conv_every=dirty_detach_cadences["conv"],
        dirty_detach_attn_every=dirty_detach_cadences["attn"],
        dirty_detach_events=dirty_detach_events,
        dirty_detach_time_s=dirty_detach_time_s,
        dirty_detach_arrays=dirty_detach_arrays,
        dirty_detach_bytes=dirty_detach_bytes,
        live_output_detach_enabled=live_output_detach_enabled,
        live_output_detach_mode=live_output_detach_mode,
        live_output_detach_events=live_output_detach_events,
        live_output_detach_time_s=live_output_detach_time_s,
        live_output_detach_arrays=live_output_detach_arrays,
        live_output_detach_bytes=live_output_detach_bytes,
        state_rebase_every=state_rebase_every,
        state_rebase_events=state_rebase_events,
        state_rebase_time_s=state_rebase_time_s,
        state_root_eval_enabled=state_root_eval_enabled,
        state_root_eval_include_mtp=state_root_eval_include_mtp,
        state_root_eval_events=state_root_eval_events,
        state_root_eval_time_s=state_root_eval_time_s,
        state_root_eval_arrays=state_root_eval_arrays,
        capture_commit_detach_components=capture_commit_detach_enabled_components,
        capture_commit_detach_mode=capture_commit_detach_mode,
        capture_commit_detach_gdn_every=capture_commit_detach_cadences["gdn"],
        capture_commit_detach_conv_every=capture_commit_detach_cadences["conv"],
        capture_commit_detach_events=capture_commit_detach_events,
        capture_commit_detach_time_s=capture_commit_detach_time_s,
        capture_commit_detach_arrays=capture_commit_detach_arrays,
        capture_commit_detach_bytes=capture_commit_detach_bytes,
        trace_accounting_time_s=trace_accounting_time_s,
        decode_trace_path=str(trace.path) if trace.path is not None else None,
        decode_trace_run_id=trace.run_id if trace.enabled else None,
        bonus_time_s=bonus_time,
        online_hidden_corrector_time_s=online_hidden_corrector_time,
        peak_memory_bytes=mx.get_peak_memory(),
        speculative_depth=speculative_depth,
        requested_speculative_depth=requested_speculative_depth,
        long_context_mtp_depth_policy=long_context_depth_policy,
        accepted_by_depth=accepted_by_depth,
        drafted_by_depth=drafted_by_depth,
        accept_probability_sum_by_depth=accept_probability_sum_by_depth,
        mean_accept_probability_by_depth=_mean_accept_probability_by_depth(
            accept_probability_sum_by_depth,
            drafted_by_depth,
        ),
        bonus_tokens=bonus_tokens,
        correction_tokens=correction_tokens,
        verify_calls=verify_calls,
        context_copy_active=bool(ccopy_active),
        context_copy_probes=ccopy_probes,
        context_copy_rounds=ccopy_rounds,
        context_copy_drafted_tokens=ccopy_drafted,
        context_copy_accepted_blocks=ccopy_blocks_accepted,
        context_copy_accepted_tokens=ccopy_accepted,
        context_copy_suspensions=ccopy_suspensions,
        context_copy_suspended=len(tokens) < ccopy_suspend_until,
        context_copy_backoff_tokens=ccopy_backoff if ccopy_index is not None else 0,
        context_copy_disabled_reason=ccopy_disabled_reason,
        graphbank={
            **(graphbank.to_dict() if graphbank is not None else {}),
            **(
                {"compiled_verify": compiled_verify_report}
                if compiled_verify_report is not None
                else {}
            ),
        },
        reject_path_counts=reject_path_counts,
        repair_time_by_reject_depth_s=repair_time_by_reject_depth,
        deferred_correction_repairs=deferred_correction_repairs,
        online_correction_cache={
            "enabled": online_correction_cache,
            "prompt_enabled": prompt_correction_cache,
            "hits": correction_cache_hits,
            "stores": correction_cache_stores,
            "entries": len(correction_cache),
            "key_policy": online_correction_cache_key,
            "prompt_hits": prompt_correction_cache_hits,
            "prompt_stores": int(prompt_seed_stats.get("stores", 0)),
            "prompt_collisions": int(prompt_seed_stats.get("collisions", 0)),
            "prompt_skipped": int(prompt_seed_stats.get("skipped", 0)),
            "prompt_min_depth": prompt_correction_cache_min_depth,
        },
        adapter_ensemble_q={
            "enabled": adapter_ensemble_q,
            "epsilon": float(adapter_ensemble_epsilon),
            "min_depth": int(adapter_ensemble_min_depth),
            "calls": adapter_ensemble_calls,
            "changed": adapter_ensemble_changed,
            "base_selected": adapter_ensemble_base_selected,
            "adapter_selected": adapter_ensemble_adapter_selected,
            "shared_selected": adapter_ensemble_shared_selected,
            "fallbacks": adapter_ensemble_fallbacks,
        },
        mtp_topk_reranker={
            "enabled": mtp_topk_reranker is not None,
            "calls": topk_reranker_calls,
            "changed": topk_reranker_changed,
            "fallbacks": topk_reranker_fallbacks,
            "selected_rank_sum": topk_reranker_selected_rank_sum,
            "mean_selected_rank": (
                topk_reranker_selected_rank_sum / topk_reranker_calls
                if topk_reranker_calls
                else None
            ),
            **(mtp_topk_reranker.to_dict() if mtp_topk_reranker is not None else {}),
        },
        draft_core={
            "requested": draft_core,
            "device_d2_calls": device_d2_calls,
            "device_d2_fallbacks": device_d2_fallbacks,
            "device_d2_compile_time_s": device_d2_compile_time,
            "device_calls": device_core_calls,
            "device_fallbacks": device_core_fallbacks,
            "device_compile_time_s": device_core_compile_time,
        },
        owned_recurrent_state=owned_recurrent_state_stats(cache),
        owned_attn_kv=tail_owned_attention_kv_stats(cache),
        repetition_stop_triggered=repetition_result is not None,
        repetition_stop_reason=(
            "exact_repeated_token_suffix" if repetition_result is not None else None
        ),
        repetition_stop_block_tokens=(
            0 if repetition_result is None else repetition_result.block_tokens
        ),
        repetition_stop_repeats=(
            0 if repetition_result is None else repetition_result.repeats
        ),
        repetition_stop_trimmed_tokens=(
            0 if repetition_result is None else repetition_result.repeated_tokens
        ),
        repetition_stop_raw_tokens=(
            0
            if repetition_result is None
            else len(tokens) + repetition_result.repeated_tokens
        ),
        loop_guard=(_loop_guard.summary() if _loop_guard is not None else {}),
        thinking_guard=(
            _thinking_guard.summary() if _thinking_guard is not None else {}
        ),
        events=events,
    )
    _attach_runtime_diagnostics(stats, rt, counter_start)
    return GenerationOutput(
        tokens=tokens,
        text=_decode(rt.tokenizer, _strip_terminal_stop(tokens, stop_token_ids)),
        stats=stats,
        final_state=final_state,
        finish_reason=finish_reason,
    )


def generate_mtpa(
    rt: MTPLXRuntime,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    sampler: SamplerConfig,
    max_depth: int,
    seed: int = 0,
    stop_token_ids: set[int] | None = None,
    mtp_hidden_variant: str = "post_norm",
    mtp_cache_policy: str = "persistent",
    draft_sampler: SamplerConfig | None = None,
    min_depth: int = 1,
    start_depth: int = 1,
    increase_after: int = 4,
    decrease_after: int = 1,
) -> GenerationOutput:
    """Generate with a simple adaptive native-MTP depth policy."""
    if not rt.mtp_enabled:
        raise RuntimeError("generate_mtpa requires an MTP-enabled runtime")
    if max_depth < 1:
        raise ValueError("max_depth must be >= 1")
    if mtp_cache_policy not in {"persistent", "fresh"}:
        raise ValueError("mtp_cache_policy must be 'persistent' or 'fresh'")

    counter_start = _runtime_counter_snapshot(rt)
    rng = np.random.default_rng(seed)
    draft_sampler = _env_scaled_draft_sampler(sampler, draft_sampler)
    policy = AdaptiveDepthPolicy(
        max_depth=max_depth,
        min_depth=min_depth,
        start_depth=start_depth,
        increase_after=increase_after,
        decrease_after=decrease_after,
    )
    stop_token_ids = (
        _default_stop_tokens(rt.tokenizer) if stop_token_ids is None else stop_token_ids
    )
    started_all = time.perf_counter()
    cache, logits, hidden, target_time = _prefill(rt, prompt_ids, return_hidden=True)
    prompt_eval_time = target_time
    tokens: list[int] = []
    events: list[dict] = []
    accepted = rejected = drafted = 0
    accepted_by_depth = [0 for _ in range(max_depth)]
    drafted_by_depth = [0 for _ in range(max_depth)]
    accept_probability_sum_by_depth = [0.0 for _ in range(max_depth)]
    draft_time = verify_time = 0.0
    snapshot_time = accept_time = rollback_time = repair_time = 0.0

    step = 0
    while len(tokens) < max_tokens:
        primary, _ = _sample_from_logits(logits[0], sampler, rng)
        planned_depth = policy.current_depth
        event = {
            "step": step,
            "primary": primary,
            "depth": planned_depth,
            "max_depth": max_depth,
            "drafts": [],
            "accepted_depths": 0,
            "rejected_at_depth": None,
        }
        step += 1
        tokens.append(primary)
        if len(tokens) >= max_tokens or _is_stop(primary, stop_token_ids):
            events.append(event)
            break

        cycle_depth = min(planned_depth, max_tokens - len(tokens))
        draft_tokens: list[int] = []
        draft_probs: list[np.ndarray | SparseDistribution | None] = []
        mtp_cache = rt.make_mtp_cache() if mtp_cache_policy == "persistent" else None
        draft_hidden = hidden
        next_token = primary

        for depth_index in range(cycle_depth):
            step_mtp_cache = (
                mtp_cache if mtp_cache_policy == "persistent" else rt.make_mtp_cache()
            )
            started = time.perf_counter()
            draft_logits, draft_hidden_next = rt.draft_mtp(
                draft_hidden,
                mx.array([[next_token]]),
                mtp_cache=step_mtp_cache,
                return_hidden=True,
                mtp_hidden_variant=mtp_hidden_variant,
            )
            draft_token, draft_q = _sample_draft_from_logits(
                draft_logits[:, -1, :][0],
                draft_sampler,
                rng,
                need_distribution=sampler.temperature > 0,
            )
            elapsed_draft = time.perf_counter() - started
            draft_time += elapsed_draft
            draft_tokens.append(draft_token)
            draft_probs.append(draft_q)
            draft_hidden = draft_hidden_next[:, -1:, :]
            next_token = draft_token
            drafted += 1
            drafted_by_depth[depth_index] += 1
            event["drafts"].append(
                {
                    "depth": depth_index + 1,
                    "token": draft_token,
                    "timing_s": {"draft": elapsed_draft},
                }
            )

        started = time.perf_counter()
        before_verify = snapshot_untrimmable_cache(cache)
        elapsed_snapshot = time.perf_counter() - started
        snapshot_time += elapsed_snapshot
        _add_timing(event, "snapshot", elapsed_snapshot)
        verify_input = [primary] + draft_tokens
        started = time.perf_counter()
        with attention_phase("decode_verify"):
            verify_logits, verify_hidden = rt.forward_ar(
                mx.array([verify_input]),
                cache=cache,
                return_hidden=True,
            )
        _eval(verify_logits, verify_hidden)
        elapsed_verify = time.perf_counter() - started
        verify_time += elapsed_verify
        target_time += elapsed_verify

        accepted_count = 0
        rejection_correction: int | None = None
        started_accept = time.perf_counter()
        for depth_index, draft_token in enumerate(draft_tokens):
            target_logits_for_draft = verify_logits[:, depth_index, :]
            if sampler.temperature <= 0:
                target_token = int(
                    mx.argmax(target_logits_for_draft[0], axis=-1).item()
                )
                accepted_now = draft_token == target_token
                accept_prob = 1.0 if accepted_now else 0.0
                correction = target_token
            else:
                target_p = _distribution_from_mlx_logits(
                    target_logits_for_draft[0], sampler
                )
                draft_q = draft_probs[depth_index]
                if draft_q is None:
                    raise RuntimeError("non-greedy MTP requires draft distributions")
                accept_prob = compute_acceptance_probability(
                    target_p, draft_q, draft_token
                )
                accepted_now = float(rng.random()) <= accept_prob
                correction = (
                    draft_token
                    if accepted_now
                    else sample_from_distribution(
                        residual_distribution(target_p, draft_q), rng
                    )
                )

            event["drafts"][depth_index]["accepted"] = accepted_now
            event["drafts"][depth_index]["accept_probability"] = float(accept_prob)
            event["drafts"][depth_index]["correction"] = int(correction)
            accept_probability_sum_by_depth[depth_index] += float(accept_prob)

            if accepted_now:
                accepted += 1
                accepted_count += 1
                accepted_by_depth[depth_index] += 1
                if _is_stop(draft_token, stop_token_ids):
                    break
                continue

            rejected += 1
            event["rejected_at_depth"] = depth_index + 1
            if sampler.temperature > 0:
                rejection_correction = int(correction)
            break
        elapsed_accept = time.perf_counter() - started_accept
        accept_time += elapsed_accept
        _add_timing(event, "accept", elapsed_accept)

        event["accepted_depths"] = accepted_count
        event["policy"] = policy.observe(
            attempted_depth=cycle_depth,
            accepted_depths=accepted_count,
        )

        if accepted_count == len(draft_tokens):
            tokens.extend(draft_tokens)
            logits = verify_logits[:, len(draft_tokens), :]
            hidden = verify_hidden[:, -1:, :]
            events.append(event)
            if any(_is_stop(token, stop_token_ids) for token in draft_tokens):
                tokens = _truncate_after_first_stop(tokens, stop_token_ids)
                break
            continue

        committed = [primary] + draft_tokens[:accepted_count]
        if rejection_correction is not None:
            committed.append(rejection_correction)
        tokens.extend(committed[1:])

        started_rollback = time.perf_counter()
        rollback_after_verify(cache, before_verify, verified_tokens=len(verify_input))
        elapsed_rollback = time.perf_counter() - started_rollback
        rollback_time += elapsed_rollback
        _add_timing(event, "rollback", elapsed_rollback)
        started = time.perf_counter()
        with attention_phase("decode_verify"):
            repair_logits, repair_hidden = rt.forward_ar(
                mx.array([committed]),
                cache=cache,
                return_hidden=True,
            )
        _eval(repair_logits, repair_hidden)
        elapsed_repair = time.perf_counter() - started
        target_time += elapsed_repair
        repair_time += elapsed_repair
        _add_timing(event, "repair_forward", elapsed_repair)
        logits = repair_logits[:, -1, :]
        hidden = repair_hidden[:, -1:, :]
        events.append(event)

        if any(_is_stop(token, stop_token_ids) for token in committed):
            stop_index = next(
                i for i, token in enumerate(tokens) if _is_stop(token, stop_token_ids)
            )
            tokens = tokens[: stop_index + 1]
            break

    elapsed = time.perf_counter() - started_all
    stats = GenerationStats(
        mode="mtpa",
        generated_tokens=len(tokens),
        elapsed_s=elapsed,
        **_generation_rate_fields(
            generated_tokens=len(tokens),
            elapsed_s=elapsed,
            prompt_eval_time_s=prompt_eval_time,
        ),
        accepted_drafts=accepted,
        rejected_drafts=rejected,
        drafted_tokens=drafted,
        verify_time_s=verify_time,
        draft_time_s=draft_time,
        target_forward_time_s=target_time,
        prompt_eval_time_s=prompt_eval_time,
        prompt_tps=(
            len(prompt_ids) / prompt_eval_time if prompt_eval_time > 0 else 0.0
        ),
        prompt_target_prefill_time_s=prompt_eval_time,
        prompt_target_prefill_tok_s=(
            len(prompt_ids) / prompt_eval_time if prompt_eval_time > 0 else 0.0
        ),
        snapshot_time_s=snapshot_time,
        accept_time_s=accept_time,
        rollback_time_s=rollback_time,
        repair_time_s=repair_time,
        peak_memory_bytes=mx.get_peak_memory(),
        speculative_depth=max_depth,
        accepted_by_depth=accepted_by_depth,
        drafted_by_depth=drafted_by_depth,
        accept_probability_sum_by_depth=accept_probability_sum_by_depth,
        mean_accept_probability_by_depth=_mean_accept_probability_by_depth(
            accept_probability_sum_by_depth,
            drafted_by_depth,
        ),
        events=events,
    )
    _attach_runtime_diagnostics(stats, rt, counter_start)
    finish_reason = _finish_reason_from_tokens(
        tokens,
        stop_token_ids=stop_token_ids,
        max_tokens=max_tokens,
    )
    return GenerationOutput(
        tokens=tokens,
        text=_decode(rt.tokenizer, _strip_terminal_stop(tokens, stop_token_ids)),
        stats=stats,
        finish_reason=finish_reason,
    )
