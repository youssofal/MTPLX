"""Handlers for the product-facing MTPLX CLI surface."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import signal
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import importlib
import importlib.metadata
import importlib.util
import re
import webbrowser
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from mtplx.artifacts import inspect_model
from mtplx.benchmarks.validators.basic import (
    summarize_benchmark_quality,
    validate_balanced_delimiters,
    validate_no_degenerate_loop,
    validate_python_syntax,
)
from mtplx.constants import DEFAULT_RUNTIME_MODEL_DIR
from mtplx.default_models import (
    OPTIMIZED_QUALITY_DESCRIPTION,
    is_verified_default_model_ref,
    optimized_quality_model_ref,
    public_model_id_for_ref,
    select_default_model,
)
from mtplx.env import collect_environment
from mtplx.fan_mode import FAN_MODE_MAX, FAN_MODE_SMART, fan_mode_from_args
from mtplx.kpi import (
    EXIT_EXACTNESS,
    EXIT_QUALITY,
    EXIT_STRICT_GATE,
    EXIT_TELEMETRY,
    EXIT_UNSUPPORTED_MODEL,
    build_benchmark_envelope,
    default_output_path,
    exact_paged_attention_env,
    prompt_suite_path,
    run_exactness_smoke,
    summarize_vllm_reference,
    write_json,
)
from mtplx.kpi.runtime_kpis import (
    distribution_suite_names,
    repo_root,
)
from mtplx.backends.registry import (
    TIER_ARCH_COMPATIBLE_UNVERIFIED,
    architecture_catalog,
)
from mtplx.backends.descriptors import (
    QWEN3_8_DRAFT_TEMPERATURE,
    descriptor_for_architecture_id,
    descriptor_for_backend_id,
    descriptor_from_inspection,
    draft_semantics_for_model,
    model_controls_for_descriptor,
    model_family_from_inspection,
    reasoning_policy_for_model,
    sampler_defaults_for_model,
    tune_policy_for_model,
)
from mtplx.profiles import (
    DEFAULT_FP16_HF_MODEL_ID,
    DEFAULT_FP16_PUBLIC_MODEL_ID,
    DEFAULT_HF_MODEL_ID,
    DEFAULT_MODEL_ID,
    DEFAULT_PROFILE_NAME,
    DEFAULT_PUBLIC_MODEL_ID,
    LEGACY_OPTIMIZED_HF_MODEL_ID,
    LEGACY_OPTIMIZED_PUBLIC_MODEL_ID,
    OPTIMIZED_SPEED_V1_HF_MODEL_ID,
    OPTIMIZED_SPEED_V1_PUBLIC_MODEL_ID,
    OPTIMIZED_SPEED_V2_HF_MODEL_ID,
    OPTIMIZED_SPEED_V2_PUBLIC_MODEL_ID,
    QUALITY_FP16_HF_MODEL_ID,
    QUALITY_FP16_PUBLIC_MODEL_ID,
    QUALITY_HF_MODEL_ID,
    QUALITY_PUBLIC_MODEL_ID,
    QWEN35_9B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
    QWEN35_9B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID,
    QWEN35_9B_OPTIMIZED_SPEED_HF_MODEL_ID,
    QWEN35_9B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
    QWEN38_BARE_SPEED_FP16_HF_MODEL_ID,
    QWEN38_BARE_SPEED_FP16_PUBLIC_MODEL_ID,
    QWEN38_BARE_SPEED_HF_MODEL_ID,
    QWEN38_BARE_SPEED_PUBLIC_MODEL_ID,
    QWEN38_OPTIMIZED_QUALITY_FP16_HF_MODEL_ID,
    QWEN38_OPTIMIZED_QUALITY_FP16_PUBLIC_MODEL_ID,
    QWEN38_OPTIMIZED_QUALITY_HF_MODEL_ID,
    QWEN38_OPTIMIZED_QUALITY_PUBLIC_MODEL_ID,
    QWEN38_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
    QWEN38_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID,
    QWEN38_OPTIMIZED_SPEED_HF_MODEL_ID,
    QWEN38_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
    QWEN36_35B_OPTIMIZED_BALANCE_FP16_HF_MODEL_ID,
    QWEN36_35B_OPTIMIZED_BALANCE_FP16_PUBLIC_MODEL_ID,
    QWEN36_35B_OPTIMIZED_BALANCE_HF_MODEL_ID,
    QWEN36_35B_OPTIMIZED_BALANCE_PUBLIC_MODEL_ID,
    QWEN36_35B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
    QWEN36_35B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID,
    QWEN36_35B_OPTIMIZED_SPEED_HF_MODEL_ID,
    QWEN36_35B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
    apply_profile_env,
    get_profile,
    restore_profile_env,
    runtime_env_with_contract_overrides,
)
from mtplx.reasoning_effort import REASONING_EFFORT_CHOICES
from mtplx.server_urls import (
    bind_label,
    connect_host_for_bind,
    is_wildcard_bind,
    local_url_for_bind,
    network_url_for_bind,
)
from mtplx.kv_quant import paged_kv_quant_mode_from_env
from mtplx.runtime_options import (
    generate_api_key_file,
    normalize_paged_kv_quantization,
    paged_kv_quantization_env,
    resolve_api_key,
)


DEFAULT_CHAMPION = DEFAULT_MODEL_ID
QUICKSTART_SPEED_MIN_TOKENS = 64
QUICKSTART_SPEED_MAX_TOKENS = 192
QUICKSTART_SPEED_PROMPT = (
    "Create a compact single-file HTML5 Canvas Flappy Bird game. "
    "Draw visuals procedurally, include physics, score, restart, and no prose."
)
QUICKSTART_TARGETS = {
    "terminal",
    "openwebui",
    "open-webui",
    "pi",
    "opencode",
    "swival",
    "hermes",
    "dashboard",
}
HERMES_PROFILE_NAME = "mtplx"
HERMES_LOCAL_API_KEY = "mtplx-local"
HERMES_CODING_TOOLSETS = ("terminal", "file", "web", "browser", "messaging")
HERMES_CODING_TOOLSETS_TEXT = ",".join(HERMES_CODING_TOOLSETS)
HERMES_CAPABILITY_SUMMARY = "Terminal, file, web, browser, and messaging tools."
HERMES_GATEWAY_STATUS_COMMAND = "env -u HERMES_HOME hermes gateway status"
HERMES_GATEWAY_TRUTH_HINT = (
    "MTPLX uses a profile-scoped HERMES_HOME for model routing, while Hermes "
    "Gateway runs from the root ~/.hermes LaunchAgent. For live messaging "
    "truth, use `env -u HERMES_HOME hermes gateway status` and "
    "send_message(action='list')."
)
HERMES_MESSAGING_SETUP_HINT = (
    "Messaging uses Hermes Gateway. Setup is `hermes gateway setup`; choose "
    "Telegram, provide the required token/user/channel values, and then run "
    "`hermes gateway start` if the LaunchAgent is stale. Never print token, "
    "user id, channel id, webhook URL, API key, or other secret values from "
    ".env; report only configured, missing, connected, not connected, or "
    "needs repair unless the user explicitly asks for a redacted diagnostic."
)
HERMES_SYSTEM_PROMPT = (
    "You are Hermes inside MTPLX on macOS. You have terminal, file, web, "
    "browser, and messaging tools. MTPLX owns model routing through the local "
    "OpenAI-compatible server. Do not send external messages unless the user "
    "explicitly gives both the destination and content. Do not print token, "
    "user id, channel id, webhook URL, API key, or other secret values from "
    ".env; report only configured, missing, connected, not connected, or "
    "needs repair unless explicitly asked for a redacted diagnostic. "
    + HERMES_GATEWAY_TRUTH_HINT
)
LONG_RESPONSE_DIRECT_PROFILE = (
    "vllm_metal_paged_attn_partitioned_block_16_blocks_1024_"
    "partition_threshold_2048_impl_mlx_vector_paged"
)
BENCH_SUSTAINED_DEFAULT_SUITES = {
    "flappy",
    "long_code_uncapped",
    "long-code-uncapped",
    "python_modules_long",
    "python-modules-long",
}
BENCH_SUSTAINED_LENGTH_SENSITIVE_SUITES = {"long_code", "long-code"}
BENCH_SUSTAINED_MAX_TOKENS_THRESHOLD = 512
BENCH_SUITE_FULL_EXACTNESS_CONTEXTS = "64,2048,6144,10240"
BENCH_SUITE_QUICK_EXACTNESS_CONTEXTS = "64,2048"
EXTERNAL_RUNTIME_ENV_KEYS = (
    "MTPLX_VERIFY_OUTPUT_DEPENDS",
    "MTPLX_VERIFY_OUTPUT_DEPENDS_AFTER_TOKENS",
    "MTPLX_VERIFY_OUTPUT_DEPENDS_EVERY",
    "MTPLX_VERIFY_OUTPUT_DEPENDS_INCLUDE_MTP",
    "MTPLX_STATE_ROOT_EVAL_MODE",
    "MTPLX_EVAL_STATE_ROOTS_INCLUDE_MTP",
    "MTPLX_EVAL_STATE_ROOTS_INCLUDE_LIVE",
    "MTPLX_TARGET_LAYER_EVAL_MODE",
    "MTPLX_FUSE_ATTN_QKV_PROJECTIONS",
    "MTPLX_SDPA_2PASS_BLOCKS",
    "MTPLX_SDPA_DYNAMIC_OFFSET_ACTIVE_BLOCKS",
    "MTPLX_EXPORT_VERIFY_DOT_DIR",
    "MTPLX_EXPORT_VERIFY_DOT_CYCLES",
    "MTPLX_EXPORT_VERIFY_DOT_INCLUDE_CACHE",
    "MTPLX_EXPORT_VERIFY_DOT_INCLUDE_CAPTURES",
)
LOCALHOST_BINDS = {"", "127.0.0.1", "::1", "localhost"}
MAX_PUBLIC_SPECULATIVE_DEPTH = 3
MAX_GEMMA4_SPECULATIVE_DEPTH = 8
TUNE_DEFAULT_DEPTHS = "1,2,3"
TUNE_DEFAULT_SUITE = "cold-long-code-192"
TUNE_DEFAULT_MAX_TOKENS = 512
TUNE_DEFAULT_LIMIT = 1
TUNE_DEFAULT_SEED = 0
TUNE_STATE_PATH = Path("~/.mtplx/tuning.json").expanduser()
TUNE_TELEMETRY_SAMPLE_INTERVAL_S = 0.75
TUNE_POWERMETRICS_SAMPLE_INTERVAL_S = 1.5
TUNE_CANDIDATE_SETTLE_S = 5.0
TUNE_TIE_PREFER_DEEPER_WITHIN_PCT = 2.0
TUNE_ACCEPTANCE_COLLAPSE_THRESHOLD = 0.05
# Measured verdicts that affirmatively prove MTP loses in this environment;
# only these clear a previously saved winner (a failed tune never does).
TUNE_RECORD_CLEARING_VERDICTS = frozenset(
    {"no_mtp_depth_beat_ar", "mtp_acceptance_collapsed"}
)
TUNE_TELEMETRY_ENV = "MTPLX_BENCH_TUNE_TELEMETRY"
GENERATION_MODE_MTP = "mtp"
GENERATION_MODE_AR = "ar"
GENERATION_MODES = {GENERATION_MODE_MTP, GENERATION_MODE_AR}
OPENCODE_CHAT_TEMPLATE_PROFILE_DEFAULT = "local_qwen36"
OPENCODE_FAIR_BATCHING_DEFAULTS: dict[str, Any] = {
    # 2026-07-16 agent-lane TPS alignment: `mtplx start opencode` now matches
    # the app's OpenCode launch card (serial + latency). The ar_batch/agent
    # lane co-schedules OpenCode's title/summarize side calls with the main
    # turn and measured 36.8 vs 51.4 decode tok/s at 8k (31.6 vs 42.4 at 33k)
    # against the serial turbo lane on the same daemon/model — single-stream
    # coding turns are the product path, and the app's launch card comment
    # documents the same measured call. Explicit --scheduler-mode/--batching-
    # preset flags still win (per-flag skip in _apply_opencode_fair_defaults).
    "scheduler_mode": "serial",
    "batching_preset": "latency",
    "decode_batch_max": None,
    "batch_wait_ms": None,
    "prefill_chunk_tokens": 2048,
    "ssd_session_cache": "on",
    "ssd_session_cache_max_size": "32GB",
    "ssd_session_cache_min_prefix_tokens": 1024,
}
HERMES_LATENCY_DEFAULTS: dict[str, Any] = {
    "scheduler_mode": "serial",
    "batching_preset": "latency",
    "max_active_requests": None,
    "decode_batch_max": None,
    "batch_wait_ms": None,
    "prefill_chunk_tokens": 2048,
    "ssd_session_cache": "on",
    "ssd_session_cache_max_size": "100GB",
    "ssd_session_cache_min_prefix_tokens": 512,
    "temperature": 0.6,
    "top_p": 1.0,
    "top_k": 20,
    "tool_prompt_mode": "hybrid",
    "chat_template_profile": OPENCODE_CHAT_TEMPLATE_PROFILE_DEFAULT,
    "adaptive_policy": "expected_value",
    "adaptive_min_depth": 1,
    "adaptive_ev_base_depth": 2,
    "adaptive_ev_warmup_full_depth_cycles": 4,
    "adaptive_ev_exploration_interval": 32,
    "reasoning": "auto",
    "preserve_thinking": "auto",
}
_OPENCODE_HIGH_MEMORY_THRESHOLD_BYTES = 96 * 1024**3
_OPENCODE_HIGH_MEMORY_MAX_BYTES = "24G"
_OPENCODE_HIGH_MEMORY_PER_SESSION_BYTES = "16G"
_OPENCODE_DEFAULT_MAX_ENTRIES = "6"
# 2026-07-04 multitask finding: one busy OpenCode conversation banked 13
# entries (20.6 GB) before prefix-supersede landed; 16 slots left nothing for
# a second or third project. With supersede a conversation settles around
# 2-4 live entries, so 32 slots hold several projects warm while the 24G
# byte budget remains the real guard.
_OPENCODE_HIGH_MEMORY_MAX_ENTRIES = "32"


def _detect_total_ram_bytes_for_opencode_defaults() -> int | None:
    if sys.platform != "darwin":
        return None
    try:
        # Absolute path: app-owned daemons run with a sanitized PATH that
        # lacks /usr/sbin (see engine_session RAM detection).
        output = subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "hw.memsize"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
        total = int(str(output).strip())
    except Exception:
        return None
    return total if total > 0 else None


def _opencode_memory_env_defaults() -> dict[str, str]:
    total_ram = _detect_total_ram_bytes_for_opencode_defaults()
    high_memory = (
        total_ram is not None and total_ram >= _OPENCODE_HIGH_MEMORY_THRESHOLD_BYTES
    )
    max_entries = (
        _OPENCODE_HIGH_MEMORY_MAX_ENTRIES
        if high_memory
        else _OPENCODE_DEFAULT_MAX_ENTRIES
    )
    return {
        # Long-context decode route: route only — the MIN_CONTEXT/MIN_Q/MAX_Q
        # overrides (32768/3/5, unmeasured 1.0.0 launch values) are gone in
        # lockstep with the app's codingAgentRuntimeEnvironment so the engine
        # defaults (65536/4/5) govern. Issue #228 measured async_per_head
        # below 64k at 4-7x SLOWER decode at 43k ctx.
        "MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE": "async_per_head",
        "MTPLX_SESSION_BLOCK_PREFIX_RESTORE": "1",
        "MTPLX_SESSION_BANK_MAX_ENTRIES": max_entries,
        # "auto" = the engine budgets half the RAM surplus left after the
        # model weights (floor 1 GiB, cap 48 GiB). Replaces the flat
        # 24G/8G tier that ignored the loaded model's size (founder memory
        # ruling 2026-07-05: 55 GB total was fine on 128 GB, lethal on 32).
        "MTPLX_SESSION_BANK_MAX_BYTES": "auto",
        "MTPLX_SESSION_BANK_PER_SESSION_BYTES": "auto",
        "MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S": "30.0",
        "MTPLX_DYNAMIC_PAGED_KV_MAX_INITIAL_NEW_TOKENS": "4096",
        "MTPLX_LAZY_TARGET_DISTRIBUTIONS": "1",
        "MTPLX_LAZY_BONUS_VERIFY": "1",
        "MTPLX_OPENCODE_TOOL_HISTORY_LIVE_FRONTIER": "1",
        "MTPLX_SESSION_LIVE_FRONTIER_REFERENCE_RESTORE": "1",
        # The read-inspection compaction battery is gone (#282): an explicit
        # env re-arms that compactor past the engine's passthrough default,
        # so exporting the battery here silently rewrote agent transcripts.
        # In lockstep with the app's codingAgentRuntimeEnvironment.
        "MTPLX_TOOL_PROMPT_MODE": "hybrid",
        "MTPLX_CHAT_TEMPLATE_PROFILE": OPENCODE_CHAT_TEMPLATE_PROFILE_DEFAULT,
    }


def _absolute_user_path(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded
    return (Path.cwd() / expanded).resolve()


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _is_localhost_bind(host: str | None) -> bool:
    return str(host or "").strip().lower().strip("[]") in LOCALHOST_BINDS


def _benchmark_seed(args: Any, *, runtime_profile: str, harness: str) -> int:
    explicit = getattr(args, "seed", None)
    if explicit is not None:
        return int(explicit)
    if runtime_profile == "native_mtp_60_cold" or harness == "depth-sweep":
        return 0
    return 42


def _depths_for_bench_run(args: Any) -> str:
    explicit_depths = getattr(args, "depths", None)
    if explicit_depths:
        return str(explicit_depths)
    depth = int(getattr(args, "speculative_depth", 0) or 0)
    return str(depth) if depth > 0 else "3"


def _runtime_env_with_external_overrides(runtime_env: dict[str, str]) -> dict[str, str]:
    merged = dict(runtime_env)
    for key in EXTERNAL_RUNTIME_ENV_KEYS:
        value = os.environ.get(key)
        if value is not None and value != "":
            merged[key] = value
    return merged


def _model_runtime_contract(inspection: dict[str, Any]) -> dict[str, Any] | None:
    compatibility = (
        inspection.get("compatibility") if isinstance(inspection, dict) else None
    )
    if isinstance(compatibility, dict):
        contract = compatibility.get("runtime_contract")
        if isinstance(contract, dict):
            return contract
    contract = (
        inspection.get("runtime_contract") if isinstance(inspection, dict) else None
    )
    return contract if isinstance(contract, dict) else None


def _profile_scoped_model_runtime_contract(
    inspection: dict[str, Any],
    profile: Any,
) -> dict[str, Any] | None:
    contract = _model_runtime_contract(inspection)
    if not isinstance(contract, dict):
        return None
    recommended = str(contract.get("recommended_profile") or "").strip()
    if not recommended:
        return contract
    active = str(getattr(profile, "name", profile) or "").strip()
    if recommended == active:
        return contract
    return None


def _runtime_env_with_model_contract_overrides(
    runtime_env: dict[str, str],
    inspection: dict[str, Any],
    profile: Any,
) -> dict[str, str]:
    return runtime_env_with_contract_overrides(
        runtime_env,
        _profile_scoped_model_runtime_contract(inspection, profile),
    )


def _bench_run_console_summary(envelope: dict[str, Any]) -> dict[str, Any]:
    runtime = envelope.get("runtime") or {}
    trace = envelope.get("decode_trace") or {}
    quality = envelope.get("quality") or {}
    correctness = envelope.get("correctness") or {}
    smoke = correctness.get("exactness_smoke") or {}
    return {
        "run_id": envelope.get("run_id"),
        "harness": envelope.get("harness") or "depth-sweep",
        "runtime_profile": envelope.get("runtime_profile"),
        "tok_s": runtime.get("tok_s") or runtime.get("mean_tok_s"),
        "generated_tokens": runtime.get("generated_tokens")
        or trace.get("generated_tokens"),
        "first64_tok_s": trace.get("first64_tok_s"),
        "last64_tok_s": trace.get("last64_tok_s"),
        "last64_over_first64": trace.get("last64_over_first64"),
        "last10_over_first10": trace.get("last10_over_first10"),
        "late_verify_ms": trace.get("late_verify_ms") or runtime.get("late_verify_ms"),
        "quality_passed": quality.get("passed"),
        "quality_failures": quality.get("failures") or [],
        "exactness_smoke_passed": smoke.get("passed"),
        "strict_passed": envelope.get("strict_passed"),
        "artifacts": envelope.get("artifacts") or {},
    }


def _model_gate(
    model: str,
    *,
    unsafe_force_unverified: bool = False,
    yes: bool = False,
) -> tuple[dict[str, Any], int | None]:
    try:
        inspection = inspect_model(model).to_dict()
    except Exception as exc:
        return {"error": "inspect failed", "model": model, "detail": str(exc)}, 1
    compatibility = inspection.get("compatibility") or {}
    tier = compatibility.get("tier")
    exit_code = int(compatibility.get("exit_code", EXIT_UNSUPPORTED_MODEL))
    # The gate asks one question: can this artifact execute. Verification
    # tier stays a label (inspect exit codes still report it); it must
    # never block loading.
    if compatibility.get("can_run"):
        if compatibility.get("unverified_model"):
            print(
                "WARNING: running a family-compatible MTPLX model without a "
                "recorded mtplx_runtime.json exactness baseline; stats will be "
                "marked unverified until this artifact is smoke-verified.",
                file=sys.stderr,
            )
        return inspection, None
    if unsafe_force_unverified and yes and tier == TIER_ARCH_COMPATIBLE_UNVERIFIED:
        print(
            "WARNING: attempting an architecture-compatible but unverified MTPLX "
            "model; startup will continue and the loader result is authoritative.",
            file=sys.stderr,
        )
        return inspection, None
    return inspection, exit_code


def _compact_model_summary(inspection: dict[str, Any]) -> dict[str, Any]:
    compatibility = inspection.get("compatibility") or {}
    return {
        "source": inspection.get("source"),
        "model_dir": inspection.get("model_dir"),
        "architecture": inspection.get("architecture"),
        "model_type": inspection.get("model_type"),
        "mtp_arch": inspection.get("mtp_arch"),
        "mtp_supported": inspection.get("mtp_supported"),
        "recommended_backend": inspection.get("recommended_backend"),
        "recommended_profile": (
            compatibility.get("recommended_profile")
            or inspection.get("recommended_profile")
        ),
        "runtime_compatibility": inspection.get("runtime_compatibility"),
        "runtime_contract_path": (
            compatibility.get("runtime_contract_path")
            or inspection.get("runtime_contract_path")
        ),
        "compatibility": {
            "tier": compatibility.get("tier"),
            "can_run": compatibility.get("can_run"),
            "supported": compatibility.get("supported"),
            "recognized": compatibility.get("recognized"),
            "exit_code": compatibility.get("exit_code"),
            "message": compatibility.get("message"),
            "arch_id": compatibility.get("arch_id"),
            "unsafe_force_required": compatibility.get("unsafe_force_required"),
            "unverified_model": compatibility.get("unverified_model"),
            "runtime_compatibility": compatibility.get("runtime_compatibility"),
            "support_level": compatibility.get("support_level"),
            "support_notes": compatibility.get("support_notes"),
        },
    }


def _model_gate_error_lines(inspection: dict[str, Any]) -> list[str]:
    compatibility = inspection.get("compatibility") or {}
    mtp = inspection.get("mtp") or {}
    lines = [
        "error: model cannot run with MTPLX",
        f"model: {inspection.get('model_dir') or inspection.get('model') or 'unknown'}",
        f"tier: {compatibility.get('tier') or 'unknown'}",
    ]
    runtime_compatibility = compatibility.get(
        "runtime_compatibility"
    ) or inspection.get("runtime_compatibility")
    mtp_layers = int(inspection.get("mtp_num_hidden_layers") or 0)
    missing_mtp_weights = (
        mtp_layers > 0
        and bool(mtp)
        and not bool(mtp.get("exists"))
        and compatibility.get("tier") != "no-MTP"
    )
    if missing_mtp_weights:
        runtime_compatibility = "missing-mtp-weights"
    if runtime_compatibility:
        lines.append(f"runtime: {runtime_compatibility}")
    if missing_mtp_weights:
        message = (
            "This model's config advertises MTP, but MTPLX did not find "
            "runnable MTP weights in the folder. mtplx_runtime.json is "
            "optional metadata; the blocker is missing MTP weights."
        )
    else:
        message = compatibility.get("message") or inspection.get("detail")
    if message:
        lines.append(f"reason: {message}")
    if mtp:
        lines.append(
            "mtp: "
            f"exists={str(bool(mtp.get('exists'))).lower()}, "
            f"tensors={mtp.get('tensor_count', 0)}, "
            f"gate={str(bool(mtp.get('passes_tensor_gate'))).lower()}"
        )
    if runtime_compatibility == "missing-mtp-weights":
        lines.append(
            "fix: use a complete model with matching MTP weights, or build and "
            "verify one from its original source with mtplx forge. MTPLX does not "
            "attach arbitrary sidecars because config/tensor checks cannot prove "
            "their trunk lineage."
        )
    elif compatibility.get("tier") == TIER_ARCH_COMPATIBLE_UNVERIFIED:
        lines.append(
            "try: add --unsafe-force-unverified --yes to run without support guarantees"
        )
    else:
        lines.append("try: mtplx inspect MODEL")
    return lines


def _print_model_gate_error(
    inspection: dict[str, Any],
    *,
    printer=print,
    json_output: bool = False,
) -> None:
    if json_output:
        _print(
            {
                "error": "model failed MTPLX compatibility gate",
                "model": _compact_model_summary(inspection),
            }
        )
        return
    for line in _model_gate_error_lines(inspection):
        printer(line)


def _print_command_error(
    payload: dict[str, Any],
    *,
    command: str,
    json_output: bool = False,
) -> None:
    if json_output:
        _print(payload)
        return
    model = payload.get("model")
    if isinstance(model, dict):
        _print_model_gate_error(model, json_output=False)
        return
    error = str(payload.get("error") or "model is not available locally")
    print(f"error: {error}")
    if model:
        print(f"model: {model}")
    detail = payload.get("detail")
    if detail:
        print(f"detail: {detail}")
    if error == "model is not available locally":
        print(f"try: mtplx {command} --download --model {model}")
        print("try: mtplx models")


def _looks_like_gemma4_model_ref(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    normalized = text.replace("_", "-").lower()
    if "gemma4" in normalized or "gemma-4" in normalized:
        return True
    try:
        path = Path(text).expanduser()
    except (TypeError, ValueError):
        return False
    return (path / "mtplx_pair.json").exists()


def _public_depth_ceiling(args: Any) -> int:
    refs = (getattr(args, "model", None), getattr(args, "model_id", None))
    if any(_looks_like_gemma4_model_ref(ref) for ref in refs):
        return MAX_GEMMA4_SPECULATIVE_DEPTH
    for ref in refs:
        if not ref:
            continue
        family = model_family_from_inspection(None, model_ref=str(ref))
        if family == "qwen3_8":
            descriptor = descriptor_for_architecture_id("qwen3-next-mtp")
            return draft_semantics_for_model(
                str(ref), descriptor=descriptor
            ).maximum
        # The artifact ref decides the ceiling. Once the default served
        # name became the 3.8 id (2026-08-14 flip), letting the model_id
        # alias widen the gate would grant D4-D6 to any non-3.8 artifact
        # served under the default identity; keep the pre-3.8 ceiling and
        # let the post-inspection contract path raise it when the real
        # artifact supports it.
        break
    return MAX_PUBLIC_SPECULATIVE_DEPTH


def _validate_public_depth(args: Any, *, printer=print) -> int | None:
    try:
        depth = int(getattr(args, "depth", 3))
    except (TypeError, ValueError):
        printer("error: --depth must be an integer")
        return 2
    depth_ceiling = _public_depth_ceiling(args)
    if depth < 1 or depth > depth_ceiling:
        printer(
            "error: --depth must be between "
            f"1 and {depth_ceiling} for the selected MTPLX runtime"
        )
        printer("hint: omit --depth to use the model contract default")
        return 2
    args.depth = depth
    return None


def _normalize_generation_mode(value: Any) -> str:
    text = str(value or GENERATION_MODE_MTP).strip().lower()
    if text == "auto":
        # Older app builds persisted "auto"; it means the engine default.
        return GENERATION_MODE_MTP
    if text not in GENERATION_MODES:
        raise ValueError("generation mode must be 'mtp' or 'ar'")
    return text


def _generation_mode_from_args(args: Any) -> str:
    if bool(getattr(args, "stock_ar", False)):
        return GENERATION_MODE_AR
    explicit = getattr(args, "generation_mode", None)
    if explicit is not None:
        if str(explicit).strip().lower() == "auto" and (
            getattr(args, "load_mtp", True) is False
            or bool(getattr(args, "no_mtp", False))
        ):
            return GENERATION_MODE_AR
        return _normalize_generation_mode(explicit)
    if getattr(args, "load_mtp", True) is False:
        return GENERATION_MODE_AR
    return (
        GENERATION_MODE_AR
        if bool(getattr(args, "no_mtp", False))
        else GENERATION_MODE_MTP
    )


def _apply_runtime_compatibility_mode(
    args: Any,
    inspection: dict[str, Any],
    *,
    printer=print,
) -> int | None:
    compatibility = inspection.get("compatibility")
    if isinstance(compatibility, dict):
        runtime_compatibility = compatibility.get(
            "runtime_compatibility"
        ) or inspection.get("runtime_compatibility")
    else:
        # inspect's four-tier contract returns ``compatibility`` as a plain
        # string tier; the runtime-lane marker then lives at top level.
        runtime_compatibility = inspection.get("runtime_compatibility")
    if runtime_compatibility == "native-ar-only-missing-mtp":
        # Founder directive 2026-08-09: a missing MTP head degrades, never
        # blocks. Announce loudly, then serve the trunk autoregressive.
        if _generation_mode_from_args(args) != GENERATION_MODE_AR:
            printer(
                "mtp_heads not found -> mtp_off: serving autoregressive "
                "(no speculative decode acceleration; build an MTP artifact "
                "with Forge for full speed)."
            )
            _set_generation_mode_on_args(args, GENERATION_MODE_AR)
            setattr(args, "depth", 0)
        setattr(args, "load_mtp", False)
        return None
    if runtime_compatibility != "native-ar-only":
        return None
    if _generation_mode_from_args(args) != GENERATION_MODE_AR:
        # Same founder directive as the missing-head case: target-only AR
        # architectures degrade loudly instead of blocking on --no-mtp.
        printer(
            "target-only AR architecture -> mtp_off: serving autoregressive "
            "(this checkpoint family has no native MTP head)."
        )
        _set_generation_mode_on_args(args, GENERATION_MODE_AR)
        setattr(args, "depth", 0)
    setattr(args, "load_mtp", False)
    return None


def _fan_mode_from_args(args: Any) -> str:
    mode = fan_mode_from_args(args)
    setattr(args, "fan_mode", mode)
    setattr(args, "max", mode == FAN_MODE_MAX)
    return mode


def _set_generation_mode_on_args(args: Any, mode: str) -> None:
    normalized = _normalize_generation_mode(mode)
    setattr(args, "generation_mode", normalized)
    setattr(args, "no_mtp", normalized == GENERATION_MODE_AR)


def _generation_mode_label(mode: str) -> str:
    return "AR target-only" if mode == GENERATION_MODE_AR else "MTP"


def _format_bytes(size_bytes: int | float | None) -> str:
    if not isinstance(size_bytes, (int, float)):
        return "unknown size"
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1000.0 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1000.0
    return f"{size:.1f} TB"


def _download_progress_callback(*, printer=print):
    """Plain-text download progress callback for non-interactive paths."""

    def emit(event: dict[str, Any]) -> None:
        kind = event.get("event")
        size = _format_bytes(event.get("size_bytes"))
        if kind == "start":
            printer(
                f"[1/4] Download started: {event.get('repo_id')} -> {event.get('path')}"
            )
        elif kind == "resume":
            printer(
                "[1/4] Resuming partial download: "
                f"{event.get('repo_id')} ({size} already on disk)"
            )
        elif kind == "progress":
            interval = max(1, int(round(float(event.get("interval_s") or 0))))
            delta = float(event.get("delta_bytes") or 0)
            if delta > 0:
                printer(
                    "[1/4] Still downloading: "
                    f"{size} on disk (+{_format_bytes(delta)} in {interval}s)"
                )
            else:
                printer(
                    "[1/4] Still downloading: "
                    f"{size} on disk (no byte change in {interval}s; waiting on Hugging Face)"
                )
        elif kind == "complete":
            printer(f"[1/4] Download complete: {size} on disk")

    return emit


def _rich_download_progress_callback(*, repo_id: str, total_bytes: int | None = None):
    """Single-line live progress callback for interactive downloads."""

    from mtplx.ui.download_progress import (
        RichDownloadProgress,
        from_progress_event_callback,
    )

    progress = RichDownloadProgress(repo_id=repo_id, total_bytes=total_bytes)
    callback = from_progress_event_callback(progress=progress)

    def finalize() -> None:
        progress.stop()

    return callback, finalize


def _profile_draft_lm_head_spec(profile: Any) -> dict[str, Any] | None:
    draft = getattr(profile, "draft_lm_head", None)
    if draft is None:
        return None
    return {
        "bits": int(draft.bits),
        "group_size": int(draft.group_size),
        "mode": str(draft.mode),
    }


def _profile_draft_sampler_spec(profile: Any) -> dict[str, Any] | None:
    draft = getattr(profile, "draft_sampler", None)
    if draft is None:
        return None
    return {
        "temperature": float(draft.temperature),
        "top_p": float(draft.top_p),
        "top_k": int(draft.top_k),
    }


def _model_draft_lm_head_spec(
    inspection: dict[str, Any],
    profile: Any,
) -> dict[str, Any] | None:
    """Use profile-matching model contract draft-head metadata, else default."""
    fallback = _profile_draft_lm_head_spec(profile)
    try:
        from mtplx.draft_lm_head import draft_lm_head_spec_from_runtime_contract

        contract = _profile_scoped_model_runtime_contract(inspection, profile)
        return draft_lm_head_spec_from_runtime_contract(contract, fallback=fallback)
    except ImportError:
        return fallback


def _model_draft_sampler_spec(
    inspection: dict[str, Any],
    profile: Any,
) -> dict[str, Any] | None:
    """Use profile-matching model contract draft-sampler metadata, else default."""
    fallback = _profile_draft_sampler_spec(profile)
    try:
        from mtplx.draft_sampling import draft_sampler_spec_from_runtime_contract

        contract = _profile_scoped_model_runtime_contract(inspection, profile)
        if not isinstance(contract, dict):
            # A profile mismatch (artifact recommends another profile) hides
            # the typed contract, but the recommended draft sampler is a
            # property of the ARTIFACT, not of the profile match — exactly
            # like _model_contract_depth above: keep resolving from the
            # top-level mtplx_runtime.json metadata so serving --profile
            # turbo against a sustained-stamped artifact does not silently
            # drop the artifact's draft-sampler stamp.
            contract = _artifact_runtime_metadata(inspection)
        try:
            return draft_sampler_spec_from_runtime_contract(
                contract, fallback=fallback
            )
        except (TypeError, ValueError):
            # Artifact metadata is fail-safe by contract; a malformed
            # recommended_draft_sampler degrades to the profile default
            # instead of failing the launch.
            return fallback
    except ImportError:
        return fallback


# Artifacts whose fastest measured mode is shallower than their sidecar
# ceiling. Same measured-win-only shape as _TURBO_DEFAULT_PUBLIC_MODEL_IDS
# below, keyed by public model id; artifacts that declare
# ``mtp_depth_default`` in their runtime contract take precedence.
#
# Qwen3.6-35B-A3B: per-level acceptance decays steeply on this MoE, so the
# D3 ceiling is never the fastest mode. Two independent sweeps, greedy:
#   M5 Max tune, pinned fans, 2026-07-19 (isolated candidates):
#     AR 93.9 | D1 130.0 (1.39x) | D2 145.0 (1.54x) | D3 132.2 (1.41x)
#     acceptance D2 [0.813, 0.575]; D3 tail level 0.218
#   PR #174 reporter's sweep: AR 94.46 | D1 138.39 | D2 135.66 | D3 107.67
# D2 is at or within noise of best in both datasets while D1 loses ~10% on
# the M5 measurement and the D3 ceiling loses 9-22% everywhere, so D2 is
# the fleet default.
# Ceiling-vs-default split contributed by davidtai (PR #174).
_MODEL_CONTRACT_DEPTH_DEFAULTS: dict[str, int] = {
    QWEN36_35B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID: 2,
    QWEN36_35B_OPTIMIZED_BALANCE_PUBLIC_MODEL_ID: 2,
    # Qwen3.8 27B Optimized Quality briefly carried a :2 entry here from the
    # drop-day forge-verify rows (D3 "18.8" vs D2 33.9). The gated live ABBA
    # then measured D2 39.5 vs D3 40.6 matched-window — the collapse was
    # order/JIT confound in single in-process tune rows, so the family D3
    # ceiling stands and the entry was removed the same day. Never pin from
    # forge-verify rows (mistakes ledger, 2026-08-14).
}


def _artifact_runtime_metadata(inspection: dict[str, Any]) -> dict[str, Any]:
    """Top-level ``mtplx_runtime.json`` of the inspected artifact, or ``{}``.

    The typed contract keeps only schema fields; identity and depth-default
    keys live beside them at the top level of the artifact json. Reading it
    here is fail-safe: any I/O or parse problem returns an empty dict and the
    caller falls back to contract-only behavior.
    """
    model_dir = inspection.get("model_dir") if isinstance(inspection, dict) else None
    if not model_dir:
        return {}
    try:
        path = Path(str(model_dir)) / "mtplx_runtime.json"
        if not path.is_file():
            return {}
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _model_contract_depth(
    inspection: dict[str, Any],
    *,
    profile: Any,
    fallback: int = 3,
) -> int:
    if int(fallback) == 0:
        # fallback 0 is the missing-MTP degrade pin (AR mode); artifact
        # metadata must not resurrect a draft depth past it.
        return 0
    contract = _profile_scoped_model_runtime_contract(inspection, profile)
    if not isinstance(contract, dict):
        # A profile mismatch (artifact recommends another profile) hides the
        # typed contract, but the measured depth default is a property of the
        # ARTIFACT, not of the profile match: keep resolving from the
        # top-level mtplx_runtime.json metadata. Repro: 3.8 Optimized
        # Quality stamps recommended_profile=sustained + mtp_depth_default=2;
        # serving --profile turbo used to early-return here and launch the
        # measured-worst depth 3.
        contract = {}
    metadata_for_max = _artifact_runtime_metadata(inspection)
    if not contract and not metadata_for_max:
        # No typed contract and no artifact metadata: exact legacy behavior.
        return int(fallback)
    try:
        depth_max = int(
            contract.get(
                "mtp_depth_max", metadata_for_max.get("mtp_depth_max", fallback)
            )
        )
    except (TypeError, ValueError):
        return int(fallback)
    # ``mtp_depth_max`` is a CEILING (the deepest sidecar the artifact
    # supports), not a recommendation. Artifacts whose fastest mode is
    # shallower declare ``mtp_depth_default``; the ceiling then only bounds
    # it. Without the split every artifact runs at its maximum depth, a
    # measured loss whenever per-level acceptance decays quickly.
    #
    # The typed RuntimeContract serialization carries only its schema fields,
    # so identity and depth-default keys living at the top level of
    # ``mtplx_runtime.json`` never reach this dict — which silently killed the
    # measured-depth map for every artifact (35B-A3B quickstart launched at
    # its D3 ceiling, the exact -22% regression repro_a3b_depth_default.py
    # documents). Resolve both from the artifact metadata when the contract
    # dict lacks them; ``recommended_mtp_depth`` is honored as the historical
    # spelling of ``mtp_depth_default`` (the Balance artifact ships it).
    metadata = _artifact_runtime_metadata(inspection)
    public_id = str(
        contract.get("public_model_id") or metadata.get("public_model_id") or ""
    ).strip()
    measured_default = _MODEL_CONTRACT_DEPTH_DEFAULTS.get(public_id)
    declared_default = next(
        (
            source.get(key)
            for source in (contract, metadata)
            for key in ("mtp_depth_default", "recommended_mtp_depth")
            if source.get(key) is not None
        ),
        None,
    )
    try:
        depth = int(
            declared_default
            if declared_default is not None
            else (measured_default or depth_max)
        )
    except (TypeError, ValueError):
        depth = measured_default or depth_max
    depth = min(depth, depth_max)
    model_ref = inspection.get("model_dir") or inspection.get("runtime_model")
    descriptor = descriptor_from_inspection(inspection)
    depth_ceiling = draft_semantics_for_model(
        str(model_ref or ""), inspection, descriptor
    ).maximum
    return max(1, min(depth_ceiling, depth))


def _apply_model_contract_depth_default(
    args: Any,
    inspection: dict[str, Any],
    profile: Any,
) -> None:
    cli_flags = getattr(args, "_cli_flags", set()) or set()
    if "depth" in cli_flags:
        return
    args.depth = _model_contract_depth(
        inspection,
        profile=profile,
        fallback=int(getattr(args, "depth", 3)),
    )


# Quantized 27B flagships whose measured default profile is turbo. This is
# the same launch rule the macOS app applies in MTPLXCommandBuilder
# (Speed/Quality/Speed-FP16 -> turbo): NAX verify kernels +22-40% chat decode
# on q8 (ULP-exact) and vk_k on q4, plus compiled verify behind its per-model
# quant-bits gate. Before 2026-07-05 the bare CLI (`mtplx serve` / quickstart
# / start) silently stayed on sustained, so every OpenAI-API consumer —
# including third-party benchmark harnesses — measured the slow path while
# the app ran turbo. Keep this list measured-win only: 6-bit small models
# (vk covers 4/8-bit affine only), 35B MoE (experts bypass the NAX patch),
# Gemma, and third-party artifacts keep the sustained default.
_TURBO_DEFAULT_PUBLIC_MODEL_IDS = frozenset(
    {
        # 27B Optimized Speed V2 (hybrid 4-bit). Named explicitly: this
        # entry used to ride on DEFAULT_PUBLIC_MODEL_ID, so the 2026-08-14
        # default flip to Qwen 3.8 would have silently demoted V2 to
        # sustained (the turbo-default-parity ledger class).
        OPTIMIZED_SPEED_V2_PUBLIC_MODEL_ID,
        OPTIMIZED_SPEED_V1_PUBLIC_MODEL_ID,  # original 27B Optimized Speed
        QUALITY_PUBLIC_MODEL_ID,  # 27B Optimized-Quality (8-bit)
        LEGACY_OPTIMIZED_PUBLIC_MODEL_ID,  # 27B Optimized (gdn8 hybrid, 8/4-bit)
        # 27B Speed-FP16 (INT4/g64 weights, fp16 activations — the M1/M2
        # routing target). Promoted 2026-07-07 after the first-ever e2e
        # measurement: turbo 1.98-2.08x over true AR on M5 Max (55-60 tok/s
        # D3 vs 27-29 AR), acceptance [86/66/51]%, peak 18.6 GB, vs only
        # 1.34x on sustained (fp16 verify_hidden_eval tax ~8s/384-tok run).
        # The 9B FP16 (6-bit) and 35B FP16 (MoE) siblings are NOT promoted.
        DEFAULT_FP16_PUBLIC_MODEL_ID,
        # 27B Quality-FP16 (q8/g64 weights, fp16 activations — the M1/M2
        # quality pick, built 2026-07-07). Measured on its own artifact:
        # turbo MTP D3 43.8/42.1 tok/s vs true AR 17.4 (2.5x), selfcheck
        # all vk q8 lanes ok, parent logit-diff argmax 16/16.
        QUALITY_FP16_PUBLIC_MODEL_ID,
        # 9B (6-bit/g64) promoted 2026-07-07 after the 6-bit hexpack
        # split-K kernels landed: live ABBA on the 9B measured turbo MTP
        # D3 110/102 tok/s vs sustained 90/69 (+34%), true AR 62-65 flat
        # both profiles -> multiplier 1.25x -> 1.68x. Exactness: 54
        # synthetic {bf16,fp16}x{gs32,64,128} cases dmax <= 0.027 vs
        # stock. The FP16 sibling shares the 6-bit packs and fp16 lanes
        # are exactness-proven, so both promote together.
        QWEN35_9B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
        QWEN35_9B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID,
        # Qwen3.8 27B family (2026-08-14). Trunk geometry is identical to the
        # Qwen3.6 27B flagships above, so the vk/NAX verify kernels and their
        # quant-bits gates carry over; the day-one A/B on the real artifacts
        # replaces this rationale with measured numbers before release.
        QWEN38_BARE_SPEED_PUBLIC_MODEL_ID,
        QWEN38_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
        QWEN38_OPTIMIZED_QUALITY_PUBLIC_MODEL_ID,
        # Qwen3.8 27B FP16 precision siblings (2026-08-15, the M1/M2 routing
        # targets): byte-identical quantized packs, bf16 -> fp16 for every
        # 16-bit tensor, so they ride the same fp16-templated vk/NAX lanes the
        # 3.6 FP16 siblings above already run under turbo. Promoted together
        # with their parents on the day-one FP16 parity campaign (ABBA parent
        # vs sibling on the turbo serve path, receipts in the release notes).
        QWEN38_BARE_SPEED_FP16_PUBLIC_MODEL_ID,
        QWEN38_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID,
        QWEN38_OPTIMIZED_QUALITY_FP16_PUBLIC_MODEL_ID,
    }
)


def _apply_model_default_profile(args: Any, model_id: str) -> bool:
    """Give the quantized 27B flagships the app's turbo default on the CLI.

    Returns True when the profile default was rewritten (caller must
    re-resolve its ``profile`` object). An explicit ``--profile`` flag or an
    interactive wizard choice always wins — both are recorded in
    ``args._cli_flags``.
    """

    cli_flags = getattr(args, "_cli_flags", set()) or set()
    if "profile" in cli_flags:
        return False
    if getattr(args, "_profile_from_config", None):
        # A profile from config.toml is the user's standing pin (stamped by
        # config._apply_profile_default). Honor it even when it equals the
        # parser default — config "sustained" used to be silently promoted.
        return False
    model_ref = getattr(args, "model", None)
    if (
        model_id not in _TURBO_DEFAULT_PUBLIC_MODEL_IDS
        and _artifact_recommended_profile(model_ref) != "turbo"
    ):
        return False
    current = str(getattr(args, "profile", None) or DEFAULT_PROFILE_NAME)
    if current != DEFAULT_PROFILE_NAME:
        # A non-default profile that arrived without a CLI flag came from a
        # programmatic caller (app command builder, tests); respect it.
        return False
    args.profile = "turbo"
    return True


def _resolved_default_profile_name(args: Any, model: str | None = None) -> str:
    """Profile name the launch will actually resolve — for display strings.

    Printed handoff/server commands used to bake the raw parser default
    ("--profile sustained") even though serve-time per-model resolution
    picks turbo for the quantized 27B flagships; users copying the printed
    command then pinned the slower profile explicitly (2026-07-16
    agent-lane TPS investigation). Mirrors _apply_model_default_profile
    without mutating args.
    """

    current = str(getattr(args, "profile", None) or DEFAULT_PROFILE_NAME)
    cli_flags = getattr(args, "_cli_flags", set()) or set()
    if (
        "profile" in cli_flags
        or current != DEFAULT_PROFILE_NAME
        or getattr(args, "_profile_from_config", None)
    ):
        return current
    model_ref = str(model if model is not None else getattr(args, "model", "") or "")
    if not model_ref:
        return current
    try:
        model_id = _public_model_id_for_args(args, model_ref)
    except Exception:
        return current
    if (
        model_id in _TURBO_DEFAULT_PUBLIC_MODEL_IDS
        or _artifact_recommended_profile(model_ref) == "turbo"
    ):
        return "turbo"
    return current


def resolved_default_profile_name_for_ref(model_ref: str | Path | None) -> str:
    """Default profile per-model launch resolution picks for an artifact ref.

    The args-free core of ``_resolved_default_profile_name`` for surfaces
    that report or stamp a profile for an artifact without a CLI namespace
    (cache listings, doctor's support matrix, forge runtime stamps). No
    user override can exist on those surfaces, so the answer is exactly
    the per-model turbo promotion over the served public id — the same
    ``public_model_id_for_ref`` mapping serve-time resolution uses.
    """

    if (
        public_model_id_for_ref(model_ref) in _TURBO_DEFAULT_PUBLIC_MODEL_IDS
        or _artifact_recommended_profile(model_ref) == "turbo"
    ):
        return "turbo"
    return DEFAULT_PROFILE_NAME


def _artifact_recommended_profile(model_ref: str | Path | None) -> str | None:
    """Read a local Forge artifact's measured launch profile, if present."""

    if model_ref is None:
        return None
    runtime, _ = _local_runtime_metadata(str(model_ref))
    if not isinstance(runtime, dict):
        return None
    profile = str(runtime.get("recommended_profile") or "").strip().lower()
    return profile if profile in {"stable", "sustained", "turbo"} else None


def _apply_qwen36_35b_optimized_speed_defaults(args: Any, model_id: str) -> None:
    # The -FP16 sibling shares the byte-identical INT packs and the measured
    # launch defaults; the app's substring detection already applied them to
    # it while this exact-id gate skipped it (2026-08-03 parity audit).
    if model_id not in {
        QWEN36_35B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
        QWEN36_35B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID,
    }:
        return
    cli_flags = getattr(args, "_cli_flags", set()) or set()
    injected = set(getattr(args, "_injected_default_flags", set()) or set())
    if "depth" not in cli_flags:
        args.depth = 1
    if "verify-strategy" not in cli_flags:
        args.verify_strategy = "target_prefix"
    if "draft-temperature" not in cli_flags:
        args.draft_temperature = 0.6
        injected.add("draft-temperature")
    if "draft-top-p" not in cli_flags:
        args.draft_top_p = 0.95
        injected.add("draft-top-p")
    if "draft-top-k" not in cli_flags:
        args.draft_top_k = 20
        injected.add("draft-top-k")
    # Measured product defaults must reach the daemon even when the model
    # contract carries no recommended_draft_sampler; record them so the
    # draft-sampler resolution treats them like requested values while
    # user-typed flags still win.
    args._injected_default_flags = injected
    if "chat-template-profile" not in cli_flags and getattr(
        args, "chat_template_profile", None
    ) in (None, "local_qwen36"):
        args.chat_template_profile = "local_qwen36"


def _inspection_backend_id(inspection: dict[str, Any]) -> str:
    compatibility = (
        inspection.get("compatibility")
        if isinstance(inspection.get("compatibility"), dict)
        else {}
    )
    return str(
        inspection.get("recommended_backend")
        or compatibility.get("recommended_backend")
        or ""
    )


def _inspection_is_gemma4_assistant(inspection: dict[str, Any]) -> bool:
    return _inspection_backend_id(inspection) == "gemma4_assistant"


def _gemma4_pair_sampler(inspection: dict[str, Any]) -> dict[str, Any]:
    sampler = inspection.get("recommended_sampler")
    if not isinstance(sampler, dict):
        pair = inspection.get("gemma4_pair")
        sampler = pair.get("sampler") if isinstance(pair, dict) else None
    if isinstance(sampler, dict):
        try:
            return {
                "temperature": float(sampler["temperature"]),
                "top_p": float(sampler["top_p"]),
                "top_k": int(sampler["top_k"]),
            }
        except (KeyError, TypeError, ValueError):
            pass
    return {"temperature": 1.0, "top_p": 0.95, "top_k": 64}


def _gemma4_pair_draft_block_size(inspection: dict[str, Any]) -> int:
    pair = inspection.get("gemma4_pair")
    benchmark = pair.get("benchmark") if isinstance(pair, dict) else None
    if isinstance(benchmark, dict):
        try:
            return max(2, min(8, int(benchmark["best_block_size"])))
        except (KeyError, TypeError, ValueError):
            pass
    return 4


def _apply_backend_serve_defaults(args: Any, inspection: dict[str, Any]) -> None:
    descriptor = descriptor_from_inspection(inspection)
    cli_flags = getattr(args, "_cli_flags", set()) or set()
    # Family-aware policy, not the raw lane descriptor: shared lanes (mlx_lm_ar)
    # pin parser=none while a family on that lane (lfm2) has a verified codec.
    # Stamping the lane's "none" here reads as an operator override downstream
    # and permanently disables reasoning for the child daemon.
    reasoning = reasoning_policy_for_model(
        inspection=inspection,
        descriptor=descriptor,
    )
    if "reasoning" not in cli_flags and getattr(args, "reasoning", None) is None:
        args.reasoning = reasoning.default_mode if reasoning.supported else "off"
    if "reasoning-parser" not in cli_flags and getattr(
        args, "reasoning_parser", None
    ) in (None, "qwen3"):
        args.reasoning_parser = reasoning.parser
    if (
        "reasoning-effort" not in cli_flags
        and getattr(args, "reasoning_effort", None) in (None, "auto")
        and reasoning.default_effort
    ):
        args.reasoning_effort = reasoning.default_effort
    required_tool_prompt_mode = descriptor.required_tool_prompt_mode
    if required_tool_prompt_mode is not None:
        requested_tool_prompt_mode = str(
            getattr(args, "tool_prompt_mode", required_tool_prompt_mode)
            or required_tool_prompt_mode
        )
        if (
            "tool-prompt-mode" in cli_flags
            and requested_tool_prompt_mode != required_tool_prompt_mode
        ):
            raise ValueError(
                f"{descriptor.display_name} requires --tool-prompt-mode "
                f"{required_tool_prompt_mode}"
            )
        args.tool_prompt_mode = required_tool_prompt_mode
    elif "tool-prompt-mode" not in cli_flags and not getattr(
        args, "tool_prompt_mode", None
    ):
        args.tool_prompt_mode = descriptor.default_tool_prompt_mode
    required_chat_template_profile = descriptor.required_chat_template_profile
    if required_chat_template_profile is not None:
        requested_profile = str(
            getattr(args, "chat_template_profile", required_chat_template_profile)
            or required_chat_template_profile
        )
        has_conflicting_profile = (
            "chat-template-profile" in cli_flags
            and requested_profile != required_chat_template_profile
        )
        has_custom_path = bool(getattr(args, "chat_template_path", None))
        if has_conflicting_profile or (
            has_custom_path and not descriptor.allows_chat_template_path
        ):
            raise ValueError(
                f"{descriptor.display_name} requires its tokenizer chat template"
            )
        args.chat_template_profile = required_chat_template_profile
        args.chat_template_path = None

    # Family-aware for the same reason as the reasoning policy above: the
    # qwen3_next lane serves qwen3_5/3_6 (0.6 coding sampler) and qwen3_8
    # (official 1.0 thinking sampler) alike.
    sampler = sampler_defaults_for_model(
        inspection=inspection,
        descriptor=descriptor,
    ).to_dict()
    draft_semantics = draft_semantics_for_model(
        inspection=inspection,
        descriptor=descriptor,
    )
    if descriptor.default_max_response_tokens is not None:
        if (
            "max-tokens" not in cli_flags
            and hasattr(args, "max_tokens")
            and getattr(args, "max_tokens", None) is None
        ):
            args.max_tokens = int(descriptor.default_max_response_tokens)
        if (
            "max-response-tokens" not in cli_flags
            and hasattr(args, "max_response_tokens")
            and getattr(args, "max_response_tokens", None) is None
        ):
            args.max_response_tokens = int(descriptor.default_max_response_tokens)
    # config.toml presence beats value-sentinels for the launch sampler trio:
    # apply_user_config stamps the parsed file onto ``args.mtplx_config``, so
    # a config value that happens to EQUAL the parser default (0.6/0.95/20)
    # is still the user's standing pin — the bare ``in (None, 0.6)`` checks
    # read it as "unset" and silently replaced it with the family sampler.
    # Injected family values are recorded in ``_injected_default_flags``
    # (the same provenance contract as the draft trio below).
    mtplx_config = getattr(args, "mtplx_config", None)
    config_pinned = {
        key
        for key in ("temperature", "top_p", "top_k")
        if isinstance(mtplx_config, dict)
        and mtplx_config.get(key) is not None
        and getattr(args, key, None) is not None
    }
    injected = set(getattr(args, "_injected_default_flags", set()) or set())
    if (
        "temperature" not in cli_flags
        and "default-temperature" not in cli_flags
        and "temperature" not in config_pinned
        and getattr(args, "temperature", None) in (None, 0.6)
    ):
        args.temperature = sampler["temperature"]
        injected.add("temperature")
    if (
        "top-p" not in cli_flags
        and "default-top-p" not in cli_flags
        and "top_p" not in config_pinned
        and getattr(args, "top_p", None) in (None, 0.95)
    ):
        args.top_p = sampler["top_p"]
        injected.add("top-p")
    if (
        "top-k" not in cli_flags
        and "top_k" not in config_pinned
        and getattr(args, "top_k", None) in (None, 20)
    ):
        args.top_k = sampler["top_k"]
        injected.add("top-k")
    if (
        "depth" not in cli_flags
        and draft_semantics.request_field == "depth"
        and getattr(args, "depth", None) in (None, 3)
    ):
        args.depth = draft_semantics.default
    # Injected-default provenance (same contract as
    # _apply_qwen36_35b_optimized_speed_defaults): the family draft values
    # below must reach the daemon as the launch draft sampler even when the
    # model contract carries no recommended_draft_sampler — and telemetry
    # must be able to tell an injected default from an artifact stamp.
    # _injected_default_flags feeds _explicit_draft_sampler_override, while
    # --draft-sampler-source stays "default" (user-typed _cli_flags only),
    # so injected values never pin and the family curve stays live.
    if "draft-temperature" not in cli_flags and getattr(
        args, "draft_temperature", None
    ) in (None, 0.6):
        # Qwen3.8 owns a measured family value even though it currently
        # matches the target sampler. Keeping one policy owner prevents the
        # app and CLI from drifting when later calibration changes it.
        family = model_family_from_inspection(
            inspection,
            descriptor=descriptor,
        )
        if family == "qwen3_8":
            args.draft_temperature = QWEN3_8_DRAFT_TEMPERATURE
        else:
            args.draft_temperature = sampler["temperature"]
        injected.add("draft-temperature")
    if "draft-top-p" not in cli_flags and getattr(args, "draft_top_p", None) is None:
        args.draft_top_p = sampler["top_p"]
        injected.add("draft-top-p")
    if "draft-top-k" not in cli_flags and getattr(args, "draft_top_k", None) in (
        None,
        20,
    ):
        args.draft_top_k = sampler["top_k"]
        injected.add("draft-top-k")
    args._injected_default_flags = injected
    if (
        "chat-template-profile" not in cli_flags
        and model_family_from_inspection(
            inspection,
            model_ref=str(getattr(args, "model", None) or ""),
            descriptor=descriptor,
        )
        not in {"qwen3_5", "qwen3_6"}
        and getattr(args, "chat_template_profile", None) == "local_qwen36"
    ):
        args.chat_template_profile = "tokenizer"

    if not descriptor.supports("native_adaptive_depth_policy"):
        if (
            "adaptive-policy" not in cli_flags
            and getattr(args, "adaptive_policy", None) == "expected_value"
        ):
            args.adaptive_policy = "none"

    if not _inspection_is_gemma4_assistant(inspection):
        return
    sampler = _gemma4_pair_sampler(inspection)
    draft_block_size = _gemma4_pair_draft_block_size(inspection)
    if getattr(args, "model_id", None) == DEFAULT_PUBLIC_MODEL_ID:
        args.model_id = "mtplx-gemma4-31b-assistant-mtp"
    if getattr(args, "temperature", None) in (None, 0.6):
        args.temperature = sampler["temperature"]
    if getattr(args, "top_p", None) is None:
        args.top_p = sampler["top_p"]
    if getattr(args, "top_k", None) in (None, 20):
        args.top_k = sampler["top_k"]
    if getattr(args, "depth", None) in (None, 3):
        args.depth = draft_block_size
    # Same injected-default provenance as the family block above: the
    # gemma4 pair draft values are launch defaults, not user pins.
    injected = set(getattr(args, "_injected_default_flags", set()) or set())
    if getattr(args, "draft_temperature", None) in (None, 0.6):
        args.draft_temperature = sampler["temperature"]
        injected.add("draft-temperature")
    if getattr(args, "draft_top_p", None) is None:
        args.draft_top_p = sampler["top_p"]
        injected.add("draft-top-p")
    if getattr(args, "draft_top_k", None) in (None, 20):
        args.draft_top_k = sampler["top_k"]
        injected.add("draft-top-k")
    args._injected_default_flags = injected
    if getattr(args, "chat_template_profile", None) == "local_qwen36":
        args.chat_template_profile = "tokenizer"
    if getattr(args, "adaptive_policy", None) == "expected_value":
        args.adaptive_policy = "none"


def _draft_sampler_from_spec(spec: dict[str, Any] | None) -> Any | None:
    if spec is None:
        return None
    from mtplx.sampling import SamplerConfig

    return SamplerConfig(
        temperature=float(spec["temperature"]),
        top_p=float(spec["top_p"]),
        top_k=int(spec["top_k"]),
    )


_DRAFT_SAMPLER_FLAG_ATTRS = {
    "draft-temperature": "draft_temperature",
    "draft-top-p": "draft_top_p",
    "draft-top-k": "draft_top_k",
}


def _explicit_draft_sampler_override(
    args: Any,
    base_sampler: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return a user-requested draft sampler override, not an internal default.

    Provenance order: user-typed CLI flags always win; defaults injected by
    `_apply_*_defaults` helpers (tracked via ``args._injected_default_flags``)
    only FILL THE GAP when no contract/profile spec exists, so the
    benchmarked launch configuration still reaches the daemon when the model
    contract carries no ``recommended_draft_sampler`` — and an injected
    generic default can never clobber an artifact's stamped value.
    """

    cli_flags = set(getattr(args, "_cli_flags", set()) or set())
    if base_sampler is None:
        cli_flags |= set(getattr(args, "_injected_default_flags", set()) or set())
    if not any(flag in cli_flags for flag in _DRAFT_SAMPLER_FLAG_ATTRS):
        return None
    base = base_sampler or {
        "temperature": getattr(args, "temperature", 0.6),
        "top_p": getattr(args, "top_p", 0.95),
        "top_k": getattr(args, "top_k", 20),
    }
    return {
        "temperature": (
            getattr(args, "draft_temperature", None)
            if "draft-temperature" in cli_flags
            and getattr(args, "draft_temperature", None) is not None
            else base["temperature"]
        ),
        "top_p": (
            getattr(args, "draft_top_p", None)
            if "draft-top-p" in cli_flags
            and getattr(args, "draft_top_p", None) is not None
            else base["top_p"]
        ),
        "top_k": (
            getattr(args, "draft_top_k", None)
            if "draft-top-k" in cli_flags
            and getattr(args, "draft_top_k", None) is not None
            else base["top_k"]
        ),
    }


def _resolve_model_context_window(tokenizer: Any, model_path: str | Path) -> int:
    candidates: list[int] = []
    tokenizer_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_max, int):
        candidates.append(tokenizer_max)

    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            config = {}
        for section in (config, config.get("text_config", {})):
            if not isinstance(section, dict):
                continue
            for key in (
                "max_position_embeddings",
                "model_max_length",
                "max_sequence_length",
                "seq_length",
                "context_length",
            ):
                value = section.get(key)
                if isinstance(value, int):
                    candidates.append(value)

    sane = [value for value in candidates if 0 < value <= 1_048_576]
    return max(sane) if sane else 262_144


def _cli_generation_budget(
    *,
    tokenizer: Any,
    model_path: str | Path,
    prompt_token_count: int,
    explicit_max_tokens: int | None,
) -> dict[str, int | bool | None]:
    context_window = _resolve_model_context_window(tokenizer, model_path)
    remaining_context = max(1, int(context_window) - max(0, int(prompt_token_count)))
    requested_max = remaining_context
    request_max_tokens = None
    if explicit_max_tokens is not None:
        request_max_tokens = int(explicit_max_tokens)
        if request_max_tokens < 1:
            raise ValueError(
                "--max-tokens must be >= 1; omit it to use the model context"
            )
        requested_max = request_max_tokens
    effective_max = max(1, min(requested_max, remaining_context))
    return {
        "request_max_tokens": request_max_tokens,
        "effective_max_tokens": int(effective_max),
        "context_window": int(context_window),
        "remaining_context_tokens": int(remaining_context),
        "context_cap_applied": bool(effective_max < requested_max),
    }


def _reasoning_mode(args: Any, *, default: str = "on") -> str:
    raw = getattr(args, "reasoning", None)
    mode = str(raw or default).strip().lower()
    if mode not in {"auto", "on", "off"}:
        return default
    return mode


def _preserve_thinking_policy(args: Any) -> str:
    if bool(getattr(args, "strip_assistant_reasoning_history", False)):
        return "off"
    raw = getattr(args, "preserve_thinking", None)
    mode = str(raw or "auto").strip().lower()
    return mode if mode in {"auto", "on", "off", "scoped"} else "auto"


def _apply_pi_history_budget_env_defaults(env: dict[str, str]) -> None:
    """Pi lane = the shared coding-agent engine block, nothing more.

    Mirrors the app's composition exactly: codingAgentRuntimeEnvironment,
    then a .pi case that sets no env overrides. The Pi compaction battery
    (compact threshold 1200 plus the line caps) is gone (#282): an explicit
    env re-arms its compactor past the engine's passthrough default, which
    silently rewrote Pi transcripts from both launchers.
    """
    _apply_opencode_memory_env_defaults(env)


def _enable_thinking_for_reasoning(mode: str) -> bool | None:
    if mode == "auto":
        return None
    return mode == "on"


def _redact_secret_value(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(
                marker in lowered
                for marker in (
                    "token",
                    "api_key",
                    "apikey",
                    "auth",
                    "secret",
                    "password",
                )
            ):
                redacted[str(key)] = "[redacted]"
            else:
                redacted[str(key)] = _redact_secret_value(item)
        return redacted
    if isinstance(value, list):
        return [_redact_secret_value(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_secret_value(item) for item in value]
    if isinstance(value, str) and len(value) > 12:
        lowered = value.lower()
        if any(
            marker in lowered
            for marker in ("hf_", "bearer ", "api-key", "password", "secret")
        ):
            return "[redacted]"
    if isinstance(value, str):
        # Support bundles are shared on GitHub issues; the user's home
        # directory is identifying. Keep the tail (still diagnostic),
        # drop the account name (QA-120).
        home = os.path.expanduser("~")
        if home and home != "~" and home in value:
            return value.replace(home, "~")
    return value


def _write_json_redacted(path: Path, value: Any) -> None:
    write_json(path, _redact_secret_value(value))


def _deep_doctor_report(args: Any, base: dict[str, Any]) -> dict[str, Any]:
    from mtplx.config import load_user_config

    config = load_user_config()
    default_model = config.model or DEFAULT_CHAMPION
    model_report: dict[str, Any]
    try:
        model_report = _compact_model_summary(inspect_model(default_model).to_dict())
    except Exception as exc:
        model_report = {
            "model": default_model,
            "ok": False,
            "error": str(exc),
        }
    server_deps = {
        name: importlib.util.find_spec(name) is not None
        for name in ("fastapi", "uvicorn")
    }
    launchers = {
        "path_lookup": shutil.which("mtplx"),
        "homebrew_bin": {
            "path": "/opt/homebrew/bin/mtplx",
            "exists": Path("/opt/homebrew/bin/mtplx").exists(),
        },
        "local_bin": {
            "path": str(Path.home() / ".local/bin/mtplx"),
            "exists": (Path.home() / ".local/bin/mtplx").exists(),
        },
    }
    base["deep"] = {
        "config": config.to_dict(),
        "default_model": model_report,
        "server_dependencies": server_deps,
        "launchers": launchers,
        "integrations": {
            "openwebui": "mtplx integrate openwebui --port 8000",
            "claude_code": "mtplx integrate claude-code --port 8000",
            "opencode": "mtplx start opencode --port 18083",
        },
        "product_policy": {
            "fanmax_counts_for_product_gate": False,
            "safe_mode_drops_per_cycle_events": os.environ.get("MTPLX_DROP_EVENTS")
            == "1",
        },
    }
    return base


def _doctor_model_id_candidates(value: str | None) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    raw_candidates = {
        text,
        text.replace("\\", "/").rsplit("/", 1)[-1],
        public_model_id_for_ref(text, default_model_id=DEFAULT_PUBLIC_MODEL_ID),
    }
    candidates: set[str] = set()
    for raw in raw_candidates:
        normalized = str(raw).strip().lower().replace("_", "-")
        normalized = normalized.replace("\\", "/").rsplit("/", 1)[-1]
        normalized = re.sub(r"[^a-z0-9.-]+", "-", normalized)
        normalized = re.sub(r"-{2,}", "-", normalized).strip("-.")
        if not normalized:
            continue
        candidates.add(normalized)
        if normalized.startswith("youssofal-") and "-mtplx-" in normalized:
            candidates.add(normalized.removeprefix("youssofal-"))
    return candidates


def _doctor_model_ids_match(left: str | None, right: str | None) -> bool:
    left_ids = _doctor_model_id_candidates(left)
    right_ids = _doctor_model_id_candidates(right)
    return bool(left_ids and right_ids and left_ids.intersection(right_ids))


def _opencode_doctor_report(args: Any) -> dict[str, Any]:
    from mtplx.opencode import (
        detect_opencode_desktop,
        opencode_config_path,
    )

    config_path = opencode_config_path()
    parsed: dict[str, Any] | None = None
    error: str | None = None
    if config_path.exists():
        try:
            value = json.loads(config_path.read_text(encoding="utf-8"))
            parsed = value if isinstance(value, dict) else {}
        except Exception as exc:
            error = str(exc)
    provider = (
        ((parsed or {}).get("provider") or {}).get("mtplx")
        if isinstance((parsed or {}).get("provider"), dict)
        else None
    )
    model_ref = (parsed or {}).get("model") if isinstance(parsed, dict) else None
    model_id = (
        str(model_ref or "").split("/", 1)[-1] if model_ref else DEFAULT_PUBLIC_MODEL_ID
    )
    configured_model_id = str(model_id) if model_ref else None
    model_config = None
    if isinstance(provider, dict):
        models = provider.get("models")
        if isinstance(models, dict):
            model_config = models.get(model_id) or next(iter(models.values()), None)
    options = provider.get("options") if isinstance(provider, dict) else {}
    base_url = ""
    api_key: str | None = None
    headers: dict[str, Any] = {}
    if isinstance(options, dict):
        base_url = str(options.get("baseURL") or "")
        raw_api_key = options.get("apiKey")
        if raw_api_key:
            api_key = str(raw_api_key)
        raw_headers = options.get("headers")
        if isinstance(raw_headers, dict):
            headers = raw_headers
    server_url = base_url.rstrip("/")
    if server_url.endswith("/v1"):
        server_url = server_url[:-3]
    health = (
        _http_json(server_url + "/health", timeout=1.5, api_key=api_key)
        if server_url
        else {}
    )
    live_model_id = None
    if isinstance(health, dict):
        live_model = (
            health.get("model")
            or health.get("model_id")
            or health.get("served_model_id")
        )
        if live_model:
            live_model_id = str(live_model).split("/", 1)[-1]
    model_matches_live_server = None
    stale_model_warning = None
    if configured_model_id and live_model_id:
        model_matches_live_server = _doctor_model_ids_match(
            configured_model_id,
            live_model_id,
        )
        if not model_matches_live_server:
            stale_model_warning = (
                "OpenCode config points at "
                f"{configured_model_id}, but the live MTPLX server reports "
                f"{live_model_id}. Launch OpenCode from the MTPLX app, or rerun "
                "`mtplx start opencode` for the intended model before judging "
                "client behavior."
            )
    plugin_setting = (parsed or {}).get("plugin") if isinstance(parsed, dict) else None
    if isinstance(plugin_setting, list):
        plugin_paths = [item for item in plugin_setting if isinstance(item, str)]
    elif isinstance(plugin_setting, str):
        plugin_paths = [plugin_setting]
    else:
        plugin_paths = []
    deprecated_plugin_configured = any(
        Path(path).name == "mtplx-session-headers.js" for path in plugin_paths
    )
    client_header_ready = headers.get("x-mtplx-client") == "opencode"
    return {
        "config_path": str(config_path),
        "config_exists": config_path.exists(),
        "config_error": error,
        "detected": detect_opencode_desktop(),
        "provider_present": isinstance(provider, dict),
        "model_ref": model_ref,
        "configured_model_id": configured_model_id,
        "base_url": base_url,
        "server_url": server_url,
        "server_health": health,
        "api_key_configured": bool(api_key),
        "live_model_id": live_model_id,
        "model_matches_live_server": model_matches_live_server,
        "stale_model_warning": stale_model_warning,
        "transport_headers": headers,
        "mtplx_client_header_configured": client_header_ready,
        "reasoning_field": None,
        "reasoning_enabled": (
            bool((model_config or {}).get("reasoning"))
            if isinstance(model_config, dict)
            else False
        ),
        "tool_call_enabled": (
            bool((model_config or {}).get("tool_call"))
            if isinstance(model_config, dict)
            else False
        ),
        "has_hidden_max_tokens": "maxTokens" in json.dumps(model_config or {}),
        "plugin_paths": plugin_paths,
        "deprecated_session_headers_plugin_configured": deprecated_plugin_configured,
        "session_headers_ready": False,
        "session_headers_status": "retired",
        "expected_start_command": "mtplx start opencode --port 18083 --max",
    }


def _doctor_port_from_base_url(base_url: str, args: Any) -> int:
    if base_url:
        try:
            port = urllib.parse.urlsplit(base_url).port
        except ValueError:
            port = None
        if port:
            return int(port)
    return int(getattr(args, "port", None) or 8000)


def _pi_doctor_report(args: Any) -> dict[str, Any]:
    from mtplx.pi import pi_models_json_path, pi_model_ref

    config_path = pi_models_json_path()
    parsed: dict[str, Any] | None = None
    error: str | None = None
    if config_path.exists():
        try:
            value = json.loads(config_path.read_text(encoding="utf-8"))
            parsed = value if isinstance(value, dict) else {}
        except Exception as exc:
            error = str(exc)
    providers = (parsed or {}).get("providers") if isinstance(parsed, dict) else None
    provider = providers.get("mtplx") if isinstance(providers, dict) else None
    models = provider.get("models") if isinstance(provider, dict) else None
    model_config = None
    if isinstance(models, list) and models:
        first = models[0]
        model_config = first if isinstance(first, dict) else None
    configured_model_id = (
        str(model_config.get("id"))
        if isinstance(model_config, dict) and model_config.get("id")
        else None
    )
    model_ref = pi_model_ref(configured_model_id) if configured_model_id else None
    base_url = str(provider.get("baseUrl") or "") if isinstance(provider, dict) else ""
    api_key = None
    headers: dict[str, Any] = {}
    auth_header = False
    if isinstance(provider, dict):
        raw_api_key = provider.get("apiKey")
        if raw_api_key:
            api_key = str(raw_api_key)
        auth_header = bool(provider.get("authHeader"))
        raw_headers = provider.get("headers")
        if isinstance(raw_headers, dict):
            headers = raw_headers
    server_url = base_url.rstrip("/")
    if server_url.endswith("/v1"):
        server_url = server_url[:-3]
    health = (
        _http_json(server_url + "/health", timeout=1.5, api_key=api_key)
        if server_url
        else {}
    )
    live_model_id = None
    if isinstance(health, dict):
        live_model = (
            health.get("model")
            or health.get("model_id")
            or health.get("served_model_id")
        )
        if live_model:
            live_model_id = str(live_model).split("/", 1)[-1]
    model_matches_live_server = None
    stale_model_warning = None
    if configured_model_id and live_model_id:
        model_matches_live_server = _doctor_model_ids_match(
            configured_model_id,
            live_model_id,
        )
        if not model_matches_live_server:
            stale_model_warning = (
                "Pi config points at "
                f"{configured_model_id}, but the live MTPLX server reports "
                f"{live_model_id}. Launch Pi from the MTPLX app, or rerun "
                "`mtplx start pi` for the intended model before judging "
                "client behavior."
            )
    return {
        "config_path": str(config_path),
        "config_exists": config_path.exists(),
        "config_error": error,
        "provider_present": isinstance(provider, dict),
        "model_ref": model_ref,
        "configured_model_id": configured_model_id,
        "base_url": base_url,
        "server_url": server_url,
        "server_health": health,
        "api_key_configured": bool(api_key),
        "auth_header": auth_header,
        "live_model_id": live_model_id,
        "model_matches_live_server": model_matches_live_server,
        "stale_model_warning": stale_model_warning,
        "transport_headers": headers,
        "mtplx_client_header_configured": headers.get("x-mtplx-client") == "pi",
        "reasoning_enabled": (
            bool(model_config.get("reasoning"))
            if isinstance(model_config, dict)
            else False
        ),
        # Presence is the healthy state for Pi: without advertised maxTokens
        # metadata Pi silently serializes a 16,384 output ceiling, and the
        # request-policy extension strips only the generated wire cap.
        "advertised_max_tokens": (
            model_config.get("maxTokens") if isinstance(model_config, dict) else None
        ),
        # The restart hint must name the port the config actually points at
        # (the app serves on 8002); 8000 is only the bare-CLI fallback.
        "expected_start_command": (
            f"mtplx start pi --port {_doctor_port_from_base_url(base_url, args)} --max"
        ),
    }


def _android_studio_doctor_report(args: Any) -> dict[str, Any]:
    host = str(getattr(args, "host", None) or "127.0.0.1")
    port = int(getattr(args, "port", None) or 8008)
    base_url = str(
        getattr(args, "base_url", None) or f"http://{host}:{port}/v1"
    ).rstrip("/")
    models = _http_json(f"{base_url}/models", timeout=3.0)
    model_id = DEFAULT_PUBLIC_MODEL_ID
    data = models.get("data") if isinstance(models, dict) else None
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and first.get("id"):
            model_id = str(first["id"])
    chat_payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "hi"}],
        "max_completion_tokens": 16,
    }
    stream_payload = {
        **chat_payload,
        "max_completion_tokens": 8,
        "enable_thinking": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    tool_payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Say OK. Do not call tools."}],
        "max_completion_tokens": 8,
        "enable_thinking": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "android_studio_check",
                    "description": "No-op compatibility check for Android Studio.",
                    "parameters": {
                        "type": "object",
                        "properties": {"ok": {"type": "string"}},
                        "required": ["ok"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "response_format": {"type": "text"},
    }
    return {
        "base_url": base_url,
        "paste_url": base_url,
        "url_schema": "OpenAI-compatible",
        "api_key": "blank for localhost unless MTPLX was started with --api-key",
        "model": model_id,
        "models": models,
        "chat_nonstream": _http_post_json(
            f"{base_url}/chat/completions", chat_payload, timeout=20.0
        ),
        "chat_stream": _http_post_text(
            f"{base_url}/chat/completions", stream_payload, timeout=20.0
        ),
        "tool_request": _http_post_json(
            f"{base_url}/chat/completions", tool_payload, timeout=30.0
        ),
        "expected_start_command": "mtplx start --port 8008",
    }


def _git_value(args: list[str], *, cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _resolve_runtime_model_path(
    model: str, *, cache_dir: str | None = None
) -> tuple[str, dict[str, Any] | None]:
    from mtplx.hf_loader import resolve_model_path

    try:
        return str(resolve_model_path(model, cache_dir=cache_dir)), None
    except Exception as exc:
        return model, {
            "error": "model is not available locally",
            "model": model,
            "detail": str(exc),
        }


def _exactness_profile_kwargs(args: Any) -> dict[str, Any]:
    return {
        "attention_impl": getattr(args, "exactness_attention_impl", "mlx_vector_paged"),
        "block_size": int(getattr(args, "exactness_block_size", 16)),
        "num_blocks": int(getattr(args, "exactness_num_blocks", 1024)),
        "partitioned": not bool(getattr(args, "exactness_no_partitioned", False)),
        "partition_threshold": int(
            getattr(args, "exactness_partition_threshold", 2048)
        ),
        "partition_size": int(getattr(args, "exactness_partition_size", 512)),
    }


def _exact_paged_env_from_args(args: Any) -> dict[str, str]:
    return exact_paged_attention_env(**_exactness_profile_kwargs(args))


def _depth_sweep_native60(
    *,
    model: str,
    prompt_suite: str,
    depths: str = "3",
    max_tokens: int,
    limit: int | None,
    seed: int,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    draft_lm_head: dict[str, Any] | None = None,
    draft_sampler: dict[str, Any] | None = None,
    base_hidden_variant: str | None = None,
    mtp_hidden_variant: str | None = None,
    concat_order: str | None = None,
    mtp_cache_policy: str = "persistent",
    mtp_history_policy: str = "committed",
    compare_ar: bool = False,
    ar_only: bool = False,
    gemma4_draft_block_size: int | None = None,
    runtime_env: dict[str, str] | None = None,
    draft_core: str = "stock",
) -> dict[str, Any]:
    from mtplx.benchmarks.runners.mtp_depth_sweep import run_mtp_depth_sweep

    previous = apply_profile_env("performance-cold")
    if runtime_env:
        os.environ.update({key: str(value) for key, value in runtime_env.items()})
    draft_lm_head = draft_lm_head or {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
    }
    # Serve-path memory discipline (#261, F7): the depth-sweep harness loads
    # the model in-process; pin the serve-path Metal allocator caps first.
    from mtplx.server.openai import apply_memory_caps_preflight

    memory_preflight = apply_memory_caps_preflight(
        entry="bench.depth_sweep",
        model=str(model),
    )
    try:
        result = run_mtp_depth_sweep(
            model,
            prompt_suite,
            depths=depths,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            max_tokens=max_tokens,
            seed=seed,
            limit=limit,
            enable_thinking=False,
            compare_ar=compare_ar,
            ar_only=ar_only,
            gemma4_draft_block_size=gemma4_draft_block_size,
            base_hidden_variant=base_hidden_variant,
            mtp_hidden_variant=mtp_hidden_variant,
            concat_order=concat_order,
            mtp_cache_policy=mtp_cache_policy,
            mtp_history_policy=mtp_history_policy,
            min_speculative_depth=1,
            verify_strategy="capture_commit",
            draft_core=str(draft_core or "stock"),
            verify_core="linear-gdn-from-conv-tape",
            draft_lm_head_bits=int(draft_lm_head["bits"]),
            draft_lm_head_group_size=int(draft_lm_head["group_size"]),
            draft_lm_head_mode=str(draft_lm_head["mode"]),
            draft_temperature=(
                None if draft_sampler is None else float(draft_sampler["temperature"])
            ),
            draft_top_p=None
            if draft_sampler is None
            else float(draft_sampler["top_p"]),
            draft_top_k=None if draft_sampler is None else int(draft_sampler["top_k"]),
        )
        if isinstance(result, dict):
            result.setdefault("memory_preflight", memory_preflight)
        return result
    finally:
        restore_profile_env(previous)


class _temporary_env:
    def __init__(self, updates: dict[str, str]) -> None:
        self.updates = updates
        self.previous: dict[str, str | None] = {}

    def __enter__(self) -> None:
        self.previous = {key: os.environ.get(key) for key in self.updates}
        os.environ.update(self.updates)

    def __exit__(self, *_exc: object) -> None:
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _compiled_verify_fence_report(args: Any) -> dict[str, Any]:
    """Static compiled-verify fence status for doctor (issue #255).

    The fence is the context ceiling of the compiled verify step: above it
    every verify call falls back to the eager path (graphbank
    ``_compiled_verify_max_context``; engine default 6144 tokens). The
    founder's public commitment on #255 is that doctor prints the live
    value. Resolution mirrors the launch: an operator env always beats the
    profile value (both envs are in profiles.py's passthrough set), else the
    profile the default launch resolves for this Mac's verified default
    model, else the engine default. Mode mapping mirrors
    ``graphbank.compiled_verify_mode``. Static env + profile state only —
    no GPU probing.
    """

    engine_default_max_context = 6144
    try:
        default_model = str(select_default_model().model)
    except Exception:
        default_model = None
    profile_name = DEFAULT_PROFILE_NAME
    if default_model:
        try:
            profile_name = _resolved_default_profile_name(args, default_model)
        except Exception:
            profile_name = DEFAULT_PROFILE_NAME
    try:
        profile_env = get_profile(profile_name).env_dict()
    except Exception:
        profile_env = {}

    def _resolve(name: str, fallback: str) -> tuple[str, str]:
        if name in os.environ:
            return str(os.environ[name]).strip(), f"{name} env"
        if name in profile_env:
            return str(profile_env[name]).strip(), f"{profile_name} profile"
        return fallback, "engine default"

    mode_raw, mode_source = _resolve("MTPLX_COMPILED_VERIFY", "")
    lowered = mode_raw.lower()
    if lowered in {"", "0", "false", "no", "off"}:
        mode = "off"
    elif lowered in {"parity", "parity2"}:
        mode = lowered
    else:
        mode = "on"
    raw_max, max_source = _resolve(
        "MTPLX_COMPILED_VERIFY_MAX_CONTEXT", str(engine_default_max_context)
    )
    try:
        max_context = max(0, int(raw_max))
    except (TypeError, ValueError):
        max_context = engine_default_max_context
        max_source = "engine default"
    return {
        "mode": mode,
        "mode_source": mode_source,
        "max_context_tokens": max_context,
        "max_context_source": max_source,
        # max_context == 0 disables the ceiling (experiments only).
        "fenced": bool(max_context),
        "resolved_default_profile": profile_name,
        "default_model": default_model,
        "above_fence_behavior": "eager verify per call",
    }


def cmd_doctor(args: Any) -> int:
    # --json promises machine-parseable stdout. Probes import third-party
    # packages whose lazy loaders print() import errors straight to stdout
    # (huggingface_hub does, and a split/partially broken install makes it
    # certain), which used to prefix the JSON document with prose and break
    # every consumer. Build the whole report with stdout routed to stderr,
    # then emit only the document.
    if getattr(args, "json", False):
        with contextlib.redirect_stdout(sys.stderr):
            report = _build_doctor_report(args)
        _print(report)
        return 0
    report = _build_doctor_report(args)
    return _render_doctor_report(args, report)


def _build_doctor_report(args: Any) -> dict[str, Any]:
    env = collect_environment(args.project_root).to_dict()
    from mtplx.hf_loader import hf_cache_report
    from mtplx.thermal import detect_thermal_control
    from mtplx.diagnostics import build_diagnostics_payload, write_doctor_bundle

    smc_path_raw = getattr(args, "smc_path", None) or shutil.which("smc") or ""
    sovereign_path_raw = (
        getattr(args, "sovereign_path", None) or shutil.which("sovereign") or ""
    )
    smc_path = Path(smc_path_raw) if smc_path_raw else None
    sovereign_path = Path(sovereign_path_raw) if sovereign_path_raw else None
    thermal_control = detect_thermal_control()
    server_deps = {
        name: importlib.util.find_spec(name) is not None
        for name in ("fastapi", "uvicorn")
    }

    report = {
        "environment": env,
        "huggingface": hf_cache_report(cache_dir=getattr(args, "model_cache", None)),
        "thermal_control": thermal_control,
        "tools": {
            "python": sys.executable,
            "powermetrics": shutil.which("powermetrics"),
            "sudo": shutil.which("sudo"),
            "smc_atlas": str(smc_path) if smc_path else None,
            "smc_atlas_exists": bool(smc_path and smc_path.exists()),
            "sovereign": str(sovereign_path) if sovereign_path else None,
            "sovereign_exists": bool(sovereign_path and sovereign_path.exists()),
        },
        "policy": {
            "fanmax_counts_for_product_gate": False,
            "benchmark_exactness_smoke_context": 2048,
        },
        "compiled_verify": _compiled_verify_fence_report(args),
    }
    cli_flags = getattr(args, "_cli_flags", set()) or set()
    report["diagnostics"] = build_diagnostics_payload(
        model_cache=getattr(args, "model_cache", None),
        include_startup_default_model="model-cache" not in cli_flags,
        deep=bool(getattr(args, "deep", False)),
        # An explicit --port aims the server checks; the bare default (8008)
        # exists for the topic bridges and must not move the server probe off
        # the shipped :8000 default.
        server_port=(int(getattr(args, "port", 8000)) if "port" in cli_flags else 8000),
        mlx_info=env.get("mlx") if isinstance(env.get("mlx"), dict) else None,
        thermal_control=thermal_control,
        server_dependencies=server_deps if getattr(args, "deep", False) else None,
    )
    if getattr(args, "topic", None) == "opencode":
        report["opencode"] = _opencode_doctor_report(args)
    if getattr(args, "topic", None) == "pi":
        report["pi"] = _pi_doctor_report(args)
    if getattr(args, "topic", None) in {"android-studio", "android_studio"}:
        report["android_studio"] = _android_studio_doctor_report(args)
    if getattr(args, "deep", False):
        report = _deep_doctor_report(args, report)
    if getattr(args, "bundle", False):
        report["bundle"] = write_doctor_bundle(
            report=report,
            output_dir=getattr(args, "output_dir", None),
            include_paths=bool(getattr(args, "include_paths", False)),
        )
    return report


def _render_doctor_report(args: Any, report: dict[str, Any]) -> int:
    if getattr(args, "summary", False):
        diagnostics = report["diagnostics"]
        print(f"MTPLX doctor: {diagnostics['overall']}")
        for check in diagnostics["checks"]:
            marker = check["status"].upper()
            print(f"{marker:4} {check['id']}: {check['observed']}")
            if check.get("fix") and check["status"] != "pass":
                print(f"     fix: {check['fix']}")
        # The compiled-verify fence is the answer to "why does long context
        # feel different" (#255) — the summary view must carry it too, not
        # only the full report (F15).
        fence = report.get("compiled_verify") or {}
        if fence.get("fenced"):
            print(
                "compiled verify fence: "
                f"<= {fence.get('max_context_tokens')} tokens "
                f"({fence.get('max_context_source')})"
            )
        if report.get("bundle"):
            print(f"bundle: {report['bundle']['bundle_dir']}")
            print(f"zip: {report['bundle']['bundle_zip']}")
    else:
        env_info = report.get("environment") or {}
        hf = report.get("huggingface") or {}
        thermal = report.get("thermal_control") or {}
        selected = thermal.get("selected") or {}
        tools = report.get("tools") or {}
        print("MTPLX status")
        print(f"python: {tools.get('python') or sys.executable}")
        print(
            f"platform: {env_info.get('platform') or env_info.get('system') or 'unknown'}"
        )
        print(f"project: {env_info.get('project_root') or os.getcwd()}")
        print(f"model cache: {hf.get('cache_dir') or 'default'}")
        print(f"cached models: {hf.get('cached_models', 'unknown')}")
        print(
            "thermal: "
            f"{'available' if thermal.get('available') else 'not configured'}"
            f" ({selected.get('kind') or 'none'})"
        )
        fence = report.get("compiled_verify") or {}
        if fence:
            print(
                "compiled verify: "
                f"{fence.get('mode')} ({fence.get('mode_source')})"
            )
            if fence.get("fenced"):
                print(
                    "compiled verify fence: "
                    f"<= {fence.get('max_context_tokens')} tokens "
                    f"({fence.get('max_context_source')}); above it each "
                    "verify call falls back to eager"
                )
            else:
                print(
                    "compiled verify fence: disabled "
                    f"({fence.get('max_context_source')}); no context ceiling"
                )
        if getattr(args, "deep", False):
            launchers = report.get("launchers") or {}
            config = report.get("config") or {}
            print(
                f"launcher: {launchers.get('global_launcher') or launchers.get('global') or 'unknown'}"
            )
            print(f"config: {config.get('path') or 'default'}")
        if report.get("opencode"):
            opencode = report["opencode"]
            print("OpenCode:")
            print(f"  config: {opencode.get('config_path')}")
            print(
                f"  provider: {'present' if opencode.get('provider_present') else 'missing'}"
            )
            print(f"  model: {opencode.get('model_ref') or 'missing'}")
            if opencode.get("model_matches_live_server") is True:
                print("  model sync: ok")
            elif opencode.get("model_matches_live_server") is False:
                print("  model sync: stale")
            else:
                print("  model sync: unknown")
            if opencode.get("live_model_id"):
                print(f"  live model: {opencode.get('live_model_id')}")
            print(f"  base URL: {opencode.get('base_url') or 'missing'}")
            print(f"  reasoning field: {opencode.get('reasoning_field') or 'missing'}")
            print(
                f"  hidden maxTokens: {str(bool(opencode.get('has_hidden_max_tokens'))).lower()}"
            )
            print(
                "  MTPLX client header: "
                + (
                    "ready"
                    if opencode.get("mtplx_client_header_configured")
                    else "missing"
                )
            )
            if opencode.get("stale_model_warning"):
                print(f"  warning: {opencode.get('stale_model_warning')}")
        if report.get("pi"):
            pi = report["pi"]
            print("Pi:")
            print(f"  config: {pi.get('config_path')}")
            print(
                f"  provider: {'present' if pi.get('provider_present') else 'missing'}"
            )
            print(f"  model: {pi.get('model_ref') or 'missing'}")
            if pi.get("model_matches_live_server") is True:
                print("  model sync: ok")
            elif pi.get("model_matches_live_server") is False:
                print("  model sync: stale")
            else:
                print("  model sync: unknown")
            if pi.get("live_model_id"):
                print(f"  live model: {pi.get('live_model_id')}")
            print(f"  base URL: {pi.get('base_url') or 'missing'}")
            print(f"  auth header: {str(bool(pi.get('auth_header'))).lower()}")
            advertised_max_tokens = pi.get("advertised_max_tokens")
            print(
                "  advertised maxTokens: "
                + (
                    str(advertised_max_tokens)
                    if advertised_max_tokens
                    else "missing (Pi will inject a 16,384 output ceiling)"
                )
            )
            print(
                "  MTPLX client header: "
                + ("ready" if pi.get("mtplx_client_header_configured") else "missing")
            )
            if pi.get("stale_model_warning"):
                print(f"  warning: {pi.get('stale_model_warning')}")
        if report.get("android_studio"):
            android = report["android_studio"]
            print("Android Studio:")
            print(f"  URL: {android.get('paste_url')}")
            print(f"  URL schema: {android.get('url_schema')}")
            print(f"  model: {android.get('model')}")
            print(
                f"  /v1/models: {'ok' if android.get('models', {}).get('data') else 'check failed'}"
            )
            print(
                "  /v1/chat/completions: "
                f"{'ok' if android.get('chat_nonstream', {}).get('ok') else 'check failed'}"
            )
    return 0


def cmd_inspect_model_public(args: Any) -> int:
    model_args = list(getattr(args, "model_args", []) or [])
    if model_args and model_args[0] == "model":
        model_args = model_args[1:]
    model = (
        args.model
        or getattr(args, "model_arg", None)
        or (model_args[0] if model_args else None)
    )
    if len(model_args) > 1:
        raise SystemExit("inspect accepts exactly one model path/repo id")
    if not model:
        raise SystemExit("inspect requires MODEL or --model MODEL")
    try:
        inspection = inspect_model(model).to_dict()
    except Exception as exc:
        _print({"error": "inspect failed", "model": model, "detail": str(exc)})
        return 1
    if getattr(args, "json", False):
        _print(inspection)
    else:
        _print_inspect_human(inspection)
    compatibility = inspection.get("compatibility") or {}
    exit_code = int(compatibility.get("exit_code", 0))
    if args.require_mtp or getattr(args, "strict_exit_code", True):
        return exit_code
    return 0


def _print_inspect_human(inspection: dict[str, Any]) -> None:
    compatibility = inspection.get("compatibility") or {}
    mtp = inspection.get("mtp") or {}
    architecture = (
        inspection.get("architecture") or inspection.get("model_type") or "unknown"
    )
    tensor_count = mtp.get("tensor_count")
    mtp_layers = inspection.get("mtp_num_hidden_layers")
    runtime_contract_present = bool(
        inspection.get("runtime_contract_path")
        or compatibility.get("runtime_contract_path")
    )
    print("MTPLX inspect")
    print(f"model: {inspection.get('model_dir')}")
    print(f"source: {inspection.get('source') or 'local'}")
    print(f"architecture: {architecture}")
    print(f"tier: {compatibility.get('tier', 'unknown')}")
    print(f"recognized: {str(bool(compatibility.get('recognized'))).lower()}")
    print(f"can_run: {str(bool(compatibility.get('can_run'))).lower()}")
    runtime_status = compatibility.get("runtime_compatibility")
    if runtime_status:
        print(f"runtime_compatibility: {runtime_status}")
    support_level = compatibility.get("support_level")
    if support_level:
        print(f"support_level: {support_level}")
    if mtp_layers is not None:
        print(f"mtp_layers: {mtp_layers}")
    if tensor_count is not None:
        print(f"mtp_tensors_present: {tensor_count}")
    print(f"runtime_contract: {str(runtime_contract_present).lower()}")
    recommended = compatibility.get("recommended_profile") or inspection.get(
        "recommended_profile"
    )
    if recommended:
        print(f"recommended_profile: {recommended}")
    message = compatibility.get("message")
    if message:
        print(f"message: {message}")


def cmd_stop_public(args: Any) -> int:
    """Stop a running MTPLX server via its health-reported pid."""

    from mtplx.daemon_client import (
        probe_running_daemons,
        stop_daemon,
    )

    host = str(getattr(args, "host", "127.0.0.1"))
    port = getattr(args, "port", None)
    json_output = bool(getattr(args, "json", False))

    def emit(payload: dict[str, Any], *, lines: list[str]) -> None:
        if json_output:
            _print(payload)
        else:
            for line in lines:
                print(line)

    if port is None:
        # The app persists its (possibly port-preflight-bumped) port in
        # its settings; without this a bumped daemon is invisible to
        # `mtplx stop` and the user is told nothing is running (QA-121).
        from mtplx.daemon_client import default_probe_ports

        probe_ports = default_probe_ports()
        daemons = probe_running_daemons(host=host, ports=probe_ports)
        if not daemons:
            emit(
                {"ok": False, "reason": "no_server"},
                lines=[
                    "No running MTPLX server found on ports "
                    + ", ".join(str(p) for p in probe_ports)
                    + ".",
                ],
            )
            return 1
        if len(daemons) > 1:
            lines = ["Multiple MTPLX servers are running:"]
            for daemon in daemons:
                lines.append(
                    f"  port {daemon.port}  ·  model {daemon.model or '?'}"
                    f"  ·  started by {daemon.owner_label}"
                )
            lines.append("Pick one with: mtplx stop --port <port>")
            emit(
                {
                    "ok": False,
                    "reason": "multiple_servers",
                    "ports": [daemon.port for daemon in daemons],
                },
                lines=lines,
            )
            return 2
        port = daemons[0].port

    grace_s = float(getattr(args, "grace_seconds", 10.0))
    result = stop_daemon(host, int(port), grace_s=grace_s)
    if result.get("ok"):
        emit(
            result,
            lines=[
                f"Stopped the MTPLX server on port {port} "
                f"(pid {result.get('pid')}, {result.get('signal')})."
            ],
        )
        return 0
    reason = str(result.get("reason") or "unknown")
    reason_lines = {
        "no_server": [f"No MTPLX server is listening on port {port}."],
        "not_mtplx": [
            f"Port {port} is in use, but not by an MTPLX server. Not touching it."
        ],
        "no_pid": [
            f"The MTPLX server on port {port} does not report a pid; "
            "it is likely an older version. Stop it from the terminal "
            "that started it (Ctrl-C)."
        ],
        "permission_denied": [
            f"No permission to stop the MTPLX server on port {port} "
            f"(pid {result.get('pid')}). It belongs to another user."
        ],
    }
    emit(result, lines=reason_lines.get(reason, [f"Could not stop: {reason}"]))
    return 1


def _parse_settings_pairs(pairs: list[str]) -> tuple[dict[str, Any], list[str]]:
    """Parse ``key=value`` pairs; values decode as JSON with string fallback."""

    parsed: dict[str, Any] = {}
    errors: list[str] = []
    for pair in pairs:
        key, separator, raw_value = str(pair).partition("=")
        key = key.strip()
        if not separator or not key:
            errors.append(pair)
            continue
        value_text = raw_value.strip()
        try:
            parsed[key] = json.loads(value_text)
        except json.JSONDecodeError:
            parsed[key] = value_text
    return parsed, errors


def cmd_settings_public(args: Any) -> int:
    """Read or change live server settings over /v1/mtplx/settings."""

    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 8000))
    base = _server_url(host, port)
    json_output = bool(getattr(args, "json", False))
    action = str(getattr(args, "settings_action", None) or "get")
    pairs = list(getattr(args, "pairs", None) or [])
    if action == "get" and pairs:
        # `mtplx settings depth=2` reads as intent to set.
        action = "set"

    def fail_unreachable() -> int:
        print(f"No MTPLX server is responding on {base}.")
        print("Start one with the MTPLX app or: mtplx start")
        return 1

    if action == "get":
        payload = _http_json(base + "/v1/mtplx/settings", timeout=5.0)
        if not payload.get("ok"):
            return fail_unreachable()
        if json_output:
            _print(payload)
            return 0
        print(f"MTPLX server settings  ·  {base}")
        for key in sorted(payload):
            if key in {"ok"}:
                continue
            print(f"  {key} = {json.dumps(payload[key], default=str)}")
        return 0

    update, malformed = _parse_settings_pairs(pairs)
    if malformed or not update:
        for pair in malformed:
            print(f"error: not a key=value pair: {pair!r}")
        if not update:
            print("usage: mtplx settings set key=value [key=value ...]")
            print("example: mtplx settings set depth=2 reasoning=off")
        return 2
    response = _http_post_json(base + "/v1/mtplx/settings", update, timeout=10.0)
    if response.get("ok"):
        body = response.get("json") or {}
        applied = body.get("applied") or {}
        if json_output:
            _print(body)
            return 0
        if applied:
            for key in sorted(applied):
                print(f"applied: {key} = {json.dumps(applied[key], default=str)}")
        else:
            print("nothing to apply")
        return 0
    error = response.get("error")
    if isinstance(error, dict) and isinstance(error.get("error"), dict):
        # _http_post_json stores the whole response body; the daemon
        # wraps errors in the OpenAI envelope {"error": {...}} with the
        # structured detail inside it (QA-105).
        error = error["error"]
    detail = error.get("detail") if isinstance(error, dict) else None
    if isinstance(detail, dict):
        kind = detail.get("error")
        keys = detail.get("keys") or []
        if kind == "restart_required":
            print(
                "error: these settings need a server restart: "
                + ", ".join(str(key) for key in keys)
            )
            print(
                "Change them in the MTPLX app's settings, or restart "
                "`mtplx serve` with the matching flags."
            )
            return 2
        if kind == "unknown_settings":
            print("error: unknown settings: " + ", ".join(str(key) for key in keys))
            supported = detail.get("supported") or []
            if supported:
                print("supported: " + ", ".join(str(key) for key in supported))
            return 2
    if isinstance(detail, str) and detail:
        print(f"error: {detail}")
        return 2
    if response.get("status") is None:
        return fail_unreachable()
    print(f"error: settings update failed ({response.get('status')})")
    return 1


def _format_aime_question_line(event: dict[str, Any]) -> str:
    idx = int(event.get("idx") or 0)
    status = str(event.get("status") or "?")
    extracted = event.get("extracted")
    expected = event.get("expected")
    duration_ms = event.get("duration_ms")
    tokens = int(event.get("reasoning_token_count") or 0) + int(
        event.get("answer_token_count") or 0
    )
    parts = [f"Q{idx:>2}", f"{status:<9}"]
    parts.append(f"answer={extracted if extracted is not None else '—'}")
    if status != "correct" and expected is not None:
        parts.append(f"expected={expected}")
    if isinstance(duration_ms, (int, float)) and duration_ms > 0:
        seconds = float(duration_ms) / 1000.0
        parts.append(f"{seconds:6.1f}s")
        if tokens:
            parts.append(f"{tokens / seconds:6.1f} tok/s")
    return "  ".join(parts)


def _format_aime_grid(per_question: list[dict[str, Any]]) -> list[str]:
    marks: dict[int, str] = {}
    for row in per_question:
        idx = int(row.get("idx") or 0)
        status = str(row.get("status") or "")
        marks[idx] = "✓" if status == "correct" else "·" if status == "skipped" else "✗"
    if not marks:
        return []
    highest = max(marks)
    lines: list[str] = []
    for start in range(1, highest + 1, 10):
        cells = [marks.get(i, " ") for i in range(start, min(start + 10, highest + 1))]
        lines.append(f"  Q{start:>2}-{min(start + 9, highest):<2}  " + " ".join(cells))
    return lines


def _print_aime_summary(summary: dict[str, Any]) -> None:
    score = summary.get("score")
    total = summary.get("total")
    accuracy = summary.get("accuracy")
    duration_ms = summary.get("duration_ms")
    print()
    for line in _format_aime_grid(list(summary.get("per_question") or [])):
        print(line)
    print()
    headline = f"AIME {summary.get('state') or 'done'}: {score}/{total}"
    if isinstance(accuracy, (int, float)):
        headline += f"  ({float(accuracy) * 100:.1f}%)"
    if isinstance(duration_ms, (int, float)) and duration_ms > 0:
        headline += f"  in {float(duration_ms) / 60000.0:.1f} min"
    print(headline)
    if summary.get("model"):
        print(f"model: {summary.get('model')}")


def _cmd_bench_aime(args: Any) -> int:
    """Run the app's AIME benchmark from the terminal over the daemon API."""

    cli_flags = getattr(args, "_cli_flags", set()) or set()
    if "port" in cli_flags:
        base = _server_url("127.0.0.1", int(getattr(args, "port", 8000)))
    else:
        base = str(getattr(args, "url", "http://127.0.0.1:8000")).rstrip("/")
    health = _http_json(base + "/health", timeout=3.0)
    if not health.get("ok"):
        print(f"No MTPLX server is responding on {base}.")
        print("Start one with the MTPLX app or: mtplx start")
        return 1
    print(f"MTPLX AIME benchmark  ·  {base}  ·  model {health.get('model')}")

    active = _http_json(base + "/v1/mtplx/benchmarks/aime/active", timeout=3.0)
    run_id = active.get("active_run_id")
    if run_id:
        print(f"Attaching to the AIME run already in progress ({run_id}).")
    else:
        body: dict[str, Any] = {}
        if bool(getattr(args, "quick", False)):
            body["question_limit"] = 5
        started = _http_post_json(
            base + "/v1/mtplx/benchmarks/aime/start", body, timeout=30.0
        )
        if not started.get("ok"):
            error = started.get("error")
            if isinstance(error, dict) and error.get("error") == "run_in_progress":
                run_id = error.get("active_run_id")
            else:
                detail = (
                    error.get("detail")
                    if isinstance(error, dict)
                    else error or started.get("status")
                )
                print(f"error: could not start the AIME run: {detail}")
                return 1
        else:
            payload = started.get("json") or {}
            run_id = payload.get("run_id")
            total = payload.get("total")
            print(f"run {run_id}  ·  {total} questions  ·  Ctrl-C cancels")
    if not run_id:
        print("error: no run id returned by the server")
        return 1

    stream_url = f"{base}/v1/mtplx/benchmarks/aime/{run_id}/stream"
    request = urllib.request.Request(stream_url)
    summary: dict[str, Any] | None = None
    try:
        with urllib.request.urlopen(request, timeout=None) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                kind = str(event.get("event") or "")
                if kind == "question_started":
                    idx = int(event.get("idx") or 0)
                    attempt = int(event.get("attempt") or 1)
                    suffix = f" (attempt {attempt})" if attempt > 1 else ""
                    print(f"Q{idx:>2} running{suffix}...", flush=True)
                elif kind == "question_done":
                    print(_format_aime_question_line(event), flush=True)
                elif kind in {"run_done", "run_cancelled"}:
                    summary = event
                    break
                elif kind == "error":
                    print(f"error: {event.get('message') or event}")
                    return 1
    except KeyboardInterrupt:
        print()
        print("Cancelling the run...")
        cancelled = _http_post_json(
            base + f"/v1/mtplx/benchmarks/aime/{run_id}/cancel", {}, timeout=10.0
        )
        if cancelled.get("ok"):
            summary = cancelled.get("json") or {}
        else:
            print("error: could not cancel; the run may still be active")
            return 130
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"error: lost connection to the server: {exc}")
        return 1
    if summary is not None:
        _print_aime_summary(summary)
        if getattr(args, "json", False):
            _print(summary)
    return 0


def cmd_bench_public(args: Any) -> int:
    action = args.bench_action
    if action == "aime":
        return _cmd_bench_aime(args)
    if action == "prefill-ladder":
        from mtplx.prefill_bench import (
            UnsafePrefillDiagnosticError,
            emit_prefill_ladder,
            run_prefill_ladder,
            write_prefill_ladder,
        )

        try:
            payload = run_prefill_ladder(args)
        except UnsafePrefillDiagnosticError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if getattr(args, "output", None):
            write_prefill_ladder(args.output, payload)
        emit_prefill_ladder(payload, json_output=bool(getattr(args, "json", False)))
        return 0
    if action in {"run", "context"}:
        return _cmd_bench_run(args)
    if action == "tune":
        return _cmd_bench_tune(args)
    if action in {"nightly", "suite"}:
        return _cmd_bench_nightly(args)
    if action == "compare":
        return _cmd_bench_compare(args)
    if action == "serve":
        return _cmd_bench_serve(args)
    if action == "reference":
        return _cmd_bench_reference(args)
    if action == "reference-vllm":
        return _cmd_bench_reference_vllm(args)
    raise SystemExit(f"unknown bench action: {action}")


def _tune_requested_model(args: Any) -> str:
    cli_flags = getattr(args, "_cli_flags", set()) or set()
    if "model" not in cli_flags:
        configured_model = (getattr(args, "mtplx_config", None) or {}).get("model")
        current_model = getattr(args, "model", None)
        if configured_model and str(current_model) == str(configured_model):
            return str(configured_model)
        try:
            selection = select_default_model()
            model = getattr(selection, "model", None) or getattr(
                selection, "hf_model", None
            )
            if model:
                return str(model)
        except Exception:
            pass
    return str(getattr(args, "model", None) or DEFAULT_CHAMPION)


def _descriptor_for_tune_family(
    family: str,
    inspection: dict[str, Any] | None,
) -> Any:
    if inspection:
        return descriptor_from_inspection(inspection)
    if family == "gemma4":
        return descriptor_for_backend_id("gemma4_assistant")
    if family == "step":
        return descriptor_for_backend_id("step3p5_mtp")
    if family == "glm":
        return descriptor_for_backend_id("glm_mtp")
    if family == "deepseek":
        return descriptor_for_backend_id("deepseek_mtp")
    return descriptor_for_backend_id("qwen3_next")


def _local_runtime_metadata(model: str) -> tuple[dict[str, Any] | None, Path | None]:
    path = Path(str(model)).expanduser()
    if not path.is_dir():
        return None, None
    runtime_path = path / "mtplx_runtime.json"
    if not runtime_path.is_file():
        return None, None
    try:
        data = json.loads(runtime_path.read_text(encoding="utf-8"))
    except Exception:
        return None, runtime_path
    return data if isinstance(data, dict) else None, runtime_path


def _identity_text_parts(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for key, item in value.items():
            out.append(str(key))
            out.extend(_identity_text_parts(item))
        return out
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            out.extend(_identity_text_parts(item))
        return out
    if value is None:
        return []
    return [str(value)]


def _mtplx_tune_family_from_text(text: str) -> str | None:
    if any(marker in text for marker in ("qwen3.8", "qwen3_8", "qwen38")):
        return "qwen3_8"
    if any(marker in text for marker in ("qwen3.6", "qwen3_6", "qwen3-6", "qwen36")):
        return "qwen3_6"
    if any(marker in text for marker in ("qwen3.5", "qwen3_5", "qwen3-5", "qwen35")):
        return "qwen3_5"
    if "gemma4" in text or "gemma-4" in text or "gemma 4" in text:
        return "gemma4"
    if "step3p5" in text or "step3p7" in text or "step-3.7" in text:
        return "step"
    if "deepseek" in text:
        return "deepseek"
    if "glm" in text:
        return "glm"
    return None


def _fast_mtplx_tune_inspection(model: str) -> dict[str, Any] | None:
    runtime, runtime_path = _local_runtime_metadata(model)
    parts = [str(model)]
    parts.extend(_identity_text_parts(runtime))
    text = " ".join(parts).lower()
    if "mtplx" not in text:
        return None

    arch_id = str((runtime or {}).get("arch_id") or "").strip() or None
    descriptor = descriptor_for_architecture_id(arch_id) if arch_id else None
    family = _mtplx_tune_family_from_text(text)
    if family is None and descriptor is not None:
        if descriptor.model_family == "qwen":
            family = "qwen3_6"
        elif descriptor.model_family:
            family = descriptor.model_family
    if family is None:
        return None

    descriptor = descriptor or _descriptor_for_tune_family(family, None)
    runtime_contract = dict(runtime) if isinstance(runtime, dict) else None
    compatibility = {
        "tier": "verified",
        "can_run": True,
        "supported": True,
        "recognized": True,
        "exit_code": 0,
        "arch_id": arch_id or descriptor.architecture_id,
        "recommended_backend": descriptor.backend_id,
        "recommended_profile": (
            (runtime or {}).get("recommended_profile") or DEFAULT_PROFILE_NAME
        ),
        "runtime_contract": runtime_contract,
        "runtime_contract_path": str(runtime_path)
        if runtime_path is not None
        else None,
        "runtime_compatibility": "native",
        "support_level": "mtplx-fast-tune",
    }
    model_path = Path(str(model)).expanduser()
    return {
        "source": "mtplx-fast-tune",
        "model_dir": str(model_path) if model_path.is_dir() else str(model),
        "runtime_model": str(model),
        "architecture": arch_id or descriptor.architecture_id,
        "model_type": family,
        "mtp_arch": arch_id or descriptor.architecture_id,
        "recommended_backend": descriptor.backend_id,
        "recommended_profile": compatibility["recommended_profile"],
        "runtime_contract": runtime_contract,
        "runtime_contract_path": str(runtime_path)
        if runtime_path is not None
        else None,
        "compatibility": compatibility,
    }


def _tune_support_payload(
    model: str,
    *,
    inspect_local: bool = False,
) -> dict[str, Any]:
    inspection: dict[str, Any] | None = _fast_mtplx_tune_inspection(str(model))
    if inspect_local or Path(str(model)).expanduser().exists():
        if inspection is None:
            try:
                inspection = inspect_model(str(model)).to_dict()
            except Exception:
                inspection = None
    family = model_family_from_inspection(inspection, model_ref=str(model))
    descriptor = _descriptor_for_tune_family(family, inspection)
    policy = tune_policy_for_model(str(model), inspection, descriptor)
    controls = model_controls_for_descriptor(
        descriptor,
        model_ref=str(model),
        inspection=inspection,
    )
    unsupported_reason = policy.unsupported_reason or (
        "Tune is supported for Qwen 3.5, Qwen 3.6, Qwen 3.8, and Gemma 4 MTPLX models only."
    )
    return {
        "ok": bool(policy.supported),
        "model": str(model),
        "model_family": controls["model_family"],
        "backend_id": controls["backend_id"],
        "architecture_id": controls["architecture_id"],
        "tune_supported": bool(policy.supported),
        "supported_families": list(policy.supported_families),
        "unsupported_reason": None if policy.supported else unsupported_reason,
        "model_controls": controls,
    }


def _unsupported_tune_model_error(
    payload: dict[str, Any],
    *,
    json_output: bool,
) -> int:
    message = (
        "Tune is supported for Qwen 3.5, Qwen 3.6, Qwen 3.8, "
        "and Gemma 4 MTPLX models only."
    )
    family = str(payload.get("model_family") or "unknown")
    detail = str(payload.get("unsupported_reason") or message)
    body = {
        "ok": False,
        "error": "unsupported_tune_model",
        "model": payload.get("model"),
        "model_family": family,
        "supported_families": payload.get("supported_families")
        or ["qwen3_5", "qwen3_6", "qwen3_8", "gemma4"],
        "message": message,
        "detail": detail,
        "model_controls": payload.get("model_controls"),
    }
    if json_output:
        _print(body)
    else:
        print(message, file=sys.stderr)
        print(f"Selected model family: {family}.", file=sys.stderr)
        if detail and detail != message:
            print(detail, file=sys.stderr)
    return 2


def _tune_control_field(support_payload: dict[str, Any] | None) -> str:
    controls = (support_payload or {}).get("model_controls")
    tune = controls.get("tune") if isinstance(controls, dict) else None
    field = tune.get("control_field") if isinstance(tune, dict) else None
    if isinstance(field, str) and field.strip():
        return field.strip()
    return "depth"


def _tune_default_candidate_values(support_payload: dict[str, Any] | None) -> list[int]:
    controls = (support_payload or {}).get("model_controls")
    tune = controls.get("tune") if isinstance(controls, dict) else None
    draft = controls.get("draft_control") if isinstance(controls, dict) else None
    field = _tune_control_field(support_payload)
    if isinstance(tune, dict):
        values: list[int] = []
        for label in tune.get("candidates") or []:
            text = str(label)
            digits = "".join(ch for ch in text if ch.isdigit())
            if digits:
                value = int(digits)
                if value not in values:
                    values.append(value)
        if values:
            return values
    if field == "draft_block_size" and isinstance(draft, dict):
        minimum = int(draft.get("minimum") or 2)
        maximum = int(draft.get("maximum") or 8)
        return list(range(minimum, maximum + 1))
    return _parse_tune_depths(TUNE_DEFAULT_DEPTHS)


def _apply_tune_sampling_defaults(
    args: Any, support_payload: dict[str, Any] | None
) -> None:
    """Resolve tune sampling from the same family contract used by serve.

    Parser defaults predate Qwen3.8 and are therefore not evidence that the
    operator explicitly selected the legacy 0.6 sampler. Explicit CLI flags
    always win.
    """

    controls = (support_payload or {}).get("model_controls")
    sampling = controls.get("sampling") if isinstance(controls, dict) else None
    if not isinstance(sampling, dict):
        return
    cli_flags = getattr(args, "_cli_flags", set()) or set()
    for attribute, flag in (
        ("temperature", "temperature"),
        ("top_p", "top-p"),
        ("top_k", "top-k"),
    ):
        if flag not in cli_flags and sampling.get(attribute) is not None:
            setattr(args, attribute, sampling[attribute])


def _parse_tune_candidate_values(
    raw: Any,
    *,
    support_payload: dict[str, Any] | None,
) -> list[int]:
    field = _tune_control_field(support_payload)
    allowed_values = _tune_default_candidate_values(support_payload)
    if raw is None or str(raw).strip() == "":
        return allowed_values
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not parts:
        raise ValueError("tune candidates must include at least one value")
    values: list[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:
            if field == "draft_block_size":
                raise ValueError(
                    "Gemma tune blocks must be integers from 2 to 8"
                ) from exc
            raise ValueError("tune depths must be integers") from exc
        if field == "draft_block_size":
            if value < 2 or value > 8:
                raise ValueError("Gemma tune blocks must be between 2 and 8")
        else:
            if value not in allowed_values:
                allowed = ",".join(str(item) for item in allowed_values)
                raise ValueError(f"tune depths must be one of {allowed}")
        if value not in values:
            values.append(value)
    return values


def cmd_tune_public(args: Any) -> int:
    if getattr(args, "_tune_candidate", None):
        return _cmd_tune_candidate(args)
    return _cmd_tune(
        args,
        action="tune",
        save_default=not bool(getattr(args, "no_save", False)),
        verbose_default=bool(getattr(args, "verbose", False)),
    )


def _cmd_bench_tune(args: Any) -> int:
    child = SimpleNamespace(**vars(args))
    cli_flags = getattr(child, "_cli_flags", set()) or set()
    if "max-tokens" not in cli_flags:
        child.max_tokens = TUNE_DEFAULT_MAX_TOKENS
    return _cmd_tune(
        child,
        action="bench tune",
        save_default=False,
        verbose_default=True,
    )


def _cmd_tune(
    args: Any,
    *,
    action: str,
    save_default: bool,
    verbose_default: bool,
) -> int:
    model = _tune_requested_model(args)
    run_id = getattr(args, "run_id", None) or f"tune-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir = _absolute_user_path(
        getattr(args, "output_dir", None) or "outputs/cli/tune"
    )
    output_root = output_dir / run_id
    output_path = (
        _absolute_user_path(getattr(args, "output"))
        if getattr(args, "output", None)
        else output_root / "tune.json"
    )
    json_output = bool(getattr(args, "json", False))
    verbose = bool(getattr(args, "verbose", verbose_default) or verbose_default)
    collect_telemetry = _tune_collect_telemetry(args, action=action)

    if bool(getattr(args, "dry_run", False)):
        support_payload = _tune_support_payload(model)
        if not support_payload["tune_supported"]:
            return _unsupported_tune_model_error(
                support_payload,
                json_output=json_output or action == "bench tune",
            )
        _apply_tune_sampling_defaults(args, support_payload)
        try:
            depths = _parse_tune_candidate_values(
                getattr(args, "depths", None),
                support_payload=support_payload,
            )
        except ValueError as exc:
            return _tune_error(str(exc), json_output=json_output)
        settings = _tune_settings(
            args,
            model=model,
            depths=depths,
            control_field=_tune_control_field(support_payload),
        )
        model_source_notes = _tune_model_source_notes(args, runtime_model=model)
        payload = _tune_dry_run_payload(
            args,
            action=action,
            model=model,
            run_id=run_id,
            output_root=output_root,
            output_path=output_path,
            settings=settings,
            depths=depths,
            save_default=save_default,
            collect_telemetry=collect_telemetry,
            model_source_notes=model_source_notes,
            support_payload=support_payload,
        )
        if json_output or action == "bench tune":
            _print(payload)
        else:
            _print_tune_dry_run_human(payload)
        return 0

    runtime_model, resolve_error = _resolve_runtime_model_path(
        model,
        cache_dir=getattr(args, "cache_dir", None),
    )
    if resolve_error is not None:
        return _tune_error(
            resolve_error.get("error") or "model is not available locally",
            detail=resolve_error.get("detail"),
            json_output=json_output,
        )
    support_payload = _tune_support_payload(runtime_model, inspect_local=True)
    if not support_payload["tune_supported"]:
        return _unsupported_tune_model_error(
            support_payload,
            json_output=json_output,
        )
    _apply_tune_sampling_defaults(args, support_payload)
    try:
        depths = _parse_tune_candidate_values(
            getattr(args, "depths", None),
            support_payload=support_payload,
        )
    except ValueError as exc:
        return _tune_error(str(exc), json_output=json_output)
    settings = _tune_settings(
        args,
        model=runtime_model,
        depths=depths,
        control_field=_tune_control_field(support_payload),
    )
    model_source_notes = _tune_model_source_notes(args, runtime_model=runtime_model)

    hardware: dict[str, Any] | None = None
    software: dict[str, Any] | None = None
    backend: dict[str, Any] | None = None
    state_key: str | None = None
    key_material: dict[str, Any] | None = None

    def _resolve_tune_state_context() -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        str,
        dict[str, Any],
    ]:
        nonlocal hardware, software, backend, state_key, key_material
        if (
            hardware is None
            or software is None
            or backend is None
            or state_key is None
            or key_material is None
        ):
            # Shared constructor so save and wizard-lookup keys can never
            # drift (issue #280 re-tune loop). Falls back to the inline
            # construction only if the shared path cannot resolve — the
            # model is already resolved and support-checked by this point,
            # so that fallback should be unreachable.
            context = _tune_state_context_for_args(args)
            if context is not None:
                hardware, software, backend, state_key, key_material = context
            else:
                hardware = _apple_hardware_context()
                software = _software_context()
                backend = _mlx_backend_context()
                state_key, key_material = _tune_state_key(
                    runtime_model,
                    settings=settings,
                    hardware=hardware,
                    software=software,
                    backend=backend,
                )
        return hardware, software, backend, state_key, key_material

    cached = None
    if save_default and not bool(getattr(args, "retune", False)):
        _hardware, _software, _backend, _state_key, _key_material = (
            _resolve_tune_state_context()
        )
        cached = _load_tune_record(_state_key)
    if save_default and cached is not None:
        payload = dict(cached.get("payload") or {})
        payload["from_cache"] = True
        payload["state_path"] = str(_tune_state_path())
        if json_output:
            _print(payload)
        else:
            _print_tune_human(payload, verbose=verbose)
        return 0

    from mtplx.thermal import MaxSession

    def _emit(line: str) -> None:
        print(line, file=sys.stderr, flush=True)

    if not json_output:
        _emit(f"[tune] model: {runtime_model}")
        for note in model_source_notes:
            _emit(note)
        if action == "bench tune":
            if collect_telemetry:
                _emit(
                    "[tune] diagnostic telemetry active; use --no-telemetry for clean speed comparison"
                )
            else:
                _emit(
                    "[tune] diagnostic telemetry disabled; speed comparison is cleaner"
                )
        _emit(
            "[tune] close heavy apps now for cleaner results before measurements start"
        )
        _emit("[tune] fans may get loud during tuning and will be restored afterward")
    max_session = MaxSession(log=_emit)
    fans_pinned = bool(max_session.start())
    if not fans_pinned:
        verified = max_session.thermal.get("verified") or {}
        if bool(getattr(args, "require_max_fans", False)):
            return _tune_error(
                "verified max-fan mode did not start and --require-max-fans is set",
                detail=verified.get("message"),
                actionable=verified.get("actionable"),
                thermal=max_session.thermal,
                json_output=json_output,
            )
        # Onboarding has to finish on every Mac. Pinned fans give the
        # cleanest timing, but a depth measured on auto fans still beats
        # a dead setup wizard (a real M5 Max user hit exactly this when
        # the helper could not verify a ramp). Continue unpinned and
        # record the honesty flag in the summary; --require-max-fans
        # keeps the strict behavior for benchmarking.
        if not json_output:
            _emit("[tune] fan pinning unavailable; tuning with fans on auto")
            reason = verified.get("message")
            if reason:
                _emit(f"[tune]   reason: {reason}")
            actionable = verified.get("actionable")
            if actionable:
                _emit(f"[tune]   to enable pinned-fan tuning later: {actionable}")

    try:
        if not json_output:
            _emit(f"[tune] artifacts: {output_root}")
            _emit(
                "[tune] running isolated candidates: "
                + ", ".join(
                    _tune_candidate_label(candidate, settings.get("control_field"))
                    for candidate in ["ar", *[str(depth) for depth in depths]]
                )
            )
        candidate_rows = _run_tune_candidates(
            args,
            runtime_model=runtime_model,
            run_id=run_id,
            output_root=output_root,
            depths=depths,
            settings=settings,
            progress=_emit if not json_output else None,
            collect_telemetry=collect_telemetry,
        )
    finally:
        max_session.stop()

    hardware, software, backend, state_key, key_material = _resolve_tune_state_context()
    payload = _tune_payload(
        action=action,
        run_id=run_id,
        model=runtime_model,
        settings=settings,
        rows=candidate_rows,
        output_root=output_root,
        output_path=output_path,
        hardware=hardware,
        software=software,
        backend=backend,
        thermal=max_session.thermal,
        state_key=state_key,
        diagnostics={
            "telemetry_enabled": collect_telemetry,
            "telemetry_env": TUNE_TELEMETRY_ENV,
            "model_source_notes": model_source_notes,
            "fans_pinned": fans_pinned,
        },
    )
    payload["fans_pinned"] = fans_pinned
    if not fans_pinned:
        payload["fans_note"] = (
            "Tuned with fans on auto; pinned-fan timing was unavailable. "
            "Results are valid for how this Mac actually runs."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, payload)

    if (
        payload.get("best")
        and save_default
        and not bool(getattr(args, "no_save", False))
    ):
        payload["saved"] = True
        payload["state_path"] = str(_tune_state_path())
        _save_tune_record(state_key, key_material=key_material, payload=payload)
        write_json(output_path, payload)
    else:
        payload["saved"] = False
        payload["state_path"] = str(_tune_state_path())
        if not any(
            isinstance(row.get("tok_s"), (int, float)) for row in candidate_rows
        ):
            payload["save_skipped_reason"] = (
                "tune failed; no candidate produced usable tokens"
            )
        elif not payload.get("best"):
            verdict = str((payload.get("best_multiplier") or {}).get("verdict") or "")
            payload["save_skipped_reason"] = (
                "no quality-passed MTP depth beat AR"
                if verdict == "no_quality_passed_mtp_depth_beat_ar"
                else "no MTP depth beat AR"
            )
            # A measured loss supersedes any stored winner for the same
            # environment; otherwise a record poisoned by GPU contention
            # replays forever (#177).
            if (
                save_default
                and not bool(getattr(args, "no_save", False))
                and (verdict in TUNE_RECORD_CLEARING_VERDICTS)
            ):
                cleared = _clear_tune_record(state_key)
                if cleared is not None:
                    payload["cleared_saved_record"] = cleared
                    write_json(output_path, payload)
        elif bool(getattr(args, "no_save", False)) or not save_default:
            payload["save_skipped_reason"] = "save disabled"
        write_json(output_path, payload)

    if json_output:
        _print(payload)
    else:
        _print_tune_human(payload, verbose=verbose)
    if not candidate_rows or not any(
        isinstance(row.get("tok_s"), (int, float)) for row in candidate_rows
    ):
        return 1
    return 0


def _cmd_tune_candidate(args: Any) -> int:
    candidate = str(getattr(args, "_tune_candidate", "")).lower()
    output = Path(getattr(args, "_tune_candidate_output", None) or "")
    if candidate not in {"ar", "1", "2", "3", "4", "5", "6", "7", "8"}:
        return _tune_error("invalid tune candidate", json_output=True)
    if not output:
        return _tune_error("missing tune candidate output path", json_output=True)
    model = getattr(args, "model", None) or DEFAULT_CHAMPION
    runtime_model, resolve_error = _resolve_runtime_model_path(
        model,
        cache_dir=getattr(args, "cache_dir", None),
    )
    if resolve_error is not None:
        _print(resolve_error)
        return 1
    fast_inspection = _fast_mtplx_tune_inspection(runtime_model)
    if fast_inspection is not None:
        inspection, gate_exit = fast_inspection, None
    else:
        inspection, gate_exit = _model_gate(
            runtime_model,
            unsafe_force_unverified=bool(
                getattr(args, "unsafe_force_unverified", False)
            ),
            yes=True,
        )
    if gate_exit is not None or inspection is None:
        _print({"error": "model failed MTP primary gate", "model": inspection})
        return gate_exit or 1
    support_payload = _tune_support_payload(runtime_model, inspect_local=True)
    if not support_payload["tune_supported"]:
        return _unsupported_tune_model_error(support_payload, json_output=True)
    control_field = _tune_control_field(support_payload)
    if candidate != "ar":
        value = int(candidate)
        if control_field == "draft_block_size":
            if value < 2 or value > 8:
                return _tune_error(
                    "Gemma tune blocks must be between 2 and 8", json_output=True
                )
        elif value not in _tune_default_candidate_values(support_payload):
            allowed = ",".join(
                str(item) for item in _tune_default_candidate_values(support_payload)
            )
            return _tune_error(
                f"tune depths must be one of {allowed}", json_output=True
            )
    # The parent tune run always passes --profile explicitly; this default
    # covers direct candidate invocations and must match what serve resolves
    # (the hidden performance-cold default tuned depth under different
    # kernels than the launch profile actually uses).
    profile = get_profile(_resolved_default_profile_name(args, runtime_model))
    runtime_env = _runtime_env_with_external_overrides(
        _runtime_env_with_model_contract_overrides(
            profile.env_dict(),
            inspection,
            profile,
        )
    )
    draft_lm_head = _model_draft_lm_head_spec(inspection, profile)
    draft_sampler = _model_draft_sampler_spec(inspection, profile)
    if any(
        getattr(args, name, None) is not None
        for name in ("draft_temperature", "draft_top_p", "draft_top_k")
    ):
        draft_sampler = {
            "temperature": float(
                getattr(args, "draft_temperature", None)
                if getattr(args, "draft_temperature", None) is not None
                else (draft_sampler or {}).get("temperature", 0.6)
            ),
            "top_p": float(
                getattr(args, "draft_top_p", None)
                if getattr(args, "draft_top_p", None) is not None
                else (draft_sampler or {}).get("top_p", 0.95)
            ),
            "top_k": int(
                getattr(args, "draft_top_k", None)
                if getattr(args, "draft_top_k", None) is not None
                else (draft_sampler or {}).get("top_k", 20)
            ),
        }
    result = _depth_sweep_native60(
        model=runtime_model,
        prompt_suite=prompt_suite_path(
            getattr(args, "prompt_suite", None) or TUNE_DEFAULT_SUITE
        ),
        depths="1" if candidate == "ar" else candidate,
        max_tokens=int(
            getattr(args, "max_tokens", TUNE_DEFAULT_MAX_TOKENS)
            or TUNE_DEFAULT_MAX_TOKENS
        ),
        limit=int(getattr(args, "limit", TUNE_DEFAULT_LIMIT) or TUNE_DEFAULT_LIMIT),
        seed=int(getattr(args, "seed", TUNE_DEFAULT_SEED) or TUNE_DEFAULT_SEED),
        temperature=float(getattr(args, "temperature", 0.6)),
        top_p=float(getattr(args, "top_p", 0.95)),
        top_k=int(getattr(args, "top_k", 20)),
        draft_lm_head=draft_lm_head,
        draft_sampler=draft_sampler,
        mtp_hidden_variant=getattr(args, "mtp_hidden_variant", None),
        base_hidden_variant=getattr(args, "base_hidden_variant", None),
        concat_order=getattr(args, "concat_order", None),
        mtp_cache_policy=str(getattr(args, "mtp_cache_policy", None) or "persistent"),
        mtp_history_policy=str(
            getattr(args, "mtp_history_policy", None) or "committed"
        ),
        compare_ar=candidate == "ar",
        ar_only=candidate == "ar",
        gemma4_draft_block_size=(
            None
            if candidate == "ar" or control_field != "draft_block_size"
            else int(candidate)
        ),
        runtime_env=runtime_env,
        draft_core=str(getattr(args, "draft_core", None) or "stock"),
    )
    from mtplx.benchmarks.runners.mtp_depth_sweep import write_depth_sweep

    write_depth_sweep(output, result)
    _print({"candidate": candidate, "output": str(output)})
    return 0


def _parse_tune_depths(raw: Any) -> list[int]:
    if raw is None:
        raw = TUNE_DEFAULT_DEPTHS
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not parts:
        raise ValueError("tune depths must include at least one of 1,2,3")
    depths: list[int] = []
    for part in parts:
        try:
            depth = int(part)
        except ValueError as exc:
            raise ValueError("tune depths must be integers from 1 to 3") from exc
        if depth < 1 or depth > MAX_PUBLIC_SPECULATIVE_DEPTH:
            raise ValueError("tune depths must be between 1 and 3")
        if depth not in depths:
            depths.append(depth)
    return depths


def _tune_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    text = str(raw).strip().lower()
    if text in {"0", "off", "false", "none", "no"}:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return default


def _tune_env_bool(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled", "none"}:
        return False
    return default


def _tune_collect_telemetry(args: Any, *, action: str) -> bool:
    return (
        action == "bench tune"
        and not bool(getattr(args, "no_telemetry", False))
        and _tune_env_bool(TUNE_TELEMETRY_ENV, default=True)
    )


def _normalize_model_ref_for_compare(value: str | Path | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith(("~", "/", "./", "../")):
        path = Path(text).expanduser()
        try:
            return str(path.resolve())
        except OSError:
            return str(path)
    return text


def _same_model_ref(left: str | Path | None, right: str | Path | None) -> bool:
    return _normalize_model_ref_for_compare(left) == _normalize_model_ref_for_compare(
        right
    )


def _tune_model_source_notes(args: Any, *, runtime_model: str) -> list[str]:
    cli_flags = getattr(args, "_cli_flags", set()) or set()
    if "model" in cli_flags:
        return []
    config = getattr(args, "mtplx_config", None)
    config_model = config.get("model") if isinstance(config, dict) else None
    if not config_model or not _same_model_ref(runtime_model, config_model):
        return []
    try:
        default_selection = select_default_model()
    except Exception:
        return []
    default_model = default_selection.model
    if _same_model_ref(runtime_model, default_model):
        return []
    config_path = (
        config.get("path") if isinstance(config, dict) else "~/.mtplx/config.toml"
    )
    return [
        f"[tune] using configured model from {config_path}: {runtime_model}",
        f"[tune] verified default for this Mac is: {default_model}",
        "[tune] pass --model <path> to benchmark a different artifact explicitly",
    ]


def _tune_settings(
    args: Any,
    *,
    model: str,
    depths: list[int],
    control_field: str = "depth",
) -> dict[str, Any]:
    suite = (
        getattr(args, "prompt_suite", None)
        or getattr(args, "suite", None)
        or TUNE_DEFAULT_SUITE
    )
    # Tune must measure under the profile serve will actually resolve for
    # this model (the macOS app already guards this); the old hidden
    # performance-cold default tuned depth under different kernels than the
    # launch profile uses. An explicit --profile always wins.
    profile = get_profile(_resolved_default_profile_name(args, model))
    return {
        "profile": profile.name,
        "suite": str(suite),
        "depths": ",".join(str(depth) for depth in depths),
        "control_field": control_field,
        "max_tokens": int(
            getattr(args, "max_tokens", TUNE_DEFAULT_MAX_TOKENS)
            or TUNE_DEFAULT_MAX_TOKENS
        ),
        "limit": int(getattr(args, "limit", TUNE_DEFAULT_LIMIT) or TUNE_DEFAULT_LIMIT),
        "seed": int(getattr(args, "seed", TUNE_DEFAULT_SEED) or TUNE_DEFAULT_SEED),
        "temperature": float(getattr(args, "temperature", 0.6)),
        "top_p": float(getattr(args, "top_p", 0.95)),
        "top_k": int(getattr(args, "top_k", 20)),
        "thinking": "disabled",
        "mtp_hidden_variant": (
            str(getattr(args, "mtp_hidden_variant"))
            if getattr(args, "mtp_hidden_variant", None)
            else None
        ),
        "base_hidden_variant": (
            str(getattr(args, "base_hidden_variant"))
            if getattr(args, "base_hidden_variant", None)
            else None
        ),
        "concat_order": (
            str(getattr(args, "concat_order"))
            if getattr(args, "concat_order", None)
            else None
        ),
        "mtp_cache_policy": str(
            getattr(args, "mtp_cache_policy", None) or "persistent"
        ),
        "mtp_history_policy": str(
            getattr(args, "mtp_history_policy", None) or "committed"
        ),
        "draft_temperature": getattr(args, "draft_temperature", None),
        "draft_top_p": getattr(args, "draft_top_p", None),
        "draft_top_k": getattr(args, "draft_top_k", None),
        "draft_core": (
            str(getattr(args, "draft_core", None))
            if getattr(args, "draft_core", None) not in (None, "stock")
            else None
        ),
        "candidate_settle_s": max(
            0.0,
            _tune_env_float("MTPLX_TUNE_CANDIDATE_SETTLE_S", TUNE_CANDIDATE_SETTLE_S),
        ),
        "tie_prefer_deeper_within_pct": max(
            0.0,
            _tune_env_float(
                "MTPLX_TUNE_TIE_PREFER_DEEPER_WITHIN_PCT",
                TUNE_TIE_PREFER_DEEPER_WITHIN_PCT,
            ),
        ),
    }


def _tune_state_path() -> Path:
    env = os.environ.get("MTPLX_TUNE_STATE")
    return Path(env).expanduser() if env else TUNE_STATE_PATH


def _load_tune_state() -> dict[str, Any]:
    path = _tune_state_path()
    if not path.exists():
        return {"schema_version": 1, "records": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "records": {}}
    if not isinstance(data, dict):
        return {"schema_version": 1, "records": {}}
    records = data.get("records")
    if not isinstance(records, dict):
        data["records"] = {}
    return data


def _tune_record_winner_collapsed(payload: dict[str, Any]) -> bool:
    """True when a saved record's own results show its winner measured ~zero
    acceptance — a poisoned artifact of pre-guard tunes under GPU contention
    (#177). Records without acceptance evidence are trusted as-is."""
    best = payload.get("best") or {}
    if not isinstance(best, dict):
        return False
    mode = best.get("mode")
    depth = best.get("depth")
    rows = [row for row in payload.get("results") or [] if isinstance(row, dict)]
    match = None
    if mode is not None:
        match = next((row for row in rows if row.get("mode") == mode), None)
    if match is None and depth is not None:
        match = next((row for row in rows if row.get("depth") == depth), None)
    return _tune_row_acceptance_collapsed(match) if match is not None else False


def _load_tune_record(state_key: str) -> dict[str, Any] | None:
    record = (_load_tune_state().get("records") or {}).get(state_key)
    if not isinstance(record, dict):
        return None
    payload = record.get("payload")
    best = payload.get("best") if isinstance(payload, dict) else None
    if not isinstance(best, dict):
        return None
    if not isinstance(best.get("depth"), int):
        return None
    # Quarantine, don't replay: every consumer (cached tune replay, quickstart
    # and Web UI depth application) treats a poisoned winner as absent, so an
    # already-affected install falls back to defaults instead of running a
    # depth whose measured acceptance was zero (#177).
    if _tune_record_winner_collapsed(payload):
        return None
    return record


def _clear_tune_record(state_key: str) -> dict[str, Any] | None:
    """Remove a previously saved winner for this state key, returning a short
    summary of what was cleared (None when nothing was stored). A measured
    no-winner verdict is affirmative evidence that the stored depth no longer
    wins in this exact environment; without this, a record poisoned by GPU
    contention outlives every honest retune (#177)."""
    state = _load_tune_state()
    records = state.get("records")
    if not isinstance(records, dict) or state_key not in records:
        return None
    old = records.pop(state_key)
    path = _tune_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, state)
    old_payload = old.get("payload") if isinstance(old, dict) else None
    old_best = old_payload.get("best") if isinstance(old_payload, dict) else None
    return {
        "state_key": state_key,
        "previous_best": old_best if isinstance(old_best, dict) else None,
        "saved_at": old.get("saved_at") if isinstance(old, dict) else None,
    }


def _save_tune_record(
    state_key: str, *, key_material: dict[str, Any], payload: dict[str, Any]
) -> None:
    state = _load_tune_state()
    records = state.setdefault("records", {})
    records[state_key] = {
        "schema_version": 1,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "key": state_key,
        "key_material": key_material,
        "payload": payload,
    }
    path = _tune_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, state)


def _tune_state_context_for_args(
    args: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, dict[str, Any]] | None:
    """Resolve (hardware, software, backend, state_key, key_material) exactly as
    a live ``mtplx tune`` save would for these args.

    Every reader of the tune-record store MUST derive its key through this
    function. The 2.8.0/2.8.1 quickstart wizard rebuilt the settings dict by
    hand (stale ``performance-cold`` profile, unjoined depths, missing keys),
    so its lookup hash could never match the record ``tune`` had just saved —
    users were re-offered tuning on every start (issue #280).
    """
    model, resolve_error = _resolve_runtime_model_path(
        _tune_requested_model(args),
        cache_dir=getattr(args, "cache_dir", None),
    )
    if resolve_error is not None:
        return None
    support_payload = _tune_support_payload(model, inspect_local=True)
    if not support_payload["tune_supported"]:
        return None
    _apply_tune_sampling_defaults(args, support_payload)
    try:
        depths = _parse_tune_candidate_values(
            getattr(args, "depths", None),
            support_payload=support_payload,
        )
    except ValueError:
        return None
    settings = _tune_settings(
        args,
        model=model,
        depths=depths,
        control_field=_tune_control_field(support_payload),
    )
    hardware = _apple_hardware_context()
    software = _software_context()
    backend = _mlx_backend_context()
    state_key, key_material = _tune_state_key(
        model,
        settings=settings,
        hardware=hardware,
        software=software,
        backend=backend,
    )
    return hardware, software, backend, state_key, key_material


def _tune_state_key(
    model: str,
    *,
    settings: dict[str, Any],
    hardware: dict[str, Any],
    software: dict[str, Any],
    backend: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    model_identity = (
        str(Path(model).expanduser().resolve())
        if Path(model).expanduser().exists()
        else str(model)
    )
    key_material = {
        "model": model_identity,
        "hardware": {
            "chip": hardware.get("chip"),
            "chip_family": hardware.get("chip_family"),
            "hw_model": hardware.get("hw_model"),
            "machine": hardware.get("machine"),
        },
        "software": {
            "mtplx_version": software.get("mtplx_version"),
            "mlx_version": software.get("mlx_version"),
            "mlx_lm_version": software.get("mlx_lm_version"),
        },
        "backend": {
            "mlx_core_path": backend.get("mlx_core_path"),
            "stock_mlx_likely": backend.get("stock_mlx_likely"),
        },
        "settings": settings,
    }
    encoded = json.dumps(key_material, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), key_material


def _tune_dry_run_payload(
    args: Any,
    *,
    action: str,
    model: str,
    run_id: str,
    output_root: Path,
    output_path: Path,
    settings: dict[str, Any],
    depths: list[int],
    save_default: bool,
    collect_telemetry: bool,
    model_source_notes: list[str] | None = None,
    support_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    commands = []
    for candidate in ["ar", *[str(depth) for depth in depths]]:
        candidate_output = output_root / (
            _tune_candidate_file_stem(candidate, settings.get("control_field"))
            + ".json"
        )
        commands.append(
            {
                "candidate": _tune_candidate_label(
                    candidate, settings.get("control_field")
                ),
                "command": _tune_candidate_command(
                    args,
                    candidate=candidate,
                    model=model,
                    output=candidate_output,
                    settings=settings,
                ),
                "output": str(candidate_output),
            }
        )
    return {
        "dry_run": True,
        "action": action,
        "run_id": run_id,
        "model": model,
        "settings": settings,
        "model_family": (support_payload or {}).get("model_family"),
        "backend_id": (support_payload or {}).get("backend_id"),
        "tune_supported": (support_payload or {}).get("tune_supported", True),
        "unsupported_reason": (support_payload or {}).get("unsupported_reason"),
        "model_controls": (support_payload or {}).get("model_controls"),
        "candidates": commands,
        "output": str(output_path),
        "state_path": str(_tune_state_path()),
        "save_default": save_default,
        "fan_control": "verified max-fan required before model load",
        "diagnostics": {
            "telemetry_enabled": collect_telemetry,
            "telemetry_env": TUNE_TELEMETRY_ENV,
            "model_source_notes": model_source_notes or [],
            "description": (
                "bench tune records per-candidate power, frequency, temperature, "
                "utilization, and fan telemetry when not in dry-run"
                if action == "bench tune" and collect_telemetry
                else (
                    "bench tune telemetry disabled; candidate commands match clean tune timing more closely"
                    if action == "bench tune"
                    else None
                )
            ),
        },
    }


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _series_stats(values: list[Any]) -> dict[str, Any] | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    if not numeric:
        return None
    return {
        "avg": sum(numeric) / len(numeric),
        "min": min(numeric),
        "max": max(numeric),
        "last": numeric[-1],
        "samples": len(numeric),
    }


def _generation_windows_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    windows: list[dict[str, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        start = _float_or_none(row.get("generation_started_at"))
        end = _float_or_none(row.get("generation_ended_at"))
        if start is None or end is None or end <= start:
            continue
        windows.append({"start": start, "end": end, "duration_s": end - start})
    return windows


def _convert_power_to_watts(value: str, unit: str | None) -> float:
    watts = float(value)
    if (unit or "").lower() == "mw":
        watts /= 1000.0
    return watts


def _convert_frequency_to_ghz(value: str, unit: str | None) -> float:
    ghz = float(value)
    if (unit or "").lower() == "mhz":
        ghz /= 1000.0
    return ghz


def _parse_powermetrics_text(text: str) -> dict[str, Any]:
    """Extract the MX Power Gadget-style rails from macOS powermetrics text."""

    power: dict[str, float] = {}
    for match in re.finditer(
        r"(?m)^(CPU|GPU|ANE) Power:\s*([0-9]+(?:\.[0-9]+)?)\s*(mW|W)\b",
        text,
    ):
        rail = match.group(1).lower()
        power[rail] = _convert_power_to_watts(match.group(2), match.group(3))
    package_match = re.search(
        r"(?m)^Combined Power \(CPU \+ GPU \+ ANE\):\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*(mW|W)\b",
        text,
    )
    if package_match:
        power["package"] = _convert_power_to_watts(
            package_match.group(1),
            package_match.group(2),
        )

    p_frequencies: list[float] = []
    m_frequencies: list[float] = []
    p_utilization: list[float] = []
    m_utilization: list[float] = []
    for match in re.finditer(
        r"(?m)^([A-Za-z0-9]+)-Cluster HW active frequency:\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*(MHz|GHz)\b",
        text,
    ):
        name = match.group(1).upper()
        ghz = _convert_frequency_to_ghz(match.group(2), match.group(3))
        if name == "P":
            p_frequencies.append(ghz)
        else:
            m_frequencies.append(ghz)
    for match in re.finditer(
        r"(?m)^([A-Za-z0-9]+)-Cluster HW active residency:\s*"
        r"([0-9]+(?:\.[0-9]+)?)%",
        text,
    ):
        name = match.group(1).upper()
        residency = float(match.group(2))
        if name == "P":
            p_utilization.append(residency)
        else:
            m_utilization.append(residency)

    frequency: dict[str, float] = {}
    utilization: dict[str, float] = {}
    p_cluster = _avg(p_frequencies)
    m_cluster = _avg(m_frequencies)
    p_core = _avg(p_utilization)
    m_core = _avg(m_utilization)
    if p_cluster is not None:
        frequency["p_cluster"] = p_cluster
    if m_cluster is not None:
        frequency["m_cluster"] = m_cluster
    if p_core is not None:
        utilization["p_core"] = p_core
    if m_core is not None:
        utilization["m_core"] = m_core

    gpu_freq_match = re.search(
        r"(?m)^GPU HW active frequency:\s*([0-9]+(?:\.[0-9]+)?)\s*(MHz|GHz)\b",
        text,
    )
    if gpu_freq_match:
        frequency["gpu"] = _convert_frequency_to_ghz(
            gpu_freq_match.group(1),
            gpu_freq_match.group(2),
        )
    gpu_residency_match = re.search(
        r"(?m)^GPU HW active residency:\s*([0-9]+(?:\.[0-9]+)?)%",
        text,
    )
    if gpu_residency_match:
        utilization["gpu"] = float(gpu_residency_match.group(1))

    payload: dict[str, Any] = {
        "source": "powermetrics",
        "power_w": power,
        "frequency_ghz": frequency,
        "utilization_pct": utilization,
    }
    pressure_match = re.search(r"(?m)^Current pressure level:\s*(.+)$", text)
    if pressure_match:
        payload["thermal_pressure"] = pressure_match.group(1).strip()
    return payload


def _thermalforge_binary() -> str | None:
    owned = Path("~/.mtplx/bin/thermalforge").expanduser()
    if owned.exists():
        return str(owned)
    return shutil.which("thermalforge")


def _temperature_groups_from_thermalforge(
    temperatures: dict[str, Any],
) -> tuple[list[float], list[float]]:
    core_values: list[float] = []
    fallback_core_values: list[float] = []
    gpu_values: list[float] = []
    for key, raw_value in temperatures.items():
        value = _float_or_none(raw_value)
        if value is None:
            continue
        name = str(key)
        upper = name.upper()
        if upper.startswith("TG") or name.startswith("Tg"):
            gpu_values.append(value)
        elif upper.startswith("TC") or name.startswith(("Tp", "Tm")):
            core_values.append(value)
        else:
            fallback_core_values.append(value)
    return core_values or fallback_core_values, gpu_values


def _sample_thermalforge_status() -> dict[str, Any]:
    binary = _thermalforge_binary()
    if not binary:
        return {"source": "thermalforge", "error": "thermalforge not found"}
    try:
        proc = subprocess.run(
            [binary, "status"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=1.0,
        )
    except Exception as exc:
        return {"source": "thermalforge", "error": str(exc)}
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        return {"source": "thermalforge", "error": detail or f"exit {proc.returncode}"}
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {"source": "thermalforge", "error": f"invalid JSON: {exc}"}

    fans = data.get("fans") if isinstance(data, dict) else []
    fan_rpms: list[float] = []
    if isinstance(fans, dict):
        fans = list(fans.values())
    for fan in fans if isinstance(fans, list) else []:
        if not isinstance(fan, dict):
            continue
        rpm = _float_or_none(fan.get("actual_rpm"))
        if rpm is None:
            rpm = _float_or_none(fan.get("target_rpm"))
        if rpm is not None:
            fan_rpms.append(rpm)

    temperatures = data.get("temperatures") if isinstance(data, dict) else {}
    core_values, gpu_values = (
        _temperature_groups_from_thermalforge(temperatures)
        if isinstance(temperatures, dict)
        else ([], [])
    )
    temperature: dict[str, float] = {}
    if core_values:
        temperature["core_avg"] = sum(core_values) / len(core_values)
        temperature["core_max"] = max(core_values)
        temperature["core_min"] = min(core_values)
    if gpu_values:
        temperature["gpu_avg"] = sum(gpu_values) / len(gpu_values)

    payload: dict[str, Any] = {
        "source": "thermalforge",
        "fans_rpm": {
            "min": min(fan_rpms),
            "max": max(fan_rpms),
            "avg": sum(fan_rpms) / len(fan_rpms),
        }
        if fan_rpms
        else {},
        "temperature_c": temperature,
    }
    if isinstance(temperatures, dict):
        payload["temperature_sensors_c"] = {
            str(key): value
            for key, value in temperatures.items()
            if isinstance(value, (int, float))
        }
    return payload


def _sample_powermetrics_once() -> dict[str, Any]:
    if not shutil.which("powermetrics"):
        return {"source": "powermetrics", "error": "powermetrics not found"}
    if not shutil.which("sudo"):
        return {"source": "powermetrics", "error": "sudo not found"}
    command = [
        "sudo",
        "-n",
        "powermetrics",
        "-n",
        "1",
        "-i",
        "250",
        "--samplers",
        "cpu_power,gpu_power,ane_power,thermal",
    ]
    try:
        proc = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=2.0,
        )
    except Exception as exc:
        return {"source": "powermetrics", "error": str(exc)}
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        return {"source": "powermetrics", "error": detail or f"exit {proc.returncode}"}
    return _parse_powermetrics_text(proc.stdout)


def _summarize_tune_telemetry_samples(
    samples: list[dict[str, Any]],
    *,
    errors: list[str] | None = None,
    started_at: float | None = None,
    ended_at: float | None = None,
) -> dict[str, Any]:
    power_keys = ("package", "cpu", "ane", "gpu")
    frequency_keys = ("p_cluster", "m_cluster", "gpu")
    temperature_keys = ("core_avg", "core_max", "core_min", "gpu_avg")
    utilization_keys = ("p_core", "m_core", "gpu")

    summary: dict[str, Any] = {
        "enabled": True,
        "scope": "candidate_process",
        "sample_count": len(samples),
        "duration_s": (
            float(ended_at - started_at)
            if started_at is not None and ended_at is not None
            else None
        ),
        "samples": samples,
        "sources": {
            "thermalforge": any(sample.get("thermalforge_ok") for sample in samples),
            "powermetrics": any(sample.get("powermetrics_ok") for sample in samples),
        },
    }
    errors = list(errors or [])
    if errors:
        summary["errors"] = sorted(set(errors))

    power = {
        key: _series_stats(
            [((sample.get("power_w") or {}).get(key)) for sample in samples]
        )
        for key in power_keys
    }
    frequency = {
        key: _series_stats(
            [((sample.get("frequency_ghz") or {}).get(key)) for sample in samples]
        )
        for key in frequency_keys
    }
    temperature = {
        key: _series_stats(
            [((sample.get("temperature_c") or {}).get(key)) for sample in samples]
        )
        for key in temperature_keys
    }
    utilization = {
        key: _series_stats(
            [((sample.get("utilization_pct") or {}).get(key)) for sample in samples]
        )
        for key in utilization_keys
    }
    fans = {
        key: _series_stats(
            [((sample.get("fans_rpm") or {}).get(key)) for sample in samples]
        )
        for key in ("avg", "min", "max")
    }
    summary["power_w"] = {key: value for key, value in power.items() if value}
    summary["frequency_ghz"] = {key: value for key, value in frequency.items() if value}
    summary["temperature_c"] = {
        key: value for key, value in temperature.items() if value
    }
    summary["utilization_pct"] = {
        key: value for key, value in utilization.items() if value
    }
    summary["fans_rpm"] = {key: value for key, value in fans.items() if value}

    pressures = [
        str(sample.get("thermal_pressure"))
        for sample in samples
        if sample.get("thermal_pressure")
    ]
    if pressures:
        summary["thermal_pressure"] = {
            "last": pressures[-1],
            "observed": sorted(set(pressures)),
        }

    latest_sensors = next(
        (
            sample.get("temperature_sensors_c")
            for sample in reversed(samples)
            if sample.get("temperature_sensors_c")
        ),
        None,
    )
    if isinstance(latest_sensors, dict):
        summary["latest_temperature_sensors_c"] = latest_sensors
    return summary


def _sample_in_any_window(
    sample: dict[str, Any], windows: list[dict[str, float]]
) -> bool:
    timestamp = _float_or_none(sample.get("timestamp"))
    if timestamp is None:
        return False
    return any(window["start"] <= timestamp <= window["end"] for window in windows)


def _attach_generation_telemetry(
    telemetry: dict[str, Any] | None,
    row: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(telemetry, dict):
        return telemetry
    windows = [
        window
        for window in row.get("generation_windows", [])
        if isinstance(window, dict)
        and isinstance(window.get("start"), (int, float))
        and isinstance(window.get("end"), (int, float))
        and float(window["end"]) > float(window["start"])
    ]
    if not windows:
        telemetry["generation"] = {
            "enabled": True,
            "scope": "generation_window",
            "sample_count": 0,
            "note": "candidate artifact did not include generation timing windows",
        }
        return telemetry
    samples = [
        sample
        for sample in telemetry.get("samples", [])
        if isinstance(sample, dict) and _sample_in_any_window(sample, windows)
    ]
    started_at = min(float(window["start"]) for window in windows)
    ended_at = max(float(window["end"]) for window in windows)
    generation = _summarize_tune_telemetry_samples(
        samples,
        errors=telemetry.get("errors") or [],
        started_at=started_at,
        ended_at=ended_at,
    )
    generation["scope"] = "generation_window"
    generation["windows"] = windows
    generation["window_count"] = len(windows)
    generation["window_duration_s"] = sum(
        float(window["end"]) - float(window["start"]) for window in windows
    )
    if not samples:
        generation["note"] = (
            "no powermetrics/thermalforge sample landed inside the generation window"
        )
    telemetry["generation"] = generation
    return telemetry


class _TuneTelemetrySampler:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, Any]] = []
        self._errors: list[str] = []
        self._lock = threading.Lock()
        self._powermetrics_disabled = False
        self._started_at: float | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        ended_at = time.monotonic()
        with self._lock:
            samples = list(self._samples)
            errors = list(self._errors)
        return _summarize_tune_telemetry_samples(
            samples,
            errors=errors,
            started_at=self._started_at,
            ended_at=ended_at,
        )

    def _loop(self) -> None:
        next_powermetrics = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            sample_started_at = time.time()
            sample: dict[str, Any] = {}
            thermal = _sample_thermalforge_status()
            if thermal.get("error"):
                self._add_error(f"thermalforge: {thermal.get('error')}")
            else:
                sample["thermalforge_ok"] = True
                sample.update({k: v for k, v in thermal.items() if k != "source"})
            if not self._powermetrics_disabled and now >= next_powermetrics:
                power = _sample_powermetrics_once()
                next_powermetrics = now + TUNE_POWERMETRICS_SAMPLE_INTERVAL_S
                if power.get("error"):
                    self._powermetrics_disabled = True
                    self._add_error(f"powermetrics: {power.get('error')}")
                else:
                    sample["powermetrics_ok"] = True
                    for key in (
                        "power_w",
                        "frequency_ghz",
                        "utilization_pct",
                        "thermal_pressure",
                    ):
                        if key in power:
                            sample[key] = power[key]
            sample_ended_at = time.time()
            sample["sample_started_at"] = sample_started_at
            sample["sample_ended_at"] = sample_ended_at
            sample["timestamp"] = (sample_started_at + sample_ended_at) / 2.0
            with self._lock:
                self._samples.append(sample)
            self._stop.wait(TUNE_TELEMETRY_SAMPLE_INTERVAL_S)

    def _add_error(self, message: str) -> None:
        with self._lock:
            if message not in self._errors:
                self._errors.append(message)


def _run_tune_candidates(
    args: Any,
    *,
    runtime_model: str,
    run_id: str,
    output_root: Path,
    depths: list[int],
    settings: dict[str, Any],
    progress: Callable[[str], None] | None = None,
    collect_telemetry: bool = False,
) -> list[dict[str, Any]]:
    output_root = _absolute_user_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    total = 1 + len(depths)
    settle_s = float(settings.get("candidate_settle_s") or 0.0)
    control_field = str(settings.get("control_field") or "depth")
    for candidate in ["ar", *[str(depth) for depth in depths]]:
        label = _tune_candidate_label(candidate, control_field)
        stem = _tune_candidate_file_stem(candidate, control_field)
        candidate_output = output_root / f"{stem}.json"
        stdout_path = output_root / f"{stem}.log"
        command = _tune_candidate_command(
            args,
            candidate=candidate,
            model=runtime_model,
            output=candidate_output,
            settings=settings,
        )
        if rows and settle_s > 0.0:
            if progress is not None:
                progress(f"[tune] settling {settle_s:.1f}s before {label}")
            time.sleep(settle_s)
        if progress is not None:
            progress(
                f"[tune] {label} ({len(rows) + 1}/{total}) starting; log: {stdout_path}"
            )
        started = time.monotonic()
        telemetry_sampler = _TuneTelemetrySampler(enabled=collect_telemetry)
        telemetry_sampler.start()
        try:
            proc = subprocess.run(
                command,
                cwd=repo_root(),
                env={**os.environ, "MTPLX_TUNE_CHILD": "1"},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
        finally:
            telemetry = telemetry_sampler.stop()
        elapsed_s = time.monotonic() - started
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        row = _tune_candidate_summary(
            candidate,
            candidate_output,
            control_field=control_field,
            returncode=proc.returncode,
            stdout_path=stdout_path,
            command=command,
        )
        if telemetry is not None:
            row["telemetry"] = _attach_generation_telemetry(telemetry, row)
        rows.append(row)
        if progress is not None:
            if isinstance(row.get("tok_s"), (int, float)):
                progress(
                    f"[tune] {label} finished in {elapsed_s:.1f}s: {float(row['tok_s']):.2f} tok/s"
                )
            else:
                progress(
                    f"[tune] {label} failed in {elapsed_s:.1f}s: {row.get('error') or 'no token rate'}"
                )
            telemetry_line = _format_tune_telemetry_inline(row.get("telemetry"))
            if telemetry_line:
                progress(f"[tune] {label} telemetry: {telemetry_line}")
    return rows


def _tune_candidate_command(
    args: Any,
    *,
    candidate: str,
    model: str,
    output: Path,
    settings: dict[str, Any],
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "mtplx.cli",
        "tune",
        "--_candidate",
        candidate,
        "--_candidate-output",
        str(output),
        "--model",
        str(model),
        "--profile",
        str(settings.get("profile") or "performance-cold"),
        "--max-tokens",
        str(int(settings["max_tokens"])),
        "--limit",
        str(int(settings["limit"])),
        "--seed",
        str(int(settings["seed"])),
        "--depths",
        str(settings["depths"]),
        "--yes",
    ]
    cache_dir = getattr(args, "cache_dir", None)
    if cache_dir:
        command.extend(["--cache-dir", str(cache_dir)])
    if bool(getattr(args, "unsafe_force_unverified", False)):
        command.append("--unsafe-force-unverified")
    suite = settings.get("suite")
    if suite and str(suite) != TUNE_DEFAULT_SUITE:
        command.extend(["--prompt-suite", str(suite)])
    for key, flag in (
        ("base_hidden_variant", "--base-hidden-variant"),
        ("mtp_hidden_variant", "--mtp-hidden-variant"),
        ("concat_order", "--concat-order"),
        ("mtp_cache_policy", "--mtp-cache-policy"),
        ("mtp_history_policy", "--mtp-history-policy"),
        ("temperature", "--temperature"),
        ("top_p", "--top-p"),
        ("top_k", "--top-k"),
        ("draft_temperature", "--draft-temperature"),
        ("draft_top_p", "--draft-top-p"),
        ("draft_top_k", "--draft-top-k"),
        ("draft_core", "--draft-core"),
    ):
        value = settings.get(key)
        if value is not None:
            command.extend([flag, str(value)])
    return command


def _tune_candidate_label(candidate: str, control_field: str | None = "depth") -> str:
    if candidate == "ar":
        return "AR"
    if str(control_field or "depth") == "draft_block_size":
        return f"Block {candidate}"
    return f"D{candidate}"


def _tune_candidate_file_stem(
    candidate: str, control_field: str | None = "depth"
) -> str:
    if candidate == "ar":
        return "ar"
    if str(control_field or "depth") == "draft_block_size":
        return f"block{candidate}"
    return f"d{candidate}"


def _row_validations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        validation
        for row in rows
        if isinstance(row, dict)
        for validation in (row.get("validations") or [])
        if isinstance(validation, dict)
    ]


def _row_finish_reason_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = row.get("finish_reason") if isinstance(row, dict) else None
        if not isinstance(reason, str) or not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _row_hit_token_budget_count(rows: list[dict[str, Any]]) -> int:
    return sum(
        1 for row in rows if isinstance(row, dict) and row.get("hit_token_budget")
    )


def _tune_candidate_summary(
    candidate: str,
    path: Path,
    *,
    control_field: str = "depth",
    returncode: int,
    stdout_path: Path,
    command: list[str],
) -> dict[str, Any]:
    label = _tune_candidate_label(candidate, control_field)
    candidate_value = None if candidate == "ar" else int(candidate)
    base = {
        "mode": label,
        "depth": candidate_value,
        "control_field": control_field,
        "draft_block_size": candidate_value
        if control_field == "draft_block_size"
        else None,
        "candidate": candidate,
        "returncode": returncode,
        "artifact": str(path),
        "stdout": str(stdout_path),
        "command": command,
    }
    if not path.exists():
        child_error = _tune_child_error_from_stdout(stdout_path)
        if child_error is not None:
            return {**base, **child_error}
        return {**base, "error": "candidate did not write an artifact"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {**base, "error": f"candidate artifact is not valid JSON: {exc}"}
    if candidate == "ar":
        ar_rows = data.get("ar_rows") or []
        quality = summarize_benchmark_quality(
            [row for row in ar_rows if isinstance(row, dict)]
        )
        tok_values = _row_metric_values(ar_rows, "decode_tok_s", "tok_s")
        end_to_end_values = _row_metric_values(ar_rows, "end_to_end_tok_s")
        elapsed_values = _row_metric_values(ar_rows, "elapsed_s")
        prompt_values = _row_metric_values(ar_rows, "prompt_eval_time_s")
        decode_elapsed_values = _row_metric_values(ar_rows, "decode_elapsed_s")
        generation_windows = _generation_windows_from_rows(ar_rows)
        hit_token_budget_count = _row_hit_token_budget_count(ar_rows)
        return {
            **base,
            "tok_s": (sum(tok_values) / len(tok_values)) if tok_values else None,
            "decode_tok_s": (sum(tok_values) / len(tok_values)) if tok_values else None,
            "end_to_end_tok_s": (
                (sum(end_to_end_values) / len(end_to_end_values))
                if end_to_end_values
                else None
            ),
            "generated_tokens": sum(
                int(row.get("generated_tokens") or 0) for row in ar_rows
            ),
            "elapsed_s": sum(elapsed_values) if elapsed_values else None,
            "prompt_eval_time_s": sum(prompt_values) if prompt_values else None,
            "decode_elapsed_s": sum(decode_elapsed_values)
            if decode_elapsed_values
            else None,
            "generation_windows": generation_windows,
            "generation_window_s": sum(
                window["duration_s"] for window in generation_windows
            ),
            "hit_token_budget": hit_token_budget_count > 0,
            "hit_token_budget_count": hit_token_budget_count,
            "finish_reasons": _row_finish_reason_counts(ar_rows),
            "verify_ms_per_call": None,
            **quality,
        }
    depth_rows = data.get("depths") or data.get("depth_results") or []
    row = depth_rows[0] if depth_rows else {}
    summary = row.get("summary") or {}
    generation_rows = [
        result_row
        for depth_row in depth_rows
        for result_row in depth_row.get("rows", [])
        if isinstance(result_row, dict)
    ]
    prompt_values = _row_metric_values(generation_rows, "prompt_eval_time_s")
    decode_elapsed_values = _row_metric_values(generation_rows, "decode_elapsed_s")
    generation_windows = _generation_windows_from_rows(generation_rows)
    hit_token_budget_count = _row_hit_token_budget_count(generation_rows)
    verify_calls = int(summary.get("verify_calls") or 0)
    quality = summarize_benchmark_quality(generation_rows)
    acceptance = summary.get("acceptance_by_depth")
    if acceptance is None:
        acceptance = _rate_lists(
            summary.get("accepted_by_depth") or [],
            summary.get("drafted_by_depth") or [],
        )
    return {
        **base,
        "tok_s": _first_number(summary, "mean_decode_tok_s", "mean_tok_s"),
        "decode_tok_s": _first_number(summary, "mean_decode_tok_s", "mean_tok_s"),
        "end_to_end_tok_s": _first_number(summary, "mean_end_to_end_tok_s"),
        "generated_tokens": summary.get("generated_tokens"),
        "elapsed_s": summary.get("elapsed_s"),
        "prompt_eval_time_s": sum(prompt_values) if prompt_values else None,
        "decode_elapsed_s": sum(decode_elapsed_values)
        if decode_elapsed_values
        else None,
        "generation_windows": generation_windows,
        "generation_window_s": sum(
            window["duration_s"] for window in generation_windows
        ),
        "hit_token_budget": hit_token_budget_count > 0,
        "hit_token_budget_count": hit_token_budget_count,
        "finish_reasons": summary.get("finish_reasons")
        or _row_finish_reason_counts(generation_rows),
        "verify_time_s": summary.get("verify_time_s"),
        "verify_calls": verify_calls,
        "verify_ms_per_call": _ms_per_call(summary.get("verify_time_s"), verify_calls),
        "verify_forward_ms_per_call": _ms_per_call(
            summary.get("verify_forward_time_s"), verify_calls
        ),
        "verify_eval_ms_per_call": _ms_per_call(
            summary.get("verify_eval_time_s"), verify_calls
        ),
        "verify_hidden_eval_ms_per_call": _ms_per_call(
            summary.get("verify_hidden_eval_time_s"), verify_calls
        ),
        "accepted_by_depth": summary.get("accepted_by_depth"),
        "drafted_by_depth": summary.get("drafted_by_depth"),
        "acceptance_by_depth": acceptance,
        "acceptance_percent_by_depth": [
            (None if value is None else 100.0 * float(value)) for value in acceptance
        ],
        "mean_accept_probability_by_depth": summary.get(
            "mean_accept_probability_by_depth"
        ),
        **quality,
    }


def _tune_child_error_from_stdout(stdout_path: Path) -> dict[str, Any] | None:
    try:
        text = stdout_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    data = _load_first_json_object(text)
    if not isinstance(data, dict):
        return None
    error = str(data.get("error") or "").strip()
    if not error:
        return None
    detail = str(data.get("detail") or "").strip()
    model = data.get("model")
    if isinstance(model, dict):
        compatibility = (
            model.get("compatibility")
            if isinstance(model.get("compatibility"), dict)
            else {}
        )
        detail = detail or str(compatibility.get("message") or "").strip()
        model_ref = str(model.get("model_dir") or "").strip()
    else:
        model_ref = str(model or "").strip()
    summary = error
    if detail:
        summary = f"{summary}: {detail}"
    payload: dict[str, Any] = {
        "error": summary,
        "child_error": data,
    }
    if detail:
        payload["detail"] = detail
    if model_ref:
        payload["model"] = model_ref
    return payload


def _load_first_json_object(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _tune_payload(
    *,
    action: str,
    run_id: str,
    model: str,
    settings: dict[str, Any],
    rows: list[dict[str, Any]],
    output_root: Path,
    output_path: Path,
    hardware: dict[str, Any],
    software: dict[str, Any],
    backend: dict[str, Any],
    thermal: dict[str, Any],
    state_key: str,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = _annotate_multipliers(rows)
    best = _best_multiplier_summary(rows)
    return {
        "action": action,
        "run_id": run_id,
        "model": model,
        "profile": str(settings.get("profile") or "performance-cold"),
        "suite": settings["suite"],
        "control_field": settings.get("control_field") or "depth",
        "settings": settings,
        "results": rows,
        "best": best.get("winner"),
        "best_multiplier": best,
        "hardware": hardware,
        "software": software,
        "backend": backend,
        "thermal": thermal,
        "diagnostics": diagnostics or {},
        "state_key": state_key,
        "artifacts": {
            "root": str(output_root),
            "summary": str(output_path),
        },
    }


def _annotate_multipliers(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ar_tok_s = None
    for row in results:
        if row.get("mode") == "AR":
            ar_tok_s = row.get("tok_s")
            break
    annotated = []
    for row in results:
        item = dict(row)
        if item.get("mode") == "AR":
            item["multiplier_vs_ar"] = (
                1.0 if isinstance(item.get("tok_s"), (int, float)) else None
            )
        else:
            item["multiplier_vs_ar"] = item.get("speedup_vs_ar") or _safe_ratio(
                item.get("tok_s"), ar_tok_s
            )
        annotated.append(item)
    return annotated


def _best_multiplier_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    annotated = _annotate_multipliers(results)
    ar = next((row for row in annotated if row.get("mode") == "AR"), None)
    if not ar or not isinstance(ar.get("tok_s"), (int, float)):
        return {
            "available": False,
            "ar_tok_s": None,
            "winner": None,
            "verdict": "tune_failed_no_ar_result",
        }
    faster_than_ar = [
        row
        for row in annotated
        if row.get("depth") is not None
        and isinstance(row.get("multiplier_vs_ar"), (int, float))
        and float(row["multiplier_vs_ar"]) > 1.0
    ]
    quality_rejected = [
        row for row in faster_than_ar if row.get("quality_passed") is False
    ]
    acceptance_collapsed = _tune_acceptance_collapsed_rows(annotated)
    candidates = [
        row
        for row in faster_than_ar
        if row.get("quality_passed") is not False
        and not _tune_row_acceptance_collapsed(row)
    ]
    raw_winner = max(
        candidates, key=lambda row: float(row["multiplier_vs_ar"]), default=None
    )
    winner = raw_winner
    tie_margin_pct = max(
        0.0,
        _tune_env_float(
            "MTPLX_TUNE_TIE_PREFER_DEEPER_WITHIN_PCT",
            TUNE_TIE_PREFER_DEEPER_WITHIN_PCT,
        ),
    )
    if raw_winner is not None and tie_margin_pct > 0.0:
        raw_tok_s = raw_winner.get("tok_s")
        if isinstance(raw_tok_s, (int, float)) and float(raw_tok_s) > 0.0:
            floor = float(raw_tok_s) * (1.0 - tie_margin_pct / 100.0)
            tied = [
                row
                for row in candidates
                if isinstance(row.get("tok_s"), (int, float))
                and float(row["tok_s"]) >= floor
            ]
            winner = max(
                tied,
                key=lambda row: (
                    int(row.get("depth") or 0),
                    float(row.get("tok_s") or 0.0),
                ),
                default=raw_winner,
            )
    if winner is None:
        verdict = (
            "no_quality_passed_mtp_depth_beat_ar"
            if quality_rejected
            else (
                "mtp_acceptance_collapsed"
                if acceptance_collapsed
                else "no_mtp_depth_beat_ar"
            )
        )
        return {
            "available": bool(ar and isinstance(ar.get("tok_s"), (int, float))),
            "ar_tok_s": ar.get("tok_s") if ar else None,
            "winner": None,
            "quality_rejected": [
                {
                    "mode": row.get("mode"),
                    "depth": row.get("depth"),
                    "tok_s": row.get("tok_s"),
                    "multiplier_vs_ar": row.get("multiplier_vs_ar"),
                    "hit_token_budget": row.get("hit_token_budget"),
                    "hit_token_budget_count": row.get("hit_token_budget_count"),
                    "finish_reasons": row.get("finish_reasons"),
                }
                for row in quality_rejected
            ],
            "acceptance_collapsed": acceptance_collapsed,
            "failure_reasons": _tune_failure_reasons(
                annotated,
                quality_rejected=quality_rejected,
                acceptance_collapsed=acceptance_collapsed,
                winner=None,
            ),
            "verdict": verdict,
        }
    raw_winner_changed = raw_winner is not None and raw_winner.get(
        "mode"
    ) != winner.get("mode")
    return {
        "available": True,
        "ar_tok_s": ar.get("tok_s") if ar else None,
        "winner": {
            "mode": winner.get("mode"),
            "depth": winner.get("depth"),
            "control_field": winner.get("control_field"),
            "draft_block_size": winner.get("draft_block_size"),
            "tok_s": winner.get("tok_s"),
            "multiplier_vs_ar": winner.get("multiplier_vs_ar"),
        },
        "raw_winner": {
            "mode": raw_winner.get("mode"),
            "depth": raw_winner.get("depth"),
            "control_field": raw_winner.get("control_field"),
            "draft_block_size": raw_winner.get("draft_block_size"),
            "tok_s": raw_winner.get("tok_s"),
            "multiplier_vs_ar": raw_winner.get("multiplier_vs_ar"),
        }
        if raw_winner is not None
        else None,
        "quality_rejected": [
            {
                "mode": row.get("mode"),
                "depth": row.get("depth"),
                "tok_s": row.get("tok_s"),
                "multiplier_vs_ar": row.get("multiplier_vs_ar"),
                "hit_token_budget": row.get("hit_token_budget"),
                "hit_token_budget_count": row.get("hit_token_budget_count"),
                "finish_reasons": row.get("finish_reasons"),
            }
            for row in quality_rejected
        ],
        "acceptance_collapsed": acceptance_collapsed,
        "failure_reasons": _tune_failure_reasons(
            annotated,
            quality_rejected=quality_rejected,
            acceptance_collapsed=acceptance_collapsed,
            winner=winner,
        ),
        "tie_breaker": {
            "applied": raw_winner_changed,
            "prefer_deeper_within_pct": tie_margin_pct,
        },
        "verdict": "mtp_depth_wins",
    }


def _tune_row_acceptance_collapsed(row: dict[str, Any]) -> bool:
    """True when a depth row carries acceptance evidence and it is ~zero at
    every level. With zero accepted drafts every MTP depth does strictly more
    work than AR per emitted token, so a >1.0x multiplier on such a row can
    only be wall-clock noise (another workload suppressing the AR window),
    never a true result (#177)."""
    if row.get("depth") is None:
        return False
    acceptance = [
        float(value)
        for value in row.get("acceptance_by_depth") or []
        if isinstance(value, (int, float))
    ]
    if not acceptance:
        return False
    return max(acceptance) <= TUNE_ACCEPTANCE_COLLAPSE_THRESHOLD


def _tune_acceptance_collapsed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collapsed: list[dict[str, Any]] = []
    for row in rows:
        if not _tune_row_acceptance_collapsed(row):
            continue
        acceptance = [
            float(value)
            for value in row.get("acceptance_by_depth") or []
            if isinstance(value, (int, float))
        ]
        collapsed.append(
            {
                "mode": row.get("mode"),
                "depth": row.get("depth"),
                "tok_s": row.get("tok_s"),
                "multiplier_vs_ar": row.get("multiplier_vs_ar"),
                "acceptance_by_depth": acceptance,
                "quality_passed": row.get("quality_passed"),
            }
        )
    return collapsed


def _tune_failure_reasons(
    rows: list[dict[str, Any]],
    *,
    quality_rejected: list[dict[str, Any]],
    acceptance_collapsed: list[dict[str, Any]],
    winner: dict[str, Any] | None,
) -> list[str]:
    if winner is not None:
        return []
    reasons: list[str] = []
    if acceptance_collapsed:
        reasons.append("mtp_acceptance_collapsed")
    if quality_rejected:
        reasons.append("quality_failed_fast_mtp")
    if any(row.get("depth") is not None for row in rows):
        reasons.append("no_mtp_depth_beat_ar")
    return reasons


def _print_tune_dry_run_human(payload: dict[str, Any]) -> None:
    print("MTPLX Tune")
    print("dry-run: no model will be loaded")
    print(f"model: {payload.get('model')}")
    print(f"output: {payload.get('output')}")
    print("candidate commands:")
    for row in payload.get("candidates", []):
        print(f"  {row.get('candidate')}: {shlex.join(row.get('command') or [])}")


def _stat_avg(stats: Any) -> float | None:
    if isinstance(stats, dict) and isinstance(stats.get("avg"), (int, float)):
        return float(stats["avg"])
    return None


def _format_tune_telemetry_inline(telemetry: Any) -> str | None:
    if not isinstance(telemetry, dict) or not telemetry.get("enabled"):
        return None
    display = telemetry
    scope = "candidate"
    generation = telemetry.get("generation")
    if isinstance(generation, dict) and generation.get("sample_count"):
        display = generation
        scope = "generation"
    parts: list[str] = []
    power = display.get("power_w") or {}
    power_parts = []
    for label, key in (
        ("pkg", "package"),
        ("cpu", "cpu"),
        ("ane", "ane"),
        ("gpu", "gpu"),
    ):
        value = _stat_avg(power.get(key))
        if value is not None:
            power_parts.append(f"{label}={value:.1f}W")
    if power_parts:
        parts.append("power " + " ".join(power_parts))

    frequency = display.get("frequency_ghz") or {}
    frequency_parts = []
    for label, key in (("P", "p_cluster"), ("M", "m_cluster"), ("GPU", "gpu")):
        value = _stat_avg(frequency.get(key))
        if value is not None:
            frequency_parts.append(f"{label}={value:.2f}GHz")
    if frequency_parts:
        parts.append("freq " + " ".join(frequency_parts))

    temperature = display.get("temperature_c") or {}
    temp_parts = []
    core_avg = _stat_avg(temperature.get("core_avg"))
    core_max = _stat_avg(temperature.get("core_max"))
    gpu_avg = _stat_avg(temperature.get("gpu_avg"))
    if core_avg is not None:
        temp_parts.append(f"core_avg={core_avg:.1f}C")
    if core_max is not None:
        temp_parts.append(f"core_max={core_max:.1f}C")
    if gpu_avg is not None:
        temp_parts.append(f"gpu_avg={gpu_avg:.1f}C")
    if temp_parts:
        parts.append("temp " + " ".join(temp_parts))

    utilization = display.get("utilization_pct") or {}
    utilization_parts = []
    for label, key in (("P", "p_core"), ("M", "m_core"), ("GPU", "gpu")):
        value = _stat_avg(utilization.get(key))
        if value is not None:
            utilization_parts.append(f"{label}={value:.1f}%")
    if utilization_parts:
        parts.append("util " + " ".join(utilization_parts))

    fans = display.get("fans_rpm") or {}
    fan_avg = _stat_avg(fans.get("avg"))
    if fan_avg is not None:
        parts.append(f"fans={fan_avg:.0f}rpm")

    sample_count = display.get("sample_count")
    if isinstance(sample_count, int):
        parts.append(f"samples={sample_count}")
    window_duration = display.get("window_duration_s")
    if scope == "generation" and isinstance(window_duration, (int, float)):
        parts.append(f"window={float(window_duration):.1f}s")
    errors = display.get("errors") or telemetry.get("errors") or []
    if parts and errors:
        parts.append("notes=" + "; ".join(str(error) for error in errors[:2]))
    if not parts:
        if errors:
            return f"unavailable ({'; '.join(str(error) for error in errors[:2])})"
        return None
    parts.insert(0, f"scope={scope}")
    return " | ".join(parts)


def _print_tune_human(payload: dict[str, Any], *, verbose: bool = False) -> None:
    print("MTPLX Tune")
    if payload.get("from_cache"):
        print("Using saved tuning. Run `mtplx tune --retune` to measure again.")
    else:
        artifacts = payload.get("artifacts") or {}
        if artifacts.get("root"):
            print(f"Results written to {artifacts.get('root')}")
    print()
    best = payload.get("best") or {}
    for row in payload.get("results", []):
        mode = str(row.get("mode") or "?")
        tok_s = _fmt_metric(row.get("tok_s"), digits=1)
        multiplier = _fmt_metric(row.get("multiplier_vs_ar"), digits=2)
        marker = "  BEST" if best.get("mode") == mode else ""
        print(f"{mode:<4} {tok_s:>6} tok/s   {multiplier}x{marker}")
    print("Speed shown is decode tok/s; prefill is tracked separately.")
    print()
    errors = [row for row in payload.get("results", []) if row.get("error")]
    best_multiplier = payload.get("best_multiplier") or {}
    quality_rejected = best_multiplier.get("quality_rejected") or []
    if quality_rejected:
        modes = ", ".join(
            (
                str(row.get("mode") or f"D{row.get('depth')}")
                + (" (hit token budget)" if row.get("hit_token_budget") else "")
            )
            for row in quality_rejected
        )
        print(f"Rejected quality-failing MTP depths: {modes}")
    if errors:
        print("Tune failed for one or more candidates:")
        for row in errors:
            print(f"  {row.get('mode')}: {row.get('error')} (log: {row.get('stdout')})")
    elif best:
        print(
            f"Best for this Mac: {best.get('mode')}, "
            f"{_fmt_metric(best.get('multiplier_vs_ar'), digits=2)}x AR"
        )
        tie_breaker = best_multiplier.get("tie_breaker") or {}
        raw_winner = best_multiplier.get("raw_winner") or {}
        if tie_breaker.get("applied") and raw_winner.get("mode"):
            print(
                f"Tie-break: {best.get('mode')} was within "
                f"{_fmt_metric(tie_breaker.get('prefer_deeper_within_pct'), digits=1)}% "
                f"of raw fastest {raw_winner.get('mode')}; preferred the deeper depth."
            )
    else:
        verdict = str(best_multiplier.get("verdict") or "")
        if verdict == "no_quality_passed_mtp_depth_beat_ar":
            print("No quality-passed MTP depth beat AR on this run")
        elif verdict == "mtp_acceptance_collapsed":
            print("No MTP depth beat AR; draft acceptance collapsed")
        else:
            print("No MTP depth beat AR on this run")
    cleared = payload.get("cleared_saved_record") or {}
    if cleared:
        previous = cleared.get("previous_best") or {}
        previous_label = previous.get("mode") or (
            f"D{previous.get('depth')}"
            if previous.get("depth") is not None
            else "record"
        )
        print(
            f"Cleared saved default {previous_label}: "
            "this run's measured verdict supersedes the stored winner."
        )
    if payload.get("saved") and best:
        control_field = str(payload.get("control_field") or "").strip()
        control_label = (
            "draft block" if control_field == "draft_block_size" else "depth"
        )
        print(
            f"Saved: Web UI starts will use {control_label} {best.get('depth')} for this model."
        )
    elif payload.get("save_skipped_reason"):
        print(f"Not saved: {payload.get('save_skipped_reason')}.")
    if verbose:
        print()
        for row in payload.get("results", []):
            if row.get("depth") is not None:
                print(
                    f"{row.get('mode')}: verify_ms={_fmt_metric(row.get('verify_ms_per_call'), digits=2)} "
                    f"acceptance={_format_depth_acceptance(row)}"
                )
            telemetry_line = _format_tune_telemetry_inline(row.get("telemetry"))
            if telemetry_line:
                print(f"{row.get('mode')}: telemetry={telemetry_line}")
        artifacts = payload.get("artifacts") or {}
        if artifacts:
            print(f"artifacts: {artifacts.get('root')}")


def _tune_error(
    message: str,
    *,
    detail: str | None = None,
    actionable: str | None = None,
    thermal: dict[str, Any] | None = None,
    json_output: bool = False,
) -> int:
    payload: dict[str, Any] = {"error": message}
    if detail:
        payload["detail"] = detail
    if actionable:
        payload["actionable"] = actionable
    if thermal is not None:
        payload["thermal"] = thermal
    if json_output:
        _print(payload)
    else:
        print(f"error: {message}", file=sys.stderr)
        if detail:
            print(f"detail: {detail}", file=sys.stderr)
        if actionable:
            print(f"action: {actionable}", file=sys.stderr)
    return 1


_PULL_NETWORK_FAILURE_MARKERS = (
    "timed out",
    "timeout",
    "connection",
    "network",
    "name resolution",
    "unreachable",
    "max retries",
    "cannot find the requested files in the local cache",
)


def _pull_failure_hint(exc: BaseException) -> str | None:
    """Mirror hint for network-shaped pull failures (#259, #96).

    ``mtplx pull`` goes through ``huggingface_hub``, which honors
    ``HF_ENDPOINT`` natively; the app exposes the same knob as Settings ->
    Advanced -> HF download mirror. Users on networks where huggingface.co
    is blocked only ever see the raw connection error, so name the knob at
    the point of failure. Silent when an endpoint is already configured
    (the mirror itself is what failed) or when the failure is not
    network-shaped (a placeholder repo with missing shards, a bad repo id).
    """
    if os.environ.get("HF_ENDPOINT", "").strip():
        return None
    text = str(exc).lower()
    if not any(marker in text for marker in _PULL_NETWORK_FAILURE_MARKERS):
        return None
    return (
        "Hugging Face was unreachable. If huggingface.co is blocked on your "
        "network, set HF_ENDPOINT=https://hf-mirror.com and rerun "
        "(in the app: Settings -> Advanced -> HF download mirror)."
    )


def cmd_pull_public(args: Any) -> int:
    from mtplx.hf_loader import pull_model, repo_id_from_model_ref

    json_mode = bool(getattr(args, "json", False))
    progress_json = bool(getattr(args, "progress_json", False))
    callback = None

    def finalize() -> None:
        return None

    progress_interval_s = 10.0

    def emit_progress_json(event: dict[str, Any]) -> None:
        print(json.dumps(event, sort_keys=True), flush=True)

    if progress_json:
        repo_id = repo_id_from_model_ref(args.model) or args.model
        callback = emit_progress_json
        progress_interval_s = 0.4
        emit_progress_json({"event": "resolving", "repo_id": repo_id})
    elif not json_mode:
        callback, finalize = _rich_download_progress_callback(
            repo_id=repo_id_from_model_ref(args.model) or args.model,
        )
        progress_interval_s = 0.4
    try:
        result = pull_model(
            args.model,
            cache_dir=args.cache_dir,
            revision=args.revision,
            progress_callback=callback,
            progress_interval_s=progress_interval_s,
        )
    except KeyboardInterrupt:
        finalize()
        if progress_json:
            emit_progress_json({"event": "cancelled", "model": args.model})
        else:
            print("download cancelled")
        return 130
    except Exception as exc:
        finalize()
        hint = _pull_failure_hint(exc)
        if progress_json:
            emit_progress_json(
                {
                    "event": "failed",
                    "error": "pull_failed",
                    "model": args.model,
                    "message": str(exc),
                    "detail": str(exc),
                }
            )
        elif json_mode:
            payload = {"error": "pull failed", "model": args.model, "detail": str(exc)}
            if hint:
                payload["hint"] = hint
            _print(payload)
        else:
            print("error: pull failed")
            print(f"model: {args.model}")
            print(f"detail: {exc}")
            if hint:
                print(f"hint: {hint}")
        return 1
    finalize()
    if progress_json:
        emit_progress_json({"event": "result", **result})
    elif json_mode:
        _print(result)
    else:
        print("MTPLX pull")
        print(f"model: {result.get('repo_id')}")
        print(f"path: {result.get('path')}")
        print(f"size: {_format_bytes(result.get('size_bytes'))}")
        print(
            f"runtime contract: {str(bool(result.get('has_runtime_contract'))).lower()}"
        )
    return 0


def _cmd_models_check(args: Any) -> int:
    from mtplx.hf_loader import model_cache_dir
    from mtplx.model_updates import (
        ENGINE_VERSION,
        STATE_UPDATE_AVAILABLE,
        check_model_updates,
    )

    rows = check_model_updates(cache_dir=args.cache_dir)
    stale = [row for row in rows if row.state == STATE_UPDATE_AVAILABLE]
    payload = {
        "cache_dir": str(model_cache_dir(args.cache_dir)),
        "engine_version": ENGINE_VERSION,
        "updates_available": len(stale),
        "models": [row.to_dict() for row in rows],
    }
    if getattr(args, "json", False):
        _print(payload)
        return 0
    print("MTPLX model updates")
    print(f"cache: {payload['cache_dir']}")
    if not rows:
        print("no tracked models (pull a model to start update tracking)")
        return 0
    for row in rows:
        local = (row.local_revision or "untracked")[:10]
        remote = (row.remote_revision or "unknown")[:10]
        line = f"- {row.repo_id}  {row.state}  {local} -> {remote}"
        if row.update_bytes:
            line += f"  ({_format_bytes(row.update_bytes)})"
        print(line)
        if row.note:
            print(f"  {row.note}")
        if row.state == "engine-update-required" and row.min_engine_version:
            print(f"  requires MTPLX >= {row.min_engine_version}")
    if stale:
        print(f"updates available: {len(stale)} — run: mtplx models --update")
    else:
        print("all tracked packs are current")
    return 0


def _cmd_models_update(args: Any, targets: list[str]) -> int:
    from mtplx.model_updates import (
        STATE_UPDATE_AVAILABLE,
        check_model_updates,
        fetch_models_manifest,
        update_cached_model,
    )

    json_mode = bool(getattr(args, "json", False))
    progress_json = bool(getattr(args, "progress_json", False))
    installed_path = getattr(args, "installed_path", None)
    downloaded_by_repo: dict[str, int] = {}

    def emit_progress_json(event: dict[str, Any]) -> None:
        payload = dict(event)
        event_repo = payload.get("repo_id")
        if isinstance(event_repo, str):
            downloaded = downloaded_by_repo.get(event_repo, 0)
            if payload.get("event") == "progress":
                downloaded += max(0, int(payload.get("delta_bytes") or 0))
                downloaded_by_repo[event_repo] = downloaded
            payload["downloaded_bytes"] = downloaded
        print(json.dumps(payload, sort_keys=True), flush=True)

    manifest = fetch_models_manifest()
    if targets:
        repos = list(dict.fromkeys(targets))
    else:
        rows = check_model_updates(cache_dir=args.cache_dir, manifest=manifest)
        repos = [row.repo_id for row in rows if row.state == STATE_UPDATE_AVAILABLE]
        if not repos:
            if progress_json:
                emit_progress_json({"event": "result", "updated": []})
            elif json_mode:
                _print({"updated": [], "message": "all tracked packs are current"})
            else:
                print("all tracked packs are current")
            return 0
    if installed_path and len(repos) != 1:
        message = "--installed-path requires exactly one --update REPO"
        if progress_json:
            emit_progress_json({"event": "failed", "error": "invalid_request", "message": message})
        else:
            print(f"error: {message}", file=sys.stderr)
        return 2
    results: list[dict[str, Any]] = []
    failed = False
    for repo in repos:
        callback = None
        finalize: Callable[[], None] = lambda: None  # noqa: E731
        if progress_json:
            # The app's update stream parses the same event schema as
            # `pull --progress-json`; update_cached_model forwards these
            # straight from pull_model, so the shapes match by construction.
            callback = emit_progress_json
            emit_progress_json({"event": "resolving", "repo_id": repo})
        elif not json_mode:
            print(f"updating {repo}")
            callback, finalize = _rich_download_progress_callback(repo_id=repo)
        try:
            result = update_cached_model(
                repo,
                cache_dir=args.cache_dir,
                destination_path=installed_path,
                manifest=manifest,
                progress_callback=callback,
                progress_interval_s=0.4 if callback else 10.0,
            )
        except Exception as exc:
            finalize()
            failed = True
            results.append({"repo_id": repo, "error": str(exc)})
            if progress_json:
                emit_progress_json(
                    {
                        "event": "failed",
                        "error": "update_failed",
                        "model": repo,
                        "message": str(exc),
                        "detail": str(exc),
                    }
                )
            elif not json_mode:
                print(f"error: update failed for {repo}: {exc}")
            continue
        finalize()
        row = {
            "repo_id": result.get("repo_id", repo),
            "path": result.get("path"),
            "resolved_sha": result.get("resolved_sha"),
            "delta_bytes": max(
                0,
                int(result.get("size_bytes") or 0)
                - int(result.get("started_size_bytes") or 0),
            ),
        }
        results.append(row)
        if progress_json:
            emit_progress_json({"event": "result", **row})
        elif not json_mode:
            print(f"updated {repo} -> {str(result.get('resolved_sha') or 'unknown')[:10]}")
    if json_mode and not progress_json:
        _print({"updated": results})
    return 1 if failed else 0


def cmd_list_public(args: Any) -> int:
    from mtplx.hf_loader import list_cached_models, model_cache_dir

    update_targets = getattr(args, "update", None)
    if update_targets is not None:
        return _cmd_models_update(args, update_targets)
    if getattr(args, "check", False):
        return _cmd_models_check(args)
    models = [row.to_dict() for row in list_cached_models(cache_dir=args.cache_dir)]
    payload = {"cache_dir": str(model_cache_dir(args.cache_dir)), "models": models}
    if getattr(args, "json", False):
        _print(payload)
    else:
        print("MTPLX models")
        print(f"cache: {payload['cache_dir']}")
        if not models:
            print("no cached models")
        for row in models:
            print(
                f"- {row.get('repo_id')}  "
                f"{_format_bytes(row.get('size_bytes'))}  "
                f"contract={str(bool(row.get('has_runtime_contract'))).lower()}"
            )
            print(f"  {row.get('path')}")
    return 0


def cmd_remove_public(args: Any) -> int:
    from mtplx.hf_loader import remove_cached_model

    result = remove_cached_model(args.model, cache_dir=args.cache_dir)
    if getattr(args, "json", False):
        _print(result)
    else:
        print("MTPLX remove")
        print(f"model: {result.get('repo_id')}")
        print(f"path: {result.get('path')}")
        if result.get("removed"):
            print(f"removed: {_format_bytes(result.get('size_bytes_removed'))}")
        else:
            print("removed: false")
    return 0 if result["removed"] or args.missing_ok else 1


def _cmd_bench_run(args: Any) -> int:
    model = args.model or DEFAULT_CHAMPION
    suite = args.suite or "default"
    selected_profile = get_profile(_bench_run_profile_name(args, suite=suite))
    args.profile = selected_profile.name
    prompt_suite = prompt_suite_path(suite)
    run_id = args.run_id or f"cli-bench-{suite}-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir = Path(args.output_dir or "outputs/cli/bench") / run_id
    output = Path(args.output) if args.output else output_dir / "depth-sweep.json"
    envelope_output = output_dir / "envelope.json"
    decode_trace = output_dir / "decode-trace.jsonl"
    exact_paged_env = _exact_paged_env_from_args(args)
    runtime_profile = selected_profile.runtime_profile
    runtime_env = selected_profile.env_dict()
    if selected_profile.name in {"safe", "exact", "max-diagnostic"}:
        runtime_env.update(exact_paged_env)
    runtime_env = _runtime_env_with_external_overrides(runtime_env)
    harness = getattr(args, "harness", "auto")
    if harness == "auto":
        harness = (
            "depth-sweep"
            if selected_profile.name == "performance-cold"
            else "direct-http"
        )
    benchmark_seed = _benchmark_seed(
        args, runtime_profile=runtime_profile, harness=harness
    )
    depths = _depths_for_bench_run(args)

    if args.dry_run:
        direct_command = _direct_http_bench_command(
            args,
            model=model,
            suite=suite,
            run_id=run_id,
            output_dir=output_dir,
            seed=benchmark_seed,
        )
        _print(
            {
                "dry_run": True,
                "action": f"bench {getattr(args, 'bench_action', None) or 'run'}",
                "model": model,
                "suite": suite,
                "prompt_suite": prompt_suite,
                "exactness_smoke": {
                    "context": 2048,
                    "automatic": True,
                    "profile": _exactness_profile_kwargs(args),
                },
                "harness": harness,
                "depths": depths if harness == "depth-sweep" else None,
                "compare_ar": bool(getattr(args, "compare_ar", False))
                if harness == "depth-sweep"
                else None,
                "seed": benchmark_seed,
                "profile": selected_profile.to_dict(),
                "runtime_profile": runtime_profile,
                "runtime_env": runtime_env,
                "direct_http_command": direct_command
                if harness == "direct-http"
                else None,
                "exact_paged_env": exact_paged_env,
                "output": str(output),
                "envelope": str(envelope_output),
                "decode_trace": str(decode_trace),
            }
        )
        return 0
    runtime_model, resolve_error = _resolve_runtime_model_path(
        model,
        cache_dir=getattr(args, "cache_dir", None),
    )
    if resolve_error is not None:
        _print(resolve_error)
        return 1
    inspection, gate_exit = _model_gate(
        runtime_model,
        unsafe_force_unverified=bool(getattr(args, "unsafe_force_unverified", False)),
        yes=bool(getattr(args, "yes", False)),
    )
    if gate_exit is not None:
        _print({"error": "model failed MTP primary gate", "model": inspection})
        return gate_exit
    if (inspection.get("compatibility") or {}).get(
        "runtime_compatibility"
    ) == "native-ar-only":
        _print(
            {
                "error": "bench run requires an MTP-capable runtime",
                "detail": "Laguna-S-2.1 is target-only AR; use mtplx run --no-mtp",
            }
        )
        return EXIT_UNSUPPORTED_MODEL
    runtime_env = _runtime_env_with_model_contract_overrides(
        runtime_env,
        inspection,
        selected_profile,
    )
    draft_lm_head = _model_draft_lm_head_spec(inspection, selected_profile)
    draft_sampler = _model_draft_sampler_spec(inspection, selected_profile)

    from mtplx.benchmarks.runners.preflight import run_preflight

    preflight = run_preflight(".")
    smoke = run_exactness_smoke(
        runtime_model,
        context=2048,
        prompt_suite=prompt_suite_path("flappy"),
        output=output_dir / "exactness-smoke.json",
        **_exactness_profile_kwargs(args),
    )
    if not smoke["passed"]:
        write_json(
            envelope_output,
            {"run_id": run_id, "correctness": {"exactness_smoke": smoke}},
        )
        _print({"error": "Phase 0H exactness smoke failed", "exactness_smoke": smoke})
        return EXIT_EXACTNESS

    if harness == "direct-http":
        return _cmd_bench_run_direct_http(
            args,
            model=runtime_model,
            suite=suite,
            run_id=run_id,
            output_dir=output_dir,
            envelope_output=envelope_output,
            exactness_smoke=smoke,
            runtime_profile=runtime_profile,
            runtime_env=runtime_env,
            preflight=preflight,
            model_inspection=inspection,
            seed=benchmark_seed,
        )

    from mtplx.benchmarks.runners.mtp_depth_sweep import write_depth_sweep

    decode_trace.parent.mkdir(parents=True, exist_ok=True)
    with _temporary_env(
        {
            "MTPLX_DECODE_TRACE_JSONL": str(decode_trace),
            "MTPLX_DECODE_TRACE_INTERVAL_S": str(args.trace_interval_s),
            "MTPLX_DECODE_TRACE_LABEL": run_id,
            **runtime_env,
        }
    ):
        result = _depth_sweep_native60(
            model=runtime_model,
            prompt_suite=prompt_suite,
            depths=depths,
            max_tokens=args.max_tokens,
            limit=args.limit,
            seed=benchmark_seed,
            temperature=float(getattr(args, "temperature", 0.6)),
            top_p=float(getattr(args, "top_p", 0.95)),
            top_k=int(getattr(args, "top_k", 20)),
            draft_lm_head=draft_lm_head,
            draft_sampler=draft_sampler,
            compare_ar=bool(getattr(args, "compare_ar", False)),
            runtime_env=runtime_env,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_depth_sweep(output, result)
    envelope = build_benchmark_envelope(
        result=result,
        model_inspection=inspection,
        run_id=run_id,
        suite=suite,
        runtime_profile=runtime_profile,
        runtime_env=runtime_env,
        exactness_smoke=smoke,
        fan_controlled=bool(args.fanmax),
        strict=bool(args.strict),
        strict_cold=bool(args.strict_cold),
        telemetry={
            "telemetry_unavailable": False,
            "source": "bench-preflight",
            "power": preflight.get("power"),
            "issues": preflight.get("issues"),
            "deep_smc_trace_attached": False,
        },
        decode_trace_path=decode_trace,
    )
    envelope["artifacts"] = {
        "depth_sweep": str(output),
        "envelope": str(envelope_output),
        "decode_trace": str(decode_trace),
    }
    write_json(envelope_output, envelope)
    _print(_bench_run_console_summary(envelope))

    if not envelope["quality"]["passed"]:
        return EXIT_QUALITY
    if args.strict or args.strict_cold:
        if envelope.get("strict_passed") is False:
            return EXIT_STRICT_GATE
    return 0


def _bench_run_profile_name(args: Any, *, suite: str) -> str:
    requested = getattr(args, "profile", None)
    if requested:
        return str(requested)
    # Founder order (2026-08-16, redp314 board): our flagship models default
    # to turbo across EVERY feature — context suites included. The old bare
    # sustained defaults here meant exactly the people producing public
    # numbers benchmarked the slow profile. Explicit --profile always wins;
    # non-flagship models keep the memory-safe sustained defaults below.
    resolved = _resolved_default_profile_name(args)
    if resolved == "turbo":
        return "turbo"
    if suite in BENCH_SUSTAINED_DEFAULT_SUITES:
        return "sustained"
    try:
        max_tokens = int(getattr(args, "max_tokens", 0) or 0)
    except (TypeError, ValueError):
        max_tokens = 0
    if (
        suite in BENCH_SUSTAINED_LENGTH_SENSITIVE_SUITES
        and max_tokens > BENCH_SUSTAINED_MAX_TOKENS_THRESHOLD
    ):
        return "sustained"
    return resolved


def _direct_http_bench_command(
    args: Any,
    *,
    model: str,
    suite: str,
    run_id: str,
    output_dir: Path,
    seed: int,
) -> list[str]:
    if suite in {"python_modules_long", "python-modules-long"}:
        test_name = "python_modules_long"
    elif suite in {"long_code_uncapped", "long-code-uncapped"}:
        test_name = "long_code_uncapped"
    elif suite in {"long_code", "long-code", "cold-long-code-192"}:
        test_name = "long_code"
    else:
        test_name = suite
    if test_name not in {
        "flappy",
        "python_modules_long",
        "long_code_uncapped",
        "long_code",
    }:
        test_name = "flappy"
    generation_mode = _generation_mode_from_args(args)
    load_mtp_flag = (
        "--no-load-mtp" if bool(getattr(args, "stock_ar", False)) else "--load-mtp"
    )
    ablation_profile = (
        "sustained"
        if getattr(args, "profile", None) == "sustained"
        else LONG_RESPONSE_DIRECT_PROFILE
    )
    command = [
        sys.executable,
        str(repo_root() / "scripts" / "run_context_degradation_diagnostics.py"),
        "local-ablation",
        "--label",
        run_id,
        "--run-id",
        run_id,
        "--output-dir",
        str(output_dir / "direct-http"),
        "--port",
        str(getattr(args, "port", 8041) or 8041),
        "--model",
        model,
        "--model-id",
        Path(model).name,
        "--generation-mode",
        generation_mode,
        load_mtp_flag,
        "--depth",
        "3",
        "--verify-strategy",
        "capture_commit",
        "--verify-core",
        "linear-gdn-from-conv-tape",
        "--profiles",
        ablation_profile,
        "--tests",
        test_name,
        "--temperature",
        "0.6",
        "--top-p",
        "0.95",
        "--top-k",
        "20",
        "--seed",
        str(seed),
        "--trace-interval-s",
        str(getattr(args, "trace_interval_s", 1.0)),
        "--python-bin",
        sys.executable,
        "--request-timeout-s",
        "2400",
        "--startup-timeout-s",
        "600",
    ]
    max_tokens = getattr(args, "max_tokens", None)
    if max_tokens is not None and test_name != "long_code_uncapped":
        command.extend(["--max-tokens", str(int(max_tokens))])
    headers_json = getattr(args, "headers_json", None)
    if headers_json:
        command.extend(["--headers-json", str(headers_json)])
    metadata_json = getattr(args, "metadata_json", None)
    if metadata_json:
        command.extend(["--metadata-json", str(metadata_json)])
    cache_mode = getattr(args, "cache_mode", None)
    if cache_mode and str(cache_mode) != "default":
        command.extend(["--cache-mode", str(cache_mode)])
    if not bool(getattr(args, "strict_mlx_fork_assert", False)):
        command.append("--no-strict-mlx-fork-assert")
    return command


def _validation_dicts_for_text(text: str, *, suite: str) -> list[dict[str, Any]]:
    validations = [
        validate_no_degenerate_loop(text),
        validate_balanced_delimiters(text),
    ]
    if suite in {
        "python_modules_long",
        "python-modules-long",
        "long_code",
        "long-code",
        "cold-long-code-192",
    }:
        validations.append(validate_python_syntax(text))
    return [validation.__dict__ for validation in validations]


def _cmd_bench_run_direct_http(
    args: Any,
    *,
    model: str,
    suite: str,
    run_id: str,
    output_dir: Path,
    envelope_output: Path,
    exactness_smoke: dict[str, Any],
    runtime_profile: str,
    runtime_env: dict[str, str],
    preflight: dict[str, Any],
    model_inspection: dict[str, Any],
    seed: int,
) -> int:
    command = _direct_http_bench_command(
        args,
        model=model,
        suite=suite,
        run_id=run_id,
        output_dir=output_dir,
        seed=seed,
    )
    proc = subprocess.run(
        command,
        cwd=repo_root(),
        env={**os.environ, **runtime_env},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    context_summary = output_dir / "direct-http" / run_id / "ablation-summary.json"
    row: dict[str, Any] = {}
    if context_summary.exists():
        summary = json.loads(context_summary.read_text(encoding="utf-8"))
        rows = summary.get("rows") or []
        row = dict(rows[0]) if rows else {}
    row_error = row.get("error") if isinstance(row.get("error"), dict) else None
    text = ""
    content_path = row.get("content_path")
    if content_path and Path(content_path).exists():
        text = Path(content_path).read_text(encoding="utf-8", errors="replace")
    validations = _validation_dicts_for_text(text, suite=suite) if text else []
    quality_failures = [
        {
            "prompt_id": row.get("test") or row.get("suite") or suite,
            "validation": validation.get("name"),
            "detail": validation.get("detail", ""),
        }
        for validation in validations
        if not validation.get("passed")
    ]
    if row_error is not None:
        quality_failures.append(
            {
                "prompt_id": row.get("test") or row.get("suite") or suite,
                "validation": "bench_runtime_error",
                "detail": str(row_error.get("message") or row_error),
            }
        )
    trace_summary = row.get("trace_summary") or {}
    runtime = {
        "generated_tokens": row.get("completion_tokens"),
        "elapsed_s": row.get("request_elapsed_s"),
        "tok_s": row.get("decode_tok_s"),
        "mean_tok_s": row.get("decode_tok_s"),
        "verify_ms_per_call": None,
        "late_verify_ms": row.get("late_verify_ms"),
        "accepted_by_depth": None,
        "drafted_by_depth": None,
        "mean_accept_probability_by_depth": None,
        "acceptance_by_depth": None,
        "correction_tokens": None,
        "bonus_tokens": None,
    }
    trace = {
        "path": trace_summary.get("trace_path"),
        "available": bool(trace_summary),
        "buckets": trace_summary.get("trace_rows"),
        "positive_buckets": trace_summary.get("trace_usable_rows"),
        "generated_tokens": trace_summary.get("trace_generated_tokens")
        or row.get("completion_tokens"),
        "first64_tok_s": row.get("first64"),
        "last64_tok_s": row.get("last64"),
        "last64_over_first64": row.get("last64_over_first64"),
        "first10_tok_s": row.get("first10_tok_s"),
        "last10_tok_s": row.get("last10_tok_s"),
        "last10_over_first10": row.get("last10_over_first10"),
        "late_verify_ms": row.get("late_verify_ms"),
        "cache_gib_last": row.get("last_cache_gib"),
    }
    strict_gates: dict[str, bool] = {}
    if args.strict:
        strict_gates = {
            "flappy_tok_s_ge_50": bool(
                runtime["tok_s"] is not None and float(runtime["tok_s"]) >= 50.0
            ),
            "last64_over_first64_ge_0_90": bool(
                trace["last64_over_first64"] is not None
                and float(trace["last64_over_first64"]) >= 0.90
            ),
            "last10_over_first10_ge_0_85": bool(
                trace["last10_over_first10"] is not None
                and float(trace["last10_over_first10"]) >= 0.85
            ),
            "late_verify_le_75ms": bool(
                trace["late_verify_ms"] is not None
                and float(trace["late_verify_ms"]) <= 75.0
            ),
            "telemetry_available": False,
            "no_fan_control": not bool(args.fanmax),
        }
    envelope = {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "suite": suite,
        "model": model_inspection,
        "fan_controlled": bool(args.fanmax),
        "harness": "direct-http",
        "runtime_profile": runtime_profile,
        "runtime_env": runtime_env,
        "fast_path_env": row.get("fast_path_env"),
        "runtime": runtime,
        "decode_trace": trace,
        "thermal": {
            "telemetry_unavailable": False,
            "source": "bench-preflight",
            "power": preflight.get("power"),
            "issues": preflight.get("issues"),
            "deep_smc_trace_attached": False,
        },
        "dispatch": {
            "dispatch_trace_attached": False,
            "command_buffers_per_token": None,
        },
        "quality": {
            "passed": not quality_failures,
            "failures": quality_failures,
            "validations": validations,
            "acceptance_smells": [],
        },
        "correctness": {
            "exactness_smoke": exactness_smoke,
            "full_exactness": None,
            "distribution_exactness": None,
        },
        "strict_gates": strict_gates,
        "strict_passed": all(strict_gates.values()) if strict_gates else None,
        "artifacts": {
            "context_summary": str(context_summary),
            "envelope": str(envelope_output),
            "server_stdout": str(output_dir / "direct-http-command.log"),
            "content": content_path,
            "events": row.get("events_path"),
        },
        "direct_returncode": proc.returncode,
        "error": row_error,
    }
    (output_dir / "direct-http-command.log").write_text(proc.stdout, encoding="utf-8")
    write_json(envelope_output, envelope)
    _print(_bench_run_console_summary(envelope))
    if (
        proc.returncode != 0
        or row_error is not None
        or runtime["generated_tokens"] is None
    ):
        return EXIT_STRICT_GATE
    if not envelope["quality"]["passed"]:
        return EXIT_QUALITY
    if args.strict and envelope.get("strict_passed") is False:
        return EXIT_STRICT_GATE
    return 0


def _suite_default_profile_name(args: Any, *, model: str) -> str:
    """Launch-rule profile for suite tasks built without an explicit flag.

    Suite builders set ``child.profile`` explicitly, which bypasses the
    serve-time per-model resolution — so the default must be resolved HERE,
    against the model the suite will actually run, or the flagships silently
    benchmark sustained (the 25.3 tok/s shape). An explicit --profile always
    wins via ``_resolved_default_profile_name``.
    """

    return get_profile(_resolved_default_profile_name(args, model)).name


def _nightly_tasks(args: Any, *, model: str) -> list[dict[str, Any]]:
    default_profile = _suite_default_profile_name(args, model=model)
    return [
        {
            "label": "cold-long-code-192",
            "suite": "cold-long-code-192",
            "max_tokens": 192,
            "profile": "performance-cold",
            "strict": False,
            "strict_cold": True,
            "harness": "auto",
        },
        {
            "label": "flappy-6k",
            "suite": "flappy",
            "max_tokens": 6000,
            "profile": default_profile,
            "strict": bool(getattr(args, "strict", False)),
            "strict_cold": False,
            "harness": "direct-http",
        },
        {
            "label": "flappy-10k",
            "suite": "flappy",
            "max_tokens": 10000,
            "profile": default_profile,
            "strict": bool(getattr(args, "strict", False)),
            "strict_cold": False,
            "harness": "direct-http",
        },
        {
            "label": "python-modules-6k",
            "suite": "python_modules_long",
            "max_tokens": 6000,
            "profile": default_profile,
            "strict": False,
            "strict_cold": False,
            "harness": "direct-http",
        },
    ]


def _bench_suite_is_quick(args: Any) -> bool:
    return bool(getattr(args, "quick", False))


def _client_contract_task(
    label: str, client: str, *, max_tokens: int, profile: str
) -> dict[str, Any]:
    return {
        "label": label,
        "suite": "flappy",
        "max_tokens": max_tokens,
        "profile": profile,
        "strict": False,
        "strict_cold": False,
        "harness": "direct-http",
        "category": "client_contract",
        "client": client,
        "headers": {"x-mtplx-client": client},
        "metadata": {
            "mtplx_bench_client": client,
            "mtplx_bench_lane": "quick-suite-client-contract",
        },
        "warn_gates": {
            "tok_s_ge": 35.0,
            "last64_over_first64_ge": 0.80,
        },
        "must": [
            "request client hint reaches the app-compatible server path",
            "quality validators pass",
            "no hidden cap or malformed output regression in the response text",
        ],
    }


def _quick_suite_tasks(args: Any, *, model: str) -> list[dict[str, Any]]:
    default_profile = _suite_default_profile_name(args, model=model)
    return [
        {
            "label": "short-context-384",
            "suite": "flappy",
            "max_tokens": 384,
            "profile": default_profile,
            "strict": False,
            "strict_cold": False,
            "harness": "direct-http",
            "category": "short_context_speed",
            "warn_gates": {
                "tok_s_ge": 45.0,
                "last64_over_first64_ge": 0.85,
            },
            "must": [
                "quality validators pass",
                "first useful app-style response stays above the comfort floor",
            ],
        },
        {
            "label": "long-tool-history-1536",
            "suite": "python_modules_long",
            "max_tokens": 1536,
            "profile": default_profile,
            "strict": False,
            "strict_cold": False,
            "harness": "direct-http",
            "category": "long_tool_history",
            "warn_gates": {
                "tok_s_ge": 35.0,
                "last64_over_first64_ge": 0.80,
                "late_verify_ms_le": 120.0,
            },
            "must": [
                "large coding prompt validates cleanly",
                "late verify cost does not collapse the tail",
            ],
        },
        _client_contract_task(
            "opencode-contract-1024",
            "opencode",
            max_tokens=1024,
            profile=default_profile,
        ),
        _client_contract_task(
            "pi-contract-1024", "pi", max_tokens=1024, profile=default_profile
        ),
        _client_contract_task(
            "hermes-contract-1024", "hermes", max_tokens=1024, profile=default_profile
        ),
    ]


def _bench_suite_tasks(args: Any, *, model: str) -> list[dict[str, Any]]:
    return (
        _quick_suite_tasks(args, model=model)
        if _bench_suite_is_quick(args)
        else _nightly_tasks(args, model=model)
    )


def _bench_suite_exactness_contexts(args: Any) -> str:
    configured = str(
        getattr(args, "nightly_exactness_contexts", BENCH_SUITE_FULL_EXACTNESS_CONTEXTS)
        or BENCH_SUITE_FULL_EXACTNESS_CONTEXTS
    )
    if (
        _bench_suite_is_quick(args)
        and configured == BENCH_SUITE_FULL_EXACTNESS_CONTEXTS
    ):
        return BENCH_SUITE_QUICK_EXACTNESS_CONTEXTS
    return configured


def _apply_bench_suite_task(child: Any, task: dict[str, Any]) -> None:
    child.suite = task["suite"]
    child.max_tokens = task["max_tokens"]
    child.profile = task["profile"]
    child.strict = task["strict"]
    child.strict_cold = task["strict_cold"]
    child.harness = task["harness"]
    child.headers_json = (
        json.dumps(task["headers"], sort_keys=True) if task.get("headers") else None
    )
    child.metadata_json = (
        json.dumps(task["metadata"], sort_keys=True) if task.get("metadata") else None
    )
    child.cache_mode = task.get("cache_mode")


def _bench_suite_task_gates(
    task: dict[str, Any],
    envelope: dict[str, Any],
    *,
    exit_code: int,
) -> dict[str, bool]:
    runtime = envelope.get("runtime") or {}
    trace = envelope.get("decode_trace") or {}
    quality = envelope.get("quality") or {}
    warn = task.get("warn_gates") or {}

    def number_at(source: dict[str, Any], key: str) -> float | None:
        value = source.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    tok_s = number_at(runtime, "tok_s") or number_at(runtime, "mean_tok_s")
    last64_ratio = number_at(trace, "last64_over_first64")
    late_verify_ms = number_at(trace, "late_verify_ms") or number_at(
        runtime, "late_verify_ms"
    )
    gates = {
        "exit_zero": int(exit_code) == 0,
        "quality_passed": bool(quality.get("passed")),
    }
    if "tok_s_ge" in warn:
        gates["tok_s_ge_warn_floor"] = bool(
            tok_s is not None and tok_s >= float(warn["tok_s_ge"])
        )
    if "last64_over_first64_ge" in warn:
        gates["last64_over_first64_ge_warn_floor"] = bool(
            last64_ratio is not None
            and last64_ratio >= float(warn["last64_over_first64_ge"])
        )
    if "late_verify_ms_le" in warn:
        gates["late_verify_ms_le_warn_floor"] = bool(
            late_verify_ms is not None
            and late_verify_ms <= float(warn["late_verify_ms_le"])
        )
    return gates


def _bench_suite_task_status(gates: dict[str, bool]) -> str:
    if not gates.get("exit_zero") or not gates.get("quality_passed"):
        return "FAIL"
    return "PASS" if all(gates.values()) else "WARN"


def _bench_suite_model(args: Any) -> str:
    model = getattr(args, "model", None) or DEFAULT_CHAMPION
    cli_flags = getattr(args, "_cli_flags", set()) or set()
    if "model" not in cli_flags and is_verified_default_model_ref(model):
        selection = select_default_model()
        to_dict = getattr(selection, "to_dict", None)
        args._mtplx_default_model_selection = (
            to_dict() if callable(to_dict) else dict(vars(selection))
        )
        return str(selection.model)
    return str(model)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _cmd_bench_nightly(args: Any) -> int:
    model = _bench_suite_model(args)
    action_name = _bench_suite_action_name(args)
    default_prefix = "cli-suite" if action_name == "bench suite" else "cli-nightly"
    run_id = args.run_id or f"{default_prefix}-{time.strftime('%Y%m%d-%H%M%S')}"
    tasks = _bench_suite_tasks(args, model=model)
    default_root = Path(
        "outputs/cli/suite" if action_name == "bench suite" else "outputs/cli/nightly"
    )
    output = Path(args.output or default_root / run_id / "summary.json")
    task_root = Path(args.output_dir or output.parent)
    rows_jsonl = output.parent / "rows.jsonl"
    exactness_output = output.parent / "phase0h-full-exactness.json"
    exactness_contexts = _bench_suite_exactness_contexts(args)
    exactness_cmd = [
        "qa",
        "exactness",
        "--model",
        model,
        "--contexts",
        exactness_contexts,
        "--output",
        str(exactness_output),
    ]
    if args.dry_run:
        dry_tasks = []
        for task in tasks:
            child = type("BenchArgs", (), vars(args).copy())()
            child.model = model
            _apply_bench_suite_task(child, task)
            child.run_id = f"{run_id}-{task['label']}"
            child.output_dir = str(task_root)
            dry_tasks.append(
                {
                    **task,
                    "run_id": child.run_id,
                    "direct_http_command": _direct_http_bench_command(
                        child,
                        model=model,
                        suite=task["suite"],
                        run_id=child.run_id,
                        output_dir=task_root / child.run_id,
                        seed=_benchmark_seed(
                            child,
                            runtime_profile=get_profile(
                                task["profile"]
                            ).runtime_profile,
                            harness=task["harness"],
                        ),
                    )
                    if task["harness"] == "direct-http"
                    else None,
                }
            )
        _print(
            {
                "dry_run": True,
                "action": action_name,
                "quick": _bench_suite_is_quick(args),
                "status": "PLAN",
                "model": model,
                "default_model_selection": getattr(
                    args, "_mtplx_default_model_selection", None
                ),
                "run_id": run_id,
                "tasks": dry_tasks,
                "full_exactness_command": exactness_cmd,
                "output": str(output),
                "rows_jsonl": str(rows_jsonl),
                "policy": {
                    "fanmax_counts_for_product_gate": False,
                    "cold_floor_tok_s": 59.0,
                    "sustained_target_tok_s": 50.0,
                    "sustained_first_stage_tok_s": 45.0,
                },
            }
        )
        return 0

    results: list[dict[str, Any]] = []
    worst_exit = 0
    for task in tasks:
        child = type("BenchArgs", (), vars(args).copy())()
        child.model = model
        _apply_bench_suite_task(child, task)
        child.fanmax = False
        child.run_id = f"{run_id}-{task['label']}"
        child.output_dir = str(task_root)
        env_updates = task.get("env") or {}
        if env_updates:
            with _temporary_env({str(k): str(v) for k, v in env_updates.items()}):
                code = _cmd_bench_run(child)
        else:
            code = _cmd_bench_run(child)
        envelope_path = task_root / child.run_id / "envelope.json"
        envelope = (
            json.loads(envelope_path.read_text(encoding="utf-8"))
            if envelope_path.exists()
            else {}
        )
        task_gates = _bench_suite_task_gates(task, envelope, exit_code=code)
        task_status = _bench_suite_task_status(task_gates)
        results.append(
            {
                "label": task["label"],
                "suite": task["suite"],
                "max_tokens": task["max_tokens"],
                "profile": task["profile"],
                "category": task.get("category"),
                "client": task.get("client"),
                "exit_code": code,
                "status": task_status,
                "gates": task_gates,
                "must": task.get("must") or [],
                "envelope": envelope,
                "envelope_path": str(envelope_path),
            }
        )
        worst_exit = max(worst_exit, int(code))

    exact_proc = _run_exactness_command(
        [
            "--model",
            model,
            "--contexts",
            exactness_contexts,
            "--attention-impl",
            str(getattr(args, "exactness_attention_impl", "mlx_vector_paged")),
            "--block-size",
            str(getattr(args, "exactness_block_size", 16)),
            "--num-blocks",
            str(getattr(args, "exactness_num_blocks", 1024)),
            "--partition-threshold",
            str(getattr(args, "exactness_partition_threshold", 2048)),
            "--partition-size",
            str(getattr(args, "exactness_partition_size", 512)),
            "--output",
            str(exactness_output),
            *(
                ["--no-partitioned"]
                if getattr(args, "exactness_no_partitioned", False)
                else ["--partitioned"]
            ),
        ]
    )
    cold = next((row for row in results if row["label"] == "cold-long-code-192"), {})
    flappy_6k = next((row for row in results if row["label"] == "flappy-6k"), {})
    flappy_10k = next((row for row in results if row["label"] == "flappy-10k"), {})
    quality_passed = all(
        bool((row.get("envelope") or {}).get("quality", {}).get("passed"))
        for row in results
    )
    task_failures_absent = all(row.get("status") != "FAIL" for row in results)
    task_warnings_absent = all(row.get("status") == "PASS" for row in results)
    cold_tok_s = ((cold.get("envelope") or {}).get("runtime") or {}).get("tok_s")
    f6_tok_s = ((flappy_6k.get("envelope") or {}).get("runtime") or {}).get("tok_s")
    f10_tok_s = ((flappy_10k.get("envelope") or {}).get("runtime") or {}).get("tok_s")
    f10_ratio = ((flappy_10k.get("envelope") or {}).get("decode_trace") or {}).get(
        "last64_over_first64"
    )
    gates = {
        "full_exactness_passed": exact_proc.returncode == 0,
        "cold_tok_s_ge_59": bool(cold_tok_s is not None and float(cold_tok_s) >= 59.0),
        "flappy_6k_tok_s_ge_45": bool(f6_tok_s is not None and float(f6_tok_s) >= 45.0),
        "flappy_10k_tok_s_ge_45": bool(
            f10_tok_s is not None and float(f10_tok_s) >= 45.0
        ),
        "flappy_10k_decay_ratio_ge_0_85": bool(
            f10_ratio is not None and float(f10_ratio) >= 0.85
        ),
        "quality_passed": quality_passed,
        "no_fan_product_gate": not bool(getattr(args, "fanmax", False)),
        "task_failures_absent": task_failures_absent,
        "task_warnings_absent": task_warnings_absent,
    }
    summary = {
        "action": action_name,
        "quick": _bench_suite_is_quick(args),
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": model,
        "default_model_selection": getattr(
            args, "_mtplx_default_model_selection", None
        ),
        "tasks": results,
        "rows_jsonl": str(rows_jsonl),
        "full_exactness": {
            "returncode": exact_proc.returncode,
            "passed": exact_proc.returncode == 0,
            "output": str(exactness_output),
            "stdout_tail": exact_proc.stdout[-4000:],
        },
        "gates": gates,
        "passed": all(gates.values()),
        "status": _bench_suite_status(gates),
        "policy": {
            "fanmax_counts_for_product_gate": False,
            "cold_floor_tok_s": 59.0,
            "sustained_target_tok_s": 50.0,
            "sustained_first_stage_tok_s": 45.0,
        },
    }
    _write_jsonl(rows_jsonl, results)
    write_json(output, summary)
    _print(
        {
            "action": action_name,
            "status": summary["status"],
            "quick": summary["quick"],
            "run_id": run_id,
            "output": str(output),
            "rows_jsonl": str(rows_jsonl),
            "passed": summary["passed"],
            "gates": gates,
            "task_outputs": [
                {
                    "label": row["label"],
                    "exit_code": row["exit_code"],
                    "status": row["status"],
                    "envelope": row["envelope_path"],
                }
                for row in results
            ],
        }
    )
    if exact_proc.returncode != 0:
        worst_exit = max(worst_exit, EXIT_EXACTNESS)
    if getattr(args, "strict", False) and not summary["passed"]:
        worst_exit = max(worst_exit, EXIT_STRICT_GATE)
    return worst_exit


def _bench_suite_action_name(args: Any) -> str:
    return (
        "bench suite"
        if getattr(args, "bench_action", None) == "suite"
        else "bench nightly"
    )


def _bench_suite_status(gates: dict[str, Any]) -> str:
    if all(bool(value) for value in gates.values()):
        return "PASS"
    hard_gates = (
        "full_exactness_passed",
        "quality_passed",
        "no_fan_product_gate",
        "task_failures_absent",
    )
    if any(key in gates and not bool(gates.get(key)) for key in hard_gates):
        return "FAIL"
    return "WARN"


def _load_benchmark_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise SystemExit(f"expected JSON object in {path}")
    return data


def _envelope_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if payload.get("action") == "bench nightly" or "tasks" in payload:
        rows: dict[str, dict[str, Any]] = {}
        for task in payload.get("tasks") or []:
            label = str(task.get("label") or task.get("suite") or "unknown")
            envelope = task.get("envelope") or task
            if isinstance(envelope, dict):
                rows[label] = envelope
        return rows
    label = str(payload.get("suite") or payload.get("run_id") or "envelope")
    return {label: payload}


def _metric_float(row: dict[str, Any], *path: str) -> float | None:
    value: Any = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cmd_bench_compare_envelopes(args: Any) -> int:
    if not args.before or not args.after:
        raise SystemExit("bench compare envelope mode requires --before and --after")
    before = _load_benchmark_json(args.before)
    after = _load_benchmark_json(args.after)
    before_rows = _envelope_rows(before)
    after_rows = _envelope_rows(after)
    labels = sorted(set(before_rows) & set(after_rows))
    if not labels:
        raise SystemExit("no matching benchmark labels between --before and --after")
    comparisons = []
    gates: dict[str, bool] = {}
    tolerance_pct = float(getattr(args, "cold_regression_tolerance_pct", 2.0))
    for label in labels:
        left = before_rows[label]
        right = after_rows[label]
        before_tok_s = _metric_float(left, "runtime", "tok_s")
        after_tok_s = _metric_float(right, "runtime", "tok_s")
        before_ratio = _metric_float(left, "decode_trace", "last64_over_first64")
        after_ratio = _metric_float(right, "decode_trace", "last64_over_first64")
        tok_s_delta = (
            after_tok_s - before_tok_s
            if before_tok_s is not None and after_tok_s is not None
            else None
        )
        tok_s_delta_pct = (
            (tok_s_delta / before_tok_s) * 100.0
            if tok_s_delta is not None and before_tok_s
            else None
        )
        quality_after = bool((right.get("quality") or {}).get("passed", True))
        exactness_after = bool(
            ((right.get("correctness") or {}).get("exactness_smoke") or {}).get(
                "passed",
                True,
            )
        )
        label_gates = {
            "quality_after": quality_after,
            "exactness_after": exactness_after,
        }
        if getattr(args, "strict_cold", False) and "cold" in label:
            label_gates["cold_floor_ge_59"] = bool(
                after_tok_s is not None and after_tok_s >= 59.0
            )
            label_gates["cold_regression_within_tolerance"] = bool(
                tok_s_delta_pct is not None and tok_s_delta_pct >= -tolerance_pct
            )
        if getattr(args, "strict", False) and (
            "flappy" in label or "10k" in label or "6k" in label
        ):
            label_gates["sustained_tok_s_ge_45"] = bool(
                after_tok_s is not None and after_tok_s >= 45.0
            )
            label_gates["decay_ratio_ge_0_85"] = bool(
                after_ratio is not None and after_ratio >= 0.85
            )
        if getattr(args, "strict_exactness", False):
            label_gates["strict_exactness_after"] = exactness_after
        gates[label] = all(label_gates.values())
        comparisons.append(
            {
                "label": label,
                "before_tok_s": before_tok_s,
                "after_tok_s": after_tok_s,
                "tok_s_delta": tok_s_delta,
                "tok_s_delta_pct": tok_s_delta_pct,
                "before_last64_over_first64": before_ratio,
                "after_last64_over_first64": after_ratio,
                "gates": label_gates,
                "passed": gates[label],
            }
        )
    report = {
        "action": "bench compare envelopes",
        "before": str(args.before),
        "after": str(args.after),
        "strict": bool(args.strict),
        "strict_cold": bool(args.strict_cold),
        "strict_exactness": bool(getattr(args, "strict_exactness", False)),
        "comparisons": comparisons,
        "passed": all(gates.values()),
    }
    output = Path(args.output) if args.output else None
    if output is not None:
        write_json(output, report)
    _print(report)
    return 0 if report["passed"] else EXIT_STRICT_GATE


def _cmd_bench_compare(args: Any) -> int:
    if getattr(args, "before", None) or getattr(args, "after", None):
        return _cmd_bench_compare_envelopes(args)
    models = args.models or []
    if not models:
        raise SystemExit("bench compare requires --models PATH_A PATH_B [...]")
    suite = args.suite or "champion-bakeoff"
    run_id = args.run_id or f"cli-compare-{time.strftime('%Y%m%d-%H%M%S')}"
    if suite == "champion-bakeoff":
        tasks = [
            {
                "label": "flappy-10k",
                "suite": "flappy",
                "max_tokens": 10000,
                "strict": bool(args.strict),
                "strict_cold": False,
            },
            {
                "label": "python-modules-long",
                "suite": "python_modules_long",
                "max_tokens": 6000,
                "strict": False,
                "strict_cold": False,
            },
            {
                "label": "cold-long-code-192",
                "suite": "cold-long-code-192",
                "max_tokens": 192,
                "strict": False,
                "strict_cold": True,
            },
        ]
    else:
        tasks = [
            {
                "label": suite,
                "suite": suite,
                "max_tokens": int(args.max_tokens),
                "strict": bool(args.strict),
                "strict_cold": bool(args.strict_cold),
            }
        ]
    if args.dry_run:
        _print(
            {
                "dry_run": True,
                "action": "bench compare",
                "models": models,
                "suite": suite,
                "tasks": tasks,
                "exactness_smoke_per_model": True,
            }
        )
        return 0
    results = []
    worst_exit = 0
    for model in models:
        for task in tasks:
            child = type("BenchArgs", (), vars(args).copy())()
            child.model = model
            child.suite = task["suite"]
            child.max_tokens = task["max_tokens"]
            child.strict = task["strict"]
            child.strict_cold = task["strict_cold"]
            child.run_id = f"{run_id}-{Path(model).name}-{task['label']}"
            child.output_dir = str(
                Path(args.output_dir or "outputs/cli/compare") / run_id
            )
            code = _cmd_bench_run(child)
            envelope_path = Path(child.output_dir) / child.run_id / "envelope.json"
            envelope = (
                json.loads(envelope_path.read_text(encoding="utf-8"))
                if envelope_path.exists()
                else {}
            )
            results.append(
                {
                    "model": model,
                    "task": task,
                    "exit_code": code,
                    "envelope": envelope,
                }
            )
            worst_exit = max(worst_exit, code)

    scorecards = []
    for model in models:
        model_rows = [row for row in results if row["model"] == model]
        by_label = {row["task"]["label"]: row for row in model_rows}
        quality_passed = all(
            row.get("envelope", {}).get("quality", {}).get("passed")
            for row in model_rows
        )
        cold_row = by_label.get("cold-long-code-192", {})
        cold_gate = (
            cold_row.get("envelope", {}).get("strict_gates", {}).get("cold_tok_s_ge_55")
        )
        metrics = {
            label: row.get("envelope", {}).get("runtime", {})
            for label, row in by_label.items()
        }
        eligible = bool(model_rows) and quality_passed and cold_gate is not False
        scorecards.append(
            {
                "model": model,
                "eligible": eligible,
                "quality_passed": quality_passed,
                "cold_tok_s_ge_55": cold_gate,
                "metrics": metrics,
            }
        )
    eligible_scorecards = [row for row in scorecards if row["eligible"]]
    recommended = None
    if eligible_scorecards:
        recommended = max(
            eligible_scorecards,
            key=lambda row: (
                float(row["metrics"].get("flappy-10k", {}).get("tok_s") or 0.0),
                float(
                    row["metrics"].get("python-modules-long", {}).get("tok_s") or 0.0
                ),
                float(row["metrics"].get("cold-long-code-192", {}).get("tok_s") or 0.0),
            ),
        )
    summary = {
        "run_id": run_id,
        "suite": suite,
        "results": results,
        "scorecards": scorecards,
        "recommended_champion": recommended["model"] if recommended else None,
        "champion_policy": "No-fan sustained and quality outrank fanmax tie-breakers.",
    }
    champion_path = Path("mtplx/champion.json")
    write_json(
        Path(args.output or Path("outputs/cli/compare") / run_id / "summary.json"),
        summary,
    )
    if args.record_champion and recommended:
        write_json(
            champion_path,
            {
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "model": recommended["model"],
                "suite": suite,
                "reason": "highest eligible no-fan champion-bakeoff scorecard",
                "run_id": run_id,
                "scorecard": recommended,
            },
        )
        summary["recorded_champion"] = str(champion_path)
    _print(summary)
    return worst_exit


def _http_json(
    url: str,
    *,
    timeout: float = 15.0,
    api_key: str | None = None,
) -> dict[str, Any]:
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            headers["X-API-Key"] = api_key
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": str(exc), "url": url}


def _http_post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float = 15.0,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8")
            return {
                "ok": 200 <= int(response.status) < 300,
                "status": int(response.status),
                "json": json.loads(text),
            }
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            parsed: Any = json.loads(text)
        except json.JSONDecodeError:
            parsed = text
        return {"ok": False, "status": exc.code, "error": parsed, "url": url}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": str(exc), "url": url}


def _http_post_text(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float = 15.0,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
            return {
                "ok": 200 <= int(response.status) < 300,
                "status": int(response.status),
                "preview": text[:1200],
            }
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "status": exc.code, "error": text[:1200], "url": url}
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"ok": False, "error": str(exc), "url": url}


def _cmd_bench_serve(args: Any) -> int:
    base = args.url.rstrip("/")
    health = _http_json(base + "/health")
    metrics = _http_json(base + "/metrics")
    report = {
        "suite": args.suite or "multiturn-flappy",
        "turns": args.turns,
        "health": health,
        "metrics": metrics,
        "required_cache_hit_ratio_turn_2_plus": 0.85,
        "note": "This smoke validates the server surface; full multi-turn generation remains the serving QA expansion point.",
    }
    _print(report)
    return 0 if health.get("ok") else EXIT_STRICT_GATE


def _cmd_bench_reference(args: Any) -> int:
    _print(
        {
            "action": "bench reference",
            "champion": args.champion,
            "references": args.references,
            "suite": args.suite,
            "max_tokens": args.max_tokens,
            "product_gate": False,
            "note": "Reference commands are diagnostic floors and do not promote MTPLX runs.",
        }
    )
    return 0


def _ssh_command(host: str, remote_script: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        host,
        f"bash -lc {shlex.quote(remote_script)}",
    ]


def _run_ssh(
    host: str, remote_script: str, *, timeout_s: int
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _ssh_command(host, remote_script),
        cwd=repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=timeout_s,
    )


def _remote_read_text(
    host: str, path: str, *, timeout_s: int = 30
) -> tuple[str | None, str | None]:
    proc = _run_ssh(host, f"cat {shlex.quote(path)}", timeout_s=timeout_s)
    if proc.returncode != 0:
        return None, proc.stdout[-2000:]
    return proc.stdout, None


def _remote_stat(host: str, path: str, *, timeout_s: int = 30) -> dict[str, Any] | None:
    proc = _run_ssh(host, f"stat -c '%Y %s' {shlex.quote(path)}", timeout_s=timeout_s)
    if proc.returncode != 0:
        return None
    parts = proc.stdout.strip().split()
    if len(parts) < 2:
        return None
    try:
        return {"mtime_epoch": int(parts[0]), "size_bytes": int(parts[1])}
    except ValueError:
        return None


def _remote_epoch(host: str) -> int | None:
    proc = _run_ssh(host, "date +%s", timeout_s=10)
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def _remote_probe_script(args: Any) -> str:
    venv_python = str(Path(args.remote_venv) / "bin" / "python")
    return "\n".join(
        [
            "set -e",
            "echo host=$(hostname)",
            "date -Is",
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
            f"{shlex.quote(venv_python)} - <<'PY'",
            "import importlib.metadata as md",
            "for pkg in ('vllm', 'torch', 'transformers', 'flashinfer-python'):",
            "    try:",
            "        print(f'{pkg}=' + md.version(pkg))",
            "    except Exception as exc:",
            "        print(f'{pkg}=unavailable:{exc}')",
            "PY",
            "command -v nsys",
            f"test -d {shlex.quote(args.remote_phase_dir)}",
            f"test -f {shlex.quote(str(Path(args.remote_phase_dir) / args.remote_run_script))}",
        ]
    )


def _write_remote_artifact(local_dir: Path, name: str, text: str | None) -> str | None:
    if text is None:
        return None
    path = local_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _remote_reference_prompt_line(args: Any) -> str:
    prompt_path = Path(prompt_suite_path(args.suite or "flappy"))
    with prompt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                break
        else:
            raise SystemExit(f"empty prompt suite: {prompt_path}")
    row["suite"] = args.suite or row.get("suite") or "reference"
    row["max_tokens"] = int(args.max_tokens)
    row.setdefault("id", f"reference_{args.suite or 'prompt'}")
    if "prompt" not in row:
        raise SystemExit(
            f"remote vLLM reference requires prompt field in {prompt_path}"
        )
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def _remote_capture_script(args: Any, *, remote_run_cmd: str) -> str:
    prompt_line = _remote_reference_prompt_line(args)
    prompt_file = str(Path(args.remote_phase_dir) / "profile_prompt.jsonl")
    backup_file = str(
        Path(args.remote_phase_dir) / ".profile_prompt.jsonl.mtplx_backup"
    )
    return "\n".join(
        [
            "set -e",
            f"if [ -f {shlex.quote(prompt_file)} ]; then cp {shlex.quote(prompt_file)} {shlex.quote(backup_file)}; fi",
            "restore_prompt() {",
            f"  if [ -f {shlex.quote(backup_file)} ]; then mv {shlex.quote(backup_file)} {shlex.quote(prompt_file)}; fi",
            "}",
            "trap restore_prompt EXIT",
            f"printf '%s\\n' {shlex.quote(prompt_line)} > {shlex.quote(prompt_file)}",
            remote_run_cmd,
        ]
    )


def _remote_offline_capture_script(args: Any, *, remote_out_dir: str) -> str:
    prompt_line = _remote_reference_prompt_line(args)
    offline_python = r"""from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["no-mtp", "mtp5"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.prompt_file.read_text().splitlines() if line.strip()]
    row = rows[0]
    prompt = row["prompt"]
    spec_summary = None
    if args.mode == "mtp5":
        spec_summary = {"method": "mtp", "num_speculative_tokens": 5}
    llm = LLM(
        model=args.model,
        served_model_name="qwen3.6-27b",
        tokenizer=args.model,
        quantization="compressed-tensors",
        tensor_parallel_size=2,
        max_model_len=32768,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        block_size=32,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.90,
        trust_remote_code=False,
        dtype="bfloat16",
        speculative_config=dict(spec_summary) if spec_summary else None,
        profiler_config={"profiler": "cuda"},
        disable_log_stats=True,
    )
    warm = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=min(64, max(1, args.max_tokens)),
        seed=42,
    )
    llm.generate([prompt], warm, use_tqdm=False)
    sampling = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=args.max_tokens,
        seed=42,
    )
    llm.start_profile()
    started = time.perf_counter()
    outputs = llm.generate([prompt], sampling, use_tqdm=False)
    elapsed = time.perf_counter() - started
    completion = outputs[0].outputs[0]
    completion_tokens = len(completion.token_ids)
    result = {
        "summary": {
            "label": f"offline-nsys-{args.mode}",
            "model": "qwen3.6-27b",
            "prompt_count": 1,
            "ok_count": 1,
            "total_completion_tokens": completion_tokens,
            "mean_decode_tok_s": completion_tokens / elapsed if elapsed else None,
            "mean_end_to_end_tok_s": completion_tokens / elapsed if elapsed else None,
        },
        "rows": [
            {
                "id": row.get("id"),
                "suite": row.get("suite"),
                "status": "ok",
                "max_tokens": args.max_tokens,
                "completion_tokens": completion_tokens,
                "wall_s": elapsed,
                "decode_tok_s": completion_tokens / elapsed if elapsed else None,
                "end_to_end_tok_s": completion_tokens / elapsed if elapsed else None,
                "text_prefix": completion.text[:240],
            }
        ],
        "sampler": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 42},
        "speculative_config": spec_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    llm.stop_profile()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
    return f"""set -e
OUT_DIR={shlex.quote(remote_out_dir)}
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
cleanup_vllm_children() {{
  for pattern in "[V]LLM::EngineCore" "[V]LLM::Worker" "[v]llm serve" "offline_capture.py" "multiprocessing.resource_tracker"; do
    pids="$(pgrep -f "$pattern" 2>/dev/null | awk -v self="$$" '$1 != self {{print}}' || true)"
    if [ -n "$pids" ]; then kill -TERM $pids 2>/dev/null || true; fi
  done
  sleep 2
  for pattern in "[V]LLM::EngineCore" "[V]LLM::Worker" "[v]llm serve" "offline_capture.py" "multiprocessing.resource_tracker"; do
    pids="$(pgrep -f "$pattern" 2>/dev/null | awk -v self="$$" '$1 != self {{print}}' || true)"
    if [ -n "$pids" ]; then kill -KILL $pids 2>/dev/null || true; fi
  done
}}
trap cleanup_vllm_children EXIT
PROMPT_FILE="$OUT_DIR/profile_prompt.jsonl"
OFFLINE_SCRIPT="$OUT_DIR/offline_capture.py"
BENCH_JSON="$OUT_DIR/bench.json"
STDOUT_LOG="$OUT_DIR/stdout.log"
REPORT_BASE="$OUT_DIR/nsys-{shlex.quote(args.remote_mode)}"
printf '%s\\n' {shlex.quote(prompt_line)} > "$PROMPT_FILE"
cat > "$OFFLINE_SCRIPT" <<'PY'
{offline_python}
PY
source {shlex.quote(str(Path(args.remote_venv) / "bin" / "activate"))}
export CUDA_VISIBLE_DEVICES=0,1
export RAY_memory_monitor_refresh_ms=0
export NCCL_CUMEM_ENABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export VLLM_SLEEP_WHEN_IDLE=0
export VLLM_ENABLE_CUDAGRAPH_GC=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export VLLM_ATTENTION_BACKEND=FLASHINFER
set +e
nsys profile --force-overwrite=true --wait=primary --trace=cuda,nvtx --cuda-graph-trace=graph --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown -o "$REPORT_BASE" \\
  {shlex.quote(str(Path(args.remote_venv) / "bin" / "python"))} "$OFFLINE_SCRIPT" --mode {shlex.quote(args.remote_mode)} --model /home/youssof/ai/models/Qwen3.6-27B-AWQ-BF16-INT4 --prompt-file "$PROMPT_FILE" --output "$BENCH_JSON" --max-tokens {int(args.max_tokens)} > "$STDOUT_LOG" 2>&1
NSYS_STATUS=$?
cleanup_vllm_children
set -e
REP="${{REPORT_BASE}}.nsys-rep"
if [ -f "$REP" ]; then
  nsys stats --force-overwrite=true --force-export=true --report cuda_gpu_kern_sum "$REP" > "$OUT_DIR/cuda_gpu_kern_sum.txt" 2>&1 || true
  nsys stats --force-overwrite=true --force-export=true --report cuda_api_sum "$REP" > "$OUT_DIR/cuda_api_sum.txt" 2>&1 || true
  nsys stats --force-overwrite=true --force-export=true --report cuda_gpu_trace "$REP" > "$OUT_DIR/cuda_gpu_trace.txt" 2>&1 || true
  nsys stats --force-overwrite=true --force-export=true --report nvtx_sum "$REP" > "$OUT_DIR/nvtx_sum.txt" 2>&1 || true
fi
find "$OUT_DIR" -maxdepth 1 -type f -printf '%f %s\\n' | sort
exit "$NSYS_STATUS"
"""


def _cmd_bench_reference_vllm(args: Any) -> int:
    run_id = (
        args.run_id
        or f"vllm-reference-{args.remote_mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    local_dir = Path(args.remote_output_dir or "outputs/cli/reference-vllm") / run_id
    remote_kind = getattr(args, "remote_capture_kind", "offline")
    remote_out_dir = str(
        Path(args.remote_phase_dir) / f"nsys-v4-{remote_kind}-{args.remote_mode}"
    )
    remote_script = str(Path(args.remote_phase_dir) / args.remote_run_script)
    remote_run_cmd = (
        f"cd {shlex.quote(args.remote_phase_dir)} && "
        f"bash {shlex.quote(remote_script)} {shlex.quote(args.remote_mode)} {int(args.remote_port)}"
    )
    capture_script = (
        _remote_offline_capture_script(args, remote_out_dir=remote_out_dir)
        if remote_kind == "offline"
        else _remote_capture_script(args, remote_run_cmd=remote_run_cmd)
    )
    if args.dry_run:
        _print(
            {
                "dry_run": True,
                "action": "bench reference-vllm",
                "ssh_host": args.ssh_host,
                "remote_capture_kind": remote_kind,
                "probe_command": _ssh_command(
                    args.ssh_host, _remote_probe_script(args)
                ),
                "capture_command": _ssh_command(args.ssh_host, capture_script),
                "remote_prompt_override": json.loads(
                    _remote_reference_prompt_line(args)
                ),
                "local_output_dir": str(local_dir),
                "product_gate": False,
            }
        )
        return 0

    local_dir.mkdir(parents=True, exist_ok=True)
    probe = _run_ssh(args.ssh_host, _remote_probe_script(args), timeout_s=60)
    _write_remote_artifact(local_dir, "remote-probe.txt", probe.stdout)
    if probe.returncode != 0:
        report = {
            "run_id": run_id,
            "action": "bench reference-vllm",
            "passed": False,
            "error": "remote 3090 probe failed",
            "returncode": probe.returncode,
            "stdout_tail": probe.stdout[-4000:],
        }
        write_json(local_dir / "summary.json", report)
        _print(report)
        return EXIT_STRICT_GATE

    capture_error = None
    capture_started_epoch = None
    if args.capture_dispatch:
        capture_started_epoch = _remote_epoch(args.ssh_host)
        try:
            capture = _run_ssh(
                args.ssh_host, capture_script, timeout_s=int(args.remote_timeout_s)
            )
        except subprocess.TimeoutExpired as exc:
            capture_error = {
                "error": "remote capture timed out",
                "timeout_s": args.remote_timeout_s,
                "stdout_tail": (exc.stdout or "")[-4000:]
                if isinstance(exc.stdout, str)
                else None,
            }
        else:
            _write_remote_artifact(local_dir, "remote-capture.txt", capture.stdout)
            if capture.returncode != 0:
                capture_error = {
                    "error": "remote Nsight capture failed",
                    "returncode": capture.returncode,
                    "stdout_tail": capture.stdout[-4000:],
                }

    kernel_text, kernel_error = _remote_read_text(
        args.ssh_host,
        str(Path(remote_out_dir) / "cuda_gpu_kern_sum.txt"),
    )
    cuda_api_text, cuda_api_error = _remote_read_text(
        args.ssh_host,
        str(Path(remote_out_dir) / "cuda_api_sum.txt"),
    )
    bench_text, bench_error = _remote_read_text(
        args.ssh_host,
        str(Path(remote_out_dir) / "bench.json"),
    )
    client_log, _client_error = _remote_read_text(
        args.ssh_host,
        str(Path(remote_out_dir) / "client.log"),
    )
    stdout_log, _stdout_error = _remote_read_text(
        args.ssh_host,
        str(Path(remote_out_dir) / "stdout.log"),
    )
    kernel_stat = _remote_stat(
        args.ssh_host, str(Path(remote_out_dir) / "cuda_gpu_kern_sum.txt")
    )
    cuda_api_stat = _remote_stat(
        args.ssh_host, str(Path(remote_out_dir) / "cuda_api_sum.txt")
    )
    bench_stat = _remote_stat(args.ssh_host, str(Path(remote_out_dir) / "bench.json"))
    kernel_stale = bool(
        args.capture_dispatch
        and capture_started_epoch is not None
        and kernel_stat
        and int(kernel_stat["mtime_epoch"]) < int(capture_started_epoch)
    )
    cuda_api_stale = bool(
        args.capture_dispatch
        and capture_started_epoch is not None
        and cuda_api_stat
        and int(cuda_api_stat["mtime_epoch"]) < int(capture_started_epoch)
    )
    _write_remote_artifact(local_dir, "cuda_gpu_kern_sum.txt", kernel_text)
    _write_remote_artifact(local_dir, "cuda_api_sum.txt", cuda_api_text)
    _write_remote_artifact(local_dir, "bench.json", bench_text)
    _write_remote_artifact(local_dir, "client.log", client_log)
    _write_remote_artifact(local_dir, "stdout.log", stdout_log)

    reference = summarize_vllm_reference(
        cuda_kernel_summary_text=kernel_text,
        cuda_api_summary_text=cuda_api_text,
        bench_json_text=bench_text,
    )
    capture_artifacts_valid = bool(
        kernel_text
        and bench_text
        and not kernel_stale
        and (not cuda_api_text or not cuda_api_stale)
    )
    capture_warning = None
    if capture_error and capture_artifacts_valid:
        capture_warning = {
            "warning": "remote Nsight command returned nonzero but fresh artifacts were recovered",
            "capture_error": capture_error,
        }
    report = {
        "run_id": run_id,
        "action": "bench reference-vllm",
        "ssh_host": args.ssh_host,
        "remote_mode": args.remote_mode,
        "remote_capture_kind": remote_kind,
        "remote_phase_dir": args.remote_phase_dir,
        "remote_out_dir": remote_out_dir,
        "suite": args.suite,
        "max_tokens": args.max_tokens,
        "capture_dispatch": bool(args.capture_dispatch),
        "capture_started_epoch": capture_started_epoch,
        "product_gate": False,
        "remote_prompt_override": json.loads(_remote_reference_prompt_line(args)),
        "capture_error": None if capture_warning else capture_error,
        "capture_warning": capture_warning,
        "probe_stdout_tail": probe.stdout[-4000:],
        "artifact_errors": {
            "cuda_gpu_kern_sum": kernel_error,
            "cuda_api_sum": cuda_api_error,
            "bench_json": bench_error,
        },
        "artifact_stats": {
            "cuda_gpu_kern_sum": kernel_stat,
            "cuda_api_sum": cuda_api_stat,
            "bench_json": bench_stat,
            "cuda_gpu_kern_sum_stale_for_capture": kernel_stale,
            "cuda_api_sum_stale_for_capture": cuda_api_stale,
        },
        "reference": reference,
        "local_output_dir": str(local_dir),
    }
    write_json(local_dir / "summary.json", report)
    _print(report)
    if args.capture_dispatch:
        return 0 if capture_artifacts_valid else EXIT_STRICT_GATE
    return 0 if bench_text else EXIT_STRICT_GATE


def cmd_qa_public(args: Any) -> int:
    if args.qa_action == "exactness":
        return _cmd_qa_exactness(args)
    if args.qa_action == "distribution":
        return _cmd_qa_distribution(args)
    raise SystemExit(f"unknown qa action: {args.qa_action}")


def _run_exactness_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(repo_root() / "scripts" / "phase0h_paged_verifier_exactness.py"),
            *args,
        ],
        cwd=repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _cmd_qa_exactness(args: Any) -> int:
    output = Path(args.output) if args.output else default_output_path("qa-exactness")
    cmd = [
        "--model",
        args.model,
        "--contexts",
        args.contexts,
        "--attention-impl",
        str(args.exactness_attention_impl),
        "--block-size",
        str(args.exactness_block_size),
        "--num-blocks",
        str(args.exactness_num_blocks),
        "--partition-threshold",
        str(args.exactness_partition_threshold),
        "--partition-size",
        str(args.exactness_partition_size),
        "--output",
        str(output),
    ]
    if args.exactness_no_partitioned:
        cmd.append("--no-partitioned")
    else:
        cmd.append("--partitioned")
    if args.prompt_suite:
        cmd.extend(["--prompt-suite", args.prompt_suite])
    proc = _run_exactness_command(cmd)
    if proc.returncode == 0:
        print(proc.stdout, end="")
        return 0
    output_text = proc.stdout or ""
    print("error: exactness check failed")
    if "float_to_fp8_e4m3" in output_text:
        print(
            "detail: the selected paged-attention exactness path failed to "
            "compile in the Metal backend on this machine."
        )
    else:
        candidate_lines = [
            line.strip()
            for line in output_text.splitlines()
            if any(
                marker in line.lower()
                for marker in (
                    "runtimeerror:",
                    "error:",
                    "failed",
                    "undeclared identifier",
                )
            )
        ]
        for line in reversed(candidate_lines or output_text.splitlines()):
            stripped = line.strip()
            if stripped:
                print(f"detail: {stripped[:240]}")
                break
    print(f"output: {output}")
    print(
        "try: a different --exactness-attention-impl "
        "(run `mtplx qa exactness --help` for the choices)"
    )
    return EXIT_EXACTNESS


def _cmd_qa_distribution(args: Any) -> int:
    suite_names = distribution_suite_names(args.suite)
    rows = []
    worst = 0
    for suite_name in suite_names:
        output = (
            Path(args.output_dir or "outputs/cli/qa-distribution")
            / f"{suite_name}-{time.strftime('%Y%m%d-%H%M%S')}.json"
        )
        proc = _run_exactness_command(
            [
                "--model",
                args.model,
                "--contexts",
                args.contexts,
                "--attention-impl",
                str(args.exactness_attention_impl),
                "--block-size",
                str(args.exactness_block_size),
                "--num-blocks",
                str(args.exactness_num_blocks),
                "--partition-threshold",
                str(args.exactness_partition_threshold),
                "--partition-size",
                str(args.exactness_partition_size),
                "--prompt-suite",
                prompt_suite_path(suite_name),
                "--output",
                str(output),
                *(
                    ["--no-partitioned"]
                    if args.exactness_no_partitioned
                    else ["--partitioned"]
                ),
            ]
        )
        rows.append(
            {
                "suite": suite_name,
                "output": str(output),
                "returncode": proc.returncode,
                "passed": proc.returncode == 0,
                "stdout_tail": proc.stdout[-2000:],
            }
        )
        worst = max(worst, proc.returncode)
    report = {
        "model": args.model,
        "reference_stack": args.reference_stack,
        "contexts": args.contexts,
        "tolerance": args.tolerance,
        "rows": rows,
        "passed": worst == 0,
    }
    _print(report)
    return 0 if worst == 0 else EXIT_EXACTNESS


def cmd_profile_public(args: Any) -> int:
    if args.profile_action == "dispatch":
        return _cmd_profile_dispatch(args)
    if args.profile_action == "thermal":
        return _cmd_profile_thermal(args)
    if args.profile_action == "compile-audit":
        return _cmd_profile_compile_audit(args)
    if args.profile_action == "eval-attribution":
        return _cmd_profile_eval_attribution(args)
    raise SystemExit(f"unknown profile action: {args.profile_action}")


def _research_script(name: str) -> Path | None:
    """Resolve a research-workspace helper script, or None when not shipped.

    Several diagnostic subcommands drive scripts that live in the MTPLX
    research workspace and are not part of the distributed package (a wheel
    install has no scripts/ tree at all). Resolving through this check keeps
    those commands honest: run the real script, or say exactly why not —
    never hand python3 a phantom path.
    """
    candidate = repo_root() / "scripts" / name
    return candidate if candidate.is_file() else None


def _research_script_unavailable(action: str, script: str) -> int:
    _print(
        {
            "action": action,
            "available": False,
            "reason": (
                f"scripts/{script} is a research-workspace tool and is not "
                "included in this installation"
            ),
            "hint": "Run from an MTPLX source checkout that provides this script.",
        }
    )
    return 2


def _cmd_profile_dispatch(args: Any) -> int:
    trace_script = _research_script("analyze_metal_command_trace.py")
    if args.trace:
        if trace_script is None:
            return _research_script_unavailable(
                "profile dispatch --trace", "analyze_metal_command_trace.py"
            )
        out_dir = Path(args.output_dir or "outputs/cli/dispatch") / time.strftime(
            "%Y%m%d-%H%M%S"
        )
        proc = subprocess.run(
            [
                sys.executable,
                str(trace_script),
                args.trace,
                "--out-dir",
                str(out_dir),
            ],
            cwd=repo_root(),
            text=True,
            check=False,
        )
        return proc.returncode
    _print(
        {
            "action": "profile dispatch",
            "model": args.model,
            "suite": args.suite,
            "max_tokens": args.max_tokens,
            "implemented_capture": False,
            "next": (
                "Run with --trace PATH to analyze an existing MLX Metal command trace."
                if trace_script is not None
                else "Trace analysis needs the research-workspace script "
                "scripts/analyze_metal_command_trace.py, which is not included "
                "in this installation."
            ),
        }
    )
    return 0


def _cmd_profile_thermal(args: Any) -> int:
    script = _research_script("run_flappy_smc_thermal_diagnostics.py")
    if script is None:
        return _research_script_unavailable(
            "profile thermal", "run_flappy_smc_thermal_diagnostics.py"
        )
    cmd = [
        sys.executable,
        str(script),
        "--model",
        args.model,
        "--run-id",
        args.run_id or f"cli-thermal-{time.strftime('%Y%m%d-%H%M%S')}",
        "--output-dir",
        args.output_dir or "outputs/cli/thermal",
    ]
    if args.dry_run:
        _print({"dry_run": True, "command": cmd})
        return 0
    return subprocess.call(cmd, cwd=repo_root())


def _reject_native_ar_for_mtp_diagnostic(
    inspection: dict[str, Any],
    *,
    action: str,
) -> int | None:
    if (inspection.get("compatibility") or {}).get(
        "runtime_compatibility"
    ) != "native-ar-only":
        return None
    _print(
        {
            "error": f"{action} requires an MTP-capable runtime",
            "detail": "Laguna-S-2.1 is installed as target-only AR",
        }
    )
    return EXIT_UNSUPPORTED_MODEL


def _cmd_profile_compile_audit(args: Any) -> int:
    inspection, gate_exit = _model_gate(args.model)
    native_ar_exit = _reject_native_ar_for_mtp_diagnostic(
        inspection,
        action="profile compile-audit",
    )
    if native_ar_exit is not None:
        return native_ar_exit
    output = (
        Path(args.output)
        if args.output
        else (
            Path(args.output_dir or "outputs/cli/compile-audit")
            / f"compile-audit-{time.strftime('%Y%m%d-%H%M%S')}.json"
        )
    )
    cmd = [
        sys.executable,
        str(repo_root() / "scripts" / "probe_mx_compile_buckets.py"),
        "--model",
        args.model,
        "--prompts",
        args.prompts,
        "--prompt-index",
        str(args.prompt_index),
        "--prefill-chunks",
        args.prefill_chunks,
        "--depths",
        args.depths,
        "--max-tokens",
        str(args.max_tokens),
        "--repeats",
        str(args.repeats),
        "--warmup",
        str(args.warmup),
        "--verify-core",
        args.verify_core,
        "--output",
        str(output),
    ]
    if args.disable_thinking:
        cmd.append("--disable-thinking")
    if args.skip_prefill:
        cmd.append("--skip-prefill")
    if args.skip_verify:
        cmd.append("--skip-verify")
    if args.dry_run:
        _print(
            {
                "dry_run": True,
                "action": "profile compile-audit",
                "model": inspection,
                "exactness_smoke": {
                    "automatic": not args.skip_exactness_smoke,
                    "context": 2048,
                    "profile": _exactness_profile_kwargs(args),
                },
                "exact_paged_env": _exact_paged_env_from_args(args),
                "command": cmd,
                "output": str(output),
            }
        )
        return 0
    if gate_exit is not None:
        _print({"error": "model failed MTP primary gate", "model": inspection})
        return gate_exit
    if not args.skip_exactness_smoke:
        smoke = run_exactness_smoke(
            args.model,
            context=2048,
            prompt_suite=prompt_suite_path("flappy"),
            output=output.with_name(output.stem + "-exactness-smoke.json"),
            **_exactness_profile_kwargs(args),
        )
        if not smoke["passed"]:
            write_json(
                output.with_name(output.stem + "-failed.json"),
                {"error": "Phase 0H exactness smoke failed", "exactness_smoke": smoke},
            )
            _print(
                {"error": "Phase 0H exactness smoke failed", "exactness_smoke": smoke}
            )
            return EXIT_EXACTNESS
    proc = subprocess.run(
        cmd,
        cwd=repo_root(),
        env={**os.environ, **_exact_paged_env_from_args(args)},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(proc.stdout, end="")
    return proc.returncode


def _cmd_profile_eval_attribution(args: Any) -> int:
    inspection, gate_exit = _model_gate(args.model)
    native_ar_exit = _reject_native_ar_for_mtp_diagnostic(
        inspection,
        action="profile eval-attribution",
    )
    if native_ar_exit is not None:
        return native_ar_exit
    output = (
        Path(args.output)
        if args.output
        else (
            Path(args.output_dir or "outputs/cli/eval-attribution")
            / f"eval-attribution-{time.strftime('%Y%m%d-%H%M%S')}.json"
        )
    )
    script = _research_script("probe_eval_attribution.py")
    if script is None:
        return _research_script_unavailable(
            "profile eval-attribution", "probe_eval_attribution.py"
        )
    cmd = [
        sys.executable,
        str(script),
        "--model",
        args.model,
        "--prefix-tokens",
        str(args.prefix_tokens),
        "--verify-tokens",
        str(args.verify_tokens),
        "--seed",
        str(args.seed),
        "--depth",
        str(args.depth),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--verify-strategy",
        args.verify_strategy,
        "--verify-core",
        args.verify_core,
        "--mtp-history-policy",
        args.mtp_history_policy,
        "--orders",
        args.orders,
        "--output",
        str(output),
    ]
    if args.no_serving_fast_defaults:
        cmd.append("--no-serving-fast-defaults")
    if args.prompt:
        cmd.extend(["--prompt", args.prompt])
    payload = {
        "action": "profile eval-attribution",
        "model": inspection,
        "command": cmd,
        "output": str(output),
        "purpose": (
            "Attribute verify-cycle first-eval debt across verifier outputs, "
            "attention cache, and recurrent GDN/conv state before committing "
            "to a larger owned kernel boundary."
        ),
    }
    if args.dry_run:
        _print({"dry_run": True, **payload})
        return 0
    if gate_exit is not None:
        _print({"error": "model failed MTP primary gate", "model": inspection})
        return gate_exit
    proc = subprocess.run(
        cmd,
        cwd=repo_root(),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(proc.stdout, end="")
    return proc.returncode


def cmd_thermal_public(args: Any) -> int:
    if args.thermal_action != "fanmax-run":
        raise SystemExit(f"unknown thermal action: {args.thermal_action}")
    run_id = args.run_id or f"cli-fanmax-{time.strftime('%Y%m%d-%H%M%S')}"
    child = [
        sys.executable,
        "-m",
        "mtplx.cli",
        "bench",
        "run",
        "--model",
        args.model,
        "--suite",
        args.suite,
        "--max-tokens",
        str(args.max_tokens),
        "--run-id",
        run_id,
        "--fanmax",
    ]
    script = _research_script("run_fanmax_command.py")
    if script is None:
        return _research_script_unavailable(
            "thermal fanmax-run", "run_fanmax_command.py"
        )
    cmd = [
        sys.executable,
        str(script),
        "--output-dir",
        args.output_dir or "outputs/cli/fanmax",
        "--",
        *child,
    ]
    if args.dry_run:
        _print({"dry_run": True, "command": cmd})
        return 0
    return subprocess.call(cmd, cwd=repo_root())


def cmd_max_public(args: Any) -> int:
    from mtplx.thermal import (
        _read_max_marker,
        check_and_recover_stale_max,
        install_passwordless_sudoers_rule,
        install_thermal_control_homebrew,
        remove_passwordless_sudoers_rule,
        set_thermal_profile,
        thermal_status,
    )

    action = args.max_action
    if action == "install":
        payload = install_thermal_control_homebrew(
            install_daemon=not bool(getattr(args, "no_daemon", False)),
        )
        code = 0 if payload.get("ok") else 1
    elif action == "grant_sudo":
        payload = install_passwordless_sudoers_rule()
        code = 0 if payload.get("ok") else 1
    elif action == "revoke_sudo":
        payload = remove_passwordless_sudoers_rule()
        code = 0 if payload.get("ok") else 1
    elif action == "status":
        # Run stale-max recovery before reporting status so `mtplx max --status`
        # also doubles as the "fix my fans!" command after a crash.
        recovery = check_and_recover_stale_max()
        payload = thermal_status()
        payload["max_marker"] = _read_max_marker()
        payload["recovered_stale_max"] = recovery
        code = 0
    elif action == "silent":
        # `mtplx max --off` — explicit user request to restore fans now.
        # Also clear the marker so `--status` doesn't report stale state.
        from mtplx.thermal import _clear_max_marker

        payload = set_thermal_profile(
            "silent", dry_run=bool(getattr(args, "dry_run", False))
        )
        if payload.get("ok") and not getattr(args, "dry_run", False):
            _clear_max_marker()
        code = 0 if payload.get("ok") or getattr(args, "dry_run", False) else 1
    else:
        payload = set_thermal_profile(
            action, dry_run=bool(getattr(args, "dry_run", False))
        )
        code = 0 if payload.get("ok") or getattr(args, "dry_run", False) else 1
    if getattr(args, "json", False):
        _print(payload)
    else:
        if action in ("install", "grant_sudo", "revoke_sudo"):
            print(f"thermal {action} ok: {str(bool(payload.get('ok'))).lower()}")
            if payload.get("message"):
                print(payload["message"])
        else:
            selected = (payload.get("detection") or {}).get("selected") or {}
            tool = selected.get("kind", "none")
            print(f"thermal tool: {tool}")
            if action == "status":
                print(
                    f"available: {str(bool((payload.get('detection') or {}).get('available'))).lower()}"
                )
            else:
                print(f"profile: {action}")
                print(f"ok: {str(bool(payload.get('ok'))).lower()}")
            if payload.get("message"):
                print(payload["message"])
            elif not bool((payload.get("detection") or {}).get("available")) and (
                payload.get("detection") or {}
            ).get("instructions"):
                print((payload.get("detection") or {})["instructions"])
    return code


def _server_url(host: str, port: int) -> str:
    return local_url_for_bind(host, int(port))


def _chat_url(host: str, port: int) -> str:
    return _server_url(host, port) + "/"


def _openwebui_docker_api_base_url(port: int) -> str:
    return f"http://host.docker.internal:{int(port)}/v1"


def _openwebui_docker_command(
    *,
    mtplx_port: int,
    webui_port: int = 3000,
    single_user: bool = False,
    api_key: str = "mtplx-local",
) -> list[str]:
    command = [
        "docker",
        "run",
        "-d",
        "--name",
        "open-webui",
        "--restart",
        "unless-stopped",
        "-p",
        f"{int(webui_port)}:8080",
        "--add-host=host.docker.internal:host-gateway",
        "-e",
        "ENABLE_OLLAMA_API=False",
        "-e",
        "ENABLE_OPENAI_API=True",
        "-e",
        f"OPENAI_API_BASE_URLS={_openwebui_docker_api_base_url(mtplx_port)}",
        "-e",
        f"OPENAI_API_KEYS={api_key}",
        "-e",
        "ENABLE_TITLE_GENERATION=False",
        "-e",
        "ENABLE_TAGS_GENERATION=False",
        "-e",
        "ENABLE_FOLLOW_UP_GENERATION=False",
        "-e",
        "ENABLE_AUTOCOMPLETE_GENERATION=False",
    ]
    if single_user:
        command.extend(["-e", "WEBUI_AUTH=False"])
    command.extend(
        [
            "-v",
            "open-webui:/app/backend/data",
            "ghcr.io/open-webui/open-webui:main",
        ]
    )
    return command


def _shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _open_browser_url(url: str) -> None:
    try:
        webbrowser.open(url, new=2, autoraise=True)
    except Exception as exc:
        _print_serve_start_line(f"warning: could not open browser automatically: {exc}")


def _dashboard_url(host: str, port: int) -> str:
    return _server_url(host, port) + "/dashboard/"


def cmd_dashboard_public(args: Any) -> int:
    """Open the live MTPLX dashboard in the browser.

    The dashboard is mounted at ``/dashboard`` on the running MTPLX server.
    This command is a thin opener: probe ``/health`` to confirm the server
    is up, then ``webbrowser.open`` the dashboard URL. It deliberately
    does *not* import MLX or start a server; if MTPLX isn't running it
    tells the user how to start it.
    """

    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 8000))
    timeout = float(getattr(args, "timeout", 2.5))
    json_output = bool(getattr(args, "json", False))

    base = _server_url(host, port)
    health_url = f"{base}/health"
    dashboard_url = _dashboard_url(host, port)

    health = _http_json(health_url, timeout=timeout)
    server_up = bool(
        isinstance(health, dict) and "error" not in health and health.get("ok")
    )

    payload: dict[str, Any] = {
        "command": "dashboard",
        "ok": server_up,
        "dashboard_url": dashboard_url,
        "health_url": health_url,
        "host": host,
        "port": port,
    }
    if server_up:
        payload["model"] = health.get("model")
        profile = health.get("profile")
        payload["profile"] = profile.get("name") if isinstance(profile, dict) else None
    else:
        payload["error"] = "MTPLX server is not reachable"
        payload["detail"] = health.get("error") if isinstance(health, dict) else None

    if json_output:
        _print(payload)
    elif server_up:
        print(f"MTPLX dashboard: {dashboard_url}")
        if payload.get("model"):
            print(f"model: {payload['model']}")
        print("opening in your browser...")
    else:
        print(f"error: MTPLX server is not reachable at {base}")
        print("try: mtplx start  (then re-run `mtplx dashboard`)")
        print("try: mtplx quickstart --host 127.0.0.1 --port 8000")

    if not server_up:
        return 1

    if not bool(getattr(args, "no_browser", False)):
        _open_browser_url(dashboard_url)
    return 0


def _connect_host_for_bind(host: str) -> str:
    return connect_host_for_bind(host)


def _port_is_busy(host: str, port: int) -> bool:
    try:
        with socket.create_connection(
            (_connect_host_for_bind(host), int(port)), timeout=0.2
        ):
            return True
    except OSError:
        return False


def _apple_hardware_context() -> dict[str, Any]:
    mem_bytes = _sysctl_int("hw.memsize")
    chip = _sysctl_text("machdep.cpu.brand_string")
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "macos_version": _command_text(["sw_vers", "-productVersion"]),
        "macos_build": _command_text(["sw_vers", "-buildVersion"]),
        "chip": chip,
        "chip_family": _apple_chip_family(chip),
        "hw_model": _sysctl_text("hw.model"),
        "memory_gib": (mem_bytes / (1024**3)) if mem_bytes else None,
        "logical_cpu": _sysctl_int("hw.logicalcpu"),
        "physical_cpu": _sysctl_int("hw.physicalcpu"),
        "perf_cores": _sysctl_int("hw.perflevel0.physicalcpu"),
        "efficiency_cores": _sysctl_int("hw.perflevel1.physicalcpu"),
        "arm64": _sysctl_int("hw.optional.arm64"),
    }


def _software_context() -> dict[str, Any]:
    env = collect_environment(".").to_dict()
    return {
        "python_executable": env.get("python_executable"),
        "python_version": env.get("python_version"),
        "mtplx_version": _package_version("mtplx"),
        "mlx_version": _package_version("mlx"),
        "mlx_lm_version": _package_version("mlx-lm"),
        "numpy_version": _package_version("numpy"),
        "git_branch": env.get("git_branch"),
        "git_status": env.get("git_status"),
    }


def _mlx_backend_context() -> dict[str, Any]:
    # MTPLX runs on stock PyPI MLX; no profile requires an MLX fork or a
    # patched qmm build. Report the imported runtime as a plain diagnostic.
    path = None
    try:
        spec = importlib.util.find_spec("mlx.core")
        path = str(Path(spec.origin).resolve()) if spec and spec.origin else None
    except Exception as exc:  # pragma: no cover - host dependent
        path = f"ERROR: {exc}"
    custom_env = {
        key: os.environ.get(key)
        for key in (
            "MTPLX_GDN_OUT_QMV8",
            "MTPLX_SOURCE_QMM_MANY",
            "MTPLX_SOURCE_QMM_MANY_STRICT",
            "MTPLX_NATIVE_MLP",
            "MTPLX_VERIFY_MLP_FUSED",
        )
        if os.environ.get(key)
    }
    stock_layout = bool(path and ("site-packages" in path or "dist-packages" in path))
    return {
        "mlx_core_path": path,
        "mlx_version": _package_version("mlx"),
        "mlx_lm_version": _package_version("mlx-lm"),
        "stock_mlx_likely": stock_layout,
        "custom_qmv_or_qmm_env": custom_env,
    }


def _ms_per_call(seconds: Any, calls: int) -> float | None:
    if not isinstance(seconds, (int, float)) or not calls:
        return None
    return 1000.0 * float(seconds) / calls


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if not isinstance(numerator, (int, float)) or not isinstance(
        denominator, (int, float)
    ):
        return None
    if float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def _first_number(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _row_metric_values(rows: list[dict[str, Any]], *keys: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _first_number(row, *keys)
        if value is not None:
            values.append(value)
    return values


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def _command_text(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            args,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        ).strip()
    except Exception:
        return None


def _sysctl_text(name: str) -> str | None:
    # Absolute path: app-owned daemons run with a sanitized PATH that lacks
    # /usr/sbin, which silently broke every sysctl-derived hardware detection.
    return _command_text(["/usr/sbin/sysctl", "-n", name])


def _sysctl_int(name: str) -> int | None:
    text = _sysctl_text(name)
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None


def _apple_chip_family(chip: str | None) -> str | None:
    if not chip:
        return None
    words = str(chip).replace("(TM)", "").split()
    if len(words) >= 2 and words[0] == "Apple" and words[1].startswith("M"):
        return " ".join(words[:3]) if len(words) >= 3 else " ".join(words[:2])
    return chip


def _rate_lists(numerators: list[Any], denominators: list[Any]) -> list[float | None]:
    rates: list[float | None] = []
    for numerator, denominator in zip(numerators, denominators):
        den = float(denominator or 0)
        rates.append((float(numerator) / den) if den else None)
    return rates


def _format_depth_acceptance(row: dict[str, Any]) -> str:
    accepted = row.get("accepted_by_depth") or []
    drafted = row.get("drafted_by_depth") or []
    percentages = row.get("acceptance_percent_by_depth") or []
    parts = []
    for index, (acc, draft) in enumerate(zip(accepted, drafted), start=1):
        pct = percentages[index - 1] if index - 1 < len(percentages) else None
        pct_text = _fmt_metric(pct, digits=2)
        parts.append(f"MTP{index} {acc}/{draft} ({pct_text}%)")
    return " | ".join(parts) if parts else "n/a"


def _fmt_metric(value: Any, *, digits: int) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return "n/a"


def _print_serve_start_line(text: str = "") -> None:
    print(text, flush=True)


_PROFILE_SHORT_SUMMARIES = {
    "safe": "Stable: exact/staged long-reply path, no fan control",
    "performance-cold": "Burst: max-fan short-context lane, not recommended beyond 8K context",
    "sustained": "Sustained: long-context native-MTP path with bounded memory",
    "exact": "QA-only exact paged verifier",
    "max-diagnostic": "Diagnostic fan-control profile",
}


def _runtime_mode_display(
    profile_name: str,
    *,
    max_mode: bool = False,
    generation_mode: str | None = None,
) -> str:
    mode = "AR" if str(generation_mode or "").lower() == GENERATION_MODE_AR else "MTP"
    if profile_name == "sustained" and max_mode:
        return f"Sustained Max {mode}"
    if profile_name == "sustained":
        return f"Sustained {mode}"
    if profile_name == "performance-cold" and max_mode:
        return f"Burst {mode}"
    if profile_name == "performance-cold":
        return f"Performance-cold {mode}"
    return f"{profile_name} {mode}"


def _print_serve_start_banner(args: Any) -> None:
    from mtplx.version import DISPLAY_VERSION
    from mtplx.ui import render_banner, render_startup_panel

    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 8000))
    profile_name = getattr(args, "profile", None) or DEFAULT_PROFILE_NAME
    warmup_tokens = int(getattr(args, "warmup_tokens", 16) or 0)
    generation_mode = _generation_mode_from_args(args)
    mode_label = _runtime_mode_display(
        profile_name,
        max_mode=bool(getattr(args, "max", False)),
        generation_mode=generation_mode,
    )
    model_label = getattr(args, "model_id", None) or DEFAULT_PUBLIC_MODEL_ID
    runtime_model = getattr(args, "model", DEFAULT_RUNTIME_MODEL_DIR)
    api_url = f"{_server_url(host, port)}/v1"
    chat_url = _chat_url(host, port)
    api_key = getattr(args, "api_key", None)
    api_note = "API key required" if api_key else "API key: leave blank for localhost"

    extra_lines: list[tuple[str, str]] = [
        ("Listening", bind_label(host, port)),
        ("Loading", str(runtime_model)),
        ("Warmup", f"{warmup_tokens} tokens"),
        ("Auth", api_note),
    ]

    render_banner()
    render_startup_panel(
        version=DISPLAY_VERSION,
        model=model_label,
        profile=profile_name,
        profile_summary=_PROFILE_SHORT_SUMMARIES.get(profile_name),
        api_url=api_url,
        chat_url=chat_url,
        mode_label=mode_label,
        extra_lines=extra_lines,
    )


def _print_serve_handoff(args: Any, runtime_model: str, profile_name: str) -> None:
    if is_wildcard_bind(getattr(args, "host", None)):
        _print_serve_start_line(
            "[1/6] Server config ready: listening on "
            f"{bind_label(args.host, int(args.port))}"
        )
        _print_serve_start_line(
            f"      Local API Base URL: {_server_url(args.host, int(args.port))}/v1"
        )
        network_url = network_url_for_bind(args.host, int(args.port), path="/v1")
        if network_url:
            _print_serve_start_line(
                f"      Network API Base URL: {network_url} (other devices + VM guests)"
            )
    else:
        _print_serve_start_line(
            f"[1/6] Server config ready: {_server_url(args.host, int(args.port))}/v1"
        )
    _print_serve_start_line(f"[2/6] Model resolved: {runtime_model}")
    _print_serve_start_line(f"[3/6] Runtime contract verified — profile: {profile_name}")
    _print_serve_start_line(
        "      Loading the model can take about a minute on first start."
    )
    _print_serve_start_line()


def _server_command_name(args: Any) -> str:
    command = str(getattr(args, "command", None) or "quickstart")
    if command == "quick-start":
        return "quickstart"
    if command in {"quickstart", "serve"}:
        return command
    return "quickstart"


_SECRET_COMMAND_VALUE_FLAGS = {"--api-key"}
_SECRET_ENV_NAME_PARTS = ("API_KEY", "AUTH", "PASSWORD", "SECRET", "TOKEN")


def _redact_command_tokens(cmd: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for raw_token in cmd:
        token = str(raw_token)
        if redact_next:
            redacted.append("[redacted]")
            redact_next = False
            continue
        for flag in _SECRET_COMMAND_VALUE_FLAGS:
            prefix = flag + "="
            if token.startswith(prefix):
                redacted.append(prefix + "[redacted]")
                break
        else:
            redacted.append(token)
            if token in _SECRET_COMMAND_VALUE_FLAGS:
                redact_next = True
    return redacted


def _serve_dry_run_env_delta(env: dict[str, str]) -> dict[str, str]:
    delta: dict[str, str] = {}
    for key, value in sorted(env.items()):
        if not key.startswith("MTPLX_"):
            continue
        if os.environ.get(key) == value:
            continue
        if any(part in key.upper() for part in _SECRET_ENV_NAME_PARTS):
            delta[key] = "[redacted]"
        else:
            delta[key] = str(value)
    return delta


def _serve_dry_run_payload(
    args: Any,
    *,
    runtime_model: str,
    profile_name: str,
    model_id: str,
    generation_mode: str,
    cmd: list[str],
    env: dict[str, str],
) -> dict[str, Any]:
    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 8000))
    base_url = _server_url(host, port)
    argv = _redact_command_tokens(cmd)
    payload: dict[str, Any] = {
        "dry_run": True,
        "target": "server",
        "command": _server_command_name(args),
        "model": str(runtime_model),
        "model_id": str(model_id),
        "profile": str(profile_name),
        "host": host,
        "port": port,
        "api_base_url": f"{base_url}/v1",
        "chat_url": _chat_url(host, port),
        "fan_mode": str(getattr(args, "fan_mode", "default") or "default"),
        "generation_mode": str(generation_mode),
        "depth": int(getattr(args, "depth", 3)),
        "argv": argv,
        "server_command": shlex.join(argv),
        "env": _serve_dry_run_env_delta(env),
    }
    if bool(getattr(args, "download", False)):
        payload["download_requested"] = True
    return payload


def _print_serve_dry_run_human(payload: dict[str, Any]) -> None:
    _print_serve_start_line("MTPLX quickstart dry run")
    _print_serve_start_line(f"Model: {payload['model']}")
    _print_serve_start_line(f"OpenAI API Base URL: {payload['api_base_url']}")
    _print_serve_start_line("Server command:")
    _print_serve_start_line(f"  {payload['server_command']}")


def _serve_should_onboard(args: Any) -> bool:
    """Return whether bare interactive ``mtplx serve`` should run setup."""

    if getattr(args, "command", None) != "serve":
        return False
    if bool(getattr(args, "yes", False)):
        return False
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    cli_flags = getattr(args, "_cli_flags", set()) or set()
    if {"model", "model-id", "profile", "max", "fan-mode"} & set(cli_flags):
        return False
    return True


def _model_ref_from_public_model_id(model_id: str | None) -> str | None:
    text = str(model_id or "").strip()
    if not text:
        return None
    key = text.replace("_", "-").lower()
    lookup_keys = [key]
    if key.startswith("mtplx/"):
        lookup_keys.append(key.split("/", 1)[1])
    sanitized_key = re.sub(r"[^a-z0-9.-]+", "-", key).strip("-.")
    if sanitized_key:
        lookup_keys.append(sanitized_key)
    mapping = {
        DEFAULT_PUBLIC_MODEL_ID.lower(): DEFAULT_HF_MODEL_ID,
        DEFAULT_HF_MODEL_ID.lower(): DEFAULT_HF_MODEL_ID,
        DEFAULT_MODEL_ID.lower(): DEFAULT_HF_MODEL_ID,
        Path(DEFAULT_HF_MODEL_ID).name.lower(): DEFAULT_HF_MODEL_ID,
        OPTIMIZED_SPEED_V1_PUBLIC_MODEL_ID.lower(): OPTIMIZED_SPEED_V1_HF_MODEL_ID,
        OPTIMIZED_SPEED_V1_HF_MODEL_ID.lower(): OPTIMIZED_SPEED_V1_HF_MODEL_ID,
        Path(
            OPTIMIZED_SPEED_V1_HF_MODEL_ID
        ).name.lower(): OPTIMIZED_SPEED_V1_HF_MODEL_ID,
        OPTIMIZED_SPEED_V2_PUBLIC_MODEL_ID.lower(): OPTIMIZED_SPEED_V2_HF_MODEL_ID,
        OPTIMIZED_SPEED_V2_HF_MODEL_ID.lower(): OPTIMIZED_SPEED_V2_HF_MODEL_ID,
        Path(
            OPTIMIZED_SPEED_V2_HF_MODEL_ID
        ).name.lower(): OPTIMIZED_SPEED_V2_HF_MODEL_ID,
        DEFAULT_FP16_PUBLIC_MODEL_ID.lower(): DEFAULT_FP16_HF_MODEL_ID,
        DEFAULT_FP16_HF_MODEL_ID.lower(): DEFAULT_FP16_HF_MODEL_ID,
        Path(DEFAULT_FP16_HF_MODEL_ID).name.lower(): DEFAULT_FP16_HF_MODEL_ID,
        LEGACY_OPTIMIZED_PUBLIC_MODEL_ID.lower(): LEGACY_OPTIMIZED_HF_MODEL_ID,
        LEGACY_OPTIMIZED_HF_MODEL_ID.lower(): LEGACY_OPTIMIZED_HF_MODEL_ID,
        Path(LEGACY_OPTIMIZED_HF_MODEL_ID).name.lower(): LEGACY_OPTIMIZED_HF_MODEL_ID,
        QUALITY_PUBLIC_MODEL_ID.lower(): QUALITY_HF_MODEL_ID,
        QUALITY_HF_MODEL_ID.lower(): QUALITY_HF_MODEL_ID,
        Path(QUALITY_HF_MODEL_ID).name.lower(): QUALITY_HF_MODEL_ID,
        QUALITY_FP16_PUBLIC_MODEL_ID.lower(): QUALITY_FP16_HF_MODEL_ID,
        QUALITY_FP16_HF_MODEL_ID.lower(): QUALITY_FP16_HF_MODEL_ID,
        Path(QUALITY_FP16_HF_MODEL_ID).name.lower(): QUALITY_FP16_HF_MODEL_ID,
        QWEN35_9B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID.lower(): QWEN35_9B_OPTIMIZED_SPEED_HF_MODEL_ID,
        QWEN35_9B_OPTIMIZED_SPEED_HF_MODEL_ID.lower(): QWEN35_9B_OPTIMIZED_SPEED_HF_MODEL_ID,
        Path(
            QWEN35_9B_OPTIMIZED_SPEED_HF_MODEL_ID
        ).name.lower(): QWEN35_9B_OPTIMIZED_SPEED_HF_MODEL_ID,
        QWEN35_9B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID.lower(): QWEN35_9B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
        QWEN35_9B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID.lower(): QWEN35_9B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
        Path(
            QWEN35_9B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID
        ).name.lower(): QWEN35_9B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
        QWEN36_35B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID.lower(): QWEN36_35B_OPTIMIZED_SPEED_HF_MODEL_ID,
        QWEN36_35B_OPTIMIZED_SPEED_HF_MODEL_ID.lower(): QWEN36_35B_OPTIMIZED_SPEED_HF_MODEL_ID,
        Path(
            QWEN36_35B_OPTIMIZED_SPEED_HF_MODEL_ID
        ).name.lower(): QWEN36_35B_OPTIMIZED_SPEED_HF_MODEL_ID,
        QWEN36_35B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID.lower(): QWEN36_35B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
        QWEN36_35B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID.lower(): QWEN36_35B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
        Path(
            QWEN36_35B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID
        ).name.lower(): QWEN36_35B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
        QWEN36_35B_OPTIMIZED_BALANCE_PUBLIC_MODEL_ID.lower(): QWEN36_35B_OPTIMIZED_BALANCE_HF_MODEL_ID,
        QWEN36_35B_OPTIMIZED_BALANCE_HF_MODEL_ID.lower(): QWEN36_35B_OPTIMIZED_BALANCE_HF_MODEL_ID,
        Path(
            QWEN36_35B_OPTIMIZED_BALANCE_HF_MODEL_ID
        ).name.lower(): QWEN36_35B_OPTIMIZED_BALANCE_HF_MODEL_ID,
        QWEN36_35B_OPTIMIZED_BALANCE_FP16_PUBLIC_MODEL_ID.lower(): QWEN36_35B_OPTIMIZED_BALANCE_FP16_HF_MODEL_ID,
        QWEN36_35B_OPTIMIZED_BALANCE_FP16_HF_MODEL_ID.lower(): QWEN36_35B_OPTIMIZED_BALANCE_FP16_HF_MODEL_ID,
        Path(
            QWEN36_35B_OPTIMIZED_BALANCE_FP16_HF_MODEL_ID
        ).name.lower(): QWEN36_35B_OPTIMIZED_BALANCE_FP16_HF_MODEL_ID,
        QWEN38_BARE_SPEED_PUBLIC_MODEL_ID.lower(): QWEN38_BARE_SPEED_HF_MODEL_ID,
        QWEN38_BARE_SPEED_HF_MODEL_ID.lower(): QWEN38_BARE_SPEED_HF_MODEL_ID,
        Path(
            QWEN38_BARE_SPEED_HF_MODEL_ID
        ).name.lower(): QWEN38_BARE_SPEED_HF_MODEL_ID,
        QWEN38_OPTIMIZED_SPEED_PUBLIC_MODEL_ID.lower(): QWEN38_OPTIMIZED_SPEED_HF_MODEL_ID,
        QWEN38_OPTIMIZED_SPEED_HF_MODEL_ID.lower(): QWEN38_OPTIMIZED_SPEED_HF_MODEL_ID,
        Path(
            QWEN38_OPTIMIZED_SPEED_HF_MODEL_ID
        ).name.lower(): QWEN38_OPTIMIZED_SPEED_HF_MODEL_ID,
        QWEN38_OPTIMIZED_QUALITY_PUBLIC_MODEL_ID.lower(): QWEN38_OPTIMIZED_QUALITY_HF_MODEL_ID,
        QWEN38_OPTIMIZED_QUALITY_HF_MODEL_ID.lower(): QWEN38_OPTIMIZED_QUALITY_HF_MODEL_ID,
        Path(
            QWEN38_OPTIMIZED_QUALITY_HF_MODEL_ID
        ).name.lower(): QWEN38_OPTIMIZED_QUALITY_HF_MODEL_ID,
        QWEN38_BARE_SPEED_FP16_PUBLIC_MODEL_ID.lower(): QWEN38_BARE_SPEED_FP16_HF_MODEL_ID,
        QWEN38_BARE_SPEED_FP16_HF_MODEL_ID.lower(): QWEN38_BARE_SPEED_FP16_HF_MODEL_ID,
        Path(
            QWEN38_BARE_SPEED_FP16_HF_MODEL_ID
        ).name.lower(): QWEN38_BARE_SPEED_FP16_HF_MODEL_ID,
        QWEN38_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID.lower(): QWEN38_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
        QWEN38_OPTIMIZED_SPEED_FP16_HF_MODEL_ID.lower(): QWEN38_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
        Path(
            QWEN38_OPTIMIZED_SPEED_FP16_HF_MODEL_ID
        ).name.lower(): QWEN38_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
        QWEN38_OPTIMIZED_QUALITY_FP16_PUBLIC_MODEL_ID.lower(): QWEN38_OPTIMIZED_QUALITY_FP16_HF_MODEL_ID,
        QWEN38_OPTIMIZED_QUALITY_FP16_HF_MODEL_ID.lower(): QWEN38_OPTIMIZED_QUALITY_FP16_HF_MODEL_ID,
        Path(
            QWEN38_OPTIMIZED_QUALITY_FP16_HF_MODEL_ID
        ).name.lower(): QWEN38_OPTIMIZED_QUALITY_FP16_HF_MODEL_ID,
        "qwen3.6-35b-a3b-mtplx-official4-cyankiwimtp-cleanrecipe": QWEN36_35B_OPTIMIZED_SPEED_HF_MODEL_ID,
    }
    for candidate in lookup_keys:
        model_ref = mapping.get(candidate)
        if model_ref:
            return model_ref
    return None


def _apply_model_id_as_model_default(args: Any, *, has_explicit_model: bool) -> bool:
    """Use known MTPLX public model ids as model refs when --model is omitted."""

    if has_explicit_model:
        return False
    cli_flags = getattr(args, "_cli_flags", set()) or set()
    if "model-id" not in cli_flags:
        return False
    model_ref = _model_ref_from_public_model_id(getattr(args, "model_id", None))
    if not model_ref:
        return False
    args.model = model_ref
    args._model_from_model_id = True
    return True


def _resolve_runtime_options_on_args(
    args: Any,
    *,
    printer: Callable[[str], None],
) -> int | None:
    # --api-key-file pointing at a missing path creates the file with a fresh
    # key instead of erroring: the non-localhost refusal below suggests
    # `--api-key-file ~/.mtplx/api-key` as the recovery command, and that
    # suggestion must work on a machine that has never had a key. The key is
    # printed once, at generation — it never appears again on later launches,
    # and the file path stays the durable copy. Only server entrypoints get
    # this behavior; read-only key-file consumers keep strict missing-file
    # errors.
    api_key_file = getattr(args, "api_key_file", None)
    if api_key_file and not getattr(args, "api_key", None):
        key_path = Path(str(api_key_file)).expanduser()
        if not key_path.exists():
            try:
                generated = generate_api_key_file(key_path)
            except OSError as exc:
                printer(f"error: could not create API key file: {exc}")
                return 2
            printer(f"Generated a new API key and saved it to {key_path}")
            printer(f"API key (use as Bearer token from other machines): {generated}")
    try:
        resolved_key = resolve_api_key(
            explicit_api_key=getattr(args, "api_key", None),
            api_key_file=getattr(args, "api_key_file", None),
        )
    except (OSError, ValueError) as exc:
        printer(f"error: {exc}")
        return 2
    setattr(args, "api_key", resolved_key.value)
    setattr(args, "api_key_source", resolved_key.source)
    try:
        kv_mode = normalize_paged_kv_quantization(
            getattr(args, "paged_kv_quantization", None),
            allow_none=True,
        )
        if kv_mode is None:
            # No explicit --paged-kv-quantization: inherit the launcher's
            # environment. The app (and any wrapper tooling) communicates the
            # KV-quant choice through the env pair, and rewriting the absent
            # flag to an explicit "off" here used to clobber that env in the
            # rebuilt child argv/env, killing the toggle engine-wide. An
            # explicit flag still wins over the environment.
            kv_mode = paged_kv_quant_mode_from_env()
    except ValueError as exc:
        printer(f"error: {exc}")
        return 2
    setattr(args, "paged_kv_quantization", kv_mode)
    return None


def cmd_serve_public(args: Any) -> int:
    dry_run = bool(getattr(args, "dry_run", False))
    quiet_json = dry_run and bool(getattr(args, "json", False))
    agent_rewrites = getattr(args, "agent_rewrites", None)
    if agent_rewrites:
        # The server child inherits os.environ; the env var is the single
        # source of truth so in-process helpers and the spawned daemon agree.
        os.environ["MTPLX_AGENT_REWRITES"] = str(agent_rewrites)
    runtime_options_error = _resolve_runtime_options_on_args(
        args,
        printer=_print_serve_start_line,
    )
    if runtime_options_error is not None:
        return runtime_options_error
    try:
        fan_mode = _fan_mode_from_args(args)
    except ValueError as exc:
        _print_serve_start_line(f"error: {exc}")
        return 2
    api_key = getattr(args, "api_key", None)
    if not _is_localhost_bind(getattr(args, "host", None)) and not api_key:
        payload = {
            "error": "--api-key or --api-key-file is required when --host is not localhost",
            "host": getattr(args, "host", None),
        }
        if getattr(args, "json", False):
            _print(payload)
        else:
            print(
                "error: --api-key or --api-key-file is required when --host is not localhost"
            )
            print(f"host: {getattr(args, 'host', None)}")
            server_command = _server_command_name(args)
            print(f"try: mtplx {server_command} --host 127.0.0.1")
            print(
                f"try: mtplx {server_command} --host 0.0.0.0 --api-key-file ~/.mtplx/api-key"
            )
        return 2
    cli_flags = getattr(args, "_cli_flags", set()) or set()
    _apply_model_id_as_model_default(
        args,
        has_explicit_model="model" in cli_flags,
    )
    if _serve_should_onboard(args):
        from mtplx.ui.onboarding import PROFILE_AUTO, run_serve_flow

        choice = run_serve_flow(
            configured_model=getattr(args, "model", None),
            host=str(getattr(args, "host", "127.0.0.1")),
            port=int(getattr(args, "port", 8000)),
            default_open_browser=bool(getattr(args, "open_browser", False)),
        )
        if choice is None:
            _print_serve_start_line("aborted")
            return 130
        chosen_model = choice.get("model")
        if chosen_model:
            args.model = chosen_model
            # Explicitness markers: see the matching stamp in
            # cmd_quickstart_public (issues #279/#280 — a wizard pick must
            # never be re-resolved to the hardware default downstream).
            args._model_explicit = True
            args._cli_flags = set(getattr(args, "_cli_flags", set()) or set())
            args._cli_flags.add("model")
            try:
                from mtplx.hf_loader import repo_id_from_model_ref

                if repo_id_from_model_ref(chosen_model):
                    args.download = True
            except Exception:
                pass
        chosen_profile = choice.get("profile")
        if chosen_profile and chosen_profile != PROFILE_AUTO:
            args.profile = chosen_profile
            # An explicit wizard pick is a user decision: record it so
            # per-model default-profile resolution never overrides it. Auto
            # is the opposite decision — stamp nothing, so the engine keeps
            # resolving the launch profile per model.
            args._cli_flags = set(getattr(args, "_cli_flags", set()) or set())
            args._cli_flags.add("profile")
        args.max = bool(choice.get("max"))
        args.fan_mode = FAN_MODE_MAX if args.max else "default"
        args.open_browser = bool(choice.get("open_browser"))
        args._onboarded = True
    depth_error = _validate_public_depth(args, printer=_print_serve_start_line)
    if depth_error is not None:
        return depth_error
    generation_mode = _generation_mode_from_args(args)
    fan_mode = _fan_mode_from_args(args)
    if (
        generation_mode == GENERATION_MODE_MTP
        and getattr(args, "load_mtp", True) is False
    ):
        _print_serve_start_line("error: --generation-mode mtp requires --load-mtp")
        _print_serve_start_line("try: mtplx serve --generation-mode ar --no-load-mtp")
        return 2
    args.model_id = _public_model_id_for_args(
        args,
        str(getattr(args, "model", "")),
    )
    # Resolve the per-model default profile BEFORE any display surface
    # renders. The startup banner used to print the raw parser default
    # ("Profile sustained") while serve-time resolution then launched
    # turbo; users echoed the banner in bug reports and pinned
    # --profile sustained explicitly, really landing on the slow path.
    _apply_model_default_profile(args, str(args.model_id))
    if not quiet_json:
        _print_serve_start_banner(args)
    if not dry_run and _port_is_busy(
        str(getattr(args, "host", "127.0.0.1")), int(getattr(args, "port", 8000))
    ):
        if bool(getattr(args, "quickstart_pi", False)):
            base = _server_url(
                str(getattr(args, "host", "127.0.0.1")),
                int(getattr(args, "port", 8000)),
            )
            health = _http_json(base + "/health", timeout=1.5, api_key=api_key)
            if health.get("ok"):
                from mtplx.pi import pi_launch_command, pi_model_ref

                model_id = (
                    health.get("model")
                    or getattr(args, "model_id", None)
                    or DEFAULT_PUBLIC_MODEL_ID
                )
                _print_serve_start_line("MTPLX is already running.")
                _print_serve_start_line(f"OpenAI API Base URL: {base}/v1")
                _print_serve_start_line(f"Pi model: {pi_model_ref(str(model_id))}")
                _quickstart_launch_pi_now(model_id=str(model_id))
                _print_serve_start_line(
                    f"Manual fallback: {pi_launch_command(str(model_id))}"
                )
                _print_serve_start_line(
                    "Use the existing server, or stop that terminal with Ctrl-C to restart."
                )
                return 0
        if bool(getattr(args, "quickstart_openwebui", False)):
            base = _server_url(
                str(getattr(args, "host", "127.0.0.1")),
                int(getattr(args, "port", 8000)),
            )
            health = _http_json(base + "/health", timeout=1.5, api_key=api_key)
            if health.get("ok"):
                model_id = (
                    health.get("model")
                    or getattr(args, "model_id", None)
                    or DEFAULT_PUBLIC_MODEL_ID
                )
                chat_url = _chat_url(
                    str(getattr(args, "host", "127.0.0.1")),
                    int(getattr(args, "port", 8000)),
                )
                _print_serve_start_line("MTPLX is already running.")
                _print_serve_start_line(f"Chat URL: {chat_url}")
                _print_serve_start_line(f"OpenAI API Base URL: {base}/v1")
                _print_serve_start_line(f"Model: {model_id}")
                _print_serve_start_line("API key: leave blank for localhost")
                _print_serve_start_line("Opening chat UI in your browser...")
                _open_browser_url(chat_url)
                _print_serve_start_line(
                    "Use the existing server, or stop that terminal with Ctrl-C to restart."
                )
                return 0
        if bool(getattr(args, "quickstart_opencode", False)):
            base = _server_url(
                str(getattr(args, "host", "127.0.0.1")),
                int(getattr(args, "port", 8000)),
            )
            health = _http_json(base + "/health", timeout=1.5, api_key=api_key)
            if health.get("ok"):
                model_id = (
                    health.get("model")
                    or getattr(args, "model_id", None)
                    or DEFAULT_PUBLIC_MODEL_ID
                )
                _print_serve_start_line("MTPLX is already running.")
                _print_serve_start_line(f"OpenAI API Base URL: {base}/v1")
                _print_serve_start_line(f"OpenCode model: mtplx/{model_id}")
                _quickstart_launch_opencode_now()
                _print_serve_start_line(
                    "Use the existing server, or stop that terminal with Ctrl-C to restart."
                )
                return 0
        if bool(getattr(args, "quickstart_swival", False)):
            base = _server_url(
                str(getattr(args, "host", "127.0.0.1")),
                int(getattr(args, "port", 8000)),
            )
            health = _http_json(base + "/health", timeout=1.5, api_key=api_key)
            if health.get("ok"):
                from mtplx.swival import shell_swival_command

                model_id = (
                    health.get("model")
                    or getattr(args, "model_id", None)
                    or DEFAULT_PUBLIC_MODEL_ID
                )
                context_window = int(health.get("context_window") or 262144)
                _print_serve_start_line("MTPLX is already running.")
                _print_serve_start_line(f"OpenAI API Base URL: {base}/v1")
                _print_serve_start_line(
                    "Swival command: "
                    + shell_swival_command(
                        base_url=base,
                        model_id=str(model_id),
                        context_window=context_window,
                    )
                )
                _print_serve_start_line(
                    "Use the existing server, or stop that terminal with Ctrl-C to restart."
                )
                return 0
        if bool(getattr(args, "quickstart_hermes", False)):
            base = _server_url(
                str(getattr(args, "host", "127.0.0.1")),
                int(getattr(args, "port", 8000)),
            )
            health = _http_json(base + "/health", timeout=1.5, api_key=api_key)
            if health.get("ok"):
                command = str(getattr(args, "hermes_launch_command", "") or "").strip()
                model_id = (
                    health.get("model")
                    or getattr(args, "model_id", None)
                    or DEFAULT_PUBLIC_MODEL_ID
                )
                _print_serve_start_line("MTPLX is already running.")
                _print_serve_start_line(f"OpenAI API Base URL: {base}/v1")
                _print_serve_start_line(f"Hermes model: {model_id}")
                if command:
                    from mtplx.pi import launch_pi_in_terminal

                    result = launch_pi_in_terminal(command)
                    if result.get("ok"):
                        _print_serve_start_line("Opening Hermes Agent in Terminal...")
                    else:
                        _print_serve_start_line(
                            f"Could not open Hermes automatically: {result.get('error')}"
                        )
                        _print_serve_start_line(f"Manual fallback: {command}")
                else:
                    _print_serve_start_line(
                        "Open Hermes manually with the MTPLX profile."
                    )
                _print_serve_start_line(
                    "Use the existing server, or stop that terminal with Ctrl-C to restart."
                )
                return 0
        _print_serve_start_line(f"error: port {int(args.port)} is already in use")
        try:
            from mtplx.daemon_client import classify_port_occupant, port_busy_advice

            occupant = classify_port_occupant(
                str(getattr(args, "host", "127.0.0.1")),
                int(getattr(args, "port", 8000)),
                api_key=api_key,
            )
            for line in port_busy_advice(occupant, port=int(args.port)):
                _print_serve_start_line(line)
        except Exception:
            _print_serve_start_line("try: mtplx status")
        server_command = _server_command_name(args)
        # Include --profile in the retry advice only when it carries real
        # signal (explicit flag or a non-default from user config). Echoing
        # the raw parser default here baked "--profile sustained" into copy-
        # paste advice and pinned 27B users off the turbo model-default
        # (2026-07-16). No model resolution may happen on this path.
        advice_profile = str(getattr(args, "profile", None) or "")
        profile_arg = (
            f" --profile {advice_profile}"
            if advice_profile
            and (
                advice_profile != DEFAULT_PROFILE_NAME
                or "profile" in (getattr(args, "_cli_flags", set()) or set())
            )
            else ""
        )
        max_arg = " --max" if bool(getattr(args, "max", False)) else ""
        _print_serve_start_line(
            f"try: mtplx {server_command}{profile_arg}{max_arg} --port {int(args.port) + 1}"
        )
        return 2
    profile = get_profile(_resolved_default_profile_name(args))
    cache_dir = getattr(args, "cache_dir", None)
    if bool(getattr(args, "download", False)) and not dry_run:
        try:
            runtime_model, resolution = _quickstart_resolve_model(
                args.model,
                cache_dir=cache_dir,
                download=True,
            )
        except KeyboardInterrupt:
            _print_serve_start_line("download cancelled")
            return 130
        except Exception as exc:
            _print_serve_start_line(f"error: {exc}")
            return 1
        if runtime_model is None:
            if resolution.get("cancelled"):
                _print_serve_start_line("download cancelled")
                return 130
            gate_inspection = resolution.get("gate_inspection")
            if isinstance(gate_inspection, dict):
                _print_model_gate_error(
                    gate_inspection,
                    printer=_print_serve_start_line,
                    json_output=bool(getattr(args, "json", False)),
                )
                compatibility = gate_inspection.get("compatibility") or {}
                return int(compatibility.get("exit_code") or 1)
            resolution_error = resolution.get("error")
            if isinstance(resolution_error, dict):
                _print_command_error(
                    resolution_error,
                    command=_server_command_name(args),
                    json_output=bool(getattr(args, "json", False)),
                )
            else:
                _print_serve_start_line("error: model is not available locally")
            return 1
        if resolution.get("downloaded"):
            _print_serve_start_line(f"downloaded: {resolution.get('download_ref')}")
    else:
        runtime_model, resolve_error = _resolve_runtime_model_path(
            args.model,
            cache_dir=cache_dir,
        )
        if resolve_error is not None:
            _print_command_error(
                resolve_error,
                command=_server_command_name(args),
                json_output=bool(getattr(args, "json", False)),
            )
            return 1
    inspection, gate_exit = _model_gate(
        runtime_model,
        unsafe_force_unverified=bool(getattr(args, "unsafe_force_unverified", False)),
        yes=bool(getattr(args, "yes", False)),
    )
    if gate_exit is not None:
        _print_model_gate_error(inspection, printer=_print_serve_start_line)
        return gate_exit
    mode_exit = _apply_runtime_compatibility_mode(
        args,
        inspection,
        printer=_print_serve_start_line,
    )
    if mode_exit is not None:
        return mode_exit
    # The missing-MTP degrade above may have flipped args to AR; the local
    # snapshot taken before model resolution would otherwise hand the child
    # a stale "--generation-mode mtp" with load_mtp already stripped.
    generation_mode = _generation_mode_from_args(args)
    model_id = _public_model_id_for_args(args, str(runtime_model))
    args.model_id = model_id
    if _apply_model_default_profile(args, model_id):
        profile = get_profile(args.profile)
    _apply_model_contract_depth_default(args, inspection, profile)
    _apply_backend_serve_defaults(args, inspection)
    _apply_qwen36_35b_optimized_speed_defaults(args, model_id)
    backend_descriptor = descriptor_from_inspection(inspection)
    draft_lm_head = _model_draft_lm_head_spec(inspection, profile) or {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
    }
    draft_sampler = _model_draft_sampler_spec(inspection, profile)
    draft_sampler_override = _explicit_draft_sampler_override(args, draft_sampler)
    if draft_sampler_override is not None:
        draft_sampler = draft_sampler_override
    if not quiet_json:
        _print_serve_handoff(args, runtime_model, profile.name)
    cmd = [
        sys.executable,
        "-m",
        "mtplx.server.openai",
        "--model",
        runtime_model,
        "--backend-id",
        backend_descriptor.backend_id,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--depth",
        str(args.depth),
        "--generation-mode",
        generation_mode,
        "--profile",
        profile.name,
        "--reasoning-mode",
        _reasoning_mode(args, default="auto"),
        "--preserve-thinking",
        _preserve_thinking_policy(args),
        "--verify-strategy",
        str(getattr(args, "verify_strategy", "capture_commit") or "capture_commit"),
        "--verify-core",
        str(
            getattr(args, "verify_core", "linear-gdn-from-conv-tape")
            or "linear-gdn-from-conv-tape"
        ),
        "--draft-lm-head-bits",
        str(draft_lm_head["bits"]),
        "--draft-lm-head-group-size",
        str(draft_lm_head["group_size"]),
        "--draft-lm-head-mode",
        str(draft_lm_head["mode"]),
        "--rate-limit",
        str(getattr(args, "rate_limit", 0)),
        "--stream-interval",
        str(getattr(args, "stream_interval", 1)),
        "--scheduler-mode",
        str(getattr(args, "scheduler_mode", "serial") or "serial"),
        "--batching-preset",
        str(getattr(args, "batching_preset", "latency") or "latency"),
        "--mtp-batch-numerics",
        str(getattr(args, "mtp_batch_numerics", "throughput") or "throughput"),
        "--warmup-tokens",
        str(getattr(args, "warmup_tokens", 16)),
        "--model-id",
        str(model_id),
        "--paged-kv-quantization",
        str(getattr(args, "paged_kv_quantization", "off") or "off"),
        "--fan-mode",
        fan_mode,
    ]
    for attr, flag in (
        ("max_active_requests", "--max-active-requests"),
        ("decode_batch_max", "--decode-batch-max"),
        ("batch_wait_ms", "--batch-wait-ms"),
        ("prefill_chunk_tokens", "--prefill-chunk-tokens"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            cmd.extend([flag, str(value)])
    # Retrieval models. The server runs as a subprocess with an explicitly
    # rebuilt argv, so anything not forwarded here never reaches it — the
    # endpoints would stay unconfigured on every path but a bare `mtplx serve`.
    for flag, attr in (("--embedding-model", "embedding_model"), ("--reranker-model", "reranker_model")):
        for reference in getattr(args, attr, None) or []:
            if str(reference).strip():
                cmd.extend([flag, str(reference)])
    for flag, attr in (
        ("--retrieval-max-resident", "retrieval_max_resident"),
        ("--retrieval-max-tokens", "retrieval_max_tokens"),
        ("--retrieval-idle-timeout", "retrieval_idle_timeout"),
    ):
        value = getattr(args, attr, None)
        if value:
            cmd.extend([flag, str(value)])
    if bool(getattr(args, "retrieval_trust_remote_code", False)):
        cmd.append("--retrieval-trust-remote-code")
    # The 2.5.4 notes promised `mtplx serve --no-auth`; the flag lived only on
    # the server module's parser until this forward existed, so the promised
    # spelling died at this parser with an argparse error.
    if bool(getattr(args, "no_auth", False)):
        cmd.append("--no-auth")
    # The chat model is already an absolute path by this point, but retrieval
    # references are resolved inside the server, which has no cache directory
    # of its own — so a model pulled into a custom --cache-dir would not be
    # found unless the directory travels with them.
    retrieval_cache_dir = getattr(args, "cache_dir", None)
    if retrieval_cache_dir and (
        getattr(args, "embedding_model", None) or getattr(args, "reranker_model", None)
    ):
        cmd.extend(["--retrieval-cache-dir", str(retrieval_cache_dir)])
    context_window = getattr(args, "context_window", None)
    if context_window is not None:
        cmd.extend(["--context-window", str(context_window)])
    mtp_adapter = getattr(args, "mtp_adapter", None)
    if mtp_adapter:
        cmd.extend(["--mtp-adapter", str(mtp_adapter)])
    if bool(getattr(args, "merge_mtp_adapter", False)):
        cmd.append("--merge-mtp-adapter")
    mtp_quant_bits = getattr(args, "mtp_quant_bits", None)
    if mtp_quant_bits is not None:
        cmd.extend(["--mtp-quant-bits", str(mtp_quant_bits)])
        cmd.extend(
            [
                "--mtp-quant-group-size",
                str(getattr(args, "mtp_quant_group_size", 64) or 64),
            ]
        )
        cmd.extend(
            [
                "--mtp-quant-mode",
                str(getattr(args, "mtp_quant_mode", "affine") or "affine"),
            ]
        )
    if bool(getattr(args, "experimental_mtp_cohorts", False)):
        cmd.append("--experimental-mtp-cohorts")
    ssd_session_cache = str(getattr(args, "ssd_session_cache", "on") or "on")
    cmd.extend(["--ssd-session-cache", ssd_session_cache])
    ssd_dir = getattr(args, "ssd_session_cache_dir", None)
    if ssd_dir:
        cmd.extend(["--ssd-session-cache-dir", str(ssd_dir)])
    ssd_max_size = getattr(args, "ssd_session_cache_max_size", None)
    if ssd_max_size:
        cmd.extend(["--ssd-session-cache-max-size", str(ssd_max_size)])
    ssd_min_prefix = getattr(args, "ssd_session_cache_min_prefix_tokens", None)
    if ssd_min_prefix is not None:
        cmd.extend(["--ssd-session-cache-min-prefix-tokens", str(ssd_min_prefix)])
    adaptive_policy = str(getattr(args, "adaptive_policy", "none") or "none")
    if adaptive_policy != "none":
        cmd.extend(["--adaptive-policy", adaptive_policy])
        cmd.extend(
            [
                "--adaptive-min-depth",
                str(getattr(args, "adaptive_min_depth", 1)),
            ]
        )
        if adaptive_policy == "streak":
            for attr, flag in (
                ("adaptive_start_depth", "--adaptive-start-depth"),
                ("adaptive_increase_after", "--adaptive-increase-after"),
                ("adaptive_decrease_after", "--adaptive-decrease-after"),
            ):
                cmd.extend([flag, str(getattr(args, attr))])
        elif adaptive_policy == "position_ema":
            cmd.extend(
                [
                    "--adaptive-position-depth-cap",
                    str(getattr(args, "adaptive_position_depth_cap", 4)),
                ]
            )
        elif adaptive_policy == "expected_value":
            for attr, flag in (
                ("adaptive_ev_base_depth", "--adaptive-ev-base-depth"),
                ("adaptive_ev_accept_priors", "--adaptive-ev-accept-priors"),
                ("adaptive_ev_draft_cost_s", "--adaptive-ev-draft-cost-s"),
                (
                    "adaptive_ev_extra_verify_cost_s",
                    "--adaptive-ev-extra-verify-cost-s",
                ),
                ("adaptive_ev_baseline_tok_s", "--adaptive-ev-baseline-tok-s"),
                ("adaptive_ev_safety_margin", "--adaptive-ev-safety-margin"),
                ("adaptive_ev_margin_center", "--adaptive-ev-margin-center"),
                ("adaptive_ev_margin_scale", "--adaptive-ev-margin-scale"),
                ("adaptive_ev_confidence_weight", "--adaptive-ev-confidence-weight"),
                (
                    "adaptive_ev_min_extra_accept_probability",
                    "--adaptive-ev-min-extra-accept-probability",
                ),
                (
                    "adaptive_ev_warmup_full_depth_cycles",
                    "--adaptive-ev-warmup-full-depth-cycles",
                ),
                (
                    "adaptive_ev_exploration_interval",
                    "--adaptive-ev-exploration-interval",
                ),
            ):
                cmd.extend([flag, str(getattr(args, attr))])
    qwen38_q4_mtp_block = getattr(args, "qwen38_q4_mtp_block", None)
    if qwen38_q4_mtp_block is not None:
        cmd.extend(["--qwen38-q4-mtp-block", str(qwen38_q4_mtp_block)])
    if draft_sampler is not None:
        # Provenance for the dynamic draft-temperature curve: only a
        # user-typed CLI draft flag pins the sampler. Injected measured
        # defaults and the app's boilerplate launch flag (the app always
        # emits --draft-temperature from its preset/target mirror,
        # identified by --app-launch-id) are curve anchors, not pins.
        user_typed_draft = any(
            flag in (getattr(args, "_cli_flags", set()) or set())
            for flag in _DRAFT_SAMPLER_FLAG_ATTRS
        )
        launched_by_app = bool(
            str(getattr(args, "app_launch_id", "") or "").strip()
        )
        cmd.extend(
            [
                "--draft-temperature",
                str(float(draft_sampler["temperature"])),
                "--draft-top-p",
                str(float(draft_sampler["top_p"])),
                "--draft-top-k",
                str(int(draft_sampler["top_k"])),
                "--draft-sampler-source",
                (
                    "explicit"
                    if user_typed_draft and not launched_by_app
                    else "default"
                ),
            ]
        )
    if getattr(args, "tool_prompt_mode", None):
        cmd.extend(["--tool-prompt-mode", str(args.tool_prompt_mode)])
    if getattr(args, "chat_template_profile", None):
        cmd.extend(["--chat-template-profile", str(args.chat_template_profile)])
    if getattr(args, "chat_template_path", None):
        cmd.extend(["--chat-template-path", str(args.chat_template_path)])
    if bool(getattr(args, "open_browser", False)):
        cmd.append("--open-browser")
    if bool(getattr(args, "open_dashboard", False)):
        cmd.append("--open-dashboard")
    if bool(getattr(args, "enable_thermal_poll", False)):
        cmd.append("--enable-thermal-poll")
    app_launch_id = str(getattr(args, "app_launch_id", "") or "").strip()
    if app_launch_id:
        cmd.extend(["--app-launch-id", app_launch_id])
    if bool(getattr(args, "quickstart_pi", False)):
        from mtplx.pi import pi_launch_command

        cmd.extend(
            [
                "--launch-pi",
                "--server-console",
                "--pi-launch-command",
                pi_launch_command(
                    str(getattr(args, "model_id", None) or DEFAULT_PUBLIC_MODEL_ID)
                ),
            ]
        )
    if bool(getattr(args, "quickstart_opencode", False)):
        cmd.extend(["--launch-opencode", "--server-console"])
    if bool(getattr(args, "quickstart_swival", False)):
        cmd.append("--server-console")
    if bool(getattr(args, "quickstart_hermes", False)):
        cmd.extend(["--launch-hermes", "--server-console"])
        hermes_launch_command = str(
            getattr(args, "hermes_launch_command", "") or ""
        ).strip()
        if hermes_launch_command:
            cmd.extend(["--hermes-launch-command", hermes_launch_command])
    if bool(getattr(args, "stock_ar", False)):
        cmd.append("--stock-ar")
    elif getattr(args, "load_mtp", True) is False:
        cmd.append("--no-load-mtp")
    api_key_source = str(getattr(args, "api_key_source", "none") or "none")
    api_key_file = getattr(args, "api_key_file", None)
    if api_key and api_key_source == "flag":
        cmd.extend(["--api-key", str(api_key)])
    elif api_key and api_key_source == "file" and api_key_file:
        cmd.extend(["--api-key-file", str(api_key_file)])
    if getattr(args, "max_response_tokens", None) is not None:
        cmd.extend(["--max-response-tokens", str(args.max_response_tokens)])
    if getattr(args, "temperature", None) is not None:
        cmd.extend(["--temperature", str(args.temperature)])
    if getattr(args, "top_p", None) is not None:
        cmd.extend(["--top-p", str(args.top_p)])
    if getattr(args, "top_k", None) is not None:
        cmd.extend(["--top-k", str(args.top_k)])
    if getattr(args, "default_presence_penalty", None):
        cmd.extend(["--default-presence-penalty", str(args.default_presence_penalty)])
    if getattr(args, "default_frequency_penalty", None):
        cmd.extend(["--default-frequency-penalty", str(args.default_frequency_penalty)])
    if getattr(args, "reasoning", None) is not None:
        reasoning_mode = _reasoning_mode(args, default="auto")
        if reasoning_mode == "on":
            cmd.append("--enable-thinking")
        elif reasoning_mode == "off":
            cmd.append("--no-enable-thinking")
    if getattr(args, "reasoning_parser", None):
        cmd.extend(["--reasoning-parser", str(args.reasoning_parser)])
    if getattr(args, "reasoning_effort", None):
        cmd.extend(["--reasoning-effort", str(args.reasoning_effort)])
    if not getattr(args, "stats_footer", True):
        cmd.append("--no-stats-footer")
    if getattr(args, "strict_warmup", False):
        cmd.append("--strict-warmup")
    child_env_base = os.environ.copy()
    child_env_base.update(
        paged_kv_quantization_env(
            getattr(args, "paged_kv_quantization", "off") or "off"
        )
    )
    _apply_ram_session_cache_env(child_env_base, args)
    if app_launch_id:
        child_env_base["MTPLX_APP_LAUNCH_ID"] = app_launch_id
    if bool(getattr(args, "quickstart_pi", False)):
        _apply_pi_history_budget_env_defaults(child_env_base)
    if bool(getattr(args, "quickstart_opencode", False)):
        _apply_opencode_memory_env_defaults(child_env_base)
    if bool(getattr(args, "quickstart_hermes", False)):
        _apply_hermes_memory_env_defaults(child_env_base)
    if fan_mode in {FAN_MODE_MAX, FAN_MODE_SMART}:
        child_env_base["MTPLX_FAN_MODE"] = fan_mode
    if fan_mode == FAN_MODE_MAX:
        child_env_base["MTPLX_MAX_REQUESTED"] = "1"
    if dry_run:
        payload = _serve_dry_run_payload(
            args,
            runtime_model=runtime_model,
            profile_name=profile.name,
            model_id=str(model_id),
            generation_mode=generation_mode,
            cmd=cmd,
            env=child_env_base,
        )
        if bool(getattr(args, "json", False)):
            _print(payload)
        else:
            _print_serve_dry_run_human(payload)
        return 0
    if fan_mode == FAN_MODE_MAX:
        from mtplx.thermal import MaxSession

        def _emit(line: str) -> None:
            _safe_serve_watchdog_log(line)

        max_session = MaxSession(log=_emit)
        if not max_session.start():
            verified = max_session.thermal.get("verified") or {}
            _emit("")
            _emit("[max] !!! FAN CONTROL DID NOT TAKE EFFECT !!!")
            _emit(f"[max]   reason: {verified.get('message')}")
            actionable = verified.get("actionable")
            if actionable:
                _emit(f"[max]   action: {actionable}")
            if bool(getattr(args, "require_max_fans", False)):
                _emit("[max] strict startup requested; refusing to load the model.")
                _emit("")
                return 2
            _emit("[max] continuing the server WITHOUT fan boost.")
            _emit("")
            args.max = False  # don't lie to the watchdog about fan state
            args.fan_mode = "default"
            child_env_base.pop("MTPLX_FAN_MODE", None)
            child_env_base.pop("MTPLX_MAX_REQUESTED", None)
            if "--fan-mode" in cmd:
                try:
                    cmd[cmd.index("--fan-mode") + 1] = "default"
                except Exception:
                    pass
        # Only spin up the idle watchdog when verification confirmed fans
        # are actually pinned. If args.max was just disabled above, fall
        # through to the no-fan-control path.
        if getattr(args, "max", False):
            child_env = child_env_base.copy()
            child_env["MTPLX_FAN_MODE"] = FAN_MODE_MAX
            child_env["MTPLX_MAX_ACTUAL_RAMP_VERIFIED"] = "1"
            child_env["MTPLX_MAX_VERIFIED_AT"] = str(time.time())
            try:
                child_env["MTPLX_MAX_VERIFIED_JSON"] = json.dumps(
                    max_session.thermal.get("verified") or {},
                    sort_keys=True,
                    default=str,
                )
            except Exception:
                pass
            idle_minutes = int(getattr(args, "max_idle_min", 15))
            watchdog = _MaxIdleWatchdog(
                host=str(getattr(args, "host", "127.0.0.1")),
                port=int(getattr(args, "port", 8000)),
                idle_seconds=max(60, idle_minutes * 60),
                api_key=getattr(args, "api_key", None),
            )
            watchdog.start()
            app_parent_pid = _app_parent_pid_from_env(child_env)
            try:
                return _run_server_child_with_app_parent_watchdog(
                    cmd,
                    env=child_env,
                    cwd=repo_root(),
                    app_parent_pid=app_parent_pid,
                )
            finally:
                watchdog.stop()
                max_session.stop()  # belt-and-suspenders alongside atexit
    app_parent_pid = _app_parent_pid_from_env(child_env_base)
    if app_parent_pid is not None:
        return _run_server_child_with_app_parent_watchdog(
            cmd,
            env=child_env_base,
            cwd=repo_root(),
            app_parent_pid=app_parent_pid,
        )
    os.execvpe(sys.executable, cmd, child_env_base)
    return 0


def _app_parent_pid_from_env(env: dict[str, str]) -> int | None:
    raw = str(env.get("MTPLX_APP_PARENT_PID") or "").strip()
    if not raw:
        return None
    try:
        pid = int(raw)
    except ValueError:
        return None
    if pid <= 1 or pid == os.getpid():
        return None
    return pid


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        proc = subprocess.run(
            ["/bin/ps", "-p", str(pid), "-o", "stat="],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=0.5,
            check=False,
        )
    except Exception:
        return True
    if proc.returncode != 0:
        return False
    state = proc.stdout.strip()
    return "Z" not in state


def _terminate_server_child(proc: subprocess.Popen[Any], *, grace_s: float) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=max(0.1, float(grace_s)))
        return
    except subprocess.TimeoutExpired:
        pass
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=0.5)
        return
    except (subprocess.TimeoutExpired, ProcessLookupError):
        pass
    if proc.poll() is not None:
        return
    try:
        proc.kill()
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass


def _safe_serve_watchdog_log(message: str) -> None:
    try:
        print(message, file=sys.stderr, flush=True)
    except Exception:
        pass


def _run_server_child_with_app_parent_watchdog(
    cmd: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    app_parent_pid: int | None,
    poll_seconds: float = 1.0,
    shutdown_grace_s: float = 5.0,
) -> int:
    """Run the real server under the serve wrapper.

    Native MTPLXApp launches use the product CLI wrapper, not
    ``mtplx.server.openai`` directly, because the wrapper owns model
    resolution and max-fan lifecycle hooks. When the GUI disappears by
    normal quit, force quit, or crash, this watchdog makes the wrapper stop the
    child daemon so the wrapper's existing ``MaxSession`` cleanup can restore
    fans.
    """

    if app_parent_pid is not None and not _pid_is_alive(app_parent_pid):
        _safe_serve_watchdog_log(
            "[mtplx] app parent is gone; refusing to leave an app-owned daemon running.",
        )
        return 130

    proc = subprocess.Popen(cmd, env=env, cwd=cwd)
    stop = threading.Event()
    triggered = [False]
    received_signal: list[int | None] = [None]
    thread: threading.Thread | None = None
    previous_handlers: dict[int, Any] = {}

    def handle_shutdown_signal(signum: int, _frame: Any) -> None:
        if received_signal[0] is None:
            received_signal[0] = signum
            _safe_serve_watchdog_log(
                f"[mtplx] serve wrapper received signal {signum}; stopping child daemon.",
            )
            _terminate_server_child(proc, grace_s=shutdown_grace_s)

    if app_parent_pid is not None:

        def watch_parent() -> None:
            while not stop.wait(max(0.05, float(poll_seconds))):
                if proc.poll() is not None:
                    return
                if _pid_is_alive(app_parent_pid):
                    continue
                triggered[0] = True
                _safe_serve_watchdog_log(
                    "[mtplx] app parent exited; stopping app-owned daemon."
                )
                _terminate_server_child(proc, grace_s=shutdown_grace_s)
                return

        thread = threading.Thread(
            target=watch_parent,
            name="mtplx-app-parent-watchdog",
            daemon=True,
        )
        thread.start()

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handle_shutdown_signal)
        returncode = int(proc.wait())
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        stop.set()
        if thread is not None:
            thread.join(timeout=1.0)

    if received_signal[0] is not None:
        return 128 + int(received_signal[0])
    if triggered[0] and returncode < 0:
        return 130
    return returncode


class _MaxIdleWatchdog:
    """Background thread that drops fans to auto after ``idle_seconds`` of no
    chat activity, then ramps back to max on the next request.

    Polls the server's ``/health`` endpoint every 30s to read
    ``last_request_at`` / ``idle_seconds`` published by the OpenAI server.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        idle_seconds: int,
        poll_seconds: int = 30,
        api_key: str | None = None,
    ) -> None:
        self.url = f"http://{host}:{port}/health"
        self.idle_seconds = int(idle_seconds)
        self.poll_seconds = max(1, int(poll_seconds))
        self.api_key = api_key
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._is_max = True  # the parent set fans to max before spawning us

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="mtplx-max-watchdog", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        from mtplx.thermal import set_thermal_profile

        last_completed: int | None = None
        last_started_at: float | None = None
        last_active = time.time()
        while not self._stop.wait(self.poll_seconds):
            if self.api_key:
                payload = _http_json(self.url, timeout=2.0, api_key=self.api_key)
            else:
                payload = _http_json(self.url, timeout=2.0)
            if not payload.get("ok"):
                continue
            current = int(payload.get("requests_completed") or 0)
            started_at = float(payload.get("last_request_started_at") or 0.0)
            active_requests = int(
                payload.get("foreground_active")
                if payload.get("foreground_active") is not None
                else payload.get("active_requests") or 0
            )
            if last_completed is None:
                last_completed = current
            if last_started_at is None:
                last_started_at = started_at
            now = time.time()
            if active_requests > 0:
                last_active = now
                last_started_at = started_at
                if not self._is_max:
                    set_thermal_profile("performance")
                    self._is_max = True
                continue
            if started_at and started_at != last_started_at:
                last_started_at = started_at
                last_active = now
                if not self._is_max:
                    set_thermal_profile("performance")
                    self._is_max = True
                continue
            if current != last_completed:
                last_completed = current
                last_active = now
                if not self._is_max:
                    set_thermal_profile("performance")
                    self._is_max = True
                continue
            if (now - last_active) >= self.idle_seconds and self._is_max:
                set_thermal_profile("silent")
                self._is_max = False


def _generate_one_shot_public(
    args: Any, *, command: str
) -> tuple[int, dict[str, Any], list[Any]]:
    fan_mode = _fan_mode_from_args(args)
    prompt = getattr(args, "prompt", None) or getattr(args, "prompt_arg", None)
    if not prompt:
        raise SystemExit(f"mtplx {command} requires a prompt")
    depth_error = _validate_public_depth(args, printer=lambda _line: None)
    if depth_error is not None:
        depth_ceiling = _public_depth_ceiling(args)
        return (
            depth_error,
            {
                "error": "invalid depth",
                "detail": (
                    "--depth must be between "
                    f"1 and {depth_ceiling} for the current MTPLX runtime"
                ),
            },
            [],
        )
    runtime_model, resolve_error = _resolve_runtime_model_path(
        args.model,
        cache_dir=getattr(args, "cache_dir", None),
    )
    if resolve_error is not None:
        return EXIT_TELEMETRY, resolve_error, []
    inspection, gate_exit = _model_gate(
        runtime_model,
        unsafe_force_unverified=bool(getattr(args, "unsafe_force_unverified", False)),
        yes=bool(getattr(args, "yes", False)),
    )
    if gate_exit is not None:
        return (
            gate_exit,
            {"error": "model failed MTP primary gate", "model": inspection},
            [],
        )
    mode_exit = _apply_runtime_compatibility_mode(
        args,
        inspection,
        printer=lambda _line: None,
    )
    if mode_exit is not None:
        return (
            mode_exit,
            {
                "error": "model requires target-only AR",
                "detail": "rerun with --no-mtp",
                "model": inspection,
            },
            [],
        )
    _apply_backend_serve_defaults(args, inspection)
    # Per-model default, same rule as serve: turbo for the quantized
    # flagships unless --profile was given. The raw sustained fallback here
    # made `mtplx run` silently benchmark the slow profile (2026-08-16
    # redp314 board investigation).
    profile = get_profile(_resolved_default_profile_name(args))
    apply_profile_env(profile.name)
    generation_mode = _generation_mode_from_args(args)
    draft_lm_head = (
        _model_draft_lm_head_spec(inspection, profile)
        if generation_mode == GENERATION_MODE_MTP
        else None
    )
    draft_sampler = (
        _model_draft_sampler_spec(inspection, profile)
        if generation_mode == GENERATION_MODE_MTP
        else None
    )

    max_session: Any | None = None
    thermal: dict[str, Any] | None = None
    if fan_mode == FAN_MODE_MAX:
        from mtplx.thermal import MaxSession

        def _emit(line: str) -> None:
            print(line, file=sys.stderr, flush=True)

        max_session = MaxSession(log=_emit)
        if max_session.start():
            thermal = max_session.thermal
        else:
            verified = max_session.thermal.get("verified") or {}
            _emit("")
            _emit("[max] fan boost unavailable; continuing this run without fan boost.")
            if verified.get("message"):
                _emit(f"[max] reason: {verified.get('message')}")
            _emit("")
            thermal = max_session.thermal
            max_session = None

    from mtplx.benchmarks.schema import PromptCase, encode_prompt_case
    from mtplx.generation import generate_ar, generate_mtpk
    from mtplx.runtime import load
    from mtplx.sampling import SamplerConfig

    # Serve-path memory discipline (#261, F7): pin the exact Metal allocator
    # caps the serve path applies at startup before this in-process load.
    from mtplx.server.openai import apply_memory_caps_preflight

    memory_preflight = apply_memory_caps_preflight(
        entry=f"cli.{command}",
        model=str(runtime_model),
    )

    try:
        rt = load(runtime_model, mtp=getattr(args, "load_mtp", True) is not False)
        draft_report = None
        if (
            draft_lm_head is not None
            and hasattr(rt, "model")
            and bool(getattr(rt, "mtp_enabled", True))
        ):
            from mtplx.draft_lm_head import _install_draft_lm_head

            draft_report = _install_draft_lm_head(
                rt,
                bits=int(draft_lm_head["bits"]),
                group_size=int(draft_lm_head["group_size"]),
                mode=str(draft_lm_head["mode"]),
            )
        messages = None
        system = getattr(args, "system", None)
        if system:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
        requested_max_tokens = getattr(args, "max_tokens", None)
        case = PromptCase(
            id=f"cli_{command}",
            category=command,
            prompt=prompt,
            max_tokens=int(requested_max_tokens or 0),
            messages=messages,
        )
        reasoning_mode = _reasoning_mode(args)
        enable_thinking = _enable_thinking_for_reasoning(reasoning_mode)
        prompt_ids = encode_prompt_case(
            rt.tokenizer,
            case,
            chat_template=True,
            enable_thinking=enable_thinking,
        )
        budget = _cli_generation_budget(
            tokenizer=rt.tokenizer,
            model_path=runtime_model,
            prompt_token_count=len(prompt_ids),
            explicit_max_tokens=requested_max_tokens,
        )
        max_tokens_value = int(budget["effective_max_tokens"])
        sampler = SamplerConfig(
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k
        )
        smart_fans = None
        smart_request_id = None
        if fan_mode == FAN_MODE_SMART:
            from mtplx.thermal import SmartFanController

            def _emit_smart(line: str) -> None:
                print(line, file=sys.stderr, flush=True)

            smart_fans = SmartFanController(log=_emit_smart)
            smart_request_id = f"cli-{command}-{int(time.time() * 1000)}"
            smart_fans.begin_request(smart_request_id)
        try:
            if generation_mode == GENERATION_MODE_AR:
                out = generate_ar(
                    rt,
                    prompt_ids,
                    max_tokens=max_tokens_value,
                    sampler=sampler,
                    seed=args.seed,
                )
            else:
                out = generate_mtpk(
                    rt,
                    prompt_ids,
                    max_tokens=max_tokens_value,
                    sampler=sampler,
                    draft_sampler=_draft_sampler_from_spec(draft_sampler),
                    speculative_depth=args.depth,
                    seed=args.seed,
                    mtp_hidden_variant="post_norm",
                    mtp_cache_policy="persistent",
                    mtp_history_policy="committed",
                    verify_strategy="capture_commit",
                    verify_core="linear-gdn-from-conv-tape",
                )
        finally:
            if smart_fans is not None and smart_request_id is not None:
                smart_fans.end_request(smart_request_id, wait_for_restore=True)
    finally:
        if max_session is not None:
            max_session.stop()
            thermal = max_session.thermal
    validations = [
        validate_no_degenerate_loop(out.text),
        validate_balanced_delimiters(out.text),
    ]
    if args.expect_python:
        validations.append(validate_python_syntax(out.text))
    payload = {
        "text": out.text,
        "model": _compact_model_summary(inspection),
        "profile": profile.to_dict(),
        "memory_preflight": memory_preflight,
        "draft_lm_head": draft_report,
        "draft_sampler": draft_sampler,
        "stats": {
            "generated_tokens": out.stats.generated_tokens,
            "max_tokens": max_tokens_value,
            "request_max_tokens": budget["request_max_tokens"],
            "context_window": budget["context_window"],
            "remaining_context_tokens": budget["remaining_context_tokens"],
            "context_cap_applied": budget["context_cap_applied"],
            "reasoning": reasoning_mode,
            "generation_mode": generation_mode,
            "mtp_depth": (
                0
                if generation_mode == GENERATION_MODE_AR
                else int(
                    getattr(out.stats, "speculative_depth", args.depth) or args.depth
                )
            ),
            "requested_mtp_depth": 0
            if generation_mode == GENERATION_MODE_AR
            else int(
                getattr(out.stats, "requested_speculative_depth", args.depth)
                or args.depth
            ),
            "long_context_mtp_depth_policy": (
                {}
                if generation_mode == GENERATION_MODE_AR
                else dict(getattr(out.stats, "long_context_mtp_depth_policy", {}) or {})
            ),
            "tok_s": getattr(out.stats, "decode_tok_s", out.stats.tok_s),
            "decode_tok_s": getattr(out.stats, "decode_tok_s", out.stats.tok_s),
            "decode_elapsed_s": getattr(out.stats, "decode_elapsed_s", None),
            "end_to_end_tok_s": getattr(out.stats, "end_to_end_tok_s", out.stats.tok_s),
            "elapsed_s": getattr(out.stats, "elapsed_s", None),
            "prompt_eval_time_s": getattr(out.stats, "prompt_eval_time_s", None),
            "verify_ms_per_call": (
                None
                if generation_mode == GENERATION_MODE_AR
                else 1000.0 * out.stats.verify_time_s / out.stats.verify_calls
                if out.stats.verify_calls
                else None
            ),
            "verify_calls": (
                0
                if generation_mode == GENERATION_MODE_AR
                else int(getattr(out.stats, "verify_calls", 0) or 0)
            ),
            "verify_time_s": (
                0.0
                if generation_mode == GENERATION_MODE_AR
                else float(getattr(out.stats, "verify_time_s", 0.0) or 0.0)
            ),
            "draft_time_s": (
                0.0
                if generation_mode == GENERATION_MODE_AR
                else float(getattr(out.stats, "draft_time_s", 0.0) or 0.0)
            ),
            "accepted_by_depth": (
                []
                if generation_mode == GENERATION_MODE_AR
                else list(getattr(out.stats, "accepted_by_depth", []) or [])
            ),
            "drafted_by_depth": (
                []
                if generation_mode == GENERATION_MODE_AR
                else list(getattr(out.stats, "drafted_by_depth", []) or [])
            ),
        },
        "validations": [v.__dict__ for v in validations],
    }
    if thermal is not None:
        payload["thermal"] = thermal
    return (
        0 if all(v.passed for v in validations) else EXIT_QUALITY,
        payload,
        validations,
    )


def cmd_run_public(args: Any) -> int:
    code, payload, _validations = _generate_one_shot_public(args, command="run")
    if "error" in payload:
        _print_command_error(
            payload,
            command="run",
            json_output=bool(getattr(args, "json", False)),
        )
        return code
    if getattr(args, "json", False):
        _print(payload)
    else:
        text = payload["text"]
        print(text, end="" if text.endswith("\n") else "\n")
        if not getattr(args, "quiet", False):
            stats = payload["stats"]
            tok_s = stats.get("tok_s")
            tok_s_text = f"{tok_s:.2f}" if isinstance(tok_s, (int, float)) else "n/a"
            mode = str(stats.get("generation_mode") or "mtp").upper()
            mtp_depth = int(stats.get("mtp_depth") or 0)
            print(
                f"\n[mtplx] profile={payload['profile']['name']} "
                f"mode={mode} mtp_depth={mtp_depth} "
                f"tokens={stats.get('generated_tokens')} tok_s={tok_s_text}"
            )
    return code


def cmd_chat_public(args: Any) -> int:
    # Still a one-shot smoke path until the interactive REPL lands in Phase 5.
    code, payload, _validations = _generate_one_shot_public(args, command="chat")
    if "error" in payload:
        _print_command_error(
            payload,
            command="chat",
            json_output=bool(getattr(args, "json", False)),
        )
        return code
    if getattr(args, "json", False):
        _print(payload)
    else:
        text = payload["text"]
        print(text, end="" if text.endswith("\n") else "\n")
    return code


def _quickstart_line(text: str = "") -> None:
    print(text, flush=True)


def _start_command_name(args: Any) -> str:
    command = str(getattr(args, "command", None) or "start")
    return "start" if command in {"start", "quickstart", "quick-start"} else command


def _start_invocation(args: Any, suffix: str = "") -> str:
    command = _start_command_name(args)
    return f"mtplx {command}{suffix}"


def _handle_quickstart_reasoning_command(args: Any, prompt: str) -> bool:
    parts = prompt.strip().split()
    if not parts or parts[0].lower() not in {"/reasoning", "--reasoning"}:
        return False
    if len(parts) == 1:
        _quickstart_line(f"Reasoning: {_reasoning_mode(args)}")
        _quickstart_line("try: /reasoning on")
        _quickstart_line("try: /reasoning off")
        _quickstart_line("try: /reasoning auto")
        return True
    if len(parts) == 2 and parts[1].lower() in {"auto", "on", "off"}:
        setattr(args, "reasoning", parts[1].lower())
        _quickstart_line(f"Reasoning: {_reasoning_mode(args)}")
        return True
    _quickstart_line("usage: /reasoning on|off|auto")
    return True


def _handle_quickstart_mtp_command(
    args: Any, prompt: str, *, runtime: Any | None = None
) -> bool:
    parts = prompt.strip().split()
    if not parts or parts[0].lower() not in {"/mtp", "--mtp"}:
        return False
    if len(parts) == 1 or parts[1].lower() == "status":
        mode = _generation_mode_from_args(args)
        _quickstart_line(
            f"MTP: {'on' if mode == GENERATION_MODE_MTP else 'off'} "
            f"({_generation_mode_label(mode)})"
        )
        return True
    if len(parts) == 2 and parts[1].lower() in {"on", "off"}:
        requested = (
            GENERATION_MODE_MTP if parts[1].lower() == "on" else GENERATION_MODE_AR
        )
        if (
            requested == GENERATION_MODE_MTP
            and runtime is not None
            and not bool(getattr(runtime, "mtp_enabled", False))
        ):
            _quickstart_line("MTP: unavailable for this loaded runtime")
            return True
        _set_generation_mode_on_args(args, requested)
        _quickstart_line(
            "MTP: on for the next turn"
            if requested == GENERATION_MODE_MTP
            else "MTP: off for the next turn (target-only AR generation)"
        )
        return True
    _quickstart_line("usage: /mtp on|off|status")
    return True


def _quickstart_console() -> Any:
    """Return a ``rich.console.Console`` if available and stdout is a TTY."""

    if not sys.stdout.isatty():
        return None
    try:
        from rich.console import Console
    except ImportError:
        return None
    try:
        return Console()
    except Exception:
        return None


def _chat_input_prompt() -> str:
    """Prompt string for the chat REPL ``input()`` call.

    Uses ANSI color when the terminal supports it; falls back to plain text.
    """

    if not sys.stdout.isatty():
        return "\nyou> "
    # ANSI bold cyan for "you", reset, then bold "> ", then reset.
    # Avoid using rich here; ``input()`` does not interact well with rich Live.
    return "\n\033[1;36myou\033[0m\033[1m>\033[0m "


def _print_assistant_fallback(label: str, text: str) -> None:
    """Print a non-streamed assistant message with the same styled label."""

    console = _quickstart_console()
    if console is None:
        _quickstart_line(f"{label}:")
        print(text, end="" if text.endswith("\n") else "\n")
        return
    console.print(f"[bold cyan]{label}[/bold cyan]", highlight=False)
    console.print(text, soft_wrap=True, highlight=False, markup=False)


def _print_stats_line(text: str) -> None:
    """Print the stats footer with a dim/colored treatment when possible."""

    console = _quickstart_console()
    if console is None:
        _quickstart_line(text)
        return
    console.print(f"  [dim]{text}[/dim]", highlight=False)


class _QuickstartHeartbeat:
    def __init__(self, label: str, *, interval_s: float = 5.0) -> None:
        script = (
            "import signal,sys,time\n"
            "label=sys.argv[1]\n"
            "interval=float(sys.argv[2])\n"
            "running=True\n"
            "def stop(_signum,_frame):\n"
            "    global running\n"
            "    running=False\n"
            "signal.signal(signal.SIGTERM, stop)\n"
            "elapsed=0.0\n"
            "while running:\n"
            "    time.sleep(interval)\n"
            "    if not running:\n"
            "        break\n"
            "    elapsed += interval\n"
            "    print(f'      {label}... {elapsed:.0f}s elapsed', flush=True)\n"
        )
        self.proc = subprocess.Popen(
            [sys.executable, "-c", script, label, str(float(interval_s))],
            stdout=None,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )

    def set(self) -> None:
        proc = self.proc
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=1.0)


def _quickstart_heartbeat(
    label: str, *, interval_s: float = 5.0
) -> _QuickstartHeartbeat:
    return _QuickstartHeartbeat(label, interval_s=interval_s)


def _quickstart_current_model(args: Any) -> str:
    model = getattr(args, "model", None)
    explicit_model = bool(getattr(args, "_model_explicit", False))
    if not explicit_model and is_verified_default_model_ref(model):
        selection = select_default_model()
        args._mtplx_default_model_selection = selection.to_dict()
        return selection.model
    return str(model or DEFAULT_MODEL_ID)


def _quickstart_download_ref(model: str) -> str:
    from mtplx.hf_loader import repo_id_from_model_ref

    if repo_id_from_model_ref(model):
        return model
    selection = select_default_model()
    default_local_refs = {
        DEFAULT_MODEL_ID,
        selection.model,
        str(DEFAULT_RUNTIME_MODEL_DIR),
        DEFAULT_CHAMPION,
        str((repo_root() / DEFAULT_MODEL_ID).resolve()),
        str((repo_root() / selection.model).resolve()),
        str((repo_root() / str(DEFAULT_RUNTIME_MODEL_DIR)).resolve()),
    }
    if model in default_local_refs:
        return selection.hf_model
    raise ValueError(
        "cannot download a local model path. Re-run with --model HF_ORG/HF_REPO "
        "or choose a folder that already exists."
    )


def _quickstart_choose_model(
    args: Any, *, target: str = "terminal"
) -> tuple[str, bool]:
    model = _quickstart_current_model(args)
    download = bool(getattr(args, "download", False))
    if (
        target == "openwebui"
        or getattr(args, "yes", False)
        or getattr(args, "prompt", None)
        or getattr(args, "dry_run", False)
        or getattr(args, "_onboarded", False)
        or not sys.stdin.isatty()
    ):
        return model, download

    _quickstart_line(f"MTPLX {_start_command_name(args)}")
    selection = select_default_model()
    _quickstart_line("Choose a model:")
    _quickstart_line(f"  1. Use verified default for this Mac ({selection.label})")
    quality_ref = optimized_quality_model_ref()
    _quickstart_line(f"  2. Optimized Quality ({OPTIMIZED_QUALITY_DESCRIPTION})")
    _quickstart_line("  3. Choose a local model folder")
    _quickstart_line(
        f"  4. Download verified default from Hugging Face ({selection.hf_model})"
    )
    choice = input("Select [1]: ").strip()
    if choice == "2":
        return quality_ref, download
    if choice == "3":
        chosen = input("Model folder: ").strip()
        return (chosen or model), False
    if choice == "4":
        return selection.hf_model, True
    return model, download


def _quickstart_resolve_model(
    model: str, *, cache_dir: str | None, download: bool
) -> tuple[str | None, dict[str, Any]]:
    runtime_model, resolve_error = _resolve_runtime_model_path(
        model, cache_dir=cache_dir
    )
    if resolve_error is None:
        return runtime_model, {
            "model": model,
            "runtime_model": runtime_model,
            "downloaded": False,
            "download_ref": None,
        }
    if not download:
        return None, {
            "model": model,
            "runtime_model": None,
            "downloaded": False,
            "download_ref": None,
            "error": resolve_error,
        }

    download_ref = _quickstart_download_ref(model)
    try:
        inspection = inspect_model(download_ref).to_dict()
    except Exception as exc:
        return None, {
            "model": model,
            "runtime_model": None,
            "downloaded": False,
            "download_ref": download_ref,
            "error": {
                "error": "model failed Hugging Face preflight",
                "model": download_ref,
                "detail": str(exc),
            },
        }
    compatibility = inspection.get("compatibility") or {}
    if not bool(compatibility.get("can_run")):
        return None, {
            "model": model,
            "runtime_model": None,
            "downloaded": False,
            "download_ref": download_ref,
            "gate_inspection": inspection,
            "error": {
                "error": "model failed MTPLX compatibility gate",
                "model": inspection,
            },
        }

    from mtplx.hf_loader import pull_model

    _quickstart_line(f"[1/4] Downloading model: {download_ref}")
    callback, finalize = _rich_download_progress_callback(repo_id=download_ref)
    try:
        try:
            result = pull_model(
                download_ref,
                cache_dir=cache_dir,
                progress_callback=callback,
                progress_interval_s=0.4,
            )
        except KeyboardInterrupt:
            return None, {
                "model": model,
                "runtime_model": None,
                "downloaded": False,
                "download_ref": download_ref,
                "cancelled": True,
                "error": {
                    "error": "download cancelled",
                    "model": download_ref,
                },
            }
    finally:
        finalize()
    runtime_model, resolve_error = _resolve_runtime_model_path(
        download_ref, cache_dir=cache_dir
    )
    if resolve_error is not None:
        return None, {
            "model": model,
            "runtime_model": None,
            "downloaded": True,
            "download_ref": download_ref,
            "download_result": result,
            "error": resolve_error,
        }
    return runtime_model, {
        "model": model,
        "runtime_model": runtime_model,
        "downloaded": True,
        "download_ref": download_ref,
        "download_result": result,
    }


def _quickstart_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _quickstart_decode_timing(
    stats: dict[str, Any],
) -> tuple[float | None, float | None]:
    generated_tokens = int(stats.get("generated_tokens") or 0)
    elapsed_s = _quickstart_number(stats.get("elapsed_s")) or 0.0
    prompt_eval_time_s = _quickstart_number(stats.get("prompt_eval_time_s"))
    if prompt_eval_time_s is None:
        target_forward_time_s = (
            _quickstart_number(stats.get("target_forward_time_s")) or 0.0
        )
        verify_time_s = _quickstart_number(stats.get("verify_time_s")) or 0.0
        repair_time_s = _quickstart_number(stats.get("repair_time_s")) or 0.0
        prompt_eval_time_s = max(
            0.0, target_forward_time_s - verify_time_s - repair_time_s
        )
    decode_elapsed_s = max(0.0, elapsed_s - prompt_eval_time_s)
    if generated_tokens <= 0 or decode_elapsed_s <= 0.0:
        return None, decode_elapsed_s
    return generated_tokens / decode_elapsed_s, decode_elapsed_s


def _quickstart_token_window_rate(token_times: list[float]) -> float | None:
    if len(token_times) < 2:
        return None
    elapsed_s = float(token_times[-1]) - float(token_times[0])
    if elapsed_s <= 0.0:
        return None
    return (len(token_times) - 1) / elapsed_s


class _QuickstartIncrementalTokenDecoder:
    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._token_cache: list[int] = []
        self._print_len = 0

    def _decode(self, tokens: list[int]) -> str:
        try:
            return self._tokenizer.decode(tokens, clean_up_tokenization_spaces=False)
        except TypeError:
            return self._tokenizer.decode(tokens)

    def feed(self, tokens: list[int]) -> str:
        if not tokens:
            return ""
        self._token_cache.extend(int(token) for token in tokens)
        text = self._decode(self._token_cache)
        if text.endswith("\n"):
            printable = text[self._print_len :]
            self._token_cache = []
            self._print_len = 0
            return printable
        if text and self._is_cjk_char(ord(text[-1])):
            printable = text[self._print_len :]
            self._print_len += len(printable)
            return printable
        close_index = text.find("</think>", self._print_len)
        if close_index >= 0:
            boundary = close_index + len("</think>")
            printable = text[self._print_len : boundary]
            self._print_len = boundary
            return printable
        boundary = -1
        for index in range(len(text) - 1, -1, -1):
            if text[index].isspace():
                boundary = index + 1
                break
        if boundary <= self._print_len:
            return ""
        printable = text[self._print_len : boundary]
        self._print_len = boundary
        return printable

    def finish(self) -> str:
        if not self._token_cache:
            return ""
        text = self._decode(self._token_cache)
        printable = text[self._print_len :]
        self._token_cache = []
        self._print_len = 0
        return printable

    @staticmethod
    def _is_cjk_char(cp: int) -> bool:
        return (
            (0x4E00 <= cp <= 0x9FFF)
            or (0x3400 <= cp <= 0x4DBF)
            or (0x20000 <= cp <= 0x2A6DF)
            or (0x2A700 <= cp <= 0x2B73F)
            or (0x2B740 <= cp <= 0x2B81F)
            or (0x2B820 <= cp <= 0x2CEAF)
            or (0xF900 <= cp <= 0xFAFF)
            or (0x2F800 <= cp <= 0x2FA1F)
        )


class _QuickstartTerminalStreamer:
    def __init__(self, tokenizer: Any, *, label: str) -> None:
        self._decoder = _QuickstartIncrementalTokenDecoder(tokenizer)
        self._label = label
        self.started = False
        self._console: Any = None
        try:
            from rich.console import Console
        except ImportError:
            self._console = None
        else:
            try:
                self._console = Console()
            except Exception:
                self._console = None

    def feed(self, tokens: list[int]) -> None:
        text = self._decoder.feed(tokens)
        if text:
            self._write(text)

    def finish(self) -> None:
        text = self._decoder.finish()
        if text:
            self._write(text)
        if self.started:
            print(flush=True)

    def _write(self, text: str) -> None:
        if not self.started:
            self._print_label()
            self.started = True
        if self._console is not None:
            self._console.print(
                text, end="", soft_wrap=True, highlight=False, markup=False
            )
        else:
            print(text, end="", flush=True)

    def _print_label(self) -> None:
        if self._console is not None:
            self._console.print(
                f"[bold cyan]{self._label}[/bold cyan]", highlight=False
            )
        else:
            print(f"{self._label}:")


def _quickstart_stats_line(payload: dict[str, Any]) -> str:
    stats = payload.get("stats") or {}
    generated_tokens = int(stats.get("generated_tokens") or 0)
    stream_tok_s = _quickstart_number(stats.get("stream_tok_s"))
    total_tok_s = _quickstart_number(stats.get("end_to_end_tok_s", stats.get("tok_s")))
    decode_tok_s = _quickstart_number(stats.get("decode_tok_s"))
    decode_elapsed_s = _quickstart_number(stats.get("decode_elapsed_s"))
    if decode_tok_s is None:
        decode_tok_s, decode_elapsed_s = _quickstart_decode_timing(stats)
    verify_ms = stats.get("verify_ms_per_call")
    verify_text = (
        f"{verify_ms:.1f} ms/verify"
        if isinstance(verify_ms, (int, float))
        else "verify n/a"
    )
    if decode_tok_s is not None:
        speed_text = f"{decode_tok_s:.2f} tok/s"
        if stream_tok_s is not None and abs(decode_tok_s - stream_tok_s) >= 0.1:
            speed_text = f"{speed_text} | live_window={stream_tok_s:.2f}"
        if total_tok_s is not None and abs(decode_tok_s - total_tok_s) >= 0.1:
            speed_text = f"{speed_text} | total={total_tok_s:.2f}"
    elif stream_tok_s is not None:
        speed_text = f"{stream_tok_s:.2f} tok/s"
        if total_tok_s is not None and abs(stream_tok_s - total_tok_s) >= 0.1:
            speed_text = f"{speed_text} | total={total_tok_s:.2f}"
    elif total_tok_s is not None:
        speed_text = f"{total_tok_s:.2f} total tok/s"
    else:
        speed_text = "TPS n/a"
    elapsed_text = (
        f"{generated_tokens} tokens in {decode_elapsed_s:.2f}s decode"
        if decode_elapsed_s is not None and decode_elapsed_s > 0.0
        else f"{generated_tokens} tokens"
    )
    detail_parts = []
    generation_mode = str(stats.get("generation_mode") or "").lower()
    if generation_mode in GENERATION_MODES:
        mode_text = "AR" if generation_mode == GENERATION_MODE_AR else "MTP"
        mtp_depth = int(stats.get("mtp_depth") or 0)
        detail_parts.append(f"mode={mode_text}")
        detail_parts.append(f"mtp_depth={mtp_depth}")
    detail_parts.append(verify_text)
    verify_calls = stats.get("verify_calls")
    if isinstance(verify_calls, int) and verify_calls:
        detail_parts.append(f"{verify_calls} verify calls")
    accepted = stats.get("accepted_by_depth")
    if isinstance(accepted, list) and accepted:
        detail_parts.append(f"accept={accepted}")
    corrections = stats.get("correction_tokens")
    if isinstance(corrections, int) and corrections:
        detail_parts.append(f"corr={corrections}")
    ttft_s = _quickstart_number(stats.get("ttft_s"))
    if ttft_s is not None:
        detail_parts.append(f"ttft={ttft_s:.2f}s")
    return (
        f"[mtplx] {elapsed_text} | "
        f"{speed_text} | {' | '.join(detail_parts)} | profile={payload['profile']['name']}"
    )


def _quickstart_generate(
    *,
    rt: Any,
    inspection: dict[str, Any],
    profile: Any,
    args: Any,
    prompt: str,
    history: list[dict[str, str]],
    turn_index: int,
    max_tokens: int | None = None,
    include_history: bool = True,
    stream_label: str | None = None,
    draft_sampler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from mtplx.benchmarks.schema import PromptCase, encode_prompt_case
    from mtplx.generation import generate_ar, generate_mtpk
    from mtplx.sampling import SamplerConfig

    messages: list[dict[str, str]] = []
    system = getattr(args, "system", None)
    if system:
        messages.append({"role": "system", "content": system})
    if include_history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    requested_max_tokens = (
        max_tokens if max_tokens is not None else getattr(args, "max_tokens", None)
    )
    case = PromptCase(
        id=f"quickstart_{turn_index}",
        category="quickstart",
        prompt=prompt,
        max_tokens=int(requested_max_tokens or 0),
        messages=messages,
    )
    reasoning_mode = _reasoning_mode(args)
    enable_thinking = _enable_thinking_for_reasoning(reasoning_mode)
    prompt_ids = encode_prompt_case(
        rt.tokenizer,
        case,
        chat_template=True,
        enable_thinking=enable_thinking,
    )
    budget = _cli_generation_budget(
        tokenizer=rt.tokenizer,
        model_path=getattr(rt, "model_path", getattr(args, "model", "")),
        prompt_token_count=len(prompt_ids),
        explicit_max_tokens=requested_max_tokens,
    )
    max_tokens_value = int(budget["effective_max_tokens"])
    request_started_s = time.perf_counter()
    token_times: list[float] = []
    terminal_streamer = (
        _QuickstartTerminalStreamer(rt.tokenizer, label=stream_label)
        if stream_label
        else None
    )

    def record_tokens(new_tokens: list[int]) -> None:
        now = time.perf_counter()
        token_times.extend([now for _token in new_tokens])
        if terminal_streamer is not None:
            terminal_streamer.feed(new_tokens)

    sampler = SamplerConfig(
        temperature=float(getattr(args, "temperature", 0.6)),
        top_p=float(getattr(args, "top_p", 0.95)),
        top_k=int(getattr(args, "top_k", 20)),
    )
    seed = int(getattr(args, "seed", 0)) + turn_index
    generation_mode = _generation_mode_from_args(args)
    fan_mode = _fan_mode_from_args(args)
    smart_fans = None
    smart_request_id = None
    try:
        if fan_mode == FAN_MODE_SMART:
            from mtplx.thermal import SmartFanController

            smart_fans = SmartFanController(log=_quickstart_line)
            smart_request_id = f"terminal-{turn_index}-{int(time.time() * 1000)}"
            smart_fans.begin_request(smart_request_id)
        if generation_mode == GENERATION_MODE_AR:
            out = generate_ar(
                rt,
                prompt_ids,
                max_tokens=max_tokens_value,
                sampler=sampler,
                seed=seed,
                token_callback=record_tokens,
            )
        else:
            out = generate_mtpk(
                rt,
                prompt_ids,
                max_tokens=max_tokens_value,
                sampler=sampler,
                draft_sampler=_draft_sampler_from_spec(draft_sampler),
                speculative_depth=int(getattr(args, "depth", 3)),
                seed=seed,
                mtp_hidden_variant="post_norm",
                mtp_cache_policy="persistent",
                mtp_history_policy="committed",
                verify_strategy="capture_commit",
                verify_core="linear-gdn-from-conv-tape",
                token_callback=record_tokens,
            )
    finally:
        if smart_fans is not None and smart_request_id is not None:
            smart_fans.end_request(smart_request_id, wait_for_restore=False)
        if terminal_streamer is not None:
            terminal_streamer.finish()
    stats = {
        "generated_tokens": out.stats.generated_tokens,
        "max_tokens": max_tokens_value,
        "request_max_tokens": budget["request_max_tokens"],
        "context_window": budget["context_window"],
        "remaining_context_tokens": budget["remaining_context_tokens"],
        "context_cap_applied": budget["context_cap_applied"],
        "reasoning": reasoning_mode,
        "generation_mode": generation_mode,
        "mtp_depth": (
            0
            if generation_mode == GENERATION_MODE_AR
            else int(
                getattr(out.stats, "speculative_depth", getattr(args, "depth", 3))
                or getattr(args, "depth", 3)
            )
        ),
        "requested_mtp_depth": (
            0
            if generation_mode == GENERATION_MODE_AR
            else int(
                getattr(
                    out.stats,
                    "requested_speculative_depth",
                    getattr(args, "depth", 3),
                )
                or getattr(args, "depth", 3)
            )
        ),
        "long_context_mtp_depth_policy": (
            {}
            if generation_mode == GENERATION_MODE_AR
            else dict(getattr(out.stats, "long_context_mtp_depth_policy", {}) or {})
        ),
        "tok_s": getattr(out.stats, "decode_tok_s", out.stats.tok_s),
        "decode_tok_s": getattr(out.stats, "decode_tok_s", out.stats.tok_s),
        "decode_elapsed_s": getattr(out.stats, "decode_elapsed_s", None),
        "end_to_end_tok_s": getattr(out.stats, "end_to_end_tok_s", out.stats.tok_s),
        "elapsed_s": out.stats.elapsed_s,
        "prompt_eval_time_s": out.stats.prompt_eval_time_s,
        "verify_time_s": 0.0
        if generation_mode == GENERATION_MODE_AR
        else out.stats.verify_time_s,
        "target_forward_time_s": out.stats.target_forward_time_s,
        "repair_time_s": out.stats.repair_time_s,
        "draft_time_s": 0.0
        if generation_mode == GENERATION_MODE_AR
        else out.stats.draft_time_s,
        "verify_calls": 0
        if generation_mode == GENERATION_MODE_AR
        else out.stats.verify_calls,
        "accepted_by_depth": (
            []
            if generation_mode == GENERATION_MODE_AR
            else list(getattr(out.stats, "accepted_by_depth", []) or [])
        ),
        "drafted_by_depth": (
            []
            if generation_mode == GENERATION_MODE_AR
            else list(getattr(out.stats, "drafted_by_depth", []) or [])
        ),
        "correction_tokens": out.stats.correction_tokens,
        "bonus_tokens": out.stats.bonus_tokens,
        "stream_tok_s": _quickstart_token_window_rate(token_times),
        "ttft_s": (token_times[0] - request_started_s) if token_times else None,
        "verify_ms_per_call": (
            None
            if generation_mode == GENERATION_MODE_AR
            else 1000.0 * out.stats.verify_time_s / out.stats.verify_calls
            if out.stats.verify_calls
            else None
        ),
    }
    decode_tok_s, decode_elapsed_s = _quickstart_decode_timing(stats)
    stats["decode_tok_s"] = decode_tok_s
    stats["decode_elapsed_s"] = decode_elapsed_s
    return {
        "text": out.text,
        "model": _compact_model_summary(inspection),
        "profile": profile.to_dict(),
        "draft_sampler": draft_sampler,
        "stats": stats,
        "streamed": bool(terminal_streamer and terminal_streamer.started),
        "validations": [
            validate_no_degenerate_loop(out.text).__dict__,
            validate_balanced_delimiters(out.text).__dict__,
        ],
    }


def _public_model_id_for_ref(model_ref: str, *, default_model_id: str) -> str:
    return public_model_id_for_ref(model_ref, default_model_id=default_model_id)


def _public_model_id_for_args(args: Any, model_ref: str | None) -> str:
    model_id = str(getattr(args, "model_id", None) or DEFAULT_PUBLIC_MODEL_ID)
    cli_flags = getattr(args, "_cli_flags", set()) or set()
    if "model-id" in cli_flags or model_id != DEFAULT_PUBLIC_MODEL_ID:
        return model_id
    return _public_model_id_for_ref(
        str(model_ref or ""),
        default_model_id=DEFAULT_PUBLIC_MODEL_ID,
    )


def _batching_command_suffix(args: Any) -> str:
    parts: list[str] = []
    scheduler_mode = str(getattr(args, "scheduler_mode", "serial") or "serial")
    batching_preset = str(getattr(args, "batching_preset", "latency") or "latency")
    if scheduler_mode != "serial":
        parts.extend(["--scheduler-mode", shlex.quote(scheduler_mode)])
    if batching_preset != "latency":
        parts.extend(["--batching-preset", shlex.quote(batching_preset)])
    for attr, flag in (
        ("max_active_requests", "--max-active-requests"),
        ("decode_batch_max", "--decode-batch-max"),
        ("batch_wait_ms", "--batch-wait-ms"),
        ("prefill_chunk_tokens", "--prefill-chunk-tokens"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            parts.extend([flag, shlex.quote(str(value))])
    if bool(getattr(args, "experimental_mtp_cohorts", False)):
        parts.append("--experimental-mtp-cohorts")
    ssd_session_cache = str(getattr(args, "ssd_session_cache", "on") or "on")
    # Emit the mode unconditionally, "off" included: these generated
    # commands are re-parsed by CLIs whose own default is "on"
    # (kvcache-v2), so omitting an explicit "off" silently re-enables
    # the cache (issue #140 class).
    parts.extend(["--ssd-session-cache", shlex.quote(ssd_session_cache)])
    if ssd_session_cache != "off":
        ssd_dir = getattr(args, "ssd_session_cache_dir", None)
        if ssd_dir:
            parts.extend(["--ssd-session-cache-dir", shlex.quote(str(ssd_dir))])
        ssd_max_size = getattr(args, "ssd_session_cache_max_size", None)
        if ssd_max_size:
            parts.extend(
                ["--ssd-session-cache-max-size", shlex.quote(str(ssd_max_size))]
            )
        ssd_min_prefix = getattr(args, "ssd_session_cache_min_prefix_tokens", None)
        if ssd_min_prefix is not None:
            parts.extend(
                [
                    "--ssd-session-cache-min-prefix-tokens",
                    shlex.quote(str(ssd_min_prefix)),
                ]
            )
    paged_kv_quantization = str(getattr(args, "paged_kv_quantization", "off") or "off")
    if paged_kv_quantization != "off":
        parts.extend(["--paged-kv-quantization", shlex.quote(paged_kv_quantization)])
    return (" " + " ".join(parts)) if parts else ""


def _api_key_command_suffix(args: Any) -> str:
    api_key = str(getattr(args, "api_key", "") or "").strip()
    api_key_source = str(getattr(args, "api_key_source", "none") or "none")
    api_key_file = str(getattr(args, "api_key_file", "") or "").strip()
    if api_key_source == "file" and api_key_file:
        return f"--api-key-file {shlex.quote(api_key_file)} "
    if api_key == "mtplx-local":
        return "--api-key mtplx-local "
    if api_key and api_key_source in {"flag", "none"}:
        return "--api-key $MTPLX_API_KEY "
    return ""


def _api_key_display_value(value: str | None) -> str:
    if not value:
        return ""
    if value == "mtplx-local":
        return value
    return "<configured>"


def _redact_secret_from_payload(value: Any, secret: str | None) -> Any:
    if not secret or secret == "mtplx-local":
        return value
    if isinstance(value, str):
        return "<configured>" if value == secret else value
    if isinstance(value, list):
        return [_redact_secret_from_payload(item, secret) for item in value]
    if isinstance(value, dict):
        return {
            key: _redact_secret_from_payload(item, secret)
            for key, item in value.items()
        }
    return value


def _hermes_yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _hermes_dotenv_quote(value: str) -> str:
    return json.dumps(str(value))


def _hermes_home() -> Path:
    return Path.home() / ".hermes"


def _hermes_profile_dir() -> Path:
    return _hermes_home() / "profiles" / HERMES_PROFILE_NAME


def _hermes_workspace_path(args: Any) -> str:
    workspace = str(os.environ.get("HERMES_WORKSPACE") or "").strip()
    if workspace:
        return str(Path(workspace).expanduser())
    return str(Path.cwd())


def _hermes_launch_command(*, model_id: str, source: str = "mtplx-cli") -> str:
    parts = [
        "hermes",
        "-p",
        HERMES_PROFILE_NAME,
        "chat",
        "--model",
        model_id,
        "--toolsets",
        HERMES_CODING_TOOLSETS_TEXT,
        "--yolo",
        "--source",
        source,
    ]
    return _shell_join(parts)


def _hermes_terminal_command(*, model_id: str, workspace_path: str) -> str:
    profile_dir = _hermes_profile_dir()
    root_channel_dir = _hermes_home() / "channel_directory.json"
    profile_channel_dir = profile_dir / "channel_directory.json"
    commands = [
        f"cd {_shell_join([workspace_path])}",
        (
            f"if [ -f {_shell_join([str(root_channel_dir)])} ]; then "
            f"cp {_shell_join([str(root_channel_dir)])} "
            f"{_shell_join([str(profile_channel_dir)])}; "
            f"chmod 600 {_shell_join([str(profile_channel_dir)])} 2>/dev/null || true; fi"
        ),
        f"export HERMES_HOME={_shell_join([str(profile_dir)])}",
        f"exec {_hermes_launch_command(model_id=model_id)}",
    ]
    return "; ".join(commands)


def _hermes_config_yaml(
    *,
    model_id: str,
    base_url: str,
    api_key: str,
    workspace_path: str,
    reasoning_effort: str | None = None,
) -> str:
    # SYNC PAIR: HermesIntegration.configYAML — both writers must emit the
    # same template shape or the shared merge sweeps each other's lines.
    # model.default_headers is the only client-side identity hook hermes
    # exposes; without x-mtplx-client every hermes-conditional server branch
    # (tool contract, managed-thinking carve-out, injected-cap strip) is dead.
    # Reasoning effort must sit under agent: — hermes reads
    # CLI_CONFIG["agent"]["reasoning_effort"]; a model.reasoning_effort line
    # is silently ignored.
    effort_line = (
        f"  reasoning_effort: {_hermes_yaml_quote(reasoning_effort)}\n"
        if reasoning_effort
        else ""
    )
    return (
        "model:\n"
        f"  default: {_hermes_yaml_quote(model_id)}\n"
        "  provider: custom\n"
        f"  base_url: {_hermes_yaml_quote(base_url)}\n"
        f"  api_key: {_hermes_yaml_quote(api_key)}\n"
        "  api_mode: chat_completions\n"
        "  default_headers:\n"
        "    x-mtplx-client: hermes\n"
        "toolsets:\n"
        + "".join(f"  - {toolset}\n" for toolset in HERMES_CODING_TOOLSETS)
        + "agent:\n"
        f"  system_prompt: {_hermes_yaml_quote(HERMES_SYSTEM_PROMPT)}\n"
        "  max_turns: 200\n"
        "  tool_use_enforcement: auto\n"
        + effort_line
        + "terminal:\n"
        "  backend: local\n"
        f"  cwd: {_hermes_yaml_quote(workspace_path)}\n"
        "  timeout: 180\n"
        "  persistent_shell: true\n"
        "display:\n"
        "  streaming: true\n"
        "  show_reasoning: true\n"
        "  tool_progress: all\n"
    )


def _hermes_dotenv(
    *,
    model_id: str,
    base_url: str,
    api_key: str,
    workspace_path: str,
) -> str:
    return (
        f"OPENAI_BASE_URL={_hermes_dotenv_quote(base_url)}\n"
        f"CUSTOM_BASE_URL={_hermes_dotenv_quote(base_url)}\n"
        f"OPENAI_API_KEY={_hermes_dotenv_quote(api_key)}\n"
        f"HERMES_MODEL={_hermes_dotenv_quote(model_id)}\n"
        f"HERMES_INFERENCE_MODEL={_hermes_dotenv_quote(model_id)}\n"
        "HERMES_INFERENCE_PROVIDER=custom\n"
        "HERMES_YOLO_MODE=1\n"
        f"HERMES_MTPLX_TOOLSETS={_hermes_dotenv_quote(HERMES_CODING_TOOLSETS_TEXT)}\n"
        f"HERMES_MTPLX_CAPABILITIES={_hermes_dotenv_quote(HERMES_CAPABILITY_SUMMARY)}\n"
        f"HERMES_MTPLX_MESSAGING_NOTE={_hermes_dotenv_quote(HERMES_MESSAGING_SETUP_HINT)}\n"
        f"HERMES_MTPLX_GATEWAY_STATUS_COMMAND={_hermes_dotenv_quote(HERMES_GATEWAY_STATUS_COMMAND)}\n"
        f"HERMES_MTPLX_GATEWAY_TRUTH_NOTE={_hermes_dotenv_quote(HERMES_GATEWAY_TRUTH_HINT)}\n"
        f"HERMES_WORKSPACE={_hermes_dotenv_quote(workspace_path)}\n"
        f"TERMINAL_CWD={_hermes_dotenv_quote(workspace_path)}\n"
    )


# Issue #131: the Hermes profile config is user-editable. Regenerating it
# from the template alone silently wiped every top-level section the app/CLI
# does not own (memory/providers/delegation/…) and user-added child keys such
# as model.max_tokens. The template is merged over the existing file instead:
# app-owned keys are rewritten, everything else is preserved byte-for-byte.
# SYNC PAIR: HermesIntegration.mergedConfigYAML in
# apps/MTPLXApp/Sources/MTPLXAppCore/Services/HermesIntegration.swift — the
# app and the CLI write the same file and must share merge semantics.

# Children owned under a template section even when the current template does
# not emit them — conditional lines must be able to disappear instead of
# being resurrected as user content. Both writers emit agent.reasoning_effort
# only while an effort is configured. "model" stays owned because
# pre-2026-08-22 writers emitted reasoning_effort under model: (a key hermes
# never read); owning it sweeps the stale line from user files.
_HERMES_CONDITIONALLY_OWNED_CHILD_KEYS: dict[str, frozenset[str]] = {
    "model": frozenset({"reasoning_effort"}),
    "agent": frozenset({"reasoning_effort"}),
}


def _hermes_top_level_key(line: str) -> str | None:
    if not line or line[0] in (" ", "\t", "#", "-", "%"):
        return None
    head, colon, _tail = line.partition(":")
    if not colon:
        return None
    key = head.strip().strip("\"'")
    return key or None


def _hermes_direct_child_key(line: str) -> str | None:
    if not line.startswith("  ") or line.startswith("   "):
        return None
    rest = line[2:]
    if not rest or rest[0] in (" ", "\t", "#", "-"):
        return None
    head, colon, _tail = rest.partition(":")
    if not colon:
        return None
    key = head.strip().strip("\"'")
    return key or None


def _hermes_parse_top_level_blocks(
    text: str,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """Split YAML text into (preamble, top-level blocks, trailing lines).

    Each block is {"leading": [...], "key": str, "lines": [...]} with the key
    line and everything under it kept verbatim; comment/blank lines directly
    above a block travel with it.
    """
    preamble: list[str] = []
    blocks: list[dict[str, Any]] = []
    pending: list[str] = []
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for line in lines:
        key = _hermes_top_level_key(line)
        if key is not None:
            blocks.append({"leading": pending, "key": key, "lines": [line]})
            pending = []
        elif not line or line.startswith("#"):
            # Could belong to the current block or introduce the next one;
            # decided when the following line arrives (or at EOF).
            if blocks:
                pending.append(line)
            else:
                preamble.append(line)
        elif blocks:
            # Indented content or a column-zero continuation (sequence
            # items, flow scalars): body of the current block, together with
            # any buffered comment/blank lines above it.
            blocks[-1]["lines"].extend(pending)
            pending = []
            blocks[-1]["lines"].append(line)
        else:
            preamble.extend(pending)
            pending = []
            preamble.append(line)
    return preamble, blocks, pending


def _hermes_direct_child_blocks(
    block: dict[str, Any],
) -> list[tuple[str, list[str]]]:
    children: list[tuple[str, list[str]]] = []
    for line in block["lines"][1:]:
        key = _hermes_direct_child_key(line)
        if key is not None:
            children.append((key, [line]))
        elif children:
            children[-1][1].append(line)
    return children


def _hermes_merged_config_yaml(existing: str | None, template: str) -> str:
    """Merge the generated template over the existing profile config.

    Template sections and their child keys are rewritten every sync; unknown
    top-level sections, unknown child keys under owned sections, comments,
    and blank lines are preserved byte-for-byte. Sequence-shaped owned
    sections (toolsets) are rewritten wholly. Idempotent, so an unchanged
    configuration produces an unchanged file and the write is skipped.
    """
    if existing is None or not existing.strip():
        return template
    _t_preamble, t_blocks, _t_trailing = _hermes_parse_top_level_blocks(template)
    e_preamble, e_blocks, e_trailing = _hermes_parse_top_level_blocks(existing)
    if not e_blocks:
        return template
    template_keys = {block["key"] for block in t_blocks}
    out: list[str] = list(e_preamble)
    for t_block in t_blocks:
        e_block = next(
            (block for block in e_blocks if block["key"] == t_block["key"]), None
        )
        if e_block is not None:
            out.extend(e_block["leading"])
        out.extend(t_block["lines"])
        if e_block is None:
            continue
        owned = {key for key, _lines in _hermes_direct_child_blocks(t_block)}
        owned |= _HERMES_CONDITIONALLY_OWNED_CHILD_KEYS.get(t_block["key"], frozenset())
        # A template section without mapping children is sequence- or
        # scalar-shaped; its full body is owned.
        if not owned:
            continue
        for key, child_lines in _hermes_direct_child_blocks(e_block):
            if key not in owned:
                out.extend(child_lines)
    for e_block in e_blocks:
        if e_block["key"] not in template_keys:
            out.extend(e_block["leading"])
            out.extend(e_block["lines"])
    out.extend(e_trailing)
    return "\n".join(out) + "\n"


def _write_if_changed(path: Path, text: str, *, mode: int = 0o600) -> bool:
    existing = None
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError:
            existing = None
    changed = existing != text
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    try:
        path.chmod(mode)
    except OSError:
        pass
    return changed


def _hermes_client_reasoning_effort(args: Any) -> str | None:
    """Explicit effort for the hermes profile; "auto" stays server-side.

    hermes validates ``agent.reasoning_effort`` against its fixed ladder and
    warns + falls back to medium on unknown values, so the "auto" sentinel
    (server-resolved family default) must never be written client-side.
    """

    reasoning_effort = getattr(args, "reasoning_effort", None)
    if not reasoning_effort or str(reasoning_effort) == "auto":
        return None
    return str(reasoning_effort)


def _sync_hermes_profile(
    *,
    model_id: str,
    base_url: str,
    api_key: str,
    workspace_path: str,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    profile_dir = _hermes_profile_dir()
    config_path = profile_dir / "config.yaml"
    env_path = profile_dir / ".env"
    profile_dir.mkdir(parents=True, exist_ok=True)
    existing_config: str | None = None
    if config_path.exists():
        try:
            existing_config = config_path.read_text(encoding="utf-8")
        except OSError:
            existing_config = None
    config_changed = _write_if_changed(
        config_path,
        _hermes_merged_config_yaml(
            existing_config,
            _hermes_config_yaml(
                model_id=model_id,
                base_url=base_url,
                api_key=api_key,
                workspace_path=workspace_path,
                reasoning_effort=reasoning_effort,
            ),
        ),
    )
    env_changed = _write_if_changed(
        env_path,
        _hermes_dotenv(
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            workspace_path=workspace_path,
        ),
    )
    did_mirror_channels = False
    root_channel_dir = _hermes_home() / "channel_directory.json"
    profile_channel_dir = profile_dir / "channel_directory.json"
    if root_channel_dir.exists():
        try:
            shutil.copy2(root_channel_dir, profile_channel_dir)
            profile_channel_dir.chmod(0o600)
            did_mirror_channels = True
        except OSError:
            did_mirror_channels = False
    return {
        "profile_name": HERMES_PROFILE_NAME,
        "profile_path": str(profile_dir),
        "config_path": str(config_path),
        "env_path": str(env_path),
        "did_change": config_changed or env_changed or did_mirror_channels,
        "did_mirror_channels": did_mirror_channels,
    }


def _apply_ram_session_cache_env(env: dict[str, str], args: Any) -> None:
    policy = str(getattr(args, "ram_session_cache_policy", "") or "").strip().lower()
    if not policy or policy == "target-default":
        return
    if policy == "minimal":
        env["MTPLX_SESSION_BLOCK_PREFIX_RESTORE"] = "0"
        env["MTPLX_SESSION_BANK_MAX_ENTRIES"] = "1"
        env["MTPLX_SESSION_BANK_MAX_BYTES"] = "1G"
        env["MTPLX_SESSION_BANK_PER_SESSION_BYTES"] = "1G"
        return
    if policy != "bounded":
        return
    block_prefix_restore = getattr(args, "ram_session_block_prefix_restore", None)
    env["MTPLX_SESSION_BLOCK_PREFIX_RESTORE"] = (
        "1" if block_prefix_restore is not False else "0"
    )
    env["MTPLX_SESSION_BANK_MAX_ENTRIES"] = str(
        int(getattr(args, "ram_session_cache_max_entries", None) or 4)
    )
    env["MTPLX_SESSION_BANK_MAX_BYTES"] = str(
        getattr(args, "ram_session_cache_max_size", None) or "8G"
    )
    env["MTPLX_SESSION_BANK_PER_SESSION_BYTES"] = str(
        getattr(args, "ram_session_cache_per_session_max_size", None) or "4G"
    )


def _adaptive_command_suffix(args: Any) -> str:
    parts: list[str] = []
    policy = str(getattr(args, "adaptive_policy", "none") or "none")
    if policy == "none":
        return ""
    parts.extend(["--adaptive-policy", shlex.quote(policy)])
    parts.extend(
        [
            "--adaptive-min-depth",
            shlex.quote(str(getattr(args, "adaptive_min_depth", 1))),
        ]
    )
    if policy == "streak":
        for attr, flag in (
            ("adaptive_start_depth", "--adaptive-start-depth"),
            ("adaptive_increase_after", "--adaptive-increase-after"),
            ("adaptive_decrease_after", "--adaptive-decrease-after"),
        ):
            parts.extend([flag, shlex.quote(str(getattr(args, attr)))])
    elif policy == "position_ema":
        parts.extend(
            [
                "--adaptive-position-depth-cap",
                shlex.quote(str(getattr(args, "adaptive_position_depth_cap", 4))),
            ]
        )
        qwen38_q4_mtp_block = getattr(args, "qwen38_q4_mtp_block", None)
        if qwen38_q4_mtp_block is not None:
            parts.extend(
                [
                    "--qwen38-q4-mtp-block",
                    shlex.quote(str(qwen38_q4_mtp_block)),
                ]
            )
    elif policy == "expected_value":
        for attr, flag in (
            ("adaptive_ev_base_depth", "--adaptive-ev-base-depth"),
            ("adaptive_ev_accept_priors", "--adaptive-ev-accept-priors"),
            ("adaptive_ev_draft_cost_s", "--adaptive-ev-draft-cost-s"),
            ("adaptive_ev_extra_verify_cost_s", "--adaptive-ev-extra-verify-cost-s"),
            ("adaptive_ev_baseline_tok_s", "--adaptive-ev-baseline-tok-s"),
            ("adaptive_ev_safety_margin", "--adaptive-ev-safety-margin"),
            ("adaptive_ev_margin_center", "--adaptive-ev-margin-center"),
            ("adaptive_ev_margin_scale", "--adaptive-ev-margin-scale"),
            ("adaptive_ev_confidence_weight", "--adaptive-ev-confidence-weight"),
            (
                "adaptive_ev_min_extra_accept_probability",
                "--adaptive-ev-min-extra-accept-probability",
            ),
            (
                "adaptive_ev_warmup_full_depth_cycles",
                "--adaptive-ev-warmup-full-depth-cycles",
            ),
            ("adaptive_ev_exploration_interval", "--adaptive-ev-exploration-interval"),
        ):
            parts.extend([flag, shlex.quote(str(getattr(args, attr)))])
    return (" " + " ".join(parts)) if parts else ""


def _reasoning_command_suffix(
    args: Any,
    *,
    default: str = "auto",
    include_mode: bool = True,
) -> str:
    parts: list[str] = []
    reasoning_mode = _reasoning_mode(args, default=default)
    if include_mode and getattr(args, "reasoning", None) is not None:
        parts.extend(["--reasoning", shlex.quote(reasoning_mode)])
    reasoning_parser = getattr(args, "reasoning_parser", None)
    if reasoning_parser:
        parts.extend(["--reasoning-parser", shlex.quote(str(reasoning_parser))])
    reasoning_effort = getattr(args, "reasoning_effort", None)
    cli_flags = getattr(args, "_cli_flags", set()) or set()
    if reasoning_effort and (
        str(reasoning_effort) != "auto" or "reasoning-effort" in cli_flags
    ):
        parts.extend(["--reasoning-effort", shlex.quote(str(reasoning_effort))])
    return (" " + " ".join(parts)) if parts else ""


def _fan_mode_command_suffix(args: Any) -> str:
    mode = _fan_mode_from_args(args)
    return f"--fan-mode {shlex.quote(mode)} "


def _server_sampler_command_suffix(args: Any, *, include_draft: bool) -> str:
    parts: list[str] = []
    for attr, flag in (
        ("temperature", "--default-temperature"),
        ("top_p", "--default-top-p"),
        ("top_k", "--top-k"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            parts.extend([flag, shlex.quote(str(value))])
    if include_draft:
        for attr, flag in (
            ("draft_temperature", "--draft-temperature"),
            ("draft_top_p", "--draft-top-p"),
            ("draft_top_k", "--draft-top-k"),
        ):
            value = getattr(args, attr, None)
            if value is not None:
                parts.extend([flag, shlex.quote(str(value))])
    return (" " + " ".join(parts)) if parts else ""


def _bridge_prompt_command_suffix(args: Any) -> str:
    parts: list[str] = []
    tool_prompt_mode = getattr(args, "tool_prompt_mode", None)
    if tool_prompt_mode:
        parts.extend(["--tool-prompt-mode", shlex.quote(str(tool_prompt_mode))])
    chat_template_profile = getattr(args, "chat_template_profile", None)
    if chat_template_profile:
        parts.extend(
            ["--chat-template-profile", shlex.quote(str(chat_template_profile))]
        )
    chat_template_path = getattr(args, "chat_template_path", None)
    if chat_template_path:
        parts.extend(["--chat-template-path", shlex.quote(str(chat_template_path))])
    return (" " + " ".join(parts)) if parts else ""


def _quickstart_openwebui_payload(
    args: Any,
    *,
    inspection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 8000))
    model_id = _public_model_id_for_args(args, str(getattr(args, "model", "")))
    base = f"http://{_connect_host_for_bind(host)}:{port}"
    context_window = _inspection_context_window(inspection, args=args)
    return {
        "integration": "openwebui",
        "server_url": base,
        "base_url": base + "/v1",
        "api_base_url": base + "/v1",
        "chat_url": base + "/",
        "model_id": model_id,
        "context_window": context_window,
        "api_key": "not required for localhost",
        "server_command": (
            f"mtplx quickstart --host {host} --port {port} "
            f"--model {shlex.quote(str(getattr(args, 'model', DEFAULT_RUNTIME_MODEL_DIR)))} "
            f"--profile {_resolved_default_profile_name(args)} "
            f"{_fan_mode_command_suffix(args)}"
            f"{'--no-mtp ' if _generation_mode_from_args(args) == GENERATION_MODE_AR else ''}"
            f"--context-window {context_window} "
            f"{_batching_command_suffix(args)} "
            f"{_server_sampler_command_suffix(args, include_draft=True)} "
            f"{_reasoning_command_suffix(args)} "
            f"{_bridge_prompt_command_suffix(args)} "
            "--no-stats-footer --open-browser"
        ),
        "openwebui_steps": [
            f"Open chat UI: {base}/",
            f"OpenAI-compatible API base URL: {base}/v1",
            f"Model: {model_id}",
        ],
    }


def _pi_sampler_temperature(args: Any) -> float:
    return float(getattr(args, "temperature", 0.6))


def _pi_sampler_top_p(args: Any) -> float:
    return float(getattr(args, "top_p", 0.95))


def _pi_sampler_top_k(args: Any) -> int:
    return int(getattr(args, "top_k", 20))


def _quickstart_pi_payload(
    args: Any,
    *,
    write_config: bool = False,
    inspection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from mtplx.pi import (
        PI_LOCAL_API_KEY,
        build_pi_provider_config,
        pi_launch_command,
        pi_model_ref,
        pi_models_json_path,
        pi_request_policy_extension_path,
        write_pi_models_config,
    )

    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 8000))
    model_id = _public_model_id_for_args(args, str(getattr(args, "model", "")))
    base_url = f"http://{_connect_host_for_bind(host)}:{port}/v1"
    api_key = str(getattr(args, "api_key", None) or PI_LOCAL_API_KEY)
    pi_temperature = _pi_sampler_temperature(args)
    pi_top_p = _pi_sampler_top_p(args)
    pi_top_k = _pi_sampler_top_k(args)
    # Pi shares the general auto resolution: the family contract governs the
    # reasoning-history policy (Qwen 3.8 preserves, checkpoint templates run
    # scoped). The 1.0.0-era Pi-only hard "off" predated scoped mode (2.0.2)
    # and kept actively stripping Pi's echoed reasoning_content history after
    # every other lane moved to the trained contract (issue #310 receipts).
    pi_preserve_thinking = _preserve_thinking_policy(args)
    context_window = _inspection_context_window(inspection, args=args)
    api_key_command_suffix = _api_key_command_suffix(args) or "--api-key mtplx-local "
    provider = build_pi_provider_config(
        base_url=base_url,
        model_id=model_id,
        model_name=f"MTPLX {model_id}",
        api_key=api_key,
        context_window=context_window,
    )
    payload = {
        "integration": "pi",
        "server_url": f"http://{_connect_host_for_bind(host)}:{port}",
        "base_url": base_url,
        "api_base_url": base_url,
        "model_id": model_id,
        "model_ref": pi_model_ref(model_id),
        "context_window": context_window,
        "api_key": _api_key_display_value(api_key),
        "config_path": str(pi_models_json_path()),
        "provider": _redact_secret_from_payload(provider, api_key),
        "no_hidden_max_tokens": True,
        "request_policy_extension_path": str(pi_request_policy_extension_path()),
        "launch_command": pi_launch_command(model_id),
        "server_console": True,
        "server_controls": [
            "/reasoning on|off|auto|status",
            "/mtp on|off|status",
            "/stats",
            "/help",
        ],
        "server_command": (
            f"mtplx quickstart --host {host} --port {port} "
            f"--model {shlex.quote(str(getattr(args, 'model', DEFAULT_RUNTIME_MODEL_DIR)))} "
            f"--profile {_resolved_default_profile_name(args)} "
            f"{_fan_mode_command_suffix(args)}"
            f"{'--no-mtp ' if _generation_mode_from_args(args) == GENERATION_MODE_AR else ''}"
            f"--context-window {context_window} "
            f"{_batching_command_suffix(args)} "
            f"--default-temperature {pi_temperature} "
            f"--default-top-p {pi_top_p} --top-k {pi_top_k} "
            f"--preserve-thinking {pi_preserve_thinking} "
            f"{_reasoning_command_suffix(args)} "
            f"{_bridge_prompt_command_suffix(args)} "
            f"{api_key_command_suffix}--no-stats-footer"
        ),
        "pi_steps": [
            f"Pi config: {pi_models_json_path()}",
            f"Model in Pi: {pi_model_ref(model_id)}",
            f"Start Pi: {pi_launch_command(model_id)}",
        ],
    }
    if write_config:
        payload["config_write"] = write_pi_models_config(
            base_url=base_url,
            model_id=model_id,
            model_name=f"MTPLX {model_id}",
            api_key=api_key,
            context_window=context_window,
        )
    return payload


def _inspection_context_window(
    inspection: dict[str, Any] | None,
    *,
    args: Any | None = None,
) -> int:
    descriptor = descriptor_from_inspection(inspection)
    requested = getattr(args, "context_window", None) if args is not None else None
    if requested is not None and int(requested) > 0:
        return min(int(requested), int(descriptor.context_window_policy.maximum))
    if not isinstance(inspection, dict):
        return int(descriptor.context_window_policy.default)
    candidates: list[int] = []
    for key in (
        "context_window",
        "context_length",
        "max_context_length",
        "max_model_len",
        "model_max_length",
    ):
        value = inspection.get(key)
        if isinstance(value, int):
            candidates.append(value)
    compatibility = inspection.get("compatibility")
    if isinstance(compatibility, dict):
        for key in ("context_window", "context_length", "max_context_length"):
            value = compatibility.get(key)
            if isinstance(value, int):
                candidates.append(value)
    sane = [value for value in candidates if 0 < value <= 1_048_576]
    return max(sane) if sane else int(descriptor.context_window_policy.default)


def _inspection_tool_prompt_mode(
    args: Any,
    inspection: dict[str, Any] | None,
) -> str:
    descriptor = descriptor_from_inspection(inspection)
    if descriptor.required_tool_prompt_mode is not None:
        return descriptor.required_tool_prompt_mode
    cli_flags = getattr(args, "_cli_flags", set()) or set()
    if "tool-prompt-mode" in cli_flags:
        return str(
            getattr(args, "tool_prompt_mode", descriptor.default_tool_prompt_mode)
            or descriptor.default_tool_prompt_mode
        )
    return descriptor.default_tool_prompt_mode


def _quickstart_opencode_payload(
    args: Any,
    *,
    write_config: bool = False,
    inspection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from mtplx.opencode import (
        build_opencode_provider_config,
        detect_opencode_desktop,
        opencode_config_path,
        opencode_model_ref,
        write_opencode_config,
    )

    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 8000))
    model_id = _public_model_id_for_args(args, str(getattr(args, "model", "")))
    base_url = f"http://{_connect_host_for_bind(host)}:{port}/v1"
    context_window = _inspection_context_window(inspection, args=args)
    reasoning_mode = _reasoning_mode(args, default="auto")
    # The declared OpenCode reasoning capability mirrors the model contract:
    # a family with a verified codec (unless the user forced --reasoning off),
    # never an unknown model. The resolved public id is the fallback ref so
    # inspection-less lanes (`mtplx integrate opencode`) still resolve the
    # family from its marker.
    reasoning_policy = reasoning_policy_for_model(
        model_ref=str(getattr(args, "model", "") or "") or model_id,
        inspection=inspection,
    )
    enable_thinking = reasoning_mode != "off" and reasoning_policy.supported
    # The app/CLI dial is OpenCode's source of truth for reasoning effort:
    # an explicit --reasoning-effort wins, otherwise the family default from
    # the descriptor codec (Qwen3.8: medium). The family's effort levels
    # drive OpenCode's effort picker so it mirrors the MTPLX dial.
    reasoning_effort = getattr(args, "reasoning_effort", None)
    if reasoning_effort in (None, "auto"):
        reasoning_effort = reasoning_policy.default_effort
    if not enable_thinking:
        reasoning_effort = None
    reasoning_effort_levels = (
        tuple(reasoning_policy.effort_levels) if reasoning_policy.supported else None
    )
    tool_prompt_mode = _inspection_tool_prompt_mode(args, inspection)
    chat_template_profile = str(
        getattr(args, "chat_template_profile", OPENCODE_CHAT_TEMPLATE_PROFILE_DEFAULT)
        or OPENCODE_CHAT_TEMPLATE_PROFILE_DEFAULT
    )
    opencode_max_response_tokens = getattr(args, "max_response_tokens", None)
    output_limit = min(
        context_window,
        int(opencode_max_response_tokens or context_window),
    )
    max_response_suffix = (
        f"--max-response-tokens {int(opencode_max_response_tokens)} "
        if opencode_max_response_tokens is not None
        else ""
    )
    api_key_suffix = _api_key_command_suffix(args)
    profile = get_profile(_resolved_default_profile_name(args))
    generation_mode = _generation_mode_from_args(args)
    target_sampler = {
        "temperature": float(getattr(args, "temperature", 0.6)),
        "top_p": float(getattr(args, "top_p", 0.95)),
        "top_k": int(getattr(args, "top_k", 20)),
    }
    draft_sampler = (
        _model_draft_sampler_spec(inspection, profile)
        if inspection is not None
        else None
    )
    draft_sampler_source = (
        "model_contract_or_profile" if draft_sampler is not None else None
    )
    draft_sampler_override = _explicit_draft_sampler_override(args, draft_sampler)
    if draft_sampler_override is not None:
        draft_sampler = draft_sampler_override
        draft_sampler_source = "explicit_cli"
    sampler_suffix = (
        f"--depth {int(getattr(args, 'depth', 3))} "
        f"--temperature {target_sampler['temperature']} "
        f"--top-p {target_sampler['top_p']} "
        f"--top-k {target_sampler['top_k']} "
    )
    draft_sampler_suffix = (
        (
            f"--draft-temperature {float(draft_sampler['temperature'])} "
            f"--draft-top-p {float(draft_sampler['top_p'])} "
            f"--draft-top-k {int(draft_sampler['top_k'])} "
        )
        if draft_sampler is not None and generation_mode == GENERATION_MODE_MTP
        else ""
    )
    provider_fragment = build_opencode_provider_config(
        base_url=base_url,
        model_id=model_id,
        model_name=f"MTPLX {model_id}",
        api_key=getattr(args, "api_key", None),
        context_window=context_window,
        output_limit=output_limit,
        enable_thinking=enable_thinking,
        top_p=float(getattr(args, "top_p", 0.95)),
        top_k=int(getattr(args, "top_k", 20)),
        reasoning_effort=reasoning_effort,
        reasoning_effort_levels=reasoning_effort_levels,
    )
    payload = {
        "integration": "opencode",
        "server_url": f"http://{_connect_host_for_bind(host)}:{port}",
        "base_url": base_url,
        "api_base_url": base_url,
        "model_id": model_id,
        "model_ref": opencode_model_ref(model_id),
        "config_path": str(opencode_config_path()),
        "provider": _redact_secret_from_payload(
            provider_fragment["provider"]["mtplx"],
            getattr(args, "api_key", None),
        ),
        "config": _redact_secret_from_payload(
            provider_fragment,
            getattr(args, "api_key", None),
        ),
        "detected": detect_opencode_desktop(),
        "context_window": context_window,
        "output_limit": output_limit,
        "transport_headers": {"x-mtplx-client": "opencode"},
        "reasoning_field": "reasoning_content",
        "reasoning_effort": reasoning_effort,
        "no_hidden_max_tokens": True,
        "tool_prompt_mode": tool_prompt_mode,
        "chat_template_profile": chat_template_profile,
        "mtp_depth": int(getattr(args, "depth", 3)),
        "target_sampler": target_sampler,
        "draft_sampler": draft_sampler,
        "draft_sampler_source": draft_sampler_source,
        "server_console": True,
        "server_controls": [
            "/reasoning on|off|auto|status",
            "/mtp on|off|status",
            "/stats",
            "/help",
        ],
        "server_command": (
            f"mtplx start opencode --host {host} --port {port} "
            f"--model {shlex.quote(str(getattr(args, 'model', DEFAULT_RUNTIME_MODEL_DIR)))} "
            f"--profile {_resolved_default_profile_name(args)} "
            f"{_fan_mode_command_suffix(args)}"
            f"{'--no-mtp ' if generation_mode == GENERATION_MODE_AR else ''}"
            f"{api_key_suffix}"
            f"--context-window {context_window} "
            f"{_batching_command_suffix(args)} "
            f"{_adaptive_command_suffix(args)} "
            f"{sampler_suffix}"
            f"{draft_sampler_suffix}"
            f"{max_response_suffix}"
            f"--tool-prompt-mode {tool_prompt_mode} "
            f"--chat-template-profile {chat_template_profile} "
            f"--reasoning {reasoning_mode} "
            f"{_reasoning_command_suffix(args, default='auto', include_mode=False)} --no-stats"
        ),
        "opencode_steps": [
            f"OpenCode config: {opencode_config_path()}",
            "Transport header: x-mtplx-client=opencode",
            f"Model in OpenCode: {opencode_model_ref(model_id)}",
            f"OpenAI-compatible API base URL: {base_url}",
            "Reasoning is controlled by MTPLX server settings.",
        ],
    }
    if write_config:
        payload["config_write"] = write_opencode_config(
            base_url=base_url,
            model_id=model_id,
            model_name=f"MTPLX {model_id}",
            api_key=getattr(args, "api_key", None),
            context_window=context_window,
            output_limit=output_limit,
            enable_thinking=enable_thinking,
            top_p=float(getattr(args, "top_p", 0.95)),
            top_k=int(getattr(args, "top_k", 20)),
            reasoning_effort=reasoning_effort,
            reasoning_effort_levels=reasoning_effort_levels,
        )
    return payload


def _quickstart_swival_payload(
    args: Any,
    *,
    inspection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from mtplx.swival import (
        build_swival_command,
        detect_swival_cli,
        shell_swival_command,
    )

    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 8000))
    model_id = _public_model_id_for_args(args, str(getattr(args, "model", "")))
    server_url = f"http://{_connect_host_for_bind(host)}:{port}"
    context_window = _inspection_context_window(inspection, args=args)
    command_argv = build_swival_command(
        base_url=server_url,
        model_id=model_id,
        context_window=context_window,
    )
    launch_command = shell_swival_command(
        base_url=server_url,
        model_id=model_id,
        context_window=context_window,
    )
    return {
        "integration": "swival",
        "server_url": server_url,
        "base_url": server_url,
        "api_base_url": server_url.rstrip("/") + "/v1",
        "model_id": model_id,
        "context_window": context_window,
        "detected": detect_swival_cli(),
        "launch_command": launch_command,
        "command_argv": command_argv,
        "no_hidden_max_tokens": True,
        "server_console": True,
        "server_controls": [
            "/reasoning on|off|auto|status",
            "/mtp on|off|status",
            "/stats",
            "/help",
        ],
        "server_command": (
            f"mtplx start swival --host {host} --port {port} "
            f"--model {shlex.quote(str(getattr(args, 'model', DEFAULT_RUNTIME_MODEL_DIR)))} "
            f"--profile {_resolved_default_profile_name(args)} "
            f"{_fan_mode_command_suffix(args)}"
            f"{'--no-mtp ' if _generation_mode_from_args(args) == GENERATION_MODE_AR else ''}"
            f"--context-window {context_window} "
            f"{_batching_command_suffix(args)} "
            "--no-stats"
        ),
        "swival_steps": [
            f"Start MTPLX server: {server_url}",
            f"Run Swival: {launch_command}",
            "Provider: generic",
        ],
    }


def _quickstart_hermes_payload(
    args: Any,
    *,
    write_config: bool = False,
    inspection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 8000))
    model_id = _public_model_id_for_args(args, str(getattr(args, "model", "")))
    server_url = f"http://{_connect_host_for_bind(host)}:{port}"
    base_url = server_url.rstrip("/") + "/v1"
    api_key = str(getattr(args, "api_key", None) or HERMES_LOCAL_API_KEY)
    workspace_path = _hermes_workspace_path(args)
    context_window = _inspection_context_window(inspection, args=args)
    launch_command = _hermes_launch_command(model_id=model_id)
    terminal_command = _hermes_terminal_command(
        model_id=model_id,
        workspace_path=workspace_path,
    )
    api_key_suffix = _api_key_command_suffix(args) or "--api-key mtplx-local "
    draft_sampler_suffix = "".join(
        f"{flag} {getattr(args, attr)} "
        for attr, flag in (
            ("draft_temperature", "--draft-temperature"),
            ("draft_top_p", "--draft-top-p"),
            ("draft_top_k", "--draft-top-k"),
        )
        if getattr(args, attr, None) is not None
    )
    payload = {
        "integration": "hermes",
        "server_url": server_url,
        "base_url": base_url,
        "api_base_url": base_url,
        "model_id": model_id,
        "profile_name": HERMES_PROFILE_NAME,
        "profile_path": str(_hermes_profile_dir()),
        "config_path": str(_hermes_profile_dir() / "config.yaml"),
        "env_path": str(_hermes_profile_dir() / ".env"),
        "workspace_path": workspace_path,
        "toolsets": list(HERMES_CODING_TOOLSETS),
        "toolsets_arg": HERMES_CODING_TOOLSETS_TEXT,
        "capability_summary": HERMES_CAPABILITY_SUMMARY,
        "context_window": context_window,
        "api_key": _api_key_display_value(api_key),
        "detected": {
            "installed": shutil.which("hermes") is not None,
            "path": shutil.which("hermes"),
        },
        "launch_command": launch_command,
        "terminal_command": terminal_command,
        "gateway_status_command": HERMES_GATEWAY_STATUS_COMMAND,
        "messaging_setup": HERMES_MESSAGING_SETUP_HINT,
        "server_console": True,
        "server_controls": [
            "/reasoning on|off|auto|status",
            "/mtp on|off|status",
            "/stats",
            "/help",
        ],
        "server_command": (
            f"mtplx start hermes --host {host} --port {port} "
            f"--model {shlex.quote(str(getattr(args, 'model', DEFAULT_RUNTIME_MODEL_DIR)))} "
            f"--profile {_resolved_default_profile_name(args)} "
            f"{_fan_mode_command_suffix(args)}"
            f"{'--no-mtp ' if _generation_mode_from_args(args) == GENERATION_MODE_AR else ''}"
            f"{api_key_suffix}"
            f"--context-window {context_window} "
            f"--scheduler-mode {str(getattr(args, 'scheduler_mode', 'serial'))} "
            f"--batching-preset {str(getattr(args, 'batching_preset', 'latency'))} "
            f"{_batching_command_suffix(args)} "
            f"{_adaptive_command_suffix(args)} "
            f"--temperature {float(getattr(args, 'temperature', 0.6))} "
            f"--top-p {float(getattr(args, 'top_p', 1.0))} "
            f"--top-k {int(getattr(args, 'top_k', 20))} "
            f"{draft_sampler_suffix}"
            f"--tool-prompt-mode {str(getattr(args, 'tool_prompt_mode', 'hybrid'))} "
            f"--chat-template-profile {str(getattr(args, 'chat_template_profile', OPENCODE_CHAT_TEMPLATE_PROFILE_DEFAULT))} "
            f"--reasoning {_reasoning_mode(args, default='auto')} "
            f"{_reasoning_command_suffix(args, default='auto', include_mode=False)} --no-stats"
        ),
        "hermes_steps": [
            f"Hermes profile: {_hermes_profile_dir()}",
            f"OpenAI-compatible API base URL: {base_url}",
            f"Run Hermes: {launch_command}",
            "Tools: " + HERMES_CODING_TOOLSETS_TEXT,
        ],
    }
    if write_config:
        payload["config_write"] = _sync_hermes_profile(
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            workspace_path=workspace_path,
            reasoning_effort=_hermes_client_reasoning_effort(args),
        )
    return payload


def _quickstart_print_openwebui_handoff(args: Any, *, runtime_model: str) -> None:
    # The full banner + status panel are rendered by `_print_serve_start_banner`
    # inside `cmd_serve_public`, so this hand-off only emits a brief progress
    # marker. Keeping it minimal avoids visual duplication of the panel.
    _quickstart_line("[2/3] Starting local MTPLX server for the browser chat...")
    _quickstart_line(f"      Loading model: {runtime_model}")
    _quickstart_line("      Keep this terminal open. The next step is the model load.")
    _quickstart_line()


def _quickstart_print_pi_handoff(
    args: Any, *, runtime_model: str, pi: dict[str, Any]
) -> None:
    _quickstart_line("[2/3] Connecting MTPLX to Pi...")
    config_write = (
        pi.get("config_write") if isinstance(pi.get("config_write"), dict) else {}
    )
    config_path = config_write.get("config_path") or pi.get("config_path")
    backup_path = config_write.get("backup_path")
    _quickstart_line(f"      Pi config: {config_path}")
    if backup_path:
        _quickstart_line(f"      Backed up unreadable old Pi config: {backup_path}")
    _quickstart_line(f"      Pi model: {pi.get('model_ref')}")
    _quickstart_line(f"      Loading model: {runtime_model}")
    _quickstart_line("      Keep this terminal open for the MTPLX server.")
    _quickstart_line("      Pi will open automatically when MTPLX is ready.")
    _quickstart_line("      Then type /help here for reasoning and MTP controls.")
    _quickstart_line(f"      Manual fallback: {pi.get('launch_command')}")
    _quickstart_line()


def _quickstart_launch_pi_now(*, model_id: str) -> None:
    from mtplx.pi import launch_pi_in_terminal, pi_launch_command, pi_model_ref

    model_ref = pi_model_ref(str(model_id))
    command = pi_launch_command(str(model_id))
    result = launch_pi_in_terminal(command, model_ref=model_ref)
    if result.get("ok"):
        _print_serve_start_line("Opening Pi in Terminal...")
    else:
        _print_serve_start_line(
            f"Could not open Pi automatically: {result.get('error')}"
        )
        _print_serve_start_line(f"Run manually: {command}")


def _quickstart_print_opencode_handoff(
    args: Any,
    *,
    runtime_model: str,
    opencode: dict[str, Any],
) -> None:
    _quickstart_line("[2/3] Connecting MTPLX to OpenCode Desktop...")
    config_write = (
        opencode.get("config_write")
        if isinstance(opencode.get("config_write"), dict)
        else {}
    )
    config_path = config_write.get("config_path") or opencode.get("config_path")
    backup_path = config_write.get("backup_path")
    _quickstart_line(f"      OpenCode config: {config_path}")
    _quickstart_line("      MTPLX client header: x-mtplx-client=opencode")
    if backup_path:
        _quickstart_line(
            f"      Backed up unreadable old OpenCode config: {backup_path}"
        )
    _quickstart_line(f"      OpenCode model: {opencode.get('model_ref')}")
    _quickstart_line(f"      API base URL: {opencode.get('api_base_url')}")
    _quickstart_line(
        "      Reasoning: "
        + (
            "enabled on the MTPLX server"
            if _reasoning_mode(args, default="auto") == "on"
            else "controlled by MTPLX server settings"
        )
    )
    _quickstart_line("      Response cap: none hidden by MTPLX")
    _quickstart_line(f"      Loading model: {runtime_model}")
    _quickstart_line("      Keep this terminal open for the MTPLX server.")
    _quickstart_line("      OpenCode will open automatically when MTPLX is ready.")
    _quickstart_line()


def _quickstart_launch_opencode_now() -> None:
    from mtplx.opencode import launch_opencode_app

    result = launch_opencode_app()
    if result.get("ok"):
        _print_serve_start_line("Opening OpenCode Desktop...")
    else:
        _print_serve_start_line(
            f"Could not open OpenCode automatically: {result.get('error')}"
        )
        _print_serve_start_line("Open OpenCode manually and select the MTPLX model.")


def _quickstart_print_swival_handoff(
    args: Any,
    *,
    runtime_model: str,
    swival: dict[str, Any],
) -> None:
    _quickstart_line("[2/3] Preparing Swival generic provider command...")
    _quickstart_line(f"      MTPLX server: {swival.get('server_url')}")
    _quickstart_line(f"      Swival model: {swival.get('model_id')}")
    _quickstart_line(f"      Context window: {swival.get('context_window')} tokens")
    _quickstart_line("      Provider: generic")
    _quickstart_line("      Response cap: none hidden by MTPLX")
    _quickstart_line(f"      Loading model: {runtime_model}")
    _quickstart_line("      Keep this terminal open for the MTPLX server.")
    _quickstart_line(
        f"      Run Swival in another terminal: {swival.get('launch_command')}"
    )
    _quickstart_line()


def _quickstart_print_hermes_handoff(
    args: Any,
    *,
    runtime_model: str,
    hermes: dict[str, Any],
) -> None:
    config_write = (
        hermes.get("config_write")
        if isinstance(hermes.get("config_write"), dict)
        else {}
    )
    config_path = config_write.get("config_path") or hermes.get("config_path")
    env_path = config_write.get("env_path") or hermes.get("env_path")
    _quickstart_line("[2/3] Connecting MTPLX to Hermes Agent...")
    _quickstart_line(f"      Hermes profile: {hermes.get('profile_name')}")
    _quickstart_line(f"      Hermes config: {config_path}")
    _quickstart_line(f"      Hermes env: {env_path}")
    _quickstart_line(f"      Hermes tools: {hermes.get('toolsets_arg')}")
    _quickstart_line(f"      API base URL: {hermes.get('api_base_url')}")
    _quickstart_line(f"      Loading model: {runtime_model}")
    _quickstart_line("      Keep this terminal open for the MTPLX server.")
    _quickstart_line("      Hermes will open automatically when MTPLX is ready.")
    _quickstart_line(f"      Manual fallback: {hermes.get('launch_command')}")
    _quickstart_line()


def _quickstart_require_pi_cli(args: Any) -> bool:
    """Return True when the Pi CLI is available, or install it interactively."""

    if shutil.which("pi"):
        return True

    from mtplx.pi import pi_install_command

    _quickstart_line()
    _quickstart_line("[2/3] Pi is not installed")
    _quickstart_line("      MTPLX has not loaded the model yet.")
    _quickstart_line(
        "      Install Pi first, then MTPLX will connect it to the local server."
    )
    _quickstart_line()
    _quickstart_line(f"      Install command: {pi_install_command()}")
    _quickstart_line(f"      Then re-run: {_start_invocation(args, ' pi')}")
    _quickstart_line()

    if not sys.stdin.isatty():
        return False

    try:
        answer = input("  Install Pi now? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        _quickstart_line("aborted")
        return False
    if answer not in {"", "y", "yes"}:
        _quickstart_line(
            "Pi install skipped. Re-run `mtplx start pi` after installing Pi."
        )
        return False

    npm = shutil.which("npm")
    if not npm:
        _quickstart_line(
            "error: npm is not on PATH, so MTPLX cannot install Pi automatically."
        )
        _quickstart_line("Install Node.js/npm, then run:")
        _quickstart_line(f"  {pi_install_command()}")
        return False

    _quickstart_line(f"Running: {pi_install_command()}")
    try:
        result = subprocess.run(
            [npm, "install", "-g", "@earendil-works/pi-coding-agent"], check=False
        )
    except KeyboardInterrupt:
        _quickstart_line("Pi install cancelled.")
        return False
    except OSError as exc:
        _quickstart_line(f"error: Pi install failed to start: {exc}")
        return False
    if result.returncode != 0:
        _quickstart_line(f"error: Pi install failed with exit code {result.returncode}")
        return False
    if not shutil.which("pi"):
        _quickstart_line("Pi installed, but `pi` is still not visible on this PATH.")
        _quickstart_line("Open a new terminal, then run:")
        _quickstart_line(f"  {_start_invocation(args, ' pi')}")
        return False
    _quickstart_line("Pi installed. Continuing with MTPLX setup.")
    _quickstart_line()
    return True


def _quickstart_apply_local_model_defaults(
    args: Any,
    *,
    model: str,
) -> dict[str, Any] | None:
    model_path = Path(str(model)).expanduser()
    if not model_path.exists():
        return None
    inspection, gate_exit = _model_gate(
        str(model_path),
        unsafe_force_unverified=bool(getattr(args, "unsafe_force_unverified", False)),
        yes=bool(getattr(args, "yes", False)),
    )
    profile = get_profile(_resolved_default_profile_name(args))
    _apply_model_contract_depth_default(args, inspection, profile)
    _apply_backend_serve_defaults(args, inspection)
    if gate_exit is not None:
        return inspection
    return inspection


def _quickstart_served_model_id(args: Any, runtime_model: str) -> str:
    model_id = _public_model_id_for_args(args, str(runtime_model))
    args.model_id = model_id
    return model_id


def _with_batching_args(target: Any, source: Any) -> Any:
    for attr, default in (
        ("scheduler_mode", "serial"),
        ("batching_preset", "latency"),
        ("max_active_requests", None),
        ("decode_batch_max", None),
        ("batch_wait_ms", None),
        ("prefill_chunk_tokens", None),
        ("experimental_mtp_cohorts", False),
        ("ssd_session_cache", "on"),
        ("ssd_session_cache_dir", None),
        ("ssd_session_cache_max_size", None),
        ("ssd_session_cache_min_prefix_tokens", 512),
    ):
        setattr(target, attr, getattr(source, attr, default))
    return target


def _with_server_policy_args(target: Any, source: Any) -> Any:
    setattr(target, "_cli_flags", getattr(source, "_cli_flags", set()) or set())
    # The config-pin marker must survive the quickstart -> serve handoff or
    # the child would re-promote over a config.toml profile pin.
    setattr(
        target, "_profile_from_config", getattr(source, "_profile_from_config", None)
    )
    # Same contract for the sampler pins: _apply_backend_serve_defaults reads
    # config presence from args.mtplx_config, and these handoffs forward the
    # raw sampler VALUES (a config temperature equal to the 0.6 parser
    # default is indistinguishable from unset without the parsed file).
    setattr(target, "mtplx_config", getattr(source, "mtplx_config", None))
    _with_batching_args(target, source)
    for attr, default in (
        # Retrieval models: quickstart builds its serve namespace field by
        # field, so without forwarding these the endpoints would silently stay
        # unconfigured on every path except a bare `mtplx serve`.
        ("embedding_model", []),
        ("reranker_model", []),
        ("retrieval_max_resident", 2),
        ("retrieval_max_tokens", 0),
        ("retrieval_idle_timeout", 0.0),
        ("retrieval_trust_remote_code", False),
        ("api_key_file", None),
        ("api_key_source", "none"),
        ("default_presence_penalty", 0.0),
        ("default_frequency_penalty", 0.0),
        ("paged_kv_quantization", "off"),
        ("tool_prompt_mode", "hybrid"),
        ("chat_template_profile", "local_qwen36"),
        ("chat_template_path", None),
        ("adaptive_policy", "none"),
        ("adaptive_min_depth", 1),
        ("adaptive_start_depth", 1),
        ("adaptive_increase_after", 4),
        ("adaptive_decrease_after", 1),
        ("adaptive_position_depth_cap", 4),
        ("adaptive_ev_base_depth", 2),
        ("adaptive_ev_accept_priors", "0.92,0.64,0.32"),
        ("adaptive_ev_draft_cost_s", 0.0048),
        ("adaptive_ev_extra_verify_cost_s", 0.006),
        ("adaptive_ev_baseline_tok_s", 40.0),
        ("adaptive_ev_safety_margin", 0.10),
        ("adaptive_ev_margin_center", 1.0),
        ("adaptive_ev_margin_scale", 2.0),
        ("adaptive_ev_confidence_weight", 0.35),
        ("adaptive_ev_min_extra_accept_probability", 0.18),
        ("adaptive_ev_warmup_full_depth_cycles", 4),
        ("adaptive_ev_exploration_interval", 32),
    ):
        setattr(target, attr, getattr(source, attr, default))
    return target


def _apply_opencode_fair_defaults(args: Any) -> None:
    """Make ``mtplx start opencode`` match the app's measured OpenCode lane.

    Every key is skipped when the user passed the matching flag, so explicit
    ar_batch/agent experiments keep working. (The old early-return for
    explicit ``--scheduler-mode serial`` predates serial being the default;
    per-key skipping now covers that case and those users additionally get
    the SSD/prefill defaults they were silently missing.)
    """

    cli_flags = getattr(args, "_cli_flags", set()) or set()
    for attr, value in OPENCODE_FAIR_BATCHING_DEFAULTS.items():
        flag = attr.replace("_", "-")
        if flag in cli_flags:
            continue
        setattr(args, attr, value)


def _apply_hermes_latency_defaults(args: Any) -> None:
    """Make ``mtplx start hermes`` match the native app's foreground agent lane."""

    cli_flags = getattr(args, "_cli_flags", set()) or set()
    for attr, value in HERMES_LATENCY_DEFAULTS.items():
        flag = attr.replace("_", "-")
        if flag in cli_flags:
            continue
        setattr(args, attr, value)


def _apply_opencode_memory_env_defaults(env: dict[str, str]) -> None:
    for key, value in _opencode_memory_env_defaults().items():
        env.setdefault(key, value)


def _apply_hermes_memory_env_defaults(env: dict[str, str]) -> None:
    total_ram = _detect_total_ram_bytes_for_opencode_defaults()
    high_memory = (
        total_ram is not None and total_ram >= _OPENCODE_HIGH_MEMORY_THRESHOLD_BYTES
    )
    # Route only; thresholds defer to engine defaults (65536/4/5) — see the
    # OpenCode lane note and issue #228.
    env.setdefault("MTPLX_VLLM_METAL_PAGED_GQA_SDPA_ROUTE", "async_per_head")
    env.setdefault("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", "1")
    env.setdefault(
        "MTPLX_SESSION_BANK_MAX_ENTRIES",
        _OPENCODE_HIGH_MEMORY_MAX_ENTRIES
        if high_memory
        else _OPENCODE_DEFAULT_MAX_ENTRIES,
    )
    # Model-aware auto budget (see _opencode_memory_env_defaults).
    env.setdefault("MTPLX_SESSION_BANK_MAX_BYTES", "auto")
    env.setdefault("MTPLX_SESSION_BANK_PER_SESSION_BYTES", "auto")
    env.setdefault("MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S", "30.0")
    env.setdefault("MTPLX_DYNAMIC_PAGED_KV_MAX_INITIAL_NEW_TOKENS", "4096")
    env.setdefault("MTPLX_LAZY_BONUS_VERIFY", "1")
    env.setdefault("MTPLX_OPENCODE_TOOL_HISTORY_LIVE_FRONTIER", "1")
    env.setdefault("MTPLX_SESSION_LIVE_FRONTIER_REFERENCE_RESTORE", "1")
    env.setdefault("MTPLX_ACTIVE_READ_INSPECTION_TOTAL_MAX_LINES", "72")
    env.setdefault("MTPLX_ACTIVE_READ_INSPECTION_MIN_LINES_PER_FILE", "8")
    env.setdefault("MTPLX_ACTIVE_READ_INSPECTION_MULTI_FILE_LINE_MAX_CHARS", "120")
    env.setdefault("MTPLX_READ_ONLY_INSPECTION_FORCE_ANSWER_AFTER_TOOLS", "12")
    env.setdefault("MTPLX_TOOL_PROMPT_MODE", "hybrid")
    env.setdefault(
        "MTPLX_CHAT_TEMPLATE_PROFILE", OPENCODE_CHAT_TEMPLATE_PROFILE_DEFAULT
    )
    env.setdefault("MTPLX_CLIENT", "hermes")


def _quickstart_run_openwebui(
    args: Any, *, runtime_model: str, inspection: dict[str, Any]
) -> int:
    model_id = _quickstart_served_model_id(args, runtime_model)
    _quickstart_print_openwebui_handoff(args, runtime_model=runtime_model)
    open_dashboard = bool(getattr(args, "open_dashboard", False))
    serve_args = SimpleNamespace(
        model=runtime_model,
        cache_dir=getattr(args, "cache_dir", None),
        profile=getattr(args, "profile", None) or DEFAULT_PROFILE_NAME,
        model_id=model_id,
        unsafe_force_unverified=bool(getattr(args, "unsafe_force_unverified", False)),
        yes=True,
        host=str(getattr(args, "host", "127.0.0.1")),
        port=int(getattr(args, "port", 8000)),
        api_key=getattr(args, "api_key", None),
        depth=int(getattr(args, "depth", 3)),
        no_mtp=bool(getattr(args, "no_mtp", False)),
        rate_limit=int(getattr(args, "rate_limit", 0)),
        stream_interval=int(getattr(args, "stream_interval", 1)),
        warmup_tokens=int(getattr(args, "warmup_tokens", 16)),
        max_response_tokens=getattr(args, "max_response_tokens", None),
        temperature=float(getattr(args, "temperature", 0.6)),
        top_p=float(getattr(args, "top_p", 0.95)),
        top_k=int(getattr(args, "top_k", 20)),
        draft_temperature=getattr(args, "draft_temperature", None),
        draft_top_p=getattr(args, "draft_top_p", None),
        draft_top_k=getattr(args, "draft_top_k", None),
        reasoning=getattr(args, "reasoning", None),
        preserve_thinking=_preserve_thinking_policy(args),
        reasoning_parser=getattr(args, "reasoning_parser", "qwen3"),
        reasoning_effort=getattr(args, "reasoning_effort", None),
        stats_footer=False,
        strict_warmup=bool(getattr(args, "strict_warmup", False)),
        quickstart_openwebui=True,
        open_browser=True,
        open_dashboard=open_dashboard,
        enable_thermal_poll=bool(getattr(args, "enable_thermal_poll", False)),
        fan_mode=getattr(args, "fan_mode", "default"),
        max=bool(getattr(args, "max", False)),
        max_idle_min=int(getattr(args, "max_idle_min", 15)),
    )
    return cmd_serve_public(_with_server_policy_args(serve_args, args))


def _quickstart_run_pi(
    args: Any, *, runtime_model: str, inspection: dict[str, Any]
) -> int:
    model_id = _quickstart_served_model_id(args, runtime_model)
    if not getattr(args, "api_key", None):
        from mtplx.pi import PI_LOCAL_API_KEY

        args.api_key = PI_LOCAL_API_KEY
    pi = _quickstart_pi_payload(args, write_config=True, inspection=inspection)
    _quickstart_print_pi_handoff(args, runtime_model=runtime_model, pi=pi)
    pi_temperature = _pi_sampler_temperature(args)
    pi_top_p = _pi_sampler_top_p(args)
    pi_top_k = _pi_sampler_top_k(args)
    serve_args = SimpleNamespace(
        model=runtime_model,
        cache_dir=getattr(args, "cache_dir", None),
        profile=getattr(args, "profile", None) or DEFAULT_PROFILE_NAME,
        model_id=model_id,
        unsafe_force_unverified=bool(getattr(args, "unsafe_force_unverified", False)),
        yes=True,
        host=str(getattr(args, "host", "127.0.0.1")),
        port=int(getattr(args, "port", 8000)),
        api_key=getattr(args, "api_key", None),
        depth=int(getattr(args, "depth", 3)),
        no_mtp=bool(getattr(args, "no_mtp", False)),
        rate_limit=int(getattr(args, "rate_limit", 0)),
        stream_interval=int(getattr(args, "stream_interval", 1)),
        warmup_tokens=int(getattr(args, "warmup_tokens", 16)),
        max_response_tokens=getattr(args, "max_response_tokens", None),
        temperature=pi_temperature,
        top_p=pi_top_p,
        top_k=pi_top_k,
        draft_temperature=getattr(args, "draft_temperature", None),
        draft_top_p=getattr(args, "draft_top_p", None),
        draft_top_k=getattr(args, "draft_top_k", None),
        reasoning=getattr(args, "reasoning", None),
        preserve_thinking=_preserve_thinking_policy(args),
        reasoning_parser=getattr(args, "reasoning_parser", "qwen3"),
        reasoning_effort=getattr(args, "reasoning_effort", None),
        stats_footer=False,
        strict_warmup=bool(getattr(args, "strict_warmup", False)),
        quickstart_openwebui=False,
        quickstart_pi=True,
        open_browser=False,
        open_dashboard=bool(getattr(args, "open_dashboard", False)),
        enable_thermal_poll=bool(getattr(args, "enable_thermal_poll", False)),
        fan_mode=getattr(args, "fan_mode", "default"),
        max=bool(getattr(args, "max", False)),
        max_idle_min=int(getattr(args, "max_idle_min", 15)),
    )
    return cmd_serve_public(_with_server_policy_args(serve_args, args))


def _quickstart_run_opencode(
    args: Any, *, runtime_model: str, inspection: dict[str, Any]
) -> int:
    model_id = _quickstart_served_model_id(args, runtime_model)
    opencode_tool_prompt_mode = _inspection_tool_prompt_mode(args, inspection)
    opencode_chat_template_profile = str(
        getattr(args, "chat_template_profile", OPENCODE_CHAT_TEMPLATE_PROFILE_DEFAULT)
        or OPENCODE_CHAT_TEMPLATE_PROFILE_DEFAULT
    )
    opencode_max_response_tokens = getattr(args, "max_response_tokens", None)
    opencode = _quickstart_opencode_payload(
        args,
        write_config=True,
        inspection=inspection,
    )
    _quickstart_print_opencode_handoff(
        args,
        runtime_model=runtime_model,
        opencode=opencode,
    )
    serve_args = SimpleNamespace(
        model=runtime_model,
        cache_dir=getattr(args, "cache_dir", None),
        profile=getattr(args, "profile", None) or DEFAULT_PROFILE_NAME,
        model_id=model_id,
        unsafe_force_unverified=bool(getattr(args, "unsafe_force_unverified", False)),
        yes=True,
        host=str(getattr(args, "host", "127.0.0.1")),
        port=int(getattr(args, "port", 8000)),
        api_key=getattr(args, "api_key", None),
        depth=int(getattr(args, "depth", 3)),
        no_mtp=bool(getattr(args, "no_mtp", False)),
        rate_limit=int(getattr(args, "rate_limit", 0)),
        stream_interval=int(getattr(args, "stream_interval", 1)),
        warmup_tokens=int(getattr(args, "warmup_tokens", 16)),
        max_response_tokens=opencode_max_response_tokens,
        temperature=float(getattr(args, "temperature", 0.6)),
        top_p=float(getattr(args, "top_p", 0.95)),
        top_k=int(getattr(args, "top_k", 20)),
        draft_temperature=getattr(args, "draft_temperature", None),
        draft_top_p=getattr(args, "draft_top_p", None),
        draft_top_k=getattr(args, "draft_top_k", None),
        reasoning=getattr(args, "reasoning", None),
        preserve_thinking=_preserve_thinking_policy(args),
        reasoning_parser=getattr(args, "reasoning_parser", "qwen3"),
        reasoning_effort=getattr(args, "reasoning_effort", None),
        tool_prompt_mode=opencode_tool_prompt_mode,
        chat_template_profile=opencode_chat_template_profile,
        chat_template_path=getattr(args, "chat_template_path", None),
        stats_footer=False,
        strict_warmup=bool(getattr(args, "strict_warmup", False)),
        quickstart_openwebui=False,
        quickstart_pi=False,
        quickstart_opencode=True,
        open_browser=False,
        open_dashboard=bool(getattr(args, "open_dashboard", False)),
        enable_thermal_poll=bool(getattr(args, "enable_thermal_poll", False)),
        fan_mode=getattr(args, "fan_mode", "default"),
        max=bool(getattr(args, "max", False)),
        max_idle_min=int(getattr(args, "max_idle_min", 15)),
    )
    return cmd_serve_public(_with_server_policy_args(serve_args, args))


def _quickstart_run_swival(
    args: Any, *, runtime_model: str, inspection: dict[str, Any]
) -> int:
    model_id = _quickstart_served_model_id(args, runtime_model)
    swival = _quickstart_swival_payload(args, inspection=inspection)
    _quickstart_print_swival_handoff(
        args,
        runtime_model=runtime_model,
        swival=swival,
    )
    serve_args = SimpleNamespace(
        model=runtime_model,
        cache_dir=getattr(args, "cache_dir", None),
        profile=getattr(args, "profile", None) or DEFAULT_PROFILE_NAME,
        model_id=model_id,
        unsafe_force_unverified=bool(getattr(args, "unsafe_force_unverified", False)),
        yes=True,
        host=str(getattr(args, "host", "127.0.0.1")),
        port=int(getattr(args, "port", 8000)),
        api_key=getattr(args, "api_key", None),
        depth=int(getattr(args, "depth", 3)),
        no_mtp=bool(getattr(args, "no_mtp", False)),
        rate_limit=int(getattr(args, "rate_limit", 0)),
        stream_interval=int(getattr(args, "stream_interval", 1)),
        warmup_tokens=int(getattr(args, "warmup_tokens", 16)),
        max_response_tokens=getattr(args, "max_response_tokens", None),
        temperature=float(getattr(args, "temperature", 0.6)),
        top_p=float(getattr(args, "top_p", 0.95)),
        top_k=int(getattr(args, "top_k", 20)),
        draft_temperature=getattr(args, "draft_temperature", None),
        draft_top_p=getattr(args, "draft_top_p", None),
        draft_top_k=getattr(args, "draft_top_k", None),
        reasoning=getattr(args, "reasoning", None),
        preserve_thinking=_preserve_thinking_policy(args),
        reasoning_parser=getattr(args, "reasoning_parser", "qwen3"),
        reasoning_effort=getattr(args, "reasoning_effort", None),
        stats_footer=False,
        strict_warmup=bool(getattr(args, "strict_warmup", False)),
        quickstart_openwebui=False,
        quickstart_pi=False,
        quickstart_opencode=False,
        quickstart_swival=True,
        open_browser=False,
        open_dashboard=bool(getattr(args, "open_dashboard", False)),
        enable_thermal_poll=bool(getattr(args, "enable_thermal_poll", False)),
        fan_mode=getattr(args, "fan_mode", "default"),
        max=bool(getattr(args, "max", False)),
        max_idle_min=int(getattr(args, "max_idle_min", 15)),
    )
    return cmd_serve_public(_with_server_policy_args(serve_args, args))


def _quickstart_run_hermes(
    args: Any, *, runtime_model: str, inspection: dict[str, Any]
) -> int:
    model_id = _quickstart_served_model_id(args, runtime_model)
    if not getattr(args, "api_key", None):
        args.api_key = HERMES_LOCAL_API_KEY
    hermes = _quickstart_hermes_payload(
        args,
        write_config=True,
        inspection=inspection,
    )
    _quickstart_print_hermes_handoff(
        args,
        runtime_model=runtime_model,
        hermes=hermes,
    )
    serve_args = SimpleNamespace(
        model=runtime_model,
        cache_dir=getattr(args, "cache_dir", None),
        profile=getattr(args, "profile", None) or DEFAULT_PROFILE_NAME,
        model_id=model_id,
        unsafe_force_unverified=bool(getattr(args, "unsafe_force_unverified", False)),
        yes=True,
        host=str(getattr(args, "host", "127.0.0.1")),
        port=int(getattr(args, "port", 8000)),
        api_key=getattr(args, "api_key", None),
        depth=int(getattr(args, "depth", 3)),
        no_mtp=bool(getattr(args, "no_mtp", False)),
        rate_limit=int(getattr(args, "rate_limit", 0)),
        stream_interval=int(getattr(args, "stream_interval", 1)),
        warmup_tokens=int(getattr(args, "warmup_tokens", 16)),
        max_response_tokens=getattr(args, "max_response_tokens", None),
        temperature=float(getattr(args, "temperature", 0.6)),
        top_p=float(getattr(args, "top_p", 1.0)),
        top_k=int(getattr(args, "top_k", 20)),
        draft_temperature=getattr(args, "draft_temperature", None),
        draft_top_p=getattr(args, "draft_top_p", None),
        draft_top_k=getattr(args, "draft_top_k", None),
        reasoning=getattr(args, "reasoning", "auto"),
        preserve_thinking=getattr(args, "preserve_thinking", "auto"),
        reasoning_parser=getattr(args, "reasoning_parser", "qwen3"),
        reasoning_effort=getattr(args, "reasoning_effort", None),
        tool_prompt_mode=getattr(args, "tool_prompt_mode", "hybrid"),
        chat_template_profile=getattr(
            args,
            "chat_template_profile",
            OPENCODE_CHAT_TEMPLATE_PROFILE_DEFAULT,
        ),
        chat_template_path=getattr(args, "chat_template_path", None),
        stats_footer=False,
        strict_warmup=bool(getattr(args, "strict_warmup", False)),
        quickstart_openwebui=False,
        quickstart_pi=False,
        quickstart_opencode=False,
        quickstart_swival=False,
        quickstart_hermes=True,
        hermes_launch_command=hermes.get("terminal_command"),
        open_browser=False,
        open_dashboard=bool(getattr(args, "open_dashboard", False)),
        enable_thermal_poll=bool(getattr(args, "enable_thermal_poll", False)),
        fan_mode=getattr(args, "fan_mode", "default"),
        max=bool(getattr(args, "max", False)),
        max_idle_min=int(getattr(args, "max_idle_min", 15)),
    )
    return cmd_serve_public(_with_server_policy_args(serve_args, args))


def _quickstart_print_dashboard_handoff(args: Any, *, runtime_model: str) -> None:
    from mtplx.ui import pretty_path

    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 8000))
    dashboard_url = _dashboard_url(host, port)
    _quickstart_line("[2/3] Starting local MTPLX server for the live dashboard...")
    _quickstart_line(f"      Loading model: {pretty_path(str(runtime_model))}")
    _quickstart_line(f"      Dashboard URL: {dashboard_url}")
    _quickstart_line(
        "      Keep this terminal open. Drive load from any client "
        "(Web UI, Pi, OpenCode, hippo, OpenAI SDK)."
    )
    _quickstart_line()


def _quickstart_run_dashboard(
    args: Any, *, runtime_model: str, inspection: dict[str, Any]
) -> int:
    model_id = _quickstart_served_model_id(args, runtime_model)
    _quickstart_print_dashboard_handoff(args, runtime_model=runtime_model)
    serve_args = SimpleNamespace(
        model=runtime_model,
        cache_dir=getattr(args, "cache_dir", None),
        profile=getattr(args, "profile", None) or DEFAULT_PROFILE_NAME,
        model_id=model_id,
        unsafe_force_unverified=bool(getattr(args, "unsafe_force_unverified", False)),
        yes=True,
        host=str(getattr(args, "host", "127.0.0.1")),
        port=int(getattr(args, "port", 8000)),
        api_key=getattr(args, "api_key", None),
        depth=int(getattr(args, "depth", 3)),
        no_mtp=bool(getattr(args, "no_mtp", False)),
        rate_limit=int(getattr(args, "rate_limit", 0)),
        stream_interval=int(getattr(args, "stream_interval", 1)),
        warmup_tokens=int(getattr(args, "warmup_tokens", 16)),
        max_response_tokens=getattr(args, "max_response_tokens", None),
        temperature=float(getattr(args, "temperature", 0.6)),
        top_p=float(getattr(args, "top_p", 0.95)),
        top_k=int(getattr(args, "top_k", 20)),
        draft_temperature=getattr(args, "draft_temperature", None),
        draft_top_p=getattr(args, "draft_top_p", None),
        draft_top_k=getattr(args, "draft_top_k", None),
        reasoning=getattr(args, "reasoning", None),
        preserve_thinking=_preserve_thinking_policy(args),
        reasoning_parser=getattr(args, "reasoning_parser", "qwen3"),
        reasoning_effort=getattr(args, "reasoning_effort", None),
        stats_footer=False,
        strict_warmup=bool(getattr(args, "strict_warmup", False)),
        # Bypass the busy-port openwebui short-circuit; the dashboard target
        # always wants a fresh server in the foreground so the user sees
        # logs while driving load from another terminal.
        quickstart_openwebui=False,
        quickstart_pi=False,
        quickstart_opencode=False,
        quickstart_swival=False,
        open_browser=False,
        open_dashboard=True,
        enable_thermal_poll=bool(getattr(args, "enable_thermal_poll", False)),
        fan_mode=getattr(args, "fan_mode", "default"),
        max=bool(getattr(args, "max", False)),
        max_idle_min=int(getattr(args, "max_idle_min", 15)),
    )
    return cmd_serve_public(_with_server_policy_args(serve_args, args))


# Targets that spawn a local server and therefore need a usable port.
_QUICKSTART_SPAWNING_TARGETS = frozenset(
    {"openwebui", "pi", "opencode", "swival", "hermes", "dashboard"}
)


def _quickstart_autoselect_busy_port(
    args: Any,
    *,
    target: str,
    cli_flags: set[str],
) -> None:
    """Auto-bump a default port held by a non-MTPLX app.

    Only fires for spawning targets when the user did not pass ``--port``.
    A healthy MTPLX daemon on the port is left alone: the downstream
    "MTPLX is already running" reuse path attaches to it instead of
    spawning a duplicate.
    """

    if target not in _QUICKSTART_SPAWNING_TARGETS or "port" in cli_flags:
        return
    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 8000))
    try:
        from mtplx.daemon_client import (
            PORT_FOREIGN,
            classify_port_occupant,
            find_free_port,
        )

        occupant = classify_port_occupant(host, port)
        if occupant.kind != PORT_FOREIGN:
            return
        free_port = find_free_port(host, port + 1)
    except Exception:
        return
    if free_port is None:
        return
    _quickstart_line(
        f"Port {port} is in use by another app — using port {free_port} instead."
    )
    args.port = free_port


def _attach_api_key(daemon: Any) -> str | None:
    env_key = str(os.environ.get("MTPLX_API_KEY") or "").strip()
    if env_key:
        return env_key
    if not getattr(daemon, "api_key_required", False):
        return None
    try:
        from mtplx.app_settings import read_app_settings

        settings = read_app_settings()
    except Exception:
        return None
    return settings.api_key if settings is not None else None


def _run_attach_chat_for_args(daemon: Any, args: Any) -> int:
    from mtplx.daemon_client import run_attach_chat

    prompt = getattr(args, "prompt", None)
    return run_attach_chat(
        daemon,
        api_key=_attach_api_key(daemon),
        prompt=str(prompt) if prompt else None,
    )


def _quickstart_attach_to_daemon(info: dict[str, Any], args: Any) -> int:
    """Resolve an onboarding attach request into a live chat session."""

    from mtplx.daemon_client import fetch_daemon_health

    host = str(info.get("host") or "127.0.0.1")
    port = int(info.get("port") or 8000)
    daemon = fetch_daemon_health(host, port)
    if daemon is None:
        _quickstart_line(f"error: the running server on port {port} went away")
        _quickstart_line(f"try: {_start_invocation(args)}")
        return 1
    return _run_attach_chat_for_args(daemon, args)


def _daemon_runs_model(daemon: Any, runtime_model: str) -> bool:
    model_path = getattr(daemon, "model_path", None)
    if model_path and str(model_path) == str(runtime_model):
        return True
    daemon_model = getattr(daemon, "model", None)
    if not daemon_model:
        return False
    try:
        from mtplx.default_models import public_model_id_for_ref

        return public_model_id_for_ref(runtime_model) == str(daemon_model)
    except Exception:
        return False


def _terminal_chat_attach_guard(args: Any, *, runtime_model: str) -> int | None:
    """Never double-load: route terminal chat through a running daemon.

    Returns an exit code when the request was fully handled (attached,
    cancelled, or refused), or ``None`` when in-process loading should
    proceed (no daemon, or the user chose to stop it).
    """

    try:
        from mtplx.daemon_client import detect_attachable_daemon

        daemon = detect_attachable_daemon()
    except Exception:
        return None
    if daemon is None or not getattr(daemon, "model", None):
        return None
    same_model = _daemon_runs_model(daemon, runtime_model)
    owner = "the MTPLX app" if daemon.owned_by_app else "another terminal"
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if not interactive or bool(getattr(args, "yes", False)):
        if same_model or not bool(getattr(args, "_model_explicit", False)):
            _quickstart_line(
                f"MTPLX is already running (model: {daemon.model}, "
                f"port {daemon.port}, started by {owner})."
            )
            _quickstart_line(
                "Attaching to the running server instead of loading a second copy."
            )
            return _run_attach_chat_for_args(daemon, args)
        _quickstart_line(
            f"error: an MTPLX server is already running on port {daemon.port} "
            f"with model {daemon.model}"
        )
        _quickstart_line(
            "Loading a different model alongside it would double memory use."
        )
        _quickstart_line(f"try: mtplx stop --port {daemon.port}")
        _quickstart_line("try: mtplx start cli   # chat with the running model")
        return 2
    _quickstart_line()
    _quickstart_line(
        f"MTPLX is already running (model: {daemon.model}, port {daemon.port}, "
        f"started by {owner})."
    )
    _quickstart_line("Loading another copy here would double memory use.")
    _quickstart_line("  1. Chat with the running model (recommended)")
    _quickstart_line("  2. Stop that server and load here")
    _quickstart_line("  3. Cancel")
    while True:
        try:
            answer = input("  Type 1-3 and press Enter [default 1]: ").strip() or "1"
        except (EOFError, KeyboardInterrupt):
            _quickstart_line()
            return 130
        if answer in {"1", "2", "3"}:
            break
    if answer == "1":
        return _run_attach_chat_for_args(daemon, args)
    if answer == "3":
        _quickstart_line("cancelled")
        return 0
    from mtplx.daemon_client import stop_daemon

    _quickstart_line(f"Stopping the server on port {daemon.port}...")
    result = stop_daemon(daemon.host, daemon.port)
    if not result.get("ok"):
        _quickstart_line(f"error: could not stop the server ({result.get('reason')})")
        return 1
    _quickstart_line("Server stopped.")
    return None


def _quickstart_run_terminal_chat(
    args: Any, *, runtime_model: str, inspection: dict[str, Any]
) -> int:
    guard_exit = _terminal_chat_attach_guard(args, runtime_model=runtime_model)
    if guard_exit is not None:
        return guard_exit
    fan_mode = _fan_mode_from_args(args)
    max_session: Any | None = None
    if fan_mode == FAN_MODE_MAX:
        from mtplx.thermal import MaxSession

        max_session = MaxSession(log=_quickstart_line)
        if not max_session.start():
            verified = max_session.thermal.get("verified") or {}
            _quickstart_line()
            _quickstart_line(
                "[max] fan boost unavailable; terminal chat will continue without fan boost."
            )
            if verified.get("message"):
                _quickstart_line(f"[max] reason: {verified.get('message')}")
            _quickstart_line()
            args.max = False
            max_session = None
    try:
        return _quickstart_run_terminal_chat_body(
            args,
            runtime_model=runtime_model,
            inspection=inspection,
        )
    finally:
        if max_session is not None:
            max_session.stop()


def _quickstart_run_terminal_chat_body(
    args: Any, *, runtime_model: str, inspection: dict[str, Any]
) -> int:
    _apply_model_default_profile(
        args, _public_model_id_for_args(args, str(runtime_model))
    )
    profile = get_profile(_resolved_default_profile_name(args))
    apply_profile_env(profile.name)
    generation_mode = _generation_mode_from_args(args)
    draft_lm_head = (
        _model_draft_lm_head_spec(inspection, profile)
        if getattr(args, "load_mtp", True) is not False
        else None
    )
    draft_sampler = _model_draft_sampler_spec(inspection, profile)

    from mtplx.runtime import load
    from mtplx.ui import ModelLoadProgress, render_banner, render_startup_panel
    from mtplx.version import DISPLAY_VERSION

    if sys.stdout.isatty():
        render_banner()
        render_startup_panel(
            version=DISPLAY_VERSION,
            model=str(runtime_model),
            profile=profile.name,
            profile_summary=_PROFILE_SHORT_SUMMARIES.get(profile.name),
            api_url="terminal chat (no server)",
            mode_label=(
                "AR target-only"
                if generation_mode == GENERATION_MODE_AR
                else _runtime_mode_display(
                    profile.name,
                    max_mode=bool(getattr(args, "max", False)),
                    generation_mode=generation_mode,
                )
            ),
            extra_lines=[
                (
                    "Sampler",
                    (
                        f"temp={float(getattr(args, 'temperature', 0.6)):.2f} "
                        f"top_p={float(getattr(args, 'top_p', 0.95)):.2f} "
                        f"top_k={int(getattr(args, 'top_k', 20))} "
                        f"depth={int(getattr(args, 'depth', 3))}"
                    ),
                ),
                ("Reasoning", _reasoning_mode(args)),
            ],
        )

    # Serve-path memory discipline (#261, F7): terminal chat loads the model
    # in-process with no server; pin the exact Metal allocator caps the serve
    # path applies at startup so long chats cannot balloon past serve limits.
    from mtplx.server.openai import apply_memory_caps_preflight

    apply_memory_caps_preflight(
        entry="quickstart.terminal_chat",
        model=str(runtime_model),
    )

    started = time.perf_counter()
    quiet_progress = not sys.stdout.isatty()
    with ModelLoadProgress("Loading model", quiet=quiet_progress) as progress:
        progress.set_subtitle(f"profile {profile.name}")
        rt = load(runtime_model, mtp=getattr(args, "load_mtp", True) is not False)
        progress.set_subtitle("ready")
    _quickstart_line(f"Model ready in {time.perf_counter() - started:.1f}s")
    _quickstart_line(f"Generation mode: {_generation_mode_label(generation_mode)}")
    draft_report = None
    if draft_lm_head is not None:
        _quickstart_line(
            "[3/4] Installing fast draft head: "
            f"{int(draft_lm_head['bits'])}-bit gs{int(draft_lm_head['group_size'])}"
        )
        draft_started = time.perf_counter()
        from mtplx.draft_lm_head import _install_draft_lm_head

        draft_report = _install_draft_lm_head(
            rt,
            bits=int(draft_lm_head["bits"]),
            group_size=int(draft_lm_head["group_size"]),
            mode=str(draft_lm_head["mode"]),
        )
        _quickstart_line(
            f"      draft head ready in {time.perf_counter() - draft_started:.1f}s"
        )
    _quickstart_line(
        f"Sampler: temp={float(getattr(args, 'temperature', 0.6)):.2f} "
        f"top_p={float(getattr(args, 'top_p', 0.95)):.2f} "
        f"top_k={int(getattr(args, 'top_k', 20))} depth={int(getattr(args, 'depth', 3))}"
    )
    _quickstart_line(f"Reasoning: {_reasoning_mode(args)}")
    if draft_report is not None:
        if generation_mode == GENERATION_MODE_AR:
            _quickstart_line(
                "Draft-only LM head loaded for /mtp on; current AR mode bypasses it."
            )
        else:
            _quickstart_line("Native-MTP speed path: draft-only LM head is active.")
    if draft_sampler is not None and generation_mode == GENERATION_MODE_MTP:
        _quickstart_line(
            "Draft sampler: "
            f"temp={float(draft_sampler['temperature']):.2f} "
            f"top_p={float(draft_sampler['top_p']):.2f} "
            f"top_k={int(draft_sampler['top_k'])}"
        )
    _quickstart_line()

    history: list[dict[str, str]] = []
    last_payload: dict[str, Any] | None = None

    def run_turn(
        prompt: str,
        turn_index: int,
        *,
        max_tokens: int | None = None,
        include_history: bool = True,
        record_history: bool = True,
        quality_gate: bool = True,
        response_label: str = "MTPLX",
    ) -> int:
        nonlocal last_payload
        _quickstart_line("[4/4] generating response...")
        active_mode = _generation_mode_from_args(args)
        turn_label = (
            f"MTPLX {_generation_mode_label(active_mode)}"
            if response_label == "MTPLX"
            else response_label
        )
        payload = _quickstart_generate(
            rt=rt,
            inspection=inspection,
            profile=profile,
            args=args,
            prompt=prompt,
            history=history,
            turn_index=turn_index,
            max_tokens=max_tokens,
            include_history=include_history,
            stream_label=turn_label,
            draft_sampler=draft_sampler,
        )
        last_payload = payload
        text = str(payload["text"])
        if not payload.get("streamed"):
            _print_assistant_fallback(turn_label, text)
        if bool(getattr(args, "show_stats", True)):
            _quickstart_line()
            _print_stats_line(_quickstart_stats_line(payload))
        if record_history:
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": text})
        failures = [row for row in payload["validations"] if not row.get("passed")]
        if failures and quality_gate:
            _quickstart_line(
                "[mtplx] quality warning: response failed a basic output validator"
            )
            return EXIT_QUALITY
        return 0

    first_prompt = getattr(args, "prompt", None)
    if first_prompt:
        return run_turn(str(first_prompt), 0)

    if not sys.stdin.isatty():
        _quickstart_line("error: no interactive terminal detected")
        prompt_hint = _start_invocation(args, ' --prompt "Say hi"')
        _quickstart_line(f"try: {prompt_hint}")
        return 2

    _quickstart_line(
        "Chat is ready. Type /mtp on|off|status, /stats, /speed, /reasoning on|off|auto, or /exit."
    )
    turn_index = 0
    worst_code = 0
    while True:
        try:
            prompt = input(_chat_input_prompt()).strip()
        except (EOFError, KeyboardInterrupt):
            _quickstart_line()
            _quickstart_line("bye")
            return worst_code
        if not prompt:
            continue
        if prompt.lower() in {"/exit", "exit", "quit", "/quit"}:
            _quickstart_line("bye")
            return worst_code
        if _handle_quickstart_reasoning_command(args, prompt):
            continue
        if _handle_quickstart_mtp_command(args, prompt, runtime=rt):
            continue
        if prompt.lower() in {"/stats", "stats"}:
            if last_payload is None:
                _quickstart_line("No stats yet.")
            else:
                _print_stats_line(_quickstart_stats_line(last_payload))
            continue
        if prompt.lower() in {"/speed", "speed", "/bench", "/benchmark"}:
            _quickstart_line(
                f"Running a {QUICKSTART_SPEED_MAX_TOKENS}-token speed sample without chat history."
            )
            code = run_turn(
                QUICKSTART_SPEED_PROMPT,
                turn_index,
                max_tokens=QUICKSTART_SPEED_MAX_TOKENS,
                include_history=False,
                record_history=False,
                quality_gate=False,
                response_label="MTPLX speed sample",
            )
            worst_code = max(worst_code, code)
            turn_index += 1
            continue
        code = run_turn(prompt, turn_index)
        worst_code = max(worst_code, code)
        turn_index += 1


def _quickstart_apply_tuned_depth(
    args: Any,
    *,
    runtime_model: str,
    target: str,
    can_prompt: bool,
) -> None:
    if target not in {"openwebui", "terminal"}:
        return
    if bool(getattr(args, "_explicit_depth", False)):
        return
    tune_args = SimpleNamespace(
        command="tune",
        model=runtime_model,
        # The wizard's pick IS an explicit model selection. Without this
        # marker _tune_requested_model treats the namespace model as
        # unrequested and re-resolves the hardware default, so picking the
        # 4B tuned (and keyed) the 27B (2026-08-17 wizard repro).
        _cli_flags={"model"},
        cache_dir=getattr(args, "cache_dir", None),
        profile=getattr(args, "profile", None),
        depths=TUNE_DEFAULT_DEPTHS,
        max_tokens=TUNE_DEFAULT_MAX_TOKENS,
        limit=TUNE_DEFAULT_LIMIT,
        seed=TUNE_DEFAULT_SEED,
        run_id=None,
        output_dir=None,
        output=None,
        json=False,
        verbose=False,
        dry_run=False,
        no_save=False,
        retune=False,
        unsafe_force_unverified=bool(getattr(args, "unsafe_force_unverified", False)),
        yes=True,
    )
    context = _tune_state_context_for_args(tune_args)
    if context is None:
        return
    _hardware, _software, _backend, state_key, _key_material = context
    record = _load_tune_record(state_key)
    if record is not None:
        payload = record.get("payload") or {}
        best = payload.get("best") or {}
        depth = best.get("depth")
        if isinstance(depth, int):
            args.depth = depth
            _quickstart_line(
                f"tuned depth: D{depth} "
                f"({_fmt_metric(best.get('multiplier_vs_ar'), digits=2)}x AR)"
            )
            return
    if not can_prompt:
        return
    try:
        from mtplx.ui.onboarding import screen_tuning_offer

        should_tune = screen_tuning_offer()
    except (KeyboardInterrupt, EOFError):
        _quickstart_line()
        _quickstart_line("tuning skipped")
        return
    if not should_tune:
        _quickstart_line("tuning skipped; using default depth")
        return
    code = _cmd_tune(
        tune_args,
        action="tune",
        save_default=True,
        verbose_default=False,
    )
    if code != 0:
        _quickstart_line("tuning failed; using default depth")
        return
    record = _load_tune_record(state_key)
    payload = record.get("payload") if isinstance(record, dict) else None
    best = payload.get("best") if isinstance(payload, dict) else None
    depth = best.get("depth") if isinstance(best, dict) else None
    if isinstance(depth, int):
        args.depth = depth


def cmd_quickstart_public(args: Any) -> int:
    raw_target = getattr(args, "target", None)
    runtime_options_error = _resolve_runtime_options_on_args(
        args,
        printer=_quickstart_line,
    )
    if runtime_options_error is not None:
        return runtime_options_error
    try:
        fan_mode = _fan_mode_from_args(args)
    except ValueError as exc:
        _quickstart_line(f"error: {exc}")
        return 2

    # Start is the most common reason fans get blasted, so this is the
    # right place to scrub a stale --max marker left behind by a previously
    # killed session. No-op when the previous run exited cleanly.
    try:
        from mtplx.thermal import check_and_recover_stale_max

        recovery = check_and_recover_stale_max()
        if recovery.get("recovered"):
            sys.stderr.write(
                "[max] restored fans from a previous --max session that "
                f"did not exit cleanly (stale pid {recovery.get('stale_pid')})\n"
            )
    except Exception:
        pass  # best-effort — never block start on cleanup

    # Onboarding flow: only when interactive, no explicit CLI overrides, and not
    # in dry-run / one-shot prompt / non-interactive automation. We detect
    # explicit CLI flags by scanning the raw argv tokens (stashed by main()),
    # because parser defaults and ``apply_user_config`` both overwrite the
    # parsed Namespace and would otherwise mask the user's actual intent.
    cli_flags = getattr(args, "_cli_flags", set()) or set()
    is_tty = sys.stdin.isatty() and sys.stdout.isatty()
    has_explicit_target = raw_target is not None
    has_explicit_model = "model" in cli_flags
    model_from_model_id = _apply_model_id_as_model_default(
        args,
        has_explicit_model=has_explicit_model,
    )
    args._model_explicit = has_explicit_model or model_from_model_id
    has_explicit_profile_flag = "profile" in cli_flags
    has_explicit_depth = "depth" in cli_flags
    args._explicit_depth = has_explicit_depth
    has_explicit_max = "max" in cli_flags
    has_explicit_fan_mode = "fan-mode" in cli_flags
    has_prompt = bool(getattr(args, "prompt", None))
    is_dry_run = bool(getattr(args, "dry_run", False))
    is_yes = bool(getattr(args, "yes", False))
    fresh = bool(getattr(args, "fresh", False))

    skip_onboarding = (
        is_dry_run
        or has_prompt
        or is_yes
        or has_explicit_target
        or has_explicit_model
        or model_from_model_id
        or has_explicit_profile_flag
        or has_explicit_max
        or has_explicit_fan_mode
        or not is_tty
    )

    if not skip_onboarding:
        from mtplx.ui.onboarding import PROFILE_AUTO, run_quickstart_flow

        configured_model = getattr(args, "model", None)
        # `--open-dashboard` / `--no-open-dashboard` on the CLI is the
        # user explicitly skipping the wizard's companion prompt. None
        # means "ask".
        explicit_open_dashboard = getattr(args, "open_dashboard", None)
        choice = run_quickstart_flow(
            fresh=fresh,
            configured_model=configured_model,
            open_dashboard_override=explicit_open_dashboard,
            host=str(getattr(args, "host", "127.0.0.1")),
            port=int(getattr(args, "port", 8000)),
        )
        if choice is None:
            _quickstart_line("aborted")
            return 130
        attach_request = choice.get("attach")
        if isinstance(attach_request, dict):
            return _quickstart_attach_to_daemon(attach_request, args)
        chosen_model = choice.get("model")
        if chosen_model:
            args.model = chosen_model
            # A wizard pick is an explicit user decision, exactly like the
            # profile stamp below. Both explicitness markers must be set:
            # without _model_explicit, _quickstart_current_model re-routes a
            # default-NAMED pick (e.g. an LM Studio folder whose basename
            # matches the verified default) back through
            # select_default_model(), replacing the picked path with the
            # canonical repo id — "Model is missing. Download?" for a model
            # that is on disk (issue #279). Without "model" in _cli_flags,
            # _tune_requested_model re-resolves the hardware default and
            # tunes a different model than the one being launched
            # (2026-08-17 wizard repro: picked 4B, tuned 27B).
            args._model_explicit = True
            args._cli_flags = set(getattr(args, "_cli_flags", set()) or set())
            args._cli_flags.add("model")
            # Auto-pull policy: the user has explicitly picked this model in
            # the onboarding wizard; if it isn't on disk we fetch it without
            # re-prompting. The legacy "Model is missing. Download? [Y/n]"
            # fallback below is reserved for non-onboarded shortcuts.
            try:
                from mtplx.hf_loader import repo_id_from_model_ref

                if repo_id_from_model_ref(chosen_model):
                    args.download = True
            except Exception:
                # Best-effort: never let an import problem break the wizard.
                pass
        chosen_profile = choice.get("profile")
        if chosen_profile and chosen_profile != PROFILE_AUTO:
            args.profile = chosen_profile
            # An explicit wizard pick is a user decision: record it so
            # per-model default-profile resolution never overrides it. Auto
            # is the opposite decision — stamp nothing, so the engine keeps
            # resolving the launch profile per model.
            args._cli_flags = set(getattr(args, "_cli_flags", set()) or set())
            args._cli_flags.add("profile")
        if choice.get("max"):
            args.max = True
            args.fan_mode = FAN_MODE_MAX
        else:
            # If onboarding declined a fan-backed mode (e.g. ThermalForge
            # install was refused or failed), make sure --max from a stale
            # config or shell alias does not silently re-enable it.
            args.max = False
            args.fan_mode = "default"
        chosen_target = choice.get("target")
        if chosen_target:
            raw_target = chosen_target
        # Dashboard companion: wizard may say "also open the dashboard"
        # alongside an openwebui / pi / opencode / swival target.
        # Stash on args so the per-target _quickstart_run_* helpers below
        # propagate it into serve_args.
        if choice.get("open_dashboard"):
            args.open_dashboard = True
        elif "open_dashboard" in choice:
            # Explicit "No" from wizard; squash any stale CLI alias.
            args.open_dashboard = False
        # The new onboarding has already collected the user's choices, so the
        # legacy ``_quickstart_choose_model`` picker must not prompt again.
        args._onboarded = True
    else:
        # Skipped onboarding (explicit flags). If the user passed --max but
        # has no fan controller, offer to auto-install before MTPLX boots
        # rather than silently dumping the JSON warning later.
        fan_mode = _fan_mode_from_args(args)
        if (
            (has_explicit_max or has_explicit_fan_mode)
            and fan_mode == FAN_MODE_MAX
            and is_tty
        ):
            from mtplx.thermal import detect_thermal_control

            detection = detect_thermal_control()
            if not detection.get("available"):
                from mtplx.ui.onboarding import ensure_thermal_control_installed

                if not ensure_thermal_control_installed():
                    args.max = False
                    args.fan_mode = "default"

    if raw_target is None:
        raw_target = "cli" if has_prompt else "web"
    raw_target = str(raw_target).lower()
    if raw_target in {"open-webui", "openwebui", "web"}:
        target = "openwebui"
    elif raw_target in {"cli", "terminal"}:
        target = "terminal"
    elif raw_target in {"pi", "pie"}:
        target = "pi"
    elif raw_target in {"opencode", "open-code", "oc"}:
        target = "opencode"
    elif raw_target in {"swival", "sv"}:
        target = "swival"
    elif raw_target in {"hermes", "hermes-agent"}:
        target = "hermes"
    elif raw_target in {"dashboard", "live-dashboard", "live"}:
        target = "dashboard"
    else:
        target = raw_target
    if target not in QUICKSTART_TARGETS:
        _quickstart_line(f"error: unknown start target: {raw_target}")
        _quickstart_line(f"try: {_start_invocation(args)}")
        _quickstart_line(f"try: {_start_invocation(args, ' cli')}")
        return 2
    if target == "opencode" and "port" not in cli_flags:
        args.port = 18083
    if target == "opencode":
        _apply_opencode_fair_defaults(args)
    if target == "swival" and "port" not in cli_flags:
        args.port = 18084
    if target == "hermes" and "port" not in cli_flags:
        args.port = 18085
    if target == "hermes":
        _apply_hermes_latency_defaults(args)
    if not getattr(args, "dry_run", False):
        _quickstart_autoselect_busy_port(args, target=target, cli_flags=cli_flags)
    depth_error = _validate_public_depth(args, printer=_quickstart_line)
    if depth_error is not None:
        return depth_error
    model, download = _quickstart_choose_model(args, target=target)
    default_selection = getattr(args, "_mtplx_default_model_selection", None)
    if isinstance(default_selection, dict) and model != default_selection.get("model"):
        default_selection = None
    # Downstream integration payloads build copy-pasteable server commands from
    # args.model. When the user accepts the verified default, argparse leaves
    # args.model unset, so keep the resolved model on the namespace as soon as
    # the selection is known.
    args.model = model
    cache_dir = getattr(args, "cache_dir", None)
    if getattr(args, "dry_run", False):
        dry_run_inspection = _quickstart_apply_local_model_defaults(
            args,
            model=model,
        )
        openwebui = (
            _quickstart_openwebui_payload(args, inspection=dry_run_inspection)
            if target == "openwebui"
            else None
        )
        pi = (
            _quickstart_pi_payload(args, inspection=dry_run_inspection)
            if target == "pi"
            else None
        )
        opencode = (
            _quickstart_opencode_payload(args, inspection=dry_run_inspection)
            if target == "opencode"
            else None
        )
        swival = (
            _quickstart_swival_payload(args, inspection=dry_run_inspection)
            if target == "swival"
            else None
        )
        hermes = (
            _quickstart_hermes_payload(args, inspection=dry_run_inspection)
            if target == "hermes"
            else None
        )
        payload = {
            "action": _start_command_name(args),
            "target": target,
            "model": model,
            "cache_dir": cache_dir,
            # Display the profile the launch will actually resolve (per-model
            # turbo rewrite included) — the raw parser default here made the
            # dry-run advertise "sustained" for the turbo-default flagships
            # (the 2026-07-16 stale-display bug class, on one more surface).
            "profile": _resolved_default_profile_name(args, model),
            "generation_mode": _generation_mode_from_args(args),
            "max": bool(getattr(args, "max", False)),
            "download_if_missing": download,
            "default_model_selection": default_selection,
            "terminal_chat": target == "terminal",
            "openwebui": openwebui,
            "pi": pi,
            "opencode": opencode,
            "swival": swival,
            "hermes": hermes,
            # The dashboard target *always* opens the dashboard (that's what
            # it's for). Otherwise honor the explicit user/wizard choice,
            # but only for server-spawning targets — terminal/CLI has no
            # server so there's nothing to attach to.
            "open_dashboard": (
                True
                if target == "dashboard"
                else bool(getattr(args, "open_dashboard", False))
                and target in {"openwebui", "pi", "opencode", "swival", "hermes"}
            ),
            "enable_thermal_poll": bool(getattr(args, "enable_thermal_poll", False)),
            "stats_visible": bool(getattr(args, "show_stats", True)),
            "next": (
                _start_invocation(args)
                if target == "openwebui"
                else _start_invocation(args, " swival")
                if target == "swival"
                else _start_invocation(args, " opencode")
                if target == "opencode"
                else _start_invocation(args, " pi")
                if target == "pi"
                else _start_invocation(args, " hermes")
                if target == "hermes"
                else _start_invocation(args, " cli")
            ),
        }
        if getattr(args, "json", False):
            _print(payload)
        else:
            _quickstart_line(f"MTPLX {_start_command_name(args)}")
            _quickstart_line(f"model: {model}")
            if isinstance(default_selection, dict):
                _quickstart_line(
                    "verified default for this Mac: "
                    f"{default_selection.get('display_name')} "
                    f"({default_selection.get('reason')})"
                )
            _quickstart_line(f"profile: {payload['profile']}")
            _quickstart_line(
                "mode: "
                + _runtime_mode_display(
                    str(payload["profile"]),
                    max_mode=bool(payload["max"]),
                    generation_mode=str(payload["generation_mode"]),
                )
            )
            _quickstart_line(
                f"generation: {_generation_mode_label(payload['generation_mode'])}"
            )
            _quickstart_line(f"download if missing: {str(download).lower()}")
            if target == "openwebui":
                _quickstart_line(
                    f"then: start local server -> open browser chat at {openwebui['chat_url']}"
                )
            elif target == "opencode":
                _quickstart_line(
                    f"then: write OpenCode config -> start local server -> open {opencode['model_ref']}"
                )
            elif target == "swival":
                _quickstart_line(
                    f"then: start local server -> run {swival['launch_command']}"
                )
            elif target == "pi":
                _quickstart_line(
                    f"then: write Pi config -> start local server -> run {pi['launch_command']}"
                )
            elif target == "hermes":
                _quickstart_line(
                    f"then: write Hermes profile -> start local server -> run {hermes['launch_command']}"
                )
            else:
                _quickstart_line(
                    "then: load once -> chat in this terminal -> stream output -> show speed stats"
                )
        return 0

    if target == "pi" and not _quickstart_require_pi_cli(args):
        return 2

    _quickstart_line(f"MTPLX {_start_command_name(args)}")
    if isinstance(default_selection, dict):
        _quickstart_line(
            "Verified default for this Mac: "
            f"{default_selection.get('display_name')} "
            f"({default_selection.get('reason')})"
        )
    _quickstart_line(f"[1/4] Checking model: {model}")
    try:
        runtime_model, resolution = _quickstart_resolve_model(
            model, cache_dir=cache_dir, download=download
        )
    except KeyboardInterrupt:
        _quickstart_line("download cancelled")
        return 130
    except Exception as exc:
        _quickstart_line(f"error: {exc}")
        return 1
    if runtime_model is None:
        if resolution.get("cancelled"):
            _quickstart_line("download cancelled")
            return 130
        gate_inspection = resolution.get("gate_inspection")
        if isinstance(gate_inspection, dict):
            _print_model_gate_error(
                gate_inspection,
                printer=_quickstart_line,
                json_output=bool(getattr(args, "json", False)),
            )
            compatibility = gate_inspection.get("compatibility") or {}
            return int(compatibility.get("exit_code") or 1)
        resolution_error = resolution.get("error")
        if isinstance(resolution_error, dict) and resolution_error.get("error") not in {
            None,
            "model is not available locally",
        }:
            _print_command_error(
                resolution_error,
                command="start",
                json_output=bool(getattr(args, "json", False)),
            )
            return 1
        if sys.stdin.isatty() and not getattr(args, "prompt", None):
            try:
                download_model = _quickstart_download_ref(model)
                label = (
                    "selected model" if download_model == model else "verified default"
                )
            except ValueError:
                download_model = select_default_model().hf_model
                label = "verified default"
            answer = (
                input(
                    f"Model is missing. Download the {label} ({download_model}) now? [Y/n] "
                )
                .strip()
                .lower()
            )
            if answer in {"", "y", "yes"}:
                try:
                    runtime_model, resolution = _quickstart_resolve_model(
                        download_model, cache_dir=cache_dir, download=True
                    )
                except KeyboardInterrupt:
                    _quickstart_line("download cancelled")
                    return 130
                except Exception as exc:
                    _quickstart_line(f"error: {exc}")
                    return 1
        if runtime_model is None and resolution.get("cancelled"):
            _quickstart_line("download cancelled")
            return 130
        gate_inspection = resolution.get("gate_inspection")
        if runtime_model is None and isinstance(gate_inspection, dict):
            _print_model_gate_error(
                gate_inspection,
                printer=_quickstart_line,
                json_output=bool(getattr(args, "json", False)),
            )
            compatibility = gate_inspection.get("compatibility") or {}
            return int(compatibility.get("exit_code") or 1)
        resolution_error = resolution.get("error")
        if (
            runtime_model is None
            and isinstance(resolution_error, dict)
            and resolution_error.get("error")
            not in {None, "model is not available locally"}
        ):
            _print_command_error(
                resolution_error,
                command="start",
                json_output=bool(getattr(args, "json", False)),
            )
            return 1
        if runtime_model is not None:
            _quickstart_line(f"model ready: {runtime_model}")
            if resolution.get("downloaded"):
                _quickstart_line(f"downloaded: {resolution.get('download_ref')}")
            inspection, gate_exit = _model_gate(
                runtime_model,
                unsafe_force_unverified=bool(
                    getattr(args, "unsafe_force_unverified", False)
                ),
                yes=bool(getattr(args, "yes", False)),
            )
            if gate_exit is not None:
                _print_model_gate_error(
                    inspection,
                    printer=_quickstart_line,
                    json_output=bool(getattr(args, "json", False)),
                )
                return gate_exit
            mode_exit = _apply_runtime_compatibility_mode(
                args,
                inspection,
                printer=_quickstart_line,
            )
            if mode_exit is not None:
                return mode_exit
            profile = get_profile(_resolved_default_profile_name(args))
            _apply_model_contract_depth_default(args, inspection, profile)
            _apply_backend_serve_defaults(args, inspection)
            _quickstart_apply_tuned_depth(
                args,
                runtime_model=runtime_model,
                target=target,
                can_prompt=bool(
                    getattr(args, "_onboarded", False) and is_tty and not is_yes
                ),
            )
            if target == "openwebui":
                args.model = runtime_model
                return _quickstart_run_openwebui(
                    args, runtime_model=runtime_model, inspection=inspection
                )
            if target == "pi":
                args.model = runtime_model
                return _quickstart_run_pi(
                    args, runtime_model=runtime_model, inspection=inspection
                )
            if target == "opencode":
                args.model = runtime_model
                return _quickstart_run_opencode(
                    args, runtime_model=runtime_model, inspection=inspection
                )
            if target == "swival":
                args.model = runtime_model
                return _quickstart_run_swival(
                    args, runtime_model=runtime_model, inspection=inspection
                )
            if target == "hermes":
                args.model = runtime_model
                return _quickstart_run_hermes(
                    args, runtime_model=runtime_model, inspection=inspection
                )
            if target == "dashboard":
                args.model = runtime_model
                return _quickstart_run_dashboard(
                    args, runtime_model=runtime_model, inspection=inspection
                )
            return _quickstart_run_terminal_chat(
                args, runtime_model=runtime_model, inspection=inspection
            )
        detail = (
            (resolution.get("error") or {}).get("detail")
            if isinstance(resolution.get("error"), dict)
            else None
        )
        _quickstart_line("model is not available locally")
        if detail:
            _quickstart_line(f"detail: {detail}")
        _quickstart_line(f"try: {_start_invocation(args, ' --download')}")
        _quickstart_line(
            "try: "
            f"{_start_invocation(args, ' cli' if target == 'terminal' else '')} "
            f"--model {shlex.quote(str(model))} --download"
        )
        _quickstart_line(f"try: {_start_invocation(args, ' --model /path/to/model')}")
        return 1

    inspection, gate_exit = _model_gate(
        runtime_model,
        unsafe_force_unverified=bool(getattr(args, "unsafe_force_unverified", False)),
        yes=bool(getattr(args, "yes", False)),
    )
    if gate_exit is not None:
        _print_model_gate_error(
            inspection,
            printer=_quickstart_line,
            json_output=bool(getattr(args, "json", False)),
        )
        return gate_exit
    mode_exit = _apply_runtime_compatibility_mode(
        args,
        inspection,
        printer=_quickstart_line,
    )
    if mode_exit is not None:
        return mode_exit
    profile = get_profile(_resolved_default_profile_name(args))
    _apply_model_contract_depth_default(args, inspection, profile)
    _apply_backend_serve_defaults(args, inspection)
    _quickstart_apply_tuned_depth(
        args,
        runtime_model=runtime_model,
        target=target,
        can_prompt=bool(getattr(args, "_onboarded", False) and is_tty and not is_yes),
    )
    _quickstart_line(f"model ready: {runtime_model}")
    if resolution.get("downloaded"):
        _quickstart_line(f"downloaded: {resolution.get('download_ref')}")
    if target == "openwebui":
        args.model = runtime_model
        return _quickstart_run_openwebui(
            args, runtime_model=runtime_model, inspection=inspection
        )
    if target == "pi":
        args.model = runtime_model
        return _quickstart_run_pi(
            args, runtime_model=runtime_model, inspection=inspection
        )
    if target == "opencode":
        args.model = runtime_model
        return _quickstart_run_opencode(
            args, runtime_model=runtime_model, inspection=inspection
        )
    if target == "swival":
        args.model = runtime_model
        return _quickstart_run_swival(
            args, runtime_model=runtime_model, inspection=inspection
        )
    if target == "hermes":
        args.model = runtime_model
        return _quickstart_run_hermes(
            args, runtime_model=runtime_model, inspection=inspection
        )
    if target == "dashboard":
        args.model = runtime_model
        return _quickstart_run_dashboard(
            args, runtime_model=runtime_model, inspection=inspection
        )
    return _quickstart_run_terminal_chat(
        args, runtime_model=runtime_model, inspection=inspection
    )


def cmd_metrics_public(args: Any) -> int:
    if args.metrics_action != "watch":
        raise SystemExit(f"unknown metrics action: {args.metrics_action}")
    base = args.url.rstrip("/")
    count = int(getattr(args, "count", 0) or 0)
    interval = float(getattr(args, "interval", 1.0))
    seen = 0
    while True:
        payload = _http_json(
            base + "/metrics", timeout=float(getattr(args, "timeout", 5.0))
        )
        latest = payload.get("latest") if isinstance(payload, dict) else None
        row = {
            "url": base + "/metrics",
            "ok": bool(isinstance(payload, dict) and "error" not in payload),
            "latest": latest,
        }
        if getattr(args, "json", False):
            print(json.dumps(row, sort_keys=True))
        else:
            if not row["ok"]:
                print(f"metrics: cannot reach {base}/metrics")
                if isinstance(payload, dict) and payload.get("detail"):
                    print(f"detail: {payload.get('detail')}")
            elif latest:
                tok_s = (
                    latest.get("tok_s")
                    or latest.get("decode_tok_s")
                    or latest.get("server_tok_s")
                )
                generated = latest.get("completion_tokens") or latest.get(
                    "generated_tokens"
                )
                verify_ms = latest.get("verify_ms_per_call") or latest.get(
                    "late_verify_ms"
                )
                cache_ratio = latest.get("cache_ratio")
                print(
                    "tok_s={tok_s} tokens={tokens} verify_ms={verify_ms} cache_ratio={cache_ratio}".format(
                        tok_s=tok_s if tok_s is not None else "n/a",
                        tokens=generated if generated is not None else "n/a",
                        verify_ms=verify_ms if verify_ms is not None else "n/a",
                        cache_ratio=cache_ratio if cache_ratio is not None else "n/a",
                    )
                )
            else:
                print("metrics: no generation recorded yet")
        seen += 1
        if count and seen >= count:
            break
        time.sleep(interval)
    return 0 if row.get("ok") else 1


def cmd_integrate_public(args: Any) -> int:
    action = args.integration
    server_url = f"http://{args.host}:{args.port}"
    api_base_url = server_url.rstrip("/") + "/v1"
    model_id = getattr(args, "model_id", None) or DEFAULT_PUBLIC_MODEL_ID
    if action == "openwebui":
        docker_command = _openwebui_docker_command(
            mtplx_port=int(args.port),
            webui_port=int(getattr(args, "webui_port", 3000) or 3000),
            single_user=bool(getattr(args, "single_user", False)),
            api_key=(
                f"${args.api_key_env}"
                if getattr(args, "api_key", None)
                else "mtplx-local"
            ),
        )
        payload = {
            "integration": "openwebui",
            "server_url": server_url,
            "base_url": api_base_url,
            "api_base_url": api_base_url,
            "docker_api_base_url": _openwebui_docker_api_base_url(int(args.port)),
            "model_id": model_id,
            "server_command": (
                f"mtplx quickstart --profile {_resolved_default_profile_name(args)} --host {args.host} --port {args.port} "
                "--no-stats-footer"
            ),
            "docker_command": _shell_join(docker_command),
            "docker_command_argv": docker_command,
            "single_user_warning": (
                "WEBUI_AUTH=False creates a single-user Open WebUI instance and "
                "cannot be safely toggled on an existing shared data volume."
                if getattr(args, "single_user", False)
                else None
            ),
            "api_key": {
                "required_for_localhost": False,
                "env": args.api_key_env,
            },
            "notes": [
                "Use the /v1 base URL as the OpenAI-compatible endpoint.",
                "Dockerized Open WebUI must use host.docker.internal, not 127.0.0.1, to reach MTPLX on the Mac host.",
                "The Docker command disables Open WebUI's Ollama probe and background title/tag/follow-up/autocomplete generations so MTPLX only serves the chat turn.",
                "Keep --no-stats-footer enabled for UI clients.",
            ],
        }
    elif action == "claude-code":
        payload = {
            "integration": "claude-code",
            "base_url": server_url,
            "model_id": model_id,
            "environment": {
                "ANTHROPIC_BASE_URL": server_url,
                "ANTHROPIC_AUTH_TOKEN": f"${args.api_key_env}",
                "ANTHROPIC_API_KEY": "",
                "ANTHROPIC_MODEL": model_id,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": model_id,
                "ANTHROPIC_DEFAULT_SONNET_MODEL": model_id,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": model_id,
                "CLAUDE_CODE_SUBAGENT_MODEL": model_id,
                "API_TIMEOUT_MS": "3000000",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            },
            "server_command": (
                f"mtplx quickstart --profile {_resolved_default_profile_name(args)} --host {args.host} --port {args.port} "
                "--no-stats-footer"
            ),
            "smoke": {
                "root_probe": f"curl {server_url}/",
                "messages": f"curl {server_url}/v1/messages",
            },
        }
    elif action == "opencode":
        from mtplx.opencode import build_opencode_provider_config

        api_key_suffix = _api_key_command_suffix(args)
        reasoning_policy = reasoning_policy_for_model(model_ref=model_id)
        payload = {
            "integration": "opencode",
            "server_url": server_url,
            "base_url": api_base_url,
            "api_base_url": api_base_url,
            "model_id": model_id,
            "config_path": "~/.config/opencode/opencode.json",
            "server_command": (
                f"mtplx quickstart --profile {_resolved_default_profile_name(args)} --host {args.host} --port {args.port} "
                f"{api_key_suffix}--reasoning auto --no-stats-footer"
            ),
            "config": build_opencode_provider_config(
                base_url=api_base_url,
                model_id=model_id,
                model_name="MTPLX local",
                api_key=(
                    f"${args.api_key_env}"
                    if getattr(args, "api_key", None)
                    else "mtplx-local"
                ),
                enable_thinking=reasoning_policy.supported,
                reasoning_effort=reasoning_policy.default_effort,
                reasoning_effort_levels=(
                    tuple(reasoning_policy.effort_levels)
                    if reasoning_policy.supported
                    else None
                ),
            ),
            "notes": [
                "OpenCode identifies itself with x-mtplx-client; the MTPLX app/CLI dial is the source of truth for reasoning effort and the family sampler stays server-side.",
                "options.reasoningEffort mirrors the MTPLX dial; an effort picked inside OpenCode overrides it for that request.",
                "Use MTPLX server settings or --reasoning on|off to change reasoning policy.",
            ],
        }
    elif action == "swival":
        from mtplx.swival import (
            build_swival_command,
            detect_swival_cli,
            shell_swival_command,
        )

        context_window = int(getattr(args, "context_window", None) or 262144)
        payload = {
            "integration": "swival",
            "server_url": server_url,
            "base_url": server_url,
            "api_base_url": api_base_url,
            "model_id": model_id,
            "context_window": context_window,
            "detected": detect_swival_cli(),
            "launch_command": shell_swival_command(
                base_url=server_url,
                model_id=model_id,
                context_window=context_window,
            ),
            "command_argv": build_swival_command(
                base_url=server_url,
                model_id=model_id,
                context_window=context_window,
            ),
            "server_command": (
                f"mtplx quickstart --profile {_resolved_default_profile_name(args)} --host {args.host} --port {args.port} "
                "--no-stats-footer"
            ),
            "notes": [
                "Swival generic provider receives the root server URL; Swival handles the OpenAI-compatible /v1 path.",
                "No MTPLX config file is written for Swival.",
                "No hidden output cap is added.",
            ],
        }
    else:
        raise SystemExit(f"unknown integration: {action}")
    if getattr(args, "smoke", False):
        payload["smoke_result"] = {
            "health": _http_json(server_url + "/health", timeout=float(args.timeout)),
            "models": _http_json(api_base_url + "/models", timeout=float(args.timeout)),
        }
    if getattr(args, "json", False):
        _print(payload)
    else:
        print(f"MTPLX connect: {action}")
        print(
            f"base URL: {api_base_url if action in {'openwebui', 'opencode'} else server_url}"
        )
        print(f"model: {model_id}")
        print(f"start server: {payload.get('server_command')}")
        if action == "openwebui":
            print("Open WebUI:")
            print("  Settings -> Connections -> OpenAI API")
            print(f"  API base URL: {api_base_url}")
            print("  API key: leave blank for localhost")
            if getattr(args, "docker", False):
                print("Docker:")
                print(f"  {_shell_join(payload['docker_command_argv'])}")
        elif action == "opencode":
            print("OpenCode:")
            print("  Config path: ~/.config/opencode/opencode.json")
            print("  Provider: mtplx")
            print("  Reasoning: controlled by MTPLX server settings")
        elif action == "swival":
            print("Swival:")
            print(f"  {payload.get('launch_command')}")
        else:
            env = payload.get("environment") or {}
            print("Claude Code environment:")
            for key, value in env.items():
                print(f"  {key}={value}")
        smoke = payload.get("smoke_result")
        if isinstance(smoke, dict):
            health = (
                smoke.get("health") if isinstance(smoke.get("health"), dict) else {}
            )
            models = (
                smoke.get("models") if isinstance(smoke.get("models"), dict) else {}
            )
            print("smoke:")
            print(f"  health: {'ok' if health.get('ok') else 'failed'}")
            data = models.get("data") if isinstance(models, dict) else None
            if isinstance(data, list):
                print(f"  models endpoint: ok ({len(data)} model(s))")
            elif models.get("error"):
                print(f"  models endpoint: failed ({models.get('error')})")
            else:
                print("  models endpoint: unknown")
    return 0


def cmd_openwebui_public(args: Any) -> int:
    if args.openwebui_action != "docker-command":
        raise SystemExit(f"unknown openwebui action: {args.openwebui_action}")
    command = _openwebui_docker_command(
        mtplx_port=int(getattr(args, "mtplx_port", 8000)),
        webui_port=int(getattr(args, "webui_port", 3000)),
        single_user=bool(getattr(args, "single_user", False)),
        api_key=str(getattr(args, "api_key", None) or "mtplx-local"),
    )
    payload = {
        "action": "openwebui docker-command",
        "docker_command": _shell_join(command),
        "docker_command_argv": command,
        "openwebui_url": f"http://127.0.0.1:{int(getattr(args, 'webui_port', 3000))}",
        "mtplx_api_base_url_for_container": _openwebui_docker_api_base_url(
            int(getattr(args, "mtplx_port", 8000))
        ),
        "single_user_warning": (
            "WEBUI_AUTH=False creates a single-user Open WebUI instance and cannot be safely toggled on an existing shared data volume."
            if getattr(args, "single_user", False)
            else None
        ),
    }
    if getattr(args, "json", False):
        _print(payload)
    else:
        print(payload["docker_command"])
        print(f"Open WebUI: {payload['openwebui_url']}")
        print(
            f"MTPLX API from container: {payload['mtplx_api_base_url_for_container']}"
        )
        if payload["single_user_warning"]:
            print(f"warning: {payload['single_user_warning']}")
    return 0


def _write_fixture_runtime_contract(
    path: Path, *, arch_id: str, profile: str = DEFAULT_PROFILE_NAME
) -> None:
    write_json(
        path / "mtplx_runtime.json",
        {
            "mtplx_version": "0.2.0",
            "arch_id": arch_id,
            "mtp_depth_max": 3,
            "recommended_profile": profile,
            "exactness_baseline": {"phase0h": "synthetic-smoke", "max_abs_diff": 0.0},
            "verified_on": {
                "timestamp": "2026-05-02T00:00:00Z",
                "hardware": "synthetic-fixture",
                "macos": "synthetic-fixture",
            },
        },
    )


def _write_fixture_weights(path: Path, keys: list[str]) -> None:
    import numpy as np
    from safetensors.numpy import save_file

    save_file({key: np.ones((1,), dtype=np.float32) for key in keys}, path)


def _architecture_qa_fixtures() -> list[dict[str, Any]]:
    from mtplx.constants import EXPECTED_MTP_KEYS

    return [
        {
            "label": "qwen3-next-verified-sidecar",
            "config": {
                "architectures": ["Qwen3NextForCausalLM"],
                "model_type": "qwen3_next",
                "mtp_num_hidden_layers": 1,
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "vocab_size": 32,
                "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
            },
            "contract": "qwen3-next-mtp",
            "safetensors": {"mtp.safetensors": list(EXPECTED_MTP_KEYS)},
            "expect": {
                "tier": "verified",
                "arch_id": "qwen3-next-mtp",
                "can_run": True,
                "recommended_backend": "qwen3_next",
                "runtime_compatibility": "native",
            },
        },
        {
            "label": "deepseek-v3-contract-gated",
            "config": {
                "architectures": ["DeepseekV3ForCausalLM"],
                "model_type": "deepseek_v3",
                "num_nextn_predict_layers": 2,
                "num_hidden_layers": 61,
            },
            "contract": "deepseek-v3-mtp",
            "safetensors": {
                "model.safetensors": [
                    "model.layers.61.enorm.weight",
                    "model.layers.62.enorm.weight",
                ]
            },
            "expect": {
                "tier": "verified",
                "arch_id": "deepseek-v3-mtp",
                "can_run": True,
                "recommended_backend": "deepseek_mtp",
                "runtime_compatibility": "native-contract-gated",
            },
        },
        {
            "label": "deepseek-v32-contract-gated",
            "config": {
                "architectures": ["DeepseekV32ForCausalLM"],
                "model_type": "deepseek_v32",
                "num_nextn_predict_layers": 2,
                "num_hidden_layers": 61,
            },
            "contract": "deepseek-v3-mtp",
            "safetensors": {
                "model.safetensors": [
                    "model.layers.61.enorm.weight",
                    "model.layers.62.enorm.weight",
                ]
            },
            "expect": {
                "tier": "verified",
                "arch_id": "deepseek-v3-mtp",
                "can_run": True,
                "recommended_backend": "deepseek_mtp",
                "runtime_compatibility": "native-contract-gated",
            },
        },
        {
            "label": "glm-moe-dsa-contract-gated",
            "config": {
                "architectures": ["GlmMoeDsaForCausalLM"],
                "model_type": "glm_moe_dsa",
                "num_nextn_predict_layers": 2,
                "num_hidden_layers": 61,
            },
            "contract": "glm-moe-dsa-mtp",
            "safetensors": {
                "model.safetensors": [
                    "model.layers.61.enorm.weight",
                    "model.layers.62.enorm.weight",
                ]
            },
            "expect": {
                "tier": "verified",
                "arch_id": "glm-moe-dsa-mtp",
                "can_run": True,
                "recommended_backend": "deepseek_mtp",
                "runtime_compatibility": "native-contract-gated",
            },
        },
        {
            "label": "glm4-moe-contract-gated",
            "config": {
                "architectures": ["Glm4MoeForCausalLM"],
                "model_type": "glm4_moe",
                "num_nextn_predict_layers": 1,
                "num_hidden_layers": 47,
            },
            "contract": "glm4-moe-mtp",
            "safetensors": {"model.safetensors": ["model.layers.47.enorm.weight"]},
            "expect": {
                "tier": "verified",
                "arch_id": "glm4-moe-mtp",
                "can_run": True,
                "recommended_backend": "glm_mtp",
                "runtime_compatibility": "native-contract-gated",
            },
        },
        {
            "label": "glm4-moe-family-gated",
            "config": {
                "architectures": ["Glm4MoeForCausalLM"],
                "model_type": "glm4_moe",
                "num_nextn_predict_layers": 1,
                "num_hidden_layers": 47,
            },
            "safetensors": {"model.safetensors": ["model.layers.47.hnorm.weight"]},
            "expect": {
                "tier": "family-compatible-unverified",
                "arch_id": "glm4-moe-mtp",
                "can_run": True,
                "recommended_backend": "glm_mtp",
                "runtime_compatibility": "native-family-gated",
            },
        },
        {
            "label": "glm4-moe-lite-contract-gated",
            "config": {
                "architectures": ["Glm4MoeLiteForCausalLM"],
                "model_type": "glm4_moe_lite",
                "num_nextn_predict_layers": 1,
                "num_hidden_layers": 47,
            },
            "contract": "glm4-moe-lite-mtp",
            "safetensors": {"model.safetensors": ["model.layers.47.enorm.weight"]},
            "expect": {
                "tier": "verified",
                "arch_id": "glm4-moe-lite-mtp",
                "can_run": True,
                "recommended_backend": "glm_mtp",
                "runtime_compatibility": "native-contract-gated",
            },
        },
        {
            "label": "glm4-moe-lite-family-gated",
            "config": {
                "architectures": ["Glm4MoeLiteForCausalLM"],
                "model_type": "glm4_moe_lite",
                "num_nextn_predict_layers": 1,
                "num_hidden_layers": 47,
            },
            "safetensors": {"model.safetensors": ["model.layers.47.eh_proj.weight"]},
            "expect": {
                "tier": "family-compatible-unverified",
                "arch_id": "glm4-moe-lite-mtp",
                "can_run": True,
                "recommended_backend": "glm_mtp",
                "runtime_compatibility": "native-family-gated",
            },
        },
        {
            "label": "mimo-contract-gated",
            "config": {
                "architectures": ["MiMoForCausalLM"],
                "model_type": "mimo",
                "num_nextn_predict_layers": 2,
                "num_hidden_layers": 46,
            },
            "contract": "mimo-mtp",
            "safetensors": {
                "model.safetensors": ["model.mtp_layers.0.token_layernorm.weight"]
            },
            "expect": {
                "tier": "verified",
                "arch_id": "mimo-mtp",
                "can_run": True,
                "recommended_backend": "mimo_mtp",
                "runtime_compatibility": "native-contract-gated",
            },
        },
        {
            "label": "mimo-family-gated",
            "config": {
                "architectures": ["MiMoForCausalLM"],
                "model_type": "mimo",
                "num_nextn_predict_layers": 2,
                "num_hidden_layers": 46,
            },
            "safetensors": {
                "model.safetensors": ["model.mtp_layers.0.hidden_layernorm.weight"]
            },
            "expect": {
                "tier": "family-compatible-unverified",
                "arch_id": "mimo-mtp",
                "can_run": True,
                "recommended_backend": "mimo_mtp",
                "runtime_compatibility": "native-family-gated",
            },
        },
        {
            "label": "minimax-m2-num-mtp-modules-recognized-pending",
            "config": {
                "architectures": ["MiniMaxM2ForCausalLM"],
                "model_type": "minimax_m2",
                "num_mtp_modules": 2,
            },
            "expect": {
                "tier": "architecture-compatible-but-unverified",
                "arch_id": "minimax-m2-mtp",
                "can_run": False,
                "runtime_compatibility": "recognized-backend-pending",
            },
        },
        {
            "label": "nemotron-h-contract-gated",
            "config": {
                "architectures": ["NemotronHForCausalLM"],
                "model_type": "nemotron_h",
                "num_nextn_predict_layers": 1,
                "num_hidden_layers": 52,
                "mtp_hybrid_override_pattern": "*E",
            },
            "contract": "nemotron-h-mtp",
            "safetensors": {
                "mtp.safetensors": [
                    "mtp.layers.0.enorm.weight",
                    "mtp.layers.0.hnorm.weight",
                    "mtp.layers.0.eh_proj.weight",
                    "mtp.layers.0.norm.weight",
                    "mtp.layers.0.mixer.q_proj.weight",
                    "mtp.layers.1.norm.weight",
                    "mtp.layers.1.mixer.gate.weight",
                    "mtp.layers.1.final_layernorm.weight",
                ]
            },
            "expect": {
                "tier": "verified",
                "arch_id": "nemotron-h-mtp",
                "can_run": True,
                "recommended_backend": "nemotron_h_mtp",
                "runtime_compatibility": "native-contract-gated",
            },
        },
        {
            "label": "nemotron-h-family-gated",
            "config": {
                "architectures": ["NemotronHForCausalLM"],
                "model_type": "nemotron_h",
                "num_nextn_predict_layers": 1,
                "num_hidden_layers": 52,
                "mtp_hybrid_override_pattern": "*E",
            },
            "safetensors": {
                "mtp.safetensors": [
                    "mtp.layers.0.enorm.weight",
                    "mtp.layers.0.hnorm.weight",
                    "mtp.layers.0.eh_proj.weight",
                    "mtp.layers.0.norm.weight",
                    "mtp.layers.0.mixer.q_proj.weight",
                    "mtp.layers.1.norm.weight",
                    "mtp.layers.1.mixer.gate.weight",
                    "mtp.layers.1.final_layernorm.weight",
                ]
            },
            "expect": {
                "tier": "family-compatible-unverified",
                "arch_id": "nemotron-h-mtp",
                "can_run": True,
                "recommended_backend": "nemotron_h_mtp",
                "runtime_compatibility": "native-family-gated",
            },
        },
        {
            "label": "step3p7-contract-gated",
            "config": {
                "architectures": ["Step3p7ForConditionalGeneration"],
                "model_type": "step3p7",
                "num_nextn_predict_layers": 3,
                "num_hidden_layers": 45,
            },
            "contract": "step3p5-mtp",
            "safetensors": {
                "model.safetensors": [
                    "model.layers.45.enorm.weight",
                    "model.layers.46.hnorm.weight",
                    "model.layers.47.transformer.shared_head.output.weight",
                ]
            },
            "expect": {
                "tier": "verified",
                "arch_id": "step3p5-mtp",
                "can_run": True,
                "recommended_backend": "step3p5_mtp",
                "runtime_compatibility": "native-contract-gated",
            },
        },
        {
            "label": "step3p7-family-gated",
            "config": {
                "architectures": ["Step3p7ForConditionalGeneration"],
                "model_type": "step3p7",
                "num_nextn_predict_layers": 3,
                "num_hidden_layers": 45,
            },
            "safetensors": {
                "model.safetensors": [
                    "model.layers.45.enorm.weight",
                    "model.layers.46.eh_proj.weight",
                    "model.layers.47.transformer.shared_head.output.weight",
                ]
            },
            "expect": {
                "tier": "family-compatible-unverified",
                "arch_id": "step3p5-mtp",
                "can_run": True,
                "recommended_backend": "step3p5_mtp",
                "runtime_compatibility": "native-family-gated",
            },
        },
        {
            "label": "gemma4-without-mtp-stays-no-mtp",
            "config": {
                "architectures": ["Gemma4ForCausalLM"],
                "model_type": "gemma4",
            },
            "expect": {
                "tier": "no-MTP",
                "arch_id": None,
                "can_run": False,
                "runtime_compatibility": "unsupported",
            },
        },
        {
            "label": "gemma4-mtp-marker-recognized-pending",
            "config": {
                "architectures": ["Gemma4ForCausalLM"],
                "model_type": "gemma4",
                "num_nextn_predict_layers": 1,
            },
            "expect": {
                "tier": "architecture-compatible-but-unverified",
                "arch_id": "gemma-mtp",
                "can_run": False,
                "runtime_compatibility": "recognized-backend-pending",
            },
        },
    ]


def _compact_qa_observed(inspection: dict[str, Any]) -> dict[str, Any]:
    compatibility = inspection.get("compatibility") or {}
    return {
        "architecture": inspection.get("architecture"),
        "model_type": inspection.get("model_type"),
        "mtp_num_hidden_layers": inspection.get("mtp_num_hidden_layers"),
        "tier": compatibility.get("tier"),
        "arch_id": compatibility.get("arch_id"),
        "can_run": compatibility.get("can_run"),
        "recommended_backend": compatibility.get("recommended_backend"),
        "runtime_compatibility": compatibility.get("runtime_compatibility"),
        "support_level": compatibility.get("support_level"),
        "message": compatibility.get("message"),
    }


def _qa_expectation_passed(
    observed: dict[str, Any], expected: dict[str, Any]
) -> tuple[bool, list[str]]:
    failures = []
    for key, expected_value in expected.items():
        if observed.get(key) != expected_value:
            failures.append(
                f"{key}: expected {expected_value!r}, got {observed.get(key)!r}"
            )
    return not failures, failures


def _run_architecture_fixture_qa() -> list[dict[str, Any]]:
    rows = []
    with tempfile.TemporaryDirectory(prefix="mtplx-arch-qa-") as tmp:
        root = Path(tmp)
        for spec in _architecture_qa_fixtures():
            model_dir = root / spec["label"]
            model_dir.mkdir(parents=True, exist_ok=True)
            write_json(model_dir / "config.json", spec["config"])
            if spec.get("contract"):
                _write_fixture_runtime_contract(
                    model_dir, arch_id=str(spec["contract"])
                )
            for filename, keys in (spec.get("safetensors") or {}).items():
                _write_fixture_weights(model_dir / filename, list(keys))
            try:
                inspection = inspect_model(str(model_dir)).to_dict()
                observed = _compact_qa_observed(inspection)
                passed, failures = _qa_expectation_passed(observed, spec["expect"])
            except Exception as exc:
                observed = {"error": str(exc)}
                passed = False
                failures = [str(exc)]
            rows.append(
                {
                    "label": spec["label"],
                    "expected": spec["expect"],
                    "observed": observed,
                    "passed": passed,
                    "failures": failures,
                }
            )
    return rows


def _run_runtime_import_smoke() -> list[dict[str, Any]]:
    smokes = []
    for module_name, class_name in (
        ("mtplx.backends.deepseek_mtp", "DeepSeekMTPBackend"),
        ("mtplx.backends.glm_mtp", "GLMMTPBackend"),
        ("mtplx.backends.mimo_mtp", "MiMoMTPBackend"),
        ("mtplx.backends.nemotron_h_mtp", "NemotronHMTPBackend"),
        ("mtplx.backends.step3p5_mtp", "Step3p5MTPBackend"),
        ("mtplx.backends.hy_v3_mtp", "HyV3MTPBackend"),
    ):
        try:
            module = importlib.import_module(module_name)
            backend = getattr(module, class_name)()
            health = backend.health()
            smokes.append(
                {
                    "module": module_name,
                    "class": class_name,
                    "passed": bool(health.get("contract_required")),
                    "health": health,
                }
            )
        except Exception as exc:
            smokes.append(
                {
                    "module": module_name,
                    "class": class_name,
                    "passed": False,
                    "error": str(exc),
                }
            )
    return smokes


def _cmd_model_qa_architectures(args: Any) -> int:
    catalog = architecture_catalog()
    ids = {row["arch_id"] for row in catalog}
    required_catalog = {
        "qwen3-next-mtp",
        "deepseek-v3-mtp",
        "glm-moe-dsa-mtp",
        "glm4-moe-mtp",
        "glm4-moe-lite-mtp",
        "mimo-mtp",
        "nemotron-h-mtp",
        "step3p5-mtp",
        "minimax-m2-mtp",
        "gemma-mtp",
    }
    required_verified = {
        "qwen3-next-mtp",
        "deepseek-v3-mtp",
        "glm-moe-dsa-mtp",
        "glm4-moe-mtp",
        "glm4-moe-lite-mtp",
        "mimo-mtp",
        "nemotron-h-mtp",
        "step3p5-mtp",
    }
    verified_ids = {row["arch_id"] for row in catalog if row.get("can_run_verified")}
    fixture_rows = _run_architecture_fixture_qa()
    runtime_smokes = (
        _run_runtime_import_smoke()
        if getattr(args, "runtime_import_smoke", False)
        else []
    )
    gates = {
        "catalog_has_main_families": required_catalog.issubset(ids),
        "verified_contract_gated_families_listed": required_verified.issubset(
            verified_ids
        ),
        "fixture_inspections_passed": all(row["passed"] for row in fixture_rows),
        "runtime_import_smoke_passed": (
            all(row["passed"] for row in runtime_smokes) if runtime_smokes else True
        ),
    }
    payload = {
        "action": "model qa-architectures",
        "fixture_count": len(fixture_rows),
        "verified_runtime_arch_ids": sorted(verified_ids),
        "recognized_backend_pending_arch_ids": sorted(
            row["arch_id"]
            for row in catalog
            if row.get("runtime_compatibility") == "recognized-backend-pending"
        ),
        "gates": gates,
        "fixtures": fixture_rows,
        "runtime_import_smokes": runtime_smokes,
        "passed": all(gates.values()),
    }
    output = Path(args.output) if getattr(args, "output", None) else None
    if output is not None:
        write_json(output, payload)
    if getattr(args, "json", False):
        _print(payload)
    else:
        print("MTPLX architecture QA")
        for row in fixture_rows:
            status = "pass" if row["passed"] else "fail"
            print(f"- {row['label']}: {status}")
        print(f"passed: {str(payload['passed']).lower()}")
        if output is not None:
            print(f"output: {output}")
    return 0 if payload["passed"] else EXIT_STRICT_GATE


def cmd_model_public(args: Any) -> int:
    if args.model_action == "architectures":
        rows = architecture_catalog()
        payload = {
            "action": "model architectures",
            "verified_runtime_arch_ids": [
                row["arch_id"] for row in rows if row.get("can_run_verified")
            ],
            "recognized_backend_pending_arch_ids": [
                row["arch_id"]
                for row in rows
                if row.get("runtime_compatibility") == "recognized-backend-pending"
            ],
            "architectures": rows,
        }
        if getattr(args, "json", False):
            _print(payload)
        else:
            print("MTPLX architecture support")
            for row in rows:
                status = (
                    "runnable" if row.get("can_run_verified") else "backend-pending"
                )
                print(
                    f"- {row['arch_id']}: {row['display_name']} "
                    f"({status}, backend={row.get('backend') or 'none'})"
                )
        return 0
    if args.model_action == "qa-architectures":
        return _cmd_model_qa_architectures(args)
    if args.model_action != "publish-check":
        raise SystemExit(f"unknown model action: {args.model_action}")
    staging = Path(args.staging_dir)
    manifest_path = staging / "MTPLX_PUBLISH_MANIFEST.json"
    runtime_contract_path = staging / "mtplx_runtime.json"
    symlinks = (
        [str(path) for path in staging.iterdir() if path.is_symlink()]
        if staging.exists()
        else []
    )
    manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inspection: dict[str, Any] | None = None
    inspect_error = None
    if staging.exists():
        try:
            inspection = _compact_model_summary(inspect_model(str(staging)).to_dict())
        except Exception as exc:
            inspect_error = str(exc)
    gates = {
        "staging_exists": staging.exists(),
        "manifest_exists": manifest_path.exists(),
        "runtime_contract_exists": runtime_contract_path.exists(),
        "no_symlinks": not symlinks,
        "repo_id_explicit": bool(args.repo_id or (manifest or {}).get("repo_id")),
        "inspect_verified": bool(
            inspection
            and (inspection.get("compatibility") or {}).get("tier") == "verified"
            and (inspection.get("compatibility") or {}).get("can_run")
        ),
    }
    payload = {
        "action": "model publish-check",
        "staging_dir": str(staging),
        "repo_id": args.repo_id or (manifest or {}).get("repo_id"),
        "manifest": str(manifest_path),
        "runtime_contract": str(runtime_contract_path),
        "symlinks": symlinks,
        "inspection": inspection,
        "inspect_error": inspect_error,
        "size_bytes": (manifest or {}).get("size_bytes"),
        "weight_size_bytes": (manifest or {}).get("weight_size_bytes"),
        "uploaded": (manifest or {}).get("upload_policy", {}).get("uploaded"),
        "gates": gates,
        "passed": all(gates.values()),
    }
    _print(payload)
    return 0 if payload["passed"] else EXIT_STRICT_GATE


def cmd_config_public(args: Any) -> int:
    from mtplx.config import CONFIG_VALUE_KEYS, load_user_config, user_config_path
    from mtplx.profiles import resolve_profile_name

    path = user_config_path(getattr(args, "config", None))
    if args.config_action == "show":
        payload = load_user_config(path).to_dict()
        if getattr(args, "json", False):
            _print(payload)
        else:
            for key, value in payload.items():
                print(f"{key}: {value}")
        return 0
    if args.config_action != "set":
        raise SystemExit(f"unknown config action: {args.config_action}")
    current = load_user_config(path)
    values = {key: getattr(current, key) for key in CONFIG_VALUE_KEYS}
    key = str(args.key).strip()
    if key not in values:
        raise SystemExit(
            "config set key must be one of: " + ", ".join(CONFIG_VALUE_KEYS)
        )
    value = str(args.value).strip()
    if key == "profile":
        value = resolve_profile_name(value)
    if key == "thermal_control" and value not in {"auto", "none"}:
        raise SystemExit("thermal_control must be auto or none")
    if key == "paged_kv_quantization":
        value = normalize_paged_kv_quantization(value)
    if key == "scheduler_mode" and value not in {
        "serial",
        "cooperative",
        "ar_batch",
        "mtp_batch",
        "mtp_cohort_experimental",
    }:
        raise SystemExit(
            "scheduler_mode must be serial, cooperative, ar_batch, mtp_batch, "
            "or mtp_cohort_experimental"
        )
    if key == "batching_preset" and value not in {
        "solo",
        "latency",
        "agent",
        "throughput",
    }:
        raise SystemExit("batching_preset must be solo, latency, agent, or throughput")
    if key == "ssd_session_cache" and value not in {"off", "on", "write-only"}:
        raise SystemExit("ssd_session_cache must be off, on, or write-only")
    if key == "ram_session_cache_policy" and value not in {
        "target-default",
        "minimal",
        "bounded",
    }:
        raise SystemExit(
            "ram_session_cache_policy must be target-default, minimal, or bounded"
        )
    if key == "reasoning" and value not in {"auto", "on", "off"}:
        raise SystemExit("reasoning must be auto, on, or off")
    if key == "reasoning_effort" and value not in REASONING_EFFORT_CHOICES:
        raise SystemExit(
            "reasoning_effort must be " + ", ".join(REASONING_EFFORT_CHOICES)
        )
    if key in {
        "max_active_requests",
        "decode_batch_max",
        "prefill_chunk_tokens",
        "ssd_session_cache_min_prefix_tokens",
        "ram_session_cache_max_entries",
        "context_window",
        "top_k",
    }:
        value = int(value)
        if value < 1:
            raise SystemExit(f"{key} must be >= 1")
    if key in {"batch_wait_ms", "temperature", "top_p"}:
        value = float(value)
    if key in {"experimental_mtp_cohorts", "ram_session_block_prefix_restore"}:
        value = _parse_config_bool(value, key=key)
    values[key] = value
    if not getattr(args, "dry_run", False):
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# MTPLX user configuration"]
        for item_key in CONFIG_VALUE_KEYS:
            item_value = values.get(item_key)
            if item_value is not None:
                lines.append(f"{item_key} = {json.dumps(item_value)}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "path": str(path),
        "dry_run": bool(getattr(args, "dry_run", False)),
        "updated": {key: value},
        "config": values,
    }
    _print(payload)
    return 0


def _parse_config_bool(value: str, *, key: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"{key} must be true or false")


def _source_contains(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def _hotpath_boundary_report() -> dict[str, Any]:
    root = repo_root()
    paths = {
        "generation": root / "mtplx" / "generation.py",
        "gdn_capture": root / "mtplx" / "gdn_capture.py",
        "paged_cache": root / "mtplx" / "cache_state.py",
        "paged_sdpa": root / "mtplx" / "kernels" / "sdpa_2pass_paged.py",
        "verify_qmv": root / "mtplx" / "verify_qmv.py",
        "native_mlp_cpp": root
        / "native_extensions"
        / "verify_mlp"
        / "gate_up"
        / "gate_up.cpp",
        "native_gdn_cpp": root
        / "native_extensions"
        / "verify_mlp"
        / "gdn_tail"
        / "gdn_tail.cpp",
        "logits_topk": root / "mtplx" / "kernels" / "logits_topk.py",
    }
    boundaries = [
        {
            "name": "verify_output_eval",
            "file": str(paths["generation"].relative_to(root)),
            "status": "intentional-single-sync",
            "default_hot_path": True,
            "evidence": "_eval_verify_outputs materializes verifier logits/hidden once per verify cycle for exact sampling.",
            "next_action": "Only remove by moving sampler/residual correction onto device or owning a larger verify-cycle boundary.",
        },
        {
            "name": "gdn_capture_kernels",
            "file": str(paths["gdn_capture"].relative_to(root)),
            "status": "mlx-fast-metal-kernel",
            "default_hot_path": True,
            "evidence": "GDN capture backends are registered through mx.fast.metal_kernel and remain lazy until verifier eval.",
            "next_action": "Do not wrap again; only revisit as a larger GDN layer/state primitive.",
        },
        {
            "name": "paged_attention_mlx_vector",
            "file": str(paths["paged_sdpa"].relative_to(root)),
            "status": "mlx-fast-metal-kernel",
            "default_hot_path": True,
            "evidence": "The product long-response profile uses the local mlx_vector_paged SDPA kernels, not the raw external fallback.",
            "next_action": "Closed as the main decay target unless attention attribution changes.",
        },
        {
            "name": "qmv_custom_kernels",
            "file": str(paths["verify_qmv"].relative_to(root)),
            "status": "mlx-fast-metal-kernel",
            "default_hot_path": False,
            "evidence": "QMV probes are mx.fast.metal_kernel helpers; prior live screens show wrapper-level qmv changes regress or tie.",
            "next_action": "Keep default-off unless a larger primitive amortizes the boundary.",
        },
        {
            "name": "fused_logits_topk_distribution",
            "file": str(paths["logits_topk"].relative_to(root)),
            "status": "mlx-fast-metal-kernel-default-off-closed",
            "default_hot_path": False,
            "evidence": "Dense-logit tile top-k plus logsumexp preserved sparse target distributions but was slower than the stock batched MLX sampler on the 4x151936 verifier shape.",
            "next_action": "Do not promote; only revisit if the target-distribution boundary is fused with more of accept/reject or LM-head work.",
        },
        {
            "name": "native_rowwise_mlp",
            "file": str(paths["native_mlp_cpp"].relative_to(root)),
            "status": "cpp-mlx-primitive-default-off",
            "default_hot_path": False,
            "evidence": "Native MLP returns mx::array backed by GateUpSwiGLU* primitives, but the current rowwise boundary is closed live.",
            "next_action": "Next MLP work must be a larger verify-layer/MLX-source primitive, not the same rowwise wrapper.",
        },
        {
            "name": "native_residual_mlp",
            "file": str(paths["native_mlp_cpp"].relative_to(root)),
            "status": "cpp-mlx-primitive-default-off-closed",
            "default_hot_path": False,
            "evidence": "Residual+MLP layer-boundary primitive was exact and isolated-fast, but integrated D3/192 live generation regressed.",
            "next_action": "Do not tune this two-dispatch scratch design further; move to lower-level source work or a larger verify-cycle boundary.",
        },
        {
            "name": "native_gdn_tail",
            "file": str(paths["native_gdn_cpp"].relative_to(root)),
            "status": "cpp-mlx-primitive-default-off",
            "default_hot_path": False,
            "evidence": "Native GDN tail is a C++ MLX primitive with internal scratch, but isolated/live results were slower.",
            "next_action": "Only revisit as packed GDN family/state ownership, not tail-only.",
        },
        {
            "name": "external_vllm_partitioned_fallback",
            "file": str(paths["paged_cache"].relative_to(root)),
            "status": "raw-sync-hazard-default-off-or-fallback",
            "default_hot_path": False,
            "evidence": "Fallback path contains explicit mx.eval before raw op and mx.synchronize after it.",
            "next_action": "Never use as product path; require local primitive path or wrap the raw op before benchmarking.",
        },
        {
            "name": "state_root_eval_sync",
            "file": str(paths["generation"].relative_to(root)),
            "status": "diagnostic-boundary",
            "default_hot_path": True,
            "evidence": "Stable staged profiles use state-root eval; depends/async variants were exact but closed as product fixes.",
            "next_action": "Keep for exact state ownership until a larger owned verify-cycle boundary replaces it.",
        },
    ]
    raw_sync_markers = {
        "explicit_mx_synchronize_in_cache_state": _source_contains(
            paths["paged_cache"], "mx.synchronize()"
        ),
        "external_partitioned_raw_call_present": _source_contains(
            paths["paged_cache"], "paged_attention_v2_online_partitioned"
        ),
        "native_mlp_is_mlx_primitive": _source_contains(
            paths["native_mlp_cpp"], "std::make_shared<GateUpSwiGLU"
        ),
        "native_gdn_is_mlx_primitive": _source_contains(
            paths["native_gdn_cpp"], "std::make_shared<GdnNormGateOutQMV8"
        ),
    }
    return {
        "action": "debug hotpath",
        "boundaries": boundaries,
        "raw_sync_markers": raw_sync_markers,
        "verdict": {
            "do_not_loop": [
                "dynamic SDPA partition/block gridding",
                "existing native rowwise MLP",
                "native residual MLP layer-boundary fusion",
                "native GDN tail-only",
                "wrapper-level qmv/RMSNorm microkernels",
                "standalone dense-logit top-k distribution kernels",
                "state-root include/exclude or depends toggles",
            ],
            "highest_upside_next": [
                "larger owned verify-layer or verify-cycle primitive",
                "MLX-source/native primitive that fuses enough MLP/GDN work to amortize dispatch",
                "device-side sampler/residual boundary only if exactness can be preserved",
            ],
            "cold_speed_policy": "default-off diagnostics only; promote nothing without full exactness and same-night cold gate.",
        },
    }


def cmd_debug_public(args: Any) -> int:
    if args.debug_action == "hotpath":
        payload = _hotpath_boundary_report()
        output = Path(args.output) if getattr(args, "output", None) else None
        if output is not None:
            write_json(output, payload)
        _print(payload)
        return 0
    if args.debug_action != "bundle":
        raise SystemExit(f"unknown debug action: {args.debug_action}")
    bundle_id = args.run_id or f"debug-bundle-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = Path(args.output_dir or "outputs/cli/debug-bundles") / bundle_id
    out_dir.mkdir(parents=True, exist_ok=True)
    doctor_args = type("DoctorArgs", (), vars(args).copy())()
    doctor_args.project_root = getattr(args, "project_root", ".")
    doctor_args.model_cache = getattr(args, "model_cache", None)
    doctor_args.deep = True
    env = collect_environment(doctor_args.project_root).to_dict()
    from mtplx.hf_loader import hf_cache_report
    from mtplx.thermal import detect_thermal_control

    doctor = {
        "environment": env,
        "huggingface": hf_cache_report(cache_dir=doctor_args.model_cache),
        "thermal_control": detect_thermal_control(),
        "tools": {
            "python": sys.executable,
            "powermetrics": shutil.which("powermetrics"),
            "sudo": shutil.which("sudo"),
        },
    }
    doctor = _deep_doctor_report(doctor_args, doctor)
    _write_json_redacted(out_dir / "doctor.json", doctor)
    _write_json_redacted(
        out_dir / "metrics.json",
        _http_json(args.url.rstrip("/") + "/metrics") if args.url else {},
    )
    _write_json_redacted(
        out_dir / "summary.json",
        {
            "bundle_id": bundle_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "redacted": True,
            "files": ["doctor.json", "metrics.json", "summary.json"],
        },
    )
    archive = out_dir.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(out_dir.iterdir()):
            tar.add(path, arcname=f"{bundle_id}/{path.name}")
    payload = {
        "action": "debug bundle",
        "bundle_dir": str(out_dir),
        "archive": str(archive),
        "redacted": True,
    }
    _print(payload)
    return 0
