"""No-MLX user configuration helpers."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mtplx.constants import DEFAULT_RUNTIME_MODEL_DIR
from mtplx.default_models import is_verified_default_model_ref
from mtplx.profiles import DEFAULT_HF_MODEL_ID, DEFAULT_MODEL_ID, DEFAULT_PROFILE_NAME, resolve_profile_name
from mtplx.runtime_options import normalize_paged_kv_quantization


DEFAULT_CONFIG_PATH = Path("~/.mtplx/config.toml").expanduser()
RUNTIME_MODEL_COMMANDS = {"ask", "run", "chat", "start", "serve", "quickstart", "quick-start", "tune"}
CACHE_COMMANDS = {"pull", "list", "models", "remove"}
LEGACY_DEFAULT_MODEL_REFS = {
    "models/Qwen3.6-27B-MTPLX-GDN8-Speed4",
    "models/Qwen3.6-27B-MTPLX-Optimized-Speed",
    "Youssofal/Qwen3.6-27B-MTPLX-Optimized",
}
CONFIG_VALUE_KEYS = (
    "model",
    "model_dir",
    "profile",
    "thermal_control",
    "paged_kv_quantization",
    "scheduler_mode",
    "batching_preset",
    "max_active_requests",
    "decode_batch_max",
    "batch_wait_ms",
    "prefill_chunk_tokens",
    "experimental_mtp_cohorts",
    "ssd_session_cache",
    "ssd_session_cache_dir",
    "ssd_session_cache_max_size",
    "ssd_session_cache_min_prefix_tokens",
    "ram_session_cache_policy",
    "ram_session_cache_max_entries",
    "ram_session_cache_max_size",
    "ram_session_cache_per_session_max_size",
    "ram_session_block_prefix_restore",
    "context_window",
    "reasoning",
    "reasoning_effort",
    "temperature",
    "top_p",
    "top_k",
    "api_key_file",
    "embedding_models",
    "reranker_models",
    "retrieval_max_resident",
)


@dataclass(frozen=True)
class UserConfig:
    path: Path
    exists: bool
    model: str | None = None
    model_dir: str | None = None
    profile: str | None = None
    thermal_control: str | None = None
    paged_kv_quantization: str | None = None
    scheduler_mode: str | None = None
    batching_preset: str | None = None
    max_active_requests: int | None = None
    decode_batch_max: int | None = None
    batch_wait_ms: float | None = None
    prefill_chunk_tokens: int | None = None
    experimental_mtp_cohorts: bool | None = None
    ssd_session_cache: str | None = None
    ssd_session_cache_dir: str | None = None
    ssd_session_cache_max_size: str | None = None
    ssd_session_cache_min_prefix_tokens: int | None = None
    ram_session_cache_policy: str | None = None
    ram_session_cache_max_entries: int | None = None
    ram_session_cache_max_size: str | None = None
    ram_session_cache_per_session_max_size: str | None = None
    ram_session_block_prefix_restore: bool | None = None
    context_window: int | None = None
    reasoning: str | None = None
    reasoning_effort: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    api_key_file: str | None = None
    embedding_models: tuple[str, ...] = ()
    reranker_models: tuple[str, ...] = ()
    retrieval_max_resident: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "path": str(self.path),
            "exists": self.exists,
        }
        for key in CONFIG_VALUE_KEYS:
            payload[key] = getattr(self, key)
        return payload


def user_config_path(value: str | Path | None = None) -> Path:
    if value:
        return Path(value).expanduser()
    env = os.environ.get("MTPLX_CONFIG")
    if env:
        return Path(env).expanduser()
    return DEFAULT_CONFIG_PATH


def load_user_config(path: str | Path | None = None) -> UserConfig:
    resolved = user_config_path(path)
    if not resolved.exists():
        return UserConfig(path=resolved, exists=False)
    with resolved.open("rb") as handle:
        data = tomllib.load(handle)
    model = data.get("model")
    model_dir = data.get("model_dir")
    profile = data.get("profile")
    thermal_control = data.get("thermal_control")
    paged_kv_quantization = data.get("paged_kv_quantization")
    if profile is not None:
        try:
            profile = resolve_profile_name(str(profile))
        except ValueError:
            # Config files can outlive profile names. Keep the raw value for
            # diagnostics, but do not let a stale saved profile break unrelated
            # commands before they can parse or explicitly choose a profile.
            profile = str(profile)
    if paged_kv_quantization is not None:
        paged_kv_quantization = normalize_paged_kv_quantization(paged_kv_quantization)
    return UserConfig(
        path=resolved,
        exists=True,
        model=str(model) if model else None,
        model_dir=str(model_dir) if model_dir else None,
        profile=str(profile) if profile else None,
        thermal_control=str(thermal_control) if thermal_control else None,
        paged_kv_quantization=str(paged_kv_quantization) if paged_kv_quantization else None,
        scheduler_mode=_str_or_none(data.get("scheduler_mode")),
        batching_preset=_str_or_none(data.get("batching_preset")),
        max_active_requests=_int_or_none(data.get("max_active_requests")),
        decode_batch_max=_int_or_none(data.get("decode_batch_max")),
        batch_wait_ms=_float_or_none(data.get("batch_wait_ms")),
        prefill_chunk_tokens=_int_or_none(data.get("prefill_chunk_tokens")),
        experimental_mtp_cohorts=_bool_or_none(data.get("experimental_mtp_cohorts")),
        ssd_session_cache=_str_or_none(data.get("ssd_session_cache")),
        ssd_session_cache_dir=_str_or_none(data.get("ssd_session_cache_dir")),
        ssd_session_cache_max_size=_str_or_none(data.get("ssd_session_cache_max_size")),
        ssd_session_cache_min_prefix_tokens=_int_or_none(data.get("ssd_session_cache_min_prefix_tokens")),
        ram_session_cache_policy=_str_or_none(data.get("ram_session_cache_policy")),
        ram_session_cache_max_entries=_int_or_none(data.get("ram_session_cache_max_entries")),
        ram_session_cache_max_size=_str_or_none(data.get("ram_session_cache_max_size")),
        ram_session_cache_per_session_max_size=_str_or_none(data.get("ram_session_cache_per_session_max_size")),
        ram_session_block_prefix_restore=_bool_or_none(data.get("ram_session_block_prefix_restore")),
        context_window=_int_or_none(data.get("context_window")),
        reasoning=_str_or_none(data.get("reasoning")),
        reasoning_effort=_str_or_none(data.get("reasoning_effort")),
        temperature=_float_or_none(data.get("temperature")),
        top_p=_float_or_none(data.get("top_p")),
        top_k=_int_or_none(data.get("top_k")),
        api_key_file=_str_or_none(data.get("api_key_file")),
        embedding_models=_str_tuple(data.get("embedding_models")),
        reranker_models=_str_tuple(data.get("reranker_models")),
        retrieval_max_resident=_int_or_none(data.get("retrieval_max_resident")),
    )


def apply_user_config(args: Any, *, config_path: str | Path | None = None) -> UserConfig:
    config = load_user_config(config_path)
    setattr(args, "mtplx_config", config.to_dict())
    if not config.exists:
        return config

    command = getattr(args, "command", None)
    if command in RUNTIME_MODEL_COMMANDS:
        _apply_model_default(args, config)
        _apply_cache_default(args, config)
        _apply_profile_default(args, config)
        _apply_runtime_defaults(args, config)
    elif command == "bench" and getattr(args, "bench_action", None) in {"run", "tune"}:
        _apply_model_default(args, config)
        _apply_cache_default(args, config)
        _apply_profile_default(args, config)
        _apply_runtime_defaults(args, config)
    elif command in CACHE_COMMANDS:
        _apply_cache_default(args, config)
    elif command == "doctor" and getattr(args, "model_cache", None) is None and config.model_dir:
        args.model_cache = config.model_dir
    return config


def _apply_model_default(args: Any, config: UserConfig) -> None:
    cli_flags = getattr(args, "_cli_flags", set())
    if "model" in cli_flags:
        return
    current = getattr(args, "model", None)
    default_refs = {None, str(DEFAULT_RUNTIME_MODEL_DIR), DEFAULT_HF_MODEL_ID, DEFAULT_MODEL_ID}
    if (
        config.model
        and (current in default_refs or is_verified_default_model_ref(current))
        and not _is_legacy_default_model_ref(config.model)
    ):
        args.model = config.model


def _is_legacy_default_model_ref(model: str) -> bool:
    normalized = str(Path(model).expanduser()) if model.startswith(("~", "/")) else model
    return any(
        normalized == ref or normalized.endswith("/" + ref)
        for ref in LEGACY_DEFAULT_MODEL_REFS
    )


def _apply_cache_default(args: Any, config: UserConfig) -> None:
    if hasattr(args, "cache_dir") and getattr(args, "cache_dir", None) is None and config.model_dir:
        args.cache_dir = config.model_dir


def _apply_profile_default(args: Any, config: UserConfig) -> None:
    cli_flags = getattr(args, "_cli_flags", set())
    if "profile" in cli_flags:
        return
    command = getattr(args, "command", None)
    if command in {"start", "serve", "quickstart", "quick-start"} and "max" in cli_flags:
        return
    current = getattr(args, "profile", None)
    if config.profile and current == DEFAULT_PROFILE_NAME:
        try:
            args.profile = resolve_profile_name(config.profile)
        except ValueError:
            return


_RUNTIME_DEFAULTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "paged_kv_quantization": ("paged_kv_quantization", ("paged-kv-quantization", "paged-kv-quant", "kv-quant")),
    "scheduler_mode": ("scheduler_mode", ("scheduler-mode",)),
    "batching_preset": ("batching_preset", ("batching-preset",)),
    "max_active_requests": ("max_active_requests", ("max-active-requests",)),
    "decode_batch_max": ("decode_batch_max", ("decode-batch-max",)),
    "batch_wait_ms": ("batch_wait_ms", ("batch-wait-ms",)),
    "prefill_chunk_tokens": ("prefill_chunk_tokens", ("prefill-chunk-tokens",)),
    "experimental_mtp_cohorts": ("experimental_mtp_cohorts", ("experimental-mtp-cohorts",)),
    "embedding_models": ("embedding_model", ("embedding-model",)),
    "reranker_models": ("reranker_model", ("reranker-model",)),
    "retrieval_max_resident": ("retrieval_max_resident", ("retrieval-max-resident",)),
    "ssd_session_cache": ("ssd_session_cache", ("ssd-session-cache",)),
    "ssd_session_cache_dir": ("ssd_session_cache_dir", ("ssd-session-cache-dir",)),
    "ssd_session_cache_max_size": ("ssd_session_cache_max_size", ("ssd-session-cache-max-size",)),
    "ssd_session_cache_min_prefix_tokens": ("ssd_session_cache_min_prefix_tokens", ("ssd-session-cache-min-prefix-tokens",)),
    "ram_session_cache_policy": ("ram_session_cache_policy", ("ram-session-cache-policy",)),
    "ram_session_cache_max_entries": ("ram_session_cache_max_entries", ("ram-session-cache-max-entries",)),
    "ram_session_cache_max_size": ("ram_session_cache_max_size", ("ram-session-cache-max-size",)),
    "ram_session_cache_per_session_max_size": ("ram_session_cache_per_session_max_size", ("ram-session-cache-per-session-max-size",)),
    "ram_session_block_prefix_restore": ("ram_session_block_prefix_restore", ("ram-session-block-prefix-restore",)),
    "context_window": ("context_window", ("context-window",)),
    "reasoning": ("reasoning", ("reasoning",)),
    "reasoning_effort": ("reasoning_effort", ("reasoning-effort",)),
    "temperature": ("temperature", ("temperature", "default-temperature")),
    "top_p": ("top_p", ("top-p", "default-top-p")),
    "top_k": ("top_k", ("top-k", "default-top-k")),
    "api_key_file": ("api_key_file", ("api-key-file",)),
}


def _apply_runtime_defaults(args: Any, config: UserConfig) -> None:
    cli_flags = getattr(args, "_cli_flags", set()) or set()
    for config_key, (attr, flags) in _RUNTIME_DEFAULTS.items():
        if not hasattr(args, attr) and not config_key.startswith("ram_session_"):
            continue
        if any(flag in cli_flags for flag in flags):
            continue
        value = getattr(config, config_key, None)
        if value is not None:
            setattr(args, attr, value)


def _str_tuple(value: Any) -> tuple[str, ...]:
    """Read a config list of model references, tolerating a bare string."""
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value if str(item).strip())


def _str_or_none(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected boolean value, got {value!r}")
