"""No-MLX user configuration helpers."""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from mtplx.constants import DEFAULT_RUNTIME_MODEL_DIR
from mtplx.default_models import is_verified_default_model_ref
from mtplx.profiles import DEFAULT_HF_MODEL_ID, DEFAULT_MODEL_ID, DEFAULT_PROFILE_NAME, resolve_profile_name
from mtplx.runtime_options import normalize_paged_kv_quantization


DEFAULT_CONFIG_PATH = Path("~/.mtplx/config.toml").expanduser()
RUNTIME_MODEL_COMMANDS = {"ask", "run", "chat", "start", "serve", "quickstart", "quick-start", "tune"}
CACHE_COMMANDS = {"pull", "list", "models", "remove"}
# Two default repo ids that early `mtplx init` builds wrote into config.toml:
# the v0.1.0-preview default and a 2026-05-03 development build's. Such a line
# is the old default, not a choice, so the command's own default applies.
# Later default ids are decided where the model is chosen
# (`is_verified_default_model_ref`). Exact ids only: MTPLX never wrote a path
# into config.toml (`mtplx init` and `mtplx setup` write the --model given or
# the default repo id, `mtplx config set` the value typed), so a folder there
# is the user's, whatever its name (#573).
LEGACY_DEFAULT_MODEL_REFS = {
    "mtplx/Qwen3.6-27B-MTPLX-GDN8-Speed4-CyanKiwiMTP",
    "Youssofal/Qwen3.6-27B-MTPLX-Optimized",
}
CONFIG_VALUE_KEYS = (
    "model",
    "model_dir",
    "model_dirs",
    "profile",
    "thermal_control",
    "paged_kv_quantization",
    "scheduler_mode",
    "batching_preset",
    "mtp_batch_numerics",
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
    "retrieval_trust_remote_code",
    # Where each agent client opens (mtplx/client_opening.py), set from the
    # app's Settings or `mtplx config set pi_opens_in Ghostty`.
    "pi_opens_in",
    "opencode_opens_in",
    "hermes_opens_in",
    "claude_code_opens_in",
    "codex_opens_in",
    "deepseek_harness_opens_in",
)


@dataclass(frozen=True)
class UserConfig:
    path: Path
    exists: bool
    model: str | None = None
    model_dir: str | None = None
    model_dirs: tuple[str, ...] = ()
    profile: str | None = None
    thermal_control: str | None = None
    paged_kv_quantization: str | None = None
    scheduler_mode: str | None = None
    batching_preset: str | None = None
    mtp_batch_numerics: str | None = None
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
    retrieval_trust_remote_code: bool | None = None
    pi_opens_in: str | None = None
    opencode_opens_in: str | None = None
    hermes_opens_in: str | None = None
    claude_code_opens_in: str | None = None
    codex_opens_in: str | None = None
    deepseek_harness_opens_in: str | None = None

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


def _warn_bad_config(resolved: Path, detail: str, *, key: str | None = None) -> None:
    """One stderr line, never a traceback. See ``load_user_config``.

    With ``key`` the line says which single key is being ignored, because the
    rest of the file still applies and the user must not read "ignoring
    <file>" as "none of my settings are active".
    """

    if key:
        subject, remedy = f"{key} in {resolved}", "fix or remove that line"
    else:
        subject, remedy = str(resolved), "fix or delete the file"
    print(
        f"mtplx: ignoring {subject}: {detail}; {remedy}; "
        "`mtplx config show` prints the active config",
        file=sys.stderr,
    )


def load_user_config(path: str | Path | None = None) -> UserConfig:
    resolved = user_config_path(path)
    if not resolved.exists():
        return UserConfig(path=resolved, exists=False)
    # This runs on EVERY CLI dispatch. A truncated or hand-mangled config.toml
    # used to raise TOMLDecodeError straight through `mtplx status`, `doctor`,
    # and `stop` — every command bricked by one bad file, with no way to read
    # the hint out of a traceback. Degrade to defaults and say so once instead.
    try:
        with resolved.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
        _warn_bad_config(resolved, str(exc))
        return UserConfig(path=resolved, exists=False)
    if not isinstance(data, dict):
        _warn_bad_config(resolved, "top level is not a table")
        return UserConfig(path=resolved, exists=False)
    model = data.get("model")
    model_dir = data.get("model_dir")
    profile = data.get("profile")
    thermal_control = data.get("thermal_control")
    if profile is not None:
        try:
            profile = resolve_profile_name(str(profile))
        except ValueError:
            # Config files can outlive profile names. Keep the raw value for
            # diagnostics, but do not let a stale saved profile break unrelated
            # commands before they can parse or explicitly choose a profile.
            profile = str(profile)

    def read(key: str, convert: Callable[[Any], Any], default: Any = None) -> Any:
        # Every typed key is converted inside this try. A badly typed saved
        # value (`context_window = "64k"`) degrades to that key's default with
        # one warning, matching the stale-profile handling above; the rest of
        # the file still applies. One key must not brick every command.
        try:
            return convert(data.get(key))
        except (TypeError, ValueError) as exc:
            _warn_bad_config(resolved, str(exc), key=key)
            return default

    return UserConfig(
        path=resolved,
        exists=True,
        model=str(model) if model else None,
        model_dir=str(model_dir) if model_dir else None,
        model_dirs=_str_tuple(data.get("model_dirs")),
        profile=str(profile) if profile else None,
        thermal_control=str(thermal_control) if thermal_control else None,
        paged_kv_quantization=read("paged_kv_quantization", _paged_kv_quantization_or_none),
        scheduler_mode=read("scheduler_mode", _str_or_none),
        batching_preset=read("batching_preset", _str_or_none),
        mtp_batch_numerics=read("mtp_batch_numerics", _str_or_none),
        max_active_requests=read("max_active_requests", _int_or_none),
        decode_batch_max=read("decode_batch_max", _int_or_none),
        batch_wait_ms=read("batch_wait_ms", _float_or_none),
        prefill_chunk_tokens=read("prefill_chunk_tokens", _int_or_none),
        experimental_mtp_cohorts=read("experimental_mtp_cohorts", _bool_or_none),
        ssd_session_cache=read("ssd_session_cache", _str_or_none),
        ssd_session_cache_dir=read("ssd_session_cache_dir", _str_or_none),
        ssd_session_cache_max_size=read("ssd_session_cache_max_size", _str_or_none),
        ssd_session_cache_min_prefix_tokens=read("ssd_session_cache_min_prefix_tokens", _int_or_none),
        ram_session_cache_policy=read("ram_session_cache_policy", _str_or_none),
        ram_session_cache_max_entries=read("ram_session_cache_max_entries", _int_or_none),
        ram_session_cache_max_size=read("ram_session_cache_max_size", _str_or_none),
        ram_session_cache_per_session_max_size=read("ram_session_cache_per_session_max_size", _str_or_none),
        ram_session_block_prefix_restore=read("ram_session_block_prefix_restore", _bool_or_none),
        context_window=read("context_window", _int_or_none),
        reasoning=read("reasoning", _str_or_none),
        reasoning_effort=read("reasoning_effort", _str_or_none),
        temperature=read("temperature", _float_or_none),
        top_p=read("top_p", _float_or_none),
        top_k=read("top_k", _int_or_none),
        api_key_file=read("api_key_file", _str_or_none),
        embedding_models=read("embedding_models", _str_tuple, ()),
        reranker_models=read("reranker_models", _str_tuple, ()),
        retrieval_max_resident=read("retrieval_max_resident", _int_or_none),
        retrieval_trust_remote_code=read("retrieval_trust_remote_code", _bool_or_none),
        pi_opens_in=read("pi_opens_in", _str_or_none),
        opencode_opens_in=read("opencode_opens_in", _str_or_none),
        hermes_opens_in=read("hermes_opens_in", _str_or_none),
        claude_code_opens_in=read("claude_code_opens_in", _str_or_none),
        codex_opens_in=read("codex_opens_in", _str_or_none),
        deepseek_harness_opens_in=read("deepseek_harness_opens_in", _str_or_none),
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
        _apply_model_search_defaults(args, config)
        _apply_profile_default(args, config)
        _apply_runtime_defaults(args, config)
    elif command == "bench" and getattr(args, "bench_action", None) in {"run", "tune"}:
        _apply_model_default(args, config)
        _apply_cache_default(args, config)
        _apply_model_search_defaults(args, config)
        _apply_profile_default(args, config)
        _apply_runtime_defaults(args, config)
    elif command in CACHE_COMMANDS:
        _apply_cache_default(args, config)
        _apply_model_search_defaults(args, config)
    elif command in {"doctor", "report", "status"}:
        if getattr(args, "model_cache", None) is None and config.model_dir:
            args.model_cache = config.model_dir
        _apply_model_search_defaults(args, config)
    elif command == "forge":
        if getattr(args, "model_root", None) is None:
            args.model_root = config.model_dir
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
        and config.model not in LEGACY_DEFAULT_MODEL_REFS
    ):
        args.model = config.model


def _apply_cache_default(args: Any, config: UserConfig) -> None:
    if hasattr(args, "cache_dir") and getattr(args, "cache_dir", None) is None and config.model_dir:
        args.cache_dir = config.model_dir


def _apply_model_search_defaults(args: Any, config: UserConfig) -> None:
    if not hasattr(args, "model_search_dirs"):
        return
    cli_flags = getattr(args, "_cli_flags", set()) or set()
    if "model-search-dir" in cli_flags:
        return
    if config.model_dirs:
        args.model_search_dirs = list(config.model_dirs)


def _apply_profile_default(args: Any, config: UserConfig) -> None:
    cli_flags = getattr(args, "_cli_flags", set())
    if "profile" in cli_flags:
        return
    command = getattr(args, "command", None)
    if command in {"start", "serve", "quickstart", "quick-start"} and "max" in cli_flags:
        return
    current = getattr(args, "profile", None)
    if config.profile and current in (None, DEFAULT_PROFILE_NAME):
        try:
            args.profile = resolve_profile_name(config.profile)
        except ValueError:
            return
        # A config-file profile is the user's standing pin: per-model
        # default-profile promotion must honor it (config "sustained" was
        # silently promoted to turbo), and a pin that sticks must be
        # visible (config "stable" silently defeated turbo). The marker is
        # deliberately not ``_cli_flags`` — that set records typed argv
        # only, and the onboarding gates depend on the distinction.
        args._profile_from_config = str(config.path)
        if not getattr(args, "json", False):
            print(f"profile: {args.profile} (from {config.path.name})", flush=True)


_RUNTIME_DEFAULTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "paged_kv_quantization": ("paged_kv_quantization", ("paged-kv-quantization", "paged-kv-quant", "kv-quant")),
    "scheduler_mode": ("scheduler_mode", ("scheduler-mode",)),
    "batching_preset": ("batching_preset", ("batching-preset",)),
    "mtp_batch_numerics": ("mtp_batch_numerics", ("mtp-batch-numerics",)),
    "max_active_requests": ("max_active_requests", ("max-active-requests",)),
    "decode_batch_max": ("decode_batch_max", ("decode-batch-max",)),
    "batch_wait_ms": ("batch_wait_ms", ("batch-wait-ms",)),
    "prefill_chunk_tokens": ("prefill_chunk_tokens", ("prefill-chunk-tokens",)),
    "experimental_mtp_cohorts": ("experimental_mtp_cohorts", ("experimental-mtp-cohorts",)),
    "embedding_models": ("embedding_model", ("embedding-model",)),
    "reranker_models": ("reranker_model", ("reranker-model",)),
    "retrieval_max_resident": ("retrieval_max_resident", ("retrieval-max-resident",)),
    "retrieval_trust_remote_code": ("retrieval_trust_remote_code", ("retrieval-trust-remote-code",)),
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
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"expected a list of model names, got {value!r}")
    return tuple(str(item) for item in value if str(item).strip())


def _str_or_none(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


def _paged_kv_quantization_or_none(value: Any) -> str | None:
    if value is None:
        return None
    normalized = normalize_paged_kv_quantization(value)
    return str(normalized) if normalized else None


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"expected a whole number, got {value!r}") from None


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"expected a number, got {value!r}") from None


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
