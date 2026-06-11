#!/usr/bin/env python3
"""Serve MTPLX through a minimal OpenAI-compatible API.

This is intentionally small: it exists so local UI clients such as Open WebUI
can exercise the native-MTP runtime without turning MTPLX into a deployment
server yet. The default live path preserves the single-user MTP oracle; the
opt-in concurrent lane batches AR fallback work on the same single MLX owner
thread so coding-agent bursts do not require parallel model loops.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import builtins
import errno
import gc
import hashlib
import html
import json
import logging
import os
import re
import secrets
import socket
import subprocess
import sys
import time
import urllib.parse
import uuid
import webbrowser
from concurrent.futures import Future
from contextlib import asynccontextmanager, contextmanager, nullcontext, suppress
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from threading import Condition, Event, Lock, Thread, Timer
from typing import Any, Callable, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from mtplx.adaptive import AdaptiveDepthPolicy, ExpectedValueDepthPolicy
from mtplx.attention_context import attention_phase
from mtplx.cache_state import snapshot_cache
from mtplx.mtp_patch import MTPContract
from mtplx.backends.descriptors import (
    BackendDescriptor,
    assistant_target_distribution_choices,
    descriptor_for_backend_id,
    descriptor_from_runtime,
    model_controls_for_descriptor,
    set_draft_control_arg,
    sync_backend_arg_aliases,
    target_distribution_mode_from_args,
)
from mtplx.backends.registry import load_runtime_contract
from mtplx.batching import BatchSchedulerConfig, SchedulerMode, SchedulerPreset
from mtplx.chat_encoding import encode_chat_messages, is_gemma4_tokenizer
from mtplx.gemma4_pair import (
    GEMMA4_BACKEND,
    gemma4_pair_sampler_defaults,
    is_gemma4_pair_repo_id,
    resolve_gemma4_pair_paths,
)
from mtplx.model_scheduler import ModelWorkScheduler
from mtplx.sampling import SamplerConfig
from mtplx.profiles import (
    DEFAULT_HF_MODEL_ID,
    DEFAULT_PROFILE_NAME,
    NATIVE_MTP_60_MLX_FORK_COMMIT,
    NATIVE_MTP_60_MLX_FORK_FRAGMENT,
    PROFILE_CHOICES,
    apply_profile_env,
    get_profile,
    profile_env_status,
    resolve_long_context_mtp_depth,
)
from mtplx.runtime_options import (
    apply_paged_kv_quantization_env,
    normalize_paged_kv_quantization,
    resolve_api_key,
)
from mtplx.draft_lm_head import _install_draft_lm_head
from mtplx.fan_mode import (
    FAN_MODE_CHOICES,
    FAN_MODE_DEFAULT,
    FAN_MODE_MAX,
    FAN_MODE_SMART,
    normalize_fan_mode,
)
from mtplx.reasoning_codecs import (
    QWEN_STYLE_REASONING_BLOCK_RE,
    QWEN_STYLE_REASONING_CLOSE_RE,
    QWEN_STYLE_REASONING_CONTROL_RE,
    QWEN_STYLE_REASONING_OPEN_RE,
    QWEN_STYLE_REASONING_TAG_NAMES,
    normalize_qwen_thinking_tags,
    normalize_reasoning_tags as normalize_backend_reasoning_tags,
    split_reasoning_text,
    strip_qwen_style_reasoning_control_markup,
    strip_qwen_style_reasoning_from_content,
    stream_splitter_for_parser,
)
from mtplx.server.dashboard_state import DashboardState, InFlightHandle
from mtplx.server.omlx_bridge import (
    ToolCallStreamFilter as OMLXToolCallStreamFilter,
    extract_thinking as omlx_extract_thinking,
    extract_tool_calls_with_thinking as omlx_extract_tool_calls_with_thinking,
    normalize_messages_for_template as omlx_normalize_messages_for_template,
)
from mtplx.server_urls import bind_label, is_wildcard_bind, local_url_for_bind

LOGGER = logging.getLogger("mtplx.server.openai")

SCHEDULER_MODE_CHOICES = tuple(mode.value for mode in SchedulerMode)
BATCHING_PRESET_CHOICES = tuple(preset.value for preset in SchedulerPreset)
_STDOUT_LOGGING_BROKEN = False


def _safe_stdout_print(*values: Any, **kwargs: Any) -> bool:
    """Write daemon diagnostics without ever poisoning a user stream.

    OpenCode and the native app can own the daemon through pipes. If that pipe
    is closed or rotated while a streaming request is finalizing, a plain
    ``print(..., flush=True)`` raises ``BrokenPipeError``. That must never be
    converted into an OpenAI stream error: logging is observability, not part of
    the model/protocol contract.
    """

    global _STDOUT_LOGGING_BROKEN
    if _STDOUT_LOGGING_BROKEN:
        return False
    kwargs.setdefault("flush", True)
    try:
        builtins.print(*values, **kwargs)
        return True
    except OSError as exc:
        if exc.errno in {errno.EPIPE, errno.ECONNRESET}:
            _STDOUT_LOGGING_BROKEN = True
        return False
    except Exception:
        return False

try:
    from mtplx.generation import (
        PostcommitAbort,
        _default_stop_tokens,
        _sample_from_logits,
        _strip_terminal_stop,
        generate_ar,
        generate_mtpk,
        prefill_chunk_size_override,
        restore_or_prefill_prompt_state,
    )
    from mtplx.native_mlp import native_mlp_stats
    from mtplx.engine_session import (
        EngineSessionBusy,
        EngineSessionManager,
        hash_text,
        is_background_request,
        system_prompt_hash,
    )
    from mtplx.runtime import load
    from mtplx.session_bank import CacheMissReason, common_prefix_len
    from mtplx.cache_state import restore_cache, snapshot_cache

    _RUNTIME_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    _RUNTIME_IMPORT_ERROR = exc

    def _missing_runtime(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            f"MTPLX runtime dependencies are unavailable: {_RUNTIME_IMPORT_ERROR}"
        ) from _RUNTIME_IMPORT_ERROR

    generate_ar = _missing_runtime
    generate_mtpk = _missing_runtime
    prefill_chunk_size_override = nullcontext
    restore_or_prefill_prompt_state = _missing_runtime
    _default_stop_tokens = _missing_runtime
    _sample_from_logits = _missing_runtime
    _strip_terminal_stop = _missing_runtime
    common_prefix_len = _missing_runtime
    restore_cache = _missing_runtime
    snapshot_cache = _missing_runtime
    load = _missing_runtime

    class PostcommitAbort(RuntimeError):
        pass

    def native_mlp_stats() -> dict[str, Any]:
        return {"available": False, "error": repr(_RUNTIME_IMPORT_ERROR)}

    class EngineSessionBusy(RuntimeError):
        pass

    class EngineSessionManager:
        def __init__(self) -> None:
            _missing_runtime()

    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def is_background_request(*_args: Any, **_kwargs: Any) -> bool:
        return False

    def system_prompt_hash(messages: list[Any]) -> str | None:
        for message in messages:
            role = getattr(message, "role", None)
            if role == "system":
                return hash_text(_content_to_text(getattr(message, "content", "")))
        return None

    class CacheMissReason(Enum):
        NEW_SESSION = "new_session"
        SESSION_BUSY = "session_busy"
        BACKGROUND_BYPASS = "background_bypass"


FAST_PATH_ENV = {
    "MTPLX_LAZY_VERIFY_LOGITS": "1",
    "MTPLX_BATCH_TARGET_ARRAYS": "1",
    "MTPLX_LAZY_TARGET_DISTRIBUTIONS": "1",
    "MTPLX_LAZY_MTP_HISTORY_APPEND": "1",
    "MTPLX_DROP_EVENTS": "1",
    "MTPLX_SKIP_VERIFY_SNAPSHOT": "1",
}
VERIFY_SNAPSHOT_REQUIRED_STRATEGIES = {"trim_commit", "target_prefix"}
EXPECTED_MLX_QMV_FORK_COMMIT = NATIVE_MTP_60_MLX_FORK_COMMIT
EXPECTED_MLX_QMV_FORK_FRAGMENT = NATIVE_MTP_60_MLX_FORK_FRAGMENT
STATS_FOOTER_MARKER = "\n---\n⚡ **MTPLX TPS:**"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
CHAT_TEMPLATE_TURN_SENTINEL_RE = re.compile(
    r"<\|im_start\|>\s*(?:system|user|assistant|tool)?\s*|<\|im_end\|>",
    re.IGNORECASE,
)
CHAT_TEMPLATE_EMPTY_SENTINEL_MARKERS = ("<|third_empty|>",)
CHAT_TEMPLATE_SENTINEL_MARKERS = (
    "<|im_start|>",
    "<|im_end|>",
    *CHAT_TEMPLATE_EMPTY_SENTINEL_MARKERS,
)
CHAT_TEMPLATE_SENTINEL_RE = re.compile(
    rf"{CHAT_TEMPLATE_TURN_SENTINEL_RE.pattern}|<\|third_empty\|>",
    re.IGNORECASE,
)
STREAM_HEARTBEAT_INTERVAL_S = 10.0
STREAM_SILENCE_WARN_S = 30.0
STREAM_SILENCE_WARN_INTERVAL_S = 60.0
STREAM_HIDDEN_TOOL_GUARD_TOKENS = 2048
STREAM_HIDDEN_TOOL_GUARD_S = 30.0
STREAM_TOOL_CALL_FINISH_GRACE_S = 0.05
TOOL_PROTOCOL_BOUNDARY_GRACE_S = 0.05
_REASONING_DETAILS_RE = re.compile(
    r"<details\b(?=[^>]*\btype=[\"']reasoning[\"'])[^>]*>.*?</details>",
    re.IGNORECASE | re.DOTALL,
)
_REASONING_DETAILS_UNCLOSED_RE = re.compile(
    r"<details\b(?=[^>]*\btype=[\"']reasoning[\"'])[^>]*>.*\Z",
    re.IGNORECASE | re.DOTALL,
)
_REASONING_TAG_RE = re.compile(
    r"<(think|thinks|thinking|reason|reasoning|thought|thoughts)\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_REASONING_CONTROL_TAG_RE = re.compile(
    r"</?\s*(?:think|thinks|thinking|reason|reasoning|thought|thoughts)\b[^>\n]*>",
    re.IGNORECASE,
)


class _StreamCancelled(RuntimeError):
    """Raised inside the generation worker after a request is cancelled."""


class _StopSequenceHit(_StreamCancelled):
    """Raised by non-stream token monitors when a client stop string matches.

    Subclasses ``_StreamCancelled`` so the generation pipeline unwinds through
    the exact same battle-tested cancellation path that client disconnects
    already use; the endpoint handlers catch it first and turn the partial
    output into a normal ``finish_reason="stop"`` response instead of a 499.
    """

    def __init__(self, matched_stop: str) -> None:
        super().__init__(f"stop sequence matched: {matched_stop!r}")
        self.matched_stop = matched_stop


def _normalize_stop_sequences(stop: Any) -> list[str]:
    """Normalize an OpenAI ``stop`` value into a bounded list of stop strings.

    OpenAI accepts a string or an array of up to 4 strings. Empty entries are
    dropped and unparseable payloads are ignored rather than rejected, so an
    exotic client value can degrade to "no stop sequences" but never to a 4xx.
    """
    if stop is None:
        return []
    if isinstance(stop, str):
        values: list[Any] = [stop]
    elif isinstance(stop, (list, tuple)):
        values = list(stop)
    else:
        return []
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        if value not in normalized:
            normalized.append(value)
        if len(normalized) >= 4:
            break
    return normalized


def _trim_text_at_stop_sequences(
    text: str, stop_sequences: list[str]
) -> tuple[str, str | None]:
    """Trim ``text`` at the earliest stop-sequence match.

    Returns the trimmed text and the matched stop string (``None`` when no
    stop sequence occurs in the text).
    """
    if not text or not stop_sequences:
        return text, None
    earliest_index = -1
    matched: str | None = None
    for stop in stop_sequences:
        if not stop:
            continue
        index = text.find(stop)
        if index != -1 and (earliest_index == -1 or index < earliest_index):
            earliest_index = index
            matched = stop
    if matched is None:
        return text, None
    return text[:earliest_index], matched


class _StopSequenceStreamMonitor:
    """Incremental stop-string detector over a streamed text channel.

    ``feed`` returns the emittable portion of ``text``: everything before a
    stop match, with a possible partial-match suffix held back so a stop
    string split across deltas is never partially emitted to the client.
    Once ``stopped`` flips, ``feed`` returns "" forever; ``flush`` releases
    any held-back suffix when the stream ends without a match.
    """

    def __init__(self, stop_sequences: list[str]) -> None:
        self._stops = [stop for stop in stop_sequences if stop]
        self._max_hold = max((len(stop) for stop in self._stops), default=1) - 1
        self._pending = ""
        self.stopped = False
        self.matched_stop: str | None = None
        self.emitted_text = ""

    @property
    def active(self) -> bool:
        return bool(self._stops)

    def _emit(self, text: str) -> str:
        if text:
            self.emitted_text += text
        return text

    def feed(self, text: str) -> str:
        if self.stopped:
            return ""
        if not self._stops:
            return self._emit(text)
        if not text:
            return ""
        self._pending += text
        earliest_index = -1
        earliest_stop: str | None = None
        for stop in self._stops:
            index = self._pending.find(stop)
            if index != -1 and (earliest_index == -1 or index < earliest_index):
                earliest_index = index
                earliest_stop = stop
        if earliest_stop is not None:
            self.stopped = True
            self.matched_stop = earliest_stop
            emit = self._pending[:earliest_index]
            self._pending = ""
            return self._emit(emit)
        hold = 0
        max_hold = min(len(self._pending), self._max_hold)
        for size in range(max_hold, 0, -1):
            suffix = self._pending[-size:]
            if any(stop.startswith(suffix) for stop in self._stops):
                hold = size
                break
        if hold:
            emit = self._pending[:-hold]
            self._pending = self._pending[-hold:]
        else:
            emit = self._pending
            self._pending = ""
        return self._emit(emit)

    def flush(self) -> str:
        if self.stopped:
            self._pending = ""
            return ""
        emit = self._pending
        self._pending = ""
        return self._emit(emit)


def _stream_cancelled_queue_item(exc: _StreamCancelled) -> tuple[str, str]:
    reason = str(exc)
    exc.__traceback__ = None
    return ("cancelled", reason)


def _raise_if_stream_cancelled(
    cancel_event: Event, message: str = "stream client disconnected"
) -> None:
    if cancel_event.is_set():
        raise _StreamCancelled(message)


def _cancel_stream_generation(
    cancel_event: Event, generation_future: Any | None
) -> None:
    cancel_event.set()
    if generation_future is None:
        return
    try:
        generation_future.cancel()
    except Exception:
        return


async def _monitor_request_disconnect(
    raw_request: Request,
    cancel_event: Event,
    *,
    poll_s: float = 0.25,
    on_disconnect: Callable[[], None] | None = None,
) -> bool:
    while not cancel_event.is_set():
        try:
            disconnected = await raw_request.is_disconnected()
        except Exception:
            return False
        if disconnected:
            if on_disconnect is not None:
                on_disconnect()
            cancel_event.set()
            return True
        await asyncio.sleep(max(0.01, float(poll_s)))
    return False


def _comma_floats(value: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


_BEGIN_END_THOUGHT_RE = re.compile(
    r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>",
    re.IGNORECASE | re.DOTALL,
)
_STATS_FOOTER_RE = re.compile(
    r"\n---\s*\n⚡\s*(?:\*\*)?MTPLX TPS:(?:\*\*)?.*\Z",
    re.IGNORECASE | re.DOTALL,
)


def _fast_path_env_status() -> dict[str, dict[str, Any]]:
    return {
        key: {
            "expected": expected,
            "observed": os.environ.get(key),
            "ok": os.environ.get(key) == expected,
        }
        for key, expected in FAST_PATH_ENV.items()
    }


def _server_runtime_env_overrides(
    args: argparse.Namespace,
    model_runtime_env_overrides: Mapping[str, str] | None,
) -> dict[str, str]:
    overrides = dict(model_runtime_env_overrides or {})
    verify_strategy = (
        str(getattr(args, "verify_strategy", "") or "")
        .strip()
        .lower()
        .replace("-", "_")
    )
    generation_mode = (
        str(getattr(args, "generation_mode", "") or "")
        .strip()
        .lower()
        .replace("-", "_")
    )
    if generation_mode == "mtp" and verify_strategy in VERIFY_SNAPSHOT_REQUIRED_STRATEGIES:
        overrides["MTPLX_SKIP_VERIFY_SNAPSHOT"] = "0"
    return overrides


def _assert_fast_path_env() -> dict[str, dict[str, Any]]:
    status = _fast_path_env_status()
    bad = {key: value for key, value in status.items() if not value["ok"]}
    if bad:
        raise RuntimeError(
            "MTPLX fast-path env is incomplete: " + json.dumps(bad, sort_keys=True)
        )
    return status


def _template_hash(tokenizer: Any) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        template = repr(type(tokenizer))
    return hash_text(str(template))


def _array_bytes(value: Any) -> bytes:
    if value is None:
        return b""
    try:
        return np.asarray(value).tobytes()
    except Exception:
        return repr(value).encode("utf-8")


def _draft_head_identity(runtime: Any) -> str | None:
    text_model = getattr(runtime.model, "language_model", runtime.model)
    draft_head = getattr(text_model, "_mtplx_draft_lm_head", None)
    if draft_head is None:
        return None
    h = hashlib.sha256()
    for name in ("weight", "scales", "biases", "bias"):
        if hasattr(draft_head, name):
            h.update(_array_bytes(getattr(draft_head, name)))
    return h.hexdigest()[:16]


def _mlx_fork_status() -> dict[str, Any]:
    try:
        import mlx.core as mx

        path = Path(mx.__file__).resolve()
        version = getattr(mx, "__version__", None)
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}
    commit = None
    for parent in [path.parent, *path.parents]:
        if (
            EXPECTED_MLX_QMV_FORK_FRAGMENT in parent.name
            or EXPECTED_MLX_QMV_FORK_FRAGMENT in str(parent)
        ):
            try:
                # Bounded timeout: if git is slow or hung (lock
                # contention, zombie pickaxe process holding pack
                # files, slow disk), do not let the daemon block on
                # startup forever. The hash is diagnostic; failing
                # closed with `commit = None` is safe and matches the
                # `prefill_bench.py` pattern.
                commit = subprocess.check_output(
                    ["git", "-C", str(parent), "rev-parse", "--short", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=2.0,
                ).strip()
            except (
                subprocess.SubprocessError,
                FileNotFoundError,
                OSError,
            ):
                commit = None
            break
    path_active = EXPECTED_MLX_QMV_FORK_FRAGMENT in str(path)
    commit_matches = commit in {None, EXPECTED_MLX_QMV_FORK_COMMIT}
    ok = path_active and commit_matches
    return {
        "ok": ok,
        "path_active": path_active,
        "commit_matches": commit_matches,
        "path": str(path),
        "version": version,
        "expected_path_fragment": EXPECTED_MLX_QMV_FORK_FRAGMENT,
        "expected_commit": EXPECTED_MLX_QMV_FORK_COMMIT,
        "observed_commit": commit,
    }


def _parse_byte_limit(value: str | int | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", "")
    if not text:
        return None
    multipliers = {
        "b": 1,
        "kb": 1024,
        "kib": 1024,
        "mb": 1024**2,
        "mib": 1024**2,
        "gb": 1024**3,
        "gib": 1024**3,
    }
    for suffix, multiplier in sorted(
        multipliers.items(), key=lambda item: -len(item[0])
    ):
        if text.endswith(suffix):
            number = text[: -len(suffix)]
            return int(float(number) * multiplier)
    return int(float(text))


def _configure_mlx_cache_limit(args: argparse.Namespace) -> dict[str, Any]:
    raw = args.mlx_cache_limit or os.environ.get("MTPLX_MLX_CACHE_LIMIT")
    requested = _parse_byte_limit(raw)
    if requested is None:
        return {"requested": raw, "configured": False}
    import mlx.core as mx

    old_limit = int(mx.set_cache_limit(int(requested)))
    return {
        "requested": raw,
        "configured": True,
        "limit_bytes": int(requested),
        "previous_limit_bytes": old_limit,
    }


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = Field(
        default=None, validation_alias=AliasChoices("top_p", "topP")
    )
    top_k: int | None = Field(
        default=None, validation_alias=AliasChoices("top_k", "topK")
    )
    depth: int | None = None
    draft_block_size: int | None = None
    gemma_draft_block_size: int | None = None
    generation_mode: str | None = None
    seed: int | None = None
    enable_thinking: bool | None = None
    reasoning_effort: str | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None
    stop: Any = None
    stream_options: dict[str, Any] | None = None
    response_format: Any = None
    metadata: dict[str, Any] | None = None
    user: str | None = None


@dataclass
class AgentTranscriptCanonicalization:
    raw_message_chars: int = 0
    canonical_message_chars: int = 0
    assistant_reasoning_history_messages: int = 0
    assistant_reasoning_history_chars: int = 0
    assistant_structured_thinking_blocks: int = 0
    stripped_tool_preamble_messages: int = 0
    stripped_tool_preamble_chars: int = 0
    skipped_aborted_assistant_messages: int = 0
    skipped_orphan_chitchat_assistant_messages: int = 0
    dropped_simple_chitchat_history_messages: int = 0
    dropped_simple_chitchat_history_chars: int = 0
    replaced_simple_chitchat_system_messages: int = 0
    replaced_simple_chitchat_system_chars: int = 0
    injected_simple_chitchat_system_chars: int = 0
    replaced_client_system_messages: int = 0
    replaced_client_system_chars: int = 0
    injected_client_system_chars: int = 0
    skipped_repeated_assistant_messages: int = 0
    skipped_stalled_agent_preamble_messages: int = 0
    skipped_stalled_agent_preamble_chars: int = 0
    collapsed_repeated_user_messages: int = 0
    collapsed_repeated_user_chars: int = 0
    dropped_duplicate_user_messages: int = 0
    dropped_duplicate_user_chars: int = 0
    merged_consecutive_user_messages: int = 0
    merged_consecutive_user_chars: int = 0
    compacted_repeated_timeout_tool_messages: int = 0
    compacted_tool_result_messages: int = 0
    compacted_tool_result_chars: int = 0
    compacted_active_tool_result_messages: int = 0
    compacted_active_tool_result_chars: int = 0
    compacted_active_tool_result_read_hints: int = 0
    compacted_active_read_messages: int = 0
    compacted_active_read_chars: int = 0
    compacted_active_read_inspection_messages: int = 0
    compacted_active_read_inspection_chars: int = 0
    inspection_read_budget_candidate_messages: int = 0
    inspection_read_budget_max_lines_per_file: int = 0
    skipped_verbatim_tool_output_assistant_messages: int = 0
    skipped_verbatim_tool_output_assistant_chars: int = 0
    compacted_repeated_read_inspection_messages: int = 0
    compacted_repeated_read_inspection_chars: int = 0

    def to_metrics(self) -> dict[str, Any]:
        return {
            "transcript_raw_message_chars": int(self.raw_message_chars),
            "transcript_canonical_message_chars": int(self.canonical_message_chars),
            "transcript_assistant_reasoning_history_messages": int(
                self.assistant_reasoning_history_messages
            ),
            "transcript_assistant_reasoning_history_chars": int(
                self.assistant_reasoning_history_chars
            ),
            "transcript_assistant_structured_thinking_blocks": int(
                self.assistant_structured_thinking_blocks
            ),
            "transcript_canonicalized": bool(
                self.stripped_tool_preamble_messages
                or self.skipped_aborted_assistant_messages
                or self.skipped_orphan_chitchat_assistant_messages
                or self.dropped_simple_chitchat_history_messages
                or self.replaced_simple_chitchat_system_messages
                or self.injected_simple_chitchat_system_chars
                or self.replaced_client_system_messages
                or self.injected_client_system_chars
                or self.skipped_repeated_assistant_messages
                or self.skipped_stalled_agent_preamble_messages
                or self.collapsed_repeated_user_messages
                or self.dropped_duplicate_user_messages
                or self.merged_consecutive_user_messages
                or self.compacted_repeated_timeout_tool_messages
                or self.compacted_tool_result_messages
                or self.compacted_active_tool_result_messages
                or self.compacted_active_read_messages
                or self.compacted_active_read_inspection_messages
                or self.skipped_verbatim_tool_output_assistant_messages
                or self.compacted_repeated_read_inspection_messages
            ),
            "transcript_stripped_tool_preamble_messages": int(
                self.stripped_tool_preamble_messages
            ),
            "transcript_stripped_tool_preamble_chars": int(
                self.stripped_tool_preamble_chars
            ),
            "transcript_skipped_aborted_assistant_messages": int(
                self.skipped_aborted_assistant_messages
            ),
            "transcript_skipped_orphan_chitchat_assistant_messages": int(
                self.skipped_orphan_chitchat_assistant_messages
            ),
            "transcript_dropped_simple_chitchat_history_messages": int(
                self.dropped_simple_chitchat_history_messages
            ),
            "transcript_dropped_simple_chitchat_history_chars": int(
                self.dropped_simple_chitchat_history_chars
            ),
            "transcript_replaced_simple_chitchat_system_messages": int(
                self.replaced_simple_chitchat_system_messages
            ),
            "transcript_replaced_simple_chitchat_system_chars": int(
                self.replaced_simple_chitchat_system_chars
            ),
            "transcript_injected_simple_chitchat_system_chars": int(
                self.injected_simple_chitchat_system_chars
            ),
            "transcript_replaced_client_system_messages": int(
                self.replaced_client_system_messages
            ),
            "transcript_replaced_client_system_chars": int(
                self.replaced_client_system_chars
            ),
            "transcript_injected_client_system_chars": int(
                self.injected_client_system_chars
            ),
            "transcript_replaced_initial_client_system_messages": int(
                self.replaced_client_system_messages
            ),
            "transcript_replaced_initial_client_system_chars": int(
                self.replaced_client_system_chars
            ),
            "transcript_injected_initial_client_system_chars": int(
                self.injected_client_system_chars
            ),
            "transcript_skipped_repeated_assistant_messages": int(
                self.skipped_repeated_assistant_messages
            ),
            "transcript_skipped_stalled_agent_preamble_messages": int(
                self.skipped_stalled_agent_preamble_messages
            ),
            "transcript_skipped_stalled_agent_preamble_chars": int(
                self.skipped_stalled_agent_preamble_chars
            ),
            "transcript_collapsed_repeated_user_messages": int(
                self.collapsed_repeated_user_messages
            ),
            "transcript_collapsed_repeated_user_chars": int(
                self.collapsed_repeated_user_chars
            ),
            "transcript_dropped_duplicate_user_messages": int(
                self.dropped_duplicate_user_messages
            ),
            "transcript_dropped_duplicate_user_chars": int(
                self.dropped_duplicate_user_chars
            ),
            "transcript_merged_consecutive_user_messages": int(
                self.merged_consecutive_user_messages
            ),
            "transcript_merged_consecutive_user_chars": int(
                self.merged_consecutive_user_chars
            ),
            "transcript_compacted_repeated_timeout_tool_messages": int(
                self.compacted_repeated_timeout_tool_messages
            ),
            "transcript_compacted_tool_result_messages": int(
                self.compacted_tool_result_messages
            ),
            "transcript_compacted_tool_result_chars": int(
                self.compacted_tool_result_chars
            ),
            "transcript_compacted_active_tool_result_messages": int(
                self.compacted_active_tool_result_messages
            ),
            "transcript_compacted_active_tool_result_chars": int(
                self.compacted_active_tool_result_chars
            ),
            "transcript_compacted_active_tool_result_read_hints": int(
                self.compacted_active_tool_result_read_hints
            ),
            "transcript_compacted_active_read_messages": int(
                self.compacted_active_read_messages
            ),
            "transcript_compacted_active_read_chars": int(
                self.compacted_active_read_chars
            ),
            "transcript_compacted_active_read_inspection_messages": int(
                self.compacted_active_read_inspection_messages
            ),
            "transcript_compacted_active_read_inspection_chars": int(
                self.compacted_active_read_inspection_chars
            ),
            "transcript_inspection_read_budget_candidate_messages": int(
                self.inspection_read_budget_candidate_messages
            ),
            "transcript_inspection_read_budget_max_lines_per_file": int(
                self.inspection_read_budget_max_lines_per_file
            ),
            "transcript_skipped_verbatim_tool_output_assistant_messages": int(
                self.skipped_verbatim_tool_output_assistant_messages
            ),
            "transcript_skipped_verbatim_tool_output_assistant_chars": int(
                self.skipped_verbatim_tool_output_assistant_chars
            ),
            "transcript_compacted_repeated_read_inspection_messages": int(
                self.compacted_repeated_read_inspection_messages
            ),
            "transcript_compacted_repeated_read_inspection_chars": int(
                self.compacted_repeated_read_inspection_chars
            ),
        }


class MTPLXSettingsUpdate(BaseModel):
    # ``extra="allow"`` lets the handler return its own structured 400
    # for restart-required / unknown keys instead of Pydantic's generic
    # 422 — important so the dashboard can route the error precisely.
    model_config = ConfigDict(extra="allow")

    reasoning: str | None = None
    generation_mode: str | None = None
    depth: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_response_tokens: int | None = None
    stream_interval: int | None = None
    enable_thinking: bool | None = None
    reasoning_parser: str | None = None
    reasoning_effort: str | None = None
    prefill_chunk_tokens: int | None = None
    draft_temperature: float | None = None
    draft_top_p: float | None = None
    draft_top_k: int | None = None


class FanModeRequest(BaseModel):
    """Body for ``POST /v1/mtplx/thermal/fan_mode`` — V1 native app
    surface around ``mtplx.thermal``. ``mode`` is ``"max"`` (verified
    ramp), ``"smart"`` (request-scoped boost), or ``"default"``/``"auto"``
    (Apple-default restore)."""

    mode: str = Field(..., pattern="^(default|smart|max|auto)$")
    require_actual_ramp: bool = False
    timeout_s: float | None = Field(default=None, ge=0.0, le=120.0)


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    prompt: str | list[int] | list[str] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    depth: int | None = None
    draft_block_size: int | None = None
    gemma_draft_block_size: int | None = None
    generation_mode: str | None = None
    seed: int | None = None
    stop: Any = None
    stream: bool = False


class AnthropicMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = ""


class AnthropicMessagesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    max_tokens: int | None = None
    messages: list[AnthropicMessage] = Field(default_factory=list)
    system: Any | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | str | None = None
    metadata: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    thinking: Any | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    depth: int | None = None
    draft_block_size: int | None = None
    gemma_draft_block_size: int | None = None
    generation_mode: str | None = None
    stream: bool = False


def _startup_line(text: str = "") -> None:
    _safe_stdout_print(text)


def _startup_server_url(args: argparse.Namespace) -> str:
    return local_url_for_bind(
        str(getattr(args, "host", "127.0.0.1")),
        int(getattr(args, "port", 8000)),
    )


def _startup_bind_label(args: argparse.Namespace) -> str:
    return bind_label(
        str(getattr(args, "host", "127.0.0.1")),
        int(getattr(args, "port", 8000)),
    )


def _startup_openai_base_url(args: argparse.Namespace) -> str:
    return _startup_server_url(args) + "/v1"


def _startup_dashboard_url(args: argparse.Namespace) -> str:
    return _startup_server_url(args) + "/dashboard/"


def _startup_chat_url(args: argparse.Namespace) -> str:
    return _startup_server_url(args) + "/"


_BROWSER_AUTH_PATH = "/mtplx/browser-auth"
_BROWSER_AUTH_QUERY_PARAM = "mtplx_api_key"
_BROWSER_AUTH_COOKIE = "mtplx_browser_api_key"
_BROWSER_AUTH_COOKIE_MAX_AGE_SECONDS = 12 * 60 * 60


def _safe_browser_auth_next_path(value: Any) -> str:
    candidate = str(value or "/").strip() or "/"
    if (
        not candidate.startswith("/")
        or candidate.startswith("//")
        or "\r" in candidate
        or "\n" in candidate
    ):
        return "/"
    return candidate


def _browser_auth_url(
    server_url: str,
    api_key: str | None,
    *,
    next_path: str = "/",
) -> str:
    next_path = _safe_browser_auth_next_path(next_path)
    server_url = server_url.rstrip("/")
    if not api_key:
        return server_url + next_path
    query = urllib.parse.urlencode(
        {
            _BROWSER_AUTH_QUERY_PARAM: api_key,
            "next": next_path,
        }
    )
    return f"{server_url}{_BROWSER_AUTH_PATH}?{query}"


def _startup_browser_chat_url(args: argparse.Namespace) -> str:
    return _browser_auth_url(
        _startup_server_url(args),
        getattr(args, "api_key", None),
        next_path="/",
    )


def _startup_printable_chat_url(args: argparse.Namespace) -> str:
    if getattr(args, "api_key", None):
        return _startup_browser_chat_url(args)
    return _startup_chat_url(args)


def _startup_browser_dashboard_url(args: argparse.Namespace) -> str:
    return _browser_auth_url(
        _startup_server_url(args),
        getattr(args, "api_key", None),
        next_path="/dashboard/",
    )


def _health_runtime_mode_label(
    profile_name: str | None,
    generation_mode: str | None,
    *,
    fan_boost_active: bool,
) -> str:
    mode = "AR" if str(generation_mode or "").lower() == "ar" else "MTP"
    if profile_name == "sustained" and fan_boost_active:
        return f"Sustained Max {mode}"
    if profile_name == "sustained":
        return f"Sustained {mode}"
    if profile_name == "performance-cold" and fan_boost_active:
        return f"Burst {mode}"
    if profile_name == "performance-cold":
        return f"Performance-cold {mode}"
    if profile_name == "stable":
        return f"Stable {mode}"
    if profile_name:
        return f"{profile_name} {mode}"
    return mode


def _backend_descriptor(state: "ServerState") -> BackendDescriptor:
    descriptor = getattr(state, "backend_descriptor", None)
    if descriptor is not None:
        return descriptor
    return descriptor_from_runtime(
        getattr(state, "runtime", None),
        getattr(state, "args", None),
    )


def _reasoning_parser_for_state(state: "ServerState") -> str:
    parser = str(getattr(state.args, "reasoning_parser", "qwen3") or "qwen3")
    if parser == "none":
        return "none"
    backend = _backend_descriptor(state)
    if parser != backend.reasoning_codec.parser:
        return backend.reasoning_codec.parser
    return parser


def _open_browser_later(url: str, *, delay_s: float = 1.0) -> None:
    def open_url() -> None:
        try:
            webbrowser.open(url, new=2, autoraise=True)
        except Exception as exc:
            _safe_stdout_print(f"[mtplx] could not open browser: {exc}")

    timer = Timer(delay_s, open_url)
    timer.daemon = True
    timer.start()


def _open_pi_later(command: str, *, model_id: str, delay_s: float = 1.0) -> None:
    def open_pi() -> None:
        try:
            from mtplx.pi import launch_pi_in_terminal, pi_model_ref

            result = launch_pi_in_terminal(command, model_ref=pi_model_ref(model_id))
            if result.get("ok"):
                _startup_line("Pi opened in Terminal.")
            else:
                _startup_line(
                    f"warning: could not open Pi automatically: {result.get('error')}"
                )
                _startup_line(f"run manually: {command}")
        except Exception as exc:
            _startup_line(f"warning: could not open Pi automatically: {exc}")
            _startup_line(f"run manually: {command}")

    timer = Timer(delay_s, open_pi)
    timer.daemon = True
    timer.start()


def _open_opencode_later(*, delay_s: float = 1.0) -> None:
    def open_opencode() -> None:
        try:
            from mtplx.opencode import launch_opencode_app

            result = launch_opencode_app()
            if result.get("ok"):
                _startup_line("OpenCode Desktop opened.")
            else:
                _startup_line(
                    f"warning: could not open OpenCode automatically: {result.get('error')}"
                )
                _startup_line("open OpenCode manually and select the MTPLX model.")
        except Exception as exc:
            _startup_line(f"warning: could not open OpenCode automatically: {exc}")
            _startup_line("open OpenCode manually and select the MTPLX model.")

    timer = Timer(delay_s, open_opencode)
    timer.daemon = True
    timer.start()


def _open_hermes_later(command: str, *, delay_s: float = 1.0) -> None:
    def open_hermes() -> None:
        try:
            from mtplx.pi import launch_pi_in_terminal

            result = launch_pi_in_terminal(command)
            if result.get("ok"):
                _startup_line("Hermes Agent opened in Terminal.")
            else:
                _startup_line(
                    f"warning: could not open Hermes automatically: {result.get('error')}"
                )
                _startup_line(f"run manually: {command}")
        except Exception as exc:
            _startup_line(f"warning: could not open Hermes automatically: {exc}")
            _startup_line(f"run manually: {command}")

    timer = Timer(delay_s, open_hermes)
    timer.daemon = True
    timer.start()


def _server_console_enabled(state: Any) -> bool:
    return bool(getattr(getattr(state, "args", None), "server_console", False))


class _StartupHeartbeat:
    def __init__(self, label: str, *, interval_s: float = 10.0) -> None:
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


def _startup_heartbeat(label: str, *, interval_s: float = 10.0) -> _StartupHeartbeat:
    return _StartupHeartbeat(label, interval_s=interval_s)


def _parse_metal_memory_size_bytes(raw: str | int | None, default_bytes: int) -> int:
    text = str(raw or "").strip().upper().replace("_", "")
    if not text:
        return int(default_bytes)
    suffix_map = {
        "K": 1024,
        "KB": 1024,
        "M": 1024**2,
        "MB": 1024**2,
        "G": 1024**3,
        "GB": 1024**3,
        "T": 1024**4,
        "TB": 1024**4,
    }
    for suffix, multiplier in sorted(
        suffix_map.items(), key=lambda item: -len(item[0])
    ):
        if text.endswith(suffix):
            try:
                return max(1, int(float(text[: -len(suffix)]) * multiplier))
            except (TypeError, ValueError):
                return int(default_bytes)
    try:
        return max(1, int(float(text)))
    except (TypeError, ValueError):
        return int(default_bytes)


def _detect_total_ram_bytes_for_metal_caps() -> tuple[int | None, str]:
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            total = int(psutil.virtual_memory().total)
            if total > 0:
                return total, "psutil"
        except Exception:
            pass
    if sys.platform == "darwin":
        try:
            output = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            total = int(str(output).strip())
            if total > 0:
                return total, "sysctl_hw_memsize"
        except Exception:
            pass
    return None, "unknown"


def _metal_is_available(mx: Any) -> bool:
    metal = getattr(mx, "metal", None)
    is_available = getattr(metal, "is_available", None)
    if callable(is_available):
        try:
            return bool(is_available())
        except Exception:
            return False
    return metal is not None


def _set_metal_memory_limit(mx: Any, name: str, value: int) -> str:
    top_level = getattr(mx, name, None)
    if callable(top_level):
        top_level(int(value))
        return f"mx.{name}"
    metal = getattr(mx, "metal", None)
    metal_level = getattr(metal, name, None)
    if callable(metal_level):
        metal_level(int(value))
        return f"mx.metal.{name}"
    raise AttributeError(f"MLX memory cap API {name} is unavailable")


def _apply_metal_memory_caps(
    *,
    mx_module: Any | None = None,
    total_ram_bytes: int | None = None,
) -> dict[str, Any]:
    """Pin MLX Metal allocator caps at startup to avoid wired-memory swap-out
    pathologies under sustained long-context inference.

    On Apple Silicon, MLX's Metal allocator can grow the wired pool past safe
    headroom under back-to-back >30 K-token requests. When the OS starts
    swapping, decode collapses ~10x (50 t/s -> 2.5 t/s) and the kernel may kill
    the process. Setting both caps at startup keeps the allocator inside a
    fixed budget; ``clear_cache`` periodically drops idle pool memory.

    Operators can override via env:
      MTPLX_MEMORY_LIMIT_BYTES   - hard cap, default 75% of total RAM,
                                   capped at 192 GiB on very large Macs
      MTPLX_WIRED_LIMIT_BYTES    - wired (resident) cap, default 60% of total
                                   RAM, capped at 160 GiB on very large Macs

    Both accept plain bytes or K/M/G/T suffix.
    """
    if mx_module is None:
        try:
            import mlx.core as mx
        except Exception as exc:
            return {
                "applied": False,
                "reason": "mlx_unavailable",
                "error": repr(exc),
            }
    else:
        mx = mx_module
    if not _metal_is_available(mx):
        return {"applied": False, "reason": "metal_unavailable"}
    if total_ram_bytes is None:
        total_ram, total_ram_source = _detect_total_ram_bytes_for_metal_caps()
    else:
        total_ram = int(total_ram_bytes)
        total_ram_source = "explicit"
    mem_raw = os.environ.get("MTPLX_MEMORY_LIMIT_BYTES")
    wired_raw = os.environ.get("MTPLX_WIRED_LIMIT_BYTES")
    if total_ram is None or total_ram <= 0:
        if not mem_raw and not wired_raw:
            return {"applied": False, "reason": "ram_unknown"}
        default_mem = 1
        default_wired = 1
    else:
        # Percentage-only caps scale badly on 512 GiB M3 Ultra systems: 75% /
        # 60% permits hundreds of GiB of allocator high-water before MLX is
        # forced to release pressure. Keep the old behavior on 64-128 GiB Macs,
        # but bound the default resident budget on large unified-memory boxes.
        default_mem = min(
            total_ram,
            max(8 * 1024**3, int(total_ram * 0.75)),
            192 * 1024**3,
        )
        default_wired = min(
            default_mem,
            max(4 * 1024**3, int(total_ram * 0.60)),
            160 * 1024**3,
        )
    mem_limit = _parse_metal_memory_size_bytes(mem_raw, default_mem)
    wired_limit = _parse_metal_memory_size_bytes(wired_raw, default_wired)

    applied: dict[str, Any] = {
        "applied": True,
        "total_ram_bytes": total_ram,
        "total_ram_source": total_ram_source,
    }
    if wired_limit > mem_limit:
        wired_limit = mem_limit
        applied["wired_limit_clamped_to_memory_limit"] = True
    # Prefer the new top-level mx.set_memory_limit / mx.set_wired_limit; fall
    # back to the deprecated mx.metal.* names if running on an older MLX.
    try:
        applied["memory_limit_api"] = _set_metal_memory_limit(
            mx, "set_memory_limit", int(mem_limit)
        )
        applied["memory_limit_bytes"] = int(mem_limit)
    except Exception as exc:
        applied["memory_limit_error"] = str(exc)
    try:
        applied["wired_limit_api"] = _set_metal_memory_limit(
            mx, "set_wired_limit", int(wired_limit)
        )
        applied["wired_limit_bytes"] = int(wired_limit)
    except Exception as exc:
        applied["wired_limit_error"] = str(exc)
    if "memory_limit_bytes" not in applied and "wired_limit_bytes" not in applied:
        applied["applied"] = False
        applied["reason"] = "cap_api_unavailable"
    return applied


class ServerState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        try:
            args.paged_kv_quantization = normalize_paged_kv_quantization(
                getattr(args, "paged_kv_quantization", "off")
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        apply_paged_kv_quantization_env(args.paged_kv_quantization)
        self.model_id = args.model_id
        self.started_at_s = time.time()
        self.lock = Lock()
        self.foreground_lock = Lock()
        self.foreground_active = 0
        self.model_scheduler = ModelWorkScheduler(name="mtplx-model")
        # Compatibility shim for older tests/helpers that expect an executor
        # with submit()/shutdown(). New serving code uses model_scheduler
        # explicitly for foreground-vs-idle admission.
        self.generation_executor = self.model_scheduler
        self.postcommit_executor = None
        self.rate_limiter = _RateLimiter(args.rate_limit)
        self.metal_memory_caps = _apply_metal_memory_caps()
        self.profile = get_profile(args.profile)
        startup_backend = descriptor_for_backend_id(getattr(args, "backend_id", None))
        runtime_label = _health_runtime_mode_label(
            self.profile.name,
            getattr(args, "generation_mode", None),
            fan_boost_active=bool(getattr(args, "max", False)),
        )
        _startup_line(f"[4/6] Preparing {runtime_label} runtime")
        if args.generation_mode == "mtp" and not args.load_mtp:
            raise ValueError("--generation-mode mtp requires --load-mtp")
        if args.diagnostic_env_ablation:
            self.profile_env_status = profile_env_status(self.profile.name)
            self.fast_path_env_status = _fast_path_env_status()
        else:
            self.model_runtime_env_overrides: dict[str, str] = {}
            contract, _contract_error = load_runtime_contract(args.model)
            if contract is not None:
                self.model_runtime_env_overrides = dict(
                    contract.runtime_env_overrides or {}
                )
            runtime_env_overrides = _server_runtime_env_overrides(
                args,
                self.model_runtime_env_overrides,
            )
            clear_cache_every = getattr(args, "clear_cache_every", None)
            if clear_cache_every is not None:
                if clear_cache_every < 0:
                    raise ValueError("--clear-cache-every must be >= 0")
                runtime_env_overrides["MTPLX_CLEAR_CACHE_EVERY"] = str(
                    clear_cache_every
                )
            self.runtime_env_overrides = runtime_env_overrides
            apply_profile_env(
                self.profile.name,
                runtime_env_overrides=self.runtime_env_overrides,
            )
            self.profile_env_status = profile_env_status(
                self.profile.name,
                runtime_env_overrides=self.runtime_env_overrides,
            )
            if args.strict_startup_asserts:
                bad_profile_env = {
                    key: value
                    for key, value in self.profile_env_status.items()
                    if not value["ok"]
                }
                if bad_profile_env:
                    raise RuntimeError(
                        "MTPLX profile env is incomplete: "
                        + json.dumps(bad_profile_env, sort_keys=True)
                    )
            self.fast_path_env_status = _fast_path_env_status()
        _startup_line("[4/6] Checking local acceleration runtime")
        _startup_line("      This may take a few seconds.")
        self.mlx_fork_status = _mlx_fork_status()
        if (
            args.strict_mlx_fork_assert
            and startup_backend.requires_native_mlx_fork
            and self.profile.required_mlx_fork_commit
            and not self.mlx_fork_status.get("ok")
        ):
            raise RuntimeError(
                "Patched MLX qmv fork is not active: "
                + json.dumps(self.mlx_fork_status, sort_keys=True)
            )
        self.mlx_cache_limit_status = _configure_mlx_cache_limit(args)
        _startup_line("[4/6] Runtime checks complete")
        started = time.perf_counter()
        _startup_line(f"[5/6] Loading model weights: {args.model}")
        _startup_line(
            "      This is the long step; MTPLX is mapping the model into MLX."
        )
        _startup_line("      Model load in progress (this may take a minute).")
        load_heartbeat = _startup_heartbeat("Model still loading")
        try:
            self.runtime = self.model_scheduler.submit_foreground(
                load,
                args.model,
                mtp=bool(args.load_mtp),
                contract=MTPContract(
                    mtp_quant_bits=getattr(args, "mtp_quant_bits", None),
                    mtp_quant_group_size=getattr(args, "mtp_quant_group_size", 64),
                    mtp_quant_mode=getattr(args, "mtp_quant_mode", "affine"),
                ),
                mtp_adapter=getattr(args, "mtp_adapter", None),
                merge_mtp_adapter=bool(getattr(args, "merge_mtp_adapter", False)),
                gemma4_draft_block_size=getattr(args, "draft_block_size", None),
                gemma4_target_distribution_mode=target_distribution_mode_from_args(
                    args,
                    startup_backend,
                ),
                batch_key="startup.load",
            ).result()
        except BaseException as exc:
            elapsed_s = time.perf_counter() - started
            _startup_line(
                f"[5/6] Model load failed after {elapsed_s:.1f}s: {type(exc).__name__}: {exc}"
            )
            self.model_scheduler.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            load_heartbeat.set()
        self.load_time_s = time.perf_counter() - started
        _startup_line(f"[5/6] Model loaded in {self.load_time_s:.1f}s")
        self.backend_descriptor = descriptor_from_runtime(self.runtime, args)
        args.backend_id = self.backend_descriptor.backend_id
        if self.backend_descriptor.uses_draft_lm_head:
            _startup_line("[5/6] Installing native-MTP draft head")
        else:
            _startup_line(
                f"[5/6] {self.backend_descriptor.display_name} drafter is active"
            )
        if self.backend_descriptor.uses_draft_lm_head and self.runtime.mtp_enabled:
            self.draft_lm_head = self.model_scheduler.submit_foreground(
                _install_draft_lm_head,
                self.runtime,
                bits=args.draft_lm_head_bits,
                group_size=args.draft_lm_head_group_size,
                mode=args.draft_lm_head_mode,
                batch_key="startup.draft_head",
            ).result()
        elif self.runtime.mtp_enabled:
            self.draft_lm_head = {
                "installed": False,
                "reason": f"{self.backend_descriptor.backend_id}_external_assistant",
            }
        else:
            self.draft_lm_head = {"installed": False, "reason": "mtp_disabled"}
        if self.backend_descriptor.uses_draft_lm_head and self.runtime.mtp_enabled:
            self.draft_head_identity = (
                self.model_scheduler.submit_foreground(
                    _draft_head_identity,
                    self.runtime,
                    batch_key="startup.draft_head_identity",
                ).result()
            )
        elif self.backend_descriptor.uses_external_assistant and self.runtime.mtp_enabled:
            runtime_config = getattr(self.runtime, "config", None)
            assistant_path = getattr(runtime_config, "assistant_model_path", None)
            draft_block_size = getattr(runtime_config, "draft_block_size", None)
            target_mode = getattr(runtime_config, "target_distribution_mode", None)
            self.draft_head_identity = (
                f"{self.backend_descriptor.backend_id}:"
                f"{assistant_path}:block={draft_block_size}:target={target_mode}"
            )
        else:
            self.draft_head_identity = None
        self.chat_template_profile = _normalize_chat_template_profile(
            getattr(args, "chat_template_profile", None)
        )
        self.chat_template_report = self.model_scheduler.submit_foreground(
            _apply_chat_template_profile,
            self.runtime.tokenizer,
            args,
            batch_key="startup.chat_template_profile",
        ).result()
        if self.chat_template_report.get("profile") == _CHAT_TEMPLATE_PROFILE_CUSTOM:
            self.chat_template_profile = _CHAT_TEMPLATE_PROFILE_CUSTOM
        _startup_line(
            "[5/6] Chat template profile: "
            + str(self.chat_template_report.get("profile") or self.chat_template_profile)
        )
        self.template_hash = (
            self.model_scheduler.submit_foreground(
                _template_hash,
                self.runtime.tokenizer,
                batch_key="startup.template_hash",
            ).result()
            if self.runtime is not None
            else None
        )
        self.draft_sampler = (
            SamplerConfig(
                temperature=float(args.draft_temperature),
                top_p=float(args.draft_top_p),
                top_k=int(args.draft_top_k),
            )
            if args.draft_temperature is not None
            else None
        )
        self.model_context_window_max = _resolve_context_window(
            self.runtime.tokenizer,
            args.model,
        )
        requested_context_window = int(getattr(args, "context_window", None) or 0)
        self.context_window = (
            max(4_096, min(int(self.model_context_window_max), requested_context_window))
            if requested_context_window > 0
            else int(self.model_context_window_max)
        )
        _startup_line(f"[5/6] Context window: {self.context_window} tokens")
        self.session_bank_cold_tier = _session_bank_cold_tier_from_args(args)
        self.sessions = EngineSessionManager(cold_tier=self.session_bank_cold_tier)
        self.last_metrics: list[dict[str, Any]] = []
        self.tool_parse_counters = {key: 0 for key in _TOOL_PARSE_COUNTER_KEYS}
        # Activity timestamps used by the parent-process thermal watchdog to
        # decide when to drop fans back to auto after an idle period.
        self.last_request_started_at: float = 0.0
        self.last_request_at: float = 0.0
        self.requests_completed: int = 0
        self.requests_cancelled: int = 0
        self.main_system_prompt_hash: str | None = None
        self.fan_mode = normalize_fan_mode(
            getattr(args, "fan_mode", None)
            or os.environ.get("MTPLX_FAN_MODE")
            or FAN_MODE_DEFAULT
        )
        self.args.fan_mode = self.fan_mode
        from mtplx.thermal import SmartFanController

        self.smart_fans = SmartFanController(
            log=lambda line: LOGGER.info("%s", line)
        )
        # Dashboard primitives: pub/sub bus, in-flight registry, 5-min rolling
        # TPS window, lifetime counters, prefill history. Created before
        # warmup so the optional warmup metrics can flow through the same
        # surface as user requests.
        self.dashboard = DashboardState()
        self.ar_batch_service = _BatchedARGenerationService(self)
        self.warmup_status = _run_startup_warmup(self)

    def begin_foreground(self) -> None:
        with self.foreground_lock:
            self.foreground_active += 1
            self.last_request_started_at = time.time()

    def end_foreground(self) -> None:
        with self.foreground_lock:
            self.foreground_active = max(0, self.foreground_active - 1)

    def has_foreground(self) -> bool:
        with self.foreground_lock:
            return self.foreground_active > 0

    def foreground_count(self) -> int:
        with self.foreground_lock:
            return int(self.foreground_active)


def _submit_foreground_model_work(
    state: Any,
    fn: Callable[..., Any],
    *args: Any,
    batch_key: str | None = None,
    **kwargs: Any,
) -> Any:
    scheduler = getattr(state, "model_scheduler", None)
    if scheduler is not None and hasattr(scheduler, "submit_foreground"):
        return scheduler.submit_foreground(fn, *args, batch_key=batch_key, **kwargs)
    executor = getattr(state, "generation_executor", None)
    if executor is None:
        raise RuntimeError("state has no model work executor")
    return executor.submit(fn, *args, **kwargs)


def _session_bank_cold_tier_from_args(args: argparse.Namespace) -> Any | None:
    mode = str(getattr(args, "ssd_session_cache", "off") or "off").strip().lower()
    if mode == "off":
        return None
    from mtplx.cache_bank import (
        DEFAULT_COLD_TIER_DIR,
        DEFAULT_COLD_TIER_MAX_BYTES,
        DEFAULT_COLD_TIER_MIN_PREFIX_TOKENS,
        SessionBankColdTier,
        parse_size_bytes,
    )

    cache_dir = Path(
        str(getattr(args, "ssd_session_cache_dir", "") or DEFAULT_COLD_TIER_DIR)
    ).expanduser()
    max_bytes = parse_size_bytes(
        getattr(args, "ssd_session_cache_max_size", None),
        DEFAULT_COLD_TIER_MAX_BYTES,
    )
    min_prefix_tokens = int(
        getattr(
            args,
            "ssd_session_cache_min_prefix_tokens",
            DEFAULT_COLD_TIER_MIN_PREFIX_TOKENS,
        )
        or DEFAULT_COLD_TIER_MIN_PREFIX_TOKENS
    )
    return SessionBankColdTier(
        base_dir=cache_dir,
        mode=mode,
        max_bytes=max_bytes,
        min_prefix_tokens=min_prefix_tokens,
    )


class _BatchedARJob:
    """One OpenAI request admitted into the live AR batch lane."""

    def __init__(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        max_tokens: int,
        sampler: SamplerConfig,
        seed: int,
        stop_token_ids: set[int],
        token_callback: Callable[[list[int]], None] | None,
        prefill_callback: Callable[[dict[str, Any]], None] | None,
        request_observability: dict[str, Any] | None,
        mtp_disabled_reason: str | None,
        generation_limits: dict[str, Any],
        cancel_event: Event | None = None,
        session_id: str | None = None,
        session_bank: Any | None = None,
        session_restore_mode: str = "cold",
        session_template_hash: str | None = None,
        session_draft_head_identity: str | None = None,
        session_policy_fingerprint: str | None = None,
    ) -> None:
        self.request_id = request_id
        self.prompt_ids = [int(token) for token in prompt_ids]
        self.max_tokens = max(1, int(max_tokens))
        self.sampler = sampler
        self.seed = int(seed)
        self.stop_token_ids = {int(token) for token in stop_token_ids}
        self.token_callback = token_callback
        self.prefill_callback = prefill_callback
        self.request_observability = dict(request_observability or {})
        self.mtp_disabled_reason = mtp_disabled_reason
        self.generation_limits = dict(generation_limits)
        self.cancel_event = cancel_event
        self.session_id = session_id
        self.session_bank = session_bank
        self.session_restore_mode = session_restore_mode
        self.session_template_hash = session_template_hash
        self.session_draft_head_identity = session_draft_head_identity
        self.session_policy_fingerprint = session_policy_fingerprint
        self.future: Future = Future()
        self.tokens: list[int] = []
        self.token_times: list[float] = []
        self.created_s = time.perf_counter()
        self.admitted_s: float | None = None
        self.prefill_started_s: float | None = None
        self.prefill_done_s: float | None = None
        self.completed_s: float | None = None
        self.uid: int | None = None
        self.max_batch_size_observed = 1
        self.insert_prompt_ids = list(self.prompt_ids)
        self.insert_all_tokens: list[int] = []
        self.insert_cache: list[Any] | None = None
        self.cached_tokens = 0
        self.cache_miss_reason: str | None = self.request_observability.get(
            "cache_miss_reason"
        )
        self.session_cache_hit = False
        self.effective_restore_mode = session_restore_mode
        self.cache_source = "none"
        self.ssd_cache_hit = False
        self.ssd_cached_tokens = 0
        self.ssd_restore_s = 0.0
        self.ssd_suffix_tokens = 0
        self.shared_prefix_tokens = 0
        self.shared_prefix_prefill_s = 0.0
        self.shared_prefix_snapshot_s = 0.0
        self.prompt_prepare_s = 0.0

    def cancel_requested(self) -> bool:
        return bool(self.cancel_event is not None and self.cancel_event.is_set())

    def emit_prefill(self, payload: dict[str, Any]) -> None:
        if self.prefill_callback is None:
            return
        try:
            self.prefill_callback(payload)
        except Exception:
            pass

    def emit_token(self, token: int) -> None:
        token = int(token)
        self.tokens.append(token)
        if token not in self.stop_token_ids and self.token_callback is not None:
            self.token_callback([token])
        self.token_times.append(time.perf_counter())


class _BatchedARGenerationService:
    """Live AR continuous-batching pump owned by ``ModelWorkScheduler``.

    FastAPI/request worker threads enqueue jobs here. The service schedules one
    foreground pump on the existing model owner thread, then uses mlx-lm's
    ``BatchGenerator`` for the AR lane. That gives MTPLX a real V1 concurrent
    path without running multiple generation loops or moving native MTP off the
    solo oracle.
    """

    def __init__(self, state: "ServerState") -> None:
        self.state = state
        self._condition = Condition()
        self._pending: list[_BatchedARJob] = []
        self._active: dict[int, _BatchedARJob] = {}
        self._pump_scheduled = False
        self._last_batch_size = 0
        self._last_error: str | None = None

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "pending": len(self._pending),
                "active": len(self._active),
                "pump_scheduled": bool(self._pump_scheduled),
                "last_batch_size": int(self._last_batch_size),
                "last_error": self._last_error,
            }

    def submit(self, job: _BatchedARJob) -> Future:
        with self._condition:
            self._pending.append(job)
            if not self._pump_scheduled:
                self._pump_scheduled = True
                _submit_foreground_model_work(
                    self.state,
                    self._pump,
                    batch_key="ar_batch.pump",
                )
            self._condition.notify_all()
        return job.future

    def _make_sampler(self, job: _BatchedARJob) -> Callable[[Any], Any]:
        import mlx.core as mx

        if float(job.sampler.temperature) <= 0:
            return lambda logprobs: mx.argmax(logprobs, axis=-1)
        rng = np.random.default_rng(job.seed)

        def sample_one(logprobs: Any) -> Any:
            token, _distribution = _sample_from_logits(logprobs[0], job.sampler, rng)
            return mx.array([int(token)])

        return sample_one

    def _stop_sequences(self) -> list[list[int]]:
        stop_token_ids = _default_stop_tokens(self.state.runtime.tokenizer)
        return [[int(token)] for token in sorted(stop_token_ids)]

    def _wait_for_initial_cohort(self, config_dict: dict[str, Any]) -> None:
        max_batch = max(1, int(config_dict["decode_batch_max"]))
        if max_batch <= 1:
            return
        max_active = max(1, int(config_dict["max_active_requests"]))
        target_batch = max(1, min(max_batch, max_active))
        wait_s = self._initial_cohort_wait_s(config_dict)
        if wait_s <= 0.0:
            return
        deadline = time.perf_counter() + wait_s
        with self._condition:
            while not self._active:
                pending_count = sum(
                    1 for job in self._pending if not job.cancel_requested()
                )
                if pending_count <= 0 or pending_count >= target_batch:
                    return
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    return
                self._condition.wait(timeout=min(remaining, 0.001))

    def _cancelled_error(self, job: _BatchedARJob) -> _StreamCancelled:
        return _StreamCancelled(f"request {job.request_id} cancelled")

    @staticmethod
    def _shared_prefix_min_tokens() -> int:
        raw = os.environ.get("MTPLX_AR_BATCH_SHARED_PREFILL_MIN_TOKENS")
        if raw is None or raw == "":
            return 512
        try:
            return max(0, int(raw))
        except ValueError:
            return 512

    @staticmethod
    def _long_burst_prompt_tokens() -> int:
        raw = os.environ.get("MTPLX_AR_BATCH_LONG_BURST_PROMPT_TOKENS")
        if raw is None or raw == "":
            return 4096
        try:
            return max(0, int(raw))
        except ValueError:
            return 4096

    @staticmethod
    def _long_burst_wait_ms() -> float:
        raw = os.environ.get("MTPLX_AR_BATCH_LONG_BURST_WAIT_MS")
        if raw is None or raw == "":
            return 1000.0
        try:
            return max(0.0, min(5000.0, float(raw)))
        except ValueError:
            return 1000.0

    def _initial_cohort_wait_s(self, config_dict: dict[str, Any]) -> float:
        wait_ms = max(0.0, float(config_dict["batch_wait_ms"]))
        with self._condition:
            pending = [job for job in self._pending if not job.cancel_requested()]
        if len(pending) >= 2:
            max_prompt_tokens = max(len(job.prompt_ids) for job in pending)
            if max_prompt_tokens >= self._long_burst_prompt_tokens():
                wait_ms = max(wait_ms, self._long_burst_wait_ms())
        return wait_ms / 1000.0

    @staticmethod
    def _common_prefix_for_jobs(jobs: list[_BatchedARJob]) -> int:
        if len(jobs) < 2:
            return 0
        prefix = list(jobs[0].prompt_ids)
        for job in jobs[1:]:
            prefix = prefix[: common_prefix_len(prefix, job.prompt_ids)]
            if not prefix:
                return 0
        # BatchGenerator expects at least one prompt token after any restored
        # cache. If prompts are identical, share up to the token before the
        # generation start token.
        max_prefix = min(max(0, len(job.prompt_ids) - 1) for job in jobs)
        return min(len(prefix), max_prefix)

    def _prepare_session_bank_restore(self, job: _BatchedARJob) -> bool:
        if job.session_bank is None or len(job.prompt_ids) < 2:
            return False
        started = time.perf_counter()
        try:
            restored = job.session_bank.restore(
                self.state.runtime,
                job.prompt_ids,
                mode=_session_bank_restore_mode(job.session_restore_mode),
                session_id=job.session_id,
                template_hash=job.session_template_hash,
                policy_fingerprint=job.session_policy_fingerprint,
            )
        except Exception as exc:
            job.cache_miss_reason = f"ar_batch_restore_error:{type(exc).__name__}"
            return False
        finally:
            job.prompt_prepare_s += time.perf_counter() - started
        if restored is None:
            job.cache_miss_reason = getattr(
                job.session_bank,
                "last_miss_reason",
                job.cache_miss_reason,
            )
            return False
        if not self._cache_supports_batch_history_merge(restored.cache):
            # mlx-lm BatchGenerator requires history caches to expose merge().
            # MTPLX's sustained paged KV cache is exact for the serial MTP path
            # but not mergeable in BatchGenerator yet, so AR batching must treat
            # this as a miss and prefill from prompt tokens rather than crash.
            job.cache_miss_reason = "ar_batch_nonmergeable_history_cache"
            job.request_observability["ar_batch_cache_restore_skipped"] = (
                "nonmergeable_history_cache"
            )
            return False
        prefix_len = int(restored.entry.prefix_len)
        if prefix_len <= 0 or prefix_len >= len(job.prompt_ids):
            # A full-prefix cache does not expose the pre-last-token state that
            # mlx-lm BatchGenerator needs to start generation. Shared-prefix
            # preparation below can still help identical concurrent prompts by
            # prefilling prompt[:-1] once for the cohort.
            job.cache_miss_reason = "ar_batch_full_prefix_not_insertable"
            return False
        job.insert_cache = restored.cache
        job.insert_all_tokens = list(restored.entry.token_ids)
        job.insert_prompt_ids = list(job.prompt_ids[prefix_len:])
        job.cached_tokens = prefix_len
        job.session_cache_hit = True
        job.cache_miss_reason = None
        job.effective_restore_mode = str(restored.restore_mode)
        job.cache_source = str(getattr(restored, "cache_source", "ram") or "ram")
        job.ssd_cache_hit = bool(getattr(restored, "ssd_cache_hit", False))
        job.ssd_cached_tokens = int(getattr(restored, "ssd_cached_tokens", 0) or 0)
        job.ssd_restore_s = float(getattr(restored, "ssd_restore_s", 0.0) or 0.0)
        job.ssd_suffix_tokens = max(0, len(job.prompt_ids) - prefix_len)
        job.request_observability["cache_source"] = job.cache_source
        job.request_observability["ssd_cache_hit"] = job.ssd_cache_hit
        job.request_observability["ssd_cached_tokens"] = job.ssd_cached_tokens
        job.request_observability["ssd_restore_s"] = job.ssd_restore_s
        job.request_observability["ssd_suffix_tokens"] = job.ssd_suffix_tokens
        return True

    def _prepare_shared_prefix(self, jobs: list[_BatchedARJob]) -> None:
        candidates = [
            job
            for job in jobs
            if job.insert_cache is None
            and not job.cancel_requested()
            and len(job.prompt_ids) >= 2
        ]
        if len(candidates) < 2:
            return
        prefix_len = self._common_prefix_for_jobs(candidates)
        if prefix_len < self._shared_prefix_min_tokens():
            return
        prefix_tokens = list(candidates[0].prompt_ids[:prefix_len])
        if not prefix_tokens:
            return
        cache = self.state.runtime.make_cache()
        if not self._cache_supports_batch_history_merge(cache):
            for job in candidates:
                job.request_observability["ar_batch_shared_prefix_skipped"] = (
                    "nonmergeable_history_cache"
                )
            return
        prepare_started = time.perf_counter()
        try:
            import mlx.core as mx

            prefill_started = time.perf_counter()
            with attention_phase("ar_batch_shared_prefill"):
                logits = self.state.runtime.forward_ar(
                    mx.array([prefix_tokens]),
                    cache=cache,
                    return_hidden=False,
                )
            mx.eval(logits, [entry.state for entry in cache])
            prefill_s = time.perf_counter() - prefill_started
            snapshot_started = time.perf_counter()
            snapshot = snapshot_cache(cache)
            mx.eval(snapshot.states, snapshot.meta_states)
            snapshot_s = time.perf_counter() - snapshot_started
        except Exception as exc:
            for job in candidates:
                job.request_observability["ar_batch_shared_prefix_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
            return
        bank_stored = False
        bank_error: str | None = None
        bank_owner = next(
            (job for job in candidates if job.session_bank is not None),
            None,
        )
        if bank_owner is not None:
            try:
                put_snapshot = getattr(bank_owner.session_bank, "put_snapshot", None)
                if callable(put_snapshot):
                    entry = put_snapshot(
                        runtime=self.state.runtime,
                        token_ids=prefix_tokens,
                        cache_snapshot=snapshot,
                        logits=None,
                        hidden=None,
                        session_id=(
                            "ar_batch_shared:"
                            + hash_text(f"{prefix_len}:{prefix_tokens[:64]}")
                        ),
                        template_hash=bank_owner.session_template_hash,
                        policy_fingerprint=bank_owner.session_policy_fingerprint,
                        snapshot_epoch=prefix_len,
                    )
                    bank_stored = entry is not None
            except Exception as exc:
                bank_error = f"{type(exc).__name__}: {exc}"
        for index, job in enumerate(candidates):
            if index == 0:
                job.insert_cache = cache
            else:
                cloned_cache = self.state.runtime.make_cache()
                restore_cache(cloned_cache, snapshot)
                job.insert_cache = cloned_cache
            job.insert_all_tokens = list(prefix_tokens)
            job.insert_prompt_ids = list(job.prompt_ids[prefix_len:])
            job.cached_tokens = prefix_len
            job.session_cache_hit = True
            job.cache_miss_reason = None
            job.effective_restore_mode = "ar_batch_shared_prefix"
            job.cache_source = "ram"
            job.ssd_cache_hit = False
            job.ssd_cached_tokens = 0
            job.ssd_restore_s = 0.0
            job.ssd_suffix_tokens = 0
            job.shared_prefix_tokens = prefix_len
            job.shared_prefix_prefill_s = prefill_s
            job.shared_prefix_snapshot_s = snapshot_s
            job.prompt_prepare_s += time.perf_counter() - prepare_started
            job.request_observability["ar_batch_shared_prefix_tokens"] = prefix_len
            job.request_observability["ar_batch_shared_prefix_prefill_s"] = prefill_s
            job.request_observability["ar_batch_shared_prefix_snapshot_s"] = snapshot_s
            job.request_observability["ar_batch_shared_prefix_bank_stored"] = (
                bank_stored
            )
            job.request_observability["cache_source"] = job.cache_source
            job.request_observability["ssd_cache_hit"] = False
            job.request_observability["ssd_cached_tokens"] = 0
            job.request_observability["ssd_restore_s"] = 0.0
            job.request_observability["ssd_suffix_tokens"] = 0
            if bank_error is not None:
                job.request_observability["ar_batch_shared_prefix_bank_error"] = (
                    bank_error
                )

    def _prepare_prompt_inputs(self, jobs: list[_BatchedARJob]) -> None:
        for job in jobs:
            self._prepare_session_bank_restore(job)
        self._prepare_shared_prefix(jobs)

    @staticmethod
    def _cache_supports_batch_history_merge(cache: list[Any] | None) -> bool:
        if cache is None:
            return True
        return all(hasattr(entry, "merge") for entry in cache)

    def _split_unmergeable_history_batch(
        self, jobs: list[_BatchedARJob]
    ) -> tuple[list[_BatchedARJob], list[_BatchedARJob]]:
        for job in jobs:
            if job.insert_cache is not None and not self._cache_supports_batch_history_merge(
                job.insert_cache
            ):
                job.insert_cache = None
                job.insert_all_tokens = []
                job.insert_prompt_ids = list(job.prompt_ids)
                job.cached_tokens = 0
                job.session_cache_hit = False
                job.cache_source = "none"
                job.ssd_cache_hit = False
                job.ssd_cached_tokens = 0
                job.ssd_restore_s = 0.0
                job.ssd_suffix_tokens = 0
                job.effective_restore_mode = "cold"
                job.cache_miss_reason = "ar_batch_nonmergeable_history_cache"
                job.request_observability["ar_batch_cache_restore_skipped"] = (
                    "nonmergeable_history_cache"
                )
                job.request_observability["ar_batch_bypass_reason"] = (
                    "nonmergeable_history_cache"
                )
        return jobs, []

    def _admit_pending(self, generator: Any, config_dict: dict[str, Any]) -> None:
        with self._condition:
            max_active = max(1, int(config_dict["max_active_requests"]))
            capacity = max(0, max_active - len(self._active))
            if capacity <= 0:
                pending = []
            else:
                pending = self._pending[:capacity]
                del self._pending[:capacity]
        if not pending:
            return
        now = time.perf_counter()
        cancelled = [job for job in pending if job.cancel_requested()]
        pending = [job for job in pending if not job.cancel_requested()]
        for job in cancelled:
            if not job.future.done():
                job.future.set_exception(self._cancelled_error(job))
        pending = [job for job in pending if not job.future.cancelled()]
        if not pending:
            return
        self._prepare_prompt_inputs(pending)
        pending, requeued = self._split_unmergeable_history_batch(pending)
        if requeued:
            with self._condition:
                self._pending = [*requeued, *self._pending]
                self._condition.notify_all()
        for job in pending:
            job.admitted_s = now
            job.prefill_started_s = now
            job.emit_prefill(
                {
                    "phase": "started",
                    "tokens_total": int(len(job.prompt_ids)),
                    "started_s": now,
                    "scheduler_lane": "ar_batch",
                    "request_id": job.request_id,
                }
            )
        uids = generator.insert(
            [job.insert_prompt_ids for job in pending],
            max_tokens=[job.max_tokens for job in pending],
            caches=[job.insert_cache for job in pending],
            all_tokens=[job.insert_all_tokens for job in pending],
            samplers=[self._make_sampler(job) for job in pending],
        )
        with self._condition:
            for uid, job in zip(uids, pending):
                job.uid = int(uid)
                self._active[int(uid)] = job
            self._condition.notify_all()

    def _complete_job(self, job: _BatchedARJob, *, finish_reason: str) -> None:
        if job.future.done():
            return
        completed = time.perf_counter()
        job.completed_s = completed
        elapsed_s = max(0.0, completed - job.created_s)
        prompt_eval_time_s = (
            max(0.0, (job.prefill_done_s or completed) - (job.prefill_started_s or job.created_s))
            if job.prefill_started_s is not None
            else 0.0
        )
        completion_tokens = len(job.tokens)
        decode_elapsed_s = max(0.0, elapsed_s - prompt_eval_time_s)
        decode_tok_s = (
            completion_tokens / decode_elapsed_s if decode_elapsed_s > 0 else 0.0
        )
        stripped_tokens = _strip_terminal_stop(job.tokens, job.stop_token_ids)
        text = self.state.runtime.tokenizer.decode(stripped_tokens)
        stats: dict[str, Any] = {
            "mode": "ar",
            "generation_mode": "ar",
            "generated_tokens": int(completion_tokens),
            "elapsed_s": float(elapsed_s),
            "tok_s": float(decode_tok_s),
            "decode_elapsed_s": float(decode_elapsed_s),
            "decode_tok_s": float(decode_tok_s),
            "end_to_end_tok_s": (
                float(completion_tokens) / elapsed_s if elapsed_s > 0 else 0.0
            ),
            "target_forward_time_s": float(prompt_eval_time_s + decode_elapsed_s),
            "prompt_eval_time_s": float(prompt_eval_time_s),
            "prompt_tps": (
                len(job.prompt_ids) / prompt_eval_time_s
                if prompt_eval_time_s > 0 and job.prompt_ids
                else 0.0
            ),
            "prompt_target_prefill_time_s": float(prompt_eval_time_s),
            "prompt_target_prefill_tok_s": (
                max(0, len(job.prompt_ids) - int(job.cached_tokens)) / prompt_eval_time_s
                if prompt_eval_time_s > 0 and job.prompt_ids
                else 0.0
            ),
            "verify_calls": 0,
            "verify_time_s": 0.0,
            "accepted_by_depth": [],
            "drafted_by_depth": [],
            "draft_time_s": 0.0,
            "mtp_depth": 0,
            "requested_mtp_depth": 0,
            "speculative_depth": 0,
            "requested_speculative_depth": 0,
            "cached_tokens": int(job.cached_tokens),
            "new_prefill_tokens": max(0, len(job.prompt_ids) - int(job.cached_tokens)),
            "session_cache_hit": bool(job.session_cache_hit),
            "cache_source": job.cache_source,
            "ssd_cache_hit": bool(job.ssd_cache_hit),
            "ssd_cached_tokens": int(job.ssd_cached_tokens),
            "ssd_restore_s": float(job.ssd_restore_s),
            "ssd_suffix_tokens": int(job.ssd_suffix_tokens),
            "cache_miss_reason": job.cache_miss_reason,
            "session_restore_mode": job.effective_restore_mode,
            "ar_batch_shared_prefix_tokens": int(job.shared_prefix_tokens),
            "ar_batch_shared_prefix_prefill_s": float(job.shared_prefix_prefill_s),
            "ar_batch_shared_prefix_snapshot_s": float(job.shared_prefix_snapshot_s),
            "ar_batch_prompt_prepare_s": float(job.prompt_prepare_s),
            "scheduler_lane": "ar_batch",
            "scheduler_mode": str(getattr(self.state.args, "scheduler_mode", "")),
            "batching_preset": str(getattr(self.state.args, "batching_preset", "")),
            "scheduler_policy": _scheduler_policy_label(
                _scheduler_config_from_args(self.state.args)
            ),
            "request_id": job.request_id,
            "client_label": job.request_observability.get("request_client_label")
            or job.request_observability.get("request_client_hint")
            or "openai",
            "ar_batch_max_observed": int(job.max_batch_size_observed),
            "active_batch_size": int(job.max_batch_size_observed),
            "mtp_disabled_reason": job.mtp_disabled_reason,
            "queue_wait_s": max(
                0.0, (job.admitted_s or job.created_s) - job.created_s
            ),
            "request_started_s": float(job.created_s),
            "server_seed": int(job.seed),
        }
        stats.update(job.request_observability)
        stats.update(
            {
                "cached_tokens": int(job.cached_tokens),
                "new_prefill_tokens": max(
                    0, len(job.prompt_ids) - int(job.cached_tokens)
                ),
                "session_cache_hit": bool(job.session_cache_hit),
                "cache_source": job.cache_source,
                "ssd_cache_hit": bool(job.ssd_cache_hit),
                "ssd_cached_tokens": int(job.ssd_cached_tokens),
                "ssd_restore_s": float(job.ssd_restore_s),
                "ssd_suffix_tokens": int(job.ssd_suffix_tokens),
                "cache_miss_reason": (
                    None if job.session_cache_hit else job.cache_miss_reason
                ),
                "session_restore_mode": job.effective_restore_mode,
                "ar_batch_shared_prefix_tokens": int(job.shared_prefix_tokens),
                "ar_batch_shared_prefix_prefill_s": float(
                    job.shared_prefix_prefill_s
                ),
                "ar_batch_shared_prefix_snapshot_s": float(
                    job.shared_prefix_snapshot_s
                ),
                "ar_batch_prompt_prepare_s": float(job.prompt_prepare_s),
            }
        )
        job.future.set_result(
            {
                "text": text,
                "tokens": list(job.tokens),
                "stats": stats,
                "prompt_tokens": len(job.prompt_ids),
                "completion_tokens": completion_tokens,
                "elapsed_s": elapsed_s,
                "tok_s": decode_tok_s,
                "end_to_end_tok_s": (
                    completion_tokens / elapsed_s if elapsed_s > 0 else 0.0
                ),
                "_final_state": None,
                "_token_times": list(job.token_times),
                "_generation_limits": dict(job.generation_limits),
                "finish_reason": finish_reason,
            }
        )

    def _fail_all(self, exc: BaseException) -> None:
        with self._condition:
            pending = list(self._pending)
            active = list(self._active.values())
            self._pending.clear()
            self._active.clear()
            self._last_error = f"{type(exc).__name__}: {exc}"
        for job in [*pending, *active]:
            if not job.future.done():
                job.future.set_exception(exc)

    def _pump(self) -> None:
        import mlx.core as mx
        from mlx_lm.generate import BatchGenerator

        config = _scheduler_config_from_args(self.state.args)
        config_dict = config.to_dict()
        generator_max_tokens = int(
            getattr(self.state.args, "max_response_tokens", None)
            or getattr(self.state, "context_window", 0)
            or 1
        )
        generator = BatchGenerator(
            self.state.runtime.model,
            max_tokens=max(1, generator_max_tokens),
            stop_tokens=self._stop_sequences(),
            completion_batch_size=max(1, int(config_dict["decode_batch_max"])),
            prefill_batch_size=max(1, min(2, int(config_dict["decode_batch_max"]))),
            prefill_step_size=max(1, int(config_dict["prefill_chunk_tokens"])),
        )
        idle_deadline_s: float | None = None
        try:
            while True:
                self._wait_for_initial_cohort(config_dict)
                self._admit_pending(generator, config_dict)
                with self._condition:
                    active_count = len(self._active)
                    pending_count = len(self._pending)
                if active_count == 0 and pending_count == 0:
                    if idle_deadline_s is None:
                        idle_deadline_s = time.perf_counter() + (
                            max(0.0, float(config_dict["batch_wait_ms"])) / 1000.0
                        )
                    if time.perf_counter() >= idle_deadline_s:
                        return
                    time.sleep(0.001)
                    continue
                idle_deadline_s = None
                self._remove_cancelled_active(generator)
                prompt_responses, generation_responses = generator.next()
                for response in prompt_responses:
                    with self._condition:
                        job = self._active.get(int(response.uid))
                    if job is None:
                        continue
                    if bool(getattr(response, "end_of_prompt", False)):
                        done_s = time.perf_counter()
                        job.prefill_done_s = done_s
                        elapsed = max(
                            0.0,
                            done_s - (job.prefill_started_s or job.created_s),
                        )
                        new_prefill_tokens = max(
                            0,
                            int(len(job.prompt_ids)) - int(job.cached_tokens),
                        )
                        prefill_tok_s = (
                            new_prefill_tokens / elapsed
                            if elapsed > 0 and job.prompt_ids
                            else None
                        )
                        job.emit_prefill(
                            {
                                "phase": "completed",
                                "tokens_total": int(len(job.prompt_ids)),
                                "new_prefill_tokens": new_prefill_tokens,
                                "cached_tokens": int(job.cached_tokens),
                                "elapsed_s": elapsed,
                                "prompt_eval_time_s": elapsed,
                                "prefill_tok_s": prefill_tok_s,
                                "prefill_compute_tok_s": prefill_tok_s,
                                "prefill_wall_tok_s": prefill_tok_s,
                                "cache_hit": bool(job.session_cache_hit),
                                "scheduler_lane": "ar_batch",
                                "request_id": job.request_id,
                            }
                        )
                batch_size = len(generation_responses)
                if batch_size:
                    with self._condition:
                        self._last_batch_size = int(batch_size)
                    scheduler = getattr(self.state, "model_scheduler", None)
                    if scheduler is not None and hasattr(
                        scheduler, "record_batch_step"
                    ):
                        scheduler.record_batch_step(
                            size=batch_size,
                            batch_key="ar_batch.decode",
                        )
                for response in generation_responses:
                    uid = int(response.uid)
                    with self._condition:
                        job = self._active.get(uid)
                        active_size = max(1, len(self._active))
                    if job is None:
                        continue
                    if job.cancel_requested():
                        with self._condition:
                            self._active.pop(uid, None)
                        try:
                            generator.remove([uid])
                        except BaseException:
                            pass
                        if not job.future.done():
                            job.future.set_exception(self._cancelled_error(job))
                        continue
                    job.max_batch_size_observed = max(
                        job.max_batch_size_observed,
                        active_size,
                        batch_size,
                    )
                    try:
                        job.emit_token(int(response.token))
                    except BaseException as exc:
                        with self._condition:
                            self._active.pop(uid, None)
                        if not job.future.done():
                            job.future.set_exception(exc)
                        continue
                    finish_reason = getattr(response, "finish_reason", None)
                    if finish_reason is not None:
                        with self._condition:
                            self._active.pop(uid, None)
                        self._complete_job(job, finish_reason=str(finish_reason))
                mx.eval([])
        except BaseException as exc:
            self._fail_all(exc)
            raise
        finally:
            try:
                generator.close()
            except BaseException:
                pass
            with self._condition:
                self._pump_scheduled = False
                if self._pending:
                    self._pump_scheduled = True
                    _submit_foreground_model_work(
                        self.state,
                        self._pump,
                        batch_key="ar_batch.pump",
                    )
                self._condition.notify_all()

    def _remove_cancelled_active(self, generator: Any) -> None:
        with self._condition:
            cancelled = [
                (uid, job)
                for uid, job in list(self._active.items())
                if job.cancel_requested()
            ]
            for uid, _job in cancelled:
                self._active.pop(uid, None)
        if not cancelled:
            return
        try:
            generator.remove([uid for uid, _job in cancelled])
        except BaseException:
            pass
        for _uid, job in cancelled:
            if not job.future.done():
                job.future.set_exception(self._cancelled_error(job))


def _submit_idle_postcommit_model_work(
    state: Any,
    fn: Callable[..., Any],
    *args: Any,
    batch_key: str | None = None,
    **kwargs: Any,
) -> Any:
    scheduler = getattr(state, "model_scheduler", None)
    if scheduler is not None and hasattr(scheduler, "submit_idle_postcommit"):
        return scheduler.submit_idle_postcommit(
            fn, *args, batch_key=batch_key, **kwargs
        )
    executor = getattr(state, "postcommit_executor", None)
    if executor is None:
        executor = getattr(state, "generation_executor", None)
    if executor is None:
        raise RuntimeError("state has no idle postcommit executor")
    return executor.submit(fn, *args, **kwargs)


def _foreground_model_work_pending(state: Any) -> bool:
    scheduler = getattr(state, "model_scheduler", None)
    if scheduler is not None and hasattr(scheduler, "foreground_pending_or_active"):
        try:
            return bool(scheduler.foreground_pending_or_active())
        except BaseException:
            pass
    if scheduler is not None and hasattr(scheduler, "has_foreground_pending"):
        return bool(scheduler.has_foreground_pending())
    return False


LOCALHOST_BINDS = {"", "127.0.0.1", "::1", "localhost"}


def _is_localhost_bind(host: str | None) -> bool:
    return str(host or "").strip().lower().strip("[]") in LOCALHOST_BINDS


def validate_server_security_args(args: argparse.Namespace) -> None:
    if not _is_localhost_bind(getattr(args, "host", None)) and not getattr(
        args, "api_key", None
    ):
        raise SystemExit("--api-key or --api-key-file is required when --host is not localhost")
    if int(getattr(args, "stream_interval", 1)) < 1:
        raise SystemExit("--stream-interval must be >= 1")
    if int(getattr(args, "rate_limit", 0)) < 0:
        raise SystemExit("--rate-limit must be >= 0")
    if int(getattr(args, "warmup_tokens", 0)) < 0:
        raise SystemExit("--warmup-tokens must be >= 0")


def _request_api_key(request: Request) -> str | None:
    auth_header = request.headers.get("authorization", "")
    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() == "bearer" and token:
        return token
    api_key = request.headers.get("x-api-key")
    if api_key:
        return api_key
    cookies = getattr(request, "cookies", {}) or {}
    cookie_key = cookies.get(_BROWSER_AUTH_COOKIE)
    return cookie_key or None


def _request_browser_auth_query_key(request: Request) -> str | None:
    query_params = getattr(request, "query_params", {}) or {}
    return query_params.get(_BROWSER_AUTH_QUERY_PARAM) or None


def _request_is_authorized(request: Request, configured_api_key: str | None) -> bool:
    if not configured_api_key:
        return True
    candidate = _request_api_key(request)
    return bool(candidate and secrets.compare_digest(candidate, configured_api_key))


def _request_is_browser_auth_bootstrap(
    request: Request, configured_api_key: str | None
) -> bool:
    if not configured_api_key:
        return True
    if request.url.path != _BROWSER_AUTH_PATH:
        return False
    candidate = _request_browser_auth_query_key(request)
    return bool(candidate and secrets.compare_digest(candidate, configured_api_key))


def _rate_limit_key(request: Request) -> str:
    api_key = _request_api_key(request)
    if api_key:
        return "key:" + hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    client = request.client.host if request.client is not None else "unknown"
    return f"client:{client}"


class _RateLimiter:
    def __init__(self, requests_per_minute: int | None) -> None:
        self.requests_per_minute = max(0, int(requests_per_minute or 0))
        self._lock = Lock()
        self._events: dict[str, list[float]] = {}

    @property
    def enabled(self) -> bool:
        return self.requests_per_minute > 0

    def check(self, key: str, *, now: float | None = None) -> tuple[bool, int]:
        if not self.enabled:
            return True, 0
        timestamp = time.monotonic() if now is None else float(now)
        window_start = timestamp - 60.0
        with self._lock:
            events = [item for item in self._events.get(key, []) if item > window_start]
            if len(events) >= self.requests_per_minute:
                retry_after = max(1, int(60.0 - (timestamp - events[0])))
                self._events[key] = events
                return False, retry_after
            events.append(timestamp)
            self._events[key] = events
            return True, 0


def _resolve_context_window(tokenizer: Any, model_path: str) -> int:
    candidates: list[int] = []
    tokenizer_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_max, int):
        candidates.append(tokenizer_max)

    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
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

    sane = [value for value in candidates if 0 < value <= 1_000_000]
    return max(sane) if sane else 262_144


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _anthropic_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                block_type = str(item.get("type") or "")
                if block_type == "text" or "text" in item:
                    parts.append(str(item.get("text", "")))
                elif block_type == "thinking":
                    parts.append(str(item.get("thinking", "")))
                elif block_type == "tool_use":
                    continue
                elif block_type == "tool_result":
                    parts.append(_anthropic_content_to_text(item.get("content")))
                else:
                    parts.append(json.dumps(item, sort_keys=True))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(content, dict):
        if content.get("type") == "text" or "text" in content:
            return str(content.get("text", ""))
        return json.dumps(content, sort_keys=True)
    return str(content)


_ANTHROPIC_SERVER_SIDE_TOOL_TYPE_PREFIXES = (
    "web_search_",
    "code_execution_",
    "bash_",
    "text_editor_",
    "computer_",
)


def _anthropic_content_blocks(content: Any) -> list[Any]:
    if content is None:
        return []
    if isinstance(content, list):
        return content
    return [{"type": "text", "text": content}]


def _anthropic_filter_system_text(text: str) -> str:
    lines = []
    for line in str(text).splitlines():
        if line.strip().lower().startswith("x-anthropic-billing-header:"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _anthropic_system_to_text(system: Any) -> str:
    if system is None:
        return ""
    if isinstance(system, list):
        parts: list[str] = []
        for block in system:
            if isinstance(block, dict):
                text = _anthropic_filter_system_text(
                    _anthropic_content_to_text([block])
                )
            else:
                text = _anthropic_filter_system_text(str(block))
            if text:
                parts.append(text)
        return "\n\n".join(parts).strip()
    return _anthropic_filter_system_text(_anthropic_content_to_text(system))


def _anthropic_tool_use_to_openai(block: dict[str, Any]) -> dict[str, Any]:
    name = str(block.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="tool_use block is missing a name")
    call_id = str(block.get("id") or f"call_{uuid.uuid4().hex[:24]}")
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": _json_object_string(
                block.get("input", {}),
                context=f"tool_use '{name}'",
            ),
        },
    }


def _anthropic_tool_result_id(block: dict[str, Any]) -> str:
    return str(block.get("tool_use_id") or block.get("id") or "").strip()


def _anthropic_message_to_chat_messages(
    message: AnthropicMessage,
) -> list[ChatMessage]:
    role = str(message.role or "").strip().lower()
    if role == "assistant":
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in _anthropic_content_blocks(message.content):
            if isinstance(block, dict):
                block_type = str(block.get("type") or "")
                if block_type == "tool_use":
                    tool_calls.append(_anthropic_tool_use_to_openai(block))
                elif block_type == "thinking":
                    thinking_parts.append(str(block.get("thinking") or ""))
                else:
                    text_parts.append(_anthropic_content_to_text([block]))
            else:
                text_parts.append(str(block))
        kwargs: dict[str, Any] = {}
        if thinking := "".join(thinking_parts).strip():
            kwargs["reasoning_content"] = thinking
        return [
            ChatMessage(
                role="assistant",
                content="".join(text_parts),
                tool_calls=tool_calls or None,
                **kwargs,
            )
        ]
    if role == "user":
        messages: list[ChatMessage] = []
        text_parts: list[str] = []
        for block in _anthropic_content_blocks(message.content):
            if (
                isinstance(block, dict)
                and str(block.get("type") or "") == "tool_result"
            ):
                if text_parts:
                    messages.append(
                        ChatMessage(role="user", content="".join(text_parts))
                    )
                    text_parts = []
                tool_call_id = _anthropic_tool_result_id(block)
                if not tool_call_id:
                    raise HTTPException(
                        status_code=400,
                        detail="tool_result block is missing tool_use_id",
                    )
                messages.append(
                    ChatMessage(
                        role="tool",
                        tool_call_id=tool_call_id,
                        content=_anthropic_content_to_text(block.get("content")),
                    )
                )
            elif isinstance(block, dict):
                text_parts.append(_anthropic_content_to_text([block]))
            else:
                text_parts.append(str(block))
        if text_parts or not messages:
            messages.append(ChatMessage(role="user", content="".join(text_parts)))
        return messages
    return [
        ChatMessage(
            role=role or "user", content=_anthropic_content_to_text(message.content)
        )
    ]


def _anthropic_is_server_side_tool(tool: dict[str, Any]) -> bool:
    tool_type = str(tool.get("type") or "").strip()
    return bool(tool_type) and any(
        tool_type.startswith(prefix)
        for prefix in _ANTHROPIC_SERVER_SIDE_TOOL_TYPE_PREFIXES
    )


def _anthropic_tools_to_openai(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    converted: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise HTTPException(
                status_code=400, detail=f"tools[{index}] must be an object"
            )
        if _anthropic_is_server_side_tool(tool):
            LOGGER.info(
                "dropping unsupported Anthropic server-side tool type=%s name=%s",
                tool.get("type"),
                tool.get("name"),
            )
            continue
        name = str(tool.get("name") or "").strip()
        if not name:
            raise HTTPException(
                status_code=400, detail=f"tools[{index}] is missing a name"
            )
        input_schema = tool.get("input_schema")
        if input_schema is None:
            input_schema = tool.get("parameters")
        parameters = (
            input_schema if isinstance(input_schema, dict) else {"type": "object"}
        )
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(tool.get("description") or ""),
                    "parameters": parameters,
                },
            }
        )
    return converted or None


def _anthropic_tool_choice_to_openai(tool_choice: Any) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        value = tool_choice.strip().lower()
        if value in {"auto", "none"}:
            return value
        if value in {"any", "required"}:
            return "required"
        return tool_choice
    if isinstance(tool_choice, dict):
        value = (
            str(tool_choice.get("type") or tool_choice.get("mode") or "")
            .strip()
            .lower()
        )
        if value in {"auto", "none"}:
            return value
        if value in {"any", "required"}:
            return "required"
        if value == "tool":
            name = str(tool_choice.get("name") or "").strip()
            if not name:
                raise HTTPException(
                    status_code=400,
                    detail="tool_choice tool must include a name",
                )
            return {"type": "function", "function": {"name": name}}
    return tool_choice


def _anthropic_thinking_to_enable_thinking(thinking: Any) -> bool | None:
    if not isinstance(thinking, dict):
        return None
    mode = str(thinking.get("type") or "").strip().lower()
    if mode == "enabled":
        return True
    if mode == "disabled":
        return False
    return None


def _anthropic_to_chat_request(
    request: AnthropicMessagesRequest,
) -> ChatCompletionRequest:
    messages: list[ChatMessage] = []
    system_text = _anthropic_system_to_text(request.system)
    if system_text:
        messages.append(ChatMessage(role="system", content=system_text))
    for message in request.messages:
        messages.extend(_anthropic_message_to_chat_messages(message))
    enable_thinking = _anthropic_thinking_to_enable_thinking(request.thinking)
    return ChatCompletionRequest(
        model=request.model,
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        tools=_anthropic_tools_to_openai(request.tools),
        tool_choice=_anthropic_tool_choice_to_openai(request.tool_choice),
        stop=request.stop_sequences,
        metadata=request.metadata,
        enable_thinking=enable_thinking,
        depth=request.depth,
        draft_block_size=request.draft_block_size,
        gemma_draft_block_size=request.gemma_draft_block_size,
        generation_mode=request.generation_mode,
        stream=False,
    )


def _anthropic_tool_input_from_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _anthropic_tool_use_block(tool_call: dict[str, Any]) -> dict[str, Any] | None:
    function = tool_call.get("function") if isinstance(tool_call, dict) else None
    if not isinstance(function, dict):
        return None
    name = str(function.get("name") or "").strip()
    if not name:
        return None
    return {
        "type": "tool_use",
        "id": str(tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}"),
        "name": name,
        "input": _anthropic_tool_input_from_arguments(function.get("arguments")),
    }


def _anthropic_stop_reason(
    finish_reason: Any,
    *,
    has_tool_calls: bool,
    matched_stop: str | None = None,
) -> str:
    if has_tool_calls or finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    if matched_stop:
        # A client stop_sequences match must surface as stop_sequence per
        # the Anthropic wire contract, not as a natural end_turn (QA-117).
        return "stop_sequence"
    if finish_reason == "stop":
        return "end_turn"
    return "end_turn"


def _matched_stop_sequence(openai_payload: dict[str, Any]) -> str | None:
    stats = openai_payload.get("mtplx_stats")
    if isinstance(stats, dict):
        matched = stats.get("stop_sequence_matched")
        if isinstance(matched, str) and matched:
            return matched
    return None


def _anthropic_payload_from_openai(openai_payload: dict[str, Any]) -> dict[str, Any]:
    choices = openai_payload.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    text = str(message.get("content") or choice.get("text") or "")
    tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
    content: list[dict[str, Any]] = []
    reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
    if isinstance(reasoning, str) and reasoning.strip():
        content.append(
            {
                "type": "thinking",
                "thinking": reasoning,
                "signature": "mtplx-reasoning",
            }
        )
    if text:
        content.append({"type": "text", "text": text})
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            block = _anthropic_tool_use_block(tool_call)
            if block is not None:
                content.append(block)
    if not content:
        content = [{"type": "text", "text": ""}]
    usage = openai_payload.get("usage") or {}
    finish_reason = choice.get("finish_reason") or "stop"
    matched_stop = _matched_stop_sequence(openai_payload)
    stop_reason = _anthropic_stop_reason(
        finish_reason,
        has_tool_calls=any(block.get("type") == "tool_use" for block in content),
        matched_stop=matched_stop,
    )
    return {
        "id": "msg_" + uuid.uuid4().hex,
        "type": "message",
        "role": "assistant",
        "model": openai_payload.get("model"),
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": matched_stop,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
        "mtplx_stats": openai_payload.get("mtplx_stats"),
    }


def _anthropic_sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


async def _iter_sse_data(body_iterator: Any):
    buffer = ""
    async for raw_chunk in body_iterator:
        chunk = (
            raw_chunk.decode("utf-8")
            if isinstance(raw_chunk, bytes)
            else str(raw_chunk)
        )
        buffer += chunk
        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            data_lines = [
                line.removeprefix("data:").strip()
                for line in frame.splitlines()
                if line.startswith("data:")
            ]
            if data_lines:
                yield "\n".join(data_lines)
    if buffer.strip():
        data_lines = [
            line.removeprefix("data:").strip()
            for line in buffer.splitlines()
            if line.startswith("data:")
        ]
        if data_lines:
            yield "\n".join(data_lines)


async def _anthropic_stream_from_openai_sse(body_iterator: Any, *, model: str):
    message_id = "msg_" + uuid.uuid4().hex
    next_block_index = 0
    active_text_index: int | None = None
    active_thinking_index: int | None = None
    opened_any_block = False
    opened_tool_block = False
    stop_reason = "end_turn"
    usage = {"input_tokens": 0, "output_tokens": 0}
    mtplx_stats: dict[str, Any] | None = None
    tool_blocks: dict[int, dict[str, Any]] = {}

    yield _anthropic_sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": usage,
            },
        },
    )

    def start_content_block(index: int, block: dict[str, Any]) -> str:
        return _anthropic_sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": block,
            },
        )

    def stop_content_block(index: int) -> str:
        return _anthropic_sse(
            "content_block_stop", {"type": "content_block_stop", "index": index}
        )

    try:
        async for data in _iter_sse_data(body_iterator):
            if data == "[DONE]":
                break
            try:
                payload = json.loads(data)
            except json.JSONDecodeError as exc:
                yield _anthropic_sse(
                    "error",
                    {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": f"failed to parse upstream SSE chunk: {exc}",
                        },
                    },
                )
                return
            if "error" in payload:
                error = payload.get("error") or {}
                yield _anthropic_sse(
                    "error",
                    {
                        "type": "error",
                        "error": {
                            "type": str(error.get("type") or "api_error"),
                            "message": str(error.get("message") or error),
                        },
                    },
                )
                return
            if payload.get("usage"):
                upstream_usage = payload.get("usage") or {}
                usage = {
                    "input_tokens": int(upstream_usage.get("prompt_tokens") or 0),
                    "output_tokens": int(upstream_usage.get("completion_tokens") or 0),
                }
            if payload.get("mtplx_stats") is not None:
                mtplx_stats = payload.get("mtplx_stats")
            for choice in payload.get("choices") or []:
                delta = choice.get("delta") or {}
                reasoning_text = str(delta.get("reasoning_content") or "")
                if reasoning_text:
                    if active_text_index is not None:
                        yield stop_content_block(active_text_index)
                        active_text_index = None
                    if active_thinking_index is None:
                        active_thinking_index = next_block_index
                        next_block_index += 1
                        opened_any_block = True
                        yield start_content_block(
                            active_thinking_index,
                            {
                                "type": "thinking",
                                "thinking": "",
                                "signature": "mtplx-reasoning",
                            },
                        )
                    yield _anthropic_sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": active_thinking_index,
                            "delta": {
                                "type": "thinking_delta",
                                "thinking": reasoning_text,
                            },
                        },
                    )
                text = str(delta.get("content") or "")
                if text:
                    if active_thinking_index is not None:
                        yield stop_content_block(active_thinking_index)
                        active_thinking_index = None
                    if active_text_index is None:
                        active_text_index = next_block_index
                        next_block_index += 1
                        opened_any_block = True
                        yield start_content_block(
                            active_text_index,
                            {"type": "text", "text": ""},
                        )
                    yield _anthropic_sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": active_text_index,
                            "delta": {"type": "text_delta", "text": text},
                        },
                    )
                tool_call_deltas = delta.get("tool_calls")
                if isinstance(tool_call_deltas, list) and tool_call_deltas:
                    if active_text_index is not None:
                        yield stop_content_block(active_text_index)
                        active_text_index = None
                    if active_thinking_index is not None:
                        yield stop_content_block(active_thinking_index)
                        active_thinking_index = None
                    for raw_tool_delta in tool_call_deltas:
                        if not isinstance(raw_tool_delta, dict):
                            continue
                        upstream_index = int(raw_tool_delta.get("index") or 0)
                        state = tool_blocks.setdefault(
                            upstream_index,
                            {
                                "id": None,
                                "name": None,
                                "block_index": None,
                                "pending_arguments": "",
                            },
                        )
                        if raw_tool_delta.get("id"):
                            state["id"] = str(raw_tool_delta.get("id"))
                        function_delta = raw_tool_delta.get("function")
                        arguments_delta = ""
                        if isinstance(function_delta, dict):
                            if function_delta.get("name"):
                                state["name"] = str(function_delta.get("name"))
                            if function_delta.get("arguments") is not None:
                                arguments_delta = str(
                                    function_delta.get("arguments") or ""
                                )
                        if state["block_index"] is None:
                            if not state.get("name"):
                                state["pending_arguments"] += arguments_delta
                                continue
                            state["block_index"] = next_block_index
                            next_block_index += 1
                            opened_any_block = True
                            opened_tool_block = True
                            yield start_content_block(
                                int(state["block_index"]),
                                {
                                    "type": "tool_use",
                                    "id": state.get("id")
                                    or f"call_{uuid.uuid4().hex[:24]}",
                                    "name": state["name"],
                                    "input": {},
                                },
                            )
                            pending = str(state.get("pending_arguments") or "")
                            if pending:
                                yield _anthropic_sse(
                                    "content_block_delta",
                                    {
                                        "type": "content_block_delta",
                                        "index": state["block_index"],
                                        "delta": {
                                            "type": "input_json_delta",
                                            "partial_json": pending,
                                        },
                                    },
                                )
                                state["pending_arguments"] = ""
                        if arguments_delta:
                            yield _anthropic_sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": state["block_index"],
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": arguments_delta,
                                    },
                                },
                            )
                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    stop_reason = _anthropic_stop_reason(
                        finish_reason,
                        has_tool_calls=opened_tool_block,
                        matched_stop=_matched_stop_sequence(
                            {"mtplx_stats": mtplx_stats}
                        ),
                    )
    finally:
        if hasattr(body_iterator, "aclose"):
            try:
                await body_iterator.aclose()
            except Exception:
                pass

    if active_text_index is not None:
        yield stop_content_block(active_text_index)
    if active_thinking_index is not None:
        yield stop_content_block(active_thinking_index)
    for state in tool_blocks.values():
        block_index = state.get("block_index")
        if block_index is not None:
            yield stop_content_block(int(block_index))
    if not opened_any_block:
        empty_index = next_block_index
        yield start_content_block(empty_index, {"type": "text", "text": ""})
        yield stop_content_block(empty_index)
    delta_payload: dict[str, Any] = {
        "type": "message_delta",
        "delta": {
            "stop_reason": stop_reason,
            "stop_sequence": _matched_stop_sequence({"mtplx_stats": mtplx_stats}),
        },
        "usage": {"output_tokens": usage["output_tokens"]},
    }
    if mtplx_stats is not None:
        delta_payload["mtplx_stats"] = mtplx_stats
    yield _anthropic_sse("message_delta", delta_payload)
    yield _anthropic_sse("message_stop", {"type": "message_stop"})


def _strip_stats_footer(text: str) -> str:
    marker_index = text.rfind(STATS_FOOTER_MARKER)
    if marker_index < 0:
        return _STATS_FOOTER_RE.sub("", text).rstrip()
    return text[:marker_index].rstrip()


def _details_block_to_think(match: re.Match[str]) -> str:
    block = match.group(0)
    block = re.sub(
        r"<summary\b[^>]*>.*?</summary>", "", block, flags=re.IGNORECASE | re.DOTALL
    )
    block = re.sub(r"</?details\b[^>]*>", "", block, flags=re.IGNORECASE | re.DOTALL)
    block = re.sub(r"<br\s*/?>", "\n", block, flags=re.IGNORECASE)
    block = re.sub(r"<[^>]+>", "", block)
    lines = []
    for line in html.unescape(block).splitlines():
        stripped = line.strip()
        if stripped.startswith(">"):
            stripped = stripped[1:].lstrip()
        if stripped:
            lines.append(stripped)
        elif lines and lines[-1]:
            lines.append("")
    reasoning = "\n".join(lines).strip()
    if not reasoning:
        return ""
    return f"{THINK_OPEN}\n{reasoning}\n{THINK_CLOSE}"


def _normalize_openwebui_reasoning_details(text: str) -> str:
    return _REASONING_DETAILS_RE.sub(_details_block_to_think, text)


def _strip_assistant_history_baggage(text: str) -> str:
    """Remove UI-only metadata before feeding assistant turns back to Qwen."""
    text = _strip_stats_footer(text)
    previous = None
    while previous != text:
        previous = text
        text = _REASONING_DETAILS_RE.sub("", text)
        text = _REASONING_TAG_RE.sub("", text)
        text = _BEGIN_END_THOUGHT_RE.sub("", text)
    text = _REASONING_DETAILS_UNCLOSED_RE.sub("", text)
    return text.strip()


_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_NAMESPACED_TOOL_CALL_BLOCK_RE = re.compile(
    r"<([A-Za-z_][\w.-]*):tool_call>\s*(.*?)\s*</\1:tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_TOOL_FUNCTION_BLOCK_RE = re.compile(
    r"^\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*$",
    re.IGNORECASE | re.DOTALL,
)
_TOOL_FUNCTION_START_RE = re.compile(
    r"^\s*<function=([^>\s]+)>\s*(.*?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_TOOL_PARAMETER_BLOCK_RE = re.compile(
    r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>",
    re.IGNORECASE | re.DOTALL,
)
_TOOL_PARAMETER_TYPE_WRAPPER_RE = re.compile(
    r"^\s*<([A-Za-z_][\w.-]*)>\s*(.*?)\s*</\1>\s*$",
    re.IGNORECASE | re.DOTALL,
)
_INVOKE_TOOL_BLOCK_RE = re.compile(
    r'^\s*<invoke\s+name="([^"]+)">\s*(.*?)\s*</invoke>\s*$',
    re.IGNORECASE | re.DOTALL,
)
_INVOKE_PARAMETER_BLOCK_RE = re.compile(
    r'<parameter\s+name="([^"]+)">\s*(.*?)\s*</parameter>',
    re.IGNORECASE | re.DOTALL,
)
_BRACKET_TOOL_CALL_RE = re.compile(
    r"\[(?:Calling tool|Tool call):\s*([A-Za-z_][\w.-]*)(?:\(({.*?})\))?\]",
    re.IGNORECASE | re.DOTALL,
)
_BRACKET_TOOL_PREFIXES = ("[Calling tool:", "[Tool call:")
_NAMESPACED_TOOL_CALL_START_RE = re.compile(
    r"<[A-Za-z_][\w.-]*:tool_call",
    re.IGNORECASE,
)
_MTPLX_TOOL_CONTRACT_SENTINEL = "MTPLX tool contract:"
_MTPLX_NO_TOOL_CONTRACT_SENTINEL = "MTPLX direct reply turn:"
_MTPLX_SIMPLE_CHAT_SYSTEM_PROMPT = (
    "You are MTPLX. Answer the latest user message directly and naturally."
)
_MTPLX_STEP_LANGUAGE_POLICY_SENTINEL = "MTPLX Step language policy:"
_MTPLX_STEP_LANGUAGE_POLICY = (
    f"{_MTPLX_STEP_LANGUAGE_POLICY_SENTINEL} Use English by default. If the "
    "latest user message is clearly written in another natural language, reply "
    "in that language; otherwise reply in English. Short greetings such as hi, "
    "hey, hello, yo, and how are you are English. Never answer in Chinese for "
    "English or ambiguous input."
)
_MTPLX_READ_ONLY_FORCE_ANSWER_SENTINEL = "MTPLX read-only answer turn:"
_MTPLX_READ_ONLY_FORCE_ANSWER_USER_SENTINEL = (
    "MTPLX read-only final answer instruction:"
)
_MTPLX_READ_ONLY_FORCE_ANSWER_STREAM_MARKER = "<mtplx_final_answer>"
_MTPLX_READ_ONLY_FORCE_ANSWER_STREAM_MARKER_RE = re.compile(
    r"<\s*/?\s*mtplx_final_answer\s*/?\s*>",
    re.IGNORECASE,
)
_MTPLX_PI_CONVERGENCE_SENTINEL = "MTPLX Pi convergence turn:"
_MTPLX_PI_CONVERGENCE_USER_SENTINEL = "MTPLX Pi convergence instruction:"
_MTPLX_CODING_AGENT_TAIL_SENTINEL = "MTPLX coding-agent tool protocol reminder:"
_MTPLX_TOOL_RESULT_CONTINUATION_SENTINEL = "MTPLX tool-result continuation:"
_MTPLX_TOOL_RESULT_CONTINUATION_BLOCK_RE = re.compile(
    r"\s*<\s*mtplx_tool_result_continuation\s*>.*?</\s*mtplx_tool_result_continuation\s*>\s*",
    re.IGNORECASE | re.DOTALL,
)
_MTPLX_TOOL_RESULT_CONTINUATION_TAG_RE = re.compile(
    r"</?\s*mtplx_tool_result_continuation\s*>\s*",
    re.IGNORECASE,
)
_OPENAI_BRIDGE_POLICY_VERSION = (
    "omlx_style:preserve_history:parse_at_completion:tool_digest:v4"
)
_MTPLX_TOOL_CONTRACT_POLICY_VERSION = (
    "soft_schema_contract:native_xml:targeted_reads:post_tool_continue:agent_tail:v11"
)
_MTPLX_NO_TOOL_CONTRACT_POLICY_VERSION = "no_tool_direct_reply:v1"
_MTPLX_OPENCODE_AGENT_CONTRACT_PROFILE = "opencode_agent"
_MTPLX_READ_ONLY_FORCE_ANSWER_POLICY_VERSION = "read_only_force_answer:v1"
_MTPLX_PI_CONVERGENCE_POLICY_VERSION = "pi_convergence:v1"
_NATIVE_TOOL_PROMPT_POLICY_VERSION = "native_template_tools:agent_tail:v7"
_COMPACT_TOOL_PROMPT_POLICY_VERSION = "compact_tool_contract:schema_free:v1"
_TOOL_PROMPT_MODE_HYBRID = "hybrid"
_TOOL_PROMPT_MODE_NATIVE = "native"
_TOOL_PROMPT_MODE_COMPACT = "compact"
_TOOL_PROMPT_MODES = {
    _TOOL_PROMPT_MODE_HYBRID,
    _TOOL_PROMPT_MODE_NATIVE,
    _TOOL_PROMPT_MODE_COMPACT,
}
_TOOL_CONTRACT_AGENT_CLIENT_HINTS = ("opencode", "pi", "hermes")
_TOOL_PROMPT_MODE_REQUEST_HEADERS = (
    "x-mtplx-tool-prompt-mode",
    "X-MTPLX-Tool-Prompt-Mode",
)
_CHAT_TEMPLATE_PROFILE_LOCAL = "local_qwen36"
_CHAT_TEMPLATE_PROFILE_FROGGERIC = "froggeric_v19"
_CHAT_TEMPLATE_PROFILE_CUSTOM = "custom"
_CHAT_TEMPLATE_PROFILE_TOKENIZER = "tokenizer"
_CHAT_TEMPLATE_PROFILES = {
    _CHAT_TEMPLATE_PROFILE_LOCAL,
    _CHAT_TEMPLATE_PROFILE_FROGGERIC,
    _CHAT_TEMPLATE_PROFILE_TOKENIZER,
}
_TOOL_PARSE_COUNTER_KEYS = (
    "tool_parse_success",
    "tool_parse_fallback",
    "unknown_tool_name",
    "malformed_tool_call",
    "unclosed_tool_call",
    "tool_stream_xml_started",
    "tool_stream_json_buffered",
    "tool_template_fallback",
    "android_studio_request_detected",
    "openai_error_response",
)


def _tool_protocol_error(message: str) -> HTTPException:
    return HTTPException(status_code=422, detail=f"malformed tool_call: {message}")


def _normalize_tool_prompt_mode(value: Any, *, default: str = _TOOL_PROMPT_MODE_HYBRID) -> str:
    mode = str(value or default).strip().lower()
    if mode not in _TOOL_PROMPT_MODES:
        allowed = ", ".join(sorted(_TOOL_PROMPT_MODES))
        raise ValueError(f"tool_prompt_mode must be one of: {allowed}")
    return mode


def _tool_prompt_mode_from_args(args: argparse.Namespace) -> str:
    return _normalize_tool_prompt_mode(
        getattr(args, "tool_prompt_mode", None),
        default=os.environ.get("MTPLX_TOOL_PROMPT_MODE", _TOOL_PROMPT_MODE_HYBRID),
    )


def _tool_contract_active_for_mode(
    *,
    tools_active: bool,
    tool_prompt_mode: str,
) -> bool:
    return bool(
        tools_active
        and tool_prompt_mode in {_TOOL_PROMPT_MODE_HYBRID, _TOOL_PROMPT_MODE_COMPACT}
    )


def _template_tools_for_prompt_mode(
    tools: list[dict[str, Any]] | None,
    *,
    tool_prompt_mode: str,
) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    if _normalize_tool_prompt_mode(tool_prompt_mode) == _TOOL_PROMPT_MODE_COMPACT:
        return None
    return tools


def _tool_prompt_policy_version(*, tools_active: bool, tool_prompt_mode: str) -> str:
    return _tool_prompt_policy_version_for_request(
        tools_active=tools_active,
        tool_prompt_mode=tool_prompt_mode,
        no_tools_contract_active=False,
    )


def _tool_prompt_policy_version_for_request(
    *,
    tools_active: bool,
    tool_prompt_mode: str,
    no_tools_contract_active: bool,
    read_only_force_answer_contract_active: bool = False,
    pi_convergence_contract_active: bool = False,
) -> str:
    if read_only_force_answer_contract_active and not tools_active:
        return _MTPLX_READ_ONLY_FORCE_ANSWER_POLICY_VERSION
    if no_tools_contract_active and not tools_active:
        return _MTPLX_NO_TOOL_CONTRACT_POLICY_VERSION
    if not tools_active:
        return "none"
    if tool_prompt_mode == _TOOL_PROMPT_MODE_NATIVE:
        base = _NATIVE_TOOL_PROMPT_POLICY_VERSION
    elif tool_prompt_mode == _TOOL_PROMPT_MODE_COMPACT:
        base = _COMPACT_TOOL_PROMPT_POLICY_VERSION
    else:
        base = _MTPLX_TOOL_CONTRACT_POLICY_VERSION
    if pi_convergence_contract_active:
        return f"{base}+{_MTPLX_PI_CONVERGENCE_POLICY_VERSION}"
    return base


def _normalize_chat_template_profile(value: Any) -> str:
    profile = str(value or _CHAT_TEMPLATE_PROFILE_LOCAL).strip().lower()
    if profile not in _CHAT_TEMPLATE_PROFILES:
        allowed = ", ".join(sorted(_CHAT_TEMPLATE_PROFILES))
        raise ValueError(f"chat_template_profile must be one of: {allowed}")
    return profile


def _chat_template_profile_path(profile: str) -> Path | None:
    if profile == _CHAT_TEMPLATE_PROFILE_FROGGERIC:
        return ROOT / "templates" / "qwen36_froggeric_v19" / "chat_template.jinja"
    return None


def _apply_chat_template_profile(
    tokenizer: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    explicit_path = str(getattr(args, "chat_template_path", "") or "").strip()
    requested_profile = _normalize_chat_template_profile(
        getattr(args, "chat_template_profile", None)
    )
    if explicit_path:
        profile = _CHAT_TEMPLATE_PROFILE_CUSTOM
        path = Path(explicit_path).expanduser()
    else:
        profile = requested_profile
        path = _chat_template_profile_path(profile)

    report: dict[str, Any] = {
        "profile": profile,
        "source": "tokenizer",
        "path": None,
        "applied": False,
    }
    if path is None:
        return report
    if not path.exists():
        raise RuntimeError(f"chat template profile file is missing: {path}")
    template = path.read_text(encoding="utf-8")
    setattr(tokenizer, "chat_template", template)
    report.update(
        {
            "source": "file",
            "path": str(path),
            "applied": True,
            "sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        }
    )
    return report


def _tool_protocol_reason(exc: BaseException) -> str:
    detail = getattr(exc, "detail", None)
    text = str(detail if detail is not None else exc)
    prefix = "malformed tool_call: "
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def _tool_parse_counter_key(reason: str) -> str:
    lowered = reason.lower()
    if "unknown tool" in lowered:
        return "unknown_tool_name"
    if "unclosed" in lowered:
        return "unclosed_tool_call"
    return "malformed_tool_call"


def _visible_malformed_tool_content(text: str, tokenizer: Any | None) -> str:
    """Preserve visible prose around malformed tool markup, but never the markup."""

    if not text:
        return ""
    tool_filter = OMLXToolCallStreamFilter(tokenizer)
    filtered = tool_filter.feed(text) + tool_filter.finish()
    visible = _strip_orphan_tool_control_markup(filtered).strip()
    if visible and not re.search(r"[A-Za-z0-9]", visible):
        return ""
    return visible


_ORPHAN_TOOL_EXEC_BLOCK_RE = re.compile(
    r"<\s*toolExec\b[^>]*>.*?</\s*toolExec\s*>",
    re.IGNORECASE | re.DOTALL,
)
_ORPHAN_TOOL_CONTROL_TAG_RE = re.compile(
    r"</?\s*(?:tool_call|toolExec|function|parameter|invoke|result|value|type|func)\b[^>\n]*>"
    r"|</<result>",
    re.IGNORECASE,
)
_ORPHAN_TOOL_CONTROL_BARE_NAMES = (
    "tool_call",
    "toolExec",
    "function",
    "parameter",
    "invoke",
    "result",
    "value",
    "type",
    "func",
)
_ORPHAN_TOOL_CONTROL_BARE_TAG_RE = re.compile(
    r"(^|[\r\n])[ \t]*/?(?:"
    + "|".join(re.escape(name) for name in _ORPHAN_TOOL_CONTROL_BARE_NAMES)
    + r")\b(?:[ \t]*=[^>\r\n]*)?>[ \t]*",
    re.IGNORECASE,
)
_ORPHAN_TOOL_CONTROL_INITIAL_TAG_RE = re.compile(
    r"^\s*</?\s*(?:"
    + "|".join(re.escape(name) for name in _ORPHAN_TOOL_CONTROL_BARE_NAMES)
    + r")\b[^>\n]*>",
    re.IGNORECASE,
)
_ORPHAN_TOOL_CONTROL_INITIAL_BARE_RE = re.compile(
    r"^\s*/?(?:"
    + "|".join(re.escape(name) for name in _ORPHAN_TOOL_CONTROL_BARE_NAMES)
    + r")\b(?:[ \t]*=[^>\r\n]*)?>",
    re.IGNORECASE,
)


def _strip_orphan_tool_control_markup(text: str) -> str:
    if not text:
        return text
    text = _ORPHAN_TOOL_CONTROL_BARE_TAG_RE.sub(lambda match: match.group(1), text)
    if "<" in text:
        text = _ORPHAN_TOOL_EXEC_BLOCK_RE.sub("", text)
        text = _MTPLX_TOOL_RESULT_CONTINUATION_BLOCK_RE.sub("", text)
        text = _MTPLX_TOOL_RESULT_CONTINUATION_TAG_RE.sub("", text)
        text = _ORPHAN_TOOL_CONTROL_TAG_RE.sub("", text)
        text = _REASONING_CONTROL_TAG_RE.sub("", text)
    return text


def _has_orphan_tool_control_marker(text: str) -> bool:
    if not text:
        return False
    return bool(
        _ORPHAN_TOOL_CONTROL_TAG_RE.search(text)
        or _ORPHAN_TOOL_CONTROL_BARE_TAG_RE.search(text)
    )


def _looks_like_tool_control_payload_only(text: str) -> bool:
    if not _has_orphan_tool_control_marker(text):
        return False
    visible = _strip_orphan_tool_control_markup(
        text.replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
    ).strip()
    if not visible:
        return True
    if not re.search(r"[A-Za-z]", visible):
        return True
    if "\n" not in visible and not re.search(r"\s", visible) and len(visible) <= 160:
        return True
    return False


def _tool_fed_degenerate_completion_reason(text: str) -> str | None:
    cleaned = _strip_mtplx_internal_continuation_markers(
        _strip_generated_chat_template_sentinels(str(text or ""))
    )
    if not cleaned.strip():
        return "empty_tool_fed_completion"
    if _looks_like_tool_control_payload_only(cleaned):
        return "orphan_tool_control_markup"
    return None


def _initial_orphan_tool_control_state(text: str) -> str:
    """Classify the beginning of streamed text for dangling tool-control residue."""

    if not text:
        return "hold"
    stripped = text.lstrip()
    if not stripped:
        return "hold"
    if (
        _ORPHAN_TOOL_CONTROL_INITIAL_TAG_RE.match(stripped)
        or _ORPHAN_TOOL_CONTROL_INITIAL_BARE_RE.match(stripped)
    ):
        return "orphan"
    lowered = stripped.lower()
    partial_markers = [
        f"</{name.lower()}>"
        for name in _ORPHAN_TOOL_CONTROL_BARE_NAMES
    ] + [
        f"<{name.lower()}"
        for name in _ORPHAN_TOOL_CONTROL_BARE_NAMES
    ]
    if lowered.startswith("<"):
        if any(marker.startswith(lowered) for marker in partial_markers):
            return "hold"
        return "normal"
    first_line = lowered.splitlines()[0]
    for name in (name.lower() for name in _ORPHAN_TOOL_CONTROL_BARE_NAMES):
        if name.startswith(first_line):
            return "hold"
        if first_line.startswith(name):
            rest = first_line[len(name) :]
            if not rest or rest.isspace():
                return "hold"
            rest = rest.lstrip()
            if rest.startswith("=") and ">" not in rest:
                return "hold"
    return "normal"


class _InitialOrphanToolControlStreamGuard:
    """Hold an initial dangling tool fragment until it is safe to emit or drop."""

    def __init__(self) -> None:
        self._buffer = ""
        self._mode = "undecided"
        self.suppressed = False

    def feed(self, text: str) -> str:
        if not text:
            return ""
        if self._mode == "pass":
            return text
        self._buffer += text
        if self._mode == "orphan":
            self.suppressed = True
            return ""
        state = _initial_orphan_tool_control_state(self._buffer)
        if state == "hold":
            return ""
        if state == "orphan":
            self._mode = "orphan"
            self.suppressed = True
            return ""
        self._mode = "pass"
        emitted = self._buffer
        self._buffer = ""
        return emitted

    def finish(self) -> str:
        if not self._buffer:
            return ""
        buffered = self._buffer
        self._buffer = ""
        if self._mode == "orphan":
            self.suppressed = True
            if _looks_like_tool_control_payload_only(buffered):
                return ""
            return _strip_orphan_tool_control_markup(buffered)
        self._mode = "pass"
        return buffered


def _tool_parse_counters_for(state: Any) -> dict[str, int] | None:
    if state is None:
        return None
    counters = getattr(state, "tool_parse_counters", None)
    if not isinstance(counters, dict):
        counters = {key: 0 for key in _TOOL_PARSE_COUNTER_KEYS}
        try:
            setattr(state, "tool_parse_counters", counters)
        except Exception:
            return None
    for key in _TOOL_PARSE_COUNTER_KEYS:
        counters.setdefault(key, 0)
    return counters


def _record_tool_parse_event(
    state: Any,
    *,
    event: str,
    reason: str | None = None,
    response_id: str | None = None,
    stream: bool | None = None,
) -> None:
    counters = _tool_parse_counters_for(state)
    if counters is not None:
        counters[event] = int(counters.get(event, 0) or 0) + 1
        if event in {"unknown_tool_name", "malformed_tool_call", "unclosed_tool_call"}:
            counters["tool_parse_fallback"] = (
                int(counters.get("tool_parse_fallback", 0) or 0) + 1
            )
    if event in {
        "tool_parse_success",
        "tool_template_fallback",
        "tool_stream_xml_started",
        "tool_stream_json_buffered",
        "android_studio_request_detected",
        "openai_error_response",
    } or _server_console_enabled(state):
        return
    try:
        _safe_stdout_print(
            json.dumps(
                {
                    "event": "mtplx_tool_parse_fallback",
                    "response_id": response_id,
                    "stream": bool(stream),
                    "reason": reason,
                    "kind": event,
                },
                ensure_ascii=False,
            )
        )
    except BaseException:
        pass


def _openai_error_type(status_code: int) -> str:
    if status_code == 401:
        return "authentication_error"
    if status_code == 403:
        return "permission_error"
    if status_code == 404:
        return "not_found_error"
    if status_code == 409:
        return "conflict_error"
    if status_code == 422:
        return "invalid_request_error"
    if 400 <= status_code < 500:
        return "invalid_request_error"
    return "server_error"


def _openai_error_content(
    message: str,
    *,
    status_code: int,
    code: str | None = None,
    param: str | None = None,
    error_type: str | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {
        "message": message,
        "type": error_type or _openai_error_type(status_code),
        "code": code or str(status_code),
        "param": param,
    }
    # Structured HTTPException details (restart_required,
    # unknown_settings, ...) ride along additively so clients can
    # render them without parsing a repr out of `message`.
    if detail is not None:
        error["detail"] = detail
    return {"error": error}


def _tool_spec_name(tool: dict[str, Any]) -> str | None:
    function = tool.get("function")
    if isinstance(function, dict):
        name = function.get("name")
    else:
        name = tool.get("name")
    if name is None:
        return None
    text = str(name).strip()
    return text or None


def _tool_names(tools: list[dict[str, Any]] | None) -> list[str]:
    return [name for tool in (tools or []) if (name := _tool_spec_name(tool))]


def _canonical_tool_name_for_model_output(
    name: str,
    tools: list[dict[str, Any]],
) -> str | None:
    raw = str(name or "").strip()
    if not raw:
        return None
    known = _tool_names(tools)
    if raw in known:
        return raw
    casefolded = {tool_name.casefold(): tool_name for tool_name in known}
    if canonical := casefolded.get(raw.casefold()):
        return canonical

    # Qwen sometimes turns OpenCode's `bash` tool into a natural-language
    # "Shell" label after the user asks for shell/terminal work. Keep this
    # mapping narrow so unknown tool names still fail loudly.
    alias_targets = {
        "shell": ("bash",),
        "terminal": ("bash",),
        "sh": ("bash",),
        "run_shell": ("bash",),
        "run_command": ("bash",),
    }
    for candidate in alias_targets.get(raw.casefold(), ()):
        if candidate in known:
            return candidate
    return None


def _tool_json_schema(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function")
    source = function if isinstance(function, dict) else tool
    parameters = source.get("parameters") if isinstance(source, dict) else None
    return parameters if isinstance(parameters, dict) else {}


def _tool_schema_for_name(
    tools: list[dict[str, Any]],
    *,
    tool_name: str,
) -> dict[str, Any]:
    for tool in tools:
        if _tool_spec_name(tool) == tool_name:
            return _tool_json_schema(tool)
    return {}


def _validate_tool_arguments_for_schema(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    tools: list[dict[str, Any]],
    context: str,
) -> None:
    schema = _tool_schema_for_name(tools, tool_name=tool_name)
    required = schema.get("required") if isinstance(schema, dict) else None
    if not isinstance(required, list):
        return
    missing = [
        str(name)
        for name in required
        if isinstance(name, str) and name not in arguments
    ]
    if missing:
        joined = ", ".join(missing)
        raise _tool_protocol_error(
            f"{context} is missing required argument(s): {joined}"
        )
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict):
        return
    if schema.get("additionalProperties") is False:
        extra = sorted(str(name) for name in arguments if name not in properties)
        if extra:
            joined = ", ".join(extra)
            raise _tool_protocol_error(
                f"{context} contains unknown argument(s): {joined}"
            )
    for name, value in arguments.items():
        param_schema = properties.get(name)
        if not isinstance(param_schema, dict):
            continue
        if not _json_schema_value_matches(value, param_schema):
            expected = _schema_type_label(param_schema)
            raise _tool_protocol_error(
                f"{context}.{name} must be {expected}, got {type(value).__name__}"
            )


def _json_schema_value_matches(value: Any, schema: dict[str, Any]) -> bool:
    if not isinstance(schema, dict):
        return True
    if "anyOf" in schema and isinstance(schema["anyOf"], list):
        return any(
            isinstance(item, dict) and _json_schema_value_matches(value, item)
            for item in schema["anyOf"]
        )
    if "oneOf" in schema and isinstance(schema["oneOf"], list):
        return any(
            isinstance(item, dict) and _json_schema_value_matches(value, item)
            for item in schema["oneOf"]
        )
    enum = schema.get("enum")
    if isinstance(enum, list) and enum and value not in enum:
        return False
    schema_types = _schema_type_names(schema)
    if not schema_types:
        return True
    for schema_type in schema_types:
        if schema_type == "null" and value is None:
            return True
        if schema_type == "string" and isinstance(value, str):
            return True
        if schema_type == "boolean" and isinstance(value, bool):
            return True
        if (
            schema_type == "integer"
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            return True
        if (
            schema_type == "number"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            return True
        if schema_type == "array" and isinstance(value, list):
            return True
        if schema_type == "object" and isinstance(value, dict):
            return True
    return False


def _schema_type_label(schema: Any) -> str:
    if not isinstance(schema, dict):
        return "any"
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        values = [str(value) for value in schema["enum"][:4]]
        suffix = "|..." if len(schema["enum"]) > 4 else ""
        return "|".join(values) + suffix
    for key in ("anyOf", "oneOf"):
        variants = schema.get(key)
        if isinstance(variants, list) and variants:
            labels = [_schema_type_label(item) for item in variants[:3]]
            return "|".join(label for label in labels if label) or "any"
    raw_type = schema.get("type")
    if isinstance(raw_type, list):
        raw_type = next((item for item in raw_type if item != "null"), raw_type[0])
    if raw_type == "array":
        items = schema.get("items")
        return f"{_schema_type_label(items)}[]"
    if raw_type == "object":
        return "object"
    if isinstance(raw_type, str) and raw_type:
        return raw_type
    return "any"


def _tool_signature(tool: dict[str, Any]) -> str | None:
    name = _tool_spec_name(tool)
    if not name:
        return None
    schema = _tool_json_schema(tool)
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return f"{name}()"
    required = schema.get("required")
    required_names = (
        [str(item) for item in required] if isinstance(required, list) else []
    )
    ordered_names: list[str] = []
    for prop in required_names:
        if prop in properties and prop not in ordered_names:
            ordered_names.append(prop)
    for prop in properties:
        if prop not in ordered_names:
            ordered_names.append(str(prop))
    parts: list[str] = []
    for prop in ordered_names[:8]:
        prop_schema = properties.get(prop)
        optional = "" if prop in required_names else "?"
        parts.append(f"{prop}{optional}:{_schema_type_label(prop_schema)}")
    if len(ordered_names) > 8:
        parts.append("...")
    return f"{name}({', '.join(parts)})"


def _tool_example_value(schema: Any) -> str:
    schema_types = _schema_type_names(schema)
    if "array" in schema_types:
        return "[]"
    if "object" in schema_types:
        return "{}"
    if "boolean" in schema_types:
        return "true"
    if "integer" in schema_types:
        return "0"
    if "number" in schema_types:
        return "0"
    if "string" in schema_types:
        return "ARGUMENT_VALUE"
    return "ARGUMENT_VALUE"


def _tool_call_example(tools: list[dict[str, Any]]) -> str:
    if not tools:
        return (
            "<tool_call>\n"
            "<function=tool_name>\n"
            "</function>\n"
            "</tool_call>"
        )
    tool = tools[0]
    name = _tool_spec_name(tool) or "tool_name"
    schema = _tool_json_schema(tool)
    properties = schema.get("properties")
    required = schema.get("required")
    params: list[str] = []
    if isinstance(properties, dict):
        required_names = (
            [str(item) for item in required] if isinstance(required, list) else []
        )
        for prop in (required_names or list(properties))[:3]:
            if prop in properties:
                params.extend(
                    [
                        f"<parameter={prop}>",
                        _tool_example_value(properties.get(prop)),
                        "</parameter>",
                    ]
                )
    params_text = "\n".join(params)
    body = f"<function={name}>"
    if params_text:
        body = f"{body}\n{params_text}\n</function>"
    else:
        body = f"{body}\n</function>"
    return f"<tool_call>\n{body}\n</tool_call>"


def _forced_tool_choice_name(tool_choice: Any) -> str | None:
    if not isinstance(tool_choice, dict):
        return None
    value = tool_choice.get("type") or tool_choice.get("mode")
    if not isinstance(value, str) or value.strip().lower() != "function":
        return None
    function = tool_choice.get("function")
    if not isinstance(function, dict):
        return None
    name = str(function.get("name") or "").strip()
    return name or None


def _tool_choice_policy_signature(tool_choice: Any) -> str:
    if tool_choice is None:
        return "auto"
    if isinstance(tool_choice, str):
        return tool_choice.strip().lower() or "auto"
    if isinstance(tool_choice, dict):
        value = tool_choice.get("type") or tool_choice.get("mode")
        kind = str(value or "dict").strip().lower() or "dict"
        forced_name = _forced_tool_choice_name(tool_choice)
        return f"{kind}:{forced_name}" if forced_name else kind
    return type(tool_choice).__name__


def _forced_tool_contract_clause(tool_choice: Any) -> str:
    forced_name = _forced_tool_choice_name(tool_choice)
    if forced_name:
        return (
            f" This request requires the `{forced_name}` tool call; reason "
            "internally if needed, then emit that tool call instead of a "
            "normal text answer."
        )
    if _tool_choice_forces_tools(tool_choice):
        return (
            " This request requires one declared tool call; reason internally "
            "if needed, then emit the tool call instead of a normal text answer."
        )
    return ""


def _mtplx_tool_contract_text(
    tools: list[dict[str, Any]],
    *,
    tool_choice: Any = None,
) -> str:
    signatures = [signature for tool in tools if (signature := _tool_signature(tool))]
    allowed = "; ".join(signatures) if signatures else "(none)"
    example = _tool_call_example(tools)
    if len(allowed) > 1200:
        allowed = allowed[:1197].rstrip() + "..."
    forced_clause = _forced_tool_contract_clause(tool_choice)
    return (
        f"{_MTPLX_TOOL_CONTRACT_SENTINEL} declared tools and schemas: {allowed}. "
        "Call only these exact tool names and exact argument keys/case. "
        "Include every required key shown in the signature. "
        "For large files, search first and use the smallest read range/limit/offset "
        "the declared read tool supports. "
        "Emit tool calls using the Qwen native XML format shown by the chat template, "
        f"for example: {example}. Do not put a JSON object inside <tool_call>. "
        "Do not put full file contents, code blocks, patches, or implementation "
        "output in reasoning/thinking. When creating or editing files, emit the "
        "declared write/edit tool call with the file content as tool arguments. "
        "Never invent Agent/task/Explore or any undeclared tool. "
        "If no declared tool applies, answer normally."
        f"{forced_clause}"
    )


def _mtplx_no_tool_contract_text() -> str:
    return (
        f"{_MTPLX_NO_TOOL_CONTRACT_SENTINEL} tools are unavailable for this "
        "turn. Start with the final user-facing answer to the latest user "
        "message. For greetings, answer with one short friendly sentence. No "
        "markdown lists, analysis, examples, commands, searches, file actions, "
        "tool names, or protocol tags. Do not emit <tool_call>, <toolExec>, "
        "<invoke>, <function>, or <parameter>."
    )


def _with_mtplx_no_tool_contract(
    messages: list[ChatMessage],
) -> list[ChatMessage]:
    contract = _mtplx_no_tool_contract_text()
    if not messages:
        return [ChatMessage(role="system", content=contract)]
    updated = list(messages)
    first = updated[0]
    if str(first.role).lower() == "system":
        content = str(first.content or "")
        if _MTPLX_NO_TOOL_CONTRACT_SENTINEL not in content:
            updated[0] = _copy_chat_message(
                first,
                content=(f"{content.rstrip()}\n\n{contract}" if content else contract),
            )
        return updated
    return [ChatMessage(role="system", content=contract), *updated]


def _mtplx_read_only_force_answer_contract_text() -> str:
    return (
        f"{_MTPLX_READ_ONLY_FORCE_ANSWER_SENTINEL} tools are intentionally "
        "closed for this read-only inspection because enough project evidence "
        "has already been gathered. Answer the latest user request now from "
        "the prior tool results and conversation evidence. Do not emit tool "
        "calls, tool names, protocol tags, filePath/startLine/endingLine "
        "markup, or promises such as 'let me read', 'let me check', or "
        "'I'll inspect'. Do not request more context. If a detail is uncertain, "
        "name the uncertainty briefly and still give the best supported "
        "answer. If the user requested a fixed number of findings or changes, "
        "return exactly that number of items and any requested marker; do not "
        "add evidence inventory, planning preambles, recap tables, duplicate "
        "summaries, or extra sections unless the user asked for them. Markdown "
        "lists are allowed when they are the clearest way to satisfy the "
        "requested answer."
    )


def _mtplx_read_only_force_answer_user_instruction_text() -> str:
    return (
        f"{_MTPLX_READ_ONLY_FORCE_ANSWER_USER_SENTINEL} This read-only "
        "inspection is now closed to more tools. The only valid next assistant "
        "turn is the final user-facing answer. Do not write planning text that "
        "requests more inspection. Do not emit raw tool markup such as "
        "glob> path>, grep> path>, read> filePath>, filePath>, startingLine>, "
        "endingLine>, <tool_call>, <function>, or <parameter>. Return the "
        "requested final answer from the evidence already gathered, with the "
        "requested number of items and marker. Begin the assistant content "
        f"with the exact internal marker {_MTPLX_READ_ONLY_FORCE_ANSWER_STREAM_MARKER} "
        "and then the answer itself; MTPLX removes that marker before the user "
        "sees it. Do not include an evidence inventory, checklist, or "
        "rehearsal before the final answer."
    )


def _with_mtplx_read_only_force_answer_contract(
    messages: list[ChatMessage],
) -> list[ChatMessage]:
    contract = _mtplx_read_only_force_answer_contract_text()
    user_instruction = _mtplx_read_only_force_answer_user_instruction_text()
    final_instruction = f"{contract}\n\n{user_instruction}"
    if not messages:
        return [ChatMessage(role="user", content=final_instruction)]
    updated = list(messages)
    if not any(
        _MTPLX_READ_ONLY_FORCE_ANSWER_USER_SENTINEL in str(message.content or "")
        for message in updated
    ):
        updated.append(ChatMessage(role="user", content=final_instruction))
    return updated


def _pi_convergence_after_tools() -> int:
    return max(0, _env_int("MTPLX_PI_CONVERGENCE_AFTER_TOOLS", 14))


def _request_should_add_pi_convergence_contract(
    messages: list[ChatMessage],
    *,
    headers: Mapping[str, str],
    metadata: Mapping[str, Any],
    tools_active: bool,
) -> bool:
    if not tools_active:
        return False
    client_hint = str(_request_client_hint_from_headers(headers, metadata) or "").lower()
    if "pi" not in client_hint:
        return False
    limit = _pi_convergence_after_tools()
    if limit <= 0:
        return False
    return _tool_result_message_count(messages) >= limit


def _mtplx_pi_convergence_contract_text() -> str:
    return (
        f"{_MTPLX_PI_CONVERGENCE_SENTINEL} this Pi coding task has enough "
        "tool evidence. The next assistant turn must converge: either emit "
        "one edit/write tool call for the safest useful change, emit one bash "
        "tool call that verifies the chosen change, or give the final answer "
        "if no safe edit is warranted. Do not call grep, find, ls, cat, wc, "
        "or broad inspection-only bash commands. A single targeted read or "
        "sed/head/tail range is allowed only when an edit failed or a compiler "
        "line number is needed to construct exact edit text; after that, edit "
        "or verify immediately. Do not say 'let me read', 'let me check', or "
        "promise more inspection. If a detail is uncertain, name the "
        "uncertainty briefly and still choose the best supported next step "
        "from the evidence already gathered."
    )


def _mtplx_pi_convergence_user_instruction_text() -> str:
    return (
        f"{_MTPLX_PI_CONVERGENCE_USER_SENTINEL} Stop gathering more project "
        "context now. Use the evidence already gathered to edit, verify, or "
        "finish. The next response must not be another broad read/grep/find/ls "
        "or inspection-only shell command; only one narrow line-range refresh "
        "is allowed when it is necessary to make the edit apply."
    )


def _with_mtplx_pi_convergence_contract(
    messages: list[ChatMessage],
) -> list[ChatMessage]:
    contract = _mtplx_pi_convergence_contract_text()
    user_instruction = _mtplx_pi_convergence_user_instruction_text()
    if not messages:
        return [
            ChatMessage(role="system", content=contract),
            ChatMessage(role="user", content=user_instruction),
        ]
    updated = list(messages)
    first = updated[0]
    if str(first.role).lower() == "system":
        content = str(first.content or "")
        if _MTPLX_PI_CONVERGENCE_SENTINEL not in content:
            updated[0] = _copy_chat_message(
                first,
                content=(f"{content.rstrip()}\n\n{contract}" if content else contract),
            )
    else:
        updated = [ChatMessage(role="system", content=contract), *updated]
    if not any(
        _MTPLX_PI_CONVERGENCE_USER_SENTINEL in str(message.content or "")
        for message in updated
    ):
        updated.append(ChatMessage(role="user", content=user_instruction))
    return updated


def _mtplx_coding_agent_tail_contract_text(tools: list[dict[str, Any]]) -> str | None:
    if not _anonymous_coding_agent_tool_request(_tool_names(tools)):
        return None
    return (
        f"{_MTPLX_CODING_AGENT_TAIL_SENTINEL} the user request immediately "
        "above is the active coding task. If the next step requires inspecting "
        "files, running commands, editing code, continuing work, or checking "
        "project status, emit one declared <tool_call> now. Do not end the "
        "turn with a promise such as 'let me check', 'let me fix this', or "
        "'I'll run it' unless the same assistant turn also includes the tool "
        "call. Do not draft full files, code blocks, or patches in reasoning; "
        "put implementation payloads in the declared tool call arguments. "
        "For review, evaluation, summarize, or inspect-only tasks, use targeted "
        "tool calls and then answer once the current evidence covers the entry "
        "points, relevant definitions, or representative line ranges; do not "
        "reconstruct full files by walking adjacent read offsets only for "
        "completeness. For recommendation tasks, choose the best supported "
        "recommendation and answer once you have enough evidence; do not keep "
        "reading extra alternatives only to increase confidence. If the user "
        "asks for one of several files or options, pick the most relevant one "
        "and stop after that one unless the user explicitly asks to compare all "
        "of them. When the user asks for a fixed number of findings or changes, "
        "return that number of items and any requested marker; do not add recap "
        "tables, duplicate summaries, or extra sections unless the user asked "
        "for them. "
        "A user message of 'continue' means continue the active coding "
        "task, not answer conversationally. Return text only when it contains "
        "concrete results and no tool is needed."
    )


def _should_add_mtplx_coding_agent_tail_contract(
    normalized: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]],
) -> bool:
    if not _anonymous_coding_agent_tool_request(_tool_names(tools)):
        return False
    last_user = _last_user_text(normalized)
    if _is_simple_chitchat_text(last_user):
        return False
    return True


def _mtplx_tool_result_continuation_hint_text() -> str:
    return (
        "Continue the active coding task using the tool result immediately "
        "above. This instruction is not part of the tool output and must not "
        "be quoted or discussed. If the original user request explicitly lists "
        "required commands, files, checks, or says to use tools in a specific "
        "order, emit the next required tool call until those concrete "
        "requirements are complete. For read-only inspection, review, audit, "
        "or recommendation tasks, answer as soon as the current evidence can "
        "support a concrete result; do not read adjacent ranges, whole files, "
        "or extra candidate files only to increase confidence. If the user "
        "asks for one of several files or options, choose the most relevant "
        "one and answer after inspecting it unless the request explicitly asks "
        "to compare all of them. When the user asks for a fixed number of "
        "findings or changes, return that number of items and any requested "
        "marker; skip recap tables and duplicate summaries unless the user "
        "asked for them. Otherwise answer now in normal assistant "
        "text. Do not answer with a bare checklist number, partial label, "
        "placeholder, promise to continue later, or empty response."
    )


def _strip_mtplx_internal_continuation_markers(text: str) -> str:
    if not text:
        return ""
    cleaned = _MTPLX_TOOL_RESULT_CONTINUATION_BLOCK_RE.sub("", text)
    cleaned = _MTPLX_TOOL_RESULT_CONTINUATION_TAG_RE.sub("", cleaned)
    read_only_force_answer_phrases = (
        _MTPLX_READ_ONLY_FORCE_ANSWER_SENTINEL,
        _MTPLX_READ_ONLY_FORCE_ANSWER_USER_SENTINEL,
        _MTPLX_PI_CONVERGENCE_SENTINEL,
        _MTPLX_PI_CONVERGENCE_USER_SENTINEL,
        "This read-only inspection is now closed to more tools",
        "The only valid next assistant turn is the final user-facing answer",
        "Do not write planning text that requests more inspection",
        "Return the requested final answer from the evidence already gathered",
        "Stop gathering more project context now",
        "The next response must not be another read/grep/find/ls",
    )
    if any(phrase in cleaned for phrase in read_only_force_answer_phrases):
        cleaned = "\n".join(
            line
            for line in cleaned.splitlines()
            if not any(phrase in line for phrase in read_only_force_answer_phrases)
        )
    if _MTPLX_TOOL_RESULT_CONTINUATION_SENTINEL not in cleaned:
        return cleaned
    kept: list[str] = []
    skip_phrases = (
        _MTPLX_TOOL_RESULT_CONTINUATION_SENTINEL,
        "Internal MTPLX continuation note",
        "Internal instruction for the next assistant turn",
        "Do not quote this note",
        "not part of any tool output",
        "not part of the tool output",
        "Use that result as the newest evidence",
        "Treat the immediately preceding tool result",
        "For read-only inspection",
        "A truncated file read is enough",
        "Use another declared tool call only when",
        "If the user's latest request explicitly lists",
        "Otherwise answer now",
        "likely wrong, not merely less complete",
        "Do not restate the task",
        "placeholder, promise to continue later",
    )
    for line in cleaned.splitlines():
        if any(phrase in line for phrase in skip_phrases):
            continue
        kept.append(line)
    return "\n".join(kept)


def _looks_like_stalled_agent_tool_promise(content: str) -> bool:
    cleaned = _strip_mtplx_internal_continuation_markers(content).strip()
    if not cleaned:
        return False
    if _looks_like_stalled_agent_preamble(cleaned):
        return True
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", cleaned) if part.strip()]
    tail = paragraphs[-1] if paragraphs else cleaned[-500:]
    tail = tail.strip()
    if len(tail) > 700:
        tail = tail[-700:]
    promise_re = re.compile(
        r"(?:^|[.!?\n]\s*)"
        r"(?:actually,\s*)?"
        r"(?:(?:now\s+)?let\s+me|i\s+need\s+to|i\s+should|i(?:'ll| will))\s+"
        r"(?:also\s+)?"
        r"(?:read|check|verify|run|inspect|look\s+at|open|fix|edit|update|write)"
        r"\b.{0,220}[.!?:]?$",
        re.IGNORECASE | re.DOTALL,
    )
    if promise_re.search(tail):
        return True
    return bool(
        re.search(
            r"\b(let me|i need to|i should|i(?:'ll| will))\b.{0,160}\b"
            r"(read|check|verify|run|inspect|look at|open)\b.{0,120}$",
            tail,
            re.IGNORECASE | re.DOTALL,
        )
    )


_TOOLISH_FORCE_ANSWER_MARKUP_RE = re.compile(
    r"(<\s*(?:tool_call|toolExec|invoke|function|parameter)\b"
    r"|</\s*(?:tool_call|toolExec|invoke|function|parameter)\s*>"
    r"|\b(?:filePath|file_path|startingLine|startLine|endingLine|endLine|limit|offset)\s*>"
    r"|\b(?:filePath|file_path|startingLine|startLine|endingLine|endLine)\s*[:=]\s*)",
    re.IGNORECASE,
)


def _looks_like_read_only_force_answer_failure(content: str) -> bool:
    cleaned = _strip_mtplx_internal_continuation_markers(content).strip()
    if not cleaned:
        return False
    if _TOOLISH_FORCE_ANSWER_MARKUP_RE.search(cleaned):
        return True
    return _looks_like_stalled_agent_tool_promise(cleaned)


_READ_ONLY_FINAL_ANSWER_START_RE = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]+)?"
    r"(?:[A-Z][A-Z0-9_]{2,48}:|1[.)][ \t]+|(?:risk|finding)[ \t]+1\b)"
)


def _read_only_force_answer_after_stream_marker(content: str) -> tuple[str, int]:
    match = _MTPLX_READ_ONLY_FORCE_ANSWER_STREAM_MARKER_RE.search(
        str(content or "")
    )
    if match is None:
        return str(content or ""), 0
    return str(content or "")[match.end() :].lstrip(), match.end()


def _strip_read_only_force_answer_visible_control_tags(content: str) -> str:
    return (
        _MTPLX_READ_ONLY_FORCE_ANSWER_STREAM_MARKER_RE.sub("", str(content or ""))
        .replace(THINK_OPEN, "")
        .replace(THINK_CLOSE, "")
        .strip()
    )


def _read_only_force_answer_visible_text(content: str) -> tuple[str, int]:
    cleaned = _strip_mtplx_internal_continuation_markers(
        _strip_generated_chat_template_sentinels(str(content or ""))
    ).strip()
    if not cleaned:
        return "", 0
    marked_visible, marker_stripped_chars = _read_only_force_answer_after_stream_marker(
        cleaned
    )
    if marker_stripped_chars:
        return (
            _strip_read_only_force_answer_visible_control_tags(marked_visible),
            marker_stripped_chars,
        )
    reasoning_text, content_text = omlx_extract_thinking(cleaned)
    visible = "\n\n".join(
        part.strip()
        for part in (reasoning_text, content_text)
        if part and part.strip()
    ) or cleaned
    visible = _strip_read_only_force_answer_visible_control_tags(visible)
    marker_match = re.search(r"(?im)^[ \t]*marker\s*=[^\n\r]+", visible)
    search_end = marker_match.start() if marker_match else len(visible)
    candidates = [
        match.start()
        for match in _READ_ONLY_FINAL_ANSWER_START_RE.finditer(visible[:search_end])
    ]
    if not candidates:
        return visible, 0
    start = candidates[-1]
    return visible[start:].strip(), start


def _append_tool_result_continuation_hint(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]],
) -> None:
    if not _anonymous_coding_agent_tool_request(_tool_names(tools)):
        return
    if not messages:
        return
    last = messages[-1]
    if last.get("role") != "tool":
        return
    if any(
        "Continue the active coding task using the tool result immediately above"
        in str(message.get("content") or "")
        for message in messages
    ):
        return
    hint = _mtplx_tool_result_continuation_hint_text()
    # Keep this out of the tool result itself. OpenCode displays model
    # reasoning, and a real app-path QA run showed the model treating an
    # appended internal note as ambiguous command output. A trailing user
    # instruction is accepted by Qwen's chat template, preserves the
    # cache-friendly prefix, and keeps the evidence boundary clean.
    messages.append({"role": "user", "content": hint})


def _with_mtplx_tool_contract(
    normalized: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    tool_choice: Any = None,
) -> list[dict[str, Any]]:
    if not tools:
        return normalized
    contract = _mtplx_tool_contract_text(tools, tool_choice=tool_choice)
    tail_contract = (
        _mtplx_coding_agent_tail_contract_text(tools)
        if _should_add_mtplx_coding_agent_tail_contract(normalized, tools=tools)
        else None
    )
    if not normalized:
        return [{"role": "system", "content": contract}]
    messages = [dict(item) for item in normalized]
    _append_tool_result_continuation_hint(messages, tools=tools)
    first = messages[0]
    if first.get("role") == "system":
        content = str(first.get("content") or "")
        additions: list[str] = []
        if _MTPLX_TOOL_CONTRACT_SENTINEL not in content:
            additions.append(contract)
        if (
            tail_contract
            and _MTPLX_CODING_AGENT_TAIL_SENTINEL not in content
        ):
            additions.append(tail_contract)
        if additions:
            first["content"] = (
                f"{content.rstrip()}\n\n" + "\n\n".join(additions)
            ).strip()
        return messages
    system_parts = [contract]
    if tail_contract:
        system_parts.append(tail_contract)
    return [{"role": "system", "content": "\n\n".join(system_parts)}, *messages]


def _with_mtplx_native_agent_tail(
    normalized: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], bool]:
    """Keep native template tools, but add the coding-agent tool-start nudge.

    Native/oMLX mode should own the actual tool schema and XML format. The
    extra tail here is only the product guard that prevents coding-agent tasks
    from stopping at "let me inspect" reasoning without emitting a declared
    tool call.
    """

    if not tools:
        return normalized, False
    messages = [dict(item) for item in normalized]
    last_is_tool_result = bool(messages and messages[-1].get("role") == "tool")
    _append_tool_result_continuation_hint(messages, tools=tools)
    if last_is_tool_result:
        return messages, False
    tail_contract = (
        _mtplx_coding_agent_tail_contract_text(tools)
        if _should_add_mtplx_coding_agent_tail_contract(messages, tools=tools)
        else None
    )
    if not tail_contract:
        return messages, False
    if not messages:
        return [{"role": "system", "content": tail_contract}], True
    first = messages[0]
    if first.get("role") == "system":
        content = str(first.get("content") or "")
        if _MTPLX_CODING_AGENT_TAIL_SENTINEL not in content:
            first["content"] = (
                f"{content.rstrip()}\n\n{tail_contract}" if content else tail_contract
            )
            return messages, True
        return messages, False
    return [{"role": "system", "content": tail_contract}, *messages], True


_NO_SUBAGENTS_RE = re.compile(
    r"\bno\s+(?:sub[- ]?agents?|subagents?|sub[- ]?tasks?|agent\s+tasks)\b",
    re.IGNORECASE,
)
_EXPLICIT_SUBAGENTS_RE = re.compile(
    r"\b(?:use|launch|spawn|run|create|delegate(?:\s+to)?|ask)\s+"
    r"(?:a\s+|an\s+|the\s+)?"
    r"(?:sub[- ]?agents?|subtasks?|agent\s+tasks?|parallel\s+agents?)\b"
    r"|\bparallel\s+(?:sub[- ]?agents?|agents?)\b",
    re.IGNORECASE,
)
_EXPLICIT_TODO_TOOL_RE = re.compile(
    r"\b(?:todo|to-do|todos|checklist|task\s+list|write\s+a\s+plan|"
    r"track\s+(?:tasks|todos)|maintain\s+(?:a\s+)?plan)\b",
    re.IGNORECASE,
)
_SUBAGENT_TOOL_NAMES = {"task"}
_TODO_TOOL_NAMES = {"todowrite"}
_MUTATING_FILE_TOOL_NAMES = {
    "edit",
    "multi_edit",
    "patch",
    "str_replace_editor",
    "write",
}
_STATIC_READ_ONLY_INSPECTION_TOOL_NAMES = {
    "glob",
    "grep",
    "ls",
    "read",
    "session_status",
}
_STATIC_READ_ONLY_COMMAND_RE = re.compile(
    r"\b(?:bash|benchmark|command|execute|lint|profile|pytest|run|serve|"
    r"shell|start|terminal|test|typecheck)\b"
    r"|\bbuild\s+(?:it|them|this|the|project|app|repo|repository|workspace|"
    r"target|package)\b"
    r"|\b(?:npm|pnpm|yarn|bun|uv|python|pytest|swift|xcodebuild|cargo|go|"
    r"make)\s+(?:run|test|build|check|lint|typecheck|serve|start)\b",
    re.IGNORECASE,
)
_NARROW_READ_ONLY_SHELL_RE = re.compile(
    r"\b(?:bash|shell|terminal|command)\b|\brun\s+(?:pwd|ls|cat\b)",
    re.IGNORECASE,
)
_NARROW_READ_ONLY_READ_RE = re.compile(
    r"\b(?:read|inspect|open|view)\b.{0,80}"
    r"\b(?:[\w.-]+(?:\.[a-z0-9]{1,12})|file|package\.json)\b",
    re.IGNORECASE | re.DOTALL,
)
_NARROW_READ_ONLY_ANSWER_RE = re.compile(
    r"\b(?:answer|respond|reply|report|return|finish|summari[sz]e)\b",
    re.IGNORECASE,
)
_BROAD_READ_ONLY_DISCOVERY_RE = re.compile(
    r"\b(?:search|find|grep|glob|rg|list|ls|relevant\s+files?|all\s+files?|"
    r"scan\s+(?:the\s+)?(?:repo|repository|project|workspace))\b",
    re.IGNORECASE,
)
_DEEP_READ_ONLY_DISCOVERY_RE = re.compile(
    r"\b(?:search|find|grep|glob|rg|relevant\s+files?|all\s+files?|"
    r"scan\s+(?:the\s+)?(?:repo|repository|project|workspace))\b",
    re.IGNORECASE,
)
_SHALLOW_READ_ONLY_LISTING_RE = re.compile(
    r"\b(?:list|ls)\s+(?:the\s+)?(?:top[- ]?level|root)\s+"
    r"(?:files?|entries|directory|contents?)\b",
    re.IGNORECASE,
)
_STATIC_READ_ONLY_REVIEW_RE = re.compile(
    r"\b(?:audit|code\s+review|evaluate|identify\s+risks?|launch[- ]readiness|"
    r"quality|release\s+review|review|risks?|strengths?\s+and\s+weaknesses?)\b",
    re.IGNORECASE,
)
_EXPLICIT_SINGLE_TOOL_THEN_ANSWER_RE = re.compile(
    r"\b(?:use|call|run)\b.{0,80}\b(?:exactly\s+)?(?:once|one\s+time)\b"
    r".{0,120}\b(?:then|after(?:wards)?|and\s+then)\b.{0,80}"
    r"\b(?:answer|reply|respond|return|finish)\b",
    re.IGNORECASE | re.DOTALL,
)
_NARROW_READ_ONLY_TOOL_NAMES = {"bash", "read"}

# Read-budget force-answer turns keep the read-only inspection toolset so the
# model can still cite evidence, while explicit "use one tool then answer"
# turns generate tool-free (QA-087/QA-088 contract).
_READ_ONLY_FORCE_ANSWER_TOOL_NAMES = {"bash", "read", "glob", "grep"}
_LOCAL_READ_ONLY_PROJECT_RE = re.compile(
    r"\b(?:project|repo|repository|workspace|codebase|source|files?|package\.json|"
    r"src/|tests?/|scripts?)\b",
    re.IGNORECASE,
)
_LOCAL_READ_ONLY_ACTION_RE = re.compile(
    r"\b(?:inspect|read|search|find|grep|glob|rg|list|scan|run|test|build|"
    r"typecheck|lint|review|audit|evaluate)\b",
    re.IGNORECASE,
)
_REMOTE_OR_INTERACTIVE_TOOL_RE = re.compile(
    r"\b(?:web|webfetch|browser|internet|online|latest|docs?|documentation|"
    r"https?://|url|ask|question|clarify|skill)\b",
    re.IGNORECASE,
)
_LOCAL_READ_ONLY_TOOL_NAMES = {
    "bash",
    "glob",
    "grep",
    "ls",
    "read",
    "session_status",
}
_NO_FILE_MUTATION_RE = re.compile(
    r"\b(?:do\s+not|don['’]?t|dont|without)\s+"
    r"(?:edit|modify|change|write|create|patch|touch)\b.{0,48}"
    r"\b(?:files?|source|project|repo|repository|workspace|code)\b"
    r"|\bno\s+(?:file\s+)?(?:edits?|changes?|modifications?|writes?)\b"
    r"|\b(?:read[- ]only|inspect[- ]only)\b",
    re.IGNORECASE | re.DOTALL,
)
_NO_TOOL_USE_RE = re.compile(
    r"\b(?:do\s+not|don['’]?t|dont|never)\s+"
    r"(?:use|call|invoke)\s+(?:any\s+)?tools?\b"
    r"|\bwithout\s+(?:using\s+)?tools?\b"
    r"|\bno\s+tool\s+calls?\b"
    r"|\bno\s+tools?\b",
    re.IGNORECASE,
)
_NO_TOOL_USE_EXCEPTION_RE = re.compile(
    r"\b(?:except|other\s+than|besides|apart\s+from|but\s+use|only\s+use)\b",
    re.IGNORECASE,
)


def _request_disallows_subagents(messages: list[ChatMessage]) -> bool:
    for message in reversed(messages):
        if str(message.role).lower() != "user":
            continue
        return bool(_NO_SUBAGENTS_RE.search(_content_to_text(message.content)))
    return False


def _request_explicitly_allows_subagents(messages: list[ChatMessage]) -> bool:
    for message in reversed(messages):
        if str(message.role).lower() != "user":
            continue
        return bool(_EXPLICIT_SUBAGENTS_RE.search(_content_to_text(message.content)))
    return False


def _request_explicitly_allows_todo_tool(messages: list[ChatMessage]) -> bool:
    for message in reversed(messages):
        if str(message.role).lower() != "user":
            continue
        return bool(_EXPLICIT_TODO_TOOL_RE.search(_content_to_text(message.content)))
    return False


def _request_disallows_file_mutation(messages: list[ChatMessage]) -> bool:
    for message in reversed(messages):
        if str(message.role).lower() != "user":
            continue
        return bool(_NO_FILE_MUTATION_RE.search(_content_to_text(message.content)))
    return False


def _request_disallows_tools(messages: list[ChatMessage]) -> bool:
    for message in reversed(messages):
        if str(message.role).lower() != "user":
            continue
        text = _content_to_text(message.content)
        return bool(_NO_TOOL_USE_RE.search(text)) and not bool(
            _NO_TOOL_USE_EXCEPTION_RE.search(text)
        )
    return False


def _request_is_static_read_only_inspection(messages: list[ChatMessage]) -> bool:
    last_user = _last_user_text(messages)
    if not _is_read_only_inspection_request(last_user):
        return False
    return not bool(_STATIC_READ_ONLY_COMMAND_RE.search(last_user))


def _static_read_only_inspection_tool_names(
    messages: list[ChatMessage],
) -> set[str]:
    names = set(_STATIC_READ_ONLY_INSPECTION_TOOL_NAMES)
    if _STATIC_READ_ONLY_REVIEW_RE.search(_last_user_text(messages)):
        names.add("bash")
    return names


def _read_only_inspection_force_answer_after_tools() -> int:
    return max(
        0,
        _env_int(
            "MTPLX_READ_ONLY_INSPECTION_FORCE_ANSWER_AFTER_TOOLS",
            0,
        ),
    )


def _tool_result_message_count(messages: list[ChatMessage]) -> int:
    return sum(1 for message in messages if str(message.role).lower() == "tool")


def _request_explicit_single_tool_then_answer(messages: list[ChatMessage]) -> bool:
    return bool(_EXPLICIT_SINGLE_TOOL_THEN_ANSWER_RE.search(_last_user_text(messages)))


def _request_should_force_answer_for_read_only_inspection(
    messages: list[ChatMessage],
) -> bool:
    if (
        _tool_result_message_count(messages) > 0
        and _request_explicit_single_tool_then_answer(messages)
    ):
        return True
    if not _request_is_static_read_only_inspection(messages):
        return False
    limit = _read_only_inspection_force_answer_after_tools()
    if limit <= 0:
        return False
    return _tool_result_message_count(messages) >= limit


def _request_is_narrow_read_only_tool_choreography(
    messages: list[ChatMessage],
) -> bool:
    last_user = _last_user_text(messages)
    if not last_user or not _request_disallows_file_mutation(messages):
        return False
    if not _NARROW_READ_ONLY_SHELL_RE.search(last_user):
        return False
    if not _NARROW_READ_ONLY_READ_RE.search(last_user):
        return False
    if not _NARROW_READ_ONLY_ANSWER_RE.search(last_user):
        return False
    if _DEEP_READ_ONLY_DISCOVERY_RE.search(last_user):
        return False
    if _BROAD_READ_ONLY_DISCOVERY_RE.search(last_user):
        return bool(_SHALLOW_READ_ONLY_LISTING_RE.search(last_user))
    return True


def _request_is_local_read_only_project_workflow(
    messages: list[ChatMessage],
) -> bool:
    last_user = _last_user_text(messages)
    if not last_user or not _request_disallows_file_mutation(messages):
        return False
    if not _LOCAL_READ_ONLY_PROJECT_RE.search(last_user):
        return False
    if not _LOCAL_READ_ONLY_ACTION_RE.search(last_user):
        return False
    return not bool(_REMOTE_OR_INTERACTIVE_TOOL_RE.search(last_user))


def _without_tool_specs(
    tools: list[dict[str, Any]],
    hidden_names: set[str],
) -> list[dict[str, Any]]:
    return [
        tool
        for tool in tools
        if (_tool_spec_name(tool) or "").strip().lower() not in hidden_names
    ]


def _filter_tool_specs_for_request(
    tools: list[dict[str, Any]],
    messages: list[ChatMessage],
    *,
    tool_choice: Any = None,
) -> list[dict[str, Any]]:
    if not tools:
        return tools
    if _tool_choice_forces_tools(tool_choice):
        return tools
    if _request_disallows_tools(messages):
        return []
    hidden_tools: set[str] = set()
    if _request_disallows_subagents(messages):
        hidden_tools.update(_SUBAGENT_TOOL_NAMES)
    elif not _request_explicitly_allows_subagents(messages):
        hidden_tools.update(_SUBAGENT_TOOL_NAMES)
    if not _request_explicitly_allows_todo_tool(messages):
        hidden_tools.update(_TODO_TOOL_NAMES)
    if _request_disallows_file_mutation(messages):
        hidden_tools.update(_MUTATING_FILE_TOOL_NAMES)
    if _request_is_narrow_read_only_tool_choreography(messages):
        requested_names = {
            (_tool_spec_name(tool) or "").strip().lower()
            for tool in tools
        }
        if requested_names & _NARROW_READ_ONLY_TOOL_NAMES:
            hidden_tools.update(
                name
                for tool in tools
                if (name := (_tool_spec_name(tool) or "").strip().lower())
                and name not in _NARROW_READ_ONLY_TOOL_NAMES
            )
    elif _request_is_local_read_only_project_workflow(messages):
        requested_names = {
            (_tool_spec_name(tool) or "").strip().lower()
            for tool in tools
        }
        if requested_names & _LOCAL_READ_ONLY_TOOL_NAMES:
            hidden_tools.update(
                name
                for tool in tools
                if (name := (_tool_spec_name(tool) or "").strip().lower())
                and name not in _LOCAL_READ_ONLY_TOOL_NAMES
            )
    if _request_is_static_read_only_inspection(messages):
        # OpenCode-style "review / evaluate / audit" turns lock the model to
        # a small read-only inspection toolset. That heuristic only makes
        # sense for coding-agent clients whose toolset actually includes
        # those read-only tools. Generic clients that ship a different
        # toolset (e.g. the MTPLX in-app chat with `web_search` /
        # `fetch_url`) would otherwise see every tool filtered out — the
        # model then loses all tool definitions in its prompt and falls
        # back to emitting `<tool_call>` XML guesses from training. Require
        # at least one safe-set tool to be present before enforcing the
        # lockdown.
        requested_names = {
            (_tool_spec_name(tool) or "").strip().lower()
            for tool in tools
        }
        static_read_only_tool_names = _static_read_only_inspection_tool_names(messages)
        if requested_names & static_read_only_tool_names:
            hidden_tools.update(
                name
                for tool in tools
                if (name := (_tool_spec_name(tool) or "").strip().lower())
                and name not in static_read_only_tool_names
            )
    if hidden_tools:
        return _without_tool_specs(tools, hidden_tools)
    return tools


def _should_add_no_tool_contract(
    *,
    requested_tools: list[dict[str, Any]],
    tools_active: bool,
    messages: list[ChatMessage],
) -> bool:
    if not requested_tools or tools_active:
        return False
    if _is_simple_chitchat_text(_last_user_text(messages)):
        return False
    return True


def _opencode_prompt_contract_profile(
    messages: list[ChatMessage],
    *,
    headers: Mapping[str, str],
    metadata: Mapping[str, Any],
    tool_choice: Any,
) -> str | None:
    if not _is_opencode_client(headers=headers, metadata=metadata):
        return None
    return _MTPLX_OPENCODE_AGENT_CONTRACT_PROFILE


def _opencode_prompt_contract_system_prompt(profile: str | None) -> str | None:
    # OpenCode owns its agent/system instructions. MTPLX owns runtime policy:
    # sampler, reasoning, MTP depth, cache identity, and the extra tool contract
    # reminder injected by _with_mtplx_tool_contract. Replacing OpenCode's
    # system prompt makes the model look fast while starving it of the actual
    # agent contract.
    return None


def _normalize_tool_specs(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not tools:
        return []
    normalized: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise HTTPException(
                status_code=400, detail=f"tools[{index}] must be an object"
            )
        if not _tool_spec_name(tool):
            raise HTTPException(
                status_code=400,
                detail=f"tools[{index}] must include a function name",
            )
        normalized.append(tool)
    return normalized


def _tool_choice_disables_tools(tool_choice: Any) -> bool:
    if tool_choice is None:
        return False
    if isinstance(tool_choice, str):
        return tool_choice.strip().lower() == "none"
    if isinstance(tool_choice, dict):
        value = tool_choice.get("type") or tool_choice.get("mode")
        return isinstance(value, str) and value.strip().lower() == "none"
    return False


def _tool_choice_forces_tools(tool_choice: Any) -> bool:
    if tool_choice is None:
        return False
    if isinstance(tool_choice, str):
        return tool_choice.strip().lower() in {"required", "tool", "function"}
    if isinstance(tool_choice, dict):
        value = tool_choice.get("type") or tool_choice.get("mode")
        return isinstance(value, str) and value.strip().lower() in {
            "function",
            "tool",
            "required",
        }
    return False


def _validate_tool_choice(tool_specs: list[dict[str, Any]], tool_choice: Any) -> None:
    if not tool_specs or not isinstance(tool_choice, dict):
        return
    if str(tool_choice.get("type") or "").lower() != "function":
        return
    function = tool_choice.get("function")
    if not isinstance(function, dict):
        raise HTTPException(
            status_code=400,
            detail="tool_choice function must include a function object",
        )
    requested = str(function.get("name") or "").strip()
    if not requested:
        raise HTTPException(
            status_code=400,
            detail="tool_choice function must include a name",
        )
    known = {_tool_spec_name(tool) for tool in tool_specs}
    if requested not in known:
        raise HTTPException(
            status_code=400,
            detail=f"tool_choice requested unknown tool '{requested}'",
        )


def _tools_active_for_request(
    tools: list[dict[str, Any]],
    tool_choice: Any,
) -> bool:
    if not tools:
        return False
    if _tool_choice_disables_tools(tool_choice):
        return False
    _validate_tool_choice(tools, tool_choice)
    return True


def _json_object_value(value: Any, *, context: str) -> dict[str, Any]:
    if value is None:
        parsed: Any = {}
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            parsed = {}
        else:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise _tool_protocol_error(
                    f"{context} arguments are not valid JSON"
                ) from exc
    else:
        parsed = value
    if not isinstance(parsed, dict):
        raise _tool_protocol_error(f"{context} arguments must be a JSON object")
    return parsed


def _json_object_string(value: Any, *, context: str) -> str:
    parsed = _json_object_value(value, context=context)
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


_TOOL_ARGUMENT_PRIMARY_ORDER: dict[str, tuple[str, ...]] = {
    "read": ("filePath", "path", "offset", "limit"),
    "grep": ("pattern", "path", "include", "limit"),
    "glob": ("pattern", "path"),
    "bash": ("command", "description", "timeout"),
    "write": ("filePath", "path", "content"),
    "edit": ("filePath", "path", "oldString", "newString", "replaceAll"),
    "webfetch": ("url", "format", "timeout"),
    "skill": ("name",),
    "question": ("questions",),
}


def _order_tool_arguments_for_client_display(
    *,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if not arguments:
        return arguments
    order = _TOOL_ARGUMENT_PRIMARY_ORDER.get(str(tool_name or "").strip().lower())
    if not order:
        return arguments
    ordered: dict[str, Any] = {}
    for key in order:
        if key in arguments:
            ordered[key] = arguments[key]
    for key, value in arguments.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _schema_type_names(schema: Any) -> set[str]:
    if not isinstance(schema, dict):
        return set()
    raw_type = schema.get("type")
    if isinstance(raw_type, str):
        return {raw_type}
    if isinstance(raw_type, list):
        return {str(item) for item in raw_type if isinstance(item, str)}
    return set()


def _tool_parameter_schema(
    tools: list[dict[str, Any]],
    *,
    tool_name: str | None,
    parameter_name: str,
) -> dict[str, Any] | None:
    if not tool_name:
        return None
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue
        if str(function.get("name") or "").strip() != str(tool_name).strip():
            continue
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            return None
        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            return None
        schema = properties.get(parameter_name)
        return schema if isinstance(schema, dict) else None
    return None


def _decode_tool_parameter_value(value: str, schema: Any | None = None) -> Any:
    text = value.strip()
    if not text:
        return ""
    schema_types = _schema_type_names(schema)
    wrapper = _TOOL_PARAMETER_TYPE_WRAPPER_RE.match(text)
    if wrapper is not None:
        tag = wrapper.group(1).strip().casefold()
        expected = {item.casefold() for item in schema_types if item != "null"}
        placeholder_tags = {
            "array",
            "bool",
            "boolean",
            "float",
            "int",
            "integer",
            "json",
            "number",
            "object",
            "str",
            "string",
            "text",
            "value",
        }
        aliases = {
            "bool": "boolean",
            "float": "number",
            "int": "integer",
            "json": "object",
            "str": "string",
            "text": "string",
            "value": "string",
        }
        normalized_tag = aliases.get(tag, tag)
        if (
            tag in placeholder_tags
            and (not expected or normalized_tag in expected or tag == "value")
        ):
            text = wrapper.group(2).strip()
    text = html.unescape(text)
    if schema_types and schema_types <= {"string", "null"}:
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _normalize_tool_arguments_for_schema(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, str):
            normalized[key] = _decode_tool_parameter_value(
                value,
                schema=_tool_parameter_schema(
                    tools,
                    tool_name=tool_name,
                    parameter_name=str(key),
                ),
            )
        else:
            normalized[key] = value
    return _order_tool_arguments_for_client_display(
        tool_name=tool_name,
        arguments=normalized,
    )


def _parse_json_tool_call(block: str) -> tuple[str, Any] | None:
    try:
        payload = json.loads(block)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, list):
        for item in payload:
            parsed = _parse_json_tool_call(json.dumps(item, ensure_ascii=False))
            if parsed is not None:
                return parsed
        return None
    if not isinstance(payload, dict):
        raise _tool_protocol_error("JSON tool_call payload must be an object")
    function = payload.get("function")
    if isinstance(function, dict):
        name = function.get("name") or function.get("tool") or function.get("function")
        arguments = (
            function.get("arguments")
            if "arguments" in function
            else function.get("args", function.get("parameters", {}))
        )
    else:
        name = (
            payload.get("name")
            or payload.get("tool")
            or payload.get("function")
            or payload.get("call")
        )
        arguments = payload.get(
            "arguments",
            payload.get("args", payload.get("parameters", {})),
        )
    name_text = str(name or "").strip()
    if not name_text:
        raise _tool_protocol_error("JSON tool_call is missing a function name")
    return name_text, arguments


def _parse_xml_tool_call(
    block: str,
    *,
    allow_unclosed_function: bool = False,
) -> tuple[str, Any] | None:
    match = _TOOL_FUNCTION_BLOCK_RE.match(block)
    if match is None:
        if not allow_unclosed_function:
            return None
        match = _TOOL_FUNCTION_START_RE.match(block)
        if match is None:
            return None
    name = match.group(1).strip()
    body = match.group(2)
    if allow_unclosed_function:
        body = re.sub(r"\s*</function>\s*$", "", body, flags=re.IGNORECASE)
        body = re.sub(r"\s*</tool_call\s*$", "", body, flags=re.IGNORECASE)
    arguments: dict[str, Any] = {}
    consumed: list[tuple[int, int]] = []
    for param_match in _TOOL_PARAMETER_BLOCK_RE.finditer(body):
        param_name = param_match.group(1).strip()
        if not param_name:
            raise _tool_protocol_error(
                f"tool '{name}' contains an empty parameter name"
            )
        arguments[param_name] = _decode_tool_parameter_value(param_match.group(2))
        consumed.append(param_match.span())
    if consumed:
        residue_parts: list[str] = []
        cursor = 0
        for start, end in consumed:
            residue_parts.append(body[cursor:start])
            cursor = end
        residue_parts.append(body[cursor:])
        residue = "".join(residue_parts).strip()
        if residue:
            raise _tool_protocol_error(
                f"tool '{name}' contains text outside parameters"
            )
    elif body.strip():
        raise _tool_protocol_error(f"tool '{name}' contains unwrapped parameter text")
    return name, arguments


def _parse_invoke_tool_call(block: str) -> tuple[str, Any] | None:
    match = _INVOKE_TOOL_BLOCK_RE.match(block)
    if match is None:
        return None
    name = match.group(1).strip()
    body = match.group(2)
    arguments: dict[str, Any] = {}
    consumed: list[tuple[int, int]] = []
    for param_match in _INVOKE_PARAMETER_BLOCK_RE.finditer(body):
        param_name = param_match.group(1).strip()
        if not param_name:
            raise _tool_protocol_error(
                f"tool '{name}' contains an empty parameter name"
            )
        arguments[param_name] = _decode_tool_parameter_value(param_match.group(2))
        consumed.append(param_match.span())
    if consumed:
        residue_parts: list[str] = []
        cursor = 0
        for start, end in consumed:
            residue_parts.append(body[cursor:start])
            cursor = end
        residue_parts.append(body[cursor:])
        residue = "".join(residue_parts).strip()
        if residue:
            raise _tool_protocol_error(
                f"tool '{name}' contains text outside parameters"
            )
    elif body.strip():
        raise _tool_protocol_error(f"tool '{name}' contains unwrapped parameter text")
    return name, arguments


def _tool_marker_pairs_from_tokenizer(tokenizer: Any | None) -> list[tuple[str, str]]:
    if tokenizer is None:
        return []
    start = getattr(tokenizer, "tool_call_start", None)
    end = getattr(tokenizer, "tool_call_end", None)
    if not isinstance(start, str) or not start:
        return []
    if not isinstance(end, str) or not end:
        return []
    if start == "<tool_call>" and end == "</tool_call>":
        return []
    return [(start, end)]


def _iter_generated_tool_call_envelopes(
    text: str,
    *,
    marker_pairs: list[tuple[str, str]] | None = None,
) -> list[tuple[int, int, str, tuple[str, Any] | None]]:
    envelopes: list[tuple[int, int, str, tuple[str, Any] | None]] = []
    for match in _TOOL_CALL_BLOCK_RE.finditer(text):
        envelopes.append((match.start(), match.end(), match.group(1).strip(), None))
    for match in _NAMESPACED_TOOL_CALL_BLOCK_RE.finditer(text):
        envelopes.append((match.start(), match.end(), match.group(2).strip(), None))
    for start_marker, end_marker in marker_pairs or []:
        pattern = re.compile(
            re.escape(start_marker) + r"\s*(.*?)\s*" + re.escape(end_marker),
            re.IGNORECASE | re.DOTALL,
        )
        for match in pattern.finditer(text):
            envelopes.append((match.start(), match.end(), match.group(1).strip(), None))
    for match in _BRACKET_TOOL_CALL_RE.finditer(text):
        name = match.group(1).strip()
        raw_args = (match.group(2) or "").strip()
        if raw_args:
            try:
                arguments: Any = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                raise _tool_protocol_error(
                    f"bracket tool_call '{name}' arguments are not valid JSON"
                ) from exc
        else:
            arguments = {}
        envelopes.append((match.start(), match.end(), "", (name, arguments)))
    envelopes.sort(key=lambda item: item[0])
    for previous, current in zip(envelopes, envelopes[1:]):
        if current[0] < previous[1]:
            raise _tool_protocol_error("overlapping tool_call envelopes")
    return envelopes


def _tool_like_marker_present(text: str, marker_pairs: list[tuple[str, str]]) -> bool:
    lowered = text.lower()
    if "<tool_call" in lowered or "</tool_call>" in lowered:
        return True
    if _NAMESPACED_TOOL_CALL_START_RE.search(text):
        return True
    if re.search(r"</[A-Za-z_][\w.-]*:tool_call>", text, flags=re.IGNORECASE):
        return True
    if any(prefix.lower() in lowered for prefix in _BRACKET_TOOL_PREFIXES):
        return True
    return any(
        start.lower() in lowered or end.lower() in lowered
        for start, end in marker_pairs
    )


def _parse_tool_call_payload(
    block: str,
    *,
    allow_unclosed_function: bool = False,
) -> tuple[str, Any] | None:
    parsed = _parse_json_tool_call(block)
    if parsed is not None:
        return parsed
    parsed = _parse_xml_tool_call(
        block,
        allow_unclosed_function=allow_unclosed_function,
    )
    if parsed is not None:
        return parsed
    return _parse_invoke_tool_call(block)


def _repair_unclosed_tool_call_payload(
    text: str,
    *,
    marker_pairs: list[tuple[str, str]],
) -> str | None:
    candidates: list[tuple[int, int]] = []
    plain = re.search(r"<tool_call>\s*", text, flags=re.IGNORECASE)
    if plain:
        candidates.append((plain.start(), plain.end()))
    namespaced = re.search(
        r"<[A-Za-z_][\w.-]*:tool_call>\s*",
        text,
        flags=re.IGNORECASE,
    )
    if namespaced:
        candidates.append((namespaced.start(), namespaced.end()))
    for start_marker, _end_marker in marker_pairs:
        idx = _find_casefold(text, start_marker)
        if idx >= 0:
            candidates.append((idx, idx + len(start_marker)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    if len(candidates) > 1:
        raise _tool_protocol_error("nested or unmatched tool_call block")
    _start, payload_start = candidates[0]
    payload = text[payload_start:].strip()
    if not payload:
        return None
    return payload


def _parse_generated_tool_calls(
    text: str,
    *,
    tools: list[dict[str, Any]],
    tokenizer: Any | None = None,
) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    marker_pairs = _tool_marker_pairs_from_tokenizer(tokenizer)
    if not _tool_like_marker_present(text, marker_pairs):
        return None
    envelopes = _iter_generated_tool_call_envelopes(
        text,
        marker_pairs=marker_pairs,
    )
    if not envelopes:
        repair_payload = _repair_unclosed_tool_call_payload(
            text,
            marker_pairs=marker_pairs,
        )
        if repair_payload is None:
            raise _tool_protocol_error("unclosed tool_call block")
        envelopes = [(0, len(text), repair_payload, None)]
    residue_parts: list[str] = []
    cursor = 0
    for start, end, _block, _parsed in envelopes:
        residue_parts.append(text[cursor:start])
        cursor = end
    residue_parts.append(text[cursor:])
    residue = "".join(residue_parts)
    if _tool_like_marker_present(residue, marker_pairs):
        raise _tool_protocol_error("nested or unmatched <tool_call> block")
    calls: list[dict[str, Any]] = []
    for index, (_start, _end, block, parsed) in enumerate(envelopes):
        if parsed is None:
            parsed = _parse_tool_call_payload(
                block,
                allow_unclosed_function=True,
            )
        if parsed is None:
            raise _tool_protocol_error("unsupported tool_call payload format")
        name, arguments = parsed
        canonical_name = _canonical_tool_name_for_model_output(name, tools)
        if canonical_name is None:
            raise _tool_protocol_error(f"unknown tool '{name}'")
        arguments_value = _json_object_value(
            arguments,
            context=f"tool_call[{index}]",
        )
        arguments_value = _normalize_tool_arguments_for_schema(
            tool_name=canonical_name,
            arguments=arguments_value,
            tools=tools,
        )
        _validate_tool_arguments_for_schema(
            tool_name=canonical_name,
            arguments=arguments_value,
            tools=tools,
            context=f"tool_call[{index}]",
        )
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": canonical_name,
                    "arguments": json.dumps(
                        arguments_value,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            }
        )
    return calls


def _parse_generated_tool_calls_or_content(
    text: str,
    *,
    tools: list[dict[str, Any]],
    tokenizer: Any | None = None,
    state: Any | None = None,
    response_id: str | None = None,
    stream: bool = False,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    try:
        tool_calls = _parse_generated_tool_calls(
            text,
            tools=tools,
            tokenizer=tokenizer,
        )
    except HTTPException as exc:
        reason = _tool_protocol_reason(exc)
        _record_tool_parse_event(
            state,
            event=_tool_parse_counter_key(reason),
            reason=reason,
            response_id=response_id,
            stream=stream,
        )
        return None, reason
    if tool_calls:
        _record_tool_parse_event(
            state,
            event="tool_parse_success",
            response_id=response_id,
            stream=stream,
        )
    return tool_calls, None


def _stream_tool_call_deltas(
    tool_calls: list[dict[str, Any]],
    *,
    argument_chunk_chars: int,
):
    chunk_size = max(1, int(argument_chunk_chars))
    for index, tool_call in enumerate(tool_calls):
        function = tool_call.get("function") if isinstance(tool_call, dict) else None
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "")
        arguments = str(function.get("arguments") or "")
        yield {
            "tool_calls": [
                {
                    "index": index,
                    "id": str(tool_call.get("id") or f"call_{index}"),
                    "type": str(tool_call.get("type") or "function"),
                    "function": {"name": name, "arguments": ""},
                }
            ]
        }
        for start in range(0, len(arguments), chunk_size):
            yield {
                "tool_calls": [
                    {
                        "index": index,
                        "function": {
                            "arguments": arguments[start : start + chunk_size]
                        },
                    }
                ]
            }


def _find_casefold(haystack: str, needle: str, start: int = 0) -> int:
    return haystack.lower().find(needle.lower(), start)


def _tool_delta(
    index: int,
    *,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> dict[str, Any]:
    function: dict[str, Any] = {}
    if name is not None:
        function["name"] = name
    if arguments is not None:
        function["arguments"] = arguments
    item: dict[str, Any] = {"index": index, "function": function}
    if call_id is not None:
        item["id"] = call_id
        item["type"] = "function"
    return {"tool_calls": [item]}


class _ToolCallStreamParser:
    """Protocol-shaped base for streaming tool-call parsers."""

    dialect = "unknown"

    @property
    def tool_calls(self) -> list[dict[str, Any]] | None:
        return None

    @property
    def fallback_reason(self) -> str | None:
        return None

    @property
    def raw_text(self) -> str:
        return ""

    @property
    def started(self) -> bool:
        return False

    def feed(self, text: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    def finish(self) -> list[dict[str, Any]]:
        raise NotImplementedError


class _QwenXMLToolCallStreamParser(_ToolCallStreamParser):
    """Incrementally translate Qwen XML tool calls into OpenAI deltas.

    This parser handles the native shape Qwen tends to emit:
    `<tool_call><function=name><parameter=key>value</parameter>...</function></tool_call>`.
    It deliberately validates only the tool name early. Argument schema
    validation remains the client's job, but malformed XML never blocks the
    stream or gets treated as a successful tool call.
    """

    dialect = "qwen_xml"
    _PARAM_CLOSE = "</parameter>"
    _FUNCTION_CLOSE = "</function>"
    _TOOL_CALL_CLOSE = "</tool_call>"

    def __init__(
        self,
        *,
        tools: list[dict[str, Any]],
        call_index: int = 0,
        repair_unclosed_complete: bool = True,
    ) -> None:
        self._tools = tools
        self._known = {name for tool in tools if (name := _tool_spec_name(tool))}
        self._call_index = int(call_index)
        self._repair_unclosed_complete = bool(repair_unclosed_complete)
        self._call_id = f"call_{uuid.uuid4().hex[:24]}"
        self._buf = ""
        self._raw = ""
        self._stage = "find_function"
        self._name: str | None = None
        self._current_key: str | None = None
        self._current_value_parts: list[str] = []
        self._params: dict[str, Any] = {}
        self._done = False
        self._tool_calls: list[dict[str, Any]] | None = None
        self._fallback_reason: str | None = None
        self._started = False
        self._name_delta_emitted = False
        self._remaining_text = ""

    @property
    def tool_calls(self) -> list[dict[str, Any]] | None:
        return self._tool_calls

    @property
    def fallback_reason(self) -> str | None:
        return self._fallback_reason

    @property
    def raw_text(self) -> str:
        return self._raw

    @property
    def started(self) -> bool:
        return self._started

    @property
    def remaining_text(self) -> str:
        return self._remaining_text

    @property
    def in_known_tool_parameter(self) -> bool:
        return (
            bool(self._started)
            and bool(self._name)
            and self._name in self._known
            and self._stage == "in_parameter"
        )

    def _finish_call(self, deltas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._params = _normalize_tool_arguments_for_schema(
            tool_name=str(self._name or ""),
            arguments=self._params,
            tools=self._tools,
        )
        try:
            _validate_tool_arguments_for_schema(
                tool_name=str(self._name or ""),
                arguments=self._params,
                tools=self._tools,
                context=f"tool_call[{self._call_index}]",
            )
        except HTTPException as exc:
            self._fallback_reason = _tool_protocol_reason(exc)
            return []
        if not self._name_delta_emitted:
            deltas.append(
                _tool_delta(
                    self._call_index,
                    call_id=self._call_id,
                    name=str(self._name or ""),
                    arguments="",
                )
            )
            self._name_delta_emitted = True
        arguments = _json_object_string(
            self._params,
            context=f"tool_call[{self._call_index}]",
        )
        deltas.append(_tool_delta(self._call_index, arguments=arguments))
        self._done = True
        self._tool_calls = [
            {
                "id": self._call_id,
                "type": "function",
                "function": {
                    "name": str(self._name or ""),
                    "arguments": arguments,
                },
            }
        ]
        return deltas

    def feed(self, text: str) -> list[dict[str, Any]]:
        if self._done or self._fallback_reason:
            self._raw += text
            return []
        self._raw += text
        self._buf += text
        deltas: list[dict[str, Any]] = []
        while True:
            if self._stage == "find_function":
                function_start = _find_casefold(self._buf, "<function=")
                if function_start < 0:
                    return deltas
                function_end = self._buf.find(">", function_start)
                if function_end < 0:
                    return deltas
                raw_name = self._buf[function_start + len("<function=") : function_end]
                name = raw_name.strip()
                if not name:
                    self._fallback_reason = "tool_call function is missing a name"
                    return deltas
                canonical_name = _canonical_tool_name_for_model_output(
                    name,
                    self._tools,
                )
                if canonical_name is None:
                    self._fallback_reason = f"unknown tool '{name}'"
                    return deltas
                self._name = canonical_name
                self._started = True
                self._buf = self._buf[function_end + 1 :]
                self._stage = "find_parameter"
                continue

            if self._stage == "find_parameter":
                param_start = _find_casefold(self._buf, "<parameter=")
                function_close = _find_casefold(self._buf, self._FUNCTION_CLOSE)
                if function_close >= 0 and (
                    param_start < 0 or function_close < param_start
                ):
                    self._buf = self._buf[function_close + len(self._FUNCTION_CLOSE) :]
                    self._stage = "after_function"
                    continue
                if param_start < 0:
                    return deltas
                param_end = self._buf.find(">", param_start)
                if param_end < 0:
                    return deltas
                raw_key = self._buf[param_start + len("<parameter=") : param_end]
                key = raw_key.strip()
                if not key:
                    self._fallback_reason = (
                        f"tool '{self._name}' contains an empty parameter name"
                    )
                    return deltas
                self._current_key = key
                self._current_value_parts = []
                self._buf = self._buf[param_end + 1 :]
                self._stage = "in_parameter"
                continue

            if self._stage == "in_parameter":
                close_start = _find_casefold(self._buf, self._PARAM_CLOSE)
                if close_start < 0:
                    hold = len(self._PARAM_CLOSE) - 1
                    if len(self._buf) <= hold:
                        return deltas
                    value_piece = self._buf[:-hold]
                    self._buf = self._buf[-hold:]
                    if not self._current_value_parts and value_piece.startswith("\n"):
                        value_piece = value_piece[1:]
                    if value_piece:
                        self._current_value_parts.append(value_piece)
                    return deltas
                value_piece = self._buf[:close_start]
                # Qwen usually formats XML parameters as newline-delimited
                # blocks. Preserve user payload text, but drop the formatting
                # newline immediately before the closing tag to match the
                # existing final parser's stripped value semantics.
                if value_piece.endswith("\n"):
                    value_piece = value_piece[:-1]
                if not self._current_value_parts and value_piece.startswith("\n"):
                    value_piece = value_piece[1:]
                if value_piece:
                    self._current_value_parts.append(value_piece)
                key = str(self._current_key or "")
                self._params[key] = _decode_tool_parameter_value(
                    "".join(self._current_value_parts),
                    schema=_tool_parameter_schema(
                        self._tools,
                        tool_name=self._name,
                        parameter_name=key,
                    ),
                )
                self._buf = self._buf[close_start + len(self._PARAM_CLOSE) :]
                self._stage = "find_parameter"
                continue

            if self._stage == "after_function":
                tool_close = _find_casefold(self._buf, self._TOOL_CALL_CLOSE)
                if tool_close < 0:
                    return deltas
                self._remaining_text = self._buf[
                    tool_close + len(self._TOOL_CALL_CLOSE) :
                ]
                self._buf = ""
                return self._finish_call(deltas)

            return deltas

    def finish(self) -> list[dict[str, Any]]:
        if self._done or self._fallback_reason:
            return []
        deltas = self.feed("")
        if self._done or self._fallback_reason:
            return deltas
        if self._repair_unclosed_complete and self._started and self._name and (
            self._stage == "after_function"
            or (self._stage == "find_parameter" and bool(self._params))
        ):
            return self._finish_call(deltas)
        if self._started:
            self._fallback_reason = "unclosed <tool_call> block"
            return []
        return []


class _BufferedFallbackToolCallParser(_ToolCallStreamParser):
    """Final-parse fallback for JSON-style or unknown tool-call payloads."""

    dialect = "buffered"

    def __init__(
        self,
        *,
        tools: list[dict[str, Any]],
        argument_chunk_chars: int,
        tokenizer: Any | None = None,
        repair_unclosed_complete: bool = True,
    ) -> None:
        self._tools = tools
        self._argument_chunk_chars = max(1, int(argument_chunk_chars))
        self._tokenizer = tokenizer
        self._repair_unclosed_complete = bool(repair_unclosed_complete)
        self._raw = ""
        self._tool_calls: list[dict[str, Any]] | None = None
        self._fallback_reason: str | None = None

    @property
    def tool_calls(self) -> list[dict[str, Any]] | None:
        return self._tool_calls

    @property
    def fallback_reason(self) -> str | None:
        return self._fallback_reason

    @property
    def raw_text(self) -> str:
        return self._raw

    def feed(self, text: str) -> list[dict[str, Any]]:
        self._raw += text
        return []

    def finish(self) -> list[dict[str, Any]]:
        try:
            tool_calls = _parse_generated_tool_calls(
                self._raw,
                tools=self._tools,
                tokenizer=self._tokenizer,
            )
        except HTTPException as exc:
            self._fallback_reason = _tool_protocol_reason(exc)
            return []
        if not tool_calls:
            return []
        self._tool_calls = tool_calls
        return list(
            _stream_tool_call_deltas(
                tool_calls,
                argument_chunk_chars=self._argument_chunk_chars,
            )
        )


class _ToolAwareContentStreamTranslator:
    """Translate streamed assistant text into OpenAI tool-call deltas when needed."""

    _START_MARKER = "<tool_call"
    _CLOSE_MARKER = "</tool_call>"

    def __init__(
        self,
        *,
        tools: list[dict[str, Any]],
        argument_chunk_chars: int,
        tokenizer: Any | None = None,
        repair_unclosed_complete: bool = True,
    ) -> None:
        self._tools = tools
        self._argument_chunk_chars = max(1, int(argument_chunk_chars))
        self._tokenizer = tokenizer
        self._repair_unclosed_complete = bool(repair_unclosed_complete)
        self._marker_pairs = _tool_marker_pairs_from_tokenizer(tokenizer)
        self._pending = ""
        self._trailing = ""
        self._mode = "passthrough" if not tools else "undecided"
        self._tool_parser: _ToolCallStreamParser | None = None
        self.tool_calls: list[dict[str, Any]] | None = None
        self.fallback_reason: str | None = None
        self.tool_parser_dialect: str | None = None
        self._suppress_remaining_tool_text = False
        self._emitted_tool_deltas = False
        self._suppressed_tool_markup = False

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def has_emitted_tool_deltas(self) -> bool:
        return self._emitted_tool_deltas

    @property
    def suppressed_tool_markup(self) -> bool:
        return self._suppressed_tool_markup

    @property
    def buffering_tool_call(self) -> bool:
        return self._mode == "tool" or self._tool_parser is not None

    @property
    def tool_argument_in_progress(self) -> bool:
        parser = self._tool_parser
        return bool(getattr(parser, "in_known_tool_parameter", False))

    @property
    def ready_to_finish_tool_turn(self) -> bool:
        """A valid tool-call assistant turn has reached a protocol boundary.

        OpenAI clients do not need the model to keep free-running after it has
        emitted a complete tool-call message. Once the current parser is done
        and only whitespace remains, the server can finish the stream with
        ``finish_reason=tool_calls`` and cancel further generation. This is not
        a content guardrail; it is the protocol stop point for tool execution.
        """

        return (
            bool(self.tool_calls)
            and self._mode == "done"
            and self._tool_parser is None
            and not self._pending
            and not self._trailing.strip()
        )

    @property
    def invalid_trailing_after_tool_call(self) -> bool:
        if not self.tool_calls or self._mode != "done" or self._tool_parser is not None:
            return False
        trailing = self._trailing.lstrip()
        if not trailing:
            return False
        if self._could_grow_into_marker(trailing):
            return False
        if self._find_tool_start(trailing) == 0:
            return False
        return True

    def _fallback_raw_tool_text_as_content(
        self,
        raw_text: str,
        *,
        emitted_tool_deltas: bool,
    ) -> list[dict[str, Any]]:
        """Preserve visible prose around malformed tool attempts, never markup."""
        self._tool_parser = None
        visible_text = _visible_malformed_tool_content(raw_text, self._tokenizer)
        if visible_text != raw_text:
            self._suppressed_tool_markup = True
        self._mode = "done"
        self._suppress_remaining_tool_text = True
        if emitted_tool_deltas or self.tool_calls:
            return []
        return [{"content": visible_text}] if visible_text else []

    def _content_delta(self, text: str) -> list[dict[str, Any]]:
        visible_text = _strip_orphan_tool_control_markup(text)
        if visible_text != text:
            self._suppressed_tool_markup = True
        return [{"content": visible_text}] if visible_text else []

    def feed(self, field: str, text: str) -> list[dict[str, Any]]:
        if not text:
            return []
        if field != "content" or self._mode == "passthrough":
            return [{field: text}]
        if self._mode == "done":
            if self._suppress_remaining_tool_text:
                return []
            self._trailing += text
            idx = self._find_tool_start(self._trailing)
            if idx >= 0 and not self._trailing[:idx].strip():
                self._pending = self._trailing[idx:]
                self._trailing = ""
                self._mode = "tool"
                return self._tool_deltas_if_complete(final=False)
            return []

        self._pending += text
        if self._mode == "tool":
            return self._tool_deltas_if_complete(final=False)

        # Modes "undecided" and "content" both need to scan the pending buffer
        # for `<tool_call>` markers. Previously "content" mode locked in on the
        # first non-marker byte and never re-checked, so any preamble text
        # followed by a tool_call block (a common Qwen3.6 27B output shape when
        # given long system prompts) caused the entire response - including the
        # tool_call markup - to leak through as `delta.content`. Clients then
        # saw zero `delta.tool_calls` and the agent loop exited with no work
        # done. Now both modes look for the marker; "content" mode also holds
        # any trailing partial-marker bytes so the marker can complete on a
        # later chunk.

        idx = self._find_tool_start(self._pending)
        if idx >= 0:
            deltas: list[dict[str, Any]] = []
            if idx > 0:
                pre = self._sanitize_prefix_before_tool(self._pending[:idx])
                # In undecided mode, leading whitespace before the marker is
                # decoration, not content - drop it to match the original
                # behaviour for tool-only responses with a leading newline.
                if self._mode == "undecided" and not pre.strip():
                    pass
                else:
                    deltas.extend(self._content_delta(pre))
            self._pending = self._pending[idx:]
            self._mode = "tool"
            deltas.extend(self._tool_deltas_if_complete(final=False))
            return deltas

        # No complete marker found in the pending buffer. We may still have a
        # partial marker on the tail (e.g. pending ends with `<tool_ca`) that
        # could complete on the next chunk; hold those bytes back.
        held = self._partial_marker_tail_len(self._pending)

        if self._mode == "undecided":
            stripped = self._pending.lstrip()
            if not stripped:
                return []
            # Whole stripped pending could still grow into a marker; keep waiting
            # (preserves the original behaviour for the tool-only case).
            if self._could_grow_into_marker(stripped):
                return []
            # Commit to content mode but keep any partial-marker tail for the
            # next chunk to potentially complete the marker.
            self._mode = "content"

        if held == 0 or held >= len(self._pending):
            # Either no partial marker (emit all) or pending is entirely a
            # partial marker (hold all, wait for more).
            if held >= len(self._pending):
                return []
            content = self._pending
            self._pending = ""
            return self._content_delta(content)

        # Emit everything up to the partial-marker tail; hold the tail.
        pre = self._pending[:-held]
        self._pending = self._pending[-held:]
        return self._content_delta(pre)

    def _find_tool_start(self, text: str) -> int:
        candidates: list[int] = []
        lowered = text.lower()
        idx = lowered.find(self._START_MARKER)
        if idx >= 0:
            candidates.append(idx)
        ns_match = _NAMESPACED_TOOL_CALL_START_RE.search(text)
        if ns_match:
            candidates.append(ns_match.start())
        for prefix in _BRACKET_TOOL_PREFIXES:
            bracket_idx = lowered.find(prefix.lower())
            while bracket_idx >= 0:
                candidate = text[bracket_idx:]
                if _BRACKET_TOOL_CALL_RE.match(candidate):
                    candidates.append(bracket_idx)
                    break
                close_idx = candidate.find("]")
                if close_idx < 0:
                    break
                bracket_idx = lowered.find(prefix.lower(), bracket_idx + 1)
        for start_marker, _end_marker in self._marker_pairs:
            custom_idx = _find_casefold(text, start_marker)
            if custom_idx >= 0:
                candidates.append(custom_idx)
        return min(candidates) if candidates else -1

    def _sanitize_prefix_before_tool(self, text: str) -> str:
        if not any(prefix in text for prefix in _BRACKET_TOOL_PREFIXES):
            return text
        out: list[str] = []
        cursor = 0
        while cursor < len(text):
            bracket_idx = -1
            bracket_prefix = ""
            for prefix in _BRACKET_TOOL_PREFIXES:
                idx = text.find(prefix, cursor)
                if idx >= 0 and (bracket_idx < 0 or idx < bracket_idx):
                    bracket_idx = idx
                    bracket_prefix = prefix
            if bracket_idx < 0:
                out.append(text[cursor:])
                break
            out.append(text[cursor:bracket_idx])
            after_prefix = bracket_idx + len(bracket_prefix)
            close_idx = text.find("]", after_prefix)
            if close_idx < 0:
                cursor = after_prefix
                continue
            out.append(text[bracket_idx : close_idx + 1])
            cursor = close_idx + 1
        return "".join(out)

    def _could_grow_into_marker(self, text: str) -> bool:
        lowered = text.lower()
        if self._START_MARKER.startswith(lowered):
            return True
        if any(prefix.lower().startswith(lowered) for prefix in _BRACKET_TOOL_PREFIXES):
            return True
        if self._could_be_partial_namespaced_open(text):
            return True
        return any(
            start.lower().startswith(lowered) for start, _end in self._marker_pairs
        )

    @staticmethod
    def _partial_prefix_len(text: str, marker: str) -> int:
        max_len = min(len(text), len(marker) - 1)
        for n in range(max_len, 0, -1):
            if text.lower().endswith(marker[:n].lower()):
                return n
        return 0

    @staticmethod
    def _could_be_partial_namespaced_open(candidate: str) -> bool:
        if not candidate.startswith("<") or ">" in candidate:
            return False
        body = candidate[1:]
        if not body:
            return True
        if body.startswith("/"):
            return False
        if ":" not in body:
            return re.match(r"^[A-Za-z_][\w.-]*$", body) is not None
        namespace, suffix = body.split(":", 1)
        if not re.match(r"^[A-Za-z_][\w.-]*$", namespace):
            return False
        return "tool_call".startswith(suffix.lower())

    def _partial_marker_tail_len(self, text: str) -> int:
        """Return the length of the trailing suffix of `text` that is a
        possible tool-call opening marker. 0 if no suffix matches.

        Used to hold back bytes that *could* be the start of a tool_call
        marker spanning multiple stream chunks, so the marker can complete
        on a later chunk instead of being emitted as content prematurely."""
        keep = self._partial_prefix_len(text, self._START_MARKER)
        for start_marker, _end_marker in self._marker_pairs:
            keep = max(keep, self._partial_prefix_len(text, start_marker))
        for prefix in _BRACKET_TOOL_PREFIXES:
            keep = max(keep, self._partial_prefix_len(text, prefix))
        last_lt = text.rfind("<")
        if last_lt >= 0:
            candidate = text[last_lt:]
            if self._could_be_partial_namespaced_open(candidate):
                keep = max(keep, len(candidate))
        lowered = text.lower()
        bracket_idx = -1
        for prefix in _BRACKET_TOOL_PREFIXES:
            idx = lowered.rfind(prefix.lower())
            if idx > bracket_idx:
                bracket_idx = idx
        if bracket_idx >= 0 and "]" not in text[bracket_idx:]:
            return max(keep, len(text) - bracket_idx)
        return min(keep, 128)

    def finish(self) -> list[dict[str, Any]]:
        if self._mode == "tool":
            return self._tool_deltas_if_complete(final=True)
        if self._mode == "done":
            if self._suppress_remaining_tool_text:
                self._trailing = ""
                return []
            if self._trailing.strip():
                stripped_trailing = self._trailing.strip().lower()
                if self._CLOSE_MARKER.endswith(stripped_trailing):
                    self._trailing = ""
                    return []
                cleaned = re.sub(
                    r"^\s*</tool_call>\s*",
                    "",
                    self._trailing,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if not cleaned.strip():
                    self._trailing = ""
                    return []
                self.fallback_reason = "text after tool_call block"
                trailing = cleaned
                self._trailing = ""
                self._mode = "content"
                return self._content_delta(trailing)
            return []
        if self._pending:
            content = self._pending
            self._pending = ""
            self._mode = "content"
            return self._content_delta(content)
        return []

    def _tool_deltas_if_complete(self, *, final: bool) -> list[dict[str, Any]]:
        if self._tool_parser is None:
            self._tool_parser = _QwenXMLToolCallStreamParser(
                tools=self._tools,
                call_index=len(self.tool_calls or []),
                repair_unclosed_complete=self._repair_unclosed_complete,
            )
            self.tool_parser_dialect = self._tool_parser.dialect
        chunk = self._pending
        self._pending = ""
        deltas: list[dict[str, Any]] = []

        while True:
            assert self._tool_parser is not None
            fed_deltas = self._tool_parser.feed(chunk)
            if any(delta.get("tool_calls") for delta in fed_deltas):
                self._emitted_tool_deltas = True
            deltas.extend(fed_deltas)
            chunk = ""
            if self._tool_parser.fallback_reason:
                self.fallback_reason = self._tool_parser.fallback_reason
                return self._fallback_raw_tool_text_as_content(
                    self._tool_parser.raw_text,
                    emitted_tool_deltas=bool(deltas) or self._emitted_tool_deltas,
                )
            if self._tool_parser.tool_calls:
                self.tool_calls = (self.tool_calls or []) + self._tool_parser.tool_calls
                remaining = getattr(self._tool_parser, "remaining_text", "")
                self._tool_parser = None
                if not remaining:
                    self._mode = "done"
                    return deltas
                idx = self._find_tool_start(remaining)
                if idx >= 0 and not remaining[:idx].strip():
                    self._tool_parser = _QwenXMLToolCallStreamParser(
                        tools=self._tools,
                        call_index=len(self.tool_calls or []),
                        repair_unclosed_complete=self._repair_unclosed_complete,
                    )
                    self.tool_parser_dialect = self._tool_parser.dialect
                    chunk = remaining[idx:]
                    self._mode = "tool"
                    continue
                self._trailing += remaining
                self._mode = "done"
                return deltas
            if (
                not final
                and _find_casefold(self._tool_parser.raw_text, self._CLOSE_MARKER) >= 0
                and _find_casefold(self._tool_parser.raw_text, "<function=") < 0
            ):
                buffered_deltas = self._complete_buffered_tool_call(
                    self._tool_parser.raw_text,
                    emitted_tool_deltas=bool(deltas) or self._emitted_tool_deltas,
                )
                if any(delta.get("tool_calls") for delta in buffered_deltas):
                    self._emitted_tool_deltas = True
                return deltas + buffered_deltas
            if not final:
                return deltas

            final_deltas = self._tool_parser.finish()
            if final_deltas:
                if any(delta.get("tool_calls") for delta in final_deltas):
                    self._emitted_tool_deltas = True
                deltas.extend(final_deltas)
            if self._tool_parser.tool_calls:
                self.tool_calls = (self.tool_calls or []) + self._tool_parser.tool_calls
                remaining = getattr(self._tool_parser, "remaining_text", "")
                self._tool_parser = None
                if not remaining:
                    self._mode = "done"
                    return deltas
                idx = self._find_tool_start(remaining)
                if idx >= 0 and not remaining[:idx].strip():
                    self._tool_parser = _QwenXMLToolCallStreamParser(
                        tools=self._tools,
                        call_index=len(self.tool_calls or []),
                        repair_unclosed_complete=self._repair_unclosed_complete,
                    )
                    self.tool_parser_dialect = self._tool_parser.dialect
                    chunk = remaining[idx:]
                    self._mode = "tool"
                    continue
                self._trailing += remaining
                self._mode = "done"
                return deltas
            if self._tool_parser.fallback_reason:
                self.fallback_reason = self._tool_parser.fallback_reason
                return deltas + self._fallback_raw_tool_text_as_content(
                    self._tool_parser.raw_text,
                    emitted_tool_deltas=bool(deltas) or self._emitted_tool_deltas,
                )
            buffered = _BufferedFallbackToolCallParser(
                tools=self._tools,
                argument_chunk_chars=self._argument_chunk_chars,
                tokenizer=self._tokenizer,
                repair_unclosed_complete=self._repair_unclosed_complete,
            )
            self.tool_parser_dialect = buffered.dialect
            buffered.feed(self._tool_parser.raw_text)
            buffered_deltas = buffered.finish()
            self._tool_parser = None
            if buffered.tool_calls:
                self.tool_calls = (self.tool_calls or []) + buffered.tool_calls
                self._mode = "done"
                if any(delta.get("tool_calls") for delta in buffered_deltas):
                    self._emitted_tool_deltas = True
                return deltas + buffered_deltas
            if buffered.fallback_reason:
                self.fallback_reason = buffered.fallback_reason
                return deltas + self._fallback_raw_tool_text_as_content(
                    buffered.raw_text,
                    emitted_tool_deltas=bool(deltas) or self._emitted_tool_deltas,
                )
            return deltas + self._fallback_raw_tool_text_as_content(
                buffered.raw_text,
                emitted_tool_deltas=bool(deltas) or self._emitted_tool_deltas,
            )

    def _complete_buffered_tool_call(
        self,
        raw_text: str,
        *,
        emitted_tool_deltas: bool,
    ) -> list[dict[str, Any]]:
        """Parse a closed non-XML tool block at the stream protocol boundary."""
        buffered = _BufferedFallbackToolCallParser(
            tools=self._tools,
            argument_chunk_chars=self._argument_chunk_chars,
            tokenizer=self._tokenizer,
            repair_unclosed_complete=self._repair_unclosed_complete,
        )
        self.tool_parser_dialect = buffered.dialect
        buffered.feed(raw_text)
        buffered_deltas = buffered.finish()
        self._tool_parser = None
        if buffered.tool_calls:
            self.tool_calls = (self.tool_calls or []) + buffered.tool_calls
            self._mode = "done"
            return buffered_deltas
        if buffered.fallback_reason:
            self.fallback_reason = buffered.fallback_reason
        else:
            self.fallback_reason = "unrecognized <tool_call> payload"
        return self._fallback_raw_tool_text_as_content(
            raw_text,
            emitted_tool_deltas=emitted_tool_deltas,
        )


def _template_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function")
    if isinstance(function, dict):
        name = str(function.get("name") or "").strip()
        arguments = function.get("arguments", {})
    else:
        name = str(tool_call.get("name") or "").strip()
        arguments = tool_call.get("arguments", {})
    if not name:
        raise HTTPException(
            status_code=400, detail="assistant tool_call is missing a name"
        )
    arguments_text = _json_object_string(
        arguments, context=f"assistant tool_call '{name}'"
    )
    normalized = {
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.loads(arguments_text),
        },
    }
    call_id = tool_call.get("id")
    if call_id:
        normalized["id"] = str(call_id)
    return normalized


def _message_extra_map(message: ChatMessage) -> dict[str, Any]:
    extra = getattr(message, "model_extra", None)
    return extra if isinstance(extra, dict) else {}


def _message_extra(message: ChatMessage, key: str, default: Any = None) -> Any:
    value = getattr(message, key, None)
    if value is not None:
        return value
    return _message_extra_map(message).get(key, default)


def _copy_chat_message(message: ChatMessage, **updates: Any) -> ChatMessage:
    try:
        return message.model_copy(update=updates)
    except AttributeError:
        data = message.dict()
        data.update(updates)
        return ChatMessage(**data)


def _retry_user_key(text: str) -> str | None:
    key = re.sub(r"\s+", " ", (text or "").strip()).strip().lower()
    key = key.strip(" \t\r\n")
    if len(key) < 8 or len(key) > 4_000:
        return None
    return key


_SIMPLE_CHITCHAT_COMPACT_CANONICAL: dict[str, str] = {
    "hi": "hi",
    "hello": "hello",
    "hey": "hey",
    "yo": "yo",
    "sup": "sup",
    "howdy": "howdy",
    "hithere": "hi there",
    "hellothere": "hello there",
    "heythere": "hey there",
    "hihowareyou": "hi, how are you?",
    "hellohowareyou": "hello, how are you?",
    "heyhowareyou": "hey, how are you?",
    "hihowyou": "hi, how are you?",
    "heyhowyou": "hey, how are you?",
    "hihowareudoing": "hi, how are you doing?",
    "hihowareyoudoing": "hi, how are you doing?",
    "heyhowareyoudoing": "hey, how are you doing?",
    "howareyou": "how are you?",
    "howareu": "how are you?",
    "howyou": "how are you?",
    "howru": "how are you?",
    "howsitgoing": "how's it going?",
    "whatsup": "what's up?",
}


def _simple_chitchat_compact_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").strip().lower())


def _collapse_repeated_simple_chitchat_text(text: str) -> str | None:
    compact = _simple_chitchat_compact_key(text)
    if not compact or len(compact) > 160:
        return None
    for key, canonical in sorted(
        _SIMPLE_CHITCHAT_COMPACT_CANONICAL.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if not key or len(compact) <= len(key) or len(compact) % len(key) != 0:
            continue
        repeat_count = len(compact) // len(key)
        if 2 <= repeat_count <= 6 and compact == key * repeat_count:
            return canonical
    return None


def _is_compact_simple_chitchat_key(compact: str) -> bool:
    if compact in _SIMPLE_CHITCHAT_COMPACT_CANONICAL:
        return True
    return _collapse_repeated_simple_chitchat_text(compact) is not None


def _collapse_repeated_user_text(text: str) -> str | None:
    raw = text or ""
    stripped = raw.strip()
    key = _retry_user_key(stripped)
    if key is not None and len(stripped) >= 16:
        # OpenCode can persist a cancelled retry as one user part like
        # "Hi, how are you?Hi, how are you?". Collapse exact tandem repeats
        # first so the user's original casing/punctuation survives.
        for split in range(8, len(stripped)):
            first = stripped[:split].strip()
            second = stripped[split:].strip()
            if not first or not second:
                continue
            if _retry_user_key(first) == _retry_user_key(second):
                return first
    collapsed_chitchat = _collapse_repeated_simple_chitchat_text(stripped)
    if collapsed_chitchat is not None:
        return collapsed_chitchat
    return None


def _canonicalize_user_retry_pollution(
    messages: list[ChatMessage],
    stats: AgentTranscriptCanonicalization,
) -> list[ChatMessage]:
    if not messages:
        return messages
    canonical: list[ChatMessage] = []
    for message in messages:
        role = str(message.role).lower()
        candidate = message
        if role == "user":
            text = _content_to_text(candidate.content)
            collapsed = _collapse_repeated_user_text(text)
            if collapsed is not None:
                candidate = _copy_chat_message(candidate, content=collapsed)
                stats.collapsed_repeated_user_messages += 1
                stats.collapsed_repeated_user_chars += max(0, len(text) - len(collapsed))

        if (
            role == "user"
            and canonical
            and str(canonical[-1].role).lower() == "user"
        ):
            previous = canonical[-1]
            previous_text = _content_to_text(previous.content).strip()
            current_text = _content_to_text(candidate.content).strip()
            previous_key = _retry_user_key(previous_text)
            current_key = _retry_user_key(current_text)
            if previous_key is not None and previous_key == current_key:
                canonical[-1] = candidate
                stats.dropped_duplicate_user_messages += 1
                stats.dropped_duplicate_user_chars += len(previous_text)
                continue
            if _is_simple_chitchat_text(previous_text) and _is_simple_chitchat_text(
                current_text
            ):
                canonical[-1] = candidate
                stats.dropped_duplicate_user_messages += 1
                stats.dropped_duplicate_user_chars += len(previous_text)
                continue
            if stats.skipped_repeated_assistant_messages:
                canonical.append(candidate)
                continue
            merged = "\n\n".join(part for part in (previous_text, current_text) if part)
            canonical[-1] = _copy_chat_message(candidate, content=merged)
            stats.merged_consecutive_user_messages += 1
            stats.merged_consecutive_user_chars += len(previous_text)
            continue

        canonical.append(candidate)
    return canonical


def _canonicalize_simple_chitchat_history(
    messages: list[ChatMessage],
    stats: AgentTranscriptCanonicalization,
    *,
    replace_system_prompt: bool = False,
) -> list[ChatMessage]:
    latest_user_index: int | None = None
    latest_user_text = ""
    for index, message in enumerate(messages):
        if str(message.role).lower() != "user":
            continue
        latest_user_index = index
        latest_user_text = _content_to_text(message.content)
    if latest_user_index is None or not _is_simple_chitchat_text(latest_user_text):
        return messages

    if replace_system_prompt:
        system_messages = [
            message
            for message in messages[:latest_user_index]
            if str(message.role).lower() == "system"
        ]
        preserved = [
            ChatMessage(role="system", content=_MTPLX_SIMPLE_CHAT_SYSTEM_PROMPT)
        ]
        stats.replaced_simple_chitchat_system_messages += len(system_messages)
        stats.replaced_simple_chitchat_system_chars += sum(
            len(_content_to_text(message.content)) for message in system_messages
        )
        stats.injected_simple_chitchat_system_chars += len(
            _MTPLX_SIMPLE_CHAT_SYSTEM_PROMPT
        )
    else:
        preserved = [
            message
            for message in messages[:latest_user_index]
            if str(message.role).lower() == "system"
        ]
    preserved.append(messages[latest_user_index])
    if len(preserved) == len(messages) and not replace_system_prompt:
        return messages

    preserved_ids = {id(message) for message in preserved}
    dropped = [message for message in messages if id(message) not in preserved_ids]
    stats.dropped_simple_chitchat_history_messages += len(dropped)
    stats.dropped_simple_chitchat_history_chars += sum(
        len(_content_to_text(message.content)) for message in dropped
    )
    return preserved


def _replace_client_system_prompt(
    messages: list[ChatMessage],
    stats: AgentTranscriptCanonicalization,
    *,
    replacement: str | None,
) -> list[ChatMessage]:
    if not replacement:
        return messages
    first_non_system = 0
    while first_non_system < len(messages):
        role = str(messages[first_non_system].role).strip().lower()
        if role != "system":
            break
        first_non_system += 1
    leading_system = messages[:first_non_system]
    if (
        len(leading_system) == 1
        and _content_to_text(leading_system[0].content) == replacement
    ):
        return messages
    updated = [ChatMessage(role="system", content=replacement), *messages[first_non_system:]]
    stats.replaced_client_system_messages += len(leading_system)
    stats.replaced_client_system_chars += sum(
        len(_content_to_text(message.content)) for message in leading_system
    )
    stats.injected_client_system_chars += len(replacement)
    return updated


def _with_backend_chat_policy(
    state: "ServerState",
    messages: list[ChatMessage],
) -> tuple[list[ChatMessage], bool]:
    backend = _backend_descriptor(state)
    if backend.model_family != "step":
        return messages, False
    if not messages:
        return [ChatMessage(role="system", content=_MTPLX_STEP_LANGUAGE_POLICY)], True

    updated = list(messages)
    first = updated[0]
    if str(first.role).lower() == "system":
        content = _content_to_text(first.content)
        if _MTPLX_STEP_LANGUAGE_POLICY_SENTINEL in content:
            return messages, False
        updated[0] = _copy_chat_message(
            first,
            content=f"{content}\n\n{_MTPLX_STEP_LANGUAGE_POLICY}" if content else _MTPLX_STEP_LANGUAGE_POLICY,
        )
        return updated, True
    return [ChatMessage(role="system", content=_MTPLX_STEP_LANGUAGE_POLICY), *updated], True


def _message_declares_aborted_assistant_turn(message: ChatMessage) -> bool:
    if message.role != "assistant" or message.tool_calls:
        return False
    extra = _message_extra_map(message)
    status_values = [
        extra.get("status"),
        extra.get("finish_reason"),
        extra.get("error"),
        extra.get("stop_reason"),
        _message_extra(message, "status"),
        _message_extra(message, "finish_reason"),
        _message_extra(message, "error"),
        _message_extra(message, "stop_reason"),
    ]
    for value in status_values:
        if value is None:
            continue
        text = str(value).lower()
        if text in {"abort", "aborted", "cancel", "cancelled", "error"}:
            return True
        if "abort" in text or "cancel" in text or "messageabortederror" in text:
            return True
    return False


def _looks_like_orphan_chitchat_assistant_turn(
    messages: list[ChatMessage],
    index: int,
) -> bool:
    message = messages[index]
    if message.role != "assistant" or message.tool_calls:
        return False
    content = _content_to_text(message.content).strip()
    if not content or len(content) > 96:
        return False
    if index <= 0 or index + 1 >= len(messages):
        return False
    previous = messages[index - 1]
    following = messages[index + 1]
    if previous.role != "user" or following.role != "user":
        return False
    return _is_simple_chitchat_text(
        _content_to_text(previous.content)
    ) and _is_simple_chitchat_text(_content_to_text(following.content))


_AGENT_PREAMBLE_MARKERS = (
    "let me ",
    "i'll ",
    "i will ",
    "almost there",
    "checking ",
    "now let me ",
    "now fixing",
)
_AGENT_ACTION_MARKERS = (
    "fix",
    "check",
    "run",
    "write",
    "edit",
    "update",
    "continue",
    "remaining",
    "errors",
    "typecheck",
    "typescript",
    "build",
    "parallel",
    "tool",
)
_TOOL_RESULT_COMPACT_THRESHOLD_CHARS = 6_000
_TOOL_RESULT_COMPACT_HEAD_CHARS = 128
_TOOL_RESULT_COMPACT_TAIL_CHARS = 128
_ACTIVE_READ_COMPACT_THRESHOLD_CHARS = 4_000
_ACTIVE_READ_COMPACT_HEAD_LINES = 16
_ACTIVE_READ_COMPACT_TAIL_LINES = 8
_ACTIVE_READ_COMPACT_MAX_LINES = 72
_ACTIVE_READ_COMPACT_CONTEXT_LINES = 1
_ACTIVE_READ_INSPECTION_COMPACT_THRESHOLD_CHARS = 1_800
_ACTIVE_READ_INSPECTION_COMPACT_HEAD_LINES = 2
_ACTIVE_READ_INSPECTION_COMPACT_TAIL_LINES = 1
_ACTIVE_READ_INSPECTION_COMPACT_MAX_LINES = 48
_ACTIVE_READ_INSPECTION_COMPACT_CONTEXT_LINES = 0
_ACTIVE_READ_INSPECTION_LINE_MAX_CHARS = 180
_ACTIVE_READ_INSPECTION_TOTAL_MAX_LINES = 96
_ACTIVE_READ_INSPECTION_MIN_LINES_PER_FILE = 16
_ACTIVE_READ_INSPECTION_MULTI_FILE_LINE_MAX_CHARS = 150
_ACTIVE_TOOL_RESULT_COMPACT_THRESHOLD_CHARS = 4_000
_ACTIVE_TOOL_RESULT_COMPACT_HEAD_LINES = 8
_ACTIVE_TOOL_RESULT_COMPACT_TAIL_LINES = 4
_ACTIVE_TOOL_RESULT_COMPACT_MAX_LINES = 48
_ACTIVE_TOOL_RESULT_LINE_MAX_CHARS = 280
_LINE_NUMBERED_CONTENT_RE = re.compile(r"^\s*(\d+):\s?(.*)$")
_READ_CONTINUATION_HINT_RE = re.compile(
    r"\(\s*Showing lines\b.*?\bUse offset=\d+\s+to continue\.\s*\)",
    re.IGNORECASE | re.DOTALL,
)
_READ_ONLY_INSPECTION_REQUEST_RE = re.compile(
    r"\b("
    r"review|evaluate|evaluation|assess|audit|inspect|summari[sz]e|"
    r"quality|strengths?|weaknesses?|improvement\s+plan|diagnos(?:e|is)|"
    r"identify(?:\s+the)?\s+issue|read-only|look\s+through"
    r")\b",
    re.IGNORECASE,
)
_READ_ONLY_RECOMMENDATION_REQUEST_RE = re.compile(
    r"\b(?:what|which)\s+(?:upgrade|upgrades|improvement|improvements|change|changes)\s+"
    r"(?:(?:do|would)\s+you\s+(?:think\s+)?(?:i|we)\s+should|should\s+(?:i|we))\s+"
    r"(?:do|make)\b"
    r"|\b(?:recommend|suggest)\b.{0,80}\b(?:upgrade|upgrades|improvement|improvements|change|changes)\b",
    re.IGNORECASE,
)
_MUTATING_REQUEST_RE = re.compile(
    r"\b("
    r"fix|implement|edit|write|create|build|modify|add|remove|delete|"
    r"refactor|rewrite|generate|install|update|upgrade|change|patch"
    r")\b",
    re.IGNORECASE,
)
_ACTIVE_READ_PRIORITY_ANCHOR_RE = re.compile(
    r"\b("
    r"collid\w*|intersect\w*|distanceTo|raycast\w*|arrow\w*|terrain\w*|"
    r"obstacle\w*|hit\w*|damage\w*|health\w*|embed\w*|isEmbedded|"
    r"checkArrowCollisions|checkCollisionWithCharacter|getHeight|"
    r"addEventListener|touchstart|keydown|visibilitychange|localStorage|"
    r"requestAnimationFrame|WebGLRenderer|BoxGeometry|MeshStandardMaterial|"
    r"SphereGeometry|PlaneGeometry|ShaderMaterial|BufferGeometry|PointsMaterial|"
    r"Clock|MathUtils|Date\.now|clone\(|setPixelRatio|shadowMap|"
    r"dispose|scene\.remove"
    r")\b",
    re.IGNORECASE,
)
_ACTIVE_READ_INSPECTION_PRIORITY_ANCHOR_RE = re.compile(
    r"(?:"
    r"\bfunction\s+(?:setDifficulty|checkCollision|animate|resetGame|flap|die)\b|"
    r"\b(?:"
    r"class|constructor|interface|type|enum|export|public|private|protected|"
    r"localStorage|addEventListener|touchstart|keydown|resize|visibilitychange|"
    r"checkCollision|requestAnimationFrame|Clock|getDelta|Date\.now|"
    r"scene\.remove|dispose|pipes\.splice|pipeRemoveZ|server\.listen|"
    r"fs\.readFile|path\.join|player|ai|hud|health|damage|arrow|bow|aim|"
    r"shoot|update|render|state|score|power|difficulty"
    r")\b"
    r")",
    re.IGNORECASE,
)
_ACTIVE_READ_INSPECTION_REQUIRED_ANCHOR_RES = (
    re.compile(r"<meta\s+name=[\"']viewport[\"']", re.IGNORECASE),
    re.compile(r"<title\b", re.IGNORECASE),
    re.compile(r"three@|cdn\.jsdelivr", re.IGNORECASE),
    re.compile(r"\bimportmap\b", re.IGNORECASE),
    re.compile(r"\bWebGLRenderer\b", re.IGNORECASE),
    re.compile(r"\bsetPixelRatio\b", re.IGNORECASE),
    re.compile(r"\bShaderMaterial\b", re.IGNORECASE),
    re.compile(r"\blocalStorage\b", re.IGNORECASE),
    re.compile(r"\bpipeRemoveZ\b", re.IGNORECASE),
    re.compile(r"\bfunction\s+checkCollision\b", re.IGNORECASE),
    re.compile(r"\bscene\.remove\(pipe\)", re.IGNORECASE),
    re.compile(r"\bpipes\.splice\b", re.IGNORECASE),
    re.compile(r"\bparticles\.forEach\(p\s*=>\s*scene\.remove\(p\)\)", re.IGNORECASE),
    re.compile(r"\bparticles\.length\s*=", re.IGNORECASE),
    re.compile(r"\bscene\.remove\b|\bdispose\b", re.IGNORECASE),
    re.compile(r"\btouchstart\b", re.IGNORECASE),
    re.compile(r"\baddEventListener\s*\(\s*[\"']resize[\"']", re.IGNORECASE),
    re.compile(r"\bcamera\.aspect\b", re.IGNORECASE),
    re.compile(r"\bupdateProjectionMatrix\b", re.IGNORECASE),
    re.compile(r"\brenderer\.setSize\b", re.IGNORECASE),
    re.compile(r"\brequestAnimationFrame\b", re.IGNORECASE),
    re.compile(r"\bgetDelta\b", re.IGNORECASE),
    re.compile(r"\bDate\.now\b", re.IGNORECASE),
    re.compile(r"for\s*\(\s*let\s+\w+\s*=\s*\w+\.length\s*-\s*1", re.IGNORECASE),
    re.compile(r"\bfs\.readFile\b|\bpath\.join\b|\bserver\.listen\b", re.IGNORECASE),
)
_ACTIVE_READ_INSPECTION_CONTEXT_ANCHOR_RES = (
    (re.compile(r"\bfunction\s+checkCollision\b", re.IGNORECASE), 5),
    (re.compile(r"\bkeydown\b", re.IGNORECASE), 3),
    (re.compile(r"\btouchstart\b", re.IGNORECASE), 3),
)
_ACTIVE_READ_ANCHOR_RE = re.compile(
    r"\b("
    r"class|constructor|function|interface|type|enum|export|public|private|protected|"
    r"collid\w*|intersect\w*|distanceTo|raycast\w*|arrow\w*|terrain\w*|"
    r"player\w*|obstacle\w*|hit\w*|damage\w*|health\w*|embed\w*|"
    r"isEmbedded|check\w*|update\w*|handle\w*|getHeight|"
    r"addEventListener|touchstart|keydown|visibilitychange|localStorage|"
    r"requestAnimationFrame|WebGLRenderer|BoxGeometry|MeshStandardMaterial|"
    r"SphereGeometry|PlaneGeometry|ShaderMaterial|BufferGeometry|PointsMaterial|"
    r"Clock|MathUtils|Date\.now|clone\(|setPixelRatio|shadowMap|"
    r"dispose|scene\.remove"
    r")\b",
    re.IGNORECASE,
)
_ACTIVE_TOOL_OUTPUT_IMPORTANT_LINE_RE = re.compile(
    r"(^|/)(src|app|lib|components|pages|packages|Sources|Tests|test|tests)/|"
    r"\b(Line \d+:|class|function|export|error|warning|failed|failure|"
    r"exception|traceback|syntaxerror|typeerror|assertionerror|TS\d+|"
    r"collid\w*|terrain\w*|arrow\w*|obstacle\w*|hit\w*|damage\w*|embed\w*)\b",
    re.IGNORECASE,
)
_ACTIVE_TOOL_OUTPUT_LOW_VALUE_LINE_RE = re.compile(
    r"(^|/)(node_modules|dist|build|vendor|\.git|DerivedData)/|"
    r"\.map\b|package-lock\.json|pnpm-lock\.yaml|yarn\.lock",
    re.IGNORECASE,
)
_SOURCE_PATH_EXT_PATTERN = (
    r"ts|tsx|js|jsx|mjs|cjs|py|swift|html|css|json|yaml|yml|toml|md|"
    r"sh|zsh|rs|go|java|kt|m|mm|h|hpp|cpp|c|cs|vue|svelte|sql"
)
_ACTIVE_TOOL_OUTPUT_PATH_LINE_RES = (
    re.compile(
        rf"(?P<path>(?:/|\.{{1,2}}/|[A-Za-z0-9_.-]+/)[^\n:()<>\"'`]+?"
        rf"\.(?:{_SOURCE_PATH_EXT_PATTERN})):(?P<line>\d+)(?::(?P<col>\d+))?",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<path>(?:/|\.{{1,2}}/|[A-Za-z0-9_.-]+/)[^\n:()<>\"'`]+?"
        rf"\.(?:{_SOURCE_PATH_EXT_PATTERN}))\((?P<line>\d+),(?P<col>\d+)\)",
        re.IGNORECASE,
    ),
    re.compile(
        rf'File "(?P<path>[^"\n]+?\.(?:{_SOURCE_PATH_EXT_PATTERN}))", '
        r"line (?P<line>\d+)",
        re.IGNORECASE,
    ),
)
_ACTIVE_TOOL_OUTPUT_SOURCE_PATH_RE = re.compile(
    rf"(?P<path>(?:/|\.{{1,2}}/|[A-Za-z0-9_.-]+/)[^\n:()<>\"'`]+?"
    rf"\.(?:{_SOURCE_PATH_EXT_PATTERN}))(?=$|[\s:),])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _ActiveToolOutputHint:
    path: str
    start: int | None
    end: int | None
    source_lines: tuple[int, ...]


@dataclass(frozen=True)
class _ReadToolContentMeta:
    path: str
    file_type: str
    line_numbers: frozenset[int]
    first_line: int | None
    last_line: int | None


def _agent_preamble_key(content: str) -> str:
    text = _strip_assistant_history_baggage(content or "")
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = text.strip(" :-\t\r\n")
    return text


def _looks_like_agent_action_preamble(content: str) -> bool:
    key = _agent_preamble_key(content)
    if len(key) < 12 or len(key) > 320:
        return False
    return any(marker in key for marker in _AGENT_PREAMBLE_MARKERS) and any(
        marker in key for marker in _AGENT_ACTION_MARKERS
    )


def _looks_like_repeated_agent_preamble(content: str) -> bool:
    normalized = _agent_preamble_key(content)
    if len(normalized) < 160:
        return False
    loop_markers = (
        "let me continue",
        "write the sky, game, and utils",
        "fix the sky, game, and utils",
        "now let me fix",
        "now fixing all",
        "almost there",
    )
    if any(normalized.count(marker) >= 4 for marker in loop_markers):
        return True

    fragments = [
        fragment.strip(" :-")
        for fragment in re.split(
            r"[\r\n]+|(?=let me continue:)|(?=now let me )|(?=almost there)",
            content,
        )
        if 16 <= len(fragment.strip()) <= 180
    ]
    counts: dict[str, int] = {}
    for fragment in fragments:
        key = _agent_preamble_key(fragment)
        if key:
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return False
    repeated_fragment, repeated_count = max(counts.items(), key=lambda item: item[1])
    repeated_chars = len(repeated_fragment) * repeated_count
    return repeated_count >= 4 and repeated_chars >= max(160, len(normalized) // 3)


def _looks_like_stalled_agent_preamble(content: str) -> bool:
    key = _agent_preamble_key(content)
    if not key:
        return False
    if _looks_like_repeated_agent_preamble(content):
        return True
    if not _looks_like_agent_action_preamble(content):
        return False
    # This catches visible OpenCode stalls such as "Let me fix this:" or
    # "I need to remove the duplicates... Let me fix this:" without removing
    # concrete answers that merely mention a next step.
    if key.endswith(
        (
            ":",
            "let me fix this",
            "let me check this",
            "let me run this",
            "let me fix all remaining errors",
            "fix all remaining errors",
        )
    ):
        return True
    return bool(re.search(r"\b(let me|i(?:'ll| will)|now let me)\b.{0,80}:$", key))


def _looks_like_verbatim_tool_output_assistant_dump(content: str) -> bool:
    text = _strip_assistant_history_baggage(content or "").strip()
    if len(text) < 1_200:
        return False
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 24:
        return False
    sampled = lines[: min(96, len(lines))]
    numbered_count = sum(
        1 for line in sampled if _LINE_NUMBERED_CONTENT_RE.match(line)
    )
    if numbered_count < 20:
        return False
    first_is_numbered = bool(_LINE_NUMBERED_CONTENT_RE.match(sampled[0]))
    return first_is_numbered or numbered_count >= max(20, int(len(sampled) * 0.65))


def _is_read_only_inspection_request(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not normalized:
        return False
    recommendation_request = bool(
        _READ_ONLY_RECOMMENDATION_REQUEST_RE.search(normalized)
    )
    if not _READ_ONLY_INSPECTION_REQUEST_RE.search(normalized) and not recommendation_request:
        return False
    if _MUTATING_REQUEST_RE.search(normalized):
        return recommendation_request or bool(
            re.search(
                r"\b(no edits?|without editing|read-only)\b"
                r"|\bdo not (?:edit|modify|write|change)\b"
                r"|\bdon't (?:edit|modify|write|change)\b",
                normalized,
            )
        )
    return True


def _tool_call_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function")
    raw_args: Any
    if isinstance(function, dict):
        raw_args = function.get("arguments", {})
    else:
        raw_args = tool_call.get("arguments", {})
    if isinstance(raw_args, dict):
        return dict(raw_args)
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except Exception:
            return {"_raw": raw_args}
        if isinstance(parsed, dict):
            return dict(parsed)
        return {"_raw": raw_args}
    return {}


def _tool_call_name(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or "").strip()
    return str(tool_call.get("name") or "").strip()


def _tool_call_loop_key(tool_call: dict[str, Any]) -> tuple[str, str, str] | None:
    name = _tool_call_name(tool_call)
    if not name:
        return None
    args = _tool_call_arguments(tool_call)
    command = str(
        args.get("command")
        or args.get("cmd")
        or args.get("script")
        or args.get("pattern")
        or args.get("filePath")
        or args.get("path")
        or ""
    ).strip()
    if command:
        key_payload = command
    else:
        try:
            key_payload = json.dumps(args, sort_keys=True, separators=(",", ":"))
        except TypeError:
            key_payload = str(args)
    if not key_payload:
        return None
    return name, key_payload, command or key_payload


def _tool_result_is_timeout(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if "timeout" not in lowered and "timed out" not in lowered:
        return False
    return (
        "terminated command after exceeding timeout" in lowered
        or "command timed out" in lowered
        or ("<shell_metadata>" in lowered and "timeout" in lowered)
    )


def _compact_repeated_timeout_tool_result_text(
    text: str,
    *,
    tool_name: str,
    command: str,
    repeat_count: int,
) -> str:
    command_excerpt = html.escape(_compact_tool_excerpt_line(command, 480))
    output_excerpt = html.escape(_compact_tool_excerpt_line(text.strip(), 900))
    return (
        "<mtplx_repeated_timeout_tool_output "
        f'tool="{html.escape(tool_name)}" repeat_count="{repeat_count}" '
        f'original_chars="{len(text)}">\n'
        "[This exact tool call has timed out repeatedly in this session. "
        "Do not call the same command again unchanged. Change strategy: inspect "
        "the relevant files/scripts, run a narrower command, increase timeout "
        "only if that is clearly justified, or explain the blocker concretely.]\n"
        f"<command>{command_excerpt}</command>\n"
        f"<latest_output_excerpt>{output_excerpt}</latest_output_excerpt>\n"
        "</mtplx_repeated_timeout_tool_output>"
    )


def _latest_plain_assistant_answer_index(messages: list[ChatMessage]) -> int | None:
    latest: int | None = None
    for index, message in enumerate(messages):
        if str(message.role).lower() != "assistant" or message.tool_calls:
            continue
        if _content_to_text(message.content).strip():
            latest = index
    return latest


def _latest_assistant_step_index(messages: list[ChatMessage]) -> int | None:
    latest: int | None = None
    for index, message in enumerate(messages):
        if str(message.role).lower() == "assistant":
            latest = index
    return latest


def _compact_tool_result_text(text: str) -> str | None:
    threshold = max(
        1,
        _env_int(
            "MTPLX_TOOL_RESULT_COMPACT_THRESHOLD_CHARS",
            _TOOL_RESULT_COMPACT_THRESHOLD_CHARS,
        ),
    )
    if len(text) <= threshold:
        return None
    head_chars = max(
        0,
        _env_int("MTPLX_TOOL_RESULT_COMPACT_HEAD_CHARS", _TOOL_RESULT_COMPACT_HEAD_CHARS),
    )
    tail_chars = max(
        0,
        _env_int("MTPLX_TOOL_RESULT_COMPACT_TAIL_CHARS", _TOOL_RESULT_COMPACT_TAIL_CHARS),
    )
    head = text[:head_chars].rstrip() if head_chars else ""
    tail = text[-tail_chars:].lstrip() if tail_chars else ""
    omitted = max(0, len(text) - len(head) - len(tail))
    return (
        "<mtplx_compacted_tool_output "
        f"original_chars={len(text)} omitted_chars={omitted}>\n"
        "[Older tool result abbreviated after a later assistant step already "
        "digested it. "
        "Call the tool again with a narrower range if exact omitted text is needed.]\n"
        f"{head}\n\n"
        f"... [MTPLX compacted {omitted} older-tool-output characters.] ...\n\n"
        f"{tail}\n"
        "</mtplx_compacted_tool_output>"
    )


def _tool_output_anchor(kind: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in (kind, *parts))
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:16]


def _line_range_label(start: int | None, end: int | None) -> str:
    if start is None or end is None:
        return "unknown"
    return f"{start}-{end}" if start != end else str(start)


def _omitted_line_ranges(
    kept_lines: list[int],
    *,
    first_line: int,
    last_line: int,
) -> list[tuple[int, int]]:
    if not kept_lines or first_line > last_line:
        return []
    ranges: list[tuple[int, int]] = []
    cursor = first_line
    for line_no in sorted(set(kept_lines)):
        if line_no < first_line or line_no > last_line:
            continue
        if line_no > cursor:
            ranges.append((cursor, line_no - 1))
        cursor = max(cursor, line_no + 1)
    if cursor <= last_line:
        ranges.append((cursor, last_line))
    return ranges


def _render_next_read_hints(
    *,
    path: str,
    ranges: list[tuple[int, int]],
    max_hints: int = 4,
) -> str:
    if not ranges:
        return ""
    safe_path = html.escape(path)
    lines = ["<next_read_hints>"]
    for start, end in ranges[:max(1, max_hints)]:
        limit = max(1, end - start + 1)
        lines.append(
            f'<range start="{start}" end="{end}" limit="{limit}">'
            f'If exact omitted text matters, read filePath="{safe_path}" '
            f"offset={start} limit={limit}; otherwise synthesize from the "
            "visible anchors now."
            "</range>"
        )
    if len(ranges) > max_hints:
        lines.append(
            f'<more omitted_range_count="{len(ranges) - max_hints}">'
            "Prefer named symbols or one exact range over rereading the whole file."
            "</more>"
        )
    lines.append("</next_read_hints>")
    return "\n".join(lines)


def _compact_tool_excerpt_line(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    tail_chars = min(96, max(32, max_chars // 4))
    head_chars = max(32, max_chars - tail_chars - 48)
    omitted = max(0, len(text) - head_chars - tail_chars)
    return (
        text[:head_chars].rstrip()
        + f" ... [MTPLX truncated {omitted} chars] ... "
        + text[-tail_chars:].lstrip()
    )


def _clean_active_tool_output_path(raw_path: str) -> str | None:
    path = html.unescape(str(raw_path or "")).strip()
    path = path.strip(" \t\r\n\"'`")
    path = re.sub(r"[\),;]+$", "", path).strip()
    if not path or len(path) > 320:
        return None
    if _ACTIVE_TOOL_OUTPUT_LOW_VALUE_LINE_RE.search(path):
        return None
    return path


def _compressed_int_ranges(values: Iterable[int], *, max_ranges: int = 4) -> str:
    ordered = sorted({int(value) for value in values if int(value) > 0})
    if not ordered:
        return ""
    ranges: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append((start, previous))
        start = previous = value
    ranges.append((start, previous))
    parts = [
        str(start) if start == end else f"{start}-{end}"
        for start, end in ranges[: max(1, max_ranges)]
    ]
    if len(ranges) > max_ranges:
        parts.append(f"+{len(ranges) - max_ranges}")
    return ",".join(parts)


def _cluster_source_lines(lines: Iterable[int], *, max_gap: int = 80) -> list[list[int]]:
    clusters: list[list[int]] = []
    for line_no in sorted({int(line) for line in lines if int(line) > 0}):
        if not clusters or line_no > clusters[-1][-1] + max_gap:
            clusters.append([line_no])
        else:
            clusters[-1].append(line_no)
    return clusters


def _active_tool_output_read_hints(lines: list[str]) -> list[_ActiveToolOutputHint]:
    path_lines: dict[str, set[int]] = {}
    output_lines_by_path: dict[str, set[int]] = {}
    plain_paths: dict[str, set[int]] = {}

    for output_line_no, line in enumerate(lines, start=1):
        if _ACTIVE_TOOL_OUTPUT_LOW_VALUE_LINE_RE.search(line):
            continue
        for path_line_re in _ACTIVE_TOOL_OUTPUT_PATH_LINE_RES:
            for match in path_line_re.finditer(line):
                path = _clean_active_tool_output_path(match.group("path"))
                if path is None:
                    continue
                try:
                    source_line = int(match.group("line"))
                except (TypeError, ValueError):
                    continue
                if source_line <= 0:
                    continue
                path_lines.setdefault(path, set()).add(source_line)
                output_lines_by_path.setdefault(path, set()).add(output_line_no)
        for match in _ACTIVE_TOOL_OUTPUT_SOURCE_PATH_RE.finditer(line):
            path = _clean_active_tool_output_path(match.group("path"))
            if path is None:
                continue
            plain_paths.setdefault(path, set()).add(output_line_no)

    hints: list[_ActiveToolOutputHint] = []
    for path, source_lines in sorted(
        path_lines.items(),
        key=lambda item: (min(item[1]), item[0]),
    ):
        for cluster in _cluster_source_lines(source_lines)[:2]:
            first = min(cluster)
            last = max(cluster)
            start = max(1, first - 20)
            end = min(max(last + 20, first + 20), start + 119)
            hints.append(
                _ActiveToolOutputHint(
                    path=path,
                    start=start,
                    end=end,
                    source_lines=tuple(sorted(output_lines_by_path.get(path, set()))),
                )
            )
            if len(hints) >= 6:
                return hints

    hinted_paths = {hint.path for hint in hints}
    for path, output_lines in sorted(
        plain_paths.items(),
        key=lambda item: (min(item[1]), item[0]),
    ):
        if path in hinted_paths:
            continue
        hints.append(
            _ActiveToolOutputHint(
                path=path,
                start=None,
                end=None,
                source_lines=tuple(sorted(output_lines)),
            )
        )
        if len(hints) >= 6:
            return hints
    return hints


def _render_active_tool_output_read_hints(
    hints: list[_ActiveToolOutputHint],
) -> str:
    if not hints:
        return ""
    rendered = ["<next_read_hints>"]
    for hint in hints:
        safe_path = html.escape(hint.path, quote=True)
        source_output_lines = html.escape(_compressed_int_ranges(hint.source_lines))
        if hint.start is not None and hint.end is not None:
            limit = max(1, hint.end - hint.start + 1)
            rendered.append(
                f'<range filePath="{safe_path}" start="{hint.start}" '
                f'end="{hint.end}" limit="{limit}" '
                f'source_output_lines="{source_output_lines}">'
                "If exact context matters, read this range next. Do not rerun "
                "the broad tool command unchanged."
                "</range>"
            )
        else:
            rendered.append(
                f'<path filePath="{safe_path}" source_output_lines="{source_output_lines}">'
                "Read this file only if it is the next relevant candidate. Avoid "
                "broad list/glob/grep repeats."
                "</path>"
            )
    rendered.append("</next_read_hints>")
    return "\n".join(rendered)


def _active_tool_result_read_hint_count(compacted: str) -> int:
    match = re.search(r"\bread_hint_count=(\d+)", compacted)
    if not match:
        return 0
    try:
        return max(0, int(match.group(1)))
    except ValueError:
        return 0


def _compact_active_tool_result_text(text: str) -> str | None:
    """Compact large current grep/glob/bash outputs without hiding useful paths."""

    threshold = max(
        1,
        _env_int(
            "MTPLX_ACTIVE_TOOL_RESULT_COMPACT_THRESHOLD_CHARS",
            _ACTIVE_TOOL_RESULT_COMPACT_THRESHOLD_CHARS,
        ),
    )
    if len(text) <= threshold:
        return None
    lines = text.splitlines()
    if not lines:
        return None
    read_hints = _active_tool_output_read_hints(lines)

    head_lines = max(
        0,
        _env_int(
            "MTPLX_ACTIVE_TOOL_RESULT_COMPACT_HEAD_LINES",
            _ACTIVE_TOOL_RESULT_COMPACT_HEAD_LINES,
        ),
    )
    tail_lines = max(
        0,
        _env_int(
            "MTPLX_ACTIVE_TOOL_RESULT_COMPACT_TAIL_LINES",
            _ACTIVE_TOOL_RESULT_COMPACT_TAIL_LINES,
        ),
    )
    max_lines = max(
        8,
        _env_int(
            "MTPLX_ACTIVE_TOOL_RESULT_COMPACT_MAX_LINES",
            _ACTIVE_TOOL_RESULT_COMPACT_MAX_LINES,
        ),
    )
    line_max_chars = max(
        120,
        _env_int(
            "MTPLX_ACTIVE_TOOL_RESULT_LINE_MAX_CHARS",
            _ACTIVE_TOOL_RESULT_LINE_MAX_CHARS,
        ),
    )

    selected: set[int] = set()

    def add_index(index: int) -> None:
        if len(selected) >= max_lines:
            return
        if 0 <= index < len(lines):
            selected.add(index)

    for index in range(min(head_lines, len(lines))):
        if not _ACTIVE_TOOL_OUTPUT_LOW_VALUE_LINE_RE.search(lines[index]):
            add_index(index)
    for index in range(max(0, len(lines) - tail_lines), len(lines)):
        if not _ACTIVE_TOOL_OUTPUT_LOW_VALUE_LINE_RE.search(lines[index]):
            add_index(index)

    for index, line in enumerate(lines):
        if len(selected) >= max_lines:
            break
        if _ACTIVE_TOOL_OUTPUT_LOW_VALUE_LINE_RE.search(line):
            continue
        if _ACTIVE_TOOL_OUTPUT_IMPORTANT_LINE_RE.search(line):
            add_index(index)

    for index, line in enumerate(lines):
        if len(selected) >= max_lines:
            break
        if _ACTIVE_TOOL_OUTPUT_IMPORTANT_LINE_RE.search(line):
            if _ACTIVE_TOOL_OUTPUT_LOW_VALUE_LINE_RE.search(line):
                continue
            add_index(index)

    kept = sorted(selected)
    if not kept:
        return None

    excerpt: list[str] = []
    previous: int | None = None
    for index in kept:
        if previous is not None and index > previous + 1:
            excerpt.append(
                f"... [MTPLX omitted output lines {previous + 2}-{index}] ..."
            )
        excerpt.append(_compact_tool_excerpt_line(lines[index], line_max_chars))
        previous = index
    if previous is not None and previous < len(lines) - 1:
        excerpt.append(
            f"... [MTPLX omitted output lines {previous + 2}-{len(lines)}] ..."
        )

    body = "\n".join(excerpt).rstrip()
    omitted_lines = max(0, len(lines) - len(kept))
    read_hint_text = _render_active_tool_output_read_hints(read_hints)
    source_path_count = len({hint.path for hint in read_hints})
    anchor = _tool_output_anchor(
        "active_tool",
        len(text),
        len(lines),
        "\n".join(lines[index] for index in kept[:12]),
    )
    compacted = (
        "<mtplx_compacted_active_tool_output "
        f"original_chars={len(text)} original_lines={len(lines)} "
        f"kept_lines={len(kept)} omitted_lines={omitted_lines} "
        f"read_hint_count={len(read_hints)} source_path_count={source_path_count} "
        f'anchor="{anchor}">\n'
        "[Large current tool output abbreviated to keep OpenCode responsive. "
        "Important source paths, errors, and match lines are prioritized. Use "
        "the next_read_hints for exact follow-up reads; do not rerun broad "
        "list/grep/build commands unchanged.]\n"
        f"{body}\n"
        f"{read_hint_text + chr(10) if read_hint_text else ''}"
        "</mtplx_compacted_active_tool_output>"
    )
    return compacted if len(compacted) < len(text) else None


def _extract_tag_text(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def _tag_plain_read_tool_result_text(text: str, *, path: str) -> str | None:
    if not text or "<content>" in text or "</content>" in text:
        return None
    clean_path = (path or "").strip()
    if not clean_path:
        return None
    lines = text.splitlines()
    if len(lines) < 20:
        return None
    numbered: list[str] = []
    already_numbered = 0
    for index, line in enumerate(lines, start=1):
        if _LINE_NUMBERED_CONTENT_RE.match(line):
            already_numbered += 1
            numbered.append(line)
        else:
            numbered.append(f"{index}: {line}")
    if already_numbered >= max(20, int(len(lines) * 0.65)):
        content = "\n".join(lines)
    else:
        content = "\n".join(numbered)
    return (
        f"<path>{html.escape(clean_path)}</path>\n"
        "<type>file</type>\n"
        "<content>\n"
        f"{content}\n"
        "</content>"
    )


def _read_tool_content_meta(text: str) -> _ReadToolContentMeta | None:
    if "<content>" not in text or "</content>" not in text:
        return None
    content = _extract_tag_text(text, "content")
    if not content:
        return None
    line_numbers: set[int] = set()
    for raw_line in content.splitlines():
        match = _LINE_NUMBERED_CONTENT_RE.match(raw_line)
        if match:
            line_numbers.add(int(match.group(1)))
    if not line_numbers:
        return None
    return _ReadToolContentMeta(
        path=_extract_tag_text(text, "path") or "(unknown path)",
        file_type=_extract_tag_text(text, "type") or "file",
        line_numbers=frozenset(line_numbers),
        first_line=min(line_numbers),
        last_line=max(line_numbers),
    )


def _inspection_read_budget_for_count(candidate_count: int) -> tuple[int | None, int | None]:
    if candidate_count <= 1:
        return None, None
    total_lines = max(
        0,
        _env_int(
            "MTPLX_ACTIVE_READ_INSPECTION_TOTAL_MAX_LINES",
            _ACTIVE_READ_INSPECTION_TOTAL_MAX_LINES,
        ),
    )
    if total_lines <= 0:
        return None, None
    min_lines = max(
        8,
        _env_int(
            "MTPLX_ACTIVE_READ_INSPECTION_MIN_LINES_PER_FILE",
            _ACTIVE_READ_INSPECTION_MIN_LINES_PER_FILE,
        ),
    )
    max_lines = max(min_lines, total_lines // max(1, candidate_count))
    line_max_chars = max(
        120,
        _env_int(
            "MTPLX_ACTIVE_READ_INSPECTION_MULTI_FILE_LINE_MAX_CHARS",
            _ACTIVE_READ_INSPECTION_MULTI_FILE_LINE_MAX_CHARS,
        ),
    )
    return max_lines, line_max_chars


def _compact_repeated_inspection_read_tool_result_text(
    text: str,
    *,
    meta: _ReadToolContentMeta,
    prior_covered_lines: int,
    new_lines: int,
) -> str:
    path = html.escape(meta.path)
    file_type = html.escape(meta.file_type)
    line_range = (
        f"{meta.first_line}-{meta.last_line}"
        if meta.first_line is not None and meta.last_line is not None
        else "unknown"
    )
    anchor = _tool_output_anchor(
        "repeated_read",
        meta.path,
        line_range,
        len(text),
        prior_covered_lines,
        new_lines,
    )
    return (
        "<mtplx_repeated_read_inspection_digest "
        f'original_chars="{len(text)}" '
        f'anchor="{anchor}" '
        f'line_range="{html.escape(line_range)}" '
        f'covered_lines="{len(meta.line_numbers)}" '
        f'prior_covered_lines="{prior_covered_lines}" '
        f'new_lines="{new_lines}">\n'
        "[This read repeats source lines already covered by an earlier digest "
        "for the same read-only review/evaluation. Use the earlier digest and "
        "the other tool outputs to synthesize now. Do not call read again for "
        "this same path or adjacent ranges for completeness; only request a "
        "new named symbol or exact line range if it would materially change "
        "the answer.]\n"
        f"<path>{path}</path>\n"
        f"<type>{file_type}</type>\n"
        "</mtplx_repeated_read_inspection_digest>"
    )


def _compact_active_read_tool_result_text(
    text: str,
    *,
    inspection_request: bool = False,
    inspection_max_lines: int | None = None,
    inspection_line_max_chars: int | None = None,
) -> str | None:
    """Compact oversized current OpenCode read outputs without hiding anchors.

    Old tool results can be reduced aggressively after a plain answer. During an
    active multi-tool loop, though, the next model call still needs enough of a
    freshly read file to decide whether to answer or request narrower ranges.
    This excerpt keeps line numbers plus semantic anchors and tells the model to
    rerun `read` narrowly for omitted exact text.
    """

    force_compact = bool(_READ_CONTINUATION_HINT_RE.search(text))
    threshold = (
        max(
            1,
            _env_int(
                "MTPLX_ACTIVE_READ_INSPECTION_COMPACT_THRESHOLD_CHARS",
                _ACTIVE_READ_INSPECTION_COMPACT_THRESHOLD_CHARS,
            ),
        )
        if inspection_request
        else _ACTIVE_READ_COMPACT_THRESHOLD_CHARS
    )
    if len(text) <= threshold and not force_compact:
        return None
    if "<content>" not in text or "</content>" not in text:
        return None
    path = _extract_tag_text(text, "path") or "(unknown path)"
    file_type = _extract_tag_text(text, "type") or "file"
    content = _extract_tag_text(text, "content")
    if not content:
        return None

    numbered: list[tuple[int, str]] = []
    for raw_line in content.splitlines():
        match = _LINE_NUMBERED_CONTENT_RE.match(raw_line)
        if match:
            numbered.append((int(match.group(1)), match.group(2)))
    selected: set[int] = set()
    if inspection_request:
        head_lines = max(
            0,
            _env_int(
                "MTPLX_ACTIVE_READ_INSPECTION_COMPACT_HEAD_LINES",
                _ACTIVE_READ_INSPECTION_COMPACT_HEAD_LINES,
            ),
        )
        tail_lines = max(
            0,
            _env_int(
                "MTPLX_ACTIVE_READ_INSPECTION_COMPACT_TAIL_LINES",
                _ACTIVE_READ_INSPECTION_COMPACT_TAIL_LINES,
            ),
        )
        max_lines = max(
            8,
            _env_int(
                "MTPLX_ACTIVE_READ_INSPECTION_COMPACT_MAX_LINES",
                _ACTIVE_READ_INSPECTION_COMPACT_MAX_LINES,
            ),
        )
        if inspection_max_lines is not None:
            max_lines = min(max_lines, max(8, int(inspection_max_lines)))
        context_lines = max(
            0,
            _env_int(
                "MTPLX_ACTIVE_READ_INSPECTION_COMPACT_CONTEXT_LINES",
                _ACTIVE_READ_INSPECTION_COMPACT_CONTEXT_LINES,
            ),
        )
        line_max_chars = max(
            120,
            _env_int(
                "MTPLX_ACTIVE_READ_INSPECTION_LINE_MAX_CHARS",
                _ACTIVE_READ_INSPECTION_LINE_MAX_CHARS,
            ),
        )
        if inspection_line_max_chars is not None:
            line_max_chars = min(line_max_chars, max(120, int(inspection_line_max_chars)))
    else:
        head_lines = max(
            0,
            _env_int(
                "MTPLX_ACTIVE_READ_COMPACT_HEAD_LINES",
                _ACTIVE_READ_COMPACT_HEAD_LINES,
            ),
        )
        tail_lines = max(
            0,
            _env_int(
                "MTPLX_ACTIVE_READ_COMPACT_TAIL_LINES",
                _ACTIVE_READ_COMPACT_TAIL_LINES,
            ),
        )
        max_lines = max(
            8,
            _env_int(
                "MTPLX_ACTIVE_READ_COMPACT_MAX_LINES",
                _ACTIVE_READ_COMPACT_MAX_LINES,
            ),
        )
        context_lines = max(
            0,
            _env_int(
                "MTPLX_ACTIVE_READ_COMPACT_CONTEXT_LINES",
                _ACTIVE_READ_COMPACT_CONTEXT_LINES,
            ),
        )
        line_max_chars = 360

    def add_window(line_no: int, radius: int = 0) -> None:
        for candidate in range(line_no - radius, line_no + radius + 1):
            if len(selected) >= max_lines:
                return
            selected.add(candidate)

    for line_no, _line in numbered[:head_lines]:
        add_window(line_no)
    if tail_lines > 0:
        for line_no, _line in numbered[-tail_lines:]:
            add_window(line_no)

    def add_anchor_candidates(
        candidates: list[int],
        *,
        radius: int = 0,
        spread: bool = False,
    ) -> None:
        remaining = max_lines - len(selected)
        if remaining <= 0:
            return
        unique: list[int] = []
        seen: set[int] = set()
        for line_no in candidates:
            if line_no in seen:
                continue
            seen.add(line_no)
            if radius == 0 and line_no in selected:
                continue
            unique.append(line_no)
        if not unique:
            return
        if spread and len(unique) > remaining:
            if remaining == 1:
                unique = [unique[-1]]
            else:
                step = (len(unique) - 1) / (remaining - 1)
                chosen: list[int] = []
                for idx in range(remaining):
                    line_no = unique[round(idx * step)]
                    if line_no not in chosen:
                        chosen.append(line_no)
                unique = chosen
        for line_no in unique:
            if len(selected) >= max_lines:
                return
            before = set(selected)
            add_window(line_no, radius)
            if len(selected) > max_lines:
                selected.clear()
                selected.update(before)
                return

    if inspection_request:
        required_anchor_lines: list[int] = []
        for anchor_re in _ACTIVE_READ_INSPECTION_REQUIRED_ANCHOR_RES:
            for line_no, line in numbered:
                if anchor_re.search(line):
                    required_anchor_lines.append(line_no)
                    break
        add_anchor_candidates(required_anchor_lines)
        for anchor_re, radius in _ACTIVE_READ_INSPECTION_CONTEXT_ANCHOR_RES:
            for line_no, line in numbered:
                if anchor_re.search(line):
                    add_anchor_candidates([line_no], radius=radius)
                    break

    priority_anchor_re = (
        _ACTIVE_READ_INSPECTION_PRIORITY_ANCHOR_RE
        if inspection_request
        else _ACTIVE_READ_PRIORITY_ANCHOR_RE
    )
    priority_anchor_lines = [
        line_no
        for line_no, line in numbered
        if priority_anchor_re.search(line)
    ]
    generic_anchor_lines = [
        line_no for line_no, line in numbered if _ACTIVE_READ_ANCHOR_RE.search(line)
    ]
    add_anchor_candidates(
        priority_anchor_lines,
        radius=context_lines,
        spread=inspection_request,
    )
    add_anchor_candidates(generic_anchor_lines, spread=inspection_request)

    line_by_no = {line_no: line for line_no, line in numbered}
    kept = [line_no for line_no in sorted(selected) if line_no in line_by_no]
    if not kept:
        return None

    excerpt: list[str] = []
    previous: int | None = None
    for line_no in kept:
        if previous is not None and line_no > previous + 1:
            excerpt.append(f"... [MTPLX omitted lines {previous + 1}-{line_no - 1}] ...")
        line = _compact_tool_excerpt_line(line_by_no[line_no], line_max_chars)
        excerpt.append(f"{line_no}: {line}")
        previous = line_no
    if previous is not None and previous < numbered[-1][0]:
        excerpt.append(f"... [MTPLX omitted lines {previous + 1}-{numbered[-1][0]}] ...")

    omitted_lines = max(0, len(numbered) - len(kept))
    if inspection_request:
        evidence = "\n".join(
            f"line {line_no}: {_compact_tool_excerpt_line(line_by_no[line_no], line_max_chars)}"
            for line_no in kept
        ).rstrip()
        first_line = numbered[0][0]
        last_line = numbered[-1][0]
        anchor = _tool_output_anchor(
            "inspection_read",
            path,
            first_line,
            last_line,
            len(text),
            "|".join(str(line_no) for line_no in kept[:24]),
        )
        compacted = (
            "<mtplx_read_inspection_digest "
            f'original_chars="{len(text)}" '
            f'original_lines="{len(numbered)}" '
            f'evidence_lines="{len(kept)}" '
            f'anchor="{anchor}" '
            f'line_range="{html.escape(_line_range_label(first_line, last_line))}" '
            f'continuation_hint_removed="{str(force_compact).lower()}" '
            'full_file_expansion_needed="false">\n'
            "[MTPLX already received this source from the read tool; this is "
            "the intended read-only review/evaluation evidence digest. Treat "
            "it as sufficient for synthesis with the other already-read files. "
            "do not request adjacent ranges or the full file for completeness. "
            "do not copy the numbered evidence lines into reasoning or final "
            "content.]\n"
            f"<path>{path}</path>\n"
            f"<type>{file_type}</type>\n"
            "<evidence>\n"
            f"{evidence}\n"
            "</evidence>\n"
            "</mtplx_read_inspection_digest>"
        )
        return compacted if force_compact or len(compacted) < len(text) else None

    first_line = numbered[0][0]
    last_line = numbered[-1][0]
    omitted_ranges = _omitted_line_ranges(
        kept,
        first_line=first_line,
        last_line=last_line,
    )
    next_read_hints = _render_next_read_hints(path=path, ranges=omitted_ranges)
    anchor = _tool_output_anchor(
        "active_read",
        path,
        first_line,
        last_line,
        len(text),
        "|".join(str(line_no) for line_no in kept[:24]),
    )
    guidance = (
        "[Large current read abbreviated to keep OpenCode tool loops responsive. "
        "The excerpt preserves file line numbers, definitions, and likely "
        "collision/navigation/runtime anchors. For review or evaluation, answer "
        "from this excerpt when the relevant anchors are visible; do not "
        "reconstruct the full file from adjacent reads. Another read is only "
        "for a named missing symbol or exact line range that would change the "
        "answer.]"
    )
    compacted = (
        "<mtplx_compacted_active_read_output "
        f"original_chars={len(text)} original_lines={len(numbered)} "
        f"kept_lines={len(kept)} omitted_lines={omitted_lines} "
        f'anchor="{anchor}" '
        f'line_range="{html.escape(_line_range_label(first_line, last_line))}" '
        f'next_read_hint_count="{min(len(omitted_ranges), 4)}" '
        f'continuation_hint_removed="{str(force_compact).lower()}">\n'
        f"{guidance}\n"
        f"<path>{path}</path>\n"
        f"<type>{file_type}</type>\n"
        "<content_excerpt>\n"
        + "\n".join(excerpt).rstrip()
        + "\n</content_excerpt>\n"
        + (next_read_hints + "\n" if next_read_hints else "")
        + "</mtplx_compacted_active_read_output>"
    )
    return compacted if force_compact or len(compacted) < len(text) else None


def _assistant_reasoning_history_stats(
    message: ChatMessage,
) -> tuple[int, int, int]:
    if str(message.role).lower() != "assistant":
        return 0, 0, 0
    chars = 0
    structured_blocks = 0
    for key in ("reasoning_content", "reasoning", "thinking"):
        value = _message_extra(message, key)
        if value:
            chars += len(str(value))
            structured_blocks += 1
    content = message.content
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "") != "thinking":
                continue
            thinking = str(item.get("thinking") or "")
            if thinking:
                chars += len(thinking)
                structured_blocks += 1
    elif isinstance(content, str):
        chars += sum(len(match.group(0)) for match in _REASONING_TAG_RE.finditer(content))
    return (1 if chars > 0 else 0), chars, structured_blocks


def _canonicalize_agent_transcript(
    messages: list[ChatMessage],
    *,
    tools_active: bool,
    replace_simple_chitchat_system_prompt: bool = False,
    initial_client_system_prompt: str | None = None,
    strip_tool_call_preamble_text: bool = False,
) -> tuple[list[ChatMessage], AgentTranscriptCanonicalization]:
    stats = AgentTranscriptCanonicalization(
        raw_message_chars=sum(
            len(_content_to_text(message.content)) for message in messages
        )
    )
    canonical = list(messages)
    for message in messages:
        reasoning_messages, reasoning_chars, structured_blocks = (
            _assistant_reasoning_history_stats(message)
        )
        stats.assistant_reasoning_history_messages += reasoning_messages
        stats.assistant_reasoning_history_chars += reasoning_chars
        stats.assistant_structured_thinking_blocks += structured_blocks
    if initial_client_system_prompt:
        canonical = _replace_client_system_prompt(
            canonical,
            stats,
            replacement=initial_client_system_prompt,
        )
    if not tools_active:
        canonical = _canonicalize_simple_chitchat_history(
            canonical,
            stats,
            replace_system_prompt=replace_simple_chitchat_system_prompt,
        )
    if tools_active:
        source_messages = canonical
        # OpenCode can feed back very large grep/read results. Old raw outputs
        # punish every follow-up, while huge current full-file reads punish
        # multi-tool exploration loops before a final answer exists. Keep
        # non-read current outputs verbatim, but turn oversized line-numbered
        # reads into anchor-rich excerpts so the model can request narrower
        # ranges instead of carrying whole files forever.
        plain_answer_cutoff = _latest_plain_assistant_answer_index(source_messages)
        latest_assistant_cutoff = _latest_assistant_step_index(source_messages)
        inspection_request = _is_read_only_inspection_request(
            _last_user_text(source_messages)
        )
        tool_calls_by_id: dict[str, tuple[str, str, str]] = {}
        for message in source_messages:
            if str(message.role).lower() != "assistant" or not message.tool_calls:
                continue
            for tool_call in message.tool_calls:
                call_id = str(tool_call.get("id") or "").strip()
                signature = _tool_call_loop_key(tool_call)
                if call_id and signature is not None:
                    tool_calls_by_id[call_id] = signature
        inspection_read_candidate_count = 0
        if inspection_request:
            for message in source_messages:
                if str(message.role).lower() != "tool":
                    continue
                text = _content_to_text(message.content)
                tool_call_id = str(
                    message.tool_call_id
                    or _message_extra(message, "tool_call_id")
                    or ""
                ).strip()
                signature = tool_calls_by_id.get(tool_call_id)
                read_text = text
                if signature is not None:
                    tool_name, _key_payload, command = signature
                    if tool_name.strip().lower() == "read":
                        tagged = _tag_plain_read_tool_result_text(text, path=command)
                        if tagged is not None:
                            read_text = tagged
                read_meta = _read_tool_content_meta(read_text)
                if read_meta is not None and len(read_meta.line_numbers) >= 20:
                    inspection_read_candidate_count += 1
        inspection_max_lines, inspection_line_max_chars = (
            _inspection_read_budget_for_count(inspection_read_candidate_count)
        )
        if inspection_max_lines is not None:
            stats.inspection_read_budget_candidate_messages = (
                inspection_read_candidate_count
            )
            stats.inspection_read_budget_max_lines_per_file = inspection_max_lines
        timeout_counts_by_key: dict[str, int] = {}
        inspection_read_lines_by_path: dict[str, set[int]] = {}
        canonical = []
        for index, message in enumerate(source_messages):
            role = str(message.role).lower()
            if role == "assistant":
                content = _content_to_text(message.content)
                if _message_declares_aborted_assistant_turn(message):
                    stats.skipped_aborted_assistant_messages += 1
                    continue
                if _looks_like_orphan_chitchat_assistant_turn(source_messages, index):
                    stats.skipped_orphan_chitchat_assistant_messages += 1
                    continue
                if (
                    message.tool_calls
                    and content.strip()
                    and (inspection_request or strip_tool_call_preamble_text)
                ):
                    stats.stripped_tool_preamble_messages += 1
                    stats.stripped_tool_preamble_chars += len(content)
                    message = _copy_chat_message(message, content="")
                    content = ""
                if not message.tool_calls:
                    if _looks_like_verbatim_tool_output_assistant_dump(content):
                        stats.skipped_verbatim_tool_output_assistant_messages += 1
                        stats.skipped_verbatim_tool_output_assistant_chars += len(content)
                        continue
                    if _looks_like_repeated_agent_preamble(content):
                        stats.skipped_repeated_assistant_messages += 1
                        continue
                    if _looks_like_stalled_agent_preamble(content):
                        stats.skipped_stalled_agent_preamble_messages += 1
                        stats.skipped_stalled_agent_preamble_chars += len(content)
                        continue
            if role == "tool":
                text = _content_to_text(message.content)
                tool_call_id = str(
                    message.tool_call_id
                    or _message_extra(message, "tool_call_id")
                    or ""
                ).strip()
                signature = tool_calls_by_id.get(tool_call_id)
                if signature is not None and _tool_result_is_timeout(text):
                    tool_name, key_payload, command = signature
                    timeout_key = f"{tool_name}\n{key_payload}"
                    repeat_count = timeout_counts_by_key.get(timeout_key, 0) + 1
                    timeout_counts_by_key[timeout_key] = repeat_count
                    if repeat_count >= 2:
                        canonical.append(
                            _copy_chat_message(
                                message,
                                content=_compact_repeated_timeout_tool_result_text(
                                    text,
                                    tool_name=tool_name,
                                    command=command,
                                    repeat_count=repeat_count,
                                ),
                            )
                        )
                        stats.compacted_repeated_timeout_tool_messages += 1
                        continue
                if inspection_request:
                    read_text = text
                    if signature is not None:
                        tool_name, _key_payload, command = signature
                        if tool_name.strip().lower() == "read":
                            tagged = _tag_plain_read_tool_result_text(
                                text,
                                path=command,
                            )
                            if tagged is not None:
                                read_text = tagged
                    read_meta = _read_tool_content_meta(read_text)
                    if read_meta is not None:
                        prior_lines = inspection_read_lines_by_path.setdefault(
                            read_meta.path,
                            set(),
                        )
                        new_lines = set(read_meta.line_numbers) - prior_lines
                        duplicate_threshold = max(
                            3,
                            int(len(read_meta.line_numbers) * 0.08),
                        )
                        if prior_lines and len(new_lines) <= duplicate_threshold:
                            compacted = _compact_repeated_inspection_read_tool_result_text(
                                read_text,
                                meta=read_meta,
                                prior_covered_lines=len(prior_lines),
                                new_lines=len(new_lines),
                            )
                            prior_lines.update(read_meta.line_numbers)
                            canonical.append(
                                _copy_chat_message(message, content=compacted)
                            )
                            saved_chars = max(0, len(text) - len(compacted))
                            stats.compacted_active_read_messages += 1
                            stats.compacted_active_read_chars += saved_chars
                            stats.compacted_active_read_inspection_messages += 1
                            stats.compacted_active_read_inspection_chars += saved_chars
                            stats.compacted_repeated_read_inspection_messages += 1
                            stats.compacted_repeated_read_inspection_chars += saved_chars
                            continue
                        prior_lines.update(read_meta.line_numbers)
                    compacted = _compact_active_read_tool_result_text(
                        read_text,
                        inspection_request=True,
                        inspection_max_lines=inspection_max_lines,
                        inspection_line_max_chars=inspection_line_max_chars,
                    )
                    if compacted is not None:
                        canonical.append(_copy_chat_message(message, content=compacted))
                        stats.compacted_active_read_messages += 1
                        stats.compacted_active_read_chars += max(
                            0, len(text) - len(compacted)
                        )
                        stats.compacted_active_read_inspection_messages += 1
                        stats.compacted_active_read_inspection_chars += max(
                            0, len(text) - len(compacted)
                        )
                        continue
                if (
                    plain_answer_cutoff is not None
                    and index < plain_answer_cutoff
                    or latest_assistant_cutoff is not None
                    and index < latest_assistant_cutoff
                ):
                    compacted = _compact_tool_result_text(text)
                    if compacted is not None:
                        canonical.append(_copy_chat_message(message, content=compacted))
                        stats.compacted_tool_result_messages += 1
                        stats.compacted_tool_result_chars += len(text) - len(compacted)
                        continue
                compacted = _compact_active_read_tool_result_text(
                    text,
                    inspection_request=inspection_request,
                    inspection_max_lines=inspection_max_lines,
                    inspection_line_max_chars=inspection_line_max_chars,
                )
                if compacted is not None:
                    canonical.append(_copy_chat_message(message, content=compacted))
                    stats.compacted_active_read_messages += 1
                    stats.compacted_active_read_chars += max(
                        0, len(text) - len(compacted)
                    )
                    if inspection_request:
                        stats.compacted_active_read_inspection_messages += 1
                        stats.compacted_active_read_inspection_chars += max(
                            0, len(text) - len(compacted)
                        )
                    continue
                compacted = _compact_active_tool_result_text(text)
                if compacted is not None:
                    canonical.append(_copy_chat_message(message, content=compacted))
                    stats.compacted_active_tool_result_messages += 1
                    stats.compacted_active_tool_result_chars += len(text) - len(compacted)
                    stats.compacted_active_tool_result_read_hints += (
                        _active_tool_result_read_hint_count(compacted)
                    )
                    continue
            canonical.append(message)
    if tools_active:
        canonical = _canonicalize_user_retry_pollution(canonical, stats)
    stats.canonical_message_chars = sum(
        len(_content_to_text(message.content)) for message in canonical
    )
    return canonical, stats


def _message_to_template_dict(
    message: ChatMessage,
    *,
    strip_assistant_reasoning_history: bool,
) -> dict[str, Any] | None:
    if not message.role:
        return None
    content = _content_to_text(message.content)
    if message.role == "assistant":
        content = (
            _strip_assistant_history_baggage(content)
            if strip_assistant_reasoning_history
            else _normalize_openwebui_reasoning_details(
                _strip_stats_footer(content)
            ).strip()
        )
    item: dict[str, Any] = {"role": message.role, "content": content}
    if message.name:
        item["name"] = message.name
    if message.tool_call_id:
        item["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        item["tool_calls"] = [_template_tool_call(call) for call in message.tool_calls]
    if content or message.role == "tool" or message.tool_calls:
        return item
    return None


def _coerce_token_ids(encoded: Any) -> list[int]:
    """Normalize tokenizer outputs to a plain token-id list.

    HF slow tokenizers, fast tokenizers, and mlx-lm's TokenizerWrapper do not
    all return the same shape from apply_chat_template(). SessionBank prefix
    comparisons must never depend on that wrapper detail.
    """
    if encoded is None:
        return []
    if hasattr(encoded, "ids"):
        return [int(token) for token in getattr(encoded, "ids")]
    if hasattr(encoded, "tolist"):
        return _coerce_token_ids(encoded.tolist())
    if isinstance(encoded, dict):
        if "input_ids" in encoded:
            return _coerce_token_ids(encoded["input_ids"])
        return []
    if isinstance(encoded, (list, tuple)):
        tokens: list[int] = []
        for item in encoded:
            if isinstance(item, int):
                tokens.append(int(item))
            elif (
                hasattr(item, "ids")
                or hasattr(item, "tolist")
                or isinstance(item, (list, tuple, dict))
            ):
                tokens.extend(_coerce_token_ids(item))
            else:
                tokens.append(int(item))
        return tokens
    return [int(encoded)]


def _encode_rendered_chat_text(tokenizer: Any, text: str) -> list[int]:
    try:
        return _coerce_token_ids(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return _coerce_token_ids(tokenizer.encode(text))


def _encode_plain_text(tokenizer: Any, text: str) -> list[int]:
    return _coerce_token_ids(tokenizer.encode(text))


_QWEN_ASSISTANT_THINK_PROMPT = "<|im_start|>assistant\n<think>\n"
_QWEN_IM_END = "<|im_end|>"
_DISABLED_THINK_GENERATION_PROMPT_RE = re.compile(
    r"(?is)(<\|im_start\|>assistant[^\n\r]*[\r\n]+)<think>\s*$"
)
_DISABLED_THINK_GENERATION_PROMPT_REPLACEMENT = (
    r"\1<think>\n\n</think>\n\n"
)


def _render_messages_with_chat_template(
    tokenizer: Any,
    normalized: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
    reasoning_effort: str | None,
    preserve_thinking: bool,
    tools: list[dict[str, Any]] | None,
    template_observability: dict[str, Any] | None = None,
) -> str | None:
    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
        "preserve_thinking": preserve_thinking,
    }
    if reasoning_effort:
        template_kwargs["reasoning_effort"] = reasoning_effort
    if tools:
        template_kwargs["tools"] = tools
    try:
        rendered = tokenizer.apply_chat_template(normalized, **template_kwargs)
    except TypeError:
        fallback_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if tools:
            fallback_kwargs["tools"] = tools
        try:
            rendered = tokenizer.apply_chat_template(normalized, **fallback_kwargs)
        except Exception:
            if not tools:
                return None
            try:
                if template_observability is not None:
                    template_observability["tool_template_fallback"] = True
                rendered = tokenizer.apply_chat_template(
                    normalized,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception:
                return None
    except Exception:
        if not tools:
            return None
        try:
            if template_observability is not None:
                template_observability["tool_template_fallback"] = True
            rendered = tokenizer.apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            return None
    if not isinstance(rendered, str):
        return None
    return _close_disabled_think_generation_prompt(
        rendered,
        enable_thinking=enable_thinking,
        add_generation_prompt=add_generation_prompt,
        template_observability=template_observability,
    )


def _close_disabled_think_generation_prompt(
    rendered: str,
    *,
    enable_thinking: bool,
    add_generation_prompt: bool,
    template_observability: dict[str, Any] | None = None,
) -> str:
    """Pre-close backend templates that open hidden reasoning when it is off.

    Step's tokenizer template currently appends ``<think>`` for generation
    regardless of ``enable_thinking``. Removing the tag entirely leaves the
    model in an unfamiliar assistant-turn format and it can generate a stray
    ``</think>``/``</thinks>`` marker itself. Qwen-style no-thinking turns are
    more stable when the empty thought block is already closed in the prompt.
    """

    if enable_thinking or not add_generation_prompt:
        return rendered
    closed = _DISABLED_THINK_GENERATION_PROMPT_RE.sub(
        _DISABLED_THINK_GENERATION_PROMPT_REPLACEMENT,
        rendered,
    )
    if closed != rendered and template_observability is not None:
        template_observability["disabled_thinking_prompt_closed"] = True
    return closed


def _tool_history_generation_boundaries(rendered: str) -> list[int]:
    """Find assistant tool-call history boundaries that must not be retokenized.

    Qwen's generation prompt ends with ``<think>\n``. On the next tool-result
    turn the structured assistant ``tool_calls`` history renders as the same
    visible text, but tokenizing the whole transcript can merge the newline
    after ``<think>`` with the generated ``</think>``/tool XML that follows.
    That one-token boundary drift is enough to make SessionBank miss and force
    OpenCode/Pi-style agents to cold-prefill the full tool schema again.

    Splitting exactly after the assistant generation prompt preserves the
    generation-time token boundary while keeping the rendered prompt text
    unchanged.
    """

    boundaries: list[int] = []
    marker = _QWEN_ASSISTANT_THINK_PROMPT
    marker_len = len(marker)
    search_from = 0
    while True:
        marker_at = rendered.find(marker, search_from)
        if marker_at < 0:
            break
        boundary = marker_at + marker_len
        block_end = rendered.find(_QWEN_IM_END, boundary)
        search_end = block_end if block_end >= 0 else len(rendered)
        if rendered.find("<tool_call>", boundary, search_end) >= 0:
            boundaries.append(boundary)
        search_from = boundary
    return boundaries


def _qwen_assistant_generation_boundaries(rendered: str) -> list[int]:
    boundaries: list[int] = []
    marker = _QWEN_ASSISTANT_THINK_PROMPT
    marker_len = len(marker)
    search_from = 0
    while True:
        marker_at = rendered.find(marker, search_from)
        if marker_at < 0:
            break
        boundary = marker_at + marker_len
        if 0 < boundary < len(rendered):
            boundaries.append(boundary)
        search_from = boundary
    return boundaries


def _qwen_plain_assistant_content_boundaries(rendered: str) -> list[int]:
    """Find Qwen no-thinking plain-text assistant generation boundaries.

    In no-thinking mode the request prompt already contains the closed empty
    thought block. Plain assistant text starts after that whole scaffold. If a
    postcommit snapshot splits after ``<think>\n`` instead, the newline pair is
    tokenized differently from the next request and SessionBank misses even
    though the rendered transcript is identical.
    """

    boundaries: list[int] = []
    marker = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    marker_len = len(marker)
    search_from = 0
    while True:
        marker_at = rendered.find(marker, search_from)
        if marker_at < 0:
            break
        boundary = marker_at + marker_len
        block_end = rendered.find(_QWEN_IM_END, boundary)
        search_end = block_end if block_end >= 0 else len(rendered)
        if (
            0 < boundary < len(rendered)
            and rendered.find("<tool_call>", boundary, search_end) < 0
        ):
            boundaries.append(boundary)
        search_from = boundary
    return boundaries


def _last_qwen_assistant_generation_boundary(rendered: str) -> int | None:
    marker_at = rendered.rfind(_QWEN_ASSISTANT_THINK_PROMPT)
    if marker_at < 0:
        return None
    boundary = marker_at + len(_QWEN_ASSISTANT_THINK_PROMPT)
    if boundary <= 0 or boundary >= len(rendered):
        return None
    return boundary


def _encode_rendered_chat_text_segmented(
    tokenizer: Any,
    rendered: str,
    boundaries: list[int],
) -> list[int]:
    if not boundaries:
        return _encode_rendered_chat_text(tokenizer, rendered)
    token_ids: list[int] = []
    start = 0
    for boundary in sorted(set(int(boundary) for boundary in boundaries)):
        if boundary <= start or boundary >= len(rendered):
            continue
        token_ids.extend(
            _encode_rendered_chat_text(tokenizer, rendered[start:boundary])
        )
        start = boundary
    if start < len(rendered):
        token_ids.extend(_encode_rendered_chat_text(tokenizer, rendered[start:]))
    return token_ids


def _encode_generation_compatible_tool_history(
    tokenizer: Any,
    normalized: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
    reasoning_effort: str | None,
    preserve_thinking: bool,
    tools: list[dict[str, Any]] | None,
    template_observability: dict[str, Any] | None = None,
) -> list[int] | None:
    if not tools or not add_generation_prompt:
        return None
    if not any(message.get("tool_calls") for message in normalized):
        return None
    rendered = _render_messages_with_chat_template(
        tokenizer,
        normalized,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        preserve_thinking=preserve_thinking,
        tools=tools,
        template_observability=template_observability,
    )
    if not rendered:
        return None
    boundaries = _qwen_assistant_generation_boundaries(rendered)
    if not boundaries:
        return None
    return _encode_rendered_chat_text_segmented(tokenizer, rendered, boundaries)


def _encode_messages(
    tokenizer: Any,
    messages: list[ChatMessage],
    *,
    enable_thinking: bool,
    reasoning_effort: str | None = None,
    strip_assistant_reasoning_history: bool = False,
    add_generation_prompt: bool = True,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    tool_prompt_mode: str = _TOOL_PROMPT_MODE_HYBRID,
    template_observability: dict[str, Any] | None = None,
) -> list[int]:
    prepared_messages: list[dict[str, Any]] = []
    for message in messages:
        item = _message_to_template_dict(
            message,
            strip_assistant_reasoning_history=strip_assistant_reasoning_history,
        )
        if item is not None:
            prepared_messages.append(item)
    normalized = omlx_normalize_messages_for_template(
        prepared_messages,
        tokenizer=tokenizer,
        native_reasoning_content=not strip_assistant_reasoning_history,
    )
    if strip_assistant_reasoning_history:
        for item in normalized:
            item.pop("reasoning_content", None)
    if not normalized:
        normalized = [{"role": "user", "content": ""}]
    effective_tool_prompt_mode = _normalize_tool_prompt_mode(tool_prompt_mode)
    if _tool_contract_active_for_mode(
        tools_active=bool(tools),
        tool_prompt_mode=effective_tool_prompt_mode,
    ):
        normalized = _with_mtplx_tool_contract(
            normalized,
            tools=tools,
            tool_choice=tool_choice,
        )
    elif effective_tool_prompt_mode == _TOOL_PROMPT_MODE_NATIVE and tools:
        normalized, native_tail_added = _with_mtplx_native_agent_tail(
            normalized,
            tools=tools,
        )
        if template_observability is not None:
            template_observability["native_agent_tail_contract_active"] = bool(
                native_tail_added
            )
    if is_gemma4_tokenizer(tokenizer):
        if template_observability is not None:
            template_observability["backend_chat_encoding"] = "gemma4"
        native_tools = (
            tools
            if effective_tool_prompt_mode == _TOOL_PROMPT_MODE_NATIVE and tools
            else None
        )
        return encode_chat_messages(
            tokenizer,
            normalized,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            add_generation_prompt=add_generation_prompt,
            preserve_thinking=not strip_assistant_reasoning_history,
            tools=native_tools,
        )
    template_tools = _template_tools_for_prompt_mode(
        tools,
        tool_prompt_mode=effective_tool_prompt_mode,
    )
    segmented_tool_history = _encode_generation_compatible_tool_history(
        tokenizer,
        normalized,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        preserve_thinking=not strip_assistant_reasoning_history,
        tools=template_tools,
        template_observability=template_observability,
    )
    if segmented_tool_history is not None:
        return segmented_tool_history
    if not enable_thinking:
        rendered = _render_messages_with_chat_template(
            tokenizer,
            normalized,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            preserve_thinking=not strip_assistant_reasoning_history,
            tools=template_tools,
            template_observability=template_observability,
        )
        if rendered is not None:
            return _encode_rendered_chat_text(tokenizer, rendered)
    template_kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
        "preserve_thinking": not strip_assistant_reasoning_history,
    }
    if reasoning_effort:
        template_kwargs["reasoning_effort"] = reasoning_effort
    if template_tools:
        template_kwargs["tools"] = template_tools
    try:
        return _coerce_token_ids(
            tokenizer.apply_chat_template(
                normalized,
                **template_kwargs,
            )
        )
    except TypeError:
        try:
            fallback_kwargs: dict[str, Any] = {
                "tokenize": True,
                "add_generation_prompt": add_generation_prompt,
            }
            if template_tools:
                fallback_kwargs["tools"] = template_tools
            return _coerce_token_ids(
                tokenizer.apply_chat_template(
                    normalized,
                    **fallback_kwargs,
                )
            )
        except (TypeError, Exception):
            if template_tools:
                try:
                    if template_observability is not None:
                        template_observability["tool_template_fallback"] = True
                    return _coerce_token_ids(
                        tokenizer.apply_chat_template(
                            normalized,
                            tokenize=True,
                            add_generation_prompt=add_generation_prompt,
                        )
                    )
                except Exception as schema_free_exc:
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            "tokenizer chat template failed after removing native "
                            f"tool schemas: {schema_free_exc}"
                        ),
                    ) from schema_free_exc
            pass
    except Exception:
        if template_tools:
            try:
                if template_observability is not None:
                    template_observability["tool_template_fallback"] = True
                return _coerce_token_ids(
                    tokenizer.apply_chat_template(
                        normalized,
                        tokenize=True,
                        add_generation_prompt=add_generation_prompt,
                    )
                )
            except Exception as schema_free_exc:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "tokenizer chat template failed after removing native "
                        f"tool schemas: {schema_free_exc}"
                    ),
                ) from schema_free_exc
            pass
    prompt = "\n".join(f"{item['role']}: {item['content']}" for item in normalized)
    if add_generation_prompt:
        prompt += "\nassistant:"
    return _encode_rendered_chat_text(tokenizer, prompt)


def _render_messages_for_postcommit(
    tokenizer: Any,
    normalized: list[dict[str, Any]],
    *,
    enable_thinking: bool,
    reasoning_effort: str | None = None,
    preserve_thinking: bool,
    tools: list[dict[str, Any]] | None,
    tool_prompt_mode: str = _TOOL_PROMPT_MODE_HYBRID,
) -> str | None:
    effective_tool_prompt_mode = _normalize_tool_prompt_mode(tool_prompt_mode)
    if _tool_contract_active_for_mode(
        tools_active=bool(tools),
        tool_prompt_mode=effective_tool_prompt_mode,
    ):
        normalized = _with_mtplx_tool_contract(normalized, tools=tools)
    template_tools = _template_tools_for_prompt_mode(
        tools,
        tool_prompt_mode=effective_tool_prompt_mode,
    )
    return _render_messages_with_chat_template(
        tokenizer,
        normalized,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        preserve_thinking=preserve_thinking,
        tools=template_tools,
    )


_POSTCOMMIT_SENTINEL_CONTENT = "__MTPLX_POSTCOMMIT_SENTINEL_4f02c7d2__"


def _sentinel_next_turn_start(
    rendered: str,
    *,
    sentinel_role: str,
    sentinel_content: str = _POSTCOMMIT_SENTINEL_CONTENT,
) -> int | None:
    sentinel_at = rendered.find(sentinel_content)
    if sentinel_at < 0:
        return None
    role = sentinel_role
    role_title = role[:1].upper() + role[1:]
    strict_markers = [
        f"<|im_start|>{role}\n",
        f"<|im_start|>{role}\r\n",
        f"<|start_header_id|>{role}<|end_header_id|>\n\n",
        f"\n### {role_title}:\n",
    ]
    loose_markers = [
        f"\n{role}:",
        f"\n{role_title}:",
        f"{role}:",
        f"{role_title}:",
    ]
    candidates = [
        idx
        for marker in strict_markers
        if (idx := rendered.rfind(marker, 0, sentinel_at)) >= 0
    ]
    if candidates:
        return max(candidates)

    # Loose "tool:" / "User:" style markers are fallback support for simple
    # templates only. With OpenCode/Qwen tool schemas, the rendered prompt can
    # contain words like "tool:" thousands of characters before the synthetic
    # sentinel. Treating those as a chat boundary stores a short unreachable
    # SessionBank prefix and forces the next tool-result request to cold
    # prefill. Only accept loose markers that are close to the sentinel.
    loose_lookback_chars = 1024
    window_start = max(0, sentinel_at - loose_lookback_chars)
    candidates = [
        idx
        for marker in loose_markers
        if (idx := rendered.rfind(marker, window_start, sentinel_at)) >= 0
    ]
    if not candidates:
        return None
    return max(candidates)


def _postcommit_next_turn_prefix_ids(
    tokenizer: Any,
    history_messages: list[ChatMessage],
    *,
    enable_thinking: bool,
    reasoning_effort: str | None = None,
    strip_assistant_reasoning_history: bool,
    tools: list[dict[str, Any]] | None,
    assistant_tool_calls: list[dict[str, Any]] | None,
    tool_prompt_mode: str = _TOOL_PROMPT_MODE_HYBRID,
) -> list[int] | None:
    sentinel_role = "user"
    sentinel_message = ChatMessage(role="user", content=_POSTCOMMIT_SENTINEL_CONTENT)
    if assistant_tool_calls:
        first_tool_call = assistant_tool_calls[0] if assistant_tool_calls else {}
        sentinel_role = "tool"
        sentinel_message = ChatMessage(
            role="tool",
            content=_POSTCOMMIT_SENTINEL_CONTENT,
            tool_call_id=str(first_tool_call.get("id") or "call_mtplx_postcommit"),
        )

    normalized: list[dict[str, Any]] = []
    last_history_role: str | None = None
    for message in history_messages:
        item = _message_to_template_dict(
            message,
            strip_assistant_reasoning_history=strip_assistant_reasoning_history,
        )
        if item is not None:
            normalized.append(item)
            last_history_role = str(item.get("role") or "")
    item = _message_to_template_dict(
        sentinel_message,
        strip_assistant_reasoning_history=strip_assistant_reasoning_history,
    )
    if item is not None:
        normalized.append(item)
    if not normalized:
        return None

    rendered = _render_messages_for_postcommit(
        tokenizer,
        normalized,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        preserve_thinking=not strip_assistant_reasoning_history,
        tools=tools,
        tool_prompt_mode=tool_prompt_mode,
    )
    if not rendered:
        return None
    sentinel_at = rendered.find(_POSTCOMMIT_SENTINEL_CONTENT)
    turn_start = _sentinel_next_turn_start(
        rendered,
        sentinel_role=sentinel_role,
    )
    if turn_start is None and sentinel_role == "tool":
        # Qwen-style templates render OpenAI tool-result messages as a user
        # turn containing <tool_response>. The sentinel is still a tool
        # message semantically, but the rendered boundary to cut at is user.
        user_turn_start = _sentinel_next_turn_start(rendered, sentinel_role="user")
        if (
            user_turn_start is not None
            and sentinel_at >= 0
            and (sentinel_at - user_turn_start) <= 2048
        ):
            turn_start = user_turn_start
    if turn_start is None:
        return None
    prefix_text = rendered[:turn_start]
    if not prefix_text:
        return None
    # Match _encode_messages(): assistant-boundary segmentation is only used
    # when native template tools are active. Compact OpenCode history uses the
    # schema-free contract and the normal prompt path plain-tokenizes the
    # rendered chat, so segmenting here would create a different cache key for
    # identical rendered text.
    template_tools = _template_tools_for_prompt_mode(
        tools,
        tool_prompt_mode=tool_prompt_mode,
    )
    boundaries: list[int] = []
    if template_tools:
        boundaries = _tool_history_generation_boundaries(prefix_text)
        if not enable_thinking:
            boundaries.extend(_qwen_plain_assistant_content_boundaries(prefix_text))
        elif last_history_role == "assistant":
            boundary = _last_qwen_assistant_generation_boundary(prefix_text)
            if boundary is not None:
                boundaries.append(boundary)
    return _encode_rendered_chat_text_segmented(tokenizer, prefix_text, boundaries)


def _encode_prompt(
    tokenizer: Any, prompt: str | list[int] | list[str] | None
) -> list[int]:
    if prompt is None:
        return []
    if isinstance(prompt, str):
        return _encode_plain_text(tokenizer, prompt)
    if isinstance(prompt, list) and all(isinstance(item, int) for item in prompt):
        return [int(item) for item in prompt]
    if isinstance(prompt, list):
        return _encode_plain_text(
            tokenizer,
            "\n".join(str(item) for item in prompt),
        )
    return _encode_plain_text(tokenizer, str(prompt))


def _count_text_tokens(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(text))
    except Exception:
        return 0


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return int(value)
    except Exception:
        return str(value)


def _request_extra(model: BaseModel, key: str, default: Any = None) -> Any:
    if hasattr(model, key):
        value = getattr(model, key)
        if value is not None:
            return value
    extra = getattr(model, "model_extra", None) or {}
    return extra.get(key, default)


def _request_metadata(model: BaseModel) -> dict[str, Any]:
    metadata = _request_extra(model, "metadata", {})
    return metadata if isinstance(metadata, dict) else {}


_CLIENT_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")


def _response_id_from_client_hint(
    *,
    prefix: str,
    headers: Mapping[str, str],
    metadata: Mapping[str, Any],
) -> str:
    raw = headers.get("x-mtplx-request-id") or metadata.get("mtplx_request_id")
    if raw is None:
        return f"{prefix}-{uuid.uuid4().hex}"
    hint = str(raw).strip()
    if not _CLIENT_REQUEST_ID_RE.fullmatch(hint):
        return f"{prefix}-{uuid.uuid4().hex}"
    prefix_with_dash = f"{prefix}-"
    return hint if hint.startswith(prefix_with_dash) else f"{prefix_with_dash}{hint}"


def _request_max_tokens(request: BaseModel) -> int | None:
    value = getattr(request, "max_tokens", None)
    if value is not None:
        return int(value)
    alias = getattr(request, "max_completion_tokens", None)
    return None if alias is None else int(alias)


def _is_opencode_title_request(request: ChatCompletionRequest) -> bool:
    """Detect OpenCode's auxiliary thread-title call.

    OpenCode sends this beside the real coding request. Letting the 27B model
    spend seconds on a cosmetic title blocks the single model-owner lane and
    makes users stare at "Thinking..." before their actual request starts.
    This fast path only handles the exact title-generator prompt shape and
    never touches normal user chat/tool requests.
    """
    if request.tools or len(request.messages) < 2:
        return False
    first = request.messages[0]
    if str(first.role).lower() != "system":
        return False
    system_text = _content_to_text(first.content)
    return (
        "You are a title generator" in system_text
        and "Generate a brief title" in system_text
        and "Never use tools" in system_text
    )


def _opencode_fast_title(request: ChatCompletionRequest) -> str:
    user_text = ""
    for message in reversed(request.messages):
        if str(message.role).lower() != "user":
            continue
        text = _content_to_text(message.content).strip()
        if not text or text.lower().startswith("generate a title"):
            continue
        user_text = text
        break
    cleaned = re.sub(r"\s+", " ", user_text).strip(" `\"'")
    lowered = cleaned.lower().strip()
    if lowered in {"hi", "hello", "hey", "yo", "howdy", "sup", "what's up", "whats up"}:
        return "Greeting"
    if not cleaned:
        return "Quick chat"
    words = cleaned.split()
    title = " ".join(words[:8])
    if len(title) > 50:
        title = title[:50].rstrip(" ,.;:-")
    return title or "Quick chat"


def _opencode_title_response(
    state: ServerState,
    request: ChatCompletionRequest,
    *,
    model: str,
    response_id: str,
    created: int,
    request_max_tokens: int | None,
) -> Response:
    title = _opencode_fast_title(request)
    stats = {
        "opencode_title_fast_path": True,
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "new_prefill_tokens": 0,
        "completion_tokens": len(title.split()),
        "ttft_s": 0.0,
        "prompt_tps": 0.0,
        "decode_tok_s": 0.0,
        "request_max_tokens": request_max_tokens,
        "server_max_response_tokens": getattr(state.args, "max_response_tokens", None),
        "effective_max_tokens": request_max_tokens,
        "request_message_count": len(request.messages),
        "request_message_roles": [message.role for message in request.messages],
        "request_tool_count": 0,
        "request_client_hint": "opencode_title",
    }
    state.last_metrics.append(stats)
    state.requests_completed = int(getattr(state, "requests_completed", 0) or 0) + 1
    state.last_request_at = time.time()
    usage = {
        "prompt_tokens": 0,
        "completion_tokens": stats["completion_tokens"],
        "total_tokens": stats["completion_tokens"],
    }
    if request.stream:

        def stream():
            first = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
            content = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"content": title}, "finish_reason": None}
                ],
            }
            done = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": usage,
                "mtplx_stats": stats,
            }
            for payload in (first, content, done):
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")
    return JSONResponse(
        {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": title},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
            "mtplx_stats": stats,
        }
    )


def _normalize_generation_mode(value: Any, *, default: str = "mtp") -> str:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text not in {"mtp", "ar"}:
        raise HTTPException(
            status_code=400,
            detail="generation_mode must be 'mtp' or 'ar'",
        )
    return text


def _request_generation_mode_value(request: BaseModel) -> Any:
    value = getattr(request, "generation_mode", None)
    if value is None:
        value = _request_extra(request, "generation_mode")
    return value


def _request_generation_mode_for_generation(
    state: ServerState,
    request: BaseModel,
    *,
    allow_client_controls: bool = True,
) -> str:
    default = _normalize_generation_mode(getattr(state.args, "generation_mode", "mtp"))
    mode = _normalize_generation_mode(
        _request_generation_mode_value(request) if allow_client_controls else None,
        default=default,
    )
    if mode == "mtp" and not bool(getattr(state.runtime, "mtp_enabled", False)):
        raise HTTPException(
            status_code=400,
            detail="generation_mode 'mtp' requires a runtime loaded with MTP",
        )
    return mode


def _request_depth_value(request: BaseModel) -> Any:
    for key in (
        "depth",
        "mtp_depth",
        "speculative_depth",
        "draft_block_size",
        "gemma_draft_block_size",
    ):
        value = getattr(request, key, None)
        if value is None:
            value = _request_extra(request, key)
        if value is not None:
            return value
    return None


def _request_draft_control_value(
    request: BaseModel,
    descriptor: BackendDescriptor,
) -> Any:
    if descriptor.draft_semantics.unit == "block":
        ordered = (
            descriptor.draft_semantics.request_field,
            "draft_block_size",
            "gemma_draft_block_size",
            "depth",
            "mtp_depth",
            "speculative_depth",
        )
    else:
        ordered = (
            descriptor.draft_semantics.request_field,
            "depth",
            "mtp_depth",
            "speculative_depth",
            "draft_block_size",
            "gemma_draft_block_size",
        )
    seen: set[str] = set()
    for key in ordered:
        if key in seen:
            continue
        seen.add(key)
        value = getattr(request, key, None)
        if value is None:
            value = _request_extra(request, key)
        if value is not None:
            return value
    return None


def _request_client_hint_from_headers(
    headers: Mapping[str, str],
    metadata: Mapping[str, Any],
) -> str | None:
    user_agent = headers.get("user-agent") or headers.get("User-Agent") or ""
    user_agent_lower = user_agent.lower()
    explicit_client = (
        headers.get("x-mtplx-client")
        or headers.get("X-MTPLX-Client")
        or headers.get("x-client-name")
        or headers.get("X-Client-Name")
        or metadata.get("client")
        or metadata.get("client_label")
    )
    if explicit_client:
        return str(explicit_client).strip().lower().replace(" ", "_")
    launch_client = os.getenv("MTPLX_CLIENT", "").strip()
    if launch_client:
        return launch_client.lower().replace(" ", "_")
    if "opencode" in user_agent_lower:
        return "opencode"
    if "android" in user_agent_lower or "jetbrains" in user_agent_lower:
        return "android_studio"
    if "ai-sdk" in user_agent_lower:
        return "ai_sdk_agent"
    return None


def _truthy_control_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "allow"}


_MTPLX_MANAGED_CLIENT_HINTS = {
    "browser",
    "chat",
    "hermes",
    "mtplx",
    "mtplx_app",
    "mtplxapp",
    "opencode",
    "openwebui",
    "pi",
}


def _app_managed_client_hint(
    headers: Mapping[str, str],
    metadata: Mapping[str, Any],
) -> str | None:
    hint = _request_client_hint_from_headers(headers, metadata)
    if not hint:
        return None
    normalized = str(hint).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in _MTPLX_MANAGED_CLIENT_HINTS:
        return normalized
    if normalized.startswith("mtplx_"):
        return normalized
    return None


def _client_controls_allowed(
    headers: Mapping[str, str],
    metadata: Mapping[str, Any],
) -> bool:
    if _app_managed_client_hint(headers, metadata):
        return False
    value = (
        headers.get("x-mtplx-allow-client-controls")
        or headers.get("X-MTPLX-Allow-Client-Controls")
        or metadata.get("allow_client_controls")
        or metadata.get("mtplx_allow_client_controls")
    )
    return _truthy_control_value(value)


def _ignored_client_control_fields(request: BaseModel) -> list[str]:
    """Request controls ignored unless the caller explicitly opts in.

    MTPLX-owned launch/live settings are the authority for generation
    policy. Client payload controls become observable hints by default.
    """

    fields: list[str] = []
    if getattr(request, "temperature", None) is not None:
        fields.append("temperature")
    if getattr(request, "top_p", None) is not None:
        fields.append("top_p")
    if getattr(request, "top_k", None) is not None:
        fields.append("top_k")
    if getattr(request, "enable_thinking", None) is not None:
        fields.append("enable_thinking")
    if getattr(request, "reasoning_effort", None) is not None:
        fields.append("reasoning_effort")
    if _request_generation_mode_value(request) is not None:
        fields.append("generation_mode")
    if _request_depth_value(request) is not None:
        fields.append("draft_control")
    return fields


def _request_depth_for_generation(
    state: ServerState,
    request: BaseModel,
    *,
    generation_mode: str,
    allow_client_controls: bool = True,
) -> int:
    descriptor = _backend_descriptor(state)
    if generation_mode == "ar":
        return 0
    value = (
        _request_draft_control_value(request, descriptor)
        if allow_client_controls
        else None
    )
    if value is None:
        default_value = getattr(
            state.args,
            descriptor.draft_semantics.request_field,
            None,
        )
        if default_value is None:
            default_value = getattr(
                state.args,
                "depth",
                descriptor.draft_semantics.default,
            )
        return descriptor.draft_semantics.clamp(default_value)
    try:
        depth = int(value)
    except (TypeError, ValueError) as exc:
        detail = f"{descriptor.draft_semantics.display_label.lower()} must be an integer"
        raise HTTPException(status_code=400, detail=detail) from exc
    minimum = descriptor.draft_semantics.minimum
    maximum = descriptor.draft_semantics.maximum
    if depth < minimum or depth > maximum:
        detail = (
            f"{descriptor.draft_semantics.display_label.lower()} must be "
            f"between {minimum} and {maximum}"
        )
        raise HTTPException(status_code=400, detail=detail)
    return depth


def _opencode_short_context_depth_policy(
    request: BaseModel,
    *,
    headers: dict[str, str],
    metadata: dict[str, Any],
    generation_mode: str,
    request_depth: int,
    prompt_tokens: int,
) -> tuple[int, dict[str, Any]]:
    client_hint = _request_client_hint_from_headers(headers, metadata)
    policy = {
        "active": False,
        "client": client_hint,
        "effective_depth": int(request_depth),
        "explicit_depth": _request_depth_value(request) is not None,
        "prompt_tokens": int(prompt_tokens),
        "reason": "disabled_depth_preservation",
        "requested_depth": int(request_depth),
        "threshold": None,
    }
    if generation_mode == "ar" or request_depth <= 0:
        policy["reason"] = "generation_mode_ar"
        return request_depth, policy
    if policy["explicit_depth"]:
        policy["reason"] = "explicit_depth"
        return request_depth, policy
    if client_hint is None or "opencode" not in client_hint:
        policy["reason"] = "not_opencode"
        return request_depth, policy
    return request_depth, policy


def _long_context_mtp_depth_policy_for_request(
    state: ServerState,
    *,
    generation_mode: str,
    request_depth: int,
    prompt_tokens: int,
) -> tuple[int, dict[str, Any]]:
    if generation_mode == "ar" or request_depth <= 0:
        return 0, {}
    descriptor = _backend_descriptor(state)
    if not descriptor.supports("native_adaptive_depth_policy"):
        return request_depth, {
            "active": False,
            "reason": f"{descriptor.backend_id}_owns_draft_policy",
        }
    effective_depth, policy = resolve_long_context_mtp_depth(
        prompt_tokens=prompt_tokens,
        requested_depth=request_depth,
        min_depth=1,
    )
    return int(effective_depth), dict(policy)


def _token_window_rate(token_times: list[float], window: int) -> float | None:
    if len(token_times) < 2:
        return None
    subset = token_times[-window:]
    if len(subset) < 2:
        return None
    elapsed = subset[-1] - subset[0]
    if elapsed <= 0:
        return None
    return (len(subset) - 1) / elapsed


def _token_window_rate_first(token_times: list[float], window: int) -> float | None:
    if len(token_times) < 2:
        return None
    subset = token_times[:window]
    if len(subset) < 2:
        return None
    elapsed = subset[-1] - subset[0]
    if elapsed <= 0:
        return None
    return (len(subset) - 1) / elapsed


MAINTENANCE_TIMING_STATS_KEYS = (
    "mtp_history_materialize_every",
    "mtp_history_materialize_events",
    "clear_cache_every",
    "clear_cache_events",
    "clear_cache_time_s",
    "trunk_cache_materialize_every",
    "trunk_cache_materialize_events",
    "trunk_cache_materialize_time_s",
    "dirty_detach_components",
    "dirty_detach_mode",
    "dirty_detach_gdn_every",
    "dirty_detach_conv_every",
    "dirty_detach_attn_every",
    "dirty_detach_events",
    "dirty_detach_time_s",
    "dirty_detach_arrays",
    "dirty_detach_bytes",
    "live_output_detach_enabled",
    "live_output_detach_mode",
    "live_output_detach_events",
    "live_output_detach_time_s",
    "live_output_detach_arrays",
    "live_output_detach_bytes",
    "state_rebase_every",
    "state_rebase_events",
    "state_rebase_time_s",
    "state_root_eval_enabled",
    "state_root_eval_include_mtp",
    "state_root_eval_events",
    "state_root_eval_time_s",
    "state_root_eval_arrays",
    "capture_commit_detach_components",
    "capture_commit_detach_mode",
    "capture_commit_detach_gdn_every",
    "capture_commit_detach_conv_every",
    "capture_commit_detach_events",
    "capture_commit_detach_time_s",
    "capture_commit_detach_arrays",
    "capture_commit_detach_bytes",
    "trace_accounting_time_s",
)


def _maintenance_timing_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        key: stats[key]
        for key in MAINTENANCE_TIMING_STATS_KEYS
        if key in stats
    }


def _metrics_envelope(
    *,
    stats: dict[str, Any],
    prompt_tokens: int,
    completion_tokens: int,
    request_elapsed_s: float,
    token_times: list[float],
    request_started_s: float,
    lock_wait_time_s: float,
    session_id: str | None,
    session_cache_hit: bool,
    cache_miss_reason: str | None,
    session_restore_mode: str,
    mtp_depth: int,
    generation_limits: dict[str, Any],
) -> dict[str, Any]:
    decode_tok_s, decode_elapsed_s = _decode_timing(stats)
    sliding_decode_tok_s_first_32 = _token_window_rate_first(token_times, 32)
    sliding_decode_tok_s_first_64 = _token_window_rate_first(token_times, 64)
    sliding_decode_tok_s_first_128 = _token_window_rate_first(token_times, 128)
    sliding_decode_tok_s_first_256 = _token_window_rate_first(token_times, 256)
    sliding_decode_tok_s_last_32 = _token_window_rate(token_times, 32)
    sliding_decode_tok_s_last_64 = _token_window_rate(token_times, 64)
    sliding_decode_tok_s_last_128 = _token_window_rate(token_times, 128)
    sliding_decode_tok_s_last_256 = _token_window_rate(token_times, 256)
    # Keep this legacy field equal to the completed-request generation TPS.
    # The sliding-window rates below remain available for diagnostics, but
    # consumer UI must not present a token-window burst as "TPS".
    display_decode_tok_s = decode_tok_s
    prompt_eval_time_s = float(stats.get("prompt_eval_time_s") or 0.0)
    ttft_s = max(0.0, token_times[0] - request_started_s) if token_times else None
    cached_tokens = int(stats.get("cached_tokens") or 0)
    new_prefill_tokens = int(
        stats.get("new_prefill_tokens") or (prompt_tokens - cached_tokens)
    )
    prefill_tok_s = (
        max(0, new_prefill_tokens) / prompt_eval_time_s
        if prompt_eval_time_s > 0
        else None
    )
    return {
        "prompt_tokens": int(prompt_tokens),
        "cached_tokens": cached_tokens,
        "new_prefill_tokens": max(0, new_prefill_tokens),
        "cache_source": str(stats.get("cache_source") or ("ram" if session_cache_hit else "none")),
        "ssd_cache_hit": bool(stats.get("ssd_cache_hit") or False),
        "ssd_cached_tokens": int(stats.get("ssd_cached_tokens") or 0),
        "ssd_restore_s": float(stats.get("ssd_restore_s") or 0.0),
        "ssd_suffix_tokens": int(stats.get("ssd_suffix_tokens") or 0),
        "completion_tokens": int(completion_tokens),
        "prompt_eval_time_s": prompt_eval_time_s,
        "cache_restore_time_s": max(
            float(stats.get("cache_restore_time_s") or 0.0),
            float(stats.get("ssd_restore_s") or 0.0),
        ),
        "prefill_tok_s": prefill_tok_s,
        "prefill_compute_tok_s": prefill_tok_s,
        "prefill_wall_tok_s": stats.get("prefill_wall_tok_s"),
        "prompt_tps": prefill_tok_s,
        "ttft_s": ttft_s,
        "decode_elapsed_s": decode_elapsed_s,
        "request_elapsed_s": request_elapsed_s,
        "request_tok_s": completion_tokens / request_elapsed_s
        if request_elapsed_s > 0
        else 0.0,
        "decode_tok_s": decode_tok_s,
        "display_decode_tok_s": display_decode_tok_s,
        "sliding_decode_tok_s_first_32": sliding_decode_tok_s_first_32,
        "sliding_decode_tok_s_first_64": sliding_decode_tok_s_first_64,
        "sliding_decode_tok_s_first_128": sliding_decode_tok_s_first_128,
        "sliding_decode_tok_s_first_256": sliding_decode_tok_s_first_256,
        "sliding_decode_tok_s_last_32": sliding_decode_tok_s_last_32,
        "sliding_decode_tok_s_last_64": sliding_decode_tok_s_last_64,
        "sliding_decode_tok_s_last_128": sliding_decode_tok_s_last_128,
        "sliding_decode_tok_s_last_256": sliding_decode_tok_s_last_256,
        "mtp_depth": int(mtp_depth),
        "verify_calls": int(stats.get("verify_calls") or 0),
        "accepted_by_depth": stats.get("accepted_by_depth") or [],
        "drafted_by_depth": stats.get("drafted_by_depth") or [],
        "mean_accept_probability_by_depth": (
            stats.get("mean_accept_probability_by_depth") or []
        ),
        "correction_tokens": int(stats.get("correction_tokens") or 0),
        "bonus_tokens": int(stats.get("bonus_tokens") or 0),
        "verify_time_s": float(stats.get("verify_time_s") or 0.0),
        "verify_forward_time_s": float(stats.get("verify_forward_time_s") or 0.0),
        "verify_eval_time_s": float(stats.get("verify_eval_time_s") or 0.0),
        "verify_logits_eval_time_s": float(
            stats.get("verify_logits_eval_time_s") or 0.0
        ),
        "verify_hidden_eval_time_s": float(
            stats.get("verify_hidden_eval_time_s") or 0.0
        ),
        "verify_joint_eval_time_s": float(stats.get("verify_joint_eval_time_s") or 0.0),
        "verify_target_distribution_time_s": float(
            stats.get("verify_target_distribution_time_s") or 0.0
        ),
        "target_distribution_materialized_rows": int(
            stats.get("target_distribution_materialized_rows") or 0
        ),
        "target_distribution_materialized_windows": int(
            stats.get("target_distribution_materialized_windows") or 0
        ),
        "target_distribution_share": float(
            stats.get("target_distribution_share") or 0.0
        ),
        "lazy_bonus_verify_calls": int(stats.get("lazy_bonus_verify_calls") or 0),
        "lazy_bonus_commit_time_s": float(
            stats.get("lazy_bonus_commit_time_s") or 0.0
        ),
        "verify_eval_unattributed_time_s": float(
            stats.get("verify_eval_unattributed_time_s") or 0.0
        ),
        "target_forward_time_s": float(stats.get("target_forward_time_s") or 0.0),
        "draft_time_s": float(stats.get("draft_time_s") or 0.0),
        "accept_time_s": float(stats.get("accept_time_s") or 0.0),
        "repair_time_s": float(stats.get("repair_time_s") or 0.0),
        "mtp_history_policy": str(stats.get("mtp_history_policy") or ""),
        "mtp_history_window_tokens": int(
            stats.get("mtp_history_window_tokens") or 0
        ),
        "mtp_history_position_base": int(
            stats.get("mtp_history_position_base") or 0
        ),
        **_maintenance_timing_stats(stats),
        "session_cache_hit": bool(session_cache_hit),
        "cache_miss_reason": cache_miss_reason,
        "session_restore_mode": session_restore_mode,
        "context_len": int(prompt_tokens + completion_tokens),
        "repetition_stop_triggered": bool(
            stats.get("repetition_stop_triggered") or False
        ),
        "repetition_stop_reason": stats.get("repetition_stop_reason"),
        "repetition_stop_block_tokens": int(
            stats.get("repetition_stop_block_tokens") or 0
        ),
        "repetition_stop_repeats": int(stats.get("repetition_stop_repeats") or 0),
        "repetition_stop_trimmed_tokens": int(
            stats.get("repetition_stop_trimmed_tokens") or 0
        ),
        "repetition_stop_raw_tokens": int(
            stats.get("repetition_stop_raw_tokens") or 0
        ),
        "lock_wait_time_s": lock_wait_time_s,
        "session_id": session_id,
        **generation_limits,
    }


def _effective_completion_tokens(
    *,
    generated_tokens: list[int],
    streamed_token_times: list[float],
) -> int:
    return max(len(generated_tokens), len(streamed_token_times))


def _repair_streamed_generation_stats(
    stats: dict[str, Any],
    *,
    completion_tokens: int,
    elapsed_s: float,
) -> dict[str, Any]:
    repaired = dict(stats)
    raw_generated = int(repaired.get("generated_tokens") or 0)
    if completion_tokens > raw_generated:
        prompt_eval_time_s = float(repaired.get("prompt_eval_time_s") or 0.0)
        decode_elapsed_s = max(0.0, elapsed_s - prompt_eval_time_s)
        decode_tok_s = (
            float(completion_tokens) / decode_elapsed_s
            if decode_elapsed_s > 0.0
            else 0.0
        )
        repaired["generated_tokens_raw"] = raw_generated
        repaired["generated_tokens_recovered_from_stream"] = True
        repaired["generated_tokens"] = int(completion_tokens)
        repaired["tok_s"] = decode_tok_s
        repaired["decode_tok_s"] = decode_tok_s
        repaired["decode_elapsed_s"] = decode_elapsed_s
        repaired["end_to_end_tok_s"] = (
            float(completion_tokens) / elapsed_s if elapsed_s > 0 else 0.0
        )
    return repaired


_MACHINE_INFO_CACHE: dict[str, Any] = {}


def _machine_info() -> dict[str, Any]:
    """Resolve hardware identity (chip, model, and unified-memory bytes), cached.

    These values come from ``sysctl`` on macOS; on non-macOS the fields are
    filled with best-effort fallbacks so the dashboard's HardwareBanner
    always has something honest to render.
    """

    if _MACHINE_INFO_CACHE:
        return dict(_MACHINE_INFO_CACHE)
    chip: str | None = None
    model: str | None = None
    mem_bytes: int | None = None
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            text=True,
            capture_output=True,
            timeout=1.0,
        ).stdout.strip() or None
    except Exception:
        pass
    try:
        model = subprocess.run(
            ["sysctl", "-n", "hw.model"],
            check=True,
            text=True,
            capture_output=True,
            timeout=1.0,
        ).stdout.strip() or None
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=True,
            text=True,
            capture_output=True,
            timeout=1.0,
        ).stdout.strip()
        mem_bytes = int(result) if result.isdigit() else None
    except Exception:
        pass
    if model is None:
        try:
            import platform as _platform

            model = _platform.machine() or _platform.processor() or None
        except Exception:
            model = None
    if mem_bytes is None:
        try:
            mem_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except Exception:
            mem_bytes = None
    info = {"chip": chip, "machine_model": model, "unified_memory_bytes": mem_bytes}
    _MACHINE_INFO_CACHE.update(info)
    return dict(info)


def _mlx_memory_stats_live() -> dict[str, Any]:
    """Snapshot of MLX's live memory accessors (active, peak, cache).

    Returns ``{"ok": False, ...}`` if MLX is unavailable or any accessor
    raises; the dashboard surfaces ``ok`` as a graceful empty state.

    Prefers the top-level ``mx.get_*`` accessors and falls back to
    ``mx.metal.get_*`` for older MLX versions; the metal namespace was
    deprecated in mlx 0.30+.
    """

    try:
        import mlx.core as _mx
    except Exception as exc:
        return {"ok": False, "error": f"mlx unavailable: {exc!r}"}
    snapshot: dict[str, Any] = {"ok": True}
    metal_ns = getattr(_mx, "metal", None)
    for attr in ("get_active_memory", "get_cache_memory", "get_peak_memory"):
        fn = getattr(_mx, attr, None) or getattr(metal_ns, attr, None)
        try:
            snapshot[attr.removeprefix("get_") + "_bytes"] = int(fn()) if fn else None
        except Exception:
            snapshot[attr.removeprefix("get_") + "_bytes"] = None
    return snapshot


def _dashboard_prompt_preview(
    request: Any, tokenizer: Any, *, max_chars: int = 96
) -> str:
    """Best-effort short preview of the last user message for the in-flight panel."""

    del tokenizer  # reserved for future token-level previews
    try:
        messages = getattr(request, "messages", None) or []
        prompt = getattr(request, "prompt", None)
        text: str = ""
        if isinstance(prompt, str) and prompt:
            text = prompt
        else:
            for message in reversed(messages):
                if getattr(message, "role", None) == "user":
                    content = getattr(message, "content", None)
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text = str(part.get("text") or "")
                                if text:
                                    break
                    if text:
                        break
        text = text.replace("\n", " ").replace("\r", " ").strip()
        if len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."
        return text
    except Exception:
        return ""


def _dashboard_record_completion(
    state: "ServerState",
    *,
    envelope: dict[str, Any],
    stats: dict[str, Any],
) -> None:
    """Feed a finished generation's envelope into the dashboard primitives.

    Called once per generation completion. Safe to call when the dashboard
    primitives are not yet initialized (early warmup): we short-circuit if
    ``state.dashboard`` is unavailable. The function is deliberately
    tolerant — a missing/malformed metric is treated as zero rather than
    raising, so a transient stats glitch never breaks a user-visible
    response.
    """

    dashboard = getattr(state, "dashboard", None)
    if dashboard is None:
        return
    try:
        request_id = envelope.get("request_id") or stats.get("request_id")
        if isinstance(request_id, str) and request_id:
            dashboard.in_flight.deregister(request_id)
            dashboard.progress_events.forget(request_id)
        prompt_tokens = int(envelope.get("prompt_tokens") or 0)
        completion_tokens = int(envelope.get("completion_tokens") or 0)
        cached_tokens = int(envelope.get("cached_tokens") or 0)
        raw_decode_tok_s = envelope.get("decode_tok_s")
        display_decode_tok_s = envelope.get("display_decode_tok_s")
        decode_tok_s = (
            raw_decode_tok_s
            if isinstance(raw_decode_tok_s, (int, float)) and raw_decode_tok_s > 0
            else display_decode_tok_s
        )
        session_id = envelope.get("session_id")
        dashboard.lifetime.record_completion(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
        )
        is_new_max = False
        if isinstance(decode_tok_s, (int, float)) and decode_tok_s > 0:
            is_new_max = dashboard.rolling.append(float(decode_tok_s), session_id)
        prefill_row = {
            "t": time.time(),
            "session_id": session_id,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "new_prefill_tokens": int(envelope.get("new_prefill_tokens") or 0),
            "prompt_eval_time_s": float(envelope.get("prompt_eval_time_s") or 0.0),
            "prefill_tok_s": envelope.get("prefill_tok_s"),
            "prefill_compute_tok_s": envelope.get("prefill_compute_tok_s")
            or envelope.get("prefill_tok_s"),
            "prefill_wall_tok_s": envelope.get("prefill_wall_tok_s"),
            "ttft_s": envelope.get("ttft_s"),
            "session_cache_hit": bool(envelope.get("session_cache_hit")),
            "cache_miss_reason": envelope.get("cache_miss_reason"),
            "context_len": int(envelope.get("context_len") or 0),
            "model_id": state.model_id,
        }
        dashboard.prefill_history.append(prefill_row)
        dashboard.bus.publish(
            {
                "kind": "completed",
                "when_s": time.time(),
                "envelope": dict(envelope),
            }
        )
        if is_new_max and isinstance(decode_tok_s, (int, float)):
            dashboard.bus.publish(
                {
                    "kind": "new_max_tps",
                    "when_s": time.time(),
                    "tok_s": float(decode_tok_s),
                    "session_id": session_id,
                    "raw_decode_tok_s": raw_decode_tok_s,
                }
            )
    except Exception as exc:
        # Never let a metrics bug break a user response.
        _safe_stdout_print(f"[dashboard] record_completion suppressed error: {exc!r}")


def _dashboard_publish_prefill(
    state: "ServerState",
    *,
    request_id: str,
    payload: dict[str, Any],
    session_id: str | None,
) -> None:
    """Forward a chunked-prefill progress event into the bus + registry.

    Payload shape (from generation.py):
      - phase: "started" | "chunk" | "completed"
      - tokens_total: prompt size in tokens
      - tokens_done: tokens processed so far (chunk + completed phases)
      - cached_tokens: tokens served from cache (no compute)
      - elapsed_s: wall time since prefill_started
      - prefill_tok_s: tokens/sec (computed at completion; cumulative on chunks)
      - prefill_compute_tok_s: new prefilled tokens / model prompt eval time
      - prefill_wall_tok_s: new prefilled tokens / wall prefill phase time
      - cumulative_prefill_tok_s/live_prefill_tok_s: chunk-phase display rates
      - chunk_size: most recent chunk size (chunk phase only)
      - chunk_elapsed_s/chunk_prefill_tok_s: most recent chunk timing
      - cache_hit: whether the bank served any prefix (completed)
      - started_s: monotonic timestamp at prefill start

    We enrich with request_id + session_id and a derived live tok/s so the
    dashboard doesn't have to recompute on every event.
    """

    dashboard = getattr(state, "dashboard", None)
    if dashboard is None:
        return
    try:
        enriched = dict(payload)
        enriched["request_id"] = request_id
        enriched["session_id"] = session_id
        # Live tok/s during chunked prefill (completion provides its own).
        if enriched.get("phase") == "chunk":
            tokens_done = float(enriched.get("tokens_done") or 0)
            elapsed = float(enriched.get("elapsed_s") or 0)
            if tokens_done > 0 and elapsed > 0:
                cumulative_tok_s = tokens_done / elapsed
                enriched.setdefault("prefill_tok_s", cumulative_tok_s)
                enriched.setdefault("cumulative_prefill_tok_s", cumulative_tok_s)
                enriched.setdefault("prefill_wall_tok_s", cumulative_tok_s)
                chunk_tok_s = enriched.get("chunk_prefill_tok_s")
                if isinstance(chunk_tok_s, (int, float)) and chunk_tok_s > 0:
                    enriched.setdefault("live_prefill_tok_s", float(chunk_tok_s))
                else:
                    enriched.setdefault("live_prefill_tok_s", cumulative_tok_s)
        # Update the in-flight handle so a poll of /v1/mtplx/snapshot
        # immediately reflects the current prefill state without waiting
        # for the SSE stream.
        if enriched.get("phase") == "completed":
            dashboard.in_flight.update_prefill(request_id, None)
        else:
            dashboard.in_flight.update_prefill(request_id, enriched)
        dashboard.bus.publish({"kind": "prefill", "when_s": time.time(), **enriched})
    except Exception as exc:
        _safe_stdout_print(f"[dashboard] publish_prefill suppressed error: {exc!r}")


def _dashboard_publish_progress(
    state: "ServerState",
    *,
    request_id: str,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Forward a streaming progress chunk into the dashboard bus + registry."""

    dashboard = getattr(state, "dashboard", None)
    if dashboard is None:
        return None
    decision_started_s = time.perf_counter()
    completion_tokens = int(payload.get("completion_tokens") or 0)
    try:
        should_publish = dashboard.progress_events.should_publish(
            request_id,
            completion_tokens=completion_tokens,
        )
        decision_time_s = time.perf_counter() - decision_started_s
        if not should_publish:
            dashboard.progress_events.record_overhead(
                request_id,
                published=False,
                completion_tokens=completion_tokens,
                decision_time_s=decision_time_s,
            )
            return None

        enriched = dict(payload)
        enriched["dashboard_progress_published"] = True
        registry_started_s = time.perf_counter()
        dashboard.in_flight.update_progress(request_id, enriched)
        registry_update_time_s = time.perf_counter() - registry_started_s
        decode_tok_s = payload.get("decode_tok_s")
        is_new_max = False
        rolling_update_time_s = 0.0
        if isinstance(decode_tok_s, (int, float)) and decode_tok_s > 0:
            rolling_started_s = time.perf_counter()
            is_new_max = dashboard.rolling.observe_progress(
                float(decode_tok_s),
                payload.get("session_id") or request_id,
            )
            rolling_update_time_s = time.perf_counter() - rolling_started_s
        bus_started_s = time.perf_counter()
        dashboard.bus.publish(
            {
                "kind": "progress",
                "when_s": time.time(),
                "request_id": request_id,
                "progress": dict(enriched),
            }
        )
        if is_new_max and isinstance(decode_tok_s, (int, float)):
            dashboard.bus.publish(
                {
                    "kind": "new_max_tps",
                    "when_s": time.time(),
                    "tok_s": float(decode_tok_s),
                    "session_id": payload.get("session_id"),
                }
            )
        bus_publish_time_s = time.perf_counter() - bus_started_s
        dashboard.progress_events.record_overhead(
            request_id,
            published=True,
            completion_tokens=completion_tokens,
            decision_time_s=decision_time_s,
            registry_update_time_s=registry_update_time_s,
            rolling_update_time_s=rolling_update_time_s,
            bus_publish_time_s=bus_publish_time_s,
        )
        enriched["dashboard_progress_decision_time_s"] = decision_time_s
        enriched["dashboard_progress_registry_update_time_s"] = (
            registry_update_time_s
        )
        enriched["dashboard_progress_rolling_update_time_s"] = (
            rolling_update_time_s
        )
        enriched["dashboard_progress_bus_publish_time_s"] = bus_publish_time_s
        return enriched
    except Exception as exc:
        _safe_stdout_print(f"[dashboard] publish_progress suppressed error: {exc!r}")
        return None


def _attach_dashboard_progress_stats(
    state: "ServerState",
    *,
    request_id: str,
    stats: dict[str, Any],
) -> None:
    dashboard = getattr(state, "dashboard", None)
    if dashboard is None:
        return
    try:
        stats.update(dashboard.progress_events.stats_for(request_id))
    except Exception as exc:
        _safe_stdout_print(f"[dashboard] progress stats suppressed error: {exc!r}")


# --- Mutable settings surface for the dashboard ----------------------------
#
# Profile, model, MTP loading, host, port and a handful of other knobs are
# immutable at startup (re-applying mid-run would need a model/runtime
# reload). The dashboard sidebar splits "mutable" from "restart required";
# this helper enforces the same split server-side.
DASHBOARD_MUTABLE_SETTINGS_KEYS: tuple[str, ...] = (
    "reasoning",
    "generation_mode",
    "depth",
    "temperature",
    "top_p",
    "top_k",
    "max_response_tokens",
    "stream_interval",
    "enable_thinking",
    "reasoning_parser",
    "reasoning_effort",
    "prefill_chunk_tokens",
    "draft_temperature",
    "draft_top_p",
    "draft_top_k",
)
DASHBOARD_READ_ONLY_SETTINGS_KEYS: tuple[str, ...] = (
    "architecture_id",
    "backend_id",
    "context_window",
    "context_window_policy",
    "depth_max",
    "draft_control",
    "kv_quant_policy",
    "model_controls",
    "model_family",
    "reasoning_policy",
    "sampling_defaults",
    "support_level",
    "tune_policy",
)
DASHBOARD_RESTART_REQUIRED_KEYS: tuple[str, ...] = (
    "profile",
    "model",
    "host",
    "port",
    "load_mtp",
    "verify_core",
    "verify_strategy",
    "scheduler_mode",
    "batching_preset",
    "max_active_requests",
    "decode_batch_max",
    "batch_wait_ms",
    "experimental_mtp_cohorts",
    "ram_session_cache_policy",
    "ram_session_block_prefix_restore",
    "ram_session_cache_max_entries",
    "ram_session_cache_max_size",
    "ram_session_cache_per_session_max_size",
    "ssd_session_cache",
    "ssd_session_cache_dir",
    "ssd_session_cache_max_size",
    "ssd_session_cache_min_prefix_tokens",
    "paged_kv_quantization",
    "context_window",
    "api_key",
)
DASHBOARD_API_VERSION = 1
DASHBOARD_SNAPSHOT_INTERVAL_DEFAULT_MS = 200
DASHBOARD_SNAPSHOT_INTERVAL_MIN_MS = 100
DASHBOARD_SNAPSHOT_INTERVAL_MAX_MS = 5000


def _dashboard_snapshot_interval_s(snapshot_interval_ms: int | None) -> float:
    """Coerce the dashboard/native-app snapshot cadence into safe bounds."""

    if snapshot_interval_ms is None:
        value = DASHBOARD_SNAPSHOT_INTERVAL_DEFAULT_MS
    else:
        value = int(snapshot_interval_ms)
    value = max(
        DASHBOARD_SNAPSHOT_INTERVAL_MIN_MS,
        min(DASHBOARD_SNAPSHOT_INTERVAL_MAX_MS, value),
    )
    return value / 1000.0


def _mtplx_app_capabilities() -> dict[str, Any]:
    """Return the stable backend contract consumed by native app shells."""

    from mtplx.dashboard import has_static_bundle

    endpoints = {
        "health": "/health",
        "metrics": "/metrics",
        "sessions": "/admin/sessions",
        "session_clear": "/admin/sessions/{session_id}/clear",
        "cache_clear": "/admin/cache/clear",
        "ssd_cache": "/admin/cache/ssd",
        "ssd_cache_archive": "/admin/cache/ssd/archive",
        "snapshot": "/v1/mtplx/snapshot",
        "metrics_stream": "/v1/mtplx/metrics/stream",
        "prefill_history": "/v1/mtplx/prefill_history",
        "settings": "/v1/mtplx/settings",
        "cancel": "/v1/mtplx/cancel/{request_id}",
        "dashboard": "/dashboard/",
        "app_capabilities": "/v1/mtplx/app/capabilities",
    }
    return {
        "ok": True,
        "name": "MTPLX App Backend",
        "api_version": DASHBOARD_API_VERSION,
        "endpoints": endpoints,
        "mutable_settings": list(DASHBOARD_MUTABLE_SETTINGS_KEYS),
        "restart_required_settings": list(DASHBOARD_RESTART_REQUIRED_KEYS),
        "snapshot_interval": {
            "default_ms": DASHBOARD_SNAPSHOT_INTERVAL_DEFAULT_MS,
            "min_ms": DASHBOARD_SNAPSHOT_INTERVAL_MIN_MS,
            "max_ms": DASHBOARD_SNAPSHOT_INTERVAL_MAX_MS,
            "native_default_ms": 500,
            "performance_lock_ms": 1000,
        },
        "features": {
            "sse_metrics": True,
            "request_cancel": True,
            "cache_clear": True,
            "ssd_session_cache": True,
            "ram_session_cache_controls": True,
            "paged_kv_quantization": True,
            "ssd_cache_archive": True,
            "session_clear": True,
            "prefill_history": True,
            "settings_mutation": True,
            "thermal_polling": True,
            "dashboard_static_bundle": has_static_bundle(),
            "scheduler_telemetry": True,
            "batching_policy": True,
            "cooperative_scheduler_core": True,
            "ar_batching_core": True,
            "ar_batching_live": True,
            "concurrent_mtp_ar_fallback": True,
            "mtp_cohorts_experimental": True,
            "mtp_cohorts_default_enabled": False,
            "startup_ownership": True,
            "strict_max_fan_startup": True,
            "thermal_actual_ramp_verification": True,
            "omlx_style_tool_parser": True,
            "parse_tools_at_completion": True,
            "early_tool_cancel_default": False,
            "hidden_generation_repair_default": False,
        },
        "openai_bridge": {
            "mode": "omlx_style",
            "preserve_reasoning_content": True,
            "preserve_tool_results": True,
            "parse_order": ["native", "qwen_xml", "namespaced", "bracket"],
            "malformed_tool_markup": "content_with_telemetry",
            "early_tool_cancel_default": False,
            "hidden_generation_repair_default": False,
            "legacy_bridge_default": False,
        },
        "scheduler": {
            "modes": list(SCHEDULER_MODE_CHOICES),
            "presets": list(BATCHING_PRESET_CHOICES),
            "v1_done_path": "path_a_solo_mtp_plus_cooperative_ar",
            "default_ux": "coding_agents",
            "default_policy": "solo_mtp_oracle",
            "solo_mtp_protected": True,
            "batched_mtp_required_for_v1": False,
        },
    }


def _json_env(name: str) -> dict[str, Any] | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _int_env(name: str) -> int | None:
    try:
        return int(str(os.environ.get(name) or "").strip())
    except (TypeError, ValueError):
        return None


def _startup_health_payload(state: "ServerState") -> dict[str, Any]:
    chat_template_report = getattr(state, "chat_template_report", {}) or {}
    tool_prompt_mode = _tool_prompt_mode_from_args(state.args)
    backend = _backend_descriptor(state)
    model_ref = str(getattr(state.args, "model", None) or state.model_id)
    model_context_window_max = getattr(state, "model_context_window_max", None)
    model_controls = model_controls_for_descriptor(
        backend,
        model_ref=model_ref,
        inspection=(
            {"model_context_window": int(model_context_window_max)}
            if model_context_window_max
            else None
        ),
    )
    return {
        "launch_id": getattr(state.args, "app_launch_id", None)
        or os.environ.get("MTPLX_APP_LAUNCH_ID"),
        "pid": os.getpid(),
        "app_parent_pid": _int_env("MTPLX_APP_PARENT_PID"),
        "started_at": getattr(state, "started_at_s", None),
        "model_id": state.model_id,
        "api_key_required": bool(getattr(state.args, "api_key", None)),
        "api_key_source": str(getattr(state.args, "api_key_source", "none") or "none"),
        "paged_kv_quantization": _effective_paged_kv_quantization(),
        "backend": backend.to_dict(),
        "model_controls": model_controls,
        "warmup": state.warmup_status,
        "tool_prompt_mode": tool_prompt_mode,
        "tool_contract_active": _tool_contract_active_for_mode(
            tools_active=True,
            tool_prompt_mode=tool_prompt_mode,
        ),
        "tool_contract_policy_version": _tool_prompt_policy_version(
            tools_active=True,
            tool_prompt_mode=tool_prompt_mode,
        ),
        "chat_template_profile": chat_template_report.get("profile")
        or getattr(state, "chat_template_profile", _CHAT_TEMPLATE_PROFILE_LOCAL),
        "chat_template_source": chat_template_report.get("source"),
        "chat_template_path": chat_template_report.get("path"),
        "chat_template_hash": getattr(state, "template_hash", None),
    }


def _server_fan_mode(state: Any) -> str:
    try:
        return normalize_fan_mode(
            getattr(state, "fan_mode", None)
            or getattr(getattr(state, "args", None), "fan_mode", None)
            or os.environ.get("MTPLX_FAN_MODE")
            or FAN_MODE_DEFAULT
        )
    except ValueError:
        return FAN_MODE_DEFAULT


def _smart_fan_status(state: Any) -> dict[str, Any]:
    controller = getattr(state, "smart_fans", None)
    if controller is None:
        return {
            "active": False,
            "active_count": 0,
            "commanded_max": False,
        }
    try:
        return dict(controller.status())
    except Exception as exc:
        return {
            "active": False,
            "active_count": 0,
            "commanded_max": False,
            "last_error": f"{type(exc).__name__}: {exc}",
        }


def _thermal_health_payload(*, fan_mode: str, smart_status: dict[str, Any] | None = None) -> dict[str, Any]:
    max_verified = _json_env("MTPLX_MAX_VERIFIED_JSON")
    fan_summary = (
        max_verified.get("after")
        if isinstance(max_verified, dict) and isinstance(max_verified.get("after"), dict)
        else None
    )
    actual_ramp_verified = os.environ.get("MTPLX_MAX_ACTUAL_RAMP_VERIFIED") == "1"
    smart = smart_status or {}
    smart_boost_active = fan_mode == FAN_MODE_SMART and bool(
        smart.get("commanded_max") or smart.get("active")
    )
    return {
        "max_requested": fan_mode == FAN_MODE_MAX
        or smart_boost_active
        or os.environ.get("MTPLX_MAX_REQUESTED") == "1",
        "max_verified": fan_mode == FAN_MODE_MAX and bool(max_verified is None or max_verified.get("ok", True)),
        "actual_ramp_verified": actual_ramp_verified,
        "smart": smart,
        "fan_summary": fan_summary,
        "verified_at": os.environ.get("MTPLX_MAX_VERIFIED_AT"),
        "verified": max_verified,
    }


def _coerce_setting(name: str, value: Any) -> Any:
    """Apply minimal type coercion so JSON ``"3"`` becomes int ``3``."""

    if name in {
        "depth",
        "top_k",
        "max_response_tokens",
        "stream_interval",
        "prefill_chunk_tokens",
        "draft_top_k",
    }:
        if value is None:
            return None
        try:
            coerced = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if name == "depth" and not 1 <= coerced <= 8:
            raise ValueError("depth must be between 1 and 8")
        if name == "prefill_chunk_tokens" and not 128 <= coerced <= 32768:
            raise ValueError("prefill_chunk_tokens must be between 128 and 32768")
        if name == "draft_top_k" and coerced < 0:
            raise ValueError("draft_top_k must be non-negative")
        return coerced
    if name == "generation_mode":
        text = str(value).strip().lower()
        if text not in {"mtp", "ar"}:
            raise ValueError("generation_mode must be 'mtp' or 'ar'")
        return text
    if name in {"temperature", "top_p", "draft_temperature", "draft_top_p"}:
        try:
            coerced = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number") from exc
        if name in {"top_p", "draft_top_p"} and coerced <= 0:
            raise ValueError(f"{name} must be positive")
        if name == "draft_temperature" and coerced < 0:
            raise ValueError("draft_temperature must be non-negative")
        return coerced
    if name == "enable_thinking":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if name == "reasoning_parser":
        text = str(value)
        if text not in {"qwen3", "step3p5", "gemma4", "none"}:
            raise ValueError(
                "reasoning_parser must be 'qwen3', 'step3p5', 'gemma4', or 'none'"
            )
        return text
    if name == "reasoning_effort":
        text = str(value).strip().lower()
        if text not in {"auto", "low", "medium", "high"}:
            raise ValueError(
                "reasoning_effort must be 'auto', 'low', 'medium', or 'high'"
            )
        return text
    return value


def _mtplx_apply_settings_payload(
    state: "ServerState", payload: dict[str, Any]
) -> dict[str, Any]:
    """Apply a partial settings update; respects restart-required keys.

    Reasoning is delegated to the existing ``_set_server_reasoning_mode``
    so behavior matches the prior ``/v1/mtplx/settings`` contract.
    """

    payload = {
        key: value
        for key, value in payload.items()
        if key not in DASHBOARD_READ_ONLY_SETTINGS_KEYS
    }
    restart_required = sorted(
        set(payload.keys()) & set(DASHBOARD_RESTART_REQUIRED_KEYS)
    )
    if restart_required:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "restart_required",
                "keys": restart_required,
                "message": (
                    "the following settings require a server restart: "
                    + ", ".join(restart_required)
                ),
            },
        )
    unknown = sorted(
        set(payload.keys())
        - set(DASHBOARD_MUTABLE_SETTINGS_KEYS)
        - set(DASHBOARD_READ_ONLY_SETTINGS_KEYS)
        - set(DASHBOARD_RESTART_REQUIRED_KEYS)
    )
    if unknown:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unknown_settings",
                "keys": unknown,
                "supported": list(DASHBOARD_MUTABLE_SETTINGS_KEYS),
            },
        )
    applied: dict[str, Any] = {}
    with state.lock:
        backend = _backend_descriptor(state)
        for key, raw in payload.items():
            if raw is None:
                continue
            if key == "reasoning":
                try:
                    _set_server_reasoning_mode(state, str(raw))
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                applied[key] = str(raw)
                continue
            try:
                value = _coerce_setting(key, raw)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if key == "depth":
                minimum = int(backend.draft_semantics.minimum)
                maximum = int(backend.draft_semantics.maximum)
                if not minimum <= int(value) <= maximum:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"depth must be between {minimum} and {maximum} "
                            "for the loaded model"
                        ),
                    )
            if (
                key == "generation_mode"
                and value == "mtp"
                and not bool(getattr(getattr(state, "runtime", None), "mtp_enabled", False))
            ):
                raise HTTPException(
                    status_code=400,
                    detail="generation_mode 'mtp' requires a runtime loaded with MTP",
                )
            setattr(state.args, key, value)
            applied[key] = value
        implicit_draft_updates: dict[str, Any] = {}
        if getattr(state, "draft_sampler", None) is not None:
            for source, target in (
                ("temperature", "draft_temperature"),
                ("top_p", "draft_top_p"),
                ("top_k", "draft_top_k"),
            ):
                if source in applied and target not in applied:
                    implicit_draft_updates[target] = applied[source]
        for key, value in implicit_draft_updates.items():
            setattr(state.args, key, value)
        if (
            {"draft_temperature", "draft_top_p", "draft_top_k"} & set(applied)
        ) or implicit_draft_updates:
            current = getattr(state, "draft_sampler", None)
            draft_temperature = getattr(state.args, "draft_temperature", None)
            draft_top_p = getattr(state.args, "draft_top_p", None)
            draft_top_k = getattr(state.args, "draft_top_k", None)
            if draft_temperature is None:
                draft_temperature = (
                    getattr(current, "temperature", None)
                    if current is not None
                    else getattr(state.args, "temperature", 0.6)
                )
            if draft_top_p is None:
                draft_top_p = (
                    getattr(current, "top_p", None)
                    if current is not None
                    else getattr(state.args, "top_p", 0.95)
                )
            if draft_top_k is None:
                draft_top_k = (
                    getattr(current, "top_k", None)
                    if current is not None
                    else getattr(state.args, "top_k", 20)
                )
            state.draft_sampler = SamplerConfig(
                temperature=float(draft_temperature),
                top_p=float(draft_top_p),
                top_k=int(draft_top_k),
            )
    return applied


def _env_bool_setting(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _effective_ram_session_cache_settings() -> dict[str, Any]:
    entries_raw = os.environ.get("MTPLX_SESSION_BANK_MAX_ENTRIES")
    max_bytes = os.environ.get("MTPLX_SESSION_BANK_MAX_BYTES") or "8G"
    per_session_bytes = os.environ.get("MTPLX_SESSION_BANK_PER_SESSION_BYTES") or "4G"
    block_prefix_restore = _env_bool_setting(
        "MTPLX_SESSION_BLOCK_PREFIX_RESTORE",
        default=True,
    )
    try:
        entries = max(1, int(entries_raw)) if entries_raw is not None else 4
    except ValueError:
        entries = 4
    if (
        entries_raw is None
        and "MTPLX_SESSION_BANK_MAX_BYTES" not in os.environ
        and "MTPLX_SESSION_BANK_PER_SESSION_BYTES" not in os.environ
        and "MTPLX_SESSION_BLOCK_PREFIX_RESTORE" not in os.environ
    ):
        policy = "target-default"
    elif entries <= 1 and max_bytes.upper() == "1G" and not block_prefix_restore:
        policy = "minimal"
    else:
        policy = "bounded"
    return {
        "ram_session_cache_policy": policy,
        "ram_session_block_prefix_restore": block_prefix_restore,
        "ram_session_cache_max_entries": entries,
        "ram_session_cache_max_size": max_bytes,
        "ram_session_cache_per_session_max_size": per_session_bytes,
    }


def _effective_paged_kv_quantization() -> str:
    raw = (
        os.environ.get("MTPLX_VLLM_METAL_PAGED_KV_QUANT")
        or os.environ.get("MTPLX_PAGED_KV_QUANT")
        or ""
    )
    return str(normalize_paged_kv_quantization(raw))


def _mtplx_current_settings(state: "ServerState") -> dict[str, Any]:
    """Read the live mutable settings for the dashboard."""

    args = state.args
    backend = _backend_descriptor(state)
    model_ref = str(getattr(args, "model", None) or getattr(state, "model_id", None) or "")
    model_context_window_max = getattr(state, "model_context_window_max", None)
    model_controls = model_controls_for_descriptor(
        backend,
        model_ref=model_ref,
        inspection=(
            {"model_context_window": int(model_context_window_max)}
            if model_context_window_max
            else None
        ),
    )
    reasoning = getattr(args, "reasoning", None)
    if reasoning not in {"auto", "on", "off"}:
        reasoning = "on" if bool(getattr(args, "enable_thinking", True)) else "off"
    tool_prompt_mode = _tool_prompt_mode_from_args(args)
    return {
        "reasoning": reasoning,
        "tool_prompt_mode": tool_prompt_mode,
        "tool_contract_active": _tool_contract_active_for_mode(
            tools_active=True,
            tool_prompt_mode=tool_prompt_mode,
        ),
        "tool_contract_policy_version": _tool_prompt_policy_version(
            tools_active=True,
            tool_prompt_mode=tool_prompt_mode,
        ),
        "chat_template_profile": getattr(
            state,
            "chat_template_profile",
            _CHAT_TEMPLATE_PROFILE_LOCAL,
        ),
        "chat_template_hash": getattr(state, "template_hash", None),
        "generation_mode": str(getattr(args, "generation_mode", "mtp") or "mtp"),
        "depth": int(getattr(args, "depth", 3) or 3),
        "depth_max": int(backend.draft_semantics.maximum),
        "draft_control": backend.draft_semantics.to_dict(),
        "backend_id": backend.backend_id,
        "architecture_id": backend.architecture_id,
        "model_family": model_controls["model_family"],
        "support_level": model_controls["support_level"],
        "model_controls": model_controls,
        "reasoning_policy": model_controls["reasoning"],
        "kv_quant_policy": model_controls["kv_quant"],
        "tune_policy": model_controls["tune"],
        "context_window_policy": model_controls["context_window"],
        "sampling_defaults": model_controls["sampling"],
        "temperature": float(getattr(args, "temperature", 0.6) or 0.0),
        "top_p": float(getattr(args, "top_p", 0.95) or 0.0),
        "top_k": int(getattr(args, "top_k", 20) or 0),
        "max_response_tokens": getattr(args, "max_response_tokens", None),
        "stream_interval": int(getattr(args, "stream_interval", 1) or 1),
        "enable_thinking": bool(getattr(args, "enable_thinking", False)),
        "api_key_required": bool(getattr(args, "api_key", None)),
        "api_key_source": str(getattr(args, "api_key_source", "none") or "none"),
        "reasoning_parser": str(getattr(args, "reasoning_parser", "qwen3")),
        "reasoning_effort": str(getattr(args, "reasoning_effort", "auto") or "auto"),
        "draft_temperature": (
            float(getattr(state.draft_sampler, "temperature"))
            if getattr(state, "draft_sampler", None) is not None
            else None
        ),
        "draft_top_p": (
            float(getattr(state.draft_sampler, "top_p"))
            if getattr(state, "draft_sampler", None) is not None
            else None
        ),
        "draft_top_k": (
            int(getattr(state.draft_sampler, "top_k"))
            if getattr(state, "draft_sampler", None) is not None
            else None
        ),
        "prefill_chunk_tokens": int(
            getattr(args, "prefill_chunk_tokens", None) or 2048
        ),
        "ssd_session_cache": str(getattr(args, "ssd_session_cache", "off") or "off"),
        "ssd_session_cache_dir": str(
            getattr(args, "ssd_session_cache_dir", "~/.mtplx/session-bank")
        ),
        "ssd_session_cache_max_size": str(
            getattr(args, "ssd_session_cache_max_size", "100GB")
        ),
        "ssd_session_cache_min_prefix_tokens": int(
            getattr(args, "ssd_session_cache_min_prefix_tokens", 512) or 512
        ),
        "paged_kv_quantization": _effective_paged_kv_quantization(),
        "restart_required_settings": [
            "model",
            "backend_id",
            "context_window",
            "paged_kv_quantization",
            "ssd_session_cache",
        ],
        **_effective_ram_session_cache_settings(),
    }


def _scheduler_config_from_args(args: Any) -> BatchSchedulerConfig:
    mode = str(getattr(args, "scheduler_mode", "serial") or "serial")
    if mode not in SCHEDULER_MODE_CHOICES:
        mode = "serial"
    preset = str(getattr(args, "batching_preset", "latency") or "latency")
    if preset not in BATCHING_PRESET_CHOICES:
        preset = "latency"
    try:
        return BatchSchedulerConfig.from_values(
            mode=mode,
            preset=preset,
            max_active_requests=getattr(args, "max_active_requests", None),
            decode_batch_max=getattr(args, "decode_batch_max", None),
            batch_wait_ms=getattr(args, "batch_wait_ms", None),
            prefill_chunk_tokens=getattr(args, "prefill_chunk_tokens", None),
            experimental_mtp_cohorts=bool(
                getattr(args, "experimental_mtp_cohorts", False)
            ),
        )
    except ValueError:
        return BatchSchedulerConfig.from_values(mode="serial", preset="latency")


def _scheduler_policy_label(config: BatchSchedulerConfig) -> str:
    if (
        config.mode
        in {SchedulerMode.AR_BATCH, SchedulerMode.MTP_COHORT_EXPERIMENTAL}
        and config.preset == SchedulerPreset.AGENT
    ):
        return "open_code_fair"
    if config.mode == SchedulerMode.AR_BATCH:
        return "fair_ar_batch"
    if config.mode == SchedulerMode.MTP_COHORT_EXPERIMENTAL:
        return "experimental_mtp_cohort"
    if config.mode == SchedulerMode.COOPERATIVE:
        return "cooperative"
    return "solo_mtp_oracle"


def _mtplx_scheduler_state(state: "ServerState") -> dict[str, Any]:
    config = _scheduler_config_from_args(state.args)
    scheduler = getattr(state, "model_scheduler", None)
    scheduler_stats: dict[str, Any] = {}
    if scheduler is not None and hasattr(scheduler, "stats"):
        try:
            scheduler_stats = dict(scheduler.stats())
        except Exception as exc:
            scheduler_stats = {"error": str(exc)}
    ar_batch_stats: dict[str, Any] = {}
    ar_batch_service = getattr(state, "ar_batch_service", None)
    if ar_batch_service is not None and hasattr(ar_batch_service, "snapshot"):
        try:
            ar_batch_stats = dict(ar_batch_service.snapshot())
        except Exception as exc:
            ar_batch_stats = {"error": str(exc)}
    try:
        active_requests = int(state.dashboard.in_flight.count())
    except Exception:
        active_requests = 0
    if not active_requests and hasattr(state, "foreground_count"):
        try:
            active_requests = int(state.foreground_count())
        except Exception:
            active_requests = 0
    mtp_available = bool(
        getattr(getattr(state, "runtime", None), "mtp_enabled", False)
        and str(getattr(state.args, "generation_mode", "mtp")) == "mtp"
    )
    if active_requests <= 1 and mtp_available:
        active_lane = "solo_mtp"
        mtp_disabled_reason = None
    elif active_requests > 1 and mtp_available:
        active_lane = (
            "ar_batch"
            if config.mode
            in {SchedulerMode.AR_BATCH, SchedulerMode.MTP_COHORT_EXPERIMENTAL}
            else "cooperative_ar"
        )
        mtp_disabled_reason = "batch_size_gt_1"
    elif config.mode == SchedulerMode.AR_BATCH:
        active_lane = "ar_batch"
        mtp_disabled_reason = "generation_mode_ar"
    else:
        active_lane = "serial_ar" if not mtp_available else "serial_mtp"
        mtp_disabled_reason = None if mtp_available else "generation_mode_ar"
    return {
        "config": config.to_dict(),
        "mode": config.mode.value,
        "preset": config.preset.value,
        "scheduler_policy": _scheduler_policy_label(config),
        "active_lane": active_lane,
        "active_requests": active_requests,
        "mtp_available": mtp_available,
        "mtp_disabled_reason": mtp_disabled_reason,
        "path": "path_a",
        "path_a": {
            "solo_mtp_protected": True,
            "concurrent_strategy": "cooperative_ar_batch",
            "batched_mtp_required_for_v1": False,
        },
        "path_b": {
            "experimental_mtp_cohorts": bool(config.experimental_mtp_cohorts),
            "default_enabled": False,
        },
        "telemetry": scheduler_stats,
        "ar_batch": ar_batch_stats,
    }


def _mtplx_dashboard_snapshot(state: "ServerState") -> dict[str, Any]:
    """Aggregate the dashboard snapshot served by SSE + the polling endpoint."""

    dashboard = state.dashboard
    bank_dict: dict[str, Any]
    sessions_dict: dict[str, Any]
    try:
        sessions_dict = state.sessions.list_sessions()
        bank_dict = sessions_dict.get("session_bank") or {}
    except Exception as exc:
        sessions_dict = {
            "sessions": [],
            "count": 0,
            "session_bank": {},
            "error": str(exc),
        }
        bank_dict = {}
    return {
        "ts": time.time(),
        "model_id": state.model_id,
        "profile": state.profile.to_dict()
        if hasattr(state.profile, "to_dict")
        else {"name": getattr(state.profile, "name", "unknown")},
        "context_window": state.context_window,
        "active_requests": state.dashboard.in_flight.count(),
        "in_flight": dashboard.in_flight.snapshot(),
        "latest": state.last_metrics[-1] if state.last_metrics else None,
        "recent": state.last_metrics[-32:],
        "rolling": dashboard.rolling.snapshot(),
        "lifetime": dashboard.lifetime.snapshot(),
        "sessions": sessions_dict,
        "session_bank": bank_dict,
        "mem": _mlx_memory_stats_live(),
        "thermal": dashboard.last_thermal,
        "thermal_when_s": dashboard.last_thermal_when_s,
        "settings": _mtplx_current_settings(state),
        "scheduler": _mtplx_scheduler_state(state),
        "machine": _machine_info(),
        "uptime_s": dashboard.lifetime.snapshot()["uptime_s"],
    }


async def _thermal_poll_loop(state: "ServerState", *, interval_s: float = 1.0) -> None:
    """Optional background sampler that publishes fan snapshots to the bus.

    Disabled by default because ``thermal.fan_summary()`` shells out to
    ``thermalforge status`` and adds ~10-30 ms of subprocess churn per
    poll. Enabled by ``--enable-thermal-poll``.
    """

    from mtplx.thermal import fan_summary

    while True:
        try:
            snapshot = await asyncio.to_thread(fan_summary)
        except asyncio.CancelledError:
            raise
        except Exception:
            snapshot = {"ok": False, "fans": [], "error": "fan_summary_failed"}
        state.dashboard.last_thermal = snapshot
        state.dashboard.last_thermal_when_s = time.time()
        state.dashboard.bus.publish(
            {"kind": "thermal", "thermal": snapshot, "when_s": time.time()}
        )
        try:
            await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            raise


def _stream_progress_payload(
    *,
    completion_tokens: int,
    decode_started_s: float | None,
    now_s: float,
) -> dict[str, Any]:
    decode_elapsed_s = (
        max(0.0, float(now_s) - float(decode_started_s))
        if decode_started_s is not None
        else 0.0
    )
    decode_tok_s = (
        float(completion_tokens) / decode_elapsed_s
        if completion_tokens > 0 and decode_elapsed_s > 0.0
        else None
    )
    return {
        "completion_tokens": int(completion_tokens),
        "decode_elapsed_s": decode_elapsed_s,
        "decode_tok_s": decode_tok_s,
    }


def _stream_heartbeat_payload(
    *,
    completion_tokens: int,
    stream_started_s: float,
    last_token_s: float | None,
    now_s: float,
) -> dict[str, Any]:
    last_activity_s = last_token_s if last_token_s is not None else stream_started_s
    return {
        "heartbeat": True,
        "phase": "generating",
        "completion_tokens": int(completion_tokens),
        "elapsed_s": max(0.0, float(now_s) - float(stream_started_s)),
        "seconds_since_last_token": max(0.0, float(now_s) - float(last_activity_s)),
    }


@contextmanager
def _temporary_env(updates: dict[str, str | None]):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _dynamic_paged_kv_initial_new_token_budget(
    max_new_tokens: int,
) -> tuple[int, int | None]:
    raw = (
        (os.environ.get("MTPLX_DYNAMIC_PAGED_KV_MAX_INITIAL_NEW_TOKENS") or "16384")
        .strip()
        .lower()
    )
    requested = max(0, int(max_new_tokens))
    if raw in {"0", "off", "false", "no", "none", "unlimited"}:
        return requested, None
    try:
        cap = max(1, int(raw))
    except (TypeError, ValueError):
        cap = 16384
    return min(requested, cap), cap


def _dynamic_paged_kv_reservation(
    *,
    prompt_tokens: int,
    max_new_tokens: int,
    mtp_depth: int,
) -> dict[str, Any]:
    requested_new = max(0, int(max_new_tokens))
    reserved_new, cap = _dynamic_paged_kv_initial_new_token_budget(requested_new)
    reserved_tokens = (
        max(0, int(prompt_tokens)) + max(0, int(reserved_new)) + max(0, int(mtp_depth))
    )
    return {
        "env": {"MTPLX_DYNAMIC_PAGED_KV_TOKENS": str(reserved_tokens)},
        "requested_new_tokens": int(requested_new),
        "reserved_new_tokens": int(reserved_new),
        "initial_new_token_cap": cap,
        "reservation_capped": bool(reserved_new < requested_new),
        "reserved_total_tokens": int(reserved_tokens),
    }


def _dynamic_paged_kv_env(
    *,
    prompt_tokens: int,
    max_new_tokens: int,
    mtp_depth: int,
) -> dict[str, str]:
    return _dynamic_paged_kv_reservation(
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        mtp_depth=mtp_depth,
    )["env"]


def _session_keep_live_refs_for_request(
    *,
    session_source: str | None,
    session_id: str | None,
    tool_names: list[str] | tuple[str, ...] | None = None,
) -> bool:
    if os.environ.get(
        "MTPLX_SESSIONBANK_LIVE_REFS_FOR_IMPLICIT_SESSIONS", ""
    ).strip().lower() in {"1", "true", "yes", "on"}:
        return True
    source = str(session_source or "")
    if source.startswith("header.") or source.startswith("metadata."):
        return True
    if source in {"user", "chat_id", "conversation_id"}:
        return True
    if _anonymous_coding_agent_tool_request(tool_names):
        return True
    # Anonymous auto sessions are useful as cloneable prefix snapshots, but
    # retaining their live paged-KV containers preserves the full request
    # capacity. A benchmark with 65k max_tokens and many one-off prompts can
    # otherwise pin huge unused buffers for no reuse benefit.
    return bool(
        session_id and not session_id.startswith("anon-") and source == "longest_prefix"
    )


def _commit_prompt_prefix_for_request(
    state: Any,
    *,
    prompt_ids: list[int],
    tools_active: bool,
) -> bool:
    if tools_active:
        return True
    if not prompt_ids:
        return False
    tier = getattr(state, "session_bank_cold_tier", None)
    if tier is None or not bool(getattr(tier, "enabled", False)):
        return False
    min_prefix_tokens = int(
        getattr(tier, "min_prefix_tokens", 512) or 512
    )
    return len(prompt_ids) >= max(512, min_prefix_tokens)


def _anonymous_coding_agent_tool_request(
    tool_names: list[str] | tuple[str, ...] | None,
) -> bool:
    names = {
        str(name).strip().lower() for name in (tool_names or []) if str(name).strip()
    }
    if not names:
        return False
    coding_agent_tools = {
        "bash",
        "edit",
        "glob",
        "grep",
        "ls",
        "multi_edit",
        "patch",
        "read",
        "str_replace_editor",
        "task",
        "todowrite",
        "webfetch",
        "write",
    }
    return bool(names & coding_agent_tools)


def _tool_call_ids_from_messages(messages: list[ChatMessage]) -> set[str]:
    ids: set[str] = set()
    for message in messages:
        if str(message.role).lower() != "assistant" or not message.tool_calls:
            continue
        for tool_call in message.tool_calls:
            call_id = str(tool_call.get("id") or "").strip()
            if call_id:
                ids.add(call_id)
    return ids


def _tool_result_ids_from_messages(messages: list[ChatMessage]) -> set[str]:
    ids: set[str] = set()
    for message in messages:
        if str(message.role).lower() != "tool":
            continue
        tool_call_id = str(
            message.tool_call_id
            or _message_extra(message, "tool_call_id")
            or ""
        ).strip()
        if tool_call_id:
            ids.add(tool_call_id)
    return ids


def _live_frontier_miss_reason_for_request(
    *,
    messages: list[ChatMessage],
    cache_miss_reason: str | None,
    session_source: str | None,
    session_keep_live_ref: bool,
) -> str | None:
    """Translate a cache miss into the reason an agent frontier could not resume."""

    assistant_tool_ids = _tool_call_ids_from_messages(messages)
    tool_result_ids = _tool_result_ids_from_messages(messages)
    return _live_frontier_miss_reason_from_counts(
        assistant_tool_call_count=len(assistant_tool_ids),
        tool_result_count=len(tool_result_ids),
        unknown_tool_result_count=len(tool_result_ids - assistant_tool_ids),
        cache_miss_reason=cache_miss_reason,
        session_source=session_source,
        session_keep_live_ref=session_keep_live_ref,
    )


def _live_frontier_miss_reason_from_counts(
    *,
    assistant_tool_call_count: int,
    tool_result_count: int,
    unknown_tool_result_count: int,
    cache_miss_reason: str | None,
    session_source: str | None,
    session_keep_live_ref: bool,
) -> str | None:
    """Translate frontier counters plus a cache miss into a user-facing reason."""

    if tool_result_count <= 0:
        return "miss_no_tool_result"
    if assistant_tool_call_count <= 0:
        return "miss_no_assistant_tool_frontier"
    if unknown_tool_result_count > 0:
        return "miss_unknown_tool_id"
    if cache_miss_reason is None:
        if not session_keep_live_ref:
            return "miss_live_frontier_not_armed"
        return None

    reason = str(cache_miss_reason or "").strip().lower()
    reason_map = {
        "template_mismatch": "miss_template_changed",
        "policy_mismatch": "miss_policy_changed",
        "model_mismatch": "miss_model_changed",
        "evicted": "miss_cache_evicted",
        "snapshot_desync": "miss_snapshot_desync",
        "no_snapshot_coverage": "miss_live_frontier_consumed_or_missing",
        "prefix_divergence_at_token": "miss_prompt_prefix_changed",
        "new_session": "miss_wrong_session_or_no_prior_frontier",
    }
    if reason in reason_map:
        return reason_map[reason]
    source = str(session_source or "")
    if not session_keep_live_ref:
        return "miss_live_frontier_not_armed"
    if source in {"", "new", "implicit_hash"}:
        return "miss_wrong_session_or_no_prior_frontier"
    return f"miss_{reason}" if reason else "miss_unknown"


def _clear_mlx_cache_after_request(
    state: Any,
    *,
    reason: str,
) -> dict[str, Any]:
    raw = (os.environ.get("MTPLX_CLEAR_CACHE_AFTER_REQUEST") or "auto").strip().lower()
    if raw in {"0", "false", "no", "off", "never"}:
        return {"cleared": False, "reason": "disabled"}
    lock = getattr(state, "lock", None)
    acquired = False
    if lock is not None and hasattr(lock, "acquire"):
        acquired = bool(lock.acquire(blocking=False))
        if not acquired:
            return {"cleared": False, "reason": "model_lock_busy", "trigger": reason}
    try:
        try:
            import mlx.core as mx
        except Exception as exc:
            return {
                "cleared": False,
                "reason": "mlx_unavailable",
                "trigger": reason,
                "error": repr(exc),
            }
        synchronize = getattr(mx, "synchronize", None)
        if callable(synchronize):
            synchronize()
        clear_cache = getattr(mx, "clear_cache", None)
        if not callable(clear_cache):
            return {
                "cleared": False,
                "reason": "clear_cache_unavailable",
                "trigger": reason,
            }
        clear_cache()
        return {"cleared": True, "reason": reason}
    except Exception as exc:
        return {
            "cleared": False,
            "reason": "clear_cache_error",
            "trigger": reason,
            "error": repr(exc),
        }
    finally:
        if acquired:
            lock.release()


def _auto_clear_mlx_cache_after_completed_request(
    state: Any,
    *,
    session_id: str | None,
    request_observability: dict[str, Any] | None,
) -> dict[str, Any] | None:
    raw = (os.environ.get("MTPLX_CLEAR_CACHE_AFTER_REQUEST") or "auto").strip().lower()
    if raw in {"0", "false", "no", "off", "never"}:
        return None
    request_observability = request_observability or {}
    client = str(
        request_observability.get("request_client_hint")
        or request_observability.get("request_client_label")
        or ""
    ).lower()
    if raw in {"1", "true", "yes", "always"}:
        reason = "after_request_forced"
    elif raw == "auto":
        if client != "aime" or session_id is not None:
            return None
        reason = "aime_stateless_question"
    elif raw == "aime":
        if client != "aime":
            return None
        reason = "aime_configured"
    else:
        return None
    return _clear_mlx_cache_after_request(state, reason=reason)


def _attach_skipped_postcommit_cleanup(
    state: Any,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    if (
        snapshot.get("mode") == "async_skipped"
        and snapshot.get("reason") == "stop_token_boundary_mismatch"
    ):
        snapshot["mlx_cache_cleanup"] = _clear_mlx_cache_after_request(
            state,
            reason="stop_token_boundary_mismatch_postcommit_skipped",
        )
    return snapshot


def _mlx_allocator_public_stats() -> dict[str, int]:
    try:
        import mlx.core as mx
    except Exception:
        return {}
    stats: dict[str, int] = {}
    for name, attr in (
        ("active_memory_bytes", "get_active_memory"),
        ("peak_memory_bytes", "get_peak_memory"),
        ("cache_memory_bytes", "get_cache_memory"),
    ):
        fn = getattr(mx, attr, None)
        if callable(fn):
            try:
                stats[name] = int(fn())
            except Exception:
                pass
    return stats


def _generation_truth_stats(
    state: "ServerState", effective_mode: str
) -> dict[str, Any]:
    load_mtp = bool(getattr(state.args, "load_mtp", True))
    runtime_mtp_enabled = bool(getattr(state.runtime, "mtp_enabled", False))
    draft_head = getattr(state, "draft_lm_head", None)
    draft_head_installed = (
        bool(draft_head.get("installed", "draft_only" in draft_head))
        if isinstance(draft_head, dict)
        else bool(draft_head)
    )
    if effective_mode == "mtp" and load_mtp and runtime_mtp_enabled:
        benchmark_mode = "mtplx_mtp_loaded_mtp_decode"
    elif effective_mode == "ar" and load_mtp and runtime_mtp_enabled:
        benchmark_mode = "mtplx_mtp_loaded_target_ar"
    elif effective_mode == "ar" and not load_mtp and not runtime_mtp_enabled:
        benchmark_mode = "mtplx_stock_ar_unloaded"
    else:
        benchmark_mode = "mtplx_unclassified"
    return {
        "benchmark_mode": benchmark_mode,
        "load_mtp": load_mtp,
        "runtime_mtp_enabled": runtime_mtp_enabled,
        "draft_head_installed": draft_head_installed,
        "profile": getattr(getattr(state, "profile", None), "name", None),
    }


PUBLIC_MTPLX_STATS_KEYS = (
    "mode",
    "profile",
    "benchmark_mode",
    "load_mtp",
    "runtime_mtp_enabled",
    "draft_head_installed",
    "ar_return_hidden",
    "forward_ar_hidden_calls",
    "forward_ar_plain_calls",
    "mtp_forward_calls",
    "make_mtp_cache_calls",
    "update_mtp_cache_calls",
    "mtp_history_append_calls",
    "full_logits_tokens_emitted",
    "final_logits_tokens_emitted",
    "logits_tokens_emitted",
    "prefill_chunk_size",
    "prefill_chunks",
    "paged_kv_capacity_tokens",
    "paged_kv_num_blocks",
    "paged_active_array_calls",
    "paged_active_array_time_s",
    "paged_turboquant",
    "paged_turboquant_k_quant",
    "paged_turboquant_v_quant",
    "paged_turboquant_attention_calls",
    "paged_kv_quant",
    "paged_kv_quant_mode",
    "paged_kv_quant_attention_calls",
    "paged_kv_quant_dequant_calls",
    "paged_kv_quant_dequant_time_s",
    "paged_kv_quant_dequant_tokens",
    "paged_gqa_sdpa_calls",
    "paged_gqa_sdpa_calls_by_route",
    "paged_gqa_sdpa_calls_by_phase",
    "paged_gqa_sdpa_route_misses_by_phase_reason",
    "paged_gqa_sdpa_route_misses_by_q_len",
    "paged_gqa_sdpa_last_route_miss",
    "attention_dense_fallback_calls",
    "prefill_dense_fallback_calls",
    "decode_dense_fallback_calls",
    "ar_dense_fallback_calls",
    "postcommit_dense_fallback_calls",
    "paged_attention_bailouts_by_phase_reason",
    "paged_attention_large_q_path",
    "large_q_split_sdpa_fallback_calls",
    "large_q_split_sdpa_fallback_calls_by_phase",
    "prefill_large_q_split_sdpa_fallback_calls",
    "decode_large_q_split_sdpa_fallback_calls",
    "partitioned_paged_calls",
    "partitioned_paged_calls_by_phase",
    "prefill_partitioned_paged_calls",
    "decode_partitioned_paged_calls",
    "sessionbank_snapshot_bytes",
    "sessionbank_skipped_oversized_snapshot",
    "hardware_acceleration_eligible",
    "hardware_acceleration_confirmed",
    "prefill_route",
    "generation_mode",
    "generated_tokens",
    "prompt_tokens",
    "completion_tokens",
    "elapsed_s",
    "tok_s",
    "end_to_end_tok_s",
    "finish_reason",
    "stop_sequence_hit",
    "stop_sequence_matched",
    "prompt_eval_time_s",
    "cache_restore_time_s",
    "prompt_target_prefill_time_s",
    "prompt_mtp_history_time_s",
    "prompt_target_prefill_tok_s",
    "prompt_mtp_history_tok_s",
    "prompt_tps",
    "prefill_tok_s",
    "prefill_compute_tok_s",
    "prefill_wall_tok_s",
    "ttft_s",
    "decode_elapsed_s",
    "request_elapsed_s",
    "request_tok_s",
    "decode_tok_s",
    "sliding_decode_tok_s_first_32",
    "sliding_decode_tok_s_first_64",
    "sliding_decode_tok_s_first_128",
    "sliding_decode_tok_s_first_256",
    "sliding_decode_tok_s_last_32",
    "sliding_decode_tok_s_last_64",
    "sliding_decode_tok_s_last_128",
    "sliding_decode_tok_s_last_256",
    "accepted_drafts",
    "rejected_drafts",
    "drafted_tokens",
    "verify_calls",
    "accepted_by_depth",
    "drafted_by_depth",
    "mean_accept_probability_by_depth",
    "correction_tokens",
    "bonus_tokens",
    "verify_time_s",
    "draft_time_s",
    "accept_time_s",
    "repair_time_s",
    # Verify-cycle decomposition for the dashboard's waterfall chart.
    # All exist on `GenerationStats` but were not part of the public envelope
    # before the dashboard work. Additive for existing clients (OpenWebUI,
    # Pi, hippo, opencode) — they tolerate extras and ignore unknown keys.
    "target_forward_time_s",
    "verify_forward_time_s",
    "verify_eval_time_s",
    "verify_logits_eval_time_s",
    "verify_hidden_eval_time_s",
    "verify_target_distribution_time_s",
    "target_distribution_materialized_rows",
    "target_distribution_materialized_windows",
    "target_distribution_share",
    "lazy_bonus_verify_calls",
    "lazy_bonus_commit_time_s",
    "verify_eval_unattributed_time_s",
    "mtp_history_policy",
    "mtp_history_window_tokens",
    "mtp_history_position_base",
    "snapshot_time_s",
    "commit_time_s",
    "capture_commit_time_s",
    "rollback_time_s",
    "graphbank",
    "repair_time_by_reject_depth_s",
    *MAINTENANCE_TIMING_STATS_KEYS,
    "session_cache_hit",
    "session_prompt_prefix_bank_commit",
    "cached_tokens",
    "new_prefill_tokens",
    "cache_source",
    "ssd_cache_hit",
    "ssd_cached_tokens",
    "ssd_restore_s",
    "ssd_suffix_tokens",
    "cache_miss_reason",
    "session_restore_mode",
    "ar_batch_shared_prefix_tokens",
    "ar_batch_shared_prefix_prefill_s",
    "ar_batch_shared_prefix_snapshot_s",
    "ar_batch_shared_prefix_bank_stored",
    "ar_batch_shared_prefix_bank_error",
    "ar_batch_prompt_prepare_s",
    "session_id",
    "context_len",
    "lock_wait_time_s",
    "request_max_tokens",
    "server_max_response_tokens",
    "effective_max_tokens",
    "decode_lease_tokens",
    "uncapped_response_requested",
    "uncapped_response_lease_tokens",
    "uncapped_response_lease_applied",
    "remaining_context_tokens",
    "server_cap_applied",
    "context_cap_applied",
    "server_elapsed_s",
    "server_tok_s",
    "server_seed",
    "server_attempts",
    "server_blank_retries",
    "server_blank_retry_suppressed",
    "scheduler_lane",
    "ar_batch_bypass_reason",
    "scheduler_mode",
    "batching_preset",
    "scheduler_policy",
    "request_id",
    "request_model",
    "served_model_id",
    "request_model_matches_served_model",
    "request_message_count",
    "request_message_roles",
    "request_message_chars",
    "request_effective_message_count",
    "request_effective_message_roles",
    "request_effective_message_chars",
    "request_metadata_keys",
    "request_client_hint",
    "request_client_label",
    "mtplx_control_owner",
    "client_controls_allowed",
    "client_control_fields_ignored",
    "client_sampler_fields_ignored",
    "request_enable_thinking",
    "request_enable_thinking_override",
    "request_reasoning_mode",
    "request_reasoning_parser",
    "request_temperature",
    "request_top_p",
    "request_top_k",
    "effective_temperature",
    "effective_top_p",
    "effective_top_k",
    "sampler_policy",
    "sampler_policy_reason",
    "sampler_policy_request_temperature",
    "sampler_policy_request_top_p",
    "sampler_policy_request_top_k",
    "sampler_policy_temperature",
    "sampler_policy_top_p",
    "sampler_policy_top_k",
    "mlx_cache_cleanup",
    "request_cancelled",
    "cancellation_reason",
    "stream_cancelled_by_client",
    "streamed_completion_tokens",
    "partial_decode_tok_s",
    "partial_request_tok_s",
    "dashboard_progress_published_events",
    "dashboard_progress_throttled_events",
    "dashboard_progress_last_completion_tokens",
    "dashboard_progress_decision_time_s",
    "dashboard_progress_registry_update_time_s",
    "dashboard_progress_rolling_update_time_s",
    "dashboard_progress_bus_publish_time_s",
    "cancellation_elapsed_s",
    "client_label",
    "queue_wait_s",
    "active_batch_size",
    "ar_batch_max_observed",
    "mtp_disabled_reason",
    "mtp_depth",
    "speculative_depth",
    "requested_mtp_depth",
    "requested_speculative_depth",
    "long_context_mtp_depth_policy",
    "opencode_short_context_depth_policy",
    "active_memory_bytes",
    "peak_memory_bytes",
    "cache_memory_bytes",
    "reasoning_reentries",
    "reasoning_tokens",
    "answer_tokens",
    "reasoning_completion_repair_attempted",
    "reasoning_completion_repair_succeeded",
    "reasoning_completion_repair_skipped",
    "reasoning_completion_repair_reason",
    "reasoning_completion_repair_first_completion_tokens",
    "reasoning_completion_repair_first_decode_tok_s",
    "reasoning_completion_repair_prompt_tokens",
    "reasoning_completion_repair_completion_tokens",
    "reasoning_completion_repair_finish_reason",
    "reasoning_completion_repair_decode_tok_s",
    "inspection_empty_retry_attempted",
    "inspection_empty_retry_succeeded",
    "inspection_empty_retry_reason",
    "inspection_empty_retry_first_completion_tokens",
    "inspection_empty_retry_first_decode_tok_s",
    "tool_fed_empty_retry_attempted",
    "tool_fed_empty_retry_succeeded",
    "tool_fed_empty_retry_reason",
    "tool_fed_empty_retry_first_completion_tokens",
    "tool_fed_empty_retry_first_decode_tok_s",
    "tool_fed_empty_retry_prompt_tokens",
    "tool_fed_empty_retry_completion_tokens",
    "tool_fed_empty_retry_finish_reason",
    "stalled_agent_retry_attempted",
    "stalled_agent_retry_succeeded",
    "stalled_agent_retry_reason",
    "stalled_agent_retry_first_completion_tokens",
    "stalled_agent_retry_first_decode_tok_s",
    "stalled_agent_retry_prompt_tokens",
    "stalled_agent_retry_completion_tokens",
    "stalled_agent_retry_finish_reason",
    "visible_reasoning_stripped",
    "nonstream_reasoning_content_routed",
    "tool_parse_success",
    "tool_parse_fallback",
    "tool_parse_fallback_reason",
    "tool_parse_fallback_kind",
    "tool_parser_dialect",
    "tool_stream_early_finish",
    "tool_call_count",
    "openai_bridge_mode",
    "tool_parser_source",
    "tool_parse_status",
    "tool_calls_emitted",
    "raw_tool_markup_suppressed",
    "legacy_bridge_used",
    "hidden_generation_repair_used",
    "early_tool_cancel_used",
    "openai_bridge_policy_version",
    "tool_prompt_mode",
    "tool_contract_policy_version",
    "tool_contract_active",
    "chat_template_profile",
    "chat_template_source",
    "chat_template_path",
    "chat_template_hash",
    "request_tool_count",
    "request_tool_choice",
    "request_tool_choice_forced",
    "request_tool_choice_forced_name",
    "request_filtered_tool_count",
    "request_tools_hidden_by_bridge",
    "request_read_only_inspection_force_answer",
    "request_read_only_inspection_tool_result_count",
    "request_read_only_inspection_force_answer_after_tools",
    "read_only_force_answer_contract_active",
    "request_pi_convergence_contract",
    "request_pi_convergence_tool_result_count",
    "request_pi_convergence_after_tools",
    "pi_convergence_contract_active",
    "opencode_simple_chat_contract_active",
    "opencode_prompt_contract_profile",
    "read_only_force_answer_retry_attempted",
    "read_only_force_answer_retry_succeeded",
    "read_only_force_answer_retry_reason",
    "read_only_force_answer_retry_first_completion_tokens",
    "read_only_force_answer_retry_first_decode_tok_s",
    "read_only_force_answer_retry_prompt_tokens",
    "read_only_force_answer_retry_completion_tokens",
    "read_only_force_answer_retry_finish_reason",
    "read_only_force_answer_buffered_stream",
    "read_only_force_answer_marker_stream_started",
    "read_only_force_answer_stream_marker_stripped_chars",
    "read_only_force_answer_visible_prefix_stripped_chars",
    "read_only_force_answer_visible_tokens",
    "request_session_source",
    "opencode_tool_history_cache_bypass",
    "opencode_tool_history_force_clone_restore",
    "opencode_tool_history_live_frontier_restore",
    "transcript_raw_message_chars",
    "transcript_canonical_message_chars",
    "transcript_canonicalized",
    "transcript_stripped_tool_preamble_messages",
    "transcript_stripped_tool_preamble_chars",
    "transcript_skipped_aborted_assistant_messages",
    "transcript_skipped_orphan_chitchat_assistant_messages",
    "transcript_dropped_simple_chitchat_history_messages",
    "transcript_dropped_simple_chitchat_history_chars",
    "transcript_replaced_simple_chitchat_system_messages",
    "transcript_replaced_simple_chitchat_system_chars",
    "transcript_injected_simple_chitchat_system_chars",
    "transcript_replaced_client_system_messages",
    "transcript_replaced_client_system_chars",
    "transcript_injected_client_system_chars",
    "transcript_replaced_initial_client_system_messages",
    "transcript_replaced_initial_client_system_chars",
    "transcript_injected_initial_client_system_chars",
    "transcript_skipped_repeated_assistant_messages",
    "transcript_skipped_stalled_agent_preamble_messages",
    "transcript_skipped_stalled_agent_preamble_chars",
    "transcript_collapsed_repeated_user_messages",
    "transcript_collapsed_repeated_user_chars",
    "transcript_dropped_duplicate_user_messages",
    "transcript_dropped_duplicate_user_chars",
    "transcript_merged_consecutive_user_messages",
    "transcript_merged_consecutive_user_chars",
    "transcript_compacted_repeated_timeout_tool_messages",
    "transcript_compacted_tool_result_messages",
    "transcript_compacted_tool_result_chars",
    "transcript_compacted_active_tool_result_messages",
    "transcript_compacted_active_tool_result_chars",
    "transcript_compacted_active_read_messages",
    "transcript_compacted_active_read_chars",
    "transcript_compacted_active_read_inspection_messages",
    "transcript_compacted_active_read_inspection_chars",
    "transcript_inspection_read_budget_candidate_messages",
    "transcript_inspection_read_budget_max_lines_per_file",
    "transcript_skipped_verbatim_tool_output_assistant_messages",
    "transcript_skipped_verbatim_tool_output_assistant_chars",
    "transcript_compacted_repeated_read_inspection_messages",
    "transcript_compacted_repeated_read_inspection_chars",
    "request_session_keep_live_ref",
    "request_session_keep_live_ref_reason",
    "request_session_bank_bypass",
    "request_session_prefix_diagnostic",
    "live_frontier_candidate",
    "live_frontier_result_turn",
    "live_frontier_policy",
    "live_frontier_hit",
    "live_frontier_restore_mode",
    "live_frontier_miss_reason",
    "live_frontier_assistant_tool_call_count",
    "live_frontier_tool_result_count",
    "live_frontier_unknown_tool_result_count",
    "dynamic_paged_kv",
    "session_prompt_prefix_commit",
)
PUBLIC_POSTCOMMIT_KEYS = (
    "stored",
    "mode",
    "reason",
    "prefix_len",
    "nbytes",
    "elapsed_s",
    "history_suffix_tokens",
    "cache_hit",
    "cached_tokens",
    "suffix_tokens",
    "cache_source",
    "ssd_cache_hit",
    "ssd_cached_tokens",
    "ssd_restore_s",
    "ssd_suffix_tokens",
    "cache_miss_reason",
    "mlx_cache_cleanup",
    "error",
)


def _public_mtplx_stats(generated: dict[str, Any]) -> dict[str, Any]:
    stats = generated.get("stats") or {}
    public = {key: stats[key] for key in PUBLIC_MTPLX_STATS_KEYS if key in stats}
    postcommit = stats.get("session_postcommit_snapshot")
    if isinstance(postcommit, dict):
        public["session_postcommit_snapshot"] = {
            key: postcommit[key] for key in PUBLIC_POSTCOMMIT_KEYS if key in postcommit
        }
    return _json_safe(public)


def _merge_final_bridge_stats_into_latest_metrics(
    state: ServerState,
    stats: dict[str, Any],
) -> None:
    if not getattr(state, "last_metrics", None):
        return
    latest = state.last_metrics[-1]
    for key in (
        "openai_bridge_mode",
        "tool_parser_source",
        "tool_parse_status",
        "tool_calls_emitted",
        "tool_parse_success",
        "tool_parse_fallback",
        "tool_parse_fallback_reason",
        "tool_parse_fallback_kind",
        "raw_tool_markup_suppressed",
        "tool_call_count",
        "read_only_force_answer_buffered_stream",
        "read_only_force_answer_visible_prefix_stripped_chars",
        "read_only_force_answer_visible_tokens",
        "session_prompt_prefix_commit",
        "session_postcommit_snapshot",
        "finish_reason",
    ):
        if key in stats:
            latest[key] = stats[key]


def _record_stream_cancellation_metric(
    state: ServerState,
    *,
    response_id: str,
    session_id: str | None,
    prompt_tokens: int,
    streamed_completion_tokens: int,
    stream_started_s: float,
    reason: str,
    request_observability: dict[str, Any],
    client_disconnected: bool,
) -> None:
    elapsed_s = max(0.0, time.perf_counter() - stream_started_s)
    streamed_tokens = int(streamed_completion_tokens)
    partial_tok_s = streamed_tokens / elapsed_s if elapsed_s > 0.0 else 0.0
    envelope: dict[str, Any] = {
        "mode": "stream",
        "request_id": response_id,
        "session_id": session_id,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": streamed_tokens,
        "generated_tokens": streamed_tokens,
        "request_elapsed_s": elapsed_s,
        "elapsed_s": elapsed_s,
        "server_elapsed_s": elapsed_s,
        "decode_elapsed_s": elapsed_s,
        "cancellation_elapsed_s": elapsed_s,
        "request_cancelled": True,
        "cancellation_reason": reason,
        "stream_cancelled_by_client": bool(client_disconnected),
        "streamed_completion_tokens": streamed_tokens,
        "decode_tok_s": partial_tok_s,
        "request_tok_s": partial_tok_s,
        "tok_s": partial_tok_s,
        "server_tok_s": partial_tok_s,
        "partial_decode_tok_s": partial_tok_s,
        "partial_request_tok_s": partial_tok_s,
        "session_cache_hit": False,
        "cache_miss_reason": None,
    }
    cleanup = _auto_clear_mlx_cache_after_completed_request(
        state,
        session_id=session_id,
        request_observability=request_observability,
    )
    if cleanup is not None:
        envelope["mlx_cache_cleanup"] = cleanup
    envelope.update(_mlx_allocator_public_stats())
    envelope.update(request_observability)
    state.last_metrics.append(_json_safe(envelope))
    state.last_metrics = state.last_metrics[-100:]
    state.last_request_at = time.time()
    state.requests_cancelled = int(getattr(state, "requests_cancelled", 0) or 0) + 1
    try:
        state.dashboard.bus.publish(
            {
                "kind": "cancelled",
                "when_s": time.time(),
                "envelope": dict(envelope),
            }
        )
    except BaseException:
        pass


def _request_observability(
    request: ChatCompletionRequest,
    *,
    headers: dict[str, str],
    metadata: dict[str, Any],
    session_source: str | None,
    request_generation_mode: str,
    request_depth: int,
) -> dict[str, Any]:
    declared_extra_keys = [
        key
        for key in (
            "max_completion_tokens",
            "stream_options",
            "response_format",
            "metadata",
            "parallel_tool_calls",
            "user",
        )
        if getattr(request, key, None) is not None
    ]
    model_extra_keys = sorted((getattr(request, "model_extra", None) or {}).keys())
    client_hint = _request_client_hint_from_headers(headers, metadata)
    user_texts = [
        _content_to_text(message.content)
        for message in request.messages
        if message.role == "user"
    ]
    candidate_headers = {
        key: value
        for key, value in headers.items()
        if key.lower()
        in {
            "x-mtplx-session-id",
            "x-openwebui-chat-id",
            "x-openwebui-user-id",
            "x-openwebui-task",
            "x-mtplx-request-id",
        }
    }
    return {
        "request_message_count": len(request.messages),
        "request_message_roles": [message.role for message in request.messages],
        "request_message_chars": [
            len(_content_to_text(message.content)) for message in request.messages
        ],
        "request_extra_keys": sorted(set(model_extra_keys + declared_extra_keys)),
        "request_metadata_keys": sorted(metadata.keys()),
        "request_client_hint": client_hint,
        "request_client_label": client_hint or "openai",
        "request_tool_count": len(request.tools or []),
        "request_tool_names": [
            name for tool in (request.tools or []) if (name := _tool_spec_name(tool))
        ],
        "request_tool_choice": _tool_choice_policy_signature(request.tool_choice),
        "request_tool_choice_forced": _tool_choice_forces_tools(request.tool_choice),
        "request_tool_choice_forced_name": _forced_tool_choice_name(
            request.tool_choice
        ),
        "request_session_source": session_source,
        "request_session_candidate_headers": candidate_headers,
        "request_generation_mode": request_generation_mode,
        "request_depth": int(request_depth),
        "request_last_user_preview": user_texts[-1][:180] if user_texts else None,
        "request_last_user_chars": len(user_texts[-1]) if user_texts else 0,
    }


def _last_user_text(messages: list[ChatMessage] | list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        role = (
            getattr(message, "role", None)
            if not isinstance(message, dict)
            else message.get("role")
        )
        if role != "user":
            continue
        content = (
            getattr(message, "content", "")
            if not isinstance(message, dict)
            else message.get("content", "")
        )
        return _content_to_text(content).strip()
    return ""


def _is_simple_chitchat_text(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    normalized = normalized.strip(" .!?")
    if not normalized or len(normalized) > 80:
        return False
    if re.search(
        r"\b("
        r"read|write|edit|fix|build|run|test|debug|implement|create|search|find|"
        r"open|inspect|refactor|package|file|folder|directory|code|script|npm|git"
        r")\b",
        normalized,
    ):
        return False
    if _is_compact_simple_chitchat_key(_simple_chitchat_compact_key(normalized)):
        return True
    return bool(
        re.fullmatch(
            r"(hi|hello|hey|yo|sup|howdy)"
            r"([, ]+(there|codex|mtplx|please|thanks|thank you))*"
            r"([, ]+how ((are|r) )?(you|u)( doing| today)?)?",
            normalized,
        )
        or re.fullmatch(
            r"(how ((are|r) )?(you|u)( doing| today)?|how'?s it going|what'?s up)",
            normalized,
        )
    )


def _opencode_default_sampler_override(
    *,
    messages: list[ChatMessage],
    tools_active: bool,
    request_temperature: float | None,
    request_top_p: float | None,
    request_top_k: int | None,
    request_observability: dict[str, Any],
    default_temperature: float,
    default_top_p: float,
    default_top_k: int,
) -> SamplerConfig | None:
    client_hint = str(request_observability.get("request_client_hint") or "").lower()
    if "opencode" not in client_hint:
        return None
    simple_chitchat = _is_simple_chitchat_text(_last_user_text(messages))
    opencode_default_sampler = (
        (request_temperature is None or abs(float(request_temperature) - 0.55) < 1e-9)
        and (request_top_p is None or abs(float(request_top_p) - 1.0) < 1e-9)
        and (
            request_top_k is None
            or int(request_top_k) == int(default_top_k)
        )
    )
    if not tools_active and not simple_chitchat:
        return None
    if not opencode_default_sampler:
        return None
    request_observability["sampler_policy"] = "opencode_default_sampler"
    request_observability["sampler_policy_reason"] = (
        "OpenCode sent its implicit default sampler; normalize target sampling "
        "to the launched MTPLX defaults"
    )
    request_observability["sampler_policy_request_temperature"] = request_temperature
    request_observability["sampler_policy_request_top_p"] = request_top_p
    request_observability["sampler_policy_request_top_k"] = request_top_k
    temperature = float(default_temperature)
    top_p = float(default_top_p)
    top_k = int(default_top_k)
    request_observability["sampler_policy_temperature"] = temperature
    request_observability["sampler_policy_top_p"] = top_p
    request_observability["sampler_policy_top_k"] = top_k
    return SamplerConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )


def _opencode_default_draft_sampler_for_request(
    state: ServerState,
    request_observability: dict[str, Any],
) -> SamplerConfig | None:
    launched = getattr(state, "draft_sampler", None)
    if not isinstance(launched, SamplerConfig):
        return None
    request_observability["draft_sampler_policy"] = "launch_default"
    request_observability["draft_sampler_policy_reason"] = (
        "OpenCode default target sampler normalized; keep the launched "
        "model-contract proposal sampler for speculative decoding"
    )
    request_observability["draft_sampler_policy_temperature"] = float(
        launched.temperature
    )
    request_observability["draft_sampler_policy_top_p"] = float(launched.top_p)
    request_observability["draft_sampler_policy_top_k"] = int(launched.top_k)
    return launched


def _policy_fingerprint(
    state: ServerState,
    *,
    thinking_enabled: bool,
    generation_mode: str | None = None,
    depth: int | None = None,
    tools_active: bool = False,
    tool_prompt_mode: str | None = None,
    tool_choice: Any = None,
    no_tools_contract_active: bool = False,
    read_only_force_answer_contract_active: bool = False,
    pi_convergence_contract_active: bool = False,
    simple_chat_contract_active: bool = False,
    opencode_prompt_contract_profile: str | None = None,
    cache_scope: str | None = None,
) -> str:
    effective_tool_prompt_mode = _normalize_tool_prompt_mode(
        tool_prompt_mode,
        default=_tool_prompt_mode_from_args(state.args),
    )
    effective_mode = _normalize_generation_mode(
        generation_mode,
        default=getattr(state.args, "generation_mode", "mtp"),
    )
    effective_depth = (
        0
        if effective_mode == "ar"
        else int(depth if depth is not None else getattr(state.args, "depth", 3))
    )
    adaptive = _adaptive_config(state.args, max_depth=effective_depth)
    proposal_cache = _proposal_cache_config(state.args)
    online_hidden = _online_hidden_config(state.args)
    parts = [
        f"template={state.template_hash}",
        f"thinking={int(bool(thinking_enabled))}",
        f"strip_reasoning={int(bool(state.args.strip_assistant_reasoning_history))}",
        f"openai_bridge={_OPENAI_BRIDGE_POLICY_VERSION}",
        f"tool_prompt_mode={effective_tool_prompt_mode}",
        "tool_contract="
        + _tool_prompt_policy_version_for_request(
            tools_active=tools_active,
            tool_prompt_mode=effective_tool_prompt_mode,
            no_tools_contract_active=no_tools_contract_active,
            read_only_force_answer_contract_active=read_only_force_answer_contract_active,
            pi_convergence_contract_active=pi_convergence_contract_active,
        ),
        f"tool_choice={_tool_choice_policy_signature(tool_choice)}",
        f"no_tools_contract={int(bool(no_tools_contract_active))}",
        "read_only_force_answer_contract="
        f"{int(bool(read_only_force_answer_contract_active))}",
        f"pi_convergence_contract={int(bool(pi_convergence_contract_active))}",
        f"simple_chat_contract={int(bool(simple_chat_contract_active))}",
        f"opencode_prompt_contract={opencode_prompt_contract_profile or 'none'}",
        f"generation_mode={effective_mode}",
        f"depth={effective_depth}",
        "hidden_variant=post_norm",
        "mtp_history_policy=committed",
        f"draft_head={state.draft_head_identity}",
        f"adaptive={json.dumps(adaptive, sort_keys=True, separators=(',', ':'))}",
        f"proposal_cache={json.dumps(proposal_cache, sort_keys=True, separators=(',', ':'))}",
        f"online_hidden={json.dumps(online_hidden, sort_keys=True, separators=(',', ':'))}",
    ]
    normalized_cache_scope = str(cache_scope or "").strip()
    if normalized_cache_scope and normalized_cache_scope != "stable":
        parts.append(f"cache_scope={normalized_cache_scope}")
    return ";".join(parts)


def _session_cache_scope_for_request(
    state: ServerState,
    *,
    headers: Mapping[str, str],
    metadata: Mapping[str, Any],
) -> str:
    client_hint = str(_request_client_hint_from_headers(headers, metadata) or "")
    if "opencode" not in client_hint:
        return "stable"
    launch_id = str(getattr(state.args, "app_launch_id", "") or "").strip()
    if not launch_id:
        launch_id = f"pid:{os.getpid()}"
    return f"opencode_process_cache:v1:{launch_id}"


def _is_opencode_client(
    *,
    headers: Mapping[str, str],
    metadata: Mapping[str, Any],
) -> bool:
    client_hint = str(_request_client_hint_from_headers(headers, metadata) or "")
    return "opencode" in client_hint


def _request_tool_prompt_mode_override(
    *,
    headers: Mapping[str, str],
    metadata: Mapping[str, Any],
) -> str | None:
    raw = None
    for header in _TOOL_PROMPT_MODE_REQUEST_HEADERS:
        raw = headers.get(header)
        if raw:
            break
    if raw is None:
        raw = metadata.get("tool_prompt_mode")
    if raw is None:
        return None
    try:
        return _normalize_tool_prompt_mode(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _agent_tool_contract_client_hint(
    *,
    headers: Mapping[str, str],
    metadata: Mapping[str, Any],
) -> str | None:
    client_hint = str(_request_client_hint_from_headers(headers, metadata) or "")
    for marker in _TOOL_CONTRACT_AGENT_CLIENT_HINTS:
        if marker in client_hint:
            return marker
    return None


def _tool_prompt_mode_for_request(
    args: argparse.Namespace,
    *,
    headers: Mapping[str, str],
    metadata: Mapping[str, Any],
    tools_active: bool,
) -> tuple[str, dict[str, Any]]:
    launch_mode = _tool_prompt_mode_from_args(args)
    requested_mode = _request_tool_prompt_mode_override(
        headers=headers,
        metadata=metadata,
    )
    client_hint = _agent_tool_contract_client_hint(
        headers=headers,
        metadata=metadata,
    )
    if requested_mode is not None:
        mode = requested_mode
        source = "request"
    elif tools_active and client_hint == "opencode":
        mode = _TOOL_PROMPT_MODE_COMPACT
        source = "client:opencode"
    elif tools_active and client_hint is not None:
        mode = _TOOL_PROMPT_MODE_HYBRID
        source = f"client:{client_hint}"
    else:
        mode = launch_mode
        source = "launch"
    return mode, {
        "tool_prompt_mode_launch": launch_mode,
        "tool_prompt_mode_source": source,
        "tool_prompt_mode_client": client_hint,
        "tool_prompt_mode_request_override": requested_mode,
        "tool_prompt_mode_client_repaired": bool(
            mode != launch_mode and requested_mode is None
        ),
    }


def _should_bypass_session_cache_for_opencode_tool_history(
    *,
    headers: Mapping[str, str],
    metadata: Mapping[str, Any],
    tool_result_history_present: bool,
) -> bool:
    if not _should_force_clone_session_cache_for_opencode_tool_history(
        headers=headers,
        metadata=metadata,
        tool_result_history_present=tool_result_history_present,
    ):
        return False
    return _env_bool_setting(
        "MTPLX_OPENCODE_TOOL_HISTORY_SESSIONBANK_BYPASS",
        default=False,
    )


def _opencode_tool_history_live_frontier_enabled() -> bool:
    return _env_bool_setting(
        "MTPLX_OPENCODE_TOOL_HISTORY_LIVE_FRONTIER",
        default=False,
    )


def _should_force_clone_session_cache_for_opencode_tool_history(
    *,
    headers: Mapping[str, str],
    metadata: Mapping[str, Any],
    tool_result_history_present: bool,
) -> bool:
    if not tool_result_history_present:
        return False
    return _is_opencode_client(headers=headers, metadata=metadata)


def _opencode_tool_history_restore_policy(
    *,
    headers: Mapping[str, str],
    metadata: Mapping[str, Any],
    tool_result_history_present: bool,
) -> dict[str, bool]:
    eligible = _should_force_clone_session_cache_for_opencode_tool_history(
        headers=headers,
        metadata=metadata,
        tool_result_history_present=tool_result_history_present,
    )
    cache_bypass = eligible and _should_bypass_session_cache_for_opencode_tool_history(
        headers=headers,
        metadata=metadata,
        tool_result_history_present=tool_result_history_present,
    )
    live_frontier_restore = (
        eligible
        and not cache_bypass
        and _opencode_tool_history_live_frontier_enabled()
    )
    return {
        "eligible": bool(eligible),
        "cache_bypass": bool(cache_bypass),
        "live_frontier_restore": bool(live_frontier_restore),
        "force_clone_restore": bool(
            eligible and not cache_bypass and not live_frontier_restore
        ),
    }


def _proposal_cache_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "online_correction_cache": bool(args.online_correction_cache),
        "online_correction_cache_min_depth": int(
            args.online_correction_cache_min_depth
        ),
        "online_correction_cache_key": str(args.online_correction_cache_key),
        "prompt_correction_cache": bool(args.prompt_correction_cache),
        "prompt_correction_cache_min_depth": int(
            args.prompt_correction_cache_min_depth
        ),
    }


def _online_hidden_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "alpha": float(getattr(args, "online_hidden_corrector_alpha", 0.0)),
        "decay": float(getattr(args, "online_hidden_corrector_decay", 0.8)),
        "warmup": int(getattr(args, "online_hidden_corrector_warmup", 1)),
        "max_feed_depth": getattr(args, "online_hidden_corrector_max_feed_depth", None),
        "key": str(getattr(args, "online_hidden_corrector_key", "global")),
    }


def _bridge_policy_observability(
    *,
    tools_active: bool,
    tool_prompt_mode: str,
    no_tools_contract_active: bool = False,
    read_only_force_answer_contract_active: bool = False,
    pi_convergence_contract_active: bool = False,
) -> dict[str, Any]:
    effective_tool_prompt_mode = _normalize_tool_prompt_mode(tool_prompt_mode)
    return {
        "openai_bridge_policy_version": _OPENAI_BRIDGE_POLICY_VERSION,
        "tool_prompt_mode": effective_tool_prompt_mode,
        "tool_contract_policy_version": _tool_prompt_policy_version_for_request(
            tools_active=tools_active,
            tool_prompt_mode=effective_tool_prompt_mode,
            no_tools_contract_active=no_tools_contract_active,
            read_only_force_answer_contract_active=read_only_force_answer_contract_active,
            pi_convergence_contract_active=pi_convergence_contract_active,
        ),
        "tool_contract_active": _tool_contract_active_for_mode(
            tools_active=tools_active,
            tool_prompt_mode=effective_tool_prompt_mode,
        ),
        "no_tools_contract_active": bool(no_tools_contract_active),
        "read_only_force_answer_contract_active": bool(
            read_only_force_answer_contract_active
        ),
        "pi_convergence_contract_active": bool(pi_convergence_contract_active),
    }


def _adaptive_config(
    args: argparse.Namespace,
    *,
    max_depth: int | None = None,
) -> dict[str, Any]:
    policy = str(getattr(args, "adaptive_policy", "none") or "none")
    if policy == "none":
        return {"policy": "none"}
    effective_max_depth = max(
        1,
        int(max_depth if max_depth is not None else getattr(args, "depth", 3)),
    )
    configured_min_depth = max(1, int(args.adaptive_min_depth))
    effective_min_depth = min(configured_min_depth, effective_max_depth)
    config: dict[str, Any] = {
        "policy": policy,
        "max_depth": effective_max_depth,
        "min_depth": effective_min_depth,
    }
    if effective_min_depth != configured_min_depth:
        config["configured_min_depth"] = configured_min_depth
    if policy == "streak":
        config.update(
            {
                "start_depth": int(args.adaptive_start_depth),
                "increase_after": int(args.adaptive_increase_after),
                "decrease_after": int(args.adaptive_decrease_after),
            }
        )
    elif policy == "expected_value":
        configured_base_depth = max(1, int(args.adaptive_ev_base_depth))
        effective_base_depth = max(
            effective_min_depth,
            min(configured_base_depth, effective_max_depth),
        )
        config.update(
            {
                "base_depth": effective_base_depth,
                "accept_priors": [float(v) for v in args.adaptive_ev_accept_priors],
                "draft_cost_s": float(args.adaptive_ev_draft_cost_s),
                "extra_verify_cost_s": float(args.adaptive_ev_extra_verify_cost_s),
                "baseline_tok_s": float(args.adaptive_ev_baseline_tok_s),
                "safety_margin": float(args.adaptive_ev_safety_margin),
                "margin_center": float(args.adaptive_ev_margin_center),
                "margin_scale": float(args.adaptive_ev_margin_scale),
                "confidence_weight": float(args.adaptive_ev_confidence_weight),
                "min_extra_accept_probability": float(
                    args.adaptive_ev_min_extra_accept_probability
                ),
                "warmup_full_depth_cycles": int(
                    args.adaptive_ev_warmup_full_depth_cycles
                ),
                "exploration_interval": int(args.adaptive_ev_exploration_interval),
            }
        )
        if effective_base_depth != configured_base_depth:
            config["configured_base_depth"] = configured_base_depth
    return config


def _make_adaptive_policy(
    args: argparse.Namespace,
    *,
    max_depth: int | None = None,
) -> AdaptiveDepthPolicy | ExpectedValueDepthPolicy | None:
    policy = str(getattr(args, "adaptive_policy", "none") or "none")
    if policy == "none":
        return None
    effective_max_depth = max(
        1,
        int(max_depth if max_depth is not None else getattr(args, "depth", 3)),
    )
    effective_min_depth = min(
        max(1, int(args.adaptive_min_depth)),
        effective_max_depth,
    )
    if policy == "streak":
        return AdaptiveDepthPolicy(
            max_depth=effective_max_depth,
            min_depth=effective_min_depth,
            start_depth=int(args.adaptive_start_depth),
            increase_after=int(args.adaptive_increase_after),
            decrease_after=int(args.adaptive_decrease_after),
        )
    if policy == "expected_value":
        effective_base_depth = max(
            effective_min_depth,
            min(max(1, int(args.adaptive_ev_base_depth)), effective_max_depth),
        )
        return ExpectedValueDepthPolicy(
            max_depth=effective_max_depth,
            min_depth=effective_min_depth,
            base_depth=effective_base_depth,
            accept_priors=tuple(float(v) for v in args.adaptive_ev_accept_priors),
            draft_cost_s=float(args.adaptive_ev_draft_cost_s),
            extra_verify_cost_s=float(args.adaptive_ev_extra_verify_cost_s),
            baseline_tok_s=float(args.adaptive_ev_baseline_tok_s),
            safety_margin=float(args.adaptive_ev_safety_margin),
            margin_center=float(args.adaptive_ev_margin_center),
            margin_scale=float(args.adaptive_ev_margin_scale),
            confidence_weight=float(args.adaptive_ev_confidence_weight),
            min_extra_accept_probability=float(
                args.adaptive_ev_min_extra_accept_probability
            ),
            warmup_full_depth_cycles=int(
                args.adaptive_ev_warmup_full_depth_cycles
            ),
            exploration_interval=int(args.adaptive_ev_exploration_interval),
        )
    raise ValueError(f"unknown adaptive policy: {policy}")


def _store_retokenized_history_snapshot(
    state: ServerState,
    *,
    session_id: str | None,
    messages: list[ChatMessage],
    assistant_content: str,
    assistant_tool_calls: list[dict[str, Any]] | None = None,
    thinking_enabled: bool,
    policy_fingerprint: str,
    acquire_model_lock_blocking: bool = True,
    tool_specs: list[dict[str, Any]] | None = None,
    session: Any | None = None,
    expected_session_revision: int | None = None,
    abort_check: Callable[[], bool] | None = None,
    abort_reason: Callable[[], str] | None = None,
    pending_record: Any | None = None,
    keep_live_ref: bool = True,
    tool_prompt_mode: str | None = None,
    strip_tool_call_preamble_text: bool = False,
) -> dict[str, Any]:
    if session_id is None:
        return {"stored": False, "reason": "no_session_id"}

    def _abort_requested() -> bool:
        if abort_check is None:
            return False
        try:
            return bool(abort_check())
        except BaseException:
            return True

    def _abort_reason() -> str:
        if abort_reason is not None:
            try:
                reason = str(abort_reason() or "")
                if reason:
                    return reason
            except BaseException:
                pass
        return "foreground_preempted_postcommit"

    history_ids = _history_ids_for_postcommit(
        state,
        messages=messages,
        assistant_content=assistant_content,
        assistant_tool_calls=assistant_tool_calls,
        thinking_enabled=thinking_enabled,
        tool_specs=tool_specs,
        tool_prompt_mode=tool_prompt_mode,
        strip_tool_call_preamble_text=strip_tool_call_preamble_text,
    )
    if not history_ids:
        return {"stored": False, "reason": "empty_boundary_prefix"}
    history_tokens = len(history_ids)
    if pending_record is not None and hasattr(pending_record, "update_token_count"):
        try:
            pending_record.update_token_count(history_tokens)
        except BaseException:
            pass
    best_prefix_len = 0
    best_prefix_nbytes = 0
    try:
        best_prefix = state.sessions.bank.longest_prefix(history_ids)
        if best_prefix is not None:
            best_prefix_len = int(getattr(best_prefix, "prefix_len", 0) or 0)
            best_prefix_nbytes = int(getattr(best_prefix, "nbytes", 0) or 0)
    except BaseException:
        best_prefix_len = 0
        best_prefix_nbytes = 0
    prefix_probe = {
        "best_prefix_len": int(best_prefix_len),
        "history_tokens": int(history_tokens),
        "suffix_tokens": max(0, int(history_tokens) - int(best_prefix_len)),
    }
    bank_budget = int(getattr(state.sessions.bank, "per_session_max_bytes", 0) or 0)
    estimated_nbytes = 0
    if best_prefix_len > 0 and best_prefix_nbytes > 0:
        # SessionBank snapshots scale roughly with prefix length. If the
        # previous committed boundary is already close to the per-session
        # cap, attempting to materialize a larger postcommit snapshot can
        # burn tens of seconds only to be rejected as oversized. Skip that
        # best-effort cache maintenance before touching MLX arrays; the
        # foreground user request can still reuse the existing shorter
        # prefix and prefill only the suffix.
        estimated_nbytes = int(
            (float(best_prefix_nbytes) * float(history_tokens) / float(best_prefix_len))
            * 1.03
        )
    if bank_budget > 0 and estimated_nbytes > bank_budget:
        return {
            "stored": False,
            "mode": "retokenized_history",
            "reason": "estimated_oversized_snapshot",
            "estimated_nbytes": int(estimated_nbytes),
            "budget": int(bank_budget),
            "best_prefix_nbytes": int(best_prefix_nbytes),
            **prefix_probe,
        }
    if _abort_requested():
        return {
            "stored": False,
            "mode": "aborted",
            "reason": _abort_reason(),
            **prefix_probe,
        }
    started = time.perf_counter()
    state.begin_foreground()
    acquired = state.lock.acquire(blocking=bool(acquire_model_lock_blocking))
    if not acquired:
        state.end_foreground()
        return {
            "stored": False,
            "mode": "retokenized_history",
            "reason": "model_lock_busy_before_retokenized_commit",
            "elapsed_s": time.perf_counter() - started,
            **prefix_probe,
        }
    # Pass `session_bank` (and the matching identity / policy fingerprints)
    # so the postcommit re-prefill reuses the longest matching prefix from
    # a prior turn instead of re-prefilling the full ~18K-token history
    # from scratch. On consecutive tool-calling turns this collapses
    # postcommit cost from ~27 s (full re-prefill) to ~1 s (suffix forward
    # only). The next foreground request, which the model_scheduler admits
    # only after this idle task completes, therefore queues ~1 s instead
    # of ~30 s behind the postcommit.
    #
    # The bank already contains an entry for the previous turn's history
    # (this same function stored it on the prior postcommit). The new
    # turn's history starts with that previous-turn prefix verbatim
    # (chat-template encoding is deterministic for the same messages and
    # tools), so `longest_prefix` matches and only the new user turn +
    # assistant turn need to be forward-AR'd.
    try:
        try:
            if _abort_requested():
                raise PostcommitAbort(_abort_reason())
            with attention_phase("postcommit"):
                prompt_state = restore_or_prefill_prompt_state(
                    state.runtime,
                    history_ids,
                    mtp_hidden_variant="post_norm",
                    mtp_history_policy="committed",
                    session_bank=state.sessions.bank,
                    template_hash=state.template_hash,
                    draft_head_identity=state.draft_head_identity,
                    policy_fingerprint=policy_fingerprint,
                    abort_check=abort_check,
                )
            if _abort_requested():
                raise PostcommitAbort(_abort_reason())
            mtp_snapshot = (
                snapshot_cache(prompt_state.committed_mtp_cache)
                if prompt_state.committed_mtp_cache is not None
                else None
            )
            if _abort_requested():
                raise PostcommitAbort(_abort_reason())
            entry = state.sessions.bank.put(
                runtime=state.runtime,
                token_ids=history_ids,
                cache=prompt_state.trunk_cache,
                logits=prompt_state.logits,
                hidden=prompt_state.hidden,
                hidden_variant="post_norm",
                keep_live_ref=bool(keep_live_ref),
                session_id=session_id,
                template_hash=state.template_hash,
                mtp_history_policy="committed",
                draft_head_identity=state.draft_head_identity,
                policy_fingerprint=policy_fingerprint,
                mtp_history_snapshot=mtp_snapshot,
                snapshot_epoch=len(history_ids),
                mtp_snapshot_epoch=len(history_ids)
                if mtp_snapshot is not None
                else None,
            )
        except PostcommitAbort:
            return {
                "stored": False,
                "mode": "aborted",
                "reason": _abort_reason(),
                "elapsed_s": time.perf_counter() - started,
                **prefix_probe,
            }
    finally:
        state.lock.release()
        state.end_foreground()
    if entry is None:
        return {
            "stored": False,
            "mode": "retokenized_history",
            "reason": "sessionbank_snapshot_skipped",
            "elapsed_s": time.perf_counter() - started,
            **prefix_probe,
        }
    session_commit: dict[str, Any] | None = None
    if session is not None:
        try:
            if _abort_requested():
                return {
                    "stored": False,
                    "mode": "aborted",
                    "reason": _abort_reason(),
                    "elapsed_s": time.perf_counter() - started,
                    **prefix_probe,
                }
            commit = session.commit_retokenized_prefix(
                token_ids=history_ids,
                expected_revision=expected_session_revision,
                nbytes=int(entry.nbytes),
            )
            session_commit = {
                "committed": bool(commit.committed),
                "reason": commit.reason,
                "prefix_len": int(commit.prefix_len),
            }
        except BaseException as exc:
            session_commit = {
                "committed": False,
                "reason": f"session_commit_error:{type(exc).__name__}",
                "prefix_len": int(getattr(session, "prefix_len", 0) or 0),
            }
    return {
        "stored": True,
        "mode": "retokenized_history",
        "prefix_len": entry.prefix_len,
        "nbytes": entry.nbytes,
        "elapsed_s": time.perf_counter() - started,
        "token_hash": entry.token_hash,
        # Observability for the prefix-reuse shortcut: lets operators tell
        # at a glance (in the `[mtplx] idle async session postcommit ...`
        # log line) whether a given postcommit hit the warm-prefix path or
        # had to do a full re-prefill, and if it missed, why. Without
        # cache_miss_reason a regression where the shortcut stops firing
        # (e.g. policy mismatch, template mismatch, snapshot desync, or
        # genuine prefix divergence) is invisible in production logs - all
        # that surfaces is `elapsed_s` drifting back to ~30 s.
        **prefix_probe,
        "cache_hit": bool(getattr(prompt_state, "cache_hit", False)),
        "cached_tokens": int(getattr(prompt_state, "cached_tokens", 0) or 0),
        "suffix_tokens": int(getattr(prompt_state, "suffix_tokens", 0) or 0),
        "cache_source": getattr(prompt_state, "cache_source", "none"),
        "ssd_cache_hit": bool(getattr(prompt_state, "ssd_cache_hit", False)),
        "ssd_cached_tokens": int(getattr(prompt_state, "ssd_cached_tokens", 0) or 0),
        "ssd_restore_s": float(getattr(prompt_state, "ssd_restore_s", 0.0) or 0.0),
        "ssd_suffix_tokens": (
            int(getattr(prompt_state, "suffix_tokens", 0) or 0)
            if bool(getattr(prompt_state, "ssd_cache_hit", False))
            else 0
        ),
        "cache_miss_reason": getattr(prompt_state, "cache_miss_reason", None),
        "session_commit": session_commit,
    }


def _history_ids_for_postcommit(
    state: ServerState,
    *,
    messages: list[ChatMessage],
    assistant_content: str,
    assistant_tool_calls: list[dict[str, Any]] | None,
    thinking_enabled: bool,
    tool_specs: list[dict[str, Any]] | None = None,
    tool_prompt_mode: str | None = None,
    strip_tool_call_preamble_text: bool = False,
) -> list[int]:
    reasoning_effort = _reasoning_effort_for_state(
        state,
        thinking_enabled=thinking_enabled,
    )
    effective_tool_prompt_mode = _normalize_tool_prompt_mode(
        tool_prompt_mode,
        default=_tool_prompt_mode_from_args(state.args),
    )
    history_messages = list(messages) + [
        ChatMessage(
            role="assistant",
            content=assistant_content,
            tool_calls=assistant_tool_calls,
        ),
    ]
    if tool_specs:
        # The generation prompt may compact the current large read as an
        # active-read excerpt. Once the assistant response is appended, that
        # same tool result is historical context for the next OpenCode turn
        # and must use the older-tool digest shape the next request will build.
        canonicalization_messages = history_messages
        if not assistant_tool_calls:
            canonicalization_messages = [
                *history_messages,
                ChatMessage(role="user", content=_POSTCOMMIT_SENTINEL_CONTENT),
            ]
        history_messages, _stats = _canonicalize_agent_transcript(
            canonicalization_messages,
            tools_active=True,
            strip_tool_call_preamble_text=strip_tool_call_preamble_text,
        )
        if (
            not assistant_tool_calls
            and history_messages
            and str(history_messages[-1].role).lower() == "user"
            and _content_to_text(history_messages[-1].content)
            == _POSTCOMMIT_SENTINEL_CONTENT
        ):
            history_messages = history_messages[:-1]
    next_turn_prefix_ids = _postcommit_next_turn_prefix_ids(
        state.runtime.tokenizer,
        history_messages,
        enable_thinking=thinking_enabled,
        reasoning_effort=reasoning_effort,
        strip_assistant_reasoning_history=state.args.strip_assistant_reasoning_history,
        tools=tool_specs,
        assistant_tool_calls=assistant_tool_calls,
        tool_prompt_mode=effective_tool_prompt_mode,
    )
    if next_turn_prefix_ids:
        return next_turn_prefix_ids
    return _encode_messages(
        state.runtime.tokenizer,
        history_messages,
        enable_thinking=thinking_enabled,
        reasoning_effort=reasoning_effort,
        strip_assistant_reasoning_history=state.args.strip_assistant_reasoning_history,
        add_generation_prompt=False,
        tools=tool_specs,
        tool_prompt_mode=effective_tool_prompt_mode,
    )


def _generation_final_postcommit_compatibility(
    state: ServerState,
    *,
    prompt_ids: list[int],
    generated: dict[str, Any],
    messages: list[ChatMessage],
    assistant_content: str,
    assistant_tool_calls: list[dict[str, Any]] | None = None,
    thinking_enabled: bool,
    tool_specs: list[dict[str, Any]] | None = None,
    tool_prompt_mode: str | None = None,
    strip_tool_call_preamble_text: bool = False,
) -> dict[str, Any]:
    if assistant_tool_calls:
        return {
            "safe": False,
            "mode": "unsafe",
            "reason": "tool_call_history_rewrite",
        }
    if (
        _STATS_FOOTER_RE.search(assistant_content)
        or STATS_FOOTER_MARKER in assistant_content
    ):
        return {
            "safe": False,
            "mode": "unsafe",
            "reason": "stats_footer_in_assistant_history",
        }
    if _reasoning_parser_for_state(state) == "gemma4" and thinking_enabled:
        return {
            "safe": False,
            "mode": "unsafe",
            "reason": "gemma4_reasoning_history_retokenize",
        }
    final_state = generated.get("_final_state")
    if final_state is None:
        return {
            "safe": False,
            "mode": "unsafe",
            "reason": "missing_generation_final_state",
        }
    if not bool(getattr(final_state, "safe_to_commit", False)):
        return {
            "safe": False,
            "mode": "unsafe",
            "reason": "generation_final_state_unsafe",
        }
    generated_tokens = [int(token) for token in generated.get("tokens") or []]
    final_generated_tokens = [
        int(token)
        for token in getattr(
            final_state, "generated_token_ids", tuple(generated_tokens)
        )
    ]
    if final_generated_tokens != generated_tokens:
        return {
            "safe": False,
            "mode": "unsafe",
            "reason": "generated_token_mismatch",
        }
    final_token_ids = [int(token) for token in prompt_ids] + final_generated_tokens
    if not final_token_ids:
        return {"safe": False, "mode": "unsafe", "reason": "empty_generation_boundary"}
    history_ids = _history_ids_for_postcommit(
        state,
        messages=messages,
        assistant_content=assistant_content,
        assistant_tool_calls=assistant_tool_calls,
        thinking_enabled=thinking_enabled,
        tool_specs=tool_specs,
        tool_prompt_mode=tool_prompt_mode,
        strip_tool_call_preamble_text=strip_tool_call_preamble_text,
    )
    if history_ids == final_token_ids:
        return {
            "safe": True,
            "mode": "generation_final_exact",
            "reason": "token_identical",
            "token_ids": final_token_ids,
            "history_suffix_tokens": 0,
        }
    if (
        len(history_ids) >= len(final_token_ids)
        and history_ids[: len(final_token_ids)] == final_token_ids
    ):
        return {
            "safe": True,
            "mode": "generation_final_prefix",
            "reason": "generation_boundary_prefix_of_history",
            "token_ids": final_token_ids,
            "history_suffix_tokens": len(history_ids) - len(final_token_ids),
        }
    reason = "retokenized_history_mismatch"
    if bool(state.args.strip_assistant_reasoning_history) and thinking_enabled:
        reason = "reasoning_history_stripping_mismatch"
    elif str(generated.get("finish_reason") or "") == "stop":
        reason = "stop_token_boundary_mismatch"
    return {
        "safe": False,
        "mode": "unsafe",
        "reason": reason,
        "history_tokens": len(history_ids),
        "generation_boundary_tokens": len(final_token_ids),
    }


def _store_generation_final_history_snapshot(
    state: ServerState,
    *,
    session_id: str | None,
    prompt_ids: list[int],
    generated: dict[str, Any],
    messages: list[ChatMessage],
    assistant_content: str,
    assistant_tool_calls: list[dict[str, Any]] | None = None,
    thinking_enabled: bool,
    policy_fingerprint: str,
    tool_specs: list[dict[str, Any]] | None = None,
    keep_live_ref: bool = True,
    tool_prompt_mode: str | None = None,
    strip_tool_call_preamble_text: bool = False,
) -> dict[str, Any]:
    if session_id is None:
        return {"stored": False, "mode": "unsafe", "reason": "no_session_id"}
    started = time.perf_counter()
    compatibility = _generation_final_postcommit_compatibility(
        state,
        prompt_ids=prompt_ids,
        generated=generated,
        messages=messages,
        assistant_content=assistant_content,
        assistant_tool_calls=assistant_tool_calls,
        thinking_enabled=thinking_enabled,
        tool_specs=tool_specs,
        tool_prompt_mode=tool_prompt_mode,
        strip_tool_call_preamble_text=strip_tool_call_preamble_text,
    )
    if not bool(compatibility.get("safe")):
        return {
            "stored": False,
            "mode": compatibility.get("mode", "unsafe"),
            "reason": compatibility.get("reason", "unsafe_history"),
            "elapsed_s": time.perf_counter() - started,
        }
    final_state = generated["_final_state"]
    token_ids = [int(token) for token in compatibility["token_ids"]]
    acquired = state.lock.acquire(blocking=False)
    if not acquired:
        return {
            "stored": False,
            "mode": "unsafe",
            "reason": "model_lock_busy_before_generation_final_commit",
            "elapsed_s": time.perf_counter() - started,
        }
    try:
        mtp_snapshot = (
            snapshot_cache(final_state.final_committed_mtp_cache)
            if final_state.final_committed_mtp_cache is not None
            else None
        )
        entry = state.sessions.bank.put(
            runtime=state.runtime,
            token_ids=token_ids,
            cache=final_state.final_trunk_cache,
            logits=final_state.final_logits,
            hidden=final_state.final_hidden,
            hidden_variant="post_norm",
            keep_live_ref=bool(keep_live_ref),
            session_id=session_id,
            template_hash=state.template_hash,
            mtp_history_policy="committed",
            draft_head_identity=state.draft_head_identity,
            policy_fingerprint=policy_fingerprint,
            mtp_history_snapshot=mtp_snapshot,
            snapshot_epoch=len(token_ids),
            mtp_snapshot_epoch=len(token_ids) if mtp_snapshot is not None else None,
            extra_state=getattr(final_state, "extra_state", None),
        )
    finally:
        state.lock.release()
    if entry is None:
        return {
            "stored": False,
            "mode": compatibility["mode"],
            "reason": "sessionbank_snapshot_skipped",
            "elapsed_s": time.perf_counter() - started,
        }
    return {
        "stored": True,
        "mode": compatibility["mode"],
        "reason": compatibility["reason"],
        "prefix_len": entry.prefix_len,
        "nbytes": entry.nbytes,
        "elapsed_s": time.perf_counter() - started,
        "history_suffix_tokens": int(compatibility.get("history_suffix_tokens") or 0),
        "token_hash": entry.token_hash,
    }


_IDLE_POSTCOMMIT_MAX_WAIT_S = 30.0
_IDLE_POSTCOMMIT_POLL_INTERVAL_S = 0.25


def _schedule_idle_postcommit_snapshot(
    state: ServerState,
    *,
    session_id: str | None,
    messages: list[ChatMessage],
    assistant_content: str,
    assistant_tool_calls: list[dict[str, Any]] | None = None,
    thinking_enabled: bool,
    policy_fingerprint: str,
    unsafe_reason: str,
    tool_specs: list[dict[str, Any]] | None = None,
    session: Any | None = None,
    expected_session_revision: int | None = None,
    keep_live_ref: bool = True,
    tool_prompt_mode: str | None = None,
    strip_tool_call_preamble_text: bool = False,
) -> dict[str, Any]:
    """Schedule a background SessionBank commit for a response the
    generation-final compatibility check rejected as unsafe (most commonly
    because the response contained tool_calls, which carries a non-trivial
    next-turn retokenization risk).

    The retokenized-history path canonicalises tool_call responses into the
    exact prefix the next request will send, so the commit is safe - it just
    must not run on the request thread (would extend stream latency). This
    function dispatches that work to the low-priority model scheduler lane.
    The scheduler admits it only after foreground work drains; the job then
    rechecks that no newer foreground is queued and that the session did not
    advance before it builds a new cache.
    """
    pending = {
        "stored": False,
        "mode": "async_pending",
        "reason": unsafe_reason,
    }
    abort_event = Event()
    pending_record_holder: dict[str, Any] = {}

    def _log(outcome: dict[str, Any]) -> None:
        record = pending_record_holder.get("record")
        if (
            session is not None
            and record is not None
            and hasattr(session, "finish_pending_postcommit")
        ):
            try:
                session.finish_pending_postcommit(record, outcome)
            except BaseException:
                pass
        elif record is not None and hasattr(record, "mark_finished"):
            try:
                record.mark_finished(outcome)
            except BaseException:
                pass
        if _server_console_enabled(state):
            return
        try:
            _safe_stdout_print(
                "[mtplx] idle async session postcommit "
                + json.dumps(
                    {
                        "session_id": session_id,
                        "unsafe_reason": unsafe_reason,
                        **outcome,
                    },
                    sort_keys=True,
                    default=str,
                )
            )
        except BaseException:
            # Logging must never bring down the executor; fail silently.
            pass

    def _observed_session_revision() -> int | None:
        observed = getattr(session, "revision", None) if session is not None else None
        if observed is None:
            return None
        try:
            return int(observed)
        except (TypeError, ValueError):
            return None

    def _stale_session_revision() -> bool:
        observed = _observed_session_revision()
        return (
            expected_session_revision is not None
            and observed is not None
            and int(observed) != int(expected_session_revision)
        )

    def _postcommit_abort_reason() -> str:
        if _stale_session_revision():
            return "stale_session_revision"
        if abort_event.is_set() or _foreground_model_work_pending(state):
            return "foreground_preempted_postcommit"
        return "postcommit_abort_requested"

    def _postcommit_abort_check() -> bool:
        return bool(
            abort_event.is_set()
            or _stale_session_revision()
            or _foreground_model_work_pending(state)
        )

    def async_postcommit() -> None:
        deadline = time.monotonic() + _IDLE_POSTCOMMIT_MAX_WAIT_S
        record = pending_record_holder.get("record")
        if record is not None and hasattr(record, "mark_started"):
            try:
                record.mark_started()
            except BaseException:
                pass
        try:
            while True:
                if _stale_session_revision():
                    observed_revision = _observed_session_revision()
                    _log(
                        {
                            "stored": False,
                            "mode": "abandoned_stale",
                            "reason": "stale_session_revision",
                            "expected_session_revision": int(expected_session_revision),
                            "observed_session_revision": int(observed_revision or -1),
                        }
                    )
                    return
                if abort_event.is_set() or _foreground_model_work_pending(state):
                    _log(
                        {
                            "stored": False,
                            "mode": "abandoned_foreground_busy",
                            "reason": _postcommit_abort_reason(),
                        }
                    )
                    return
                postcommit = _store_retokenized_history_snapshot(
                    state,
                    session_id=session_id,
                    messages=messages,
                    assistant_content=assistant_content,
                    assistant_tool_calls=assistant_tool_calls,
                    thinking_enabled=thinking_enabled,
                    policy_fingerprint=policy_fingerprint,
                    acquire_model_lock_blocking=False,
                    tool_specs=tool_specs,
                    session=session,
                    expected_session_revision=expected_session_revision,
                    abort_check=_postcommit_abort_check,
                    abort_reason=_postcommit_abort_reason,
                    pending_record=record,
                    keep_live_ref=bool(keep_live_ref),
                    tool_prompt_mode=tool_prompt_mode,
                    strip_tool_call_preamble_text=strip_tool_call_preamble_text,
                )
                if postcommit.get("stored"):
                    _log(postcommit)
                    return
                if (
                    postcommit.get("reason")
                    != "model_lock_busy_before_retokenized_commit"
                ):
                    _log(postcommit)
                    return
                if time.monotonic() >= deadline:
                    _log(
                        {
                            "stored": False,
                            "mode": "abandoned_foreground_busy",
                            "reason": "model_lock_busy_past_deadline",
                        }
                    )
                    return
                time.sleep(_IDLE_POSTCOMMIT_POLL_INTERVAL_S)
        except BaseException as exc:
            _log(
                {
                    "stored": False,
                    "mode": "async_error",
                    "reason": f"async_postcommit_raised:{type(exc).__name__}",
                }
            )

    future = _submit_idle_postcommit_model_work(
        state,
        async_postcommit,
        batch_key=f"postcommit:{session_id or 'stateless'}",
    )
    # Stash the future on the EngineSession so the next request in this
    # session can wait briefly for it before acquiring the session lock.
    # The wait is bounded and best-effort: if the postcommit raises or
    # times out the next request just falls through to a cold prefill.
    if session is not None:
        try:
            record = session.set_pending_postcommit(
                future,
                abort_event=abort_event,
                reason=unsafe_reason,
            )
            pending_record_holder["record"] = record
        except BaseException:
            # Telemetry plumbing must never break the request path.
            pass
    return pending


def _skipped_idle_postcommit_snapshot(
    *,
    state: ServerState | None = None,
    unsafe_reason: str,
    assistant_tool_calls: list[dict[str, Any]] | None = None,
    prompt_prefix_len: int | None = None,
) -> dict[str, Any] | None:
    backend_id = getattr(getattr(state, "backend_descriptor", None), "backend_id", "")
    if backend_id == GEMMA4_BACKEND:
        return {
            "stored": False,
            "mode": "skipped",
            "reason": "gemma4_retokenized_postcommit_unsupported",
            "unsafe_reason": unsafe_reason,
            "assistant_tool_calls": len(assistant_tool_calls or []),
            "prompt_prefix_len": int(prompt_prefix_len or 0),
        }
    del unsafe_reason, assistant_tool_calls, prompt_prefix_len
    return None


_UNCAPPED_RESPONSE_LEASE_DISABLED_VALUES = {
    "0",
    "off",
    "false",
    "no",
    "none",
    "unlimited",
}


def _uncapped_response_lease_tokens_from_env() -> int | None:
    raw_value = os.environ.get("MTPLX_UNCAPPED_RESPONSE_LEASE_TOKENS")
    if raw_value is None:
        return None
    raw = raw_value.strip().lower()
    if raw in _UNCAPPED_RESPONSE_LEASE_DISABLED_VALUES:
        return None
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return None


def _generation_params(
    state: ServerState,
    *,
    prompt_token_count: int,
    max_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
) -> tuple[int, SamplerConfig, dict[str, Any]]:
    remaining_context = max(1, int(state.context_window) - int(prompt_token_count))
    request_max_tokens = None if max_tokens is None else int(max_tokens)
    semantic_requested_max = (
        remaining_context if request_max_tokens is None else request_max_tokens
    )
    before_server_cap = semantic_requested_max
    server_max_response_tokens = state.args.max_response_tokens
    if state.args.max_response_tokens is not None:
        semantic_requested_max = min(
            semantic_requested_max, int(state.args.max_response_tokens)
        )
    after_server_cap = semantic_requested_max
    semantic_effective_max = max(1, min(after_server_cap, remaining_context))
    decode_lease_tokens = semantic_effective_max
    uncapped_response_requested = request_max_tokens is None
    uncapped_response_lease_tokens: int | None = None
    uncapped_response_lease_applied = False
    if uncapped_response_requested and server_max_response_tokens is None:
        uncapped_response_lease_tokens = _uncapped_response_lease_tokens_from_env()
        if uncapped_response_lease_tokens is not None:
            decode_lease_tokens = max(
                1, min(semantic_effective_max, uncapped_response_lease_tokens)
            )
            uncapped_response_lease_applied = decode_lease_tokens < semantic_effective_max
    sampler_temperature = (
        state.args.temperature if temperature is None else float(temperature)
    )
    sampler_top_p = state.args.top_p if top_p is None else float(top_p)
    sampler_top_k = state.args.top_k if top_k is None else int(top_k)
    sampler = SamplerConfig(
        temperature=sampler_temperature,
        top_p=sampler_top_p,
        top_k=sampler_top_k,
    )
    return (
        decode_lease_tokens,
        sampler,
        {
            "request_max_tokens": request_max_tokens,
            "server_max_response_tokens": (
                None
                if server_max_response_tokens is None
                else int(server_max_response_tokens)
            ),
            "effective_max_tokens": int(semantic_effective_max),
            "decode_lease_tokens": int(decode_lease_tokens),
            "uncapped_response_requested": bool(uncapped_response_requested),
            "uncapped_response_lease_tokens": (
                None
                if uncapped_response_lease_tokens is None
                else int(uncapped_response_lease_tokens)
            ),
            "uncapped_response_lease_applied": bool(
                uncapped_response_lease_applied
            ),
            "remaining_context_tokens": int(remaining_context),
            "server_cap_applied": bool(
                server_max_response_tokens is not None
                and after_server_cap < before_server_cap
            ),
            "context_cap_applied": bool(semantic_effective_max < after_server_cap),
            "effective_temperature": float(sampler.temperature),
            "effective_top_p": float(sampler.top_p),
            "effective_top_k": int(sampler.top_k),
        },
    )


def _uncapped_repetition_stop_enabled(generation_limits: dict[str, Any]) -> bool:
    if not bool(generation_limits.get("uncapped_response_requested")):
        return False
    if generation_limits.get("server_max_response_tokens") is not None:
        return False
    raw = os.environ.get("MTPLX_UNCAPPED_REPETITION_STOP", "1").strip().lower()
    return raw not in _UNCAPPED_RESPONSE_LEASE_DISABLED_VALUES


def _fresh_seed() -> int:
    # Keep within signed 31-bit range for downstream RNG compatibility.
    return secrets.randbelow(2**31 - 1)


def _session_bank_restore_mode(mode: str | None) -> str:
    normalized = str(mode or "clone").replace("-", "_")
    if normalized in {"reference", "reference_lease"}:
        return "reference"
    if normalized == "clone":
        return "clone"
    return "clone"


def _resolve_seed(state: ServerState, request_seed: int | None) -> tuple[int, bool]:
    if request_seed is not None:
        return int(request_seed), True
    if state.args.seed is not None:
        return int(state.args.seed), True
    return _fresh_seed(), False


def _dashboard_in_flight_count(state: ServerState) -> int:
    try:
        return int(state.dashboard.in_flight.count())
    except Exception:
        return 0


def _ar_batch_mtp_fallback_reason(state: ServerState) -> str | None:
    config = _scheduler_config_from_args(state.args)
    if config.mode not in {SchedulerMode.AR_BATCH, SchedulerMode.MTP_COHORT_EXPERIMENTAL}:
        return None
    burst_reason = (
        "open_code_fair_burst"
        if _scheduler_policy_label(config) == "open_code_fair"
        else "batch_size_gt_1"
    )
    wait_s = max(0.0, float(config.to_dict()["batch_wait_ms"]) / 1000.0)
    deadline = time.perf_counter() + wait_s
    while True:
        service = getattr(state, "ar_batch_service", None)
        service_snapshot: dict[str, Any] = {}
        if service is not None and hasattr(service, "snapshot"):
            try:
                service_snapshot = dict(service.snapshot())
            except Exception:
                service_snapshot = {}
        if (
            _dashboard_in_flight_count(state) > 1
            or int(service_snapshot.get("pending") or 0) > 0
            or int(service_snapshot.get("active") or 0) > 0
        ):
            return burst_reason
        if time.perf_counter() >= deadline:
            return None
        time.sleep(0.001)


def _use_live_ar_batch(
    state: ServerState,
    *,
    effective_mode: str,
) -> tuple[bool, str | None]:
    config = _scheduler_config_from_args(state.args)
    if config.mode not in {SchedulerMode.AR_BATCH, SchedulerMode.MTP_COHORT_EXPERIMENTAL}:
        return False, None
    if effective_mode == "ar":
        return True, "generation_mode_ar"
    fallback_reason = _ar_batch_mtp_fallback_reason(state)
    if fallback_reason is None:
        return False, None
    return True, fallback_reason


def _ar_batch_history_bypass_reason(
    request_observability: dict[str, Any] | None,
) -> str | None:
    """Return why a request must stay out of the live AR batch lane.

    The live AR batch lane is an agent fairness feature, not a generic OpenAI
    API compatibility mode. Anonymous benchmark/API calls should keep the solo
    MTP path so concurrent `mtplx serve` users do not see hidden AR fallback.

    Tool/history turns are still plain prompt tokens when no restored cache is
    handed to mlx-lm's ``BatchGenerator``. The unsafe object is the restored
    non-mergeable paged KV history cache, not the existence of assistant/tool
    roles in the prompt. Keep this hook for future explicit bypass reasons, but
    do not serialize OpenCode follow-up turns by role alone.
    """

    request_observability = request_observability or {}
    client_hint = str(request_observability.get("request_client_hint") or "").lower()
    client_label = str(request_observability.get("request_client_label") or "").lower()
    client = client_hint or client_label
    if client in {"", "openai"}:
        return "generic_openai_solo_mtp"
    return None


def _finalize_batched_ar_generation(
    state: ServerState,
    prompt_ids: list[int],
    generated: dict[str, Any],
    *,
    session_id: str | None,
    session_cache_hit: bool,
    cache_miss_reason: str | None,
    session_restore_mode: str,
    request_observability: dict[str, Any] | None,
) -> dict[str, Any]:
    token_times = [float(value) for value in generated.pop("_token_times", [])]
    generation_limits = dict(generated.pop("_generation_limits", {}) or {})
    completion_tokens = _effective_completion_tokens(
        generated_tokens=list(generated.get("tokens") or []),
        streamed_token_times=token_times,
    )
    elapsed_s = float(generated.get("elapsed_s") or 0.0)
    stats = _repair_streamed_generation_stats(
        dict(generated.get("stats") or {}),
        completion_tokens=completion_tokens,
        elapsed_s=elapsed_s,
    )
    envelope = _metrics_envelope(
        stats=stats,
        prompt_tokens=len(prompt_ids),
        completion_tokens=completion_tokens,
        request_elapsed_s=elapsed_s,
        token_times=token_times,
        request_started_s=float(stats.get("request_started_s") or time.perf_counter()),
        lock_wait_time_s=float(stats.get("queue_wait_s") or 0.0),
        session_id=session_id,
        session_cache_hit=session_cache_hit,
        cache_miss_reason=cache_miss_reason,
        session_restore_mode=session_restore_mode,
        mtp_depth=0,
        generation_limits=generation_limits,
    )
    envelope["generation_mode"] = "ar"
    envelope["requested_mtp_depth"] = 0
    envelope["long_context_mtp_depth_policy"] = {}
    envelope["mtp_depth"] = 0
    envelope["verify_calls"] = 0
    envelope["verify_time_s"] = 0.0
    envelope["accepted_by_depth"] = []
    envelope["draft_time_s"] = 0.0
    for key in (
        "cached_tokens",
        "new_prefill_tokens",
        "session_cache_hit",
        "cache_source",
        "ssd_cache_hit",
        "ssd_cached_tokens",
        "ssd_restore_s",
        "ssd_suffix_tokens",
        "cache_miss_reason",
        "session_restore_mode",
        "ar_batch_shared_prefix_tokens",
        "ar_batch_shared_prefix_prefill_s",
        "ar_batch_shared_prefix_snapshot_s",
        "ar_batch_prompt_prepare_s",
        "scheduler_lane",
        "scheduler_mode",
        "batching_preset",
        "scheduler_policy",
        "request_id",
        "client_label",
        "queue_wait_s",
        "active_batch_size",
        "ar_batch_max_observed",
        "mtp_disabled_reason",
    ):
        if key in stats:
            envelope[key] = stats[key]
    if request_observability:
        envelope.update(request_observability)
    cleanup = _auto_clear_mlx_cache_after_completed_request(
        state,
        session_id=session_id,
        request_observability=request_observability,
    )
    if cleanup is not None:
        envelope["mlx_cache_cleanup"] = cleanup
    envelope.update(_mlx_allocator_public_stats())
    for key in (
        "cached_tokens",
        "new_prefill_tokens",
        "session_cache_hit",
        "cache_source",
        "ssd_cache_hit",
        "ssd_cached_tokens",
        "ssd_restore_s",
        "ssd_suffix_tokens",
        "cache_miss_reason",
        "session_restore_mode",
        "ar_batch_shared_prefix_tokens",
        "ar_batch_shared_prefix_prefill_s",
        "ar_batch_shared_prefix_snapshot_s",
        "ar_batch_prompt_prepare_s",
    ):
        if key in stats:
            envelope[key] = stats[key]
    if bool(envelope.get("session_cache_hit")):
        envelope["cache_miss_reason"] = None
    stats.update(envelope)
    stats.update(_generation_truth_stats(state, "ar"))
    stats["server_elapsed_s"] = elapsed_s
    stats["server_tok_s"] = (
        completion_tokens / elapsed_s if elapsed_s > 0 else 0.0
    )
    state.last_metrics.append(dict(envelope))
    state.last_metrics = state.last_metrics[-100:]
    state.last_request_at = time.time()
    state.requests_completed += 1
    _dashboard_record_completion(state, envelope=envelope, stats=stats)
    generated["stats"] = _json_safe(stats)
    generated["completion_tokens"] = completion_tokens
    generated["tok_s"] = stats.get("decode_tok_s") or generated.get("tok_s") or 0.0
    generated["end_to_end_tok_s"] = stats["server_tok_s"]
    if not bool((request_observability or {}).get("warmup")) and not _server_console_enabled(state):
        _safe_stdout_print(
            json.dumps(
                {
                    "event": "mtplx_openai_generation",
                    "scheduler_lane": "ar_batch",
                    "prompt_tokens": generated.get("prompt_tokens"),
                    "completion_tokens": completion_tokens,
                    "elapsed_s": round(elapsed_s, 6),
                    "tok_s": round(float(generated.get("tok_s") or 0.0), 6),
                    "end_to_end_tok_s": round(float(generated["end_to_end_tok_s"]), 6),
                    "seed": stats.get("server_seed"),
                    "mtp_disabled_reason": stats.get("mtp_disabled_reason"),
                    "text_preview": str(generated.get("text") or "")[:120],
                },
                ensure_ascii=False,
            )
        )
    return generated


def _smart_fan_request_id(response_id: str | None, fallback_prefix: str) -> str:
    return response_id or f"{fallback_prefix}-{uuid.uuid4().hex}"


def _begin_smart_fan_request(
    state: Any,
    *,
    request_id: str | None,
    background_request: bool = False,
    request_observability: dict[str, Any] | None = None,
) -> str | None:
    if background_request:
        return None
    if bool((request_observability or {}).get("warmup")):
        return None
    try:
        mode = normalize_fan_mode(getattr(state, "fan_mode", FAN_MODE_DEFAULT))
    except ValueError:
        mode = FAN_MODE_DEFAULT
    if mode != FAN_MODE_SMART:
        return None
    controller = getattr(state, "smart_fans", None)
    if controller is None:
        return None
    lease_id = request_id or f"request-{uuid.uuid4().hex}"
    try:
        controller.begin_request(lease_id)
    except Exception as exc:
        LOGGER.warning("Smart fan begin failed: %s", exc)
    return lease_id


def _end_smart_fan_request(
    state: Any,
    lease_id: str | None,
    *,
    wait_for_restore: bool = False,
) -> None:
    if lease_id is None:
        return
    controller = getattr(state, "smart_fans", None)
    if controller is None:
        return
    try:
        controller.end_request(lease_id, wait_for_restore=wait_for_restore)
    except Exception as exc:
        LOGGER.warning("Smart fan restore failed: %s", exc)


def _run_generation_dispatched(
    state: ServerState,
    prompt_ids: list[int],
    *,
    batch_key: str,
    response_id: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    effective_mode = _normalize_generation_mode(
        kwargs.get("generation_mode"),
        default=getattr(state.args, "generation_mode", "mtp"),
    )
    request_observability_for_lane = dict(kwargs.get("request_observability") or {})
    if response_id:
        request_observability_for_lane.setdefault("request_id", response_id)
    kwargs["request_observability"] = request_observability_for_lane
    history_bypass_reason = _ar_batch_history_bypass_reason(
        request_observability_for_lane
    )
    if history_bypass_reason is not None:
        use_ar_batch = False
        mtp_disabled_reason = None
        request_observability_for_lane["scheduler_lane"] = (
            "solo_mtp_generic"
            if history_bypass_reason == "generic_openai_solo_mtp"
            else "solo_mtp_history"
        )
        request_observability_for_lane["ar_batch_bypass_reason"] = (
            history_bypass_reason
        )
    else:
        use_ar_batch, mtp_disabled_reason = _use_live_ar_batch(
            state,
            effective_mode=effective_mode,
        )
    if use_ar_batch:
        if mtp_disabled_reason not in {None, "generation_mode_ar"}:
            LOGGER.warning(
                "Solo MTP bypassed for fair AR batching",
                extra={
                    "request_id": response_id,
                    "mtp_disabled_reason": mtp_disabled_reason,
                    "scheduler_policy": _scheduler_policy_label(
                        _scheduler_config_from_args(state.args)
                    ),
                },
            )
        response_max, sampler, generation_limits = _generation_params(
            state,
            prompt_token_count=len(prompt_ids),
            max_tokens=kwargs.get("max_tokens"),
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            top_k=kwargs.get("top_k"),
        )
        generation_seed, _seed_is_explicit = _resolve_seed(state, kwargs.get("seed"))
        request_observability = dict(kwargs.get("request_observability") or {})
        request_observability["scheduler_lane"] = "ar_batch"
        if kwargs.get("cache_miss_reason") is not None:
            request_observability.setdefault(
                "cache_miss_reason", kwargs.get("cache_miss_reason")
            )
        request_observability.setdefault(
            "session_restore_mode", str(kwargs.get("session_restore_mode") or "cold")
        )
        if mtp_disabled_reason:
            request_observability["mtp_disabled_reason"] = mtp_disabled_reason
        job = _BatchedARJob(
            request_id=response_id or f"arbatch-{uuid.uuid4().hex}",
            prompt_ids=prompt_ids,
            max_tokens=response_max,
            sampler=sampler,
            seed=generation_seed,
            stop_token_ids=_default_stop_tokens(state.runtime.tokenizer),
            token_callback=kwargs.get("token_callback"),
            prefill_callback=kwargs.get("prefill_callback"),
            request_observability=request_observability,
            mtp_disabled_reason=mtp_disabled_reason,
            generation_limits=generation_limits,
            cancel_event=kwargs.get("cancel_event"),
            session_id=kwargs.get("session_id"),
            session_bank=kwargs.get("session_bank"),
            session_restore_mode=str(kwargs.get("session_restore_mode") or "cold"),
            session_template_hash=kwargs.get("session_template_hash"),
            session_draft_head_identity=kwargs.get("session_draft_head_identity"),
            session_policy_fingerprint=kwargs.get("session_policy_fingerprint"),
        )
        ar_request_id = response_id or job.request_id
        smart_fan_lease = _begin_smart_fan_request(
            state,
            request_id=_smart_fan_request_id(ar_request_id, "arbatch"),
            request_observability=request_observability,
        )
        state.begin_foreground()
        try:
            future = state.ar_batch_service.submit(job)
            generated = future.result()
        finally:
            state.end_foreground()
            _end_smart_fan_request(state, smart_fan_lease)
        ar_stats = dict(generated.get("stats") or {})
        ar_session_cache_hit = bool(ar_stats.get("session_cache_hit") or False)
        ar_cache_miss_reason = ar_stats.get("cache_miss_reason")
        if ar_session_cache_hit:
            ar_cache_miss_reason = None
        elif ar_cache_miss_reason is None:
            ar_cache_miss_reason = kwargs.get("cache_miss_reason")
        return _finalize_batched_ar_generation(
            state,
            prompt_ids,
            generated,
            session_id=kwargs.get("session_id"),
            session_cache_hit=ar_session_cache_hit,
            cache_miss_reason=ar_cache_miss_reason,
            session_restore_mode=str(
                ar_stats.get("session_restore_mode")
                or kwargs.get("session_restore_mode")
                or "ar_batch"
            ),
            request_observability=request_observability,
        )

    def run() -> dict[str, Any]:
        return _run_generation(state, prompt_ids, **kwargs)

    scheduler = getattr(state, "model_scheduler", None)
    if scheduler is not None and hasattr(scheduler, "is_owner_thread") and scheduler.is_owner_thread():
        return run()
    return _submit_foreground_model_work(
        state,
        run,
        batch_key=batch_key,
    ).result()


def _run_generation(
    state: ServerState,
    prompt_ids: list[int],
    *,
    max_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    seed: int | None,
    draft_sampler: SamplerConfig | None = None,
    generation_mode: str | None = None,
    depth: int | None = None,
    resolved_mtp_depth: int | None = None,
    token_callback: Callable[[list[int]], None] | None = None,
    session_id: str | None = None,
    session_cache_hit: bool = False,
    cache_miss_reason: str | None = CacheMissReason.NEW_SESSION.value,
    session_restore_mode: str = "cold",
    session_bank: Any | None = None,
    session_template_hash: str | None = None,
    session_draft_head_identity: str | None = None,
    session_policy_fingerprint: str | None = None,
    background_request: bool = False,
    commit_final_state_to_bank: bool = True,
    commit_prompt_prefix_to_bank: bool = False,
    session_keep_live_ref: bool = True,
    request_observability: dict[str, Any] | None = None,
    prefill_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_event: Event | None = None,
    streaming_response: bool | None = None,
) -> dict[str, Any]:
    response_max, sampler, generation_limits = _generation_params(
        state,
        prompt_token_count=len(prompt_ids),
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    uncapped_repetition_stop = _uncapped_repetition_stop_enabled(generation_limits)
    generation_limits["uncapped_repetition_stop_enabled"] = bool(
        uncapped_repetition_stop
    )
    effective_draft_sampler = draft_sampler if draft_sampler is not None else state.draft_sampler
    effective_mode = _normalize_generation_mode(
        generation_mode,
        default=getattr(state.args, "generation_mode", "mtp"),
    )
    requested_depth = (
        0
        if effective_mode == "ar"
        else int(depth if depth is not None else getattr(state.args, "depth", 3))
    )
    effective_depth = requested_depth
    if (
        effective_mode != "ar"
        and resolved_mtp_depth is not None
        and requested_depth > 0
    ):
        effective_depth = max(1, min(int(resolved_mtp_depth), int(requested_depth)))
    started = time.perf_counter()
    token_times: list[float] = []
    lock_wait_time_s = 0.0

    def record_tokens(new_tokens: list[int]) -> None:
        now = time.perf_counter()
        token_times.extend([now for _token in new_tokens])
        if token_callback is not None:
            token_callback(new_tokens)

    blank_retry_budget = max(0, int(state.args.blank_retry_attempts))
    response_is_streaming = (
        token_callback is not None
        if streaming_response is None
        else bool(streaming_response)
    )
    max_attempts = 1 if response_is_streaming else 1 + blank_retry_budget
    last: dict[str, Any] | None = None
    trace_preview = (
        str((request_observability or {}).get("request_last_user_preview") or "")
        .replace("\n", " ")
        .strip()
    )
    trace_label = (
        f"{session_id or 'stateless'}:{trace_preview[:64]}"
        if trace_preview
        else session_id
    )
    trace_metadata = {
        "session_id": session_id,
        "session_restore_mode": session_restore_mode,
        "background_request": bool(background_request),
        "cache_bypass": session_bank is None,
        "generation_mode": effective_mode,
        **(request_observability or {}),
    }
    for attempt in range(max_attempts):
        generation_seed, seed_is_explicit = _resolve_seed(state, seed)
        lock_started = time.perf_counter()
        smart_fan_lease: str | None = None
        if background_request:
            if state.has_foreground() or state.lock.locked():
                raise HTTPException(
                    status_code=503,
                    detail="background request bypassed while foreground generation is busy",
                    headers={"Retry-After": "1"},
                )
            acquired = state.lock.acquire(blocking=False)
            if not acquired:
                raise HTTPException(
                    status_code=503,
                    detail="background request bypassed while model lock is busy",
                    headers={"Retry-After": "1"},
                )
        else:
            smart_request_id = str((request_observability or {}).get("request_id") or "")
            smart_fan_lease = _begin_smart_fan_request(
                state,
                request_id=_smart_fan_request_id(
                    smart_request_id or session_id,
                    "generation",
                ),
                request_observability=request_observability,
            )
            state.begin_foreground()
            state.lock.acquire()
        lock_wait_time_s += time.perf_counter() - lock_started
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise _StreamCancelled("request cancelled before generation")
            dynamic_kv_reservation = _dynamic_paged_kv_reservation(
                prompt_tokens=len(prompt_ids),
                max_new_tokens=response_max,
                mtp_depth=effective_depth,
            )
            prefill_chunk_tokens = getattr(state.args, "prefill_chunk_tokens", None)
            with _temporary_env(
                dynamic_kv_reservation["env"]
            ), prefill_chunk_size_override(prefill_chunk_tokens):
                if effective_mode == "ar":
                    out = generate_ar(
                        state.runtime,
                        prompt_ids,
                        max_tokens=response_max,
                        sampler=sampler,
                        seed=generation_seed,
                        token_callback=record_tokens,
                        trace_label=trace_label,
                        trace_metadata=trace_metadata,
                        prefill_callback=prefill_callback,
                        repetition_stop=uncapped_repetition_stop,
                    )
                else:
                    adaptive_policy = _make_adaptive_policy(
                        state.args, max_depth=effective_depth
                    )
                    out = generate_mtpk(
                        state.runtime,
                        prompt_ids,
                        max_tokens=response_max,
                        sampler=sampler,
                        draft_sampler=effective_draft_sampler,
                        speculative_depth=effective_depth,
                        seed=generation_seed,
                        mtp_hidden_variant="post_norm",
                        mtp_cache_policy="persistent",
                        mtp_history_policy="committed",
                        verify_strategy=state.args.verify_strategy,
                        verify_core=state.args.verify_core,
                        token_callback=record_tokens,
                        session_bank=session_bank,
                        session_id=session_id,
                        session_restore_mode=_session_bank_restore_mode(
                            session_restore_mode
                        ),
                        session_template_hash=session_template_hash,
                        session_draft_head_identity=session_draft_head_identity,
                        session_policy_fingerprint=session_policy_fingerprint,
                        capture_final_state=session_bank is not None,
                        commit_prompt_state_to_bank=(
                            commit_prompt_prefix_to_bank
                            and session_bank is not None
                            and session_id is not None
                        ),
                        # Prompt-prefix commits happen before decode mutates
                        # the same KV/MTP cache objects. They must snapshot or
                        # skip, not live-lease the mutable prompt cache.
                        commit_prompt_state_keep_live_ref=False,
                        trace_label=trace_label,
                        trace_metadata=trace_metadata,
                        prefill_callback=prefill_callback,
                        adaptive_policy=adaptive_policy,
                        repetition_stop=uncapped_repetition_stop,
                        online_correction_cache=bool(
                            state.args.online_correction_cache
                        ),
                        online_correction_cache_min_depth=int(
                            state.args.online_correction_cache_min_depth
                        ),
                        online_correction_cache_key=str(
                            state.args.online_correction_cache_key
                        ),
                        prompt_correction_cache=bool(
                            state.args.prompt_correction_cache
                        ),
                        prompt_correction_cache_min_depth=int(
                            state.args.prompt_correction_cache_min_depth
                        ),
                        online_hidden_corrector_alpha=float(
                            state.args.online_hidden_corrector_alpha
                        ),
                        online_hidden_corrector_decay=float(
                            state.args.online_hidden_corrector_decay
                        ),
                        online_hidden_corrector_warmup=int(
                            state.args.online_hidden_corrector_warmup
                        ),
                        online_hidden_corrector_max_feed_depth=(
                            state.args.online_hidden_corrector_max_feed_depth
                        ),
                        online_hidden_corrector_key=str(
                            state.args.online_hidden_corrector_key
                        ),
                    )
        finally:
            state.lock.release()
            if not background_request:
                state.end_foreground()
                _end_smart_fan_request(state, smart_fan_lease)
        elapsed_s = time.perf_counter() - started
        completion_tokens = _effective_completion_tokens(
            generated_tokens=list(out.tokens),
            streamed_token_times=token_times,
        )
        server_tok_s = completion_tokens / elapsed_s if elapsed_s > 0 else 0.0
        stats = _repair_streamed_generation_stats(
            out.stats.to_dict(),
            completion_tokens=completion_tokens,
            elapsed_s=elapsed_s,
        )
        if effective_mode == "mtp":
            stats["requested_speculative_depth"] = int(requested_depth)
        if session_bank is not None:
            cache_miss_reason = stats.get("cache_miss_reason") or cache_miss_reason
            session_cache_hit = bool(stats.get("session_cache_hit") or False)
            if session_cache_hit:
                cache_miss_reason = None
            session_restore_mode = str(
                stats.get("session_restore_mode") or session_restore_mode
            )
        final_state = out.final_state
        if (
            commit_final_state_to_bank
            and session_bank is not None
            and session_id is not None
            and final_state is not None
            and final_state.safe_to_commit
        ):
            final_token_ids = list(prompt_ids) + list(out.tokens)
            mtp_snapshot = (
                snapshot_cache(final_state.final_committed_mtp_cache)
                if final_state.final_committed_mtp_cache is not None
                else None
            )
            session_bank.put(
                runtime=state.runtime,
                token_ids=final_token_ids,
                cache=final_state.final_trunk_cache,
                logits=final_state.final_logits,
                hidden=final_state.final_hidden,
                hidden_variant="post_norm",
                keep_live_ref=bool(session_keep_live_ref),
                session_id=session_id,
                template_hash=session_template_hash,
                mtp_history_policy="committed",
                draft_head_identity=session_draft_head_identity,
                policy_fingerprint=session_policy_fingerprint,
                mtp_history_snapshot=mtp_snapshot,
                snapshot_epoch=len(final_token_ids),
                mtp_snapshot_epoch=len(final_token_ids)
                if mtp_snapshot is not None
                else None,
                extra_state=getattr(final_state, "extra_state", None),
            )
            stats["sessionbank_snapshot_bytes"] = int(
                getattr(session_bank, "last_put_nbytes", 0) or 0
            )
            stats["sessionbank_skipped_oversized_snapshot"] = bool(
                getattr(session_bank, "last_put_skipped_oversized_snapshot", False)
            )
        actual_mtp_depth = int(
            stats.get("speculative_depth")
            or (effective_depth if effective_mode == "mtp" else 0)
            or 0
        )
        requested_mtp_depth = int(
            stats.get("requested_speculative_depth")
            or (requested_depth if effective_mode == "mtp" else 0)
            or 0
        )
        envelope = _metrics_envelope(
            stats=stats,
            prompt_tokens=len(prompt_ids),
            completion_tokens=completion_tokens,
            request_elapsed_s=elapsed_s,
            token_times=token_times,
            request_started_s=started,
            lock_wait_time_s=lock_wait_time_s,
            session_id=session_id,
            session_cache_hit=session_cache_hit,
            cache_miss_reason=cache_miss_reason,
            session_restore_mode=session_restore_mode,
            mtp_depth=actual_mtp_depth if effective_mode == "mtp" else 0,
            generation_limits=generation_limits,
        )
        envelope["dynamic_paged_kv"] = {
            key: value for key, value in dynamic_kv_reservation.items() if key != "env"
        }
        for key in (
            "paged_kv_capacity_tokens",
            "paged_kv_num_blocks",
            "paged_active_array_calls",
            "paged_active_array_time_s",
            "paged_turboquant",
            "paged_turboquant_k_quant",
            "paged_turboquant_v_quant",
            "paged_turboquant_attention_calls",
            "paged_kv_quant",
            "paged_kv_quant_mode",
            "paged_kv_quant_attention_calls",
            "paged_kv_quant_dequant_calls",
            "paged_kv_quant_dequant_time_s",
            "paged_kv_quant_dequant_tokens",
            "paged_gqa_sdpa_calls",
            "paged_gqa_sdpa_calls_by_route",
            "paged_gqa_sdpa_calls_by_phase",
            "paged_gqa_sdpa_route_misses_by_phase_reason",
            "paged_gqa_sdpa_route_misses_by_q_len",
            "paged_gqa_sdpa_last_route_miss",
            "attention_dense_fallback_calls",
            "prefill_dense_fallback_calls",
            "decode_dense_fallback_calls",
            "ar_dense_fallback_calls",
            "postcommit_dense_fallback_calls",
            "paged_attention_bailouts_by_phase_reason",
            "paged_attention_large_q_path",
            "large_q_split_sdpa_fallback_calls",
            "large_q_split_sdpa_fallback_calls_by_phase",
            "prefill_large_q_split_sdpa_fallback_calls",
            "decode_large_q_split_sdpa_fallback_calls",
            "partitioned_paged_calls",
            "partitioned_paged_calls_by_phase",
            "prefill_partitioned_paged_calls",
            "decode_partitioned_paged_calls",
        ):
            if key in stats:
                envelope[key] = stats[key]
        envelope["generation_mode"] = effective_mode
        envelope["requested_mtp_depth"] = (
            requested_mtp_depth if effective_mode == "mtp" else 0
        )
        envelope["long_context_mtp_depth_policy"] = (
            (stats.get("long_context_mtp_depth_policy") or {})
            if effective_mode == "mtp"
            else {}
        )
        if effective_mode == "ar":
            envelope["mtp_depth"] = 0
            envelope["requested_mtp_depth"] = 0
            envelope["long_context_mtp_depth_policy"] = {}
            envelope["verify_calls"] = 0
            for key in (
                "verify_time_s",
                "verify_forward_time_s",
                "verify_eval_time_s",
                "verify_logits_eval_time_s",
                "verify_hidden_eval_time_s",
                "verify_joint_eval_time_s",
                "verify_target_distribution_time_s",
                "target_distribution_materialized_rows",
                "target_distribution_materialized_windows",
                "target_distribution_share",
                "lazy_bonus_verify_calls",
                "lazy_bonus_commit_time_s",
                "verify_eval_unattributed_time_s",
            ):
                envelope[key] = 0 if key.endswith(("rows", "windows", "calls")) else 0.0
            envelope["accepted_by_depth"] = []
            envelope["draft_time_s"] = 0.0
        if request_observability:
            envelope.update(request_observability)
            if request_observability.get("live_frontier_result_turn"):
                frontier_hit = bool(session_cache_hit)
                envelope["live_frontier_hit"] = frontier_hit
                envelope["live_frontier_restore_mode"] = session_restore_mode
                envelope["live_frontier_miss_reason"] = (
                    None
                    if frontier_hit
                    else _live_frontier_miss_reason_from_counts(
                        assistant_tool_call_count=int(
                            request_observability.get(
                                "live_frontier_assistant_tool_call_count"
                            )
                            or 0
                        ),
                        tool_result_count=int(
                            request_observability.get("live_frontier_tool_result_count")
                            or 0
                        ),
                        unknown_tool_result_count=int(
                            request_observability.get(
                                "live_frontier_unknown_tool_result_count"
                            )
                            or 0
                        ),
                        cache_miss_reason=cache_miss_reason,
                        session_source=str(
                            request_observability.get("request_session_source") or ""
                        ),
                        session_keep_live_ref=session_keep_live_ref,
                    )
                )
        cleanup = _auto_clear_mlx_cache_after_completed_request(
            state,
            session_id=session_id,
            request_observability=request_observability,
        )
        if cleanup is not None:
            envelope["mlx_cache_cleanup"] = cleanup
        envelope.update(_mlx_allocator_public_stats())
        stats["generation_mode"] = effective_mode
        stats.update(envelope)
        stats.update(_generation_truth_stats(state, effective_mode))
        if effective_mode == "ar":
            stats["mtp_depth"] = 0
            stats["verify_calls"] = 0
            for key in (
                "verify_time_s",
                "verify_forward_time_s",
                "verify_eval_time_s",
                "verify_logits_eval_time_s",
                "verify_hidden_eval_time_s",
                "verify_joint_eval_time_s",
                "verify_target_distribution_time_s",
                "target_distribution_materialized_rows",
                "target_distribution_materialized_windows",
                "target_distribution_share",
                "lazy_bonus_verify_calls",
                "lazy_bonus_commit_time_s",
                "verify_eval_unattributed_time_s",
            ):
                stats[key] = 0 if key.endswith(("rows", "windows", "calls")) else 0.0
            stats["accepted_by_depth"] = []
            stats["drafted_by_depth"] = []
            stats["mean_accept_probability_by_depth"] = []
            stats["draft_time_s"] = 0.0
        stats["server_elapsed_s"] = elapsed_s
        stats["server_tok_s"] = server_tok_s
        stats["server_seed"] = generation_seed
        stats["server_attempts"] = attempt + 1
        stats["server_blank_retries"] = attempt
        stats["server_blank_retry_suppressed"] = bool(
            response_is_streaming and blank_retry_budget
        )
        state.last_metrics.append(dict(envelope))
        state.last_metrics = state.last_metrics[-100:]
        state.last_request_at = time.time()
        state.requests_completed += 1
        _dashboard_record_completion(state, envelope=envelope, stats=stats)
        last = {
            "text": out.text,
            "tokens": out.tokens,
            "stats": _json_safe(stats),
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": completion_tokens,
            "elapsed_s": elapsed_s,
            "tok_s": stats.get("decode_tok_s") or server_tok_s,
            "end_to_end_tok_s": server_tok_s,
            "_final_state": final_state,
            "finish_reason": (
                out.final_state.finish_reason if out.final_state is not None else "stop"
            ),
        }
        if seed_is_explicit or out.text.strip():
            break
    assert last is not None
    if not bool(
        (request_observability or {}).get("warmup")
    ) and not _server_console_enabled(state):
        _safe_stdout_print(
            json.dumps(
                {
                    "event": "mtplx_openai_generation",
                    "prompt_tokens": last["prompt_tokens"],
                    "completion_tokens": last["completion_tokens"],
                    "elapsed_s": round(float(last["elapsed_s"]), 6),
                    "tok_s": round(float(last["tok_s"]), 6),
                    "end_to_end_tok_s": round(float(last["end_to_end_tok_s"]), 6),
                    "seed": last["stats"].get("server_seed"),
                    "attempts": last["stats"].get("server_attempts"),
                    "blank_retries": last["stats"].get("server_blank_retries"),
                    "text_preview": str(last["text"])[:120],
                },
                ensure_ascii=False,
            )
        )
    return last


def _run_startup_warmup(state: ServerState) -> dict[str, Any]:
    warmup_tokens = int(getattr(state.args, "warmup_tokens", 0) or 0)
    status: dict[str, Any] = {
        "enabled": warmup_tokens > 0,
        "ran": False,
        "tokens": warmup_tokens,
        "elapsed_s": 0.0,
        "error": None,
    }
    if warmup_tokens <= 0:
        _startup_line("[6/6] Warmup skipped (--warmup-tokens 0)")
        return status
    started = time.perf_counter()
    _startup_line(f"[6/6] Warming model with {warmup_tokens} tokens")
    warmup_heartbeat = _startup_heartbeat("warmup still running", interval_s=5.0)
    try:
        prompt_ids = _encode_prompt(state.runtime.tokenizer, "MTPLX warmup.")
        generated = _submit_foreground_model_work(
            state,
            lambda: _run_generation(
                state,
                prompt_ids,
                max_tokens=warmup_tokens,
                temperature=state.args.temperature,
                top_p=state.args.top_p,
                top_k=state.args.top_k,
                seed=0,
                request_observability={"warmup": True},
            ),
            batch_key="startup.warmup",
        ).result()
    except BaseException as exc:
        status.update(
            {
                "elapsed_s": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        warmup_heartbeat.set()
        _startup_line(
            f"[6/6] Warmup failed after {status['elapsed_s']:.1f}s: {status['error']}"
        )
        if getattr(state.args, "strict_warmup", False):
            raise
        return status
    finally:
        warmup_heartbeat.set()
    status.update(
        {
            "ran": True,
            "elapsed_s": time.perf_counter() - started,
            "completion_tokens": generated.get("completion_tokens"),
            "tok_s": generated.get("tok_s"),
        }
    )
    tok_s = status.get("tok_s")
    tok_s_text = "unknown tok/s" if tok_s is None else f"{float(tok_s):.2f} tok/s"
    _startup_line(f"[6/6] Warmup complete in {status['elapsed_s']:.1f}s ({tok_s_text})")
    return status


def _decode_timing(stats: dict[str, Any]) -> tuple[float, float]:
    generated_tokens = int(stats.get("generated_tokens") or 0)
    elapsed_s = float(stats.get("elapsed_s") or 0.0)
    if "prompt_eval_time_s" in stats:
        prompt_eval_time_s = float(stats.get("prompt_eval_time_s") or 0.0)
    else:
        target_forward_time_s = float(stats.get("target_forward_time_s") or 0.0)
        verify_time_s = float(stats.get("verify_time_s") or 0.0)
        repair_time_s = float(stats.get("repair_time_s") or 0.0)
        prompt_eval_time_s = max(
            0.0, target_forward_time_s - verify_time_s - repair_time_s
        )
    cache_restore_time_s = max(
        float(stats.get("cache_restore_time_s") or 0.0),
        float(stats.get("ssd_restore_s") or 0.0),
    )
    decode_elapsed_s = max(0.0, elapsed_s - prompt_eval_time_s - cache_restore_time_s)
    if decode_elapsed_s <= 0.0:
        return 0.0, 0.0
    return generated_tokens / decode_elapsed_s, decode_elapsed_s


def _stats_footer_text(state: ServerState, generated: dict[str, Any]) -> str:
    if not state.args.stats_footer:
        return ""
    stats = generated["stats"]
    tok_s, decode_elapsed_s = _decode_timing(stats)
    completion_tokens = int(generated.get("completion_tokens") or 0)
    footer = f"**{tok_s:.1f} tok/s** · {completion_tokens} tokens · {decode_elapsed_s:.2f}s decode"
    return f"{STATS_FOOTER_MARKER} {footer}"


def _usage_payload(generated: dict[str, Any]) -> dict[str, int]:
    prompt_tokens = int(generated.get("prompt_tokens") or 0)
    completion_tokens = int(generated.get("completion_tokens") or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _strip_generated_chat_template_sentinels(text: str) -> str:
    if not text:
        return ""
    return CHAT_TEMPLATE_SENTINEL_RE.sub("", text)


def _clean_generated_assistant_text(text: str) -> str:
    cleaned = strip_qwen_style_reasoning_control_markup(text)
    cleaned = _REASONING_CONTROL_TAG_RE.sub("", cleaned)
    return _strip_generated_chat_template_sentinels(cleaned)


def _split_thinking_segments(text: str, *, thinking_enabled: bool) -> tuple[str, str]:
    if not thinking_enabled:
        return "", strip_qwen_style_reasoning_from_content(
            _strip_generated_chat_template_sentinels(text)
        )
    reasoning_parts: list[str] = []
    content_parts: list[str] = []

    def append_reasoning(segment: str) -> None:
        if not segment:
            return
        if (
            reasoning_parts
            and not reasoning_parts[-1].endswith(("\n", " "))
            and not segment.startswith(("\n", " "))
        ):
            reasoning_parts.append("\n")
        reasoning_parts.append(segment)

    position = 0
    inside_thinking = True
    while position < len(text):
        if inside_thinking:
            close_match = QWEN_STYLE_REASONING_CLOSE_RE.search(text, position)
            if close_match is None:
                segment = _clean_generated_assistant_text(text[position:])
                append_reasoning(segment)
                break
            segment = _clean_generated_assistant_text(text[position : close_match.start()])
            append_reasoning(segment)
            position = close_match.end()
            inside_thinking = False
            continue

        open_match = QWEN_STYLE_REASONING_OPEN_RE.search(text, position)
        if open_match is None:
            segment = _clean_generated_assistant_text(text[position:])
            if segment:
                content_parts.append(segment)
            break
        segment = _clean_generated_assistant_text(text[position : open_match.start()])
        if segment:
            content_parts.append(segment)
        position = open_match.end()
        inside_thinking = True
    return "".join(reasoning_parts).strip(), "".join(content_parts).strip()


def _normalize_thinking_tags(text: str, *, thinking_enabled: bool) -> str:
    """Make Qwen thinking output parseable by Open WebUI's native tag handler.

    Qwen's chat template can put the opening <think> tag in the prompt, so the
    generated text may begin inside the reasoning block and only emit </think>.
    Open WebUI's built-in parser needs a start tag in the streamed content.
    """
    reasoning, content = _split_thinking_segments(
        text, thinking_enabled=thinking_enabled
    )
    if not thinking_enabled:
        return content
    pieces: list[str] = []
    if reasoning:
        pieces.append(f"{THINK_OPEN}\n{reasoning}\n{THINK_CLOSE}")
    if content:
        pieces.append(content)
    return "\n\n".join(pieces)


def _normalize_reasoning_tags_for_state(
    state: ServerState,
    text: str,
    *,
    thinking_enabled: bool,
) -> str:
    return normalize_backend_reasoning_tags(
        text,
        parser=_reasoning_parser_for_state(state),
        thinking_enabled=thinking_enabled,
    )


def _split_backend_reasoning_for_state(
    state: ServerState,
    text: str,
    *,
    thinking_enabled: bool,
) -> tuple[str, str]:
    parser = _reasoning_parser_for_state(state)
    if parser in {"qwen3", "step3p5"}:
        text = normalize_qwen_thinking_tags(
            text,
            thinking_enabled=thinking_enabled,
        )
    parts = split_reasoning_text(
        text,
        parser=parser,
        thinking_enabled=thinking_enabled,
    )
    return parts.reasoning, parts.content


def _tool_extraction_text_parts(
    state: ServerState,
    text: str,
    *,
    thinking_enabled: bool,
) -> tuple[str, str]:
    if _reasoning_parser_for_state(state) == "gemma4":
        return _split_backend_reasoning_for_state(
            state,
            text,
            thinking_enabled=thinking_enabled,
        )
    return omlx_extract_thinking(text)


class _IncrementalTokenDecoder:
    """Small TextStreamer-style decoder for committed-token SSE streaming.

    The previous bridge decoded the entire generated token buffer after every
    callback. That is prefix-stable, but it becomes O(n^2) tokenizer work during
    long reasoning streams. This keeps only the current partial word and flushes
    finalized text as soon as whitespace or CJK boundaries make it safe.
    """

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
        close_match = QWEN_STYLE_REASONING_CLOSE_RE.search(text, self._print_len)
        if close_match is not None:
            boundary = close_match.end()
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
        if (
            (0x4E00 <= cp <= 0x9FFF)
            or (0x3400 <= cp <= 0x4DBF)
            or (0x20000 <= cp <= 0x2A6DF)
            or (0x2A700 <= cp <= 0x2B73F)
            or (0x2B740 <= cp <= 0x2B81F)
            or (0x2B820 <= cp <= 0x2CEAF)
            or (0xF900 <= cp <= 0xFAFF)
            or (0x2F800 <= cp <= 0x2FA1F)
        ):
            return True
        return False


class _NonDuplicatingTokenDecoder(_IncrementalTokenDecoder):
    """Compatibility alias for old bridge tests/imports."""

    def feed(self, tokens: list[int]) -> str:
        text = super().feed(tokens)
        if text:
            return text
        if not self._token_cache:
            return ""
        joined = self._decode(self._token_cache)
        if QWEN_STYLE_REASONING_CLOSE_RE.search(joined):
            return self.finish()
        return ""


class _ThinkingContentStreamSplitter:
    _TOOL_CALL_MARKER = "<tool_call"
    _TOOL_CALL_CLOSE_MARKER = "</tool_call>"
    _TOOL_CONTROL_MARKERS = (
        "<tool_call",
        "<function=",
        "<parameter=",
        "</parameter>",
        "</function>",
        "</tool_call>",
        "parameter=",
        "function=",
    )

    def __init__(
        self,
        *,
        thinking_enabled: bool,
        recover_unclosed_reasoning_as_content: bool = True,
        start_inside_thinking: bool = True,
    ) -> None:
        self._thinking_enabled = thinking_enabled
        self._recover_unclosed_reasoning_as_content = (
            recover_unclosed_reasoning_as_content
        )
        self._inside_thinking = thinking_enabled and start_inside_thinking
        self._inside_tool_call = False
        self._tool_call_tail = ""
        self._pending = ""
        self._disabled_inside_reasoning = False
        self._disabled_visible_started = False
        self._reentry_count = 0
        self._reasoning_accumulated: list[str] = []
        self._content_emitted = False
        self._content_history_tail = ""
        self._post_orphan_close_duplicate_tail = ""
        self._saw_chat_template_sentinel = False

    @property
    def reentry_count(self) -> int:
        return self._reentry_count

    def start(self) -> list[tuple[str, str]]:
        return []

    def feed(self, text: str) -> list[tuple[str, str]]:
        if not text:
            return []
        if not self._thinking_enabled:
            self._pending += text
            return self._drain_disabled(final=False)
        self._pending += text
        return self._drain(final=False)

    def finish(
        self,
        *,
        recover_unclosed_reasoning_as_content: bool | None = None,
    ) -> list[tuple[str, str]]:
        chunks = (
            self._drain(final=True)
            if self._thinking_enabled
            else self._drain_disabled(final=True)
        )
        recover_unclosed_reasoning = (
            self._recover_unclosed_reasoning_as_content
            if recover_unclosed_reasoning_as_content is None
            else recover_unclosed_reasoning_as_content
        )
        if (
            self._thinking_enabled
            and recover_unclosed_reasoning
            and not self._content_emitted
            and self._reasoning_accumulated
            and not self._saw_chat_template_sentinel
        ):
            recovered = "".join(self._reasoning_accumulated).strip()
            if recovered:
                self._content_emitted = True
                chunks.append(("content", recovered))
        self._inside_thinking = False
        return chunks

    def _append_chunk(
        self,
        chunks: list[tuple[str, str]],
        field: str,
        text: str,
    ) -> None:
        cleaned = _clean_generated_assistant_text(text)
        if cleaned:
            if field == "reasoning_content":
                self._reasoning_accumulated.append(cleaned)
            elif field == "content":
                self._content_emitted = True
                self._content_history_tail = (
                    self._content_history_tail + cleaned
                )[-2048:]
            chunks.append((field, cleaned))

    @classmethod
    def _tool_control_marker_index(cls, text: str) -> int:
        text_lower = text.lower()
        indexes = [
            index
            for marker in cls._TOOL_CONTROL_MARKERS
            if (index := text_lower.find(marker)) >= 0
        ]
        return min(indexes) if indexes else -1

    @classmethod
    def _tool_control_marker_has_partial_prefix(cls, text: str) -> bool:
        text_lower = text.lower()
        return any(marker.startswith(text_lower) for marker in cls._TOOL_CONTROL_MARKERS)

    @staticmethod
    def _reasoning_control_marker_has_partial_prefix(text: str) -> bool:
        text_lower = text.lower()
        if not text_lower.startswith("<"):
            return False
        markers: list[str] = []
        for name in QWEN_STYLE_REASONING_TAG_NAMES:
            markers.append(f"<{name}")
            markers.append(f"</{name}")
        return any(marker.startswith(text_lower) for marker in markers)

    @staticmethod
    def _partial_marker_tail_len(text: str, markers: Iterable[str]) -> int:
        text_lower = text.lower()
        keep = 0
        for marker in markers:
            marker_lower = marker.lower()
            max_len = min(len(text_lower), len(marker_lower) - 1)
            for n in range(max_len, 0, -1):
                if text_lower.endswith(marker_lower[:n]):
                    keep = max(keep, n)
                    break
        return min(keep, 128)

    @classmethod
    def _disabled_reasoning_tail_len(cls, text: str) -> int:
        markers: list[str] = list(CHAT_TEMPLATE_SENTINEL_MARKERS)
        for name in QWEN_STYLE_REASONING_TAG_NAMES:
            markers.extend((f"<{name}", f"<{name}>", f"</{name}", f"</{name}>"))
        return cls._partial_marker_tail_len(text, markers)

    @classmethod
    def _disabled_reasoning_close_tail_len(cls, text: str) -> int:
        markers: list[str] = []
        for name in QWEN_STYLE_REASONING_TAG_NAMES:
            markers.extend((f"</{name}", f"</{name}>"))
        return cls._partial_marker_tail_len(text, markers)

    def _drop_duplicate_after_orphan_close(self, text: str) -> str:
        cleaned = _clean_generated_assistant_text(text).lstrip()
        if not cleaned:
            return ""
        emitted = self._content_history_tail.strip()
        candidate = cleaned.strip()
        if not emitted or not candidate:
            return cleaned
        if emitted.endswith(candidate):
            return ""
        if candidate.startswith(emitted):
            return candidate[len(emitted) :].lstrip()
        return cleaned

    def _consume_post_orphan_close_duplicate_prefix(self) -> bool:
        target = self._post_orphan_close_duplicate_tail
        if not target or not self._pending:
            return False
        changed = False
        while self._pending and target:
            if self._pending[0].isspace() and not target.startswith(self._pending[0]):
                self._pending = self._pending[1:]
                changed = True
                continue
            if target.startswith(self._pending):
                self._post_orphan_close_duplicate_tail = target[len(self._pending) :]
                self._pending = ""
                return True
            if self._pending.startswith(target):
                self._pending = self._pending[len(target) :].lstrip()
                self._post_orphan_close_duplicate_tail = ""
                return True
            common = 0
            max_common = min(len(self._pending), len(target))
            while (
                common < max_common
                and self._pending[common] == target[common]
            ):
                common += 1
            if common:
                self._pending = self._pending[common:]
                self._post_orphan_close_duplicate_tail = target[common:]
                changed = True
                continue
            self._post_orphan_close_duplicate_tail = ""
            return changed
        return changed

    def _strip_stream_sentinels_from_pending(self) -> bool:
        sentinel_cleaned = _strip_generated_chat_template_sentinels(self._pending)
        if sentinel_cleaned == self._pending:
            return False
        turn_sentinel_cleaned = CHAT_TEMPLATE_TURN_SENTINEL_RE.sub("", self._pending)
        if turn_sentinel_cleaned != self._pending:
            self._saw_chat_template_sentinel = True
        self._pending = sentinel_cleaned
        return True

    def _drain_disabled(self, *, final: bool) -> list[tuple[str, str]]:
        chunks: list[tuple[str, str]] = []
        while self._pending:
            pending_lower = self._pending.lower()
            if not final and any(
                marker.lower().startswith(pending_lower)
                for marker in CHAT_TEMPLATE_SENTINEL_MARKERS
            ):
                break
            if self._strip_stream_sentinels_from_pending():
                if not self._pending:
                    break
                continue
            if self._consume_post_orphan_close_duplicate_prefix():
                if not self._pending:
                    break
                continue
            if self._disabled_inside_reasoning:
                close_match = QWEN_STYLE_REASONING_CLOSE_RE.search(self._pending)
                if close_match is None:
                    if final:
                        self._pending = ""
                    else:
                        hold = self._disabled_reasoning_close_tail_len(self._pending)
                        self._pending = self._pending[-hold:] if hold else ""
                    break
                self._pending = self._pending[close_match.end() :].lstrip()
                self._disabled_inside_reasoning = False
                self._disabled_visible_started = True
                continue
            block_cleaned = QWEN_STYLE_REASONING_BLOCK_RE.sub("", self._pending)
            if block_cleaned != self._pending:
                self._pending = block_cleaned
                if not self._pending:
                    break
                continue
            close_match = QWEN_STYLE_REASONING_CLOSE_RE.search(self._pending)
            open_match = QWEN_STYLE_REASONING_OPEN_RE.search(self._pending)
            if close_match is not None and (
                open_match is None or close_match.start() <= open_match.start()
            ):
                after_close = self._pending[close_match.end() :].lstrip()
                if self._disabled_visible_started:
                    after_close = self._drop_duplicate_after_orphan_close(after_close)
                    if not after_close and self._content_history_tail.strip():
                        self._post_orphan_close_duplicate_tail = (
                            self._content_history_tail.strip()
                        )
                self._pending = after_close
                self._disabled_visible_started = True
                continue
            if open_match is not None:
                before = self._pending[: open_match.start()]
                if before:
                    self._append_chunk(chunks, "content", before)
                self._pending = self._pending[open_match.end() :]
                self._disabled_inside_reasoning = True
                self._reentry_count += 1
                continue
            if (
                not final
                and not self._disabled_visible_started
                and (
                    not (stripped := self._pending.lstrip())
                    or (
                        stripped.startswith("<")
                        and self._disabled_reasoning_tail_len(stripped)
                        >= len(stripped)
                    )
                )
            ):
                break
            if final:
                emit_len = len(self._pending)
            else:
                hold = self._disabled_reasoning_tail_len(self._pending)
                emit_len = len(self._pending) - hold
            if emit_len <= 0:
                break
            self._append_chunk(chunks, "content", self._pending[:emit_len])
            self._disabled_visible_started = True
            self._pending = self._pending[emit_len:]
            break
        return chunks

    def _drain(self, *, final: bool) -> list[tuple[str, str]]:
        chunks: list[tuple[str, str]] = []
        sentinel_keep = max(
            len(marker) + len("assistant") + 2
            for marker in CHAT_TEMPLATE_SENTINEL_MARKERS
        )
        tag_keep = max(len(name) for name in QWEN_STYLE_REASONING_TAG_NAMES) + len("</>")
        keep = max(
            tag_keep,
            sentinel_keep,
        )
        while self._pending:
            pending_lower = self._pending.lower()
            if not final and any(
                marker.lower().startswith(pending_lower)
                for marker in CHAT_TEMPLATE_SENTINEL_MARKERS
            ):
                break
            if self._strip_stream_sentinels_from_pending():
                if not self._pending:
                    break
                continue
            if self._inside_thinking:
                pending_lower = self._pending.lower()
                close_match = QWEN_STYLE_REASONING_CLOSE_RE.search(self._pending)
                close_index = -1 if close_match is None else close_match.start()
                tool_control_index = self._tool_control_marker_index(self._pending)
                if tool_control_index >= 0 and (
                    close_index < 0 or tool_control_index < close_index
                ):
                    if tool_control_index > 0:
                        self._append_chunk(
                            chunks,
                            "reasoning_content",
                            self._pending[:tool_control_index],
                        )
                        self._pending = self._pending[tool_control_index:]
                        continue
                    self._inside_thinking = False
                    self._inside_tool_call = True
                    self._tool_call_tail = ""
                    continue
                if (
                    not final
                    and pending_lower
                    and self._tool_control_marker_has_partial_prefix(pending_lower)
                ):
                    break
                open_match_at_start = QWEN_STYLE_REASONING_OPEN_RE.match(self._pending)
                if open_match_at_start is not None:
                    self._pending = self._pending[open_match_at_start.end() :]
                    self._reentry_count += 1
                    continue
                if (
                    not final
                    and self._reasoning_control_marker_has_partial_prefix(self._pending)
                ):
                    break
                if close_match is None:
                    emit_len = (
                        len(self._pending)
                        if final
                        else max(0, len(self._pending) - keep)
                    )
                    if emit_len <= 0:
                        break
                    self._append_chunk(
                        chunks, "reasoning_content", self._pending[:emit_len]
                    )
                    self._pending = self._pending[emit_len:]
                    break
                self._append_chunk(
                    chunks, "reasoning_content", self._pending[:close_index]
                )
                self._pending = self._pending[close_match.end() :].lstrip()
                self._inside_thinking = False
                continue

            open_match = QWEN_STYLE_REASONING_OPEN_RE.search(self._pending)
            if open_match is None:
                pending_lower = self._pending.lower()
                tool_close_index = pending_lower.find(self._TOOL_CALL_CLOSE_MARKER)
                tool_passthrough = (
                    self._inside_tool_call
                    or self._TOOL_CALL_MARKER in pending_lower
                )
                emit_len = (
                    len(self._pending)
                    if final or tool_passthrough
                    else (
                        tool_close_index + len(self._TOOL_CALL_CLOSE_MARKER)
                        if tool_close_index >= 0
                        else max(0, len(self._pending) - keep)
                    )
                )
                if emit_len <= 0:
                    break
                emitted = self._pending[:emit_len]
                if tool_passthrough:
                    self._tool_call_tail = (
                        self._tool_call_tail + emitted.lower()
                    )[-len(self._TOOL_CALL_CLOSE_MARKER) :]
                if self._TOOL_CALL_CLOSE_MARKER in emitted.lower() or (
                    tool_passthrough
                    and self._tool_call_tail.endswith(self._TOOL_CALL_CLOSE_MARKER)
                ):
                    self._inside_tool_call = False
                    self._tool_call_tail = ""
                self._append_chunk(chunks, "content", emitted)
                self._pending = self._pending[emit_len:]
                break
            self._append_chunk(chunks, "content", self._pending[: open_match.start()])
            self._pending = self._pending[open_match.end() :]
            self._inside_thinking = True
            self._reentry_count += 1
        return chunks


class _ThinkingContentStreamNormalizer(_ThinkingContentStreamSplitter):
    """Compatibility wrapper returning only text chunks."""

    def start(self) -> list[str]:
        return [text for _, text in super().start()]

    def feed(self, text: str) -> list[str]:
        return [chunk for _, chunk in super().feed(text)]

    def finish(self) -> list[str]:
        return [chunk for _, chunk in super().finish()]


def _stream_splitter_for_state(
    state: ServerState,
    *,
    thinking_enabled: bool,
    recover_unclosed_reasoning_as_content: bool = True,
    start_inside_thinking: bool = True,
) -> Any:
    parser = _reasoning_parser_for_state(state)
    if parser == "gemma4":
        return stream_splitter_for_parser(parser, thinking_enabled=thinking_enabled)
    return _ThinkingContentStreamSplitter(
        thinking_enabled=thinking_enabled,
        recover_unclosed_reasoning_as_content=recover_unclosed_reasoning_as_content,
        start_inside_thinking=start_inside_thinking,
    )


def _finish_stream_splitter(splitter: Any, *, recover_unclosed_reasoning: bool) -> list[tuple[str, str]]:
    try:
        return splitter.finish(
            recover_unclosed_reasoning_as_content=recover_unclosed_reasoning
        )
    except TypeError:
        return splitter.finish()


def _reasoning_completion_repair_needed(
    *,
    thinking_enabled: bool,
    reasoning_text: str,
    answer_text: str,
    assistant_tool_calls: list[dict[str, Any]] | None,
) -> bool:
    """Return true when a Qwen thinking turn ended before producing an answer.

    This is a protocol-completion check, not a content guardrail. Qwen's
    template opens ``<think>`` in the prompt, so a valid assistant turn must
    eventually produce answer content or a tool call after the thinking block.
    A stop token while only reasoning has streamed leaves OpenAI clients with
    an empty assistant message.
    """
    if not thinking_enabled:
        return False
    if assistant_tool_calls:
        return False
    return bool(reasoning_text.strip()) and not bool(answer_text.strip())


def _reasoning_completion_repair_prompt_ids(
    tokenizer: Any,
    prompt_ids: list[int],
    generated_tokens: list[int],
) -> list[int]:
    stop_token_ids = _default_stop_tokens(tokenizer)
    generated_without_stop = _strip_terminal_stop(generated_tokens, stop_token_ids)
    generated_text = tokenizer.decode(generated_without_stop)
    repair_suffix = "\n\n"
    if THINK_CLOSE not in generated_text:
        repair_suffix = f"\n{THINK_CLOSE}\n\n"
    return [
        *[int(token) for token in prompt_ids],
        *[int(token) for token in generated_without_stop],
        *_encode_rendered_chat_text(tokenizer, repair_suffix),
    ]


def _display_text(
    state: ServerState,
    generated: dict[str, Any],
    *,
    thinking_enabled: bool = False,
) -> str:
    raw_text = str(generated["text"])
    text = (
        _normalize_reasoning_tags_for_state(
            state,
            raw_text,
            thinking_enabled=thinking_enabled,
        )
        if state.args.normalize_thinking_tags
        else raw_text
    )
    if not state.args.stats_footer:
        return text
    footer = _stats_footer_text(state, generated)
    separator = "\n\n" if text.endswith("\n") else "\n\n"
    return f"{text}{separator}{footer}"


def _nonstream_chat_message_parts(
    state: ServerState,
    generated: dict[str, Any],
    *,
    thinking_enabled: bool,
    suppress_visible_reasoning: bool = False,
) -> tuple[str, str]:
    raw_text = _strip_generated_chat_template_sentinels(
        str(generated.get("text") or "")
    )
    reasoning_text = ""
    display_text = raw_text
    parser = _reasoning_parser_for_state(state)
    parser_enabled = parser != "none"
    has_qwen_style_reasoning_marker = bool(
        QWEN_STYLE_REASONING_CONTROL_RE.search(raw_text)
    )
    # Qwen opens <think> in the prompt, so non-stream generations often begin
    # inside hidden reasoning and only emit the closing tag. If a thinking tag
    # is present anywhere in the final text, route it like the streaming path:
    # reasoning_content is reasoning, message.content is the visible answer.
    if parser_enabled and parser == "gemma4":
        reasoning_text, display_text = _split_backend_reasoning_for_state(
            state,
            raw_text,
            thinking_enabled=thinking_enabled,
        )
        stats = generated.setdefault("stats", {})
        stats["reasoning_tokens"] = _count_text_tokens(
            state.runtime.tokenizer,
            reasoning_text,
        )
        stats["answer_tokens"] = _count_text_tokens(
            state.runtime.tokenizer,
            display_text,
        )
        stats["nonstream_reasoning_content_routed"] = bool(reasoning_text)
        stats["visible_reasoning_stripped"] = bool(
            reasoning_text and display_text != raw_text
        )
    elif parser_enabled and parser in {"qwen3", "step3p5"}:
        if thinking_enabled and has_qwen_style_reasoning_marker:
            reasoning_text, display_text = _split_backend_reasoning_for_state(
                state,
                raw_text,
                thinking_enabled=True,
            )
        elif not thinking_enabled:
            reasoning_text = ""
            display_text = strip_qwen_style_reasoning_from_content(raw_text)
        stats = generated.setdefault("stats", {})
        if reasoning_text or display_text != raw_text:
            stats["reasoning_tokens"] = _count_text_tokens(
                state.runtime.tokenizer,
                reasoning_text,
            )
            stats["answer_tokens"] = _count_text_tokens(
                state.runtime.tokenizer,
                display_text,
            )
            stats["nonstream_reasoning_content_routed"] = bool(reasoning_text)
            stats["visible_reasoning_stripped"] = bool(display_text != raw_text)
    elif thinking_enabled and parser_enabled and (THINK_OPEN in raw_text or THINK_CLOSE in raw_text):
        reasoning_text, display_text = _split_thinking_segments(
            raw_text,
            thinking_enabled=True,
        )
        stats = generated.setdefault("stats", {})
        stats["reasoning_tokens"] = _count_text_tokens(
            state.runtime.tokenizer,
            reasoning_text,
        )
        stats["answer_tokens"] = _count_text_tokens(
            state.runtime.tokenizer,
            display_text,
        )
        stats["nonstream_reasoning_content_routed"] = bool(reasoning_text)
        stats["visible_reasoning_stripped"] = bool(
            reasoning_text and display_text != raw_text
        )
    elif getattr(state.args, "normalize_thinking_tags", False):
        display_text = _normalize_reasoning_tags_for_state(
            state,
            raw_text,
            thinking_enabled=thinking_enabled,
        )

    display_text = _strip_mtplx_internal_continuation_markers(display_text)
    if suppress_visible_reasoning:
        reasoning_text = ""
    if not getattr(state.args, "stats_footer", False):
        return display_text, reasoning_text
    footer = _stats_footer_text(state, generated)
    if not footer:
        return display_text, reasoning_text
    return f"{display_text}\n\n{footer}", reasoning_text


def _chat_ui_html(
    *,
    model_id: str,
    server_url: str,
    api_key_required: bool,
    default_settings: dict[str, Any],
) -> str:
    api_note = "API key required" if api_key_required else "local · no API key"
    depth_max = max(1, int(default_settings.get("depth_max", 3) or 3))
    default_depth = max(1, min(depth_max, int(default_settings.get("depth", 3))))
    default_settings = {
        "mtp_enabled": bool(default_settings.get("mtp_enabled", True)),
        "temperature": float(default_settings.get("temperature", 0.6)),
        "top_p": float(default_settings.get("top_p", 0.95)),
        "top_k": int(default_settings.get("top_k", 20)),
        "depth": default_depth,
        "max_tokens": int(default_settings.get("max_tokens", 16384)),
        "reasoning": str(default_settings.get("reasoning", "auto")),
        "system": str(default_settings.get("system", "")),
    }
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>MTPLX</title>
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Cdefs%3E%3ClinearGradient id='g' x1='0' x2='1' y1='0' y2='1'%3E%3Cstop offset='0' stop-color='%23a5b4fc'/%3E%3Cstop offset='1' stop-color='%235b8dee'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='32' height='32' rx='9' fill='url(%23g)'/%3E%3Ctext x='50%25' y='59%25' font-family='-apple-system,Segoe UI,sans-serif' font-size='15' font-weight='800' fill='white' text-anchor='middle'%3EM%3C/text%3E%3C/svg%3E">
  <style>
    :root {
      color-scheme: dark;
      --bg: #0a0b0d;
      --surface: #14161a;
      --surface-2: #1a1d22;
      --line: rgba(255, 255, 255, 0.07);
      --line-strong: rgba(255, 255, 255, 0.14);
      --text: #ececec;
      --muted: #9ca3af;
      --muted-2: #6b7280;
      --accent: #5b8dee;
      --accent-strong: #7aa3f3;
      --accent-soft: rgba(91, 141, 238, 0.10);
      --user-tint: rgba(91, 141, 238, 0.09);
      --reason-bg: #1a1726;
      --reason-text: #c5b8e0;
      --reason-border: rgba(165, 132, 245, 0.22);
      --code-bg: #0a0b0d;
      --code-border: rgba(255, 255, 255, 0.06);
      --ok: #5fd28b;
      --warn: #f0b85c;
      --error: #ef6868;
      --font-sans: -apple-system, BlinkMacSystemFont, "Inter", "SF Pro Text", "Segoe UI", system-ui, sans-serif;
      --font-mono: "SF Mono", ui-monospace, Menlo, Monaco, "Cascadia Code", Consolas, monospace;
      --sidebar-w: 268px;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: var(--font-sans);
      font-size: 15px;
      line-height: 1.6;
      display: grid;
      grid-template-rows: 48px 1fr;
      grid-template-columns: var(--sidebar-w) 1fr;
      grid-template-areas:
        "topbar topbar"
        "sidebar chat";
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }
    @media (max-width: 900px) {
      body { grid-template-columns: 1fr; grid-template-areas: "topbar" "chat"; }
      aside#sidebar { display: none; }
      aside#sidebar.open { display: block; position: fixed; inset: 48px 0 0 0; z-index: 30; }
    }

    /* Topbar */
    header.topbar {
      grid-area: topbar;
      padding: 0 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(20, 22, 26, 0.85);
      backdrop-filter: blur(12px);
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      font-weight: 700;
      font-size: 14px;
      letter-spacing: 0.2px;
    }
    .brand .logo {
      width: 24px;
      height: 24px;
      border-radius: 7px;
      background: linear-gradient(135deg, #a5b4fc, var(--accent));
      display: flex; align-items: center; justify-content: center;
      color: white; font-weight: 800; font-size: 13px;
    }
    .topbar-meta { color: var(--muted); font-size: 13px; flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .topbar-meta .sep { color: var(--muted-2); margin: 0 8px; }
    .topbar-meta .dim { color: var(--muted-2); }
    .topbar-actions { display: flex; align-items: center; gap: 4px; }
    .runtime-pill {
      display: inline-flex; align-items: center;
      min-height: 24px;
      padding: 0 9px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      background: var(--surface);
      font-size: 12px;
      font-weight: 600;
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
    }
    .runtime-pill[hidden] { display: none; }
    .icon-btn {
      width: 32px; height: 32px;
      border: 0; border-radius: 8px;
      background: transparent; color: var(--muted);
      cursor: pointer;
      display: inline-flex; align-items: center; justify-content: center;
      transition: background 0.12s, color 0.12s;
    }
    .icon-btn:hover { background: var(--surface-2); color: var(--text); }
    .icon-btn svg { width: 16px; height: 16px; }

    /* Sidebar */
    aside#sidebar {
      grid-area: sidebar;
      background: var(--surface);
      border-right: 1px solid var(--line);
      overflow-y: auto;
      padding: 18px 16px 24px;
    }
    .sb-section { margin-bottom: 20px; }
    .sb-title {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      color: var(--muted-2);
      margin: 0 0 10px;
      padding: 0 2px;
    }
    .sb-row { margin-bottom: 14px; padding: 0 2px; }
    .sb-row label {
      display: flex; justify-content: space-between; align-items: baseline;
      font-size: 13px; color: var(--text); margin-bottom: 6px;
    }
    .sb-row label .v {
      color: var(--muted);
      font-weight: 600; font-size: 12px;
      font-variant-numeric: tabular-nums;
    }
    .sb-row textarea, .sb-row select {
      width: 100%;
      background: var(--bg);
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--text);
      padding: 9px 11px;
      font: inherit; font-size: 13px;
      outline: none;
      transition: border-color 0.12s, box-shadow 0.12s;
    }
    .sb-row textarea:focus, .sb-row select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-soft);
    }
    .sb-row textarea { min-height: 64px; max-height: 200px; resize: vertical; line-height: 1.45; }
    .sb-row .help { color: var(--muted-2); font-size: 11px; margin-top: 5px; line-height: 1.4; }
    .switch-row {
      display: flex; align-items: center; justify-content: space-between;
      gap: 12px; margin-bottom: 12px;
    }
    .switch-row label { margin: 0; display: inline-flex; align-items: baseline; gap: 6px; }
    .switch {
      position: relative; display: inline-flex; align-items: center;
      width: 40px; height: 24px; flex: 0 0 auto;
    }
    .switch input { opacity: 0; width: 0; height: 0; }
    .switch .track {
      position: absolute; inset: 0;
      border-radius: 999px;
      background: var(--surface-2);
      border: 1px solid var(--line);
      transition: background 0.12s, border-color 0.12s;
    }
    .switch .thumb {
      position: absolute; top: 4px; left: 4px;
      width: 16px; height: 16px; border-radius: 50%;
      background: var(--muted);
      transition: transform 0.12s, background 0.12s;
    }
    .switch input:checked + .track { background: var(--accent); border-color: var(--accent); }
    .switch input:checked + .track + .thumb { transform: translateX(16px); background: white; }
    .switch input:focus-visible + .track { box-shadow: 0 0 0 3px var(--accent-soft); }
    .sb-row select {
      appearance: none; -webkit-appearance: none; padding-right: 28px;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 10 10'%3E%3Cpath d='M2 4l3 3 3-3' stroke='%239ca3af' stroke-width='1.4' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
      background-repeat: no-repeat; background-position: right 10px center; background-size: 10px;
    }

    /* Slider — thin track + dot thumb (Open WebUI style) */
    input[type="range"] {
      -webkit-appearance: none; appearance: none;
      width: 100%; height: 18px;
      background: transparent; cursor: pointer;
      margin: 0;
    }
    input[type="range"]::-webkit-slider-runnable-track {
      height: 3px; border-radius: 999px;
      background: linear-gradient(to right, var(--accent) 0%, var(--accent) var(--filled, 0%), var(--surface-2) var(--filled, 0%), var(--surface-2) 100%);
    }
    input[type="range"]::-moz-range-track { height: 3px; border-radius: 999px; background: var(--surface-2); }
    input[type="range"]::-moz-range-progress { height: 3px; border-radius: 999px; background: var(--accent); }
    input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 14px; height: 14px;
      border-radius: 50%;
      background: white; border: 0;
      margin-top: -5.5px;
      box-shadow: 0 1px 4px rgba(0, 0, 0, 0.25);
      cursor: grab;
      transition: transform 0.1s;
    }
    input[type="range"]:active::-webkit-slider-thumb { transform: scale(1.18); cursor: grabbing; }
    input[type="range"]::-moz-range-thumb {
      width: 14px; height: 14px; border-radius: 50%;
      background: white; border: 0;
      box-shadow: 0 1px 4px rgba(0, 0, 0, 0.25);
      cursor: grab;
    }
    input[type="range"]:disabled { cursor: default; opacity: 0.45; }

    .sb-actions { margin-top: 18px; padding-top: 14px; border-top: 1px solid var(--line); }
    .sb-btn {
      width: 100%;
      background: transparent; border: 1px solid var(--line);
      color: var(--muted);
      padding: 9px 12px; border-radius: 8px;
      cursor: pointer; font: inherit; font-size: 12px;
      transition: border-color 0.12s, color 0.12s, background 0.12s;
    }
    .sb-btn:hover { border-color: var(--line-strong); color: var(--text); background: var(--surface-2); }

    /* Chat area */
    main.chat-area {
      grid-area: chat; min-height: 0;
      display: grid; grid-template-rows: 1fr auto;
      position: relative; overflow: hidden;
    }
    #messages { overflow-y: auto; scroll-behavior: auto; padding: 24px 24px 8px; }
    .messages-inner { max-width: 760px; margin: 0 auto; }
    #messages-bottom { height: 1px; }

    /* Turns — borderless, ChatGPT/Open-WebUI style */
    .turn { padding: 14px 0; }
    .turn-user { display: flex; justify-content: flex-end; }
    .turn-user .turn-body {
      max-width: 85%;
      background: var(--user-tint);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px 16px;
      white-space: pre-wrap;
    }
    .turn-assistant {
      display: grid;
      grid-template-columns: 32px 1fr;
      gap: 14px; align-items: start;
    }
    .avatar {
      width: 32px; height: 32px;
      border-radius: 9px;
      background: linear-gradient(135deg, #a5b4fc, var(--accent));
      display: flex; align-items: center; justify-content: center;
      color: white; font-weight: 800; font-size: 13px;
      letter-spacing: 0.5px; flex-shrink: 0; margin-top: 2px;
    }
    .turn-assistant .turn-body { min-width: 0; }

    /* Reasoning is its own card ABOVE the answer, not nested inside */
    .reasoning-block {
      margin: 0 0 12px;
      background: var(--reason-bg);
      border: 1px solid var(--reason-border);
      border-radius: 10px;
      overflow: hidden;
    }
    .reasoning-block[hidden] { display: none; }
    .reasoning-summary {
      cursor: pointer;
      padding: 10px 14px;
      color: var(--reason-text);
      font-size: 12px; font-weight: 600;
      letter-spacing: 0.2px;
      display: flex; align-items: center; gap: 8px;
      user-select: none;
    }
    .reasoning-summary .chev {
      width: 10px; height: 10px;
      transition: transform 0.15s;
      opacity: 0.75;
    }
    .reasoning-block.open .reasoning-summary .chev { transform: rotate(90deg); }
    .reasoning-summary .label { flex: 1; }
    .reasoning-summary .meta { color: var(--muted-2); font-size: 11px; font-weight: 500; }
    .reasoning-body {
      padding: 0 14px 12px;
      color: var(--reason-text);
      font-size: 13px; line-height: 1.6;
      white-space: pre-wrap;
      max-height: 320px; overflow-y: auto;
    }
    .reasoning-block:not(.open) .reasoning-body { display: none; }

    /* Answer body — markdown rendered */
    .answer { color: var(--text); }
    .answer.streaming-plain { white-space: pre-wrap; }
    .answer p { margin: 0 0 0.75em; }
    .answer p:last-child { margin-bottom: 0; }
    .answer h1, .answer h2, .answer h3, .answer h4 { margin: 1em 0 0.5em; line-height: 1.3; font-weight: 700; }
    .answer h1 { font-size: 1.5em; }
    .answer h2 { font-size: 1.3em; }
    .answer h3 { font-size: 1.13em; }
    .answer ul, .answer ol { padding-left: 1.4em; margin: 0.4em 0 0.8em; }
    .answer li { margin: 0.2em 0; }
    .answer a { color: var(--accent-strong); text-decoration: underline; text-underline-offset: 3px; text-decoration-thickness: 1px; }
    .answer a:hover { color: var(--accent); }
    .answer blockquote {
      border-left: 3px solid var(--line-strong);
      margin: 0.6em 0; padding: 0.2em 0 0.2em 1em;
      color: var(--muted);
    }
    .answer code {
      font-family: var(--font-mono); font-size: 0.92em;
      background: var(--code-bg);
      border: 1px solid var(--code-border);
      border-radius: 4px;
      padding: 1.5px 6px;
    }
    .answer pre {
      position: relative;
      margin: 0.9em 0; padding: 0;
      border: 1px solid var(--code-border);
      border-radius: 10px;
      background: var(--code-bg);
      overflow: hidden;
    }
    .answer pre code {
      display: block;
      padding: 12px 14px;
      font-size: 13px; line-height: 1.6;
      overflow-x: auto;
      border: 0; background: transparent; border-radius: 0;
      white-space: pre;
    }
    .copy-btn {
      position: absolute; top: 7px; right: 7px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 11px; font-weight: 500;
      padding: 3px 9px; border-radius: 5px;
      cursor: pointer;
      opacity: 0;
      transition: opacity 0.12s, color 0.12s, border-color 0.12s;
    }
    .answer pre:hover .copy-btn { opacity: 1; }
    .copy-btn:hover { color: var(--text); border-color: var(--line-strong); }
    .copy-btn.copied { color: var(--ok); border-color: rgba(95, 210, 139, 0.4); }

    /* Stats below the answer — minimal inline pills, no border boxes */
    .stats {
      margin-top: 10px;
      display: flex; flex-wrap: wrap;
      gap: 4px 12px;
      font-size: 12px;
      color: var(--muted-2);
      font-variant-numeric: tabular-nums;
    }
    .stats[hidden] { display: none; }
    .stats .stat-tps { color: var(--ok); font-weight: 600; }
    .stats .stat-ttft { color: var(--muted); }

    /* Composer */
    .composer-wrap {
      padding: 12px 24px 18px;
      background: linear-gradient(to top, var(--bg) 0%, var(--bg) 75%, transparent);
    }
    .composer { max-width: 760px; margin: 0 auto; position: relative; }
    #status-row {
      min-height: 18px;
      display: flex; align-items: center; gap: 8px;
      font-size: 12px; color: var(--muted-2);
      padding: 0 4px 6px;
    }
    #status-row.ready { color: var(--muted); }
    #status-row.streaming { color: var(--accent-strong); }
    #status-row .dot {
      width: 6px; height: 6px;
      border-radius: 50%;
      background: currentColor; flex-shrink: 0;
    }
    #live-stats { margin-left: auto; color: var(--muted-2); font-variant-numeric: tabular-nums; }
    #live-stats .tps { color: var(--ok); font-weight: 600; }

    .composer-box {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px; align-items: end;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 8px 8px 8px 14px;
      transition: border-color 0.12s, box-shadow 0.12s;
    }
    .composer-box:focus-within { border-color: var(--line-strong); }
    #prompt {
      min-height: 38px; max-height: 220px;
      resize: none;
      border: 0;
      background: transparent;
      color: var(--text);
      padding: 8px 0;
      font: inherit; font-size: 15px; line-height: 1.5;
      outline: none;
    }
    #prompt::placeholder { color: var(--muted-2); }
    .send-btn {
      width: 36px; height: 36px;
      border: 0; border-radius: 9px;
      background: var(--accent); color: white;
      cursor: pointer;
      display: inline-flex; align-items: center; justify-content: center;
      transition: background 0.12s, transform 0.05s;
    }
    .send-btn:hover { background: var(--accent-strong); }
    .send-btn:active { transform: scale(0.95); }
    .send-btn:disabled { opacity: 0.45; cursor: default; }
    .send-btn svg { width: 16px; height: 16px; }
    .send-btn.stop { background: var(--error); }
    .send-btn.stop:hover { background: #f47e7e; }

    @media (max-width: 900px) {
      header.topbar { padding: 0 12px; }
      .topbar-meta { display: none; }
      .runtime-pill { max-width: 45vw; overflow: hidden; text-overflow: ellipsis; }
      #messages { padding: 18px 14px 6px; }
      .composer-wrap { padding: 10px 14px 14px; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <span class="brand"><span class="logo">M</span>MTPLX</span>
    <span class="topbar-meta"><span>__MODEL__</span><span class="sep">·</span><span class="dim">__API_NOTE__</span><span class="sep">·</span><span class="dim">__SERVER_URL__/v1</span></span>
    <div class="topbar-actions">
      <span id="runtime-pill" class="runtime-pill" hidden>Runtime</span>
      <button id="sidebar-toggle" class="icon-btn" title="Toggle settings" aria-label="Toggle settings">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
      </button>
      <button id="new-chat-btn" class="icon-btn" title="Start a new conversation" aria-label="New conversation">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14M5 12h14"/></svg>
      </button>
    </div>
  </header>
  <aside id="sidebar">
    <div class="sb-section">
      <p class="sb-title">Sampling</p>
      <div class="sb-row">
        <label for="ctl-temp">Temperature <span class="v" id="val-temp">0.60</span></label>
        <input id="ctl-temp" type="range" min="0" max="2" step="0.05" value="0.6">
      </div>
      <div class="sb-row">
        <label for="ctl-top-p">Top P <span class="v" id="val-top-p">0.95</span></label>
        <input id="ctl-top-p" type="range" min="0" max="1" step="0.01" value="0.95">
      </div>
      <div class="sb-row">
        <label for="ctl-top-k">Top K <span class="v" id="val-top-k">20</span></label>
        <input id="ctl-top-k" type="range" min="0" max="100" step="1" value="20">
        <p class="help">0 disables top-k.</p>
      </div>
    </div>
    <div class="sb-section">
      <p class="sb-title">Speculative</p>
      <div class="sb-row switch-row">
        <label for="ctl-mtp">MTP <span class="v" id="val-mtp">on</span></label>
        <span class="switch">
          <input id="ctl-mtp" type="checkbox" checked>
          <span class="track"></span>
          <span class="thumb"></span>
        </span>
      </div>
      <div class="sb-row">
        <label for="ctl-depth">Draft depth <span class="v" id="val-depth">__DEPTH_VALUE__</span></label>
        <input id="ctl-depth" type="range" min="1" max="__DEPTH_MAX__" step="1" value="__DEPTH_VALUE__">
        <p class="help">MTP draft tokens per verify cycle.</p>
      </div>
    </div>
    <div class="sb-section">
      <p class="sb-title">Output</p>
      <div class="sb-row">
        <label for="ctl-max-tokens">Max tokens <span class="v" id="val-max-tokens">8k</span></label>
        <input id="ctl-max-tokens" type="range" min="256" max="32768" step="256" value="8192">
        <p class="help" id="max-tokens-help">Detecting context length…</p>
      </div>
      <div class="sb-row">
        <label for="ctl-think">Reasoning</label>
        <select id="ctl-think">
          <option value="auto" selected>Auto</option>
          <option value="on">Always show thinking</option>
          <option value="off">Hide thinking</option>
        </select>
      </div>
    </div>
    <div class="sb-section">
      <p class="sb-title">System prompt</p>
      <div class="sb-row">
        <textarea id="ctl-system" placeholder="Optional. Overrides the model default."></textarea>
      </div>
    </div>
    <div class="sb-actions">
      <button id="reset-defaults" class="sb-btn" type="button">Reset to defaults</button>
    </div>
  </aside>
  <main class="chat-area">
    <section id="messages" aria-live="polite">
      <div class="messages-inner">
        <div class="turn turn-assistant turn-greeting">
          <div class="avatar">M</div>
          <div class="turn-body">
            <div class="answer"><p>Ready when you are. Settings mirror the running MTPLX app.</p></div>
          </div>
        </div>
      </div>
      <div id="messages-bottom" aria-hidden="true"></div>
    </section>
    <div class="composer-wrap">
      <div class="composer">
        <div id="status-row" class="ready" role="status">
          <span class="dot"></span>
          <span id="status-text">Ready</span>
          <span id="live-stats"></span>
        </div>
        <form id="chat-form" autocomplete="off">
          <div class="composer-box">
            <textarea id="prompt" placeholder="Message MTPLX (Enter to send · Shift+Enter for newline)" rows="1" autofocus></textarea>
            <button id="send" class="send-btn" type="submit" aria-label="Send">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 5l7 7-7 7"/></svg>
            </button>
          </div>
        </form>
      </div>
    </div>
  </main>
  <script src="https://cdn.jsdelivr.net/npm/marked@15/marked.min.js" defer></script>
  <script>
    "use strict";
    const MODEL_ID = __MODEL_JSON__;
    const messagesEl = document.getElementById("messages");
    const messagesInner = messagesEl.querySelector(".messages-inner");
    const messagesBottom = document.getElementById("messages-bottom");
    const form = document.getElementById("chat-form");
    const promptEl = document.getElementById("prompt");
    const sendBtn = document.getElementById("send");
    const statusRow = document.getElementById("status-row");
    const statusText = document.getElementById("status-text");
    const liveStatsEl = document.getElementById("live-stats");
    const newChatBtn = document.getElementById("new-chat-btn");
    const sidebarToggleBtn = document.getElementById("sidebar-toggle");
    const sidebarEl = document.getElementById("sidebar");
    const runtimePillEl = document.getElementById("runtime-pill");
    const history = [];
    let activeAbort = null;
    let pinnedToBottom = true;
    let forceAutoScroll = false;
    let scrollFrame = null;
    let postLayoutScrollTimer = null;

    const SVG_SEND = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 5l7 7-7 7"/></svg>';
    const SVG_STOP = '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>';

    // ---------- settings ------------------------------------------------------
    const SETTINGS_KEY = "mtplx.chat.settings.v5:" + MODEL_ID;
    const LEGACY_SETTINGS_KEY = "mtplx.chat.settings.v4";
    const DEFAULTS = __DEFAULT_SETTINGS_JSON__;
    // RANGES is mutable so we can rewrite max_tokens.max after we discover
    // the model's real context window via /health. Hardcoding a 32768 cap
    // (our previous default) lied about a 256k-context model and stopped
    // users from raising the answer budget for long replies.
    const RANGES = {
      temperature: {min: 0, max: 2},
      top_p: {min: 0, max: 1},
      top_k: {min: 0, max: 100},
      depth: {min: 1, max: __DEPTH_MAX__},
      max_tokens: {min: 256, max: 32768}
    };
    const maxTokensHelpEl = document.getElementById("max-tokens-help");
    function formatTokens(n) {
      if (n >= 1000) return (n / 1000).toFixed(1).replace(/\\.0$/, "") + "k";
      return String(n);
    }
    function runtimeLabelFromHealth(health) {
      if (health && health.runtime_mode) return String(health.runtime_mode);
      const profileName = health && health.profile && health.profile.name ? String(health.profile.name) : "";
      const mode = String((health && health.generation_mode) || "").toLowerCase() === "ar" ? "AR" : "MTP";
      const fanBoost = Boolean(health && (health.fan_boost_active || health.fan_mode === "max"));
      if (profileName === "sustained" && fanBoost) return "Sustained Max " + mode;
      if (profileName === "sustained") return "Sustained " + mode;
      if (profileName === "performance-cold" && fanBoost) return "Burst " + mode;
      if (profileName === "performance-cold") return "Performance-cold " + mode;
      if (profileName === "stable") return "Stable " + mode;
      return profileName ? profileName + " " + mode : mode;
    }
    function applyRuntimeHealth(health) {
      if (!runtimePillEl || !health) return;
      runtimePillEl.textContent = runtimeLabelFromHealth(health);
      runtimePillEl.hidden = false;
      runtimePillEl.title = runtimePillEl.textContent;
    }
    async function discoverServerLimits() {
      try {
        const res = await fetch("/health", {cache: "no-store"});
        if (!res.ok) throw new Error("health " + res.status);
        const health = await res.json();
        applyRuntimeHealth(health);
        const ctx = parseInt(health.context_window, 10);
        const serverCap = parseInt(health.max_response_tokens, 10);
        if (Number.isFinite(ctx) && ctx > 0) {
          // Allow up to (context - some headroom) tokens of output. We leave
          // 4k for the prompt so users typing a long question don't get a
          // 400 from the server even at the slider's max.
          let cap = Math.max(1024, ctx - 4096);
          if (Number.isFinite(serverCap) && serverCap > 0) cap = Math.min(cap, serverCap);
          // Round down to a clean step.
          cap = Math.floor(cap / 256) * 256;
          RANGES.max_tokens.max = cap;
          if (ctlEls.max_tokens) {
            ctlEls.max_tokens.max = String(cap);
            // Re-clamp the current value so a saved value above the new cap
            // doesn't render off the right edge of the slider.
            ctlEls.max_tokens.value = String(Math.min(parseInt(ctlEls.max_tokens.value, 10) || DEFAULTS.max_tokens, cap));
          }
          if (maxTokensHelpEl) {
            maxTokensHelpEl.textContent =
              "Cap is the model's " + formatTokens(ctx) + " context (slider tops out at " +
              formatTokens(cap) + ").";
          }
          refreshLabels();
          refreshSliderFills();
        }
      } catch (err) {
        if (maxTokensHelpEl) maxTokensHelpEl.textContent =
          "Could not detect context length; using a 32k default.";
      }
    }
    const ctlEls = {
      temperature: document.getElementById("ctl-temp"),
      top_p: document.getElementById("ctl-top-p"),
      top_k: document.getElementById("ctl-top-k"),
      mtp_enabled: document.getElementById("ctl-mtp"),
      depth: document.getElementById("ctl-depth"),
      max_tokens: document.getElementById("ctl-max-tokens"),
      reasoning: document.getElementById("ctl-think"),
      system: document.getElementById("ctl-system")
    };
    const valEls = {
      temperature: document.getElementById("val-temp"),
      top_p: document.getElementById("val-top-p"),
      top_k: document.getElementById("val-top-k"),
      mtp_enabled: document.getElementById("val-mtp"),
      depth: document.getElementById("val-depth"),
      max_tokens: document.getElementById("val-max-tokens")
    };
    function loadStoredSystemPrompt() {
      try {
        const raw = window.localStorage.getItem(SETTINGS_KEY);
        const fallback = window.localStorage.getItem(LEGACY_SETTINGS_KEY);
        const parsed = JSON.parse(raw || fallback || "{}");
        return parsed && typeof parsed.system === "string" ? parsed.system : "";
      } catch (err) {
        console.warn("settings prompt load failed", err);
        return "";
      }
    }
    function loadSettings() {
      return Object.assign({}, DEFAULTS, {system: loadStoredSystemPrompt()});
    }
    function settingsFromDaemonPayload(payload) {
      payload = payload || {};
      const rawMode = payload.generation_mode == null ? "" : String(payload.generation_mode);
      const mode = rawMode.toLowerCase();
      if (payload.depth_max) {
        RANGES.depth.max = Math.max(1, parseInt(payload.depth_max, 10) || RANGES.depth.max);
        if (ctlEls.depth) ctlEls.depth.max = String(RANGES.depth.max);
      }
      return Object.assign({}, DEFAULTS, {
        temperature: payload.temperature,
        top_p: payload.top_p,
        top_k: payload.top_k,
        mtp_enabled: mode ? mode === "mtp" : DEFAULTS.mtp_enabled,
        depth: payload.depth,
        max_tokens: payload.max_response_tokens == null ? DEFAULTS.max_tokens : payload.max_response_tokens,
        reasoning: payload.reasoning || DEFAULTS.reasoning,
        system: loadStoredSystemPrompt()
      });
    }
    function daemonSettingsPayload(s) {
      const normalized = normalizeSettings(s || {});
      return {
        temperature: normalized.temperature,
        top_p: normalized.top_p,
        top_k: normalized.top_k,
        generation_mode: normalized.mtp_enabled ? "mtp" : "ar",
        depth: normalized.depth,
        max_response_tokens: normalized.max_tokens,
        reasoning: normalized.reasoning
      };
    }
    function daemonSettingsSignature(s) {
      return JSON.stringify(daemonSettingsPayload(s));
    }
    async function fetchDaemonSettings() {
      const res = await fetch("/v1/mtplx/settings", {cache: "no-store"});
      if (!res.ok) throw new Error("settings " + res.status);
      const payload = await res.json();
      return settingsFromDaemonPayload(payload);
    }
    function saveSettings(s) {
      try {
        window.localStorage.setItem(
          SETTINGS_KEY,
          JSON.stringify({system: String((s && s.system) || "")})
        );
      } catch (_e) { /* ignore quota */ }
    }
    function clamp(value, min, max, fallback, isInt) {
      const n = isInt ? parseInt(value, 10) : parseFloat(value);
      if (!Number.isFinite(n)) return fallback;
      return Math.min(max, Math.max(min, n));
    }
    function normalizeSettings(s) {
      s = s || {};
      return {
        temperature: clamp(s.temperature, RANGES.temperature.min, RANGES.temperature.max, DEFAULTS.temperature, false),
        top_p: clamp(s.top_p, RANGES.top_p.min, RANGES.top_p.max, DEFAULTS.top_p, false),
        top_k: clamp(s.top_k, RANGES.top_k.min, RANGES.top_k.max, DEFAULTS.top_k, true),
        mtp_enabled: s.mtp_enabled == null ? DEFAULTS.mtp_enabled !== false : s.mtp_enabled !== false,
        depth: clamp(s.depth, RANGES.depth.min, RANGES.depth.max, DEFAULTS.depth, true),
        max_tokens: clamp(s.max_tokens, RANGES.max_tokens.min, RANGES.max_tokens.max, DEFAULTS.max_tokens, true),
        reasoning: ["auto", "on", "off"].includes(String(s.reasoning || "")) ? String(s.reasoning) : DEFAULTS.reasoning,
        system: String(s.system || "")
      };
    }
    function applySettingsToUI(s) {
      const normalized = normalizeSettings(s);
      ctlEls.temperature.value = normalized.temperature;
      ctlEls.top_p.value = normalized.top_p;
      ctlEls.top_k.value = normalized.top_k;
      ctlEls.mtp_enabled.checked = normalized.mtp_enabled;
      ctlEls.depth.value = normalized.depth;
      ctlEls.max_tokens.value = normalized.max_tokens;
      ctlEls.reasoning.value = normalized.reasoning;
      ctlEls.system.value = normalized.system;
      refreshLabels();
      refreshSliderFills();
    }
    let lastSyncedSettingsSignature = "";
    let settingsSyncTimer = null;
    let settingsSyncSeq = 0;
    let lastLocalSettingsEditAt = 0;
    async function refreshDaemonSettings(options) {
      const opts = options || {};
      if (!opts.force) {
        const recentlyEdited = performance.now() - lastLocalSettingsEditAt < 700;
        if (activeAbort || settingsSyncTimer || recentlyEdited) return;
      }
      try {
        const serverSettings = normalizeSettings(await fetchDaemonSettings());
        settings = Object.assign({}, serverSettings, {system: loadStoredSystemPrompt()});
        lastSyncedSettingsSignature = daemonSettingsSignature(settings);
        applySettingsToUI(settings);
        saveSettings(settings);
      } catch (err) {
        console.warn("daemon settings refresh failed", err);
      }
    }
    async function syncDaemonSettings(nextSettings, options) {
      const opts = options || {};
      const signature = daemonSettingsSignature(nextSettings);
      if (!opts.force && signature === lastSyncedSettingsSignature) return normalizeSettings(nextSettings);
      const seq = ++settingsSyncSeq;
      const response = await fetch("/v1/mtplx/settings", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(daemonSettingsPayload(nextSettings))
      });
      if (!response.ok) {
        let detail = "settings update failed: " + response.status;
        try {
          const errBody = await response.json();
          if (errBody?.error?.message) detail = errBody.error.message;
        } catch (_e) { /* ignore */ }
        throw new Error(detail);
      }
      const payload = await response.json();
      if (seq !== settingsSyncSeq) return normalizeSettings(nextSettings);
      const serverSettings = normalizeSettings(settingsFromDaemonPayload(payload));
      lastSyncedSettingsSignature = daemonSettingsSignature(serverSettings);
      return serverSettings;
    }
    function scheduleDaemonSettingsSync(nextSettings, options) {
      const opts = options || {};
      if (settingsSyncTimer) {
        clearTimeout(settingsSyncTimer);
        settingsSyncTimer = null;
      }
      settingsSyncTimer = setTimeout(() => {
        settingsSyncTimer = null;
        syncDaemonSettings(nextSettings).then((serverSettings) => {
          settings = Object.assign({}, serverSettings, {system: nextSettings.system});
          applySettingsToUI(settings);
          saveSettings(settings);
        }).catch((err) => {
          console.warn("daemon settings sync failed", err);
          setStatus("Settings update failed", "error");
        });
      }, opts.immediate ? 0 : 180);
    }
    function refreshLabels() {
      valEls.temperature.textContent = Number(ctlEls.temperature.value).toFixed(2);
      valEls.top_p.textContent = Number(ctlEls.top_p.value).toFixed(2);
      const tk = parseInt(ctlEls.top_k.value, 10) || 0;
      valEls.top_k.textContent = tk === 0 ? "off" : String(tk);
      const mtpOn = Boolean(ctlEls.mtp_enabled.checked);
      valEls.mtp_enabled.textContent = mtpOn ? "on" : "off";
      ctlEls.depth.disabled = !mtpOn;
      valEls.depth.textContent = mtpOn ? String(parseInt(ctlEls.depth.value, 10) || 0) : "off";
      const mt = parseInt(ctlEls.max_tokens.value, 10) || 0;
      valEls.max_tokens.textContent = mt >= 1000 ? (mt / 1000).toFixed(1).replace(/\\.0$/, "") + "k" : String(mt);
    }
    function refreshSliderFills() {
      for (const key of ["temperature", "top_p", "top_k", "depth", "max_tokens"]) {
        const el = ctlEls[key];
        const range = RANGES[key];
        if (!el || !range) continue;
        if (key === "depth" && !ctlEls.mtp_enabled.checked) {
          el.style.setProperty("--filled", "0%");
          continue;
        }
        const value = Number(el.value);
        const span = range.max - range.min;
        const pct = span > 0 ? ((value - range.min) / span) * 100 : 100;
        el.style.setProperty("--filled", pct.toFixed(2) + "%");
      }
    }
    function readSettings() {
      const s = {
        temperature: clamp(ctlEls.temperature.value, RANGES.temperature.min, RANGES.temperature.max, DEFAULTS.temperature, false),
        top_p: clamp(ctlEls.top_p.value, RANGES.top_p.min, RANGES.top_p.max, DEFAULTS.top_p, false),
        top_k: clamp(ctlEls.top_k.value, RANGES.top_k.min, RANGES.top_k.max, DEFAULTS.top_k, true),
        mtp_enabled: Boolean(ctlEls.mtp_enabled.checked),
        depth: clamp(ctlEls.depth.value, RANGES.depth.min, RANGES.depth.max, DEFAULTS.depth, true),
        max_tokens: clamp(ctlEls.max_tokens.value, RANGES.max_tokens.min, RANGES.max_tokens.max, DEFAULTS.max_tokens, true),
        reasoning: ctlEls.reasoning.value || "auto",
        system: (ctlEls.system.value || "").trim()
      };
      refreshLabels();
      refreshSliderFills();
      return s;
    }
    let settings = loadSettings();
    applySettingsToUI(settings);
    fetchDaemonSettings().catch((err) => {
      console.warn("daemon settings hydrate failed", err);
      return loadSettings();
    }).then((serverSettings) => {
      settings = normalizeSettings(serverSettings);
      lastSyncedSettingsSignature = daemonSettingsSignature(settings);
      applySettingsToUI(settings);
      saveSettings(settings);
      return discoverServerLimits();
    }).then(() => {
      // Re-clamp + redraw after the real context window arrives so users
      // who reload the page don't see "8k" sitting under a fresh 256k cap.
      settings = readSettings();
      saveSettings(settings);
    });
    window.setInterval(() => refreshDaemonSettings(), 1500);
    window.addEventListener("focus", () => refreshDaemonSettings({force: true}));
    function handleSettingsControlEdit(key) {
      return () => {
        lastLocalSettingsEditAt = performance.now();
        settings = readSettings();
        saveSettings(settings);
        if (key !== "system") {
          scheduleDaemonSettingsSync(settings, {immediate: key === "mtp_enabled" || key === "reasoning"});
        }
      };
    }
    for (const key of Object.keys(ctlEls)) {
      const handler = handleSettingsControlEdit(key);
      ctlEls[key].addEventListener("input", handler);
      ctlEls[key].addEventListener("change", handler);
    }
    document.getElementById("reset-defaults").addEventListener("click", () => {
      settings = Object.assign({}, DEFAULTS);
      lastLocalSettingsEditAt = performance.now();
      applySettingsToUI(settings);
      saveSettings(settings);
      scheduleDaemonSettingsSync(settings, {immediate: true});
    });
    sidebarToggleBtn.addEventListener("click", () => sidebarEl.classList.toggle("open"));

    // ---------- markdown ------------------------------------------------------
    function escapeHtml(text) {
      return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }
    function renderMarkdown(text) {
      try {
        if (typeof window.marked !== "undefined") {
          window.marked.setOptions({
            gfm: true,
            breaks: true,
            mangle: false,
            headerIds: false
          });
          return window.marked.parse(String(text || ""));
        }
      } catch (err) {
        console.warn("markdown parse failed; falling back to text", err);
      }
      // Fallback: escape and convert simple newlines to <br>
      return escapeHtml(text).replace(/\\n/g, "<br>");
    }
    function attachCopyButtons(scope) {
      for (const pre of scope.querySelectorAll("pre")) {
        if (pre.querySelector(".copy-btn")) continue;
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "copy-btn";
        btn.textContent = "Copy";
        btn.addEventListener("click", async () => {
          const code = pre.querySelector("code");
          const text = code ? code.textContent || "" : pre.textContent || "";
          try {
            await navigator.clipboard.writeText(text);
            btn.textContent = "Copied";
            btn.classList.add("copied");
            setTimeout(() => { btn.textContent = "Copy"; btn.classList.remove("copied"); }, 1400);
          } catch (err) { btn.textContent = "Error"; }
        });
        pre.appendChild(btn);
      }
    }

    // ---------- scroll handling ----------------------------------------------
    const SCROLL_PIN_THRESHOLD = 160;
    function isPinned() {
      const remaining = messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight;
      return remaining <= SCROLL_PIN_THRESHOLD;
    }
    messagesEl.addEventListener("scroll", () => {
      if (forceAutoScroll) return;
      pinnedToBottom = isPinned();
    });
    function scrollToBottom(opts) {
      const options = opts || {};
      if (!options.force && !forceAutoScroll && !pinnedToBottom) return;
      messagesEl.scrollTop = messagesEl.scrollHeight;
      if (messagesBottom && messagesBottom.scrollIntoView) {
        messagesBottom.scrollIntoView({block: "end", inline: "nearest", behavior: "auto"});
      }
      messagesEl.scrollTop = messagesEl.scrollHeight;
      pinnedToBottom = true;
    }
    function scheduleScrollToBottom(opts) {
      const options = opts || {};
      if (!options.force && !forceAutoScroll && !pinnedToBottom) return;
      if (options.force) pinnedToBottom = true;
      if (postLayoutScrollTimer) {
        clearTimeout(postLayoutScrollTimer);
        postLayoutScrollTimer = null;
      }
      if (scrollFrame !== null) return;
      scrollFrame = requestAnimationFrame(() => {
        scrollFrame = null;
        scrollToBottom({force: options.force});
        // Streamed code blocks and final markdown can grow after the text node
        // update that triggered this scroll. Recheck on the next frame and
        // once more after layout settles so coding output keeps following.
        requestAnimationFrame(() => scrollToBottom({force: options.force}));
        postLayoutScrollTimer = setTimeout(() => {
          postLayoutScrollTimer = null;
          scrollToBottom({force: options.force});
        }, 40);
      });
    }
    if (window.ResizeObserver) {
      const scrollObserver = new ResizeObserver(() => scheduleScrollToBottom());
      scrollObserver.observe(messagesInner);
    }

    // ---------- DOM helpers ---------------------------------------------------
    function setStatus(text, kind) {
      statusText.textContent = text;
      statusRow.className = kind || "";
    }
    function appendUser(text) {
      const node = document.createElement("div");
      node.className = "turn turn-user";
      const body = document.createElement("div");
      body.className = "turn-body";
      body.textContent = text;
      node.appendChild(body);
      messagesInner.appendChild(node);
      scheduleScrollToBottom();
    }
    function appendAssistantTurn() {
      const node = document.createElement("div");
      node.className = "turn turn-assistant";
      const avatar = document.createElement("div");
      avatar.className = "avatar";
      avatar.textContent = "M";
      const body = document.createElement("div");
      body.className = "turn-body";

      const reasoningBlock = document.createElement("div");
      reasoningBlock.className = "reasoning-block";
      reasoningBlock.hidden = true;
      const reasoningSummary = document.createElement("div");
      reasoningSummary.className = "reasoning-summary";
      reasoningSummary.innerHTML =
        '<svg class="chev" viewBox="0 0 10 10" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 2l4 3-4 3"/></svg>' +
        '<span class="label">Thinking</span>' +
        '<span class="meta"></span>';
      const reasoningBody = document.createElement("div");
      reasoningBody.className = "reasoning-body";
      reasoningBlock.appendChild(reasoningSummary);
      reasoningBlock.appendChild(reasoningBody);
      reasoningSummary.addEventListener("click", () => reasoningBlock.classList.toggle("open"));

      const answerBody = document.createElement("div");
      answerBody.className = "answer streaming-plain";

      const stats = document.createElement("div");
      stats.className = "stats";
      stats.hidden = true;

      body.appendChild(reasoningBlock);
      body.appendChild(answerBody);
      body.appendChild(stats);
      node.appendChild(avatar);
      node.appendChild(body);
      messagesInner.appendChild(node);
      scheduleScrollToBottom();
      return {node, reasoningBlock, reasoningSummary, reasoningBody, answerBody, stats};
    }
    function setReasoningMeta(turn, text) {
      const meta = turn.reasoningSummary.querySelector(".meta");
      if (meta) meta.textContent = text;
    }
    function renderStats(statsEl, stats) {
      if (!stats) return;
      const verifyMs = stats.verify_calls ? (1000 * Number(stats.verify_time_s || 0) / Number(stats.verify_calls)) : null;
      const tps = Number(stats.decode_tok_s ?? stats.request_tok_s);
      const generationMode = String(stats.generation_mode || "").toLowerCase();
      const mtpDepth = Number(stats.mtp_depth ?? stats.speculative_depth);
      const parts = [];
      if (generationMode === "ar") parts.push("AR");
      else if (generationMode === "mtp") parts.push("MTP depth " + (Number.isFinite(mtpDepth) ? mtpDepth : "?"));
      if (Number.isFinite(tps)) parts.push('<span class="stat-tps">' + tps.toFixed(1) + ' tok/s</span>');
      if (Number.isFinite(Number(stats.completion_tokens))) parts.push(Number(stats.completion_tokens) + ' tokens');
      if (Number.isFinite(Number(stats.reasoning_tokens)) && Number(stats.reasoning_tokens) > 0) parts.push(Number(stats.reasoning_tokens) + ' thinking');
      if (Number.isFinite(Number(stats.ttft_s))) parts.push('<span class="stat-ttft">ttft ' + Number(stats.ttft_s).toFixed(2) + 's</span>');
      if (Number.isFinite(verifyMs)) parts.push(verifyMs.toFixed(1) + ' ms/verify');
      if (Number.isFinite(Number(stats.verify_calls))) parts.push(Number(stats.verify_calls) + ' verifies');
      statsEl.innerHTML = parts.join(' <span style="color:var(--muted-2)">·</span> ');
      statsEl.hidden = parts.length === 0;
    }
    function renderLiveStats(state) {
      const explicitTps = Number(state.tps);
      const tps = Number.isFinite(explicitTps)
        ? explicitTps
        : (state.tokens > 0 && state.elapsed > 0 ? (state.tokens / state.elapsed) : null);
      const parts = [];
      if (tps !== null) parts.push('<span class="tps">' + tps.toFixed(1) + ' tok/s</span>');
      if (state.tokens > 0) parts.push(state.tokens + ' tokens');
      liveStatsEl.innerHTML = parts.length ? '· ' + parts.join(' · ') : '';
    }
    function applyProgressToLiveState(liveState, progress) {
      if (!progress) return;
      const tokens = Number(progress.completion_tokens ?? progress.generated_tokens);
      const elapsed = Number(progress.decode_elapsed_s ?? progress.elapsed_s);
      const tps = Number(progress.decode_tok_s ?? progress.tok_s);
      if (Number.isFinite(tokens) && tokens >= 0) liveState.tokens = tokens;
      if (Number.isFinite(elapsed) && elapsed >= 0) liveState.elapsed = elapsed;
      if (Number.isFinite(tps) && tps >= 0) liveState.tps = tps;
      liveState.hasServerProgress = true;
      renderLiveStats(liveState);
    }
    function applyFinalStatsToLiveState(liveState, stats) {
      if (!stats) return;
      const tokens = Number(stats.completion_tokens ?? stats.generated_tokens);
      const elapsed = Number(stats.decode_elapsed_s ?? stats.request_elapsed_s ?? stats.elapsed_s);
      const tps = Number(stats.decode_tok_s ?? stats.request_tok_s ?? stats.tok_s);
      if (Number.isFinite(tokens) && tokens >= 0) liveState.tokens = tokens;
      if (Number.isFinite(elapsed) && elapsed >= 0) liveState.elapsed = elapsed;
      if (Number.isFinite(tps) && tps >= 0) liveState.tps = tps;
      liveState.hasServerProgress = true;
      renderLiveStats(liveState);
    }

    // ---------- streaming -----------------------------------------------------
    function splitFallbackReasoning(reasoningText, answerText) {
      if (answerText.trim() || !reasoningText.trim()) return {reasoningText, answerText};
      const blocks = reasoningText.trim().split(/\\n\\s*\\n/).filter(Boolean);
      if (blocks.length < 2) return {reasoningText, answerText};
      const answer = blocks[blocks.length - 1].trim();
      const reasoning = blocks.slice(0, -1).join("\\n\\n").trim();
      return {reasoningText: reasoning, answerText: answer};
    }
    function drainSse(buffer, onPayload) {
      let offset = 0;
      while (true) {
        const index = buffer.indexOf("\\n\\n", offset);
        if (index < 0) break;
        const eventText = buffer.slice(offset, index);
        offset = index + 2;
        for (const line of eventText.split("\\n")) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6).trim();
          if (data === "[DONE]") continue;
          try { onPayload(JSON.parse(data)); } catch (err) { console.warn(err, data); }
        }
      }
      return buffer.slice(offset);
    }
    function setSendButton(mode) {
      if (mode === "stop") {
        sendBtn.classList.add("stop");
        sendBtn.dataset.action = "stop";
        sendBtn.innerHTML = SVG_STOP;
        sendBtn.setAttribute("aria-label", "Stop generation");
      } else {
        sendBtn.classList.remove("stop");
        sendBtn.dataset.action = "send";
        sendBtn.innerHTML = SVG_SEND;
        sendBtn.setAttribute("aria-label", "Send");
      }
    }
    async function sendMessage(text) {
      // Remove the greeting bubble on first send.
      const greeting = messagesInner.querySelector(".turn-greeting");
      if (greeting) greeting.remove();

      pinnedToBottom = true;
      forceAutoScroll = true;
      appendUser(text);
      history.push({role: "user", content: text});
      const turn = appendAssistantTurn();
      scheduleScrollToBottom({force: true});
      setSendButton("stop");
      promptEl.disabled = false;
      setStatus("Thinking", "streaming");

      activeAbort = new AbortController();
      let assistantText = "";
      let reasoningText = "";
      let finalStats = null;
      let firstTokenAt = null;
      const startedAt = performance.now();
      const liveState = {tokens: 0, elapsed: 0, tps: null, hasServerProgress: false};
      // Transport watchdog: normal long generations should receive server
      // heartbeat/progress frames. Abort only when the stream itself goes
      // quiet, not when the model is still actively working.
      let lastChunkAt = performance.now();
      const STALL_WARN_MS = 30000;
      const STALL_ABORT_MS = 90000;
      let stallTimer = null;
      function armStallWatchdog() {
        if (stallTimer) clearInterval(stallTimer);
        stallTimer = setInterval(() => {
          const idle = performance.now() - lastChunkAt;
          if (idle > STALL_ABORT_MS) {
            clearInterval(stallTimer);
            stallTimer = null;
            try { activeAbort && activeAbort.abort("stalled"); } catch (_e) {}
          } else if (idle > STALL_WARN_MS) {
            setStatus("Waiting for stream data (" + Math.round(idle / 1000) + "s)", "streaming");
          }
        }, 1000);
      }
      function disarmStallWatchdog() {
        if (stallTimer) { clearInterval(stallTimer); stallTimer = null; }
      }
      try {
        let settingsNow = readSettings();
        if (settingsSyncTimer) {
          clearTimeout(settingsSyncTimer);
          settingsSyncTimer = null;
        }
        const serverSettings = await syncDaemonSettings(settingsNow);
        settingsNow = Object.assign({}, serverSettings, {system: settingsNow.system});
        settings = settingsNow;
        applySettingsToUI(settings);
        saveSettings(settings);
        const messages = settingsNow.system
          ? [{role: "system", content: settingsNow.system}, ...history]
          : history.slice();
        const requestBody = {
          model: MODEL_ID,
          messages,
          stream: true,
          temperature: settingsNow.temperature,
          top_p: settingsNow.top_p,
          generation_mode: settingsNow.mtp_enabled ? "mtp" : "ar",
          depth: settingsNow.depth,
          max_tokens: settingsNow.max_tokens
        };
        if (settingsNow.top_k > 0) requestBody.top_k = settingsNow.top_k;
        if (settingsNow.reasoning === "on") requestBody.enable_thinking = true;
        else if (settingsNow.reasoning === "off") requestBody.enable_thinking = false;

        armStallWatchdog();
        const response = await fetch("/v1/chat/completions", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-MTPLX-Client": "mtplx_browser",
            "X-MTPLX-Allow-Client-Controls": "1"
          },
          body: JSON.stringify(requestBody),
          signal: activeAbort.signal
        });
        if (!response.ok || !response.body) {
          let detail = "Request failed: " + response.status;
          try {
            const errBody = await response.json();
            if (errBody?.error?.message) detail = errBody.error.message;
          } catch (_e) { /* ignore */ }
          throw new Error(detail);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
          const {value, done} = await reader.read();
          if (done) break;
          lastChunkAt = performance.now();
          buffer += decoder.decode(value, {stream: true});
          buffer = drainSse(buffer, (payload) => {
            if (payload.error) throw new Error(payload.error.message || "generation failed");
            if (payload.mtplx_progress) {
              applyProgressToLiveState(liveState, payload.mtplx_progress);
              if (payload.mtplx_progress.heartbeat) {
                setStatus("Still working", "streaming");
              } else {
                setStatus("Thinking", "streaming");
              }
            }
            if (payload.mtplx_stats) {
              finalStats = payload.mtplx_stats;
              renderStats(turn.stats, finalStats);
              applyFinalStatsToLiveState(liveState, finalStats);
            }
            const delta = payload.choices?.[0]?.delta || {};
            const reasoningPiece = delta.reasoning_content || "";
            const answerPiece = delta.content || "";
            if (reasoningPiece) {
              reasoningText += reasoningPiece;
              turn.reasoningBody.textContent = reasoningText;
              if (turn.reasoningBlock.hidden) {
                turn.reasoningBlock.hidden = false;
                turn.reasoningBlock.classList.add("open");
              }
              setReasoningMeta(turn, "streaming…");
              setStatus("Thinking", "streaming");
            }
            if (answerPiece) {
              if (firstTokenAt === null) firstTokenAt = performance.now();
              assistantText += answerPiece;
              turn.answerBody.textContent = assistantText;
              if (!liveState.hasServerProgress) {
                liveState.tokens += 1;
                liveState.elapsed = (performance.now() - startedAt) / 1000;
                liveState.tps = null;
                renderLiveStats(liveState);
              }
              setStatus("Streaming", "streaming");
              // Auto-collapse the reasoning block once the answer starts (still expandable).
              if (!turn.reasoningBlock.hidden) {
                turn.reasoningBlock.classList.remove("open");
                setReasoningMeta(turn, "click to expand");
              }
            }
            scheduleScrollToBottom({force: true});
          });
        }
        const separated = splitFallbackReasoning(reasoningText, assistantText);
        reasoningText = separated.reasoningText;
        assistantText = separated.answerText;
        turn.reasoningBody.textContent = reasoningText;
        turn.reasoningBlock.hidden = !reasoningText.trim();
        if (!turn.reasoningBlock.hidden) setReasoningMeta(turn, "click to expand");
        const finalText = assistantText.trim() ? assistantText : "(No text returned.)";
        turn.answerBody.classList.remove("streaming-plain");
        turn.answerBody.innerHTML = renderMarkdown(finalText);
        attachCopyButtons(turn.answerBody);
        if (finalStats) renderStats(turn.stats, finalStats);
        history.push({role: "assistant", content: assistantText});
        setStatus("Ready", "ready");
        renderLiveStats(liveState);
      } catch (err) {
        const aborted = err && (err.name === "AbortError" || /aborted/i.test(String(err.message || "")));
        const stalled =
          aborted &&
          (performance.now() - lastChunkAt) > STALL_ABORT_MS;
        if (stalled) {
          turn.answerBody.classList.remove("streaming-plain");
          turn.answerBody.innerHTML = renderMarkdown(
            "*[stream connection went quiet for " + Math.round(STALL_ABORT_MS / 1000) + "s - request stopped]*\\n\\n" +
              "The browser did not receive stream data from MTPLX. Check the server terminal, " +
              "then retry or restart only if the MTPLX process has exited."
          );
          attachCopyButtons(turn.answerBody);
          setStatus("Stream disconnected", "");
        } else if (aborted) {
          turn.answerBody.classList.remove("streaming-plain");
          turn.answerBody.innerHTML = renderMarkdown(assistantText || "*[stopped]*");
          attachCopyButtons(turn.answerBody);
          history.push({role: "assistant", content: assistantText});
          setStatus("Stopped", "ready");
        } else {
          turn.answerBody.textContent = "Error: " + (err?.message || String(err));
          setStatus("Error", "");
        }
      } finally {
        disarmStallWatchdog();
        activeAbort = null;
        setSendButton("send");
        promptEl.disabled = false;
        promptEl.focus();
        autoResizePrompt();
        scheduleScrollToBottom({force: true});
        forceAutoScroll = false;
      }
    }

    // ---------- form wiring ---------------------------------------------------
    function autoResizePrompt() {
      promptEl.style.height = "auto";
      promptEl.style.height = Math.min(promptEl.scrollHeight, 220) + "px";
    }
    promptEl.addEventListener("input", autoResizePrompt);

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      if (sendBtn.dataset.action === "stop") {
        if (activeAbort) activeAbort.abort();
        return;
      }
      const text = promptEl.value.trim();
      if (!text) return;
      promptEl.value = "";
      autoResizePrompt();
      sendMessage(text);
    });
    promptEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        form.requestSubmit();
      }
    });

    newChatBtn.addEventListener("click", () => {
      if (activeAbort) activeAbort.abort();
      history.length = 0;
      messagesInner.innerHTML = '<div class="turn turn-assistant turn-greeting"><div class="avatar">M</div><div class="turn-body"><div class="answer"><p>New conversation. Settings on the left are unchanged.</p></div></div></div>';
      pinnedToBottom = true;
      forceAutoScroll = false;
      setStatus("Ready", "ready");
      renderLiveStats({tokens: 0, elapsed: 0});
      promptEl.focus();
      autoResizePrompt();
      scheduleScrollToBottom({force: true});
    });

    setSendButton("send");
    autoResizePrompt();
  </script>
</body>
</html>"""
    return (
        template.replace("__MODEL__", html.escape(model_id))
        .replace("__API_NOTE__", html.escape(api_note))
        .replace("__SERVER_URL__", html.escape(server_url))
        .replace("__MODEL_JSON__", json.dumps(model_id))
        .replace(
            "__DEFAULT_SETTINGS_JSON__", json.dumps(default_settings, sort_keys=True)
        )
        .replace("__DEPTH_VALUE__", str(default_depth))
        .replace("__DEPTH_MAX__", str(depth_max))
    )


def _thinking_enabled_for_request(
    state: ServerState,
    request: ChatCompletionRequest,
    *,
    allow_client_controls: bool = True,
) -> bool:
    if _reasoning_parser_for_state(state) == "none":
        return False
    return (
        state.args.enable_thinking
        if request.enable_thinking is None or not allow_client_controls
        else bool(request.enable_thinking)
    )


def _normalize_reasoning_effort(value: Any, *, default: str = "auto") -> str:
    effort = str(value or default).strip().lower()
    if effort not in {"auto", "low", "medium", "high"}:
        raise ValueError("reasoning_effort must be one of: auto, low, medium, high")
    return effort


def _reasoning_effort_for_state(
    state: ServerState,
    *,
    thinking_enabled: bool,
    request_effort: str | None = None,
    allow_client_controls: bool = True,
) -> str | None:
    if not thinking_enabled:
        return None
    backend = _backend_descriptor(state)
    levels = set(backend.reasoning_codec.effort_levels)
    if not levels:
        return None
    raw = (
        request_effort
        if request_effort is not None and allow_client_controls
        else getattr(state.args, "reasoning_effort", None)
    )
    effort = _normalize_reasoning_effort(
        raw,
        default=backend.reasoning_codec.default_effort or "auto",
    )
    if effort == "auto":
        effort = backend.reasoning_codec.default_effort or "low"
    return effort if effort in levels else backend.reasoning_codec.default_effort


def _aime_visible_working_for_request(metadata: Mapping[str, Any]) -> bool:
    if not isinstance(metadata, Mapping):
        return False
    client = str(metadata.get("client") or "").strip().lower()
    if client != "aime":
        return False
    raw = metadata.get("aime_visible_working")
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_reasoning_mode(value: Any, *, default: str = "auto") -> str:
    mode = str(value or default).strip().lower()
    if mode not in {"auto", "on", "off"}:
        raise ValueError("reasoning must be one of: auto, on, off")
    return mode


def _normalize_preserve_thinking_policy(value: Any, *, default: str = "auto") -> str:
    mode = str(value or default).strip().lower()
    if mode not in {"auto", "on", "off"}:
        raise ValueError("preserve_thinking must be one of: auto, on, off")
    return mode


def _preserve_thinking_effective(args: argparse.Namespace) -> bool:
    return _normalize_preserve_thinking_policy(
        getattr(args, "preserve_thinking", "auto")
    ) in {"auto", "on"}


def _set_server_reasoning_mode(state: ServerState, mode: str) -> None:
    normalized = _normalize_reasoning_mode(mode)
    state.args.reasoning = normalized
    # Browser/Pi "auto" means no per-request override. The product default is
    # thinking-capable so OpenAI-compatible agent clients can receive the raw
    # reasoning stream unless they explicitly disable it.
    state.args.enable_thinking = False if normalized == "off" else True


def _server_settings_payload(state: ServerState) -> dict[str, Any]:
    reasoning = getattr(state.args, "reasoning", None)
    if reasoning not in {"auto", "on", "off"}:
        reasoning = (
            "on" if bool(getattr(state.args, "enable_thinking", True)) else "off"
        )
    payload = {
        "ok": True,
        "reasoning": reasoning,
        "enable_thinking": bool(getattr(state.args, "enable_thinking", True)),
        "preserve_thinking": getattr(state.args, "preserve_thinking", "auto"),
        "preserve_thinking_effective": _preserve_thinking_effective(state.args),
        "reasoning_parser": state.args.reasoning_parser,
        "generation_mode": state.args.generation_mode,
        "depth": state.args.depth,
        "model": state.model_id,
        "metal_memory_caps": getattr(
            state,
            "metal_memory_caps",
            {"applied": False, "reason": "unavailable"},
        ),
    }
    payload.update(_mtplx_current_settings(state))
    return payload


def _server_console_help() -> str:
    return (
        "MTPLX server controls: /reasoning on|off|auto|status, "
        "/mtp on|off|status, /stats, /help"
    )


def _server_console_handle_command(state: ServerState, raw: str) -> str | None:
    text = raw.strip()
    if not text:
        return None
    parts = text.split()
    command = parts[0].lower()
    arg = parts[1].lower() if len(parts) > 1 else "status"
    if command in {"/help", "help", "?"}:
        return _server_console_help()
    if command in {"/reasoning", "reasoning"}:
        if arg in {"", "status"}:
            payload = _server_settings_payload(state)
            return (
                f"Reasoning: {payload['reasoning']} "
                f"(enable_thinking={str(payload['enable_thinking']).lower()})"
            )
        if arg in {"on", "off", "auto"}:
            _set_server_reasoning_mode(state, arg)
            payload = _server_settings_payload(state)
            return (
                f"Reasoning: {payload['reasoning']} "
                f"(enable_thinking={str(payload['enable_thinking']).lower()})"
            )
        return "usage: /reasoning on|off|auto|status"
    if command in {"/mtp", "mtp"}:
        mode = str(getattr(state.args, "generation_mode", "mtp")).lower()
        if arg in {"", "status"}:
            return f"MTP: {'on' if mode == 'mtp' else 'off'} (generation_mode={mode})"
        if arg == "off":
            state.args.generation_mode = "ar"
            return "MTP: off (AR target-only mode for new requests)"
        if arg == "on":
            if not bool(getattr(state.runtime, "mtp_enabled", False)):
                return "MTP is not available for this loaded model."
            state.args.generation_mode = "mtp"
            return f"MTP: on (depth={int(getattr(state.args, 'depth', 1) or 1)})"
        return "usage: /mtp on|off|status"
    if command in {"/stats", "stats"}:
        latest = (
            state.last_metrics[-1] if getattr(state, "last_metrics", None) else None
        )
        if not latest:
            return "No generation stats yet."
        public = {
            key: latest[key]
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "tok_s",
                "accept_rate",
                "generation_mode",
                "ttft_s",
            )
            if key in latest
        }
        return json.dumps(public or latest, sort_keys=True, default=str)
    if command in {"/exit", "/quit", "exit", "quit"}:
        return "Use Ctrl-C in this MTPLX terminal to stop the server."
    return "Unknown MTPLX server command. Try /help"


def _start_server_console(state: ServerState) -> None:
    if not sys.stdin.isatty():
        return

    def console_loop() -> None:
        _startup_line(_server_console_help())
        while True:
            try:
                raw = input("mtplx> ")
            except EOFError:
                return
            except BaseException:
                return
            try:
                response = _server_console_handle_command(state, raw)
            except BaseException as exc:
                response = f"error: {type(exc).__name__}: {exc}"
            if response:
                _startup_line(response)

    thread = Thread(target=console_loop, name="mtplx-server-console", daemon=True)
    thread.start()


def create_app(state: ServerState) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Bind the running asyncio loop to the dashboard bus so generation
        # workers (sync threads) can publish via call_soon_threadsafe without
        # needing to grab the loop themselves.
        dashboard = getattr(state, "dashboard", None)
        if dashboard is not None:
            dashboard.bus.attach_loop(asyncio.get_running_loop())
        bg_tasks: list[asyncio.Task[Any]] = []
        if dashboard is not None and bool(getattr(state.args, "enable_thermal_poll", False)):
            bg_tasks.append(asyncio.create_task(_thermal_poll_loop(state)))
        try:
            yield
        finally:
            # Cancel any in-flight AIME benchmark runs so the daemon
            # doesn't leak runner asyncio tasks past shutdown.
            try:
                from mtplx.benchmarks.runners import aime as _aime_runner

                await _aime_runner.stop_runs()
            except Exception:
                pass
            for task in bg_tasks:
                task.cancel()
            scheduler = getattr(state, "model_scheduler", None)
            if scheduler is not None:
                scheduler.shutdown(wait=False, cancel_futures=True)
            else:
                postcommit_executor = getattr(state, "postcommit_executor", None)
                generation_executor = getattr(state, "generation_executor", None)
                if postcommit_executor is not None:
                    postcommit_executor.shutdown(wait=False, cancel_futures=True)
                if generation_executor is not None:
                    generation_executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="MTPLX OpenAI-compatible server", lifespan=lifespan)
    app.state.mtplx = state
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def api_key_and_rate_limit(
        request: Request, call_next: Callable[[Request], Any]
    ) -> Any:
        if not _request_is_authorized(
            request, state.args.api_key
        ) and not _request_is_browser_auth_bootstrap(request, state.args.api_key):
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "missing or invalid API key",
                        "type": "authentication_error",
                    }
                },
                headers={"WWW-Authenticate": "Bearer"},
            )
        allowed, retry_after = state.rate_limiter.check(_rate_limit_key(request))
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": "rate limit exceeded",
                        "type": "rate_limit_error",
                    }
                },
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)

    @app.get(_BROWSER_AUTH_PATH)
    def browser_auth(request: Request) -> Response:
        configured_api_key = getattr(state.args, "api_key", None)
        if configured_api_key and not _request_is_browser_auth_bootstrap(
            request, configured_api_key
        ):
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "missing or invalid API key",
                        "type": "authentication_error",
                    }
                },
                headers={"WWW-Authenticate": "Bearer"},
            )
        next_path = _safe_browser_auth_next_path(request.query_params.get("next"))
        response = RedirectResponse(url=next_path, status_code=303)
        if configured_api_key:
            response.set_cookie(
                _BROWSER_AUTH_COOKIE,
                configured_api_key,
                max_age=_BROWSER_AUTH_COOKIE_MAX_AGE_SECONDS,
                httponly=True,
                samesite="lax",
                secure=False,
                path="/",
            )
        return response

    @app.get("/", response_class=HTMLResponse)
    def root(request: Request) -> HTMLResponse:
        server_url = str(request.base_url).rstrip("/")
        return HTMLResponse(
            _chat_ui_html(
                model_id=state.model_id,
                server_url=server_url,
                api_key_required=bool(state.args.api_key),
                default_settings={
                    "temperature": float(state.args.temperature),
                    "top_p": float(state.args.top_p),
                    "top_k": int(state.args.top_k),
                    "depth": int(state.args.depth),
                    "depth_max": int(_backend_descriptor(state).draft_semantics.maximum),
                    "mtp_enabled": str(getattr(state.args, "generation_mode", "mtp"))
                    == "mtp",
                    "max_tokens": int(state.args.max_response_tokens or 16384),
                    "reasoning": str(getattr(state.args, "reasoning", None) or "auto"),
                    "system": "",
                },
            )
        )

    @app.head("/")
    def root_head() -> Response:
        return Response(status_code=200)

    @app.get("/v1")
    @app.get("/v1/")
    def v1_landing(request: Request) -> dict[str, Any]:
        server_url = str(request.base_url).rstrip("/")
        return {
            "ok": True,
            "name": "MTPLX OpenAI-compatible API",
            "model": state.model_id,
            "message": "Do not use this as a browser chat page. Paste this URL into Open WebUI Settings > Connections as the OpenAI API Base URL.",
            "mtplx_api_base_url": f"{server_url}/v1",
            "openwebui": {
                "where": "Open WebUI -> Settings -> Connections -> OpenAI-compatible connection",
                "base_url": f"{server_url}/v1",
                "api_key": "leave blank for localhost"
                if not state.args.api_key
                else "use the API key you started MTPLX with",
                "model": state.model_id,
            },
            "endpoints": {
                "models": f"{server_url}/v1/models",
                "chat_completions": f"{server_url}/v1/chat/completions",
                "health": f"{server_url}/health",
            },
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        runtime = getattr(state, "runtime", None)
        if hasattr(state, "foreground_count"):
            foreground_active = int(state.foreground_count())
        else:
            foreground_active = int(getattr(state, "foreground_active", 0) or 0)
        dashboard_active = _dashboard_in_flight_count(state)
        scheduler_state = _mtplx_scheduler_state(state)
        scheduler_active = int(scheduler_state.get("active_requests") or 0)
        active_requests = max(foreground_active, dashboard_active, scheduler_active)
        fan_mode = _server_fan_mode(state)
        smart_status = _smart_fan_status(state)
        fan_boost_active = fan_mode == FAN_MODE_MAX or (
            fan_mode == FAN_MODE_SMART
            and bool(smart_status.get("commanded_max") or smart_status.get("active"))
        )
        runtime_mode = _health_runtime_mode_label(
            state.profile.name,
            state.args.generation_mode,
            fan_boost_active=fan_boost_active,
        )
        profile_payload = state.profile.to_dict()
        profile_default_model_id = profile_payload.get("model_id")
        if profile_default_model_id != state.model_id:
            profile_payload["profile_default_model_id"] = profile_default_model_id
            profile_payload["model_id"] = state.model_id
        active_sampler = {
            "temperature": float(state.args.temperature),
            "top_p": float(state.args.top_p),
            "top_k": int(state.args.top_k),
        }
        profile_default_sampler = profile_payload.get("sampler")
        if profile_default_sampler != active_sampler:
            profile_payload["profile_default_sampler"] = profile_default_sampler
            profile_payload["sampler"] = active_sampler
        return {
            "ok": True,
            "model": state.model_id,
            "model_path": str(
                getattr(runtime, "model_path", None)
                or getattr(state.args, "model", "")
            ),
            "generation_mode": state.args.generation_mode,
            "default_generation_mode": state.args.generation_mode,
            "runtime_mode": runtime_mode,
            "parent_runtime_released_for_aime": bool(
                getattr(state, "aime_parent_runtime_released", False)
            ),
            "fan_mode": fan_mode,
            "fan_boost_active": fan_boost_active,
            "smart_fan_active_count": int(smart_status.get("active_count") or 0),
            "smart_fan_last_transition_at": smart_status.get("last_transition_at"),
            "smart_fan_last_error": smart_status.get("last_error"),
            "startup": _startup_health_payload(state),
            "thermal": _thermal_health_payload(
                fan_mode=fan_mode,
                smart_status=smart_status,
            ),
            "available_generation_modes": ["mtp", "ar"],
            "load_mtp": bool(state.args.load_mtp),
            "mtp_enabled": bool(
                getattr(runtime, "mtp_enabled", False)
                if runtime is not None
                else getattr(state.args, "load_mtp", False)
            ),
            "depth": state.args.depth,
            "profile": profile_payload,
            "adaptive": _adaptive_config(state.args),
            "proposal_cache": _proposal_cache_config(state.args),
            "online_hidden": _online_hidden_config(state.args),
            "verify_core": state.args.verify_core,
            "verify_strategy": state.args.verify_strategy,
            "reasoning": getattr(state.args, "reasoning", None)
            or ("on" if bool(state.args.enable_thinking) else "off"),
            "enable_thinking": state.args.enable_thinking,
            "preserve_thinking": getattr(state.args, "preserve_thinking", "auto"),
            "preserve_thinking_effective": _preserve_thinking_effective(state.args),
            "strip_assistant_reasoning_history": bool(
                state.args.strip_assistant_reasoning_history
            ),
            "context_window": state.context_window,
            "max_response_tokens": state.args.max_response_tokens,
            "api_key_required": bool(state.args.api_key),
            "api_key_source": str(
                getattr(state.args, "api_key_source", "none") or "none"
            ),
            "paged_kv_quantization": _effective_paged_kv_quantization(),
            "rate_limit_per_minute": int(state.args.rate_limit),
            "stream_interval": int(state.args.stream_interval),
            "warmup": state.warmup_status,
            "foreground_active": foreground_active,
            "dashboard_active_requests": dashboard_active,
            "active_requests": active_requests,
            "scheduler": scheduler_state,
            "session_bank": (
                state.sessions.bank.to_dict()
                if hasattr(getattr(state, "sessions", None), "bank")
                and hasattr(state.sessions.bank, "to_dict")
                else {}
            ),
            "ssd_session_cache": (
                state.session_bank_cold_tier.stats()
                if getattr(state, "session_bank_cold_tier", None) is not None
                else {"enabled": False}
            ),
            "last_request_started_at": getattr(state, "last_request_started_at", 0.0),
            "requests_completed": getattr(state, "requests_completed", 0),
            "requests_cancelled": getattr(state, "requests_cancelled", 0),
            "last_request_at": getattr(state, "last_request_at", 0.0),
            "idle_seconds": (
                time.time() - getattr(state, "last_request_at", 0.0)
                if getattr(state, "last_request_at", 0.0) > 0
                else None
            ),
            "reasoning_parser": state.args.reasoning_parser,
            "load_time_s": getattr(state, "load_time_s", None),
            "draft_lm_head": state.draft_lm_head,
            "draft_sampler": (
                asdict(getattr(state, "draft_sampler", None))
                if is_dataclass(getattr(state, "draft_sampler", None))
                else None
            ),
            "draft_head_identity": state.draft_head_identity,
            "tokenizer_template_hash": state.template_hash,
            "fast_path_env": state.fast_path_env_status,
            "profile_env": state.profile_env_status,
            "diagnostic_env_ablation": bool(state.args.diagnostic_env_ablation),
            "mtp_history_materialize_every": (
                os.environ.get("MTPLX_MTP_HISTORY_MATERIALIZE_EVERY")
            ),
            "clear_cache_every": os.environ.get("MTPLX_CLEAR_CACHE_EVERY"),
            "trunk_cache_materialize_every": (
                os.environ.get("MTPLX_TRUNK_CACHE_MATERIALIZE_EVERY")
            ),
            "state_rebase_every": os.environ.get("MTPLX_STATE_REBASE_EVERY"),
            "state_root_eval": os.environ.get("MTPLX_EVAL_STATE_ROOTS_ON_COMMIT"),
            "state_root_eval_include_mtp": os.environ.get(
                "MTPLX_EVAL_STATE_ROOTS_INCLUDE_MTP"
            ),
            "state_root_eval_include_live": os.environ.get(
                "MTPLX_EVAL_STATE_ROOTS_INCLUDE_LIVE"
            ),
            "target_layer_eval_every": os.environ.get("MTPLX_TARGET_LAYER_EVAL_EVERY"),
            "target_layer_eval_schedule": os.environ.get(
                "MTPLX_TARGET_LAYER_EVAL_SCHEDULE"
            ),
            "target_layer_eval_context_threshold": os.environ.get(
                "MTPLX_TARGET_LAYER_EVAL_CONTEXT_THRESHOLD"
            ),
            "target_layer_eval_max_q": os.environ.get("MTPLX_TARGET_LAYER_EVAL_MAX_Q"),
            "defer_verify_hidden_eval": os.environ.get(
                "MTPLX_DEFER_VERIFY_HIDDEN_EVAL"
            ),
            "late_depth_switch_after_tokens": os.environ.get(
                "MTPLX_LATE_DEPTH_SWITCH_AFTER_TOKENS"
            ),
            "late_depth_before": os.environ.get("MTPLX_LATE_DEPTH_BEFORE"),
            "late_depth_after": os.environ.get("MTPLX_LATE_DEPTH_AFTER"),
            "long_context_mtp_depth_policy": os.environ.get(
                "MTPLX_LONG_CONTEXT_MTP_DEPTH_POLICY"
            ),
            "long_context_mtp_depth_threshold": os.environ.get(
                "MTPLX_LONG_CONTEXT_MTP_DEPTH_THRESHOLD"
            ),
            "long_context_mtp_depth": os.environ.get("MTPLX_LONG_CONTEXT_MTP_DEPTH"),
            "opencode_short_context_depth_policy": {
                "active": False,
                "reason": "disabled_depth_preservation",
                "default_depth": int(getattr(state.args, "depth", 3)),
            },
            "opencode_short_context_depth2_tokens": None,
            "mtp_position_mode": os.environ.get("MTPLX_MTP_POSITION_MODE"),
            "mtp_position_cap": os.environ.get("MTPLX_MTP_POSITION_CAP"),
            "mtp_position_period": os.environ.get("MTPLX_MTP_POSITION_PERIOD"),
            "mtp_position_base": os.environ.get("MTPLX_MTP_POSITION_BASE"),
            "split_verify_eval": os.environ.get("MTPLX_SPLIT_VERIFY_EVAL"),
            "split_full_attn": os.environ.get("MTPLX_SPLIT_FULL_ATTN"),
            "split_full_attn_chunk_size": os.environ.get(
                "MTPLX_SPLIT_FULL_ATTN_CHUNK_SIZE"
            ),
            "split_full_attn_threshold": os.environ.get(
                "MTPLX_SPLIT_FULL_ATTN_THRESHOLD"
            ),
            "sdpa_2pass": os.environ.get("MTPLX_SDPA_2PASS"),
            "sdpa_2pass_threshold": os.environ.get("MTPLX_SDPA_2PASS_THRESHOLD"),
            "sdpa_2pass_max_q": os.environ.get("MTPLX_SDPA_2PASS_MAX_Q"),
            "blockwise_attn": os.environ.get("MTPLX_BLOCKWISE_ATTN"),
            "blockwise_attn_threshold": os.environ.get(
                "MTPLX_BLOCKWISE_ATTN_THRESHOLD"
            ),
            "dirty_detach_components": os.environ.get("MTPLX_DETACH_COMPONENTS"),
            "dirty_detach_mode": os.environ.get("MTPLX_DETACH_MODE"),
            "dirty_detach_gdn_every": os.environ.get("MTPLX_DETACH_GDN_EVERY"),
            "dirty_detach_conv_every": os.environ.get("MTPLX_DETACH_CONV_EVERY"),
            "dirty_detach_attn_every": os.environ.get("MTPLX_DETACH_ATTN_EVERY"),
            "capture_commit_detach_components": os.environ.get(
                "MTPLX_CAPTURE_COMMIT_DETACH_COMPONENTS"
            ),
            "capture_commit_detach_mode": os.environ.get(
                "MTPLX_CAPTURE_COMMIT_DETACH_MODE"
            ),
            "capture_commit_detach_gdn_every": os.environ.get(
                "MTPLX_CAPTURE_COMMIT_DETACH_GDN_EVERY"
            ),
            "capture_commit_detach_conv_every": os.environ.get(
                "MTPLX_CAPTURE_COMMIT_DETACH_CONV_EVERY"
            ),
            "owned_recurrent_state": os.environ.get("MTPLX_OWNED_RECURRENT_STATE"),
            "owned_recurrent_state_mode": os.environ.get(
                "MTPLX_OWNED_RECURRENT_STATE_MODE"
            ),
            "owned_attn_kv": os.environ.get("MTPLX_OWNED_ATTN_KV"),
            "owned_attn_kv_mode": os.environ.get("MTPLX_OWNED_ATTN_KV_MODE"),
            "owned_attn_kv_step": os.environ.get("MTPLX_OWNED_ATTN_KV_STEP"),
            "owned_attn_kv_block_size": os.environ.get(
                "MTPLX_OWNED_ATTN_KV_BLOCK_SIZE"
            ),
            "vllm_metal_paged_attn": os.environ.get("MTPLX_VLLM_METAL_PAGED_ATTN"),
            "vllm_metal_paged_block_size": os.environ.get(
                "MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE"
            ),
            "vllm_metal_paged_num_blocks": os.environ.get(
                "MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS"
            ),
            "vllm_metal_paged_sliding_window": os.environ.get(
                "MTPLX_VLLM_METAL_PAGED_SLIDING_WINDOW"
            ),
            "vllm_metal_paged_partitioned_attn": os.environ.get(
                "MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN"
            ),
            "vllm_metal_paged_partition_threshold": os.environ.get(
                "MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD"
            ),
            "vllm_metal_paged_partition_size": os.environ.get(
                "MTPLX_VLLM_METAL_PAGED_PARTITION_SIZE"
            ),
            "vllm_metal_paged_mtp_attn": os.environ.get(
                "MTPLX_VLLM_METAL_PAGED_MTP_ATTN"
            ),
            "vllm_metal_paged_mtp_block_size": os.environ.get(
                "MTPLX_VLLM_METAL_PAGED_MTP_BLOCK_SIZE"
            ),
            "vllm_metal_paged_mtp_num_blocks": os.environ.get(
                "MTPLX_VLLM_METAL_PAGED_MTP_NUM_BLOCKS"
            ),
            "vllm_metal_paged_turboquant": os.environ.get(
                "MTPLX_VLLM_METAL_PAGED_TURBOQUANT"
            ),
            "vllm_metal_paged_turboquant_k_quant": os.environ.get(
                "MTPLX_VLLM_METAL_PAGED_TURBOQUANT_K_QUANT"
            ),
            "vllm_metal_paged_turboquant_v_quant": os.environ.get(
                "MTPLX_VLLM_METAL_PAGED_TURBOQUANT_V_QUANT"
            ),
            "vllm_metal_paged_kv_quant": os.environ.get(
                "MTPLX_VLLM_METAL_PAGED_KV_QUANT"
            ),
            "native_mlp_rowwise": os.environ.get("MTPLX_NATIVE_MLP_ROWWISE"),
            "native_mlp_min_m": os.environ.get("MTPLX_NATIVE_MLP_MIN_M"),
            "native_mlp_max_m": os.environ.get("MTPLX_NATIVE_MLP_MAX_M"),
            "native_mlp_context_threshold": os.environ.get(
                "MTPLX_NATIVE_MLP_CONTEXT_THRESHOLD"
            ),
            "mlp_call_variant": os.environ.get("MTPLX_MLP_CALL_VARIANT"),
            "mlp_variant_stats": native_mlp_stats(),
            "native_gdn_tail": os.environ.get("MTPLX_NATIVE_GDN_TAIL"),
            "native_gdn_tail_simdgroups": os.environ.get(
                "MTPLX_NATIVE_GDN_TAIL_SIMDGROUPS"
            ),
            "live_output_detach": os.environ.get("MTPLX_DETACH_LIVE_OUTPUTS"),
            "live_output_detach_mode": os.environ.get("MTPLX_DETACH_LIVE_OUTPUTS_MODE"),
            "aime_process_isolation": os.environ.get("MTPLX_AIME_PROCESS_ISOLATION"),
            "metal_memory_caps": getattr(
                state,
                "metal_memory_caps",
                {"applied": False, "reason": "unavailable"},
            ),
            "mlx_cache_limit": state.mlx_cache_limit_status,
            "mlx_fork": state.mlx_fork_status,
            # Hardware fields surfaced for the dashboard's HardwareBanner
            # and MemoryStackedBar. Cached after the first lookup.
            **_machine_info(),
        }

    @app.get("/v1/mtplx/settings")
    @app.get("/mtplx/settings")
    def get_mtplx_settings() -> dict[str, Any]:
        return _server_settings_payload(state)

    @app.post("/v1/mtplx/settings")
    @app.post("/mtplx/settings")
    def update_mtplx_settings(update: MTPLXSettingsUpdate) -> dict[str, Any]:
        # `update.model_dump(exclude_none=True)` keeps the public contract
        # unchanged: omitting a field leaves the existing server value alone.
        payload = update.model_dump(exclude_none=True)
        applied = _mtplx_apply_settings_payload(state, payload) if payload else {}
        result = _server_settings_payload(state)
        if applied:
            result["applied"] = applied
        return result

    @app.get("/v1/mtplx/snapshot")
    def mtplx_snapshot() -> dict[str, Any]:
        return _mtplx_dashboard_snapshot(state)

    @app.get("/v1/mtplx/prefill_history")
    def mtplx_prefill_history() -> dict[str, Any]:
        return {
            "capacity": state.dashboard.prefill_history.capacity(),
            "history": state.dashboard.prefill_history.snapshot(),
        }

    @app.post("/v1/mtplx/cancel/{request_id}")
    def mtplx_cancel(request_id: str) -> dict[str, Any]:
        handle = state.dashboard.in_flight.get(request_id)
        cancelled = state.dashboard.in_flight.cancel(request_id)
        if cancelled:
            scheduler = getattr(state, "model_scheduler", None)
            if scheduler is not None and hasattr(scheduler, "record_request_cancelled"):
                latency_s = (
                    max(0.0, time.time() - float(handle.started_s))
                    if handle is not None
                    else None
                )
                scheduler.record_request_cancelled(latency_s=latency_s)
        return {
            "ok": cancelled,
            "request_id": request_id,
            "cancelled": cancelled,
            "active_requests": state.dashboard.in_flight.count(),
        }

    @app.get("/v1/mtplx/app/capabilities")
    def mtplx_app_capabilities() -> dict[str, Any]:
        return _mtplx_app_capabilities()

    @app.post("/v1/mtplx/thermal/fan_mode")
    @app.post("/mtplx/thermal/fan_mode")
    def mtplx_thermal_fan_mode(request: FanModeRequest) -> dict[str, Any]:
        """Set fan mode to ``max``, ``smart``, or ``default``.

        Thin HTTP wrapper around ``mtplx.thermal``'s verified helpers.
        Returns the verified result plus a fresh ``fan_summary`` so the
        UI can render the new ramp state immediately.
        """

        from mtplx.thermal import (
            fan_summary as _fan_summary,
            restore_thermal_profile_verified,
            set_thermal_profile_verified,
        )

        try:
            mode = normalize_fan_mode(request.mode)
            if mode == FAN_MODE_MAX:
                if getattr(state, "smart_fans", None) is not None:
                    state.smart_fans.restore_now(wait=False)
                kwargs: dict[str, Any] = {
                    "require_actual_ramp": bool(request.require_actual_ramp)
                }
                if request.timeout_s is not None:
                    kwargs["actual_ramp_timeout_s"] = float(request.timeout_s)
                result = set_thermal_profile_verified("performance", **kwargs)
                if result.get("ok"):
                    state.fan_mode = FAN_MODE_MAX
                    state.args.fan_mode = FAN_MODE_MAX
            elif mode == FAN_MODE_SMART:
                state.fan_mode = FAN_MODE_SMART
                state.args.fan_mode = FAN_MODE_SMART
                smart_status = (
                    state.smart_fans.restore_now(wait=False)
                    if getattr(state, "smart_fans", None) is not None
                    else {}
                )
                result = {
                    "ok": True,
                    "profile": FAN_MODE_SMART,
                    "message": "smart fan mode enabled",
                    "smart": smart_status,
                }
            else:
                result = restore_thermal_profile_verified()
                if getattr(state, "smart_fans", None) is not None:
                    state.smart_fans.restore_now(wait=False)
                if result.get("ok"):
                    state.fan_mode = FAN_MODE_DEFAULT
                    state.args.fan_mode = FAN_MODE_DEFAULT
        except Exception as exc:
            return {
                "verified": False,
                "current_mode": None,
                "error": f"{type(exc).__name__}: {exc}",
                "fan_summary": _fan_summary(),
            }

        return {
            "verified": bool(result.get("ok")),
            "current_mode": mode if result.get("ok") else None,
            "result": result,
            "smart": _smart_fan_status(state),
            "fan_summary": _fan_summary(),
        }

    @app.get("/v1/mtplx/thermal/status")
    @app.get("/mtplx/thermal/status")
    def mtplx_thermal_status_endpoint() -> dict[str, Any]:
        """Snapshot of fan-control detection and current readings.

        Returns ``ok=False`` (not HTTP 500) when ``thermalforge`` isn't
        installed so the UI can render an unavailable state cleanly.
        """

        from mtplx.thermal import (
            detect_thermal_control,
            fan_summary as _fan_summary,
            thermal_status as _thermal_status,
        )

        try:
            detection = detect_thermal_control()
            status = _thermal_status()
            summary = _fan_summary()
        except Exception as exc:
            return {
                "ok": False,
                "detection": None,
                "current_mode": None,
                "fan_summary": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

        return {
            "ok": bool(status.get("ok", False) or detection.get("available", False)),
            "detection": detection,
            "current_mode": _server_fan_mode(state),
            "smart": _smart_fan_status(state),
            "thermal_status": status,
            "fan_summary": summary,
        }

    # ----- AIME 2026 benchmark endpoints -----------------------------------
    # See mtplx/benchmarks/runners/aime.py for runner semantics and
    # mtplx/benchmarks/prompts/aime_2026.jsonl for the problem dataset.
    # SwiftUI BenchmarkOverlay in apps/MTPLXApp/.../Benchmark/ consumes
    # this surface; the in-browser chat-UI launcher is intentionally NOT
    # wired (the SwiftUI app is the V1 product).

    def _free_loopback_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return int(sock.getsockname()[1])

    def _aime_process_isolation_mode(body: "_AIMEStartBody | None") -> str:
        raw = None
        if body is not None:
            raw = getattr(body, "question_process_isolation", None)
        if raw is None:
            raw = os.environ.get("MTPLX_AIME_PROCESS_ISOLATION")
        mode = str(raw or "off").strip().lower()
        if mode in {"1", "true", "yes", "on", "per-question", "per_question"}:
            return "per_question"
        return "off"

    def _aime_clear_cache_every() -> int:
        aime_raw = os.environ.get("MTPLX_AIME_CLEAR_CACHE_EVERY")
        inherited_raw = os.environ.get("MTPLX_CLEAR_CACHE_EVERY")
        raw = aime_raw if aime_raw is not None else inherited_raw
        normalized = str(raw or "").strip().lower()
        if not normalized or normalized == "auto":
            return 512
        if normalized in {"off", "false", "no"}:
            return 0
        try:
            return max(0, int(normalized))
        except ValueError:
            return 512

    def _aime_worker_drain_s() -> float:
        raw = str(os.environ.get("MTPLX_AIME_WORKER_DRAIN_S") or "0.75").strip()
        try:
            return max(0.0, min(5.0, float(raw)))
        except ValueError:
            return 0.75

    def _aime_release_parent_runtime_enabled(body: "_AIMEStartBody | None") -> bool:
        if _aime_process_isolation_mode(body) != "per_question":
            return False
        raw = str(
            os.environ.get("MTPLX_AIME_RELEASE_PARENT_RUNTIME") or "auto"
        ).strip().lower()
        return raw not in {"0", "false", "no", "off", "never"}

    def _release_parent_runtime_for_aime() -> dict[str, Any]:
        if getattr(state, "aime_parent_runtime_released", False):
            return {
                "released": False,
                "reason": "already_released",
                "allocator_after": _mlx_allocator_public_stats(),
            }
        runtime = getattr(state, "runtime", None)
        if runtime is None:
            state.aime_parent_runtime_released = True
            return {
                "released": False,
                "reason": "runtime_missing",
                "allocator_after": _mlx_allocator_public_stats(),
            }
        started = time.perf_counter()
        model_path = str(getattr(runtime, "model_path", getattr(state.args, "model", "")))
        mtp_enabled = bool(getattr(runtime, "mtp_enabled", False))
        lock = getattr(state, "lock", None)
        acquired = False
        if lock is not None and hasattr(lock, "acquire"):
            acquired = bool(lock.acquire(blocking=False))
            if not acquired:
                return {
                    "released": False,
                    "reason": "model_lock_busy",
                    "allocator_after": _mlx_allocator_public_stats(),
                }
        try:
            state.runtime = None
            state.draft_lm_head = {
                "installed": False,
                "reason": "aime_parent_runtime_released",
            }
            state.draft_head_identity = None
            state.template_hash = None
            state.aime_parent_runtime_released = True
            gc.collect()
            cleanup = _clear_mlx_cache_after_request(
                state,
                reason="aime_parent_runtime_release",
            )
            return {
                "released": True,
                "model_path": model_path,
                "mtp_enabled": mtp_enabled,
                "duration_ms": int(round((time.perf_counter() - started) * 1000.0)),
                "mlx_cache_cleanup": cleanup,
                "allocator_after": _mlx_allocator_public_stats(),
            }
        finally:
            if acquired:
                lock.release()

    def _aime_worker_argv(*, port: int, app_launch_id: str) -> list[str]:
        raw_args = list(getattr(state.args, "_raw_args", sys.argv[1:]))
        skip_value_flags = {
            "--host",
            "--port",
            "--app-launch-id",
            "--clear-cache-every",
        }
        drop_flags = {
            "--open-browser",
            "--open-dashboard",
            "--launch-pi",
            "--launch-opencode",
            "--server-console",
        }
        child_args: list[str] = []
        skip_next = False
        for arg in raw_args:
            if skip_next:
                skip_next = False
                continue
            if arg in drop_flags:
                continue
            if arg in skip_value_flags:
                skip_next = True
                continue
            if any(arg.startswith(flag + "=") for flag in skip_value_flags):
                continue
            child_args.append(arg)
        child_args.extend(
            [
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--app-launch-id",
                app_launch_id,
                "--clear-cache-every",
                str(_aime_clear_cache_every()),
            ]
        )
        return child_args

    async def _wait_for_aime_worker_health(
        proc: subprocess.Popen[Any],
        *,
        base_url: str,
        api_key: str | None,
        timeout_s: float = 180.0,
    ) -> dict[str, Any]:
        import httpx

        deadline = time.monotonic() + timeout_s
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        last_error: str | None = None
        async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as client:
            while time.monotonic() < deadline:
                returncode = proc.poll()
                if returncode is not None:
                    raise RuntimeError(
                        f"AIME worker exited before health check: returncode={returncode}"
                    )
                try:
                    response = await client.get(base_url.rstrip("/") + "/health", headers=headers)
                    if response.status_code == 200:
                        payload = response.json()
                        if isinstance(payload, dict) and payload.get("ok"):
                            return payload
                        last_error = f"unhealthy payload: {payload!r}"
                    else:
                        last_error = f"HTTP {response.status_code}"
                except Exception as exc:  # noqa: BLE001 - startup polls are best effort
                    last_error = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(0.25)
        raise RuntimeError(
            "AIME worker did not become healthy"
            + (f": {last_error}" if last_error else "")
        )

    def _aime_worker_log_tail(path: Path, *, max_bytes: int = 4000) -> str:
        try:
            data = path.read_bytes()
        except OSError:
            return ""
        return data[-max_bytes:].decode("utf-8", errors="replace").strip()

    async def _stop_aime_worker(proc: subprocess.Popen[Any]) -> dict[str, Any]:
        started = time.monotonic()
        if proc.poll() is None:
            proc.terminate()
            try:
                await asyncio.to_thread(proc.wait, timeout=12.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                await asyncio.to_thread(proc.wait, timeout=12.0)
        returncode = proc.poll()
        drain_s = _aime_worker_drain_s()
        if drain_s > 0:
            await asyncio.sleep(drain_s)
        return {
            "ok": returncode is not None,
            "pid": proc.pid,
            "returncode": returncode,
            "post_exit_drain_ms": int(round(drain_s * 1000.0)),
            "duration_ms": int(round((time.monotonic() - started) * 1000.0)),
        }

    async def _aime_question_process_runtime_factory(
        runner: Any,
        problem: Any,
    ) -> Any:
        from mtplx.benchmarks.runners.aime import AIMEQuestionRuntime

        port = _free_loopback_port()
        idx = int(getattr(runner, "current_idx", None) or getattr(problem, "index", 0) or 0)
        attempt = int(getattr(runner, "current_attempt", None) or 1)
        parent_launch_id = (
            str(getattr(state.args, "app_launch_id", None) or "").strip()
            or "mtplx"
        )
        app_launch_id = (
            f"{parent_launch_id}-aime-q{idx}-a{attempt}-{uuid.uuid4().hex[:6]}"
        )
        child_args = _aime_worker_argv(port=port, app_launch_id=app_launch_id)
        env = dict(os.environ)
        env["MTPLX_AIME_PROCESS_ISOLATION"] = "off"
        env["MTPLX_APP_LAUNCH_ID"] = app_launch_id
        env["MTPLX_AIME_PARENT_PID"] = str(os.getpid())
        base_url = f"http://127.0.0.1:{port}"
        log_dir = Path.home() / ".mtplx" / "benchmarks" / "aime" / "worker-logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{app_launch_id}.log"
        log_file = log_path.open("ab", buffering=0)
        proc = subprocess.Popen(
            [sys.executable, "-m", "mtplx.server.openai", *child_args],
            cwd=str(ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            health = await _wait_for_aime_worker_health(
                proc,
                base_url=base_url,
                api_key=getattr(state.args, "api_key", None),
            )
        except Exception as exc:
            await _stop_aime_worker(proc)
            log_file.close()
            log_tail = _aime_worker_log_tail(log_path)
            if log_tail:
                raise RuntimeError(
                    f"{exc}; AIME worker log {log_path}: {log_tail}"
                ) from exc
            raise

        async def cleanup() -> dict[str, Any]:
            try:
                return await _stop_aime_worker(proc)
            finally:
                log_file.close()

        return AIMEQuestionRuntime(
            base_url=base_url,
            cleanup=cleanup,
            metadata={
                "mode": "per_question_process",
                "pid": proc.pid,
                "port": port,
                "app_launch_id": app_launch_id,
                "log_path": str(log_path),
                "model": health.get("model"),
                "profile_name": (health.get("profile") or {}).get("name")
                if isinstance(health.get("profile"), dict)
                else None,
                "generation_mode": health.get("generation_mode"),
                "clear_cache_every": health.get("clear_cache_every")
                or env.get("MTPLX_CLEAR_CACHE_EVERY"),
            },
        )

    class _AIMEStartBody(BaseModel):
        model_config = ConfigDict(extra="ignore")
        year: int = 2026
        # Sampler overrides; omit any to inherit the daemon's preset
        # default (matches commit 25ae0fe "Restore product sampler
        # defaults for launch QA").
        temperature: float | None = None
        top_p: float | None = None
        top_k: int | None = None
        max_tokens: int | None = None
        enable_thinking: bool | None = None
        answer_verification: str | None = None
        answer_verification_attempts: int | None = None
        cap_recovery: str | None = None
        visible_submission_max_tokens: int | None = None
        question_process_isolation: str | None = None
        question_limit: int | None = None

    async def _aime_question_isolation_cleanup(
        _runner: Any,
        _result: Any,
        request_id: str,
    ) -> dict[str, Any]:
        """Hard isolation boundary between native AIME questions.

        AIME rows are intentionally stateless. The OpenAI bridge already
        bypasses SessionBank for AIME requests and clears idle MLX cache after
        natural completion; this boundary also catches handoff/cancel paths and
        clears app-owned session state before the next question starts.
        """

        sessions_cleared: Any = None
        try:
            sessions_cleared = state.sessions.clear_all()
        except Exception as exc:  # noqa: BLE001 - keep the benchmark alive
            sessions_cleared = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        cleanup_started_s = time.perf_counter()
        cleanup_attempts: list[dict[str, Any]] = []
        cleanup: dict[str, Any] = {}
        for _ in range(20):
            cleanup = _clear_mlx_cache_after_request(
                state,
                reason="aime_question_boundary",
            )
            cleanup_attempts.append(cleanup)
            if cleanup.get("reason") != "model_lock_busy":
                break
            await asyncio.sleep(0.05)
        if len(cleanup_attempts) > 1:
            cleanup = dict(cleanup)
            cleanup["attempts"] = len(cleanup_attempts)
            cleanup["waited_s"] = round(time.perf_counter() - cleanup_started_s, 3)
            cleanup["attempt_reasons"] = [
                str(attempt.get("reason") or "") for attempt in cleanup_attempts
            ]
        return {
            "ok": bool(cleanup.get("cleared")),
            "request_id": request_id,
            "session_cache_clear": sessions_cleared,
            "mlx_cache_cleanup": cleanup,
            "allocator_after": _mlx_allocator_public_stats(),
        }

    def _aime_runner_kwargs(body: "_AIMEStartBody | None") -> dict[str, Any]:
        decode_profile = getattr(state.args, "profile", None) or "sustained"
        mtp_enabled: bool | None
        try:
            mtp_enabled = bool(
                getattr(state, "mtp_enabled", None)
                if getattr(state, "mtp_enabled", None) is not None
                else not bool(getattr(state.args, "no_mtp", False))
            )
        except Exception:
            mtp_enabled = None
        try:
            depth_raw = int(
                getattr(state, "depth", 0) or getattr(state.args, "depth", 0)
            )
        except Exception:
            depth_raw = 0
        kwargs: dict[str, Any] = {
            "decode_profile": decode_profile,
            "mtp_enabled": mtp_enabled,
            "depth": depth_raw if depth_raw > 0 else None,
            "model_id": state.model_id,
            "base_url": _startup_server_url(state.args),
            "api_key": getattr(state.args, "api_key", None),
            "question_isolation_factory": _aime_question_isolation_cleanup,
        }
        if _aime_process_isolation_mode(body) == "per_question":
            kwargs["question_runtime_factory"] = (
                _aime_question_process_runtime_factory
            )
        if body is not None:
            if body.temperature is not None:
                kwargs["temperature"] = body.temperature
            if body.top_p is not None:
                kwargs["top_p"] = body.top_p
            if body.top_k is not None:
                kwargs["top_k"] = body.top_k
            if body.max_tokens is not None:
                kwargs["max_tokens"] = body.max_tokens
            if body.enable_thinking is not None:
                kwargs["enable_thinking"] = body.enable_thinking
            else:
                kwargs["enable_thinking"] = bool(
                    getattr(state.args, "enable_thinking", True)
                )
            if body.answer_verification is not None:
                kwargs["answer_verification"] = body.answer_verification
            if body.answer_verification_attempts is not None:
                kwargs["answer_verification_attempts"] = (
                    body.answer_verification_attempts
                )
            if body.cap_recovery is not None:
                kwargs["cap_recovery"] = body.cap_recovery
            if body.visible_submission_max_tokens is not None:
                kwargs["visible_submission_max_tokens"] = (
                    body.visible_submission_max_tokens
                )
        return kwargs

    @app.post("/v1/mtplx/benchmarks/aime/start")
    async def aime_start(
        body: dict[str, Any] | None = Body(default=None),
    ) -> JSONResponse:
        from mtplx.benchmarks.runners import aime as aime_runner

        parsed_body = _AIMEStartBody.model_validate(body or {})
        year = int(parsed_body.year) if parsed_body.year else 2026
        if year != 2026:
            raise HTTPException(
                status_code=400,
                detail=f"only AIME 2026 is shipped (got year={year})",
            )
        kwargs = _aime_runner_kwargs(parsed_body)
        if parsed_body.question_limit is not None:
            question_limit = int(parsed_body.question_limit)
            if question_limit < 1 or question_limit > 30:
                raise HTTPException(
                    status_code=400,
                    detail="question_limit must be between 1 and 30",
                )
            kwargs["problems"] = aime_runner.load_dataset()[:question_limit]
        parent_runtime_release: dict[str, Any] | None = None
        if _aime_release_parent_runtime_enabled(parsed_body):
            parent_runtime_release = _release_parent_runtime_for_aime()
            if parent_runtime_release.get("reason") == "model_lock_busy":
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "aime_parent_runtime_release_busy",
                        "release": parent_runtime_release,
                    },
                )
        try:
            runner = await aime_runner.start_run(year=year, **kwargs)
        except aime_runner.ConcurrentRunError as exc:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "run_in_progress",
                    "active_run_id": exc.active_run_id,
                },
            )
        snapshot = runner.snapshot()
        return JSONResponse(
            status_code=200,
            content={
                "run_id": runner.run_id,
                "total": runner.total,
                "model": runner.model_id,
                "year": runner.year,
                "started_at": snapshot.get("started_at"),
                "state": runner.state.value,
                "parent_runtime_release": parent_runtime_release,
            },
        )

    @app.get("/v1/mtplx/benchmarks/aime/active")
    def aime_active() -> dict[str, Any]:
        from mtplx.benchmarks.runners import aime as aime_runner

        return {"active_run_id": aime_runner.list_active_run_id()}

    @app.get("/v1/mtplx/benchmarks/aime/history")
    def aime_history(limit: int = 5) -> dict[str, Any]:
        """Return summary lines of the most recent N AIME runs.

        Reads JSONL files in `~/.mtplx/benchmarks/aime/` and returns the
        last-line `summary` entry from each (in mtime-descending order).
        """
        from mtplx.benchmarks.runners import aime as aime_runner

        directory = aime_runner.DEFAULT_PERSIST_DIR
        if not directory.is_dir():
            return {"runs": []}
        files = sorted(
            (p for p in directory.glob("*.jsonl") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        runs: list[dict[str, Any]] = []
        capped = max(1, min(int(limit or 5), 50))
        for path in files[:capped]:
            try:
                with path.open(encoding="utf-8") as handle:
                    last_summary: dict[str, Any] | None = None
                    for raw in handle:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            obj = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(obj, dict) and "summary" in obj:
                            last_summary = obj["summary"]
                if last_summary is not None:
                    runs.append({"run_id": path.stem, "path": str(path), **last_summary})
            except OSError:
                continue
        return {"runs": runs}

    @app.get("/v1/mtplx/benchmarks/aime/{run_id}")
    def aime_snapshot(run_id: str) -> dict[str, Any]:
        from mtplx.benchmarks.runners import aime as aime_runner

        run = aime_runner.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"unknown run_id {run_id}")
        return run.snapshot()

    @app.post("/v1/mtplx/benchmarks/aime/{run_id}/pause")
    async def aime_pause(run_id: str) -> dict[str, Any]:
        from mtplx.benchmarks.runners import aime as aime_runner

        run = aime_runner.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"unknown run_id {run_id}")
        await run.pause()
        return run.snapshot()

    @app.post("/v1/mtplx/benchmarks/aime/{run_id}/resume")
    async def aime_resume(run_id: str) -> dict[str, Any]:
        from mtplx.benchmarks.runners import aime as aime_runner

        run = aime_runner.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"unknown run_id {run_id}")
        await run.resume()
        return run.snapshot()

    @app.post("/v1/mtplx/benchmarks/aime/{run_id}/skip")
    async def aime_skip(run_id: str) -> dict[str, Any]:
        from mtplx.benchmarks.runners import aime as aime_runner

        run = aime_runner.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"unknown run_id {run_id}")
        await run.skip_current()
        return run.snapshot()

    @app.post("/v1/mtplx/benchmarks/aime/{run_id}/cancel")
    async def aime_cancel(run_id: str) -> dict[str, Any]:
        from mtplx.benchmarks.runners import aime as aime_runner

        run = aime_runner.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"unknown run_id {run_id}")
        await run.cancel()
        return run.snapshot()

    @app.get("/v1/mtplx/benchmarks/aime/{run_id}/stream")
    async def aime_stream(run_id: str) -> StreamingResponse:
        from mtplx.benchmarks.runners import aime as aime_runner

        run = aime_runner.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"unknown run_id {run_id}")

        terminal_kinds = {"run_done", "run_cancelled", "error"}
        queue, replay = run.subscribe()

        async def event_stream():
            saw_terminal = False
            try:
                for ev in replay:
                    yield (
                        f"event: {ev.get('event', 'message')}\n"
                        f"data: {json.dumps(_json_safe(ev))}\n\n"
                    )
                    if ev.get("event") in terminal_kinds:
                        saw_terminal = True
                if saw_terminal:
                    return
                while True:
                    try:
                        ev = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    yield (
                        f"event: {ev.get('event', 'message')}\n"
                        f"data: {json.dumps(_json_safe(ev))}\n\n"
                    )
                    if ev.get("event") in terminal_kinds:
                        break
            except asyncio.CancelledError:
                raise
            finally:
                run.unsubscribe(queue)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/v1/mtplx/metrics/stream")
    async def mtplx_metrics_stream(
        snapshot_interval_ms: int | None = None,
    ) -> StreamingResponse:
        bus = state.dashboard.bus
        queue = bus.subscribe()
        snapshot_interval_s = _dashboard_snapshot_interval_s(snapshot_interval_ms)

        async def event_stream():
            try:
                snapshot = _mtplx_dashboard_snapshot(state)
                yield (
                    "event: snapshot\n"
                    f"data: {json.dumps(_json_safe(snapshot))}\n\n"
                )
                last_snapshot_s = time.perf_counter()
                while True:
                    timeout_s = max(
                        0.01,
                        snapshot_interval_s
                        - (time.perf_counter() - last_snapshot_s),
                    )
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=timeout_s)
                        yield (
                            f"event: {event.get('kind', 'event')}\n"
                            f"data: {json.dumps(_json_safe(event))}\n\n"
                        )
                    except asyncio.TimeoutError:
                        pass
                    if (
                        time.perf_counter() - last_snapshot_s
                    ) >= snapshot_interval_s:
                        snapshot = _mtplx_dashboard_snapshot(state)
                        yield (
                            "event: snapshot\n"
                            f"data: {json.dumps(_json_safe(snapshot))}\n\n"
                        )
                        last_snapshot_s = time.perf_counter()
            except asyncio.CancelledError:
                raise
            finally:
                bus.unsubscribe(queue)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return {
            "latest": state.last_metrics[-1] if state.last_metrics else None,
            "recent": state.last_metrics[-32:],
            "tool_parse_counters": dict(
                getattr(state, "tool_parse_counters", {}) or {}
            ),
        }

    @app.get("/admin/sessions")
    def admin_sessions() -> dict[str, Any]:
        return state.sessions.list_sessions()

    @app.post("/admin/sessions/{session_id}/clear")
    def admin_clear_session(session_id: str) -> dict[str, Any]:
        return state.sessions.clear_session(session_id)

    @app.post("/admin/cache/clear")
    def admin_clear_cache() -> dict[str, Any]:
        cleared = state.sessions.clear_all()
        if isinstance(cleared, dict):
            cleared["mlx_cache_cleanup"] = _clear_mlx_cache_after_request(
                state,
                reason="admin_cache_clear",
            )
            return cleared
        return {
            "cleared": cleared,
            "mlx_cache_cleanup": _clear_mlx_cache_after_request(
                state,
                reason="admin_cache_clear",
            ),
        }

    @app.get("/admin/cache/ssd")
    def admin_ssd_cache() -> dict[str, Any]:
        tier = getattr(state, "session_bank_cold_tier", None)
        if tier is None or not hasattr(tier, "stats"):
            return {"enabled": False}
        return tier.stats()

    @app.post("/admin/cache/ssd/archive")
    def admin_archive_ssd_cache() -> dict[str, Any]:
        archived = state.sessions.archive_cold_tier()
        archived["ram_cache_unchanged"] = True
        return archived

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": state.model_id,
                    "object": "model",
                    "created": now,
                    "owned_by": "mtplx",
                    "context_length": state.context_window,
                    "max_context_length": state.context_window,
                    "max_model_len": state.context_window,
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        raw_request: Request, request: ChatCompletionRequest
    ) -> Any:
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        headers = dict(raw_request.headers)
        metadata = _request_metadata(request)
        request_max_tokens = _request_max_tokens(request)
        requested_model = request.model
        model = state.model_id
        response_id = _response_id_from_client_hint(
            prefix="chatcmpl",
            headers=headers,
            metadata=metadata,
        )
        created = int(time.time())
        if _is_opencode_title_request(request):
            return _opencode_title_response(
                state,
                request,
                model=model,
                response_id=response_id,
                created=created,
                request_max_tokens=request_max_tokens,
            )
        cache_bypass = headers.get("x-mtplx-cache-mode", "").lower() in {
            "bypass",
            "stateless",
            "off",
        } or str(metadata.get("cache_mode", "")).lower() in {
            "bypass",
            "stateless",
            "off",
        }
        opencode_client = _is_opencode_client(headers=headers, metadata=metadata)
        requested_tool_specs = _normalize_tool_specs(request.tools)
        tool_specs = _filter_tool_specs_for_request(
            requested_tool_specs,
            request.messages,
            tool_choice=request.tool_choice,
        )
        tools_active = _tools_active_for_request(tool_specs, request.tool_choice)
        raw_tool_result_history_present = any(
            str(message.role).lower() == "tool" for message in request.messages
        )
        agent_transcript_tools_active = bool(
            tools_active
            or (
                requested_tool_specs
                and raw_tool_result_history_present
                and _is_read_only_inspection_request(_last_user_text(request.messages))
            )
        )
        read_only_force_answer_contract_active = (
            _request_should_force_answer_for_read_only_inspection(request.messages)
        )
        if read_only_force_answer_contract_active:
            if (
                _tool_result_message_count(request.messages) > 0
                and _request_explicit_single_tool_then_answer(request.messages)
            ):
                # Explicit "use one tool then answer": the forced final turn
                # generates tool-free, and turn-level tool state/observability
                # must agree (zero remaining tools, read_only_force_answer:v1
                # policy version).
                tool_specs = []
                tools_active = False
            else:
                # Read-budget force answer: keep the read-only inspection
                # toolset so cited evidence stays greppable, instead of
                # returning zero tools.
                tool_specs = [
                    tool
                    for tool in requested_tool_specs
                    if (_tool_spec_name(tool) or "").strip().lower()
                    in _READ_ONLY_FORCE_ANSWER_TOOL_NAMES
                ]
                tools_active = _tools_active_for_request(
                    tool_specs, request.tool_choice
                )
        no_tools_contract_active = bool(
            not read_only_force_answer_contract_active
            and _should_add_no_tool_contract(
                requested_tools=requested_tool_specs,
                tools_active=tools_active,
                messages=request.messages,
            )
        )
        client_controls_allowed = _client_controls_allowed(headers, metadata)
        pi_convergence_contract_active = bool(
            not read_only_force_answer_contract_active
            and not no_tools_contract_active
            and _request_should_add_pi_convergence_contract(
                request.messages,
                headers=headers,
                metadata=metadata,
                tools_active=agent_transcript_tools_active,
            )
        )
        opencode_prompt_contract_profile = _opencode_prompt_contract_profile(
            request.messages,
            headers=headers,
            metadata=metadata,
            tool_choice=request.tool_choice,
        )
        opencode_prompt_contract_system_prompt = (
            _opencode_prompt_contract_system_prompt(opencode_prompt_contract_profile)
        )
        opencode_simple_chat_contract_active = False
        messages_for_generation, transcript_stats = _canonicalize_agent_transcript(
            request.messages,
            tools_active=agent_transcript_tools_active,
            replace_simple_chitchat_system_prompt=False,
            initial_client_system_prompt=opencode_prompt_contract_system_prompt,
            strip_tool_call_preamble_text=opencode_client,
        )
        messages_for_generation, backend_chat_policy_active = _with_backend_chat_policy(
            state,
            messages_for_generation,
        )
        if read_only_force_answer_contract_active:
            messages_for_generation = _with_mtplx_read_only_force_answer_contract(
                messages_for_generation
            )
        elif no_tools_contract_active:
            messages_for_generation = _with_mtplx_no_tool_contract(
                messages_for_generation
            )
        elif pi_convergence_contract_active:
            messages_for_generation = _with_mtplx_pi_convergence_contract(
                messages_for_generation
            )
        read_only_inspection_request = _is_read_only_inspection_request(
            _last_user_text(messages_for_generation)
        )
        tool_result_history_present = any(
            str(message.role).lower() == "tool" for message in messages_for_generation
        )
        raw_messages_for_postcommit = (
            list(request.messages)
            if read_only_force_answer_contract_active
            else (
                list(messages_for_generation)
                if (
                    no_tools_contract_active
                    or pi_convergence_contract_active
                    or opencode_prompt_contract_profile is not None
                    or backend_chat_policy_active
                )
                else list(request.messages)
            )
        )
        postcommit_tool_specs = (
            tool_specs
            if tools_active
            else (requested_tool_specs if agent_transcript_tools_active else None)
        )
        background = is_background_request(
            messages=messages_for_generation,
            max_tokens=request_max_tokens,
            headers=headers,
            metadata=metadata,
            main_system_hash=state.main_system_prompt_hash,
        )
        if background and (
            state.has_foreground()
            or state.lock.locked()
            or _foreground_model_work_pending(state)
        ):
            return JSONResponse(
                status_code=503,
                headers={"Retry-After": "1"},
                content={
                    "error": {
                        "message": "background Open WebUI task bypassed while foreground generation is busy",
                        "type": CacheMissReason.SESSION_BUSY.value,
                    },
                    "mtplx_stats": {
                        "session_cache_hit": False,
                        "cache_miss_reason": CacheMissReason.SESSION_BUSY.value,
                        "session_restore_mode": "background_bypass",
                    },
                },
            )
        thinking_enabled = _thinking_enabled_for_request(
            state,
            request,
            allow_client_controls=client_controls_allowed,
        )
        reasoning_effort = _reasoning_effort_for_state(
            state,
            thinking_enabled=thinking_enabled,
            request_effort=request.reasoning_effort,
            allow_client_controls=client_controls_allowed,
        )
        if (
            read_only_force_answer_contract_active
            and _reasoning_parser_for_state(state) == "gemma4"
        ):
            thinking_enabled = False
        aime_visible_working = (
            _aime_visible_working_for_request(metadata)
            and thinking_enabled
            and _reasoning_parser_for_state(state) in {"qwen3", "step3p5"}
        )
        tool_prompt_mode, tool_prompt_mode_resolution = _tool_prompt_mode_for_request(
            state.args,
            headers=headers,
            metadata=metadata,
            tools_active=tools_active,
        )
        template_tool_prompt_mode = tool_prompt_mode
        if read_only_force_answer_contract_active and tools_active:
            # Read-budget force-answer turns keep the read-only toolset with
            # real schemas in the template; the compact schema-free contract
            # would strip them and defeat the evidence-citing final turn.
            # Only the template/observability lane switches to hybrid — the
            # policy fingerprints keep the resolved launch/client mode so
            # SessionBank restore still matches postcommit.
            template_tool_prompt_mode = _TOOL_PROMPT_MODE_HYBRID
            tool_prompt_mode_resolution = {
                **tool_prompt_mode_resolution,
                "tool_prompt_mode_source": "read_only_force_answer",
            }
        postcommit_tool_prompt_mode = tool_prompt_mode
        if postcommit_tool_specs and not tools_active:
            postcommit_tool_prompt_mode, _ = _tool_prompt_mode_for_request(
                state.args,
                headers=headers,
                metadata=metadata,
                tools_active=True,
            )
        request_generation_mode = _request_generation_mode_for_generation(
            state,
            request,
            allow_client_controls=client_controls_allowed,
        )
        request_depth = _request_depth_for_generation(
            state,
            request,
            generation_mode=request_generation_mode,
            allow_client_controls=client_controls_allowed,
        )
        template_observability: dict[str, Any] = {}
        prompt_ids = _encode_messages(
            state.runtime.tokenizer,
            messages_for_generation,
            enable_thinking=thinking_enabled,
            reasoning_effort=reasoning_effort,
            strip_assistant_reasoning_history=state.args.strip_assistant_reasoning_history,
            tools=tool_specs if tools_active else None,
            tool_choice=request.tool_choice,
            tool_prompt_mode=template_tool_prompt_mode,
            template_observability=template_observability,
        )
        if aime_visible_working:
            prompt_ids = [
                *prompt_ids,
                *_encode_rendered_chat_text(
                    state.runtime.tokenizer,
                    f"{THINK_CLOSE}\n",
                ),
            ]
            template_observability["aime_visible_working"] = True
            template_observability["aime_visible_working_prompt_close"] = True
        request_depth, short_depth_policy = _opencode_short_context_depth_policy(
            request,
            headers=headers,
            metadata=metadata,
            generation_mode=request_generation_mode,
            request_depth=request_depth,
            prompt_tokens=len(prompt_ids),
        )
        effective_request_depth, long_context_depth_policy = (
            _long_context_mtp_depth_policy_for_request(
                state,
                generation_mode=request_generation_mode,
                request_depth=request_depth,
                prompt_tokens=len(prompt_ids),
            )
        )
        current_system_hash = system_prompt_hash(messages_for_generation)
        if current_system_hash is not None and not background:
            state.main_system_prompt_hash = current_system_hash
        session_id: str | None = None
        session_source: str | None = None
        session = None
        cache_miss_reason = (
            CacheMissReason.BACKGROUND_BYPASS.value
            if background
            else CacheMissReason.NEW_SESSION.value
        )
        session_restore_mode = "background_bypass" if background else "cold"
        session_cache_scope = _session_cache_scope_for_request(
            state,
            headers=headers,
            metadata=metadata,
        )
        policy_fingerprint = _policy_fingerprint(
            state,
            thinking_enabled=thinking_enabled,
            generation_mode=request_generation_mode,
            depth=effective_request_depth,
            tools_active=tools_active,
            tool_prompt_mode=tool_prompt_mode,
            tool_choice=request.tool_choice,
            no_tools_contract_active=no_tools_contract_active,
            read_only_force_answer_contract_active=read_only_force_answer_contract_active,
            pi_convergence_contract_active=pi_convergence_contract_active,
            simple_chat_contract_active=opencode_simple_chat_contract_active,
            opencode_prompt_contract_profile=opencode_prompt_contract_profile,
            cache_scope=session_cache_scope,
        )
        postcommit_policy_fingerprint = policy_fingerprint
        if read_only_force_answer_contract_active:
            postcommit_policy_fingerprint = _policy_fingerprint(
                state,
                thinking_enabled=thinking_enabled,
                generation_mode=request_generation_mode,
                depth=effective_request_depth,
                tools_active=bool(postcommit_tool_specs),
                tool_prompt_mode=postcommit_tool_prompt_mode,
                tool_choice=request.tool_choice,
                no_tools_contract_active=False,
                read_only_force_answer_contract_active=False,
                pi_convergence_contract_active=False,
                simple_chat_contract_active=opencode_simple_chat_contract_active,
                opencode_prompt_contract_profile=opencode_prompt_contract_profile,
                cache_scope=session_cache_scope,
            )
        session_restore_policy_fingerprint = (
            postcommit_policy_fingerprint
            if read_only_force_answer_contract_active
            else policy_fingerprint
        )
        request_observability = _request_observability(
            request,
            headers=headers,
            metadata=metadata,
            session_source=session_source,
            request_generation_mode=request_generation_mode,
            request_depth=request_depth,
        )
        if read_only_force_answer_contract_active:
            request_observability["request_session_restore_policy"] = (
                "stable_without_transient_force_answer"
            )
            request_observability[
                "request_session_restore_policy_matches_postcommit"
            ] = bool(session_restore_policy_fingerprint == postcommit_policy_fingerprint)
        opencode_tool_history_policy = (
            _opencode_tool_history_restore_policy(
                headers=headers,
                metadata=metadata,
                tool_result_history_present=tool_result_history_present,
            )
            if not background and not cache_bypass
            else {
                "eligible": False,
                "cache_bypass": False,
                "live_frontier_restore": False,
                "force_clone_restore": False,
            }
        )
        opencode_tool_history_cache_bypass = bool(
            opencode_tool_history_policy["cache_bypass"]
        )
        opencode_tool_history_live_frontier_restore = bool(
            opencode_tool_history_policy["live_frontier_restore"]
        )
        opencode_tool_history_force_clone_restore = bool(
            opencode_tool_history_policy["force_clone_restore"]
        )
        if not background and not cache_bypass:
            requested_restore_mode = headers.get(
                "x-mtplx-restore-mode", "reference_lease"
            )
            requested_restore_mode = requested_restore_mode.replace("-", "_")
            if opencode_tool_history_live_frontier_restore:
                requested_restore_mode = "reference_lease"
            elif opencode_tool_history_force_clone_restore:
                requested_restore_mode = "clone"
            session_restore_mode = (
                "clone" if requested_restore_mode == "clone" else "reference_lease"
            )
            if opencode_tool_history_cache_bypass:
                cache_miss_reason = "opencode_tool_history_cache_bypass"
                session_restore_mode = "opencode_tool_history_bypass"
            session_id, session_source = state.sessions.resolve_session_id(
                headers=headers,
                metadata=metadata,
                user=_request_extra(request, "user"),
                chat_id=_request_extra(request, "chat_id"),
                conversation_id=_request_extra(request, "conversation_id"),
                prompt_ids=prompt_ids,
            )
            session = state.sessions.get_or_create(session_id)
            session.last_cache_miss_reason = cache_miss_reason
            session.last_restore_mode = session_restore_mode
        if requested_model:
            request_observability["request_model"] = requested_model
            request_observability["served_model_id"] = state.model_id
            request_observability["request_model_matches_served_model"] = (
                requested_model == state.model_id
            )
        server_reasoning_mode = getattr(state.args, "reasoning", None)
        if server_reasoning_mode not in {"auto", "on", "off"}:
            server_reasoning_mode = (
                "on" if bool(getattr(state.args, "enable_thinking", True)) else "off"
            )
        request_observability["request_effective_mtp_depth"] = int(
            effective_request_depth
        )
        if not client_controls_allowed:
            request_reasoning_mode = (
                "off" if not thinking_enabled else server_reasoning_mode
            )
        elif request.enable_thinking is False:
            request_reasoning_mode = "off"
        elif request.enable_thinking is True and server_reasoning_mode == "auto":
            request_reasoning_mode = "on"
        else:
            request_reasoning_mode = server_reasoning_mode
        request_observability["request_reasoning_mode"] = request_reasoning_mode
        request_observability["request_enable_thinking"] = bool(thinking_enabled)
        request_observability["request_reasoning_effort"] = reasoning_effort
        request_observability["request_enable_thinking_override"] = (
            request.enable_thinking is not None and client_controls_allowed
        )
        request_observability["mtplx_control_owner"] = (
            "client" if client_controls_allowed else "server"
        )
        request_observability["client_controls_allowed"] = bool(
            client_controls_allowed
        )
        if not client_controls_allowed:
            ignored_fields = _ignored_client_control_fields(request)
            if ignored_fields:
                request_observability["client_control_fields_ignored"] = (
                    ignored_fields
                )
        request_observability["request_reasoning_parser"] = (
            _reasoning_parser_for_state(state)
        )
        request_observability["request_read_only_inspection_force_answer"] = bool(
            read_only_force_answer_contract_active
        )
        request_observability["request_read_only_inspection_tool_result_count"] = (
            _tool_result_message_count(request.messages)
        )
        request_observability[
            "request_read_only_inspection_force_answer_after_tools"
        ] = _read_only_inspection_force_answer_after_tools()
        request_observability["request_pi_convergence_contract"] = bool(
            pi_convergence_contract_active
        )
        request_observability["request_pi_convergence_tool_result_count"] = (
            _tool_result_message_count(request.messages)
        )
        request_observability["request_pi_convergence_after_tools"] = (
            _pi_convergence_after_tools()
        )
        request_observability["opencode_simple_chat_contract_active"] = bool(
            opencode_simple_chat_contract_active
        )
        request_observability["opencode_prompt_contract_profile"] = (
            opencode_prompt_contract_profile or "none"
        )
        request_observability["backend_chat_policy_active"] = bool(
            backend_chat_policy_active
        )
        request_observability["request_effective_message_count"] = len(
            messages_for_generation
        )
        request_observability["request_effective_message_roles"] = [
            message.role for message in messages_for_generation
        ]
        request_observability["request_effective_message_chars"] = [
            len(_content_to_text(message.content))
            for message in messages_for_generation
        ]
        request_observability["preserve_thinking"] = getattr(
            state.args, "preserve_thinking", "auto"
        )
        request_observability["preserve_thinking_effective"] = (
            _preserve_thinking_effective(state.args)
        )
        request_observability["strip_assistant_reasoning_history"] = bool(
            state.args.strip_assistant_reasoning_history
        )
        request_observability["long_context_mtp_depth_policy"] = (
            long_context_depth_policy
        )
        request_observability.update(
            _bridge_policy_observability(
                tools_active=tools_active,
                tool_prompt_mode=template_tool_prompt_mode,
                no_tools_contract_active=no_tools_contract_active,
                read_only_force_answer_contract_active=read_only_force_answer_contract_active,
                pi_convergence_contract_active=pi_convergence_contract_active,
            )
        )
        request_observability.update(tool_prompt_mode_resolution)
        request_observability["session_cache_scope"] = session_cache_scope
        request_observability["opencode_tool_history_cache_bypass"] = bool(
            opencode_tool_history_cache_bypass
        )
        request_observability["opencode_tool_history_force_clone_restore"] = bool(
            opencode_tool_history_force_clone_restore
        )
        request_observability["opencode_tool_history_live_frontier_restore"] = bool(
            opencode_tool_history_live_frontier_restore
        )
        requested_tool_names = list(request_observability.get("request_tool_names") or [])
        filtered_tool_names = _tool_names(tool_specs) if tools_active else []
        hidden_tool_names = [
            name for name in requested_tool_names if name not in filtered_tool_names
        ]
        request_observability.update(
            {
                "request_filtered_tool_count": len(filtered_tool_names),
                "request_filtered_tool_names": filtered_tool_names,
                "request_hidden_tool_names": hidden_tool_names,
                "request_tools_hidden_by_bridge": bool(hidden_tool_names),
            }
        )
        chat_template_report = getattr(state, "chat_template_report", {}) or {}
        request_observability.update(
            {
                "chat_template_profile": str(
                    chat_template_report.get("profile")
                    or getattr(state, "chat_template_profile", _CHAT_TEMPLATE_PROFILE_LOCAL)
                ),
                "chat_template_source": chat_template_report.get("source"),
                "chat_template_path": chat_template_report.get("path"),
                "chat_template_hash": state.template_hash,
            }
        )
        request_observability["opencode_short_context_depth_policy"] = (
            short_depth_policy
        )
        request_observability.update(transcript_stats.to_metrics())
        request_observability.update(template_observability)
        if template_observability.get("tool_template_fallback"):
            _record_tool_parse_event(state, event="tool_template_fallback")
        if request_observability.get("request_client_hint") == "android_studio":
            _record_tool_parse_event(state, event="android_studio_request_detected")
        prefix_diagnostic = getattr(state.sessions, "last_prefix_diagnostic", None)
        if isinstance(prefix_diagnostic, dict):
            request_observability["request_session_prefix_diagnostic"] = (
                prefix_diagnostic
            )
        session_keep_live_ref = _session_keep_live_refs_for_request(
            session_source=session_source,
            session_id=session_id,
            tool_names=_tool_names(tool_specs) if tools_active else None,
        )
        live_frontier_policy = "none"
        if agent_transcript_tools_active:
            live_frontier_policy = (
                "live_reference_lease"
                if session_keep_live_ref
                else "snapshot_only"
            )
        if (
            _is_opencode_client(headers=headers, metadata=metadata)
            and agent_transcript_tools_active
        ):
            if _opencode_tool_history_live_frontier_enabled():
                session_keep_live_ref = True
                live_frontier_policy = "opencode_live_reference_lease"
                request_observability["request_session_keep_live_ref_reason"] = (
                    "opencode_tool_live_frontier"
                )
            else:
                session_keep_live_ref = False
                live_frontier_policy = "opencode_snapshot_only"
                request_observability["request_session_keep_live_ref_reason"] = (
                    "opencode_tool_snapshot_only"
                )
        request_observability["request_session_keep_live_ref"] = bool(
            session_keep_live_ref
        )
        request_observability["live_frontier_candidate"] = bool(
            agent_transcript_tools_active
        )
        request_observability["live_frontier_result_turn"] = bool(
            agent_transcript_tools_active and tool_result_history_present
        )
        request_observability["live_frontier_policy"] = live_frontier_policy
        if agent_transcript_tools_active:
            live_frontier_tool_call_ids = _tool_call_ids_from_messages(request.messages)
            live_frontier_tool_result_ids = _tool_result_ids_from_messages(
                request.messages
            )
            request_observability["live_frontier_assistant_tool_call_count"] = len(
                live_frontier_tool_call_ids
            )
            request_observability["live_frontier_tool_result_count"] = len(
                live_frontier_tool_result_ids
            )
            request_observability["live_frontier_unknown_tool_result_count"] = len(
                live_frontier_tool_result_ids - live_frontier_tool_call_ids
            )
        session_bank_for_generation = (
            None
            if background or cache_bypass or opencode_tool_history_cache_bypass
            else state.sessions.bank
        )
        request_observability["request_session_bank_bypass"] = (
            session_bank_for_generation is None
        )
        commit_prompt_prefix = _commit_prompt_prefix_for_request(
            state,
            prompt_ids=prompt_ids,
            tools_active=tools_active,
        )
        if read_only_force_answer_contract_active:
            # The forced-answer contract is a transient generation aid. OpenCode
            # will not echo it on the next turn, so caching it as a session
            # prefix turns a real follow-up into a low-boundary cache hit.
            commit_prompt_prefix = False
        request_observability["request_commit_prompt_prefix"] = bool(
            commit_prompt_prefix
        )
        sampler_temperature = request.temperature if client_controls_allowed else None
        sampler_top_p = request.top_p if client_controls_allowed else None
        sampler_top_k = request.top_k if client_controls_allowed else None
        request_observability["request_temperature"] = request.temperature
        request_observability["request_top_p"] = request.top_p
        request_observability["request_top_k"] = request.top_k
        ignored_sampler_fields = [
            name
            for name, value in (
                ("temperature", request.temperature),
                ("top_p", request.top_p),
                ("top_k", request.top_k),
            )
            if value is not None and not client_controls_allowed
        ]
        if ignored_sampler_fields:
            request_observability["client_sampler_fields_ignored"] = (
                ignored_sampler_fields
            )
        request_draft_sampler = _opencode_default_sampler_override(
            messages=messages_for_generation,
            tools_active=tools_active,
            request_temperature=request.temperature,
            request_top_p=request.top_p,
            request_top_k=request.top_k,
            request_observability=request_observability,
            default_temperature=getattr(state.args, "temperature", 0.6),
            default_top_p=getattr(state.args, "top_p", 0.95),
            default_top_k=getattr(state.args, "top_k", 20),
        )
        if request_draft_sampler is not None:
            target_sampler_override = request_draft_sampler
            sampler_temperature = target_sampler_override.temperature
            sampler_top_p = target_sampler_override.top_p
            sampler_top_k = target_sampler_override.top_k
            launch_draft_sampler = _opencode_default_draft_sampler_for_request(
                state,
                request_observability,
            )
            request_draft_sampler = launch_draft_sampler or target_sampler_override
            request_observability["draft_sampler_override"] = asdict(
                request_draft_sampler
            )
        if sampler_temperature is None:
            default_temperature = getattr(state.args, "temperature", None)
            sampler_temperature = (
                0.6 if default_temperature is None else default_temperature
            )
        if sampler_top_p is None:
            default_top_p = getattr(state.args, "top_p", None)
            sampler_top_p = 0.95 if default_top_p is None else default_top_p
        if sampler_top_k is None:
            default_top_k = getattr(state.args, "top_k", None)
            sampler_top_k = 20 if default_top_k is None else default_top_k
        request_observability["effective_temperature"] = float(sampler_temperature)
        request_observability["effective_top_p"] = float(sampler_top_p)
        request_observability["effective_top_k"] = int(sampler_top_k)
        suppress_visible_reasoning = False
        stop_sequences = _normalize_stop_sequences(request.stop)

        def _nonstream_on_prefill(progress: dict[str, Any]) -> None:
            _dashboard_publish_prefill(
                state,
                request_id=response_id,
                session_id=session_id,
                payload=progress,
            )

        nonstream_cancel_event = Event()
        nonstream_completion_tokens = 0
        nonstream_cancel_reason = "request_cancelled"
        nonstream_client_disconnected = False
        # Stop sequences are detected on the visible content channel as it
        # decodes, so a match aborts generation immediately instead of
        # burning tokens until EOS/max_tokens. Reasoning text never matches:
        # OpenAI clients only ever receive content, so stop strings only
        # apply there.
        nonstream_stop_monitor: _StopSequenceStreamMonitor | None = None
        nonstream_stop_decoder: _IncrementalTokenDecoder | None = None
        nonstream_stop_splitter: Any = None
        nonstream_stop_reasoning_chunks: list[str] = []
        if stop_sequences and not request.stream:
            nonstream_stop_monitor = _StopSequenceStreamMonitor(stop_sequences)
            nonstream_stop_decoder = _IncrementalTokenDecoder(
                state.runtime.tokenizer
            )
            nonstream_stop_splitter = _stream_splitter_for_state(
                state,
                thinking_enabled=thinking_enabled,
                recover_unclosed_reasoning_as_content=False,
                start_inside_thinking=not aime_visible_working,
            )

        def attach_response_observability(generated: dict[str, Any]) -> dict[str, Any]:
            stats = generated.setdefault("stats", {})
            for key in (
                "request_model",
                "served_model_id",
                "request_model_matches_served_model",
            ):
                if key in request_observability:
                    stats.setdefault(key, request_observability[key])
            return generated

        def _nonstream_on_tokens(new_tokens: list[int]) -> None:
            nonlocal nonstream_completion_tokens
            nonstream_completion_tokens += len(new_tokens)
            cancel_message = (
                "client disconnected"
                if nonstream_client_disconnected
                else "request cancelled"
            )
            _raise_if_stream_cancelled(
                nonstream_cancel_event, cancel_message
            )
            if nonstream_stop_monitor is not None:
                delta = nonstream_stop_decoder.feed(
                    [int(token) for token in new_tokens]
                )
                if not delta:
                    return
                for field, text in nonstream_stop_splitter.feed(delta):
                    if not text:
                        continue
                    if field == "reasoning_content":
                        nonstream_stop_reasoning_chunks.append(text)
                        continue
                    if field != "content":
                        continue
                    nonstream_stop_monitor.feed(text)
                    if nonstream_stop_monitor.stopped:
                        raise _StopSequenceHit(
                            nonstream_stop_monitor.matched_stop or ""
                        )

        def adopt_forked_session(acquired_session: Any) -> None:
            nonlocal session, session_id, session_keep_live_ref
            if acquired_session is session:
                return
            session = acquired_session
            session_id = session.session_id
            session_keep_live_ref = False
            request_observability["request_session_forked_busy"] = True
            request_observability["request_session_keep_live_ref"] = False
            request_observability["request_session_keep_live_ref_reason"] = (
                "forked_busy_session"
            )

        def run_generation_for_response() -> dict[str, Any]:
            if session is None:
                return attach_response_observability(
                    _run_generation_dispatched(
                        state,
                        prompt_ids,
                        batch_key="chat.nonstream",
                        response_id=response_id,
                        max_tokens=request_max_tokens,
                        temperature=sampler_temperature,
                        top_p=sampler_top_p,
                        top_k=sampler_top_k,
                        seed=request.seed,
                        draft_sampler=request_draft_sampler,
                        generation_mode=request_generation_mode,
                        depth=request_depth,
                        resolved_mtp_depth=effective_request_depth,
                        session_id=session_id,
                        cache_miss_reason=cache_miss_reason,
                        session_restore_mode=session_restore_mode,
                        session_bank=session_bank_for_generation,
                        session_template_hash=state.template_hash,
                        session_draft_head_identity=state.draft_head_identity,
                        session_policy_fingerprint=session_restore_policy_fingerprint,
                        background_request=background,
                        commit_prompt_prefix_to_bank=commit_prompt_prefix,
                        session_keep_live_ref=session_keep_live_ref,
                        request_observability=request_observability,
                        token_callback=_nonstream_on_tokens,
                        prefill_callback=_nonstream_on_prefill,
                        cancel_event=nonstream_cancel_event,
                        streaming_response=False,
                    )
                )
            with state.sessions.generation_slot(
                session, source=session_source
            ) as acquired_session:
                adopt_forked_session(acquired_session)
                generated_result = _run_generation_dispatched(
                    state,
                    prompt_ids,
                    batch_key="chat.nonstream",
                    response_id=response_id,
                    max_tokens=request_max_tokens,
                    temperature=sampler_temperature,
                    top_p=sampler_top_p,
                    top_k=sampler_top_k,
                    seed=request.seed,
                    draft_sampler=request_draft_sampler,
                    generation_mode=request_generation_mode,
                    depth=request_depth,
                    resolved_mtp_depth=effective_request_depth,
                    session_id=session_id,
                    cache_miss_reason=cache_miss_reason,
                    session_restore_mode=session_restore_mode,
                    session_bank=session_bank_for_generation,
                    session_template_hash=state.template_hash,
                    session_draft_head_identity=state.draft_head_identity,
                    session_policy_fingerprint=session_restore_policy_fingerprint,
                    commit_prompt_prefix_to_bank=commit_prompt_prefix,
                    session_keep_live_ref=session_keep_live_ref,
                    request_observability=request_observability,
                    token_callback=_nonstream_on_tokens,
                    prefill_callback=_nonstream_on_prefill,
                    cancel_event=nonstream_cancel_event,
                    streaming_response=False,
                )
                generated_result = attach_response_observability(generated_result)
                session.commit(
                    prompt_ids=prompt_ids,
                    generated_ids=generated_result["tokens"],
                    finish_reason=generated_result.get("finish_reason", "stop"),
                )
                return generated_result

        async def store_postcommit_snapshot(
            generated: dict[str, Any],
            *,
            assistant_content: str,
            assistant_tool_calls: list[dict[str, Any]] | None = None,
            stream_response: bool = False,
        ) -> None:
            if session is None:
                return
            started = time.perf_counter()
            compatibility = _generation_final_postcommit_compatibility(
                state,
                prompt_ids=prompt_ids,
                generated=generated,
                messages=raw_messages_for_postcommit,
                assistant_content=assistant_content,
                assistant_tool_calls=assistant_tool_calls,
                thinking_enabled=thinking_enabled,
                tool_specs=postcommit_tool_specs,
                tool_prompt_mode=postcommit_tool_prompt_mode,
                strip_tool_call_preamble_text=opencode_client,
            )
            if compatibility.get("safe"):
                generated["stats"]["session_postcommit_snapshot"] = {
                    "stored": True,
                    "mode": compatibility["mode"],
                    "reason": compatibility["reason"],
                    "prefix_len": len(compatibility["token_ids"]),
                    "elapsed_s": time.perf_counter() - started,
                    "history_suffix_tokens": int(
                        compatibility.get("history_suffix_tokens") or 0
                    ),
                }
                return
            unsafe_reason = str(compatibility.get("reason") or "unsafe_history")
            generated_mode = str(
                generated.get("stats", {}).get("generation_mode") or ""
            )
            skipped = _skipped_idle_postcommit_snapshot(
                state=state,
                unsafe_reason=unsafe_reason,
                assistant_tool_calls=assistant_tool_calls,
                prompt_prefix_len=len(prompt_ids),
            )
            if skipped is not None:
                generated["stats"]["session_postcommit_snapshot"] = (
                    _attach_skipped_postcommit_cleanup(state, skipped)
                )
                return
            if state.args.session_postcommit_mode == "async" and generated_mode != "ar":
                generated["stats"]["session_postcommit_snapshot"] = (
                    _schedule_idle_postcommit_snapshot(
                        state,
                        session_id=session_id,
                        messages=raw_messages_for_postcommit,
                        assistant_content=assistant_content,
                        assistant_tool_calls=assistant_tool_calls,
                        thinking_enabled=thinking_enabled,
                        policy_fingerprint=postcommit_policy_fingerprint,
                        unsafe_reason=unsafe_reason,
                        tool_specs=postcommit_tool_specs,
                        session=session,
                        expected_session_revision=getattr(session, "revision", None),
                        keep_live_ref=session_keep_live_ref,
                        tool_prompt_mode=postcommit_tool_prompt_mode,
                        strip_tool_call_preamble_text=opencode_client,
                    )
                )
                return
            postcommit = await asyncio.wrap_future(
                _submit_foreground_model_work(
                    state,
                    lambda: _store_retokenized_history_snapshot(
                        state,
                        session_id=session_id,
                        messages=raw_messages_for_postcommit,
                        assistant_content=assistant_content,
                        assistant_tool_calls=assistant_tool_calls,
                        thinking_enabled=thinking_enabled,
                        policy_fingerprint=postcommit_policy_fingerprint,
                        tool_specs=postcommit_tool_specs,
                        keep_live_ref=session_keep_live_ref,
                        tool_prompt_mode=postcommit_tool_prompt_mode,
                        strip_tool_call_preamble_text=opencode_client,
                    ),
                    batch_key=f"postcommit.inline:{session_id or 'stateless'}",
                ),
            )
            generated["stats"]["session_postcommit_snapshot"] = postcommit

        # Bounded wait for the prior turn's postcommit to land before we
        # admit this request to the model scheduler. Done HERE - off the
        # scheduler-owner thread, before any foreground submit and before
        # the session lock is acquired - so a slow postcommit cannot deadlock
        # against this request. The wait is best-effort: timeouts fall
        # through to a cold prefill, never a hang.
        postcommit_wait_outcome: dict[str, Any] | None = None
        if session is not None:
            postcommit_wait_outcome = await asyncio.to_thread(
                session.wait_for_pending_postcommit
            )
            request_observability["postcommit_wait"] = postcommit_wait_outcome
            if (
                postcommit_wait_outcome is not None
                and postcommit_wait_outcome.get("waited")
                and not _server_console_enabled(state)
            ):
                try:
                    _safe_stdout_print(
                        "[mtplx] postcommit-wait "
                        + json.dumps(
                            {
                                "session_id": session_id,
                                **postcommit_wait_outcome,
                            },
                            sort_keys=True,
                            default=str,
                        )
                    )
                except BaseException:
                    pass

        if request.stream:

            async def event_stream():
                stream_started_s = time.perf_counter()
                last_sse_sent_s = stream_started_s
                last_token_s: float | None = None
                next_silence_warn_s = stream_started_s + STREAM_SILENCE_WARN_S

                def mark_sse_sent(chunk: str) -> str:
                    nonlocal last_sse_sent_s
                    last_sse_sent_s = time.perf_counter()
                    return chunk

                first = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                yield mark_sse_sent(f"data: {json.dumps(first)}\n\n")

                queue: Queue[tuple[str, Any]] = Queue()
                cancel_event = Event()
                # Register this request in the dashboard's in-flight registry
                # so external cancel (`POST /v1/mtplx/cancel/{id}`) can flip
                # the same per-request cancel_event the worker already checks.
                # Deregistered in the stream's `finally`.
                in_flight_handle = InFlightHandle(
                    request_id=response_id,
                    cancel_event=cancel_event,
                    started_s=time.time(),
                    session_id=session_id,
                    model=model,
                    prompt_preview=_dashboard_prompt_preview(
                        request, state.runtime.tokenizer
                    ),
                    prompt_tokens=len(prompt_ids),
                )
                state.dashboard.in_flight.register(in_flight_handle)
                decoder = _IncrementalTokenDecoder(state.runtime.tokenizer)
                splitter = _stream_splitter_for_state(
                    state,
                    thinking_enabled=thinking_enabled,
                    recover_unclosed_reasoning_as_content=False,
                    start_inside_thinking=not aime_visible_working,
                )
                # Client stop sequences gate the visible content channel.
                # Forced final-answer turns own their visibility through the
                # buffered marker path, so they bypass stop monitoring (the
                # combination is an internal agent contract, not a client
                # completion surface).
                stop_monitor: _StopSequenceStreamMonitor | None = (
                    _StopSequenceStreamMonitor(stop_sequences)
                    if stop_sequences
                    and not read_only_force_answer_contract_active
                    else None
                )
                stop_sequence_cancel_fired = False

                def fire_stop_sequence_cancel() -> None:
                    nonlocal stop_sequence_cancel_fired
                    if stop_sequence_cancel_fired:
                        return
                    stop_sequence_cancel_fired = True
                    _cancel_stream_generation(cancel_event, generation_future)

                stream_interval = max(1, int(state.args.stream_interval))
                content_tool_translator = (
                    _ToolAwareContentStreamTranslator(
                        tools=tool_specs,
                        argument_chunk_chars=stream_interval,
                        tokenizer=state.runtime.tokenizer,
                        repair_unclosed_complete=(
                            str(raw_request.url.path or "") != "/v1/messages"
                        ),
                    )
                    # Forced final-answer turns stream sanitized visible text
                    # through the buffered marker path; the tool-call
                    # translator must not interleave with that buffer even
                    # when read-only tools remain active for the template.
                    if tools_active and not read_only_force_answer_contract_active
                    else None
                )
                stream_client_hint = str(
                    request_observability.get("request_client_hint") or ""
                ).lower()
                single_tool_call_stream = (
                    "pi" in stream_client_hint
                    or (
                        "opencode" in stream_client_hint
                        and _request_explicit_single_tool_then_answer(
                            messages_for_generation
                        )
                    )
                )
                orphan_stream_guard_enabled = bool(
                    tools_active and tool_result_history_present
                )
                orphan_reasoning_stream_guard = _InitialOrphanToolControlStreamGuard()
                orphan_content_stream_guard = _InitialOrphanToolControlStreamGuard()
                stream_orphan_tool_markup_suppressed = False
                pending_stream_tokens: list[int] = []
                commit_event = Event()
                commit_state = {
                    "commit": False,
                    "assistant_history_content": None,
                    "assistant_tool_calls": None,
                    "postcommit_snapshot": None,
                    "retokenize_inline": False,
                }

                def on_tokens(new_tokens: list[int]) -> None:
                    _raise_if_stream_cancelled(cancel_event)
                    int_tokens = [int(token) for token in new_tokens]
                    queue.put(
                        (
                            "tokens",
                            {
                                "tokens": list(int_tokens),
                                "timestamp_s": time.perf_counter(),
                            },
                        )
                    )
                    _raise_if_stream_cancelled(cancel_event)

                def on_prefill(progress: dict[str, Any]) -> None:
                    # Runs on the generation thread. Safe to call from any
                    # thread because _dashboard_publish_prefill uses the
                    # bus's loop.call_soon_threadsafe delivery.
                    _dashboard_publish_prefill(
                        state,
                        request_id=response_id,
                        session_id=session_id,
                        payload=progress,
                    )

                def maybe_retry_degenerate_read_only_inspection(
                    generated: dict[str, Any],
                ) -> dict[str, Any]:
                    if (
                        not tools_active
                        or not read_only_inspection_request
                        or not tool_result_history_present
                        or request.seed is not None
                    ):
                        return generated
                    first_text = _strip_generated_chat_template_sentinels(
                        str(generated.get("text") or "")
                    )
                    if first_text.strip():
                        return generated
                    first_completion_tokens = int(
                        generated.get("completion_tokens") or 0
                    )
                    if first_completion_tokens > 4:
                        return generated
                    first_stats = dict(generated.get("stats") or {})
                    retry_observability = dict(request_observability)
                    retry_observability.update(
                        {
                            "inspection_empty_retry_attempted": True,
                            "inspection_empty_retry_reason": (
                                "empty_tool_fed_read_only_inspection"
                            ),
                            "inspection_empty_retry_first_completion_tokens": (
                                first_completion_tokens
                            ),
                            "inspection_empty_retry_first_decode_tok_s": first_stats.get(
                                "decode_tok_s"
                            ),
                        }
                    )
                    retry_generated = _run_generation_dispatched(
                        state,
                        prompt_ids,
                        batch_key="chat.stream.inspection_empty_retry",
                        response_id=response_id,
                        max_tokens=request_max_tokens,
                        temperature=sampler_temperature,
                        top_p=sampler_top_p,
                        top_k=sampler_top_k,
                        seed=None,
                        draft_sampler=request_draft_sampler,
                        generation_mode=request_generation_mode,
                        depth=request_depth,
                        resolved_mtp_depth=effective_request_depth,
                        token_callback=on_tokens,
                        session_id=session_id,
                        cache_miss_reason=cache_miss_reason,
                        session_restore_mode=session_restore_mode,
                        session_bank=session_bank_for_generation,
                        session_template_hash=state.template_hash,
                        session_draft_head_identity=state.draft_head_identity,
                        session_policy_fingerprint=session_restore_policy_fingerprint,
                        background_request=background,
                        commit_final_state_to_bank=False,
                        commit_prompt_prefix_to_bank=commit_prompt_prefix,
                        session_keep_live_ref=session_keep_live_ref,
                        request_observability=retry_observability,
                        prefill_callback=on_prefill,
                        cancel_event=cancel_event,
                    )
                    retry_text = _strip_generated_chat_template_sentinels(
                        str(retry_generated.get("text") or "")
                    )
                    retry_stats = retry_generated.setdefault("stats", {})
                    retry_stats.update(retry_observability)
                    retry_succeeded = bool(retry_text.strip())
                    retry_stats["inspection_empty_retry_succeeded"] = retry_succeeded
                    if state.last_metrics:
                        state.last_metrics[-1].update(
                            {
                                "inspection_empty_retry_attempted": True,
                                "inspection_empty_retry_succeeded": retry_succeeded,
                                "inspection_empty_retry_reason": (
                                    "empty_tool_fed_read_only_inspection"
                                ),
                                "inspection_empty_retry_first_completion_tokens": (
                                    first_completion_tokens
                                ),
                                "inspection_empty_retry_first_decode_tok_s": (
                                    first_stats.get("decode_tok_s")
                                ),
                            }
                        )
                    return retry_generated

                def maybe_retry_degenerate_tool_fed_empty_completion(
                    generated: dict[str, Any],
                ) -> dict[str, Any]:
                    if (
                        not tools_active
                        or read_only_inspection_request
                        or not tool_result_history_present
                        or request.seed is not None
                    ):
                        return generated
                    first_text = _strip_mtplx_internal_continuation_markers(
                        _strip_generated_chat_template_sentinels(
                            str(generated.get("text") or "")
                        )
                    )
                    retry_reason = _tool_fed_degenerate_completion_reason(first_text)
                    if retry_reason is None:
                        return generated
                    first_stats = dict(generated.get("stats") or {})
                    repair_messages = list(messages_for_generation)
                    repair_messages.append(
                        ChatMessage(
                            role="user",
                            content=(
                                "Complete the active coding task now. Your previous "
                                "assistant turn ended with an empty or orphaned "
                                "tool-control response after tool results. Answer "
                                "with concrete final results, including files "
                                "changed, checks run, and any remaining caveats. "
                                "If more work is truly needed, emit exactly one "
                                "declared tool call. Do not return raw tool markup "
                                "or an empty response."
                            ),
                        )
                    )
                    repair_observability: dict[str, Any] = {}
                    repair_prompt_ids = _encode_messages(
                        state.runtime.tokenizer,
                        repair_messages,
                        enable_thinking=thinking_enabled,
                        reasoning_effort=reasoning_effort,
                        strip_assistant_reasoning_history=state.args.strip_assistant_reasoning_history,
                        tools=tool_specs,
                        tool_prompt_mode=tool_prompt_mode,
                        template_observability=repair_observability,
                    )
                    retry_observability = dict(request_observability)
                    retry_observability.update(
                        {
                            "tool_fed_empty_retry_attempted": True,
                            "tool_fed_empty_retry_reason": retry_reason,
                            "tool_fed_empty_retry_first_completion_tokens": int(
                                generated.get("completion_tokens") or 0
                            ),
                            "tool_fed_empty_retry_first_decode_tok_s": first_stats.get(
                                "decode_tok_s"
                            ),
                            "tool_fed_empty_retry_prompt_tokens": len(
                                repair_prompt_ids
                            ),
                        }
                    )
                    queue.put(("reset_orphan_stream_guards", None))
                    retry_generated = _run_generation_dispatched(
                        state,
                        repair_prompt_ids,
                        batch_key="chat.stream.tool_fed_empty_retry",
                        response_id=response_id,
                        max_tokens=request_max_tokens,
                        temperature=sampler_temperature,
                        top_p=sampler_top_p,
                        top_k=sampler_top_k,
                        seed=None,
                        draft_sampler=request_draft_sampler,
                        generation_mode=request_generation_mode,
                        depth=request_depth,
                        resolved_mtp_depth=effective_request_depth,
                        token_callback=on_tokens,
                        session_id=session_id,
                        cache_miss_reason=cache_miss_reason,
                        session_restore_mode=session_restore_mode,
                        session_bank=session_bank_for_generation,
                        session_template_hash=state.template_hash,
                        session_draft_head_identity=state.draft_head_identity,
                        session_policy_fingerprint=session_restore_policy_fingerprint,
                        background_request=background,
                        commit_final_state_to_bank=False,
                        commit_prompt_prefix_to_bank=commit_prompt_prefix,
                        session_keep_live_ref=session_keep_live_ref,
                        request_observability=retry_observability,
                        prefill_callback=on_prefill,
                        cancel_event=cancel_event,
                    )
                    retry_text = _strip_mtplx_internal_continuation_markers(
                        _strip_generated_chat_template_sentinels(
                            str(retry_generated.get("text") or "")
                        )
                    )
                    retry_stats = retry_generated.setdefault("stats", {})
                    retry_stats.update(retry_observability)
                    retry_succeeded = bool(retry_text.strip())
                    retry_stats["tool_fed_empty_retry_succeeded"] = retry_succeeded
                    retry_stats["tool_fed_empty_retry_completion_tokens"] = int(
                        retry_generated.get("completion_tokens") or 0
                    )
                    retry_stats["tool_fed_empty_retry_finish_reason"] = str(
                        retry_generated.get("finish_reason") or "stop"
                    )
                    if state.last_metrics:
                        state.last_metrics[-1].update(
                            {
                                "tool_fed_empty_retry_attempted": True,
                                "tool_fed_empty_retry_succeeded": retry_succeeded,
                                "tool_fed_empty_retry_reason": retry_reason,
                                "tool_fed_empty_retry_first_completion_tokens": int(
                                    generated.get("completion_tokens") or 0
                                ),
                                "tool_fed_empty_retry_first_decode_tok_s": first_stats.get(
                                    "decode_tok_s"
                                ),
                                "tool_fed_empty_retry_prompt_tokens": len(
                                    repair_prompt_ids
                                ),
                            }
                        )
                    return retry_generated

                def maybe_repair_tool_fed_reasoning_only_completion(
                    generated: dict[str, Any],
                ) -> dict[str, Any]:
                    if (
                        not tools_active
                        or not tool_result_history_present
                        or not thinking_enabled
                        or request.seed is not None
                        or _reasoning_parser_for_state(state) not in {"qwen3", "step3p5"}
                        # Forced final-answer turns intentionally rehearse
                        # before the visible marker; the buffered marker
                        # stream owns visibility, so a reasoning-shaped first
                        # pass is expected and must not trigger a raw retry.
                        or read_only_force_answer_contract_active
                    ):
                        return generated
                    first_text = _strip_mtplx_internal_continuation_markers(
                        _strip_generated_chat_template_sentinels(
                            str(generated.get("text") or "")
                        )
                    )
                    if not first_text.strip():
                        return generated
                    raw_reasoning_text, raw_content_text = _tool_extraction_text_parts(
                        state,
                        first_text,
                        thinking_enabled=thinking_enabled,
                    )
                    extraction = omlx_extract_tool_calls_with_thinking(
                        raw_reasoning_text,
                        raw_content_text,
                        state.runtime.tokenizer,
                        tool_specs,
                    )
                    if extraction.tool_calls:
                        return generated
                    reasoning_text, answer_text = _split_backend_reasoning_for_state(
                        state,
                        first_text,
                        thinking_enabled=thinking_enabled,
                    )
                    if not _reasoning_completion_repair_needed(
                        thinking_enabled=thinking_enabled,
                        reasoning_text=reasoning_text,
                        answer_text=answer_text,
                        assistant_tool_calls=None,
                    ):
                        return generated
                    first_stats = dict(generated.get("stats") or {})
                    repair_prompt_ids = _reasoning_completion_repair_prompt_ids(
                        state.runtime.tokenizer,
                        prompt_ids,
                        [int(token) for token in generated.get("tokens") or []],
                    )
                    retry_observability = dict(request_observability)
                    retry_observability.update(
                        {
                            "reasoning_completion_repair_attempted": True,
                            "reasoning_completion_repair_reason": (
                                "tool_fed_reasoning_only_completion"
                            ),
                            "reasoning_completion_repair_first_completion_tokens": int(
                                generated.get("completion_tokens") or 0
                            ),
                            "reasoning_completion_repair_first_decode_tok_s": first_stats.get(
                                "decode_tok_s"
                            ),
                            "reasoning_completion_repair_prompt_tokens": len(
                                repair_prompt_ids
                            ),
                        }
                    )
                    queue.put(("close_unclosed_reasoning_for_repair", None))
                    retry_generated = _run_generation_dispatched(
                        state,
                        repair_prompt_ids,
                        batch_key="chat.stream.reasoning_completion_repair",
                        response_id=response_id,
                        max_tokens=request_max_tokens,
                        temperature=sampler_temperature,
                        top_p=sampler_top_p,
                        top_k=sampler_top_k,
                        seed=None,
                        draft_sampler=request_draft_sampler,
                        generation_mode=request_generation_mode,
                        depth=request_depth,
                        resolved_mtp_depth=effective_request_depth,
                        token_callback=on_tokens,
                        session_id=session_id,
                        cache_miss_reason=cache_miss_reason,
                        session_restore_mode=session_restore_mode,
                        session_bank=session_bank_for_generation,
                        session_template_hash=state.template_hash,
                        session_draft_head_identity=state.draft_head_identity,
                        session_policy_fingerprint=session_restore_policy_fingerprint,
                        background_request=background,
                        commit_final_state_to_bank=False,
                        commit_prompt_prefix_to_bank=False,
                        session_keep_live_ref=session_keep_live_ref,
                        request_observability=retry_observability,
                        prefill_callback=on_prefill,
                        cancel_event=cancel_event,
                    )
                    retry_text = _strip_mtplx_internal_continuation_markers(
                        _strip_generated_chat_template_sentinels(
                            str(retry_generated.get("text") or "")
                        )
                    )
                    retry_reasoning, retry_content = _tool_extraction_text_parts(
                        state,
                        retry_text,
                        thinking_enabled=thinking_enabled,
                    )
                    retry_extraction = omlx_extract_tool_calls_with_thinking(
                        retry_reasoning,
                        retry_content,
                        state.runtime.tokenizer,
                        tool_specs,
                    )
                    retry_visible_text = _clean_generated_assistant_text(
                        retry_content or retry_text
                    ).strip()
                    retry_stats = retry_generated.setdefault("stats", {})
                    retry_stats.update(retry_observability)
                    retry_succeeded = bool(
                        retry_extraction.tool_calls
                    ) or bool(retry_visible_text)
                    retry_stats["reasoning_completion_repair_succeeded"] = (
                        retry_succeeded
                    )
                    retry_stats["reasoning_completion_repair_completion_tokens"] = int(
                        retry_generated.get("completion_tokens") or 0
                    )
                    retry_stats["reasoning_completion_repair_finish_reason"] = str(
                        retry_generated.get("finish_reason") or "stop"
                    )
                    retry_stats["reasoning_completion_repair_decode_tok_s"] = (
                        retry_stats.get("decode_tok_s")
                    )
                    if state.last_metrics:
                        state.last_metrics[-1].update(
                            {
                                "reasoning_completion_repair_attempted": True,
                                "reasoning_completion_repair_succeeded": (
                                    retry_succeeded
                                ),
                                "reasoning_completion_repair_reason": (
                                    "tool_fed_reasoning_only_completion"
                                ),
                                "reasoning_completion_repair_first_completion_tokens": int(
                                    generated.get("completion_tokens") or 0
                                ),
                                "reasoning_completion_repair_first_decode_tok_s": (
                                    first_stats.get("decode_tok_s")
                                ),
                                "reasoning_completion_repair_prompt_tokens": len(
                                    repair_prompt_ids
                                ),
                            }
                        )
                    return retry_generated

                def maybe_retry_stalled_agent_tool_promise(
                    generated: dict[str, Any],
                ) -> dict[str, Any]:
                    if (
                        not tools_active
                        or not tool_result_history_present
                        or request.seed is not None
                    ):
                        return generated
                    raw_text = _strip_mtplx_internal_continuation_markers(
                        _strip_generated_chat_template_sentinels(
                            str(generated.get("text") or "")
                        )
                    )
                    if not raw_text.strip():
                        return generated
                    raw_reasoning_text, raw_content_text = _tool_extraction_text_parts(
                        state,
                        raw_text,
                        thinking_enabled=thinking_enabled,
                    )
                    extraction = omlx_extract_tool_calls_with_thinking(
                        raw_reasoning_text,
                        raw_content_text,
                        state.runtime.tokenizer,
                        tool_specs,
                    )
                    if extraction.tool_calls:
                        return generated
                    visible_candidate = "\n\n".join(
                        part.strip()
                        for part in (raw_reasoning_text, raw_content_text)
                        if part and part.strip()
                    ) or raw_text
                    if not _looks_like_stalled_agent_tool_promise(visible_candidate):
                        return generated

                    repair_messages = list(messages_for_generation)
                    repair_messages.append(
                        ChatMessage(
                            role="user",
                            content=(
                                "Continue the active coding task now. Your previous "
                                "draft ended by promising to inspect, run, edit, or "
                                "check more work, but it did not include a tool call. "
                                "If more work is needed, emit exactly one declared "
                                "tool call now. If no more tool is needed, answer "
                                "with concrete final results. Do not say \"let me\" "
                                "and do not quote MTPLX internal notes."
                            ),
                        )
                    )
                    repair_observability: dict[str, Any] = {}
                    repair_prompt_ids = _encode_messages(
                        state.runtime.tokenizer,
                        repair_messages,
                        enable_thinking=thinking_enabled,
                        reasoning_effort=reasoning_effort,
                        strip_assistant_reasoning_history=state.args.strip_assistant_reasoning_history,
                        tools=tool_specs,
                        tool_prompt_mode=tool_prompt_mode,
                        template_observability=repair_observability,
                    )
                    first_stats = dict(generated.get("stats") or {})
                    retry_observability = dict(request_observability)
                    retry_observability.update(
                        {
                            "stalled_agent_retry_attempted": True,
                            "stalled_agent_retry_reason": "tool_promise_without_tool_call",
                            "stalled_agent_retry_first_completion_tokens": int(
                                generated.get("completion_tokens") or 0
                            ),
                            "stalled_agent_retry_first_decode_tok_s": first_stats.get(
                                "decode_tok_s"
                            ),
                            "stalled_agent_retry_prompt_tokens": len(
                                repair_prompt_ids
                            ),
                        }
                    )
                    retry_generated = _run_generation_dispatched(
                        state,
                        repair_prompt_ids,
                        batch_key="chat.stream.stalled_agent_retry",
                        response_id=response_id,
                        max_tokens=request_max_tokens,
                        temperature=sampler_temperature,
                        top_p=sampler_top_p,
                        top_k=sampler_top_k,
                        seed=None,
                        draft_sampler=request_draft_sampler,
                        generation_mode=request_generation_mode,
                        depth=request_depth,
                        resolved_mtp_depth=effective_request_depth,
                        token_callback=on_tokens,
                        session_id=session_id,
                        cache_miss_reason=cache_miss_reason,
                        session_restore_mode=session_restore_mode,
                        session_bank=session_bank_for_generation,
                        session_template_hash=state.template_hash,
                        session_draft_head_identity=state.draft_head_identity,
                        session_policy_fingerprint=session_restore_policy_fingerprint,
                        background_request=background,
                        commit_final_state_to_bank=False,
                        commit_prompt_prefix_to_bank=commit_prompt_prefix,
                        session_keep_live_ref=session_keep_live_ref,
                        request_observability=retry_observability,
                        prefill_callback=on_prefill,
                        cancel_event=cancel_event,
                    )
                    retry_text = _strip_mtplx_internal_continuation_markers(
                        _strip_generated_chat_template_sentinels(
                            str(retry_generated.get("text") or "")
                        )
                    )
                    retry_reasoning, retry_content = _tool_extraction_text_parts(
                        state,
                        retry_text,
                        thinking_enabled=thinking_enabled,
                    )
                    retry_extraction = omlx_extract_tool_calls_with_thinking(
                        retry_reasoning,
                        retry_content,
                        state.runtime.tokenizer,
                        tool_specs,
                    )
                    retry_stats = retry_generated.setdefault("stats", {})
                    retry_stats.update(retry_observability)
                    retry_succeeded = bool(
                        retry_extraction.tool_calls
                    ) or not _looks_like_stalled_agent_tool_promise(retry_text)
                    retry_stats["stalled_agent_retry_succeeded"] = retry_succeeded
                    retry_stats["stalled_agent_retry_completion_tokens"] = int(
                        retry_generated.get("completion_tokens") or 0
                    )
                    retry_stats["stalled_agent_retry_finish_reason"] = str(
                        retry_generated.get("finish_reason") or "stop"
                    )
                    if state.last_metrics:
                        state.last_metrics[-1].update(
                            {
                                "stalled_agent_retry_attempted": True,
                                "stalled_agent_retry_succeeded": retry_succeeded,
                                "stalled_agent_retry_reason": (
                                    "tool_promise_without_tool_call"
                                ),
                                "stalled_agent_retry_first_completion_tokens": int(
                                    generated.get("completion_tokens") or 0
                                ),
                                "stalled_agent_retry_first_decode_tok_s": first_stats.get(
                                    "decode_tok_s"
                                ),
                                "stalled_agent_retry_prompt_tokens": len(
                                    repair_prompt_ids
                                ),
                            }
                        )
                    return retry_generated

                def maybe_retry_read_only_force_answer(
                    generated: dict[str, Any],
                ) -> dict[str, Any]:
                    if (
                        not read_only_force_answer_contract_active
                        or request.seed is not None
                    ):
                        return generated
                    raw_text = _strip_mtplx_internal_continuation_markers(
                        _strip_generated_chat_template_sentinels(
                            str(generated.get("text") or "")
                        )
                    )
                    if not raw_text.strip():
                        return generated
                    raw_reasoning_text, raw_content_text = _tool_extraction_text_parts(
                        state,
                        raw_text,
                        thinking_enabled=thinking_enabled,
                    )
                    visible_candidate = "\n\n".join(
                        part.strip()
                        for part in (raw_reasoning_text, raw_content_text)
                        if part and part.strip()
                    ) or raw_text
                    if not _looks_like_read_only_force_answer_failure(
                        visible_candidate
                    ):
                        return generated

                    repair_messages = list(messages_for_generation)
                    repair_messages.append(
                        ChatMessage(
                            role="user",
                            content=(
                                "Answer now. Tools are closed for this read-only "
                                "inspection. Do not read, check, inspect, run, or "
                                "request more files. Do not emit tool calls, tool "
                                "names, protocol tags, or filePath/startLine/"
                                "endingLine markup. Provide the requested final "
                                "answer from the evidence already gathered, "
                                "including the requested number of items and marker."
                            ),
                        )
                    )
                    repair_observability: dict[str, Any] = {}
                    repair_prompt_ids = _encode_messages(
                        state.runtime.tokenizer,
                        repair_messages,
                        enable_thinking=thinking_enabled,
                        reasoning_effort=reasoning_effort,
                        strip_assistant_reasoning_history=state.args.strip_assistant_reasoning_history,
                        tools=None,
                        tool_prompt_mode=tool_prompt_mode,
                        template_observability=repair_observability,
                    )
                    first_stats = dict(generated.get("stats") or {})
                    retry_observability = dict(request_observability)
                    retry_observability.update(
                        {
                            "read_only_force_answer_retry_attempted": True,
                            "read_only_force_answer_retry_reason": (
                                "toolish_draft_after_tools_closed"
                            ),
                            "read_only_force_answer_retry_first_completion_tokens": int(
                                generated.get("completion_tokens") or 0
                            ),
                            "read_only_force_answer_retry_first_decode_tok_s": (
                                first_stats.get("decode_tok_s")
                            ),
                            "read_only_force_answer_retry_prompt_tokens": len(
                                repair_prompt_ids
                            ),
                        }
                    )
                    retry_generated = _run_generation_dispatched(
                        state,
                        repair_prompt_ids,
                        batch_key="chat.stream.read_only_force_answer_retry",
                        response_id=response_id,
                        max_tokens=request_max_tokens,
                        temperature=sampler_temperature,
                        top_p=sampler_top_p,
                        top_k=sampler_top_k,
                        seed=None,
                        draft_sampler=request_draft_sampler,
                        generation_mode=request_generation_mode,
                        depth=request_depth,
                        resolved_mtp_depth=effective_request_depth,
                        token_callback=on_tokens,
                        session_id=session_id,
                        cache_miss_reason=cache_miss_reason,
                        session_restore_mode=session_restore_mode,
                        session_bank=session_bank_for_generation,
                        session_template_hash=state.template_hash,
                        session_draft_head_identity=state.draft_head_identity,
                        session_policy_fingerprint=session_restore_policy_fingerprint,
                        background_request=background,
                        commit_final_state_to_bank=False,
                        commit_prompt_prefix_to_bank=commit_prompt_prefix,
                        session_keep_live_ref=session_keep_live_ref,
                        request_observability=retry_observability,
                        prefill_callback=on_prefill,
                        cancel_event=cancel_event,
                    )
                    retry_text = _strip_mtplx_internal_continuation_markers(
                        _strip_generated_chat_template_sentinels(
                            str(retry_generated.get("text") or "")
                        )
                    )
                    retry_stats = retry_generated.setdefault("stats", {})
                    retry_stats.update(retry_observability)
                    retry_succeeded = not _looks_like_read_only_force_answer_failure(
                        retry_text
                    )
                    retry_stats["read_only_force_answer_retry_succeeded"] = (
                        retry_succeeded
                    )
                    retry_stats["read_only_force_answer_retry_completion_tokens"] = int(
                        retry_generated.get("completion_tokens") or 0
                    )
                    retry_stats["read_only_force_answer_retry_finish_reason"] = str(
                        retry_generated.get("finish_reason") or "stop"
                    )
                    if state.last_metrics:
                        state.last_metrics[-1].update(
                            {
                                "read_only_force_answer_retry_attempted": True,
                                "read_only_force_answer_retry_succeeded": (
                                    retry_succeeded
                                ),
                                "read_only_force_answer_retry_reason": (
                                    "toolish_draft_after_tools_closed"
                                ),
                                "read_only_force_answer_retry_first_completion_tokens": int(
                                    generated.get("completion_tokens") or 0
                                ),
                                "read_only_force_answer_retry_first_decode_tok_s": (
                                    first_stats.get("decode_tok_s")
                                ),
                                "read_only_force_answer_retry_prompt_tokens": len(
                                    repair_prompt_ids
                                ),
                            }
                        )
                    return retry_generated

                def worker() -> None:
                    try:
                        _raise_if_stream_cancelled(cancel_event)
                        if session is None:
                            generated = _run_generation_dispatched(
                                state,
                                prompt_ids,
                                batch_key="chat.stream",
                                response_id=response_id,
                                max_tokens=request_max_tokens,
                                temperature=sampler_temperature,
                                top_p=sampler_top_p,
                                top_k=sampler_top_k,
                                seed=request.seed,
                                draft_sampler=request_draft_sampler,
                                generation_mode=request_generation_mode,
                                depth=request_depth,
                                resolved_mtp_depth=effective_request_depth,
                                token_callback=on_tokens,
                                session_id=session_id,
                                cache_miss_reason=cache_miss_reason,
                                session_restore_mode=session_restore_mode,
                                session_bank=session_bank_for_generation,
                                session_template_hash=state.template_hash,
                                session_draft_head_identity=state.draft_head_identity,
                                session_policy_fingerprint=session_restore_policy_fingerprint,
                                background_request=background,
                                commit_prompt_prefix_to_bank=commit_prompt_prefix,
                                session_keep_live_ref=session_keep_live_ref,
                                request_observability=request_observability,
                                prefill_callback=on_prefill,
                                cancel_event=cancel_event,
                            )
                            generated = maybe_retry_degenerate_read_only_inspection(
                                generated
                            )
                            generated = maybe_retry_degenerate_tool_fed_empty_completion(
                                generated
                            )
                            generated = maybe_repair_tool_fed_reasoning_only_completion(
                                generated
                            )
                            generated = maybe_retry_read_only_force_answer(generated)
                            generated = maybe_retry_stalled_agent_tool_promise(
                                generated
                            )
                        else:
                            with state.sessions.generation_slot(
                                session, source=session_source
                            ) as acquired_session:
                                adopt_forked_session(acquired_session)
                                generated = _run_generation_dispatched(
                                    state,
                                    prompt_ids,
                                    batch_key="chat.stream",
                                    response_id=response_id,
                                    max_tokens=request_max_tokens,
                                    temperature=sampler_temperature,
                                    top_p=sampler_top_p,
                                    top_k=sampler_top_k,
                                    seed=request.seed,
                                    draft_sampler=request_draft_sampler,
                                    generation_mode=request_generation_mode,
                                    depth=request_depth,
                                    resolved_mtp_depth=effective_request_depth,
                                    token_callback=on_tokens,
                                    session_id=session_id,
                                    cache_miss_reason=cache_miss_reason,
                                    session_restore_mode=session_restore_mode,
                                    session_bank=session_bank_for_generation,
                                    session_template_hash=state.template_hash,
                                    session_draft_head_identity=state.draft_head_identity,
                                    session_policy_fingerprint=session_restore_policy_fingerprint,
                                    commit_final_state_to_bank=False,
                                    commit_prompt_prefix_to_bank=commit_prompt_prefix,
                                    session_keep_live_ref=session_keep_live_ref,
                                    request_observability=request_observability,
                                    prefill_callback=on_prefill,
                                    cancel_event=cancel_event,
                                )
                                generated = maybe_retry_degenerate_read_only_inspection(
                                    generated
                                )
                                generated = maybe_retry_degenerate_tool_fed_empty_completion(
                                    generated
                                )
                                generated = maybe_repair_tool_fed_reasoning_only_completion(
                                    generated
                                )
                                generated = maybe_retry_read_only_force_answer(
                                    generated
                                )
                                generated = maybe_retry_stalled_agent_tool_promise(
                                    generated
                                )
                                queue.put(("done", generated))
                                commit_event.wait()
                                if commit_state["commit"]:
                                    assistant_history_content = str(
                                        commit_state.get("assistant_history_content")
                                        or ""
                                    ) or (
                                        _normalize_reasoning_tags_for_state(
                                            state,
                                            str(generated["text"]),
                                            thinking_enabled=thinking_enabled,
                                        )
                                        if state.args.normalize_thinking_tags
                                        else str(generated["text"])
                                    )
                                    assistant_tool_calls = commit_state.get(
                                        "assistant_tool_calls"
                                    )
                                    if bool(commit_state.get("retokenize_inline")):
                                        postcommit = _submit_foreground_model_work(
                                            state,
                                            lambda: _store_retokenized_history_snapshot(
                                                state,
                                                session_id=session_id,
                                                messages=raw_messages_for_postcommit,
                                                assistant_content=(
                                                    assistant_history_content
                                                ),
                                                assistant_tool_calls=(
                                                    assistant_tool_calls
                                                ),
                                                thinking_enabled=thinking_enabled,
                                                policy_fingerprint=postcommit_policy_fingerprint,
                                                tool_specs=postcommit_tool_specs,
                                                keep_live_ref=session_keep_live_ref,
                                                tool_prompt_mode=postcommit_tool_prompt_mode,
                                                strip_tool_call_preamble_text=opencode_client,
                                            ),
                                            batch_key=(
                                                f"postcommit.stream.inline:"
                                                f"{session_id or 'stateless'}"
                                            ),
                                        ).result()
                                    else:
                                        postcommit = _submit_foreground_model_work(
                                            state,
                                            lambda: _store_generation_final_history_snapshot(
                                                state,
                                                session_id=session_id,
                                                prompt_ids=prompt_ids,
                                                generated=generated,
                                                messages=raw_messages_for_postcommit,
                                                assistant_content=(
                                                    assistant_history_content
                                                ),
                                                assistant_tool_calls=(
                                                    assistant_tool_calls
                                                ),
                                                thinking_enabled=thinking_enabled,
                                                policy_fingerprint=postcommit_policy_fingerprint,
                                                tool_specs=postcommit_tool_specs,
                                                keep_live_ref=session_keep_live_ref,
                                                tool_prompt_mode=postcommit_tool_prompt_mode,
                                                strip_tool_call_preamble_text=opencode_client,
                                            ),
                                            batch_key=(
                                                f"postcommit.stream.final:"
                                                f"{session_id or 'stateless'}"
                                            ),
                                        ).result()
                                        if not postcommit.get("stored"):
                                            generated["stats"][
                                                "session_postcommit_snapshot"
                                            ] = postcommit
                                            queue.put(
                                                (
                                                    "released",
                                                    {
                                                        "generated": generated,
                                                        "postcommit": postcommit,
                                                    },
                                                )
                                            )
                                            return
                                    generated["stats"][
                                        "session_postcommit_snapshot"
                                    ] = postcommit
                                    session.commit(
                                        prompt_ids=prompt_ids,
                                        generated_ids=generated["tokens"],
                                        finish_reason=generated.get(
                                            "finish_reason", "stop"
                                        ),
                                        nbytes=int(postcommit.get("nbytes") or 0),
                                    )
                                    queue.put(("committed", generated))
                                else:
                                    queue.put(("released", None))
                                return
                    except _StreamCancelled as exc:
                        queue.put(_stream_cancelled_queue_item(exc))
                    except EngineSessionBusy as exc:
                        queue.put(
                            ("error", HTTPException(status_code=409, detail=str(exc)))
                        )
                    except BaseException as exc:
                        if (
                            session is not None
                            and commit_event.is_set()
                            and state.args.session_postcommit_mode == "async"
                        ):
                            _safe_stdout_print(
                                f"[mtplx] async session postcommit failed: {exc!r}"
                            )
                        queue.put(("error", exc))
                    else:
                        queue.put(("done", generated))

                generation_future: Future = Future()

                def run_worker_thread() -> None:
                    try:
                        worker()
                    except BaseException as exc:
                        queue.put(("error", exc))
                        if not generation_future.done():
                            generation_future.set_exception(exc)
                    else:
                        if not generation_future.done():
                            generation_future.set_result(None)

                Thread(
                    target=run_worker_thread,
                    name=f"mtplx-stream-worker-{response_id[-8:]}",
                    daemon=True,
                ).start()

                def delta_payload_chunk(delta: dict[str, Any]) -> str:
                    payload = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": None,
                            }
                        ],
                    }
                    return f"data: {json.dumps(payload)}\n\n"

                def delta_chunk(field: str, text: str) -> str:
                    return delta_payload_chunk({field: text})

                def progress_chunk(progress: dict[str, Any]) -> str:
                    payload = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": None,
                            }
                        ],
                        "mtplx_progress": _json_safe(progress),
                    }
                    return f"data: {json.dumps(payload)}\n\n"

                def error_chunk(exc: BaseException) -> str:
                    if isinstance(exc, HTTPException):
                        message = str(exc.detail)
                        status_code = exc.status_code
                    else:
                        message = str(exc)
                        status_code = 500
                    payload = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": "error"}
                        ],
                        **_openai_error_content(
                            message,
                            status_code=status_code,
                            code=type(exc).__name__,
                        ),
                    }
                    return f"data: {json.dumps(payload)}\n\n"

                def maybe_log_stream_silence(now_s: float) -> None:
                    nonlocal next_silence_warn_s
                    last_activity_s = (
                        last_token_s if last_token_s is not None else stream_started_s
                    )
                    seconds_since_last_token = max(0.0, now_s - last_activity_s)
                    if (
                        seconds_since_last_token < STREAM_SILENCE_WARN_S
                        or now_s < next_silence_warn_s
                        or _server_console_enabled(state)
                    ):
                        return
                    scheduler_stats: dict[str, Any] = {}
                    scheduler = getattr(state, "model_scheduler", None)
                    if scheduler is not None and hasattr(scheduler, "stats"):
                        try:
                            scheduler_stats = dict(scheduler.stats())
                        except BaseException:
                            scheduler_stats = {}
                    pending_postcommit_detail = None
                    if session is not None and hasattr(
                        session, "pending_postcommit_admin"
                    ):
                        try:
                            pending_postcommit_detail = (
                                session.pending_postcommit_admin()
                            )
                        except BaseException:
                            pending_postcommit_detail = None
                    _safe_stdout_print(
                        json.dumps(
                            {
                                "event": "mtplx_stream_silence",
                                "response_id": response_id,
                                "session_id": session_id,
                                "elapsed_s": round(now_s - stream_started_s, 6),
                                "completion_tokens": int(streamed_progress_tokens),
                                "seconds_since_last_token": round(
                                    seconds_since_last_token,
                                    6,
                                ),
                                "scheduler_active_kind": scheduler_stats.get(
                                    "active_kind"
                                ),
                                "scheduler_foreground_pending": scheduler_stats.get(
                                    "foreground_pending"
                                ),
                                "scheduler_idle_pending": scheduler_stats.get(
                                    "idle_pending"
                                ),
                                "postcommit_active": bool(
                                    pending_postcommit_detail
                                    and pending_postcommit_detail.get("active")
                                ),
                                "pending_postcommit": pending_postcommit_detail,
                            },
                            ensure_ascii=False,
                        )
                    )
                    next_silence_warn_s = now_s + STREAM_SILENCE_WARN_INTERVAL_S

                generated: dict[str, Any] | None = None
                history_reasoning_chunks: list[str] = []
                history_content_chunks: list[str] = []
                streamed_token_ids: list[int] = []
                streamed_progress_tokens = 0
                streamed_decode_started_s: float | None = None
                streamed_assistant_tool_calls: list[dict[str, Any]] | None = None
                streamed_tool_deltas_emitted = False
                early_tool_cancel_used = False
                pending_tool_cancel_started_s: float | None = None
                hidden_tool_guard_started_s: float | None = None
                hidden_tool_guard_started_tokens: int | None = None
                buffer_read_only_force_answer_stream = bool(
                    read_only_force_answer_contract_active
                )
                read_only_force_answer_stream_buffer = ""
                read_only_force_answer_stream_started = False
                read_only_force_answer_stream_marker_stripped_chars = 0

                def remember_stream_delta(delta: dict[str, Any]) -> None:
                    reasoning = delta.get("reasoning_content")
                    if isinstance(reasoning, str) and reasoning:
                        history_reasoning_chunks.append(reasoning)
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        history_content_chunks.append(content)

                def reset_orphan_stream_guards() -> None:
                    nonlocal orphan_reasoning_stream_guard
                    nonlocal orphan_content_stream_guard
                    orphan_reasoning_stream_guard = (
                        _InitialOrphanToolControlStreamGuard()
                    )
                    orphan_content_stream_guard = (
                        _InitialOrphanToolControlStreamGuard()
                    )

                def apply_orphan_stream_guard(field: str, text: str) -> str:
                    nonlocal stream_orphan_tool_markup_suppressed
                    if not orphan_stream_guard_enabled:
                        return text
                    if field == "reasoning_content":
                        guard = orphan_reasoning_stream_guard
                    elif field == "content":
                        guard = orphan_content_stream_guard
                    else:
                        return text
                    guarded = guard.feed(text)
                    if guard.suppressed:
                        stream_orphan_tool_markup_suppressed = True
                    return guarded

                def stream_content_delta_chunks(
                    field: str,
                    text: str,
                    *,
                    use_orphan_guard: bool = True,
                    use_tool_translator: bool = True,
                    monitor_stop: bool = True,
                ) -> list[str]:
                    nonlocal streamed_assistant_tool_calls
                    nonlocal streamed_tool_deltas_emitted
                    nonlocal early_tool_cancel_used
                    nonlocal pending_tool_cancel_started_s
                    if not text:
                        return []
                    if use_orphan_guard:
                        text = apply_orphan_stream_guard(field, text)
                        if not text:
                            return []
                    if field == "reasoning_content" and suppress_visible_reasoning:
                        remember_stream_delta({field: text})
                        return []
                    if (
                        field == "content"
                        and monitor_stop
                        and stop_monitor is not None
                    ):
                        if stop_monitor.stopped:
                            return []
                        text = stop_monitor.feed(text)
                        if stop_monitor.stopped:
                            fire_stop_sequence_cancel()
                        if not text:
                            return []
                    if (
                        field == "content"
                        and content_tool_translator is not None
                        and use_tool_translator
                    ):
                        if early_tool_cancel_used and streamed_assistant_tool_calls:
                            return []
                        chunks: list[str] = []
                        for delta in content_tool_translator.feed(field, text):
                            if not delta:
                                continue
                            if (
                                single_tool_call_stream
                                and streamed_assistant_tool_calls
                                and "tool_calls" in delta
                            ):
                                indexes = [
                                    int(item.get("index") or 0)
                                    for item in delta.get("tool_calls", [])
                                    if isinstance(item, dict)
                                ]
                                if indexes and all(
                                    index >= len(streamed_assistant_tool_calls)
                                    for index in indexes
                                ):
                                    continue
                            if (
                                early_tool_cancel_used
                                and streamed_assistant_tool_calls
                                and "tool_calls" not in delta
                            ):
                                continue
                            if "content" in delta or "reasoning_content" in delta:
                                remember_stream_delta(delta)
                            if "tool_calls" in delta:
                                streamed_tool_deltas_emitted = True
                                parsed_calls = (
                                    content_tool_translator.tool_calls
                                    or streamed_assistant_tool_calls
                                )
                                if single_tool_call_stream and parsed_calls:
                                    parsed_calls = parsed_calls[:1]
                                streamed_assistant_tool_calls = parsed_calls
                            chunks.append(delta_payload_chunk(delta))
                        if (
                            single_tool_call_stream
                            and streamed_assistant_tool_calls
                            and not early_tool_cancel_used
                        ):
                            early_tool_cancel_used = True
                            pending_tool_cancel_started_s = None
                            _cancel_stream_generation(cancel_event, generation_future)
                            return chunks
                        if content_tool_translator.invalid_trailing_after_tool_call:
                            streamed_assistant_tool_calls = (
                                content_tool_translator.tool_calls
                                or streamed_assistant_tool_calls
                            )
                            if single_tool_call_stream and streamed_assistant_tool_calls:
                                streamed_assistant_tool_calls = (
                                    streamed_assistant_tool_calls[:1]
                                )
                            early_tool_cancel_used = True
                            pending_tool_cancel_started_s = None
                            _cancel_stream_generation(cancel_event, generation_future)
                        elif content_tool_translator.ready_to_finish_tool_turn:
                            streamed_assistant_tool_calls = (
                                content_tool_translator.tool_calls
                                or streamed_assistant_tool_calls
                            )
                            if single_tool_call_stream and streamed_assistant_tool_calls:
                                streamed_assistant_tool_calls = (
                                    streamed_assistant_tool_calls[:1]
                                )
                                early_tool_cancel_used = True
                                pending_tool_cancel_started_s = None
                                _cancel_stream_generation(
                                    cancel_event, generation_future
                                )
                            elif pending_tool_cancel_started_s is None:
                                pending_tool_cancel_started_s = time.perf_counter()
                        else:
                            pending_tool_cancel_started_s = None
                        return chunks
                    delta = {field: text}
                    remember_stream_delta(delta)
                    return [delta_payload_chunk(delta)]

                def finish_orphan_stream_guards() -> list[str]:
                    nonlocal stream_orphan_tool_markup_suppressed
                    if not orphan_stream_guard_enabled:
                        return []
                    chunks: list[str] = []
                    for field, guard in (
                        ("reasoning_content", orphan_reasoning_stream_guard),
                        ("content", orphan_content_stream_guard),
                    ):
                        flushed = guard.finish()
                        if guard.suppressed:
                            stream_orphan_tool_markup_suppressed = True
                        if not flushed:
                            continue
                        chunks.extend(
                            stream_content_delta_chunks(
                                field,
                                flushed,
                                use_orphan_guard=False,
                            )
                        )
                    return chunks

                def read_only_force_answer_text_slices(
                    text: str,
                    *,
                    max_chars: int = 512,
                ) -> list[str]:
                    if not text:
                        return []
                    return [
                        text[index : index + max_chars]
                        for index in range(0, len(text), max_chars)
                    ]

                def stream_read_only_force_answer_text(
                    text: str,
                    *,
                    force: bool = False,
                ) -> list[str]:
                    nonlocal read_only_force_answer_stream_buffer
                    nonlocal read_only_force_answer_stream_started
                    nonlocal read_only_force_answer_stream_marker_stripped_chars
                    chunks: list[str] = []
                    emit_text = ""
                    if read_only_force_answer_stream_started:
                        emit_text = text
                    else:
                        if text:
                            read_only_force_answer_stream_buffer += text
                        marked_visible, marker_stripped_chars = (
                            _read_only_force_answer_after_stream_marker(
                                read_only_force_answer_stream_buffer
                            )
                        )
                        if marker_stripped_chars:
                            read_only_force_answer_stream_started = True
                            read_only_force_answer_stream_marker_stripped_chars = (
                                marker_stripped_chars
                            )
                            emit_text = marked_visible
                            read_only_force_answer_stream_buffer = ""
                        elif force and read_only_force_answer_stream_buffer:
                            visible_text, stripped_chars = (
                                _read_only_force_answer_visible_text(
                                    read_only_force_answer_stream_buffer
                                )
                            )
                            read_only_force_answer_stream_started = bool(visible_text)
                            read_only_force_answer_stream_marker_stripped_chars = int(
                                stripped_chars
                            )
                            emit_text = visible_text
                            read_only_force_answer_stream_buffer = ""
                    if not emit_text:
                        return chunks
                    for part in read_only_force_answer_text_slices(emit_text):
                        chunks.extend(
                            stream_content_delta_chunks(
                                "content",
                                part,
                                use_orphan_guard=False,
                                use_tool_translator=False,
                            )
                        )
                    return chunks

                def emit_read_only_force_answer_visible_text(text: str) -> list[str]:
                    nonlocal read_only_force_answer_stream_buffer
                    nonlocal read_only_force_answer_stream_started
                    if not text:
                        return []
                    read_only_force_answer_stream_buffer = ""
                    read_only_force_answer_stream_started = True
                    chunks: list[str] = []
                    for part in read_only_force_answer_text_slices(text):
                        chunks.extend(
                            stream_content_delta_chunks(
                                "content",
                                part,
                                use_orphan_guard=False,
                                use_tool_translator=False,
                            )
                        )
                    return chunks

                def finish_translated_stream_chunks() -> list[str]:
                    nonlocal streamed_assistant_tool_calls
                    nonlocal streamed_tool_deltas_emitted
                    chunks: list[str] = []
                    if early_tool_cancel_used and streamed_assistant_tool_calls:
                        return chunks
                    if content_tool_translator is not None:
                        for delta in content_tool_translator.finish():
                            if not delta:
                                continue
                            if "content" in delta or "reasoning_content" in delta:
                                remember_stream_delta(delta)
                            if "tool_calls" in delta:
                                streamed_tool_deltas_emitted = True
                                streamed_assistant_tool_calls = (
                                    content_tool_translator.tool_calls
                                    or streamed_assistant_tool_calls
                                )
                            chunks.append(delta_payload_chunk(delta))
                    return chunks

                def drain_stream_tokens(
                    tokens: list[int],
                    *,
                    force: bool = False,
                ) -> list[tuple[str, str]]:
                    pending_stream_tokens.extend(tokens)
                    chunks: list[tuple[str, str]] = []
                    while pending_stream_tokens and (
                        force or len(pending_stream_tokens) >= stream_interval
                    ):
                        batch_size = (
                            len(pending_stream_tokens) if force else stream_interval
                        )
                        batch = pending_stream_tokens[:batch_size]
                        del pending_stream_tokens[:batch_size]
                        delta = decoder.feed(batch)
                        chunks.extend(
                            (field, text)
                            for field, text in splitter.feed(delta)
                            if text
                        )
                    return chunks

                def streamed_history_content() -> str:
                    # Always capture the natural-language portion of the
                    # response. Previously this returned "" whenever
                    # tool parsing found calls, which dropped any preamble
                    # text (e.g. "Let me check..." before <tool_call>) from
                    # the stored assistant_content. The next turn's lookup
                    # encodes the same assistant message WITH the preamble
                    # (clients echo back content + tool_calls), so the
                    # prefix diverged and every tool-using turn paid a cold
                    # prefill. tool_call markup itself is suppressed by the
                    # bridge filter and not in history_content_chunks, so
                    # this is safe to return for the tool-call case too.
                    #
                    # Deliberately DO NOT fold reasoning_content back into
                    # assistant content. OpenAI-compatible clients such as
                    # OpenCode render `reasoning_content` separately and do
                    # not echo it as normal assistant `content` on the next
                    # request. Storing it here creates a SessionBank prefix
                    # the next turn can never match, which forces full cold
                    # prefill in multi-turn tool sessions.
                    content = (
                        "".join(history_content_chunks)
                        .replace(THINK_OPEN, "")
                        .replace(THINK_CLOSE, "")
                        .strip()
                    )
                    return content

                stream_cancelled_by_client = False
                cancelled_metric_recorded = False
                for field, text in splitter.start():
                    if text:
                        for chunk in stream_content_delta_chunks(field, text):
                            yield mark_sse_sent(chunk)

                try:
                    while True:
                        try:
                            kind, item = await asyncio.to_thread(queue.get, True, 0.25)
                        except Empty:
                            if stop_monitor is not None and stop_monitor.stopped:
                                # Stop-sequence cancel in flight: keep
                                # draining until the worker acknowledges with
                                # "cancelled"/"done" so the client still gets
                                # the terminal finish_reason="stop" chunk.
                                if await raw_request.is_disconnected():
                                    stream_cancelled_by_client = True
                                    return
                                continue
                            if (
                                cancel_event.is_set()
                                or await raw_request.is_disconnected()
                            ):
                                stream_cancelled_by_client = True
                                _cancel_stream_generation(
                                    cancel_event, generation_future
                                )
                                if session is not None and hasattr(
                                    session, "abort_pending_postcommit"
                                ):
                                    session.abort_pending_postcommit(
                                        "stream_client_disconnected"
                                    )
                                return
                            now_s = time.perf_counter()
                            if (
                                pending_tool_cancel_started_s is not None
                                and not generation_future.done()
                                and now_s - pending_tool_cancel_started_s
                                >= STREAM_TOOL_CALL_FINISH_GRACE_S
                            ):
                                early_tool_cancel_used = True
                                pending_tool_cancel_started_s = None
                                _cancel_stream_generation(
                                    cancel_event, generation_future
                                )
                                continue
                            if (
                                not generation_future.done()
                                and now_s - last_sse_sent_s
                                >= STREAM_HEARTBEAT_INTERVAL_S
                            ):
                                maybe_log_stream_silence(now_s)
                                yield mark_sse_sent(
                                    progress_chunk(
                                        _stream_heartbeat_payload(
                                            completion_tokens=streamed_progress_tokens,
                                            stream_started_s=stream_started_s,
                                            last_token_s=last_token_s,
                                            now_s=now_s,
                                        )
                                    )
                                )
                            continue
                        if kind == "tokens":
                            if isinstance(item, dict):
                                stream_tokens = list(item.get("tokens") or [])
                                token_timestamp_s = float(
                                    item.get("timestamp_s") or time.perf_counter()
                                )
                            else:
                                stream_tokens = list(item or [])
                                token_timestamp_s = time.perf_counter()
                            if stream_tokens:
                                streamed_token_ids.extend(int(t) for t in stream_tokens)
                                last_token_s = token_timestamp_s
                                next_silence_warn_s = (
                                    token_timestamp_s + STREAM_SILENCE_WARN_S
                                )
                                if streamed_decode_started_s is None:
                                    streamed_decode_started_s = token_timestamp_s
                                streamed_progress_tokens += len(stream_tokens)
                                progress_payload = _stream_progress_payload(
                                    completion_tokens=streamed_progress_tokens,
                                    decode_started_s=streamed_decode_started_s,
                                    now_s=token_timestamp_s,
                                )
                                progress_payload["request_id"] = response_id
                                progress_payload["session_id"] = session_id
                                published_progress = _dashboard_publish_progress(
                                    state,
                                    request_id=response_id,
                                    payload=progress_payload,
                                )
                                if (
                                    published_progress is not None
                                    and not buffer_read_only_force_answer_stream
                                ):
                                    yield mark_sse_sent(
                                        progress_chunk(published_progress)
                                    )
                            if not buffer_read_only_force_answer_stream:
                                for field, text in drain_stream_tokens(stream_tokens):
                                    for chunk in stream_content_delta_chunks(
                                        field, text
                                    ):
                                        yield mark_sse_sent(chunk)
                            else:
                                for _field, text in drain_stream_tokens(stream_tokens):
                                    for chunk in stream_read_only_force_answer_text(
                                        text
                                    ):
                                        yield mark_sse_sent(chunk)
                            if (
                                content_tool_translator is not None
                                and content_tool_translator.buffering_tool_call
                                and not streamed_tool_deltas_emitted
                                and not content_tool_translator.tool_argument_in_progress
                            ):
                                now_s = time.perf_counter()
                                if hidden_tool_guard_started_s is None:
                                    hidden_tool_guard_started_s = now_s
                                    hidden_tool_guard_started_tokens = (
                                        streamed_progress_tokens
                                    )
                                hidden_tokens = streamed_progress_tokens - int(
                                    hidden_tool_guard_started_tokens
                                    or streamed_progress_tokens
                                )
                                hidden_elapsed_s = now_s - hidden_tool_guard_started_s
                                if (
                                    hidden_tokens >= STREAM_HIDDEN_TOOL_GUARD_TOKENS
                                    and hidden_elapsed_s >= STREAM_HIDDEN_TOOL_GUARD_S
                                ):
                                    for chunk in finish_translated_stream_chunks():
                                        yield mark_sse_sent(chunk)
                                    if streamed_assistant_tool_calls:
                                        early_tool_cancel_used = True
                                        _cancel_stream_generation(
                                            cancel_event, generation_future
                                        )
                                    else:
                                        reason = (
                                            "unterminated tool_call stream suppressed "
                                            f"{hidden_tokens} tokens for "
                                            f"{hidden_elapsed_s:.1f}s"
                                        )
                                        _record_tool_parse_event(
                                            state,
                                            event="unclosed_tool_call",
                                            reason=reason,
                                            response_id=response_id,
                                            stream=True,
                                        )
                                        _cancel_stream_generation(
                                            cancel_event, generation_future
                                        )
                                        yield mark_sse_sent(
                                            error_chunk(
                                                HTTPException(
                                                    status_code=422,
                                                    detail=(
                                                        "malformed tool_call: "
                                                        "unterminated stream"
                                                    ),
                                                )
                                            )
                                        )
                                        yield mark_sse_sent("data: [DONE]\n\n")
                                        return
                            else:
                                hidden_tool_guard_started_s = None
                                hidden_tool_guard_started_tokens = None
                        elif kind == "done":
                            generated = item
                            generated = attach_response_observability(generated)
                            if stop_monitor is not None and stop_monitor.stopped:
                                # Generation finished before the cancel landed
                                # (or the match arrived in the final batch).
                                # The client already received exactly the text
                                # before the stop string; close out with the
                                # OpenAI stop contract and skip the session
                                # commit machinery, matching the cancel path.
                                generated["text"] = streamed_history_content()
                                generated["finish_reason"] = "stop"
                                stats = generated.setdefault("stats", {})
                                stats["stop_sequence_hit"] = True
                                stats["stop_sequence_matched"] = (
                                    stop_monitor.matched_stop
                                )
                                _attach_dashboard_progress_stats(
                                    state,
                                    request_id=response_id,
                                    stats=stats,
                                )
                                _merge_final_bridge_stats_into_latest_metrics(
                                    state, stats
                                )
                                break
                            if buffer_read_only_force_answer_stream:
                                for _field, text in drain_stream_tokens([], force=True):
                                    for chunk in stream_read_only_force_answer_text(
                                        text
                                    ):
                                        yield mark_sse_sent(chunk)
                                tail = decoder.finish()
                                if tail:
                                    for _field, text in splitter.feed(tail):
                                        if text:
                                            for chunk in stream_read_only_force_answer_text(
                                                text
                                            ):
                                                yield mark_sse_sent(chunk)
                                visible_text, stripped_chars = (
                                    _read_only_force_answer_visible_text(
                                        str(generated.get("text") or "")
                                    )
                                )
                                visible_tokens = _encode_rendered_chat_text(
                                    state.runtime.tokenizer,
                                    visible_text,
                                )
                                stats = generated.setdefault("stats", {})
                                stats["read_only_force_answer_buffered_stream"] = True
                                stats[
                                    "read_only_force_answer_marker_stream_started"
                                ] = bool(read_only_force_answer_stream_started)
                                stats[
                                    "read_only_force_answer_stream_marker_stripped_chars"
                                ] = int(
                                    read_only_force_answer_stream_marker_stripped_chars
                                )
                                stats[
                                    "read_only_force_answer_visible_prefix_stripped_chars"
                                ] = int(stripped_chars)
                                stats["read_only_force_answer_visible_tokens"] = len(
                                    visible_tokens
                                )
                                if visible_text:
                                    generated["text"] = visible_text
                                    generated["tokens"] = visible_tokens
                                    if read_only_force_answer_stream_started:
                                        streamed_visible_text = (
                                            streamed_history_content()
                                        )
                                        missing_visible_text = ""
                                        if visible_text.startswith(
                                            streamed_visible_text
                                        ):
                                            missing_visible_text = visible_text[
                                                len(streamed_visible_text) :
                                            ]
                                        if missing_visible_text:
                                            for part in read_only_force_answer_text_slices(
                                                missing_visible_text
                                            ):
                                                for chunk in stream_content_delta_chunks(
                                                    "content",
                                                    part,
                                                    use_orphan_guard=False,
                                                    use_tool_translator=False,
                                                ):
                                                    yield mark_sse_sent(chunk)
                                    else:
                                        for chunk in emit_read_only_force_answer_visible_text(
                                            visible_text
                                        ):
                                            yield mark_sse_sent(chunk)
                                    stats[
                                        "read_only_force_answer_marker_stream_started"
                                    ] = bool(read_only_force_answer_stream_started)
                                    stats[
                                        "read_only_force_answer_stream_marker_stripped_chars"
                                    ] = int(
                                        read_only_force_answer_stream_marker_stripped_chars
                                    )
                            else:
                                for field, text in drain_stream_tokens([], force=True):
                                    for chunk in stream_content_delta_chunks(
                                        field, text
                                    ):
                                        yield mark_sse_sent(chunk)
                                tail = decoder.finish()
                                if tail:
                                    for field, text in splitter.feed(tail):
                                        if text:
                                            for chunk in stream_content_delta_chunks(
                                                field, text
                                            ):
                                                yield mark_sse_sent(chunk)
                                recover_unclosed_reasoning = (
                                    str(generated.get("finish_reason") or "") == "stop"
                                    and not tools_active
                                )
                                for field, text in _finish_stream_splitter(
                                    splitter,
                                    recover_unclosed_reasoning=(
                                        recover_unclosed_reasoning
                                    ),
                                ):
                                    if text:
                                        for chunk in stream_content_delta_chunks(
                                            field, text
                                        ):
                                            yield mark_sse_sent(chunk)
                                for chunk in finish_orphan_stream_guards():
                                    yield mark_sse_sent(chunk)
                                if (
                                    stop_monitor is not None
                                    and not stop_monitor.stopped
                                ):
                                    held_text = stop_monitor.flush()
                                    if held_text:
                                        for chunk in stream_content_delta_chunks(
                                            "content",
                                            held_text,
                                            use_orphan_guard=False,
                                            monitor_stop=False,
                                        ):
                                            yield mark_sse_sent(chunk)
                                for chunk in finish_translated_stream_chunks():
                                    yield mark_sse_sent(chunk)
                            raw_generated_text = _strip_mtplx_internal_continuation_markers(
                                _strip_generated_chat_template_sentinels(
                                    str(generated.get("text") or "")
                                )
                            )
                            raw_reasoning_text, raw_content_text = (
                                _tool_extraction_text_parts(
                                    state,
                                    raw_generated_text,
                                    thinking_enabled=thinking_enabled,
                                )
                            )
                            extraction = (
                                omlx_extract_tool_calls_with_thinking(
                                    raw_reasoning_text,
                                    raw_content_text,
                                    state.runtime.tokenizer,
                                    tool_specs,
                                )
                                # Forced final-answer turns already streamed
                                # sanitized visible text through the buffered
                                # marker path; running tool extraction over the
                                # raw rehearsal text would re-emit it as a
                                # malformed-as-content fallback.
                                if tools_active and not read_only_force_answer_contract_active
                                else None
                            )
                            assistant_tool_calls = streamed_assistant_tool_calls or (
                                extraction.tool_calls if extraction is not None else None
                            )
                            stats = generated.setdefault("stats", {})
                            stats["openai_bridge_mode"] = "omlx_style"
                            stats["legacy_bridge_used"] = False
                            stats["hidden_generation_repair_used"] = False
                            stats["early_tool_cancel_used"] = bool(
                                early_tool_cancel_used
                            )
                            if extraction is not None:
                                stats["tool_parser_source"] = (
                                    "streaming_translator"
                                    if streamed_assistant_tool_calls
                                    else extraction.parser_source
                                )
                                stats["tool_parse_status"] = (
                                    "success"
                                    if streamed_assistant_tool_calls
                                    else extraction.status
                                )
                                stats["tool_calls_emitted"] = len(
                                    assistant_tool_calls or []
                                )
                                stats["raw_tool_markup_suppressed"] = bool(
                                    extraction.raw_tool_markup_suppressed
                                    or streamed_tool_deltas_emitted
                                    or stream_orphan_tool_markup_suppressed
                                    or (
                                        content_tool_translator is not None
                                        and content_tool_translator.suppressed_tool_markup
                                    )
                                )
                                if (
                                    extraction.status == "malformed_as_content"
                                    and extraction.cleaned_text
                                    and (
                                        fallback_visible_text := _visible_malformed_tool_content(
                                            extraction.cleaned_text,
                                            state.runtime.tokenizer,
                                        )
                                    )
                                    and fallback_visible_text
                                    not in "".join(history_content_chunks)
                                ):
                                    delta = {"content": fallback_visible_text}
                                    remember_stream_delta(delta)
                                    yield mark_sse_sent(delta_payload_chunk(delta))
                            if assistant_tool_calls:
                                if not streamed_tool_deltas_emitted:
                                    for delta in _stream_tool_call_deltas(
                                        assistant_tool_calls,
                                        argument_chunk_chars=stream_interval,
                                    ):
                                        yield mark_sse_sent(delta_payload_chunk(delta))
                                _record_tool_parse_event(
                                    state,
                                    event="tool_parse_success",
                                    response_id=response_id,
                                    stream=True,
                                )
                                stats["tool_parse_success"] = True
                                stats["tool_call_count"] = len(assistant_tool_calls)
                                generated["finish_reason"] = "tool_calls"
                            elif (
                                extraction is not None
                                and extraction.status == "malformed_as_content"
                            ):
                                fallback_reason = (
                                    extraction.malformed_reason
                                    or "malformed_tool_call"
                                )
                                fallback_kind = _tool_parse_counter_key(
                                    fallback_reason
                                )
                                _record_tool_parse_event(
                                    state,
                                    event=fallback_kind,
                                    reason=fallback_reason,
                                    response_id=response_id,
                                    stream=True,
                                )
                                stats["tool_parse_fallback"] = True
                                stats["tool_parse_fallback_reason"] = fallback_reason
                                stats["tool_parse_fallback_kind"] = fallback_kind
                            _merge_final_bridge_stats_into_latest_metrics(
                                state, stats
                            )
                            if session is not None:
                                assistant_history_content = streamed_history_content()
                                commit_state["assistant_history_content"] = (
                                    assistant_history_content
                                )
                                commit_state["assistant_tool_calls"] = (
                                    assistant_tool_calls
                                )
                                commit_state["retokenize_inline"] = (
                                    state.args.session_postcommit_mode == "inline"
                                )
                                commit_state["commit"] = True
                                commit_event.set()
                                commit_kind, commit_item = await asyncio.to_thread(
                                    queue.get
                                )
                                if commit_kind == "committed":
                                    generated = commit_item
                                elif commit_kind == "error":
                                    yield mark_sse_sent(error_chunk(commit_item))
                                    yield mark_sse_sent("data: [DONE]\n\n")
                                    return
                                elif commit_kind == "released":
                                    release = (
                                        commit_item
                                        if isinstance(commit_item, dict)
                                        else {
                                            "generated": generated,
                                            "postcommit": {},
                                        }
                                    )
                                    generated = release.get("generated") or generated
                                    postcommit = release.get("postcommit") or {}
                                    prompt_prefix_boundary_kind = (
                                        "tool_call_prompt_prefix"
                                        if assistant_tool_calls
                                        else "postcommit_prompt_prefix"
                                    )
                                    if read_only_force_answer_contract_active:
                                        prompt_prefix_commit_info = {
                                            "committed": False,
                                            "reason": "transient_generation_contract",
                                            "prefix_len": int(
                                                getattr(session, "prefix_len", 0) or 0
                                            ),
                                            "boundary_kind": prompt_prefix_boundary_kind,
                                        }
                                        prompt_prefix_len = int(
                                            prompt_prefix_commit_info["prefix_len"]
                                        )
                                    else:
                                        prompt_prefix_commit = session.commit_prompt_prefix(
                                            prompt_ids=prompt_ids,
                                            finish_reason=str(
                                                generated.get("finish_reason") or "stop"
                                            ),
                                            boundary_kind=prompt_prefix_boundary_kind,
                                        )
                                        prompt_prefix_commit_info = {
                                            "committed": bool(
                                                prompt_prefix_commit.committed
                                            ),
                                            "reason": prompt_prefix_commit.reason,
                                            "prefix_len": int(
                                                prompt_prefix_commit.prefix_len
                                            ),
                                            "boundary_kind": prompt_prefix_boundary_kind,
                                        }
                                        prompt_prefix_len = int(
                                            prompt_prefix_commit.prefix_len
                                        )
                                    generated["stats"][
                                        "session_prompt_prefix_commit"
                                    ] = prompt_prefix_commit_info
                                    unsafe_reason = str(
                                        postcommit.get("reason") or "unsafe_history"
                                    )
                                    postcommit_snapshot = (
                                        _skipped_idle_postcommit_snapshot(
                                            state=state,
                                            unsafe_reason=unsafe_reason,
                                            assistant_tool_calls=assistant_tool_calls,
                                            prompt_prefix_len=(
                                                prompt_prefix_len
                                            ),
                                        )
                                    )
                                    if postcommit_snapshot is not None:
                                        postcommit_snapshot = (
                                            _attach_skipped_postcommit_cleanup(
                                                state,
                                                postcommit_snapshot,
                                            )
                                        )
                                    else:
                                        postcommit_snapshot = _schedule_idle_postcommit_snapshot(
                                            state,
                                            session_id=session_id,
                                            messages=raw_messages_for_postcommit,
                                            assistant_content=(
                                                assistant_history_content
                                            ),
                                            assistant_tool_calls=assistant_tool_calls,
                                            thinking_enabled=thinking_enabled,
                                            policy_fingerprint=postcommit_policy_fingerprint,
                                            unsafe_reason=unsafe_reason,
                                            tool_specs=postcommit_tool_specs,
                                            session=session,
                                            expected_session_revision=getattr(
                                                session, "revision", None
                                            ),
                                            keep_live_ref=session_keep_live_ref,
                                            tool_prompt_mode=postcommit_tool_prompt_mode,
                                            strip_tool_call_preamble_text=opencode_client,
                                        )
                                    generated["stats"][
                                        "session_postcommit_snapshot"
                                    ] = postcommit_snapshot
                                else:
                                    yield mark_sse_sent(
                                        error_chunk(
                                            RuntimeError(
                                                f"unexpected commit event: {commit_kind}"
                                            )
                                        )
                                    )
                                    yield mark_sse_sent("data: [DONE]\n\n")
                                    return
                            generated = attach_response_observability(generated)
                            _attach_dashboard_progress_stats(
                                state,
                                request_id=response_id,
                                stats=generated.get("stats") or {},
                            )
                            _merge_final_bridge_stats_into_latest_metrics(
                                state, generated.get("stats") or {}
                            )
                            generated["stats"]["reasoning_reentries"] = (
                                splitter.reentry_count
                            )
                            reasoning_text = "".join(history_reasoning_chunks).strip()
                            answer_text = "".join(history_content_chunks).strip()
                            generated["stats"]["reasoning_tokens"] = _count_text_tokens(
                                state.runtime.tokenizer,
                                reasoning_text,
                            )
                            generated["stats"]["answer_tokens"] = _count_text_tokens(
                                state.runtime.tokenizer,
                                answer_text,
                            )
                            if state.last_metrics:
                                state.last_metrics[-1]["reasoning_reentries"] = (
                                    splitter.reentry_count
                                )
                            footer = _stats_footer_text(state, generated)
                            if footer and not assistant_tool_calls:
                                # The footer is server-injected, not model
                                # output: bypass stop monitoring so a stop
                                # string like "\n\n" can neither suppress it
                                # nor false-match against it.
                                for chunk in stream_content_delta_chunks(
                                    "content",
                                    f"\n\n{footer}",
                                    monitor_stop=False,
                                ):
                                    yield mark_sse_sent(chunk)
                            break
                        elif kind == "reset_orphan_stream_guards":
                            reset_orphan_stream_guards()
                            continue
                        elif kind == "close_unclosed_reasoning_for_repair":
                            if _reasoning_parser_for_state(state) not in {"qwen3", "step3p5"}:
                                continue
                            for field, text in drain_stream_tokens([], force=True):
                                for chunk in stream_content_delta_chunks(field, text):
                                    yield mark_sse_sent(chunk)
                            tail = decoder.finish()
                            if tail:
                                for field, text in splitter.feed(tail):
                                    if text:
                                        for chunk in stream_content_delta_chunks(
                                            field, text
                                        ):
                                            yield mark_sse_sent(chunk)
                            for field, text in splitter.feed(THINK_CLOSE):
                                if text:
                                    for chunk in stream_content_delta_chunks(field, text):
                                        yield mark_sse_sent(chunk)
                            decoder = _IncrementalTokenDecoder(state.runtime.tokenizer)
                            continue
                        elif kind == "error":
                            yield mark_sse_sent(error_chunk(item))
                            yield mark_sse_sent("data: [DONE]\n\n")
                            return
                        elif kind == "cancelled":
                            if stop_monitor is not None and stop_monitor.stopped:
                                # Stop-sequence cancel: the worker unwound
                                # through the normal cancellation path, but for
                                # the client this is a successful completion
                                # that ended at the stop string.
                                generated = {
                                    "text": streamed_history_content(),
                                    "tokens": list(streamed_token_ids),
                                    "prompt_tokens": len(prompt_ids),
                                    "completion_tokens": int(
                                        streamed_progress_tokens
                                    ),
                                    "finish_reason": "stop",
                                    "stats": {
                                        "generation_mode": request_generation_mode,
                                        "mtp_depth": request_depth,
                                        "prompt_tokens": len(prompt_ids),
                                        "completion_tokens": int(
                                            streamed_progress_tokens
                                        ),
                                        "stop_sequence_hit": True,
                                        "stop_sequence_matched": (
                                            stop_monitor.matched_stop
                                        ),
                                        "openai_bridge_mode": "omlx_style",
                                        "legacy_bridge_used": False,
                                        "hidden_generation_repair_used": False,
                                        "early_tool_cancel_used": False,
                                    },
                                }
                                generated = attach_response_observability(
                                    generated
                                )
                                _attach_dashboard_progress_stats(
                                    state,
                                    request_id=response_id,
                                    stats=generated["stats"],
                                )
                                _merge_final_bridge_stats_into_latest_metrics(
                                    state, generated["stats"]
                                )
                                break
                            if early_tool_cancel_used and streamed_assistant_tool_calls:
                                generated = {
                                    "text": streamed_history_content(),
                                    "tokens": list(streamed_token_ids),
                                    "prompt_tokens": len(prompt_ids),
                                    "completion_tokens": int(streamed_progress_tokens),
                                    "finish_reason": "tool_calls",
                                    "stats": {
                                        "generation_mode": request_generation_mode,
                                        "mtp_depth": request_depth,
                                        "prompt_tokens": len(prompt_ids),
                                        "completion_tokens": int(
                                            streamed_progress_tokens
                                        ),
                                        "tool_parse_success": True,
                                        "tool_call_count": len(
                                            streamed_assistant_tool_calls
                                        ),
                                        "tool_parser_source": (
                                            "streaming_translator"
                                        ),
                                        "tool_parse_status": "success",
                                        "tool_calls_emitted": len(
                                            streamed_assistant_tool_calls
                                        ),
                                        "raw_tool_markup_suppressed": True,
                                        "early_tool_cancel_used": True,
                                        "openai_bridge_mode": "omlx_style",
                                        "legacy_bridge_used": False,
                                        "hidden_generation_repair_used": False,
                                    },
                                }
                                generated = attach_response_observability(generated)
                                _attach_dashboard_progress_stats(
                                    state,
                                    request_id=response_id,
                                    stats=generated["stats"],
                                )
                                _merge_final_bridge_stats_into_latest_metrics(
                                    state, generated["stats"]
                                )
                                break
                            return
                        else:
                            yield mark_sse_sent(
                                error_chunk(
                                    RuntimeError(f"unexpected stream event: {kind}")
                                )
                            )
                            yield mark_sse_sent("data: [DONE]\n\n")
                            return
                except asyncio.CancelledError:
                    stream_cancelled_by_client = True
                    _cancel_stream_generation(cancel_event, generation_future)
                    raise
                except BaseException as exc:
                    yield mark_sse_sent(error_chunk(exc))
                    yield mark_sse_sent("data: [DONE]\n\n")
                    return
                finally:
                    nonlocal_cancel_reason = (
                        "client_disconnected"
                        if stream_cancelled_by_client
                        else "stream_cancelled"
                    )
                    _cancel_stream_generation(cancel_event, generation_future)
                    if (
                        stream_cancelled_by_client
                        and session is not None
                        and hasattr(session, "abort_pending_postcommit")
                    ):
                        session.abort_pending_postcommit("stream_cancelled")
                    if session is not None and not commit_event.is_set():
                        commit_state["commit"] = False
                        commit_event.set()
                    if cancel_event.is_set() and generated is None:
                        state.dashboard.lifetime.record_cancellation()
                        if not cancelled_metric_recorded:
                            _record_stream_cancellation_metric(
                                state,
                                response_id=response_id,
                                session_id=session_id,
                                prompt_tokens=len(prompt_ids),
                                streamed_completion_tokens=streamed_progress_tokens,
                                stream_started_s=stream_started_s,
                                reason=nonlocal_cancel_reason,
                                request_observability=request_observability,
                                client_disconnected=stream_cancelled_by_client,
                            )
                            cancelled_metric_recorded = True
                    state.dashboard.in_flight.deregister(response_id)
                    state.dashboard.progress_events.forget(response_id)

                if generated is None:
                    yield mark_sse_sent(
                        error_chunk(RuntimeError("generation ended without a result"))
                    )
                    yield mark_sse_sent("data: [DONE]\n\n")
                    return
                finish_reason = generated.get("finish_reason") or "stop"
                generated["finish_reason"] = finish_reason
                generated.setdefault("stats", {})["finish_reason"] = finish_reason
                _merge_final_bridge_stats_into_latest_metrics(
                    state, {"finish_reason": finish_reason}
                )
                done = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": _usage_payload(generated),
                    "mtplx_stats": _public_mtplx_stats(generated),
                }
                yield mark_sse_sent(f"data: {json.dumps(done)}\n\n")
                yield mark_sse_sent("data: [DONE]\n\n")

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        def run_nonstream_generation() -> dict[str, Any]:
            return run_generation_for_response()

        nonstream_handle = InFlightHandle(
            request_id=response_id,
            cancel_event=nonstream_cancel_event,
            started_s=time.time(),
            session_id=session_id,
            model=model,
            prompt_preview=_dashboard_prompt_preview(request, state.runtime.tokenizer),
            prompt_tokens=len(prompt_ids),
        )
        state.dashboard.in_flight.register(nonstream_handle)
        nonstream_started_s = time.perf_counter()

        def mark_nonstream_client_disconnected() -> None:
            nonlocal nonstream_cancel_reason
            nonlocal nonstream_client_disconnected
            nonstream_cancel_reason = "client_disconnected"
            nonstream_client_disconnected = True

        disconnect_monitor_task = asyncio.create_task(
            _monitor_request_disconnect(
                raw_request,
                nonstream_cancel_event,
                on_disconnect=mark_nonstream_client_disconnected,
            )
        )
        try:
            generated = await asyncio.to_thread(run_nonstream_generation)
        except EngineSessionBusy as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except _StopSequenceHit:
            # A client stop string matched mid-generation: this is a normal
            # completion that ends at the stop boundary, not a cancellation.
            # The monitor holds exactly the visible text before the match;
            # session postcommit is skipped, matching the streaming stop path.
            assert nonstream_stop_monitor is not None
            stop_reasoning_text = "".join(nonstream_stop_reasoning_chunks).strip()
            stop_generated: dict[str, Any] = {
                "text": nonstream_stop_monitor.emitted_text,
                "tokens": [],
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": int(nonstream_completion_tokens),
                "finish_reason": "stop",
                "stats": {
                    "generation_mode": request_generation_mode,
                    "mtp_depth": request_depth,
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": int(nonstream_completion_tokens),
                    "stop_sequence_hit": True,
                    "stop_sequence_matched": (
                        nonstream_stop_monitor.matched_stop
                    ),
                    "openai_bridge_mode": "omlx_style",
                    "legacy_bridge_used": False,
                    "hidden_generation_repair_used": False,
                    "early_tool_cancel_used": False,
                    "finish_reason": "stop",
                },
            }
            stop_generated = attach_response_observability(stop_generated)
            _merge_final_bridge_stats_into_latest_metrics(
                state, stop_generated["stats"]
            )
            stop_message: dict[str, Any] = {
                "role": "assistant",
                "content": nonstream_stop_monitor.emitted_text,
            }
            if stop_reasoning_text and not suppress_visible_reasoning:
                stop_message["reasoning_content"] = stop_reasoning_text
            return JSONResponse(
                {
                    "id": response_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": stop_message,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": _usage_payload(stop_generated),
                    "mtplx_stats": _public_mtplx_stats(stop_generated),
                }
            )
        except _StreamCancelled as exc:
            state.dashboard.lifetime.record_cancellation()
            _record_stream_cancellation_metric(
                state,
                response_id=response_id,
                session_id=session_id,
                prompt_tokens=len(prompt_ids),
                streamed_completion_tokens=nonstream_completion_tokens,
                stream_started_s=nonstream_started_s,
                reason=nonstream_cancel_reason,
                request_observability=request_observability,
                client_disconnected=nonstream_client_disconnected,
            )
            return JSONResponse(
                status_code=499,
                content=_openai_error_content(
                    str(exc),
                    status_code=499,
                    code="request_cancelled",
                    error_type="request_cancelled",
                ),
            )
        finally:
            disconnect_monitor_task.cancel()
            with suppress(asyncio.CancelledError, TimeoutError):
                await asyncio.wait_for(disconnect_monitor_task, timeout=0.25)
            state.dashboard.in_flight.deregister(response_id)
            state.dashboard.progress_events.forget(response_id)
        generated.setdefault("stats", {})
        generated["stats"]["openai_bridge_mode"] = "omlx_style"
        generated["stats"]["legacy_bridge_used"] = False
        generated["stats"]["hidden_generation_repair_used"] = False
        generated["stats"]["early_tool_cancel_used"] = False
        if tools_active:
            raw_text = _strip_mtplx_internal_continuation_markers(
                _strip_generated_chat_template_sentinels(
                    str(generated.get("text") or "")
                )
            )
            raw_reasoning_text, raw_content_text = _tool_extraction_text_parts(
                state,
                raw_text,
                thinking_enabled=thinking_enabled,
            )
            extraction = omlx_extract_tool_calls_with_thinking(
                raw_reasoning_text,
                raw_content_text,
                state.runtime.tokenizer,
                tool_specs,
            )
            tool_calls = extraction.tool_calls
            generated["stats"]["tool_parser_source"] = extraction.parser_source
            generated["stats"]["tool_parse_status"] = extraction.status
            generated["stats"]["tool_calls_emitted"] = len(tool_calls or [])
            generated["stats"]["raw_tool_markup_suppressed"] = bool(
                extraction.raw_tool_markup_suppressed
            )
        else:
            tool_calls = None
            extraction = None
        if tool_calls:
            generated["finish_reason"] = "tool_calls"
            generated["stats"]["tool_parse_success"] = True
            generated["stats"]["tool_call_count"] = len(tool_calls)
            _record_tool_parse_event(
                state,
                event="tool_parse_success",
                response_id=response_id,
                stream=False,
            )
            _merge_final_bridge_stats_into_latest_metrics(
                state, generated["stats"]
            )
            assistant_content = (
                extraction.cleaned_text.strip()
                if extraction is not None and extraction.cleaned_text
                else ""
            )
            await store_postcommit_snapshot(
                generated,
                assistant_content=assistant_content,
                assistant_tool_calls=tool_calls,
            )
            _merge_final_bridge_stats_into_latest_metrics(
                state, generated["stats"]
            )
            message: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_content or None,
                "tool_calls": tool_calls,
            }
            finish_reason = "tool_calls"
        else:
            reasoning_text = ""
            if (
                extraction is not None
                and extraction.status == "malformed_as_content"
            ):
                display_text = _visible_malformed_tool_content(
                    extraction.cleaned_text,
                    state.runtime.tokenizer,
                )
                display_text = _strip_mtplx_internal_continuation_markers(
                    display_text
                )
                fallback_reason = extraction.malformed_reason or "malformed_tool_call"
                fallback_kind = _tool_parse_counter_key(fallback_reason)
                generated["stats"]["tool_parse_fallback"] = True
                generated["stats"]["tool_parse_fallback_reason"] = fallback_reason
                generated["stats"]["tool_parse_fallback_kind"] = fallback_kind
                generated["stats"]["raw_tool_markup_suppressed"] = bool(
                    generated["stats"].get("raw_tool_markup_suppressed")
                    or display_text != extraction.cleaned_text
                )
                _record_tool_parse_event(
                    state,
                    event=fallback_kind,
                    reason=fallback_reason,
                    response_id=response_id,
                    stream=False,
                )
            else:
                display_text, reasoning_text = _nonstream_chat_message_parts(
                    state,
                    generated,
                    thinking_enabled=thinking_enabled,
                    suppress_visible_reasoning=suppress_visible_reasoning,
                )
            if stop_sequences:
                # Post-trim safety net for matches the incremental monitor
                # cannot see (e.g. a stop string completed only by the
                # decoder's held-back partial word at end of generation).
                trimmed_text, matched_stop = _trim_text_at_stop_sequences(
                    display_text, stop_sequences
                )
                if matched_stop is not None:
                    display_text = trimmed_text
                    generated["finish_reason"] = "stop"
                    generated["stats"]["stop_sequence_hit"] = True
                    generated["stats"]["stop_sequence_matched"] = matched_stop
            _merge_final_bridge_stats_into_latest_metrics(
                state, generated["stats"]
            )
            await store_postcommit_snapshot(
                generated,
                assistant_content=display_text,
            )
            _merge_final_bridge_stats_into_latest_metrics(
                state, generated["stats"]
            )
            message = {"role": "assistant", "content": display_text}
            if reasoning_text:
                message["reasoning_content"] = reasoning_text
            finish_reason = generated.get("finish_reason", "stop")
        return JSONResponse(
            {
                "id": response_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": _usage_payload(generated),
                "mtplx_stats": _public_mtplx_stats(generated),
            }
        )

    @app.post("/v1/messages")
    async def anthropic_messages(
        raw_request: Request, request: AnthropicMessagesRequest
    ) -> Any:
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        chat_request = _anthropic_to_chat_request(request)
        chat_request.stream = bool(request.stream)
        response = await chat_completions(raw_request, chat_request)
        if request.stream:
            if not isinstance(response, StreamingResponse):
                return response
            return StreamingResponse(
                _anthropic_stream_from_openai_sse(
                    response.body_iterator,
                    model=state.model_id,
                ),
                media_type="text/event-stream",
            )
        if not isinstance(response, JSONResponse):
            return response
        try:
            openai_payload = json.loads(response.body)
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"failed to translate response: {exc}"
            ) from exc
        if response.status_code >= 400:
            return response
        payload = _anthropic_payload_from_openai(openai_payload)
        return JSONResponse(payload, status_code=response.status_code)

    @app.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens(
        raw_request: Request, request: AnthropicMessagesRequest
    ) -> Any:
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        chat_request = _anthropic_to_chat_request(request)
        headers = dict(raw_request.headers)
        metadata = _request_metadata(chat_request)
        requested_tool_specs = _normalize_tool_specs(chat_request.tools)
        tools_active = _tools_active_for_request(
            requested_tool_specs,
            chat_request.tool_choice,
        )
        messages_for_generation, _transcript_stats = _canonicalize_agent_transcript(
            chat_request.messages,
            tools_active=tools_active,
        )
        messages_for_generation, _backend_chat_policy_active = _with_backend_chat_policy(
            state,
            messages_for_generation,
        )
        client_controls_allowed = _client_controls_allowed(headers, metadata)
        thinking_enabled = _thinking_enabled_for_request(
            state,
            chat_request,
            allow_client_controls=client_controls_allowed,
        )
        reasoning_effort = _reasoning_effort_for_state(
            state,
            thinking_enabled=thinking_enabled,
            request_effort=chat_request.reasoning_effort,
            allow_client_controls=client_controls_allowed,
        )
        tool_prompt_mode, _tool_prompt_mode_resolution = _tool_prompt_mode_for_request(
            state.args,
            headers=headers,
            metadata=metadata,
            tools_active=tools_active,
        )
        prompt_ids = _encode_messages(
            state.runtime.tokenizer,
            messages_for_generation,
            enable_thinking=thinking_enabled,
            reasoning_effort=reasoning_effort,
            strip_assistant_reasoning_history=state.args.strip_assistant_reasoning_history,
            tools=requested_tool_specs if tools_active else None,
            tool_choice=chat_request.tool_choice,
            tool_prompt_mode=tool_prompt_mode,
        )
        return {"input_tokens": len(prompt_ids)}

    @app.post("/v1/completions")
    async def completions(raw_request: Request, request: CompletionRequest) -> Any:
        headers = dict(raw_request.headers)
        raw_metadata = _request_extra(request, "metadata", {})
        metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
        client_controls_allowed = _client_controls_allowed(headers, metadata)
        prompt_ids = _encode_prompt(state.runtime.tokenizer, request.prompt)
        request_generation_mode = _request_generation_mode_for_generation(
            state,
            request,
            allow_client_controls=client_controls_allowed,
        )
        request_depth = _request_depth_for_generation(
            state,
            request,
            generation_mode=request_generation_mode,
            allow_client_controls=client_controls_allowed,
        )
        effective_request_depth, _ = _long_context_mtp_depth_policy_for_request(
            state,
            generation_mode=request_generation_mode,
            request_depth=request_depth,
            prompt_tokens=len(prompt_ids),
        )
        sampler_temperature = request.temperature if client_controls_allowed else None
        sampler_top_p = request.top_p if client_controls_allowed else None
        sampler_top_k = request.top_k if client_controls_allowed else None
        request_observability = {
            "request_client_hint": _request_client_hint_from_headers(headers, metadata),
            "request_client_label": _request_client_hint_from_headers(headers, metadata)
            or "openai",
            "request_generation_mode": request_generation_mode,
            "request_depth": int(request_depth),
            "request_effective_mtp_depth": int(effective_request_depth),
            "request_temperature": request.temperature,
            "request_top_p": request.top_p,
            "request_top_k": request.top_k,
            "mtplx_control_owner": (
                "client" if client_controls_allowed else "server"
            ),
            "client_controls_allowed": bool(client_controls_allowed),
        }
        if not client_controls_allowed:
            ignored_fields = _ignored_client_control_fields(request)
            if ignored_fields:
                request_observability["client_control_fields_ignored"] = ignored_fields
                request_observability["client_sampler_fields_ignored"] = [
                    field
                    for field in ignored_fields
                    if field in {"temperature", "top_p", "top_k"}
                ]
        stop_sequences = _normalize_stop_sequences(request.stop)
        model = state.model_id
        response_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        if request.stream:
            # Real incremental streaming: tokens flow through a queue from the
            # generation worker and are decoded as they arrive, mirroring the
            # chat pattern. The previous implementation generated everything
            # first and then re-chunked the final text, which kept clients
            # staring at a silent stream for the whole generation.
            async def event_stream():
                queue: Queue[tuple[str, Any]] = Queue()
                cancel_event = Event()
                decoder = _IncrementalTokenDecoder(state.runtime.tokenizer)
                stop_monitor = _StopSequenceStreamMonitor(stop_sequences)
                streamed_completion_tokens = 0
                generated: dict[str, Any] | None = None
                stop_hit = False

                def on_tokens(new_tokens: list[int]) -> None:
                    _raise_if_stream_cancelled(cancel_event)
                    queue.put(("tokens", [int(token) for token in new_tokens]))
                    _raise_if_stream_cancelled(cancel_event)

                def worker() -> None:
                    try:
                        result = _run_generation_dispatched(
                            state,
                            prompt_ids,
                            batch_key="completion.stream",
                            response_id=response_id,
                            max_tokens=request.max_tokens,
                            temperature=sampler_temperature,
                            top_p=sampler_top_p,
                            top_k=sampler_top_k,
                            seed=request.seed,
                            generation_mode=request_generation_mode,
                            depth=request_depth,
                            resolved_mtp_depth=effective_request_depth,
                            token_callback=on_tokens,
                            request_observability=request_observability,
                            cancel_event=cancel_event,
                        )
                    except _StreamCancelled as exc:
                        queue.put(_stream_cancelled_queue_item(exc))
                    except BaseException as exc:
                        queue.put(("error", exc))
                    else:
                        queue.put(("done", result))

                generation_future: Future = Future()

                def run_worker_thread() -> None:
                    try:
                        worker()
                    except BaseException as exc:
                        queue.put(("error", exc))
                        if not generation_future.done():
                            generation_future.set_exception(exc)
                    else:
                        if not generation_future.done():
                            generation_future.set_result(None)

                Thread(
                    target=run_worker_thread,
                    name=f"mtplx-completion-worker-{response_id[-8:]}",
                    daemon=True,
                ).start()

                def text_chunk(text: str) -> str:
                    payload = {
                        "id": response_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [
                            {"index": 0, "text": text, "finish_reason": None}
                        ],
                    }
                    return f"data: {json.dumps(payload)}\n\n"

                def error_chunk(exc: BaseException) -> str:
                    if isinstance(exc, HTTPException):
                        message = str(exc.detail)
                        status_code = exc.status_code
                    else:
                        message = str(exc)
                        status_code = 500
                    payload = {
                        "id": response_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [
                            {"index": 0, "text": "", "finish_reason": "error"}
                        ],
                        **_openai_error_content(
                            message,
                            status_code=status_code,
                            code=type(exc).__name__,
                        ),
                    }
                    return f"data: {json.dumps(payload)}\n\n"

                def emit_text(text: str, *, monitor: bool = True) -> list[str]:
                    nonlocal stop_hit
                    if not text:
                        return []
                    if monitor and stop_monitor.active:
                        if stop_monitor.stopped:
                            return []
                        text = stop_monitor.feed(text)
                        if stop_monitor.stopped and not stop_hit:
                            stop_hit = True
                            _cancel_stream_generation(
                                cancel_event, generation_future
                            )
                        if not text:
                            return []
                    return [text_chunk(text)]

                try:
                    while True:
                        try:
                            kind, item = await asyncio.to_thread(
                                queue.get, True, 0.25
                            )
                        except Empty:
                            if (
                                cancel_event.is_set() and not stop_hit
                            ) or await raw_request.is_disconnected():
                                _cancel_stream_generation(
                                    cancel_event, generation_future
                                )
                                return
                            continue
                        if kind == "tokens":
                            streamed_completion_tokens += len(item)
                            for chunk in emit_text(decoder.feed(item)):
                                yield chunk
                        elif kind == "done":
                            generated = item
                            for chunk in emit_text(decoder.finish()):
                                yield chunk
                            held_text = stop_monitor.flush()
                            for chunk in emit_text(held_text, monitor=False):
                                yield chunk
                            break
                        elif kind == "cancelled":
                            if stop_hit:
                                generated = {
                                    "text": stop_monitor.emitted_text,
                                    "tokens": [],
                                    "prompt_tokens": len(prompt_ids),
                                    "completion_tokens": int(
                                        streamed_completion_tokens
                                    ),
                                    "finish_reason": "stop",
                                    "stats": {
                                        "generation_mode": request_generation_mode,
                                        "mtp_depth": request_depth,
                                        "prompt_tokens": len(prompt_ids),
                                        "completion_tokens": int(
                                            streamed_completion_tokens
                                        ),
                                    },
                                }
                                break
                            return
                        elif kind == "error":
                            yield error_chunk(item)
                            yield "data: [DONE]\n\n"
                            return
                        else:
                            yield error_chunk(
                                RuntimeError(f"unexpected stream event: {kind}")
                            )
                            yield "data: [DONE]\n\n"
                            return
                except asyncio.CancelledError:
                    _cancel_stream_generation(cancel_event, generation_future)
                    raise
                except BaseException as exc:
                    yield error_chunk(exc)
                    yield "data: [DONE]\n\n"
                    return
                finally:
                    _cancel_stream_generation(cancel_event, generation_future)

                if generated is None:
                    yield error_chunk(
                        RuntimeError("generation ended without a result")
                    )
                    yield "data: [DONE]\n\n"
                    return
                finish_reason = str(generated.get("finish_reason") or "stop")
                stats = generated.setdefault("stats", {})
                if stop_hit or stop_monitor.stopped:
                    finish_reason = "stop"
                    generated["finish_reason"] = "stop"
                    stats["stop_sequence_hit"] = True
                    stats["stop_sequence_matched"] = stop_monitor.matched_stop
                else:
                    footer = (
                        _stats_footer_text(state, generated)
                        if state.args.stats_footer
                        else ""
                    )
                    if footer:
                        for chunk in emit_text(f"\n\n{footer}", monitor=False):
                            yield chunk
                stats["finish_reason"] = finish_reason
                final_payload = {
                    "id": response_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {"index": 0, "text": "", "finish_reason": finish_reason}
                    ],
                    "usage": _usage_payload(generated),
                    "mtplx_stats": _public_mtplx_stats(generated),
                }
                yield f"data: {json.dumps(final_payload)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        nonstream_stop_monitor: _StopSequenceStreamMonitor | None = None
        nonstream_stop_decoder: _IncrementalTokenDecoder | None = None
        nonstream_completion_tokens = 0
        if stop_sequences:
            nonstream_stop_monitor = _StopSequenceStreamMonitor(stop_sequences)
            nonstream_stop_decoder = _IncrementalTokenDecoder(
                state.runtime.tokenizer
            )

        def nonstream_stop_on_tokens(new_tokens: list[int]) -> None:
            nonlocal nonstream_completion_tokens
            nonstream_completion_tokens += len(new_tokens)
            if nonstream_stop_monitor is None or nonstream_stop_decoder is None:
                return
            delta = nonstream_stop_decoder.feed(
                [int(token) for token in new_tokens]
            )
            if not delta:
                return
            nonstream_stop_monitor.feed(delta)
            if nonstream_stop_monitor.stopped:
                raise _StopSequenceHit(
                    nonstream_stop_monitor.matched_stop or ""
                )

        try:
            generated = await asyncio.to_thread(
                lambda: _run_generation_dispatched(
                    state,
                    prompt_ids,
                    batch_key="completion",
                    response_id=response_id,
                    max_tokens=request.max_tokens,
                    temperature=sampler_temperature,
                    top_p=sampler_top_p,
                    top_k=sampler_top_k,
                    seed=request.seed,
                    generation_mode=request_generation_mode,
                    depth=request_depth,
                    resolved_mtp_depth=effective_request_depth,
                    request_observability=request_observability,
                    token_callback=(
                        nonstream_stop_on_tokens
                        if nonstream_stop_monitor is not None
                        else None
                    ),
                )
            )
        except _StopSequenceHit:
            # A stop string matched mid-generation: return the text before
            # the match as a normal completion instead of burning tokens
            # until EOS/max_tokens.
            assert nonstream_stop_monitor is not None
            generated = {
                "text": nonstream_stop_monitor.emitted_text,
                "tokens": [],
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": int(nonstream_completion_tokens),
                "finish_reason": "stop",
                "stats": {
                    "generation_mode": request_generation_mode,
                    "mtp_depth": request_depth,
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": int(nonstream_completion_tokens),
                    "stop_sequence_hit": True,
                    "stop_sequence_matched": (
                        nonstream_stop_monitor.matched_stop
                    ),
                },
            }
        finish_reason = str(generated.get("finish_reason") or "stop")
        if stop_sequences and not generated.get("stats", {}).get(
            "stop_sequence_hit"
        ):
            # Post-trim safety net for matches the incremental monitor cannot
            # see (e.g. completed only by the decoder's held-back tail).
            trimmed_text, matched_stop = _trim_text_at_stop_sequences(
                str(generated.get("text") or ""), stop_sequences
            )
            if matched_stop is not None:
                generated["text"] = trimmed_text
                finish_reason = "stop"
                generated["finish_reason"] = "stop"
                generated.setdefault("stats", {})["stop_sequence_hit"] = True
                generated["stats"]["stop_sequence_matched"] = matched_stop
        generated.setdefault("stats", {})["finish_reason"] = finish_reason
        display_text = _display_text(state, generated)
        return JSONResponse(
            {
                "id": response_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "text": display_text, "finish_reason": finish_reason}
                ],
                "usage": _usage_payload(generated),
                "mtplx_stats": _public_mtplx_stats(generated),
            }
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        _request: Request, exc: HTTPException
    ) -> JSONResponse:
        _record_tool_parse_event(state, event="openai_error_response")
        detail_payload = exc.detail if isinstance(exc.detail, dict) else None
        message = str(exc.detail)
        if detail_payload is not None:
            detail_message = detail_payload.get("message")
            if isinstance(detail_message, str) and detail_message:
                message = detail_message
        return JSONResponse(
            status_code=exc.status_code,
            headers=getattr(exc, "headers", None),
            content=_openai_error_content(
                message,
                status_code=exc.status_code,
                code=type(exc).__name__,
                detail=detail_payload,
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        _record_tool_parse_event(state, event="openai_error_response")
        first = exc.errors()[0] if exc.errors() else {}
        loc = first.get("loc") if isinstance(first, dict) else None
        param = (
            ".".join(str(item) for item in loc)
            if isinstance(loc, (list, tuple))
            else None
        )
        message = first.get("msg") if isinstance(first, dict) else str(exc)
        return JSONResponse(
            status_code=422,
            content=_openai_error_content(
                str(message),
                status_code=422,
                code="request_validation_error",
                param=param,
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception(_request: Request, exc: Exception) -> JSONResponse:
        _record_tool_parse_event(state, event="openai_error_response")
        request_id = uuid.uuid4().hex[:12]
        return JSONResponse(
            status_code=500,
            content=_openai_error_content(
                f"{type(exc).__name__}: {exc} (request_id={request_id})",
                status_code=500,
                code=type(exc).__name__,
            ),
        )

    # Mount the dashboard SPA last so its catch-all does not shadow the
    # explicit routes above (FastAPI/Starlette route resolution is order-
    # sensitive once a StaticFiles mount with html=True is involved).
    _mount_dashboard(app)

    return app


def _mount_dashboard(app: FastAPI) -> None:
    """Mount the dashboard SPA at ``/dashboard`` if the bundle is present.

    When the bundle is missing (e.g. running from source without a
    ``bun run build``) we register a fallback HTML route at the same path
    with build instructions, so curl ``/dashboard`` always returns
    something useful.
    """

    from mtplx.dashboard import DASHBOARD_STATIC_DIR, has_static_bundle

    if has_static_bundle():
        from fastapi.staticfiles import StaticFiles

        app.mount(
            "/dashboard",
            StaticFiles(directory=str(DASHBOARD_STATIC_DIR), html=True),
            name="dashboard",
        )
        return

    fallback_html = (
        "<!doctype html><meta charset=utf-8><title>MTPLX Dashboard</title>"
        "<body style='font-family:ui-sans-serif,system-ui;background:#0a0a0a;"
        "color:#e8eef3;padding:48px;max-width:720px'>"
        "<h1>MTPLX Dashboard</h1>"
        "<p>The dashboard SPA bundle is missing from this MTPLX install.</p>"
        "<pre style='background:#14181d;padding:16px;border-radius:8px'>"
        "cd dashboard\nbun install\nbun run build</pre>"
        "<p>Then restart <code>mtplx serve</code> and reload this page.</p>"
        f"<p style='color:#8d97a3;font-size:13px'>Expected bundle: "
        f"<code>{DASHBOARD_STATIC_DIR}</code></p>"
        "</body>"
    )

    @app.get("/dashboard", response_class=HTMLResponse)
    @app.get("/dashboard/", response_class=HTMLResponse)
    def dashboard_fallback() -> HTMLResponse:
        return HTMLResponse(fallback_html, status_code=200)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _explicit_server_flags(raw_args: list[str]) -> set[str]:
    flags: set[str] = set()
    for token in raw_args:
        if not token.startswith("-") or token in {"-", "--"}:
            continue
        head = token.split("=", 1)[0]
        if head.startswith("--"):
            flags.add(head[2:])
        else:
            flags.add(head[1:])
    return flags


def _server_flag_present(flags: set[str], *names: str) -> bool:
    candidates = set(names)
    candidates.update(name.replace("_", "-") for name in names)
    candidates.update(name.replace("-", "_") for name in names)
    return any(name in flags for name in candidates)


def _model_ref_is_gemma4_pair(model_ref: str | None) -> bool:
    if is_gemma4_pair_repo_id(model_ref):
        return True
    if not model_ref:
        return False
    try:
        return resolve_gemma4_pair_paths(model_ref) is not None
    except Exception:
        return False


def _gemma4_bundle_defaults(model_ref: str | None) -> tuple[dict[str, Any] | None, int | None]:
    if not model_ref:
        return None, None
    pair = resolve_gemma4_pair_paths(model_ref)
    if pair is None:
        return None, None
    metadata = pair["metadata"]
    sampler = gemma4_pair_sampler_defaults(
        target_model=pair["target_model"],
        metadata=metadata,
    )
    benchmark = metadata.get("benchmark") if isinstance(metadata, dict) else {}
    draft_block_size = None
    if isinstance(benchmark, dict):
        try:
            draft_block_size = int(benchmark.get("best_block_size"))
        except (TypeError, ValueError):
            draft_block_size = None
    return sampler, draft_block_size


def _apply_backend_server_defaults(
    args: argparse.Namespace,
    *,
    explicit_flags: set[str],
) -> None:
    if (
        not _server_flag_present(explicit_flags, "backend-id")
        and _model_ref_is_gemma4_pair(getattr(args, "model", None))
    ):
        args.backend_id = GEMMA4_BACKEND

    sync_backend_arg_aliases(args)
    backend = descriptor_for_backend_id(getattr(args, "backend_id", None))
    if not _server_flag_present(explicit_flags, "reasoning-parser"):
        args.reasoning_parser = backend.reasoning_codec.parser
    if (
        not _server_flag_present(explicit_flags, "reasoning-effort")
        and getattr(args, "reasoning_effort", None) in (None, "auto")
        and backend.reasoning_codec.default_effort
    ):
        args.reasoning_effort = backend.reasoning_codec.default_effort
    if backend.backend_id != GEMMA4_BACKEND:
        return

    sampler, draft_block_size = _gemma4_bundle_defaults(getattr(args, "model", None))
    sampler = sampler or backend.sampler_defaults.to_dict()
    draft_block_size = backend.draft_semantics.clamp(
        draft_block_size or backend.draft_semantics.default
    )
    if not _server_flag_present(explicit_flags, "model-id"):
        args.model_id = "mtplx-gemma4-31b-assistant-mtp"
    if not _server_flag_present(explicit_flags, "temperature", "default-temperature"):
        args.temperature = float(sampler["temperature"])
    if not _server_flag_present(explicit_flags, "top-p", "default-top-p"):
        args.top_p = float(sampler["top_p"])
    if not _server_flag_present(explicit_flags, "top-k"):
        args.top_k = int(sampler["top_k"])
    if not _server_flag_present(explicit_flags, "draft-top-p"):
        args.draft_top_p = float(sampler["top_p"])
    if not _server_flag_present(explicit_flags, "draft-top-k"):
        args.draft_top_k = int(sampler["top_k"])
    if (
        not _server_flag_present(explicit_flags, "chat-template-profile")
        or getattr(args, "chat_template_profile", None) == _CHAT_TEMPLATE_PROFILE_LOCAL
    ):
        args.chat_template_profile = _CHAT_TEMPLATE_PROFILE_TOKENIZER
    if not _server_flag_present(
        explicit_flags,
        "reasoning",
        "reasoning-mode",
        "enable-thinking",
        "no-enable-thinking",
    ):
        args.reasoning = backend.reasoning_codec.default_mode
        args.reasoning_mode = backend.reasoning_codec.default_mode
        args.enable_thinking = backend.reasoning_codec.default_mode != "off"
    if not _server_flag_present(
        explicit_flags,
        "depth",
        "mtp-depth",
        "speculative-depth",
        backend.draft_semantics.request_field,
        "draft-block-size",
        "gemma-draft-block-size",
    ):
        set_draft_control_arg(args, backend, int(draft_block_size))
    else:
        sync_backend_arg_aliases(args)
        if (
            getattr(args, "draft_block_size", None) is None
            and _server_flag_present(
                explicit_flags,
                "depth",
                "mtp-depth",
                "speculative-depth",
            )
            and getattr(args, "depth", None) is not None
        ):
            set_draft_control_arg(args, backend, int(args.depth))
    if not _server_flag_present(
        explicit_flags,
        "target-distribution-mode",
        "gemma-target-distribution-mode",
    ):
        target_mode = target_distribution_mode_from_args(args, backend)
        if target_mode is not None:
            args.target_distribution_mode = target_mode
            args.gemma_target_distribution_mode = target_mode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    postcommit_default = os.environ.get("MTPLX_SESSION_POSTCOMMIT_MODE", "async")
    if postcommit_default not in {"inline", "async"}:
        postcommit_default = "async"
    parser.add_argument("--model", default=DEFAULT_HF_MODEL_ID)
    parser.add_argument("--model-id", default="mtplx-qwen36-27b-native-mtp")
    parser.add_argument("--backend-id", default="qwen3_next", help=argparse.SUPPRESS)
    parser.add_argument(
        "--assistant-model",
        "--gemma-assistant-model",
        dest="assistant_model",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--draft-block-size",
        "--gemma-draft-block-size",
        dest="draft_block_size",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--target-distribution-mode",
        "--gemma-target-distribution-mode",
        dest="target_distribution_mode",
        choices=list(assistant_target_distribution_choices()),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--profile", choices=PROFILE_CHOICES, default=DEFAULT_PROFILE_NAME
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--api-key",
        default=None,
        help="Require Bearer or X-API-Key auth. Required for non-localhost binds.",
    )
    parser.add_argument(
        "--api-key-file",
        help="Read the API key from a local file instead of argv/env.",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=_env_int("MTPLX_RATE_LIMIT_PER_MINUTE", 0),
        help="Requests per minute per client/API key. Use 0 to disable.",
    )
    parser.add_argument(
        "--stream-interval",
        type=int,
        default=_env_int("MTPLX_STREAM_INTERVAL", 1),
        help="Committed-token batch size per chat SSE chunk. Default: 1.",
    )
    parser.add_argument(
        "--scheduler-mode",
        choices=SCHEDULER_MODE_CHOICES,
        default=os.environ.get("MTPLX_SCHEDULER_MODE", "serial"),
        help=(
            "Generation scheduler mode. The live default remains serial so "
            "single-request MTP stays the oracle while batching modes are "
            "brought online behind explicit flags."
        ),
    )
    parser.add_argument(
        "--batching-preset",
        choices=BATCHING_PRESET_CHOICES,
        default=os.environ.get("MTPLX_BATCHING_PRESET", "latency"),
        help="Concurrent batching policy preset: solo, latency, agent, or throughput.",
    )
    parser.add_argument("--max-active-requests", type=int)
    parser.add_argument("--decode-batch-max", type=int)
    parser.add_argument("--batch-wait-ms", type=float)
    parser.add_argument("--prefill-chunk-tokens", type=int)
    parser.add_argument(
        "--experimental-mtp-cohorts",
        action="store_true",
        help="Expose future batched-MTP verify cohorts as experimental; off by default.",
    )
    parser.add_argument(
        "--ssd-session-cache",
        choices=["off", "on", "write-only"],
        default=os.environ.get("MTPLX_SSD_SESSION_CACHE", "off"),
        help="Persistent SessionBank cold tier. Default off for raw serve.",
    )
    parser.add_argument(
        "--ssd-session-cache-dir",
        default=os.environ.get("MTPLX_SSD_SESSION_CACHE_DIR", "~/.mtplx/session-bank"),
        help="Directory for persistent SessionBank snapshots.",
    )
    parser.add_argument(
        "--ssd-session-cache-max-size",
        default=os.environ.get("MTPLX_SSD_SESSION_CACHE_MAX_SIZE", "100GB"),
        help="Soft maximum SSD SessionBank cache size.",
    )
    parser.add_argument(
        "--ssd-session-cache-min-prefix-tokens",
        type=int,
        default=_env_int("MTPLX_SSD_SESSION_CACHE_MIN_PREFIX_TOKENS", 512),
        help="Minimum committed prefix length before writing to the SSD SessionBank cache.",
    )
    parser.add_argument(
        "--paged-kv-quantization",
        "--paged-kv-quant",
        "--kv-quant",
        dest="paged_kv_quantization",
        metavar="{off,q8,q4}",
        default=os.environ.get("MTPLX_VLLM_METAL_PAGED_KV_QUANT")
        or os.environ.get("MTPLX_PAGED_KV_QUANT")
        or "off",
        help="Paged KV cache quantization mode: off, q8, or q4.",
    )
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=_env_int("MTPLX_WARMUP_COUNT", 16),
        help="Startup warmup generation length. Use 0 to disable.",
    )
    parser.add_argument(
        "--strict-warmup",
        action="store_true",
        help="Fail server startup if the warmup pass fails.",
    )
    parser.add_argument(
        "--generation-mode",
        choices=["mtp", "ar"],
        default="mtp",
        help="Generation mode. 'ar' uses target-only AR generation while keeping the same loaded runtime.",
    )
    parser.add_argument(
        "--stock-ar",
        action="store_true",
        help=(
            "Diagnostic only: run target AR without loading the MTP sidecar. "
            "Equivalent to --generation-mode ar --no-load-mtp."
        ),
    )
    parser.add_argument(
        "--load-mtp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load and inject the native MTP sidecar. Disable only for stock AR diagnostics.",
    )
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument(
        "--max-response-tokens",
        "--max-tokens",
        dest="max_response_tokens",
        type=int,
        default=None,
        help="Optional emergency generation cap. Unset means no bridge cap; requests are limited only by remaining context.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=0,
        help="Override context window. Default reads the model/tokenizer config.",
    )
    parser.add_argument(
        "--temperature",
        "--default-temperature",
        dest="temperature",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--top-p", "--default-top-p", dest="top_p", type=float, default=0.95
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--adaptive-policy",
        choices=["none", "streak", "expected_value"],
        default="none",
        help="Optional per-request native-MTP depth policy. Exact sampler semantics remain unchanged.",
    )
    parser.add_argument("--adaptive-min-depth", type=int, default=1)
    parser.add_argument("--adaptive-start-depth", type=int, default=1)
    parser.add_argument("--adaptive-increase-after", type=int, default=4)
    parser.add_argument("--adaptive-decrease-after", type=int, default=1)
    parser.add_argument("--adaptive-ev-base-depth", type=int, default=2)
    parser.add_argument(
        "--adaptive-ev-accept-priors",
        type=_comma_floats,
        default=(0.92, 0.64, 0.32),
    )
    parser.add_argument("--adaptive-ev-draft-cost-s", type=float, default=0.0048)
    parser.add_argument("--adaptive-ev-extra-verify-cost-s", type=float, default=0.0060)
    parser.add_argument("--adaptive-ev-baseline-tok-s", type=float, default=40.0)
    parser.add_argument("--adaptive-ev-safety-margin", type=float, default=0.10)
    parser.add_argument("--adaptive-ev-margin-center", type=float, default=1.0)
    parser.add_argument("--adaptive-ev-margin-scale", type=float, default=2.0)
    parser.add_argument("--adaptive-ev-confidence-weight", type=float, default=0.35)
    parser.add_argument(
        "--adaptive-ev-min-extra-accept-probability",
        type=float,
        default=0.18,
    )
    parser.add_argument("--adaptive-ev-warmup-full-depth-cycles", type=int, default=4)
    parser.add_argument("--adaptive-ev-exploration-interval", type=int, default=32)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional deterministic seed. Omit for fresh per-request interactive sampling.",
    )
    parser.add_argument(
        "--blank-retry-attempts",
        type=int,
        default=3,
        help="Retry empty unseeded generations this many times with fresh seeds.",
    )
    parser.add_argument(
        "--no-stats-footer",
        action="store_false",
        dest="stats_footer",
        help="Do not append the visible MTPLX TPS footer to returned text.",
    )
    parser.add_argument(
        "--session-postcommit-mode",
        choices=["inline", "async"],
        default=postcommit_default,
        help=(
            "SessionBank postcommit policy for streaming fallbacks. The default "
            "'async' uses the generation final state when token-safe and never "
            "holds the stream open for unsafe retokenized prefill work; 'inline' "
            "is a diagnostic mode that preserves the old blocking snapshot."
        ),
    )
    parser.add_argument(
        "--reasoning-mode",
        choices=["auto", "on", "off"],
        default="auto",
        help="Server-default Qwen thinking mode for clients that do not send enable_thinking.",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass enable_thinking to the Qwen chat template for visible <think> reasoning blocks.",
    )
    parser.add_argument(
        "--reasoning",
        choices=["auto", "on", "off"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--reasoning-parser",
        choices=["qwen3", "step3p5", "gemma4", "none"],
        default="qwen3",
        help="Parser for streamed reasoning tags. Use 'none' to stream all text as content.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["auto", "low", "medium", "high"],
        default="auto",
        help="Backend reasoning effort. Step-3.7 Flash maps this to low/medium/high in its chat template.",
    )
    parser.add_argument(
        "--preserve-thinking",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Preserve prior assistant <think>/reasoning blocks in chat-template "
            "history. Default auto preserves them for Qwen reasoning templates."
        ),
    )
    parser.add_argument(
        "--tool-prompt-mode",
        choices=sorted(_TOOL_PROMPT_MODES),
        default=os.environ.get("MTPLX_TOOL_PROMPT_MODE", _TOOL_PROMPT_MODE_HYBRID),
        help=(
            "Tool prompt contract mode. native passes tools only to the model "
            "chat template; hybrid keeps the legacy MTPLX contract for rollback."
        ),
    )
    parser.add_argument(
        "--chat-template-profile",
        choices=sorted(_CHAT_TEMPLATE_PROFILES),
        default=os.environ.get(
            "MTPLX_CHAT_TEMPLATE_PROFILE",
            _CHAT_TEMPLATE_PROFILE_LOCAL,
        ),
        help="Chat template profile to apply at model load.",
    )
    parser.add_argument(
        "--chat-template-path",
        default=os.environ.get("MTPLX_CHAT_TEMPLATE_PATH"),
        help="Explicit chat_template.jinja path for diagnostics.",
    )
    parser.add_argument(
        "--strip-assistant-reasoning-history",
        action="store_true",
        help=("Backward-compatible alias for --preserve-thinking off."),
    )
    parser.add_argument(
        "--normalize-thinking-tags",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Normalize non-stream text with explicit <think> tags. Default keeps "
            "raw generated text so assistant history remains byte/token stable."
        ),
    )
    parser.add_argument("--verify-strategy", default="capture_commit")
    parser.add_argument("--verify-core", default="linear-gdn-from-conv-tape")
    parser.add_argument("--mtp-adapter", default=None)
    parser.add_argument("--merge-mtp-adapter", action="store_true")
    parser.add_argument("--mtp-quant-bits", type=int, default=None)
    parser.add_argument("--mtp-quant-group-size", type=int, default=64)
    parser.add_argument(
        "--mtp-quant-mode",
        choices=["affine", "symmetric"],
        default="affine",
    )
    parser.add_argument(
        "--online-correction-cache",
        action="store_true",
        help=(
            "Enable the exact target-feedback proposal cache. Diagnostic: changes "
            "draft q only; target verification and residual correction remain exact."
        ),
    )
    parser.add_argument("--online-correction-cache-min-depth", type=int, default=1)
    parser.add_argument(
        "--online-correction-cache-key",
        choices=["local_prefix", "source_token", "primary_source"],
        default="local_prefix",
    )
    parser.add_argument(
        "--prompt-correction-cache",
        action="store_true",
        help=(
            "Seed the exact proposal cache from prompt-local n-grams. Diagnostic "
            "only; historically not additive on short-code screens."
        ),
    )
    parser.add_argument("--prompt-correction-cache-min-depth", type=int, default=2)
    parser.add_argument(
        "--online-hidden-corrector-alpha",
        type=float,
        default=0.0,
        help=(
            "Apply an online hidden residual to MTP draft hidden states. "
            "Diagnostic: changes proposal q only; target verification and "
            "residual correction remain authoritative."
        ),
    )
    parser.add_argument("--online-hidden-corrector-decay", type=float, default=0.8)
    parser.add_argument("--online-hidden-corrector-warmup", type=int, default=1)
    parser.add_argument("--online-hidden-corrector-max-feed-depth", type=int)
    parser.add_argument(
        "--online-hidden-corrector-key",
        choices=["global", "token"],
        default="global",
    )
    parser.add_argument("--draft-lm-head-bits", type=int, default=4)
    parser.add_argument("--draft-lm-head-group-size", type=int, default=64)
    parser.add_argument("--draft-lm-head-mode", default="affine")
    parser.add_argument("--draft-temperature", type=float)
    parser.add_argument("--draft-top-p", type=float, default=0.95)
    parser.add_argument("--draft-top-k", type=int, default=20)
    parser.add_argument(
        "--mlx-cache-limit",
        help=(
            "Optional MLX allocator cache limit, e.g. 0, 512MB, 1GB. "
            "Defaults to MTPLX_MLX_CACHE_LIMIT when set."
        ),
    )
    parser.add_argument(
        "--clear-cache-every",
        type=int,
        default=None,
        help=(
            "Diagnostic override for MTPLX_CLEAR_CACHE_EVERY during generation "
            "after profile env is applied. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--strict-startup-asserts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refuse startup unless the selected MTPLX profile env is active after profile setup.",
    )
    parser.add_argument(
        "--diagnostic-env-ablation",
        action="store_true",
        help=(
            "Diagnostic mode: report observed MTPLX profile/fast-path env exactly as launched "
            "without asserting or filling missing flags."
        ),
    )
    parser.add_argument(
        "--strict-mlx-fork-assert",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refuse startup unless the selected profile's required MLX fork is active.",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the local MTPLX browser chat UI after startup.",
    )
    parser.add_argument(
        "--open-dashboard",
        action="store_true",
        help=(
            "Open the live MTPLX dashboard (/dashboard) after startup "
            "instead of the chat UI."
        ),
    )
    parser.add_argument(
        "--enable-thermal-poll",
        action="store_true",
        help=(
            "Enable a 1 Hz background poll of thermal.fan_summary() so the "
            "dashboard's fan panel updates live. Off by default because the "
            "poll shells out to `thermalforge status` (~10-30 ms per tick)."
        ),
    )
    parser.add_argument(
        "--fan-mode",
        choices=FAN_MODE_CHOICES,
        default=normalize_fan_mode(os.environ.get("MTPLX_FAN_MODE") or FAN_MODE_DEFAULT),
        help=(
            "Fan policy: default leaves Apple fan control alone, smart boosts "
            "only while visible requests generate, max reports sustained max mode."
        ),
    )
    parser.add_argument(
        "--app-launch-id",
        default=os.environ.get("MTPLX_APP_LAUNCH_ID"),
        help="Opaque native-app launch id echoed by /health for daemon ownership checks.",
    )
    parser.add_argument(
        "--launch-pi",
        action="store_true",
        help="Open Pi in Terminal after the MTPLX server is ready.",
    )
    parser.add_argument(
        "--launch-opencode",
        action="store_true",
        help="Open OpenCode Desktop after the MTPLX server is ready.",
    )
    parser.add_argument(
        "--launch-hermes",
        action="store_true",
        help="Open Hermes Agent in Terminal after the MTPLX server is ready.",
    )
    parser.add_argument(
        "--server-console",
        action="store_true",
        help="Accept live server-control commands such as /reasoning and /mtp on stdin.",
    )
    parser.add_argument(
        "--pi-launch-command",
        default="",
        help="Pi command to open when --launch-pi is set.",
    )
    parser.add_argument(
        "--hermes-launch-command",
        default="",
        help="Hermes command to open when --launch-hermes is set.",
    )
    args = parser.parse_args(raw_args)
    args._raw_args = list(raw_args)
    args._cli_flags = _explicit_server_flags(raw_args)
    try:
        resolved_key = resolve_api_key(
            explicit_api_key=getattr(args, "api_key", None),
            api_key_file=getattr(args, "api_key_file", None),
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.api_key = resolved_key.value
    args.api_key_source = resolved_key.source
    try:
        args.paged_kv_quantization = normalize_paged_kv_quantization(
            getattr(args, "paged_kv_quantization", "off")
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.fan_mode = normalize_fan_mode(getattr(args, "fan_mode", FAN_MODE_DEFAULT))
    except ValueError as exc:
        parser.error(str(exc))
    _apply_backend_server_defaults(args, explicit_flags=args._cli_flags)
    sync_backend_arg_aliases(args)
    if args.stock_ar:
        args.generation_mode = "ar"
        args.load_mtp = False
    if getattr(args, "reasoning", None) is None:
        args.reasoning = args.reasoning_mode
    args.reasoning = _normalize_reasoning_mode(args.reasoning)
    if args.reasoning == "off":
        args.enable_thinking = False
    elif args.reasoning == "on":
        args.enable_thinking = True
    args.preserve_thinking = _normalize_preserve_thinking_policy(
        "off" if args.strip_assistant_reasoning_history else args.preserve_thinking
    )
    args.strip_assistant_reasoning_history = not _preserve_thinking_effective(args)
    return args


def _start_aime_parent_watchdog_from_env() -> None:
    raw = str(os.environ.get("MTPLX_AIME_PARENT_PID") or "").strip()
    if not raw:
        return
    try:
        parent_pid = int(raw)
    except ValueError:
        return
    if parent_pid <= 1 or parent_pid == os.getpid():
        return

    def watch_parent() -> None:
        while True:
            if os.getppid() == 1:
                os._exit(0)
            try:
                os.kill(parent_pid, 0)
            except OSError as exc:
                if exc.errno == errno.ESRCH:
                    os._exit(0)
            time.sleep(1.0)

    Thread(
        target=watch_parent,
        name="aime-parent-watchdog",
        daemon=True,
    ).start()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_server_security_args(args)
    _start_aime_parent_watchdog_from_env()
    try:
        state = ServerState(args)
    except RuntimeError as exc:
        if str(exc).startswith("Patched MLX qmv fork is not active:"):
            _startup_line("error: fast MLX fork is not active")
            _startup_line(str(exc))
            _startup_line("try: mtplx start --profile sustained")
            _startup_line("try: mtplx start --profile stable")
            _startup_line("try: mtplx start --profile performance-cold --max")
            _startup_line(
                "     (public start disables the strict fork assert when the fork is missing)"
            )
            raise SystemExit(2) from None
        raise
    app = create_app(state)
    import uvicorn

    _startup_line()
    _startup_line("MTPLX is ready.")
    chat_url = _startup_printable_chat_url(args)
    if is_wildcard_bind(getattr(args, "host", None)):
        _startup_line("Listening: " + _startup_bind_label(args))
        _startup_line("Local Chat UI: " + chat_url)
        _startup_line("Local OpenAI API Base URL: " + _startup_openai_base_url(args))
    else:
        _startup_line("Chat UI: " + chat_url)
        _startup_line("OpenAI API Base URL: " + _startup_openai_base_url(args))
    _startup_line("Model: " + str(args.model_id))
    _startup_line(
        "Reasoning history: "
        + ("preserve" if _preserve_thinking_effective(args) else "strip")
        + f" (policy {getattr(args, 'preserve_thinking', 'auto')})"
    )
    if getattr(args, "api_key", None):
        _startup_line("API key: required")
    else:
        _startup_line("API key: leave blank for localhost")
    _startup_line("Health check: " + _startup_server_url(args) + "/health")
    _startup_line("Keep this terminal open. Press Ctrl-C to stop MTPLX.")
    if args.server_console:
        _startup_line("Type /help here for live MTPLX controls.")
        _start_server_console(state)
    # Open chat and dashboard independently: a user with --open-browser
    # AND --open-dashboard gets both tabs (chat first, then dashboard with
    # a small stagger so neither becomes the dropped focus).
    if args.open_browser:
        _startup_line("Opening chat UI in your browser...")
        _open_browser_later(_startup_browser_chat_url(args))
    if getattr(args, "open_dashboard", False):
        _startup_line("Live Dashboard: " + _startup_dashboard_url(args))
        _startup_line("Opening live dashboard in your browser...")
        _open_browser_later(_startup_browser_dashboard_url(args), delay_s=1.6)
    if args.launch_pi:
        command = str(args.pi_launch_command or "").strip()
        if command:
            _startup_line("Opening Pi in Terminal...")
            _open_pi_later(command, model_id=str(args.model_id))
        else:
            _startup_line(
                "warning: --launch-pi was set but no Pi command was provided."
            )
    if args.launch_opencode:
        _startup_line("Opening OpenCode Desktop...")
        _open_opencode_later()
    if args.launch_hermes:
        command = str(args.hermes_launch_command or "").strip()
        if command:
            _startup_line("Opening Hermes Agent in Terminal...")
            _open_hermes_later(command)
        else:
            _startup_line(
                "warning: --launch-hermes was set but no Hermes command was provided."
            )
    uvicorn.run(
        app, host=args.host, port=args.port, log_level="warning", access_log=False
    )


if __name__ == "__main__":
    main()
