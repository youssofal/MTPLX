#!/usr/bin/env python3
"""Serve MTPLX through a minimal OpenAI-compatible API.

This is intentionally small: it exists so local UI clients such as Open WebUI
can exercise the native-MTP runtime without turning MTPLX into a deployment
server yet. The generation path is serialized because the current cache and
GraphBank machinery has not been audited for concurrent requests.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import json
import os
import re
import secrets
import subprocess
import sys
import time
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Timer
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from mtplx.adaptive import AdaptiveDepthPolicy, ExpectedValueDepthPolicy
from mtplx.cache_state import snapshot_cache
from mtplx.mtp_patch import MTPContract
from mtplx.sampling import SamplerConfig
from mtplx.profiles import (
    DEFAULT_PROFILE_NAME,
    PROFILE_CHOICES,
    apply_profile_env,
    get_profile,
    profile_env_status,
)
from mtplx.draft_lm_head import _install_draft_lm_head

try:
    from mtplx.generation import generate_ar, generate_mtpk, restore_or_prefill_prompt_state
    from mtplx.native_mlp import native_mlp_stats
    from mtplx.engine_session import (
        EngineSessionBusy,
        EngineSessionManager,
        hash_text,
        is_background_request,
        system_prompt_hash,
    )
    from mtplx.runtime import load
    from mtplx.session_bank import CacheMissReason
    _RUNTIME_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    _RUNTIME_IMPORT_ERROR = exc

    def _missing_runtime(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(f"MTPLX runtime dependencies are unavailable: {_RUNTIME_IMPORT_ERROR}") from _RUNTIME_IMPORT_ERROR

    generate_ar = _missing_runtime
    generate_mtpk = _missing_runtime
    restore_or_prefill_prompt_state = _missing_runtime
    load = _missing_runtime

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
    "MTPLX_LAZY_MTP_HISTORY_APPEND": "1",
    "MTPLX_DROP_EVENTS": "1",
    "MTPLX_SKIP_VERIFY_SNAPSHOT": "1",
}
EXPECTED_MLX_QMV_FORK_COMMIT = "2377a99f"
EXPECTED_MLX_QMV_FORK_FRAGMENT = "mlx-mtplx-0.31.2-qmm"
STATS_FOOTER_MARKER = "\n---\n⚡ **MTPLX TPS:**"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
_REASONING_DETAILS_RE = re.compile(
    r"<details\b(?=[^>]*\btype=[\"']reasoning[\"'])[^>]*>.*?</details>",
    re.IGNORECASE | re.DOTALL,
)
_REASONING_DETAILS_UNCLOSED_RE = re.compile(
    r"<details\b(?=[^>]*\btype=[\"']reasoning[\"'])[^>]*>.*\Z",
    re.IGNORECASE | re.DOTALL,
)
_REASONING_TAG_RE = re.compile(
    r"<(think|thinking|reason|reasoning|thought)\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)


class _StreamCancelled(RuntimeError):
    """Raised inside the generation worker after the SSE client disconnects."""


def _raise_if_stream_cancelled(cancel_event: Event) -> None:
    if cancel_event.is_set():
        raise _StreamCancelled("stream client disconnected")


def _cancel_stream_generation(cancel_event: Event, generation_future: Any | None) -> None:
    cancel_event.set()
    if generation_future is None:
        return
    try:
        generation_future.cancel()
    except Exception:
        return


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


def _assert_fast_path_env() -> dict[str, dict[str, Any]]:
    status = _fast_path_env_status()
    bad = {key: value for key, value in status.items() if not value["ok"]}
    if bad:
        raise RuntimeError(
            "MTPLX fast-path env is incomplete: "
            + json.dumps(bad, sort_keys=True)
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
        if EXPECTED_MLX_QMV_FORK_FRAGMENT in parent.name or EXPECTED_MLX_QMV_FORK_FRAGMENT in str(parent):
            try:
                commit = subprocess.check_output(
                    ["git", "-C", str(parent), "rev-parse", "--short", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except Exception:
                commit = None
            break
    ok = EXPECTED_MLX_QMV_FORK_FRAGMENT in str(path) and (
        commit in {None, EXPECTED_MLX_QMV_FORK_COMMIT}
    )
    return {
        "ok": ok,
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
    for suffix, multiplier in sorted(multipliers.items(), key=lambda item: -len(item[0])):
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
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    depth: int | None = None
    generation_mode: str | None = None
    seed: int | None = None
    enable_thinking: bool | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    prompt: str | list[int] | list[str] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    depth: int | None = None
    generation_mode: str | None = None
    seed: int | None = None
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
    depth: int | None = None
    generation_mode: str | None = None
    stream: bool = False


def _startup_line(text: str = "") -> None:
    print(text, flush=True)


def _startup_server_url(args: argparse.Namespace) -> str:
    host = str(getattr(args, "host", "127.0.0.1"))
    if host.strip() in {"", "0.0.0.0", "::"}:
        host = "127.0.0.1"
    return f"http://{host}:{int(getattr(args, 'port', 8000))}"


def _startup_openai_base_url(args: argparse.Namespace) -> str:
    return _startup_server_url(args) + "/v1"


def _startup_chat_url(args: argparse.Namespace) -> str:
    return _startup_server_url(args) + "/"


def _open_browser_later(url: str, *, delay_s: float = 1.0) -> None:
    def open_url() -> None:
        try:
            webbrowser.open(url, new=2, autoraise=True)
        except Exception as exc:
            print(f"[mtplx] could not open browser: {exc}", flush=True)

    timer = Timer(delay_s, open_url)
    timer.daemon = True
    timer.start()


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


class ServerState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model_id = args.model_id
        self.lock = Lock()
        self.foreground_lock = Lock()
        self.foreground_active = 0
        self.rate_limiter = _RateLimiter(args.rate_limit)
        self.profile = get_profile(args.profile)
        runtime_label = "Medium MTP" if self.profile.name == "performance-cold" else self.profile.name
        _startup_line(f"[4/6] Preparing {runtime_label} runtime")
        if args.generation_mode == "mtp" and not args.load_mtp:
            raise ValueError("--generation-mode mtp requires --load-mtp")
        if args.diagnostic_env_ablation:
            self.profile_env_status = profile_env_status(self.profile.name)
            self.fast_path_env_status = _fast_path_env_status()
        else:
            apply_profile_env(self.profile.name)
            self.profile_env_status = profile_env_status(self.profile.name)
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
        _startup_line("      This is the long step; MTPLX is mapping the model into MLX.")
        _startup_line("      Model load in progress (this may take a minute).")
        load_heartbeat = _startup_heartbeat("Model still loading")
        try:
            self.runtime = load(args.model, mtp=bool(args.load_mtp), contract=MTPContract())
        except BaseException as exc:
            elapsed_s = time.perf_counter() - started
            _startup_line(f"[5/6] Model load failed after {elapsed_s:.1f}s: {type(exc).__name__}: {exc}")
            raise
        finally:
            load_heartbeat.set()
        self.load_time_s = time.perf_counter() - started
        _startup_line(f"[5/6] Model loaded in {self.load_time_s:.1f}s")
        _startup_line("[5/6] Installing native-MTP draft head")
        self.draft_lm_head = (
            _install_draft_lm_head(
                self.runtime,
                bits=args.draft_lm_head_bits,
                group_size=args.draft_lm_head_group_size,
                mode=args.draft_lm_head_mode,
            )
            if self.runtime.mtp_enabled
            else {"installed": False, "reason": "mtp_disabled"}
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
        self.draft_head_identity = _draft_head_identity(self.runtime)
        self.template_hash = _template_hash(self.runtime.tokenizer)
        self.context_window = (
            int(args.context_window)
            if int(args.context_window) > 0
            else _resolve_context_window(self.runtime.tokenizer, args.model)
        )
        _startup_line(f"[5/6] Context window: {self.context_window} tokens")
        self.sessions = EngineSessionManager()
        self.last_metrics: list[dict[str, Any]] = []
        # Activity timestamps used by the parent-process thermal watchdog to
        # decide when to drop fans back to auto after an idle period.
        self.last_request_started_at: float = 0.0
        self.last_request_at: float = 0.0
        self.requests_completed: int = 0
        self.main_system_prompt_hash: str | None = None
        self.generation_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mtplx-generation",
        )
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


LOCALHOST_BINDS = {"", "127.0.0.1", "::1", "localhost"}


def _is_localhost_bind(host: str | None) -> bool:
    return str(host or "").strip().lower().strip("[]") in LOCALHOST_BINDS


def validate_server_security_args(args: argparse.Namespace) -> None:
    if not _is_localhost_bind(getattr(args, "host", None)) and not getattr(args, "api_key", None):
        raise SystemExit("--api-key is required when --host is not localhost")
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
    return api_key or None


def _request_is_authorized(request: Request, configured_api_key: str | None) -> bool:
    if not configured_api_key:
        return True
    candidate = _request_api_key(request)
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


def _anthropic_to_chat_request(request: AnthropicMessagesRequest) -> ChatCompletionRequest:
    messages: list[ChatMessage] = []
    system_text = _anthropic_content_to_text(request.system).strip()
    if system_text:
        messages.append(ChatMessage(role="system", content=system_text))
    for message in request.messages:
        role = "assistant" if message.role == "assistant" else "user"
        messages.append(
            ChatMessage(
                role=role,
                content=_anthropic_content_to_text(message.content),
            )
        )
    return ChatCompletionRequest(
        model=request.model,
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        depth=request.depth,
        generation_mode=request.generation_mode,
        stream=False,
    )


def _anthropic_payload_from_openai(openai_payload: dict[str, Any]) -> dict[str, Any]:
    choices = openai_payload.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    text = str(message.get("content") or choice.get("text") or "")
    usage = openai_payload.get("usage") or {}
    finish_reason = choice.get("finish_reason") or "stop"
    stop_reason = "max_tokens" if finish_reason == "length" else "end_turn"
    return {
        "id": "msg_" + uuid.uuid4().hex,
        "type": "message",
        "role": "assistant",
        "model": openai_payload.get("model"),
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
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
        chunk = raw_chunk.decode("utf-8") if isinstance(raw_chunk, bytes) else str(raw_chunk)
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
    content_started = False
    stop_reason = "end_turn"
    usage = {"input_tokens": 0, "output_tokens": 0}
    mtplx_stats: dict[str, Any] | None = None

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

    def start_content_block() -> str:
        return _anthropic_sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
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
                text = ""
                if delta.get("reasoning_content"):
                    text += str(delta.get("reasoning_content") or "")
                if delta.get("content"):
                    text += str(delta.get("content") or "")
                if text:
                    if not content_started:
                        content_started = True
                        yield start_content_block()
                    yield _anthropic_sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": text},
                        },
                    )
                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    stop_reason = "max_tokens" if finish_reason == "length" else "end_turn"
    finally:
        if hasattr(body_iterator, "aclose"):
            try:
                await body_iterator.aclose()
            except Exception:
                pass

    if not content_started:
        yield start_content_block()
    yield _anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    delta_payload: dict[str, Any] = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
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
    block = re.sub(r"<summary\b[^>]*>.*?</summary>", "", block, flags=re.IGNORECASE | re.DOTALL)
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
_TOOL_FUNCTION_BLOCK_RE = re.compile(
    r"^\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*$",
    re.IGNORECASE | re.DOTALL,
)
_TOOL_PARAMETER_BLOCK_RE = re.compile(
    r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>",
    re.IGNORECASE | re.DOTALL,
)


def _tool_protocol_error(message: str) -> HTTPException:
    return HTTPException(status_code=422, detail=f"malformed tool_call: {message}")


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


def _normalize_tool_specs(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not tools:
        return []
    normalized: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise HTTPException(status_code=400, detail=f"tools[{index}] must be an object")
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


def _json_object_string(value: Any, *, context: str) -> str:
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
                raise _tool_protocol_error(f"{context} arguments are not valid JSON") from exc
    else:
        parsed = value
    if not isinstance(parsed, dict):
        raise _tool_protocol_error(f"{context} arguments must be a JSON object")
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def _decode_tool_parameter_value(value: str) -> Any:
    text = value.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _parse_json_tool_call(block: str) -> tuple[str, Any] | None:
    try:
        payload = json.loads(block)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        raise _tool_protocol_error("JSON tool_call payload must be an object")
    function = payload.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        name = payload.get("name")
        arguments = payload.get("arguments", {})
    name_text = str(name or "").strip()
    if not name_text:
        raise _tool_protocol_error("JSON tool_call is missing a function name")
    return name_text, arguments


def _parse_xml_tool_call(block: str) -> tuple[str, Any] | None:
    match = _TOOL_FUNCTION_BLOCK_RE.match(block)
    if match is None:
        return None
    name = match.group(1).strip()
    body = match.group(2)
    arguments: dict[str, Any] = {}
    consumed: list[tuple[int, int]] = []
    for param_match in _TOOL_PARAMETER_BLOCK_RE.finditer(body):
        param_name = param_match.group(1).strip()
        if not param_name:
            raise _tool_protocol_error(f"tool '{name}' contains an empty parameter name")
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
            raise _tool_protocol_error(f"tool '{name}' contains text outside parameters")
    elif body.strip():
        raise _tool_protocol_error(f"tool '{name}' contains unwrapped parameter text")
    return name, arguments


def _parse_generated_tool_calls(
    text: str,
    *,
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    lowered = text.lower()
    if "<tool_call" not in lowered and "</tool_call>" not in lowered:
        return None
    blocks = list(_TOOL_CALL_BLOCK_RE.finditer(text))
    if not blocks:
        raise _tool_protocol_error("unclosed <tool_call> block")
    residue = _TOOL_CALL_BLOCK_RE.sub("", text)
    if "<tool_call" in residue.lower() or "</tool_call>" in residue.lower():
        raise _tool_protocol_error("nested or unmatched <tool_call> block")
    known = {name for tool in tools if (name := _tool_spec_name(tool))}
    calls: list[dict[str, Any]] = []
    for index, block_match in enumerate(blocks):
        block = block_match.group(1).strip()
        parsed = _parse_json_tool_call(block)
        if parsed is None:
            parsed = _parse_xml_tool_call(block)
        if parsed is None:
            raise _tool_protocol_error("unsupported tool_call payload format")
        name, arguments = parsed
        if name not in known:
            raise _tool_protocol_error(f"unknown tool '{name}'")
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": _json_object_string(
                        arguments,
                        context=f"tool_call[{index}]",
                    ),
                },
            }
        )
    return calls


def _template_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function")
    if isinstance(function, dict):
        name = str(function.get("name") or "").strip()
        arguments = function.get("arguments", {})
    else:
        name = str(tool_call.get("name") or "").strip()
        arguments = tool_call.get("arguments", {})
    if not name:
        raise HTTPException(status_code=400, detail="assistant tool_call is missing a name")
    arguments_text = _json_object_string(arguments, context=f"assistant tool_call '{name}'")
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
            else _normalize_openwebui_reasoning_details(_strip_stats_footer(content)).strip()
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


def _encode_messages(
    tokenizer: Any,
    messages: list[ChatMessage],
    *,
    enable_thinking: bool,
    strip_assistant_reasoning_history: bool = False,
    add_generation_prompt: bool = True,
    tools: list[dict[str, Any]] | None = None,
) -> list[int]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        item = _message_to_template_dict(
            message,
            strip_assistant_reasoning_history=strip_assistant_reasoning_history,
        )
        if item is not None:
            normalized.append(item)
    if not normalized:
        normalized = [{"role": "user", "content": ""}]
    template_kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
        "preserve_thinking": not strip_assistant_reasoning_history,
    }
    if tools:
        template_kwargs["tools"] = tools
    try:
        return list(
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
            if tools:
                fallback_kwargs["tools"] = tools
            return list(
                tokenizer.apply_chat_template(
                    normalized,
                    **fallback_kwargs,
                )
            )
        except TypeError as exc:
            if tools:
                raise HTTPException(
                    status_code=500,
                    detail="tokenizer chat template does not support tool schemas",
                ) from exc
        except Exception as exc:
            if tools:
                raise HTTPException(
                    status_code=500,
                    detail=f"tokenizer chat template failed with tool schemas: {exc}",
                ) from exc
            pass
    except Exception as exc:
        if tools:
            raise HTTPException(
                status_code=500,
                detail=f"tokenizer chat template failed with tool schemas: {exc}",
            ) from exc
        pass
    prompt = "\n".join(f"{item['role']}: {item['content']}" for item in normalized)
    if add_generation_prompt:
        prompt += "\nassistant:"
    return list(tokenizer.encode(prompt))


def _encode_prompt(tokenizer: Any, prompt: str | list[int] | list[str] | None) -> list[int]:
    if prompt is None:
        return []
    if isinstance(prompt, str):
        return list(tokenizer.encode(prompt))
    if isinstance(prompt, list) and all(isinstance(item, int) for item in prompt):
        return [int(item) for item in prompt]
    if isinstance(prompt, list):
        return list(tokenizer.encode("\n".join(str(item) for item in prompt)))
    return list(tokenizer.encode(str(prompt)))


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
    extra = getattr(model, "model_extra", None) or {}
    return extra.get(key, default)


def _request_metadata(model: BaseModel) -> dict[str, Any]:
    metadata = _request_extra(model, "metadata", {})
    return metadata if isinstance(metadata, dict) else {}


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


def _request_generation_mode_for_generation(state: ServerState, request: BaseModel) -> str:
    default = _normalize_generation_mode(getattr(state.args, "generation_mode", "mtp"))
    mode = _normalize_generation_mode(_request_generation_mode_value(request), default=default)
    if mode == "mtp" and not bool(getattr(state.runtime, "mtp_enabled", False)):
        raise HTTPException(
            status_code=400,
            detail="generation_mode 'mtp' requires a runtime loaded with MTP",
        )
    return mode


def _request_depth_value(request: BaseModel) -> Any:
    for key in ("depth", "mtp_depth", "speculative_depth"):
        value = getattr(request, key, None)
        if value is None:
            value = _request_extra(request, key)
        if value is not None:
            return value
    return None


def _request_depth_for_generation(
    state: ServerState,
    request: BaseModel,
    *,
    generation_mode: str,
) -> int:
    if generation_mode == "ar":
        return 0
    value = _request_depth_value(request)
    if value is None:
        return int(getattr(state.args, "depth", 3))
    try:
        depth = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="depth must be an integer") from exc
    if depth < 1 or depth > 3:
        raise HTTPException(status_code=400, detail="depth must be between 1 and 3")
    return depth


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
    prompt_eval_time_s = float(stats.get("prompt_eval_time_s") or 0.0)
    ttft_s = (
        max(0.0, token_times[0] - request_started_s)
        if token_times
        else None
    )
    cached_tokens = int(stats.get("cached_tokens") or 0)
    new_prefill_tokens = int(stats.get("new_prefill_tokens") or (prompt_tokens - cached_tokens))
    prefill_tok_s = (
        max(0, new_prefill_tokens) / prompt_eval_time_s
        if prompt_eval_time_s > 0
        else None
    )
    return {
        "prompt_tokens": int(prompt_tokens),
        "cached_tokens": cached_tokens,
        "new_prefill_tokens": max(0, new_prefill_tokens),
        "completion_tokens": int(completion_tokens),
        "prompt_eval_time_s": prompt_eval_time_s,
        "prefill_tok_s": prefill_tok_s,
        "ttft_s": ttft_s,
        "decode_elapsed_s": decode_elapsed_s,
        "request_elapsed_s": request_elapsed_s,
        "request_tok_s": completion_tokens / request_elapsed_s if request_elapsed_s > 0 else 0.0,
        "decode_tok_s": decode_tok_s,
        "sliding_decode_tok_s_first_32": _token_window_rate_first(token_times, 32),
        "sliding_decode_tok_s_first_64": _token_window_rate_first(token_times, 64),
        "sliding_decode_tok_s_last_32": _token_window_rate(token_times, 32),
        "sliding_decode_tok_s_last_64": _token_window_rate(token_times, 64),
        "mtp_depth": int(mtp_depth),
        "verify_calls": int(stats.get("verify_calls") or 0),
        "accepted_by_depth": stats.get("accepted_by_depth") or [],
        "correction_tokens": int(stats.get("correction_tokens") or 0),
        "bonus_tokens": int(stats.get("bonus_tokens") or 0),
        "verify_time_s": float(stats.get("verify_time_s") or 0.0),
        "draft_time_s": float(stats.get("draft_time_s") or 0.0),
        "accept_time_s": float(stats.get("accept_time_s") or 0.0),
        "repair_time_s": float(stats.get("repair_time_s") or 0.0),
        "session_cache_hit": bool(session_cache_hit),
        "cache_miss_reason": cache_miss_reason,
        "session_restore_mode": session_restore_mode,
        "context_len": int(prompt_tokens + completion_tokens),
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
        repaired["generated_tokens_raw"] = raw_generated
        repaired["generated_tokens_recovered_from_stream"] = True
        repaired["generated_tokens"] = int(completion_tokens)
        repaired["tok_s"] = (
            float(completion_tokens) / elapsed_s if elapsed_s > 0 else 0.0
        )
    return repaired


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


PUBLIC_MTPLX_STATS_KEYS = (
    "mode",
    "generation_mode",
    "generated_tokens",
    "prompt_tokens",
    "completion_tokens",
    "elapsed_s",
    "tok_s",
    "prompt_eval_time_s",
    "prefill_tok_s",
    "ttft_s",
    "decode_elapsed_s",
    "request_elapsed_s",
    "request_tok_s",
    "decode_tok_s",
    "sliding_decode_tok_s_first_32",
    "sliding_decode_tok_s_first_64",
    "sliding_decode_tok_s_last_32",
    "sliding_decode_tok_s_last_64",
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
    "session_cache_hit",
    "cached_tokens",
    "new_prefill_tokens",
    "cache_miss_reason",
    "session_restore_mode",
    "session_id",
    "context_len",
    "lock_wait_time_s",
    "request_max_tokens",
    "server_max_response_tokens",
    "effective_max_tokens",
    "remaining_context_tokens",
    "server_cap_applied",
    "context_cap_applied",
    "server_elapsed_s",
    "server_tok_s",
    "server_seed",
    "server_attempts",
    "server_blank_retries",
    "server_blank_retry_suppressed",
    "mtp_depth",
    "speculative_depth",
    "peak_memory_bytes",
    "reasoning_reentries",
    "reasoning_tokens",
    "answer_tokens",
)
PUBLIC_POSTCOMMIT_KEYS = (
    "stored",
    "mode",
    "reason",
    "prefix_len",
    "nbytes",
    "elapsed_s",
    "error",
)


def _public_mtplx_stats(generated: dict[str, Any]) -> dict[str, Any]:
    stats = generated.get("stats") or {}
    public = {
        key: stats[key]
        for key in PUBLIC_MTPLX_STATS_KEYS
        if key in stats
    }
    postcommit = stats.get("session_postcommit_snapshot")
    if isinstance(postcommit, dict):
        public["session_postcommit_snapshot"] = {
            key: postcommit[key]
            for key in PUBLIC_POSTCOMMIT_KEYS
            if key in postcommit
        }
    return _json_safe(public)


def _request_observability(
    request: ChatCompletionRequest,
    *,
    headers: dict[str, str],
    metadata: dict[str, Any],
    session_source: str | None,
    request_generation_mode: str,
    request_depth: int,
) -> dict[str, Any]:
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
        }
    }
    return {
        "request_message_count": len(request.messages),
        "request_message_roles": [message.role for message in request.messages],
        "request_message_chars": [
            len(_content_to_text(message.content)) for message in request.messages
        ],
        "request_extra_keys": sorted((getattr(request, "model_extra", None) or {}).keys()),
        "request_metadata_keys": sorted(metadata.keys()),
        "request_session_source": session_source,
        "request_session_candidate_headers": candidate_headers,
        "request_generation_mode": request_generation_mode,
        "request_depth": int(request_depth),
        "request_last_user_preview": user_texts[-1][:180] if user_texts else None,
        "request_last_user_chars": len(user_texts[-1]) if user_texts else 0,
    }


def _policy_fingerprint(
    state: ServerState,
    *,
    thinking_enabled: bool,
    generation_mode: str | None = None,
    depth: int | None = None,
) -> str:
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
    return ";".join(
        [
            f"template={state.template_hash}",
            f"thinking={int(bool(thinking_enabled))}",
            f"strip_reasoning={int(bool(state.args.strip_assistant_reasoning_history))}",
            f"generation_mode={effective_mode}",
            f"depth={effective_depth}",
            "hidden_variant=post_norm",
            "mtp_history_policy=committed",
            f"draft_head={state.draft_head_identity}",
            f"adaptive={json.dumps(adaptive, sort_keys=True, separators=(',', ':'))}",
            f"proposal_cache={json.dumps(proposal_cache, sort_keys=True, separators=(',', ':'))}",
            f"online_hidden={json.dumps(online_hidden, sort_keys=True, separators=(',', ':'))}",
        ]
    )


def _proposal_cache_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "online_correction_cache": bool(args.online_correction_cache),
        "online_correction_cache_min_depth": int(args.online_correction_cache_min_depth),
        "online_correction_cache_key": str(args.online_correction_cache_key),
        "prompt_correction_cache": bool(args.prompt_correction_cache),
        "prompt_correction_cache_min_depth": int(args.prompt_correction_cache_min_depth),
    }


def _online_hidden_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "alpha": float(getattr(args, "online_hidden_corrector_alpha", 0.0)),
        "decay": float(getattr(args, "online_hidden_corrector_decay", 0.8)),
        "warmup": int(getattr(args, "online_hidden_corrector_warmup", 1)),
        "max_feed_depth": getattr(args, "online_hidden_corrector_max_feed_depth", None),
        "key": str(getattr(args, "online_hidden_corrector_key", "global")),
    }


def _adaptive_config(
    args: argparse.Namespace,
    *,
    max_depth: int | None = None,
) -> dict[str, Any]:
    policy = str(getattr(args, "adaptive_policy", "none") or "none")
    if policy == "none":
        return {"policy": "none"}
    effective_max_depth = int(max_depth if max_depth is not None else getattr(args, "depth", 3))
    config: dict[str, Any] = {
        "policy": policy,
        "max_depth": effective_max_depth,
        "min_depth": int(args.adaptive_min_depth),
    }
    if policy == "streak":
        config.update(
            {
                "start_depth": int(args.adaptive_start_depth),
                "increase_after": int(args.adaptive_increase_after),
                "decrease_after": int(args.adaptive_decrease_after),
            }
        )
    elif policy == "expected_value":
        config.update(
            {
                "base_depth": int(args.adaptive_ev_base_depth),
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
            }
        )
    return config


def _make_adaptive_policy(
    args: argparse.Namespace,
    *,
    max_depth: int | None = None,
) -> AdaptiveDepthPolicy | ExpectedValueDepthPolicy | None:
    policy = str(getattr(args, "adaptive_policy", "none") or "none")
    if policy == "none":
        return None
    effective_max_depth = int(max_depth if max_depth is not None else getattr(args, "depth", 3))
    if policy == "streak":
        return AdaptiveDepthPolicy(
            max_depth=effective_max_depth,
            min_depth=int(args.adaptive_min_depth),
            start_depth=int(args.adaptive_start_depth),
            increase_after=int(args.adaptive_increase_after),
            decrease_after=int(args.adaptive_decrease_after),
        )
    if policy == "expected_value":
        return ExpectedValueDepthPolicy(
            max_depth=effective_max_depth,
            min_depth=int(args.adaptive_min_depth),
            base_depth=int(args.adaptive_ev_base_depth),
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
) -> dict[str, Any]:
    if session_id is None:
        return {"stored": False, "reason": "no_session_id"}
    history_messages = list(messages) + [
        ChatMessage(
            role="assistant",
            content=assistant_content,
            tool_calls=assistant_tool_calls,
        ),
    ]
    encoded_with_sentinel = _encode_messages(
        state.runtime.tokenizer,
        history_messages,
        enable_thinking=thinking_enabled,
        strip_assistant_reasoning_history=state.args.strip_assistant_reasoning_history,
        add_generation_prompt=False,
    )
    history_ids = encoded_with_sentinel
    if not history_ids:
        return {"stored": False, "reason": "empty_boundary_prefix"}
    started = time.perf_counter()
    state.begin_foreground()
    state.lock.acquire()
    try:
        prompt_state = restore_or_prefill_prompt_state(
            state.runtime,
            history_ids,
            mtp_hidden_variant="post_norm",
            mtp_history_policy="committed",
        )
        mtp_snapshot = (
            snapshot_cache(prompt_state.committed_mtp_cache)
            if prompt_state.committed_mtp_cache is not None
            else None
        )
        entry = state.sessions.bank.put(
            runtime=state.runtime,
            token_ids=history_ids,
            cache=prompt_state.trunk_cache,
            logits=prompt_state.logits,
            hidden=prompt_state.hidden,
            hidden_variant="post_norm",
            keep_live_ref=True,
            session_id=session_id,
            template_hash=state.template_hash,
            mtp_history_policy="committed",
            draft_head_identity=state.draft_head_identity,
            policy_fingerprint=policy_fingerprint,
            mtp_history_snapshot=mtp_snapshot,
            snapshot_epoch=len(history_ids),
            mtp_snapshot_epoch=len(history_ids) if mtp_snapshot is not None else None,
        )
    finally:
        state.lock.release()
        state.end_foreground()
    return {
        "stored": True,
        "prefix_len": entry.prefix_len,
        "nbytes": entry.nbytes,
        "elapsed_s": time.perf_counter() - started,
        "token_hash": entry.token_hash,
    }


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
    requested_max = remaining_context if request_max_tokens is None else request_max_tokens
    before_server_cap = requested_max
    server_max_response_tokens = state.args.max_response_tokens
    if state.args.max_response_tokens is not None:
        requested_max = min(requested_max, int(state.args.max_response_tokens))
    after_server_cap = requested_max
    requested_max = max(1, min(after_server_cap, remaining_context))
    sampler = SamplerConfig(
        temperature=state.args.temperature if temperature is None else float(temperature),
        top_p=state.args.top_p if top_p is None else float(top_p),
        top_k=state.args.top_k if top_k is None else int(top_k),
    )
    return requested_max, sampler, {
        "request_max_tokens": request_max_tokens,
        "server_max_response_tokens": (
            None if server_max_response_tokens is None else int(server_max_response_tokens)
        ),
        "effective_max_tokens": int(requested_max),
        "remaining_context_tokens": int(remaining_context),
        "server_cap_applied": bool(
            server_max_response_tokens is not None and after_server_cap < before_server_cap
        ),
        "context_cap_applied": bool(requested_max < after_server_cap),
    }


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


def _run_generation(
    state: ServerState,
    prompt_ids: list[int],
    *,
    max_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    seed: int | None,
    generation_mode: str | None = None,
    depth: int | None = None,
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
    request_observability: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response_max, sampler, generation_limits = _generation_params(
        state,
        prompt_token_count=len(prompt_ids),
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
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
    started = time.perf_counter()
    token_times: list[float] = []
    lock_wait_time_s = 0.0

    def record_tokens(new_tokens: list[int]) -> None:
        now = time.perf_counter()
        token_times.extend([now for _token in new_tokens])
        if token_callback is not None:
            token_callback(new_tokens)

    blank_retry_budget = max(0, int(state.args.blank_retry_attempts))
    streaming_response = token_callback is not None
    max_attempts = 1 if streaming_response else 1 + blank_retry_budget
    last: dict[str, Any] | None = None
    trace_preview = (
        str((request_observability or {}).get("request_last_user_preview") or "")
        .replace("\n", " ")
        .strip()
    )
    trace_label = f"{session_id or 'stateless'}:{trace_preview[:64]}" if trace_preview else session_id
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
            state.begin_foreground()
            state.lock.acquire()
        lock_wait_time_s += time.perf_counter() - lock_started
        try:
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
                )
            else:
                adaptive_policy = _make_adaptive_policy(state.args, max_depth=effective_depth)
                out = generate_mtpk(
                    state.runtime,
                    prompt_ids,
                    max_tokens=response_max,
                    sampler=sampler,
                    draft_sampler=state.draft_sampler,
                    speculative_depth=effective_depth,
                    seed=generation_seed,
                    mtp_hidden_variant="post_norm",
                    mtp_cache_policy="persistent",
                    mtp_history_policy="committed",
                    verify_strategy=state.args.verify_strategy,
                    verify_core=state.args.verify_core,
                    token_callback=record_tokens,
                    session_bank=session_bank,
                    session_restore_mode=_session_bank_restore_mode(session_restore_mode),
                    session_template_hash=session_template_hash,
                    session_draft_head_identity=session_draft_head_identity,
                    session_policy_fingerprint=session_policy_fingerprint,
                    capture_final_state=session_bank is not None,
                    trace_label=trace_label,
                    trace_metadata=trace_metadata,
                    adaptive_policy=adaptive_policy,
                    online_correction_cache=bool(state.args.online_correction_cache),
                    online_correction_cache_min_depth=int(
                        state.args.online_correction_cache_min_depth
                    ),
                    online_correction_cache_key=str(state.args.online_correction_cache_key),
                    prompt_correction_cache=bool(state.args.prompt_correction_cache),
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
                    online_hidden_corrector_key=str(state.args.online_hidden_corrector_key),
                )
        finally:
            state.lock.release()
            if not background_request:
                state.end_foreground()
        elapsed_s = time.perf_counter() - started
        completion_tokens = _effective_completion_tokens(
            generated_tokens=list(out.tokens),
            streamed_token_times=token_times,
        )
        tok_s = completion_tokens / elapsed_s if elapsed_s > 0 else 0.0
        stats = _repair_streamed_generation_stats(
            out.stats.to_dict(),
            completion_tokens=completion_tokens,
            elapsed_s=elapsed_s,
        )
        if session_bank is not None:
            cache_miss_reason = stats.get("cache_miss_reason") or cache_miss_reason
            session_cache_hit = bool(stats.get("session_cache_hit") or False)
            if session_cache_hit:
                cache_miss_reason = None
            session_restore_mode = str(stats.get("session_restore_mode") or session_restore_mode)
        final_state = out.final_state
        if (
            commit_final_state_to_bank
            and
            session_bank is not None
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
                keep_live_ref=True,
                session_id=session_id,
                template_hash=session_template_hash,
                mtp_history_policy="committed",
                draft_head_identity=session_draft_head_identity,
                policy_fingerprint=session_policy_fingerprint,
                mtp_history_snapshot=mtp_snapshot,
                snapshot_epoch=len(final_token_ids),
                mtp_snapshot_epoch=len(final_token_ids) if mtp_snapshot is not None else None,
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
            mtp_depth=effective_depth if effective_mode == "mtp" else 0,
            generation_limits=generation_limits,
        )
        envelope["generation_mode"] = effective_mode
        if effective_mode == "ar":
            envelope["mtp_depth"] = 0
            envelope["verify_calls"] = 0
            envelope["verify_time_s"] = 0.0
            envelope["accepted_by_depth"] = []
            envelope["draft_time_s"] = 0.0
        if request_observability:
            envelope.update(request_observability)
        stats["generation_mode"] = effective_mode
        stats.update(envelope)
        if effective_mode == "ar":
            stats["mtp_depth"] = 0
            stats["verify_calls"] = 0
            stats["verify_time_s"] = 0.0
            stats["accepted_by_depth"] = []
            stats["drafted_by_depth"] = []
            stats["mean_accept_probability_by_depth"] = []
            stats["draft_time_s"] = 0.0
        stats["server_elapsed_s"] = elapsed_s
        stats["server_tok_s"] = tok_s
        stats["server_seed"] = generation_seed
        stats["server_attempts"] = attempt + 1
        stats["server_blank_retries"] = attempt
        stats["server_blank_retry_suppressed"] = bool(streaming_response and blank_retry_budget)
        state.last_metrics.append(dict(envelope))
        state.last_metrics = state.last_metrics[-100:]
        state.last_request_at = time.time()
        state.requests_completed += 1
        last = {
            "text": out.text,
            "tokens": out.tokens,
            "stats": _json_safe(stats),
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": completion_tokens,
            "elapsed_s": elapsed_s,
            "tok_s": tok_s,
            "finish_reason": (
                out.final_state.finish_reason
                if out.final_state is not None
                else "stop"
            ),
        }
        if seed_is_explicit or out.text.strip():
            break
    assert last is not None
    if not bool((request_observability or {}).get("warmup")):
        print(
            json.dumps(
                {
                    "event": "mtplx_openai_generation",
                    "prompt_tokens": last["prompt_tokens"],
                    "completion_tokens": last["completion_tokens"],
                    "elapsed_s": round(float(last["elapsed_s"]), 6),
                    "tok_s": round(float(last["tok_s"]), 6),
                    "seed": last["stats"].get("server_seed"),
                    "attempts": last["stats"].get("server_attempts"),
                    "blank_retries": last["stats"].get("server_blank_retries"),
                    "text_preview": str(last["text"])[:120],
                },
                ensure_ascii=False,
            ),
            flush=True,
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
        generated = _run_generation(
            state,
            prompt_ids,
            max_tokens=warmup_tokens,
            temperature=state.args.temperature,
            top_p=state.args.top_p,
            top_k=state.args.top_k,
            seed=0,
            request_observability={"warmup": True},
        )
    except BaseException as exc:
        status.update(
            {
                "elapsed_s": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        warmup_heartbeat.set()
        _startup_line(f"[6/6] Warmup failed after {status['elapsed_s']:.1f}s: {status['error']}")
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


def _chunk_text(text: str, chunk_chars: int = 24) -> list[str]:
    if not text:
        return [""]
    return [text[index : index + chunk_chars] for index in range(0, len(text), chunk_chars)]


def _decode_timing(stats: dict[str, Any]) -> tuple[float, float]:
    generated_tokens = int(stats.get("generated_tokens") or 0)
    elapsed_s = float(stats.get("elapsed_s") or 0.0)
    if "prompt_eval_time_s" in stats:
        prompt_eval_time_s = float(stats.get("prompt_eval_time_s") or 0.0)
    else:
        target_forward_time_s = float(stats.get("target_forward_time_s") or 0.0)
        verify_time_s = float(stats.get("verify_time_s") or 0.0)
        repair_time_s = float(stats.get("repair_time_s") or 0.0)
        prompt_eval_time_s = max(0.0, target_forward_time_s - verify_time_s - repair_time_s)
    decode_elapsed_s = max(0.0, elapsed_s - prompt_eval_time_s)
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


def _split_thinking_segments(text: str, *, thinking_enabled: bool) -> tuple[str, str]:
    if not thinking_enabled:
        return "", text
    reasoning_parts: list[str] = []
    content_parts: list[str] = []

    def append_reasoning(segment: str) -> None:
        if not segment:
            return
        if reasoning_parts and not reasoning_parts[-1].endswith(("\n", " ")) and not segment.startswith(("\n", " ")):
            reasoning_parts.append("\n")
        reasoning_parts.append(segment)

    position = 0
    inside_thinking = True
    while position < len(text):
        if inside_thinking:
            close_index = text.find(THINK_CLOSE, position)
            if close_index < 0:
                segment = text[position:].replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
                append_reasoning(segment)
                break
            segment = text[position:close_index].replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
            append_reasoning(segment)
            position = close_index + len(THINK_CLOSE)
            inside_thinking = False
            continue

        open_index = text.find(THINK_OPEN, position)
        if open_index < 0:
            segment = text[position:].replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
            if segment:
                content_parts.append(segment)
            break
        segment = text[position:open_index].replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
        if segment:
            content_parts.append(segment)
        position = open_index + len(THINK_OPEN)
        inside_thinking = True
    return "".join(reasoning_parts).strip(), "".join(content_parts).strip()


def _normalize_thinking_tags(text: str, *, thinking_enabled: bool) -> str:
    """Make Qwen thinking output parseable by Open WebUI's native tag handler.

    Qwen's chat template can put the opening <think> tag in the prompt, so the
    generated text may begin inside the reasoning block and only emit </think>.
    Open WebUI's built-in parser needs a start tag in the streamed content.
    """
    reasoning, content = _split_thinking_segments(text, thinking_enabled=thinking_enabled)
    if not thinking_enabled:
        return content
    pieces: list[str] = []
    if reasoning:
        pieces.append(f"{THINK_OPEN}\n{reasoning}\n{THINK_CLOSE}")
    if content:
        pieces.append(content)
    return "\n\n".join(pieces)


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
        close_index = text.find(THINK_CLOSE, self._print_len)
        if close_index >= 0:
            boundary = close_index + len(THINK_CLOSE)
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
        if THINK_CLOSE in joined:
            return self.finish()
        return ""


class _ThinkingContentStreamSplitter:
    def __init__(self, *, thinking_enabled: bool) -> None:
        self._thinking_enabled = thinking_enabled
        self._inside_thinking = thinking_enabled
        self._pending = ""
        self._reentry_count = 0

    @property
    def reentry_count(self) -> int:
        return self._reentry_count

    def start(self) -> list[tuple[str, str]]:
        return []

    def feed(self, text: str) -> list[tuple[str, str]]:
        if not text:
            return []
        if not self._thinking_enabled:
            return [("content", text)]
        self._pending += text
        return self._drain(final=False)

    def finish(self) -> list[tuple[str, str]]:
        chunks = self._drain(final=True)
        self._inside_thinking = False
        return chunks

    def _append_chunk(
        self,
        chunks: list[tuple[str, str]],
        field: str,
        text: str,
    ) -> None:
        cleaned = text.replace(THINK_OPEN, "").replace(THINK_CLOSE, "")
        if cleaned:
            chunks.append((field, cleaned))

    def _drain(self, *, final: bool) -> list[tuple[str, str]]:
        chunks: list[tuple[str, str]] = []
        keep = max(len(THINK_OPEN), len(THINK_CLOSE)) - 1
        while self._pending:
            if self._inside_thinking:
                close_index = self._pending.find(THINK_CLOSE)
                if close_index < 0:
                    emit_len = len(self._pending) if final else max(0, len(self._pending) - keep)
                    if emit_len <= 0:
                        break
                    self._append_chunk(chunks, "reasoning_content", self._pending[:emit_len])
                    self._pending = self._pending[emit_len:]
                    break
                self._append_chunk(chunks, "reasoning_content", self._pending[:close_index])
                self._pending = self._pending[close_index + len(THINK_CLOSE) :].lstrip()
                self._inside_thinking = False
                continue

            open_index = self._pending.find(THINK_OPEN)
            if open_index < 0:
                emit_len = len(self._pending) if final else max(0, len(self._pending) - keep)
                if emit_len <= 0:
                    break
                self._append_chunk(chunks, "content", self._pending[:emit_len])
                self._pending = self._pending[emit_len:]
                break
            self._append_chunk(chunks, "content", self._pending[:open_index])
            self._pending = self._pending[open_index + len(THINK_OPEN) :]
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


def _display_text(
    state: ServerState,
    generated: dict[str, Any],
    *,
    thinking_enabled: bool = False,
) -> str:
    raw_text = str(generated["text"])
    text = (
        _normalize_thinking_tags(
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


def _chat_ui_html(
    *,
    model_id: str,
    server_url: str,
    api_key_required: bool,
    default_settings: dict[str, Any],
) -> str:
    api_note = "API key required" if api_key_required else "local · no API key"
    default_depth = max(1, min(3, int(default_settings.get("depth", 3))))
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
    .switch-row label { margin: 0; }
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
            <div class="answer"><p>Ready when you are. Settings on the left persist between sessions.</p></div>
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
    const history = [];
    let activeAbort = null;
    let pinnedToBottom = true;
    let forceAutoScroll = false;
    let scrollFrame = null;
    let postLayoutScrollTimer = null;

    const SVG_SEND = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 5l7 7-7 7"/></svg>';
    const SVG_STOP = '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>';

    // ---------- settings ------------------------------------------------------
    const SETTINGS_KEY = "mtplx.chat.settings.v4";
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
    async function discoverServerLimits() {
      try {
        const res = await fetch("/health", {cache: "no-store"});
        if (!res.ok) throw new Error("health " + res.status);
        const health = await res.json();
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
    function loadSettings() {
      try {
        const raw = window.localStorage.getItem(SETTINGS_KEY);
        if (!raw) return Object.assign({}, DEFAULTS);
        const parsed = JSON.parse(raw);
        return Object.assign({}, DEFAULTS, parsed && typeof parsed === "object" ? parsed : {});
      } catch (err) {
        console.warn("settings load failed", err);
        return Object.assign({}, DEFAULTS);
      }
    }
    function saveSettings(s) {
      try { window.localStorage.setItem(SETTINGS_KEY, JSON.stringify(s)); } catch (_e) { /* ignore quota */ }
    }
    function clamp(value, min, max, fallback, isInt) {
      const n = isInt ? parseInt(value, 10) : parseFloat(value);
      if (!Number.isFinite(n)) return fallback;
      return Math.min(max, Math.max(min, n));
    }
    function applySettingsToUI(s) {
      ctlEls.temperature.value = clamp(s.temperature, RANGES.temperature.min, RANGES.temperature.max, DEFAULTS.temperature, false);
      ctlEls.top_p.value = clamp(s.top_p, RANGES.top_p.min, RANGES.top_p.max, DEFAULTS.top_p, false);
      ctlEls.top_k.value = clamp(s.top_k, RANGES.top_k.min, RANGES.top_k.max, DEFAULTS.top_k, true);
      ctlEls.mtp_enabled.checked = s.mtp_enabled !== false;
      ctlEls.depth.value = clamp(s.depth, RANGES.depth.min, RANGES.depth.max, DEFAULTS.depth, true);
      ctlEls.max_tokens.value = clamp(s.max_tokens, RANGES.max_tokens.min, RANGES.max_tokens.max, DEFAULTS.max_tokens, true);
      ctlEls.reasoning.value = String(s.reasoning || "auto");
      ctlEls.system.value = String(s.system || "");
      refreshLabels();
      refreshSliderFills();
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
    discoverServerLimits().then(() => {
      // Re-clamp + redraw after the real context window arrives so users
      // who reload the page don't see "8k" sitting under a fresh 256k cap.
      settings = readSettings();
      saveSettings(settings);
    });
    for (const key of Object.keys(ctlEls)) {
      ctlEls[key].addEventListener("input", () => { settings = readSettings(); saveSettings(settings); });
    }
    document.getElementById("reset-defaults").addEventListener("click", () => {
      settings = Object.assign({}, DEFAULTS);
      applySettingsToUI(settings);
      saveSettings(settings);
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

      const settingsNow = readSettings();
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

      activeAbort = new AbortController();
      let assistantText = "";
      let reasoningText = "";
      let finalStats = null;
      let firstTokenAt = null;
      const startedAt = performance.now();
      const liveState = {tokens: 0, elapsed: 0, tps: null, hasServerProgress: false};
      // Stall watchdog: if no SSE chunk arrives within this window, abort
      // and surface a clear error instead of leaving the UI parked on
      // "Thinking" forever (the failure mode the user reported after a
      // settings change).
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
            setStatus("Waiting on server (" + Math.round(idle / 1000) + "s)…", "streaming");
          }
        }, 1000);
      }
      function disarmStallWatchdog() {
        if (stallTimer) { clearInterval(stallTimer); stallTimer = null; }
      }
      try {
        armStallWatchdog();
        const response = await fetch("/v1/chat/completions", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
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
            "*[no response from server in " + Math.round(STALL_ABORT_MS / 1000) + "s — request aborted]*\\n\\n" +
              "If MTPLX is still loading the model, wait a few seconds and retry. " +
              "If the server has crashed, restart it: `mtplx start --max` (or just `mtplx start`)."
          );
          attachCopyButtons(turn.answerBody);
          setStatus("Server unresponsive", "");
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
        template
        .replace("__MODEL__", html.escape(model_id))
        .replace("__API_NOTE__", html.escape(api_note))
        .replace("__SERVER_URL__", html.escape(server_url))
        .replace("__MODEL_JSON__", json.dumps(model_id))
        .replace("__DEFAULT_SETTINGS_JSON__", json.dumps(default_settings, sort_keys=True))
        .replace("__DEPTH_VALUE__", str(default_depth))
        .replace("__DEPTH_MAX__", str(default_depth))
    )


def _thinking_enabled_for_request(
    state: ServerState,
    request: ChatCompletionRequest,
) -> bool:
    if state.args.reasoning_parser == "none":
        return False
    return state.args.enable_thinking if request.enable_thinking is None else bool(request.enable_thinking)


def create_app(state: ServerState) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            state.generation_executor.shutdown(wait=False, cancel_futures=True)

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
    async def api_key_and_rate_limit(request: Request, call_next: Callable[[Request], Any]) -> Any:
        if not _request_is_authorized(request, state.args.api_key):
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
                    "mtp_enabled": str(getattr(state.args, "generation_mode", "mtp")) == "mtp",
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
                "api_key": "leave blank for localhost" if not state.args.api_key else "use the API key you started MTPLX with",
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
        if hasattr(state, "foreground_count"):
            foreground_active = int(state.foreground_count())
        else:
            foreground_active = int(getattr(state, "foreground_active", 0) or 0)
        return {
            "ok": True,
            "model": state.model_id,
            "model_path": str(state.runtime.model_path),
            "generation_mode": state.args.generation_mode,
            "default_generation_mode": state.args.generation_mode,
            "available_generation_modes": ["mtp", "ar"],
            "load_mtp": bool(state.args.load_mtp),
            "mtp_enabled": bool(state.runtime.mtp_enabled),
            "depth": state.args.depth,
            "profile": state.profile.to_dict(),
            "adaptive": _adaptive_config(state.args),
            "proposal_cache": _proposal_cache_config(state.args),
            "online_hidden": _online_hidden_config(state.args),
            "verify_core": state.args.verify_core,
            "verify_strategy": state.args.verify_strategy,
            "enable_thinking": state.args.enable_thinking,
            "context_window": state.context_window,
            "max_response_tokens": state.args.max_response_tokens,
            "api_key_required": bool(state.args.api_key),
            "rate_limit_per_minute": int(state.args.rate_limit),
            "stream_interval": int(state.args.stream_interval),
            "warmup": state.warmup_status,
            "foreground_active": foreground_active,
            "active_requests": foreground_active,
            "last_request_started_at": getattr(state, "last_request_started_at", 0.0),
            "requests_completed": getattr(state, "requests_completed", 0),
            "last_request_at": getattr(state, "last_request_at", 0.0),
            "idle_seconds": (
                time.time() - getattr(state, "last_request_at", 0.0)
                if getattr(state, "last_request_at", 0.0) > 0
                else None
            ),
            "reasoning_parser": state.args.reasoning_parser,
            "load_time_s": state.load_time_s,
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
            "defer_verify_hidden_eval": os.environ.get("MTPLX_DEFER_VERIFY_HIDDEN_EVAL"),
            "late_depth_switch_after_tokens": os.environ.get(
                "MTPLX_LATE_DEPTH_SWITCH_AFTER_TOKENS"
            ),
            "late_depth_before": os.environ.get("MTPLX_LATE_DEPTH_BEFORE"),
            "late_depth_after": os.environ.get("MTPLX_LATE_DEPTH_AFTER"),
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
            "live_output_detach_mode": os.environ.get(
                "MTPLX_DETACH_LIVE_OUTPUTS_MODE"
            ),
            "mlx_cache_limit": state.mlx_cache_limit_status,
            "mlx_fork": state.mlx_fork_status,
        }

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return {
            "latest": state.last_metrics[-1] if state.last_metrics else None,
            "recent": state.last_metrics[-32:],
        }

    @app.get("/admin/sessions")
    def admin_sessions() -> dict[str, Any]:
        return state.sessions.list_sessions()

    @app.post("/admin/sessions/{session_id}/clear")
    def admin_clear_session(session_id: str) -> dict[str, Any]:
        return state.sessions.clear_session(session_id)

    @app.post("/admin/cache/clear")
    def admin_clear_cache() -> dict[str, Any]:
        return state.sessions.clear_all()

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
    async def chat_completions(raw_request: Request, request: ChatCompletionRequest) -> Any:
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        headers = dict(raw_request.headers)
        metadata = _request_metadata(request)
        cache_bypass = (
            headers.get("x-mtplx-cache-mode", "").lower() in {"bypass", "stateless", "off"}
            or str(metadata.get("cache_mode", "")).lower() in {"bypass", "stateless", "off"}
        )
        tool_specs = _normalize_tool_specs(request.tools)
        tools_active = _tools_active_for_request(tool_specs, request.tool_choice)
        background = is_background_request(
            messages=request.messages,
            max_tokens=request.max_tokens,
            headers=headers,
            metadata=metadata,
            main_system_hash=state.main_system_prompt_hash,
        )
        if background and (state.has_foreground() or state.lock.locked()):
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
        thinking_enabled = _thinking_enabled_for_request(state, request)
        if tools_active and request.enable_thinking is None:
            thinking_enabled = False
        request_generation_mode = _request_generation_mode_for_generation(state, request)
        request_depth = _request_depth_for_generation(
            state,
            request,
            generation_mode=request_generation_mode,
        )
        prompt_ids = _encode_messages(
            state.runtime.tokenizer,
            request.messages,
            enable_thinking=thinking_enabled,
            strip_assistant_reasoning_history=state.args.strip_assistant_reasoning_history,
            tools=tool_specs if tools_active else None,
        )
        current_system_hash = system_prompt_hash(request.messages)
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
        policy_fingerprint = _policy_fingerprint(
            state,
            thinking_enabled=thinking_enabled,
            generation_mode=request_generation_mode,
            depth=request_depth,
        )
        if not background and not cache_bypass:
            requested_restore_mode = headers.get("x-mtplx-restore-mode", "reference_lease")
            requested_restore_mode = requested_restore_mode.replace("-", "_")
            session_restore_mode = (
                "clone"
                if requested_restore_mode == "clone"
                else "reference_lease"
            )
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
        request_observability = _request_observability(
            request,
            headers=headers,
            metadata=metadata,
            session_source=session_source,
            request_generation_mode=request_generation_mode,
            request_depth=request_depth,
        )
        model = request.model or state.model_id
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        def run_generation_for_response() -> dict[str, Any]:
            if session is None:
                return _run_generation(
                    state,
                    prompt_ids,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    seed=request.seed,
                    generation_mode=request_generation_mode,
                    depth=request_depth,
                    session_id=session_id,
                    cache_miss_reason=cache_miss_reason,
                    session_restore_mode=session_restore_mode,
                    session_bank=None if background or cache_bypass else state.sessions.bank,
                    session_template_hash=state.template_hash,
                    session_draft_head_identity=state.draft_head_identity,
                    session_policy_fingerprint=policy_fingerprint,
                    background_request=background,
                    request_observability=request_observability,
                )
            with session.in_flight_generation():
                generated_result = _run_generation(
                    state,
                    prompt_ids,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    seed=request.seed,
                    generation_mode=request_generation_mode,
                    depth=request_depth,
                    session_id=session_id,
                    cache_miss_reason=cache_miss_reason,
                    session_restore_mode=session_restore_mode,
                    session_bank=state.sessions.bank,
                    session_template_hash=state.template_hash,
                    session_draft_head_identity=state.draft_head_identity,
                    session_policy_fingerprint=policy_fingerprint,
                    request_observability=request_observability,
                )
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
        ) -> None:
            if session is None:
                return
            loop = asyncio.get_running_loop()
            if state.args.session_postcommit_mode == "async":
                generated["stats"]["session_postcommit_snapshot"] = {
                    "stored": False,
                    "mode": "async_pending",
                }

                def async_postcommit() -> None:
                    try:
                        _store_retokenized_history_snapshot(
                            state,
                            session_id=session_id,
                            messages=request.messages,
                            assistant_content=assistant_content,
                            assistant_tool_calls=assistant_tool_calls,
                            thinking_enabled=thinking_enabled,
                            policy_fingerprint=policy_fingerprint,
                        )
                    except BaseException as exc:
                        print(
                            f"[mtplx] async session postcommit failed: {exc!r}",
                            flush=True,
                        )

                state.generation_executor.submit(async_postcommit)
                return
            postcommit = await loop.run_in_executor(
                state.generation_executor,
                lambda: _store_retokenized_history_snapshot(
                    state,
                    session_id=session_id,
                    messages=request.messages,
                    assistant_content=assistant_content,
                    assistant_tool_calls=assistant_tool_calls,
                    thinking_enabled=thinking_enabled,
                    policy_fingerprint=policy_fingerprint,
                ),
            )
            generated["stats"]["session_postcommit_snapshot"] = postcommit

        if request.stream and tools_active:
            async def tool_event_stream():
                def stream_chunk(
                    *,
                    delta: dict[str, Any],
                    finish_reason: str | None = None,
                    usage: dict[str, int] | None = None,
                    mtplx_stats: dict[str, Any] | None = None,
                ) -> str:
                    payload: dict[str, Any] = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": finish_reason,
                            }
                        ],
                    }
                    if usage is not None:
                        payload["usage"] = usage
                    if mtplx_stats is not None:
                        payload["mtplx_stats"] = mtplx_stats
                    return f"data: {json.dumps(payload)}\n\n"

                def stream_error(exc: BaseException) -> str:
                    if isinstance(exc, HTTPException):
                        message = str(exc.detail)
                        error_type = str(exc.status_code)
                        status_code = exc.status_code
                    else:
                        message = str(exc)
                        error_type = type(exc).__name__
                        status_code = 500
                    payload = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                        "error": {
                            "message": message,
                            "type": error_type,
                            "status_code": status_code,
                        },
                    }
                    return f"data: {json.dumps(payload)}\n\n"

                yield stream_chunk(delta={"role": "assistant"})
                loop = asyncio.get_running_loop()
                try:
                    generated = await loop.run_in_executor(
                        state.generation_executor,
                        run_generation_for_response,
                    )
                    tool_calls = _parse_generated_tool_calls(
                        str(generated["text"]),
                        tools=tool_specs,
                    )
                    if tool_calls:
                        generated["finish_reason"] = "tool_calls"
                        await store_postcommit_snapshot(
                            generated,
                            assistant_content="",
                            assistant_tool_calls=tool_calls,
                        )
                        yield stream_chunk(
                            delta={
                                "tool_calls": [
                                    {"index": index, **tool_call}
                                    for index, tool_call in enumerate(tool_calls)
                                ]
                            }
                        )
                        finish_reason = "tool_calls"
                    else:
                        display_text = _display_text(
                            state,
                            generated,
                            thinking_enabled=thinking_enabled,
                        )
                        await store_postcommit_snapshot(
                            generated,
                            assistant_content=display_text,
                        )
                        if display_text:
                            yield stream_chunk(delta={"content": display_text})
                        finish_reason = generated.get("finish_reason", "stop")
                except EngineSessionBusy as exc:
                    yield stream_error(HTTPException(status_code=409, detail=str(exc)))
                    yield "data: [DONE]\n\n"
                    return
                except BaseException as exc:
                    yield stream_error(exc)
                    yield "data: [DONE]\n\n"
                    return

                yield stream_chunk(
                    delta={},
                    finish_reason=finish_reason,
                    usage=_usage_payload(generated),
                    mtplx_stats=_public_mtplx_stats(generated),
                )
                yield "data: [DONE]\n\n"

            return StreamingResponse(tool_event_stream(), media_type="text/event-stream")

        if request.stream:
            async def event_stream():
                first = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(first)}\n\n"

                queue: Queue[tuple[str, Any]] = Queue()
                cancel_event = Event()
                decoder = _IncrementalTokenDecoder(state.runtime.tokenizer)
                splitter = _ThinkingContentStreamSplitter(thinking_enabled=thinking_enabled)
                stream_interval = max(1, int(state.args.stream_interval))
                pending_stream_tokens: list[int] = []
                commit_event = Event()
                commit_state = {"commit": False, "assistant_history_content": None}

                def on_tokens(new_tokens: list[int]) -> None:
                    _raise_if_stream_cancelled(cancel_event)
                    queue.put(
                        (
                            "tokens",
                            {
                                "tokens": list(new_tokens),
                                "timestamp_s": time.perf_counter(),
                            },
                        )
                    )
                    _raise_if_stream_cancelled(cancel_event)

                def worker() -> None:
                    try:
                        _raise_if_stream_cancelled(cancel_event)
                        if session is None:
                            generated = _run_generation(
                                state,
                                prompt_ids,
                                max_tokens=request.max_tokens,
                                temperature=request.temperature,
                                top_p=request.top_p,
                                top_k=request.top_k,
                                seed=request.seed,
                                generation_mode=request_generation_mode,
                                depth=request_depth,
                                token_callback=on_tokens,
                                session_id=session_id,
                                cache_miss_reason=cache_miss_reason,
                                session_restore_mode=session_restore_mode,
                                session_bank=None if background or cache_bypass else state.sessions.bank,
                                session_template_hash=state.template_hash,
                                session_draft_head_identity=state.draft_head_identity,
                                session_policy_fingerprint=policy_fingerprint,
                                background_request=background,
                                request_observability=request_observability,
                            )
                        else:
                            with session.in_flight_generation():
                                generated = _run_generation(
                                    state,
                                    prompt_ids,
                                    max_tokens=request.max_tokens,
                                    temperature=request.temperature,
                                    top_p=request.top_p,
                                    top_k=request.top_k,
                                    seed=request.seed,
                                    generation_mode=request_generation_mode,
                                    depth=request_depth,
                                    token_callback=on_tokens,
                                    session_id=session_id,
                                    cache_miss_reason=cache_miss_reason,
                                    session_restore_mode=session_restore_mode,
                                    session_bank=state.sessions.bank,
                                    session_template_hash=state.template_hash,
                                    session_draft_head_identity=state.draft_head_identity,
                                    session_policy_fingerprint=policy_fingerprint,
                                    commit_final_state_to_bank=False,
                                    request_observability=request_observability,
                                )
                                queue.put(("done", generated))
                                commit_event.wait()
                                if commit_state["commit"]:
                                    assistant_history_content = (
                                        str(commit_state.get("assistant_history_content") or "")
                                        or (
                                            _normalize_thinking_tags(
                                                str(generated["text"]),
                                                thinking_enabled=thinking_enabled,
                                            )
                                            if state.args.normalize_thinking_tags
                                            else str(generated["text"])
                                        )
                                    )
                                    postcommit = _store_retokenized_history_snapshot(
                                        state,
                                        session_id=session_id,
                                        messages=request.messages,
                                        assistant_content=assistant_history_content,
                                        thinking_enabled=thinking_enabled,
                                        policy_fingerprint=policy_fingerprint,
                                    )
                                    generated["stats"]["session_postcommit_snapshot"] = postcommit
                                    session.commit(
                                        prompt_ids=prompt_ids,
                                        generated_ids=generated["tokens"],
                                        finish_reason=generated.get("finish_reason", "stop"),
                                    )
                                    queue.put(("committed", generated))
                                else:
                                    queue.put(("released", None))
                                return
                    except _StreamCancelled as exc:
                        queue.put(("cancelled", exc))
                    except EngineSessionBusy as exc:
                        queue.put(("error", HTTPException(status_code=409, detail=str(exc))))
                    except BaseException as exc:
                        if (
                            session is not None
                            and commit_event.is_set()
                            and state.args.session_postcommit_mode == "async"
                        ):
                            print(
                                f"[mtplx] async session postcommit failed: {exc!r}",
                                flush=True,
                            )
                        queue.put(("error", exc))
                    else:
                        queue.put(("done", generated))

                generation_future = state.generation_executor.submit(worker)

                def delta_chunk(field: str, text: str) -> str:
                    payload = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                                {
                                    "index": 0,
                                    "delta": {field: text},
                                    "finish_reason": None,
                                }
                        ],
                    }
                    return f"data: {json.dumps(payload)}\n\n"

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
                        error_type = str(exc.status_code)
                        status_code = exc.status_code
                    else:
                        message = str(exc)
                        error_type = type(exc).__name__
                        status_code = 500
                    payload = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                        "error": {
                            "message": message,
                            "type": error_type,
                            "status_code": status_code,
                        },
                    }
                    return f"data: {json.dumps(payload)}\n\n"

                generated: dict[str, Any] | None = None
                history_reasoning_chunks: list[str] = []
                history_content_chunks: list[str] = []
                streamed_progress_tokens = 0
                streamed_decode_started_s: float | None = None

                def remember_stream_chunk(field: str, text: str) -> None:
                    if field == "reasoning_content":
                        history_reasoning_chunks.append(text)
                    elif field == "content":
                        history_content_chunks.append(text)

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
                        batch_size = len(pending_stream_tokens) if force else stream_interval
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
                    reasoning = (
                        "".join(history_reasoning_chunks)
                        .replace(THINK_OPEN, "")
                        .replace(THINK_CLOSE, "")
                        .strip()
                    )
                    content = (
                        "".join(history_content_chunks)
                        .replace(THINK_OPEN, "")
                        .replace(THINK_CLOSE, "")
                        .strip()
                    )
                    pieces: list[str] = []
                    if reasoning:
                        pieces.append(f"{THINK_OPEN}\n{reasoning}\n{THINK_CLOSE}")
                    if content:
                        pieces.append(content)
                    return "\n\n".join(pieces)

                for field, text in splitter.start():
                    if text:
                        remember_stream_chunk(field, text)
                        yield delta_chunk(field, text)

                try:
                    while True:
                        try:
                            kind, item = await asyncio.to_thread(queue.get, True, 0.25)
                        except Empty:
                            if cancel_event.is_set() or await raw_request.is_disconnected():
                                _cancel_stream_generation(cancel_event, generation_future)
                                return
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
                                if streamed_decode_started_s is None:
                                    streamed_decode_started_s = token_timestamp_s
                                streamed_progress_tokens += len(stream_tokens)
                                yield progress_chunk(
                                    _stream_progress_payload(
                                        completion_tokens=streamed_progress_tokens,
                                        decode_started_s=streamed_decode_started_s,
                                        now_s=token_timestamp_s,
                                    )
                                )
                            for field, text in drain_stream_tokens(stream_tokens):
                                remember_stream_chunk(field, text)
                                yield delta_chunk(field, text)
                        elif kind == "done":
                            generated = item
                            for field, text in drain_stream_tokens([], force=True):
                                remember_stream_chunk(field, text)
                                yield delta_chunk(field, text)
                            tail = decoder.finish()
                            if tail:
                                for field, text in splitter.feed(tail):
                                    if text:
                                        remember_stream_chunk(field, text)
                                        yield delta_chunk(field, text)
                            for field, text in splitter.finish():
                                if text:
                                    remember_stream_chunk(field, text)
                                    yield delta_chunk(field, text)
                            if session is not None:
                                commit_state["assistant_history_content"] = streamed_history_content()
                                commit_state["commit"] = True
                                commit_event.set()
                                if state.args.session_postcommit_mode == "async":
                                    generated["stats"]["session_postcommit_snapshot"] = {
                                        "stored": False,
                                        "mode": "async_pending",
                                    }
                                else:
                                    commit_kind, commit_item = await asyncio.to_thread(queue.get)
                                    if commit_kind == "committed":
                                        generated = commit_item
                                    elif commit_kind == "error":
                                        yield error_chunk(commit_item)
                                        yield "data: [DONE]\n\n"
                                        return
                                    else:
                                        yield error_chunk(
                                            RuntimeError(f"unexpected commit event: {commit_kind}")
                                        )
                                        yield "data: [DONE]\n\n"
                                        return
                            generated["stats"]["reasoning_reentries"] = splitter.reentry_count
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
                                state.last_metrics[-1]["reasoning_reentries"] = splitter.reentry_count
                            footer = _stats_footer_text(state, generated)
                            if footer:
                                yield delta_chunk("content", f"\n\n{footer}")
                            break
                        elif kind == "error":
                            yield error_chunk(item)
                            yield "data: [DONE]\n\n"
                            return
                        elif kind == "cancelled":
                            return
                        else:
                            yield error_chunk(RuntimeError(f"unexpected stream event: {kind}"))
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
                    if session is not None and not commit_event.is_set():
                        commit_state["commit"] = False
                        commit_event.set()

                if generated is None:
                    yield error_chunk(RuntimeError("generation ended without a result"))
                    yield "data: [DONE]\n\n"
                    return
                done = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": generated.get("finish_reason", "stop"),
                        }
                    ],
                    "usage": _usage_payload(generated),
                    "mtplx_stats": _public_mtplx_stats(generated),
                }
                yield f"data: {json.dumps(done)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        def run_nonstream_generation() -> dict[str, Any]:
            return run_generation_for_response()

        loop = asyncio.get_running_loop()
        try:
            generated = await loop.run_in_executor(
                state.generation_executor,
                run_nonstream_generation,
            )
        except EngineSessionBusy as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        tool_calls = (
            _parse_generated_tool_calls(str(generated["text"]), tools=tool_specs)
            if tools_active
            else None
        )
        if tool_calls:
            generated["finish_reason"] = "tool_calls"
            await store_postcommit_snapshot(
                generated,
                assistant_content="",
                assistant_tool_calls=tool_calls,
            )
            message: dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            }
            finish_reason = "tool_calls"
        else:
            display_text = _display_text(
                state,
                generated,
                thinking_enabled=thinking_enabled,
            )
            await store_postcommit_snapshot(
                generated,
                assistant_content=display_text,
            )
            message = {"role": "assistant", "content": display_text}
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
    async def anthropic_messages(raw_request: Request, request: AnthropicMessagesRequest) -> Any:
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
                    model=request.model or state.model_id,
                ),
                media_type="text/event-stream",
            )
        if not isinstance(response, JSONResponse):
            return response
        try:
            openai_payload = json.loads(response.body)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"failed to translate response: {exc}") from exc
        if response.status_code >= 400:
            return response
        payload = _anthropic_payload_from_openai(openai_payload)
        return JSONResponse(payload, status_code=response.status_code)

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest) -> Any:
        prompt_ids = _encode_prompt(state.runtime.tokenizer, request.prompt)
        request_generation_mode = _request_generation_mode_for_generation(state, request)
        request_depth = _request_depth_for_generation(
            state,
            request,
            generation_mode=request_generation_mode,
        )
        generated = _run_generation(
            state,
            prompt_ids,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            seed=request.seed,
            generation_mode=request_generation_mode,
            depth=request_depth,
        )
        model = request.model or state.model_id
        response_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        display_text = _display_text(state, generated)
        if request.stream:
            async def event_stream():
                chunk_chars = 24 * max(1, int(state.args.stream_interval))
                for chunk in _chunk_text(display_text, chunk_chars=chunk_chars):
                    payload = {
                        "id": response_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "text": chunk, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")
        return JSONResponse(
            {
                "id": response_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "text": display_text, "finish_reason": "stop"}],
                "usage": _usage_payload(generated),
                "mtplx_stats": _public_mtplx_stats(generated),
            }
        )

    @app.exception_handler(Exception)
    async def unhandled_exception(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(exc), "type": type(exc).__name__}},
        )

    return app


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    postcommit_default = os.environ.get("MTPLX_SESSION_POSTCOMMIT_MODE", "inline")
    if postcommit_default not in {"inline", "async"}:
        postcommit_default = "inline"
    parser.add_argument("--model", default="models/Qwen3.6-27B-MTPLX-Flat4-CyanKiwiMTP")
    parser.add_argument("--model-id", default="mtplx-qwen36-27b-native-mtp")
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default=DEFAULT_PROFILE_NAME)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MTPLX_AUTH"),
        help="Require Bearer or X-API-Key auth. Required for non-localhost binds.",
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
    parser.add_argument("--temperature", "--default-temperature", dest="temperature", type=float, default=0.6)
    parser.add_argument("--top-p", "--default-top-p", dest="top_p", type=float, default=0.95)
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
            "SessionBank postcommit policy. 'inline' preserves existing behavior; "
            "'async' returns the stream/response before the retokenized history "
            "snapshot finishes, while the single generation executor completes "
            "the exact snapshot in the background."
        ),
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass enable_thinking to the Qwen chat template for visible <think> reasoning blocks.",
    )
    parser.add_argument(
        "--reasoning-parser",
        choices=["qwen3", "none"],
        default="qwen3",
        help="Parser for streamed reasoning tags. Use 'none' to stream all text as content.",
    )
    parser.add_argument(
        "--strip-assistant-reasoning-history",
        action="store_true",
        help=(
            "Opt-in speed/debug mode: remove prior assistant <think>/reasoning blocks "
            "before re-encoding chat history. Default preserves reasoning context."
        ),
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_server_security_args(args)
    try:
        state = ServerState(args)
    except RuntimeError as exc:
        if str(exc).startswith("Patched MLX qmv fork is not active:"):
            _startup_line("error: fast MLX fork is not active")
            _startup_line(str(exc))
            _startup_line("try: mtplx start --profile safe")
            _startup_line("try: mtplx start --profile performance-cold")
            _startup_line("     (public start disables the strict fork assert when the fork is missing)")
            raise SystemExit(2) from None
        raise
    app = create_app(state)
    import uvicorn

    _startup_line()
    _startup_line("MTPLX is ready.")
    _startup_line("Chat UI: " + _startup_chat_url(args))
    _startup_line("OpenAI API Base URL: " + _startup_openai_base_url(args))
    _startup_line("Model: " + str(args.model_id))
    _startup_line("API key: leave blank for localhost")
    _startup_line("Health check: " + _startup_server_url(args) + "/health")
    _startup_line("Keep this terminal open. Press Ctrl-C to stop MTPLX.")
    if args.open_browser:
        _startup_line("Opening chat UI in your browser...")
        _open_browser_later(_startup_chat_url(args))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
