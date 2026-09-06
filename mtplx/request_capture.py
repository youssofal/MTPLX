"""Privacy-safe per-request capture for deterministic local replay.

The capture ring is off by default. When enabled, content-bearing fields are
reduced to counts and stable digests unless each content class is explicitly
enabled. This keeps captures useful for correlation while preventing prompts,
responses, token IDs, messages, and exception text from being persisted by
accident.

Enable the ring with ``MTPLX_REQUEST_CAPTURE_DIR=<dir>``. The newest
``MTPLX_REQUEST_CAPTURE_KEEP`` files are retained, with a default of 200. Older
files are moved into a ``pruned/`` subdirectory rather than deleted.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

_LOCK = threading.Lock()
_PATHS_BY_ID: dict[str, str] = {}

_TRUE = {"1", "true", "yes", "on"}
_PROMPT_TEXT_KEYS = {"prompt", "prompt_text", "rendered_prompt"}
_RESPONSE_TEXT_KEYS = {
    "text",
    "text_head",
    "text_tail",
    "response_text",
    "completion_text",
}
_MESSAGE_KEYS = {"messages"}
_EXCEPTION_KEYS = {"error", "exception", "exception_message", "traceback"}
_SECRET_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "auth_token",
    "access_token",
    "refresh_token",
    "password",
    "passwd",
    "secret",
    "credential",
    "cookie",
)
_REDACTED = "<redacted>"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in _TRUE


@dataclass(frozen=True)
class CaptureOptions:
    """Explicit content controls. Every content-bearing option defaults off."""

    include_prompt_tokens: bool = False
    include_completion_tokens: bool = False
    include_prompt_text: bool = False
    include_response_text: bool = False
    include_messages: bool = False
    include_exception_text: bool = False
    prompt_token_limit: int = 2048
    completion_token_limit: int = 2048
    text_head_chars: int = 1000
    text_tail_chars: int = 1000

    @classmethod
    def from_env(cls) -> CaptureOptions:
        return cls(
            include_prompt_tokens=_env_flag(
                "MTPLX_REQUEST_CAPTURE_INCLUDE_PROMPT_TOKENS"
            ),
            include_completion_tokens=_env_flag(
                "MTPLX_REQUEST_CAPTURE_INCLUDE_COMPLETION_TOKENS"
            ),
            include_prompt_text=_env_flag("MTPLX_REQUEST_CAPTURE_INCLUDE_PROMPT_TEXT"),
            include_response_text=_env_flag(
                "MTPLX_REQUEST_CAPTURE_INCLUDE_RESPONSE_TEXT"
            ),
            include_messages=_env_flag("MTPLX_REQUEST_CAPTURE_INCLUDE_MESSAGES"),
            include_exception_text=_env_flag(
                "MTPLX_REQUEST_CAPTURE_INCLUDE_EXCEPTION_TEXT"
            ),
        )


def capture_dir() -> str | None:
    raw = str(os.environ.get("MTPLX_REQUEST_CAPTURE_DIR", "")).strip()
    return raw or None


def _keep_count() -> int:
    try:
        return max(1, int(os.environ.get("MTPLX_REQUEST_CAPTURE_KEEP", "200")))
    except ValueError:
        return 200


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_json_digest(value: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-safe value."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _safe_json_digest(value: Any) -> str:
    try:
        return stable_json_digest(value)
    except (TypeError, ValueError, OverflowError):
        return hashlib.sha256(repr(type(value)).encode("utf-8")).hexdigest()


def _text_digest(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _sensitive_key(key: Any) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return any(part in normalized for part in _SECRET_PARTS)


def _sanitize_nested(value: Any, options: CaptureOptions) -> Any:
    if isinstance(value, Mapping):
        nested = sanitize_capture_payload(value, options=options)
        nested.pop("capture_content_policy", None)
        return nested
    if isinstance(value, (list, tuple)):
        return [_sanitize_nested(item, options) for item in value]
    return value


def _coerce_token_ids(value: Any) -> list[int]:
    if value is None or isinstance(value, (str, bytes, bytearray, Mapping)):
        return []
    try:
        return [int(token) for token in value]
    except Exception:
        return []


def _content_limit(value: Any, default: int = 2048) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _sanitize_tokens(
    value: Any,
    *,
    kind: str,
    include: bool,
    limit: int,
) -> dict[str, Any]:
    tokens = _coerce_token_ids(value)
    result: dict[str, Any] = {
        f"{kind}_token_count": len(tokens),
        f"{kind}_tokens_sha256": stable_json_digest(tokens),
    }
    if include:
        bounded_limit = _content_limit(limit)
        result[f"{kind}_token_ids"] = tokens[:bounded_limit]
        result[f"{kind}_tokens_clipped"] = len(tokens) > bounded_limit
    return result


def _sanitize_text(
    key: str,
    value: Any,
    *,
    include: bool,
    options: CaptureOptions,
) -> dict[str, Any]:
    text = str(value or "")
    result: dict[str, Any] = {
        f"{key}_chars": len(text),
        f"{key}_sha256": _text_digest(text),
    }
    if include:
        result[f"{key}_excerpt"] = clip_text_head_tail(
            text,
            head=options.text_head_chars,
            tail=options.text_tail_chars,
        )
    return result


def sanitize_capture_payload(
    payload: Mapping[str, Any],
    *,
    options: CaptureOptions | None = None,
    outcome: bool = False,
) -> dict[str, Any]:
    """Copy and sanitize a capture payload under an explicit content policy."""

    options = options or CaptureOptions.from_env()
    try:
        source = deepcopy(dict(payload))
    except Exception as exc:
        return {
            "capture_sanitization_error": type(exc).__name__,
            "payload_type_sha256": _text_digest(repr(type(payload))),
        }

    sanitized: dict[str, Any] = {}
    derived: dict[str, Any] = {}
    for key, value in source.items():
        output_key = str(key)
        normalized = output_key.strip().lower()
        if normalized == "prompt_token_ids":
            derived.update(
                _sanitize_tokens(
                    value,
                    kind="prompt",
                    include=options.include_prompt_tokens,
                    limit=options.prompt_token_limit,
                )
            )
            continue
        if normalized == "completion_token_ids":
            derived.update(
                _sanitize_tokens(
                    value,
                    kind="completion",
                    include=options.include_completion_tokens,
                    limit=options.completion_token_limit,
                )
            )
            continue
        if normalized in _PROMPT_TEXT_KEYS:
            derived.update(
                _sanitize_text(
                    normalized,
                    value,
                    include=options.include_prompt_text,
                    options=options,
                )
            )
            continue
        if normalized in _RESPONSE_TEXT_KEYS:
            derived.update(
                _sanitize_text(
                    normalized,
                    value,
                    include=options.include_response_text,
                    options=options,
                )
            )
            continue
        if normalized in _MESSAGE_KEYS:
            count = len(value) if isinstance(value, (list, tuple)) else 0
            derived["messages_count"] = count
            derived["messages_sha256"] = _safe_json_digest(value)
            if options.include_messages:
                derived["messages"] = _sanitize_nested(value, options)
            continue
        if normalized in _EXCEPTION_KEYS:
            derived[f"{normalized}_sha256"] = _text_digest(value)
            if options.include_exception_text:
                derived[normalized] = str(value)
            continue
        sanitized[output_key] = (
            _REDACTED
            if _sensitive_key(output_key)
            else _sanitize_nested(value, options)
        )

    sanitized.update(derived)
    sanitized["capture_content_policy"] = {
        "prompt_tokens": options.include_prompt_tokens,
        "completion_tokens": options.include_completion_tokens,
        "prompt_text": options.include_prompt_text,
        "response_text": options.include_response_text,
        "messages": options.include_messages,
        "exception_text": options.include_exception_text,
        "phase": "outcome" if outcome else "request",
    }
    return sanitized


def _atomic_write(path: str, payload: dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _prune_locked(directory: str) -> None:
    try:
        entries = sorted(
            name
            for name in os.listdir(directory)
            if name.startswith("req-") and name.endswith(".json")
        )
    except OSError:
        return
    excess = len(entries) - _keep_count()
    if excess <= 0:
        return
    pruned_dir = os.path.join(directory, "pruned")
    os.makedirs(pruned_dir, exist_ok=True)
    for name in entries[:excess]:
        source_path = os.path.join(directory, name)
        try:
            os.replace(source_path, os.path.join(pruned_dir, name))
        except OSError:
            continue
        for request_id, mapped_path in tuple(_PATHS_BY_ID.items()):
            if mapped_path == source_path:
                _PATHS_BY_ID.pop(request_id, None)


def capture_request(
    request_id: str | None,
    payload: Mapping[str, Any],
    *,
    options: CaptureOptions | None = None,
) -> None:
    """Persist a sanitized reproduction envelope at dispatch time. Never raises."""

    directory = capture_dir()
    if not directory or not request_id:
        return
    try:
        os.makedirs(directory, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        safe_id = "".join(
            ch if ch.isalnum() or ch in "-_" else "_" for ch in str(request_id)
        )[:80]
        path = os.path.join(directory, f"req-{stamp}-{safe_id}.json")
        record = {
            "capture_version": 2,
            "captured_at_utc": stamp,
            "request_id": str(request_id),
            "phase": "dispatched",
            **sanitize_capture_payload(payload, options=options),
        }
        with _LOCK:
            _atomic_write(path, record)
            _PATHS_BY_ID[str(request_id)] = path
            _prune_locked(directory)
    except Exception:
        return


def capture_outcome(
    request_id: str | None,
    outcome: Mapping[str, Any],
    *,
    options: CaptureOptions | None = None,
) -> None:
    """Merge a sanitized completion outcome into its capture file. Never raises."""

    if not capture_dir() or not request_id:
        return
    try:
        with _LOCK:
            request_key = str(request_id)
            path = _PATHS_BY_ID.get(request_key)
            if not path or not os.path.exists(path):
                _PATHS_BY_ID.pop(request_key, None)
                return
            with open(path, "r", encoding="utf-8") as handle:
                record = json.load(handle)
            record["phase"] = "completed"
            record["outcome"] = sanitize_capture_payload(
                outcome,
                options=options,
                outcome=True,
            )
            _atomic_write(path, record)
            _PATHS_BY_ID.pop(request_key, None)
    except Exception:
        return


def completion_token_ids(
    tokens: Any,
    *,
    options: CaptureOptions | None = None,
) -> dict[str, Any]:
    """Return privacy-safe completion token metadata for a capture outcome.

    The existing one-argument call remains valid. Raw token IDs are included
    only when ``include_completion_tokens`` or its environment flag is enabled.
    """

    try:
        resolved = options or CaptureOptions.from_env()
        return _sanitize_tokens(
            tokens,
            kind="completion",
            include=resolved.include_completion_tokens,
            limit=resolved.completion_token_limit,
        )
    except Exception:
        return _sanitize_tokens(
            [],
            kind="completion",
            include=False,
            limit=0,
        )


def clip_text_head_tail(
    text: str,
    head: int = 2000,
    tail: int = 2000,
) -> dict[str, Any]:
    """Return a bounded diagnostic excerpt for an opted-in capture."""

    text = str(text or "")
    head = max(0, int(head))
    tail = max(0, int(tail))
    if len(text) <= head + tail:
        return {"text": text, "text_clipped": False}
    return {
        "text_head": text[:head],
        "text_tail": text[-tail:] if tail else "",
        "text_chars": len(text),
        "text_clipped": True,
    }


__all__ = [
    "CaptureOptions",
    "capture_dir",
    "capture_outcome",
    "capture_request",
    "clip_text_head_tail",
    "completion_token_ids",
    "sanitize_capture_payload",
    "stable_json_digest",
]
