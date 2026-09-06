"""Dependency-free, privacy-first OTLP/HTTP trace export.

The exporter uses only the Python standard library, remains disabled until an
endpoint is configured, bounds memory with a finite queue, and never lets an
observability outage fail a generation request.  Content-bearing values are
reduced to length and SHA-256 metadata by default; ordinary numeric metrics
such as ``prompt_tokens`` and ``completion_tokens`` remain exportable.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

_SECRET_PARTS = (
    "authorization",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "auth_token",
    "password",
    "passwd",
    "secret",
    "credential",
    "cookie",
    "session_token",
)
_CONTENT_EXACT = {
    "prompt",
    "prompt_text",
    "messages",
    "message",
    "input",
    "input_text",
    "output",
    "output_text",
    "response",
    "response_text",
    "completion",
    "completion_text",
    "generated_text",
    "text",
    "exception_message",
    "error_message",
    "stacktrace",
    "traceback",
    "tool_arguments",
    "tool_result",
    "reasoning",
    "thinking",
}
_NUMERIC_SUFFIXES = (
    "_tokens",
    "_bytes",
    "_count",
    "_total",
    "_rate",
    "_ratio",
    "_ms",
    "_ns",
    "_seconds",
    "_s",
    "_tps",
)


def _normalized_key(key: Any) -> str:
    return str(key).strip().lower().replace("-", "_").replace(".", "_")


def _is_secret_key(key: Any) -> bool:
    normalized = _normalized_key(key)
    return any(part in normalized for part in _SECRET_PARTS)


def _is_content_key(key: Any, value: Any) -> bool:
    normalized = _normalized_key(key)
    if isinstance(value, (int, float, bool)) and (
        normalized.endswith(_NUMERIC_SUFFIXES)
        or normalized in {"latency", "duration", "temperature", "top_p", "top_k"}
    ):
        return False
    if normalized in _CONTENT_EXACT:
        return True
    if normalized.endswith(("_text", "_message", "_messages", "_prompt", "_response")):
        return True
    return False


def _content_digest(value: Any) -> dict[str, Any]:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    except Exception:
        encoded = repr(type(value)).encode("utf-8")
    return {
        "content_redacted": True,
        "content_bytes": len(encoded),
        "content_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def sanitize_attributes(
    value: Any,
    *,
    allow_content: bool = False,
    maximum_string_bytes: int = 512,
    maximum_collection_items: int = 64,
    _key: str = "",
    _depth: int = 0,
) -> Any:
    """Return a bounded JSON-safe value suitable for telemetry export."""

    if _depth > 6:
        return "<depth-limit>"
    if _is_secret_key(_key):
        return "<redacted>"
    if not allow_content and _is_content_key(_key, value):
        return _content_digest(value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return str(value)
        return value
    if isinstance(value, bytes):
        if allow_content:
            return value[:maximum_string_bytes].decode("utf-8", errors="replace")
        return _content_digest(value.hex())
    if isinstance(value, str):
        encoded = value.encode("utf-8", errors="replace")
        if len(encoded) <= maximum_string_bytes:
            return value
        prefix = encoded[:maximum_string_bytes].decode("utf-8", errors="replace")
        return {
            "truncated": True,
            "prefix": prefix,
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= maximum_collection_items:
                result["_truncated_items"] = len(value) - maximum_collection_items
                break
            key_text = str(key)
            result[key_text] = sanitize_attributes(
                item,
                allow_content=allow_content,
                maximum_string_bytes=maximum_string_bytes,
                maximum_collection_items=maximum_collection_items,
                _key=key_text,
                _depth=_depth + 1,
            )
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        result = [
            sanitize_attributes(
                item,
                allow_content=allow_content,
                maximum_string_bytes=maximum_string_bytes,
                maximum_collection_items=maximum_collection_items,
                _key=_key,
                _depth=_depth + 1,
            )
            for item in items[:maximum_collection_items]
        ]
        if len(items) > maximum_collection_items:
            result.append({"truncated_items": len(items) - maximum_collection_items})
        return result
    return sanitize_attributes(
        str(value),
        allow_content=allow_content,
        maximum_string_bytes=maximum_string_bytes,
        maximum_collection_items=maximum_collection_items,
        _key=_key,
        _depth=_depth + 1,
    )


@dataclass(frozen=True)
class OTLPExporterConfig:
    endpoint: str | None = None
    service_name: str = "mtplx"
    service_version: str = "unknown"
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout_s: float = 2.0
    batch_size: int = 32
    flush_interval_s: float = 1.0
    maximum_queue_size: int = 1024
    maximum_attribute_bytes: int = 512
    maximum_collection_items: int = 64
    allow_content: bool = False
    enabled: bool = False

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.flush_interval_s <= 0:
            raise ValueError("flush_interval_s must be positive")
        if self.maximum_queue_size < 1:
            raise ValueError("maximum_queue_size must be at least 1")
        if self.maximum_attribute_bytes < 32:
            raise ValueError("maximum_attribute_bytes must be at least 32")
        if self.maximum_collection_items < 1:
            raise ValueError("maximum_collection_items must be at least 1")
        if self.endpoint:
            parsed = urllib.parse.urlparse(self.endpoint)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("OTLP endpoint must be an absolute http(s) URL")

    @classmethod
    def from_env(cls) -> OTLPExporterConfig:
        endpoint = os.environ.get("MTPLX_OTLP_ENDPOINT") or os.environ.get(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
        )
        if endpoint and not endpoint.rstrip("/").endswith("/v1/traces"):
            endpoint = endpoint.rstrip("/") + "/v1/traces"
        raw_headers = os.environ.get("MTPLX_OTLP_HEADERS") or os.environ.get(
            "OTEL_EXPORTER_OTLP_HEADERS", ""
        )
        headers: dict[str, str] = {}
        for item in raw_headers.split(","):
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            if key.strip():
                headers[key.strip()] = urllib.parse.unquote(value.strip())
        truthy = {"1", "true", "yes", "on"}
        return cls(
            endpoint=endpoint,
            service_name=os.environ.get("OTEL_SERVICE_NAME", "mtplx"),
            service_version=os.environ.get("MTPLX_VERSION", "unknown"),
            headers=headers,
            timeout_s=float(os.environ.get("MTPLX_OTLP_TIMEOUT_S", "2")),
            batch_size=int(os.environ.get("MTPLX_OTLP_BATCH_SIZE", "32")),
            flush_interval_s=float(os.environ.get("MTPLX_OTLP_FLUSH_INTERVAL_S", "1")),
            maximum_queue_size=int(os.environ.get("MTPLX_OTLP_QUEUE_SIZE", "1024")),
            allow_content=os.environ.get("MTPLX_OTLP_ALLOW_CONTENT", "").lower()
            in truthy,
            enabled=bool(endpoint)
            and os.environ.get("MTPLX_OTLP_DISABLED", "").lower() not in truthy,
        )


@dataclass(frozen=True)
class OTLPSpan:
    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    start_time_unix_nano: int
    end_time_unix_nano: int
    attributes: Mapping[str, Any]
    status_code: int = 1
    status_message: str = ""


class OTLPExporter:
    """Bounded asynchronous OTLP/HTTP JSON exporter."""

    def __init__(self, config: OTLPExporterConfig | None = None) -> None:
        self.config = config or OTLPExporterConfig.from_env()
        self._queue: queue.Queue[OTLPSpan] = queue.Queue(self.config.maximum_queue_size)
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._exported_spans = 0
        self._dropped_spans = 0
        self._failed_exports = 0
        self._active_exports = 0
        self._last_error_type: str | None = None
        self._last_export_s: float | None = None
        if self.config.enabled:
            self.start()

    def start(self) -> None:
        if not self.config.enabled:
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._worker,
                name="mtplx-otlp",
                daemon=True,
            )
            self._thread.start()

    def emit(self, span: OTLPSpan) -> bool:
        if not self.config.enabled:
            return False
        try:
            self._queue.put_nowait(span)
        except queue.Full:
            with self._lock:
                self._dropped_spans += 1
            return False
        if self._queue.qsize() >= self.config.batch_size:
            self._wake.set()
        return True

    @contextmanager
    def span(
        self,
        name: str,
        *,
        attributes: Mapping[str, Any] | None = None,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        started = time.time_ns()
        mutable = dict(attributes or {})
        status_code = 1
        status_message = ""
        try:
            yield mutable
        except Exception as exc:
            status_code = 2
            status_message = type(exc).__name__
            mutable.setdefault("exception_type", type(exc).__name__)
            raise
        finally:
            clean = sanitize_attributes(
                mutable,
                allow_content=self.config.allow_content,
                maximum_string_bytes=self.config.maximum_attribute_bytes,
                maximum_collection_items=self.config.maximum_collection_items,
            )
            assert isinstance(clean, Mapping)
            self.emit(
                OTLPSpan(
                    name=str(name)[:256],
                    trace_id=trace_id or secrets.token_hex(16),
                    span_id=secrets.token_hex(8),
                    parent_span_id=parent_span_id,
                    start_time_unix_nano=started,
                    end_time_unix_nano=time.time_ns(),
                    attributes=dict(clean),
                    status_code=status_code,
                    status_message=status_message,
                )
            )

    @staticmethod
    def _any_value(value: Any) -> dict[str, Any]:
        if value is None:
            return {"stringValue": "null"}
        if isinstance(value, bool):
            return {"boolValue": value}
        if isinstance(value, int):
            return {"intValue": str(value)}
        if isinstance(value, float):
            return {"doubleValue": value}
        if isinstance(value, str):
            return {"stringValue": value}
        return {
            "stringValue": json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        }

    def _payload(self, spans: Sequence[OTLPSpan]) -> bytes:
        resource_attributes = [
            {"key": "service.name", "value": {"stringValue": self.config.service_name}},
            {
                "key": "service.version",
                "value": {"stringValue": self.config.service_version},
            },
        ]
        rows = []
        for span in spans:
            row: dict[str, Any] = {
                "traceId": span.trace_id,
                "spanId": span.span_id,
                "name": span.name,
                "kind": 1,
                "startTimeUnixNano": str(span.start_time_unix_nano),
                "endTimeUnixNano": str(span.end_time_unix_nano),
                "attributes": [
                    {"key": str(key), "value": self._any_value(value)}
                    for key, value in sorted(span.attributes.items())
                ],
                "status": {
                    "code": span.status_code,
                    "message": span.status_message,
                },
            }
            if span.parent_span_id:
                row["parentSpanId"] = span.parent_span_id
            rows.append(row)
        payload = {
            "resourceSpans": [
                {
                    "resource": {"attributes": resource_attributes},
                    "scopeSpans": [
                        {
                            "scope": {"name": "mtplx.native", "version": "1"},
                            "spans": rows,
                        }
                    ],
                }
            ]
        }
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )

    def _send(self, spans: Sequence[OTLPSpan]) -> None:
        if not spans or not self.config.endpoint:
            return
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "mtplx-native-otlp/1",
            **dict(self.config.headers),
        }
        request = urllib.request.Request(
            self.config.endpoint,
            data=self._payload(spans),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_s
            ) as response:
                if not 200 <= int(response.status) < 300:
                    raise urllib.error.HTTPError(
                        self.config.endpoint,
                        int(response.status),
                        "OTLP export failed",
                        response.headers,
                        None,
                    )
        except Exception as exc:
            with self._lock:
                self._failed_exports += 1
                self._last_error_type = type(exc).__name__
            return
        with self._lock:
            self._exported_spans += len(spans)
            self._last_export_s = time.time()
            self._last_error_type = None

    def _send_tracked(self, spans: Sequence[OTLPSpan]) -> None:
        with self._lock:
            self._active_exports += 1
        try:
            self._send(spans)
        finally:
            with self._lock:
                self._active_exports = max(0, self._active_exports - 1)

    def _drain(self, limit: int) -> list[OTLPSpan]:
        rows: list[OTLPSpan] = []
        while len(rows) < limit:
            try:
                rows.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return rows

    def _worker(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self.config.flush_interval_s)
            self._wake.clear()
            rows = self._drain(self.config.batch_size)
            if rows:
                self._send_tracked(rows)
        while True:
            rows = self._drain(self.config.batch_size)
            if not rows:
                break
            self._send_tracked(rows)

    def flush(self, *, timeout_s: float = 5.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_s)
        self._wake.set()
        while time.monotonic() < deadline:
            rows = self._drain(self.config.batch_size)
            if rows:
                self._send_tracked(rows)
                continue
            with self._lock:
                active = self._active_exports
            if self._queue.empty() and active == 0:
                return True
            time.sleep(0.005)
        with self._lock:
            active = self._active_exports
        return self._queue.empty() and active == 0

    def shutdown(self, *, timeout_s: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(max(0.0, timeout_s))
        self.flush(timeout_s=timeout_s)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "available": True,
                "enabled": self.config.enabled,
                "endpoint_configured": bool(self.config.endpoint),
                "content_export_enabled": self.config.allow_content,
                "queue_size": self._queue.qsize(),
                "queue_capacity": self.config.maximum_queue_size,
                "exported_spans": self._exported_spans,
                "dropped_spans": self._dropped_spans,
                "failed_exports": self._failed_exports,
                "active_exports": self._active_exports,
                "last_error_type": self._last_error_type,
                "last_export_s": self._last_export_s,
                "failure_is_request_fatal": False,
            }


_DEFAULT_EXPORTER: OTLPExporter | None = None
_DEFAULT_LOCK = threading.Lock()


def default_exporter() -> OTLPExporter:
    global _DEFAULT_EXPORTER
    with _DEFAULT_LOCK:
        if _DEFAULT_EXPORTER is None:
            _DEFAULT_EXPORTER = OTLPExporter()
        return _DEFAULT_EXPORTER


def reset_default_exporter_for_tests() -> None:
    global _DEFAULT_EXPORTER
    with _DEFAULT_LOCK:
        if _DEFAULT_EXPORTER is not None:
            _DEFAULT_EXPORTER.shutdown(timeout_s=0.2)
        _DEFAULT_EXPORTER = None
