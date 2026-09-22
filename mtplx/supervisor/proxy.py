"""Pure-ASGI front-door proxy: resolve a request's model, JIT-load it if
needed, and stream the request through to the chosen engine.

No Starlette ``BaseHTTPMiddleware`` anywhere (that wraps every SSE frame in
a rendezvous relay); this reads the request head and body once, then
forwards with ``httpx.AsyncClient`` streaming so response bytes pass
through chunk by chunk without ever being buffered in full.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import threading
import time
import urllib.parse
from typing import TYPE_CHECKING, Any

import httpx
from starlette.requests import Request

if TYPE_CHECKING:
    from .registry import EngineRecord
    from .service import Supervisor

logger = logging.getLogger("mtplx.supervisor")

# Request headers never forwarded to the engine: hop-by-hop per RFC 7230,
# plus content-length (httpx recomputes it from the body it sends).
_REQUEST_HOP_BY_HOP = {
    "host",
    "connection",
    "content-length",
    "transfer-encoding",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "upgrade",
}

# Response headers never passed back to the client: same set, since the
# supervisor is itself now the hop boundary for the response side too.
_RESPONSE_HOP_BY_HOP = _REQUEST_HOP_BY_HOP

_LOCALHOST_BINDS = {"", "127.0.0.1", "::1", "localhost"}

# Front-door and admin paths, lowercased. Matched against a normalized path
# (see ``normalize_path``) so a differently-cased or slash-doubled spelling
# still lands on the admin app instead of being forwarded to an engine as a
# literal path (SECURITY_REVIEW L1).
_FRONT_DOOR_EXACT_PATHS = ("/health", "/v1/models")
_ADMIN_PREFIX = "/mtplx/admin"

_MULTI_SLASH_RE = re.compile(r"/{2,}")

# Body size cap enforced before buffering (H1): a request larger than this
# gets a 413, never fully read into memory. Not currently CLI-configurable;
# raise this constant if a legitimate payload (e.g. large image/audio input)
# needs more room.
_MAX_BODY_BYTES = 32 * 1024 * 1024  # 32 MiB


def normalize_path(path: str) -> str:
    """Lowercase and collapse duplicate slashes, so ``/HEALTH`` and
    ``//v1//models`` match the same routes as ``/health`` and
    ``/v1/models``."""
    collapsed = _MULTI_SLASH_RE.sub("/", path or "/")
    return collapsed.lower()


def is_front_door_or_admin_path(path: str) -> bool:
    """True when a normalized path names ``/health``, ``/v1/models``, or
    anything under ``/mtplx/admin``."""
    return path in _FRONT_DOOR_EXACT_PATHS or path.startswith(_ADMIN_PREFIX)


def is_localhost_bind(host: str | None) -> bool:
    """True when ``host`` is a loopback bind address, same set as the
    engine's own ``_is_localhost_bind`` in ``mtplx/server/openai.py``."""
    return str(host or "").strip().lower().strip("[]") in _LOCALHOST_BINDS


class InsufficientMemory(Exception):
    """Raised by ``Supervisor.ensure_loaded`` when a model does not fit."""

    def __init__(self, admission: Any) -> None:
        self.admission = admission
        super().__init__(admission.reason)


class EngineUnavailable(Exception):
    """Raised when a JIT load fails, times out, or the engine dies."""


class ModelNotFound(Exception):
    """Raised when a model id is not registered at all."""


def request_api_key(request: Request) -> str | None:
    """Same header forms as the engine's ``_request_api_key``: ``Authorization:
    Bearer <key>`` or ``x-api-key``."""
    auth_header = request.headers.get("authorization", "")
    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() == "bearer" and token:
        return token
    api_key = request.headers.get("x-api-key")
    return api_key or None


def is_authorized(request: Request, configured_api_key: str | None) -> bool:
    """Same match semantics as the engine's ``_request_is_authorized``."""
    if not configured_api_key:
        return True
    candidate = request_api_key(request)
    return bool(candidate and secrets.compare_digest(candidate, configured_api_key))


class RateLimiter:
    """In-process sliding-window limiter, keyed per caller (client host).

    Not shared across processes or restarts; good enough to blunt a local
    abuser hammering the admin API or brute-forcing the key (M1). Mirrors
    the algorithm of ``mtplx/server/openai.py``'s ``_RateLimiter`` without
    importing that module, which would pull in the model runtime.
    """

    def __init__(self, limit_per_minute: int) -> None:
        self.limit_per_minute = limit_per_minute
        self._lock = threading.Lock()
        self._events: dict[str, list[float]] = {}

    def _trim(self, key: str, now: float) -> list[float]:
        window_start = now - 60.0
        events = [item for item in self._events.get(key, []) if item > window_start]
        self._events[key] = events
        return events

    def allow(self, key: str, *, now: float | None = None) -> tuple[bool, int]:
        """Record one hit against ``key``; returns ``(allowed, retry_after_s)``."""
        timestamp = time.monotonic() if now is None else float(now)
        with self._lock:
            events = self._trim(key, timestamp)
            if len(events) >= self.limit_per_minute:
                retry_after = max(1, int(60.0 - (timestamp - events[0])))
                return False, retry_after
            events.append(timestamp)
            return True, 0

    def is_over(self, key: str, *, now: float | None = None) -> tuple[bool, int]:
        """Read-only: is ``key`` already over budget, without recording a hit."""
        timestamp = time.monotonic() if now is None else float(now)
        with self._lock:
            events = self._trim(key, timestamp)
            if len(events) >= self.limit_per_minute:
                retry_after = max(1, int(60.0 - (timestamp - events[0])))
                return True, retry_after
            return False, 0


def client_host(request: Request) -> str:
    return request.client.host if request.client is not None else "unknown"


# Shared across ProxyApp and AdminApp so a host that racks up failed-auth
# attempts on one gets throttled on the other too (M1: "10/minute then 429
# on any route for that host"). Lives here, not in admin.py, so proxy.py
# never needs to import admin.py (admin.py already imports this module).
auth_fail_limiter = RateLimiter(10)


class _BodyTooLarge(Exception):
    """Raised by ``_read_body_capped`` when the body exceeds the cap."""


async def _read_body_capped(request: Request, limit: int) -> bytes:
    """Read the request body up to ``limit`` bytes, never buffering more.

    Checks ``Content-Length`` first as a cheap early-out, then still caps
    the actual read in case the header is absent, wrong, or chunked (H1).
    """
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > limit:
                raise _BodyTooLarge()
        except ValueError:
            pass
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise _BodyTooLarge()
        chunks.append(chunk)
    return b"".join(chunks)


def _json_error(status: int, code: str, message: str | None = None, **extra: Any) -> dict:
    error: dict[str, Any] = {"code": code}
    if message:
        error["message"] = message
    error.update(extra)
    return {"error": error}


async def _send_json(
    scope, receive, send, status: int, payload: dict, headers: dict | None = None
) -> None:
    from starlette.responses import JSONResponse

    response = JSONResponse(payload, status_code=status, headers=headers or {})
    await response(scope, receive, send)


def _extract_requested_model(method: str, body: bytes, query_string: bytes) -> str | None:
    if method in ("POST", "PUT", "PATCH") and body:
        try:
            data = json.loads(body)
        except ValueError:
            data = None
        if isinstance(data, dict):
            model = data.get("model")
            if isinstance(model, str) and model:
                return model
    if query_string:
        parsed = urllib.parse.parse_qs(query_string.decode("utf-8", "replace"))
        values = parsed.get("model")
        if values:
            return values[0]
    return None


class ProxyApp:
    """Front-door routing for everything that is not an admin route.

    ``/v1/*`` paths resolve a requested model (body for POST, query for
    GET) via ``registry.resolve``; every other path (dashboard, an
    engine's own ``/health``, etc.) always goes to the default engine.
    """

    def __init__(self, supervisor: "Supervisor") -> None:
        self.supervisor = supervisor

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            return
        request = Request(scope, receive=receive)
        method = str(scope.get("method") or "GET").upper()
        path = normalize_path(str(scope.get("path") or "/"))
        query_string = scope.get("query_string") or b""

        if is_front_door_or_admin_path(path):
            # The outer Supervisor dispatch only recognizes the exact/prefix
            # spelling of these paths; a case- or slash-varied spelling
            # (``/HEALTH``, ``//v1//models``) still names a front-door or
            # admin route, not a literal path to forward to an engine
            # (SECURITY_REVIEW L1).
            from .admin import AdminApp

            await AdminApp(self.supervisor)(scope, receive, send)
            return

        host_key = client_host(request)
        blocked, retry_after = auth_fail_limiter.is_over(host_key)
        if blocked:
            await _send_json(
                scope,
                receive,
                send,
                429,
                _json_error(429, "rate_limited", "too many failed auth attempts"),
                headers={"Retry-After": str(retry_after)},
            )
            return

        if not self._is_authorized(request):
            allowed, fail_retry_after = auth_fail_limiter.allow(host_key)
            if not allowed:
                await _send_json(
                    scope,
                    receive,
                    send,
                    429,
                    _json_error(429, "rate_limited", "too many failed auth attempts"),
                    headers={"Retry-After": str(fail_retry_after)},
                )
                return
            await _send_json(
                scope,
                receive,
                send,
                401,
                _json_error(401, "unauthorized", "missing or invalid API key"),
                headers={"WWW-Authenticate": "Bearer"},
            )
            return

        try:
            body = await _read_body_capped(request, _MAX_BODY_BYTES)
        except _BodyTooLarge:
            await _send_json(
                scope,
                receive,
                send,
                413,
                _json_error(
                    413,
                    "request_too_large",
                    f"request body exceeds {_MAX_BODY_BYTES} bytes",
                ),
            )
            return

        registry = self.supervisor.registry
        default_id = self.supervisor.default_model_id
        is_v1 = path.startswith("/v1/")

        if is_v1:
            requested = _extract_requested_model(method, body, query_string)
            record, kind = registry.resolve(requested, default_id, self.supervisor.config.strict_model)
        else:
            requested = None
            record, kind = registry.resolve(None, default_id, False)

        if kind == "draining":
            await _send_json(
                scope,
                receive,
                send,
                503,
                _json_error(503, "engine_draining"),
                headers={"Retry-After": "2"},
            )
            return

        if kind == "unknown" or record is None:
            await _send_json(scope, receive, send, 404, _json_error(404, "model_not_found"))
            return

        extra_headers: dict[str, str] = {}
        if is_v1 and kind == "fallback":
            extra_headers["x-mtplx-routed-model"] = record.spec.model_id

        model_id = record.spec.model_id

        if kind == "installed":
            try:
                record = await self.supervisor.ensure_loaded(model_id)
            except InsufficientMemory as exc:
                admission = exc.admission
                await _send_json(
                    scope,
                    receive,
                    send,
                    507,
                    _json_error(
                        507,
                        "insufficient_memory",
                        admission.reason,
                        needed_bytes=admission.needed_bytes,
                        available_bytes=admission.available_bytes,
                        would_free=admission.would_free,
                    ),
                )
                return
            except EngineUnavailable as exc:
                await _send_json(
                    scope,
                    receive,
                    send,
                    503,
                    _json_error(503, "engine_unavailable", str(exc)),
                    headers={"Retry-After": "5"},
                )
                return

        registry.pin(model_id)
        try:
            await self._forward(scope, receive, send, request, method, path, query_string, body, record, extra_headers)
        finally:
            registry.unpin(model_id)
            registry.touch(model_id)

    def _is_authorized(self, request: Request) -> bool:
        config = self.supervisor.config
        if config.insecure_lan or is_localhost_bind(config.host):
            return True
        return is_authorized(request, config.api_key)

    async def _forward(
        self,
        scope,
        receive,
        send,
        request: Request,
        method: str,
        path: str,
        query_string: bytes,
        body: bytes,
        record: "EngineRecord",
        extra_headers: dict[str, str],
    ) -> None:
        client = self.supervisor.http_client
        if client is None:
            await _send_json(scope, receive, send, 503, _json_error(503, "engine_unavailable", "supervisor not started"))
            return
        url = f"http://127.0.0.1:{record.port}{path}"
        if query_string:
            url += "?" + query_string.decode("utf-8", "replace")
        headers = [
            (name, value)
            for name, value in request.headers.items()
            if name.lower() not in _REQUEST_HOP_BY_HOP
        ]
        started = False
        try:
            async with client.stream(method, url, headers=headers, content=body) as response:
                out_headers = [
                    (name.decode("latin-1"), value.decode("latin-1"))
                    for name, value in response.headers.raw
                    if name.decode("latin-1").lower() not in _RESPONSE_HOP_BY_HOP
                ]
                out_headers.extend(extra_headers.items())
                await send(
                    {
                        "type": "http.response.start",
                        "status": response.status_code,
                        "headers": [(k.encode("latin-1"), v.encode("latin-1")) for k, v in out_headers],
                    }
                )
                started = True
                async for chunk in response.aiter_raw():
                    await send({"type": "http.response.body", "body": chunk, "more_body": True})
                await send({"type": "http.response.body", "body": b"", "more_body": False})
        except httpx.HTTPError as exc:
            if started:
                # ``http.response.start`` already went out: a second one
                # would break the ASGI connection (C2). Close the body
                # cleanly instead and let the client see a short read.
                logger.warning(
                    "engine connection dropped mid-stream for %s: %s", record.spec.model_id, exc
                )
                try:
                    await send({"type": "http.response.body", "body": b"", "more_body": False})
                except Exception:
                    pass
                return
            logger.warning("engine connection error for %s: %s", record.spec.model_id, exc)
            await _send_json(scope, receive, send, 503, _json_error(503, "engine_unavailable", str(exc)))
