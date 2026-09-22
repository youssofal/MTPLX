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
import secrets
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


def _json_error(status: int, code: str, message: str | None = None, **extra: Any) -> dict:
    error: dict[str, Any] = {"code": code}
    if message:
        error["message"] = message
    error.update(extra)
    return {"error": error}


async def _send_json(scope, receive, send, status: int, payload: dict, headers: dict | None = None) -> None:
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
        path = str(scope.get("path") or "/")
        query_string = scope.get("query_string") or b""
        body = await request.body()

        if not self._is_authorized(request):
            await _send_json(
                scope,
                receive,
                send,
                401,
                _json_error(401, "unauthorized", "missing or invalid API key"),
                headers={"WWW-Authenticate": "Bearer"},
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
                async for chunk in response.aiter_raw():
                    await send({"type": "http.response.body", "body": chunk, "more_body": True})
                await send({"type": "http.response.body", "body": b"", "more_body": False})
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError) as exc:
            logger.warning("engine connection error for %s: %s", record.spec.model_id, exc)
            await _send_json(scope, receive, send, 503, _json_error(503, "engine_unavailable", str(exc)))
