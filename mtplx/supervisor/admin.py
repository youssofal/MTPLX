"""Admin routes and the front-door ``/health`` / ``/v1/models`` endpoints.

Same pure-ASGI shape as ``proxy.py``: one callable, routed on path, JSON
responses built with Starlette's ``JSONResponse`` (not
``BaseHTTPMiddleware``).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from .proxy import (
    EngineUnavailable,
    InsufficientMemory,
    RateLimiter,
    auth_fail_limiter,
    client_host,
    is_authorized,
    is_localhost_bind,
    normalize_path,
)
from .registry import EngineState

if TYPE_CHECKING:
    from .service import Supervisor

logger = logging.getLogger("mtplx.supervisor")

_LOAD_POLL_S = 0.05

# Admin routes (``/mtplx/admin/*``) only; front-door ``/health``/``/v1/models``
# are not throttled by this one (M1).
admin_rate_limiter = RateLimiter(30)


def _error(
    status: int,
    code: str,
    message: str | None = None,
    *,
    headers: dict | None = None,
    **extra: Any,
) -> JSONResponse:
    error: dict[str, Any] = {"code": code}
    if message:
        error["message"] = message
    error.update(extra)
    return JSONResponse({"error": error}, status_code=status, headers=headers)


class AdminApp:
    """``/mtplx/admin/*``, front-door ``/health``, and ``/v1/models``."""

    def __init__(self, supervisor: "Supervisor") -> None:
        self.supervisor = supervisor

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            return
        request = Request(scope, receive=receive)
        path = normalize_path(str(scope.get("path") or "/"))
        method = str(scope.get("method") or "GET").upper()
        host_key = client_host(request)

        blocked, retry_after = auth_fail_limiter.is_over(host_key)
        if blocked:
            response = _error(
                429,
                "rate_limited",
                "too many failed auth attempts",
                headers={"Retry-After": str(retry_after)},
            )
            await response(scope, receive, send)
            return

        if path.startswith("/mtplx/admin"):
            allowed, admin_retry_after = admin_rate_limiter.allow(host_key)
            if not allowed:
                response = _error(
                    429,
                    "rate_limited",
                    "too many admin requests",
                    headers={"Retry-After": str(admin_retry_after)},
                )
                await response(scope, receive, send)
                return

        if path == "/health":
            response = await self._health(request)
        elif path == "/v1/models":
            response = await self._list_models(request, host_key)
        elif path == "/mtplx/admin/models" and method == "GET":
            response = await self._admin_models(request, host_key)
        elif path == "/mtplx/admin/load" and method == "POST":
            response = await self._load(request, host_key)
        elif path == "/mtplx/admin/unload" and method == "POST":
            response = await self._unload(request, host_key)
        elif path == "/mtplx/admin/restart" and method == "POST":
            response = await self._restart(request, host_key)
        elif path == "/mtplx/admin/health" and method == "GET":
            response = await self._admin_health(request, host_key)
        else:
            response = _error(404, "not_found")

        await response(scope, receive, send)

    # -- auth -----------------------------------------------------------

    def _unauthorized(self, host_key: str, *, message: str) -> JSONResponse:
        allowed, retry_after = auth_fail_limiter.allow(host_key)
        if not allowed:
            return _error(
                429,
                "rate_limited",
                "too many failed auth attempts",
                headers={"Retry-After": str(retry_after)},
            )
        return _error(401, "unauthorized", message)

    def _admin_authorized(self, request: Request, host_key: str) -> JSONResponse | None:
        """Returns an error response if unauthorized, else None."""
        config = self.supervisor.config
        if not config.api_key:
            return _error(401, "unauthorized", "admin API needs --api-key-file")
        if not is_authorized(request, config.api_key):
            return self._unauthorized(host_key, message="missing or invalid API key")
        return None

    def _inference_authorized(self, request: Request, host_key: str) -> JSONResponse | None:
        config = self.supervisor.config
        if config.insecure_lan or is_localhost_bind(config.host):
            return None
        if not is_authorized(request, config.api_key):
            return self._unauthorized(host_key, message="missing or invalid API key")
        return None

    # -- front door -------------------------------------------------------

    async def _health(self, request: Request) -> JSONResponse:
        # /health always answers (even with no or a wrong key): daemon
        # ownership checks (mtplx.daemon_client.fetch_daemon_health) need
        # ``ok``/``model``/``startup`` unauthenticated. What varies with the
        # admin key is how much of the engine topology it discloses (M2).
        config = self.supervisor.config
        admin_ok = bool(config.api_key) and is_authorized(request, config.api_key)
        payload = {
            "ok": True,
            "model": self.supervisor.default_model_id,
            "startup": {"pid": os.getpid(), "launch_id": config.launch_id},
        }
        if admin_ok:
            payload["supervisor"] = self.supervisor.snapshot()
        else:
            payload["supervisor"] = {
                "engines": [
                    {"model": record.spec.model_id, "state": record.state.value}
                    for record in self.supervisor.registry.records()
                ]
            }
        if config.insecure_lan:
            payload["insecure_lan"] = True
        return JSONResponse(payload)

    async def _list_models(self, request: Request, host_key: str) -> JSONResponse:
        unauthorized = self._inference_authorized(request, host_key)
        if unauthorized is not None:
            return unauthorized
        data = [
            {
                "id": record.spec.model_id,
                "object": "model",
                "owned_by": "mtplx",
                "loaded": record.state == EngineState.READY,
                "state": record.state.value,
                # ``aliases`` lands on EngineRecord alongside registry
                # add_alias (R1b); default to empty until it does.
                "aliases": list(getattr(record, "aliases", None) or []),
            }
            for record in self.supervisor.registry.records()
        ]
        return JSONResponse({"object": "list", "data": data})

    # -- admin ------------------------------------------------------------

    async def _admin_models(self, request: Request, host_key: str) -> JSONResponse:
        unauthorized = self._admin_authorized(request, host_key)
        if unauthorized is not None:
            return unauthorized
        records = self.supervisor.registry.records()
        loaded = [r.spec.model_id for r in records if r.state == EngineState.READY]
        installed = [r.spec.model_id for r in records if r.state != EngineState.READY]
        snapshot = self.supervisor.snapshot()
        return JSONResponse({"loaded": loaded, "installed": installed, "budget": snapshot["budget"]})

    async def _admin_health(self, request: Request, host_key: str) -> JSONResponse:
        unauthorized = self._admin_authorized(request, host_key)
        if unauthorized is not None:
            return unauthorized
        return JSONResponse(self.supervisor.snapshot())

    async def _body_model_id(self, request: Request) -> str | None:
        try:
            data = await request.json()
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        model_id = data.get("model")
        return model_id if isinstance(model_id, str) and model_id else None

    async def _load(self, request: Request, host_key: str) -> JSONResponse:
        unauthorized = self._admin_authorized(request, host_key)
        if unauthorized is not None:
            return unauthorized
        model_id = await self._body_model_id(request)
        if not model_id:
            return _error(400, "bad_request", "missing 'model'")
        record = self.supervisor.registry.get(model_id)
        if record is None:
            return _error(404, "model_not_found")
        if record.state == EngineState.READY:
            return JSONResponse({"model": model_id, "state": "ready"}, status_code=200)

        task = asyncio.ensure_future(self.supervisor.ensure_loaded(model_id))
        self.supervisor.track_background(task)
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=_LOAD_POLL_S)
        except asyncio.TimeoutError:
            return JSONResponse({"model": model_id, "state": "loading"}, status_code=202)
        except InsufficientMemory as exc:
            admission = exc.admission
            return _error(
                507,
                "insufficient_memory",
                admission.reason,
                needed_bytes=admission.needed_bytes,
                available_bytes=admission.available_bytes,
                would_free=admission.would_free,
            )
        except EngineUnavailable as exc:
            return _error(503, "engine_unavailable", str(exc))
        else:
            return JSONResponse({"model": model_id, "state": "ready"}, status_code=200)

    async def _unload(self, request: Request, host_key: str) -> JSONResponse:
        unauthorized = self._admin_authorized(request, host_key)
        if unauthorized is not None:
            return unauthorized
        model_id = await self._body_model_id(request)
        if not model_id:
            return _error(400, "bad_request", "missing 'model'")
        record = self.supervisor.registry.get(model_id)
        if record is None:
            return _error(404, "model_not_found")
        if model_id == self.supervisor.default_model_id and not self.supervisor.config.unload_default:
            return _error(409, "cannot_unload_default")
        await self.supervisor.unload(model_id)
        return JSONResponse({"model": model_id, "state": "stopped"})

    async def _restart(self, request: Request, host_key: str) -> JSONResponse:
        unauthorized = self._admin_authorized(request, host_key)
        if unauthorized is not None:
            return unauthorized
        model_id = await self._body_model_id(request)
        if not model_id:
            return _error(400, "bad_request", "missing 'model'")
        if self.supervisor.registry.get(model_id) is None:
            return _error(404, "model_not_found")
        try:
            record = await self.supervisor.restart(model_id)
        except InsufficientMemory as exc:
            admission = exc.admission
            return _error(
                507,
                "insufficient_memory",
                admission.reason,
                needed_bytes=admission.needed_bytes,
                available_bytes=admission.available_bytes,
                would_free=admission.would_free,
            )
        except EngineUnavailable as exc:
            return _error(503, "engine_unavailable", str(exc))
        return JSONResponse({"model": model_id, "state": record.state.value})
