"""Just-in-time model loading for ``mtplx serve --jit``.

One MTPLX server process owns one model: its Metal buffers are wired, its
family patches are process-global, and a 20-70 GB pack cannot be released
reliably inside a live interpreter. So switching models means switching
processes. This module is a small front door that owns the public port,
reads the ``model`` field of each request, and keeps exactly one ordinary
``mtplx serve`` child running for the model that was asked for. A request
for a different model drains the requests in flight, stops the old child
(whole process group, so the wrapper cannot orphan the wired server),
starts the new one through the same ``mtplx serve --model`` path a user
would type, and replays the request against it.

Stdlib only, like ``daemon_client``: the router must run in every install
flavor and must never import MLX.
"""

from __future__ import annotations

import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

# Bodies that carry a ``model`` the router must route on.
MODEL_ROUTED_PATHS = frozenset(
    {
        "/v1/chat/completions",
        "/v1/responses",
        "/v1/completions",
        "/v1/messages",
        "/v1/messages/count_tokens",
    }
)
# Cadence of SSE comments while a streaming request waits for a model load,
# well under the 120 s client inactivity timeout and the 5 s the server itself
# uses before its first token.
KEEPALIVE_INTERVAL_S = 5.0
CHILD_READY_TIMEOUT_S = 900.0
CHILD_STOP_GRACE_S = 20.0
_CATALOG_TTL_S = 10.0
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "content-length",
    }
)
# serve flags the router owns and must not forward to a child. Every one
# takes a value except the ones in _JIT_VALUELESS_FLAGS.
_JIT_STRIPPED_VALUE_FLAGS = frozenset(
    {
        "--model",
        "--model-id",
        "--host",
        "--port",
        "--jit-models",
        "--jit-idle-unload",
        "--api-key",
        "--api-key-file",
    }
)
_JIT_VALUELESS_FLAGS = frozenset({"--jit", "--open-browser", "--dry-run"})


class JitError(Exception):
    """A request the router can answer without touching a child."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@dataclass(frozen=True)
class JitModel:
    """One model the router can load, as a client names it."""

    model_id: str
    ref: str
    peak_memory_gib: float | None
    names: frozenset[str]


def _key(value: str) -> str:
    return value.strip().lower().replace("\\", "/").replace("--", "/")


def discover_models(
    *,
    cache_dir: str | None = None,
    search_dirs: Iterable[str] | None = None,
    allow: Iterable[str] | None = None,
) -> list[JitModel]:
    """Complete local installs, optionally limited to an allowlist.

    Only models that are on disk are listed: advertising a model the daemon
    cannot load ends in a 400 or a download in the middle of a request.
    """

    from mtplx.default_models import public_model_id_for_ref
    from mtplx.model_catalog import catalog_model_matching, scan_installed_models

    allowed: set[str] | None = None
    if allow:
        allowed = set()
        for ref in allow:
            catalog = catalog_model_matching(ref)
            allowed.add(_key(catalog.hf_model_id if catalog else ref))

    models: list[JitModel] = []
    for installed in scan_installed_models(cache_dir, search_dirs=search_dirs):
        catalog = installed.catalog
        model_id = catalog.hf_model_id if catalog else installed.name
        if allowed is not None and _key(model_id) not in allowed:
            continue
        names = {_key(model_id), _key(installed.name), _key(str(installed.path))}
        names.add(_key(public_model_id_for_ref(installed.path)))
        if catalog is not None:
            names.update(
                {
                    _key(catalog.id),
                    _key(catalog.display_name),
                    *(_key(alias) for alias in catalog.aliases),
                    _key(public_model_id_for_ref(catalog.hf_model_id)),
                }
            )
        models.append(
            JitModel(
                model_id=model_id,
                ref=str(installed.path),
                peak_memory_gib=catalog.peak_memory_gib if catalog else None,
                names=frozenset(names),
            )
        )
    return models


def child_serve_argv(
    raw_argv: Iterable[str], *, model_ref: str, port: int
) -> list[str]:
    """``mtplx serve`` for one model: the user's flags minus the router's."""

    passthrough: list[str] = []
    tokens = list(raw_argv)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        head = token.split("=", 1)[0]
        if head in _JIT_STRIPPED_VALUE_FLAGS:
            if "=" not in token:
                index += 1
            continue
        if token in _JIT_VALUELESS_FLAGS:
            continue
        passthrough.append(token)
    if passthrough and passthrough[0] == "serve":
        passthrough = passthrough[1:]
    return [
        sys.executable,
        "-P",
        "-m",
        "mtplx.cli",
        "serve",
        "--model",
        model_ref,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        *passthrough,
    ]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _total_ram_gib() -> float | None:
    try:
        from mtplx.memory_plan import detect_total_ram_bytes

        total = detect_total_ram_bytes()
    except Exception:
        return None
    return float(total) / 1_073_741_824.0 if total else None


class ModelSupervisor:
    """Keeps at most one ``mtplx serve`` child, swapping it on demand."""

    def __init__(
        self,
        *,
        raw_argv: list[str],
        discover: Callable[[], list[JitModel]],
        default_ref: str | None = None,
        spawn: Callable[[list[str]], subprocess.Popen[Any]] | None = None,
        ready_timeout_s: float = CHILD_READY_TIMEOUT_S,
        stop_grace_s: float = CHILD_STOP_GRACE_S,
        ram_gib: float | None = None,
        idle_unload_s: float = 0.0,
    ) -> None:
        self._raw_argv = raw_argv
        self._discover = discover
        self._default_ref = default_ref
        self._spawn = spawn or self._spawn_child
        self._ready_timeout_s = ready_timeout_s
        self._stop_grace_s = stop_grace_s
        self._ram_gib = ram_gib if ram_gib is not None else _total_ram_gib()
        self._cond = threading.Condition()
        self._catalog: list[JitModel] = []
        self._catalog_at = 0.0
        self._current: JitModel | None = None
        self._proc: subprocess.Popen[Any] | None = None
        self._port: int | None = None
        self._leases = 0
        # A swap in progress blocks new leases on the old model too, so a
        # steady stream of requests for it cannot starve the swap.
        self._swapping_to: str | None = None
        self._idle_unload_s = max(0.0, float(idle_unload_s))
        self._last_active = time.monotonic()
        self._reaper_stop = threading.Event()
        if self._idle_unload_s > 0:
            threading.Thread(
                target=self._reap_idle, name="mtplx-jit-reaper", daemon=True
            ).start()

    # -- model resolution -------------------------------------------------

    def models(self, *, refresh: bool = False) -> list[JitModel]:
        with self._cond:
            if refresh or time.monotonic() - self._catalog_at > _CATALOG_TTL_S:
                self._catalog = self._discover()
                self._catalog_at = time.monotonic()
            return list(self._catalog)

    def resolve(self, requested: str | None) -> JitModel:
        wanted = (requested or "").strip()
        if not wanted:
            wanted = self._current.model_id if self._current else (self._default_ref or "")
        if not wanted:
            raise JitError(
                400,
                "model_required",
                "this server loads models on demand, so the request must name one "
                "in its 'model' field; see GET /v1/models",
            )
        key = _key(wanted)
        for refresh in (False, True):
            for model in self.models(refresh=refresh):
                if key in model.names:
                    return model
        available = ", ".join(model.model_id for model in self.models()) or "none"
        raise JitError(
            404,
            "model_not_found",
            f"model {wanted!r} is not installed; installed models: {available}. "
            "Install one with: mtplx pull <repo>",
        )

    # -- state ------------------------------------------------------------

    @property
    def current(self) -> JitModel | None:
        with self._cond:
            return self._current if self._alive() else None

    @property
    def port(self) -> int | None:
        with self._cond:
            return self._port if self._alive() else None

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # -- leases -----------------------------------------------------------

    def acquire(
        self, model: JitModel, *, tick: Callable[[], None] | None = None
    ) -> int:
        """Load ``model`` if needed and lease it; returns the child's port.

        ``tick`` runs about every KEEPALIVE_INTERVAL_S while this waits, so a
        streaming caller can keep its connection warm.
        """

        next_tick = time.monotonic() + KEEPALIVE_INTERVAL_S
        with self._cond:
            while True:
                if (
                    self._alive()
                    and self._current == model
                    and self._swapping_to in (None, model.model_id)
                ):
                    self._leases += 1
                    assert self._port is not None
                    return self._port
                if self._swapping_to is None and self._leases == 0:
                    self._swapping_to = model.model_id
                    break
                self._cond.wait(timeout=max(0.05, next_tick - time.monotonic()))
                if tick is not None and time.monotonic() >= next_tick:
                    next_tick = time.monotonic() + KEEPALIVE_INTERVAL_S
                    self._cond.release()
                    try:
                        tick()
                    finally:
                        self._cond.acquire()
        try:
            self._swap_to(model, tick)
        except BaseException:
            with self._cond:
                self._swapping_to = None
                self._cond.notify_all()
            raise
        with self._cond:
            self._swapping_to = None
            self._leases += 1
            self._cond.notify_all()
            assert self._port is not None
            return self._port

    def release(self) -> None:
        with self._cond:
            self._leases = max(0, self._leases - 1)
            self._last_active = time.monotonic()
            self._cond.notify_all()

    # -- swap -------------------------------------------------------------

    def _swap_to(self, model: JitModel, tick: Callable[[], None] | None) -> None:
        self._check_fits(model)
        self._stop_child()
        port = _free_port()
        argv = child_serve_argv(self._raw_argv, model_ref=model.ref, port=port)
        print(f"[mtplx-jit] loading {model.model_id}", file=sys.stderr, flush=True)
        proc = self._spawn(argv)
        try:
            self._wait_ready(proc, port, tick)
        except BaseException:
            self._terminate(proc)
            raise
        with self._cond:
            self._proc, self._port, self._current = proc, port, model
            self._last_active = time.monotonic()
        print(f"[mtplx-jit] {model.model_id} ready", file=sys.stderr, flush=True)

    def _check_fits(self, model: JitModel) -> None:
        # Checked before the working model is stopped: a refused swap must
        # leave the previous model serving.
        if (
            self._ram_gib is not None
            and model.peak_memory_gib is not None
            and model.peak_memory_gib > self._ram_gib
        ):
            raise JitError(
                503,
                "model_does_not_fit",
                f"{model.model_id} needs about {model.peak_memory_gib:.0f} GiB "
                f"and this Mac has {self._ram_gib:.0f} GiB",
            )

    @staticmethod
    def _spawn_child(argv: list[str]) -> subprocess.Popen[Any]:
        # Own session: the `mtplx serve` wrapper and the server module it
        # spawns die together on killpg. Killing the wrapper alone leaves the
        # module holding the whole model wired.
        return subprocess.Popen(argv, start_new_session=True)

    def _wait_ready(
        self,
        proc: subprocess.Popen[Any],
        port: int,
        tick: Callable[[], None] | None,
    ) -> None:
        deadline = time.monotonic() + self._ready_timeout_s
        next_tick = time.monotonic() + KEEPALIVE_INTERVAL_S
        while True:
            code = proc.poll()
            if code is not None:
                raise JitError(
                    502,
                    "model_failed_to_load",
                    f"the model server exited with status {code} while loading; "
                    "its output is in the router's terminal",
                )
            if _child_healthy(port):
                return
            if time.monotonic() > deadline:
                raise JitError(
                    504,
                    "model_load_timeout",
                    f"the model did not become ready within {self._ready_timeout_s:.0f}s",
                )
            if tick is not None and time.monotonic() >= next_tick:
                tick()
                next_tick = time.monotonic() + KEEPALIVE_INTERVAL_S
            time.sleep(0.5)

    def _reap_idle(self) -> None:
        interval = min(5.0, max(0.05, self._idle_unload_s / 4))
        while not self._reaper_stop.wait(interval):
            with self._cond:
                if (
                    not self._alive()
                    or self._leases
                    or self._swapping_to is not None
                    or time.monotonic() - self._last_active < self._idle_unload_s
                ):
                    continue
                # Claim the swap slot so a request arriving mid-stop waits
                # for the unload, then loads the model afresh.
                self._swapping_to = "<idle-unload>"
                name = self._current.model_id if self._current else "model"
            print(
                f"[mtplx-jit] unloading {name} after {self._idle_unload_s:.0f}s idle",
                file=sys.stderr,
                flush=True,
            )
            try:
                self._stop_child()
            finally:
                with self._cond:
                    self._swapping_to = None
                    self._cond.notify_all()

    def _stop_child(self) -> None:
        with self._cond:
            proc, self._proc, self._port, self._current = self._proc, None, None, None
        if proc is not None:
            self._terminate(proc)

    def _terminate(self, proc: subprocess.Popen[Any]) -> None:
        if proc.poll() is None:
            self._signal_group(proc, signal.SIGTERM)
            try:
                proc.wait(timeout=self._stop_grace_s)
            except subprocess.TimeoutExpired:
                self._signal_group(proc, signal.SIGKILL)
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    pass

    @staticmethod
    def _signal_group(proc: subprocess.Popen[Any], sig: int) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            try:
                proc.send_signal(sig)
            except ProcessLookupError:
                pass

    def shutdown(self) -> None:
        self._reaper_stop.set()
        self._stop_child()


def _child_healthy(port: int) -> bool:
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
        try:
            conn.request("GET", "/health")
            return conn.getresponse().status == 200
        finally:
            conn.close()
    except OSError:
        return False


def _error_body(status: int, code: str, message: str) -> bytes:
    return json.dumps(
        {"error": {"message": message, "type": code, "code": code}}
    ).encode("utf-8")


def make_handler(
    supervisor: ModelSupervisor, *, api_key: str | None
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "mtplx-jit"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        # -- plumbing -----------------------------------------------------

        def _authorized(self) -> bool:
            if not api_key:
                return True
            bearer = self.headers.get("Authorization", "")
            supplied = bearer[7:] if bearer.lower().startswith("bearer ") else None
            supplied = supplied or self.headers.get("x-api-key")
            return supplied == api_key

        def _send_json(self, status: int, payload: Any) -> None:
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, error: JitError) -> None:
            self._send_json(error.status, _error_body(error.status, error.code, error.message))

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if length else b""

        # -- routes -------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch(b"")

        def do_DELETE(self) -> None:  # noqa: N802
            self._dispatch(b"")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch(self._read_body())

        def _dispatch(self, body: bytes) -> None:
            path = self.path.split("?", 1)[0]
            # /health stays open like the server's own, so supervisors and
            # `mtplx` can probe the daemon without a key.
            if path != "/health" and not self._authorized():
                self._send_json(401, _error_body(401, "invalid_api_key", "invalid API key"))
                return
            try:
                if self.command == "GET" and path == "/v1/models":
                    self._models()
                elif self.command == "GET" and path == "/health":
                    self._health()
                elif self.command == "POST" and path in MODEL_ROUTED_PATHS:
                    self._routed(path, body)
                else:
                    self._passthrough_current(body)
            except JitError as error:
                self._send_error(error)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

        def _models(self) -> None:
            query = self.path.partition("?")[2]
            capability = next(
                (p.split("=", 1)[1] for p in query.split("&") if p.startswith("capability=")),
                None,
            )
            if capability not in (None, "chat"):
                self._passthrough_current(b"")
                return
            port = supervisor.port
            loaded = supervisor.current
            child_entry: dict[str, Any] = {}
            if port is not None:
                child_entry = _child_chat_entry(port)
            now = int(time.time())
            data = []
            for model in supervisor.models(refresh=True):
                entry: dict[str, Any] = {
                    "id": model.model_id,
                    "object": "model",
                    "created": now,
                    "owned_by": "mtplx",
                    "capability": "chat",
                    "loaded": loaded == model,
                }
                if loaded == model:
                    for field in (
                        "supports_vision",
                        "modalities",
                        "context_length",
                        "max_context_length",
                        "max_model_len",
                    ):
                        if field in child_entry:
                            entry[field] = child_entry[field]
                data.append(entry)
            self._send_json(200, {"object": "list", "data": data})

        def _health(self) -> None:
            loaded = supervisor.current
            port = supervisor.port
            payload: dict[str, Any] = {}
            if port is not None:
                payload = _child_json(port, "/health") or {}
            payload.setdefault("ok", True)
            payload.setdefault("model", loaded.model_id if loaded else None)
            payload["jit"] = {
                "enabled": True,
                "loaded_model": loaded.model_id if loaded else None,
                "router_pid": os.getpid(),
            }
            self._send_json(200, payload)

        def _passthrough_current(self, body: bytes) -> None:
            port = supervisor.port
            if port is None:
                raise JitError(
                    503,
                    "no_model_loaded",
                    "no model is loaded yet; send a request that names one",
                )
            self._proxy(port, body, committed=False)

        def _routed(self, path: str, body: bytes) -> None:
            try:
                parsed = json.loads(body) if body else {}
            except ValueError:
                parsed = {}
            requested = parsed.get("model") if isinstance(parsed, dict) else None
            streaming = bool(isinstance(parsed, dict) and parsed.get("stream"))
            model = supervisor.resolve(requested if isinstance(requested, str) else None)
            committed = False

            def tick() -> None:
                # A client that hung up must not abort a load other
                # requests are waiting on.
                if committed:
                    try:
                        self.wfile.write(b": loading model\n\n")
                        self.wfile.flush()
                    except OSError:
                        pass

            if streaming:
                # Open the event stream now: a model load takes far longer
                # than a client waits for response headers.
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                committed = True
                self.wfile.write(b": loading model\n\n")
                self.wfile.flush()
            try:
                port = supervisor.acquire(model, tick=tick)
            except JitError as error:
                if not committed:
                    raise
                self._stream_error(path, error.code, error.message)
                return
            try:
                self._proxy(port, body, committed=committed, path=path)
            finally:
                supervisor.release()

        def _stream_error(self, path: str, code: str, message: str) -> None:
            payload = json.dumps({"error": {"message": message, "type": code, "code": code}})
            if path.startswith("/v1/messages"):
                frame = f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': code, 'message': message}})}\n\n"
            else:
                frame = f"data: {payload}\n\ndata: [DONE]\n\n"
            self.wfile.write(frame.encode("utf-8"))
            self.wfile.flush()

        def _proxy(
            self, port: int, body: bytes, *, committed: bool, path: str = ""
        ) -> None:
            headers = {
                k: v
                for k, v in self.headers.items()
                if k.lower() not in _HOP_BY_HOP and k.lower() != "host"
            }
            if body:
                headers["Content-Length"] = str(len(body))
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=None)
            try:
                try:
                    conn.request(self.command, self.path, body=body or None, headers=headers)
                    upstream = conn.getresponse()
                except OSError as exc:
                    error = JitError(502, "model_unreachable", f"model server unreachable: {exc}")
                    if committed:
                        self._stream_error(path, error.code, error.message)
                        return
                    raise error from exc
                if committed:
                    if upstream.status != 200:
                        detail = upstream.read().decode("utf-8", "replace")
                        self._stream_error(path, "upstream_error", detail)
                        return
                else:
                    self.send_response(upstream.status)
                    for name, value in upstream.getheaders():
                        if name.lower() not in _HOP_BY_HOP:
                            self.send_header(name, value)
                    length = upstream.getheader("Content-Length")
                    if length is not None:
                        self.send_header("Content-Length", length)
                    else:
                        self.send_header("Connection", "close")
                        self.close_connection = True
                    self.end_headers()
                while True:
                    chunk = upstream.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            finally:
                conn.close()

    return Handler


def _child_json(port: int, path: str) -> dict[str, Any] | None:
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
        try:
            conn.request("GET", path)
            response = conn.getresponse()
            data = json.loads(response.read())
            return data if isinstance(data, dict) else None
        finally:
            conn.close()
    except (OSError, ValueError):
        return None


def _child_chat_entry(port: int) -> dict[str, Any]:
    listing = _child_json(port, "/v1/models") or {}
    for entry in listing.get("data") or []:
        if isinstance(entry, dict) and entry.get("capability") == "chat":
            return entry
    return {}


def run_jit_router(args: Any, raw_argv: list[str]) -> int:
    """Serve the router on ``--host``/``--port`` until interrupted."""

    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 8000))
    # cmd_serve_public already resolved the key (flag, file, env) onto args.
    api_key = getattr(args, "api_key", None) or None
    allow = [m for m in str(getattr(args, "jit_models", "") or "").split(",") if m.strip()]
    cache_dir = getattr(args, "cache_dir", None)
    search_dirs = getattr(args, "model_search_dirs", None)
    cli_flags = getattr(args, "_cli_flags", set()) or set()
    default_ref = str(args.model) if "model" in cli_flags else None
    # The child cannot see the router's api key (it binds loopback and the
    # router is its only intended caller), so a key-protected router is
    # the single place the key is checked.
    supervisor = ModelSupervisor(
        raw_argv=raw_argv,
        discover=lambda: discover_models(
            cache_dir=cache_dir, search_dirs=search_dirs, allow=allow or None
        ),
        default_ref=default_ref,
        idle_unload_s=float(getattr(args, "jit_idle_unload", 0) or 0),
    )
    server = ThreadingHTTPServer((host, port), make_handler(supervisor, api_key=api_key))
    server.daemon_threads = True

    def stop(*_: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    installed = supervisor.models(refresh=True)
    print(
        f"[mtplx-jit] listening on http://{host}:{port} "
        f"({len(installed)} installed model(s); loaded on first request)",
        file=sys.stderr,
        flush=True,
    )
    for model in installed:
        print(f"[mtplx-jit]   {model.model_id}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        supervisor.shutdown()
    return 0
