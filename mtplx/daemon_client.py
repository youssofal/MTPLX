"""Client-side helpers for cooperating with an already-running MTPLX daemon.

``mtplx start`` historically assumed it owned the machine: it loaded models
in-process and treated a busy port as a fatal error. With the macOS app
keeping a daemon alive, the CLI must instead detect that daemon, attach to
it for chat, classify who actually owns a busy port, and stop daemons
cleanly. Everything here is stdlib-only (urllib/http.client) because these
paths must work in every install flavor, including the minimal venv the app
bootstraps.
"""

from __future__ import annotations

import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping

# The app daemon's default port plus the per-target defaults `mtplx start`
# assigns (opencode/swival/hermes).
DAEMON_PROBE_PORTS: tuple[int, ...] = (8000, 18083, 18084, 18085)
ATTACH_PROBE_ENV = "MTPLX_START_ATTACH_PROBE"
_PROBE_DISABLED_VALUES = frozenset({"0", "off", "no", "false", "disabled"})
# Attached chat runs against a daemon on this machine, so the TCP connect
# either succeeds at once or the daemon is gone.
ATTACH_CHAT_CONNECT_TIMEOUT_S = 5.0
# A healthy streamed request never goes quiet for long: the server writes an
# SSE keep-alive comment every 5 s before the first token (10 s progress
# chunks fill any later gap), and its own stall watchdog fails a frozen model
# owner with an error frame well inside 300 s. Only a daemon that is wedged
# (or a dead connection) puts nothing on the wire for this long, so waiting
# it out never trips on a slow prefill of a large prompt.
ATTACH_CHAT_INACTIVITY_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class RunningDaemon:
    """A healthy MTPLX server discovered over ``/health``."""

    host: str
    port: int
    model: str | None
    model_path: str | None
    launch_id: str | None
    pid: int | None
    api_key_required: bool
    health: Mapping[str, Any]

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def owned_by_app(self) -> bool:
        return bool(self.launch_id)

    @property
    def owner_label(self) -> str:
        return "the MTPLX app" if self.owned_by_app else "another MTPLX server"


def _connect_host(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "::"} else host


def port_is_open(host: str, port: int, *, timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection((_connect_host(host), int(port)), timeout=timeout):
            return True
    except OSError:
        return False


@dataclass(frozen=True)
class _HealthProbeResult:
    daemon: RunningDaemon | None = None
    auth_required: bool = False


def _probe_daemon_health(
    host: str,
    port: int,
    *,
    timeout: float = 1.5,
    api_key: str | None = None,
) -> _HealthProbeResult:
    """Query ``/health`` on ``host:port`` and preserve auth rejections."""

    url = f"http://{_connect_host(host)}:{int(port)}/health"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        with suppress(Exception):
            exc.close()
        return _HealthProbeResult(auth_required=exc.code in (401, 403))
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        OSError,
    ):
        return _HealthProbeResult()
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return _HealthProbeResult()
    startup = payload.get("startup")
    startup = startup if isinstance(startup, dict) else {}
    launch_id = startup.get("launch_id")
    pid = startup.get("pid")
    return _HealthProbeResult(
        daemon=RunningDaemon(
            host=_connect_host(host),
            port=int(port),
            model=(str(payload.get("model")) if payload.get("model") else None),
            model_path=(
                str(payload.get("model_path")) if payload.get("model_path") else None
            ),
            launch_id=(str(launch_id) if launch_id else None),
            pid=(int(pid) if isinstance(pid, int) else None),
            api_key_required=bool(startup.get("api_key_required")),
            health=payload,
        )
    )


def fetch_daemon_health(
    host: str,
    port: int,
    *,
    timeout: float = 1.5,
    api_key: str | None = None,
) -> RunningDaemon | None:
    """Return the daemon behind ``host:port`` or ``None`` when it is not one."""

    return _probe_daemon_health(host, port, timeout=timeout, api_key=api_key).daemon


def attach_probe_enabled() -> bool:
    raw = str(os.environ.get(ATTACH_PROBE_ENV) or "").strip().lower()
    return raw not in _PROBE_DISABLED_VALUES


def probe_running_daemons(
    *,
    host: str = "127.0.0.1",
    ports: tuple[int, ...] = DAEMON_PROBE_PORTS,
    timeout: float = 1.0,
    api_key: str | None = None,
) -> list[RunningDaemon]:
    """All healthy MTPLX daemons on the given ports, probe order preserved."""

    daemons: list[RunningDaemon] = []
    for port in ports:
        if not port_is_open(host, port):
            continue
        daemon = fetch_daemon_health(host, port, timeout=timeout, api_key=api_key)
        if daemon is not None:
            daemons.append(daemon)
    return daemons


def _app_persisted_port() -> int | None:
    """The MTPLX app's persisted daemon port, if any.

    The app's port preflight can bump the daemon off 8000 and persist
    the new port; CLI lanes that only probe the static defaults then
    miss a perfectly attachable daemon (QA-121/QA-122).
    """

    try:
        from mtplx.app_settings import read_app_settings

        port = int(getattr(read_app_settings(), "port", 0) or 0)
        return port if port > 0 else None
    except Exception:
        return None


def default_probe_ports() -> tuple[int, ...]:
    """Static probe ports plus the app's persisted (possibly bumped) port."""

    app_port = _app_persisted_port()
    if app_port and app_port not in DAEMON_PROBE_PORTS:
        return (app_port, *DAEMON_PROBE_PORTS)
    return DAEMON_PROBE_PORTS


def detect_attachable_daemon(
    *,
    host: str = "127.0.0.1",
    ports: tuple[int, ...] | None = None,
    api_key: str | None = None,
) -> RunningDaemon | None:
    """First attachable daemon, honoring the ``MTPLX_START_ATTACH_PROBE`` kill switch."""

    if not attach_probe_enabled():
        return None
    if ports is None:
        ports = default_probe_ports()
    daemons = probe_running_daemons(host=host, ports=ports, api_key=api_key)
    return daemons[0] if daemons else None


# ---------- port classification ---------------------------------------------

PORT_FREE = "free"
PORT_APP_DAEMON = "app_daemon"
PORT_MTPLX_SERVER = "mtplx_server"
PORT_AUTH_REQUIRED = "auth_required"
PORT_FOREIGN = "foreign"


@dataclass(frozen=True)
class PortOccupant:
    kind: str  # free | app_daemon | mtplx_server | auth_required | foreign
    daemon: RunningDaemon | None = None


def classify_port_occupant(
    host: str,
    port: int,
    *,
    timeout: float = 1.5,
    api_key: str | None = None,
) -> PortOccupant:
    """Identify who owns a port so error copy can be actionable.

    ``app_daemon``: a healthy MTPLX server launched by the macOS app
    (``startup.launch_id`` present). ``mtplx_server``: a healthy MTPLX
    server without a launch id (an older CLI ``mtplx serve`` terminal).
    ``auth_required``: a server on the port answered HTTP 401/403 to ``/health``.
    ``foreign``: something is listening but does not speak MTPLX health.
    """

    if not port_is_open(host, port):
        return PortOccupant(kind=PORT_FREE)
    probe = _probe_daemon_health(host, port, timeout=timeout, api_key=api_key)
    if probe.auth_required:
        return PortOccupant(kind=PORT_AUTH_REQUIRED)
    daemon = probe.daemon
    if daemon is None:
        return PortOccupant(kind=PORT_FOREIGN)
    if daemon.owned_by_app:
        return PortOccupant(kind=PORT_APP_DAEMON, daemon=daemon)
    return PortOccupant(kind=PORT_MTPLX_SERVER, daemon=daemon)


def app_configured_port() -> int | None:
    """The port the macOS app has persisted, or None when there is no app
    setting. Public alias so caller lanes can ask "did the user configure
    this port?" without reaching into a private helper."""

    return _app_persisted_port()


# A stopping MTPLX server keeps its listener while it drains, but its
# /health stops answering first, so classify_port_occupant reads it as
# PORT_FOREIGN for those few seconds. Issue #409: every stop/start cycle
# then tripped the "in use by another app" fallback and silently moved a
# configured 1234 to 1235. Long enough to cover a normal drain, short
# enough that a genuinely foreign listener does not stall a launch.
PORT_SETTLE_TIMEOUT_S = 5.0
PORT_SETTLE_POLL_S = 0.25


def wait_for_port_settle(
    host: str,
    port: int,
    *,
    timeout_s: float = PORT_SETTLE_TIMEOUT_S,
    poll_s: float = PORT_SETTLE_POLL_S,
    api_key: str | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> PortOccupant:
    """Re-classify a foreign-looking port until it settles, or time out.

    Returns the last classification. Anything other than PORT_FOREIGN is
    terminal and returns immediately.

    This does NOT claim to identify the owning process; it separates a
    TRANSIENT occupant that clears on its own (our own draining server,
    a socket in the tail of TIME_WAIT, a listener mid-restart) from a
    STEADY foreign listener that is still there after the window. That is
    the distinction the caller's decision actually needs, and it needs no
    lsof, no psutil, and no new dependency.
    """

    deadline = clock() + max(0.0, float(timeout_s))
    occupant = classify_port_occupant(host, port, api_key=api_key)
    while occupant.kind == PORT_FOREIGN and clock() < deadline:
        sleep(max(0.0, float(poll_s)))
        occupant = classify_port_occupant(host, port, api_key=api_key)
    return occupant


@dataclass(frozen=True)
class ForeignListener:
    """Who holds a port that does not answer ``/health`` (issue #503).

    ``launch_id`` is the ``MTPLX_APP_LAUNCH_ID`` the macOS app gave the
    process at launch: set, the "foreign" listener is one of our own daemons
    that wedged (socket alive, ``/health`` dead); None, it is a stranger's
    process or a CLI-started server. SYNC PAIR: PortPreflight.appOwnedListener
    in the app resolves the same identity and reaps its own wedged daemon in
    place; the CLI names it and leaves the process alone.
    """

    pid: int
    launch_id: str | None

    @property
    def owned_by_app(self) -> bool:
        return bool(self.launch_id)


def listening_process_ids(port: int) -> list[int]:
    """PIDs with a TCP listener on ``port`` (``/usr/sbin/lsof``; empty on failure)."""

    try:
        proc = subprocess.run(
            ["/usr/sbin/lsof", "-nP", f"-iTCP:{int(port)}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    pids: list[int] = []
    for line in proc.stdout.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid > 1:
            pids.append(pid)
    return pids


def process_app_launch_id(pid: int) -> str | None:
    """The ``MTPLX_APP_LAUNCH_ID`` in ``pid``'s environment, or None.

    Read from the kernel's own argv/env image (``KERN_PROCARGS2``), never
    from ``ps`` text, so an argument that merely contains the token cannot
    pass as ownership. None off macOS, for another user's process, or for a
    process without the marker.
    """

    if sys.platform != "darwin":
        return None
    try:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        ctl_kern, kern_procargs2 = 1, 49
        mib = (ctypes.c_int * 3)(ctl_kern, kern_procargs2, int(pid))
        size = ctypes.c_size_t(0)
        if libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0:
            return None
        buffer = ctypes.create_string_buffer(max(1, size.value))
        if libc.sysctl(mib, 3, buffer, ctypes.byref(size), None, 0) != 0:
            return None
        raw = buffer.raw[: size.value]
    except (OSError, AttributeError, ValueError):
        return None
    if len(raw) < 4:
        return None
    argc = int.from_bytes(raw[:4], "little", signed=True)
    if argc < 0:
        return None
    cursor = 4

    def skip_cstring(at: int) -> int | None:
        end = raw.find(b"\0", at)
        return None if end < 0 else end + 1

    nxt = skip_cstring(cursor)  # the exec path
    if nxt is None:
        return None
    cursor = nxt
    while cursor < len(raw) and raw[cursor] == 0:
        cursor += 1
    for _ in range(argc):
        nxt = skip_cstring(cursor)
        if nxt is None:
            return None
        cursor = nxt
    prefix = b"MTPLX_APP_LAUNCH_ID="
    while cursor < len(raw):
        while cursor < len(raw) and raw[cursor] == 0:
            cursor += 1
        if cursor >= len(raw):
            break
        end = raw.find(b"\0", cursor)
        if end < 0:
            end = len(raw)
        entry = raw[cursor:end]
        cursor = end + 1
        if entry.startswith(prefix):
            value = entry[len(prefix):].decode("utf-8", "replace").strip()
            return value or None
    return None


def describe_foreign_listener(port: int) -> ForeignListener | None:
    """Identity of the process holding a foreign-looking ``port``, or None."""

    pids = listening_process_ids(port)
    if not pids:
        return None
    for pid in pids:
        launch_id = process_app_launch_id(pid)
        if launch_id:
            return ForeignListener(pid=pid, launch_id=launch_id)
    return ForeignListener(pid=pids[0], launch_id=None)


def port_busy_advice(
    occupant: PortOccupant,
    *,
    port: int,
    listener: ForeignListener | None = None,
) -> list[str]:
    """Actionable, occupant-aware copy for a busy port.

    ``listener`` (``describe_foreign_listener``) turns the generic "another
    app" line into the truth when the port is held by a daemon the macOS app
    launched that stopped answering (issue #503): the pid, and how to clear
    it, instead of a hunt through other apps.
    """

    if occupant.kind == PORT_APP_DAEMON:
        model = occupant.daemon.model if occupant.daemon else None
        suffix = f" (model: {model})" if model else ""
        return [
            f"Port {port} is the MTPLX app's running server{suffix}.",
            "Use the app's Stop button, or run: "
            f"mtplx stop --port {port}",
        ]
    if occupant.kind == PORT_MTPLX_SERVER:
        return [
            f"Port {port} is an MTPLX server started outside the app.",
            "Press Ctrl-C in that server's terminal, or run: "
            f"mtplx stop --port {port}",
        ]
    if occupant.kind == PORT_AUTH_REQUIRED:
        return [
            f"Port {port} is a running MTPLX server that requires an API key.",
            f"Stop it with: mtplx stop --port {port} --api-key-file <path> "
            "(or --api-key <key>).",
        ]
    if listener is not None and listener.owned_by_app:
        return [
            (
                f"Port {port} is held by pid {listener.pid}, an MTPLX daemon "
                "the app launched that is no longer answering (it wedged)."
            ),
            (
                "Press Stop in the MTPLX app (or quit the app), or run: "
                f"kill {listener.pid}"
            ),
        ]
    if listener is not None:
        return [
            f"Port {port} is in use by another app (not MTPLX): pid {listener.pid}.",
        ]
    return [
        f"Port {port} is in use by another app (not MTPLX).",
    ]


def find_free_port(
    host: str,
    start_port: int,
    *,
    attempts: int = 50,
) -> int | None:
    """First bindable port at or above ``start_port``; ``None`` when exhausted."""

    bind_host = _connect_host(host)
    for candidate in range(int(start_port), int(start_port) + int(attempts)):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((bind_host, candidate))
            return candidate
        except OSError:
            continue
    return None


# ---------- stopping a daemon -------------------------------------------------


def stop_daemon(
    host: str,
    port: int,
    *,
    grace_s: float = 10.0,
    timeout: float = 1.5,
    api_key: str | None = None,
    kill: Callable[[int, int], None] = os.kill,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Stop the MTPLX daemon on ``host:port`` via its health-reported pid.

    SIGTERM first, ``grace_s`` of polling for exit, SIGKILL as a last
    resort. Refuses (with a reason) when the port has no MTPLX daemon or
    the daemon does not report a pid.
    """

    occupant = classify_port_occupant(host, port, timeout=timeout, api_key=api_key)
    if occupant.kind == PORT_FREE:
        return {"ok": False, "reason": "no_server", "port": int(port)}
    if occupant.kind == PORT_AUTH_REQUIRED:
        return {"ok": False, "reason": "auth_required", "port": int(port)}
    if occupant.kind == PORT_FOREIGN:
        return {"ok": False, "reason": "not_mtplx", "port": int(port)}
    daemon = occupant.daemon
    if daemon is None or daemon.pid is None:
        return {"ok": False, "reason": "no_pid", "port": int(port)}

    def process_alive() -> bool:
        try:
            kill(daemon.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    try:
        kill(daemon.pid, signal.SIGTERM)
    except ProcessLookupError:
        return {
            "ok": True,
            "pid": daemon.pid,
            "port": int(port),
            "signal": "none",
            "reason": "already_exited",
        }
    except PermissionError:
        return {
            "ok": False,
            "reason": "permission_denied",
            "pid": daemon.pid,
            "port": int(port),
        }
    deadline = time.monotonic() + max(0.0, float(grace_s))
    while time.monotonic() < deadline:
        if not process_alive():
            return {
                "ok": True,
                "pid": daemon.pid,
                "port": int(port),
                "signal": "SIGTERM",
            }
        sleep(0.2)
    try:
        kill(daemon.pid, signal.SIGKILL)
    except ProcessLookupError:
        return {
            "ok": True,
            "pid": daemon.pid,
            "port": int(port),
            "signal": "SIGTERM",
        }
    return {
        "ok": True,
        "pid": daemon.pid,
        "port": int(port),
        "signal": "SIGKILL",
    }


# ---------- attach chat REPL ---------------------------------------------------


class AttachChatError(RuntimeError):
    pass


@dataclass
class _TurnResult:
    content: str
    reasoning: str
    finish_reason: str | None
    stats: Mapping[str, Any] | None
    cancelled: bool = False


def _sse_events(response: Any) -> Iterator[dict[str, Any]]:
    """Decode ``data:`` SSE frames from a streaming HTTP response."""

    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            return
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload


class AttachChatSession:
    """Streaming chat client against a running daemon's OpenAI endpoint.

    History is client-side: the daemon stays stateless from the CLI's point
    of view and its session cache keys off the transcript like any other
    OpenAI client.
    """

    def __init__(
        self,
        daemon: RunningDaemon,
        *,
        api_key: str | None = None,
        connect_timeout: float = ATTACH_CHAT_CONNECT_TIMEOUT_S,
        inactivity_timeout: float = ATTACH_CHAT_INACTIVITY_TIMEOUT_S,
    ) -> None:
        self.daemon = daemon
        self.api_key = api_key
        self.connect_timeout = float(connect_timeout)
        self.inactivity_timeout = float(inactivity_timeout)
        self.history: list[dict[str, str]] = []
        self.last_stats: Mapping[str, Any] | None = None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def run_turn(
        self,
        prompt: str,
        *,
        on_reasoning: Callable[[str], None] | None = None,
        on_content: Callable[[str], None] | None = None,
    ) -> _TurnResult:
        messages = [*self.history, {"role": "user", "content": prompt}]
        body = json.dumps(
            {
                "model": self.daemon.model,
                "messages": messages,
                "stream": True,
            }
        )
        connection = http.client.HTTPConnection(
            self.daemon.host,
            self.daemon.port,
            timeout=self.connect_timeout,
        )
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason: str | None = None
        stats: Mapping[str, Any] | None = None
        cancelled = False
        try:
            try:
                connection.connect()
            except TimeoutError:
                raise AttachChatError(
                    f"could not connect to the server within {self.connect_timeout:.0f}s"
                ) from None
            # The short timeout above governed the TCP connect only. From here
            # on the socket waits at most the inactivity deadline for the
            # response headers and then between SSE frames: a wedged daemon
            # used to leave this loop blocked forever with a blank answer.
            connection.sock.settimeout(self.inactivity_timeout)
            connection.request(
                "POST", "/v1/chat/completions", body=body, headers=self._headers()
            )
            response = connection.getresponse()
            if response.status != 200:
                detail = response.read().decode("utf-8", errors="replace")
                raise AttachChatError(
                    f"server returned HTTP {response.status}: {detail[:300]}"
                )
            try:
                for payload in _sse_events(response):
                    choices = payload.get("choices") or []
                    choice = choices[0] if choices else {}
                    delta = choice.get("delta") or {}
                    reasoning_text = delta.get("reasoning_content")
                    if isinstance(reasoning_text, str) and reasoning_text:
                        reasoning_parts.append(reasoning_text)
                        if on_reasoning is not None:
                            on_reasoning(reasoning_text)
                    content_text = delta.get("content")
                    if isinstance(content_text, str) and content_text:
                        content_parts.append(content_text)
                        if on_content is not None:
                            on_content(content_text)
                    if choice.get("finish_reason"):
                        finish_reason = str(choice["finish_reason"])
                    if isinstance(payload.get("mtplx_stats"), dict):
                        stats = payload["mtplx_stats"]
                    if isinstance(payload.get("error"), dict):
                        message = str(
                            payload["error"].get("message") or "server error"
                        )
                        raise AttachChatError(message)
            except KeyboardInterrupt:
                # Closing the connection triggers the server's
                # disconnect-cancel path, so generation stops promptly.
                cancelled = True
        except TimeoutError:
            raise AttachChatError(
                "server stopped responding "
                f"(nothing received for {self.inactivity_timeout:.0f}s)"
            ) from None
        finally:
            try:
                connection.close()
            except Exception:
                pass
        content = "".join(content_parts)
        if not cancelled:
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": content})
            if stats is not None:
                self.last_stats = stats
        return _TurnResult(
            content=content,
            reasoning="".join(reasoning_parts),
            finish_reason=finish_reason,
            stats=stats,
            cancelled=cancelled,
        )


def _stats_line_fields(stats: Mapping[str, Any]) -> dict[str, Any]:
    def _float(key: str) -> float | None:
        value = stats.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    depth = stats.get("mtp_depth")
    return {
        "tok_s": _float("decode_tok_s") or _float("tok_s"),
        "accept_rate": _float("accept_rate"),
        "depth": int(depth) if isinstance(depth, int) else None,
        "time_to_first_token_s": _float("ttft_s"),
        "elapsed_s": _float("request_elapsed_s") or _float("elapsed_s"),
    }


def run_attach_chat(
    daemon: RunningDaemon,
    *,
    api_key: str | None = None,
    prompt: str | None = None,
    input_fn: Callable[[str], str] = input,
    printer: Any | None = None,
) -> int:
    """Interactive (or one-shot) chat attached to a running daemon.

    The model never loads a second time: every turn is an HTTP streaming
    request against the daemon that already holds it.
    """

    if printer is None:
        from mtplx.ui.chat_printer import ChatPrinter

        printer = ChatPrinter()
    session = AttachChatSession(daemon, api_key=api_key)
    printer.print_info(
        f"Attached to {daemon.owner_label} on port {daemon.port}"
        + (f"  ·  model {daemon.model}" if daemon.model else ""),
        dim=True,
    )

    def render_turn(text: str) -> int:
        reasoning_open = False
        assistant_open = False

        def on_reasoning(chunk: str) -> None:
            nonlocal reasoning_open
            if not reasoning_open:
                printer.begin_reasoning()
                reasoning_open = True
            printer.stream_reasoning_chunk(chunk)

        def on_content(chunk: str) -> None:
            nonlocal reasoning_open, assistant_open
            if reasoning_open:
                printer.end_reasoning()
                reasoning_open = False
            if not assistant_open:
                printer.begin_assistant()
                assistant_open = True
            printer.stream_chunk(chunk)

        try:
            result = session.run_turn(
                text,
                on_reasoning=on_reasoning,
                on_content=on_content,
            )
        except AttachChatError as exc:
            if reasoning_open:
                printer.end_reasoning()
            if assistant_open:
                printer.end_assistant()
            printer.print_error(str(exc))
            return 1
        except OSError as exc:
            if reasoning_open:
                printer.end_reasoning()
            if assistant_open:
                printer.end_assistant()
            printer.print_error(f"lost connection to the server: {exc}")
            return 1
        if reasoning_open:
            printer.end_reasoning()
        if assistant_open:
            printer.end_assistant()
        if result.cancelled:
            printer.print_info("(cancelled)", dim=True)
            return 0
        if not assistant_open and result.content:
            printer.begin_assistant()
            printer.stream_chunk(result.content)
            printer.end_assistant()
        if result.stats is not None:
            printer.print_stats(**_stats_line_fields(result.stats))
        return 0

    if prompt is not None:
        return render_turn(str(prompt))

    printer.print_info(
        "Chat is ready. Type /stats for the last turn's speed, /exit to leave.",
        dim=True,
    )
    exit_code = 0
    while True:
        try:
            text = input_fn("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            printer.print_info("bye", dim=True)
            return exit_code
        if not text:
            continue
        lowered = text.lower()
        if lowered in {"/exit", "exit", "/quit", "quit"}:
            printer.print_info("bye", dim=True)
            return exit_code
        if lowered in {"/stats", "stats"}:
            if session.last_stats is None:
                printer.print_info("No stats yet.", dim=True)
            else:
                printer.print_stats(**_stats_line_fields(session.last_stats))
            continue
        exit_code = max(exit_code, render_turn(text))
