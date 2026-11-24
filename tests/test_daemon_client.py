from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx import daemon_client
from mtplx.commands import public
from mtplx.daemon_client import (
    PORT_APP_DAEMON,
    PORT_AUTH_REQUIRED,
    PORT_FOREIGN,
    PORT_FREE,
    PORT_MTPLX_SERVER,
    AttachChatError,
    AttachChatSession,
    PortOccupant,
    RunningDaemon,
    classify_port_occupant,
    detect_attachable_daemon,
    fetch_daemon_health,
    find_free_port,
    port_busy_advice,
    probe_running_daemons,
    wait_for_port_settle,
    run_attach_chat,
    stop_daemon,
)


class _StubDaemonHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *_args) -> None:  # noqa: N802 - stdlib signature
        return

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        config = self.server.config  # type: ignore[attr-defined]
        if self.path != "/health":
            self.send_error(404)
            return
        if config.get("garbage"):
            body = b"<html>definitely not mtplx</html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        required_key = config.get("required_api_key")
        if required_key and self.headers.get("Authorization") != f"Bearer {required_key}":
            status = int(config.get("auth_status", 401))
            body = json.dumps(
                {
                    "error": {
                        "message": "missing or invalid API key",
                        "type": "authentication_error",
                    }
                }
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        payload = {
            "ok": True,
            "model": config.get("model"),
            "model_path": config.get("model_path"),
            "startup": {
                "launch_id": config.get("launch_id"),
                "pid": config.get("pid"),
                "api_key_required": bool(config.get("api_key_required")),
            },
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        length = int(self.headers.get("Content-Length") or 0)
        request_body = (
            json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        )
        self.server.requests.append(  # type: ignore[attr-defined]
            {"path": self.path, "body": request_body, "headers": dict(self.headers)}
        )
        config = self.server.config  # type: ignore[attr-defined]
        release = self.server.release  # type: ignore[attr-defined]
        if config.get("hang_before_headers"):
            # A wedged daemon: the request is read and nothing ever comes back.
            release.wait(30)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def sse(payload: dict) -> None:
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
            self.wfile.flush()

        sse(
            {
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ]
            }
        )
        if config.get("hang_after_first_chunk"):
            sse(
                {
                    "choices": [
                        {"index": 0, "delta": {"content": "Hello "}, "finish_reason": None}
                    ]
                }
            )
            release.wait(30)
            return
        sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning_content": "thinking..."},
                        "finish_reason": None,
                    }
                ]
            }
        )
        sse(
            {
                "choices": [
                    {"index": 0, "delta": {"content": "Hello "}, "finish_reason": None}
                ]
            }
        )
        sse(
            {
                "choices": [
                    {"index": 0, "delta": {"content": "there."}, "finish_reason": None}
                ]
            }
        )
        sse(
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "mtplx_stats": {
                    "decode_tok_s": 42.0,
                    "mtp_depth": 3,
                    "request_elapsed_s": 1.25,
                },
            }
        )
        self.wfile.write(b"data: [DONE]\n\n")


@contextlib.contextmanager
def _stub_daemon(config: dict | None = None):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubDaemonHandler)
    server.config = dict(config or {})  # type: ignore[attr-defined]
    server.requests = []  # type: ignore[attr-defined]
    server.release = threading.Event()  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.release.set()  # type: ignore[attr-defined]
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class _RecordingPrinter:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def print_info(self, text, *, dim=False):
        self.events.append(("info", text))

    def print_warning(self, text):
        self.events.append(("warning", text))

    def print_error(self, text):
        self.events.append(("error", text))

    def begin_assistant(self):
        self.events.append(("begin_assistant", None))

    def stream_chunk(self, text):
        self.events.append(("content", text))

    def end_assistant(self):
        self.events.append(("end_assistant", None))

    def begin_reasoning(self):
        self.events.append(("begin_reasoning", None))

    def stream_reasoning_chunk(self, text):
        self.events.append(("reasoning", text))

    def end_reasoning(self):
        self.events.append(("end_reasoning", None))

    def print_stats(self, **fields):
        self.events.append(("stats", fields))


def test_fetch_daemon_health_parses_startup_fields():
    with _stub_daemon(
        {
            "model": "mtplx-test-model",
            "model_path": "/models/example",
            "launch_id": "native-123",
            "pid": 4242,
        }
    ) as server:
        port = server.server_address[1]
        daemon = fetch_daemon_health("127.0.0.1", port)

    assert daemon is not None
    assert daemon.model == "mtplx-test-model"
    assert daemon.model_path == "/models/example"
    assert daemon.launch_id == "native-123"
    assert daemon.pid == 4242
    assert daemon.owned_by_app is True
    assert daemon.base_url == f"http://127.0.0.1:{port}"


def test_classify_port_occupant_covers_all_kinds():
    with _stub_daemon({"model": "m", "launch_id": "native-1", "pid": 1}) as server:
        app = classify_port_occupant("127.0.0.1", server.server_address[1])
    assert app.kind == PORT_APP_DAEMON

    with _stub_daemon({"model": "m", "launch_id": None, "pid": 1}) as server:
        cli = classify_port_occupant("127.0.0.1", server.server_address[1])
    assert cli.kind == PORT_MTPLX_SERVER

    with _stub_daemon({"garbage": True}) as server:
        foreign = classify_port_occupant("127.0.0.1", server.server_address[1])
    assert foreign.kind == PORT_FOREIGN

    free_port = find_free_port("127.0.0.1", 49500)
    assert free_port is not None
    assert classify_port_occupant("127.0.0.1", free_port).kind == PORT_FREE


def test_port_busy_advice_is_occupant_aware():
    app = port_busy_advice(
        PortOccupant(
            kind=PORT_APP_DAEMON,
            daemon=SimpleNamespace(model="mtplx-test"),
        ),
        port=8000,
    )
    assert any("MTPLX app" in line for line in app)
    assert any("mtplx stop --port 8000" in line for line in app)

    cli = port_busy_advice(PortOccupant(kind=PORT_MTPLX_SERVER), port=8000)
    assert any("Ctrl-C" in line for line in cli)

    foreign = port_busy_advice(PortOccupant(kind=PORT_FOREIGN), port=8000)
    assert any("another app" in line for line in foreign)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_port_busy_advice_names_a_wedged_app_daemon_by_pid():
    from mtplx.daemon_client import ForeignListener

    wedged = port_busy_advice(
        PortOccupant(kind=PORT_FOREIGN),
        port=8001,
        listener=ForeignListener(pid=4242, launch_id="launch-abc"),
    )
    assert any("4242" in line and "no longer answering" in line for line in wedged)
    assert any("kill 4242" in line for line in wedged)
    assert not any("another app" in line for line in wedged)

    stranger = port_busy_advice(
        PortOccupant(kind=PORT_FOREIGN),
        port=8001,
        listener=ForeignListener(pid=77, launch_id=None),
    )
    assert any("another app" in line and "pid 77" in line for line in stranger)


def _wedged_listener(port: int, launch_id: str | None) -> subprocess.Popen:
    """A listener that accepts connections and never answers (issue #503),
    spawned with the app's launch marker in its environment like a daemon."""

    env = dict(os.environ)
    env.pop("MTPLX_APP_LAUNCH_ID", None)
    if launch_id:
        env["MTPLX_APP_LAUNCH_ID"] = launch_id
    script = (
        "import socket, time\n"
        "s = socket.socket()\n"
        "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
        f"s.bind(('127.0.0.1', {port}))\n"
        "s.listen(16)\n"
        "print('ready', flush=True)\n"
        "time.sleep(600)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "ready"
    return proc


@pytest.mark.skipif(sys.platform != "darwin", reason="KERN_PROCARGS2 and lsof are Darwin")
def test_describe_foreign_listener_tells_the_apps_wedged_daemon_from_a_stranger():
    from mtplx.daemon_client import describe_foreign_listener, process_app_launch_id

    port = _free_port()
    ours = _wedged_listener(port, "cli-wedged-launch")
    try:
        assert classify_port_occupant("127.0.0.1", port, timeout=0.5).kind == PORT_FOREIGN
        assert process_app_launch_id(ours.pid) == "cli-wedged-launch"
        listener = describe_foreign_listener(port)
        assert listener is not None
        assert listener.pid == ours.pid
        assert listener.launch_id == "cli-wedged-launch"
        assert listener.owned_by_app
    finally:
        ours.kill()
        ours.wait(timeout=5)

    port = _free_port()
    stranger = _wedged_listener(port, None)
    try:
        assert process_app_launch_id(stranger.pid) is None
        listener = describe_foreign_listener(port)
        assert listener is not None
        assert listener.pid == stranger.pid
        assert listener.launch_id is None
        assert not listener.owned_by_app
    finally:
        stranger.kill()
        stranger.wait(timeout=5)
    assert describe_foreign_listener(port) is None


def test_find_free_port_skips_bound_ports():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        bound_port = sock.getsockname()[1]
        free = find_free_port("127.0.0.1", bound_port)
    assert free is not None
    assert free != bound_port


def test_detect_attachable_daemon_honors_kill_switch(monkeypatch):
    with _stub_daemon({"model": "m", "pid": 1}) as server:
        port = server.server_address[1]
        monkeypatch.setenv("MTPLX_START_ATTACH_PROBE", "off")
        assert detect_attachable_daemon(ports=(port,)) is None
        monkeypatch.setenv("MTPLX_START_ATTACH_PROBE", "on")
        daemon = detect_attachable_daemon(ports=(port,))
        assert daemon is not None and daemon.port == port


def test_probe_running_daemons_skips_closed_and_foreign_ports(monkeypatch):
    monkeypatch.setenv("MTPLX_START_ATTACH_PROBE", "on")
    closed_port = find_free_port("127.0.0.1", 49600)
    assert closed_port is not None
    with _stub_daemon({"model": "m", "pid": 7}) as healthy:
        with _stub_daemon({"garbage": True}) as garbage:
            daemons = probe_running_daemons(
                ports=(
                    closed_port,
                    garbage.server_address[1],
                    healthy.server_address[1],
                )
            )
    assert [daemon.port for daemon in daemons] == [healthy.server_address[1]]


def test_stop_daemon_terminates_with_sigterm():
    with _stub_daemon({"model": "m", "pid": 4242}) as server:
        port = server.server_address[1]
        signals: list[int] = []
        alive = {"value": True}

        def fake_kill(pid: int, sig: int) -> None:
            assert pid == 4242
            if sig == 0:
                if not alive["value"]:
                    raise ProcessLookupError
                return
            signals.append(sig)
            if sig == signal.SIGTERM:
                alive["value"] = False

        result = stop_daemon(
            "127.0.0.1", port, kill=fake_kill, sleep=lambda _s: None
        )

    assert result["ok"] is True
    assert result["signal"] == "SIGTERM"
    assert signals == [signal.SIGTERM]


def test_stop_daemon_escalates_to_sigkill_after_grace():
    with _stub_daemon({"model": "m", "pid": 4242}) as server:
        port = server.server_address[1]
        signals: list[int] = []

        def fake_kill(pid: int, sig: int) -> None:
            if sig != 0:
                signals.append(sig)

        result = stop_daemon(
            "127.0.0.1",
            port,
            grace_s=0.0,
            kill=fake_kill,
            sleep=lambda _s: None,
        )

    assert result["ok"] is True
    assert result["signal"] == "SIGKILL"
    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_stop_daemon_refuses_free_foreign_and_pidless_ports():
    free_port = find_free_port("127.0.0.1", 49700)
    assert free_port is not None
    assert stop_daemon("127.0.0.1", free_port)["reason"] == "no_server"

    with _stub_daemon({"garbage": True}) as server:
        result = stop_daemon("127.0.0.1", server.server_address[1])
    assert result["reason"] == "not_mtplx"

    with _stub_daemon({"model": "m", "pid": None}) as server:
        result = stop_daemon("127.0.0.1", server.server_address[1])
    assert result["reason"] == "no_pid"


def test_attach_chat_session_streams_and_keeps_history():
    with _stub_daemon({"model": "mtplx-test-model", "pid": 1}) as server:
        port = server.server_address[1]
        daemon = fetch_daemon_health("127.0.0.1", port)
        assert daemon is not None
        session = AttachChatSession(daemon, api_key="secret-key")
        content_chunks: list[str] = []
        reasoning_chunks: list[str] = []

        result = session.run_turn(
            "hi",
            on_content=content_chunks.append,
            on_reasoning=reasoning_chunks.append,
        )
        second = session.run_turn("again")

        requests = server.requests  # type: ignore[attr-defined]

    assert result.content == "Hello there."
    assert result.reasoning == "thinking..."
    assert result.finish_reason == "stop"
    assert result.stats is not None and result.stats["decode_tok_s"] == 42.0
    assert "".join(content_chunks) == "Hello there."
    assert "".join(reasoning_chunks) == "thinking..."
    assert second.content == "Hello there."
    assert requests[0]["headers"].get("Authorization") == "Bearer secret-key"
    assert requests[0]["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert requests[1]["body"]["messages"][:2] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello there."},
    ]
    assert requests[1]["body"]["messages"][2] == {"role": "user", "content": "again"}


def test_run_attach_chat_one_shot_prompt():
    with _stub_daemon({"model": "mtplx-test-model", "pid": 1}) as server:
        daemon = fetch_daemon_health("127.0.0.1", server.server_address[1])
        assert daemon is not None
        printer = _RecordingPrinter()

        code = run_attach_chat(daemon, prompt="hi", printer=printer)

    assert code == 0
    streamed = "".join(
        str(text) for kind, text in printer.events if kind == "content"
    )
    assert streamed == "Hello there."
    stats_events = [fields for kind, fields in printer.events if kind == "stats"]
    assert stats_events and stats_events[0]["tok_s"] == 42.0


def test_run_attach_chat_repl_supports_stats_and_exit():
    with _stub_daemon({"model": "mtplx-test-model", "pid": 1}) as server:
        daemon = fetch_daemon_health("127.0.0.1", server.server_address[1])
        assert daemon is not None
        printer = _RecordingPrinter()
        answers = iter(["hi", "/stats", "/exit"])

        code = run_attach_chat(
            daemon,
            printer=printer,
            input_fn=lambda _prompt: next(answers),
        )

    assert code == 0
    stats_events = [fields for kind, fields in printer.events if kind == "stats"]
    # Once after the turn, once for /stats.
    assert len(stats_events) == 2
    reasoning = "".join(
        str(text) for kind, text in printer.events if kind == "reasoning"
    )
    assert reasoning == "thinking..."


def test_attach_chat_gives_up_on_a_daemon_that_stops_mid_stream():
    # The session used to open the connection with timeout=None and block in
    # `for raw_line in response:` forever when the daemon stopped emitting
    # frames: a blank answer until Ctrl-C. The socket now waits at most the
    # inactivity deadline between frames and the turn fails with a plain line.
    with _stub_daemon(
        {"model": "mtplx-test-model", "pid": 1, "hang_after_first_chunk": True}
    ) as server:
        daemon = fetch_daemon_health("127.0.0.1", server.server_address[1])
        assert daemon is not None
        session = AttachChatSession(daemon, inactivity_timeout=0.3)
        chunks: list[str] = []
        started = time.monotonic()
        with pytest.raises(AttachChatError) as excinfo:
            session.run_turn("hi", on_content=chunks.append)
        elapsed = time.monotonic() - started

    assert "server stopped responding" in str(excinfo.value)
    assert "0s" in str(excinfo.value)
    assert chunks == ["Hello "]  # what did arrive was shown as it arrived
    assert 0.2 < elapsed < 5.0
    assert session.history == []  # a failed turn is not remembered as an exchange


def test_attach_chat_gives_up_on_a_daemon_that_never_answers():
    with _stub_daemon(
        {"model": "mtplx-test-model", "pid": 1, "hang_before_headers": True}
    ) as server:
        daemon = fetch_daemon_health("127.0.0.1", server.server_address[1])
        assert daemon is not None
        session = AttachChatSession(daemon, inactivity_timeout=0.3)
        with pytest.raises(AttachChatError, match="server stopped responding"):
            session.run_turn("hi")


def test_run_attach_chat_reports_a_stalled_daemon_and_exits_nonzero():
    with _stub_daemon(
        {"model": "mtplx-test-model", "pid": 1, "hang_after_first_chunk": True}
    ) as server:
        daemon = fetch_daemon_health("127.0.0.1", server.server_address[1])
        assert daemon is not None
        printer = _RecordingPrinter()
        original = daemon_client.AttachChatSession

        def short_deadline(*args, **kwargs):
            kwargs.setdefault("inactivity_timeout", 0.3)
            return original(*args, **kwargs)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(daemon_client, "AttachChatSession", short_deadline)
            code = run_attach_chat(daemon, prompt="hi", printer=printer)

    assert code == 1
    kinds = [kind for kind, _ in printer.events]
    assert kinds.index("content") < kinds.index("end_assistant") < kinds.index("error")
    errors = [str(text) for kind, text in printer.events if kind == "error"]
    assert errors == ["server stopped responding (nothing received for 0s)"]


def test_attach_chat_connect_and_inactivity_timeouts_are_wired(monkeypatch):
    # The connect deadline is short and separate from the inactivity deadline
    # that governs the wait for headers and every frame after them.
    recorded: dict[str, object] = {}

    class FakeSocket:
        def settimeout(self, value):
            recorded["socket_timeout"] = value

    class FakeResponse:
        status = 200

        def __iter__(self):
            yield b"data: [DONE]\n"

    class FakeConnection:
        def __init__(self, host, port, timeout=None):
            recorded["connect_timeout"] = timeout
            self.sock = None

        def connect(self):
            self.sock = FakeSocket()

        def request(self, *_args, **_kwargs):
            recorded["requested"] = True

        def getresponse(self):
            return FakeResponse()

        def close(self):
            recorded["closed"] = True

    monkeypatch.setattr(daemon_client.http.client, "HTTPConnection", FakeConnection)
    daemon = RunningDaemon(
        host="127.0.0.1", port=1, model="m", model_path=None, launch_id=None,
        pid=None, api_key_required=False, health={},
    )

    AttachChatSession(daemon).run_turn("hi")

    assert recorded["connect_timeout"] == daemon_client.ATTACH_CHAT_CONNECT_TIMEOUT_S == 5.0
    assert recorded["socket_timeout"] == daemon_client.ATTACH_CHAT_INACTIVITY_TIMEOUT_S == 120.0
    assert recorded["requested"] and recorded["closed"]


# ---------- public.py integration: guard + port auto-select -----------------


def _guard_args(**overrides):
    defaults = {
        "prompt": None,
        "yes": False,
        "_model_explicit": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_terminal_chat_attach_guard_attaches_noninteractively(monkeypatch):
    daemon = SimpleNamespace(
        model="mtplx-test-model",
        model_path="/models/example",
        port=8000,
        host="127.0.0.1",
        owned_by_app=True,
        api_key_required=False,
    )
    monkeypatch.setattr(
        "mtplx.daemon_client.detect_attachable_daemon", lambda: daemon
    )
    attached: list[object] = []

    def fake_attach(target_daemon, _args):
        attached.append(target_daemon)
        return 0

    monkeypatch.setattr(public, "_run_attach_chat_for_args", fake_attach)

    code = public._terminal_chat_attach_guard(
        _guard_args(), runtime_model="/models/example"
    )

    assert code == 0
    assert attached == [daemon]


def test_terminal_chat_attach_guard_rejects_different_explicit_model(monkeypatch):
    daemon = SimpleNamespace(
        model="mtplx-test-model",
        model_path="/models/example",
        port=8000,
        host="127.0.0.1",
        owned_by_app=False,
        api_key_required=False,
    )
    monkeypatch.setattr(
        "mtplx.daemon_client.detect_attachable_daemon", lambda: daemon
    )

    code = public._terminal_chat_attach_guard(
        _guard_args(_model_explicit=True),
        runtime_model="/models/a-very-different-model",
    )

    assert code == 2


def test_terminal_chat_attach_guard_passes_through_without_daemon(monkeypatch):
    monkeypatch.setattr(
        "mtplx.daemon_client.detect_attachable_daemon", lambda: None
    )

    assert (
        public._terminal_chat_attach_guard(
            _guard_args(), runtime_model="/models/example"
        )
        is None
    )


def test_quickstart_autoselect_busy_port_bumps_only_foreign(monkeypatch):
    # #409: the sweep now settles a foreign reading before believing it, and
    # treats the app's persisted port as user-configured. Both are stubbed
    # here so this test keeps pinning the original bump contract (and so it
    # never reads the developer's real app settings).
    monkeypatch.setattr(
        "mtplx.daemon_client.wait_for_port_settle",
        lambda host, port, **_kw: daemon_client.classify_port_occupant(host, port),
    )
    monkeypatch.setattr("mtplx.daemon_client.app_configured_port", lambda: None)
    monkeypatch.setattr(
        "mtplx.daemon_client.classify_port_occupant",
        lambda _host, _port: PortOccupant(kind=PORT_FOREIGN),
    )
    monkeypatch.setattr(
        "mtplx.daemon_client.find_free_port", lambda _host, _start: 8010
    )
    args = SimpleNamespace(host="127.0.0.1", port=8000)
    public._quickstart_autoselect_busy_port(args, target="openwebui", cli_flags=set())
    assert args.port == 8010

    monkeypatch.setattr(
        "mtplx.daemon_client.classify_port_occupant",
        lambda _host, _port: PortOccupant(kind=PORT_APP_DAEMON),
    )
    reused = SimpleNamespace(host="127.0.0.1", port=8000)
    public._quickstart_autoselect_busy_port(
        reused, target="openwebui", cli_flags=set()
    )
    assert reused.port == 8000

    explicit = SimpleNamespace(host="127.0.0.1", port=8000)
    public._quickstart_autoselect_busy_port(
        explicit, target="openwebui", cli_flags={"port"}
    )
    assert explicit.port == 8000

    terminal = SimpleNamespace(host="127.0.0.1", port=8000)
    public._quickstart_autoselect_busy_port(
        terminal, target="terminal", cli_flags=set()
    )
    assert terminal.port == 8000


def test_daemon_runs_model_matches_path_and_public_id(tmp_path):
    model_dir = tmp_path / "Qwen3.6-27B-MTPLX-Optimized-Speed"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3NextForCausalLM"],
                "model_type": "qwen3_next",
                "quantization": {"bits": 4},
            }
        ),
        encoding="utf-8",
    )
    daemon = SimpleNamespace(
        model="mtplx-qwen36-27b-optimized-speed",
        model_path="/somewhere/else",
    )
    assert public._daemon_runs_model(daemon, str(model_dir)) is True

    path_daemon = SimpleNamespace(model=None, model_path=str(model_dir))
    assert public._daemon_runs_model(path_daemon, str(model_dir)) is True

    other = SimpleNamespace(model="mtplx-other", model_path="/x")
    assert public._daemon_runs_model(other, str(tmp_path / "unrelated")) is False


# ---------------------------------------------------------------------------
# Issue #409 (reporter kmei3560): "Port XXXX was in use by another app. MTPLX
# now uses port (XXXX+1)" on almost every stop/start, forcing him to configure
# 1233 so the bump landed on the 1234 he actually wanted. Two defects: the
# probe misreads our own DRAINING server as a foreign app (its listener
# outlives its /health), and a port the user configured is moved silently.


def test_wait_for_port_settle_returns_when_the_drain_finishes():
    """A draining MTPLX reads foreign until its listener closes."""
    readings = [
        PortOccupant(kind=PORT_FOREIGN),
        PortOccupant(kind=PORT_FOREIGN),
        PortOccupant(kind=PORT_FREE),
        PortOccupant(kind=PORT_FOREIGN),  # never reached
    ]
    calls: list[tuple[str, int]] = []

    def fake_classify(host, port, *, api_key=None):
        calls.append((host, port))
        return readings[len(calls) - 1]

    import mtplx.daemon_client as dc

    original = dc.classify_port_occupant
    dc.classify_port_occupant = fake_classify
    try:
        slept: list[float] = []
        occupant = wait_for_port_settle(
            "127.0.0.1",
            1234,
            timeout_s=5.0,
            poll_s=0.25,
            clock=lambda: 0.0,
            sleep=slept.append,
        )
    finally:
        dc.classify_port_occupant = original

    assert occupant.kind == PORT_FREE
    assert len(calls) == 3
    assert slept == [0.25, 0.25]


def test_wait_for_port_settle_gives_up_on_a_steady_foreign_listener():
    """The window is bounded: a real stranger still resolves as foreign."""
    import mtplx.daemon_client as dc

    ticks = iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    calls = {"n": 0}

    def fake_classify(host, port, *, api_key=None):
        calls["n"] += 1
        return PortOccupant(kind=PORT_FOREIGN)

    original = dc.classify_port_occupant
    dc.classify_port_occupant = fake_classify
    try:
        occupant = wait_for_port_settle(
            "127.0.0.1",
            1234,
            timeout_s=3.0,
            poll_s=0.25,
            clock=lambda: next(ticks),
            sleep=lambda _s: None,
        )
    finally:
        dc.classify_port_occupant = original

    assert occupant.kind == PORT_FOREIGN
    # Bounded, not a spin: the deadline is honored.
    assert calls["n"] <= 6


def test_wait_for_port_settle_short_circuits_on_a_healthy_daemon():
    """A terminal classification returns at once and never sleeps."""

    def _no_sleep(_seconds: float) -> None:
        raise AssertionError("a terminal classification must not sleep")

    import mtplx.daemon_client as dc

    original = dc.classify_port_occupant
    dc.classify_port_occupant = lambda _h, _p, **_kw: PortOccupant(
        kind=PORT_APP_DAEMON
    )
    try:
        occupant = wait_for_port_settle("127.0.0.1", 1234, sleep=_no_sleep)
    finally:
        dc.classify_port_occupant = original
    assert occupant.kind == PORT_APP_DAEMON


def test_quickstart_waits_out_our_own_drain_instead_of_bumping(monkeypatch):
    """The reporter's stop/start cycle: foreign for a moment, then free.

    The first classification still says foreign (the old server's listener
    is draining); the settle window then sees it clear, so the launch keeps
    the port instead of moving to +1.
    """
    monkeypatch.setattr("mtplx.daemon_client.app_configured_port", lambda: None)
    monkeypatch.setattr(
        "mtplx.daemon_client.classify_port_occupant",
        lambda _host, _port, **_kw: PortOccupant(kind=PORT_FOREIGN),
    )
    monkeypatch.setattr(
        "mtplx.daemon_client.wait_for_port_settle",
        lambda _host, _port, **_kw: PortOccupant(kind=PORT_FREE),
    )
    monkeypatch.setattr(
        "mtplx.daemon_client.find_free_port",
        lambda _host, _start: (_ for _ in ()).throw(
            AssertionError("must not look for another port")
        ),
    )
    args = SimpleNamespace(host="127.0.0.1", port=1234)
    public._quickstart_autoselect_busy_port(args, target="openwebui", cli_flags=set())
    assert args.port == 1234


def test_quickstart_never_moves_the_app_configured_port(monkeypatch):
    """His exact seat: the port lives in app settings, not on the CLI."""
    printed: list[str] = []
    monkeypatch.setattr(public, "_quickstart_line", printed.append)
    monkeypatch.setattr("mtplx.daemon_client.app_configured_port", lambda: 1234)
    monkeypatch.setattr(
        "mtplx.daemon_client.classify_port_occupant",
        lambda _host, _port, **_kw: PortOccupant(kind=PORT_FOREIGN),
    )
    monkeypatch.setattr(
        "mtplx.daemon_client.wait_for_port_settle",
        lambda _host, _port, **_kw: PortOccupant(kind=PORT_FOREIGN),
    )
    monkeypatch.setattr(
        "mtplx.daemon_client.find_free_port",
        lambda _host, _start: (_ for _ in ()).throw(
            AssertionError("a configured port is never relocated")
        ),
    )
    args = SimpleNamespace(host="127.0.0.1", port=1234)
    public._quickstart_autoselect_busy_port(args, target="openwebui", cli_flags=set())

    assert args.port == 1234
    joined = " ".join(printed)
    assert "not MTPLX" in joined  # the actionable occupant copy
    assert "Keeping the configured port 1234" in joined


def test_quickstart_still_bumps_an_unconfigured_default_port(monkeypatch):
    """Auto-pick survives for a port nobody chose."""
    monkeypatch.setattr(public, "_quickstart_line", lambda _line: None)
    monkeypatch.setattr("mtplx.daemon_client.app_configured_port", lambda: 1234)
    monkeypatch.setattr(
        "mtplx.daemon_client.classify_port_occupant",
        lambda _host, _port, **_kw: PortOccupant(kind=PORT_FOREIGN),
    )
    monkeypatch.setattr(
        "mtplx.daemon_client.wait_for_port_settle",
        lambda _host, _port, **_kw: PortOccupant(kind=PORT_FOREIGN),
    )
    monkeypatch.setattr(
        "mtplx.daemon_client.find_free_port", lambda _host, _start: 8010
    )
    args = SimpleNamespace(host="127.0.0.1", port=8000)
    public._quickstart_autoselect_busy_port(args, target="openwebui", cli_flags=set())
    assert args.port == 8010


# ---------- keyed server stop + auth_required classification ----------------


def test_classify_port_occupant_distinguishes_401_and_403_from_foreign():
    for status in (401, 403):
        with _stub_daemon(
            {
                "model": "m",
                "pid": 4321,
                "required_api_key": "secret-key",
                "auth_status": status,
            }
        ) as server:
            port = server.server_address[1]
            unauthed = classify_port_occupant("127.0.0.1", port)
            wrong_key = classify_port_occupant("127.0.0.1", port, api_key="wrong")
            authed = classify_port_occupant("127.0.0.1", port, api_key="secret-key")
        assert unauthed.kind == PORT_AUTH_REQUIRED
        assert wrong_key.kind == PORT_AUTH_REQUIRED
        assert authed.kind == PORT_MTPLX_SERVER
        assert authed.daemon is not None and authed.daemon.pid == 4321

    with _stub_daemon({"garbage": True}) as server:
        assert classify_port_occupant("127.0.0.1", server.server_address[1]).kind == PORT_FOREIGN

    advice = port_busy_advice(PortOccupant(kind=PORT_AUTH_REQUIRED), port=8091)
    joined = " ".join(advice)
    assert "--api-key-file" in joined and "--api-key" in joined


def test_stop_daemon_reports_auth_required_and_stops_when_keyed():
    with _stub_daemon(
        {"model": "m", "pid": 9876, "required_api_key": "top-secret"}
    ) as server:
        port = server.server_address[1]
        kills: list[tuple[int, int]] = []

        def fake_kill(pid: int, sig: int) -> None:
            kills.append((pid, sig))
            if sig == 0:
                raise ProcessLookupError

        unauthed = stop_daemon("127.0.0.1", port, kill=fake_kill, sleep=lambda _s: None)
        assert unauthed == {"ok": False, "reason": "auth_required", "port": port}
        assert kills == []

        authed = stop_daemon(
            "127.0.0.1",
            port,
            api_key="top-secret",
            grace_s=1.0,
            kill=fake_kill,
            sleep=lambda _s: None,
        )
        assert authed["ok"] is True
        assert authed["pid"] == 9876
        assert authed["signal"] == "SIGTERM"
        assert kills[0] == (9876, signal.SIGTERM)


def test_cli_stop_parser_accepts_api_key_and_api_key_file():
    from mtplx.cli import build_parser

    parser = build_parser()
    ns_key = parser.parse_args(["stop", "--port", "8091", "--api-key", "abc"])
    assert ns_key.port == 8091
    assert ns_key.api_key == "abc"
    assert ns_key.api_key_file is None

    ns_file = parser.parse_args(["stop", "--port", "8091", "--api-key-file", "/tmp/k"])
    assert ns_file.port == 8091
    assert ns_file.api_key_file == "/tmp/k"
    assert ns_file.api_key is None


def test_cmd_stop_public_handles_keyed_server_and_missing_key(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("MTPLX_API_KEY", raising=False)
    monkeypatch.delenv("MTPLX_AUTH", raising=False)
    key_file = tmp_path / "api-key"
    key_file.write_text("file-secret-key\n", encoding="utf-8")

    with _stub_daemon(
        {"model": "m", "pid": 5555, "required_api_key": "file-secret-key"}
    ) as server:
        port = server.server_address[1]

        # (a) Unauthenticated stop on a keyed server reports auth_required (not not_mtplx).
        rc_unauthed = public.cmd_stop_public(
            SimpleNamespace(host="127.0.0.1", port=port, json=True)
        )
        assert rc_unauthed == 1
        payload_unauthed = json.loads(capsys.readouterr().out)
        assert payload_unauthed["ok"] is False
        assert payload_unauthed["reason"] == "auth_required"

        rc_text = public.cmd_stop_public(
            SimpleNamespace(host="127.0.0.1", port=port, json=False)
        )
        assert rc_text == 1
        text_out = capsys.readouterr().out
        assert "--api-key-file" in text_out
        assert "--api-key" in text_out
        assert "not by an MTPLX server" not in text_out

        # (b) Stop with --api-key succeeds.
        stop_calls: list[tuple[int, str | None]] = []

        def fake_stop(host: str, target_port: int, *, grace_s: float = 10.0, api_key: str | None = None):
            stop_calls.append((target_port, api_key))
            return stop_daemon(
                host,
                target_port,
                grace_s=grace_s,
                api_key=api_key,
                kill=lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError())
                if sig == 0
                else None,
                sleep=lambda _s: None,
            )

        monkeypatch.setattr("mtplx.daemon_client.stop_daemon", fake_stop)
        rc_key = public.cmd_stop_public(
            SimpleNamespace(
                host="127.0.0.1", port=port, api_key="file-secret-key", json=True
            )
        )
        assert rc_key == 0
        payload_key = json.loads(capsys.readouterr().out)
        assert payload_key["ok"] is True
        assert payload_key["pid"] == 5555

        # (c) Stop with --api-key-file reads key from file and succeeds.
        rc_file = public.cmd_stop_public(
            SimpleNamespace(
                host="127.0.0.1", port=port, api_key_file=str(key_file), json=True
            )
        )
        assert rc_file == 0
        payload_file = json.loads(capsys.readouterr().out)
        assert payload_file["ok"] is True
        assert stop_calls == [(port, "file-secret-key"), (port, "file-secret-key")]

        # (d) Missing --api-key-file fails cleanly without creating the file.
        missing_file = tmp_path / "does-not-exist"
        rc_missing = public.cmd_stop_public(
            SimpleNamespace(
                host="127.0.0.1", port=port, api_key_file=str(missing_file), json=True
            )
        )
        assert rc_missing == 2
        assert not missing_file.exists()
        payload_missing = json.loads(capsys.readouterr().out)
        assert payload_missing["reason"] == "invalid_api_key"

        # (e) Bare `mtplx stop` (no --port) surfaces auth_required when probed port needs a key.
        monkeypatch.setattr("mtplx.daemon_client.default_probe_ports", lambda: (port,))
        rc_bare = public.cmd_stop_public(
            SimpleNamespace(host="127.0.0.1", port=None, json=True)
        )
        assert rc_bare == 1
        payload_bare = json.loads(capsys.readouterr().out)
        assert payload_bare["reason"] == "auth_required"
        assert payload_bare["port"] == port
