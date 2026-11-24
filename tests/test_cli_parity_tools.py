from __future__ import annotations

import contextlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

from mtplx.commands import public
from mtplx.commands.public import (
    _cmd_bench_aime,
    _parse_settings_pairs,
    cmd_settings_public,
    cmd_stop_public,
)


class _ParityStubHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *_args) -> None:  # noqa: N802 - stdlib signature
        return

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        config = self.server.config  # type: ignore[attr-defined]
        if self.path == "/health":
            self._json(
                {
                    "ok": True,
                    "model": "mtplx-test-model",
                    "startup": {"launch_id": None, "pid": 4242},
                }
            )
        elif self.path == "/v1/mtplx/settings":
            self._json(
                {
                    "ok": True,
                    "depth": 3,
                    "reasoning": "auto",
                    "generation_mode": "mtp",
                    "model": "mtplx-test-model",
                }
            )
        elif self.path == "/v1/mtplx/benchmarks/aime/active":
            self._json({"active_run_id": config.get("active_run_id")})
        elif self.path == "/v1/mtplx/benchmarks/aime/run-1/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for event in config.get("stream_events", []):
                self.wfile.write(
                    (
                        f"event: {event.get('event')}\n"
                        f"data: {json.dumps(event)}\n\n"
                    ).encode("utf-8")
                )
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        config = self.server.config  # type: ignore[attr-defined]
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        self.server.posts.append({"path": self.path, "body": body})  # type: ignore[attr-defined]
        if self.path == "/v1/mtplx/settings":
            mode = config.get("settings_mode", "ok")
            # Mirror the daemon's real wire shape: the OpenAI error
            # envelope with the structured HTTPException detail riding
            # along as error.detail (QA-105 — the old stub returned a
            # bare {"detail": ...} body the daemon never sends, which
            # masked the CLI rendering bug).
            if mode == "restart_required":
                detail = {
                    "error": "restart_required",
                    "keys": ["model"],
                    "message": "the following settings require a server restart: model",
                }
                self._json(
                    {
                        "error": {
                            "message": detail["message"],
                            "type": "invalid_request_error",
                            "code": "HTTPException",
                            "param": None,
                            "detail": detail,
                        }
                    },
                    status=400,
                )
            elif mode == "unknown_settings":
                detail = {
                    "error": "unknown_settings",
                    "keys": ["bogus"],
                    "supported": ["depth", "reasoning"],
                }
                self._json(
                    {
                        "error": {
                            "message": str(detail),
                            "type": "invalid_request_error",
                            "code": "HTTPException",
                            "param": None,
                            "detail": detail,
                        }
                    },
                    status=400,
                )
            else:
                self._json({"ok": True, "applied": body, **body})
        elif self.path == "/v1/mtplx/benchmarks/aime/start":
            self._json(
                {
                    "run_id": "run-1",
                    "total": config.get("total", 2),
                    "model": "mtplx-test-model",
                    "state": "running",
                }
            )
        elif self.path == "/v1/mtplx/benchmarks/aime/run-1/cancel":
            self._json(config.get("cancel_snapshot", {"state": "cancelled"}))
        else:
            self.send_error(404)


@contextlib.contextmanager
def _stub_server(config: dict | None = None):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ParityStubHandler)
    server.config = dict(config or {})  # type: ignore[attr-defined]
    server.posts = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# ---------- mtplx stop --------------------------------------------------------


def test_stop_reports_no_server(monkeypatch, capsys):
    monkeypatch.setattr(
        "mtplx.daemon_client.probe_running_daemons", lambda **_kwargs: []
    )
    args = SimpleNamespace(host="127.0.0.1", port=None, grace_seconds=0.1, json=False)

    assert cmd_stop_public(args) == 1
    assert "No running MTPLX server found" in capsys.readouterr().out


def test_stop_lists_multiple_servers(monkeypatch, capsys):
    daemons = [
        SimpleNamespace(port=8000, model="m-a", owner_label="the MTPLX app"),
        SimpleNamespace(port=18083, model="m-b", owner_label="another MTPLX server"),
    ]
    monkeypatch.setattr(
        "mtplx.daemon_client.probe_running_daemons", lambda **_kwargs: daemons
    )
    args = SimpleNamespace(host="127.0.0.1", port=None, grace_seconds=0.1, json=False)

    assert cmd_stop_public(args) == 2
    output = capsys.readouterr().out
    assert "port 8000" in output and "port 18083" in output
    assert "mtplx stop --port" in output


def test_stop_stops_single_discovered_server(monkeypatch, capsys):
    daemon = SimpleNamespace(port=8000, model="m", owner_label="the MTPLX app")
    monkeypatch.setattr(
        "mtplx.daemon_client.probe_running_daemons", lambda **_kwargs: [daemon]
    )
    calls: list[tuple] = []

    def fake_stop(host, port, *, grace_s, api_key=None):
        assert api_key is None
        calls.append((host, port, grace_s))
        return {"ok": True, "pid": 4242, "port": port, "signal": "SIGTERM"}

    monkeypatch.setattr("mtplx.daemon_client.stop_daemon", fake_stop)
    args = SimpleNamespace(host="127.0.0.1", port=None, grace_seconds=2.0, json=False)

    assert cmd_stop_public(args) == 0
    assert calls == [("127.0.0.1", 8000, 2.0)]
    output = capsys.readouterr().out
    assert "Stopped the MTPLX server on port 8000" in output
    assert "pid 4242" in output


def test_stop_refuses_foreign_port(monkeypatch, capsys):
    monkeypatch.setattr(
        "mtplx.daemon_client.stop_daemon",
        lambda host, port, *, grace_s, api_key=None: {"ok": False, "reason": "not_mtplx"},
    )
    args = SimpleNamespace(host="127.0.0.1", port=9999, grace_seconds=0.1, json=False)

    assert cmd_stop_public(args) == 1
    assert "not by an MTPLX server" in capsys.readouterr().out


# ---------- mtplx settings ----------------------------------------------------


def test_parse_settings_pairs_coerces_json_values():
    parsed, errors = _parse_settings_pairs(
        ["depth=2", "reasoning=off", "top_p=0.9", "flag=true"]
    )
    assert parsed == {
        "depth": 2,
        "reasoning": "off",
        "top_p": 0.9,
        "flag": True,
    }
    assert errors == []

    parsed, errors = _parse_settings_pairs(["nonsense"])
    assert parsed == {}
    assert errors == ["nonsense"]


def test_settings_get_renders_sorted_keys(capsys):
    with _stub_server() as server:
        args = SimpleNamespace(
            host="127.0.0.1",
            port=server.server_address[1],
            settings_action="get",
            pairs=[],
            json=False,
        )
        assert cmd_settings_public(args) == 0
    output = capsys.readouterr().out
    assert "depth = 3" in output
    assert 'reasoning = "auto"' in output


def test_settings_set_applies_pairs(capsys):
    with _stub_server() as server:
        args = SimpleNamespace(
            host="127.0.0.1",
            port=server.server_address[1],
            settings_action="set",
            pairs=["depth=2", "reasoning=off"],
            json=False,
        )
        assert cmd_settings_public(args) == 0
        posts = server.posts  # type: ignore[attr-defined]
    assert posts[0]["body"] == {"depth": 2, "reasoning": "off"}
    output = capsys.readouterr().out
    assert "applied: depth = 2" in output


def test_settings_set_renders_restart_required(capsys):
    with _stub_server({"settings_mode": "restart_required"}) as server:
        args = SimpleNamespace(
            host="127.0.0.1",
            port=server.server_address[1],
            settings_action="set",
            pairs=["model=other"],
            json=False,
        )
        assert cmd_settings_public(args) == 2
    output = capsys.readouterr().out
    assert "need a server restart: model" in output
    assert "MTPLX app" in output


def test_settings_set_renders_unknown_settings(capsys):
    with _stub_server({"settings_mode": "unknown_settings"}) as server:
        args = SimpleNamespace(
            host="127.0.0.1",
            port=server.server_address[1],
            settings_action="set",
            pairs=["bogus=1"],
            json=False,
        )
        assert cmd_settings_public(args) == 2
    output = capsys.readouterr().out
    assert "unknown settings: bogus" in output
    assert "supported: depth, reasoning" in output


def test_settings_unreachable_server_is_actionable(capsys):
    from mtplx.daemon_client import find_free_port

    port = find_free_port("127.0.0.1", 49800)
    assert port is not None
    args = SimpleNamespace(
        host="127.0.0.1",
        port=port,
        settings_action="get",
        pairs=[],
        json=False,
    )
    assert cmd_settings_public(args) == 1
    output = capsys.readouterr().out
    assert "No MTPLX server is responding" in output
    assert "mtplx start" in output


def test_settings_bare_pairs_imply_set(capsys):
    with _stub_server() as server:
        args = SimpleNamespace(
            host="127.0.0.1",
            port=server.server_address[1],
            settings_action="get",
            pairs=["depth=1"],
            json=False,
        )
        assert cmd_settings_public(args) == 0
        posts = server.posts  # type: ignore[attr-defined]
    assert posts and posts[0]["body"] == {"depth": 1}


# ---------- mtplx bench aime --------------------------------------------------


def _aime_stream_events() -> list[dict]:
    return [
        {"event": "question_started", "run_id": "run-1", "idx": 1, "attempt": 1},
        {
            "event": "question_done",
            "run_id": "run-1",
            "idx": 1,
            "status": "correct",
            "extracted": 204,
            "expected": 204,
            "duration_ms": 12_000,
            "reasoning_token_count": 500,
            "answer_token_count": 100,
        },
        {"event": "question_started", "run_id": "run-1", "idx": 2, "attempt": 1},
        {
            "event": "question_done",
            "run_id": "run-1",
            "idx": 2,
            "status": "incorrect",
            "extracted": 11,
            "expected": 42,
            "duration_ms": 8_000,
            "reasoning_token_count": 300,
            "answer_token_count": 60,
        },
        {
            "event": "run_done",
            "run_id": "run-1",
            "state": "done",
            "score": 1,
            "total": 2,
            "accuracy": 0.5,
            "duration_ms": 20_000,
            "model": "mtplx-test-model",
            "per_question": [
                {"idx": 1, "status": "correct"},
                {"idx": 2, "status": "incorrect"},
            ],
        },
    ]


def _aime_args(server, **overrides) -> SimpleNamespace:
    defaults = {
        "url": f"http://127.0.0.1:{server.server_address[1]}",
        "port": 8041,
        "quick": False,
        "json": False,
        "_cli_flags": set(),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_bench_aime_streams_per_question_lines_and_summary(capsys):
    with _stub_server({"stream_events": _aime_stream_events()}) as server:
        args = _aime_args(server)
        assert _cmd_bench_aime(args) == 0
        posts = server.posts  # type: ignore[attr-defined]

    assert posts[0]["path"] == "/v1/mtplx/benchmarks/aime/start"
    assert posts[0]["body"] == {}
    output = capsys.readouterr().out
    assert "run run-1" in output
    assert "Q 1" in output and "correct" in output
    assert "answer=204" in output
    assert "expected=42" in output
    assert "tok/s" in output
    assert "AIME done: 1/2  (50.0%)" in output
    assert "✓ ✗" in output


def test_bench_aime_quick_caps_questions(capsys):
    with _stub_server({"stream_events": _aime_stream_events()}) as server:
        args = _aime_args(server, quick=True)
        assert _cmd_bench_aime(args) == 0
        posts = server.posts  # type: ignore[attr-defined]
    assert posts[0]["body"] == {"question_limit": 5}


def test_bench_aime_attaches_to_active_run(capsys):
    with _stub_server(
        {
            "active_run_id": "run-1",
            "stream_events": _aime_stream_events(),
        }
    ) as server:
        args = _aime_args(server)
        assert _cmd_bench_aime(args) == 0
        posts = server.posts  # type: ignore[attr-defined]
    # No start POST: it attached to the run already in progress.
    assert all(post["path"] != "/v1/mtplx/benchmarks/aime/start" for post in posts)
    assert "already in progress" in capsys.readouterr().out


def test_bench_aime_without_server_is_actionable(capsys):
    from mtplx.daemon_client import find_free_port

    port = find_free_port("127.0.0.1", 49900)
    assert port is not None
    args = SimpleNamespace(
        url=f"http://127.0.0.1:{port}",
        port=8041,
        quick=False,
        json=False,
        _cli_flags=set(),
    )
    assert _cmd_bench_aime(args) == 1
    output = capsys.readouterr().out
    assert "No MTPLX server is responding" in output


def test_bench_aime_routes_through_bench_dispatcher(monkeypatch):
    calls: list[object] = []
    monkeypatch.setattr(public, "_cmd_bench_aime", lambda args: calls.append(args) or 0)
    args = SimpleNamespace(bench_action="aime")
    assert public.cmd_bench_public(args) == 0
    assert calls == [args]


# ---------- parser wiring -----------------------------------------------------


def test_cli_parses_stop_settings_and_bench_aime():
    from mtplx.cli import build_parser

    parser = build_parser()
    stop = parser.parse_args(["stop", "--port", "8010"])
    assert stop.port == 8010
    assert stop.func.__name__ == "cmd_stop_public"

    settings = parser.parse_args(["settings", "set", "depth=2"])
    assert settings.settings_action == "set"
    assert settings.pairs == ["depth=2"]
    assert settings.func.__name__ == "cmd_settings_public"

    bench = parser.parse_args(["bench", "aime", "--quick"])
    assert bench.bench_action == "aime"
    assert bench.quick is True
