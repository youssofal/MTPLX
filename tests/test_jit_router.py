"""`mtplx serve --jit`: the router swaps one `mtplx serve` child per model.

The child is a stand-in HTTP server, so nothing here loads a model.
"""

from __future__ import annotations

import http.client
import json
import subprocess
import sys
import textwrap
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from mtplx.server import jit_router
from mtplx.server.jit_router import JitModel, ModelSupervisor, child_serve_argv

FAKE_CHILD = textwrap.dedent(
    """
    import json, sys, time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    argv = sys.argv
    port = int(argv[argv.index("--port") + 1])
    model = argv[argv.index("--model") + 1]

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *a): pass
        def _json(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            if self.path == "/health":
                self._json({"ok": True, "model": model, "pid": 1})
            elif self.path.startswith("/v1/models"):
                self._json({"object": "list", "data": [
                    {"id": model, "capability": "chat", "context_length": 4096}]})
            else:
                self._json({"path": self.path})
        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                for piece in ("a", "b"):
                    self.wfile.write(("data: " + json.dumps({"model": model, "t": piece}) + "\\n\\n").encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\\n\\n")
                self.close_connection = True
            else:
                time.sleep(req.get("sleep", 0))
                self._json({"served_by": model, "path": self.path})

    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
    """
)


@pytest.fixture
def fake_script(tmp_path):
    path = tmp_path / "fake_child.py"
    path.write_text(FAKE_CHILD)
    return path


def _models():
    return [
        JitModel("Org/Alpha", "/m/alpha", 10.0, frozenset({"org/alpha", "alpha"})),
        JitModel("Org/Beta", "/m/beta", 20.0, frozenset({"org/beta", "beta"})),
        JitModel("Org/Huge", "/m/huge", 900.0, frozenset({"org/huge"})),
    ]


@pytest.fixture
def supervisor(fake_script):
    spawned: list[subprocess.Popen] = []

    def spawn(argv):
        # argv is the real `mtplx serve` command; run the stand-in instead.
        port = argv[argv.index("--port") + 1]
        model = argv[argv.index("--model") + 1]
        proc = subprocess.Popen(
            [sys.executable, str(fake_script), "--port", port, "--model", model],
            start_new_session=True,
        )
        spawned.append(proc)
        return proc

    sup = ModelSupervisor(
        raw_argv=["serve", "--jit", "--port", "8000", "--depth", "3"],
        discover=_models,
        spawn=spawn,
        ready_timeout_s=30.0,
        stop_grace_s=5.0,
        ram_gib=128.0,
    )
    sup.spawned = spawned
    yield sup
    sup.shutdown()


@pytest.fixture
def router(supervisor):
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), jit_router.make_handler(supervisor, api_key=None)
    )
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def _request(port, method, path, payload=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    body = json.dumps(payload).encode() if payload is not None else None
    conn.request(method, path, body=body, headers=headers or {})
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, data


def _json(port, method, path, payload=None, headers=None):
    status, data = _request(port, method, path, payload, headers)
    return status, json.loads(data)


def test_child_argv_keeps_user_flags_and_drops_router_flags():
    argv = child_serve_argv(
        [
            "serve",
            "--jit",
            "--jit-models=a,b",
            "--model",
            "x",
            "--host=0.0.0.0",
            "--port",
            "8000",
            "--api-key",
            "secret",
            "--depth",
            "3",
            "--no-stats-footer",
        ],
        model_ref="/m/alpha",
        port=5555,
    )
    assert argv[1:5] == ["-P", "-m", "mtplx.cli", "serve"]
    assert argv[5:] == [
        "--model", "/m/alpha", "--host", "127.0.0.1", "--port", "5555",
        "--depth", "3", "--no-stats-footer",
    ]
    assert "secret" not in argv


def test_first_request_loads_the_named_model(router, supervisor):
    status, body = _json(
        router, "POST", "/v1/chat/completions", {"model": "Org/Alpha"}
    )
    assert status == 200
    assert body["served_by"] == "/m/alpha"
    assert supervisor.current.model_id == "Org/Alpha"


def test_switching_models_replaces_the_child(router, supervisor):
    _json(router, "POST", "/v1/chat/completions", {"model": "Org/Alpha"})
    first = supervisor.spawned[0]
    status, body = _json(router, "POST", "/v1/chat/completions", {"model": "beta"})
    assert status == 200 and body["served_by"] == "/m/beta"
    assert first.poll() is not None, "the old child must be gone, not orphaned"
    assert len(supervisor.spawned) == 2
    # Same model again reuses the running child.
    _json(router, "POST", "/v1/chat/completions", {"model": "Org/Beta"})
    assert len(supervisor.spawned) == 2


def test_swap_waits_for_requests_in_flight(router, supervisor):
    results = {}

    def slow():
        results["slow"] = _json(
            router, "POST", "/v1/chat/completions", {"model": "Org/Alpha", "sleep": 1.5}
        )

    thread = threading.Thread(target=slow)
    thread.start()
    time.sleep(0.5)
    started = time.monotonic()
    status, body = _json(router, "POST", "/v1/chat/completions", {"model": "Org/Beta"})
    thread.join()
    assert status == 200 and body["served_by"] == "/m/beta"
    assert results["slow"][0] == 200, "the in-flight request must finish"
    assert results["slow"][1]["served_by"] == "/m/alpha"
    assert time.monotonic() - started > 0.8


def test_unknown_model_is_404_and_leaves_loaded_model_alone(router, supervisor):
    _json(router, "POST", "/v1/chat/completions", {"model": "Org/Alpha"})
    status, body = _json(router, "POST", "/v1/chat/completions", {"model": "gpt-4"})
    assert status == 404
    assert body["error"]["code"] == "model_not_found"
    assert "Org/Alpha" in body["error"]["message"]
    assert supervisor.current.model_id == "Org/Alpha"


def test_model_too_big_is_refused_before_stopping_the_loaded_one(router, supervisor):
    _json(router, "POST", "/v1/chat/completions", {"model": "Org/Alpha"})
    status, body = _json(router, "POST", "/v1/chat/completions", {"model": "Org/Huge"})
    assert status == 503 and body["error"]["code"] == "model_does_not_fit"
    assert supervisor.current.model_id == "Org/Alpha"
    status, _ = _json(router, "POST", "/v1/chat/completions", {"model": "Org/Alpha"})
    assert status == 200 and len(supervisor.spawned) == 1


def test_model_less_request_uses_loaded_model_else_asks(router, supervisor):
    status, body = _json(router, "POST", "/v1/chat/completions", {})
    assert status == 400 and body["error"]["code"] == "model_required"
    _json(router, "POST", "/v1/chat/completions", {"model": "Org/Beta"})
    status, body = _json(router, "POST", "/v1/chat/completions", {})
    assert status == 200 and body["served_by"] == "/m/beta"


def test_idle_model_is_unloaded_and_reloaded_on_demand(fake_script, supervisor):
    sup = ModelSupervisor(
        raw_argv=["serve", "--jit"],
        discover=_models,
        spawn=supervisor._spawn,
        ready_timeout_s=30.0,
        stop_grace_s=5.0,
        ram_gib=128.0,
        idle_unload_s=0.6,
    )
    try:
        model = sup.resolve("Org/Alpha")
        sup.acquire(model)
        first = supervisor.spawned[-1]
        time.sleep(1.2)
        assert sup.current == model, "a request in flight keeps the model loaded"
        sup.release()
        deadline = time.monotonic() + 10
        while sup.current is not None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert sup.current is None and first.poll() is not None
        sup.acquire(model)
        assert sup.current == model and supervisor.spawned[-1] is not first
        sup.release()
    finally:
        sup.shutdown()


def test_idle_flag_is_not_forwarded_to_children():
    argv = child_serve_argv(
        ["serve", "--jit", "--jit-idle-unload", "300", "--depth", "3"],
        model_ref="/m/a",
        port=1,
    )
    assert "--jit-idle-unload" not in argv and "300" not in argv


def test_streaming_request_gets_events_and_a_keepalive_while_loading(
    router, supervisor, monkeypatch
):
    monkeypatch.setattr(jit_router, "KEEPALIVE_INTERVAL_S", 0.2)
    status, data = _request(
        router, "POST", "/v1/chat/completions", {"model": "Org/Alpha", "stream": True}
    )
    text = data.decode()
    assert status == 200
    assert text.startswith(": loading model")
    assert '"t": "a"' in text and text.rstrip().endswith("data: [DONE]")


def test_streaming_load_failure_is_an_sse_error_frame(router, supervisor):
    supervisor._check_fits = lambda model: None
    supervisor._spawn = lambda argv: subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.exit(3)"]
    )
    status, data = _request(
        router, "POST", "/v1/chat/completions", {"model": "Org/Alpha", "stream": True}
    )
    assert status == 200
    assert "model_failed_to_load" in data.decode()
    assert data.decode().rstrip().endswith("data: [DONE]")


def test_models_lists_every_installed_model_and_marks_the_loaded_one(router, supervisor):
    status, body = _json(router, "GET", "/v1/models")
    assert status == 200
    assert [m["id"] for m in body["data"]] == ["Org/Alpha", "Org/Beta", "Org/Huge"]
    assert not any(m["loaded"] for m in body["data"])
    _json(router, "POST", "/v1/chat/completions", {"model": "Org/Beta"})
    _, body = _json(router, "GET", "/v1/models")
    loaded = [m for m in body["data"] if m["loaded"]]
    assert [m["id"] for m in loaded] == ["Org/Beta"]
    assert loaded[0]["context_length"] == 4096


def test_health_reports_router_with_or_without_a_model(router, supervisor):
    _, body = _json(router, "GET", "/health")
    assert body["ok"] is True and body["model"] is None
    assert body["jit"]["loaded_model"] is None
    _json(router, "POST", "/v1/messages", {"model": "Org/Alpha"})
    _, body = _json(router, "GET", "/health")
    assert body["model"] == "/m/alpha" and body["jit"]["loaded_model"] == "Org/Alpha"


def test_other_paths_pass_through_once_a_model_is_loaded(router, supervisor):
    status, body = _json(router, "GET", "/metrics")
    assert status == 503 and body["error"]["code"] == "no_model_loaded"
    _json(router, "POST", "/v1/chat/completions", {"model": "Org/Alpha"})
    status, body = _json(router, "GET", "/admin/sessions")
    assert status == 200 and body == {"path": "/admin/sessions"}


def test_api_key_is_enforced_by_the_router(supervisor):
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), jit_router.make_handler(supervisor, api_key="s3cret")
    )
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        assert _request(port, "GET", "/v1/models")[0] == 401
        assert _request(port, "GET", "/health")[0] == 200
        ok = _request(
            port, "GET", "/v1/models", headers={"Authorization": "Bearer s3cret"}
        )
        assert ok[0] == 200
    finally:
        server.shutdown()
        server.server_close()


def test_discover_lists_only_installed_models_and_honours_the_allowlist(tmp_path):
    for name in (
        "Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed",
        "Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Balance",
    ):
        (tmp_path / name).mkdir()
    from mtplx import model_catalog

    real = model_catalog._is_complete_install
    model_catalog._is_complete_install = lambda path: True
    try:
        found = jit_router.discover_models(cache_dir=str(tmp_path))
        ids = {m.model_id for m in found}
        assert ids == {
            "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed",
            "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance",
        }
        speed = next(m for m in found if m.model_id.endswith("27B-MTPLX-Optimized-Speed"))
        assert "mtplx-qwen38-27b-optimized-speed" in speed.names
        limited = jit_router.discover_models(
            cache_dir=str(tmp_path), allow=["Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"]
        )
        assert [m.model_id for m in limited] == [
            "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"
        ]
    finally:
        model_catalog._is_complete_install = real
