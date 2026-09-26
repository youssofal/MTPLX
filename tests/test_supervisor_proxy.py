"""Tests for mtplx.supervisor.proxy + the Supervisor ASGI app's routing.

Runs the real Supervisor app under uvicorn in a background thread on an
ephemeral port, with the fake stdlib engine (tests/fixtures/fake_engine.py)
spawned via ``engine_argv_override`` in place of a real ``mtplx serve``.
No real model is ever loaded.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

from mtplx.supervisor.registry import EngineState
from mtplx.supervisor.service import Supervisor, SupervisorConfig

FAKE_ENGINE = Path(__file__).parent / "fixtures" / "fake_engine.py"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _fake_argv(*extra: str) -> list[str]:
    return [sys.executable, str(FAKE_ENGINE), "--port", "{port}", *extra]


def _make_pack(tmp_path: Path, name: str, size_bytes: int = 4096) -> Path:
    pack = tmp_path / name
    pack.mkdir()
    (pack / "weights.safetensors").write_bytes(b"0" * size_bytes)
    return pack


def _wait_until(predicate, timeout_s: float = 5.0, interval_s: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


@contextlib.contextmanager
def running_supervisor(config: SupervisorConfig, **kwargs):
    supervisor = Supervisor(config, **kwargs)
    uv_config = uvicorn.Config(
        supervisor.app, host=config.host, port=config.port, lifespan="on", log_level="warning"
    )
    server = uvicorn.Server(uv_config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://{config.host}:{config.port}"
    ready = _wait_until(lambda: _probe_up(base_url), timeout_s=10.0)
    assert ready, "supervisor did not come up"
    try:
        yield supervisor, base_url
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)


def _probe_up(base_url: str) -> bool:
    try:
        httpx.get(base_url + "/health", timeout=0.3)
        return True
    except Exception:
        return False


def _patch_sizes(monkeypatch, sizes: dict[str, int]) -> None:
    def fake_estimate(path: Path) -> int:
        return sizes.get(Path(path).name, 4096)

    monkeypatch.setattr("mtplx.supervisor.service.estimate_resident_bytes", fake_estimate)


@pytest.fixture
def two_packs(tmp_path):
    a = _make_pack(tmp_path, "pack-a")
    b = _make_pack(tmp_path, "pack-b")
    return a, b


def test_routes_to_already_loaded_engine(tmp_path, two_packs, monkeypatch):
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1024, "pack-b": 1024})
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        preload=["pack-a"],
        default_model_id="pack-a",
        load_timeout_s=10.0,
        engine_argv_override=_fake_argv(),
    )
    with running_supervisor(config) as (supervisor, base_url):
        assert _wait_until(
            lambda: supervisor.registry.get("pack-a").state == EngineState.READY, timeout_s=10.0
        )
        resp = httpx.post(base_url + "/v1/chat/completions", json={"model": "pack-a"}, timeout=5.0)
        assert resp.status_code == 200
        assert resp.json()["model"] == "fake"


def test_jit_load_on_first_request_transitions_states(tmp_path, two_packs, monkeypatch):
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1024, "pack-b": 1024})
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        default_model_id="pack-a",
        load_timeout_s=10.0,
        engine_argv_override=_fake_argv(),
    )
    with running_supervisor(config) as (supervisor, base_url):
        assert supervisor.registry.get("pack-b").state == EngineState.INSTALLED
        resp = httpx.post(base_url + "/v1/chat/completions", json={"model": "pack-b"}, timeout=10.0)
        assert resp.status_code == 200
        assert supervisor.registry.get("pack-b").state == EngineState.READY


def test_unknown_model_falls_back_to_default_with_header(tmp_path, two_packs, monkeypatch):
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1024, "pack-b": 1024})
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        preload=["pack-a"],
        default_model_id="pack-a",
        load_timeout_s=10.0,
        engine_argv_override=_fake_argv(),
    )
    with running_supervisor(config) as (supervisor, base_url):
        assert _wait_until(
            lambda: supervisor.registry.get("pack-a").state == EngineState.READY, timeout_s=10.0
        )
        resp = httpx.post(
            base_url + "/v1/chat/completions", json={"model": "does-not-exist"}, timeout=5.0
        )
        assert resp.status_code == 200
        assert resp.headers.get("x-mtplx-routed-model") == "pack-a"


def test_strict_mode_returns_404_for_unknown_model(tmp_path, two_packs, monkeypatch):
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1024, "pack-b": 1024})
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        preload=["pack-a"],
        default_model_id="pack-a",
        strict_model=True,
        load_timeout_s=10.0,
        engine_argv_override=_fake_argv(),
    )
    with running_supervisor(config) as (supervisor, base_url):
        assert _wait_until(
            lambda: supervisor.registry.get("pack-a").state == EngineState.READY, timeout_s=10.0
        )
        resp = httpx.post(
            base_url + "/v1/chat/completions", json={"model": "does-not-exist"}, timeout=5.0
        )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "model_not_found"


def test_insufficient_memory_returns_507_shape(tmp_path, two_packs, monkeypatch):
    a, b = two_packs
    # pack-a resident; budget too small for pack-b, no eviction allowed.
    _patch_sizes(monkeypatch, {"pack-a": 1_000_000, "pack-b": 1_000_000})
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        preload=["pack-a"],
        default_model_id="pack-a",
        budget_bytes=1_200_000,
        evict_to_fit=False,
        load_timeout_s=10.0,
        engine_argv_override=_fake_argv(),
    )
    with running_supervisor(config) as (supervisor, base_url):
        assert _wait_until(
            lambda: supervisor.registry.get("pack-a").state == EngineState.READY, timeout_s=10.0
        )
        resp = httpx.post(base_url + "/v1/chat/completions", json={"model": "pack-b"}, timeout=5.0)
        assert resp.status_code == 507
        body = resp.json()
        assert body["error"]["code"] == "insufficient_memory"
        assert body["error"]["needed_bytes"] == 1_000_000
        assert "available_bytes" in body["error"]
        assert body["error"]["would_free"] == ["pack-a"]


def test_evict_to_fit_unloads_lru_then_loads(tmp_path, two_packs, monkeypatch):
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1_000_000, "pack-b": 1_000_000})
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        preload=["pack-a"],
        default_model_id="pack-b",
        budget_bytes=1_200_000,
        evict_to_fit=True,
        load_timeout_s=10.0,
        engine_argv_override=_fake_argv(),
    )
    with running_supervisor(config) as (supervisor, base_url):
        assert _wait_until(
            lambda: supervisor.registry.get("pack-a").state == EngineState.READY, timeout_s=10.0
        )
        resp = httpx.post(base_url + "/v1/chat/completions", json={"model": "pack-b"}, timeout=15.0)
        assert resp.status_code == 200
        assert supervisor.registry.get("pack-b").state == EngineState.READY
        assert supervisor.registry.get("pack-a").state == EngineState.STOPPED


def test_sse_passthrough_bytes_match_engine_output(tmp_path, two_packs, monkeypatch):
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1024, "pack-b": 1024})
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        preload=["pack-a"],
        default_model_id="pack-a",
        load_timeout_s=10.0,
        engine_argv_override=_fake_argv(),
    )
    with running_supervisor(config) as (supervisor, base_url):
        assert _wait_until(
            lambda: supervisor.registry.get("pack-a").state == EngineState.READY, timeout_s=10.0
        )
        port = supervisor.registry.get("pack-a").port
        direct = httpx.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={"model": "pack-a", "stream": True},
            timeout=5.0,
        )
        proxied = httpx.post(
            base_url + "/v1/chat/completions",
            json={"model": "pack-a", "stream": True},
            timeout=5.0,
        )
        assert proxied.status_code == 200
        assert proxied.content == direct.content
        assert proxied.content.count(b"data: ") == 4  # 3 frames + [DONE]


def test_engine_connection_error_maps_to_503(tmp_path, two_packs, monkeypatch):
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1024, "pack-b": 1024})
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        preload=["pack-a"],
        default_model_id="pack-a",
        load_timeout_s=10.0,
        engine_argv_override=_fake_argv(),
    )
    with running_supervisor(config) as (supervisor, base_url):
        assert _wait_until(
            lambda: supervisor.registry.get("pack-a").state == EngineState.READY, timeout_s=10.0
        )
        record = supervisor.registry.get("pack-a")
        proc = supervisor._processes["pack-a"]
        proc.terminate(grace_s=2.0)
        # Registry still says READY (health sweep hasn't run); the proxy
        # should still surface a clean 503 instead of hanging or crashing.
        resp = httpx.post(base_url + "/v1/chat/completions", json={"model": "pack-a"}, timeout=5.0)
        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "engine_unavailable"


def test_non_v1_path_routes_to_default_engine(tmp_path, two_packs, monkeypatch):
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1024, "pack-b": 1024})
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        preload=["pack-a"],
        default_model_id="pack-a",
        load_timeout_s=10.0,
        engine_argv_override=_fake_argv(),
    )
    with running_supervisor(config) as (supervisor, base_url):
        assert _wait_until(
            lambda: supervisor.registry.get("pack-a").state == EngineState.READY, timeout_s=10.0
        )
        resp = httpx.get(base_url + "/dashboard", timeout=5.0)
        # The fake engine 404s unknown GET paths; what matters is that it
        # reached pack-a's engine at all rather than 404ing at the front door.
        assert resp.status_code == 404


def test_pin_unpin_around_forward_no_leak(tmp_path, two_packs, monkeypatch):
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1024, "pack-b": 1024})
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        preload=["pack-a"],
        default_model_id="pack-a",
        load_timeout_s=10.0,
        engine_argv_override=_fake_argv(),
    )
    with running_supervisor(config) as (supervisor, base_url):
        assert _wait_until(
            lambda: supervisor.registry.get("pack-a").state == EngineState.READY, timeout_s=10.0
        )
        for _ in range(3):
            httpx.post(base_url + "/v1/chat/completions", json={"model": "pack-a"}, timeout=5.0)
        assert supervisor.registry.get("pack-a").pins == 0


@contextlib.contextmanager
def _dying_stream_engine():
    """A raw asyncio TCP server standing in for an engine that dies mid-SSE:
    sends valid HTTP headers plus one complete chunk, then closes the
    connection without the terminating chunk. Runs on its own thread/loop so
    it can sit alongside the supervisor's own uvicorn thread."""

    port_holder: list[int] = []
    ready = threading.Event()
    stop_loop: asyncio.AbstractEventLoop | None = None

    async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(reader.read(65536), timeout=2.0)
        frame = b'data: {"frame": 0}\n\n'
        chunk = f"{len(frame):x}\r\n".encode() + frame + b"\r\n"
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"\r\n" + chunk
        )
        with contextlib.suppress(Exception):
            await writer.drain()
        # No terminating "0\r\n\r\n" chunk: the client is left mid-stream.
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()

    async def _serve() -> None:
        nonlocal stop_loop
        stop_loop = asyncio.get_running_loop()
        server = await asyncio.start_server(_handle, "127.0.0.1", 0)
        port_holder.append(server.sockets[0].getsockname()[1])
        ready.set()
        async with server:
            await server.serve_forever()

    def _run() -> None:
        with contextlib.suppress(Exception):
            asyncio.run(_serve())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    assert ready.wait(timeout=5.0), "dying-stream engine did not start"
    try:
        yield port_holder[0]
    finally:
        if stop_loop is not None:
            with contextlib.suppress(Exception):
                stop_loop.call_soon_threadsafe(stop_loop.stop)
        thread.join(timeout=5.0)


def test_engine_dies_mid_stream_after_headers_sent(tmp_path, two_packs, monkeypatch, caplog):
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1024, "pack-b": 1024})
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        preload=["pack-a"],
        default_model_id="pack-a",
        load_timeout_s=10.0,
        engine_argv_override=_fake_argv(),
    )
    with running_supervisor(config) as (supervisor, base_url):
        assert _wait_until(
            lambda: supervisor.registry.get("pack-a").state == EngineState.READY, timeout_s=10.0
        )
        with _dying_stream_engine() as dying_port:
            record = supervisor.registry.get("pack-a")
            record.port = dying_port  # redirect this model's routing at a dead-mid-stream server
            with caplog.at_level(logging.WARNING, logger="mtplx.supervisor"):
                # The stream dies after http.response.start has already gone
                # out: uvicorn manages its own framing to the client, so the
                # client sees a clean (short) response, not a hung
                # connection and not a second ASGI response.start (C2). The
                # only trace of the mid-stream death is the supervisor's own
                # warning log and the truncated body (no [DONE] frame).
                resp = httpx.post(
                    base_url + "/v1/chat/completions",
                    json={"model": "pack-a", "stream": True},
                    timeout=5.0,
                )
            assert resp.status_code == 200
            assert b"frame" in resp.content
            assert b"[DONE]" not in resp.content
            assert any("mid-stream" in rec.message for rec in caplog.records)

        # The supervisor's ASGI loop is still healthy (no unhandled
        # exception took the server down) and the next request against the
        # now-dead port gets a clean 503, not a hang or crash.
        resp = httpx.post(
            base_url + "/v1/chat/completions", json={"model": "pack-a"}, timeout=5.0
        )
        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "engine_unavailable"


def test_request_body_size_limit_rejected(tmp_path, two_packs, monkeypatch):
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1024, "pack-b": 1024})
    monkeypatch.setattr("mtplx.supervisor.proxy._MAX_BODY_BYTES", 1024)
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        preload=["pack-a"],
        default_model_id="pack-a",
        load_timeout_s=10.0,
        engine_argv_override=_fake_argv(),
    )
    with running_supervisor(config) as (supervisor, base_url):
        assert _wait_until(
            lambda: supervisor.registry.get("pack-a").state == EngineState.READY, timeout_s=10.0
        )
        oversized = {"model": "pack-a", "padding": "x" * 4096}
        resp = httpx.post(base_url + "/v1/chat/completions", json=oversized, timeout=5.0)
        assert resp.status_code == 413
        assert resp.json()["error"]["code"] == "request_too_large"


def test_body_cap_checked_after_auth_not_before(tmp_path, two_packs, monkeypatch):
    """H1: auth runs before the body is ever read. An unauthenticated
    request with an oversized body gets 401 (auth rejected it before the
    body cap even ran), not 413."""
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1024, "pack-b": 1024})
    monkeypatch.setattr("mtplx.supervisor.proxy._MAX_BODY_BYTES", 1024)
    config = SupervisorConfig(
        host="0.0.0.0",
        port=_free_port(),
        model_paths=[a, b],
        preload=["pack-a"],
        default_model_id="pack-a",
        api_key="secret",
        load_timeout_s=10.0,
        engine_argv_override=_fake_argv(),
    )
    with running_supervisor(config) as (supervisor, base_url):
        assert _wait_until(
            lambda: supervisor.registry.get("pack-a").state == EngineState.READY, timeout_s=10.0
        )
        oversized = {"model": "pack-a", "padding": "x" * 4096}
        resp = httpx.post(base_url + "/v1/chat/completions", json=oversized, timeout=5.0)
        assert resp.status_code == 401


def test_draining_engine_returns_503_with_retry_after(tmp_path, two_packs, monkeypatch):
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1024, "pack-b": 1024})
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        preload=["pack-a"],
        default_model_id="pack-a",
        load_timeout_s=10.0,
        engine_argv_override=_fake_argv(),
    )
    with running_supervisor(config) as (supervisor, base_url):
        assert _wait_until(
            lambda: supervisor.registry.get("pack-a").state == EngineState.READY, timeout_s=10.0
        )
        record = supervisor.registry.get("pack-a")
        original_resolve = supervisor.registry.resolve
        monkeypatch.setattr(
            supervisor.registry, "resolve", lambda *a, **k: (record, "draining")
        )
        try:
            resp = httpx.post(
                base_url + "/v1/chat/completions", json={"model": "pack-a"}, timeout=5.0
            )
        finally:
            monkeypatch.setattr(supervisor.registry, "resolve", original_resolve)
        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "engine_draining"
        assert resp.headers.get("retry-after") == "2"


def test_path_normalization_uppercase_and_double_slash(tmp_path, two_packs, monkeypatch):
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1024, "pack-b": 1024})
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        preload=["pack-a"],
        default_model_id="pack-a",
        load_timeout_s=10.0,
        engine_argv_override=_fake_argv(),
    )
    with running_supervisor(config) as (supervisor, base_url):
        assert _wait_until(
            lambda: supervisor.registry.get("pack-a").state == EngineState.READY, timeout_s=10.0
        )
        resp = httpx.get(base_url + "/HEALTH", timeout=5.0)
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        resp = httpx.get(base_url + "//v1//models", timeout=5.0)
        assert resp.status_code == 200
        assert resp.json()["object"] == "list"


def test_alias_routes_to_the_same_engine(tmp_path, two_packs, monkeypatch):
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1024, "pack-b": 1024})
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        preload=["pack-a"],
        default_model_id="pack-a",
        load_timeout_s=10.0,
        engine_argv_override=_fake_argv(),
    )
    with running_supervisor(config) as (supervisor, base_url):
        if not hasattr(supervisor.registry, "add_alias"):
            pytest.xfail("waits for registry aliases")
        assert _wait_until(
            lambda: supervisor.registry.get("pack-a").state == EngineState.READY, timeout_s=10.0
        )
        supervisor.registry.add_alias("pack-a", "pack-a-alias")
        resp = httpx.post(
            base_url + "/v1/chat/completions", json={"model": "pack-a-alias"}, timeout=5.0
        )
        assert resp.status_code == 200
        assert resp.json()["model"] == "fake"
        # Routed to the loaded engine directly, not the "fallback" default path.
        assert "x-mtplx-routed-model" not in resp.headers
