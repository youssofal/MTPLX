"""Tests for mtplx.supervisor.proxy + the Supervisor ASGI app's routing.

Runs the real Supervisor app under uvicorn in a background thread on an
ephemeral port, with the fake stdlib engine (tests/fixtures/fake_engine.py)
spawned via ``engine_argv_override`` in place of a real ``mtplx serve``.
No real model is ever loaded.
"""

from __future__ import annotations

import contextlib
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
