"""Tests for mtplx.supervisor.admin: admin routes, /health, /v1/models,
crash-reap-and-restart, OOM handling, and idle unload.
"""

from __future__ import annotations

import contextlib
import os
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


def test_admin_401_without_key(tmp_path, two_packs, monkeypatch):
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1024, "pack-b": 1024})
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        preload=["pack-a"],
        default_model_id="pack-a",
        api_key="secret",
        engine_argv_override=_fake_argv(),
    )
    with running_supervisor(config) as (supervisor, base_url):
        resp = httpx.get(base_url + "/mtplx/admin/models", timeout=5.0)
        assert resp.status_code == 401


def test_admin_401_when_no_key_configured(tmp_path, two_packs, monkeypatch):
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1024, "pack-b": 1024})
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        preload=["pack-a"],
        default_model_id="pack-a",
        api_key=None,
        engine_argv_override=_fake_argv(),
    )
    with running_supervisor(config) as (supervisor, base_url):
        resp = httpx.get(base_url + "/mtplx/admin/models", timeout=5.0)
        assert resp.status_code == 401
        assert "api-key-file" in resp.json()["error"]["message"]


def test_admin_200_with_bearer_key(tmp_path, two_packs, monkeypatch):
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1024, "pack-b": 1024})
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        preload=["pack-a"],
        default_model_id="pack-a",
        api_key="secret",
        engine_argv_override=_fake_argv(),
    )
    with running_supervisor(config) as (supervisor, base_url):
        assert _wait_until(
            lambda: supervisor.registry.get("pack-a").state == EngineState.READY, timeout_s=10.0
        )
        resp = httpx.get(
            base_url + "/mtplx/admin/models",
            headers={"Authorization": "Bearer secret"},
            timeout=5.0,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["loaded"] == ["pack-a"]
        assert body["installed"] == ["pack-b"]
        assert "budget" in body


def test_admin_load_unload_restart(tmp_path, two_packs, monkeypatch):
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1024, "pack-b": 1024})
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        preload=["pack-a"],
        default_model_id="pack-a",
        api_key="secret",
        load_timeout_s=10.0,
        engine_argv_override=_fake_argv(),
    )
    headers = {"x-api-key": "secret"}
    with running_supervisor(config) as (supervisor, base_url):
        assert _wait_until(
            lambda: supervisor.registry.get("pack-a").state == EngineState.READY, timeout_s=10.0
        )
        load_resp = httpx.post(
            base_url + "/mtplx/admin/load", json={"model": "pack-b"}, headers=headers, timeout=5.0
        )
        assert load_resp.status_code in (200, 202)
        assert _wait_until(
            lambda: supervisor.registry.get("pack-b").state == EngineState.READY, timeout_s=10.0
        )
        load_again = httpx.post(
            base_url + "/mtplx/admin/load", json={"model": "pack-b"}, headers=headers, timeout=5.0
        )
        assert load_again.status_code == 200

        unload_resp = httpx.post(
            base_url + "/mtplx/admin/unload", json={"model": "pack-b"}, headers=headers, timeout=15.0
        )
        assert unload_resp.status_code == 200
        assert supervisor.registry.get("pack-b").state == EngineState.STOPPED

        restart_resp = httpx.post(
            base_url + "/mtplx/admin/restart", json={"model": "pack-a"}, headers=headers, timeout=15.0
        )
        assert restart_resp.status_code == 200
        assert restart_resp.json()["state"] == "ready"


def test_admin_unload_default_returns_409(tmp_path, two_packs, monkeypatch):
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1024, "pack-b": 1024})
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        preload=["pack-a"],
        default_model_id="pack-a",
        api_key="secret",
        engine_argv_override=_fake_argv(),
    )
    headers = {"x-api-key": "secret"}
    with running_supervisor(config) as (supervisor, base_url):
        assert _wait_until(
            lambda: supervisor.registry.get("pack-a").state == EngineState.READY, timeout_s=10.0
        )
        resp = httpx.post(
            base_url + "/mtplx/admin/unload", json={"model": "pack-a"}, headers=headers, timeout=5.0
        )
        assert resp.status_code == 409


def test_v1_models_merge_shows_loaded_flags(tmp_path, two_packs, monkeypatch):
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1024, "pack-b": 1024})
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        preload=["pack-a"],
        default_model_id="pack-a",
        engine_argv_override=_fake_argv(),
    )
    with running_supervisor(config) as (supervisor, base_url):
        assert _wait_until(
            lambda: supervisor.registry.get("pack-a").state == EngineState.READY, timeout_s=10.0
        )
        resp = httpx.get(base_url + "/v1/models", timeout=5.0)
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        by_id = {row["id"]: row for row in body["data"]}
        assert by_id["pack-a"]["loaded"] is True
        assert by_id["pack-b"]["loaded"] is False


def test_health_shape(tmp_path, two_packs, monkeypatch):
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1024, "pack-b": 1024})
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        preload=["pack-a"],
        default_model_id="pack-a",
        launch_id="launch-123",
        engine_argv_override=_fake_argv(),
    )
    with running_supervisor(config) as (supervisor, base_url):
        resp = httpx.get(base_url + "/health", timeout=5.0)
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["model"] == "pack-a"
        assert body["startup"]["pid"] == os.getpid()
        assert body["startup"]["launch_id"] == "launch-123"
        assert "supervisor" in body
        assert "engines" in body["supervisor"]


def test_insecure_lan_flag_shows_in_health(tmp_path, two_packs, monkeypatch):
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1024, "pack-b": 1024})
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        preload=["pack-a"],
        default_model_id="pack-a",
        insecure_lan=True,
        engine_argv_override=_fake_argv(),
    )
    with running_supervisor(config) as (supervisor, base_url):
        resp = httpx.get(base_url + "/health", timeout=5.0)
        assert resp.json()["insecure_lan"] is True


def test_crash_reap_and_restart(tmp_path, two_packs, monkeypatch):
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1024, "pack-b": 1024})
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        preload=["pack-a"],
        default_model_id="pack-a",
        load_timeout_s=10.0,
        engine_argv_override=_fake_argv("--exit-after-s", "1"),
    )
    with running_supervisor(config, health_sweep_interval_s=0.2) as (supervisor, base_url):
        assert _wait_until(
            lambda: supervisor.registry.get("pack-a").state == EngineState.READY, timeout_s=10.0
        )
        first_pid = supervisor.registry.get("pack-a").pid
        # The fake engine exits at 1s; the health sweep should notice, mark
        # a crash, schedule a restart, and bring it back to READY under a
        # different pid (proving a fresh process was spawned, not reuse).
        assert _wait_until(
            lambda: supervisor.registry.get("pack-a").state == EngineState.READY
            and supervisor.registry.get("pack-a").pid != first_pid,
            timeout_s=15.0,
        )


def test_oom_marks_failed_without_restart(tmp_path, two_packs, monkeypatch):
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1024, "pack-b": 1024})
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        preload=[],
        default_model_id="pack-a",
        load_timeout_s=10.0,
        engine_argv_override=_fake_argv("--oom"),
    )
    with running_supervisor(config, health_sweep_interval_s=0.2) as (supervisor, base_url):
        resp = httpx.post(base_url + "/v1/chat/completions", json={"model": "pack-a"}, timeout=15.0)
        assert resp.status_code == 503
        record = supervisor.registry.get("pack-a")
        assert record.state == EngineState.FAILED
        assert record.failure_reason == "out_of_memory"
        # Give the sweep a couple of cycles: it must NOT resurrect an OOM engine.
        time.sleep(1.0)
        assert supervisor.registry.get("pack-a").state == EngineState.FAILED


def test_idle_sweep_unloads_after_ttl(tmp_path, two_packs, monkeypatch):
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1024, "pack-b": 1024})
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        preload=["pack-a", "pack-b"],
        default_model_id="pack-a",
        idle_ttl_s=1.0,
        load_timeout_s=10.0,
        engine_argv_override=_fake_argv(),
    )
    with running_supervisor(config, idle_sweep_interval_s=0.3) as (supervisor, base_url):
        assert _wait_until(
            lambda: supervisor.registry.get("pack-b").state == EngineState.READY, timeout_s=10.0
        )
        # pack-b is not the default; once idle past the ttl it should unload.
        assert _wait_until(
            lambda: supervisor.registry.get("pack-b").state == EngineState.STOPPED, timeout_s=10.0
        )
        # the default engine is exempt and stays READY.
        assert supervisor.registry.get("pack-a").state == EngineState.READY


def test_pins_block_idle_unload(tmp_path, two_packs, monkeypatch):
    a, b = two_packs
    _patch_sizes(monkeypatch, {"pack-a": 1024, "pack-b": 1024})
    config = SupervisorConfig(
        host="127.0.0.1",
        port=_free_port(),
        model_paths=[a, b],
        preload=["pack-a", "pack-b"],
        default_model_id="pack-a",
        idle_ttl_s=1.0,
        load_timeout_s=10.0,
        engine_argv_override=_fake_argv(),
    )
    with running_supervisor(config, idle_sweep_interval_s=0.3) as (supervisor, base_url):
        assert _wait_until(
            lambda: supervisor.registry.get("pack-b").state == EngineState.READY, timeout_s=10.0
        )
        supervisor.registry.pin("pack-b")
        try:
            time.sleep(2.0)
            assert supervisor.registry.get("pack-b").state == EngineState.READY
        finally:
            supervisor.registry.unpin("pack-b")
