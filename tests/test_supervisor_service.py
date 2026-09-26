"""Tests for mtplx.supervisor.service.Supervisor: the concurrency-sensitive
seams found by the code/security review that admin/proxy integration tests
don't exercise directly -- C1 (unload vs JIT-load race), H3/H4 (restart
locking + RestartTracker.generation), M4 (port-collision retry), R1 (alias
registration), and L2 (concurrent health-sweep probing).

Drives ``Supervisor`` methods directly with asyncio rather than through
HTTP, per the fix backlog's guidance -- these are unit tests of the
Supervisor class, not proxy/admin integration tests (those stay in
test_supervisor_proxy.py / test_supervisor_admin.py, owned elsewhere).

No pytest-asyncio plugin is installed in this project, so each test is a
plain sync function that drives its async body with ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
import socket
import sys
from pathlib import Path

import httpx

from mtplx.supervisor import service as service_mod
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


async def _wait_until(predicate, timeout_s: float = 5.0, interval_s: float = 0.05) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval_s)
    return predicate()


async def _concurrent_unload_and_jit_load_same_model(tmp_path):
    pack = _make_pack(tmp_path, "model-a")
    config = SupervisorConfig(
        model_paths=[pack], engine_argv_override=_fake_argv(), load_timeout_s=10.0
    )
    supervisor = Supervisor(config)
    await supervisor.start()
    try:
        await supervisor.ensure_loaded("model-a")
        proc1 = supervisor._processes["model-a"]
        assert proc1.is_alive()

        supervisor.registry.pin("model-a")
        unload_task = asyncio.ensure_future(supervisor._unload("model-a", drain_s=2.0))
        assert await _wait_until(
            lambda: supervisor.registry.get("model-a").state == EngineState.DRAINING
        )

        load_task = asyncio.ensure_future(supervisor.ensure_loaded("model-a"))
        await asyncio.sleep(0.2)
        # The JIT load must not have raced ahead while still draining.
        assert not load_task.done()

        supervisor.registry.unpin("model-a")
        await unload_task
        record2 = await load_task

        assert record2.state == EngineState.READY
        proc2 = supervisor._processes["model-a"]
        assert proc2 is not proc1
        assert not proc1.is_alive()
        assert proc2.is_alive()
    finally:
        await supervisor.stop()


def test_concurrent_unload_and_jit_load_same_model(tmp_path):
    """C1: a request that arrives while a model is DRAINING must not spawn
    a second engine underneath the drain. It waits for the drain's load
    lock, then loads a fresh engine only once the old one is fully
    terminated -- exactly one live process survives, and it is not the
    process the drain terminated."""
    asyncio.run(_concurrent_unload_and_jit_load_same_model(tmp_path))


async def _restart_races_inflight_ensure_loaded(tmp_path):
    pack = _make_pack(tmp_path, "model-a")
    config = SupervisorConfig(
        model_paths=[pack], engine_argv_override=_fake_argv(), load_timeout_s=10.0
    )
    supervisor = Supervisor(config)
    await supervisor.start()
    try:
        await supervisor.ensure_loaded("model-a")
        proc1 = supervisor._processes["model-a"]
        tracker = supervisor._trackers["model-a"]
        generation_before = tracker.generation

        restart_task = asyncio.ensure_future(supervisor.restart("model-a"))
        load_task = asyncio.ensure_future(supervisor.ensure_loaded("model-a"))
        record_restart, record_load = await asyncio.gather(restart_task, load_task)

        assert record_restart.state == EngineState.READY
        assert record_load.state == EngineState.READY
        assert tracker.generation == generation_before + 1

        live = [p for p in [proc1, supervisor._processes["model-a"]] if p.is_alive()]
        assert len(live) == 1
    finally:
        await supervisor.stop()


def test_restart_races_inflight_ensure_loaded(tmp_path):
    """H3/H4: restart() takes the same per-model load lock as
    ensure_loaded, so a restart racing a concurrent JIT load for the same
    model id can never interleave with it -- both complete cleanly and
    exactly one engine ends up READY, and RestartTracker.generation bumps
    from reset()."""
    asyncio.run(_restart_races_inflight_ensure_loaded(tmp_path))


async def _port_collision_retries_on_a_fresh_port(monkeypatch, tmp_path):
    pack = _make_pack(tmp_path, "model-a")
    busy_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    busy_socket.bind(("127.0.0.1", 0))
    busy_socket.listen(1)
    busy_port = busy_socket.getsockname()[1]

    real_free_port = service_mod._free_port
    ports = iter([busy_port])

    def _fake_free_port() -> int:
        try:
            return next(ports)
        except StopIteration:
            return real_free_port()

    monkeypatch.setattr(service_mod, "_free_port", _fake_free_port)

    config = SupervisorConfig(
        model_paths=[pack], engine_argv_override=_fake_argv(), load_timeout_s=10.0
    )
    supervisor = Supervisor(config)
    await supervisor.start()
    try:
        record = await supervisor.ensure_loaded("model-a")
        assert record.state == EngineState.READY
    finally:
        busy_socket.close()
        await supervisor.stop()


def test_port_collision_retries_on_a_fresh_port(monkeypatch, tmp_path):
    """M4: _free_port()'s TOCTOU can hand out a port that is already bound
    by the time the child tries to use it. ensure_loaded must retry on a
    fresh port instead of surfacing an opaque failure."""
    asyncio.run(_port_collision_retries_on_a_fresh_port(monkeypatch, tmp_path))


async def _service_registers_engine_alias_on_ready(tmp_path):
    pack = _make_pack(tmp_path, "model-a")
    config = SupervisorConfig(
        model_paths=[pack],
        engine_argv_override=_fake_argv("--alias-id", "served-name"),
        load_timeout_s=10.0,
    )
    supervisor = Supervisor(config)
    await supervisor.start()
    try:
        await supervisor.ensure_loaded("model-a")
        assert "served-name" in supervisor.registry.get("model-a").aliases
        record, verdict = supervisor.registry.resolve(
            "served-name", default_id="model-a", strict=False
        )
        assert verdict == "loaded"
        assert record.spec.model_id == "model-a"
    finally:
        await supervisor.stop()


def test_service_registers_engine_alias_on_ready(tmp_path):
    """R1: once an engine reaches READY, the supervisor registers the
    engine's own reported served id (first entry of its `/v1/models`) as
    an alias, so a client can address the engine by either name."""
    asyncio.run(_service_registers_engine_alias_on_ready(tmp_path))


async def _ensure_loaded_waits_for_warmup_before_ready(tmp_path):
    pack = _make_pack(tmp_path, "model-a")
    config = SupervisorConfig(
        model_paths=[pack],
        engine_argv_override=_fake_argv("--warmup-s", "0.8"),
        load_timeout_s=10.0,
    )
    supervisor = Supervisor(config)
    # Deliberately skip start(): it would auto-preload this single pack as
    # the default model, finishing before this test can measure the
    # warmup-gated load time. A bare http_client is all ensure_loaded needs.
    supervisor.http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0, read=None))
    try:
        loop = asyncio.get_running_loop()
        started = loop.time()
        record = await supervisor.ensure_loaded("model-a")
        elapsed = loop.time() - started
        assert record.state == EngineState.READY
        assert elapsed >= 0.7
    finally:
        if supervisor.http_client is not None:
            await supervisor.http_client.aclose()
        proc = supervisor._processes.get("model-a")
        if proc is not None:
            await asyncio.to_thread(proc.terminate, 2.0)


def test_ensure_loaded_waits_for_warmup_before_ready(tmp_path):
    """R3: ensure_loaded's own probe loop stays LOADING (not READY) until
    the engine's warmup signal flips true, mirroring process.py's gate."""
    asyncio.run(_ensure_loaded_waits_for_warmup_before_ready(tmp_path))


async def _health_sweep_reaps_dead_engine_without_disturbing_others(tmp_path):
    pack_a = _make_pack(tmp_path, "model-a")
    pack_b = _make_pack(tmp_path, "model-b")
    config = SupervisorConfig(
        model_paths=[pack_a, pack_b], engine_argv_override=_fake_argv(), load_timeout_s=10.0
    )
    supervisor = Supervisor(config)
    await supervisor.start()
    try:
        await supervisor.ensure_loaded("model-a")
        await supervisor.ensure_loaded("model-b")
        proc_a = supervisor._processes["model-a"]
        proc_b = supervisor._processes["model-b"]

        await asyncio.to_thread(proc_a.terminate, 2.0)
        await supervisor._health_sweep_once()

        record_a = supervisor.registry.get("model-a")
        record_b = supervisor.registry.get("model-b")
        assert record_a.state in (EngineState.INSTALLED, EngineState.FAILED)
        assert record_b.state == EngineState.READY
        assert proc_b.is_alive()
    finally:
        await supervisor.stop()


def test_health_sweep_reaps_dead_engine_without_disturbing_others(tmp_path):
    """L2: probes run concurrently (asyncio.gather); this checks the sweep
    is still correct under concurrency -- one dead engine among several is
    reaped without touching a healthy sibling probed in the same pass."""
    asyncio.run(_health_sweep_reaps_dead_engine_without_disturbing_others(tmp_path))
