from __future__ import annotations

import socket
import sys
import time
from pathlib import Path

import pytest

from mtplx.supervisor.process import (
    EngineProcess,
    Liveness,
    RestartPolicy,
    RestartTracker,
)

FAKE_ENGINE = Path(__file__).parent / "fixtures" / "fake_engine.py"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _fake_argv(port: int, *extra: str) -> list[str]:
    return [sys.executable, str(FAKE_ENGINE), "--port", str(port), *extra]


def _wait_until(predicate, timeout_s: float = 5.0, interval_s: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


@pytest.fixture
def spawned():
    """Track EngineProcess instances so every test kills its child."""
    procs: list[EngineProcess] = []

    def make(port: int, argv_override: list[str]) -> EngineProcess:
        proc = EngineProcess("unused.gguf", port, [], argv_override=argv_override)
        proc.spawn()
        procs.append(proc)
        return proc

    yield make
    for proc in procs:
        try:
            proc.terminate(grace_s=1.0)
        except Exception:
            pass


def test_argv_default_shape():
    proc = EngineProcess(
        "/models/x.gguf", 5001, ["--profile", "fast"], python="/usr/bin/python3.11"
    )
    assert proc.argv == [
        "/usr/bin/python3.11",
        "-m",
        "mtplx.cli",
        "serve",
        "--model",
        "/models/x.gguf",
        "--host",
        "127.0.0.1",
        "--port",
        "5001",
        "--no-auth",
        "--yes",
        "--profile",
        "fast",
    ]


def test_argv_override():
    override = [sys.executable, "-c", "pass"]
    proc = EngineProcess("/models/x.gguf", 5001, [], argv_override=override)
    assert proc.argv == override


def test_spawn_probe_reaches_ready(spawned):
    port = _free_port()
    proc = spawned(port, _fake_argv(port))
    ready = _wait_until(lambda: proc.probe(timeout_s=0.3) == Liveness.READY, timeout_s=5.0)
    assert ready


def test_ready_after_delay_shows_not_ready_then_ready(spawned):
    port = _free_port()
    proc = spawned(port, _fake_argv(port, "--ready-after-s", "0.6"))
    results: list[Liveness] = []
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        verdict = proc.probe(timeout_s=0.2)
        results.append(verdict)
        if verdict == Liveness.READY:
            break
        time.sleep(0.05)
    assert results[-1] == Liveness.READY
    assert any(v in (Liveness.PORT_CLOSED, Liveness.BUSY) for v in results[:-1])


def test_hang_health_yields_busy_not_gone(spawned):
    port = _free_port()
    proc = spawned(port, _fake_argv(port, "--hang-health"))
    # Give the server a moment to bind before probing.
    _wait_until(lambda: proc.probe(timeout_s=0.2) != Liveness.PORT_CLOSED, timeout_s=3.0)
    verdict = proc.probe(timeout_s=0.3)
    assert verdict == Liveness.BUSY
    assert proc.is_alive()


def test_hang_health_unresponsive_for_s_grows(spawned):
    port = _free_port()
    proc = spawned(port, _fake_argv(port, "--hang-health"))
    _wait_until(lambda: proc.probe(timeout_s=0.2) != Liveness.PORT_CLOSED, timeout_s=3.0)
    first_verdict = proc.probe(timeout_s=0.2)
    assert first_verdict == Liveness.BUSY
    first = proc.unresponsive_for_s(time.monotonic())
    time.sleep(0.3)
    proc.probe(timeout_s=0.2)
    second = proc.unresponsive_for_s(time.monotonic())
    assert second > first


def test_exit_after_yields_gone_and_death_reason_exit(spawned):
    port = _free_port()
    proc = spawned(port, _fake_argv(port, "--exit-after-s", "0.2"))
    gone = _wait_until(lambda: proc.probe(timeout_s=0.2) == Liveness.GONE, timeout_s=5.0)
    assert gone
    assert proc.death_reason() == "exit"


def test_oom_death_reason(spawned):
    port = _free_port()
    proc = spawned(port, _fake_argv(port, "--oom"))
    gone = _wait_until(lambda: not proc.is_alive(), timeout_s=5.0)
    assert gone
    # Let the stderr reader thread catch up with the printed line.
    _wait_until(lambda: proc.death_reason() == "out_of_memory", timeout_s=2.0)
    assert proc.death_reason() == "out_of_memory"


def test_terminate_graceful_stop(spawned):
    port = _free_port()
    proc = spawned(port, _fake_argv(port))
    _wait_until(lambda: proc.probe(timeout_s=0.2) == Liveness.READY, timeout_s=5.0)
    code = proc.terminate(grace_s=2.0)
    assert not proc.is_alive()
    assert code is not None
    # Idempotent: calling again on an already-dead child is a no-op.
    assert proc.terminate(grace_s=1.0) == code


def test_terminate_escalates_to_sigkill_for_stubborn_child(tmp_path):
    script = tmp_path / "stubborn.py"
    script.write_text(
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "while True:\n"
        "    time.sleep(0.05)\n",
        encoding="utf-8",
    )
    proc = EngineProcess(
        "unused.gguf", _free_port(), [], argv_override=[sys.executable, str(script)]
    )
    proc.spawn()
    try:
        _wait_until(lambda: proc.is_alive(), timeout_s=3.0)
        started = time.monotonic()
        code = proc.terminate(grace_s=0.3)
        elapsed = time.monotonic() - started
        assert not proc.is_alive()
        assert code is not None
        assert elapsed < 5.0
    finally:
        if proc.is_alive():
            proc.terminate(grace_s=0.1)


def test_restart_tracker_delay_progression():
    tracker = RestartTracker(RestartPolicy())
    t0 = 1_000.0
    assert tracker.record_crash(t0) == pytest.approx(1.0)
    assert tracker.record_crash(t0 + 1) == pytest.approx(2.0)
    assert tracker.record_crash(t0 + 2) == pytest.approx(4.0)


def test_restart_tracker_exhausted_returns_none():
    tracker = RestartTracker(RestartPolicy())
    t0 = 1_000.0
    tracker.record_crash(t0)
    tracker.record_crash(t0 + 1)
    tracker.record_crash(t0 + 2)
    assert tracker.record_crash(t0 + 3) is None


def test_restart_tracker_generation_increments():
    tracker = RestartTracker(RestartPolicy())
    assert tracker.generation == 0
    tracker.record_crash(1_000.0)
    tracker.record_crash(1_001.0)
    assert tracker.generation == 2


def test_restart_tracker_forgets_crashes_outside_window():
    policy = RestartPolicy(
        max_attempts=3, initial_delay_s=1.0, max_delay_s=30.0, crash_window_s=10.0
    )
    tracker = RestartTracker(policy)
    tracker.record_crash(0.0)
    tracker.record_crash(5.0)
    # Both prior crashes are now older than crash_window_s; only this one counts.
    delay = tracker.record_crash(20.0)
    assert delay == pytest.approx(1.0)


def test_restart_tracker_reset():
    tracker = RestartTracker(RestartPolicy())
    tracker.record_crash(1_000.0)
    tracker.record_crash(1_001.0)
    tracker.reset()
    assert tracker.record_crash(2_000.0) == pytest.approx(1.0)
