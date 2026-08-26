from __future__ import annotations

from threading import Event, get_ident
import time
from types import SimpleNamespace

from mtplx.model_scheduler import ModelWorkScheduler
from mtplx.server import openai


def test_foreground_runs_before_pending_idle_postcommit():
    scheduler = ModelWorkScheduler(name="test-model-scheduler", idle_grace_s=0.05)
    order: list[str] = []
    try:
        idle = scheduler.submit_idle_postcommit(lambda: order.append("idle"))
        foreground = scheduler.submit_foreground(lambda: order.append("foreground"))

        foreground.result(timeout=2)
        idle.result(timeout=2)

        assert order == ["foreground", "idle"]
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_idle_postcommit_does_not_start_while_foreground_is_queued():
    scheduler = ModelWorkScheduler(name="test-model-scheduler", idle_grace_s=0.01)
    started = Event()
    release = Event()
    order: list[str] = []

    def first_foreground() -> None:
        order.append("foreground-1-start")
        started.set()
        assert release.wait(timeout=2)
        order.append("foreground-1-end")

    try:
        first = scheduler.submit_foreground(first_foreground)
        assert started.wait(timeout=2)
        idle = scheduler.submit_idle_postcommit(lambda: order.append("idle"))
        second = scheduler.submit_foreground(lambda: order.append("foreground-2"))
        time.sleep(0.05)

        assert not idle.done()
        release.set()
        first.result(timeout=2)
        second.result(timeout=2)
        idle.result(timeout=2)

        assert order == [
            "foreground-1-start",
            "foreground-1-end",
            "foreground-2",
            "idle",
        ]
    finally:
        release.set()
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_foreground_fifo_order_is_preserved():
    scheduler = ModelWorkScheduler(name="test-model-scheduler", idle_grace_s=0.0)
    order: list[int] = []
    try:
        futures = [
            scheduler.submit_foreground(lambda value=value: order.append(value))
            for value in (1, 2, 3)
        ]
        for future in futures:
            future.result(timeout=2)

        assert order == [1, 2, 3]
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_model_scheduler_exposes_queue_and_batch_telemetry():
    scheduler = ModelWorkScheduler(name="test-model-scheduler", idle_grace_s=0.0)
    try:
        future = scheduler.submit_foreground(
            lambda: "ok",
            batch_key="chat.stream",
        )
        assert future.result(timeout=2) == "ok"
        scheduler.record_batch_step(size=3, batch_key="ar_batch.decode")
        stats = scheduler.stats()

        assert stats["started"] == 1
        assert stats["completed"] == 1
        assert stats["started_by_batch_key"]["chat.stream"] == 1
        assert stats["started_by_batch_key"]["ar_batch.decode"] == 1
        assert stats["batch_histogram"] == {"1": 1, "3": 1}
        assert stats["queue_wait_s"]["count"] == 1
        assert stats["run_duration_s"]["count"] == 1
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_model_work_runs_on_one_owner_thread():
    scheduler = ModelWorkScheduler(name="test-model-scheduler", idle_grace_s=0.0)
    try:
        foreground_thread = scheduler.submit_foreground(get_ident).result(timeout=2)
        idle_thread = scheduler.submit_idle_postcommit(get_ident).result(timeout=2)

        assert foreground_thread == idle_thread
        assert foreground_thread == scheduler.owner_thread_id
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_idle_postcommit_aborts_when_session_revision_is_stale(monkeypatch):
    scheduler = ModelWorkScheduler(name="test-model-scheduler", idle_grace_s=0.02)
    session = SimpleNamespace(revision=0)
    state = SimpleNamespace(
        session_cache_route=openai._GENERIC_SESSION_CACHE_ROUTE,
        model_scheduler=scheduler,
        generation_executor=scheduler,
        lock=None,
        args=SimpleNamespace(server_console=True),
    )
    calls: list[dict] = []

    def fake_store(*_args, **kwargs):
        calls.append(kwargs)
        return {"stored": True, "mode": "retokenized_history"}

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fake_store)

    try:
        pending = openai._schedule_idle_postcommit_snapshot(
            state,
            session_id="session-1",
            messages=[],
            assistant_content="ok",
            thinking_enabled=False,
            policy_fingerprint="policy",
            unsafe_reason="retokenized_history_mismatch",
            session=session,
            expected_session_revision=session.revision,
        )
        session.revision += 1
        scheduler.shutdown(wait=True)

        assert pending["mode"] == "async_pending"
        assert calls == []
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_batch_key_telemetry_collapses_per_session_suffixes():
    # postcommit:{session_id}-style keys must not grow the started-by counter
    # by one entry per session for the daemon's lifetime (external review F5):
    # the counter records the stable class before ':', while dotted static
    # keys ("chat.stream", "ar_batch.decode") pass through unchanged.
    scheduler = ModelWorkScheduler(name="test-model-scheduler", idle_grace_s=0.0)
    try:
        for index in range(64):
            scheduler.submit_foreground(
                lambda: "ok", batch_key=f"postcommit:session-{index}"
            ).result(timeout=2)
            scheduler.record_batch_step(
                size=1, batch_key=f"stream_tail:session-{index}"
            )
        stats = scheduler.stats()
        assert stats["started_by_batch_key"]["postcommit"] == 64
        assert stats["started_by_batch_key"]["stream_tail"] == 64
        session_keys = [
            key for key in stats["started_by_batch_key"] if "session-" in key
        ]
        assert session_keys == []
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_park_shutdown_keeps_owner_thread_alive_and_rejects_new_work():
    # #303: at process exit the owner thread must never pthread_exit — a
    # clean exit during interpreter finalization runs mlx's TLS destructor
    # into _Py_Dealloc. park=True leaves the thread blocked on a never-set
    # Event instead.
    scheduler = ModelWorkScheduler(name="test-park-scheduler", idle_grace_s=0.01)
    done = scheduler.submit_foreground(lambda: "ok")
    assert done.result(timeout=2) == "ok"

    scheduler.shutdown(wait=False, cancel_futures=True, park=True)
    time.sleep(0.2)
    assert scheduler._thread.is_alive()

    late = scheduler.submit_foreground(lambda: "never")
    assert late.cancelled() or isinstance(late.exception(timeout=2), Exception)


def test_release_mlx_thread_state_calls_clear_streams(monkeypatch):
    import sys

    from mtplx import model_scheduler

    calls: list[str] = []
    fake_core = SimpleNamespace(clear_streams=lambda: calls.append("cleared"))
    fake_mlx = SimpleNamespace(core=fake_core)
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)

    model_scheduler._release_mlx_thread_state()
    assert calls == ["cleared"]


def test_release_mlx_thread_state_swallows_missing_and_raising(monkeypatch):
    import sys

    from mtplx import model_scheduler

    # Older mlx without clear_streams: getattr-guarded no-op.
    bare_core = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=bare_core))
    monkeypatch.setitem(sys.modules, "mlx.core", bare_core)
    model_scheduler._release_mlx_thread_state()

    # clear_streams that raises must never propagate into teardown.
    def boom() -> None:
        raise RuntimeError("stream teardown")

    raising_core = SimpleNamespace(clear_streams=boom)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=raising_core))
    monkeypatch.setitem(sys.modules, "mlx.core", raising_core)
    model_scheduler._release_mlx_thread_state()
