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


# --- idle keepalive (GPU residency) -----------------------------------------


class _KeepaliveClock:
    """Advance scheduler time without imposing sub-second host deadlines."""

    def __init__(self, monkeypatch):
        from mtplx import model_scheduler

        self.now = 100.0
        # Replace only this module's clock, not time used by Condition waits.
        monkeypatch.setattr(
            model_scheduler, "time", SimpleNamespace(monotonic=lambda: self.now)
        )

    def settle(self, scheduler):
        with scheduler._condition:
            assert scheduler._condition.wait_for(
                lambda: not scheduler._owner_work_claimed, timeout=5
            )

    def advance(self, scheduler, seconds, *, beats=None, errors=None):
        with scheduler._condition:
            self.now += seconds
            scheduler._condition.notify_all()
            if beats is not None or errors is not None:
                assert scheduler._condition.wait_for(
                    lambda: (
                        (beats is None or scheduler._keepalive_beats >= beats)
                        and (errors is None or scheduler._keepalive_errors >= errors)
                        and not scheduler._owner_work_claimed
                    ),
                    timeout=5,
                )


def test_idle_keepalive_beats_on_owner_thread_only_while_attentive(monkeypatch):
    clock = _KeepaliveClock(monkeypatch)
    scheduler = ModelWorkScheduler(name="test-model-scheduler", idle_grace_s=0.01)
    beat_threads: list[int] = []
    try:
        scheduler.arm_idle_keepalive(
            lambda: beat_threads.append(get_ident()),
            interval_s=0.05,
            attentive_s=0.4,
        )
        # Arming starts the window; each completed beat gets a fresh interval.
        for count in range(1, 4):
            clock.advance(scheduler, 0.06, beats=count)
        state = scheduler.keepalive_state()
        assert state["armed"] and state["attentive"]
        assert state["beats"] == 3, state
        assert set(beat_threads) == {scheduler.owner_thread_id}
        clock.advance(scheduler, 0.5)
        assert not scheduler.keepalive_state()["attentive"]
        clock.advance(scheduler, 0.2)
        assert scheduler.keepalive_state()["beats"] == 3
        # A foreground completion re-opens the window and beats resume.
        scheduler.submit_foreground(lambda: None).result(timeout=5)
        clock.settle(scheduler)
        clock.advance(scheduler, 0.06, beats=4)
        assert scheduler.keepalive_state()["attentive"]
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_idle_keepalive_never_runs_while_work_is_queued_and_yields_to_foreground(monkeypatch):
    clock = _KeepaliveClock(monkeypatch)
    scheduler = ModelWorkScheduler(name="test-model-scheduler", idle_grace_s=0.01)
    events: list[str] = []
    release, started = Event(), Event()

    def long_foreground() -> None:
        events.append("fg-start")
        started.set()
        assert release.wait(timeout=5)
        events.append("fg-end")

    try:
        scheduler.arm_idle_keepalive(
            lambda: events.append("beat"), interval_s=0.05, attentive_s=10.0
        )
        future = scheduler.submit_foreground(long_foreground)
        assert started.wait(timeout=5)
        clock.advance(scheduler, 0.15)
        assert events == ["fg-start"]
        release.set()
        future.result(timeout=5)
        clock.settle(scheduler)
        clock.advance(scheduler, 0.06, beats=1)
        assert events == ["fg-start", "fg-end", "beat"]
    finally:
        release.set()
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_idle_keepalive_disarms_after_repeated_failures_and_reports_them(monkeypatch):
    clock = _KeepaliveClock(monkeypatch)
    scheduler = ModelWorkScheduler(name="test-model-scheduler", idle_grace_s=0.01)

    def boom() -> None:
        raise RuntimeError("no metal device")

    try:
        scheduler.arm_idle_keepalive(boom, interval_s=0.05, attentive_s=10.0)
        for count in range(1, 4):
            clock.advance(scheduler, 0.06, errors=count)
        state = scheduler.keepalive_state()
        assert state["errors"] == 3
        assert state["armed"] is False
        assert "no metal device" in (state["last_error"] or "")
        assert scheduler.submit_foreground(lambda: 7).result(timeout=5) == 7
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_idle_keepalive_warm_flag_tracks_recent_owner_activity(monkeypatch):
    clock = _KeepaliveClock(monkeypatch)
    scheduler = ModelWorkScheduler(name="test-model-scheduler", idle_grace_s=0.01)
    try:
        assert scheduler.keepalive_state()["warm"] is False
        scheduler.arm_idle_keepalive(lambda: None, interval_s=0.05, attentive_s=10.0)
        clock.advance(scheduler, 0.06, beats=1)
        assert scheduler.keepalive_state()["warm"] is True
        # Once attention expires, no beat refreshes the residency signal.
        clock.advance(scheduler, 11.0)
        assert scheduler.keepalive_state()["warm"] is False
        scheduler.disarm_idle_keepalive()
        assert scheduler.keepalive_state()["armed"] is False
        assert "idle_keepalive" in scheduler.stats()
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)
