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
