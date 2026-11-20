"""A request that needs the running end-of-turn job's work waits for it.

With thinking on, the Qwen 3.6 template drops the previous user turn's
thinking from the next prompt, so at every user-turn boundary the history
after the turn's first thinking block is read again. The end-of-turn idle job
re-reads exactly that history (its target is the next prompt minus the new
user message). In the founder's Pi task on the 35B (rows
barsx-1117-0500/01-cand, requests 14 and 17) the next turn arrived while the
job was 2.5 s into that work: the request waited the 0.6 s arrival window,
aborted the job, and redid the same tokens from an older checkpoint (first
text 4.85 s and 8.08 s). The job's remaining work is a subset of the
request's own, so a request whose prompt starts with the job's target now
waits for it while it keeps reaching chunk boundaries; any other request keeps
the bounded abort.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from mtplx.engine_session import EngineSession
from mtplx.server import openai

PAD = 151655


def _running(session: EngineSession, future: Future, target, *, pad=None):
    future.set_running_or_notify_cancel()
    record = session.set_pending_postcommit(
        future, reason="reasoning_history_scoping_mismatch", token_count=len(target)
    )
    record.mark_started()
    record.note_target(target, image_pad_token_id=pad)
    return record


def _finish_after(record, future: Future, *, ticks: int, every_s: float) -> threading.Thread:
    def work() -> None:
        for _ in range(ticks):
            time.sleep(every_s)
            record.note_progress()
        future.set_result({"stored": True})

    thread = threading.Thread(target=work)
    thread.start()
    return thread


@pytest.fixture(autouse=True)
def _short_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MTPLX_POSTCOMMIT_WAIT_TIMEOUT_S", raising=False)
    monkeypatch.setenv("MTPLX_POSTCOMMIT_ARRIVAL_WAIT_S", "0.1")
    monkeypatch.setenv("MTPLX_POSTCOMMIT_WAIT_STALL_S", "0.4")
    monkeypatch.delenv("MTPLX_POSTCOMMIT_ADOPTION", raising=False)


def test_a_request_extending_the_jobs_target_waits_for_the_job_to_finish() -> None:
    session = EngineSession("sess-adopt")
    future: Future = Future()
    target = list(range(1, 200))
    record = _running(session, future, target)
    worker = _finish_after(record, future, ticks=6, every_s=0.1)
    try:
        started = time.monotonic()
        outcome = session.resolve_pending_postcommit_for_request(
            prompt_ids=target + [7, 8, 9]
        )
        elapsed = time.monotonic() - started
    finally:
        worker.join(timeout=5)
    assert outcome["outcome"] == "completed"
    assert outcome["adopted"] is True
    assert elapsed >= 0.5, "waited past the 0.1 s arrival window for the job"
    assert not record.abort_event.is_set()
    assert session.pending_postcommit is None


def test_a_request_with_another_history_keeps_the_bounded_abort() -> None:
    session = EngineSession("sess-other")
    future: Future = Future()
    target = list(range(1, 200))
    record = _running(session, future, target)
    prompt = target[:50] + [999_999] + target[51:] + [7]
    started = time.monotonic()
    outcome = session.resolve_pending_postcommit_for_request(prompt_ids=prompt)
    assert time.monotonic() - started < 0.4
    assert outcome["outcome"] == "aborted_for_foreground"
    assert "adopted" not in outcome
    assert record.abort_event.is_set()


def test_a_shorter_prompt_or_no_prompt_or_no_target_keeps_the_bounded_abort() -> None:
    for prompt, target in (([1, 2, 3], [1, 2, 3, 4]), (None, [1, 2, 3]), ([1, 2, 3, 4], None)):
        session = EngineSession("sess-short")
        future: Future = Future()
        future.set_running_or_notify_cancel()
        record = session.set_pending_postcommit(future, reason="postcommit")
        record.mark_started()
        if target is not None:
            record.note_target(target)
        outcome = session.resolve_pending_postcommit_for_request(prompt_ids=prompt)
        assert outcome["outcome"] == "aborted_for_foreground", (prompt, target)
        assert record.abort_event.is_set()


def test_an_adopted_job_that_stops_making_progress_is_aborted() -> None:
    session = EngineSession("sess-stall")
    future: Future = Future()
    target = list(range(1, 50))
    record = _running(session, future, target)
    record.note_progress()  # one chunk boundary, then silence
    started = time.monotonic()
    outcome = session.resolve_pending_postcommit_for_request(prompt_ids=target + [1])
    elapsed = time.monotonic() - started
    assert outcome["outcome"] == "aborted_for_foreground"
    assert outcome["adopted"] is True
    assert outcome["adoption_ended"] == "job_stopped_progressing"
    assert elapsed < 2.0
    assert record.abort_event.is_set()
    assert record.adopted is False


def test_the_switch_restores_the_bounded_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MTPLX_POSTCOMMIT_ADOPTION", "0")
    session = EngineSession("sess-off")
    future: Future = Future()
    target = list(range(1, 50))
    record = _running(session, future, target)
    outcome = session.resolve_pending_postcommit_for_request(prompt_ids=target + [1])
    assert outcome["outcome"] == "aborted_for_foreground"
    assert record.abort_event.is_set()


def test_image_histories_compare_one_pad_per_image() -> None:
    """The job's target carries each image's full run of pad tokens; a request
    is compared before its images are expanded (one pad per image) or after."""

    session = EngineSession("sess-image")
    future: Future = Future()
    expanded = [1, 2, PAD, PAD, PAD, PAD, 3, 4, PAD, PAD, 5]
    record = _running(session, future, expanded, pad=PAD)
    assert record.builds_prefix_of([1, 2, PAD, 3, 4, PAD, 5, 6, 7])
    assert record.builds_prefix_of(expanded + [6])
    assert not record.builds_prefix_of([1, 2, PAD, 3, 9, PAD, 5, 6])
    assert not record.builds_prefix_of([1, 2, PAD, 3, 4])


def test_an_adopted_job_does_not_yield_to_the_request_waiting_for_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The idle job yields to queued foreground work after its grace; while
    the request it serves waits for it, that request is not a reason to stop."""

    monkeypatch.setenv("MTPLX_POSTCOMMIT_FOREGROUND_GRACE_S", "0")
    scheduler = openai.ModelWorkScheduler(name="postcommit-adopt-test", idle_grace_s=0.0)
    state = SimpleNamespace(
        lock=threading.Lock(),
        has_foreground=lambda: False,
        generation_executor=scheduler,
        postcommit_executor=None,
        model_scheduler=scheduler,
        args=SimpleNamespace(server_console=True),
    )
    session = EngineSession("sess-adopt-yield")
    started = threading.Event()
    release = threading.Event()
    outcomes: list[dict] = []

    def fake_store(*_args, **kwargs):
        record = kwargs["pending_record"]
        record.note_target(list(range(1, 40)))
        started.set()
        abort_check = kwargs["abort_check"]
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not release.is_set():
            if abort_check():
                outcome = {"stored": False, "mode": "aborted", "reason": kwargs["abort_reason"]()}
                outcomes.append(outcome)
                return outcome
            time.sleep(0.01)
        outcome = {"stored": True, "mode": "retokenized_history"}
        outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fake_store)
    monkeypatch.setattr(openai, "_server_console_enabled", lambda _state: True)
    try:
        openai._schedule_idle_postcommit_snapshot(
            state,
            session_id="sess-adopt-yield",
            messages=[],
            assistant_content="done",
            assistant_tool_calls=None,
            thinking_enabled=True,
            policy_fingerprint="test-policy",
            unsafe_reason="reasoning_history_scoping_mismatch",
            session=session,
            expected_session_revision=session.revision,
        )
        assert started.wait(timeout=2.0)
        record = session._pending_postcommit
        record.adopted = True
        # Foreground work queued behind the job: an unadopted job yields at once.
        queued = scheduler.submit_foreground(lambda: None)
        time.sleep(0.2)
        assert not outcomes, "the adopted job kept working under foreground pressure"
        release.set()
        queued.result(timeout=3.0)
        assert outcomes[-1]["stored"] is True
    finally:
        release.set()
        scheduler.shutdown(wait=True, cancel_futures=True)
