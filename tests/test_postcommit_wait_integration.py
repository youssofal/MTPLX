"""Integration tests: scheduler-side wiring of postcommit-wait.

These tests exercise the seam between `_schedule_idle_postcommit_snapshot`
and `EngineSession.set_pending_postcommit`. Together they prove that:

  1. Scheduling an idle postcommit stashes the work future on the session.
  2. The next request's `wait_for_pending_postcommit()` observes the same
     future and bails out cleanly when the postcommit returns.
  3. A foreground submission (chat handler path) does not need to wait on
     idle work that the scheduler has not started yet, because the
     foreground lane has admission priority - we only assert wiring here.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from mtplx.engine_session import EngineSession, EngineSessionManager
from mtplx.server import openai


def _fake_state_with_executor():
    """Stand up a state object good enough for `_schedule_idle_postcommit_snapshot`.

    `_submit_idle_postcommit_model_work` falls back to `state.generation_executor`
    when no model_scheduler is present. Tests run cheaply against this seam
    rather than spinning up the real ModelWorkScheduler.
    """
    return SimpleNamespace(
        session_cache_route=openai._GENERIC_SESSION_CACHE_ROUTE,
        lock=threading.Lock(),
        has_foreground=lambda: False,
        generation_executor=ThreadPoolExecutor(max_workers=1),
        postcommit_executor=None,
        model_scheduler=None,
        args=SimpleNamespace(server_console=False),
    )


def _kwargs(session_id: str = "sess-int", session: EngineSession | None = None):
    return dict(
        session_id=session_id,
        messages=[],
        assistant_content="hello",
        assistant_tool_calls=None,
        thinking_enabled=False,
        policy_fingerprint="test-policy",
        unsafe_reason="retokenized_history_mismatch",
        session=session,
        expected_session_revision=getattr(session, "revision", 0) if session else None,
    )


def test_scheduling_stashes_future_on_session(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _fake_state_with_executor()
    barrier = threading.Event()

    def fake_store(*_args, **_kwargs):
        # Block until the test releases us so the future is observably
        # pending on the session for at least one wait cycle.
        barrier.wait(timeout=2.0)
        return {
            "stored": True,
            "mode": "retokenized_history",
            "prefix_len": 1,
            "nbytes": 1,
        }

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fake_store)
    monkeypatch.setattr(openai, "_server_console_enabled", lambda _state: True)

    session = EngineSession("sess-int")
    assert session.pending_postcommit is None

    openai._schedule_idle_postcommit_snapshot(state, **_kwargs(session=session))

    # The schedule call returns immediately; the future is on the session.
    assert session.pending_postcommit is not None

    # Release the worker; the wait observes a clean completion.
    barrier.set()
    outcome = session.wait_for_pending_postcommit(timeout_s=2.0)
    assert outcome["outcome"] == "completed"
    assert session.pending_postcommit is None

    state.generation_executor.shutdown(wait=True)


def test_tool_call_prompt_prefix_anchors_next_waiter() -> None:
    """Streaming tool calls publish the prompt prefix before async postcommit.

    The structured assistant/tool history is retokenized in the idle
    postcommit lane, but the immediate tool-result request must still resolve
    to the same session so it can wait for that pending work.
    """
    session = EngineSession("sess-tool")

    commit = session.commit_prompt_prefix(
        prompt_ids=[1, 2, 3, 4],
        finish_reason="tool_calls",
        boundary_kind="tool_call_prompt_prefix",
    )

    assert commit.committed is True
    assert commit.reason == "committed_prompt_prefix"
    assert session.committed_token_ids == (1, 2, 3, 4)
    assert session.last_finish_reason == "tool_calls"
    assert session.revision == 1
    assert session.boundaries[-1].kind == "tool_call_prompt_prefix"


def test_tool_call_prompt_prefix_does_not_rewind_session() -> None:
    session = EngineSession("sess-tool")
    session.commit_prompt_prefix(prompt_ids=[1, 2, 3, 4], finish_reason="tool_calls")

    commit = session.commit_prompt_prefix(
        prompt_ids=[1, 2],
        finish_reason="tool_calls",
    )

    assert commit.committed is False
    assert commit.reason == "prompt_prefix_older_than_session"
    assert session.committed_token_ids == (1, 2, 3, 4)
    assert session.revision == 1


def test_pending_near_prefix_resolves_tool_result_waiter() -> None:
    manager = EngineSessionManager()
    session = manager.get_or_create("sess-tool")
    session.commit_prompt_prefix(
        prompt_ids=[1, 2, 3, 4],
        finish_reason="tool_calls",
        boundary_kind="tool_call_prompt_prefix",
    )
    future: Future = Future()
    session.set_pending_postcommit(future)

    session_id, source = manager.resolve_session_id(prompt_ids=[1, 2, 3, 99, 100])

    assert session_id == "sess-tool"
    assert source == "pending_postcommit_near_prefix"
    assert manager.last_prefix_diagnostic is not None
    assert manager.last_prefix_diagnostic["reason"] == "pending_postcommit_near_prefix_match"
    assert manager.last_prefix_diagnostic["near_prefix_gap"] == 1

    future.set_result(None)
    session.wait_for_pending_postcommit(timeout_s=1.0)


def test_pending_near_prefix_tolerates_small_tool_boundary_gap() -> None:
    manager = EngineSessionManager()
    session = manager.get_or_create("sess-tool")
    prompt = list(range(200))
    session.commit_prompt_prefix(
        prompt_ids=prompt,
        finish_reason="tool_calls",
        boundary_kind="tool_call_prompt_prefix",
    )
    future: Future = Future()
    session.set_pending_postcommit(future)
    tool_result_prompt = prompt[:-3] + [10_001, 10_002, 10_003, 10_004]

    session_id, source = manager.resolve_session_id(prompt_ids=tool_result_prompt)

    assert session_id == "sess-tool"
    assert source == "pending_postcommit_near_prefix"
    assert manager.last_prefix_diagnostic is not None
    assert manager.last_prefix_diagnostic["near_prefix_gap"] == 3

    future.set_result(None)
    session.wait_for_pending_postcommit(timeout_s=1.0)


def test_retokenized_prefix_replaces_prompt_anchor_boundary() -> None:
    session = EngineSession("sess-tool")
    session.commit_prompt_prefix(prompt_ids=[1, 2, 3, 4], finish_reason="tool_calls")

    commit = session.commit_retokenized_prefix(
        token_ids=[1, 2, 3, 99, 100],
        expected_revision=1,
        nbytes=123,
    )

    assert commit.committed is True
    assert commit.reason == "committed_retokenized_prefix"
    assert session.committed_token_ids == (1, 2, 3, 99, 100)
    assert session.prefix_len == 5
    assert session.bytes_estimate == 123
    assert session.revision == 2


def test_retokenized_prefix_tolerates_small_tool_boundary_gap() -> None:
    session = EngineSession("sess-tool")
    prompt = list(range(200))
    session.commit_prompt_prefix(prompt_ids=prompt, finish_reason="tool_calls")

    retokenized = prompt[:-3] + [10_001, 10_002, 10_003, 10_004]
    commit = session.commit_retokenized_prefix(
        token_ids=retokenized,
        expected_revision=1,
        nbytes=123,
    )

    assert commit.committed is True
    assert commit.reason == "committed_retokenized_prefix"
    assert session.committed_token_ids == tuple(retokenized)
    assert session.revision == 2


def test_retokenized_prefix_does_not_rewind_prompt_anchor() -> None:
    session = EngineSession("sess-tool")
    session.commit_prompt_prefix(prompt_ids=[1, 2, 3, 4], finish_reason="tool_calls")

    commit = session.commit_retokenized_prefix(
        token_ids=[1, 2, 3],
        expected_revision=1,
    )

    assert commit.committed is False
    assert commit.reason == "retokenized_prefix_older_than_session"
    assert session.committed_token_ids == (1, 2, 3, 4)
    assert session.revision == 1


def test_wait_times_out_when_postcommit_hangs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operator safety net: if the postcommit job hangs (e.g. the scheduler
    has a foreground backlog draining), the next request bails out and
    proceeds cold instead of stalling.
    """
    state = _fake_state_with_executor()
    block_forever = threading.Event()

    def fake_store(*_args, **_kwargs):
        # Simulate a postcommit job that is stuck (e.g. waiting on lock).
        block_forever.wait(timeout=5.0)
        return {"stored": False, "reason": "stuck"}

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fake_store)
    monkeypatch.setattr(openai, "_server_console_enabled", lambda _state: True)

    session = EngineSession("sess-int-timeout")
    openai._schedule_idle_postcommit_snapshot(state, **_kwargs(session=session))
    assert session.pending_postcommit is not None

    started = time.monotonic()
    outcome = session.wait_for_pending_postcommit(timeout_s=0.1)
    elapsed = time.monotonic() - started

    assert outcome["outcome"] == "timeout"
    assert outcome["waited"] is True
    assert outcome["abort_reason"] == "foreground_preempted_postcommit"
    assert elapsed < 0.5, "wait must be bounded by the timeout"
    # Future stays referenced by the executor but is removed from the
    # session - the request can now proceed cold without re-blocking.
    assert session.pending_postcommit is None

    block_forever.set()
    state.generation_executor.shutdown(wait=True)


def test_running_idle_postcommit_yields_to_queued_foreground(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Strict immediate-yield semantics (2026-07-02 starvation guard) are
    # preserved behind grace=0; the default bounded grace is covered by
    # test_running_idle_postcommit_finishes_within_foreground_grace below.
    monkeypatch.setenv("MTPLX_POSTCOMMIT_FOREGROUND_GRACE_S", "0")
    scheduler = openai.ModelWorkScheduler(name="postcommit-yield-test", idle_grace_s=0.0)
    state = SimpleNamespace(
        session_cache_route=openai._GENERIC_SESSION_CACHE_ROUTE,
        lock=threading.Lock(),
        has_foreground=lambda: False,
        generation_executor=scheduler,
        postcommit_executor=None,
        model_scheduler=scheduler,
        args=SimpleNamespace(server_console=True),
    )
    session = EngineSession("sess-yield")
    started = threading.Event()
    outcomes: list[dict] = []

    def fake_store(*_args, **kwargs):
        started.set()
        abort_check = kwargs["abort_check"]
        abort_reason = kwargs["abort_reason"]
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if abort_check():
                outcome = {
                    "stored": False,
                    "mode": "aborted",
                    "reason": abort_reason(),
                }
                outcomes.append(outcome)
                return outcome
            time.sleep(0.01)
        outcome = {"stored": True, "mode": "retokenized_history"}
        outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fake_store)
    monkeypatch.setattr(openai, "_server_console_enabled", lambda _state: True)

    try:
        openai._schedule_idle_postcommit_snapshot(state, **_kwargs(session=session))
        assert started.wait(timeout=2.0)

        foreground_ran = threading.Event()
        foreground = scheduler.submit_foreground(lambda: foreground_ran.set())
        foreground.result(timeout=2.0)

        assert foreground_ran.is_set()
        assert outcomes
        assert outcomes[-1]["reason"] == "foreground_preempted_postcommit"
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_running_idle_postcommit_aborts_when_revision_advances_mid_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = openai.ModelWorkScheduler(name="postcommit-stale-test", idle_grace_s=0.0)
    state = SimpleNamespace(
        session_cache_route=openai._GENERIC_SESSION_CACHE_ROUTE,
        lock=threading.Lock(),
        has_foreground=lambda: False,
        generation_executor=scheduler,
        postcommit_executor=None,
        model_scheduler=scheduler,
        args=SimpleNamespace(server_console=True),
    )
    session = EngineSession("sess-stale-mid")
    started = threading.Event()
    outcomes: list[dict] = []

    def fake_store(*_args, **kwargs):
        started.set()
        abort_check = kwargs["abort_check"]
        abort_reason = kwargs["abort_reason"]
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if abort_check():
                outcome = {
                    "stored": False,
                    "mode": "aborted",
                    "reason": abort_reason(),
                }
                outcomes.append(outcome)
                return outcome
            time.sleep(0.01)
        outcome = {"stored": True, "mode": "retokenized_history"}
        outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fake_store)
    monkeypatch.setattr(openai, "_server_console_enabled", lambda _state: True)

    try:
        future_info = openai._schedule_idle_postcommit_snapshot(
            state,
            **_kwargs(session=session),
        )
        assert future_info["mode"] == "async_pending"
        assert started.wait(timeout=2.0)
        session.revision += 1
        session.pending_postcommit.result(timeout=2.0)

        assert outcomes
        assert outcomes[-1]["reason"] == "stale_session_revision"
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)


def test_wait_metrics_attach_to_request_observability_shape() -> None:
    """The chat handler attaches the wait outcome to request_observability
    which then merges into the metrics envelope. We assert the shape here
    rather than spin up a full FastAPI test client - the shape is what
    /metrics consumers depend on.
    """
    session = EngineSession("sess-int-metrics")
    # Simulate "no_pending" path the chat handler hits on a brand-new session.
    outcome = session.wait_for_pending_postcommit(timeout_s=1.0)

    # request_observability is just a plain dict in the handler; we mimic that.
    request_observability: dict = {}
    request_observability["postcommit_wait"] = outcome

    assert "postcommit_wait" in request_observability
    pw = request_observability["postcommit_wait"]
    assert set(pw.keys()) == {"waited", "elapsed_s", "outcome", "timeout_s"}
    assert pw["outcome"] == "no_pending"


def test_postcommit_skips_estimated_oversized_snapshot_before_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large OpenCode histories must not spend foreground-visible time
    materializing a SessionBank snapshot that the byte cap will reject.
    """

    class FakeBank:
        per_session_max_bytes = 8 * 1024 * 1024 * 1024

        def longest_prefix(self, _history_ids):
            return SimpleNamespace(prefix_len=117_793, nbytes=8_227_240_960)

    state = SimpleNamespace(
        session_cache_route=openai._GENERIC_SESSION_CACHE_ROUTE,
        sessions=SimpleNamespace(bank=FakeBank()),
    )
    monkeypatch.setattr(
        openai,
        "_history_ids_for_postcommit",
        lambda *_args, **_kwargs: (list(range(121_704)), None),
    )
    restore_called = False

    def fake_restore(*_args, **_kwargs):
        nonlocal restore_called
        restore_called = True
        raise AssertionError("oversized postcommit should skip before prefill")

    monkeypatch.setattr(openai, "restore_or_prefill_prompt_state", fake_restore)

    outcome = openai._store_retokenized_history_snapshot(
        state,
        session_id="sess-int",
        messages=[],
        assistant_content="hello",
        thinking_enabled=False,
        policy_fingerprint="test-policy",
    )

    assert outcome["stored"] is False
    assert outcome["reason"] == "estimated_oversized_snapshot"
    assert outcome["estimated_nbytes"] > outcome["budget"]
    assert outcome["best_prefix_len"] == 117_793
    assert outcome["history_tokens"] == 121_704
    assert restore_called is False


def test_common_prefix_reuse_survives_mid_history_mutation() -> None:
    """A mutated mid-prefix (compaction flip / edit rewrite) must not fork a
    new anon session id — the 2026-07-20 chess session rotated through six
    ids and quadrupled the RAM bank (2026-07-20 live-session forensics)."""
    manager = EngineSessionManager()
    session = manager.get_or_create("sess-mutated")
    prompt = list(range(9000))
    session.commit_prompt_prefix(
        prompt_ids=prompt,
        finish_reason="tool_calls",
        boundary_kind="tool_call_prompt_prefix",
    )
    # History mutates 6000 tokens in (an old tool result re-rendered), the
    # tail is rewritten, and the conversation continues longer than before.
    mutated = prompt[:6000] + [50_000 + i for i in range(4000)]

    session_id, source = manager.resolve_session_id(prompt_ids=mutated)

    assert session_id == "sess-mutated"
    assert source == "common_prefix_reuse"
    assert manager.last_prefix_diagnostic is not None
    assert manager.last_prefix_diagnostic["reason"] == "common_prefix_reuse"
    assert manager.last_prefix_diagnostic["matched_prefix_len"] == 6000


def test_common_prefix_reuse_rejects_unrelated_conversations() -> None:
    manager = EngineSessionManager()
    session = manager.get_or_create("sess-a")
    session.commit_prompt_prefix(
        prompt_ids=list(range(5000)),
        finish_reason="stop",
        boundary_kind="retokenized_history",
    )
    unrelated = [90_000 + i for i in range(5000)]

    session_id, source = manager.resolve_session_id(prompt_ids=unrelated)

    assert source == "new"
    assert session_id != "sess-a"


def test_common_prefix_reuse_requires_threshold() -> None:
    manager = EngineSessionManager()
    session = manager.get_or_create("sess-short")
    session.commit_prompt_prefix(
        prompt_ids=list(range(2000)),
        finish_reason="stop",
        boundary_kind="retokenized_history",
    )
    # Shares only 1000 tokens — below both the absolute and fractional bars.
    candidate = list(range(1000)) + [70_000 + i for i in range(9000)]

    session_id, source = manager.resolve_session_id(prompt_ids=candidate)

    assert source == "new"
    assert session_id != "sess-short"


def test_running_idle_postcommit_finishes_within_foreground_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 2026-08-01: in real agent loops the next request arrives within ~0.5s,
    # which preempted every tool-turn commit and forced a 2-4k-token salvage
    # re-prefill per turn. A short commit now finishes inside the bounded
    # grace even with foreground queued; the foreground still runs right
    # after (delay bounded by the grace, not unbounded starvation).
    monkeypatch.setenv("MTPLX_POSTCOMMIT_FOREGROUND_GRACE_S", "5")
    scheduler = openai.ModelWorkScheduler(name="postcommit-grace-test", idle_grace_s=0.0)
    state = SimpleNamespace(
        session_cache_route=openai._GENERIC_SESSION_CACHE_ROUTE,
        lock=threading.Lock(),
        has_foreground=lambda: False,
        generation_executor=scheduler,
        postcommit_executor=None,
        model_scheduler=scheduler,
        args=SimpleNamespace(server_console=True),
    )
    session = EngineSession("sess-grace")
    started = threading.Event()
    outcomes: list[dict] = []

    def fake_store(*_args, **kwargs):
        started.set()
        abort_check = kwargs["abort_check"]
        deadline = time.monotonic() + 0.4
        while time.monotonic() < deadline:
            if abort_check():
                outcome = {
                    "stored": False,
                    "mode": "aborted",
                    "reason": kwargs["abort_reason"](),
                }
                outcomes.append(outcome)
                return outcome
            time.sleep(0.01)
        outcome = {"stored": True, "mode": "retokenized_history"}
        outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fake_store)
    monkeypatch.setattr(openai, "_server_console_enabled", lambda _state: True)

    try:
        openai._schedule_idle_postcommit_snapshot(state, **_kwargs(session=session))
        assert started.wait(timeout=2.0)

        foreground_ran = threading.Event()
        foreground = scheduler.submit_foreground(lambda: foreground_ran.set())
        foreground.result(timeout=5.0)

        assert foreground_ran.is_set()
        assert outcomes
        assert outcomes[-1] == {"stored": True, "mode": "retokenized_history"}
    finally:
        scheduler.shutdown(wait=True, cancel_futures=True)
