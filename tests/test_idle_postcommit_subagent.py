"""Regression tests for `_schedule_idle_postcommit_snapshot`.

The async path used to gate on `state.has_foreground()` clearing before it
ran the retokenized commit. In OpenAI-compatible subagent fan-out (opencode
researcher / planner / fixer / bug-patcher) the parent re-dispatches the
next request within milliseconds of the previous response completing, so
the foreground flag never clears - every subagent's snapshot was abandoned
with `foreground_busy_past_deadline` and the SessionBank never accumulated
an entry for those session ids.

These tests cover the post-fix behaviour: the async path now retries the
existing non-blocking lock acquire inside `_store_retokenized_history_snapshot`
(which is the real correctness gate) until either the snapshot stores or
the deadline expires.
"""

from concurrent.futures import ThreadPoolExecutor
import threading
import time
from types import SimpleNamespace

import pytest

from mtplx.server import openai


def _fake_subagent_state(*, foreground_always: bool = False):
    """Build the minimum stub needed for `_schedule_idle_postcommit_snapshot`.

    `foreground_always=True` simulates the opencode subagent regression:
    `state.has_foreground()` returns True for the entire test, which under
    the old code blocked the retokenized commit indefinitely.
    """
    foreground_lock = threading.Lock()

    def has_foreground() -> bool:
        return True if foreground_always else False

    return SimpleNamespace(
        lock=foreground_lock,
        has_foreground=has_foreground,
        generation_executor=ThreadPoolExecutor(max_workers=1),
        # Console disabled so `_log` exercises the print branch (and stays
        # silent under pytest's captured stdout).
        args=SimpleNamespace(server_console=False),
    )


def _wait_for_executor_drain(state, *, timeout_s: float = 5.0) -> None:
    """Shut the executor down (waiting for the submitted task to finish).

    Using `executor.shutdown(wait=True)` gives us a deterministic point
    after which `_log` has already been called for the submitted job.
    """
    state.generation_executor.shutdown(wait=True)
    _ = timeout_s  # symmetric with similar helpers; kept for readability.


def _kwargs_for_schedule():
    return dict(
        session_id="opencode-researcher",
        messages=[],
        assistant_content="<tool_call>...</tool_call>",
        assistant_tool_calls=[{"name": "grep", "arguments": "{}"}],
        thinking_enabled=False,
        policy_fingerprint="test-policy",
        unsafe_reason="retokenized_history_mismatch",
    )


def test_idle_postcommit_stores_when_lock_briefly_free(monkeypatch):
    """Happy path: even if `has_foreground()` says True, a successful store
    on the first non-blocking lock acquire ends the loop and is logged.
    """
    state = _fake_subagent_state(foreground_always=True)
    calls: list[dict] = []

    def fake_store(*_args, **kwargs):
        calls.append(kwargs)
        return {
            "stored": True,
            "mode": "retokenized_history",
            "prefix_len": 17,
            "nbytes": 4242,
        }

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fake_store)
    monkeypatch.setattr(openai, "_server_console_enabled", lambda _state: True)

    pending = openai._schedule_idle_postcommit_snapshot(
        state, **_kwargs_for_schedule()
    )

    _wait_for_executor_drain(state)

    assert pending == {
        "stored": False,
        "mode": "async_pending",
        "reason": "retokenized_history_mismatch",
    }
    assert len(calls) == 1
    assert calls[0]["session_id"] == "opencode-researcher"
    assert calls[0]["acquire_model_lock_blocking"] is False


def test_idle_postcommit_retries_until_deadline_under_lock_busy(monkeypatch):
    """If the lock stays busy the entire deadline, the loop logs
    `model_lock_busy_past_deadline` and returns - it does NOT spin
    forever.
    """
    state = _fake_subagent_state(foreground_always=True)
    log_lines: list[dict] = []
    call_count = {"n": 0}

    def fake_store(*_args, **_kwargs):
        call_count["n"] += 1
        return {
            "stored": False,
            "mode": "retokenized_history",
            "reason": "model_lock_busy_before_retokenized_commit",
        }

    def capture_console_disabled(_state):
        # Keep `_log` going through the print branch so we know the
        # function reached the deadline-abandon log line. We also
        # monkeypatch the print in the real module so we can inspect.
        return False

    real_print = openai.print if hasattr(openai, "print") else print

    def captured_print(message, *_args, **_kwargs):
        # Lines are formatted as
        # `[mtplx] idle async session postcommit {json...}`.
        prefix = "[mtplx] idle async session postcommit "
        if isinstance(message, str) and message.startswith(prefix):
            import json

            log_lines.append(json.loads(message[len(prefix):]))
        else:
            real_print(message, *_args, **_kwargs)

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fake_store)
    monkeypatch.setattr(openai, "_server_console_enabled", capture_console_disabled)
    monkeypatch.setattr("builtins.print", captured_print)
    monkeypatch.setattr(openai, "_IDLE_POSTCOMMIT_MAX_WAIT_S", 0.5)
    monkeypatch.setattr(openai, "_IDLE_POSTCOMMIT_POLL_INTERVAL_S", 0.05)

    started = time.monotonic()
    pending = openai._schedule_idle_postcommit_snapshot(
        state, **_kwargs_for_schedule()
    )
    _wait_for_executor_drain(state)
    elapsed = time.monotonic() - started

    assert pending["mode"] == "async_pending"
    # Loop should retry many times under the 0.5s deadline / 0.05s poll
    # interval (we expect ~10 calls; allow generous slack for CI jitter).
    assert call_count["n"] >= 3, f"expected several retries, got {call_count['n']}"
    assert elapsed < 5.0, f"loop ran too long: {elapsed:.2f}s"
    # We must see exactly one terminal log line declaring the deadline
    # abandon, and crucially the reason must be the new one.
    assert log_lines, "expected at least one log line"
    last = log_lines[-1]
    assert last["mode"] == "abandoned_foreground_busy"
    assert last["reason"] == "model_lock_busy_past_deadline"
    assert last["session_id"] == "opencode-researcher"


def test_idle_postcommit_returns_immediately_on_non_recoverable_failure(monkeypatch):
    """`no_session_id` (and similar non-retryable errors) must short-circuit
    the loop after exactly one call - retrying will not help.
    """
    state = _fake_subagent_state(foreground_always=True)
    call_count = {"n": 0}

    def fake_store(*_args, **_kwargs):
        call_count["n"] += 1
        return {"stored": False, "reason": "no_session_id"}

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fake_store)
    monkeypatch.setattr(openai, "_server_console_enabled", lambda _state: True)
    # Use a long deadline so a buggy implementation that DID retry would
    # show up as call_count["n"] > 1.
    monkeypatch.setattr(openai, "_IDLE_POSTCOMMIT_MAX_WAIT_S", 5.0)
    monkeypatch.setattr(openai, "_IDLE_POSTCOMMIT_POLL_INTERVAL_S", 0.01)

    openai._schedule_idle_postcommit_snapshot(state, **_kwargs_for_schedule())
    _wait_for_executor_drain(state)

    assert call_count["n"] == 1


def test_idle_postcommit_no_longer_blocks_on_has_foreground(monkeypatch):
    """Regression test: even if `state.has_foreground()` returns True for
    the entire duration of the test, the snapshot must still be stored.

    Pre-fix this scenario was the bug - the loop would spin on
    `has_foreground()` until the 30s deadline expired without ever calling
    `_store_retokenized_history_snapshot`.
    """
    state = _fake_subagent_state(foreground_always=True)
    calls: list[dict] = []

    def fake_store(*_args, **kwargs):
        calls.append(kwargs)
        return {
            "stored": True,
            "mode": "retokenized_history",
            "prefix_len": 8,
            "nbytes": 64,
        }

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fake_store)
    monkeypatch.setattr(openai, "_server_console_enabled", lambda _state: True)
    # Force tight bounds so a regression to the old wait-on-foreground
    # code would time out and the snapshot would NOT be stored.
    monkeypatch.setattr(openai, "_IDLE_POSTCOMMIT_MAX_WAIT_S", 0.5)
    monkeypatch.setattr(openai, "_IDLE_POSTCOMMIT_POLL_INTERVAL_S", 0.05)

    started = time.monotonic()
    openai._schedule_idle_postcommit_snapshot(state, **_kwargs_for_schedule())
    _wait_for_executor_drain(state)
    elapsed = time.monotonic() - started

    assert calls, "snapshot must be attempted even when has_foreground() is True"
    assert len(calls) == 1
    assert elapsed < 1.0, (
        f"snapshot ran too long ({elapsed:.2f}s) - did the loop fall back to "
        "the deprecated has_foreground() wait?"
    )
