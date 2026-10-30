"""Cross-session postcommit yield (2026-08-05 showdown fix).

The idle postcommit's foreground grace is a same-session bargain: waiting
<=grace pays off because the commit makes THIS session's next request fast.
A request from a DIFFERENT session gains nothing from a stranger's commit —
it just pays the commit's remaining runtime in TTFT and its bandwidth
residue in decode. These tests prove the admission-time sweep aborts every
other session's pending commit, spares the admitting session's own, and
respects the env kill switch.
"""

from __future__ import annotations

import time
from concurrent.futures import Future

import pytest

from mtplx.engine_session import EngineSessionManager
from mtplx.server import openai


def _manager() -> EngineSessionManager:
    return EngineSessionManager(bank=None, idle_ttl_s=60.0)


def _pending(manager: EngineSessionManager, session_id: str):
    session = manager.get_or_create(session_id)
    future: Future = Future()  # never resolved = commit in flight
    record = session.set_pending_postcommit(future, reason="test-commit")
    return session, record


def test_cross_session_pending_postcommit_aborted_on_admission() -> None:
    manager = _manager()
    other_session, other_record = _pending(manager, "sess-a")
    manager.get_or_create("sess-b")

    outcome = manager.abort_cross_session_postcommits(except_session_id="sess-b")

    assert outcome is not None
    assert outcome["count"] == 1
    assert outcome["sessions"] == ["sess-a"]
    assert outcome["reason"] == "cross_session_foreground_preempted"
    assert other_record.abort_event.is_set()
    assert other_record.last_abort_reason == "cross_session_foreground_preempted"


def test_same_session_pending_postcommit_survives_sweep() -> None:
    manager = _manager()
    own_session, own_record = _pending(manager, "sess-b")

    outcome = manager.abort_cross_session_postcommits(except_session_id="sess-b")

    assert outcome is None
    assert not own_record.abort_event.is_set()
    assert own_session.has_pending_postcommit()


def test_sweep_with_no_pending_commits_returns_none() -> None:
    manager = _manager()
    manager.get_or_create("sess-a")
    manager.get_or_create("sess-b")

    assert manager.abort_cross_session_postcommits(except_session_id="sess-b") is None


def test_stateless_admission_aborts_all_sessions() -> None:
    manager = _manager()
    _, record_a = _pending(manager, "sess-a")
    _, record_b = _pending(manager, "sess-b")

    outcome = manager.abort_cross_session_postcommits(except_session_id=None)

    assert outcome is not None
    assert outcome["count"] == 2
    assert record_a.abort_event.is_set()
    assert record_b.abort_event.is_set()


def test_cross_session_yield_env_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MTPLX_POSTCOMMIT_CROSS_SESSION_YIELD", raising=False)
    assert openai._postcommit_cross_session_yield_enabled() is True

    for off in ("0", "false", "off", "no"):
        monkeypatch.setenv("MTPLX_POSTCOMMIT_CROSS_SESSION_YIELD", off)
        assert openai._postcommit_cross_session_yield_enabled() is False

    monkeypatch.setenv("MTPLX_POSTCOMMIT_CROSS_SESSION_YIELD", "1")
    assert openai._postcommit_cross_session_yield_enabled() is True


# ---------------------------------------------------------------------------
# /v1/completions request path (2.5.3 pre-merge correction): the sessionless
# endpoint sweeps ALL pending commits, and a sweep failure SURFACES exactly
# like the chat path — no silent swallow.

def _completions_app_state(monkeypatch, sweep):
    from types import SimpleNamespace  # noqa: PLC0415

    from test_server_openai import _fake_generation, _fake_state  # noqa: PLC0415

    state = _fake_state()
    monkeypatch.setattr(
        state.sessions,
        "abort_cross_session_postcommits",
        sweep,
        raising=False,
    )
    monkeypatch.setattr(
        openai, "_run_generation", lambda *a, **k: _fake_generation("ok")
    )
    return state


def test_completions_admission_sweeps_all_sessions(monkeypatch):
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from mtplx.server.openai import create_app  # noqa: PLC0415

    calls: list[object] = []

    def sweep(*, except_session_id, reason="cross_session_foreground_preempted"):
        calls.append(except_session_id)
        return {"count": 0, "sessions": [], "reason": reason}

    state = _completions_app_state(monkeypatch, sweep)
    client = TestClient(create_app(state))
    r = client.post("/v1/completions", json={"prompt": [1, 2, 3], "max_tokens": 4})
    assert r.status_code == 200
    assert calls == [None]  # sessionless: every pending commit is foreign

    # env kill switch: no sweep call
    calls.clear()
    monkeypatch.setenv("MTPLX_POSTCOMMIT_CROSS_SESSION_YIELD", "0")
    r = client.post("/v1/completions", json={"prompt": [1, 2, 3], "max_tokens": 4})
    assert r.status_code == 200
    assert calls == []


def test_completions_sweep_failure_surfaces_not_swallowed(monkeypatch):
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from mtplx.server.openai import create_app  # noqa: PLC0415

    def sweep(**_kw):
        raise RuntimeError("sweep exploded")

    state = _completions_app_state(monkeypatch, sweep)
    client = TestClient(create_app(state), raise_server_exceptions=False)
    r = client.post("/v1/completions", json={"prompt": [1, 2, 3], "max_tokens": 4})
    # Surfaces through the sanitized 500 handler — the request does NOT
    # proceed as if the sweep succeeded (parity with the chat path).
    assert r.status_code == 500


# ---------------------------------------------------------------------------
# Marathon protection on the cross-session path (#432, reporter
# nomishbhardwaj). The 2026-08-05 bargain prices the arriving request's TTFT
# but not the checkpoint it destroys: a 130-token vision request from another
# session killed a 200k-token commit at admission, and the deep session then
# paid a 316s full re-prefill. MTPLX_POSTCOMMIT_MARATHON_PROTECT_TOKENS now
# guards this path too, under a bounded, non-refreshing grace.

MARATHON_ENV = "MTPLX_POSTCOMMIT_MARATHON_PROTECT_TOKENS"
MARATHON_GRACE_ENV = "MTPLX_POSTCOMMIT_MARATHON_WAIT_S"


def _pending_sized(manager: EngineSessionManager, session_id: str, token_count: int):
    session = manager.get_or_create(session_id)
    future: Future = Future()  # never resolved = commit in flight
    record = session.set_pending_postcommit(
        future,
        reason="retokenized_history_mismatch",
        token_count=token_count,
    )
    return session, record


def test_marathon_protection_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset threshold keeps the exact 2026-08-05 behavior: everything aborts."""
    monkeypatch.delenv(MARATHON_ENV, raising=False)
    manager = _manager()
    _, deep_record = _pending_sized(manager, "sess-deep", 200_000)
    manager.get_or_create("sess-vision")

    outcome = manager.abort_cross_session_postcommits(except_session_id="sess-vision")

    assert outcome is not None
    assert outcome["count"] == 1
    assert outcome["sessions"] == ["sess-deep"]
    assert "marathon_protected" not in outcome
    assert deep_record.abort_event.is_set()


def test_marathon_commit_survives_foreign_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MARATHON_ENV, "50000")
    monkeypatch.delenv(MARATHON_GRACE_ENV, raising=False)
    manager = _manager()
    deep_session, deep_record = _pending_sized(manager, "sess-deep", 200_000)
    manager.get_or_create("sess-vision")

    outcome = manager.abort_cross_session_postcommits(except_session_id="sess-vision")

    assert outcome is not None
    assert outcome["count"] == 0
    assert outcome["sessions"] == []
    assert outcome["marathon_protected_count"] == 1
    grant = outcome["marathon_protected"][0]
    assert grant["session_id"] == "sess-deep"
    assert grant["token_count"] == 200_000
    assert grant["protect_tokens"] == 50_000
    assert grant["grace_s"] == 30.0
    assert 0.0 < grant["grace_remaining_s"] <= 30.0
    # The commit lives: no abort signal, still the session's pending record.
    assert not deep_record.abort_event.is_set()
    assert deep_record.last_abort_reason is None
    assert deep_session.has_pending_postcommit()


def test_below_threshold_commit_still_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Protection is size-keyed, not a global kill switch for the sweep."""
    monkeypatch.setenv(MARATHON_ENV, "50000")
    manager = _manager()
    _, small_record = _pending_sized(manager, "sess-shallow", 4_096)
    manager.get_or_create("sess-vision")

    outcome = manager.abort_cross_session_postcommits(except_session_id="sess-vision")

    assert outcome is not None
    assert outcome["count"] == 1
    assert "marathon_protected" not in outcome
    assert small_record.abort_event.is_set()


def test_protection_grace_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wedged commit cannot starve foreign traffic forever: past the grace
    the foreground wins exactly as it did before."""
    monkeypatch.setenv(MARATHON_ENV, "50000")
    monkeypatch.setenv(MARATHON_GRACE_ENV, "0.05")
    manager = _manager()
    _, deep_record = _pending_sized(manager, "sess-deep", 200_000)
    manager.get_or_create("sess-vision")

    first = manager.abort_cross_session_postcommits(except_session_id="sess-vision")
    assert first is not None
    assert first["marathon_protected_count"] == 1
    assert not deep_record.abort_event.is_set()

    time.sleep(0.06)
    second = manager.abort_cross_session_postcommits(except_session_id="sess-vision")
    assert second is not None
    assert second["count"] == 1
    assert "marathon_protected" not in second
    assert deep_record.abort_event.is_set()


def test_rearmed_commit_inherits_the_spent_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The abort/re-arm chain shares ONE window. An aborted commit re-arms
    with a fresh record up to 16 times; a per-record deadline would multiply
    the bound by the retry chain, which is the starvation the reporter hit
    with MTPLX_POSTCOMMIT_FOREGROUND_GRACE_S."""
    monkeypatch.setenv(MARATHON_ENV, "50000")
    monkeypatch.setenv(MARATHON_GRACE_ENV, "0.05")
    manager = _manager()
    deep_session, _ = _pending_sized(manager, "sess-deep", 200_000)
    manager.get_or_create("sess-vision")

    manager.abort_cross_session_postcommits(except_session_id="sess-vision")
    time.sleep(0.06)
    manager.abort_cross_session_postcommits(except_session_id="sess-vision")

    # The yielded job re-arms with a brand-new record of the same size.
    retry_record = deep_session.set_pending_postcommit(
        Future(),
        reason="retokenized_history_mismatch",
        token_count=200_000,
    )
    third = manager.abort_cross_session_postcommits(except_session_id="sess-vision")

    assert third is not None
    assert third["count"] == 1
    assert "marathon_protected" not in third
    assert retry_record.abort_event.is_set()


def test_landed_commit_rearms_the_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a commit that actually lands refreshes the window, so the next
    marathon turn is protected again."""
    monkeypatch.setenv(MARATHON_ENV, "50000")
    monkeypatch.setenv(MARATHON_GRACE_ENV, "0.05")
    manager = _manager()
    deep_session, first_record = _pending_sized(manager, "sess-deep", 200_000)
    manager.get_or_create("sess-vision")

    manager.abort_cross_session_postcommits(except_session_id="sess-vision")
    time.sleep(0.06)
    manager.abort_cross_session_postcommits(except_session_id="sess-vision")
    deep_session.finish_pending_postcommit(
        first_record,
        {"stored": True, "mode": "retokenized_history", "prefix_len": 200_000},
    )

    next_record = deep_session.set_pending_postcommit(
        Future(),
        reason="retokenized_history_mismatch",
        token_count=200_000,
    )
    outcome = manager.abort_cross_session_postcommits(except_session_id="sess-vision")

    assert outcome is not None
    assert outcome["marathon_protected_count"] == 1
    assert not next_record.abort_event.is_set()


def test_aborted_outcome_does_not_rearm_the_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MARATHON_ENV, "50000")
    monkeypatch.setenv(MARATHON_GRACE_ENV, "0.05")
    manager = _manager()
    deep_session, first_record = _pending_sized(manager, "sess-deep", 200_000)
    manager.get_or_create("sess-vision")

    manager.abort_cross_session_postcommits(except_session_id="sess-vision")
    time.sleep(0.06)
    manager.abort_cross_session_postcommits(except_session_id="sess-vision")
    deep_session.finish_pending_postcommit(
        first_record,
        {"stored": False, "mode": "aborted", "reason": "foreground_preempted_postcommit"},
    )

    retry_record = deep_session.set_pending_postcommit(
        Future(),
        reason="retokenized_history_mismatch",
        token_count=200_000,
    )
    outcome = manager.abort_cross_session_postcommits(except_session_id="sess-vision")

    assert outcome is not None
    assert outcome["count"] == 1
    assert retry_record.abort_event.is_set()


def test_same_session_marathon_path_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The admitting session's own commit is still spared by identity, before
    any size question is asked."""
    monkeypatch.setenv(MARATHON_ENV, "50000")
    manager = _manager()
    own_session, own_record = _pending_sized(manager, "sess-deep", 1_024)

    outcome = manager.abort_cross_session_postcommits(except_session_id="sess-deep")

    assert outcome is None
    assert not own_record.abort_event.is_set()
    assert own_session.has_pending_postcommit()


def test_a_commit_cancelled_before_it_ran_stops_reading_as_pending() -> None:
    """#640: the scheduler skips a job cancelled before it started, so the job
    never finished its own record. The record read as pending for the
    session's life, and every later admission aborted it again: the report's
    ``"count": 262`` on every request was such records, not live commits."""
    manager = _manager()
    session, record = _pending(manager, "sess-a")
    manager.get_or_create("sess-b")

    first = manager.abort_cross_session_postcommits(except_session_id="sess-b")

    assert first is not None and first["count"] == 1
    assert record.future.cancelled()
    assert not session.has_pending_postcommit()
    assert session.last_postcommit_outcome["mode"] == "cancelled_before_start"
    # The next admission finds nothing left to abort.
    assert manager.abort_cross_session_postcommits(except_session_id="sess-b") is None


def test_a_running_commit_keeps_its_record_until_it_finishes() -> None:
    """A job already running is not cancelled: it sees its abort event at the
    next chunk boundary and finishes its own record, as before."""
    manager = _manager()
    session, record = _pending(manager, "sess-a")
    assert record.future.set_running_or_notify_cancel()
    manager.get_or_create("sess-b")

    outcome = manager.abort_cross_session_postcommits(except_session_id="sess-b")

    assert outcome is not None and outcome["count"] == 1
    assert record.abort_event.is_set()
    assert session.has_pending_postcommit()
    session.finish_pending_postcommit(record, {"stored": False, "mode": "aborted"})
    assert not session.has_pending_postcommit()
