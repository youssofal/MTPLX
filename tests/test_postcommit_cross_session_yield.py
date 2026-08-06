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
