"""Claude Code names its session in a header; MTPLX takes it as the session id."""

import threading

import pytest

from mtplx import engine_session
from mtplx.engine_session import EngineSessionBusy, EngineSessionManager
from mtplx.server import openai

CLAUDE_CODE_SESSION = "e2617d79-cf1b-42df-b4fe-b8fa4d9d6168"


def test_the_claude_code_header_names_the_session():
    manager = EngineSessionManager()

    session_id, source = manager.resolve_session_id(
        headers={"X-Claude-Code-Session-Id": CLAUDE_CODE_SESSION},
        prompt_ids=[1, 2, 3],
    )

    assert session_id == CLAUDE_CODE_SESSION
    assert source == "header.x-claude-code-session-id"


def test_the_claude_code_header_wins_over_prompt_inference():
    manager = EngineSessionManager()
    other = manager.get_or_create("inferred")
    other.commit_prompt_prefix(
        prompt_ids=[1, 2, 3, 4],
        finish_reason="stop",
        boundary_kind="final",
    )

    session_id, source = manager.resolve_session_id(
        headers={"x-claude-code-session-id": CLAUDE_CODE_SESSION},
        prompt_ids=[1, 2, 3, 4, 5],
    )

    assert session_id == CLAUDE_CODE_SESSION
    assert source == "header.x-claude-code-session-id"


def test_an_explicit_mtplx_session_header_still_comes_first():
    manager = EngineSessionManager()

    session_id, source = manager.resolve_session_id(
        headers={
            "x-claude-code-session-id": CLAUDE_CODE_SESSION,
            "x-mtplx-session-id": "chosen-by-client",
        },
    )

    assert session_id == "chosen-by-client"
    assert source == "header.x-mtplx-session-id"


def test_a_blank_claude_code_header_falls_through():
    manager = EngineSessionManager()

    _, source = manager.resolve_session_id(
        headers={"x-claude-code-session-id": "  "},
    )

    assert source != "header.x-claude-code-session-id"


def test_a_claude_code_session_counts_as_named_by_the_client():
    assert openai._session_named_by_client("header.x-claude-code-session-id") is True


def test_a_claude_code_subagent_gets_its_own_session():
    manager = EngineSessionManager()

    parent, _ = manager.resolve_session_id(
        headers={"x-claude-code-session-id": CLAUDE_CODE_SESSION},
    )
    first, source = manager.resolve_session_id(
        headers={
            "x-claude-code-session-id": CLAUDE_CODE_SESSION,
            "X-Claude-Code-Agent-Id": "a1adfaff0e8902c9e",
        },
    )
    second, _ = manager.resolve_session_id(
        headers={
            "x-claude-code-session-id": CLAUDE_CODE_SESSION,
            "x-claude-code-agent-id": "ad76c715f46cff638",
        },
    )

    assert parent == CLAUDE_CODE_SESSION
    assert first == f"{CLAUDE_CODE_SESSION}:agent:a1adfaff0e8902c9e"
    assert len({parent, first, second}) == 3
    assert source == "header.x-claude-code-session-id"


def test_a_blank_agent_id_keeps_the_parent_session():
    manager = EngineSessionManager()

    session_id, _ = manager.resolve_session_id(
        headers={
            "x-claude-code-session-id": CLAUDE_CODE_SESSION,
            "x-claude-code-agent-id": " ",
        },
    )

    assert session_id == CLAUDE_CODE_SESSION


def test_a_busy_claude_code_session_forks_instead_of_refusing():
    manager = EngineSessionManager()
    named = manager.get_or_create(CLAUDE_CODE_SESSION)
    assert named.try_begin_generation()
    try:
        with manager.generation_slot(
            named, source="header.x-claude-code-session-id"
        ) as acquired:
            assert acquired.session_id != CLAUDE_CODE_SESSION
            assert acquired.session_id.startswith("anon-")
            assert acquired.in_flight
        assert not acquired.in_flight
    finally:
        named.end_generation()


def test_a_cancelled_claude_code_holder_is_waited_for_not_forked(monkeypatch):
    monkeypatch.setattr(engine_session, "_cancelled_holder_handoff_wait_s", lambda: 0.05)
    manager = EngineSessionManager()
    named = manager.get_or_create(CLAUDE_CODE_SESSION)
    holder_cancel = threading.Event()
    holder_cancel.set()
    assert named.try_begin_generation(cancel_event=holder_cancel)
    try:
        with (
            pytest.raises(EngineSessionBusy, match="did not stop"),
            manager.generation_slot(named, source="header.x-claude-code-session-id"),
        ):
            pass
    finally:
        named.end_generation()


def test_other_named_sessions_still_refuse_when_busy():
    manager = EngineSessionManager()
    named = manager.get_or_create("chosen-by-client")
    assert named.try_begin_generation()
    try:
        with (
            pytest.raises(EngineSessionBusy),
            manager.generation_slot(named, source="header.x-mtplx-session-id"),
        ):
            pass
    finally:
        named.end_generation()
