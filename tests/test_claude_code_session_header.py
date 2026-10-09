"""Claude Code names its session in a header; MTPLX takes it as the session id."""

from mtplx.engine_session import EngineSessionManager
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
