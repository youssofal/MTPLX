"""Claude Code side requests stay off the conversation's named session.

Claude Code stamps X-Claude-Code-Session-Id on every request of a session,
including its side requests: the session title, the auto-mode safety
classifier, the Bash command-prefix check. Their prompts share nothing with
the conversation's. If one took the conversation's session, its generation
commit would replace the committed stream that the next turn's reasoning
re-insertion and token splice read. Header values are the ones claude-cli
2.1.287 sent in the capture behind PR #606.
"""

from __future__ import annotations

from mtplx.engine_session import _COMMON_PREFIX_REUSE_MIN_TOKENS, EngineSessionManager

CLAUDE_CODE_SESSION = "e2617d79-cf1b-42df-b4fe-b8fa4d9d6168"
SUBAGENT = "a1adfaff0e8902c9e"
HEADERS = {"X-Claude-Code-Session-Id": CLAUDE_CODE_SESSION}
SOURCE = "header.x-claude-code-session-id"

# The conversation: system prompt and tool contract (Claude Code's run to
# well over 4K tokens), then history. A side request: its own system prompt.
CONVERSATION = list(range(1000, 1000 + 6000))
SIDE_REQUEST = list(range(50_000, 50_300))


def _turn(manager, headers, prompt, generated, diagnostic=None):
    session_id, source = manager.resolve_session_id(
        headers=headers, prompt_ids=prompt, diagnostic_out=diagnostic
    )
    session = manager.get_or_create(session_id)
    session.commit(prompt_ids=prompt, generated_ids=generated, finish_reason="tool_calls")
    return session_id, source


def test_a_side_request_does_not_take_an_established_conversation():
    manager = EngineSessionManager()
    first, source = _turn(manager, HEADERS, CONVERSATION, [7, 8, 9])
    assert (first, source) == (CLAUDE_CODE_SESSION, SOURCE)
    conversation_stream = manager.peek(CLAUDE_CODE_SESSION).committed_token_ids

    diagnostic: dict = {}
    side, side_source = _turn(manager, HEADERS, SIDE_REQUEST, [1, 2], diagnostic)
    assert side != CLAUDE_CODE_SESSION
    assert side_source == "new"
    assert diagnostic["claude_code_side_request"]["named_session_id"] == CLAUDE_CODE_SESSION
    # The side request's own commit landed on its own session.
    assert manager.peek(CLAUDE_CODE_SESSION).committed_token_ids == conversation_stream

    follow_up = [*CONVERSATION, 7, 8, 9, 4242, 4243]
    assert manager.resolve_session_id(headers=HEADERS, prompt_ids=follow_up) == (
        CLAUDE_CODE_SESSION,
        SOURCE,
    )


def test_a_compacted_conversation_keeps_its_session():
    manager = EngineSessionManager()
    _turn(manager, HEADERS, CONVERSATION, [7, 8, 9])
    # Compaction keeps the system prompt and tools and rewrites the history.
    keep = _COMMON_PREFIX_REUSE_MIN_TOKENS + 200
    compacted = [*CONVERSATION[:keep], *range(90_000, 90_800)]
    assert manager.resolve_session_id(headers=HEADERS, prompt_ids=compacted) == (
        CLAUDE_CODE_SESSION,
        SOURCE,
    )


def test_a_session_shorter_than_the_conversation_prefix_is_taken_as_before():
    manager = EngineSessionManager()
    # A side request reached the named session first (nothing else had).
    assert _turn(manager, HEADERS, SIDE_REQUEST, [1]) == (CLAUDE_CODE_SESSION, SOURCE)
    assert manager.resolve_session_id(headers=HEADERS, prompt_ids=CONVERSATION) == (
        CLAUDE_CODE_SESSION,
        SOURCE,
    )


def test_a_subagent_side_request_stays_off_the_subagent_session():
    manager = EngineSessionManager()
    headers = {**HEADERS, "X-Claude-Code-Agent-Id": SUBAGENT}
    subagent = f"{CLAUDE_CODE_SESSION}:agent:{SUBAGENT}"
    assert _turn(manager, headers, CONVERSATION, [5]) == (subagent, SOURCE)
    side, _ = manager.resolve_session_id(headers=headers, prompt_ids=SIDE_REQUEST)
    assert side not in {subagent, CLAUDE_CODE_SESSION}


def test_a_request_without_prompt_ids_keeps_the_header():
    manager = EngineSessionManager()
    _turn(manager, HEADERS, CONVERSATION, [7])
    assert manager.resolve_session_id(headers=HEADERS) == (CLAUDE_CODE_SESSION, SOURCE)
