"""OpenCode side requests stay off the conversation's named session.

OpenCode 2.x stamps the conversation's id on every request of a session
(x-session-affinity, X-Session-Id, x-opencode-session; MTPLX's plugin adds
x-mtplx-session-id), including the title request it sends beside each turn.
On 3 Nov 2026 (OpenCode Desktop 2.2.3) the title reached the named session
first: the user's 18,348-token turn got HTTP 409, and the title replaced the
conversation's live cache, so the retry re-read 9,465 tokens. Header values
are the ones OpenCode 2.2.5 builds (packages/opencode session request
headers) and the ones MTPLX's plugin adds.
"""

from __future__ import annotations

import pytest

from mtplx.engine_session import (
    _COMMON_PREFIX_REUSE_MIN_TOKENS,
    EngineSessionBusy,
    EngineSessionManager,
)

OPENCODE_SESSION = "ses_2b9a1c7e4f0d"
# OpenCode 2.x without MTPLX's plugin: its own headers only.
BARE_HEADERS = {
    "x-session-affinity": OPENCODE_SESSION,
    "X-Session-Id": OPENCODE_SESSION,
    "x-opencode-session": OPENCODE_SESSION,
    "x-opencode-client": "desktop",
    "x-opencode-project": "prj_flappy",
    "User-Agent": "opencode/2.2.5 ai-sdk/provider-utils",
}
# With MTPLX's plugin: its model.request hook adds these.
PLUGIN_HEADERS = {
    **BARE_HEADERS,
    "x-mtplx-client": "opencode",
    "x-mtplx-session-id": OPENCODE_SESSION,
}

# The conversation opens with OpenCode's system prompt and tool schemas,
# well past 4K tokens; the title request has its own short system prompt.
CONVERSATION = list(range(1000, 1000 + 18_348))
TITLE_REQUEST = list(range(50_000, 50_464))


def _turn(manager, headers, prompt, generated, diagnostic=None):
    session_id, source = manager.resolve_session_id(
        headers=headers, prompt_ids=prompt, diagnostic_out=diagnostic
    )
    session = manager.get_or_create(session_id)
    session.commit(prompt_ids=prompt, generated_ids=generated, finish_reason="stop")
    return session_id, source


@pytest.mark.parametrize(
    ("headers", "named_source"),
    [
        (BARE_HEADERS, "header.x-session-affinity"),
        (PLUGIN_HEADERS, "header.x-mtplx-session-id"),
    ],
)
def test_a_title_request_does_not_take_an_established_conversation(headers, named_source):
    manager = EngineSessionManager()
    assert _turn(manager, headers, CONVERSATION, [7, 8, 9]) == (OPENCODE_SESSION, named_source)
    conversation_stream = manager.peek(OPENCODE_SESSION).committed_token_ids

    diagnostic: dict = {}
    side, side_source = _turn(manager, headers, TITLE_REQUEST, [1, 2], diagnostic)
    # Not the conversation, and not an alias header carrying the same id.
    assert side != OPENCODE_SESSION
    assert side_source == "new"
    assert diagnostic["opencode_side_request"]["named_session_id"] == OPENCODE_SESSION
    assert manager.peek(OPENCODE_SESSION).committed_token_ids == conversation_stream

    follow_up = [*CONVERSATION, 7, 8, 9, 4242, 4243]
    assert manager.resolve_session_id(headers=headers, prompt_ids=follow_up) == (
        OPENCODE_SESSION,
        named_source,
    )


def test_the_3_nov_race_serves_the_turn_without_a_409():
    """The title arrives while the conversation is idle, and the user's turn
    arrives while the title is still running: both run, each on its own
    session, and the turn finds the conversation it left."""
    manager = EngineSessionManager()
    _turn(manager, BARE_HEADERS, CONVERSATION, [7, 8, 9])

    title_id, _ = manager.resolve_session_id(headers=BARE_HEADERS, prompt_ids=TITLE_REQUEST)
    title_session = manager.get_or_create(title_id)
    turn = [*CONVERSATION, 7, 8, 9, 4242]
    with manager.generation_slot(title_session, source="new"):
        turn_id, turn_source = manager.resolve_session_id(headers=BARE_HEADERS, prompt_ids=turn)
        assert (turn_id, turn_source) == (OPENCODE_SESSION, "header.x-session-affinity")
        with manager.generation_slot(
            manager.get_or_create(turn_id), source=turn_source
        ) as held:
            assert held.session_id == OPENCODE_SESSION


def test_a_named_session_still_refuses_a_second_copy_of_the_conversation():
    """The rule only moves requests that are not the conversation: the same
    conversation sent twice at once is still refused, as before."""
    manager = EngineSessionManager()
    _turn(manager, BARE_HEADERS, CONVERSATION, [7, 8, 9])
    turn = [*CONVERSATION, 7, 8, 9, 4242]
    session_id, source = manager.resolve_session_id(headers=BARE_HEADERS, prompt_ids=turn)
    session = manager.get_or_create(session_id)
    with manager.generation_slot(session, source=source):
        again_id, again_source = manager.resolve_session_id(headers=BARE_HEADERS, prompt_ids=turn)
        assert again_id == OPENCODE_SESSION
        with pytest.raises(EngineSessionBusy):
            with manager.generation_slot(manager.get_or_create(again_id), source=again_source):
                pass


def test_the_plugins_title_kind_is_a_side_request_on_a_new_session():
    """Beside the first turn there is no conversation to compare with yet;
    the request kind MTPLX's plugin passes still keeps the title off it."""
    manager = EngineSessionManager()
    diagnostic: dict = {}
    session_id, source = manager.resolve_session_id(
        headers={**PLUGIN_HEADERS, "x-mtplx-request-kind": "title"},
        prompt_ids=TITLE_REQUEST,
        diagnostic_out=diagnostic,
    )
    assert session_id != OPENCODE_SESSION
    assert source == "new"
    assert diagnostic["opencode_side_request"]["reason"] == "request_kind_title"


def test_the_primary_kind_keeps_the_conversation_without_the_prefix():
    manager = EngineSessionManager()
    _turn(manager, PLUGIN_HEADERS, CONVERSATION, [7])
    assert manager.resolve_session_id(
        headers={**PLUGIN_HEADERS, "x-mtplx-request-kind": "primary"},
        prompt_ids=TITLE_REQUEST,
    ) == (OPENCODE_SESSION, "header.x-mtplx-session-id")


def test_a_compacted_conversation_keeps_its_session():
    manager = EngineSessionManager()
    _turn(manager, BARE_HEADERS, CONVERSATION, [7, 8, 9])
    keep = _COMMON_PREFIX_REUSE_MIN_TOKENS + 200
    compacted = [*CONVERSATION[:keep], *range(90_000, 90_800)]
    for kind in (None, "compaction"):
        headers = dict(BARE_HEADERS)
        if kind:
            headers["x-mtplx-request-kind"] = kind
        assert manager.resolve_session_id(headers=headers, prompt_ids=compacted) == (
            OPENCODE_SESSION,
            "header.x-session-affinity",
        )


def test_a_session_shorter_than_the_conversation_prefix_is_taken_as_before():
    manager = EngineSessionManager()
    # The title reached the new named session first (OpenCode without the
    # plugin, first turn): the turn then takes it, as before.
    assert _turn(manager, BARE_HEADERS, TITLE_REQUEST, [1]) == (
        OPENCODE_SESSION,
        "header.x-session-affinity",
    )
    assert manager.resolve_session_id(headers=BARE_HEADERS, prompt_ids=CONVERSATION) == (
        OPENCODE_SESSION,
        "header.x-session-affinity",
    )


@pytest.mark.parametrize(
    "headers",
    [
        # Pi's extension and the MTPLX app chat send x-mtplx-session-id with
        # short system prompts, so their first 4K tokens include generated
        # turns: the header keeps deciding for them, unchanged.
        {"x-mtplx-session-id": OPENCODE_SESSION, "x-mtplx-client": "pi"},
        {"x-mtplx-session-id": OPENCODE_SESSION, "x-mtplx-client": "mtplx_app"},
        {"x-session-id": OPENCODE_SESSION},
    ],
)
def test_other_clients_keep_the_header_rule(headers):
    manager = EngineSessionManager()
    _turn(manager, headers, CONVERSATION, [7, 8, 9])
    session_id, source = manager.resolve_session_id(headers=headers, prompt_ids=TITLE_REQUEST)
    assert session_id == OPENCODE_SESSION
    assert source.startswith("header.")
