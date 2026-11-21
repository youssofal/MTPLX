"""A conversation under 4,096 tokens keeps its session when the client resends
it and the resent history leaves the generated stream inside the model's answer.

Hermes sends no session id and re-serializes its tool calls. On its first
tool-result turn the resent history left the generated stream 63 tokens into
the answer, at 4,087 shared tokens: under the 4,096-token floor of the
turn-boundary rule, so the request became a new session and the turn re-read
the model's own 345-token answer. A match that runs at least
_GENERATED_TURN_MATCH_MIN_TOKENS into the tokens the session generated is that
conversation's own answer, at any length. A match that ends on a turn's prompt
is not: two conversations that open with the same system prompt and user
message share exactly that, so they stay apart under the floor as before.
"""

from __future__ import annotations

import random

from mtplx.engine_session import (
    _COMMON_PREFIX_REUSE_MIN_TOKENS,
    _GENERATED_TURN_MATCH_MIN_TOKENS,
    EngineSessionManager,
)

PROMPT = 4_024  # Hermes's system prompt, tools and first user message
ANSWER = 345  # the model's first answer: its reasoning and the tool call
LEFT_AT = 63  # where the resent tool call leaves the generated stream


def _tokens(seed: int, count: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(5, 50_000) for _ in range(count)]


def _differing(token: int) -> int:
    return (int(token) % 50_000) + 50_001


def _resolve(manager: EngineSessionManager, prompt: list[int]) -> tuple[str, str, dict]:
    diagnostic: dict = {}
    session_id, source = manager.resolve_session_id(prompt_ids=prompt, diagnostic_out=diagnostic)
    return session_id, source, diagnostic


def _answered(manager: EngineSessionManager, prompt: list[int], answer: list[int]) -> str:
    session_id, source, _ = _resolve(manager, prompt)
    assert source == "new"
    session = manager.get_or_create(session_id)
    assert session.commit(prompt_ids=prompt, generated_ids=answer, finish_reason="tool_calls").committed
    return session_id


def _resent(prompt: list[int], answer: list[int], left_at: int, tail_seed: int) -> list[int]:
    """The client's next request: the answer re-rendered from ``left_at`` on, then the tool result."""
    return prompt + answer[:left_at] + [_differing(answer[left_at])] + _tokens(tail_seed, 160)


def test_the_tool_result_turn_keeps_the_session_under_the_floor() -> None:
    manager = EngineSessionManager()
    prompt, answer = _tokens(1, PROMPT), _tokens(2, ANSWER)
    session_id = _answered(manager, prompt, answer)
    resent = _resent(prompt, answer, LEFT_AT, 3)
    assert PROMPT + LEFT_AT < _COMMON_PREFIX_REUSE_MIN_TOKENS

    resolved, source, diagnostic = _resolve(manager, resent)

    assert resolved == session_id
    assert source == "common_prefix_reuse"
    assert diagnostic["reuse_rule"] == "inside_generated_turn"
    assert diagnostic["matched_prefix_len"] == PROMPT + LEFT_AT
    assert diagnostic["turn_boundary"] == PROMPT


def test_the_match_must_run_the_minimum_into_the_answer() -> None:
    manager = EngineSessionManager()
    prompt, answer = _tokens(1, PROMPT), _tokens(2, ANSWER)
    session_id = _answered(manager, prompt, answer)

    short = _resent(prompt, answer, _GENERATED_TURN_MATCH_MIN_TOKENS - 1, 3)
    resolved, source, _ = _resolve(manager, short)
    assert resolved != session_id
    assert source == "new"

    enough = _resent(prompt, answer, _GENERATED_TURN_MATCH_MIN_TOKENS, 3)
    resolved, source, diagnostic = _resolve(manager, enough)
    assert resolved == session_id
    assert diagnostic["reuse_rule"] == "inside_generated_turn"


def test_a_conversation_that_opens_the_same_way_stays_new_under_the_floor() -> None:
    """Same system prompt and user message, a different answer of its own."""
    manager = EngineSessionManager()
    prompt = _tokens(1, PROMPT)
    first = _answered(manager, prompt, _tokens(2, ANSWER))

    resolved, source, diagnostic = _resolve(manager, list(prompt))
    assert resolved != first
    assert source == "new"
    assert diagnostic["matched_prefix_len"] == PROMPT

    other_answer = _tokens(4, ANSWER)
    other = _answered(manager, list(prompt), other_answer)
    assert other != first
    resolved, source, _ = _resolve(manager, _resent(prompt, other_answer, LEFT_AT, 5))
    assert resolved == other
    assert source == "common_prefix_reuse"


def test_the_deepest_answer_match_wins() -> None:
    """Two conversations whose answers open alike: the resent one is the one it matches furthest."""
    manager = EngineSessionManager()
    prompt = _tokens(1, PROMPT)
    opening = _tokens(2, 20)
    first_answer = opening + _tokens(3, ANSWER - 20)
    second_answer = opening + _tokens(4, ANSWER - 20)
    first = _answered(manager, prompt, first_answer)
    second = _answered(manager, list(prompt), second_answer)
    assert first != second

    resolved, source, diagnostic = _resolve(manager, _resent(prompt, second_answer, LEFT_AT, 5))
    assert resolved == second
    assert diagnostic["matched_prefix_len"] == PROMPT + LEFT_AT


def test_a_short_conversation_sharing_only_a_system_prompt_stays_new() -> None:
    manager = EngineSessionManager()
    prompt, answer = _tokens(1, PROMPT), _tokens(2, ANSWER)
    session_id = _answered(manager, prompt, answer)
    shared = 2_000
    other = prompt[:shared] + [_differing(prompt[shared])] + _tokens(3, 2_000)

    resolved, source, diagnostic = _resolve(manager, other)

    assert resolved != session_id
    assert source == "new"
    assert diagnostic["matched_prefix_len"] == shared


def test_a_later_turn_under_the_floor_keeps_the_session() -> None:
    """Turn two of the same short conversation: resent inside the second answer."""
    manager = EngineSessionManager()
    prompt, answer = _tokens(1, 1_500), _tokens(2, 200)
    session_id = _answered(manager, prompt, answer)
    session = manager.get_or_create(session_id)
    second_prompt = prompt + answer + _tokens(3, 300)
    second_answer = _tokens(4, 250)
    resolved, source, _ = _resolve(manager, second_prompt)
    assert resolved == session_id
    assert source == "longest_prefix"
    assert session.commit(prompt_ids=second_prompt, generated_ids=second_answer, finish_reason="tool_calls").committed

    resolved, source, diagnostic = _resolve(manager, _resent(second_prompt, second_answer, 40, 5))

    assert resolved == session_id
    assert source == "common_prefix_reuse"
    assert diagnostic["reuse_rule"] == "inside_generated_turn"
    assert diagnostic["turn_boundary"] == len(second_prompt)
