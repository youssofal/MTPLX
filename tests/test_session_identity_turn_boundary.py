"""Issue #446: a long conversation keeps its session once the shared prefix is
a small fraction of the prompt.

A live session's committed stream carries the reasoning it streamed; clients
resend the history without it, so every later turn's raw common prefix with
the session ends exactly where the first turn started generating. The
fraction rule alone forked the identity at turn 5 of a 5 x 23k chain
(22,437 shared tokens: 25.06 % of an 89k prompt passed, 20.1 % of a 112k
prompt did not) and the restore fell to a 2,048-token block, 105 s of
prefill. A shared prefix on a recorded turn boundary is the conversation's
own signature at any length.
"""

from __future__ import annotations

import random

from mtplx.engine_session import (
    _COMMON_PREFIX_PROBE_TOKENS,
    _COMMON_PREFIX_REUSE_MIN_TOKENS,
    _TURN_PROMPT_LENS_MAX,
    EngineSession,
    EngineSessionManager,
    _common_prefix_reuse_threshold,
)


def _tokens(seed: int, count: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(5, 50_000) for _ in range(count)]


def _differing(token: int) -> int:
    return (int(token) % 50_000) + 50_001


def _resolve(manager: EngineSessionManager, prompt: list[int]) -> tuple[str, str, dict]:
    diagnostic: dict = {}
    session_id, source = manager.resolve_session_id(prompt_ids=prompt, diagnostic_out=diagnostic)
    return session_id, source, diagnostic


def test_resent_history_keeps_the_session_past_the_fraction_threshold() -> None:
    """The #446 chain shape: reasoning streamed, history resent without it."""
    manager = EngineSessionManager()
    first_prompt = _tokens(1, 5_000)
    streamed = _tokens(2, 40)
    session_id, source, _ = _resolve(manager, first_prompt)
    assert source == "new"
    session = manager.get_or_create(session_id)
    assert session.commit(prompt_ids=first_prompt, generated_ids=streamed, finish_reason="stop").committed
    assert session.turn_prompt_lens == [5_000]

    canonical = first_prompt + streamed  # what the session committed
    client = first_prompt + [_differing(streamed[0]), 7, 8]  # the client's rendering
    for turn in range(2, 6):
        user = _tokens(10 + turn, 20_000)
        prompt = client + user
        # The shared prefix is 5,000 tokens: below the fraction rule from turn 2 on.
        assert 5_000 < _common_prefix_reuse_threshold(len(prompt))
        resolved, source, diagnostic = _resolve(manager, prompt)
        assert resolved == session_id, turn
        assert source == "common_prefix_reuse"
        assert diagnostic["reason"] == "common_prefix_reuse"
        assert diagnostic["reuse_rule"] == "turn_boundary"
        assert diagnostic["turn_boundary"] == 5_000
        assert diagnostic["matched_prefix_len"] == 5_000
        generated = _tokens(20 + turn, 40)
        assert session.commit(
            prompt_ids=canonical + user, generated_ids=generated, finish_reason="stop"
        ).committed
        canonical = canonical + user + generated
        client = prompt + [_differing(generated[0]), 7, 8]


def test_without_a_recorded_turn_boundary_the_fraction_rule_alone_forks() -> None:
    """The 2.11.1 behaviour, kept as the receipt of what the boundary adds."""
    manager = EngineSessionManager()
    first_prompt = _tokens(1, 5_000)
    streamed = _tokens(2, 40)
    session_id, _, _ = _resolve(manager, first_prompt)
    session = manager.get_or_create(session_id)
    session.commit(prompt_ids=first_prompt, generated_ids=streamed, finish_reason="stop")
    prompt = first_prompt + [_differing(streamed[0]), 7, 8] + _tokens(3, 20_000)

    session.turn_prompt_lens.clear()
    resolved, source, diagnostic = _resolve(manager, prompt)
    assert resolved != session_id
    assert source == "new"
    assert diagnostic["reason"] == "prefix_divergence_at_token"

    session.note_turn_prompt_len(5_000)
    resolved, source, _ = _resolve(manager, prompt)
    assert resolved == session_id
    assert source == "common_prefix_reuse"


def test_fraction_reuse_is_unchanged_for_an_edited_history() -> None:
    manager = EngineSessionManager()
    prompt = _tokens(1, 10_000)
    session_id, _, _ = _resolve(manager, prompt)
    session = manager.get_or_create(session_id)
    session.commit(prompt_ids=prompt, generated_ids=_tokens(2, 40), finish_reason="stop")
    edited = prompt[:8_000] + [_differing(prompt[8_000])] + _tokens(3, 1_999)
    resolved, source, diagnostic = _resolve(manager, edited)
    assert resolved == session_id
    assert source == "common_prefix_reuse"
    assert diagnostic["reuse_rule"] == "fraction"
    assert diagnostic["turn_boundary"] is None
    assert diagnostic["matched_prefix_len"] == 8_000


def test_unrelated_conversation_sharing_a_long_system_prompt_stays_new() -> None:
    manager = EngineSessionManager()
    prompt = _tokens(1, 6_000)
    session_id, _, _ = _resolve(manager, prompt)
    session = manager.get_or_create(session_id)
    session.commit(prompt_ids=prompt, generated_ids=_tokens(2, 40), finish_reason="stop")
    shared = 4_500
    assert shared >= _COMMON_PREFIX_REUSE_MIN_TOKENS
    other = prompt[:shared] + [_differing(prompt[shared])] + _tokens(3, 20_000)
    resolved, source, diagnostic = _resolve(manager, other)
    assert resolved != session_id
    assert source == "new"
    assert diagnostic["matched_prefix_len"] == shared


def test_the_boundary_window_is_the_probe_width() -> None:
    manager = EngineSessionManager()
    prompt = _tokens(1, 6_000)
    generated = _tokens(2, 200)
    session_id, _, _ = _resolve(manager, prompt)
    session = manager.get_or_create(session_id)
    session.commit(prompt_ids=prompt, generated_ids=generated, finish_reason="stop")
    tail = _tokens(3, 20_000)

    inside = _COMMON_PREFIX_PROBE_TOKENS
    resent = prompt + generated[:inside] + [_differing(generated[inside])] + tail
    resolved, source, diagnostic = _resolve(manager, resent)
    assert resolved == session_id
    assert source == "common_prefix_reuse"
    assert diagnostic["reuse_rule"] == "turn_boundary"
    assert diagnostic["matched_prefix_len"] == 6_000 + inside

    past = _COMMON_PREFIX_PROBE_TOKENS + 1
    resent = prompt + generated[:past] + [_differing(generated[past])] + tail
    resolved, source, _ = _resolve(manager, resent)
    assert resolved != session_id
    assert source == "new"


def test_turn_prompt_lens_come_from_both_commit_paths_and_stay_bounded() -> None:
    session = EngineSession("s")
    assert session.commit_prompt_prefix(prompt_ids=_tokens(1, 5_000), finish_reason="tool_calls").committed
    assert session.turn_prompt_lens == [5_000]
    assert session.commit(
        prompt_ids=_tokens(1, 5_000) + _tokens(2, 300), generated_ids=_tokens(3, 10), finish_reason="stop"
    ).committed
    assert session.turn_prompt_lens == [5_000, 5_300]
    assert session.turn_boundary_at(5_300) == 5_300
    assert session.turn_boundary_at(5_000 + _COMMON_PREFIX_PROBE_TOKENS) == 5_000
    assert session.turn_boundary_at(4_999) is None
    assert session.turn_boundary_at(5_300 + _COMMON_PREFIX_PROBE_TOKENS + 1) is None

    for prompt_len in range(6_000, 6_000 + _TURN_PROMPT_LENS_MAX + 50):
        session.note_turn_prompt_len(prompt_len)
    assert len(session.turn_prompt_lens) == _TURN_PROMPT_LENS_MAX
    assert session.turn_prompt_lens[-1] == 6_000 + _TURN_PROMPT_LENS_MAX + 49
    assert session.to_admin_dict()["turn_prompt_lens"] == session.turn_prompt_lens[-8:]
