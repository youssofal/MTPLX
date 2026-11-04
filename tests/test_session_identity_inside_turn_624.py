"""Issue #624: a long conversation without a session id keeps its session when
the client's rendering of a turn diverges inside the tokens MTPLX generated.

The #446 rule accepts a shared prefix that ends within 64 tokens past a
recorded turn start. In #624 the raw divergence sat 96 tokens into MTPLX's
own re-rendering of the first assistant turn after a compaction (at its
tool-call block), so only the 25% fraction rule kept the session, and it
stopped holding once the prompt passed 4 x 40,709 tokens: request 150
(163,242 tokens, 99.9% shared with request 149) got a new anon id and
restored 35,440 tokens instead of about 163,000 (TTFT 95.6 s).
"""

from __future__ import annotations

import random

from mtplx.engine_session import (
    _COMMON_PREFIX_PROBE_TOKENS,
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


def test_a_divergence_inside_a_generated_turn_keeps_the_session_at_any_length() -> None:
    """The #624 shape: the client re-sends turn 1, MTPLX re-renders it and the
    raw stream diverges 96 tokens into what MTPLX generated."""
    manager = EngineSessionManager()
    first_prompt = _tokens(1, 40_613)
    generated = _tokens(2, 400)  # a tool-call turn with text before it
    inside = 96
    assert inside > _COMMON_PREFIX_PROBE_TOKENS
    session_id, source, _ = _resolve(manager, first_prompt)
    assert source == "new"
    session = manager.get_or_create(session_id)
    assert session.commit(
        prompt_ids=first_prompt, generated_ids=generated, finish_reason="tool_calls"
    ).committed

    cp_raw = len(first_prompt) + inside
    client = first_prompt + generated[:inside] + [_differing(generated[inside])] + _tokens(3, 300)
    canonical = first_prompt + generated
    grown = 0
    turn = 0
    # Append tool results until the prompt is well past 4 x cp_raw.
    while len(client) <= 4 * cp_raw + 20_000:
        turn += 1
        result = _tokens(100 + turn, 9_000)
        prompt = client + result
        resolved, source, diagnostic = _resolve(manager, prompt)
        assert resolved == session_id, (turn, len(prompt), diagnostic)
        assert source == "common_prefix_reuse"
        assert diagnostic["matched_prefix_len"] == cp_raw
        if cp_raw < _common_prefix_reuse_threshold(len(prompt)):
            grown += 1
            assert diagnostic["reuse_rule"] == "turn_boundary", diagnostic
            assert diagnostic["turn_boundary"] == len(first_prompt)
        reply = _tokens(200 + turn, 120)
        assert session.commit(
            prompt_ids=canonical + result, generated_ids=reply, finish_reason="tool_calls"
        ).committed
        canonical = canonical + result + reply
        client = prompt + reply
    # The loop must have crossed the point where the fraction rule alone forks.
    assert grown > 0


def test_a_divergence_in_a_message_the_client_wrote_is_still_a_new_conversation() -> None:
    """Accepting the generated span must not accept a different client message:
    a shared prefix that ends inside the next user message (after turn 1's
    last generated token and before turn 2's prompt end) gets only the
    fraction rule, as before."""
    manager = EngineSessionManager()
    first_prompt = _tokens(1, 6_000)
    generated = _tokens(2, 200)
    session_id, _, _ = _resolve(manager, first_prompt)
    session = manager.get_or_create(session_id)
    session.commit(prompt_ids=first_prompt, generated_ids=generated, finish_reason="stop")
    user2 = _tokens(4, 300)
    second_prompt = first_prompt + generated + user2
    session.commit(prompt_ids=second_prompt, generated_ids=_tokens(5, 50), finish_reason="stop")
    # Shares turn 1 and the first 40 tokens of the second user message, then differs.
    shared = len(first_prompt) + len(generated) + 40
    other = second_prompt[:shared] + [_differing(second_prompt[shared])] + _tokens(3, 40_000)
    assert shared < _common_prefix_reuse_threshold(len(other))
    resolved, source, diagnostic = _resolve(manager, other)
    assert resolved != session_id
    assert source == "new"
    assert diagnostic["matched_prefix_len"] == shared
