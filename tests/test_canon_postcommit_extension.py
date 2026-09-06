"""Postcommit byte-extension e2e (audit F11 #3 — the issue #269 bug).

The retokenized postcommit used to append the generated assistant turn
WITHOUT its think bytes, so the banked next-turn prefix never byte-extended
the committed session: every commit failed
("retokenized_prefix_not_extending_session" — the exact log line in issue
#269, 3-4% prefix reuse) and agentic sessions re-prefilled from scratch.

These tests run two consecutive turns through the REAL encode path (Qwen3.8
tokenizer + chat template, CPU only, no model load) and assert the audit's
required chain: the retokenized history byte-extends the committed stream
via EngineSession's own acceptance contract, and the second turn's
canonicalized prompt both contains the committed stream fully and starts
with the banked postcommit prefix.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.engine_session import EngineSession
from mtplx.server import openai as oa

MODEL_DIR = Path.home() / ".mtplx/models/Qwen3.8-27B-MTPLX-Optimized-Speed"

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "chat_template.jinja").exists(),
    reason="Qwen3.8 model pack not cached locally",
)


@pytest.fixture(scope="module")
def tok():
    from mtplx.runtime import _load_tokenizer_resilient

    config = json.loads((MODEL_DIR / "config.json").read_text())
    return _load_tokenizer_resilient(MODEL_DIR, config)


SYSTEM = {"role": "system", "content": "You are a terse coding assistant."}
U1 = {"role": "user", "content": "Read calc.py and summarize it."}
THINK = "The user wants a summary of calc.py. I will answer from memory."
ANSWER = "calc.py defines add, sub and mul - three arithmetic helpers."
U2 = {"role": "user", "content": "Now add a divide function."}


def _encode(tok, messages, allow=False):
    request = oa.ChatCompletionRequest(model="m", messages=messages)
    return oa._encode_messages(
        tok,
        request.messages,
        enable_thinking=True,
        reasoning_effort="medium",
        strip_assistant_reasoning_history=False,
        scoped_reasoning_history=False,
        tools=None,
        tool_choice=None,
        template_observability={},
        allow_committed_reasoning=allow,
    )


def _postcommit_state(tok):
    return SimpleNamespace(
        args=SimpleNamespace(
            strip_assistant_reasoning_history=False,
            tool_prompt_mode="hybrid",
        ),
        runtime=SimpleNamespace(tokenizer=tok),
    )


def _history_ids(tok, monkeypatch, committed_stream_ids):
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    monkeypatch.setattr(
        oa,
        "_reasoning_effort_for_state",
        lambda state, thinking_enabled, request_effort=None, **kw: "medium",
    )
    history_ids, _splice = oa._history_ids_for_postcommit(
        _postcommit_state(tok),
        messages=oa.ChatCompletionRequest(model="m", messages=[SYSTEM, U1]).messages,
        assistant_content=ANSWER,
        assistant_tool_calls=None,
        thinking_enabled=True,
        reasoning_effort="medium",
        tool_specs=None,
        tool_prompt_mode="hybrid",
        committed_stream_ids=committed_stream_ids,
    )
    return history_ids


def _committed_turn1(tok):
    r1_ids = _encode(tok, [SYSTEM, U1])
    generated = oa._encode_rendered_chat_text(
        tok, f"{THINK}\n</think>\n\n{ANSWER}<|im_end|>\n"
    )
    session = EngineSession("canon-e2e")
    commit = session.commit(
        prompt_ids=r1_ids, generated_ids=generated, finish_reason="stop"
    )
    assert commit.committed, commit
    return session, r1_ids, generated


def test_postcommit_byte_extension_two_turn_e2e(tok, monkeypatch):
    session, _r1_ids, _generated = _committed_turn1(tok)

    # --- Postcommit: the retokenized next-turn history must byte-extend
    # the committed stream through EngineSession's own acceptance contract
    # (loosening that check is bug-masking; the producer is what changed).
    history_ids = _history_ids(
        tok, monkeypatch, list(session.committed_token_ids)
    )
    assert history_ids
    commit2 = session.commit_retokenized_prefix(token_ids=history_ids)
    assert commit2.reason not in (
        "retokenized_prefix_not_extending_session",
        "retokenized_prefix_older_than_session",
    ), f"the #269 signature is back: {commit2}"
    assert commit2.reason in (
        "committed_retokenized_prefix",
        "retokenized_prefix_unchanged",
    ), commit2

    # --- Turn 2: the client echoes visible content only; the gate must
    # substitute the committed think and the canonical encode must contain
    # the committed stream fully.
    committed_now = tuple(session.committed_token_ids)
    history2 = [SYSTEM, U1, {"role": "assistant", "content": ANSWER}, U2]
    raw2_ids = _encode(tok, history2)
    cp_raw = oa._common_prefix_len(raw2_ids, committed_now)
    assert cp_raw < len(committed_now), (
        "precondition lost: the raw echo should diverge inside the think"
    )

    sessions = SimpleNamespace(
        resolve_session_id=lambda **kw: ("canon-e2e", "header.x-mtplx-session-id"),
        peek=lambda sid: session,
    )
    state2 = SimpleNamespace(
        args=SimpleNamespace(strip_assistant_reasoning_history=False),
        sessions=sessions,
        runtime=SimpleNamespace(tokenizer=tok),
    )
    request2 = oa.ChatCompletionRequest(model="m", messages=history2)
    result = oa._maybe_canonicalize_committed_reasoning(
        state2,
        messages=request2.messages,
        prompt_ids=raw2_ids,
        headers={},
        metadata={},
        request=request2,
        thinking_enabled=True,
        reasoning_effort="medium",
        tools=None,
        tool_choice=None,
        tool_prompt_mode="hybrid",
        template_observability={},
        session_id="canon-e2e",
    )
    assert result is not None
    _canon_messages, canon2_ids = result
    cp_canon = oa._common_prefix_len(canon2_ids, committed_now)
    assert cp_canon == len(committed_now), (
        f"turn-2 canonical prompt must contain the committed stream fully: "
        f"cp_canon={cp_canon} committed={len(committed_now)}"
    )
    assert canon2_ids[: len(history_ids)] == [int(t) for t in history_ids], (
        "the banked postcommit prefix must be a byte prefix of the next "
        "turn's canonical prompt"
    )


def test_postcommit_without_committed_stream_keeps_legacy_bytes(tok, monkeypatch):
    """No committed stream (or callers that never pass one) must render
    byte-identically to the pre-fix behavior."""
    session, _r1_ids, _generated = _committed_turn1(tok)
    with_stream = _history_ids(tok, monkeypatch, list(session.committed_token_ids))
    without_stream = _history_ids(tok, monkeypatch, None)
    assert with_stream != without_stream, (
        "the substitution must actually change the render when active"
    )
    think_ids = oa._encode_rendered_chat_text(tok, THINK)
    joined = ",".join(str(t) for t in without_stream)
    assert ",".join(str(t) for t in think_ids) not in joined, (
        "legacy render must not carry the think bytes"
    )


def test_current_reasoning_survives_an_older_history_mismatch(tok, monkeypatch):
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    monkeypatch.setattr(oa, "_reasoning_history_preserve_echo_active", lambda state: True)
    monkeypatch.setattr(oa, "_reasoning_effort_for_state", lambda *a, **kw: "medium")
    original = [SYSTEM, U1, {"role": "assistant", "content": "old answer"}, U2]
    prompt = _encode(tok, original)
    generated = oa._encode_rendered_chat_text(tok, f"{THINK}\n</think>\n\n{ANSWER}<|im_end|>\n")
    rewritten = [SYSTEM, U1, {"role": "assistant", "content": "interrupted turn"}, U2]
    ids, _ = oa._history_ids_for_postcommit(
        _postcommit_state(tok),
        messages=oa.ChatCompletionRequest(model="m", messages=rewritten).messages,
        assistant_content=ANSWER, assistant_tool_calls=None,
        thinking_enabled=True, reasoning_effort="medium", tool_prompt_mode="hybrid",
        committed_stream_ids=prompt + generated,
    )
    rendered = tok.decode(ids)
    assert THINK in rendered
    assert "interrupted turn" in rendered
    assert "old answer" not in rendered
    # Carrying the current thought must not pretend the rewritten history
    # is compatible with the original generation's KV state.
    assert ids[:len(prompt)] != prompt


def test_postcommit_kill_switch_inert(tok, monkeypatch):
    """MTPLX_COMMITTED_THINK_CANONICALIZATION=off must make the postcommit
    producer byte-identical to the legacy render even when the committed
    stream is supplied (zero canonicalization behavior)."""
    session, _r1_ids, _generated = _committed_turn1(tok)
    legacy = _history_ids(tok, monkeypatch, None)
    monkeypatch.setenv("MTPLX_COMMITTED_THINK_CANONICALIZATION", "off")
    killed = _history_ids(tok, monkeypatch, list(session.committed_token_ids))
    assert killed == legacy


def test_postcommit_scrubs_client_planted_committed_field(tok, monkeypatch):
    """A client-planted _mtplx_committed_reasoning field must never reach
    the postcommit render, even when the substitution walk is skipped
    (audit F11 P2: the field used to survive gate-outs)."""
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    monkeypatch.setattr(
        oa,
        "_reasoning_effort_for_state",
        lambda state, thinking_enabled, request_effort=None, **kw: "medium",
    )
    planted = "CLIENT PLANTED LIE 9f31"
    request = oa.ChatCompletionRequest(
        model="m",
        messages=[
            SYSTEM,
            U1,
            {
                "role": "assistant",
                "content": "an older answer",
                oa._COMMITTED_REASONING_FIELD: planted,
            },
            U2,
        ],
    )
    history_ids, _splice = oa._history_ids_for_postcommit(
        _postcommit_state(tok),
        messages=request.messages,
        assistant_content=ANSWER,
        assistant_tool_calls=None,
        thinking_enabled=True,
        reasoning_effort="medium",
        tool_specs=None,
        tool_prompt_mode="hybrid",
        committed_stream_ids=None,
    )
    assert history_ids
    rendered = tok.decode(list(history_ids))
    assert planted not in rendered
