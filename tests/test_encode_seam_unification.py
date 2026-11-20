"""Encode-seam unification (2026-08-21 founder-session walls 3913/4041/4444).

Under tool_prompt_mode=compact (the repaired OpenCode lane) the raw
transcript encode and the postcommit encode used to plain-tokenize while the
committed-reasoning canonical encode segmented at assistant '<think>\n'
generation seams. A single-pass BPE merges that seam newline with what
follows, so identical rendered text produced different TOKENS per path and
the committed prefix died at the first assistant seam on every request.

These tests run the REAL Qwen3.8 tokenizer + chat template (CPU only):
one segmentation policy means raw == canonical for identical rendered text,
the compact-lane postcommit byte-extends the committed session, and the
postcommit backfills think interiors from the session's committed stream so
a canon-miss request can no longer regress the committed session
(real->empty->real oscillation).
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
THINK2 = "A divide helper needs a zero guard before the division itself."
ANSWER2 = "Added divide(a, b) with a ZeroDivisionError guard."
TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "read",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {}},
    },
}


def _encode(tok, messages, *, allow=False, preserve=False, tool_prompt_mode="compact"):
    request = oa.ChatCompletionRequest(model="m", messages=messages)
    return oa._encode_messages(
        tok,
        request.messages,
        enable_thinking=True,
        reasoning_effort="medium",
        strip_assistant_reasoning_history=False,
        scoped_reasoning_history=False,
        preserve_reasoning_history=preserve,
        tools=[TOOL_SPEC],
        tool_choice=None,
        tool_prompt_mode=tool_prompt_mode,
        template_observability={},
        allow_committed_reasoning=allow,
    )


def _postcommit_state(tok, tool_prompt_mode="compact"):
    return SimpleNamespace(
        args=SimpleNamespace(
            strip_assistant_reasoning_history=False,
            tool_prompt_mode=tool_prompt_mode,
        ),
        runtime=SimpleNamespace(tokenizer=tok),
    )


def _history_ids(
    tok,
    monkeypatch,
    committed_stream_ids,
    *,
    messages,
    assistant_content,
    session_committed_ids=None,
    tool_prompt_mode="compact",
):
    monkeypatch.setattr(oa, "_reasoning_history_scoped_active", lambda state: False)
    monkeypatch.setattr(
        oa,
        "_reasoning_effort_for_state",
        lambda state, thinking_enabled, request_effort=None, **kw: "medium",
    )
    history_ids, _splice = oa._history_ids_for_postcommit(
        _postcommit_state(tok, tool_prompt_mode),
        messages=oa.ChatCompletionRequest(model="m", messages=messages).messages,
        assistant_content=assistant_content,
        assistant_tool_calls=None,
        thinking_enabled=True,
        reasoning_effort="medium",
        tool_specs=[TOOL_SPEC],
        tool_prompt_mode=tool_prompt_mode,
        committed_stream_ids=committed_stream_ids,
        session_committed_ids=session_committed_ids,
    )
    return history_ids


def _committed_turn1(tok, *, tool_prompt_mode="compact"):
    r1_ids = _encode(tok, [SYSTEM, U1], tool_prompt_mode=tool_prompt_mode)
    generated = oa._encode_rendered_chat_text(
        tok, f"{THINK}\n</think>\n\n{ANSWER}<|im_end|>\n"
    )
    session = EngineSession("seam-e2e")
    commit = session.commit(
        prompt_ids=r1_ids, generated_ids=generated, finish_reason="stop"
    )
    assert commit.committed, commit
    return session, r1_ids, generated


def test_raw_encode_matches_canonical_encode_identity(tok):
    """One segmentation policy: with no committed-reasoning fields planted the
    canonical encode renders the identical text, so raw and canonical ids must
    be IDENTICAL. Pre-fix, compact mode plain-tokenized the raw path and the
    '<think>\\n' seam merged into a different token at every assistant turn."""
    history = [SYSTEM, U1, {"role": "assistant", "content": ANSWER}, U2]
    for mode in ("compact", "hybrid"):
        raw = _encode(tok, history, allow=False, tool_prompt_mode=mode)
        canon = _encode(tok, history, allow=True, tool_prompt_mode=mode)
        assert raw == canon, (
            f"raw/canonical encode diverged under {mode}: first mismatch at "
            f"{oa.common_prefix_len(raw, canon)} of {len(raw)}/{len(canon)}"
        )


def test_postcommit_byte_extension_compact_lane(tok, monkeypatch):
    """The founder-lane shape: tool_prompt_mode=compact (no native template
    tools). The retokenized postcommit must byte-extend the committed stream
    and the banked prefix must be a byte prefix of the next turn's canonical
    prompt — pre-fix the compact postcommit plain-tokenized and the #269
    signature came back on this exact lane."""
    session, _r1_ids, _generated = _committed_turn1(tok)

    history_ids = _history_ids(
        tok,
        monkeypatch,
        list(session.committed_token_ids),
        messages=[SYSTEM, U1],
        assistant_content=ANSWER,
    )
    assert history_ids
    commit2 = session.commit_retokenized_prefix(token_ids=history_ids)
    assert commit2.reason not in (
        "retokenized_prefix_not_extending_session",
        "retokenized_prefix_older_than_session",
    ), f"the #269 signature is back on the compact lane: {commit2}"

    committed_now = tuple(session.committed_token_ids)
    history2 = [SYSTEM, U1, {"role": "assistant", "content": ANSWER}, U2]
    raw2_ids = _encode(tok, history2)
    sessions = SimpleNamespace(
        resolve_session_id=lambda **kw: ("seam-e2e", "header.x-mtplx-session-id"),
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
        tools=[TOOL_SPEC],
        tool_choice=None,
        tool_prompt_mode="compact",
        template_observability={},
        session_id="seam-e2e",
    )
    assert result is not None, "canonicalization must apply on the compact lane"
    _canon_messages, canon2_ids = result
    cp_canon = oa.common_prefix_len(canon2_ids, committed_now)
    assert cp_canon == len(committed_now), (
        f"turn-2 canonical prompt must contain the committed stream fully: "
        f"cp_canon={cp_canon} committed={len(committed_now)}"
    )
    assert canon2_ids[: len(history_ids)] == [int(t) for t in history_ids], (
        "the banked compact postcommit prefix must be a byte prefix of the "
        "next turn's canonical prompt"
    )


def test_postcommit_backfills_empty_interiors_from_session_stream(tok, monkeypatch):
    """A canon-miss request renders history think interiors EMPTY; its
    request-local stream must not regress the committed session. The session's
    own committed stream (KV-true) backfills those ordinals, so the published
    postcommit keeps every recovered interior (kills real->empty->real)."""
    session, _r1_ids, _generated = _committed_turn1(tok)

    # Request-local stream of a canon-miss turn 2: A1 rendered with an EMPTY
    # think scaffold, plus this request's generated A2 with a real think.
    canon_miss_prompt = _encode(
        tok, [SYSTEM, U1, {"role": "assistant", "content": ANSWER}, U2]
    )
    generated2 = oa._encode_rendered_chat_text(
        tok, f"{THINK2}\n</think>\n\n{ANSWER2}<|im_end|>\n"
    )
    request_local = list(canon_miss_prompt) + list(generated2)
    messages2 = [SYSTEM, U1, {"role": "assistant", "content": ANSWER}, U2]

    without_backfill = _history_ids(
        tok,
        monkeypatch,
        request_local,
        messages=messages2,
        assistant_content=ANSWER2,
    )
    with_backfill = _history_ids(
        tok,
        monkeypatch,
        request_local,
        messages=messages2,
        assistant_content=ANSWER2,
        session_committed_ids=list(session.committed_token_ids),
    )
    assert with_backfill and without_backfill
    assert THINK not in tok.decode(list(without_backfill)), (
        "precondition lost: the request-local stream alone should render "
        "A1's think empty (the oscillation)"
    )
    rendered = tok.decode(list(with_backfill))
    assert THINK in rendered, "A1's interior must be backfilled from the session"
    assert THINK2 in rendered, "the request's own generated interior must survive"


ECHO1 = "ECHO_THINK_ONE the summary came from reading calc directly"
U3 = {"role": "user", "content": "Now add a modulo helper too."}


def test_raw_encode_carries_preserve_echo_and_matches_canonical(tok):
    """Preserve echo-carry, the spiral killer: the raw encode renders the
    client's echoed reasoning for history turns (an uncovered turn is no
    longer an empty scaffold), and the canonical encode of the identical
    messages produces IDENTICAL ids - one segmentation policy holds with the
    echo present."""
    history = [
        SYSTEM,
        U1,
        {"role": "assistant", "content": ANSWER, "reasoning_content": ECHO1},
        U2,
    ]
    raw = _encode(tok, history, preserve=True)
    assert ECHO1 in tok.decode(raw), "echoed reasoning must reach the render"
    dropped = _encode(tok, history)
    assert ECHO1 not in tok.decode(dropped), (
        "flag-off must stay the legacy drop (rollback lane)"
    )
    canon = _encode(tok, history, allow=True, preserve=True)
    assert raw == canon, (
        f"raw/canonical diverged with echo present: first mismatch at "
        f"{oa.common_prefix_len(raw, canon)} of {len(raw)}/{len(canon)}"
    )


def test_postcommit_prefix_carries_echo_and_prefixes_next_turn(tok, monkeypatch):
    """The postcommit prediction must render echoed turns exactly like the
    next request's encode, or the banked prefix dies at the first echoed
    turn. Also pins the state-level wiring: the stub state resolves
    echo-carry ON (preserve mode, auto policy)."""
    messages2 = [
        SYSTEM,
        U1,
        {"role": "assistant", "content": ANSWER, "reasoning_content": ECHO1},
        U2,
    ]
    served_prompt = _encode(tok, messages2, preserve=True)
    generated2 = oa._encode_rendered_chat_text(
        tok, f"{THINK2}\n</think>\n\n{ANSWER2}<|im_end|>\n"
    )
    request_local = list(served_prompt) + list(generated2)
    history_ids = _history_ids(
        tok,
        monkeypatch,
        request_local,
        messages=messages2,
        assistant_content=ANSWER2,
    )
    assert history_ids
    rendered = tok.decode(list(history_ids))
    assert ECHO1 in rendered, "postcommit render must keep the echoed turn"
    assert THINK2 in rendered, "the request's own generated interior must survive"

    history3 = messages2 + [
        {"role": "assistant", "content": ANSWER2, "reasoning_content": THINK2},
        U3,
    ]
    raw3 = _encode(tok, history3, preserve=True)
    assert raw3[: len(history_ids)] == [int(t) for t in history_ids], (
        "the postcommit prefix must be a byte prefix of the next turn's "
        "echo-carrying encode"
    )
