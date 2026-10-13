"""Stop, then send again: the conversation keeps its session (2.11.4).

Receipt, founder's native app chat on Flash-Next, 2026-09-20, request log
rows by structure:

1. Turn 1: prompt 90 tokens, 49,879 generated, session ``9040A954``.
2. Turn 2 (a follow-up): ``committed_reasoning_canonicalization`` =
   ``{applied, cp_raw: 90, committed_len: 49968, turns_substituted: 1,
   cp_canon: 49968}``, prompt 50,045 tokens, same session. Stop pressed after
   858 streamed tokens (``cancellation_reason: client_disconnected``).
3. Turn 3 (the follow-up sent again; five messages, because the app keeps
   the interrupted answer as a turn): prompt 26,294 tokens, NO
   canonicalization record, ``ssd_prefix_miss``, 24.5 s to the first token,
   and ``x-mtplx-session-id: 29FE770B``: a different session.

The server did nothing wrong with the cancelled turn. ``EngineSession.commit``
refuses a cancelled finish and the stream's teardown sets ``commit = False``,
so the session stayed at turn 1's committed stream, findable under the
conversation's id. The app asked under a NEW id: it replaced the conversation's
session id with a fresh UUID after every Stop. ``sessions.peek(new_id)`` is
``None``, which is the canonicalizer's silent "first turn" exit, so turn 1's
reasoning was not put back, the prompt no longer extended anything the bank
held, and the whole history was prefilled again.

The fix has two halves. The app keeps the conversation's session id across a
Stop (Swift, ``ChatViewModel.cancelTurn``; its test is
``StopKeepsSessionTests``). That makes a second thing reachable: the follow-up
can now arrive while the cancelled generation still holds the session (the
engine notices a cancel at its next decode round or prefill chunk boundary),
and a named session that is busy used to be refused with "already in flight".
A holder that has already been cancelled is now waited for instead.

Everything here runs on the fake streaming session state: no model, CPU only.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from threading import Event

import pytest

pytest.importorskip("fastapi")
from fastapi.responses import Response
from fastapi.testclient import TestClient

from mtplx.engine_session import (
    EngineSession,
    EngineSessionBusy,
    EngineSessionManager,
)
from mtplx.server import openai
from mtplx.server.openai import create_app

from test_server_openai import (  # noqa: E402 - shared fixtures
    _fake_final_state,
    _fake_streaming_session_state,
    _stream_payloads,
)

CONVERSATION = "9040A954-conversation"
ROTATED = "29FE770B-rotated-after-stop"
HEADER_SOURCE = "header.x-mtplx-session-id"

U1 = "Build a small flappy bird game."
THINK1 = "The user wants a game. Plan the loop, the pipes and the scoring first."
ANSWER1 = "Here is the game: one canvas, one loop, pipes that scroll left."
U2 = "Make the gap between the pipes a little wider."
THINK2 = "Widen the gap constant."
PARTIAL2 = "Sure, the gap is set by"


class ChatMLThinkingTokenizer:
    """One token per character, rendered the way the Qwen ChatML template
    renders a thinking conversation: every assistant turn carries a think
    block (empty when the client sent no reasoning), and the generation
    prompt ends with the open ``<think>`` scaffold."""

    chat_template = "fake-chatml-thinking"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=True,
        **_kwargs,
    ):
        parts: list[str] = []
        for message in messages:
            role = message["role"]
            content = message.get("content") or ""
            if role == "assistant":
                reasoning = str(message.get("reasoning_content") or "").strip()
                parts.append(
                    "<|im_start|>assistant\n<think>\n"
                    + reasoning
                    + "\n</think>\n\n"
                    + content.strip()
                    + "<|im_end|>\n"
                )
            else:
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n<think>\n")
        text = "".join(parts)
        return [ord(char) for char in text] if tokenize else text

    def encode(self, text, **_kwargs):
        return [ord(char) for char in str(text)]

    def decode(self, tokens, **_kwargs):
        return "".join(chr(int(token)) for token in tokens)


def _text(ids) -> str:
    return "".join(chr(int(token)) for token in ids)


def _ids(text: str) -> list[int]:
    return [ord(char) for char in text]


def _state():
    state = _fake_streaming_session_state()
    state.runtime.tokenizer = ChatMLThinkingTokenizer()
    state.args.stream_interval = 1
    return state


def _messages(*turns):
    roles = ("user", "assistant")
    return [
        {"role": roles[index % 2], "content": content}
        for index, content in enumerate(turns)
    ]


TURN2_MESSAGES = _messages(U1, ANSWER1, U2)
# What the app really sends after a Stop: the interrupted answer is kept as
# a turn of its own (visible text only), then the follow-up again.
TURN3_MESSAGES = _messages(U1, ANSWER1, U2, PARTIAL2, U2)


class _Engine:
    """Scripted stand-in for ``_run_generation``: one entry per turn. It
    keeps what each turn was handed: the prompt, and the two receipts the
    request carried by the time it reached the engine."""

    def __init__(self):
        self.prompts: list[list[int]] = []
        self.canon: list[dict | None] = []
        self.handoffs: list[dict | None] = []
        self.release_cancelled = Event()
        self.cancelled_running = Event()
        self.cancelled_unwound = Event()

    def __call__(self, _state, prompt_ids, **kwargs):
        observability = kwargs.get("request_observability") or {}
        self.prompts.append([int(token) for token in prompt_ids])
        self.canon.append(
            observability.get("committed_reasoning_canonicalization") or None
        )
        self.handoffs.append(
            observability.get("request_session_cancel_handoff") or None
        )
        turn = len(self.prompts)
        token_callback = kwargs.get("token_callback")
        if turn == 2:
            # The turn the user stops: some tokens reach the client, then
            # the engine is still busy (mid decode round, mid prefill
            # chunk) until the test lets it reach its next cancel check.
            if token_callback is not None:
                token_callback(_ids(THINK2 + "\n</think>\n\n" + PARTIAL2))
            self.cancelled_running.set()
            self.release_cancelled.wait(20.0)
            self.cancelled_unwound.set()
            raise openai._StreamCancelled("stream client disconnected")
        think, answer = (THINK1, ANSWER1) if turn == 1 else ("Again.", "Done.")
        tokens = _ids(f"{think}\n</think>\n\n{answer}")
        if token_callback is not None:
            token_callback(tokens)
        return {
            "text": _text(tokens),
            "tokens": tokens,
            "stats": {
                **(kwargs.get("request_observability") or {}),
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": len(tokens),
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(tokens),
            "finish_reason": "stop",
            "_final_state": _fake_final_state(tokens),
        }


def _install(monkeypatch, engine: _Engine) -> None:
    monkeypatch.setattr(openai, "_run_generation", engine)
    monkeypatch.setattr(
        openai,
        "_store_generation_final_history_snapshot",
        lambda *_args, **_kwargs: {
            "stored": True,
            "mode": "generation_final_exact",
            "reason": "compatible",
            "nbytes": 123,
        },
    )


def _chat(client, session_id: str, messages) -> str:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"x-mtplx-session-id": session_id},
        json={
            "messages": messages,
            "enable_thinking": True,
            "stream": True,
            "max_tokens": 256,
        },
    ) as response:
        assert response.status_code == 200
        return response.read().decode()


async def _never_disconnected(self) -> bool:
    return False


def _stop_mid_stream(monkeypatch, client, engine: _Engine) -> None:
    """Turn 2, stopped the way the app stops it: the transport goes away
    while tokens are streaming (Starlette closes the stream generator)."""

    captured: dict[str, object] = {}
    with monkeypatch.context() as patch:
        patch.setattr(
            openai,
            "StreamingResponse",
            lambda content, **_kwargs: captured.update(generator=content)
            or Response("captured"),
        )
        # The captured generator runs outside a live transport; the
        # disconnect is delivered as the generator close below.
        patch.setattr(
            "starlette.requests.Request.is_disconnected", _never_disconnected
        )
        response = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-session-id": CONVERSATION},
            json={
                "messages": TURN2_MESSAGES,
                "enable_thinking": True,
                "stream": True,
                "max_tokens": 256,
            },
        )
        assert response.status_code == 200 and response.text == "captured"
        generator = captured["generator"]

        async def drive() -> None:
            # The role frame precedes the stream's guarded block; every
            # later frame comes from the token queue inside it. Close only
            # once the engine has streamed tokens and sits mid-generation,
            # so the close lands where a real Stop lands.
            await asyncio.wait_for(generator.__anext__(), 10.0)
            await asyncio.wait_for(generator.__anext__(), 10.0)
            while not engine.cancelled_running.is_set():
                await asyncio.sleep(0.01)
            await generator.aclose()

        asyncio.run(drive())
    assert engine.cancelled_running.is_set()


def _three_turns(monkeypatch, *, resend_session: str, release_before_resend: bool):
    state = _state()
    engine = _Engine()
    _install(monkeypatch, engine)
    with TestClient(create_app(state)) as client:
        _chat(client, CONVERSATION, _messages(U1))
        # A named session's stream ends at its last token and commits after
        # it (aed64295): read the session once that commit landed, as the
        # next request of the session does.
        state.sessions.peek(CONVERSATION).wait_for_response_tail(10.0)
        committed = tuple(state.sessions.peek(CONVERSATION).committed_token_ids)
        _stop_mid_stream(monkeypatch, client, engine)
        if release_before_resend:
            engine.release_cancelled.set()
            assert engine.cancelled_unwound.wait(10.0)
            deadline = time.monotonic() + 10.0
            while state.sessions.peek(CONVERSATION).in_flight:
                assert time.monotonic() < deadline
                time.sleep(0.01)
        else:
            # The engine reaches its next cancel check only AFTER the
            # follow-up has arrived and found the session held: the release
            # is keyed on the follow-up entering the wait, not on a clock.
            waiting = _observe_handoff_wait(monkeypatch)

            def release_once_waiting() -> None:
                if waiting.wait(10.0):
                    time.sleep(0.2)
                engine.release_cancelled.set()

            threading.Thread(target=release_once_waiting, daemon=True).start()
        try:
            body = _chat(client, resend_session, TURN3_MESSAGES)
        finally:
            engine.release_cancelled.set()
    return state, engine, committed, body


def _observe_handoff_wait(monkeypatch) -> Event:
    """An event set the moment a request starts waiting for a cancelled
    holder; the real wait still runs."""
    waiting = Event()
    original = EngineSession.wait_for_cancelled_holder

    def observed(self, **kwargs):
        waiting.set()
        return original(self, **kwargs)

    monkeypatch.setattr(EngineSession, "wait_for_cancelled_holder", observed)
    return waiting


# --- The server half of the receipt ----------------------------------------


def test_a_stopped_turn_leaves_the_session_at_its_last_finished_turn(monkeypatch):
    state, engine, committed, _body = _three_turns(
        monkeypatch, resend_session=CONVERSATION, release_before_resend=True
    )
    session = state.sessions.peek(CONVERSATION)
    # Turn 1 committed prompt + generation; the stopped turn 2 changed
    # nothing (no partial output is ever a finished assistant turn) until
    # turn 3 finished and committed in its turn.
    assert _text(committed).endswith(f"{THINK1}\n</think>\n\n{ANSWER1}")
    assert PARTIAL2 not in _text(committed)
    assert tuple(session.committed_token_ids[: len(committed)]) == committed
    assert session.last_finish_reason == "stop"
    # Turn 2 was the healthy shape of the receipt: turn 1's reasoning put
    # back, the prompt extending the committed stream to its last token.
    turn2 = engine.prompts[1]
    assert tuple(turn2[: len(committed)]) == committed
    assert engine.canon[0] is None, "a first turn has nothing to canonicalize"
    assert engine.canon[1] == {
        "applied": True,
        "cp_raw": len(engine.prompts[0]),
        "committed_len": len(committed),
        "turns_substituted": 1,
        "cp_canon": len(committed),
    }
    # The Stop is on the record as the client's, and nothing was waited for.
    cancelled = [row for row in state.last_metrics if row.get("request_cancelled")]
    assert cancelled and cancelled[-1]["cancellation_reason"] == "client_disconnected"
    assert engine.handoffs == [None, None, None]


def test_the_follow_up_after_a_stop_is_canonicalized_under_the_same_session(
    monkeypatch,
):
    _server_state, engine, committed, body = _three_turns(
        monkeypatch, resend_session=CONVERSATION, release_before_resend=True
    )
    assert "data: [DONE]" in body and "already in flight" not in body
    turn2, turn3 = engine.prompts[1], engine.prompts[2]
    record = engine.canon[2]
    assert record is not None, "the follow-up must leave a canonicalization record"
    assert record["applied"] is True
    assert record["turns_substituted"] == 1
    assert record["cp_canon"] == record["committed_len"] == len(committed)
    # The prompt the engine gets extends the committed stream, so the bank
    # restores turn 1 instead of prefilling it again ...
    assert tuple(turn3[: len(committed)]) == committed
    assert f"<think>\n{THINK1}\n</think>" in _text(turn3)
    # ... and it extends the stopped turn's own prompt too, so the prompt
    # prefix the engine banks once a prefill completes can serve it.
    assert turn3[: len(turn2)] == turn2
    # The interrupted answer is history the client sent, nothing more: it is
    # rendered as sent, with no reasoning invented for it.
    assert _text(turn3[len(turn2) :]).startswith(f"\n</think>\n\n{PARTIAL2}<|im_end|>")
    assert THINK2 not in _text(turn3)


def test_a_renamed_session_is_the_silent_exit_of_the_receipt(monkeypatch):
    """What the app did until 2.11.4, reproduced: the follow-up under a fresh
    session id. The session store still holds the conversation, but nobody
    asks for it, and the gate takes its ``peek(...) is None`` exit: no
    record, no reasoning put back, a prompt that extends nothing."""

    state, engine, committed, body = _three_turns(
        monkeypatch, resend_session=ROTATED, release_before_resend=True
    )
    assert "data: [DONE]" in body
    # The conversation's session is intact and findable; it was not asked.
    assert state.sessions.peek(CONVERSATION).committed_token_ids == committed
    turn3 = engine.prompts[2]
    assert engine.canon[2] is None, "the silent exit leaves no record at all"
    # The engine got the client's messages as sent, turn 1's reasoning gone.
    assert turn3 == ChatMLThinkingTokenizer().apply_chat_template(
        TURN3_MESSAGES, tokenize=True, add_generation_prompt=True
    )
    assert THINK1 not in _text(turn3)
    # That prompt leaves the committed stream where turn 1's reasoning began
    # (the receipt's cp_raw: 90 of 49,968); all the rest is a prefill.
    assert openai.common_prefix_len(turn3, committed) == len(engine.prompts[0])


def test_the_follow_up_waits_for_the_stopped_generation_to_let_go(monkeypatch):
    """Same session, and the follow-up arrives while the cancelled generation
    is still inside the engine. It used to be refused ("already in flight")."""

    state, engine, committed, body = _three_turns(
        monkeypatch, resend_session=CONVERSATION, release_before_resend=False
    )
    assert "already in flight" not in body
    assert "data: [DONE]" in body
    finishes = [
        payload["choices"][0].get("finish_reason")
        for payload in _stream_payloads(body)
        if payload.get("choices")
    ]
    assert finishes[-1] == "stop"
    assert engine.cancelled_unwound.is_set()
    turn3 = engine.prompts[2]
    assert tuple(turn3[: len(committed)]) == committed
    assert engine.canon[2]["applied"] is True
    # The receipt rides the request observability the engine is handed,
    # which the real engine folds into the request-log row as it is
    # (envelope.update(request_observability)); the fake engine above
    # returns it as stats, so the client's final frame shows it is on the
    # public stats allowlist too.
    handoff = engine.handoffs[2]
    assert handoff is not None, "the follow-up must carry the handoff receipt"
    assert handoff["outcome"] == "handed_off" and handoff["acquired"] is True
    assert 0.15 <= handoff["waited_s"] < 10.0
    assert handoff["timeout_s"] == 30.0
    assert engine.handoffs[:2] == [None, None]
    final = [
        payload
        for payload in _stream_payloads(body)
        if payload.get("choices") and payload["choices"][0].get("finish_reason")
    ][-1]
    assert final["mtplx_stats"]["request_session_cancel_handoff"] == handoff
    assert state.sessions.peek(CONVERSATION).last_cancel_handoff["outcome"] == "handed_off"


# --- The session slot itself -------------------------------------------------


def _held(manager: EngineSessionManager, *, cancel_event):
    session = manager.get_or_create("named")
    assert session.try_begin_generation(cancel_event=cancel_event)
    return session


def test_a_cancelled_holder_hands_a_named_session_over():
    manager = EngineSessionManager()
    holder_cancel = Event()
    session = _held(manager, cancel_event=holder_cancel)
    holder_cancel.set()
    threading.Timer(0.15, session.end_generation).start()
    receipt: dict = {}
    started = time.monotonic()
    with manager.generation_slot(
        session, source=HEADER_SOURCE, cancel_event=Event(), handoff_out=receipt
    ) as acquired:
        assert acquired is session and session.in_flight
        assert not session.holder_cancel_requested(), "the slot is the new request's"
    assert not session.in_flight
    assert time.monotonic() - started >= 0.1
    assert receipt["outcome"] == "handed_off" and receipt["acquired"] is True
    assert session.to_admin_dict()["last_cancel_handoff"]["outcome"] == "handed_off"


def test_a_holder_that_was_not_cancelled_is_still_a_collision():
    manager = EngineSessionManager()
    session = _held(manager, cancel_event=Event())
    receipt: dict = {}
    started = time.monotonic()
    try:
        with pytest.raises(EngineSessionBusy, match="already in flight"):
            with manager.generation_slot(
                session, source=HEADER_SOURCE, cancel_event=Event(), handoff_out=receipt
            ):
                pass
    finally:
        session.end_generation()
    assert time.monotonic() - started < 1.0, "no wait for a generation that is wanted"
    assert receipt == {}
    assert session.last_cancel_handoff is None


def test_a_cancelled_holder_that_never_stops_is_refused_at_the_bound(monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_CANCEL_HANDOFF_WAIT_S", "0.2")
    manager = EngineSessionManager()
    holder_cancel = Event()
    session = _held(manager, cancel_event=holder_cancel)
    holder_cancel.set()
    receipt: dict = {}
    started = time.monotonic()
    try:
        with pytest.raises(EngineSessionBusy, match="did not stop within 0.2s"):
            with manager.generation_slot(
                session, source=HEADER_SOURCE, cancel_event=Event(), handoff_out=receipt
            ):
                pass
    finally:
        session.end_generation()
    assert 0.2 <= time.monotonic() - started < 5.0
    assert receipt["outcome"] == "timeout" and receipt["acquired"] is False


def test_the_wait_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_CANCEL_HANDOFF_WAIT_S", "0")
    manager = EngineSessionManager()
    holder_cancel = Event()
    session = _held(manager, cancel_event=holder_cancel)
    holder_cancel.set()
    started = time.monotonic()
    try:
        with pytest.raises(EngineSessionBusy):
            with manager.generation_slot(
                session, source=HEADER_SOURCE, cancel_event=Event()
            ):
                pass
    finally:
        session.end_generation()
    assert time.monotonic() - started < 1.0


def test_a_second_stop_ends_the_wait():
    manager = EngineSessionManager()
    holder_cancel = Event()
    session = _held(manager, cancel_event=holder_cancel)
    holder_cancel.set()
    waiter_cancel = Event()
    threading.Timer(0.1, waiter_cancel.set).start()
    receipt: dict = {}
    started = time.monotonic()
    try:
        with pytest.raises(EngineSessionBusy):
            with manager.generation_slot(
                session,
                source=HEADER_SOURCE,
                cancel_event=waiter_cancel,
                handoff_out=receipt,
            ):
                pass
    finally:
        session.end_generation()
    assert time.monotonic() - started < 5.0
    assert receipt["outcome"] == "waiter_cancelled"


def test_a_third_request_that_took_the_slot_is_not_waited_for():
    manager = EngineSessionManager()
    holder_cancel = Event()
    session = _held(manager, cancel_event=holder_cancel)
    holder_cancel.set()

    def hand_to_a_live_generation() -> None:
        session.end_generation()
        # May lose the slot to the waiter's own poll; either outcome below
        # is correct, and neither may wait out the full bound.
        session.try_begin_generation(cancel_event=Event())

    threading.Timer(0.1, hand_to_a_live_generation).start()
    started = time.monotonic()
    won = False
    try:
        with manager.generation_slot(
            session, source=HEADER_SOURCE, cancel_event=Event()
        ):
            won = True
    except EngineSessionBusy:
        pass
    assert time.monotonic() - started < 5.0
    if not won:
        assert session.in_flight, "the live generation still holds the session"
        session.end_generation()


def test_an_inferred_session_still_forks_instead_of_waiting():
    manager = EngineSessionManager()
    holder_cancel = Event()
    session = _held(manager, cancel_event=holder_cancel)
    holder_cancel.set()
    started = time.monotonic()
    try:
        with manager.generation_slot(
            session, source="longest_prefix", cancel_event=Event()
        ) as acquired:
            assert acquired is not session
            assert acquired.session_id.startswith("anon-")
    finally:
        session.end_generation()
    assert time.monotonic() - started < 1.0


def test_a_cancelled_finish_is_never_a_commit():
    session = EngineSession("named")
    assert session.commit(
        prompt_ids=[1, 2, 3], generated_ids=[4, 5], finish_reason="stop"
    ).committed
    for finish in ("cancelled", "client_disconnected", "abort", "error"):
        refused = session.commit(
            prompt_ids=[1, 2, 3, 4, 5, 6], generated_ids=[7], finish_reason=finish
        )
        assert not refused.committed and refused.reason == f"unsafe_finish:{finish}"
    assert session.committed_token_ids == (1, 2, 3, 4, 5)


# --- The same three turns through the real chat template ---------------------

MODEL_DIR = Path.home() / ".mtplx/models/Qwen3.8-27B-MTPLX-Optimized-Speed"


def _reach(record: dict) -> int:
    """How far into the committed stream the served prompt reaches: the
    canonical encode's own prefix, or the committed-id splice past a token
    seam of the model's own making."""
    return max(int(record.get("cp_canon") or 0), int(record.get("cp_spliced") or 0))


@pytest.mark.skipif(
    not (MODEL_DIR / "chat_template.jinja").exists(),
    reason="Qwen3.8 model pack not cached locally",
)
def test_real_template_follow_up_extends_the_committed_stream_and_the_stopped_prompt(
    monkeypatch,
):
    """Tokenizer and chat template only, no weights. Flash-Next ships the
    same template. The follow-up, canonicalized against the session the
    Stop left behind, must extend turn 1's committed stream AND the stopped
    turn's prompt token for token: those are the two states the bank holds."""

    from types import SimpleNamespace

    from mtplx.runtime import _load_tokenizer_resilient

    config = json.loads((MODEL_DIR / "config.json").read_text())
    tok = _load_tokenizer_resilient(MODEL_DIR, config)
    monkeypatch.setattr(openai, "_reasoning_history_scoped_active", lambda state: False)
    monkeypatch.setattr(
        openai, "_reasoning_history_preserve_echo_active", lambda state: True
    )

    manager = EngineSessionManager()
    state = SimpleNamespace(
        args=SimpleNamespace(strip_assistant_reasoning_history=False),
        sessions=manager,
        runtime=SimpleNamespace(tokenizer=tok, model_path=MODEL_DIR),
    )

    def encode(messages):
        request = openai.ChatCompletionRequest(model="m", messages=messages)
        return request, openai._encode_messages(
            tok,
            request.messages,
            enable_thinking=True,
            reasoning_effort="medium",
            strip_assistant_reasoning_history=False,
            scoped_reasoning_history=False,
            preserve_reasoning_history=True,
            tools=None,
            tool_choice=None,
            template_observability={},
        )

    def canonicalize(messages, session_id):
        request, raw_ids = encode(messages)
        observability: dict = {}
        result = openai._maybe_canonicalize_committed_reasoning(
            state,
            messages=request.messages,
            prompt_ids=list(raw_ids),
            headers={"x-mtplx-session-id": session_id},
            metadata={},
            request=request,
            thinking_enabled=True,
            reasoning_effort="medium",
            tools=None,
            tool_choice=None,
            tool_prompt_mode="hybrid",
            template_observability={},
            request_observability=observability,
            session_id=session_id,
        )
        record = observability.get("committed_reasoning_canonicalization")
        return raw_ids, (result[1] if result is not None else None), record

    _request, prompt1 = encode(_messages(U1))
    generated1 = openai._encode_rendered_chat_text(
        tok, f"{THINK1}\n</think>\n\n{ANSWER1}"
    )
    session = manager.get_or_create(CONVERSATION)
    assert session.commit(
        prompt_ids=prompt1, generated_ids=generated1, finish_reason="stop"
    ).committed
    committed = list(session.committed_token_ids)

    _raw2, turn2, record2 = canonicalize(TURN2_MESSAGES, CONVERSATION)
    assert record2["applied"] is True and _reach(record2) == len(committed)
    assert turn2[: len(committed)] == committed

    # The Stop: nothing is committed, the session is where turn 1 left it.
    assert list(session.committed_token_ids) == committed

    raw3, turn3, record3 = canonicalize(TURN3_MESSAGES, CONVERSATION)
    assert record3["applied"] is True
    assert record3["turns_substituted"] == 1
    assert _reach(record3) == len(committed)
    assert turn3[: len(committed)] == committed
    assert turn3[: len(turn2)] == turn2, (
        "the follow-up must extend the stopped turn's already-prefilled prompt"
    )
    new_tokens = len(turn3) - len(turn2)
    assert 0 < new_tokens < len(raw3) - record3["cp_raw"], (
        "what is left to prefill is the interrupted answer and the follow-up, "
        "not the conversation"
    )

    # Under a renamed session the gate is silent and the prompt is the raw one.
    _raw, renamed, renamed_record = canonicalize(TURN3_MESSAGES, ROTATED)
    assert renamed is None and renamed_record is None
    assert openai.common_prefix_len(raw3, committed) == record3["cp_raw"]
    assert record3["cp_raw"] < len(committed)
