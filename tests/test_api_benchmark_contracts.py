"""API-edge contracts the 2.8 benchmark wave leans on (charlatan defense).

Every test here pins a wire behavior an external benchmark harness consumes
blindly: echo+logprobs array alignment (KL quality lane), over-context 400
vs silent 1-token rows, finish_reason parity across the three dialects,
the completions logprobs gate at its falsy boundaries, client-hint
detection for Claude Code, and repetition-stop visibility. The engine is
always monkeypatched (no model loads); where a test needs the real
``_run_generation`` envelope assembly it fakes the generators instead,
exactly like tests/test_request_observability_golden.py.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.server import openai
from mtplx.server.openai import create_app

from test_server_openai import (  # noqa: E402 - shared fixtures
    CaptureTokenizer,
    ForegroundState,
    _fake_state,
    _fake_streaming_session_state,
    _stream_payloads,
    _tool_schema,
)


# --- shared harness ---------------------------------------------------------


def _scoring_state(tokenizer=None):
    state = _fake_state()
    state.runtime.tokenizer = tokenizer or CaptureTokenizer()
    state.begin_foreground = lambda: None
    state.end_foreground = lambda: None
    state.requests_completed = 0
    state.last_request_at = 0.0
    return state


def _fake_generation_output(**stats_overrides):
    """Stand-in for generate_mtpk/generate_ar on the REAL stats dataclass so
    _run_generation's envelope assembly runs exactly as in production."""
    from mtplx.generation import GenerationStats

    stats = GenerationStats(
        mode="mtpk",
        generated_tokens=2,
        elapsed_s=0.01,
        tok_s=200.0,
        decode_elapsed_s=0.005,
        decode_tok_s=400.0,
        prompt_eval_time_s=0.005,
        prompt_tps=600.0,
        verify_calls=1,
        accepted_by_depth=[1],
        **stats_overrides,
    )

    def _generate(*_args, **_kwargs):
        return SimpleNamespace(
            tokens=[79, 75],
            text="OK",
            stats=stats,
            final_state=None,
            finish_reason="stop",
        )

    return _generate


def _envelope_client(monkeypatch, *, generator=None, prompt_tokens=3):
    """TestClient where the real _run_generation runs over faked generators
    (the golden-matrix harness), so generation_limits and the public stats
    whitelist behave as in production."""
    monkeypatch.delenv("MTPLX_CLIENT", raising=False)
    state = _fake_state()
    foreground = ForegroundState()
    state.lock = foreground.lock
    state.begin_foreground = foreground.begin_foreground
    state.end_foreground = foreground.end_foreground
    state.has_foreground = foreground.has_foreground
    state.foreground_count = foreground.foreground_count
    state.requests_completed = 0
    state.requests_cancelled = 0
    state.last_request_at = 0.0
    state.last_request_started_at = 0.0
    state.active_requests = 0
    state.runtime.tokenizer.encode = lambda _text, **_kwargs: list(
        range(prompt_tokens)
    )
    monkeypatch.setattr(
        openai,
        "_encode_messages",
        lambda *_args, **_kwargs: list(range(prompt_tokens)),
    )
    fake = generator or _fake_generation_output()
    monkeypatch.setattr(openai, "generate_mtpk", fake)
    monkeypatch.setattr(openai, "generate_ar", fake)
    monkeypatch.setattr(openai, "generate_mtp1", fake, raising=False)
    return TestClient(create_app(state)), state


BYPASS = {"x-mtplx-cache-mode": "bypass"}


# --- F4: /v1/completions echo+logprobs KL-lane alignment --------------------


def _fake_score(positions_by_index=None):
    """Engine-shaped scoring result: n-1 positions, position i predicts
    prompt token i+1 (the read-only score_prompt_logprobs contract)."""

    def score(_runtime, prompt_ids, *, top_k):
        n = len(prompt_ids)
        positions = []
        for i in range(n - 1):
            if positions_by_index and i in positions_by_index:
                positions.append(positions_by_index[i])
            else:
                positions.append(
                    [(prompt_ids[i + 1], -0.1), (prompt_ids[0], -2.0)][:top_k or 2]
                )
        return {
            "positions": positions,
            "token_logprobs": [-0.1] * (n - 1),
            "prompt_tokens": n,
            "elapsed_s": 0.01,
        }

    return score


def test_prompt_scoring_arrays_are_openai_aligned(monkeypatch):
    """All four logprobs arrays have length n with nulls at index 0, so a
    harness zipping tokens[i] with token_logprobs[i] reads the logprob OF
    tokens[i] — never the next token's (the bogus-KL headline)."""

    state = _scoring_state()
    monkeypatch.setattr(openai, "score_prompt_logprobs", _fake_score())
    client = TestClient(create_app(state))

    response = client.post(
        "/v1/completions",
        json={
            "prompt": "abcd",
            "echo": True,
            "logprobs": 2,
            "max_tokens": 0,
            "temperature": 0,
        },
    )

    assert response.status_code == 200
    logprobs = response.json()["choices"][0]["logprobs"]
    assert logprobs["tokens"] == ["a", "b", "c", "d"]
    assert len(logprobs["token_logprobs"]) == 4
    assert logprobs["token_logprobs"][0] is None
    assert logprobs["token_logprobs"][1:] == [-0.1, -0.1, -0.1]
    assert len(logprobs["top_logprobs"]) == 4
    # {} at index 0, never null: nothing predicts position 0, but harness
    # KL parsers iterate entries with .items() and a null crashes them.
    assert logprobs["top_logprobs"][0] == {}
    # top_logprobs[i] is the distribution that predicted tokens[i]: the
    # token's own string is a key, mapped to its own logprob.
    for index in (1, 2, 3):
        entry = logprobs["top_logprobs"][index]
        assert isinstance(entry, dict)
        assert entry[logprobs["tokens"][index]] == pytest.approx(-0.1)
    assert logprobs["text_offset"] == [0, 1, 2, 3]
    # Stable identity rides along: string keys collapse (multi-byte pieces
    # all decode to U+FFFD), token ids never do.
    assert logprobs["token_ids"] == [97, 98, 99, 100]


def test_prompt_scoring_actual_token_always_in_its_top_map(monkeypatch):
    """A prompt token ranked below top-K must still appear in its own
    top_logprobs map with its own logprob (OpenAI: 'up to k+1 entries'),
    otherwise measured KL inflates for every ranked-out token."""

    state = _scoring_state()
    # Position 0 predicts token "b" (98) but its top-2 excludes it.
    rigged = {0: [(120, -0.05), (121, -0.9)]}
    score = _fake_score(positions_by_index=rigged)

    def score_with_true_logprob(runtime, prompt_ids, *, top_k):
        result = score(runtime, prompt_ids, top_k=top_k)
        result["token_logprobs"] = [-7.5] + [-0.1] * (len(prompt_ids) - 2)
        return result

    monkeypatch.setattr(openai, "score_prompt_logprobs", score_with_true_logprob)
    client = TestClient(create_app(state))

    response = client.post(
        "/v1/completions",
        json={"prompt": "abcd", "echo": True, "logprobs": 2, "max_tokens": 0},
    )

    assert response.status_code == 200
    logprobs = response.json()["choices"][0]["logprobs"]
    first_map = logprobs["top_logprobs"][1]
    assert first_map["x"] == pytest.approx(-0.05)
    assert first_map["y"] == pytest.approx(-0.9)
    # The actual token was outside top-K yet is present with its own value.
    assert first_map["b"] == pytest.approx(-7.5)
    assert logprobs["token_logprobs"][1] == pytest.approx(-7.5)


class _SpacedJoinTokenizer:
    """Tokenizer whose batch decode diverges from per-token decode joins
    (SentencePiece-style spacing): slicing the echoed text by text_offset is
    only exact when text and offsets come from the same decomposition."""

    pieces = {200: "héllo", 201: "wörld", 202: "!"}

    def encode(self, _text, **_kwargs):
        return [200, 201, 202]

    def decode(self, tokens, **_kwargs):
        parts = [self.pieces[int(token)] for token in tokens]
        return " ".join(parts) if len(parts) > 1 else parts[0]


def test_prompt_scoring_text_offsets_slice_the_returned_text(monkeypatch):
    state = _scoring_state(tokenizer=_SpacedJoinTokenizer())
    monkeypatch.setattr(openai, "score_prompt_logprobs", _fake_score())
    client = TestClient(create_app(state))

    response = client.post(
        "/v1/completions",
        json={"prompt": "héllo wörld!", "echo": True, "logprobs": 1, "max_tokens": 0},
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    logprobs = choice["logprobs"]
    text = choice["text"]
    offsets = logprobs["text_offset"]
    tokens = logprobs["tokens"]
    assert len(offsets) == len(tokens) == 3
    for index, token_text in enumerate(tokens):
        start = offsets[index]
        assert text[start : start + len(token_text)] == token_text


# --- F5: over-context 400 and visible max_tokens clamp ----------------------


def test_completions_prompt_over_context_is_400_not_one_token(monkeypatch):
    client, state = _envelope_client(monkeypatch, prompt_tokens=6000)
    assert state.context_window == 4096

    response = client.post(
        "/v1/completions",
        headers=BYPASS,
        json={"prompt": "x" * 6000, "max_tokens": 128},
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "context_length_exceeded"
    assert "4096" in error["message"]
    assert "6000" in error["message"]


def test_chat_prompt_over_context_is_400_not_one_token(monkeypatch):
    client, _state = _envelope_client(monkeypatch, prompt_tokens=6000)

    response = client.post(
        "/v1/chat/completions",
        headers=BYPASS,
        json={
            "messages": [{"role": "user", "content": "long"}],
            "max_tokens": 128,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_length_exceeded"


def test_streaming_over_context_is_a_real_400_before_the_stream(monkeypatch):
    client, _state = _envelope_client(monkeypatch, prompt_tokens=6000)

    chat = client.post(
        "/v1/chat/completions",
        headers=BYPASS,
        json={
            "messages": [{"role": "user", "content": "long"}],
            "stream": True,
        },
    )
    completions = client.post(
        "/v1/completions",
        headers=BYPASS,
        json={"prompt": "y" * 6000, "stream": True},
    )

    assert chat.status_code == 400
    assert completions.status_code == 400
    assert chat.json()["error"]["code"] == "context_length_exceeded"
    assert completions.json()["error"]["code"] == "context_length_exceeded"


def test_messages_prompt_over_context_is_400(monkeypatch):
    client, _state = _envelope_client(monkeypatch, prompt_tokens=6000)

    response = client.post(
        "/v1/messages",
        headers=BYPASS,
        json={
            "model": "default",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "long"}],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_length_exceeded"


def test_exactly_full_context_prompt_is_400(monkeypatch):
    client, _state = _envelope_client(monkeypatch, prompt_tokens=4096)

    response = client.post(
        "/v1/completions",
        headers=BYPASS,
        json={"prompt": "z" * 4096, "max_tokens": 1},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_length_exceeded"


def test_max_tokens_clamped_to_remaining_context_is_visible(monkeypatch, caplog):
    client, _state = _envelope_client(monkeypatch, prompt_tokens=4000)

    with caplog.at_level(logging.WARNING, logger="mtplx.server.openai"):
        response = client.post(
            "/v1/completions",
            headers=BYPASS,
            json={"prompt": "w" * 4000, "max_tokens": 50_000},
        )

    assert response.status_code == 200
    stats = response.json()["mtplx_stats"]
    # Existing stats idiom for the clamp: the flag plus the effective value.
    assert stats["context_cap_applied"] is True
    assert stats["effective_max_tokens"] == 96  # 4096 - 4000
    assert stats["remaining_context_tokens"] == 96
    assert any(
        "max_tokens clamped to remaining context" in record.message
        for record in caplog.records
    ), "clamp must produce one server log line"


def test_fitting_request_reports_no_context_clamp(monkeypatch):
    client, _state = _envelope_client(monkeypatch, prompt_tokens=8)

    response = client.post(
        "/v1/completions",
        headers=BYPASS,
        json={"prompt": "short", "max_tokens": 16},
    )

    assert response.status_code == 200
    stats = response.json()["mtplx_stats"]
    assert stats["context_cap_applied"] is False
    assert stats["effective_max_tokens"] == 16


# --- F28a: non-stream chat finish_reason cap-hit wins over tool_calls -------


def _single_call_extraction(*_args, **_kwargs):
    return SimpleNamespace(
        cleaned_text="",
        cleaned_thinking="",
        tool_calls=[
            {
                "id": "call_status",
                "type": "function",
                "function": {"name": "session_status", "arguments": "{}"},
            }
        ],
        parser_source="native",
        status="parsed",
        malformed_reason=None,
        raw_tool_markup_suppressed=True,
    )


def _nonstream_tool_response(monkeypatch, *, finish_reason: str):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    generated = {
        "text": "x",
        "tokens": [4],
        "stats": {
            "generation_mode": "ar",
            "mtp_depth": 0,
            "completion_tokens": 1,
        },
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "finish_reason": finish_reason,
    }
    monkeypatch.setattr(openai, "_run_generation", lambda *a, **k: dict(generated))
    monkeypatch.setattr(
        openai, "omlx_extract_tool_calls_with_thinking", _single_call_extraction
    )
    client = TestClient(create_app(state))
    return client.post(
        "/v1/chat/completions",
        headers=BYPASS,
        json={
            "messages": [{"role": "user", "content": "status"}],
            "tools": [_tool_schema()],
        },
    )


def test_nonstream_chat_length_cap_beats_tool_calls(monkeypatch):
    """#196/#197 non-stream twin: a length-truncated turn that still parsed
    complete tool calls must report "length" so agent clients continue —
    the stream path already does (openai.py stream twin)."""

    response = _nonstream_tool_response(monkeypatch, finish_reason="length")

    assert response.status_code == 200
    body = response.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "length"
    assert choice["message"]["tool_calls"], "tool calls themselves must survive"
    assert body["mtplx_stats"]["tool_calls_truncated_by_length"] is True


def test_nonstream_chat_natural_stop_with_tools_reports_tool_calls(monkeypatch):
    response = _nonstream_tool_response(monkeypatch, finish_reason="stop")

    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "tool_calls"


# --- F28b: /v1/messages stop_reason priority --------------------------------


def test_anthropic_stop_reason_priority_unit():
    stop_reason = openai._anthropic_stop_reason
    # Budget cut wins even when tool_use blocks are present (Anthropic wire
    # semantics: a truncated turn is max_tokens, whatever it contains).
    assert stop_reason("length", has_tool_calls=True) == "max_tokens"
    assert stop_reason("length", has_tool_calls=False) == "max_tokens"
    # Client stop_sequences match outranks tool_use (QA-117 stays intact).
    assert (
        stop_reason("stop", has_tool_calls=True, matched_stop="STOP")
        == "stop_sequence"
    )
    assert stop_reason("stop", has_tool_calls=True) == "tool_use"
    assert stop_reason("tool_calls", has_tool_calls=False) == "tool_use"
    assert stop_reason("stop", has_tool_calls=False) == "end_turn"


def test_messages_length_with_tool_use_maps_to_max_tokens(monkeypatch):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    generated = {
        "text": "x",
        "tokens": [4],
        "stats": {
            "generation_mode": "ar",
            "mtp_depth": 0,
            "completion_tokens": 1,
        },
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "finish_reason": "length",
    }
    monkeypatch.setattr(openai, "_run_generation", lambda *a, **k: dict(generated))
    monkeypatch.setattr(
        openai, "omlx_extract_tool_calls_with_thinking", _single_call_extraction
    )
    client = TestClient(create_app(state))

    response = client.post(
        "/v1/messages",
        headers=BYPASS,
        json={
            "model": "default",
            "max_tokens": 4,
            "messages": [{"role": "user", "content": "status"}],
            "tools": [
                {
                    "name": "session_status",
                    "description": "status",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert any(block["type"] == "tool_use" for block in body["content"])
    assert body["stop_reason"] == "max_tokens"


# --- F28c: completions stream stop-string parity with non-stream trim -------


def _stream_completion(monkeypatch, fake_run_generation, *, stop):
    state = _fake_streaming_session_state()
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)
    response = client.post(
        "/v1/completions",
        json={"prompt": "go", "max_tokens": 32, "stream": True, "stop": stop},
    )
    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    streamed = "".join(
        payload["choices"][0]["text"]
        for payload in payloads
        if payload["choices"][0].get("text")
    )
    final = [
        payload for payload in payloads if payload["choices"][0].get("finish_reason")
    ][-1]
    return streamed, final


def test_stream_stop_in_final_batch_without_cancellation(monkeypatch):
    """The engine finishes on its own right after emitting the stop string
    (no cancel handshake): the emitted stream must still exclude it."""

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        token_callback = kwargs["token_callback"]
        token_callback([ord(char) for char in "Hello STOP tail\n"])
        text = "Hello STOP tail\n"
        return {
            "text": text,
            "tokens": [ord(char) for char in text],
            "stats": {"generation_mode": "ar", "mtp_depth": 0},
            "prompt_tokens": 2,
            "completion_tokens": len(text),
            "finish_reason": "length",
        }

    streamed, final = _stream_completion(
        monkeypatch, fake_run_generation, stop=["STOP"]
    )

    assert "STOP" not in streamed
    assert streamed == "Hello "
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["mtplx_stats"]["stop_sequence_hit"] is True


def test_stream_stop_split_across_final_batches(monkeypatch):
    def fake_run_generation(_state, _prompt_ids, **kwargs):
        token_callback = kwargs["token_callback"]
        token_callback([ord(char) for char in "Hello ST"])
        token_callback([ord(char) for char in "OP tail\n"])
        text = "Hello STOP tail\n"
        return {
            "text": text,
            "tokens": [ord(char) for char in text],
            "stats": {"generation_mode": "ar", "mtp_depth": 0},
            "prompt_tokens": 2,
            "completion_tokens": len(text),
            "finish_reason": "length",
        }

    streamed, final = _stream_completion(
        monkeypatch, fake_run_generation, stop=["STOP"]
    )

    assert "STOP" not in streamed
    assert streamed == "Hello "
    assert final["choices"][0]["finish_reason"] == "stop"


def test_stream_stop_completed_only_in_engine_final_text(monkeypatch):
    """Parity with the non-stream post-trim net: the callbacks never
    delivered the full stop string but the engine's final text contains it —
    the held tail must not be flushed to the client and the finish must be
    an honest "stop" with the stop stats stamped."""

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        token_callback = kwargs["token_callback"]
        token_callback([ord(char) for char in "Hello ST"])
        text = "Hello STOP tail"
        return {
            "text": text,
            "tokens": [ord(char) for char in text],
            "stats": {"generation_mode": "ar", "mtp_depth": 0},
            "prompt_tokens": 2,
            "completion_tokens": len(text),
            "finish_reason": "length",
        }

    streamed, final = _stream_completion(
        monkeypatch, fake_run_generation, stop=["STOP"]
    )

    assert "STOP" not in streamed
    assert streamed == "Hello "
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["mtplx_stats"]["stop_sequence_hit"] is True
    assert final["mtplx_stats"]["stop_sequence_matched"] == "STOP"


def test_stream_partial_stop_prefix_tail_is_still_released(monkeypatch):
    """A held tail that never completes a stop is legitimate output and must
    be flushed at the end of the stream, exactly as before."""

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        token_callback = kwargs["token_callback"]
        token_callback([ord(char) for char in "value: ST"])
        text = "value: ST"
        return {
            "text": text,
            "tokens": [ord(char) for char in text],
            "stats": {"generation_mode": "ar", "mtp_depth": 0},
            "prompt_tokens": 2,
            "completion_tokens": len(text),
            "finish_reason": "stop",
        }

    streamed, final = _stream_completion(
        monkeypatch, fake_run_generation, stop=["STOP"]
    )

    assert streamed == "value: ST"
    assert final["choices"][0]["finish_reason"] == "stop"


# --- F30: completions logprobs gate at falsy boundaries ---------------------


def test_logprobs_zero_is_a_real_request_not_absent(monkeypatch):
    """OpenAI semantics: logprobs=0 means 'sampled token logprob only'.
    Without echo it is served for max_tokens=1 only, so max_tokens=8 must be
    the loud 400 — never a silent fall-through into generation that returns
    no logprobs at all."""

    def _explode(*_args, **_kwargs):
        raise AssertionError("logprobs=0 must never reach generation silently")

    monkeypatch.setattr(openai, "_run_generation_dispatched", _explode)
    client = TestClient(create_app(_scoring_state()))

    response = client.post(
        "/v1/completions",
        json={"prompt": "hi", "logprobs": 0, "max_tokens": 8},
    )

    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert "echo=true" in message
    assert "max_tokens" in message


def test_echo_logprobs_zero_runs_prompt_scoring(monkeypatch):
    """echo + logprobs=0 + max_tokens=0 is prompt scoring with no top-K
    alternatives: token_logprobs still aligned, each top map carrying only
    the actual token. Previously this fell through into normal generation
    and produced one sampled token."""

    state = _scoring_state()
    monkeypatch.setattr(openai, "score_prompt_logprobs", _fake_score())
    client = TestClient(create_app(state))

    response = client.post(
        "/v1/completions",
        json={"prompt": "abcd", "echo": True, "logprobs": 0, "max_tokens": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["mtplx_stats"]["mode"] == "prompt_scoring"
    logprobs = body["choices"][0]["logprobs"]
    assert logprobs["token_logprobs"][0] is None
    assert logprobs["token_logprobs"][1:] == [-0.1, -0.1, -0.1]
    for index in (1, 2, 3):
        entry = logprobs["top_logprobs"][index]
        assert entry == {logprobs["tokens"][index]: pytest.approx(-0.1)}


def test_logprobs_none_generates_without_logprobs(monkeypatch):
    client, _state = _envelope_client(monkeypatch, prompt_tokens=3)

    response = client.post(
        "/v1/completions",
        headers=BYPASS,
        json={"prompt": "hi", "max_tokens": 4},
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice.get("logprobs") is None


def test_negative_logprobs_is_rejected(monkeypatch):
    client = TestClient(create_app(_scoring_state()))

    response = client.post(
        "/v1/completions",
        json={"prompt": "hi", "logprobs": -1, "echo": True, "max_tokens": 0},
    )

    assert response.status_code == 400


def test_boolean_logprobs_is_rejected_by_validation():
    client = TestClient(create_app(_scoring_state()))

    response = client.post(
        "/v1/completions",
        json={"prompt": "hi", "logprobs": False, "max_tokens": 4},
    )

    # Pydantic types the field int|None; a boolean is not silently coerced
    # into a logprobs request or into absence.
    assert response.status_code in (400, 422)


def test_echo_logprobs_still_requires_zero_max_tokens():
    client = TestClient(create_app(_scoring_state()))

    response = client.post(
        "/v1/completions",
        json={"prompt": "hi", "echo": True, "logprobs": 2, "max_tokens": 8},
    )

    assert response.status_code == 400
    assert "max_tokens" in response.json()["error"]["message"]


# --- F32: Claude Code client-hint detection ---------------------------------


def test_claude_cli_user_agent_maps_to_claude_code_hint():
    hint = openai._request_client_hint_from_headers(
        {"user-agent": "claude-cli/1.0.44 (external, cli)"},
        {},
    )
    assert hint == "claude_code"


def test_claude_code_hint_is_not_a_managed_surface():
    """Detection is observability only: Claude Code keeps OpenAI-API
    control semantics (not server-owned like the app/browser surfaces)."""

    managed = openai._app_managed_client_hint(
        {"user-agent": "claude-cli/1.0.44 (external, cli)"},
        {},
    )
    assert managed is None


def test_unrelated_user_agents_still_unmatched():
    hint = openai._request_client_hint_from_headers(
        {"user-agent": "python-httpx/0.27"},
        {},
    )
    assert hint is None


# --- F10: repetition-stop visibility in public stats ------------------------


def test_repetition_stop_surfaces_in_public_stats(monkeypatch):
    client, _state = _envelope_client(
        monkeypatch,
        generator=_fake_generation_output(
            repetition_stop_triggered=True,
            repetition_stop_reason="block_repeat",
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        headers=BYPASS,
        json={
            "messages": [{"role": "user", "content": "loop"}],
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    stats = response.json()["mtplx_stats"]
    assert stats["repetition_stop_triggered"] is True
    assert stats["repetition_stop_reason"] == "block_repeat"


def test_quiet_requests_do_not_carry_repetition_keys(monkeypatch):
    """Additive visibility: requests the guard never touched keep their
    envelope byte-stable (golden matrix stays untouched)."""

    client, _state = _envelope_client(monkeypatch)

    response = client.post(
        "/v1/chat/completions",
        headers=BYPASS,
        json={
            "messages": [{"role": "user", "content": "ok"}],
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    stats = response.json()["mtplx_stats"]
    assert "repetition_stop_triggered" not in stats
    assert "repetition_stop_reason" not in stats
