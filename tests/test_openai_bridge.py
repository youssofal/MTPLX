import asyncio
import json
from threading import Event, Lock
from types import SimpleNamespace

import pytest

from mtplx.server.openai import (
    AnthropicMessage,
    AnthropicMessagesRequest,
    ChatMessage,
    STATS_FOOTER_MARKER,
    _RateLimiter,
    _anthropic_content_to_text,
    _anthropic_payload_from_openai,
    _anthropic_stream_from_openai_sse,
    _anthropic_to_chat_request,
    _IncrementalTokenDecoder,
    _StreamCancelled,
    _ThinkingContentStreamSplitter,
    _cancel_stream_generation,
    _encode_messages,
    _effective_completion_tokens,
    _generation_params,
    _generation_final_postcommit_compatibility,
    _normalize_thinking_tags,
    _online_hidden_config,
    _policy_fingerprint,
    _public_mtplx_stats,
    _raise_if_stream_cancelled,
    _repair_streamed_generation_stats,
    _request_is_authorized,
    _schedule_idle_postcommit_snapshot,
    _store_generation_final_history_snapshot,
    _strip_assistant_history_baggage,
    _stream_heartbeat_payload,
    _usage_payload,
    parse_args,
    validate_server_security_args,
)


class TinyTokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(chr(int(token)) for token in tokens)


class ChatTemplateTokenizer(TinyTokenizer):
    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, **_kwargs
    ):
        assert tokenize is True
        text = "\n".join(
            f"{message['role']}:{message.get('content') or ''}" for message in messages
        )
        if add_generation_prompt:
            text = f"{text}\nassistant:" if text else "assistant:"
        return _ids(text)

    def encode(self, text, **_kwargs):
        return _ids(str(text))


def _ids(text: str) -> list[int]:
    return [ord(ch) for ch in text]


class RecordingBank:
    def __init__(self) -> None:
        self.puts: list[dict] = []

    def put(self, **kwargs):
        self.puts.append(kwargs)
        return SimpleNamespace(
            prefix_len=len(kwargs["token_ids"]),
            nbytes=123,
            token_hash="test-token-hash",
        )


def _postcommit_state(*, tokenizer=None):
    args = parse_args(["--warmup-tokens", "0"])
    bank = RecordingBank()
    return SimpleNamespace(
        args=args,
        runtime=SimpleNamespace(
            tokenizer=tokenizer or ChatTemplateTokenizer(),
            model_path="models/test",
            mtp_enabled=True,
        ),
        sessions=SimpleNamespace(bank=bank),
        template_hash="template",
        draft_head_identity="draft-head",
        lock=Lock(),
        generation_executor=SimpleNamespace(
            submit=lambda fn, *args, **kwargs: fn(*args, **kwargs)
        ),
        postcommit_executor=SimpleNamespace(
            submit=lambda fn, *args, **kwargs: fn(*args, **kwargs)
        ),
        has_foreground=lambda: False,
    )


def _final_state(tokens, *, safe=True):
    return SimpleNamespace(
        final_trunk_cache=["cache"],
        final_logits="logits",
        final_hidden="hidden",
        final_committed_mtp_cache=None,
        generated_token_ids=tuple(tokens),
        safe_to_commit=safe,
        finish_reason="stop",
    )


def test_server_parse_args_exposes_product_flags():
    args = parse_args(
        [
            "--host",
            "0.0.0.0",
            "--api-key",
            "test-key",
            "--rate-limit",
            "60",
            "--stream-interval",
            "3",
            "--max-tokens",
            "256",
            "--default-temperature",
            "0.5",
            "--default-top-p",
            "0.8",
            "--reasoning-parser",
            "none",
            "--warmup-tokens",
            "4",
        ]
    )

    assert args.api_key == "test-key"
    assert args.rate_limit == 60
    assert args.stream_interval == 3
    assert args.max_response_tokens == 256
    assert args.temperature == 0.5
    assert args.top_p == 0.8
    assert args.reasoning_parser == "none"
    assert args.warmup_tokens == 4
    assert args.session_postcommit_mode == "async"
    validate_server_security_args(args)

    stock = parse_args(["--stock-ar"])
    assert stock.stock_ar is True
    assert stock.generation_mode == "ar"
    assert stock.load_mtp is False


def test_generation_final_postcommit_exact_stores_final_state_without_retokenized_prefill():
    state = _postcommit_state()
    messages = [ChatMessage(role="user", content="hi")]
    prompt_ids = _encode_messages(
        state.runtime.tokenizer,
        messages,
        enable_thinking=False,
        add_generation_prompt=True,
    )
    generated_tokens = _ids("ok")
    generated = {
        "tokens": generated_tokens,
        "_final_state": _final_state(generated_tokens),
    }

    result = _store_generation_final_history_snapshot(
        state,
        session_id="session-1",
        prompt_ids=prompt_ids,
        generated=generated,
        messages=messages,
        assistant_content="ok",
        thinking_enabled=False,
        policy_fingerprint="policy",
    )

    assert result["stored"] is True
    assert result["mode"] == "generation_final_exact"
    assert result["history_suffix_tokens"] == 0
    assert state.sessions.bank.puts[0]["token_ids"] == prompt_ids + generated_tokens


def test_generation_final_postcommit_prefix_stores_boundary_and_reports_suffix():
    state = _postcommit_state()
    messages = [ChatMessage(role="user", content="hi")]
    prompt_ids = _encode_messages(
        state.runtime.tokenizer,
        messages,
        enable_thinking=False,
        add_generation_prompt=True,
    )
    generated_tokens = _ids("ok")
    generated = {
        "tokens": generated_tokens,
        "_final_state": _final_state(generated_tokens),
    }

    result = _store_generation_final_history_snapshot(
        state,
        session_id="session-1",
        prompt_ids=prompt_ids,
        generated=generated,
        messages=messages,
        assistant_content="ok!",
        thinking_enabled=False,
        policy_fingerprint="policy",
    )

    assert result["stored"] is True
    assert result["mode"] == "generation_final_prefix"
    assert result["history_suffix_tokens"] == 1
    assert state.sessions.bank.puts[0]["token_ids"] == prompt_ids + generated_tokens


def test_generation_final_postcommit_rejects_tool_call_history_rewrite():
    state = _postcommit_state()
    messages = [ChatMessage(role="user", content="call tool")]
    prompt_ids = _encode_messages(
        state.runtime.tokenizer,
        messages,
        enable_thinking=False,
        add_generation_prompt=True,
    )
    generated_tokens = _ids('{"name":"lookup"}')
    generated = {
        "tokens": generated_tokens,
        "_final_state": _final_state(generated_tokens),
    }

    compatibility = _generation_final_postcommit_compatibility(
        state,
        prompt_ids=prompt_ids,
        generated=generated,
        messages=messages,
        assistant_content="",
        assistant_tool_calls=[
            {
                "type": "function",
                "function": {"name": "lookup", "arguments": {}},
            }
        ],
        thinking_enabled=False,
    )

    assert compatibility["safe"] is False
    assert compatibility["reason"] == "tool_call_history_rewrite"
    assert state.sessions.bank.puts == []


def test_idle_async_postcommit_returns_pending_and_dispatches_retokenized_commit(
    capsys, monkeypatch
):
    """When the foreground is idle the async postcommit should attempt the
    retokenized commit (not silently abandon as the old build did)."""
    state = _postcommit_state()

    captured_calls = []

    def fake_retokenized_commit(state, **kwargs):
        captured_calls.append(kwargs)
        return {
            "stored": True,
            "mode": "retokenized_history",
            "prefix_len": 5,
            "nbytes": 123,
        }

    monkeypatch.setattr(
        "mtplx.server.openai._store_retokenized_history_snapshot",
        fake_retokenized_commit,
    )

    pending = _schedule_idle_postcommit_snapshot(
        state,
        session_id="session-1",
        messages=[ChatMessage(role="user", content="hi")],
        assistant_content="ok",
        thinking_enabled=False,
        policy_fingerprint="policy",
        unsafe_reason="retokenized_history_mismatch",
    )

    assert pending == {
        "stored": False,
        "mode": "async_pending",
        "reason": "retokenized_history_mismatch",
    }
    assert len(captured_calls) == 1
    assert captured_calls[0]["session_id"] == "session-1"
    assert captured_calls[0]["assistant_content"] == "ok"
    assert captured_calls[0]["acquire_model_lock_blocking"] is False
    log = capsys.readouterr().out
    assert '"stored": true' in log
    assert "retokenized_history" in log


def test_idle_async_postcommit_attempts_commit_for_tool_call_responses(
    capsys, monkeypatch
):
    """Tool-call responses must reach the retokenized commit path. This is the
    regression case for the async-postcommit fix: the unpatched build would
    log 'abandoned_foreground_busy' and never call bank.put."""
    state = _postcommit_state()

    captured_calls = []

    def fake_retokenized_commit(state, **kwargs):
        captured_calls.append(kwargs)
        return {"stored": True, "mode": "retokenized_history", "prefix_len": 8}

    monkeypatch.setattr(
        "mtplx.server.openai._store_retokenized_history_snapshot",
        fake_retokenized_commit,
    )

    pending = _schedule_idle_postcommit_snapshot(
        state,
        session_id="session-tool",
        messages=[ChatMessage(role="user", content="call lookup")],
        assistant_content="",
        assistant_tool_calls=[
            {"type": "function", "function": {"name": "lookup", "arguments": {}}}
        ],
        thinking_enabled=False,
        policy_fingerprint="policy",
        unsafe_reason="tool_call_history_rewrite",
    )

    assert pending["mode"] == "async_pending"
    assert pending["reason"] == "tool_call_history_rewrite"
    assert len(captured_calls) == 1
    # Tool calls must be forwarded so the canonical encoding includes them.
    assert captured_calls[0]["assistant_tool_calls"] == [
        {"type": "function", "function": {"name": "lookup", "arguments": {}}}
    ]
    assert captured_calls[0]["acquire_model_lock_blocking"] is False
    log = capsys.readouterr().out
    assert "tool_call_history_rewrite" in log
    assert '"stored": true' in log


def test_idle_async_postcommit_abandons_when_model_lock_stays_busy(
    capsys, monkeypatch
):
    """If the model lock never frees, the async commit must abandon
    rather than block forever. The loop now drives the non-blocking acquire
    inside `_store_retokenized_history_snapshot` as the correctness gate."""
    state = _postcommit_state()
    state.has_foreground = lambda: True  # intentionally ignored by this path

    # Make the wait short so the test stays fast.
    monkeypatch.setattr(
        "mtplx.server.openai._IDLE_POSTCOMMIT_MAX_WAIT_S", 0.1
    )
    monkeypatch.setattr(
        "mtplx.server.openai._IDLE_POSTCOMMIT_POLL_INTERVAL_S", 0.05
    )

    called: list[dict] = []

    def fake_store_lock_busy(state, **kwargs):
        called.append(kwargs)
        return {
            "stored": False,
            "mode": "retokenized_history",
            "reason": "model_lock_busy_before_retokenized_commit",
        }

    monkeypatch.setattr(
        "mtplx.server.openai._store_retokenized_history_snapshot",
        fake_store_lock_busy,
    )

    pending = _schedule_idle_postcommit_snapshot(
        state,
        session_id="session-busy",
        messages=[ChatMessage(role="user", content="hi")],
        assistant_content="ok",
        thinking_enabled=False,
        policy_fingerprint="policy",
        unsafe_reason="retokenized_history_mismatch",
    )

    assert pending["mode"] == "async_pending"
    assert called, "loop must attempt the non-blocking commit before abandoning"
    log = capsys.readouterr().out
    assert "abandoned_foreground_busy" in log
    assert "model_lock_busy_past_deadline" in log


def test_server_security_requires_api_key_for_non_localhost_bind():
    args = parse_args(["--host", "0.0.0.0"])

    with pytest.raises(SystemExit):
        validate_server_security_args(args)


def test_server_auth_accepts_bearer_and_x_api_key():
    assert _request_is_authorized(
        SimpleNamespace(headers={"authorization": "Bearer test-key"}),
        "test-key",
    )
    assert _request_is_authorized(
        SimpleNamespace(headers={"x-api-key": "test-key"}),
        "test-key",
    )
    assert not _request_is_authorized(
        SimpleNamespace(headers={"authorization": "Bearer wrong"}),
        "test-key",
    )


def test_rate_limiter_enforces_window():
    limiter = _RateLimiter(2)

    assert limiter.check("client", now=100.0) == (True, 0)
    assert limiter.check("client", now=101.0) == (True, 0)
    allowed, retry_after = limiter.check("client", now=102.0)
    assert allowed is False
    assert retry_after > 0
    assert limiter.check("client", now=161.5) == (True, 0)


def test_stream_cancel_helper_marks_event_and_cancels_future():
    cancel_event = Event()

    class Future:
        cancelled = False

        def cancel(self):
            self.cancelled = True
            return True

    future = Future()
    _raise_if_stream_cancelled(cancel_event)

    _cancel_stream_generation(cancel_event, future)

    assert cancel_event.is_set()
    assert future.cancelled is True
    with pytest.raises(_StreamCancelled):
        _raise_if_stream_cancelled(cancel_event)


def test_stream_heartbeat_payload_is_progress_only():
    payload = _stream_heartbeat_payload(
        completion_tokens=42,
        stream_started_s=100.0,
        last_token_s=125.0,
        now_s=140.0,
    )

    assert payload == {
        "heartbeat": True,
        "phase": "generating",
        "completion_tokens": 42,
        "elapsed_s": 40.0,
        "seconds_since_last_token": 15.0,
    }


def test_anthropic_content_blocks_convert_to_text():
    text = _anthropic_content_to_text(
        [
            {"type": "text", "text": "hello"},
            {"type": "tool_result", "content": [{"type": "text", "text": " world"}]},
        ]
    )

    assert text == "hello world"


def test_anthropic_request_translates_to_openai_chat_request():
    request = AnthropicMessagesRequest(
        model="mtplx",
        system=[{"type": "text", "text": "system"}],
        max_tokens=64,
        messages=[
            AnthropicMessage(role="user", content=[{"type": "text", "text": "hi"}]),
            AnthropicMessage(role="assistant", content="hello"),
        ],
        temperature=0.4,
        top_p=0.9,
    )

    chat = _anthropic_to_chat_request(request)

    assert chat.model == "mtplx"
    assert chat.max_tokens == 64
    assert chat.temperature == 0.4
    assert chat.top_p == 0.9
    assert [(message.role, message.content) for message in chat.messages] == [
        ("system", "system"),
        ("user", "hi"),
        ("assistant", "hello"),
    ]


def test_anthropic_payload_from_openai_response():
    payload = _anthropic_payload_from_openai(
        {
            "model": "mtplx",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            "mtplx_stats": {"tok_s": 42.0},
        }
    )

    assert payload["type"] == "message"
    assert payload["role"] == "assistant"
    assert payload["content"] == [{"type": "text", "text": "hello"}]
    assert payload["usage"] == {"input_tokens": 12, "output_tokens": 3}
    assert payload["mtplx_stats"] == {"tok_s": 42.0}


def test_public_mtplx_stats_excludes_internal_trace_fields():
    stats = _public_mtplx_stats(
        {
            "stats": {
                "generated_tokens": 4,
                "decode_tok_s": 18.25,
                "session_cache_hit": True,
                "cached_tokens": 128,
                "session_restore_mode": "reference_lease",
                "events": [{"step": 0, "drafts": [{"token": 1}]}],
                "owned_attn_kv": {"bytes": 1024},
                "graphbank": {"debug": "internal"},
                "session_postcommit_snapshot": {
                    "stored": True,
                    "prefix_len": 64,
                    "nbytes": 1234,
                    "token_hash": "internal",
                },
            }
        }
    )

    assert stats["generated_tokens"] == 4
    assert stats["decode_tok_s"] == 18.25
    assert stats["session_cache_hit"] is True
    assert stats["cached_tokens"] == 128
    assert "events" not in stats
    assert "owned_attn_kv" not in stats
    assert "graphbank" not in stats
    assert stats["session_postcommit_snapshot"] == {
        "stored": True,
        "prefix_len": 64,
        "nbytes": 1234,
    }


def test_anthropic_stream_translates_openai_sse_events():
    async def upstream():
        yield (
            'data: {"choices":[{"delta":{"role":"assistant"},'
            '"finish_reason":null}]}\n\n'
        )
        yield (
            'data: {"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}\n\n'
        )
        yield (
            'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}\n\n'
        )
        yield (
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":5,"completion_tokens":2},'
            '"mtplx_stats":{"tok_s":12.5}}\n\n'
        )
        yield "data: [DONE]\n\n"

    async def collect():
        return [
            chunk
            async for chunk in _anthropic_stream_from_openai_sse(
                upstream(),
                model="mtplx",
            )
        ]

    chunks = asyncio.run(collect())
    frames = [frame for frame in "".join(chunks).split("\n\n") if frame]
    events = []
    for frame in frames:
        lines = frame.splitlines()
        event = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        events.append((event, data))

    assert [event for event, _data in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[0][1]["message"]["model"] == "mtplx"
    assert events[2][1]["delta"] == {"type": "text_delta", "text": "Hel"}
    assert events[3][1]["delta"] == {"type": "text_delta", "text": "lo"}
    assert events[5][1]["delta"]["stop_reason"] == "end_turn"
    assert events[5][1]["usage"] == {"output_tokens": 2}
    assert events[5][1]["mtplx_stats"] == {"tok_s": 12.5}


class RecordingTokenizer:
    def __init__(self):
        self.normalized = None
        self.kwargs = None

    def apply_chat_template(self, normalized, **_kwargs):
        self.normalized = normalized
        self.kwargs = _kwargs
        return [1, 2, 3]


def test_strip_assistant_history_baggage_removes_openwebui_reasoning_and_footer():
    text = (
        '<details type="reasoning" done="true"><summary>Thought for 4 seconds</summary>'
        "private chain</details>\n"
        "<think>old hidden reasoning</think>\n"
        "Visible answer."
        f"{STATS_FOOTER_MARKER} **62.0 tok/s** · 10 tokens · 0.16s decode"
    )

    stripped = _strip_assistant_history_baggage(text)

    assert stripped == "Visible answer."


def test_encode_messages_preserves_assistant_reasoning_context_by_default():
    tokenizer = RecordingTokenizer()
    _encode_messages(
        tokenizer,
        [
            ChatMessage(
                role="assistant",
                content="<think>useful prior reasoning</think>\nVisible answer.",
            )
        ],
        enable_thinking=True,
    )

    assert tokenizer.normalized == [
        {
            "role": "assistant",
            "content": "<think>useful prior reasoning</think>\nVisible answer.",
        }
    ]
    assert tokenizer.kwargs["preserve_thinking"] is True


def test_encode_messages_normalizes_openwebui_reasoning_details_for_prefix_lookup():
    tokenizer = RecordingTokenizer()
    _encode_messages(
        tokenizer,
        [
            ChatMessage(
                role="assistant",
                content=(
                    '<details type="reasoning" done="true">'
                    "<summary>Thought for 2 seconds</summary>"
                    "&gt; useful prior reasoning"
                    "</details>\nVisible answer."
                    f"{STATS_FOOTER_MARKER} **62.0 tok/s** · 10 tokens · 0.16s decode"
                ),
            )
        ],
        enable_thinking=True,
    )

    assert tokenizer.normalized == [
        {
            "role": "assistant",
            "content": "<think>\nuseful prior reasoning\n</think>\nVisible answer.",
        }
    ]


def test_normalize_thinking_tags_wraps_capped_reasoning_for_history_match():
    assert (
        _normalize_thinking_tags("unfinished reasoning", thinking_enabled=True)
        == "<think>\nunfinished reasoning\n</think>"
    )


def test_normalize_thinking_tags_canonicalizes_reentrant_thinking_blocks():
    normalized = _normalize_thinking_tags(
        "<think>first hidden</think>OK<think>second hidden</think>Done",
        thinking_enabled=True,
    )

    assert normalized == "<think>\nfirst hidden\nsecond hidden\n</think>\n\nOKDone"


def test_encode_messages_can_strip_assistant_reasoning_context_when_requested():
    tokenizer = RecordingTokenizer()
    _encode_messages(
        tokenizer,
        [
            ChatMessage(
                role="assistant",
                content="<think>old reasoning</think>\nVisible answer.",
            )
        ],
        enable_thinking=True,
        strip_assistant_reasoning_history=True,
    )

    assert tokenizer.normalized == [{"role": "assistant", "content": "Visible answer."}]
    assert tokenizer.kwargs["preserve_thinking"] is False


def test_incremental_token_decoder_does_not_redecode_cumulative_history():
    decoder = _IncrementalTokenDecoder(TinyTokenizer())

    assert decoder.feed(_ids("hello ")) == "hello "
    assert decoder.feed(_ids("wor")) == ""
    assert decoder.feed(_ids("ld ")) == "world "
    assert decoder.finish() == ""


def test_incremental_token_decoder_flushes_think_close_without_waiting_for_space():
    decoder = _IncrementalTokenDecoder(TinyTokenizer())

    assert decoder.feed(_ids("reasoning ")) == "reasoning "
    assert decoder.feed(_ids("</think>")) == "</think>"
    assert decoder.feed(_ids("Answer ")) == "Answer "


def test_thinking_stream_splitter_keeps_reasoning_out_of_content():
    splitter = _ThinkingContentStreamSplitter(thinking_enabled=True)

    chunks = []
    chunks.extend(splitter.start())
    chunks.extend(splitter.feed("first thought "))
    chunks.extend(splitter.feed("still thought</think>Final answer."))
    chunks.extend(splitter.finish())

    reasoning = "".join(text for field, text in chunks if field == "reasoning_content")
    content = "".join(text for field, text in chunks if field == "content")

    assert reasoning == "first thought still thought"
    assert content == "Final answer."


def test_thinking_stream_splitter_routes_reentrant_thinking_out_of_content():
    splitter = _ThinkingContentStreamSplitter(thinking_enabled=True)

    chunks = []
    for piece in ["first </thi", "nk>Visible <thi", "nk>second</thi", "nk>More"]:
        chunks.extend(splitter.feed(piece))
    chunks.extend(splitter.finish())

    reasoning = "".join(text for field, text in chunks if field == "reasoning_content")
    content = "".join(text for field, text in chunks if field == "content")

    assert reasoning == "first second"
    assert content == "Visible More"
    assert "<think>" not in content
    assert "</think>" not in content
    assert splitter.reentry_count == 1


def test_generation_params_exposes_no_server_cap_when_unset():
    state = SimpleNamespace(
        context_window=1000,
        args=SimpleNamespace(
            max_response_tokens=None,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        ),
    )

    response_max, _sampler, limits = _generation_params(
        state,
        prompt_token_count=100,
        max_tokens=None,
        temperature=None,
        top_p=None,
        top_k=None,
    )

    assert response_max == 900
    assert limits["request_max_tokens"] is None
    assert limits["server_max_response_tokens"] is None
    assert limits["effective_max_tokens"] == 900
    assert limits["server_cap_applied"] is False
    assert limits["context_cap_applied"] is False


def test_generation_params_marks_server_cap_when_configured():
    state = SimpleNamespace(
        context_window=1000,
        args=SimpleNamespace(
            max_response_tokens=384,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        ),
    )

    response_max, _sampler, limits = _generation_params(
        state,
        prompt_token_count=100,
        max_tokens=None,
        temperature=None,
        top_p=None,
        top_k=None,
    )

    assert response_max == 384
    assert limits["server_max_response_tokens"] == 384
    assert limits["effective_max_tokens"] == 384
    assert limits["server_cap_applied"] is True
    assert limits["context_cap_applied"] is False


def test_online_hidden_config_and_policy_fingerprint_track_proposal_policy():
    args = SimpleNamespace(
        strip_assistant_reasoning_history=False,
        adaptive_policy="none",
        online_correction_cache=False,
        online_correction_cache_min_depth=1,
        online_correction_cache_key="local_prefix",
        prompt_correction_cache=False,
        prompt_correction_cache_min_depth=2,
        online_hidden_corrector_alpha=0.25,
        online_hidden_corrector_decay=0.7,
        online_hidden_corrector_warmup=2,
        online_hidden_corrector_max_feed_depth=2,
        online_hidden_corrector_key="token",
    )
    state = SimpleNamespace(
        args=args,
        template_hash="template",
        draft_head_identity="draft",
    )

    config = _online_hidden_config(args)
    fingerprint = _policy_fingerprint(state, thinking_enabled=True)

    assert config == {
        "alpha": 0.25,
        "decay": 0.7,
        "warmup": 2,
        "max_feed_depth": 2,
        "key": "token",
    }
    assert 'online_hidden={"alpha":0.25' in fingerprint
    assert '"key":"token"' in fingerprint


def test_streamed_generation_stats_recover_zero_final_token_count():
    completion_tokens = _effective_completion_tokens(
        generated_tokens=[],
        streamed_token_times=[1.0, 1.1, 1.2],
    )
    stats = _repair_streamed_generation_stats(
        {"generated_tokens": 0, "elapsed_s": 0.5, "tok_s": 0.0},
        completion_tokens=completion_tokens,
        elapsed_s=0.5,
    )

    assert completion_tokens == 3
    assert stats["generated_tokens"] == 3
    assert stats["generated_tokens_raw"] == 0
    assert stats["generated_tokens_recovered_from_stream"] is True
    assert stats["tok_s"] == 6.0


def test_streamed_generation_stats_keep_real_generation_count_when_larger():
    completion_tokens = _effective_completion_tokens(
        generated_tokens=[1, 2, 3, 4],
        streamed_token_times=[1.0, 1.1, 1.2],
    )
    stats = _repair_streamed_generation_stats(
        {"generated_tokens": 4, "tok_s": 8.0},
        completion_tokens=completion_tokens,
        elapsed_s=0.5,
    )

    assert completion_tokens == 4
    assert stats["generated_tokens"] == 4
    assert "generated_tokens_recovered_from_stream" not in stats


def test_usage_payload_uses_repaired_completion_tokens():
    assert _usage_payload({"prompt_tokens": 12, "completion_tokens": 34}) == {
        "prompt_tokens": 12,
        "completion_tokens": 34,
        "total_tokens": 46,
    }
