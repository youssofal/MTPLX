import asyncio
import gc
import json
from threading import Event, Lock
from types import SimpleNamespace
import weakref

import pytest

from mtplx.generation import (
    RepetitionStopConfig,
    _detect_repeated_token_suffix,
    _trim_repeated_suffix,
)
from mtplx.reasoning_codecs import split_reasoning_text, stream_splitter_for_parser
from mtplx.server.openai import (
    AnthropicMessage,
    AnthropicMessagesRequest,
    ChatMessage,
    STATS_FOOTER_MARKER,
    _BROWSER_AUTH_COOKIE,
    _RateLimiter,
    _adaptive_config,
    _aime_visible_working_for_request,
    _anthropic_content_to_text,
    _anthropic_payload_from_openai,
    _anthropic_stream_from_openai_sse,
    _anthropic_to_chat_request,
    _IncrementalTokenDecoder,
    _StreamCancelled,
    _ThinkingContentStreamSplitter,
    _cancel_stream_generation,
    _canonicalize_agent_transcript,
    _compact_tool_result_text,
    _encode_messages,
    _effective_completion_tokens,
    _generation_params,
    _generation_final_postcommit_compatibility,
    _merge_final_bridge_stats_into_latest_metrics,
    _metrics_envelope,
    _monitor_request_disconnect,
    _nonstream_chat_message_parts,
    _normalize_thinking_tags,
    _online_hidden_config,
    _opencode_tool_history_restore_policy,
    _policy_fingerprint,
    _public_mtplx_stats,
    _raise_if_stream_cancelled,
    _render_messages_for_postcommit,
    _repair_streamed_generation_stats,
    _request_is_authorized,
    _response_id_from_client_hint,
    _schedule_idle_postcommit_snapshot,
    _session_cache_scope_for_request,
    _should_bypass_session_cache_for_opencode_tool_history,
    _should_force_clone_session_cache_for_opencode_tool_history,
    _store_generation_final_history_snapshot,
    _strip_assistant_history_baggage,
    _stream_cancelled_queue_item,
    _stream_heartbeat_payload,
    _usage_payload,
    _uncapped_repetition_stop_enabled,
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


def test_parse_args_preserve_thinking_defaults_to_safe_auto():
    args = parse_args(["--model", "dummy"])

    assert args.preserve_thinking == "auto"
    assert args.strip_assistant_reasoning_history is False


def test_parse_args_preserve_thinking_off_strips_assistant_reasoning_history():
    args = parse_args(["--model", "dummy", "--preserve-thinking", "off"])

    assert args.preserve_thinking == "off"
    assert args.strip_assistant_reasoning_history is True


def test_parse_args_strip_assistant_reasoning_history_alias_sets_preserve_off():
    args = parse_args(["--model", "dummy", "--strip-assistant-reasoning-history"])

    assert args.preserve_thinking == "off"
    assert args.strip_assistant_reasoning_history is True


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
    assert args.session_cache_mode == "on"
    validate_server_security_args(args)

    stock = parse_args(["--stock-ar"])
    assert stock.stock_ar is True
    assert stock.generation_mode == "ar"
    assert stock.load_mtp is False


def test_server_parse_args_accepts_construction_time_session_cache_disable():
    args = parse_args(["--session-cache-mode", "off"])

    assert args.session_cache_mode == "off"


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


def test_idle_async_postcommit_abandons_when_model_lock_stays_busy(capsys, monkeypatch):
    """If the model lock never frees, the async commit must abandon
    rather than block forever. The loop now drives the non-blocking acquire
    inside `_store_retokenized_history_snapshot` as the correctness gate."""
    state = _postcommit_state()
    state.has_foreground = lambda: True  # intentionally ignored by this path

    # Make the wait short so the test stays fast.
    monkeypatch.setattr("mtplx.server.openai._IDLE_POSTCOMMIT_MAX_WAIT_S", 0.1)
    monkeypatch.setattr("mtplx.server.openai._IDLE_POSTCOMMIT_POLL_INTERVAL_S", 0.05)

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
    assert _request_is_authorized(
        SimpleNamespace(headers={}, cookies={_BROWSER_AUTH_COOKIE: "test-key"}),
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


def test_stream_cancel_queue_item_drops_traceback_frames():
    class Payload:
        pass

    def make_cancelled_exc():
        payload = Payload()
        payload_ref = weakref.ref(payload)
        try:
            raise _StreamCancelled("request cancelled")
        except _StreamCancelled as exc:
            return exc, payload_ref

    exc, payload_ref = make_cancelled_exc()
    assert payload_ref() is not None

    item = _stream_cancelled_queue_item(exc)

    assert item == ("cancelled", "request cancelled")
    assert exc.__traceback__ is None
    del exc
    gc.collect()
    assert payload_ref() is None


def test_request_disconnect_monitor_sets_cancel_event():
    class FakeRequest:
        def __init__(self):
            self.calls = 0

        async def is_disconnected(self):
            self.calls += 1
            return self.calls >= 2

    cancel_event = Event()
    disconnected = []

    result = asyncio.run(
        _monitor_request_disconnect(
            FakeRequest(),
            cancel_event,
            poll_s=0,
            on_disconnect=lambda: disconnected.append(True),
        )
    )

    assert result is True
    assert cancel_event.is_set()
    assert disconnected == [True]


def test_response_id_from_client_hint_sanitizes_cancelable_ids():
    assert (
        _response_id_from_client_hint(
            prefix="chatcmpl",
            headers={"x-mtplx-request-id": "aime-row-1"},
            metadata={},
        )
        == "chatcmpl-aime-row-1"
    )
    assert (
        _response_id_from_client_hint(
            prefix="chatcmpl",
            headers={},
            metadata={"mtplx_request_id": "chatcmpl-existing"},
        )
        == "chatcmpl-existing"
    )
    invalid = _response_id_from_client_hint(
        prefix="chatcmpl",
        headers={"x-mtplx-request-id": "../bad"},
        metadata={},
    )
    assert invalid.startswith("chatcmpl-")
    assert invalid != "chatcmpl-../bad"


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


def test_anthropic_request_translates_claude_code_tools_and_history():
    request = AnthropicMessagesRequest(
        model="mtplx",
        system=[
            {"type": "text", "text": "x-anthropic-billing-header: noise"},
            {"type": "text", "text": "real system"},
        ],
        max_tokens=64,
        stop_sequences=["</stop>"],
        metadata={"session_id": "claude-code-smoke"},
        thinking={"type": "enabled"},
        tools=[
            {
                "name": "Bash",
                "description": "Run a shell command",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
            {
                "type": "web_search_20250305",
                "name": "web_search",
            },
        ],
        tool_choice={"type": "tool", "name": "Bash"},
        messages=[
            AnthropicMessage(role="user", content="Run ./test.sh"),
            AnthropicMessage(
                role="assistant",
                content=[
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Bash",
                        "input": {"command": "./test.sh"},
                    }
                ],
            ),
            AnthropicMessage(
                role="user",
                content=[
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"type": "text", "text": "ok"}],
                    }
                ],
            ),
        ],
    )

    chat = _anthropic_to_chat_request(request)

    assert [(message.role, message.content) for message in chat.messages] == [
        ("system", "real system"),
        ("user", "Run ./test.sh"),
        ("assistant", ""),
        ("tool", "ok"),
    ]
    assert chat.messages[2].tool_calls == [
        {
            "id": "toolu_1",
            "type": "function",
            "function": {"name": "Bash", "arguments": '{"command":"./test.sh"}'},
        }
    ]
    assert chat.messages[3].tool_call_id == "toolu_1"
    assert chat.tools == [
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "description": "Run a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    assert chat.tool_choice == {"type": "function", "function": {"name": "Bash"}}
    assert chat.stop == ["</stop>"]
    assert chat.metadata == {"session_id": "claude-code-smoke"}
    assert chat.enable_thinking is True


def test_anthropic_request_keeps_broader_claude_code_client_tools():
    tool_names = [
        "Bash",
        "Read",
        "Edit",
        "Write",
        "MultiEdit",
        "Glob",
        "Grep",
        "LS",
        "TodoWrite",
    ]
    request = AnthropicMessagesRequest(
        model="mtplx",
        max_tokens=64,
        messages=[AnthropicMessage(role="user", content="work")],
        tools=[
            {
                "name": name,
                "description": f"{name} tool",
                "input_schema": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
            }
            for name in tool_names
        ],
    )

    chat = _anthropic_to_chat_request(request)

    assert [tool["function"]["name"] for tool in chat.tools or []] == tool_names
    assert all(tool["type"] == "function" for tool in chat.tools or [])


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
    assert payload["usage"] == {
        "input_tokens": 12,
        "output_tokens": 3,
        # #121/#144 telemetry: session-cache hit mirrored Anthropic-style.
        "cache_read_input_tokens": 0,
    }
    assert payload["mtplx_stats"] == {"tok_s": 42.0}


def test_anthropic_payload_from_openai_tool_call_response():
    payload = _anthropic_payload_from_openai(
        {
            "model": "mtplx",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "Bash",
                                    "arguments": '{"command":"./test.sh"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        }
    )

    assert payload["content"] == [
        {
            "type": "tool_use",
            "id": "call_1",
            "name": "Bash",
            "input": {"command": "./test.sh"},
        }
    ]
    assert payload["stop_reason"] == "tool_use"


def test_public_mtplx_stats_excludes_internal_trace_fields():
    # ``graphbank`` is now intentionally part of the public surface so the
    # dashboard's verify-waterfall + GraphBank panel can render it. The
    # rest of the "internal trace" contract (events, owned_attn_kv,
    # postcommit token_hash) still applies.
    stats = _public_mtplx_stats(
        {
            "stats": {
                "generated_tokens": 4,
                "decode_tok_s": 18.25,
                "session_cache_hit": True,
                "cached_tokens": 128,
                "session_restore_mode": "reference_lease",
                "mtp_history_policy": "last_window",
                "mtp_history_window_tokens": 8192,
                "mtp_history_position_base": 11835,
                "opencode_tool_history_cache_bypass": False,
                "opencode_tool_history_force_clone_restore": True,
                "opencode_tool_history_live_frontier_restore": False,
                "request_session_bank_bypass": False,
                "request_client_hint": "aime",
                "request_enable_thinking": True,
                "request_temperature": 1.0,
                "request_top_p": 0.9,
                "request_top_k": 40,
                "mlx_cache_cleanup": {
                    "cleared": True,
                    "reason": "aime_stateless_question",
                },
                "request_session_keep_live_ref": False,
                "request_session_keep_live_ref_reason": "opencode_tool_snapshot_only",
                "live_frontier_policy": "opencode_snapshot_only",
                "live_frontier_result_turn": True,
                "live_frontier_hit": False,
                "live_frontier_restore_mode": "opencode_tool_history_bypass",
                "live_frontier_miss_reason": "opencode_tool_history_cache_bypass",
                "live_frontier_assistant_tool_call_count": 2,
                "live_frontier_tool_result_count": 1,
                "live_frontier_unknown_tool_result_count": 0,
                "transcript_inspection_read_budget_candidate_messages": 6,
                "transcript_inspection_read_budget_max_lines_per_file": 16,
                "visible_reasoning_stripped": True,
                "nonstream_reasoning_content_routed": True,
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
    assert stats["mtp_history_policy"] == "last_window"
    assert stats["mtp_history_window_tokens"] == 8192
    assert stats["mtp_history_position_base"] == 11835
    assert stats["opencode_tool_history_cache_bypass"] is False
    assert stats["opencode_tool_history_force_clone_restore"] is True
    assert stats["opencode_tool_history_live_frontier_restore"] is False
    assert stats["request_session_bank_bypass"] is False
    assert stats["request_client_hint"] == "aime"
    assert stats["request_enable_thinking"] is True
    assert stats["request_temperature"] == 1.0
    assert stats["request_top_p"] == 0.9
    assert stats["request_top_k"] == 40
    assert stats["mlx_cache_cleanup"] == {
        "cleared": True,
        "reason": "aime_stateless_question",
    }
    assert stats["request_session_keep_live_ref"] is False
    assert stats["request_session_keep_live_ref_reason"] == "opencode_tool_snapshot_only"
    assert stats["live_frontier_policy"] == "opencode_snapshot_only"
    assert stats["live_frontier_result_turn"] is True
    assert stats["live_frontier_hit"] is False
    assert stats["live_frontier_restore_mode"] == "opencode_tool_history_bypass"
    assert stats["live_frontier_miss_reason"] == "opencode_tool_history_cache_bypass"
    assert stats["live_frontier_assistant_tool_call_count"] == 2
    assert stats["live_frontier_tool_result_count"] == 1
    assert stats["live_frontier_unknown_tool_result_count"] == 0
    assert stats["transcript_inspection_read_budget_candidate_messages"] == 6
    assert stats["transcript_inspection_read_budget_max_lines_per_file"] == 16
    assert stats["visible_reasoning_stripped"] is True
    assert stats["nonstream_reasoning_content_routed"] is True
    assert "events" not in stats
    assert "owned_attn_kv" not in stats
    assert stats["graphbank"] == {"debug": "internal"}
    assert stats["session_postcommit_snapshot"] == {
        "stored": True,
        "prefix_len": 64,
        "nbytes": 1234,
    }


def test_final_bridge_stats_update_latest_metrics_with_frontier_cache_fields():
    state = SimpleNamespace(last_metrics=[{"request_id": "resp_1"}])

    _merge_final_bridge_stats_into_latest_metrics(
        state,
        {
            "tool_parse_success": True,
            "session_prompt_prefix_commit": {
                "committed": True,
                "reason": "committed_prompt_prefix",
                "prefix_len": 2048,
                "boundary_kind": "tool_call_prompt_prefix",
            },
            "session_postcommit_snapshot": {
                "stored": False,
                "mode": "async_pending",
                "reason": "tool_call_history_rewrite",
            },
        },
    )

    latest = state.last_metrics[-1]
    assert latest["tool_parse_success"] is True
    assert latest["session_prompt_prefix_commit"]["prefix_len"] == 2048
    assert latest["session_prompt_prefix_commit"]["boundary_kind"] == (
        "tool_call_prompt_prefix"
    )
    assert latest["session_postcommit_snapshot"] == {
        "stored": False,
        "mode": "async_pending",
        "reason": "tool_call_history_rewrite",
    }


def _anthropic_stream_events(chunks):
    frames = [frame for frame in "".join(chunks).split("\n\n") if frame]
    events = []
    for frame in frames:
        lines = frame.splitlines()
        event = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        events.append((event, data))
    return events


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
    events = _anthropic_stream_events(chunks)

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


def test_anthropic_stream_translates_openai_tool_call_deltas():
    async def upstream():
        yield (
            'data: {"choices":[{"delta":{"role":"assistant"},'
            '"finish_reason":null}]}\n\n'
        )
        yield (
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"id":"call_1","type":"function","function":{"name":"Bash",'
            '"arguments":""}}]},"finish_reason":null}]}\n\n'
        )
        yield (
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":"{\\"command\\":"}}]},'
            '"finish_reason":null}]}\n\n'
        )
        yield (
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":"\\"./test.sh\\"}"}}]},'
            '"finish_reason":null}]}\n\n'
        )
        yield (
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],'
            '"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
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
    joined = "".join(chunks)
    events = _anthropic_stream_events(chunks)

    assert [event for event, _data in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[1][1]["index"] == 0
    assert events[1][1]["content_block"] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "Bash",
        "input": {},
    }
    assert events[2][1]["delta"] == {
        "type": "input_json_delta",
        "partial_json": '{"command":',
    }
    assert events[3][1]["delta"] == {
        "type": "input_json_delta",
        "partial_json": '"./test.sh"}',
    }
    assert events[5][1]["delta"]["stop_reason"] == "tool_use"
    assert "<tool_call" not in joined
    assert '"type": "text"' not in joined


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


def test_transcript_metrics_count_assistant_reasoning_history():
    _canonical, stats = _canonicalize_agent_transcript(
        [
            ChatMessage(
                role="assistant",
                content=[
                    {"type": "thinking", "thinking": "structured hidden"},
                    {"type": "text", "text": "Visible answer."},
                ],
            ),
            ChatMessage(
                role="assistant",
                content="<think>tagged hidden</think>\nVisible answer.",
            ),
            ChatMessage(
                role="assistant",
                content="Visible answer.",
                reasoning_content="native hidden",
            ),
        ],
        tools_active=True,
    )

    metrics = stats.to_metrics()
    assert metrics["transcript_assistant_reasoning_history_messages"] == 3
    assert metrics["transcript_assistant_structured_thinking_blocks"] == 2
    assert metrics["transcript_assistant_reasoning_history_chars"] == (
        len("structured hidden")
        + len("<think>tagged hidden</think>")
        + len("native hidden")
    )


def test_old_tool_result_compaction_threshold_is_env_tunable(monkeypatch):
    monkeypatch.setenv("MTPLX_TOOL_RESULT_COMPACT_THRESHOLD_CHARS", "20")
    monkeypatch.setenv("MTPLX_TOOL_RESULT_COMPACT_HEAD_CHARS", "4")
    monkeypatch.setenv("MTPLX_TOOL_RESULT_COMPACT_TAIL_CHARS", "4")

    compacted = _compact_tool_result_text("abcdefghijklmnopqrstuvwxyz")

    assert compacted is not None
    assert "abcd" in compacted
    assert "wxyz" in compacted
    assert "MTPLX compacted" in compacted


def test_postcommit_render_uses_same_preserve_thinking_policy():
    tokenizer = RecordingTokenizer()

    _render_messages_for_postcommit(
        tokenizer,
        [{"role": "assistant", "content": "<think>old</think>\nVisible."}],
        enable_thinking=True,
        preserve_thinking=False,
        tools=None,
    )

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


def test_step3p5_stream_splitter_uses_think_tags():
    splitter = stream_splitter_for_parser("step3p5", thinking_enabled=True)

    chunks = []
    chunks.extend(splitter.feed("<think>check the user request</think>Actual answer."))
    chunks.extend(splitter.finish())

    reasoning = "".join(text for field, text in chunks if field == "reasoning_content")
    content = "".join(text for field, text in chunks if field == "content")

    assert reasoning == "check the user request"
    assert content == "Actual answer."


def test_step3p5_reasoning_off_strips_orphan_close_from_stream():
    splitter = stream_splitter_for_parser("step3p5", thinking_enabled=False)

    chunks = []
    for piece in [
        "I'm doing well, thank you for asking. ",
        "</thi",
        "nks> I'm doing well, thank you for asking.",
    ]:
        chunks.extend(splitter.feed(piece))
    chunks.extend(splitter.finish())

    reasoning = "".join(text for field, text in chunks if field == "reasoning_content")
    content = "".join(text for field, text in chunks if field == "content")

    assert reasoning == ""
    assert content == "I'm doing well, thank you for asking."
    assert "</think" not in content
    assert "</thinks" not in content


def test_openai_splitter_reasoning_off_streams_plain_text_incrementally():
    splitter = _ThinkingContentStreamSplitter(
        thinking_enabled=False,
        recover_unclosed_reasoning_as_content=False,
    )

    chunks = []
    for piece in ["Count: ", "1, ", "2, ", "3.\n"]:
        chunks.extend(splitter.feed(piece))
    chunks.extend(splitter.finish(recover_unclosed_reasoning_as_content=False))

    content_deltas = [text for field, text in chunks if field == "content"]

    assert content_deltas == ["Count: ", "1, ", "2, ", "3.\n"]


def test_openai_splitter_reasoning_off_suppresses_split_orphan_close_duplicate():
    splitter = _ThinkingContentStreamSplitter(
        thinking_enabled=False,
        recover_unclosed_reasoning_as_content=False,
    )

    chunks = []
    for piece in [
        "I'm doing well, thank you for asking. ",
        "</thi",
        "nks> I'm doing well, thank you for asking.",
    ]:
        chunks.extend(splitter.feed(piece))
    chunks.extend(splitter.finish(recover_unclosed_reasoning_as_content=False))

    content = "".join(text for field, text in chunks if field == "content")

    assert content == "I'm doing well, thank you for asking. "
    assert "</think" not in content
    assert "</thinks" not in content


def test_step3p5_reasoning_off_strips_orphan_close_from_final_text():
    parts = split_reasoning_text(
        "I'm doing well, thank you for asking. </thinks> I'm doing well, thank you for asking.",
        parser="step3p5",
        thinking_enabled=False,
    )

    assert parts.reasoning == ""
    assert parts.content == "I'm doing well, thank you for asking."
    assert "</think" not in parts.content
    assert "</thinks" not in parts.content


def test_step3p5_reasoning_on_routes_plural_close_to_reasoning():
    parts = split_reasoning_text(
        "Check language and answer briefly.</thinks> I'm doing well.",
        parser="step3p5",
        thinking_enabled=True,
    )

    assert parts.reasoning == "Check language and answer briefly."
    assert parts.content == "I'm doing well."


def test_thinking_stream_splitter_can_start_in_visible_content_for_aime():
    splitter = _ThinkingContentStreamSplitter(
        thinking_enabled=True,
        start_inside_thinking=False,
    )

    chunks = []
    chunks.extend(splitter.feed("Visible solution. "))
    chunks.extend(splitter.feed("<think>private check</think>Final answer."))
    chunks.extend(splitter.finish())

    reasoning = "".join(text for field, text in chunks if field == "reasoning_content")
    content = "".join(text for field, text in chunks if field == "content")

    assert reasoning == "private check"
    assert content == "Visible solution. Final answer."


def test_aime_visible_working_requires_explicit_aime_metadata():
    assert (
        _aime_visible_working_for_request(
            {"client": "aime", "aime_visible_working": True}
        )
        is True
    )
    assert (
        _aime_visible_working_for_request(
            {"client": "aime", "aime_visible_working": "on"}
        )
        is True
    )
    assert _aime_visible_working_for_request({"client": "aime"}) is False
    assert (
        _aime_visible_working_for_request(
            {"client": "opencode", "aime_visible_working": True}
        )
        is False
    )


def test_gemma4_stream_splitter_keeps_channel_reasoning_out_of_content():
    splitter = stream_splitter_for_parser("gemma4", thinking_enabled=True)

    chunks = []
    for piece in [
        "<|chan",
        "nel>thought\nUse a compact derivation. ",
        "<channel|>candidate_answer=277\n\\boxed{277}",
    ]:
        chunks.extend(splitter.feed(piece))
    chunks.extend(splitter.finish())

    reasoning = "".join(text for field, text in chunks if field == "reasoning_content")
    content = "".join(text for field, text in chunks if field == "content")

    assert reasoning == "Use a compact derivation. "
    assert content == "candidate_answer=277\n\\boxed{277}"
    assert "<|channel>" not in content
    assert "<channel|>" not in content


def test_gemma4_stream_splitter_strips_channel_markup_when_thinking_disabled():
    splitter = stream_splitter_for_parser("gemma4", thinking_enabled=False)

    chunks = []
    chunks.extend(splitter.feed("<|channel>thought\nhidden<channel|>visible"))
    chunks.extend(splitter.finish())

    reasoning = "".join(text for field, text in chunks if field == "reasoning_content")
    content = "".join(text for field, text in chunks if field == "content")

    assert reasoning == ""
    assert content == "visible"
    assert "<|channel>" not in content
    assert "<channel|>" not in content


def test_nonstream_chat_routes_qwen_thinking_out_of_content():
    state = _postcommit_state()
    state.args.stats_footer = False
    generated = {
        "text": "private math notes</think>\n\n42",
        "stats": {},
    }

    content, reasoning = _nonstream_chat_message_parts(
        state,
        generated,
        thinking_enabled=True,
    )

    assert content == "42"
    assert reasoning == "private math notes"
    assert "<think>" not in content
    assert "</think>" not in content
    assert generated["stats"]["nonstream_reasoning_content_routed"] is True
    assert generated["stats"]["visible_reasoning_stripped"] is True


def test_nonstream_chat_leaves_untagged_content_visible():
    state = _postcommit_state()
    state.args.stats_footer = False
    generated = {"text": "Plain final answer.", "stats": {}}

    content, reasoning = _nonstream_chat_message_parts(
        state,
        generated,
        thinking_enabled=True,
    )

    assert content == "Plain final answer."
    assert reasoning == ""
    assert "nonstream_reasoning_content_routed" not in generated["stats"]


def test_thinking_stream_splitter_can_keep_unclosed_reasoning_private():
    splitter = _ThinkingContentStreamSplitter(
        thinking_enabled=True,
        recover_unclosed_reasoning_as_content=False,
    )

    chunks = []
    chunks.extend(splitter.feed("unfinished private plan"))
    chunks.extend(splitter.finish())

    reasoning = "".join(text for field, text in chunks if field == "reasoning_content")
    content = "".join(text for field, text in chunks if field == "content")

    assert reasoning == "unfinished private plan"
    assert content == ""


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


def test_thinking_stream_splitter_keeps_tool_call_markup_out_of_reasoning():
    splitter = _ThinkingContentStreamSplitter(thinking_enabled=True)

    chunks = []
    chunks.extend(splitter.feed("I should inspect the file.\n\n<tool_call>\n"))
    chunks.extend(splitter.feed("<function=read>\n<parameter=path>\nsrc/Game.ts\n"))
    chunks.extend(splitter.feed("</parameter>\n</function>\n</tool_call>"))
    chunks.extend(splitter.finish())

    reasoning = "".join(text for field, text in chunks if field == "reasoning_content")
    content = "".join(text for field, text in chunks if field == "content")

    assert "I should inspect the file." in reasoning
    assert "<tool_call>" not in reasoning
    assert "<function=read>" not in reasoning
    assert "<tool_call>" in content
    assert "<function=read>" in content


def test_thinking_stream_splitter_keeps_orphan_parameter_markup_out_of_reasoning():
    splitter = _ThinkingContentStreamSplitter(thinking_enabled=True)

    chunks = []
    chunks.extend(splitter.feed("I should inspect the tool result.\n\n<par"))
    chunks.extend(splitter.feed("ameter=keys>\npath\n</parameter>\n"))
    chunks.extend(splitter.feed("</function>\n</tool_call>"))
    chunks.extend(splitter.finish())

    reasoning = "".join(text for field, text in chunks if field == "reasoning_content")
    content = "".join(text for field, text in chunks if field == "content")

    assert "I should inspect the tool result." in reasoning
    assert "<parameter=keys>" not in reasoning
    assert "</parameter>" not in reasoning
    assert "<parameter=keys>" in content
    assert "</tool_call>" in content


def test_thinking_stream_splitter_strips_generated_chat_template_sentinels():
    splitter = _ThinkingContentStreamSplitter(thinking_enabled=True)

    chunks = []
    for piece in ["<|im_sta", "rt|>user sent a follow-up\n"]:
        chunks.extend(splitter.feed(piece))
    chunks.extend(splitter.finish())

    reasoning = "".join(text for field, text in chunks if field == "reasoning_content")
    content = "".join(text for field, text in chunks if field == "content")

    assert reasoning == "sent a follow-up\n"
    assert content == ""
    assert "<|im_start|>" not in reasoning


def test_thinking_stream_splitter_strips_qwen_empty_sentinels_without_poisoning_answer():
    splitter = _ThinkingContentStreamSplitter(thinking_enabled=True)

    chunks = []
    for piece in ["<|third_", "empty|>Final answer"]:
        chunks.extend(splitter.feed(piece))
    chunks.extend(splitter.finish())

    reasoning = "".join(text for field, text in chunks if field == "reasoning_content")
    content = "".join(text for field, text in chunks if field == "content")

    assert reasoning == "Final answer"
    assert content == "Final answer"
    assert "<|third_empty|>" not in reasoning
    assert "<|third_empty|>" not in content


def test_generation_params_exposes_no_server_cap_when_unset(monkeypatch):
    monkeypatch.delenv("MTPLX_UNCAPPED_RESPONSE_LEASE_TOKENS", raising=False)
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
    assert limits["decode_lease_tokens"] == 900
    assert limits["uncapped_response_requested"] is True
    assert limits["uncapped_response_lease_tokens"] is None
    assert limits["uncapped_response_lease_applied"] is False
    assert limits["server_cap_applied"] is False
    assert limits["context_cap_applied"] is False
    assert limits["effective_temperature"] == 0.6
    assert limits["effective_top_p"] == 0.95
    assert limits["effective_top_k"] == 20


def test_generation_params_reports_explicit_effective_sampler(monkeypatch):
    monkeypatch.delenv("MTPLX_UNCAPPED_RESPONSE_LEASE_TOKENS", raising=False)
    state = SimpleNamespace(
        context_window=1000,
        args=SimpleNamespace(
            max_response_tokens=None,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        ),
    )

    _response_max, sampler, limits = _generation_params(
        state,
        prompt_token_count=100,
        max_tokens=None,
        temperature=1.0,
        top_p=0.8,
        top_k=64,
    )

    assert sampler.temperature == 1.0
    assert sampler.top_p == 0.8
    assert sampler.top_k == 64
    assert limits["effective_temperature"] == 1.0
    assert limits["effective_top_p"] == 0.8
    assert limits["effective_top_k"] == 64


def test_generation_params_leaves_large_uncapped_requests_unleased_by_default(
    monkeypatch,
):
    monkeypatch.delenv("MTPLX_UNCAPPED_RESPONSE_LEASE_TOKENS", raising=False)
    state = SimpleNamespace(
        context_window=262144,
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

    assert response_max == 262044
    assert limits["request_max_tokens"] is None
    assert limits["server_max_response_tokens"] is None
    assert limits["effective_max_tokens"] == 262044
    assert limits["decode_lease_tokens"] == 262044
    assert limits["remaining_context_tokens"] == 262044
    assert limits["uncapped_response_requested"] is True
    assert limits["uncapped_response_lease_tokens"] is None
    assert limits["uncapped_response_lease_applied"] is False
    assert limits["server_cap_applied"] is False
    assert limits["context_cap_applied"] is False


def test_generation_params_can_explicitly_lease_uncapped_requests(monkeypatch):
    monkeypatch.setenv("MTPLX_UNCAPPED_RESPONSE_LEASE_TOKENS", "8192")
    state = SimpleNamespace(
        context_window=262144,
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

    assert response_max == 8192
    assert limits["effective_max_tokens"] == 262044
    assert limits["decode_lease_tokens"] == 8192
    assert limits["uncapped_response_requested"] is True
    assert limits["uncapped_response_lease_tokens"] == 8192
    assert limits["uncapped_response_lease_applied"] is True


def test_generation_params_marks_server_cap_when_configured(monkeypatch):
    monkeypatch.delenv("MTPLX_UNCAPPED_RESPONSE_LEASE_TOKENS", raising=False)
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
    assert limits["decode_lease_tokens"] == 384
    assert limits["uncapped_response_requested"] is True
    assert limits["uncapped_response_lease_applied"] is False
    assert limits["server_cap_applied"] is True
    assert limits["context_cap_applied"] is False


def test_uncapped_repetition_stop_only_enables_for_uncapped_requests(monkeypatch):
    monkeypatch.delenv("MTPLX_UNCAPPED_REPETITION_STOP", raising=False)
    assert (
        _uncapped_repetition_stop_enabled(
            {
                "uncapped_response_requested": True,
                "server_max_response_tokens": None,
            }
        )
        is True
    )
    assert (
        _uncapped_repetition_stop_enabled(
            {
                "uncapped_response_requested": False,
                "server_max_response_tokens": None,
            }
        )
        is False
    )
    assert (
        _uncapped_repetition_stop_enabled(
            {
                "uncapped_response_requested": True,
                "server_max_response_tokens": 4096,
            }
        )
        is False
    )
    monkeypatch.setenv("MTPLX_UNCAPPED_REPETITION_STOP", "off")
    assert (
        _uncapped_repetition_stop_enabled(
            {
                "uncapped_response_requested": True,
                "server_max_response_tokens": None,
            }
        )
        is False
    )


def test_repetition_stop_detects_and_trims_exact_token_loop():
    config = RepetitionStopConfig(
        enabled=True,
        min_tokens=12,
        min_repeated_tokens=8,
        min_repeats=4,
        min_block_tokens=2,
        max_block_tokens=4,
    )
    tokens = [101, 102, 103, 104, 7, 8, 7, 8, 7, 8, 7, 8]

    detected = _detect_repeated_token_suffix(tokens, config)

    assert detected is not None
    assert detected.trim_start == 4
    assert detected.block_tokens == 2
    assert detected.repeats == 4
    assert detected.repeated_tokens == 8
    trimmed = list(tokens)
    trimmed_result = _trim_repeated_suffix(trimmed, config)
    assert trimmed_result == detected
    assert trimmed == [101, 102, 103, 104]
    disabled = RepetitionStopConfig(
        enabled=False,
        min_tokens=12,
        min_repeated_tokens=8,
        min_repeats=4,
        min_block_tokens=2,
        max_block_tokens=4,
    )
    assert _detect_repeated_token_suffix(tokens, disabled) is None


def test_metrics_envelope_exposes_repetition_stop_fields():
    envelope = _metrics_envelope(
        stats={
            "repetition_stop_triggered": True,
            "repetition_stop_reason": "exact_repeated_token_suffix",
            "repetition_stop_block_tokens": 6,
            "repetition_stop_repeats": 7,
            "repetition_stop_trimmed_tokens": 42,
            "repetition_stop_raw_tokens": 96,
        },
        prompt_tokens=12,
        completion_tokens=54,
        request_elapsed_s=2.0,
        token_times=[],
        request_started_s=100.0,
        lock_wait_time_s=0.0,
        session_id="session-a",
        session_cache_hit=False,
        cache_miss_reason="new_session",
        session_restore_mode="cold",
        mtp_depth=3,
        generation_limits={"uncapped_repetition_stop_enabled": True},
    )

    assert envelope["repetition_stop_triggered"] is True
    assert envelope["repetition_stop_reason"] == "exact_repeated_token_suffix"
    assert envelope["repetition_stop_block_tokens"] == 6
    assert envelope["repetition_stop_repeats"] == 7
    assert envelope["repetition_stop_trimmed_tokens"] == 42
    assert envelope["repetition_stop_raw_tokens"] == 96
    assert envelope["uncapped_repetition_stop_enabled"] is True


def test_metrics_envelope_exposes_aime_start_windows():
    token_times = [100.0 + (idx * 0.02) for idx in range(300)]
    envelope = _metrics_envelope(
        stats={"generated_tokens": 300, "elapsed_s": 6.0, "prompt_eval_time_s": 0.0},
        prompt_tokens=12,
        completion_tokens=300,
        request_elapsed_s=6.0,
        token_times=token_times,
        request_started_s=99.5,
        lock_wait_time_s=0.0,
        session_id="session-a",
        session_cache_hit=False,
        cache_miss_reason="new_session",
        session_restore_mode="cold",
        mtp_depth=3,
        generation_limits={},
    )

    assert envelope["sliding_decode_tok_s_first_128"] == pytest.approx(50.0)
    assert envelope["sliding_decode_tok_s_first_256"] == pytest.approx(50.0)
    assert envelope["sliding_decode_tok_s_last_128"] == pytest.approx(50.0)
    assert envelope["sliding_decode_tok_s_last_256"] == pytest.approx(50.0)


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
    assert (
        "openai_bridge=omlx_style:preserve_history:parse_at_completion:"
        "tool_digest:v4"
    ) in fingerprint
    assert "tool_contract=none" in fingerprint
    assert 'online_hidden={"alpha":0.25' in fingerprint
    assert '"key":"token"' in fingerprint


def test_adaptive_ev_config_tracks_warmup_and_exploration_in_fingerprint():
    args = parse_args(
        [
            "--adaptive-policy",
            "expected_value",
            "--adaptive-ev-warmup-full-depth-cycles",
            "7",
            "--adaptive-ev-exploration-interval",
            "19",
        ]
    )
    state = SimpleNamespace(
        args=args,
        template_hash="template",
        draft_head_identity="draft",
    )

    config = _adaptive_config(args, max_depth=3)
    fingerprint = _policy_fingerprint(state, thinking_enabled=True)

    assert config["warmup_full_depth_cycles"] == 7
    assert config["exploration_interval"] == 19
    assert '"warmup_full_depth_cycles":7' in fingerprint
    assert '"exploration_interval":19' in fingerprint


def test_adaptive_ev_config_clamps_base_depth_to_request_depth():
    args = parse_args(
        [
            "--adaptive-policy",
            "expected_value",
            "--adaptive-ev-base-depth",
            "2",
        ]
    )

    config = _adaptive_config(args, max_depth=1)

    assert config["max_depth"] == 1
    assert config["min_depth"] == 1
    assert config["base_depth"] == 1
    assert config["configured_base_depth"] == 2


def test_policy_fingerprint_separates_tool_contract_cache_identity():
    args = SimpleNamespace(
        generation_mode="mtp",
        depth=3,
        strip_assistant_reasoning_history=False,
        adaptive_policy="none",
        online_correction_cache=False,
        online_correction_cache_min_depth=1,
        online_correction_cache_key="local_prefix",
        prompt_correction_cache=False,
        prompt_correction_cache_min_depth=2,
        online_hidden_corrector_alpha=0.0,
        online_hidden_corrector_decay=0.8,
        online_hidden_corrector_warmup=1,
        online_hidden_corrector_max_feed_depth=None,
        online_hidden_corrector_key="global",
        tool_prompt_mode="hybrid",
    )
    state = SimpleNamespace(
        args=args,
        template_hash="template",
        draft_head_identity="draft",
    )

    plain = _policy_fingerprint(state, thinking_enabled=True, tools_active=False)
    tools = _policy_fingerprint(state, thinking_enabled=True, tools_active=True)

    assert plain != tools
    assert "tool_contract=none" in plain
    assert "tool_prompt_mode=hybrid" in tools
    from mtplx.server.openai import _MTPLX_TOOL_CONTRACT_POLICY_VERSION

    # The fingerprint must carry the current contract policy version so a
    # contract-text change rotates cache identity (reference the constant,
    # not a literal — the version bumps whenever the contract text changes).
    assert f"tool_contract={_MTPLX_TOOL_CONTRACT_POLICY_VERSION}" in tools
    native = _policy_fingerprint(
        state,
        thinking_enabled=True,
        tools_active=True,
        tool_prompt_mode="native",
    )
    assert native != tools
    assert "tool_prompt_mode=native" in native
    assert "tool_contract=native_template_tools:agent_tail:v7" in native
    assert (
        "openai_bridge=omlx_style:preserve_history:parse_at_completion:"
        "tool_digest:v4"
    ) in tools


def test_opencode_session_cache_scope_is_launch_bound():
    args = SimpleNamespace(
        generation_mode="mtp",
        depth=3,
        strip_assistant_reasoning_history=False,
        adaptive_policy="none",
        online_correction_cache=False,
        online_correction_cache_min_depth=1,
        online_correction_cache_key="local_prefix",
        prompt_correction_cache=False,
        prompt_correction_cache_min_depth=2,
        online_hidden_corrector_alpha=0.0,
        online_hidden_corrector_decay=0.8,
        online_hidden_corrector_warmup=1,
        online_hidden_corrector_max_feed_depth=None,
        online_hidden_corrector_key="global",
        tool_prompt_mode="hybrid",
        app_launch_id="launch-a",
    )
    state = SimpleNamespace(
        args=args,
        template_hash="template",
        draft_head_identity="draft",
    )

    stable_scope = _session_cache_scope_for_request(state, headers={}, metadata={})
    opencode_scope = _session_cache_scope_for_request(
        state,
        headers={"x-mtplx-client": "opencode"},
        metadata={},
    )
    stable = _policy_fingerprint(
        state,
        thinking_enabled=True,
        tools_active=True,
        cache_scope=stable_scope,
    )
    scoped = _policy_fingerprint(
        state,
        thinking_enabled=True,
        tools_active=True,
        cache_scope=opencode_scope,
    )

    assert stable_scope == "stable"
    assert opencode_scope == "opencode_process_cache:v1:launch-a"
    assert stable != scoped
    assert "cache_scope=" not in stable
    assert "cache_scope=opencode_process_cache:v1:launch-a" in scoped


def test_opencode_tool_history_forces_clone_restore_without_default_bypass(monkeypatch):
    monkeypatch.delenv("MTPLX_OPENCODE_TOOL_HISTORY_SESSIONBANK_BYPASS", raising=False)
    monkeypatch.delenv("MTPLX_OPENCODE_TOOL_HISTORY_LIVE_FRONTIER", raising=False)
    assert _should_force_clone_session_cache_for_opencode_tool_history(
        headers={"x-mtplx-client": "opencode"},
        metadata={},
        tool_result_history_present=True,
    )
    assert not _should_force_clone_session_cache_for_opencode_tool_history(
        headers={"x-mtplx-client": "opencode"},
        metadata={},
        tool_result_history_present=False,
    )
    assert not _should_force_clone_session_cache_for_opencode_tool_history(
        headers={},
        metadata={},
        tool_result_history_present=True,
    )
    assert not _should_bypass_session_cache_for_opencode_tool_history(
        headers={"x-mtplx-client": "opencode"},
        metadata={},
        tool_result_history_present=True,
    )

    monkeypatch.setenv("MTPLX_OPENCODE_TOOL_HISTORY_SESSIONBANK_BYPASS", "1")
    assert _should_bypass_session_cache_for_opencode_tool_history(
        headers={"x-mtplx-client": "opencode"},
        metadata={},
        tool_result_history_present=True,
    )


def test_opencode_tool_history_live_frontier_restore_beats_clone(monkeypatch):
    monkeypatch.delenv("MTPLX_OPENCODE_TOOL_HISTORY_SESSIONBANK_BYPASS", raising=False)
    monkeypatch.setenv("MTPLX_OPENCODE_TOOL_HISTORY_LIVE_FRONTIER", "1")

    policy = _opencode_tool_history_restore_policy(
        headers={"x-mtplx-client": "opencode"},
        metadata={},
        tool_result_history_present=True,
    )

    assert policy == {
        "eligible": True,
        "cache_bypass": False,
        "live_frontier_restore": True,
        "force_clone_restore": False,
    }


def test_opencode_tool_history_explicit_bypass_beats_live_frontier(monkeypatch):
    monkeypatch.setenv("MTPLX_OPENCODE_TOOL_HISTORY_SESSIONBANK_BYPASS", "1")
    monkeypatch.setenv("MTPLX_OPENCODE_TOOL_HISTORY_LIVE_FRONTIER", "1")

    policy = _opencode_tool_history_restore_policy(
        headers={"x-mtplx-client": "opencode"},
        metadata={},
        tool_result_history_present=True,
    )

    assert policy == {
        "eligible": True,
        "cache_bypass": True,
        "live_frontier_restore": False,
        "force_clone_restore": False,
    }


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
