from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
import time
from threading import Lock
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.profiles import get_profile
from mtplx.server import openai
from mtplx.server.openai import _RateLimiter, create_app, parse_args


def test_runtime_mode_label_distinguishes_sustained_max_and_burst():
    assert (
        openai._health_runtime_mode_label(
            "sustained", "mtp", fan_boost_active=False
        )
        == "Sustained MTP"
    )
    assert (
        openai._health_runtime_mode_label(
            "sustained", "mtp", fan_boost_active=True
        )
        == "Sustained Max MTP"
    )
    assert (
        openai._health_runtime_mode_label(
            "performance-cold", "mtp", fan_boost_active=True
        )
        == "Burst MTP"
    )
    assert (
        openai._health_runtime_mode_label(
            "sustained", "ar", fan_boost_active=False
        )
        == "Sustained AR"
    )


class FakeExecutor:
    def submit(self, fn, *args, **kwargs):
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # pragma: no cover - surfaced by caller
            future.set_exception(exc)
        return future

    def shutdown(self, **_kwargs):
        return None


class StreamingTokenizer:
    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, **_kwargs
    ):
        assert tokenize is True
        text = "\n".join(
            f"{message['role']}:{message.get('content') or ''}" for message in messages
        )
        if add_generation_prompt:
            text = f"{text}\nassistant:" if text else "assistant:"
        return [ord(char) for char in text]

    def encode(self, text):
        return [ord(char) for char in str(text)]

    def decode(self, tokens, **_kwargs):
        return "".join(chr(int(token)) for token in tokens)


class RecordingBank:
    def __init__(self):
        self.puts: list[dict] = []

    def put(self, **kwargs):
        self.puts.append(kwargs)
        return SimpleNamespace(
            prefix_len=len(kwargs["token_ids"]),
            nbytes=123,
            token_hash="test-token-hash",
        )


class ForegroundState:
    def __init__(self) -> None:
        self.lock = Lock()
        self.foreground_active = 0
        self.last_request_started_at = 0.0

    def begin_foreground(self) -> None:
        self.foreground_active += 1

    def end_foreground(self) -> None:
        self.foreground_active = max(0, self.foreground_active - 1)

    def has_foreground(self) -> bool:
        return self.foreground_active > 0

    def foreground_count(self) -> int:
        return self.foreground_active


def _fake_state(*, api_key: str | None = None, rate_limit: int = 0):
    argv = ["--warmup-tokens", "0", "--rate-limit", str(rate_limit)]
    if api_key:
        argv.extend(["--api-key", api_key])
    args = parse_args(argv)
    return SimpleNamespace(
        args=args,
        model_id="mtplx-test-model",
        runtime=SimpleNamespace(
            model_path=Path("models/example"),
            mtp_enabled=True,
            tokenizer=SimpleNamespace(
                decode=lambda tokens, **_kwargs: "".join(
                    chr(int(token)) for token in tokens
                ),
            ),
        ),
        profile=get_profile(args.profile),
        context_window=4096,
        load_time_s=0.25,
        draft_lm_head={"installed": False, "reason": "test"},
        draft_head_identity="test-head",
        template_hash="test-template",
        main_system_prompt_hash=None,
        fast_path_env_status={},
        profile_env_status={},
        mlx_cache_limit_status={"configured": False},
        mlx_fork_status={"ok": False},
        warmup_status={"enabled": False, "ran": False, "tokens": 0},
        last_metrics=[{"tok_s": 12.5, "accept_rate": 0.75}],
        rate_limiter=_RateLimiter(rate_limit),
        sessions=SimpleNamespace(
            list_sessions=lambda: {"sessions": []},
            clear_session=lambda session_id: {"cleared": session_id},
            clear_all=lambda: {"cleared": True},
        ),
        generation_executor=FakeExecutor(),
    )


def _fake_streaming_session_state():
    state = _fake_state()
    foreground = ForegroundState()
    state.lock = foreground.lock
    state.begin_foreground = foreground.begin_foreground
    state.end_foreground = foreground.end_foreground
    state.has_foreground = foreground.has_foreground
    state.foreground_count = foreground.foreground_count
    state.runtime.tokenizer = StreamingTokenizer()
    state.sessions = openai.EngineSessionManager(bank=RecordingBank())
    state.generation_executor = ThreadPoolExecutor(max_workers=1)
    state.postcommit_executor = FakeExecutor()
    state.args.stats_footer = False
    return state


def _fake_final_state(tokens):
    return SimpleNamespace(
        final_trunk_cache=["cache"],
        final_logits="logits",
        final_hidden="hidden",
        final_committed_mtp_cache=None,
        generated_token_ids=tuple(tokens),
        safe_to_commit=True,
        finish_reason="stop",
    )


def test_mtplx_settings_endpoint_controls_server_reasoning():
    state = _fake_state(api_key="mtplx-local")
    client = TestClient(create_app(state))

    initial = client.get(
        "/v1/mtplx/settings", headers={"Authorization": "Bearer mtplx-local"}
    )
    assert initial.status_code == 200
    assert initial.json()["reasoning"] == "auto"

    off = client.post(
        "/v1/mtplx/settings",
        json={"reasoning": "off"},
        headers={"Authorization": "Bearer mtplx-local"},
    )
    assert off.status_code == 200
    assert off.json()["reasoning"] == "off"
    assert off.json()["enable_thinking"] is False
    assert state.args.enable_thinking is False

    on = client.post(
        "/v1/mtplx/settings",
        json={"reasoning": "on"},
        headers={"Authorization": "Bearer mtplx-local"},
    )
    assert on.status_code == 200
    assert on.json()["reasoning"] == "on"
    assert on.json()["enable_thinking"] is True
    assert state.args.enable_thinking is True


def test_server_console_controls_reasoning_and_mtp_defaults():
    state = _fake_state()

    assert "Reasoning: off" in openai._server_console_handle_command(
        state, "/reasoning off"
    )
    assert state.args.reasoning == "off"
    assert state.args.enable_thinking is False

    assert "Reasoning: auto" in openai._server_console_handle_command(
        state, "/reasoning auto"
    )
    assert state.args.reasoning == "auto"
    assert state.args.enable_thinking is True

    assert "MTP: off" in openai._server_console_handle_command(state, "/mtp off")
    assert state.args.generation_mode == "ar"

    assert "MTP: on" in openai._server_console_handle_command(state, "/mtp on")
    assert state.args.generation_mode == "mtp"


def test_openai_server_health_metrics_and_models_fake_state():
    client = TestClient(create_app(_fake_state()))

    root_head = client.head("/")
    assert root_head.status_code == 200

    root = client.get("/")
    assert root.status_code == 200
    assert "text/html" in root.headers["content-type"]
    # Brand and chat scaffold
    assert "<title>MTPLX</title>" in root.text
    assert "MTPLX" in root.text
    assert 'id="messages"' in root.text
    assert 'id="prompt"' in root.text
    assert "Message MTPLX" in root.text
    assert "/v1/chat/completions" in root.text
    assert "reasoning_content" in root.text
    # Inference settings sidebar — all sliders now
    assert 'id="ctl-temp"' in root.text
    assert 'id="ctl-top-p"' in root.text
    assert 'id="ctl-top-k" type="range"' in root.text
    assert 'id="ctl-mtp" type="checkbox"' in root.text
    assert 'id="ctl-depth" type="range"' in root.text
    assert 'id="ctl-max-tokens" type="range"' in root.text
    assert 'id="ctl-system"' in root.text
    assert 'id="reset-defaults"' in root.text
    # New layout: avatar circles + reasoning-as-its-own-block + turn-* classes
    assert "turn turn-assistant" in root.text
    assert 'class="avatar"' in root.text
    assert "reasoning-block" in root.text
    # Auto-scroll, stop, new-chat, persistence
    assert 'id="jump-pill"' not in root.text
    assert 'id="messages-bottom"' in root.text
    assert "ResizeObserver" in root.text
    assert "scrollIntoView" in root.text
    assert "forceAutoScroll" in root.text
    assert "SCROLL_PIN_THRESHOLD = 160" in root.text
    assert 'id="new-chat-btn"' in root.text
    assert "AbortController" in root.text
    # SETTINGS_KEY bumped to v4 when MTP on/off settings landed; bumping the
    # version invalidates saved sidebar settings without a generation-mode bit.
    assert "mtplx.chat.settings.v4" in root.text
    # Auto-detect of context length must be hooked up so the slider isn't
    # capped at a stale 32k for a 256k-context model.
    assert "discoverServerLimits" in root.text
    assert "/health" in root.text
    # Transport watchdog plus heartbeat handling keep long active generations
    # from looking like crashed servers.
    assert "armStallWatchdog" in root.text
    assert "mtplx_progress.heartbeat" in root.text
    assert "Still working" in root.text
    assert "stream connection went quiet" in root.text
    assert "server has crashed" not in root.text
    # Markdown via marked.js
    assert "marked.min.js" in root.text
    # Live tps element
    assert 'id="live-stats"' in root.text
    assert "tok/s" in root.text
    assert '"mtp_enabled": true' in root.text
    assert 'generation_mode: settingsNow.mtp_enabled ? "mtp" : "ar"' in root.text
    assert '"depth": 3' in root.text
    assert 'id="ctl-depth" type="range" min="1" max="3" step="1" value="3"' in root.text
    # Updated max-tokens default and cap.
    assert 'value="8192"' in root.text
    assert 'min="256" max="32768"' in root.text

    v1 = client.get("/v1")
    assert v1.status_code == 200
    assert v1.json()["openwebui"]["base_url"].endswith("/v1")
    assert "Paste this URL into Open WebUI" in v1.json()["message"]

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model"] == "mtplx-test-model"
    assert health.json()["api_key_required"] is False
    assert health.json()["warmup"]["ran"] is False
    assert health.json()["foreground_active"] == 0
    assert health.json()["active_requests"] == 0
    assert health.json()["last_request_started_at"] == 0.0

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["latest"]["tok_s"] == 12.5

    models = client.get("/v1/models")
    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "mtplx-test-model"


def test_openai_server_auth_and_rate_limit_fake_state():
    client = TestClient(create_app(_fake_state(api_key="test-key", rate_limit=1)))

    assert client.get("/v1/models").status_code == 401
    assert (
        client.get(
            "/v1/models", headers={"Authorization": "Bearer test-key"}
        ).status_code
        == 200
    )

    limited = client.get("/v1/models", headers={"Authorization": "Bearer test-key"})
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) > 0


def test_anthropic_messages_rejects_empty_request_before_generation():
    client = TestClient(create_app(_fake_state()))

    response = client.post(
        "/v1/messages",
        json={
            "model": "mtplx-test-model",
            "max_tokens": 8,
            "stream": True,
            "messages": [],
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "messages must not be empty"


def test_chat_ui_uses_server_depth_default():
    state = _fake_state()
    state.args.depth = 2
    client = TestClient(create_app(state))

    root = client.get("/")

    assert root.status_code == 200
    assert '"depth": 2' in root.text
    assert '"mtp_enabled": true' in root.text
    assert 'id="ctl-depth" type="range" min="1" max="2" step="1" value="2"' in root.text


def test_chat_generation_mode_request_override_routes_ar(monkeypatch):
    captured: dict[str, object] = {}
    state = _fake_state()
    client = TestClient(create_app(state))

    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])

    def fake_run_generation(_state, prompt_ids, **kwargs):
        captured["prompt_ids"] = prompt_ids
        captured["generation_mode"] = kwargs["generation_mode"]
        captured["depth"] = kwargs["depth"]
        return {
            "text": "ok",
            "tokens": [4],
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 1,
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 1,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Say READY"}],
            "max_tokens": 4,
            "generation_mode": "ar",
            "depth": 3,
        },
    )

    assert response.status_code == 200
    assert captured["generation_mode"] == "ar"
    assert captured["depth"] == 0
    assert response.json()["mtplx_stats"]["generation_mode"] == "ar"
    assert response.json()["mtplx_stats"]["mtp_depth"] == 0


def test_generation_truth_stats_distinguish_stock_ar_from_target_ar():
    target_state = _fake_state()
    target_state.args.load_mtp = True
    target_state.runtime.mtp_enabled = True
    target_state.draft_lm_head = {"draft_only": {"bits": 4}}

    stock_state = _fake_state()
    stock_state.args.load_mtp = False
    stock_state.runtime.mtp_enabled = False

    target = openai._generation_truth_stats(target_state, "ar")
    assert target["benchmark_mode"] == "mtplx_mtp_loaded_target_ar"
    assert target["draft_head_installed"] is True
    stock = openai._generation_truth_stats(stock_state, "ar")
    assert stock["benchmark_mode"] == "mtplx_stock_ar_unloaded"
    assert stock["load_mtp"] is False
    assert stock["runtime_mtp_enabled"] is False


def test_chat_generation_mode_request_override_routes_mtp_depth(monkeypatch):
    captured: dict[str, object] = {}
    client = TestClient(create_app(_fake_state()))

    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        captured["generation_mode"] = kwargs["generation_mode"]
        captured["depth"] = kwargs["depth"]
        return {
            "text": "ok",
            "tokens": [4],
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 1,
            },
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Say READY"}],
            "max_tokens": 4,
            "generation_mode": "mtp",
            "depth": 1,
        },
    )

    assert response.status_code == 200
    assert captured == {"generation_mode": "mtp", "depth": 1}
    assert response.json()["mtplx_stats"]["generation_mode"] == "mtp"
    assert response.json()["mtplx_stats"]["mtp_depth"] == 1


def test_streaming_session_uses_generation_final_postcommit_without_retokenized_tail(
    monkeypatch,
):
    state = _fake_streaming_session_state()
    captured: dict[str, object] = {}

    def fail_retokenized(*_args, **_kwargs):
        raise AssertionError(
            "streaming fast path must not retokenize/prefill postcommit"
        )

    def fake_run_generation(_state, prompt_ids, **kwargs):
        captured.setdefault(
            "commit_final_state_to_bank",
            kwargs.get("commit_final_state_to_bank"),
        )
        token_callback = kwargs.get("token_callback")
        tokens = [ord("O"), ord("K")]
        if token_callback is not None:
            token_callback(tokens[:1])
            token_callback(tokens[1:])
        return {
            "text": "OK",
            "tokens": tokens,
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 2,
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 2,
            "finish_reason": "stop",
            "_final_state": _fake_final_state(tokens),
        }

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fail_retokenized)
    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    with TestClient(create_app(state)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-session-id": "stream-session"},
            json={
                "messages": [{"role": "user", "content": "Say OK"}],
                "enable_thinking": False,
                "stream": True,
                "max_tokens": 4,
            },
        )
        second = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-session-id": "stream-session"},
            json={
                "messages": [
                    {"role": "user", "content": "Say OK"},
                    {"role": "assistant", "content": "OK"},
                    {"role": "user", "content": "Again"},
                ],
                "enable_thinking": False,
                "max_tokens": 4,
            },
        )

    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    assert '"content": "OK"' in response.text or (
        '"content": "O"' in response.text and '"content": "K"' in response.text
    )
    assert '"mode": "generation_final_exact"' in response.text
    assert captured["commit_final_state_to_bank"] is False
    assert second.status_code == 200
    assert "already in flight" not in second.text


def test_streaming_unsafe_postcommit_releases_without_blocking_second_request(
    monkeypatch,
):
    state = _fake_streaming_session_state()
    scheduled: list[dict] = []

    def fake_store_generation_final(*_args, **_kwargs):
        return {
            "stored": False,
            "mode": "unsafe",
            "reason": "retokenized_history_mismatch",
        }

    def fake_schedule(*_args, **kwargs):
        scheduled.append(kwargs)
        return {
            "stored": False,
            "mode": "async_pending",
            "reason": kwargs["unsafe_reason"],
        }

    def fake_run_generation(_state, prompt_ids, **kwargs):
        token_callback = kwargs.get("token_callback")
        tokens = [ord("O"), ord("K")]
        if token_callback is not None:
            token_callback(tokens)
        return {
            "text": "OK",
            "tokens": tokens,
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 2,
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 2,
            "finish_reason": "stop",
            "_final_state": _fake_final_state(tokens),
        }

    monkeypatch.setattr(
        openai, "_store_generation_final_history_snapshot", fake_store_generation_final
    )
    monkeypatch.setattr(openai, "_schedule_idle_postcommit_snapshot", fake_schedule)
    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    with TestClient(create_app(state)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-session-id": "unsafe-session"},
            json={
                "messages": [{"role": "user", "content": "Say OK"}],
                "enable_thinking": False,
                "stream": True,
                "max_tokens": 4,
            },
        )
        second = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-session-id": "unsafe-session"},
            json={
                "messages": [{"role": "user", "content": "Say OK again"}],
                "enable_thinking": False,
                "stream": True,
                "max_tokens": 4,
            },
        )

    assert response.status_code == 200
    assert '"mode": "async_pending"' in response.text
    assert '"reason": "retokenized_history_mismatch"' in response.text
    assert scheduled
    assert second.status_code == 200
    assert "already in flight" not in second.text


def test_streaming_ar_keeps_retokenized_postcommit_path(monkeypatch):
    state = _fake_streaming_session_state()
    retokenized_calls: list[dict] = []

    def fake_retokenized(*_args, **kwargs):
        retokenized_calls.append(kwargs)
        return {
            "stored": True,
            "mode": "retokenized_history",
            "prefix_len": 32,
            "nbytes": 99,
        }

    def fake_run_generation(_state, prompt_ids, **kwargs):
        token_callback = kwargs.get("token_callback")
        tokens = [ord("A"), ord("R")]
        if token_callback is not None:
            token_callback(tokens)
        return {
            "text": "AR",
            "tokens": tokens,
            "stats": {
                "generation_mode": "ar",
                "mtp_depth": 0,
                "completion_tokens": 2,
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 2,
            "finish_reason": "stop",
            "_final_state": None,
        }

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fake_retokenized)
    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    with TestClient(create_app(state)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-session-id": "ar-session"},
            json={
                "messages": [{"role": "user", "content": "Say AR"}],
                "enable_thinking": False,
                "generation_mode": "ar",
                "stream": True,
                "max_tokens": 4,
            },
        )

    assert response.status_code == 200
    assert retokenized_calls
    assert '"mode": "retokenized_history"' in response.text
    assert '"generation_mode": "ar"' in response.text


def test_invalid_generation_mode_returns_400():
    client = TestClient(create_app(_fake_state()))

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Say READY"}],
            "generation_mode": "off",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "generation_mode must be 'mtp' or 'ar'"


class CaptureTokenizer:
    def __init__(self):
        self.calls: list[tuple[list[dict[str, object]], dict[str, object]]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [1, 2, 3]

    def encode(self, text):
        return [ord(char) for char in str(text)]


def _tool_schema():
    return {
        "type": "function",
        "function": {
            "name": "session_status",
            "description": "Show the current agent session status.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }


def _add_tool_schema():
    return {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    }


def _fake_generation(text: str):
    return {
        "text": text,
        "tokens": [4],
        "stats": {
            "generation_mode": "ar",
            "mtp_depth": 0,
            "completion_tokens": 1,
        },
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "finish_reason": "stop",
    }


def _fake_streaming_generation(text: str):
    tokens = [ord(char) for char in text]

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        token_callback = kwargs.get("token_callback")
        if token_callback is not None:
            for token in tokens:
                token_callback([token])
        return {
            "text": text,
            "tokens": tokens,
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": len(tokens),
            },
            "prompt_tokens": 3,
            "completion_tokens": len(tokens),
            "finish_reason": "stop",
        }

    return fake_run_generation


def _stream_payloads(response_text: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in response_text.splitlines()
        if line.startswith("data: {")
    ]


def test_chat_tools_are_passed_to_qwen_template_and_disable_default_thinking(
    monkeypatch,
):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    client = TestClient(create_app(state))
    monkeypatch.setattr(
        openai, "_run_generation", lambda *_args, **_kwargs: _fake_generation("ok")
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Use the tool."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    _messages, kwargs = state.runtime.tokenizer.calls[0]
    assert kwargs["tools"] == [_tool_schema()]
    assert kwargs["enable_thinking"] is False


def test_chat_template_preserves_assistant_tool_calls_and_tool_results():
    tokenizer = CaptureTokenizer()

    openai._encode_messages(
        tokenizer,
        [
            openai.ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    {
                        "id": "call_test",
                        "type": "function",
                        "function": {
                            "name": "session_status",
                            "arguments": "{}",
                        },
                    }
                ],
            ),
            openai.ChatMessage(
                role="tool",
                tool_call_id="call_test",
                content='{"status":"ok"}',
            ),
        ],
        enable_thinking=False,
        add_generation_prompt=False,
    )

    messages, _kwargs = tokenizer.calls[0]
    assert messages[0]["role"] == "assistant"
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {}
    assert messages[1]["role"] == "tool"
    assert messages[1]["tool_call_id"] == "call_test"


def test_chat_tool_xml_returns_openai_tool_calls_nonstream(monkeypatch):
    client = TestClient(create_app(_fake_state()))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        lambda *_args, **_kwargs: _fake_generation(
            "<tool_call>\n<function=session_status>\n</function>\n</tool_call>"
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Status."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"][0]["function"] == {
        "name": "session_status",
        "arguments": "{}",
    }
    assert "<tool_call>" not in json.dumps(payload)


def test_chat_tool_json_returns_openai_tool_calls_nonstream(monkeypatch):
    client = TestClient(create_app(_fake_state()))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        lambda *_args, **_kwargs: _fake_generation(
            '<tool_call>{"name":"session_status","arguments":{}}</tool_call>'
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Status."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_chat_stream_tool_calls_emit_delta_tool_calls(monkeypatch):
    client = TestClient(create_app(_fake_state()))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation(
            "<tool_call>\n<function=session_status>\n</function>\n</tool_call>"
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Status."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    assert '"tool_calls"' in response.text
    assert '"finish_reason": "tool_calls"' in response.text
    assert "<tool_call>" not in response.text


def test_chat_stream_tools_plain_content_stays_incremental(monkeypatch):
    state = _fake_state()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation("Count: 1, 2, 3.\n"),
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Count."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 16,
            "enable_thinking": False,
        },
    )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    content_deltas = [
        payload["choices"][0]["delta"]["content"]
        for payload in payloads
        if payload["choices"][0]["delta"].get("content")
    ]
    assert len(content_deltas) > 1
    assert "".join(content_deltas) == "Count: 1, 2, 3.\n"
    assert not any(
        payload["choices"][0]["delta"].get("tool_calls") for payload in payloads
    )
    assert any(
        payload["choices"][0].get("finish_reason") == "stop" for payload in payloads
    )


def test_chat_stream_tool_call_arguments_are_incremental(monkeypatch):
    state = _fake_state()
    state.args.stream_interval = 1
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation(
            '<tool_call>{"name":"add","arguments":{"a":25,"b":17}}</tool_call>'
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Use add."}],
            "tools": [_add_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    tool_deltas = [
        payload["choices"][0]["delta"]["tool_calls"][0]
        for payload in payloads
        if payload["choices"][0]["delta"].get("tool_calls")
    ]
    assert tool_deltas[0]["id"].startswith("call_")
    assert tool_deltas[0]["type"] == "function"
    assert tool_deltas[0]["function"] == {"name": "add", "arguments": ""}

    argument_fragments = [
        delta["function"]["arguments"]
        for delta in tool_deltas
        if delta.get("function", {}).get("arguments") is not None
    ]
    assert "".join(argument_fragments) == '{"a":25,"b":17}'
    assert len([fragment for fragment in argument_fragments if fragment]) > 1
    assert '{"a":25,"b":17}' not in argument_fragments


def test_chat_stream_emits_heartbeat_during_alive_silence(monkeypatch):
    state = _fake_state()
    state.args.stats_footer = False
    state.generation_executor = ThreadPoolExecutor(max_workers=1)
    client = TestClient(create_app(state))
    tokens = [ord("o"), ord("k"), ord("\n")]

    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(openai, "STREAM_HEARTBEAT_INTERVAL_S", 0.0)
    monkeypatch.setattr(openai, "STREAM_SILENCE_WARN_S", 0.01)
    monkeypatch.setattr(openai, "STREAM_SILENCE_WARN_INTERVAL_S", 60.0)

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        # Leave enough wall time for the ASGI stream loop to observe an empty
        # token queue even on slower GitHub macOS runners.
        time.sleep(1.25)
        kwargs["token_callback"](tokens)
        return {
            "text": "ok\n",
            "tokens": tokens,
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": len(tokens),
            },
            "prompt_tokens": 3,
            "completion_tokens": len(tokens),
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    try:
        response = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-cache-mode": "bypass"},
            json={
                "messages": [{"role": "user", "content": "Say ok."}],
                "stream": True,
                "max_tokens": 16,
                "enable_thinking": False,
            },
        )
    finally:
        state.generation_executor.shutdown(wait=True)

    assert response.status_code == 200
    payloads = []
    for event in response.text.split("\n\n"):
        if not event.startswith("data: {"):
            continue
        payloads.append(json.loads(event.removeprefix("data: ")))

    heartbeats = [
        payload
        for payload in payloads
        if payload.get("mtplx_progress", {}).get("heartbeat") is True
    ]
    content = "".join(
        payload["choices"][0].get("delta", {}).get("content", "")
        for payload in payloads
    )
    final_chunks = [
        payload
        for payload in payloads
        if payload["choices"][0].get("finish_reason") == "stop"
    ]

    assert heartbeats
    assert heartbeats[0]["mtplx_progress"]["phase"] == "generating"
    assert heartbeats[0]["mtplx_progress"]["completion_tokens"] == 0
    assert content == "ok\n"
    assert final_chunks
    assert "data: [DONE]" in response.text


def test_chat_tools_malformed_tool_call_falls_back_to_content(monkeypatch):
    """Malformed tool-call markup should fall back to plain content, not 422."""
    client = TestClient(create_app(_fake_state()))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        lambda *_args, **_kwargs: _fake_generation("<tool_call>not json</tool_call>"),
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Status."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] != "tool_calls"
    assert choice["message"]["content"] is not None


def test_server_state_emits_startup_progress(monkeypatch, capsys):
    monkeypatch.setattr(openai, "apply_profile_env", lambda _profile: None)
    monkeypatch.setattr(openai, "profile_env_status", lambda _profile: {})
    monkeypatch.setattr(openai, "_fast_path_env_status", lambda: {})
    monkeypatch.setattr(openai, "_mlx_fork_status", lambda: {"ok": True})
    monkeypatch.setattr(
        openai, "_configure_mlx_cache_limit", lambda _args: {"configured": False}
    )
    monkeypatch.setattr(
        openai,
        "load",
        lambda model, mtp, contract: SimpleNamespace(
            model_path=Path(model),
            mtp_enabled=mtp,
            tokenizer=SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        openai, "_install_draft_lm_head", lambda *_args, **_kwargs: {"installed": True}
    )
    monkeypatch.setattr(openai, "_draft_head_identity", lambda _runtime: "draft-head")
    monkeypatch.setattr(openai, "_template_hash", lambda _tokenizer: "template")
    monkeypatch.setattr(
        openai, "_resolve_context_window", lambda _tokenizer, _model: 32768
    )
    monkeypatch.setattr(openai, "EngineSessionManager", lambda: SimpleNamespace())

    args = parse_args(["--model", "models/example", "--warmup-tokens", "0"])
    state = openai.ServerState(args)

    captured = capsys.readouterr().out
    assert "[4/6] Preparing Sustained MTP runtime" in captured
    assert "[5/6] Loading model weights: models/example" in captured
    assert "This is the long step" in captured
    assert "Model load in progress (this may take a minute)" in captured
    assert "[5/6] Model loaded" in captured
    assert "[6/6] Warmup skipped" in captured
    assert state.context_window == 32768


def test_server_state_reports_model_load_failure(monkeypatch, capsys):
    monkeypatch.setattr(openai, "apply_profile_env", lambda _profile: None)
    monkeypatch.setattr(openai, "profile_env_status", lambda _profile: {})
    monkeypatch.setattr(openai, "_fast_path_env_status", lambda: {})
    monkeypatch.setattr(openai, "_mlx_fork_status", lambda: {"ok": True})
    monkeypatch.setattr(
        openai, "_configure_mlx_cache_limit", lambda _args: {"configured": False}
    )

    def fail_load(model, mtp, contract):
        assert model == "models/example"
        assert mtp is True
        assert contract is not None
        raise RuntimeError("boom")

    monkeypatch.setattr(openai, "load", fail_load)

    args = parse_args(["--model", "models/example", "--warmup-tokens", "0"])
    with pytest.raises(RuntimeError, match="boom"):
        openai.ServerState(args)

    captured = capsys.readouterr().out
    assert "[5/6] Model load failed" in captured
    assert "RuntimeError: boom" in captured
