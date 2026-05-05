from concurrent.futures import Future
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.profiles import get_profile
from mtplx.server import openai
from mtplx.server.openai import _RateLimiter, create_app, parse_args


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
            tokenizer=SimpleNamespace(),
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
    # Stall watchdog so the UI surfaces a real error instead of parking on
    # "Thinking" forever when the server hangs (also user-reported).
    assert "armStallWatchdog" in root.text
    assert "no response from server" in root.text
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
    assert client.get("/v1/models", headers={"Authorization": "Bearer test-key"}).status_code == 200

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


def test_chat_tools_are_passed_to_qwen_template_and_disable_default_thinking(monkeypatch):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_run_generation", lambda *_args, **_kwargs: _fake_generation("ok"))

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
            "stream": True,
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    assert '"tool_calls"' in response.text
    assert '"finish_reason": "tool_calls"' in response.text
    assert "<tool_call>" not in response.text


def test_chat_tools_malformed_tool_call_returns_422(monkeypatch):
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

    assert response.status_code == 422
    assert response.json()["detail"].startswith("malformed tool_call")


def test_server_state_emits_startup_progress(monkeypatch, capsys):
    monkeypatch.setattr(openai, "apply_profile_env", lambda _profile: None)
    monkeypatch.setattr(openai, "profile_env_status", lambda _profile: {})
    monkeypatch.setattr(openai, "_fast_path_env_status", lambda: {})
    monkeypatch.setattr(openai, "_mlx_fork_status", lambda: {"ok": True})
    monkeypatch.setattr(openai, "_configure_mlx_cache_limit", lambda _args: {"configured": False})
    monkeypatch.setattr(
        openai,
        "load",
        lambda model, mtp, contract: SimpleNamespace(
            model_path=Path(model),
            mtp_enabled=mtp,
            tokenizer=SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(openai, "_install_draft_lm_head", lambda *_args, **_kwargs: {"installed": True})
    monkeypatch.setattr(openai, "_draft_head_identity", lambda _runtime: "draft-head")
    monkeypatch.setattr(openai, "_template_hash", lambda _tokenizer: "template")
    monkeypatch.setattr(openai, "_resolve_context_window", lambda _tokenizer, _model: 32768)
    monkeypatch.setattr(openai, "EngineSessionManager", lambda: SimpleNamespace())

    args = parse_args(["--model", "models/example", "--warmup-tokens", "0"])
    state = openai.ServerState(args)

    captured = capsys.readouterr().out
    assert "[4/6] Preparing Medium MTP runtime" in captured
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
    monkeypatch.setattr(openai, "_configure_mlx_cache_limit", lambda _args: {"configured": False})

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
