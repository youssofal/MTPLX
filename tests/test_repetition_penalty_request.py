"""``repetition_penalty`` is refused instead of silently dropped.

The request models are ``extra="allow"``, so the vLLM/HF-style field used to
parse on every endpoint and never reach the sampler. With client sampler
controls applied, any value other than the no-op 1.0 is now a 400; with
server-owned controls it is listed with the other ignored sampler fields.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.server import openai
from mtplx.server.openai import create_app
from tests.test_server_openai import _fake_state


def _client(monkeypatch, captured):
    def fake_run_generation(_state, prompt_ids, **kwargs):
        captured["called"] = True
        captured["request_observability"] = kwargs.get("request_observability")
        return {
            "text": "ok",
            "tokens": [4],
            "stats": {"completion_tokens": 1},
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 1,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_encode_messages", lambda *_a, **_k: [1, 2, 3])
    monkeypatch.setattr(openai, "_encode_prompt", lambda *_a, **_k: [1, 2, 3])
    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)
    return TestClient(create_app(_fake_state()))


def _chat(client, **fields):
    return client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, **fields},
    )


@pytest.mark.parametrize("value", [1.05, 1.3, 0.9, "high"])
def test_chat_rejects_repetition_penalty_that_would_be_ignored(monkeypatch, value):
    captured: dict[str, object] = {}
    response = _chat(_client(monkeypatch, captured), repetition_penalty=value)

    assert response.status_code == 400
    assert "repetition_penalty is not supported" in response.json()["error"]["message"]
    assert "called" not in captured


def test_chat_accepts_repetition_penalty_noop(monkeypatch):
    captured: dict[str, object] = {}
    response = _chat(_client(monkeypatch, captured), repetition_penalty=1.0)

    assert response.status_code == 200
    assert captured["called"] is True


def test_completions_rejects_repetition_penalty(monkeypatch):
    captured: dict[str, object] = {}
    response = _client(monkeypatch, captured).post(
        "/v1/completions",
        json={"prompt": "hello", "max_tokens": 4, "repetition_penalty": 1.05},
    )

    assert response.status_code == 400
    assert "repetition_penalty is not supported" in response.json()["error"]["message"]
    assert "called" not in captured


def test_messages_rejects_repetition_penalty(monkeypatch):
    captured: dict[str, object] = {}
    response = _client(monkeypatch, captured).post(
        "/v1/messages",
        json={
            "model": "mtplx-test-model",
            "max_tokens": 4,
            "messages": [{"role": "user", "content": "hi"}],
            "repetition_penalty": 1.05,
        },
    )

    assert response.status_code == 400
    assert "repetition_penalty is not supported" in response.json()["error"]["message"]
    assert "called" not in captured


def test_server_owned_controls_list_repetition_penalty_as_ignored(monkeypatch):
    # Pre-2.5.3 'hints' policy: client sampler fields are observability only,
    # so the field joins the ignored list instead of failing the request.
    monkeypatch.setenv("MTPLX_CLIENT_CONTROLS_DEFAULT", "hints")
    captured: dict[str, object] = {}
    response = _chat(_client(monkeypatch, captured), repetition_penalty=1.3)

    assert response.status_code == 200
    ignored = captured["request_observability"]["client_sampler_fields_ignored"]
    assert "repetition_penalty" in ignored


def test_anthropic_bridge_carries_repetition_penalty():
    request = openai.AnthropicMessagesRequest(
        model="m",
        max_tokens=16,
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        repetition_penalty=1.05,
    )
    chat = openai._anthropic_to_chat_request(request)
    assert openai._request_extra(chat, "repetition_penalty") == 1.05

    without = openai.AnthropicMessagesRequest(
        model="m",
        max_tokens=16,
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    )
    chat = openai._anthropic_to_chat_request(without)
    assert openai._request_extra(chat, "repetition_penalty") is None
