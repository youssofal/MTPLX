"""Unclosed reasoning is recovered as content on agent turns too (#583).

A thinking model that answers after a tool result inside the template-opened
think block, without ``</think>``, left the visible content empty whenever the
request declared tools, which is every agent turn. The recovery now also runs
with tools declared, on a natural stop and when the turn holds no tool markup.

Engine always monkeypatched, CPU only.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from test_server_openai import (
    _fake_state,
    _fake_streaming_generation,
    _stream_payloads,
)

from mtplx.server import openai
from mtplx.server.openai import _unclosed_reasoning_recovery_allowed, create_app

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_service",
            "description": "Look up a service port.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    }
]


# --- the gate ---------------------------------------------------------------


def test_gate_tools_active_clean_stop_recovers():
    assert _unclosed_reasoning_recovery_allowed(
        finish_reason="stop", tools_active=True, generated_text="PORT=7505"
    )


def test_gate_length_never_recovers():
    for tools_active in (True, False):
        assert not _unclosed_reasoning_recovery_allowed(
            finish_reason="length",
            tools_active=tools_active,
            generated_text="PORT=7505",
        )


def test_gate_tool_call_in_turn_does_not_recover():
    text = (
        "I will look it up.<tool_call><function=lookup_service>"
        "<parameter=name>billing</parameter></function></tool_call>"
    )
    assert not _unclosed_reasoning_recovery_allowed(
        finish_reason="tool_calls", tools_active=True, generated_text=text
    )
    assert not _unclosed_reasoning_recovery_allowed(
        finish_reason="stop", tools_active=True, generated_text=text
    )


def test_gate_tool_markup_in_text_does_not_recover():
    assert not _unclosed_reasoning_recovery_allowed(
        finish_reason="stop",
        tools_active=True,
        generated_text="PORT=1 <FUNCTION=lookup_service>",
    )


def test_gate_without_tools_ignores_markup():
    # No tools declared: orphan markup is stripped downstream, as before.
    assert _unclosed_reasoning_recovery_allowed(
        finish_reason="stop", tools_active=False, generated_text="<tool_call>"
    )


# --- through the endpoints --------------------------------------------------


def _state():
    state = _fake_state()
    state.args.stats_footer = False
    return state


def _patch_nonstream(monkeypatch, text: str, finish_reason: str):
    monkeypatch.setattr(
        openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3]
    )
    monkeypatch.setattr(
        openai,
        "_run_generation",
        lambda *_args, **_kwargs: {
            "text": text,
            "tokens": [4],
            "stats": {"generation_mode": "ar", "mtp_depth": 0, "completion_tokens": 8},
            "prompt_tokens": 3,
            "completion_tokens": 8,
            "finish_reason": finish_reason,
        },
    )


def _post(client, *, stream: bool = False):
    return client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [
                {"role": "user", "content": "Which port? Answer only with PORT=<n>."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup_service",
                                "arguments": '{"name": "billing"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": '{"port": 7505}'},
            ],
            "tools": TOOLS,
            "stream": stream,
            "max_tokens": 128,
        },
    )


def test_nonstream_with_tools_recovers_unclosed_answer(monkeypatch):
    _patch_nonstream(monkeypatch, "<think>PORT=7505", "stop")
    response = _post(TestClient(create_app(_state())))

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"]["content"] == "PORT=7505"
    assert not choice["message"].get("tool_calls")


def test_nonstream_with_tools_length_stays_empty(monkeypatch):
    _patch_nonstream(monkeypatch, "<think>PORT=7505", "length")
    response = _post(TestClient(create_app(_state())))

    choice = response.json()["choices"][0]
    assert not (choice["message"].get("content") or "")
    assert choice["finish_reason"] == "length"


def test_nonstream_with_tools_tool_call_is_not_recovered(monkeypatch):
    _patch_nonstream(
        monkeypatch,
        "<think>checking<tool_call><function=lookup_service>"
        "<parameter=name>billing</parameter></function></tool_call>",
        "stop",
    )
    response = _post(TestClient(create_app(_state())))

    choice = response.json()["choices"][0]
    assert choice["message"]["tool_calls"]
    assert "PORT" not in (choice["message"].get("content") or "")
    assert "tool_call" not in (choice["message"].get("content") or "")


def _stream_content(response) -> str:
    return "".join(
        frame["choices"][0]["delta"].get("content") or ""
        for frame in _stream_payloads(response.text)
        if frame.get("choices") and frame["choices"][0].get("delta")
    )


def test_stream_with_tools_recovers_unclosed_answer(monkeypatch):
    monkeypatch.setattr(
        openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3]
    )
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation("PORT=7505", finish_reason="stop"),
    )
    response = _post(TestClient(create_app(_state())), stream=True)

    assert response.status_code == 200
    assert _stream_content(response) == "PORT=7505"


def test_stream_with_tools_length_stays_empty(monkeypatch):
    monkeypatch.setattr(
        openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3]
    )
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation("PORT=7505", finish_reason="length"),
    )
    response = _post(TestClient(create_app(_state())), stream=True)

    assert _stream_content(response) == ""
