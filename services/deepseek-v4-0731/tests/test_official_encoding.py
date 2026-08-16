"""Byte-exact gates for the official DeepSeek-V4-Flash-0731 encoder."""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENCODING = ROOT / "encoding"
VECTORS = ENCODING / "tests"


def _official():
    path = ENCODING / "encoding_dsv4.py"
    module = ModuleType("encoding_dsv4")
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("case", [1, 2, 3, 4])
def test_official_vectors_are_byte_exact(case: int) -> None:
    encoding = _official()
    payload = json.loads((VECTORS / f"test_input_{case}.json").read_text(encoding="utf-8"))
    if case == 1:
        messages = payload["messages"]
        messages[0]["tools"] = payload["tools"]
    else:
        messages = payload
    mode = "chat" if case == 4 else "thinking"
    expected = (VECTORS / f"test_output_{case}.txt").read_text(encoding="utf-8")
    assert encoding.encode_messages(messages, thinking_mode=mode) == expected


def test_dsml_tool_call_result_merge_and_completion_parse() -> None:
    encoding = _official()
    payload = json.loads((VECTORS / "test_input_1.json").read_text(encoding="utf-8"))
    messages = payload["messages"]
    messages[0]["tools"] = payload["tools"]
    prompt = encoding.encode_messages(messages, thinking_mode="thinking")
    assert "<｜DSML｜tool_calls>" in prompt
    assert "<tool_result>" in prompt
    assert not any(message.get("role") == "tool" for message in encoding.merge_tool_messages(messages))

    marker = "<｜Assistant｜><think>"
    first_start = prompt.find(marker) + len(marker)
    first_end = prompt.find("<｜User｜>", first_start)
    parsed = encoding.parse_message_from_completion_text(prompt[first_start:first_end], thinking_mode="thinking")
    assert parsed["reasoning_content"].startswith("The user wants")
    assert parsed["content"] == ""
    assert parsed["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(parsed["tool_calls"][0]["function"]["arguments"]) == {
        "location": "Beijing",
        "unit": "celsius",
    }


@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_reasoning_prefixes_and_thinking_modes(effort: str) -> None:
    encoding = _official()
    messages = [{"role": "user", "content": "hi"}]
    thinking = encoding.encode_messages(messages, thinking_mode="thinking", reasoning_effort=effort)
    chat = encoding.encode_messages(messages, thinking_mode="chat", reasoning_effort=effort)
    assert thinking.startswith(encoding.bos_token + encoding.REASONING_EFFORT_PROMPTS[effort])
    assert thinking.endswith(encoding.ASSISTANT_SP_TOKEN + encoding.thinking_start_token)
    assert chat.endswith(encoding.ASSISTANT_SP_TOKEN + encoding.thinking_end_token)


def test_invalid_reasoning_effort_fails_closed() -> None:
    encoding = _official()
    with pytest.raises(AssertionError, match="Invalid reasoning effort"):
        encoding.encode_messages(
            [{"role": "user", "content": "hi"}],
            thinking_mode="thinking",
            reasoning_effort="medium",
        )
