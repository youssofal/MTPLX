"""The tool-call example shows the parameter syntax whenever a tool has one (#584).

OpenCode's requests run in compact tool mode, where the chat template is given
no tools, so the contract's single example is the only place the prompt shows
how a call is written. It was built from the first declared tool. In the #584
repro that is take_screenshot, which takes no parameters, so the prompt never
showed a <parameter=...> block, and after a screenshot in the history the
model wrote click_xy's numbers as "<parameter=290>". The tools and messages
are the repro's own (repro.py in #584).
"""

from __future__ import annotations

from mtplx.server import openai
from tests.test_server_openai import CaptureTokenizer

TAKE_SCREENSHOT = {
    "type": "function",
    "function": {
        "name": "take_screenshot",
        "description": "Take a screenshot",
        "parameters": {"type": "object", "properties": {}},
    },
}
CLICK_XY = {
    "type": "function",
    "function": {
        "name": "click_xy",
        "description": "Click at x, y in CSS pixels",
        "parameters": {
            "type": "object",
            "required": ["x", "y"],
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
        },
    },
}
REPRO_TOOLS = [TAKE_SCREENSHOT, CLICK_XY]
REPRO_MESSAGES = [
    openai.ChatMessage(role="system", content="You are a browser agent."),
    openai.ChatMessage(role="user", content="Click the «Delete account» button on the open page."),
    openai.ChatMessage(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "take_screenshot", "arguments": "{}"},
            }
        ],
    ),
    openai.ChatMessage(role="tool", tool_call_id="call_1", content="Screenshot taken"),
    openai.ChatMessage(role="user", content="Attached media from tool result:"),
]


def test_the_example_shows_the_parameter_syntax_for_the_repro_tools():
    example = openai._tool_call_example(REPRO_TOOLS)
    assert example == (
        "<tool_call>\n<function=click_xy>\n"
        "<parameter=x>\n0\n</parameter>\n"
        "<parameter=y>\n0\n</parameter>\n"
        "</function>\n</tool_call>"
    )


def test_opencode_compact_prompt_shows_the_parameter_syntax():
    tokenizer = CaptureTokenizer()
    openai._encode_messages_uncached(
        tokenizer,
        REPRO_MESSAGES,
        enable_thinking=False,
        tools=REPRO_TOOLS,
        tool_prompt_mode="compact",
    )
    rendered, kwargs = tokenizer.calls[-1]
    # Compact mode: the template gets no tools, the contract carries them.
    assert kwargs.get("tools") is None
    system = rendered[0]["content"]
    assert "click_xy(" in system
    assert "<parameter=x>" in system


def test_the_example_is_unchanged_when_the_first_tool_has_parameters():
    bash = {
        "type": "function",
        "function": {
            "name": "bash",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
        },
    }
    assert openai._tool_call_example([bash, CLICK_XY, TAKE_SCREENSHOT]) == (
        "<tool_call>\n<function=bash>\n"
        "<parameter=command>\nARGUMENT_VALUE\n</parameter>\n"
        "</function>\n</tool_call>"
    )


def test_tools_without_parameters_keep_the_first_tool():
    other = {
        "type": "function",
        "function": {"name": "refresh", "parameters": {"type": "object"}},
    }
    assert openai._tool_call_example([TAKE_SCREENSHOT, other]) == (
        "<tool_call>\n<function=take_screenshot>\n</function>\n</tool_call>"
    )
