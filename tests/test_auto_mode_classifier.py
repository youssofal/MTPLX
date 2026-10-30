"""Claude Code auto-mode classifier requests on /v1/messages (issue #669).

The classifier runs with thinking off, which changes generation, so only its
exact shape may match. The shapes below are claude-cli 2.1.201's own requests,
captured against a local recording server: the classifier's two stages (a
2,112-token verdict that stops at </block>, then a 10,240-token review), and
the main agent turn sent in the same session. The classifier's 78,698-character
system prompt is cut to its first sentence and a stand-in; every field that
decides the match is as captured.
"""

from __future__ import annotations

import pytest

from mtplx.server import openai

BILLING = {
    "type": "text",
    "text": "x-anthropic-billing-header: cc_version=2.1.201.af3; cc_entrypoint=sdk-cli;",
}
MONITOR = (
    "You are a security monitor for autonomous AI coding agents.\n\n## Context\n\n"
    "The agent you are monitoring is an **autonomous coding agent** with shell "
    "access, file system access, and API credentials."
)
TRANSCRIPT = [
    {"type": "text", "text": "<transcript>\n"},
    {"type": "text", "text": "User: Use the Bash tool to run `git push --force origin main`.\n"},
    {"type": "text", "text": "Bash git push --force origin main\n", "cache_control": {"type": "ephemeral"}},
    {"type": "text", "text": "</transcript>\n"},
]
STAGE_ONE_TAIL = {"type": "text", "text": "\nErr on the side of blocking. <block> immediately."}
STAGE_TWO_TAIL = {"type": "text", "text": "\nUse <thinking> before responding with <block>."}
USER_ID = '{"device_id":"d","account_uuid":"","session_id":"a4bf8043-d676-458d-8c2c-6199ccdebb4c"}'


def _classifier(stage: int = 1, **overrides):
    payload = {
        "model": "mtplx-capture-model",
        "max_tokens": 2112 if stage == 1 else 10240,
        "system": [BILLING, {"type": "text", "text": MONITOR, "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {"role": "user", "content": [*TRANSCRIPT, STAGE_ONE_TAIL if stage == 1 else STAGE_TWO_TAIL]}
        ],
        "metadata": {"user_id": USER_ID},
    }
    if stage == 1:
        payload["stop_sequences"] = ["</block>"]
    payload.update(overrides)
    return openai.AnthropicMessagesRequest(**payload)


def _is_classifier(request) -> bool:
    return openai._anthropic_to_chat_request(request).auto_mode_classifier


@pytest.mark.parametrize("stage", [1, 2])
def test_both_classifier_stages_are_detected(stage):
    assert _is_classifier(_classifier(stage)) is True


def test_empty_tools_list_still_counts_as_classifier_request():
    assert _is_classifier(_classifier(tools=[])) is True


def test_the_main_claude_code_turn_of_the_same_session_is_untouched():
    main = openai.AnthropicMessagesRequest(
        model="mtplx-capture-model",
        max_tokens=32000,
        stream=True,
        system=[
            {**BILLING, "text": "x-anthropic-billing-header: cc_version=2.1.201.8ca; cc_entrypoint=sdk-cli;"},
            {"type": "text", "text": "You are a Claude agent, built on Anthropic's Claude Agent SDK."},
            {"type": "text", "text": "You are an interactive agent that helps users with software engineering tasks."},
        ],
        messages=[{"role": "user", "content": "Use the Bash tool to run `echo hi`."}],
        tools=[{"name": "Bash", "description": "Run a shell command.", "input_schema": {"type": "object"}}],
        thinking={"type": "adaptive", "display": "omitted"},
        output_config={"effort": "high"},
    )
    assert _is_classifier(main) is False


@pytest.mark.parametrize(
    "near_miss",
    [
        # The classifier's words from a client that is not Claude Code.
        {"system": [{"type": "text", "text": MONITOR}]},
        {"system": MONITOR},
        # Claude Code's billing block, but the sentence is not where the
        # classifier's system prompt starts.
        {"system": [BILLING, {"type": "text", "text": "Unrelated preamble. " + MONITOR}]},
        {"system": [BILLING, {"type": "text", "text": "Review this. " + MONITOR.lower()}]},
        {"system": [BILLING, {"type": "text", "text": MONITOR.lower()}]},
        # The exact shape, plus something the classifier never sends.
        {"tools": [{"name": "Bash", "description": "Run.", "input_schema": {"type": "object"}}]},
        {"stream": True},
        {"thinking": {"type": "enabled", "budget_tokens": 1024}},
        {"thinking": {"type": "adaptive"}},
    ],
    ids=[
        "no-billing-block",
        "no-billing-string-system",
        "sentence-not-first",
        "lowercase-not-first",
        "lowercase-first",
        "tools",
        "stream",
        "thinking-enabled",
        "thinking-adaptive",
    ],
)
def test_requests_near_the_classifier_shape_keep_thinking(near_miss):
    assert _is_classifier(_classifier(**near_miss)) is False


def test_no_system_is_not_a_classifier_request():
    assert _is_classifier(_classifier(system=None)) is False
