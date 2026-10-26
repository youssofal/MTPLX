"""Claude Code auto-mode classifier requests on /v1/messages (issue #1)."""

from __future__ import annotations

from mtplx.server import openai

CLASSIFIER_SYSTEM = (
    "You are a security monitor for autonomous AI coding agents. "
    "Decide whether the next action is safe."
)
BILLING_BLOCK = {
    "type": "text",
    "text": "x-anthropic-billing-header: cc_version=2.1.0; cch=abc123;",
}
USER_MESSAGE = [{"role": "user", "content": "Is `rm -rf build` safe?"}]


def _request(system, *, tools=None, stream=False):
    payload = {
        "model": "m",
        "max_tokens": 64,
        "messages": USER_MESSAGE,
        "stream": stream,
    }
    if system is not None:
        payload["system"] = system
    if tools is not None:
        payload["tools"] = tools
    return openai.AnthropicMessagesRequest(**payload)


def _bash_tool():
    return {
        "name": "Bash",
        "description": "Run a shell command.",
        "input_schema": {"type": "object", "properties": {}},
    }


def _is_classifier(request) -> bool:
    return openai._anthropic_to_chat_request(request).auto_mode_classifier


def test_detects_string_system_with_marker():
    assert _is_classifier(_request(CLASSIFIER_SYSTEM)) is True


def test_detects_marker_in_a_non_first_system_block():
    request = _request(
        [
            {"type": "text", "text": "Unrelated preamble."},
            {"type": "text", "text": CLASSIFIER_SYSTEM},
        ]
    )
    assert _is_classifier(request) is True


def test_billing_block_alone_is_not_a_classifier_request():
    assert _is_classifier(_request([BILLING_BLOCK])) is False


def test_marker_absent_is_not_a_classifier_request():
    assert _is_classifier(_request("You are a helpful coding assistant.")) is False


def test_streaming_request_is_not_a_classifier_request():
    assert _is_classifier(_request(CLASSIFIER_SYSTEM, stream=True)) is False


def test_request_with_tools_is_not_a_classifier_request():
    assert _is_classifier(_request(CLASSIFIER_SYSTEM, tools=[_bash_tool()])) is False


def test_empty_tools_list_still_counts_as_classifier_request():
    assert _is_classifier(_request(CLASSIFIER_SYSTEM, tools=[])) is True


def test_no_system_is_not_a_classifier_request():
    assert _is_classifier(_request(None)) is False
