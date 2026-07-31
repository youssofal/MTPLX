"""Bracket-dialect tool calls (`[Calling tool: name({...})]`) parse via the
balanced scanner.

Laguna drifts into this textual dialect on large writes (observed live on a
game.js apply_patch, 2026-07-25). The old non-greedy regex ended the block at
the first `})]`, which any code-file argument contains inside a string, so
every large bracket call failed JSON decode and fell back to prose — burning
a Cline mistake strike.
"""

import json

from mtplx.server.openai import (
    _parse_generated_tool_calls,
    _scan_bracket_tool_call,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "parameters": {
                "type": "object",
                "properties": {"input": {"type": "string"}},
            },
        },
    }
]


def test_bracket_call_with_close_sequence_inside_string_parses():
    # The argument string contains the literal "})]"; the old regex ended the
    # envelope there and the decode failed.
    payload = {"input": "const f = (x) => ({y: x});\ncall(f(1))]... })] tail"}
    text = "Prose before. [Calling tool: apply_patch(" + json.dumps(payload) + ")]"
    calls = _parse_generated_tool_calls(text, tools=TOOLS, tokenizer=None)
    assert calls is not None and len(calls) == 1
    assert calls[0]["function"]["name"] == "apply_patch"
    assert json.loads(calls[0]["function"]["arguments"]) == payload


def test_bracket_call_with_checklist_brackets_parses():
    payload = {
        "input": "*** Begin Patch\n+code\n*** End Patch",
        "task_progress": "- [x] step one\n- [ ] step two",
    }
    text = "[Calling tool: apply_patch(" + json.dumps(payload) + ")]"
    calls = _parse_generated_tool_calls(text, tools=TOOLS, tokenizer=None)
    assert calls is not None
    assert json.loads(calls[0]["function"]["arguments"]) == payload


def test_unterminated_bracket_call_is_not_an_envelope():
    import pytest
    from fastapi.exceptions import HTTPException

    text = '[Calling tool: apply_patch({"input": "never closed'
    assert _scan_bracket_tool_call(text, 0) is None
    # The parser classifies it as an unclosed block; the catch layer above
    # turns that into visible content plus a logged fallback reason.
    with pytest.raises(HTTPException):
        _parse_generated_tool_calls(text, tools=TOOLS, tokenizer=None)


def test_zero_argument_bracket_call_parses():
    end, name, arguments = _scan_bracket_tool_call("[Calling tool: noop()]", 0)
    assert (name, arguments) == ("noop", {})
    assert end == len("[Calling tool: noop()]")
