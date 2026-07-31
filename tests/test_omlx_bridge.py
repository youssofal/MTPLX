from __future__ import annotations

import json

from mtplx.server.omlx_bridge import (
    ToolCallStreamFilter,
    extract_thinking,
    normalize_messages_for_template,
    parse_tool_calls,
)


def test_omlx_adapter_preserves_reasoning_tool_calls_and_tool_results():
    messages = [
        {"role": "system", "content": "You are OpenCode."},
        {"role": "developer", "content": "Keep changes narrow."},
        {"role": "user", "content": "status?"},
        {
            "role": "assistant",
            "content": "Let me inspect.",
            "reasoning_content": "Need to read files first.",
            "tool_calls": [
                {
                    "id": "call_read",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{\"path\":\"x\"}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_read", "content": "file text"},
    ]

    normalized = normalize_messages_for_template(messages)

    assert normalized[0]["role"] == "system"
    assert "You are OpenCode." in normalized[0]["content"]
    assert "Keep changes narrow." in normalized[0]["content"]
    assert normalized[2]["reasoning_content"] == "Need to read files first."
    assert normalized[2]["tool_calls"][0]["id"] == "call_read"
    assert normalized[3]["role"] == "tool"
    assert normalized[3]["tool_call_id"] == "call_read"


def test_omlx_thinking_unclosed_recovers_visible_content():
    reasoning, visible = extract_thinking("<think>Need to answer naturally.")

    assert reasoning == "Need to answer naturally."
    assert visible == "Need to answer naturally."


def test_omlx_tool_parser_tries_qwen_xml_and_normalizes_arguments():
    extraction = parse_tool_calls(
        "<tool_call>{\"name\":\"add\",\"arguments\":{\"a\":1,\"b\":2}}</tool_call>",
        tokenizer=None,
        tools=[{"type": "function", "function": {"name": "add"}}],
    )

    assert extraction.status == "parsed"
    assert extraction.parser_source == "qwen_xml"
    arguments = extraction.tool_calls[0]["function"]["arguments"]
    assert json.loads(arguments) == {"a": 1, "b": 2}


def test_omlx_tool_parser_accepts_opencode_drifted_json_shapes():
    tools = [{"type": "function", "function": {"name": "read"}}]
    samples = [
        '<tool_call>{"tool":"read","args":{"filePath":"src/game/Game.ts"}}</tool_call>',
        '<tool_call>{"function":{"name":"read","arguments":{"filePath":"src/game/Game.ts"}}}</tool_call>',
        '<tool_call>[{"name":"read","parameters":{"filePath":"src/game/Game.ts"}}]</tool_call>',
        '<tool_call>read({"filePath":"src/game/Game.ts"})</tool_call>',
    ]

    for sample in samples:
        extraction = parse_tool_calls(sample, tokenizer=None, tools=tools)
        assert extraction.status == "parsed"
        assert extraction.parser_source == "qwen_xml"
        assert extraction.tool_calls[0]["function"]["name"] == "read"
        arguments = extraction.tool_calls[0]["function"]["arguments"]
        assert json.loads(arguments) == {"filePath": "src/game/Game.ts"}


def test_omlx_tool_parser_accepts_name_attribute_function_shape():
    extraction = parse_tool_calls(
        '<tool_call><function name="read"><parameter name="filePath">src/game/Game.ts</parameter></function></tool_call>',
        tokenizer=None,
        tools=[{"type": "function", "function": {"name": "read"}}],
    )

    assert extraction.status == "parsed"
    assert extraction.tool_calls[0]["function"]["name"] == "read"
    assert json.loads(extraction.tool_calls[0]["function"]["arguments"]) == {
        "filePath": "src/game/Game.ts"
    }


def test_omlx_tool_parser_orders_opencode_read_target_before_limit():
    extraction = parse_tool_calls(
        "<tool_call>\n"
        "<function=read>\n"
        "<parameter=limit>\n100\n</parameter>\n"
        "<parameter=filePath>\nsrc/game/Game.ts\n</parameter>\n"
        "</function>\n"
        "</tool_call>",
        tokenizer=None,
        tools=[{"type": "function", "function": {"name": "read"}}],
    )

    assert extraction.status == "parsed"
    arguments = extraction.tool_calls[0]["function"]["arguments"]
    assert arguments.startswith('{"filePath":"src/game/Game.ts","limit":100')
    assert json.loads(arguments) == {"filePath": "src/game/Game.ts", "limit": 100}


def test_omlx_tool_parser_malformed_markup_remains_content():
    text = "<tool_call>not json</tool_call>"
    extraction = parse_tool_calls(
        text,
        tokenizer=None,
        tools=[{"type": "function", "function": {"name": "add"}}],
    )

    assert extraction.status == "malformed_as_content"
    assert extraction.tool_calls is None
    assert extraction.cleaned_text == text


def test_omlx_stream_filter_suppresses_tool_markup_without_cancelling():
    stream_filter = ToolCallStreamFilter()
    visible = []
    for char in "Before <tool_call>{\"name\":\"add\"}</tool_call> after":
        chunk = stream_filter.feed(char)
        if chunk:
            visible.append(chunk)
    tail = stream_filter.finish()
    if tail:
        visible.append(tail)

    assert "".join(visible) == "Before  after"
    assert stream_filter.suppressed_markup is True


# ---------- #170: the extraction lane must not fabricate or deliver
# impossible calls (contract parity with the strict streaming/final parsers) --

EDIT_FILE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "search": {"type": "string"},
                                "replace": {"type": "string"},
                            },
                            "required": ["search", "replace"],
                        },
                    },
                },
                "required": ["path", "edits"],
            },
        },
    }
]


def test_omlx_tool_parser_json_body_in_function_envelope():
    nested = {
        "path": "config.py",
        "edits": [{"search": "a = 1", "replace": 'a = {"b": 2}'}],
    }
    extraction = parse_tool_calls(
        "<tool_call>\n<function=edit_file>\n"
        + json.dumps(nested)
        + "\n</function>\n</tool_call>",
        tokenizer=None,
        tools=EDIT_FILE_TOOLS,
    )
    assert extraction.status == "parsed"
    assert json.loads(extraction.tool_calls[0]["function"]["arguments"]) == nested


def test_omlx_tool_parser_never_fabricates_empty_arguments():
    """#170 delivery lane: a recognized envelope with an unreadable body used
    to come back as a schema-less call with arguments {}."""
    extraction = parse_tool_calls(
        "<tool_call>\n<function=edit_file>\nnot a payload at all\n</function>\n</tool_call>",
        tokenizer=None,
        tools=EDIT_FILE_TOOLS,
    )
    assert extraction.status == "malformed_as_content"
    assert extraction.tool_calls is None
    assert "unwrapped parameter text" in (extraction.malformed_reason or "")


def test_omlx_tool_parser_delivers_partial_arguments_faithfully():
    """OpenAI-protocol contract: arguments carry the model's actual output and
    schema validation is the client's job (test_chat_stream_missing_required_
    tool_argument_still_emits_model_tool_call pins the same rule at the stream
    level). What the parser must never do is FABRICATE arguments — a partial
    call here must be the model's own partial payload, not an invented {}."""
    extraction = parse_tool_calls(
        "<tool_call>\n<function=edit_file>\n"
        "<parameter=path>\nconfig.py\n</parameter>\n"
        "</function>\n</tool_call>",
        tokenizer=None,
        tools=EDIT_FILE_TOOLS,
    )
    assert extraction.status == "parsed"
    assert json.loads(extraction.tool_calls[0]["function"]["arguments"]) == {
        "path": "config.py"
    }


def test_omlx_tool_parser_empty_body_no_arg_tool_still_parses():
    extraction = parse_tool_calls(
        "<tool_call>\n<function=list_files>\n</function>\n</tool_call>",
        tokenizer=None,
        tools=[{"type": "function", "function": {"name": "list_files"}}],
    )
    assert extraction.status == "parsed"
    assert json.loads(extraction.tool_calls[0]["function"]["arguments"]) == {}


class _NativeToolTokenizer:
    """Tokenizer double for the native tool_parser branch: mirrors the live
    mlx-lm TokenizerWrapper shape that returned empty arguments for JSON-object
    function bodies (the probe-2 live receipt behind the #170 fix)."""

    has_tool_calling = True
    tool_call_start = "<tool_call>"
    tool_call_end = "</tool_call>"

    @staticmethod
    def tool_parser(_text, _tools):
        return {"name": "grep", "arguments": {}}


GREP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "grep",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    }
]


def test_omlx_native_branch_recovers_json_body_instead_of_empty_args():
    extraction = parse_tool_calls(
        '<tool_call>\n<function=grep>\n{"path": "config.py"}\n</function>\n</tool_call>',
        tokenizer=_NativeToolTokenizer(),
        tools=GREP_TOOLS,
    )
    assert extraction.status == "parsed"
    assert json.loads(extraction.tool_calls[0]["function"]["arguments"]) == {
        "path": "config.py"
    }


def test_omlx_native_branch_never_fabricates_empty_args_from_garbage_body():
    extraction = parse_tool_calls(
        "<tool_call>\n<function=grep>\nnot a payload\n</function>\n</tool_call>",
        tokenizer=_NativeToolTokenizer(),
        tools=GREP_TOOLS,
    )
    assert extraction.status == "malformed_as_content"
    assert extraction.tool_calls is None


def test_omlx_native_branch_keeps_blank_body_no_arg_call():
    extraction = parse_tool_calls(
        "<tool_call>\n<function=grep>\n</function>\n</tool_call>",
        tokenizer=_NativeToolTokenizer(),
        tools=GREP_TOOLS,
    )
    assert extraction.status == "parsed"
    assert json.loads(extraction.tool_calls[0]["function"]["arguments"]) == {}


def test_omlx_tool_parser_accepts_poolside_arg_pairs():
    extraction = parse_tool_calls(
        "I'll list the files.<tool_call>list_files"
        "<arg_key>path</arg_key><arg_value>src</arg_value></tool_call>",
        tokenizer=None,
        tools=[{"type": "function", "function": {"name": "list_files"}}],
    )

    assert extraction.status == "parsed"
    assert json.loads(extraction.tool_calls[0]["function"]["arguments"]) == {
        "path": "src"
    }
    assert extraction.cleaned_text == "I'll list the files."


def test_omlx_tool_parser_poolside_residue_stays_content():
    text = (
        "<tool_call>foo<arg_key>k</arg_key><arg_value>v</arg_value>"
        "garbage</tool_call>"
    )
    extraction = parse_tool_calls(
        text,
        tokenizer=None,
        tools=[{"type": "function", "function": {"name": "foo"}}],
    )

    assert extraction.status == "malformed_as_content"
    assert extraction.tool_calls is None


def test_omlx_tool_parser_passes_unknown_tool_name_through():
    extraction = parse_tool_calls(
        "<tool_call>task_progress<arg_key>steps</arg_key>"
        "<arg_value>plan</arg_value></tool_call>",
        tokenizer=None,
        tools=[{"type": "function", "function": {"name": "list_files"}}],
    )

    assert extraction.status == "parsed"
    assert extraction.tool_calls[0]["function"]["name"] == "task_progress"
