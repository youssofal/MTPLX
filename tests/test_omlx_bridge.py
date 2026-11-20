from __future__ import annotations

import json

import pytest

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


def test_omlx_tool_parser_accepts_hy3_suffixed_native_protocol():
    extraction = parse_tool_calls(
        "<tool_calls:opensource>"
        "<tool_call:opensource>read<tool_sep:opensource>"
        "<arg_key:opensource>filePath</arg_key:opensource>"
        "<arg_value:opensource>/tmp/brief.md</arg_value:opensource>"
        "</tool_call:opensource>"
        "</tool_calls:opensource>",
        tokenizer=None,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "parameters": {
                        "type": "object",
                        "properties": {"filePath": {"type": "string"}},
                        "required": ["filePath"],
                    },
                },
            }
        ],
    )

    assert extraction.status == "parsed"
    assert extraction.parser_source == "suffixed_native"
    assert extraction.cleaned_text == ""
    assert extraction.tool_calls[0]["function"]["name"] == "read"
    assert json.loads(extraction.tool_calls[0]["function"]["arguments"]) == {
        "filePath": "/tmp/brief.md"
    }


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


@pytest.mark.parametrize(
    "exact, prefix",
    [
        # #691: the model's own order, so the call the client sends back
        # renders as the tokens the model generated.
        ("1", '{"limit":100,"filePath":"src/game/Game.ts"'),
        # MTPLX_TOOL_ARGUMENTS_EXACT=0: the earlier path-first order.
        ("0", '{"filePath":"src/game/Game.ts","limit":100'),
    ],
)
def test_omlx_tool_parser_read_argument_order(monkeypatch, exact, prefix):
    monkeypatch.setenv("MTPLX_TOOL_ARGUMENTS_EXACT", exact)
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
    assert arguments.startswith(prefix)
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


def test_omlx_tool_parser_accepts_lfm_pythonic_marker_protocol():
    extraction = parse_tool_calls(
        "<think>plan the lookup</think>I'll check the weather."
        "<|tool_call_start|>[get_weather(city='Paris', days=3, "
        'opts={"detail": true, "fallback": null}, tags=[\'a\', \'b\'], '
        "verbose=True, region=None)]<|tool_call_end|>",
        tokenizer=None,
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
    )

    assert extraction.status == "parsed"
    assert extraction.parser_source == "pythonic_marker"
    assert extraction.raw_tool_markup_suppressed is True
    assert "tool_call" not in extraction.cleaned_text
    assert "I'll check the weather." in extraction.cleaned_text
    call = extraction.tool_calls[0]["function"]
    assert call["name"] == "get_weather"
    assert json.loads(call["arguments"]) == {
        "city": "Paris",
        "days": 3,
        "opts": {"detail": True, "fallback": None},
        "tags": ["a", "b"],
        "verbose": True,
        "region": None,
    }


def test_omlx_tool_parser_lfm_pythonic_multiple_calls_in_one_envelope():
    extraction = parse_tool_calls(
        "<|tool_call_start|>[first(x=1), second(y='z')]<|tool_call_end|>",
        tokenizer=None,
        tools=[
            {"type": "function", "function": {"name": "first"}},
            {"type": "function", "function": {"name": "second"}},
        ],
    )

    assert extraction.status == "parsed"
    names = [call["function"]["name"] for call in extraction.tool_calls]
    assert names == ["first", "second"]
    assert json.loads(extraction.tool_calls[1]["function"]["arguments"]) == {"y": "z"}


def test_omlx_tool_parser_lfm_pythonic_sole_positional_maps_to_parameter():
    extraction = parse_tool_calls(
        "<|tool_call_start|>[read_file('notes.md')]<|tool_call_end|>",
        tokenizer=None,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ],
    )

    assert extraction.status == "parsed"
    assert json.loads(extraction.tool_calls[0]["function"]["arguments"]) == {
        "path": "notes.md"
    }


def test_omlx_tool_parser_lfm_pythonic_ambiguous_positional_is_malformed():
    text = "<|tool_call_start|>[edit('a', 'b')]<|tool_call_end|>"
    extraction = parse_tool_calls(
        text,
        tokenizer=None,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "edit",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "old": {"type": "string"},
                            "new": {"type": "string"},
                        },
                    },
                },
            }
        ],
    )

    assert extraction.status == "malformed_as_content"
    assert extraction.tool_calls is None
    assert "positional" in (extraction.malformed_reason or "")
    assert extraction.cleaned_text == text


def test_omlx_tool_parser_lfm_pythonic_unknown_tool_refuses():
    extraction = parse_tool_calls(
        "<|tool_call_start|>[made_up(x=1)]<|tool_call_end|>",
        tokenizer=None,
        tools=[{"type": "function", "function": {"name": "real_tool"}}],
    )

    assert extraction.status == "malformed_as_content"
    assert extraction.tool_calls is None
    assert "no declared tool" in (extraction.malformed_reason or "")


def test_omlx_tool_parser_lfm_pythonic_unclosed_envelope_is_malformed():
    extraction = parse_tool_calls(
        "<|tool_call_start|>[get_weather(city='Par",
        tokenizer=None,
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
    )

    assert extraction.status == "malformed_as_content"
    assert extraction.tool_calls is None


def test_omlx_stream_filter_suppresses_lfm_pythonic_envelope_across_chunks():
    stream = ToolCallStreamFilter(tokenizer=None)
    visible = ""
    for chunk in (
        "Checking now. <|tool_call_st",
        "art|>[get_weather(city='Paris')]<|tool_call_e",
        "nd|> Done.",
    ):
        visible += stream.feed(chunk)
    visible += stream.finish()

    assert "tool_call" not in visible
    assert "get_weather" not in visible
    assert "Checking now." in visible
    assert "Done." in visible
    assert stream.suppressed_markup is True


def test_omlx_tool_parser_lfm_pythonic_call_inside_thinking_is_recovered():
    from mtplx.server.omlx_bridge import extract_tool_calls_with_thinking

    extraction = extract_tool_calls_with_thinking(
        "I should look this up "
        "<|tool_call_start|>[get_weather(city='Oslo')]<|tool_call_end|>",
        "",
        tokenizer=None,
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
    )

    assert extraction.tool_calls is not None
    assert extraction.tool_calls[0]["function"]["name"] == "get_weather"
    assert "tool_call" not in extraction.cleaned_thinking


def _stream_through_filter(text: str, chunk: int) -> tuple[str, str, ToolCallStreamFilter]:
    stream = ToolCallStreamFilter()
    streamed = ""
    for index in range(0, len(text), chunk):
        streamed += stream.feed(text[index : index + chunk])
    return streamed, stream.finish(), stream


@pytest.mark.parametrize("chunk", [1, 7, 13, 500])
def test_omlx_stream_filter_keeps_prose_that_quotes_the_bracket_marker(chunk):
    # A literal "[Calling tool:" with no "]" used to put the filter into an
    # unbounded hold (the bracket branch skipped the 128-char cap) and
    # finish() then dropped the whole tail: the answer stopped mid-sentence
    # with a normal finish and no error. Prose that merely quotes the marker
    # is the answer's own words and streams through in full.
    text = (
        "MTPLX prints [Calling tool: read_file when it starts a tool. The marker "
        "is followed by the tool name and a closing bracket once the call is "
        "complete, so a log line like this one is not a call."
    )

    streamed, tail, stream = _stream_through_filter(text, chunk)

    assert streamed + tail == text
    assert stream.suppressed_markup is False


def test_omlx_stream_filter_keeps_a_quoted_marker_split_at_the_prefix():
    stream = ToolCallStreamFilter()
    visible = stream.feed("Type [Calling to")
    visible += stream.feed("ol: and then the tool name.")
    visible += stream.finish()

    assert visible == "Type [Calling tool: and then the tool name."
    assert stream.suppressed_markup is False


def test_omlx_stream_filter_sanitize_keeps_whole_text_that_quotes_the_marker():
    from mtplx.server.omlx_bridge.tool_calling import sanitize_tool_call_markup

    text = "The [Tool call: marker appears when a call starts; nothing else changes."

    assert sanitize_tool_call_markup(text) == text


@pytest.mark.parametrize("chunk", [1, 5, 9, 500])
def test_omlx_stream_filter_still_suppresses_a_complete_bracket_call_split_across_chunks(chunk):
    text = 'Let me check. [Calling tool: lookup({"q": "scene [1]"})] Done.'

    streamed, tail, stream = _stream_through_filter(text, chunk)

    assert streamed + tail == "Let me check.  Done."
    assert stream.suppressed_markup is True


def test_omlx_stream_filter_holds_long_multiline_json_arguments_until_the_call_closes():
    # The hold is bounded by the call's grammar, not by a character count or
    # the current line: a real call with a whole file body in its arguments
    # is held until "]" and suppressed as one unit.
    payload = {"input": "*** Begin Patch\n" + "\n".join(f"+line {i} => ({i});" for i in range(60)) + "\n*** End Patch"}
    text = "Applying: [Calling tool: apply_patch(" + json.dumps(payload) + ")] Applied."
    assert len(text) > 600

    streamed, tail, stream = _stream_through_filter(text, 11)

    assert streamed + tail == "Applying:  Applied."
    assert "Begin Patch" not in streamed
    assert stream.suppressed_markup is True


def test_omlx_stream_filter_still_suppresses_an_unclosed_call_shaped_block():
    # Same contract as the translator: a block that is structurally a call
    # but never closes is markup, not prose, and stays out of the transcript.
    text = 'And now: [Calling tool: apply_patch({"input": "never closed'

    streamed, tail, stream = _stream_through_filter(text, 9)

    assert streamed + tail == "And now: "
    assert stream.suppressed_markup is True


def test_omlx_stream_filter_releases_the_hold_once_a_block_can_no_longer_be_a_call():
    # Name, then "(" with no "{": not the dialect, so the words flow.
    text = "Call read_file(path) via [Calling tool: read_file(path) and see."
    streamed, tail, _stream = _stream_through_filter(text, 6)
    assert streamed + tail == text

    # Balanced JSON followed by anything but ")]" is not a call either.
    text = 'Use [Calling tool: lookup({"q": 1}) then go on.'
    streamed, tail, _stream = _stream_through_filter(text, 6)
    assert streamed + tail == text
