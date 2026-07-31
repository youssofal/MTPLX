"""Unit tests for _ToolAwareContentStreamTranslator covering both
tool-only responses (the original v0.2.0 fix) and mixed text+tool responses
(this fix - issue #20)."""

import json

from mtplx.server.openai import _ToolAwareContentStreamTranslator, _tool_call_example


TOOL_SPECS = [{"function": {"name": "lookup", "parameters": {"type": "object"}}}]
WRITE_FILE_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "contents": {"type": "string"},
                },
                "required": ["path", "contents"],
            },
        },
    }
]
MIXED_TOOL_SPECS = [
    {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}},
    {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}},
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "contents": {"type": "string"},
                },
                "required": ["path", "contents"],
            },
        },
    },
]
OPENCODE_WRITE_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "write",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["filePath", "content"],
            },
        },
    }
]
OPENCODE_SHELL_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "description": {"type": "string"},
                    "timeout": {"type": "number"},
                },
                "required": ["command", "description"],
            },
        },
    }
]
OPENCODE_READ_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                },
                "required": ["filePath"],
            },
        },
    }
]
OPENCODE_READ_LIMIT_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["filePath"],
            },
        },
    }
]
OPENCODE_QUESTIONS_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "question",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {"type": "array"},
                },
                "required": ["questions"],
            },
        },
    }
]


def _make(*, tools=TOOL_SPECS, tokenizer=None, suppress_tool_call_preamble=False):
    return _ToolAwareContentStreamTranslator(
        tools=tools,
        argument_chunk_chars=64,
        tokenizer=tokenizer,
        suppress_tool_call_preamble=suppress_tool_call_preamble,
    )


def _argument_text(deltas):
    return "".join(
        item.get("function", {}).get("arguments", "")
        for delta in deltas
        for item in delta.get("tool_calls", [])
    )


def _feed_in_chunks(translator, text, chunks):
    deltas = []
    offset = 0
    for size in chunks:
        if offset >= len(text):
            break
        deltas.extend(translator.feed("content", text[offset : offset + size]))
        offset += size
    if offset < len(text):
        deltas.extend(translator.feed("content", text[offset:]))
    deltas.extend(translator.finish())
    return deltas


def _content_text(deltas):
    return "".join(delta.get("content", "") for delta in deltas)


def _assert_no_tool_markup_leaked(content):
    assert "<tool_call" not in content
    assert "_call>" not in content
    assert "<function=" not in content
    assert "</think>" not in content


# ---------- existing behaviour (regression coverage) ----------

def test_no_tools_passthrough():
    """Without tools, every chunk passes through as-is."""
    t = _make(tools=[])
    assert t.feed("content", "hello ") == [{"content": "hello "}]
    assert t.feed("content", "world") == [{"content": "world"}]
    assert t.finish() == []


def test_pure_text_response():
    """Text-only response with tools available: emit as content."""
    t = _make()
    out = t.feed("content", "Plain answer.")
    assert {"content": "Plain answer."} in out
    assert t.finish() == []
    assert t.has_tool_calls is False


def test_pure_tool_call_response_streamed_in_pieces():
    """Tool-only response (the v0.2.0 happy path) still works after the patch."""
    t = _make()
    deltas = []
    # Stream the marker in pieces so we exercise the prefix-hold path
    assert t.feed("content", "<tool_") == []
    deltas.extend(t.feed("content", "call>\n<function=lookup>\n"))
    deltas.extend(t.feed("content", "<parameter=q>\nhello\n</parameter>\n"))
    deltas.extend(t.feed("content", "</function>\n</tool_call>"))
    deltas.extend(t.finish())
    assert t.has_tool_calls is True
    assert any("tool_calls" in d for d in deltas)


def test_opencode_read_tool_call_stream_orders_target_before_limit():
    t = _make(tools=OPENCODE_READ_LIMIT_TOOL_SPECS)
    out = t.feed(
        "content",
        "<tool_call>\n"
        "<function=read>\n"
        "<parameter=limit>\n100\n</parameter>\n"
        "<parameter=filePath>\nsrc/game/Game.ts\n</parameter>\n"
        "</function>\n"
        "</tool_call>",
    )
    out.extend(t.finish())

    arguments = _argument_text(out)
    assert arguments.startswith('{"filePath":"src/game/Game.ts","limit":100')
    assert json.loads(arguments) == {"filePath": "src/game/Game.ts", "limit": 100}
    assert _content_text(out) == ""


# ---------- the NEW bug-fix coverage ----------

def test_mixed_text_then_tool_call_in_one_chunk():
    """Issue #20: model emits preamble text + tool_call in one chunk.
    Old code emitted the entire chunk as content (markup leaked).
    New code: emit preamble as content, switch to tool mode, parse correctly."""
    t = _make()
    out = t.feed(
        "content",
        "Let me search.\n<tool_call>\n<function=lookup>\n"
        "<parameter=q>\nhello\n</parameter>\n</function>\n</tool_call>",
    )
    # Should have emitted the preamble as content
    contents = [d["content"] for d in out if "content" in d]
    assert "Let me search.\n" in "".join(contents)
    finish_deltas = t.finish()
    all_deltas = out + finish_deltas
    assert t.has_tool_calls is True
    assert any("tool_calls" in d for d in all_deltas), \
        "tool_calls should be emitted as soon as the XML function appears"


def test_mixed_text_then_tool_call_streamed_in_pieces():
    """Same as above but with the marker arriving in a separate chunk
    (verifies the partial-marker hold logic in content mode)."""
    t = _make()
    deltas_a = t.feed("content", "I will investigate. ")
    # Preamble emits as content
    assert any(d.get("content") == "I will investigate. " for d in deltas_a)
    # Now the marker arrives
    all_deltas = list(deltas_a)
    all_deltas.extend(t.feed(
        "content",
        "<tool_call>\n<function=lookup>\n"
        "<parameter=q>\ny\n</parameter>\n"
        "</function>\n</tool_call>",
    ))
    all_deltas.extend(t.finish())
    assert t.has_tool_calls is True
    assert any("tool_calls" in d for d in all_deltas)


def test_hermes_mode_suppresses_preamble_when_turn_contains_tool_calls():
    """Hermes renders streamed content immediately, so internal tool-plan labels
    must not be emitted as visible text alongside structured tool calls."""
    t = _make(suppress_tool_call_preamble=True)

    assert t.feed("content", "search\n\nMac Studio M4 Max\n\n") == []
    out = t.feed(
        "content",
        "<tool_call>\n<function=lookup>\n"
        "<parameter=q>\nMac Studio M4 Max\n</parameter>\n"
        "</function>\n</tool_call>",
    )
    out.extend(t.finish())

    assert t.has_tool_calls is True
    assert any("tool_calls" in delta for delta in out)
    assert _content_text(out) == ""


def test_hermes_mode_releases_buffer_for_genuine_text_only_turn():
    t = _make(suppress_tool_call_preamble=True)

    assert t.feed("content", "A normal ") == []
    assert t.feed("content", "answer.") == []
    assert t.finish() == [{"content": "A normal answer."}]


def test_partial_marker_held_across_chunks_in_content_mode():
    """If text + partial marker arrive together, the partial bytes must be
    held so the marker can complete on the next chunk."""
    t = _make()
    out_a = t.feed("content", "preamble<tool_ca")
    contents_a = [d["content"] for d in out_a if "content" in d]
    # 'preamble' should be emitted, '<tool_ca' should be held
    assert "preamble" in "".join(contents_a)
    assert not any("<tool_ca" in c for c in contents_a)

    # Complete the marker
    all_deltas = list(out_a)
    all_deltas.extend(t.feed(
        "content",
        "ll>\n<function=lookup>\n"
        "<parameter=q>\ny\n</parameter>\n"
        "</function>\n</tool_call>",
    ))
    all_deltas.extend(t.finish())
    assert t.has_tool_calls is True
    assert any("tool_calls" in d for d in all_deltas)


def test_lookalike_marker_prefix_in_content_does_not_block_emission():
    """Text containing chars that look like marker bytes but cannot complete
    must eventually be emitted, not held forever."""
    t = _make()
    out = t.feed("content", "this is text with <html tag>")
    # The trailing "<" might briefly be held but a follow-up that disambiguates
    # should release it. In a single-chunk feed, the held tail is "<" because
    # "<" is a prefix of "<tool_call". Subsequent text resolves it.
    out_next = t.feed("content", " and more text")
    # By the end of these two feeds, all the content text should have been emitted
    contents = [d["content"] for d in out + out_next if "content" in d]
    full = "".join(contents)
    assert "this is text with <html tag>" in full
    # No tool_calls
    assert t.finish() == []
    assert t.has_tool_calls is False


def test_marker_split_across_three_chunks_after_text():
    """`text` then `<tool_` then `call>...` - exercises the held-suffix path
    multiple times across content-mode feeds."""
    t = _make()
    a = t.feed("content", "hello ")
    b = t.feed("content", "<tool_")
    c = t.feed("content", "call>\n<function=lookup>\n"
                          "<parameter=q>\ny\n</parameter>\n"
                          "</function>\n</tool_call>")
    finish_deltas = t.finish()
    contents = [d["content"] for d in (a + b + c) if "content" in d]
    assert "hello " in "".join(contents)
    assert t.has_tool_calls is True
    assert any("tool_calls" in d for d in (a + b + c + finish_deltas))


def test_leading_whitespace_before_marker_still_dropped():
    """Existing behaviour: a tool-only response with leading whitespace
    drops the whitespace (it isn't content). Make sure the patch preserves
    this for the tool-only happy path."""
    t = _make()
    out = t.feed(
        "content",
        "\n\n<tool_call>\n<function=lookup>\n"
        "<parameter=q>\nhello\n</parameter>\n</function>\n</tool_call>",
    )
    contents = [d["content"] for d in out if "content" in d]
    # No whitespace-only content should leak (was dropped in original undecided path)
    assert all(c.strip() for c in contents) or contents == []
    finish_deltas = t.finish()
    assert t.has_tool_calls is True
    assert any("tool_calls" in d for d in (out + finish_deltas))


def test_unknown_tool_name_passes_through_as_tool_call():
    # OpenAI-compatible contract: an unknown-named call is surfaced to the
    # client, which owns the rejection (and answers the model so it can
    # self-correct). Suppressing it degraded the whole turn to prose and cost
    # agent clients (Cline) a consecutive-mistake strike per occurrence.
    t = _make()
    text = (
        "<tool_call>\n<function=Agent>\n"
        "<parameter=description>\nList files\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    out = t.feed("content", text)
    finish_deltas = t.finish()
    assert t.has_tool_calls is True
    assert t.fallback_reason is None
    named = [
        d for d in (out + finish_deltas)
        if any(
            (c.get("function") or {}).get("name") == "Agent"
            for c in (d.get("tool_calls") or [])
        )
    ]
    assert named, "unknown-named call should stream as a tool_call delta"
    assert _content_text(out) == ""


def test_unclosed_tool_call_suppresses_raw_tool_markup():
    t = _make()
    text = "<tool_call>\n<function=lookup>\n"
    assert t.feed("content", text) == []
    out = t.finish()
    assert t.has_tool_calls is False
    assert t.fallback_reason == "unclosed <tool_call> block"
    assert _content_text(out) == ""


def test_missing_required_tool_argument_suppresses_raw_tool_markup():
    t = _make(tools=OPENCODE_SHELL_TOOL_SPECS)
    text = "<tool_call>\n<function=bash>\n</function>\n</tool_call>"
    out = t.feed("content", text)
    assert t.finish() == []
    assert t.has_tool_calls is False
    assert "missing required argument" in (t.fallback_reason or "")
    assert "command" in (t.fallback_reason or "")
    assert _content_text(out) == ""


def test_qwen_xml_tool_call_emits_after_arguments_validate():
    t = _make(tools=WRITE_FILE_TOOL_SPECS)
    assert t.feed("content", "<tool_call>\n") == []
    assert t.feed("content", "<function=write_file>\n") == []
    name_deltas = t.feed("content", "<parameter=path>")
    assert name_deltas == []
    arg_deltas = []
    arg_deltas.extend(t.feed("content", "app.js</parameter>\n"))
    arg_deltas.extend(t.feed("content", "<parameter=contents>const label = 'he"))
    arg_deltas.extend(t.feed("content", "llo 🌍';\n</parameter>\n"))
    arg_deltas.extend(t.feed("content", "</function>\n</tool_call>"))
    arg_deltas.extend(t.finish())

    args = json.loads(_argument_text(name_deltas + arg_deltas))
    assert args == {"path": "app.js", "contents": "const label = 'hello 🌍';"}
    assert "\\ud83c" not in _argument_text(name_deltas + arg_deltas)
    assert t.has_tool_calls is True
    assert t.tool_parser_dialect == "qwen_xml"


def test_consecutive_qwen_xml_tool_calls_do_not_leak_as_content():
    t = _make()
    text = (
        "<tool_call>\n<function=lookup>\n"
        "<parameter=q>\none\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=lookup>\n"
        "<parameter=q>\ntwo\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    out = t.feed("content", text)
    out.extend(t.finish())

    assert not any("<tool_call" in d.get("content", "") for d in out)
    assert t.tool_calls is not None
    assert len(t.tool_calls) == 2
    args = [
        json.loads(call["function"]["arguments"])
        for call in t.tool_calls
    ]
    assert args == [{"q": "one"}, {"q": "two"}]
    indices = [
        item.get("index")
        for delta in out
        for item in delta.get("tool_calls", [])
        if item.get("function", {}).get("name") == "lookup"
    ]
    assert indices == [0, 1]


def test_consecutive_qwen_xml_tool_call_close_split_does_not_leak_tail():
    t = _make()
    out = []
    out.extend(t.feed("content", "<tool_call>\n<function=lookup>\n"))
    out.extend(t.feed("content", "<parameter=q>\none\n</parameter>\n</function>\n</tool"))
    out.extend(t.feed("content", "_call>\n<tool_call>\n<function=lookup>\n"))
    out.extend(t.feed("content", "<parameter=q>\ntwo\n</parameter>\n</function>\n</tool_call>"))
    out.extend(t.finish())

    content = "".join(d.get("content", "") for d in out)
    assert "_call>" not in content
    assert "<tool_call" not in content
    assert t.tool_calls is not None
    assert len(t.tool_calls) == 2


def test_qwen_xml_three_tool_calls_streamed_5char_chunks_no_leak():
    """Regression for Daniel's in-the-wild hippo-code/OpenAI stream shape.

    v0.3.3 parsed the first Qwen XML tool call and leaked the following two as
    visible content when the stream arrived in tiny chunks. Keep this exact
    same-name, blank-line-separated, 5-character stream shape locked down.
    """
    t = _make()
    text = (
        "<tool_call>\n<function=lookup>\n<parameter=q>\n"
        "src/scene\n</parameter>\n</function>\n</tool_call>\n\n"
        "<tool_call>\n<function=lookup>\n<parameter=q>\n"
        "src/entities\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=lookup>\n<parameter=q>\n"
        "src/systems\n</parameter>\n</function>\n</tool_call>"
    )
    out = _feed_in_chunks(t, text, [5] * ((len(text) // 5) + 1))

    content = _content_text(out)
    _assert_no_tool_markup_leaked(content)
    assert t.tool_calls is not None
    assert len(t.tool_calls) == 3
    args = [json.loads(call["function"]["arguments"]) for call in t.tool_calls]
    assert args == [
        {"q": "src/scene"},
        {"q": "src/entities"},
        {"q": "src/systems"},
    ]
    indices = [
        item.get("index")
        for delta in out
        for item in delta.get("tool_calls", [])
        if item.get("function", {}).get("name") == "lookup"
    ]
    assert indices == [0, 1, 2]


def test_qwen_xml_mixed_tool_calls_uneven_chunks_no_leak():
    """Mixed tool names should remain ordered when chunks cut across tags."""
    t = _make(tools=MIXED_TOOL_SPECS)
    text = (
        "<tool_call>\n<function=lookup>\n<parameter=q>\n"
        "entities\n</parameter>\n</function>\n</tool_call>\n\n"
        "<tool_call>\n<function=read_file>\n<parameter=path>\n"
        "src/game.ts\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=write_file>\n<parameter=path>\n"
        "src/out.ts\n</parameter>\n<parameter=contents>\n"
        "export const ok = true;\n</parameter>\n</function>\n</tool_call>"
    )
    out = _feed_in_chunks(t, text, [1, 9, 2, 17, 4, 31, 3, 8, 5, 21])

    content = _content_text(out)
    _assert_no_tool_markup_leaked(content)
    assert t.tool_calls is not None
    assert [call["function"]["name"] for call in t.tool_calls] == [
        "lookup",
        "read_file",
        "write_file",
    ]
    args = [json.loads(call["function"]["arguments"]) for call in t.tool_calls]
    assert args == [
        {"q": "entities"},
        {"path": "src/game.ts"},
        {"path": "src/out.ts", "contents": "export const ok = true;"},
    ]


def test_qwen_xml_opening_marker_split_after_preamble_no_leak():
    t = _make()
    out = []
    out.extend(t.feed("content", "Checking sources. <too"))
    out.extend(t.feed("content", "l_call>\n<function=lookup>\n"))
    out.extend(t.feed("content", "<parameter=q>\nscene\n</parameter>\n"))
    out.extend(t.feed("content", "</function>\n</tool_call>"))
    out.extend(t.finish())

    content = _content_text(out)
    assert "Checking sources. " in content
    _assert_no_tool_markup_leaked(content)
    assert t.tool_calls is not None
    assert len(t.tool_calls) == 1
    assert json.loads(t.tool_calls[0]["function"]["arguments"]) == {"q": "scene"}


def test_opencode_style_long_write_arguments_stream_without_raw_xml():
    """OpenCode-style write(filePath, content) args can be large and chunked."""
    t = _make(tools=OPENCODE_WRITE_TOOL_SPECS)
    long_content = "\n".join(
        [
            "export function hello(name) {",
            "  const payload = { quote: \"hello\", emoji: \"🌍\" };",
            "  return `${payload.quote}, ${name}!`;",
            "}",
        ]
        * 8
    )
    text = (
        "<tool_call>\n<function=write>\n"
        "<parameter=filePath>\nsrc/hello.ts\n</parameter>\n"
        f"<parameter=content>\n{long_content}\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    out = _feed_in_chunks(t, text, [7, 13, 1, 5, 23, 3, 19, 2, 29, 11])

    content = _content_text(out)
    _assert_no_tool_markup_leaked(content)
    assert t.tool_calls is not None
    assert len(t.tool_calls) == 1
    args = json.loads(t.tool_calls[0]["function"]["arguments"])
    assert args == {"filePath": "src/hello.ts", "content": long_content}
    streamed_args = _argument_text(out)
    assert "\\ud83c" not in streamed_args
    assert json.loads(streamed_args) == args


def test_opencode_style_content_first_long_write_marks_argument_in_progress():
    """Content-first file writes are recognized as active tool arguments."""
    t = _make(tools=OPENCODE_WRITE_TOOL_SPECS)
    long_content = "\n".join(
        [
            "<!DOCTYPE html>",
            "<html>",
            "<body>",
            "<script>",
            "const label = 'flappy';",
            "</script>",
            "</body>",
            "</html>",
        ]
        * 32
    )
    out = []
    out.extend(t.feed("content", "<tool_call>\n<function=write>\n"))
    out.extend(t.feed("content", "<parameter=content>\n"))
    assert out == []
    assert t.tool_argument_in_progress is True

    offset = 0
    for size in [3, 41, 7, 19, 2, 53]:
        out.extend(t.feed("content", long_content[offset : offset + size]))
        offset += size
    if offset < len(long_content):
        out.extend(t.feed("content", long_content[offset:]))
    out.extend(t.feed("content", "\n</parameter>\n"))
    out.extend(t.feed("content", "<parameter=filePath>\nindex.html\n</parameter>\n"))
    out.extend(t.feed("content", "</function>\n</tool_call>"))
    out.extend(t.finish())

    content = _content_text(out)
    _assert_no_tool_markup_leaked(content)
    assert t.tool_calls is not None
    assert len(t.tool_calls) == 1
    args = json.loads(t.tool_calls[0]["function"]["arguments"])
    assert args == {"content": long_content, "filePath": "index.html"}
    assert json.loads(_argument_text(out)) == args


def test_opencode_shell_xml_stream_emits_complete_typed_arguments():
    t = _make(tools=OPENCODE_SHELL_TOOL_SPECS)
    text = (
        "<tool_call>\n<function=bash>\n"
        "<parameter=command>\nnpm run build\n</parameter>\n"
        "<parameter=description>\nBuild the project\n</parameter>\n"
        "<parameter=timeout>\n60000\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    out = _feed_in_chunks(t, text, [11, 2, 19, 5, 7, 3, 23, 1])

    args_text = _argument_text(out)
    args = json.loads(args_text)
    assert args == {
        "command": "npm run build",
        "description": "Build the project",
        "timeout": 60000,
    }
    assert isinstance(args["timeout"], int)
    assert '"timeout":"60000"' not in args_text


def test_opencode_shell_xml_unwraps_schema_type_placeholder_values():
    t = _make(tools=OPENCODE_SHELL_TOOL_SPECS)
    text = (
        "<tool_call>\n<function=bash>\n"
        "<parameter=command>\n<string>\npwd && ls -F\n</string>\n</parameter>\n"
        "<parameter=description>\n<string>\nRun pwd and list files\n</string>\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    out = _feed_in_chunks(t, text, [13, 5, 29, 7, 11, 3, 17])

    args = json.loads(_argument_text(out))
    assert args == {
        "command": "pwd && ls -F",
        "description": "Run pwd and list files",
    }
    assert "<string>" not in args["command"]


def test_opencode_shell_xml_unescapes_parameter_entities():
    t = _make(tools=OPENCODE_SHELL_TOOL_SPECS)
    text = (
        "<tool_call>\n<function=bash>\n"
        "<parameter=command>\npwd &amp;&amp; ls -F\n</parameter>\n"
        "<parameter=description>\nRun pwd and list files\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    out = _feed_in_chunks(t, text, [7, 19, 3, 11, 5, 17, 2])

    args = json.loads(_argument_text(out))
    assert args["command"] == "pwd && ls -F"


def test_opencode_json_tool_call_unwraps_schema_type_placeholder_values():
    t = _make(tools=OPENCODE_READ_TOOL_SPECS)
    text = (
        '<tool_call>{"name":"read","arguments":'
        '{"filePath":"<string>package.json</string>"}}</tool_call>'
    )
    out = _feed_in_chunks(t, text, [9, 13, 7, 5, 31, 11])

    args = json.loads(_argument_text(out))
    assert args == {"filePath": "package.json"}


def test_tool_call_example_does_not_invite_literal_type_wrappers():
    example = _tool_call_example(OPENCODE_SHELL_TOOL_SPECS)

    assert "<string>" not in example
    assert "ARGUMENT_VALUE" in example
    assert "<parameter=command>" in example


def test_opencode_questions_xml_stream_emits_array_not_string():
    t = _make(tools=OPENCODE_QUESTIONS_TOOL_SPECS)
    questions = (
        '[{"header":"Scope","id":"scope","question":"Pick one",'
        '"options":[{"label":"A","description":"first"},'
        '{"label":"B","description":"second"}]}]'
    )
    text = (
        "<tool_call>\n<function=question>\n"
        f"<parameter=questions>\n{questions}\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    out = _feed_in_chunks(t, text, [4, 4, 9, 2, 17, 1, 31, 8])

    args = json.loads(_argument_text(out))
    assert isinstance(args["questions"], list)
    assert args["questions"][0]["options"][1]["label"] == "B"


def test_existing_json_tool_call_final_parse_still_works():
    t = _make()
    text = '<tool_call>{"name":"lookup","arguments":{"q":"hello"}}</tool_call>'
    deltas = t.feed("content", text)
    deltas.extend(t.finish())
    assert t.has_tool_calls is True
    assert json.loads(_argument_text(deltas)) == {"q": "hello"}
    assert t.tool_parser_dialect == "buffered"


def test_closed_json_tool_call_stream_finishes_without_waiting_for_model_stop():
    t = _make()
    text = (
        'Let me check. <tool_call>{"tool":"lookup",'
        '"args":{"q":"hello"}}</tool_call>'
    )
    deltas = _feed_in_chunks(t, text, [5, 12, 8, 9, 11, 7])

    assert t.has_tool_calls is True
    assert t.ready_to_finish_tool_turn is True
    assert t.tool_parser_dialect == "buffered"
    assert _content_text(deltas) == "Let me check. "
    assert json.loads(_argument_text(deltas)) == {"q": "hello"}


def test_unclosed_complete_qwen_xml_tool_call_is_repaired_at_finish():
    t = _make(tools=OPENCODE_SHELL_TOOL_SPECS)
    text = (
        "<tool_call>\n<function=bash>\n"
        "<parameter=command>\npwd\n</parameter>\n"
        "<parameter=description>\nPrint working directory\n</parameter>"
    )
    assert t.feed("content", text) == []
    deltas = t.finish()

    assert t.has_tool_calls is True
    args = json.loads(_argument_text(deltas))
    assert args == {
        "command": "pwd",
        "description": "Print working directory",
    }


def test_unclosed_incomplete_parameter_suppresses_raw_tool_markup():
    t = _make(tools=OPENCODE_SHELL_TOOL_SPECS)
    text = (
        "<tool_call>\n<function=bash>\n"
        "<parameter=command>\npwd\n</parameter>\n"
        "<parameter=description>\n"
    )
    deltas = t.feed("content", text)
    deltas.extend(t.finish())
    assert t.has_tool_calls is False
    assert t.fallback_reason == "unclosed <tool_call> block"
    content = _content_text(deltas)
    assert content == ""
    _assert_no_tool_markup_leaked(content)


def test_namespaced_tool_call_parses_without_markup_leak():
    t = _make()
    text = (
        "Let me check."
        "<foo-bar:tool_call>"
        '<invoke name="lookup">'
        '<parameter name="q">"scene"</parameter>'
        "</invoke>"
        "</foo-bar:tool_call>"
    )
    out = _feed_in_chunks(t, text, [3, 2, 11, 5, 7, 13])

    content = _content_text(out)
    assert content == "Let me check."
    _assert_no_tool_markup_leaked(content)
    assert "<foo-bar:tool_call>" not in content
    assert t.tool_calls is not None
    assert json.loads(t.tool_calls[0]["function"]["arguments"]) == {"q": "scene"}


def test_bracket_tool_call_parses_and_suppresses_partial_prefix():
    t = _make()
    out = []
    out.extend(t.feed("content", "Let me check. [Calling tool:"))
    out.extend(t.feed("content", ' lookup({"q":"scene"})]'))
    out.extend(t.finish())

    content = _content_text(out)
    assert content == "Let me check. "
    assert "[Calling tool:" not in content
    assert t.tool_calls is not None
    assert json.loads(t.tool_calls[0]["function"]["arguments"]) == {"q": "scene"}


def test_literal_bracket_prefix_before_valid_tool_call_is_sanitized_not_leaked():
    t = _make()
    text = (
        "Before [Calling tool: unfinished and then "
        '[Calling tool: lookup({"q":"scene"})] done'
    )
    out = _feed_in_chunks(t, text, [len(text)])

    content = _content_text(out)
    assert "[Calling tool:" not in content
    assert content == "Before  unfinished and then "
    assert t.tool_calls is not None
    assert json.loads(t.tool_calls[0]["function"]["arguments"]) == {"q": "scene"}


class _CustomToolTokenizer:
    tool_call_start = "<|tool|>"
    tool_call_end = "<|/tool|>"


def test_tokenizer_defined_tool_delimiters_parse_json_payload():
    t = _make(tokenizer=_CustomToolTokenizer())
    text = 'Before <|tool|>{"name":"lookup","arguments":{"q":"scene"}}<|/tool|>'
    out = _feed_in_chunks(t, text, [6, 4, 9, 3, 5, 8])

    content = _content_text(out)
    assert content == "Before "
    assert "<|tool|>" not in content
    assert t.tool_calls is not None
    assert json.loads(t.tool_calls[0]["function"]["arguments"]) == {"q": "scene"}


def test_schema_type_mismatch_suppresses_raw_tool_markup():
    t = _make(tools=OPENCODE_SHELL_TOOL_SPECS)
    text = (
        "<tool_call>\n<function=bash>\n"
        "<parameter=command>\npwd\n</parameter>\n"
        "<parameter=description>\nPrint working directory\n</parameter>\n"
        "<parameter=timeout>\nsoon\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    out = t.feed("content", text)
    assert t.finish() == []
    assert t.has_tool_calls is False
    assert "timeout must be number" in (t.fallback_reason or "")
    assert _content_text(out) == ""


def test_bracket_call_streams_as_tool_call_not_content():
    # Live shape 2026-07-25: prose, then a complete [Calling tool: ...] block
    # whose arguments contain `]` (task_progress checklist) and `})]` inside
    # strings. The old start detector skipped the prefix on the early `]`,
    # so the block streamed as content AND the finish-time rescue emitted the
    # same call — a double delivery.
    payload = {
        "input": "*** Begin Patch\n+const f = (x) => ({y: x});\ncall(f(1))]\n*** End Patch",
        "task_progress": "- [x] one\n- [ ] two",
    }
    text = (
        "Let me start with the math utilities: [Calling tool: apply_patch("
        + json.dumps(payload)
        + ")]"
    )
    t = _make(
        tools=[
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
    )
    out = []
    for i in range(0, len(text), 7):  # simulate streaming in small chunks
        out += t.feed("content", text[i : i + 7])
    out += t.finish()
    assert t.has_tool_calls is True, t.fallback_reason
    contents = _content_text(out)
    assert "[Calling tool:" not in contents
    assert "Begin Patch" not in contents
    args = _argument_text(out)
    assert json.loads(args) == payload


def test_unterminated_bracket_call_suppresses_like_unclosed_markup():
    # Same contract as test_unclosed_tool_call_suppresses_raw_tool_markup:
    # a never-completing block is suppressed with a recorded reason, keeping
    # the drift dialect out of the visible transcript (and the model's
    # conversation history).
    text = 'And now: [Calling tool: apply_patch({"input": "never closed'
    t = _make()
    out = []
    for i in range(0, len(text), 9):
        out += t.feed("content", text[i : i + 9])
    out += t.finish()
    assert t.has_tool_calls is False
    assert t.fallback_reason
    assert _content_text(out) == "And now: "
