"""#691: tool-call arguments reach the client exactly as the model wrote them.

The Qwen 3.x template writes a parameter as ``<parameter=k>\\n{value}\\n</parameter>``,
so the exact inverse removes one newline after the opening tag and one before
the closing tag, nothing else. Before the fix every parser path trimmed all
whitespace from a value: a file an agent wrote lost its last newline, a
value's leading indentation was cut, a run of blank lines at the end vanished,
and an HTML entity in written text was decoded. In streams the visible-text
edge trim also dropped the newlines before ``</parameter>`` at a chunk edge
before the tool parser saw them; the final parser read a written JSON file or
a file holding only "42" as an object or a number and then refused the call;
and the arguments were reordered (path first), so the call the client sent
back no longer rendered as the tokens the model generated.

Paths covered: the bridge's XML parser (non-stream fallback and recovery),
the server's final parser (JSON-dialect and unclosed calls in streams), the
streaming translator at every chunking, and the visible-text splitter feeding
the translator as the server wires them. Values the schema types as numbers,
booleans or arrays are still read as JSON, and visible text before a call is
still trimmed.
"""

from __future__ import annotations

import json

import pytest

from mtplx.server import openai as oa
from mtplx.server.omlx_bridge import tool_calling as bridge

TOOLS = [
    {"type": "function", "function": {"name": "write", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"},
        "raw": {"type": "boolean"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "edit", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "oldString": {"type": "string"}, "newString": {"type": "string"},
        "lines": {"type": "array", "items": {"type": "integer"}}}}}},
]

# Values that differ from their trimmed or JSON-read form, plus ordinary ones.
CONTENTS = [
    "class X:\n    pass\n",
    "x = 1\n\n",
    "  x = 1  ",
    "\n\nstarts with a blank line\n",
    "<p>a &lt;b&gt; &amp; c</p>\n",
    '{\n  "name": "x"\n}\n',
    "42\n",
    "true",
    "no newline at all",
    "\n",
    " ",
]


def _call(name: str, params: list[tuple[str, str]]) -> str:
    """What the model writes, in the template's own layout."""
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params)
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def _args(call: dict) -> dict:
    return json.loads(call["function"]["arguments"])


# ---------- the bridge's XML parser (the reporter's offline repro) ----------


@pytest.mark.parametrize("content", CONTENTS)
def test_bridge_keeps_a_written_file_exactly(content):
    result = bridge.extract_tool_calls_with_thinking(
        "", _call("write", [("path", "a.py"), ("content", content)]), None, TOOLS
    )
    assert _args(result.tool_calls[0]) == {"path": "a.py", "content": content}


def test_bridge_still_reads_typed_values_as_json():
    result = bridge.extract_tool_calls_with_thinking(
        "", _call("read", [("path", "a.py"), ("offset", "147"), ("limit", " 50 "), ("raw", "true")]),
        None, TOOLS,
    )
    assert _args(result.tool_calls[0]) == {"path": "a.py", "offset": 147, "limit": 50, "raw": True}


def test_bridge_without_a_schema_keeps_text_and_reads_json():
    result = bridge.extract_tool_calls_with_thinking(
        "", _call("anything", [("text", "keep me\n"), ("n", "3")]), None, None
    )
    assert _args(result.tool_calls[0]) == {"text": "keep me\n", "n": 3}


def test_bridge_keeps_the_models_argument_order():
    result = bridge.extract_tool_calls_with_thinking(
        "", _call("read", [("limit", "50"), ("offset", "147"), ("path", "a.py")]), None, TOOLS
    )
    assert list(_args(result.tool_calls[0])) == ["limit", "offset", "path"]


# ---------- the server's final parser and the schema pass ----------


@pytest.mark.parametrize("content", CONTENTS)
def test_final_parser_keeps_a_written_file_exactly(content):
    calls = oa._parse_generated_tool_calls(
        _call("write", [("path", "a.py"), ("content", content)]), tools=TOOLS
    )
    assert _args(calls[0]) == {"path": "a.py", "content": content}


def test_final_parser_reads_typed_values_and_keeps_edit_strings():
    calls = oa._parse_generated_tool_calls(
        _call("edit", [("path", "a.py"), ("oldString", "    return x\n"),
                       ("newString", "    return y\n"), ("lines", "[3, 4]")]),
        tools=TOOLS,
    )
    assert _args(calls[0]) == {
        "path": "a.py", "oldString": "    return x\n", "newString": "    return y\n", "lines": [3, 4],
    }


def test_final_parser_keeps_the_models_argument_order():
    calls = oa._parse_generated_tool_calls(
        _call("read", [("limit", "50"), ("offset", "147"), ("path", "a.py")]), tools=TOOLS
    )
    assert json.loads(calls[0]["function"]["arguments"]) == {"limit": 50, "offset": 147, "path": "a.py"}
    assert list(_args(calls[0])) == ["limit", "offset", "path"]


def test_a_decoded_string_decodes_to_itself():
    """The stream parser decodes with the schema and the schema pass runs
    over its result again: a second pass must change nothing."""
    for content in CONTENTS:
        once = oa._decode_tool_parameter_value(content, {"type": "string"})
        assert once == content
        assert oa._decode_tool_parameter_value(once, {"type": "string"}) == content
    assert oa._decode_tool_parameter_value("\n") == "\n"
    assert oa._decode_tool_parameter_value(" 50 ", {"type": "integer"}) == 50


def test_a_placeholder_type_wrapper_is_still_unwrapped():
    assert oa._decode_tool_parameter_value("<integer>50</integer>", {"type": "integer"}) == 50
    assert oa._decode_tool_parameter_value("<string>a.py</string>", {"type": "string"}) == "a.py"


# ---------- the streaming translator, every chunking ----------


def _streamed_arguments(deltas: list[dict]) -> str:
    return "".join(
        item.get("function", {}).get("arguments", "")
        for delta in deltas
        for item in delta.get("tool_calls", [])
    )


def _stream(text: str, chunk: int) -> tuple[oa._ToolAwareContentStreamTranslator, str]:
    translator = oa._ToolAwareContentStreamTranslator(tools=TOOLS, argument_chunk_chars=64, tokenizer=None)
    deltas: list[dict] = []
    for start in range(0, len(text), chunk):
        deltas.extend(translator.feed("content", text[start : start + chunk]))
    deltas.extend(translator.finish())
    return translator, _streamed_arguments(deltas)


@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 7, 11, 12, 13, 64, 10_000])
@pytest.mark.parametrize("content", CONTENTS)
def test_streaming_translator_keeps_a_written_file_exactly(content, chunk):
    translator, arguments = _stream(_call("write", [("path", "a.py"), ("content", content)]), chunk)
    assert translator.has_tool_calls, translator.fallback_reason
    assert json.loads(arguments) == {"path": "a.py", "content": content}


@pytest.mark.parametrize("chunk", [1, 4, 10_000])
def test_streaming_translator_keeps_the_models_argument_order(chunk):
    _translator, arguments = _stream(_call("read", [("limit", "50"), ("offset", "147"), ("path", "a.py")]), chunk)
    assert arguments == '{"limit":50,"offset":147,"path":"a.py"}'


def test_a_chunk_ending_inside_the_closing_tag_adds_no_newline():
    # The newline before "</parameter>" is the template's, also when a chunk
    # ends one character short of the closing tag.
    text = _call("write", [("path", "a.py"), ("content", "x\n")])
    cut = text.index("</parameter>") + len("</parameter")
    translator = oa._ToolAwareContentStreamTranslator(tools=TOOLS, argument_chunk_chars=64, tokenizer=None)
    deltas = translator.feed("content", text[:cut]) + translator.feed("content", text[cut:]) + translator.finish()
    assert json.loads(_streamed_arguments(deltas)) == {"path": "a.py", "content": "x\n"}


# ---------- the splitter feeding the translator, as the server wires them ----------


def _splitter() -> oa._ThinkingContentStreamSplitter:
    return oa._ThinkingContentStreamSplitter(
        thinking_enabled=True,
        recover_unclosed_reasoning_as_content=False,
        start_inside_thinking=True,
        suppress_orphan_tool_markup=False,
        trim_visible_content_edges=True,
        tool_names=[tool["function"]["name"] for tool in TOOLS],
    )


def _server_stream(model_text: str, pieces: list[str]) -> tuple[str, str, oa._ToolAwareContentStreamTranslator]:
    """Model text -> visible/reasoning splitter (edge trim on, as served) ->
    tool translator, fed in the given pieces."""
    assert "".join(pieces) == model_text
    splitter = _splitter()
    translator = oa._ToolAwareContentStreamTranslator(tools=TOOLS, argument_chunk_chars=64, tokenizer=None)
    chunks = list(splitter.start())
    for piece in pieces:
        chunks.extend(splitter.feed(piece))
    chunks.extend(splitter.finish())
    deltas: list[dict] = []
    for field, text in chunks:
        if field == "content":
            deltas.extend(translator.feed("content", text))
    deltas.extend(translator.finish())
    visible = "".join(delta.get("content", "") for delta in deltas)
    return visible, _streamed_arguments(deltas), translator


def _sized(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def _cut(text: str, offsets: list[int]) -> list[str]:
    bounds = [0, *offsets, len(text)]
    return [text[a:b] for a, b in zip(bounds, bounds[1:])]


@pytest.mark.parametrize("content", CONTENTS)
def test_splitter_and_translator_keep_a_written_file_exactly(content):
    call = _call("write", [("path", "a.py"), ("content", content)])
    model_text = "Plan the file.\n</think>\n\nWriting it now.\n\n" + call
    close = model_text.index("</parameter>\n</function>")
    cuttings = [_sized(model_text, size) for size in (1, 2, 3, 5, 7, 9, 13, 64, 10_000)]
    cuttings += [
        _cut(model_text, [close]),  # the edge the old trim held and dropped
        _cut(model_text, [close - 1, close]),
        _cut(model_text, [model_text.index("<tool_call>"), close]),
    ]
    for pieces in cuttings:
        visible, arguments, translator = _server_stream(model_text, pieces)
        assert translator.has_tool_calls, (translator.fallback_reason, pieces)
        assert json.loads(arguments) == {"path": "a.py", "content": content}, pieces


@pytest.mark.parametrize("size", [1, 3, 7, 10_000])
def test_visible_text_before_a_call_is_still_trimmed(size):
    model_text = "Plan.\n</think>\n\nWriting it now.\n\n" + _call("write", [("path", "a.py"), ("content", "x\n")])
    visible, arguments, _ = _server_stream(model_text, _sized(model_text, size))
    assert visible.strip() == "Writing it now."
    assert not visible.startswith(("\n", " "))
    assert json.loads(arguments) == {"path": "a.py", "content": "x\n"}


def test_visible_text_of_a_turn_without_tool_calls_is_unchanged():
    model_text = "Plan.\n</think>\n\n  Line one.\n\n  Line two.\n\n"
    for size in (1, 4, 10_000):
        visible, arguments, _ = _server_stream(model_text, _sized(model_text, size))
        assert visible == "Line one.\n\n  Line two."
        assert arguments == ""


def test_two_calls_in_one_turn_both_keep_their_files():
    model_text = (
        "Plan.\n</think>\n\n"
        + _call("write", [("path", "a.py"), ("content", "a\n")])
        + "\n"
        + _call("write", [("path", "b.py"), ("content", "b\n\n")])
    )
    for size in (1, 4, 10_000):
        splitter = _splitter()
        chunks = list(splitter.start())
        for piece in _sized(model_text, size):
            chunks.extend(splitter.feed(piece))
        chunks.extend(splitter.finish())
        content = "".join(text for field, text in chunks if field == "content")
        calls = oa._parse_generated_tool_calls(content, tools=TOOLS)
        assert [_args(call) for call in calls] == [
            {"path": "a.py", "content": "a\n"},
            {"path": "b.py", "content": "b\n\n"},
        ], size


# ---------- identifiers, escaped values and the rollback switch ----------

IDENTIFIERS = {
    "path": "src/game/Game.ts",
    "command": "cd app && npm test 2>&1 | tail -5",
    "pattern": "^def \\w+\\(",
    "url": "https://example.com/a?b=1&c=2",
}
IDENTIFIER_TOOLS = [
    {"type": "function", "function": {"name": "probe", "parameters": {"type": "object", "properties": {
        key: {"type": "string"} for key in IDENTIFIERS}}}},
]


def _all_paths(text: str, tools: list[dict]) -> dict[str, dict]:
    """The arguments each parser path hands the client for one call."""
    results = {}
    results["final"] = _args(oa._parse_generated_tool_calls(text, tools=tools)[0])
    results["bridge"] = _args(bridge.extract_tool_calls_with_thinking("", text, None, tools).tool_calls[0])
    for size in (1, 5, 10_000):
        translator = oa._ToolAwareContentStreamTranslator(tools=tools, argument_chunk_chars=64, tokenizer=None)
        deltas: list[dict] = []
        for start in range(0, len(text), size):
            deltas.extend(translator.feed("content", text[start : start + size]))
        deltas.extend(translator.finish())
        results[f"stream/{size}"] = json.loads(_streamed_arguments(deltas))
    return results


def test_paths_commands_patterns_and_urls_arrive_exactly_as_before():
    # A well-formed call (the template's layout) gives no client a leading or
    # trailing space or newline on a path, a command, a pattern or a URL: the
    # template's own newlines are the only ones removed, and an "&" that does
    # not begin an entity (a shell "&&", a query string) stays.
    text = _call("probe", list(IDENTIFIERS.items()))
    for path, arguments in _all_paths(text, IDENTIFIER_TOOLS).items():
        assert arguments == IDENTIFIERS, path


def test_a_wholly_escaped_command_is_still_decoded():
    # MTPLX's own tool-result notices escape the commands they quote; a model
    # copying that must not hand a shell "&amp;&amp;".
    tools = [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {
        "command": {"type": "string"}}}}}]
    text = _call("bash", [("command", "pwd &amp;&amp; ls -F 2&gt;&amp;1")])
    for path, arguments in _all_paths(text, tools).items():
        assert arguments == {"command": "pwd && ls -F 2>&1"}, path


def test_an_html_page_keeps_its_entities():
    page = "<p>Use &lt;canvas&gt; &amp; JS</p>\n"
    text = _call("write", [("path", "index.html"), ("content", page)])
    for path, arguments in _all_paths(text, TOOLS).items():
        assert arguments == {"path": "index.html", "content": page}, path


def test_the_rollback_switch_restores_the_earlier_handling(monkeypatch):
    monkeypatch.setenv("MTPLX_TOOL_ARGUMENTS_EXACT", "0")
    text = _call("write", [("content", "a &amp; b\n\n"), ("path", " a.py ")])
    for path, arguments in _all_paths(text, TOOLS).items():
        expected = {"path": "a.py", "content": "a & b"}
        if path == "bridge":
            expected = {"path": "a.py", "content": "a &amp; b"}  # the bridge never decoded entities
        assert arguments == expected, path
        assert list(arguments) == ["path", "content"], path  # path first again
    # and the splitter trims inside a call again: the newlines held at a chunk
    # edge before "</parameter>" are dropped (harmless once values are trimmed)
    assert _edge_trimmed(_splitter(), _VALUE_AT_A_CHUNK_EDGE) == (
        "<tool_call>\n<function=write>\n<parameter=content>\nx</parameter>\n</function>\n</tool_call>"
    )


# A value that arrives as its own chunk, ending in its newlines, right before
# a chunk that starts with "</parameter>": the shape that lost every written
# file's last newline in streams.
_VALUE_AT_A_CHUNK_EDGE = [
    "<tool_call>\n<function=write>\n<parameter=content>\n",
    "x\n\n\n",
    "</parameter>\n</function>\n</tool_call>",
]


def _edge_trimmed(splitter, chunks: list[str]) -> str:
    return "".join(text for _field, text in splitter._trim_visible_edges([("content", c) for c in chunks]))


def test_the_edge_trim_passes_a_call_through_untouched():
    assert _edge_trimmed(_splitter(), _VALUE_AT_A_CHUNK_EDGE) == "".join(_VALUE_AT_A_CHUNK_EDGE)
    # also when the opening marker is split across two chunks
    split = ["Done.\n\n<tool_", "call>\n<function=write>\n<parameter=content>\n", "x\n\n\n",
             "</parameter>\n</function>\n</tool_call>"]
    assert _edge_trimmed(_splitter(), split) == "".join(split)


# ---------- end to end: the server's routes, in each client's request shape ----------
#
# A scripted model (one character per token, as the server's other stream
# tests drive it) writes a file whose last line ends in a newline; the client
# must receive that file byte for byte on every route a coding agent uses.

from fastapi.testclient import TestClient  # noqa: E402

from test_server_openai import (  # noqa: E402 - shared fixtures
    _anthropic_events,
    _fake_state,
    _fake_streaming_generation,
    _stream_payloads,
)

_FILE = "def main():\n    print('hi')\n\n\nmain()\n"


def _model_writes(path_key: str, path: str, tool: str = "write") -> str:
    return "I will write the file.\n\n" + _call(tool, [(path_key, path), ("content", _FILE)])


def _client(monkeypatch, model_text: str) -> TestClient:
    state = _fake_state()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    monkeypatch.setattr(oa, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(oa, "_run_generation", _fake_streaming_generation(model_text))
    return TestClient(oa.create_app(state))


@pytest.mark.parametrize(
    "path_key, tool",
    [("path", "write"), ("filePath", "write")],  # Pi's write tool, OpenCode's
)
def test_chat_completions_stream_delivers_the_file_exactly(monkeypatch, path_key, tool):
    client = _client(monkeypatch, _model_writes(path_key, "hello.py", tool))
    schema = {"type": "object", "properties": {path_key: {"type": "string"}, "content": {"type": "string"}},
              "required": [path_key, "content"]}
    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Write hello.py."}],
            "tools": [{"type": "function", "function": {"name": tool, "parameters": schema}}],
            "stream": True,
            "max_tokens": 512,
        },
    )
    assert response.status_code == 200, response.text
    payloads = _stream_payloads(response.text)
    arguments = "".join(
        call.get("function", {}).get("arguments", "")
        for payload in payloads
        for call in payload["choices"][0]["delta"].get("tool_calls") or []
    )
    assert json.loads(arguments) == {path_key: "hello.py", "content": _FILE}
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_chat_completions_without_streaming_delivers_the_file_exactly(monkeypatch):
    client = _client(monkeypatch, _model_writes("path", "hello.py"))
    schema = {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}
    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Write hello.py."}],
            "tools": [{"type": "function", "function": {"name": "write", "parameters": schema}}],
            "max_tokens": 512,
        },
    )
    assert response.status_code == 200, response.text
    call = response.json()["choices"][0]["message"]["tool_calls"][0]
    assert _args(call) == {"path": "hello.py", "content": _FILE}


def test_anthropic_messages_stream_delivers_the_file_exactly(monkeypatch):
    # Claude Code's Write tool, over /v1/messages.
    client = _client(monkeypatch, _model_writes("file_path", "/tmp/hello.py", tool="Write"))
    response = client.post(
        "/v1/messages",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "model": "mtplx-test-model",
            "max_tokens": 512,
            "stream": True,
            "messages": [{"role": "user", "content": "Write hello.py."}],
            "tools": [{"name": "Write", "input_schema": {"type": "object", "properties": {
                "file_path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["file_path", "content"]}}],
        },
    )
    assert response.status_code == 200, response.text
    partial = "".join(
        data["delta"].get("partial_json", "")
        for event, data in _anthropic_events(response.text)
        if event == "content_block_delta" and data["delta"].get("type") == "input_json_delta"
    )
    assert json.loads(partial) == {"file_path": "/tmp/hello.py", "content": _FILE}


def test_responses_stream_delivers_the_file_exactly(monkeypatch):
    # Codex, over /v1/responses.
    client = _client(monkeypatch, _model_writes("path", "hello.py"))
    response = client.post(
        "/v1/responses",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "model": "mtplx-test-model",
            "stream": True,
            "input": [{"role": "user", "content": "Write hello.py."}],
            "tools": [{"type": "function", "name": "write", "parameters": {"type": "object", "properties": {
                "path": {"type": "string"}, "content": {"type": "string"}}}}],
        },
    )
    assert response.status_code == 200, response.text
    events = _stream_payloads(response.text)
    completed = [event for event in events if event.get("type") == "response.completed"]
    calls = [item for item in completed[-1]["response"]["output"] if item.get("type") == "function_call"]
    assert json.loads(calls[0]["arguments"]) == {"path": "hello.py", "content": _FILE}
