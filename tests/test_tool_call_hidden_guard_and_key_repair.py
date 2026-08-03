"""Issues #196/#197: hidden-tool-guard misfire on JSON-dialect bodies, and
near-miss argument-key corruption silently dropping arguments.

#196 (hard-error layer): the reporter's exact client error —
``malformed tool_call: unterminated stream`` — is produced by MTPLX's own
stream hidden-tool guard (openai.py, STREAM_HIDDEN_TOOL_GUARD_*). The guard
stands down while the parser is inside a ``<parameter=>`` value
(``tool_argument_in_progress``), but the #170 JSON-dialect body
(``<function=write>{"filePath": ..., "content": "..."}``) waits in the
``find_parameter`` stage, so the stand-down never engaged. A large write body
(>= 2048 hidden tokens and >= 30 s — guaranteed for multi-KB code payloads on
slower hardware) crossed the guard budget and generation was cancelled
mid-call with the 422. This is exactly the reported shape: small prose writes
succeed, large code/markup writes fail. Contract after the fix: a JSON-dialect
function body for a known tool is argument payload, and the guard stands down
over it exactly as it does for ``<parameter=>`` values.

#197 (silent-drop layer): corrupted argument keys (``offsets`` for ``offset``,
``offset `` with trailing whitespace, ``offset >`` with the template's
tag-close byte bled in) pass schema validation whenever the real parameter is
optional (OpenCode's ``read``), so the client silently dropped the argument
and every paginated read returned the same top-of-file window. Contract after
the fix: an unambiguous near-miss key is repaired to the schema property
(trim / case / single trailing "s"); anything ambiguous or already-supplied
passes through verbatim.
"""

import json

from mtplx.server.openai import (
    _ToolAwareContentStreamTranslator,
    _parse_generated_tool_calls,
    _repair_tool_argument_keys_for_schema,
)


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

# OpenCode's real read tool shape: offset/limit are OPTIONAL, so schema
# validation cannot catch a corrupted key — the silent-drop lane of #197.
OPENCODE_READ_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["filePath"],
            },
        },
    }
]

# The issue-197 curl repro shape: offset IS required. Before the repair the
# corrupted key failed required-validation and the whole call fell back.
READ_ALL_REQUIRED_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["filePath", "offset", "limit"],
            },
        },
    }
]

LONG_HTML = "\n".join(
    [
        "<!DOCTYPE html>",
        "<html>",
        "  <head>",
        '    <style>body { font-family: "Menlo", monospace; }</style>',
        "  </head>",
        "  <body>",
        "    <script>",
        '      const config = { "mode": "dark", "depth": 3 };',
        "    </script>",
        "  </body>",
        "</html>",
    ]
    * 24
)


def _make(tools):
    return _ToolAwareContentStreamTranslator(
        tools=tools,
        argument_chunk_chars=64,
        tokenizer=None,
    )


def _argument_text(deltas):
    return "".join(
        item.get("function", {}).get("arguments", "")
        for delta in deltas
        for item in delta.get("tool_calls", [])
    )


def _content_text(deltas):
    return "".join(delta.get("content", "") for delta in deltas)


def _feed_in_chunks(translator, text, size):
    deltas = []
    for start in range(0, len(text), size):
        deltas.extend(translator.feed("content", text[start : start + size]))
    return deltas


# ---------- #196: hidden-tool-guard stand-down over JSON-dialect bodies ----------


def test_json_body_write_marks_argument_in_progress_while_streaming():
    """The guard's stand-down signal must hold across a streaming JSON body."""
    t = _make(OPENCODE_WRITE_TOOL_SPECS)
    payload = {"filePath": "index.html", "content": LONG_HTML}
    body = json.dumps(payload, ensure_ascii=False)

    out = []
    out.extend(t.feed("content", "<tool_call>\n<function=write>\n"))
    # Before any body byte arrives there is nothing hidden yet.
    assert t.tool_argument_in_progress is False

    for start in range(0, len(body), 53):
        out.extend(t.feed("content", body[start : start + 53]))
        # Mid-body: this is the exact predicate the stream hidden-tool guard
        # checks. False here is what cancelled large writes with
        # "malformed tool_call: unterminated stream".
        assert t.tool_argument_in_progress is True, (
            f"guard stand-down dropped at body offset {start}"
        )

    out.extend(t.feed("content", "\n</function>\n</tool_call>"))
    out.extend(t.finish())

    assert t.has_tool_calls is True, t.fallback_reason
    assert json.loads(_argument_text(out)) == payload
    content = _content_text(out)
    assert "<tool_call" not in content
    assert "<function=" not in content


def test_json_body_unknown_tool_keeps_guard_armed():
    """Hidden runaways on tools absent from the request stay guarded."""
    t = _make(OPENCODE_READ_TOOL_SPECS)
    t.feed("content", "<tool_call>\n<function=deploy>\n")
    t.feed("content", '{"target": "prod", "notes": "')
    assert t.tool_argument_in_progress is False


def test_xml_parameter_stand_down_unchanged():
    """The original <parameter=> stand-down contract is untouched."""
    t = _make(OPENCODE_WRITE_TOOL_SPECS)
    t.feed("content", "<tool_call>\n<function=write>\n<parameter=content>\n")
    t.feed("content", "line one\nline two\n")
    assert t.tool_argument_in_progress is True


# ---------- #197: near-miss argument-key repair ----------


def test_streaming_xml_offsets_plural_key_repaired_for_optional_param():
    """The reporter's exact symptom: offsets(plural) silently dropped."""
    t = _make(OPENCODE_READ_TOOL_SPECS)
    text = (
        "<tool_call>\n<function=read>\n"
        "<parameter=filePath>\nplan/architecture.md\n</parameter>\n"
        "<parameter=limit>\n40\n</parameter>\n"
        "<parameter=offsets>\n45\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    out = _feed_in_chunks(t, text, 7)
    out.extend(t.finish())

    assert t.has_tool_calls is True, t.fallback_reason
    args = json.loads(_argument_text(out))
    assert args == {"filePath": "plan/architecture.md", "limit": 40, "offset": 45}
    assert "offsets" not in args


def test_streaming_xml_offsets_key_repaired_for_required_param():
    """Issue-197 curl shape: with offset required, the corrupted key used to
    fail required-validation and swallow the whole call."""
    t = _make(READ_ALL_REQUIRED_TOOL_SPECS)
    text = (
        "<tool_call>\n<function=read>\n"
        "<parameter=filePath>\nplan/architecture.md\n</parameter>\n"
        "<parameter=offsets>\n45\n</parameter>\n"
        "<parameter=limit>\n40\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    out = _feed_in_chunks(t, text, 11)
    out.extend(t.finish())

    assert t.has_tool_calls is True, t.fallback_reason
    args = json.loads(_argument_text(out))
    assert args["offset"] == 45


def test_json_body_corrupted_keys_repaired_all_reporter_shapes():
    """All three corruption shapes from #197, through the JSON-dialect body
    (json.loads preserves corrupted keys verbatim, so they reach the
    normalizer untouched)."""
    for corrupted in ("offsets", "offset ", "offset >"):
        t = _make(OPENCODE_READ_TOOL_SPECS)
        body = json.dumps(
            {"filePath": "plan/architecture.md", "limit": 40, corrupted: 45},
            ensure_ascii=False,
        )
        text = f"<tool_call>\n<function=read>\n{body}\n</function>\n</tool_call>"
        out = _feed_in_chunks(t, text, 13)
        out.extend(t.finish())

        assert t.has_tool_calls is True, (corrupted, t.fallback_reason)
        args = json.loads(_argument_text(out))
        assert args == {
            "filePath": "plan/architecture.md",
            "limit": 40,
            "offset": 45,
        }, f"key {corrupted!r} was not repaired: {args!r}"


def test_final_parser_offsets_key_repaired():
    """Non-stream parity: the final parser repairs the same key."""
    text = (
        "<tool_call>\n<function=read>\n"
        "<parameter=filePath>\nplan/architecture.md\n</parameter>\n"
        "<parameter=offsets>\n45\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    calls = _parse_generated_tool_calls(text, tools=OPENCODE_READ_TOOL_SPECS)
    assert calls is not None and len(calls) == 1
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"filePath": "plan/architecture.md", "offset": 45}


def test_key_repair_is_conservative():
    """Repair only fires on unambiguous mappings."""
    tools = OPENCODE_READ_TOOL_SPECS

    # Target already supplied by the model: never clobber it.
    args = {"offset": 1, "offsets": 2}
    assert _repair_tool_argument_keys_for_schema(
        tool_name="read", arguments=args, tools=tools
    ) == {"offset": 1, "offsets": 2}

    # No near-miss match: unknown keys pass through (client owns rejection).
    args = {"filePath": "a.md", "randomkey": 1}
    assert _repair_tool_argument_keys_for_schema(
        tool_name="read", arguments=args, tools=tools
    ) == {"filePath": "a.md", "randomkey": 1}

    # Two corrupted keys collapsing onto one property: repair neither.
    args = {"offsets": 1, "Offset": 2}
    assert _repair_tool_argument_keys_for_schema(
        tool_name="read", arguments=args, tools=tools
    ) == {"offsets": 1, "Offset": 2}

    # A schema that legitimately has both singular and plural: untouched.
    both_tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "offset": {"type": "integer"},
                        "offsets": {"type": "array"},
                    },
                },
            },
        }
    ]
    args = {"offsets": [1, 2]}
    assert _repair_tool_argument_keys_for_schema(
        tool_name="read", arguments=args, tools=both_tools
    ) == {"offsets": [1, 2]}
