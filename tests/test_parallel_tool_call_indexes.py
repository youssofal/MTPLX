"""Issue #709: parallel tool calls keep their own index, id and JSON object.

Inkling 2 Mini's channels commit each call's JSON at its own delimiter, so two
calls reach the stream translator as two ``<tool_call>`` blocks in separate
chunks. Each block was parsed on its own and numbered from 0, so both calls
went out as ``index`` 0: an OpenAI client assembles them into one call whose
arguments are two JSON objects back to back, and the Anthropic stream
translator (which keys tool blocks by that index) into one ``tool_use`` block
with the same concatenated input. These tests cover both endpoints, streaming
and not, the same tool twice, two tools, three calls, and calls split across
chunk boundaries, plus the other tool dialects that parse one block at a time.
"""

import asyncio
import json

import pytest

from mtplx.server.openai import (
    _anthropic_stream_from_openai_sse,
    _anthropic_tool_use_block,
    _parse_generated_tool_calls,
    _ToolAwareContentStreamTranslator,
)

READ = {
    "type": "function",
    "function": {
        "name": "Read",
        "parameters": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    },
}
LIST = {
    "type": "function",
    "function": {
        "name": "List",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}

CASES = {
    "same tool twice": (
        [READ],
        [("Read", {"file_path": "/tmp/a.txt"}), ("Read", {"file_path": "/tmp/b.txt"})],
    ),
    "two different tools": (
        [READ, LIST],
        [("Read", {"file_path": "/tmp/a.txt"}), ("List", {"path": "/tmp"})],
    ),
    "three calls": (
        [READ],
        [("Read", {"file_path": f"/tmp/{c}.txt"}) for c in "abc"],
    ),
}


def _json_block(name, arguments):
    """What Inkling's channels hand over for one call."""
    body = json.dumps({"name": name, "arguments": arguments}, separators=(",", ":"))
    return f"<tool_call>{body}</tool_call>"


def _chunkings(blocks):
    joined = "".join(blocks)
    yield "one chunk", [joined]
    yield "one chunk a block (as the channels commit them)", list(blocks)
    yield "three-character chunks", [joined[i : i + 3] for i in range(0, len(joined), 3)]
    cut = len(blocks[0]) + len(blocks[1]) // 2
    yield "split inside the second call's JSON", [joined[:cut], joined[cut:]]


def _translate(tools, chunks):
    translator = _ToolAwareContentStreamTranslator(tools=tools, argument_chunk_chars=8)
    deltas = []
    for chunk in chunks:
        deltas.extend(translator.feed("content", chunk))
    deltas.extend(translator.finish())
    return translator, deltas


def _assemble_like_an_openai_client(deltas):
    calls = {}
    for delta in deltas:
        for item in delta.get("tool_calls") or []:
            call = calls.setdefault(item["index"], {"id": None, "name": "", "arguments": ""})
            if item.get("id"):
                assert call["id"] is None, "a second id at one index: two calls merged"
                call["id"] = item["id"]
            function = item.get("function") or {}
            call["name"] += function.get("name") or ""
            call["arguments"] += function.get("arguments") or ""
    return [calls[index] for index in sorted(calls)]


def _openai_sse(deltas):
    chunks = [
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
        + "\n\n"
        for delta in deltas
    ]
    chunks.append(
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
        + "\n\n"
    )
    chunks.append("data: [DONE]\n\n")
    return chunks


def _anthropic_events(sse_chunks):
    async def upstream():
        for chunk in sse_chunks:
            yield chunk

    async def collect():
        return [chunk async for chunk in _anthropic_stream_from_openai_sse(upstream(), model="m")]

    text = "".join(c.decode() if isinstance(c, bytes) else c for c in asyncio.run(collect()))
    events = []
    for block in text.split("\n\n"):
        data = [line[5:].strip() for line in block.splitlines() if line.startswith("data:")]
        if data:
            events.append(json.loads("\n".join(data)))
    return events


@pytest.mark.parametrize("case", list(CASES))
def test_openai_stream_gives_each_call_its_own_index(case):
    tools, expected = CASES[case]
    for how, chunks in _chunkings([_json_block(n, a) for n, a in expected]):
        _, deltas = _translate(tools, chunks)
        calls = _assemble_like_an_openai_client(deltas)
        assert [(c["name"], json.loads(c["arguments"])) for c in calls] == expected, how
        assert len({c["id"] for c in calls}) == len(expected), how


@pytest.mark.parametrize("case", list(CASES))
def test_anthropic_stream_gives_each_call_its_own_tool_use_block(case):
    tools, expected = CASES[case]
    for how, chunks in _chunkings([_json_block(n, a) for n, a in expected]):
        _, deltas = _translate(tools, chunks)
        events = _anthropic_events(_openai_sse(deltas))
        blocks = {}
        for event in events:
            if event.get("type") == "content_block_start" and event["content_block"].get("type") == "tool_use":
                blocks[event["index"]] = {
                    "id": event["content_block"]["id"],
                    "name": event["content_block"]["name"],
                    "json": "",
                }
            elif event.get("type") == "content_block_delta" and event["delta"].get("type") == "input_json_delta":
                blocks[event["index"]]["json"] += event["delta"]["partial_json"]
        got = [blocks[index] for index in sorted(blocks)]
        assert [(b["name"], json.loads(b["json"])) for b in got] == expected, how
        assert len({b["id"] for b in got}) == len(expected), how
        stop = [e for e in events if e.get("type") == "message_delta"][-1]["delta"]["stop_reason"]
        assert stop == "tool_use", how


@pytest.mark.parametrize("case", list(CASES))
def test_non_streaming_answer_keeps_one_call_per_block_on_both_endpoints(case):
    tools, expected = CASES[case]
    calls = _parse_generated_tool_calls("".join(_json_block(n, a) for n, a in expected), tools=tools)
    assert [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls] == expected
    assert len({c["id"] for c in calls}) == len(expected)
    blocks = [_anthropic_tool_use_block(call) for call in calls]
    assert [(b["name"], b["input"]) for b in blocks] == expected
    assert len({b["id"] for b in blocks}) == len(expected)


def _xml_block(name, arguments):
    params = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in arguments.items())
    return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


def _suffixed_block(name, arguments):
    pairs = "".join(
        f"<arg_key:opensource>{k}</arg_key:opensource>\n<arg_value:opensource>{v}</arg_value:opensource>\n"
        for k, v in arguments.items()
    )
    return (
        "<tool_calls:opensource>\n"
        f"<tool_call:opensource>{name}<tool_sep:opensource>\n{pairs}</tool_call:opensource>\n"
        "</tool_calls:opensource>"
    )


@pytest.mark.parametrize("dialect, make_block", [("qwen_xml", _xml_block), ("suffixed_native", _suffixed_block)])
def test_other_dialects_number_consecutive_blocks_across_the_response(dialect, make_block):
    tools, expected = CASES["three calls"]
    blocks = [make_block(n, a) for n, a in expected]
    for how, chunks in (("one chunk a block", blocks), ("three-character chunks", None)):
        if chunks is None:
            joined = "\n".join(blocks)
            chunks = [joined[i : i + 3] for i in range(0, len(joined), 3)]
        _, deltas = _translate(tools, chunks)
        calls = _assemble_like_an_openai_client(deltas)
        assert [(c["name"], json.loads(c["arguments"])) for c in calls] == expected, (dialect, how)
        assert len({c["id"] for c in calls}) == len(expected), (dialect, how)
