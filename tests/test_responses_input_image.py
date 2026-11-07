"""Responses input_image parts reach the chat vision path.

User messages and function/custom tool outputs may carry ``input_image``
parts (screenshots from computer-use tools, for example). They translate to
Chat ``image_url`` parts; text-only content still translates to the same
plain string as before.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from mtplx.server import responses as r

DATA = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO+aL1sAAAAASUVORK5CYII="
)
IMG = {"type": "input_image", "image_url": DATA}
TXT = {"type": "input_text", "text": "Grün\n"}


def convert(items, **kw):
    return r.translate_request(r.ResponsesRequest(input=items, **kw))


def tool_items(kind="function", output=None):
    if kind == "custom":
        call = {"type": "custom_tool_call", "call_id": "exact-id", "name": "screen", "input": "capture"}
        out = "custom_tool_call_output"
    else:
        call = {"type": "function_call", "call_id": "exact-id", "name": "screen", "arguments": "{}"}
        out = "function_call_output"
    return [call, {"type": out, "call_id": "exact-id", "output": output}]


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Text\n ä", "Text\n ä"),
        (None, ""),
        ([], ""),
        ([TXT], "Grün\n"),
        ([TXT, {"type": "output_text", "text": "\n尾"}], "Grün\n\n尾"),
    ],
)
def test_text_only_content_is_still_a_plain_string(value, expected):
    assert convert([{"role": "user", "content": value}]).chat["messages"][0]["content"] == expected
    for kind in ("function", "custom"):
        assert convert(tool_items(kind, value)).chat["messages"][1]["content"] == expected


@pytest.mark.parametrize("url", [DATA, "https://example.org/a.png?b=2", "http://example.org:8080/a"])
def test_user_image_becomes_image_url(url):
    content = [{"type": "input_image", "image_url": url, "detail": "high"}]
    message = convert([{"role": "user", "content": content}]).chat["messages"][0]
    assert message["content"] == [{"type": "image_url", "image_url": {"url": url, "detail": "high"}}]


def test_order_is_kept_and_adjacent_text_coalesced():
    content = [TXT, TXT, IMG, TXT, IMG]
    parts = convert([{"role": "user", "content": content}]).chat["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url", "text", "image_url"]
    assert parts[0]["text"] == TXT["text"] * 2
    assert parts[2]["text"] == TXT["text"]
    assert content == [TXT, TXT, IMG, TXT, IMG]


@pytest.mark.parametrize("kind", ["function", "custom"])
def test_tool_output_image(kind):
    messages = convert(tool_items(kind, [TXT, IMG])).chat["messages"]
    assert messages[0]["tool_calls"][0]["id"] == "exact-id"
    assert messages[1]["role"] == "tool"
    assert messages[1]["tool_call_id"] == "exact-id"
    assert messages[1]["content"][1]["image_url"]["url"] == DATA


def test_namespace_tool_output_image():
    tools = [{"type": "namespace", "name": "computer", "description": "Desktop",
              "tools": [{"type": "function", "name": "screen", "parameters": {"type": "object"}}]}]
    items = tool_items(output=[IMG])
    items[0]["namespace"] = "computer"
    result = convert(items, tools=tools)
    flat = result.chat["messages"][0]["tool_calls"][0]["function"]["name"]
    assert result.namespace_functions[flat].namespace == "computer"
    assert result.namespace_functions[flat].name == "screen"


BAD_PARTS = [
    {"type": "input_image"},
    {"type": "input_image", "file_id": "file-123"},
    {**IMG, "file_id": "file-123"},
    {**IMG, "image_url": {"url": DATA}},
    {**IMG, "image_url": ""},
    {**IMG, "image_url": "file:///tmp/p.png"},
    {**IMG, "image_url": "https:///no-host"},
    {**IMG, "image_url": "http://host:bad/p"},
    {**IMG, "image_url": "https://u:p@host/p"},
    {**IMG, "image_url": "https://host/a b"},
    {**IMG, "image_url": "data:text/plain;base64,YQ=="},
    {**IMG, "image_url": "data:image/png;base64,?invalid"},
    {**IMG, "image_url": "data:image/png;base64,"},
    {**IMG, "detail": []},
    {**IMG, "detail": "invalid"},
    {**IMG, "extra": True},
    {"type": "input_audio", "data": "xxx"},
    {"type": "input_file", "file_id": "file-123"},
    {"type": "input_text", "text": 7},
    5,
]


@pytest.mark.parametrize("part", BAD_PARTS, ids=range(len(BAD_PARTS)))
def test_malformed_parts_are_refused(part):
    for items in ([{"role": "user", "content": [part]}], tool_items(output=[part])):
        with pytest.raises(r.ResponsesProtocolError):
            convert(items)


@pytest.mark.parametrize("role", ["assistant", "system", "developer"])
def test_images_stay_refused_outside_user_and_tool_output(role):
    with pytest.raises(r.ResponsesProtocolError):
        convert([{"role": role, "content": [IMG]}])


def test_no_relaxation_for_orphan_output_or_custom_call_input():
    with pytest.raises(r.ResponsesProtocolError):
        convert([{"type": "function_call_output", "call_id": "missing", "output": [IMG]}])
    items = tool_items("custom", "okay")
    items[0]["input"] = [IMG]
    with pytest.raises(r.ResponsesProtocolError):
        convert(items)


def test_streamed_answer_after_tool_image():
    async def run():
        async def chunks():
            for obj in [
                {"choices": [{"delta": {"content": "Green"}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}},
            ]:
                yield ("data: " + json.dumps(obj) + "\n\n").encode()
            yield b"data: [DONE]\n\n"

        conversion = convert(tool_items(output=[IMG]), stream=True)
        frames = [
            frame
            async for frame in r.stream_from_chat_sse(
                chunks(), response_id="resp_test", model="test",
                response_fields=conversion.response_fields,
                custom_tool_names=conversion.custom_tool_names,
                namespace_functions=conversion.namespace_functions,
            )
        ]
        events = [json.loads(f.split("data: ", 1)[1]) for f in frames if "data: " in f]
        final = events[-1]["response"]
        assert events[-1]["type"] == "response.completed"
        assert final["output"][0]["content"][0]["text"] == "Green"
        assert final["usage"]["total_tokens"] == 11

    asyncio.run(run())
