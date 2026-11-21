"""The publisher's typed message records mapped to the serving layer's content channels."""

import json
import re

CONTROL = re.compile(
    r"<\|(?:message_\w+|content_\w+|end_message|unused_200054|begin_of_text)\|>"
)


def literal(text):
    return CONTROL.sub(lambda match: match.group().replace("<|", "< |", 1), text)


def prepare_message(message, images, fetch_image):
    item = (
        dict(message)
        if isinstance(message, dict)
        else message.model_dump(exclude_none=True)
    )
    content = item.get("content")
    if isinstance(content, str):
        item["content"] = literal(content)
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append({"type": "text", "text": literal(part)})
            elif part.get("type") in ("image_url", "input_image"):
                locator = part.get("image_url")
                url = locator.get("url") if isinstance(locator, dict) else locator
                images.append(fetch_image(str(url or "")))
                parts.append({"type": "image"})
            elif part.get("type") in ("text", "input_text"):
                parts.append(
                    {"type": "text", "text": literal(str(part.get("text", "")))}
                )
            else:
                raise ValueError(
                    f"Unsupported Inkling content part: {part.get('type')}"
                )
        item["content"] = parts
    if isinstance(message, dict):
        return item
    return message.model_copy(update={"content": item.get("content")})


def encode(
    tokenizer,
    messages,
    *,
    tools,
    effort,
    thinking,
    strip_reasoning,
    add_generation_prompt,
):
    normalized = []
    for message in messages:
        item = (
            dict(message)
            if isinstance(message, dict)
            else message.model_dump(exclude_none=True)
        )
        if item.get("role") == "developer":
            item["role"] = "system"
        if strip_reasoning:
            item.pop("reasoning_content", None)
        calls = []
        for call in item.get("tool_calls") or []:
            function = dict(call["function"])
            args = function.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args or "{}")
            if not isinstance(args, dict):
                raise ValueError("Inkling tool arguments must be a JSON object")
            calls.append({**call, "function": {**function, "arguments": args}})
        if calls:
            item["tool_calls"] = calls
        normalized.append(item)
    rendered = tokenizer.apply_chat_template(
        normalized,
        tools=tools,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        reasoning_effort=(effort or "high") if thinking else "none",
    )
    return tokenizer.encode(rendered, add_special_tokens=False)


class Channels:
    """Streaming state machine. Text and reasoning stream immediately, tool JSON commits at its delimiter."""

    def __init__(self, tokenizer):
        self.ids = {
            int(tokenizer.convert_tokens_to_ids("<|" + name + "|>")): name
            for name in (
                "message_model",
                "content_text",
                "content_thinking",
                "content_invoke_tool_json",
                "content_invoke_tool_text",
                "end_message",
                "content_model_end_sampling",
            )
        }
        self.mode = "text"
        self.payload = []

    def _content(self, text):
        if self.mode in ("header", "tool"):
            self.payload.append(text)
            return ""
        return text

    def _close(self):
        if self.mode == "reasoning":
            return "</think>"
        if self.mode == "tool" and self.payload:
            body = "".join(self.payload)
            try:
                call = json.loads(body)
            except json.JSONDecodeError:
                return body
            if (
                isinstance(call, dict)
                and isinstance(call.get("name"), str)
                and isinstance(call.get("args"), dict)
            ):
                return (
                    "<tool_call>"
                    + json.dumps(
                        {"name": call["name"], "arguments": call["args"]},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "</tool_call>"
                )
            return body
        return ""

    def feed(self, tokens, decode, flush):
        pieces, text_ids = [], []
        for token in tokens:
            kind = self.ids.get(int(token))
            if kind is None:
                text_ids.append(token)
                continue
            pieces.append(self._content(decode(text_ids) + flush()))
            text_ids.clear()
            pieces.append(self._close())
            self.payload = []
            self.mode = {
                "message_model": "header",
                "content_thinking": "reasoning",
                "content_invoke_tool_json": "tool",
                "content_invoke_tool_text": "text",
            }.get(kind, "text")
            if self.mode == "reasoning":
                pieces.append("<think>")
        pieces.append(self._content(decode(text_ids)))
        return "".join(pieces)

    def finish(self, flush):
        result = self._content(flush()) + self._close()
        self.mode, self.payload = "text", []
        return result


def completion_text(tokenizer, tokens):
    channels = Channels(tokenizer)
    return channels.feed(
        tokens, lambda ids: tokenizer.decode(ids) if ids else "", lambda: ""
    ) + channels.finish(lambda: "")
