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


# The session bank holds an answer as the ids the model wrote. The client sends that turn back
# as parsed fields and the chat template renders them again (each argument through `tojson`,
# keys sorted), so the next prompt can leave the bank anywhere inside the answer even when it
# means exactly the same thing: the model wrote a run with a token split the encoder never makes
# (a 60-character `=` banner begun with ` =` where the encoder writes ` ` + `========`), escaped a
# character the template writes raw, or ordered keys differently. Everything after that point
# was read again: 65.5 s and 128.9 s to the first token after two big file writes in a Pi
# session. `splice_committed` compares the prompt with the bank one assistant turn at a time,
# framed by the control tokens neither side can write as text, and serves the bank's ids for
# every turn that says the same thing, so only the new tokens are read.

_HEADERS = ("message_user", "message_model", "message_system", "message_tool")
_OPENERS = ("content_text", "content_image", "content_thinking", "content_tool_error",
            "content_xml", "content_invoke_tool_json", "content_invoke_tool_text")
_MARKS = ("end_message", "content_model_end_sampling", "begin_of_text", "endoftext")
_VISIBLE = ("content_text", "content_invoke_tool_text", "content_tool_error")


class Dialect:
    """The control-token ids of Inkling's chat format for one tokenizer."""

    def __init__(self, tokenizer):
        ids = {name: tokenizer.convert_tokens_to_ids("<|" + name + "|>")
               for name in _HEADERS + _OPENERS + _MARKS}
        if not all(isinstance(value, int) for value in ids.values()) or len(set(ids.values())) != len(ids):
            raise ValueError("the tokenizer does not carry Inkling's control tokens")
        self.model = ids["message_model"]
        self.end = ids["end_message"]
        self.turn_end = ids["content_model_end_sampling"]
        self.thinking = ids["content_thinking"]
        self.tool_json = ids["content_invoke_tool_json"]
        self.visible = frozenset(ids[name] for name in _VISIBLE)
        self.openers = frozenset(ids[name] for name in _OPENERS)
        self.others = tuple(ids[name] for name in ("message_user", "message_system", "message_tool"))
        special = getattr(tokenizer, "all_special_ids", None) or ()
        self.control = frozenset(ids.values()).union(int(token) for token in special)


def dialect(tokenizer):
    """The tokenizer's `Dialect`, built once per tokenizer."""
    found = getattr(tokenizer, "_mtplx_inkling_dialect", None)
    if found is None:
        found = Dialect(tokenizer)
        try:
            tokenizer._mtplx_inkling_dialect = found
        except AttributeError:
            pass  # a tokenizer that takes no attributes builds it per call
    return found


def same_value(left, right):
    """Whether two parsed JSON values are the same value: types count (a string is never a
    number, `true` is never `1`), except that 1 and 1.0 are one JSON number."""
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(same_value(left[key], right[key]) for key in left))
    if isinstance(left, list):
        return (isinstance(right, list) and len(left) == len(right)
                and all(same_value(a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


def _unique_keys(pairs):
    found = {}
    for key, value in pairs:
        if key in found:
            raise ValueError(f"duplicate key {key!r}")
        found[key] = value
    return found


def _tool_call(text):
    """The `{"name": ..., "args": {...}}` payload of a tool-call segment, or None when the text
    is not one. A repeated key makes the meaning depend on the parser, so it is none."""
    try:
        value = json.loads(text, object_pairs_hook=_unique_keys)
    except ValueError:
        return None
    if isinstance(value, dict) and isinstance(value.get("name"), str) and isinstance(value.get("args"), dict):
        return value
    return None


def _text(tokenizer, ids):
    """A segment's text, or None when it cannot be compared as text: an image row, or a split
    multi-byte character."""
    if not ids:
        return ""
    from mtplx.vision.splice import is_image_key

    if is_image_key(max(ids)):
        return None
    text = tokenizer.decode(list(ids))
    if not isinstance(text, str) or "\ufffd" in text:
        return None
    return text


def _segments(stream, start, stop, words):
    """(opener, header ids, payload ids, closed) for each segment of the assistant turn
    `stream[start:stop]` (its closing marker excluded), or None when the span is not a run of
    model segments. Only the last one may be open: an answer cut at its length."""
    segments = []
    at = start
    while at < stop:
        if stream[at] != words.model:
            return None
        opener = at + 1
        while opener < stop and stream[opener] not in words.control:
            opener += 1
        if opener == stop or stream[opener] not in words.openers:
            return None
        try:
            close = stream.index(words.end, opener + 1, stop)
        except ValueError:
            close = stop
        payload = stream[opener + 1 : close]
        if not words.control.isdisjoint(payload):
            return None
        segments.append((stream[opener], stream[at + 1 : opener], payload, close < stop))
        at = close + 1 if close < stop else stop
    return segments


class _Turn:
    """What an assistant turn says: its reasoning, its visible text, its tool calls (name and
    parsed payload, in order) and any assistant name on its other segments."""

    __slots__ = ("reasoning", "visible", "calls", "names", "open_tail")

    def __init__(self, reasoning, visible, calls, names, open_tail):
        self.reasoning, self.visible, self.calls = reasoning, visible, calls
        self.names, self.open_tail = names, open_tail


def _read_turn(stream, start, stop, words, tokenizer):
    segments = _segments(stream, start, stop, words)
    if segments is None:
        return None
    reasoning, visible, calls, names = [], [], [], set()
    for opener, header, payload, closed in segments:
        name, text = _text(tokenizer, header), _text(tokenizer, payload)
        if name is None or text is None:
            return None
        if opener == words.tool_json:
            call = _tool_call(text) if closed else None
            if call is None:
                return None
            calls.append((name, call))
            continue
        if name:
            names.add(name)
        if opener == words.thinking:
            reasoning.append(text)
        elif opener in words.visible:
            visible.append(text)
        else:
            return None
    return _Turn("".join(reasoning), "".join(visible), calls, names,
                 bool(segments) and not segments[-1][3])


def _verdict(sent, banked, substitute_reasoning, visible_gate):
    """(why the bank's turn may not stand for the sent one, or None; whether it puts back text
    or reasoning the client did not send). Leading and trailing whitespace never count: the
    response strips it."""
    if len(sent.calls) != len(banked.calls) or not all(
        sent_name == banked_name and same_value(sent_call, banked_call)
        for (sent_name, sent_call), (banked_name, banked_call) in zip(sent.calls, banked.calls)
    ):
        return "tool_call_changed", False
    if sent.names != banked.names:
        return "turn_name_changed", False
    put_back = False
    if sent.visible.strip() != banked.visible.strip():
        gate = visible_gate if substitute_reasoning else None
        if gate is None or gate(sent.visible, bool(sent.calls)) != gate(banked.visible, bool(banked.calls)):
            return "text_changed", False
        put_back = True
    if sent.reasoning.strip() != banked.reasoning.strip():
        if not substitute_reasoning:
            return "reasoning_changed", False
        put_back = True
    return None, put_back


def _turn_start(stream, at, floor, words):
    """The first `<|message_model|>` of the assistant turn that holds `stream[at - 1]`, looking
    no further back than `floor`; None when that token is outside an assistant turn."""
    start = None
    for index in range(at - 1, floor - 1, -1):
        token = stream[index]
        if token == words.model:
            start = index
        elif token == words.turn_end or token in words.others:
            break
    return start


def _turn_end(stream, start, words, *, open_ok):
    """One past the turn's closing marker, or the stream's end for a turn the bank holds open
    when `open_ok`; None when another message starts inside the turn."""
    try:
        end = stream.index(words.turn_end, start) + 1
    except ValueError:
        if not open_ok:
            return None
        end = len(stream)
    for header in words.others:
        try:
            stream.index(header, start, end)
        except ValueError:
            continue
        return None
    return end


def splice_committed(prompt_ids, committed, tokenizer, *, substitute_reasoning=False, visible_gate=None):
    """Re-express `prompt_ids` with the bank's ids for every assistant turn that says the same
    thing, and return the ids and a receipt.

    The contract. Wherever the two streams differ, the assistant turn around that point is read
    from both sides. The bank's turn is served in place of the sent one when
      - every tool call has the same name and the same parsed payload (`same_value`), in order,
      - its visible text is the same (or, with `substitute_reasoning`, the same after
        `visible_gate`, the normalizer the client-echo gate uses), and
      - its reasoning is the same, or `substitute_reasoning` is set (preserve mode: the model
        gets back the reasoning the client dropped or changed).
    The first turn that fails, or a difference outside any assistant turn (the client changed
    a user, tool or system message), ends the walk: the rest of the prompt is served as sent and
    read again, so the model always sees what the client sent. The result is the bank's ids up
    to where the walk ended followed by the prompt's ids from the same point.
    """
    from mtplx.token_prefix import common_prefix_len, common_run_len

    prompt = [int(token) for token in prompt_ids]
    stream = committed if isinstance(committed, (list, tuple)) else tuple(committed)
    receipt = {"spans": 0, "tokens_in": 0, "tokens_out": 0}
    if not prompt or not stream:
        return prompt, receipt
    common = common_prefix_len(prompt, stream)
    if common >= len(prompt) or common >= len(stream):
        return prompt, receipt
    words = dialect(tokenizer)
    p = c = common
    floor = put_back = 0
    stopped = None
    while p < len(prompt) and c < len(stream):
        if prompt[p] == stream[c]:
            run = common_run_len(prompt, p, stream, c)
            p += run
            c += run
            continue
        start = _turn_start(stream, c, floor, words)
        if start is None:
            stopped = "outside_assistant_turn"
            break
        sent_start = p - (c - start)
        sent_end = _turn_end(prompt, sent_start, words, open_ok=False)
        banked_end = _turn_end(stream, start, words, open_ok=True)
        if sent_end is None or banked_end is None:
            stopped = "turn_shape"
            break
        closed = stream[banked_end - 1] == words.turn_end
        sent = _read_turn(prompt, sent_start, sent_end - 1, words, tokenizer)
        banked = _read_turn(stream, start, banked_end - 1 if closed else banked_end, words, tokenizer)
        if sent is None or banked is None:
            stopped = "turn_unreadable"
            break
        reason, restored = _verdict(sent, banked, substitute_reasoning, visible_gate)
        if reason is not None:
            stopped = reason
            break
        resume = sent_end
        if not closed:
            # The bank holds this answer open (cut at its length): the prompt goes on from its own
            # closing marker, or from the end of the segment the bank stops inside.
            resume, mark = (sent_end - 2, words.end) if banked.open_tail else (sent_end - 1, words.turn_end)
            if resume < sent_start or prompt[resume] != mark:
                stopped = "turn_shape"
                break
        receipt["spans"] += 1
        receipt["tokens_in"] += resume - sent_start
        receipt["tokens_out"] += banked_end - start
        put_back += int(restored)
        p, c, floor = resume, banked_end, banked_end
    if receipt["spans"]:
        receipt["first_divergence"] = int(common)
        receipt["cp_after"] = int(c)
    if put_back:
        receipt["put_back_turns"] = put_back
    if stopped is not None:
        receipt["stopped"] = stopped
    return list(stream[:c]) + prompt[p:], receipt
