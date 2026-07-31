# SPDX-License-Identifier: Apache-2.0
"""Tool-call parsing and stream filtering adapted from oMLX.

Source inspiration: oMLX ``omlx/api/tool_calling.py`` on origin/main,
Apache-2.0. MTPLX intentionally uses this module as a protocol adapter, not as
a model babysitter: visible text is preserved, raw control markup is filtered,
and valid tool calls are parsed at completion.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCallExtraction:
    cleaned_text: str
    tool_calls: list[dict[str, Any]] | None
    cleaned_thinking: str
    parser_source: str = "none"
    status: str = "no_tool"
    malformed_reason: str | None = None
    raw_tool_markup_suppressed: bool = False


def _serialize_tool_call_arguments(arguments: Any) -> str:
    if isinstance(arguments, dict):
        return json.dumps(
            _order_tool_arguments_for_client_display("", arguments),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    return "{}"


_TOOL_ARGUMENT_PRIMARY_ORDER: dict[str, tuple[str, ...]] = {
    "read": ("filePath", "path", "offset", "limit"),
    "grep": ("pattern", "path", "include", "limit"),
    "glob": ("pattern", "path"),
    "bash": ("command", "description", "timeout"),
    "write": ("filePath", "path", "content"),
    "edit": ("filePath", "path", "oldString", "newString", "replaceAll"),
    "webfetch": ("url", "format", "timeout"),
    "skill": ("name",),
    "question": ("questions",),
}


def _order_tool_arguments_for_client_display(
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Return stable argument order so clients do not lead with display noise.

    OpenCode renders tool argument values in JSON insertion order. Qwen can emit
    optional knobs such as ``limit`` before the useful target path, producing
    transcript rows that start with a bare ``100``. This preserves every
    argument and only moves the user-facing identifiers first.
    """

    if not arguments:
        return arguments
    order = _TOOL_ARGUMENT_PRIMARY_ORDER.get(str(tool_name or "").strip().lower())
    if not order:
        return arguments
    ordered: dict[str, Any] = {}
    for key in order:
        if key in arguments:
            ordered[key] = arguments[key]
    for key, value in arguments.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _tool_call(name: str, arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        arguments = _order_tool_arguments_for_client_display(str(name), arguments)
    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": str(name),
            "arguments": _serialize_tool_call_arguments(arguments),
        },
    }


def _coerce_json_tool_payload(parsed: Any) -> dict[str, Any] | None:
    if isinstance(parsed, list):
        for item in parsed:
            coerced = _coerce_json_tool_payload(item)
            if coerced is not None:
                return coerced
        return None
    if not isinstance(parsed, dict):
        return None
    function = parsed.get("function")
    if isinstance(function, dict):
        name = function.get("name") or function.get("tool") or function.get("function")
        arguments = (
            function.get("arguments")
            if "arguments" in function
            else function.get("args", function.get("parameters", {}))
        )
        if name:
            return _tool_call(str(name), arguments)
    name = (
        parsed.get("name")
        or parsed.get("tool")
        or parsed.get("function")
        or parsed.get("call")
    )
    if not name:
        return None
    arguments = parsed.get("arguments", parsed.get("args", parsed.get("parameters", {})))
    return _tool_call(str(name), arguments)


def _parse_json_tool_payload(content: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        return None
    return _coerce_json_tool_payload(parsed)


def _parse_bare_tool_payload(content: str) -> dict[str, Any] | None:
    stripped = content.strip()
    match = re.match(
        r"^([A-Za-z_][\w.-]*)\s*(?:\((.*)\)|:\s*(\{.*\})|(\{.*\}))\s*$",
        stripped,
        re.DOTALL,
    )
    if not match:
        return None
    name = match.group(1)
    raw_args = next((group for group in match.groups()[1:] if group), "{}")
    try:
        arguments = json.loads(raw_args)
    except (TypeError, ValueError):
        arguments = {"_raw": raw_args}
    return _tool_call(name, arguments)


def _parse_xml_tool_calls(text: str) -> tuple[str, list[dict[str, Any]] | None, str | None]:
    calls: list[dict[str, Any]] = []
    malformed_reason: str | None = None
    for match in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        content = match.strip()
        if parsed := _parse_json_tool_payload(content):
            calls.append(parsed)
            continue
        if parsed := _parse_bare_tool_payload(content):
            calls.append(parsed)
            continue
        func_match = re.match(
            r"<function=([^>\s]+)>\s*(.*?)\s*</function>",
            content,
            re.DOTALL,
        )
        if func_match is None:
            func_match = re.match(
                r'<function\s+name="([^"]+)">\s*(.*?)\s*</function>',
                content,
                re.DOTALL,
            )
        if func_match:
            name = func_match.group(1)
            params = {}
            param_patterns = (
                r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>",
                r'<parameter\s+name="([^"]+)">\s*(.*?)\s*</parameter>',
            )
            for pattern in param_patterns:
                for param in re.finditer(pattern, func_match.group(2), re.DOTALL):
                    key = param.group(1)
                    value = param.group(2).strip()
                    try:
                        params[key] = json.loads(value)
                    except (TypeError, ValueError):
                        params[key] = value
            if not params:
                body = func_match.group(2).strip()
                if body:
                    # #170: a pure JSON-object body inside the envelope is the
                    # arguments payload (same contract as the strict parsers).
                    parsed_body = None
                    if body.startswith("{"):
                        try:
                            parsed_body = json.loads(body)
                        except (TypeError, ValueError):
                            parsed_body = None
                    if isinstance(parsed_body, dict):
                        params = parsed_body
                    else:
                        # Never fabricate a {}-arguments call out of a body the
                        # parser could not read — that silent empty call is the
                        # measured #170 client shape. The turn stays visible
                        # content instead.
                        malformed_reason = (
                            f"tool '{name}' contains unwrapped parameter text"
                        )
                        calls = []
                        break
            calls.append(_tool_call(name, params))
            continue
        invoke_match = re.match(
            r'<invoke\s+name="([^"]+)">\s*(.*?)\s*</invoke>',
            content,
            re.DOTALL,
        )
        if invoke_match:
            name = invoke_match.group(1)
            params = {}
            for param in re.finditer(
                r'<parameter\s+name="([^"]+)">\s*(.*?)\s*</parameter>',
                invoke_match.group(2),
                re.DOTALL,
            ):
                key = param.group(1)
                value = param.group(2).strip()
                try:
                    params[key] = json.loads(value)
                except (TypeError, ValueError):
                    params[key] = value
            if not params and invoke_match.group(2).strip():
                malformed_reason = (
                    f"tool '{name}' contains unwrapped parameter text"
                )
                calls = []
                break
            calls.append(_tool_call(name, params))
            continue
        # Poolside dialect: `name<arg_key>k</arg_key><arg_value>v</arg_value>...`
        # (the Laguna family's native emission; same contract as the strict
        # parser in openai.py — every non-pair byte after the name is residue
        # and marks the call malformed rather than silently dropping args).
        poolside_match = re.match(r"\s*([A-Za-z_][\w.-]*)", content)
        if poolside_match:
            poolside_name = poolside_match.group(1)
            body = content[poolside_match.end():]
            pairs = list(
                re.finditer(
                    r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>",
                    body,
                    re.DOTALL,
                )
            )
            if pairs:
                params = {}
                residue_parts = []
                cursor = 0
                valid = True
                for pair in pairs:
                    key = pair.group(1).strip()
                    if not key or key in params:
                        valid = False
                        break
                    value = pair.group(2).strip()
                    try:
                        params[key] = json.loads(value)
                    except (TypeError, ValueError):
                        params[key] = value
                    residue_parts.append(body[cursor:pair.start()])
                    cursor = pair.end()
                residue_parts.append(body[cursor:])
                if valid and not "".join(residue_parts).strip():
                    calls.append(_tool_call(poolside_name, params))
                    continue
                malformed_reason = (
                    f"tool '{poolside_name}' contains malformed Poolside arguments"
                )
                calls = []
                break
        malformed_reason = "unrecognized <tool_call> payload"
    if not calls:
        return text, None, malformed_reason
    cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()
    return cleaned, calls, None


def _parse_namespaced_tool_calls(text: str) -> tuple[str, list[dict[str, Any]] | None]:
    calls: list[dict[str, Any]] = []
    pattern = r"<([A-Za-z_][\w.-]*):tool_call>\s*(.*?)\s*</\1:tool_call>"
    for _namespace, content in re.findall(pattern, text, re.DOTALL):
        for invoke in re.finditer(
            r'<invoke\s+name="([^"]+)">\s*(.*?)\s*</invoke>',
            content,
            re.DOTALL,
        ):
            params = {}
            for param in re.finditer(
                r'<parameter\s+name="([^"]+)">\s*(.*?)\s*</parameter>',
                invoke.group(2),
                re.DOTALL,
            ):
                value = param.group(2).strip()
                try:
                    params[param.group(1)] = json.loads(value)
                except (TypeError, ValueError):
                    params[param.group(1)] = value
            if not params and invoke.group(2).strip():
                # No fabricated {}-arguments calls from unread bodies (#170).
                continue
            calls.append(_tool_call(invoke.group(1), params))
    if not calls:
        return text, None
    cleaned = re.sub(pattern, "", text, flags=re.DOTALL).strip()
    return cleaned, calls


def _parse_bracket_tool_calls(text: str) -> tuple[str, list[dict[str, Any]] | None]:
    calls: list[dict[str, Any]] = []
    pattern = r"\[(?:Calling tool|Tool call):\s*([A-Za-z_][\w.-]*)(?:\(({.*?})\))?\]"
    for name, args_text in re.findall(pattern, text, re.DOTALL):
        arguments: Any = {}
        if args_text:
            try:
                arguments = json.loads(args_text)
            except (TypeError, ValueError):
                arguments = {}
        calls.append(_tool_call(name, arguments))
    if not calls:
        return text, None
    return re.sub(pattern, "", text, flags=re.DOTALL).strip(), calls


def _allowed_tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    names: set[str] = set()
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and function.get("name"):
            names.add(str(function["name"]))
        elif isinstance(tool, dict) and tool.get("name"):
            names.add(str(tool["name"]))
    return names


def _filter_known_tools(
    calls: list[dict[str, Any]] | None,
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if not calls or not tools:
        return calls
    allowed = _allowed_tool_names(tools)
    filtered = [
        call
        for call in calls
        if call.get("function", {}).get("name") in allowed
    ]
    return filtered or None


def _function_body_is_blank(envelope: str) -> bool:
    """True when the tool envelope carries no payload beyond its tags.

    An empty ``<function=name></function>`` block is a legitimate no-argument
    call and must keep parsing to ``{}`` (the stream-level contract pinned by
    test_chat_stream_missing_required_tool_argument_still_emits_model_tool_call).
    A non-blank body that no parser could read must never become ``{}``.
    """
    inner = re.match(
        r"\s*<function(?:=[^>\s]+|\s+name=\"[^\"]+\")>\s*(.*?)\s*</function>\s*$",
        envelope.strip(),
        re.DOTALL,
    )
    if inner is None:
        return not envelope.strip()
    return not inner.group(1).strip()


def parse_tool_calls(
    text: str,
    tokenizer: Any | None,
    tools: list[dict[str, Any]] | None = None,
) -> ToolCallExtraction:
    cleaned_text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    raw_markup = any(
        marker in (text or "")
        for marker in ("<tool_call", "</tool_call>", "[Calling tool:", "[Tool call:")
    )

    if tokenizer is not None and getattr(tokenizer, "has_tool_calling", False):
        start = getattr(tokenizer, "tool_call_start", None)
        end = getattr(tokenizer, "tool_call_end", None)
        parser = getattr(tokenizer, "tool_parser", None)
        if start and parser:
            matches: list[str] = []
            if end:
                matches = re.findall(
                    rf"{re.escape(start)}(.*?){re.escape(end)}",
                    text or "",
                    flags=re.DOTALL,
                )
            elif start in (text or ""):
                matches = [
                    part
                    for part in re.split(re.escape(start), text or "")[1:]
                    if part.strip()
                ]
            calls: list[dict[str, Any]] = []
            for match in matches:
                try:
                    parsed = parser(match.strip(), tools)
                except Exception:
                    parsed = None
                items = parsed if isinstance(parsed, list) else [parsed]
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name")
                    if not name:
                        continue
                    arguments = item.get("arguments", {})
                    if not arguments and match.strip():
                        # #170: the native tokenizer parser returns an empty
                        # arguments object for envelope bodies it cannot read
                        # (a JSON-object body inside <function=...> being the
                        # common case). Never fabricate a {}-arguments call —
                        # give our own envelope parser a second opinion and
                        # adopt its arguments, or skip the item entirely.
                        _cleaned, recovered, _reason = _parse_xml_tool_calls(
                            "<tool_call>" + match.strip() + "</tool_call>"
                        )
                        recovered_args = None
                        for candidate in recovered or []:
                            candidate_fn = candidate.get("function") or {}
                            if str(candidate_fn.get("name")) == str(name):
                                try:
                                    recovered_args = json.loads(
                                        candidate_fn.get("arguments") or "{}"
                                    )
                                except (TypeError, ValueError):
                                    recovered_args = None
                                break
                        if recovered_args:
                            arguments = recovered_args
                        elif not _function_body_is_blank(match):
                            continue
                    calls.append(_tool_call(str(name), arguments))
            calls = _filter_known_tools(calls, tools) or []
            if calls:
                if end:
                    cleaned = re.sub(
                        rf"{re.escape(start)}.*?{re.escape(end)}",
                        "",
                        cleaned_text,
                        flags=re.DOTALL,
                    ).strip()
                else:
                    cleaned = cleaned_text.split(start, 1)[0].strip()
                return ToolCallExtraction(
                    cleaned_text=cleaned,
                    tool_calls=calls,
                    cleaned_thinking="",
                    parser_source="native",
                    status="parsed",
                    raw_tool_markup_suppressed=raw_markup,
                )

    if "<tool_call" in cleaned_text:
        cleaned, calls, malformed = _parse_xml_tool_calls(cleaned_text)
        filtered_calls = _filter_known_tools(calls, tools)
        if calls and not filtered_calls and tools:
            # OpenAI passes unknown-named calls through; the client owns the
            # rejection (and answers the model, letting it self-correct).
            # Degrading the whole turn to prose costs the agent a strike.
            filtered_calls = calls
        if filtered_calls:
            return ToolCallExtraction(
                cleaned_text=cleaned,
                tool_calls=filtered_calls,
                cleaned_thinking="",
                parser_source="qwen_xml",
                status="parsed",
                raw_tool_markup_suppressed=True,
            )
        return ToolCallExtraction(
            cleaned_text=cleaned_text,
            tool_calls=None,
            cleaned_thinking="",
            parser_source="qwen_xml",
            status="malformed_as_content",
            malformed_reason=malformed or "unclosed or invalid tool_call markup",
            raw_tool_markup_suppressed=False,
        )

    if re.search(r"<[A-Za-z_][\w.-]*:tool_call>", cleaned_text):
        cleaned, calls = _parse_namespaced_tool_calls(cleaned_text)
        calls = _filter_known_tools(calls, tools)
        if calls:
            return ToolCallExtraction(
                cleaned_text=cleaned,
                tool_calls=calls,
                cleaned_thinking="",
                parser_source="namespaced",
                status="parsed",
                raw_tool_markup_suppressed=True,
            )

    if "[Calling tool:" in cleaned_text or "[Tool call:" in cleaned_text:
        cleaned, calls = _parse_bracket_tool_calls(cleaned_text)
        calls = _filter_known_tools(calls, tools)
        if calls:
            return ToolCallExtraction(
                cleaned_text=cleaned,
                tool_calls=calls,
                cleaned_thinking="",
                parser_source="bracket",
                status="parsed",
                raw_tool_markup_suppressed=True,
            )

    return ToolCallExtraction(
        cleaned_text=cleaned_text,
        tool_calls=None,
        cleaned_thinking="",
        parser_source="none",
        status="no_tool",
        raw_tool_markup_suppressed=raw_markup,
    )


def sanitize_tool_call_markup(text: str, tokenizer: Any | None = None) -> str:
    if not text:
        return ""
    filtered = ToolCallStreamFilter(tokenizer)
    return filtered.feed(text) + filtered.finish()


def extract_tool_calls_with_thinking(
    thinking_content: str,
    regular_content: str,
    tokenizer: Any | None,
    tools: list[dict[str, Any]] | None = None,
) -> ToolCallExtraction:
    result = parse_tool_calls(regular_content, tokenizer, tools)
    cleaned_thinking = sanitize_tool_call_markup(thinking_content, tokenizer)
    calls = result.tool_calls
    status = result.status
    source = result.parser_source
    malformed = result.malformed_reason
    if not calls and thinking_content:
        thinking_result = parse_tool_calls(thinking_content, tokenizer, tools)
        if thinking_result.tool_calls and not regular_content.strip():
            calls = thinking_result.tool_calls
            source = thinking_result.parser_source
            status = thinking_result.status
        elif thinking_result.status == "malformed_as_content" and status == "no_tool":
            status = thinking_result.status
            malformed = thinking_result.malformed_reason
            source = thinking_result.parser_source
    return ToolCallExtraction(
        cleaned_text=result.cleaned_text,
        tool_calls=calls,
        cleaned_thinking=cleaned_thinking,
        parser_source=source,
        status=status,
        malformed_reason=malformed,
        raw_tool_markup_suppressed=(
            result.raw_tool_markup_suppressed or cleaned_thinking != thinking_content
        ),
    )


class ToolCallStreamFilter:
    """Suppress tool-control markup while preserving normal streamed text."""

    def __init__(self, tokenizer: Any | None = None) -> None:
        start = getattr(tokenizer, "tool_call_start", None) if tokenizer is not None else None
        end = getattr(tokenizer, "tool_call_end", None) if tokenizer is not None else None
        self._marker_pairs: list[tuple[str, str]] = [("<tool_call>", "</tool_call>")]
        self._suppress_after_markers: list[str] = []
        if start:
            if end:
                self._marker_pairs.insert(0, (str(start), str(end)))
            else:
                self._suppress_after_markers.append(str(start))
        self._namespaced_open_re = re.compile(r"<([A-Za-z_][\w.-]*):tool_call>")
        self._bracket_prefixes = ["[Calling tool:", "[Tool call:"]
        self._bracket_call_re = re.compile(
            r"^\[(?:Calling tool|Tool call):\s*([A-Za-z_][\w.-]*)(?:\(({.*?})\))?\]",
            re.DOTALL,
        )
        self._buffer = ""
        self._suppressing_until: str | None = None
        self._suppressing = False
        self.suppressed_markup = False

    def _find_start_envelope(self, text: str) -> tuple[int, int, str | None] | None:
        starts: list[tuple[int, int, str | None]] = []
        for marker, close in self._marker_pairs:
            index = text.find(marker)
            if index >= 0:
                starts.append((index, len(marker), close))
        if match := self._namespaced_open_re.search(text):
            namespace = match.group(1)
            starts.append((match.start(), len(match.group(0)), f"</{namespace}:tool_call>"))
        for prefix in self._bracket_prefixes:
            index = text.find(prefix)
            while index >= 0:
                candidate = text[index:]
                bracket = self._bracket_call_re.match(candidate)
                if bracket:
                    starts.append((index, bracket.end(), None))
                index = text.find(prefix, index + 1)
        for marker in self._suppress_after_markers:
            index = text.find(marker)
            if index >= 0:
                starts.append((index, len(text) - index, "__suppress_permanently__"))
        return min(starts, key=lambda item: item[0]) if starts else None

    @staticmethod
    def _partial_prefix_len(text: str, marker: str) -> int:
        max_len = min(len(text), len(marker) - 1)
        for size in range(max_len, 0, -1):
            if text.endswith(marker[:size]):
                return size
        return 0

    @staticmethod
    def _could_be_partial_namespaced_open(candidate: str) -> bool:
        if not candidate.startswith("<") or ">" in candidate:
            return False
        body = candidate[1:]
        if not body or body.startswith("/"):
            return bool(not body)
        if ":" not in body:
            return re.match(r"^[A-Za-z_][\w.-]*$", body) is not None
        namespace, suffix = body.split(":", 1)
        return bool(
            re.match(r"^[A-Za-z_][\w.-]*$", namespace)
            and "tool_call".startswith(suffix)
        )

    def _partial_suffix_len(self, text: str) -> int:
        keep = 0
        for marker, _close in self._marker_pairs:
            keep = max(keep, self._partial_prefix_len(text, marker))
        for marker in self._suppress_after_markers:
            keep = max(keep, self._partial_prefix_len(text, marker))
        if (last_lt := text.rfind("<")) >= 0:
            candidate = text[last_lt:]
            if self._could_be_partial_namespaced_open(candidate):
                keep = max(keep, len(candidate))
        for prefix in self._bracket_prefixes:
            keep = max(keep, self._partial_prefix_len(text, prefix))
            if (index := text.rfind(prefix)) >= 0 and "]" not in text[index:]:
                return max(keep, len(text[index:]))
        return min(keep, 128)

    def _should_drop_tail_at_finish(self, tail: str) -> bool:
        if not tail:
            return False
        for marker, _close in self._marker_pairs:
            if marker.startswith(tail):
                return True
        for prefix in self._bracket_prefixes:
            if tail.startswith(prefix):
                return True
        return tail.startswith("<") and ">" not in tail and ":" in tail

    def feed(self, text: str) -> str:
        if self._suppressing or not text:
            return ""
        self._buffer += text
        out: list[str] = []
        while self._buffer:
            if self._suppressing_until == "__suppress_permanently__":
                self.suppressed_markup = True
                self._suppressing = True
                self._suppressing_until = None
                self._buffer = ""
                break
            if self._suppressing_until is not None:
                end_index = self._buffer.find(self._suppressing_until)
                if end_index < 0:
                    keep = self._partial_prefix_len(self._buffer, self._suppressing_until)
                    self._buffer = self._buffer[-keep:] if keep else ""
                    break
                self.suppressed_markup = True
                self._buffer = self._buffer[end_index + len(self._suppressing_until) :]
                self._suppressing_until = None
                continue
            start = self._find_start_envelope(self._buffer)
            if start:
                index, consume_len, close_marker = start
                if index > 0:
                    out.append(self._buffer[:index])
                self.suppressed_markup = True
                self._buffer = self._buffer[index + consume_len :]
                if close_marker is not None:
                    self._suppressing_until = close_marker
                continue
            keep = self._partial_suffix_len(self._buffer)
            if keep == 0:
                out.append(self._buffer)
                self._buffer = ""
                break
            if len(self._buffer) > keep:
                out.append(self._buffer[:-keep])
                self._buffer = self._buffer[-keep:]
            break
        return "".join(out)

    def finish(self) -> str:
        if self._suppressing or self._suppressing_until is not None:
            self.suppressed_markup = True
            self._buffer = ""
            self._suppressing_until = None
            return ""
        keep = self._partial_suffix_len(self._buffer)
        if keep >= len(self._buffer):
            tail = self._buffer
            self._buffer = ""
            if self._should_drop_tail_at_finish(tail):
                self.suppressed_markup = True
                return ""
            return tail
        text = self._buffer[:-keep] if keep else self._buffer
        self._buffer = ""
        return text
