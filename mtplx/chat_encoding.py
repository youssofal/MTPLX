"""Backend-aware chat prompt encoding helpers.

This module is deliberately MLX-free. Public surfaces that need to turn chat
messages into token ids should come through here instead of each surface
recreating Qwen-shaped assumptions.
"""

from __future__ import annotations

import json
from typing import Any

GEMMA4_THINK_OPEN = "<|channel>"
GEMMA4_THINK_CLOSE = "<channel|>"
GEMMA4_THOUGHT_PREFIX = "thought\n"
GEMMA4_EMPTY_THOUGHT_BLOCK = (
    f"{GEMMA4_THINK_OPEN}{GEMMA4_THOUGHT_PREFIX}{GEMMA4_THINK_CLOSE}"
)
QWEN_THINK_OPEN = "<think>"
QWEN_THINK_CLOSE = "</think>"


def is_gemma4_tokenizer(tokenizer: Any) -> bool:
    try:
        special = getattr(tokenizer, "model_specific_special_tokens", None) or {}
    except Exception:
        special = {}
    try:
        if (
            special.get("think_token") == "<|think|>"
            and special.get("soc_token") == "<|channel>"
            and special.get("eoc_token") == "<channel|>"
        ):
            return True
        if (
            str(getattr(tokenizer, "think_token", "")) == "<|think|>"
            and str(getattr(tokenizer, "soc_token", "")) == "<|channel>"
            and str(getattr(tokenizer, "eoc_token", "")) == "<channel|>"
        ):
            return True
        vocab = tokenizer.get_vocab()
        return all(
            token in vocab
            for token in ("<|think|>", "<|channel>", "<channel|>", "<|turn>", "<turn|>")
        )
    except Exception:
        return False


def encode_without_added_special_tokens(tokenizer: Any, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def strip_gemma4_thinking_text(text: str) -> str:
    while GEMMA4_THINK_OPEN in text and GEMMA4_THINK_CLOSE in text:
        start = text.find(GEMMA4_THINK_OPEN)
        end = text.find(GEMMA4_THINK_CLOSE, start)
        if end < 0:
            break
        text = text[:start] + text[end + len(GEMMA4_THINK_CLOSE) :]
    while QWEN_THINK_OPEN in text and QWEN_THINK_CLOSE in text:
        start = text.find(QWEN_THINK_OPEN)
        end = text.find(QWEN_THINK_CLOSE, start)
        if end < 0:
            break
        text = text[:start] + text[end + len(QWEN_THINK_CLOSE) :]
    return text


def _tool_instruction_message(tools: list[dict[str, Any]]) -> dict[str, str]:
    lines = [
        "Tool calling is available. If a tool is needed, respond with exactly one XML tool call in this format:",
        "<tool_call>",
        "<function=TOOL_NAME>",
        "<parameter=ARGUMENT_NAME>",
        "ARGUMENT_VALUE",
        "</parameter>",
        "</function>",
        "</tool_call>",
        "",
        "Available tools:",
    ]
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        function = function if isinstance(function, dict) else {}
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        description = str(function.get("description") or "").strip()
        parameters = function.get("parameters") or {}
        try:
            parameters_text = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        except TypeError:
            parameters_text = str(parameters)
        line = f"- {name}"
        if description:
            line += f": {description}"
        lines.append(line)
        lines.append(f"  parameters: {parameters_text}")
    return {"role": "system", "content": "\n".join(lines)}


def _prepend_tool_instruction(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not tools:
        return messages
    return [_tool_instruction_message(tools), *messages]


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    pieces.append(str(item.get("text") or ""))
                elif "text" in item:
                    pieces.append(str(item.get("text") or ""))
            elif item is not None:
                pieces.append(str(item))
        return "".join(pieces)
    return str(content)


def _tool_call_arguments(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"arguments": raw}
        return parsed if isinstance(parsed, dict) else {"arguments": parsed}
    return {"arguments": raw}


def _tool_argument_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _gemma4_tool_call_text(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function") if isinstance(tool_call, dict) else None
    if not isinstance(function, dict):
        return ""
    name = str(function.get("name") or "").strip()
    if not name:
        return ""
    lines = ["<tool_call>", f"<function={name}>"]
    for key, value in _tool_call_arguments(function.get("arguments")).items():
        argument_name = str(key).strip()
        if not argument_name:
            continue
        lines.extend(
            [
                f"<parameter={argument_name}>",
                _tool_argument_text(value),
                "</parameter>",
            ]
        )
    lines.extend(["</function>", "</tool_call>"])
    return "\n".join(lines)


def _gemma4_tool_calls_text(tool_calls: Any) -> str:
    if not isinstance(tool_calls, list):
        return ""
    rendered: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        text = _gemma4_tool_call_text(tool_call)
        if text:
            rendered.append(text)
    return "\n".join(rendered)


def encode_gemma4_messages(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    enable_thinking: bool,
    add_generation_prompt: bool,
) -> list[int]:
    """Encode Gemma 4 text chat turns when a converted artifact has no template."""

    bos = str(getattr(tokenizer, "bos_token", None) or "<bos>")
    parts: list[str] = [bos]
    body_messages = list(messages)
    system_text = ""

    if body_messages and str(body_messages[0].get("role") or "") in {
        "system",
        "developer",
    }:
        system_text = _content_to_text(body_messages[0].get("content")).strip()
        body_messages = body_messages[1:]

    if enable_thinking or system_text:
        parts.append("<|turn>system\n")
        if enable_thinking:
            parts.append("<|think|>\n")
        if system_text:
            parts.append(system_text)
        parts.append("<turn|>\n")

    for item in body_messages:
        role = str(item.get("role") or "")
        content = _content_to_text(item.get("content"))
        if role == "assistant":
            role = "model"
            content = strip_gemma4_thinking_text(content)
            reasoning = _content_to_text(
                item.get("reasoning_content") or item.get("reasoning")
            )
            if enable_thinking and reasoning:
                content = (
                    f"{GEMMA4_THINK_OPEN}{GEMMA4_THOUGHT_PREFIX}"
                    f"{reasoning}{GEMMA4_THINK_CLOSE}{content}"
                )
            tool_call_text = _gemma4_tool_calls_text(item.get("tool_calls"))
            if tool_call_text:
                content = (
                    f"{content.rstrip()}\n{tool_call_text}"
                    if content.strip()
                    else tool_call_text
                )
        elif role == "tool":
            role = "tool_response"
        if role not in {"user", "model", "tool_response"}:
            continue
        parts.append(f"<|turn>{role}\n")
        if content:
            parts.append(content if role == "user" else content.strip())
        parts.append("<turn|>\n")

    if add_generation_prompt:
        parts.append("<|turn>model\n")
        if not enable_thinking:
            parts.append(GEMMA4_EMPTY_THOUGHT_BLOCK)

    return encode_without_added_special_tokens(tokenizer, "".join(parts))


def encode_chat_messages(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    enable_thinking: bool | None,
    reasoning_effort: str | None = None,
    add_generation_prompt: bool = True,
    preserve_thinking: bool = True,
    tools: list[dict[str, Any]] | None = None,
) -> list[int]:
    if not messages:
        messages = [{"role": "user", "content": ""}]
    thinking = bool(enable_thinking)
    if is_gemma4_tokenizer(tokenizer):
        return encode_gemma4_messages(
            tokenizer,
            _prepend_tool_instruction(messages, tools),
            enable_thinking=thinking,
            add_generation_prompt=add_generation_prompt,
        )

    template_kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
        "preserve_thinking": preserve_thinking,
    }
    if reasoning_effort:
        template_kwargs["reasoning_effort"] = reasoning_effort
    if tools:
        template_kwargs["tools"] = tools
    try:
        return list(tokenizer.apply_chat_template(messages, **template_kwargs))
    except TypeError:
        fallback_kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
        }
        if tools:
            fallback_kwargs["tools"] = tools
        return list(tokenizer.apply_chat_template(messages, **fallback_kwargs))
    except Exception:
        if getattr(tokenizer, "chat_template", None):
            raise
        prompt = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}" for item in messages
        )
        if add_generation_prompt:
            prompt += "\nassistant:"
        return list(tokenizer.encode(prompt))
