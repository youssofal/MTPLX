"""Stateless OpenAI Responses API translation for the MTPLX chat runtime.

This module deliberately knows nothing about ``openai.py``'s request models or
inference implementation.  It converts Responses requests into protocol-neutral
chat data and translates chat payloads/SSE back to Responses objects.  The
server injects those data into its existing chat-completions path.

Supported tools are client-executed top-level function/custom tools and
function tools grouped in a namespace.  Hosted tools (including web search and
tool search) are rejected because MTPLX cannot execute them.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Mapping

from pydantic import BaseModel, ConfigDict


class ResponsesRequest(BaseModel):
    """The deliberately bounded Responses request surface supported by MTPLX."""

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    input: Any = None
    instructions: str | None = None
    max_output_tokens: int | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    stop: Any = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None
    metadata: dict[str, Any] | None = None
    user: str | None = None
    include: list[str] | None = None
    stream_options: dict[str, Any] | None = None
    prompt_cache_key: str | None = None
    client_metadata: dict[str, Any] | None = None
    service_tier: str | None = None
    reasoning: dict[str, Any] | None = None
    text: dict[str, Any] | None = None
    store: bool | None = None
    background: bool | None = None
    previous_response_id: str | None = None


class ResponsesProtocolError(ValueError):
    """A caller-visible incompatibility with the supported local subset."""


@dataclass(frozen=True)
class NamespaceFunction:
    """Original Responses identity for a flattened chat function name."""

    namespace: str
    name: str


@dataclass
class ToolConversion:
    chat_tools: list[dict[str, Any]] | None
    top_level_tool_names: set[str] = field(default_factory=set)
    custom_tool_names: set[str] = field(default_factory=set)
    namespace_functions: dict[str, NamespaceFunction] = field(default_factory=dict)
    namespace_flat_names: dict[tuple[str, str], str] = field(default_factory=dict)
    hosted_tool_types: list[str] = field(default_factory=list)


@dataclass
class ResponsesConversion:
    """Protocol-neutral chat request plus the reversible output contract."""

    chat: dict[str, Any]
    response_fields: dict[str, Any]
    custom_tool_names: set[str]
    namespace_functions: dict[str, NamespaceFunction]


def unsupported(message: str) -> ResponsesProtocolError:
    return ResponsesProtocolError(message)


def _text(value: Any, *, field_name: str) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if not isinstance(value, list):
        raise unsupported(f"{field_name} must be text or a list of text parts")
    chunks: list[str] = []
    for index, part in enumerate(value):
        if not isinstance(part, Mapping):
            raise unsupported(f"{field_name}[{index}] must be a text part")
        part_type = str(part.get("type") or "").strip().lower()
        if part_type not in {"input_text", "output_text", "text"}:
            raise unsupported(
                f"{field_name}[{index}].type={part_type or '<missing>'!r} is not "
                "supported; only text content is supported"
            )
        text = part.get("text")
        if not isinstance(text, str):
            raise unsupported(f"{field_name}[{index}].text must be a string")
        chunks.append(text)
    return "".join(chunks)


def _arguments(value: Any, *, field_name: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    raise unsupported(f"{field_name} must be a JSON string or object")


def _function_chat_tool(
    tool: Mapping[str, Any],
    *,
    field_name: str,
    chat_name: str | None = None,
) -> dict[str, Any]:
    unknown = sorted(
        set(tool)
        - {
            "type",
            "name",
            "description",
            "parameters",
            "strict",
            "defer_loading",
        }
    )
    if unknown:
        raise unsupported(
            f"{field_name} function tool has unsupported key(s): " + ", ".join(unknown)
        )
    if tool.get("defer_loading"):
        raise unsupported(
            f"{field_name}.defer_loading=true requires hosted tool_search"
        )
    name = str(tool.get("name") or "").strip()
    if not name:
        raise unsupported(f"{field_name}.name is required")
    parameters = tool.get("parameters", {"type": "object", "properties": {}})
    if not isinstance(parameters, Mapping):
        raise unsupported(f"{field_name}.parameters must be an object")
    function: dict[str, Any] = {
        "name": chat_name or name,
        "parameters": dict(parameters),
    }
    if isinstance(tool.get("description"), str):
        function["description"] = tool["description"]
    if "strict" in tool:
        if not isinstance(tool["strict"], bool):
            raise unsupported(f"{field_name}.strict must be boolean")
        function["strict"] = tool["strict"]
    return {"type": "function", "function": function}


def _custom_chat_tool(tool: Mapping[str, Any], *, field_name: str) -> dict[str, Any]:
    unknown = sorted(set(tool) - {"type", "name", "description", "format"})
    if unknown:
        raise unsupported(
            f"{field_name} custom tool has unsupported key(s): " + ", ".join(unknown)
        )
    name = str(tool.get("name") or "").strip()
    if not name:
        raise unsupported(f"{field_name}.name is required")
    custom_format = tool.get("format")
    if custom_format is not None and custom_format != {"type": "text"}:
        raise unsupported(f"{field_name}.format supports only freeform text")
    description = str(tool.get("description") or "").strip()
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (description + "\n\n" if description else "")
            + "Return the custom tool's freeform input in the input string.",
            "parameters": {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
                "additionalProperties": False,
            },
        },
    }


_NAME_PART_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _name_part(value: str) -> str:
    return _NAME_PART_RE.sub("_", value).strip("_") or "tool"


def _namespace_chat_name(namespace: str, name: str, used: set[str]) -> str:
    digest = hashlib.sha256(f"{namespace}\0{name}".encode()).hexdigest()[:12]
    base = f"ns_{_name_part(namespace)[:18]}_{_name_part(name)[:18]}_{digest}"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base[:60]}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def convert_tools(tools: list[dict[str, Any]] | None) -> ToolConversion:
    """Convert client tools without rejecting hosted entries yet.

    Keeping hosted-tool detection in the result makes the contract testable:
    all function and namespace definitions are validated/flattened before the
    caller rejects the hosted capability that MTPLX cannot execute.
    """

    if tools is None:
        return ToolConversion(chat_tools=None)
    converted: list[dict[str, Any]] = []
    custom_names: set[str] = set()
    namespace_functions: dict[str, NamespaceFunction] = {}
    namespace_flat_names: dict[tuple[str, str], str] = {}
    hosted: list[str] = []
    declared_top_level: dict[str, tuple[int, str]] = {}
    for index, tool in enumerate(tools):
        if not isinstance(tool, Mapping):
            continue
        tool_type = str(tool.get("type") or "").strip().lower()
        if tool_type not in {"function", "custom", "namespace"}:
            continue
        name = str(tool.get("name") or "").strip()
        if not name:
            continue
        previous = declared_top_level.get(name)
        if previous is not None:
            previous_index, previous_type = previous
            raise unsupported(
                f"tools[{index}].name={name!r} duplicates "
                f"tools[{previous_index}].name ({tool_type} vs {previous_type}); "
                "top-level tool names must be unique across function, custom, "
                "and namespace tools"
            )
        declared_top_level[name] = (index, tool_type)
    used_names = set(declared_top_level)
    top_level_names = {
        name
        for name, (_index, tool_type) in declared_top_level.items()
        if tool_type in {"function", "custom"}
    }
    for index, tool in enumerate(tools):
        field_name = f"tools[{index}]"
        if not isinstance(tool, Mapping):
            raise unsupported(f"{field_name} must be an object")
        tool_type = str(tool.get("type") or "").strip().lower()
        if tool_type == "function":
            converted.append(_function_chat_tool(tool, field_name=field_name))
            continue
        if tool_type == "custom":
            converted.append(_custom_chat_tool(tool, field_name=field_name))
            custom_names.add(str(tool.get("name") or "").strip())
            continue
        if tool_type == "namespace":
            unknown = sorted(set(tool) - {"type", "name", "description", "tools"})
            if unknown:
                raise unsupported(
                    f"{field_name} namespace has unsupported key(s): "
                    + ", ".join(unknown)
                )
            namespace = str(tool.get("name") or "").strip()
            if not namespace:
                raise unsupported(f"{field_name}.name is required")
            if not isinstance(tool.get("description"), str):
                raise unsupported(
                    f"{field_name}.description is required for namespace tools"
                )
            nested_tools = tool.get("tools")
            if not isinstance(nested_tools, list) or not nested_tools:
                raise unsupported(f"{field_name}.tools must be a non-empty list")
            for nested_index, nested in enumerate(nested_tools):
                nested_field = f"{field_name}.tools[{nested_index}]"
                if not isinstance(nested, Mapping):
                    raise unsupported(f"{nested_field} must be an object")
                nested_type = str(nested.get("type") or "").strip().lower()
                if nested_type != "function":
                    raise unsupported(
                        f"{nested_field}.type={nested_type or '<missing>'!r} is not "
                        "supported; namespace entries must be client-executed functions"
                    )
                nested_name = str(nested.get("name") or "").strip()
                if not nested_name:
                    raise unsupported(f"{nested_field}.name is required")
                pair = (namespace, nested_name)
                if pair in namespace_flat_names:
                    raise unsupported(
                        f"{field_name} repeats namespace function {namespace}.{nested_name}"
                    )
                flat_name = _namespace_chat_name(namespace, nested_name, used_names)
                converted.append(
                    _function_chat_tool(
                        nested, field_name=nested_field, chat_name=flat_name
                    )
                )
                identity = NamespaceFunction(namespace=namespace, name=nested_name)
                namespace_functions[flat_name] = identity
                namespace_flat_names[pair] = flat_name
            continue
        if not tool_type:
            raise unsupported(f"{field_name}.type is required")
        hosted.append(tool_type)
    return ToolConversion(
        chat_tools=converted,
        top_level_tool_names=top_level_names,
        custom_tool_names=custom_names,
        namespace_functions=namespace_functions,
        namespace_flat_names=namespace_flat_names,
        hosted_tool_types=sorted(set(hosted)),
    )


def _input_to_chat_messages(
    input_value: Any,
    *,
    namespace_flat_names: Mapping[tuple[str, str], str],
) -> list[dict[str, Any]]:
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]
    if not isinstance(input_value, list) or not input_value:
        raise unsupported("input must be a non-empty text string or item list")

    messages: list[dict[str, Any]] = []
    seen_function_calls: set[str] = set()
    for index, item in enumerate(input_value):
        if not isinstance(item, Mapping):
            raise unsupported(f"input[{index}] must be an object")
        item_type = str(item.get("type") or "message").strip().lower()
        if item_type in {"function_call", "custom_tool_call"}:
            call_id = str(item.get("call_id") or "").strip()
            name = str(item.get("name") or "").strip()
            if not call_id or not name:
                raise unsupported(
                    f"input[{index}] {item_type} requires call_id and name"
                )
            namespace = str(item.get("namespace") or "").strip()
            chat_name = name
            if namespace:
                chat_name = namespace_flat_names.get((namespace, name), "")
                if not chat_name:
                    raise unsupported(
                        f"input[{index}] references undeclared namespace function "
                        f"{namespace}.{name}"
                    )
            arguments = (
                json.dumps(
                    {
                        "input": _text(
                            item.get("input"), field_name=f"input[{index}].input"
                        )
                    },
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                if item_type == "custom_tool_call"
                else _arguments(
                    item.get("arguments", ""),
                    field_name=f"input[{index}].arguments",
                )
            )
            tool_call = {
                "id": call_id,
                "type": "function",
                "function": {"name": chat_name, "arguments": arguments},
            }
            if (
                messages
                and messages[-1]["role"] == "assistant"
                and messages[-1].get("tool_calls")
            ):
                messages[-1]["tool_calls"].append(tool_call)
            else:
                messages.append(
                    {"role": "assistant", "content": "", "tool_calls": [tool_call]}
                )
            seen_function_calls.add(call_id)
            continue
        if item_type in {"function_call_output", "custom_tool_call_output"}:
            call_id = str(item.get("call_id") or "").strip()
            if not call_id:
                raise unsupported(f"input[{index}] {item_type} requires call_id")
            if call_id not in seen_function_calls:
                raise unsupported(
                    "function_call_output continuation requires the matching "
                    "function_call earlier in this request; previous responses "
                    "are not stored"
                )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _text(
                        item.get("output"), field_name=f"input[{index}].output"
                    ),
                }
            )
            continue
        if item_type != "message":
            raise unsupported(f"input[{index}].type={item_type!r} is not supported")
        role = str(item.get("role") or "user").strip().lower()
        if role == "developer":
            role = "system"
        if role not in {"system", "user", "assistant"}:
            raise unsupported(f"input[{index}].role={role!r} is not supported")
        messages.append(
            {
                "role": role,
                "content": _text(
                    item.get("content"), field_name=f"input[{index}].content"
                ),
            }
        )
    if not messages:
        raise unsupported("input must contain at least one supported item")
    return messages


def _tool_choice_to_chat(tool_choice: Any, tools: ToolConversion) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice not in {"auto", "none", "required"}:
            raise unsupported(
                "tool_choice must be auto, none, required, or a function selector"
            )
        return tool_choice
    if not isinstance(tool_choice, Mapping):
        raise unsupported("tool_choice must be a string or function selector")
    unknown = sorted(set(tool_choice) - {"type", "name"})
    if unknown:
        if "namespace" in unknown:
            raise unsupported(
                "tool_choice cannot qualify a namespace function: the current "
                "Responses function-selector schema has only type and name; use "
                "auto or required for namespace tools"
            )
        raise unsupported(
            "tool_choice function selector has unsupported key(s): "
            + ", ".join(unknown)
        )
    choice_type = str(tool_choice.get("type") or "").strip().lower()
    if choice_type not in {"function", "custom"}:
        raise unsupported(
            f"tool_choice.type={choice_type or '<missing>'!r} is not supported"
        )
    name = str(tool_choice.get("name") or "").strip()
    if not name:
        raise unsupported("tool_choice function selector requires name")
    nested_matches = {
        identity.namespace
        for identity in tools.namespace_functions.values()
        if identity.name == name
    }
    if name not in tools.top_level_tool_names and nested_matches:
        raise unsupported(
            "tool_choice cannot explicitly select namespace function "
            f"{name!r}: the current Responses selector has no namespace field; "
            "use auto or required"
        )
    return {"type": "function", "function": {"name": name}}


def _text_format_to_chat(text: Any) -> tuple[Any, str | None]:
    if text is None:
        return None, None
    if not isinstance(text, Mapping):
        raise unsupported("text must be an object")
    unknown = sorted(set(text) - {"format", "verbosity"})
    if unknown:
        raise unsupported("text has unsupported key(s): " + ", ".join(unknown))
    verbosity = text.get("verbosity")
    if verbosity is not None and verbosity not in {"low", "medium", "high"}:
        raise unsupported("text.verbosity must be low, medium, or high")
    response_format = text.get("format")
    if response_format is None:
        return None, verbosity
    if not isinstance(response_format, Mapping):
        raise unsupported("text.format must be an object")
    format_type = str(response_format.get("type") or "").strip().lower()
    if format_type == "text":
        return {"type": "text"}, verbosity
    if format_type == "json_object":
        return {"type": "json_object"}, verbosity
    if format_type != "json_schema":
        raise unsupported(
            f"text.format.type={format_type or '<missing>'!r} is not supported"
        )
    schema = response_format.get("schema")
    if not isinstance(schema, Mapping):
        raise unsupported("text.format json_schema requires schema")
    wrapper: dict[str, Any] = {"schema": dict(schema)}
    for key in ("name", "description", "strict"):
        if key in response_format:
            wrapper[key] = response_format[key]
    return {"type": "json_schema", "json_schema": wrapper}, verbosity


def _reasoning_to_chat(reasoning: Any) -> tuple[str | None, str | None]:
    if reasoning is None:
        return None, None
    if not isinstance(reasoning, Mapping):
        raise unsupported("reasoning must be an object")
    unknown = sorted(set(reasoning) - {"effort", "summary"})
    if unknown:
        raise unsupported("reasoning has unsupported key(s): " + ", ".join(unknown))
    summary = reasoning.get("summary")
    if summary not in {None, "none", "auto"}:
        raise unsupported(
            "reasoning.summary supports only auto/none; MTPLX emits no separate "
            "reasoning summary item"
        )
    raw_effort = reasoning.get("effort")
    if raw_effort is None:
        return None, None
    requested = str(raw_effort).strip().lower()
    if requested not in {"auto", "low", "medium", "high", "xhigh"}:
        raise unsupported(
            "reasoning.effort must be one of: auto, low, medium, high, xhigh"
        )
    # RequestPolicy resolves this vocabulary against the loaded backend. Keep
    # only the client request here so observability cannot claim an effective
    # level before family policy has accepted, clamped, or rejected it.
    return requested, requested


def _validate_stream_options(stream_options: Any) -> None:
    if stream_options is None:
        return
    if not isinstance(stream_options, Mapping):
        raise unsupported("stream_options must be an object")
    unknown = sorted(set(stream_options) - {"include_usage", "include_obfuscation"})
    if unknown:
        raise unsupported(
            "stream_options has unsupported key(s): " + ", ".join(unknown)
        )
    for key in ("include_usage", "include_obfuscation"):
        if key in stream_options and not isinstance(stream_options[key], bool):
            raise unsupported(f"stream_options.{key} must be a boolean")
    if stream_options.get("include_obfuscation") is True:
        raise unsupported(
            "stream_options.include_obfuscation=true is not supported: "
            "MTPLX does not implement Responses stream padding"
        )


def response_fields(request: ResponsesRequest) -> dict[str, Any]:
    max_output_tokens = next(
        (
            value
            for value in (
                request.max_output_tokens,
                request.max_tokens,
                request.max_completion_tokens,
            )
            if value is not None
        ),
        None,
    )
    return {
        "instructions": request.instructions,
        "metadata": (
            {str(key): str(value) for key, value in request.metadata.items()}
            if request.metadata is not None
            else None
        ),
        "temperature": request.temperature,
        "tool_choice": request.tool_choice or "auto",
        "tools": list(request.tools or []),
        "parallel_tool_calls": (
            True if request.parallel_tool_calls is None else request.parallel_tool_calls
        ),
        "top_p": request.top_p,
        "background": False,
        "max_output_tokens": max_output_tokens,
        "previous_response_id": None,
        "prompt_cache_key": request.prompt_cache_key,
        "reasoning": request.reasoning,
        "service_tier": request.service_tier,
        "text": request.text,
        "user": request.user,
    }


def translate_request(request: ResponsesRequest) -> ResponsesConversion:
    extras = sorted((request.model_extra or {}).keys())
    if extras:
        raise unsupported("unsupported field(s): " + ", ".join(extras))
    if request.background:
        raise unsupported("background=true is not supported")
    if request.store not in {None, False}:
        raise unsupported("store=true is not supported; MTPLX has no Responses storage")
    if request.previous_response_id:
        raise unsupported(
            "previous_response_id is not supported; send the full conversation in input"
        )
    unsupported_includes = sorted(
        set(request.include or []) - {"reasoning.encrypted_content"}
    )
    if unsupported_includes:
        raise unsupported(
            "include has unsupported value(s): " + ", ".join(unsupported_includes)
        )
    _validate_stream_options(request.stream_options)
    if request.prompt_cache_key is not None and not request.prompt_cache_key.strip():
        raise unsupported("prompt_cache_key must not be empty")
    if request.client_metadata is not None and not isinstance(
        request.client_metadata, Mapping
    ):
        raise unsupported("client_metadata must be an object")
    if request.service_tier not in {None, "auto", "default"}:
        raise unsupported(
            "service_tier is only supported as auto or default on this local server"
        )
    max_values = [
        value
        for value in (
            request.max_output_tokens,
            request.max_tokens,
            request.max_completion_tokens,
        )
        if value is not None
    ]
    if len(set(max_values)) > 1:
        raise unsupported(
            "max_output_tokens, max_tokens, and max_completion_tokens disagree"
        )

    tool_conversion = convert_tools(request.tools)
    messages = _input_to_chat_messages(
        request.input,
        namespace_flat_names=tool_conversion.namespace_flat_names,
    )
    if request.instructions is not None:
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = (
                f"{request.instructions}\n\n{messages[0].get('content') or ''}"
            )
        else:
            messages.insert(0, {"role": "system", "content": request.instructions})
    response_format, verbosity = _text_format_to_chat(request.text)
    reasoning_effort, reasoning_requested = _reasoning_to_chat(request.reasoning)
    metadata = dict(request.metadata or {})
    for internal_key in (
        "responses_reasoning_effort_requested",
        "responses_reasoning_effort_effective",
        "responses_reasoning_effort_downgraded",
    ):
        metadata.pop(internal_key, None)
    if request.prompt_cache_key is not None:
        metadata.setdefault("session_id", request.prompt_cache_key)
    if request.client_metadata is not None:
        metadata["responses_client_metadata"] = dict(request.client_metadata)
    if request.include:
        metadata["responses_include"] = list(request.include)
    if request.service_tier is not None:
        metadata["responses_service_tier"] = request.service_tier
    if verbosity is not None:
        metadata["responses_text_verbosity"] = verbosity
    if request.reasoning and request.reasoning.get("summary") is not None:
        metadata["responses_reasoning_summary"] = request.reasoning["summary"]
    if reasoning_requested is not None:
        metadata["responses_reasoning_effort_requested"] = reasoning_requested
    if request.stream_options:
        metadata["responses_stream_options"] = dict(request.stream_options)
    if any(
        value is not None
        for value in (
            request.temperature,
            request.top_p,
            request.top_k,
            request.presence_penalty,
            request.frequency_penalty,
            reasoning_effort,
        )
    ):
        metadata["allow_client_controls"] = True

    # Reject only after all client-executed tools and the real structured input
    # have been converted.  This distinguishes unavailable hosted execution
    # from a false claim that Codex's namespace schema is unsupported.
    if tool_conversion.hosted_tool_types:
        joined = ", ".join(tool_conversion.hosted_tool_types)
        raise unsupported(
            f"hosted tool type(s) unavailable on local MTPLX: {joined}; "
            "client-executed function and namespace tools are supported"
        )

    chat = {
        "model": request.model,
        "messages": messages,
        "max_tokens": max_values[0] if max_values else None,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "presence_penalty": request.presence_penalty,
        "frequency_penalty": request.frequency_penalty,
        "seed": request.seed,
        "stop": request.stop,
        "stream": bool(request.stream),
        "tools": tool_conversion.chat_tools,
        "tool_choice": _tool_choice_to_chat(request.tool_choice, tool_conversion),
        "parallel_tool_calls": request.parallel_tool_calls,
        "response_format": response_format,
        "metadata": metadata or None,
        "user": request.user,
        "reasoning_effort": reasoning_effort,
        "suppress_stats_footer": True,
    }
    return ResponsesConversion(
        chat=chat,
        response_fields=response_fields(request),
        custom_tool_names=tool_conversion.custom_tool_names,
        namespace_functions=tool_conversion.namespace_functions,
    )


def _usage(chat_usage: Any) -> dict[str, Any]:
    usage = chat_usage if isinstance(chat_usage, Mapping) else {}
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    cached_tokens = int(
        (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    )
    completion_details = usage.get("completion_tokens_details")
    completion_details = (
        completion_details if isinstance(completion_details, Mapping) else {}
    )
    try:
        reasoning_tokens = int(completion_details.get("reasoning_tokens") or 0)
    except (TypeError, ValueError, OverflowError):
        reasoning_tokens = 0
    reasoning_tokens = max(0, min(reasoning_tokens, max(0, output_tokens)))
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {
            "cached_tokens": cached_tokens,
            "cache_write_tokens": 0,
        },
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": input_tokens + output_tokens,
    }


def _custom_input(arguments: Any) -> str:
    text = str(arguments or "")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(parsed, Mapping) and isinstance(parsed.get("input"), str):
        return parsed["input"]
    return text


def _function_identity(
    chat_name: str,
    namespace_functions: Mapping[str, NamespaceFunction],
) -> tuple[str, str | None]:
    identity = namespace_functions.get(chat_name)
    if identity is None:
        return chat_name, None
    return identity.name, identity.namespace


def output_from_chat(
    chat_payload: Mapping[str, Any],
    *,
    custom_tool_names: set[str] | None = None,
    namespace_functions: Mapping[str, NamespaceFunction] | None = None,
) -> list[dict[str, Any]]:
    choice = (chat_payload.get("choices") or [{}])[0]
    message = choice.get("message") if isinstance(choice, Mapping) else {}
    message = message if isinstance(message, Mapping) else {}
    output: list[dict[str, Any]] = []
    custom_tool_names = custom_tool_names or set()
    namespace_functions = namespace_functions or {}
    item_status = (
        "incomplete" if choice.get("finish_reason") == "length" else "completed"
    )
    text = message.get("content")
    if text is not None:
        output.append(
            {
                "id": "msg_" + uuid.uuid4().hex,
                "type": "message",
                "status": item_status,
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": str(text),
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        )
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, Mapping):
            continue
        function = tool_call.get("function")
        if not isinstance(function, Mapping):
            continue
        chat_name = str(function.get("name") or "").strip()
        if not chat_name:
            continue
        name, namespace = _function_identity(chat_name, namespace_functions)
        if chat_name in custom_tool_names:
            item = {
                "id": "ctc_" + uuid.uuid4().hex,
                "type": "custom_tool_call",
                "call_id": str(tool_call.get("id") or "call_" + uuid.uuid4().hex),
                "name": name,
                "input": _custom_input(function.get("arguments")),
            }
            if namespace is not None:
                item["namespace"] = namespace
            output.append(item)
            continue
        item = {
            "id": "fc_" + uuid.uuid4().hex,
            "type": "function_call",
            "status": item_status,
            "call_id": str(tool_call.get("id") or "call_" + uuid.uuid4().hex),
            "name": name,
            "arguments": str(function.get("arguments") or ""),
        }
        if namespace is not None:
            item["namespace"] = namespace
        output.append(item)
    return output


def payload_from_chat(
    chat_payload: Mapping[str, Any],
    *,
    response_id: str,
    created_at: int | None = None,
    status: str | None = None,
    output: list[dict[str, Any]] | None = None,
    response_fields: Mapping[str, Any] | None = None,
    custom_tool_names: set[str] | None = None,
    namespace_functions: Mapping[str, NamespaceFunction] | None = None,
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    choice = (chat_payload.get("choices") or [{}])[0]
    finish_reason = choice.get("finish_reason") if isinstance(choice, Mapping) else None
    final_status = status or (
        "incomplete" if finish_reason == "length" else "completed"
    )
    fields = {"tool_choice": "auto", "tools": [], **dict(response_fields or {})}
    payload: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": int(
            created_at
            if created_at is not None
            else (chat_payload.get("created") or time.time())
        ),
        "status": final_status,
        "error": dict(error) if error is not None else None,
        "incomplete_details": (
            {"reason": "max_output_tokens"} if final_status == "incomplete" else None
        ),
        "model": chat_payload.get("model"),
        "output": (
            output_from_chat(
                chat_payload,
                custom_tool_names=custom_tool_names,
                namespace_functions=namespace_functions,
            )
            if output is None
            else output
        ),
        "parallel_tool_calls": bool(fields.pop("parallel_tool_calls", True)),
        "usage": _usage(chat_payload.get("usage")),
        **fields,
    }
    if final_status == "completed":
        payload["completed_at"] = time.time()
    return payload


def _sse(event: str, payload: Mapping[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


async def _iter_sse_data(body_iterator: Any) -> AsyncIterator[str]:
    pending = ""
    async for raw in body_iterator:
        pending += raw.decode() if isinstance(raw, bytes) else str(raw)
        while "\n\n" in pending:
            frame, pending = pending.split("\n\n", 1)
            data_lines = [
                line[6:] for line in frame.splitlines() if line.startswith("data: ")
            ]
            if data_lines:
                yield "\n".join(data_lines)
    if pending.strip():
        data_lines = [
            line[6:] for line in pending.splitlines() if line.startswith("data: ")
        ]
        if data_lines:
            yield "\n".join(data_lines)


async def stream_from_chat_sse(
    body_iterator: Any,
    *,
    response_id: str,
    model: str,
    response_fields: Mapping[str, Any],
    custom_tool_names: set[str],
    namespace_functions: Mapping[str, NamespaceFunction],
) -> AsyncIterator[str]:
    created = int(time.time())
    output: list[dict[str, Any]] = []
    text_item: dict[str, Any] | None = None
    text_chunks: list[str] = []
    tool_items: dict[int, dict[str, Any]] = {}
    final_chat: dict[str, Any] | None = None
    sequence_number = 0

    def event_payload(event_type: str, **payload: Any) -> dict[str, Any]:
        nonlocal sequence_number
        sequence_number += 1
        return {"type": event_type, "sequence_number": sequence_number, **payload}

    def base(status: str) -> dict[str, Any]:
        fields = dict(response_fields)
        return {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "status": status,
            "error": None,
            "incomplete_details": None,
            "model": model,
            "output": output,
            "parallel_tool_calls": bool(fields.pop("parallel_tool_calls", True)),
            "usage": None,
            **fields,
        }

    yield _sse(
        "response.created",
        event_payload("response.created", response=base("in_progress")),
    )
    yield _sse(
        "response.in_progress",
        event_payload("response.in_progress", response=base("in_progress")),
    )
    try:
        async for data in _iter_sse_data(body_iterator):
            if data == "[DONE]":
                break
            try:
                chat_chunk = json.loads(data)
            except json.JSONDecodeError as exc:
                message = f"failed to parse MTPLX SSE: {exc}"
                yield _sse(
                    "error",
                    event_payload(
                        "error", code="server_error", message=message, param=None
                    ),
                )
                failed = base("failed")
                failed["error"] = {"code": "server_error", "message": message}
                yield _sse(
                    "response.failed",
                    event_payload("response.failed", response=failed),
                )
                return
            if "error" in chat_chunk:
                raw_error = chat_chunk.get("error") or {}
                if not isinstance(raw_error, Mapping):
                    raw_error = {"message": str(raw_error)}
                message = str(raw_error.get("message") or raw_error)
                code = str(raw_error.get("code") or "server_error")
                yield _sse(
                    "error",
                    event_payload(
                        "error",
                        code=code,
                        message=message,
                        param=raw_error.get("param"),
                    ),
                )
                failed = base("failed")
                failed["error"] = {"code": "server_error", "message": message}
                yield _sse(
                    "response.failed",
                    event_payload("response.failed", response=failed),
                )
                return
            final_chat = chat_chunk
            choice = (chat_chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") if isinstance(choice, Mapping) else {}
            delta = delta if isinstance(delta, Mapping) else {}
            text = delta.get("content")
            if isinstance(text, str) and text:
                if text_item is None:
                    text_item = {
                        "id": "msg_" + uuid.uuid4().hex,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    }
                    output.append(text_item)
                    output_index = len(output) - 1
                    yield _sse(
                        "response.output_item.added",
                        event_payload(
                            "response.output_item.added",
                            output_index=output_index,
                            item=text_item,
                        ),
                    )
                    part = {"type": "output_text", "text": "", "annotations": []}
                    text_item["content"].append(part)
                    yield _sse(
                        "response.content_part.added",
                        event_payload(
                            "response.content_part.added",
                            item_id=text_item["id"],
                            output_index=output_index,
                            content_index=0,
                            part=part,
                        ),
                    )
                text_chunks.append(text)
                text_item["content"][0]["text"] = "".join(text_chunks)
                yield _sse(
                    "response.output_text.delta",
                    event_payload(
                        "response.output_text.delta",
                        item_id=text_item["id"],
                        output_index=output.index(text_item),
                        content_index=0,
                        delta=text,
                        logprobs=[],
                    ),
                )
            for raw_tool in delta.get("tool_calls") or []:
                if not isinstance(raw_tool, Mapping):
                    continue
                index = int(raw_tool.get("index") or 0)
                item = tool_items.setdefault(
                    index,
                    {
                        "id": None,
                        "call_id": None,
                        "chat_name": None,
                        "name": None,
                        "namespace": None,
                        "arguments": "",
                        "raw_arguments": "",
                        "custom": False,
                        "output_index": None,
                    },
                )
                if raw_tool.get("id"):
                    item["call_id"] = str(raw_tool["id"])
                function = raw_tool.get("function")
                if isinstance(function, Mapping):
                    if function.get("name"):
                        item["chat_name"] = str(function["name"])
                        item["name"], item["namespace"] = _function_identity(
                            item["chat_name"], namespace_functions
                        )
                    arguments = function.get("arguments")
                    if arguments is not None:
                        argument_text = str(arguments)
                        item["arguments"] += argument_text
                        item["raw_arguments"] += argument_text
                if item["output_index"] is None and item["name"]:
                    item["custom"] = item["chat_name"] in custom_tool_names
                    item["id"] = (
                        "ctc_" if item["custom"] else "fc_"
                    ) + uuid.uuid4().hex
                    item["call_id"] = item["call_id"] or "call_" + uuid.uuid4().hex
                    item["output_index"] = len(output)
                    if item["custom"]:
                        output_item = {
                            "id": item["id"],
                            "type": "custom_tool_call",
                            "call_id": item["call_id"],
                            "name": item["name"],
                            "input": "",
                        }
                    else:
                        output_item = {
                            "id": item["id"],
                            "type": "function_call",
                            "status": "in_progress",
                            "call_id": item["call_id"],
                            "name": item["name"],
                            "arguments": "",
                        }
                    if item["namespace"] is not None:
                        output_item["namespace"] = item["namespace"]
                    output.append(output_item)
                    yield _sse(
                        "response.output_item.added",
                        event_payload(
                            "response.output_item.added",
                            output_index=item["output_index"],
                            item=output[-1],
                        ),
                    )
                if (
                    item["output_index"] is not None
                    and item["arguments"]
                    and not item["custom"]
                ):
                    output[item["output_index"]]["arguments"] += item["arguments"]
                    yield _sse(
                        "response.function_call_arguments.delta",
                        event_payload(
                            "response.function_call_arguments.delta",
                            item_id=item["id"],
                            output_index=item["output_index"],
                            delta=item["arguments"],
                        ),
                    )
                    item["arguments"] = ""
                elif item["custom"]:
                    item["arguments"] = ""
    finally:
        if hasattr(body_iterator, "aclose"):
            with suppress(Exception):
                await body_iterator.aclose()

    if final_chat is None:
        message = "MTPLX stream ended without a terminal chat chunk"
        yield _sse(
            "error",
            event_payload("error", code="server_error", message=message, param=None),
        )
        failed = base("failed")
        failed["error"] = {"code": "server_error", "message": message}
        yield _sse("response.failed", event_payload("response.failed", response=failed))
        return

    final_choice = (final_chat.get("choices") or [{}])[0]
    finish_reason = (
        final_choice.get("finish_reason") if isinstance(final_choice, Mapping) else None
    )
    terminal_status = "incomplete" if finish_reason == "length" else "completed"
    if text_item is not None:
        output_index = output.index(text_item)
        yield _sse(
            "response.output_text.done",
            event_payload(
                "response.output_text.done",
                item_id=text_item["id"],
                output_index=output_index,
                content_index=0,
                text=text_item["content"][0]["text"],
                logprobs=[],
            ),
        )
        yield _sse(
            "response.content_part.done",
            event_payload(
                "response.content_part.done",
                item_id=text_item["id"],
                output_index=output_index,
                content_index=0,
                part=text_item["content"][0],
            ),
        )
        text_item["status"] = terminal_status
        yield _sse(
            "response.output_item.done",
            event_payload(
                "response.output_item.done", output_index=output_index, item=text_item
            ),
        )
    for item in tool_items.values():
        output_index = item.get("output_index")
        if output_index is None:
            continue
        output_item = output[int(output_index)]
        if item["custom"]:
            custom_input = _custom_input(item["raw_arguments"])
            if custom_input:
                yield _sse(
                    "response.custom_tool_call_input.delta",
                    event_payload(
                        "response.custom_tool_call_input.delta",
                        item_id=item["id"],
                        output_index=output_index,
                        delta=custom_input,
                    ),
                )
            output_item["input"] = custom_input
            yield _sse(
                "response.custom_tool_call_input.done",
                event_payload(
                    "response.custom_tool_call_input.done",
                    item_id=item["id"],
                    output_index=output_index,
                    input=custom_input,
                ),
            )
            yield _sse(
                "response.output_item.done",
                event_payload(
                    "response.output_item.done",
                    output_index=output_index,
                    item=output_item,
                ),
            )
            continue
        yield _sse(
            "response.function_call_arguments.done",
            event_payload(
                "response.function_call_arguments.done",
                item_id=item["id"],
                output_index=output_index,
                name=item["name"],
                arguments=output_item["arguments"],
            ),
        )
        output_item["status"] = terminal_status
        yield _sse(
            "response.output_item.done",
            event_payload(
                "response.output_item.done",
                output_index=output_index,
                item=output_item,
            ),
        )
    final = payload_from_chat(
        final_chat,
        response_id=response_id,
        created_at=created,
        output=output,
        response_fields=response_fields,
        custom_tool_names=custom_tool_names,
        namespace_functions=namespace_functions,
    )
    terminal_event = (
        "response.incomplete"
        if final["status"] == "incomplete"
        else "response.completed"
    )
    yield _sse(terminal_event, event_payload(terminal_event, response=final))
