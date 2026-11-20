"""One attempt at one task: the conversation, the grade, and its row.

The row records what the founder's questions need: did it pass, how many
tokens did it take (thinking and answer apart, summed over every request of
the task), how long did it take, and how the draft head fared by step. It
also keeps the finished conversation, which is the text the divergence
corpus is built from.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

from mtplx.benchmarks.quality.client import Server, ServerError
from mtplx.benchmarks.quality.grading import (
    Verdict,
    grade_agent,
    grade_answer,
    grade_code,
    grade_tool,
    parse_tool_calls,
)
from mtplx.benchmarks.quality.sandbox import Workspace, run_python
from mtplx.benchmarks.quality.suite import Suite, Task, render_image

ROW_SCHEMA = "mtplx.quality.task/1"
AGENT_SYSTEM = (
    "You are a coding agent working in a small Python project. Use the tools to list, read and "
    "write files and to run the project's tests. When the task is done, reply with a short "
    "summary of what you changed and make no further tool call."
)
TOOL_OUTPUT_LIMIT = 6_000  # characters of a tool result shown to the model
_STAT_KEYS = (
    "mtp_depth",
    "verify_calls",
    "accepted_by_depth",
    "drafted_by_depth",
    "decode_tok_s",
    "ttft_s",
    "prompt_tps",
    "effective_temperature",
    "effective_top_p",
    "effective_top_k",
    "repetition_stop_triggered",
    # The route each request took: a pack whose bits a fast lane refuses shows
    # here, not as an unexplained speed gap.
    "demotions",
    "compiled_verify",
)


def _function(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


AGENT_TOOLS = [
    _function("list_files", "List the files in the project.", {}, []),
    _function("read_file", "Read a file of the project.", {"path": {"type": "string"}}, ["path"]),
    _function(
        "write_file",
        "Create or replace a file of the project with the given content.",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        ["path", "content"],
    ),
    _function("run_tests", "Run the project's tests with unittest and return their output.", {}, []),
]


def _agent_tool(workspace: Workspace, call: dict[str, Any]) -> str:
    args = call.get("arguments") or {}
    try:
        if call["name"] == "list_files":
            return "\n".join(workspace.names())
        if call["name"] == "read_file":
            return workspace.read(str(args["path"]))
        if call["name"] == "write_file":
            workspace.write(str(args["path"]), str(args["content"]))
            return f"wrote {args['path']}"
        if call["name"] == "run_tests":
            result = run_python(workspace.root, ["-m", "unittest", "-q"], timeout_s=60.0)
            output = (result.stdout + result.stderr).strip()
            return f"exit status {result.returncode} ({result.status})\n{output}"[-TOOL_OUTPUT_LIMIT:]
        return f"error: there is no tool named {call['name']!r}"
    except (KeyError, ValueError, OSError) as error:
        return f"error: {type(error).__name__}: {error}"


def _user_message(task: Task) -> dict[str, Any]:
    if task.kind != "image":
        return {"role": "user", "content": task.prompt.strip()}
    picture = base64.b64encode(render_image(task.image)).decode()
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{picture}"}},
            {"type": "text", "text": task.prompt.strip()},
        ],
    }


def _turn_record(response: dict[str, Any], wall_s: float) -> dict[str, Any]:
    usage = response.get("usage") or {}
    stats = response.get("mtplx_stats") or {}
    choice = (response.get("choices") or [{}])[0]
    completion = int(usage.get("completion_tokens") or 0)
    reasoning = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
    record = {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "cached_tokens": int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "answer_tokens": completion - reasoning,
        "finish_reason": choice.get("finish_reason"),
        "tool_calls": len((choice.get("message") or {}).get("tool_calls") or []),
        "wall_s": round(wall_s, 3),
    }
    record.update({key: stats[key] for key in _STAT_KEYS if key in stats})
    return record


def _for_history(message: dict[str, Any]) -> dict[str, Any]:
    """The assistant message as a client sends it back, thinking included."""

    kept = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("reasoning_content"):
        kept["reasoning_content"] = message["reasoning_content"]
    if message.get("tool_calls"):
        kept["tool_calls"] = message["tool_calls"]
    return kept


def _without_pictures(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for message in messages:
        if isinstance(message.get("content"), list):
            parts = [
                {"type": "image_url", "image_url": {"url": "(task image)"}} if p.get("type") == "image_url" else p
                for p in message["content"]
            ]
            message = {**message, "content": parts}
        out.append(message)
    return out


def run_episode(
    server: Server,
    suite: Suite,
    task: Task,
    *,
    seed: int,
    scratch: Path,
    request_extras: dict[str, Any],
) -> dict[str, Any]:
    """Run ``task`` to its end on ``server`` and return its row."""

    messages: list[dict[str, Any]] = []
    if task.kind == "agent":
        messages.append({"role": "system", "content": AGENT_SYSTEM})
    elif task.system:
        messages.append({"role": "system", "content": task.system})
    messages.append(_user_message(task))
    tools = AGENT_TOOLS if task.kind == "agent" else task.tools
    workspace = Workspace(scratch / f"{task.id}-s{seed}", task.files) if task.kind == "agent" else None

    turns: list[dict[str, Any]] = []
    outcome, error, last = "finished", None, {}
    started = time.monotonic()
    request_limit = suite.max_agent_turns if task.kind == "agent" else 1
    while True:
        body = {"messages": messages, "seed": seed, "max_tokens": suite.budget_tokens, **request_extras}
        if tools:
            body["tools"] = tools
        t0 = time.monotonic()
        try:
            response = server.chat(body)
        except ServerError as failure:
            outcome, error = "error", str(failure)
            break
        turns.append(_turn_record(response, time.monotonic() - t0))
        last = (response.get("choices") or [{}])[0].get("message") or {}
        messages.append(_for_history(last))
        if turns[-1]["finish_reason"] == "length":
            outcome = "runaway"
            break
        calls = parse_tool_calls(last)
        if task.kind != "agent" or not calls:
            break
        if len(turns) >= request_limit:
            outcome = "unfinished"
            break
        for raw, call in zip(last.get("tool_calls") or [], calls):
            messages.append(
                {"role": "tool", "tool_call_id": raw.get("id"), "content": _agent_tool(workspace, call)}
            )

    if outcome != "finished":
        verdict = Verdict(False, outcome, error or "")
    elif task.kind == "agent":
        verdict = grade_agent(task, workspace)
    elif task.kind == "tool":
        verdict = grade_tool(task, last)
    elif task.kind == "code":
        verdict = grade_code(task, last.get("content") or "", scratch / f"{task.id}-s{seed}")
    else:
        verdict = grade_answer(task, last.get("content") or "")

    completion = sum(t["completion_tokens"] for t in turns)
    reasoning = sum(t["reasoning_tokens"] for t in turns)
    return {
        "schema": ROW_SCHEMA,
        "suite": suite.name,
        "task": task.id,
        "kind": task.kind,
        "seed": seed,
        "passed": verdict.passed,
        "status": verdict.status,
        "detail": verdict.detail,
        "outcome": outcome,
        "requests": len(turns),
        "tokens": {"completion": completion, "reasoning": reasoning, "answer": completion - reasoning},
        "wall_s": round(time.monotonic() - started, 3),
        "turns": turns,
        "request_extras": request_extras,
        "tools": tools,
        "messages": _without_pictures(messages),
    }


def append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
