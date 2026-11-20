"""Whether a finished conversation solved its task.

Each kind has one check, applied to what a user would see: the answer text
(never the thinking), the tool calls of the reply, or the project files the
agent left behind. A verdict carries a short status so a regression can be
told apart from a format slip (``no_code``, ``syntax_error``) when two packs
are compared.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mtplx.benchmarks.quality.sandbox import Workspace, run_python
from mtplx.benchmarks.quality.suite import Task

_FENCE = re.compile(r"```[ \t]*(?:python|py)?[ \t]*\n(.*?)(?:\n```|\Z)", re.DOTALL)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_HIDDEN_CHECK = ".hidden_check.py"


@dataclass(frozen=True)
class Verdict:
    passed: bool
    status: str
    detail: str = ""


def extract_code(content: str) -> str | None:
    """The fenced block that defines something, else the first fenced block."""

    blocks = [match.group(1) for match in _FENCE.finditer(content)]
    for block in blocks:
        if re.search(r"^\s*(def|class)\s", block, re.MULTILINE):
            return block
    return blocks[0] if blocks else None


def _run_check(workdir: Path, name: str, source: str) -> Verdict:
    (workdir / name).write_text(source)
    result = run_python(workdir, [name])
    if result.status == "ok":
        return Verdict(True, "passed")
    tail = (result.stderr.strip().splitlines() or [f"exit {result.returncode}"])[-1]
    if result.status == "timeout":
        return Verdict(False, "timeout", tail)
    if re.search(r"^(SyntaxError|IndentationError|TabError)", result.stderr, re.MULTILINE):
        return Verdict(False, "syntax_error", tail)
    return Verdict(False, "failed", tail[:300])


def grade_code(task: Task, content: str, workdir: Path) -> Verdict:
    code = extract_code(content)
    if code is None:
        return Verdict(False, "no_code", "no fenced code block in the answer")
    workdir.mkdir(parents=True, exist_ok=True)
    return _run_check(workdir, "candidate.py", f"{code}\n\n{task.tests}")


def grade_agent(task: Task, workspace: Workspace) -> Verdict:
    return _run_check(workspace.root, _HIDDEN_CHECK, task.hidden_tests)


# --------------------------------------------------------------------------
# tool calls
# --------------------------------------------------------------------------


def _normal(value: Any) -> Any:
    return value.strip().lower() if isinstance(value, str) else value


def _matches(rule: dict[str, Any], value: Any) -> bool:
    if "equals" in rule:
        want = rule["equals"]
        if isinstance(want, (int, float)) and not isinstance(want, bool):
            try:
                return float(value) == float(want)
            except (TypeError, ValueError):
                return False
        return _normal(value) == _normal(want)
    if "contains" in rule:
        return isinstance(value, str) and _normal(rule["contains"]) in _normal(value)
    if "one_of" in rule:
        return _normal(value) in {_normal(v) for v in rule["one_of"]}
    if "set_equals" in rule:
        return isinstance(value, list) and {_normal(v) for v in value} == {
            _normal(v) for v in rule["set_equals"]
        }
    raise ValueError(f"unknown argument rule {rule!r}")


def _call_matches(expected: dict[str, Any], call: dict[str, Any]) -> bool:
    if call.get("name") != expected["name"]:
        return False
    args = call.get("arguments") or {}
    return all(key in args and _matches(rule, args[key]) for key, rule in expected.get("args", {}).items())


def parse_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """``[{name, arguments}]`` from an OpenAI assistant message."""

    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        raw = function.get("arguments")
        try:
            arguments = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        except (TypeError, ValueError):
            arguments = {"__unparsed__": raw}
        calls.append({"name": function.get("name"), "arguments": arguments})
    return calls


def grade_tool(task: Task, message: dict[str, Any]) -> Verdict:
    """Every expected call made by a distinct call, and nothing else called."""

    calls = parse_tool_calls(message)
    if not calls:
        return Verdict(False, "no_call", "the reply called no tool")
    unmatched = list(range(len(calls)))
    for expected in task.expect_calls:
        hit = next((i for i in unmatched if _call_matches(expected, calls[i])), None)
        if hit is None:
            return Verdict(False, "wrong_call", json.dumps(calls)[:300])
        unmatched.remove(hit)
    stray = [calls[i] for i in unmatched if not any(_call_matches(e, calls[i]) for e in task.expect_calls)]
    if stray:
        return Verdict(False, "extra_call", json.dumps(stray)[:300])
    return Verdict(True, "passed")


# --------------------------------------------------------------------------
# answers
# --------------------------------------------------------------------------


def _last_line(content: str) -> str:
    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def grade_answer(task: Task, content: str) -> Verdict:
    if not content.strip():
        return Verdict(False, "no_answer", "the answer is empty")
    expect = task.expect
    kind = expect["type"]
    if kind == "number":
        # The task asks for the number on the last line; the whole answer is
        # the fallback when that line holds none.
        numbers = _NUMBER.findall(_last_line(content).replace(",", "")) or _NUMBER.findall(
            content.replace(",", "")
        )
        if not numbers:
            return Verdict(False, "no_answer", "no number in the answer")
        passed = float(numbers[-1]) == float(expect["value"])
        return Verdict(passed, "passed" if passed else "wrong_answer", numbers[-1])
    if kind == "last_line":
        line = _last_line(content).strip("`*. ")
        passed = re.sub(r"\s", "", str(expect["value"])) in re.sub(r"\s", "", line)
        return Verdict(passed, "passed" if passed else "wrong_answer", line[:200])
    if kind == "sequence":
        vocabulary = [w.lower() for w in expect["words"]]
        pattern = re.compile(r"\b(" + "|".join(map(re.escape, vocabulary)) + r")\b", re.IGNORECASE)
        lines = [line for line in content.strip().splitlines() if pattern.search(line)]
        found = [w.lower() for w in pattern.findall(lines[-1])] if lines else []
        passed = found == [w.lower() for w in expect["value"]]
        return Verdict(passed, "passed" if passed else "wrong_answer", ", ".join(found))
    raise ValueError(f"unknown answer type {kind!r}")
