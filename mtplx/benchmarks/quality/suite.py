"""The fixed tasks: what the model is asked, and what counts as done.

A suite is a folder of TOML files, one task each (``mtplx/benchmarks/prompts/
quality/`` is the default). Five kinds share one shape, a conversation the
model must finish:

- ``code``: write a function or class; hidden tests run against the code it
  returns.
- ``agent``: a small project with failing tests; the model works through
  tools (list, read and write files, run the tests) until it says it is done;
  hidden tests then run against the files it left.
- ``tool``: tools are declared but not run; the model's first reply must call
  the right ones with the right arguments.
- ``answer``: a question with one checkable answer.
- ``image``: the same, about a picture drawn here from the task's shapes.

Every task carries what the model never sees (tests, expected calls,
answers) and, for code and agent tasks, a reference solution that the suite's
own unit tests use to prove the checks accept a correct answer and refuse the
starting state.
"""

from __future__ import annotations

import json
import struct
import tomllib
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KINDS = ("code", "agent", "tool", "answer", "image")
DEFAULT_SUITE_DIR = Path(__file__).resolve().parent.parent / "prompts" / "quality"
SUITE_FILE = "suite.toml"  # suite-wide settings, beside the task files


class SuiteError(ValueError):
    pass


@dataclass(frozen=True)
class Task:
    id: str
    kind: str
    prompt: str
    system: str | None = None
    tests: str = ""  # code: appended after the model's code
    reference: str = ""  # code: a correct solution (suite tests only)
    files: dict[str, str] = field(default_factory=dict)  # agent: the project
    hidden_tests: str = ""  # agent: run against the final files
    reference_files: dict[str, str] = field(default_factory=dict)  # agent
    tools: list[dict[str, Any]] = field(default_factory=list)  # tool: declared
    expect_calls: list[dict[str, Any]] = field(default_factory=list)  # tool
    expect: dict[str, Any] = field(default_factory=dict)  # answer, image
    image: dict[str, Any] = field(default_factory=dict)  # image: the drawing


@dataclass(frozen=True)
class Suite:
    name: str
    budget_tokens: int  # per request; a reply that reaches it is a runaway
    max_agent_turns: int  # requests per agent task before it counts unfinished
    tasks: tuple[Task, ...]

    def select(self, ids: list[str] | None = None, kinds: list[str] | None = None) -> list[Task]:
        chosen = [t for t in self.tasks if (not ids or t.id in ids) and (not kinds or t.kind in kinds)]
        if ids:
            missing = sorted(set(ids) - {t.id for t in chosen})
            if missing:
                raise SuiteError(f"unknown task ids: {', '.join(missing)}")
        return chosen


def _task_from(data: dict[str, Any], source: Path) -> Task:
    kind = data.get("kind")
    if kind not in KINDS:
        raise SuiteError(f"{source.name}: kind must be one of {KINDS}, got {kind!r}")
    if not data.get("id") or not data.get("prompt"):
        raise SuiteError(f"{source.name}: a task needs an id and a prompt")
    required = {
        "code": ("tests", "reference"),
        "agent": ("files", "hidden_tests", "reference_files"),
        "tool": ("tools_json", "expect_calls"),
        "answer": ("expect",),
        "image": ("expect", "image"),
    }[kind]
    absent = [key for key in required if not data.get(key)]
    if absent:
        raise SuiteError(f"{source.name}: a {kind} task needs {', '.join(absent)}")
    tools = json.loads(data["tools_json"]) if data.get("tools_json") else []
    return Task(
        id=str(data["id"]),
        kind=kind,
        prompt=str(data["prompt"]),
        system=data.get("system"),
        tests=str(data.get("tests", "")),
        reference=str(data.get("reference", "")),
        files=dict(data.get("files", {})),
        hidden_tests=str(data.get("hidden_tests", "")),
        reference_files=dict(data.get("reference_files", {})),
        tools=tools,
        expect_calls=list(data.get("expect_calls", [])),
        expect=dict(data.get("expect", {})),
        image=dict(data.get("image", {})),
    )


def load_suite(folder: Path | str = DEFAULT_SUITE_DIR) -> Suite:
    folder = Path(folder)
    settings = tomllib.loads((folder / SUITE_FILE).read_text())
    tasks = []
    for path in sorted(folder.glob("*.toml")):
        if path.name == SUITE_FILE:
            continue
        tasks.append(_task_from(tomllib.loads(path.read_text()), path))
    ids = [t.id for t in tasks]
    if len(set(ids)) != len(ids):
        raise SuiteError("task ids must be unique")
    if not tasks:
        raise SuiteError(f"no tasks in {folder}")
    return Suite(
        name=str(settings["name"]),
        budget_tokens=int(settings["budget_tokens"]),
        max_agent_turns=int(settings["max_agent_turns"]),
        tasks=tuple(tasks),
    )


# --------------------------------------------------------------------------
# images: drawn from the task's shapes, so the expected answer is known
# --------------------------------------------------------------------------

COLORS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (220, 30, 30),
    "green": (30, 160, 60),
    "blue": (30, 70, 210),
    "yellow": (240, 200, 20),
}


def render_image(spec: dict[str, Any]) -> bytes:
    """A PNG of filled rectangles on a plain background (no imaging library)."""

    width, height = int(spec["width"]), int(spec["height"])
    background = COLORS[spec.get("background", "white")]
    pixels = [bytearray(bytes(background) * width) for _ in range(height)]
    for rect in spec.get("rects", []):
        color = bytes(COLORS[rect["color"]])
        x0, y0 = int(rect["x"]), int(rect["y"])
        x1, y1 = min(width, x0 + int(rect["w"])), min(height, y0 + int(rect["h"]))
        for y in range(max(0, y0), y1):
            pixels[y][3 * max(0, x0) : 3 * x1] = color * (x1 - max(0, x0))
    raw = b"".join(b"\x00" + bytes(row) for row in pixels)

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
