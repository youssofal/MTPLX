"""The suite's own proof: every check accepts a correct answer and refuses the
starting state, so a task can only be failed by the model, never by its test."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from mtplx.benchmarks.quality import grading
from mtplx.benchmarks.quality.sandbox import Workspace, run_python
from mtplx.benchmarks.quality.suite import KINDS, load_suite, render_image

SUITE = load_suite()
CODE = [t for t in SUITE.tasks if t.kind == "code"]
AGENT = [t for t in SUITE.tasks if t.kind == "agent"]


def test_suite_shape():
    assert SUITE.name == "quality-v1"
    assert len(SUITE.tasks) == 20
    assert {t.kind for t in SUITE.tasks} == set(KINDS)
    assert SUITE.budget_tokens > 4096 and SUITE.max_agent_turns >= 8


@pytest.mark.parametrize("task", CODE, ids=lambda t: t.id)
def test_code_reference_passes_and_nothing_else_does(task, tmp_path):
    fenced = f"Here it is.\n\n```python\n{task.reference}\n```\n"
    assert grading.grade_code(task, fenced, tmp_path / "ref") == grading.Verdict(True, "passed")
    wrong = grading.grade_code(task, "```python\nx = 1\n```", tmp_path / "wrong")
    assert not wrong.passed and wrong.status == "failed"
    assert grading.grade_code(task, "I would write a loop.", tmp_path / "none").status == "no_code"


@pytest.mark.parametrize("task", AGENT, ids=lambda t: t.id)
def test_agent_start_fails_and_reference_passes(task, tmp_path):
    start = Workspace(tmp_path / "start", task.files)
    assert not grading.grade_agent(task, start).passed
    fixed = Workspace(tmp_path / "fixed", {**task.files, **task.reference_files})
    assert grading.grade_agent(task, fixed) == grading.Verdict(True, "passed")
    # The project's own tests, as the agent's run_tests tool runs them.
    assert run_python(fixed.root, ["-m", "unittest", "-q"], timeout_s=60).status == "ok"


def _message(*calls):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": args}}
            for i, (name, args) in enumerate(calls)
        ],
    }


def test_tool_grading():
    calendar = next(t for t in SUITE.tasks if t.id == "tool-calendar")
    good = _message(("create_event", '{"title": "Design Review", "date": "2026-12-03", "start_time": "14:30", '
                                     '"duration_minutes": 45, "attendees": ["li@example.com", "ana@example.com"]}'))
    assert grading.grade_tool(calendar, good).passed
    late = _message(("create_event", '{"title": "Design review", "date": "2026-12-03", "start_time": "02:30", '
                                     '"duration_minutes": 45, "attendees": ["li@example.com", "ana@example.com"]}'))
    assert grading.grade_tool(calendar, late).status == "wrong_call"
    assert grading.grade_tool(calendar, {"content": "Done."}).status == "no_call"

    currency = next(t for t in SUITE.tasks if t.id == "tool-currency")
    eur = ("convert_currency", '{"amount": 250, "from_currency": "USD", "to_currency": "EUR"}')
    jpy = ("convert_currency", '{"amount": "250", "from_currency": "usd", "to_currency": "JPY"}')
    assert grading.grade_tool(currency, _message(jpy, eur)).passed
    assert grading.grade_tool(currency, _message(eur)).status == "wrong_call"
    gbp = ("convert_currency", '{"amount": 250, "from_currency": "USD", "to_currency": "GBP"}')
    assert grading.grade_tool(currency, _message(eur, jpy, gbp)).status == "extra_call"
    assert grading.grade_tool(currency, _message(eur, jpy, eur)).passed  # a repeat is not a wrong call


def test_answer_grading():
    count = next(t for t in SUITE.tasks if t.id == "answer-divisible-count")
    assert grading.grade_answer(count, "333 + 200 - 66 = 467, minus 66 gives\n\n**401**").passed
    assert grading.grade_answer(count, "So 401 numbers... wait, 467.\n467").status == "wrong_answer"
    assert grading.grade_answer(count, "").status == "no_answer"
    late = next(t for t in SUITE.tasks if t.id == "answer-late-binding")
    assert grading.grade_answer(late, "Each lambda sees the final i.\n\n`[4,4,4]`").passed
    assert not grading.grade_answer(late, "[0, 2, 4]").passed
    stripes = next(t for t in SUITE.tasks if t.id == "image-stripes")
    assert grading.grade_answer(stripes, "From left to right:\nGreen, blue, red").passed
    assert not grading.grade_answer(stripes, "Red, blue, green").passed


def _pixel(png: bytes, x: int, y: int) -> tuple[int, int, int]:
    width, height = struct.unpack(">II", png[16:24])
    data = zlib.decompress(png[png.index(b"IDAT") + 4 : png.index(b"IEND") - 8])
    stride = 1 + 3 * width
    offset = y * stride + 1 + 3 * x
    return tuple(data[offset : offset + 3])


def test_images_are_drawn_as_specified():
    squares = next(t for t in SUITE.tasks if t.id == "image-count-squares")
    png = render_image(squares.image)
    assert png.startswith(b"\x89PNG\r\n\x1a\n") and struct.unpack(">II", png[16:24]) == (320, 240)
    assert _pixel(png, 30, 30) == (220, 30, 30)  # inside the first red square
    assert _pixel(png, 110, 40) == (30, 70, 210)  # inside a blue one
    assert _pixel(png, 5, 5) == (255, 255, 255)
    assert sum(r["color"] == "red" for r in squares.image["rects"]) == squares.expect["value"]


@pytest.mark.skipif(not Path("/usr/bin/sandbox-exec").exists(), reason="macOS sandbox only")
def test_sandbox_confines_model_code(tmp_path):
    outside = tmp_path / "outside.txt"
    work = tmp_path / "work"
    work.mkdir()
    (work / "escape.py").write_text(f"open({str(outside)!r}, 'w').write('x')\n")
    assert run_python(work, ["escape.py"]).status == "failed" and not outside.exists()
    (work / "spawn.py").write_text("import subprocess\nsubprocess.run(['/bin/echo', 'hi'], check=True)\n")
    assert run_python(work, ["spawn.py"]).status == "failed"
    (work / "inside.py").write_text("open('here.txt', 'w').write('x')\n")
    assert run_python(work, ["inside.py"]).status == "ok" and (work / "here.txt").exists()
    (work / "net.py").write_text("import socket\nsocket.create_connection(('127.0.0.1', 9), timeout=2)\n")
    assert run_python(work, ["net.py"]).status == "failed"
    (work / "fork.py").write_text("import os\nos.fork()\n")
    assert run_python(work, ["fork.py"]).status == "failed"
    if str(Path(__file__).resolve()).startswith(("/Users/", "/Volumes/")):
        (work / "peek.py").write_text(f"print(open({str(Path(__file__).resolve())!r}).read())\n")
        assert run_python(work, ["peek.py"]).status == "failed"


def test_sandbox_timeout_kills(tmp_path):
    (tmp_path / "spin.py").write_text("while True:\n    pass\n")
    result = run_python(tmp_path, ["spin.py"], timeout_s=2)
    assert result.status == "timeout" and result.seconds < 10


def test_a_program_that_prints_forever_is_stopped_and_its_output_bounded(tmp_path):
    from mtplx.benchmarks.quality.sandbox import OUTPUT_LIMIT

    (tmp_path / "flood.py").write_text("import sys\nwhile True:\n    sys.stdout.write('x' * 1_000_000)\n")
    result = run_python(tmp_path, ["flood.py"], timeout_s=20)
    assert result.status == "failed" and result.seconds < 15
    assert 0 < len(result.stdout) <= OUTPUT_LIMIT
    assert not (tmp_path / ".stdout").exists()


def test_workspace_refuses_paths_outside(tmp_path):
    space = Workspace(tmp_path / "p", {"a.py": "x = 1\n"})
    for bad in ["../x.py", "/etc/hosts", "sub/../../x"]:
        with pytest.raises(ValueError):
            space.write(bad, "")
    space.write("pkg/b.py", "y = 2\n")
    assert space.names() == ["a.py", "pkg/b.py"]
