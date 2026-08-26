"""Deterministic, common-vocabulary coding-agent benchmark context."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


PROGRAMMING_ARTIFACT_KINDS = (
    "source",
    "test",
    "config",
    "documentation",
    "diagnostic",
    "review",
)


@dataclass(frozen=True)
class ProgrammingArtifact:
    kind: str
    path: str
    body: str

    def render(self, cycle: int) -> str:
        return (
            f"\n\n## Repository artifact: workspace_{cycle}/{self.path}\n"
            f"Artifact type: {self.kind}.\n```text\n{self.body.rstrip()}\n```\n"
        )


def _artifacts() -> tuple[ProgrammingArtifact, ...]:
    return (
        ProgrammingArtifact(
            "documentation",
            "README.md",
            """# Task Queue
A small Python service accepts jobs, validates input, stores state, and writes
structured logs. Keep public behavior stable and make failures explicit.

Development uses Python 3.11 and pytest. Run the unit tests before changing the
command line interface or the JSON record schema.""",
        ),
        ProgrammingArtifact(
            "source",
            "src/task_queue/models.py",
            """from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class Job:
    job_id: str
    command: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.job_id.strip():
            raise ValueError("job_id must not be empty")
        if not self.command.strip():
            raise ValueError("command must not be empty")""",
        ),
        ProgrammingArtifact(
            "source",
            "src/task_queue/store.py",
            """from collections import OrderedDict

class JobStore:
    def __init__(self, capacity: int = 128) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items = OrderedDict()

    def get(self, key: str):
        value = self._items.pop(key)
        self._items[key] = value
        return value

    def put(self, key: str, value) -> None:
        self._items.pop(key, None)
        self._items[key] = value
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)""",
        ),
        ProgrammingArtifact(
            "test",
            "tests/test_store.py",
            """import pytest
from task_queue.store import JobStore

def test_store_rejects_invalid_capacity():
    with pytest.raises(ValueError, match="positive"):
        JobStore(0)

def test_store_evicts_the_oldest_item():
    store = JobStore(capacity=2)
    store.put("first", 1)
    store.put("second", 2)
    store.put("third", 3)
    with pytest.raises(KeyError):
        store.get("first")""",
        ),
        ProgrammingArtifact(
            "config",
            "pyproject.toml",
            """[project]
name = "task-queue"
requires-python = ">=3.11"
dependencies = []

[project.scripts]
task-queue = "task_queue.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
""",
        ),
        ProgrammingArtifact(
            "diagnostic",
            "logs/failed-run.log",
            """INFO request accepted job_id=demo-17
INFO state loaded records=42 elapsed_ms=3
WARNING retry scheduled attempt=2 delay_ms=50
ERROR state write failed reason=temporary_io_error
INFO request finished status=failed elapsed_ms=61""",
        ),
        ProgrammingArtifact(
            "review",
            "docs/review-notes.md",
            """The patch must preserve insertion order, reject invalid limits,
use atomic file replacement, and add a regression test for duplicate job
identifiers. Avoid a new dependency when the standard library is sufficient.
Keep error messages useful to both command-line users and automated clients.""",
        ),
        ProgrammingArtifact(
            "source",
            "src/task_queue/codec.py",
            """import json
from typing import Any

def encode_record(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))

def decode_record(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("record must be an object")
    return value""",
        ),
        ProgrammingArtifact(
            "test",
            "tests/test_codec.py",
            """from task_queue.codec import decode_record, encode_record

def test_codec_is_deterministic():
    assert encode_record({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert decode_record('{"ok":true}') == {"ok": True}

def test_decoder_rejects_a_list():
    with pytest.raises(ValueError, match="object"):
        decode_record("[]")""",
        ),
        ProgrammingArtifact(
            "documentation",
            "docs/api.md",
            """# Command line contract

The run command reads newline-delimited JSON, validates each object, and prints
a summary. Exit code 0 means success, 2 means invalid input, and 3 means a
storage failure. Standard output contains data; diagnostics use standard error.""",
        ),
        ProgrammingArtifact(
            "config",
            "config/example.json",
            """{
  "capacity": 128,
  "retry_limit": 3,
  "retry_delay_ms": 50,
  "log_level": "INFO",
  "output_path": "var/jobs.jsonl"
}""",
        ),
        ProgrammingArtifact(
            "diagnostic",
            "docs/incident.md",
            """# Interrupted state write

A process interruption between writing data and renaming the temporary file
left stale state. The fix must flush, fsync, and replace the destination without
exposing partial JSON. A failed replacement must leave the old file readable.""",
        ),
        ProgrammingArtifact(
            "review",
            "docs/acceptance.md",
            """Run unit tests, type checks, and the command-line smoke test.
Confirm deterministic output, helpful error messages, no network access, and no
changes to the public schema. Test both a clean run and recovery from invalid
JSON. Record the exact command and result in the pull request.""",
        ),
    )


def build_programming_context(*, minimum_characters: int) -> str:
    """Return at least ``minimum_characters`` of deterministic repository text."""

    if minimum_characters <= 0:
        raise ValueError("minimum_characters must be positive")
    rendered = [
        "You are reviewing a normal Python repository. Read the source, tests, "
        "configuration, documentation, and diagnostics before making a small, "
        "production-safe change. Preserve public behavior and explain errors clearly."
    ]
    total_characters = len(rendered[0])
    artifacts = _artifacts()
    index = 0
    while total_characters < minimum_characters:
        artifact = artifacts[index % len(artifacts)]
        generation = index // len(artifacts)
        section = artifact.render(generation)
        rendered.append(section)
        total_characters += len(section)
        index += 1
    return "".join(rendered)


def build_unique_programming_context() -> str:
    """Return one non-repeating pass over the deterministic Python artifacts."""

    rendered = [
        "You are reviewing a normal Python repository. Read every distinct "
        "source file, test, configuration, document, and diagnostic below. "
        "Then implement the final request without changing unrelated behavior."
    ]
    for artifact in _artifacts():
        rendered.append(artifact.render(0))
    rendered.append(
        "\n\n# Final user request\n"
        "Implement a complete Python 3.11 patch for this repository. Repair "
        "atomic persistence, preserve insertion order, reject duplicate job "
        "identifiers and invalid limits, keep the JSON schema stable, and use "
        "only the standard library. Include focused pytest coverage for the "
        "failure and recovery paths. Return the changed Python files and tests, "
        "with concise comments only where the invariants are not obvious."
    )
    return "".join(rendered)


def programming_context_stats(text: str) -> dict[str, object]:
    """Return structural diagnostics used to qualify generated prompt context."""

    paths = [
        line.split(": ", 1)[1]
        for line in text.splitlines()
        if line.startswith("## Repository artifact: ")
    ]
    kinds = [
        kind
        for kind in PROGRAMMING_ARTIFACT_KINDS
        if f"Artifact type: {kind}." in text
    ]
    counts = {path: paths.count(path) for path in set(paths)}
    return {
        "artifact_count": len(paths),
        "artifact_kinds": kinds,
        "largest_duplicate_count": max(counts.values(), default=0),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
