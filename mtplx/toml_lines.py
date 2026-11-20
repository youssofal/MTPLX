"""Line-level TOML editing that keeps everything it does not own.

MTPLX edits a few keys in files people and other programs also write: the
model folders in ``~/.mtplx/config.toml`` (shared by the CLI and the app) and
MTPLX's provider in the Codex desktop instance's ``config.toml``. A TOML
writer would reformat the whole file and drop comments; these helpers split
a file into statements (a table header, a key whose value may span lines, or
a comment/blank line) so a caller replaces only its own statements.

The app has a Swift twin (``TOMLLines`` in MTPLXAppCore); fixture tests keep
the two byte-identical. Pure: no MLX, no I/O.
"""

from __future__ import annotations

import json
from typing import Any, NamedTuple


class Statement(NamedTuple):
    table: str  # "" for the top level
    key: str | None  # None for headers, comments and blank lines
    lines: list[str]


def value_open_depth(text: str, depth: int, quote: str | None) -> tuple[int, str | None]:
    """Bracket depth after ``text``, outside strings and comments.

    ``quote`` is a multi-line string delimiter (``\"\"\"`` or ``'''``)
    carried over from the line before, or None.
    """

    i = 0
    while i < len(text):
        if quote is not None:
            end = text.find(quote, i)
            if end < 0:
                return depth, quote
            i = end + len(quote)
            quote = None
            continue
        ch = text[i]
        if text.startswith('"""', i) or text.startswith("'''", i):
            quote = text[i : i + 3]
            i += 3
            continue
        if ch == '"':
            i += 1
            while i < len(text) and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
            i += 1
            continue
        if ch == "'":
            end = text.find("'", i + 1)
            i = len(text) if end < 0 else end + 1
            continue
        if ch == "#":
            break
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        i += 1
    return depth, quote


def statements(text: str) -> list[Statement]:
    out: list[Statement] = []
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    table = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("["):
            header = stripped.split("#", 1)[0].strip()
            header = header[2:-2] if header.startswith("[[") else header[1:-1]
            table = header.strip()
            out.append(Statement(table, None, [line]))
            i += 1
            continue
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out.append(Statement(table, None, [line]))
            i += 1
            continue
        key = stripped.split("=", 1)[0].strip().strip('"').strip("'")
        block = [line]
        depth, quote = value_open_depth(stripped.split("=", 1)[1], 0, None)
        while (depth > 0 or quote is not None) and i + 1 < len(lines):
            i += 1
            block.append(lines[i])
            depth, quote = value_open_depth(lines[i], depth, quote)
        out.append(Statement(table, key, block))
        i += 1
    return out


def render_value(value: Any) -> str:
    """A TOML value as MTPLX writes it: JSON strings, numbers, booleans and
    arrays of those (the same lines ``mtplx config set`` writes)."""

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(render_value(item) for item in value) + "]"
    return json.dumps(value)


def set_top_level_values(text: str | None, updates: dict[str, Any]) -> str:
    """Set top-level keys (a value of None removes the key), keep the rest.

    A key that exists is replaced where it stands; a new key goes after the
    last top-level line, before the first table.
    """

    pending = dict(updates)
    out: list[str] = []
    first_table = None
    for statement in statements(text or ""):
        if statement.table == "" and statement.key in pending:
            value = pending.pop(statement.key)
            if value is not None:
                out.append(f"{statement.key} = {render_value(value)}")
            continue
        if first_table is None and statement.key is None and statement.lines[0].strip().startswith("["):
            first_table = len(out)
        out.extend(statement.lines)
    added = [f"{key} = {render_value(value)}" for key, value in pending.items() if value is not None]
    if added:
        if first_table is None:
            out.extend(added)
        else:
            insert_at = first_table
            while insert_at > 0 and not out[insert_at - 1].strip():
                insert_at -= 1
            out[insert_at:insert_at] = added
    return "\n".join(out) + ("\n" if out else "")
