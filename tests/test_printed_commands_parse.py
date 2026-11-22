"""Every `mtplx ...` command the CLI prints for a user to run is one the CLI accepts.

`mtplx connect <client>` and `mtplx integrate <client>` print the server command to start, and `mtplx start
<client> --dry-run` prints the command the start runs. Through 2.13's first candidates a chosen window put
`--context-window N` into a printed `mtplx quickstart` or `mtplx start` command, which neither takes (`mtplx
serve` does), so the command stopped with "unrecognized arguments" when run as printed. Each printed command
here goes through the CLI's own parser, with and without a chosen window, for every client this release offers
and, with the MTPLX 3 switch on, for the clients behind it.
"""

from __future__ import annotations

import io
import json
import re
import shlex
from contextlib import redirect_stderr, redirect_stdout

import pytest

from mtplx import cli, release_gates

# A printed command: on its own line, or after a label ("start server: ..."). A sentence that names a command
# ("mtplx start pi opens Pi for you.") ends with a full stop and is prose, not a command to run.
_PRINTED = re.compile(r"^(?:\s*|.*?:\s+)(mtplx (?:quickstart|start|serve)\b.*?)\s*$")


def _run(argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(out):
        try:
            code = cli.main(argv)
        except SystemExit as exc:
            code = exc.code
    return int(code or 0), out.getvalue()


def _commands(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _commands(item)
    elif isinstance(value, list):
        for item in value:
            yield from _commands(item)
    elif isinstance(value, str) and value.strip().startswith("mtplx ") and not value.strip().endswith("."):
        yield value.strip()


def _parses(command: str) -> None:
    argv = shlex.split(command)
    assert argv[0] == "mtplx", command
    try:
        cli.build_parser().parse_args(argv[1:])
    except SystemExit as exc:
        pytest.fail(f"a printed command does not parse (exit {exc.code}): {command}")


def _check_output(argv: list[str]) -> int:
    """Run ``argv`` with --json and without; every mtplx command either prints must parse. Returns how many."""

    code, out = _run([*argv, "--json"])
    assert code == 0, out
    commands = list(_commands(json.loads(out)))
    code, text = _run(argv)
    assert code == 0, text
    for line in text.splitlines():
        match = _PRINTED.match(line)
        if match and not match.group(1).endswith("."):
            commands.append(match.group(1))
    for command in commands:
        _parses(command)
    return len(commands)


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    monkeypatch.setenv("MTPLX_OPENCODE_CONFIG", str(tmp_path / "opencode.json"))
    monkeypatch.setenv("MTPLX_OPENCODE_DESKTOP_SETTINGS_STORE", str(tmp_path / "default.dat"))
    monkeypatch.setenv("MTPLX_AGENT_CLIENTS_DIR", str(tmp_path / "clients"))


CHOSEN = ["--context-window", "100000"]


@pytest.mark.parametrize("window", ([], CHOSEN), ids=("default-window", "chosen-window"))
@pytest.mark.parametrize("name", ("openwebui", "claude-code", "opencode", "swival"))
@pytest.mark.parametrize("verb", ("connect", "integrate"))
def test_every_command_connect_prints_parses(verb, name, window):
    # Port 1: nothing answers, so the live-server probe fails at once and the pack metadata answers.
    assert _check_output([verb, name, "--port", "1", *window]) > 0


@pytest.mark.parametrize("target", ("pi", "opencode", "swival", "hermes"))
def test_every_command_a_start_preview_prints_parses(target):
    assert _check_output(["start", target, "--dry-run", "--model", "models/example", "--yes"]) > 0


def test_a_chosen_window_sets_the_client_and_never_the_printed_start_command():
    """`mtplx connect opencode --context-window N` writes N into OpenCode's config; the server command it prints
    passes no window (the server serves the model's window as far as the Mac's memory allows)."""

    code, out = _run(["connect", "opencode", "--port", "1", "--context-window", "100000", "--json"])
    assert code == 0, out
    payload = json.loads(out)
    assert "--context-window" not in payload["server_command"]
    code, out = _run(["start", "opencode", "--dry-run", "--json", "--model", "models/example", "--yes"])
    assert code == 0, out
    assert "--context-window" not in json.loads(out)["opencode"]["server_command"]


@pytest.mark.parametrize("target", ("claude-code", "codex"))
def test_with_the_mtplx_3_switch_on_their_printed_commands_parse(monkeypatch, target):
    monkeypatch.setattr(release_gates, "V3_AGENT_CLIENTS", True)
    assert _check_output(["start", target, "--dry-run", "--model", "models/example", "--yes"]) > 0
    assert _check_output(["connect", target, "--port", "1"]) > 0
