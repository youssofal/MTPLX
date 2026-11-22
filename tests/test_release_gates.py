"""What MTPLX 2.13 offers a user to pick (mtplx/release_gates.py).

The founder, 22 November 2026: Claude Code, Codex and DeepSeek Harness as harnesses MTPLX starts arrive in
MTPLX 3. The CLI offers none of them (`mtplx start`, `mtplx connect codex|deepseek-harness`, `mtplx integrate
codex`, `mtplx clients`), its help names none, a direct call says in one line that they arrive in MTPLX 3, and
`mtplx connect claude-code` (2.12.3's) stays, pointing at a server command 2.13 has. Their launch code keeps its
own tests with the switch on (test_agent_clients.py, test_client_opening.py, test_deepseek_harness.py). The app's
twin: apps/MTPLXApp/Tests/MTPLXAppCoreTests/ReleaseGateTests.swift.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout

import pytest

from mtplx import cli, release_gates

V3_STARTS = ("claude-code", "claude", "codex", "deepseek-harness", "deepseek", "dsh")
V3_CONNECTS = ("codex", "deepseek-harness", "dsh")
V3_WORDS = ("claude code", "codex", "deepseek", "dsh")


def _run(argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(out):
        try:
            code = cli.main(argv)
        except SystemExit as exc:  # argparse's --help
            code = exc.code
    return int(code or 0), out.getvalue()


@pytest.fixture(autouse=True)
def _fresh_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "mtplx-config.json"))
    monkeypatch.setenv("MTPLX_AGENT_CLIENTS_DIR", str(tmp_path / "clients"))


def test_the_release_switch_is_off():
    assert release_gates.V3_AGENT_CLIENTS is False


@pytest.mark.parametrize("target", V3_STARTS)
def test_a_direct_start_says_in_one_line_that_it_arrives_in_mtplx_3(target):
    code, out = _run(["start", target, "--dry-run"])
    assert code == 2
    assert out.strip().count("\n") == 0, out
    assert "arrives in MTPLX 3" in out
    assert "mtplx start pi" in out


@pytest.mark.parametrize("action", V3_CONNECTS)
def test_a_direct_connect_says_in_one_line_that_it_arrives_in_mtplx_3(action):
    code, out = _run(["connect", action])
    assert code == 2
    assert out.strip().count("\n") == 0, out
    assert "arrives in MTPLX 3" in out


def test_integrate_codex_arrives_in_mtplx_3():
    code, out = _run(["integrate", "codex"])
    assert code == 2 and "arrives in MTPLX 3" in out


def test_claude_codes_connect_stays_and_points_at_a_server_command_this_release_has():
    code, out = _run(["connect", "claude-code"])
    assert code == 0, out
    assert "MTPLX connect: claude-code" in out
    assert "start server: mtplx quickstart " in out
    assert "mtplx start claude-code" not in out
    code, out = _run(["connect", "claude-code", "--json"])
    payload = json.loads(out)
    assert payload["server_command"].startswith("mtplx quickstart ")
    assert not any("mtplx start claude-code" in note for note in payload["notes"])


def test_the_connect_overview_lists_what_2123_listed():
    code, out = _run(["connect", "--json"])
    assert code == 0
    names = [item["name"] for item in json.loads(out)["integrations"]]
    assert names == ["openwebui", "claude-code", "opencode", "swival"]
    code, out = _run(["connect"])
    lowered = out.lower()
    assert "codex" not in lowered and "deepseek" not in lowered


def test_the_clients_listing_names_none_of_them():
    code, out = _run(["clients"])
    assert code == 0, out
    lowered = out.lower()
    for word in V3_WORDS:
        assert word not in lowered, (word, out)
    code, out = _run(["clients", "--json"])
    payload = json.loads(out)
    assert [row["client"] for row in payload["openings"]["clients"]] == ["pi", "opencode", "hermes"]
    assert payload["clients"] == []


@pytest.mark.parametrize("argv", (["--help"], ["help", "start"], ["start", "--help"], ["connect", "--help"],
                                  ["clients", "--help"], ["integrate", "--help"], ["help", "commands"]))
def test_the_help_names_none_of_them(argv):
    code, out = _run(argv)
    lowered = out.lower()
    for word in ("codex", "deepseek", "dsh"):
        assert word not in lowered, (argv, word)
    # Claude Code appears only as 2.12.3's `mtplx connect claude-code` and `mtplx integrate claude-code`.
    for line in lowered.splitlines():
        if "claude" in line:
            assert ("connect" in line or "copy settings" in line
                    or "{openwebui,claude-code,opencode,swival}" in line), (argv, line)


def test_the_switch_offers_them_again(monkeypatch):
    monkeypatch.setattr(release_gates, "V3_AGENT_CLIENTS", True)
    for name in (*V3_STARTS, *V3_CONNECTS):
        assert release_gates.offered(name)
    assert "mtplx start claude-code" in cli._format_start_help()
    code, out = _run(["connect", "--json"])
    names = [item["name"] for item in json.loads(out)["integrations"]]
    assert {"codex", "codex-desktop", "deepseek-harness"} <= set(names)
