"""Where each client opens: the terminal the user picks, or the client's app."""

from __future__ import annotations

import importlib.util
import json
import plistlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx import agent_clients as ac
from mtplx import client_opening as co
from mtplx.cli import main

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "apps" / "MTPLXApp" / "Tests" / "Fixtures" / "client_opening_cases.json"
TERMINAL, GHOSTTY, WARP = co.FIXTURE_TERMINALS


@pytest.fixture(autouse=True)
def _mtplx3_agent_clients_on(monkeypatch):
    """This file covers the agent clients that arrive in MTPLX 3, with their release switch on
    (mtplx/release_gates.py); tests/test_release_gates.py covers the release with it off."""
    monkeypatch.setattr("mtplx.release_gates.V3_AGENT_CLIENTS", True)


def _bundle(root: Path, name: str, info: dict) -> Path:
    bundle = root / f"{name}.app"
    (bundle / "Contents").mkdir(parents=True)
    with (bundle / "Contents" / "Info.plist").open("wb") as handle:
        plistlib.dump(info, handle)
    return bundle


def _shell_role(bundle_id: str, name: str, role: str = "Shell") -> dict:
    return {
        "CFBundleIdentifier": bundle_id,
        "CFBundleName": name,
        "CFBundleDocumentTypes": [
            {"CFBundleTypeRole": role, "LSItemContentTypes": [co.SHELL_SCRIPT_TYPE]}
        ],
    }


def test_terminals_are_found_from_what_each_app_declares(tmp_path):
    system, apps = tmp_path / "system", tmp_path / "apps"
    _bundle(system, "Terminal", _shell_role("com.apple.Terminal", "Terminal"))
    _bundle(apps, "Ghostty", _shell_role("com.mitchellh.ghostty", "Ghostty"))
    _bundle(apps, "Alacritty", _shell_role("org.alacritty", "Alacritty"))
    _bundle(apps, "Editor", _shell_role("com.example.editor", "Editor", role="Editor"))
    _bundle(apps, "Notes", {"CFBundleIdentifier": "com.example.notes"})
    (apps / "Broken.app").mkdir()
    found = co.installed_terminals([system, apps, tmp_path / "missing"])
    assert [t.name for t in found] == ["Terminal", "Alacritty", "Ghostty"]
    assert found[2].path == str(apps / "Ghostty.app")


def test_the_same_terminal_in_two_folders_is_listed_once(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    _bundle(first, "Ghostty", _shell_role("com.mitchellh.ghostty", "Ghostty"))
    _bundle(second, "Ghostty", _shell_role("com.mitchellh.ghostty", "Ghostty"))
    (found,) = co.installed_terminals([first, second])
    assert found.path == str(first / "Ghostty.app")


@pytest.mark.parametrize(
    "client, choice, surface, terminal",
    [
        ("pi", None, co.TERMINAL, "Terminal"),
        ("pi", "Ghostty", co.TERMINAL, "Ghostty"),
        ("pi", " ghostty ", co.TERMINAL, "Ghostty"),
        ("pi", "com.mitchellh.ghostty", co.TERMINAL, "Ghostty"),
        ("opencode", None, co.OWN_APP, None),
        ("opencode", "Warp", co.TERMINAL, "Warp"),
        ("hermes", None, co.OWN_APP, None),
        ("claude-code", None, co.TERMINAL, "Terminal"),
        ("claude-code", "Ghostty", co.TERMINAL, "Ghostty"),
        ("codex", None, co.MTPLX_APP, None),
        ("codex", "Ghostty", co.TERMINAL, "Ghostty"),
    ],
)
def test_each_client_opens_where_it_was_set(client, choice, surface, terminal):
    opening = co.resolve(client, choice, list(co.FIXTURE_TERMINALS))
    assert opening.surface == surface and opening.note is None
    assert (opening.terminal.name if opening.terminal else None) == terminal


def test_a_setting_never_makes_a_launch_fail():
    terminals = list(co.FIXTURE_TERMINALS)
    gone = co.resolve("pi", "iTerm", terminals)
    assert gone.terminal == TERMINAL and gone.note == "iTerm is not installed; opening Terminal"
    no_app = co.resolve("pi", co.OWN_APP, terminals)
    assert no_app.surface == co.TERMINAL and "has no app" in no_app.note
    # Codex's command (`--cli`) goes to the chosen terminal, or Terminal when
    # the setting picks the app copy.
    assert co.terminal_for("codex", "Warp", terminals).terminal == WARP
    assert co.terminal_for("codex", None, terminals).terminal == TERMINAL
    assert co.terminal_for("codex", None, []).terminal_name == "Terminal"


def test_the_setting_is_read_from_the_shared_config(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text('pi_opens_in = "Ghostty"\ncodex_opens_in = "terminal"\n', encoding="utf-8")
    monkeypatch.setenv("MTPLX_CONFIG", str(config))
    assert co.saved_choice("pi") == "Ghostty"
    assert co.saved_choice("codex") == co.TERMINAL
    assert co.saved_choice("hermes") is None
    assert ac.desktop_app_for("codex", choice=co.saved_choice("codex")) is None


def test_config_set_accepts_only_what_this_mac_can_open(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setattr(co, "installed_terminals", lambda search_dirs=None: list(co.FIXTURE_TERMINALS))
    assert main(["config", "set", "pi_opens_in", "ghostty"]) == 0
    assert json.loads(capsys.readouterr().out)["updated"] == {"pi_opens_in": "Ghostty"}
    assert co.saved_choice("pi") == "Ghostty"
    with pytest.raises(SystemExit) as refused:
        main(["config", "set", "pi_opens_in", "iTerm"])
    assert str(refused.value) == "pi_opens_in must be one of: Terminal, Ghostty, Warp"
    with pytest.raises(SystemExit) as refused:
        main(["config", "set", "pi_opens_in", co.MTPLX_APP])
    assert "must be one of" in str(refused.value)
    assert main(["config", "set", "codex_opens_in", co.MTPLX_APP]) == 0
    capsys.readouterr()
    # Another `config set` keeps the choice (it rewrites the file).
    assert main(["config", "set", "context_window", "65536"]) == 0
    assert co.saved_choice("pi") == "Ghostty" and co.saved_choice("codex") == co.MTPLX_APP


def test_mtplx_clients_shows_where_each_client_opens(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "config.toml"))
    (tmp_path / "config.toml").write_text('hermes_opens_in = "Warp"\n', encoding="utf-8")
    monkeypatch.setattr(co, "installed_terminals", lambda search_dirs=None: list(co.FIXTURE_TERMINALS))
    codex = {"path": "/Applications/ChatGPT.app", "supported": True, "executable": "/x/ChatGPT"}
    monkeypatch.setattr(ac, "find_codex_desktop_app", lambda: codex)
    monkeypatch.setattr(ac, "codex_desktop_instances", lambda app, user_data=None: [])
    assert main(["clients", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    opens_in = {row["client"]: row["opens_in"] for row in payload["openings"]["clients"]}
    assert opens_in == {
        "pi": "Terminal",
        "opencode": "the OpenCode app",
        "hermes": "Warp",
        "claude-code": "Terminal",
        "codex": "MTPLX's own Codex app",
        "deepseek-harness": "the DeepSeek Harness web app",
    }
    # With no Codex app MTPLX can open, `mtplx start codex` opens the CLI.
    monkeypatch.setattr(ac, "find_codex_desktop_app", lambda: None)
    assert main(["clients", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)["openings"]["clients"]
    codex_row = next(row for row in rows if row["client"] == "codex")
    assert codex_row["opens_in"] == "Terminal (no Codex app MTPLX can open)"
    assert main(["clients"]) == 0
    out = capsys.readouterr().out
    assert "terminals found: Terminal, Ghostty, Warp" in out
    # One column for every client's choice, the longest name included.
    lines = out.splitlines()
    start = lines.index("Where each client opens (the app's Settings shares these):") + 1
    names = [row["name"] for row in rows]
    listed = lines[start : start + len(names)]
    assert [line.split("  ")[1].strip() for line in listed] == names
    columns = {len(line) - len(line[2:].split("  ", 1)[1].lstrip()) for line in listed}
    assert len(columns) == 1
    assert listed[names.index("Hermes")].endswith(" Warp")


def test_a_client_command_opens_in_the_chosen_terminal(monkeypatch, tmp_path):
    from mtplx import pi

    monkeypatch.setattr(co.sys, "platform", "darwin")
    monkeypatch.setattr(co, "installed_terminals", lambda search_dirs=None: list(co.FIXTURE_TERMINALS))
    monkeypatch.setattr(co, "saved_choice", lambda client, config=None: {"hermes": "Ghostty"}.get(client))
    calls = []
    monkeypatch.setattr(
        co.subprocess,
        "run",
        lambda argv, **_kw: calls.append(argv) or subprocess.CompletedProcess(argv, 0, "", ""),
    )
    result = pi.launch_pi_in_terminal("exec hermes chat", client="hermes", workspace=str(tmp_path))
    assert result["ok"] and result["terminal"] == "Ghostty" and result["note"] is None
    assert calls[-1][:3] == ["/usr/bin/open", "-a", GHOSTTY.path]
    script = Path(calls[-1][3])
    assert script.read_text() == f"#!/bin/zsh\ncd {tmp_path} || exit 1\nexec hermes chat\n"
    assert script.stat().st_mode & 0o777 == 0o700
    assert pi.launch_pi_in_terminal("pi", workspace=str(tmp_path))["terminal"] == "Terminal"
    assert calls[-1][:3] == ["/usr/bin/open", "-a", TERMINAL.path]


def test_opencode_opens_its_app_or_its_terminal_ui(monkeypatch, tmp_path):
    from mtplx import opencode

    monkeypatch.setattr(co.sys, "platform", "darwin")
    monkeypatch.setattr(co, "installed_terminals", lambda search_dirs=None: list(co.FIXTURE_TERMINALS))
    monkeypatch.setattr(opencode, "launch_opencode_app", lambda: {"ok": True, "app": "OpenCode"})
    choice = {"opencode": None}
    monkeypatch.setattr(co, "saved_choice", lambda client, config=None: choice.get(client))
    assert opencode.open_opencode("m")["opened"] == "OpenCode Desktop"
    calls = []
    monkeypatch.setattr(
        co.subprocess,
        "run",
        lambda argv, **_kw: calls.append(argv) or subprocess.CompletedProcess(argv, 0, "", ""),
    )
    choice["opencode"] = "Warp"
    monkeypatch.chdir(tmp_path)
    result = opencode.open_opencode("mtplx-qwen38-27b-optimized-speed")
    assert result["ok"] and result["opened"] == "OpenCode in Warp"
    assert calls[-1][:3] == ["/usr/bin/open", "-a", WARP.path]
    script = Path(calls[-1][3]).read_text()
    assert script.endswith("exec opencode --model mtplx/mtplx-qwen38-27b-optimized-speed\n")
    assert "command -v opencode" in script


def test_the_swift_fixture_is_current():
    spec = importlib.util.spec_from_file_location(
        "write_client_opening_cases", ROOT / "scripts" / "write_client_opening_cases.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert FIXTURE.read_text(encoding="utf-8") == module.render(), (
        "stale fixture: run `python scripts/write_client_opening_cases.py`"
    )
