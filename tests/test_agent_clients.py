"""Claude Code and Codex on the local server: one launch plan, no user config edited."""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import stat
import tomllib
from pathlib import Path

import pytest

from mtplx import agent_clients as ac
from mtplx import launch_lane as ll
from mtplx.cli import main

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "apps" / "MTPLXApp" / "Tests" / "Fixtures" / "agent_client_plans.json"
BASE = "http://127.0.0.1:8000"
MODEL = "mtplx-qwen38-flash-next-optimized-speed"


def _plan(client, **overrides):
    kwargs = {"base_url": BASE, "model_id": MODEL, "context_window": 262144, "api_key": None}
    kwargs.update(overrides)
    return ac.build_plan(client, **kwargs)


# --- Claude Code ------------------------------------------------------------


def test_claude_code_always_carries_a_gateway_credential():
    # A base URL with no credential keeps the claude.ai login active, and
    # Claude Code would send that login to the local server.
    plan = _plan("claude-code")
    assert plan["env"]["ANTHROPIC_AUTH_TOKEN"] == ac.LOCAL_API_KEY
    assert _plan("claude-code", api_key="sk-x")["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-x"
    for name in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK",
                 "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY"):
        assert name in plan["unset_env"]


def test_claude_code_routes_every_model_alias_to_the_served_model():
    env = _plan("claude-code")["env"]
    assert env["ANTHROPIC_BASE_URL"] == BASE  # Anthropic root, no /v1
    for key in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
                "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                "CLAUDE_CODE_SUBAGENT_MODEL"):
        assert env[key] == MODEL


def test_claude_code_keeps_mcp_tools_behind_its_tool_search():
    # Against a host that is not Anthropic's, Claude Code sends every MCP tool
    # definition in full unless this is set: 820 tools, a 254,636-token first
    # prompt (12 Nov). The server keeps the deferred ones out of the prompt.
    plan = _plan("claude-code")
    assert plan["env"]["ENABLE_TOOL_SEARCH"] == "true"
    settings = json.loads(next(iter(plan["files"].values())))
    assert settings["env"]["ENABLE_TOOL_SEARCH"] == "true"


def test_claude_code_tool_search_stays_off_where_the_server_cannot_write_a_loaded_tool():
    # /health says anthropic_tool_search false when the served template does
    # not list tools the way a loaded one is written (Gemma 4, for one):
    # every tool goes up front instead.
    plan = _plan("claude-code", tool_search=False)
    assert plan["env"]["ENABLE_TOOL_SEARCH"] == "false"
    settings = json.loads(next(iter(plan["files"].values())))
    assert settings["env"]["ENABLE_TOOL_SEARCH"] == "false"


def test_claude_code_learns_the_served_window_and_a_proportional_answer_budget():
    env = _plan("claude-code", context_window=262144)["env"]
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "262144"
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "65536"
    assert ac.claude_code_output_tokens(65536) == 32000  # never below its own default
    assert ac.claude_code_output_tokens(2_000_000) == 128000  # its documented cap
    unknown = _plan("claude-code", context_window=None)["env"]
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in unknown
    assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS" not in unknown


def test_claude_code_settings_file_is_mtplx_owned_and_carries_the_same_env():
    plan = _plan("claude-code")
    (path, text), = plan["files"].items()
    assert plan["argv"] == [
        "claude",
        "--settings",
        path,
        "--append-system-prompt",
        ac.CLAUDE_CODE_TERMINAL_NOTE,
    ]
    assert Path(path) == ac.clients_dir() / "claude-code" / "settings.json"
    assert json.loads(text) == {"env": plan["env"]}
    # Never the user's own Claude Code files.
    assert ".claude" not in Path(path).parts


def test_claude_code_is_told_its_terminal_draws_no_latex_and_to_keep_a_no_search_request():
    """16 Nov: Flash-Next answered a maths question in Claude Code with raw
    LaTeX, and searched the web on the follow-up to "without searching". The
    note rides on Claude Code's own flag, the same text on every request."""

    plan = _plan("claude-code")
    note = plan["argv"][plan["argv"].index("--append-system-prompt") + 1]
    assert note == ac.CLAUDE_CODE_TERMINAL_NOTE
    assert "not LaTeX" in note and "$...$" in note
    assert "not to search" in note
    assert "\n" not in note  # one line in the Terminal script MTPLX writes
    assert _plan("codex")["argv"].count("--append-system-prompt") == 0


def test_plan_files_are_written_owner_only(tmp_path):
    plan = _plan("claude-code", api_key="sk-secret", settings_path=str(tmp_path / "s.json"))
    (written,) = ac.write_plan_files(plan)
    mode = stat.S_IMODE(os.stat(written).st_mode)
    assert mode == 0o600
    assert json.loads(Path(written).read_text())["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-secret"


def test_credentials_never_ride_on_argv():
    for client in ac.AGENT_CLIENTS:
        plan = _plan(client, api_key="sk-secret")
        assert not any("sk-secret" in str(arg) for arg in plan["argv"])
        assert "sk-secret" not in ac.display_command(plan)


# --- Codex -------------------------------------------------------------------


def test_codex_gets_the_provider_from_command_line_overrides_only():
    plan = _plan("codex")
    assert plan["files"] == {}  # ~/.codex/config.toml is never touched
    assert plan["env"] == {ac.CODEX_API_KEY_ENV: ac.LOCAL_API_KEY}
    argv = plan["argv"]
    assert argv[0] == "codex" and argv[-2:] == ["-m", MODEL]
    overrides = [argv[i + 1] for i, arg in enumerate(argv) if arg == "-c"]
    parsed: dict = {}
    for override in overrides:
        # Codex parses each value as TOML; so must every one we emit.
        parsed = _merge(parsed, tomllib.loads(override))
    assert parsed["model_provider"] == "mtplx"
    provider = parsed["model_providers"]["mtplx"]
    assert provider["base_url"] == BASE + "/v1"
    assert provider["wire_api"] == "responses"
    assert provider["env_key"] == ac.CODEX_API_KEY_ENV
    assert provider["requires_openai_auth"] is False
    assert provider["http_headers"] == {"x-mtplx-client": "codex"}
    # A hosted web_search tool is refused by a local server (400 on every turn).
    assert parsed["web_search"] == "disabled"
    assert parsed["model_context_window"] == 262144
    assert "model_context_window=" not in " ".join(_plan("codex", context_window=None)["argv"])


def _merge(left: dict, right: dict) -> dict:
    out = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


# --- Terminal script ----------------------------------------------------------


def test_terminal_script_sets_env_runs_the_users_copy_and_explains_a_missing_client(tmp_path):
    plan = _plan("claude-code", settings_path=str(tmp_path / "s.json"))
    script = ac.terminal_script(plan, workspace="/tmp/work dir", executable="/opt/x/claude")
    assert script.startswith("#!/bin/zsh\ncd '/tmp/work dir' || exit 1\n")
    assert "unset ANTHROPIC_API_KEY" in script
    assert f"export ANTHROPIC_MODEL={MODEL}" in script
    assert 'command -v claude' in script and "/opt/x/claude" in script
    assert "exit 127" in script and "code.claude.com" in script
    assert script.rstrip().endswith(
        f"--settings {tmp_path / 's.json'} --append-system-prompt "
        + shlex.quote(ac.CLAUDE_CODE_TERMINAL_NOTE)
    )


def test_open_in_terminal_writes_into_mtplx_owned_folders_only(tmp_path, monkeypatch):
    calls = []

    class Done:
        returncode = 0
        stderr = ""

    from mtplx import client_opening

    monkeypatch.setattr(ac.sys, "platform", "darwin")
    monkeypatch.setattr(ac.subprocess, "run", lambda argv, **kw: calls.append(argv) or Done())
    monkeypatch.setattr(client_opening, "installed_terminals", lambda search_dirs=None: [])
    result = ac.open_in_terminal(_plan("claude-code"), workspace=str(tmp_path))
    assert result["ok"]
    assert calls == [["/usr/bin/open", "-a", "Terminal", result["script"]]]
    assert result["terminal"] == "Terminal"
    script = Path(result["script"])
    assert script.parent == ac.handoff_dir()
    assert stat.S_IMODE(script.stat().st_mode) == 0o700
    assert (ac.clients_dir() / "claude-code" / "settings.json").is_file()


# --- Lane --------------------------------------------------------------------


@pytest.mark.parametrize("client", ac.AGENT_CLIENTS)
def test_agent_clients_take_the_serial_coding_agent_lane(client):
    lane = ll.comparable_lane(
        ll.resolve_launch_lane(client=client, family="qwen4_exp", memory_bytes=128 * ll.GIB)
    )
    assert lane["scheduler_mode"] == "serial" and lane["batching_preset"] == "latency"
    assert lane["ssd_session_cache"] == "on"
    assert lane["max_active_requests"] is None
    assert lane["temperature"] is None  # the family owns its sampler
    assert lane["env"] == ll.agent_env_block("opencode", 128 * ll.GIB)


def test_start_dry_run_reports_the_plan_and_lane(monkeypatch, capsys, tmp_path):
    from mtplx.commands import public

    monkeypatch.setattr(
        public, "_detect_total_ram_bytes_for_opencode_defaults", lambda: 128 * ll.GIB
    )
    monkeypatch.setattr(ac, "find_codex_desktop_app", lambda: None)
    for client in ac.AGENT_CLIENTS:
        code = main(["start", client, "--dry-run", "--json", "--model", "models/example", "--yes"])
        payload = json.loads(capsys.readouterr().out)
        assert code == 0
        assert payload["target"] == client
        agent = payload["agent_client"]
        assert agent["edits_user_config"] is False
        assert agent["plan"]["client"] == client
        assert "--ssd-session-cache on" in agent["server_command"]
        assert "--scheduler-mode serial" in agent["server_command"]
        assert payload["next"] == f"mtplx start {client}"


def test_start_accepts_the_short_claude_alias(capsys):
    code = main(["start", "claude", "--dry-run", "--json", "--model", "models/example", "--yes"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["target"] == "claude-code"


def test_connect_writes_only_mtplx_files_and_masks_keys(monkeypatch, capsys):
    monkeypatch.setenv("MTPLX_API_KEY", "sk-from-env")
    code = main(["connect", "claude-code", "--port", "18012", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["environment"]["ANTHROPIC_AUTH_TOKEN"] == "<api key>"
    assert "sk-from-env" not in payload["shell"]
    (written,) = payload["files_written"]
    assert Path(written).is_relative_to(ac.clients_dir())
    assert json.loads(Path(written).read_text())["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-from-env"
    code = main(["connect", "codex", "--port", "18012", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0 and payload["files_written"] == []
    assert payload["environment"]["MTPLX_API_KEY"] == "<api key>"


# --- Codex desktop app: MTPLX's own instance -------------------------------------


def _without_owned(data: dict) -> dict:
    data = dict(data)
    for key in ac.CODEX_DESKTOP_OWNED_KEYS:
        data.pop(key, None)
    providers = dict(data.get("model_providers", {}))
    providers.pop("mtplx", None)
    if providers:
        data["model_providers"] = providers
    else:
        data.pop("model_providers", None)
    return data


@pytest.mark.parametrize("existing", ac.FIXTURE_CODEX_DESKTOP_CONFIGS)
def test_codex_desktop_config_sets_mtplx_and_keeps_every_other_setting(existing):
    text = ac.codex_desktop_config_text(
        existing, base_url=BASE, model_id=MODEL, context_window=65536, api_key=None
    )
    parsed = tomllib.loads(text)
    assert parsed["model"] == MODEL and parsed["model_provider"] == "mtplx"
    assert parsed["web_search"] == "disabled" and parsed["model_context_window"] == 65536
    provider = parsed["model_providers"]["mtplx"]
    assert provider["base_url"] == BASE + "/v1" and provider["wire_api"] == "responses"
    assert provider["requires_openai_auth"] is False
    assert "experimental_bearer_token" not in provider and "env_key" not in provider
    # A new copy starts at MTPLX's coding effort; a line the user (or the
    # app) already wrote is theirs and is kept.
    before = tomllib.loads(existing or "")
    assert parsed["model_reasoning_effort"] == before.get(
        "model_reasoning_effort", ac.CODEX_DESKTOP_DEFAULT_EFFORT
    )
    parsed.pop("model_reasoning_effort")
    before.pop("model_reasoning_effort", None)
    # The app's plugins and MCP servers, the user's own provider and settings:
    # every value outside MTPLX's keys survives, and a second launch is a no-op.
    assert _without_owned(parsed) == _without_owned(before)
    assert ac.codex_desktop_config_text(
        text, base_url=BASE, model_id=MODEL, context_window=65536, api_key=None
    ) == text


def test_codex_desktop_key_lives_in_the_owner_only_file_never_on_argv():
    plan = ac.build_codex_desktop_plan(
        app_path="/Applications/ChatGPT.app", base_url=BASE, model_id=MODEL,
        api_key="sk-secret",
    )
    assert not any("sk-secret" in arg for arg in plan["argv"])
    (path, text), = plan["files"].items()
    assert tomllib.loads(text)["model_providers"]["mtplx"]["experimental_bearer_token"] == "sk-secret"
    assert Path(path) == ac.codex_desktop_home() / "config.toml"
    assert plan["argv"][:2] == ["/usr/bin/open", "-n"]
    assert f"CODEX_HOME={ac.codex_desktop_home()}" in plan["argv"]
    assert f"CODEX_ELECTRON_USER_DATA_PATH={ac.codex_desktop_user_data()}" in plan["argv"]
    assert plan["argv"][-1] == f"--user-data-dir={ac.codex_desktop_user_data()}"
    for folder in plan["folders"]:
        assert Path(folder).is_relative_to(ac.clients_dir())
        assert ".codex" not in Path(folder).parts


def _fake_app(root: Path, name: str, bundle_id: str, version: str) -> None:
    import plistlib

    contents = root / name / "Contents"
    contents.mkdir(parents=True)
    with (contents / "Info.plist").open("wb") as handle:
        plistlib.dump(
            {
                "CFBundleIdentifier": bundle_id,
                "CFBundleShortVersionString": version,
                "CFBundleExecutable": Path(name).stem,
            },
            handle,
        )


def test_codex_desktop_app_is_found_by_bundle_id_and_fenced_to_the_proven_version(tmp_path):
    _fake_app(tmp_path, "ChatGPT.app", "com.openai.chat", "1.0")
    assert ac.find_codex_desktop_app((tmp_path,)) is None
    other = tmp_path / "b"
    _fake_app(other, "Codex.app", ac.CODEX_DESKTOP_BUNDLE_ID, "26.1106.51428")
    found = ac.find_codex_desktop_app((tmp_path, other))
    assert found == {
        "path": str(other / "Codex.app"),
        "version": "26.1106.51428",
        "supported": True,
        "executable": str(other / "Codex.app" / "Contents" / "MacOS" / "Codex"),
    }
    old = tmp_path / "c"
    _fake_app(old, "ChatGPT.app", ac.CODEX_DESKTOP_BUNDLE_ID, "26.1005.1")
    assert ac.find_codex_desktop_app((old,))["supported"] is False


def test_open_codex_desktop_keeps_the_previous_config_and_writes_owner_only(monkeypatch):
    calls = []

    class Done:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(ac.sys, "platform", "darwin")
    monkeypatch.setattr(ac.subprocess, "run", lambda argv, **kw: calls.append(argv) or Done())
    home = ac.codex_desktop_home()
    home.mkdir(parents=True)
    previous = '[mcp_servers.mine]\ncommand = "x"\n'
    (home / "config.toml").write_text(previous, encoding="utf-8")
    app = {"path": "/Applications/ChatGPT.app", "version": "26.1106.51428", "supported": True}
    result = ac.open_codex_desktop(
        base_url=BASE, model_id=MODEL, context_window=None, api_key=None, app=app,
        instance_timeout_s=0,
    )
    assert result["ok"] and calls == [result["plan"]["argv"]]
    assert result["pid"] is None and result["closed_previous"] == []
    assert (home / "config.toml.mtplx-previous").read_text(encoding="utf-8") == previous
    written = home / "config.toml"
    assert stat.S_IMODE(written.stat().st_mode) == 0o600
    assert tomllib.loads(written.read_text(encoding="utf-8"))["mcp_servers"]["mine"]["command"] == "x"
    unsupported = ac.open_codex_desktop(
        base_url=BASE, model_id=MODEL, context_window=None, api_key=None,
        app={**app, "supported": False},
    )
    assert not unsupported["ok"] and len(calls) == 1


def _launch_copy(marker: str):
    import subprocess

    # bash runs two commands, so it never execs sleep and keeps its argv.
    return subprocess.Popen(["/bin/bash", "-c", "sleep 30; exit 0", marker])


def test_the_codex_copy_is_the_main_process_with_mtplxs_folder_only():
    main_exe = "/Applications/ChatGPT.app/Contents/MacOS/ChatGPT"
    folder = "/Users/someone/.mtplx/clients/codex-desktop/user-data"

    def check(executable, argv):
        return ac.is_codex_desktop_instance(
            4242, main_exe, folder, read_arguments=lambda pid: (executable, argv)
        )

    assert check(main_exe, [main_exe, f"--user-data-dir={folder}"])
    assert not check(main_exe, [main_exe])  # the user's own Codex app
    assert not check(main_exe, [main_exe, f"--user-data-dir={folder}-other"])
    assert not check(main_exe, [f"--user-data-dir={folder}"])  # argv[0] is no argument
    helper = "/Applications/ChatGPT.app/Contents/Frameworks/ChatGPT Helper.app/Contents/MacOS/ChatGPT Helper"
    assert not check(helper, [helper, "--type=renderer", f"--user-data-dir={folder}"])
    assert not ac.is_codex_desktop_instance(4242, main_exe, folder, read_arguments=lambda pid: None)
    # A KERN_PROCARGS2 image: argc, exec path, padding, argv, then the
    # environment, which never counts.
    image = (2).to_bytes(4, "little") + b"\0".join(
        s.encode() for s in [main_exe, "", "", main_exe, f"--user-data-dir={folder}",
                             "MARK=--user-data-dir=/x", ""]
    )
    assert ac._parse_procargs(image) == (main_exe, [main_exe, f"--user-data-dir={folder}"])
    assert ac._parse_procargs(b"\x01\x00") is None


@pytest.mark.skipif(not Path("/bin/bash").exists(), reason="needs /bin/bash")
def test_closing_the_codex_copy_ends_exactly_that_process(tmp_path):
    if ac.sys.platform != "darwin":
        pytest.skip("KERN_PROCARGS2 is macOS")
    folder = tmp_path / "codex-desktop" / "user-data"
    app = {"executable": "/bin/bash"}
    copy = _launch_copy(f"--user-data-dir={folder}")
    lookalike = _launch_copy(f"--user-data-dir={folder}-not-mtplx")
    try:
        import time

        deadline = time.monotonic() + 5
        while ac.codex_desktop_instances(app, folder) != [copy.pid] and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ac.codex_desktop_instances(app, folder) == [copy.pid]
        assert ac.end_codex_desktop_instance(copy.pid, app, folder)
        copy.wait(timeout=5)
        assert lookalike.poll() is None, "never signal a process that is not the copy"
        assert not ac.end_codex_desktop_instance(lookalike.pid, app, folder)
        assert lookalike.poll() is None
        assert ac.codex_desktop_instances(app, folder) == []
    finally:
        for process in (copy, lookalike):
            if process.poll() is None:
                process.kill()
                process.wait()


def test_start_codex_opens_the_app_copy_by_default_and_the_cli_with_cli(monkeypatch, capsys):
    from types import SimpleNamespace

    from mtplx.commands import public

    monkeypatch.setattr(
        public, "_detect_total_ram_bytes_for_opencode_defaults", lambda: 128 * ll.GIB
    )
    app = {
        "path": "/Applications/ChatGPT.app",
        "version": "26.1106.51428",
        "supported": True,
        "executable": "/Applications/ChatGPT.app/Contents/MacOS/ChatGPT",
    }
    monkeypatch.setattr(ac, "find_codex_desktop_app", lambda: app)
    code = main(["start", "codex", "--dry-run", "--json", "--model", "models/example", "--yes"])
    agent = json.loads(capsys.readouterr().out)["agent_client"]
    assert code == 0 and agent["surface"] == "desktop"
    assert agent["desktop"]["argv"][:2] == ["/usr/bin/open", "-n"]
    assert agent["desktop"]["folder"] == str(ac.codex_desktop_root())
    assert agent["desktop"]["separate_history"] is True
    # Claude Code and `--cli` stay in Terminal; no installed CLI is needed
    # for the app copy.
    code = main(["start", "codex", "--cli", "--dry-run", "--json", "--model", "models/example", "--yes"])
    assert code == 0 and json.loads(capsys.readouterr().out)["agent_client"]["surface"] == "terminal"
    code = main(["start", "claude-code", "--dry-run", "--json", "--model", "models/example", "--yes"])
    assert code == 0 and json.loads(capsys.readouterr().out)["agent_client"]["surface"] == "terminal"
    monkeypatch.setattr(ac, "resolve_executable", lambda client: None)
    assert public._quickstart_require_agent_cli(SimpleNamespace(agent_client_cli=False), "codex")
    assert not public._quickstart_require_agent_cli(SimpleNamespace(agent_client_cli=True), "codex")
    capsys.readouterr()
    # The server child gets --agent-client-cli only when asked.
    from mtplx.server import openai as server

    assert server._agent_client_desktop_app(
        SimpleNamespace(launch_agent_client="codex", agent_client_cli=False)
    ) == app
    assert server._agent_client_desktop_app(
        SimpleNamespace(launch_agent_client="codex", agent_client_cli=True)
    ) is None
    assert server._agent_client_desktop_app(
        SimpleNamespace(launch_agent_client="claude-code", agent_client_cli=False)
    ) is None


def test_clients_lists_the_codex_copy_and_removes_it_only_when_asked(monkeypatch, capsys):
    monkeypatch.setattr(ac, "find_codex_desktop_app", lambda: None)
    code = main(["clients", "--json"])
    rows = json.loads(capsys.readouterr().out)["clients"]
    assert code == 0 and rows[0]["name"] == "codex-desktop" and rows[0]["exists"] is False
    root = ac.codex_desktop_root()
    (root / "user-data" / "plugins" / "locked").mkdir(parents=True)
    (root / "user-data" / "plugins" / "locked" / "a.bin").write_bytes(b"x" * 10)
    (root / "user-data" / "plugins" / "locked").chmod(0o555)  # as the app unpacks some
    (root / "home").mkdir()
    code = main(["clients", "--json"])
    row = json.loads(capsys.readouterr().out)["clients"][0]
    assert code == 0 and row["exists"] and row["bytes"] == 10
    # Non-interactive removal needs --yes; an open copy is never removed.
    monkeypatch.setattr(ac.sys.stdin, "isatty", lambda: False)
    assert main(["clients", "remove", "codex-desktop"]) == 1
    assert "--yes" in capsys.readouterr().err and root.is_dir()
    monkeypatch.setattr(ac, "codex_desktop_instances", lambda app, user_data=None: [4242])
    monkeypatch.setattr(ac, "find_codex_desktop_app", lambda: {"executable": "/x"})
    assert main(["clients", "remove", "codex-desktop", "--yes"]) == 1
    assert "is open" in capsys.readouterr().err and root.is_dir()
    monkeypatch.setattr(ac, "codex_desktop_instances", lambda app, user_data=None: [])
    assert main(["clients", "remove", "codex-desktop", "--yes", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["bytes"] == 10 and not root.exists()
    assert main(["clients", "remove", "nothing-here"]) == 1


def test_connect_codex_desktop_says_what_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(ac, "find_codex_desktop_app", lambda: None)
    code = main(["connect", "codex", "--desktop", "--port", "18012", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1 and payload["surface"] == "desktop"
    assert "not installed" in payload["error"] and payload["edits_user_config"] is False


def test_the_swift_fixture_is_current():
    spec = importlib.util.spec_from_file_location(
        "write_agent_client_plans", ROOT / "scripts" / "write_agent_client_plans.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert FIXTURE.read_text(encoding="utf-8") == module.render(), (
        "stale fixture: run `python scripts/write_agent_client_plans.py`"
    )


def test_the_server_opens_the_codex_copy_and_closes_it_when_it_exits(monkeypatch, capsys):
    """`mtplx start codex`: the server opens the copy once ready and registers
    its pid to be closed at exit, as the app's Stop does."""

    import atexit
    from types import SimpleNamespace

    from mtplx.server import openai as server

    app = {"path": "/Applications/ChatGPT.app", "supported": True, "executable": "/x/ChatGPT"}
    monkeypatch.setattr(ac, "find_codex_desktop_app", lambda: app)
    opened, registered = [], []
    monkeypatch.setattr(
        ac, "open_codex_desktop", lambda **kw: opened.append(kw) or {"ok": True, "pid": 4242}
    )
    monkeypatch.setattr(atexit, "register", lambda fn, *a: registered.append((fn, a)))

    class Now:
        def __init__(self, _delay, fn):
            self.fn = fn
            self.daemon = True

        def start(self):
            self.fn()

    monkeypatch.setattr(server, "Timer", Now)
    monkeypatch.setattr(server, "served_execution_window", lambda state: {"tokens": 65536})
    args = SimpleNamespace(
        launch_agent_client="codex", agent_client_cli=False, model_id="m",
        host="127.0.0.1", port=8123, api_key=None, agent_client_workspace="",
    )
    server._open_agent_client_later(SimpleNamespace(args=args))
    assert opened and opened[0]["context_window"] == 65536 and opened[0]["app"] == app
    assert registered == [(ac.end_desktop_copy, ("codex", 4242, app))]
    assert "closes when this server stops" in capsys.readouterr().out


# --- CLI and app: every client on both surfaces (D6) ------------------------

# Each lane client the app's Start menu offers (mtplx/launch_lane.py CLIENTS,
# the list the Swift parity test maps to LaunchTarget) and how the CLI starts
# it; and each `mtplx start` target and the lane it runs. A client added to
# one surface without the other fails here until it is placed or declared.
APP_CLIENT_ON_THE_CLI = {
    "chat": "start cli",          # the app's own chat; the terminal chat is the CLI's
    "openwebui": "start web",
    "opencode": "start opencode",
    "pi": "start pi",
    "hermes": "start hermes",
    "claude-code": "start claude-code",
    "codex": "start codex",
    "deepseek-harness": "start deepseek-harness",
    "other": "serve",             # any OpenAI/Anthropic client: a bare server
    "benchmark": "bench",          # the app's AIME overlay; `mtplx bench` on the CLI
}
CLI_TARGET_IN_THE_APP = {
    "terminal": "chat",
    "openwebui": "openwebui",
    "pi": "pi",
    "opencode": "opencode",
    "hermes": "hermes",
    "claude-code": "claude-code",
    "codex": "codex",
    "deepseek-harness": "deepseek-harness",
    # Declared CLI-only: Swival is left out of the app at the founder's
    # request; the app shows its own live dashboard.
    "swival": None,
    "dashboard": None,
}


def test_every_client_is_on_both_surfaces_or_declared():
    from mtplx.commands.public import QUICKSTART_TARGETS

    assert set(APP_CLIENT_ON_THE_CLI) == set(ll.CLIENTS)
    assert set(CLI_TARGET_IN_THE_APP) == set(QUICKSTART_TARGETS) - {"open-webui"}
    for target, client in CLI_TARGET_IN_THE_APP.items():
        assert client is None or client in ll.CLIENTS, target


@pytest.mark.parametrize("alias", ["web", "webui", "openwebui", "open-webui"])
def test_the_apps_web_ui_name_works_on_the_cli(alias, capsys):
    code = main(["start", alias, "--dry-run", "--json", "--model", "models/example", "--yes"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["target"] == "openwebui"
