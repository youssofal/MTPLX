"""DeepSeek Harness on the local server: MTPLX's own profile, telemetry off.

The unit tests run anywhere. ``test_the_installed_harness_composes_nothing_unreviewed``
runs against a real ``dsh`` (on PATH, or named by ``DSH_TEST_EXECUTABLE``) and
fails when a newer harness adds a row MTPLX has not reviewed, so a version
bump is read before it ships; it is skipped where no harness is installed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx import agent_clients as ac
from mtplx import client_opening as co
from mtplx import deepseek_harness as dh
from mtplx import launch_lane as ll
from mtplx.cli import main

BASE = "http://127.0.0.1:8000"
MODEL = "mtplx-qwen38-flash-next-optimized-speed"
# Every row the web profile of 0.1.2-rc.1 composes with MTPLX's overlay that
# stays on, each read and found local (the browser app's plugins, the agent,
# its tools, the session store, the local web server), with what it does. A
# newer harness that adds a row fails the installed-harness test until the
# row is read and placed there or switched off in OFF_ROWS.
REVIEWED_ROWS_FILE = Path(__file__).parent / "fixtures" / "deepseek_harness_reviewed_rows.json"
REVIEWED_ROWS = frozenset(json.loads(REVIEWED_ROWS_FILE.read_text())["rows"])


@pytest.fixture(autouse=True)
def _mtplx3_agent_clients_on(monkeypatch):
    """This file covers the agent clients that arrive in MTPLX 3, with their release switch on
    (mtplx/release_gates.py); tests/test_release_gates.py covers the release with it off."""
    monkeypatch.setattr("mtplx.release_gates.V3_AGENT_CLIENTS", True)


def _rows(**overrides):
    kwargs = {"base_url": BASE, "model_id": MODEL, "context_window": 262144, "vision": True}
    kwargs.update(overrides)
    return dh.overlay_rows(**kwargs)


def test_the_profile_routes_the_harness_to_mtplx_and_makes_it_the_default():
    rows = {row["id"]: row for row in _rows()}
    provider = rows["llm-pi-ai"]["config"]["providers"][dh.PROVIDER_ID]
    assert provider["baseURL"] == BASE + "/v1"
    assert provider["api"] == "openai-completions"
    assert provider["apiKeyEnv"] == dh.API_KEY_ENV  # the key travels in the environment
    assert provider["headers"] == {"x-mtplx-client": "deepseek_harness"}
    # The wire contract MTPLX's Pi provider is proven on.
    assert provider["compat"] == {
        "supportsDeveloperRole": False,
        "supportsReasoningEffort": True,
        "thinkingFormat": "qwen",
        "maxTokensField": "max_tokens",
    }
    (model,) = provider["models"]
    assert model["id"] == MODEL and model["contextWindow"] == 262144
    # The cap the library writes into every request is the window: it never binds.
    assert model["maxTokens"] == 262144
    assert model["reasoningEfforts"]["off"] is None
    assert rows["agent-default-model"]["config"] == {
        "provider": dh.PROVIDER_ID,
        "model": MODEL,
        "reasoningEffort": dh.DEFAULT_REASONING_EFFORT,
    }


def test_images_are_offered_only_when_the_pack_reads_them():
    with_vision = _rows(vision=True)[0]["config"]["providers"]["mtplx"]["models"][0]
    without = _rows(vision=False)[0]["config"]["providers"]["mtplx"]["models"][0]
    assert with_vision["input"] == ["text", "image"]
    assert without["input"] == ["text"]


def test_no_window_means_no_invented_size():
    model = _rows(context_window=None)[0]["config"]["providers"]["mtplx"]["models"][0]
    assert "contextWindow" not in model and "maxTokens" not in model


def test_every_reporting_piece_is_switched_off():
    rows = {row["id"]: row for row in _rows()}
    for row_id in (
        "session-telemetry-otel",
        "session-log-deepseek",
        "plugin-package-inventory-deepseek",
        "command-feedback",
        "message-feedback",
        "ui-message-feedback",
        "llm-deepseek",
        "web-search-deepseek",
    ):
        assert rows[row_id] == {"id": row_id, "disabled": True}, row_id
    env = dh.launch_env(api_key=None)
    assert env["DSH_TELEMETRY_DISABLED"] == "1"
    assert env["DSH_TELEMETRY_MODE"] == "DISABLED"
    assert env["DSH_TELEMETRY_OTLP_URL"].startswith("http://127.0.0.1:9/")
    assert env[dh.API_KEY_ENV] == dh.LOCAL_API_KEY


def test_the_overlay_file_is_yaml_the_harness_reads_and_carries_no_secret():
    plan = dh.build_web_plan(base_url=BASE, model_id=MODEL, port=4321, api_key="sk-secret")
    (path, text) = next(iter(plan["files"].items()))
    assert path == str(dh.overlay_path())
    body = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    assert json.loads(body) == _rows(vision=False, context_window=None)
    assert "sk-secret" not in text and not any("sk-secret" in arg for arg in plan["argv"])
    assert plan["env"][dh.API_KEY_ENV] == "sk-secret"
    assert plan["env"]["DSH_HOME"] == str(dh.home_dir())
    assert plan["unset_env"] == ["DEEPSEEK_API_KEY"]


def test_launcher_flags_come_before_the_apps_own():
    # dsh reads --patch only before the first flag it does not know.
    web = dh.build_web_plan(base_url=BASE, model_id=MODEL, port=4321)
    # MTPLX opens the browser itself once the server answers.
    assert web["argv"] == [
        "dsh", "web", "--patch", str(dh.overlay_path()), "--port", "4321", "--no-open",
    ]
    terminal = dh.build_terminal_plan(base_url=BASE, model_id=MODEL, port=4322)
    assert terminal["argv"] == ["dsh", "web", "--patch", str(dh.overlay_path()), "--port", "4322"]
    assert terminal["surface"] == "terminal"


def test_everything_lives_in_mtplxs_own_folder():
    plan = dh.build_web_plan(base_url=BASE, model_id=MODEL, port=4321)
    for path in [*plan["files"], *plan["folders"], plan["env"]["DSH_HOME"]]:
        assert Path(path).is_relative_to(ac.clients_dir()), path
        assert ".dsh" not in Path(path).parts


DUMP = """\
# == /x/dsh-base/cordis.patch.yml
- id: timer
  name: '@deepseek-ai/cordis-plugin-timer'
- id: session-telemetry-otel
  name: '@deepseek-ai/dsh-session-telemetry-otel'
  disabled: true
  config:
    mode: !!js process.env.DSH_TELEMETRY_MODE || 'FEEDBACK_ONLY'
- id: tool-bash
  name: '@deepseek-ai/dsh-tool-bash'
  disabled: !!js process.platform === 'win32'
- id: token-meter
  name: '@deepseek-ai/dsh-token-meter'
- id: ui-sidebar
  name: "@deepseek-ai/dsh-client-ui-sidebar"
"""


def test_the_dump_parser_reads_ids_packages_and_switches():
    rows = dh.parse_dump_rows(DUMP)
    assert [row["id"] for row in rows] == [
        "timer", "session-telemetry-otel", "tool-bash", "token-meter", "ui-sidebar",
    ]
    by_id = {row["id"]: row for row in rows}
    assert by_id["session-telemetry-otel"]["disabled"] is True
    # A platform expression is read as on: the audit errs that way.
    assert by_id["tool-bash"]["disabled"] is False
    assert by_id["ui-sidebar"]["name"] == "@deepseek-ai/dsh-client-ui-sidebar"


def test_the_audit_refuses_a_reporting_row_that_is_on():
    clean = dh.audit_rows(dh.parse_dump_rows(DUMP))
    assert clean["ok"] and clean["reporting"] == []
    # A newer version adds an update checker and an analytics exporter.
    newer = DUMP + (
        "- id: update-check\n  name: '@deepseek-ai/dsh-update-check'\n"
        "- id: usage-analytics\n  name: '@deepseek-ai/dsh-analytics'\n"
    )
    audit = dh.audit_rows(dh.parse_dump_rows(newer))
    assert not audit["ok"]
    assert {row["id"] for row in audit["reporting"]} == {"update-check", "usage-analytics"}
    # A row MTPLX switches off that is composed on means the overlay missed it.
    missed = DUMP.replace("  disabled: true\n", "")
    audit = dh.audit_rows(dh.parse_dump_rows(missed))
    assert not audit["ok"] and audit["reporting"][0]["id"] == "session-telemetry-otel"


def test_the_audit_names_rows_nobody_reviewed():
    audit = dh.audit_rows(dh.parse_dump_rows(DUMP), reviewed={"timer", "tool-bash", "token-meter"})
    assert [row["id"] for row in audit["unreviewed"]] == ["ui-sidebar"]


def test_a_refused_composition_never_starts_the_harness(monkeypatch):
    monkeypatch.setattr(dh, "resolve_executable", lambda: "/usr/local/bin/dsh")
    monkeypatch.setattr(
        dh,
        "dump_config",
        lambda plan, **kw: {"ok": True, "text": DUMP + "- id: crash-reporter\n  name: '@x/crash'\n"},
    )
    started = []
    monkeypatch.setattr(dh, "start_web", lambda *a, **kw: started.append(a) or {"ok": True})
    result = ac.open_deepseek_harness_web(
        base_url=BASE, model_id=MODEL, context_window=65536, api_key=None,
        vision=True, workspace="/tmp",
    )
    assert result["ok"] is False and "crash-reporter" in result["error"]
    assert started == []


def test_the_server_drops_the_harness_injected_cap_and_keeps_a_users_own():
    from mtplx.server import openai as server

    headers = {"x-mtplx-client": "deepseek_harness"}
    for injected in (32768, 65536):
        assert server._strip_client_injected_output_cap(
            injected, headers=headers, metadata={}, context_window=65536
        ) == (None, True)
    # Any other value is the user's choice and passes through.
    assert server._strip_client_injected_output_cap(
        4096, headers=headers, metadata={}, context_window=65536
    ) == (4096, False)
    # Another client's 32768 is untouched.
    assert server._strip_client_injected_output_cap(
        32768, headers={"x-mtplx-client": "pi"}, metadata={}, context_window=65536
    ) == (32768, False)


# --- where it opens: one setting, the app and `mtplx start` ----------------------


def test_it_opens_as_its_browser_app_by_default_and_in_a_terminal_when_picked(monkeypatch):
    monkeypatch.setattr(dh, "resolve_executable", lambda: "/usr/local/bin/dsh")
    assert co.resolve(dh.CLIENT_ID, None, []).surface == co.OWN_APP
    assert ac.desktop_app_for(dh.CLIENT_ID) == {"kind": "web", "path": "/usr/local/bin/dsh"}
    assert ac.desktop_app_for(dh.CLIENT_ID, cli=True) is None
    assert ac.desktop_app_for(dh.CLIENT_ID, choice="terminal") is None
    assert ac.desktop_app_for(dh.CLIENT_ID, choice="terminal", app=True) is not None
    assert co.CONFIG_KEYS["deepseek_harness_opens_in"] == dh.CLIENT_ID
    monkeypatch.setattr(dh, "resolve_executable", lambda: None)
    assert ac.desktop_app_for(dh.CLIENT_ID) is None


@pytest.mark.parametrize("alias", ["deepseek-harness", "dsh", "deepseek"])
def test_start_knows_it_by_every_name(alias, monkeypatch, capsys):
    from mtplx.commands import public

    monkeypatch.setattr(public, "_detect_total_ram_bytes_for_opencode_defaults", lambda: 128 * ll.GIB)
    code = main(["start", alias, "--dry-run", "--json", "--model", "models/example", "--yes"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0 and payload["target"] == dh.CLIENT_ID
    assert "--scheduler-mode serial" in payload["agent_client"]["server_command"]


def test_start_app_and_cli_choose_for_this_launch_only(monkeypatch, capsys):
    from mtplx.commands import public

    monkeypatch.setattr(public, "_detect_total_ram_bytes_for_opencode_defaults", lambda: 128 * ll.GIB)
    monkeypatch.setattr(dh, "resolve_executable", lambda: "/usr/local/bin/dsh")
    run = ["start", "deepseek-harness", "--dry-run", "--json", "--model", "models/example", "--yes"]
    assert main(run) == 0
    assert json.loads(capsys.readouterr().out)["agent_client"]["surface"] == "desktop"
    assert main([*run, "--cli"]) == 0
    assert json.loads(capsys.readouterr().out)["agent_client"]["surface"] == "terminal"
    co_save = co.validate_choice(dh.CLIENT_ID, "terminal", [])
    assert co_save == "terminal"
    with pytest.raises(SystemExit):
        main([*run, "--cli", "--app"])  # one or the other


def test_the_server_opens_its_browser_app_and_ends_it_when_it_exits(monkeypatch, capsys):
    import atexit

    from mtplx.server import openai as server

    monkeypatch.setattr(dh, "resolve_executable", lambda: "/usr/local/bin/dsh")
    opened, registered = [], []
    monkeypatch.setattr(
        ac,
        "open_deepseek_harness_web",
        lambda **kw: opened.append(kw) or {"ok": True, "pid": 4343, "url": "http://127.0.0.1:5/"},
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
    monkeypatch.setattr(server, "_server_vision_spec", lambda state: object())
    args = SimpleNamespace(
        launch_agent_client=dh.CLIENT_ID, agent_client_cli=False, agent_client_app=False,
        model_id="m", host="127.0.0.1", port=8123, api_key=None, agent_client_workspace="/w",
    )
    server._open_agent_client_later(SimpleNamespace(args=args))
    assert opened and opened[0]["context_window"] == 65536 and opened[0]["vision"] is True
    assert opened[0]["workspace"] == "/w"
    (fn, call), = registered
    assert fn is ac.end_desktop_copy and call[:2] == (dh.CLIENT_ID, 4343)
    out = capsys.readouterr().out
    assert "browser app" in out and "telemetry" in out


def test_ending_the_browser_app_never_touches_a_harness_it_did_not_start(tmp_path, monkeypatch):
    import sys

    overlay = tmp_path / "mtplx.patch.yml"
    overlay.write_text("[]\n")
    # Each sleeper says so once it runs: until the child has exec'd, its
    # argv is still this test process's, without the overlay path (under a
    # loaded parallel suite the check ran before the exec and failed).
    sleeper = [sys.executable, "-c", "import time; print('up', flush=True); time.sleep(30)"]
    ours = subprocess.Popen(
        [*sleeper, str(overlay)], start_new_session=True, stdout=subprocess.PIPE, text=True
    )
    theirs = subprocess.Popen(
        [*sleeper, str(tmp_path / "theirs.yml")],
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert ours.stdout.readline().strip() == "up"
        assert theirs.stdout.readline().strip() == "up"
        assert dh.is_mtplx_web_process(ours.pid, overlay)
        assert not dh.is_mtplx_web_process(theirs.pid, overlay)
        assert dh.end_web(theirs.pid, overlay) is False and theirs.poll() is None
        assert dh.end_web(ours.pid, overlay) is True
        ours.wait(timeout=5)
    finally:
        for process in (ours, theirs):
            if process.poll() is None:
                process.kill()
                process.wait()


# --- the harness itself -----------------------------------------------------------


def _installed_dsh() -> str | None:
    # Not an MTPLX_ name: the suite drops those from the environment.
    named = os.environ.get("DSH_TEST_EXECUTABLE")
    if named:
        return named
    return shutil.which("dsh")


@pytest.mark.skipif(_installed_dsh() is None, reason="DeepSeek Harness is not installed here")
def test_the_installed_harness_composes_nothing_unreviewed(tmp_path):
    """With MTPLX's overlay, the installed harness's web tree has every
    reporting piece off and no row MTPLX has not read."""

    exe = _installed_dsh()
    plan = dh.build_web_plan(
        base_url=BASE, model_id=MODEL, port=4321, context_window=65536, vision=True,
        home=tmp_path / "home", overlay=tmp_path / "mtplx.patch.yml",
    )
    checked = dh.check_composition(plan, profile=dh.WEB_PROFILE, executable=exe)
    assert checked.get("ok"), checked
    dumped = dh.dump_config(plan, profile=dh.WEB_PROFILE, executable=exe)
    rows = dh.parse_dump_rows(dumped["text"])
    assert REVIEWED_ROWS, "the reviewed rows file is empty"
    audit = dh.audit_rows(rows, reviewed=REVIEWED_ROWS)
    assert audit["ok"], audit["reporting"]
    assert audit["unreviewed"] == [], (
        "this DeepSeek Harness composes rows MTPLX has not read; read each and add it to "
        "tests/fixtures/deepseek_harness_reviewed_rows.json (or switch it off in OFF_ROWS)"
    )
    assert not (tmp_path / "home" / ".anonymous-user-id").exists()


def test_first_open_installs_privately_and_reuses_it(monkeypatch, tmp_path):
    from mtplx import deepseek_install as installer

    monkeypatch.setattr(dh, "root_dir", lambda: tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _: None)
    is_file = Path.is_file
    def first_open_file(path):
        if path.name == dh.EXECUTABLE and not path.is_relative_to(tmp_path):
            return False
        return is_file(path)
    monkeypatch.setattr(Path, "is_file", first_open_file)
    attempts = []

    def install(root, env):
        attempts.append((root, env))
        binary = installer.executable(root)
        node = installer.runtime_directory(root) / "node/bin/node"
        for path in (binary, node):
            path.parent.mkdir(exist_ok=True, parents=True)
            path.touch(mode=0o700)
        return binary

    monkeypatch.setattr(installer, "install", install)
    assert dh.resolve_executable() is None
    assert dh.ensure_executable() == str(installer.executable(tmp_path))
    assert dh.ensure_executable() == str(installer.executable(tmp_path))
    assert attempts == [(tmp_path, dh.TELEMETRY_ENV)]
    node_bin = installer.runtime_directory(tmp_path) / "node/bin"
    assert dh.launch_env(api_key=None)["PATH"].split(os.pathsep)[0] == str(node_bin)


def test_existing_harness_keeps_its_own_node(monkeypatch, tmp_path):
    from mtplx import deepseek_install as installer

    monkeypatch.setattr(dh, "root_dir", lambda: tmp_path)
    (installer.runtime_directory(tmp_path) / "node/bin").mkdir(parents=True)
    monkeypatch.setattr(shutil, "which", lambda _: "/custom/bin/dsh")
    assert dh.ensure_executable() == "/custom/bin/dsh"
    assert "PATH" not in dh.launch_env(api_key=None)
