"""Claude Code, Codex and DeepSeek Harness on the local MTPLX server: one
launch plan.

The app's Start menu, ``mtplx start claude-code|codex|deepseek-harness`` and
``mtplx connect claude-code|codex`` all open these clients from the plan
built here (the app's Swift twin is tested against
``apps/MTPLXApp/Tests/Fixtures/agent_client_plans.json``, which
``scripts/write_agent_client_plans.py`` writes from this module). DeepSeek
Harness's own profile, its telemetry switches and its browser app live in
``mtplx.deepseek_harness``.

Nothing in the user's own Claude Code or Codex setup is edited. Their usual
cloud sessions keep working exactly as before; only the session MTPLX opens
talks to the local server:

* Claude Code reads a per-launch environment plus ``--settings <file>``. The
  file is MTPLX's own (``~/.mtplx/clients/claude-code/settings.json``) and
  only its ``env`` block is set. ``--settings`` sits above the user's and the
  project's settings files, so a gateway the user configured there cannot
  silently take this session elsewhere. A gateway credential is always set
  (``mtplx-local`` when the server has no key), because a base URL without
  one keeps the claude.ai login active and would send that login to the
  local server.
* Codex gets the ``mtplx`` provider through ``-c`` overrides on its command
  line (TOML values, one run only) and the key through an environment
  variable the provider names. No file is written.

The user's own desktop apps are never pointed at MTPLX either. The Codex
desktop app opens as a separate copy on MTPLX's own folders (see "Codex
desktop app" below): its own history, a second Dock icon, the user's own copy
untouched. The Claude desktop app reads a gateway only from its app-wide
third-party inference configuration, so Claude Code opens as its terminal
program.

Pure data: no MLX import, no server import, no subprocess at import time.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from mtplx import deepseek_harness, toml_lines

AGENT_CLIENTS: tuple[str, ...] = ("claude-code", "codex", deepseek_harness.CLIENT_ID)
LOCAL_API_KEY = "mtplx-local"

# Claude Code ---------------------------------------------------------------

CLAUDE_CODE_EXECUTABLE = "claude"
# Test and power-user override for every file this module writes (the
# settings file and the Terminal scripts); the default is ~/.mtplx.
CLIENTS_DIR_ENV = "MTPLX_AGENT_CLIENTS_DIR"
# Five minutes of headers-wait per attempt would end a long cold prefill on a
# small Mac; the server streams keep-alives but the headers wait is separate.
CLAUDE_CODE_API_TIMEOUT_MS = "3000000"
# Claude Code's own answer cap for a model it does not know is 32,000 tokens;
# an xhigh answer from a reasoning model runs past that. A quarter of the
# served window keeps most of the window for the conversation.
CLAUDE_CODE_DEFAULT_OUTPUT_TOKENS = 32_000
CLAUDE_CODE_MAX_OUTPUT_TOKENS_CAP = 128_000
# Claude Code keeps MCP tools behind its tool search on Anthropic's own API,
# and against any other host sends every definition in full on every request
# unless this is set: 820 tools made a 254,636-token first prompt (12 Nov).
# The server keeps deferred tools out of the prompt and writes a tool's
# definition where Claude Code loads it.
CLAUDE_CODE_TOOL_SEARCH = "true"
# Appended to Claude Code's system prompt (`--append-system-prompt`). Its own
# prompt says only that replies are "Github-flavored markdown in a terminal";
# a local model writes math in LaTeX, which the terminal prints raw (16 Nov,
# the Jacobian conjecture answer). And a user's "without searching" held for
# one answer only: the follow-up ran two web searches. Two sentences, the same
# on every request, so the cached prefix holds.
CLAUDE_CODE_TERMINAL_NOTE = (
    "Your replies are shown in a terminal that draws Markdown but not LaTeX: write math "
    "as plain text with Unicode symbols (x², √2, ∑, ≤, →, ∂f/∂x), never with $...$, "
    "\\(...\\) or other LaTeX commands. When the user asks you not to search or not to "
    "use a tool, keep to that for the rest of the topic unless they say otherwise."
)
# Variables in the user's shell that would take this session somewhere else
# or send their real credential to the local server.
CLAUDE_CODE_UNSET_ENV: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_ANTHROPIC_AWS",
)

# Codex ---------------------------------------------------------------------

CODEX_EXECUTABLE = "codex"
CODEX_PROVIDER_ID = "mtplx"
CODEX_API_KEY_ENV = "MTPLX_API_KEY"
# Codex adds a server-hosted web_search tool by default; a local server cannot
# run it and refuses the request, so the MTPLX session turns it off.
CODEX_WEB_SEARCH = "disabled"

INSTALL_HINTS: dict[str, str] = {
    "claude-code": "Install Claude Code from https://code.claude.com/docs/en/setup",
    "codex": "Install the Codex CLI from https://developers.openai.com/codex/cli",
    deepseek_harness.CLIENT_ID: deepseek_harness.INSTALL_HINT,
}
DISPLAY_NAMES: dict[str, str] = {
    "claude-code": "Claude Code",
    "codex": "Codex",
    deepseek_harness.CLIENT_ID: deepseek_harness.DISPLAY_NAME,
}
EXECUTABLES: dict[str, str] = {
    "claude-code": "claude",
    "codex": "codex",
    deepseek_harness.CLIENT_ID: deepseek_harness.EXECUTABLE,
}


def normalize_client(name: str) -> str:
    value = str(name or "").strip().lower().replace("_", "-")
    if value in {"claude", "claude-code", "claudecode"}:
        return "claude-code"
    if value in {"codex", "codex-cli"}:
        return "codex"
    if value in {"deepseek-harness", "deepseekharness", "deepseek", "dsh"}:
        return deepseek_harness.CLIENT_ID
    raise ValueError(
        f"unknown agent client {name!r}; choose claude-code, codex or deepseek-harness"
    )


def claude_code_output_tokens(context_window: int | None) -> int | None:
    if not context_window or int(context_window) <= 0:
        return None
    return max(
        CLAUDE_CODE_DEFAULT_OUTPUT_TOKENS,
        min(CLAUDE_CODE_MAX_OUTPUT_TOKENS_CAP, int(context_window) // 4),
    )


def claude_code_env(
    *,
    base_url: str,
    model_id: str,
    context_window: int | None,
    api_key: str | None,
    tool_search: bool = True,
) -> dict[str, str]:
    """``tool_search``: the server can write a tool Claude Code loads in the
    served template's own tool encoding (its health's
    ``anthropic_tool_search``); off, every tool is sent up front."""
    env = {
        "ANTHROPIC_BASE_URL": base_url.rstrip("/"),
        "ANTHROPIC_AUTH_TOKEN": api_key or LOCAL_API_KEY,
        "ANTHROPIC_MODEL": model_id,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model_id,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model_id,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model_id,
        "CLAUDE_CODE_SUBAGENT_MODEL": model_id,
        "API_TIMEOUT_MS": CLAUDE_CODE_API_TIMEOUT_MS,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "ENABLE_TOOL_SEARCH": CLAUDE_CODE_TOOL_SEARCH if tool_search else "false",
    }
    if context_window and int(context_window) > 0:
        env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(int(context_window))
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(claude_code_output_tokens(context_window))
    return env


def codex_config_overrides(
    *, base_url: str, context_window: int | None
) -> list[str]:
    """The ``-c key=value`` overrides, as TOML values, in a fixed order."""

    api_base = base_url.rstrip("/") + "/v1"
    provider = f"model_providers.{CODEX_PROVIDER_ID}"
    overrides = [
        f'model_provider="{CODEX_PROVIDER_ID}"',
        f'{provider}.name="MTPLX"',
        f"{provider}.base_url={json.dumps(api_base)}",
        f'{provider}.wire_api="responses"',
        f'{provider}.env_key="{CODEX_API_KEY_ENV}"',
        f"{provider}.requires_openai_auth=false",
        f'{provider}.http_headers={{ "x-mtplx-client" = "codex" }}',
        f'web_search="{CODEX_WEB_SEARCH}"',
    ]
    if context_window and int(context_window) > 0:
        overrides.append(f"model_context_window={int(context_window)}")
    return overrides


def build_plan(
    client: str,
    *,
    base_url: str,
    model_id: str,
    context_window: int | None = None,
    api_key: str | None = None,
    settings_path: str | None = None,
    executable: str | None = None,
    tool_search: bool = True,
    vision: bool = False,
) -> dict[str, Any]:
    """The launch plan: argv, env to set, env to remove, files to write.

    ``api_key`` is the key the server requires (None for an open localhost
    server). It travels in the environment or in an MTPLX-owned 0600 file,
    never on the command line. ``vision``: the served model reads images
    (DeepSeek Harness offers image input only then; the other two clients
    learn it from the server).
    """

    client = normalize_client(client)
    if client == deepseek_harness.CLIENT_ID:
        return deepseek_harness.build_terminal_plan(
            base_url=base_url,
            model_id=model_id,
            context_window=context_window,
            api_key=api_key,
            vision=vision,
            executable=executable,
        )
    if client == "claude-code":
        env = claude_code_env(
            base_url=base_url,
            model_id=model_id,
            context_window=context_window,
            api_key=api_key,
            tool_search=tool_search,
        )
        settings = str(settings_path or default_claude_code_settings_path())
        return {
            "client": client,
            "argv": [
                executable or CLAUDE_CODE_EXECUTABLE,
                "--settings",
                settings,
                "--append-system-prompt",
                CLAUDE_CODE_TERMINAL_NOTE,
            ],
            "env": env,
            "unset_env": list(CLAUDE_CODE_UNSET_ENV),
            "files": {settings: json.dumps({"env": env}, indent=2, sort_keys=True) + "\n"},
        }
    argv = [executable or CODEX_EXECUTABLE]
    for override in codex_config_overrides(base_url=base_url, context_window=context_window):
        argv.extend(["-c", override])
    argv.extend(["-m", model_id])
    return {
        "client": client,
        "argv": argv,
        "env": {CODEX_API_KEY_ENV: api_key or LOCAL_API_KEY},
        "unset_env": [],
        "files": {},
    }


def clients_dir() -> Path:
    override = os.environ.get(CLIENTS_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".mtplx" / "clients"


def handoff_dir() -> Path:
    override = os.environ.get(CLIENTS_DIR_ENV)
    if override:
        return Path(override).expanduser() / "handoffs"
    return Path.home() / ".mtplx" / "handoffs"


def default_claude_code_settings_path() -> Path:
    return clients_dir() / "claude-code" / "settings.json"


def resolve_executable(client: str) -> str | None:
    """The copy the user's own terminal would run, else a known install path."""

    client = normalize_client(client)
    if client == deepseek_harness.CLIENT_ID:
        return deepseek_harness.resolve_executable()
    name = EXECUTABLES[client]
    found = shutil.which(name)
    if found:
        return found
    home = Path.home()
    for candidate in (
        home / ".local" / "bin" / name,
        Path("/opt/homebrew/bin") / name,
        Path("/usr/local/bin") / name,
        home / ".npm-global" / "bin" / name,
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def write_plan_files(plan: dict[str, Any]) -> list[str]:
    """Write the plan's MTPLX-owned files (0600). Returns their paths."""

    written: list[str] = []
    for raw_path, text in (plan.get("files") or {}).items():
        path = Path(raw_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(path, 0o600)
        written.append(str(path))
    return written


def shell_command(plan: dict[str, Any], *, workspace: str | None = None) -> str:
    """One line a user can paste in their own terminal (secrets included)."""

    parts: list[str] = []
    if workspace:
        parts.append(f"cd {shlex.quote(workspace)} &&")
    parts.append("env")
    for name in plan.get("unset_env") or ():
        parts.extend(["-u", name])
    for name, value in sorted((plan.get("env") or {}).items()):
        parts.append(f"{name}={shlex.quote(str(value))}")
    parts.append(" ".join(shlex.quote(str(arg)) for arg in plan["argv"]))
    return " ".join(parts)


def display_command(plan: dict[str, Any]) -> str:
    """The command with credentials masked, for logs and dry runs."""

    masked = dict(plan.get("env") or {})
    for name in ("ANTHROPIC_AUTH_TOKEN", CODEX_API_KEY_ENV):
        if name in masked and masked[name] != LOCAL_API_KEY:
            masked[name] = "<api key>"
    return shell_command({**plan, "env": masked})


def terminal_script(plan: dict[str, Any], *, workspace: str, executable: str) -> str:
    """The zsh script Terminal runs: cd, set the environment, exec the client.

    The executable is resolved when the script runs, so the user's own PATH
    picks the copy they would get by typing the name, and a missing client
    prints how to install it instead of an opaque error.
    """

    name = plan["argv"][0]
    lines = ["#!/bin/zsh", f"cd {shlex.quote(workspace)} || exit 1"]
    for unset in plan.get("unset_env") or ():
        lines.append(f"unset {unset}")
    for key, value in sorted((plan.get("env") or {}).items()):
        lines.append(f"export {key}={shlex.quote(str(value))}")
    lines.append(f'_mtplx_bin="$(command -v {shlex.quote(name)} 2>/dev/null)"')
    lines.append(f'[[ -z "$_mtplx_bin" && -x {shlex.quote(executable)} ]] && _mtplx_bin={shlex.quote(executable)}')
    hint = INSTALL_HINTS[plan["client"]]
    lines.append(
        f'if [[ -z "$_mtplx_bin" ]]; then print -r -- {shlex.quote(hint)}; exit 127; fi'
    )
    lines.append(
        'exec "$_mtplx_bin" ' + " ".join(shlex.quote(str(arg)) for arg in plan["argv"][1:])
    )
    return "\n".join(lines) + "\n"


def open_in_terminal(
    plan: dict[str, Any], *, workspace: str, handoff_dir_override: Path | None = None
) -> dict[str, Any]:
    """Write the plan's files and a 0700 script, then open it in the terminal
    app the user chose for this client (Terminal by default). ``terminal`` in
    the result names the app, ``note`` says why it is not the saved choice."""

    from mtplx import client_opening

    if sys.platform != "darwin":
        return {"ok": False, "error": "opening a terminal client needs macOS"}
    executable = resolve_executable(plan["client"]) or plan["argv"][0]
    opening = client_opening.terminal_for(plan["client"], client_opening.saved_choice(plan["client"]))
    for folder in plan.get("folders") or ():
        Path(folder).mkdir(parents=True, exist_ok=True, mode=0o700)
    if plan["client"] == deepseek_harness.CLIENT_ID:
        deepseek_harness.prepare_home(plan["env"]["DSH_HOME"])
    write_plan_files(plan)
    directory = handoff_dir_override or handoff_dir()
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / f"open-{plan['client']}-{uuid.uuid4().hex}.command"
    fd = os.open(script, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o700)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(terminal_script(plan, workspace=workspace, executable=executable))
    result = client_opening.open_script(script, opening.terminal)
    return {
        **result,
        "script": str(script),
        "executable": executable,
        "terminal": opening.terminal_name,
        "note": opening.note,
    }


# Codex desktop app: an isolated MTPLX instance ---------------------------------
#
# The Codex desktop app (bundle com.openai.codex, shipped as ChatGPT.app) has
# its own isolated-launch path: started with `open -n` and the environment
# variables CODEX_HOME and CODEX_ELECTRON_USER_DATA_PATH plus a matching
# --user-data-dir, it runs as a separate instance on those folders, beside the
# user's own instance, and leaves ~/.codex and its normal data folder alone.
# MTPLX owns both folders (~/.mtplx/clients/codex-desktop/). Proven on
# 26.1106.51428 on 12 Nov 2026 (receipts/codex-desktop-isolation-0553): both
# folders honoured, two instances side by side, no sign-in screen with the
# mtplx provider, the user's folders unchanged. An older build may not know
# the user-data variable, so it gets the Codex CLI in Terminal instead.

CODEX_DESKTOP_BUNDLE_ID = "com.openai.codex"
CODEX_DESKTOP_MIN_VERSION: tuple[int, ...] = (26, 1106)
CODEX_DESKTOP_APP_NAMES: tuple[str, ...] = ("ChatGPT.app", "Codex.app")
# Top-level keys and the provider table MTPLX writes; everything else in the
# instance's config.toml (the app adds plugins, MCP servers and a notify hook
# at start, the user may add more) is kept as it is.
CODEX_DESKTOP_OWNED_KEYS: tuple[str, ...] = (
    "model",
    "model_provider",
    "web_search",
    "model_context_window",
)
CODEX_DESKTOP_PROVIDER_TABLE = f"model_providers.{CODEX_PROVIDER_ID}"
# The Codex app's composer starts a custom model at "Light" (low effort).
# MTPLX's coding-agent lanes default to medium (Flash-Next 2026-08-28 A/B:
# the same correct multifile answer in 44 s at medium against 150 s at
# xhigh), so a new copy starts there; once the line exists it is the user's
# and is never rewritten.
CODEX_DESKTOP_DEFAULT_EFFORT = "medium"
CODEX_DESKTOP_CONFIG_HEADER = (
    "# MTPLX sets model, model_provider, web_search, model_context_window and "
    "[model_providers.mtplx] on each launch; everything else here is kept."
)


def _version_tuple(text: str | None) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in str(text or "").split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def find_codex_desktop_app(search_dirs: tuple[Path, ...] | None = None) -> dict[str, Any] | None:
    """The installed Codex desktop app: its path, version and whether MTPLX can
    open it as an isolated instance (``supported``)."""

    import plistlib

    roots = search_dirs or (Path("/Applications"), Path.home() / "Applications")
    for root in roots:
        for name in CODEX_DESKTOP_APP_NAMES:
            bundle = Path(root) / name
            try:
                with (bundle / "Contents" / "Info.plist").open("rb") as handle:
                    info = plistlib.load(handle)
            except (OSError, ValueError, plistlib.InvalidFileException):
                continue
            if info.get("CFBundleIdentifier") != CODEX_DESKTOP_BUNDLE_ID:
                continue
            version = str(info.get("CFBundleShortVersionString") or "")
            executable = str(info.get("CFBundleExecutable") or "")
            return {
                "path": str(bundle),
                "version": version,
                # The main process is how MTPLX finds and closes its copy.
                "supported": bool(executable)
                and _version_tuple(version) >= CODEX_DESKTOP_MIN_VERSION,
                "executable": str(bundle / "Contents" / "MacOS" / executable),
            }
    return None


def codex_desktop_root() -> Path:
    return clients_dir() / "codex-desktop"


def codex_desktop_home() -> Path:
    return clients_dir() / "codex-desktop" / "home"


def codex_desktop_user_data() -> Path:
    return clients_dir() / "codex-desktop" / "user-data"


def _is_owned_table(table: str | None) -> bool:
    return table == CODEX_DESKTOP_PROVIDER_TABLE or str(table or "").startswith(
        CODEX_DESKTOP_PROVIDER_TABLE + "."
    )


def codex_desktop_config_text(
    existing: str | None,
    *,
    base_url: str,
    model_id: str,
    context_window: int | None,
    api_key: str | None,
) -> str:
    """The instance's config.toml with MTPLX's keys set and the rest kept.

    MTPLX's top-level keys go first, then every other top-level line as found,
    then every other table as found, then ``[model_providers.mtplx]``. An
    open localhost server needs no credential; a server with a key gets it as
    the provider's bearer token inside this 0600 file, never on a command
    line.
    """

    owned = [
        f"model = {json.dumps(model_id)}",
        f'model_provider = "{CODEX_PROVIDER_ID}"',
        f'web_search = "{CODEX_WEB_SEARCH}"',
    ]
    if context_window and int(context_window) > 0:
        owned.append(f"model_context_window = {int(context_window)}")
    provider = [
        f"[{CODEX_DESKTOP_PROVIDER_TABLE}]",
        'name = "MTPLX"',
        f"base_url = {json.dumps(base_url.rstrip('/') + '/v1')}",
        'wire_api = "responses"',
        "requires_openai_auth = false",
        'http_headers = { "x-mtplx-client" = "codex" }',
    ]
    if api_key:
        provider.append(f"experimental_bearer_token = {json.dumps(api_key)}")

    top: list[str] = []
    tables: list[str] = []
    has_effort = False
    for table, key, block in toml_lines.statements(existing or ""):
        if table == "":
            if key in CODEX_DESKTOP_OWNED_KEYS or block == [CODEX_DESKTOP_CONFIG_HEADER]:
                continue
            has_effort = has_effort or key == "model_reasoning_effort"
            top.extend(block)
        elif not _is_owned_table(table):
            tables.extend(block)
    if not has_effort:
        owned.append(f'model_reasoning_effort = "{CODEX_DESKTOP_DEFAULT_EFFORT}"')
    while top and not top[0].strip():
        top.pop(0)
    while top and not top[-1].strip():
        top.pop()
    while tables and not tables[-1].strip():
        tables.pop()
    out = [CODEX_DESKTOP_CONFIG_HEADER, *owned, *top, ""]
    if tables:
        out += tables + [""]
    out += provider
    return "\n".join(out) + "\n"


def build_codex_desktop_plan(
    *,
    app_path: str,
    base_url: str,
    model_id: str,
    context_window: int | None = None,
    api_key: str | None = None,
    existing_config: str | None = None,
    home: str | Path | None = None,
    user_data: str | Path | None = None,
) -> dict[str, Any]:
    """The isolated-instance launch: one ``open -n`` and one config file."""

    home_dir = str(home or codex_desktop_home())
    data_dir = str(user_data or codex_desktop_user_data())
    config_path = str(Path(home_dir) / "config.toml")
    return {
        "client": "codex",
        "surface": "desktop",
        "argv": [
            "/usr/bin/open",
            "-n",
            "--env",
            f"CODEX_HOME={home_dir}",
            "--env",
            f"CODEX_ELECTRON_USER_DATA_PATH={data_dir}",
            app_path,
            "--args",
            f"--user-data-dir={data_dir}",
        ],
        "env": {},
        "unset_env": [],
        "files": {
            config_path: codex_desktop_config_text(
                existing_config,
                base_url=base_url,
                model_id=model_id,
                context_window=context_window,
                api_key=api_key,
            )
        },
        "folders": [home_dir, data_dir],
    }


# The running copy: the app's main process with MTPLX's own --user-data-dir in
# its argv, read from the kernel's argv image (KERN_PROCARGS2), never from
# `ps` text. The user's own Codex app never carries that argument, so closing
# the copy can never close theirs. Mirrored by CodexDesktopIntegration.swift.


def _parse_procargs(raw: bytes) -> tuple[str, list[str]] | None:
    """Exec path and argv from a KERN_PROCARGS2 image (argc, exec path,
    padding, argc NUL-terminated strings, then the environment, not read)."""

    if len(raw) < 4:
        return None
    argc = int.from_bytes(raw[:4], "little", signed=True)
    if argc < 0:
        return None
    cursor = 4

    def take() -> str | None:
        nonlocal cursor
        end = raw.find(b"\0", cursor)
        if end < 0:
            return None
        text = raw[cursor:end].decode("utf-8", "replace")
        cursor = end + 1
        return text

    executable = take()
    if executable is None:
        return None
    while cursor < len(raw) and raw[cursor] == 0:
        cursor += 1
    argv: list[str] = []
    for _ in range(argc):
        argument = take()
        if argument is None:
            return None
        argv.append(argument)
    return executable, argv


def _process_arguments(pid: int) -> tuple[str, list[str]] | None:
    if sys.platform != "darwin" or int(pid) <= 1:
        return None
    try:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        ctl_kern, kern_procargs2 = 1, 49
        mib = (ctypes.c_int * 3)(ctl_kern, kern_procargs2, int(pid))
        size = ctypes.c_size_t(0)
        if libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0:
            return None
        buffer = ctypes.create_string_buffer(max(1, size.value))
        if libc.sysctl(mib, 3, buffer, ctypes.byref(size), None, 0) != 0:
            return None
    except (OSError, AttributeError, ValueError):
        return None
    return _parse_procargs(buffer.raw[: size.value])


def is_codex_desktop_instance(
    pid: int,
    executable: str,
    user_data: str | Path | None = None,
    *,
    read_arguments: Any = None,
) -> bool:
    """True while ``pid`` is the app's main process on MTPLX's folder."""

    image = (read_arguments or _process_arguments)(int(pid))
    if not image:
        return False
    marker = f"--user-data-dir={user_data or codex_desktop_user_data()}"
    return os.path.realpath(image[0]) == os.path.realpath(executable) and marker in image[1][1:]


def codex_desktop_instances(
    app: dict[str, Any], user_data: str | Path | None = None
) -> list[int]:
    """Pids of MTPLX's running copies of the app (normally one)."""

    executable = str(app.get("executable") or "")
    if sys.platform != "darwin" or not executable:
        return []
    try:
        listing = subprocess.run(
            ["/bin/ps", "-axww", "-o", "pid=,comm="],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    pids: list[int] = []
    for line in listing.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        # `comm` is only a filter; the kernel's argv image decides.
        if os.path.basename(parts[1]) != os.path.basename(executable):
            continue
        if is_codex_desktop_instance(int(parts[0]), executable, user_data):
            pids.append(int(parts[0]))
    return pids


def end_codex_desktop_instance(
    pid: int,
    app: dict[str, Any],
    user_data: str | Path | None = None,
    *,
    grace_s: float = 3.0,
) -> bool:
    """Close one copy MTPLX opened: SIGTERM (it closes its windows and
    databases), SIGKILL only if it is still that same process after
    ``grace_s``. Never signals a process that is not the copy."""

    executable = str(app.get("executable") or "")
    if not executable or not is_codex_desktop_instance(pid, executable, user_data):
        return False
    try:
        os.kill(int(pid), signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not is_codex_desktop_instance(pid, executable, user_data):
            return True
        time.sleep(0.1)
    if is_codex_desktop_instance(pid, executable, user_data):
        try:
            os.kill(int(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    return True


def open_codex_desktop(
    *,
    base_url: str,
    model_id: str,
    context_window: int | None,
    api_key: str | None,
    app: dict[str, Any] | None = None,
    instance_timeout_s: float = 15.0,
) -> dict[str, Any]:
    """Open the Codex desktop app as MTPLX's own instance. Never ~/.codex.

    A copy MTPLX opened earlier is closed first, so the new one starts on
    this server's address, model and window. ``pid`` in the result is the
    new copy's main process (None if it did not appear in time).
    """

    if sys.platform != "darwin":
        return {"ok": False, "error": "the Codex desktop app needs macOS"}
    app = app or find_codex_desktop_app()
    if not app or not app.get("supported"):
        return {"ok": False, "error": "no supported Codex desktop app", "app": app}
    closed = []
    for pid in codex_desktop_instances(app):
        if end_codex_desktop_instance(pid, app):
            closed.append(pid)
    config_path = codex_desktop_home() / "config.toml"
    try:
        existing = config_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        existing = None
    except (OSError, UnicodeDecodeError) as exc:
        return {"ok": False, "error": f"could not read {config_path}: {exc}"}
    plan = build_codex_desktop_plan(
        app_path=app["path"],
        base_url=base_url,
        model_id=model_id,
        context_window=context_window,
        api_key=api_key,
        existing_config=existing,
    )
    for folder in plan["folders"]:
        Path(folder).mkdir(parents=True, exist_ok=True, mode=0o700)
    if existing is not None:
        backup = config_path.with_name("config.toml.mtplx-previous")
        fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(existing)
    write_plan_files(plan)
    try:
        completed = subprocess.run(
            plan["argv"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc), "plan": plan}
    if completed.returncode != 0:
        return {
            "ok": False,
            "error": (completed.stderr or "").strip() or f"open exited {completed.returncode}",
            "plan": plan,
        }
    # `open` returns once LaunchServices has the request; the copy's main
    # process appears a moment later.
    deadline = time.monotonic() + instance_timeout_s
    pid = None
    while pid is None:
        found = codex_desktop_instances(app)
        pid = found[0] if found else None
        if pid is not None or time.monotonic() >= deadline:
            break
        time.sleep(0.25)
    return {"ok": True, "plan": plan, "app": app, "pid": pid, "closed_previous": closed}


def folder_size_bytes(path: Path) -> int:
    """Bytes under ``path`` (files only, links not followed); 0 if absent."""

    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                continue
    return total


def client_copies() -> list[dict[str, Any]]:
    """The client folders MTPLX keeps beside the user's own apps, for
    ``mtplx clients``: where each is, its size and whether it is open."""

    rows: list[dict[str, Any]] = []
    root = codex_desktop_root()
    app = find_codex_desktop_app()
    rows.append(
        {
            "name": "codex-desktop",
            "description": "the Codex app's MTPLX copy: its own history, settings and plugins",
            "path": str(root),
            "exists": root.is_dir(),
            "bytes": folder_size_bytes(root) if root.is_dir() else 0,
            "running_pids": codex_desktop_instances(app) if app else [],
        }
    )
    harness = deepseek_harness.root_dir()
    if harness.is_dir():
        rows.append(
            {
                "name": "deepseek-harness",
                "description": "DeepSeek Harness's MTPLX home: its sessions, settings and the profile MTPLX writes",
                "path": str(harness),
                "exists": True,
                "bytes": folder_size_bytes(harness),
                "running_pids": [],
            }
        )
    return rows


def desktop_app_for(
    client: str, *, cli: bool = False, choice: str | None = None, app: bool = False
) -> dict[str, Any] | None:
    """The app ``mtplx start <client>`` opens, or None for the client's CLI
    in a terminal: ``--cli``, a terminal chosen in ``<client>_opens_in``, or
    no app MTPLX can open. ``app`` (``--app``) asks for the app whatever the
    setting says. ``choice`` is the saved setting
    (``client_opening.saved_choice``). Codex's default is MTPLX's own copy of
    its app, DeepSeek Harness's its browser app, Claude Code's the terminal."""

    from mtplx import client_opening

    if cli or client not in AGENT_CLIENTS:
        return None
    # Only the surface matters here, so no terminal scan.
    surface = client_opening.resolve(client, choice, []).surface
    wanted = client_opening.CLIENTS_BY_ID[client].app_surfaces
    if not app and surface not in wanted:
        return None
    if client == deepseek_harness.CLIENT_ID:
        executable = deepseek_harness.resolve_executable()
        return {"kind": "web", "path": executable} if executable else None
    if client == "codex":
        found = find_codex_desktop_app()
        return found if found and found.get("supported") else None
    return None


def open_desktop_copy(
    client: str,
    app: dict[str, Any],
    *,
    base_url: str,
    model_id: str,
    context_window: int | None,
    api_key: str | None,
    vision: bool = False,
    workspace: str | None = None,
) -> dict[str, Any]:
    """Open the app ``desktop_app_for(client)`` found, on this server."""

    if client == deepseek_harness.CLIENT_ID:
        return open_deepseek_harness_web(
            base_url=base_url,
            model_id=model_id,
            context_window=context_window,
            api_key=api_key,
            vision=vision,
            workspace=workspace or os.getcwd(),
        )
    return open_codex_desktop(
        base_url=base_url,
        model_id=model_id,
        context_window=context_window,
        api_key=api_key,
        app=app,
    )


def end_desktop_copy(client: str, pid: int, app: dict[str, Any]) -> bool:
    """Close the app copy MTPLX opened for ``client`` (when its server stops)."""

    if client == deepseek_harness.CLIENT_ID:
        return deepseek_harness.end_web(int(pid))
    if client == "codex":
        return end_codex_desktop_instance(int(pid), app)
    return False


def open_deepseek_harness_web(
    *,
    base_url: str,
    model_id: str,
    context_window: int | None,
    api_key: str | None,
    vision: bool,
    workspace: str,
) -> dict[str, Any]:
    """DeepSeek Harness's browser app on this server: its composed tree is
    checked first (nothing that reports to the vendor may be on), then
    ``dsh web`` starts in the background and opens the browser itself."""

    plan = deepseek_harness.build_web_plan(
        base_url=base_url,
        model_id=model_id,
        port=deepseek_harness.free_local_port(),
        context_window=context_window,
        api_key=api_key,
        vision=vision,
    )
    checked = deepseek_harness.check_composition(plan, profile=deepseek_harness.WEB_PROFILE)
    if not checked.get("ok"):
        return {"ok": False, "error": checked.get("error"), "audit": checked, "plan": plan}
    started = deepseek_harness.start_web(plan, workspace=workspace)
    return {**started, "plan": plan, "audit": checked}


def desktop_copy_root(client: str) -> Path:
    if client == deepseek_harness.CLIENT_ID:
        return deepseek_harness.root_dir()
    return codex_desktop_root()


# The fixture the app's Swift twin is compared against ------------------------

FIXTURE_CASES: tuple[dict[str, Any], ...] = (
    {"base_url": "http://127.0.0.1:8000", "model_id": "mtplx-qwen38-flash-next-optimized-speed", "context_window": 262144, "api_key": None},
    {"base_url": "http://127.0.0.1:8123", "model_id": "mtplx-qwen38-27b-optimized-speed", "context_window": 65536, "api_key": "sk-local-test"},
    {"base_url": "http://127.0.0.1:8000", "model_id": "mtplx-qwen35-4b-speed", "context_window": None, "api_key": None},
)
FIXTURE_SETTINGS_PATH = "<settings>"


# Configs the Codex desktop merge must handle: none yet, the file the app
# wrote on its first start in the 12 Nov test (provider block kept, plugins
# and an MCP server added), a user's own provider and settings, multi-line
# values with brackets and quotes, and an old provider written as subtables.
FIXTURE_CODEX_DESKTOP_CONFIGS: tuple[str | None, ...] = (
    None,
    'model = "mtplx-mock-model"\nmodel_provider = "mtplx"\nweb_search = "disabled"\n'
    "model_context_window = 65536\n\n"
    'notify = ["/x/SkyComputerUseClient", "turn-ended"]\n\n'
    '[model_providers.mtplx]\nname = "MTPLX"\nbase_url = "http://127.0.0.1:51738/v1"\n'
    'wire_api = "responses"\nenv_key = "MTPLX_API_KEY"\nrequires_openai_auth = false\n'
    'http_headers = { "x-mtplx-client" = "codex" }\n\n'
    '[plugins."browser@openai-bundled"]\nenabled = true\n\n'
    '[mcp_servers.node_repl]\nargs = []\ncommand = "/Applications/ChatGPT.app/node_repl"\n\n'
    '[mcp_servers.node_repl.env]\nCODEX_HOME = "/x/home"\n',
    '# my settings\nmodel_reasoning_effort = "high"\nmodel = "gpt-old"\n\n'
    '[model_providers.other]\nname = "Other"\nbase_url = "https://example.invalid/v1"\n',
    'approval_policy = "never"\nnotify = [\n  "a[1]",\n  ["nested", "]"],\n  """x\n[not.a.table]\n"""\n]\n'
    '[tui]\nnotifications = true\n',
    'model_provider = "mtplx"\n[model_providers.mtplx]\nname = "Old"\n\n'
    '[model_providers.mtplx.http_headers]\nX-Old = "1"\n\n[history]\npersistence = "none"\n',
)


def app_parity_fixture() -> dict[str, Any]:
    rows = []
    for case in FIXTURE_CASES:
        # The clients the app builds plans for itself; it opens DeepSeek
        # Harness through `mtplx connect deepseek-harness`, so its profile and
        # audit have one owner.
        for client in ("claude-code", "codex"):
            plan = build_plan(client, settings_path=FIXTURE_SETTINGS_PATH, **case)
            rows.append({"inputs": {"client": client, **case}, "plan": plan})
    desktop_rows = []
    for existing in FIXTURE_CODEX_DESKTOP_CONFIGS:
        for case in FIXTURE_CASES:
            plan = build_codex_desktop_plan(
                app_path="/Applications/ChatGPT.app",
                existing_config=existing,
                home="<home>",
                user_data="<user-data>",
                **case,
            )
            desktop_rows.append(
                {"inputs": {**case, "existing_config": existing}, "plan": plan}
            )
    return {
        "version": 2,
        "generated_by": "scripts/write_agent_client_plans.py",
        "local_api_key": LOCAL_API_KEY,
        "codex_desktop_bundle_id": CODEX_DESKTOP_BUNDLE_ID,
        "codex_desktop_min_version": list(CODEX_DESKTOP_MIN_VERSION),
        "rows": rows,
        "codex_desktop_rows": desktop_rows,
    }
