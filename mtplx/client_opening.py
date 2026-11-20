"""Where each agent client opens: a terminal app the user picks, or an app.

One setting per client, shared by the app's Settings and the CLI through
``~/.mtplx/config.toml`` (``pi_opens_in = "Ghostty"``, ``codex_opens_in =
"mtplx-app"``). A terminal is any installed app that runs a ``.command``
script: Terminal, and Ghostty, iTerm, Warp or whichever others the user has,
found from what each app declares in its Info.plist, never from a list of
names. With no choice saved, a client opens where it always has.

The app's Swift twin (``ClientOpening.swift``) reads the same file the same
way; ``apps/MTPLXApp/Tests/Fixtures/client_opening_cases.json``, written from
``app_parity_fixture`` here, keeps the two in step.

Pure data and one ``open`` call: no MLX import, nothing run at import time.
"""

from __future__ import annotations

import os
import plistlib
import shlex
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

APPLE_TERMINAL_BUNDLE_ID = "com.apple.Terminal"
# A ``.command`` file is this content type. An app that declares it with the
# Shell role runs the script in a new window, as Terminal does; an editor
# declares it as Editor or Viewer and is not a terminal.
SHELL_SCRIPT_TYPE = "com.apple.terminal.shell-script"
TERMINAL_SEARCH_DIRS: tuple[Path, ...] = (
    Path("/System/Applications/Utilities"),
    Path("/Applications"),
    Path("/Applications/Utilities"),
    Path.home() / "Applications",
)

TERMINAL = "terminal"  # Terminal, the terminal every Mac has
MTPLX_APP = "mtplx-app"  # MTPLX's own copy of the client's desktop app
OWN_APP = "app"  # the client's own desktop app (OpenCode, Hermes)


@dataclass(frozen=True)
class Client:
    id: str
    name: str
    config_key: str
    app_surfaces: tuple[str, ...]  # what it can open as besides a terminal
    default: str  # where it opens with no choice saved
    app_name: str | None = None  # its desktop app, when it opens as one


CLIENTS: tuple[Client, ...] = (
    Client("pi", "Pi", "pi_opens_in", (), TERMINAL),
    Client("opencode", "OpenCode", "opencode_opens_in", (OWN_APP,), OWN_APP, "OpenCode"),
    Client("hermes", "Hermes", "hermes_opens_in", (OWN_APP,), OWN_APP, "Hermes"),
    Client("claude-code", "Claude Code", "claude_code_opens_in", (), TERMINAL),
    Client("codex", "Codex", "codex_opens_in", (MTPLX_APP,), MTPLX_APP, "Codex"),
    # Its app is its browser app (`dsh web`), which MTPLX starts and stops.
    Client(
        "deepseek-harness",
        "DeepSeek Harness",
        "deepseek_harness_opens_in",
        (OWN_APP,),
        OWN_APP,
        "DeepSeek Harness web",
    ),
)
CLIENTS_BY_ID: dict[str, Client] = {client.id: client for client in CLIENTS}
CONFIG_KEYS: dict[str, str] = {client.config_key: client.id for client in CLIENTS}


@dataclass(frozen=True)
class TerminalApp:
    bundle_id: str
    name: str
    path: str

    def matches(self, choice: str) -> bool:
        """A choice names a terminal by bundle id or app name, any case."""

        wanted = choice.strip().casefold()
        return wanted in (self.bundle_id.casefold(), self.name.casefold())


def terminal_from_info(info: Mapping[str, Any], path: str) -> TerminalApp | None:
    """The terminal an app bundle's Info.plist describes, or None when the
    app does not run shell scripts."""

    bundle_id = str(info.get("CFBundleIdentifier") or "")
    if not bundle_id:
        return None
    runs_scripts = any(
        isinstance(kind, dict)
        and kind.get("CFBundleTypeRole") == "Shell"
        and SHELL_SCRIPT_TYPE in (kind.get("LSItemContentTypes") or ())
        for kind in info.get("CFBundleDocumentTypes") or ()
    )
    if not runs_scripts:
        return None
    name = str(info.get("CFBundleDisplayName") or info.get("CFBundleName") or Path(path).stem)
    return TerminalApp(bundle_id=bundle_id, name=name, path=path)


def terminal_from_bundle(bundle: Path) -> TerminalApp | None:
    try:
        with (bundle / "Contents" / "Info.plist").open("rb") as handle:
            info = plistlib.load(handle)
    except (OSError, ValueError, plistlib.InvalidFileException):
        return None
    return terminal_from_info(info, str(bundle)) if isinstance(info, dict) else None


def sorted_terminals(terminals: Iterable[TerminalApp]) -> list[TerminalApp]:
    """Terminal first, the rest by name; one entry per bundle id (the first
    one seen wins)."""

    unique: dict[str, TerminalApp] = {}
    for terminal in terminals:
        unique.setdefault(terminal.bundle_id, terminal)
    return sorted(
        unique.values(),
        key=lambda t: (t.bundle_id != APPLE_TERMINAL_BUNDLE_ID, t.name.casefold()),
    )


def installed_terminals(search_dirs: Iterable[Path] | None = None) -> list[TerminalApp]:
    """Every installed app that runs a ``.command`` script."""

    found: list[TerminalApp] = []
    for root in search_dirs or TERMINAL_SEARCH_DIRS:
        try:
            bundles = sorted(Path(root).glob("*.app"))
        except OSError:
            continue
        found.extend(t for t in map(terminal_from_bundle, bundles) if t is not None)
    return sorted_terminals(found)


@dataclass(frozen=True)
class Opening:
    """Where one launch of a client goes, and why when it is not the choice."""

    client: str
    surface: str  # TERMINAL, MTPLX_APP or OWN_APP
    terminal: TerminalApp | None = None  # None: Terminal by name
    note: str | None = None

    @property
    def terminal_name(self) -> str:
        return self.terminal.name if self.terminal is not None else "Terminal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "client": self.client,
            "surface": self.surface,
            "terminal": None if self.terminal is None else asdict(self.terminal),
            "note": self.note,
        }


def _apple_terminal(terminals: list[TerminalApp]) -> TerminalApp | None:
    return next((t for t in terminals if t.bundle_id == APPLE_TERMINAL_BUNDLE_ID), None)


def resolve(
    client_id: str,
    choice: str | None,
    terminals: list[TerminalApp] | None = None,
) -> Opening:
    """The opening for ``client_id`` from its saved choice.

    No choice is the client's default. A terminal that is no longer installed
    opens Terminal and says so, so a launch never fails over a setting.
    """

    client = CLIENTS_BY_ID[client_id]
    value = (choice or "").strip() or client.default
    if value in client.app_surfaces:
        return Opening(client.id, value)
    terminals = installed_terminals() if terminals is None else terminals
    fallback = _apple_terminal(terminals)
    if value == TERMINAL:
        return Opening(client.id, TERMINAL, fallback)
    if value in (MTPLX_APP, OWN_APP):
        return Opening(client.id, TERMINAL, fallback, f"{client.name} has no app to open; opening Terminal")
    chosen = next((t for t in terminals if t.matches(value)), None)
    if chosen is not None:
        return Opening(client.id, TERMINAL, chosen)
    return Opening(client.id, TERMINAL, fallback, f"{value} is not installed; opening Terminal")


def terminal_for(
    client_id: str,
    choice: str | None,
    terminals: list[TerminalApp] | None = None,
) -> Opening:
    """The terminal a client's command runs in: the one the user chose, else
    Terminal (no choice, or an app that this caller is not opening)."""

    terminals = installed_terminals() if terminals is None else terminals
    opening = resolve(client_id, choice, terminals)
    if opening.surface == TERMINAL:
        return opening
    return Opening(client_id, TERMINAL, _apple_terminal(terminals))


def saved_choice(client_id: str, config: Any | None = None) -> str | None:
    """The client's choice from the user's config (``load_user_config``)."""

    if config is None:
        from mtplx.config import load_user_config

        config = load_user_config()
    value = str(getattr(config, CLIENTS_BY_ID[client_id].config_key, None) or "").strip()
    return value or None


def validate_choice(client_id: str, value: str, terminals: list[TerminalApp] | None = None) -> str:
    """The value to save for ``mtplx config set <client>_opens_in VALUE``:
    "terminal", one of the client's apps, or an installed terminal's name.
    Raises ValueError with the choices this Mac has."""

    client = CLIENTS_BY_ID[client_id]
    wanted = value.strip()
    if wanted in (TERMINAL, *client.app_surfaces):
        return wanted
    terminals = installed_terminals() if terminals is None else terminals
    chosen = next((t for t in terminals if t.matches(wanted)), None)
    if chosen is not None:
        return chosen.name
    options = [*client.app_surfaces, *(t.name for t in terminals)]
    raise ValueError(
        f"{client.config_key} must be one of: " + ", ".join(dict.fromkeys(options or [TERMINAL]))
    )


def write_command_script(
    command: str, *, client: str, workspace: str | Path, directory: Path
) -> Path:
    """A one-use ``.command`` script (0700) that runs ``command`` in
    ``workspace``: the file every terminal app is handed."""

    directory.mkdir(parents=True, exist_ok=True)
    script = directory / f"open-{client}-{uuid.uuid4().hex}.command"
    fd = os.open(script, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o700)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(f"#!/bin/zsh\ncd {shlex.quote(str(workspace))} || exit 1\n{command}\n")
    return script


def open_argv(script: str | Path, terminal: TerminalApp | None) -> list[str]:
    application = terminal.path if terminal is not None else "Terminal"
    return ["/usr/bin/open", "-a", application, str(script)]


def open_script(script: str | Path, terminal: TerminalApp | None) -> dict[str, Any]:
    """Run a ``.command`` script in ``terminal`` (Terminal when None)."""

    if sys.platform != "darwin":
        return {"ok": False, "error": "opening a terminal client needs macOS"}
    argv = open_argv(script, terminal)
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc), "argv": argv}
    if completed.returncode != 0:
        error = (completed.stderr or "").strip() or f"open exited {completed.returncode}"
        return {"ok": False, "error": error, "argv": argv}
    return {"ok": True, "argv": argv}


def open_command(
    command: str,
    *,
    client: str,
    workspace: str | Path | None = None,
    opening: Opening | None = None,
) -> dict[str, Any]:
    """Run an interactive client's command in ``workspace`` (the current
    folder by default), in the terminal the user chose for ``client``.
    ``terminal`` in the result is the app it opened in, ``note`` why that is
    not the saved choice."""

    from mtplx.agent_clients import handoff_dir

    if sys.platform != "darwin":
        return {
            "ok": False,
            "status": "unsupported_platform",
            "command": command,
            "error": "opening a terminal client needs macOS",
        }
    opening = opening or terminal_for(client, saved_choice(client))
    try:
        script = write_command_script(
            command, client=client, workspace=workspace or os.getcwd(), directory=handoff_dir()
        )
    except OSError as exc:
        return {"ok": False, "status": "launch_failed", "command": command, "error": str(exc)}
    result = open_script(script, opening.terminal)
    return {
        **result,
        "status": "launched" if result.get("ok") else "launch_failed",
        "command": command,
        "terminal": opening.terminal_name,
        "note": opening.note,
    }


def describe(opening: Opening) -> str:
    """"Ghostty", "MTPLX's own Codex app", ...: for CLI lines and listings."""

    app = CLIENTS_BY_ID[opening.client].app_name
    if opening.surface == MTPLX_APP:
        return f"MTPLX's own {app} app"
    if opening.surface == OWN_APP:
        return f"the {app} app"
    return opening.terminal_name


def summary(config: Any | None = None, terminals: list[TerminalApp] | None = None) -> dict[str, Any]:
    """Every client's opening and this Mac's terminals, for ``mtplx clients``."""

    from mtplx.agent_clients import desktop_app_for

    terminals = installed_terminals() if terminals is None else terminals
    rows = []
    for client in CLIENTS:
        choice = saved_choice(client.id, config)
        opening = resolve(client.id, choice, terminals)
        opens_in = describe(opening)
        if opening.surface == MTPLX_APP and desktop_app_for(client.id, choice=MTPLX_APP) is None:
            # What `mtplx start` does then: the client's CLI in a terminal.
            fallback = terminal_for(client.id, None, terminals).terminal_name
            opens_in = f"{fallback} (no {client.app_name} app MTPLX can open)"
        rows.append(
            {
                "client": client.id,
                "name": client.name,
                "config_key": client.config_key,
                "choice": choice,
                "opens_in": opens_in,
                "opening": opening.to_dict(),
                "choices": [TERMINAL, *client.app_surfaces, *(t.name for t in terminals)],
            }
        )
    return {"clients": rows, "terminals": [asdict(t) for t in terminals]}


# The cases the app's Swift twin is compared against --------------------------

FIXTURE_TERMINALS: tuple[TerminalApp, ...] = (
    TerminalApp("com.apple.Terminal", "Terminal", "/System/Applications/Utilities/Terminal.app"),
    TerminalApp("com.mitchellh.ghostty", "Ghostty", "/Applications/Ghostty.app"),
    TerminalApp("dev.warp.Warp-Stable", "Warp", "/Applications/Warp.app"),
)
FIXTURE_CHOICES: tuple[str | None, ...] = (
    None, "", TERMINAL, OWN_APP, MTPLX_APP, "Ghostty", " ghostty ",
    "com.mitchellh.ghostty", "iTerm", "Warp",
)
FIXTURE_INFOS: tuple[dict[str, Any], ...] = (
    {
        "CFBundleIdentifier": "com.apple.Terminal",
        "CFBundleName": "Terminal",
        "CFBundleDocumentTypes": [
            {"CFBundleTypeRole": "Shell", "LSItemContentTypes": [SHELL_SCRIPT_TYPE]},
        ],
    },
    {
        "CFBundleIdentifier": "dev.warp.Warp-Stable",
        "CFBundleDisplayName": "Warp",
        "CFBundleName": "Warp-Stable",
        "CFBundleDocumentTypes": [
            {"CFBundleTypeRole": "Viewer", "LSItemContentTypes": ["public.folder"]},
            {"CFBundleTypeRole": "Shell", "LSItemContentTypes": [SHELL_SCRIPT_TYPE]},
        ],
    },
    {  # an editor that opens .command files to edit them: not a terminal
        "CFBundleIdentifier": "com.example.editor",
        "CFBundleName": "Editor",
        "CFBundleDocumentTypes": [
            {"CFBundleTypeRole": "Editor", "LSItemContentTypes": [SHELL_SCRIPT_TYPE]},
        ],
    },
    {"CFBundleName": "No Id"},
)


def app_parity_fixture() -> dict[str, Any]:
    terminals = list(FIXTURE_TERMINALS)
    return {
        "version": 1,
        "generated_by": "scripts/write_client_opening_cases.py",
        "clients": [asdict(client) for client in CLIENTS],
        "terminals": [asdict(t) for t in terminals],
        "resolve": [
            {
                "client": client.id,
                "choice": choice,
                "opening": resolve(client.id, choice, terminals).to_dict(),
                "terminal_for": terminal_for(client.id, choice, terminals).to_dict(),
            }
            for client in CLIENTS
            for choice in FIXTURE_CHOICES
        ],
        "infos": [
            {
                "info": info,
                "terminal": (
                    None
                    if (found := terminal_from_info(info, "/Applications/X.app")) is None
                    else asdict(found)
                ),
            }
            for info in FIXTURE_INFOS
        ],
    }
