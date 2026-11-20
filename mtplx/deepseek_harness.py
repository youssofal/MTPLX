"""DeepSeek Harness on the local MTPLX server, with its telemetry off.

DeepSeek Harness is the npm package ``@deepseek-ai/dsh`` (MIT). Its ``dsh``
command boots a *profile*: an ordered stack of patch layers over a tree of
plugin rows (``$DSH_HOME/profiles/<name>``), then any ``--patch`` overlays the
launcher names, applied last. MTPLX never edits the user's own harness home
(``~/.dsh`` by default) or their own profiles. A session MTPLX opens runs on
MTPLX's own harness home, ``~/.mtplx/clients/deepseek-harness/home``, with one
overlay MTPLX writes on each launch (``mtplx.patch.yml`` beside it) that:

* points the harness's OpenAI-compatible route (the ``llm-pi-ai`` row) at the
  local server, the model it serves, the window it executes and whether it
  reads images, and makes that the default model; and
* switches off every piece that reports to the vendor or phones home
  (``OFF_ROWS``, each with the reason), on top of the harness's own telemetry
  switch in the environment (``TELEMETRY_ENV``).

``audit_rows`` reads the tree the harness actually composes with that overlay
(``dsh --dump-config``) and names any row that looks like reporting and is not
switched off, and any row MTPLX has not reviewed: a newer harness that adds a
piece is refused at launch (reporting) or fails the test suite (unreviewed),
never shipped quietly.

Two surfaces, one setting (``deepseek_harness_opens_in``): the harness's
browser app (``dsh web``, MTPLX keeps its server in the background and
closes it on Stop) by default, or the same server run in a terminal window
the user picks, where its own log shows (0.1.2 has no terminal app of its
own; see ``TERMINAL_PROFILE``).

Pure data plus subprocess calls made only when asked; nothing at import time.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

CLIENT_ID = "deepseek-harness"
DISPLAY_NAME = "DeepSeek Harness"
EXECUTABLE = "dsh"
NPM_PACKAGE = "@deepseek-ai/dsh"
INSTALL_HINT = "Install DeepSeek Harness with: npm install -g @deepseek-ai/dsh"
# The client name the server sees (x-mtplx-client), for its request policy
# and the request log.
CLIENT_HEADER_VALUE = "deepseek_harness"
PROVIDER_ID = "mtplx"
API_KEY_ENV = "MTPLX_API_KEY"
LOCAL_API_KEY = "mtplx-local"
# The harness's profile for each surface. 0.1.2 ships no interactive terminal
# app (its shipped profiles are web, headless for one task, acp and sdk for
# other programs), so "in the terminal" runs its own browser-app server in a
# terminal window: its log shows there, it opens the browser itself, and
# closing the window ends it.
WEB_PROFILE = "web"
TERMINAL_PROFILE = WEB_PROFILE
# MTPLX's coding-agent default effort (the same evidence as the Codex copy:
# medium gave the same correct multi-file answer in a third of xhigh's time).
DEFAULT_REASONING_EFFORT = "medium"
# A long cold read on a small Mac sends no bytes for minutes before its first
# token; an xhigh answer streams for longer than ten minutes. Neither is a
# reason to cut a request off (Claude Code gets the same 50 minutes).
REQUEST_TIMEOUT_MS = 3_000_000
STREAM_IDLE_TIMEOUT_MS = 3_000_000

# The harness's own switch (any non-empty value disables its session
# telemetry row), the mode it would otherwise read (FEEDBACK_ONLY by
# default), and the collector address pointed at the discard port, so even a
# future exporter that only reads the address sends nothing anywhere.
TELEMETRY_ENV: dict[str, str] = {
    "DSH_TELEMETRY_DISABLED": "1",
    "DSH_TELEMETRY_MODE": "DISABLED",
    "DSH_TELEMETRY_OTLP_URL": "http://127.0.0.1:9/v1/logs",
}
# The user's own DeepSeek key would let a vendor route work (and reach the
# vendor) inside a session MTPLX opened; that session has no use for it.
UNSET_ENV: tuple[str, ...] = ("DEEPSEEK_API_KEY",)

# Rows MTPLX switches off, by row id, with why (reviewed in 0.1.2-rc.1). A row
# a profile does not compose is skipped by the harness with a warning.
OFF_ROWS: dict[str, str] = {
    "session-telemetry-otel": (
        "session telemetry exporter: uploads session records to the vendor's "
        "collector, tagged with the harness home's anonymous user id"
    ),
    "session-log-deepseek": "session log addressed to the vendor",
    "plugin-package-inventory-deepseek": "plugin inventory attached to vendor requests",
    "command-feedback": "/feedback, which hands the session to the telemetry upload",
    "message-feedback": "like and dislike on answers, which feeds the same upload",
    "ui-message-feedback": "the like and dislike buttons in the browser app",
    "llm-deepseek": "the vendor's own model API (the session runs on MTPLX)",
    "web-search-deepseek": "web search through the vendor's API",
    # Not reporting, but a second model request at the start of each session
    # that would queue behind (or delay) the real turn on a local server.
    "session-title-llm": "a separate model request per session to write its title",
}

# Names that read as reporting. An enabled row matching one of these that is
# not in OFF_ROWS or REVIEWED_SAFE_ROWS stops the launch (fail closed).
REPORTING_PATTERN = re.compile(
    r"telemetry|otel|analytic|metric|feedback|crash|sentry|report|inventory|"
    r"session-log|update|upgrade|beacon|track|usage|ping|-deepseek\b|deepseek-",
    re.IGNORECASE,
)
# Enabled rows whose names match REPORTING_PATTERN but were read and found
# local (row id -> what it does).
REVIEWED_SAFE_ROWS: dict[str, str] = {
    "deepseek-llm-api-extensions": "request-shape extensions for the vendor model route, which is off",
    "token-meter": "counts tokens for the local context meter",
    "plugin-inventory": "read-only list of loaded plugins for the local browser app",
    "ui-settings-plugin-inventory": "the settings page that shows that local list",
    "session-log-download": "exports a session to a file the user downloads",
}


# --- paths --------------------------------------------------------------------


def root_dir() -> Path:
    from mtplx.agent_clients import clients_dir

    return clients_dir() / CLIENT_ID


def home_dir() -> Path:
    """MTPLX's own harness home (``DSH_HOME`` for every session it opens)."""

    return root_dir() / "home"


def overlay_path() -> Path:
    return root_dir() / "mtplx.patch.yml"


def resolve_executable() -> str | None:
    """The ``dsh`` the user's own terminal would run, else a known install path."""

    found = shutil.which(EXECUTABLE)
    if found:
        return found
    from mtplx import deepseek_install

    home = Path.home()
    for candidate in (
        home / ".local" / "bin" / EXECUTABLE,
        Path("/opt/homebrew/bin") / EXECUTABLE,
        Path("/usr/local/bin") / EXECUTABLE,
        home / ".npm-global" / "bin" / EXECUTABLE,
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    root = root_dir()
    return str(deepseek_install.executable(root)) if deepseek_install.installed(root) else None


# --- the profile MTPLX writes ---------------------------------------------------


def ensure_executable() -> str:
    found = resolve_executable()
    if found:
        return found
    from mtplx import deepseek_install

    return str(deepseek_install.install(root_dir(), TELEMETRY_ENV))


def provider_profile(
    *,
    base_url: str,
    model_id: str,
    context_window: int | None,
    vision: bool,
    model_name: str | None = None,
) -> dict[str, Any]:
    """The harness's route to MTPLX (an ``llm-pi-ai`` provider).

    The wire contract is the one MTPLX's Pi provider is proven on (the harness
    speaks through the same pi-ai library): system messages instead of
    developer ones, ``max_tokens``, and the Qwen thinking fields
    ``enable_thinking`` / ``reasoning_effort``. ``maxTokens`` is the served
    window, so the cap the library writes into every request can never bind;
    the server drops it for this client (it is not the user's choice).
    """

    window = int(context_window) if context_window and int(context_window) > 0 else None
    model: dict[str, Any] = {
        "id": str(model_id),
        "name": model_name or f"MTPLX {model_id}",
        # Engine capability (Law 1): offer images when the pack reads them.
        "input": ["text", "image"] if vision else ["text"],
        # Key = the level a user picks, value = what the server is sent; off
        # sends enable_thinking false. The server narrows to its family's tiers.
        "reasoningEfforts": {
            "off": None,
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh",
        },
    }
    if window:
        model["contextWindow"] = window
        model["maxTokens"] = window
    return {
        "displayName": "MTPLX",
        "apiKeyEnv": API_KEY_ENV,
        "api": "openai-completions",
        "baseURL": base_url.rstrip("/") + "/v1",
        "headers": {"x-mtplx-client": CLIENT_HEADER_VALUE},
        "compat": {
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": True,
            "thinkingFormat": "qwen",
            "maxTokensField": "max_tokens",
        },
        "timeoutMs": REQUEST_TIMEOUT_MS,
        "streamIdleTimeoutMs": STREAM_IDLE_TIMEOUT_MS,
        "models": [model],
    }


def overlay_rows(
    *,
    base_url: str,
    model_id: str,
    context_window: int | None,
    vision: bool,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """The overlay's patch rows: the MTPLX route, the default model, then
    every row in ``OFF_ROWS`` switched off."""

    rows: list[dict[str, Any]] = [
        {
            "id": "llm-pi-ai",
            "config": {
                "providers": {
                    PROVIDER_ID: provider_profile(
                        base_url=base_url,
                        model_id=model_id,
                        context_window=context_window,
                        vision=vision,
                        model_name=model_name,
                    )
                }
            },
        },
        {
            "id": "agent-default-model",
            "config": {
                "provider": PROVIDER_ID,
                "model": str(model_id),
                "reasoningEffort": DEFAULT_REASONING_EFFORT,
            },
        },
    ]
    rows.extend({"id": row_id, "disabled": True} for row_id in OFF_ROWS)
    return rows


def overlay_text(rows: list[dict[str, Any]]) -> str:
    """The overlay file: a YAML patch list (written as JSON, which YAML reads
    as is, so every value is quoted exactly)."""

    header = [
        "# MTPLX writes this file each time it opens DeepSeek Harness on the local",
        "# MTPLX server (mtplx start deepseek-harness, or the MTPLX app). It routes",
        "# the session to MTPLX and switches off the harness's telemetry, its",
        "# vendor session log, feedback uploads and vendor routes. Edits here are",
        "# replaced on the next launch; settings you change in the harness itself",
        "# are kept in its home folder beside this file.",
    ]
    return "\n".join(header) + "\n" + json.dumps(rows, indent=2) + "\n"


def launch_env(*, api_key: str | None, home: str | Path | None = None) -> dict[str, str]:
    env = {
        "DSH_HOME": str(home or home_dir()),
        API_KEY_ENV: api_key or LOCAL_API_KEY,
    }
    from mtplx import deepseek_install

    root = root_dir()
    if resolve_executable() == str(deepseek_install.executable(root)):
        node_bin = deepseek_install.runtime_directory(root) / "node" / "bin"
        env["PATH"] = os.pathsep.join((str(node_bin), os.environ.get("PATH", os.defpath)))
    env.update(TELEMETRY_ENV)
    return env


def _plan(
    launcher: list[str],
    app_args: list[str],
    *,
    surface: str,
    base_url: str,
    model_id: str,
    context_window: int | None,
    api_key: str | None,
    vision: bool,
    model_name: str | None,
    overlay: str | Path | None,
    home: str | Path | None,
) -> dict[str, Any]:
    overlay_file = str(overlay or overlay_path())
    rows = overlay_rows(
        base_url=base_url,
        model_id=model_id,
        context_window=context_window,
        vision=vision,
        model_name=model_name,
    )
    # The launcher reads its own flags (--profile, --patch) only up to the
    # first one it does not know; everything after goes to the app. So the
    # overlay comes before the app's own flags (`web --patch F --port N`).
    return {
        "client": CLIENT_ID,
        "surface": surface,
        "argv": [*launcher, "--patch", overlay_file, *app_args],
        "env": launch_env(api_key=api_key, home=home),
        "unset_env": list(UNSET_ENV),
        "files": {overlay_file: overlay_text(rows)},
        "folders": [str(home or home_dir())],
    }


def build_terminal_plan(
    *,
    base_url: str,
    model_id: str,
    context_window: int | None = None,
    api_key: str | None = None,
    vision: bool = False,
    model_name: str | None = None,
    executable: str | None = None,
    overlay: str | Path | None = None,
    home: str | Path | None = None,
    port: int | None = None,
) -> dict[str, Any]:
    """The harness in a terminal window on MTPLX: ``dsh web`` run where the
    user sees it (see ``TERMINAL_PROFILE``), on a free local port."""

    return _plan(
        [executable or EXECUTABLE, "web"],
        ["--port", str(int(port if port is not None else free_local_port()))],
        surface="terminal",
        base_url=base_url,
        model_id=model_id,
        context_window=context_window,
        api_key=api_key,
        vision=vision,
        model_name=model_name,
        overlay=overlay,
        home=home,
    )


def build_web_plan(
    *,
    base_url: str,
    model_id: str,
    port: int,
    context_window: int | None = None,
    api_key: str | None = None,
    vision: bool = False,
    model_name: str | None = None,
    executable: str | None = None,
    overlay: str | Path | None = None,
    home: str | Path | None = None,
) -> dict[str, Any]:
    """The harness's browser app on MTPLX: ``dsh web`` on a free local port.
    It opens the user's browser on its own address once it is ready."""

    plan = _plan(
        [executable or EXECUTABLE, "web"],
        # MTPLX opens the browser itself once the server answers.
        ["--port", str(int(port)), "--no-open"],
        surface="web",
        base_url=base_url,
        model_id=model_id,
        context_window=context_window,
        api_key=api_key,
        vision=vision,
        model_name=model_name,
        overlay=overlay,
        home=home,
    )
    plan["port"] = int(port)
    return plan


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


# --- what the harness composes, checked --------------------------------------------


def parse_dump_rows(text: str) -> list[dict[str, Any]]:
    """Rows of a ``dsh --dump-config`` listing: id, package name and whether
    the row is switched off (``disabled: true``). A row whose ``disabled`` is
    an expression (a platform check) counts as on: the audit errs that way.

    Top-level entries are the rows. An entry nested inside a row's config
    that names a plugin package (a group's children) is listed too, as on
    unless it says otherwise, with ``nested`` set, so the audit sees every
    plugin the tree can mount; a nested entry without a package (a model in
    a provider's list) is not a row."""

    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_indent = -1
    pending_nested: dict[str, Any] | None = None
    for line in text.splitlines():
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        match = re.match(r"- id:\s*(.+?)\s*$", stripped)
        if match:
            entry = {"id": _scalar(match.group(1)), "name": None, "disabled": False}
            if indent == 0:
                current, current_indent, pending_nested = entry, 2, None
                rows.append(entry)
            else:
                pending_nested = {**entry, "nested": True, "_indent": indent + 2}
            continue
        key, _, value = stripped.partition(":")
        if pending_nested is not None and indent == pending_nested["_indent"]:
            if key == "name":
                name = _scalar(value.strip())
                if name.startswith("@") or name.startswith("cordis:"):
                    pending_nested["name"] = name
                    rows.append(pending_nested)
            elif key == "disabled":
                pending_nested["disabled"] = value.strip() == "true"
            continue
        if current is None or indent != current_indent:
            continue
        if key == "name":
            current["name"] = _scalar(value.strip())
        elif key == "disabled":
            current["disabled"] = value.strip() == "true"
    for row in rows:
        row.pop("_indent", None)
    return rows


def _scalar(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def audit_rows(rows: Iterable[Mapping[str, Any]], reviewed: Iterable[str] = ()) -> dict[str, Any]:
    """What a composed tree still leaves on.

    ``reporting``: enabled rows whose id or package reads as reporting and
    that MTPLX neither switches off nor has read and found local. A launch
    with any is refused. ``unreviewed``: enabled rows not in ``reviewed``
    (the rows of the version MTPLX was checked against); the tests fail on
    any, so a new harness version is read before it ships.
    """

    known = set(reviewed)
    reporting: list[dict[str, Any]] = []
    unreviewed: list[dict[str, Any]] = []
    enabled = 0
    for row in rows:
        if row.get("disabled"):
            continue
        enabled += 1
        row_id = str(row.get("id") or "")
        package = str(row.get("name") or "").removeprefix("@deepseek-ai/")
        if row_id in OFF_ROWS:
            # Switched off by the overlay yet composed on: the overlay did not
            # reach this row, which is exactly what must not launch.
            reporting.append({"id": row_id, "name": row.get("name"), "why": OFF_ROWS[row_id]})
            continue
        if row_id not in REVIEWED_SAFE_ROWS and (
            REPORTING_PATTERN.search(row_id) or REPORTING_PATTERN.search(package)
        ):
            reporting.append({"id": row_id, "name": row.get("name"), "why": "reads as reporting"})
        if known and row_id not in known:
            unreviewed.append({"id": row_id, "name": row.get("name")})
    return {"ok": not reporting, "enabled": enabled, "reporting": reporting, "unreviewed": unreviewed}


def dump_config(
    plan: Mapping[str, Any], *, profile: str, executable: str | None = None, timeout_s: float = 60.0
) -> dict[str, Any]:
    """The tree ``plan`` would boot, from the harness itself (nothing runs:
    ``--dump-config`` composes without starting any plugin)."""

    exe = executable or resolve_executable()
    if not exe:
        return {"ok": False, "error": INSTALL_HINT}
    overlay = next(iter(plan["files"]))
    argv = [exe, "--profile", profile, "--dump-config", "--patch", overlay]
    if profile == WEB_PROFILE:
        argv = [exe, "web", "--dump-config", "--patch", overlay]
    env = _child_env(plan)
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_s, check=False, env=env,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc), "argv": argv}
    if completed.returncode != 0:
        return {
            "ok": False,
            "error": (completed.stderr or "").strip()[-2000:] or f"exit {completed.returncode}",
            "argv": argv,
        }
    return {"ok": True, "argv": argv, "text": completed.stdout, "warnings": completed.stderr}


def _child_env(plan: Mapping[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    for name in plan.get("unset_env") or ():
        env.pop(name, None)
    env.update({key: str(value) for key, value in (plan.get("env") or {}).items()})
    return env


def prepare_home(home: str | Path) -> Path:
    """MTPLX's harness home, with its web profile set to read patch files at
    start only.

    The shipped web profile re-reads them live, which needs the harness's
    module-reload service; on a fresh home 0.1.2-rc.1 prints its address and
    then exits ("user patch-layer watching requires the Cordis HMR service").
    MTPLX writes its overlay before each launch, so nothing needs the live
    reload. The harness never rewrites an existing manifest, so this one
    stays; any plugin the user added to it is kept.
    """

    profile = Path(home) / "profiles" / WEB_PROFILE
    profile.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest_path = profile / "package.json"
    manifest: dict[str, Any] = {"name": f"dsh-profile-{WEB_PROFILE}", "private": True, "dependencies": {}}
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest = loaded
        except (OSError, ValueError):
            pass
    section = manifest.setdefault("dsh", {}).setdefault("profile", {})
    section.setdefault("bundles", ["@deepseek-ai/dsh-base", "@deepseek-ai/dsh-web-app"])
    section["patchReload"] = "startup"
    text = json.dumps(manifest, indent=2) + "\n"
    if not manifest_path.exists() or manifest_path.read_text(encoding="utf-8") != text:
        manifest_path.write_text(text, encoding="utf-8")
    return profile


def write_files(plan: Mapping[str, Any]) -> None:
    from mtplx.agent_clients import write_plan_files

    for folder in plan.get("folders") or ():
        Path(folder).mkdir(parents=True, exist_ok=True, mode=0o700)
    prepare_home(plan["env"]["DSH_HOME"])
    write_plan_files(dict(plan))


def check_composition(plan: Mapping[str, Any], *, profile: str, executable: str | None = None) -> dict[str, Any]:
    """Write the plan's files, compose its tree and audit it. ``ok`` False
    with ``reporting`` rows means the launch must not happen."""

    write_files(plan)
    dumped = dump_config(plan, profile=profile, executable=executable)
    if not dumped.get("ok"):
        return {"ok": False, "error": dumped.get("error"), "argv": dumped.get("argv")}
    audit = audit_rows(parse_dump_rows(dumped["text"]))
    if not audit["ok"]:
        names = ", ".join(row["id"] for row in audit["reporting"])
        return {
            **audit,
            "error": (
                f"this DeepSeek Harness version adds pieces that report to its vendor and "
                f"MTPLX does not switch off yet ({names}); MTPLX will not open it"
            ),
        }
    return audit


# --- the browser app: a server MTPLX owns ------------------------------------------


_URL_LINE = re.compile(r"dsh web: (http://127\.0\.0\.1:\d+/\S*)")


def start_web(
    plan: Mapping[str, Any],
    *,
    workspace: str,
    log_path: str | Path | None = None,
    timeout_s: float = 60.0,
    open_browser: bool = True,
) -> dict[str, Any]:
    """Start ``dsh web`` for ``plan`` in the background, in ``workspace``,
    and open the user's browser on it once it answers.

    It runs in its own process group, so ``end_web`` ends the server and
    anything it started, and only ever this one: the process is matched by
    MTPLX's overlay path in its own argv before any signal. ``url`` in the
    result carries the browser app's sign-in token (it stays on this Mac);
    ``address`` is the same without it, for logs.
    """

    if sys.platform != "darwin":
        return {"ok": False, "error": "DeepSeek Harness through MTPLX needs macOS"}
    exe = resolve_executable()
    if not exe:
        return {"ok": False, "error": INSTALL_HINT}
    argv = [exe, *list(plan["argv"])[1:]]
    log = Path(log_path) if log_path else root_dir() / "web.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    start = log.stat().st_size if log.exists() else 0
    try:
        handle = open(log, "ab")
        process = subprocess.Popen(
            argv,
            cwd=workspace,
            env=_child_env(plan),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        handle.close()
    except OSError as exc:
        return {"ok": False, "error": str(exc), "argv": argv}
    url = _wait_until_serving(process, log, start, timeout_s)
    if url is None:
        tail = _log_tail(log, start)
        if process.poll() is None:
            end_web(process.pid, next(iter(plan["files"])))
        return {
            "ok": False,
            "error": f"DeepSeek Harness did not start: {tail or 'no answer in time'}",
            "argv": argv,
            "log": str(log),
        }
    if open_browser:
        subprocess.run(["/usr/bin/open", url], capture_output=True, timeout=30, check=False)
    return {
        "ok": True,
        "pid": process.pid,
        "argv": argv,
        "port": plan.get("port"),
        "url": url,
        "address": url.split("?", 1)[0],
        "log": str(log),
    }


def _wait_until_serving(process: subprocess.Popen, log: Path, start: int, timeout_s: float) -> str | None:
    """The address ``dsh web`` prints, once a request to it answers 200; None
    if the process exits or nothing answers within ``timeout_s``."""

    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout_s
    url: str | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return None
        if url is None:
            found = _URL_LINE.findall(_log_text(log, start))
            url = found[-1] if found else None
        if url is not None:
            try:
                opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
                with opener.open(url, timeout=2) as response:
                    if response.status == 200 and process.poll() is None:
                        return url
            except (OSError, urllib.error.URLError):
                pass
        time.sleep(0.25)
    return None


def _log_text(log: Path, start: int) -> str:
    try:
        with open(log, "rb") as handle:
            handle.seek(start)
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _log_tail(log: Path, start: int) -> str:
    lines = [line for line in _log_text(log, start).splitlines() if line.strip()]
    # The address carries a token; the reason is in the last lines.
    lines = [line for line in lines if not line.startswith("dsh web: ")]
    return " | ".join(lines[-3:])[-600:]


def is_mtplx_web_process(pid: int, overlay: str | Path | None = None) -> bool:
    """True while ``pid`` is a harness MTPLX started (its argv names MTPLX's
    overlay), never the user's own."""

    from mtplx.agent_clients import _process_arguments

    image = _process_arguments(int(pid))
    if not image:
        return False
    return str(overlay or overlay_path()) in image[1]


def end_web(pid: int, overlay: str | Path | None = None, *, grace_s: float = 5.0) -> bool:
    """End the harness MTPLX started (SIGTERM to its group, then SIGKILL if it
    is still that same process after ``grace_s``)."""

    if not is_mtplx_web_process(pid, overlay):
        return False
    try:
        os.killpg(int(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return True
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not is_mtplx_web_process(pid, overlay):
            return True
        time.sleep(0.1)
    if is_mtplx_web_process(pid, overlay):
        try:
            os.killpg(int(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    return True
