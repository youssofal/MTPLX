"""DeepSeek Harness (dsh) coding-agent integration helpers.

The public CLI uses this module to make ``mtplx start dsh`` a real connection
flow: register an MTPLX provider in DSH's ``settings.yaml`` (and its API key
in ``.credentials.yaml``), then start the OpenAI-compatible MTPLX server with
matching settings and launch the DSH web app.

DSH (https://github.com/deepseek-ai/deepseek-harness) is a plugin-bundle
harness whose LLM layer is pi-ai based. A provider profile there uses the
same Chat Completions wire contract as Pi — ``api: openai-completions`` plus
a ``compat`` block — so MTPLX's Qwen thinking vocabulary
(``thinkingFormat: qwen``) wires through unchanged.

Unlike Pi's ``models.json`` (JSON, flat ``providers`` root), DSH keeps its
config in ``<dshHome>/settings.yaml`` under the ``llm-pi-ai`` namespace with a
``providers`` map, and secrets in a versioned ``<dshHome>/.credentials.yaml``
reference store (the profile names an environment variable via ``apiKeyEnv``;
the variable's value lives in the credentials file, mode 0600).
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

DSH_PROVIDER_ID = "mtplx"
DSH_CLIENT_NAME = "dsh"
DSH_LOCAL_API_KEY = "mtplx-local"
# DSH profiles name credentials by environment variable (``apiKeyEnv``); the
# value is resolved from ``.credentials.yaml`` refs under that variable name.
DSH_API_KEY_ENV = "MTPLX_API_KEY"
DSH_NPM_PACKAGE = "@deepseek-ai/dsh"
DSH_DEFAULT_CONTEXT_WINDOW = 262_144
DSH_HOME_DIRNAME = ".dsh"
DSH_SETTINGS_NAMESPACE = "llm-pi-ai"
DSH_SETTINGS_FILENAME = "settings.yaml"
DSH_CREDENTIALS_FILENAME = ".credentials.yaml"
# Connection identity MTPLX must keep correct for the integration to work at
# all (ports move between launches). Everything else belongs to the user once
# they edit it (#282: silent clobber of user edits).
DSH_OWNED_PROVIDER_CONNECTION_KEYS = ("baseURL", "api", "apiKeyEnv", "headers", "compat")


def dsh_install_command() -> str:
    return f"npm install -g {DSH_NPM_PACKAGE}"


def dsh_home(path: str | Path | None = None) -> Path:
    """Return the DSH home directory.

    DSH's own precedence: an explicitly configured home, then ``$DSH_HOME``
    (a whitespace-only value means unset), then ``~/.dsh``.
    ``MTPLX_DSH_HOME`` exists only for tests and power-user overrides.
    """

    if path is not None:
        return Path(path).expanduser()
    for env_name in ("MTPLX_DSH_HOME", "DSH_HOME"):
        env = os.environ.get(env_name)
        if env is None or not env.strip():
            continue
        return Path(env).expanduser()
    return Path.home() / DSH_HOME_DIRNAME


def dsh_settings_path(path: str | Path | None = None) -> Path:
    """Return DSH's settings file path (``<dshHome>/settings.yaml``).

    ``path`` overrides the whole location (tests). Otherwise ``MTPLX_DSH_HOME``
    / ``DSH_HOME`` / ``~/.dsh`` apply (see :func:`dsh_home`).
    """

    if path is not None:
        return Path(path).expanduser()
    return dsh_home() / DSH_SETTINGS_FILENAME


def dsh_credentials_path(path: str | Path | None = None) -> Path:
    """Return DSH's credentials store path (``<dshHome>/.credentials.yaml``)."""

    if path is not None:
        return Path(path).expanduser()
    return dsh_home() / DSH_CREDENTIALS_FILENAME


def dsh_model_ref(model_id: str, *, provider_id: str = DSH_PROVIDER_ID) -> str:
    return f"{provider_id}/{model_id}"


def dsh_launch_command() -> str:
    """The DSH surface command.

    ``dsh web`` boots the web profile (auto-initializing it on first use) and
    serves the browser app; the MTPLX provider and model show up in the app's
    model picker through ``settings.yaml``. The web surface takes no model
    argument — model selection is an in-app choice.
    """

    return "dsh web"


def launch_dsh_in_terminal(command: str, *, model_ref: str | None = None) -> dict[str, Any]:
    """Open DSH in a macOS Terminal window/tab without blocking MTPLX.

    DSH's web app is a browser surface, but hosting the server process in a
    visible Terminal tab keeps the log readable and Ctrl-C stopping trivial.
    Mirror the Pi launcher: always try to open it; a false "already running"
    is much worse than an extra tab. On non-macOS systems, return a clear
    fallback payload.
    """

    _ = model_ref  # kept for call-site clarity and future platform-specific launchers.
    if sys.platform != "darwin":
        return {
            "ok": False,
            "status": "unsupported_platform",
            "command": command,
            "error": "automatic DSH launch currently requires macOS Terminal",
        }
    script = "\n".join(
        [
            'tell application "Terminal"',
            "  activate",
            f"  do script {json.dumps(command)}",
            "end tell",
        ]
    )
    try:
        subprocess.Popen(
            ["osascript", "-e", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        return {"ok": False, "status": "launch_failed", "command": command, "error": str(exc)}
    return {"ok": True, "status": "launched", "command": command}


def build_dsh_provider_profile(
    *,
    base_url: str,
    model_id: str,
    model_name: str | None = None,
    context_window: int = DSH_DEFAULT_CONTEXT_WINDOW,
) -> dict[str, Any]:
    """Build the DSH provider profile block MTPLX needs.

    DSH's pi-ai-backed transport needs the Chat Completions API name, a
    credentials reference (``apiKeyEnv`` — the value lives in
    ``.credentials.yaml``, never in settings), and the compatibility flags so
    it sends ``system`` instead of ``developer`` and ``max_tokens`` instead of
    the newer OpenAI field. The Qwen thinking format wires DSH's
    thinking-level picker to the server's ``enable_thinking``/
    ``reasoning_effort`` request fields — the same contract Pi uses.

    No ``maxTokens`` metadata is advertised: DSH then sends no per-request
    output cap at all, and the MTPLX server applies the loaded model's own
    limit. (Pi's 16,384 injected default has no DSH equivalent to strip.)
    """

    model_config: dict[str, Any] = {
        "id": str(model_id),
        "name": model_name or f"MTPLX {model_id}",
        "contextWindow": int(context_window),
    }

    return {
        "displayName": "MTPLX",
        "apiKeyEnv": DSH_API_KEY_ENV,
        "api": "openai-completions",
        "baseURL": str(base_url).rstrip("/"),
        "headers": {
            "x-mtplx-client": DSH_CLIENT_NAME,
        },
        # DSH (pi-ai) with thinkingFormat "qwen" serializes exactly the fields
        # the MTPLX server accepts: top-level ``enable_thinking`` plus
        # ``reasoning_effort``. Same wire contract as Pi's compat block.
        "compat": {
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": True,
            "thinkingFormat": "qwen",
            "maxTokensField": "max_tokens",
        },
        "models": [model_config],
    }


def _backup_snapshot(path: Path) -> Path:
    """Timestamped snapshot of an existing file, taken before MTPLX rewrites it."""

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.{stamp}.bak")
    counter = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}.{stamp}-{counter}.bak")
        counter += 1
    try:
        backup.write_bytes(path.read_bytes())
    except OSError:
        return path
    try:
        backup.chmod(path.stat().st_mode & 0o777 or 0o600)
    except OSError:
        pass
    return backup


def _backup_invalid_config(path: Path) -> Path:
    """Quarantine a file MTPLX cannot parse (invalid YAML/JSON)."""

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.invalid-{stamp}.bak")
    counter = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}.invalid-{stamp}-{counter}.bak")
        counter += 1
    path.replace(backup)
    return backup


def _fill_missing_deep(existing: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """Existing user values win; defaults only fill gaps, recursively.

    (Same user-preserving merge policy as :mod:`mtplx.pi`; duplicated so the
    two integrations stay independent — Pi is JSON, DSH is YAML.)
    """

    merged = dict(existing)
    for key, default_value in defaults.items():
        if key not in merged:
            merged[key] = default_value
        elif isinstance(merged[key], dict) and isinstance(default_value, dict):
            merged[key] = _fill_missing_deep(merged[key], default_value)
    return merged


def merge_dsh_provider_profile(
    existing_provider: Any,
    fresh: dict[str, Any],
) -> dict[str, Any]:
    """User-preserving merge of the MTPLX DSH provider profile (#282 clobber fix).

    MTPLX owns the connection identity (``baseURL``/``api``/``apiKeyEnv`` and
    the ``x-mtplx-client`` header, plus the ``compat`` transport contract)
    because ports move between launches and the integration must keep working.
    Every other key the user edited wins: our values only fill missing keys,
    recursively. Model entries merge by ``id`` the same way, and user-added
    models or fields survive a sync untouched.
    """

    if not isinstance(existing_provider, dict):
        return fresh
    merged = _fill_missing_deep(
        existing_provider,
        {
            key: value
            for key, value in fresh.items()
            if key not in ("models", "headers", "compat")
        },
    )
    for key in DSH_OWNED_PROVIDER_CONNECTION_KEYS:
        if key in fresh:
            merged[key] = fresh[key]
    # ``compat`` is MTPLX's transport contract with DSH (which wire fields the
    # server supports), not a user preference: a stale block written by an
    # older MTPLX must not outlive the engine that wrote it. Our keys win;
    # user-added extra compat keys still survive.
    existing_compat = (
        dict(existing_provider.get("compat"))
        if isinstance(existing_provider.get("compat"), dict)
        else {}
    )
    existing_compat.update(fresh.get("compat") or {})
    if existing_compat:
        merged["compat"] = existing_compat
    headers = (
        {
            key: value
            for key, value in (existing_provider.get("headers") or {}).items()
            if str(key).lower() != "x-mtplx-client"
        }
        if isinstance(existing_provider.get("headers"), dict)
        else {}
    )
    headers.update(fresh.get("headers") or {})
    merged["headers"] = headers

    fresh_models = fresh.get("models") or []
    existing_models = existing_provider.get("models")
    if not isinstance(existing_models, list):
        merged["models"] = fresh_models
        return merged
    fresh_ids = {str(model.get("id")) for model in fresh_models}
    # Stale MTPLX entries are pruned so switching models does not pile up dead
    # picker rows — the same policy Pi applies to models.json. Every model the
    # MTPLX server serves carries the "mtplx-" id prefix, so the prune covers
    # all MTPLX-served models, even ones added to the picker by hand; models
    # under the provider without the prefix (foreign entries) are kept.
    result_models = [
        entry
        for entry in existing_models
        if not (
            isinstance(entry, dict)
            and str(entry.get("id", "")).startswith("mtplx-")
            and str(entry.get("id")) not in fresh_ids
        )
    ]
    for fresh_model in fresh_models:
        fresh_id = str(fresh_model.get("id"))
        for index, entry in enumerate(result_models):
            if isinstance(entry, dict) and str(entry.get("id")) == fresh_id:
                result_models[index] = _fill_missing_deep(entry, fresh_model)
                break
        else:
            result_models.append(fresh_model)
    merged["models"] = result_models
    return merged


def merge_dsh_settings(
    existing: dict[str, Any] | None,
    *,
    provider_profile: dict[str, Any],
    provider_id: str = DSH_PROVIDER_ID,
    write_default_model: bool = False,
) -> dict[str, Any]:
    """Merge or create a DSH ``settings.yaml`` payload.

    MTPLX owns only ``llm-pi-ai.providers.mtplx`` (inside it, only the
    connection identity — see :func:`merge_dsh_provider_profile`). Other
    namespaces, other providers, and any user-added values are untouched.
    With ``write_default_model`` MTPLX additionally sets
    ``llm-pi-ai.agent-default-model`` to the MTPLX model so the web app opens
    on it; the bundle default (an official DeepSeek model) is left alone
    otherwise.
    """

    payload = dict(existing or {})
    namespace = payload.get(DSH_SETTINGS_NAMESPACE)
    if not isinstance(namespace, dict):
        namespace = {}
    else:
        namespace = dict(namespace)
    providers = namespace.get("providers")
    if not isinstance(providers, dict):
        providers = {}
    else:
        providers = dict(providers)
    providers[str(provider_id)] = merge_dsh_provider_profile(
        providers.get(str(provider_id)),
        provider_profile,
    )
    namespace["providers"] = providers
    if write_default_model:
        model_id = str((provider_profile.get("models") or [{}])[0].get("id", ""))
        if model_id:
            namespace["agent-default-model"] = {
                "provider": str(provider_id),
                "model": model_id,
            }
    payload[DSH_SETTINGS_NAMESPACE] = namespace
    return payload


def _load_yaml_document(path: Path) -> tuple[dict[str, Any] | None, Path | None]:
    """Load a YAML document, quarantining unparseable files.

    Returns ``(document, backup_path)``; a missing file yields ``({}, None)``.
    """

    backup_path: Path | None = None
    if not path.exists():
        return {}, None
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        backup_path = _backup_invalid_config(path)
        return {}, backup_path
    return (parsed if isinstance(parsed, dict) else {}), backup_path


def _dump_yaml_document(document: dict[str, Any]) -> str:
    return yaml.safe_dump(
        document,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=120,
    )


def write_dsh_credentials(
    *,
    api_key: str = DSH_LOCAL_API_KEY,
    env_name: str = DSH_API_KEY_ENV,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Record the MTPLX API key in DSH's ``.credentials.yaml`` (v1 layout).

    The file holds ``refs`` (POSIX-identifier name → value) and ``records``
    (richer credential objects); MTPLX touches only the one ref named by
    ``env_name`` and preserves everything else, including the version marker.
    Written 0600. Returns ``{"path", "backup_path", "written"}``.
    """

    credentials_path = dsh_credentials_path(path)
    existing, backup_path = _load_yaml_document(credentials_path)
    document: dict[str, Any] = {"version": 1}
    refs = existing.get("refs") if isinstance(existing, dict) else None
    if isinstance(refs, dict):
        document["refs"] = dict(refs)
    else:
        document["refs"] = {}
    records = existing.get("records") if isinstance(existing, dict) else None
    if records is not None:
        document["records"] = records
    document["refs"][str(env_name)] = str(api_key)
    credentials_path.parent.mkdir(parents=True, exist_ok=True)
    credentials_path.write_text(_dump_yaml_document(document) + "\n", encoding="utf-8")
    try:
        credentials_path.chmod(0o600)
    except OSError:
        pass
    return {
        "path": str(credentials_path),
        "backup_path": str(backup_path) if backup_path is not None else None,
        "written": True,
    }


def write_dsh_settings(
    *,
    base_url: str,
    model_id: str,
    model_name: str | None = None,
    api_key: str = DSH_LOCAL_API_KEY,
    path: str | Path | None = None,
    credentials_path: str | Path | None = None,
    provider_id: str = DSH_PROVIDER_ID,
    context_window: int = DSH_DEFAULT_CONTEXT_WINDOW,
    write_default_model: bool = False,
) -> dict[str, Any]:
    """Write the MTPLX provider into DSH's config and return a handoff payload.

    Two files are written (both under the DSH home unless overridden):
    ``settings.yaml`` gains the ``llm-pi-ai.providers.mtplx`` profile (values
    user-preserving; a pre-write snapshot is taken because the YAML round-trip
    cannot preserve comments), and ``.credentials.yaml`` records the API key
    under ``MTPLX_API_KEY``. Both 0600.
    """

    settings_path = dsh_settings_path(path)
    existing, invalid_backup = _load_yaml_document(settings_path)

    provider_profile = build_dsh_provider_profile(
        base_url=base_url,
        model_id=model_id,
        model_name=model_name,
        context_window=context_window,
    )
    merged = merge_dsh_settings(
        existing,
        provider_profile=provider_profile,
        provider_id=provider_id,
        write_default_model=write_default_model,
    )
    backup_path: Path | None = None
    if settings_path.exists():
        backup_path = _backup_snapshot(settings_path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(_dump_yaml_document(merged) + "\n", encoding="utf-8")
    try:
        settings_path.chmod(0o600)
    except OSError:
        pass

    credentials = write_dsh_credentials(
        api_key=api_key,
        path=credentials_path,
    )

    return {
        "config_path": str(settings_path),
        "credentials_path": str(credentials["path"]),
        "backup_path": str(backup_path) if backup_path is not None else None,
        "invalid_backup_path": str(invalid_backup) if invalid_backup is not None else None,
        "credentials_backup_path": credentials["backup_path"],
        "provider_id": provider_id,
        "base_url": provider_profile["baseURL"],
        "model_id": str(model_id),
        "model_ref": dsh_model_ref(str(model_id), provider_id=provider_id),
        "launch_command": dsh_launch_command(),
        "api_key": api_key,
        "api_key_env": DSH_API_KEY_ENV,
        "context_window": int(context_window),
        "no_hidden_max_tokens": True,
        "agent_default_model_written": write_default_model,
        "written": True,
    }