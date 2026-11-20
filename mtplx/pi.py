"""Pi coding-agent integration helpers.

The public CLI uses this module to make ``mtplx start pi`` a real connection
flow: merge an MTPLX provider into Pi's ``models.json`` and then start the
OpenAI-compatible MTPLX server with matching settings.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
from pathlib import Path
from typing import Any

from mtplx.jsonc import load_config_file

PI_PROVIDER_ID = "mtplx"
PI_LOCAL_API_KEY = "mtplx-local"
PI_NPM_PACKAGE = "@earendil-works/pi-coding-agent"
PI_DEFAULT_CONTEXT_WINDOW = 131_072
PI_DEFAULT_MAX_TOKENS: int | None = None
# Legacy Pi fallback when model metadata omits maxTokens. Configured models
# pass their advertised value to the bridge instead of assuming this value.
PI_INJECTED_DEFAULT_MAX_TOKENS = 16_384
PI_REQUEST_POLICY_EXTENSION_NAME = "mtplx-request-policy.ts"
# Connection identity MTPLX must keep correct for the integration to work at
# all (ports move between launches). Everything else belongs to the user once
# they edit it (#282: silent clobber of user edits in models.json).
PI_OWNED_PROVIDER_CONNECTION_KEYS = ("baseUrl", "api", "apiKey", "authHeader")
PI_EXTENSION_MANAGED_MARKER = "MTPLX-managed"


def pi_install_command() -> str:
    return f"npm install -g {PI_NPM_PACKAGE}"


def pi_models_json_path(path: str | Path | None = None) -> Path:
    """Return Pi's custom models config path.

    ``MTPLX_PI_MODELS_JSON`` exists only for tests and power-user overrides.
    Normal users get Pi's documented ``~/.pi/agent/models.json`` path.
    """

    if path is not None:
        return Path(path).expanduser()
    env = os.environ.get("MTPLX_PI_MODELS_JSON")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".pi" / "agent" / "models.json"


def pi_request_policy_extension_path(path: str | Path | None = None) -> Path:
    """Return the MTPLX-owned Pi extension next to ``models.json``."""

    return pi_models_json_path(path).parent / "extensions" / PI_REQUEST_POLICY_EXTENSION_NAME


def build_pi_settings_extension_source() -> str:
    """Independent of the user-owned request bridge; never overwrite customization."""
    return r'''// MTPLX-managed settings mirror. Remove this marker to take ownership.
export default function (pi: any) {
  let timer: any;
  let syncInFlight: any;
  let clientThinking: any;
  let sessionKey: any;

  const readSettings = async (ctx: any) => {
    const model = ctx.model;
    if (model?.provider !== "mtplx") {
      ctx.ui.setStatus("mtplx-controls", undefined);
      return;
    }
    try {
      const key = String(ctx.sessionManager.getSessionId()) + ":" + model.id;
      if (sessionKey !== key) {
        sessionKey = key;
        clientThinking = undefined;
      }
      const registry = ctx.modelRegistry;
      const auth = registry.getApiKeyAndHeaders
        ? await registry.getApiKeyAndHeaders(model)
        : { ok: true, apiKey: await registry.getApiKey(model) };
      if (!auth.ok) throw new Error(auth.error);
      const headers: any = {};
      for (const [name, value] of Object.entries({ ...model.headers, ...auth.headers })) {
        if (typeof value === "string") headers[name] = value;
      }
      if (auth.apiKey) headers.Authorization = "Bearer " + auth.apiKey;
      // Pi installs a global HTTP dispatcher whose idle connection reuse can
      // delay these small polls. Keep control-plane reads off pooled sockets.
      headers.Connection = "close";
      const base = String(auth.baseUrl || model.baseUrl).replace(/[/]+$/, "");
      const response = await fetch(base + "/mtplx/settings", {
        headers, signal: AbortSignal.timeout(5000),
      });
      if (!response.ok) throw new Error("settings HTTP " + response.status);
      const settings = await response.json();
      if (settings.managed_client_controls === "app") {
        const effort = settings.enable_thinking === false ? "off" : settings.reasoning_effort;
        if (!["off", "low", "medium", "high", "xhigh"].includes(effort)) {
          throw new Error("unsupported MTPLX reasoning effort: " + effort);
        }
        clientThinking ??= pi.getThinkingLevel();
        if (pi.getThinkingLevel() !== effort) pi.setThinkingLevel(effort);
        ctx.ui.setStatus("mtplx-controls", "MTPLX controls reasoning: " + effort);
      } else {
        if (clientThinking !== undefined) pi.setThinkingLevel(clientThinking);
        clientThinking = undefined;
        ctx.ui.setStatus("mtplx-controls", undefined);
      }
    } catch (error) {
      ctx.ui.setStatus("mtplx-controls", "MTPLX settings sync failed: " + String(error));
    }
  };
  const syncThinking = (ctx: any) => {
    syncInFlight ??= readSettings(ctx).finally(() => { syncInFlight = undefined; });
    return syncInFlight;
  };
  const startSync = async (_event: any, ctx: any) => {
    if (timer) clearInterval(timer);
    await syncThinking(ctx);
    timer = setInterval(() => { if (ctx.isIdle()) void syncThinking(ctx); }, 3000);
    timer.unref?.();
  };
  pi.on("session_start", startSync);
  pi.on("session_switch", startSync);
  pi.on("model_select", startSync);
  pi.on("before_agent_start", async (_event: any, ctx: any) => { await syncThinking(ctx); });
  pi.on("session_shutdown", () => { if (timer) clearInterval(timer); });

}
'''


def build_pi_request_policy_extension_source(
    model_id: str,
    *,
    uncapped: bool,
    injected_max_tokens: int = PI_INJECTED_DEFAULT_MAX_TOKENS,
) -> str:
    """Build Pi's request/session bridge for the configured MTPLX model.

    Pi serializes the model's advertised ``maxTokens`` (or its legacy default)
    on every request. The extension removes only Pi's generated output
    ceiling for the exact MTPLX model while leaving explicit user caps alone.
    It also gives MTPLX Pi's real session id so prompt-cache reuse is stable.
    """

    model_literal = json.dumps(str(model_id))
    uncapped_literal = "true" if uncapped else "false"
    return f"""// {PI_EXTENSION_MANAGED_MARKER} Pi extension. MTPLX keeps this file up to
// date on every sync. To take ownership (or disable it), edit it and delete
// this marker line: MTPLX never touches the file again once the marker and
// the mtplx identifiers below are gone from it.
const mtplxModelID = {model_literal};
const mtplxUncapped = {uncapped_literal};
const mtplxPiInjectedDefaultMaxTokens = {int(injected_max_tokens)};

export default function (pi: any) {{
  pi.on("before_provider_headers", (event: any, ctx: any) => {{
    const headers = event?.headers;
    if (!headers || typeof headers !== "object") return;
    const client = Object.entries(headers).find(
      ([key]) => key.toLowerCase() === "x-mtplx-client",
    )?.[1];
    if (client !== "pi") return;
    event.headers["x-mtplx-session-id"] = String(
      ctx.sessionManager.getSessionId(),
    );
    const leaf = ctx.sessionManager.getLeafId();
    if (leaf) event.headers["x-mtplx-client-entry-id"] = String(leaf);
  }});

  pi.on("before_provider_request", (event: any) => {{
    const payload = event?.payload;
    if (!mtplxUncapped || !payload || typeof payload !== "object") return;
    if (payload.model !== mtplxModelID) return;
    // Strip only Pi's serialized default ceiling; an explicit user cap (any
    // other value) is honored end to end.
    const request = {{ ...payload }};
    let changed = false;
    if (request.max_tokens === mtplxPiInjectedDefaultMaxTokens) {{
      delete request.max_tokens;
      changed = true;
    }}
    if (request.max_completion_tokens === mtplxPiInjectedDefaultMaxTokens) {{
      delete request.max_completion_tokens;
      changed = true;
    }}
    if (!changed) return;
    return request;
  }});
}}
"""


def write_pi_request_policy_extension(
    *,
    model_id: str,
    uncapped: bool,
    injected_max_tokens: int = PI_INJECTED_DEFAULT_MAX_TOKENS,
    path: str | Path | None = None,
) -> Path:
    """Install the small Pi bridge owned by the MTPLX provider config."""

    extension_path = pi_request_policy_extension_path(path)
    source = build_pi_request_policy_extension_source(
        model_id, uncapped=uncapped, injected_max_tokens=injected_max_tokens,
    )
    return _write_pi_managed_extension(extension_path, source)


def write_pi_settings_extension(path: str | Path | None = None) -> Path:
    extension_path = pi_models_json_path(path).parent / "extensions" / "mtplx-settings-sync.ts"
    return _write_pi_managed_extension(extension_path, build_pi_settings_extension_source())


def _write_pi_managed_extension(extension_path: Path, source: str) -> Path:
    if extension_path.exists():
        try:
            current = extension_path.read_text(encoding="utf-8")
        except OSError:
            current = ""
        managed = (
            PI_EXTENSION_MANAGED_MARKER in current
            or "mtplxPiInjectedDefaultMaxTokens" in current
        )
        if not managed:
            # The user replaced the extension with their own content: it is
            # theirs now. Never overwrite a user-owned file (#282).
            return extension_path
        if current == source:
            return extension_path
    extension_path.parent.mkdir(parents=True, exist_ok=True)
    extension_path.write_text(source, encoding="utf-8")
    try:
        extension_path.chmod(0o600)
    except OSError:
        pass
    return extension_path


def pi_model_ref(model_id: str, *, provider_id: str = PI_PROVIDER_ID) -> str:
    return f"{provider_id}/{model_id}"


def pi_launch_command(model_id: str, *, provider_id: str = PI_PROVIDER_ID) -> str:
    return f"pi --model {pi_model_ref(model_id, provider_id=provider_id)}"


def launch_pi_in_terminal(
    command: str,
    *,
    model_ref: str | None = None,
    client: str = "pi",
    workspace: str | None = None,
) -> dict[str, Any]:
    """Open an interactive client (Pi; Hermes passes ``client="hermes"``) in
    the terminal app the user chose for it (``pi_opens_in`` and
    ``hermes_opens_in`` in config.toml), Terminal by default, in
    ``workspace`` (the folder the command was run from by default).

    Pi is an interactive terminal client, so spawning it as a silent background
    process would be worse UX than doing nothing. Always try to open it: a
    false "already running" is much worse than an extra Pi tab. On non-macOS
    systems, return a clear fallback payload. ``terminal`` in the result is
    the app it opened in, ``note`` why that is not the saved choice.
    """

    from mtplx.client_opening import open_command

    _ = model_ref  # kept for call-site clarity
    return open_command(command, client=client, workspace=workspace)


def build_pi_provider_config(
    *,
    base_url: str,
    model_id: str,
    model_name: str | None = None,
    api_key: str = PI_LOCAL_API_KEY,
    context_window: int = PI_DEFAULT_CONTEXT_WINDOW,
    max_tokens: int | None = PI_DEFAULT_MAX_TOKENS,
    vision: bool = False,
) -> dict[str, Any]:
    """Build the Pi provider block MTPLX needs.

    Pi's OpenAI-compatible transport currently needs the Chat Completions API
    name, a dummy-or-real API key, and compatibility flags so it sends
    ``system`` instead of ``developer`` and ``max_tokens`` instead of the newer
    OpenAI field. The Qwen thinking format wires Pi's thinking-level picker to
    the server's ``enable_thinking``/``reasoning_effort`` request fields.
    """

    model_config: dict[str, Any] = {
        "id": str(model_id),
        "name": model_name or f"MTPLX {model_id}",
        "reasoning": True,
        # Pi's effort ladder is off/minimal/low/medium/high/xhigh/max; the
        # MTPLX vocabulary is low/medium/high/xhigh (mtplx/reasoning_effort.py)
        # and the server narrows to the loaded family's declared tiers.
        # "minimal": null hides Pi's duplicate below-low tier; "xhigh" must be
        # mapped to appear in Pi's picker at all (Qwen 3.8's top tier); "max"
        # stays unmapped, so hidden. Unmapped levels pass through verbatim.
        "thinkingLevelMap": {
            "minimal": None,
            "xhigh": "xhigh",
        },
        # Engine capability, not a preference: Pi only offers/sends image
        # parts when this lists "image". A hardcoded ["text"] here kept Pi
        # text-only even for vision-enabled packs (issue #328) while the same
        # model served images fine through the built-in chat.
        "input": ["text", "image"] if vision else ["text"],
        "contextWindow": int(context_window),
        "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0,
        },
    }
    # Pi requires output metadata and otherwise silently substitutes 16,384.
    # Advertise the real context ceiling; the MTPLX-owned request extension
    # omits the generated wire cap when the user did not explicitly request one.
    model_config["maxTokens"] = int(
        context_window if max_tokens is None else max_tokens
    )

    return {
        "baseUrl": str(base_url).rstrip("/"),
        "api": "openai-completions",
        "apiKey": str(api_key),
        "authHeader": True,
        "headers": {
            "x-mtplx-client": "pi",
        },
        # Pi 0.84.x with thinkingFormat "qwen" serializes exactly the fields
        # the MTPLX server accepts: top-level ``enable_thinking`` (true when a
        # thinking level is selected, false for Pi's "off" level) plus
        # ``reasoning_effort`` mapped through thinkingLevelMap
        # (pi-ai openai-completions buildParams). Pi's default level is
        # "medium" — the Qwen 3.8 family coding default.
        "compat": {
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": True,
            "thinkingFormat": "qwen",
            "maxTokensField": "max_tokens",
        },
        "models": [model_config],
    }


def _unique_backup(path: Path, reason: str) -> Path:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.{reason}-{stamp}.bak")
    counter = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}.{reason}-{stamp}-{counter}.bak")
        counter += 1
    return backup


def _fill_missing_deep(existing: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """Existing user values win; defaults only fill gaps, recursively."""

    merged = dict(existing)
    for key, default_value in defaults.items():
        if key not in merged:
            merged[key] = default_value
        elif isinstance(merged[key], dict) and isinstance(default_value, dict):
            merged[key] = _fill_missing_deep(merged[key], default_value)
    return merged


def _window_fields_written_by_mtplx(entry: dict[str, Any]) -> bool:
    """Whether a model entry's ``contextWindow``/``maxTokens`` pair is MTPLX's.

    MTPLX writes ``maxTokens`` equal to the window. Any other pair is a user
    edit and survives every sync (#282). SYNC PAIR:
    ``PiIntegration.windowFieldsWereWrittenByMTPLX``.
    """

    window = entry.get("contextWindow")
    max_tokens = entry.get("maxTokens")
    if not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (window, max_tokens)
    ):
        return False
    return window > 0 and max_tokens == window


def merge_pi_provider_config(
    existing_provider: Any,
    fresh: dict[str, Any],
    *,
    refresh_window: bool = True,
) -> dict[str, Any]:
    """User-preserving merge of the MTPLX provider block (#282 clobber fix).

    MTPLX owns the connection identity (``baseUrl``/``api``/``apiKey``/
    ``authHeader`` and the ``x-mtplx-client`` header) because ports move
    between launches and the integration must keep working. Every other key
    the user edited wins: our values only fill missing keys, recursively.
    Model entries merge by ``id`` the same way, and user-added models or
    fields (custom ``thinkingLevelMap``, explicit ``maxTokens``) survive a
    sync untouched. ``input`` is half an exception: it states what the
    ENGINE supports, so when the engine POSITIVELY knows the pack does vision
    the fresh ``["text", "image"]`` wins — a stale ``["text"]`` written by a
    pre-vision MTPLX must not outlive the engine that wrote it (issue #328).
    When the engine does NOT advertise vision (which includes "could not
    resolve the model dir"), a user-taught ``input`` survives like any other
    edit (#282): the user may proxy to a capable endpoint, and an engine
    "unknown" must never delete what a human wrote.

    A ``contextWindow``/``maxTokens`` pair MTPLX wrote takes the fresh
    window with ``refresh_window``; a pair the user edited stays theirs.
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
    for key in PI_OWNED_PROVIDER_CONNECTION_KEYS:
        if key in fresh:
            merged[key] = fresh[key]
    # ``compat`` is MTPLX's transport contract with Pi (which wire fields the
    # server supports), not a user preference: a stale block written by an
    # older MTPLX must not outlive the engine that wrote it. Receipt
    # 2026-08-28: a 2.9.x-era ``supportsReasoningEffort: false`` survived
    # every re-sync and silently killed Pi's effort dial after upgrade. Our
    # keys win; user-added extra compat keys still survive.
    existing_compat = (
        dict(existing_provider.get("compat"))
        if isinstance(existing_provider.get("compat"), dict)
        else {}
    )
    existing_compat.update(fresh.get("compat") or {})
    if existing_compat:
        merged["compat"] = existing_compat
    headers = {
        key: value
        for key, value in (
            existing_provider.get("headers") or {}
        ).items()
        if str(key).lower() != "x-mtplx-client"
    } if isinstance(existing_provider.get("headers"), dict) else {}
    headers.update(fresh.get("headers") or {})
    merged["headers"] = headers

    fresh_models = fresh.get("models") or []
    existing_models = existing_provider.get("models")
    if not isinstance(existing_models, list):
        merged["models"] = fresh_models
        return merged
    fresh_ids = {str(model.get("id")) for model in fresh_models}
    # Stale MTPLX-owned entries (our own previous model ids, always
    # "mtplx-"-prefixed) are pruned so switching models does not pile up
    # dead picker rows; user-added models never match the prefix and stay.
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
                merged_model = _fill_missing_deep(entry, fresh_model)
                # ``input`` upgrade rule (see docstring): the engine's
                # positive vision knowledge wins; its absence never
                # downgrades a user-taught value.
                fresh_input = fresh_model.get("input")
                if isinstance(fresh_input, list) and "image" in fresh_input:
                    merged_model["input"] = fresh_input
                if refresh_window and _window_fields_written_by_mtplx(entry):
                    merged_model["contextWindow"] = fresh_model.get("contextWindow")
                    merged_model["maxTokens"] = fresh_model.get("maxTokens")
                result_models[index] = merged_model
                break
        else:
            result_models.append(fresh_model)
    merged["models"] = result_models
    return merged


def merge_pi_models_config(
    existing: dict[str, Any] | None,
    *,
    provider_config: dict[str, Any],
    provider_id: str = PI_PROVIDER_ID,
    refresh_window: bool = True,
) -> dict[str, Any]:
    """Merge or create a Pi ``models.json`` payload.

    MTPLX owns only the ``providers.mtplx`` block, and inside it only the
    connection identity: user edits within the block are preserved via
    :func:`merge_pi_provider_config`. Other providers are untouched.
    """

    payload = dict(existing or {})
    providers = payload.get("providers")
    if not isinstance(providers, dict):
        providers = {}
    else:
        providers = dict(providers)
    providers[str(provider_id)] = merge_pi_provider_config(
        providers.get(str(provider_id)),
        provider_config,
        refresh_window=refresh_window,
    )
    payload["providers"] = providers
    return payload


def _write_models_json(
    config_path: Path, existing: dict[str, Any] | None, merged: dict[str, Any]
) -> tuple[bool, Path | None]:
    """Write ``merged`` unless it equals ``existing``; return (written, backup).

    Unchanged content leaves the file exactly as the user wrote it. A
    rewrite keeps the previous file next to it.
    """

    written = existing is None or merged != existing
    backup_path: Path | None = None
    if written:
        if existing is not None:
            backup_path = _unique_backup(config_path, "before-mtplx")
            shutil.copy2(config_path, backup_path)
        config_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    try:
        config_path.chmod(0o600)
    except OSError:
        pass
    return written, backup_path


def pi_lists_model(
    model_id: str,
    *,
    path: str | Path | None = None,
    provider_id: str = PI_PROVIDER_ID,
) -> bool:
    """Whether Pi's config lists ``model_id`` under MTPLX's provider.

    Raises ``InvalidConfigFile`` for a file Pi could not read either.
    """

    config_path = pi_models_json_path(path)
    if not config_path.exists():
        return False
    existing, _existing_text = load_config_file(config_path)
    providers = existing.get("providers")
    provider = providers.get(provider_id) if isinstance(providers, dict) else None
    models = provider.get("models") if isinstance(provider, dict) else None
    return isinstance(models, list) and any(
        isinstance(entry, dict) and str(entry.get("id")) == str(model_id)
        for entry in models
    )


def refresh_pi_models_window(
    model_id: str,
    window: int,
    *,
    path: str | Path | None = None,
    provider_id: str = PI_PROVIDER_ID,
) -> dict[str, Any]:
    """Give MTPLX's own Pi entry for ``model_id`` the window the server serves.

    That window (``served_execution_window``) exists only once the model is
    loaded, so the CLI's Pi handoff calls this after startup, as the app
    re-syncs Pi once its daemon answers. Only a pair MTPLX wrote changes;
    the entry's other fields, other models and a pair the user edited stay.
    """

    config_path = pi_models_json_path(path)
    result: dict[str, Any] = {
        "config_path": str(config_path),
        "context_window": int(window),
        "written": False,
        "backup_path": None,
    }
    if int(window) <= 0 or not config_path.exists():
        return result
    existing, _existing_text = load_config_file(config_path)
    providers = existing.get("providers")
    provider = providers.get(provider_id) if isinstance(providers, dict) else None
    models = provider.get("models") if isinstance(provider, dict) else None
    if not isinstance(models, list):
        return result
    refreshed = [
        {**entry, "contextWindow": int(window), "maxTokens": int(window)}
        if isinstance(entry, dict)
        and str(entry.get("id")) == str(model_id)
        and _window_fields_written_by_mtplx(entry)
        else entry
        for entry in models
    ]
    merged = {
        **existing,
        "providers": {**providers, provider_id: {**provider, "models": refreshed}},
    }
    written, backup_path = _write_models_json(config_path, existing, merged)
    result["written"] = written
    result["backup_path"] = str(backup_path) if backup_path is not None else None
    return result


def write_pi_models_config(
    *,
    base_url: str,
    model_id: str,
    model_name: str | None = None,
    api_key: str = PI_LOCAL_API_KEY,
    path: str | Path | None = None,
    provider_id: str = PI_PROVIDER_ID,
    context_window: int = PI_DEFAULT_CONTEXT_WINDOW,
    max_tokens: int | None = PI_DEFAULT_MAX_TOKENS,
    vision: bool = False,
    keep_window: bool = False,
) -> dict[str, Any]:
    """Write the MTPLX provider into Pi's config and return a handoff payload.

    ``keep_window`` leaves the window of an existing MTPLX entry alone: the
    write before the server starts has only a guess at the window, and the
    handoff then brings it to the served one (``refresh_pi_models_window``),
    so a launch never rewrites it twice.
    """

    config_path = pi_models_json_path(path)
    existing: dict[str, Any] | None = None
    if config_path.exists():
        # Pi strips // comments and trailing commas from models.json, so MTPLX
        # reads it the same way. A file that still does not parse is the
        # user's to fix: InvalidConfigFile propagates and nothing here is
        # moved or written.
        existing, _existing_text = load_config_file(config_path)

    provider_config = build_pi_provider_config(
        base_url=base_url,
        model_id=model_id,
        model_name=model_name,
        api_key=api_key,
        context_window=context_window,
        max_tokens=max_tokens,
        vision=vision,
    )
    merged = merge_pi_models_config(
        existing,
        provider_config=provider_config,
        provider_id=provider_id,
        refresh_window=not keep_window,
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    written, backup_path = _write_models_json(config_path, existing, merged)
    # With keep_window the file may keep another window than the one passed.
    written_window = next(
        (
            entry.get("contextWindow")
            for entry in merged["providers"][str(provider_id)].get("models") or []
            if isinstance(entry, dict) and str(entry.get("id")) == str(model_id)
        ),
        None,
    )
    request_policy_extension_path = write_pi_request_policy_extension(
        model_id=model_id,
        uncapped=max_tokens is None,
        injected_max_tokens=int(provider_config["models"][0]["maxTokens"]),
        path=config_path,
    )
    settings_extension_path = write_pi_settings_extension(config_path)
    return {
        "config_path": str(config_path),
        "backup_path": str(backup_path) if backup_path is not None else None,
        "provider_id": provider_id,
        "base_url": provider_config["baseUrl"],
        "model_id": model_id,
        "model_ref": pi_model_ref(model_id, provider_id=provider_id),
        "launch_command": pi_launch_command(model_id, provider_id=provider_id),
        "api_key": api_key,
        "context_window": (
            written_window if isinstance(written_window, int) else int(context_window)
        ),
        "max_tokens": None if max_tokens is None else int(max_tokens),
        "no_hidden_max_tokens": max_tokens is None,
        "request_policy_extension_path": str(request_policy_extension_path),
        "settings_extension_path": str(settings_extension_path),
        "uncapped_request_policy": max_tokens is None,
        "written": written,
    }
