"""OpenCode Desktop integration helpers.

The public CLI uses this module to make ``mtplx start opencode`` a real
connection flow: merge an MTPLX OpenAI-compatible provider into OpenCode's
JSON config, point OpenCode at the local MTPLX server, then launch OpenCode
when possible.
"""

from __future__ import annotations

import datetime
import base64
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mtplx.jsonc import load_config_file

OPENCODE_PROVIDER_ID = "mtplx"
OPENCODE_NPM_PACKAGE = "@ai-sdk/openai-compatible"
OPENCODE_DEFAULT_CONTEXT_WINDOW = 262_144
OPENCODE_DEFAULT_CHUNK_TIMEOUT_MS = 900_000
# OpenCode's own injected output ceiling when the user never set a cap. The
# plugin strips exactly this value, and below a 64K window the smaller reply
# reserve MTPLX writes as limit.output (opencode_output_limit): anything
# else is a deliberate client cap and must reach MTPLX intact.
# Receipts: sst/opencode v1.18.21
# provider/transform.ts `OUTPUT_TOKEN_MAX = 32_000` (min'd against
# limit.output on every request), and request-log-8002.jsonl records 313-327
# all showing request_max_tokens=32000. The earlier 32_768 guess never
# matched the wire, so the guard silently stripped nothing.
OPENCODE_INJECTED_OUTPUT_CAP = 32_000
# The answer cap a user asked for with --max-response-tokens, written as the
# MTPLX model's own request header. OpenCode hands the model, headers
# included, to the plugin, which sends this cap as the request's max_tokens:
# it reaches a server that was started without the flag, and it can never be
# mistaken for the reply reserve MTPLX writes as limit.output. OpenCode also
# sends the header on every request; the server ignores it.
OPENCODE_REQUESTED_OUTPUT_HEADER = "x-mtplx-max-response-tokens"


def opencode_output_limit(context_window: int, requested: int | None = None) -> int:
    """``limit.output`` for the MTPLX model: OpenCode's reply reserve.

    OpenCode 1.18.29 keeps ``min(limit.output, 32_000)`` of ``limit.context``
    for the reply and compacts the moment a turn's total tokens reach the
    rest (session/overflow.ts ``usable`` / ``isOverflow``). Mirroring the
    context into ``limit.output`` therefore left a zero-token conversation
    window on any context <= 32K (8,192 on a 32 GB seat), and the compaction
    agent ran after every reply (issue #480: 48 summaries in 98 turns, no
    turn past 7,801 tokens). By default reserve at most half the window,
    capped at the 32,000 OpenCode injects on large windows. OpenCode also
    sends this reserve as every request's max_tokens; the session-headers
    plugin strips it there (``mtplxReserve``), so the server's own limits
    apply at every window. SYNC:
    ``OpenCodeIntegration.outputLimit(forContextWindow:)``.

    ``requested`` is an answer cap the user asked for (--max-response-tokens)
    and is written as is: OpenCode then keeps that much of the window for
    each reply, the user's own trade against compaction.
    """
    if requested is not None and int(requested) > 0:
        return int(requested)
    context = max(1, int(context_window))
    return max(1, min(OPENCODE_INJECTED_OUTPUT_CAP, context // 2))


def opencode_requested_output(model: Any) -> int | None:
    """The answer cap recorded on an MTPLX model entry, if the user set one."""

    headers = model.get("headers") if isinstance(model, dict) else None
    value = headers.get(OPENCODE_REQUESTED_OUTPUT_HEADER) if isinstance(headers, dict) else None
    try:
        cap = int(str(value))
    except (TypeError, ValueError):
        return None
    return cap if cap > 0 else None
# OpenCode <= 1.18.20 (including Desktop 1.18.18) injects a qwen-keyed
# sampler for any model id containing "qwen" (provider/transform.ts
# `temperature()`/`topP()` at v1.18.18); 1.18.21 removed the rule. The plugin
# strips exactly this injected pair so the server's family-native sampler
# (the app's source of truth) applies; any other value is a deliberate
# client choice and passes through.
OPENCODE_INJECTED_QWEN_TEMPERATURE = 0.55
OPENCODE_INJECTED_QWEN_TOP_P = 1
# OpenCode's built-in effort tiers for reasoning-capable openai-compatible
# models (provider/transform.ts OPENAI_EFFORTS at v1.18.18/v1.18.21). The
# generated config disables the tiers a family contract does not define so
# OpenCode's effort picker mirrors the MTPLX dial exactly.
OPENCODE_OPENAI_COMPATIBLE_DEFAULT_EFFORTS = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
)
OPENCODE_SESSION_HEADERS_PLUGIN_NAME = "mtplx-session-headers.js"
OPENCODE_SESSION_HEADERS_PACKAGE_NAME = "mtplx-session-headers"
OPENCODE_DESKTOP_SETTINGS_STORE_NAME = "default.dat"
OPENCODE_DESKTOP_SETTINGS_KEY = "settings.v3"
OPENCODE_DESKTOP_GLOBAL_STORE_NAME = "opencode.global.dat"
OPENCODE_SESSION_HEADERS_PLUGIN_SOURCE = (
    """const mtplxProviderID = (input) =>
  input?.model?.providerID || input?.provider?.id;

const mtplxInjectedOutputCap = __MTPLX_INJECTED_OUTPUT_CAP__;
const mtplxRequestedOutputHeader = "__MTPLX_REQUESTED_OUTPUT_HEADER__";
const mtplxInjectedQwenTemperature = __MTPLX_INJECTED_QWEN_TEMPERATURE__;
const mtplxInjectedQwenTopP = __MTPLX_INJECTED_QWEN_TOP_P__;

export const MTPLXSessionHeaders = async () => ({
  "chat.headers": async (input, output) => {
    output.headers ||= {};
    const providerID = mtplxProviderID(input);
    if (providerID && providerID !== "mtplx") return;
    output.headers["x-mtplx-client"] = "opencode";
    if (input?.sessionID) {
      output.headers["x-mtplx-session-id"] = String(input.sessionID);
    }
    if (input?.message?.id) {
      output.headers["x-mtplx-client-turn-id"] = String(input.message.id);
    }
  },
  "chat.params": async (input, output) => {
    const providerID = mtplxProviderID(input);
    if (providerID && providerID !== "mtplx") return;
    // OpenCode sends maxOutputTokens = min(limit.output, 32000) with every
    // request and hands this hook the model with its limit and headers.
    // MTPLX writes limit.output as OpenCode's reply reserve, half the window
    // and at most 32,000 (#480), so by default that value is OpenCode's own
    // and is stripped: the server's own limits apply. An answer cap the user
    // asked MTPLX for (--max-response-tokens) is the model's
    // x-mtplx-max-response-tokens header and is sent whole, past OpenCode's
    // 32,000 ceiling; a limit.output that is not MTPLX's reserve is sent the
    // same way. Any other value is a cap set in OpenCode itself and passes
    // through untouched.
    const positive = (value) => (Number.isInteger(value) && value > 0 ? value : null);
    const limit = input?.model?.limit;
    const configuredOutput = positive(limit?.output);
    const opencodeDefault = configuredOutput === null
      ? mtplxInjectedOutputCap
      : Math.min(configuredOutput, mtplxInjectedOutputCap);
    if (output.maxOutputTokens === opencodeDefault) {
      const context = positive(limit?.context);
      const mtplxReserve = context === null
        ? null
        : Math.min(mtplxInjectedOutputCap, Math.max(1, Math.floor(context / 2)));
      const requested = positive(Number(input?.model?.headers?.[mtplxRequestedOutputHeader]));
      output.maxOutputTokens = requested
        ?? (configuredOutput !== null && configuredOutput !== mtplxReserve
          ? configuredOutput
          : undefined);
    }
    // OpenCode <= 1.18.20 (Desktop 1.18.18 included) injects a qwen-keyed
    // sampler (temperature 0.55, topP 1) for any model id containing
    // "qwen"; 1.18.21 removed the rule. Strip exactly that injected pair so
    // the MTPLX server's family-native sampler applies; any other value is
    // a deliberate client choice and passes through untouched.
    const modelID = String(input?.model?.id ?? input?.model?.modelID ?? "").toLowerCase();
    if (modelID.includes("qwen")) {
      if (output.temperature === mtplxInjectedQwenTemperature) {
        output.temperature = undefined;
      }
      if (output.topP === mtplxInjectedQwenTopP) {
        output.topP = undefined;
      }
    }
  }
});
export default MTPLXSessionHeaders;
"""
    .replace("__MTPLX_INJECTED_OUTPUT_CAP__", str(OPENCODE_INJECTED_OUTPUT_CAP))
    .replace("__MTPLX_REQUESTED_OUTPUT_HEADER__", OPENCODE_REQUESTED_OUTPUT_HEADER)
    .replace(
        "__MTPLX_INJECTED_QWEN_TEMPERATURE__",
        str(OPENCODE_INJECTED_QWEN_TEMPERATURE),
    )
    .replace("__MTPLX_INJECTED_QWEN_TOP_P__", str(OPENCODE_INJECTED_QWEN_TOP_P))
)


OPENCODE_SESSION_HEADERS_V2_SOURCE = """// Older V1 imports index.js; modern V1 and V2 resolve this entrypoint.
// No prompt, tool-schema or generation-option rewriting belongs here.
import { MTPLXSessionHeaders } from "./index.js";
export default {
  id: "mtplx.session-headers",
  server: MTPLXSessionHeaders,
  async setup(ctx) {
    await ctx.session.hook("model.request", (event) => {
      event.headers["x-mtplx-client"] = "opencode";
      event.headers["x-mtplx-session-id"] = String(event.sessionID);
      if (event.kind) event.headers["x-mtplx-request-kind"] = String(event.kind);
    }, { providerID: "mtplx" });
  }
};
"""


def opencode_config_path(path: str | Path | None = None) -> Path:
    """Return OpenCode's JSON config path.

    ``MTPLX_OPENCODE_CONFIG`` exists for tests and power users. Normal users
    get OpenCode's shared config path under ``~/.config/opencode``.
    """

    if path is not None:
        return Path(path).expanduser()
    env = os.environ.get("MTPLX_OPENCODE_CONFIG")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".config" / "opencode" / "opencode.json"


def opencode_session_headers_plugin_path(path: str | Path | None = None) -> Path:
    """Return the managed package, discoverable by V2 and configured for V1."""

    return (
        opencode_config_path(path).parent / "plugins"
        / OPENCODE_SESSION_HEADERS_PACKAGE_NAME
    )


def opencode_desktop_settings_store_path(path: str | Path | None = None) -> Path:
    """Return OpenCode Desktop's renderer persisted-settings store path."""

    if path is not None:
        return Path(path).expanduser()
    env = os.environ.get("MTPLX_OPENCODE_DESKTOP_SETTINGS_STORE")
    if env:
        return Path(env).expanduser()
    return (
        Path.home()
        / "Library"
        / "Application Support"
        / "ai.opencode.desktop"
        / OPENCODE_DESKTOP_SETTINGS_STORE_NAME
    )


def opencode_desktop_app_support_path(path: str | Path | None = None) -> Path:
    """Return OpenCode Desktop's application support directory."""

    if path is not None:
        return Path(path).expanduser()
    env = os.environ.get("MTPLX_OPENCODE_DESKTOP_APP_SUPPORT")
    if env:
        return Path(env).expanduser()
    return (
        Path.home()
        / "Library"
        / "Application Support"
        / "ai.opencode.desktop"
    )


def opencode_model_ref(model_id: str, *, provider_id: str = OPENCODE_PROVIDER_ID) -> str:
    return f"{provider_id}/{model_id}"


def detect_opencode_desktop() -> dict[str, Any]:
    """Best-effort OpenCode Desktop detection for UX messages.

    Launching through macOS ``open -a`` is still attempted even when this
    returns missing; Spotlight/app registration can know about apps outside
    the common Applications paths.
    """

    if sys.platform != "darwin":
        return {"available": False, "kind": "unsupported_platform"}
    candidates = [
        Path("/Applications/OpenCode.app"),
        Path.home() / "Applications" / "OpenCode.app",
        Path("/Applications/OpenCode Desktop.app"),
        Path.home() / "Applications" / "OpenCode Desktop.app",
    ]
    for candidate in candidates:
        if candidate.exists():
            return {"available": True, "kind": "app", "path": str(candidate)}
    if shutil.which("opencode"):
        return {"available": True, "kind": "cli", "path": shutil.which("opencode")}
    return {"available": False, "kind": "not_found"}


def launch_opencode_app() -> dict[str, Any]:
    """Open OpenCode Desktop without blocking the MTPLX server."""

    if sys.platform != "darwin":
        return {
            "ok": False,
            "status": "unsupported_platform",
            "error": "automatic OpenCode launch currently requires macOS",
        }
    state_repair = repair_opencode_desktop_state()
    for app_name in ("OpenCode", "OpenCode Desktop"):
        try:
            subprocess.Popen(
                ["open", "-a", app_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return {
                "ok": True,
                "status": "launched",
                "app": app_name,
                "desktop_state_repair": state_repair,
            }
        except OSError as exc:
            last_error = str(exc)
    return {
        "ok": False,
        "status": "launch_failed",
        "error": last_error,
        "desktop_state_repair": state_repair,
    }


def _opencode_effort_variants(
    effort_levels: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Mirror the family effort dial into OpenCode's variant picker.

    Disable the built-in tiers the family contract does not define, and write
    an EXPLICIT variant for every tier it does. The earlier form declared a
    family tier only when it was outside OPENAI_EFFORTS, trusting OpenCode to
    surface the rest — but Desktop 1.18.21's picker does not offer its full
    built-in list for a custom openai-compatible provider (observed live:
    xhigh missing for the Flash-Next dial while low/medium rendered). An
    explicit `{"reasoningEffort": tier}` variant always renders and merges
    over any same-named built-in, so declaring every tier is correct on both
    behaviors.
    """

    allowed = {str(level) for level in effort_levels}
    variants: dict[str, dict[str, Any]] = {
        effort: {"disabled": True}
        for effort in OPENCODE_OPENAI_COMPATIBLE_DEFAULT_EFFORTS
        if effort not in allowed
    }
    for level in effort_levels:
        variants[str(level)] = {"reasoningEffort": str(level)}
    return variants


def build_opencode_provider_config(
    *,
    base_url: str,
    model_id: str,
    model_name: str | None = None,
    api_key: str | None = None,
    context_window: int = OPENCODE_DEFAULT_CONTEXT_WINDOW,
    output_limit: int | None = None,
    chunk_timeout_ms: int = OPENCODE_DEFAULT_CHUNK_TIMEOUT_MS,
    enable_thinking: bool = True,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int | None = None,
    reasoning_effort: str | None = None,
    reasoning_effort_levels: Sequence[str] | None = None,
    vision: bool = False,
) -> dict[str, Any]:
    """Build the OpenCode provider/config fragment MTPLX owns.

    OpenCode's `limit` object is model metadata, not a server-side generation
    cap. We intentionally do not write hidden maxTokens/maxOutput caps:
    ``output_limit`` is an answer cap the user asked for
    (--max-response-tokens), written as ``limit.output`` and as the model's
    ``x-mtplx-max-response-tokens`` header, which the session-headers plugin
    sends as the request's cap.

    ``reasoning``/``temperature`` are declared capable so OpenCode round-trips
    assistant reasoning_content (preserve_thinking) and transmits explicit
    client-side choices; with nothing chosen, OpenCode 1.18.21 sends no
    sampler for MTPLX model ids and the server's family defaults (the app's
    source of truth) apply. ``reasoning_effort`` is the app's current dial and
    rides per-model ``options.reasoningEffort`` (@ai-sdk/openai-compatible
    maps it to the wire's ``reasoning_effort``); an effort variant picked
    inside OpenCode merges after model options and wins for that request.
    The family sampler args are accepted for caller symmetry but deliberately
    not written: @ai-sdk/openai-compatible 2.0.41 has no per-model sampler
    transport (its provider-options schema is user/reasoningEffort/
    textVerbosity/strictJsonSchema only), so writing them would be dead
    config posing as policy.
    """

    context = int(context_window or OPENCODE_DEFAULT_CONTEXT_WINDOW)
    output = opencode_output_limit(context, output_limit)
    _ = (temperature, top_p, top_k)
    options: dict[str, Any] = {
        "baseURL": str(base_url).rstrip("/"),
        "timeout": False,
        "chunkTimeout": int(chunk_timeout_ms),
        "headers": {
            "x-mtplx-client": "opencode",
        },
    }
    if api_key:
        options["apiKey"] = str(api_key)
    model: dict[str, Any] = {
        "name": model_name or f"MTPLX {model_id}",
        "reasoning": bool(enable_thinking),
        "tool_call": True,
        "temperature": True,
        "limit": {
            "context": context,
            "output": output,
        },
        "modalities": {
            "input": ["text", "image"] if vision else ["text"],
            "output": ["text"],
        },
    }
    if output_limit is not None and int(output_limit) > 0:
        model["headers"] = {OPENCODE_REQUESTED_OUTPUT_HEADER: str(int(output_limit))}
    if enable_thinking:
        if reasoning_effort:
            model["options"] = {"reasoningEffort": str(reasoning_effort)}
        if reasoning_effort_levels is not None:
            variants = _opencode_effort_variants(reasoning_effort_levels)
            if variants:
                model["variants"] = variants
    return {
        "provider": {
            OPENCODE_PROVIDER_ID: {
                "npm": OPENCODE_NPM_PACKAGE,
                "name": "MTPLX (local)",
                "options": options,
                "models": {
                    str(model_id): model,
                },
            }
        },
        "model": opencode_model_ref(str(model_id)),
        "small_model": opencode_model_ref(str(model_id)),
    }


def merge_opencode_config(
    existing: dict[str, Any] | None,
    *,
    config_fragment: dict[str, Any],
    provider_id: str = OPENCODE_PROVIDER_ID,
    session_headers_plugin_path: str | Path | None = None,
) -> dict[str, Any]:
    """Merge or create OpenCode config while preserving unrelated providers."""

    payload = dict(existing or {})
    providers = payload.get("provider")
    if not isinstance(providers, dict):
        providers = {}
    else:
        providers = dict(providers)
    fragment_providers = config_fragment.get("provider")
    if not isinstance(fragment_providers, dict) or provider_id not in fragment_providers:
        raise ValueError(f"config_fragment must include provider.{provider_id}")
    providers[str(provider_id)] = fragment_providers[provider_id]
    payload["provider"] = providers
    payload["model"] = config_fragment["model"]
    payload["small_model"] = config_fragment["small_model"]
    if session_headers_plugin_path is not None:
        plugin_path = str(Path(session_headers_plugin_path).expanduser())
        existing_plugins = payload.get("plugin")
        if isinstance(existing_plugins, list):
            plugins = list(existing_plugins)
        elif isinstance(existing_plugins, str):
            plugins = [existing_plugins]
        elif existing_plugins is None:
            plugins = []
        else:
            plugins = [existing_plugins]
        # Canonicalize: stale copies of the managed plugin registered under
        # other paths would double-fire the hooks, so keep exactly one entry
        # at the managed location.
        plugins = [
            item
            for item in plugins
            if not (
                isinstance(item, str)
                and item != plugin_path
                and Path(item).name in {
                    OPENCODE_SESSION_HEADERS_PLUGIN_NAME,
                    OPENCODE_SESSION_HEADERS_PACKAGE_NAME,
                }
            )
        ]
        if plugin_path not in [item for item in plugins if isinstance(item, str)]:
            plugins.append(plugin_path)
        payload["plugin"] = plugins
    return payload


def _opencode_project_key_to_path(key: str) -> str | None:
    project_key = str(key).split("/ses_", 1)[0]
    if not project_key:
        return None
    padding = "=" * (-len(project_key) % 4)
    try:
        decoded = base64.urlsafe_b64decode((project_key + padding).encode()).decode()
    except Exception:
        return None
    if not decoded.startswith("/"):
        return None
    return decoded


def _remove_missing_session_keys(value: Any, missing_session_ids: set[str]) -> tuple[Any, int]:
    if not isinstance(value, dict) or not missing_session_ids:
        return value, 0
    next_value: dict[str, Any] = {}
    removed = 0
    for key, item in value.items():
        key_text = str(key)
        if any(session_id in key_text for session_id in missing_session_ids):
            removed += 1
            continue
        next_value[key] = item
    return next_value, removed


def _repair_global_store_payload(
    root: dict[str, Any],
    *,
    missing_paths: set[str],
) -> tuple[dict[str, Any], int]:
    changed_entries = 0
    payload = dict(root)

    layout_text = payload.get("layout")
    if isinstance(layout_text, str):
        try:
            layout = json.loads(layout_text)
        except json.JSONDecodeError:
            layout = None
        if isinstance(layout, dict):
            for section_name in ("sessionTabs", "sessionView"):
                section = layout.get(section_name)
                if isinstance(section, dict):
                    next_section = {}
                    for key, value in section.items():
                        decoded_path = _opencode_project_key_to_path(str(key))
                        if decoded_path in missing_paths:
                            changed_entries += 1
                            continue
                        next_section[key] = value
                    layout[section_name] = next_section
            if changed_entries:
                payload["layout"] = json.dumps(layout, separators=(",", ":"))

    page_text = payload.get("layout.page")
    if isinstance(page_text, str):
        try:
            page = json.loads(page_text)
        except json.JSONDecodeError:
            page = None
        if isinstance(page, dict):
            last_project_session = page.get("lastProjectSession")
            if isinstance(last_project_session, dict):
                next_last = {
                    key: value
                    for key, value in last_project_session.items()
                    if key not in missing_paths
                }
                changed_entries += len(last_project_session) - len(next_last)
                page["lastProjectSession"] = next_last
            for map_name in ("workspaceOrder", "workspaceName", "workspaceBranchName", "workspaceExpanded"):
                current = page.get(map_name)
                if isinstance(current, dict):
                    next_map = {
                        key: value for key, value in current.items() if key not in missing_paths
                    }
                    changed_entries += len(current) - len(next_map)
                    page[map_name] = next_map
            payload["layout.page"] = json.dumps(page, separators=(",", ":"))

    server_text = payload.get("server")
    if isinstance(server_text, str):
        try:
            server = json.loads(server_text)
        except json.JSONDecodeError:
            server = None
        if isinstance(server, dict):
            projects = server.get("projects")
            if isinstance(projects, dict):
                for group, entries in list(projects.items()):
                    if not isinstance(entries, list):
                        continue
                    next_entries = []
                    for entry in entries:
                        if isinstance(entry, dict) and entry.get("worktree") in missing_paths:
                            changed_entries += 1
                            continue
                        next_entries.append(entry)
                    projects[group] = next_entries
            payload["server"] = json.dumps(server, separators=(",", ":"))

    return payload, changed_entries


def repair_opencode_desktop_state(
    app_support_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Remove dead project references from OpenCode Desktop renderer state.

    OpenCode Desktop can get stuck on its splash screen when its saved layout
    tries to reopen a project directory that no longer exists. MTPLX only
    removes renderer references to missing workspaces; it does not delete
    OpenCode history or database rows.
    """

    support = opencode_desktop_app_support_path(app_support_dir)
    global_store = support / OPENCODE_DESKTOP_GLOBAL_STORE_NAME
    if not global_store.exists():
        return {
            "status": "missing_store",
            "path": str(global_store),
            "did_change": False,
            "backup_path": None,
            "removed_entries": 0,
            "missing_paths": [],
        }

    try:
        root = json.loads(global_store.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "status": "unreadable_store",
            "path": str(global_store),
            "did_change": False,
            "backup_path": None,
            "removed_entries": 0,
            "missing_paths": [],
        }
    if not isinstance(root, dict):
        return {
            "status": "unsupported_store",
            "path": str(global_store),
            "did_change": False,
            "backup_path": None,
            "removed_entries": 0,
            "missing_paths": [],
        }

    candidate_paths: set[str] = set()
    for key in ("layout", "layout.page", "server"):
        value = root.get(key)
        if not isinstance(value, str):
            continue
        try:
            text = json.dumps(json.loads(value))
        except json.JSONDecodeError:
            text = value
        for token in text.replace('":"', '": "').replace('","', '", "').split('"'):
            if token.startswith("/") and ("/private/tmp/" in token or token.startswith(str(Path.home()))):
                candidate_paths.add(token)
        if key == "layout":
            try:
                layout = json.loads(value)
            except json.JSONDecodeError:
                layout = {}
            if isinstance(layout, dict):
                for section_name in ("sessionTabs", "sessionView"):
                    section = layout.get(section_name)
                    if isinstance(section, dict):
                        for project_key in section:
                            decoded = _opencode_project_key_to_path(str(project_key))
                            if decoded:
                                candidate_paths.add(decoded)

    missing_paths = {
        path
        for path in candidate_paths
        if path.startswith("/")
        and not Path(path).exists()
    }
    if not missing_paths:
        return {
            "status": "clean",
            "path": str(global_store),
            "did_change": False,
            "backup_path": None,
            "removed_entries": 0,
            "missing_paths": [],
        }

    repaired, removed_entries = _repair_global_store_payload(root, missing_paths=missing_paths)
    if removed_entries <= 0:
        return {
            "status": "no_matching_entries",
            "path": str(global_store),
            "did_change": False,
            "backup_path": None,
            "removed_entries": 0,
            "missing_paths": sorted(missing_paths),
        }

    backup = _unique_backup(global_store, "dead-workspaces")
    shutil.copy2(global_store, backup)
    global_store.write_text(json.dumps(repaired, indent=2) + "\n", encoding="utf-8")
    try:
        global_store.chmod(0o600)
    except OSError:
        pass
    return {
        "status": "repaired",
        "path": str(global_store),
        "did_change": True,
        "backup_path": str(backup),
        "removed_entries": removed_entries,
        "missing_paths": sorted(missing_paths),
    }


def _unique_backup(path: Path, reason: str) -> Path:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.{reason}-{stamp}.bak")
    counter = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}.{reason}-{stamp}-{counter}.bak")
        counter += 1
    return backup


def _write_config_json(
    config_path: Path, existing: dict[str, Any] | None, merged: dict[str, Any]
) -> tuple[bool, Path | None]:
    """Write ``merged`` unless it equals ``existing``; return (written, backup).

    A file whose content already matches is left untouched, comments and
    formatting included. When a rewrite is needed the previous file is kept
    next to it and the copy's path is reported so every renderer can say so.
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


def _mtplx_model_entry(
    config: dict[str, Any] | None, provider_id: str, model_id: str
) -> dict[str, Any] | None:
    providers = (config or {}).get("provider")
    provider = providers.get(provider_id) if isinstance(providers, dict) else None
    models = provider.get("models") if isinstance(provider, dict) else None
    model = models.get(str(model_id)) if isinstance(models, dict) else None
    return model if isinstance(model, dict) else None


def opencode_selects_model(
    model_id: str,
    *,
    path: str | Path | None = None,
    provider_id: str = OPENCODE_PROVIDER_ID,
) -> bool:
    """Whether OpenCode's config lists ``model_id`` under MTPLX's provider
    and selects it as the model OpenCode opens on.

    Raises ``InvalidConfigFile`` for a file OpenCode could not read either.
    """

    config_path = opencode_config_path(path)
    if not config_path.exists():
        return False
    existing, _existing_text = load_config_file(config_path)
    return _mtplx_model_entry(existing, provider_id, model_id) is not None and (
        existing.get("model") == opencode_model_ref(model_id, provider_id=provider_id)
    )


def refresh_opencode_window(
    model_id: str,
    window: int,
    *,
    requested_output: int | None = None,
    path: str | Path | None = None,
    provider_id: str = OPENCODE_PROVIDER_ID,
) -> dict[str, Any]:
    """Give MTPLX's OpenCode model the window the server serves.

    That window (``served_execution_window``) exists only once the model is
    loaded, so the CLI's OpenCode handoff calls this after startup, as the
    app re-syncs OpenCode once its daemon answers. ``limit.context`` becomes
    the window and ``limit.output`` its reply reserve, or the answer cap the
    user asked for: ``requested_output`` when this launch was given one
    (recorded as the model's header), else the one the entry records
    (:func:`opencode_requested_output`). Nothing else in the file changes.
    """

    config_path = opencode_config_path(path)
    result: dict[str, Any] = {
        "config_path": str(config_path),
        "context_window": int(window),
        "output_limit": opencode_output_limit(window),
        "written": False,
        "backup_path": None,
    }
    if int(window) <= 0 or not config_path.exists():
        return result
    existing, _existing_text = load_config_file(config_path)
    model = _mtplx_model_entry(existing, provider_id, model_id)
    if model is None:
        return result
    typed = None
    if requested_output is not None and int(requested_output) > 0:
        typed = int(requested_output)
    output = opencode_output_limit(window, typed or opencode_requested_output(model))
    result["output_limit"] = output
    refreshed = {**model, "limit": {"context": int(window), "output": output}}
    if typed is not None:
        headers = model.get("headers") if isinstance(model.get("headers"), dict) else {}
        refreshed["headers"] = {**headers, OPENCODE_REQUESTED_OUTPUT_HEADER: str(typed)}
    provider = existing["provider"][provider_id]
    merged = {
        **existing,
        "provider": {
            **existing["provider"],
            provider_id: {
                **provider,
                "models": {**provider["models"], str(model_id): refreshed},
            },
        },
    }
    written, backup_path = _write_config_json(config_path, existing, merged)
    result["written"] = written
    result["backup_path"] = str(backup_path) if backup_path is not None else None
    return result


def write_opencode_session_headers_plugin(
    path: str | Path | None = None,
) -> Path:
    """Install versioned entrypoints without dropping older V1 support.

    V1 imports the package main (the original function API); V2's plugin
    host resolves the server subpath first. Both use the same registration
    path. See opencode.ai/v2/docs/build/plugins/migrate-v1 and @opencode/plugin
    Host.resolve. No npm dependency or runtime version sniffing is needed.
    """

    plugin_path = opencode_session_headers_plugin_path(path)
    plugin_path.mkdir(parents=True, exist_ok=True)
    files = {
        "package.json": json.dumps({
            "name": OPENCODE_SESSION_HEADERS_PACKAGE_NAME,
            "private": True,
            "type": "module",
            "main": "./index.js",
            "exports": {".": "./index.js", "./server": "./server.js"},
        }, indent=2) + "\n",
        "index.js": OPENCODE_SESSION_HEADERS_PLUGIN_SOURCE,
        "server.js": OPENCODE_SESSION_HEADERS_V2_SOURCE,
    }
    for name, source in files.items():
        target = plugin_path / name
        if not target.exists() or target.read_text(encoding="utf-8") != source:
            target.write_text(source, encoding="utf-8")
        try:
            target.chmod(0o600)
        except OSError:
            pass
    return plugin_path


def ensure_opencode_reasoning_summaries_visible(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Enable visible reasoning parts in OpenCode Desktop's UI."""

    explicit_path = path is not None or bool(
        os.environ.get("MTPLX_OPENCODE_DESKTOP_SETTINGS_STORE")
    )
    if sys.platform != "darwin" and not explicit_path:
        return {
            "supported": False,
            "status": "unsupported_platform",
            "setting": "settings.v3.general.showReasoningSummaries",
        }

    store_path = opencode_desktop_settings_store_path(path)
    backup_path: Path | None = None
    root: dict[str, Any] = {}
    existing_data: str | None = None

    if store_path.exists():
        # A store that cannot be read is OpenCode Desktop's to repair. It used
        # to be moved aside and replaced with a store holding only this one
        # setting, which threw away every other Desktop setting for a
        # cosmetic tweak; now the tweak is skipped and the store left alone.
        try:
            existing_data = store_path.read_text(encoding="utf-8")
            parsed = json.loads(existing_data)
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "supported": True,
                "status": "unreadable_store",
                "path": str(store_path),
                "did_change": False,
                "backup_path": None,
                "setting": "settings.v3.general.showReasoningSummaries",
                "error": str(exc),
            }
        root = parsed if isinstance(parsed, dict) else {}

    raw_settings = root.get(OPENCODE_DESKTOP_SETTINGS_KEY)
    settings: dict[str, Any] = {}
    if isinstance(raw_settings, str) and raw_settings.strip():
        try:
            parsed_settings = json.loads(raw_settings)
            settings = parsed_settings if isinstance(parsed_settings, dict) else {}
        except json.JSONDecodeError:
            settings = {}
    elif isinstance(raw_settings, dict):
        settings = dict(raw_settings)

    general = settings.get("general")
    if not isinstance(general, dict):
        general = {}
    else:
        general = dict(general)

    if general.get("showReasoningSummaries") is True:
        return {
            "supported": True,
            "status": "already_visible",
            "path": str(store_path),
            "did_change": False,
            "backup_path": None,
            "setting": "settings.v3.general.showReasoningSummaries",
        }

    general["showReasoningSummaries"] = True
    settings["general"] = general
    root[OPENCODE_DESKTOP_SETTINGS_KEY] = json.dumps(settings, separators=(",", ":"))

    next_data = json.dumps(root, indent=2, sort_keys=True) + "\n"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    if existing_data is not None and backup_path is None:
        backup_path = _unique_backup(store_path, "reasoning-visible")
        shutil.copy2(store_path, backup_path)
    store_path.write_text(next_data, encoding="utf-8")
    try:
        store_path.chmod(0o600)
    except OSError:
        pass

    return {
        "supported": True,
        "status": "enabled",
        "path": str(store_path),
        "did_change": True,
        "backup_path": str(backup_path) if backup_path is not None else None,
        "setting": "settings.v3.general.showReasoningSummaries",
    }


def write_opencode_config(
    *,
    base_url: str,
    model_id: str,
    model_name: str | None = None,
    api_key: str | None = None,
    path: str | Path | None = None,
    provider_id: str = OPENCODE_PROVIDER_ID,
    context_window: int = OPENCODE_DEFAULT_CONTEXT_WINDOW,
    output_limit: int | None = None,
    chunk_timeout_ms: int = OPENCODE_DEFAULT_CHUNK_TIMEOUT_MS,
    enable_thinking: bool = True,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    reasoning_effort: str | None = None,
    reasoning_effort_levels: Sequence[str] | None = None,
    vision: bool = False,
    keep_window: bool = False,
) -> dict[str, Any]:
    """Write MTPLX into OpenCode config and return a handoff payload.

    ``keep_window`` keeps the window (``limit.context``) this model already
    has: the write before the server starts has only a guess at the window,
    and the handoff then brings it to the served one
    (``refresh_opencode_window``), so a launch never rewrites it twice.
    ``limit.output`` follows the kept window and ``output_limit``, the cap
    requested with ``--max-response-tokens``.
    """

    config_path = opencode_config_path(path)
    existing: dict[str, Any] | None = None
    if config_path.exists():
        # OpenCode reads this file as JSONC (comments, trailing commas), so
        # MTPLX does too. A file that still does not parse is the user's to
        # fix: InvalidConfigFile propagates and nothing here is moved or
        # written, instead of the old move-aside that replaced their
        # providers, agents and keybinds with an MTPLX-only config.
        existing, _existing_text = load_config_file(config_path)

    fragment = build_opencode_provider_config(
        base_url=base_url,
        model_id=model_id,
        model_name=model_name,
        api_key=api_key,
        context_window=context_window,
        output_limit=output_limit,
        chunk_timeout_ms=chunk_timeout_ms,
        enable_thinking=enable_thinking,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        reasoning_effort=reasoning_effort,
        reasoning_effort_levels=reasoning_effort_levels,
        vision=vision,
    )
    limit = fragment["provider"][OPENCODE_PROVIDER_ID]["models"][str(model_id)]["limit"]
    kept = (_mtplx_model_entry(existing, provider_id, model_id) or {}).get("limit")
    kept_window = kept.get("context") if isinstance(kept, dict) else None
    if keep_window and isinstance(kept_window, int) and kept_window > 0:
        limit.update(
            context=kept_window,
            output=opencode_output_limit(kept_window, output_limit),
        )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    session_headers_plugin_path = write_opencode_session_headers_plugin(config_path)
    merged = merge_opencode_config(
        existing,
        config_fragment=fragment,
        provider_id=provider_id,
        session_headers_plugin_path=session_headers_plugin_path,
    )
    written, backup_path = _write_config_json(config_path, existing, merged)
    reasoning_visibility = ensure_opencode_reasoning_summaries_visible()
    return {
        "config_path": str(config_path),
        "backup_path": str(backup_path) if backup_path is not None else None,
        "provider_id": provider_id,
        "base_url": str(base_url).rstrip("/"),
        "model_id": model_id,
        "model_ref": opencode_model_ref(model_id, provider_id=provider_id),
        "context_window": int(limit["context"]),
        "output_limit": int(limit["output"]),
        "chunk_timeout_ms": int(chunk_timeout_ms),
        "reasoning_field": "reasoning_content",
        "reasoning_effort": reasoning_effort,
        "session_headers_plugin_path": str(session_headers_plugin_path),
        "reasoning_visibility": reasoning_visibility,
        "no_hidden_max_tokens": True,
        "written": written,
    }
