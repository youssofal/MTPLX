from __future__ import annotations

import base64
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from mtplx.opencode import (
    OPENCODE_INJECTED_OUTPUT_CAP,
    OPENCODE_INJECTED_QWEN_TEMPERATURE,
    OPENCODE_INJECTED_QWEN_TOP_P,
    OPENCODE_REQUESTED_OUTPUT_HEADER,
    OPENCODE_SESSION_HEADERS_PLUGIN_SOURCE,
    build_opencode_provider_config,
    ensure_opencode_reasoning_summaries_visible,
    merge_opencode_config,
    opencode_model_ref,
    opencode_session_headers_plugin_path,
    refresh_opencode_window,
    repair_opencode_desktop_state,
    write_opencode_config,
    write_opencode_session_headers_plugin,
)


def test_opencode_injected_output_cap_matches_client_wire_truth():
    # sst/opencode provider/transform.ts OUTPUT_TOKEN_MAX = 32_000 (v1.18.21),
    # live receipt request-log-8002.jsonl records 313-327: request_max_tokens
    # = 32000. The 32_768 guess never matched, so the strip was a no-op and
    # five marathon generations truncated mid-think.
    assert OPENCODE_INJECTED_OUTPUT_CAP == 32_000


def test_opencode_model_ref_uses_provider_namespace():
    assert (
        opencode_model_ref("mtplx-qwen36-27b-optimized-quality")
        == "mtplx/mtplx-qwen36-27b-optimized-quality"
    )


def test_build_opencode_config_keeps_sampler_policy_server_side():
    payload = build_opencode_provider_config(
        base_url="http://127.0.0.1:18083/v1",
        model_id="mtplx-qwen36-27b-optimized-quality",
        api_key="1234",
        context_window=262144,
    )

    provider = payload["provider"]["mtplx"]
    model = provider["models"]["mtplx-qwen36-27b-optimized-quality"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == "http://127.0.0.1:18083/v1"
    assert provider["options"]["timeout"] is False
    assert provider["options"]["chunkTimeout"] == 900000
    assert provider["options"]["headers"]["x-mtplx-client"] == "opencode"
    assert provider["options"]["apiKey"] == "1234"
    # Reasoning + temperature are declared capable so reasoning_content
    # round-trips and explicit client choices transmit; the family sampler
    # itself stays server-side (no per-model sampler transport in
    # @ai-sdk/openai-compatible 2.0.41).
    assert model["reasoning"] is True
    assert model["tool_call"] is True
    assert model["temperature"] is True
    assert model["limit"] == {"context": 262144, "output": 32000}
    assert "interleaved" not in model
    assert "options" not in model
    assert "variants" not in model
    assert "maxTokens" not in json.dumps(payload)


def test_opencode_writer_upgrades_text_only_model_registration(tmp_path):
    path = tmp_path / "opencode.json"
    kwargs = dict(base_url="http://127.0.0.1:8000/v1", model_id="vision-model", path=path)
    write_opencode_config(**kwargs)
    assert json.loads(path.read_text())["provider"]["mtplx"]["models"]["vision-model"]["modalities"]["input"] == ["text"]
    write_opencode_config(**kwargs, vision=True)
    assert json.loads(path.read_text())["provider"]["mtplx"]["models"]["vision-model"]["modalities"]["input"] == ["text", "image"]


def test_build_opencode_config_carries_family_effort_dial():
    payload = build_opencode_provider_config(
        base_url="http://127.0.0.1:18083/v1",
        model_id="mtplx-qwen38-27b-optimized-speed",
        context_window=262144,
        reasoning_effort="medium",
        reasoning_effort_levels=("xhigh", "medium", "low"),
    )

    model = payload["provider"]["mtplx"]["models"]["mtplx-qwen38-27b-optimized-speed"]
    assert model["reasoning"] is True
    # The app dial rides options.reasoningEffort (the SDK's reasoning_effort
    # transport); an effort variant picked inside OpenCode merges after model
    # options and wins for that request.
    assert model["options"] == {"reasoningEffort": "medium"}
    # OpenCode's effort picker is trimmed to the family dial: tiers outside
    # xhigh/medium/low are disabled, and every family tier is declared as an
    # explicit variant — Desktop 1.18.21 does not surface its built-in list
    # for custom openai-compatible providers (xhigh was missing live), and an
    # explicit variant renders on every version.
    assert model["variants"] == {
        "none": {"disabled": True},
        "minimal": {"disabled": True},
        "high": {"disabled": True},
        "xhigh": {"reasoningEffort": "xhigh"},
        "medium": {"reasoningEffort": "medium"},
        "low": {"reasoningEffort": "low"},
    }


def test_build_opencode_config_no_effort_dial_disables_effort_picker():
    payload = build_opencode_provider_config(
        base_url="http://127.0.0.1:18083/v1",
        model_id="mtplx-qwen36-27b-optimized-quality",
        context_window=262144,
        reasoning_effort=None,
        reasoning_effort_levels=(),
    )

    model = payload["provider"]["mtplx"]["models"]["mtplx-qwen36-27b-optimized-quality"]
    assert model["reasoning"] is True
    assert "options" not in model
    assert model["variants"] == {
        "none": {"disabled": True},
        "minimal": {"disabled": True},
        "low": {"disabled": True},
        "medium": {"disabled": True},
        "high": {"disabled": True},
        "xhigh": {"disabled": True},
    }


def test_build_opencode_config_keeps_gemma_policy_server_side():
    payload = build_opencode_provider_config(
        base_url="http://127.0.0.1:18108/v1",
        model_id="gemma4-mtplx-optimized-speed",
        context_window=262144,
        enable_thinking=False,
        reasoning_effort="medium",
        reasoning_effort_levels=("low", "medium"),
    )

    provider = payload["provider"]["mtplx"]
    assert "apiKey" not in provider["options"]
    model = payload["provider"]["mtplx"]["models"]["gemma4-mtplx-optimized-speed"]
    assert model["reasoning"] is False
    assert model["temperature"] is True
    assert "interleaved" not in model
    # Non-reasoning families carry no effort dial even when a caller passes
    # one: effort is a reasoning control.
    assert "options" not in model
    assert "variants" not in model


def test_ensure_opencode_reasoning_summaries_visible_enables_desktop_store(tmp_path):
    store = tmp_path / "default.dat"
    store.write_text(
        json.dumps(
            {
                "settings.v3": json.dumps(
                    {
                        "general": {
                            "autoSave": True,
                            "showReasoningSummaries": False,
                        },
                        "appearance": {"fontSize": 14},
                    }
                ),
                "highlights.v1": json.dumps({"version": "1.15.7"}),
            }
        ),
        encoding="utf-8",
    )

    result = ensure_opencode_reasoning_summaries_visible(store)

    assert result["status"] == "enabled"
    assert result["did_change"] is True
    assert result["backup_path"]
    root = json.loads(store.read_text(encoding="utf-8"))
    settings = json.loads(root["settings.v3"])
    assert settings["general"]["showReasoningSummaries"] is True
    assert settings["general"]["autoSave"] is True
    assert root["highlights.v1"] == json.dumps({"version": "1.15.7"})

    second = ensure_opencode_reasoning_summaries_visible(store)
    assert second["status"] == "already_visible"
    assert second["did_change"] is False


def test_merge_opencode_config_preserves_other_providers():
    fragment = build_opencode_provider_config(
        base_url="http://127.0.0.1:18083/v1",
        model_id="mtplx-qwen36-27b-optimized-quality",
    )

    merged = merge_opencode_config(
        {
            "provider": {"lmstudio": {"name": "LM Studio"}},
            "model": "lmstudio/foo",
        },
        config_fragment=fragment,
    )

    assert merged["provider"]["lmstudio"] == {"name": "LM Studio"}
    assert merged["provider"]["mtplx"]["models"]
    assert merged["model"] == "mtplx/mtplx-qwen36-27b-optimized-quality"
    assert merged["small_model"] == "mtplx/mtplx-qwen36-27b-optimized-quality"


def test_merge_opencode_config_preserves_existing_plugins_and_injects_session_headers():
    fragment = build_opencode_provider_config(
        base_url="http://127.0.0.1:18083/v1",
        model_id="mtplx-qwen36-27b-optimized-quality",
    )

    merged = merge_opencode_config(
        {
            "plugin": ["/existing/plugin.js"],
            "provider": {"lmstudio": {"name": "LM Studio"}},
        },
        config_fragment=fragment,
        session_headers_plugin_path="/tmp/mtplx-session-headers.js",
    )

    assert merged["plugin"] == ["/existing/plugin.js", "/tmp/mtplx-session-headers.js"]


def test_merge_opencode_config_canonicalizes_stale_session_headers_entries():
    fragment = build_opencode_provider_config(
        base_url="http://127.0.0.1:18083/v1",
        model_id="mtplx-qwen36-27b-optimized-quality",
    )

    merged = merge_opencode_config(
        {
            "plugin": [
                "/existing/plugin.js",
                "/stale/location/mtplx-session-headers.js",
            ],
        },
        config_fragment=fragment,
        session_headers_plugin_path="/managed/mtplx-session-headers.js",
    )

    # A stale registration under another path would double-fire the hooks;
    # exactly one entry survives, at the managed location.
    assert merged["plugin"] == [
        "/existing/plugin.js",
        "/managed/mtplx-session-headers.js",
    ]


def test_write_opencode_config_refuses_unreadable_json_and_leaves_it_alone(tmp_path, monkeypatch):
    """C-12: an opencode.json that does not parse used to be moved to a
    .invalid-<stamp>.bak and replaced with an MTPLX-only config, so the
    user's other providers, agents and keybinds vanished from the live file.
    Now nothing is moved or written and the caller gets the path and error."""
    from mtplx.jsonc import InvalidConfigFile

    path = tmp_path / "opencode.json"
    settings_store = tmp_path / "default.dat"
    path.write_text('{"provider": {"lmstudio": {}}, bad json', encoding="utf-8")
    monkeypatch.setenv("MTPLX_OPENCODE_CONFIG", str(path))
    monkeypatch.setenv("MTPLX_OPENCODE_DESKTOP_SETTINGS_STORE", str(settings_store))

    with pytest.raises(InvalidConfigFile) as excinfo:
        write_opencode_config(
            base_url="http://127.0.0.1:18083/v1",
            model_id="mtplx-qwen36-27b-optimized-quality",
            api_key="1234",
        )

    assert str(path) in str(excinfo.value)
    assert "line 1, column 32" in str(excinfo.value)
    assert path.read_text(encoding="utf-8") == '{"provider": {"lmstudio": {}}, bad json'
    assert sorted(p.name for p in tmp_path.iterdir()) == ["opencode.json"], "nothing else written"
    assert not settings_store.exists()


def test_write_opencode_config_merges_a_jsonc_config_instead_of_replacing_it(
    tmp_path, monkeypatch
):
    """OpenCode reads opencode.json as JSONC (jsonc-parser, allowTrailingComma).
    A config with comments and trailing commas is a working config; MTPLX must
    merge into it, and keep the user's original text next to the rewrite."""
    path = tmp_path / "opencode.json"
    settings_store = tmp_path / "default.dat"
    original = (
        "{\n"
        "  // my providers\n"
        '  "provider": {"lmstudio": {"name": "LM Studio",},},\n'
        '  "keybinds": {"leader": "ctrl+x",}, /* keep */\n'
        "}\n"
    )
    path.write_text(original, encoding="utf-8")
    monkeypatch.setenv("MTPLX_OPENCODE_CONFIG", str(path))
    monkeypatch.setenv("MTPLX_OPENCODE_DESKTOP_SETTINGS_STORE", str(settings_store))

    result = write_opencode_config(
        base_url="http://127.0.0.1:18083/v1",
        model_id="mtplx-qwen36-27b-optimized-quality",
        api_key="1234",
    )

    assert result["written"] is True
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["provider"]["lmstudio"] == {"name": "LM Studio"}
    assert payload["keybinds"] == {"leader": "ctrl+x"}
    assert payload["provider"]["mtplx"]["options"]["baseURL"] == "http://127.0.0.1:18083/v1"
    assert not list(tmp_path.glob("*.invalid-*")), "a readable config is never treated as invalid"
    backup = Path(result["backup_path"])
    assert backup.parent == tmp_path
    assert "before-mtplx" in backup.name
    assert backup.read_text(encoding="utf-8") == original


def test_write_opencode_config_leaves_an_up_to_date_file_untouched(tmp_path, monkeypatch):
    path = tmp_path / "opencode.json"
    settings_store = tmp_path / "default.dat"
    monkeypatch.setenv("MTPLX_OPENCODE_CONFIG", str(path))
    monkeypatch.setenv("MTPLX_OPENCODE_DESKTOP_SETTINGS_STORE", str(settings_store))
    kwargs = dict(
        base_url="http://127.0.0.1:18083/v1",
        model_id="mtplx-qwen36-27b-optimized-quality",
        api_key="1234",
    )

    first = write_opencode_config(**kwargs)
    assert first["written"] is True
    assert first["backup_path"] is None  # nothing existed to keep
    # The user adds a comment; the content MTPLX cares about is unchanged.
    commented = "// mine\n" + path.read_text(encoding="utf-8")
    path.write_text(commented, encoding="utf-8")

    second = write_opencode_config(**kwargs)

    assert second["written"] is False
    assert second["backup_path"] is None
    assert path.read_text(encoding="utf-8") == commented
    assert not list(tmp_path.glob("*.bak"))


def test_ensure_opencode_reasoning_summaries_visible_leaves_unreadable_store_alone(tmp_path):
    store = tmp_path / "default.dat"
    store.write_text("{not json", encoding="utf-8")

    result = ensure_opencode_reasoning_summaries_visible(store)

    assert result["status"] == "unreadable_store"
    assert result["did_change"] is False
    assert result["backup_path"] is None
    assert store.read_text(encoding="utf-8") == "{not json"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["default.dat"]


def test_write_opencode_config_installs_session_headers_plugin(tmp_path, monkeypatch):
    path = tmp_path / "opencode.json"
    settings_store = tmp_path / "default.dat"
    path.write_text(
        json.dumps({"plugin": [str(path.parent / "mtplx-session-headers.js")]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MTPLX_OPENCODE_CONFIG", str(path))
    monkeypatch.setenv("MTPLX_OPENCODE_DESKTOP_SETTINGS_STORE", str(settings_store))

    result = write_opencode_config(
        base_url="http://127.0.0.1:18083/v1",
        model_id="mtplx-qwen36-27b-optimized-quality",
        api_key="1234",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["provider"]["mtplx"]["options"]["apiKey"] == "1234"
    assert payload["provider"]["mtplx"]["options"]["headers"]["x-mtplx-client"] == "opencode"
    plugin_path = str(opencode_session_headers_plugin_path(path))
    assert plugin_path in payload["plugin"]
    package = Path(plugin_path)
    assert package.parent.name == "plugins"
    assert json.loads((package / "package.json").read_text())["exports"] == {
        ".": "./index.js", "./server": "./server.js"
    }
    assert (package / "server.js").exists()
    assert str(path.parent / "mtplx-session-headers.js") not in payload["plugin"]
    assert result["reasoning_visibility"]["path"] == str(settings_store)
    assert result["reasoning_visibility"]["did_change"] is True
    assert payload["provider"]["mtplx"]["models"]
    assert result["session_headers_plugin_path"] == plugin_path
    assert not (path.parent / "package.json").exists()
    plugin_source = (package / "index.js").read_text(
        encoding="utf-8"
    )
    assert 'output.headers["x-mtplx-session-id"]' in plugin_source
    assert '"chat.params"' in plugin_source
    # The rewrite is guarded: only OpenCode's own default for the model,
    # min(limit.output, 32,000), is replaced (by the cap the user asked MTPLX
    # for, or by nothing); an explicit client cap passes through (issue:
    # unconditional delete erased deliberate user caps). The node-executed
    # tests below pin the behavior.
    assert "if (output.maxOutputTokens === opencodeDefault)" in plugin_source
    assert (
        f'const mtplxRequestedOutputHeader = "{OPENCODE_REQUESTED_OUTPUT_HEADER}";'
        in plugin_source
    )
    assert f"const mtplxInjectedOutputCap = {OPENCODE_INJECTED_OUTPUT_CAP};" in plugin_source
    # Same guarded-strip contract for the qwen sampler OpenCode <= 1.18.20
    # injects (temperature 0.55 / topP 1): exactly the injected pair is
    # cleared so the server's family-native sampler applies.
    assert (
        f"const mtplxInjectedQwenTemperature = {OPENCODE_INJECTED_QWEN_TEMPERATURE};"
        in plugin_source
    )
    assert (
        f"const mtplxInjectedQwenTopP = {OPENCODE_INJECTED_QWEN_TOP_P};"
        in plugin_source
    )
    assert "output.temperature === mtplxInjectedQwenTemperature" in plugin_source
    assert "output.topP === mtplxInjectedQwenTopP" in plugin_source
    assert "process.stdout.write" not in plugin_source
    assert "message.updated" not in plugin_source


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_plugin_package_registers_v1_and_provider_scoped_v2_headers(tmp_path):
    package = write_opencode_session_headers_plugin(tmp_path / "opencode.json")
    script = f"""
import plugin from {json.dumps((package / 'server.js').as_uri())};
import legacy from {json.dumps((package / 'index.js').as_uri())};
const registrations = [];
await plugin.setup({{session: {{hook: async (...args) => registrations.push(args)}}}});
const [name, callback, filter] = registrations[0];
const event = {{sessionID: 'ses_v2', headers: {{existing: 'kept'}},
  model: {{providerID: 'mtplx', id: 'flash'}}, kind: 'primary'}};
await callback(event);
const hooks = await plugin.server();
const output = {{headers: {{existing: 'kept'}}}};
await hooks['chat.headers']({{sessionID: 'ses_v1', model: {{providerID: 'mtplx'}},
  message: {{id: 'msg_user'}}}}, output);
console.log(JSON.stringify({{id: plugin.id, name, filter, count: registrations.length,
  sameLegacy: plugin.server === legacy, event, output}}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    assert data["sameLegacy"] is True
    assert data["count"] == 1
    assert data["name"] == "model.request"
    assert data["filter"] == {"providerID": "mtplx"}
    # OpenCode 2.x's own request kind (primary, title, compaction) reaches
    # the server, which keeps a title off the conversation's session.
    assert data["event"]["headers"] == {
        "existing": "kept", "x-mtplx-client": "opencode",
        "x-mtplx-session-id": "ses_v2", "x-mtplx-request-kind": "primary",
    }
    assert data["output"]["headers"]["x-mtplx-session-id"] == "ses_v1"
    assert data["output"]["headers"]["x-mtplx-client-turn-id"] == "msg_user"
    assert set(data["event"]) == {"sessionID", "headers", "model", "kind"}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_opencode_plugin_cap_guard_three_payload_shapes(tmp_path):
    """Execute the real plugin under node for the three payload shapes:
    injected default (stripped), explicit cap (preserved), absent (untouched).
    """

    plugin = tmp_path / "plugin.mjs"
    plugin.write_text(OPENCODE_SESSION_HEADERS_PLUGIN_SOURCE, encoding="utf-8")
    harness = tmp_path / "harness.mjs"
    harness.write_text(
        f"""
import plugin from {json.dumps(str(plugin))};
const hooks = await plugin();
const run = async (params) => {{
  const output = {{ ...params }};
  await hooks["chat.params"]({{ model: {{ providerID: "mtplx" }} }}, output);
  return output;
}};
const results = {{
  injected: await run({{ maxOutputTokens: {OPENCODE_INJECTED_OUTPUT_CAP} }}),
  explicit: await run({{ maxOutputTokens: 9000 }}),
  absent: await run({{}}),
}};
console.log(JSON.stringify(results));
""",
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["node", str(harness)], capture_output=True, text=True, check=True
    )
    results = json.loads(proc.stdout)
    # JSON.stringify drops undefined-valued keys.
    assert "maxOutputTokens" not in results["injected"]
    assert results["explicit"]["maxOutputTokens"] == 9000
    assert "maxOutputTokens" not in results["absent"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_opencode_plugin_strips_only_client_injected_qwen_sampler(tmp_path):
    """Payload shapes mirror oc-recorder chat.params records: OpenCode
    <= 1.18.20 injects temperature 0.55 / topP 1 for qwen model ids. Exactly
    that pair is stripped for MTPLX qwen models; explicit values and
    non-qwen models pass through untouched.
    """

    plugin = tmp_path / "plugin.mjs"
    plugin.write_text(OPENCODE_SESSION_HEADERS_PLUGIN_SOURCE, encoding="utf-8")
    harness = tmp_path / "harness.mjs"
    harness.write_text(
        f"""
import plugin from {json.dumps(str(plugin))};
const hooks = await plugin();
const run = async (modelID, params) => {{
  const output = {{ ...params }};
  await hooks["chat.params"](
    {{ model: {{ providerID: "mtplx", id: modelID }} }},
    output,
  );
  return output;
}};
const qwen = "mtplx-qwen38-27b-optimized-speed";
const results = {{
  injected: await run(qwen, {{
    temperature: {OPENCODE_INJECTED_QWEN_TEMPERATURE},
    topP: {OPENCODE_INJECTED_QWEN_TOP_P},
    maxOutputTokens: {OPENCODE_INJECTED_OUTPUT_CAP},
  }}),
  explicit: await run(qwen, {{ temperature: 0.9, topP: 0.8 }}),
  nonQwen: await run("step-3.7-flash-mtplx-step3p5", {{
    temperature: {OPENCODE_INJECTED_QWEN_TEMPERATURE},
    topP: {OPENCODE_INJECTED_QWEN_TOP_P},
  }}),
}};
const foreign = {{ temperature: 0.55, topP: 1 }};
await hooks["chat.params"]({{ model: {{ providerID: "anthropic", id: "qwen-x" }} }}, foreign);
results.foreign = foreign;
console.log(JSON.stringify(results));
""",
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["node", str(harness)], capture_output=True, text=True, check=True
    )
    results = json.loads(proc.stdout)
    # JSON.stringify drops undefined-valued keys: the injected pair is gone.
    assert "temperature" not in results["injected"]
    assert "topP" not in results["injected"]
    assert "maxOutputTokens" not in results["injected"]
    assert results["explicit"]["temperature"] == 0.9
    assert results["explicit"]["topP"] == 0.8
    # Non-qwen MTPLX models never had the client-injected qwen sampler; any
    # matching values there are deliberate and pass through.
    assert results["nonQwen"]["temperature"] == OPENCODE_INJECTED_QWEN_TEMPERATURE
    assert results["nonQwen"]["topP"] == OPENCODE_INJECTED_QWEN_TOP_P
    # Other providers are untouched entirely.
    assert results["foreign"]["temperature"] == 0.55
    assert results["foreign"]["topP"] == 1


def _chat_params_after_hook(tmp_path: Path, plugin: Path, cases: dict) -> dict:
    """Run the plugin's chat.params hook under node on each (input, output)."""

    harness = tmp_path / "chat-params-harness.mjs"
    harness.write_text(
        f"""
import plugin from {json.dumps(str(plugin))};
const hooks = await plugin();
const cases = {json.dumps(cases)};
const results = {{}};
for (const [name, [input, output]] of Object.entries(cases)) {{
  await hooks["chat.params"](input, output);
  results[name] = output;
}}
console.log(JSON.stringify(results));
""",
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["node", str(harness)], capture_output=True, text=True, check=True
    )
    return json.loads(proc.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("window", [20_480, 32_768, 65_536, 262_144])
def test_opencode_plugin_strips_the_output_limit_mtplx_configured(tmp_path, window):
    """OpenCode sends maxOutputTokens = min(limit.output, 32000) with every
    request and hands the hook the model with its limit (provider/transform.ts
    maxOutputTokens; CLI 1.18.29 and Desktop 1.18.31). Below a 64K window the
    limit.output MTPLX configures is half the window (#480), and a hook that
    stripped only 32,000 left every answer capped there: 16,384 on 32,768.
    The generated plugin runs on the request OpenCode builds from the config
    MTPLX wrote: the injected value goes at every window, a user cap stays,
    and so does a limit.output the user chose (--max-response-tokens)."""

    def written_model(config_path, **kwargs):
        write_opencode_config(
            base_url="http://127.0.0.1:8000/v1",
            model_id="mtplx-qwen",
            path=config_path,
            context_window=window,
            **kwargs,
        )
        entry = json.loads(config_path.read_text())["provider"]["mtplx"]["models"][
            "mtplx-qwen"
        ]
        # What OpenCode hands the hook: the model with its limit and headers.
        return {
            "providerID": "mtplx",
            "id": "mtplx-qwen",
            "limit": entry["limit"],
            "headers": entry.get("headers", {}),
        }

    config_path = tmp_path / "opencode.json"
    model = written_model(config_path)
    assert model["limit"] == {"context": window, "output": min(32_000, window // 2)}
    chosen = written_model(tmp_path / "chosen" / "opencode.json", output_limit=9_000)
    assert chosen["limit"] == {"context": window, "output": 9_000}

    def sent(model):
        # What OpenCode puts in the request before the hook runs.
        return {"maxOutputTokens": min(model["limit"]["output"], OPENCODE_INJECTED_OUTPUT_CAP)}

    results = _chat_params_after_hook(
        tmp_path,
        opencode_session_headers_plugin_path(config_path) / "index.js",
        {
            "injected": [{"model": model}, sent(model)],
            "user_cap": [{"model": model}, {"maxOutputTokens": 9_000}],
            "absent": [{"model": model}, {}],
            "chosen_limit": [{"model": chosen}, sent(chosen)],
        },
    )
    # JSON.stringify drops undefined-valued keys.
    assert "maxOutputTokens" not in results["injected"]
    assert results["user_cap"]["maxOutputTokens"] == 9_000
    assert "maxOutputTokens" not in results["absent"]
    assert results["chosen_limit"]["maxOutputTokens"] == 9_000


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize(
    "window,cap",
    [
        (16_384, 9_000),
        (32_768, 20_000),
        (262_144, 50_000),
        (32_768, 16_384),
        (262_144, 32_000),
    ],
)
def test_an_explicit_cap_reaches_a_running_server_through_opencode(
    tmp_path, monkeypatch, window, cap
):
    """`mtplx start opencode --max-response-tokens N` against a server
    already running, started without the flag: only OpenCode's request can
    carry N. It used to be clamped into the reply reserve (8,192 for 9,000 on
    a 16,384 window), and the plugin then removed it as OpenCode's default.
    Now it is written as limit.output and as the model's header, and the
    generated plugin sends N whole, a cap equal to the reserve and one above
    OpenCode's 32,000 ceiling included. Without the flag nothing changes:
    the reserve is the default and is stripped."""
    from types import SimpleNamespace

    import mtplx.opencode as opencode_module
    from mtplx.commands import public

    config_path = tmp_path / "opencode.json"
    monkeypatch.setenv("MTPLX_OPENCODE_CONFIG", str(config_path))
    monkeypatch.setattr(public, "_model_vision_enabled", lambda model: False)
    monkeypatch.setattr(opencode_module, "detect_opencode_desktop", dict)

    def handoff(**flags):
        args = SimpleNamespace(
            model="Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed",
            context_window=window,
            max_response_tokens=flags.get("value"),
            _cli_flags={"max-response-tokens"} if flags.get("typed") else set(),
        )
        model_id = public._quickstart_opencode_payload(args, write_config=True)["model_id"]
        health = {"ok": True, "model": model_id, "execution_window": {"tokens": window}}
        public._quickstart_sync_client_window("opencode", health, model_id=model_id)
        entry = json.loads(config_path.read_text())["provider"]["mtplx"]["models"][model_id]
        model = {
            "providerID": "mtplx",
            "id": model_id,
            "limit": entry["limit"],
            "headers": entry.get("headers", {}),
        }
        sent = {"maxOutputTokens": min(entry["limit"]["output"], OPENCODE_INJECTED_OUTPUT_CAP)}
        return entry, model, sent

    reserve = min(32_000, window // 2)
    typed_entry, typed_model, typed_sent = handoff(value=cap, typed=True)
    assert typed_entry["limit"] == {"context": window, "output": cap}
    default_entry, default_model, default_sent = handoff()
    assert default_entry["limit"] == {"context": window, "output": reserve}
    # A pack's own default cap (set on args without the flag) is the server's.
    pack_entry, pack_model, pack_sent = handoff(value=32_768)
    assert pack_entry["limit"] == {"context": window, "output": reserve}
    assert "headers" not in default_entry and "headers" not in pack_entry

    results = _chat_params_after_hook(
        tmp_path,
        opencode_session_headers_plugin_path(config_path) / "index.js",
        {
            "typed": [{"model": typed_model}, typed_sent],
            "default": [{"model": default_model}, default_sent],
            "pack_default": [{"model": pack_model}, pack_sent],
        },
    )
    assert results["typed"]["maxOutputTokens"] == cap
    assert "maxOutputTokens" not in results["default"]
    assert "maxOutputTokens" not in results["pack_default"]


def test_refresh_opencode_window_sets_the_served_limits_and_nothing_else(tmp_path):
    config_path = tmp_path / "opencode.json"
    write_opencode_config(
        base_url="http://127.0.0.1:8000/v1",
        model_id="mtplx-qwen",
        path=config_path,
        context_window=262_144,
    )
    before = json.loads(config_path.read_text())
    result = refresh_opencode_window("mtplx-qwen", 32_768, path=config_path)
    after = json.loads(config_path.read_text())
    assert result["written"] is True and result["backup_path"]
    model = after["provider"]["mtplx"]["models"]["mtplx-qwen"]
    assert model.pop("limit") == {"context": 32_768, "output": 16_384}
    before["provider"]["mtplx"]["models"]["mtplx-qwen"].pop("limit")
    assert after == before
    # An answer cap the user asked for is recorded on the entry and survives
    # a new window; without one, limit.output is the reserve for that window.
    write_opencode_config(
        base_url="http://127.0.0.1:8000/v1",
        model_id="mtplx-qwen",
        path=config_path,
        context_window=262_144,
        output_limit=20_000,
    )
    capped = refresh_opencode_window("mtplx-qwen", 16_384, path=config_path)
    model = json.loads(config_path.read_text())["provider"]["mtplx"]["models"]["mtplx-qwen"]
    assert capped["output_limit"] == 20_000
    assert model["limit"] == {"context": 16_384, "output": 20_000}
    assert model["headers"] == {OPENCODE_REQUESTED_OUTPUT_HEADER: "20000"}
    # A cap typed for this launch (a running server started without it).
    refresh_opencode_window("mtplx-qwen", 32_768, requested_output=8_000, path=config_path)
    model = json.loads(config_path.read_text())["provider"]["mtplx"]["models"]["mtplx-qwen"]
    assert model["limit"] == {"context": 32_768, "output": 8_000}
    assert model["headers"] == {OPENCODE_REQUESTED_OUTPUT_HEADER: "8000"}
    assert refresh_opencode_window("mtplx-other", 32_768, path=config_path)["written"] is False
    assert refresh_opencode_window("mtplx-qwen", 32_768, path=tmp_path / "none.json")["written"] is False


def test_write_opencode_config_keep_window_keeps_the_served_limits(tmp_path):
    """The write before the server starts has only a guess at the window;
    with keep_window it keeps the window the last handoff set, so a launch
    does not rewrite it twice, and a cap asked for with
    --max-response-tokens still sets limit.output on that window."""
    config_path = tmp_path / "opencode.json"
    kwargs = {
        "base_url": "http://127.0.0.1:8000/v1",
        "model_id": "mtplx-qwen",
        "path": config_path,
        "context_window": 262_144,
    }
    write_opencode_config(**kwargs)
    refresh_opencode_window("mtplx-qwen", 32_768, path=config_path)
    kept = write_opencode_config(**kwargs, keep_window=True)
    assert kept["written"] is False
    assert (kept["context_window"], kept["output_limit"]) == (32_768, 16_384)
    capped = write_opencode_config(**kwargs, output_limit=8_000, keep_window=True)
    assert (capped["context_window"], capped["output_limit"]) == (32_768, 8_000)
    fresh = write_opencode_config(**kwargs)
    assert (fresh["context_window"], fresh["output_limit"]) == (262_144, 32_000)


def test_repair_opencode_desktop_state_prunes_missing_workspace(tmp_path, monkeypatch):
    app_support = tmp_path / "OpenCodeSupport"
    app_support.mkdir()
    present = tmp_path / "present-project"
    present.mkdir()
    missing_path = "/private/tmp/mtplx-opencode-desktop-qa"
    missing_key = base64.urlsafe_b64encode(missing_path.encode()).decode().rstrip("=")
    present_key = base64.urlsafe_b64encode(str(present).encode()).decode().rstrip("=")
    store = app_support / "opencode.global.dat"
    store.write_text(
        json.dumps(
            {
                "layout": json.dumps(
                    {
                        "sessionTabs": {
                            f"{missing_key}/ses_dead": {"all": []},
                            f"{present_key}/ses_live": {"all": ["context"]},
                        },
                        "sessionView": {
                            f"{missing_key}/ses_dead": {"scroll": {}},
                            f"{present_key}/ses_live": {"scroll": {}},
                        },
                    }
                ),
                "layout.page": json.dumps(
                    {
                        "lastProjectSession": {
                            missing_path: {
                                "directory": missing_path,
                                "id": "ses_dead",
                            },
                            str(present): {
                                "directory": str(present),
                                "id": "ses_live",
                            },
                        },
                        "workspaceExpanded": {
                            missing_path: True,
                            str(present): True,
                        },
                    }
                ),
                "server": json.dumps(
                    {
                        "projects": {
                            "local": [
                                {
                                    "worktree": missing_path,
                                    "expanded": True,
                                },
                                {"worktree": str(present), "expanded": True},
                            ]
                        },
                        "lastProject": {"local": str(present)},
                    }
                ),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MTPLX_OPENCODE_DESKTOP_APP_SUPPORT", str(app_support))

    result = repair_opencode_desktop_state()

    assert result["status"] == "repaired"
    assert result["did_change"] is True
    assert result["backup_path"]
    repaired = json.loads(store.read_text(encoding="utf-8"))
    assert "mtplx-opencode-desktop-qa" not in repaired["layout"]
    assert "mtplx-opencode-desktop-qa" not in repaired["layout.page"]
    assert "mtplx-opencode-desktop-qa" not in repaired["server"]
    assert str(present) in repaired["layout.page"]


def test_opencode_output_limit_keeps_half_the_window():
    """Issue #480: OpenCode reserves the output limit out of the context before
    deciding whether the conversation still fits; equal numbers left nothing.
    The default reserve is half the window, at most 32,000. An answer cap the
    user asked for is written as is: the compaction trade is theirs."""
    from mtplx.opencode import opencode_output_limit

    assert opencode_output_limit(262_144) == 32_000
    assert opencode_output_limit(57_344) == 28_672
    assert opencode_output_limit(8_192) == 4_096
    assert opencode_output_limit(8_192, 6_000) == 6_000
    assert opencode_output_limit(262_144, 8_000) == 8_000
    assert opencode_output_limit(262_144, 50_000) == 50_000
    assert opencode_output_limit(262_144, 0) == 32_000


def test_written_config_reserves_half_of_a_small_window(tmp_path):
    from mtplx.opencode import write_opencode_config

    config_path = tmp_path / "opencode.json"
    result = write_opencode_config(
        path=config_path,
        base_url="http://127.0.0.1:8000/v1",
        model_id="mtplx-qwen38-27b-optimized-speed-fp16",
        context_window=8_192,
    )
    assert result["output_limit"] == 4_096
    written = json.loads(config_path.read_text())
    model = written["provider"]["mtplx"]["models"]["mtplx-qwen38-27b-optimized-speed-fp16"]
    assert model["limit"] == {"context": 8192, "output": 4096}
