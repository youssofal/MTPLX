"""How the server was started, as the dashboard's Settings tab shows it.

Covers the startup capture (argv, the MTPLX_/MLX_/PYTHONPATH environment and
each variable's source), redaction of secrets, the ``mtplx serve`` handoff
and the read-only ``GET /v1/mtplx/launch`` endpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from mtplx.profiles import apply_profile_env, get_profile
from mtplx.server.launch_info import (
    REDACTED,
    SERVE_LAUNCH_ENV,
    build_launch_info,
    capture_launch,
    is_secret_name,
    redact_argv,
    serve_invocation_argv,
    serve_launch_handoff,
)

FIXED = {"profile": "turbo", "port": 8123}


def _rows(info):
    return {row["name"]: row for row in info["env"]}


def test_capture_keeps_only_mtplx_mlx_and_pythonpath():
    capture = capture_launch(
        {
            "MTPLX_THINKING_BUDGET": "6144",
            "MLX_SDPA_D256_MINQ": "64",
            "PYTHONPATH": "/opt/mlx:/opt/mtplx",
            "HOME": "/Users/someone",
            "PATH": "/usr/bin",
        },
        ["--port", "8123"],
        executable="/venv/bin/python",
    )

    assert dict(capture.env) == {
        "MTPLX_THINKING_BUDGET": "6144",
        "MLX_SDPA_D256_MINQ": "64",
        "PYTHONPATH": "/opt/mlx:/opt/mtplx",
    }
    assert capture.argv == (
        "/venv/bin/python",
        "-m",
        "mtplx.server.openai",
        "--port",
        "8123",
    )
    assert capture.launcher == "server"


def test_sources_tell_launch_profile_both_and_flag_apart():
    capture = capture_launch(
        {
            "MTPLX_THINKING_BUDGET": "6144",
            "MTPLX_PREFILL_CHUNK_SIZE_DENSE": "4096",
            "MTPLX_NAX_VERIFY": "1",
        },
        [],
        executable="python",
    )
    final_env = {
        "MTPLX_THINKING_BUDGET": "6144",
        "MTPLX_PREFILL_CHUNK_SIZE_DENSE": "4096",
        "MTPLX_NAX_VERIFY": "1",
        "MTPLX_DROP_EVENTS": "1",
        "MTPLX_MEMORY_BUDGET": "1000",
    }
    profile_env = {
        "MTPLX_PREFILL_CHUNK_SIZE_DENSE": "2048",
        "MTPLX_NAX_VERIFY": "1",
        "MTPLX_DROP_EVENTS": "1",
    }

    info = build_launch_info(
        capture, environ=final_env, profile_env=profile_env, fixed=FIXED
    )
    rows = _rows(info)

    assert rows["MTPLX_THINKING_BUDGET"]["source"] == "launch"
    assert rows["MTPLX_DROP_EVENTS"]["source"] == "profile"
    assert rows["MTPLX_MEMORY_BUDGET"]["source"] == "flag"
    dense = rows["MTPLX_PREFILL_CHUNK_SIZE_DENSE"]
    assert dense["source"] == "both"
    assert dense["value"] == "4096"
    assert dense["launch_value"] == "4096"
    assert dense["profile_value"] == "2048"
    assert rows["MTPLX_NAX_VERIFY"]["source"] == "both"
    assert info["fixed"] == FIXED
    # The restart command carries what the operator passed, never what the
    # profile or a flag derives at startup.
    assert [entry["name"] for entry in info["restart"]["env"]] == [
        "MTPLX_NAX_VERIFY",
        "MTPLX_PREFILL_CHUNK_SIZE_DENSE",
        "MTPLX_THINKING_BUDGET",
    ]


def test_sources_follow_the_real_profile_applier():
    environ = {
        # Operator-overridable: the launch value wins over the profile.
        "MTPLX_PREFILL_CHUNK_SIZE_DENSE": "4096",
        # Not in the profile at all.
        "MTPLX_THINKING_BUDGET": "6144",
    }
    capture = capture_launch(environ, [], executable="python")
    profile = get_profile("turbo")
    apply_profile_env(profile.name, environ=environ)

    info = build_launch_info(
        capture, environ=environ, profile_env=profile.env_dict(), fixed=FIXED
    )
    rows = _rows(info)

    dense = rows["MTPLX_PREFILL_CHUNK_SIZE_DENSE"]
    assert dense["source"] == "both"
    assert dense["value"] == "4096"
    assert (
        dense["profile_value"] == profile.env_dict()["MTPLX_PREFILL_CHUNK_SIZE_DENSE"]
    )
    assert rows["MTPLX_THINKING_BUDGET"]["source"] == "launch"
    for name in profile.env_dict():
        if name not in capture.env:
            assert rows[name]["source"] == "profile"


def test_pythonpath_is_listed_and_kept_in_the_restart_command():
    capture = capture_launch(
        {"PYTHONPATH": "/opt/mlx:/opt/mtplx"}, [], executable="python"
    )
    info = build_launch_info(
        capture,
        environ={"PYTHONPATH": "/opt/mlx:/opt/mtplx"},
        profile_env={},
        fixed=FIXED,
    )

    assert _rows(info)["PYTHONPATH"] == {
        "name": "PYTHONPATH",
        "value": "/opt/mlx:/opt/mtplx",
        "source": "launch",
    }
    assert info["restart"]["env"] == [
        {"name": "PYTHONPATH", "value": "/opt/mlx:/opt/mtplx"}
    ]


@pytest.mark.parametrize(
    ("name", "secret"),
    [
        ("MTPLX_API_KEY", True),
        ("MTPLX_HF_TOKEN", True),
        ("MTPLX_CLIENT_SECRET", True),
        ("MTPLX_PROXY_PASSWORD", True),
        ("--api-key", True),
        ("MTPLX_WARMUP_TOKENS", False),
        ("MTPLX_PAGED_KV_QUANT", False),
        ("MTPLX_API_KEY_FILE", False),
        ("--api-key-file", False),
    ],
)
def test_secret_names_match_whole_words(name, secret):
    assert is_secret_name(name) is secret


def test_secret_values_are_redacted_everywhere():
    environ = {
        "MTPLX_API_KEY": "sk-live",
        "MTPLX_HF_TOKEN": "hf_abc",
        "MTPLX_WARMUP_TOKENS": "64",
    }
    capture = capture_launch(
        environ,
        ["--api-key", "sk-live", "--port", "8123"],
        executable="python",
    )
    info = build_launch_info(capture, environ=environ, profile_env={}, fixed=FIXED)
    payload = json.dumps(info)

    assert "sk-live" not in payload
    assert "hf_abc" not in payload
    rows = _rows(info)
    assert rows["MTPLX_API_KEY"]["value"] == REDACTED
    assert rows["MTPLX_HF_TOKEN"]["value"] == REDACTED
    assert rows["MTPLX_WARMUP_TOKENS"]["value"] == "64"
    assert info["argv"][-4:] == ["--api-key", REDACTED, "--port", "8123"]


def test_redact_argv_handles_both_flag_spellings():
    assert redact_argv(["--api-key=abc", "--api-key", "def", "--no-auth"]) == [
        f"--api-key={REDACTED}",
        "--api-key",
        REDACTED,
        "--no-auth",
    ]
    # A secret flag without a value leaves the next flag alone.
    assert redact_argv(["--api-key", "--port", "1"]) == ["--api-key", "--port", "1"]


def test_serve_handoff_restores_the_operator_command():
    parent_env = {"MTPLX_THINKING_BUDGET": "6144", "MTPLX_PAGED_KV_QUANT": "q8"}
    child_env = {
        **parent_env,
        "MTPLX_PAGED_KV_QUANT": "off",
        "MTPLX_VLLM_METAL_PAGED_KV_QUANT": "off",
        "HOME": "/Users/someone",
    }
    serve_argv = ["/venv/bin/mtplx", "serve", "--port", "8123", "--no-auth"]
    child_env[SERVE_LAUNCH_ENV] = serve_launch_handoff(
        argv=serve_argv, parent_env=parent_env, child_env=child_env
    )

    capture = capture_launch(
        child_env, ["--model", "/m", "--port", "8123"], executable="python"
    )
    info = build_launch_info(capture, environ=child_env, profile_env={}, fixed=FIXED)

    assert capture.launcher == "mtplx serve"
    assert capture.argv == tuple(serve_argv)
    assert info["restart"]["argv"] == serve_argv
    assert [entry["name"] for entry in info["restart"]["env"]] == [
        "MTPLX_THINKING_BUDGET"
    ]
    rows = _rows(info)
    assert rows["MTPLX_PAGED_KV_QUANT"]["source"] == "flag"
    assert rows["MTPLX_VLLM_METAL_PAGED_KV_QUANT"]["source"] == "flag"
    assert SERVE_LAUNCH_ENV not in rows


def test_malformed_serve_handoff_falls_back_to_the_server_command():
    capture = capture_launch(
        {SERVE_LAUNCH_ENV: "{not json", "MTPLX_THINKING_BUDGET": "1"},
        ["--port", "8123"],
        executable="python",
    )

    assert capture.launcher == "server"
    assert capture.argv[-2:] == ("--port", "8123")
    assert dict(capture.env) == {"MTPLX_THINKING_BUDGET": "1"}


def test_serve_invocation_argv_makes_python_m_runnable():
    assert serve_invocation_argv(
        ["/x/mtplx/__main__.py", "serve", "--port", "1"], executable="/py"
    ) == ["/py", "-m", "mtplx", "serve", "--port", "1"]
    assert serve_invocation_argv(["/venv/bin/mtplx", "serve"], executable="/py") == [
        "/venv/bin/mtplx",
        "serve",
    ]


def test_serve_wrapper_hands_its_command_to_the_child(monkeypatch):
    from mtplx.commands import public

    monkeypatch.setattr(sys, "argv", ["/venv/bin/mtplx", "serve", "--no-auth"])
    monkeypatch.delenv("MTPLX_PAGED_KV_QUANT", raising=False)
    child_env = {"MTPLX_PAGED_KV_QUANT": "off"}

    env = public._with_serve_launch_handoff(child_env)

    assert SERVE_LAUNCH_ENV not in child_env
    assert json.loads(env[SERVE_LAUNCH_ENV]) == {
        "argv": ["/venv/bin/mtplx", "serve", "--no-auth"],
        "env_keys": ["MTPLX_PAGED_KV_QUANT"],
    }


# ---- GET /v1/mtplx/launch ------------------------------------------------


def _client(state):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from mtplx.server.openai import create_app

    return TestClient(create_app(state))


def _fake_state():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_server_openai import _fake_state as fake_state

    return fake_state()


def test_launch_endpoint_returns_the_startup_capture():
    from mtplx.server.openai import _launch_fixed_settings

    state = _fake_state()
    environ = {"MTPLX_THINKING_BUDGET": "6144", "MTPLX_API_KEY": "sk-live"}
    state.launch_info = build_launch_info(
        capture_launch(environ, ["--port", "8123"], executable="python"),
        environ=environ,
        profile_env={},
        fixed=_launch_fixed_settings(state.args),
    )

    response = _client(state).get("/v1/mtplx/launch")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["fixed"]["profile"] == state.args.profile
    assert body["fixed"]["port"] == state.args.port
    assert body["fixed"]["api_key_required"] is False
    assert _rows(body)["MTPLX_API_KEY"]["value"] == REDACTED
    assert "sk-live" not in response.text


def test_launch_endpoint_without_capture_says_so():
    response = _client(_fake_state()).get("/v1/mtplx/launch")

    assert response.status_code == 200
    assert response.json() == {"ok": False, "reason": "launch_info_unavailable"}


def test_app_capabilities_list_the_launch_endpoint():
    response = _client(_fake_state()).get("/v1/mtplx/app/capabilities")

    assert response.json()["endpoints"]["launch"] == "/v1/mtplx/launch"
