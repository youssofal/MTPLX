"""Serving without an API key on a trusted network takes two explicit flags (#491).

A non-localhost bind still refuses to start without a key. `--no-auth` alone
keeps its localhost-only meaning (#235), and `--trust-network` alone opens
nothing: only both together serve keyless beyond this Mac, and the startup
banner then says that anyone who can reach the port can use the model.
"""

from __future__ import annotations

import pytest

from mtplx.cli import build_parser
from mtplx.commands import public
from mtplx.server import openai
from tests.test_public_cli import _serve_dry_run_payload_for_model


def _serve_args(tmp_path, *flags):
    return build_parser().parse_args(
        ["serve", "--model", str(tmp_path), "--host", "0.0.0.0", "--yes", *flags]
    )


def test_serve_on_the_network_with_both_flags_starts_keyless_and_forwards_them(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    monkeypatch.setenv("MTPLX_API_KEY", "a-key-the-user-chose-to-switch-off")
    model_dir = tmp_path / "example-model"
    model_dir.mkdir()
    payload = _serve_dry_run_payload_for_model(
        monkeypatch,
        capsys,
        model_dir,
        extra_args=("--host", "0.0.0.0", "--no-auth", "--trust-network"),
    )
    command = payload["server_command"]
    assert "--host 0.0.0.0" in command
    assert "--no-auth" in command and "--trust-network" in command


@pytest.mark.parametrize("flags", [(), ("--no-auth",), ("--trust-network",)])
def test_serve_on_the_network_without_both_flags_still_refuses(
    monkeypatch, tmp_path, capsys, flags
):
    monkeypatch.setattr(public, "_serve_should_onboard", lambda _args: False)
    monkeypatch.delenv("MTPLX_API_KEY", raising=False)
    assert public.cmd_serve_public(_serve_args(tmp_path, *flags)) == 2
    out = capsys.readouterr().out
    assert "--api-key or --api-key-file is required when --host is not localhost" in out
    assert "--no-auth --trust-network" in out


def test_serve_help_documents_the_opt_out(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["serve", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "--trust-network" in help_text
    assert "anyone who can reach the port can use the model" in help_text.lower()


def test_the_server_accepts_a_keyless_network_bind_only_with_both_flags(monkeypatch):
    monkeypatch.setenv("MTPLX_API_KEY", "a-key-the-user-chose-to-switch-off")
    args = openai.parse_args(["--host", "0.0.0.0", "--no-auth", "--trust-network"])
    openai.validate_server_security_args(args)
    assert args.api_key is None
    assert args.api_key_source == "disabled_by_no_auth_flag"

    monkeypatch.delenv("MTPLX_API_KEY")
    for flags in ([], ["--no-auth"], ["--trust-network"]):
        with pytest.raises(SystemExit, match="--no-auth --trust-network"):
            openai.validate_server_security_args(
                openai.parse_args(["--host", "0.0.0.0", *flags])
            )


def test_trust_network_never_drops_a_configured_key(monkeypatch):
    monkeypatch.setenv("MTPLX_API_KEY", "kept")
    args = openai.parse_args(["--host", "0.0.0.0", "--trust-network"])
    openai.validate_server_security_args(args)
    assert args.api_key == "kept"


def test_startup_banner_says_the_network_bind_is_open():
    keyless = openai.parse_args(["--host", "0.0.0.0", "--no-auth", "--trust-network"])
    assert openai._startup_api_key_line(keyless) == (
        "API key: off (--no-auth --trust-network): anyone who can reach port "
        "8000 can use the model"
    )
    local = openai.parse_args(["--no-auth"])
    assert openai._startup_api_key_line(local) == "API key: leave blank for localhost"
