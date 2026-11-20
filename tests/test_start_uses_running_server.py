"""`mtplx start <client>` beside a running MTPLX server uses that server.

OpenCode, Swival and Hermes start their own server on 18083-18085 while the
app's daemon sits on 8000 (or the port it persisted), so `mtplx start
opencode` with the app running loaded a second copy of the model. With no
``--port``, a running server now becomes the target's port and the existing
"MTPLX is already running" path sets the client up against it, as terminal
chat already does. A different model named with ``--model`` still starts
its own server.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mtplx import daemon_client
from mtplx.cli import main
from mtplx.commands import public
from mtplx.default_models import public_model_id_for_ref

REPO = "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"


def _daemon(port, model=None, *, launch_id="app-launch", model_path=None):
    return daemon_client.RunningDaemon(
        host="127.0.0.1",
        port=port,
        model=model or public_model_id_for_ref(REPO),
        model_path=model_path,
        launch_id=launch_id,
        pid=4242,
        api_key_required=False,
        health={"ok": True},
    )


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.setenv("MTPLX_START_ATTACH_PROBE", "on")
    monkeypatch.setattr(daemon_client, "default_probe_ports", lambda: (8000, 18083, 18084, 18085))
    calls = []
    running: list[daemon_client.RunningDaemon] = []

    def fake_probe(*, host, ports, **_kwargs):
        calls.append(tuple(ports))
        return [d for port in ports for d in running if d.port == port]

    monkeypatch.setattr(daemon_client, "probe_running_daemons", fake_probe)
    return SimpleNamespace(calls=calls, running=running)


def _args(port, *, model=REPO, explicit=False):
    return SimpleNamespace(host="127.0.0.1", port=port, model=model, _model_explicit=explicit)


@pytest.mark.parametrize("target,port", [("opencode", 18083), ("swival", 18084), ("hermes", 18085)])
def test_a_client_on_its_own_port_uses_the_app_daemon(probe, capsys, target, port):
    probe.running.append(_daemon(8000))
    args = _args(port)
    public._quickstart_use_running_server(args, target=target, cli_flags=set())
    assert args.port == 8000
    assert probe.calls == [(port, 8000, *(p for p in (18083, 18084, 18085) if p != port))]
    out = capsys.readouterr().out
    assert "MTPLX is already running on port 8000" in out
    assert "started by the MTPLX app" in out
    assert f"instead of loading a second copy of the model on port {port}" in out


def test_nothing_running_keeps_the_targets_own_port(probe):
    args = _args(18083)
    public._quickstart_use_running_server(args, target="opencode", cli_flags=set())
    assert args.port == 18083


def test_a_server_on_the_targets_own_port_is_left_to_the_reuse_path(probe, capsys):
    probe.running.extend([_daemon(18083), _daemon(8000)])
    args = _args(18083)
    public._quickstart_use_running_server(args, target="opencode", cli_flags=set())
    assert args.port == 18083
    assert capsys.readouterr().out == ""


def test_the_same_model_named_explicitly_still_uses_the_running_server(probe):
    probe.running.append(_daemon(8000))
    args = _args(18085, explicit=True)
    public._quickstart_use_running_server(args, target="hermes", cli_flags={"model"})
    assert args.port == 8000


def test_another_model_named_explicitly_starts_its_own_server(probe):
    probe.running.append(_daemon(8000, model="mtplx-some-other-model"))
    args = _args(18083, explicit=True)
    public._quickstart_use_running_server(args, target="opencode", cli_flags={"model"})
    assert args.port == 18083


def test_a_port_the_user_named_is_never_moved(probe):
    probe.running.append(_daemon(8000))
    args = _args(18083)
    public._quickstart_use_running_server(args, target="opencode", cli_flags={"port"})
    assert args.port == 18083
    assert probe.calls == []


def test_the_probe_kill_switch_and_non_client_targets_are_respected(probe, monkeypatch):
    probe.running.append(_daemon(8000))
    for target in ("dashboard", "terminal", "cli"):
        args = _args(8001)
        public._quickstart_use_running_server(args, target=target, cli_flags=set())
        assert args.port == 8001
    monkeypatch.setenv("MTPLX_START_ATTACH_PROBE", "off")
    args = _args(18083)
    public._quickstart_use_running_server(args, target="opencode", cli_flags=set())
    assert args.port == 18083
    assert probe.calls == []


def test_the_app_daemon_on_a_moved_port_is_found_for_pi(probe, monkeypatch):
    """The app persists a port it had to move off 8000; `mtplx start pi` on
    8000 used to start a second server beside it."""
    monkeypatch.setattr(daemon_client, "default_probe_ports", lambda: (8001, 8000, 18083, 18084, 18085))
    probe.running.append(_daemon(8001))
    args = _args(8000)
    public._quickstart_use_running_server(args, target="pi", cli_flags=set())
    assert args.port == 8001


def test_without_a_model_the_running_servers_model_is_the_one_checked(probe, tmp_path):
    """No model menu and no offer to download this Mac's default when the
    model that will answer is the running server's, here on the target's own
    port as well as on another one."""
    pack = tmp_path / "Qwen3.8-Flash-Next-MTPLX-Optimized-Speed"
    pack.mkdir()
    for own, running in ((18083, 8000), (8000, 8000)):
        probe.running[:] = [_daemon(running, model_path=str(pack))]
        args = _args(own, model=None)
        args._cli_flags = set()
        public._quickstart_use_running_server(args, target="opencode" if own != 8000 else "pi", cli_flags=set())
        assert (args.port, args.model, args._model_explicit) == (8000, str(pack), True)
        assert "model" in args._cli_flags and args._running_server_model is True
    # A model the user named stays theirs; a path this Mac cannot read is
    # left to the usual check.
    probe.running[:] = [_daemon(8000, model_path=str(pack))]
    named = _args(18083, model=REPO, explicit=True)
    public._quickstart_use_running_server(named, target="opencode", cli_flags={"model"})
    assert named.model == REPO
    probe.running[:] = [_daemon(8000, model_path=str(tmp_path / "gone"))]
    unread = _args(18083, model=None)
    public._quickstart_use_running_server(unread, target="opencode", cli_flags=set())
    assert unread.port == 8000 and unread.model is None


def test_the_model_menu_is_skipped_for_the_running_servers_model(monkeypatch, tmp_path):
    monkeypatch.setattr(public.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda *_a: (_ for _ in ()).throw(AssertionError("asked")))
    args = SimpleNamespace(model=str(tmp_path), _model_explicit=True, _running_server_model=True, download=False)
    assert public._quickstart_choose_model(args, target="pi") == (str(tmp_path), False)


def test_start_opencode_beside_the_app_hands_opencode_to_the_app_daemon(probe, monkeypatch, tmp_path):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(public.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    resolved = []

    def fake_resolve(model, *a, **k):
        resolved.append(model)
        return model, {}

    monkeypatch.setattr(public, "_quickstart_resolve_model", fake_resolve)
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda *a, **k: ({"model_dir": "models/example", "compatibility": {"can_run": True}}, None),
    )
    captured = {}

    def fake_serve(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(public, "cmd_serve_public", fake_serve)
    pack = tmp_path / "Qwen3.8-Flash-Next-MTPLX-Optimized-Speed"
    pack.mkdir()
    probe.running.append(_daemon(8000, model_path=str(pack)))

    assert main(["start", "opencode", "--yes"]) == 0

    serve_args = captured["args"]
    assert serve_args.quickstart_opencode is True
    assert serve_args.port == 8000
    assert resolved == [str(pack)]
