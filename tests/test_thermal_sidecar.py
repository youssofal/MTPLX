"""Tests for the detached fan-restore sidecar.

The sidecar is the only piece of the crash-safety machinery that handles
SIGKILL and terminal-close (signal handlers can't catch those, and the
marker-file recovery only fires on the *next* MTPLX invocation). We
exercise it without spawning real subprocesses by importing the module
and driving its building blocks directly.
"""

from __future__ import annotations

import json
import os
import subprocess
import time

import pytest

from mtplx import thermal_sidecar


@pytest.fixture(autouse=True)
def _default_no_daemon_socket(monkeypatch):
    """Default every test to "no ThermalForge daemon socket" so the suite never
    touches a real daemon on the dev machine. The socket-path test opts in."""
    monkeypatch.setattr(thermal_sidecar, "_daemon_socket_send", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _scratch_home_and_verified_fans(monkeypatch, tmp_path):
    """Keep the sidecar log under a scratch HOME (never the real ~/.mtplx) and
    default fan verification to "fans report auto" so no test reads the real
    fan controller. Tests about unverified restores override it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(thermal_sidecar, "wait_for_auto_fans", lambda **k: True)


def _sidecar_log(tmp_path) -> str:
    path = tmp_path / ".mtplx" / "logs" / "thermal-sidecar.log"
    return path.read_text() if path.exists() else ""


def _socket_ok(*a, **k):
    return {"ok": True, "response": "ok", "command": ["<sock>", "auto"]}


class _FakeProc:
    def __init__(self, returncode: int = 0, stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = ""
        self.stderr = stderr


def test_parent_alive_returns_true_for_self():
    assert thermal_sidecar._parent_alive(os.getpid()) is True


def test_restore_fans_prefers_daemon_socket(monkeypatch):
    """When the daemon socket answers and the fans verify back on auto, restore
    through it and never shell out to sudo (which needs a password and would
    run the app-killing CLI)."""
    monkeypatch.setattr(thermal_sidecar, "_daemon_socket_send", _socket_ok)

    def boom(*a, **k):
        raise AssertionError("must not shell out when the daemon socket handles it")

    monkeypatch.setattr(subprocess, "run", boom)
    assert thermal_sidecar._restore_fans("/path/to/thermalforge") == (True, "daemon socket")


def test_parent_alive_returns_false_for_dead_pid():
    # 999_999_999 is far above kernel.pid_max on macOS — definitely free.
    assert thermal_sidecar._parent_alive(999_999_999) is False


def test_parent_alive_returns_false_for_zero_or_negative():
    assert thermal_sidecar._parent_alive(0) is False
    assert thermal_sidecar._parent_alive(-1) is False


def test_clear_marker_no_op_when_missing(tmp_path):
    marker = tmp_path / "missing.json"
    thermal_sidecar._clear_marker(str(marker))  # must not raise
    assert not marker.exists()


def test_clear_marker_deletes_existing_file(tmp_path):
    marker = tmp_path / "active.json"
    marker.write_text("{}")
    thermal_sidecar._clear_marker(str(marker))
    assert not marker.exists()


def test_clear_marker_handles_none():
    thermal_sidecar._clear_marker(None)  # must not raise


def test_restore_fans_runs_sudo_thermalforge_auto(monkeypatch):
    """Sidecar must invoke ``sudo -n <binary> auto`` (passwordless),
    never a plain ``thermalforge auto`` (which fails with "Run with
    sudo"), and never an interactive ``sudo`` (which would prompt
    against /dev/null and hang)."""

    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(list(cmd))
        assert kwargs.get("stdin") is subprocess.DEVNULL
        assert kwargs.get("timeout"), "sudo -n must be bounded"
        return _FakeProc(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    verified, detail = thermal_sidecar._restore_fans("/path/to/thermalforge")

    assert verified is True
    assert detail == "sudo -n /path/to/thermalforge auto"
    assert captured == [["sudo", "-n", "/path/to/thermalforge", "auto"]]


def test_restore_fans_reports_subprocess_exceptions_as_unverified(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("simulated")

    monkeypatch.setattr(subprocess, "run", boom)
    verified, detail = thermal_sidecar._restore_fans("/path/to/thermalforge")
    assert verified is False
    assert "no daemon socket" in detail
    assert "could not run: simulated" in detail


def test_main_exits_immediately_when_parent_already_dead(monkeypatch, tmp_path):
    """If the parent is gone before the sidecar's first poll, restore
    must run on iteration 1 and the sidecar must exit with 0."""

    marker = tmp_path / "active.json"
    marker.write_text("{}")
    captured: list[list[str]] = []

    monkeypatch.setattr(thermal_sidecar, "_detach_from_terminal", lambda: None)
    monkeypatch.setattr(thermal_sidecar, "_parent_alive", lambda pid: False)

    def fake_run(cmd, *args, **kwargs):
        captured.append(list(cmd))
        return _FakeProc(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = thermal_sidecar.main(
        [
            "--parent-pid",
            "1",
            "--binary",
            "/path/to/thermalforge",
            "--marker",
            str(marker),
            "--poll-seconds",
            "0.1",
        ]
    )

    assert rc == 0
    assert captured == [["sudo", "-n", "/path/to/thermalforge", "auto"]]
    assert not marker.exists()  # marker cleared
    assert "fans restored to auto via sudo -n /path/to/thermalforge auto" in _sidecar_log(tmp_path)


def test_main_keeps_marker_when_restore_command_fails(monkeypatch, tmp_path):
    marker = tmp_path / "active.json"
    marker.write_text("{}")

    monkeypatch.setattr(thermal_sidecar, "_detach_from_terminal", lambda: None)
    monkeypatch.setattr(thermal_sidecar, "_parent_alive", lambda pid: False)

    def fake_run(cmd, *args, **kwargs):
        return _FakeProc(returncode=1, stderr="sudo: a password is required")

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = thermal_sidecar.main(
        [
            "--parent-pid",
            "1",
            "--binary",
            "/path/to/thermalforge",
            "--marker",
            str(marker),
            "--poll-seconds",
            "0.1",
        ]
    )

    assert rc == 1
    assert marker.exists()
    # The sidecar's stdio is /dev/null; the reason must land in the log file.
    log = _sidecar_log(tmp_path)
    assert "fan restore NOT verified, marker kept" in log
    assert "exited 1: sudo: a password is required" in log


# -- C-08: an unverified "ok" never clears the recovery marker -----------------


def test_unverified_daemon_ok_does_not_clear_marker(monkeypatch, tmp_path):
    """A wedged ThermalForge daemon can answer ok without touching the fans.
    The sidecar used to return 0 on that reply and delete the marker, so the
    next `mtplx start` found nothing to recover and fans stayed at maximum.
    Now the reply is held to the fan rows, the CLI fallback runs and is held
    to the same proof, and the marker survives an unverified restore."""
    marker = tmp_path / "active.json"
    marker.write_text(json.dumps({"pid": 1, "owner_token": "lease"}))
    monkeypatch.setattr(thermal_sidecar, "_daemon_socket_send", _socket_ok)
    monkeypatch.setattr(thermal_sidecar, "wait_for_auto_fans", lambda **k: False)
    monkeypatch.setattr(thermal_sidecar, "_detach_from_terminal", lambda: None)
    monkeypatch.setattr(thermal_sidecar, "_parent_alive", lambda pid: False)
    ran: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        ran.append(list(cmd))
        return _FakeProc(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = thermal_sidecar.main(
        [
            "--parent-pid",
            "1",
            "--binary",
            "/path/to/thermalforge",
            "--marker",
            str(marker),
            "--owner-token",
            "lease",
            "--poll-seconds",
            "0.1",
        ]
    )

    assert rc == 1
    assert marker.exists(), "an unverified restore must leave the marker for the next start"
    assert json.loads(marker.read_text())["owner_token"] == "lease"
    assert ran == [["sudo", "-n", "/path/to/thermalforge", "auto"]]
    log = _sidecar_log(tmp_path)
    assert "marker kept" in log
    assert "daemon replied ok but the fans did not report auto" in log
    assert "exited 0 but the fans did not report auto" in log


def test_daemon_ok_unverified_falls_through_to_sudo_and_verified_restore_clears_marker(
    monkeypatch, tmp_path
):
    marker = tmp_path / "active.json"
    marker.write_text(json.dumps({"pid": 1, "owner_token": "lease"}))
    monkeypatch.setattr(thermal_sidecar, "_daemon_socket_send", _socket_ok)
    verdicts = iter([False, True])  # socket reply: not verified; after sudo: verified
    monkeypatch.setattr(thermal_sidecar, "wait_for_auto_fans", lambda **k: next(verdicts))
    ran: list[list[str]] = []
    monkeypatch.setattr(
        subprocess, "run", lambda cmd, **k: ran.append(list(cmd)) or _FakeProc(0)
    )

    rc = thermal_sidecar._restore_owned_fans("/path/to/thermalforge", str(marker), "lease")

    assert rc == 0
    assert ran == [["sudo", "-n", "/path/to/thermalforge", "auto"]]
    assert not marker.exists()
    assert "fans restored to auto via sudo -n /path/to/thermalforge auto; marker cleared" in (
        _sidecar_log(tmp_path)
    )


def test_verified_daemon_restore_clears_marker_and_logs(monkeypatch, tmp_path):
    marker = tmp_path / "active.json"
    marker.write_text(json.dumps({"pid": 1, "owner_token": "lease"}))
    monkeypatch.setattr(thermal_sidecar, "_daemon_socket_send", _socket_ok)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("no CLI fallback after a verified socket restore")
    )

    rc = thermal_sidecar._restore_owned_fans("/path/to/thermalforge", str(marker), "lease")

    assert rc == 0
    assert not marker.exists()
    assert "fans restored to auto via daemon socket; marker cleared" in _sidecar_log(tmp_path)


def test_old_sidecar_does_not_restore_newer_owner(monkeypatch, tmp_path):
    """The old watchdog can observe parent death after a new daemon starts."""

    marker = tmp_path / "active.json"
    marker.write_text(json.dumps({"pid": 2, "owner_token": "new-owner"}))
    captured: list[list[str]] = []

    monkeypatch.setattr(thermal_sidecar, "_detach_from_terminal", lambda: None)
    monkeypatch.setattr(thermal_sidecar, "_parent_alive", lambda pid: False)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, *args, **kwargs: captured.append(list(cmd)),
    )

    rc = thermal_sidecar.main(
        [
            "--parent-pid",
            "1",
            "--binary",
            "/path/to/thermalforge",
            "--marker",
            str(marker),
            "--owner-token",
            "old-owner",
            "--poll-seconds",
            "0.1",
        ]
    )

    assert rc == 0
    assert captured == []
    assert json.loads(marker.read_text())["owner_token"] == "new-owner"


def test_main_polls_until_parent_dies(monkeypatch, tmp_path):
    """Sidecar must keep polling while the parent is alive and only
    fire the restore once it's gone."""

    polls = {"n": 0}

    def fake_alive(pid: int) -> bool:
        polls["n"] += 1
        return polls["n"] < 3  # die after 2 alive polls

    captured: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        captured.append(list(cmd))
        class _P:
            returncode = 0
        return _P()

    monkeypatch.setattr(thermal_sidecar, "_detach_from_terminal", lambda: None)
    monkeypatch.setattr(thermal_sidecar, "_parent_alive", fake_alive)
    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = thermal_sidecar.main(
        [
            "--parent-pid",
            "12345",
            "--binary",
            "/path/to/thermalforge",
            "--poll-seconds",
            "0.05",
        ]
    )

    assert rc == 0
    assert polls["n"] == 3
    assert len(captured) == 1


def test_main_respects_max_lifetime(monkeypatch):
    """Hard ceiling so a buggy sidecar can never live forever."""

    monkeypatch.setattr(thermal_sidecar, "_detach_from_terminal", lambda: None)
    monkeypatch.setattr(thermal_sidecar, "_parent_alive", lambda pid: True)

    started = time.time()
    rc = thermal_sidecar.main(
        [
            "--parent-pid",
            str(os.getpid()),
            "--binary",
            "/path/to/thermalforge",
            "--poll-seconds",
            "0.1",
            "--max-lifetime-seconds",
            "0.3",
        ]
    )
    elapsed = time.time() - started
    assert rc == 0
    assert elapsed < 2.0, f"sidecar overran its lifetime ceiling ({elapsed:.2f}s)"


def test_install_max_lifecycle_hooks_spawns_sidecar(monkeypatch, tmp_path):
    """``install_max_lifecycle_hooks`` must call ``_spawn_thermal_sidecar``
    so the parent process gets crash-safety coverage on terminal close."""

    from mtplx import thermal

    monkeypatch.setattr(thermal, "MAX_MARKER_FILE", tmp_path / "max-active.json")
    spawned: list[bool] = []

    def fake_spawn(owner_token=None):
        spawned.append(bool(owner_token))
        return None  # we don't care about the Popen return for this test

    monkeypatch.setattr(thermal, "_spawn_thermal_sidecar", fake_spawn)
    monkeypatch.setattr(
        "mtplx.thermal.set_thermal_profile",
        lambda profile, **kw: {"ok": True},
    )
    monkeypatch.setattr("mtplx.thermal.signal.signal", lambda *a, **kw: None)
    monkeypatch.setattr("mtplx.thermal.atexit.register", lambda *a, **kw: None)

    cleanup = thermal.install_max_lifecycle_hooks()

    assert spawned == [True], "sidecar was not spawned by install_max_lifecycle_hooks"
    cleanup()


def test_spawn_sidecar_passes_owner_token(monkeypatch, tmp_path):
    """The detached watchdog must receive the same lease as its parent."""

    from mtplx import thermal

    captured: list[list[str]] = []

    class _FakeProc:
        pass

    monkeypatch.setattr(thermal, "MAX_MARKER_FILE", tmp_path / "max-active.json")
    monkeypatch.setattr(
        thermal,
        "detect_thermal_control",
        lambda: {
            "available": True,
            "selected": {"kind": "thermalforge", "path": "/path/to/thermalforge"},
        },
    )
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda cmd, **kwargs: captured.append(list(cmd)) or _FakeProc(),
    )

    assert thermal._spawn_thermal_sidecar("lease-123") is not None
    assert captured
    assert captured[0][-2:] == ["--owner-token", "lease-123"]


def _poll_with_parent_alive(monkeypatch, marker, *, lifetime_s: float) -> tuple[int, float, list]:
    captured: list[list[str]] = []
    monkeypatch.setattr(thermal_sidecar, "_detach_from_terminal", lambda: None)
    monkeypatch.setattr(thermal_sidecar, "_parent_alive", lambda pid: True)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, *args, **kwargs: captured.append(list(cmd)),
    )
    started = time.time()
    rc = thermal_sidecar.main(
        [
            "--parent-pid",
            str(os.getpid()),
            "--binary",
            "/path/to/thermalforge",
            "--marker",
            str(marker),
            "--owner-token",
            "ramp-1",
            "--poll-seconds",
            "0.05",
            "--max-lifetime-seconds",
            str(lifetime_s),
        ]
    )
    return rc, time.time() - started, captured


def test_sidecar_exits_once_a_newer_lease_owns_the_fans(monkeypatch, tmp_path):
    """#602: Smart fan mode takes a new lease, with its own sidecar, for every
    busy stretch. The old sidecar's lease is over while the server lives on,
    so it exits instead of waiting for the server, and leaves the new lease
    alone."""

    marker = tmp_path / "active.json"
    marker.write_text(json.dumps({"pid": os.getpid(), "owner_token": "ramp-2"}))

    rc, elapsed, captured = _poll_with_parent_alive(monkeypatch, marker, lifetime_s=30.0)

    assert rc == 0
    assert elapsed < 5.0, f"sidecar outlived its lease by {elapsed:.1f}s"
    assert captured == []
    assert json.loads(marker.read_text())["owner_token"] == "ramp-2"


def test_sidecar_exits_once_its_lease_was_restored_and_cleared(monkeypatch, tmp_path):
    """The parent's own cleanup restored the fans and cleared the marker:
    nothing is left to guard."""

    marker = tmp_path / "active.json"

    rc, elapsed, captured = _poll_with_parent_alive(monkeypatch, marker, lifetime_s=30.0)

    assert rc == 0
    assert elapsed < 5.0
    assert captured == []
    assert not marker.exists()


def test_sidecar_keeps_watching_while_its_lease_is_current(monkeypatch, tmp_path):
    """Control: while the marker still carries this sidecar's lease it keeps
    polling (here until its lifetime ceiling), so a SIGKILL of the parent is
    still covered."""

    marker = tmp_path / "active.json"
    marker.write_text(json.dumps({"pid": os.getpid(), "owner_token": "ramp-1"}))

    rc, elapsed, captured = _poll_with_parent_alive(monkeypatch, marker, lifetime_s=0.4)

    assert rc == 0
    assert elapsed >= 0.4
    assert captured == []
    assert json.loads(marker.read_text())["owner_token"] == "ramp-1"
