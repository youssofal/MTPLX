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


# -- #602: a sidecar exits once its lease is released --------------------------


def _main_args(marker, owner_token: str) -> list[str]:
    return [
        "--parent-pid",
        "1",
        "--binary",
        "/path/to/thermalforge",
        "--marker",
        str(marker),
        "--owner-token",
        owner_token,
        "--poll-seconds",
        "0.1",
    ]


def _record_fan_commands(monkeypatch) -> list[list[str]]:
    commands: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        commands.append(list(cmd))
        return _FakeProc(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return commands


def test_main_exits_without_restoring_once_its_lease_is_cleared(monkeypatch, tmp_path):
    """Smart fan mode takes a lease per ramp, and a verified restore clears the
    marker. The lease's sidecar used to keep polling the live server until it
    exited, one leftover process per ramp (#602). It now exits on the next poll
    and leaves the fans alone."""
    marker = tmp_path / "active.json"
    marker.write_text(json.dumps({"pid": 1, "owner_token": "lease-a"}))
    fan_commands = _record_fan_commands(monkeypatch)
    sleeps: list[float] = []

    def fake_sleep(seconds):
        if sleeps:
            raise AssertionError("sidecar kept polling after its lease was cleared")
        sleeps.append(seconds)
        marker.unlink()  # the in-process restore cleared the lease

    monkeypatch.setattr(thermal_sidecar, "_detach_from_terminal", lambda: None)
    monkeypatch.setattr(thermal_sidecar, "_parent_alive", lambda pid: True)
    monkeypatch.setattr(time, "sleep", fake_sleep)

    assert thermal_sidecar.main(_main_args(marker, "lease-a")) == 0
    assert len(sleeps) == 1  # held the lease on the first poll, saw it gone on the second
    assert fan_commands == []


def test_main_exits_without_restoring_when_a_newer_lease_replaces_it(monkeypatch, tmp_path):
    marker = tmp_path / "active.json"
    marker.write_text(json.dumps({"pid": 1, "owner_token": "lease-a"}))
    newer = json.dumps({"pid": 1, "owner_token": "lease-b"})
    fan_commands = _record_fan_commands(monkeypatch)

    monkeypatch.setattr(thermal_sidecar, "_detach_from_terminal", lambda: None)
    monkeypatch.setattr(thermal_sidecar, "_parent_alive", lambda pid: True)
    replaced: list[bool] = []

    def fake_sleep(seconds):
        if replaced:
            raise AssertionError("sidecar kept polling after a newer lease replaced it")
        replaced.append(True)
        marker.write_text(newer)

    monkeypatch.setattr(time, "sleep", fake_sleep)

    assert thermal_sidecar.main(_main_args(marker, "lease-a")) == 0
    assert fan_commands == []
    assert marker.read_text() == newer  # the newer lease is untouched


def test_main_keeps_guarding_a_held_lease_until_the_parent_dies(monkeypatch, tmp_path):
    """The lease check must not cost crash safety: while the sidecar still holds
    the lease it keeps polling, and restores the fans when the parent dies."""
    marker = tmp_path / "active.json"
    marker.write_text(json.dumps({"pid": 1, "owner_token": "lease-a"}))
    fan_commands = _record_fan_commands(monkeypatch)
    sleeps: list[float] = []

    monkeypatch.setattr(thermal_sidecar, "_detach_from_terminal", lambda: None)
    monkeypatch.setattr(thermal_sidecar, "_parent_alive", lambda pid: len(sleeps) < 3)
    monkeypatch.setattr(time, "sleep", sleeps.append)

    assert thermal_sidecar.main(_main_args(marker, "lease-a")) == 0
    assert len(sleeps) == 3
    assert fan_commands == [["sudo", "-n", "/path/to/thermalforge", "auto"]]
    assert not marker.exists()  # verified restore cleared it


def test_lease_check_reads_the_marker_under_the_lock(monkeypatch, tmp_path):
    """Writers rewrite the marker in place under the marker lock; reading it
    outside the lock could see a half-written file and exit early."""
    import contextlib

    marker = tmp_path / "active.json"
    marker.write_text(json.dumps({"pid": 1, "owner_token": "lease-a"}))
    locked: list[str] = []

    @contextlib.contextmanager
    def fake_lock(marker_path=None):
        locked.append(marker_path)
        yield

    monkeypatch.setattr(thermal_sidecar, "_max_marker_lock", fake_lock)

    assert thermal_sidecar._lease_released(str(marker), "lease-a") is False
    assert thermal_sidecar._lease_released(str(marker), "lease-b") is True
    assert locked == [str(marker), str(marker)]
