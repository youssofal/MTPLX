"""Standalone fan-restore sidecar for ``--max`` sessions.

Spawned as a detached child by ``cmd_serve_public`` when MAX mode pins
the fans. The sidecar:

  1. Detaches itself from the parent's controlling terminal via ``setsid``
     so closing the terminal window or sending SIGHUP to the process
     group does NOT kill it too.
  2. Polls the parent PID every ``poll_seconds``, and exits quietly once
     its Max lease is released: cleared by a verified restore, or replaced
     by a newer lease. Smart fan mode takes a lease per ramp, so a sidecar
     that outlived its lease would linger until the server exited (#602).
  3. The moment the parent is gone (any cause: clean exit, SIGINT,
     SIGTERM, SIGHUP, SIGKILL, OOM, terminal closed, kernel panic
     followed by reboot — well, except that last one), it restores Auto
     only if its ownership token still matches the global Max marker.
     A newer MTPLX process can therefore take over without the old
     watchdog undoing the new Max pin.

This is the only piece of the crash-safety machinery that handles
SIGKILL of the parent. The signal-handler / atexit path covers
recoverable signals; the marker-file path covers "user runs MTPLX
again later"; the sidecar covers "user closes the terminal and walks
away".

The sidecar runs as the unprivileged user and relies on the
``/etc/sudoers.d/mtplx-thermalforge`` NOPASSWD rule installed by
``mtplx max --install`` to do its restore without a password.

A restore only counts once the fan rows report the automatic curve again
(#201); the Max marker is cleared on a verified restore only, so a wedged
daemon that answers "ok" without touching the fans leaves the marker for
the next ``mtplx start`` to recover from. Because stdio is detached, the
outcome is appended to ``~/.mtplx/logs/thermal-sidecar.log``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from mtplx.thermal import (
    _daemon_socket_send,
    _max_marker_lock,
    _max_marker_owned_by,
    wait_for_auto_fans,
)

# The sidecar fires once and has nowhere to retry from, so it waits a little
# longer than the in-process path for the daemon to act.
RESTORE_VERIFY_TIMEOUT_S = 5.0
SUDO_AUTO_TIMEOUT_S = 20.0


def _log_path() -> str:
    return os.path.expanduser("~/.mtplx/logs/thermal-sidecar.log")


def _log(message: str) -> None:
    """Append one timestamped line to the sidecar log.

    The sidecar's stdio points at /dev/null, so this file is the only place
    a failed restore can be reported. Writing the log is itself best-effort:
    there is nowhere further to report a log that cannot be written.
    """

    path = _log_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            handle.write(f"{stamp} pid={os.getpid()} {message}\n")
    except OSError:
        pass


def _detach_from_terminal() -> None:
    """Best-effort detach from the controlling terminal.

    A second fork would be the textbook double-fork daemonization, but
    ``setsid`` + closing stdio is enough for our use case: we only need
    to survive the parent's death, not to live forever as a system
    daemon.
    """

    try:
        os.setsid()
    except OSError:
        pass

    devnull = os.open(os.devnull, os.O_RDWR)
    try:
        for fd in (0, 1, 2):
            try:
                os.dup2(devnull, fd)
            except OSError:
                pass
    finally:
        if devnull > 2:
            try:
                os.close(devnull)
            except OSError:
                pass


def _parent_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # The pid exists but isn't ours. Treat as alive — better to
        # leave fans pinned for an extra poll cycle than to restore
        # while the real owner is still running.
        return True
    except OSError:
        return False


def _restore_fans(binary: str) -> tuple[bool, str]:
    """Restore Apple-auto fans and prove it.

    Returns ``(verified, detail)``: ``detail`` names the path that restored the
    fans, or lists every attempt and why it did not count.

    Prefer the ThermalForge daemon socket: it resets fans as root without sudo
    and, unlike ``thermalforge auto``, never quits the menu bar app. Its "ok"
    reply is believed only once the fan rows report auto (#201); otherwise
    fall through to ``sudo -n <binary> auto``, which never prompts, and hold
    that to the same proof.
    """

    reasons: list[str] = []
    reset = _daemon_socket_send("auto")
    if reset is None:
        reasons.append("no daemon socket")
    elif not reset["ok"]:
        reasons.append(f"daemon replied {reset.get('response') or 'nothing'!r}")
    elif wait_for_auto_fans(timeout_s=RESTORE_VERIFY_TIMEOUT_S):
        return True, "daemon socket"
    else:
        reasons.append(
            "daemon replied ok but the fans did not report auto within "
            f"{RESTORE_VERIFY_TIMEOUT_S:g}s"
        )

    cli = f"sudo -n {binary} auto"
    try:
        proc = subprocess.run(
            ["sudo", "-n", binary, "auto"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=SUDO_AUTO_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        reasons.append(f"{cli} did not finish within {SUDO_AUTO_TIMEOUT_S:g}s")
        return False, "; ".join(reasons)
    except OSError as exc:
        reasons.append(f"{cli} could not run: {exc}")
        return False, "; ".join(reasons)
    if proc.returncode != 0:
        said = (proc.stderr or proc.stdout or "").strip()
        reasons.append(f"{cli} exited {proc.returncode}" + (f": {said}" if said else ""))
        return False, "; ".join(reasons)
    if wait_for_auto_fans(timeout_s=RESTORE_VERIFY_TIMEOUT_S):
        return True, cli
    reasons.append(
        f"{cli} exited 0 but the fans did not report auto within {RESTORE_VERIFY_TIMEOUT_S:g}s"
    )
    return False, "; ".join(reasons)


def _clear_marker(marker_path: str | None) -> None:
    if not marker_path:
        return
    try:
        if os.path.exists(marker_path):
            os.unlink(marker_path)
    except OSError:
        pass


def _restore_owned_fans(
    binary: str,
    marker_path: str | None,
    owner_token: str | None,
) -> int:
    """Restore only while this sidecar still owns the global Max lease.

    The marker is cleared only after the restore is verified; an unverified
    attempt leaves it in place so the next ``mtplx start`` recovers the fans,
    and the reason is written to the sidecar log.
    """

    with _max_marker_lock(marker_path):
        if not _max_marker_owned_by(marker_path, owner_token):
            return 0
        try:
            verified, detail = _restore_fans(binary)
        except Exception as exc:  # last resort: a detached process has no other channel
            verified, detail = False, f"{type(exc).__name__}: {exc}"
        if verified:
            _clear_marker(marker_path)
            _log(f"fans restored to auto via {detail}; marker cleared")
            return 0
        _log(
            "fan restore NOT verified, marker kept so the next mtplx start "
            f"recovers the fans ({marker_path or 'no marker'}): {detail}"
        )
        return 1


def _lease_released(marker_path: str | None, owner_token: str) -> bool:
    """True once the Max lease this sidecar guards is gone.

    Writers rewrite the marker in place under the marker lock, so it is read
    under the lock too: a half-written marker must not read as a lost lease.
    """

    with _max_marker_lock(marker_path):
        return not _max_marker_owned_by(marker_path, owner_token)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--binary", required=True, help="Path to thermalforge CLI")
    parser.add_argument("--marker", default=None, help="Marker file to delete after restore")
    parser.add_argument(
        "--owner-token",
        default=None,
        help="Opaque marker lease; prevents an old sidecar restoring a newer session",
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--max-lifetime-seconds", type=float, default=24 * 3600.0,
                        help="Hard ceiling on sidecar lifetime; ensures we eventually die even on bugs")
    args = parser.parse_args(argv)

    _detach_from_terminal()
    started_at = time.time()

    while True:
        if not _parent_alive(args.parent_pid):
            return _restore_owned_fans(
                args.binary,
                args.marker,
                args.owner_token,
            )
        if args.owner_token and _lease_released(args.marker, args.owner_token):
            return 0
        if (time.time() - started_at) > args.max_lifetime_seconds:
            return 0
        try:
            time.sleep(max(0.5, float(args.poll_seconds)))
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
