"""Engine child process lifecycle: spawn, health probe, terminate, restart backoff.

Generalizes the single-daemon wrapper in `mtplx/commands/public.py`
(`_run_server_child_with_app_parent_watchdog`, `_terminate_server_child`,
`_pid_is_alive`) from one wrapped `mtplx serve` process to N supervisor-owned
engine children, and replicates the macOS app's busy-vs-dead liveness split
(`DaemonLivenessPolicy.swift`, issue #487: a probe timeout alone never means
dead) and restart backoff (`DaemonSupervisor.swift`'s `DaemonRestartPolicy`)
in Python. Health is fetched the same way `daemon_client.fetch_daemon_health`
does: stdlib HTTP, no new dependency.

Orphan protection: every child is spawned in its own process group
(``start_new_session=True``), so ``terminate()`` can ``os.killpg`` the whole
group (the engine and anything it may have spawned) rather than just the one
pid. The child also inherits ``MTPLX_APP_PARENT_PID`` (and, informationally,
``MTPLX_SUPERVISOR_PID``) set to the supervisor's own pid. `mtplx serve`
already understands ``MTPLX_APP_PARENT_PID`` on its own: `cmd_serve_public`
in `mtplx/commands/public.py` (`_app_parent_pid_from_env`,
`_run_server_child_with_app_parent_watchdog`) watches that pid and tears
down the real server child itself if it vanishes. Setting it to the
supervisor's pid means an engine self-terminates if the supervisor is
SIGKILLed or OOM-killed, without the supervisor having to do anything.

Warmup gating: `probe()` treats a `READY` verdict as gated on any warmup
signal the payload carries. The real engine's `/health` (see
`mtplx/server/openai.py`'s `_startup_health_payload`, ~line 16925, and the
`/health` route at ~line 29674) nests `startup.warmup` = `{"enabled",
"ran", "tokens", "elapsed_s", "error"}`. In practice that blocking startup
warmup (`_run_startup_warmup`) runs during `ServerState.__init__`, before
uvicorn ever starts serving, so `ran` is already True the first time a
probe can succeed at all -- gating on it does not change observed JIT-load
latency against today's real engine. The gate is implemented anyway,
generically, against both that nested shape and a simpler top-level
`{"warmup": {"ready": bool}}` shape (used by the test fixture and left
available for a future engine build that reports async warmup progress),
so a not-yet-ready engine is reported BUSY rather than READY.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Keep only the last N stderr lines in memory; enough to see an OOM message
# or a traceback without letting a wedged child fill an unread pipe.
_STDERR_TAIL_LINES = 200

# Case-insensitive: matched against the joined stderr tail in death_reason().
_OOM_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"out of memory",
        r"kIOGPUCommandBufferCallbackErrorOutOfMemory",
        r"Metal.*memory",
        r"cannot load inside the available Metal memory budget",
        r"MemoryError",
    )
)

# Case-insensitive: matched against the joined stderr tail in death_reason(),
# ahead of the OOM patterns. Lets a bind-collision (see budget/service
# M4 -- two concurrent JIT loads picking the same free port) be retried on a
# fresh port instead of being reported as a generic "exit" failure.
_PORT_IN_USE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"address already in use",
        r"errno 48",
        r"errno 98",
    )
)

# H2 (security review): any env var whose name contains one of these,
# case-insensitive, is stripped before an engine child is spawned so a
# `--no-auth` engine never inherits the supervisor's own credential (e.g.
# MTPLX_API_KEY) via os.environ / /proc/<pid>/environ.
_SECRET_ENV_NAME_RE = re.compile(r"(API_KEY|TOKEN|SECRET)", re.IGNORECASE)


def _scrubbed_env(overrides: dict[str, str] | None) -> dict[str, str]:
    """Parent environment minus anything secret-shaped, with `overrides`
    applied last so they always win (used for MTPLX_APP_PARENT_PID etc,
    none of which match the secret pattern anyway)."""
    env = {key: value for key, value in os.environ.items() if not _SECRET_ENV_NAME_RE.search(key)}
    if overrides:
        env.update(overrides)
    return env


def _warmup_ready(payload: dict) -> bool:
    """True unless `payload` carries an explicit not-yet-ready warmup
    signal. See the module docstring for the two shapes recognized."""
    warmup = payload.get("warmup")
    if isinstance(warmup, dict) and "ready" in warmup:
        return bool(warmup["ready"])
    startup = payload.get("startup")
    if isinstance(startup, dict):
        nested = startup.get("warmup")
        if isinstance(nested, dict) and nested.get("enabled") and "ran" in nested:
            return bool(nested["ran"])
    return True


class Liveness(str, Enum):
    """Verdict from one probe. UNRESPONSIVE is a caller-computed state (see
    `unresponsive_for_s`) for when silence outlasts the grace period; probe()
    itself only ever returns the other four."""

    READY = "ready"
    BUSY = "busy"
    GONE = "gone"
    PORT_CLOSED = "port_closed"
    UNRESPONSIVE = "unresponsive"


@dataclass
class RestartPolicy:
    """Backoff shape for one engine's crash-restart loop. Mirrors the Swift
    app's `DaemonRestartPolicy` defaults."""

    max_attempts: int = 3
    initial_delay_s: float = 1.0
    max_delay_s: float = 30.0
    crash_window_s: float = 120.0


class RestartTracker:
    """Counts recent crashes against a `RestartPolicy` and computes backoff.

    Crashes older than `crash_window_s` are forgotten, so a flaky engine that
    recovers and runs cleanly for a while gets a fresh budget. `generation`
    increments on every recorded crash so a caller that scheduled a delayed
    restart can tell it is stale (mirrors the Swift `restartGeneration`).
    """

    def __init__(self, policy: RestartPolicy) -> None:
        self.policy = policy
        self.generation = 0
        self._crash_times: list[float] = []

    def record_crash(self, now: float) -> float | None:
        """Record a crash at `now`; return the delay before the next restart
        attempt, or None when the policy is exhausted."""
        self.generation += 1
        self._crash_times.append(now)
        window_start = now - self.policy.crash_window_s
        self._crash_times = [t for t in self._crash_times if t >= window_start]
        count = len(self._crash_times)
        if count > self.policy.max_attempts:
            return None
        delay = self.policy.initial_delay_s * (2 ** (count - 1))
        return min(self.policy.max_delay_s, delay)

    def reset(self) -> None:
        """Clear crash history and bump `generation` (H4), so any
        already-scheduled `_restart_after_delay` from a crash that predates
        this reset is provably stale by the generation counter itself,
        rather than by an incidental registry-state check at the call
        site."""
        self._crash_times.clear()
        self.generation += 1


class EngineProcess:
    """One spawned `mtplx serve` child, running as a supervisor-owned engine."""

    def __init__(
        self,
        model_path: str | Path,
        port: int,
        extra_args: list[str] | None = None,
        *,
        python: str = sys.executable,
        env: dict[str, str] | None = None,
        argv_override: list[str] | None = None,
    ) -> None:
        self.model_path = model_path
        self.port = port
        self.extra_args = list(extra_args or [])
        self.python = python
        self.env = env
        # Test-only hook: replaces the whole argv, e.g. to spawn the fake
        # engine fixture instead of the real `mtplx serve`.
        self._argv_override = argv_override
        self._proc: subprocess.Popen[str] | None = None
        self._stderr_tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        self._reader_thread: threading.Thread | None = None
        self.last_ready_at: float | None = None
        self.unresponsive_since: float | None = None

    @property
    def argv(self) -> list[str]:
        if self._argv_override is not None:
            return list(self._argv_override)
        return [
            self.python,
            "-m",
            "mtplx.cli",
            "serve",
            "--model",
            str(self.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--no-auth",
            "--yes",
            *self.extra_args,
        ]

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def spawn(self) -> None:
        """Start the child. Stdout is discarded; stderr is drained on a
        background thread into a bounded tail so the pipe never fills.

        Spawned with ``start_new_session=True`` so the child (and anything
        it spawns) lands in its own process group, letting ``terminate()``
        kill the whole group by pgid instead of just this one pid. The env
        is always scrubbed of secret-shaped names (H2); ``self.env`` (if
        given) is applied on top as explicit overrides."""
        self._proc = subprocess.Popen(
            self.argv,
            env=_scrubbed_env(self.env),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        self._reader_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._reader_thread.start()

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                self._stderr_tail.append(line.rstrip("\n"))
        except (OSError, ValueError):
            pass

    def is_alive(self) -> bool:
        if self._proc is None:
            return False
        return self._proc.poll() is None

    def exit_code(self) -> int | None:
        if self._proc is None:
            return None
        return self._proc.poll()

    def probe(self, timeout_s: float = 1.5) -> Liveness:
        """Process gone -> GONE. Else try a TCP connect: refused -> PORT_CLOSED.
        Else GET /health: `ok: true` -> READY, timeout or non-ok -> BUSY (per
        the Swift busy-vs-dead policy, a probe timeout alone is never death)."""
        now = time.monotonic()
        if not self.is_alive():
            self.unresponsive_since = None
            return Liveness.GONE
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=timeout_s):
                pass
        except OSError:
            if self.unresponsive_since is None:
                self.unresponsive_since = now
            return Liveness.PORT_CLOSED
        if self._health_ok(timeout_s):
            self.last_ready_at = now
            self.unresponsive_since = None
            return Liveness.READY
        if self.unresponsive_since is None:
            self.unresponsive_since = now
        return Liveness.BUSY

    def _health_ok(self, timeout_s: float) -> bool:
        """`ok: true` alone is not enough: a payload carrying an explicit
        not-yet-ready warmup signal (`_warmup_ready`, see module docstring)
        keeps this False -- and the caller's probe() reports BUSY, not
        READY -- until the engine reports warmup complete."""
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout_s)
        try:
            conn.request("GET", "/health")
            response = conn.getresponse()
            body = response.read()
            if response.status != 200:
                return False
            payload = json.loads(body.decode("utf-8"))
            if not isinstance(payload, dict) or payload.get("ok") is not True:
                return False
            return _warmup_ready(payload)
        except (OSError, ValueError):
            return False
        finally:
            conn.close()

    def unresponsive_for_s(self, now: float) -> float:
        """Seconds of unbroken silence since the last READY probe, or 0.0 if
        currently READY or never unresponsive. A caller reaps once this
        outlasts its own grace period."""
        if self.unresponsive_since is None:
            return 0.0
        return max(0.0, now - self.unresponsive_since)

    def _signal_group(self, sig: int) -> None:
        """Signal the whole process group (H3: the child was spawned with
        start_new_session=True, so its pgid equals its own pid, and this
        also reaches anything the engine itself spawned). Falls back to
        signaling just the one pid if the group is already gone or
        killpg is unavailable (non-POSIX)."""
        proc = self._proc
        if proc is None:
            return
        killpg = getattr(os, "killpg", None)
        if killpg is not None:
            try:
                killpg(proc.pid, sig)
                return
            except (ProcessLookupError, PermissionError, OSError):
                pass
        proc.send_signal(sig)

    def terminate(self, grace_s: float = 10.0) -> int | None:
        """SIGTERM, wait grace_s; SIGINT, wait grace_s/2; SIGKILL, wait
        grace_s/2. Signals the whole process group each step. Idempotent.
        Returns the exit code (None if never spawned)."""
        proc = self._proc
        if proc is None:
            return None
        if proc.poll() is not None:
            return proc.returncode
        self._signal_group(signal.SIGTERM)
        try:
            proc.wait(timeout=max(0.1, grace_s))
            return proc.returncode
        except subprocess.TimeoutExpired:
            pass
        if proc.poll() is not None:
            return proc.returncode
        try:
            self._signal_group(signal.SIGINT)
            proc.wait(timeout=max(0.1, grace_s / 2))
            return proc.returncode
        except (subprocess.TimeoutExpired, ProcessLookupError):
            pass
        if proc.poll() is not None:
            return proc.returncode
        try:
            self._signal_group(signal.SIGKILL)
        except ProcessLookupError:
            return proc.poll()
        try:
            proc.wait(timeout=max(0.1, grace_s / 2))
        except subprocess.TimeoutExpired:
            pass
        return proc.poll()

    def death_reason(self) -> str:
        """"port_in_use" if the stderr tail matches a bind-collision message
        (M4: the caller can retry on a fresh port); else "out_of_memory" if
        it matches a known OOM pattern; else "signal" for a negative return
        code; else "exit"."""
        tail = "\n".join(self._stderr_tail)
        for pattern in _PORT_IN_USE_PATTERNS:
            if pattern.search(tail):
                return "port_in_use"
        for pattern in _OOM_PATTERNS:
            if pattern.search(tail):
                return "out_of_memory"
        code = self.exit_code()
        if code is not None and code < 0:
            return "signal"
        return "exit"
