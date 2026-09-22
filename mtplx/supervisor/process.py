"""Engine child process lifecycle: spawn, health probe, terminate, restart backoff.

Generalizes the single-daemon wrapper in `mtplx/commands/public.py`
(`_run_server_child_with_app_parent_watchdog`, `_terminate_server_child`,
`_pid_is_alive`) from one wrapped `mtplx serve` process to N supervisor-owned
engine children, and replicates the macOS app's busy-vs-dead liveness split
(`DaemonLivenessPolicy.swift`, issue #487: a probe timeout alone never means
dead) and restart backoff (`DaemonSupervisor.swift`'s `DaemonRestartPolicy`)
in Python. Health is fetched the same way `daemon_client.fetch_daemon_health`
does: stdlib HTTP, no new dependency.
"""

from __future__ import annotations

import http.client
import json
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
        self._crash_times.clear()


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
        background thread into a bounded tail so the pipe never fills."""
        self._proc = subprocess.Popen(
            self.argv,
            env=self.env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
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
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout_s)
        try:
            conn.request("GET", "/health")
            response = conn.getresponse()
            body = response.read()
            if response.status != 200:
                return False
            payload = json.loads(body.decode("utf-8"))
            return isinstance(payload, dict) and payload.get("ok") is True
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

    def terminate(self, grace_s: float = 10.0) -> int | None:
        """SIGTERM, wait grace_s; SIGINT, wait grace_s/2; SIGKILL, wait
        grace_s/2. Idempotent. Returns the exit code (None if never spawned)."""
        proc = self._proc
        if proc is None:
            return None
        if proc.poll() is not None:
            return proc.returncode
        proc.terminate()
        try:
            proc.wait(timeout=max(0.1, grace_s))
            return proc.returncode
        except subprocess.TimeoutExpired:
            pass
        if proc.poll() is not None:
            return proc.returncode
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=max(0.1, grace_s / 2))
            return proc.returncode
        except (subprocess.TimeoutExpired, ProcessLookupError):
            pass
        if proc.poll() is not None:
            return proc.returncode
        try:
            proc.kill()
        except ProcessLookupError:
            return proc.poll()
        try:
            proc.wait(timeout=max(0.1, grace_s / 2))
        except subprocess.TimeoutExpired:
            pass
        return proc.poll()

    def death_reason(self) -> str:
        """"out_of_memory" if the captured stderr tail matches a known OOM
        pattern; else "signal" for a negative return code; else "exit"."""
        tail = "\n".join(self._stderr_tail)
        for pattern in _OOM_PATTERNS:
            if pattern.search(tail):
                return "out_of_memory"
        code = self.exit_code()
        if code is not None and code < 0:
            return "signal"
        return "exit"
