# SPDX-License-Identifier: Apache-2.0
"""Lifecycle for a Splash engine owned by the bridge.

Splash has no runtime model-swap API: a server serves exactly the package it
was started with. Loading and unloading a model is therefore process
lifecycle, and that is what this class is.

The bridge starts Splash's inner server (`server/server.py`) rather than the
`splash serve` CLI, for one reason: the 1.0 CLI hardcodes port 8000 and takes
an exclusive lock on it, which would collide with the port the bridge itself
serves the MTPLX contract on. The inner server accepts `--port`, so Splash
runs on a private loopback port and the bridge keeps the public one. Every
argument below is the one the CLI would have passed.
"""

from __future__ import annotations

import collections
import json
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from .install import SplashInstall, SplashUnavailable

# Splash prints a memory budget and exits if the model does not fit, but a
# cold 27B on a busy machine can still take a while to map and warm.
DEFAULT_READY_TIMEOUT_S = 900.0
STOP_GRACE_S = 20.0


def free_port() -> int:
    """A loopback port free right now, for the private Splash listener."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _get(url: str, *, timeout: float, api_key: str | None = None) -> tuple[int, bytes]:
    request = urllib.request.Request(url)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as error:
        return int(error.code), error.read()


@dataclass
class EngineState:
    """What the bridge knows about the engine right now."""

    model_id: str
    phase: str = "stopped"  # stopped | installing | starting | ready | failed
    detail: str = ""
    started_at: float | None = None
    ready_at: float | None = None
    pid: int | None = None
    port: int | None = None
    error: str | None = None
    logs: collections.deque = field(default_factory=lambda: collections.deque(maxlen=400))


class SplashEngine:
    """Owns one `splash` engine process and the health view of it."""

    def __init__(
        self,
        install: SplashInstall,
        model_id: str,
        *,
        port: int | None = None,
        max_memory: str | None = None,
        max_context: str | None = None,
        api_key: str | None = None,
        ready_timeout_s: float = DEFAULT_READY_TIMEOUT_S,
        on_log: Callable[[str], None] | None = None,
    ) -> None:
        self.install = install
        self.model_id = model_id
        self.port = port or free_port()
        # Callers hand these over however they parsed them: the MTPLX CLI
        # types --context-window as an int, and an int in argv fails both the
        # log line's join and Popen itself. This class owns the command line,
        # so it normalizes here rather than trusting every caller to.
        self.max_memory = self._argv_value(max_memory)
        self.max_context = self._argv_value(max_context)
        self.api_key = api_key
        self.ready_timeout_s = ready_timeout_s
        self._on_log = on_log
        self._process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._lock = threading.RLock()
        self.state = EngineState(model_id=model_id, port=self.port)

    @staticmethod
    def _argv_value(value: Any) -> str:
        """A Splash size/limit argument as text; unset or zero means auto."""
        if value is None or value == "" or value == 0:
            return "auto"
        return str(value)

    # -- addresses --------------------------------------------------------

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Download if needed, spawn the engine, and block until it is ready."""
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            if not self.install.is_installed(self.model_id):
                self._set(phase="installing", detail=f"downloading {self.model_id}")
                self.install.prepare(self.model_id, on_line=self._log)
            self._spawn()
        self._await_ready()

    def _spawn(self) -> None:
        root = self.install.package_root(self.model_id)
        command = [
            str(self.install.python),
            "-u",
            str(self.install.server_script),
            str(root / "target"),
            str(root / "draft"),
            "--tokenizer",
            str(root / "tokenizer"),
            "--model",
            self.model_id,
            "--binary",
            str(self.install.engine_binary),
            "--max-memory",
            self.max_memory,
            "--max-context",
            self.max_context,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            # The bridge serves the chat UI; Splash's own page would be a
            # second, divergent one on a port nobody is told about.
            "--no-webui",
        ]
        environment = dict(
            os.environ, PYTHONUNBUFFERED="1", TRANSFORMERS_VERBOSITY="error"
        )
        if self.api_key:
            environment["SPLASH_API_KEY"] = self.api_key
        self._set(phase="starting", detail="starting engine", error=None)
        self._log("$ " + " ".join(command))
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=environment,
                # Its own group, so stopping takes the engine child with it.
                start_new_session=True,
            )
        except OSError as error:
            self._set(phase="failed", error=f"could not start Splash: {error}")
            raise SplashUnavailable(f"could not start Splash: {error}") from error
        self.state.pid = self._process.pid
        self.state.started_at = time.time()
        self._reader = threading.Thread(
            target=self._drain, name="splash-logs", daemon=True
        )
        self._reader.start()

    def _drain(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            self._log(line.rstrip())

    def _await_ready(self) -> None:
        deadline = time.monotonic() + self.ready_timeout_s
        while time.monotonic() < deadline:
            process = self._process
            if process is None:
                raise SplashUnavailable("Splash was stopped before it became ready")
            code = process.poll()
            if code is not None:
                tail = "\n".join(list(self.state.logs)[-15:])
                self._set(phase="failed", error=f"Splash exited with code {code}")
                raise SplashUnavailable(
                    f"Splash exited with code {code} before becoming ready.\n{tail}"
                )
            try:
                status, _ = _get(
                    f"{self.base_url}/ready", timeout=3.0, api_key=self.api_key
                )
            except (OSError, urllib.error.URLError):
                status = 0
            if status == 200:
                self.state.ready_at = time.time()
                self._set(phase="ready", detail=f"serving {self.model_id}", error=None)
                return
            time.sleep(0.5)
        self.stop()
        raise SplashUnavailable(
            f"Splash did not become ready within {self.ready_timeout_s:.0f}s"
        )

    def stop(self) -> None:
        """Stop the engine and free its unified memory."""
        with self._lock:
            process = self._process
            self._process = None
            if process is None:
                self._set(phase="stopped", detail="", error=None)
                return
            if process.poll() is None:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    process.terminate()
                try:
                    process.wait(timeout=STOP_GRACE_S)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        process.kill()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            self.state.pid = None
            self.state.ready_at = None
            self._set(phase="stopped", detail="", error=None)

    def is_running(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None

    def is_ready(self) -> bool:
        return self.state.phase == "ready" and self.is_running()

    # -- engine telemetry -------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Splash's own /status, or {} when it is not answering."""
        if not self.is_running():
            return {}
        try:
            code, body = _get(
                f"{self.base_url}/status", timeout=3.0, api_key=self.api_key
            )
        except (OSError, urllib.error.URLError):
            return {}
        if code != 200:
            return {}
        try:
            parsed = json.loads(body)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def metrics_text(self) -> str:
        if not self.is_running():
            return ""
        try:
            code, body = _get(
                f"{self.base_url}/metrics", timeout=3.0, api_key=self.api_key
            )
        except (OSError, urllib.error.URLError):
            return ""
        return body.decode("utf-8", "replace") if code == 200 else ""

    # -- internals --------------------------------------------------------

    def _set(self, **fields: Any) -> None:
        for key, value in fields.items():
            setattr(self.state, key, value)

    def _log(self, line: str) -> None:
        if not line:
            return
        self.state.logs.append(line)
        if self._on_log is not None:
            self._on_log(line)
