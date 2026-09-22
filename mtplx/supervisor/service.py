"""``Supervisor``: wires registry + budget + process management into one
ASGI app, and ``run_supervisor``, the blocking uvicorn entry point.

Spawning and probing (``EngineProcess.spawn``/``probe``) are blocking
calls; every use here runs them in ``asyncio.to_thread`` so the event loop
serving other requests never stalls on a subprocess call.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import uvicorn

from .admin import AdminApp
from .budget import GIB, admit, estimate_resident_bytes, total_ram_bytes
from .process import EngineProcess, Liveness, RestartPolicy, RestartTracker
from .proxy import EngineUnavailable, InsufficientMemory, ModelNotFound, ProxyApp
from .registry import EngineRecord, EngineRegistry, EngineSpec, EngineState

logger = logging.getLogger("mtplx.supervisor")

_HEALTH_SWEEP_INTERVAL_S = 2.0
_IDLE_SWEEP_INTERVAL_S = 30.0

# M4: _free_port()'s bind-then-close probe has a TOCTOU window -- two
# concurrent JIT loads can be handed the same "free" port before either
# child binds it. Retry the spawn on a fresh port a bounded number of times
# rather than surfacing an opaque load_timeout/GONE to the caller.
_MAX_PORT_COLLISION_RETRIES = 3

__all__ = ["SupervisorConfig", "Supervisor", "run_supervisor"]


@dataclass
class SupervisorConfig:
    """Everything ``mtplx supervise`` needs to start the front door.

    Frozen interface (see ``docs/features/model-supervisor/COMPONENT_DAG.md``):
    the CLI codes against exactly these fields.
    """

    host: str = "127.0.0.1"
    port: int = 8000
    model_paths: list[Path] = field(default_factory=list)
    preload: list[str] = field(default_factory=list)
    default_model_id: str | None = None
    api_key: str | None = None
    budget_bytes: int | None = None
    idle_ttl_s: float = 1800.0
    load_timeout_s: float = 600.0
    unresponsive_grace_s: float = 90.0
    drain_s: float = 15.0
    evict_to_fit: bool = False
    strict_model: bool = False
    insecure_lan: bool = False
    unload_default: bool = False
    engine_extra_args: list[str] = field(default_factory=list)
    launch_id: str | None = None
    engine_argv_override: list[str] | None = None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _substitute_argv(template: list[str], port: int, model_path: Path) -> list[str]:
    return [arg.replace("{port}", str(port)).replace("{model}", str(model_path)) for arg in template]


class Supervisor:
    """Owns the registry, the budget, every ``EngineProcess``, and the
    background health/idle sweep loops. Also the ASGI app itself: it
    handles the ``lifespan`` scope directly and dispatches ``http`` scopes
    to the admin or proxy sub-apps."""

    def __init__(
        self,
        config: SupervisorConfig,
        *,
        health_sweep_interval_s: float = _HEALTH_SWEEP_INTERVAL_S,
        idle_sweep_interval_s: float = _IDLE_SWEEP_INTERVAL_S,
    ) -> None:
        self.config = config
        self.registry = EngineRegistry()
        self.budget_bytes = (
            config.budget_bytes
            if config.budget_bytes is not None
            else int(total_ram_bytes() * 0.75) - 6 * GIB
        )
        for path in config.model_paths:
            path = Path(path)
            spec = EngineSpec(
                model_id=path.name,
                path=path,
                resident_bytes=estimate_resident_bytes(path),
            )
            self.registry.add(spec)

        self.default_model_id = self._resolve_default_id()

        self._health_sweep_interval_s = health_sweep_interval_s
        self._idle_sweep_interval_s = idle_sweep_interval_s

        self._processes: dict[str, EngineProcess] = {}
        self._trackers: dict[str, RestartTracker] = {}
        self._load_locks: dict[str, asyncio.Lock] = {}
        self._background_tasks: set[asyncio.Task] = set()

        self.http_client: httpx.AsyncClient | None = None
        self._health_task: asyncio.Task | None = None
        self._idle_task: asyncio.Task | None = None

        self._admin_app = AdminApp(self)
        self._proxy_app = ProxyApp(self)
        self.app = self

        # H3: orphan protection for the unclean-shutdown path (SIGKILL/OOM
        # bypasses lifespan shutdown entirely, so stop() never runs). Each
        # EngineProcess is also spawned with MTPLX_APP_PARENT_PID set to our
        # own pid (see process.py's module docstring), which gives a second,
        # independent line of defense: the engine's own `mtplx serve`
        # watchdog tears itself down if this process disappears. This hook
        # is belt-and-suspenders and is idempotent (guarded by
        # ``_atexit_done``; ``EngineProcess.terminate`` is idempotent too).
        self._atexit_done = False
        atexit.register(self._atexit_cleanup)

    def _resolve_default_id(self) -> str | None:
        if self.config.default_model_id:
            return self.config.default_model_id
        if self.config.preload:
            return self.config.preload[0]
        if self.config.model_paths:
            return Path(self.config.model_paths[0]).name
        return None

    # -- ASGI entry point -----------------------------------------------

    async def __call__(self, scope, receive, send) -> None:
        scope_type = scope.get("type")
        if scope_type == "lifespan":
            await self._handle_lifespan(receive, send)
            return
        if scope_type != "http":
            return
        path = str(scope.get("path") or "/")
        if path.startswith("/mtplx/admin") or path in ("/health", "/v1/models"):
            await self._admin_app(scope, receive, send)
        else:
            await self._proxy_app(scope, receive, send)

    async def _handle_lifespan(self, receive, send) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                try:
                    await self.start()
                except Exception as exc:  # noqa: BLE001 - report to the server
                    await send({"type": "lifespan.startup.failed", "message": str(exc)})
                    return
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                try:
                    await self.stop()
                except Exception as exc:  # noqa: BLE001
                    await send({"type": "lifespan.shutdown.failed", "message": str(exc)})
                    return
                await send({"type": "lifespan.shutdown.complete"})
                return

    # -- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        if self.config.insecure_lan:
            logger.warning(
                "insecure-lan: inference routes accept unauthenticated requests "
                "from any host that can reach %s:%s",
                self.config.host,
                self.config.port,
            )
        self.http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0, read=None))
        preload_ids = self.config.preload or (
            [self.default_model_id] if self.default_model_id else []
        )
        for model_id in preload_ids:
            if self.registry.get(model_id) is None:
                logger.warning("preload model %s is not a registered pack", model_id)
                continue
            try:
                await self.ensure_loaded(model_id)
            except (InsufficientMemory, EngineUnavailable, ModelNotFound) as exc:
                logger.error("preload of %s failed: %s", model_id, exc)
        self._health_task = asyncio.create_task(self._health_sweep_loop())
        self._idle_task = asyncio.create_task(self._idle_sweep_loop())

    async def stop(self) -> None:
        for task in (self._health_task, self._idle_task):
            if task is not None:
                task.cancel()
        for task in list(self._background_tasks):
            task.cancel()
        deadline = time.monotonic() + self.config.drain_s
        for record in self.registry.records():
            if record.state in (EngineState.READY, EngineState.LOADING):
                remaining = max(0.0, deadline - time.monotonic())
                await self._unload(record.spec.model_id, drain_s=remaining)
        if self.http_client is not None:
            await self.http_client.aclose()
            self.http_client = None
        self._atexit_done = True

    def _atexit_cleanup(self) -> None:
        """Synchronous, idempotent last resort: terminate every spawned
        engine's process group. Runs on normal interpreter exit (including
        an uncaught exception), not on SIGKILL -- see the class docstring
        note above and process.py's MTPLX_APP_PARENT_PID mechanism for the
        SIGKILL case."""
        if self._atexit_done:
            return
        self._atexit_done = True
        for model_id, proc in list(self._processes.items()):
            try:
                proc.terminate(grace_s=5.0)
            except Exception:  # noqa: BLE001 - best-effort cleanup
                logger.exception("atexit cleanup failed to terminate engine %s", model_id)

    def track_background(self, task: asyncio.Task) -> None:
        """Keep a strong reference to a fire-and-forget task and log if it fails."""
        self._background_tasks.add(task)

        def _done(t: asyncio.Task) -> None:
            self._background_tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None and not isinstance(exc, (InsufficientMemory, EngineUnavailable, ModelNotFound)):
                logger.exception("background task failed", exc_info=exc)

        task.add_done_callback(_done)

    # -- loading ------------------------------------------------------------

    async def ensure_loaded(self, model_id: str) -> EngineRecord:
        """Idempotent, concurrency-safe JIT load. Concurrent callers for the
        same model id share one asyncio.Lock and await the same outcome."""
        record = self.registry.get(model_id)
        if record is None:
            raise ModelNotFound(model_id)
        if record.state == EngineState.READY:
            return record

        lock = self._load_locks.setdefault(model_id, asyncio.Lock())
        async with lock:
            record = self.registry.get(model_id)
            if record is None:
                raise ModelNotFound(model_id)
            if record.state == EngineState.READY:
                return record

            spec = record.spec
            admission = admit(
                self.registry, spec, self.budget_bytes, evict_to_fit=self.config.evict_to_fit
            )
            if not admission.ok:
                raise InsufficientMemory(admission)
            for victim_id in admission.would_free:
                await self._unload(victim_id)

            port = _free_port()
            self.registry.set_state(model_id, EngineState.LOADING, port=port)
            proc = self._spawn_engine(model_id, spec, port)
            self._trackers.setdefault(model_id, RestartTracker(RestartPolicy()))
            await asyncio.to_thread(proc.spawn)

            deadline = time.monotonic() + self.config.load_timeout_s
            port_collision_attempts = 0
            while True:
                liveness = await asyncio.to_thread(proc.probe, 2.0)
                if liveness == Liveness.READY:
                    self.registry.set_state(model_id, EngineState.READY, pid=proc.pid)
                    self.registry.touch(model_id)
                    await self._register_engine_alias(model_id)
                    return self.registry.get(model_id)
                if liveness == Liveness.GONE:
                    reason = proc.death_reason()
                    if (
                        reason == "port_in_use"
                        and port_collision_attempts < _MAX_PORT_COLLISION_RETRIES
                    ):
                        port_collision_attempts += 1
                        logger.warning(
                            "port collision loading %s (attempt %d/%d); retrying on a fresh port",
                            model_id,
                            port_collision_attempts,
                            _MAX_PORT_COLLISION_RETRIES,
                        )
                        port = _free_port()
                        self.registry.set_state(model_id, EngineState.LOADING, port=port)
                        proc = self._spawn_engine(model_id, spec, port)
                        await asyncio.to_thread(proc.spawn)
                        deadline = time.monotonic() + self.config.load_timeout_s
                        continue
                    self.registry.set_state(model_id, EngineState.FAILED, failure_reason=reason)
                    raise EngineUnavailable(reason)
                if time.monotonic() >= deadline:
                    self.registry.set_state(model_id, EngineState.FAILED, failure_reason="load_timeout")
                    raise EngineUnavailable("load_timeout")
                await asyncio.sleep(0.2)

    def _spawn_engine(self, model_id: str, spec: EngineSpec, port: int) -> EngineProcess:
        """Build (but do not yet spawn) the EngineProcess for `model_id` on
        `port`, wiring in the H3 parent-pid env so the child can self-clean
        if this supervisor disappears (see process.py's module docstring)."""
        env_overrides = {
            "MTPLX_APP_PARENT_PID": str(os.getpid()),
            "MTPLX_SUPERVISOR_PID": str(os.getpid()),
        }
        if self.config.engine_argv_override:
            argv = _substitute_argv(self.config.engine_argv_override, port, spec.path)
            proc = EngineProcess(
                spec.path,
                port,
                self.config.engine_extra_args,
                env=env_overrides,
                argv_override=argv,
            )
        else:
            proc = EngineProcess(
                spec.path, port, self.config.engine_extra_args, env=env_overrides
            )
        self._processes[model_id] = proc
        return proc

    async def _register_engine_alias(self, model_id: str) -> None:
        """R1: an engine answers with its own served model id (first entry
        of its own `/v1/models`), which can differ from the pack directory
        name the supervisor routes by. Register that id as an alias so
        clients can send either. Best-effort: any failure just means no
        alias, never a failed load."""
        proc = self._processes.get(model_id)
        record = self.registry.get(model_id)
        if proc is None or record is None or self.http_client is None:
            return
        try:
            response = await self.http_client.get(
                f"http://127.0.0.1:{proc.port}/v1/models", timeout=5.0
            )
            response.raise_for_status()
            data = response.json().get("data") or []
            if data and isinstance(data[0], dict):
                served_id = data[0].get("id")
                if isinstance(served_id, str) and served_id.strip():
                    self.registry.add_alias(model_id, served_id.strip())
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
            logger.warning("could not register alias for %s: %s", model_id, exc)

    async def unload(self, model_id: str) -> None:
        await self._unload(model_id)

    async def _unload(self, model_id: str, *, drain_s: float | None = None) -> None:
        """C1: shares the per-model load lock with ensure_loaded/restart so
        a JIT load for the same model id can never interleave with a drain
        in progress -- it blocks until the drain finishes, then loads a
        fresh engine against the now-STOPPED record. `proc` is captured
        before the drain wait (not after), so termination always targets
        the engine this call actually meant to drain."""
        lock = self._load_locks.setdefault(model_id, asyncio.Lock())
        async with lock:
            record = self.registry.get(model_id)
            if record is None or record.state not in (EngineState.READY, EngineState.LOADING):
                return
            proc = self._processes.get(model_id)
            self.registry.set_state(model_id, EngineState.DRAINING)
            wait_s = self.config.drain_s if drain_s is None else drain_s
            deadline = time.monotonic() + wait_s
            while time.monotonic() < deadline:
                current = self.registry.get(model_id)
                if current is None or current.pins == 0:
                    break
                await asyncio.sleep(0.05)
            if proc is not None:
                await asyncio.to_thread(proc.terminate, 10.0)
                if self._processes.get(model_id) is proc:
                    del self._processes[model_id]
            self.registry.set_state(model_id, EngineState.STOPPED)

    async def restart(self, model_id: str) -> EngineRecord:
        """H3/H4: the terminate-and-reset critical section runs under the
        same per-model load lock as ensure_loaded/_unload, so a client
        request racing an admin restart can never observe READY, get
        pinned, and then have its engine pulled out from under it. The lock
        is released before calling ensure_loaded (an asyncio.Lock is not
        reentrant); ensure_loaded's own locking still makes the reload
        itself concurrency-safe against any other caller."""
        record = self.registry.get(model_id)
        if record is None:
            raise ModelNotFound(model_id)
        lock = self._load_locks.setdefault(model_id, asyncio.Lock())
        async with lock:
            record = self.registry.get(model_id)
            if record is None:
                raise ModelNotFound(model_id)
            proc = self._processes.get(model_id)
            if proc is not None and proc.is_alive():
                await asyncio.to_thread(proc.terminate, 10.0)
            if self._processes.get(model_id) is proc:
                self._processes.pop(model_id, None)
            tracker = self._trackers.get(model_id)
            if tracker is not None:
                tracker.reset()
            self.registry.set_state(model_id, EngineState.INSTALLED)
        return await self.ensure_loaded(model_id)

    # -- background sweeps --------------------------------------------------

    async def _health_sweep_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._health_sweep_interval_s)
                await self._health_sweep_once()
        except asyncio.CancelledError:
            return

    async def _health_sweep_once(self) -> None:
        """L2: probes run concurrently (asyncio.gather over to_thread), so
        detection latency for a wedged/dead engine no longer grows linearly
        with the number of supervised engines. Reaping stays sequential
        (cheap, and each reap takes its own per-model lock)."""
        targets: list[tuple[str, EngineProcess]] = []
        for record in self.registry.records():
            if record.state not in (EngineState.READY, EngineState.LOADING):
                continue
            model_id = record.spec.model_id
            lock = self._load_locks.get(model_id)
            if record.state == EngineState.LOADING and lock is not None and lock.locked():
                # ensure_loaded owns probing and timeout for this engine.
                continue
            proc = self._processes.get(model_id)
            if proc is None:
                continue
            targets.append((model_id, proc))
        if not targets:
            return

        async def _probe(model_id: str, proc: EngineProcess) -> tuple[str, EngineProcess, Liveness]:
            liveness = await asyncio.to_thread(proc.probe, 1.5)
            return model_id, proc, liveness

        results = await asyncio.gather(*(_probe(m, p) for m, p in targets), return_exceptions=True)
        now = time.monotonic()
        for result in results:
            if isinstance(result, BaseException):
                logger.exception("health probe failed", exc_info=result)
                continue
            model_id, proc, liveness = result
            if liveness in (Liveness.GONE, Liveness.PORT_CLOSED):
                await self._reap(model_id, proc)
            elif liveness == Liveness.BUSY and proc.unresponsive_for_s(now) >= self.config.unresponsive_grace_s:
                await self._reap(model_id, proc)

    async def _reap(self, model_id: str, proc: EngineProcess) -> None:
        """C1: takes the same per-model load lock as ensure_loaded/_unload/
        restart, and re-checks that `proc` is still the tracked process for
        this model id once the lock is held -- a concurrent load or restart
        may have already superseded it, in which case this reap is stale
        and does nothing."""
        lock = self._load_locks.setdefault(model_id, asyncio.Lock())
        async with lock:
            record = self.registry.get(model_id)
            if record is None or record.state not in (EngineState.READY, EngineState.LOADING):
                return
            if self._processes.get(model_id) is not proc:
                return
            reason = proc.death_reason()
            if reason == "out_of_memory":
                self.registry.set_state(model_id, EngineState.FAILED, failure_reason="out_of_memory")
                return
            tracker = self._trackers.setdefault(model_id, RestartTracker(RestartPolicy()))
            now = time.monotonic()
            delay = tracker.record_crash(now)
            if delay is None:
                self.registry.set_state(model_id, EngineState.FAILED, failure_reason="crash_loop")
                return
            self.registry.set_state(model_id, EngineState.INSTALLED, failure_reason="crash: restart scheduled")
            generation = tracker.generation
        task = asyncio.ensure_future(self._restart_after_delay(model_id, delay, generation, tracker))
        self.track_background(task)

    async def _restart_after_delay(
        self, model_id: str, delay: float, generation: int, tracker: RestartTracker
    ) -> None:
        await asyncio.sleep(delay)
        if tracker.generation != generation:
            return  # a newer crash or a reset happened; this attempt is stale
        record = self.registry.get(model_id)
        if record is None or record.state != EngineState.INSTALLED:
            return
        try:
            await self.ensure_loaded(model_id)
        except (InsufficientMemory, EngineUnavailable, ModelNotFound) as exc:
            logger.error("scheduled restart of %s failed: %s", model_id, exc)

    async def _idle_sweep_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._idle_sweep_interval_s)
                await self._idle_sweep_once()
        except asyncio.CancelledError:
            return

    async def _idle_sweep_once(self) -> None:
        exclude: set[str] = set()
        if not self.config.unload_default and self.default_model_id:
            exclude = {self.default_model_id}
        for record in self.registry.idle(time.time(), self.config.idle_ttl_s, exclude):
            await self._unload(record.spec.model_id)

    # -- introspection --------------------------------------------------------

    def snapshot(self) -> dict:
        records = self.registry.records()
        used = sum(
            r.spec.resident_bytes
            for r in records
            if r.state in (EngineState.READY, EngineState.LOADING, EngineState.DRAINING)
        )
        engines = [
            {
                "model": r.spec.model_id,
                "state": r.state.value,
                "port": r.port,
                "pid": r.pid,
                "pins": r.pins,
                "last_used": r.last_used,
                "failure_reason": r.failure_reason,
            }
            for r in records
        ]
        return {
            "engines": engines,
            "budget": {
                "total": self.budget_bytes,
                "used": used,
                "available": self.budget_bytes - used,
            },
        }


def run_supervisor(config: SupervisorConfig) -> int:
    """Blocking entry point: run uvicorn with lifespan-driven start/stop.
    SIGTERM/SIGINT are handled by uvicorn's own default signal handlers."""
    supervisor = Supervisor(config)
    uv_config = uvicorn.Config(
        supervisor.app,
        host=config.host,
        port=config.port,
        lifespan="on",
        log_level="info",
    )
    server = uvicorn.Server(uv_config)
    try:
        server.run()
    except Exception:
        logger.exception("supervisor crashed")
        return 1
    return 0
