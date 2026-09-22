"""``Supervisor``: wires registry + budget + process management into one
ASGI app, and ``run_supervisor``, the blocking uvicorn entry point.

Spawning and probing (``EngineProcess.spawn``/``probe``) are blocking
calls; every use here runs them in ``asyncio.to_thread`` so the event loop
serving other requests never stalls on a subprocess call.
"""

from __future__ import annotations

import asyncio
import logging
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

            if self.config.engine_argv_override:
                argv = _substitute_argv(self.config.engine_argv_override, port, spec.path)
                proc = EngineProcess(spec.path, port, self.config.engine_extra_args, argv_override=argv)
            else:
                proc = EngineProcess(spec.path, port, self.config.engine_extra_args)
            self._processes[model_id] = proc
            self._trackers.setdefault(model_id, RestartTracker(RestartPolicy()))

            await asyncio.to_thread(proc.spawn)

            deadline = time.monotonic() + self.config.load_timeout_s
            while True:
                liveness = await asyncio.to_thread(proc.probe, 2.0)
                if liveness == Liveness.READY:
                    self.registry.set_state(model_id, EngineState.READY, pid=proc.pid)
                    self.registry.touch(model_id)
                    return self.registry.get(model_id)
                if liveness == Liveness.GONE:
                    reason = proc.death_reason()
                    self.registry.set_state(model_id, EngineState.FAILED, failure_reason=reason)
                    raise EngineUnavailable(reason)
                if time.monotonic() >= deadline:
                    self.registry.set_state(model_id, EngineState.FAILED, failure_reason="load_timeout")
                    raise EngineUnavailable("load_timeout")
                await asyncio.sleep(0.2)

    async def unload(self, model_id: str) -> None:
        await self._unload(model_id)

    async def _unload(self, model_id: str, *, drain_s: float | None = None) -> None:
        record = self.registry.get(model_id)
        if record is None:
            return
        self.registry.set_state(model_id, EngineState.DRAINING)
        wait_s = self.config.drain_s if drain_s is None else drain_s
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            current = self.registry.get(model_id)
            if current is None or current.pins == 0:
                break
            await asyncio.sleep(0.05)
        proc = self._processes.get(model_id)
        if proc is not None:
            await asyncio.to_thread(proc.terminate, 10.0)
        self.registry.set_state(model_id, EngineState.STOPPED)

    async def restart(self, model_id: str) -> EngineRecord:
        record = self.registry.get(model_id)
        if record is None:
            raise ModelNotFound(model_id)
        proc = self._processes.get(model_id)
        if proc is not None and proc.is_alive():
            await asyncio.to_thread(proc.terminate, 10.0)
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
            liveness = await asyncio.to_thread(proc.probe, 1.5)
            now = time.monotonic()
            if liveness in (Liveness.GONE, Liveness.PORT_CLOSED):
                await self._reap(model_id, proc)
            elif liveness == Liveness.BUSY and proc.unresponsive_for_s(now) >= self.config.unresponsive_grace_s:
                await self._reap(model_id, proc)

    async def _reap(self, model_id: str, proc: EngineProcess) -> None:
        record = self.registry.get(model_id)
        if record is None or record.state not in (EngineState.READY, EngineState.LOADING):
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
