# Component DAG — Model Supervisor

| Sub-component | Files (exclusive write scope) | Depends on | Wave |
|---|---|---|---|
| registry+budget | `mtplx/supervisor/__init__.py`, `registry.py`, `budget.py`, `tests/test_supervisor_registry.py`, `tests/test_supervisor_budget.py` | — | 1 |
| process | `mtplx/supervisor/process.py`, `tests/fixtures/fake_engine.py`, `tests/test_supervisor_process.py` | — | 1 |
| proxy+admin+service | `mtplx/supervisor/proxy.py`, `admin.py`, `service.py`, `tests/test_supervisor_proxy.py`, `tests/test_supervisor_admin.py` | registry, budget, process | 2 |
| cli+docs | `mtplx/cli.py` (supervise parser), `mtplx/commands/public.py` (`cmd_supervise_public`), `pyproject.toml` (httpx), `docs/server.md` section, `tests/test_supervisor_cli.py` | service | 2 |

## Waves
- Wave 1 (parallel): registry+budget, process. Contracts frozen in DESIGN.md (class and function names below).
- Wave 2 (parallel): proxy+admin+service, cli+docs. cli imports `mtplx.supervisor.service.Supervisor` by the frozen name.
- Wave 3: review + security pass, fixes, full suite, PR.

## Frozen contracts (wave 1 must implement exactly these names)
```python
# registry.py
class EngineState(str, Enum): INSTALLED, LOADING, READY, DRAINING, STOPPED, FAILED
@dataclass class EngineSpec: model_id: str; path: Path; resident_bytes: int
@dataclass class EngineRecord: spec: EngineSpec; state: EngineState; port: int | None; pid: int | None; pins: int; last_used: float; failure_reason: str | None
class EngineRegistry:
    def add(self, spec) -> EngineRecord; def get(self, model_id) -> EngineRecord | None
    def resolve(self, requested: str | None, default_id: str, strict: bool) -> tuple[EngineRecord | None, str]  # (record, "loaded"|"installed"|"fallback"|"unknown")
    def pin(self, model_id); def unpin(self, model_id); def touch(self, model_id)
    def lru_unpinned(self, exclude: set[str]) -> list[EngineRecord]
    def idle(self, now, ttl_s, exclude) -> list[EngineRecord]
    def set_state(self, model_id, state, *, port=None, pid=None, failure_reason=None)
# budget.py
def total_ram_bytes() -> int
def estimate_resident_bytes(path: Path) -> int
@dataclass class Admission: ok: bool; needed_bytes: int; available_bytes: int; would_free: list[str]; reason: str
def admit(registry, spec, budget_bytes, *, evict_to_fit: bool) -> Admission
# process.py
@dataclass class RestartPolicy: max_attempts=3; initial_delay_s=1.0; max_delay_s=30.0; crash_window_s=120.0
class Liveness(str, Enum): READY, BUSY, GONE, PORT_CLOSED, UNRESPONSIVE
class EngineProcess:
    def __init__(self, model_path, port, extra_args, *, python=sys.executable, env=None)
    argv: list[str]; pid: int | None
    def spawn(self); def probe(self, timeout_s=1.5) -> Liveness; def is_alive() -> bool
    def terminate(self, grace_s=10.0); def exit_code() -> int | None
    def death_reason(self) -> str  # "exit"|"signal"|"out_of_memory"
class RestartTracker: def __init__(self, policy); def record_crash(self, now) -> float | None  # next delay or None when exhausted; def reset()
```
