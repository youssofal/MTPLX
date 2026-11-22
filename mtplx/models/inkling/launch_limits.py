"""Thread groups this GPU can run, kernel pipeline by kernel pipeline (issue #706).

MLX checks a launch's thread group against the compiled pipeline's own
``maxTotalThreadsPerThreadgroup`` when it encodes the launch, and raises
``ValueError: Thread group size (N) is greater than the maximum allowed threads
per threadgroup (M)``. That limit is not a property of the device: it depends on
the pipeline's register use and on the GPU family. On an M1 Max the copy lane's
25-row attention partials (800 threads) are limited to 704, while the M5 runs
them at 1,024. MLX evaluates lazily, so the error used to surface at some later
``mx.eval``, far from the launch, and stopped the model from loading.

Every Inkling kernel is made through ``kernel()`` here. The first launch of
more than ``PROBE_ABOVE`` threads of each pipeline (kernel name, template,
input and output types, thread group) in a process is evaluated at the call,
on the call's own operands, as issue #400's fix proves a kernel by dispatching
the real pipeline: its result is the launch's result, so the check costs no
extra launch, only that first call's wait. A launch whose group this GPU cannot
run raises ``GroupRefused`` at the call itself, before anything lazy depends on
it, so the stage that made it serves its reference route for that shape
instead (counted ``fallback.<stage>.threadgroup``). Where a caller has a form
with the same bits in smaller groups (``Kernel.runs``), it takes that form
instead (counted ``fused.split.<kernel>.<group>``). Either is printed once and
listed by ``report()`` in /health and ``mtplx doctor``. Only MLX's thread-group
error is caught; every other error still raises.

``MTPLX_INKLING_TEST_THREADGROUP_LIMIT`` (tests and replays only) makes launches
above a limit refused without dispatching them, so these paths run on a GPU that
has no such limit: ``704`` for every kernel, or ``name=704`` pairs (comma
separated, a kernel-name prefix each) for a GPU known kernel by kernel, such as
the M1 Max of issue #706 (``mtplx_inkling_fast_attend_partials=704``).
"""

from __future__ import annotations

import os
import re
import threading
from typing import Any

import mlx.core as mx

#: Launches of this many threads or fewer are not checked (checking every launch would add a wait at the first
#: use of each of several hundred launch shapes). No Inkling kernel of 256 threads or fewer has been refused on
#: any Mac on record: the M1 Max in issue #706 refused 800 threads against a limit of 704 and ran the 576- and
#: 1,024-thread kernels of its first warm-up; GitHub's M1 runner refuses only the attention partials above 704
#: (768 for one float16 form) and the one-launch window attention's 1,024; the M1 runner job and the M4 gate
#: dispatch every kernel at its real widths.
PROBE_ABOVE = 256
TEST_LIMIT_ENV = "MTPLX_INKLING_TEST_THREADGROUP_LIMIT"

_ERROR = re.compile(
    r"Thread group size \((\d+)\) is greater than\s+the maximum allowed threads per threadgroup \((\d+)\)"
)

_lock = threading.Lock()
#: (kernel, group, template, input and output types) -> this pipeline's limit when it refuses the group, else 0.
_verdicts: dict[tuple, int] = {}
#: (kernel, group) -> what was refused and what served instead, for report() and the one printed line.
_refused: dict[tuple, dict[str, Any]] = {}
#: What serves a launch ``check`` refuses: the stage that made it catches GroupRefused.
STAGE_ROUTE = "its stage's reference route serves that shape"
_probes = {"count": 0}


class GroupRefused(RuntimeError):
    """A launch whose thread group this GPU cannot run for this kernel's pipeline: the stage that made it
    serves its reference route for this shape instead."""

    def __init__(self, kernel: str, group: int, limit: int, source: str):
        self.kernel, self.group, self.limit, self.source = kernel, int(group), int(limit), source
        super().__init__(
            f"{kernel}: a {group}-thread group is above what this GPU runs for that pipeline ({limit}, {source})"
        )


def test_limit(name: str | None = None) -> int | None:
    """The test hook's limit for kernel ``name`` (any kernel when None), or None; read on each check, so a test
    can set it. A malformed value raises: a hook that silently does nothing would prove nothing."""
    raw = os.environ.get(TEST_LIMIT_ENV, "").strip()
    if not raw:
        return None
    every, by_prefix = None, []
    for part in raw.split(","):
        prefix, _, value = part.strip().rpartition("=")
        if not value.isdigit() or int(value) <= 0:
            raise ValueError(f"{TEST_LIMIT_ENV}={raw!r}: expected a thread count or name=count pairs")
        if prefix:
            by_prefix.append((prefix, int(value)))
        else:
            every = int(value)
    if name is None:
        return min([value for _, value in by_prefix] + ([every] if every else []))
    for prefix, value in by_prefix:
        if name.startswith(prefix):
            return value
    return every


def _group(threadgroup) -> int:
    total = 1
    for value in tuple(threadgroup or (1,)):
        total *= int(value)
    return total


def _pipeline_key(name: str, group: int, kw: dict) -> tuple:
    """What makes a compiled pipeline: MLX builds one per kernel, template and input and output types (its
    pipeline names carry them), and the thread-group limit belongs to that pipeline."""
    template = tuple((str(k), str(v)) for k, v in (kw.get("template") or ()))
    inputs = tuple((str(a.dtype), a.ndim) if isinstance(a, mx.array) else type(a).__name__
                   for a in kw.get("inputs") or ())
    outputs = tuple(str(d) for d in kw.get("output_dtypes") or ())
    return name, group, template, inputs, outputs


def _first(launch, kw: dict):
    """Launch and evaluate now: ``(limit, None)`` when this GPU refuses the group for the pipeline, else
    ``(0, outputs)``."""
    with _lock:
        _probes["count"] += 1
    try:
        outs = launch(**kw)
        mx.eval(outs)
    except ValueError as exc:
        found = _ERROR.search(str(exc))
        if found is None:
            raise
        return int(found.group(2)), None
    return 0, outs


def _limit(launch, name: str, kw: dict):
    """``(group, limit, source, outputs)``: this GPU's limit for the launch's pipeline when it refuses the group
    (else 0), and the launch's evaluated outputs when this call was its pipeline's first (else None)."""
    group = _group(kw.get("threadgroup"))
    cap = test_limit(name)
    if cap is not None and group > cap:
        return group, cap, "test hook", None
    if group <= PROBE_ABOVE:
        return group, 0, "", None
    key = _pipeline_key(name, group, kw)
    limit = _verdicts.get(key)
    outs = None
    if limit is None:
        limit, outs = _first(launch, kw)
        with _lock:
            _verdicts[key] = limit
    return group, limit, "pipeline", outs


def _note(name: str, group: int, limit: int, source: str, served_by: str, counter: str) -> None:
    """Record, count (route counter ``counter``) and say once a process that this GPU cannot run a launch's
    thread group, and what serves it instead."""
    with _lock:
        entry = _refused.get((name, group))
        first = entry is None
        if first:
            entry = _refused[(name, group)] = {"kernel": name, "group": group, "limit": limit, "source": source,
                                                "served_by": served_by, "launches": 0}
        entry["launches"] += 1
    from .fast.routes import count

    count(counter)
    if first:
        print(f"[mtplx] inkling: {name} cannot run {group}-thread groups on this GPU (its pipeline allows {limit}"
              f"{'; test hook' if source == 'test hook' else ''}); {served_by}", flush=True)


def runs(launch, name: str, kw: dict, instead: str | None = None) -> bool:
    """Whether this GPU runs this launch's thread group for its pipeline (decided once per pipeline and group;
    deciding it runs the launch once). With ``instead`` (what the caller runs in its place, with the same bits)
    a refusal is recorded, counted ``fused.split.<kernel>.<group>`` (a stage path: replayed programs count it
    again) and said once."""
    group, limit, source, _ = _limit(launch, name, kw)
    if not limit:
        return True
    if instead:
        _note(name, group, limit, source, instead, f"fused.split.{name}.{group}")
    return False


def check(launch, name: str, kw: dict):
    """The launch's evaluated outputs when this call decided its pipeline (else None); raises ``GroupRefused``
    (recorded, counted ``threadgroup.refused.<kernel>.<group>``, said once) when this GPU cannot run its group.
    The stage that catches it counts ``fallback.<stage>.threadgroup`` for every call it serves instead."""
    group, limit, source, outs = _limit(launch, name, kw)
    if limit:
        _note(name, group, limit, source, STAGE_ROUTE, f"threadgroup.refused.{name}.{group}")
        raise GroupRefused(name, group, limit, source)
    return outs


class Kernel:
    """An ``mx.fast.metal_kernel`` whose launches are checked against this GPU's limit first (``check``)."""

    __slots__ = ("_launch", "name")

    def __init__(self, launch, name: str):
        self._launch = launch
        self.name = name

    def __call__(self, **kw):
        outs = check(self._launch, self.name, kw)
        return self._launch(**kw) if outs is None else outs

    def runs(self, instead: str | None = None, **kw) -> bool:
        """Whether this GPU runs the launch ``kw``; ``instead`` names the same-bits form the caller runs if not."""
        return runs(self._launch, self.name, kw, instead)


def kernel(*args, **kwargs) -> Kernel:
    """``mx.fast.metal_kernel(...)``, its launches checked against this GPU's per-pipeline thread-group limit."""
    name = kwargs.get("name", args[0] if args else "kernel")
    return Kernel(mx.fast.metal_kernel(*args, **kwargs), str(name))


def report() -> dict[str, Any]:
    """For /health and ``mtplx doctor``: every launch this GPU refused and what it allows, the probes made."""
    with _lock:
        refused = [dict(entry) for entry in _refused.values()]
        probed = len(_verdicts)
        probes = _probes["count"]
    return {"refused": sorted(refused, key=lambda e: (e["kernel"], e["group"])), "pipelines_checked": probed,
            "probes": probes, "probe_above_threads": PROBE_ABOVE,
            "test_limit": os.environ.get(TEST_LIMIT_ENV, "").strip() or None}


def reset_for_tests() -> None:
    with _lock:
        _verdicts.clear()
        _refused.clear()
        _probes["count"] = 0


__all__ = ["GroupRefused", "Kernel", "PROBE_ABOVE", "TEST_LIMIT_ENV", "check", "kernel", "report",
           "reset_for_tests", "runs", "test_limit"]
