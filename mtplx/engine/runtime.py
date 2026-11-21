"""The Python face of the native executor.

One place reads the switch, one place loads the compiled module, and one
``Engine`` exists per process.  Nothing here may break ``import mtplx``: when
the compiled module is missing, was built for another Python or another MLX, or
the Mac cannot run it, ``available()`` is False, ``unavailable()`` says why and
every caller stays on the MLX path.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import sys
import sysconfig
import threading
from pathlib import Path
from typing import Any

from .ir import ArrayIR, BufferIR, ProgramIR, StepIR  # noqa: F401  (re-exported)

SWITCH_ENV = "MTPLX_NATIVE_ENGINE"
DUMP_ENV = "MTPLX_NATIVE_ENGINE_DUMP"
MODULE_ENV = "MTPLX_NATIVE_ENGINE_MODULE"  # developer override: path to a built module
# Test aid, off unless set: the module's start check reports a failure, as a broken build's
# would, so a release gate can drive the fallback on an installed wheel without touching it.
TEST_FAIL_START_ENV = "MTPLX_NATIVE_ENGINE_TEST_FAIL_START"

# What this file expects from the compiled module.  A module that reports a
# different number is refused, so a stale build can never serve.
NATIVE_API = 1

_OFF = {"0", "off", "false", "no", "none", "disable", "disabled"}
_SHADOW = {"shadow"}


class EngineUnavailable(RuntimeError):
    """The native executor cannot be used in this process."""


class BuildRefused(Exception):
    """A recording cannot become a program.  ``code`` is a counter name."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------- the switch


def mode() -> str:
    """``auto`` (on wherever the module loads), ``0`` (off) or ``shadow``."""

    raw = os.environ.get(SWITCH_ENV, "").strip().lower()
    if raw in _OFF:
        return "0"
    if raw in _SHADOW:
        return "shadow"
    return "auto"


def enabled() -> bool:
    """True when the switch allows the engine and the module is usable."""

    return mode() != "0" and available()


def dump_dir() -> Path | None:
    raw = os.environ.get(DUMP_ENV, "").strip()
    return Path(raw).expanduser() if raw else None


# ------------------------------------------------------ loading the module

_LOCK = threading.Lock()
_NATIVE: Any = None
_REASON: str | None = None  # None: not tried yet; "": loaded
_ENGINE: "Engine | None" = None


def _build_tag() -> str:
    return sysconfig.get_config_var("SOABI") or f"cpython-{sys.version_info[0]}{sys.version_info[1]}"


def _candidate_paths() -> list[Path]:
    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    here = Path(__file__).resolve().parent
    package = here.parent
    paths: list[Path] = []
    override = os.environ.get(MODULE_ENV, "").strip()
    if override:
        paths.append(Path(override).expanduser())
    # Installed: the module ships inside the package (mtplx/_engine*.so).
    for suffix in suffixes:
        paths.append(package / f"_engine{suffix}")
    # Source tree: native/build/<SOABI>/_engine*.so, one folder per Python.
    build = package.parent / "native" / "build" / _build_tag()
    for suffix in suffixes:
        paths.append(build / f"_engine{suffix}")
    return paths


def _load() -> None:
    global _NATIVE, _REASON
    if _REASON is not None:
        return
    with _LOCK:
        if _REASON is not None:
            return
        if sys.platform != "darwin":
            _REASON = "not_macos"
            return
        try:
            # libmlx must be in the process before the module binds to it.
            importlib.import_module("mlx.core")
        except Exception as exc:  # pragma: no cover - mlx is a hard dependency on Macs
            _REASON = f"mlx_import_failed: {exc!r}"
            return
        tried: list[str] = []
        for path in _candidate_paths():
            if not path.is_file():
                continue
            try:
                spec = importlib.util.spec_from_file_location("mtplx._engine", path)
                if spec is None or spec.loader is None:
                    tried.append(f"{path.name}: no loader")
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as exc:
                tried.append(f"{path}: {exc!r}")
                continue
            api = getattr(module, "NATIVE_API", None)
            if api != NATIVE_API:
                tried.append(f"{path}: api {api!r}, need {NATIVE_API}")
                continue
            problem = _check_module(module)
            if problem:
                tried.append(f"{path}: {problem}")
                continue
            sys.modules["mtplx._engine"] = module
            _NATIVE = module
            _REASON = ""
            return
        _REASON = "module_missing" if not tried else "module_unusable: " + " | ".join(tried)


def _check_module(module: Any) -> str:
    """Everything that must hold before the module may serve; "" when it does."""

    try:
        import mlx.core as mx

        built = str(getattr(module, "BUILT_AGAINST_MLX", ""))
        if built != mx.__version__:
            # The module reads MLX's array layout directly.
            return f"built for mlx {built}, running mlx {mx.__version__}"
        problem = module.self_check()
        if not problem and os.environ.get(TEST_FAIL_START_ENV, "").strip() not in ("", "0"):
            problem = f"self check failed (forced by {TEST_FAIL_START_ENV})"
        if problem:
            return str(problem)
        try:
            module.abi_probe(mx.zeros((1,), dtype=mx.float32))
        except TypeError:
            return "array types are not shared with mlx.core (nanobind mismatch)"
    except Exception as exc:
        return f"start check raised {exc!r}"
    return ""


def available() -> bool:
    """True when the compiled module is loaded and passed its own start check."""

    _load()
    return _NATIVE is not None


def unavailable() -> str:
    """Why the engine cannot be used, or ``""`` when it can."""

    _load()
    return _REASON or ""


def native_module() -> Any:
    """The compiled module ``mtplx._engine`` (raises ``EngineUnavailable``)."""

    _load()
    if _NATIVE is None:
        raise EngineUnavailable(_REASON or "module_missing")
    return _NATIVE


def module_version() -> str:
    _load()
    return str(getattr(_NATIVE, "__version__", "")) if _NATIVE is not None else ""


def engine() -> "Engine":
    """The process-wide engine (raises ``EngineUnavailable``)."""

    global _ENGINE
    if _ENGINE is None:
        native_module()
        with _LOCK:
            if _ENGINE is None:
                _ENGINE = Engine(label="process")
    return _ENGINE


# ------------------------------------------------------------------ Engine


# How a replay orders its launches on the GPU.
#   concurrent  one concurrent pass with barriers: what MLX itself encodes
#   serial      one serial pass: launches run strictly in order, no barriers
#   hybrid      serial passes for runs of launches that each depend on the one
#               before, concurrent passes with barriers for the rest
DISPATCH = {"concurrent": 0, "serial": 1, "hybrid": 2}
#   recorded    barriers where MLX placed them
#   minimal     recomputed from what each launch reads and writes
#   named       barriers that name the buffers they protect
#   passes      no barrier calls: a new pass wherever a barrier would be needed
BARRIERS = {"recorded": 0, "minimal": 1, "named": 2, "passes": 4}
# The defaults since 10 November 2026 08:25: hybrid passes, recomputed
# barriers and a commit every 256 launches, measured together on the real
# pack through the server (the lead's window `wd`, rows/wd-1110-0811: engine
# wall 23.1 ms a verify forward against 24.2 for concurrent + recorded + one
# command buffer, the GPU's own time equal, the same text).  They were
# measured as a set, so they are the default as a set; each switch still
# turns its part back: MTPLX_NATIVE_ENGINE_DISPATCH=concurrent,
# MTPLX_NATIVE_ENGINE_BARRIERS=recorded, MTPLX_NATIVE_ENGINE_SPLIT=0.
DEFAULT_DISPATCH = "hybrid"
DEFAULT_BARRIERS = "minimal"
DEFAULT_SPLIT = 256


class RunRefused(RuntimeError):
    """``run`` could not replay this call.  ``reason`` is a counter name."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class EngineError(RuntimeError):
    """The engine failed in a way that is not a plain decline."""


class ReplayLost(EngineError):
    """A replay was committed and gave no outputs.

    The arrays the program writes in place (``Program.clobbered_inputs``) are
    of unknown content, and after a timeout they can still change until
    ``Engine.stuck()`` is false.  Every other input is untouched.  A caller
    that handed over live state cannot take the MLX path on it: the request
    fails and that state is dropped.  A caller that handed over private
    copies (a first-use check, shadow mode) loses nothing.

    ``code`` is one of:
      ``timeout``         the wait passed its bound or its hard limit, or the
                          request was stopped while the replay was in flight
      ``command_buffer``  Metal ended a command buffer in an error status
      ``outputs``         the GPU finished and the output arrays could not be
                          built
    ``wrote_inputs`` is False for a program that writes into none of its
    inputs: then nothing the caller holds was touched.
    """

    def __init__(self, reason: str, *, wrote_inputs: bool = True) -> None:
        super().__init__(reason)
        self.reason = reason
        self.code = reason.split(":", 2)[1].strip() if reason.startswith("failed:") else "unknown"
        self.wrote_inputs = bool(wrote_inputs)


def _failure(reason: str, program: "Program | None" = None) -> tuple[str, "ReplayLost | None"]:
    """A native reason sorted into its class: ``(decline, None)`` when nothing
    was written, ``("", ReplayLost)`` when the replay was committed."""

    if not reason.startswith("failed:"):
        return str(reason), None
    if reason.startswith("failed:prepare"):
        # The buffer table could not be made (an allocation): nothing was
        # committed and no input was touched.
        return "prepare_failed", None
    wrote = True if program is None else bool(getattr(program, "clobbered_inputs", True))
    return "", ReplayLost(reason, wrote_inputs=wrote)


class Trace:
    """What MLX launched on one thread between ``record_begin`` and ``record_end``."""

    def __init__(self, native: Any) -> None:
        self.native = native
        self._exported: dict | None = None

    def exported(self) -> dict:
        if self._exported is None:
            if getattr(_NATIVE, "SUPPORTS_INDIRECT_DISPATCH", False):
                self._exported = self.native.export(include_indirect=True)
            else:
                self._exported = self.native.export()
        return self._exported

    @property
    def launches(self) -> int:
        return int(self.native.launches())

    @property
    def mlx_command_buffers(self) -> int:
        return int(self.exported()["mlx_command_buffers"])

    @property
    def mlx_encoders(self) -> int:
        return int(self.exported()["mlx_encoders"])

    def kernel_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for step in self.exported()["steps"]:
            counts[step[1]] = counts.get(step[1], 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    def mark_counts(self) -> dict[str, int]:
        """Launches per ``record_mark`` label, in the order the marks were set."""

        counts: dict[str, int] = {}
        for step in self.exported()["steps"]:
            counts[step[11]] = counts.get(step[11], 0) + 1
        return counts


class Program:
    """A sealed program: the IR it came from and its executable form."""

    def __init__(self, native: Any, ir: ProgramIR) -> None:
        self.native = native
        self.ir = ir
        self.name = ir.name
        self.signature = ir.signature()
        self.launches = len(ir.steps)
        # Inputs the program writes into.  After a run they are consumed: the
        # caller must drop them and use the outputs.
        self.clobbered_inputs = sorted(
            buf.index for buf in ir.buffers if buf.role == "input" and buf.clobbered
        )
        # Barrier counts under each form, and how many launches sit in runs
        # where each depends on the one before (what a serial pass is for).
        self.barriers = dict(getattr(native, "barriers", {}) or {})
        # This program's own form of DISPATCH and BARRIERS (a site sets them
        # for a program whose launches the executor's setting does not suit);
        # None: the executor's.
        self.dispatch: str | None = None
        self.barriers_form: str | None = None
        self.split: int | None = None

    def __repr__(self) -> str:
        return f"<Program {self.name!r} launches={self.launches} sig={self.signature[:8]}>"


class Engine:
    """Records what MLX launches for one evaluation and replays it natively.

    See ``outputs/metal-overnight-20261109/ENGINE-CONTRACT.md`` section 3 and
    the ``API READY`` section of agent A's state file for the rules.
    """

    def __init__(self, *, label: str = "", validate: bool = False) -> None:
        module = native_module()
        self._module = module
        self._native = module.Engine(label, validate)
        self.label = label
        self.validate = validate
        self._builds_refused: dict[str, int] = {}
        self._last_reason = ""
        # How a replay orders its launches on the GPU, read once.  Every form
        # computes the same bits; see DISPATCH and BARRIERS.
        dispatch = os.environ.get("MTPLX_NATIVE_ENGINE_DISPATCH", "").strip().lower()
        self.dispatch = dispatch if dispatch in DISPATCH else DEFAULT_DISPATCH
        barriers = os.environ.get("MTPLX_NATIVE_ENGINE_BARRIERS", "").strip().lower()
        self.barriers = barriers if barriers in BARRIERS else DEFAULT_BARRIERS
        try:
            self.min_run = max(2, int(os.environ.get("MTPLX_NATIVE_ENGINE_MIN_RUN", "4") or 4))
        except ValueError:
            self.min_run = 4
        # Inline constants are bound from one buffer written at seal.  Setting
        # this sends them as bytes on every launch instead, as MLX does.
        self.copy_inline = os.environ.get("MTPLX_NATIVE_ENGINE_COPY_INLINE", "").strip() == "1"
        # Encode a replay's launches before MLX's own queue is empty and
        # commit them after (the watched run only: `submit`).  Off until the
        # lead has a row for it.
        self.encode_early = os.environ.get("MTPLX_NATIVE_ENGINE_ENCODE_EARLY", "").strip() == "1"
        # The wait for a replay: 0 sleeps until Metal's completion handler
        # wakes this thread; N microseconds makes it watch for the handler
        # from N microseconds before the replay is due (see set_spin_lead_us
        # in the module).  Off until the lead has a row for it.
        try:
            self.spin_us = max(0, int(os.environ.get("MTPLX_NATIVE_ENGINE_SPIN_US", "0") or 0))
        except ValueError:
            self.spin_us = 0
        if self.spin_us and hasattr(self._native, "set_spin_lead_us"):
            self._native.set_spin_lead_us(self.spin_us)
        # Test aid, off unless set: "N" makes the Nth replay of the process
        # that finished be reported as lost ("N:timeout" picks the code), so
        # the path a lost replay takes above the engine (a refused key, a
        # failed request, a dropped session) can be driven through the
        # product loop.  The replay itself ran and wrote its inputs.
        self._lose_at, self._lose_code, self._finished = 0, "command_buffer", 0
        raw_lose = os.environ.get("MTPLX_NATIVE_ENGINE_TEST_LOSE_REPLAY", "").strip()
        if raw_lose:
            count, _, code = raw_lose.partition(":")
            try:
                self._lose_at = max(0, int(count))
            except ValueError:
                self._lose_at = 0
            if code in {"timeout", "command_buffer", "outputs"}:
                self._lose_code = code
        # Streaming commit: commit every N launches while the rest is still
        # being encoded, so the GPU starts sooner (0: one command buffer).
        raw_split = os.environ.get("MTPLX_NATIVE_ENGINE_SPLIT", "").strip()
        self._split_explicit = bool(raw_split)
        try:
            self.split = max(0, int(raw_split)) if raw_split else DEFAULT_SPLIT
        except ValueError:
            self.split = DEFAULT_SPLIT

    def info(self) -> dict:
        info = dict(self._native.info())
        info["mode"] = mode()
        # MLX picks its kernels by this name when it is set (the M3/M4 rehearsal).
        info["mlx_gpu_arch_override"] = os.environ.get("MLX_METAL_GPU_ARCH", "")
        info["dispatch"] = self.dispatch
        info["barriers"] = self.barriers
        info["min_run"] = self.min_run
        info["split"] = self.split
        info["spin_us"] = self.spin_us
        info["encode_early"] = self.encode_early
        return info

    # ------------------------------------------------------------ recording

    def describe(self, arrays: list) -> list:
        """Where each evaluated array lives, without keeping the arrays alive.

        Take this before the MLX call you are about to record, for every input
        of the call.  Holding the input arrays themselves through the recorded
        evaluation would stop MLX writing into them in place, and the recording
        would then contain copies the live path never makes.
        """

        from .builder import ArrayRef

        keeper, rows = self._native.describe(list(arrays))
        return [ArrayRef.from_native(row, keeper) for row in rows]

    def record_begin(self, *, label: str = "", inputs: list | None = None) -> None:
        """Start following what MLX encodes on this thread."""

        del inputs  # accepted for symmetry with build(); not needed to record
        self._native.record_begin(label)

    def record_mark(self, label: str) -> None:
        """Name everything launched from here to the next mark."""

        self._native.record_mark(label)

    def record_end(self) -> Trace:
        """Stop following.  The evaluation must be complete (use ``mx.eval``)."""

        return Trace(self._native.record_end())

    def record_function(self, fn, inputs: list, *, name: str):
        """Record and seal one MLX evaluation. Return its program and live output arrays."""
        import mlx.core as mx

        refs = self.describe(inputs)
        self.settle(inputs)
        self.record_begin(label=name)
        try:
            result = fn(*inputs)
            outputs = list(result) if isinstance(result, (list, tuple)) else [result]
            mx.eval(*outputs)
        finally:
            trace = self.record_end()
        ir = self.build(trace, inputs=refs, outputs=outputs, name=name)
        return self.seal(ir), outputs

    def recording(self) -> bool:
        return bool(self._native.recording())

    def layout_key(self, arrays: list) -> int:
        """One number for what a recording depends on in its inputs.

        Type, shape and the strides that address elements (the stride of a
        dimension of extent 1 is left out).  Two calls with the same key launch
        the same kernels, so it belongs in a site's geometry key: a state leaf
        that arrives as a view of a longer buffer is another geometry than the
        same leaf arriving contiguous.  About a microsecond for 30 arrays.
        """

        return int(self._native.layout_key(arrays))

    def settle(self, arrays: list, *, timeout_ms: float = 5.0) -> bool:
        """Wait until MLX has let go of arrays it just evaluated.

        For some microseconds after ``mx.eval`` returns, MLX's command buffer
        still holds every array that evaluation produced.  MLX does not write
        into an array in that state, so a recording started right then would
        contain a copy the settled path never makes.  Call this on the inputs
        before a recorded evaluation.  (``try_run`` settles by itself.)
        """

        return bool(self._native.settle(list(arrays), int(timeout_ms * 1000)))

    # ------------------------------------------------------------- building

    def pin(self, arrays: list) -> list[int]:
        """Weights and other long-lived arrays: bound in place, kept alive.

        Returns the positions in ``arrays`` that were skipped because the array
        has not been evaluated yet (pin them again once it has).  Pinning an
        array twice is harmless.
        """

        slots = self._native.pin(list(arrays))
        return [index for index, slot in enumerate(slots) if slot == -2]

    def _refs(self, arrays: list) -> list:
        from .builder import ArrayRef

        if all(isinstance(item, ArrayRef) for item in arrays):
            return list(arrays)
        if any(isinstance(item, ArrayRef) for item in arrays):
            live = [item for item in arrays if not isinstance(item, ArrayRef)]
            described = iter(self.describe(live))
            return [item if isinstance(item, ArrayRef) else next(described) for item in arrays]
        return self.describe(arrays)

    def build(
        self,
        trace: Trace,
        *,
        inputs: list,
        outputs: list,
        name: str = "",
        const_limit: int | None = None,
    ) -> ProgramIR:
        """Turn a recording into a program, or raise ``BuildRefused``.

        ``inputs``: the ``describe()`` list taken before the recorded call (live
        arrays are accepted when nothing is written in place).  ``outputs``: the
        evaluated output arrays, in a fixed order.  ``const_limit``: the largest
        undeclared, unpinned buffer the program may carry as a constant (default
        16 MiB); set it low in a test on a small model to prove the pin list.
        """

        from . import builder

        try:
            return builder.build_ir(
                trace.exported(),
                read_buffer=trace.native.read_buffer,
                pin_slot=self._native.pin_slot,
                inputs=self._refs(inputs),
                outputs=self._refs(outputs),
                name=name,
                const_limit=const_limit,
            )
        except builder.Refused as refused:
            self._builds_refused[refused.code] = self._builds_refused.get(refused.code, 0) + 1
            raise BuildRefused(refused.code, refused.detail) from None

    def seal(self, ir: ProgramIR) -> Program:
        """Validate and freeze: allocates working memory and constants."""

        from . import builder

        problems = ir.check()
        if problems:
            raise BuildRefused("malformed_program", "; ".join(problems[:4]))
        try:
            if not getattr(_NATIVE, "SUPPORTS_INDIRECT_DISPATCH", False) and any(
                step.indirect_grid is not None for step in ir.steps
            ):
                raise RuntimeError("seal: indirect_dispatch_unsupported")
            # A module older than 0.6.4 would hand such an output a fresh buffer
            # without its bytes (builder.reused_constant): refuse instead.
            if not getattr(_NATIVE, "SUPPORTS_OUTPUT_SEEDS", False) and any(
                buf.role == "output" and buf.data is not None for buf in ir.buffers
            ):
                raise RuntimeError("seal: output_seeds_unsupported")
            native = self._native.seal(builder.to_spec(ir))
        except RuntimeError as exc:
            code = str(exc).removeprefix("seal: ").split(":")[0].strip() or "seal_failed"
            self._builds_refused[code] = self._builds_refused.get(code, 0) + 1
            raise BuildRefused(code, str(exc)) from None
        return Program(native, ir)

    # -------------------------------------------------------------- running

    def can_run(self, program: Program, inputs: list, *, extra_refs: int = 1) -> str:
        """``""`` when this call can replay; otherwise the reason code."""

        return str(self._native.can_run(program.native, inputs, extra_refs))

    def try_run(
        self,
        program: Program,
        inputs: list,
        *,
        profile: bool = False,
        extra_refs: int = 1,
    ):
        """Replay if legal.  Returns ``(outputs, "")`` or ``(None, reason)``.

        A reason means nothing was touched and the caller takes the MLX path.
        A replay that was committed and then failed raises ``ReplayLost``.

        ``inputs`` must be a list.  Each input the program writes into
        (``program.clobbered_inputs``) may be referenced by that list and by at
        most ``extra_refs`` other places (by default one: the container it
        lives in).  Any other holder would see the array change under it, so
        the call is refused with ``state_shared``, exactly where MLX would
        have made a copy.  After a run those inputs are consumed.
        """

        dispatch, barriers = self._form(program)
        reason, outputs = self._native.run(
            program.native,
            inputs,
            False,
            profile,
            False,
            self._program_split(program),
            extra_refs,
            dispatch,
            barriers,
            self.min_run,
            self.copy_inline,
        )
        if reason:
            decline, lost = _failure(reason, program)
            if lost is not None:
                raise lost
            return None, decline
        self._maybe_lose(program)
        return outputs, ""

    def _form(self, program: Program) -> tuple[int, int]:
        """The dispatch and barrier codes a replay of ``program`` runs with."""

        dispatch = getattr(program, "dispatch", None)
        barriers = getattr(program, "barriers_form", None)
        return (
            DISPATCH[dispatch if dispatch in DISPATCH else self.dispatch],
            BARRIERS[barriers if barriers in BARRIERS else self.barriers],
        )

    def _program_split(self, program: Program) -> int:
        split = getattr(program, "split", None)
        return self.split if self._split_explicit or split is None else split

    def _maybe_lose(self, program: Program | None) -> None:
        """The test switch ``MTPLX_NATIVE_ENGINE_TEST_LOSE_REPLAY`` (see __init__)."""

        if not self._lose_at:
            return
        self._finished += 1
        if self._finished == self._lose_at:
            raise _failure(f"failed:{self._lose_code}: lost on purpose (test switch)", program)[1]

    def run(
        self,
        program: Program,
        inputs: list,
        *,
        then_prepare: Any = None,
        profile: bool = False,
        extra_refs: int = 1,
    ) -> list:
        """Replay.  Blocks until the GPU is done; the interpreter lock is released.

        ``then_prepare`` is a callable that runs on this thread while the GPU
        works (the next round's host preparation, say): the launches are
        committed, it is called, and then the replay is collected.  It must
        not touch the inputs the program writes into.
        """

        if then_prepare is None or profile:
            outputs, reason = self.try_run(program, inputs, profile=profile, extra_refs=extra_refs)
            if outputs is None:
                raise RunRefused(reason)
            return outputs
        outputs, reason = self.run_patient(
            program, inputs, extra_refs=extra_refs, then_prepare=then_prepare
        )
        if outputs is None:
            raise RunRefused(reason)
        return outputs

    # A replay in two halves, so the caller can work while the GPU does.

    def submit(self, program: Program, inputs: list, *, extra_refs: int = 1) -> str:
        """Commit the program's launches and return.  ``""``: in flight.

        Any other answer is a reason as from ``try_run``: nothing was touched
        and the caller takes the MLX path.  While a replay is in flight the
        inputs it writes into belong to it, and every other ``run`` or
        ``submit`` answers ``engine_busy``.
        """

        dispatch, barriers = self._form(program)
        options = [
            program.native,
            inputs,
            self._program_split(program),
            extra_refs,
            dispatch,
            barriers,
            self.min_run,
        ]
        if self.encode_early:
            # Only a module that has the option is given it (0.6.1 on).
            options.append(True)
        reason = self._native.submit(*options)
        decline, lost = _failure(reason, program)
        if lost is not None:
            raise lost
        if not decline:
            self._flight_program = program
        return decline

    def collect(self, timeout_ms: int) -> tuple[list | None, str]:
        """Wait at most ``timeout_ms`` for the replay in flight.

        ``(outputs, "")`` when it has finished; ``(None, "pending")`` when it
        has not (nothing was touched: call again, or ``give_up``);
        ``(None, "no_flight")``.  A replay that ended in error raises
        ``ReplayLost``.
        """

        reason, outputs = self._native.collect(max(0, int(timeout_ms)))
        if reason == "pending":
            return None, "pending"
        program, self._flight_program = getattr(self, "_flight_program", None), None
        if reason:
            decline, lost = _failure(reason, program)
            if lost is not None:
                raise lost
            return None, decline
        self._maybe_lose(program)
        return outputs, ""

    def give_up(self) -> None:
        """Stop waiting for the replay in flight, as if it had passed its bound.

        The engine answers ``engine_stuck`` until the GPU has let go of it,
        and the inputs it writes into stay indeterminate until then.
        """

        self._native.give_up()

    def release_idle_temporaries(self) -> int:
        """Return scratch to the MLX allocator while keeping every checked program."""
        release = getattr(self._native, "release_idle_temporaries", None)
        return int(release()) if release is not None else 0

    def in_flight(self) -> bool:
        return bool(self._native.in_flight())

    def wait(
        self,
        *,
        slice_ms: int = 250,
        limit_ms: int | None = 30_000,
        should_stop: Any = None,
    ) -> tuple[list | None, str]:
        """Collect the replay in flight, looking up every ``slice_ms``.

        A replay that is slow (another process holding the GPU) is waited for:
        it is the same work the MLX path would be waiting for.  The wait ends
        early only when ``should_stop()`` says so (a cancelled request) or
        ``limit_ms`` passes (None: no limit); the replay is then given up and
        ``ReplayLost("failed:timeout: ...")`` raised, with the engine stuck
        until the GPU has let go.
        """

        waited = 0
        while True:
            outputs, reason = self.collect(slice_ms)
            if reason != "pending":
                return outputs, reason
            waited += slice_ms
            stopped = bool(should_stop()) if should_stop is not None else False
            if stopped or (limit_ms is not None and waited >= limit_ms):
                program, self._flight_program = getattr(self, "_flight_program", None), None
                self.give_up()
                why = "the request was stopped" if stopped else f"no completion within {waited} ms"
                raise _failure(f"failed:timeout: {why}", program)[1]

    def run_patient(
        self,
        program: Program,
        inputs: list,
        *,
        extra_refs: int = 1,
        slice_ms: int = 250,
        limit_ms: int | None = 30_000,
        should_stop: Any = None,
        then_prepare: Any = None,
    ) -> tuple[list | None, str]:
        """``try_run`` whose wait can be watched: ``(outputs, "")`` or ``(None, reason)``.

        Unlike ``try_run`` it does not give a slow replay up at 50 times the
        program's GPU time: see ``wait``.  ``then_prepare``, when given, is
        called once the launches are committed and before the wait: host work
        of the caller's that fits under the GPU's time.  It is not called when
        the replay is declined (the caller then does that work its own way).
        """

        if not hasattr(self._native, "submit"):  # a module older than 0.6.0
            outputs, reason = self.try_run(program, inputs, extra_refs=extra_refs)
            if outputs is not None and then_prepare is not None:
                then_prepare()
            return outputs, reason
        reason = self.submit(program, inputs, extra_refs=extra_refs)
        if reason:
            return None, reason
        if then_prepare is None:
            return self.wait(slice_ms=slice_ms, limit_ms=limit_ms, should_stop=should_stop)
        try:
            then_prepare()
        finally:
            # Whatever the callable did, the replay in flight is collected.
            result = self.wait(slice_ms=slice_ms, limit_ms=limit_ms, should_stop=should_stop)
        return result

    def null_run(self, program: Program, inputs: list, *, extra_refs: int = 1) -> dict:
        """Encode the program into an encoder that ignores every call.

        Nothing reaches the GPU, the inputs are not touched and no outputs are
        made.  Returns the run's figures: ``encode_ns`` is then the cost of this
        engine's own loop alone, and with the allocation probe on,
        ``mallocs_encode`` counts the allocations that loop makes (none).
        """

        reason, _outputs = self._native.run(
            program.native,
            inputs,
            False,
            False,
            False,
            0,
            extra_refs,
            DISPATCH["concurrent"],
            BARRIERS[self.barriers],
            self.min_run,
            self.copy_inline,
            True,
        )
        if reason:
            raise RunRefused(str(reason))
        return dict(self._native.last_run())

    def last_run(self) -> dict:
        return dict(self._native.last_run())

    def time_form(
        self,
        program: Program,
        inputs: list,
        *,
        dispatch: str,
        barriers: str,
        split: int,
        fill: int | None = None,
        extra_refs: int = 1,
    ) -> tuple[dict | None, str]:
        """One replay in a named form, for a census: ``(last_run figures, "")``
        or ``(None, reason)``.

        ``dispatch`` and ``barriers`` name the forms of ``DISPATCH`` and
        ``BARRIERS``; ``barriers="none"`` encodes no barrier at all, so the
        launches overlap as far as the GPU lets them and the results are wrong.
        Its time is the floor that fusing dependent launches could reach. The
        outputs are dropped, and the inputs the program writes into are
        consumed as by any run: hand in private copies.

        ``fill`` (0 to 255) starts the program's working memory and its fresh
        outputs as that byte. A barrier-free replay needs it: a launch can read
        an index buffer (expert ids, offsets) before the launch that writes it,
        and a recycled buffer's bytes there would index out of bounds; zero is
        always in range.
        """

        # The executor's kNone (native/engine/executor.h): timing only, never a
        # serving form, so it is not in BARRIERS where a switch could pick it.
        codes = dict(BARRIERS, none=3)
        reason, outputs = self._native.run(
            program.native,
            inputs,
            False,
            False,
            False,
            max(0, int(split)),
            extra_refs,
            DISPATCH[dispatch],
            codes[barriers],
            self.min_run,
            self.copy_inline,
            False,
            -1 if fill is None else int(fill) & 0xFF,
        )
        del outputs
        if reason:
            decline, lost = _failure(reason, program)
            if lost is not None:
                raise lost
            return None, decline
        return dict(self._native.last_run()), ""

    def pin_slot(self, pointer: int) -> int:
        """The pin-table slot of the Metal buffer at ``pointer`` (``ArrayRef.ptr``), -1 when unpinned."""

        return int(self._native.pin_slot(int(pointer)))

    def step_run(
        self,
        program: Program,
        inputs: list,
        *,
        after_step: Any,
        fill: int | None = None,
        extra_refs: int = 1,
    ):
        """Replay one launch per command buffer and look after each one.

        ``after_step(index)`` is called when launch ``index`` has completed on
        the GPU; inside it ``peek`` and ``peek_digest`` read any buffer of the
        program as it is at that moment.  With ``fill`` (0 to 255) the
        program's working memory and its fresh output buffers start as that
        byte, so bytes the program never writes are known.  A diagnosis tool:
        it is a real run (clobbered inputs are consumed) and it is slow.
        Returns ``(outputs, "")`` or ``(None, reason)`` like ``try_run``.
        """

        try:
            reason, outputs = self._native.run(
                program.native,
                inputs,
                False,
                True,
                False,
                0,
                extra_refs,
                DISPATCH["concurrent"],
                BARRIERS["recorded"],
                self.min_run,
                False,
                False,
                -1 if fill is None else int(fill) & 0xFF,
                after_step,
            )
        except TypeError as exc:
            raise EngineError(f"the compiled module is older than 0.4.0: {exc}") from None
        if reason:
            if reason.startswith("failed:"):
                raise EngineError(reason)
            return None, str(reason)
        return outputs, ""

    def clone_layout(self, array: Any, *, min_length: int = 0) -> Any:
        """The same bytes, shape, strides and offset in a buffer of its own:
        the whole parent buffer is copied, so a view stays the same view.
        For a first-use check and for diagnosis (a private copy that the
        executor accepts where it accepted the original). ``min_length``
        asks for a buffer at least that many bytes long (tests: an array in
        a buffer with room to spare, as MLX's allocator often leaves one)."""

        if min_length:
            return self._native.clone_layout(array, int(min_length))
        return self._native.clone_layout(array)

    def mark_tail(self, array: Any, seed: int) -> int:
        """Fill the room past ``array``'s end in its buffer (the allocator's
        slack) with a pattern made from ``seed``; the bytes marked, 0 when
        there is no such room or it is not the array's alone (a view or a
        parent shares the buffer), or the module is older than 0.6.3."""

        native = getattr(self._native, "mark_tail", None)
        if native is None:
            return 0
        return int(native(array, int(seed) & 0xFFFFFFFFFFFFFFFF))

    def tail_intact(self, array: Any, seed: int, marked: int) -> bool:
        """True when the bytes ``mark_tail`` marked past ``array`` still hold the pattern."""

        native = getattr(self._native, "tail_intact", None)
        if native is None or not marked:
            return False
        return bool(native(array, int(seed) & 0xFFFFFFFFFFFFFFFF, int(marked)))

    def relax_input(self, program: Program, index: int) -> bool:
        """After a run left input ``index``'s marked tail intact: let later calls
        hand that input in a buffer with less room than the recorded one (the
        program needs nothing past the array). False when it cannot."""

        native = getattr(self._native, "relax_input", None)
        if native is None:
            return False
        return bool(native(program.native, int(index)))

    def last_short_detail(self) -> str:
        """Which input the last ``input_buffer_small`` answer was about, and its lengths."""

        native = getattr(self._native, "last_short_detail", None)
        return str(native()) if native is not None else ""

    def set_wait_floor_ms(self, ms: int) -> None:
        """The shortest bound on any wait for the GPU (default 2,000 ms; the
        bound of a replay is the larger of this and 50 times the program's
        best GPU time).  A replay that passes its bound raises ``EngineError``
        ("failed:timeout"), and the engine declines every run with
        ``engine_stuck`` until that command buffer has left the GPU."""

        self._native.set_wait_floor_ms(int(ms))

    def stuck(self) -> bool:
        return bool(self._native.stuck())

    def peek(self, program: Program, buffer: int, offset: int = 0, length: int = -1) -> bytes:
        """Bytes of buffer ``buffer`` (its position in ``program.ir.buffers``)
        from ``offset`` of the recorded buffer.  Only inside ``after_step``."""

        return self._native.peek(program.native, int(buffer), int(offset), int(length))

    def peek_digest(
        self, program: Program, buffer: int, offset: int = 0, length: int = -1
    ) -> tuple[int, int]:
        """``(bytes read, 64-bit digest)`` of the region ``peek`` would return."""

        count, digest = self._native.peek_digest(
            program.native, int(buffer), int(offset), int(length)
        )
        return int(count), int(digest)

    def last_profile(self) -> list[dict]:
        """Per step of the last ``profile=True`` run: kernel name and GPU ns."""

        return [
            {"step": index, "kernel": kernel, "kernel_name": name, "gpu_ns": gpu, "wall_ns": wall}
            for index, (kernel, name, gpu, wall) in enumerate(self._native.last_profile())
        ]

    def profile(self, program: Program, inputs: list) -> list[dict]:
        """Run with one command buffer per launch and return the per-step times.

        This is a real run: clobbered inputs are consumed and the outputs are
        dropped.  Use ``run(..., profile=True)`` then ``last_profile()`` to keep
        the outputs.
        """

        self.run(program, inputs, profile=True)
        return self.last_profile()

    def counters(self) -> dict:
        counters = dict(self._native.counters())
        counters["refused"] = dict(counters.get("refused", {}))
        counters["builds_refused"] = dict(self._builds_refused)
        return counters

    def synchronize(self) -> None:
        self._native.synchronize()
