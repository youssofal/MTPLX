"""Metal buffers die when MLX frees them, whatever Python thread called the engine.

The engine is Objective-C++ and Python threads have no autorelease pool: an
object autoreleased on one is never released while the thread lives.  Reading
a buffer's bytes with ``-contents`` makes ARC retain and autorelease the
buffer, so every buffer the engine read that way (a first-use check's clones
and the live state they were cloned from, a program's constants, a short state
leaf copied into a longer buffer) stayed allocated after MLX freed it, and MLX
no longer counted it.

These tests call the engine from a thread of their own, as the server does.
No model is loaded.  Run with pytest, or directly (``python
tests/engine/test_buffer_lifetime.py``) in an environment without pytest.
"""

from __future__ import annotations

import ctypes
import gc
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import mlx.core as mx

import mtplx
from mtplx.engine import runtime
from mtplx.engine.selftest import bits, record

try:  # pytest is optional: the file also runs as a script
    import pytest

    pytestmark = pytest.mark.skipif(
        not runtime.available(), reason=f"native engine unavailable: {runtime.unavailable()}"
    )
except ImportError:  # pragma: no cover
    pytest = None

_HERE = Path(__file__).resolve()


class _Weak:
    """An Objective-C weak reference: tells whether an object still exists
    without keeping it alive."""

    _objc = None

    def __init__(self, address: int):
        if _Weak._objc is None:
            objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
            objc.objc_initWeak.restype = ctypes.c_void_p
            objc.objc_initWeak.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
            objc.objc_loadWeakRetained.restype = ctypes.c_void_p
            objc.objc_loadWeakRetained.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            objc.objc_destroyWeak.restype = None
            objc.objc_destroyWeak.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            objc.objc_release.restype = None
            objc.objc_release.argtypes = [ctypes.c_void_p]
            _Weak._objc = objc
        self._slot = ctypes.c_void_p(0)
        _Weak._objc.objc_initWeak(ctypes.byref(self._slot), ctypes.c_void_p(address))

    def alive(self) -> bool:
        held = _Weak._objc.objc_loadWeakRetained(ctypes.byref(self._slot))
        if held:
            _Weak._objc.objc_release(held)
            return True
        return False

    def close(self) -> None:
        _Weak._objc.objc_destroyWeak(ctypes.byref(self._slot))


def _buffer_of(engine, array) -> _Weak:
    # describe() keeps the buffer alive through its keeper: only the address
    # is taken, and the keeper goes with the row.
    row = engine.describe([array])[0]
    address = row.ptr
    del row
    return _Weak(address)


def _survivors(weak: list[_Weak]) -> int:
    """Buffers still allocated once MLX has let go of them and emptied its cache.

    MLX holds an evaluation's arrays until its command buffer's completion
    handler has run, a little after the evaluation returns: a freed buffer is
    gone within milliseconds, a held one stays."""

    gc.collect()
    mx.synchronize()
    deadline = time.monotonic() + 2.0
    while True:
        mx.clear_cache()
        alive = sum(w.alive() for w in weak)
        if alive == 0 or time.monotonic() > deadline:
            break
        time.sleep(0.02)
    for w in weak:
        w.close()
    return alive


def _on_a_thread(body):
    """Run ``body`` on a new Python thread (no autorelease pool), as the server does."""

    failure: list[BaseException] = []

    def target():
        try:
            body()
        except BaseException as exc:  # noqa: BLE001 - handed to the caller
            failure.append(exc)

    worker = threading.Thread(target=target, name="engine-caller")
    worker.start()
    worker.join()
    if failure:
        raise failure[0]


def _rand(seed: int, shape: tuple[int, ...]):
    return mx.random.normal(shape, key=mx.random.key(seed))


def test_a_cloned_state_and_its_clone_are_freed_with_their_arrays():
    """The first-use check clones every state leaf a program writes into.
    Both the live leaf and its clone must be freed when MLX frees them; they
    stayed allocated for the life of the calling thread."""

    engine = runtime.engine()
    found: dict = {}

    def body():
        sources = [mx.full((64 * 1024,), float(k + 1), dtype=mx.float32) for k in range(4)]
        mx.eval(*sources)
        clones = [engine.clone_layout(source) for source in sources]
        mx.eval(*clones)
        assert [bits(c) for c in clones] == [bits(s) for s in sources]
        weak = [_buffer_of(engine, x) for x in sources + clones]
        del sources, clones
        found["survivors"] = _survivors(weak)

    _on_a_thread(body)
    assert found["survivors"] == 0, (
        f"{found['survivors']} of 8 Metal buffers (4 states, 4 clones) outlived their arrays"
    )


def _engine_calls() -> dict:
    """One of each engine call a serving thread makes, on that thread.

    A program recorded with a constant past the snapshot size (read back by
    the build, then sealed into the program) and a state leaf it writes in
    place, recorded in a buffer with room to spare; then a round whose state
    arrives in a buffer of its own size (copied into a longer one by the run),
    and a round through submit and collect."""

    engine = runtime.engine()
    engine.info()
    shape = (36, 64)
    spare = shape[0] * shape[1] * 4 + 3072
    table = mx.arange(32 * 1024, dtype=mx.float32) * 0.25  # 128 KiB
    mx.eval(table)

    def fn(state, row):
        doubled = state * 2.0 + table[: shape[1]]  # written into the state's own buffer
        return [mx.concatenate([doubled[:, 1:], row], axis=1), doubled.sum(axis=0)]

    def spared(seed):
        plain = _rand(seed, shape)
        mx.eval(plain)
        return engine.clone_layout(plain, min_length=spare)

    program, trace, ir = record(engine, fn, lambda s: [spared(s), _rand(s + 50, (shape[0], 1))], name="pools")
    del trace, ir
    report = {"clobbered": list(program.clobbered_inputs), "const_bytes": int(program.native.const_bytes)}

    state = _rand(70, shape) + 0.0  # a buffer of the array's own size: shorter than recorded
    row = _rand(71, (shape[0], 1))
    mx.eval(state, row)
    inputs = [state, row]
    del state
    outputs, reason = engine.try_run(program, inputs, extra_refs=0)
    del inputs
    assert reason == "", reason
    report["grown"] = int(engine.last_run()["grown"])

    row = _rand(72, (shape[0], 1))
    mx.eval(row)
    inputs = [outputs[0], row]
    del outputs
    outputs, reason = engine.run_patient(program, inputs, extra_refs=0)
    del inputs
    assert reason == "", reason
    report["submitted"] = True
    report["runs"] = int(engine.counters()["runs"])
    del outputs, program, table
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    return report


def test_no_engine_call_autoreleases_outside_a_pool():
    """Every engine call a serving thread makes, in a process where the
    Objective-C runtime names each object autoreleased with no pool in place
    (OBJC_DEBUG_MISSING_POOLS): none may be named."""

    env = dict(os.environ)
    env["OBJC_DEBUG_MISSING_POOLS"] = "YES"
    # The child runs the same source and the same compiled module as this process.
    env["MTPLX_NATIVE_ENGINE_MODULE"] = str(Path(runtime.native_module().__file__).resolve())
    source_root = str(Path(mtplx.__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(
        [source_root, *[p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]]
    )
    done = subprocess.run(
        [sys.executable, str(_HERE), "--child"],
        env=env, capture_output=True, text=True, timeout=300, cwd=source_root,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    report = json.loads([line for line in done.stdout.splitlines() if line.startswith("{")][-1])
    # The calls in question were made.
    assert report["clobbered"] == [0], report
    assert report["const_bytes"] >= 128 * 1024, report
    assert report["grown"] == 1, report
    assert report["submitted"] and report["runs"] >= 2, report
    named = [line for line in done.stderr.splitlines() if "MISSING POOLS" in line]
    assert not named, f"{len(named)} objects autoreleased with no pool in place:\n" + "\n".join(named[:12])


# ------------------------------------------------------------- run as script


def _child() -> int:
    from mtplx.engine.site import load_early

    load_early()  # before any kernel is compiled, as the server and conftest do
    if not runtime.available():
        print(json.dumps({"unavailable": runtime.unavailable()}))
        return 2
    report: dict = {}
    _on_a_thread(lambda: report.update(_engine_calls()))
    print(json.dumps(report))
    return 0


def _main() -> int:
    if not runtime.available():
        print(f"native engine unavailable: {runtime.unavailable()}")
        return 2
    failures = 0
    for name, test in sorted(globals().items()):
        if not name.startswith("test_") or not callable(test):
            continue
        try:
            test()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001 - report every failure
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"{'OK' if not failures else 'FAILED'}: {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    if "--child" in sys.argv:
        sys.exit(_child())
    from mtplx.engine.site import load_early

    load_early()
    sys.exit(_main())
