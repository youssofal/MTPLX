"""A failed host read of an MLX array raises in Python and leaves CPython intact.

numpy and memoryview read an ``mlx.core.array`` through its buffer slot, and
MLX's slot evaluates the array first.  An evaluation that throws (a tracer
read inside ``mx.compile``, an allocation MLX refuses) used to leave the slot
as a C++ exception, through CPython's C frames and without their cleanup.
Under a traced call the frames it crossed were never popped and their memory
was reused under the thread (the garbage-collector SIGSEGV and data-stack
SIGBUS that ended full suite runs on 20 November); with no C++ handler above,
the process aborted.  The native module puts a C boundary on that slot, for
the test suite only: it installs it when it loads with
MTPLX_MLX_BUFFER_GUARD=1 (``tests/conftest.py`` sets it) or when
``install_mlx_buffer_guard()`` is called; a serving process keeps MLX's own
slot.  Each failing read runs in a process of its own, so a module without
the boundary fails its case instead of taking the session down.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mtplx.engine import runtime

pytestmark = [
    pytest.mark.skipif(not mx.metal.is_available(), reason="MLX evaluates on the Metal GPU here"),
    pytest.mark.skipif(not runtime.available(), reason="the compiled executor is not built"),
    # A module built from this source can install the boundary on request;
    # one built before (a stored engine a measurement reuses) has no switch to
    # test, and says so instead of failing for its age.
    pytest.mark.skipif(runtime.available() and not hasattr(runtime.native_module(), "install_mlx_buffer_guard"),
                       reason="the loaded native module predates the switchable buffer guard: rebuild native/"),
]

ROOT = Path(__file__).resolve().parents[2]

PRELUDE = """
import sys
import mlx.core as mx
import numpy
from mtplx.engine import runtime

assert runtime.available(), runtime.unavailable()
print("guard:", getattr(runtime.native_module(), "MLX_BUFFER_GUARD", "absent"))
"""

# Inside a trace: the catching frame must be the thread's current frame again.
# The check runs before any other Python call, which would pop a stale frame.
TRACED = """
def traced(x):
    {read}
    return x

def call():
    compiled = mx.compile(traced)
    try:
        compiled(mx.array([1.0, 2.0]))
    except ValueError as exc:
        current = sys._getframe().f_code.co_name
        print("raised:", type(exc).__name__, str(exc))
        print("current frame:", current)

call()
"""

# Outside any transformation: an array MLX refuses to allocate.
REFUSED = """
big = mx.zeros((1 << 20, 1 << 20), dtype=mx.uint8)  # refused before any allocation
try:
    {read}
except RuntimeError as exc:
    print("raised:", type(exc).__name__, str(exc))
"""


def _run(body: str, *, guard: bool = True) -> list[str]:
    """The script's output lines; a failed or aborted process fails the test."""

    path = os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])
    script = textwrap.dedent(PRELUDE) + textwrap.dedent(body)
    env = dict(os.environ, PYTHONPATH=path)
    env.pop("MTPLX_MLX_BUFFER_GUARD", None)
    if guard:
        env["MTPLX_MLX_BUFFER_GUARD"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr[-2000:]
    return result.stdout.splitlines()


def test_a_serving_process_keeps_mlx_s_own_slot_unless_asked():
    # No switch: the module leaves mlx.core.array as MLX made it.
    lines = _run('print("installed now:", runtime.native_module().install_mlx_buffer_guard())', guard=False)
    assert "guard: off" in lines, lines
    # Asked for: installed, and asking again is a no-op.
    assert "installed now: installed" in lines, lines


def test_the_suite_puts_the_boundary_on_mlx_arrays_when_the_module_loads():
    assert os.environ.get("MTPLX_MLX_BUFFER_GUARD") == "1"  # tests/conftest.py
    module = runtime.native_module()
    if module.MLX_BUFFER_GUARD != "installed":
        # The module was loaded before the suite set the switch.
        assert module.install_mlx_buffer_guard() == "installed"
    # A read that succeeds is MLX's own: same bytes, shape and type.
    array = mx.arange(6, dtype=mx.float32).reshape(2, 3)
    view = memoryview(array)
    assert view.shape == (2, 3) and view.format == "f"
    assert np.asarray(array).tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


@pytest.mark.parametrize("read", ["numpy.asarray(x)", "memoryview(x)"])
def test_a_host_read_inside_a_trace_raises_and_pops_every_frame_it_crossed(read):
    lines = _run(TRACED.format(read=read))
    refusal = "raised: ValueError [eval] Attempting to eval"
    assert "guard: installed" in lines, lines
    assert any(line.startswith(refusal) for line in lines), lines
    assert "current frame: call" in lines, lines


@pytest.mark.parametrize("read", ["memoryview(big)", "numpy.asarray(big)"])
def test_a_refused_evaluation_during_a_host_read_raises_instead_of_aborting(read):
    lines = _run(REFUSED.format(read=read))
    assert "guard: installed" in lines, lines
    assert any(line.startswith("raised: RuntimeError [metal::malloc]") for line in lines), lines
