"""The grid is a real GPU read, including when it is a view of a live input."""
import ctypes
from dataclasses import replace
from pathlib import Path
import subprocess

import mlx.core as mx
import pytest

from mtplx.engine import runtime
from mtplx.engine.ir import ProgramIR
from mtplx.engine.site import compare_structure


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    if not runtime.available():
        pytest.skip(runtime.unavailable())
    folder = tmp_path_factory.mktemp("indirect-grid")
    lib = folder / "grid.dylib"
    src = Path(__file__).with_name("indirect_grid_fixture.m")
    subprocess.run(["cc", "-dynamiclib", "-fobjc-arc", "-framework", "Foundation", "-framework", "Metal",
                    str(src), "-o", str(lib)], check=True)
    encode = ctypes.CDLL(str(lib)).encode_indirect
    encode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_bool]
    encode.restype = ctypes.c_int
    return encode


@pytest.mark.parametrize("external", [False, True], ids=["gpu-producer", "input-view"])
def test_live_indirect_grid(fixture, external):
    engine = runtime.engine()
    count = mx.array([5], mx.uint32)
    grid = mx.array([0, 0, 99, 5, 1, 1], mx.uint32)[2:]
    output = mx.zeros((16,), mx.uint32)
    mx.eval(count, grid, output)
    n, descriptor, result = engine.describe([count, grid, output])
    engine.record_begin(label="indirect grid")
    try:
        assert fixture(n.ptr, descriptor.ptr, descriptor.offset + 4, result.ptr, external) == 0
    finally:
        trace = engine.record_end()
    assert trace.native.export()["unsupported"] == "indirect_dispatch_unsupported"
    assert trace.exported()["calls"]["dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:"] == 1
    ir = engine.build(trace, inputs=[descriptor if external else n], outputs=[output])
    last = ir.steps[-1]
    assert last.dispatch[0] == "indirect"
    buf, offset = last.indirect_grid
    assert offset == descriptor.offset + 4
    assert ir.buffers[buf].role == ("input" if external else "temp")
    assert ProgramIR.from_json(ir.to_json()).signature() == ir.signature()
    shifted = replace(ir, steps=[*ir.steps[:-1], replace(last, indirect_grid=(buf, offset - 4))])
    assert shifted.signature() != ir.signature()
    assert "indirect grid" in compare_structure(ir, shifted)
    if not external:
        # The consumer line stays identical. Its earlier producer now wrote a different buffer.
        clone = replace(ir.buffers[buf], id=len(ir.buffers))
        producer = ir.steps[0]
        bindings = [(slot, clone.id if buffer == buf and access == "w" else buffer, at, access)
                    for slot, buffer, at, access in producer.bindings]
        orphaned = replace(ir, buffers=[*ir.buffers, clone], steps=[replace(producer, bindings=bindings), *ir.steps[1:]])
        assert "indirect grid" in compare_structure(ir, orphaned)
    for invalid in (offset + 1, ir.buffers[buf].nbytes - 8):
        bad = replace(ir, steps=[*ir.steps[:-1], replace(last, indirect_grid=(buf, invalid))])
        with pytest.raises(runtime.BuildRefused, match="indirect grid"):
            engine.seal(bad)
    program = engine.seal(ir)
    # Different view offsets exercise relocation. Different counts prove the grid
    # came from the new data, including the legal empty expert schedule.
    for live, pad in ((2, 0), (11, 3), (0, 1), (16, 4)):
        value = mx.array([0] * pad + [99, live, 1, 1], mx.uint32)[pad:] if external else mx.array([live], mx.uint32)
        mx.eval(value)
        for barriers in ("recorded", "minimal", "named"):
            program.dispatch = "concurrent"
            program.barriers_form = barriers
            got, why = engine.try_run(program, [value])
            assert why == ""
            assert got[0].tolist() == list(range(1, live + 1)) + [0] * (16 - live)
