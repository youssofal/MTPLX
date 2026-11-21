"""The mismatch diagnosis on small graphs with a known fault each.

Each case builds a fault of one kind (a host value the recording could only
bake in, a launch that computes from memory nothing wrote, a recording with a
launch the other does not make) and checks that the diagnosis names the right
launch.  No model is loaded.  Runs under pytest or as a script.
"""

from __future__ import annotations

import sys

import mlx.core as mx

from mtplx.engine import diagnose, runtime
from mtplx.engine.selftest import bits, record

try:  # pytest is optional: the file also runs as a script
    import pytest

    pytestmark = pytest.mark.skipif(
        not runtime.available(), reason=f"native engine unavailable: {runtime.unavailable()}"
    )
except ImportError:  # pragma: no cover
    pytest = None


def _engine():
    return runtime.engine()


def _rand(seed: int, shape: tuple[int, ...], dtype=None):
    x = mx.random.normal(shape, key=mx.random.key(seed))
    return x.astype(dtype) if dtype is not None else x


def _reference(fn, pristine):
    inputs = pristine()
    outputs = fn(*inputs)
    del inputs
    mx.eval(*outputs)
    return outputs


def test_clean_program_reports_nothing():
    def fn(a, b):
        h = mx.tanh(a * b + 0.5)
        return [h @ h.T, h.sum(axis=0)]

    make = lambda s: [_rand(s, (8, 64)), _rand(s + 50, (8, 64))]
    engine = _engine()
    _program, _trace, first = record(engine, fn, make, name="clean")
    _program, _trace, second = record(engine, fn, make, name="clean", warm=False)
    inputs = make(31)
    mx.eval(*inputs)
    pristine = diagnose.Pristine(inputs)
    del inputs
    reference = _reference(fn, pristine)

    static = diagnose.compare(first, second)
    assert static["same_program"], static
    report = diagnose.explain(engine, first, pristine, second=second, reference=reference)
    assert report["stored_program_against_mlx"]["differing"] == 0, report
    assert report["recording_of_this_call_against_mlx"]["differing"] == 0, report
    assert report["one_command_buffer_against_stepped"]["differing"] == 0, report
    assert report["unwritten"]["outputs_depending_on_unwritten_memory"] == 0, report
    assert report["verdict"].startswith("nothing found"), report["verdict"]


def test_host_value_baked_into_the_recording_is_named():
    box = {}

    def fn(a):
        h = mx.tanh(a) + 1.0
        scaled = h * box["scale"]  # the host rewrites this array on every call
        return [scaled.sum(axis=0), mx.exp(-mx.abs(scaled))]

    def make(seed):
        box["scale"] = mx.array([1.0 + seed, 2.0, 3.0, 4.0] * 16)
        mx.eval(box["scale"])
        return [_rand(seed, (4, 64))]

    engine = _engine()
    _program, _trace, first = record(engine, fn, make, name="baked")
    # The call that disagrees: the host value has moved on.
    inputs = make(77)
    mx.eval(*inputs)
    pristine = diagnose.Pristine(inputs)
    del inputs
    reference = _reference(fn, pristine)
    _program, _trace, second = record(
        engine, fn, lambda _s: pristine(), name="baked", warm=False
    )

    static = diagnose.compare(first, second)
    assert not static["same_program"]
    cause = static["first_cause"]
    assert cause is not None, static
    why = cause["why"][0]
    assert why["what"] == "argument_source" and why["first"][0] in ("const", "seed"), why
    multiply = cause["first"]["launch"]

    report = diagnose.explain(engine, first, pristine, second=second, reference=reference)
    assert report["stored_program_against_mlx"]["differing"] >= 1, report
    assert report["recording_of_this_call_against_mlx"]["differing"] == 0, report
    moved = report["divergence"]["first"]
    assert moved is not None and moved["first"]["launch"] == multiply, (moved, multiply)
    assert "the two programs differ" in report["verdict"], report["verdict"]
    assert f"launch {multiply}" in report["verdict"], report["verdict"]


def test_launch_that_computes_from_unwritten_memory_is_named():
    half = mx.fast.metal_kernel(
        name="diagnose_test_writes_half",
        input_names=["x"],
        output_names=["y"],
        source="""
            uint i = thread_position_in_grid.x;
            y[i] = x[i] * 3.0f;
        """,
    )

    def fn(x):
        # The kernel is launched over half of its output: the other half keeps
        # whatever the buffer held.
        (y,) = half(
            inputs=[x],
            grid=(x.size // 2, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[mx.float32],
        )
        doubled = y * 2.0  # reads all of it
        return [doubled, x + 1.0]

    make = lambda s: [_rand(s, (64,))]
    engine = _engine()
    _program, _trace, ir = record(engine, fn, make, name="unwritten")
    inputs = make(5)
    mx.eval(*inputs)
    pristine = diagnose.Pristine(inputs)
    del inputs
    reference = _reference(fn, pristine)

    found = diagnose.unwritten(engine, ir, pristine)
    assert found["outputs_depending_on_unwritten_memory"] == 1, found
    assert found["outputs"][0]["output"] == 0 and found["outputs"][0]["first_element"] == 32, found
    first = found["first_launch"]
    assert first is not None, found
    assert "Multiply" in first["launch"]["kernel"] or "multiply" in first["launch"]["kernel"].lower(), first
    # 32 float32 elements computed from memory nothing wrote
    assert first["bytes_computed_from_unwritten_memory"] > 0

    report = diagnose.explain(engine, ir, pristine, reference=reference)
    assert "never wrote" in report["verdict"], report["verdict"]
    # The elements the program does define are right either way.
    assert report.get("bytes_differing_from_mlx_that_do_not_depend_on_unwritten_memory", 0) == 0, report


def test_recordings_of_different_length_are_aligned():
    box = {"copy_first": False}

    def fn(state, row):
        if box["copy_first"]:
            # What MLX does by itself when something else still holds the
            # state: a whole copy before the write, one more launch.
            state = state * 1.0
        state[2:3, :] = row * 2.0
        return [state, state.sum(axis=0)]

    make = lambda s: [_rand(s, (8, 64)), _rand(s + 50, (1, 64))]
    engine = _engine()
    _program, _trace, in_place = record(engine, fn, make, name="lengths")
    box["copy_first"] = True
    _program, _trace, copied = record(engine, fn, make, name="lengths")
    assert len(copied.steps) == len(in_place.steps) + 1, (len(copied.steps), len(in_place.steps))

    static = diagnose.compare(in_place, copied)
    assert static["paired"] == len(in_place.steps), static
    assert static["only_first"]["count"] == 0, static["only_first"]
    assert static["only_second"]["count"] == 1, static["only_second"]
    assert any("Multiply" in name for name in static["only_second"]["kernels"]), static["only_second"]
    # No paired launch has a reason of its own to differ: the write lands in
    # the copy instead of the input, which follows from the extra launch.
    assert static["different_for_a_reason_of_their_own"] == 0, static["first_cause"]

    inputs = make(9)
    mx.eval(*inputs)
    pristine = diagnose.Pristine(inputs)
    del inputs
    reference = _reference(fn, pristine)
    report = diagnose.explain(engine, in_place, pristine, second=copied, reference=reference)
    assert report["stored_program_against_mlx"]["differing"] == 0, report
    assert report["recording_of_this_call_against_mlx"]["differing"] == 0, report
    assert report["divergence"]["outputs"]["differing"] == 0, report["divergence"]
    assert report["divergence"]["first"] is None, report["divergence"]["first"]
    assert "writes the same bytes" in report["verdict"], report["verdict"]


def test_trace_reads_what_each_launch_wrote():
    def fn(a):
        b = a * 2.0
        return [b + 1.0]

    make = lambda s: [_rand(s, (4, 16))]
    engine = _engine()
    program, _trace, ir = record(engine, fn, make, name="trace")
    inputs = make(3)
    mx.eval(*inputs)
    expected = bits(inputs[0] * 2.0 + 1.0)
    wrote, outputs, reason = diagnose.trace(engine, program, inputs)
    del inputs
    assert reason == "", reason
    assert len(wrote) == len(ir.steps)
    last = wrote[-1][0]
    assert last.data is not None and last.data[: len(expected)] == expected
    assert bits(outputs[0]) == expected


# ------------------------------------------------------------- run as script


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
    sys.exit(_main())
