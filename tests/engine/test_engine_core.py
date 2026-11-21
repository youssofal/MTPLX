"""The native executor on small MLX graphs: record, replay, compare bits.

Every case records one evaluation, replays it on other inputs and compares
each output bit for bit with what MLX computes for the same inputs.  No model
is loaded.  Run with pytest, or directly (``python tests/engine/test_engine_core.py``)
in an environment without pytest.
"""

from __future__ import annotations

import gc
import sys

import mlx.core as mx

from mtplx.engine import runtime
from mtplx.engine.selftest import bits, check, record

try:  # pytest is optional: the file also runs as a script
    import pytest

    pytestmark = pytest.mark.skipif(
        not runtime.available(), reason=f"native engine unavailable: {runtime.unavailable()}"
    )
except ImportError:  # pragma: no cover
    pytest = None


def _engine():
    return runtime.engine()


def test_replay_uses_the_sites_buffer_size_unless_the_operator_sets_one(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.delenv("MTPLX_NATIVE_ENGINE_SPLIT", raising=False)
    engine = runtime.Engine()
    assert engine._program_split(SimpleNamespace(split=None)) == runtime.DEFAULT_SPLIT
    assert engine._program_split(SimpleNamespace(split=0)) == 0

    monkeypatch.setenv("MTPLX_NATIVE_ENGINE_SPLIT", "128")
    assert runtime.Engine()._program_split(SimpleNamespace(split=0)) == 128


def _rand(seed: int, shape: tuple[int, ...], dtype=None):
    x = mx.random.normal(shape, key=mx.random.key(seed))
    return x.astype(dtype) if dtype is not None else x


def test_read_only_inputs_can_arrive_as_overlapping_views_of_one_buffer():
    engine = _engine()
    def multiply(a, b):
        return [a @ b.T]
    def separate(seed):
        return [_rand(seed, (4, 64)), _rand(seed + 1, (4, 64))]
    def shared(seed):
        bank = _rand(seed, (6, 64))
        return [bank[:4], bank[2:]]

    program, _, _ = record(engine, multiply, separate, name="separate_read_owners")
    assert not program.clobbered_inputs
    values = shared(4)
    mx.eval(*values)
    got, reason = engine.try_run(program, values)
    assert reason == "" and bits(got[0]) == bits(multiply(*values)[0])

    # A recorded shared slot has one address in the program: it cannot
    # supply two independent inputs after that relationship disappears.
    program, _, _ = record(engine, multiply, shared, name="shared_read_owner")
    values = separate(5)
    mx.eval(*values)
    assert engine.can_run(program, values) == "input_alias"


def _assert_replays(fn, make_inputs, *, name: str, pinned=None, min_launches: int = 1):
    engine = _engine()
    program, trace, ir = record(engine, fn, make_inputs, name=name, pinned=pinned)
    assert program.launches >= min_launches, (program.launches, min_launches)
    assert program.launches == trace.launches
    problem = check(engine, program, fn, make_inputs)
    assert problem == "", problem
    return program, trace, ir


# ------------------------------------------------------------ plain graphs


def test_elementwise_chain():
    def fn(a, b):
        y = a
        for k in range(6):
            y = mx.tanh(y * b + 0.25 * k) - mx.sigmoid(y)
        return [y, mx.maximum(y, b), mx.exp(-mx.abs(y))]

    _assert_replays(
        fn, lambda s: [_rand(s, (4, 96)), _rand(s + 50, (4, 96))], name="elementwise", min_launches=6
    )


def test_half_precision_types():
    def fn(a, b):
        y = a.astype(mx.bfloat16) * b.astype(mx.bfloat16) + a.astype(mx.bfloat16)
        z = y.astype(mx.float16) * 0.5
        return [y, z, mx.rsqrt(mx.abs(z).astype(mx.float32) + 1.0)]

    _assert_replays(fn, lambda s: [_rand(s, (4, 96)), _rand(s + 50, (4, 96))], name="half")


def test_reduction_and_softmax():
    def fn(a):
        return [
            a.sum(axis=-1),
            a.max(axis=0),
            mx.softmax(a, axis=-1),
            mx.logsumexp(a, axis=-1),
            mx.argmax(a, axis=-1),
            mx.cumsum(a, axis=-1),
        ]

    _assert_replays(fn, lambda s: [_rand(s, (6, 128))], name="reduce", min_launches=5)


def test_gather_and_take_along_axis():
    table = _rand(11, (64, 48))
    mx.eval(table)

    def fn(index, rows):
        picked = mx.take(table, index, axis=0)
        along = mx.take_along_axis(rows, mx.argsort(rows, axis=-1)[:, :4], axis=-1)
        return [picked, picked.sum(axis=0), along]

    def make(seed):
        index = mx.array([(seed * 7 + k * 3) % 64 for k in range(5)], dtype=mx.int32)
        return [index, _rand(seed, (5, 48))]

    _assert_replays(fn, make, name="gather", min_launches=3)


def test_matmul_with_pinned_weight():
    weight = _rand(21, (96, 96))
    mx.eval(weight)

    def fn(a, b):
        h = a @ weight
        return [h + b, (h * h).sum(axis=-1)]

    program, _trace, ir = _assert_replays(
        fn, lambda s: [_rand(s, (4, 96)), _rand(s + 50, (4, 96))], name="matmul", pinned=[weight]
    )
    assert any(buf.role == "pinned" for buf in ir.buffers), [b.role for b in ir.buffers]


def test_quantized_matmul():
    dense = _rand(31, (128, 256))
    wq, scales, biases = mx.quantize(dense, group_size=64, bits=4)
    mx.eval(wq, scales, biases)

    def fn(x):
        y = mx.quantized_matmul(x, wq, scales, biases, transpose=True, group_size=64, bits=4)
        return [y, y.astype(mx.float32).sum(axis=-1)]

    program, _trace, ir = _assert_replays(
        fn, lambda s: [_rand(s, (4, 256), mx.bfloat16)], name="qmm", pinned=[wq, scales, biases]
    )
    names = " ".join(ir.kernel_counts())
    assert "affine" in names or "qm" in names, names
    assert sum(1 for buf in ir.buffers if buf.role == "pinned") == 3


def test_custom_metal_kernel():
    kernel = mx.fast.metal_kernel(
        name="engine_test_scale_shift",
        input_names=["x", "s"],
        output_names=["y"],
        source="""
            uint i = thread_position_in_grid.x;
            y[i] = x[i] * s[0] + T(i % 7);
        """,
    )

    def fn(x, s):
        (y,) = kernel(
            inputs=[x, s],
            template=[("T", mx.float32)],
            grid=(x.size, 1, 1),
            threadgroup=(64, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[mx.float32],
        )
        return [y, y * 2.0]

    def make(seed):
        return [_rand(seed, (4, 64)), mx.array([1.0 + seed, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]

    program, _trace, ir = _assert_replays(fn, make, name="custom_kernel")
    assert any("engine_test_scale_shift" in step.kernel_name for step in ir.steps), ir.kernel_counts()


def test_const_limit_proves_the_pin_list():
    weight = _rand(22, (96, 96))
    mx.eval(weight)

    def fn(a):
        return [a @ weight]

    make = lambda s: [_rand(s, (4, 96))]
    engine = _engine()
    for seed in (1, 2):
        mx.eval(*fn(*make(seed)))
    inputs = make(3)
    mx.eval(*inputs)
    engine.settle(inputs)
    refs = engine.describe(inputs)
    outs = fn(*inputs)
    del inputs
    engine.record_begin(label="pins")
    mx.eval(*outs)
    trace = engine.record_end()
    # Unpinned, the weight (36,864 bytes) would travel as a constant; with a low
    # limit the build names it instead.
    assert engine.build(trace, inputs=refs, outputs=outs).bytes_by_role()["const"] >= weight.nbytes
    try:
        engine.build(trace, inputs=refs, outputs=outs, const_limit=1024)
    except runtime.BuildRefused as refused:
        assert refused.code == "undeclared_external", refused.code
        # The detail names the kernel, the argument and the Metal buffer's
        # length (MLX rounds a buffer up, so the array in it is no larger).
        assert "gemm" in refused.detail and "argument 1" in refused.detail, refused.detail
        length = int(refused.detail.split("a buffer of ")[1].split(" bytes")[0])
        assert length >= weight.nbytes, (length, weight.nbytes)
    else:
        raise AssertionError("an unpinned weight was accepted under a low constant limit")
    engine.pin([weight])
    pinned = engine.build(trace, inputs=refs, outputs=outs, const_limit=1024)
    assert pinned.bytes_by_role()["const"] == 0


# ------------------------------------------------ state, donation and slices


def test_in_place_slice_update_is_donated_and_replayed():
    def fn(state, row):
        state[2:3, :] = row * 2.0
        return [state, state[1:4].sum(axis=0)]

    make = lambda s: [_rand(s, (8, 64)), _rand(s + 50, (1, 64))]
    program, _trace, ir = _assert_replays(fn, make, name="slice_update")
    # MLX wrote into the state in place, so the program consumes input 0 and
    # output 0 is an array over the same data.
    assert 0 in program.clobbered_inputs, program.clobbered_inputs
    assert ir.output_alias.get(0) == 0, ir.output_alias

    # Handed its state in a buffer as long as the recorded one (the allocator
    # decides how much room a buffer has; a shorter one is copied first, see
    # test_a_state_leaf_that_arrives_in_a_shorter_buffer_is_served), the run
    # writes in place.
    engine = _engine()
    recorded = next(buf for buf in ir.buffers if buf.role == "input" and buf.index == 0).nbytes
    inputs = make(77)
    mx.eval(*inputs)
    inputs[0] = engine.clone_layout(inputs[0], min_length=recorded)
    before = engine.describe(inputs)[0].ptr
    outputs, reason = engine.try_run(program, inputs, extra_refs=0)
    assert reason == "", reason
    assert engine.last_run()["grown"] == 0, engine.last_run()
    assert engine.describe([outputs[0]])[0].ptr == before  # same Metal buffer: no copy was made


def test_dynamic_slice_and_dynamic_slice_update():
    def fn(bank, row, position):
        bank = mx.slice_update(bank, row, position, axes=(0,))
        window = mx.slice(bank, position, axes=(0,), slice_size=(2, 64))
        return [bank, window, window.sum()]

    def make(seed):
        return [
            _rand(seed, (16, 64)),
            _rand(seed + 50, (1, 64)),
            mx.array([(seed * 5) % 14], dtype=mx.int32),
        ]

    program, _trace, ir = _assert_replays(fn, make, name="dynamic_slice")
    # The start offset is computed by a kernel, so one program serves every position.
    assert any("compute_dynamic_offset" in step.kernel_name for step in ir.steps), ir.kernel_counts()
    problem = check(_engine(), program, fn, make, seeds=(0, 1, 2, 3, 4, 5, 6, 13))
    assert problem == "", problem


def test_compiled_function_with_captured_state():
    state = {"h": mx.zeros((4, 64)), "n": mx.array(0, dtype=mx.int32)}

    def step(x):
        state["h"] = state["h"] * 0.5 + mx.tanh(x)
        state["n"] = state["n"] + 1
        return (state["h"] * state["h"]).sum(axis=-1) + state["n"]

    compiled = mx.compile(step, inputs=state, outputs=state)

    def fn(x, h, n):
        state["h"], state["n"] = h, n
        y = compiled(x)
        return [y, state["h"], state["n"]]

    def make(seed):
        return [_rand(seed, (4, 64)), _rand(seed + 50, (4, 64)), mx.array(seed, dtype=mx.int32)]

    program, _trace, ir = _assert_replays(fn, make, name="compiled_state")
    assert len(ir.outputs) == 3


# ------------------------------------------------------------------ stability


def test_two_recordings_one_signature():
    weight = _rand(41, (64, 64))
    mx.eval(weight)

    def fn(a, b):
        h = mx.tanh(a @ weight) + b
        for _ in range(4):
            h = mx.sigmoid(h) * b + h
        return [h, h.sum(axis=-1)]

    make = lambda s: [_rand(s, (4, 64)), _rand(s + 50, (4, 64))]
    engine = _engine()
    first, _t1, _ir1 = record(engine, fn, make, name="stable", pinned=[weight])
    # Disturb MLX's allocator between the recordings: other buffers get handed out.
    junk = [_rand(k, (3 + k, 64)) * 2.0 for k in range(12)]
    mx.eval(*junk)
    del junk[::2]
    gc.collect()
    second, _t2, _ir2 = record(engine, fn, make, name="stable", warm=False, pinned=[weight])
    assert first.signature == second.signature
    assert check(engine, second, fn, make) == ""


def test_one_program_serves_both_layouts_of_an_extent_one_dimension():
    """MLX hands some kernels the stride of an extent-1 dimension, so a plain
    array and a view of a longer buffer record different argument bytes.  The
    two programs are interchangeable, and `input_layout()` tells a site when
    two recordings are not expected to hash the same."""

    def fn(a):
        return [a * 2.0, a.sum(axis=-1)]

    engine = _engine()
    parent = _rand(5, (3, 8, 32))
    mx.eval(parent)
    makers = {
        "plain": lambda seed: _rand(seed, (1, 6, 32)),
        "view": lambda seed: (parent + float(seed))[1:2, 2:8, :],
    }

    def record_on(kind):
        inputs = [makers[kind](2)]
        mx.eval(*inputs)
        refs = engine.describe(inputs)
        outs = fn(*inputs)
        engine.record_begin(label="layout")
        mx.eval(*outs)
        trace = engine.record_end()
        return engine.build(trace, inputs=refs, outputs=outs, name="layout")

    mx.eval(*fn(makers["plain"](1)))
    irs = {kind: record_on(kind) for kind in makers}
    assert irs["plain"].input_layout() != irs["view"].input_layout()
    assert record_on("plain").signature() == irs["plain"].signature()
    assert record_on("view").signature() == irs["view"].signature()
    for recorded, ir in irs.items():
        program = engine.seal(ir)
        for kind, maker in makers.items():
            for seed in (7, 8):
                reference = [maker(seed)]
                mx.eval(*reference)
                expected = [bits(x) for x in fn(*reference)]
                inputs = [maker(seed)]
                mx.eval(*inputs)
                outputs, reason = engine.try_run(program, inputs, extra_refs=0)
                assert reason == "", (recorded, kind, reason)
                assert [bits(x) for x in outputs] == expected, (recorded, kind)


def test_ir_round_trip_seals_and_replays():
    def fn(a):
        return [mx.tanh(a) * 3.0, a.sum(axis=0)]

    make = lambda s: [_rand(s, (4, 64))]
    engine = _engine()
    program, _trace, ir = record(engine, fn, make, name="roundtrip")
    from mtplx.engine.ir import ProgramIR

    again = ProgramIR.from_json(ir.to_json())
    assert again == ir and again.signature() == ir.signature()
    resealed = engine.seal(again)
    assert check(engine, resealed, fn, make) == ""


def test_every_dispatch_and_barrier_form_gives_the_same_bits():
    """Serial, hybrid and concurrent passes, with each kind of barrier."""

    weights = [_rand(60 + k, (96, 96)) * 0.1 for k in range(4)]
    mx.eval(*weights)

    def fn(state, x):
        h = x
        for _ in range(3):
            # four independent branches, then a join, then a dependent run
            a = mx.tanh(h @ weights[0])
            b = mx.sigmoid(h @ weights[1])
            c = h @ weights[2]
            d = h @ weights[3]
            h = a * b + c * d
            for _ in range(5):
                h = h * mx.rsqrt((h * h).mean(axis=-1, keepdims=True) + 1e-3)
        state[1:2, :] = h[0:1, :]
        return [state, h, state.sum(axis=0)]

    make = lambda s: [_rand(s, (6, 96)), _rand(s + 50, (4, 96))]
    engine = _engine()
    program, _trace, _ir = record(engine, fn, make, name="forms", pinned=weights)
    census = program.barriers
    assert census["minimal"] <= census["recorded"], census
    assert census["launches_in_dependent_runs"] > 0, census
    saved = (engine.dispatch, engine.barriers, engine.min_run)
    try:
        for dispatch in ("concurrent", "serial", "hybrid"):
            for barriers in ("recorded", "minimal", "named", "passes"):
                for min_run in (2, 4):
                    engine.dispatch, engine.barriers, engine.min_run = dispatch, barriers, min_run
                    problem = check(engine, program, fn, make, seeds=(3, 4))
                    assert problem == "", (dispatch, barriers, min_run, problem)
                    stats = engine.last_run()
                    assert stats["launches"] == program.launches
                    if dispatch == "serial":
                        assert stats["barriers"] == 0 and stats["passes"] == 1, stats
                    if barriers == "passes":
                        assert stats["barriers"] == 0, stats
                    if dispatch == "hybrid":
                        assert stats["passes"] > 1, stats
    finally:
        engine.dispatch, engine.barriers, engine.min_run = saved


def test_streaming_commit_gives_the_same_bits():
    def fn(a, b):
        h = a
        for k in range(40):
            h = mx.tanh(h * b + 0.01 * k)
        return [h, h.sum(axis=-1)]

    make = lambda s: [_rand(s, (4, 96)), _rand(s + 50, (4, 96))]
    engine = _engine()
    program, _trace, _ir = record(engine, fn, make, name="split")
    saved = engine.split
    try:
        for split in (1, 7, 16, 1000):
            engine.split = split
            assert check(engine, program, fn, make, seeds=(3,)) == "", split
            expected = -(-program.launches // split)
            assert engine.last_run()["command_buffers"] == expected, (split, engine.last_run())
        engine.split = 1
        program.split = 0
        assert check(engine, program, fn, make, seeds=(4,)) == ""
        assert engine.last_run()["command_buffers"] == 1
    finally:
        engine.split = saved


# ------------------------------------------------------------------ refusals


def test_undeclared_host_input_that_changes_is_caught():
    box = {}

    def fn(a):
        return [a * box["scale"]]

    def make(seed):
        # Written by the host on every call and not declared: the recording
        # can only see it as a constant.
        box["scale"] = mx.array([1.0 + seed, 2.0, 3.0, 4.0] * 16)
        return [_rand(seed, (4, 64))]

    engine = _engine()
    program, _trace, ir = record(engine, fn, make, name="hidden_input")
    assert any(buf.role in ("const", "seed") for buf in ir.buffers)
    # The check on another round is what refuses it: the bits differ.
    problem = check(engine, program, fn, make)
    assert "differs" in problem, problem
    # And a second recording does not hash the same.
    second, _t, _ir = record(engine, fn, make, name="hidden_input", warm=False)
    make(4242)
    third, _t, _ir = record(
        engine, fn, lambda s: make(s + 17), name="hidden_input", warm=False
    )
    assert third.signature != second.signature or second.signature != program.signature


def test_cpu_side_primitive_is_refused_or_caught():
    def fn(a):
        on_cpu = mx.add(a, 1.0, stream=mx.cpu)
        return [on_cpu * 2.0]

    make = lambda s: [_rand(s, (4, 64))]
    engine = _engine()
    try:
        program, _trace, _ir = record(engine, fn, make, name="cpu_inside")
    except runtime.BuildRefused as refused:
        assert refused.code in {
            "cpu_primitive",
            "undeclared_external",
            "output_not_computed",
            "undeclared_state",
        }, refused.code
        return
    problem = check(engine, program, fn, make)
    assert problem != "", "a program with CPU work inside replayed as if it were whole"


def test_shared_input_that_would_be_clobbered_is_not_replayed():
    def fn(state, row):
        state[2:3, :] = row * 2.0
        return [state]

    make = lambda s: [_rand(s, (8, 64)), _rand(s + 50, (1, 64))]
    engine = _engine()
    program, _trace, _ir = record(engine, fn, make, name="shared_state")
    assert 0 in program.clobbered_inputs

    # A second Python holder of the same array object.
    inputs = make(5)
    mx.eval(*inputs)
    keep = inputs[0]
    expected = bits(keep)
    outputs, reason = engine.try_run(program, inputs, extra_refs=0)
    assert outputs is None and reason == "state_shared", (outputs, reason)
    assert bits(keep) == expected  # untouched
    del keep

    # A second MLX holder of the same data (a view).
    inputs = make(6)
    mx.eval(*inputs)
    view = inputs[0][0:2]
    mx.eval(view)
    outputs, reason = engine.try_run(program, inputs, extra_refs=0)
    assert outputs is None and reason == "state_shared", (outputs, reason)
    del view

    # With the holders gone the same call replays.
    inputs = make(7)
    mx.eval(*inputs)
    mx.synchronize()
    outputs, reason = engine.try_run(program, inputs, extra_refs=0)
    assert reason == "", reason
    assert engine.counters()["refused"].get("state_shared", 0) >= 2


def test_wrong_shape_or_dtype_is_not_replayed():
    def fn(a):
        return [a * 2.0]

    engine = _engine()
    program, _trace, _ir = record(engine, fn, lambda s: [_rand(s, (4, 64))], name="shape")
    for bad, why in (
        ([_rand(1, (5, 64))], "input_shape"),
        ([_rand(1, (4, 64), mx.float16)], "input_dtype"),
        ([_rand(1, (64, 4)).T], "input_strides"),
        ([], "input_count"),
    ):
        if bad:
            mx.eval(*bad)
        assert engine.can_run(program, bad) == why, (why, engine.can_run(program, bad))
    lazy = [_rand(1, (4, 64)) + 1.0]
    assert engine.can_run(program, lazy) == "input_not_evaluated"


def test_stride_of_an_extent_one_dimension_is_ignored():
    """The same layout reached two ways: only the unused stride differs."""

    def fn(a):
        return [a * 2.0, a.sum(axis=-1)]

    engine = _engine()
    make = lambda s: [_rand(s, (1, 6, 32))]
    program, _trace, _ir = record(engine, fn, make, name="unit_stride")
    # A view of a longer buffer: shape (1, 6, 32) with the leading stride of
    # the parent (8 * 32), at a non-zero offset.
    parent = _rand(5, (3, 8, 32))
    mx.eval(parent)
    view = parent[1:2, 2:8, :]
    mx.eval(view)
    expected = [bits(x) for x in fn(view)]
    assert engine.can_run(program, [view], extra_refs=1) in ("", "state_shared")
    held = engine.describe([view])[0]
    assert held.strides[0] != 6 * 32 and held.offset != 0, (held.strides, held.offset)
    if program.clobbered_inputs:
        # The program writes into its input and the view shares its parent's
        # data: correctly not replayed in place.
        assert engine.can_run(program, [view]) == "state_shared"
        del parent
    outputs, reason = engine.try_run(program, [view], extra_refs=1)
    assert reason == "", reason
    assert [bits(x) for x in outputs] == expected


def test_layout_key_separates_what_launches_differently():
    engine = _engine()
    plain = _rand(1, (1, 6, 32))
    parent = _rand(2, (3, 8, 32))
    wide = _rand(3, (1, 8, 64))
    mx.eval(plain, parent, wide)
    unit_view = parent[1:2, 2:8, :]  # only the extent-1 stride differs
    strided = wide[:, 1:7, :32]  # a real stride differs
    mx.eval(unit_view, strided)
    other = _rand(9, (1, 6, 32))
    half = plain.astype(mx.float16)
    mx.eval(other, half)
    key = engine.layout_key
    assert key([plain]) == key([other])
    assert key([plain]) == key([unit_view])
    assert key([plain]) != key([strided])
    assert key([plain]) != key([half])
    assert key([plain, wide]) != key([wide, plain])
    assert key([plain]) != key([_rand(9, (1, 6, 32)) + 1.0])  # not evaluated: no layout yet


def test_pin_skips_and_reports_unevaluated_arrays():
    engine = _engine()
    ready = _rand(3, (4, 4))
    mx.eval(ready)
    lazy = _rand(4, (4, 4)) + 1.0
    assert engine.pin([ready, lazy, ready]) == [1]
    mx.eval(lazy)
    assert engine.pin([lazy]) == []


# ---------------------------------------------------- order between queues


def test_drain_order_between_mlx_queue_and_ours():
    """An input MLX is still computing must be waited for.

    The input of the replay is produced by a long chain that MLX has committed
    but not finished.  With the drain the replay reads the finished value; with
    the drain switched off (a test-only flag) it reads the buffer too early
    and the output is wrong.
    """

    def fn(a):
        return [a * 2.0 + 1.0]

    engine = _engine()
    # Recorded with the input still held, so MLX does not write into it and
    # the program only reads it.  (A program that writes into its input is
    # protected by the sole-holder test instead: MLX's unfinished work holds
    # the array, see the second half of this test.)
    held = [_rand(1, (4, 64))]
    mx.eval(*held)
    refs = engine.describe(held)
    lazy = fn(*held)
    engine.record_begin(label="drain")
    mx.eval(*lazy)
    trace = engine.record_end()
    program = engine.seal(engine.build(trace, inputs=refs, outputs=lazy, name="drain"))
    assert program.clobbered_inputs == []
    del held, lazy

    big = _rand(99, (2048, 2048)) * 0.02
    mx.eval(big)

    def slow_input(scale: float):
        # A chain of its own for every attempt. MLX hands a new array a buffer
        # an earlier one gave back, with that array's values still in it: with
        # one chain for all attempts, a read made too early found the last
        # attempt's finished values, the right ones, whenever this test was
        # the first of its process to use a buffer of that size.
        base = big * scale
        h = base
        for _ in range(60):
            h = mx.tanh(h @ base)
        return h[:4, :64] + 0.0

    # The values the chains converge to, computed the ordinary way, and kept
    # to the end so that no later array is handed one of their buffers.
    scales = (1.0, 1.03, 1.06, 1.09)
    references = {scale: slow_input(scale) for scale in scales}
    mx.eval(*references.values())
    expected = {scale: bits(reference * 2.0 + 1.0) for scale, reference in references.items()}
    assert len(set(expected.values())) == len(scales)

    def attempt(scale: float, skip_drain: bool) -> bytes:
        pending = slow_input(scale)
        mx.async_eval(pending)  # committed to MLX's queue, not finished
        reason, outputs = engine._native.run(
            program.native, [pending], False, False, skip_drain, 0, 1
        )
        assert reason == "", reason
        got = bits(outputs[0])
        mx.eval(pending)
        return got

    assert attempt(1.0, skip_drain=False) == expected[1.0]
    assert attempt(1.03, skip_drain=True) != expected[1.03], (
        "the replay did not need the drain: test too weak"
    )
    assert attempt(1.06, skip_drain=False) == expected[1.06]

    # A program that writes into its input, given an input MLX is still
    # computing: the run waits for it, then for MLX to let go of it, and
    # replays in place.
    writer, _trace, _ir = record(engine, fn, lambda s: [_rand(s, (4, 64))], name="drain_writer")
    assert writer.clobbered_inputs == [0]
    pending = [slow_input(1.09)]
    mx.async_eval(*pending)
    outputs, reason = engine.try_run(writer, pending, extra_refs=0)
    assert reason == "", reason
    assert bits(outputs[0]) == expected[1.09]
    del references


def test_signature_does_not_depend_on_which_buffer_becomes_an_output():
    """MLX may reuse, for a declared output, a buffer that was working memory
    earlier in the same evaluation.  Two recordings that differ only in that
    hash the same; a different producer for the output does not."""

    from mtplx.engine.ir import ArrayIR, BufferIR, ProgramIR, StepIR

    def program(recycled: bool, producer: int = 2) -> ProgramIR:
        # step 0: in -> t0 ; step 1: t0 -> t1 ; step 2: t1 -> out
        # recycled: the output lands in t0's buffer (dead after step 1).
        buffers = [
            BufferIR(0, "input", 256, index=0),
            BufferIR(1, "output" if recycled else "temp", 256, index=0 if recycled else -1),
            BufferIR(2, "temp", 256),
        ]
        out_buffer = 1
        if not recycled:
            buffers.append(BufferIR(3, "output", 256, index=0))
            out_buffer = 3
        grid = ("threads", (64, 1, 1), (64, 1, 1))
        steps = [
            StepIR(0, "k_a", [(0, 0, 0, "r"), (1, 1, 0, "w")], [], grid),
            StepIR(1, "k_b", [(0, 1, 0, "r"), (1, 2, 0, "w")], [], grid, barrier_before=True),
            StepIR(2, "k_c", [(0, 2 if producer == 2 else 0, 0, "r"), (1, out_buffer, 0, "w")], [],
                   grid, barrier_before=True),
        ]
        arr = dict(offset=0, shape=(64,), dtype="float32", strides=(1,), nbytes=256, data_size=64, flags=7)
        return ProgramIR(
            name="recycle",
            buffers=buffers,
            steps=steps,
            inputs=[ArrayIR(buffer=0, **arr)],
            outputs=[ArrayIR(buffer=out_buffer, **arr)],
            output_alias={},
        )

    plain, recycled = program(False), program(True)
    assert plain.check() == [] and recycled.check() == [], (plain.check(), recycled.check())
    assert plain.signature() == recycled.signature()
    assert program(False, producer=0).signature() != plain.signature()


def test_a_replay_that_does_not_finish_is_abandoned_within_its_bound():
    """No wait is unbounded: a launch that runs past the bound fails the run,
    the engine declines further runs while the GPU still holds that work, and
    serves again once it has left."""

    import time

    spin = mx.fast.metal_kernel(
        name="engine_test_spin",
        input_names=["x", "count"],
        output_names=["y"],
        source="""
            uint i = thread_position_in_grid.x;
            float acc = x[i];
            uint n = (uint)count[0];
            for (uint k = 0; k < n; ++k) {
                acc = acc * 1.0000001f + 0.0000001f * float(k & 7u);
            }
            y[i] = acc;
        """,
    )

    def fn(x, count):
        (y,) = spin(inputs=[x, count], grid=(x.size, 1, 1), threadgroup=(64, 1, 1),
                    output_shapes=[x.shape], output_dtypes=[mx.float32])
        return [y]

    def make(seed, count=64.0):
        return [_rand(seed, (4096,)), mx.array([count], dtype=mx.float32)]

    engine = _engine()
    program, _trace, _ir = record(engine, fn, lambda s: make(s), name="spin")
    problem = check(engine, program, fn, lambda s: make(s))
    assert problem == "", problem

    # How many iterations keep the GPU busy for a second and a half: found on
    # a warm kernel, best of two, so a slow first launch does not shorten it
    # (a replay that ends just after its bound would be served again at once).
    def seconds_for(iterations: float) -> float:
        best = 1e9
        for _ in range(2):
            probe = fn(*make(3, iterations))
            started = time.perf_counter()
            mx.eval(*probe)
            best = min(best, time.perf_counter() - started)
        return best

    count = 2.0e6
    seconds = seconds_for(count)
    while seconds < 0.15 and count < 1.0e9:
        count *= 4
        seconds = seconds_for(count)
    long_count = min(4.0e9, count * (1.5 / max(seconds, 1e-3)))

    # Everything the checks after the timeout need is made before the long
    # run: MLX work started while the GPU still runs the abandoned launch can
    # wait behind it, and by then the engine would have recovered.
    inputs = make(5, long_count)
    again = make(6)
    mx.eval(*inputs, *again)
    engine.set_wait_floor_ms(100)
    try:
        started = time.perf_counter()
        try:
            engine.try_run(program, inputs, extra_refs=0)
            raise AssertionError("the long replay returned")
        except runtime.EngineError as error:
            assert "failed:timeout" in str(error), error
        waited = time.perf_counter() - started
        assert waited < 0.6, f"returned after {waited:.3f} s, bound 0.1 s"
        assert engine.stuck(), "not marked stuck after the timeout"
        outputs, reason = engine.try_run(program, again, extra_refs=0)
        assert outputs is None and reason == "engine_stuck", f"served or other reason: {reason!r}"
        del again, inputs
        deadline = time.perf_counter() + 20.0
        outputs = None
        while outputs is None and time.perf_counter() < deadline:
            time.sleep(0.1)
            fresh = make(7)
            mx.eval(*fresh)
            outputs, reason = engine.try_run(program, fresh, extra_refs=0)
            del fresh
        assert outputs is not None, f"never served again: {reason}"
        assert not engine.stuck(), "still marked stuck after serving again"
    finally:
        engine.set_wait_floor_ms(2000)
    expected = fn(*make(7))
    mx.eval(*expected)
    assert bits(outputs[0]) == bits(expected[0]), "bits differ after the engine served again"
    assert engine.counters()["refused"].get("timeout", 0) >= 1, engine.counters()["refused"]


def _spin_program(engine, name: str):
    """A program of one launch whose length is set by its second input."""

    spin = mx.fast.metal_kernel(
        name=name,
        input_names=["x", "count"],
        output_names=["y"],
        source="""
            uint i = thread_position_in_grid.x;
            float acc = x[i];
            uint n = (uint)count[0];
            for (uint k = 0; k < n; ++k) {
                acc = acc * 1.0000001f + 0.0000001f * float(k & 7u);
            }
            y[i] = acc;
        """,
    )

    def fn(x, count):
        (y,) = spin(inputs=[x, count], grid=(x.size, 1, 1), threadgroup=(64, 1, 1),
                    output_shapes=[x.shape], output_dtypes=[mx.float32])
        return [y]

    def make(seed, count=64.0):
        return [_rand(seed, (4096,)), mx.array([count], dtype=mx.float32)]

    program, _trace, _ir = record(engine, fn, lambda s: make(s), name=name)
    return program, fn, make


def _spin_count(fn, make, seconds: float) -> float:
    """How many iterations keep the GPU busy for about ``seconds``."""

    import time

    def seconds_for(iterations: float) -> float:
        best = 1e9
        for _ in range(2):
            probe = fn(*make(3, iterations))
            started = time.perf_counter()
            mx.eval(*probe)
            best = min(best, time.perf_counter() - started)
        return best

    count = 2.0e6
    taken = seconds_for(count)
    while taken < 0.1 and count < 1.0e9:
        count *= 4
        taken = seconds_for(count)
    return min(4.0e9, count * (seconds / max(taken, 1e-3)))


def test_submit_and_collect_are_a_run_in_two_halves():
    """The caller can work between the two; the bits are the run's."""

    def fn(a, b):
        h = mx.tanh(a * b + 0.5)
        return [h @ h.T, h.sum(axis=0)]

    make = lambda s: [_rand(s, (8, 64)), _rand(s + 50, (8, 64))]
    engine = _engine()
    program, _trace, _ir = record(engine, fn, make, name="two_halves")
    assert engine.collect(1) == (None, "no_flight")
    for seed in (41, 42, 43):
        inputs = make(seed)
        mx.eval(*inputs)
        expected = fn(*make(seed))
        mx.eval(*expected)
        assert engine.submit(program, inputs, extra_refs=0) == ""
        assert engine.in_flight()
        # MLX work of the caller's own while the replay is on the GPU.
        other = mx.tanh(_rand(seed + 7, (32, 32))).sum()
        mx.eval(other)
        outputs, reason = engine.wait()
        assert reason == "" and outputs is not None, reason
        assert not engine.in_flight()
        assert [bits(o) for o in outputs] == [bits(e) for e in expected]

    # then_prepare: the callable runs while the replay is in flight.
    seen = []
    inputs = make(44)
    mx.eval(*inputs)
    expected = fn(*make(44))
    mx.eval(*expected)
    outputs = engine.run(program, inputs, then_prepare=lambda: seen.append(engine.in_flight()), extra_refs=0)
    assert seen == [True], seen
    assert [bits(o) for o in outputs] == [bits(e) for e in expected]


def test_idle_scratch_returns_to_mlx_and_sealed_programs_reacquire_their_sizes():
    engine = runtime.Engine(label="idle scratch test")

    def compute(x):
        return [mx.sum(mx.tanh(x), axis=-1) + mx.sum(mx.square(x), axis=-1)]

    programs = []
    for count in (128, 4096):
        make = lambda seed, rows=count: [_rand(seed, (rows, 32))]
        program, _, _ = record(engine, compute, make, name=f"scratch_{count}")
        programs.append((program, make))
    allocated = engine.counters()["pool_bytes"]
    assert allocated > 0
    assert engine.release_idle_temporaries() == allocated
    assert engine.counters()["pool_bytes"] == engine.release_idle_temporaries() == 0

    for program, make in programs + programs[:1]:
        inputs = make(71)
        expected = compute(*make(71))
        mx.eval(inputs, expected)
        assert engine.submit(program, inputs, extra_refs=0) == ""
        assert engine.release_idle_temporaries() == 0
        got, reason = engine.wait()
        assert not reason and bits(got[0]) == bits(expected[0])
        assert engine.release_idle_temporaries() > 0
        assert engine.counters()["pool_bytes"] == 0


def test_one_replay_is_in_flight_at_a_time():
    engine = _engine()
    program, fn, make = _spin_program(engine, "engine_test_spin_turn")
    long_count = _spin_count(fn, make, 0.4)
    inputs, other = make(5, long_count), make(6)
    expected = fn(*make(5, long_count))
    mx.eval(*inputs, *other, *expected)
    assert engine.submit(program, inputs, extra_refs=0) == ""
    outputs, reason = engine.try_run(program, other, extra_refs=0)
    assert outputs is None and reason == "engine_busy", reason
    assert engine.submit(program, other, extra_refs=0) == "engine_busy"
    outputs, reason = engine.wait()
    assert reason == "" and bits(outputs[0]) == bits(expected[0])
    outputs, reason = engine.try_run(program, other, extra_refs=0)
    assert reason == "" and outputs is not None, reason


def test_a_slow_replay_is_waited_for_and_is_right():
    """A replay far past 50 times its usual time is not a failure when the
    caller waits in slices: it finishes late with the right bits, and the
    engine was never stuck."""

    engine = _engine()
    program, fn, make = _spin_program(engine, "engine_test_spin_slow")
    long_count = _spin_count(fn, make, 0.6)
    inputs = make(5, long_count)
    expected = fn(*make(5, long_count))
    mx.eval(*inputs, *expected)
    before = engine.counters()["refused"].get("timeout", 0)
    looks = []
    outputs, reason = engine.run_patient(
        program, inputs, extra_refs=0, slice_ms=50, should_stop=lambda: looks.append(1) and False
    )
    assert reason == "" and outputs is not None, reason
    assert len(looks) >= 2, f"the wait looked up {len(looks)} times"
    assert bits(outputs[0]) == bits(expected[0])
    assert not engine.stuck()
    assert engine.counters()["refused"].get("timeout", 0) == before


def test_a_replay_that_is_given_up_leaves_the_engine_stuck_until_the_gpu_is_done():
    import time

    engine = _engine()
    program, fn, make = _spin_program(engine, "engine_test_spin_given_up")
    long_count = _spin_count(fn, make, 0.8)
    inputs, again = make(5, long_count), make(6)
    mx.eval(*inputs, *again)
    stop = {"now": False}
    assert engine.submit(program, inputs, extra_refs=0) == ""
    try:
        stop["now"] = True
        engine.wait(slice_ms=50, should_stop=lambda: stop["now"])
        raise AssertionError("the wait returned")
    except runtime.EngineError as error:
        assert "failed:timeout" in str(error) and "stopped" in str(error), error
    assert not engine.in_flight()
    assert engine.stuck(), "not stuck after giving up"
    outputs, reason = engine.try_run(program, again, extra_refs=0)
    assert outputs is None and reason == "engine_stuck", reason
    del again, inputs
    deadline = time.perf_counter() + 20.0
    while engine.stuck() and time.perf_counter() < deadline:
        time.sleep(0.05)
    assert not engine.stuck(), "still stuck after 20 s"
    fresh = make(7)
    mx.eval(*fresh)
    expected = fn(*make(7))
    mx.eval(*expected)
    outputs, reason = engine.try_run(program, fresh, extra_refs=0)
    assert reason == "" and bits(outputs[0]) == bits(expected[0])


def test_the_lost_replay_switch_reports_a_finished_replay_as_lost():
    """MTPLX_NATIVE_ENGINE_TEST_LOSE_REPLAY: the test aid for the path a lost
    replay takes above the engine.  Driven here through its two fields."""

    def fn(a, b):
        return [mx.tanh(a * b + 0.5).sum(axis=0)]

    make = lambda s: [_rand(s, (8, 64)), _rand(s + 50, (8, 64))]
    engine = _engine()
    program, _trace, _ir = record(engine, fn, make, name="lost_on_purpose")
    inputs = make(61)
    mx.eval(*inputs)
    # This program writes into its first input, so a replay needs the caller
    # to be that input's only holder. MLX lets go of an array it has just
    # evaluated when Metal runs the command buffer's completion handler, some
    # tens of microseconds after mx.eval returns, and the executor waits at
    # most 1 ms for that; on a loaded Mac the handler can be later and the
    # replay is declined as state_shared (12 Nov, the gate under three suites
    # and a build). So both replays here start after a synchronize, as the
    # one that must replay in test_shared_input_that_would_be_clobbered_is_not_replayed does.
    mx.synchronize()
    engine._lose_at, engine._finished, engine._lose_code = 1, 0, "command_buffer"
    try:
        engine.try_run(program, inputs, extra_refs=0)
        raise AssertionError("the replay was not reported lost")
    except runtime.ReplayLost as lost:
        assert lost.code == "command_buffer", lost.code
        assert lost.wrote_inputs is bool(program.clobbered_inputs)
        assert isinstance(lost, runtime.EngineError)
    finally:
        engine._lose_at = 0
    again = make(62)
    expected = fn(*make(62))
    mx.eval(*again, *expected)
    mx.synchronize()
    outputs, reason = engine.run_patient(program, again, extra_refs=0)
    assert reason == "" and bits(outputs[0]) == bits(expected[0])


def test_a_view_input_and_its_layout_keeping_clone():
    """A program recorded on a strided view of a longer buffer serves another
    such view and a clone that keeps the layout, and declines a plain copy."""

    def fn(v):
        return [v * 2.0 + 1.0, v.sum(axis=1)]

    def view(seed):
        parent = _rand(seed, (2, 64, 32))
        mx.eval(parent)
        cut = parent[:, :7, :]  # 7 of 64 rows of each half: not row-contiguous
        mx.eval(cut)
        return cut

    engine = _engine()
    program, _trace, ir = record(engine, fn, lambda s: [view(s)], name="view")
    assert tuple(ir.inputs[0].strides) == (2048, 32, 1), ir.inputs[0].strides

    for seed in (21, 22):
        expected = fn(view(seed))
        mx.eval(*expected)
        inputs = [view(seed)]
        outputs, reason = engine.try_run(program, inputs, extra_refs=0)
        assert reason == "", reason
        assert [bits(x) for x in outputs] == [bits(x) for x in expected]

        clone = [engine.clone_layout(view(seed))]
        assert tuple(clone[0].shape) == (2, 7, 32)
        outputs, reason = engine.try_run(program, clone, extra_refs=0)
        assert reason == "", reason
        assert [bits(x) for x in outputs] == [bits(x) for x in expected]

    plain = [mx.contiguous(view(23))]
    mx.eval(*plain)
    outputs, reason = engine.try_run(program, plain, extra_refs=0)
    assert outputs is None and reason == "input_strides", reason


def test_a_constant_that_changed_during_the_recording_is_refused():
    """Nothing recorded writes the buffer, and yet its bytes at the end of the
    recording are not those of its first bind: not a constant."""

    from mtplx.engine import builder

    def exported(changed: bool) -> dict:
        step = (0, "k_mul", True, "", [(0, 0, 0, "r"), (1, 1, 0, "r"), (2, 2, 0, "w")], [], [],
                True, (64, 1, 1), (64, 1, 1), False, "", [])
        return {
            "unsupported": "",
            "steps": [step],
            "buffers": [
                (0, 1000, 256, None, False),          # the declared input
                (1, 2000, 8, b"\x00" * 8, changed),   # read, never written, not declared
                (2, 3000, 256, None, False),          # the output
            ],
        }

    def ref(ptr):
        return builder.ArrayRef(ptr=ptr, offset=0, shape=(64,), dtype="float32", strides=(1,),
                                data_size=64, flags=7, nbytes=256, itemsize=4, buffer_length=256)

    kwargs = dict(read_buffer=lambda _i, _n: None, pin_slot=lambda _p: -1,
                  inputs=[ref(1000)], outputs=[ref(3000)])
    ir = builder.build_ir(exported(False), **kwargs)
    assert [buf.role for buf in ir.buffers] == ["input", "const", "output"]
    try:
        builder.build_ir(exported(True), **kwargs)
        raise AssertionError("built with a constant that changed")
    except builder.Refused as refused:
        assert refused.code == "const_changed", refused.code


def test_a_constant_whose_memory_an_output_reuses_keeps_its_own_bytes():
    """MLX frees a one-use scalar after its launch and may give the same buffer
    to an output computed later in the same evaluation. The reads before the
    output's first write see the constant; the output keeps its own buffer."""

    from mtplx.engine import builder

    def exported(read_while_writing: bool) -> dict:
        add = (0, "k_add", True, "", [(0, 0, 0, "r"), (1, 1, 0, "r"), (2, 2, 0, "w")], [], [],
               True, (1, 1, 1), (1, 1, 1), False, "", [])
        pick = [(0, 2, 0, "r"), (1, 1, 0, "w")] + ([(2, 1, 0, "r")] if read_while_writing else [])
        return {
            "unsupported": "",
            "steps": [add, (1, "k_pick", True, "", pick, [], [], True, (1, 1, 1), (1, 1, 1), True, "", [])],
            "buffers": [
                (0, 1000, 4, None, False),                # the declared input
                (1, 2000, 4, b"\x02\x00\x00\x00", True),  # a scalar constant, then the output
                (2, 3000, 4, None, False),                # working memory
            ],
        }

    def ref(ptr):
        return builder.ArrayRef(ptr=ptr, offset=0, shape=(1, 1), dtype="int32", strides=(1, 1),
                                data_size=1, flags=7, nbytes=4, itemsize=4, buffer_length=4)

    kwargs = dict(read_buffer=lambda _i, _n: None, pin_slot=lambda _p: -1,
                  inputs=[ref(1000)], outputs=[ref(2000)])
    ir = builder.build_ir(exported(False), **kwargs)
    assert [buf.role for buf in ir.buffers] == ["input", "const", "output", "temp"]
    assert ir.buffers[1].data == b"\x02\x00\x00\x00"
    assert ir.steps[0].bindings[1] == (1, 1, 0, "r")  # the add reads the constant's copy
    assert ir.steps[1].bindings[1] == (1, 2, 0, "w")  # the pick writes the output
    assert ir.outputs[0].buffer == 2
    # The output starts every run with the constant's bytes: a write that
    # covers only some of its elements leaves the rest as MLX's buffer had them.
    assert ir.buffers[2].data == b"\x02\x00\x00\x00"
    try:
        builder.build_ir(exported(True), **kwargs)
        raise AssertionError("built a write that may depend on the constant's bytes")
    except builder.Refused as refused:
        assert refused.code == "undeclared_state", refused.code


def test_an_output_given_bytes_starts_every_run_with_them(monkeypatch):
    """An output buffer the program spec gives bytes (an output seed) starts
    every run with them, after the fill: the elements the program leaves alone
    read as MLX's buffer had them. The recording: MLX donates a host-made array
    to a slice update that writes element 1 only; the seed is set here by hand
    (the builder sets one on the recycled-constant path, see
    test_a_constant_whose_memory_an_output_reuses_keeps_its_own_bytes)."""

    from mtplx.engine.ir import ProgramIR

    engine = _engine()
    constant = [1.0, 2.0, 3.0, 4.0]

    def fn(x):
        c = mx.array(constant, dtype=mx.float32)
        c[1:2] = (x * 2.0).sum(keepdims=True)
        return [c]

    make = lambda seed: [_rand(seed, (4,))]
    _program, _trace, ir = record(engine, fn, make, name="output_seed")
    [out] = [buf for buf in ir.buffers if buf.role == "output"]
    writes = [binding for step in ir.steps for binding in step.bindings if binding[1] == out.id]
    assert writes == [(1, out.id, 4, "w")], writes  # one launch writes element 1 and nothing else
    seeded = ProgramIR.from_json(ir.to_json())
    seeded.buffers[out.id].data = bits(mx.array(constant, dtype=mx.float32))
    program = engine.seal(seeded)
    for seed in (31, 32):
        inputs = make(seed)
        mx.eval(*inputs)
        expected = [bits(x) for x in fn(*inputs)]
        inputs = make(seed)
        mx.eval(*inputs)
        got, reason = engine.step_run(program, inputs, after_step=lambda _index: None, fill=0xAB, extra_refs=0)
        assert reason == "", reason
        assert [bits(x) for x in got] == expected
    assert check(engine, program, fn, make) == ""

    # A module that cannot seed an output refuses the program instead of
    # handing back a fresh buffer without the bytes.
    monkeypatch.setattr(runtime._NATIVE, "SUPPORTS_OUTPUT_SEEDS", False, raising=False)
    try:
        engine.seal(seeded)
        raise AssertionError("sealed an output seed on a module without them")
    except runtime.BuildRefused as refused:
        assert refused.code == "output_seeds_unsupported", refused.code


def test_a_live_constant_donated_to_an_update_keeps_the_elements_not_written():
    """S4 F1: a host-made array read by one launch, then donated to a slice
    update that writes one element and returned. The add reads the constant's
    copy; the output starts from the constant's bytes. MLX 0.32.2 copies such
    an array before the update instead of donating it, so this recording has
    nothing to seed there; the case runs where MLX donates."""

    engine = _engine()

    def fn(x):
        c = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
        s = x + c
        c[1:2] = s[:1]
        return [c, s]

    make = lambda seed: [_rand(seed, (4,))]
    program, _trace, ir = record(engine, fn, make, name="donated_constant_update")
    if not any(buf.role == "output" and buf.data is not None for buf in ir.buffers):
        if pytest is not None:
            pytest.skip("MLX copied the constant before the update: no recycled output to seed")
        return
    assert check(engine, program, fn, make) == ""
    inputs = make(31)
    mx.eval(*inputs)
    expected = [bits(x) for x in fn(*inputs)]
    inputs = make(31)
    mx.eval(*inputs)
    got, reason = engine.step_run(program, inputs, after_step=lambda _index: None, fill=0xAB, extra_refs=0)
    assert reason == "", reason
    assert [bits(x) for x in got] == expected


def test_an_input_buffer_used_as_working_memory_must_be_as_long_as_recorded():
    """A program recorded on a view of a longer buffer may write that whole
    buffer (MLX recycles a dead input's buffer as working memory inside the
    same evaluation).  Handed a plain copy of the array, it must decline:
    writing past the end of the smaller buffer is what made a first-round
    program compute wrong bits, or never return."""

    def fn(state, row):
        state[1:2, :] = row * 2.0
        return [state, state.sum(axis=0)]

    def view(seed):
        parent = _rand(seed, (64, 64))
        mx.eval(parent)
        cut = parent[:4, :]  # the first 4 of 64 rows: contiguous, in a 16 KiB buffer
        mx.eval(cut)
        del parent
        return cut

    make = lambda s: [view(s), _rand(s + 50, (1, 64))]
    engine = _engine()
    program, _trace, ir = record(engine, fn, make, name="view_written")
    assert 0 in program.clobbered_inputs, program.clobbered_inputs
    written = next(buf for buf in ir.buffers if buf.role == "input" and buf.index == 0)
    assert written.nbytes >= 64 * 64 * 4, written.nbytes

    # The same kind of view: served, same bits.
    expected = fn(*make(31))
    mx.eval(*expected)
    inputs = make(31)
    mx.eval(*inputs)
    outputs, reason = engine.try_run(program, inputs, extra_refs=0)
    del inputs
    assert reason == "", reason
    assert [bits(x) for x in outputs] == [bits(x) for x in expected]

    # A layout-keeping clone has the whole parent: served.
    inputs = make(32)
    mx.eval(*inputs)
    inputs[0] = engine.clone_layout(inputs[0])
    outputs, reason = engine.try_run(program, inputs, extra_refs=0)
    del inputs
    assert reason == "", reason

    # A plain copy of the 4 rows sits in a buffer of its own size.  The
    # program never writes that buffer: the run moves the array into a buffer
    # of the recorded length and works there.  Same bits, the caller's array
    # untouched, and the state output lives in the longer buffer.
    expected = fn(*make(33))
    mx.eval(*expected)
    inputs = make(33)
    plain = inputs[0] + 0.0
    mx.eval(plain, inputs[1])
    small = [plain, inputs[1]]
    del inputs
    before = bits(plain)
    described = engine.describe(small)
    assert described[0].buffer_length < written.nbytes, described[0].buffer_length
    grown_before = engine.counters()["grown"]
    outputs, reason = engine.try_run(program, small, extra_refs=1)
    assert reason == "", reason
    assert [bits(x) for x in outputs] == [bits(x) for x in expected]
    assert engine.last_run()["grown"] == 1, engine.last_run()
    assert engine.counters()["grown"] == grown_before + 1
    assert bits(plain) == before, "the run wrote into the caller's short array"
    assert engine.describe([outputs[0]])[0].buffer_length >= written.nbytes
    del small, plain, outputs


def _state_leaf_rewritten_fresh(shape=(36, 64)):
    """A state leaf the program rewrites into a fresh buffer every run, while
    its input buffer serves as working memory: what a verify step does with
    some of its recurrent leaves.  (36, 64) floats are 9,216 bytes: less than
    a page, so the allocator does not round the size.)"""

    def fn(state, row):
        doubled = state * 2.0  # written into the state's own buffer
        return [mx.concatenate([doubled[:, 1:], row], axis=1), doubled.sum(axis=0)]

    return shape, fn


def test_an_input_too_short_and_too_large_to_copy_is_declined():
    """The copy into a buffer of the recorded length has a size limit (8 MiB):
    past it the run is declined as before, and nothing is touched."""

    shape, fn = _state_leaf_rewritten_fresh((1536, 1536))  # 9 MiB
    nbytes = shape[0] * shape[1] * 4
    engine = _engine()

    def spared(seed):
        plain = _rand(seed, shape)
        mx.eval(plain)
        return engine.clone_layout(plain, min_length=nbytes + 8192)

    make = lambda s: [spared(s), _rand(s + 50, (shape[0], 1))]
    program, _trace, ir = record(engine, fn, make, name="leaf_far_too_short")
    written = next(buf for buf in ir.buffers if buf.role == "input" and buf.index == 0)
    assert written.clobbered and written.nbytes >= nbytes + 8192, (written.clobbered, written.nbytes)

    mx.clear_cache()
    short = _rand(41, shape) + 0.0
    row = _rand(91, (shape[0], 1))
    mx.eval(short, row)
    assert engine.describe([short])[0].buffer_length < written.nbytes
    before = bits(short)
    small = [short, row]
    grown_before = engine.counters()["grown"]
    outputs, reason = engine.try_run(program, small, extra_refs=1)
    assert outputs is None and reason == "input_buffer_small", reason
    assert engine.counters()["grown"] == grown_before
    assert bits(short) == before
    del small, short
    mx.clear_cache()


def test_a_state_leaf_rewritten_into_a_fresh_buffer_is_served_every_round():
    """Recorded with its state in a buffer that has room to spare (MLX's
    allocator hands out what its cache holds), a program must keep serving
    when its own output comes back as that input: the fresh output buffer is
    as long as the recorded input buffer.  Allocated at the output's own
    recorded length, every second round was declined (input_buffer_small)."""

    shape, fn = _state_leaf_rewritten_fresh()
    nbytes = shape[0] * shape[1] * 4
    spare = nbytes + 3072
    engine = _engine()

    def state(seed):
        plain = _rand(seed, shape)
        mx.eval(plain)
        return engine.clone_layout(plain, min_length=spare)

    make = lambda s: [state(s), _rand(s + 50, (shape[0], 1))]
    mx.clear_cache()
    program, _trace, ir = record(engine, fn, make, name="leaf_fresh")
    assert program.clobbered_inputs == [0], program.clobbered_inputs
    assert 0 not in ir.output_alias, ir.output_alias
    recorded_in = next(buf for buf in ir.buffers if buf.role == "input" and buf.index == 0).nbytes
    recorded_out = ir.buffers[ir.outputs[0].buffer].nbytes
    assert recorded_in >= spare, recorded_in
    # The case the test is about: the allocator gave the output less room.
    assert recorded_out < recorded_in, (recorded_out, recorded_in)

    reference = state(60)
    live = [state(60)]
    grown_before = engine.counters()["grown"]
    for round_index in range(8):
        row = _rand(200 + round_index, (shape[0], 1))
        mx.eval(row)
        expected = fn(reference, row)
        del reference
        mx.eval(*expected)
        inputs = [live.pop(), row]
        outputs, reason = engine.try_run(program, inputs, extra_refs=0)
        del inputs
        assert reason == "", f"round {round_index}: {reason}"
        assert [bits(x) for x in outputs] == [bits(x) for x in expected], round_index
        length = engine.describe([outputs[0]])[0].buffer_length
        assert length >= recorded_in, (round_index, length, recorded_in)
        reference = expected[0]
        live.append(outputs[0])
        del outputs, expected
    # Served as it came every round: no copy was needed.
    assert engine.counters()["grown"] == grown_before


def test_a_state_leaf_that_arrives_in_a_shorter_buffer_is_served():
    """The same program handed its state in a buffer of the array's own size
    (what MLX's path leaves after it served a round): the run copies the leaf
    into a buffer of the recorded length and serves, and the round after
    needs no copy."""

    shape, fn = _state_leaf_rewritten_fresh()
    nbytes = shape[0] * shape[1] * 4
    spare = nbytes + 3072
    engine = _engine()

    def spared(seed):
        plain = _rand(seed, shape)
        mx.eval(plain)
        return engine.clone_layout(plain, min_length=spare)

    make = lambda s: [spared(s), _rand(s + 50, (shape[0], 1))]
    program, _trace, ir = record(engine, fn, make, name="leaf_short")
    recorded_in = next(buf for buf in ir.buffers if buf.role == "input" and buf.index == 0).nbytes

    mx.clear_cache()
    short = _rand(70, shape) + 0.0
    mx.eval(short)
    assert engine.describe([short])[0].buffer_length < recorded_in
    reference = _rand(70, shape) + 0.0
    live = [short]
    del short
    grown_before = engine.counters()["grown"]
    for round_index in range(4):
        row = _rand(300 + round_index, (shape[0], 1))
        mx.eval(row)
        expected = fn(reference, row)
        del reference
        mx.eval(*expected)
        inputs = [live.pop(), row]
        outputs, reason = engine.try_run(program, inputs, extra_refs=0)
        del inputs
        assert reason == "", f"round {round_index}: {reason}"
        assert [bits(x) for x in outputs] == [bits(x) for x in expected], round_index
        assert engine.last_run()["grown"] == (1 if round_index == 0 else 0), round_index
        reference = expected[0]
        live.append(outputs[0])
        del outputs, expected
    assert engine.counters()["grown"] == grown_before + 1


def test_a_large_leaf_written_in_place_is_served_from_a_buffer_without_slack_once_relaxed():
    """A state bank larger than the copy limit, written in place (a verify
    step's new rows), recorded in a buffer with room to spare (MLX's
    allocator hands out what its cache holds): handed the same bank in a
    buffer of its own size (a restore, or MLX's path after a round it
    served), the run is declined, input_buffer_small, and no copy can
    repair it. A run on a marked copy that leaves the room past the array
    untouched relaxes the program: the same bank in its own size is then
    served, bit for bit with MLX, and the run writes in place.

    (16 KiB of room: MLX writes an array in place only when its buffer is at
    most 16 KiB longer than the array, so that is the room a bank it writes
    in place can have.)

    The night-4 real-pack rows: 98 of 154 rounds of a cold 130K prompt and 80
    of 81 of its follow-up turn left the engine this way (banks of 128 MiB)."""

    rows, width = 4608, 1024  # 9 MiB of float16: past the 8 MiB copy limit
    engine = _engine()

    def fn(state, row, at):
        state = mx.slice_update(state, row, at, axes=(0,))
        return [state, state.sum(axis=0)]

    def spared(seed):
        plain = _rand(seed, (rows, width), mx.float16)
        mx.eval(plain)
        return engine.clone_layout(plain, min_length=plain.nbytes + 16384)

    def make(seed):
        return [spared(seed), _rand(seed + 50, (2, width), mx.float16), mx.array(17 + seed % 5)]

    program, _trace, ir = record(engine, fn, make, name="large_leaf_in_place")
    assert program.clobbered_inputs == [0], program.clobbered_inputs
    assert ir.output_alias.get(0) == 0, ir.output_alias
    written = next(buf for buf in ir.buffers if buf.role == "input" and buf.index == 0)
    nbytes = rows * width * 2
    assert written.nbytes >= nbytes + 16384, written.nbytes

    def exact(seed):
        mx.clear_cache()  # a fresh allocation: exactly the array's own pages
        leaf = _rand(seed, (rows, width), mx.float16) + 0
        mx.eval(leaf)
        assert engine.describe([leaf])[0].buffer_length < written.nbytes
        return leaf

    def expected_for(seed):
        reference = [exact(seed), _rand(seed + 50, (2, width), mx.float16), mx.array(17 + seed % 5)]
        mx.eval(*reference)
        out = fn(*reference)
        del reference
        mx.eval(*out)
        return [bits(x) for x in out]

    # Before: a bank in its own size is declined, and nothing is touched.
    leaf = exact(61)
    before = bits(leaf)
    inputs = [leaf, _rand(111, (2, width), mx.float16), mx.array(18)]
    mx.eval(*inputs)
    outputs, reason = engine.try_run(program, inputs, extra_refs=1)
    assert outputs is None and reason == "input_buffer_small", reason
    assert "input 0" in engine.last_short_detail(), engine.last_short_detail()
    assert bits(leaf) == before
    del inputs, leaf

    # One run on a copy with room to spare, its room marked: untouched.
    probe = make(62)
    mx.eval(*probe)
    # (For a moment after an evaluation MLX's command buffer still holds what
    # it touched; a tail is marked only on data nobody else holds.)
    engine.settle(probe)
    marked = engine.mark_tail(probe[0], seed=1234)
    assert marked >= 16384, marked
    outputs, reason = engine.try_run(program, probe, extra_refs=0)
    assert reason == "", reason
    assert engine.tail_intact(probe[0], 1234, marked)
    del outputs, probe
    assert engine.relax_input(program, 0)
    position = ir.buffers.index(written)
    assert program.native.required[position] == nbytes, program.native.required[position]

    # After: served from its own size, in place, the same bits as MLX.
    for seed in (63, 64):
        expected = expected_for(seed)
        leaf = exact(seed)
        address = engine.describe([leaf])[0].ptr
        inputs = [leaf, _rand(seed + 50, (2, width), mx.float16), mx.array(17 + seed % 5)]
        mx.eval(*inputs)
        del leaf
        grown_before = engine.counters()["grown"]
        outputs, reason = engine.try_run(program, inputs, extra_refs=0)
        del inputs
        assert reason == "", reason
        assert [bits(x) for x in outputs] == expected
        assert engine.counters()["grown"] == grown_before  # no copy
        assert engine.describe([outputs[0]])[0].ptr == address  # written in place
        del outputs
    mx.clear_cache()


def test_a_run_that_writes_past_its_array_is_not_relaxed():
    """A program that uses an input's whole buffer as working memory writes
    the room past the array: its mark is gone, and the program keeps the
    recorded length."""

    def fn(state, row):
        doubled = state * 2.0  # MLX writes this into the state's own buffer
        return [mx.concatenate([doubled[:, 1:], row], axis=1), doubled.sum(axis=0)]

    shape = (36, 64)
    engine = _engine()

    def spared(seed):
        plain = _rand(seed, shape)
        mx.eval(plain)
        return engine.clone_layout(plain, min_length=plain.nbytes + 3072)

    make = lambda s: [spared(s), _rand(s + 50, (shape[0], 1))]
    program, _trace, ir = record(engine, fn, make, name="tail_written")
    assert program.clobbered_inputs == [0], program.clobbered_inputs
    probe = make(71)
    mx.eval(*probe)
    engine.settle(probe)
    marked = engine.mark_tail(probe[0], seed=99)
    assert marked > 0
    outputs, reason = engine.try_run(program, probe, extra_refs=0)
    assert reason == "", reason
    intact = engine.tail_intact(probe[0], 99, marked)
    del outputs, probe
    written = next(buf for buf in ir.buffers if buf.role == "input" and buf.index == 0)
    if intact:
        # This MLX wrote the working copy within the array: relaxing is right.
        assert engine.relax_input(program, 0)
    else:
        assert program.native.required[ir.buffers.index(written)] == written.nbytes


def test_a_tail_shared_with_another_array_is_never_marked():
    """The room past a view is its parent's data: never written."""

    engine = _engine()
    parent = _rand(5, (64, 64))
    mx.eval(parent)
    view = parent[:4, :]
    mx.eval(view)
    before = bits(parent)
    assert engine.mark_tail(view, seed=7) == 0
    assert bits(parent) == before


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
