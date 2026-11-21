"""Self-test of the native executor: ``python -m mtplx.engine.selftest``.

Records a handful of small MLX graphs, replays each on other inputs and
compares every output bit for bit with what MLX computes.  Prints one line per
case and a verdict with the GPU, so a user on any Mac can say in one paste
whether the engine works there.  Loads no model; runs in about a second.

Exit status: 0 pass, 1 a case failed, 2 the engine is unavailable.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable


def bits(array: Any) -> bytes:
    """The exact bytes of an evaluated array, whatever its type."""

    import mlx.core as mx

    if not array.size:
        return b""
    flat = mx.contiguous(array).reshape(-1)
    if flat.dtype == mx.bool_:
        flat = flat.astype(mx.uint8)
    view = flat.view(mx.uint8)
    mx.eval(view)
    # Through the buffer protocol, so the self-test needs nothing but MLX.
    return bytes(memoryview(view))


def record(
    engine: Any,
    fn: Callable[..., list],
    make_inputs: Callable[[int], list],
    *,
    name: str,
    pinned: list | None = None,
    warm: bool = True,
):
    """Record one evaluation of ``fn`` and seal it.  Returns (program, trace, ir).

    ``fn`` takes the inputs positionally and returns a list of lazy arrays.
    ``make_inputs(seed)`` builds a fresh, evaluated input list.  The inputs of
    the recorded call are dropped before the evaluation, as a real call site
    drops the state it replaces, so MLX is free to write into them in place.
    """

    import mlx.core as mx

    if pinned:
        engine.pin(pinned)
    if warm:
        # Never record a first call: it compiles kernels and allocates.
        first = make_inputs(1000)
        mx.eval(*first)
        outs = fn(*first)
        del first
        mx.eval(*outs)
        del outs
    inputs = make_inputs(1001)
    mx.eval(*inputs)
    engine.settle(inputs)
    refs = engine.describe(inputs)
    outs = fn(*inputs)
    del inputs
    engine.record_begin(label=name)
    try:
        mx.eval(*outs)
    finally:
        trace = engine.record_end()
    ir = engine.build(trace, inputs=refs, outputs=outs, name=name)
    return engine.seal(ir), trace, ir


def check(
    engine: Any,
    program: Any,
    fn: Callable[..., list],
    make_inputs: Callable[[int], list],
    *,
    seeds: tuple[int, ...] = (7, 8, 9),
    extra_refs: int = 0,
) -> str:
    """Replay on other inputs and compare with MLX.  ``""`` when all bits match."""

    import mlx.core as mx

    for seed in seeds:
        reference_inputs = make_inputs(seed)
        mx.eval(*reference_inputs)
        engine.settle(reference_inputs)
        expected = fn(*reference_inputs)
        del reference_inputs
        mx.eval(*expected)
        expected_bits = [bits(x) for x in expected]
        expected_meta = [(tuple(x.shape), str(x.dtype)) for x in expected]

        inputs = make_inputs(seed)
        mx.eval(*inputs)
        outputs, reason = engine.try_run(program, inputs, extra_refs=extra_refs)
        del inputs
        if outputs is None:
            return f"seed {seed}: not replayed ({reason})"
        if len(outputs) != len(expected_bits):
            return f"seed {seed}: {len(outputs)} outputs, expected {len(expected_bits)}"
        for index, out in enumerate(outputs):
            if (tuple(out.shape), str(out.dtype)) != expected_meta[index]:
                return f"seed {seed}: output {index} is {out.shape} {out.dtype}"
            if bits(out) != expected_bits[index]:
                return f"seed {seed}: output {index} differs"
    return ""


def _cases() -> list[tuple[str, Callable[..., list], Callable[[int], list]]]:
    import mlx.core as mx

    def rand(seed: int, shape: tuple[int, ...], dtype: Any = None) -> Any:
        x = mx.random.normal(shape, key=mx.random.key(seed))
        return x.astype(dtype) if dtype is not None else x

    weight = rand(5, (64, 64), mx.float32)
    mx.eval(weight)

    def elementwise(a, b):
        y = mx.tanh(a * b + 1.5) - mx.sigmoid(a)
        return [y * y, mx.maximum(y, b)]

    def matmul_reduce(a, b):
        h = a @ weight
        return [mx.softmax(h, axis=-1), h.sum(axis=0), mx.argmax(h + b, axis=-1)]

    def slice_update(state, row):
        state[2:3, :] = row * 2.0
        return [state, state[1:4].sum(axis=0)]

    def gather(a, index):
        return [mx.take(a, index, axis=0), a[index[0]]]

    def half(a, b):
        y = (a.astype(mx.bfloat16) * b.astype(mx.bfloat16)).astype(mx.float16)
        return [y, mx.rsqrt(mx.abs(y).astype(mx.float32) + 1.0)]

    return [
        ("elementwise", elementwise, lambda s: [rand(s, (4, 64)), rand(s + 50, (4, 64))]),
        ("matmul_softmax_reduce", matmul_reduce, lambda s: [rand(s, (4, 64)), rand(s + 50, (4, 64))]),
        ("slice_update_in_place", slice_update, lambda s: [rand(s, (8, 64)), rand(s + 50, (1, 64))]),
        (
            "gather",
            gather,
            lambda s: [rand(s, (16, 32)), mx.array([(s * 3) % 16, (s * 5) % 16, 1], dtype=mx.int32)],
        ),
        ("half_types", half, lambda s: [rand(s, (4, 64)), rand(s + 50, (4, 64))]),
    ]


def run(*, verbose: bool = True) -> dict:
    from . import runtime

    report: dict[str, Any] = {"available": False, "cases": [], "passed": False}
    if not runtime.available():
        report["unavailable"] = runtime.unavailable()
        if verbose:
            print(f"mtplx engine: UNAVAILABLE ({report['unavailable']})")
        return report
    report["available"] = True
    engine = runtime.engine()
    info = engine.info()
    report["info"] = info
    if verbose:
        print(
            f"mtplx engine {info['module_version']} on {info['gpu_name']} "
            f"({info['gpu_arch']}), macOS {info['macos']}, encoder {info['encoder_class']}"
        )
    failures = 0
    for name, fn, make_inputs in _cases():
        entry: dict[str, Any] = {"name": name}
        try:
            program, trace, ir = record(engine, fn, make_inputs, name=name)
            again, _trace, _ir = record(engine, fn, make_inputs, name=name, warm=False)
            entry["launches"] = program.launches
            entry["mlx_command_buffers"] = trace.mlx_command_buffers
            entry["clobbered_inputs"] = program.clobbered_inputs
            problem = check(engine, program, fn, make_inputs)
            if not problem and again.signature != program.signature:
                problem = "a second recording gave another signature"
            entry["last_run"] = engine.last_run()
        except Exception as exc:  # a refusal is a failure of the self-test
            problem = f"{type(exc).__name__}: {exc}"
        entry["problem"] = problem
        failures += 1 if problem else 0
        report["cases"].append(entry)
        if verbose:
            verdict = "ok  " if not problem else "FAIL"
            detail = f"{entry.get('launches', '?')} launches" if not problem else problem
            print(f"  {verdict} {name}: {detail}")
    report["passed"] = failures == 0
    report["counters"] = engine.counters()
    if verbose:
        print("mtplx engine: PASS" if report["passed"] else f"mtplx engine: FAIL ({failures} cases)")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = parser.parse_args(argv)
    report = run(verbose=not args.json)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    if not report["available"]:
        return 2
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
