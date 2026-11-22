"""Inkling's short convolutions (four a layer visit) in one launch each, class A.

A visit runs four four-tap convolutions over a three-row float32 tail: key and
value (made dynamic on the eight full-attention visits by a per-head offset on
every tap), the attention output and the expert output. The parent writes each
as a chain of small launches: a cast, a two-part concatenation, a gather for
the tail of a long chunk, a weight copy, the depthwise convolution (or, when
dynamic, four rounds of offset, product and sum), an add, a divide and a cast.

One thread here owns one output row of one channel and repeats its parent's
arithmetic in its parent's order, so every bit matches at every width:

* static: ``acc`` starts at zero and takes the taps oldest first, each written
  ``acc += h * w`` in one statement, which is how the pinned library's
  depthwise kernel is compiled (``llvm.fmuladd``, read from its metallib). The
  other spelling (an explicit ``fma``) is the second form; the first-use proof
  keeps whichever gives the library's bits on this GPU, or neither;
* dynamic: the static weight plus the tap's projected offset, then the
  product, then the sum, newest tap first, each in its own statement and so
  rounded on its own, as the parent's separate launches round them;
* then ``(x + acc) / divisor`` (an IEEE add and divide in the parent, a float32
  divisor) and one cast to the input's type.

Because no output's arithmetic depends on how many rows a call holds, scalar
decode, a 2-9 row verify and a prefill chunk give the same bits for the same
token (the dynamic offset still comes from the caller's projection, which this
module does not change).

State is never written in place. The new tail is a fresh three-row float32
array, so a long chunk keeps nothing of its allocation alive and a scalar step
keeps three rows where the parent's view kept four. With a tape the full
history is written too, in the parent's layout, so a rejected verify restores
from the untouched pre-verify rows exactly as before.
"""

from __future__ import annotations

from contextvars import ContextVar
from functools import lru_cache
import math
import os
import struct
from threading import Lock
import time

import mlx.core as mx
from . import launch_limits

ROUTE_ENV = "MTPLX_INKLING_CONV"
#: Ways of writing the static four-tap sum; the first-use proof keeps the first that matches.
FORMS = ("float", "fused")
_THREADS = 256

_METAL = mx.metal.is_available()
_routes: dict[str, int] = {}
_proofs: dict[tuple, object] = {}
_proof_seconds: dict[tuple, float] = {}
_lock = Lock()
# The request being served, if any: its own route counts and the first-use proofs it paid for.
_request = ContextVar("inkling_convolution_request", default=None)


def _route():
    return os.environ.get(ROUTE_ENV, "candidate").lower()


def _count(name):
    _routes[name] = _routes.get(name, 0) + 1
    receipt = _request.get()
    if receipt is not None:
        receipt["routes"][name] = receipt["routes"].get(name, 0) + 1


def note_first_use(key, verdict, seconds):
    """A first-use proof's verdict and cost: process-wide, and in the request that paid for it."""
    _proof_seconds[key] = seconds
    receipt = _request.get()
    if receipt is not None:
        receipt["first_use"].append({"key": "/".join(map(str, key)), "verdict": verdict, "seconds": seconds})
        receipt["first_use_seconds"] += seconds


def conv_route_counts():
    """Selections by route since the last reset, plus every first-use proof verdict."""
    with _lock:
        return {
            "routes": dict(_routes),
            "proofs": {"/".join(map(str, key)): value for key, value in _proofs.items()},
            # What a process pays once per key: the proof and the kernels it compiles.
            "proof_seconds": {"/".join(map(str, key)): value for key, value in _proof_seconds.items()},
        }


def observe_convolution_request(fn, rt, *args, **kwargs):
    """Serve one request with its own convolution receipt in ``stats.inkling_convolution``.

    Nested calls (a request inside a request) count into the outer receipt.
    """
    if _request.get() is not None:
        return fn(rt, *args, **kwargs)
    receipt = {"mode": _route(), "routes": {}, "first_use": [], "first_use_seconds": 0.0}
    token = _request.set(receipt)
    try:
        output = fn(rt, *args, **kwargs)
        output.stats.inkling_convolution = receipt
        return output
    finally:
        _request.reset(token)


def reset_conv_route_counts():
    with _lock:
        _routes.clear()


@lru_cache(maxsize=8)
def _divisor_bits(size, dynamic):
    """The float32 the parent divides by: MLX rounds the Python float to the array's float32."""
    value = math.sqrt((2 * size + 1) if dynamic else (size + 1))
    return struct.unpack("<I", struct.pack("<f", value))[0]


_SOURCE = """
    const uint c = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    const uint b = thread_position_in_grid.z;
    const uint width = uint(x_shape[2]);
    const uint rows = uint(x_shape[1]);
    if (c >= width || t >= rows) {
        return;
    }
    constexpr uint keep = uint(SIZE - 1);
    // The four rows this output reads, oldest first, from [previous tail, new rows].
    // Inputs are read through their strides: the module's static weight is column-major.
    float h[SIZE];
    for (uint j = 0; j < uint(SIZE); ++j) {
        const uint i = t + j;
        if (i < keep) {
            if constexpr (HAS_PREV) {
                h[j] = prev[b * prev_strides[0] + i * prev_strides[1] + c * prev_strides[2]];
            } else {
                h[j] = 0.0f;
            }
        } else {
            h[j] = static_cast<float>(x[b * x_strides[0] + (i - keep) * x_strides[1] + c * x_strides[2]]);
        }
    }
    float acc = 0.0f;
    if constexpr (DYNAMIC) {
        // Newest tap first: offset the static weight, multiply, then add, each rounded on its own.
        const uint head = c / (width / uint(HEADS));
        for (uint lag = 0; lag < uint(SIZE); ++lag) {
            const float offset = static_cast<float>(
                change[b * change_strides[0] + t * change_strides[1] + (head * uint(SIZE) + lag) * change_strides[2]]);
            const float coefficient = weight[lag * weight_strides[0] + c * weight_strides[1]] + offset;
            const float product = h[keep - lag] * coefficient;
            acc = acc + product;
        }
    } else if constexpr (FORM == 0) {
        for (uint j = 0; j < uint(SIZE); ++j) {
            acc += h[j] * weight[j * weight_strides[0] + c * weight_strides[1]];
        }
    } else {
        for (uint j = 0; j < uint(SIZE); ++j) {
            acc = metal::fma(h[j], weight[j * weight_strides[0] + c * weight_strides[1]], acc);
        }
    }
    const float total = h[keep] + acc;
    const float value = total / as_type<float>(uint(DIVISOR));
    /*OUTPUT*/
    /*TAPE_ROW*/
    if (t + 1 == rows) {
        for (uint r = 0; r < keep; ++r) {
            tail[(size_t(b) * keep + r) * width + c] = h[r + 1];
            /*TAPE_TAIL*/
        }
    }
"""
# With a tape the full history is written in the parent's layout: each thread its oldest row, the last thread the tail.
_TAPE_ROW = "tape[(size_t(b) * (rows + keep) + t) * width + c] = h[0];"
_TAPE_TAIL = "tape[(size_t(b) * (rows + keep) + rows + r) * width + c] = h[r + 1];"
_OUTPUT = "out[(size_t(b) * rows + t) * width + c] = static_cast<T>(value);"
# The visit's residual step after the attention and expert convolutions (``model.Layer``): the output
# rounded to the input's type, then to float32, plus the other residual operand read through its strides
# (``RESIDUAL``), in the output type ``U`` (float32, or the stream's type for the visit's output).
_OUTPUT_RESIDUAL = """{
        float joined = static_cast<float>(static_cast<T>(value));
        if constexpr (RESIDUAL) {
            joined = joined + static_cast<float>(add[b * add_strides[0] + t * add_strides[1] + c * add_strides[2]]);
        }
        out[(size_t(b) * rows + t) * width + c] = static_cast<U>(joined);
    }"""


def kernel_text(tape, residual=False):
    """The kernel body; the tape's stores exist only in the kernel that has a tape output, and the residual
    step only in the kernel that takes one."""
    return (_SOURCE.replace("/*TAPE_ROW*/", _TAPE_ROW if tape else "")
            .replace("/*TAPE_TAIL*/", _TAPE_TAIL if tape else "")
            .replace("/*OUTPUT*/", _OUTPUT_RESIDUAL if residual else _OUTPUT))


@lru_cache(maxsize=4)
def _kernel(tape, residual=False):
    return launch_limits.kernel(
        name=("mtplx_inkling_short_conv_tape" if tape else "mtplx_inkling_short_conv") + ("_residual" if residual else ""),
        input_names=["x", "prev", "weight", "change"] + (["add"] if residual else []),
        output_names=["out", "tail", "tape"] if tape else ["out", "tail"],
        source=kernel_text(tape, residual),
        # Strided reads instead of a copy per call of the column-major static weight.
        ensure_row_contiguous=False,
    )


def _launch(x, prev, weight, change, tape, form, add=None, out_dtype=None):
    batch, rows, width = x.shape
    size = weight.shape[0]
    dynamic = change is not None
    heads = change.shape[-1] // size if dynamic else 1
    residual = out_dtype is not None
    shapes = [x.shape, (batch, size - 1, width)]
    dtypes = [out_dtype if residual else x.dtype, mx.float32]
    if tape:
        shapes.append((batch, rows + size - 1, width))
        dtypes.append(mx.float32)
    template = [
        ("T", x.dtype),
        ("SIZE", size),
        ("HEADS", heads),
        ("DYNAMIC", dynamic),
        ("HAS_PREV", prev is not None),
        ("FORM", FORMS.index(form)),
        ("DIVISOR", _divisor_bits(size, dynamic)),
    ]
    inputs = [x, weight if prev is None else prev, weight, change if dynamic else weight]
    if residual:
        template += [("U", out_dtype), ("RESIDUAL", add is not None)]
        inputs.append(x if add is None else add)
    return _kernel(bool(tape), residual)(
        # An absent tail or offset is never read; the weight stands in for its buffer.
        inputs=inputs,
        template=template,
        grid=(width, rows, batch),
        threadgroup=(min(_THREADS, width), 1, 1),
        output_shapes=shapes,
        output_dtypes=dtypes,
    )


def reference(x, prev, weight, change, tape):
    """The parent's statements (``model.Convolution``), for the proof and the probes.

    Returns ``(out, new tail, history or None)``. ``change`` is the projected
    offset (``linear(drive, weight_proj)``) or ``None`` for a static convolution.
    """
    size = weight.shape[0]
    xf = x.astype(mx.float32)
    if prev is None:
        prev = mx.zeros((x.shape[0], size - 1, x.shape[-1]), mx.float32)
    history = mx.concatenate([prev, xf], axis=1)
    if x.shape[1] > size:
        tail = mx.take(xf, mx.arange(x.shape[1] - size + 1, x.shape[1]), axis=1)
    else:
        tail = history[:, -(size - 1) :]
    if change is None:
        weighted = mx.conv1d(history, weight.T[:, :, None], groups=x.shape[-1])
        divisor = math.sqrt(size + 1)
    else:
        heads = change.shape[-1] // size
        offsets = change.astype(mx.float32).reshape(*x.shape[:2], heads, size)
        weighted = mx.zeros((*x.shape[:2], heads, x.shape[-1] // heads), mx.float32)
        for lag in range(size):
            current = history[:, size - 1 - lag : size - 1 - lag + x.shape[1]].reshape(
                weighted.shape
            )
            coefficients = weight[lag].reshape(heads, -1) + offsets[..., lag, None]
            weighted = weighted + current * coefficients
        weighted = weighted.reshape(x.shape)
        divisor = math.sqrt(2 * size + 1)
    out = ((xf + weighted) / divisor).astype(x.dtype)
    return out, tail, history if tape else None


def _same_bits(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    view = {2: mx.uint16, 4: mx.uint32}[left.dtype.size]
    return bool(mx.array_equal(left.view(view), right.view(view)).item())


def _proof_cases(x_dtype, change_dtype, width, heads, size):
    """Deterministic operands: a first chunk, a verify with a tape, a short chunk past the tap count."""
    key = mx.random.key(20261120 + width + 7 * heads)
    keys = mx.random.split(key, 8)
    weight = (mx.random.normal((width, size), key=keys[0]) * 0.75).astype(mx.float32)
    # The module's static weight is the transpose of the checkpoint's [channels, 1, taps]; the dynamic one is row-major.
    weight = mx.contiguous(weight.T) if change_dtype is not None else weight.T
    cases = []
    for at, (rows, has_prev, tape) in enumerate(((1, False, False), (5, True, True), (37, True, False))):
        x = (mx.random.normal((1, rows, width), key=keys[1 + at]) * 3.0).astype(x_dtype)
        # Zeros of both signs and a large value, so a sign or overflow difference shows.
        x = mx.concatenate([x[..., :-3], mx.array([[[0.0, -0.0, 6.0e4]] * rows], x_dtype)], axis=-1)
        prev = (
            mx.random.normal((1, size - 1, width), key=keys[4 + at]) * 2.0 if has_prev else None
        )
        change = None
        if change_dtype is not None:
            change = (mx.random.normal((1, rows, heads * size), key=keys[7]) * 0.5)[:, :rows]
            change = change.astype(change_dtype)
        cases.append((x, prev, weight, change, tape))
    return cases


def _prove(x_dtype, change_dtype, width, heads, size):
    """The first form whose every output bit matches the parent's at this layer's sizes, or why none."""
    cases = _proof_cases(x_dtype, change_dtype, width, heads, size)
    wants = [reference(*case) for case in cases]
    mx.eval(wants)
    forms = FORMS[:1] if change_dtype is not None else FORMS  # the dynamic sum has one spelling
    why = "no form ran"
    for form in forms:
        wrong = []
        for case, want in zip(cases, wants):
            got = _launch(*case, form)
            mx.eval(got)
            names = ("output", "tail", "history")
            for name, left, right in zip(names, want, got):
                if left is not None and not _same_bits(left, right):
                    wrong.append(f"{name} at {case[0].shape[1]} rows")
        if not wrong:
            return form
        why = f"form {form}: {', '.join(wrong)} differ from the parent's launches"
    return why


def _proven(x_dtype, change_dtype, width, heads, size):
    key = (str(x_dtype), str(change_dtype), width, heads, size)
    with _lock:
        verdict = _proofs.get(key)
        if verdict is None:
            start = time.perf_counter()
            try:
                verdict = _prove(x_dtype, change_dtype, width, heads, size)
            except Exception as exc:  # noqa: BLE001 - a kernel this GPU refuses leaves the parent serving, with its reason
                verdict = f"{type(exc).__name__}: {exc}"
            _proofs[key] = verdict
            note_first_use(key, verdict, time.perf_counter() - start)
        return verdict


def convolve(x, prev, weight, change, tape, add=None, out_dtype=None):
    """``(out, new tail, history or None)`` from one launch, or ``None`` when the parent must serve.

    ``x`` is the new rows ``[batch, rows, channels]``; ``prev`` the float32
    three-row tail or ``None`` before the first chunk; ``weight`` the float32
    ``[taps, channels]`` the module holds; ``change`` the projected dynamic
    offset ``[batch, rows, heads * taps]`` or ``None``; ``tape`` whether the
    caller keeps the full history for a partial accept.

    With ``out_dtype`` the launch also takes the visit's residual step:
    ``out`` is ``(conv.astype(float32) + add.astype(float32)).astype(out_dtype)``
    (without ``add``, ``conv.astype(out_dtype)`` through float32), the
    statements ``model.Layer`` writes after its attention and expert
    convolutions, each value rounded where they round it.
    """
    route = _route()
    if route != "candidate":
        _count("stock:" + ("chosen" if route == "stock" else "unknown_route"))
        return None
    if not _METAL or mx.default_device() != mx.gpu:
        _count("stock:not_gpu")
        return None
    if (
        x.ndim != 3
        or x.dtype not in (mx.bfloat16, mx.float16, mx.float32)
        or not x.shape[1]
        or weight.ndim != 2
        or weight.dtype != mx.float32
        or weight.shape[0] < 2
        or weight.shape[1] != x.shape[2]
    ):
        _count("stock:shape")
        return None
    size, width = weight.shape
    if prev is not None and (prev.dtype != mx.float32 or prev.shape != (x.shape[0], size - 1, width)):
        _count("stock:tail")
        return None
    heads = 1
    if change is not None:
        heads = change.shape[-1] // size if change.ndim == 3 else 0
        if (
            change.ndim != 3
            or change.shape[:2] != x.shape[:2]
            or not heads
            or change.shape[-1] != heads * size
            or width % heads
            or change.dtype not in (mx.bfloat16, mx.float16, mx.float32)
        ):
            _count("stock:offset")
            return None
    form = _proven(x.dtype, None if change is None else change.dtype, width, heads, size)
    if form not in FORMS:
        _count("stock:unproven")
        return None
    if out_dtype is not None and (
        out_dtype not in (mx.bfloat16, mx.float16, mx.float32)
        or (add is not None and (add.shape != x.shape or add.dtype not in (mx.bfloat16, mx.float16, mx.float32)))
    ):
        _count("stock:residual")
        return None
    _count("fused:" + ("dynamic" if change is not None else "static") + (":residual" if out_dtype is not None else ""))
    result = _launch(x, prev, weight, change, tape, form, add, out_dtype)
    return result[0], result[1], result[2] if tape else None
