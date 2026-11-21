"""Focused tests for Inkling's fused short convolutions.

Run from the source root: ``python -m pytest -q <this file>``. With
``MTPLX_TEST_CPU_ONLY=1`` every test that needs the GPU is skipped and the rest
run on the CPU.
"""

import os

import mlx.core as mx
import pytest

if os.environ.get("MTPLX_TEST_CPU_ONLY") == "1":
    mx.set_default_device(mx.cpu)

from mtplx.models.inkling import model as inkling_model  # noqa: E402
from mtplx.models.inkling import short_conv  # noqa: E402

GPU = mx.default_device() == mx.gpu and mx.metal.is_available()
needs_gpu = pytest.mark.skipif(not GPU, reason="the fused kernel runs on the GPU only")


def _conv(dynamic, channels, dtype=mx.bfloat16):
    key = mx.random.split(mx.random.key(3), 2)
    if dynamic:
        weights = {
            "c.weight_proj": (mx.random.normal((32, 2304), key=key[0]) * 0.02).astype(dtype),
            "c.static_w": (mx.random.normal((4, channels), key=key[1]) * 0.5).astype(dtype),
        }
    else:
        weights = {"c.weight": (mx.random.normal((channels, 1, 4), key=key[0]) * 0.5).astype(dtype)}
    return inkling_model.Convolution(weights, "c", dynamic)


def _bits(a):
    return a.view({2: mx.uint16, 4: mx.uint32}[a.dtype.size])


def _same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and mx.array_equal(_bits(a), _bits(b)).item()


def _stock(conv, x, prev, drive, tape):
    os.environ[short_conv.ROUTE_ENV] = "stock"
    cache, taped = [prev], ({} if tape else None)
    out = inkling_model.Convolution.__call__(conv, x, cache, 0, drive, taped)
    return out, cache[0], taped[0] if tape else None


@pytest.mark.parametrize("dynamic,channels", [(False, 1024), (False, 2304), (True, 1024)])
@pytest.mark.parametrize("rows,tape", [(1, False), (1, True), (5, True), (9, True), (37, False)])
@pytest.mark.parametrize("first", [False, True])
def test_reference_is_the_parents_statements(dynamic, channels, rows, tape, first):
    conv = _conv(dynamic, channels)
    keys = mx.random.split(mx.random.key(rows + 11 * channels), 3)
    x = (mx.random.normal((1, rows, channels), key=keys[0]) * 3).astype(mx.bfloat16)
    prev = None if first else mx.random.normal((1, 3, channels), key=keys[1])
    drive = mx.random.normal((1, rows, 2304), key=keys[2]).astype(mx.bfloat16) if dynamic else None
    change = inkling_model.linear(drive, conv.dynamic) if dynamic else None
    want = _stock(conv, x, prev, drive, tape)
    got = short_conv.reference(x, prev, conv.weight, change, tape)
    for left, right in zip(want, got):
        assert (left is None and right is None) or _same(left, right)


def test_kernel_text_follows_the_pinned_mlx_rules():
    for tape in (False, True):
        text = short_conv.kernel_text(tape)
        assert "#include" not in text  # a header include must end in a newline; none is needed
        assert "out[(size_t(b) * rows + t) * width + c] = static_cast<T>(value);" in text
        assert ("tape[" in text) == tape  # the plain kernel has no tape output to name


def test_parent_serves_off_the_gpu_and_counts_why():
    if GPU:
        pytest.skip("the GPU takes the fused route")
    short_conv.reset_conv_route_counts()
    os.environ[short_conv.ROUTE_ENV] = "candidate"
    conv = _conv(False, 1024)
    x = mx.ones((1, 2, 1024), mx.bfloat16)
    assert short_conv.convolve(x, None, conv.weight, None, False) is None
    assert short_conv.conv_route_counts()["routes"] == {"stock:not_gpu": 1}


@needs_gpu
@pytest.mark.parametrize("dynamic,channels", [(False, 1024), (False, 2304), (True, 1024)])
@pytest.mark.parametrize("rows,tape", [(1, False), (2, True), (9, True), (37, False), (2048, False)])
@pytest.mark.parametrize("first", [False, True])
def test_fused_matches_every_bit_and_keeps_a_fresh_tail(dynamic, channels, rows, tape, first):
    conv = _conv(dynamic, channels)
    keys = mx.random.split(mx.random.key(rows + 5 * channels), 3)
    x = (mx.random.normal((1, rows, channels), key=keys[0]) * 3).astype(mx.bfloat16)
    prev = None if first else mx.random.normal((1, 3, channels), key=keys[1])
    drive = mx.random.normal((1, rows, 2304), key=keys[2]).astype(mx.bfloat16) if dynamic else None
    change = inkling_model.linear(drive, conv.dynamic) if dynamic else None
    want = _stock(conv, x, prev, drive, tape)
    os.environ[short_conv.ROUTE_ENV] = "candidate"
    short_conv.reset_conv_route_counts()
    got = short_conv.convolve(x, prev, conv.weight, change, tape)
    assert got is not None, short_conv.conv_route_counts()
    for left, right in zip(want, got):
        assert (left is None and right is None) or _same(left, right)
    assert got[1].shape == (1, 3, channels) and got[1].dtype == mx.float32


@needs_gpu
def test_partial_accept_resumes_from_the_untouched_tape():
    conv = _conv(False, 2304)
    x = (mx.random.normal((1, 12, 2304), key=mx.random.key(9)) * 3).astype(mx.bfloat16)
    prev = mx.random.normal((1, 3, 2304), key=mx.random.key(10))
    os.environ[short_conv.ROUTE_ENV] = "candidate"
    out, _, history = short_conv.convolve(x[:, :9], prev, conv.weight, None, True)
    kept = 4
    resumed = history[:, kept : kept + 3]  # the runtime's commit of a partial accept
    after, _, _ = short_conv.convolve(x[:, kept : kept + 3], resumed, conv.weight, None, False)
    whole, _, _ = short_conv.convolve(x[:, : kept + 3], prev, conv.weight, None, False)
    assert _same(out[:, :kept], whole[:, :kept])
    assert _same(after, whole[:, kept:])
