"""The visit's residual step joined to the short convolution's launch (short_conv.convolve with out_dtype):
bit for bit the statements model.Layer writes after its attention and expert convolutions, at prompt and
decode widths, static and dynamic, with and without the other operand, into float32 or the stream's type."""

import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():  # pragma: no cover - CPU-only machines
    pytest.skip("the convolution kernel needs a Metal GPU", allow_module_level=True)

from mtplx.models.inkling import short_conv as SC  # noqa: E402


def same(a, b):
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a.view(view), b.view(view)).item())


@pytest.mark.parametrize("rows", [1, 37, 4096])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("join", ["attention", "attention_boundary", "experts"])
def test_the_joined_residual_step_has_the_layers_bits(rows, dynamic, join):
    width, size, heads = 2304, 4, 8
    ks = mx.random.split(mx.random.key(rows * 7 + dynamic), 6)
    x = (mx.random.normal((1, rows, width), key=ks[0]) * 3).astype(mx.bfloat16)
    prev = mx.random.normal((1, size - 1, width), key=ks[1]) * 2
    weight = (mx.random.normal((width, size), key=ks[2]) * 0.75).astype(mx.float32)
    weight = mx.contiguous(weight.T) if dynamic else weight.T
    change = (mx.random.normal((1, rows, heads * size), key=ks[3]) * 0.5).astype(mx.bfloat16) if dynamic else None
    stream = (mx.random.normal((1, rows, width), key=ks[4]) * 3).astype(mx.bfloat16)
    residual = mx.random.normal((1, rows, width), key=ks[5]) * 3
    conv, tail, _ = SC.reference(x, prev, weight, change, False)
    if join == "attention":
        add, dtype = stream, mx.float32
        want = conv.astype(mx.float32) + stream.astype(mx.float32)
    elif join == "attention_boundary":
        add, dtype = None, mx.float32
        want = conv.astype(mx.float32)
    else:
        add, dtype = residual, mx.bfloat16
        want = (residual + conv.astype(mx.float32)).astype(mx.bfloat16)
    got = SC.convolve(x, prev, weight, change, False, add, dtype)
    assert got is not None
    mx.eval(want, got[0], got[1], tail)
    assert same(got[0], want)
    assert same(got[1], tail)
