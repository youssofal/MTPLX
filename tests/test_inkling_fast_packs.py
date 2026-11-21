"""Every pack takes the fast step. A visit whose weights a 16-bit lane kernel cannot read
(stored codes of an affine pack, another 16-bit type) runs that stage on the pack's own
product and counts ``fallback.<stage>.<reason>``; it never reaches a kernel that would read
the codes as a matrix (the Optimized Speed recipe packs ten attention output projections)."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.models.inkling.fast import attend as fast_attend
from mtplx.models.inkling.fast import attn_output as fast_output
from mtplx.models.inkling.fast import step as fast_step
from mtplx.models.inkling.fast.routes import fast_route_counts
from mtplx.models.inkling.precision import AffineWeight


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def packed(rows, cols, bits, seed=0):
    weight = (mx.random.normal((rows, cols), key=mx.random.key(seed)) * 0.05).astype(mx.bfloat16)
    codes, scales, biases = mx.quantize(weight, group_size=64, bits=bits)
    return AffineWeight(codes, scales, biases, bits, 64)


def test_unreadable_names_the_format_a_lane_kernel_cannot_read():
    dense = mx.zeros((8, 64), mx.bfloat16)
    assert fast_step.unreadable(mx.bfloat16, dense, None, dense) is None
    assert fast_step.unreadable(mx.bfloat16, dense, packed(64, 128, 6)) == "packed_6bit"
    assert fast_step.unreadable(mx.bfloat16, mx.zeros((8, 64), mx.float16)) == "weight_dtype"


def test_packed_output_projection_keeps_fused_attention_and_counts(monkeypatch):
    rows, inner, outer = 2, 128, 64
    hg = (mx.random.normal((1, rows, inner), key=mx.random.key(1)) * 0.5).astype(mx.bfloat16)
    x = (mx.random.normal((1, rows, outer), key=mx.random.key(2))).astype(mx.bfloat16)
    tail = mx.random.normal((1, 3, outer), key=mx.random.key(3))
    convs = [SimpleNamespace(weight=mx.random.normal((4, outer), key=mx.random.key(4 + i)) * 0.3,
                             dynamic=None) for i in range(4)]
    layer = SimpleNamespace(sliding=True, boundary=False, out=packed(outer, inner, 4, seed=9),
                            convs=convs, rel_basis=None)
    state = SimpleNamespace(window_view=lambda visit: (None, None, 0, 0), tail=lambda visit, slot: tail)
    panel = fast_step.Panel(rows, mx.array(10, mx.uint32), True)
    a = SimpleNamespace(q=None, rel=None, sink=None, gate_logits=None, k_rows=None, v_rows=None)
    monkeypatch.setattr(fast_attend, "attend", lambda *args, **kw: hg)

    def dense_only(*args, **kw):
        raise AssertionError("the 16-bit output kernel must not read stored codes")

    monkeypatch.setattr(fast_output, "attention_output", dense_only)
    before = fast_route_counts().get("fallback.bc_output.packed_4bit", 0)
    partial, conv_rows = fast_step.fused_attention(layer, a, x, state, 3, panel)
    assert fast_route_counts().get("fallback.bc_output.packed_4bit", 0) == before + 1
    # The pack's own product: the quantized projection of the gated result, the attention
    # convolution on the committed tail, the residual, exactly the reference's tail.
    want_partial, want_rows = fast_step.output_projection(layer, hg, x, state, 3, panel)
    assert mx.array_equal(conv_rows, layer.out.product(hg).astype(mx.float32)).item()
    assert mx.array_equal(conv_rows, want_rows).item() and mx.array_equal(partial, want_partial).item()
    assert partial.shape == (1, rows, outer) and partial.dtype == mx.float32
