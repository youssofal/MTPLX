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
    # Stored codes are projected one row at a time, as a one-row step projects them (MLX's
    # 2 to 24 row affine product is a different kernel on M3 and up).
    one_row = mx.concatenate([layer.out.product(hg[:, r:r + 1]) for r in range(rows)], axis=1)
    assert mx.array_equal(conv_rows, one_row.astype(mx.float32)).item()
    assert mx.array_equal(conv_rows, want_rows).item() and mx.array_equal(partial, want_partial).item()
    assert partial.shape == (1, rows, outer) and partial.dtype == mx.float32


def test_a_replayed_step_counts_the_visits_that_take_the_packs_own_product():
    """The native engine replays a recorded step without running the stage functions, so the
    decoder counts, per replayed step, the visits each stage serves with the pack's product:
    the same per-visit rules as the stages (stage_fallbacks)."""
    dense = mx.zeros((8, 128), mx.bfloat16)

    def layer(q=dense, out=dense, down3=dense):
        controllers = [SimpleNamespace(down=dense) for _ in range(3)] + [SimpleNamespace(down=down3)]
        return SimpleNamespace(q=q, k=dense, v=dense, r=dense, sink=dense, gate=dense, out=out, experts=None,
                               convs=[SimpleNamespace(dynamic=None) for _ in range(4)], controllers=controllers)

    model = SimpleNamespace(
        layers=[layer(q=packed(64, 256, 4)),  # 4-bit/64 over 256 columns: stage A reads it in place
                layer(out=packed(64, 128, 6)), layer(q=packed(64, 128, 4), down3=packed(64, 128, 8))],
        visit_order=[0, 1, 2, 1, 2], start=1, end=3,  # layers 1-2 loop twice; messages on the second pass
        embed_tokens=SimpleNamespace(weight=dense), unembed=dense)
    assert fast_step.stage_fallbacks(model, "a,bc,d", 4) == {
        "fallback.a.packed_4bit_q": 2, "fallback.bc_output.packed_6bit": 2, "fallback.d.packed_8bit": 1}
    assert fast_step.stage_fallbacks(model, "bc,e", 4) == {"fallback.bc_output.packed_6bit": 2,
                                                          "fallback.experts.cpu": 5}
    model.unembed = packed(64, 256, 4)  # a 4-bit head: lane L1's row-stable launch takes it
    assert "fallback.head.packed_rows" not in fast_step.stage_fallbacks(model, "bc", 4)
    model.unembed = packed(64, 256, 6)  # a stored form the launch does not take: one row at a time
    assert fast_step.stage_fallbacks(model, "bc", 4)["fallback.head.packed_rows"] == 1
    assert "fallback.head.packed_rows" not in fast_step.stage_fallbacks(model, "bc", 1)
    # A 4-bit output projection runs as the row-stable launch beside the fused output: unfused.
    model.unembed = dense
    model.layers[1] = layer(out=packed(64, 256, 4))
    assert fast_step.stage_fallbacks(model, "bc", 4) == {"unfused.bc_output.packed_4bit": 2}


def test_a_packed_product_is_batched_only_where_its_rows_prove_equal_to_one_row():
    """MLX picks its affine kernel by shape and GPU: M1 and M2 keep the one-row kernel for small
    panels (row-stable), M3 to M5 use a wide one (not). The first use of a shape decides."""
    calls = []
    rows_equal = mx.arange(6, dtype=mx.float32).reshape(1, 2, 3).astype(mx.bfloat16)
    rows_differ = rows_equal + mx.array([[[0.0], [1.0]]], mx.bfloat16)

    def batched(value):
        return lambda: (calls.append("batched"), value)[1]

    def per_row():
        return (calls.append("rows"), rows_equal)[1]

    key_same, key_other = ("test", 2, "same"), ("test", 2, "other")
    fast_step._ROWS_PROVEN.pop(key_same, None)
    fast_step._ROWS_PROVEN.pop(key_other, None)
    assert mx.array_equal(fast_step.proven_rows(key_same, batched(rows_equal), per_row), rows_equal).item()
    assert mx.array_equal(fast_step.proven_rows(key_other, batched(rows_differ), per_row), rows_equal).item()
    calls.clear()
    fast_step.proven_rows(key_same, batched(rows_equal), per_row)   # proven: batched alone
    fast_step.proven_rows(key_other, batched(rows_differ), per_row)  # refused: rows alone
    assert calls == ["batched", "rows"]
    assert fast_route_counts().get("rows_proof.test.one_at_a_time", 0) >= 1


def test_wide_copy_rounds_need_every_stage_a_and_mlp_read_on_the_lane_kernels():
    dense = mx.zeros((8, 256), mx.bfloat16)

    def layer(v=dense):
        controllers = [SimpleNamespace(down=dense) for _ in range(4)]
        return SimpleNamespace(q=dense, k=dense, v=v, r=dense, sink=dense, gate=dense, out=dense, experts=None,
                               convs=[SimpleNamespace(dynamic=None) for _ in range(4)], controllers=controllers)

    native = SimpleNamespace(layers=[layer(), layer()], visit_order=[0, 1, 1], start=1, end=2,
                             embed_tokens=SimpleNamespace(weight=dense), unembed=dense)
    assert fast_step.wide_rows_exact(native, "a,bc,d,e")
    assert not fast_step.wide_rows_exact(native, "bc,e")  # reference stage A reads wide panels batched
    six_bit = SimpleNamespace(**{**vars(native), "layers": [layer(), layer(v=packed(64, 256, 6))]})
    assert not fast_step.wide_rows_exact(six_bit, "a,bc,d,e")  # a 6-bit v: stage A falls back
