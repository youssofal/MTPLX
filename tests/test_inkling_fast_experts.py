"""The four-launch Inkling expert block (fast/experts.py) against RC6's own stage statements.

CPU tests pin admission. Device tests (skipped without Metal) build a synthetic block at the
real shapes with 16 experts (the bias keeps routing among them), in both expert formats, and
require RC6's bits at every width 1 to 8, row stability against one-row calls, and identical
routing on exact router ties. The real-weight proof is the campaign harness
(`outputs/inkling-fast-20261121/summary/l3/l3_check.py`).
"""
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from mtplx.models.inkling import model as M  # noqa: E402
from mtplx.models.inkling.fast import experts as fx  # noqa: E402

needs_gpu = pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")


def namespace_block(mode="nvfp4", experts=128, affine=None, router=(128, 2304), count=8):
    fp4 = mode == "nvfp4"
    pack = lambda o, k: mx.zeros((experts, o, k // (8 if fp4 else 4)), mx.uint32)
    return SimpleNamespace(
        gate_up=SimpleNamespace(mode=mode, affine=affine, codes=pack(2304, 2304)),
        down=SimpleNamespace(mode=mode, affine=None, codes=pack(2304, 1152)),
        router=mx.zeros(router, mx.float32), count=count)


def test_unserved_names_each_reason(monkeypatch):
    device = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        assert fx.unserved(namespace_block(), 1) == "cpu"
    finally:
        mx.set_default_device(device)
    if not mx.metal.is_available():
        return
    assert fx.unserved(namespace_block(), 1) is None
    assert fx.unserved(namespace_block("mxfp8"), 8) is None
    assert fx.unserved(namespace_block(), 33) == "rows"
    assert fx.unserved(namespace_block(), 0) == "rows"
    assert fx.unserved(namespace_block("affine"), 1) == "format"
    assert fx.unserved(namespace_block(affine=object()), 1) == "affine_experts"
    assert fx.unserved(namespace_block(experts=64), 1) == "geometry"
    assert fx.unserved(namespace_block(count=4), 1) == "geometry"
    monkeypatch.setenv("MTPLX_INKLING_FAST_EXPERTS", "off")
    assert fx.unserved(namespace_block(), 1) == "disabled"


def test_both_product_kernels_carry_both_formats():
    for text in (fx.GATE_UP, fx.DOWN, fx.COMBINE, fx.INPUTS):
        assert "#if FP4" in text


EXPERTS = 16


def synthetic_layer(mode, seed=7, tie=False):
    fp4 = mode == "nvfp4"
    keys = iter(mx.random.split(mx.random.key(seed), 16))
    rnd = lambda shape, lo, hi: mx.random.randint(lo, hi, shape, key=next(keys)).astype(mx.uint8)
    router = (mx.random.normal((128, 2304), key=next(keys)) * 0.02).astype(mx.bfloat16)
    bias = mx.concatenate([mx.random.normal((EXPERTS,), key=next(keys)) * 0.01,
                           mx.full((128 - EXPERTS,), -100.0)]).astype(mx.float32)
    if tie:  # experts 3 and 5 score identically: the stable order must pick 3 first, both routes
        router = mx.concatenate([router[:5], router[3:4], router[6:]])
        bias = mx.concatenate([bias[:5], bias[3:4], bias[6:]])
    w = {"mlp.gate.weight": router, "mlp.gate.bias": bias}
    for stem, (out, k) in (("w13", (2304, 2304)), ("w2", (2304, 1152))):
        name = f"mlp.experts.{stem}_weight"
        if fp4:
            w[name] = rnd((EXPERTS, out, k // 2), 0, 256)
            w[name + "_scale"] = (rnd((EXPERTS, out, k // 16), 3, 9) * 8 + rnd((EXPERTS, out, k // 16), 0, 8)).astype(mx.uint8)
            w[name + "_amax"] = mx.random.uniform(1, 10, (EXPERTS,), key=next(keys))
            w[name + "_outer_scale_denominator"] = mx.array([2688.0])
        else:
            w[name] = rnd((EXPERTS, out, k), 0, 256)
            w[name + "_scale"] = rnd((EXPERTS, out, k // 32), 118, 126)
    cfg = {"num_experts_per_tok": 8, "route_scale": 8.0, "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0}
    block = M.Experts(w, cfg, mode)
    conv = M.Convolution({"mlp_sconv.weight": (mx.random.normal((2304, 1, 4), key=next(keys)) * 0.5).astype(mx.bfloat16)},
                         "mlp_sconv")
    norm_weight = (1 + 0.1 * mx.random.normal((2304,), key=next(keys))).astype(mx.bfloat16)
    return block, conv, norm_weight


def reference(block, conv, norm_weight, source, partial, tail):
    hidden = M.norm(source, norm_weight, 1e-6).astype(mx.bfloat16)
    routed = block(hidden)
    out = conv(routed, [None, None, None, tail], 3, tape={} if source.shape[1] > 1 else None)
    return (partial + out.astype(mx.float32)).astype(mx.bfloat16), routed.astype(mx.float32)


def same(a, b):
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a.view(view), b.view(view)).item())


@needs_gpu
@pytest.mark.parametrize("mode", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("tie", [False, True])
def test_bits_and_row_stability_at_every_width(mode, tie):
    block, conv, norm_weight = synthetic_layer(mode, tie=tie)
    keys = mx.random.split(mx.random.key(11), 3)
    source = mx.random.normal((1, 8, 2304), key=keys[0]) * 3.0
    partial = mx.random.normal((1, 8, 2304), key=keys[1])
    tail = mx.random.normal((1, 3, 2304), key=keys[2])
    for W in range(1, 9):
        want = reference(block, conv, norm_weight, source[:, :W], partial[:, :W], tail)
        trace = {}
        got = fx.expert_visit_parts(block, norm_weight, conv.weight, source[:, :W], partial[:, :W], tail,
                                    eps=1e-6, trace=trace)
        assert same(got[0], want[0]), (mode, W, "output")
        assert same(got[1], want[1]), (mode, W, "convolution rows")
        if tie:
            ids = trace["experts"].reshape(W, 8)
            for r in range(W):
                row = ids[r].tolist()
                if 3 in row and 5 in row:
                    assert row.index(3) < row.index(5)
    # Row stability: row r of the 8-row panel equals a one-row call on that row's own state.
    panel = fx.expert_visit_parts(block, norm_weight, conv.weight, source, partial, tail, eps=1e-6)
    history = mx.concatenate([tail, panel[1]], axis=1)
    for r in range(8):
        single = fx.expert_visit_parts(block, norm_weight, conv.weight, source[:, r:r + 1], partial[:, r:r + 1],
                                       history[:, r:r + 3], eps=1e-6)
        assert same(single[0][0, 0], panel[0][0, r]), (mode, r)
        assert same(single[1][0, 0], panel[1][0, r]), (mode, r)
