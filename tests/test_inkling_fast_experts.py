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
    assert fx.unserved(namespace_block(), 1, mx.float16) is None     # the FP16 packs' stream
    assert fx.unserved(namespace_block(), 1, mx.float32) == "dtype"
    monkeypatch.setenv("MTPLX_INKLING_FAST_EXPERTS", "off")
    assert fx.unserved(namespace_block(), 1) == "disabled"


def test_both_product_kernels_carry_both_formats():
    for text in (fx.GATE_UP, fx.DOWN, fx.COMBINE, fx.INPUTS, fx.PANEL):
        assert "#if FP4" in text
    assert set(fx.PANEL_LAYOUT) == set(fx.PANEL_GEOMETRY) == {(p, f) for p in ("gu", "dn") for f in (True, False)}


def test_panel_products_serve_two_to_thirty_two_rows(monkeypatch):
    monkeypatch.setattr(fx, "PRODUCTS", "auto")
    assert [fx.products_for(w) for w in (1, 2, 4, 8, 9, 25, 32)] == ["own"] + ["panel"] * 6
    monkeypatch.setattr(fx, "PRODUCTS", "panel")
    assert [fx.products_for(w) for w in (1, 8, 9, 32)] == ["panel"] * 4
    for pinned in ("mlx", "own"):
        monkeypatch.setattr(fx, "PRODUCTS", pinned)
        assert {fx.products_for(w) for w in (1, 4, 8, 9, 32)} == {pinned}


def test_the_wide_plan_leaves_the_narrow_kernels_text_alone():
    # 2 to 8 rows keep their recorded programs: the narrow plan and panel kernels are unchanged.
    assert "PAIRS_AT" not in fx.ROUTE_PLAN and "PAIRS_AT" not in fx.PANEL
    assert fx.ROUTE_PLAN_WIDE.replace("plan[base + PAIRS_AT + before] = p;", "plan[base + 10 + before] = p;") == fx.ROUTE_PLAN
    assert fx.PANEL_WIDE.replace("ps[PAIRS_AT + m0 + m]", "ps[10 + m0 + m]") == fx.PANEL
    assert fx.PLAN_STRIDE_WIDE == 2 + 2 * fx.MAX_ROWS and fx.WIDE_PAIRS_AT == 2 + fx.MAX_ROWS


EXPERTS = 16


def synthetic_layer(mode, seed=7, tie=False, dtype=mx.bfloat16):
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
    conv = M.Convolution({"mlp_sconv.weight": (mx.random.normal((2304, 1, 4), key=next(keys)) * 0.5).astype(dtype)},
                         "mlp_sconv")
    norm_weight = (1 + 0.1 * mx.random.normal((2304,), key=next(keys))).astype(dtype)
    return block, conv, norm_weight


def reference(block, conv, norm_weight, source, partial, tail, dtype=mx.bfloat16):
    hidden = M.norm(source, norm_weight, 1e-6).astype(dtype)
    routed = block(hidden)
    out = conv(routed, [None, None, None, tail], 3, tape={} if source.shape[1] > 1 else None)
    return (partial + out.astype(mx.float32)).astype(dtype), routed.astype(mx.float32)


def same(a, b):
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a.view(view), b.view(view)).item())


@needs_gpu
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16], ids=str)
@pytest.mark.parametrize("products", ["mlx", "own", "panel"])
@pytest.mark.parametrize("mode", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("tie", [False, True])
def test_bits_and_row_stability_at_every_width(mode, tie, products, dtype, monkeypatch):
    monkeypatch.setattr(fx, "PRODUCTS", products)
    block, conv, norm_weight = synthetic_layer(mode, tie=tie, dtype=dtype)
    keys = mx.random.split(mx.random.key(11), 3)
    source = mx.random.normal((1, 8, 2304), key=keys[0]) * 3.0
    partial = mx.random.normal((1, 8, 2304), key=keys[1])
    tail = mx.random.normal((1, 3, 2304), key=keys[2])
    for W in range(1, 9):
        want = reference(block, conv, norm_weight, source[:, :W], partial[:, :W], tail, dtype)
        trace = {}
        got = fx.expert_visit_parts(block, norm_weight, conv.weight, source[:, :W], partial[:, :W], tail,
                                    eps=1e-6, trace=trace, dtype=dtype)
        assert same(got[0], want[0]), (mode, W, "output")
        assert same(got[1], want[1]), (mode, W, "convolution rows")
        if tie:
            ids = trace["experts"].reshape(W, 8)
            for r in range(W):
                row = ids[r].tolist()
                if 3 in row and 5 in row:
                    assert row.index(3) < row.index(5)
    # Row stability: row r of the 8-row panel equals a one-row call on that row's own state.
    panel = fx.expert_visit_parts(block, norm_weight, conv.weight, source, partial, tail, eps=1e-6, dtype=dtype)
    history = mx.concatenate([tail, panel[1]], axis=1)
    for r in range(8):
        single = fx.expert_visit_parts(block, norm_weight, conv.weight, source[:, r:r + 1], partial[:, r:r + 1],
                                       history[:, r:r + 3], eps=1e-6, dtype=dtype)
        assert same(single[0][0, 0], panel[0][0, r]), (mode, r)
        assert same(single[1][0, 0], panel[1][0, r]), (mode, r)


@needs_gpu
@pytest.mark.parametrize("mode", ["nvfp4", "mxfp8"])
def test_wide_panels_give_the_gathered_products_bits_and_row_stability(mode, monkeypatch):
    """9 to 32 rows (a copy round's verify) on the wide plan: the bits of the gathered route they
    replace, and every row equal to a one-row call on its own state (past 8 rows RC6's batched body
    is not the criterion: its rows differ from its own one-row step; a copy round's rows must each
    be the one-token step's). 16 experts in play, so an expert has members in most rows (up to 32
    members a slot)."""
    block, conv, norm_weight = synthetic_layer(mode)
    keys = mx.random.split(mx.random.key(29), 3)
    source = mx.random.normal((1, 32, 2304), key=keys[0]) * 3.0
    partial = mx.random.normal((1, 32, 2304), key=keys[1])
    tail = mx.random.normal((1, 3, 2304), key=keys[2])
    for W in (9, 13, 16, 24, 25, 32):
        monkeypatch.setattr(fx, "PRODUCTS", "mlx")
        gathered = fx.expert_visit_parts(block, norm_weight, conv.weight, source[:, :W], partial[:, :W], tail, eps=1e-6)
        monkeypatch.setattr(fx, "PRODUCTS", "auto")
        assert fx.products_for(W) == "panel"
        got = fx.expert_visit_parts(block, norm_weight, conv.weight, source[:, :W], partial[:, :W], tail, eps=1e-6)
        for i, name in enumerate(("output", "convolution rows")):
            assert same(got[i], gathered[i]), (mode, W, name, "gathered")
    panel = fx.expert_visit_parts(block, norm_weight, conv.weight, source, partial, tail, eps=1e-6)
    history = mx.concatenate([tail, panel[1]], axis=1)
    for r in range(32):
        single = fx.expert_visit_parts(block, norm_weight, conv.weight, source[:, r:r + 1], partial[:, r:r + 1],
                                       history[:, r:r + 3], eps=1e-6)
        assert same(single[0][0, 0], panel[0][0, r]), (mode, r)
        assert same(single[1][0, 0], panel[1][0, r]), (mode, r)


@needs_gpu
@pytest.mark.parametrize("rows", [9, 25, 32])
@pytest.mark.parametrize("pattern", ["shared", "spread"])
def test_wide_route_plan_names_every_member(rows, pattern):
    """The wide plan: every row on the same eight experts (32 members a slot at 32 rows) and rows
    spread over all 128 experts."""
    shared = [3, 17, 40, 41, 77, 90, 100, 127]
    z = mx.random.normal((rows, 128), key=mx.random.key(100 + rows)) * 0.01
    hot = mx.zeros((rows, 128))
    for r in range(rows):
        for k, e in enumerate(shared if pattern == "shared" else [(8 * r + k * 5) % 128 for k in range(8)]):
            hot[r, e] = 5.0 + 0.1 * k
    logits = (z + hot).astype(mx.float32)
    experts, prob, plan = fx._route_plan_kernel(True)(
        inputs=[logits, mx.zeros((128,), mx.float32), fx._route_consts(8.0)],
        template=[("W", rows)], grid=(256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(rows * 8,), (rows * 8,), (1 + 128 * fx.PLAN_STRIDE_WIDE,)],
        output_dtypes=[mx.uint32, mx.float32, mx.uint32],
    )
    mx.eval(experts, plan)
    pe, plan = experts.tolist(), plan.tolist()
    want = _plan_reference(pe, rows)
    assert plan[0] == len(want)
    for slot, (e, m, member_rows, pairs) in enumerate(want):
        base = 1 + slot * fx.PLAN_STRIDE_WIDE
        assert plan[base] == e and plan[base + 1] == m
        assert plan[base + 2: base + 2 + m] == member_rows
        assert plan[base + fx.WIDE_PAIRS_AT: base + fx.WIDE_PAIRS_AT + m] == pairs


def _plan_reference(pe, rows):
    """The plan the route kernel must write, from its own expert choices: distinct experts in
    ascending order, each with its members (pair p = row * 8 + k) in pair order."""
    distinct = sorted(set(pe))
    out = []
    for e in distinct:
        members = [p for p in range(rows * 8) if pe[p] == e]
        out.append((e, len(members), [p // 8 for p in members], members))
    return out


@needs_gpu
@pytest.mark.parametrize("rows", [3, 8])
@pytest.mark.parametrize("pattern", ["shared", "disjoint"])
def test_route_plan_with_every_row_on_the_same_experts_or_on_disjoint_ones(rows, pattern):
    """Review 2, F5: every row choosing the same eight experts (the case where several threads
    once stored one flag) and rows choosing disjoint experts, at an odd and the full width."""
    shared = [3, 17, 40, 41, 77, 90, 100, 127]
    z = mx.random.normal((rows, 128), key=mx.random.key(rows)) * 0.01
    hot = mx.zeros((rows, 128))
    for r in range(rows):
        for k, e in enumerate(shared if pattern == "shared" else range(8 * r, 8 * r + 8)):
            hot[r, e] = 5.0 + 0.1 * k
    logits = (z + hot).astype(mx.float32)
    experts, prob, plan = fx._route_plan_kernel()(
        inputs=[logits, mx.zeros((128,), mx.float32), fx._route_consts(8.0)],
        template=[("W", rows)], grid=(256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(rows * 8,), (rows * 8,), (1 + 128 * fx.PLAN_STRIDE,)],
        output_dtypes=[mx.uint32, mx.float32, mx.uint32],
    )
    mx.eval(experts, plan)
    pe, plan = experts.tolist(), plan.tolist()
    want = _plan_reference(pe, rows)
    assert len(want) == (8 if pattern == "shared" else 8 * rows)
    assert plan[0] == len(want)
    for slot, (e, m, member_rows, pairs) in enumerate(want):
        base = 1 + slot * fx.PLAN_STRIDE
        assert plan[base] == e and plan[base + 1] == m
        assert plan[base + 2: base + 2 + m] == member_rows
        assert plan[base + 10: base + 10 + m] == pairs
