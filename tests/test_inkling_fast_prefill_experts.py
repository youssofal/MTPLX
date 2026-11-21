"""The sorted prompt expert route (expert_fusion.routed_sorted) against RC6's stock body.

Prompt chunks of more than 32 rows run the stock body: its two quantized gathers on pairs
sorted by expert, with MLX ops around them. The sorted route keeps those two gathers and the
sort exactly and replaces the glue with the fused input, route, SwiGLU and combine kernels; it
must give the stock body's bits at every chunk size, in both expert formats and both route
variants, and `experts_forward` must take it for 33 to 2,048 rows only.
"""

import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():  # pragma: no cover - CPU-only machines
    pytest.skip("the fused expert kernels need a Metal GPU", allow_module_level=True)

from mtplx.models.inkling import expert_fusion as EF  # noqa: E402
from mtplx.models.inkling import model as M  # noqa: E402

EXPERTS = 16


def synthetic_block(mode, seed=3):
    """Real shapes, 16 experts carrying every route (the bias keeps the rest out)."""
    fp4 = mode == "nvfp4"
    keys = iter(mx.random.split(mx.random.key(seed), 16))
    rnd = lambda shape, lo, hi: mx.random.randint(lo, hi, shape, key=next(keys)).astype(mx.uint8)
    router = (mx.random.normal((128, 2304), key=next(keys)) * 0.02).astype(mx.bfloat16)
    bias = mx.concatenate([mx.random.normal((EXPERTS,), key=next(keys)) * 0.01,
                           mx.full((128 - EXPERTS,), -100.0)]).astype(mx.float32)
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
    return M.Experts(w, cfg, mode)


def hidden_rows(rows, seed):
    x = mx.random.normal((1, rows, 2304), key=mx.random.key(seed)) * 3.0
    return M.norm(x, (1 + 0.1 * mx.random.normal((2304,), key=mx.random.key(seed + 1))).astype(mx.bfloat16),
                  1e-6).astype(mx.bfloat16)


def same(a, b):
    return a.shape == b.shape and bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item())


@pytest.mark.parametrize("mode", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("variant", ["fused", "scores"])
@pytest.mark.parametrize("rows", [33, 512])
def test_sorted_prompt_route_equals_the_stock_body(mode, variant, rows):
    block = synthetic_block(mode)
    hidden = hidden_rows(rows, seed=rows)
    flat = hidden.reshape(-1, 2304)
    assert EF.prompt_unserved(block, flat) is None
    want = block.stock(hidden).reshape(rows, 2304)
    got = EF.routed_sorted(block, flat, M.linear, variant=variant)
    mx.eval(want, got)
    assert same(got, want), (mode, variant, rows)


def test_prompt_admission(monkeypatch):
    block = synthetic_block("nvfp4")
    small, wide, huge = (mx.zeros((n, 2304), mx.bfloat16) for n in (32, 512, 2049))
    assert EF.prompt_unserved(block, small) == "prompt_rows"
    assert EF.prompt_unserved(block, wide) is None
    assert EF.prompt_unserved(block, huge) == "prompt_rows"
    monkeypatch.setenv("MTPLX_INKLING_PROMPT_EXPERTS", "stock")
    assert EF.prompt_unserved(block, wide) == "disabled"


def test_experts_forward_takes_the_prompt_route_and_keeps_stock_bits(monkeypatch):
    monkeypatch.delenv("MTPLX_INKLING_PROMPT_EXPERTS", raising=False)
    block = synthetic_block("nvfp4", seed=9)
    hidden = hidden_rows(512, seed=21)
    state = EF._state(block)
    before = sum(v for k, v in state.counts.items() if ".prompt_fused." in k)
    got = block(hidden)
    want = block.stock(hidden)
    mx.eval(got, want)
    assert same(got, want)
    assert sum(v for k, v in state.counts.items() if ".prompt_fused." in k) == before + 1
    # more than PROMPT_ROWS rows keep the stock body, counted
    assert EF.prompt_unserved(block, mx.zeros((EF.PROMPT_ROWS + 1, 2304), mx.bfloat16)) == "prompt_rows"
