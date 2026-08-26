from __future__ import annotations

import mlx.core as mx

from mtplx.deepseek_v4_mia_draft_graph import MiaPhysicalK5FullGraphDraftRoute
from mtplx.deepseek_v4_mia_draft_moe import install_mia_k64_packed_switch


class _CompileSpy:
    def __init__(self) -> None:
        self.functions = []

    def __call__(self, function):
        self.functions.append(function)
        return function


def test_fullgraph_route_compiles_once_and_consumes_live_cache_and_position():
    spy = _CompileSpy()

    def proposal_graph(primary, cache0, cache1, cache2, start_position):
        cache_total = cache0.sum() + cache1.sum() + cache2.sum()
        future = primary[:, None] + mx.arange(5, dtype=primary.dtype)[None]
        logits = future.astype(mx.float32) + cache_total + start_position[0]
        return future, logits

    route = MiaPhysicalK5FullGraphDraftRoute(
        proposal_graph,
        compile_fn=spy,
    )
    primary = mx.array([7], dtype=mx.uint32)
    first = route(
        primary,
        mx.array([1], dtype=mx.float32),
        mx.array([2], dtype=mx.float32),
        mx.array([3], dtype=mx.float32),
        11,
    )
    second = route(
        primary,
        mx.array([10], dtype=mx.float32),
        mx.array([20], dtype=mx.float32),
        mx.array([30], dtype=mx.float32),
        19,
    )
    mx.eval(*first, *second)

    assert len(spy.functions) == 1
    assert bool(mx.array_equal(first[0], second[0]).item())
    assert first[1].tolist() == [[24.0, 25.0, 26.0, 27.0, 28.0]]
    assert second[1].tolist() == [[86.0, 87.0, 88.0, 89.0, 90.0]]


def test_k64_packed_gate_up_is_bit_exact_and_preserves_named_modules():
    from mlx_lm.models.switch_layers import SwitchGLU
    from mlx.utils import tree_flatten

    mx.random.seed(47)
    switch = SwitchGLU(64, 32, 4, bias=False)
    switch.gate_proj = switch.gate_proj.to_quantized(
        group_size=32,
        bits=4,
        mode="mxfp4",
    )
    switch.up_proj = switch.up_proj.to_quantized(
        group_size=32,
        bits=4,
        mode="mxfp4",
    )
    switch.down_proj = switch.down_proj.to_quantized(
        group_size=32,
        bits=4,
        mode="mxfp4",
    )
    x = mx.random.normal((5, 64)).astype(mx.bfloat16)
    indices = mx.array(
        [[0, 1, 2, 3, 0, 1]] * 5,
        dtype=mx.uint32,
    )
    expected = switch(x, indices)

    packed = install_mia_k64_packed_switch(switch)
    actual = packed(x, indices)
    mx.eval(expected, actual)

    assert bool(mx.array_equal(actual, expected).item())
    modules = dict(packed.named_modules())
    assert modules["gate_proj"] is switch.gate_proj
    assert modules["up_proj"] is switch.up_proj
    assert modules["down_proj"] is switch.down_proj
    assert {name for name, _value in tree_flatten(packed.parameters())} == {
        "gate_proj.weight",
        "gate_proj.scales",
        "up_proj.weight",
        "up_proj.scales",
        "down_proj.weight",
        "down_proj.scales",
    }
