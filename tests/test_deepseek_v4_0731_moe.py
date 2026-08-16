"""CPU construction contracts for the receipt-backed 0731 routed-Q2 lane."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402
from mlx_lm.models.switch_layers import QuantizedSwitchLinear  # noqa: E402

from mtplx.deepseek_v4_0731_moe import (  # noqa: E402
    DeepseekV40731PackedQ2SwitchGLU,
    build_routed_q2_pair,
    build_row_owned_combine_m1,
    exact_selfcheck_row_owned_combine_m1,
    validate_routed_q2_pair,
)


@pytest.fixture(autouse=True)
def _cpu_default_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _q2(in_features: int = 256, out_features: int = 64, experts: int = 8):
    projection = QuantizedSwitchLinear(
        in_features,
        out_features,
        experts,
        bias=False,
        group_size=128,
        bits=2,
    )
    projection.scales = projection.scales.astype(mx.bfloat16)
    projection.biases = projection.biases.astype(mx.bfloat16)
    mx.eval(projection.parameters())
    return projection


def test_routed_q2_pair_requires_exact_affine_q2_group128_storage():
    gate = _q2()
    up = _q2()

    contract = validate_routed_q2_pair(
        gate,
        up,
        hidden_size=256,
        width=64,
        experts=8,
    )

    assert contract.bits == 2
    assert contract.group_size == 128
    assert contract.hidden_size == 256
    assert contract.width == 64
    assert contract.experts == 8

    up.group_size = 64
    with pytest.raises(ValueError, match="affine Q2/group-128"):
        validate_routed_q2_pair(
            gate,
            up,
            hidden_size=256,
            width=64,
            experts=8,
        )


def test_routed_q2_pair_packs_only_output_rows_and_owns_fixed_unsorted_path():
    gate = _q2()
    up = _q2()
    down = object()
    activation = object()
    switch = SimpleNamespace(
        gate_proj=gate,
        up_proj=up,
        down_proj=down,
        activation=activation,
    )

    packed = build_routed_q2_pair(
        switch,
        hidden_size=256,
        width=64,
        experts=8,
    )

    assert type(packed) is DeepseekV40731PackedQ2SwitchGLU
    assert packed._split_at == 64
    assert packed.down_proj is down
    assert packed.activation is activation
    assert packed.gate_up_proj.weight.shape == (8, 128, 16)
    assert packed.gate_up_proj.scales.shape == (8, 128, 2)
    assert packed.gate_up_proj.biases.shape == (8, 128, 2)
    hot_source = inspect.getsource(type(packed).__call__)
    assert "moe_force_unsorted_enabled" not in hot_source
    assert "environ" not in hot_source
    assert "try:" not in hot_source


def test_row_owned_m1_combine_binds_fixed_top6_geometry_without_gpu(
    monkeypatch,
):
    calls = []

    def kernel(**kwargs):
        calls.append(kwargs)
        return (mx.zeros((1, 4096), dtype=mx.bfloat16),)

    import mtplx.deepseek_v4_0731_moe as moe

    monkeypatch.setattr(moe.mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(moe, "_row_owned_combine_m1_kernel", lambda: kernel)
    combine = build_row_owned_combine_m1(hidden_size=4096, top_k=6)

    got = combine(
        mx.zeros((1, 6, 4096), dtype=mx.bfloat16),
        mx.zeros((1, 6), dtype=mx.float32),
    )

    assert got.shape == (1, 4096)
    assert calls[0]["grid"] == (4096, 1, 1)
    assert calls[0]["threadgroup"] == (128, 1, 1)
    assert calls[0]["output_dtypes"] == [mx.bfloat16]


def test_row_owned_m1_combine_rejects_every_nonreceipt_geometry(monkeypatch):
    import mtplx.deepseek_v4_0731_moe as moe

    monkeypatch.setattr(moe.mx.metal, "is_available", lambda: True)
    with pytest.raises(ValueError, match="row-owned combine geometry"):
        build_row_owned_combine_m1(hidden_size=4096, top_k=8)
    with pytest.raises(ValueError, match="row-owned combine geometry"):
        build_row_owned_combine_m1(hidden_size=4100, top_k=6)


def test_row_owned_m1_exact_selfcheck_executes_and_rejects_mismatch():
    calls = []

    def exact(routed, route_weights):
        calls.append((routed, route_weights))
        accumulator = mx.zeros((1, 4096), dtype=mx.bfloat16)
        weights = route_weights.astype(mx.bfloat16)
        for expert in range(6):
            product = (routed[:, expert] * weights[:, expert : expert + 1]).astype(
                mx.bfloat16
            )
            accumulator = (accumulator + product).astype(mx.bfloat16)
        return accumulator

    exact_selfcheck_row_owned_combine_m1(exact)
    assert len(calls) == 1

    with pytest.raises(ValueError, match="exact self-check failed"):
        exact_selfcheck_row_owned_combine_m1(
            lambda routed, _weights: mx.zeros_like(routed[:, 0])
        )
