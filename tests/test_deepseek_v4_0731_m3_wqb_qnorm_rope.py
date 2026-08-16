"""CPU contracts for the receipt-backed pre-geometry fixed-M3 WQB fusion."""

from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_default_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _projection():
    return SimpleNamespace(
        bits=6,
        group_size=128,
        mode="affine",
        bias=None,
        weight=SimpleNamespace(shape=(32768, 192), dtype=mx.uint32),
        scales=SimpleNamespace(shape=(32768, 8), dtype=mx.bfloat16),
        biases=SimpleNamespace(shape=(32768, 8), dtype=mx.bfloat16),
    )


def _layers():
    layers = []
    for index in range(43):

        def stock(qr, cos, sin, *, layer_index=index):
            return ("stock", layer_index, qr, cos, sin)

        layers.append(
            SimpleNamespace(
                attn=SimpleNamespace(
                    wq_b=_projection(),
                    _q_projection_qhead_route=stock,
                )
            )
        )
    return layers


def test_contract_binds_only_pre_geometry_q6_g128_wqb(monkeypatch):
    from mtplx import deepseek_v4_0731_m3_wqb_qnorm_rope as candidate

    calls = []
    candidate._build_kernel.cache_clear()
    monkeypatch.setattr(
        candidate.mx.fast,
        "metal_kernel",
        lambda **kwargs: calls.append(kwargs) or object(),
    )

    bound = candidate.build_0731_m3_wqb_qnorm_rope(_projection())

    assert bound.input_shape == (1, 3, 1024)
    assert bound.output_shape == (1, 3, 64, 512)
    assert bound.grid == (64 * 256, 1, 1)
    assert bound.threadgroup == (256, 1, 1)
    assert calls[0]["input_names"] == [
        "x",
        "w",
        "scales",
        "biases",
        "cos",
        "sin",
    ]
    assert calls[0]["output_names"] == ["output"]
    assert calls[0]["ensure_row_contiguous"] is False
    assert not hasattr(candidate, "m3_wqb_qhead_geometry_variants")


def test_contract_rejects_nonreceipt_storage():
    from mtplx import deepseek_v4_0731_m3_wqb_qnorm_rope as candidate

    projection = _projection()
    projection.bits = 8
    with pytest.raises(candidate.M3WQBNormRopeContractError, match="Q6/G128"):
        candidate.validate_0731_m3_wqb_qnorm_rope(projection)

    projection = _projection()
    projection.weight.shape = (32768, 191)
    with pytest.raises(candidate.M3WQBNormRopeContractError, match="packed wq_b"):
        candidate.validate_0731_m3_wqb_qnorm_rope(projection)


def test_source_preserves_pre_geometry_qmv_norm_and_rope_arithmetic():
    from mtplx import deepseek_v4_0731_m3_wqb_qnorm_rope as candidate

    assert candidate.RECORDED_PRE_GEOMETRY_SOURCE_SHA256 == (
        "2eb4ce3d5bae9c9b71574d17fedfd37b6755d94299cb4ed6b01d2015c5f8f9a1"
    )
    assert candidate.RECORDED_PRE_GEOMETRY_TEST_SHA256 == (
        "0c551aa8d7f865454d3a9b6d22f46af5115e00a2fc48c5db82225516c2a77cbb"
    )
    source = candidate.m3_wqb_qnorm_rope_metal_source()
    for fragment in (
        "constexpr uint M = 3;",
        "constexpr uint K = 1024;",
        "constexpr uint N = 32768;",
        "constexpr uint HEADS = 64;",
        "constexpr uint HEAD_DIM = 512;",
        "constexpr uint ROPE_DIM = 64;",
        "float result0[RESULTS_PER_SIMDGROUP] = {0.0f};",
        "float result1[RESULTS_PER_SIMDGROUP] = {0.0f};",
        "float result2[RESULTS_PER_SIMDGROUP] = {0.0f};",
        "constexpr uint NORM_LANES = 32;",
        "constexpr uint NORM_READS = 4;",
        "metal::precise::rsqrt(mean + EPS);",
        "float rope0_lhs = metal::precise::fma(x0, c[pair], 0.0f);",
        "float rope0_rhs = metal::precise::fma(x1, s[pair], 0.0f);",
        "float rope1_lhs = metal::precise::fma(x0, s[pair], 0.0f);",
        "float rope1_rhs = metal::precise::fma(x1, c[pair], 0.0f);",
    ):
        assert fragment in source
    assert "precise float" not in source
    assert (
        "geometry"
        not in inspect.signature(candidate.m3_wqb_qnorm_rope_metal_source).parameters
    )
    assert "parallel_norm" not in source
    assert "shared_x" not in source


def test_bound_call_is_one_fixed_m3_launch_without_hot_validation(monkeypatch):
    from mtplx import deepseek_v4_0731_m3_wqb_qnorm_rope as candidate

    launches = []

    def fake_kernel(*, inputs, grid, threadgroup, output_shapes, output_dtypes, **_):
        launches.append((inputs, grid, threadgroup, output_shapes, output_dtypes))
        return (mx.zeros(output_shapes[0], dtype=output_dtypes[0]),)

    candidate._build_kernel.cache_clear()
    monkeypatch.setattr(candidate.mx.fast, "metal_kernel", lambda **_: fake_kernel)
    bound = candidate.build_0731_m3_wqb_qnorm_rope(_projection())
    actual = bound(
        mx.zeros((1, 3, 1024), dtype=mx.bfloat16),
        mx.zeros((3, 32)),
        mx.zeros((3, 32)),
    )

    assert len(launches) == 1
    inputs, grid, threadgroup, output_shapes, output_dtypes = launches[0]
    assert tuple(inputs[0].shape) == (3, 1024)
    assert tuple(inputs[4].shape) == tuple(inputs[5].shape) == (3, 32)
    assert grid == (64 * 256, 1, 1)
    assert threadgroup == (256, 1, 1)
    assert output_shapes == [(3, 32768)]
    assert output_dtypes == [mx.bfloat16]
    assert actual.shape == (1, 3, 64, 512)
    hot_source = inspect.getsource(type(bound).__call__)
    assert "if " not in hot_source
    assert "try:" not in hot_source


def test_preparation_selfchecks_43_before_reversible_publication(monkeypatch):
    from mtplx import deepseek_v4_0731_m3_wqb_qnorm_rope as candidate

    layers = _layers()
    stocks = tuple(layer.attn._q_projection_qhead_route for layer in layers)
    built = []
    checked = []

    def build(_projection):
        index = len(built)

        def fused(qr, cos, sin):
            return ("candidate", index, qr, cos, sin)

        built.append(fused)
        return fused

    def selfcheck(stock, fused, index):
        checked.append(index)
        assert stock is stocks[index]
        assert fused is built[index]
        assert tuple(layer.attn._q_projection_qhead_route for layer in layers) == stocks
        return True

    monkeypatch.setattr(candidate, "build_0731_m3_wqb_qnorm_rope", build)
    prepared = candidate.prepare_wqb_qhead_m3(
        layers,
        exact_selfcheck=selfcheck,
    )

    assert checked == list(range(43))
    assert prepared.q6_count == 43
    assert prepared.exact_selfchecked == 43
    assert len(prepared.published_routes) == 43
    assert tuple(layer.attn._q_projection_qhead_route for layer in layers) == stocks

    prepared.publish()
    assert tuple(layer.attn._q_projection_qhead_route for layer in layers) == (
        prepared.published_routes
    )
    qr = SimpleNamespace(shape=(2, 3, 1024))
    assert layers[17].attn._q_projection_qhead_route(qr, "c", "s")[:2] == (
        "candidate",
        17,
    )
    prepared.restore()
    assert tuple(layer.attn._q_projection_qhead_route for layer in layers) == stocks


def test_preparation_failure_publishes_nothing(monkeypatch):
    from mtplx import deepseek_v4_0731_m3_wqb_qnorm_rope as candidate

    layers = _layers()
    stocks = tuple(layer.attn._q_projection_qhead_route for layer in layers)
    monkeypatch.setattr(
        candidate,
        "build_0731_m3_wqb_qnorm_rope",
        lambda _projection: lambda qr, cos, sin: (qr, cos, sin),
    )

    with pytest.raises(
        candidate.M3WQBNormRopeContractError,
        match="layer 9.*self-check failed",
    ):
        candidate.prepare_wqb_qhead_m3(
            layers,
            exact_selfcheck=lambda _stock, _fused, index: index != 9,
        )

    assert tuple(layer.attn._q_projection_qhead_route for layer in layers) == stocks


def test_restore_attempts_every_qhead_owner_before_raising():
    from mtplx import deepseek_v4_0731_m3_wqb_qnorm_rope as candidate

    good_stock = object()

    class RejectingOwner:
        def __init__(self, stock):
            object.__setattr__(self, "stock", stock)
            object.__setattr__(self, "_q_projection_qhead_route", object())

        def __setattr__(self, name, value):
            if name == "_q_projection_qhead_route" and value is self.stock:
                raise RuntimeError("first qhead restore failed")
            object.__setattr__(self, name, value)

    first_stock = object()
    first = RejectingOwner(first_stock)
    second = SimpleNamespace(_q_projection_qhead_route=object())
    prepared = candidate.PreparedWQBQHeadM3Routes(
        attentions=(first, second),
        stock_routes=(first_stock, good_stock),
        candidate_routes=(object(), object()),
        published_routes=(object(), object()),
        q6_count=2,
        exact_selfchecked=2,
    )

    with pytest.raises(ExceptionGroup, match="qhead route restoration"):
        prepared.restore()

    assert second._q_projection_qhead_route is good_stock


def test_cpu_oracle_keeps_norm_and_rope_owned_by_each_row():
    from mtplx.deepseek_v4_0731_m3_wqb_qnorm_rope import (
        q_head_norm_rope_cpu_oracle,
    )

    q = np.zeros((1, 3, 1, 4), dtype=np.float32)
    q[..., 0] = 3.0
    q[..., 1] = 4.0
    q[..., 2] = 1.0
    q[..., 3] = 2.0
    cos = np.array([[1.0], [0.0], [-1.0]], dtype=np.float32)
    sin = np.array([[0.0], [1.0], [0.0]], dtype=np.float32)

    got = q_head_norm_rope_cpu_oracle(q, cos, sin, eps=0.0, rope_dim=2)
    scale = 1.0 / np.sqrt(7.5)
    assert np.allclose(got[0, 0, 0], np.array([3, 4, 1, 2]) * scale)
    assert np.allclose(got[0, 1, 0], np.array([3, 4, -2, 1]) * scale)
    assert np.allclose(got[0, 2, 0], np.array([3, 4, -1, -2]) * scale)


@pytest.mark.skipif(
    mx.default_device() != mx.gpu or os.environ.get("MTPLX_GPU_TESTS") != "1",
    reason="requires an explicitly guarded Metal GPU lane",
)
def test_gpu_pre_geometry_candidate_is_exact_to_three_stock_m1_calls():
    pytest.skip("GPU gate is intentionally deferred to the locked lane")
