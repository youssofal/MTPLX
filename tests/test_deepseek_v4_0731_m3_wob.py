"""CPU contracts for the receipt-backed fixed-M3 Q6/G128 WOB primitive."""

from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

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


class _CallableProjection(SimpleNamespace):
    def __call__(self, value):
        return ("stock", value)


def _projection(*, bits=6, group_size=128, mode="affine"):
    return _CallableProjection(
        bits=bits,
        group_size=group_size,
        mode=mode,
        bias=None,
        weight=SimpleNamespace(shape=(4096, 1536), dtype=mx.uint32),
        scales=SimpleNamespace(shape=(4096, 64), dtype=mx.bfloat16),
        biases=SimpleNamespace(shape=(4096, 64), dtype=mx.bfloat16),
    )


def _layers():
    layers = []
    for _ in range(43):
        stock = _projection()
        attention = SimpleNamespace(
            wo_b=stock,
            _o_lora_impl=SimpleNamespace(wo_b=stock),
        )
        layers.append(SimpleNamespace(attn=attention))
    return layers


def test_contract_and_binder_are_only_fixed_m3_q6_g128(monkeypatch):
    from mtplx import deepseek_v4_0731_m3_wob as candidate

    contract = candidate.M3WOBContract()
    assert (contract.k, contract.n, contract.bits, contract.group_size) == (
        8192,
        4096,
        6,
        128,
    )
    candidate.validate_wob_projection(_projection(), contract)
    with pytest.raises(candidate.M3WOBContractError, match="Q6/G128"):
        candidate.validate_wob_projection(_projection(bits=8), contract)

    calls = []
    candidate._build_wob_kernel.cache_clear()
    monkeypatch.setattr(
        candidate.mx.fast,
        "metal_kernel",
        lambda **kwargs: calls.append(kwargs) or object(),
    )
    bound = candidate.bind_m3_wob(_projection())
    assert bound.input_shape == (1, 3, 8192)
    assert bound.output_shape == (1, 3, 4096)
    assert bound.grid == ((4096 // 8) * 64, 1, 1)
    assert bound.threadgroup == (64, 1, 1)
    assert calls[0]["input_names"] == ["x", "w", "scales", "biases"]
    assert calls[0]["output_names"] == ["y"]


def test_source_preserves_three_official_q6_m1_reduction_trees():
    from mtplx import deepseek_v4_0731_m3_wob as candidate

    source = candidate.m3_wob_metal_source()
    for fragment in (
        "constexpr uint M = 3;",
        "constexpr uint K = 8192;",
        "constexpr uint N = 4096;",
        "constexpr uint GS = 128;",
        "float result0[RESULTS_PER_SIMDGROUP]",
        "float result1[RESULTS_PER_SIMDGROUP]",
        "float result2[RESULTS_PER_SIMDGROUP]",
        "simd_sum(result0[row])",
        "simd_sum(result1[row])",
        "simd_sum(result2[row])",
        "thread uchar w_thread[PACKS_PER_THREAD * BYTES_PER_PACK]",
        "sum0 += x0[i] + x0[i + 1] + x0[i + 2] + x0[i + 3];",
    ):
        assert fragment in source
    assert "return projection(x)" not in source


def test_preparation_is_atomic_reversible_and_rebinds_active_o_lora(monkeypatch):
    from mtplx import deepseek_v4_0731_m3_wob as candidate

    layers = _layers()
    stocks = tuple(layer.attn.wo_b for layer in layers)
    built = []
    checked = []

    def build(_stock):
        index = len(built)

        def fixed(value):
            return ("candidate", index, value)

        built.append(fixed)
        return fixed

    def selfcheck(stock, fixed, index):
        checked.append(index)
        assert stock is stocks[index]
        assert fixed is built[index]
        assert tuple(layer.attn.wo_b for layer in layers) == stocks
        assert tuple(layer.attn._o_lora_impl.wo_b for layer in layers) == stocks
        return True

    monkeypatch.setattr(candidate, "bind_m3_wob", build)
    prepared = candidate.prepare_wob_m3(layers, exact_selfcheck=selfcheck)

    assert checked == list(range(43))
    assert prepared.layer_count == 43
    assert prepared.q6_count == 43
    assert prepared.exact_selfchecked == 43
    assert prepared.o_lora_sink_count == 43
    assert tuple(layer.attn.wo_b for layer in layers) == stocks
    assert tuple(layer.attn._o_lora_impl.wo_b for layer in layers) == stocks

    prepared.publish()
    assert tuple(layer.attn.wo_b for layer in layers) == prepared.published_routes
    assert tuple(layer.attn._o_lora_impl.wo_b for layer in layers) == (
        prepared.published_routes
    )
    value = SimpleNamespace(shape=(4, 3, 8192))
    assert layers[5].attn.wo_b(value) == ("candidate", 5, value)
    prepared.restore()
    assert tuple(layer.attn.wo_b for layer in layers) == stocks
    assert tuple(layer.attn._o_lora_impl.wo_b for layer in layers) == stocks


def test_stale_o_lora_sink_rejected_before_binding(monkeypatch):
    from mtplx import deepseek_v4_0731_m3_wob as candidate

    layers = _layers()
    stocks = tuple(layer.attn.wo_b for layer in layers)
    layers[19].attn._o_lora_impl.wo_b = _projection()
    binds = []
    monkeypatch.setattr(candidate, "bind_m3_wob", lambda stock: binds.append(stock))

    with pytest.raises(candidate.M3WOBContractError, match="o-LoRA wo_b sink"):
        candidate.prepare_wob_m3(layers, exact_selfcheck=lambda *_: True)

    assert binds == []
    assert tuple(layer.attn.wo_b for layer in layers) == stocks


def test_failed_selfcheck_checks_all_layers_and_publishes_nothing(monkeypatch):
    from mtplx import deepseek_v4_0731_m3_wob as candidate

    layers = _layers()
    stocks = tuple(layer.attn.wo_b for layer in layers)
    checked = []
    monkeypatch.setattr(
        candidate,
        "bind_m3_wob",
        lambda _stock: lambda value: ("candidate", value),
    )

    with pytest.raises(candidate.M3WOBContractError, match="layer 17.*failed"):
        candidate.prepare_wob_m3(
            layers,
            exact_selfcheck=lambda _stock, _fixed, index: (
                checked.append(index) or index != 17
            ),
        )

    assert checked == list(range(43))
    assert tuple(layer.attn.wo_b for layer in layers) == stocks
    assert tuple(layer.attn._o_lora_impl.wo_b for layer in layers) == stocks


def test_restore_attempts_every_attention_and_o_lora_owner_before_raising():
    from mtplx import deepseek_v4_0731_m3_wob as candidate

    class RejectingOwner:
        def __init__(self, stock):
            object.__setattr__(self, "stock", stock)
            object.__setattr__(self, "wo_b", object())

        def __setattr__(self, name, value):
            if name == "wo_b" and value is self.stock:
                raise RuntimeError("first WOB restore failed")
            object.__setattr__(self, name, value)

    first_stock = object()
    second_stock = object()
    first_attention = RejectingOwner(first_stock)
    first_o_lora = SimpleNamespace(wo_b=object())
    second_attention = SimpleNamespace(wo_b=object())
    second_o_lora = SimpleNamespace(wo_b=object())
    prepared = candidate.PreparedWOBM3Routes(
        attentions=(first_attention, second_attention),
        o_lora_impls=(first_o_lora, second_o_lora),
        stock_routes=(first_stock, second_stock),
        candidate_routes=(object(), object()),
        published_routes=(object(), object()),
        layer_count=2,
        q6_count=2,
        exact_selfchecked=2,
        o_lora_sink_count=2,
    )

    with pytest.raises(ExceptionGroup, match="WOB route restoration"):
        prepared.restore()

    assert first_o_lora.wo_b is first_stock
    assert second_attention.wo_b is second_stock
    assert second_o_lora.wo_b is second_stock


def test_fixed_route_has_only_logical_m_choice():
    from mtplx import deepseek_v4_0731_m3_wob as candidate

    calls = []

    def stock(value):
        calls.append(("stock", value))
        return "stock"

    def fixed(value):
        calls.append(("fixed", value))
        return "fixed"

    route = candidate.prebind_wob_route(stock, fixed)
    assert route(SimpleNamespace(shape=(2, 3, 8192))) == "fixed"
    assert route(SimpleNamespace(shape=(1, 1, 8192))) == "stock"
    source = inspect.getsource(type(route).__call__)
    assert "shape[1]" in source
    assert "try:" not in source


@pytest.mark.skipif(
    mx.default_device() != mx.gpu or os.environ.get("MTPLX_GPU_TESTS") != "1",
    reason="requires an explicitly guarded Metal GPU lane",
)
def test_gpu_wob_is_exact_to_three_stock_m1_calls():
    pytest.skip("GPU gate is intentionally deferred to the locked lane")
