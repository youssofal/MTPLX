"""CPU checks of the convolution receipt each request carries.

With ``MTPLX_TEST_CPU_ONLY=1`` the tests run on the CPU.
"""

import os
from dataclasses import dataclass, field

import mlx.core as mx
import pytest

if os.environ.get("MTPLX_TEST_CPU_ONLY") == "1":
    mx.set_default_device(mx.cpu)

from mtplx.models.inkling import model as inkling_model  # noqa: E402
from mtplx.models.inkling import short_conv  # noqa: E402


@pytest.fixture(autouse=True)
def _route_switch_restored(monkeypatch):
    # The tests below set the convolution route they need (os.environ, as an operator would); the
    # switch must not outlive the test: left at "stock" it sent every later test in the session to
    # MLX's convolution, and the stored stage C proof (its reference convolves) refused on every pack.
    monkeypatch.setenv(short_conv.ROUTE_ENV, "candidate")


@dataclass
class _Stats:
    inkling_convolution: dict | None = None


@dataclass
class _Output:
    stats: _Stats = field(default_factory=_Stats)


class _Runtime:
    backend_id = "inkling2_loop"


def _conv():
    return inkling_model.Convolution({"c.weight": mx.ones((16, 1, 4), mx.bfloat16)}, "c")


def _serve(calls, raise_after=False):
    def fn(rt):
        conv = _conv()
        for _ in range(calls):
            conv(mx.ones((1, 2, 16), mx.bfloat16), [None], 0)
        if raise_after:
            raise RuntimeError("request failed")
        return _Output()

    return fn


def test_each_request_gets_its_own_routes():
    os.environ[short_conv.ROUTE_ENV] = "candidate"
    first = short_conv.observe_convolution_request(_serve(3), _Runtime())
    second = short_conv.observe_convolution_request(_serve(5), _Runtime())
    expected = "stock:not_gpu" if mx.default_device() != mx.gpu else None
    if expected:
        assert first.stats.inkling_convolution["routes"] == {expected: 3}
        assert second.stats.inkling_convolution["routes"] == {expected: 5}
    assert first.stats.inkling_convolution["mode"] == "candidate"
    assert short_conv._request.get() is None


def test_nested_request_counts_into_the_outer_receipt():
    os.environ[short_conv.ROUTE_ENV] = "stock"

    def outer(rt):
        inner = short_conv.observe_convolution_request(_serve(2), rt)
        assert inner.stats.inkling_convolution is None  # the outer request owns the receipt
        _conv()(mx.ones((1, 1, 16), mx.bfloat16), [None], 0)
        return _Output()

    out = short_conv.observe_convolution_request(outer, _Runtime())
    assert out.stats.inkling_convolution["routes"] == {"stock:chosen": 3}


def test_a_failed_request_leaves_no_receipt_behind():
    with pytest.raises(RuntimeError):
        short_conv.observe_convolution_request(_serve(1, raise_after=True), _Runtime())
    assert short_conv._request.get() is None
    later = short_conv.observe_convolution_request(_serve(1), _Runtime())
    assert sum(later.stats.inkling_convolution["routes"].values()) == 1


def test_first_use_cost_lands_in_the_request_that_paid_and_in_the_process_totals():
    def fn(rt):
        short_conv.note_first_use(("bfloat16", None, 2304, 1, 4), "float", 0.25)
        return _Output()

    out = short_conv.observe_convolution_request(fn, _Runtime())
    receipt = out.stats.inkling_convolution
    assert receipt["first_use"] == [{"key": "bfloat16/None/2304/1/4", "verdict": "float", "seconds": 0.25}]
    assert receipt["first_use_seconds"] == 0.25
    assert short_conv.conv_route_counts()["proof_seconds"]["bfloat16/None/2304/1/4"] == 0.25
    short_conv._proof_seconds.clear()


def test_generation_stats_carry_the_receipt_only_when_present():
    from mtplx.generation import GenerationStats

    stats = GenerationStats.__new__(GenerationStats)
    names = {f.name for f in GenerationStats.__dataclass_fields__.values()}
    assert "inkling_convolution" in names
    field_ = GenerationStats.__dataclass_fields__["inkling_convolution"]
    assert field_.kw_only and field_.default is None
    del stats


def test_the_wrappers_are_installed_on_both_generation_entries():
    from mtplx import generation

    for name in ("generate_ar", "generate_mtpk"):
        fn = getattr(generation, name)
        chain = []
        while fn is not None:
            chain.append(fn.__code__.co_names)
            fn = getattr(fn, "__wrapped__", None)
        assert any("observe_convolution_request" in names for names in chain), name
