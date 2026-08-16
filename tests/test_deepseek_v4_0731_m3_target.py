"""CPU contracts for the single receipt-backed physical-M3 target route."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

from mtplx import deepseek_v4_0731_m3_target as target  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_default_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


class _Body:
    hc_mult = 4

    def __init__(self):
        self.layers = tuple(object() for _ in range(43))

    @staticmethod
    def embed_tokens(ids):
        return mx.broadcast_to(ids[..., None].astype(mx.float32), (*ids.shape, 2))


class _Model:
    def __init__(self):
        self.args = SimpleNamespace(
            hidden_size=4096,
            num_hidden_layers=43,
            num_attention_heads=64,
            num_key_value_heads=1,
            head_dim=512,
        )
        self.model = _Body()
        self._dspark = SimpleNamespace(
            target_layer_ids=(40, 41, 42),
            stages=(object(), object(), object()),
        )
        self.base_calls = []

        def base(owner, input_ids, cache=None):
            self.base_calls.append((owner, input_ids, cache))
            return "base-hidden", "base-taps"

        self._target_hidden_route = base

    @staticmethod
    def logits_from_hc_hidden(hidden):
        return mx.mean(hidden, axis=2)


def test_fixed_m3_route_runs_43_prebound_layers_once_and_captures_taps():
    model = _Model()
    calls = []

    def layer_route(hidden, input_ids, cache, *, layer_id):
        calls.append((layer_id, tuple(input_ids.shape), cache))
        return hidden + 1

    routes = tuple(
        lambda hidden, input_ids, cache, layer_id=layer_id: layer_route(
            hidden,
            input_ids,
            cache,
            layer_id=layer_id,
        )
        for layer_id in range(43)
    )
    route = target.build_0731_m3_target_route(
        model,
        full_layer_routes=routes,
        base_route=model._target_hidden_route,
    )
    caches = tuple(object() for _ in range(43))

    logits, hidden, taps = route.forward(mx.array([[7, 8, 9]]), caches)
    mx.eval(logits, hidden, taps)

    assert hidden.shape == (1, 3, 4, 2)
    assert logits.shape == (1, 3, 2)
    assert taps.shape == (1, 3, 6)
    assert [layer_id for layer_id, _, _ in calls] == list(range(43))
    assert [cache for _, _, cache in calls] == list(caches)
    assert all(shape == (1, 3) for _, shape, _ in calls)
    assert mx.array_equal(taps[..., :2], hidden[..., 0, :] - 2)
    assert mx.array_equal(taps[..., 2:4], hidden[..., 0, :] - 1)
    assert mx.array_equal(taps[..., 4:], hidden[..., 0, :])


def test_fixed_m3_route_delegates_only_non_m3_shapes_to_prebound_base():
    model = _Model()
    routes = tuple(lambda hidden, _ids, _cache: hidden for _ in range(43))
    route = target.build_0731_m3_target_route(
        model,
        full_layer_routes=routes,
        base_route=model._target_hidden_route,
    )
    ids = mx.array([[7]])

    assert route(model, ids, "cache") == ("base-hidden", "base-taps")
    assert model.base_calls == [(model, ids, "cache")]

    # Batch size is construction-owned; the hot route reads logical M only.
    m3_ids = mx.array([[1, 2, 3], [4, 5, 6]])
    hidden, taps = route(model, m3_ids, (None,) * 43)
    mx.eval(hidden, taps)
    assert hidden.shape == (2, 3, 4, 2)
    assert model.base_calls == [(model, ids, "cache")]


def test_bound_m3_body_directly_executes_without_hot_invariant_checks():
    source = inspect.getsource(target._M3TargetBody.__call__)
    assert "physical-M3 target route requires" not in source
    assert "len(entries)" not in source
    assert "strict=True" not in source
    assert "missed a DSpark tap" not in source

    model = _Model()
    calls = []
    routes = tuple(
        lambda hidden, _ids, cache, layer_id=layer_id: (
            calls.append((layer_id, cache)) or hidden
        )
        for layer_id in range(43)
    )
    route = target.build_0731_m3_target_route(
        model,
        full_layer_routes=routes,
        base_route=model._target_hidden_route,
    )

    class CacheEntries:
        def __iter__(self):
            return iter(range(43))

        def __len__(self):
            raise AssertionError("bound route must not revalidate cache length")

    logits, hidden, taps = route.forward(mx.array([[7, 8]]), CacheEntries())
    mx.eval(logits, hidden, taps)

    assert hidden.shape == (1, 2, 4, 2)
    assert taps.shape == (1, 2, 6)
    assert calls == list(enumerate(range(43)))


def test_fixed_m3_builder_rejects_missing_routes_and_non_0731_owner():
    model = _Model()
    with pytest.raises(target.M3TargetContractError, match="43 prebound"):
        target.build_0731_m3_target_route(
            model,
            full_layer_routes=(),
            base_route=model._target_hidden_route,
        )

    model._dspark.target_layer_ids = (39, 40, 41)
    with pytest.raises(target.M3TargetContractError, match="target taps"):
        target.build_0731_m3_target_route(
            model,
            full_layer_routes=tuple(lambda *_args: None for _ in range(43)),
            base_route=model._target_hidden_route,
        )


def test_compiled_tail_layer_keeps_one_full_width_three_boundary():
    calls = []

    class HC:
        @staticmethod
        def pre(hidden):
            calls.append(("pre", tuple(hidden.shape)))
            value = hidden[:, :, 0, :]
            return value, value + 20, value[:, :, None, :] + 30

    class Layer:
        attn_hc = HC()
        attn_norm = staticmethod(lambda value: value + 10)

        @staticmethod
        def attn(value, *, mask, cache):
            calls.append(("attention", tuple(value.shape), cache))
            return value + 100

    def compiled_tail(attn_out, residual, post, comb, input_ids):
        calls.append(
            (
                "tail",
                tuple(attn_out.shape),
                tuple(residual.shape),
                tuple(post.shape),
                tuple(comb.shape),
                tuple(input_ids.shape),
            )
        )
        return attn_out[:, :, None, :] + mx.zeros_like(residual)

    route = target.build_m3_compiled_tail_layer(Layer(), compiled_tail)
    hidden = mx.zeros((1, 3, 4, 2), dtype=mx.float32)
    ids = mx.array([[7, 8, 9]])
    cache = object()

    got = route(hidden, ids, cache)
    mx.eval(got)

    assert got.shape == (1, 3, 4, 2)
    assert calls == [
        ("pre", (1, 3, 4, 2)),
        ("attention", (1, 3, 2), cache),
        ("tail", (1, 3, 2), (1, 3, 4, 2), (1, 3, 2), (1, 3, 1, 2), (1, 3)),
    ]


def test_module_exposes_no_unmeasured_control_or_hybrid_modes():
    assert not hasattr(target, "RowExactControlLayer")
    assert not hasattr(target, "M3HybridControlLayer")
    assert not hasattr(target, "build_row_exact_control_layer")
    assert not hasattr(target, "build_m3_hybrid_control_layer")
