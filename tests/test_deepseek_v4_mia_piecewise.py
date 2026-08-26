from __future__ import annotations

from types import SimpleNamespace

import pytest

import mlx.core as mx

from mtplx.deepseek_v4_mia_piecewise import (
    MiaDFlashTargetPhaseRoute,
    MiaPhysicalM6PiecewiseTargetRoute,
)
from mtplx.attention_context import attention_phase
from mtplx.models.deepseek_v4 import DeepseekV4Model


class _ToyEmbedding:
    def __init__(self, hidden_size: int):
        self._hidden_size = hidden_size

    def __call__(self, input_ids):
        ids = input_ids.astype(mx.float32)
        columns = [ids + float(index) for index in range(self._hidden_size)]
        return mx.stack(columns, axis=-1)


class _ToyMHC:
    def __init__(self, hidden_size: int, hc_mult: int):
        self._hidden_size = hidden_size
        self._hc_mult = hc_mult

    def pre_broadcast(self, x, hc, norm):
        rows = x.shape[0] * x.shape[1]
        x = x.reshape(rows, self._hidden_size)
        residual = mx.stack(
            [x + float(copy) for copy in range(self._hc_mult)],
            axis=-2,
        )
        post = mx.broadcast_to(
            mx.arange(self._hc_mult, dtype=mx.float32)[None],
            (rows, self._hc_mult),
        )
        comb = mx.broadcast_to(
            mx.eye(self._hc_mult, dtype=mx.float32)[None],
            (rows, self._hc_mult, self._hc_mult),
        )
        value = x + hc.bias + norm.bias
        return residual, post, comb, value

    def _post_pre(self, x, residual, post, comb, hc, norm):
        rows = x.shape[0] * x.shape[1] if x.ndim == 3 else x.shape[0]
        x = x.reshape(rows, self._hidden_size)
        residual = residual.reshape(rows, self._hc_mult, self._hidden_size)
        residual = (residual + x[:, None, :]) * 0.25
        post = post + hc.bias
        comb = comb + norm.bias
        value = mx.sum(residual, axis=-2) + hc.bias + norm.bias
        return residual, post, comb, value

    post_pre_attn = _post_pre
    post_pre_ffn = _post_pre

    def post(self, x, residual, post, comb):
        rows = x.shape[0] * x.shape[1] if x.ndim == 3 else x.shape[0]
        x = x.reshape(rows, self._hidden_size)
        residual = residual.reshape(rows, self._hc_mult, self._hidden_size)
        return (
            residual
            + x[:, None, :]
            + post[:, :, None]
            + mx.sum(comb, axis=-1)[:, :, None]
        )


class _ToyAttention:
    def __init__(self, layer_id: int, compile_spy=None):
        self._layer_id = layer_id
        self._compile_spy = compile_spy
        self.observed_offsets = []
        self.observed_schedules = []
        self.generic_calls = 0

    def __call__(self, value, *, mask, cache):
        self.generic_calls += 1
        return self._forward(value, mask=mask, cache=cache)

    def _mia_m6_forward_impl(self, value, *, cache, schedule):
        self.observed_schedules.append(schedule)
        return self._forward(value, mask=None, cache=cache)

    def _forward(self, value, *, mask, cache):
        assert mask is None
        if self._compile_spy is not None:
            assert self._compile_spy.depth == 0
        self.observed_offsets.append(cache.offset)
        return value + float(self._layer_id + cache.offset) / 64.0


class _ToyFFN:
    def __init__(self, layer_id: int):
        self._layer_id = layer_id
        self.generic_calls = 0
        self.shared_input_shapes = []
        self._input_rows_impl = lambda input_ids: input_ids.reshape(-1)
        self.gate = lambda _value, input_ids: (input_ids, None)
        self.shared_experts = self._shared_experts
        self._mia_exl3_m6_fused = (
            lambda _value, input_ids, _weights, shared: (
                shared + input_ids.astype(mx.float32)[:, None] / 64.0
            )
        )

    def _shared_experts(self, value):
        self.shared_input_shapes.append(tuple(int(item) for item in value.shape))
        return value + float(self._layer_id) / 64.0

    def __call__(self, value, *, input_ids):
        self.generic_calls += 1
        token_term = input_ids.astype(mx.float32)[..., None]
        return value + token_term / 64.0 + float(self._layer_id) / 64.0


def _toy_model(*, layer_count=43, compile_spy=None):
    hidden_size = 4
    hc_mult = 2
    layers = []
    for layer_id in range(layer_count):
        layers.append(
            SimpleNamespace(
                attn_hc=SimpleNamespace(bias=float(layer_id + 1) / 64.0),
                attn_norm=SimpleNamespace(bias=0.25),
                ffn_hc=SimpleNamespace(bias=float(layer_id + 2) / 64.0),
                ffn_norm=SimpleNamespace(bias=0.5),
                attn=_ToyAttention(layer_id, compile_spy),
                ffn=_ToyFFN(layer_id),
            )
        )
    return SimpleNamespace(
        args=SimpleNamespace(hidden_size=hidden_size, hc_mult=hc_mult),
        layers=tuple(layers),
        embed_tokens=_ToyEmbedding(hidden_size),
        _mia_mhc=_ToyMHC(hidden_size, hc_mult),
    )


class _CompileSpy:
    def __init__(self):
        self.depth = 0
        self.functions = []

    def __call__(self, fn):
        self.functions.append(fn)

        def compiled(*args):
            self.depth += 1
            try:
                return fn(*args)
            finally:
                self.depth -= 1

        return compiled


class _ScheduleSpy:
    def __init__(self):
        self.accesses = 0
        self.cycle = SimpleNamespace(
            by_layer=tuple(object() for _ in range(43)),
        )

    @property
    def current_cycle(self):
        self.accesses += 1
        return self.cycle


def _caches(offset: int):
    schedule = _ScheduleSpy()
    return tuple(
        SimpleNamespace(offset=offset, _mia_m6_schedule=schedule)
        for _ in range(43)
    )


def _assert_array_equal(actual, expected):
    assert bool(mx.array_equal(actual, expected).item())


def test_piecewise_route_is_construction_bound_to_physical_m6_and_43_layers():
    with pytest.raises(ValueError, match="physical width 6"):
        MiaPhysicalM6PiecewiseTargetRoute(
            _toy_model(),
            physical_width=5,
        )
    with pytest.raises(ValueError, match="43 target layers"):
        MiaPhysicalM6PiecewiseTargetRoute(
            _toy_model(layer_count=42),
            physical_width=6,
        )


def test_dflash_phase_route_reserves_piecewise_m6_for_decode_verify():
    calls = []
    route = MiaDFlashTargetPhaseRoute(
        prefill=lambda inputs, cache: calls.append(("prefill", inputs, cache)),
        decode_verify=lambda inputs, cache: calls.append(
            ("decode_verify", inputs, cache)
        ),
    )
    inputs = object()
    cache = object()

    with attention_phase("prefill"):
        route(inputs, cache)
    with attention_phase("decode_verify"):
        route(inputs, cache)

    assert calls == [
        ("prefill", inputs, cache),
        ("decode_verify", inputs, cache),
    ]


def test_attention_is_outside_tapes_and_reads_current_cache_metadata():
    spy = _CompileSpy()
    model = _toy_model(compile_spy=spy)
    route = MiaPhysicalM6PiecewiseTargetRoute(
        model,
        physical_width=6,
        compile_fn=spy,
    )
    input_ids = mx.arange(6, dtype=mx.uint32)[None]

    first_hidden, first_taps = route(input_ids, _caches(3))
    second_hidden, second_taps = route(input_ids, _caches(19))
    mx.eval(first_hidden, *first_taps, second_hidden, *second_taps)

    assert len(spy.functions) == 44
    assert all(layer.attn.observed_offsets == [3, 19] for layer in model.layers)
    assert not bool(mx.array_equal(first_hidden, second_hidden).item())


def test_piecewise_fetches_one_cycle_and_uses_prebound_m6_attentions():
    model = _toy_model()
    route = MiaPhysicalM6PiecewiseTargetRoute(
        model,
        physical_width=6,
        compile_fn=lambda fn: fn,
    )
    caches = _caches(127)
    schedule = caches[0]._mia_m6_schedule

    hidden, taps = route(mx.arange(6, dtype=mx.uint32)[None], caches)
    mx.eval(hidden, *taps)

    assert schedule.accesses == 1
    assert all(layer.attn.generic_calls == 0 for layer in model.layers)
    assert all(
        layer.attn.observed_schedules == [schedule.cycle.by_layer[layer_id]]
        for layer_id, layer in enumerate(model.layers)
    )


def test_piecewise_regions_bind_the_fixed_m6_ffn_without_generic_routing():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        spy = _CompileSpy()
        model = _toy_model(compile_spy=spy)
        route = MiaPhysicalM6PiecewiseTargetRoute(
            model,
            physical_width=6,
            compile_fn=spy,
        )

        hidden, taps = route(mx.arange(6, dtype=mx.uint32)[None], _caches(3))
        mx.eval(hidden, *taps)

        assert all(layer.ffn.generic_calls == 0 for layer in model.layers)
        assert all(
            layer.ffn.shared_input_shapes == [(6, 4)]
            for layer in model.layers
        )
    finally:
        mx.set_default_device(previous)


def test_piecewise_outputs_and_tail_taps_match_exact_eager_route_bitwise():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        eager_model = _toy_model()
        compiled_model = _toy_model()
        input_ids = mx.arange(6, dtype=mx.uint32)[None]

        expected_hidden, expected_taps = DeepseekV4Model._run_mia_hc_target_tail_taps(
            eager_model,
            input_ids,
            _caches(7),
        )
        route = MiaPhysicalM6PiecewiseTargetRoute(
            compiled_model,
            physical_width=6,
        )
        actual_hidden, actual_taps = route(input_ids, _caches(7))
        mx.eval(
            expected_hidden,
            *expected_taps,
            actual_hidden,
            *actual_taps,
        )

        _assert_array_equal(actual_hidden, expected_hidden)
        assert len(actual_taps) == len(expected_taps) == 3
        for actual, expected in zip(actual_taps, expected_taps):
            _assert_array_equal(actual, expected)
    finally:
        mx.set_default_device(previous)
