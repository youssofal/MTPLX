from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from mtplx.generation import _sample_from_logits
from mtplx.models.inkling.sampling import NativeDrawsAhead, _accumulated, draw_native, is_native_sampler
from mtplx.sampling import NonFiniteLogitsError, SamplerConfig


NATIVE = SamplerConfig(temperature=1, top_p=1, top_k=0)


def test_native_draw_keeps_each_seeded_token_and_generator_state():
    source = np.random.default_rng(113)
    left, right = (np.random.default_rng(29) for _ in range(2))
    with mx.stream(mx.cpu):
        for scale in (0.0, 1.0, 7.0, 50.0):
            values = source.normal(0, scale, 200058).astype(np.float32)
            values[::17] = -np.inf
            logits = mx.array(values)
            for _ in range(4):
                token, _ = _sample_from_logits(logits, NATIVE, left)
                assert draw_native(logits, right) == token
    assert left.bit_generator.state == right.bit_generator.state


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_invalid_native_row_raises_without_drawing(value):
    rng = np.random.default_rng(29)
    before = rng.bit_generator.state
    with mx.stream(mx.cpu), np.errstate(invalid="ignore"), pytest.raises(NonFiniteLogitsError):
        draw_native(mx.full((10,), value), rng)
    assert rng.bit_generator.state == before


def test_other_sampler_choices_keep_the_shared_path():
    assert is_native_sampler(NATIVE)
    for setting in ({"temperature": 0}, {"temperature": 0.7}, {"top_p": 0.99},
                    {"top_k": 20}, {"frequency_penalty": 0.2}, {"presence_penalty": -1}):
        assert not is_native_sampler(replace(NATIVE, **setting))


def _reference_accumulated(row):
    # The draw's arithmetic as it was written before rows were computed ahead (in place).
    mass = row.astype(np.float64)
    mass -= mass.max()
    np.exp(mass, out=mass)
    mass /= mass.sum()
    mass /= mass.sum()
    mass /= mass.sum()
    np.cumsum(mass, out=mass)
    return mass


def _panel(source, rows, scale, vocab=200058):
    values = source.normal(0, scale, (rows, vocab)).astype(np.float32)
    values[:, ::17] = -np.inf
    return values


def test_accumulated_masses_keep_the_in_place_bits():
    source = np.random.default_rng(7)
    for scale in (0.0, 1.0, 7.0, 50.0):
        for row in _panel(source, 3, scale):
            got, want = _accumulated(row), _reference_accumulated(row)
            assert np.array_equal(got.view(np.uint64), want.view(np.uint64))


@pytest.mark.parametrize("walked", [1, 2, 9, 25, 32])
def test_draws_ahead_equal_the_row_by_row_draw(walked):
    source = np.random.default_rng(walked)
    with mx.stream(mx.cpu):
        for scale in (1.0, 7.0, 50.0):
            panel = mx.array(_panel(source, 32, scale))
            for seed in (3, 29):
                one, ahead_rng = (np.random.default_rng(seed) for _ in range(2))
                ahead = NativeDrawsAhead(panel)
                try:
                    got = [ahead.draw(pos, ahead_rng) for pos in range(walked)]
                finally:
                    ahead.close()
                want = [draw_native(panel[pos], one) for pos in range(walked)]
                assert got == want
                assert ahead_rng.bit_generator.state == one.bit_generator.state


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_draws_ahead_raise_at_a_non_finite_row_before_its_draw(value):
    source = np.random.default_rng(11)
    values = _panel(source, 6, 3.0)
    values[3, :] = value
    rng, one = np.random.default_rng(5), np.random.default_rng(5)
    with mx.stream(mx.cpu), np.errstate(invalid="ignore"):
        panel = mx.array(values)
        ahead = NativeDrawsAhead(panel)
        try:
            assert [ahead.draw(pos, rng) for pos in range(3)] == [draw_native(panel[pos], one) for pos in range(3)]
            with pytest.raises(NonFiniteLogitsError):
                ahead.draw(3, rng)
        finally:
            ahead.close()
    assert rng.bit_generator.state == one.bit_generator.state
