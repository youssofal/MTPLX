from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from mtplx.generation import _sample_from_logits
from mtplx.models.inkling.sampling import draw_native, is_native_sampler
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
