"""Review 2, finding 13: repetition penalties far from 1 keep the host's decision.

A penalty p divides a seen positive logit by p. At p = 2**-20 two logits 2**-20 apart
(12.000001907 and 12.000002861) become 12,582,914 and 12,582,915; the device's float32 law
row (scores times float32(1 / T)) rounds both to 8,388,610 at temperature 1.5, a coin flip,
where the law itself gives the larger one 0.66. The device round serves penalties in
``REPETITION_RANGE`` only; the corners of that range at the lowest served temperature are
checked against float64 on the GPU.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from mtplx.generation import ClientControlledDraw
from mtplx.models.inkling.fast import round as rnd
from mtplx.sampling import SamplerConfig

L1, L2 = 12.000001907348633, 12.00000286102295
TINY = 2.0**-20


def _device():
    try:
        return mx.metal.is_available()
    except Exception:  # noqa: BLE001
        return False


needs_gpu = pytest.mark.skipif(not _device(), reason="needs a Metal GPU")


def test_the_counterexample_is_a_coin_flip_in_float32_and_0_66_in_the_law():
    t = 1.5
    law64 = np.array([L1 / TINY, L2 / TINY]) / t
    p64 = 1.0 / (1.0 + math.exp(law64[0] - law64[1]))
    assert abs(p64 - 0.6608) < 1e-3
    scale32 = np.float32(1.0 / t)
    law32 = (np.array([L1, L2], np.float32) / np.float32(TINY)) * scale32  # the device's float32 law
    assert law32.dtype == np.float32 and law32[0] == law32[1] == np.float32(8388610.0)


def test_the_served_range():
    assert all(rnd.repetition_served(p) for p in (1.0, 0.0, None, 0.5, 0.9, 1.1, 1.3, 2.0))
    assert not any(rnd.repetition_served(p) for p in (TINY, 0.49, 2.01, 10.0, float("nan")))


@needs_gpu  # supported() declines every request on a machine without the GPU
def test_penalties_outside_the_range_keep_the_host_decision():
    served = SamplerConfig(temperature=1.5, top_p=1.0, top_k=0, repetition_penalty=1.1)
    assert rnd.supported(served) == (True, None)
    for penalty in (TINY, 4.0):
        far = SamplerConfig(temperature=1.5, top_p=1.0, top_k=0, repetition_penalty=penalty)
        assert rnd.supported(far) == (False, "repetition_penalty_range")
    greedy = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=TINY)
    assert rnd.supported(greedy) == (False, "repetition_penalty_range")


@needs_gpu
@pytest.mark.parametrize("penalty", [0.5, 2.0])
def test_the_range_corners_at_the_lowest_temperature_match_float64(penalty):
    """The device's law row from the controlled scores (``ClientControlledDraw.rows`` then
    ``law_rows``) at temperature 0.05 against the same law in float64, logits up to +-40 with
    half the vocabulary seen: the largest probability error stays below 1e-4."""
    rng = np.random.default_rng(13)
    vocab = 32_768
    t = rnd.TEMPERATURE_RANGE[0]
    worst = 0.0
    for trial in range(8):
        logits = (rng.normal(size=vocab) * 8.0).clip(-40, 40).astype(np.float32)
        top = rng.choice(vocab, 6, replace=False)
        logits[top] = 40.0 - rng.random(6).astype(np.float32) * 0.05  # a close race at the top
        if trial % 2:
            logits = -np.abs(logits)  # every logit negative: the penalty multiplies them
        seen = rng.choice(vocab, vocab // 2, replace=False)
        controls = ClientControlledDraw(SamplerConfig(temperature=t, top_p=1.0, top_k=0,
                                                      repetition_penalty=penalty), seen.tolist())
        law, _ = rnd.law_rows(controls.rows(mx.array(logits[None, :])), temperature=t)
        y32 = np.asarray(law, np.float64)[0]
        l64 = logits.astype(np.float64)
        mask = np.zeros(vocab, bool)
        mask[seen] = True
        l64 = np.where(mask, np.where(l64 > 0, l64 / penalty, l64 * penalty), l64) / t
        p64 = np.exp(l64 - l64.max())
        p64 /= p64.sum()
        p32 = np.exp(y32 - y32.max())
        p32 /= p32.sum()
        worst = max(worst, float(np.abs(p32 - p64).max()))
    assert worst < 1e-4, worst
