"""Greedy draft contracts used by expected-value depth selection."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from mtplx.adaptive import ExpectedValueDepthPolicy
from mtplx.generation import _draft_confidence_metrics, _sample_draft_from_logits
from mtplx.sampling import SamplerConfig, SparseDistribution


@pytest.fixture(autouse=True)
def _cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
@pytest.mark.parametrize(
    "values",
    [
        [1.0, 4.0, 3.25, -2.0, 0.5, 2.0, -1.0, 0.0, 1.5],
        [2.0, 2.0, 1.0, -1.0, 0.0, 0.5, 1.5, -3.0, 0.25],
    ],
)
def test_confidence_metrics_match_fp32_top8(values, dtype):
    logits = mx.array(values, dtype=dtype)

    metrics = _draft_confidence_metrics(logits, topk=8)

    rounded = np.asarray(logits.astype(mx.float32), dtype=np.float32)
    top_values = np.sort(rounded)[-8:]
    shifted = top_values[::-1].astype(np.float64) - float(top_values[-1])
    exp_values = np.exp(shifted)
    probabilities = exp_values / float(np.sum(exp_values))
    expected_entropy = -float(
        np.sum(probabilities * np.log(np.maximum(probabilities, 1e-30)))
    )

    assert metrics["top2_margin"] == pytest.approx(
        float(top_values[-1] - top_values[-2])
    )
    assert metrics["top1_prob_topk"] == pytest.approx(float(probabilities[0]))
    assert metrics["entropy_topk"] == pytest.approx(expected_entropy)


@pytest.mark.parametrize("need_distribution", [False, True])
def test_greedy_draft_reader_preserves_first_maximum(need_distribution):
    logits = mx.array([1.0, 4.0, 3.25, -2.0, 4.0], dtype=mx.float16)

    token, distribution = _sample_draft_from_logits(
        logits,
        SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
        np.random.default_rng(7),
        need_distribution=need_distribution,
    )

    assert token == 1
    if need_distribution:
        assert isinstance(distribution, SparseDistribution)
        assert distribution.vocab_size == 5
        np.testing.assert_array_equal(distribution.token_ids, np.array([1]))
        np.testing.assert_array_equal(distribution.probs, np.array([1.0]))
    else:
        assert distribution is None


def test_confidence_metrics_preserve_expected_value_decision():
    metrics = _draft_confidence_metrics(
        mx.array([0.0, 1.0, 3.5, 2.0, -1.0, 0.25, 0.5, 1.5]),
        topk=8,
    )
    policy = ExpectedValueDepthPolicy(
        max_depth=3,
        base_depth=2,
        warmup_full_depth_cycles=0,
        exploration_interval=0,
    )
    policy._attempt_counts[2] = 1
    policy._cycles_observed = 1

    decision = policy.should_continue_after_draft(
        drafted_depth=2,
        max_depth=3,
        draft_metrics=metrics,
    )

    assert math.isfinite(float(decision["confidence_factor"]))
    assert decision["drafted_depth"] == 2
    assert decision["next_depth"] == 3
    assert decision["reason"] in {"ev_pass", "ev_fail"}
