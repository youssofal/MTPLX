"""Greedy draft contracts used by expected-value depth selection."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

import mtplx.generation as generation
from mtplx.adaptive import ExpectedValueDepthPolicy
from mtplx.generation import (
    _can_combine_greedy_draft_read,
    _draft_confidence_metrics,
    _greedy_draft_token_and_metrics,
    _sample_draft_from_logits,
)
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


@pytest.mark.parametrize("need_distribution", [False, True])
def test_combined_greedy_read_matches_separate_operations(need_distribution):
    logits = mx.array(
        [1.0, 4.0, 3.25, -2.0, 4.0, 2.0, -1.0, 0.0, 1.5],
        dtype=mx.float16,
    )
    sampler = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
    expected_token, expected_distribution = _sample_draft_from_logits(
        logits,
        sampler,
        np.random.default_rng(7),
        need_distribution=need_distribution,
    )
    expected_metrics = _draft_confidence_metrics(logits)

    token, distribution, metrics = _greedy_draft_token_and_metrics(
        logits.reshape(1, 1, -1),
        need_distribution=need_distribution,
    )

    assert token == expected_token == 1
    assert metrics == expected_metrics
    if need_distribution:
        assert isinstance(distribution, SparseDistribution)
        np.testing.assert_array_equal(
            distribution.token_ids,
            expected_distribution.token_ids,
        )
        np.testing.assert_array_equal(
            distribution.probs,
            expected_distribution.probs,
        )
    else:
        assert distribution is expected_distribution is None


def test_combined_greedy_read_uses_one_sync(monkeypatch):
    original_eval = generation._eval
    evaluations = []

    def audited_eval(*values, **kwargs):
        evaluations.append(values)
        return original_eval(*values, **kwargs)

    monkeypatch.setattr(generation, "_eval", audited_eval)

    token, _, _ = _greedy_draft_token_and_metrics(
        mx.array([[[1.0, 4.0, 3.25, -2.0]]], dtype=mx.float16),
        need_distribution=False,
    )

    assert token == 1
    assert len(evaluations) == 1
    assert len(evaluations[0]) == 2
    assert evaluations[0][0].ndim == 0
    assert tuple(evaluations[0][1].shape) == (4,)
    assert evaluations[0][1].dtype == mx.float32


def _combined_read_kwargs() -> dict:
    return {
        "confidence_metrics_required": True,
        "adaptive_width_policy": None,
        "target_prefix_route": None,
        "correction_cache_enabled": False,
        "adapter_ensemble_q": False,
        "mtp_topk_reranker": None,
    }


def test_combined_greedy_read_accepts_opt_in_confidence_lane():
    assert _can_combine_greedy_draft_read(
        SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
        **_combined_read_kwargs(),
    )


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("confidence_metrics_required", False),
        ("adaptive_width_policy", object()),
        ("target_prefix_route", object()),
        ("correction_cache_enabled", True),
        ("adapter_ensemble_q", True),
        ("mtp_topk_reranker", object()),
    ],
)
def test_combined_greedy_read_rejects_alternate_selectors(option, value):
    options = _combined_read_kwargs()
    options[option] = value

    assert not _can_combine_greedy_draft_read(
        SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
        **options,
    )


def test_combined_greedy_read_requires_greedy_draft_sampler():
    assert not _can_combine_greedy_draft_read(
        SamplerConfig(temperature=0.7, top_p=0.9, top_k=20),
        **_combined_read_kwargs(),
    )


def test_expected_value_generation_preserves_control_trace(monkeypatch):
    from test_generation_sustained import AcceptingTinyMTPModel, _runtime

    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "off")
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")

    def run_once():
        return generation.generate_mtpk(
            _runtime(AcceptingTinyMTPModel()),
            [0],
            max_tokens=8,
            sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
            speculative_depth=3,
            verify_strategy="batched",
            mtp_history_policy="cycle",
            stop_token_ids=set(),
            adaptive_policy=ExpectedValueDepthPolicy(max_depth=3, base_depth=2),
        )

    original_combined_read = generation._greedy_draft_token_and_metrics
    combined_calls = 0

    def audited_combined_read(*args, **kwargs):
        nonlocal combined_calls
        combined_calls += 1
        return original_combined_read(*args, **kwargs)

    monkeypatch.setattr(
        generation,
        "_greedy_draft_token_and_metrics",
        audited_combined_read,
    )
    combined = run_once()
    assert combined_calls > 0
    assert (
        combined.stats.draft_core["greedy_confidence_sync_calls"]
        == combined_calls
    )
    assert (
        combined.stats.draft_core["greedy_confidence_token_reuses"]
        == combined_calls
    )

    monkeypatch.setattr(
        generation,
        "_can_combine_greedy_draft_read",
        lambda *_args, **_kwargs: False,
    )
    control = run_once()
    assert control.stats.draft_core["greedy_confidence_sync_calls"] == 0
    assert control.stats.draft_core["greedy_confidence_token_reuses"] == 0

    def policy_trace(output):
        return [
            {
                "depth": draft["depth"],
                "token": draft["token"],
                "top2_margin": draft.get("top2_margin"),
                "top1_prob_topk": draft.get("top1_prob_topk"),
                "entropy_topk": draft.get("entropy_topk"),
                "policy_continue": draft.get("policy_continue"),
            }
            for event in output.stats.events
            for draft in event.get("drafts", [])
        ]

    assert combined.tokens == control.tokens
    assert combined.stats.drafted_by_depth == control.stats.drafted_by_depth
    assert combined.stats.accepted_by_depth == control.stats.accepted_by_depth
    assert combined.stats.rejected_drafts == control.stats.rejected_drafts
    assert policy_trace(combined) == policy_trace(control)


@pytest.mark.parametrize(
    ("threshold", "expect_full_reuse"),
    [(0.0, True), (100.0, False)],
)
def test_margin_lane_reports_joint_read_engagement(
    monkeypatch,
    threshold,
    expect_full_reuse,
):
    from test_generation_sustained import AcceptingTinyMTPModel, _runtime

    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "off")
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")

    output = generation.generate_mtpk(
        _runtime(AcceptingTinyMTPModel()),
        [0],
        max_tokens=8,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
        speculative_depth=3,
        verify_strategy="batched",
        mtp_history_policy="cycle",
        stop_token_ids=set(),
        draft_margin_threshold=threshold,
    )

    calls = output.stats.draft_core["greedy_confidence_sync_calls"]
    reuses = output.stats.draft_core["greedy_confidence_token_reuses"]
    assert calls > 0
    if expect_full_reuse:
        assert reuses == calls
    else:
        assert 0 < reuses < calls


def test_default_greedy_lane_reports_no_confidence_sync_engagement(monkeypatch):
    from test_generation_sustained import AcceptingTinyMTPModel, _runtime

    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "off")
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")

    output = generation.generate_mtpk(
        _runtime(AcceptingTinyMTPModel()),
        [0],
        max_tokens=8,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
        speculative_depth=3,
        verify_strategy="batched",
        mtp_history_policy="cycle",
        stop_token_ids=set(),
    )

    assert output.stats.draft_core["greedy_confidence_sync_calls"] == 0
    assert output.stats.draft_core["greedy_confidence_token_reuses"] == 0
