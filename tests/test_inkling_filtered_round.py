"""The device decides a filtered law, or leaves its first uncertain position to the host."""

from fractions import Fraction

import mlx.core as mx
import numpy as np
import pytest

from mtplx.generation import ClientControlledDraw
from mtplx.models.inkling.fast import round as rnd
from mtplx.sampling import SamplerConfig, acceptance_probability, apply_top_p_top_k, residual_distribution, softmax


def test_device_accepts_filtered_requests():
    target = SamplerConfig(temperature=1.0, top_p=0.95, top_k=0)
    draft = SamplerConfig(temperature=0.7, top_p=0.9, top_k=20)
    assert rnd.supported(target, draft_sampler=draft) == (True, None)


@pytest.mark.parametrize("draft_cut", [None, (0.74, 3), (1.0, 1)])
def test_enumerate_filtered_ratio_and_residual(draft_cut):
    # Integer weights and complete enumeration of each proposal, the two
    # acceptance cells, and every correction token. Includes disjoint support.
    def distribution(weights, cut):
        scaled = np.array(weights, dtype=np.float64)
        result = apply_top_p_top_k(scaled / scaled.sum(), *cut) if cut else scaled
        kept = [w if p else 0 for w, p in zip(weights, result)]
        return [Fraction(w, sum(kept)) for w in kept]

    target = distribution([9, 2, 6, 1, 0], (0.8, 3))
    draft = distribution([2, 6, 0, 2, 12], draft_cut)
    p, q = np.array(target, dtype=float), np.array(draft, dtype=float)
    emitted = [Fraction(0) for _ in p]
    for proposal, chance in enumerate(draft):
        if not chance:
            continue
        accept = min(Fraction(1), target[proposal] / chance)
        assert acceptance_probability(p, q, proposal) == pytest.approx(float(accept))
        emitted[proposal] += chance * accept
        if accept == 1:
            continue
        excess = [max(Fraction(0), pi - qi) for pi, qi in zip(target, draft)]
        correction = [r / sum(excess) for r in excess]
        np.testing.assert_allclose(residual_distribution(p, q), [float(v) for v in correction], atol=1e-15)
        for token, mass in enumerate(correction):
            emitted[token] += chance * (1 - accept) * mass
    assert emitted == target


@pytest.mark.parametrize("p,k,min_p", [(0.72, 0, 0), (1., 2, 0), (0.99, 40, 0.1)])
def test_adversarial_cut_against_product(p, k, min_p):
    from mtplx.models.inkling.fast import nucleus

    x = np.array([[3, 3, 3, 2, 1, -np.inf], [20, 2, -np.inf, -np.inf, 1, 2],
                  [1, 1.000002, 1, .999999, 1.000001, 1]], dtype=np.float32)
    cfg = SamplerConfig(top_p=p, top_k=k, min_p=min_p, temperature=.7)
    scores = mx.array(x)
    out, _, bounds = rnd.sampled_rows(scores, temperature=cfg.temperature, filters=nucleus.request_words(cfg))
    floors = np.isfinite(np.asarray(ClientControlledDraw(cfg, []).rows(scores)))
    np.testing.assert_array_equal(np.asarray(bounds)[:, 2], np.zeros(len(x)))
    for i, row in enumerate(x):
        want = (apply_top_p_top_k(softmax(row, cfg.temperature), p, k) > 0) & floors[i]
        np.testing.assert_array_equal(np.isfinite(np.asarray(out)[i]), want)


def test_filtered_decisions_match_the_reference_at_each_seed():
    from mtplx.models.inkling.fast import nucleus

    x = np.array([[2, 4, 3, 1], [4, 1, 3, 2]], np.float32)
    qx = np.array([[4, 3, 1, 2]], np.float32)
    cfg = SamplerConfig(top_p=1., top_k=2, temperature=1.)
    words = nucleus.request_words(cfg)
    y, stats, _ = rnd.sampled_rows(mx.array(qx), temperature=1.0, filters=words)
    draft = rnd.DraftLaw(y, stats)
    masked_target = np.where(np.array([apply_top_p_top_k(softmax(row), 1., 2) > 0 for row in x]), x, -np.inf)
    masked_draft = np.where(apply_top_p_top_k(softmax(qx[0]), 1., 2) > 0, qx, -np.inf)
    accepted = set()
    for seed in range(32):
        ids, _ = rnd.sample_rows(y, stats, positions=mx.array([10], mx.uint32), seed=seed, purpose=rnd.PURPOSE_DRAFT)
        got = rnd.read_packet(rnd.decide(mx.array(x), ids, position=10, seed=seed, draft=draft, filters=words), 2)
        want, probabilities = rnd.reference_decide(masked_target, ids.tolist(), position=10, seed=seed,
                                                    draft_rows=masked_draft)
        assert got.status == 0 and got.tokens == want
        np.testing.assert_allclose(got.accept_probability, probabilities, atol=2e-6)
        accepted.add(got.accepted)
    assert accepted == {0, 1}


def test_uncertain_future_row_cannot_restart_an_earlier_draw():
    from mtplx.models.inkling.fast import nucleus

    future = mx.array([[0, 0, 0, 0], [0, 0, nucleus.STATUS_BOUNDARY, 0]], mx.uint32)
    rejected = mx.array([0, 0, 7, -1, 0], mx.int32)
    accepted = mx.array([1, 0, 5, 7, 0], mx.int32)
    assert nucleus.merge_status(rejected, future, 2).tolist() == rejected.tolist()
    assert nucleus.merge_status(accepted, future, 2).tolist() == [1, nucleus.STATUS_BOUNDARY, 5, -1, 0]


def test_filter_threadgroup_refusal_returns_to_host(monkeypatch):
    from mtplx.models.inkling import launch_limits
    from mtplx.models.inkling.fast import nucleus

    monkeypatch.setenv(launch_limits.TEST_LIMIT_ENV, "mtplx_inkling_nucleus_xmax=128")
    cfg = SamplerConfig(top_p=.9, top_k=0, temperature=1.0)
    scores = mx.array([[1., 2., 3., 4.]])
    words = nucleus.request_words(cfg)
    checked = rnd.draw_checked(scores, position=2, seed=3, filters=words)
    with pytest.raises(nucleus.FilterRefused, match="filter_threadgroup"):
        rnd.read_draw(checked)
    decision = rnd.read_packet(rnd.decide(scores, None, position=2, seed=3, filters=words), 1)
    assert decision.status == nucleus.STATUS_GROUP


def test_floor_is_applied_after_nucleus_and_nonfinite_is_preserved():
    from mtplx.models.inkling.fast import nucleus

    x = mx.array(np.log([40, 25, 15, 10, 10]).astype(np.float32)[None, :])
    cfg = SamplerConfig(top_p=.60, top_k=0, min_p=.45, temperature=1.)
    controls = ClientControlledDraw(cfg, [])
    shaped = controls.rows(x, defer_min_p=True)
    out, _, bounds = rnd.sampled_rows(shaped, temperature=1., filters=nucleus.request_words(cfg))
    assert bounds.tolist()[0][2] == 0
    assert np.isfinite(np.asarray(out)[0]).tolist() == [True, True, False, False, False]
    from mtplx.sampling import NonFiniteLogitsError

    bad = mx.array([[2., 1., mx.nan, 3., 4.]])
    with pytest.raises(NonFiniteLogitsError):
        rnd.read_draw(rnd.draw_checked(bad, position=3, seed=10, filters=nucleus.request_words(cfg)), rows=bad)


def test_top_k_beyond_float64_support_is_refused():
    from mtplx.models.inkling.fast import nucleus

    cfg = SamplerConfig(top_p=1., top_k=8, temperature=1.)
    x = mx.array([[5., 2., -900., -np.inf]])
    out, _, bounds = rnd.sampled_rows(x, temperature=1., filters=nucleus.request_words(cfg))
    assert bounds.tolist()[0][2] == nucleus.STATUS_BOUNDARY
    assert out.tolist() == x.tolist()


def test_proposal_reuses_the_same_soft_cap_bits():
    from mtplx.models.inkling.fast import nucleus

    x = mx.array(np.random.default_rng(4).normal(1.0, 10.0, (3, 12017)).astype("f4"))
    cfg = SamplerConfig(temperature=.7, top_p=.92, top_k=40)
    params = rnd.law_params(3, cfg.temperature, cap=30.0)
    words = nucleus.request_words(cfg)
    for scores in (x, x.astype(mx.bfloat16), x.astype(mx.float16)):
        reference = nucleus.filtered(scores, None, params, words)
        actual = rnd.sampled_rows(scores, temperature=cfg.temperature, cap=30.0, params=params, filters=words)
        for got, expected in zip(actual, reference):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def test_packet_uses_a_fixed_group_and_counts_pipeline_refusal(monkeypatch):
    from mtplx.models.inkling import launch_limits
    from mtplx.models.inkling.fast import nucleus

    monkeypatch.setenv(launch_limits.TEST_LIMIT_ENV, "mtplx_inkling_nucleus_merge=16")
    packet = mx.array([1, 0, 7, 9, 0], mx.int32)
    bounds = mx.zeros((2, 4), mx.uint32)
    result = nucleus.merge_status(packet, bounds, 2)
    assert result[:2].tolist() == [0, nucleus.STATUS_GROUP]


def test_request_log_describes_the_sampled_device_proposal():
    from mtplx.models.inkling.customer_controls import ProposalSampler
    from mtplx.models.inkling.generation import served_proposals

    cfg = SamplerConfig(temperature=1., top_p=.95, top_k=0)
    host = ProposalSampler(None, 29)
    fields = {**host.stats(8), **served_proposals(True, cfg.temperature, host, 8, draft_config=cfg)}
    assert (fields["method"], fields["temperature"], fields["top_p"], fields["top_k"]) == ("sampled", 1., .95, 0)


def test_top_k_with_min_p_and_no_nucleus_matches_the_host_kept_set():
    from mtplx.models.inkling.fast import nucleus
    from mtplx.sampling import min_p_filter

    # The floor binds in row zero. The k cutoff binds in row one and splits
    # four equal probabilities by ID in row two. No cumulative mass is needed.
    scores = np.log([[16, 10, 7, 5, 1, 0.5],
                     [12, 10, 9, 8, 1, 0.5],
                     [16, 10, 10, 10, 10, 0.5]]).astype(np.float32)
    config = SamplerConfig(temperature=1.0, top_p=1.0, top_k=3, min_p=0.5)
    raw = mx.array(scores)
    words = nucleus.request_words(config)
    assert words.tolist()[2] == 0
    out, _, bounds = rnd.sampled_rows(raw, temperature=config.temperature, filters=words)
    floor = np.isfinite(np.asarray(ClientControlledDraw(config, []).rows(raw)))
    np.testing.assert_array_equal(np.asarray(bounds)[:, 2], [0, 0, 0])
    for i, expected_ids in enumerate(([0, 1], [0, 1, 2], [0, 1, 2])):
        cut = apply_top_p_top_k(softmax(scores[i]), top_p=1.0, top_k=3)
        host = min_p_filter(cut, config.min_p) > 0
        np.testing.assert_array_equal(host, (cut > 0) & floor[i])
        np.testing.assert_array_equal(np.flatnonzero(host), expected_ids)
        np.testing.assert_array_equal(np.isfinite(np.asarray(out)[i]), host)
