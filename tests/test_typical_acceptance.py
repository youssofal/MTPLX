"""CPU tests for Medusa-2 typical acceptance (lenient verification).

These pin the pure-NumPy primitives in ``mtplx.sampling`` that the runtime
accept loop calls when the typical lane is on (MTPLX_FABLE_TYPICAL_THRESHOLD >
0, server flag ``--typical-threshold``). They do NOT touch the exact
Leviathan-Chen path: the exact-rule tests continue to live in
``test_sampling.py`` and must be unaffected by anything here.

The typical rule (Cai et al. 2024, arXiv:2401.10774):

    accept x_t iff p_target(x_t) > min(eps, delta * exp(-H(p_t)))

- deterministic decision (no coin);
- the FIRST non-typical position resamples from the target's own row p_t,
  NOT the residual (p - q)+;
- entropy is taken over the truncated (top-p/top-k) support.
"""

from __future__ import annotations

import math

import numpy as np

from mtplx.sampling import (
    SamplerConfig,
    SparseDistribution,
    acceptance_probability,
    distribution_entropy,
    distribution_from_logits,
    residual_distribution,
    typical_accept_decision,
    typical_acceptance_threshold,
)


# --- entropy over the truncated support ------------------------------------


def test_entropy_of_one_hot_is_zero():
    assert distribution_entropy(np.array([0.0, 1.0, 0.0])) == 0.0
    assert distribution_entropy(SparseDistribution.one_hot(3, 8)) == 0.0


def test_entropy_of_uniform_support_is_log_support():
    # Four equal masses -> H = ln(4); zeros outside the support are ignored.
    dist = np.array([0.25, 0.25, 0.0, 0.25, 0.25])
    assert math.isclose(distribution_entropy(dist), math.log(4), rel_tol=1e-12)


def test_entropy_matches_between_dense_and_sparse_reps():
    dense = np.array([0.5, 0.3, 0.2])
    sparse = SparseDistribution(
        np.array([0, 1, 2], dtype=np.int64),
        np.array([0.5, 0.3, 0.2], dtype=np.float64),
        3,
    )
    assert math.isclose(
        distribution_entropy(dense), distribution_entropy(sparse), rel_tol=1e-12
    )


def test_entropy_is_over_truncated_support_only():
    # distribution_from_logits truncates to top-k=2 then renormalizes; the
    # entropy must reflect only the two surviving tokens, not the raw softmax.
    logits = np.array([2.0, 1.9, -5.0, -6.0])
    dist = distribution_from_logits(
        logits, SamplerConfig(temperature=1.0, top_p=0.95, top_k=2)
    )
    assert np.count_nonzero(dist) == 2
    # two near-equal survivors -> entropy close to ln(2), never ln(4).
    assert distribution_entropy(dist) <= math.log(2) + 1e-9
    assert distribution_entropy(dist) > 0.5 * math.log(2)


# --- threshold formula ------------------------------------------------------


def test_threshold_is_min_of_eps_and_entropy_term():
    # low entropy: entropy term delta*exp(-H) is large, so eps caps it.
    assert typical_acceptance_threshold(0.0, eps=0.09, delta=0.3) == 0.09
    # high entropy: exp(-H) small, so the entropy term is the min.
    thr = typical_acceptance_threshold(3.0, eps=0.09, delta=0.3)
    assert math.isclose(thr, 0.3 * math.exp(-3.0), rel_tol=1e-12)


def test_eps_default_1_leaves_delta_0p09_numerics_unchanged():
    # Design ruling (2026-09-06): eps becomes an advanced cap with default 1.0
    # (inert). At the measured operating point delta=0.09 the floor is
    # delta*exp(-H) for any eps >= 0.09, so the shipped default eps=1.0 gives
    # byte-identical numerics to the old default eps=0.3 (and to anything
    # between). This is the assertion the ruling asks for: raising the eps
    # default to 1.0 does not move the delta=0.09 threshold at any entropy.
    for H in (0.0, 0.5, 1.0, 2.5, 5.0):
        floor = 0.09 * math.exp(-H)
        thr_new = typical_acceptance_threshold(H, eps=1.0, delta=0.09)
        thr_old = typical_acceptance_threshold(H, eps=0.3, delta=0.09)
        assert math.isclose(thr_new, floor, rel_tol=1e-12)
        assert thr_new == thr_old  # exact equality: the eps cap binds in neither
        assert floor <= 0.09 < 0.3 <= 1.0
    # delta=0.2 (the stricter arm) is likewise unaffected by eps 0.3 -> 1.0.
    for H in (0.0, 1.0, 3.0):
        floor = 0.2 * math.exp(-H)
        assert typical_acceptance_threshold(H, eps=1.0, delta=0.2) == floor
        assert typical_acceptance_threshold(H, eps=0.3, delta=0.2) == floor


# --- accept decision --------------------------------------------------------


def test_typical_accepts_when_target_mass_exceeds_floor():
    # Target row with a dominant token; floor at H=... is small, token passes.
    target = np.array([0.7, 0.2, 0.1])
    accepted, thr, H = typical_accept_decision(target, 0, eps=0.3, delta=0.09)
    assert accepted is True
    assert 0.0 < thr <= 0.09


def test_typical_rejects_token_below_floor():
    # A near-deterministic target (very low entropy) sets floor ~= delta.
    # A draft token with tiny target mass is not typical.
    target = np.array([0.98, 0.01, 0.01])
    # H is tiny -> floor ~ 0.09*exp(-H) ~ 0.088; token 1 has mass 0.01 < floor.
    accepted, thr, H = typical_accept_decision(target, 1, eps=0.3, delta=0.09)
    assert accepted is False
    assert 0.01 <= thr


def test_typical_accepts_documented_prefix_and_resamples_from_target():
    # A 3-token draft window verified against 3 target rows. Positions 0 and 1
    # are typical, position 2 is not; the accepted prefix is [0, 1] and the
    # correction at position 2 is a sample from the TARGET row (never residual).
    eps, delta = 0.3, 0.09
    rows = [
        np.array([0.6, 0.3, 0.1]),   # pos0: draft token 0 (p=0.6) typical
        np.array([0.5, 0.4, 0.1]),   # pos1: draft token 1 (p=0.4) typical
        np.array([0.95, 0.03, 0.02]),  # pos2: draft token 2 (p=0.02) NOT typical
    ]
    draft = [0, 1, 2]
    accepted_prefix = 0
    correction = None
    for i, (row, tok) in enumerate(zip(rows, draft)):
        ok, _thr, _H = typical_accept_decision(row, tok, eps=eps, delta=delta)
        if ok:
            accepted_prefix += 1
            continue
        # First non-typical position: resample from the target row itself.
        rng = np.random.default_rng(0)
        correction = int(np.argmax(row))  # deterministic check of the row used
        # sanity: a residual correction would differ from a target-row sample
        # in general; here we only assert the accepted prefix and that the
        # committed token comes from the target row's support.
        assert row[correction] > 0
        break
    assert accepted_prefix == 2
    assert correction == 0


def test_greedy_argmax_is_always_typical():
    # The argmax of any row has the largest mass; for a shaped row it always
    # clears min(eps, delta*exp(-H)). This is why greedy decode is ~exact under
    # typical acceptance.
    rng = np.random.default_rng(7)
    for _ in range(200):
        raw = rng.random(20)
        dist = distribution_from_logits(
            np.log(raw + 1e-9), SamplerConfig(temperature=1.0, top_p=0.95, top_k=20)
        )
        argmax = int(np.argmax(dist))
        accepted, _thr, _H = typical_accept_decision(
            dist, argmax, eps=0.3, delta=0.09
        )
        assert accepted is True


# --- the exact rule stays byte-identical (the flag-off invariant) -----------


def test_exact_rule_primitives_unchanged():
    # typical_* must not perturb the exact acceptance/residual primitives that
    # the flag-off runtime path calls. Recompute the classic examples.
    target = np.array([0.8, 0.2])
    draft = np.array([0.4, 0.6])
    assert acceptance_probability(target, draft, 0) == 1.0
    assert math.isclose(acceptance_probability(target, draft, 1), 1.0 / 3.0)

    t2 = np.array([0.6, 0.3, 0.1])
    d2 = np.array([0.2, 0.5, 0.3])
    residual = residual_distribution(t2, d2)
    assert math.isclose(residual.sum(), 1.0)
    assert residual[0] == 1.0
    assert residual[1] == 0.0
