"""Class B law gate for the GPU round's sampler, on the CPU (AGENTS-REFERENCE, Exactness proofs).

The device kernels and this file share one generator (Philox-4x32-10 keyed by seed, counter
(vocabulary index, absolute position, purpose, 0)) and one sampler (the exponential race).
Here the sampler runs in float64 NumPy, vectorised over a million independent draws:

* the race draws from its law (chi-square, 1e6 draws);
* the ratio rule's first emitted token follows p for a sampled draft law, a draft law at
  another temperature, and a point-mass (prompt-copy) proposal (1e6 draws each), and the
  acceptance rate equals 1 - TV(p, q);
* three output tokens produced by repeated rounds on a bigram target and a bigram drafter
  follow the target chain law exactly (1e6 sequences, 512 cells), which also checks that
  keying random numbers by absolute position stays exact when positions are proposed
  again after a rejection.

Each is compared with the product's own law (mtplx.sampling.speculative_output_marginal,
residual_distribution).
"""

import json
import os

import numpy as np
import pytest

from mtplx.models.inkling.fast import round as rnd
from mtplx.sampling import residual_distribution, speculative_output_marginal

DRAWS = 1_000_000
CHUNK = 125_000


def _record(name, **fields):
    """Append one result line to $INKLING_L4_LAW_RECEIPT (JSON lines) when it is set."""
    path = os.environ.get("INKLING_L4_LAW_RECEIPT")
    if path:
        with open(path, "a") as out:
            out.write(json.dumps({"test": name, "draws": DRAWS, **fields}) + "\n")


def _q9999(df):
    """Chi-square 0.9999 quantile (Wilson-Hilferty): a false alarm once in ten thousand runs."""
    z = 3.719
    return df * (1 - 2 / (9 * df) + z * np.sqrt(2 / (9 * df))) ** 3


def _softmax(y):
    z = np.exp(y - np.max(y, axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def _logs(seed, positions, purpose, vocab):
    k0, k1 = rnd.seed_pair(seed)
    j = np.arange(vocab, dtype=np.uint64)[None, :]
    r = rnd.philox_np(j, np.asarray(positions, np.uint64)[:, None], purpose, 0, k0, k1)
    return np.log(rnd.exponential_np(r[0], r[1]))


def _accept_e(seed, positions):
    k0, k1 = rnd.seed_pair(seed)
    r = rnd.philox_np(0, np.asarray(positions, np.uint64), rnd.PURPOSE_ACCEPT, 0, k0, k1)
    return rnd.exponential_np(r[0], r[1])


def _race(logw, logs):
    with np.errstate(invalid="ignore"):
        return np.argmax(logw[None, :] - logs if logw.ndim == 1 else logw - logs, axis=1)


def _chi2(counts, p, min_expected=5.0):
    """Pearson chi-square with cells expected below ``min_expected`` pooled into one;
    returns (statistic, degrees of freedom)."""
    counts = np.asarray(counts, np.float64).reshape(-1)
    expected = np.asarray(p, np.float64).reshape(-1) * counts.sum()
    big = expected >= min_expected
    obs = list(counts[big])
    exp = list(expected[big])
    if (~big).any():
        obs.append(counts[~big].sum())
        exp.append(expected[~big].sum())
    obs, exp = np.array(obs), np.array(exp)
    return float(((obs - exp) ** 2 / exp).sum()), len(obs) - 1


def _real_shaped(vocab, seed):
    rng = np.random.default_rng(seed)
    # a head of a few likely tokens and a long tail, as a coding model's rows look
    y = np.sort(rng.gumbel(size=vocab))[::-1] * 2.2
    y[:3] += np.array([3.0, 2.0, 1.2])
    return y - y.max()


def test_race_draws_its_law_1e6():
    vocab = 48
    y = _real_shaped(vocab, 1)
    p = _softmax(y)
    counts = np.zeros(vocab)
    for start in range(0, DRAWS, CHUNK):
        pos = np.arange(start, start + CHUNK)
        counts += np.bincount(_race(y, _logs(5, pos, rnd.PURPOSE_TARGET, vocab)), minlength=vocab)
    stat, df = _chi2(counts, p)
    _record("race_draws_its_law", chi2=stat, df=df, q9999=_q9999(df))
    assert stat < _q9999(df), (stat, df)


@pytest.mark.parametrize("case", ["sampled", "other_temperature", "point_mass"])
def test_ratio_rule_first_token_is_p_1e6(case):
    vocab = 48
    y = _real_shaped(vocab, 2)
    p = _softmax(y)
    rng = np.random.default_rng(3)
    if case == "sampled":
        yq = y + rng.normal(size=vocab) * 0.9
    elif case == "other_temperature":
        yq = (y + rng.normal(size=vocab) * 0.5) / 0.6  # an explicit draft temperature 0.6
    else:
        yq = None
    q = _softmax(yq) if yq is not None else None
    counts = np.zeros(vocab)
    accepted_total = 0
    seed = 0xFEEDFACECAFEBEEF
    for start in range(0, DRAWS, CHUNK):
        pos = np.arange(start, start + CHUNK)
        if q is None:
            x = np.full(CHUNK, 7)
            qx = np.ones(CHUNK)
            res = p.copy()
            res[7] = 0.0
            res /= res.sum()
        else:
            x = _race(yq, _logs(seed, pos, rnd.PURPOSE_DRAFT, vocab))
            qx = q[x]
            res = residual_distribution(p, q)
        a = np.minimum(1.0, p[x] / qx)
        accept = (a >= 1.0) | (_accept_e(seed, pos) > -np.log(a))
        with np.errstate(divide="ignore"):
            corr = _race(np.log(res), _logs(seed, pos, rnd.PURPOSE_TARGET, vocab))
        out = np.where(accept, x, corr)
        counts += np.bincount(out, minlength=vocab)
        accepted_total += int(accept.sum())
    stat, df = _chi2(counts, p)
    rate = accepted_total / DRAWS
    want = (1 - 0.5 * np.abs(p - q).sum()) if q is not None else p[7]
    _record(f"ratio_first_token_{case}", chi2=stat, df=df, q9999=_q9999(df), accept_rate=rate,
            accept_rate_expected=float(want))
    assert stat < _q9999(df), (stat, df)
    if q is not None:
        # the product's oracle: the rule's exact marginal is p, and acceptance is 1 - TV
        np.testing.assert_allclose(speculative_output_marginal(p, q), p, atol=1e-12)
        tv = 0.5 * np.abs(p - q).sum()
        rate = accepted_total / DRAWS
        assert abs(rate - (1 - tv)) < 5 * np.sqrt(tv * (1 - tv) / DRAWS)
    else:
        rate = accepted_total / DRAWS
        assert abs(rate - p[7]) < 5 * np.sqrt(p[7] * (1 - p[7]) / DRAWS)


def _decide_rows(rows_p, rows_q, x, pos, seed):
    """Vectorised ratio decision for a batch of panels: rows_p (N, k+1, V), rows_q (N, k, V)."""
    n_trials, width, vocab = rows_p.shape
    k = width - 1
    accepted = np.zeros(n_trials, np.int64)
    alive = np.ones(n_trials, bool)
    out = np.full((n_trials, width), -1)
    for i in range(k):
        p = rows_p[:, i]
        q = rows_q[:, i]
        xi = x[:, i]
        pi = p[np.arange(n_trials), xi]
        qi = q[np.arange(n_trials), xi]
        a = np.minimum(1.0, pi / qi)
        acc = (a >= 1.0) | (_accept_e(seed, pos + i) > -np.log(a))
        res = np.maximum(p - q, 0.0)
        res /= res.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore"):
            corr = _race(np.log(res), _logs(seed, pos + i, rnd.PURPOSE_TARGET, vocab))
        take = alive & acc
        stop = alive & ~acc
        out[take, i] = xi[take]
        accepted[take] += 1
        out[stop, i] = corr[stop]
        alive = take
    with np.errstate(divide="ignore"):
        bonus = _race(np.log(rows_p[:, k]), _logs(seed, pos + k, rnd.PURPOSE_TARGET, vocab))
    out[alive, k] = bonus[alive]
    return accepted, out


def test_three_tokens_over_rounds_follow_the_target_chain_1e6():
    vocab, k, anchor = 8, 3, 0
    rng = np.random.default_rng(17)
    P = _softmax(rng.normal(size=(vocab, vocab)) * 1.4)          # target p(next | last)
    Q = _softmax(np.log(P) + rng.normal(size=(vocab, vocab)) * 0.8)  # drafter q(next | last)
    seed = 1234567
    counts = np.zeros((vocab, vocab, vocab))
    trials = DRAWS
    for start in range(0, trials, CHUNK):
        n = CHUNK
        # each trial is a separate request with its own range of absolute positions
        position = np.full(n, 100, np.int64)
        need = np.ones(n, bool)
        trial_seed = seed + start  # one seed per chunk; positions differ per trial below
        offset = np.arange(n, dtype=np.int64) * 16  # disjoint position ranges per trial
        last = np.full(n, anchor)
        tokens = np.full((n, 3), -1)
        count = np.zeros(n, np.int64)
        while need.any():
            idx = np.nonzero(need)[0]
            m = len(idx)
            pos = position[idx] + offset[idx]
            # proposals from the bigram drafter by the race, positions pos .. pos+k-1
            x = np.zeros((m, k), np.int64)
            rows_q = np.zeros((m, k, vocab))
            prev = last[idx]
            for i in range(k):
                qrow = Q[prev]
                rows_q[:, i] = qrow
                x[:, i] = _race(np.log(qrow), _logs(trial_seed, pos + i, rnd.PURPOSE_DRAFT, vocab))
                prev = x[:, i]
            # target rows conditioned on the drafted prefix
            rows_p = np.zeros((m, k + 1, vocab))
            rows_p[:, 0] = P[last[idx]]
            for i in range(k):
                rows_p[:, i + 1] = P[x[:, i]]
            accepted, out = _decide_rows(rows_p, rows_q, x, pos, trial_seed)
            for t in range(m):
                g = idx[t]
                emitted = out[t, : accepted[t] + 1]
                for tok in emitted:
                    if count[g] < 3:
                        tokens[g, count[g]] = tok
                        count[g] += 1
                last[g] = emitted[-1]
                position[g] += len(emitted)
            need = count < 3
        np.add.at(counts, (tokens[:, 0], tokens[:, 1], tokens[:, 2]), 1)
    law = P[anchor][:, None, None] * P[:, :, None] * P[None, :, :]
    law = law / law.sum()
    stat, df = _chi2(counts, law)
    _record("three_tokens_over_rounds_chain", chi2=stat, df=df, q9999=_q9999(df))
    assert df > 300
    assert stat < _q9999(df), (stat, df)
