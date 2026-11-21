"""The GPU round decision (mtplx/models/inkling/fast/round.py): generator, law and packet.

CPU tests run anywhere. Device tests need Metal and run through the campaign's GPU queue.
"""

import mlx.core as mx
import numpy as np
import pytest

from mtplx.models.inkling.fast import round as rnd
from mtplx.sampling import residual_distribution, speculative_output_marginal


def _device():
    return mx.metal.is_available()


needs_gpu = pytest.mark.skipif(not _device(), reason="needs a Metal GPU")


# ------------------------------------------------------------------ CPU: the generator


def test_philox_known_answers():
    # Philox-4x32-10 known-answer vectors.
    cases = [
        ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
        ((0xFFFFFFFF,) * 4, (0xFFFFFFFF, 0xFFFFFFFF), (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
        ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0),
         (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
    ]
    for ctr, key, want in cases:
        got = rnd.philox_np(*ctr, *key)
        assert tuple(int(v) for v in got) == want


def test_exponential_edges_are_finite_and_positive():
    hi = np.array([0, 0, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xFFFFFFFF], np.uint32)
    lo = np.array([0, 1, 0xFFFFFFFF, 0, 0xFFFFFFFE, 0xFFFFFFFF], np.uint32)
    e = rnd.exponential_np(hi, lo)
    assert np.all(np.isfinite(e)) and np.all(e > 0)
    assert abs(e[0] - 65 * np.log(2)) < 1e-9  # W = 0.5 / 2^64
    assert e[-1] < 1e-19  # W = 1 - 0.5 / 2^64


# ------------------------------------------------------------------ CPU: the law


def _softmax(y):
    z = np.exp(y - y.max())
    return z / z.sum()


def test_product_oracle_returns_the_target_on_real_shaped_laws():
    rng = np.random.default_rng(3)
    for _ in range(20):
        p = _softmax(rng.normal(size=64) * 3)
        q = _softmax(rng.normal(size=64) * 3)
        np.testing.assert_allclose(speculative_output_marginal(p, q), p, atol=1e-12)


def test_reference_decision_law_matches_the_oracle_on_small_vocabulary():
    # The reference decision with the device's random numbers, as a sampler: the emitted
    # first token over many positions follows p (chi-square), and acceptance is 1 - TV.
    rng = np.random.default_rng(11)
    vocab = 16
    logits_p = rng.normal(size=vocab) * 2.0
    logits_q = logits_p + rng.normal(size=vocab) * 1.0
    p, q = _softmax(logits_p), _softmax(logits_q)
    k0, k1 = rnd.seed_pair(29)
    draws = 200_000
    positions = np.arange(draws, dtype=np.uint64)
    # proposals from q by the same race the device runs (purpose DRAFT)
    j = np.arange(vocab, dtype=np.uint64)
    r = rnd.philox_np(j[None, :], positions[:, None], rnd.PURPOSE_DRAFT, 0, k0, k1)
    logs_d = np.log(rnd.exponential_np(r[0], r[1]))
    x = np.argmax(np.log(q)[None, :] - logs_d, axis=1)
    a = np.minimum(1.0, p[x] / q[x])
    ra = rnd.philox_np(0, positions, rnd.PURPOSE_ACCEPT, 0, k0, k1)
    u = np.exp(-rnd.exponential_np(ra[0], ra[1]))
    accepted = u < a
    res = residual_distribution(p, q)
    rt = rnd.philox_np(j[None, :], positions[:, None], rnd.PURPOSE_TARGET, 0, k0, k1)
    logs_t = np.log(rnd.exponential_np(rt[0], rt[1]))
    with np.errstate(divide="ignore"):
        corr = np.argmax(np.log(res)[None, :] - logs_t, axis=1)
    out = np.where(accepted, x, corr)
    counts = np.bincount(out, minlength=vocab)
    expected = p * draws
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    # 15 degrees of freedom: the 0.999 quantile is 37.7
    assert chi2 < 37.7, chi2
    tv = 0.5 * np.abs(p - q).sum()
    assert abs(accepted.mean() - (1 - tv)) < 5 * np.sqrt((1 - tv) * tv / draws) + 1e-3


# ------------------------------------------------------------------ device


_DEBUG_PHILOX = r"""
    uint i = thread_position_in_grid.x;
    uint4 r = l4_philox(uint4(ctr[4 * i], ctr[4 * i + 1], ctr[4 * i + 2], ctr[4 * i + 3]),
                        uint2(key[0], key[1]));
    out[4 * i] = r.x; out[4 * i + 1] = r.y; out[4 * i + 2] = r.z; out[4 * i + 3] = r.w;
    expo[i] = l4_exponential(bits[2 * i], bits[2 * i + 1]);
"""


@needs_gpu
def test_device_philox_and_exponential_match_host():
    kernel = mx.fast.metal_kernel(
        name="mtplx_inkling_l4_debug_philox", input_names=["ctr", "key", "bits"],
        output_names=["out", "expo"], source=_DEBUG_PHILOX, header=rnd._HEADER)
    rng = np.random.default_rng(5)
    n = 4096
    ctr = rng.integers(0, 2**32, size=(n, 4), dtype=np.uint64).astype(np.uint32)
    key = np.array([0xA4093822, 0x299F31D0], np.uint32)
    bits = rng.integers(0, 2**32, size=(n, 2), dtype=np.uint64).astype(np.uint32)
    bits[:6] = [[0, 0], [0, 1], [0x7FFFFFFF, 0xFFFFFFFF], [0x80000000, 0], [0xFFFFFFFF, 0xFFFFFFFE],
                [0xFFFFFFFF, 0xFFFFFFFF]]
    out, expo = kernel(inputs=[mx.array(ctr.reshape(-1)), mx.array(key), mx.array(bits.reshape(-1))],
                       grid=(n, 1, 1), threadgroup=(256, 1, 1),
                       output_shapes=[(4 * n,), (n,)], output_dtypes=[mx.uint32, mx.float32])
    want = rnd.philox_np(ctr[:, 0], ctr[:, 1], ctr[:, 2], ctr[:, 3], int(key[0]), int(key[1]))
    np.testing.assert_array_equal(np.asarray(out).reshape(n, 4), np.stack(want, axis=1))
    e64 = rnd.exponential_np(bits[:, 0], bits[:, 1])
    rel = np.abs(np.asarray(expo, np.float64) - e64) / e64
    assert rel.max() < 4e-6, rel.max()


def _rows(rng, rows, vocab, spread=4.0):
    base = rng.normal(size=vocab).astype(np.float32) * spread
    out = np.stack([base + rng.normal(size=vocab).astype(np.float32) for _ in range(rows)])
    return np.tanh(out / 30.0) * 30.0  # the head's soft cap


@needs_gpu
@pytest.mark.parametrize("vocab", [1, 257, 4096, 200_058])
def test_device_lse_matches_host(vocab):
    rng = np.random.default_rng(vocab)
    x = _rows(rng, 3, vocab)
    law, stats = rnd.law_rows(mx.array(x), temperature=0.7, cap=30.0)
    _, lse = rnd.race(law, stats, modes=mx.zeros((3,), mx.int32),
                      positions=mx.arange(3, dtype=mx.uint32), seed=rnd._seed_words(1),
                      purpose=rnd.PURPOSE_TARGET)
    y = np.tanh(x.astype(np.float64) / 30.0) * 30.0 / 0.7
    m = y.max(axis=1, keepdims=True)
    want = (m + np.log(np.exp(y - m).sum(axis=1, keepdims=True)))[:, 0]
    got = np.asarray(lse)[:, 1]
    np.testing.assert_allclose(got, want, rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(np.asarray(law), y, rtol=2e-6, atol=2e-5)


@needs_gpu
def test_device_draw_matches_host_race():
    rng = np.random.default_rng(7)
    vocab = 200_058
    x = _rows(rng, 8, vocab)
    seed = 0x1234_5678_9ABC_DEF0
    got = np.asarray(rnd.draw(mx.array(x), position=1000, seed=seed))
    for r in range(8):
        logs = rnd.log_exponentials_np(seed, 1000 + r, rnd.PURPOSE_TARGET, vocab)
        y = x[r].astype(np.float64)
        keys = (y - y.max()) - logs
        assert int(got[r]) == int(np.argmax(keys)), r
    greedy = np.asarray(rnd.draw(mx.array(x), position=0, seed=seed, temperature=0))
    np.testing.assert_array_equal(greedy, np.argmax(x, axis=1))


def _panel(rng, k, vocab, closeness):
    target = _rows(rng, k + 1, vocab)
    draft = target[:k] + rng.normal(size=(k, vocab)).astype(np.float32) * closeness
    return target, draft


@needs_gpu
@pytest.mark.parametrize("k", [1, 3, 7])
def test_device_decide_matches_host_reference(k):
    rng = np.random.default_rng(100 + k)
    vocab = 32_768
    disagreements = 0
    trials = 40
    for trial in range(trials):
        target, draft = _panel(rng, k, vocab, closeness=0.6)
        seed = int(rng.integers(0, 2**63))
        position = int(rng.integers(0, 2**31))
        dlaw, dstats = rnd.law_rows(mx.array(draft), temperature=1.0)
        positions = mx.array(position, mx.uint32) + mx.arange(k, dtype=mx.uint32)
        props, _ = rnd.sample_rows(dlaw, dstats, positions=positions, seed=seed,
                                   purpose=rnd.PURPOSE_DRAFT)
        packet = rnd.decide(mx.array(target), props, position=position, seed=seed,
                            draft=rnd.DraftLaw(dlaw, dstats))
        d = rnd.read_packet(packet, k + 1)
        assert d.status == 0
        want, probs = rnd.reference_decide(target, np.asarray(props), position=position,
                                           seed=seed, draft_rows=np.asarray(dlaw))
        disagreements += int(d.tokens != want)
        np.testing.assert_allclose(d.accept_probability[: len(probs)], probs, rtol=1e-4, atol=1e-6)
    assert disagreements == 0, disagreements


@needs_gpu
def test_device_decide_greedy_and_point_mass():
    rng = np.random.default_rng(9)
    vocab = 50_000
    target = _rows(rng, 4, vocab)
    best = np.argmax(target, axis=1)
    # greedy: accept the first two, reject the third
    props = np.array([best[0], best[1], (best[2] + 1) % vocab], np.int32)
    d = rnd.read_packet(rnd.decide(mx.array(target), mx.array(props), position=5, seed=1,
                                   temperature=0), 4)
    assert d.accepted == 2 and d.tokens == [best[0], best[1], best[2]]
    # greedy, all accepted: bonus is the last argmax
    d = rnd.read_packet(rnd.decide(mx.array(target), mx.array(best[:3].astype(np.int32)),
                                   position=5, seed=1, temperature=0), 4)
    assert d.accepted == 3 and d.tokens == best.tolist()
    # point mass (prompt copy) at temperature 1 against the host reference
    for seed in range(30):
        props = rng.integers(0, vocab, size=3).astype(np.int32)
        props[0] = best[0]
        d = rnd.read_packet(rnd.decide(mx.array(target), mx.array(props), position=77,
                                       seed=seed), 4)
        want, _ = rnd.reference_decide(target, props, position=77, seed=seed)
        assert d.tokens == want


@needs_gpu
def test_device_first_token_law_chi_square():
    # One target/draft pair with most mass on a few dozen tokens: the emitted first token
    # over 20,000 independent positions follows p.
    rng = np.random.default_rng(21)
    vocab = 200_058
    y = np.full(vocab, -20.0, np.float32)
    top = rng.choice(vocab, size=40, replace=False)
    y[top] = rng.normal(size=40).astype(np.float32) * 1.5
    target = np.stack([y, y])
    dy = y.copy()
    dy[top] += rng.normal(size=40).astype(np.float32) * 0.8
    draft = dy[None, :]
    dlaw, dstats = rnd.law_rows(mx.array(draft))
    seed = 77
    draws = 20_000
    counts = {}
    accepted = 0
    for pos in range(draws):
        props, _ = rnd.sample_rows(dlaw, dstats, positions=mx.array([pos], mx.uint32), seed=seed,
                                   purpose=rnd.PURPOSE_DRAFT)
        packet = rnd.decide(mx.array(target), props, position=pos, seed=seed,
                            draft=rnd.DraftLaw(dlaw, dstats))
        mx.async_eval(packet)
        v = np.asarray(packet)
        accepted += int(v[0] >= 1)
        counts[int(v[2])] = counts.get(int(v[2]), 0) + 1
    p = _softmax(y.astype(np.float64))
    q = _softmax(dy.astype(np.float64))
    order = np.argsort(-p)[:40]
    obs = np.array([counts.get(int(t), 0) for t in order], np.float64)
    exp = p[order] * draws
    rest_obs = draws - obs.sum()
    rest_exp = draws - exp.sum()
    chi2 = float(((obs - exp) ** 2 / exp).sum() + (rest_obs - rest_exp) ** 2 / max(rest_exp, 1e-9))
    assert chi2 < 80.0, chi2  # 40 degrees of freedom: 0.999 quantile 73.4, slack for the rest bin
    tv = 0.5 * np.abs(p - q).sum()
    assert abs(accepted / draws - (1 - tv)) < 0.02


@needs_gpu
def test_device_match_rule_matches_host_reference():
    rng = np.random.default_rng(31)
    vocab = 32_768
    for trial in range(30):
        target = _rows(rng, 4, vocab, spread=6.0)
        best = np.argmax(target, axis=1).astype(np.int32)
        props = best[:3].copy()
        if trial % 3 == 0:
            props[1] = (props[1] + 7) % vocab
        seed = int(rng.integers(0, 2**63))
        d = rnd.read_packet(rnd.decide(mx.array(target), mx.array(props), position=trial * 11,
                                       seed=seed, rule="match"), 4)
        want, probs = rnd.reference_decide(target, props, position=trial * 11, seed=seed,
                                           rule="match")
        assert d.tokens == want
        np.testing.assert_allclose(d.accept_probability[: len(probs)], probs, rtol=1e-4, atol=1e-7)
