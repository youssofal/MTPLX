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


# ------------------------------------------------------------------ CPU: commit count, depth


def _loop_walk(tokens, accepted, stops, remaining):
    """The generation loop's walk over a decided packet (budget, stop), as the host runs it."""
    emitted, kept = [], 1
    for pos, token in enumerate(tokens):
        if len(emitted) == remaining:
            break
        emitted.append(token)
        if pos < accepted:
            kept += 1
        else:
            break
        if token in stops:
            break
    return kept, len(emitted)


def test_commit_count_equals_the_loop_walk():
    rng = np.random.default_rng(5)
    with mx.stream(mx.cpu):
        for case in range(3000):
            rows = int(rng.integers(1, rnd.MAX_ROWS + 1))
            k = rows - 1
            n = int(rng.integers(0, k + 1))
            emitted = [int(t) for t in rng.integers(0, 12, size=n + 1)]
            packet = np.zeros(2 + rows + k, np.int32)
            packet[0] = n
            packet[2: 2 + rows] = emitted + [-1] * (rows - n - 1)
            stops = [int(t) for t in rng.choice(12, size=int(rng.integers(0, 4)), replace=False)]
            remaining = int(rng.integers(1, 10))
            as_array = case % 2 == 0
            kept, count = rnd.commit_count(
                mx.array(packet), rows,
                stops=mx.array(stops or [-2], mx.int32) if as_array else stops,
                remaining=mx.array(remaining, mx.uint32) if as_array else remaining)
            mx.eval(kept, count)
            assert (int(kept), int(count)) == _loop_walk(emitted, n, set(stops), remaining), (
                rows, n, emitted, stops, remaining)


def test_depth_controller_follows_the_prompt_channel_and_is_deterministic():
    from mtplx.models.inkling.fast import draft as fdraft

    reasoning, tool = 200008, 200049
    ctl = fdraft.DepthByYield(cost_ms=fdraft.RC6_COST_MS_2K)
    assert ctl.follow([5, tool, 7, reasoning, 9, 11]) == "reasoning"
    assert ctl.follow([1, 2, 3]) == "reasoning"  # no channel token: stays
    assert ctl.rate() == pytest.approx(ctl.channel_prior["reasoning"])
    # code in a tool call yields more than reasoning: the controller goes deeper there
    shallow = ctl.choose()
    ctl.follow([tool])
    deep = ctl.choose()
    assert deep > shallow
    # a run of fully accepted rounds raises the depth; the budget caps it
    for _ in range(8):
        ctl.observe(deep, deep, [])
    assert ctl.choose() == 7 and ctl.choose(remaining=3) == 2 and ctl.choose(remaining=1) == 0
    # the same history gives the same choices
    def replay():
        c = fdraft.DepthByYield(cost_ms=fdraft.RC6_COST_MS_2K)
        c.follow([reasoning])
        out = []
        for accepted in (0, 1, 3, 0, 7, 7, 2, 0, 1, 5):
            k = c.choose()
            out.append(k)
            c.observe(k, min(accepted, k), [tool] if accepted == 7 else [])
        return out
    assert replay() == replay()


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
    # columns: the row maximum and the log-sum relative to it, kept apart (review 1, finding 4)
    got = np.asarray(lse)
    np.testing.assert_allclose(got[:, 0], m[:, 0], rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(got[:, 1], np.log(np.exp(y - m).sum(axis=1)), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(np.asarray(law), y, rtol=2e-6, atol=2e-5)


@needs_gpu
def test_device_draw_matches_host_race():
    rng = np.random.default_rng(7)
    vocab = 200_058
    x = _rows(rng, 8, vocab)
    seed = 0x1234_5678_9ABC_DEF0
    drawn = np.asarray(rnd.draw_checked(mx.array(x), position=1000, seed=seed))
    assert drawn.shape == (2, 8) and (drawn[1] == rnd.STATUS_OK).all()
    got = drawn[0]
    np.testing.assert_array_equal(np.asarray(rnd.draw(mx.array(x), position=1000, seed=seed)), got)
    for r in range(8):
        logs = rnd.log_exponentials_np(seed, 1000 + r, rnd.PURPOSE_TARGET, vocab)
        y = x[r].astype(np.float64)
        keys = (y - y.max()) - logs
        assert int(got[r]) == int(np.argmax(keys)), r
    greedy = np.asarray(rnd.draw_checked(mx.array(x), position=0, seed=seed, temperature=0))
    np.testing.assert_array_equal(greedy[0], np.argmax(x, axis=1))
    assert rnd.read_draw(rnd.draw_checked(mx.array(x[:1]), position=3, seed=seed)) == [int(np.asarray(
        rnd.draw(mx.array(x[:1]), position=3, seed=seed))[0])]


@needs_gpu
def test_device_draw_reports_rows_that_are_not_a_law():
    """Review 1, finding 3: an all-NaN row, an all minus-infinity row, a single NaN or a +inf
    must not stream a token; the status says so and read_draw raises the product's error."""
    from mtplx.sampling import NonFiniteLogitsError

    rng = np.random.default_rng(3)
    vocab = 4096
    good = rng.normal(size=vocab).astype(np.float32)
    rows = np.stack([good, np.full(vocab, np.nan, np.float32), np.full(vocab, -np.inf, np.float32),
                     good.copy(), good.copy(), good.copy()])
    rows[3, 1234] = np.nan
    rows[4, 7] = np.inf
    rows[5, :4000] = -np.inf  # legitimate masking: still a law over the rest
    for temperature in (1.0, 0.0):
        drawn = np.asarray(rnd.draw_checked(mx.array(rows), position=10, seed=5, temperature=temperature))
        np.testing.assert_array_equal(drawn[1], [0, 1, 1, 1, 1, 0])
        assert 4000 <= drawn[0, 5] < vocab
    with pytest.raises(NonFiniteLogitsError):
        rnd.read_draw(rnd.draw_checked(mx.array(rows[1:2]), position=10, seed=5), rows=mx.array(rows[1:2]))
    assert rnd.read_draw(rnd.draw_checked(mx.array(rows[:1]), position=10, seed=5)) == [
        int(np.asarray(rnd.draw(mx.array(rows[:1]), position=10, seed=5))[0])]


@needs_gpu
def test_device_tiny_temperature_tie_keeps_half_the_mass():
    """Review 1, finding 4, pinned against float64: two tied logits at temperature 1e-7 each
    carry half the mass. With an absolute log normaliser float32 rounded 1e8 + log 2 to 1e8
    and accepted a point-mass proposal for one of them with certainty."""
    vocab = 4096
    logits = np.full((2, vocab), -10.0, np.float32)
    logits[:, 17] = 10.0
    logits[:, 99] = 10.0
    y = logits.astype(np.float64) / 1e-7
    m = y.max(axis=1, keepdims=True)
    p64 = np.exp(y - m) / np.exp(y - m).sum(axis=1, keepdims=True)
    assert abs(p64[0, 17] - 0.5) < 1e-12
    accepted, probs = 0, []
    trials = 4000
    for trial in range(trials):
        packet = rnd.decide(mx.array(logits), mx.array([17], mx.int32), position=1000 + 3 * trial,
                            seed=0xC0FFEE, temperature=1e-7)
        decision = rnd.read_packet(packet, 2)
        assert decision.status == rnd.STATUS_OK
        accepted += decision.accepted
        probs.append(decision.accept_probability[0])
        if decision.accepted == 0:
            assert decision.tokens == [99]  # the residual is the other tied token
    assert abs(np.mean(probs) - p64[0, 17]) < 1e-6, np.mean(probs)
    sigma = np.sqrt(0.25 / trials)
    assert abs(accepted / trials - 0.5) < 5 * sigma, accepted / trials
    # and the product does not serve such a temperature on the device in the first place
    assert not rnd.temperature_served(1e-7) and not rnd.temperature_served(2.5)
    assert all(rnd.temperature_served(t) for t in (0.0, 0.05, 0.6, 1.0, 2.0))


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


@needs_gpu
def test_device_offered_count_equals_the_truncated_panel():
    # a panel of 8 rows deciding only its first k proposals gives the packet of the k+1-row panel
    rng = np.random.default_rng(41)
    vocab = 32_768
    for trial in range(24):
        k = 1 + trial % 7
        target, draft = _panel(rng, 7, vocab, closeness=0.5)
        seed = int(rng.integers(0, 2**63))
        dlaw, dstats = rnd.law_rows(mx.array(draft))
        positions = mx.array(500, mx.uint32) + mx.arange(7, dtype=mx.uint32)
        props, _ = rnd.sample_rows(dlaw, dstats, positions=positions, seed=seed, purpose=rnd.PURPOSE_DRAFT)
        full = rnd.read_packet(rnd.decide(mx.array(target), props, position=500, seed=seed,
                                          draft=rnd.DraftLaw(dlaw, dstats),
                                          offered=mx.array([k], mx.uint32)), 8)
        _, part_stats = rnd.law_rows(dlaw[:k])
        part = rnd.read_packet(rnd.decide(mx.array(target[: k + 1]), props[:k], position=500,
                                          seed=seed, draft=rnd.DraftLaw(dlaw[:k], part_stats)), k + 1)
        assert full.tokens == part.tokens and full.accepted == part.accepted


@needs_gpu
def test_device_choose_depth_follows_confidence():
    from mtplx.models.inkling.fast import draft as fdraft
    cost = fdraft.RC6_COST_MS_2K
    sure = mx.log(mx.array([0.999] * 7, mx.float32))
    unsure = mx.log(mx.array([0.1] * 7, mx.float32))
    mixed = mx.log(mx.array([0.999, 0.999, 0.999, 0.1, 0.999, 0.999, 0.999], mx.float32))
    assert int(np.asarray(fdraft.choose_depth(sure, cost))[0]) == 7
    assert int(np.asarray(fdraft.choose_depth(unsure, cost))[0]) == 1
    assert int(np.asarray(fdraft.choose_depth(mixed, cost))[0]) == 3


@needs_gpu
@pytest.mark.parametrize("temperature", [0.6, 1.3])
def test_device_decide_at_other_temperatures_matches_host(temperature):
    # a client temperature scales both laws; the draft is drawn at the same temperature
    rng = np.random.default_rng(int(temperature * 100))
    vocab = 32_768
    for trial in range(20):
        target, draft = _panel(rng, 3, vocab, closeness=0.6)
        seed = int(rng.integers(0, 2**63))
        dlaw, dstats = rnd.law_rows(mx.array(draft), temperature=temperature)
        positions = mx.array(900, mx.uint32) + mx.arange(3, dtype=mx.uint32)
        props, _ = rnd.sample_rows(dlaw, dstats, positions=positions, seed=seed, purpose=rnd.PURPOSE_DRAFT)
        d = rnd.read_packet(rnd.decide(mx.array(target), props, position=900, seed=seed,
                                       temperature=temperature, draft=rnd.DraftLaw(dlaw, dstats)), 4)
        want, probs = rnd.reference_decide(target, np.asarray(props), position=900, seed=seed,
                                           temperature=temperature, draft_rows=np.asarray(dlaw))
        assert d.tokens == want
        np.testing.assert_allclose(d.accept_probability[: len(probs)], probs, rtol=1e-4, atol=1e-6)


def test_the_wide_decision_leaves_the_narrow_kernel_alone():
    # 2 to 8 rows keep their kernel text (and so their recorded programs); 9 to 32 rows take a
    # copy whose only changes are the row loop and the per-row slots.
    assert "    if (sg < W) {\n" in rnd._DECIDE and "threadgroup int win_r[8];" in rnd._DECIDE
    assert "for (uint sg = first; sg < W; sg += L4_THREADS / 32) {" in rnd._DECIDE_WIDE
    assert "[8]" not in rnd._DECIDE_WIDE and rnd._DECIDE_WIDE.count("[32];") == 5
    with pytest.raises(ValueError):
        rnd.decide(mx.zeros((rnd.WIDE_ROWS + 1, 16)), mx.arange(rnd.WIDE_ROWS, dtype=mx.int32),
                   position=0, seed=1)


def _copy_panel(rng, rows, vocab, margin=7.0):
    """A prompt-copy panel: each copied token is the row's likeliest by ``margin`` logits (most
    of the mass), so walks run long and cross the 8-row kernel's reach."""
    target = _rows(rng, rows, vocab)
    copied = rng.integers(0, vocab, size=rows - 1).astype(np.int32)
    target[np.arange(rows - 1), copied] = target[: rows - 1].max(axis=1) + margin
    return target.astype(np.float32), copied


@needs_gpu
@pytest.mark.parametrize("rows", [9, 13, 17, 25, 32])
def test_device_decides_wide_copy_panels_like_the_host_reference(rows):
    rng = np.random.default_rng(700 + rows)
    vocab = 32_768
    longest = 0
    for trial in range(12):
        target, copied = _copy_panel(rng, rows, vocab)
        if trial % 4 == 3:
            copied[int(rng.integers(0, rows - 1))] = int(rng.integers(0, vocab))  # text it would not copy
        seed = int(rng.integers(0, 2**63))
        position = int(rng.integers(0, 2**31))
        d = rnd.read_packet(rnd.decide(mx.array(target), mx.array(copied), position=position, seed=seed), rows)
        assert d.status == rnd.STATUS_OK
        want, probs = rnd.reference_decide(target, copied, position=position, seed=seed)
        assert d.tokens == want, (trial, d.tokens, want)
        np.testing.assert_allclose(d.accept_probability[: len(probs)], probs, rtol=1e-4, atol=1e-6)
        longest = max(longest, d.accepted)
    assert longest >= rnd.MAX_ROWS  # the walk used row 8 or later: past the 8-row kernel's reach


@needs_gpu
def test_wide_decision_greedy_match_and_offered_equal_the_narrow_panels():
    rng = np.random.default_rng(77)
    vocab = 32_768
    rows = 25
    target, copied = _copy_panel(rng, rows, vocab)
    best = np.argmax(target, axis=1).astype(np.int32)
    # greedy: the walk stops at the first copied token that is not the argmax
    props = best[: rows - 1].copy()
    props[19] = (props[19] + 1) % vocab
    d = rnd.read_packet(rnd.decide(mx.array(target), mx.array(props), position=3, seed=1, temperature=0), rows)
    assert d.accepted == 19 and d.tokens == best[:20].tolist()
    d = rnd.read_packet(rnd.decide(mx.array(target), mx.array(best[: rows - 1]), position=3, seed=1,
                                   temperature=0), rows)
    assert d.accepted == rows - 1 and d.tokens == best.tolist()
    # the seed-match rule on a wide panel against the host reference
    for trial in range(8):
        target, copied = _copy_panel(rng, rows, vocab, margin=6.0)
        seed = int(rng.integers(0, 2**63))
        d = rnd.read_packet(rnd.decide(mx.array(target), mx.array(copied), position=trial * 40, seed=seed,
                                       rule="match"), rows)
        want, probs = rnd.reference_decide(target, copied, position=trial * 40, seed=seed, rule="match")
        assert d.tokens == want
        np.testing.assert_allclose(d.accept_probability[: len(probs)], probs, rtol=1e-4, atol=1e-7)
    # a 25-row panel deciding its first k proposals gives the (k + 1)-row panel's packet, bit for
    # bit: for k up to 7 that is the 8-row kernel's own answer
    for k in (1, 3, 7, 8, 12, 16, 24):
        target, copied = _copy_panel(rng, rows, vocab)
        seed = int(rng.integers(0, 2**63))
        full = np.asarray(rnd.decide(mx.array(target), mx.array(copied), position=1000, seed=seed,
                                     offered=mx.array([k], mx.uint32)))
        part = np.asarray(rnd.decide(mx.array(target[: k + 1]), mx.array(copied[:k]), position=1000,
                                     seed=seed))
        n = int(part[0])
        assert int(full[0]) == n and full[1] == part[1]
        assert full[2: 3 + n].tolist() == part[2: 3 + n].tolist()
        assert full[2 + rows: 2 + rows + k].tolist() == part[2 + k + 1: 2 + k + 1 + k].tolist()


@needs_gpu
def test_device_warm_builds_every_variant():
    rnd.warm()
