"""Client controls on Inkling's device round (logit_bias, repetition_penalty, min_p).

The device round decides a round by the ratio rule from law rows. The rule is exact for any
target law, so a request with client controls is served by it once its rows carry the
controls (``ClientControlledDraw.rows``):

* every row equals the host draw's shaped row bit for bit (bias and repetition by the same
  float32 operations), row j counting the panel's first j proposals as seen;
* min_p keeps exactly the tokens ``min_p_filter`` keeps on the tempered softmax, the others
  are -inf; NaN and infinite entries are left for the row check;
* on the GPU, the decision on those rows equals the float64 reference with the kernels'
  random numbers, and the first-token draw follows the host's min_p law.
"""

import mlx.core as mx
import numpy as np
import pytest

from mtplx.generation import ClientControlledDraw
from mtplx.models.inkling.fast import round as rnd
from mtplx.sampling import SamplerConfig, min_p_filter

CONTROLLED = SamplerConfig(temperature=0.8, top_p=1.0, top_k=0, min_p=0.1, repetition_penalty=1.3,
                           logit_bias=((5, -100.0), (11, 3.5), (63, 2.25)))


def _logits(rng, rows, vocab):
    # bf16-valued logits like the head's, spread like a real row
    return np.asarray(mx.array(rng.normal(size=(rows, vocab)).astype(np.float32) * 3.0).astype(mx.bfloat16)
                      .astype(mx.float32))


def _host_row(config, prompt, history, row):
    draw = ClientControlledDraw(config, prompt)
    for token in history:
        draw.observe(token)
    return np.asarray(draw.logits(mx.array(row)))


def _host_keep(config, shaped):
    if not config.temperature or not config.min_p:
        return np.ones(shaped.shape, bool)
    y = shaped.astype(np.float64) / config.temperature
    probs = np.exp(y - y.max())
    return min_p_filter(probs / probs.sum(), config.min_p) > 0


@pytest.fixture
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.mark.parametrize("config", [
    CONTROLLED,
    SamplerConfig(temperature=1.0, top_p=1.0, top_k=0, min_p=0.05),
    SamplerConfig(temperature=0.6, top_p=1.0, top_k=0, repetition_penalty=0.8),
    SamplerConfig(temperature=0.0, top_p=1.0, top_k=0, min_p=0.3, logit_bias=((2, 100.0),)),
])
def test_rows_equal_the_host_draws_shaped_rows(cpu, config):
    rng = np.random.default_rng(7)
    vocab, width = 64, 6
    prompt = [1, 2, 3, 40]
    emitted = [7, 8]
    target = _logits(rng, width, vocab)
    proposals = np.array([9, 3, 9, 50, 12], np.int32)
    controls = ClientControlledDraw(config, prompt)
    for token in emitted:
        controls.observe(token)
    got = np.asarray(controls.rows(mx.array(target), mx.array(proposals)))
    for j in range(width):
        want = _host_row(config, prompt, emitted + proposals[:j].tolist(), target[j])
        keep = _host_keep(config, want)
        assert keep.any()
        np.testing.assert_array_equal(got[j][keep], want[keep])
        assert np.all(got[j][~keep] == -np.inf)


def test_min_p_floor_is_the_host_filter_on_many_rows(cpu):
    rng = np.random.default_rng(11)
    for min_p in (0.01, 0.05, 0.2, 0.9, 1.0):
        for temperature in (0.05, 0.3, 1.0, 2.0):
            config = SamplerConfig(temperature=temperature, top_p=1.0, top_k=0, min_p=min_p)
            target = _logits(rng, 8, 4096)
            got = np.asarray(ClientControlledDraw(config, []).rows(mx.array(target)))
            for j in range(8):
                keep = _host_keep(config, target[j])
                assert np.array_equal(got[j] > -np.inf, keep), (min_p, temperature, j)


def test_rows_leave_non_finite_entries_for_the_row_check(cpu):
    config = SamplerConfig(temperature=1.0, top_p=1.0, top_k=0, min_p=0.5)
    row = np.array([[0.0, 5.0, np.nan, 4.0, -np.inf]], np.float32)
    got = np.asarray(ClientControlledDraw(config, []).rows(mx.array(row)))
    assert np.isnan(got[0, 2]) or np.isnan(got[0]).any()
    row = np.array([[0.0, 5.0, np.inf, 4.0]], np.float32)
    got = np.asarray(ClientControlledDraw(config, []).rows(mx.array(row)))
    assert got[0, 2] == np.inf


def test_neutral_proposals_and_no_penalty_change_nothing(cpu):
    config = SamplerConfig(temperature=1.0, top_p=1.0, top_k=0, logit_bias=((1, -2.0),))
    target = _logits(np.random.default_rng(3), 3, 32)
    got = np.asarray(ClientControlledDraw(config, [4, 5]).rows(mx.array(target), mx.array([4, 6], mx.int32)))
    want = target.copy()
    want[:, 1] += np.float32(-2.0)
    np.testing.assert_array_equal(got, want)


# ------------------------------------------------------------------ GPU: the device round

needs_gpu = pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")


@needs_gpu
@pytest.mark.parametrize("k", [1, 3, 7])
def test_device_round_on_controlled_rows_matches_the_host_reference(k):
    mx.set_default_device(mx.gpu)
    rng = np.random.default_rng(300 + k)
    vocab = 32_768
    config = SamplerConfig(temperature=1.0, top_p=1.0, top_k=0, min_p=0.02, repetition_penalty=1.2,
                           logit_bias=((5, -100.0), (11, 3.5)))
    disagreements = 0
    for trial in range(30):
        target = _logits(rng, k + 1, vocab)
        draft = target[:k] + rng.normal(size=(k, vocab)).astype(np.float32) * 0.6
        seed = int(rng.integers(0, 2**63))
        position = int(rng.integers(0, 2**31))
        controls = ClientControlledDraw(config, rng.integers(0, vocab, size=400).tolist())
        dlaw, dstats = rnd.law_rows(mx.array(draft), temperature=1.0)
        positions = mx.array(position, mx.uint32) + mx.arange(k, dtype=mx.uint32)
        props, _ = rnd.sample_rows(dlaw, dstats, positions=positions, seed=seed, purpose=rnd.PURPOSE_DRAFT)
        rows = controls.rows(mx.array(target), props)
        d = rnd.read_packet(rnd.decide(rows, props, position=position, seed=seed,
                                       draft=rnd.DraftLaw(dlaw, dstats)), k + 1)
        assert d.status == 0
        want, probs = rnd.reference_decide(np.asarray(rows), np.asarray(props), position=position,
                                           seed=seed, draft_rows=np.asarray(dlaw))
        disagreements += int(d.tokens != want)
        np.testing.assert_allclose(d.accept_probability[: len(probs)], probs, rtol=1e-4, atol=1e-6)
        # a banned proposal is never kept
        assert all(t != 5 for t in d.tokens)
    assert disagreements == 0, disagreements


@needs_gpu
def test_device_first_token_follows_the_host_min_p_law():
    mx.set_default_device(mx.gpu)
    rng = np.random.default_rng(5)
    vocab = 48
    config = SamplerConfig(temperature=0.7, top_p=1.0, top_k=0, min_p=0.08, repetition_penalty=1.5,
                           logit_bias=((0, -100.0),))
    prompt = [1, 2, 3]
    row = _logits(rng, 1, vocab)[0]
    shaped = _host_row(config, prompt, [], row)
    y = shaped.astype(np.float64) / config.temperature
    probs = np.exp(y - y.max())
    law = min_p_filter(probs / probs.sum(), config.min_p)
    controls = ClientControlledDraw(config, prompt)
    rows = controls.rows(mx.array(np.repeat(row[None, :], 4096, axis=0)))
    counts = np.zeros(vocab)
    for base in range(0, 6 * 4096, 4096):
        tokens = np.asarray(rnd.draw_checked(rows, position=base, seed=99, temperature=config.temperature)[0])
        counts += np.bincount(tokens, minlength=vocab)
    total = counts.sum()
    assert counts[law == 0].sum() == 0
    support = law > 0
    expected = law[support] * total
    chi2 = float(((counts[support] - expected) ** 2 / expected).sum())
    # chi-square with (support - 1) degrees of freedom, far tail
    assert chi2 < 3.0 * support.sum() + 30, (chi2, support.sum())
