"""The device round's top-p, top-k and min-p: every row it decides keeps the product's own kept set.

The reference is ``mtplx.sampling.apply_top_p_top_k`` on the float64 tempered softmax (then the
float32 min-p floor of ``ClientControlledDraw.rows``), through ``nucleus.kept_reference``. A row
whose crossing lies within the error margin, or whose boundary bins overflow, must say so (its
status) instead of deciding.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():  # pragma: no cover - the kernels are Metal
    pytest.skip("Metal is not available", allow_module_level=True)

from mtplx.models.inkling.fast import nucleus  # noqa: E402
from mtplx.models.inkling.fast import round as fast_round  # noqa: E402


def _rows(rng, rows, vocab, kind):
    if kind == "peaked":
        x = rng.normal(0.0, 2.0, (rows, vocab)).astype(np.float32)
        x[np.arange(rows), rng.integers(0, vocab, rows)] += 12.0
    elif kind == "flat":
        x = rng.normal(0.0, 0.6, (rows, vocab)).astype(np.float32)
    elif kind == "zipf":
        ranks = np.arange(1, vocab + 1, dtype=np.float64)
        base = (-1.1 * np.log(ranks)).astype(np.float32)
        x = np.stack([rng.permutation(base) for _ in range(rows)]) + rng.normal(0, 0.05, (rows, vocab)).astype(np.float32)
    elif kind == "model":  # a head of a few likely tokens over a long tail, as this model's rows are
        x = rng.normal(-4.0, 2.5, (rows, vocab)).astype(np.float32)
        for r in range(rows):
            head = rng.integers(1, 40)
            x[r, rng.choice(vocab, head, replace=False)] = rng.normal(8.0, 2.0, head).astype(np.float32)
    else:  # ties: bfloat16 values, so many entries share a score
        x = rng.normal(0.0, 3.0, (rows, vocab)).astype(np.float32)
        return np.asarray(mx.array(x).astype(mx.bfloat16).astype(mx.float32))
    return x.astype(np.float32)


def _device(x, temperature, top_p, top_k, min_p):
    scores = mx.array(x)
    rows = x.shape[0]
    params = fast_round.law_params(rows, temperature)
    law, _ = fast_round.law_rows(scores, params=params)
    words = mx.array(nucleus.words(top_p=top_p, top_k=top_k, min_p=min_p, temperature=temperature), mx.uint32)
    out, stats, bound = nucleus.filtered(scores, law, params, words)
    mx.eval(out, stats, bound)
    return np.asarray(out), np.asarray(stats), np.asarray(bound)


def _check(x, temperature, top_p, top_k, min_p):
    """Every decided row equals the reference; every other row keeps all and says why. Returns the statuses."""
    out, stats, bound = _device(x, temperature, top_p, top_k, min_p)
    statuses = []
    for r in range(x.shape[0]):
        status = int(bound[r, 2])
        statuses.append(status)
        kept = np.isfinite(out[r])
        if status != nucleus.STATUS_OK:
            assert status in (nucleus.STATUS_BOUNDARY, nucleus.STATUS_OVERFLOW)
            assert kept.all(), "a row the device does not decide keeps every entry"
            continue
        want = nucleus.kept_reference(x[r], temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p)
        assert np.array_equal(kept, want), (r, int(kept.sum()), int(want.sum()))
        # the stats are the masked row's: its log-sum-exp is the kept entries'
        y = out[r][kept].astype(np.float64)
        m = stats[r, :, 0].max()
        s = (stats[r, :, 1] * np.exp(stats[r, :, 0] - m)).sum()
        assert np.isclose(m + np.log(s), y.max() + np.log(np.exp(y - y.max()).sum()), rtol=0, atol=1e-5)
    return statuses


def test_exact_masses_within_their_bound():
    """The float-float masses are within 2**-42 of float64's (the bound the select step budgets is 2**-38)."""
    rng = np.random.default_rng(3)
    worst = 0.0
    for temperature in (1.0, 0.7, 0.3, 0.6, 1.5, 2.0, 0.05, 0.95):
        xm = np.float32(rng.uniform(-20.0, 40.0))
        d = np.concatenate([[0.0], rng.uniform(0.0, 43.0 * temperature, 4000), rng.uniform(0.0, 0.05, 500),
                            np.arange(0, 43 * 256) / 256.0 * temperature + 1e-7])
        x = (np.float64(xm) - d).astype(np.float32)
        got = nucleus.exact_masses(xm, x, temperature=temperature).astype(np.float64)
        exact = (np.float64(xm) - x.astype(np.float64)) / temperature
        want = np.where(exact < 43.0, np.exp(-exact) * 2.0 ** 60, 0.0)
        live = want > 2.0 ** 20
        err = np.abs(got[live] - want[live]) / want[live]
        worst = max(worst, float(err.max()))
        assert (np.abs(got[~live] - want[~live]) <= 2.0 ** 20 * 2.0 ** -40 + 4).all()
    assert worst < 2.0 ** -42, worst


CASES = [
    (1.0, 0.95, 0, 0.0), (1.0, 0.9, 0, 0.0), (1.0, 0.5, 0, 0.0), (1.0, 0.99, 0, 0.0),
    (1.0, 1.0, 40, 0.0), (1.0, 1.0, 1, 0.0), (1.0, 1.0, 1000, 0.0),
    (0.7, 0.9, 20, 0.0), (0.3, 0.95, 0, 0.0), (1.5, 0.8, 64, 0.0), (0.6, 0.95, 20, 0.0),
    (1.0, 0.95, 0, 0.05), (0.8, 0.9, 0, 0.1),
]


@pytest.mark.parametrize("kind", ["peaked", "flat", "model", "zipf", "ties"])
@pytest.mark.parametrize("temperature,top_p,top_k,min_p", CASES)
def test_kept_set_is_the_products(kind, temperature, top_p, top_k, min_p):
    rng = np.random.default_rng(abs(hash((kind, temperature, top_p, top_k, min_p))) % (2 ** 32))
    x = _rows(rng, 6, 20_011, kind)
    statuses = _check(x, temperature, top_p, top_k, min_p)
    # a margin of 2**-34 of the total leaves no realistic row undecided; only boundary bins of more than
    # CAP entries (a flat synthetic tail, or a bfloat16 tie group) go to the host
    assert nucleus.STATUS_BOUNDARY not in statuses or kind == "ties", statuses
    if kind in ("peaked", "model"):
        assert statuses == [nucleus.STATUS_OK] * 6, statuses


def test_dense_tail_is_decided():
    """2000 near-equal tail entries at the crossing (each about 2**-20 of the total, below a float32 cumulative
    mass's error): the exact masses tell their boundary, and it is the host's."""
    vocab = 4096
    x = np.full((1, vocab), -40.0, np.float32)
    x[0, 0] = 0.0
    x[0, 1:2001] = (-14.0 - np.arange(2000) * 1e-6).astype(np.float32)
    for top_p in (0.9993, 0.9995, 0.9997):
        assert _check(x, 1.0, top_p, 0, 0.0) == [nucleus.STATUS_OK]


def test_boundary_rows_say_so():
    """A row whose top-p crossing sits on the margin is refused, never decided with a doubtful set."""
    vocab = 4096
    x = np.full((1, vocab), -60.0, np.float32)
    x[0, :20] = 0.0  # twenty equal entries hold all the mass: top_p 0.5 crosses exactly between two of them
    statuses = _check(x, 1.0, 0.5, 0, 0.0)
    assert statuses == [nucleus.STATUS_BOUNDARY]


def test_full_vocabulary_rows():
    rng = np.random.default_rng(7)
    for kind in ("model", "zipf"):
        x = _rows(rng, 4, 200_058, kind)
        statuses = _check(x, 1.0, 0.95, 0, 0.0)
        if kind == "model":
            assert statuses == [nucleus.STATUS_OK] * 4, statuses
