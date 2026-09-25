"""Prompt scoring's per-chunk top-K: the block-prefiltered selection returns
what the full float32 log-softmax plus full-vocab argpartition returned.

Values are bitwise equal (both are ``float32(logit) - float32 logsumexp``);
ids are equal wherever the K-th value is not an exact tie, and within exact
ties the order is now defined (ascending token id)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mtplx.generation import (
    _TOP_K_PREFILTER_BLOCK,
    _exact_top_k_ids,
    _logprobs_at,
    _row_logsumexp_f32,
    _sorted_top_k,
)


def _full_log_softmax_top_k(logits: mx.array, k: int, targets: mx.array):
    """The path this replaced, verbatim: full f32 log-softmax, then top-K."""

    logprobs = logits.astype(mx.float32)
    logprobs = logprobs - mx.logsumexp(logprobs, axis=-1, keepdims=True)
    top_idx = mx.argpartition(-logprobs, kth=k - 1, axis=-1)[..., :k]
    top_vals = mx.take_along_axis(logprobs, top_idx, axis=-1)
    target_lp = mx.take_along_axis(logprobs, targets, axis=-1)
    mx.eval(top_idx, top_vals, target_lp)
    idx_np, vals_np = np.array(top_idx), np.array(top_vals)
    order = np.argsort(-vals_np, axis=-1)
    return (
        np.take_along_axis(idx_np, order, axis=-1),
        np.take_along_axis(vals_np, order, axis=-1),
        np.array(target_lp),
    )


def _prefiltered_top_k(logits: mx.array, k: int, targets: mx.array):
    row_lse = _row_logsumexp_f32(logits)
    top_idx = _exact_top_k_ids(logits, k)
    top_vals = _logprobs_at(logits, row_lse, top_idx)
    target_lp = _logprobs_at(logits, row_lse, targets)
    mx.eval(top_idx, top_vals, target_lp)
    idx_np, vals_np = _sorted_top_k(np.array(top_idx), np.array(top_vals))
    return idx_np, vals_np, np.array(target_lp)


def _distinct_logits(rows: int, vocab: int, seed: int) -> mx.array:
    """float32 logits with no two equal values in a row (no ties at all)."""

    rng = np.random.default_rng(seed)
    base = np.arange(vocab, dtype=np.float32) * np.float32(1e-3) - 50.0
    return mx.array(np.stack([rng.permutation(base) for _ in range(rows)]))


@pytest.mark.parametrize("vocab", [63, 64, 1000, 4096 + 37])
@pytest.mark.parametrize("k", [1, 5, 128])
def test_ids_and_values_match_the_full_log_softmax_without_ties(vocab, k):
    k = min(k, vocab)
    logits = _distinct_logits(6, vocab, seed=vocab + k)
    targets = mx.array(np.arange(6)[:, None] * 7 % vocab)

    old_ids, old_vals, old_targets = _full_log_softmax_top_k(logits, k, targets)
    new_ids, new_vals, new_targets = _prefiltered_top_k(logits, k, targets)

    np.testing.assert_array_equal(new_ids, old_ids)
    np.testing.assert_array_equal(new_vals, old_vals)
    np.testing.assert_array_equal(new_targets, old_targets)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("k", [1, 20, 128])
def test_low_precision_logits_match_up_to_exact_ties(dtype, k):
    """bf16 at vocab scale has many exact ties; outside them nothing moves."""

    mx.random.seed(k)
    logits = (mx.random.normal((8, 20_000)) * 3.0).astype(dtype)
    targets = mx.random.randint(0, 20_000, (8, 1))
    raw = np.array(logits.astype(mx.float32))

    old_ids, old_vals, old_targets = _full_log_softmax_top_k(logits, k, targets)
    new_ids, new_vals, new_targets = _prefiltered_top_k(logits, k, targets)

    np.testing.assert_array_equal(new_vals, old_vals)
    np.testing.assert_array_equal(new_targets, old_targets)
    old_raw = np.take_along_axis(raw, old_ids, axis=-1)
    new_raw = np.take_along_axis(raw, new_ids, axis=-1)
    np.testing.assert_array_equal(new_raw, old_raw)
    for row in range(raw.shape[0]):
        kth = new_raw[row, -1]
        # Ids may differ only where the logit equals the K-th value.
        differing = set(new_ids[row]) ^ set(old_ids[row])
        assert all(raw[row, token] == kth for token in differing)


def test_exact_ties_are_ordered_by_ascending_token_id():
    logits = mx.full((1, 300), -5.0)
    tied = [250, 7, 131, 64]
    logits[0, mx.array(tied)] = 3.0
    logits[0, 99] = 4.0

    ids, vals, _ = _prefiltered_top_k(logits, 5, mx.array([[0]]))

    assert ids[0].tolist() == [99, 7, 64, 131, 250]
    assert vals[0, 1] == vals[0, 4]


def test_minus_infinity_logits_are_never_chosen_over_finite_ones():
    vocab = 3 * _TOP_K_PREFILTER_BLOCK + 11
    logits = mx.full((2, vocab), -mx.inf)
    finite = [0, vocab - 1, 2 * _TOP_K_PREFILTER_BLOCK + 3]
    logits[:, mx.array(finite)] = mx.array([1.0, 2.0, 0.5])

    ids, vals, targets = _prefiltered_top_k(
        logits, 3, mx.array([[vocab - 1], [5]])
    )

    assert ids.tolist() == [[vocab - 1, 0, finite[2]]] * 2
    assert np.isfinite(vals).all()
    assert targets[1, 0] == -np.inf


def test_top_k_clustered_in_one_block_or_in_the_ragged_tail_is_found():
    block = _TOP_K_PREFILTER_BLOCK
    vocab = 40 * block + 9
    logits = mx.zeros((2, vocab))
    logits[0, 5 * block : 5 * block + 8] = mx.arange(8, dtype=mx.float32) + 1.0
    logits[1, vocab - 9 :] = mx.arange(9, dtype=mx.float32) + 1.0

    ids, _vals, _ = _prefiltered_top_k(logits, 8, mx.array([[0], [0]]))

    assert sorted(ids[0].tolist()) == list(range(5 * block, 5 * block + 8))
    assert sorted(ids[1].tolist()) == list(range(vocab - 8, vocab))


def test_k_equal_to_vocab_returns_every_token():
    logits = _distinct_logits(3, 100, seed=1)

    ids, vals, _ = _prefiltered_top_k(logits, 100, mx.array([[0], [1], [2]]))

    assert all(sorted(row) == list(range(100)) for row in ids.tolist())
    assert (np.diff(vals, axis=-1) <= 0).all()
