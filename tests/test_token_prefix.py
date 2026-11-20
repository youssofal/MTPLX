from __future__ import annotations

import random

import numpy as np
import pytest

from mtplx.token_prefix import _PREFIX_COMPARE_CHUNK, common_prefix_len


def _reference_common_prefix_len(left, right) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if int(left[index]) != int(right[index]):
            return index
    return limit


CHUNK = _PREFIX_COMPARE_CHUNK


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ((), (), 0),
        ((), (1, 2, 3), 0),
        ((1, 2, 3), (), 0),
        ((1, 2, 3), (1, 2, 3), 3),
        ((1, 2, 3), (1, 2), 2),
        ((1, 2), (1, 2, 3), 2),
        ((9, 2, 3), (1, 2, 3), 0),
        ((1, 2, 3), (1, 2, 4), 2),
    ],
)
def test_common_prefix_len_small_cases(left, right, expected):
    assert common_prefix_len(left, right) == expected


@pytest.mark.parametrize(
    "mismatch_at",
    [0, 1, CHUNK - 1, CHUNK, CHUNK + 1, 2 * CHUNK - 1, 2 * CHUNK, 5 * CHUNK + 7],
)
def test_common_prefix_len_finds_mismatch_at_chunk_edges(mismatch_at):
    left = tuple(range(1000, 1000 + 6 * CHUNK))
    right = list(left)
    right[mismatch_at] = -1
    assert common_prefix_len(left, tuple(right)) == mismatch_at


@pytest.mark.parametrize("length", [CHUNK - 1, CHUNK, CHUNK + 1, 8000])
def test_common_prefix_len_identical_and_extended(length):
    tokens = tuple(range(length))
    assert common_prefix_len(tokens, tokens) == length
    assert common_prefix_len(tokens, tokens + (7, 8)) == length
    assert common_prefix_len(tokens + (7, 8), tokens) == length


def test_common_prefix_len_accepts_mixed_lists_tuples_and_numpy_ints():
    tokens = list(range(3 * CHUNK))
    assert common_prefix_len(tokens, tuple(tokens)) == len(tokens)
    assert common_prefix_len([np.int64(t) for t in tokens], tuple(tokens)) == len(tokens)
    shifted = tokens[:CHUNK + 3] + [-5] + tokens[CHUNK + 4 :]
    assert common_prefix_len(tuple(tokens), shifted) == CHUNK + 3


def test_common_prefix_len_matches_reference_on_random_sequences():
    rng = random.Random(1234)
    for _ in range(300):
        shared = rng.randrange(0, 600)
        base = [rng.randrange(0, 50) for _ in range(shared)]
        left = base + [rng.randrange(0, 50) for _ in range(rng.randrange(0, 200))]
        right = base + [rng.randrange(0, 50) for _ in range(rng.randrange(0, 200))]
        assert common_prefix_len(tuple(left), right) == _reference_common_prefix_len(left, right)


def test_every_module_shares_the_implementation():
    from mtplx import engine_session, session_bank
    from mtplx.server import openai
    from mtplx.cache_bank import cold_tier

    assert session_bank.common_prefix_len is common_prefix_len
    assert engine_session.common_prefix_len is common_prefix_len
    assert cold_tier.common_prefix_len is common_prefix_len
    assert openai.common_prefix_len is common_prefix_len
