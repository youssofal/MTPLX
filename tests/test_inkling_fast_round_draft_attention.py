"""The drafter's key-sharing attention (``round_program.draft_attention``) against MLX's masked
attention over the concatenated history, the path it replaces: rings (full, partly filled, the
newest slot anywhere) and banks (rows past the count are not history), at the drafter's shapes
(its block is always 8 rows: the anchor and up to 7 proposals, padded), in bfloat16 and in the
FP16 siblings' float16. Float32 inside both; the outputs are 16-bit, so they agree to its rounding."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mtplx.models.inkling.fast.round_program import draft_attention

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal kernel")

HQ, HK, D = 24, 8, 128


def _reference(q, hk, hv, ok, ov, valid):
    rows = q.shape[2]
    mask = mx.concatenate([valid, mx.ones((rows,), mx.bool_)]).reshape(1, 1, 1, -1)
    return mx.fast.scaled_dot_product_attention(
        q, mx.concatenate([hk, ok], axis=2), mx.concatenate([hv, ov], axis=2), scale=D**-0.5, mask=mask
    )


@pytest.mark.parametrize(
    "slots,ring,count,newest",
    [
        (1024, True, 1024, 517),
        (1024, True, 300, 299),
        (1024, True, 1, 0),
        (4096, False, 4096, 0),
        (4096, False, 1500, 0),
        (8192, False, 4150, 0),
    ],
)
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("chunk", [None, 64, 128])
def test_draft_attention_equals_masked_attention(slots, ring, count, newest, dtype, chunk):
    rows = 8
    keys = mx.random.split(mx.random.key(slots + count + rows), 5)
    q = mx.random.normal((1, HQ, rows, D), key=keys[0]).astype(dtype)
    hk = mx.random.normal((1, HK, slots, D), key=keys[1]).astype(dtype)
    hv = mx.random.normal((1, HK, slots, D), key=keys[2]).astype(dtype)
    ok = mx.random.normal((1, HK, rows, D), key=keys[3]).astype(dtype)
    ov = mx.random.normal((1, HK, rows, D), key=keys[4]).astype(dtype)
    j = mx.arange(slots, dtype=mx.uint32)
    count_a, newest_a = mx.array(count, mx.uint32), mx.array(newest, mx.uint32)
    if ring:
        valid = ((newest_a + mx.array(slots, mx.uint32) - j) % mx.array(slots, mx.uint32)) < count_a
    else:
        valid = j < count_a
    want = np.array(_reference(q, hk, hv, ok, ov, valid).astype(mx.float32))
    out = draft_attention(q, hk, hv, ok, ov, count_a, newest_a, ring=ring, chunk=chunk)
    assert out.dtype == dtype
    got = np.array(out.astype(mx.float32))
    assert got.shape == want.shape
    # 16-bit outputs: within two units in the last place of the largest value
    ulp = 2.0**-7 if dtype == mx.bfloat16 else 2.0**-10
    assert np.abs(got - want).max() <= 2 * ulp * max(1.0, np.abs(want).max())


def test_rows_past_a_banks_count_are_never_read():
    rows, slots, count = 8, 4096, 1000
    keys = mx.random.split(mx.random.key(3), 5)
    q = mx.random.normal((1, HQ, rows, D), key=keys[0]).astype(mx.bfloat16)
    hk = mx.random.normal((1, HK, slots, D), key=keys[1]).astype(mx.bfloat16)
    hv = mx.random.normal((1, HK, slots, D), key=keys[2]).astype(mx.bfloat16)
    ok = mx.random.normal((1, HK, rows, D), key=keys[3]).astype(mx.bfloat16)
    ov = mx.random.normal((1, HK, rows, D), key=keys[4]).astype(mx.bfloat16)
    junk = mx.where(mx.arange(slots)[None, None, :, None] >= count, mx.array(float("nan"), mx.bfloat16), hv)
    count_a, newest_a = mx.array(count, mx.uint32), mx.array(0, mx.uint32)
    clean = draft_attention(q, hk, hv, ok, ov, count_a, newest_a, ring=False)
    dirty = draft_attention(q, hk, junk, ok, ov, count_a, newest_a, ring=False)
    assert np.array_equal(np.array(clean.astype(mx.float32)), np.array(dirty.astype(mx.float32)))


def test_a_query_count_the_simdgroups_cannot_share_is_refused():
    q = mx.zeros((1, HQ, 4, D), mx.bfloat16)  # 12 query vectors: not a multiple of 8
    kv = mx.zeros((1, HK, 1024, D), mx.bfloat16)
    own = mx.zeros((1, HK, 4, D), mx.bfloat16)
    with pytest.raises(ValueError):
        draft_attention(q, kv, kv, own, own, mx.array(10, mx.uint32), mx.array(9, mx.uint32), ring=True)


def test_a_chunk_that_is_not_whole_tiles_is_refused():
    q = mx.zeros((1, HQ, 8, D), mx.bfloat16)
    kv = mx.zeros((1, HK, 1024, D), mx.bfloat16)
    own = mx.zeros((1, HK, 8, D), mx.bfloat16)
    with pytest.raises(ValueError):
        draft_attention(q, kv, kv, own, own, mx.array(10, mx.uint32), mx.array(9, mx.uint32), ring=True, chunk=48)
