"""#420 follow-up: the Flash-Next QSA cache family can ride the batched AR lane.

CPU-only. ``QSACache.merge`` returns a ``BatchQSACache`` that keeps one
single-sequence cache per row; the QSA layer loops the rows. These tests pin
the entry contract mlx-lm's BatchGenerator relies on (merge / extend / filter
/ extract / prepare / finalize / state / nbytes), the ragged-prefill trim, the
per-row forward dispatch, the lengths-aware tails for the PLE history and the
short-conv window, and that the startup cache-family probe now admits the
family. No model weights are loaded.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

import mtplx.server.openai as srv
from mtplx.models.qwen4_exp import (
    BatchQSACache,
    QSACache,
    _qsa_batched_forward,
    _tail_by_lengths,
)

D_IDX = 8  # indexer head dim (tiny)
H_KV, D_KV = 2, 4


def _fill(row: QSACache, n_tokens: int, seed: int) -> QSACache:
    """Forward ``n_tokens`` through a row the way the layer does: indexer keys
    at their absolute positions, then the KV append that advances the offset."""
    mx.random.seed(seed)
    keys = mx.random.normal((1, n_tokens, D_IDX)).astype(mx.bfloat16)
    row.write_raw(keys)
    k = mx.random.normal((1, H_KV, n_tokens, D_KV)).astype(mx.bfloat16)
    v = mx.random.normal((1, H_KV, n_tokens, D_KV)).astype(mx.bfloat16)
    row.kv.update_and_fetch(k, v)
    nb = row.offset // row.ratio
    if nb > row.pooled_len:
        blocks = mx.random.normal((1, nb - row.pooled_len, D_IDX)).astype(mx.bfloat16)
        row.write_pooled(blocks, row.pooled_len, nb)
    mx.eval(row.state)
    return row


def test_merge_keeps_one_row_per_sequence_and_history():
    a = _fill(QSACache(4), 10, 1)
    b = QSACache(4)  # fresh (no history)
    merged = QSACache.merge([a, b])
    assert isinstance(merged, BatchQSACache) and len(merged) == 2
    assert merged.rows[0] is a and merged.rows[0].offset == 10 and merged.rows[0].pooled_len == 2
    assert merged.rows[1] is b and merged.rows[1].offset == 0
    # merging batches flattens rows; None stands for a fresh row
    again = QSACache.merge([merged, None])
    assert len(again) == 3 and again.rows[2].offset == 0


def test_extend_filter_extract_are_row_list_operations():
    rows = [_fill(QSACache(4), n, i) for i, n in enumerate((4, 9, 13))]
    cache = QSACache.merge(rows[:2])
    cache.extend(QSACache.merge(rows[2:]))
    assert [r.offset for r in cache.rows] == [4, 9, 13]
    cache.filter(mx.array([2, 0], mx.int32))  # keep rows 2 and 0, in that order
    assert [r.offset for r in cache.rows] == [13, 4]
    extracted = cache.extract(1)
    assert extracted is rows[0] and isinstance(extracted, QSACache)
    assert len(extracted.state) == 4, "extract hands back the single-sequence 4-leaf cache the bank snapshots"
    assert cache.nbytes == rows[2].nbytes + rows[0].nbytes


def test_ragged_prefill_finalize_trims_the_right_padding_per_row():
    long_row = _fill(QSACache(4), 12, 7)
    short_row = _fill(QSACache(4), 12, 8)  # forwarded 12 tokens, 5 of them padding
    cache = QSACache.merge([long_row, short_row])
    cache.prepare(lengths=[12, 7], right_padding=[0, 5])
    cache.finalize()
    assert long_row.offset == 12 and long_row.pooled_len == 3
    assert short_row.offset == 7 and short_row.pooled_len == 1, "pooled blocks past the real length are dropped"
    assert cache._right_padding is None
    raw, pooled = short_row.state[2], short_row.state[3]
    assert raw.shape[1] == 7 and pooled.shape[1] == 1
    cache.finalize()  # idempotent
    assert short_row.offset == 7


def test_state_is_a_flat_evaluable_leaf_list():
    cache = QSACache.merge([_fill(QSACache(4), 5, 3), QSACache(4)])
    leaves = cache.state
    assert all(isinstance(x, mx.array) for x in leaves)
    mx.eval(leaves)  # what BatchGenerator does after every prefill chunk
    with pytest.raises(ValueError):
        cache.state = leaves


def test_batched_forward_runs_each_row_on_its_own_cache_and_slice():
    calls = []

    def attn(x, cache):
        calls.append((x.shape, cache))
        return mx.full((1, x.shape[1], 3), float(cache.offset))

    rows = [_fill(QSACache(4), 6, 1), _fill(QSACache(4), 2, 2), QSACache(4)]
    cache = BatchQSACache(rows)
    x = mx.zeros((3, 5, 16))
    out = _qsa_batched_forward(attn, x, cache)
    assert out.shape == (3, 5, 3)
    assert [c for _, c in calls] == rows and all(s == (1, 5, 16) for s, _ in calls)
    assert out[:, 0, 0].tolist() == [6.0, 2.0, 0.0], "row i sees row i's offset"
    # B == 1 through a one-row batch takes the single-row path directly
    calls.clear()
    _qsa_batched_forward(attn, mx.zeros((1, 2, 16)), BatchQSACache(rows[:1]))
    assert calls == [((1, 2, 16), rows[0])]
    with pytest.raises(ValueError):
        _qsa_batched_forward(attn, mx.zeros((2, 2, 16)), BatchQSACache(rows[:1]))


def test_tail_by_lengths_stops_at_each_rows_real_length():
    prev, S, keep = 3, 4, 3
    seq = mx.arange(2 * (prev + S)).reshape(2, prev + S)  # row0: 0..6, row1: 7..13
    plain = _tail_by_lengths(seq, keep, SimpleNamespace(lengths=None), S)
    assert plain.tolist() == [[4, 5, 6], [11, 12, 13]]
    ragged = _tail_by_lengths(seq, keep, SimpleNamespace(lengths=mx.array([4, 1])), S)
    assert ragged.tolist() == [[4, 5, 6], [8, 9, 10]], "row 1 has one real token: its tail ends at prev+1"
    seq3 = mx.broadcast_to(seq[..., None], (2, prev + S, 2))
    ragged3 = _tail_by_lengths(seq3, keep, SimpleNamespace(lengths=mx.array([4, 1])), S)
    assert ragged3[..., 0].tolist() == [[4, 5, 6], [8, 9, 10]]


def test_startup_probe_now_admits_the_flash_next_cache_family():
    from mlx_lm.models.cache import ArraysCache

    runtime = SimpleNamespace(
        model=SimpleNamespace(make_cache=lambda: [QSACache(4), ArraysCache(size=4), ArraysCache(size=2)])
    )
    assert srv._ar_batch_unavailable_reason(runtime) is None
    assert srv._BatchedARGenerationService._cache_supports_batch_history_merge([QSACache(4), ArraysCache(size=2)])


def test_tiny_layer_batched_ragged_prefill_matches_solo_bit_for_bit():
    """A random tiny QSA layer run two ways: each sequence alone with its own
    cache, and both together through the batched lane's exact sequence
    (merge fresh rows -> prepare with right padding -> padded forward ->
    finalize -> a [B, 1] decode step). Row 0 is long enough to engage the
    sparse selection (T // ratio > block_topk); row 1 stays on the dense
    shortcut. Outputs, KV, indexer streams and the decode step must match
    exactly: the batched lane is the single-row code per row, and the trim
    must leave the padded row's cache identical to the solo one."""
    from mtplx.models.qwen4_exp import Attention, TextArgs

    args = TextArgs(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16, indexer_budget=16,
        indexer_compress_ratio=4, rms_norm_eps=1e-6, partial_rotary_factor=0.25, rope_theta=1e4,
    )
    mx.random.seed(0)
    attn = Attention(args)
    mx.eval(attn.parameters())
    lens = [22, 13]
    xs = [mx.random.normal((1, n, 64)) for n in lens]
    x_dec = mx.random.normal((2, 1, 64))

    solo = []
    for i, x in enumerate(xs):
        c = QSACache(4)
        y = attn(x, c)
        d = attn(x_dec[i : i + 1], c)
        mx.eval(y, d)
        solo.append((y, c, d))

    S = max(lens)
    xb = mx.concatenate([mx.concatenate([xs[i], mx.zeros((1, S - lens[i], 64))], axis=1) for i in range(2)], axis=0)
    bc = QSACache.merge([QSACache(4), QSACache(4)])
    bc.prepare(lengths=lens, right_padding=[S - n for n in lens])
    yb = attn(xb, bc)
    mx.eval(yb)
    bc.finalize()
    yd = attn(x_dec, bc)
    mx.eval(yd, bc.state)

    def same(a, b):
        return bool(mx.array_equal(a.astype(mx.float32), b.astype(mx.float32)))

    for i, (y, c, d) in enumerate(solo):
        n = lens[i]
        row = bc.rows[i]
        assert same(yb[i : i + 1, :n], y), f"row {i} prefill output"
        assert row.offset == c.offset == n + 1 and row.pooled_len == c.pooled_len
        assert same(row.kv.keys[..., :n, :], c.kv.keys[..., :n, :]) and same(row.kv.values[..., :n, :], c.kv.values[..., :n, :])
        assert same(row.raw_keys[:, :n], c.raw_keys[:, :n])
        if row.pooled_len:
            assert same(row.pooled[:, : row.pooled_len], c.pooled[:, : c.pooled_len])
        assert same(yd[i : i + 1], d), f"row {i} decode step"
    assert bc.rows[0].pooled_len > bc.rows[0].ratio and 22 // 4 > attn.indexer.block_topk, "row 0 exercised the sparse path"
