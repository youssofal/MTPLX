"""A QSA layer whose pooled block count ran past its keys never fails a turn (#592, #605).

On 2.12.2 a Flash-Next turn could end in HTTP 500 before its first token:
``ValueError: QSA pooled hold 23273 rows; resizing to 10240 would cut them``.
The one-copy store resizes a conversation's QSA buffers to the compiled
verifier's rows before the bank adopts them, and it priced and resized the
pooled index keys against ``pooled_len``, the layer's count of pooled blocks.
That count had run past the blocks the layer's keys cover (a restore from the
SSD tier below a longer entry's end handed over that entry's count, #635), so
the resize read rows of another conversation as rows it must keep, and every
retry failed the same way. Only blocks the keys cover are the layer's own:
``kv.offset // ratio``, the bound every other reader already applies (the
state getter, ``trim``, the verifier's demotion).

The end-to-end case (a real restore, then the resize) is in
``test_qsa_ssd_prefix_restore.py``. This file pins the bound where the count
is read: the resize and its bill, and the indexer's block extension.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx import one_copy
from mtplx.models.qwen4_exp import Attention, QSACache, TextArgs


def _reported_layer() -> QSACache:
    """The 2026-10-23 report: 27,957 tokens held, 19,843 pooled blocks counted."""

    cache = QSACache(4)
    cache.kv.keys = mx.zeros((1, 1, 32768, 2))
    cache.kv.values = mx.zeros((1, 1, 32768, 2))
    cache.kv.offset = 27957
    cache.raw_keys = mx.zeros((1, 32768, 8))
    cache.pooled = mx.zeros((1, 20000, 8))
    cache.pooled_len = 19843
    return cache


def _numbered_layer(*, rows: int, offset: int, pooled_rows: int, pooled_len: int) -> QSACache:
    """A layer whose every row is distinct, so a cut or a shifted row shows."""

    cache = QSACache(4)
    cache.kv.keys = mx.arange(rows * 2, dtype=mx.float32).reshape(1, 1, rows, 2)
    cache.kv.values = -cache.kv.keys
    cache.kv.offset = offset
    cache.raw_keys = mx.arange(rows * 8, dtype=mx.float32).reshape(1, rows, 8)
    cache.pooled = mx.arange(pooled_rows * 8, dtype=mx.float32).reshape(1, pooled_rows, 8) + 0.5
    cache.pooled_len = pooled_len
    mx.eval(cache.kv.keys, cache.kv.values, cache.raw_keys, cache.pooled)
    return cache


def test_the_reported_count_is_priced_and_resized_without_an_error():
    cache = _reported_layer()
    assert one_copy.resize_bill([cache], 32768) is not None
    assert one_copy.layers_to_resize([cache], 32768) == 1
    assert one_copy.resize_qsa_buffers([cache], 32768) == 0
    # The pooled keys take the bank's 8,192 blocks; the layer keeps its own
    # 6,989 and counts only those.
    assert int(cache.pooled.shape[1]) == 32768 // 4
    assert cache.pooled_len == 27957 // 4


def test_the_counts_from_the_other_reports_never_raise():
    # 2026-10-28: 25,203 blocks counted on a 24,576-row layer the bank keeps
    # at its rows (6,144 blocks).
    for blocks in (25203, 37749, 49334):
        layer = _numbered_layer(rows=24576, offset=24576, pooled_rows=blocks, pooled_len=blocks)
        assert one_copy.qsa_buffers_short([layer], 24576) is False
        assert one_copy.resize_bill([layer], 24576) is not None
        assert one_copy.layers_to_resize([layer], 24576) == 1


def test_a_resize_keeps_every_row_the_layer_holds_and_drops_only_the_rest():
    layer = _numbered_layer(rows=96, offset=70, pooled_rows=60, pooled_len=40)
    before = [mx.array(leaf) for leaf in layer.state]
    assert one_copy.resize_qsa_buffers([layer], 80) == 0
    assert int(layer.kv.keys.shape[2]) == int(layer.raw_keys.shape[1]) == 80
    assert int(layer.pooled.shape[1]) == 20
    assert layer.pooled_len == 70 // 4
    for leaf, old in zip(layer.state, before):
        assert mx.array_equal(leaf, old).item()


def test_a_consistent_layer_resizes_exactly_as_before():
    layer = _numbered_layer(rows=32, offset=17, pooled_rows=8, pooled_len=4)
    before = [mx.array(leaf) for leaf in layer.state]
    assert one_copy.resize_qsa_buffers([layer], 64) == 0
    assert layer.pooled_len == 4
    for leaf, old in zip(layer.state, before):
        assert mx.array_equal(leaf, old).item()


def test_rows_the_keys_hold_are_still_never_cut():
    layer = _numbered_layer(rows=64, offset=48, pooled_rows=16, pooled_len=12)
    with pytest.raises(ValueError, match="would cut them"):
        one_copy.resize_qsa_buffers([layer], 32)
    assert int(layer.kv.keys.shape[2]) == 64 and layer.kv.offset == 48


# -- the indexer's block extension ---------------------------------------------------


def _args() -> TextArgs:
    return TextArgs(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=16,
        indexer_compress_ratio=4,
    )


@pytest.fixture()
def attn():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(0)
    layer = Attention(_args())
    mx.eval(layer.parameters())
    yield layer
    mx.set_default_device(previous)


def _hidden(tokens: int, seed: int) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((1, tokens, 64)).astype(mx.float32)


class _AttentionWriteFailed(Exception):
    pass


def test_blocks_pooled_by_a_forward_that_failed_are_pooled_again(attn):
    """A forward that raised after the indexer pooled its blocks (memory refused
    the attention write, say) leaves the count past the keys. The next forward
    writes other tokens at those rows and must pool them, not reuse the
    failed forward's blocks."""

    prefix, failed, retried = _hidden(64, seed=1), _hidden(16, seed=2), _hidden(16, seed=3)
    cache = QSACache(4)
    attn(prefix, cache)

    def refuse(keys, values):
        raise _AttentionWriteFailed

    cache.kv.update_and_fetch = refuse
    with pytest.raises(_AttentionWriteFailed):
        attn(failed, cache)
    del cache.kv.update_and_fetch
    assert cache.offset == 64
    assert cache.pooled_len == (64 + 16) // 4  # pooled before attention refused

    out = attn(retried, cache)
    fresh = QSACache(4)
    attn(prefix, fresh)
    golden = attn(retried, fresh)
    assert mx.array_equal(out, golden).item()
    assert mx.array_equal(cache.state[3], fresh.state[3]).item()


def test_an_extension_with_no_new_block_drops_a_count_past_the_keys(attn):
    cache = QSACache(4)
    attn(_hidden(64, seed=4), cache)
    cache.pooled_len = 30  # past the 16 blocks the keys cover
    attn(_hidden(2, seed=5), cache)  # completes no block
    assert cache.pooled_len == 16


# The same failure on a decode step, through each indexer route the stock cache
# can take: the compiled indexer reads the count before its graph runs.
from test_qsa_index_block_writes_evaluated import (  # noqa: E402
    _NEEDS_GPU,
    _arm_indexer_route,
    _layer,
    restore_device,  # noqa: F401  (fixture)
)
from test_qsa_index_block_writes_evaluated import _cache as _ratio_two_cache  # noqa: E402
from test_qsa_index_block_writes_evaluated import _hidden as _rows  # noqa: E402


@pytest.mark.parametrize(
    "device,dtype,route",
    [
        pytest.param(mx.cpu, mx.float32, "eager", id="cpu-fp32-eager"),
        pytest.param(mx.gpu, mx.bfloat16, "eager", id="gpu-bf16-eager", marks=_NEEDS_GPU),
        pytest.param(mx.gpu, mx.bfloat16, "compiled", id="gpu-bf16-compiled", marks=_NEEDS_GPU),
    ],
)
def test_a_decode_step_that_failed_after_pooling_is_pooled_again(
    restore_device, monkeypatch, device, dtype, route,
):
    calls = _arm_indexer_route(monkeypatch, route)
    layer = _layer(device, dtype)
    # Ratio 2, budget 8: 13 rows hold 6 blocks, and the next row completes a
    # seventh while selection is engaged (more than 4 blocks).
    prefix = _rows(13, seed=1, dtype=dtype)
    failed, retried = _rows(1, seed=2, dtype=dtype), _rows(1, seed=3, dtype=dtype)
    cache = _ratio_two_cache()
    mx.eval(layer(prefix, cache))

    def refuse(keys, values):
        raise _AttentionWriteFailed

    cache.kv.update_and_fetch = refuse
    with pytest.raises(_AttentionWriteFailed):
        layer(failed, cache)
    del cache.kv.update_and_fetch
    assert cache.offset == 13 and cache.pooled_len == 7

    served = calls[route]
    out = layer(retried, cache)
    # The retried step itself ran on the armed route.
    assert calls[route] == served + 1, calls
    fresh = _ratio_two_cache()
    layer(prefix, fresh)
    golden = layer(retried, fresh)
    assert mx.array_equal(out, golden).item()
    assert mx.array_equal(cache.state[3], fresh.state[3]).item()
