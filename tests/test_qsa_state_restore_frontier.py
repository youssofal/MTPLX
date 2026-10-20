"""A QSACache restored from a prefix-decoded snapshot must not trust the donor's pooled table length."""
import mlx.core as mx
from mtplx.models.qwen4_exp import QSACache


def _leaves(kv_rows, donor_rows, ratio=4, dim=8):
    keys = mx.zeros((1, 2, kv_rows, 16), dtype=mx.bfloat16)
    values = mx.zeros((1, 2, kv_rows, 16), dtype=mx.bfloat16)
    raw = mx.zeros((1, donor_rows, dim), dtype=mx.bfloat16)
    pooled = mx.zeros((1, donor_rows // ratio, dim), dtype=mx.bfloat16)
    return keys, values, raw, pooled


def test_prefix_restore_clamps_pooled_len_to_the_restored_kv():
    cache = QSACache(compress_ratio=4)
    cache.state = _leaves(kv_rows=2048, donor_rows=58073)
    assert cache.offset == 2048
    assert cache.pooled_len == 2048 // 4


def test_full_state_round_trip_keeps_its_pooled_blocks():
    cache = QSACache(compress_ratio=4)
    cache.state = _leaves(kv_rows=2048, donor_rows=2048)
    assert cache.pooled_len == 512
    again = QSACache(compress_ratio=4)
    again.state = cache.state
    assert again.pooled_len == 512


def test_snapshot_without_pooled_has_no_valid_blocks():
    cache = QSACache(compress_ratio=4)
    keys, values, raw, _ = _leaves(kv_rows=256, donor_rows=256)
    cache.state = (keys, values, raw, None)
    assert cache.pooled_len == 0
