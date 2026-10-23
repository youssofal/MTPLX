"""QSA cache rewinds and declined resizes must preserve the logical frontier."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

import mtplx.models.qwen4_exp as qwen
import mtplx.one_copy as oc
from mtplx import generation


def cache_at(offset, pooled_len, capacity=80000, ratio=4):
    cache = qwen.QSACache(ratio)
    cache.kv.keys = mx.arange(capacity * 2).reshape(1, 1, capacity, 2)
    cache.kv.values = -cache.kv.keys
    cache.kv.offset = offset
    cache.raw_keys = mx.arange(capacity * 8, dtype=mx.float32).reshape(1, capacity, 8)
    cache.pooled = mx.full((1, capacity // ratio, 8), -123.0)
    cache.pooled_len = pooled_len
    mx.eval(cache.kv.keys, cache.kv.values, cache.raw_keys, cache.pooled)
    return cache


class CacheRegressionTests(unittest.TestCase):
    def test_exact_reported_stale_length_resize_preserves_active_prefix(self):
        cache = cache_at(27957, 19843)
        expected = cache.state
        self.assertIsNotNone(oc.resize_bill([cache], 32768))
        self.assertEqual(oc.resize_qsa_buffers([cache], 32768), 0)
        self.assertEqual(cache.pooled_len, 27957 // 4)
        self.assertEqual(cache.pooled.shape[1], 8192)
        for actual, before in zip(cache.state, expected):
            self.assertTrue(mx.array_equal(actual, before).item())

    def test_normal_resize_does_not_change_logical_state(self):
        cache = cache_at(17, 4, capacity=32)
        expected = cache.state
        oc.resize_qsa_buffers([cache], 64)
        self.assertEqual(cache.pooled_len, 4)
        for actual, before in zip(cache.state, expected):
            self.assertTrue(mx.array_equal(actual, before).item())

    def test_already_matching_buffers_still_normalize_stale_count(self):
        cache = cache_at(17, 7, capacity=32)
        self.assertEqual(oc.resize_qsa_buffers([cache], 32), 0)
        self.assertEqual(cache.pooled_len, 4)

    def test_other_compression_ratio(self):
        cache = cache_at(17, 4, capacity=64, ratio=8)
        oc.resize_qsa_buffers([cache], 32)
        self.assertEqual(cache.pooled_len, 2)
        self.assertEqual(cache.pooled.shape[1], 4)

    def test_real_live_rows_cannot_be_truncated(self):
        cache = cache_at(48, 12, capacity=64)
        before = cache.state
        self.assertIsNone(oc.resize_bill([cache], 32))
        with self.assertRaises(oc.QSAResizeWouldCut):
            oc.resize_qsa_buffers([cache], 32)
        self.assertEqual(cache.kv.offset, 48)
        for actual, old in zip(cache.state, before):
            self.assertTrue(mx.array_equal(actual, old).item())

    def test_promotion_estimate_falls_back_to_copy(self):
        cache = cache_at(48, 12, capacity=64)
        with (
            patch.object(
                generation, "_qwen4_fixed_m4_adoption_rows", return_value=None
            ),
            patch.object(qwen, "_qsa_gather_enabled", return_value=True),
            patch.object(qwen, "_qsa_gather_min_context", return_value=0),
        ):
            result = generation._qwen4_fixed_m4_promotion_bill(
                None, 28, SimpleNamespace(), 32, [cache], 64, 100
            )
        self.assertEqual(result, ("copied", 3200, 32))

    def test_execution_guard_keeps_eager_path_on_unsafe_resize(self):
        cache = cache_at(48, 12, capacity=64)
        receipt = {}
        with patch.object(generation, "_announce_qwen4_fixed_m4_skip"):
            result = generation._qwen4_fixed_m4_resize_held(
                None,
                [cache],
                SimpleNamespace(resize_rows=32),
                receipt,
                session_bank=None,
                protect_ids=[],
            )
        self.assertFalse(result)
        self.assertEqual(receipt["reason"], "cache_resize_guard")
        self.assertEqual(cache.kv.offset, 48)

    def indexer(self):
        return qwen.QSAIndexer(
            qwen.TextArgs(
                hidden_size=8,
                head_dim=8,
                num_hidden_layers=1,
                indexer_n_heads=1,
                indexer_kv_heads=1,
                indexer_head_dim=8,
                indexer_budget=8,
                partial_rotary_factor=1.0,
            )
        )

    def test_rewound_extension_recomputes_new_blocks_and_mirror(self):
        cache = cache_at(8, 7, capacity=32)
        indexer = self.indexer()
        # Preserve two valid blocks, overwrite stale blocks after offset 8.
        old_prefix = cache.pooled[:, :2, :]
        cache.pooled_f32_view(7)
        expected = indexer._pool_keys_eager(cache.raw_keys[:, 8:16, :], 2, 4)
        with patch.object(
            qwen.QSAIndexer, "_prepare_kernel_supported", return_value=False
        ):
            result = indexer._extend_pooled(cache, 16)
        self.assertTrue(mx.array_equal(result[:, :2, :], old_prefix).item())
        self.assertTrue(mx.array_equal(result[:, 2:4, :], expected).item())
        mirror = mx.swapaxes(cache.pooled_f32_view(4)[:, 0], 1, 2)
        self.assertTrue(mx.array_equal(mirror, result.astype(mx.float32)).item())
        self.assertEqual(cache.pooled_len, 4)

    def test_rewound_extension_without_complete_block_clears_length(self):
        cache = cache_at(0, 7, capacity=32)
        result = self.indexer()._extend_pooled(cache, 3)
        self.assertIsNone(result)
        self.assertEqual(cache.pooled_len, 0)

    def test_regular_extension_matches_full_recomputation(self):
        cache = cache_at(8, 2, capacity=32)
        indexer = self.indexer()
        cache.write_pooled(
            indexer._pool_keys_eager(cache.raw_keys[:, :8, :], 0, 2), 0, 2
        )
        expected = indexer._pool_keys_eager(cache.raw_keys[:, :16, :], 0, 4)
        with patch.object(
            qwen.QSAIndexer, "_prepare_kernel_supported", return_value=False
        ):
            result = indexer._extend_pooled(cache, 16)
        self.assertTrue(mx.array_equal(result, expected).item())


if __name__ == "__main__":
    unittest.main(verbosity=2)
