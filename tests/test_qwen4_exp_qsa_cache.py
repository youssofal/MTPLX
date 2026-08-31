"""QSACache must be a full citizen of the cache contract.

The QSA indexer keeps its own raw-key stream (and derived pooled block keys)
next to the attention KV. The serve loop rolls caches back after every
speculative verify round (``rollback_after_verify``: trim for trimmable
entries, snapshot-restore for the rest) and resumes banked sessions through
``state``. A raw-key stream that only ever appends desyncs from the KV on the
first rollback; once the context crosses the indexer's engage threshold the
selection mask is built from the raw-stream length while attention keys come
from the KV — the ``broadcast_shapes (1,1,4,3719) vs (1,24,4,3715)`` crash
OpenCode hit live at 3.7k ctx (2026-08-27). Below the threshold the same
desync corrupts pooled blocks silently instead of crashing.

All runs are CPU (M-series GPU fp32 matmul is reduced-precision; CPU is the
parity surface).
"""

import mlx.core as mx
import pytest

from mtplx.cache_state import (
    rollback_after_verify,
    snapshot_untrimmable_cache,
)
from mtplx.models.qwen4_exp import Attention, QSACache, TextArgs


def _tiny_args() -> TextArgs:
    return TextArgs(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=2,
    )


@pytest.fixture()
def attn():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(0)
    layer = Attention(_tiny_args())
    mx.eval(layer.parameters())
    yield layer
    mx.set_default_device(prev)


def _hidden(tokens: int, seed: int) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((1, tokens, 64)).astype(mx.float32)


PREFILL = 12  # engage threshold with budget=8/ratio=2 is >8 visible tokens
STEP = 4  # a depth-3 verify round: 1 committed + 3 drafts


def test_rollback_then_forward_matches_fresh_run(attn):
    """A rejected verify round must leave the QSA layer exactly where a run
    that never saw the rejected tokens would be."""
    x_pre = _hidden(PREFILL, seed=1)
    x_rejected = _hidden(STEP, seed=2)
    x_next = _hidden(STEP, seed=3)

    cache = [QSACache()]
    attn(x_pre, cache[0])
    snap = snapshot_untrimmable_cache(cache)
    attn(x_rejected, cache[0])
    rollback_after_verify(cache, snap, verified_tokens=STEP)
    assert cache[0].offset == PREFILL
    out = attn(x_next, cache[0])

    fresh = QSACache()
    attn(x_pre, fresh)
    golden = attn(x_next, fresh)

    assert out.shape == golden.shape
    assert mx.allclose(out, golden, atol=0, rtol=0).item()


def test_state_roundtrip_resumes_identically(attn):
    """Bank restore: ``state`` must carry everything the layer needs — a
    resumed session past the engage threshold selects the same blocks and
    produces the same output as the uninterrupted run."""
    x_pre = _hidden(PREFILL, seed=4)
    x_next = _hidden(STEP, seed=5)

    live = QSACache()
    attn(x_pre, live)
    golden = attn(x_next, live)

    donor = QSACache()
    attn(x_pre, donor)
    resumed = QSACache()
    resumed.state = donor.state
    assert resumed.offset == PREFILL
    out = attn(x_next, resumed)

    assert mx.allclose(out, golden, atol=0, rtol=0).item()


def test_trim_contract(attn):
    """QSACache is trimmable: trim rolls the layer back token-exactly,
    including through a pooled-block boundary."""
    cache = QSACache()
    assert cache.is_trimmable()

    x_pre = _hidden(PREFILL, seed=6)
    x_tail = _hidden(3, seed=7)  # odd length: trims back through a block edge
    x_next = _hidden(STEP, seed=8)

    attn(x_pre, cache)
    attn(x_tail, cache)
    assert cache.trim(3) == 3
    assert cache.offset == PREFILL
    out = attn(x_next, cache)

    fresh = QSACache()
    attn(x_pre, fresh)
    golden = attn(x_next, fresh)

    assert mx.allclose(out, golden, atol=0, rtol=0).item()


def test_rollback_below_engage_threshold_still_exact(attn):
    """The desync is silent below the engage threshold (dense mask hides it);
    the pooled stream must still be positionally correct once the session
    grows past it."""
    x_pre = _hidden(4, seed=9)
    x_rejected = _hidden(STEP, seed=10)
    # two accepted rounds carry the session across the threshold
    x_a = _hidden(STEP, seed=11)
    x_b = _hidden(STEP, seed=12)
    x_c = _hidden(STEP, seed=13)

    cache = [QSACache()]
    attn(x_pre, cache[0])
    snap = snapshot_untrimmable_cache(cache)
    attn(x_rejected, cache[0])
    rollback_after_verify(cache, snap, verified_tokens=STEP)
    for chunk in (x_a, x_b, x_c):
        out = attn(chunk, cache[0])

    fresh = QSACache()
    attn(x_pre, fresh)
    for chunk in (x_a, x_b, x_c):
        golden = attn(chunk, fresh)

    assert mx.allclose(out, golden, atol=0, rtol=0).item()


def test_qsa_cache_quantized_pooled_mirror():
    """Verify quantized pooled key mirror (q8 and q4) saves memory and maintains
    valid transposed view."""
    # Test q8
    cache8 = QSACache(kv_bits=8)
    assert cache8.pooled_bits == 8
    blocks = mx.random.normal((1, 64, 16)).astype(mx.float32)
    cache8.write_pooled(blocks, 0, 64)
    assert cache8.pooled_quant_t is not None
    assert cache8.pooled_f32_t is None
    view8 = cache8.pooled_f32_view(64)
    assert view8.shape == (1, 1, 16, 64)
    # Check decompression closeness
    assert mx.allclose(view8[0, 0], mx.swapaxes(blocks[0], 0, 1), atol=1e-1).item()

    # Test state roundtrip
    donor8 = QSACache(kv_bits=8)
    donor8.write_pooled(blocks, 0, 64)
    resumed8 = QSACache(kv_bits=8)
    resumed8.state = donor8.state
    assert resumed8.pooled_quant_t is not None
    view_resumed = resumed8.pooled_f32_view(64)
    assert view_resumed.shape == (1, 1, 16, 64)

    # Test q4
    cache4 = QSACache(kv_bits=4)
    assert cache4.pooled_bits == 4
    cache4.write_pooled(blocks, 0, 64)
    assert cache4.pooled_quant_t is not None
    view4 = cache4.pooled_f32_view(64)
    assert view4.shape == (1, 1, 16, 64)

    # Test arbitrary unaligned block counts (e.g. 23 blocks)
    cache_unaligned = QSACache(kv_bits=8)
    blocks_unaligned = mx.random.normal((1, 23, 16)).astype(mx.float32)
    cache_unaligned.write_pooled(blocks_unaligned, 0, 23)
    assert cache_unaligned.pooled_quant_t is not None
    view_unaligned = cache_unaligned.pooled_f32_view(23)
    assert view_unaligned.shape == (1, 1, 16, 23)
    assert mx.allclose(view_unaligned[0, 0], mx.swapaxes(blocks_unaligned[0], 0, 1), atol=1e-1).item()

    # Test incremental append over group boundary (e.g. 70 blocks then add 80 more -> 150 blocks)
    cache_inc = QSACache(kv_bits=8)
    all_blocks = mx.random.normal((1, 150, 16)).astype(mx.float32)
    cache_inc.write_pooled(all_blocks[:, :70, :], 0, 70)
    cache_inc.write_pooled(all_blocks[:, 70:150, :], 70, 150)
    view_inc = cache_inc.pooled_f32_view(150)
    assert view_inc.shape == (1, 1, 16, 150)
    assert mx.allclose(view_inc[0, 0], mx.swapaxes(all_blocks[0], 0, 1), atol=1e-1).item()


def test_select_backend_context_window_clamps_to_model_max():
    from mtplx.backends.descriptors import QWEN3_NEXT_DESCRIPTOR
    from mtplx.server.openai import _select_backend_context_window

    # Explicit request <= model_max succeeds
    assert _select_backend_context_window(QWEN3_NEXT_DESCRIPTOR, model_max=262144, requested=131072, machine_fit=65536) == 131072
    # Explicit request > model_max is clamped to model_max
    assert _select_backend_context_window(QWEN3_NEXT_DESCRIPTOR, model_max=262144, requested=1048576, machine_fit=65536) == 262144
    # Default without explicit request honors machine_fit
    assert _select_backend_context_window(QWEN3_NEXT_DESCRIPTOR, model_max=262144, requested=None, machine_fit=65536) == 65536


def test_adaptive_mtp_history_window_throttling():
    from mtplx.generation import _mtp_history_last_window_tokens

    # Standard scaling
    assert _mtp_history_last_window_tokens(1000) == 8192
    assert _mtp_history_last_window_tokens(32768) == 16384
    assert _mtp_history_last_window_tokens(65536) == 32768
    assert _mtp_history_last_window_tokens(262144) == 32768

    # Adaptive throttling at extreme depth (>262k) caps to 16,384 tokens
    assert _mtp_history_last_window_tokens(262145) == 16384
    assert _mtp_history_last_window_tokens(524288) == 16384
    assert _mtp_history_last_window_tokens(1048576) == 16384


def test_qsa_cache_empty_state_restore_clears_kv():
    cache = QSACache(kv_bits=8)
    cache.kv.offset = 128
    cache.raw_keys = mx.zeros((1, 128, 16), mx.float32)
    cache.pooled = mx.zeros((1, 32, 16), mx.float32)
    # Restore empty snapshot
    cache.state = (None, None, None, None)
    assert cache.kv.offset == 0
    assert cache.raw_keys is None
    assert cache.pooled is None
    assert cache.pooled_len == 0


def test_entry_matches_restore_lookup_history_window():
    from unittest.mock import MagicMock
    from mtplx.generation import _entry_matches_restore_lookup

    rt = MagicMock()
    rt.model_path = "test-model"

    # Stored last_window entry with 8K history (from 16K prefix)
    entry_lw = MagicMock()
    entry_lw.model_path = "test-model"
    entry_lw.hidden_variant = "post_norm"
    entry_lw.template_hash = "h1"
    entry_lw.mtp_history_policy = "last_window"
    entry_lw.draft_head_identity = "d1"
    entry_lw.policy_fingerprint = "fp1"
    entry_lw.prefix_len = 16384
    entry_lw.mtp_snapshot_epoch = 16384
    entry_lw.snapshot_epoch = 16384

    # 32K prompt requires 16K window -> 8K stored history is too narrow
    assert not _entry_matches_restore_lookup(
        entry_lw,
        rt,
        hidden_variant="post_norm",
        template_hash="h1",
        mtp_history_policy="last_window",
        draft_head_identity="d1",
        policy_fingerprint="fp1",
        prompt_tokens=32768,
    )

    # Stored committed entry with 16K history
    entry_com = MagicMock()
    entry_com.model_path = "test-model"
    entry_com.hidden_variant = "post_norm"
    entry_com.template_hash = "h1"
    entry_com.mtp_history_policy = "committed"
    entry_com.draft_head_identity = "d1"
    entry_com.policy_fingerprint = "fp1"
    entry_com.prefix_len = 16384
    entry_com.mtp_snapshot_epoch = 16384
    entry_com.snapshot_epoch = 16384

    # 32K prompt requires 16K window -> 16K stored history is sufficient
    assert _entry_matches_restore_lookup(
        entry_com,
        rt,
        hidden_variant="post_norm",
        template_hash="h1",
        mtp_history_policy="last_window",
        draft_head_identity="d1",
        policy_fingerprint="fp1",
        prompt_tokens=32768,
    )


def test_adaptive_mtp_history_deep_cap_clamped(monkeypatch):
    from mtplx.generation import _mtp_history_last_window_tokens

    monkeypatch.setenv("MTPLX_MTP_HISTORY_DEEP_CAP", "0")
    # Should clamp to at least 1, not return 0 or negative
    assert _mtp_history_last_window_tokens(300000) >= 1

    monkeypatch.setenv("MTPLX_MTP_HISTORY_DEEP_CAP", "-100")
    assert _mtp_history_last_window_tokens(300000) >= 1


def test_qsa_cache_state_restore_type_mismatch_raises():
    import pytest

    quant_cache = QSACache(kv_bits=8)
    dense_keys = mx.zeros((1, 1, 10, 64), mx.float32)
    dense_values = mx.zeros((1, 1, 10, 64), mx.float32)

    with pytest.raises(ValueError, match="Cannot restore dense KV snapshot into QuantizedKVCache"):
        quant_cache.state = (dense_keys, dense_values, None, None)

    dense_cache = QSACache(kv_bits=None)
    quant_tuple = (
        mx.zeros((1, 1, 10, 16), mx.uint32),
        mx.zeros((1, 1, 10, 1), mx.float32),
        mx.zeros((1, 1, 10, 1), mx.float32),
    )
    with pytest.raises(ValueError, match="Cannot restore quantized KV snapshot into dense KVCache"):
        dense_cache.state = (quant_tuple, quant_tuple, None, None)




