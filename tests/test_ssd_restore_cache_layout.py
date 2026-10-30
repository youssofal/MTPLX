"""An SSD restore builds the request's cache layout, the same one a RAM restore builds.

Before, the SSD restore called ``runtime.make_cache()`` and ignored the request's
``cache_factory``: for a long context that default is the paged KV cache, with about 5x slower
prefill and slower decode than the dense cache the request asked for (Qwen3.8 27B, 2026-10-08:
first token after an SSD restore at 50K and 80K 16 to 48 s instead of 5 to 8 s).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache

from mtplx.cache_bank.cold_tier import SessionBankColdTier
from mtplx.cache_state import snapshot_cache
from mtplx.session_bank import SessionBank

HEADS, DIM, ROWS = 2, 16, 700


class _DefaultLayoutKV(KVCache):
    """Stands in for the runtime's default layout (the paged cache in production)."""


class _RequestLayoutKV(KVCache):
    """Stands in for the layout the request asks for (the dense cache in production)."""


class _Runtime:
    model_path = Path("models/example")

    def __init__(self, *, mtp: bool = False) -> None:
        self.mtp_enabled = mtp
        self.made = 0
        self.made_mtp = 0

    def make_cache(self):
        self.made += 1
        return [_DefaultLayoutKV() for _ in range(2)]

    def make_mtp_cache(self):
        self.made_mtp += 1
        return [_DefaultLayoutKV()]


def _rand(seed: int, n: int) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((1, HEADS, n, DIM)).astype(np.float32)).astype(mx.bfloat16)


def _filled(layers: int, seed: int) -> list[KVCache]:
    cache = [KVCache() for _ in range(layers)]
    for i, c in enumerate(cache):
        c.update_and_fetch(_rand(seed + i, ROWS), _rand(seed + 50 + i, ROWS))
    return cache


def _tier(path: Path) -> SessionBankColdTier:
    return SessionBankColdTier(base_dir=path, mode="on", min_prefix_tokens=2, block_size=256)


def _bank_on_disk(tmp_path: Path, runtime: _Runtime, *, mtp: bool = False):
    """A conversation written to the SSD tier, then a fresh bank (empty RAM) on the same store."""
    tier = _tier(tmp_path / "tier")
    tokens = list(range(ROWS))
    cache = _filled(2, 1)
    kwargs = {}
    if mtp:
        kwargs = {
            "mtp_history_snapshot": snapshot_cache(_filled(1, 9)),
            "snapshot_epoch": ROWS,
            "mtp_snapshot_epoch": ROWS,
        }
    SessionBank(cold_tier=tier).put(
        runtime=runtime,
        token_ids=tokens,
        cache=cache,
        logits=mx.zeros((1, 8)),
        hidden=mx.ones((1, 4), dtype=mx.bfloat16) if mtp else None,
        session_id="s",
        **kwargs,
    )
    assert tier.flush(timeout_s=30.0)
    return SessionBank(cold_tier=tier), tokens, cache


def test_ssd_restore_builds_the_requests_cache_layout(tmp_path) -> None:
    runtime = _Runtime()
    bank, tokens, written = _bank_on_disk(tmp_path, runtime)
    made_before = runtime.made
    restored = bank.restore(
        runtime,
        tokens + [ROWS + 1, ROWS + 2],
        session_id="s",
        cache_factory=lambda: [_RequestLayoutKV() for _ in range(2)],
    )
    assert restored is not None and restored.restore_mode == "ssd_clone"
    assert bank.last_restore_source == "ssd"
    assert all(type(c) is _RequestLayoutKV for c in restored.cache)
    assert runtime.made == made_before  # the default layout is not built at all
    for got, want in zip(restored.cache, written):
        assert got.offset == ROWS
        gk, gv = got.state
        wk, wv = want.state
        assert mx.array_equal(gk, wk).item() and mx.array_equal(gv, wv).item()


def test_ssd_restore_without_a_factory_keeps_the_default_layout(tmp_path) -> None:
    runtime = _Runtime()
    bank, tokens, _ = _bank_on_disk(tmp_path, runtime)
    restored = bank.restore(runtime, tokens + [ROWS + 1], session_id="s")
    assert restored is not None and restored.restore_mode == "ssd_clone"
    assert all(type(c) is _DefaultLayoutKV for c in restored.cache)
    assert restored.cache[0].offset == ROWS


def test_ssd_restore_builds_the_mtp_history_with_the_requests_factory(tmp_path) -> None:
    runtime = _Runtime(mtp=True)
    bank, tokens, _ = _bank_on_disk(tmp_path, runtime, mtp=True)
    made_mtp_before = runtime.made_mtp
    restored = bank.restore(
        runtime,
        tokens + [ROWS + 1],
        session_id="s",
        cache_factory=lambda: [_RequestLayoutKV() for _ in range(2)],
        mtp_cache_factory=lambda: [_RequestLayoutKV()],
    )
    assert restored is not None and restored.restore_mode == "ssd_clone"
    assert restored.mtp_history_cache is not None
    assert type(restored.mtp_history_cache[0]) is _RequestLayoutKV
    assert restored.mtp_history_cache[0].offset == ROWS
    assert runtime.made_mtp == made_mtp_before
