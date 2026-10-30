"""Regression test: partial-prefix SSD restore must skip tuple meta states.

A partial-prefix SSD restore (decode_payload_prefix) must hand restore_cache
``None`` for every layer's meta state, including layers whose meta state is a
tuple (paged KV cache: block_size, num_blocks, offset).  Before the fix such a
layer became ``(None, None, None)``; restore_cache only skips ``None``, so the
paged cache's meta_state setter crashed in ``int(value[2])`` (TypeError) on
every partial-prefix restore.

Runs under pytest and standalone:  python tests/test_prefix_decode_meta_skip.py
"""

from __future__ import annotations

import mlx.core as mx

from mtplx.cache_bank.codec import decode_payload_prefix, encode_payload
from mtplx.cache_state import (
    CacheSnapshot,
    TailOwnedKVCache,
    VllmMetalPagedKVCache,
    restore_cache,
    snapshot_cache,
)


class _PagedMetaEntry:
    """Stands in for VllmMetalPagedKVCache: its meta_state setter int()s [2]."""

    def __init__(self) -> None:
        self.offset = 0

    @property
    def meta_state(self):
        return (str(16), str(8), str(self.offset))

    @meta_state.setter
    def meta_state(self, value) -> None:
        self.offset = int(value[2])


def test_prefix_decode_meta_states_are_none_for_tuple_layers():
    snapshot = CacheSnapshot(
        states=(
            mx.zeros((1, 1, 1024, 2), dtype=mx.float16),
            mx.zeros((1, 1, 1024, 2), dtype=mx.float16),
        ),
        meta_states=(("16", "8117", "114960"), ("16", "8117", "114960")),
    )
    mtp_snapshot = CacheSnapshot(
        states=(mx.full((1, 1, 1024, 2), 3.0, dtype=mx.float16),),
        meta_states=(("16", "8117", "114960"),),
    )
    encoded = encode_payload(
        cache_snapshot=snapshot,
        logits=None,
        hidden=None,
        mtp_history_snapshot=mtp_snapshot,
        block_size=256,
    )

    decoded = decode_payload_prefix(
        encoded.spec,
        encoded.tensors.__getitem__,
        cache_prefix_len=256,
        mtp_history_prefix_len=255,
    )

    assert tuple(decoded.cache_snapshot.meta_states) == (None, None)
    assert tuple(decoded.mtp_history_snapshot.meta_states) == (None,)


def test_restore_cache_skips_prefix_decoded_meta_states():
    snapshot = CacheSnapshot(
        states=(mx.zeros((1, 1, 1024, 2), dtype=mx.float16),),
        meta_states=(("16", "8117", "114960"),),
    )
    encoded = encode_payload(
        cache_snapshot=snapshot,
        logits=None,
        hidden=None,
        mtp_history_snapshot=None,
        block_size=256,
    )
    decoded = decode_payload_prefix(
        encoded.spec, encoded.tensors.__getitem__, cache_prefix_len=256
    )
    entry = _PagedMetaEntry()
    entry.offset = 256

    # The state is not installed here (entry has no KV pages): only the meta
    # path is under test, so give restore_cache a state-less snapshot.
    meta_only = CacheSnapshot(
        states=(None,), meta_states=tuple(decoded.cache_snapshot.meta_states)
    )
    restore_cache([entry], meta_only)  # raised TypeError before the fix

    assert entry.offset == 256


def _filled(cache, rows: int, seed: int):
    keys = mx.random.normal((1, 2, rows, 8), key=mx.random.key(seed))
    values = mx.random.normal((1, 2, rows, 8), key=mx.random.key(seed + 1))
    cache.update_and_fetch(keys, values)
    mx.eval(keys, values)
    return keys, values


def test_prefix_restore_into_real_tuple_meta_caches_without_a_factory():
    # The production shape of #655: the restore builds the runtime's default
    # layout (no cache factory, so meta states are restored) and both of its
    # attention containers carry tuple meta states: the paged cache
    # (block_size, num_blocks, offset) and the tail-owned cache
    # (step, offset, mode). The committed MTP history is always restored with
    # its meta state, factory or not. Each must land at the prefix, holding
    # exactly the stored prefix rows.
    rows, prefix = 600, 256
    paged = VllmMetalPagedKVCache(block_size=16, num_blocks=64)
    tail = TailOwnedKVCache()
    mtp = TailOwnedKVCache()
    trunk_rows = [_filled(paged, rows, 1), _filled(tail, rows, 3)]
    mtp_rows = _filled(mtp, rows, 5)
    encoded = encode_payload(
        cache_snapshot=snapshot_cache([paged, tail]),
        logits=None,
        hidden=None,
        mtp_history_snapshot=snapshot_cache([mtp]),
        block_size=128,
    )
    decoded = decode_payload_prefix(
        encoded.spec,
        encoded.tensors.__getitem__,
        cache_prefix_len=prefix,
        mtp_history_prefix_len=prefix - 1,
    )

    restored = [VllmMetalPagedKVCache(block_size=16, num_blocks=64), TailOwnedKVCache()]
    restore_cache(restored, decoded.cache_snapshot)  # TypeError before the fix
    restored_mtp = [TailOwnedKVCache()]
    restore_cache(restored_mtp, decoded.mtp_history_snapshot)  # TypeError too

    for layer, (keys, values) in zip(restored, trunk_rows):
        assert layer.offset == prefix
        got_keys, got_values = layer.state
        assert mx.array_equal(got_keys, keys[..., :prefix, :]).item()
        assert mx.array_equal(got_values, values[..., :prefix, :]).item()
    assert restored_mtp[0].offset == prefix - 1
    got_keys, got_values = restored_mtp[0].state
    assert mx.array_equal(got_keys, mtp_rows[0][..., : prefix - 1, :]).item()
    assert mx.array_equal(got_values, mtp_rows[1][..., : prefix - 1, :]).item()


if __name__ == "__main__":
    test_prefix_decode_meta_states_are_none_for_tuple_layers()
    test_restore_cache_skips_prefix_decoded_meta_states()
    test_prefix_restore_into_real_tuple_meta_caches_without_a_factory()
    print("OK")
