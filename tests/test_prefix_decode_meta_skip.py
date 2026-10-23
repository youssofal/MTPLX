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
from mtplx.cache_state import CacheSnapshot, restore_cache


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


if __name__ == "__main__":
    test_prefix_decode_meta_states_are_none_for_tuple_layers()
    test_restore_cache_skips_prefix_decoded_meta_states()
    print("OK")
