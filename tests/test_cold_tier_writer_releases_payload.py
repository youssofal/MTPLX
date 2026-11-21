"""The SSD writer keeps nothing of a write once it has written it.

``put_entry`` encodes a whole entry into host bytes and queues it; the writer
thread writes the files and moves on. Its loop variable used to hold the last
write, every encoded tensor's bytes with it, while it waited for the next one:
host memory as large as the entry, for as long as the server was idle (1.7 GB
after an Inkling 2 Mini 4K turn through ``mtplx serve``, 21 November 2026).
"""

from __future__ import annotations

import gc
import weakref

from mtplx.cache_bank import SessionBankColdTier
from mtplx.cache_bank.cold_tier import PendingWrite


def _tier(tmp_path) -> SessionBankColdTier:
    return SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        max_bytes=64 * 1024**3,
        min_prefix_tokens=2,
    )


def _pending(payload: bytes) -> PendingWrite:
    return PendingWrite(
        entry_id="entry-one",
        token_ids=(1, 2, 3),
        metadata={"prefix_len": 3, "nbytes": len(payload)},
        payload_spec=None,
        tensors={"cache/0": payload},
        pinned_nbytes=len(payload),
    )


def test_a_written_entry_is_released_once_the_write_is_done(tmp_path, monkeypatch):
    cold = _tier(tmp_path)
    try:
        written = []
        monkeypatch.setattr(cold, "_write_pending", lambda pending: written.append(pending.entry_id) or True)
        pending = _pending(b"\x01" * (1 << 20))
        alive = weakref.ref(pending)
        cold._pending_bytes += pending.pinned_nbytes
        cold._queue.put_nowait(pending)
        del pending
        assert cold.flush(timeout_s=10.0)
        gc.collect()
        assert written == ["entry-one"]
        assert alive() is None, "the writer still holds the written entry's encoded bytes"
        assert cold._pending_bytes == 0
    finally:
        cold.close()


def test_a_failed_write_is_released_too(tmp_path, monkeypatch):
    cold = _tier(tmp_path)
    try:
        def fail(pending):
            raise OSError("disk went away")

        monkeypatch.setattr(cold, "_write_pending", fail)
        pending = _pending(b"\x02" * (1 << 20))
        alive = weakref.ref(pending)
        cold._queue.put_nowait(pending)
        del pending
        assert cold.flush(timeout_s=10.0)
        gc.collect()
        assert alive() is None, "a failed write's encoded bytes stayed alive in the writer"
        assert int(cold._stats.get("write_failures", 0)) == 1
    finally:
        cold.close()
