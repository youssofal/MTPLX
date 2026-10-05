"""Incremental SSD encode (MTPLX_SSD_INCREMENTAL_ENCODE, opt-in).

The switch must leave the store byte-identical to a full encode: the same
payload spec, the same tensor names, the same blob digests and the same blob
bytes. Only the work changes: KV blocks (and large whole tensors) whose
content key matches the session's previous completed write are referenced
instead of evaluated, copied to the host and hashed again.

It stays off by default until the content key is a content identity: two
float32 sign flips leave the fingerprint unchanged (the collision tests
below), so a changed tensor could borrow its predecessor's blob.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mtplx.cache_bank.codec import content_fingerprints, encode_payload
from mtplx.cache_bank.cold_tier import SessionBankColdTier
from mtplx.cache_state import CacheSnapshot

SESSION = "sess-incremental"
MODEL = "models/example"


def _rand(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal(shape).astype(np.float32)).astype(dtype)


class _Entry:
    def __init__(self, *, tokens, kv, gdn, mtp, session_id=SESSION):
        self.token_ids = tuple(tokens)
        self.cache_snapshot = CacheSnapshot(
            states=((kv[0], kv[1]), (gdn,)),
            meta_states=({"offset": len(tokens)}, {"kind": "gdn"}),
        )
        self.logits = mx.arange(16, dtype=mx.float32).reshape(1, 16)
        self.hidden = mx.ones((1, 8), dtype=mx.bfloat16)
        self.mtp_history_snapshot = CacheSnapshot(
            states=((mtp,),), meta_states=({"offset": len(tokens)},)
        )
        self.gdn_boundaries = ()
        self.has_recurrent = True
        self.session_id = session_id
        self.model_path = MODEL
        self.mtp_enabled = True
        self.snapshot_epoch = len(tokens)
        self.mtp_snapshot_epoch = len(tokens)
        self.nbytes = sum(
            int(a.nbytes) for a in (kv[0], kv[1], gdn, mtp, self.logits, self.hidden)
        )


def _turns():
    """Turn 1: 1,024 tokens. Turn 2: the same KV bits plus 300 new tokens
    (4 full blocks shared, the 5th block and a partial 6th are new) and the
    same 1 MiB recurrent state, as after a restore from turn 1."""
    k1 = _rand((1, 2, 1024, 64), mx.bfloat16, 1)
    v1 = _rand((1, 2, 1024, 64), mx.bfloat16, 2)
    m1 = _rand((1, 1, 1024, 32), mx.bfloat16, 3)
    gdn = _rand((1, 8, 128, 256), mx.float32, 4)  # exactly 1 MiB
    turn1 = _Entry(tokens=range(1024), kv=(k1, v1), gdn=gdn, mtp=m1)
    k2 = mx.concatenate([k1, _rand((1, 2, 300, 64), mx.bfloat16, 5)], axis=2)
    v2 = mx.concatenate([v1, _rand((1, 2, 300, 64), mx.bfloat16, 6)], axis=2)
    m2 = mx.concatenate([m1, _rand((1, 1, 300, 32), mx.bfloat16, 7)], axis=2)
    turn2 = _Entry(tokens=range(1324), kv=(k2, v2), gdn=gdn, mtp=m2)
    return turn1, turn2


def _tier(path: Path) -> SessionBankColdTier:
    return SessionBankColdTier(
        base_dir=path, mode="on", min_prefix_tokens=2, block_size=256
    )


def _payload(tier: SessionBankColdTier, entry_id: str) -> dict:
    path = tier.base_dir / "entries" / entry_id[:2] / entry_id / "payload.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metadata"].pop("created_at_s")
    return payload


def _blobs(tier: SessionBankColdTier) -> dict[str, bytes]:
    return {
        p.name: p.read_bytes()
        for p in (tier.base_dir / "blobs").rglob("*.bin")
    }


def _write(tier: SessionBankColdTier, entry) -> str:
    assert tier.put_entry(entry) is True
    assert tier.flush(timeout_s=10.0) is True
    return tier._metadata_for_entry(entry, capabilities=(), payload_nbytes=0)["entry_id"]


def _bits_equal(a, b) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return bool(mx.array_equal(a.view(view), b.view(view)).item())


def test_fingerprint_sees_every_bit():
    base = mx.zeros((1, 1, 512, 4), dtype=mx.bfloat16)
    negative_zero = mx.array(np.array([-0.0], dtype=np.float32)).astype(mx.bfloat16)
    flipped = mx.concatenate(
        [base[:, :, :300], mx.broadcast_to(negative_zero, (1, 1, 1, 4)), base[:, :, 301:]],
        axis=2,
    )
    fp_base = content_fingerprints(base, rows=256)
    fp_flip = content_fingerprints(flipped, rows=256)
    assert len(fp_base) == 2
    assert fp_base[0] == fp_flip[0]
    assert fp_base[1] != fp_flip[1]  # -0.0 == +0.0 as a value, not as bits
    assert content_fingerprints(base, rows=256) == fp_base  # deterministic
    # Swapped elements inside a block change the fingerprint (positional).
    a = mx.arange(512 * 4, dtype=mx.float32).reshape(1, 1, 512, 4)
    b = mx.concatenate([a[:, :, 1:2], a[:, :, 0:1], a[:, :, 2:]], axis=2)
    assert content_fingerprints(a, rows=256)[0] != content_fingerprints(b, rows=256)[0]


def test_switch_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("MTPLX_SSD_INCREMENTAL_ENCODE", raising=False)
    tier = _tier(tmp_path / "default")
    try:
        turn1, turn2 = _turns()
        _write(tier, turn1)
        _write(tier, turn2)
        stats = tier.stats()
        assert stats["incremental_encode"] is False
        assert stats["incremental_encodes"] == 0
        assert stats["incremental_reused_blobs"] == 0
        assert tier._block_memos == {}
    finally:
        tier.close()


def _two_signs_flipped(turn):
    """``turn`` with the signs of two elements of its float32 recurrent state
    flipped. Every fingerprint weight is odd, so each flip adds 2**31 to the
    linear sum and the pair cancels modulo 2**32; (x + 2**31)**2 equals x**2
    modulo 2**32, so the squared sum cannot see a flip at all."""
    (gdn,) = turn.cache_snapshot.states[1]
    flat = np.array(gdn.reshape(-1))
    flat[[5, 77]] *= -1.0
    flipped = mx.array(flat).reshape(gdn.shape)
    turn.cache_snapshot = CacheSnapshot(
        states=(turn.cache_snapshot.states[0], (flipped,)),
        meta_states=turn.cache_snapshot.meta_states,
    )
    return gdn, flipped


def test_default_store_restores_a_state_with_two_flipped_signs(tmp_path, monkeypatch):
    """Turn 2's recurrent state differs from turn 1's in two signs. The
    fingerprint tells them apart, and the default (full) encode stores and
    restores turn 2's own bytes either way."""
    turn1, turn2 = _turns()
    gdn, flipped = _two_signs_flipped(turn2)
    assert not _bits_equal(gdn, flipped)
    assert content_fingerprints(gdn) != content_fingerprints(flipped)

    monkeypatch.delenv("MTPLX_SSD_INCREMENTAL_ENCODE", raising=False)
    tier = _tier(tmp_path / "store")
    try:
        _write(tier, turn1)
        _write(tier, turn2)
        record = tier.lookup(
            list(turn2.token_ids) + [7, 7], model_path=MODEL, mtp_enabled=True
        )
        assert record is not None
        assert record.token_ids == turn2.token_ids
        (restored,) = record.cache_snapshot.states[1]
        assert _bits_equal(restored, flipped)
    finally:
        tier.close()


def test_incremental_store_is_full_encode_when_the_fingerprint_collides(
    tmp_path, monkeypatch
):
    turn1, turn2 = _turns()
    _two_signs_flipped(turn2)
    monkeypatch.setenv("MTPLX_SSD_INCREMENTAL_ENCODE", "0")
    full = _tier(tmp_path / "full")
    monkeypatch.setenv("MTPLX_SSD_INCREMENTAL_ENCODE", "1")
    incremental = _tier(tmp_path / "incremental")
    try:
        for entry in (turn1, turn2):
            entry_id = _write(full, entry)
            assert _write(incremental, entry) == entry_id
            assert _payload(incremental, entry_id) == _payload(full, entry_id)
    finally:
        full.close()
        incremental.close()


def _flip_signs(array, positions):
    """Copy of ``array`` with the sign bit flipped at flat ``positions``."""
    view = {2: mx.uint16, 4: mx.uint32}[array.dtype.size]
    top = 1 << (8 * array.dtype.size - 1)
    mask = np.zeros(array.size, dtype=np.uint32)
    mask[list(positions)] = top
    bits = array.reshape(-1).view(view) ^ mx.array(mask.astype(np.dtype(f"u{array.dtype.size}")))
    return bits.view(array.dtype).reshape(array.shape)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("flips", [1, 2, 4])
def test_sign_flips_change_the_fingerprint(dtype, flips):
    base = _rand((1, 2, 256, 64), dtype, 21)
    rng = np.random.default_rng(flips)
    for _ in range(25):
        positions = rng.choice(base.size, flips, replace=False)
        flipped = _flip_signs(base, positions)
        assert not _bits_equal(base, flipped)
        assert content_fingerprints(base) != content_fingerprints(flipped)
        assert content_fingerprints(base, rows=256) != content_fingerprints(
            flipped, rows=256
        )


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_swapped_values_change_the_fingerprint(dtype):
    base = _rand((1, 2, 256, 64), dtype, 22)
    flat = np.array(base.astype(mx.float32).reshape(-1))
    rng = np.random.default_rng(3)
    for _ in range(25):
        i, j = rng.choice(flat.size, 2, replace=False)
        if flat[i] == flat[j]:
            continue
        swapped = flat.copy()
        swapped[[i, j]] = swapped[[j, i]]
        other = mx.array(swapped).astype(dtype).reshape(base.shape)
        assert content_fingerprints(base) != content_fingerprints(other)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_small_perturbations_change_the_fingerprint(dtype):
    base = _rand((1, 2, 256, 64), dtype, 23)
    view = {2: mx.uint16, 4: mx.uint32}[dtype.size]
    bits = np.array(base.reshape(-1).view(view))
    rng = np.random.default_rng(4)
    seen = {tuple(content_fingerprints(base))}
    for _ in range(100):
        changed = bits.copy()
        for pos in rng.choice(bits.size, int(rng.integers(1, 4)), replace=False):
            changed[pos] ^= bits.dtype.type(1 << int(rng.integers(0, 8 * dtype.size)))
        other = mx.array(changed).view(dtype).reshape(base.shape)
        fp = tuple(content_fingerprints(other))
        assert fp not in seen
        seen.add(fp)


def test_identical_content_has_identical_fingerprints():
    for dtype, shape in ((mx.bfloat16, (1, 2, 512, 64)), (mx.float32, (1, 8, 128, 256))):
        a = _rand(shape, dtype, 24)
        b = mx.array(np.array(a.astype(mx.float32))).astype(dtype)
        assert content_fingerprints(a) == content_fingerprints(b)
        assert content_fingerprints(a, rows=256 if shape[2] % 256 == 0 else 128) == (
            content_fingerprints(b, rows=256 if shape[2] % 256 == 0 else 128)
        )


def test_fingerprint_has_two_32_bit_halves_and_one_per_block():
    kv = _rand((1, 2, 768, 64), mx.bfloat16, 25)
    fps = content_fingerprints(kv, rows=256)
    assert len(fps) == 3
    assert all(0 <= fp < 1 << 64 for fp in fps)
    assert len(set(fps)) == 3
    gdn = _rand((1, 8, 128, 256), mx.float32, 26)
    (whole,) = content_fingerprints(gdn)
    assert 0 <= whole < 1 << 64


def test_uint32_multiply_wraps_modulo_2_32():
    a = mx.array([0xFFFFFFFF, 0x80000000], dtype=mx.uint32)
    product = a * mx.array(0x846CA68B, dtype=mx.uint32)
    expected = (np.array([0xFFFFFFFF, 0x80000000], dtype=np.uint64) * 0x846CA68B) & 0xFFFFFFFF
    assert product.tolist() == expected.tolist()


def _session_turns(dtype, seed):
    """Five banked turns of one session, the shapes an agent session takes:
    a first turn, an extension (the prefix restored bit for bit), an
    extension whose recurrent state changed, an extension whose block 1
    differs in one element's lowest bit (a re-prefill with other numerics),
    and a turn whose history was rewritten from block 2 on (a compaction or
    an edit)."""

    def kv(tokens, salt):
        return (
            _rand((1, 2, tokens, 64), dtype, seed * 100 + salt),
            _rand((1, 2, tokens, 64), dtype, seed * 100 + salt + 1),
        )

    def entry(tokens, k, v, gdn, mtp):
        return _Entry(tokens=range(tokens), kv=(k, v), gdn=gdn, mtp=mtp)

    k1, v1 = kv(700, 1)
    mtp1 = _rand((1, 1, 700, 32), dtype, seed * 100 + 3)
    gdn1 = _rand((1, 8, 128, 256), mx.float32, seed * 100 + 4)
    turns = [entry(700, k1, v1, gdn1, mtp1)]

    k_tail, v_tail = kv(400, 5)
    k2 = mx.concatenate([k1, k_tail], axis=2)
    v2 = mx.concatenate([v1, v_tail], axis=2)
    mtp2 = mx.concatenate([mtp1, _rand((1, 1, 400, 32), dtype, seed * 100 + 7)], axis=2)
    turns.append(entry(1100, k2, v2, gdn1, mtp2))

    gdn3 = _rand((1, 8, 128, 256), mx.float32, seed * 100 + 8)
    k_tail, v_tail = kv(300, 9)
    k3 = mx.concatenate([k2, k_tail], axis=2)
    v3 = mx.concatenate([v2, v_tail], axis=2)
    mtp3 = mx.concatenate([mtp2, _rand((1, 1, 300, 32), dtype, seed * 100 + 11)], axis=2)
    turns.append(entry(1400, k3, v3, gdn3, mtp3))

    view = {2: mx.uint16, 4: mx.uint32}[dtype.size]
    bits = k3.view(view)
    flipped = bits[:, :, 300:301] ^ mx.array(1, dtype=view)
    k_flip = mx.concatenate(
        [bits[:, :, :300], flipped, bits[:, :, 301:]], axis=2
    ).view(dtype)
    k_tail, v_tail = kv(100, 17)
    k4 = mx.concatenate([k_flip, k_tail], axis=2)
    v4 = mx.concatenate([v3, v_tail], axis=2)
    mtp4 = mx.concatenate([mtp3, _rand((1, 1, 100, 32), dtype, seed * 100 + 19)], axis=2)
    turns.append(entry(1500, k4, v4, gdn3, mtp4))

    k_new, v_new = kv(1000 - 512, 13)
    k5 = mx.concatenate([k4[:, :, :512], k_new], axis=2)
    v5 = mx.concatenate([v3[:, :, :512], v_new], axis=2)
    mtp5 = mx.concatenate(
        [mtp3[:, :, :512], _rand((1, 1, 1000 - 512, 32), dtype, seed * 100 + 15)],
        axis=2,
    )
    turns.append(entry(1000, k5, v5, gdn3, mtp5))
    return turns


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_every_turn_shape_stores_what_the_full_encode_stores(
    tmp_path, monkeypatch, dtype, seed
):
    """Across dtypes, seeds and every turn shape an agent session produces,
    each incremental write is the full encode's payload (spec, names,
    digests) and the store holds the same blob bytes; the one-bit and
    rewritten blocks were captured, not borrowed."""

    monkeypatch.setenv("MTPLX_SSD_INCREMENTAL_ENCODE", "0")
    full = _tier(tmp_path / "full")
    monkeypatch.setenv("MTPLX_SSD_INCREMENTAL_ENCODE", "1")
    incremental = _tier(tmp_path / "incremental")
    try:
        for turn in _session_turns(dtype, seed):
            entry_id = _write(full, turn)
            assert _write(incremental, turn) == entry_id
            assert _payload(incremental, entry_id) == _payload(full, entry_id)
        assert _blobs(incremental) == _blobs(full)
        stats = incremental.stats()
        assert stats["incremental_reused_blobs"] > 0
        assert stats["incremental_missing_blobs"] == 0
    finally:
        full.close()
        incremental.close()


def test_incremental_store_is_identical_to_full_encode(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_SSD_INCREMENTAL_ENCODE", "0")
    full = _tier(tmp_path / "full")
    monkeypatch.setenv("MTPLX_SSD_INCREMENTAL_ENCODE", "1")
    incremental = _tier(tmp_path / "incremental")
    try:
        assert incremental.stats()["incremental_encode"] is True
        ids = []
        for entry in _turns():
            ids.append(_write(full, entry))
            assert _write(incremental, entry) == ids[-1]
        for entry_id in ids:
            assert _payload(incremental, entry_id) == _payload(full, entry_id)
        assert _blobs(incremental) == _blobs(full)

        stats = incremental.stats()
        # Turn 2 reused 4 full blocks each of K, V and the MTP history, plus
        # the unchanged 1 MiB recurrent state.
        assert stats["incremental_encodes"] == 1
        assert stats["incremental_reused_blobs"] == 4 * 3 + 1
        assert stats["incremental_reused_bytes"] == (
            4 * (2 * 2 * 256 * 64 * 2) + 4 * (256 * 32 * 2) + 1024 * 1024
        )
        assert stats["incremental_missing_blobs"] == 0
    finally:
        full.close()
        incremental.close()


def test_changed_prefix_block_is_captured_not_reused(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_SSD_INCREMENTAL_ENCODE", "0")
    full = _tier(tmp_path / "full")
    monkeypatch.setenv("MTPLX_SSD_INCREMENTAL_ENCODE", "1")
    incremental = _tier(tmp_path / "incremental")
    try:
        turn1, turn2 = _turns()
        # One element of block 1 of K differs from turn 1 (e.g. a re-prefill
        # with other numerics): that block must be captured, never borrowed.
        k2 = turn2.cache_snapshot.states[0][0]
        bumped = k2[:, :, 300:301] + mx.array(1.0, dtype=mx.bfloat16)
        k2 = mx.concatenate([k2[:, :, :300], bumped, k2[:, :, 301:]], axis=2)
        turn2.cache_snapshot = CacheSnapshot(
            states=((k2, turn2.cache_snapshot.states[0][1]), turn2.cache_snapshot.states[1]),
            meta_states=turn2.cache_snapshot.meta_states,
        )
        ids = []
        for entry in (turn1, turn2):
            ids.append(_write(full, entry))
            _write(incremental, entry)
        for entry_id in ids:
            assert _payload(incremental, entry_id) == _payload(full, entry_id)
        assert _blobs(incremental) == _blobs(full)
        assert incremental.stats()["incremental_reused_blobs"] == 3 + 4 + 4 + 1
    finally:
        full.close()
        incremental.close()


def test_restore_from_incremental_write_is_bit_identical(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_SSD_INCREMENTAL_ENCODE", "1")
    base = tmp_path / "store"
    tier = _tier(base)
    turn1, turn2 = _turns()
    try:
        _write(tier, turn1)
        _write(tier, turn2)
        assert tier.stats()["incremental_reused_blobs"] > 0
    finally:
        tier.close()
    # After a restart, with the switch off: the store is an ordinary store.
    monkeypatch.setenv("MTPLX_SSD_INCREMENTAL_ENCODE", "0")
    reopened = _tier(base)
    try:
        record = reopened.lookup(
            list(turn2.token_ids) + [7, 7],
            model_path=MODEL,
            mtp_enabled=True,
        )
        assert record is not None
        assert record.token_ids == turn2.token_ids
        (k, v), (gdn,) = record.cache_snapshot.states
        (k2, v2), (gdn2,) = turn2.cache_snapshot.states
        assert _bits_equal(k, k2)
        assert _bits_equal(v, v2)
        assert _bits_equal(gdn, gdn2)
        assert _bits_equal(
            record.mtp_history_snapshot.states[0][0],
            turn2.mtp_history_snapshot.states[0][0],
        )
        assert _bits_equal(record.logits, turn2.logits)
        assert _bits_equal(record.hidden, turn2.hidden)
    finally:
        reopened.close()


def test_missing_reused_blob_skips_write_then_full_encode_recovers(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_SSD_INCREMENTAL_ENCODE", "1")
    tier = _tier(tmp_path / "store")
    try:
        turn1, turn2 = _turns()
        _write(tier, turn1)
        # A blob the memo points at disappears (its entries were evicted).
        memo = tier._block_memo(SESSION)
        victim = next(iter(memo.values()))["sha256"]
        tier._blob_path(victim).unlink()

        assert tier.put_entry(turn2) is True
        assert tier.flush(timeout_s=10.0) is True
        stats = tier.stats()
        assert stats["incremental_missing_blobs"] == 1
        assert stats["writes_completed"] == 1
        assert tier._block_memo(SESSION) == {}

        # Memo dropped: the retry encodes in full and lands.
        _write(tier, turn2)
        stats = tier.stats()
        assert stats["writes_completed"] == 2
        assert stats["incremental_encodes"] == 1
        record = tier.lookup(list(turn2.token_ids), model_path=MODEL, mtp_enabled=True)
        assert record is not None
        assert _bits_equal(
            record.cache_snapshot.states[0][0], turn2.cache_snapshot.states[0][0]
        )
    finally:
        tier.close()


def test_other_session_does_not_borrow(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_SSD_INCREMENTAL_ENCODE", "1")
    tier = _tier(tmp_path / "store")
    try:
        turn1, turn2 = _turns()
        _write(tier, turn1)
        turn2.session_id = "another-session"
        _write(tier, turn2)
        assert tier.stats()["incremental_reused_blobs"] == 0
    finally:
        tier.close()


def test_encode_payload_without_reuse_is_unchanged():
    turn1, _ = _turns()
    kwargs = {
        "cache_snapshot": turn1.cache_snapshot,
        "logits": turn1.logits,
        "hidden": turn1.hidden,
        "mtp_history_snapshot": turn1.mtp_history_snapshot,
        "has_recurrent": True,
        "block_size": 256,
    }
    legacy = encode_payload(**kwargs)
    empty_memo = encode_payload(**kwargs, reuse={})
    assert legacy.spec == empty_memo.spec
    assert legacy.tensors == empty_memo.tensors
    assert legacy.reused == {} and legacy.fingerprints == {}
    assert empty_memo.reused == {} and empty_memo.fingerprints
