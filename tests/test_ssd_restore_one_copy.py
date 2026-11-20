"""A conversation restored from the SSD cache is held once, not twice.

Window n4l2 (2026-11-12, 128 GB M5 Max, Flash-Next Optimized Speed): after a
247,034-token answer the memory guard emptied the RAM bank, and the follow-up
(247,590 tokens) was served from the SSD copy the cache had written a second
after the answer. Within 2.6 s the request grew 16.1 GB for an 8.66 GB entry,
the engine reached 99.8 GB, 3.2 GB past its 90 GiB limit, and the turn ended
in a 507. Three steps each held the state twice:

- the decode built every tensor from blocks read into one buffer (one buffer
  per block before #648: the same) and evaluated them all at the end, so every
  concatenate was allocated while the blocks it read were still held;
- the near-prefix lane copied the candidate the SSD tier had decoded for that
  request alone, and the exact lane banked the decoded entry beside the
  request's copy for the whole turn;
- the first forward grew every QSA layer from the restored rows to the
  request's rows target inside one evaluation, allocating every new buffer
  before the GPU let go of the old ones.

The tests store Flash-Next's attention state (QSA layers: keys and values of
two heads, raw index keys, pooled blocks) through the real codec, cold tier
and session bank, restore it the way a follow-up does, and measure MLX's peak
over what was held before against the bytes of the state restored. Small
arrays, real Metal allocations, no model: the forward is a stand-in that
writes the suffix's rows into each layer, as the model's attention does.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from mtplx import generation  # noqa: E402
from mtplx.cache_bank import SessionBankColdTier, blob_io  # noqa: E402
from mtplx.cache_bank.codec import decode_payload_prefix, encode_payload  # noqa: E402
from mtplx.cache_state import CacheSnapshot  # noqa: E402
from mtplx.models.qwen4_exp import QSACache, qsa_rows_target  # noqa: E402
from mtplx.mtp_patch import MTPContract  # noqa: E402
from mtplx.runtime import MTPLXRuntime  # noqa: E402
from mtplx.session_bank import SessionBank  # noqa: E402

LAYERS = 12  # Flash-Next's QSA layers
HEADS = 2
HEAD_DIM = 64
INDEX_DIM = 64
RATIO = 4
STORED = list(range(1, 16_385))  # the conversation on SSD: 16,384 tokens
STEP = 256


def _qsa_states(rows: int) -> tuple[tuple, int]:
    mx.random.seed(7)
    states, total = [], 0
    for _ in range(LAYERS):
        k = mx.random.normal((1, HEADS, rows, HEAD_DIM)).astype(mx.bfloat16)
        v = mx.random.normal((1, HEADS, rows, HEAD_DIM)).astype(mx.bfloat16)
        raw = mx.random.normal((1, rows, INDEX_DIM)).astype(mx.bfloat16)
        pooled = mx.random.normal((1, rows // RATIO, INDEX_DIM)).astype(mx.bfloat16)
        mx.eval(k, v, raw, pooled)
        total += sum(int(x.nbytes) for x in (k, v, raw, pooled))
        states.append((k, v, raw, pooled))
    return tuple(states), total


def _state_bytes(rows: int) -> int:
    per_row = 2 * HEADS * HEAD_DIM * 2 + INDEX_DIM * 2
    return LAYERS * (rows * per_row + (rows // RATIO) * INDEX_DIM * 2)


def _settle() -> int:
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    base = int(mx.get_active_memory())
    mx.reset_peak_memory()
    return base


def _peak_over(base: int) -> int:
    mx.synchronize()
    return int(mx.get_peak_memory()) - base


def _rows_target(prompt_tokens: int) -> int:
    # The request's rows: its prompt and a 512-token answer, at the KV step.
    return ((prompt_tokens + 512 + STEP - 1) // STEP) * STEP


@pytest.mark.parametrize("hooked", [True, False], ids=["parallel_read", "blob_by_blob"])
def test_the_decode_holds_the_state_once(tmp_path, hooked):
    states, state_bytes = _qsa_states(len(STORED))
    encoded = encode_payload(
        cache_snapshot=CacheSnapshot(states=states, meta_states=tuple(None for _ in states)),
        logits=mx.zeros((1, 8), mx.float32),
        hidden=mx.zeros((1, 8), mx.bfloat16),
        mtp_history_snapshot=None,
        block_size=STEP,
    )
    for name, raw in encoded.tensors.items():
        (tmp_path / f"{name}.bin").write_bytes(raw)
    spec = json.loads(json.dumps(encoded.spec))
    first_keys = states[0][0]
    del states, encoded

    def read(name: str) -> bytes:
        return (tmp_path / f"{name}.bin").read_bytes()

    staging = None
    if hooked:
        read.read_into = lambda items: blob_io.read_into_many(
            (tmp_path / f"{name}.bin", view) for name, view in items
        )
        staging = blob_io.StagingBuffer()
        read.allocate = staging.allocate
    base = _settle()
    decoded = decode_payload_prefix(spec, read, cache_prefix_len=len(STORED))
    if staging is not None:
        staging.release()
    peak = _peak_over(base)
    held = int(mx.get_active_memory()) - base
    assert mx.array_equal(decoded.cache_snapshot.states[0][0], first_keys).item()
    # Before: 1.79 times the state on either read path (probe, 2026-11-12).
    assert held <= state_bytes * 1.02, held / state_bytes
    assert peak <= state_bytes * 1.15, peak / state_bytes


class _QsaForward:
    """Stands in for the model: each forward writes its rows into every QSA
    layer (raw index keys first, then keys and values, as the attention
    does) and evaluates them."""

    def __init__(self):
        self.forwards = 0

    def make_cache(self):
        return [QSACache(compress_ratio=RATIO) for _ in range(LAYERS)]

    def make_mtp_cache(self):
        return []

    def __call__(self, input_ids, *, cache=None, return_hidden=False,
                 hidden_variant=None, emit_logits=True, logits_keep=None):
        self.forwards += 1
        rows = int(input_ids.shape[1])
        leaves = []
        for layer in cache or ():
            if isinstance(layer, QSACache):
                layer.write_raw(mx.zeros((1, rows, INDEX_DIM), mx.bfloat16))
                block = mx.zeros((1, HEADS, rows, HEAD_DIM), mx.bfloat16)
                layer.kv.update_and_fetch(block, block)
                leaves += [layer.kv.keys, layer.kv.values, layer.raw_keys]
        if leaves:
            mx.eval(*leaves)
        hidden = mx.zeros((1, rows, 2), dtype=mx.float32)
        keep = rows if logits_keep is None else min(rows, max(1, int(logits_keep)))
        logits = mx.zeros((1, keep, 4), dtype=mx.float32)
        if not emit_logits:
            return (None, hidden) if return_hidden else None
        return (logits, hidden) if return_hidden else logits


def _runtime(model) -> MTPLXRuntime:
    return MTPLXRuntime(
        model=model,
        tokenizer=SimpleNamespace(decode=lambda tokens, **_: ""),
        model_path=Path("models/example"),
        mtp_enabled=True,
        contract=MTPContract(),
    )


@pytest.fixture(scope="module")
def ssd_store(tmp_path_factory):
    """STORED written once by the real bank and cold tier (the encode is the
    slow part); each test reads it back through its own bank."""

    base_dir = tmp_path_factory.mktemp("ssd-store") / "session-bank"
    rt = _runtime(_QsaForward())
    cold = SessionBankColdTier(base_dir=base_dir, mode="on", min_prefix_tokens=2)
    try:
        bank = SessionBank(
            max_entries=8, max_bytes=8 << 30, per_session_max_bytes=8 << 30, cold_tier=cold
        )
        states, _total = _qsa_states(len(STORED))
        entry = bank.put_snapshot(
            runtime=rt,
            token_ids=STORED,
            cache_snapshot=CacheSnapshot(states=states, meta_states=tuple(None for _ in states)),
            logits=mx.zeros((1, 4), mx.float32),
            hidden=mx.zeros((1, 2), mx.float32),
            hidden_variant=generation._resolve_runtime_base_hidden_variant(rt, None),
            session_id="a",
            snapshot_epoch=len(STORED),
            mtp_history_policy="cycle",
        )
        assert entry is not None
        assert cold.flush(timeout_s=120.0) is True
        first_keys = states[0][0]
        bank.clear()
        del states, entry, bank
    finally:
        cold.close()
    return base_dir, first_keys


def _bank_on(ssd_store, monkeypatch):
    """A bank whose only copy of STORED is the SSD cache's."""

    monkeypatch.setenv("MTPLX_SESSION_PREFIX_BLOCK_SIZE", str(STEP))
    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS", str(STEP))
    monkeypatch.setenv("MTPLX_SESSION_NEAR_PREFIX_MIN_MATCH_TOKENS", "4")
    base_dir, first_keys = ssd_store
    cold = SessionBankColdTier(base_dir=base_dir, mode="on", min_prefix_tokens=2)
    bank = SessionBank(
        max_entries=8, max_bytes=8 << 30, per_session_max_bytes=8 << 30, cold_tier=cold
    )
    return cold, bank, first_keys


def test_a_near_prefix_follow_up_restored_from_ssd_holds_one_copy(ssd_store, monkeypatch):
    # n4l2's shape: the follow-up re-renders the end of the stored turn, so it
    # diverges just before the stored end and resumes from the SSD copy.
    model = _QsaForward()
    rt = _runtime(model)
    cold, bank, first_keys = _bank_on(ssd_store, monkeypatch)
    try:
        prompt = STORED[:16_000] + [90_000 + i for i in range(100)]
        restore_point = (16_000 // STEP) * STEP
        base = _settle()
        with qsa_rows_target(_rows_target(len(prompt))):
            state = generation._restore_near_prefix_prompt_state(
                rt,
                list(prompt),
                base_hidden_variant=generation._resolve_runtime_base_hidden_variant(rt, None),
                mtp_hidden_variant=None,
                mtp_history_policy="cycle",
                session_bank=bank,
                template_hash=None,
                draft_head_identity=None,
                policy_fingerprint=None,
                allow_block_prefix=True,
                session_id="a",
            )
        peak = _peak_over(base)
        assert state is not None and state.ssd_cache_hit
        assert state.cached_tokens == restore_point
        assert model.forwards >= 2
        # Class A: the restored rows are the stored rows.
        keys = state.trunk_cache[0].kv.keys
        assert mx.array_equal(
            keys[..., : restore_point - 1, :], first_keys[..., : restore_point - 1, :]
        ).item()
        state_bytes = _state_bytes(restore_point)
        capacity = _state_bytes(_rows_target(len(prompt)))
        # Before: about 2.2 times the state (decode at 1.8, then the copy and
        # the one-evaluation growth beside it).
        assert peak <= capacity + state_bytes * 0.25, (peak / state_bytes, capacity / state_bytes)
    finally:
        cold.close()


def test_an_exact_follow_up_restored_from_ssd_is_not_banked_beside_its_copy(
    ssd_store, monkeypatch
):
    # After a restart, or when the guard emptied the bank: the next turn
    # extends the stored conversation and is served from the SSD copy, then
    # its suffix is read on the restored cache (generation's exact lane).
    model = _QsaForward()
    rt = _runtime(model)
    cold, bank, first_keys = _bank_on(ssd_store, monkeypatch)
    try:
        prompt = STORED + [90_000 + i for i in range(100)]
        hidden_variant = generation._resolve_runtime_base_hidden_variant(rt, None)
        base = _settle()
        with qsa_rows_target(_rows_target(len(prompt))):
            restored = bank.restore(
                rt,
                prompt,
                mode="clone",
                session_id="a",
                hidden_variant=hidden_variant,
                mtp_history_policy="cycle",
                cache_factory=model.make_cache,
            )
            assert restored is not None and restored.cache_source == "ssd"
            generation._prefill_restored_prompt_suffix(
                rt,
                restored,
                list(prompt[len(STORED):]),
                base_hidden_variant=hidden_variant,
                mtp_hidden_variant=None,
                mtp_history_policy="cycle",
                tokens_total=len(prompt),
                cached_tokens=len(STORED),
            )
        peak = _peak_over(base)
        # The bank keeps no decoded copy beside the request's.
        assert bank.total_nbytes == 0 and not bank._entries
        keys = restored.cache[0].kv.keys
        assert int(keys.shape[2]) == _rows_target(len(prompt))
        assert mx.array_equal(keys[..., : len(STORED), :], first_keys).item()
        state_bytes = _state_bytes(len(STORED))
        capacity = _state_bytes(_rows_target(len(prompt)))
        # Before: the banked entry and the request's grown copy, about 2.8
        # times the state at the peak.
        assert peak <= capacity + state_bytes * 0.25, (peak / state_bytes, capacity / state_bytes)
    finally:
        cold.close()


def test_a_cache_outside_the_one_copy_store_keeps_the_banked_entry(tmp_path, monkeypatch):
    # Other models' caches keep the old contract: the decoded entry is banked
    # and the request gets its own copy.
    from mlx_lm.models.cache import KVCache

    class _Plain(_QsaForward):
        def make_cache(self):
            return [KVCache() for _ in range(LAYERS)]

    model = _Plain()
    rt = _runtime(model)
    monkeypatch.setenv("MTPLX_SESSION_PREFIX_BLOCK_SIZE", str(STEP))
    cold = SessionBankColdTier(base_dir=tmp_path / "session-bank", mode="on", min_prefix_tokens=2)
    try:
        bank = SessionBank(
            max_entries=8, max_bytes=8 << 30, per_session_max_bytes=8 << 30, cold_tier=cold
        )
        mx.random.seed(3)
        states = tuple(
            (
                mx.random.normal((1, HEADS, 1024, HEAD_DIM)).astype(mx.bfloat16),
                mx.random.normal((1, HEADS, 1024, HEAD_DIM)).astype(mx.bfloat16),
            )
            for _ in range(2)
        )
        bank.put_snapshot(
            runtime=rt,
            token_ids=STORED[:1024],
            cache_snapshot=CacheSnapshot(states=states, meta_states=(None, None)),
            logits=mx.zeros((1, 4), mx.float32),
            hidden=mx.zeros((1, 2), mx.float32),
            hidden_variant=generation._resolve_runtime_base_hidden_variant(rt, None),
            session_id="a",
            snapshot_epoch=1024,
            mtp_history_policy="cycle",
        )
        assert cold.flush(timeout_s=30.0) is True
        bank.clear()
        restored = bank.restore(
            rt,
            STORED[:1024] + [7, 8, 9],
            mode="clone",
            session_id="a",
            hidden_variant=generation._resolve_runtime_base_hidden_variant(rt, None),
            mtp_history_policy="cycle",
            cache_factory=lambda: [KVCache() for _ in range(2)],
        )
        assert restored is not None and restored.cache_source == "ssd"
        assert len(bank._entries) == 1
        assert restored.entry.cache_snapshot is not None
    finally:
        cold.close()
