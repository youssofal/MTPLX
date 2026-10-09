"""A spilled session is restored whole from SSD, not from a neighbour's head.

Measured on 2026-10-09 (Qwen3.8-27B, 7 GB bank, two conversations sharing a
4,019-token system prompt): after the bank spilled conversation A to SSD to
make room for B, A's next turn found no exact prefix in RAM and took the
near-prefix lane first. That lane asked the SSD tier for its best candidate,
which was A's own spilled entry, a whole prefix of the prompt; it hydrated
it (5.4 GB), refused it (the lane serves only matched < prefix_len; whole
prefixes are the exact lane's), and served B's checkpoint at the shared
system prompt instead. The turn restored 4,019 tokens and re-read 54,510
(143 s), while session_bank.restore() would have read A's 57,989 tokens back
from SSD. The near lane now has to beat the longest SSD whole prefix, read
from the manifest without hydrating anything.

CPU-sized: toy runtime, real SessionBank and cold tier on tmp_path.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx

from mtplx.cache_bank import SessionBankColdTier
from mtplx.cache_state import CacheSnapshot
from mtplx.generation import restore_or_prefill_prompt_state
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.session_bank import SessionBank

MODEL = str(Path("models/example"))
SPILLED = list(range(1, 42))  # conversation A, on SSD only
PROMPT = tuple(SPILLED + [70, 71, 72])  # A's next turn
NEIGHBOUR = SPILLED[:16] + [90 + i for i in range(30)]  # B shares 16 tokens


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)


def _tier(tmp_path) -> SessionBankColdTier:
    return SessionBankColdTier(
        base_dir=tmp_path / "session-bank", mode="on", min_prefix_tokens=2
    )


def _bank(cold) -> SessionBank:
    return SessionBank(
        max_entries=8, max_bytes=4096, per_session_max_bytes=4096, cold_tier=cold
    )


def _put(bank: SessionBank, token_ids, session_id: str):
    return bank.put_snapshot(
        runtime=_runtime(),
        token_ids=token_ids,
        cache_snapshot=CacheSnapshot(states=(), meta_states=()),
        logits=None,
        hidden=None,
        session_id=session_id,
        snapshot_epoch=len(token_ids),
        nbytes_override=128,
    )


def _spill_a_keep_b(tmp_path):
    cold = _tier(tmp_path)
    bank = _bank(cold)
    _put(bank, SPILLED, "a")
    assert cold.flush(timeout_s=5.0) is True
    bank.clear()
    _put(bank, NEIGHBOUR, "b")
    return cold, bank


def _count_hydrations(cold, monkeypatch) -> list:
    calls: list = []
    original = cold._restore_row

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(cold, "_restore_row", spy)
    return calls


def test_the_probe_reads_the_manifest_only(tmp_path, monkeypatch):
    cold, bank = _spill_a_keep_b(tmp_path)
    try:
        calls = _count_hydrations(cold, monkeypatch)
        before = cold.stats()
        probe = {"model_path": MODEL, "mtp_enabled": True}
        assert bank.cold_exact_prefix_len(PROMPT, **probe) == len(SPILLED)
        # B's prompt shares only the opening: no whole prefix on SSD.
        assert bank.cold_exact_prefix_len(tuple(NEIGHBOUR), **probe) == 0
        # Another identity never matches.
        assert bank.cold_exact_prefix_len(PROMPT, model_path="other", mtp_enabled=True) == 0
        assert calls == []
        after = cold.stats()
        for key in ("restore_hits", "restore_misses", "restore_failures"):
            assert after[key] == before[key]
    finally:
        cold.close()


def test_the_probe_is_zero_without_a_readable_tier(tmp_path):
    assert SessionBank(max_entries=2, max_bytes=4096).cold_exact_prefix_len(
        PROMPT, model_path=MODEL, mtp_enabled=True
    ) == 0
    cold = SessionBankColdTier(
        base_dir=tmp_path / "write-only", mode="write-only", min_prefix_tokens=2
    )
    try:
        assert cold.exact_prefix_len(PROMPT, model_path=MODEL, mtp_enabled=True) == 0
    finally:
        cold.close()


def _near(bank: SessionBank, floor: int):
    return bank.near_prefix_candidates(
        PROMPT,
        min_matched_tokens=4,
        block_size=8,
        block_min_matched_tokens=8,
        model_path=MODEL,
        mtp_enabled=True,
        min_restore_tokens=floor,
    )


def test_without_the_floor_the_spilled_session_is_hydrated_for_nothing(
    tmp_path, monkeypatch
):
    """The 2026-10-09 shape: the best candidate is A's whole prefix (which the
    near lane refuses), hydrated on the request path."""
    cold, bank = _spill_a_keep_b(tmp_path)
    try:
        calls = _count_hydrations(cold, monkeypatch)
        matches = _near(bank, 0)
        assert calls == [1]
        entry, matched = matches[0]
        assert getattr(entry, "cache_source", None) == "ssd"
        assert matched == entry.prefix_len == len(SPILLED)
    finally:
        cold.close()


def test_at_the_floor_nothing_is_hydrated_and_no_candidate_beats_it(
    tmp_path, monkeypatch
):
    cold, bank = _spill_a_keep_b(tmp_path)
    try:
        calls = _count_hydrations(cold, monkeypatch)
        floor = bank.cold_exact_prefix_len(PROMPT, model_path=MODEL, mtp_enabled=True)
        matches = _near(bank, floor)
        assert calls == []
        assert all(int(matched) <= floor for _entry, matched in matches)
    finally:
        cold.close()


# ---- restore_or_prefill_prompt_state hands the turn to the exact lane ----


class _TinyModel:
    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []

    def __call__(self, input_ids, *, cache=None, return_hidden=False,
                 hidden_variant=None, emit_logits=True, logits_keep=None):
        length = int(input_ids.shape[1])
        hidden = mx.zeros((1, length, 2), dtype=mx.float32)
        keep = length if logits_keep is None else min(length, max(1, int(logits_keep)))
        logits = mx.zeros((1, keep, 4), dtype=mx.float32)
        if not emit_logits:
            return (None, hidden) if return_hidden else None
        return (logits, hidden) if return_hidden else logits


def _tiny_runtime() -> MTPLXRuntime:
    return MTPLXRuntime(
        model=_TinyModel(),
        tokenizer=SimpleNamespace(decode=lambda tokens, **_: ""),
        model_path=Path("tiny"),
        mtp_enabled=True,
        contract=MTPContract(),
    )


class _SpilledBank:
    """RAM holds no exact prefix (``ram_exact`` 0) or a short one; SSD holds
    ``ssd_exact`` tokens, served by restore() as an SSD clone."""

    SUPPORTS_NEAR_PREFIX_MIN_RESTORE = True
    last_miss_reason = None
    cold_tier = None
    eviction_log = ()

    def __init__(self, *, ram_exact: int, ssd_exact: int):
        self.ram_exact = ram_exact
        self.ssd_exact = ssd_exact
        self.near_floors: list[int] = []
        self.probes = 0

    def longest_prefix(self, _ids):
        return SimpleNamespace(prefix_len=self.ram_exact) if self.ram_exact else None

    def cold_exact_prefix_len(self, _ids, **_kwargs):
        self.probes += 1
        return self.ssd_exact

    def near_prefix_candidates(self, _ids, **kwargs):
        self.near_floors.append(int(kwargs["min_restore_tokens"]))
        return []

    def restore(self, rt, _ids, **kwargs):
        factory = kwargs.get("cache_factory")
        return SimpleNamespace(
            entry=SimpleNamespace(prefix_len=self.ssd_exact),
            cache=factory() if callable(factory) else rt.make_cache(),
            logits=mx.zeros((1, 4), dtype=mx.float32),
            hidden=mx.zeros((1, 1, 2), dtype=mx.float32),
            mtp_history_cache=[],
            restore_mode="ssd_clone",
            cache_source="ssd",
            ssd_cache_hit=True,
            ssd_cached_tokens=self.ssd_exact,
            ssd_restore_s=0.25,
        )

    def held_by_session(self):
        return []

    def longest_shared_prefix_tokens(self, _ids, *, session_id=None):
        return self.ssd_exact


def test_the_near_lane_must_beat_the_spilled_whole_prefix():
    bank = _SpilledBank(ram_exact=0, ssd_exact=41)
    state = restore_or_prefill_prompt_state(
        _tiny_runtime(), list(PROMPT), session_bank=bank, session_id="a"
    )
    assert bank.near_floors == [41]
    assert state.ssd_cache_hit is True
    assert state.cached_tokens == 41
    assert state.suffix_tokens == len(PROMPT) - 41


def test_a_ram_exact_prefix_keeps_its_own_floor_and_skips_the_probe():
    bank = _SpilledBank(ram_exact=30, ssd_exact=41)
    restore_or_prefill_prompt_state(
        _tiny_runtime(), list(PROMPT), session_bank=bank, session_id="a"
    )
    assert bank.probes == 0
    assert bank.near_floors == [30]
