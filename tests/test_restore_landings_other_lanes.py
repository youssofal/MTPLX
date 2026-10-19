"""Restore landings and the SSD restore split on the lanes outside
generation's restore_or_prefill: the Gemma 4 target+assistant pair and the
batched MTP lane's session hooks.

Gemma 4 has no recurrent layers, so its restores land on the matched length
itself: the restore point equals the match and the checkpoint gap is 0,
recorded as such rather than left empty.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mtplx.backends.gemma4_assistant as gemma4
from mtplx.cache_bank.telemetry import ORIGIN_OTHER, ORIGIN_SAME
from mtplx.server import openai
from mtplx.session_bank import SessionBank


class _TokenCache:
    def __init__(self, token_ids=None) -> None:
        self.token_ids = list(token_ids or [])
        self.offset = len(self.token_ids)

    @property
    def state(self):
        raise RuntimeError("live reference only")

    def is_trimmable(self) -> bool:
        return True

    def trim(self, count: int) -> int:
        count = int(count)
        if count:
            del self.token_ids[-count:]
        self.offset = len(self.token_ids)
        return count


class _GemmaRuntime:
    model_path = Path("models/gemma4")
    mtp_enabled = True

    def __init__(self) -> None:
        self.target = SimpleNamespace(cache_offset=lambda cache: cache[0].offset)

    def make_cache(self):
        return [_TokenCache()]


def _fake_prefill(_runtime, prompt_ids, *, cache, phase, abort_check=None):
    cache[0].token_ids.extend(int(token) for token in prompt_ids)
    cache[0].offset = len(cache[0].token_ids)
    out = SimpleNamespace(
        logits=_Any(), hidden=_Any(), shared_kv_states={}, cache_offset=cache[0].offset
    )
    return out, 0.01


class _Any:
    def __getitem__(self, _key):
        return self


def _gemma_bank(runtime, stored, *, session_id="g1"):
    # Budgets below the entry's size keep it a live reference (no snapshot of
    # the fake cache), as in test_gemma4_session_cache.
    bank = SessionBank(max_entries=4, max_bytes=1024, per_session_max_bytes=512)
    entry = bank.put(
        runtime=runtime,
        token_ids=stored,
        cache=[_TokenCache(stored)],
        logits=("stored", len(stored)),
        hidden=("stored", len(stored)),
        hidden_variant="gemma4_pre_norm",
        keep_live_ref=True,
        session_id=session_id,
        template_hash="t",
        mtp_history_policy=gemma4.GEMMA4_SESSION_STATE_POLICY,
        draft_head_identity="d",
        policy_fingerprint="p",
        snapshot_epoch=len(stored),
        nbytes_override=1024,
        extra_state={"gemma4_shared_kv_states": {}, "gemma4_kv_offset": len(stored)},
    )
    assert entry is not None
    return bank, entry


def _gemma_restore(runtime, bank, prompt, *, session_id):
    return gemma4._restore_or_prefill_gemma4_prompt(
        runtime,
        prompt,
        session_bank=bank,
        session_restore_mode="reference",
        session_template_hash="t",
        session_draft_head_identity="d",
        session_policy_fingerprint="p",
        require_shared_kv=False,
        session_id=session_id,
    )


class TestGemma4:
    def test_prefix_restore_lands_on_the_match_with_no_gap(self, monkeypatch):
        monkeypatch.setattr(gemma4, "_gemma4_prefill_prompt", _fake_prefill)
        runtime = _GemmaRuntime()
        stored = list(range(1, 1001))
        bank, _entry = _gemma_bank(runtime, stored)
        prompt = list(range(1, 701)) + [5000 + i for i in range(40)]
        state = _gemma_restore(runtime, bank, prompt, session_id="g1")
        assert state.cache_hit and state.restore_mode.startswith("block_prefix")
        landing = state.restore_served["landing"]
        matched = landing["matched_tokens"]
        assert matched > 0
        assert landing["restore_point_tokens"] == matched
        assert landing["checkpoint_gap_tokens"] == 0
        assert landing["hybrid"] is False
        assert landing["partial"] is True
        assert landing["origin"] == ORIGIN_SAME
        assert landing["source"] == "ram"
        assert landing["suffix_tokens"] == state.suffix_tokens
        stats = gemma4._gemma4_restore_stats(state)
        assert stats["ssd_cache_hit"] is False
        assert stats["session_restore_served"]["landing"] == landing
        fields = openai._restore_telemetry_fields(stats)
        assert fields["prefix_restore_point_tokens"] == matched
        assert fields["prefix_restore_checkpoint_gap_tokens"] == 0
        counters = bank.to_dict()["restore_landings"]["ram"]
        assert counters["restores"] == 1 and counters["partial"] == 1
        assert counters["checkpoint_gap_tokens"] == 0

    def test_exact_restore_from_another_session(self, monkeypatch):
        monkeypatch.setattr(gemma4, "_gemma4_prefill_prompt", _fake_prefill)
        runtime = _GemmaRuntime()
        stored = list(range(1, 601))
        bank, _entry = _gemma_bank(runtime, stored, session_id="other")
        state = _gemma_restore(runtime, bank, stored + [7, 8, 9], session_id="g1")
        landing = state.restore_served["landing"]
        assert landing["partial"] is False
        assert landing["matched_tokens"] == landing["restore_point_tokens"] == 600
        assert landing["checkpoint_gap_tokens"] == 0
        assert landing["origin"] == ORIGIN_OTHER
        assert landing["suffix_tokens"] == 3
        assert bank.to_dict()["restore_landings"]["ram"]["exact"] == 1

    def test_ssd_sourced_prefix_restore_carries_the_split(self, monkeypatch):
        monkeypatch.setattr(gemma4, "_gemma4_prefill_prompt", _fake_prefill)
        runtime = _GemmaRuntime()
        stored = list(range(1, 1001))
        bank, entry = _gemma_bank(runtime, stored)
        # As _cold_near_prefix_candidate marks an entry read from the SSD.
        entry.cache_source = "ssd"
        entry.ssd_restore_s = 0.4
        entry.ssd_restore_split = {"read_s": 0.3, "copy_s": 0.05, "eval_s": 0.05, "bytes": 99}
        prompt = list(range(1, 701)) + [5000 + i for i in range(40)]
        state = _gemma_restore(runtime, bank, prompt, session_id="g1")
        stats = gemma4._gemma4_restore_stats(state)
        assert stats["ssd_cache_hit"] is True
        assert stats["ssd_restore_s"] == 0.4
        fields = openai._restore_telemetry_fields(stats)
        assert fields["ssd_restore_read_s"] == 0.3
        assert fields["ssd_restore_bytes"] == 99
        assert bank.to_dict()["restore_landings"]["ssd"]["restores"] == 1

    def test_cold_prefill_records_nothing(self, monkeypatch):
        monkeypatch.setattr(gemma4, "_gemma4_prefill_prompt", _fake_prefill)
        runtime = _GemmaRuntime()
        bank = SessionBank(max_entries=4, max_bytes=1 << 20, per_session_max_bytes=1 << 20)
        state = _gemma_restore(runtime, bank, list(range(1, 50)), session_id="g1")
        assert state.cache_hit is False
        assert state.restore_served == {}
        assert "ram" not in bank.to_dict()["restore_landings"]


class _BatchBank:
    checkpoint_budget_bytes = None

    def __init__(self, entry, matched, restore_point) -> None:
        self.entry = entry
        self.matched = matched
        self.restore_point = restore_point
        self.landings = []

    def near_prefix_candidates(self, tokens, **_kwargs):
        return [(self.entry, self.matched)]

    def restore_entry_prefix_cache(self, rt, entry, matched, **_kwargs):
        return ("cache", "history", "clone", self.restore_point, "hidden")

    def record_restore_landing(self, landing):
        self.landings.append(landing)


def _batch_hooks(bank, *, session_id="b1"):
    state = SimpleNamespace(
        runtime=SimpleNamespace(model_path=Path("models/qwen"), mtp_enabled=True),
        draft_head_identity="d",
        backend_descriptor=None,
    )
    lane = SimpleNamespace(
        prefill_request=SimpleNamespace(keywords={"target_cache_factory": lambda: []})
    )
    observability: dict = {}
    restore, _commit = openai._build_mtp_batch_session_hooks(
        state,
        prompt_ids=list(range(2000)),
        session_bank=bank,
        session_id=session_id,
        template_hash="t",
        policy_fingerprint="p",
        lane=lane,
        request_observability=observability,
    )
    return restore, observability


def _hybrid_entry(**extra):
    return SimpleNamespace(
        prefix_len=1800,
        session_id="other",
        has_recurrent=True,
        gdn_boundaries=[],
        hits=0,
        last_access_s=0.0,
        **extra,
    )


class TestBatchedMtpLane:
    def test_restore_hook_records_the_landing(self):
        bank = _BatchBank(_hybrid_entry(), matched=1500, restore_point=1280)
        restore, observability = _batch_hooks(bank)
        assert restore() is not None
        assert observability["mtp_batch_session_restore"]["hit"] is True
        assert bank.landings == [
            {
                "source": "ram",
                "partial": True,
                "hybrid": True,
                "matched_tokens": 1500,
                "restore_point_tokens": 1280,
                "checkpoint_gap_tokens": 220,
                "suffix_tokens": 720,
                "origin": ORIGIN_OTHER,
            }
        ]
        assert observability["prefix_restore_point_tokens"] == 1280
        assert observability["prefix_restore_checkpoint_gap_tokens"] == 220
        assert observability["prefix_restore_origin"] == ORIGIN_OTHER
        assert observability["ssd_cache_hit"] is False
        assert observability["ssd_restore_bytes"] == 0

    def test_ssd_entry_carries_the_split(self):
        entry = _hybrid_entry(
            cache_source="ssd",
            ssd_restore_s=1.5,
            ssd_restore_split={"read_s": 1.0, "copy_s": 0.2, "eval_s": 0.2, "other_s": 0.1, "bytes": 4096},
        )
        bank = _BatchBank(entry, matched=1500, restore_point=1280)
        restore, observability = _batch_hooks(bank)
        assert restore() is not None
        assert observability["ssd_cache_hit"] is True
        assert observability["ssd_cached_tokens"] == 1280
        assert observability["ssd_restore_s"] == 1.5
        assert observability["ssd_restore_read_s"] == 1.0
        assert observability["ssd_restore_bytes"] == 4096
        assert bank.landings[0]["source"] == "ssd"

    def test_telemetry_failure_keeps_the_restore(self, monkeypatch):
        bank = _BatchBank(_hybrid_entry(), matched=1500, restore_point=1280)

        def boom(*_args, **_kwargs):
            raise RuntimeError("telemetry")

        monkeypatch.setattr(openai, "_mtp_batch_restore_telemetry", boom)
        restore, observability = _batch_hooks(bank)
        assert restore() is not None
        assert observability["mtp_batch_session_restore"]["hit"] is True
