"""The RAM-lane refusal is reported next to the cold tier's miss.

When the near/block-prefix lane passed over every stored entry, the request
fell through to ``SessionBank.restore()`` and its SSD lookup; health then read
``last_miss_reason: ssd_prefix_miss``. That is true (the SSD had nothing), but
it hid why RAM could not serve the shared prompt start: a shared prefix below
the block minimum, a hybrid entry without GDN boundaries, or a candidate the
generation lane refused (``boundary_not_better``). Those reasons now land in
``last_prefix_diagnostic["ram_miss_reason"]`` and ``last_ram_miss_reason``.

Pure host tests: no model, no Metal.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx import generation as g
from mtplx.session_bank import SessionBank, block_prefix_skip_reason

RUNTIME = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
SHARED = list(range(1000, 1220))  # a 220-token shared prompt start


def _bank(**kwargs) -> SessionBank:
    defaults = {"max_entries": 8, "max_bytes": 1 << 20, "per_session_max_bytes": 1 << 20}
    defaults.update(kwargs)
    return SessionBank(**defaults)


def _put(bank: SessionBank, tokens: list[int], *, recurrent: bool = False):
    entry = bank.put(
        runtime=RUNTIME,
        token_ids=tokens,
        cache=[],
        logits=None,
        hidden=None,
        session_id="s1",
        nbytes_override=16,
    )
    assert entry is not None
    # A hybrid (GDN) entry stored without any boundary records.
    entry.has_recurrent = recurrent
    return entry


def _candidates(bank: SessionBank, prompt: list[int], **kwargs):
    defaults = {"block_size": 256, "block_min_matched_tokens": 512}
    defaults.update(kwargs)
    return bank.near_prefix_candidates(prompt, **defaults)


def _ram_reason(bank: SessionBank) -> str | None:
    assert bank.last_prefix_diagnostic is not None
    reason = bank.last_prefix_diagnostic.get("ram_miss_reason")
    assert bank.to_dict()["last_ram_miss_reason"] == reason
    return reason


# --- bank: why the block-prefix lane passed over an entry -------------------


def test_shared_prefix_below_the_block_minimum_names_the_threshold():
    bank = _bank()
    _put(bank, SHARED + list(range(80)))

    assert _candidates(bank, SHARED + [7] * 100) == []
    assert _ram_reason(bank) == "below_block_min_match:512"
    assert bank.last_prefix_diagnostic["common_prefix_tokens"] == 220


def test_threshold_is_reported_after_the_silent_raise_to_the_block_size():
    # MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS=128 is raised to the block
    # size (256); the reason shows the minimum that actually applied.
    bank = _bank()
    _put(bank, SHARED + list(range(80)))

    assert _candidates(bank, SHARED + [7] * 100, block_min_matched_tokens=128) == []
    assert _ram_reason(bank) == "below_block_min_match:256"


def test_hybrid_entry_without_gdn_boundaries_is_named():
    # 700 shared tokens clear a 600 minimum, but without boundaries the
    # restore point rounds down to the 512 block edge.
    shared = list(range(5000, 5700))
    bank = _bank()
    _put(bank, shared + list(range(300)), recurrent=True)

    assert _candidates(bank, shared + [7] * 50, block_min_matched_tokens=600) == []
    assert _ram_reason(bank) == "no_gdn_boundaries"


def test_disabled_block_prefix_lane_is_named():
    shared = list(range(5000, 5700))
    bank = _bank()
    _put(bank, shared + list(range(300)))

    assert _candidates(bank, shared + [7] * 50, allow_block_prefix=False) == []
    assert _ram_reason(bank) == "block_prefix_disabled"


def test_no_shared_prefix_and_empty_bank_keep_the_generic_reasons():
    bank = _bank()
    assert _candidates(bank, [1, 2, 3]) == []
    assert _ram_reason(bank) == "new_session"

    _put(bank, SHARED)
    assert _candidates(bank, [1, 2, 3]) == []
    assert _ram_reason(bank) == "prefix_divergence_at_token"


def test_a_served_candidate_carries_no_ram_miss_reason():
    shared = list(range(5000, 5700))
    bank = _bank()
    _put(bank, shared + list(range(300)))

    assert _candidates(bank, shared + [7] * 50)
    assert _ram_reason(bank) is None


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"matched_tokens": 0, "block_prefix_allowed": False, "exact_capable": True},
         "prefix_divergence_at_token"),
        ({"matched_tokens": 600, "block_prefix_allowed": False, "exact_capable": True},
         "block_prefix_disabled"),
        ({"matched_tokens": 600, "block_prefix_allowed": True, "exact_capable": False},
         "no_gdn_boundaries"),
        ({"matched_tokens": 300, "block_prefix_allowed": True, "exact_capable": False},
         "below_block_min_match:512"),
        ({"matched_tokens": 300, "block_prefix_allowed": True, "exact_capable": True},
         "below_block_min_match:512"),
    ],
)
def test_block_prefix_skip_reason(kwargs, expected):
    assert block_prefix_skip_reason(block_min_matched_tokens=512, **kwargs) == expected


# --- the SSD miss that follows no longer hides the RAM reason ---------------


class _MissingColdTier:
    last_miss_reason = "ssd_prefix_miss"

    def lookup(self, *_args, **_kwargs):
        return None

    def stats(self):
        return {"enabled": True, "last_miss_reason": self.last_miss_reason}


def test_ssd_prefix_miss_stays_the_cold_reason_and_the_ram_reason_survives():
    bank = _bank(cold_tier=_MissingColdTier())
    _put(bank, SHARED + list(range(80)))
    prompt = SHARED + [7] * 100

    assert _candidates(bank, prompt) == []
    assert bank.restore(RUNTIME, prompt, session_id="s1") is None

    health = bank.to_dict()
    assert health["last_miss_reason"] == "ssd_prefix_miss"
    assert health["last_ram_miss_reason"] == "below_block_min_match:512"


# --- generation: candidates the near-prefix lane refused --------------------


class _CandidateBank:
    """Duck-typed bank that offers fixed candidates, best first."""

    def __init__(self, candidates):
        self._candidates = candidates
        self.last_prefix_diagnostic = {"miss_reason": None}

    def near_prefix_candidates(self, prompt_ids, **_kwargs):
        return list(self._candidates)


def _entry(**overrides):
    fields = {
        "model_path": str(RUNTIME.model_path),
        "prefix_len": 1200,
        "hidden_variant": "b",
        "mtp_history_policy": "cycle",
        "has_recurrent": False,
        "mtp_history_snapshot": None,
        "mtp_snapshot_epoch": None,
        "snapshot_epoch": 0,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _near_lane(bank, *, prompt_len: int = 1100):
    return g._restore_near_prefix_prompt_state(
        RUNTIME,
        list(range(prompt_len)),
        base_hidden_variant="b",
        mtp_hidden_variant="m",
        mtp_history_policy="cycle",
        session_bank=bank,
        template_hash=None,
        draft_head_identity=None,
        policy_fingerprint=None,
    )


def test_boundary_not_better_is_recorded():
    no_boundary = _entry(
        has_recurrent=True, recurrent_boundary_at_or_below=lambda matched: None
    )
    bank = _CandidateBank([(no_boundary, 1000)])

    assert _near_lane(bank) is None
    assert bank.last_prefix_diagnostic["ram_miss_reason"] == "boundary_not_better:0"


def test_identity_mismatch_is_recorded():
    bank = _CandidateBank([(_entry(model_path="models/other"), 1000)])

    assert _near_lane(bank) is None
    assert bank.last_prefix_diagnostic["ram_miss_reason"] == "identity_mismatch"


def test_the_best_candidate_refusal_wins():
    best = _entry(has_recurrent=True, recurrent_boundary_at_or_below=lambda m: None)
    runner_up = _entry(model_path="models/other")
    bank = _CandidateBank([(best, 1000), (runner_up, 900)])

    assert _near_lane(bank) is None
    assert bank.last_prefix_diagnostic["ram_miss_reason"] == "boundary_not_better:0"


def test_no_candidates_leave_the_bank_reason_untouched():
    bank = _CandidateBank([])
    bank.last_prefix_diagnostic["ram_miss_reason"] = "below_block_min_match:512"

    assert _near_lane(bank) is None
    assert bank.last_prefix_diagnostic["ram_miss_reason"] == "below_block_min_match:512"


def test_duck_typed_bank_without_diagnostic_is_tolerated():
    bank = _CandidateBank([(_entry(model_path="models/other"), 1000)])
    del bank.last_prefix_diagnostic

    assert _near_lane(bank) is None
