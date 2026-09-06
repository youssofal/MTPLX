"""Boundary shedding on admission (MTPLX_SESSION_BANK_SHED_BOUNDARIES).

The bug these tests pin, from PR #391 receipts/ttft/control.json
(2026-09-01, Qwen3.8 Flash-Next, three repeats):

    cold                 19,022-tok prompt   15.47 s visible TTFT
    matching_terminal    exact restore        0.217 s
    rerendered_terminal  cached=0, cold      15.79 s   <-- 70x cliff

The bank had auto-sized to its 1 GiB floor ("model weights 107.1G"). A 19K
entry's base snapshot is ~711 MB and each GDN boundary record ~87-101 MB, so
the 8-record payload pushed every boundary-CARRYING put over the per-session
cap: eviction_log shows skipped_oversized_snapshot at 1,398,321,776 and
1,520,850,304 bytes against a 1,073,741,824 budget, while the same turn's
boundary-LESS commit (710,255,120) was admitted. The one surviving entry
reports gdn_boundaries: []. recurrent_boundary_at_or_below() then returns None,
the near-prefix lane rejects every candidate, and the request falls through to
a cold prefill.

Pure host tests. MLX arrays are used ONLY as byte-size carriers -- 64 to 512
bytes, constructed and never evaluated, so nothing reaches Metal.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.cache_state import CacheSnapshot
from mtplx.session_bank import (
    SESSION_BANK_SHED_BOUNDARIES_ENV,
    SessionBank,
)


RUNTIME = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)

#: One boundary record's payload, in bytes. Scaled down from the receipt's
#: ~87-101 MB by ~1.5e6 so the arithmetic is identical and the tensors are not.
BOUNDARY_BYTES = 64
#: The entry's base snapshot. The receipt's ~711 MB, same scaling.
BASE_BYTES = 256
#: The per-session cap. The receipt's 1 GiB, same scaling: base fits, base plus
#: the full 4-record payload does not, base plus 2 records does.
BUDGET = 400


def _bytes(n: int):
    """A never-evaluated MLX array of exactly ``n`` bytes."""

    return mx.zeros((int(n),), mx.uint8)


def _boundary(position: int, *, size: int = BOUNDARY_BYTES):
    return (int(position), CacheSnapshot(states=(_bytes(size),), meta_states=(None,)), None)


@pytest.fixture(autouse=True)
def _no_inherited_gate(monkeypatch):
    monkeypatch.delenv(SESSION_BANK_SHED_BOUNDARIES_ENV, raising=False)


def _bank(**kwargs) -> SessionBank:
    defaults = dict(max_entries=16, max_bytes=100_000, per_session_max_bytes=BUDGET)
    defaults.update(kwargs)
    return SessionBank(**defaults)


def _put(bank: SessionBank, *, boundaries, base: int = BASE_BYTES, tokens=(1, 2, 3), cache=None):
    return bank.put(
        runtime=RUNTIME,
        token_ids=list(tokens),
        cache=[] if cache is None else cache,
        logits=None,
        hidden=_bytes(base),
        session_id="s1",
        gdn_boundaries=list(boundaries),
    )


FOUR = [_boundary(p) for p in (1_000, 2_000, 3_000, 4_000)]


# --------------------------------------------------------------------------
# Gate
# --------------------------------------------------------------------------


def test_gate_defaults_off():
    assert _bank().shed_gdn_boundaries_to_fit is False


@pytest.mark.parametrize("value", ["1", "true", "YES", "On"])
def test_gate_accepts_truthy(monkeypatch, value):
    monkeypatch.setenv(SESSION_BANK_SHED_BOUNDARIES_ENV, value)
    assert _bank().shed_gdn_boundaries_to_fit is True


@pytest.mark.parametrize("value", ["0", "", "off", "nope"])
def test_gate_rejects_everything_else(monkeypatch, value):
    monkeypatch.setenv(SESSION_BANK_SHED_BOUNDARIES_ENV, value)
    assert _bank().shed_gdn_boundaries_to_fit is False


def test_gate_is_construction_time(monkeypatch):
    monkeypatch.setenv(SESSION_BANK_SHED_BOUNDARIES_ENV, "1")
    bank = _bank()
    monkeypatch.delenv(SESSION_BANK_SHED_BOUNDARIES_ENV)
    assert bank.shed_gdn_boundaries_to_fit is True
    assert _bank().shed_gdn_boundaries_to_fit is False


# --------------------------------------------------------------------------
# The control: today's behaviour, reproduced
# --------------------------------------------------------------------------


def test_control_refuses_the_whole_entry_over_its_boundary_payload():
    """Base 256 <= budget 400, yet the entry is dropped. This is the bug."""

    bank = _bank()
    assert _put(bank, boundaries=FOUR) is None
    assert len(bank) == 0
    assert bank.last_put_skipped_oversized_snapshot is True
    assert bank.last_put_nbytes == BASE_BYTES + 4 * BOUNDARY_BYTES
    assert bank.eviction_log[-1]["reason"] == "skipped_oversized_snapshot"
    assert bank.boundary_shed_puts == 0


def test_control_admits_the_same_entry_with_no_boundaries():
    """The asymmetry the receipt shows: boundary-LESS commits get in."""

    bank = _bank()
    entry = _put(bank, boundaries=[])
    assert entry is not None
    assert entry.gdn_boundaries == []
    assert entry.recurrent_boundary_at_or_below(3_500) is None


# --------------------------------------------------------------------------
# The fix
# --------------------------------------------------------------------------


def test_shed_admits_the_entry_with_the_tail_nearest_boundaries(monkeypatch):
    monkeypatch.setenv(SESSION_BANK_SHED_BOUNDARIES_ENV, "1")
    bank = _bank()

    entry = _put(bank, boundaries=FOUR)

    assert entry is not None
    assert [record[0] for record in entry.gdn_boundaries] == [3_000, 4_000]
    assert entry.nbytes == BASE_BYTES + 2 * BOUNDARY_BYTES
    assert entry.nbytes <= BUDGET
    assert bank.boundary_shed_puts == 1
    assert bank.boundary_shed_records == 2
    shed_log = bank.eviction_log[-1]
    assert shed_log["reason"] == "shed_gdn_boundaries"
    assert shed_log["boundaries_shed"] == 2
    assert shed_log["boundaries_kept"] == 2
    assert shed_log["budget"] == BUDGET


def test_shed_restores_the_capability_the_near_prefix_lane_needs(monkeypatch):
    """The whole point: recurrent_boundary_at_or_below stops returning None.

    That probe returning None is what makes _restore_near_prefix_prompt_state
    log `boundary_not_better:0` and skip the candidate, which is what sent the
    re-rendered turn to a 15.79 s cold prefill.
    """

    monkeypatch.setenv(SESSION_BANK_SHED_BOUNDARIES_ENV, "1")
    bank = _bank()
    entry = _put(bank, boundaries=FOUR)

    assert entry is not None
    assert entry.recurrent_boundary_at_or_below(3_500)[0] == 3_000
    assert entry.recurrent_boundary_at_or_below(4_000)[0] == 4_000
    # Deep divergence is the documented cost of shedding: no anchor survives,
    # so it fails closed to cold -- exactly what it does today, never worse.
    assert entry.recurrent_boundary_at_or_below(1_500) is None


def test_shed_keeps_ascending_order_for_the_entry():
    bank = _bank()
    kept, nbytes, shed = bank._shed_boundaries_to_fit(
        FOUR, BASE_BYTES + 4 * BOUNDARY_BYTES
    )
    assert [record[0] for record in kept] == [3_000, 4_000]
    assert nbytes == BASE_BYTES + 2 * BOUNDARY_BYTES
    assert shed == 2


def test_shed_is_a_no_op_when_the_entry_already_fits():
    bank = _bank()
    two = FOUR[:2]
    kept, nbytes, shed = bank._shed_boundaries_to_fit(
        two, BASE_BYTES + 2 * BOUNDARY_BYTES
    )
    assert kept == two
    assert shed == 0
    assert nbytes == BASE_BYTES + 2 * BOUNDARY_BYTES


def test_shed_stops_at_zero_records_and_does_not_loop():
    bank = _bank()
    kept, nbytes, shed = bank._shed_boundaries_to_fit(FOUR, 10_000)
    assert kept == []
    assert shed == 4
    assert nbytes == 10_000 - 4 * BOUNDARY_BYTES


def test_an_oversized_base_snapshot_is_still_refused(monkeypatch):
    """Shedding rescues entries whose PAYLOAD is the problem, nothing else."""

    monkeypatch.setenv(SESSION_BANK_SHED_BOUNDARIES_ENV, "1")
    bank = _bank()

    assert _put(bank, boundaries=FOUR, base=BUDGET + 64) is None
    assert len(bank) == 0
    assert bank.last_put_skipped_oversized_snapshot is True
    # A shed that cannot rescue the entry is not applied and not counted, so
    # boundary_shed_puts stays an honest "entries rescued" counter.
    assert bank.boundary_shed_puts == 0
    assert bank.eviction_log[-1]["reason"] == "skipped_oversized_snapshot"
    assert bank.last_put_nbytes == BUDGET + 64 + 4 * BOUNDARY_BYTES


def test_nbytes_override_is_never_shed(monkeypatch):
    """A caller that pinned the size owns it; shedding would falsify it."""

    monkeypatch.setenv(SESSION_BANK_SHED_BOUNDARIES_ENV, "1")
    bank = _bank()
    entry = bank.put(
        runtime=RUNTIME,
        token_ids=[1, 2, 3],
        cache=[],
        logits=None,
        hidden=_bytes(BASE_BYTES),
        session_id="s1",
        gdn_boundaries=list(FOUR),
        nbytes_override=BUDGET + 1,
    )
    assert entry is None
    assert bank.boundary_shed_puts == 0


def test_entries_that_fit_are_untouched_under_the_flag(monkeypatch):
    monkeypatch.setenv(SESSION_BANK_SHED_BOUNDARIES_ENV, "1")
    bank = _bank()
    two = FOUR[:2]
    entry = _put(bank, boundaries=two)
    assert entry is not None
    assert [record[0] for record in entry.gdn_boundaries] == [1_000, 2_000]
    assert bank.boundary_shed_puts == 0


# --------------------------------------------------------------------------
# The downstream consequence the receipt was missing
# --------------------------------------------------------------------------


def test_a_boundary_carrying_container_supersedes_its_prefixes(monkeypatch):
    """Why one shed put is enough under a one-entry budget.

    _supersede_contained_prefixes returns early for a boundary-LESS recurrent
    container ("without recurrent boundaries the container cannot serve
    sub-prefix restores"), so today's bank accumulates redundant boundary-less
    entries and the byte cap then evicts by LRU. Once put admits a
    boundary-carrying entry, put's own prefix_donor inheritance carries the
    records onto the longer commit and the pair collapses to one entry that
    can still serve sub-prefix restores.
    """

    monkeypatch.setenv(SESSION_BANK_SHED_BOUNDARIES_ENV, "1")
    bank = _bank(per_session_max_bytes=BUDGET, max_bytes=100_000)

    short = _put(bank, boundaries=FOUR, tokens=tuple(range(4_000)))
    assert short is not None
    assert [record[0] for record in short.gdn_boundaries] == [3_000, 4_000]

    # The generation-final commit passes NO boundaries, exactly as the receipt
    # shows; put inherits them from the longest banked prefix.
    longer = _put(bank, boundaries=[], tokens=tuple(range(5_000)))
    assert longer is not None
    assert [record[0] for record in longer.gdn_boundaries] == [3_000, 4_000]

    # The container dominates, so the shorter donor is superseded away and the
    # single surviving entry still answers the boundary probe.
    assert set(bank._entries) == {tuple(range(5_000))}
    assert longer.recurrent_boundary_at_or_below(3_500)[0] == 3_000


def test_longer_recurrent_entry_does_not_erase_uncovered_exact_prefix(monkeypatch):
    monkeypatch.setenv(SESSION_BANK_SHED_BOUNDARIES_ENV, "1")
    bank = _bank(per_session_max_bytes=10_000)
    recurrent = [SimpleNamespace(state=None, meta_state=None, is_trimmable=lambda: False)]
    short = _put(bank, boundaries=[_boundary(2_048)], tokens=range(15_688), cache=recurrent)
    longer = _put(bank, boundaries=[_boundary(2_048), _boundary(30_720)], tokens=range(32_870), cache=recurrent)
    assert short is not None and longer is not None
    next_prompt = [*range(15_702), -1]
    assert bank.longest_prefix(next_prompt) is short
    assert longer.recurrent_boundary_at_or_below(15_702)[0] == 2_048


def test_to_dict_publishes_the_engagement_receipt(monkeypatch):
    monkeypatch.setenv(SESSION_BANK_SHED_BOUNDARIES_ENV, "1")
    bank = _bank()
    _put(bank, boundaries=FOUR)
    snapshot = bank.to_dict()
    assert snapshot["shed_gdn_boundaries_to_fit"] is True
    assert snapshot["boundary_shed_puts"] == 1
    assert snapshot["boundary_shed_records"] == 2


def test_to_dict_receipt_defaults_off():
    snapshot = _bank().to_dict()
    assert snapshot["shed_gdn_boundaries_to_fit"] is False
    assert snapshot["boundary_shed_puts"] == 0
