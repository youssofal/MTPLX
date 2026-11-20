"""#690: a conversation's newest state stays in a bank whose explicit budget it alone exceeds.

The report (M1 Max 32 GB, Qwen 3.6 27B, MTPLX_SESSION_BANK_MAX_BYTES=2G):
once a session passed about 11,000 tokens, one turn re-read the whole
context (67 to 106 s instead of about 1.3 s). An explicit bank budget leaves
the per-session cap at its RAM-tier default (8 GiB), so put() sized a 2.15
to 2.29 GB entry against 8 GiB: no boundary records were shed, the entry was
stored, its SSD write queued, and _evict_if_needed then evicted it at once
because it alone was over the 2 GiB bank. The queued write still held the
arrays (no memory came back) and the next request found nothing in RAM.

When the bank's budget is the tighter cap, an entry over it now sheds
boundary records, furthest from the tail first, until it fits, whatever
MTPLX_SESSION_BANK_SHED_BOUNDARIES says. An entry whose base state alone is
over the bank keeps the earlier path (stored, SSD write queued, evicted from
RAM), so a turn that comes after the write can still restore from SSD. When
the per-session cap is the tighter one, nothing changes.

Pure host tests: MLX arrays are byte-size carriers only, never evaluated.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.cache_state import CacheSnapshot
from mtplx.session_bank import SESSION_BANK_SHED_BOUNDARIES_ENV, SessionBank

RUNTIME = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)

# The report's sizes scaled down: a 2 GiB bank, an 8 GiB per-session cap, a
# base state of about 0.77 GB and boundary records of 147 MB each.
BANK = 400
PER_SESSION = 1_600
BASE = 256
RECORD = 64
FOUR = [(p, CacheSnapshot(states=(mx.zeros((RECORD,), mx.uint8),), meta_states=(None,)), None)
        for p in (1_000, 2_000, 3_000, 4_000)]


@pytest.fixture(autouse=True)
def _default_gate(monkeypatch):
    monkeypatch.delenv(SESSION_BANK_SHED_BOUNDARIES_ENV, raising=False)


class _ColdTier:
    """Accepts the SSD write; the jobs below are only queued, never run."""

    def put_entry(self, *args, **kwargs):  # pragma: no cover - never runs
        raise AssertionError("the queued write must not run in these tests")


def _bank(*, max_bytes: int = BANK, per_session: int = PER_SESSION):
    bank = SessionBank(
        max_entries=4, max_bytes=max_bytes, per_session_max_bytes=per_session, cold_tier=_ColdTier()
    )
    jobs: list = []
    bank.cold_enqueue_dispatch = jobs.append
    return bank, jobs


def _put(bank, *, boundaries, base=BASE, tokens=(1, 2, 3)):
    return bank.put(
        runtime=RUNTIME,
        token_ids=list(tokens),
        cache=[],
        logits=None,
        hidden=mx.zeros((base,), mx.uint8),
        session_id="s1",
        gdn_boundaries=list(boundaries),
    )


def test_the_newest_state_stays_restorable_when_it_alone_is_over_the_bank_budget():
    bank, jobs = _bank()
    entry = _put(bank, boundaries=FOUR)  # 256 + 4 x 64 = 512 > 400

    assert entry is not None
    assert len(bank) == 1, "the entry was evicted right after it was stored"
    assert bank.longest_prefix((1, 2, 3, 4)) is entry
    assert [record[0] for record in entry.gdn_boundaries] == [3_000, 4_000]
    assert entry.nbytes == BASE + 2 * RECORD <= BANK
    assert bank.total_nbytes <= bank.max_bytes
    assert [log["reason"] for log in bank.eviction_log] == ["shed_gdn_boundaries"]
    assert bank.eviction_log[-1]["budget"] == BANK
    assert len(jobs) == 1  # one SSD write, for the state the bank keeps


def test_a_state_whose_base_alone_is_over_the_bank_keeps_the_earlier_path():
    # Shedding cannot make it fit, so nothing changes: it is stored, its SSD
    # write queued (a turn after the write restores from SSD), and it leaves
    # RAM, as before.
    bank, jobs = _bank()
    _put(bank, boundaries=FOUR, base=BANK + 50)
    assert len(bank) == 0
    assert len(jobs) == 1
    assert bank.boundary_shed_puts == 0


def test_a_turn_later_the_conversation_continues_from_the_bank():
    # The shape of the report: each turn's state replaces the last one.
    bank, _jobs = _bank()
    _put(bank, boundaries=FOUR, tokens=(1, 2, 3))
    later = _put(bank, boundaries=FOUR, tokens=(1, 2, 3, 4, 5))
    assert later is not None
    assert bank.longest_prefix((1, 2, 3, 4, 5, 6)) is later
    assert bank.total_nbytes <= bank.max_bytes


def test_when_the_per_session_cap_is_tighter_nothing_changes():
    # The default (auto) budgets: per-session is two thirds of the bank, and
    # the boundary-shed gate (off) still refuses an entry over its session's
    # budget, exactly as before.
    bank, jobs = _bank(max_bytes=100_000, per_session=BANK)
    assert _put(bank, boundaries=FOUR) is None
    assert bank.eviction_log[-1]["reason"] == "skipped_oversized_snapshot"
    assert bank.eviction_log[-1]["budget"] == BANK
    assert jobs == []


# The report's own byte sizes: Qwen 3.6 27B, 69,632 B a token plus 153,944,064 B
# for the tail state and for each of eight GDN boundary records, a 2 GiB bank
# beside the 8 GiB per-session default, one entry per session. The next request
# arrives within a second, before the queued SSD write could run, and asks what
# the admission path asks: what would the restore read from RAM (restore_plan).
REPORT_PER_TOKEN = 69_632
REPORT_RECORD = 153_944_064


@pytest.mark.parametrize(
    "stored, following", [(11_012, 11_040), (12_990, 13_020)], ids=["2.15GB", "2.29GB"]
)
def test_the_next_request_restores_the_reported_entries_from_ram(monkeypatch, stored, following):
    monkeypatch.setenv("MTPLX_SESSION_BANK_PER_SESSION_MAX_ENTRIES", "1")
    bank, jobs = _bank(max_bytes=2 << 30, per_session=8 << 30)
    records = [
        (1_200 * (i + 1),
         CacheSnapshot(states=(mx.zeros((REPORT_RECORD,), mx.uint8),), meta_states=(None,)), None)
        for i in range(8)
    ]
    base = REPORT_PER_TOKEN * stored + REPORT_RECORD
    assert base + 8 * REPORT_RECORD > 2 << 30  # the reported entry is over the bank alone
    _put(bank, boundaries=records, base=base, tokens=range(stored))

    plan = bank.restore_plan(
        list(range(following)), model_path=str(RUNTIME.model_path), mtp_enabled=True
    )
    assert plan["mode"] != "none", "the next request would read the whole context again"
    assert plan["reuse_tokens"] == stored
    assert bank.total_nbytes <= bank.max_bytes
    assert len(jobs) == 1
