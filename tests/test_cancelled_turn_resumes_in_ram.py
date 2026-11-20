"""After a cancelled answer the next message resumes from RAM, where the
cancelled turn began.

Found on 12 November in a Pi session at 118K tokens: the client cancelled an
answer and sent its next message. That prompt leaves the cancelled one where
the last finished turn ended. The one-copy store held the conversation once,
as the cancelled prompt's lease, and a hybrid model resumes only at a
recurrent checkpoint: the lease had one at its own end (above the new
prompt's match) and older ones wherever the budget had kept them (an answer
or more below it). The finished turn was on SSD as a whole prefix of the new
prompt, so the turn read 118,096 tokens back (0.91 s, first text 1.83 s
against about 0.5 s) and held the conversation twice until the bank let the
lease go.

Two things were missing, and each has its tests here:

* the lease keeps the checkpoint where its prompt's prefill resumed the
  stored conversation (the copy the restore already takes for the lease's
  return), ahead of the grid when the budget is full;
* a candidate in RAM only has to reach a floor that stands for a spilled
  entry, where it had to beat it.

Host only: toy caches and runtimes, a real SessionBank and cold tier on
tmp_path. The same turn on a model: tests/test_one_copy_cancelled_answer.py.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx import generation
from mtplx.cache_bank import SessionBankColdTier
from mtplx.checkpoint_anchors import record_nbytes
from mtplx.one_copy import prompt_lease_fields
from mtplx.session_bank import SessionBank
from test_checkpoint_anchors import _KV, _Recurrent, _boundary
from test_checkpoint_anchors import _runtime as _recurrent_runtime
from test_near_lane_releases_refused import PROMPT, SPILLED, _near, _put, _WatchingModel
from test_near_lane_releases_refused import _runtime as _toy_runtime
from test_near_lane_spilled_exact_prefix import _SpilledBank, _tiny_runtime

# One turn of window flashw1 (16 November, Optimized Speed): the checkpoints
# its prefill left, eight of them, the budget's worth.
PROMPT_LEN = 88_208
PREFILL_CHECKPOINTS = (4096, 16384, 32768, 49152, 65536, 82580, 88143, 88207)
LEASE_POINT = 84_770  # where the turn before it ended


def _positions(records) -> list[int]:
    return [int(record[0]) for record in records]


def _lease_fields(**kwargs):
    return prompt_lease_fields(
        [_KV(PROMPT_LEN), _Recurrent(PROMPT_LEN)],
        committed_mtp_cache=None,
        hidden=None,
        prompt_len=PROMPT_LEN,
        boundaries=[_boundary(p) for p in PREFILL_CHECKPOINTS],
        **kwargs,
    )


def test_a_prompts_lease_carries_the_checkpoint_where_its_prefill_resumed():
    fields = _lease_fields(resumed_at=LEASE_POINT, lease_point=_boundary(LEASE_POINT))
    assert fields["resumed_at"] == LEASE_POINT
    assert _positions(fields["gdn_boundaries"]) == [
        *PREFILL_CHECKPOINTS, LEASE_POINT, PROMPT_LEN,
    ]
    # A cold prompt resumed nothing and carries no such checkpoint.
    cold = _lease_fields()
    assert cold["resumed_at"] == 0
    assert _positions(cold["gdn_boundaries"]) == [*PREFILL_CHECKPOINTS, PROMPT_LEN]


def _banked(**kwargs) -> list[int]:
    one = record_nbytes(_boundary(1))
    bank = SessionBank(checkpoint_budget_bytes=len(PREFILL_CHECKPOINTS) * one)
    entry = bank.put(
        runtime=_recurrent_runtime(),
        token_ids=list(range(PROMPT_LEN)),
        cache=[_KV(PROMPT_LEN), _Recurrent(PROMPT_LEN)],
        logits=mx.zeros((1, 4)),
        hidden=None,
        mtp_history_policy="cycle",
        gdn_boundaries=[
            _boundary(p) for p in (*PREFILL_CHECKPOINTS, LEASE_POINT, PROMPT_LEN)
        ],
        **kwargs,
    )
    return _positions(entry.gdn_boundaries)


def test_a_full_budget_keeps_the_point_the_prefill_resumed_at():
    kept = _banked(resumed_at=LEASE_POINT)
    assert LEASE_POINT in kept
    assert len(kept) == len(PREFILL_CHECKPOINTS)
    # Beside the anchor at the prompt's end and the one a little before it.
    assert PROMPT_LEN in kept and 88143 in kept
    # The budget is the same, so one grid checkpoint makes room, the finest.
    assert [p for p in PREFILL_CHECKPOINTS if p not in kept] == [16384, 88207]


def test_without_it_the_grid_takes_the_place():
    """What every prompt's lease kept before: a client that cancelled its
    answer had no checkpoint between 82,580 and the prompt's end."""
    kept = _banked()
    assert LEASE_POINT not in kept
    assert len(kept) == len(PREFILL_CHECKPOINTS)


# ---- a candidate in RAM that reaches the spilled entry's length is served ----

CANCELLED = SPILLED + [60, 61, 62, 63]  # the cancelled prompt, in RAM


@pytest.fixture
def cancelled_turn(tmp_path, monkeypatch):
    """Conversation A finished and on SSD only; A's cancelled prompt in RAM.
    Yields the runtime, the bank and the list of rows read back from SSD."""
    monkeypatch.setenv("MTPLX_SESSION_PREFIX_BLOCK_SIZE", "8")
    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS", "8")
    monkeypatch.setenv("MTPLX_SESSION_NEAR_PREFIX_MIN_MATCH_TOKENS", "4")
    rt = _toy_runtime(_WatchingModel())
    cold = SessionBankColdTier(
        base_dir=tmp_path / "session-bank", mode="on", min_prefix_tokens=2
    )
    try:
        bank = SessionBank(
            max_entries=8, max_bytes=4096, per_session_max_bytes=4096, cold_tier=cold
        )
        assert _put(bank, rt, SPILLED, "a") is not None
        assert cold.flush(timeout_s=5.0) is True
        bank.clear()
        assert _put(bank, rt, CANCELLED, "a") is not None
        read_back: list[int] = []
        restore_row = cold._restore_row

        def counted(*args, **kwargs):
            read_back.append(1)
            return restore_row(*args, **kwargs)

        monkeypatch.setattr(cold, "_restore_row", counted)
        yield rt, bank, read_back
    finally:
        cold.close()


def test_the_cancelled_prompts_entry_serves_the_next_message(cancelled_turn):
    rt, bank, read_back = cancelled_turn
    state = _near(
        rt, PROMPT, bank, session_id="a",
        min_restore_tokens=len(SPILLED), floor_on_ssd=True,
    )
    assert state is not None
    assert state.cached_tokens == len(SPILLED)
    assert state.suffix_tokens == len(PROMPT) - len(SPILLED)
    assert not state.ssd_cache_hit and state.cache_source == "ram"
    assert read_back == []


def test_a_floor_held_in_ram_still_has_to_be_beaten(cancelled_turn):
    """The exact lane serves a resident whole prefix with its logits and
    without a seed forward: a candidate that only reaches it stands aside."""
    rt, bank, read_back = cancelled_turn
    state = _near(
        rt, PROMPT, bank, session_id="a", min_restore_tokens=len(SPILLED)
    )
    assert state is None
    assert read_back == []


@pytest.mark.parametrize("ram_exact, floor, on_ssd", [(0, 41, True), (30, 30, False)])
def test_the_lane_is_told_whose_floor_it_has(monkeypatch, ram_exact, floor, on_ssd):
    """A whole prefix of 41 tokens on SSD, and one of 30 in RAM or none."""
    told: list[tuple[int, bool]] = []

    def near(*_args, **kwargs):
        told.append((kwargs["min_restore_tokens"], kwargs.get("floor_on_ssd", False)))
        return None

    monkeypatch.setattr(generation, "_restore_near_prefix_prompt_state", near)
    bank = _SpilledBank(ram_exact=ram_exact, ssd_exact=41)
    generation.restore_or_prefill_prompt_state(
        _tiny_runtime(), list(PROMPT), session_bank=bank, session_id="a"
    )
    assert told[0] == (floor, on_ssd)
