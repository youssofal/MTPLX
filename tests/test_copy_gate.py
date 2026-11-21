"""The copy-round cost gate: it only vetoes or shortens, it prices rounds from the request's
own clock, it lets everything through until it has measured both kinds of round, and on
recorded requests it stops the rounds that lose and keeps the ones that pay.

Recorded figures used below (10 November 2026, M5 Max, Flash-Next Optimized Speed, 2.12.3):
an ordinary round 32.4 ms committing about 3 tokens, an eager copy round about 9 ms a verify
row (81 ms for the 9-row probation window). The cold 16K ladder request ran 9 copy rounds
that drafted 72 tokens and landed 7, in 6 rounds that landed at least one. The request log
keeps totals only, so the order of rounds inside a request is reconstructed. The clock is
a fake one here; `tests/test_copy_gate_generation.py` runs the gate inside a generation.
"""

from __future__ import annotations

import itertools
import json

import pytest

from mtplx import copy_gate
from mtplx.copy_gate import LADDER, CopyGate, CostMeter, RoundCosts

EAGER = RoundCosts(ordinary_s=0.0324, copy_fixed_s=0.0, copy_row_s=0.009)
# A compiled window priced at an ordinary 4-row round plus 3 ms for each further row.
COMPILED = RoundCosts(ordinary_s=0.0324, copy_fixed_s=0.0204, copy_row_s=0.003)


class Session:
    """Drives a gate the way the copy lane does, on a clock that advances by the true costs."""

    def __init__(self, monkeypatch, true: RoundCosts, gate: CopyGate | None = None, ordinary_tokens: int = 3):
        self.now = 100.0
        monkeypatch.setattr(copy_gate.time, "perf_counter", lambda: self.now)
        self.true, self.gate = true, gate or CopyGate()
        self.ordinary_tokens = ordinary_tokens
        self.tokens = self.rounds = 0
        self.copies: list[tuple[int, int]] = []
        self.decisions: list[int] = []

    def round(self, offer: tuple[int, int] | None = None) -> int:
        """One decode cycle. `offer` = (block length the lane offers, tokens that would land)."""
        self.gate.cycle(self.tokens, self.rounds)
        length = 0
        if offer is not None:
            length = self.gate.plan(offer[0])
            self.decisions.append(length)
        if length:
            landed = min(offer[1], length)
            self.gate.observe_copy(length, landed)
            self.copies.append((length, landed))
            self.now += self.true.copy_round_s(length + 1)
            self.tokens += landed + 1
        else:
            self.now += self.true.ordinary_s
            self.tokens += self.ordinary_tokens
        self.rounds += 1
        return length

    def run(self, rounds: int, offers: dict[int, tuple[int, int]]) -> "Session":
        for index in range(rounds):
            self.round(offers.get(index))
        return self

    @property
    def copy_s(self) -> float:
        return sum(self.true.copy_round_s(length + 1) for length, _ in self.copies)


def warm(session: Session) -> Session:
    """Enough rounds of both kinds for the gate to hold measured costs: the first of each kind is
    dropped, then two of each are timed."""
    for _ in range(3):
        session.round()
        assert session.round((8, 8)) == 8
    session.round()
    assert session.gate.costs() is not None
    return session


def test_off_by_default_and_pins_parse(monkeypatch):
    for name in ("MTPLX_CONTEXT_COPY_GATE", "MTPLX_CONTEXT_COPY_GATE_COSTS", "MTPLX_CONTEXT_COPY_GATE_LOG"):
        monkeypatch.delenv(name, raising=False)
    assert copy_gate.for_request() is None
    monkeypatch.setenv("MTPLX_CONTEXT_COPY_GATE", "1")
    gate = copy_gate.for_request()
    assert isinstance(gate, CopyGate) and gate.pinned is None and gate.costs() is None
    monkeypatch.setenv("MTPLX_CONTEXT_COPY_GATE_COSTS", "32.4,0,9")
    pinned = copy_gate.for_request().pinned
    assert (pinned.ordinary_s, pinned.copy_fixed_s, pinned.copy_row_s) == pytest.approx((0.0324, 0.0, 0.009))
    monkeypatch.setenv("MTPLX_CONTEXT_COPY_GATE_COSTS", "nonsense")
    assert copy_gate.for_request().pinned is None


def test_prior_starts_where_the_lane_starts():
    gate = CopyGate()
    # The lane's acceptance average starts at half a probation block.
    assert copy_gate._expected_landed(gate.prior_survival, LADDER[0]) == pytest.approx(4.0, abs=1e-6)
    assert gate.survival() == pytest.approx(gate.prior_survival, abs=1e-9)


def test_cold_start_lets_every_block_through_until_both_kinds_are_timed(monkeypatch):
    session = Session(monkeypatch, EAGER)
    # Copy rounds that land nothing, at every other cycle: the worst case for letting them through.
    for index in range(6):
        session.round((24, 0) if index % 2 else None)
    assert session.decisions == [24, 24, 24]            # nothing vetoed, nothing shortened
    assert session.gate.counts["passed_cold"] == 3 and session.gate.costs() is None
    session.round()                                      # closes the third copy round: two are timed now
    costs = session.gate.costs()
    assert costs.ordinary_s == pytest.approx(0.0324) and costs.copy_round_s(25) == pytest.approx(0.225)
    assert session.round((24, 0)) == 0                   # from here the request's own costs decide
    assert session.gate.counts["vetoed_rate"] == 1


def test_a_short_answer_is_the_ungated_lane(monkeypatch):
    # Fewer than three copy rounds: every decision is the lane's own, whatever lands.
    for landing in itertools.product((0, 8), repeat=2):
        session = Session(monkeypatch, EAGER)
        offers = {2: (8, landing[0]), 5: (8, landing[1])}
        session.run(12, offers)
        assert session.decisions == [8, 8]
        assert session.gate.counts == {"passed_cold": 2, "taken_rate": 0, "taken_streak": 0, "trial": 0, "vetoed_rate": 0, "shortened": 0}


@pytest.mark.parametrize("costs", [EAGER, COMPILED])
def test_it_only_vetoes_or_shortens(monkeypatch, costs):
    histories = ((), ((8, 0),), ((8, 8),), ((8, 8), (24, 24), (24, 2)))
    for offered, history in itertools.product((1, 3, 8, 9, 12, 16, 20, 24, 32), histories):
        session = Session(monkeypatch, costs, CopyGate(pinned=costs))
        session.run(3, {})
        for drafted, accepted in history:
            session.gate.observe_copy(drafted, accepted)
        for _ in range(3):
            length = session.round((offered, offered))
            assert length == 0 or length == offered or (length in LADDER and length < offered)
    assert CopyGate(pinned=costs).plan(0) == 0


def test_cold_16k_request_stops_paying_for_rounds_that_land_nothing(monkeypatch):
    # 9 probation blocks offered across 162 rounds; they landed 2, 1, 1, 1, 1, 1, 0, 0, 0 tokens.
    landed = (2, 1, 1, 1, 1, 1, 0, 0, 0)
    offers = {17 * (i + 1): (8, n) for i, n in enumerate(landed)}
    session = Session(monkeypatch, EAGER).run(162, offers)
    assert session.decisions[:3] == [8, 8, 8]            # cold start: as the lane runs today
    assert len(session.copies) <= 4                      # then at most one spaced trial
    assert session.gate.counts["vetoed_rate"] >= 5
    today_s = 9 * EAGER.copy_round_s(9)
    # Each copy round not run is made up by ordinary rounds at about 3 tokens for 32.4 ms.
    made_up = (9 - len(session.copies)) * (1 + 7 / 9) / 3.0 * EAGER.ordinary_s
    assert today_s - session.copy_s - made_up > 0.25     # of a 5.77 s decode


def test_file_rewrite_keeps_copying_when_windows_are_compiled(monkeypatch):
    session = warm(Session(monkeypatch, COMPILED))
    blocks = ((24, 24), (24, 24), (24, 15), (16, 2), (8, 0))
    before = len(session.copies)
    for block in blocks:
        session.round()
        session.round(block)
    taken = session.copies[before:]
    assert len(taken) >= 4 and sum(landed for _, landed in taken) >= 30
    assert session.gate.counts["taken_rate"] + session.gate.counts["taken_streak"] >= 4
    # The streak stepped up to time a wider window, and two widths give the cost line back.
    assert 13 in session.gate.meter._copy
    costs = session.gate.costs()
    assert (costs.copy_fixed_s, costs.copy_row_s) == pytest.approx((0.0204, 0.003), abs=1e-6)


def test_eager_streak_runs_while_each_round_pays_and_stops_when_one_does_not(monkeypatch):
    session = Session(monkeypatch, EAGER, CopyGate(pinned=EAGER))
    session.run(3, {})
    for _ in range(200):                       # the estimate says no, so a spaced trial comes first
        if session.round((8, 8)):
            break
    assert session.gate.counts["trial"] == 1 and session.gate.counts["vetoed_rate"] > 50
    assert session.round((8, 8)) == 8          # 9 tokens for 81 ms beats 3 for 32.4 ms: the streak goes on
    assert session.round((8, 1)) == 8
    assert session.gate.counts["taken_streak"] == 2
    assert session.round((8, 8)) == 0          # 2 tokens for 81 ms did not pay: back to the estimate


def test_trials_cost_a_bounded_share_when_nothing_lands(monkeypatch):
    session = Session(monkeypatch, EAGER, CopyGate(pinned=EAGER))
    session.run(3, {})
    session.run(5000, {i: (8, 0) for i in range(5000)})
    ordinary_s = (5003 - len(session.copies)) * EAGER.ordinary_s
    assert 1 <= len(session.copies) <= 8
    assert session.copy_s / (ordinary_s + session.copy_s) < 0.01
    assert session.gate.counts["trial"] == len(session.copies)


def test_compiled_windows_pick_the_length_that_pays_most(monkeypatch):
    session = Session(monkeypatch, COMPILED, CopyGate(pinned=COMPILED))
    session.run(3, {})
    assert session.round((24, 24)) == 8        # nothing proven yet: the short block has the best rate
    assert session.gate.counts["shortened"] == 1
    for _ in range(6):
        session.round((24, 24))
    assert session.decisions[-1] == 24         # a block that keeps landing whole is taken whole


def test_meter_drops_the_first_of_each_kind_and_fits_the_line():
    meter = CostMeter()
    meter.observe_ordinary(9.0)                # the first round carries one-off work
    meter.observe_copy(9, 9.0)
    for _ in range(3):
        meter.observe_ordinary(0.0324)
    assert meter.snapshot() is None            # no copy round timed yet
    meter.observe_copy(9, 0.0204 + 0.003 * 9)
    assert meter.snapshot() is None
    meter.observe_copy(9, 0.0204 + 0.003 * 9)
    one_width = meter.snapshot()               # one width: a row is priced at the mean
    assert one_width.copy_fixed_s == 0.0 and one_width.copy_round_s(9) == pytest.approx(0.0474)
    for rows in (13, 17, 25):
        meter.observe_copy(rows, 0.0204 + 0.003 * rows)
    costs = meter.snapshot()
    assert costs.ordinary_s == pytest.approx(0.0324)
    assert costs.copy_fixed_s == pytest.approx(0.0204, abs=1e-6) and costs.copy_row_s == pytest.approx(0.003, abs=1e-7)
    meter.observe_ordinary(5.0)                # a stall is not a round
    assert meter.snapshot().ordinary_s == pytest.approx(0.0324)


def test_cycle_counts_every_round_and_times_only_single_rounds(monkeypatch):
    session = Session(monkeypatch, EAGER)
    gate = session.gate
    session.run(4, {})
    gate.cycle(session.tokens, session.rounds)
    assert (gate._ordinary_rounds, gate._ordinary_tokens, gate.meter._ordinary_n) == (4, 12.0, 3)
    # The lane was suspended for five rounds: counted, not timed.
    session.now += 5 * EAGER.ordinary_s
    session.tokens += 15
    session.rounds += 5
    gate.cycle(session.tokens, session.rounds)
    assert (gate._ordinary_rounds, gate._ordinary_tokens, gate.meter._ordinary_n) == (9, 27.0, 3)
    # A copy round followed by a suspended stretch: its tokens are not ordinary tokens.
    gate.observe_copy(8, 2)
    session.tokens += 3 + 6
    session.rounds += 1 + 2
    gate.cycle(session.tokens, session.rounds)
    assert (gate._ordinary_rounds, gate._ordinary_tokens, gate._since_copy) == (11, 33.0, 2)
    assert gate.meter._copy == {}


def test_log_has_a_line_for_every_decision_with_its_reason(tmp_path, monkeypatch):
    log = tmp_path / "gate.jsonl"
    session = Session(monkeypatch, EAGER, CopyGate(pinned=EAGER, log_path=str(log)))
    session.run(3, {})
    session.round((8, 0))
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(line["ev"], line.get("why"), line.get("length")) for line in lines] == [("plan", "vetoed_rate", 0)]
    assert lines[0]["counts"]["vetoed_rate"] == 1 and lines[0]["ordinary_ms"] == 32.4 and lines[0]["tokens"] == 9
    silent = Session(monkeypatch, EAGER, CopyGate(pinned=EAGER))
    silent.run(3, {})
    silent.round((8, 0))
    assert len(log.read_text().splitlines()) == 1        # no file named: nothing written
