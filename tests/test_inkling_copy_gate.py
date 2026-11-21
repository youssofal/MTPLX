"""The copy-round cost gate (``mtplx/copy_gate.py``) on Inkling 2 Mini's round costs.

Costs used below are Bare Speed's at a 4K context, from lane C1's census (provisional, M5 Max):
an ordinary 3-guess round about 28.9 ms for about 2.9 tokens (100 tokens a second), and a copy
round on the line 17.5 ms + 4.2 ms a verify row (55 ms at 9 rows, 122.5 ms at 25 rows).
"""

import pytest

from mtplx import copy_gate
from mtplx.copy_gate import CopyGate, CostMeter, RoundCosts

INKLING_4K = RoundCosts(ordinary_s=0.0289, copy_fixed_s=0.0175, copy_row_s=0.0042)
ORDINARY_TOKENS = 2.9


def _drive(gate, rounds, offer, landed_of, quiet=4):
    """The lane's calls for ``rounds`` rounds: no match for the first ``quiet`` (ordinary rounds),
    then a block of ``offer`` tokens offered each round. Returns the planned lengths."""
    tokens = done = 0
    planned = []
    for i in range(rounds):
        gate.cycle(int(tokens), done)
        length = gate.plan(offer if i >= quiet else 0)
        if length:
            landed = landed_of(length)
            gate.observe_copy(length, landed)
            tokens += landed + 1
        else:
            tokens += ORDINARY_TOKENS
        done += 1
        planned.append(length)
    return planned


def test_off_by_default_and_pinned_inkling_costs_parse(monkeypatch):
    monkeypatch.delenv("MTPLX_CONTEXT_COPY_GATE", raising=False)
    monkeypatch.delenv("MTPLX_CONTEXT_COPY_GATE_COSTS", raising=False)
    assert copy_gate.for_request() is None
    monkeypatch.setenv("MTPLX_CONTEXT_COPY_GATE", "1")
    assert copy_gate.for_request().pinned is None
    monkeypatch.setenv("MTPLX_CONTEXT_COPY_GATE_COSTS", "28.9,17.5,4.2")
    pinned = copy_gate.for_request().pinned
    assert (pinned.ordinary_s, pinned.copy_fixed_s, pinned.copy_row_s) == pytest.approx((0.0289, 0.0175, 0.0042))
    assert pinned.copy_round_s(25) == pytest.approx(0.1225)


def test_the_meter_skips_each_kinds_first_round_and_fits_the_copy_line():
    meter = CostMeter()
    meter.observe_ordinary(1.0)          # a first round carries one-off work: not counted
    for _ in range(3):
        meter.observe_ordinary(0.0289)
    meter.observe_copy(25, 2.0)          # the first copy round: not counted
    for rows in (9, 25, 9, 25):
        meter.observe_copy(rows, INKLING_4K.copy_round_s(rows))
    costs = meter.snapshot()
    assert costs.ordinary_s == pytest.approx(0.0289)
    assert costs.copy_fixed_s == pytest.approx(0.0175)
    assert costs.copy_row_s == pytest.approx(0.0042)


def test_a_rewrite_whose_blocks_land_keeps_the_longest_block():
    # 24 copied tokens a round at 122.5 ms: about 200 tokens a second against 100 for ordinary
    # rounds. The gate starts from the lane's own prior (a probation block lands half its tokens),
    # which prices a copy round below an ordinary one: it vetoes until a trial lands, then the
    # streak keeps every block. That start costs a rewrite its first rounds (lane C1's note).
    gate = CopyGate(pinned=INKLING_4K)
    planned = _drive(gate, 80, 24, landed_of=lambda length: length)
    assert planned[-40:] == [24] * 40
    first = next(i for i, n in enumerate(planned) if n)
    assert gate.counts["vetoed_rate"] == first - 4 and first - 4 <= 20


def test_blocks_that_land_nothing_are_vetoed_and_trials_stay_rare():
    gate = CopyGate(pinned=INKLING_4K)
    planned = _drive(gate, 600, 24, landed_of=lambda length: 0)
    copies = sum(1 for n in planned if n)
    assert gate.counts["vetoed_rate"] > 500
    # trials run the shortest block, spaced so they cost about 1% of decode time
    assert copies <= 20 and all(n in (0, 8, 24) for n in planned)


@pytest.mark.parametrize("offer", [1, 7, 8, 11, 16, 24, 31])
def test_the_gate_never_lengthens_a_block(offer):
    gate = CopyGate(pinned=INKLING_4K)
    planned = _drive(gate, 80, offer, landed_of=lambda length: length // 2)
    assert all(0 <= n <= offer for n in planned)
