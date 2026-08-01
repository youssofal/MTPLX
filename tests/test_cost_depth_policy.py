"""Deterministic tests for CostModelDepthPolicy (omlx _DepthController port).

Wall-clock is injected by monkeypatching time.perf_counter so cycle costs
are exact and the tests are noise-free.
"""

from __future__ import annotations

import time

import pytest

from mtplx.adaptive import CostModelDepthPolicy


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def advance_ms(self, ms: float) -> None:
        self.now += ms / 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture()
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(time, "perf_counter", c)
    return c


def _cycle(policy, clock, *, accepted, cycle_ms):
    """Advance the injected clock by cycle_ms and observe one cycle at the
    policy's current depth."""
    d = policy.current_depth
    clock.advance_ms(cycle_ms)
    return policy.observe(attempted_depth=d, accepted_depths=min(accepted, d))


def test_warmup_sweeps_all_depths(clock):
    p = CostModelDepthPolicy(max_depth=3)
    seen = [p.current_depth]
    for _ in range(4):  # first observe cannot self-time; depth 3 repeats once
        _cycle(p, clock, accepted=3, cycle_ms=50)
        seen.append(p.current_depth)
    # starts at 3, walks 2, 1, then picks by score
    assert seen[0] == 3
    assert 2 in seen and 1 in seen
    assert set(p.t.keys()) == {1, 2, 3}


def test_prefers_deep_when_acceptance_high_and_marginal_cheap(clock):
    p = CostModelDepthPolicy(max_depth=3)
    # warmup: identical near-costs, full acceptance
    for ms in (52, 51, 50):
        _cycle(p, clock, accepted=3, cycle_ms=ms)
    for _ in range(60):
        _cycle(p, clock, accepted=3, cycle_ms=52)
    assert p.current_depth == 3


def test_drops_shallow_when_deep_acceptance_collapses(clock):
    p = CostModelDepthPolicy(max_depth=3)
    for ms in (90, 70, 50):  # depth 3 costs nearly 2x depth 1
        _cycle(p, clock, accepted=3, cycle_ms=ms)
    # depth-2/3 rejections: only the first draft position ever accepts
    for _ in range(120):
        d = p.current_depth
        clock.advance_ms(50 + 20 * (d - 1))
        p.observe(attempted_depth=d, accepted_depths=min(1, d))
    # expected tokens: d1 ~ 1+p1 vs d3 ~ 1+p1+p1p2+... with p2,p3 -> 0,
    # while t(3) ~ 1.8x t(1): depth 1 must win
    assert p.current_depth == 1


def test_probe_fires_and_returns(clock):
    p = CostModelDepthPolicy(max_depth=3)
    for ms in (60, 55, 50):
        _cycle(p, clock, accepted=3, cycle_ms=ms)
    actions = []
    for _ in range(120):
        out = _cycle(p, clock, accepted=3, cycle_ms=50)
        actions.append(out["action"])
    assert "probe" in actions, "staleness/rival probes never fired"
    assert "probe_done" in actions


def test_interface_contract(clock):
    p = CostModelDepthPolicy(max_depth=4, min_depth=2)
    for _ in range(40):
        out = _cycle(p, clock, accepted=4, cycle_ms=40)
        assert set(out) >= {
            "previous_depth", "attempted_depth", "accepted_depths",
            "next_depth", "action",
        }
        assert 2 <= p.current_depth <= 4


def test_huge_gaps_do_not_poison_cost(clock):
    p = CostModelDepthPolicy(max_depth=2)
    for ms in (50, 45):
        _cycle(p, clock, accepted=2, cycle_ms=ms)
    t_before = dict(p.t)
    # a 60s tool round-trip between cycles must not register as a cycle cost
    _cycle(p, clock, accepted=2, cycle_ms=60_000)
    assert p.t == t_before
