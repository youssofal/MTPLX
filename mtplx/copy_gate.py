"""Cost gate for context-copy rounds.

A copy round proposes a block copied from the prompt and verifies it in one wide forward.
It commits more tokens than an ordinary draft round when the block lands and fewer when
it does not, and it costs more either way because the verify is wider. The copy lane's
own rules (probation, acceptance average, suspension) look only at how much landed. This
gate adds the missing half: what each kind of round costs, measured in the request.

Costs come from the request's own rounds: the wall time of its ordinary rounds and of its
copy rounds (a line in the verify rows once two widths have been timed). The first round
of each kind is not counted, because it carries one-off work (a trace, a first allocation).
Until two ordinary rounds and two copy rounds have been timed after that, the gate lets
every block through unchanged, so a short answer runs exactly as it does without the gate.

After that a copy round runs when its expected tokens per second beat the ordinary
round's by a margin, at the ladder length that maximises that rate, or when the copy
round before it paid for itself (a streak keeps going until a round does not pay, and steps
up one ladder length at a time to time the widths it has not timed yet). While
the estimate says no, a trial of the shortest block still runs at a spacing chosen so that
trials cost a fixed small share of decode time: that is how a rewrite that starts later
is noticed.

The gate can veto a block or shorten it to a shorter ladder length. It never lengthens a
block, never proposes tokens of its own and never touches the verify or the acceptance
rule, so every emitted token still comes out of the exact verify and the output keeps the
target sampling law (distribution-exact). Which rounds run depends on measured time, so
the tokens of one seed can differ between two runs with the gate on; pin the costs to
make a run repeatable. It reads token counts only: text, tool-call JSON and image turns
go through the same rule.

MTPLX_CONTEXT_COPY_GATE=1/true/on turns it on (default off: nothing here runs).
MTPLX_CONTEXT_COPY_GATE_COSTS="ordinary_ms,copy_fixed_ms,copy_row_ms" pins the costs.
MTPLX_CONTEXT_COPY_GATE_LOG=<file> appends one JSON line per decision and per copy round.

Wiring, all inside the copy lane: `cycle(len(tokens), verify_calls)` where the lane
probes, `plan(block_length)` where it has chosen a block length, `observe_copy(drafted,
accepted)` where it books a copy round.
"""

from __future__ import annotations

import itertools
import json
import os
import time
from dataclasses import dataclass, field

LADDER = (8, 12, 16, 24, 32)   # the copy lane's block lengths (context_copy._BLOCK_LADDER)
MIN_SAMPLES = 2                 # timed rounds of each kind before the gate decides anything
_IDS = itertools.count(1)


def enabled() -> bool:
    return (os.environ.get("MTPLX_CONTEXT_COPY_GATE") or "").strip().lower() in {"1", "true", "on"}


@dataclass(frozen=True)
class RoundCosts:
    """What a round costs, in seconds. A copy round's cost is a line in its verify rows."""

    ordinary_s: float
    copy_fixed_s: float
    copy_row_s: float

    def copy_round_s(self, rows: int) -> float:
        return self.copy_fixed_s + self.copy_row_s * rows

    def valid(self) -> bool:
        return self.ordinary_s > 0.0 and self.copy_round_s(LADDER[0] + 1) > 0.0 and self.copy_row_s >= 0.0


def pinned_costs() -> RoundCosts | None:
    raw = (os.environ.get("MTPLX_CONTEXT_COPY_GATE_COSTS") or "").strip()
    if not raw:
        return None
    try:
        ordinary, fixed, row = (float(part) / 1000.0 for part in raw.split(","))
    except ValueError:
        return None
    costs = RoundCosts(ordinary, fixed, row)
    return costs if costs.valid() else None


@dataclass
class CostMeter:
    """Round times of one request, by kind. The first sample of each kind is dropped."""

    weight: float = 0.1                      # share a new sample takes once the mean has ten
    outlier: float = 5.0                     # a sample this many times the mean is a stall, not a round
    _ordinary_s: float = 0.0
    _ordinary_n: int = 0
    _ordinary_seen: int = 0
    _copy_seen: int = 0
    _copy: dict = field(default_factory=dict)   # verify rows -> [mean seconds, samples]

    def observe_ordinary(self, seconds: float) -> None:
        self._ordinary_seen += 1
        if self._ordinary_seen == 1 or seconds <= 0.0:
            return
        if self._ordinary_n >= MIN_SAMPLES and seconds > self.outlier * self._ordinary_s:
            return
        self._ordinary_n += 1
        self._ordinary_s += max(self.weight, 1.0 / self._ordinary_n) * (seconds - self._ordinary_s)

    def observe_copy(self, rows: int, seconds: float) -> None:
        self._copy_seen += 1
        if self._copy_seen == 1 or seconds <= 0.0 or rows < 2:
            return
        cell = self._copy.setdefault(int(rows), [0.0, 0])
        if cell[1] >= MIN_SAMPLES and seconds > self.outlier * cell[0]:
            return
        cell[1] += 1
        cell[0] += max(self.weight, 1.0 / cell[1]) * (seconds - cell[0])

    def snapshot(self) -> RoundCosts | None:
        """None until both kinds have `MIN_SAMPLES` timed rounds."""
        cells = [(rows, mean, n) for rows, (mean, n) in self._copy.items() if n >= 1]
        total = sum(n for _, _, n in cells)
        if self._ordinary_n < MIN_SAMPLES or total < MIN_SAMPLES:
            return None
        mean_rows = sum(rows * n for rows, _, n in cells) / total
        mean_s = sum(mean * n for _, mean, n in cells) / total
        spread = sum(n * (rows - mean_rows) ** 2 for rows, _, n in cells)
        row_s = fixed_s = -1.0
        if spread > 0.0:
            row_s = sum(n * (rows - mean_rows) * (mean - mean_s) for rows, mean, n in cells) / spread
            fixed_s = mean_s - row_s * mean_rows
        if row_s <= 0.0 or fixed_s < 0.0:
            # One width only, or a fit that makes no physical sense: price a row at the mean.
            row_s, fixed_s = mean_s / mean_rows, 0.0
        costs = RoundCosts(self._ordinary_s, fixed_s, row_s)
        return costs if costs.valid() else None


def _expected_landed(survival: float, length: int) -> float:
    """Tokens of a block of `length` expected to land when each lands with `survival` given the one before."""
    if survival >= 1.0:
        return float(length)
    if survival <= 0.0:
        return 0.0
    return survival * (1.0 - survival ** length) / (1.0 - survival)


def _survival_for_share(share: float, length: int) -> float:
    """The per-token survival at which a block of `length` lands `share` of its tokens (bisection)."""
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _expected_landed(mid, length) < share * length:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class CopyGate:
    """One request's gate."""

    pinned: RoundCosts | None = None
    margin: float = 0.05          # a copy round must beat the ordinary rate by this much
    trial_share: float = 0.01     # share of decode time spent on trials while the estimate says no
    memory: float = 0.7           # weight the lane already gives its acceptance history per round
    log_path: str = ""
    # The lane starts its acceptance average at half a probation block; start the survival
    # estimate at the same place, worth one block of evidence.
    prior_survival: float = field(default_factory=lambda: _survival_for_share(0.5, LADDER[0]))
    meter: CostMeter = field(default_factory=CostMeter)
    gate_id: int = field(default_factory=lambda: next(_IDS))
    counts: dict = field(default_factory=lambda: {
        "passed_cold": 0, "taken_rate": 0, "taken_streak": 0, "trial": 0, "vetoed_rate": 0, "shortened": 0,
    })
    _landed: float = 0.0
    _stops: float = 0.0
    _ordinary_tokens: float = 0.0
    _ordinary_rounds: int = 0
    _since_copy: int = 0
    _failed_trials: int = 0
    _last_was_trial: bool = False
    _last_paid: bool = False
    _clock: float = 0.0
    _started: bool = False
    _total: int = 0
    _rounds: int = 0
    _pending: tuple | None = None

    # ---- what the request has measured
    def costs(self) -> RoundCosts | None:
        return self.pinned or self.meter.snapshot()

    def survival(self) -> float:
        prior_landed = _expected_landed(self.prior_survival, LADDER[0])
        prior_stops = 1.0 - self.prior_survival ** LADDER[0]
        landed = self._landed + prior_landed
        return landed / (landed + self._stops + prior_stops)

    def ordinary_rate(self, costs: RoundCosts) -> float:
        """Tokens a second of this request's ordinary rounds."""
        return max(self._ordinary_tokens / self._ordinary_rounds, 1.0) / costs.ordinary_s

    def copy_rate(self, costs: RoundCosts, length: int) -> float:
        return (_expected_landed(self.survival(), length) + 1.0) / costs.copy_round_s(length + 1)

    def trial_period(self, costs: RoundCosts, length: int) -> int:
        """Ordinary rounds between trials: a trial that wastes W seconds, run once in W / (share x
        ordinary round) rounds, costs `trial_share` of decode time. Doubles after each trial that
        failed, as the lane's own suspension does."""
        made = _expected_landed(self.survival(), length) + 1.0
        wasted = costs.copy_round_s(length + 1) - made / self.ordinary_rate(costs)
        if wasted <= 0.0:
            return 1
        period = wasted / (self.trial_share * costs.ordinary_s)
        return int(min(period * 2 ** min(self._failed_trials, 6), 1e9)) + 1

    # ---- the three calls
    def cycle(self, total_tokens: int, rounds: int) -> None:
        """Where the lane probes, once a cycle while it is not suspended: `total_tokens` committed
        and `rounds` verified so far. Closes what ran since the last call. A span is timed only
        when it is exactly one round of one kind."""
        now = time.perf_counter()
        d_tokens, d_rounds = total_tokens - self._total, rounds - self._rounds
        if self._started and d_rounds > 0:
            span = now - self._clock
            ordinary, tokens = d_rounds, d_tokens
            if self._pending is not None:
                rows, landed = self._pending
                ordinary, tokens = d_rounds - 1, d_tokens - (landed + 1)
                if ordinary == 0:
                    self.meter.observe_copy(rows, span)
            elif d_rounds == 1:
                self.meter.observe_ordinary(span)
            if ordinary > 0:
                self._ordinary_rounds += ordinary
                self._ordinary_tokens += max(tokens, ordinary)
                self._since_copy += ordinary
        self._pending = None
        self._started = True
        self._total, self._rounds, self._clock = total_tokens, rounds, now

    def plan(self, offered: int) -> int:
        """Block length to copy this cycle, at most `offered`; 0 means run an ordinary round."""
        self._last_was_trial = False
        if offered <= 0:
            return 0
        costs = self.costs()
        if costs is None or self._ordinary_rounds < MIN_SAMPLES:
            return self._decide(offered, offered, "passed_cold", costs)
        lengths = [n for n in LADDER if n < offered] + [offered]
        best = max(lengths, key=lambda n: self.copy_rate(costs, n))
        if self._last_paid:
            # A streak is the cheap moment to time the next width up: without it a request that
            # has only timed probation blocks prices every row at their mean and never learns
            # that a longer window costs less per row.
            longer = [n for n in lengths if n > best and (n + 1) not in self.meter._copy]
            if longer and self.pinned is None:
                best = longer[0]
            return self._decide(offered, best, "taken_streak", costs)
        if self.copy_rate(costs, best) > self.ordinary_rate(costs) * (1.0 + self.margin):
            return self._decide(offered, best, "taken_rate", costs)
        if self._since_copy >= self.trial_period(costs, lengths[0]):
            self._last_was_trial = True
            return self._decide(offered, lengths[0], "trial", costs)
        return self._decide(offered, 0, "vetoed_rate", costs)

    def observe_copy(self, drafted: int, accepted: int) -> None:
        if drafted <= 0:
            return
        self._pending = (drafted + 1, accepted)
        self._landed = self.memory * self._landed + accepted
        self._stops = self.memory * self._stops + (1.0 if accepted < drafted else 0.0)
        costs = self.costs()
        paid = None
        if costs is not None and self._ordinary_rounds >= MIN_SAMPLES:
            paid = (accepted + 1.0) / costs.copy_round_s(drafted + 1) > self.ordinary_rate(costs)
            if self._last_was_trial:
                self._failed_trials = 0 if paid else self._failed_trials + 1
            elif paid:
                self._failed_trials = 0
        self._last_paid = bool(paid)
        self._since_copy = 0
        self._last_was_trial = False
        self._log({"ev": "copy", "drafted": drafted, "landed": accepted, "paid": paid})

    # ---- bookkeeping
    def _decide(self, offered: int, length: int, why: str, costs: RoundCosts | None) -> int:
        self.counts[why] += 1
        if 0 < length < offered:
            self.counts["shortened"] += 1
        if self.log_path:
            line = {"ev": "plan", "offered": offered, "length": length, "why": why, "survival": round(self.survival(), 4)}
            if costs is not None and self._ordinary_rounds:
                line.update(
                    ordinary_ms=round(1000.0 * costs.ordinary_s, 2),
                    copy_ms=round(1000.0 * costs.copy_round_s((length or offered) + 1), 2),
                    ordinary_tokens=round(self._ordinary_tokens / self._ordinary_rounds, 2),
                )
            self._log(line)
        return length

    def _log(self, line: dict) -> None:
        if not self.log_path:
            return
        line = {"gate": self.gate_id, "tokens": self._total, **line, "counts": dict(self.counts)}
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")

    def stats(self) -> dict:
        costs = self.costs()
        out = dict(self.counts, survival=round(self.survival(), 4), pinned=self.pinned is not None)
        if costs is not None:
            out.update(
                ordinary_ms=round(1000.0 * costs.ordinary_s, 2),
                copy_fixed_ms=round(1000.0 * costs.copy_fixed_s, 2),
                copy_row_ms=round(1000.0 * costs.copy_row_s, 3),
            )
        return out


def for_request() -> CopyGate | None:
    """None with the switch off: the lane is untouched."""
    if not enabled():
        return None
    log_path = (os.environ.get("MTPLX_CONTEXT_COPY_GATE_LOG") or "").strip()
    if log_path:
        open(log_path, "a", encoding="utf-8").close()   # a path that cannot be written fails here, before any token
    return CopyGate(pinned=pinned_costs(), log_path=log_path)
