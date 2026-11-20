"""How far an arm's next-token distributions sit from a reference's.

Both sides come from the server's teacher-forced scoring contract
(``/v1/completions`` with ``echo``, ``logprobs=K``, ``max_tokens=0``): for
every position, the top-K alternatives as a string-keyed map plus the exact
log-probability of the token that actually follows.

Only the top K of each distribution is visible, so the divergence is taken
over a partition both sides can evaluate exactly: every token string that
appears in both maps is its own cell, and everything else (the tails, and any
string the decoder collapsed to U+FFFD) is one shared "rest" cell whose mass
is one minus the named cells. Merging outcomes can only lose information, so
this partition KL is a lower bound on the true KL(reference || arm); with
K = 64 the named cells hold nearly all the mass and the bound is tight.

The log-probability of the actual token needs no partition: its difference
between arm and reference (``nll_delta``) is exact, and over text the
reference itself wrote it estimates the same KL from the other direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

# A string key holding U+FFFD stands for a byte-level piece the decoder could
# not render; distinct tokens collapse onto it, so it never names one outcome.
_COLLAPSED = "�"
# A rest cell the arm leaves empty while the reference does not would make the
# KL infinite; the floor keeps the number finite and the event is counted.
_REST_FLOOR = 1e-9


@dataclass(frozen=True)
class PositionDivergence:
    kl: float  # partition KL(reference || arm), nats
    top1_agrees: bool
    nll_delta: float  # arm NLL - reference NLL of the token that follows
    reference_rest_mass: float  # reference mass outside the shared cells
    rest_floored: bool  # the arm's rest cell was empty and got the floor


def _named(top: Mapping[str, float]) -> dict[str, float]:
    return {key: value for key, value in top.items() if _COLLAPSED not in key}


def _argmax(top: Mapping[str, float]) -> str | None:
    return max(top, key=top.__getitem__) if top else None


def position_divergence(
    reference_top: Mapping[str, float],
    arm_top: Mapping[str, float],
    *,
    reference_logprob: float,
    arm_logprob: float,
) -> PositionDivergence:
    reference_named = _named(reference_top)
    arm_named = _named(arm_top)
    shared = reference_named.keys() & arm_named.keys()

    kl = 0.0
    p_named = q_named = 0.0
    for key in shared:
        p = math.exp(reference_named[key])
        q = math.exp(arm_named[key])
        p_named += p
        q_named += q
        kl += p * (reference_named[key] - arm_named[key])

    p_rest = max(0.0, 1.0 - p_named)
    q_rest = max(0.0, 1.0 - q_named)
    floored = p_rest > _REST_FLOOR and q_rest < _REST_FLOOR
    if p_rest > 0.0:
        kl += p_rest * (math.log(p_rest) - math.log(max(q_rest, _REST_FLOOR)))

    return PositionDivergence(
        kl=max(0.0, kl),  # rounding in the reported logprobs can dip below zero
        top1_agrees=_argmax(reference_top) == _argmax(arm_top),
        nll_delta=float(reference_logprob) - float(arm_logprob),
        reference_rest_mass=p_rest,
        rest_floored=floored,
    )


@dataclass(frozen=True)
class DivergenceSummary:
    positions: int
    entries: int
    mean_kl: float
    mean_kl_stderr: float  # across entries, each entry one block of positions
    median_kl: float
    p99_kl: float
    max_kl: float
    top1_agreement: float
    mean_nll_delta: float
    mean_reference_rest_mass: float
    rest_floored_positions: int

    def as_dict(self) -> dict[str, float | int]:
        return dict(self.__dict__)


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, math.ceil(q * len(sorted_values)) - 1))
    return sorted_values[index]


def summarize(entries: Iterable[Sequence[PositionDivergence]]) -> DivergenceSummary:
    """Pool positions; the error bar treats each entry (one text) as a block."""

    per_entry: list[tuple[int, float]] = []
    kls: list[float] = []
    agree = 0
    nll = rest = 0.0
    floored = 0
    for positions in entries:
        if not positions:
            continue
        entry_kl = [p.kl for p in positions]
        per_entry.append((len(entry_kl), sum(entry_kl) / len(entry_kl)))
        kls.extend(entry_kl)
        agree += sum(p.top1_agrees for p in positions)
        nll += sum(p.nll_delta for p in positions)
        rest += sum(p.reference_rest_mass for p in positions)
        floored += sum(p.rest_floored for p in positions)

    n = len(kls)
    if n == 0:
        raise ValueError("no scored positions to summarize")
    mean = sum(kls) / n
    stderr = 0.0
    if len(per_entry) > 1:
        weights = [count / n for count, _ in per_entry]
        variance = sum(w * (m - mean) ** 2 for w, (_, m) in zip(weights, per_entry))
        stderr = math.sqrt(variance / (len(per_entry) - 1))
    ordered = sorted(kls)
    return DivergenceSummary(
        positions=n,
        entries=len(per_entry),
        mean_kl=mean,
        mean_kl_stderr=stderr,
        median_kl=_quantile(ordered, 0.5),
        p99_kl=_quantile(ordered, 0.99),
        max_kl=ordered[-1],
        top1_agreement=agree / n,
        mean_nll_delta=nll / n,
        mean_reference_rest_mass=rest / n,
        rest_floored_positions=floored,
    )
