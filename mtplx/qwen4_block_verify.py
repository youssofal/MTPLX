"""Block verification (Sun et al. 2024, arXiv:2403.10444) for the stock lane.

NO MLX.  Pure NumPy on host rows the stock native-MTP accept loop has already
built for its own decision.  This module never evaluates an ``mx`` array, never
draws a random number, and never touches the device.

Why
---
``H-tokens-per-window-design.md`` §Option B.  The shipped law decides depth
``d`` from ``x_d`` alone::

    alpha_d = min(1, rho_d)          rho_d = p_d(x_d) / q_d(x_d)

and that is already saturated at depth 1 (H §3.1: exactness forces
``q(x) A_1(x) <= p(x)``, so no law can accept the first draft more often).  The
slack is one clip deeper.  The whole M4 verify forward is finished before the
accept loop runs, so **all** target rows and the full-depth draft chain are on
the host when depth 1 is decided.  The decision may therefore legally look at
``rho_2`` and ``rho_3``; the only exactness constraint is on the *marginal*
reach probability::

    Abar_d(x_{1:d}) <= Abar_{d-1} * rho_d(x_d)   and   Abar_d <= 1

Clipping the running product instead of clipping each factor and multiplying,
and spending the resulting budget preferentially on the realisations whose
*next* drafted token the target likes, is block verification.  Measured offline
on 381 real windows: **+1.85% tokens/window (2.487 -> 2.533)**.

The law, exactly as implemented
-------------------------------
``tests/block_verify_reference.py`` is the float64 reference and
this module is its in-loop mirror -- same arithmetic, same tie ownership, same
``reach`` cap.  Per window, with ``c_0 = w_0 = 1``::

    for d in 1..D:
        rho_d = p_d(x_d) / q_d(x_d)                    # 0 when the target
                                                       # zeroed the drafted token
        A_d   = min(1, c_{d-1} * rho_d)                # reach BUDGET
        if d < D:
            base(y) = min(1, A_d * rho_{d+1}(y))       # y over draft row d+1
            lam_d   = water-fill level s.t.
                      sum_y q_{d+1}(y) * min(w_{d-1}, base(y) + lam_d) = A_d
            w_d     = min(w_{d-1}, base(x_{d+1}) + lam_d)   # REALISED reach
        else:
            w_d     = A_d
        a_d = w_d / w_{d-1}                            # CONDITIONAL accept coin
        if u_d <= a_d:  c_d = A_d ; w_d stays ; continue
        emit sample(normalise((c_{d-1} * p_d - q_d)+))  # SCALED residual
        stop
    emit bonus ~ p_{D+1}                               # unchanged

``lam_d`` is the water-filling level that holds
``E_{x_{d+1} ~ q_{d+1}}[w_d]`` at the budget ``A_d``, which is what preserves
exactness: averaged over the next drafted token the position-``d`` accept
probability is still ``min(1, rho_d)``, so ``q_d(y) P(accept | x_d = y) <=
p_d(y)`` holds pointwise and the residual stays non-negative.

**The cap is ``w_{d-1}``, not H's literal ``min(1, .)``.**  H's pseudo-code
lets ``w_d`` exceed ``w_{d-1}``, and then ``a_d = w_d / w_{d-1} > 1`` is not a
probability.  Capping the water-fill at ``w_{d-1}`` is the smallest change that
makes the law well defined; it is still budget-exact (feasible because
``w_{d-1} >= A_d``) and the coin is always a probability.  The reference calls
this ``--cap reach`` and it is its default; :data:`CAP_MODE` pins the in-loop
lane to it.

**The c = 1 identity.**  When the ladder never drops below 1 the two laws
coincide token for token: ``a_d`` collapses to ``min(1, rho_d)`` and the
residual scale to 1.  That holds on 43-47% of windows (H §1.2), and it is the
partial parity check a staged rollout gets for free -- the offline scorer fails
if it ever stops holding.

Draw accounting -- unchanged
----------------------------
Block verification consumes **exactly the same randomness as the shipped law**:
one accept coin per depth reached, one ``rng.choice`` for a correction, one for
the bonus.  Nothing in this module takes an RNG.  The water-fill is
deterministic, so arming the flag cannot shift the PCG64 stream for a given
outcome path (``mtplx/pcg64_tape.py``'s ``DRAWS_PER_CYCLE`` is a PR391-lane
constant and is untouched either way).

Row preparation
---------------
Rows arrive as ``SparseDistribution`` / dense arrays that the stock lane has
already shaped (temperature, top-p, top-k, renormalised -- ``sampling.py``).
:func:`prepared_pair` applies exactly what the reference's ``prepared_row``
applies to the *logged* copies of the same arrays: drop the zero-probability
entries, sort by token id ascending, renormalise once.  Same inputs, same
operations, same order -- so the in-loop ladder is bit-identical to the
offline reference's, which is what makes an armed K20 log replayable as an
exactness proof rather than a smoke test.

What ``accept_probability`` means when armed
--------------------------------------------
``a_d``, the **conditional** accept probability of depth ``d`` given the window
reached it.  That is the same thing the shipped law's ``min(1, rho_d)`` is (the
probability the coin at that depth accepts), so ``accept_probability_sum_by_depth``
and ``drafts[].accept_probability`` keep their arithmetic meaning; what changes
is that under BV the value depends on the depth ``d+1`` rows as well as on
``x_d``.  It is no longer an estimate of the TV overlap ``beta_d``: H §1.2's
``E[alpha] = beta`` identity is a property of ``min(1, rho)`` and does not
survive the water-fill.  Read ``alpha_uncensored`` out of the offline scorer
for that.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np

from .sampling import SparseDistribution

_ENV_VAR = "MTPLX_QWEN4_BLOCK_VERIFY"

#: The only water-fill cap the in-loop lane implements.  See the module
#: docstring: H's literal ``min(1, .)`` does not yield a probability.
CAP_MODE = "reach"

ZERO = np.float64(0.0)
ONE = np.float64(1.0)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


#: ``None`` = resolve from the environment on each read; a test may force a
#: bool via :func:`_configure_for_test`. Read at USE, never at import: the
#: server's fixed-M4 auto-arm stamps MTPLX_QWEN4_BLOCK_VERIFY into the
#: environment AFTER this module is imported, so an import-time read froze the
#: default (off) and the served accept loop never used it -- the arming audit,
#: 2026-09-07. The env is frozen once serving starts, so a per-call read is the
#: same value at every accept step, drawing the same uniforms in the same
#: order as before.
_ENABLED = None


def is_enabled() -> bool:
    """True when ``MTPLX_QWEN4_BLOCK_VERIFY`` is set for this process.

    Read at use, not frozen at import (a test may force :data:`_ENABLED`).
    """

    if _ENABLED is not None:
        return bool(_ENABLED)
    return _env_truthy(_ENV_VAR)


def _configure_for_test(enabled: bool) -> None:
    """Flip the import-time gate (tests only)."""

    global _ENABLED
    _ENABLED = bool(enabled)


#: First-use engagement latch: the accept loop has no other per-window
#: observable, so /health surfaces this so the battery can confirm the block
#: law actually ran instead of trusting the env. Read-only reporting.
_ENGAGED = [False]


def engagement_report() -> dict:
    """Install/first-use verdict for ``/health qwen4_install_reports.block_verify``.

    ``armed`` is read at use (reflects the served auto-arm stamp, gate-able
    without a request); ``engaged`` latches True the first window a block
    verifier is actually built for the accept loop.
    """

    return {"armed": is_enabled(), "engaged": bool(_ENGAGED[0])}


def reset_engagement_for_test() -> None:
    """Clear the first-use latch (tests only)."""

    _ENGAGED[0] = False


# ---------------------------------------------------------------------------
# Row preparation -- mirrors offline_block_verification.prepared_row exactly.
# ---------------------------------------------------------------------------


def renormalize(probabilities: np.ndarray) -> np.ndarray:
    """Mirror ``_renormalize_sparse_probabilities`` / ``renormalize_sparse``."""

    sanitized = np.where(
        np.isfinite(probabilities) & (probabilities > ZERO), probabilities, ZERO
    )
    return sanitized / np.sum(sanitized, dtype=np.float64)


def prepared_pair(distribution: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """``(ids ascending, probs)`` from a host row, or ``None`` when empty.

    Composes ``qwen4_k20_log._distribution_rows`` with the reference's
    ``prepared_row``: read ``token_ids``/``probs`` off a sparse row (or the
    positive entries of a dense one), drop the zero-probability padding, sort
    by token id, renormalise once.  Doing it here rather than through
    ``BatchedSparseDistributions.to_distribution`` is deliberate: that
    normalises in the row's *original* order, and float64 summation is
    order-dependent, so the ladder would drift from the offline reference in
    the last ULP.
    """

    if distribution is None:
        return None
    ids = getattr(distribution, "token_ids", None)
    probs = getattr(distribution, "probs", None)
    if ids is None or probs is None:
        dense = np.asarray(distribution, dtype=np.float64).reshape(-1)
        keep = np.flatnonzero(dense > 0.0)
        ids = keep.astype(np.int64)
        probs = dense[keep]
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    if ids.size != probs.size:
        return None
    keep = probs > ZERO
    kept_ids = ids[keep]
    kept_probs = probs[keep]
    if kept_ids.size == 0:
        return None
    order = np.argsort(kept_ids)
    return kept_ids[order], renormalize(kept_probs[order])


def lookup(token_ids: np.ndarray, probabilities: np.ndarray, token: int) -> np.float64:
    """Mirror the reference's ``lookup``."""

    hits = np.nonzero(token_ids == np.int64(token))[0]
    return ZERO if hits.size == 0 else np.float64(probabilities[int(hits[0])])


def lookup_many(
    token_ids: np.ndarray, probabilities: np.ndarray, wanted: np.ndarray
) -> np.ndarray:
    """Vectorised :func:`lookup` on a prepared (id-ascending, unique) row."""

    wanted = np.asarray(wanted, dtype=np.int64)
    out = np.zeros(wanted.size, dtype=np.float64)
    if token_ids.size == 0 or wanted.size == 0:
        return out
    position = np.searchsorted(token_ids, wanted)
    clipped = np.minimum(position, token_ids.size - 1)
    hit = token_ids[clipped] == wanted
    out[hit] = np.asarray(probabilities, dtype=np.float64)[clipped[hit]]
    return out


def water_fill_lambda(
    q: np.ndarray, base: np.ndarray, cap: np.float64, target: np.float64
) -> np.float64:
    """Smallest ``lam >= 0`` with ``sum q * min(cap, base + lam) == target``.

    Bit-for-bit the reference's ``water_fill_lambda``.  ``f(lam)`` is
    continuous, non-decreasing and piecewise linear with breakpoints at
    ``cap - base``; ``f(0) <= target`` always holds here because
    ``sum_y min(q(y), A p(y)) <= A``, and ``f(inf) = cap * sum q >= target``
    whenever ``cap >= target``, so a root exists.  The walk consumes an entire
    tie group before testing, so the result does not depend on the order
    within a group of equal breakpoints.
    """

    q = np.asarray(q, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    cap = np.float64(cap)
    target = np.float64(target)
    value = np.sum(q * np.minimum(cap, base), dtype=np.float64)
    if value >= target:
        return ZERO
    threshold = cap - base
    order = np.argsort(threshold, kind="stable")
    threshold = threshold[order]
    weights = q[order]
    active = np.sum(weights[threshold > ZERO], dtype=np.float64)
    level = ZERO
    index = int(np.searchsorted(threshold, ZERO, side="right"))
    size = int(threshold.size)
    while index < size:
        edge = np.float64(threshold[index])
        if active <= ZERO:
            return level
        gain = active * (edge - level)
        if value + gain >= target:
            return level + (target - value) / active
        value += gain
        level = edge
        while index < size and np.float64(threshold[index]) == edge:
            active -= np.float64(weights[index])
            index += 1
    if active > ZERO:
        return level + (target - value) / active
    return level


# ---------------------------------------------------------------------------
# The window ladder.
# ---------------------------------------------------------------------------


class BlockVerifier:
    """One verify window's block-verification ladder, computed up front.

    The ladder is a deterministic function of the prepared rows and the drafted
    tokens -- it consults no uniform -- so the whole thing is built before the
    accept loop starts and the loop only *reads* it.  That is not just
    convenient: the reference advances ``c`` and ``w`` only on an accept and
    stops at the first rejection, so the entries this class exposes at depth
    ``d`` are exactly the ones that can ever be consumed (depth ``d`` is
    reached only when every shallower depth accepted).
    """

    __slots__ = (
        "depth",
        "draft_tokens",
        "draft",
        "target_single",
        "target_double",
        "vocab_size",
        "rho",
        "accept_probability",
        "residual_scale",
        "budget",
        "realised",
        "clipped",
    )

    def __init__(
        self,
        *,
        draft_tokens: Sequence[int],
        draft_rows: Sequence[tuple[np.ndarray, np.ndarray]],
        target_rows: Sequence[tuple[np.ndarray, np.ndarray]],
        vocab_size: int,
    ) -> None:
        self.depth = len(draft_rows)
        if len(target_rows) < self.depth or len(draft_tokens) < self.depth:
            raise ValueError("block verification needs one row per drafted depth")
        self.draft_tokens = [int(token) for token in draft_tokens[: self.depth]]
        self.draft = list(draft_rows)
        self.target_single = list(target_rows[: self.depth])
        # The reference's `target_double`: the residual is built from the
        # DOUBLE-normalised target row while `rho` uses the single-normalised
        # one (kernel lines 281 vs 297, mirrored in the offline Window).
        self.target_double = [
            (ids, renormalize(probs)) for ids, probs in self.target_single
        ]
        self.vocab_size = int(vocab_size)

        self.rho = [ZERO] * self.depth
        self.accept_probability = [0.0] * self.depth
        self.residual_scale = [1.0] * self.depth
        self.budget = [0.0] * self.depth
        self.realised = [0.0] * self.depth
        self.clipped = [0] * self.depth
        self._build()

    # -- construction ------------------------------------------------
    def _rho(self, depth: int, token: int) -> np.float64:
        """``p_d(token) / q_d(token)`` -- the reference's ``Window.rho``."""

        target_ids, target_probs = self.target_single[depth]
        draft_ids, draft_probs = self.draft[depth]
        p_value = lookup(target_ids, target_probs, token)
        q_value = lookup(draft_ids, draft_probs, token)
        if q_value <= ZERO:
            return ONE if p_value > ZERO else ZERO
        return np.float64(p_value / q_value)

    def _realised_reach(
        self, depth: int, *, budget: np.float64, reach: np.float64
    ) -> np.float64:
        """``w_d``: the water-filled look-ahead, or the raw budget at the end."""

        if depth + 1 >= self.depth:
            return budget
        cap = reach
        if budget >= cap:
            # The draft row sums to 1, so a budget at or above the cap has
            # exactly one solution: every realisation saturates.  Short-
            # circuiting is not an optimisation -- solving for
            # `lam = cap - min(base)` and adding it back reintroduces float64
            # rounding, and at `cap = budget = 1` that rounding is precisely
            # what breaks the c = 1 identity with the shipped law.
            return cap
        draft_ids, draft_probs = self.draft[depth + 1]
        target_ids, target_probs = self.target_single[depth + 1]
        next_p = lookup_many(target_ids, target_probs, draft_ids)
        next_rho = np.divide(
            next_p, draft_probs, out=np.zeros_like(next_p), where=draft_probs > ZERO
        )
        base = np.minimum(ONE, budget * next_rho)
        level = water_fill_lambda(draft_probs, base, cap, budget)
        next_token = self.draft_tokens[depth + 1]
        hits = np.nonzero(draft_ids == np.int64(next_token))[0]
        return min(cap, base[int(hits[0])] + level) if hits.size else min(cap, level)

    def _build(self) -> None:
        credit = ONE  # c_{d-1}: the reach budget entering this depth
        reach = ONE  # w_{d-1}: the probability this depth is reached at all
        for depth in range(self.depth):
            rho = self._rho(depth, self.draft_tokens[depth])
            budget = min(ONE, credit * rho)
            realised = self._realised_reach(depth, budget=budget, reach=reach)
            # reach == 0 is a measure-zero branch (it needs an earlier realised
            # reach of exactly 0 AND a uniform of exactly 0.0, which the `<=`
            # tie ownership does accept); the conditional is arbitrary there.
            coin = ONE if reach <= ZERO else np.float64(realised / reach)
            if coin > ONE:
                self.clipped[depth] = 1
                coin = ONE
            self.rho[depth] = rho
            self.residual_scale[depth] = float(credit)
            self.accept_probability[depth] = float(coin)
            self.budget[depth] = float(budget)
            self.realised[depth] = float(realised)
            credit = budget
            reach = realised if realised < reach else reach

    # -- what the accept loop reads ----------------------------------
    def scaled_residual(self, depth: int) -> SparseDistribution:
        """``normalise((c_{d-1} p_d - q_d)+)`` -- the block law's correction.

        Mirrors the reference's ``prepare_residual(scale=credit)``, which is
        ``sampling.residual_distribution`` with one scalar multiply on the
        target term: union both supports, ``max(c p - q, 0)``, drop the
        non-positive entries, normalise twice (once here and once inside
        ``SparseDistribution``, exactly as the shipped path does).  When
        nothing survives, fall back to the double-normalised target row.
        """

        target_ids, target_probs = self.target_double[depth]
        draft_ids, draft_probs = self.draft[depth]
        scale = np.float64(self.residual_scale[depth])
        union_ids = np.union1d(target_ids, draft_ids).astype(np.int64, copy=False)
        residual = np.maximum(
            scale * lookup_many(target_ids, target_probs, union_ids)
            - lookup_many(draft_ids, draft_probs, union_ids),
            ZERO,
        )
        residual = np.where(np.isfinite(residual) & (residual > ZERO), residual, ZERO)
        keep = residual > ZERO
        first_total = np.sum(residual[keep], dtype=np.float64)
        if not np.isfinite(first_total) or first_total <= ZERO:
            return SparseDistribution(target_ids, target_probs, self.vocab_size)
        return SparseDistribution(
            union_ids[keep], residual[keep] / first_total, self.vocab_size
        )

    # -- receipts ----------------------------------------------------
    def log_arrays(self) -> dict[str, list[float] | list[int]]:
        """The ladder, for the K20 log's ``stock_prepared_bv`` layout."""

        return {
            "coin": [float(value) for value in self.accept_probability],
            "scale": [float(value) for value in self.residual_scale],
            "budget": [float(value) for value in self.budget],
            "realised": [float(value) for value in self.realised],
            "clipped": [int(value) for value in self.clipped],
        }


def build_verifier(
    *,
    draft_tokens: Sequence[int],
    draft_probs: Sequence[Any],
    target_batch: Any = None,
    target_list: Sequence[Any] | None = None,
    vocab_size: int | None = None,
) -> BlockVerifier | None:
    """Arm block verification for one window, or ``None`` to keep the shipped law.

    ``None`` is returned whenever the window does not already hold every row
    the ladder needs -- ``D`` draft rows and ``D`` target rows (the bonus row
    is never consulted).  That is not a fallback of convenience: materialising
    a target row the lane deliberately skipped would add host work to the
    lazy path and change what an un-armed run costs.  The shipped law and the
    block law are both exact samplers of the same target distribution, so
    mixing them per window changes nothing about the output distribution.
    """

    depth = len(draft_tokens)
    if depth == 0:
        return None
    draft_rows: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(depth):
        row = prepared_pair(draft_probs[index] if index < len(draft_probs) else None)
        if row is None:
            return None
        draft_rows.append(row)

    target_rows: list[tuple[np.ndarray, np.ndarray]] = []
    if target_batch is not None:
        ids = np.asarray(target_batch.token_ids, dtype=np.int64)
        probs = np.asarray(target_batch.probs, dtype=np.float64)
        if int(ids.shape[0]) < depth:
            return None
        for index in range(depth):
            row = prepared_pair(_RawRow(ids[index], probs[index]))
            if row is None:
                return None
            target_rows.append(row)
        if vocab_size is None:
            vocab_size = int(getattr(target_batch, "vocab_size", 0))
    elif target_list is not None:
        if len(target_list) < depth:
            return None
        for index in range(depth):
            row = prepared_pair(target_list[index])
            if row is None:
                return None
            target_rows.append(row)
        if vocab_size is None:
            vocab_size = int(getattr(target_list[0], "vocab_size", 0))
    else:
        return None

    if not vocab_size:
        # A SparseDistribution needs a vocabulary bound for the residual it
        # hands back; the widest id in play is a safe, exact one.
        vocab_size = 1 + int(
            max(int(ids.max()) for ids, _ in (*draft_rows, *target_rows))
        )
    if not _ENGAGED[0]:
        _ENGAGED[0] = True
        import sys as _sys

        print(
            "[mtplx] MTPLX_QWEN4_BLOCK_VERIFY armed: block verification engaged "
            f"in the accept loop (depth={depth})",
            file=_sys.stderr,
            flush=True,
        )
    return BlockVerifier(
        draft_tokens=draft_tokens,
        draft_rows=draft_rows,
        target_rows=target_rows,
        vocab_size=int(vocab_size),
    )


class _RawRow:
    """One row of a ``BatchedSparseDistributions`` as :func:`prepared_pair` input."""

    __slots__ = ("token_ids", "probs")

    def __init__(self, token_ids: np.ndarray, probs: np.ndarray) -> None:
        self.token_ids = token_ids
        self.probs = probs


__all__ = [
    "CAP_MODE",
    "BlockVerifier",
    "build_verifier",
    "is_enabled",
    "prepared_pair",
    "water_fill_lambda",
]
