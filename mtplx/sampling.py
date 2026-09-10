"""Sampler and stochastic speculative-decoding helpers.

These utilities are intentionally NumPy based for fast correctness tests. The
runtime path can later swap equivalent MLX kernels behind the same semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

# OpenAI/vLLM valid range for presence_penalty / frequency_penalty.
PENALTY_MIN = -2.0
PENALTY_MAX = 2.0


@dataclass(frozen=True)
class SamplerConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    # OpenAI-style additive penalties (default 0.0 == exact no-op). Applied to
    # raw logits before temperature; counts cover completion tokens only.
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


@dataclass(frozen=True)
class SparseDistribution:
    token_ids: np.ndarray
    probs: np.ndarray
    vocab_size: int

    def __post_init__(self):
        token_ids = np.asarray(self.token_ids, dtype=np.int64)
        probs = np.asarray(self.probs, dtype=np.float64)
        if token_ids.ndim != 1 or probs.ndim != 1:
            raise ValueError("SparseDistribution expects 1D token_ids and probs")
        if token_ids.shape[0] != probs.shape[0]:
            raise ValueError("SparseDistribution token_ids/probs length mismatch")
        if token_ids.shape[0] == 0:
            raise ValueError("SparseDistribution cannot be empty")
        if np.any(np.isfinite(probs) & (probs < 0)):
            raise ValueError("SparseDistribution probabilities must be non-negative")
        sanitized = np.where(np.isfinite(probs) & (probs > 0), probs, 0.0)
        total = sanitized.sum()
        if not np.isfinite(total) or total <= 0:
            valid_ids = token_ids[
                (token_ids >= 0) & (token_ids < int(self.vocab_size))
            ]
            if valid_ids.shape[0] == 0:
                raise ValueError("SparseDistribution probabilities must have positive mass")
            token_ids = np.array([int(valid_ids[0])], dtype=np.int64)
            sanitized = np.array([1.0], dtype=np.float64)
            total = 1.0
        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "probs", sanitized / total)

    @classmethod
    def one_hot(cls, token_id: int, vocab_size: int) -> "SparseDistribution":
        return cls(np.array([int(token_id)], dtype=np.int64), np.array([1.0], dtype=np.float64), vocab_size)

    def probability(self, token_id: int) -> float:
        hits = np.nonzero(self.token_ids == int(token_id))[0]
        if hits.size == 0:
            return 0.0
        return float(self.probs[int(hits[0])])

    def to_dense(self) -> np.ndarray:
        dense = np.zeros(int(self.vocab_size), dtype=np.float64)
        dense[self.token_ids] = self.probs
        return dense


Distribution = np.ndarray | SparseDistribution


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        out = np.zeros_like(logits, dtype=np.float64)
        out[int(np.argmax(logits))] = 1.0
        return out
    scaled = logits / float(temperature)
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    total = np.sum(exp)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Cannot normalize logits into a probability distribution")
    return exp / total


def deterministic_top_k_order(values: np.ndarray, top_k: int) -> np.ndarray:
    """Return the highest-value ids, breaking exact ties by vocabulary id."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Expected a 1D value vector")
    size = int(values.shape[0])
    count = min(max(int(top_k), 0), size)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    token_ids = np.arange(size, dtype=np.int64)
    if count == size:
        return np.lexsort((token_ids, -values)).astype(np.int64, copy=False)
    cutoff = np.partition(values, size - count)[size - count]
    higher = np.flatnonzero(values > cutoff)
    tied = np.flatnonzero(values == cutoff)
    chosen = np.concatenate((higher, tied[: count - higher.size]))
    order = np.lexsort((chosen, -values[chosen]))
    return chosen[order].astype(np.int64, copy=False)


def apply_top_p_top_k(
    probs: np.ndarray, top_p: float = 1.0, top_k: int = 0
) -> np.ndarray:
    """Apply the same top-p then top-k order used by local `mlx_lm`.

    Proper speculative sampling requires target and draft probabilities to be
    filtered with exactly the same sampler semantics. Local `mlx_lm` applies
    top-p before top-k, so MTPLX's NumPy reference path mirrors that order.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError("Expected a 1D probability vector")
    size = int(probs.shape[0])
    bounded_top_k = int(top_k) if top_k and 0 < int(top_k) < size else 0
    mask = np.ones(size, dtype=bool)
    ranked: np.ndarray | None = None
    if bounded_top_k:
        ranked = deterministic_top_k_order(probs, bounded_top_k)
    if 0 < top_p < 1.0:
        order = ranked
        if order is None:
            order = deterministic_top_k_order(probs, size)
        sorted_probs = probs[order]
        cumulative = np.cumsum(sorted_probs)
        cumulative_before = np.concatenate(([0.0], cumulative[:-1]))
        keep_sorted = cumulative_before < top_p
        nucleus_mask = np.zeros_like(mask)
        nucleus_mask[order[keep_sorted]] = True
        mask &= nucleus_mask
    if bounded_top_k:
        top_mask = np.zeros_like(mask)
        top_mask[ranked] = True
        mask &= top_mask
    filtered = np.where(mask, probs, 0.0)
    total = filtered.sum()
    if total <= 0:
        filtered[int(np.argmax(probs))] = 1.0
        return filtered
    return filtered / total


def apply_top_k_top_p(probs: np.ndarray, top_k: int = 0, top_p: float = 1.0) -> np.ndarray:
    """Backward-compatible alias for the project sampler semantics."""
    return apply_top_p_top_k(probs, top_p=top_p, top_k=top_k)


def apply_penalties(
    logits: np.ndarray,
    token_counts: Mapping[int, int] | None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    penalty_overlay: Mapping[int, float] | None = None,
) -> np.ndarray:
    """Subtract OpenAI-style additive presence/frequency penalties on raw logits.

        logit[j] -= frequency_penalty * count[j] + presence_penalty * (count[j] > 0)

    Matches OpenAI's published formula and vLLM (``model_executor/layers/utils.py``):
    a positive penalty lowers the logit of reused tokens. ``token_counts`` maps a
    token id to how many times it has appeared **in the completion so far** (the
    caller scopes this to output tokens; the prompt is excluded). Penalties are
    clamped to ``[-2, 2]``.

    ``penalty_overlay`` is an additional sparse token->positive-subtraction map
    (the Loop Guard's DRY-style steering); unlike the count penalties it is not
    clamped — the caller caps it.

    Returns ``logits`` unchanged (same object, no copy) when both penalties are
    0/no counts and there is no overlay — the exactness-preserving no-op path.
    """
    presence = float(np.clip(presence_penalty, PENALTY_MIN, PENALTY_MAX))
    frequency = float(np.clip(frequency_penalty, PENALTY_MIN, PENALTY_MAX))
    counts_active = bool(token_counts) and (presence != 0.0 or frequency != 0.0)
    overlay_active = bool(penalty_overlay)
    if not counts_active and not overlay_active:
        return logits
    out = np.array(logits, dtype=np.float64, copy=True)
    if counts_active:
        ids = np.fromiter(token_counts.keys(), dtype=np.int64, count=len(token_counts))
        counts = np.fromiter(token_counts.values(), dtype=np.float64, count=len(token_counts))
        # Sparse scatter-subtract over only the seen tokens: O(unique_seen), not O(vocab).
        out[ids] -= frequency * counts + presence * (counts > 0)
    if overlay_active:
        overlay_ids = np.fromiter(
            penalty_overlay.keys(), dtype=np.int64, count=len(penalty_overlay)
        )
        overlay_vals = np.fromiter(
            penalty_overlay.values(), dtype=np.float64, count=len(penalty_overlay)
        )
        out[overlay_ids] -= overlay_vals
    return out


def distribution_from_logits(
    logits: np.ndarray,
    config: SamplerConfig,
    *,
    token_counts: Mapping[int, int] | None = None,
    penalty_overlay: Mapping[int, float] | None = None,
) -> np.ndarray:
    logits = apply_penalties(
        logits,
        token_counts,
        config.presence_penalty,
        config.frequency_penalty,
        penalty_overlay=penalty_overlay,
    )
    probs = softmax(logits, temperature=config.temperature)
    return apply_top_p_top_k(probs, top_p=config.top_p, top_k=config.top_k)


def _probability(distribution: Distribution, token_id: int) -> float:
    if isinstance(distribution, SparseDistribution):
        return distribution.probability(token_id)
    return float(distribution[token_id])


def _vocab_size(distribution: Distribution) -> int:
    if isinstance(distribution, SparseDistribution):
        return int(distribution.vocab_size)
    return int(np.asarray(distribution).shape[0])


def _as_dense(distribution: Distribution) -> np.ndarray:
    if isinstance(distribution, SparseDistribution):
        return distribution.to_dense()
    return np.asarray(distribution, dtype=np.float64)


def acceptance_probability(target_p: Distribution, draft_q: Distribution, token_id: int) -> float:
    p = _probability(target_p, token_id)
    q = _probability(draft_q, token_id)
    if q <= 0:
        return 1.0 if p > 0 else 0.0
    return min(1.0, p / q)


def residual_distribution(target_p: Distribution, draft_q: Distribution) -> Distribution:
    if isinstance(target_p, SparseDistribution) or isinstance(draft_q, SparseDistribution):
        if isinstance(target_p, SparseDistribution) and isinstance(draft_q, SparseDistribution):
            token_ids = np.union1d(target_p.token_ids, draft_q.token_ids).astype(np.int64)
            residual = np.array(
                [max(target_p.probability(int(token)) - draft_q.probability(int(token)), 0.0) for token in token_ids],
                dtype=np.float64,
            )
            residual = np.where(np.isfinite(residual) & (residual > 0), residual, 0.0)
            keep = residual > 0
            total = residual[keep].sum()
            if not np.isfinite(total) or total <= 0:
                return target_p
            return SparseDistribution(token_ids[keep], residual[keep] / total, _vocab_size(target_p))

        dense_target = _as_dense(target_p)
        dense_draft = _as_dense(draft_q)
        residual = np.maximum(dense_target - dense_draft, 0.0)
        residual = np.where(np.isfinite(residual) & (residual > 0), residual, 0.0)
        total = residual.sum()
        if not np.isfinite(total) or total <= 0:
            residual = np.where(np.isfinite(dense_target) & (dense_target > 0), dense_target, 0.0)
            total = residual.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Cannot build residual distribution from empty target")
        return residual / total

    residual = np.maximum(np.asarray(target_p) - np.asarray(draft_q), 0.0)
    residual = np.where(np.isfinite(residual) & (residual > 0), residual, 0.0)
    total = residual.sum()
    if not np.isfinite(total) or total <= 0:
        dense_target = np.asarray(target_p, dtype=np.float64)
        residual = np.where(np.isfinite(dense_target) & (dense_target > 0), dense_target, 0.0)
        total = residual.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Cannot build residual distribution from empty target")
    return residual / total


def distribution_entropy(distribution: Distribution) -> float:
    """Shannon entropy (nats) of ``distribution`` over its positive support.

    For a top-p / top-k truncated, renormalized target row -- exactly what
    ``distribution_from_logits`` and ``BatchedSparseDistributions.to_distribution``
    produce -- this is the entropy of the *truncated support*, which is the
    quantity Medusa-2 typical acceptance sizes its per-position threshold from
    (Cai et al. 2024, arXiv:2401.10774, Eq. "typical acceptance"). A one-hot /
    empty row has entropy 0.
    """
    if isinstance(distribution, SparseDistribution):
        probs = np.asarray(distribution.probs, dtype=np.float64)
    else:
        probs = np.asarray(distribution, dtype=np.float64)
    probs = probs[np.isfinite(probs) & (probs > 0.0)]
    if probs.size == 0:
        return 0.0
    probs = probs / probs.sum()
    return float(-np.sum(probs * np.log(probs)))


def typical_acceptance_threshold(entropy_value: float, eps: float, delta: float) -> float:
    """Medusa-2 typical-acceptance floor ``min(eps, delta * exp(-H))``.

    ``eps`` is the hard probability floor; ``delta * exp(-H)`` is the
    entropy-scaled floor that drops as the target row gets more uncertain (a
    token needs less mass to be "typical" when the target itself is unsure).
    A draft token is accepted when its target probability exceeds this value.
    """
    return float(min(float(eps), float(delta) * float(np.exp(-float(entropy_value)))))


def typical_accept_decision(
    target_p: Distribution,
    token_id: int,
    *,
    eps: float,
    delta: float,
    entropy_value: float | None = None,
) -> tuple[bool, float, float]:
    """Medusa-2 typical acceptance for one position.

    Returns ``(accepted, threshold, entropy)``. The token is accepted iff the
    target assigns it more mass than the entropy-scaled floor:

        p_target(token) > min(eps, delta * exp(-H(target_p)))

    The decision is DETERMINISTIC -- it consumes no uniform, unlike the exact
    Leviathan-Chen ``min(1, p/q)`` coin. This is *not* a distribution-exact
    sampler: the caller resamples the first non-typical position from
    ``target_p`` itself (not the residual ``(p-q)+``), and the emitted stream is
    gated on task quality, not on matching the target law. Pass ``entropy_value``
    to reuse a precomputed entropy for the row.
    """
    entropy = distribution_entropy(target_p) if entropy_value is None else float(entropy_value)
    threshold = typical_acceptance_threshold(entropy, eps, delta)
    p = _probability(target_p, int(token_id))
    return (p > threshold, threshold, entropy)


def sample_from_distribution(probs: Distribution, rng: np.random.Generator | None = None) -> int:
    rng = rng or np.random.default_rng()
    if isinstance(probs, SparseDistribution):
        return int(rng.choice(probs.token_ids, p=probs.probs))
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(rng.choice(np.arange(probs.shape[0]), p=probs))


@dataclass(frozen=True)
class SpeculativeDecision:
    accepted: bool
    token_id: int
    accept_probability: float


def verify_one_token(
    target_p: np.ndarray,
    draft_q: np.ndarray,
    draft_token: int,
    rng: np.random.Generator | None = None,
) -> SpeculativeDecision:
    rng = rng or np.random.default_rng()
    accept_p = acceptance_probability(target_p, draft_q, draft_token)
    if float(rng.random()) <= accept_p:
        return SpeculativeDecision(True, int(draft_token), accept_p)
    corrected = sample_from_distribution(residual_distribution(target_p, draft_q), rng)
    return SpeculativeDecision(False, corrected, accept_p)


def speculative_output_marginal(target_p: np.ndarray, draft_q: np.ndarray) -> np.ndarray:
    """Return the exact output marginal induced by one-token spec sampling.

    This is a small-distribution correctness oracle. Summing over every possible
    draft token must recover the target distribution when acceptance and
    residual correction are implemented correctly.
    """
    target_p = _as_dense(target_p)
    draft_q = _as_dense(draft_q)
    target_p = target_p / target_p.sum()
    draft_q = draft_q / draft_q.sum()

    out = np.zeros_like(target_p)
    for token_id, q_value in enumerate(draft_q):
        accept_p = acceptance_probability(target_p, draft_q, token_id)
        out[token_id] += q_value * accept_p
        if accept_p < 1.0:
            residual = residual_distribution(target_p, draft_q)
            out += q_value * (1.0 - accept_p) * residual
    return out / out.sum()
