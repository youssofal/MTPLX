"""Sampler and stochastic speculative-decoding helpers.

These utilities are intentionally NumPy based for fast correctness tests. The
runtime path can later swap equivalent MLX kernels behind the same semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SamplerConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20

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
        if np.any(probs < 0):
            raise ValueError("SparseDistribution probabilities must be non-negative")
        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            if token_ids.size > 0:
                token_ids = token_ids[:1]
            else:
                token_ids = np.array([0], dtype=np.int64)
            probs = np.array([1.0], dtype=np.float64)
            total = 1.0
        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "probs", probs / total)

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


def apply_top_p_top_k(probs: np.ndarray, top_p: float = 1.0, top_k: int = 0) -> np.ndarray:
    """Apply the same top-p then top-k order used by local `mlx_lm`.

    Proper speculative sampling requires target and draft probabilities to be
    filtered with exactly the same sampler semantics. Local `mlx_lm` applies
    top-p before top-k, so MTPLX's NumPy reference path mirrors that order.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError("Expected a 1D probability vector")
    mask = np.ones(probs.shape[0], dtype=bool)
    if 0 < top_p < 1.0:
        order = np.argsort(-probs)
        sorted_probs = probs[order]
        cumulative = np.cumsum(sorted_probs)
        keep_sorted = cumulative <= top_p
        if keep_sorted.size:
            keep_sorted[0] = True
            first_over = np.argmax(cumulative >= top_p)
            keep_sorted[: first_over + 1] = True
        nucleus_mask = np.zeros_like(mask)
        nucleus_mask[order[keep_sorted]] = True
        mask &= nucleus_mask
    if top_k and 0 < top_k < probs.shape[0]:
        scoped_probs = np.where(mask, probs, 0.0)
        keep = np.argpartition(-scoped_probs, top_k - 1)[:top_k]
        top_mask = np.zeros_like(mask)
        top_mask[keep] = True
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


def distribution_from_logits(logits: np.ndarray, config: SamplerConfig) -> np.ndarray:
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
            keep = residual > 0
            total = residual[keep].sum()
            if total <= 0 or not np.isfinite(total):
                return target_p
            return SparseDistribution(token_ids[keep], residual[keep] / total, _vocab_size(target_p))

        dense_target = _as_dense(target_p)
        dense_draft = _as_dense(draft_q)
        residual = np.maximum(dense_target - dense_draft, 0.0)
        total = residual.sum()
        if total <= 0 or not np.isfinite(total):
            residual = dense_target.copy()
            total = residual.sum()
        if total <= 0 or not np.isfinite(total):
            raise ValueError("Cannot build residual distribution from empty target")
        return residual / total

    residual = np.maximum(np.asarray(target_p) - np.asarray(draft_q), 0.0)
    total = residual.sum()
    if total <= 0 or not np.isfinite(total):
        residual = np.asarray(target_p, dtype=np.float64).copy()
        total = residual.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Cannot build residual distribution from empty target")
    return residual / total


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
