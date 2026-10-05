"""Context-copy drafting for AR-only runtimes (no MTP head).

``generate_ar`` normally commits one token per forward pass. With
MTPLX_CONTEXT_COPY_AR=1 it borrows the context-copy lane of the MTP loops: when
the tail of the stream matches an n-gram of the PROMPT, the prompt continuation
is proposed as a block and verified in ONE multi-row target forward pass.

Everything that decides what to copy is shared with the MTP lanes
(``NgramIndex``, ``block_for_ext``, the probation/EMA/suspend policy in
``CopyGovernor`` and the MTPLX_CONTEXT_COPY_* switches). What is new here is the
verify step for a plain AR model:

* the target runs on ``[token, *block]`` (``token`` is the sampled-but-not-yet-
  forwarded token), so row ``i`` scores ``block[i]`` and row ``len(block)``
  holds the logits after the whole block;
* greedy decoding accepts the longest prefix matching the target argmax;
* sampled decoding accepts each copied token with the target's shaped
  probability and draws the first rejection's correction from the residual
  (point-mass proposal), so the output law stays the target distribution;
* the KV cache is trimmed back to the accepted prefix, and a stop token inside
  the accepted prefix ends the block (it stays out of the cache, as after a
  normal AR step).

The prompt is the only source of blocks (never the generated text), as in the
MTP lanes. Requests with penalties, constraints, guards or hidden-state capture
do not use the lane.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .context_copy import (
    CopyGovernor,
    JoinedTokens,
    NgramIndex,
    block_for_ext,
    context_copy_ar_enabled,
    context_copy_block_k,
    context_copy_min_ext,
    context_copy_ng_max,
    context_copy_ng_min,
    context_copy_probation_k,
)

# Sampled acceptance: (block_logits[len(block), vocab], block) -> (accepted, correction)
AcceptFn = Callable[[Any, Sequence[int]], "tuple[int, int | None]"]


@dataclass(frozen=True)
class CopyRoundResult:
    tokens: list[int]            # accepted block prefix, cut after a stop token
    correction: int | None       # residual sample at the first rejection (sampled only)
    stopped: bool                # the last token in ``tokens`` is a stop token
    logits: Any                  # (1, vocab): logits after the committed block
    block_len: int


def copy_ar_eligible(
    *,
    mtp_enabled: bool,
    penalized: bool,
    guarded: bool,
    sync_eval: bool,
    cache: Any,
) -> bool:
    """Whether this ``generate_ar`` request may use the copy lane."""
    if mtp_enabled or penalized or guarded or not sync_eval:
        return False
    if not context_copy_ar_enabled():
        return False
    return bool(cache) and all(
        hasattr(entry, "offset") and callable(getattr(entry, "trim", None))
        for entry in cache
    )


def accepted_prefix_greedy(block: Sequence[int], target_ids: Sequence[int]) -> int:
    """Longest prefix of ``block`` equal to the target argmax rows."""
    accepted = 0
    for proposed, chosen in zip(block, target_ids):
        if int(proposed) != int(chosen):
            break
        accepted += 1
    return accepted


class ArContextCopy:
    """Prompt n-gram index plus the shared acceptance governor, per request."""

    def __init__(self, prompt_ids: list[int]) -> None:
        self.prompt_ids = prompt_ids
        self.k = context_copy_block_k()
        self.probation_k = min(self.k, context_copy_probation_k())
        self.min_ext = context_copy_min_ext()
        self.index = NgramIndex(context_copy_ng_min(), context_copy_ng_max())
        self.index.sync(prompt_ids)
        self.governor = CopyGovernor()
        self.probes = self.rounds = self.drafted = self.accepted = 0

    def propose(self, tokens: list[int], budget: int) -> list[int]:
        """Prompt continuation to copy after ``tokens`` (at most ``budget`` long)."""
        if budget < 1 or self.governor.suspended(len(tokens)):
            return []
        self.probes += 1
        pos, ext = self.index.find(
            JoinedTokens(self.prompt_ids, tokens), max_pos=len(self.prompt_ids)
        )
        if pos is None or ext < self.min_ext:
            return []
        cap = self.governor.block_cap(self.k, self.probation_k)
        length = min(block_for_ext(ext, cap), budget)
        return [int(t) for t in self.prompt_ids[pos : pos + length]]

    def record(self, block_len: int, accepted: int, n_tokens: int) -> None:
        self.rounds += 1
        self.drafted += block_len
        self.accepted += accepted
        self.governor.record(accepted, block_len, n_tokens)

    def summary(self) -> dict[str, Any]:
        return {
            "probes": self.probes,
            "rounds": self.rounds,
            "drafted_tokens": self.drafted,
            "accepted_tokens": self.accepted,
            "acceptance_rate": self.accepted / self.drafted if self.drafted else 0.0,
            "suspensions": self.governor.suspensions,
        }


def verify_copy_block(
    forward: Callable[[Any], Any],
    trim_cache: Callable[[int], bool],
    cache_offset: int,
    token: int,
    block: Sequence[int],
    stop_token_ids: set[int] | None,
    *,
    greedy: bool,
    accept: AcceptFn | None = None,
) -> CopyRoundResult:
    """One target pass over ``[token, *block]``; commit the accepted prefix.

    ``forward(ids)`` runs the target on a ``(1, n)`` int array and returns logits
    ``(1, n, vocab)``; ``trim_cache(offset)`` rolls the KV cache back to that
    length and reports success. ``accept`` is required when not ``greedy``.
    The cache ends holding every committed token except the last, as after a
    normal AR step (``token`` and the accepted block, minus a final stop token).
    """
    import mlx.core as mx

    ids = mx.array([[int(token), *[int(t) for t in block]]])
    logits = forward(ids)
    correction: int | None = None
    if greedy:
        chosen = mx.argmax(logits[0], axis=-1)
        mx.eval(chosen)
        accepted = accepted_prefix_greedy(block, chosen.tolist())
    else:
        if accept is None:
            raise ValueError("sampled copy verification needs an accept function")
        accepted, correction = accept(logits[0, : len(block)], block)
    stopped = False
    stops = stop_token_ids or set()
    for i in range(accepted):
        if int(block[i]) in stops:
            accepted, stopped, correction = i + 1, True, None
            break
    keep = cache_offset + 1 + accepted - (1 if stopped else 0)
    if not trim_cache(keep):
        raise RuntimeError("context-copy (AR): KV cache rollback failed")
    return CopyRoundResult(
        tokens=[int(t) for t in block[:accepted]],
        correction=correction,
        stopped=stopped,
        logits=logits[:, accepted, :],
        block_len=len(block),
    )
