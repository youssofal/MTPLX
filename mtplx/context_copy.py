"""Context-copy (prompt-lookup) speculative drafting for the MTP decode loop.

Opt-in via MTPLX_CONTEXT_COPY=1. When the tail of (prompt + generated) tokens matches an
earlier n-gram, the continuation block is proposed verbatim (uncapped, K up to
MTPLX_CONTEXT_COPY_K) and verified in one forward through the existing capture_commit
verify path — no MTP-head compute for that round. When there is no match, the normal MTP
round runs unchanged. Greedy (temperature<=0) only for now.

Rationale: MTP heads draft novel tokens well (~2x ceiling at depth<=3) but cannot open a
long verbatim window; on grounded/agentic workloads (code edits, file re-emission, RAG)
most output already exists in context, where block-copy verification reaches 4-8x.
The two mechanisms compose: copy when a match exists, MTP otherwise.
"""
import os


def context_copy_enabled() -> bool:
    return (os.environ.get("MTPLX_CONTEXT_COPY") or "").strip() in {"1", "true", "on"}


def context_copy_block_k() -> int:
    try:
        return max(4, int(os.environ.get("MTPLX_CONTEXT_COPY_K") or 24))
    except ValueError:
        return 24


def context_copy_ng_min() -> int:
    try:
        return max(2, int(os.environ.get("MTPLX_CONTEXT_COPY_NGMIN") or 6))
    except ValueError:
        return 6


def context_copy_ng_max() -> int:
    try:
        return max(context_copy_ng_min(), int(os.environ.get("MTPLX_CONTEXT_COPY_NGMAX") or 10))
    except ValueError:
        return 10


def context_copy_min_ext() -> int:
    """Minimum backward match extension (beyond ng_min) required to fire a copy round.
    Short, ng_min-only matches are common in novel generation (loops, boilerplate) and
    have low acceptance — those rounds should stay with the MTP head. Default 2 =
    require an (ng_min+2)-token suffix match before copying."""
    try:
        return max(0, int(os.environ.get("MTPLX_CONTEXT_COPY_MINEXT") or 2))
    except ValueError:
        return 2


class NgramIndex:
    """Incremental ng_min-gram index over the token history: gram -> continuation
    positions. find() is O(candidates) instead of an O(L) backward scan, which
    keeps the proposer off the CPU-bound path at 16-32K contexts."""

    def __init__(self, ng_min: int, ng_max: int, max_candidates: int = 32):
        self.ng_min = ng_min
        self.ng_max = ng_max
        self.max_candidates = max_candidates
        self.grams: dict[tuple, list[int]] = {}
        self.indexed = 0

    def sync(self, history: list[int]) -> None:
        """Index grams ending at positions (self.indexed, len(history)]."""
        for e in range(max(self.indexed + 1, self.ng_min), len(history) + 1):
            self.grams.setdefault(tuple(history[e - self.ng_min:e]), []).append(e)
        self.indexed = len(history)

    def find(self, history: list[int]):
        """Best match: (continuation_pos, extension) or (None, -1). Extension =
        how many tokens beyond ng_min the match runs backwards (0..ng_max-ng_min),
        a free confidence signal (longer suffix match -> longer safe block)."""
        L = len(history)
        if L < self.ng_min + 1:
            return None, -1
        cands = self.grams.get(tuple(history[-self.ng_min:]))
        if not cands:
            return None, -1
        best_pos, best_ext = None, -1
        max_ext = self.ng_max - self.ng_min
        for pos in reversed(cands[-self.max_candidates:]):
            if pos >= L:                       # the trailing gram itself
                continue
            ext = 0                            # longest backward extension wins,
            while (ext < max_ext               # most recent wins ties
                   and pos - self.ng_min - 1 - ext >= 0
                   and history[pos - self.ng_min - 1 - ext]
                   == history[L - self.ng_min - 1 - ext]):
                ext += 1
            if ext > best_ext:
                best_ext, best_pos = ext, pos
                if ext == max_ext:
                    break
        return best_pos, best_ext
