"""Context-copy (prompt-lookup) speculative drafting for the MTP decode loop.

Enabled by default; MTPLX_CONTEXT_COPY set to 0, false, or off disables it. When the
tail of the generated stream matches an n-gram that occurs in the PROMPT, the prompt
continuation is proposed verbatim as a block (up to MTPLX_CONTEXT_COPY_K tokens, with
shorter blocks for weaker matches) and verified in one forward pass through the
existing capture-commit verify path, so the MTP head is skipped for that cycle. When
there is no match, the normal MTP round runs unchanged. Active only for greedy
decoding (temperature <= 0) without repetition penalties; a temperature path via
sample-and-match verification is a possible follow-up.

Rationale: MTP heads draft novel tokens well but commit at most mtp_depth tokens per
step, and they cannot open a long verbatim window. On grounded workloads (code edits,
file re-emission, RAG) most of the output already exists in the prompt, where a copy
block can commit far more per verify call (see the benchmarks in the pull request).
The two mechanisms compose: copy when a prompt match exists, MTP otherwise.
"""
import os


def context_copy_enabled() -> bool:
    """Enabled by default. MTPLX_CONTEXT_COPY set to 0, false, or off disables it."""
    return (os.environ.get("MTPLX_CONTEXT_COPY") or "").strip() not in {"0", "false", "off"}


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
    Default 0: weak matches are allowed but propose only a SHORT block (see
    block_for_ext), so a wrong incidental match wastes little."""
    try:
        return max(0, int(os.environ.get("MTPLX_CONTEXT_COPY_MINEXT") or 0))
    except ValueError:
        return 0


# Confidence ladder: block length by backward match extension (0..ng_max-ng_min).
# A longer suffix match earns a longer copy block, so a weak match only ever
# risks a short, cheap verify while a strong match copies a full window.
_BLOCK_LADDER = (8, 12, 16, 24, 32)


def block_for_ext(ext: int, k_cap: int) -> int:
    idx = max(0, min(int(ext), len(_BLOCK_LADDER) - 1))
    return min(_BLOCK_LADDER[idx], max(4, k_cap))


class NgramIndex:
    """ng_min-gram index, built once over the prompt at setup: gram -> continuation
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
