"""Context-copy (prompt-lookup) speculative drafting for the MTP decode loop.

Enabled by default; MTPLX_CONTEXT_COPY set to 0, false, or off disables it. When the
tail of the generated stream matches an n-gram that occurs in the PROMPT, the prompt
continuation is proposed verbatim as a block (up to MTPLX_CONTEXT_COPY_K tokens, with
shorter blocks for weaker matches) and verified in one forward pass through the
existing capture-commit verify path, so the MTP head is skipped for that cycle. When
there is no match, the normal MTP round runs unchanged. Active at any temperature:
greedy verifies by argmax match, and sampled decoding uses the same probability-ratio
acceptance as the MTP path (the copy block is a point-mass proposal, so a copied
token is accepted with the target's shaped probability and a rejection emits a
residual sample), which keeps the output law exactly the target sampling
distribution. Requests with repetition penalties fall back to the normal MTP round.

Rationale: MTP heads draft novel tokens well but commit at most mtp_depth tokens per
step, and they cannot open a long verbatim window. On grounded workloads (code edits,
file re-emission, RAG) most of the output already exists in the prompt, where a copy
block can commit far more per verify call (see the benchmarks in the pull request).
The two mechanisms compose: copy when a prompt match exists, MTP otherwise.

--- FILE MODIFIED (Apache License 2.0, section 4(b) notice) ---------------------
This file has been changed from the upstream MTPLX release to add RAMP: a
fixed/long block-length policy plus a mismatch-tolerant fuzzy re-anchor fallback
for when the exact n-gram key misses. RAMP is OFF BY DEFAULT (MTPLX_RAMP_ENABLED
unset or falsy) and, when off, every code path below reduces byte-for-byte to the
unmodified upstream behaviour -- see RampIndex.find() and block_for_ext(). See
README.md's "RAMP" section and docs/ramp/ in this fork for the measured evidence
behind the defaults (block=48, fuzzy=on): +45.9% to +71.4% decode throughput on
real repetitive/edit-shaped coding-agent workloads (128K context, temperature-0
output byte-identical to stock); roughly neutral-to-slightly-negative on
open-ended, non-repetitive generation (dark_fraction ~0.92), which is exactly why
it stays off by default rather than becoming the new default policy.
-----------------------------------------------------------------------------
"""
import os


def context_copy_enabled() -> bool:
    """Enabled by default. MTPLX_CONTEXT_COPY set to 0, false, or off disables it."""
    return (os.environ.get("MTPLX_CONTEXT_COPY") or "").strip() not in {"0", "false", "off"}


def context_copy_target_prefix_enabled() -> bool:
    """Opt-in: run context-copy on the target_prefix lane (default OFF, so the
    shipped/PR behaviour is byte-unchanged).

    On this lane context-copy is a DRAFT SOURCE, not a block-round engine: a
    prompt n-gram match starts a streak that feeds the copy continuation as
    the depth-1 draft, so every forward keeps the lane's 2-row verify
    geometry and the emitted stream is bit-exact to pure AR for any draft
    source at any temperature (the accepted token is always the pre-sampled
    target id).  Block rounds -- whose T+1-row forwards leave M>2 kernel-path
    ulps in retained cache rows and break AR-exactness -- remain
    capture_commit-only.

    The COMPILED K1 route keeps its device-draft (R1) contract, so when this
    flag takes over, the compiled route STEPS ASIDE (like the
    grammar-constraint case) and the request runs the non-compiled
    target_prefix lane.  The flag drives the lane switch REGARDLESS of
    whether streaks fire, so flag-on + MTPLX_CONTEXT_COPY=0 is a clean
    same-lane baseline.

    Precedence: whole-MoE fusion needs the compiled route, and repetition
    penalties disable context-copy; in both cases the compiled route is KEPT
    and this flag is inert -- mirrored in the exact_a3b_target_prefix_factory
    gate and the ccopy_active gate.
    """
    return (os.environ.get("MTPLX_CONTEXT_COPY_TARGET_PREFIX") or "").strip() in {
        "1",
        "true",
        "on",
    }


# ---------------------------------------------------------------------------
# RAMP -- off by default. See the file-level notice above.
# ---------------------------------------------------------------------------


def ramp_enabled() -> bool:
    """Master switch. Default OFF: every measured RAMP number is short-of-target
    evidence for open-ended (non-repetitive) generation, so this must stay an
    explicit opt-in, not a new default."""
    return (os.environ.get("MTPLX_RAMP_ENABLED") or "").strip() in {"1", "true", "on"}


def ramp_block() -> int | None:
    """Fixed block length in tokens when RAMP is enabled. 0 (or unset) keeps the
    upstream confidence ladder (_BLOCK_LADDER) instead of a fixed length."""
    try:
        v = int(os.environ.get("MTPLX_RAMP_BLOCK") or 48)
    except ValueError:
        v = 48
    return v if v > 0 else None


def ramp_fuzzy_enabled() -> bool:
    """Mismatch-tolerant short-anchor fallback for when the exact ng_min-gram key
    misses. Only a net win COMBINED WITH a long block (measured: short block +
    fuzzy is worse than doing nothing) -- do not enable without also setting a
    long MTPLX_RAMP_BLOCK."""
    return (os.environ.get("MTPLX_RAMP_FUZZY") or "1").strip() not in {"0", "false", "off"}


def ramp_anchor_len() -> int:
    try:
        return max(1, int(os.environ.get("MTPLX_RAMP_ANCHOR_LEN") or 3))
    except ValueError:
        return 3


def ramp_max_fuzzy_candidates() -> int:
    try:
        return max(1, int(os.environ.get("MTPLX_RAMP_MAX_FUZZY_CANDIDATES") or 8))
    except ValueError:
        return 8


def ramp_similarity_span() -> int:
    try:
        return max(1, int(os.environ.get("MTPLX_RAMP_SIMILARITY_SPAN") or 24))
    except ValueError:
        return 24


class _RampFuzzyAnchor:
    """Mismatch-tolerant short-anchor fallback, consulted only after the exact
    ng_min-gram index misses. Anchors on a shorter `anchor_len`-gram and ranks
    candidates by backward similarity against the corpus (the prompt), so a
    context that diverges in one token (e.g. a renamed identifier) can still
    anchor -- the exact key goes dark at every such divergence."""

    def __init__(self, anchor_len: int, max_candidates: int, similarity_span: int) -> None:
        self.anchor_len = anchor_len
        self.max_candidates = max_candidates
        self.similarity_span = similarity_span
        self.anchors: dict[tuple, list[int]] = {}
        self.indexed = 0

    def sync(self, history: list[int]) -> None:
        a = self.anchor_len
        for e in range(max(self.indexed + 1, a), len(history) + 1):
            self.anchors.setdefault(tuple(history[e - a:e]), []).append(e)
        self.indexed = len(history)

    def _similarity(self, pos: int, history: list[int], corpus: list[int]) -> float:
        hits = total = 0
        for d in range(1, self.similarity_span + 1):
            i, j = pos - d, len(history) - d
            if i < 0 or j < 0:
                break
            total += 1
            if corpus[i] == history[j]:
                hits += 1
        return hits / total if total else 0.0

    def find(self, history: list[int], corpus: list[int], *, max_pos: int | None = None):
        a = self.anchor_len
        if len(history) < a + 1:
            return None
        cands = self.anchors.get(tuple(history[-a:]))
        if not cands:
            return None
        limit = max_pos if max_pos is not None else len(corpus)
        best_pos, best_sim = None, -1.0
        for p in reversed(cands[-self.max_candidates * 8:]):
            if p >= limit or p >= len(history):
                continue
            sim = self._similarity(p, history, corpus)
            if sim > best_sim:
                best_pos, best_sim = p, sim
        return best_pos


def context_copy_block_k() -> int:
    try:
        base = max(4, int(os.environ.get("MTPLX_CONTEXT_COPY_K") or 24))
    except ValueError:
        base = 24
    fixed = ramp_block() if ramp_enabled() else None
    if fixed is not None:
        # block_for_ext(ext, ccopy_k) receives this value as its cap -- the stock
        # cap (24) would silently re-clamp a longer RAMP block if this weren't
        # widened too.
        return max(base, fixed)
    return base


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
    if ramp_enabled():
        fixed = ramp_block()
        if fixed is not None:
            return fixed
    idx = max(0, min(int(ext), len(_BLOCK_LADDER) - 1))
    return min(_BLOCK_LADDER[idx], max(4, k_cap))


class NgramIndex:
    """ng_min-gram index, built once over the prompt at setup: gram -> continuation
    positions. find() is O(candidates) instead of an O(L) backward scan, which
    keeps the proposer off the CPU-bound path at 16-32K contexts.

    When RAMP is enabled (ramp_enabled()), a miss on the exact index falls
    through to a mismatch-tolerant fuzzy re-anchor before returning (None, -1).
    When RAMP is disabled -- the default -- this class is byte-for-byte the
    upstream NgramIndex; the fuzzy machinery is not even constructed."""

    def __init__(self, ng_min: int, ng_max: int, max_candidates: int = 32):
        self.ng_min = ng_min
        self.ng_max = ng_max
        self.max_candidates = max_candidates
        self.grams: dict[tuple, list[int]] = {}
        self.indexed = 0
        self._ramp_fuzzy: _RampFuzzyAnchor | None = None
        self._ramp_corpus: list[int] = []
        if ramp_enabled() and ramp_fuzzy_enabled():
            self._ramp_fuzzy = _RampFuzzyAnchor(
                max(1, min(ramp_anchor_len(), ng_min - 1)),
                ramp_max_fuzzy_candidates(),
                ramp_similarity_span(),
            )

    def sync(self, history: list[int]) -> None:
        """Index grams ending at positions (self.indexed, len(history)]."""
        for e in range(max(self.indexed + 1, self.ng_min), len(history) + 1):
            self.grams.setdefault(tuple(history[e - self.ng_min:e]), []).append(e)
        self.indexed = len(history)
        if self._ramp_fuzzy is not None:
            self._ramp_fuzzy.sync(history)
            self._ramp_corpus = list(history)

    def find(self, history: list[int], *, max_pos: int | None = None):
        """Best match: (continuation_pos, extension) or (None, -1). Extension =
        how many tokens beyond ng_min the match runs backwards (0..ng_max-ng_min),
        a free confidence signal (longer suffix match -> longer safe block).
        max_pos (exclusive) drops candidates with no continuation left in the
        indexed region, so the best VALID match wins rather than a boundary
        match being selected and then discarded by the caller."""
        L = len(history)
        pos, ext = self._find_exact(history, L, max_pos=max_pos)
        if pos is not None:
            return pos, ext
        if self._ramp_fuzzy is not None:
            fpos = self._ramp_fuzzy.find(history, self._ramp_corpus, max_pos=max_pos)
            if fpos is not None:
                # A fuzzy hit has no exact backward extension to report; ext=0
                # is the confidence signal into block_for_ext (bypassed anyway
                # when a fixed RAMP block is configured).
                return fpos, 0
        return None, -1

    def _find_exact(self, history: list[int], L: int, *, max_pos: int | None = None):
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
            if max_pos is not None and pos >= max_pos:
                continue                       # no prompt continuation to copy
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
