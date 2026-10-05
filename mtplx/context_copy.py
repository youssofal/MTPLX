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

AR-only runtimes (no MTP head) can opt in to the same lane in generate_ar with
MTPLX_CONTEXT_COPY_AR=1 (see context_copy_ar.py); it reuses NgramIndex,
block_for_ext and CopyGovernor from here.

RAMP (adapting community PR #375 by @johninthewinter): an opt-in fixed/long
block-length policy plus a mismatch-tolerant fuzzy re-anchor fallback for when
the exact n-gram key misses. OFF BY DEFAULT (MTPLX_RAMP_ENABLED unset or
falsy); when off, every code path below reduces byte-for-byte to the prior
behavior — NgramIndex.find() and block_for_ext() — and the fuzzy machinery is
not even constructed. The author's receipts behind the defaults (block=48,
fuzzy=on): +45.9% to +71.4% decode throughput on repetitive/edit-shaped
coding-agent workloads (Qwen3.8-27B, up to 128K context, temperature-0 output
byte-identical to stock), roughly neutral-to-slightly-negative on open-ended
generation (the exact key goes dark on ~0.92 of novel output) — which is why
it stays an explicit opt-in rather than a new default. Both proposer entry
points are shared, so the batched (qwen4_exp) copy lane inherits the policy
through the same two functions.
"""
import os


def context_copy_enabled() -> bool:
    """Enabled by default. MTPLX_CONTEXT_COPY set to 0, false, or off disables it."""
    return (os.environ.get("MTPLX_CONTEXT_COPY") or "").strip() not in {"0", "false", "off"}


def context_copy_ar_enabled() -> bool:
    """Context-copy for AR-only runtimes (no MTP head), in ``generate_ar``.

    Opt-in: MTPLX_CONTEXT_COPY_AR=1 (also true/on/yes), and only while the
    global MTPLX_CONTEXT_COPY switch is not off. Default off: plain AR decoding
    stays byte-identical to before. The copy block is verified with one
    multi-row forward, whose batched numerics can differ from row-by-row
    decoding in the last bits, so a greedy stream is not guaranteed
    bit-identical to the one-token loop.
    """
    value = (os.environ.get("MTPLX_CONTEXT_COPY_AR") or "").strip().lower()
    return context_copy_enabled() and value in {"1", "true", "on", "yes"}


def context_copy_batched_enabled() -> bool:
    """Context-copy block rounds on the BATCHED verify lane (default ON;
    MTPLX_CONTEXT_COPY_BATCHED=0 disables).

    The batched lane (qwen4_exp / Flash-Next) never had the copy mechanic:
    the gate below predates the family and only knew the capture-commit and
    target_prefix lanes, so grounded re-emission (code edits, file rewrites)
    decoded at plain MTP depth while the 27B's capture lane copied 24-token
    blocks (receipted 2026-08-28: a 33.5k-token full-file rewrite ran at
    29.8 tok/s with MTP acceptance 0.89/0.78/0.69 — every token of it
    already sat in the prompt).

    On this lane a block round forwards [primary]+block through the normal
    batched verify forward and commits through the family capture-commit
    (row-count generic) or the rollback+re-forward fallback, so the lane's
    exactness contract is unchanged: acceptance is the same point-mass
    probability-ratio rule the MTP path uses, at any temperature.
    """
    return (os.environ.get("MTPLX_CONTEXT_COPY_BATCHED") or "").strip() not in {
        "0",
        "false",
        "off",
    }


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


class JoinedTokens:
    """Prompt plus generated tokens as one read-only sequence.

    The copy-lane probe runs once per decode round and only reads the last few
    tokens and a handful of prompt positions. Building ``prompt_ids + tokens``
    for it copied the whole prompt every round: about a millisecond of host
    time per round at 131K tokens, inside the gap where the GPU waits for the
    next round. Indexing and slicing return the same values the concatenated
    list would.
    """

    __slots__ = ("_head", "_tail", "_split")

    def __init__(self, head: list[int], tail: list[int]) -> None:
        self._head = head
        self._tail = tail
        self._split = len(head)

    def __len__(self) -> int:
        return self._split + len(self._tail)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("JoinedTokens index out of range")
        if index < self._split:
            return self._head[index]
        return self._tail[index - self._split]


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


def context_copy_probation_k() -> int:
    """Block cap while the copy lane is still unproven in this generation.

    Misfired copy rounds verify 16-24-token blocks — a ~4x-cost forward vs
    the normal q=4 round — and the acceptance EMA needs several rounds to
    arm a suspension, so short coding-agent turns re-paid that tuition
    every turn (2026-08-29 receipts: turn-long verify 60-88 ms/round at 8k
    ctx with 21/96 copy tokens accepted; the same lane is a +16.7% win on
    long-context re-emission). Until the EMA proves the content pays,
    blocks stay this size; a proven lane opens to the full
    MTPLX_CONTEXT_COPY_K. Winners lose at most two short blocks of upside;
    losers pay a quarter of the old misfire cost.
    """
    try:
        return max(2, int(os.environ.get("MTPLX_CONTEXT_COPY_PROBATION_K") or 8))
    except ValueError:
        return 8


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


class CopyGovernor:
    """Acceptance EMA, probation and suspend/backoff of a copy lane.

    The same policy the MTP decode loops keep inline: blocks stay at the
    probation size until two sampled rounds hold the 0.5 starting EMA, an EMA
    below 0.35 after three rounds suspends copying for ``backoff`` tokens, and
    the backoff doubles on each suspension (reset to 64 once a round accepts at
    least half its block).
    """

    def __init__(self) -> None:
        self.ema = 0.5
        self.seen = 0
        self.suspend_until = 0
        self.backoff = 64
        self.suspensions = 0

    def suspended(self, n_tokens: int) -> bool:
        return n_tokens < self.suspend_until

    def block_cap(self, k: int, probation_k: int) -> int:
        """Full block cap once proven, the probation cap before that."""
        return k if (self.seen >= 2 and self.ema >= 0.5) else probation_k

    def record(self, accepted: int, block_len: int, n_tokens: int) -> None:
        """Fold in one verified block; ``n_tokens`` counts the stream after it."""
        self.ema = 0.7 * self.ema + 0.3 * (accepted / block_len)
        self.seen += 1
        if accepted / block_len >= 0.5:
            self.backoff = 64
        if self.seen >= 3 and self.ema < 0.35:
            self.suspend_until = n_tokens + self.backoff
            self.backoff = min(self.backoff * 2, 4096)
            self.ema, self.seen = 0.5, 0
            self.suspensions += 1


def ladder_widths(k_caps) -> frozenset[int]:
    """Verify widths (primary plus block) of every full-length block these caps allow.

    The widths a copy round forwards unless its block was cut short (by the
    request's token budget, the end of the prompt, or a grammar): with the
    default caps (24, and 8 in probation) they are 9, 13, 17 and 25.
    """

    return frozenset(
        1 + block_for_ext(ext, cap)
        for cap in k_caps
        for ext in range(len(_BLOCK_LADDER))
    )


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
