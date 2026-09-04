"""Warm-prefix state reuse for MTPLX target prefill.

SessionBank is deliberately conservative in this first version: it stores
exact token-prefix entries in memory, restores cloned cache state into a fresh
runtime cache, then forwards only the suffix tokens. The benchmark gate compares
the warm result against a cold full prefill before any generation path uses it.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable

import mlx.core as mx
import numpy as np

from .cache_state import (
    CacheSnapshot,
    _clone_tree,
    _is_trimmable,
    restore_cache,
    snapshot_cache,
    snapshot_cache_lazy_hybrid,
)
from .cache_bank.codec import ColdEncodeInterrupted
from .runtime import MTPLXRuntime
from .runtime_options import block_prefix_restore_enabled


def _policy_uses_committed_history(policy: str | None) -> bool:
    """Mirror generation._mtp_history_uses_committed_cache exactly.

    (Normalization mirrors generation._normalize_mtp_history_policy: lower,
    strip, dashes to underscores, aliases full/lastwindow/window.)
    """
    normalized = (policy or "cycle").strip().lower().replace("-", "_")
    normalized = {
        "full": "committed",
        "lastwindow": "last_window",
        "window": "last_window",
    }.get(normalized, normalized)
    return normalized in {"committed", "last_window"}


def _restore_identity_compatible(
    entry: "SessionBankEntry",
    *,
    model_path: str | None,
    mtp_enabled: bool | None,
    hidden_variant: str | None,
    template_hash: str | None,
    mtp_history_policy: str | None,
    draft_head_identity: str | None,
    policy_fingerprint: str | None,
) -> bool:
    """Mirror restore()'s identity gates (None parameter = wildcard)."""
    if model_path is not None and entry.model_path != str(model_path):
        return False
    if mtp_enabled is not None and bool(entry.mtp_enabled) != bool(mtp_enabled):
        return False
    if hidden_variant is not None and entry.hidden_variant != hidden_variant:
        return False
    if template_hash is not None and entry.template_hash != template_hash:
        return False
    if mtp_history_policy is not None and not _mtp_history_policy_compatible(
        entry.mtp_history_policy, mtp_history_policy
    ):
        return False
    if (
        draft_head_identity is not None
        and entry.draft_head_identity != draft_head_identity
    ):
        return False
    if (
        policy_fingerprint is not None
        and entry.policy_fingerprint != policy_fingerprint
    ):
        return False
    return True


def _lazy_snapshot_enabled() -> bool:
    """Zero-copy KV snapshots at commit (kvcache-v2). Off-switch only."""
    raw = str(os.environ.get("MTPLX_SESSION_LAZY_SNAPSHOT", "1")).strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _snapshot_settle_enabled() -> bool:
    """Idle-lane owner-copy settling of the lazy snapshot after a put.

    Motivation: every unevaluated snapshot view holds a reference to a
    live cache buffer, blocking donation, so the next turn's first write
    pays a full COW divergence copy (measured: first slice_update with an
    alias alive = 66 ms/GB + doubled memory; plain mx.eval of a
    full-range view ALIASES and releases nothing, so only owner copies
    decouple).

    DEFAULT OFF — falsified as a default by the 2026-08-30 phase-3 A/B
    (settle_on/settle_off x2, warm 91K turns): stall magnitude is
    dominated by idle-lane/SSD scheduling nondeterminism (the next
    request queues behind multi-GB cold encodes), and adding the settle
    copy to that lane produced the worst observed stall (27.6 s) instead
    of removing the class. Kept as an opt-in instrument; the structural
    fix for the stall class is the #391-style fixed-capacity banks (no
    per-turn multi-GB snapshot at all) plus a preemptible idle lane.
    """
    raw = str(os.environ.get("MTPLX_SESSION_SNAPSHOT_SETTLE", "0")).strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _near_prefix_tiny_gap_limit() -> int:
    """Token gap treated as tokenizer-boundary drift (long-shipped tolerance)."""
    raw = os.environ.get("MTPLX_SESSION_NEAR_PREFIX_MAX_TOKEN_GAP")
    try:
        return max(0, int(str(raw).strip())) if raw is not None else 8
    except (TypeError, ValueError):
        return 8


def _boundary_true_restore_enabled() -> bool:
    """Fail-closed recurrent-boundary restores (kvcache-v2). Off-switch only.

    When ON, a sub-prefix restore of an entry whose model carries recurrent
    (non-trimmable) state requires a stored recurrent boundary at or below the
    match point; without one the entry is skipped instead of silently reusing
    recurrent state from a later boundary (the pre-v2 behavior that Desktop QA
    observed "degrading the visible answer").
    """
    raw = str(os.environ.get("MTPLX_SESSION_BOUNDARY_TRUE_RESTORE", "1")).strip().lower()
    return raw not in {"0", "false", "off", "no"}


SESSION_BANK_SHED_BOUNDARIES_ENV = "MTPLX_SESSION_BANK_SHED_BOUNDARIES"


def _shed_boundaries_enabled() -> bool:
    """Shed GDN boundary records to fit, instead of dropping the whole entry.

    THE BUG, measured 2026-09-01 (PR #391 receipts/ttft/control.json).
    A three-scenario TTFT screen on Qwen3.8 Flash-Next: a 19,022-token cold
    turn, the same conversation with the model's own reply appended (0.217 s
    visible TTFT, exact restore), and the same turn with the prior assistant
    message re-rendered -- which took **15.79 s, cached=0, cold, on all three
    repeats**. A 70x cliff on exactly the traffic shape the near-prefix lane
    exists for.

    The chain, from the receipt:

    * The bank auto-sized to its 1 GiB FLOOR ("session-bank budget: 1.0G total
      (auto: machine memory plan...), model weights 107.1G" -- the same
      resolution production gets on this box).
    * A 19K-token entry's base snapshot is ~711 MB. Its GDN boundary records
      cost ~87-101 MB EACH, and MTPLX_GDN_BOUNDARY_MAX is 8, so the boundary
      payload alone is ~700-810 MB.
    * ``put`` counts that payload into ``entry_nbytes`` and then refuses the
      ENTIRE entry: eviction_log shows ``skipped_oversized_snapshot`` at
      nbytes=1,398,321,776 and 1,520,850,304 against a 1,073,741,824 budget --
      while the same turn's boundary-LESS commit (710,255,120) was admitted.
    * So the bank only ever held boundary-less entries. The one survivor in the
      receipt reports ``gdn_boundaries: []``.
    * ``recurrent_boundary_at_or_below()`` returns None on such an entry, so
      ``_restore_near_prefix_prompt_state`` rejects every candidate
      (``boundary_not_better:0``), the request falls through to ``restore()``,
      the SSD lookup misses, and the response reports ``ssd_prefix_miss`` --
      which MASKS the real RAM-lane reason and is why this read as an SSD
      problem rather than an admission one.

    Boundary records are the *sheddable* part of an entry: they are pure
    acceleration, reconstructible by re-prefill, and an entry with FEWER
    boundaries still serves every restore an entry with none can. Dropping the
    entry to protect the budget therefore trades a 0.5 s restore for a 15.8 s
    cold prefill in order to save bytes it could have saved by keeping three
    fewer records.

    With shedding, the prompt-boundary entry is admitted with as many records
    as fit; ``put``'s existing ``prefix_donor`` inheritance then carries those
    records onto the generation-final commit, ``_supersede_contained_prefixes``
    collapses the pair (a boundary-carrying container legitimately dominates
    its contained prefixes), and the single entry the 1 GiB budget allows is a
    boundary-carrying one.

    Note the corollary: under this bug, raising MTPLX_GDN_BOUNDARY_MAX from 8 --
    the audit's "cheapest lever" -- makes things WORSE, because it enlarges the
    payload that triggers the refusal.

    Default OFF; read at SessionBank construction (see __init__) so one bank
    keeps one admission policy for its whole life and an A/B arm cannot drift
    mid-run.
    """
    raw = str(os.environ.get(SESSION_BANK_SHED_BOUNDARIES_ENV, "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


SESSION_BANK_PROTECTED_TERMINAL_ENV = "MTPLX_SESSION_BANK_PROTECTED_TERMINAL"


def _protected_terminal_enabled() -> bool:
    """Protected-terminal eviction order (oMLX PR #3330, exact_resident.py).

    Ported policy, in oMLX's words: two candidates compete under one byte
    ceiling -- the longer *matching terminal* (a banked turn that strictly
    extends the incoming prompt) and the shorter *input-prompt fallback* (the
    prompt itself, being published now). Publishing the fallback must never be
    what evicts the terminal that extends it: the terminal can serve every
    restore the fallback can (exact hits trim; boundary-true restores pick a
    boundary <= the matched point) plus the tail the fallback cannot, so
    trading it for the fallback strictly loses coverage. oMLX keeps the NEWEST
    such terminal, deliberately not letting length override insertion recency
    (`_newest_extending_entry`, exact_resident.py:87-103), and counts the
    deflections (`protected_rejections`, exact_resident.py:189-215).

    This is an eviction ORDER change and nothing else. It adds no byte ceiling
    (effective_max_bytes() and the per-session cap already exist and are
    model-aware), no background work, and no idle lane -- see the phase-3
    falsification recorded at _snapshot_settle_enabled above and in commit
    b5fac4ac: on this box, adding work to the idle lane produced the WORST
    observed stall (27.6 s) across a warm 91K-turn A/B.

    Default OFF; read at SessionBank construction (see __init__), so one bank
    keeps one policy for its whole life and an A/B arm cannot drift mid-run.
    """
    raw = str(os.environ.get(SESSION_BANK_PROTECTED_TERMINAL_ENV, "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


GIB = 1024**3
# Tool sessions store ~3 entries per turn (prompt-prefix commit, postcommit,
# generation-final); an 8-entry cap churned the whole bank every ~3 turns and
# pushed warm boundary-carrying entries to the SSD tier, which does not yet
# persist recurrent boundaries — the next divergent turn then fail-closed to
# a stale short prefix (#121, measured 2026-07-16). Memory stays bounded by
# max_bytes; the count cap only bounds scan cost.
DEFAULT_MAX_ENTRIES = 24
DEFAULT_MAX_BYTES = 24 * GIB
DEFAULT_PER_SESSION_MAX_BYTES = 8 * GIB
DEFAULT_IDLE_TTL_S = 60 * 60
DEFAULT_PREFIX_BLOCK_SIZE = 256
DEFAULT_BLOCK_PREFIX_MIN_MATCH_TOKENS = 512
DEFAULT_ACTIVE_SESSION_PIN_TTL_S = 600.0
DEFAULT_PER_SESSION_MAX_ENTRIES = 3


def _per_session_max_entries() -> int:
    """Retention cap on RAM entries per session (count, not bytes).

    2026-08-01 live leak: agent turns whose canonical transcripts diverge
    mid-stream (scoped thinking / tool-call rendering) bank one ~1.7GB
    sibling per turn that is NOT a strict prefix of the next — supersede
    never fires, so one OpenCode session accumulated 5 near-duplicate
    snapshots (8.8GB) inside its byte budget and allocator pressure added
    25-40ms/tick to every verify call. Newest-K retention bounds that while
    keeping a couple of older boundary entries for divergent restores.
    0 disables (byte budgets alone).
    """
    raw = os.environ.get("MTPLX_SESSION_BANK_PER_SESSION_MAX_ENTRIES")
    if raw is None or not str(raw).strip():
        return DEFAULT_PER_SESSION_MAX_ENTRIES
    try:
        return max(0, int(str(raw).strip()))
    except (TypeError, ValueError):
        return DEFAULT_PER_SESSION_MAX_ENTRIES


def _active_session_pin_ttl_s() -> float:
    """Sessions that touched the bank within this window are eviction-last.

    2026-07-31 live incident: a long coding session's warm entry was
    LRU-evicted by cross-session pressure mid-run, forcing an 85.6k-token
    full re-prefill on the very next turn. Recently-active sessions are
    exactly the ones about to be extended, so cross-session eviction now
    prefers idle victims. Activity-TTL rather than explicit pin/unpin so a
    cancelled or crashed request can never leak a pin. 0 disables.
    """
    raw = os.environ.get("MTPLX_SESSION_BANK_ACTIVE_PIN_TTL_S")
    if raw is None or not str(raw).strip():
        return DEFAULT_ACTIVE_SESSION_PIN_TTL_S
    try:
        return max(0.0, float(str(raw).strip()))
    except (TypeError, ValueError):
        return DEFAULT_ACTIVE_SESSION_PIN_TTL_S


class CacheMissReason(str, Enum):
    NEW_SESSION = "new_session"
    PREFIX_DIVERGENCE_AT_TOKEN = "prefix_divergence_at_token"
    MODEL_MISMATCH = "model_mismatch"
    TEMPLATE_MISMATCH = "template_mismatch"
    POLICY_MISMATCH = "policy_mismatch"
    EVICTED = "evicted"
    BACKGROUND_BYPASS = "background_bypass"
    SESSION_BUSY = "session_busy"
    SNAPSHOT_DESYNC = "snapshot_desync"
    NO_SNAPSHOT_COVERAGE = "no_snapshot_coverage"


def token_prefix_hash(token_ids: list[int] | tuple[int, ...]) -> str:
    h = hashlib.sha256()
    for token in token_ids:
        h.update(int(token).to_bytes(8, byteorder="little", signed=True))
    return h.hexdigest()


def common_prefix_len(left: list[int] | tuple[int, ...], right: list[int] | tuple[int, ...]) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if int(left[index]) != int(right[index]):
            return index
    return limit


def block_aligned_prefix_len(matched_tokens: int, *, block_size: int) -> int:
    block = max(1, int(block_size))
    matched = max(0, int(matched_tokens))
    return (matched // block) * block


# Policies that share the committed-mtp-cache representation. An entry stored
# under any of these policies can be safely reused for a lookup that requests
# any other policy in this set, because the cache snapshot shape is identical
# (``last_window`` is just a runtime trim of the same committed cache).
_COMMITTED_CACHE_POLICIES = frozenset({"committed", "last_window"})


def _mtp_history_policy_compatible(
    entry_policy: str | None, lookup_policy: str | None
) -> bool:
    """Return True if a bank entry stored under ``entry_policy`` may be reused
    for a lookup that resolved to ``lookup_policy``.

    Equality is always compatible. Beyond that, ``committed`` and
    ``last_window`` are treated as interchangeable because both rely on the
    same committed mtp-history cache shape; the only difference between them
    is a runtime trim that is applied during prefill, which is moot once the
    cache is being restored from a stored snapshot.
    """
    if entry_policy == lookup_policy:
        return True
    if entry_policy is None or lookup_policy is None:
        return False
    return (
        entry_policy in _COMMITTED_CACHE_POLICIES
        and lookup_policy in _COMMITTED_CACHE_POLICIES
    )


def _tree_nbytes(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, CacheSnapshot):
        return _tree_nbytes(value.states) + _tree_nbytes(value.meta_states)
    if isinstance(value, mx.array):
        return int(value.nbytes)
    if isinstance(value, (list, tuple)):
        return sum(_tree_nbytes(item) for item in value)
    if isinstance(value, dict):
        return sum(_tree_nbytes(item) for item in value.values())
    return 0


def _snapshot_nbytes(snapshot: CacheSnapshot) -> int:
    return _tree_nbytes(snapshot.states) + _tree_nbytes(snapshot.meta_states)


@dataclass
class SessionBankEntry:
    token_ids: tuple[int, ...]
    token_hash: str
    model_path: str
    mtp_enabled: bool
    hidden_variant: str | None
    cache_snapshot: CacheSnapshot
    logits: Any
    hidden: Any | None
    cache_ref: list[Any] | None = None
    mtp_history_cache_ref: list[Any] | None = None
    live_ref_only: bool = False
    # Live-ref leases store nbytes=0 (no snapshot copy exists), which blinds
    # byte projections that scale from `longest_prefix().nbytes` — the
    # postcommit's oversized-snapshot estimate read 0 once a lease became the
    # session's longest banked prefix and then materialized a multi-GiB
    # snapshot just to have put() reject it again (the #255 freeze family).
    # Record the rejected snapshot size that forced the lease so projections
    # stay honest across the whole oversized regime.
    oversized_nbytes: int = 0
    # Passive probe: monotonic time this ENTRY OBJECT's cold-tier encode
    # completed (the encode evals the entry's lazy roots in place), or None.
    # Kept on the exact object — Site A and Site B can create distinct
    # entries with the SAME token hash, and an old entry finishing its
    # encode must never report a newer lazy replacement as settled.
    cold_encode_completed_at: float | None = None
    # Monotonic time the idle-lane settle evaluated this entry's lazy
    # snapshot views (releasing their references to live cache buffers so
    # the next turn's writes can donate), or None. Same exact-object
    # contract as cold_encode_completed_at.
    snapshot_settled_at: float | None = None
    created_at_s: float = field(default_factory=time.time)
    last_access_s: float = field(default_factory=time.time)
    hits: int = 0
    nbytes: int = 0
    session_id: str | None = None
    template_hash: str | None = None
    mtp_history_policy: str | None = None
    draft_head_identity: str | None = None
    policy_fingerprint: str | None = None
    mtp_history_snapshot: Any | None = None
    snapshot_epoch: int = 0
    mtp_snapshot_epoch: int | None = None
    eviction_reason: str | None = None
    extra_state: dict[str, Any] | None = None
    # kvcache-v2: KV states held as zero-copy lazy views (recurrent still
    # cloned). Restores install fresh zero-copy views of the stored states —
    # never the stored objects themselves, which would hand the borrower's
    # in-place writes back into this entry (issue #247).
    lazy_kv: bool = False
    # kvcache-v2: whether the source cache carried non-trimmable (recurrent)
    # entries — recorded at put() time from the live cache, because only the
    # producer knows the container classes.
    has_recurrent: bool = False
    # kvcache-v2: (token_count, recurrent-only CacheSnapshot, hidden_last)
    # captured at interior prefill boundaries, sorted ascending. Enables exact
    # sub-prefix restores on hybrid (GDN/conv) models: trim KV to boundary
    # b <= match, install recurrent state at b, re-prefill (b, prompt_end].
    # hidden_last (base hidden of token b-1, may be None) lets committed MTP
    # history resume at b without a seed re-forward — re-running token b-1
    # would advance recurrent state twice and break exactness.
    gdn_boundaries: list[tuple[int, CacheSnapshot, Any]] = field(default_factory=list)
    # kvcache-v2: SSD-restored entries defer boundary decode (exact restores
    # never need them); the loader fills gdn_boundaries on first partial use.
    gdn_boundary_loader: Any = None

    @property
    def prefix_len(self) -> int:
        return len(self.token_ids)

    def _ensure_boundaries_loaded(self) -> None:
        if self.gdn_boundaries or self.gdn_boundary_loader is None:
            return
        loader, self.gdn_boundary_loader = self.gdn_boundary_loader, None
        try:
            self.gdn_boundaries = [
                (int(r[0]), r[1], r[2] if len(r) > 2 else None) for r in loader() or ()
            ]
        except Exception:
            # Fail closed: a missing/corrupt boundary payload just means the
            # partial-restore path declines, exactly as if none were stored.
            self.gdn_boundaries = []

    def recurrent_boundary_at_or_below(
        self, matched: int
    ) -> tuple[int, CacheSnapshot, Any] | None:
        """Newest stored recurrent boundary b <= matched, if any."""
        self._ensure_boundaries_loaded()
        best: tuple[int, CacheSnapshot, Any] | None = None
        for record in self.gdn_boundaries:
            boundary, snapshot = int(record[0]), record[1]
            hidden = record[2] if len(record) > 2 else None
            if boundary <= int(matched) and (best is None or boundary > best[0]):
                best = (boundary, snapshot, hidden)
        return best


def _empty_cache_snapshot(cache: list[Any] | None) -> CacheSnapshot:
    size = len(cache or [])
    return CacheSnapshot(states=tuple(None for _ in range(size)), meta_states=tuple(None for _ in range(size)))


def _trim_cache_ref_to_prefix(cache: list[Any] | None, prefix_len: int) -> bool:
    if cache is None:
        return False
    target_offset = max(0, int(prefix_len) - 1)
    for entry in cache:
        current = int(getattr(entry, "offset", target_offset) or 0)
        if current < target_offset:
            return False
        delta = current - target_offset
        if delta <= 0:
            continue
        trim = getattr(entry, "trim", None)
        if not callable(trim):
            return False
        if int(trim(delta)) != delta:
            return False
    return True


def _trim_cache_ref_to_tokens(cache: list[Any] | None, tokens: int) -> bool:
    """Trim offset-bearing entries to exactly `tokens` consumed tokens.

    Unlike `_trim_cache_ref_to_prefix` (which leaves one slot for a seed
    re-forward of the final prefix token), this lands the cache at the full
    boundary — used by boundary-true restores where no seed forward runs.
    """
    if cache is None:
        return False
    target_offset = max(0, int(tokens))
    for entry in cache:
        current = int(getattr(entry, "offset", target_offset) or 0)
        if current < target_offset:
            return False
        delta = current - target_offset
        if delta <= 0:
            continue
        trim = getattr(entry, "trim", None)
        if not callable(trim):
            return False
        if int(trim(delta)) != delta:
            return False
    return True


def _trim_cache_ref_by_tokens(cache: list[Any] | None, tokens: int) -> bool:
    if cache is None:
        return False
    delta = max(0, int(tokens))
    if delta <= 0:
        return True
    for entry in cache:
        trim = getattr(entry, "trim", None)
        if not callable(trim):
            return False
        if int(trim(delta)) != delta:
            return False
    return True


@dataclass
class SessionBankRestore:
    entry: SessionBankEntry
    cache: list[Any]
    logits: Any
    hidden: Any | None
    restored_nbytes: int
    restore_mode: str = "clone"
    cache_miss_reason: str | None = None
    mtp_history_snapshot: Any | None = None
    mtp_history_cache: list[Any] | None = None
    cache_source: str = "ram"
    ssd_cache_hit: bool = False
    ssd_cached_tokens: int = 0
    ssd_restore_s: float = 0.0
    extra_state: dict[str, Any] | None = None


class SessionBank:
    """In-memory exact prefix table for warm target prefill."""

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_bytes: int = DEFAULT_MAX_BYTES,
        per_session_max_bytes: int = DEFAULT_PER_SESSION_MAX_BYTES,
        idle_ttl_s: float = DEFAULT_IDLE_TTL_S,
        cold_tier: Any | None = None,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if max_bytes < 1:
            raise ValueError("max_bytes must be >= 1")
        if per_session_max_bytes < 1:
            raise ValueError("per_session_max_bytes must be >= 1")
        if idle_ttl_s <= 0:
            raise ValueError("idle_ttl_s must be > 0")
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self.per_session_max_bytes = int(per_session_max_bytes)
        self.idle_ttl_s = float(idle_ttl_s)
        self._entries: dict[tuple[int, ...], SessionBankEntry] = {}
        self.last_miss_reason: str | None = None
        self.last_put_nbytes: int = 0
        self.last_put_skipped_oversized_snapshot: bool = False
        self._oversized_warned_sessions: set[str | None] = set()
        # Bounded: appended on every eviction/skip for the daemon's lifetime;
        # health snapshots only ever read the newest entries, so an unbounded
        # list is pure retention on long-running agent servers.
        self.eviction_log: deque[dict[str, Any]] = deque(maxlen=256)
        self.active_pin_ttl_s = _active_session_pin_ttl_s()
        self._session_last_active: dict[str, float] = {}
        self.per_session_max_entries = _per_session_max_entries()
        self.cold_tier = cold_tier
        # Optional idle-lane dispatcher for SSD cold-tier enqueues. Post-#169
        # put_entry encodes the full-KV payload at enqueue time, so calling it
        # synchronously from a request/stream tail pays the byte conversion
        # there. The server wires this to the model scheduler's idle lane;
        # when unset (tests, CLI paths without a scheduler) the enqueue stays
        # synchronous, preserving legacy behavior.
        self.cold_enqueue_dispatch: Callable[[Callable[[], None]], Any] | None = None
        self.last_restore_source: str | None = None
        self.last_ssd_restore_s: float = 0.0
        self.last_prefix_diagnostic: dict[str, Any] | None = None
        # Dynamic budget ceiling (memory_plan.bank_dynamic_ceiling): the
        # bank takes all free memory while live KV is small and yields as a
        # long-context request's KV actually materializes — the #305 fix's
        # "guard that turns on". None (tests, CLI without a plan) keeps
        # max_bytes as the only bound, byte-identical to legacy behavior.
        self.dynamic_ceiling_fn: Callable[[], int] | None = None
        self.last_dynamic_ceiling_bytes: int | None = None
        self.dynamic_ceiling_errors: int = 0
        # MTPLX_SESSION_BANK_PROTECTED_TERMINAL, resolved ONCE here so a bank keeps
        # one eviction policy for its whole life (see the gate's docstring).
        self.protect_newest_extending: bool = _protected_terminal_enabled()
        # Deflections: how many times publishing a shorter input-prompt
        # fallback would have evicted the newest terminal extending it and
        # took another victim instead. oMLX's `protected_rejections`.
        self.protected_rejections: int = 0
        # MTPLX_SESSION_BANK_SHED_BOUNDARIES, resolved ONCE here (see the gate's
        # docstring for the 2026-09-01 15.79 s receipt this exists to fix).
        self.shed_gdn_boundaries_to_fit: bool = _shed_boundaries_enabled()
        self.boundary_shed_puts: int = 0
        self.boundary_shed_records: int = 0

    # Capability marker for generation: near_prefix_candidates accepts
    # min_restore_tokens so resident-duplicate eligibility mirrors the
    # caller's serve gates. Explicit attribute; no signature inspection.
    SUPPORTS_NEAR_PREFIX_MIN_RESTORE = True

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def total_nbytes(self) -> int:
        return sum(entry.nbytes for entry in self._entries.values())

    def effective_max_bytes(self) -> int:
        """The byte budget in force right now.

        min(configured max, dynamic ceiling). A failing ceiling read must
        never take a put or an eviction down with it — the budget falls
        back to the static max and the failure is counted, not swallowed
        invisibly (dynamic_ceiling_errors rides the health snapshot).
        """
        limit = int(self.max_bytes)
        fn = self.dynamic_ceiling_fn
        if fn is None:
            return limit
        try:
            ceiling = int(fn())
        except Exception:
            self.dynamic_ceiling_errors += 1
            return limit
        self.last_dynamic_ceiling_bytes = ceiling
        return max(1, min(limit, ceiling))

    def touch_sessions(self, session_ids: Any) -> None:
        """Re-stamp the active pin for in-flight sessions.

        The pin is otherwise touched only at restore() and put() time, so a
        single generation longer than the 600 s TTL lost dynamic-ceiling
        protection mid-turn — the memory guard calls this each tick with
        the live requests' sessions (xhigh turns measured 618-624 s on
        2026-08-28, already past the TTL).
        """
        for session_id in session_ids or ():
            self._touch_session(str(session_id))

    def _touch_session(self, session_id: str | None) -> None:
        if not session_id or self.active_pin_ttl_s <= 0:
            return
        now = time.monotonic()
        self._session_last_active[str(session_id)] = now
        if len(self._session_last_active) > 512:
            cutoff = now - self.active_pin_ttl_s
            self._session_last_active = {
                sid: ts
                for sid, ts in self._session_last_active.items()
                if ts >= cutoff
            }

    def _active_session_ids(self) -> set[str]:
        if self.active_pin_ttl_s <= 0 or not self._session_last_active:
            return set()
        cutoff = time.monotonic() - self.active_pin_ttl_s
        return {
            sid for sid, ts in self._session_last_active.items() if ts >= cutoff
        }

    def warn_oversized_snapshot_skip(
        self, session_id: str | None, *, needed_nbytes: int
    ) -> None:
        """Loud once per session (#229): the point where a long conversation
        stops getting durable snapshots (a live-ref lease survives only until
        restart/displacement) and users read the resulting cold prefill as
        "the cache broke". Say exactly which knob raises the ceiling. Shared
        by put()'s oversized branch and the postcommit's byte projection so
        the ceiling is never silent regardless of which gate hits first.
        """
        if session_id in self._oversized_warned_sessions:
            return
        self._oversized_warned_sessions.add(session_id)
        print(
            "[mtplx] session-bank snapshot skipped: session "
            f"{session_id or 'anon'} needs "
            f"{int(needed_nbytes) / 2**30:.1f} GiB but the "
            "per-session cap is "
            f"{self.per_session_max_bytes / 2**30:.1f} GiB — longer "
            "contexts will re-prefill after restart/eviction. Raise "
            "MTPLX_SESSION_BANK_PER_SESSION_BYTES (e.g. "
            f"{max(1, int(needed_nbytes * 1.5) >> 30)}G) to keep "
            "caching this session.",
            flush=True,
        )

    def put(
        self,
        *,
        runtime: MTPLXRuntime,
        token_ids: list[int] | tuple[int, ...],
        cache: list[Any],
        logits: Any,
        hidden: Any | None,
        hidden_variant: str | None = None,
        keep_live_ref: bool = False,
        session_id: str | None = None,
        template_hash: str | None = None,
        mtp_history_policy: str | None = None,
        draft_head_identity: str | None = None,
        policy_fingerprint: str | None = None,
        mtp_history_snapshot: Any | None = None,
        mtp_history_cache_ref: list[Any] | None = None,
        snapshot_epoch: int = 0,
        mtp_snapshot_epoch: int | None = None,
        nbytes_override: int | None = None,
        extra_state: dict[str, Any] | None = None,
        gdn_boundaries: list[tuple[int, CacheSnapshot]] | None = None,
        timing_out: dict[str, Any] | None = None,
    ) -> SessionBankEntry | None:
        # timing_out: optional request-local dict the CALLER owns (never
        # shared bank state — puts run concurrently across the foreground,
        # postcommit, and batched lanes). Keys are written progressively as
        # phases are reached: trunk_snapshot_s, entry_build_s, cold_enqueue
        # {enabled, skip_reason, deferred, dispatch_elapsed_s,
        # synchronous_serialize_elapsed_s}. Early returns leave later keys
        # absent. Deferred cold-tier serialization is never charged here —
        # only the dispatch span is; the job itself runs on the idle lane.
        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            raise ValueError("cannot store an empty prefix")
        if mtp_snapshot_epoch is not None and int(mtp_snapshot_epoch) != int(snapshot_epoch):
            raise ValueError("trunk and MTP snapshots must share the same commit boundary")
        self.last_put_nbytes = 0
        self.last_put_skipped_oversized_snapshot = False
        self._touch_session(session_id)
        cache_has_recurrent = any(not _is_trimmable(entry) for entry in (cache or []))
        normalized_boundaries = sorted(
            (
                (int(r[0]), r[1], r[2] if len(r) > 2 else None)
                for r in (gdn_boundaries or [])
                if int(r[0]) > 0
            ),
            key=lambda item: item[0],
        )
        if os.environ.get("MTPLX_DEBUG_PREFIX_DIVERGENCE"):
            print(
                f"[mtplx] bank-put: len={len(tokens)} "
                f"boundaries={[b[0] for b in normalized_boundaries]} "
                f"session={session_id}",
                file=sys.stderr,
                flush=True,
            )
        if not normalized_boundaries:
            # Same-key replacement must not lose interior boundaries: the idle
            # postcommit re-put of a prompt-boundary entry (which restores
            # instead of re-prefilling, so it captures none) would otherwise
            # strip the store-on-prefill entry's boundaries and push the next
            # RAG-shape turn to a fail-closed cold prefill (found 2026-07-03).
            # Boundaries describe the token prefix, so an identical key keeps
            # them valid.
            prior = self._entries.get(tokens)
            if prior is not None and prior.gdn_boundaries:
                normalized_boundaries = list(prior.gdn_boundaries)
            inherited_loader = (
                getattr(prior, "gdn_boundary_loader", None) if prior is not None else None
            )
            if not normalized_boundaries and inherited_loader is None:
                # Prefix-entry inheritance (#121, 2026-07-16): boundary
                # records describe token PREFIXES, so any stored entry that
                # is a strict prefix of the new tokens carries records that
                # stay valid verbatim for the new entry. Put sites that have
                # no PromptState in scope (generation-final commits) would
                # otherwise store boundary-less entries and push the next
                # divergent agent turn onto the fail-closed cold path.
                prefix_donor = self.longest_prefix(tokens)
                if prefix_donor is not None:
                    # A loader-backed donor (exact SSD restore after a
                    # restart) counts too: its records live on disk behind
                    # the loader, and sharing the loader callable is safe —
                    # it is a pure re-read of the donor's payload.
                    normalized_boundaries = list(prefix_donor.gdn_boundaries)
                    inherited_loader = getattr(
                        prefix_donor, "gdn_boundary_loader", None
                    )
        def live_ref_entry(reason: str, nbytes: int) -> SessionBankEntry | None:
            if not keep_live_ref or not cache:
                return None
            entry = SessionBankEntry(
                token_ids=tokens,
                token_hash=token_prefix_hash(tokens),
                model_path=str(runtime.model_path),
                mtp_enabled=bool(runtime.mtp_enabled),
                hidden_variant=hidden_variant,
                cache_snapshot=_empty_cache_snapshot(cache),
                logits=_clone_tree(logits),
                hidden=_clone_tree(hidden),
                cache_ref=cache,
                mtp_history_cache_ref=mtp_history_cache_ref,
                live_ref_only=True,
                nbytes=0,
                oversized_nbytes=max(0, int(nbytes)),
                session_id=session_id,
                template_hash=template_hash,
                mtp_history_policy=mtp_history_policy,
                draft_head_identity=draft_head_identity,
                policy_fingerprint=policy_fingerprint,
                mtp_history_snapshot=None,
                snapshot_epoch=int(snapshot_epoch),
                mtp_snapshot_epoch=(
                    int(mtp_snapshot_epoch)
                    if mtp_snapshot_epoch is not None
                    else (
                        int(snapshot_epoch)
                        if mtp_history_cache_ref is not None
                        else None
                    )
                ),
                extra_state=_clone_tree(extra_state),
                has_recurrent=cache_has_recurrent,
                gdn_boundaries=list(normalized_boundaries),
            )
            self.eviction_log.append(
                {
                    "reason": reason,
                    "session_id": session_id,
                    "prefix_len": len(tokens),
                    "token_hash": entry.token_hash,
                    "nbytes": int(nbytes),
                    "budget": int(self.per_session_max_bytes),
                    "fallback": "live_reference_lease",
                }
            )
            self._entries[tokens] = entry
            self._supersede_contained_prefixes(tokens)
            self._evict_if_needed(protected_tokens=tokens)
            return entry

        if nbytes_override is not None and int(nbytes_override) > self.per_session_max_bytes:
            self.last_put_nbytes = int(nbytes_override)
            self.last_put_skipped_oversized_snapshot = True
            self.warn_oversized_snapshot_skip(
                session_id, needed_nbytes=int(nbytes_override)
            )
            live_entry = live_ref_entry(
                "skipped_oversized_snapshot_live_ref",
                int(nbytes_override),
            )
            if live_entry is not None:
                self._schedule_live_ref_spill(live_entry)
                return live_entry
            self.eviction_log.append(
                {
                    "reason": "skipped_oversized_snapshot",
                    "session_id": session_id,
                    "prefix_len": len(tokens),
                    "token_hash": token_prefix_hash(tokens),
                    "nbytes": int(nbytes_override),
                    "budget": int(self.per_session_max_bytes),
                }
            )
            return None
        lazy_kv = _lazy_snapshot_enabled()
        trunk_snapshot_started = time.perf_counter()
        try:
            snapshot = (
                snapshot_cache_lazy_hybrid(cache) if lazy_kv else snapshot_cache(cache)
            )
        except RuntimeError as exc:
            if "materialize active K/V arrays" not in str(exc):
                raise
            self.last_put_skipped_oversized_snapshot = True
            live_entry = live_ref_entry(
                "skipped_dense_materializing_snapshot_live_ref",
                0,
            )
            if live_entry is not None:
                self._schedule_live_ref_spill(live_entry)
                return live_entry
            self.eviction_log.append(
                {
                    "reason": "skipped_dense_materializing_snapshot",
                    "session_id": session_id,
                    "prefix_len": len(tokens),
                    "token_hash": token_prefix_hash(tokens),
                    "nbytes": 0,
                    "budget": int(self.per_session_max_bytes),
                    "error": str(exc),
                }
            )
            return None
        trunk_snapshot_done = time.perf_counter()
        if timing_out is not None:
            timing_out["trunk_snapshot_s"] = trunk_snapshot_done - trunk_snapshot_started
        computed_nbytes = (
            _snapshot_nbytes(snapshot)
            + _tree_nbytes(logits)
            + _tree_nbytes(hidden)
            + _tree_nbytes(mtp_history_snapshot)
            + sum(
                _snapshot_nbytes(r[1]) + _tree_nbytes(r[2])
                for r in normalized_boundaries
            )
        )
        entry_nbytes = int(nbytes_override if nbytes_override is not None else computed_nbytes)
        if (
            self.shed_gdn_boundaries_to_fit
            and nbytes_override is None
            and normalized_boundaries
            and entry_nbytes > self.per_session_max_bytes
        ):
            kept, shed_nbytes, shed = self._shed_boundaries_to_fit(
                normalized_boundaries, entry_nbytes
            )
            # Only apply a shed that actually rescues the entry. When the BASE
            # snapshot alone is over budget, dropping records changes nothing
            # about the outcome, so leave the entry exactly as it was and let
            # the unchanged refusal below report the real size.
            if shed and shed_nbytes <= self.per_session_max_bytes:
                normalized_boundaries, entry_nbytes = kept, shed_nbytes
                self.boundary_shed_puts += 1
                self.boundary_shed_records += shed
                self.eviction_log.append(
                    {
                        "reason": "shed_gdn_boundaries",
                        "session_id": session_id,
                        "prefix_len": len(tokens),
                        "token_hash": token_prefix_hash(tokens),
                        "nbytes": int(entry_nbytes),
                        "budget": int(self.per_session_max_bytes),
                        "boundaries_shed": int(shed),
                        "boundaries_kept": len(normalized_boundaries),
                    }
                )
        self.last_put_nbytes = int(entry_nbytes)
        if entry_nbytes > self.per_session_max_bytes:
            self.last_put_skipped_oversized_snapshot = True
            live_entry = live_ref_entry(
                "skipped_oversized_snapshot_live_ref",
                int(entry_nbytes),
            )
            if live_entry is not None:
                self._schedule_live_ref_spill(live_entry)
                return live_entry
            self.eviction_log.append(
                {
                    "reason": "skipped_oversized_snapshot",
                    "session_id": session_id,
                    "prefix_len": len(tokens),
                    "token_hash": token_prefix_hash(tokens),
                    "nbytes": int(entry_nbytes),
                    "budget": int(self.per_session_max_bytes),
                }
            )
            return None
        entry = SessionBankEntry(
            token_ids=tokens,
            token_hash=token_prefix_hash(tokens),
            model_path=str(runtime.model_path),
            mtp_enabled=bool(runtime.mtp_enabled),
            hidden_variant=hidden_variant,
            cache_snapshot=snapshot,
            logits=_clone_tree(logits),
            hidden=_clone_tree(hidden),
            cache_ref=cache if keep_live_ref else None,
            mtp_history_cache_ref=mtp_history_cache_ref if keep_live_ref else None,
            nbytes=int(entry_nbytes),
            session_id=session_id,
            template_hash=template_hash,
            mtp_history_policy=mtp_history_policy,
            draft_head_identity=draft_head_identity,
            policy_fingerprint=policy_fingerprint,
            mtp_history_snapshot=_clone_tree(mtp_history_snapshot),
            snapshot_epoch=int(snapshot_epoch),
            mtp_snapshot_epoch=(
                int(mtp_snapshot_epoch)
                if mtp_snapshot_epoch is not None
                else (int(snapshot_epoch) if mtp_history_snapshot is not None else None)
            ),
            extra_state=_clone_tree(extra_state),
            lazy_kv=lazy_kv,
            has_recurrent=cache_has_recurrent,
            gdn_boundaries=list(normalized_boundaries),
            gdn_boundary_loader=(
                inherited_loader if not normalized_boundaries else None
            ),
        )
        if timing_out is not None:
            timing_out["entry_build_s"] = time.perf_counter() - trunk_snapshot_done
        if lazy_kv:
            self._schedule_snapshot_settle(entry, timing_out=timing_out)
        self._enqueue_cold_entry(entry, timing_out=timing_out)
        self._entries[tokens] = entry
        self._supersede_contained_prefixes(tokens)
        self._evict_if_needed(protected_tokens=tokens)
        return entry

    def put_snapshot(
        self,
        *,
        runtime: MTPLXRuntime,
        token_ids: list[int] | tuple[int, ...],
        cache_snapshot: CacheSnapshot,
        logits: Any = None,
        hidden: Any | None = None,
        hidden_variant: str | None = None,
        keep_live_ref: bool = False,
        cache_ref: list[Any] | None = None,
        session_id: str | None = None,
        template_hash: str | None = None,
        mtp_history_policy: str | None = None,
        draft_head_identity: str | None = None,
        policy_fingerprint: str | None = None,
        mtp_history_snapshot: Any | None = None,
        snapshot_epoch: int = 0,
        mtp_snapshot_epoch: int | None = None,
        nbytes_override: int | None = None,
    ) -> SessionBankEntry | None:
        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            raise ValueError("cannot store an empty prefix")
        if mtp_snapshot_epoch is not None and int(mtp_snapshot_epoch) != int(snapshot_epoch):
            raise ValueError("trunk and MTP snapshots must share the same commit boundary")
        self.last_put_nbytes = 0
        self.last_put_skipped_oversized_snapshot = False
        self._touch_session(session_id)
        computed_nbytes = (
            _snapshot_nbytes(cache_snapshot)
            + _tree_nbytes(logits)
            + _tree_nbytes(hidden)
            + _tree_nbytes(mtp_history_snapshot)
        )
        entry_nbytes = int(nbytes_override if nbytes_override is not None else computed_nbytes)
        self.last_put_nbytes = int(entry_nbytes)
        if entry_nbytes > self.per_session_max_bytes:
            self.last_put_skipped_oversized_snapshot = True
            self.eviction_log.append(
                {
                    "reason": "skipped_oversized_snapshot",
                    "session_id": session_id,
                    "prefix_len": len(tokens),
                    "token_hash": token_prefix_hash(tokens),
                    "nbytes": int(entry_nbytes),
                    "budget": int(self.per_session_max_bytes),
                }
            )
            return None
        snapshot = CacheSnapshot(
            states=tuple(_clone_tree(item) for item in cache_snapshot.states),
            meta_states=tuple(_clone_tree(item) for item in cache_snapshot.meta_states),
        )
        entry = SessionBankEntry(
            token_ids=tokens,
            token_hash=token_prefix_hash(tokens),
            model_path=str(runtime.model_path),
            mtp_enabled=bool(runtime.mtp_enabled),
            hidden_variant=hidden_variant,
            cache_snapshot=snapshot,
            logits=_clone_tree(logits),
            hidden=_clone_tree(hidden),
            cache_ref=cache_ref if keep_live_ref else None,
            nbytes=int(entry_nbytes),
            session_id=session_id,
            template_hash=template_hash,
            mtp_history_policy=mtp_history_policy,
            draft_head_identity=draft_head_identity,
            policy_fingerprint=policy_fingerprint,
            mtp_history_snapshot=_clone_tree(mtp_history_snapshot),
            snapshot_epoch=int(snapshot_epoch),
            mtp_snapshot_epoch=(
                int(mtp_snapshot_epoch)
                if mtp_snapshot_epoch is not None
                else (int(snapshot_epoch) if mtp_history_snapshot is not None else None)
            ),
        )
        self._enqueue_cold_entry(entry)
        self._entries[tokens] = entry
        self._supersede_contained_prefixes(tokens)
        self._evict_if_needed(protected_tokens=tokens)
        return entry

    def shares_ram_prefix(
        self, token_ids: list[int] | tuple[int, ...], *, min_tokens: int
    ) -> bool:
        """Whether any RAM entry shares at least ``min_tokens`` of prefix.

        The cheap question a request-arrival optimisation needs: will the
        prefill start at token 0, or will a restore (exact, near, or block
        prefix -- any of them begins by matching this many tokens) move its
        start past the first chunk?  One tuple-slice compare per entry, no
        cold-tier scan, no lookup side effects.
        """
        n = max(1, int(min_tokens))
        if len(token_ids) < n:
            return False
        head = tuple(int(token) for token in token_ids[:n])
        for prefix in self._entries:
            if len(prefix) >= n and prefix[:n] == head:
                return True
        return False

    def longest_shared_prefix_tokens(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        session_id: str | None = None,
    ) -> int:
        """Longest common prefix, in tokens, between ``token_ids`` and any RAM
        entry (optionally only this session's entries).

        The restore path serves prompts that no entry is an exact prefix of:
        a block-prefix restore rewinds to the last safe boundary under the
        common prefix and re-prefills the tail (an agent follow-up after a
        forced tool round, a retokenized tail). A memory estimate that asks
        only ``longest_prefix`` (exact containment) reads 0 for such prompts
        and calls the session compacted. One compare per entry, no cold-tier
        scan, no lookup side effects.
        """
        tokens = tuple(int(token) for token in token_ids)
        best = 0
        for prefix, entry in self._entries.items():
            if session_id is not None and entry.session_id != session_id:
                continue
            matched = common_prefix_len(tokens, prefix)
            if matched > best:
                best = matched
        return best

    def longest_prefix(self, token_ids: list[int] | tuple[int, ...]) -> SessionBankEntry | None:
        tokens = tuple(int(token) for token in token_ids)
        best: SessionBankEntry | None = None
        for prefix, entry in self._entries.items():
            if len(prefix) > len(tokens):
                continue
            if tokens[: len(prefix)] != prefix:
                continue
            if best is None or len(prefix) > len(best.token_ids):
                best = entry
        return best

    def near_prefix_candidates(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        max_token_gap: int = 8,
        min_matched_tokens: int = 64,
        block_size: int = DEFAULT_PREFIX_BLOCK_SIZE,
        block_min_matched_tokens: int = DEFAULT_BLOCK_PREFIX_MIN_MATCH_TOKENS,
        allow_block_prefix: bool = True,
        model_path: str | None = None,
        mtp_enabled: bool | None = None,
        hidden_variant: str | None = None,
        template_hash: str | None = None,
        mtp_history_policy: str | None = None,
        draft_head_identity: str | None = None,
        policy_fingerprint: str | None = None,
        min_restore_tokens: int = 0,
    ) -> list[tuple[SessionBankEntry, int]]:
        """Return entries whose divergence can be restored from a safe boundary.

        The tiny-gap path covers tokenizer-boundary drift at the very end of a
        stored transcript. The block-prefix path covers real agent follow-ups:
        if a new prompt shares a large stable token prefix, restore to the last
        full block and prefill only the changed suffix.
        """
        tokens = tuple(int(token) for token in token_ids)
        gap_limit = max(0, int(max_token_gap))
        min_match = max(1, int(min_matched_tokens))
        block = max(1, int(block_size))
        block_min_match = max(block, int(block_min_matched_tokens))
        matches: list[tuple[SessionBankEntry, int]] = []
        best_diag: dict[str, Any] | None = None
        self._purge_expired()
        for entry in self._entries.values():
            prefix = entry.token_ids
            if not prefix:
                continue
            matched = common_prefix_len(tokens, prefix)
            gap = len(prefix) - matched
            safe_block = min(
                block_aligned_prefix_len(matched, block_size=block),
                len(prefix),
                len(tokens),
            )
            diag = {
                "prompt_len": len(tokens),
                "session_id": entry.session_id,
                "stored_prefix_len": len(prefix),
                "common_prefix_tokens": int(matched),
                "nearest_boundary_tokens": int(safe_block),
                "near_prefix_gap": int(gap),
                "token_hash": entry.token_hash,
            }
            if best_diag is None or (
                int(diag["common_prefix_tokens"]),
                int(diag["nearest_boundary_tokens"]),
                int(diag["stored_prefix_len"]),
            ) > (
                int(best_diag["common_prefix_tokens"]),
                int(best_diag["nearest_boundary_tokens"]),
                int(best_diag["stored_prefix_len"]),
            ):
                best_diag = diag

            required_match = min(min_match, max(1, len(prefix) - gap_limit))
            # A stored prefix may include assistant output after the exact
            # prompt boundary. If the requested prompt is wholly contained in
            # that longer continuation, restoring at `matched` can leave decode
            # sitting on a post-answer/EOS boundary. Treat that as unsafe for
            # the tiny-gap path; long prompts can still use a block-aligned
            # restore below and re-prefill the tail to the real prompt end.
            if (
                gap >= 0
                and gap <= gap_limit
                and matched >= required_match
                and matched < len(tokens)
            ):
                matches.append((entry, matched))
                continue

            if not allow_block_prefix:
                continue
            # kvcache-v2 token-granularity: entries that can restore exactly at
            # any offset (pure-attention models) or that carry interior
            # recurrent boundaries no longer quantize the match to block edges
            # — KV trims to any token and the boundary-true restore picks the
            # actual recurrent-safe point. Legacy hybrid entries without
            # boundaries keep the block-aligned value (restore fails closed on
            # them when boundary-true is on).
            exact_capable = (not entry.has_recurrent) or bool(
                entry.gdn_boundaries or getattr(entry, "gdn_boundary_loader", None)
            )
            candidate_len = matched if exact_capable else safe_block
            if candidate_len < block_min_match:
                continue
            if candidate_len < 2:
                continue
            if candidate_len > matched:
                continue
            matches.append((entry, candidate_len))

        # Serve-equivalent RESIDENT twins of possible cold rows, built ONLY
        # from the computed matches above and only from candidates that
        # generation would ACTUALLY serve — mirroring its gates exactly:
        # min_restore_tokens, matched range, identity, committed-MTP
        # presence when the policy requires it, snapshot-epoch sync, and
        # the recurrent achievable-boundary threshold. Raw RAM matches are
        # NOT a floor: an entry generation would reject must never suppress
        # a valid cold candidate or cold-only recovery. When no eligible
        # twin exists the cold lookup runs exactly as before.
        committed_required = _policy_uses_committed_history(mtp_history_policy)
        floor = int(min_restore_tokens)
        resident_duplicates: dict[str, dict[str, Any]] | None = None
        serve_compatible_best_matched = 0
        for _entry, _candidate_len in matches:
            _cand = int(_candidate_len)
            if _cand <= floor:
                continue
            if _cand < 2 or _cand >= int(_entry.prefix_len):
                continue
            if _entry.live_ref_only:
                # Leases are single-use; only durable snapshot twins may
                # suppress a cold hydration.
                continue
            if not _restore_identity_compatible(
                _entry,
                model_path=model_path,
                mtp_enabled=mtp_enabled,
                hidden_variant=hidden_variant,
                template_hash=template_hash,
                mtp_history_policy=mtp_history_policy,
                draft_head_identity=draft_head_identity,
                policy_fingerprint=policy_fingerprint,
            ):
                continue
            _has_mtp = (
                _entry.mtp_history_snapshot is not None
                or getattr(_entry, "mtp_history_cache_ref", None) is not None
            )
            if committed_required and not _has_mtp:
                continue
            if (
                _entry.mtp_snapshot_epoch is not None
                and int(_entry.mtp_snapshot_epoch) != int(_entry.snapshot_epoch)
            ):
                continue
            if _entry.has_recurrent:
                _gap = int(_entry.prefix_len) - _cand
                if _gap > gap_limit:
                    _probe = getattr(
                        _entry, "recurrent_boundary_at_or_below", None
                    )
                    _achievable = 0
                    if callable(_probe):
                        _boundary = _probe(_cand)
                        if _boundary is not None:
                            _achievable = int(_boundary[0])
                    if _achievable <= floor:
                        continue
            if resident_duplicates is None:
                resident_duplicates = {}
            resident_duplicates[str(_entry.token_hash)] = {
                "prefix_len": int(_entry.prefix_len),
                "has_mtp_history": _has_mtp,
            }
            # Entries surviving every gate above are serve-usable for THIS
            # request; only they may raise the bar a cold candidate must
            # beat. A raw-higher but identity-incompatible RAM match must
            # never suppress a valid cold hydration
            # (test_identity_incompatible_higher_ram_match_does_not_shadow_valid_cold).
            serve_compatible_best_matched = max(
                serve_compatible_best_matched, _cand
            )
        # The best SERVE-COMPATIBLE RAM match is the bar a cold candidate
        # must beat: the stable sort below picks max (matched, prefix_len),
        # so a cold row with strictly smaller matched can never win against
        # an entry that can actually serve this request — hydrating it (a
        # multi-GB disk read + decode on the request thread) is pure waste.
        # The bar deliberately ignores incompatible/lease-only RAM entries
        # (they cannot serve, so they must not shadow a valid cold row).
        ram_best_matched = int(serve_compatible_best_matched)
        if floor > 0:
            # The caller's own gate discards every near/block candidate with
            # matched <= min_restore_tokens (its exact-prefix RAM entry
            # already serves that much), so a cold row at or below the floor
            # can never be served through this lane -- hydrating it is a
            # multi-GB SSD decode on the request thread for a candidate the
            # sort discards unread. Measured 0.59-0.66 s of unattributed
            # prompt-state wall on EVERY warm agent turn (py-spy on the
            # 2026-09-03 candidate daemon: the exact entry was skipped by
            # the `_cand <= floor` guard above, so the bar read 0 and the
            # same turn's own SSD twin hydrated each time). Strictly-better
            # cold rows and cold-only recovery (floor 0) are unchanged.
            ram_best_matched = max(ram_best_matched, floor + 1)
        cold_match = self._cold_near_prefix_candidate(
            tokens,
            max_token_gap=gap_limit,
            min_matched_tokens=min_match,
            block_size=block,
            block_min_matched_tokens=block_min_match,
            allow_block_prefix=allow_block_prefix,
            resident_duplicates=resident_duplicates,
            min_useful_matched_tokens=ram_best_matched,
            model_path=model_path,
            mtp_enabled=mtp_enabled,
            hidden_variant=hidden_variant,
            template_hash=template_hash,
            mtp_history_policy=mtp_history_policy,
            draft_head_identity=draft_head_identity,
            policy_fingerprint=policy_fingerprint,
        )
        if cold_match is not None:
            matches.append(cold_match)
        matches.sort(key=lambda item: (item[1], item[0].prefix_len), reverse=True)
        if matches:
            entry, matched = matches[0]
            cache_source = str(getattr(entry, "cache_source", "ram") or "ram")
            self.last_prefix_diagnostic = {
                "prompt_len": len(tokens),
                "session_id": entry.session_id,
                "stored_prefix_len": entry.prefix_len,
                "common_prefix_tokens": int(common_prefix_len(tokens, entry.token_ids)),
                "nearest_boundary_tokens": int(matched),
                "new_prefill_tokens": max(0, len(tokens) - int(matched)),
                "miss_reason": None,
                "restore_kind": (
                    "near_boundary"
                    if entry.prefix_len - int(matched) <= gap_limit
                    else "block_prefix"
                ),
                "cache_source": cache_source,
            }
        else:
            best = best_diag or {
                "prompt_len": len(tokens),
                "session_id": None,
                "stored_prefix_len": 0,
                "common_prefix_tokens": 0,
                "nearest_boundary_tokens": 0,
            }
            best["miss_reason"] = (
                CacheMissReason.PREFIX_DIVERGENCE_AT_TOKEN.value
                if self._entries
                else CacheMissReason.NEW_SESSION.value
            )
            self.last_prefix_diagnostic = best
        return matches

    def _cold_near_prefix_candidate(
        self,
        tokens: tuple[int, ...],
        *,
        max_token_gap: int,
        min_matched_tokens: int,
        block_size: int,
        block_min_matched_tokens: int,
        allow_block_prefix: bool,
        model_path: str | None,
        mtp_enabled: bool | None,
        hidden_variant: str | None,
        template_hash: str | None,
        mtp_history_policy: str | None,
        draft_head_identity: str | None,
        policy_fingerprint: str | None,
        resident_duplicates: dict[str, dict[str, Any]] | None = None,
        min_useful_matched_tokens: int = 0,
    ) -> tuple[SessionBankEntry, int] | None:
        if self.cold_tier is None:
            return None
        if model_path is None or mtp_enabled is None:
            return None
        if not block_prefix_restore_enabled():
            return None
        lookup = getattr(self.cold_tier, "lookup_prefix_boundary", None)
        if not callable(lookup):
            return None
        # Capability detection happens BEFORE the call via an explicit tier
        # attribute (no per-request signature inspection, never an
        # exception/retry probe after work): duck-typed tiers without the
        # marker get the pre-shadow call shape and simply hydrate as before.
        lookup_kwargs: dict[str, Any] = {}
        if resident_duplicates and getattr(
            self.cold_tier, "SUPPORTS_RESIDENT_DUPLICATE_SHADOW", False
        ):
            lookup_kwargs["resident_duplicates"] = resident_duplicates
        if min_useful_matched_tokens > 0 and getattr(
            self.cold_tier, "SUPPORTS_MIN_USEFUL_MATCHED_TOKENS", False
        ):
            lookup_kwargs["min_useful_matched_tokens"] = int(
                min_useful_matched_tokens
            )
        result = lookup(
            tokens,
            model_path=model_path,
            mtp_enabled=bool(mtp_enabled),
            hidden_variant=hidden_variant,
            template_hash=template_hash,
            mtp_history_policy=mtp_history_policy,
            draft_head_identity=draft_head_identity,
            policy_fingerprint=policy_fingerprint,
            max_token_gap=max_token_gap,
            min_matched_tokens=min_matched_tokens,
            block_size=block_size,
            block_min_matched_tokens=block_min_matched_tokens,
            allow_block_prefix=allow_block_prefix,
            **lookup_kwargs,
        )
        if result is None:
            return None
        record = getattr(result, "record", None)
        matched = int(getattr(result, "matched_tokens", 0) or 0)
        if record is None or matched <= 0:
            return None
        metadata = dict(getattr(record, "metadata", {}) or {})
        entry = SessionBankEntry(
            token_ids=tuple(int(token) for token in record.token_ids),
            token_hash=metadata.get("token_hash") or token_prefix_hash(record.token_ids),
            has_recurrent=bool(
                getattr(record, "has_recurrent", False)
                or metadata.get("has_recurrent", False)
            ),
            gdn_boundaries=list(getattr(record, "gdn_boundaries", None) or []),
            gdn_boundary_loader=getattr(record, "gdn_boundary_loader", None),
            model_path=str(metadata.get("model_path") or model_path),
            mtp_enabled=bool(metadata.get("mtp_enabled", mtp_enabled)),
            hidden_variant=metadata.get("hidden_variant"),
            cache_snapshot=record.cache_snapshot,
            logits=_clone_tree(record.logits),
            hidden=_clone_tree(record.hidden),
            nbytes=int(getattr(record, "nbytes", 0) or 0),
            session_id=metadata.get("session_id"),
            template_hash=metadata.get("template_hash"),
            mtp_history_policy=metadata.get("mtp_history_policy"),
            draft_head_identity=metadata.get("draft_head_identity"),
            policy_fingerprint=metadata.get("policy_fingerprint"),
            mtp_history_snapshot=_clone_tree(record.mtp_history_snapshot),
            snapshot_epoch=int(metadata.get("snapshot_epoch") or len(record.token_ids)),
            mtp_snapshot_epoch=(
                int(metadata["mtp_snapshot_epoch"])
                if metadata.get("mtp_snapshot_epoch") is not None
                else (
                    int(metadata.get("snapshot_epoch") or len(record.token_ids))
                    if record.mtp_history_snapshot is not None
                    else None
                )
            ),
        )
        setattr(entry, "cache_source", "ssd")
        setattr(entry, "ssd_cache_hit", True)
        setattr(entry, "ssd_cached_tokens", matched)
        setattr(entry, "ssd_restore_s", float(getattr(record, "restore_s", 0.0) or 0.0))
        return entry, matched

    def restore(
        self,
        runtime: MTPLXRuntime,
        token_ids: list[int] | tuple[int, ...],
        *,
        mode: str = "clone",
        session_id: str | None = None,
        hidden_variant: str | None = None,
        template_hash: str | None = None,
        mtp_history_policy: str | None = None,
        draft_head_identity: str | None = None,
        policy_fingerprint: str | None = None,
        cache_factory: Callable[[], list[Any]] | None = None,
        mtp_cache_factory: Callable[[], list[Any]] | None = None,
    ) -> SessionBankRestore | None:
        mode = str(mode).replace("-", "_")
        if mode == "reference_lease":
            mode = "reference"
        if mode not in {"clone", "reference"}:
            raise ValueError("mode must be 'clone', 'reference', or 'reference_lease'")
        self.last_miss_reason = None
        self._purge_expired()
        self._touch_session(session_id)

        def cold_fallback() -> SessionBankRestore | None:
            return self._restore_cold(
                runtime,
                token_ids,
                session_id=session_id,
                hidden_variant=hidden_variant,
                template_hash=template_hash,
                mtp_history_policy=mtp_history_policy,
                draft_head_identity=draft_head_identity,
                policy_fingerprint=policy_fingerprint,
            )

        entry = self.longest_prefix(token_ids)
        if entry is None:
            self.last_miss_reason = (
                CacheMissReason.PREFIX_DIVERGENCE_AT_TOKEN.value
                if self._entries
                else CacheMissReason.NEW_SESSION.value
            )
            if self.last_prefix_diagnostic is not None:
                self.last_prefix_diagnostic["miss_reason"] = self.last_miss_reason
            return cold_fallback()
        if entry.model_path != str(runtime.model_path):
            self.last_miss_reason = CacheMissReason.MODEL_MISMATCH.value
            return cold_fallback()
        if hidden_variant is not None and entry.hidden_variant != hidden_variant:
            self.last_miss_reason = CacheMissReason.POLICY_MISMATCH.value
            return cold_fallback()
        if template_hash is not None and entry.template_hash != template_hash:
            self.last_miss_reason = CacheMissReason.TEMPLATE_MISMATCH.value
            return cold_fallback()
        if mtp_history_policy is not None and not _mtp_history_policy_compatible(
            entry.mtp_history_policy, mtp_history_policy
        ):
            self.last_miss_reason = CacheMissReason.POLICY_MISMATCH.value
            return cold_fallback()
        if draft_head_identity is not None and entry.draft_head_identity != draft_head_identity:
            self.last_miss_reason = CacheMissReason.POLICY_MISMATCH.value
            return cold_fallback()
        if policy_fingerprint is not None and entry.policy_fingerprint != policy_fingerprint:
            self.last_miss_reason = CacheMissReason.POLICY_MISMATCH.value
            return cold_fallback()
        if (
            entry.mtp_snapshot_epoch is not None
            and int(entry.mtp_snapshot_epoch) != int(entry.snapshot_epoch)
        ):
            self.last_miss_reason = CacheMissReason.SNAPSHOT_DESYNC.value
            return cold_fallback()
        actual_restore_mode = "clone"
        if mode == "reference" and entry.cache_ref is not None:
            cache = entry.cache_ref
            entry.cache_ref = None
            # Trim depth is the seed-forward contract, decided by the lookup
            # shape. A lookup that EXTENDS the entry is served by the exact
            # suffix-forward lane, which forwards prompt[prefix_len:] with NO
            # seed re-forward — the cache must land at the FULL boundary or
            # the first suffix token overwrites the final prefix position
            # (integer-exact receipt: warm lease restores decoded different
            # bytes than cold, F39 lane 2026-08-16). An exact full-prefix
            # lookup keeps the pre-last-token trim: that consumer contract
            # (BatchGenerator insert / stored-boundary-logits decode start)
            # owns the final-token re-forward.
            if len(token_ids) > entry.prefix_len:
                trimmed = _trim_cache_ref_to_tokens(cache, entry.prefix_len)
            else:
                trimmed = _trim_cache_ref_to_prefix(cache, entry.prefix_len)
            if not trimmed:
                self.last_miss_reason = CacheMissReason.NO_SNAPSHOT_COVERAGE.value
                return cold_fallback()
            actual_restore_mode = "reference_lease"
        else:
            if entry.live_ref_only:
                self.last_miss_reason = CacheMissReason.NO_SNAPSHOT_COVERAGE.value
                return cold_fallback()
            cache = cache_factory() if cache_factory is not None else runtime.make_cache()
            restore_cache(
                cache,
                entry.cache_snapshot,
                restore_meta_state=cache_factory is None,
                clone_states=not entry.lazy_kv,
            )
        mtp_history_cache = None
        if mode == "reference" and entry.mtp_history_cache_ref is not None:
            mtp_history_cache = entry.mtp_history_cache_ref
            entry.mtp_history_cache_ref = None
            if not _trim_cache_ref_to_prefix(mtp_history_cache, entry.prefix_len):
                self.last_miss_reason = CacheMissReason.NO_SNAPSHOT_COVERAGE.value
                return cold_fallback()
        elif entry.mtp_history_snapshot is not None:
            mtp_history_cache = (
                mtp_cache_factory()
                if mtp_cache_factory is not None
                else runtime.make_mtp_cache()
            )
            restore_cache(mtp_history_cache, entry.mtp_history_snapshot)
        elif entry.live_ref_only and entry.mtp_snapshot_epoch is not None:
            self.last_miss_reason = CacheMissReason.NO_SNAPSHOT_COVERAGE.value
            return cold_fallback()
        entry.hits += 1
        entry.last_access_s = time.time()
        self.last_restore_source = "ram"
        self.last_ssd_restore_s = 0.0
        lookup_len = len(tuple(int(token) for token in token_ids))
        self.last_prefix_diagnostic = {
            "prompt_len": lookup_len,
            "session_id": entry.session_id,
            "stored_prefix_len": entry.prefix_len,
            "common_prefix_tokens": entry.prefix_len,
            "nearest_boundary_tokens": entry.prefix_len,
            "new_prefill_tokens": max(0, lookup_len - entry.prefix_len),
            "miss_reason": None,
            "restore_kind": "exact_prefix",
        }
        return SessionBankRestore(
            entry=entry,
            cache=cache,
            logits=_clone_tree(entry.logits),
            hidden=_clone_tree(entry.hidden),
            restored_nbytes=entry.nbytes,
            restore_mode=actual_restore_mode,
            mtp_history_snapshot=_clone_tree(entry.mtp_history_snapshot),
            mtp_history_cache=mtp_history_cache,
            cache_source="ram",
            extra_state=_clone_tree(entry.extra_state),
        )

    def restore_entry_prefix_cache(
        self,
        runtime: MTPLXRuntime,
        entry: SessionBankEntry,
        prefix_len: int,
        *,
        mode: str = "clone",
        cache_factory: Callable[[], list[Any]] | None = None,
        mtp_cache_factory: Callable[[], list[Any]] | None = None,
        served_out: dict[str, Any] | None = None,
    ) -> tuple[list[Any], list[Any] | None, str] | None:
        """Restore a cached entry to an earlier safe prefix boundary.

        Exact ``restore()`` only works when the stored token prefix is a
        literal prefix of the next prompt. Real agent transcripts often diverge
        at the assistant-generation marker while sharing almost the entire long
        user/workspace prefix. This helper lets the generation layer restore a
        block-aligned boundary from the same entry and prefill only the suffix.
        """

        mode = str(mode).replace("-", "_")
        if mode == "reference_lease":
            mode = "reference"
        if mode not in {"clone", "reference"}:
            raise ValueError("mode must be 'clone', 'reference', or 'reference_lease'")
        matched = int(prefix_len)
        if matched < 1 or matched > int(entry.prefix_len):
            return None

        # kvcache-v2 boundary-true restore: on hybrid models a sub-prefix
        # restore must land on a token where the recurrent state is *known*,
        # not merely where the KV can trim. Restoring KV to `matched` while
        # recurrent state stays at the stored end silently degrades answers
        # (Desktop QA, pre-v2). Tiny gaps (<= near-prefix gap limit) keep the
        # long-shipped tokenizer-drift tolerance; anything larger requires a
        # stored boundary <= matched and restores there instead, with the
        # caller re-prefilling (boundary, prompt_end].
        restore_point = matched
        boundary_snapshot: CacheSnapshot | None = None
        boundary_hidden: Any | None = None
        gap_from_entry = int(entry.prefix_len) - matched
        needs_boundary = (
            bool(entry.has_recurrent)
            and gap_from_entry > _near_prefix_tiny_gap_limit()
        )
        if needs_boundary:
            boundary = entry.recurrent_boundary_at_or_below(matched)
            if boundary is None:
                if os.environ.get("MTPLX_DEBUG_PREFIX_DIVERGENCE"):
                    positions = [
                        int(record[0])
                        for record in (entry.gdn_boundaries or [])
                    ]
                    print(
                        f"[mtplx] boundary-miss: entry_len={entry.prefix_len} "
                        f"matched={matched} boundary_positions={positions}",
                        file=sys.stderr,
                        flush=True,
                    )
                if _boundary_true_restore_enabled():
                    self.last_miss_reason = (
                        CacheMissReason.NO_SNAPSHOT_COVERAGE.value
                    )
                    return None
                # Legacy escape hatch (env off-switch): pre-v2 behavior.
            else:
                restore_point, boundary_snapshot, boundary_hidden = boundary
                if restore_point < 1:
                    self.last_miss_reason = (
                        CacheMissReason.NO_SNAPSHOT_COVERAGE.value
                    )
                    return None

        actual_restore_mode = "clone"
        mtp_history_trim_tokens = max(0, int(entry.prefix_len) - restore_point)
        # Boundary restores land the KV at the full boundary (no seed forward
        # will run — it would advance recurrent state past the captured
        # boundary a second time). Non-boundary restores keep the seed-forward
        # slot semantics.
        trim_to_target = (
            (lambda c: _trim_cache_ref_to_tokens(c, restore_point))
            if boundary_snapshot is not None
            else (lambda c: _trim_cache_ref_to_prefix(c, restore_point))
        )
        # Passive-probe maintenance splits: CPU-side perf_counter spans only,
        # written into the caller-owned served_out dict (request-local, same
        # non-shared contract as put's timing_out). No evaluation points are
        # added or moved — the lazy graph is observed, never perturbed.
        _mnt: dict[str, Any] | None = None
        if served_out is not None:
            _mnt = {}
            served_out["maintenance"] = _mnt
        if mode == "reference" and entry.cache_ref is not None:
            cache = entry.cache_ref
            entry.cache_ref = None
            _trim_started = time.perf_counter()
            if not trim_to_target(cache):
                self.last_miss_reason = CacheMissReason.NO_SNAPSHOT_COVERAGE.value
                return None
            if _mnt is not None:
                _mnt["trim_s"] = time.perf_counter() - _trim_started
            actual_restore_mode = "reference_lease"
        else:
            if entry.live_ref_only:
                self.last_miss_reason = CacheMissReason.NO_SNAPSHOT_COVERAGE.value
                return None
            _factory_started = time.perf_counter()
            cache = cache_factory() if cache_factory is not None else runtime.make_cache()
            _install_started = time.perf_counter()
            restore_cache(
                cache,
                entry.cache_snapshot,
                restore_meta_state=cache_factory is None,
                clone_states=not entry.lazy_kv,
            )
            _trim_started = time.perf_counter()
            if not trim_to_target(cache):
                self.last_miss_reason = CacheMissReason.NO_SNAPSHOT_COVERAGE.value
                return None
            if _mnt is not None:
                _mnt["factory_s"] = _install_started - _factory_started
                _mnt["install_s"] = _trim_started - _install_started
                _mnt["trim_s"] = time.perf_counter() - _trim_started
        if boundary_snapshot is not None:
            # Overwrite recurrent (non-trimmable) states with the interior
            # boundary capture; trimmable entries are None in these snapshots.
            _overwrite_started = time.perf_counter()
            restore_cache(cache, boundary_snapshot, restore_meta_state=False)
            if _mnt is not None:
                _mnt["recurrent_overwrite_s"] = (
                    time.perf_counter() - _overwrite_started
                )

        mtp_history_cache = None
        if mode == "reference" and entry.mtp_history_cache_ref is not None:
            mtp_history_cache = entry.mtp_history_cache_ref
            entry.mtp_history_cache_ref = None
            if not _trim_cache_ref_by_tokens(
                mtp_history_cache,
                mtp_history_trim_tokens,
            ):
                self.last_miss_reason = CacheMissReason.NO_SNAPSHOT_COVERAGE.value
                return None
        elif entry.mtp_history_snapshot is not None:
            _mtp_started = time.perf_counter()
            mtp_history_cache = (
                mtp_cache_factory()
                if mtp_cache_factory is not None
                else runtime.make_mtp_cache()
            )
            restore_cache(mtp_history_cache, entry.mtp_history_snapshot)
            if not _trim_cache_ref_by_tokens(
                mtp_history_cache,
                mtp_history_trim_tokens,
            ):
                self.last_miss_reason = CacheMissReason.NO_SNAPSHOT_COVERAGE.value
                return None
            if _mnt is not None:
                _mnt["mtp_install_s"] = time.perf_counter() - _mtp_started
        elif entry.live_ref_only and entry.mtp_snapshot_epoch is not None:
            self.last_miss_reason = CacheMissReason.NO_SNAPSHOT_COVERAGE.value
            return None

        if served_out is not None:
            served_out["restore_point"] = int(restore_point)
            served_out["boundary_used"] = boundary_snapshot is not None
            served_out["mode"] = actual_restore_mode
        return (
            cache,
            mtp_history_cache,
            actual_restore_mode,
            restore_point,
            boundary_hidden if boundary_snapshot is not None else None,
        )

    def clear(self, *, session_id: str | None = None) -> int:
        if session_id is None:
            count = len(self._entries)
            self._entries.clear()
            return count
        victims = [
            tokens
            for tokens, entry in self._entries.items()
            if entry.session_id == session_id
        ]
        for tokens in victims:
            self._entries.pop(tokens, None)
        return len(victims)

    def archive_cold_tier(self) -> dict[str, Any]:
        if self.cold_tier is None:
            return {"archived": False, "reason": "ssd_cache_disabled"}
        archive = getattr(self.cold_tier, "archive", None)
        if not callable(archive):
            return {"archived": False, "reason": "ssd_cache_archive_unavailable"}
        return archive()

    def flush_cold_tier(self, *, timeout_s: float = 30.0) -> bool:
        if self.cold_tier is None:
            return True
        flush = getattr(self.cold_tier, "flush", None)
        if not callable(flush):
            return True
        return bool(flush(timeout_s=timeout_s))

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_entries": self.max_entries,
            "max_bytes": self.max_bytes,
            "effective_max_bytes": self.effective_max_bytes(),
            "dynamic_ceiling_bytes": self.last_dynamic_ceiling_bytes,
            "dynamic_ceiling_errors": self.dynamic_ceiling_errors,
            # Engagement receipt for MTPLX_SESSION_BANK_PROTECTED_TERMINAL: a lane
            # that reads flat in an A/B must still be able to prove it ran.
            "protect_newest_extending": bool(self.protect_newest_extending),
            "protected_rejections": int(self.protected_rejections),
            # MTPLX_SESSION_BANK_SHED_BOUNDARIES: boundary_shed_puts > 0 is the
            # proof that entries which used to be refused wholesale are now
            # being admitted with a reduced boundary set.
            "shed_gdn_boundaries_to_fit": bool(self.shed_gdn_boundaries_to_fit),
            "boundary_shed_puts": int(self.boundary_shed_puts),
            "boundary_shed_records": int(self.boundary_shed_records),
            "per_session_max_bytes": self.per_session_max_bytes,
            "idle_ttl_s": self.idle_ttl_s,
            "entries": len(self._entries),
            "total_nbytes": self.total_nbytes,
            "last_miss_reason": self.last_miss_reason,
            "last_restore_source": self.last_restore_source,
            "last_ssd_restore_s": self.last_ssd_restore_s,
            "last_prefix_diagnostic": self.last_prefix_diagnostic,
            "active_pin_ttl_s": self.active_pin_ttl_s,
            "active_sessions": sorted(self._active_session_ids()),
            "recent_evictions": list(self.eviction_log)[-8:],
            "cold_tier": (
                self.cold_tier.stats()
                if self.cold_tier is not None and hasattr(self.cold_tier, "stats")
                else {"enabled": False}
            ),
            "prefixes": [
                {
                    "session_id": entry.session_id,
                    "prefix_len": entry.prefix_len,
                    "token_hash": entry.token_hash,
                    "model_path": entry.model_path,
                    "mtp_enabled": entry.mtp_enabled,
                    "hidden_variant": entry.hidden_variant,
                    "template_hash": entry.template_hash,
                    "mtp_history_policy": entry.mtp_history_policy,
                    "draft_head_identity": entry.draft_head_identity,
                    "policy_fingerprint": entry.policy_fingerprint,
                    "hits": entry.hits,
                    "nbytes": entry.nbytes,
                    "created_at_s": entry.created_at_s,
                    "last_access_s": entry.last_access_s,
                    "has_live_ref": entry.cache_ref is not None,
                    "has_mtp_history_live_ref": entry.mtp_history_cache_ref is not None,
                    "live_ref_only": bool(entry.live_ref_only),
                    "snapshot_epoch": entry.snapshot_epoch,
                    "mtp_snapshot_epoch": entry.mtp_snapshot_epoch,
                    "lazy_kv": bool(getattr(entry, "lazy_kv", False)),
                    "has_recurrent": bool(getattr(entry, "has_recurrent", False)),
                    "gdn_boundaries": [
                        int(record[0])
                        for record in (getattr(entry, "gdn_boundaries", None) or [])
                    ],
                }
                for entry in sorted(self._entries.values(), key=lambda item: item.prefix_len)
            ],
            "eviction_log": list(self.eviction_log)[-16:],
        }

    def _schedule_snapshot_settle(
        self,
        entry: SessionBankEntry,
        timing_out: dict[str, Any] | None = None,
    ) -> None:
        """Materialize the entry's lazy snapshot views off the request tail.

        Runs on the same model-owner idle lane as the cold encode, dispatched
        FIRST so the views settle before the (much heavier, coalesced) SSD
        serialize touches them. No dispatch lane means no settle — the lazy
        contract stays exactly as before rather than paying a synchronous
        eval on the response tail. Per-array evals keep any foreground
        request that lands mid-settle waiting at most one array (~10 ms),
        not the whole snapshot.
        """
        if entry.live_ref_only or not _snapshot_settle_enabled():
            return
        dispatch = self.cold_enqueue_dispatch
        if dispatch is None:
            if timing_out is not None:
                timing_out["snapshot_settle"] = {"dispatched": False}
            return

        def _settle_job() -> None:
            # Plain mx.eval of a full-range lazy view ALIASES the source
            # buffer (measured 2026-08-30: eval(base[...]) allocates
            # nothing, and mx.contiguous no-ops on already-contiguous
            # inputs), so the donation-blocking reference survives eval.
            # Only an owner copy (metal_copy_leaf) actually decouples the
            # snapshot from the live buffers; each leaf is copied and
            # evaluated individually so a foreground request that lands
            # mid-settle waits at most one leaf.
            try:
                from mtplx.kernels.copy_leaf import metal_copy_leaf

                def _own(value: Any) -> Any:
                    if value is None:
                        return None
                    if isinstance(value, CacheSnapshot):
                        return CacheSnapshot(
                            states=_own(value.states),
                            meta_states=_own(value.meta_states),
                        )
                    if isinstance(value, mx.array):
                        owned = metal_copy_leaf(value)
                        mx.eval(owned)
                        return owned
                    if isinstance(value, tuple):
                        return tuple(_own(item) for item in value)
                    if isinstance(value, list):
                        return [_own(item) for item in value]
                    if isinstance(value, dict):
                        return {key: _own(item) for key, item in value.items()}
                    return value

                # Field-at-a-time rebinding: every intermediate state is
                # valid (same values, different buffers), so a concurrent
                # restore reading the entry mid-settle stays correct.
                entry.cache_snapshot = _own(entry.cache_snapshot)
                entry.mtp_history_snapshot = _own(entry.mtp_history_snapshot)
                entry.logits = _own(entry.logits)
                entry.hidden = _own(entry.hidden)
                entry.snapshot_settled_at = time.monotonic()
            except Exception as exc:
                self.eviction_log.append(
                    {
                        "reason": "snapshot_settle_error",
                        "session_id": entry.session_id,
                        "prefix_len": entry.prefix_len,
                        "token_hash": entry.token_hash,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        job = _settle_job
        # Newest-wins per session: settling a superseded entry's snapshot
        # is pure waste, and the key namespace is disjoint from the SSD
        # encode's so a settle never coalesces away a persist (or vice
        # versa).
        job.coalesce_key = (
            f"snapshot_settle:{entry.session_id}"
            if entry.session_id
            else f"snapshot_settle:hash:{entry.token_hash}"
        )
        try:
            dispatch(job)
            if timing_out is not None:
                timing_out["snapshot_settle"] = {"dispatched": True}
        except BaseException as exc:
            self.eviction_log.append(
                {
                    "reason": "snapshot_settle_dispatch_error",
                    "session_id": entry.session_id,
                    "prefix_len": entry.prefix_len,
                    "token_hash": entry.token_hash,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    def _enqueue_cold_entry(
        self,
        entry: SessionBankEntry,
        timing_out: dict[str, Any] | None = None,
    ) -> None:
        cold: dict[str, Any] | None = None
        if timing_out is not None:
            cold = {}
            timing_out["cold_enqueue"] = cold
        if entry.live_ref_only:
            if cold is not None:
                cold["enabled"] = False
                cold["skip_reason"] = "live_ref_only"
            # #323: live-ref-only entries used to end here — the SSD tier
            # never saw the exact sessions whose re-prefill costs minutes.
            # The streaming spill re-derives a snapshot at idle time.
            self._schedule_live_ref_spill(entry)
            return
        if self.cold_tier is None:
            if cold is not None:
                cold["enabled"] = False
                cold["skip_reason"] = "no_cold_tier"
            return
        put_entry = getattr(self.cold_tier, "put_entry", None)
        if not callable(put_entry):
            if cold is not None:
                cold["enabled"] = False
                cold["skip_reason"] = "no_put_entry"
            return
        if cold is not None:
            cold["enabled"] = True
        dispatch = self.cold_enqueue_dispatch
        if dispatch is not None:
            # Idle-lane path: the job reads the immutable bank entry (its
            # arrays are settled snapshot copies, so buffer donation on the
            # live cache cannot corrupt what gets encoded) on the model
            # owner thread, keeping the full-KV byte encode out of the
            # request/stream tail.
            dispatch_started = time.perf_counter()
            try:
                job = lambda: self._cold_enqueue_job(entry, put_entry)  # noqa: E731
                # Stable logical key for newest-wins coalescing of PENDING
                # persistence work: each queued job pins its entry's
                # GB-scale snapshot until it runs, and under continuous
                # traffic the idle window may not arrive for many turns.
                # Per-session, only the newest entry's encode stays queued.
                # Attribute-carried so legacy dispatch wirings that ignore
                # it keep their exact behavior.
                job.coalesce_key = (
                    f"ssd_cold:{entry.session_id}"
                    if entry.session_id
                    else f"ssd_cold:hash:{entry.token_hash}"
                )
                dispatch(job)
                if cold is not None:
                    cold["deferred"] = True
                    cold["dispatch_elapsed_s"] = (
                        time.perf_counter() - dispatch_started
                    )
                return
            except BaseException as exc:
                self.eviction_log.append(
                    {
                        "reason": "ssd_enqueue_dispatch_error",
                        "session_id": entry.session_id,
                        "prefix_len": entry.prefix_len,
                        "token_hash": entry.token_hash,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if cold is not None:
                    cold["dispatch_elapsed_s"] = (
                        time.perf_counter() - dispatch_started
                    )
                    cold["dispatch_error"] = True
                # Fall through to the synchronous path.
        sync_started = time.perf_counter()
        self._cold_enqueue_job(entry, put_entry)
        if cold is not None:
            cold["deferred"] = False
            cold["synchronous_serialize_elapsed_s"] = (
                time.perf_counter() - sync_started
            )

    def _cold_enqueue_job(
        self, entry: SessionBankEntry, put_entry: Callable[..., Any]
    ) -> None:
        # The tier serializes only MATERIALIZED boundary records. An entry
        # whose records still sit behind the lazy loader (inherited from an
        # SSD-restored donor) would persist a boundary-less package and
        # silently downgrade its whole lineage on the next restart — so
        # hydrate first. Runs on the postcommit/idle lane; the loader fails
        # closed on a corrupt payload.
        entry._ensure_boundaries_loaded()
        capabilities = ["ar_insert"]
        if entry.logits is not None and entry.hidden is not None:
            capabilities.append("mtp_full")
        # Entries at/above the tier's staged-queue backlog budget could
        # never persist through put_entry (the fully encoded payload would
        # not fit the queue) — stream them instead. Same on-disk format.
        spill = getattr(self.cold_tier, "spill_entry", None)
        spill_threshold = getattr(self.cold_tier, "spill_threshold_bytes", None)
        use_spill = (
            callable(spill)
            and isinstance(spill_threshold, int)
            and int(entry.nbytes) >= int(spill_threshold)
        )
        try:
            try:
                if use_spill:
                    stored = spill(
                        entry, capabilities=capabilities, raise_on_yield=True
                    )
                else:
                    stored = put_entry(
                        entry, capabilities=capabilities, raise_on_yield=True
                    )
            except TypeError:
                # Cold tiers predating the foreground-yield contract (or test
                # doubles) take no raise_on_yield kwarg.
                stored = put_entry(entry, capabilities=capabilities)
            if stored:
                # On the exact entry object — never a hash-keyed map (a
                # replaced entry with the same token hash must stay lazy).
                entry.cold_encode_completed_at = time.monotonic()
        except ColdEncodeInterrupted:
            # A foreground request arrived mid-encode; the encode aborted at a
            # tensor boundary. Re-dispatch the same job for the next quiet
            # window — the coalesce key keeps at most one pending copy, and a
            # newer commit for the same session supersedes it (newest-wins).
            dispatch = self.cold_enqueue_dispatch
            if dispatch is not None:
                job = lambda: self._cold_enqueue_job(entry, put_entry)  # noqa: E731
                # Same key expression as the original dispatch site so the
                # retry coalesces with (and is superseded by) newer commits.
                job.coalesce_key = (
                    f"ssd_cold:{entry.session_id}"
                    if entry.session_id
                    else f"ssd_cold:hash:{entry.token_hash}"
                )
                try:
                    dispatch(job)
                except Exception:
                    pass
            return
        except Exception as exc:
            self.eviction_log.append(
                {
                    "reason": "ssd_enqueue_error",
                    "session_id": entry.session_id,
                    "prefix_len": entry.prefix_len,
                    "token_hash": entry.token_hash,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    def _schedule_live_ref_spill(self, entry: SessionBankEntry) -> None:
        """Queue an idle-lane streaming spill for a live-ref-only session.

        Issue #323 + #305 durability: oversized sessions hold only a live
        reference lease — displacement or restart used to cost a FULL
        re-prefill (514 s at 134k tokens in the #305 traces) because
        nothing durable ever existed. The job re-derives a lazy COW
        snapshot from the live cache AT RUN TIME (epoch-guarded, so a
        superseded commit is skipped) and streams it to the SSD tier
        tensor-by-tensor. Without an idle dispatcher there is no safe
        window for the encode, so the skip is recorded, not silent.
        """
        cold = self.cold_tier
        if cold is None or not callable(getattr(cold, "spill_entry", None)):
            return
        dispatch = self.cold_enqueue_dispatch
        coalesce_key = (
            f"ssd_cold:{entry.session_id}"
            if entry.session_id
            else f"ssd_cold:hash:{entry.token_hash}"
        )
        if dispatch is None:
            self.eviction_log.append(
                {
                    "reason": "ssd_spill_no_dispatch",
                    "session_id": entry.session_id,
                    "prefix_len": entry.prefix_len,
                    "token_hash": entry.token_hash,
                }
            )
            return
        token_ids = tuple(entry.token_ids)
        epoch = int(entry.snapshot_epoch)
        job = lambda: self.run_live_ref_spill(token_ids, epoch)  # noqa: E731
        job.coalesce_key = coalesce_key
        try:
            dispatch(job)
        except BaseException as exc:
            self.eviction_log.append(
                {
                    "reason": "ssd_spill_dispatch_error",
                    "session_id": entry.session_id,
                    "prefix_len": entry.prefix_len,
                    "token_hash": entry.token_hash,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    def run_live_ref_spill(
        self, token_ids: tuple[int, ...], snapshot_epoch: int
    ) -> bool:
        """Idle-lane body of the live-ref spill; safe to call directly.

        Looks up the CURRENT entry (never a captured one — holding the
        entry in the closure would pin a superseded session's whole KV in
        RAM), verifies the commit epoch, snapshots lazily (zero-copy COW
        views: later cache writes cannot mutate what gets encoded — the
        lazy-snapshot COW pin covers this), and streams to disk.
        """
        entry = self._entries.get(tuple(int(token) for token in token_ids))
        if entry is None or not entry.live_ref_only or entry.cache_ref is None:
            return False
        if int(entry.snapshot_epoch) != int(snapshot_epoch):
            # Superseded: the newer commit scheduled its own (coalesced) job.
            return False
        cold = self.cold_tier
        spill = getattr(cold, "spill_entry", None) if cold is not None else None
        if not callable(spill):
            return False
        try:
            snapshot = snapshot_cache_lazy_hybrid(entry.cache_ref)
            mtp_snapshot = (
                snapshot_cache_lazy_hybrid(entry.mtp_history_cache_ref)
                if entry.mtp_history_cache_ref is not None
                else None
            )
        except RuntimeError as exc:
            # e.g. the paged long-context guard refuses to materialize
            # active K/V arrays; recorded so trace can show why this
            # session stays restart-volatile.
            self.eviction_log.append(
                {
                    "reason": "ssd_spill_snapshot_unavailable",
                    "session_id": entry.session_id,
                    "prefix_len": entry.prefix_len,
                    "token_hash": entry.token_hash,
                    "error": str(exc),
                }
            )
            return False
        entry._ensure_boundaries_loaded()
        view = replace(
            entry,
            cache_snapshot=snapshot,
            mtp_history_snapshot=mtp_snapshot,
            nbytes=int(entry.oversized_nbytes or 0),
            live_ref_only=False,
            cache_ref=None,
            mtp_history_cache_ref=None,
        )
        capabilities = ["ar_insert"]
        if view.logits is not None and view.hidden is not None:
            capabilities.append("mtp_full")
        try:
            stored = spill(view, capabilities=capabilities, raise_on_yield=True)
        except ColdEncodeInterrupted:
            # A foreground request arrived mid-encode. Re-dispatch for the
            # next quiet window; the coalesce key keeps at most one pending
            # spill per session and newer commits supersede it.
            dispatch = self.cold_enqueue_dispatch
            if dispatch is not None:
                job = lambda: self.run_live_ref_spill(  # noqa: E731
                    token_ids, snapshot_epoch
                )
                job.coalesce_key = (
                    f"ssd_cold:{entry.session_id}"
                    if entry.session_id
                    else f"ssd_cold:hash:{entry.token_hash}"
                )
                try:
                    dispatch(job)
                except Exception:
                    pass
            return False
        except Exception as exc:
            self.eviction_log.append(
                {
                    "reason": "ssd_spill_error",
                    "session_id": entry.session_id,
                    "prefix_len": entry.prefix_len,
                    "token_hash": entry.token_hash,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return False
        if stored:
            entry.cold_encode_completed_at = time.monotonic()
        return bool(stored)

    def _restore_cold(
        self,
        runtime: MTPLXRuntime,
        token_ids: list[int] | tuple[int, ...],
        *,
        session_id: str | None,
        hidden_variant: str | None,
        template_hash: str | None,
        mtp_history_policy: str | None,
        draft_head_identity: str | None,
        policy_fingerprint: str | None,
    ) -> SessionBankRestore | None:
        if self.cold_tier is None:
            return None
        lookup = getattr(self.cold_tier, "lookup", None)
        if not callable(lookup):
            return None
        record = lookup(
            token_ids,
            model_path=str(runtime.model_path),
            mtp_enabled=bool(runtime.mtp_enabled),
            hidden_variant=hidden_variant,
            template_hash=template_hash,
            mtp_history_policy=mtp_history_policy,
            draft_head_identity=draft_head_identity,
            policy_fingerprint=policy_fingerprint,
        )
        if record is None:
            # Read the miss reason through the cheap accessor: stats() is the
            # observability surface and may schedule a store reconciliation
            # walk, which has no place on a lookup miss (a 41 s walk per cold
            # miss on a large bank, 2026-08-15). Duck-typed doubles that only
            # expose stats() keep working.
            if hasattr(self.cold_tier, "last_miss_reason"):
                cold_miss = self.cold_tier.last_miss_reason
            elif hasattr(self.cold_tier, "stats"):
                cold_miss = self.cold_tier.stats().get("last_miss_reason")
            else:
                cold_miss = None
            if cold_miss:
                self.last_miss_reason = str(cold_miss)
                if self.last_prefix_diagnostic is not None:
                    self.last_prefix_diagnostic["miss_reason"] = self.last_miss_reason
            return None
        if hidden_variant is not None and (
            getattr(record, "logits", None) is None
            or getattr(record, "hidden", None) is None
        ):
            self.last_miss_reason = "ssd_missing_mtp_generation_state"
            return None
        if (
            mtp_history_policy in _COMMITTED_CACHE_POLICIES
            and getattr(record, "mtp_history_snapshot", None) is None
        ):
            self.last_miss_reason = "ssd_missing_mtp_history"
            return None
        metadata = dict(getattr(record, "metadata", {}) or {})
        entry = SessionBankEntry(
            token_ids=tuple(int(token) for token in record.token_ids),
            token_hash=metadata.get("token_hash") or token_prefix_hash(record.token_ids),
            has_recurrent=bool(
                getattr(record, "has_recurrent", False)
                or metadata.get("has_recurrent", False)
            ),
            gdn_boundaries=list(getattr(record, "gdn_boundaries", None) or []),
            gdn_boundary_loader=getattr(record, "gdn_boundary_loader", None),
            model_path=str(metadata.get("model_path") or runtime.model_path),
            mtp_enabled=bool(metadata.get("mtp_enabled", runtime.mtp_enabled)),
            hidden_variant=metadata.get("hidden_variant"),
            cache_snapshot=record.cache_snapshot,
            logits=_clone_tree(record.logits),
            hidden=_clone_tree(record.hidden),
            nbytes=int(getattr(record, "nbytes", 0) or 0),
            session_id=session_id or metadata.get("session_id"),
            template_hash=metadata.get("template_hash"),
            mtp_history_policy=metadata.get("mtp_history_policy"),
            draft_head_identity=metadata.get("draft_head_identity"),
            policy_fingerprint=metadata.get("policy_fingerprint"),
            mtp_history_snapshot=_clone_tree(record.mtp_history_snapshot),
            snapshot_epoch=int(metadata.get("snapshot_epoch") or len(record.token_ids)),
            mtp_snapshot_epoch=(
                int(metadata["mtp_snapshot_epoch"])
                if metadata.get("mtp_snapshot_epoch") is not None
                else (
                    int(metadata.get("snapshot_epoch") or len(record.token_ids))
                    if record.mtp_history_snapshot is not None
                    else None
                )
            ),
        )
        if (
            entry.mtp_snapshot_epoch is not None
            and int(entry.mtp_snapshot_epoch) != int(entry.snapshot_epoch)
        ):
            self.last_miss_reason = CacheMissReason.SNAPSHOT_DESYNC.value
            return None
        cache = runtime.make_cache()
        restore_cache(cache, entry.cache_snapshot)
        mtp_history_cache = None
        if entry.mtp_history_snapshot is not None:
            mtp_history_cache = runtime.make_mtp_cache()
            restore_cache(mtp_history_cache, entry.mtp_history_snapshot)
        entry.hits += 1
        entry.last_access_s = time.time()
        self._entries[entry.token_ids] = entry
        self._evict_if_needed(protected_tokens=entry.token_ids)
        self.last_restore_source = "ssd"
        self.last_ssd_restore_s = float(getattr(record, "restore_s", 0.0) or 0.0)
        self.last_miss_reason = None
        lookup_len = len(tuple(int(token) for token in token_ids))
        self.last_prefix_diagnostic = {
            "prompt_len": lookup_len,
            "session_id": entry.session_id,
            "stored_prefix_len": entry.prefix_len,
            "common_prefix_tokens": entry.prefix_len,
            "nearest_boundary_tokens": entry.prefix_len,
            "new_prefill_tokens": max(0, lookup_len - entry.prefix_len),
            "miss_reason": None,
            "restore_kind": "ssd_prefix",
        }
        return SessionBankRestore(
            entry=entry,
            cache=cache,
            logits=_clone_tree(entry.logits),
            hidden=_clone_tree(entry.hidden),
            restored_nbytes=entry.nbytes,
            restore_mode="ssd_clone",
            mtp_history_snapshot=_clone_tree(entry.mtp_history_snapshot),
            mtp_history_cache=mtp_history_cache,
            cache_source="ssd",
            ssd_cache_hit=True,
            ssd_cached_tokens=entry.prefix_len,
            ssd_restore_s=self.last_ssd_restore_s,
            extra_state=_clone_tree(entry.extra_state),
        )

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [
            entry
            for entry in self._entries.values()
            if now - float(entry.last_access_s) > self.idle_ttl_s
        ]
        for entry in expired:
            self._evict_entry(entry, reason=CacheMissReason.EVICTED.value)

    def _session_nbytes(self, session_id: str | None) -> int:
        return sum(
            entry.nbytes
            for entry in self._entries.values()
            if entry.session_id == session_id
        )

    def _supersede_contained_prefixes(self, tokens: tuple[int, ...]) -> None:
        """Evict RAM entries that are strict token-prefixes of a new entry.

        Agent sessions bank one entry per round, and each round's canonical
        transcript strictly extends the last — measured 2026-07-04: a single
        OpenCode conversation held 13/16 RAM slots (20.6 of 24 GB), a third of
        them strict prefixes of a newer entry. Multitasking across projects
        then churned every other project out of RAM ("sometimes I see a long
        prefill"). A container entry dominates its contained prefixes for
        every restore shape (exact hits trim; boundary-true restores pick a
        boundary <= the matched point), so the contained entries are pure
        redundancy — but only when the restore-compat identity (model,
        template, policy fingerprint, MTP policy, draft head, hidden variant)
        matches; a differing fingerprint can serve requests the container
        cannot. The SSD cold tier is untouched: superseded prefixes remain
        restorable from disk.
        """
        container = self._entries.get(tokens)
        if container is None:
            return
        if container.live_ref_only:
            # A live-reference lease is consumed by its first restore; it
            # cannot stand in for durable snapshot entries.
            return
        if container.has_recurrent and not (
            container.gdn_boundaries or container.gdn_boundary_loader is not None
        ):
            # Without recurrent boundaries the container cannot serve
            # sub-prefix restores, so contained entries still add coverage.
            return
        victims = [
            entry
            for key, entry in self._entries.items()
            if key != tokens
            and entry.prefix_len < len(tokens)
            and tokens[: entry.prefix_len] == entry.token_ids
            and entry.cache_ref is None
            and entry.model_path == container.model_path
            and entry.template_hash == container.template_hash
            and entry.policy_fingerprint == container.policy_fingerprint
            and entry.mtp_history_policy == container.mtp_history_policy
            and entry.draft_head_identity == container.draft_head_identity
            and entry.hidden_variant == container.hidden_variant
        ]
        for entry in victims:
            self._evict_entry(entry, reason="superseded_by_longer_prefix")
        self._enforce_session_entry_retention(
            container.session_id, protected_tokens=tokens
        )

    def _enforce_session_entry_retention(
        self, session_id: str | None, *, protected_tokens: tuple[int, ...]
    ) -> None:
        """Keep only the newest-K RAM entries for a session (see
        _per_session_max_entries). Live-reference leases are exempt: they are
        consumed by their first restore and carry no snapshot bytes to shed.
        """
        cap = self.per_session_max_entries
        if not session_id or cap <= 0:
            return
        entries = [
            entry
            for entry in self._entries.values()
            if entry.session_id == session_id and not entry.live_ref_only
        ]
        if len(entries) <= cap:
            return
        entries.sort(
            key=lambda entry: (
                entry.token_ids == protected_tokens,
                entry.last_access_s,
                entry.created_at_s,
            ),
            reverse=True,
        )
        for entry in entries[cap:]:
            self._evict_entry(entry, reason="session_entry_retention")

    def _shed_boundaries_to_fit(
        self,
        boundaries: list[tuple[int, CacheSnapshot, Any]],
        entry_nbytes: int,
    ) -> tuple[list[tuple[int, CacheSnapshot, Any]], int, int]:
        """Drop boundary records until the entry fits its per-session budget.

        Returns ``(kept, nbytes, shed_count)``. ``kept`` keeps the input's
        ascending-position ordering, so callers see the same shape they passed.

        Records are dropped FURTHEST-FROM-THE-TAIL first, and the newest record
        is never dropped while any is kept. That is deliberately not
        ``generation._thin_gdn_boundary_records``'s geometric policy, which
        preserves a deep-divergence anchor: under byte pressure the anchor is
        the first thing that has to go, because agent divergence concentrates
        near the prompt tail (the same reason MTPLX_GDN_BOUNDARY_TAIL_INTERVAL
        gives the final chunk a finer grid). The trade is explicit: a
        deep-divergence match may find no boundary at or below it and fail
        closed to a cold prefill -- which is exactly what it does TODAY, when
        the whole entry is refused. Shedding is never worse and is usually the
        difference between a 0.5 s restore and a 15.8 s re-prefill.

        Shedding stops at zero records: an entry whose BASE snapshot already
        exceeds the budget is still refused by the caller, unchanged.
        """

        budget = int(self.per_session_max_bytes)
        sizes = [
            _snapshot_nbytes(record[1]) + _tree_nbytes(record[2])
            for record in boundaries
        ]
        total = int(entry_nbytes)
        kept = list(boundaries)
        kept_sizes = list(sizes)
        shed = 0
        # kept/kept_sizes are sorted ascending by position (put normalizes
        # them), so index 0 is the record furthest from the tail.
        while kept and total > budget:
            total -= kept_sizes.pop(0)
            kept.pop(0)
            shed += 1
        return kept, int(total), int(shed)

    def _newest_extending_entry(
        self, tokens: tuple[int, ...] | None
    ) -> SessionBankEntry | None:
        """The newest banked entry that STRICTLY extends ``tokens``.

        Port of oMLX ``_newest_extending_entry`` (exact_resident.py:87-103):
        insertion recency wins, deliberately NOT length -- a longer but older
        branch of a diverged transcript is not the branch the client is on.
        ``self._entries`` is a plain dict, so ``reversed()`` is newest-first by
        insertion for exactly the same reason oMLX's is.

        Returns None when the gate is off, so the whole rule costs one boolean
        read on the default path.
        """

        if not self.protect_newest_extending or not tokens:
            return None
        width = len(tokens)
        for key, entry in reversed(self._entries.items()):
            if len(key) > width and key[:width] == tokens:
                return entry
        return None

    def _evict_if_needed(self, *, protected_tokens: tuple[int, ...] | None = None) -> None:
        while True:
            if not self._entries:
                return
            session_over_budget = {
                entry.session_id
                for entry in self._entries.values()
                if self._session_nbytes(entry.session_id) > self.per_session_max_bytes
            }
            reason: str | None = None
            over_budget_only = False
            candidates = list(self._entries.values())
            if len(self._entries) > self.max_entries:
                reason = CacheMissReason.EVICTED.value
            elif self.total_nbytes > self.effective_max_bytes():
                reason = CacheMissReason.EVICTED.value
            elif session_over_budget:
                reason = CacheMissReason.EVICTED.value
                over_budget_only = True
                candidates = [
                    entry
                    for entry in candidates
                    if entry.session_id in session_over_budget
                ]
            else:
                return

            unprotected = [
                entry
                for entry in candidates
                if protected_tokens is None or entry.token_ids != protected_tokens
            ]
            if unprotected:
                candidates = unprotected
            elif len(candidates) == 1:
                entry = candidates[0]
                if (
                    entry.nbytes > self.per_session_max_bytes
                    or entry.nbytes > self.max_bytes
                ):
                    self._evict_entry(entry, reason=reason or CacheMissReason.EVICTED.value)
                    continue
                return
            if not over_budget_only:
                # Cross-session pressure prefers idle victims: a session that
                # touched the bank within the active-pin TTL is mid-run, and
                # evicting it forces a full re-prefill on its very next turn
                # (the 2026-07-31 85.6k live incident). A session over its own
                # per-session budget still self-evicts oldest-first above.
                active = self._active_session_ids()
                if active:
                    idle = [
                        entry
                        for entry in candidates
                        if entry.session_id not in active
                    ]
                    if idle:
                        candidates = idle
            victim = min(
                candidates,
                key=lambda entry: (entry.last_access_s, -entry.nbytes, entry.created_at_s),
            )
            terminal = self._newest_extending_entry(protected_tokens)
            if terminal is not None and victim is terminal and len(candidates) > 1:
                # Protected-terminal rule (see _protected_terminal_enabled):
                # the shorter fallback being published must not be what evicts
                # the newest terminal extending it. Order only -- if the
                # terminal is the LAST candidate standing it is still evicted,
                # because the budget has to be met and refusing here would
                # spin this loop forever.
                self.protected_rejections += 1
                candidates = [entry for entry in candidates if entry is not terminal]
                victim = min(
                    candidates,
                    key=lambda entry: (
                        entry.last_access_s,
                        -entry.nbytes,
                        entry.created_at_s,
                    ),
                )
            self._evict_entry(victim, reason=reason)

    def shrink_to_bytes(
        self,
        target_bytes: int,
        *,
        reason: str = "memory_pressure",
        protect_active: bool = False,
    ) -> int:
        """Evict least-recently-used entries until the bank fits the target.

        The memory-pressure guard calls this when macOS reports system-wide
        pressure (issue #144: a 64 GB Mac swapping 60 GB while the bank sat
        on its full budget). Returns the number of entries evicted.

        ``protect_active=True`` (the dynamic-ceiling caller) never evicts an
        active session's entries — the bank may stay above the target. The
        ceiling subtracts an instantaneous working-set reading, so a deep
        prefill's transient spike reads as a standing commitment; evicting
        the live session's own prefix chain to absorb it trades a
        seconds-long spike for a 50+ second re-prefill on the very next turn
        (2026-08-28 receipt: a 93k OpenCode session's bank was walked to 0
        bytes mid-request, TTFT 54-57 s after). Real macOS pressure keeps
        take-anything semantics — active sessions merely sort last there.
        """

        evicted = 0
        target = max(0, int(target_bytes))
        active = self._active_session_ids()
        while self._entries and self.total_nbytes > target:
            candidates = self._entries.values()
            if protect_active and active:
                candidates = [
                    entry
                    for entry in candidates
                    if entry.session_id not in active
                ]
                if not candidates:
                    break
            victim = min(
                candidates,
                # Real memory pressure may take anything, but active sessions
                # go last so the responder doesn't force a mid-run re-prefill
                # while idle entries were available.
                key=lambda entry: (
                    entry.session_id in active,
                    entry.last_access_s,
                    -entry.nbytes,
                    entry.created_at_s,
                ),
            )
            before = len(self._entries)
            self._evict_entry(victim, reason=reason)
            if len(self._entries) >= before:
                # Defensive: an entry whose dict key drifted from its
                # token_ids would make this loop spin forever while
                # total_nbytes never shrinks (allocating an eviction-log
                # record per iteration). Should be unreachable — put() keys
                # strictly by token_ids — but an infinite allocator loop is
                # never an acceptable failure mode for a pressure responder.
                break
            evicted += 1
        return evicted

    def shrink_for_admission(
        self,
        target_bytes: int,
        *,
        protect_tokens: list[int] | tuple[int, ...] | None = None,
        reason: str = "prefill_admission_chain",
    ) -> tuple[int, int]:
        """Escalating eviction for the admission shed (#447).

        Runs only when the pre-prefill projection says the request in front
        of us is likely to die on the sustained-pressure abort, after the
        superseded clear and the ``protect_active`` LRU pass both came up
        short: a deep session's sibling snapshots — forked generations of
        the same conversation whose retokenized histories diverge, so
        ``_supersede_contained_prefixes`` never collapses them — are all
        active-protected there. A 12.6 GiB bank served a 7 GiB deficit
        with zero evictions and the request 507'd.

        Phase 1 walks non-terminal entries (any session keeps its highest
        ``prefix_len`` entry — the one the protected-terminal order guards).
        Phase 2, only if the deficit stands, takes remaining entries in the
        take-anything order of real memory pressure (active sessions last).
        Both phases spare the entry the imminent prompt restores from
        (``protect_tokens``), and every eviction is RAM-only: the SSD cold
        tier still restores a walked entry, so the worst case is a disk
        read on some session's next turn, not this request's abort.
        Returns ``(non_terminal_evicted, terminal_evicted)``.
        """
        target = max(0, int(target_bytes))
        protected_keys: set[tuple[int, ...]] = set()
        if protect_tokens:
            tokens = tuple(int(token) for token in protect_tokens)
            best_key = None
            best_common = 0
            for key, entry in self._entries.items():
                common = common_prefix_len(tokens, entry.token_ids)
                if common > best_common:
                    best_common = common
                    best_key = key
            if best_key is not None:
                protected_keys.add(best_key)

        def _walk(candidates_fn, order_key) -> int:
            evicted = 0
            while self._entries and self.total_nbytes > target:
                candidates = candidates_fn()
                if not candidates:
                    break
                victim = min(candidates, key=order_key)
                before = len(self._entries)
                self._evict_entry(victim, reason=reason)
                if len(self._entries) >= before:
                    break
                evicted += 1
            return evicted

        def _evictable(entry) -> bool:
            # Entries holding a live cache reference are the live session's
            # own arrays (the same bar _supersede_contained_prefixes sets);
            # walking one frees nothing and costs the running session its
            # state. Measured: the first decode after such an eviction ran
            # at 15 tok/s against 63 stock.
            return entry.cache_ref is None and not entry.live_ref_only

        def _non_terminal_candidates():
            terminal: dict[str, int] = {}
            for entry in self._entries.values():
                lineage = entry.session_id or ""
                if entry.prefix_len > terminal.get(lineage, -1):
                    terminal[lineage] = entry.prefix_len
            return [
                entry
                for key, entry in self._entries.items()
                if key not in protected_keys
                and _evictable(entry)
                and entry.prefix_len < terminal.get(entry.session_id or "", -1)
            ]

        non_terminal = _walk(
            _non_terminal_candidates,
            lambda entry: (
                entry.last_access_s,
                -entry.nbytes,
                entry.created_at_s,
            ),
        )
        if self.total_nbytes <= target:
            return non_terminal, 0

        active = self._active_session_ids()
        terminal_evicted = _walk(
            lambda: [
                entry
                for key, entry in self._entries.items()
                if key not in protected_keys and _evictable(entry)
            ],
            lambda entry: (
                entry.session_id in active,
                entry.last_access_s,
                -entry.nbytes,
                entry.created_at_s,
            ),
        )
        return non_terminal, terminal_evicted

    def _evict_entry(self, entry: SessionBankEntry, *, reason: str) -> None:
        entry.eviction_reason = reason
        if self._entries.pop(entry.token_ids, None) is None:
            for key, value in list(self._entries.items()):
                if value is entry:
                    self._entries.pop(key, None)
                    break
        self.eviction_log.append(
            {
                "reason": reason,
                "session_id": entry.session_id,
                "prefix_len": entry.prefix_len,
                "token_hash": entry.token_hash,
                "nbytes": entry.nbytes,
                "last_access_s": entry.last_access_s,
                "session_active": bool(
                    entry.session_id
                    and entry.session_id in self._active_session_ids()
                ),
            }
        )


def prefill_target(
    runtime: MTPLXRuntime,
    token_ids: list[int],
    *,
    return_hidden: bool = True,
) -> tuple[list[Any], Any, Any | None, float]:
    """Prefill using the same all-but-last/last-token split as generation."""
    if not token_ids:
        raise ValueError("token_ids must not be empty")
    cache = runtime.make_cache()
    elapsed = 0.0
    if len(token_ids) > 1:
        started = time.perf_counter()
        prefill = runtime.forward_ar(
            mx.array([token_ids[:-1]]),
            cache=cache,
            return_hidden=False,
        )
        mx.eval(prefill)
        elapsed += time.perf_counter() - started

    started = time.perf_counter()
    result = runtime.forward_ar(
        mx.array([[token_ids[-1]]]),
        cache=cache,
        return_hidden=return_hidden,
    )
    if return_hidden:
        logits, hidden_seq = result
        mx.eval(logits, hidden_seq)
        hidden = hidden_seq[:, -1:, :]
    else:
        logits = result
        hidden = None
        mx.eval(logits)
    elapsed += time.perf_counter() - started
    return cache, logits[:, -1, :], hidden, elapsed


def prefill_target_with_session_bank(
    runtime: MTPLXRuntime,
    token_ids: list[int],
    bank: SessionBank,
    *,
    return_hidden: bool = True,
    restore_mode: str = "clone",
) -> tuple[list[Any], Any, Any | None, float, dict[str, Any]]:
    started_total = time.perf_counter()
    restored = bank.restore(runtime, token_ids, mode=restore_mode)
    if restored is None:
        cache, logits, hidden, elapsed = prefill_target(
            runtime,
            token_ids,
            return_hidden=return_hidden,
        )
        return cache, logits, hidden, elapsed, {
            "hit": False,
            "prefix_len": 0,
            "suffix_len": len(token_ids),
        }

    suffix = list(token_ids[restored.entry.prefix_len :])
    if not suffix:
        elapsed = time.perf_counter() - started_total
        return restored.cache, restored.logits, restored.hidden, elapsed, {
            "hit": True,
            "prefix_len": restored.entry.prefix_len,
            "suffix_len": 0,
            "restored_nbytes": restored.restored_nbytes,
            "restore_included_s": elapsed,
            "restore_mode": restore_mode,
        }

    elapsed_suffix = 0.0
    if len(suffix) > 1:
        started = time.perf_counter()
        prefill = runtime.forward_ar(
            mx.array([suffix[:-1]]),
            cache=restored.cache,
            return_hidden=False,
        )
        mx.eval(prefill)
        elapsed_suffix += time.perf_counter() - started

    started = time.perf_counter()
    result = runtime.forward_ar(
        mx.array([[suffix[-1]]]),
        cache=restored.cache,
        return_hidden=return_hidden,
    )
    if return_hidden:
        logits, hidden_seq = result
        mx.eval(logits, hidden_seq)
        hidden = hidden_seq[:, -1:, :]
    else:
        logits = result
        hidden = None
        mx.eval(logits)
    elapsed_suffix += time.perf_counter() - started
    elapsed_total = time.perf_counter() - started_total
    return restored.cache, logits[:, -1, :], hidden, elapsed_total, {
        "hit": True,
        "prefix_len": restored.entry.prefix_len,
        "suffix_len": len(suffix),
        "restored_nbytes": restored.restored_nbytes,
        "suffix_forward_s": elapsed_suffix,
        "restore_and_suffix_s": elapsed_total,
        "restore_mode": restore_mode,
    }


def max_abs_diff(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    diff = mx.abs(left.astype(mx.float32) - right.astype(mx.float32))
    mx.eval(diff)
    return float(np.max(np.asarray(diff)))
