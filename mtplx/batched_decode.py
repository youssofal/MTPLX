"""Multi-stream (cross-request) batched greedy decode for A3B — Phase 1.

WHY THIS EXISTS
---------------
Single-stream A3B decode is latency-bound and kernel-closed (~167 tok/s, §41 of
``claude-s3-serving-integration-build-20260721.md``).  The one remaining
throughput lever is CROSS-REQUEST BATCHING: run ``B`` concurrent decode requests
as ONE ``[B, ·]`` forward per cycle so the ~1054.8 MB of dense weights
(attn/GDN/router/shared-expert/lm_head) are read ONCE and amortized across all
``B`` streams.  The eager probe ``a3b_174_batch_upside_bench.py`` measured this
amortization at ×2.49 ideal / ×2.21 net-ragged @ B=8 (§42/§43).  This module is
the *running decode* that realizes it (the probe timed a bare ``forward_ar``; it
never decoded).

WHAT THIS IS (Phase 1) vs WHAT IT IS NOT (Phase 2)
--------------------------------------------------
This is a GREEDY, uniform-commit multi-stream driver on the BATCH-GENERIC cache
lane (stock KV / GDN caches + stock attention — NOT the served
``VllmMetalPagedKVCache``, which hard-raises at batch>1, ``cache_state.py:955``).
Each cycle:

  1. ``x0_b = argmax(logits_b)``         — the next greedy token per stream.
  2. draft ``d_b`` from the MTP head     — one ``[B,1]`` draft forward.
  3. VERIFY ``forward_ar([B,2])`` on ``[x0_b, d_b]`` — the single amortized
     weight read the probe measured; advances every stream's cache by 2.
  4. ``x1_b = argmax(verify[:,0])``      — the true 2nd greedy token per stream.
     ``accept_b = (d_b == x1_b)``.
  5. If EVERY stream accepted: the verify already put ``x1`` at position O+1 for
     all, so keep it — 1 forward committed 2 tokens for all B (the speculative
     win).  Otherwise: roll the WHOLE batch back to O and re-forward
     ``[B,2] = [x0_b, x1_b]`` (the correct 2 greedy tokens) — a UNIFORM full-B
     repair that keeps the single shared cache offset (Phase-1 constraint).

Because sampling is greedy, the committed sequence per stream is exactly the
target model's greedy-argmax continuation ``x0, x1, x2, …`` — the SAME sequence
regardless of the accept pattern, and byte-identical to that stream run alone
through this driver.  Crucially, for a stream that WOULD have accepted, the
repair re-forward of ``[x0, x1]`` is bit-identical to the verify it replaces
(same tokens, same prefix, same weights, deterministic forward), so a rejecting
neighbour never perturbs an accepting stream.  **That determinism is the Phase-1
correctness contract** (proved on CPU with a fake runtime; the per-stream sha
gate on the real model is fable-main's GPU window).

Phase-1 SCOPE HONESTY: the uniform full-B repair is CORRECT but pays the full
``[B,2]`` weight read again whenever ANY stream rejects — with independent
streams that is most cycles, so this realizes the cross-request amortization but
NOT the §43 compacted-repair economics (repair only the rejecting rows).  The
COMPACTED repair sub-batch (``filter``/merge on the batch-generic cache) is the
plan's hard Phase 2 and is deliberately NOT built here — see the module-level
``PHASE2_REMAINING`` note.  Greedy-only; a p/q ratio-accept (temperature>0) lane
is also Phase 2.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mtplx.sampling import (
    Distribution,
    SamplerConfig,
    SparseDistribution,
    apply_penalties,
    distribution_from_logits,
    sample_from_distribution,
    verify_one_token,
)

# --------------------------------------------------------------------------- #
# Env gate (fail-closed).  Phase 1 calls ``generate_greedy_batched`` directly
# from the bench; this flag is the seam a future served path (Phase 3) checks
# before routing a cohort here.  OFF => callers never touch this module, so a
# gate-off run is byte-identical to single-stream ``generate_mtpk``.
# --------------------------------------------------------------------------- #
BATCHED_DECODE_ENV = "MTPLX_A3B_BATCHED_DECODE"
# Build-1 fallback: force the exact Phase-1 SERIAL loop (4-5 blocking syncs per
# cycle) instead of the default single-sync pipelined loop.  This is the A/B
# switch — set it to compare the parallelized scheduling against the serial
# baseline WITHOUT any change to the committed token stream (the loop is a pure
# scheduling change; both commit the identical greedy sequence).  The ``serial=``
# argument overrides this env when passed explicitly.
BATCHED_DECODE_SERIAL_ENV = "MTPLX_A3B_BATCHED_DECODE_SERIAL"
# Reject-handling mode (scheme doc §2.2, Build-2 kill-check).  DEFAULT "repair"
# is fail-closed: the exact Build-1 uniform full-B repair loop, byte-identical
# when off.  "foldin" selects the FOLD-IN REPLAY loop on the ragged-KV lane: a
# missed row re-enters the next cycle one position back with [x0_prev, x1] (no
# separate repair forward), its recurrent state rewound per-row.  Env fallback;
# the ``reject_mode=`` argument overrides it when passed explicitly.
BATCHED_DECODE_REJECT_ENV = "MTPLX_A3B_BATCHED_DECODE_REJECT"
# Fallback knob (default OFF): restore the legacy eager clone snapshot for the
# fold-in loop's per-cycle recurrent snapshot.  The default (OFF) uses the lazy
# zero-copy view snapshot (COW-safe: the GDN forward and the masked REPLAY
# rewind both rebind cache slots, never mutate a snapshot buffer in place).
FOLDIN_CLONE_SNAPSHOT_ENV = "MTPLX_A3B_FOLDIN_CLONE_SNAPSHOT"
_TRUTHY = {"1", "true", "yes", "on"}
_REJECT_MODES = {"repair", "foldin"}

# What a real cross-request serving build still needs beyond this module
# (recorded so the Phase-1/Phase-2 boundary is unambiguous):
PHASE2_REMAINING = (
    "compacted repair sub-batch (filter rejecting rows -> [B_reject,2] repair -> "
    "scatter KV back, vs the uniform full-B repair here); per-stream staggered "
    "offsets / ragged-KV for long context (this driver holds ONE shared cache "
    "offset, so all prompts must be equal length); dynamic admission/departure "
    "(mtplx/batching scheduler); and a p/q ratio-accept (temperature>0) lane."
)


def batched_decode_enabled(environ: dict[str, str] | None = None) -> bool:
    """True iff ``MTPLX_A3B_BATCHED_DECODE`` is set truthy.  Fail-closed."""
    env = os.environ if environ is None else environ
    return str(env.get(BATCHED_DECODE_ENV, "")).strip().lower() in _TRUTHY


def batched_decode_serial(environ: dict[str, str] | None = None) -> bool:
    """True iff ``MTPLX_A3B_BATCHED_DECODE_SERIAL`` is set truthy.  Fail-closed.

    The default (False) selects the Build-1 single-sync pipelined loop.
    """
    env = os.environ if environ is None else environ
    return str(env.get(BATCHED_DECODE_SERIAL_ENV, "")).strip().lower() in _TRUTHY


def batched_decode_reject_mode(environ: dict[str, str] | None = None) -> str:
    """Reject-handling mode from ``MTPLX_A3B_BATCHED_DECODE_REJECT``.

    Returns ``"foldin"`` iff the env is set to ``foldin`` (case-insensitive);
    otherwise ``"repair"`` (the default, fail-closed Build-1 behaviour).  Any
    unrecognized value falls back to ``"repair"``.
    """
    env = os.environ if environ is None else environ
    value = str(env.get(BATCHED_DECODE_REJECT_ENV, "")).strip().lower()
    return value if value in _REJECT_MODES else "repair"


def foldin_clone_snapshot(environ: dict[str, str] | None = None) -> bool:
    """True iff the fold-in loop must use the legacy EAGER clone snapshot.

    From ``MTPLX_A3B_FOLDIN_CLONE_SNAPSHOT``.  Default (False) selects the lazy
    zero-copy view snapshot for the per-cycle recurrent snapshot (the new
    default); setting the env truthy restores the ``_clone_tree`` materialization
    as a fallback.  Fail-closed to the fast (lazy) path.  Affects ONLY the
    fold-in loop -- the serial/pipelined scalar-repair lanes keep the eager
    ``snapshot_untrimmable_cache`` unchanged.
    """
    env = os.environ if environ is None else environ
    return str(env.get(FOLDIN_CLONE_SNAPSHOT_ENV, "")).strip().lower() in _TRUTHY


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class BatchedStreamResult:
    index: int
    prompt_len: int
    tokens: list[int]
    finish_reason: str
    sha: str


@dataclass
class BatchedDecodeResult:
    batch_size: int
    streams: list[BatchedStreamResult]
    cycles: int
    forwards: int
    all_accept_cycles: int
    repair_cycles: int
    prefill_s: float
    decode_s: float
    generated_tokens: int
    # FOLD-IN telemetry: total REPLAY row-cycles (== total misses; each miss is
    # deferred one cycle and replayed).  Named to line up with upstream's
    # ``deferred_correction_repairs``.  Zero in repair mode (no fold-in).
    replay_rows: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def aggregate_decode_tokps(self) -> float:
        return self.generated_tokens / self.decode_s if self.decode_s > 0 else 0.0

    @property
    def shas(self) -> list[str]:
        return [s.sha for s in self.streams]


@dataclass(frozen=True)
class MTPK1RowCycle:
    """One request-owned depth-one speculative sampling decision."""

    primary_token: int
    draft_token: int
    accepted: bool
    second_token: int
    bonus_token: int | None
    accept_probability: float
    next_primary: int | None


@dataclass(frozen=True)
class _MTPK1RowProposal:
    primary_token: int
    draft_token: int
    draft_distribution: Distribution


# --------------------------------------------------------------------------- #
# Pure helpers (no MLX — unit-drivable)
# --------------------------------------------------------------------------- #
def _greedy_token_from_logits(
    logits: np.ndarray,
    sampler: SamplerConfig,
    *,
    token_counts: Counter[int] | None = None,
) -> int:
    adjusted = apply_penalties(
        np.asarray(logits),
        token_counts,
        sampler.presence_penalty,
        sampler.frequency_penalty,
    )
    return int(np.argmax(adjusted))


def token_sha(tokens: list[int]) -> str:
    """Stable 16-hex digest of a committed token sequence (per-stream gate key)."""
    payload = json.dumps([int(t) for t in tokens], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _sample_mtp_k1_primary(
    primary_logits: np.ndarray,
    *,
    sampler: SamplerConfig,
    rng: np.random.Generator,
    history_tokens: list[int],
    pending_primary: int | None = None,
) -> int:
    """Sample one request-owned primary, or reuse its emitted pending token."""
    counts = Counter(int(token) for token in history_tokens)
    if pending_primary is None:
        if sampler.temperature <= 0:
            primary = _greedy_token_from_logits(
                primary_logits, sampler, token_counts=counts
            )
        else:
            primary_p = distribution_from_logits(
                np.asarray(primary_logits, dtype=np.float64),
                sampler,
                token_counts=counts,
            )
            primary = sample_from_distribution(primary_p, rng)
    else:
        primary = int(pending_primary)
    return primary


def _sample_mtp_k1_draft(
    primary_token: int,
    draft_logits: np.ndarray,
    *,
    draft_sampler: SamplerConfig,
    rng: np.random.Generator,
) -> _MTPK1RowProposal:
    """Sample the row-owned draft after its primary has shaped the MTP forward."""

    if draft_sampler.temperature <= 0:
        draft = _greedy_token_from_logits(draft_logits, draft_sampler)
        draft_q: Distribution = SparseDistribution.one_hot(
            draft, int(np.asarray(draft_logits).shape[0])
        )
    else:
        draft_q = distribution_from_logits(
            np.asarray(draft_logits, dtype=np.float64),
            draft_sampler,
        )
        draft = sample_from_distribution(draft_q, rng)
    return _MTPK1RowProposal(
        primary_token=int(primary_token),
        draft_token=draft,
        draft_distribution=draft_q,
    )


def _sample_mtp_k1_row_proposal(
    primary_logits: np.ndarray,
    draft_logits: np.ndarray,
    *,
    sampler: SamplerConfig,
    draft_sampler: SamplerConfig,
    rng: np.random.Generator,
    history_tokens: list[int],
    pending_primary: int | None = None,
) -> _MTPK1RowProposal:
    """Sample the primary and draft needed to construct a target verify row."""
    primary = _sample_mtp_k1_primary(
        primary_logits,
        sampler=sampler,
        rng=rng,
        history_tokens=history_tokens,
        pending_primary=pending_primary,
    )
    return _sample_mtp_k1_draft(
        primary,
        draft_logits,
        draft_sampler=draft_sampler,
        rng=rng,
    )


def _finish_mtp_k1_row_cycle(
    proposal: _MTPK1RowProposal,
    verify_logits: np.ndarray,
    bonus_logits: np.ndarray | None,
    *,
    sampler: SamplerConfig,
    rng: np.random.Generator,
    history_tokens: list[int],
    omit_speculative_bonus: bool,
) -> MTPK1RowCycle:
    """Finish p/q verification after the fixed ``[B, 2]`` target forward."""
    primary = int(proposal.primary_token)
    draft = int(proposal.draft_token)
    draft_q = proposal.draft_distribution
    counts = Counter(int(token) for token in history_tokens)

    if sampler.temperature <= 0:
        target = _greedy_token_from_logits(
            verify_logits, sampler, token_counts=counts
        )
        accepted = draft == target
        second = draft if accepted else target
        accept_probability = 1.0 if accepted else 0.0
    else:
        target_p = distribution_from_logits(
            np.asarray(verify_logits, dtype=np.float64),
            sampler,
            token_counts=counts,
        )
        decision = verify_one_token(target_p, draft_q, draft, rng)
        accepted = bool(decision.accepted)
        second = int(decision.token_id)
        accept_probability = float(decision.accept_probability)
    counts[second] += 1

    bonus = None
    if accepted and not omit_speculative_bonus and bonus_logits is not None:
        if sampler.temperature <= 0:
            bonus = _greedy_token_from_logits(
                bonus_logits, sampler, token_counts=counts
            )
        else:
            bonus_p = distribution_from_logits(
                np.asarray(bonus_logits, dtype=np.float64),
                sampler,
                token_counts=counts,
            )
            bonus = sample_from_distribution(bonus_p, rng)

    return MTPK1RowCycle(
        primary_token=primary,
        draft_token=draft,
        accepted=accepted,
        second_token=second,
        bonus_token=bonus,
        accept_probability=accept_probability,
        next_primary=bonus if accepted else second,
    )


def _sample_mtp_k1_row_cycle(
    primary_logits: np.ndarray,
    draft_logits: np.ndarray,
    verify_logits: np.ndarray,
    bonus_logits: np.ndarray | None,
    *,
    sampler: SamplerConfig,
    draft_sampler: SamplerConfig,
    rng: np.random.Generator,
    history_tokens: list[int],
    omit_speculative_bonus: bool,
    pending_primary: int | None = None,
) -> MTPK1RowCycle:
    """Apply the single-request ``generate_mtpk`` K1 RNG order to one row.

    All inputs are already materialized row logits. The caller owns ``rng``;
    sharing or reordering other rows therefore cannot advance this request's
    random stream. Completion penalties cover committed output only, matching
    the solo path. A pending primary is already present in ``history_tokens``.
    Draft sampling intentionally does not consume completion counts.
    """
    proposal = _sample_mtp_k1_row_proposal(
        primary_logits,
        draft_logits,
        sampler=sampler,
        draft_sampler=draft_sampler,
        rng=rng,
        history_tokens=history_tokens,
        pending_primary=pending_primary,
    )
    committed_history = list(history_tokens)
    if pending_primary is None:
        committed_history.append(proposal.primary_token)
    return _finish_mtp_k1_row_cycle(
        proposal,
        verify_logits,
        bonus_logits,
        sampler=sampler,
        rng=rng,
        history_tokens=committed_history,
        omit_speculative_bonus=omit_speculative_bonus,
    )


def left_pad_prompts(
    prompts: list[list[int]], pad_id: int
) -> tuple[list[list[int]], list[int]]:
    """Left-pad a ragged prompt batch to a shared length with ``pad_id``.

    **On a hybrid GDN trunk this is SILENTLY INCORRECT, and it is kept only to
    reproduce historical results.** Use ``ragged_prompts=True`` on the dense MTP
    driver, or a ragged refill queue here, for anything whose output matters.

    The reason is specific to the architecture. Attention layers keep an
    addressable KV cache, so a per-row offset genuinely un-sees a pad prefix.
    GDN layers keep a RECURRENT state, into which every token this function
    prepends is folded, and no offset rewinds that -- the contamination is not
    stored positionally. A padded row's recurrent state entering its first real
    token is therefore not the zero state, and nothing complains: the model
    loads, runs, and returns fluent text conditioned on tokens the caller never
    sent.

    The guard rail for this EXISTS and is simply never switched on.
    ``create_ssm_mask(h, cache)`` is ``cache.make_mask(N)``, which returns a
    real per-row mask when ``left_padding`` or ``lengths`` is set on the cache
    entry, and threads it into ``gated_delta_update`` so a masked position
    leaves the state untouched. Nothing in this module sets either attribute,
    so ``make_mask`` returns ``None`` and every pad is folded in. Wiring it up
    would make padding CORRECT but not FREE: the rectangle is still
    ``[B, max_len]``, so a short prompt still pays the longest prompt's prefill
    flops. Prefilling each length group on its own costs neither, which is why
    that is what both drivers do.

    The batch-generic decode cache carries ONE shared offset, so every stream
    must enter the loop at the same length; that is the constraint this
    function was written for.  Left-padding keeps each stream's TRUE last token
    at the final position, so its next-token logits are its own, which is why
    the Phase-1 throughput gate could use it: the single-stream reference is fed
    the IDENTICAL padded prompt, so the gate compares batching isolation and
    makes no claim about prompt semantics.  Returns ``(padded, true_lengths)``.
    """
    if not prompts:
        raise ValueError("prompts must be non-empty")
    lengths = [len(p) for p in prompts]
    width = max(lengths)
    if width < 1:
        raise ValueError("each prompt needs at least one token")
    padded = [[int(pad_id)] * (width - len(p)) + [int(t) for t in p] for p in prompts]
    return padded, lengths


def diff_streams(
    batched: list[list[int]], reference: list[list[int]]
) -> list[dict[str, Any]]:
    """Per-stream sha comparison of a batched run vs its single-stream reference.

    Returns one record per stream with ``match`` and, on mismatch, the first
    differing position + a short window around it (so a GPU divergence is
    localized, not just pass/fail).  This is the Phase-1 correctness gate.
    """
    if len(batched) != len(reference):
        raise ValueError(
            f"stream count mismatch: batched {len(batched)} vs reference "
            f"{len(reference)}"
        )
    records: list[dict[str, Any]] = []
    for idx, (bt, rt_) in enumerate(zip(batched, reference)):
        match = bt == rt_
        record: dict[str, Any] = {
            "index": idx,
            "match": match,
            "batched_sha": token_sha(bt),
            "reference_sha": token_sha(rt_),
            "batched_len": len(bt),
            "reference_len": len(rt_),
        }
        if not match:
            first = next(
                (
                    i
                    for i in range(min(len(bt), len(rt_)))
                    if bt[i] != rt_[i]
                ),
                min(len(bt), len(rt_)),
            )
            lo = max(0, first - 2)
            record["first_divergence"] = first
            record["batched_window"] = bt[lo : first + 3]
            record["reference_window"] = rt_[lo : first + 3]
        records.append(record)
    return records


def streams_all_match(records: list[dict[str, Any]]) -> bool:
    return all(bool(r.get("match")) for r in records)


# --------------------------------------------------------------------------- #
# The driver (MLX; lazy imports keep module import cheap)
# --------------------------------------------------------------------------- #
def _argmax_ids(logits_2d: Any) -> list[int]:
    """Greedy argmax over a ``[B, V]`` logits tensor -> list of B python ints."""
    import mlx.core as mx

    ids = mx.argmax(logits_2d, axis=-1)
    mx.eval(ids)
    return [int(t) for t in ids.tolist()]


def _eval_bundle(bundle: Any) -> tuple[list[int], list[int], list[int], list[int]]:
    """THE single per-cycle critical-path sync of the pipelined loop.

    ``bundle`` is the in-graph decision tensor ``[4, B]`` stacking, per cycle,
    ``(x0, draft, x1, accept)`` — all computed on device with no host round-trip.
    One :func:`mx.eval` + one ``tolist`` reads the whole decision for the cycle;
    it is the ONLY blocking eval on the steady-state critical path (the accept
    mask must reach the host to drive the uniform full-B repair branch — that is
    why the budget is 1 sync/cycle, not 0; folding the branch on-device is
    Build-2).  Kept as a module-level seam so a test can monkeypatch it and count
    exactly one call per cycle.
    """
    import mlx.core as mx

    mx.eval(bundle)
    rows = bundle.tolist()  # [[x0...],[draft...],[x1...],[accept...]]
    return (
        [int(t) for t in rows[0]],
        [int(t) for t in rows[1]],
        [int(t) for t in rows[2]],
        [int(t) for t in rows[3]],
    )


def _run_ar_loop(
    rt: Any,
    *,
    cache: Any,
    batch: int,
    logits_last: Any,
    hidden_last: Any,
    max_new_tokens: int,
    done: list[bool],
    commit: Any,
    admit_fn: Any = None,
    work_remaining: Any = None,
    max_cycles: int | None = None,
    pin_idle_offsets: int | None = None,
) -> tuple[int, int, int, int]:
    """Plain batched AR decode: one ``[B,1]`` forward per cycle, 1 tok/stream.

    The ROW-PACKING aggregate lane.  Speculative K=1 verify spends 2 rows per
    stream for a per-row cadence of ``1/(2-a) <= 1`` token/row/cycle; plain AR
    packs one stream per row at exactly 1 token/row/cycle — so at the fixed
    16-row lane budget, 16 AR streams beat 8 spec streams on AGGREGATE for any
    accept < 1, while spec remains the per-request LATENCY SKU.  Same pipelined
    single-sync structure, ragged per-row KV, admission hooks, and idle-offset
    pin as the fold-in loop; no draft, no verify decision, no replay, no
    snapshot/restore.  Returns ``(cycles, forwards, 0, 0)``.
    """
    import mlx.core as mx

    from mtplx.attention_context import attention_phase
    from mtplx.ragged_kv_cache import RaggedBatchKVCache

    ragged = [e for e in cache if isinstance(e, RaggedBatchKVCache)]

    def _submit(ll: Any) -> dict[str, Any]:
        x_ids = mx.argmax(ll, axis=-1)  # [batch]
        if pin_idle_offsets is not None and ragged and any(done):
            idle_dev = mx.array(list(done))
            pin_off = mx.full((batch,), int(pin_idle_offsets), dtype=mx.int32)
            for rc in ragged:
                rc.offsets = mx.where(idle_dev, pin_off, rc.offsets).astype(mx.int32)
        for rc in ragged:
            rc.reserve(1)
        # No draft head runs on this lane, so the hidden state is dead weight
        # here — and a target-only runtime (Laguna) returns logits ONLY, so
        # asking for it is an unpack error rather than a wasted tensor.
        with attention_phase("decode_verify"):
            v_logits = rt.forward_ar(mx.expand_dims(x_ids, axis=1), cache=cache)
        return {
            "x": x_ids,
            "v_logits": v_logits,
            "next_ll": v_logits[:, -1, :],
            "next_hl": None,
        }

    def _read(sub: dict[str, Any]) -> list[int]:
        nonlocal forwards
        mx.eval(sub["x"])  # THE one blocking sync
        forwards += 1
        return [int(t) for t in sub["x"].tolist()]

    forwards = 0
    if max_cycles is None:
        max_cycles = max_new_tokens + 2
    _more = work_remaining if work_remaining is not None else (lambda: not all(done))
    flags_stub = [False] * batch  # admit_fn clears replay flags; AR has none
    pending: list[int] | None = None

    def _flush() -> None:
        nonlocal pending
        if pending is not None:
            for b in range(batch):
                commit(b, pending[b])
            pending = None

    sub = _submit(logits_last)
    mx.async_eval(sub["x"], sub["v_logits"])
    pending = _read(sub)
    logits_last, hidden_last = sub["next_ll"], sub["next_hl"]
    cycles = 1

    while _more() and cycles < max_cycles:
        if admit_fn is not None:
            logits_last, hidden_last = admit_fn(
                logits_last, hidden_last, flags_stub, _flush
            )
        sub = _submit(logits_last)
        mx.async_eval(sub["x"], sub["v_logits"])
        _flush()  # commit the previous cycle (one-cycle lag)
        pending = _read(sub)
        logits_last, hidden_last = sub["next_ll"], sub["next_hl"]
        cycles += 1

    _flush()
    return cycles, forwards, 0, 0


def _run_foldin_loop(
    rt: Any,
    *,
    cache: Any,
    batch: int,
    n_real: int,
    real_slots: Any,
    logits_last: Any,
    hidden_last: Any,
    use_mtp_draft: bool,
    max_new_tokens: int,
    done: list[bool],
    commit: Any,
    verify_shape_ok: Any,
    admit_fn: Any = None,
    work_remaining: Any = None,
    max_cycles: int | None = None,
    pin_idle_offsets: int | None = None,
) -> tuple[int, int, int, int]:
    """The FOLD-IN REPLAY decode loop (scheme doc §2.2; R3).

    Pipelined single-sync structure identical to the Build-1 loop (submit -> async
    kick -> drain the previous cycle one behind -> ONE bundle sync), but with NO
    repair forward: a missed row re-enters the next cycle one position back.

    Per cycle, ONE ``[B,2]`` forward.  Per-row mode from last cycle's accept mask
    (``replay_flags``, from the previous bundle sync -- dummy slots forced OFF):

    * SPEC row (accepted or fresh): input ``[x0', d']`` written at ``(L, L+1)``;
      ``x0' = argmax`` of its latest logits, ``d'`` = MTP draft on its latest
      hidden.  Commits 2 tokens on accept, 1 (``x0'``) on a miss.
    * REPLAY row (missed last cycle): input ``[x0_prev, x1]`` written at
      ``(L-1, L)`` -- ``write_start = offset-1`` overwrites the stale draft slot;
      its recurrent state is reverted per-row to the PRE-verify snapshot of the
      cycle it missed (a 1-cycle snapshot window).  Unconditional accept (both
      tokens known); commits ``x1``.

    All row-mode selection -- input tokens, ragged write offsets / new offsets, and
    the recurrent restore mask -- is DEVICE-SIDE ``mx.where`` on the accept mask;
    the only host read is the single ``[4,B]`` bundle
    ``(commit0, commit1, n_commit, next_replay)`` per cycle.  The ragged KV entries
    carry per-row offsets and roll a missed row back by OVERWRITING its draft slot
    on the replay write (no KV snapshot); only the recurrent state is snapshot-
    restored.  Returns ``(cycles, forwards, all_accept_cycles, replay_rows)``.
    """
    import mlx.core as mx

    from mtplx.attention_context import attention_phase
    from mtplx.cache_state import (
        restore_untrimmable_cache_masked,
        snapshot_untrimmable_cache,
        snapshot_untrimmable_cache_lazy,
    )
    from mtplx.ragged_kv_cache import RaggedBatchKVCache

    # FIX 2: the fold-in per-cycle recurrent snapshot uses the lazy zero-copy
    # view by default (COW-safe -- the GDN forward and the masked REPLAY rewind
    # both rebind cache slots, never mutate a snapshot buffer in place), which
    # drops the whole-batch GDN-matrix clone every cycle.  The env fallback
    # restores the eager clone.  Resolved once per decode so the per-cycle hot
    # path is a plain call.
    _snapshot_untrimmable = (
        snapshot_untrimmable_cache
        if foldin_clone_snapshot()
        else snapshot_untrimmable_cache_lazy
    )

    ragged = [e for e in cache if isinstance(e, RaggedBatchKVCache)]
    # Dummy slots (indices >= n_real) NEVER replay -- they stay inert SPEC rows.
    real_mask_host = [b < n_real for b in range(batch)]

    def _mtp_draft(hl: Any, x0_ids: Any) -> Any:
        if not use_mtp_draft:
            return x0_ids
        draft_logits = rt.draft_mtp(
            hl, mx.expand_dims(x0_ids, axis=1), mtp_cache=rt.make_mtp_cache()
        )
        return mx.argmax(draft_logits[:, -1, :], axis=-1)

    def _submit(
        ll: Any, hl: Any, replay_flags: list[bool], replay_a: Any, replay_b: Any,
        prev_snapshot: Any,
    ) -> dict[str, Any]:
        """Build one cycle's device graph (no host sync); returns lazy handles."""
        replay_dev = mx.array(replay_flags)  # [batch] bool
        # 1. SPEC candidate tokens (device).
        x0_spec = mx.argmax(ll, axis=-1)  # [batch]
        d_spec = _mtp_draft(hl, x0_spec)  # [batch]
        # 2. per-row input [batch,2]: REPLAY rows use the known [x0_prev, x1].
        a = mx.where(replay_dev, replay_a, x0_spec)
        b = mx.where(replay_dev, replay_b, d_spec)
        inp = mx.stack([a, b], axis=1)  # [batch,2]
        # 3. pre-forward ragged prep: write_start = offset-1 for REPLAY rows;
        #    reserve so the ragged mask key_len matches the post-write capacity.
        # 3a. FROZEN-capacity guard (refill lane): idle rows -- dummy slots and
        #     done-but-not-yet-readmitted real slots -- decode discarded garbage
        #     but their offsets would advance ~2/cycle without bound and overflow
        #     the frozen physical capacity (out-of-bounds scatter corrupts
        #     NEIGHBOURING rows' slabs).  Pin them to a constant in-bounds
        #     position; their content is discarded and per-row independence keeps
        #     live rows byte-stable.  ``done`` is at most one flush stale, so an
        #     unpinned overrun is bounded by ~2 positions -- always in bounds.
        if pin_idle_offsets is not None and ragged and any(done):
            idle_dev = mx.array(list(done))
            pin_off = mx.full((batch,), int(pin_idle_offsets), dtype=mx.int32)
            for rc in ragged:
                rc.offsets = mx.where(idle_dev, pin_off, rc.offsets).astype(mx.int32)
        for rc in ragged:
            rc.offsets = mx.where(replay_dev, rc.offsets - 1, rc.offsets).astype(mx.int32)
            rc.reserve(2)
        # 3b. recurrent REPLAY rewind: revert replay rows to the pre-verify snapshot
        #     of the cycle they missed (prev cycle's snapshot).
        # FIX 1: skip the whole-batch masked restore when NO real row replays --
        # ``replay_flags`` already comes from the single per-cycle bundle sync
        # (dummy slots forced OFF), so this adds no new sync.  An all-False mask
        # restore is mathematically ``mx.where(False, snap, cur) == cur``, i.e. a
        # byte-identical no-op, so gating it out cannot change the committed
        # sequence -- it only elides ~158 us/layer of pointless mx.where rebinds.
        if prev_snapshot is not None and any(replay_flags):
            restore_untrimmable_cache_masked(cache, prev_snapshot, replay_flags)
        # 4. snapshot the (rewound) recurrent state for THIS cycle's potential miss.
        snapshot = _snapshot_untrimmable(cache)
        # 5. the one [B,2] forward.
        with attention_phase("decode_verify"):
            v_logits, v_hidden = rt.forward_ar(inp, cache=cache, return_hidden=True)
        # 6. decision (device).  REPLAY rows accept unconditionally.
        x1_new = mx.argmax(v_logits[:, 0, :], axis=-1)  # [batch]
        accept = mx.where(replay_dev, mx.array(True), b == x1_new)  # [batch] bool
        spec_miss = mx.logical_and(mx.logical_not(replay_dev), mx.logical_not(accept))
        # offset fix: a SPEC miss commits only x0 -> drop the stale draft slot from
        # the logical length (REPLAY already advanced by exactly 1; SPEC accept by 2).
        for rc in ragged:
            rc.offsets = mx.where(spec_miss, rc.offsets - 1, rc.offsets).astype(mx.int32)
        # 7. commit + telemetry bundle [4,batch].
        commit0 = mx.where(replay_dev, b, a)  # REPLAY commits x1(=b); SPEC commits x0(=a)
        two = mx.logical_and(mx.logical_not(replay_dev), accept)  # SPEC accept -> 2 tokens
        n_commit = mx.where(two, mx.array(2, mx.int32), mx.array(1, mx.int32))
        next_replay = spec_miss.astype(mx.int32)
        bundle = mx.stack(
            [commit0.astype(mx.int32), x1_new.astype(mx.int32), n_commit, next_replay],
            axis=0,
        )  # [4,batch] = (commit0, commit1, n_commit, next_replay)
        return {
            "v_logits": v_logits,
            "v_hidden": v_hidden,
            "bundle": bundle,
            "snapshot": snapshot,
            "next_ll": v_logits[:, 1, :],
            "next_hl": v_hidden[:, 1:2, :],
            "replay_a": a,  # this cycle's x0 -> next-cycle REPLAY's x0_prev
            "replay_b": x1_new,  # this cycle's true x1 -> next-cycle REPLAY's x1
        }

    def _read(sub: dict[str, Any]) -> tuple[tuple[list, list, list], list[bool]]:
        nonlocal forwards, all_accept_cycles
        verify_shape_ok(sub["v_logits"], sub["v_hidden"])
        c0, c1, nc, nr = _eval_bundle(sub["bundle"])  # THE one blocking sync
        forwards += 1
        if not any(nr[b] for b in real_slots):
            all_accept_cycles += 1  # no NEW real miss this cycle
        next_flags = [bool(nr[b]) and real_mask_host[b] for b in range(batch)]
        return (c0, c1, nc), next_flags

    def _drain(pending: tuple[list, list, list]) -> None:
        c0, c1, nc = pending
        for b in range(batch):
            commit(b, c0[b])
            if nc[b] >= 2:
                commit(b, c1[b])

    forwards = 0
    all_accept_cycles = 0
    replay_rows = 0
    # Every ACTIVE row commits >=1 token/cycle (accept 2, miss 1, replay 1), so a
    # row reaches max_new_tokens in <= max_new_tokens cycles; +lag headroom.
    # A refill driver passes an override scaled by its total request count.
    if max_cycles is None:
        max_cycles = max_new_tokens + 4
    _more = work_remaining if work_remaining is not None else (lambda: not all(done))

    zeros = mx.zeros((batch,), dtype=mx.int32)
    replay_flags = [False] * batch
    replay_a, replay_b, prev_snapshot = zeros, zeros, None
    pending: tuple[list, list, list] | None = None

    def _flush() -> None:
        # Drain the held cycle's commits exactly once.  At the loop top there is
        # NO cycle in flight (the previous iteration's read synced it), so an
        # admission hook can flush here and see fully-current done flags before
        # it reassigns a slot -- stale pending commits can never leak into a
        # newly admitted request.
        nonlocal pending
        if pending is not None:
            _drain(pending)
            pending = None

    # Prologue: submit + read cycle 0 (all SPEC), then hold one pending commit.
    sub = _submit(logits_last, hidden_last, replay_flags, replay_a, replay_b, prev_snapshot)
    mx.async_eval(sub["bundle"], sub["v_logits"], sub["v_hidden"])
    pending, replay_flags = _read(sub)
    replay_rows += sum(1 for f in replay_flags if f)
    logits_last, hidden_last = sub["next_ll"], sub["next_hl"]
    replay_a, replay_b, prev_snapshot = sub["replay_a"], sub["replay_b"], sub["snapshot"]
    cycles = 1

    while _more() and cycles < max_cycles:
        if admit_fn is not None:
            logits_last, hidden_last = admit_fn(
                logits_last, hidden_last, replay_flags, _flush
            )
        sub = _submit(
            logits_last, hidden_last, replay_flags, replay_a, replay_b, prev_snapshot
        )
        mx.async_eval(sub["bundle"], sub["v_logits"], sub["v_hidden"])
        _flush()  # commit the previous cycle (one-cycle lag)
        pending, replay_flags = _read(sub)
        replay_rows += sum(1 for f in replay_flags if f)
        logits_last, hidden_last = sub["next_ll"], sub["next_hl"]
        replay_a, replay_b, prev_snapshot = (
            sub["replay_a"], sub["replay_b"], sub["snapshot"]
        )
        cycles += 1

    _flush()  # flush the final cycle's deferred commit
    return cycles, forwards, all_accept_cycles, replay_rows


def to_foldin_cache(cache: list[Any], batch_size: int) -> list[Any]:
    """Convert a stock (prefilled) cache list to the FOLD-IN lane in place (item 1).

    * Full-attention layers (trimmable KV) -> :class:`RaggedBatchKVCache`, seeded
      from the prefilled scalar KV via ``from_scalar_cache`` (uniform per-row
      offsets == the shared prefill length, host capacity bound seeded).  Their
      array ``offset`` fails every custom Metal fast-path closed
      (``_cache_offset_static_int`` -> ``None``), so only stock SDPA runs, and the
      ragged mask reaches attention through ``create_attention_mask`` ->
      ``make_mask`` with no model edit.
    * GDN/conv layers (recurrent, batch-major) -> ``OwnedRecurrentStateCache`` so
      the per-row masked restore (``restore_masked``) is available for the REPLAY
      rewind.  A non-array recurrent entry (e.g. a CPU test fake with list state)
      is left untouched -- the generic restore fallback drives it.

    Called AFTER prefill: prefill runs on the stock lane (uniform offset, plain
    causal mask), then this hands the decode loop a ragged cache whose buffers are
    the prefilled K/V.
    """
    from mtplx.cache_state import OwnedRecurrentStateCache, _is_trimmable
    from mtplx.ragged_kv_cache import RaggedBatchKVCache

    for idx, entry in enumerate(cache):
        if entry is None:
            continue
        if isinstance(entry, RaggedBatchKVCache):
            continue  # already on the fold-in lane (refill converts early)
        if _is_trimmable(entry):
            cache[idx] = RaggedBatchKVCache.from_scalar_cache(
                entry, batch_size=int(batch_size)
            )
            continue
        if isinstance(entry, OwnedRecurrentStateCache):
            continue
        # Convert an ARRAY-state recurrent cache to the owned class (so
        # restore_masked exists).  Leave list/None state (test fakes) alone.
        state = getattr(entry, "state", None)
        if (
            isinstance(state, list)
            and state
            and all(_is_array_leaf(leaf) for leaf in state)
        ):
            cache[idx] = OwnedRecurrentStateCache.from_cache(entry)
    return cache


def _is_array_leaf(leaf: Any) -> bool:
    import mlx.core as mx

    return leaf is None or isinstance(leaf, mx.array)


def _zero_untrimmable_rows(cache: list[Any], row_mask: list[bool]) -> None:
    """Masked fresh-start reset of recurrent rows (refill admission).

    Rows selected by ``row_mask`` get their recurrent state zeroed
    (``OwnedRecurrentStateCache.zero_rows`` — conv tail and GDN matrix state
    both zero-initialize, so the admission prefill over those rows reproduces a
    from-scratch prefill); a per-row Python container (the CPU test fake's
    histories) gets those rows emptied, the same fresh-start semantics.
    Trimmable / ragged KV entries are skipped — their reset is the per-row
    offset rewrite in the admission pass.
    """
    import mlx.core as mx

    from mtplx.cache_state import _is_trimmable

    for entry in cache:
        if entry is None or _is_trimmable(entry):
            continue
        zero_rows = getattr(entry, "zero_rows", None)
        if callable(zero_rows):
            zero_rows(row_mask)
            continue
        state = getattr(entry, "state", None)
        if isinstance(state, mx.array):
            mask = mx.array(row_mask).reshape(
                (len(row_mask),) + (1,) * (int(state.ndim) - 1)
            )
            entry.state = mx.where(mask, mx.zeros_like(state), state)
        elif isinstance(state, list) and all(isinstance(r, list) for r in state):
            entry.state = [[] if m else row for row, m in zip(state, row_mask)]


def generate_greedy_batched(
    rt: Any,
    prompts: list[list[int]],
    *,
    max_new_tokens: int,
    stop_token_ids: set[int] | None = None,
    use_mtp_draft: bool = True,
    collect_stats: bool = True,
    cohort_slots: int | None = None,
    pad_id: int = 0,
    serial: bool | None = None,
    reject_mode: str | None = None,
    refill_queue: list[list[int]] | None = None,
    decode_mode: str = "spec",
) -> BatchedDecodeResult:
    """Greedy multi-stream batched decode.

    ``prompts`` is a list of REAL token-id sequences that MUST share a length
    (use :func:`left_pad_prompts` first for a ragged batch).  Every real stream is
    decoded to ``max_new_tokens`` greedy tokens (or an earlier stop token),
    committing 2 greedy tokens per cycle via one ``[B,2]`` verify forward and, on
    any real-stream reject, one uniform ``[B,2]`` full-B repair forward.

    FIXED-SHAPE COHORT MODE (``cohort_slots``).  When set (e.g. ``8``), the prompt
    list is padded to exactly ``cohort_slots`` streams with DUMMY prompts
    (``[pad_id] * prompt_len``).  Dummy slots — like finished streams — keep
    occupying their row in every forward but commit nothing and never trigger the
    repair branch (they are masked out of the all-accept decision).  EVERY forward
    therefore has identical ``[cohort_slots, ·]`` shapes regardless of how many
    real streams exist, which is what makes the per-stream sha gate FIXED-SHAPE:
    stream ``b`` batched among other real prompts vs stream ``b`` alone in a cohort
    of the same slot count differ only in the OTHER rows' content, so a bitwise
    match rests solely on per-row forward independence (``do_sort`` pinned via
    ``MTPLX_A3B_MOE_FORCE_UNSORTED``).  Results report REAL streams only.

    LOOP (``serial``).  Default (``serial=False``, or the
    ``MTPLX_A3B_BATCHED_DECODE_SERIAL`` env fallback unset) runs the Build-1
    single-sync PIPELINED loop: the per-cycle decision (x0, draft, x1, accept) is
    computed on-device and read back with ONE :func:`mx.eval` (:func:`_eval_bundle`)
    — the only blocking sync on the steady-state critical path — while the previous
    cycle's commit/stop bookkeeping drains one cycle behind (a stopped stream
    over-runs a bounded ``<=2`` cycles of uncommitted garbage).  ``serial=True``
    runs the exact Phase-1 serial loop (4-5 blocking syncs/cycle) for A/B; both
    commit the IDENTICAL greedy sequence — the parallelization is a pure scheduling
    change.

    REJECT (``reject_mode``).  Default ``"repair"`` (or the
    ``MTPLX_A3B_BATCHED_DECODE_REJECT`` env fallback unset) is the Build-1 uniform
    full-B repair loop above, byte-identical when off.  ``"foldin"`` selects the
    FOLD-IN REPLAY loop on the ragged-KV lane (scheme §2.2): a missed row re-enters
    the next cycle one position back with ``[x0_prev, x1]`` — no separate repair
    forward — its recurrent state rewound per-row.  Same single-sync structure,
    same committed greedy sequence; only the per-cycle token cadence (1 on a miss +
    1 on its replay, vs 2 on accept) and the ``replay_rows`` telemetry differ.  The
    stock KV entries are converted (post-prefill) to :class:`RaggedBatchKVCache`.

    The committed sequence of real stream ``b`` is byte-identical across loops AND
    reject modes, and to running ``[prompts[b]]`` alone through this function in a
    cohort of the same slot count (the correctness contract); assert with
    :func:`diff_streams`.

    REFILL / CONTINUOUS BATCHING (``refill_queue``, fold-in + cohort mode only).
    When not ``None`` (an EMPTY list still selects refill mechanics — the
    reference arm uses that), the run serves ``prompts + refill_queue`` requests
    through the ``n_real`` slots: a finished slot is re-admitted with the next
    queued request at the following cycle boundary.  EVERY real request —
    initial cohort included — enters through one identical ADMISSION pass: a
    ``[cohort, prompt_len]`` ragged-mask prefill in which the admitted rows run
    their new prompt from per-row offset 0 over zero-reset recurrent state
    while every other row's state is masked-restored and its offsets are put
    back (its KV beyond the logical length is never attended).  The initial
    scalar prefill runs DUMMY rows only (buffer materialization — identical in
    every run), and KV capacity is FROZEN to one constant so every forward in
    candidate and reference runs has identical shapes (the §47 discipline
    extended to admission).  Results report one stream per REQUEST; admission
    passes are counted in ``meta['admission_passes']`` (not ``forwards``).

    DECODE MODE (``decode_mode``).  ``"spec"`` (default) is everything above.
    ``"ar"`` selects the plain batched AR loop (:func:`_run_ar_loop`): one
    ``[B,1]`` row per stream, no draft/verify — the ROW-PACKING aggregate lane
    (16 AR streams in the 16-row budget beat 8 spec streams on aggregate for
    any accept < 1; spec stays the latency SKU).  AR runs on the same ragged
    fold-in cache and supports the same cohort gate and ``refill_queue``.
    """
    import mlx.core as mx

    from mtplx.attention_context import attention_phase
    from mtplx.cache_state import (
        rollback_after_verify,
        snapshot_untrimmable_cache,
    )

    decode_mode = str(decode_mode).strip().lower()
    if decode_mode not in ("spec", "ar"):
        raise ValueError(f"decode_mode must be 'spec' or 'ar', got {decode_mode!r}")
    ar_mode = decode_mode == "ar"

    # The AR lane needs no draft head: `_run_ar_loop` runs one [B,1] forward per
    # cycle with no draft, no verify decision and no replay.  Requiring MTP here
    # locked out every target-only runtime (Laguna has no MTP head at all) from
    # a lane that never touches one.  The speculative lane still requires it.
    if not ar_mode and not rt.mtp_enabled:
        raise RuntimeError(
            "generate_greedy_batched(decode_mode='spec') requires an "
            "MTP-enabled runtime; use decode_mode='ar' for target-only runtimes"
        )
    if not prompts:
        raise ValueError("prompts must be non-empty")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    n_real = len(prompts)
    prompt_len = len(prompts[0])
    if prompt_len < 1:
        raise ValueError("each prompt needs at least one token")
    if any(len(p) != prompt_len for p in prompts):
        raise ValueError(
            "all prompts must share a length (shared-offset cache); "
            "left_pad_prompts() equalizes a ragged batch"
        )

    # --- fixed-shape cohort padding: append dummy slots so every forward is the
    # same shape regardless of the real-stream count.  Dummy content is a fixed
    # pad-token prompt of the shared length.
    slots: list[list[int]] = [[int(t) for t in p] for p in prompts]
    if cohort_slots is not None:
        cohort_slots = int(cohort_slots)
        if cohort_slots < n_real:
            raise ValueError(
                f"cohort_slots ({cohort_slots}) must be >= the real prompt count "
                f"({n_real})"
            )
        dummy = [int(pad_id)] * prompt_len
        slots.extend([list(dummy) for _ in range(cohort_slots - n_real)])
    batch = len(slots)  # total rows in every forward (real + dummy) = FIXED shape
    real_slots = range(n_real)  # only these commit / are reported / gate the repair

    if serial is None:
        serial = batched_decode_serial()
    if reject_mode is None:
        reject_mode = batched_decode_reject_mode()
    reject_mode = str(reject_mode).strip().lower()
    if reject_mode not in _REJECT_MODES:
        raise ValueError(
            f"reject_mode must be one of {sorted(_REJECT_MODES)}, got {reject_mode!r}"
        )
    stop = {int(t) for t in (stop_token_ids or set())}


    refill = refill_queue is not None
    if refill:
        if not ar_mode and reject_mode != "foldin":
            raise ValueError(
                "refill_queue requires reject_mode='foldin' or decode_mode='ar' "
                "(the ragged per-row offset lane is what admission resets)"
            )
        if cohort_slots is None:
            raise ValueError(
                "refill_queue requires fixed-shape cohort mode (cohort_slots)"
            )
        for q in refill_queue:
            if len(q) < 1:
                raise ValueError("each refill prompt needs at least one token")
        # Refill prompts may differ in length from the cohort and from each
        # other. Admission groups them by length and prefills each group at its
        # OWN length, so no pad token ever enters a joining row.
        #
        # This used to be an error whose message read "left_pad the whole
        # request set together", which is the one thing a caller must not do
        # here: `left_pad_prompts` folds its pads into the GDN recurrent state
        # with no mask, and fails silently. An error that refuses the safe input
        # and recommends the unsafe workaround is worse than no check.
    # One entry per REQUEST (non-refill: exactly the initial prompts).
    requests: list[list[int]] = [list(p) for p in slots[:n_real]]
    if refill:
        requests += [[int(t) for t in q] for q in refill_queue]

    started_all = time.perf_counter()
    cache = rt.make_cache()

    # --- batched prefill: one [batch, prompt_len] forward -> per-stream last logits.
    # Refill mode prefills DUMMY rows only (buffer/template materialization,
    # identical in every run); the real requests enter via the admission pass.
    prefill_rows = (
        [[int(pad_id)] * prompt_len for _ in range(batch)] if refill else slots
    )
    started = time.perf_counter()
    # Only the speculative lane consumes hidden states (the draft head reads
    # them).  The AR lane never does, and a target-only runtime such as Laguna
    # returns logits ONLY — asking it for hidden states is an unpack error at
    # the very first forward.
    with attention_phase("prefill"):
        if ar_mode:
            logits = rt.forward_ar(mx.array(prefill_rows), cache=cache)
            hidden = None
        else:
            logits, hidden = rt.forward_ar(
                mx.array(prefill_rows),
                cache=cache,
                return_hidden=True,
            )
    mx.eval(logits) if hidden is None else mx.eval(logits, hidden)
    if int(logits.shape[0]) != batch or (
        hidden is not None and int(hidden.shape[0]) != batch
    ):
        raise RuntimeError(
            f"prefill collapsed the batch dim: logits {tuple(logits.shape)} "
            f"hidden {None if hidden is None else tuple(hidden.shape)} "
            f"for B={batch}"
        )
    logits_last = logits[:, -1, :]  # [batch, V]
    hidden_last = None if hidden is None else hidden[:, -1:, :]  # [batch, 1, H]
    prefill_s = time.perf_counter() - started

    # Request-indexed results with slot indirection: slot ``b`` currently serves
    # request ``slot_request[b]`` (``None`` for a dummy slot).  Without refill
    # this is the identity map, byte-identical to the old per-slot bookkeeping.
    tokens: list[list[int]] = [[] for _ in requests]
    finish: list[str | None] = [None] * len(requests)
    done = [False] * batch
    slot_request: list[int | None] = [None] * batch
    for b in range(n_real):
        slot_request[b] = b
    # Dummy slots occupy their row but are inert: pre-marked done so they commit
    # nothing and drop out of the ``all(done)`` termination test.
    for b in range(n_real, batch):
        done[b] = True

    def _commit(b: int, tok: int) -> None:
        """Record one committed token for slot b's request, applying stop/length."""
        r = slot_request[b]
        if r is None or done[b]:
            return
        tokens[r].append(int(tok))
        if int(tok) in stop:
            done[b] = True
            finish[r] = "stop"
        elif len(tokens[r]) >= max_new_tokens:
            done[b] = True
            finish[r] = "length"

    def _commit_pair(x0: list[int], x1: list[int]) -> None:
        for b in range(batch):
            _commit(b, x0[b])
            _commit(b, x1[b])

    def _verify_shape_ok(v_logits: Any, v_hidden: Any) -> None:
        if (
            int(v_logits.shape[0]) != batch
            or int(v_hidden.shape[0]) != batch
            or int(v_hidden.shape[1]) != 2
        ):
            raise RuntimeError(
                f"verify collapsed shape: logits {tuple(v_logits.shape)} "
                f"hidden {tuple(v_hidden.shape)} for [B={batch}, rows=2]"
            )

    cycles = 0
    forwards = 0
    all_accept_cycles = 0
    repair_cycles = 0
    replay_rows = 0
    started_decode = time.perf_counter()

    admission_passes = 0
    if ar_mode or reject_mode == "foldin":
        # ================= FOLD-IN REPLAY LOOP (R3, scheme §2.2) ================
        # ONE [B,2] pass per cycle, no separate repair forward.  Per-row mode from
        # last cycle's accept mask; all row-mode selection (tokens, write_start,
        # new_offsets, recurrent restore) is DEVICE-SIDE mx.where, the single
        # [4,B]-bundle sync per cycle preserved.  The stock KV entries become
        # RaggedBatchKVCache (per-row offsets) seeded from the prefill.
        foldin_cache = to_foldin_cache(cache, batch)
        admit_fn = None
        work_remaining = None
        max_cycles_override = None
        if refill:
            # ---- REFILL / CONTINUOUS BATCHING (see the docstring) -------------
            from mtplx.cache_state import (
                restore_untrimmable_cache_masked,
                snapshot_untrimmable_cache_lazy,
            )
            from mtplx.ragged_kv_cache import RaggedBatchKVCache

            _snap = (
                snapshot_untrimmable_cache
                if foldin_clone_snapshot()
                else snapshot_untrimmable_cache_lazy
            )
            ragged_entries = [
                e for e in foldin_cache if isinstance(e, RaggedBatchKVCache)
            ]
            # Constant KV capacity = the TRUE per-slot bound: a slot admits at
            # offset 0, prefills to prompt_len, then decodes at most max_new more
            # (idle pin sits at prompt_len < cap).  prompt_len + max_new + a small
            # lag margin; a tight growth step keeps the frozen cap (= the SDPA key
            # width read EVERY cycle) near that bound, not a 256-step round-up.
            # The TRUE per-slot bound over every request a slot may serve:
            # a slot admits at offset 0, prefills to that request's length, then
            # decodes at most max_new more. With ragged refill the longest
            # request in the whole set sets the bound, not the initial cohort's
            # length -- sizing it off `prompt_len` alone under-grew the buffer
            # the moment a joiner was longer than the cohort that sealed.
            frozen_cap = (
                max(len(request) for request in requests) + max_new_tokens + 16
            )
            for rc in ragged_entries:
                rc.step = 32
                rc.freeze_capacity(frozen_cap)
            next_req = n_real

            def _admit_rows(
                assign: dict[int, int], ll: Any, hl: Any, flags: list[bool] | None
            ) -> tuple[Any, Any]:
                """Admit ``assign`` (slot -> request idx) via ONE cohort-shaped
                ragged prefill; every other row's state/offsets are put back.

                Every request in ``assign`` MUST share a length, and that is
                what makes the forward's last position each admitted row's own
                last position. With a shared length there is no padding in any
                joining row, so nothing is folded into its recurrent state that
                its caller did not send. Callers with mixed lengths split the
                assignment by length and call this once per group; see
                ``admit_fn``.

                The rows that are NOT joining are fed pad tokens because the
                rectangle has to be full, and that is safe for a different
                reason: their recurrent state is snapshotted before the forward
                and masked-restored after it, and their KV offsets are put back,
                so nothing this forward showed them survives.
                """
                nonlocal admission_passes
                widths = {len(requests[rid]) for rid in assign.values()}
                if len(widths) > 1:
                    raise RuntimeError(
                        "an admission group must share one prompt length; got "
                        f"{sorted(widths)}. Padding them to a common width would "
                        "fold pad tokens into the GDN recurrent state, which no "
                        "offset rewinds and which fails silently"
                    )
                width = widths.pop() if widths else prompt_len
                dummy_row = [int(pad_id)] * width
                admit_host = [b in assign for b in range(batch)]
                keep_host = [not m for m in admit_host]
                admit_dev = mx.array(admit_host)
                pre_state = _snap(foldin_cache)
                _zero_untrimmable_rows(foldin_cache, admit_host)
                saved_offsets = [rc.offsets for rc in ragged_entries]
                if ragged_entries:
                    zero_off = mx.zeros((batch,), dtype=mx.int32)
                    for rc in ragged_entries:
                        rc.offsets = mx.where(
                            admit_dev, zero_off, rc.offsets
                        ).astype(mx.int32)
                inp = [
                    requests[assign[b]] if b in assign else dummy_row
                    for b in range(batch)
                ]
                with attention_phase("prefill"):
                    # AR lane: no draft head consumes hidden states, and a
                    # target-only runtime (Laguna) returns logits ONLY — same
                    # conditioning as the initial-cohort prefill above.
                    if ar_mode:
                        p_logits = rt.forward_ar(mx.array(inp), cache=foldin_cache)
                        p_hidden = None
                    else:
                        p_logits, p_hidden = rt.forward_ar(
                            mx.array(inp), cache=foldin_cache, return_hidden=True
                        )
                restore_untrimmable_cache_masked(foldin_cache, pre_state, keep_host)
                if ragged_entries:
                    admitted_off = mx.full((batch,), width, dtype=mx.int32)
                    for rc, saved in zip(ragged_entries, saved_offsets):
                        rc.offsets = mx.where(
                            admit_dev, admitted_off, saved
                        ).astype(mx.int32)
                ll = mx.where(admit_dev[:, None], p_logits[:, -1, :], ll)
                if hl is not None and p_hidden is not None:
                    hl = mx.where(admit_dev[:, None, None], p_hidden[:, -1:, :], hl)
                else:
                    hl = None
                for b, rid in assign.items():
                    slot_request[b] = rid
                    done[b] = False
                    if flags is not None:
                        flags[b] = False
                admission_passes += 1
                return ll, hl

            # The INITIAL cohort enters through the same admission pass as every
            # queued request -- one prefill mechanism, one kernel schedule.
            logits_last, hidden_last = _admit_rows(
                {b: b for b in range(n_real)}, logits_last, hidden_last, None
            )

            def admit_fn(ll: Any, hl: Any, flags: list[bool], flush: Any):
                nonlocal next_req
                if next_req >= len(requests):
                    return ll, hl
                if not any(done[b] for b in range(n_real)):
                    return ll, hl
                flush()  # commits current before any slot is reassigned
                assign: dict[int, int] = {}
                for b in range(n_real):
                    if next_req >= len(requests):
                        break
                    if done[b]:
                        assign[b] = next_req
                        next_req += 1
                if not assign:
                    return ll, hl
                # One forward per DISTINCT joiner length. Joiners of the same
                # length share a forward and amortize the weight read between
                # them; differing lengths cost a forward each, which is the
                # price of every row's logits landing on its own last token
                # instead of on a pad.
                by_length: dict[int, dict[int, int]] = {}
                for slot, rid in assign.items():
                    by_length.setdefault(len(requests[rid]), {})[slot] = rid
                for length in sorted(by_length):
                    ll, hl = _admit_rows(by_length[length], ll, hl, flags)
                return ll, hl

            def work_remaining() -> bool:
                return next_req < len(requests) or not all(done)

            max_cycles_override = (max_new_tokens + 4) * max(1, len(requests))
        if ar_mode:
            cycles, forwards, all_accept_cycles, replay_rows = _run_ar_loop(
                rt,
                cache=foldin_cache,
                batch=batch,
                logits_last=logits_last,
                hidden_last=hidden_last,
                max_new_tokens=max_new_tokens,
                done=done,
                commit=_commit,
                admit_fn=admit_fn,
                work_remaining=work_remaining,
                max_cycles=max_cycles_override,
                pin_idle_offsets=(prompt_len if refill else None),
            )
        else:
            cycles, forwards, all_accept_cycles, replay_rows = _run_foldin_loop(
                rt,
                cache=foldin_cache,
                batch=batch,
                n_real=n_real,
                real_slots=real_slots,
                logits_last=logits_last,
                hidden_last=hidden_last,
                use_mtp_draft=use_mtp_draft,
                max_new_tokens=max_new_tokens,
                done=done,
                commit=_commit,
                verify_shape_ok=_verify_shape_ok,
                admit_fn=admit_fn,
                work_remaining=work_remaining,
                max_cycles=max_cycles_override,
                pin_idle_offsets=(prompt_len if refill else None),
            )
    elif serial:
        # ================= EXACT PHASE-1 SERIAL LOOP (A/B baseline) =============
        # 4-5 blocking syncs per cycle; behaviour identical to the original driver
        # except dummy slots are masked out of the all-accept decision.
        max_cycles = max_new_tokens + 1  # guards a runaway
        while not all(done) and cycles < max_cycles:
            x0 = _argmax_ids(logits_last)
            if use_mtp_draft:
                draft_logits = rt.draft_mtp(
                    hidden_last,
                    mx.array([[int(t)] for t in x0]),
                    mtp_cache=rt.make_mtp_cache(),
                )
                draft = _argmax_ids(draft_logits[:, -1, :])
            else:
                draft = list(x0)

            snapshot = snapshot_untrimmable_cache(cache)
            with attention_phase("decode_verify"):
                v_logits, v_hidden = rt.forward_ar(
                    mx.array([[x0[b], draft[b]] for b in range(batch)]),
                    cache=cache,
                    return_hidden=True,
                )
            mx.eval(v_logits, v_hidden)
            forwards += 1
            _verify_shape_ok(v_logits, v_hidden)

            x1 = _argmax_ids(v_logits[:, 0, :])
            all_accept = all(draft[b] == x1[b] for b in real_slots)

            if all_accept:
                logits_last = v_logits[:, 1, :]
                hidden_last = v_hidden[:, 1:2, :]
                all_accept_cycles += 1
            else:
                rollback_after_verify(cache, snapshot, verified_tokens=2)
                with attention_phase("decode_verify"):
                    r_logits, r_hidden = rt.forward_ar(
                        mx.array([[x0[b], x1[b]] for b in range(batch)]),
                        cache=cache,
                        return_hidden=True,
                    )
                mx.eval(r_logits, r_hidden)
                forwards += 1
                repair_cycles += 1
                logits_last = r_logits[:, 1, :]
                hidden_last = r_hidden[:, 1:2, :]

            _commit_pair(x0, x1)
            cycles += 1
    else:
        # ============ BUILD-1 SINGLE-SYNC PIPELINED LOOP (default) ==============
        # ONE blocking eval per cycle (the decision bundle); the previous cycle's
        # commit/stop bookkeeping drains one cycle behind the GPU submission.
        max_cycles = max_new_tokens + 3  # +lag headroom over the serial guard

        def _submit(ll: Any, hl: Any) -> tuple[Any, Any, Any, Any]:
            """Build one cycle's device graph (no host sync).

            Returns lazy ``(snapshot, v_logits, v_hidden, bundle)`` where
            ``bundle`` is ``[4, batch]`` = stack(x0, draft, x1, accept), all
            computed on-device.
            """
            snapshot = snapshot_untrimmable_cache(cache)
            x0_ids = mx.argmax(ll, axis=-1)  # [batch]
            if use_mtp_draft:
                draft_logits = rt.draft_mtp(
                    hl,
                    mx.expand_dims(x0_ids, axis=1),
                    mtp_cache=rt.make_mtp_cache(),
                )
                draft_ids = mx.argmax(draft_logits[:, -1, :], axis=-1)
            else:
                draft_ids = x0_ids
            with attention_phase("decode_verify"):
                v_logits, v_hidden = rt.forward_ar(
                    mx.stack([x0_ids, draft_ids], axis=1),
                    cache=cache,
                    return_hidden=True,
                )
            x1_ids = mx.argmax(v_logits[:, 0, :], axis=-1)  # [batch]
            accept = (draft_ids == x1_ids).astype(mx.int32)  # [batch]
            bundle = mx.stack([x0_ids, draft_ids, x1_ids, accept], axis=0)  # [4,batch]
            return snapshot, v_logits, v_hidden, bundle

        def _read_repair(
            snapshot: Any, v_logits: Any, v_hidden: Any, bundle: Any
        ) -> tuple[list[int], list[int], Any, Any]:
            """The single per-cycle sync + the host-side accept/repair branch.

            Returns ``(x0, x1, next_logits_last, next_hidden_last)``.
            """
            nonlocal forwards, all_accept_cycles, repair_cycles
            _verify_shape_ok(v_logits, v_hidden)
            x0, draft, x1, accept = _eval_bundle(bundle)  # THE one blocking sync
            forwards += 1
            all_accept = all(accept[b] for b in real_slots)  # dummies masked out
            if all_accept:
                next_ll = v_logits[:, 1, :]
                next_hl = v_hidden[:, 1:2, :]
                all_accept_cycles += 1
            else:
                rollback_after_verify(cache, snapshot, verified_tokens=2)
                with attention_phase("decode_verify"):
                    r_logits, r_hidden = rt.forward_ar(
                        mx.array([[x0[b], x1[b]] for b in range(batch)]),
                        cache=cache,
                        return_hidden=True,
                    )
                forwards += 1
                repair_cycles += 1
                next_ll = r_logits[:, 1, :]
                next_hl = r_hidden[:, 1:2, :]
            return x0, x1, next_ll, next_hl

        # Prologue: submit + read cycle 0 so the loop always holds one committed-
        # but-not-yet-drained cycle in ``pending``.
        snapshot, v_logits, v_hidden, bundle = _submit(logits_last, hidden_last)
        mx.async_eval(bundle, v_logits, v_hidden)
        x0, x1, logits_last, hidden_last = _read_repair(
            snapshot, v_logits, v_hidden, bundle
        )
        pending: tuple[list[int], list[int]] = (x0, x1)
        cycles = 1

        while not all(done) and cycles < max_cycles:
            # 1. Submit the NEXT cycle's forward + decision bundle (kick the GPU).
            snapshot, v_logits, v_hidden, bundle = _submit(logits_last, hidden_last)
            mx.async_eval(bundle, v_logits, v_hidden)
            # 2. Drain the PREVIOUS cycle's commit while the GPU runs this cycle
            #    (one-cycle-lag; done-flags trail by a bounded <=2 cycles).
            _commit_pair(*pending)
            # 3. The single blocking sync for this cycle + accept/repair branch.
            x0, x1, logits_last, hidden_last = _read_repair(
                snapshot, v_logits, v_hidden, bundle
            )
            pending = (x0, x1)
            cycles += 1

        # Flush the final cycle's deferred commit.
        _commit_pair(*pending)

    decode_s = time.perf_counter() - started_decode

    for r in range(len(requests)):
        if finish[r] is None:
            finish[r] = "length" if len(tokens[r]) >= max_new_tokens else "cycle_cap"

    streams = [
        BatchedStreamResult(
            index=r,
            # The REQUEST's own length. Reporting the cohort's shared length
            # was harmless while every request had to share it; with a ragged
            # refill queue it hands a caller another request's prompt length.
            prompt_len=len(requests[r]),
            tokens=tokens[r],
            finish_reason=str(finish[r]),
            sha=token_sha(tokens[r]),
        )
        for r in range(len(requests))
    ]
    generated = sum(len(tokens[r]) for r in range(len(requests)))
    foldin = reject_mode == "foldin"
    meta: dict[str, Any] = {}
    if collect_stats:
        if ar_mode:
            loop_name = "ar_batched_single_sync"
            scheme_name = "ar_row_packed"
            lane_name = "ragged_batch_kv+stock_attention"
        elif foldin:
            loop_name = "foldin_replay_single_sync"
            scheme_name = "foldin_replay_ragged_kv"
            lane_name = "ragged_batch_kv+stock_attention"
        else:
            loop_name = "serial" if serial else "pipelined_single_sync"
            scheme_name = "uniform_+2_per_cycle_full_B_repair"
            lane_name = "batch_generic_kv+stock_attention"
        meta = {
            "elapsed_s": time.perf_counter() - started_all,
            "use_mtp_draft": bool(use_mtp_draft),
            "shared_offset_lane": lane_name,
            "scheme": scheme_name,
            "reject_mode": reject_mode,
            "loop": loop_name,
            "cohort_slots": None if cohort_slots is None else int(cohort_slots),
            "real_streams": n_real,
            # fold-in and the pipelined repair loop both hold to one bundle sync
            # per cycle; the serial A/B baseline is multi-sync.
            "syncs_per_cycle": None if (serial and not foldin and not ar_mode) else 1,
            "replay_rows": int(replay_rows),
            "decode_mode": decode_mode,
            "refill": bool(refill),
            "requests": len(requests),
            "admission_passes": int(admission_passes),
            "phase": 1,
            "phase2_remaining": PHASE2_REMAINING,
        }
    return BatchedDecodeResult(
        batch_size=n_real,
        streams=streams,
        cycles=cycles,
        forwards=forwards,
        all_accept_cycles=all_accept_cycles,
        repair_cycles=repair_cycles,
        prefill_s=prefill_s,
        decode_s=decode_s,
        generated_tokens=generated,
        replay_rows=replay_rows,
        meta=meta,
    )
