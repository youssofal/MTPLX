"""Dense batched-MTP cohort service for ``mtplx serve``.

The dense counterpart of ``mtplx/server/mtp_batch.py``. Request threads enqueue
independent jobs; the model-owner thread seals an immutable cohort of
compatible jobs and runs it through ``generate_dense_mtp_batch``, closing each
caller's future with that caller's own tokens.

WHAT THIS DOES AND DOES NOT DO (item 1 of the serving-integration plan)
-----------------------------------------------------------------------
Does: concurrent HTTP requests arriving at a dense qwen3_5 server are batched
into one cohort and decoded as ONE ``[B, K+1]`` verify forward per cycle, so
the weight read is amortized across the cohort instead of being repeated per
stream. Tokens reach each caller as they are committed, not only at the end.

Does not: admit a request into a cohort that is already running. A sealed
cohort keeps its membership for its lifetime, exactly like the A3B lane's
Phase 1. Continuous batching is item 4 and is a separate decision.

WHAT DOES *NOT* CONSTRAIN ADMISSION
-----------------------------------
Item 1 shipped this lane with two deliberate temporary constraints, uniform
sampling and uniform prompt length. **Items 2 and 3 removed both**, so a cohort
now freely mixes:

* **Prompt lengths.** Rows are handed to the driver at their true lengths with
  ``ragged_prompts=True``, which groups them by length, prefills each group at
  its own length, and assembles one cache with per-row pinned offsets. Nothing
  is padded. This is not merely cheaper than the ``left_pad_prompts`` it
  replaced: padding was WRONG on this trunk, because pad tokens run through the
  recurrent GDN layers and no KV offset undoes that. Each row's output is now
  bit-identical to that row decoded alone, which is the property a serving lane
  actually needs — a caller's tokens must not depend on who else was in flight.
* **Sampling settings.** Temperature, top-k and top-p go to the driver as
  per-row vectors, so a greedy caller and a caller at temperature 0.8 share one
  cohort. Greedy is exact inside a sampling cohort, not approximated.
* **``max_tokens``.** Per-row caps, so a 64-token request and a 2000-token one
  share a cohort and each stops at its own limit.

What still binds a cohort together is the stop-token set and the route, both of
which the driver takes once for the whole batch.

ONE KNOWN LIMITATION, stated rather than hidden
------------------------------------------------
A per-request ``seed`` is not honoured inside a cohort. The driver takes one
``sampling_seed`` per run and derives per-row randomness from it, so rows are
independent of each other but a caller cannot reproduce its own stream by
pinning a seed while sharing a cohort. Solo requests (cohorts of one) are
unaffected, and the effective seed is reported per request as
``dense_mtp_batch_cohort_seed`` so this is visible rather than silent. Per-row
seeds need per-row keys in the driver's sampling core; that is not part of
items 1 to 3.
"""

from __future__ import annotations

import contextlib
import math
import os
import time
from collections import Counter, deque
from collections.abc import Callable, Hashable
from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import Condition, Event
from typing import Any

from mtplx.dense_mtp_batch import DenseBatchResult, generate_dense_mtp_batch
from mtplx.sampling import SamplerConfig
from mtplx.server.mtp_batch import MTPBatchFinalizeOwnership

__all__ = [
    "batch_generic_cache_scope",
    "DenseMTPBatchQueueFull",
    "DenseMTPBatchJob",
    "DenseMTPBatchGenerationService",
    "dense_mtp_batch_compatibility_key",
]

# Optional bound on the LENGTH SPREAD inside one cohort. Both zero means
# unlimited, which is the default after item 2: with per-row pinned offsets a
# short prompt is not padded and costs nothing extra to sit beside a long one.
# The knob survives because a cohort still finishes together, so an operator who
# cares more about short-request latency than aggregate throughput can stop a
# 24k prompt from pulling a 200-token one along for its prefill. That is a
# fairness policy, not a correctness requirement, so it is off unless asked for.
_DEFAULT_LENGTH_SPREAD_TOKENS = 0
_DEFAULT_LENGTH_SPREAD_RATIO = 0.0


# Cache layouts that exist for single-stream serving and cannot express a
# batch. The served default is the vLLM-Metal paged attention cache, which
# raises "VllmMetalPagedKVCache currently supports batch size 1" the moment a
# cohort of 2 reaches it. The owned/block-owned tail caches are the same story.
#
# `dense_batch_bench` never met any of this because it builds its own runtime
# with none of them installed, which is exactly why a library-level bench cannot
# stand in for a serving test.
_BATCH_INCAPABLE_CACHE_ENV = (
    "MTPLX_VLLM_METAL_PAGED_ATTN",
    "MTPLX_OWNED_ATTN_KV",
    "MTPLX_BLOCK_OWNED_ATTN_KV",
)


@contextlib.contextmanager
def batch_generic_cache_scope():
    """Run a cohort on the batch-generic cache lane, then restore the server's.

    Scoped rather than set once at startup, so a SOLO request on the same
    server keeps its paged KV cache and its performance. The driver builds both
    its trunk and head caches inside this scope, so both land on the stock
    lane.

    Safe against races by construction: cohorts execute on the model-owner
    thread, which is the only thread that builds caches, so no other request
    can observe the temporarily-modified environment. This mirrors
    ``_target_prefill_cache_layout_scope`` in ``mtplx/generation.py``, which
    does the same save/set/restore for the same reason.
    """

    saved = {key: os.environ.get(key) for key in _BATCH_INCAPABLE_CACHE_ENV}
    for key in _BATCH_INCAPABLE_CACHE_ENV:
        os.environ[key] = "0"
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class DenseMTPBatchQueueFull(RuntimeError):
    """The lane's admission queue is at capacity; the caller should retry.

    Deliberately a distinct type rather than a generic RuntimeError: the server
    maps it to 503 with Retry-After, which tells a client to back off, whereas
    an unbounded queue silently accepts work it may never reach.
    """


def _sampler_penalties(sampler: Any) -> tuple[float, float]:
    """(presence, frequency) for one request.

    Separate from _sampler_triple rather than widening it: every existing
    caller of the triple wants exactly the three filter parameters, and
    silently changing that tuple's arity is how a positional index elsewhere
    starts reading the wrong field.
    """

    return (
        float(getattr(sampler, "presence_penalty", 0.0) or 0.0),
        float(getattr(sampler, "frequency_penalty", 0.0) or 0.0),
    )


def _sampler_triple(sampler: Any) -> tuple[float, float, int]:
    return (
        float(getattr(sampler, "temperature", 0.0) or 0.0),
        float(getattr(sampler, "top_p", 1.0) or 1.0),
        int(getattr(sampler, "top_k", 0) or 0),
    )


def dense_mtp_batch_compatibility_key(
    lane: Any,
    sampler: Any,
    stop_token_ids: set[int] | frozenset[int],
) -> tuple[Any, ...]:
    """Cohort admission key for the dense lane.

    Only two things bind a cohort now: the route, and the stop-token set, which
    the driver takes once for the whole batch.

    The sampler is deliberately NOT in the key any more. Item 1 had it there
    because the driver applied one temperature/top-k/top-p to the whole cohort,
    so two requests that disagreed could not safely share one; item 3 made those
    per row, so they can. ``sampler`` stays in the signature because callers
    pass it and because putting it back is a one-line change if a future driver
    revision re-introduces a cohort-wide sampling stage.

    Prompt length is not in the key either, and after item 2 there is no reason
    for it to be: rows are prefilled at their true lengths and nothing is
    padded, so mixing a 200-token prompt with a 24k one costs the short row
    nothing beyond waiting for the cohort.
    """

    return (
        str(getattr(lane, "route_id", "")),
        tuple(sorted(int(t) for t in stop_token_ids)),
    )


@dataclass
class DenseMTPBatchJob:
    """One caller's request, waiting for or riding in a dense cohort."""

    request_id: str
    prompt_ids: list[int]
    max_tokens: int
    sampler: SamplerConfig
    seed: int
    stop_token_ids: set[int]
    compatibility_key: Hashable
    generation_limits: dict[str, Any]
    solo_runner: Callable[["DenseMTPBatchJob"], dict[str, Any]] | None
    cancel_error: Callable[["DenseMTPBatchJob"], BaseException]
    token_callback: Callable[[list[int]], None] | None = None
    prefill_callback: Callable[[dict[str, Any]], None] | None = None
    cancel_event: Event = field(default_factory=Event)
    finalize_ownership: MTPBatchFinalizeOwnership = field(
        default_factory=MTPBatchFinalizeOwnership
    )
    request_observability: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    future: Future = field(default_factory=Future, init=False)
    #: Which row of the cohort this request occupied. Set when the driver's
    #: streams are zipped back onto jobs, and used to attribute per-row prefix
    #: reuse to the right caller. -1 means "not yet placed in a cohort".
    cohort_row: int = field(default=-1, init=False)
    tokens: list[int] = field(default_factory=list, init=False)
    token_times: list[float] = field(default_factory=list, init=False)
    callback_error: BaseException | None = field(default=None, init=False)
    decode_started_s: float | None = field(default=None, init=False)
    created_s: float = field(default_factory=time.perf_counter, init=False)
    admitted_s: float | None = field(default=None, init=False)
    pad_tokens: int = field(default=0, init=False)
    finalize_owner_accepted: bool = field(default=False, init=False)
    finalize_owner_finished: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.prompt_ids = [int(token) for token in self.prompt_ids]
        self.max_tokens = max(1, int(self.max_tokens))
        self.stop_token_ids = {int(token) for token in self.stop_token_ids}
        self.request_observability = dict(self.request_observability)
        self.generation_limits = dict(self.generation_limits)

    def cancel_requested(self) -> bool:
        return self.cancel_event.is_set()

    def emit_token(self, token: int) -> None:
        """Record one committed token and forward it to the caller.

        Same contract as the A3B lane's ``emit_token``: stop tokens are kept in
        the job's own token list (the finish reason is derived from them) but
        are not forwarded to the client callback, and a callback that raises
        cancels the job rather than failing its cohort-mates.
        """

        if self.cancel_requested():
            return
        value = int(token)
        self.tokens.append(value)
        if value not in self.stop_token_ids and self.token_callback is not None:
            try:
                self.token_callback([value])
            except Exception as exc:
                self.callback_error = exc
                self.cancel_event.set()
                if not self.future.done():
                    self.future.set_exception(exc)
        self.token_times.append(time.perf_counter())

    def emit_prefill(self, payload: dict[str, Any]) -> None:
        if self.prefill_callback is None:
            return
        try:
            self.prefill_callback(dict(payload))
        except Exception:
            pass

    def mark_decode_started(self) -> None:
        if self.decode_started_s is None:
            self.decode_started_s = time.perf_counter()

    def close_cancelled(self) -> None:
        self.cancel_event.set()
        if not self.future.done():
            self.future.set_exception(self.cancel_error(self))

    def finish_finalize_ownership(self, *, finalized: bool) -> None:
        if not self.finalize_owner_accepted or self.finalize_owner_finished:
            return
        self.finalize_ownership.owner_finished(finalized=finalized)
        self.finalize_owner_finished = True


class DenseMTPBatchGenerationService:
    """Seal and execute dense batched-MTP cohorts on the model-owner thread."""

    def __init__(
        self,
        state: Any,
        *,
        lane: Any,
        driver: Callable[..., DenseBatchResult] = generate_dense_mtp_batch,
        batch_wait_s: float = 0.02,
        auto_schedule: bool = True,
        owner_finalize: Callable[[list[DenseMTPBatchJob]], dict[str, Any] | None]
        | None = None,
        length_spread_tokens: int = _DEFAULT_LENGTH_SPREAD_TOKENS,
        length_spread_ratio: float = _DEFAULT_LENGTH_SPREAD_RATIO,
        refill_depth: int | None = None,
        max_queue_depth: int | None = None,
        cohort_deadline_s: float | None = None,
        memory_headroom: float = 0.85,
        prefix_cache_bytes: int = 0,
        prefix_cache_min_tokens: int = 256,
        continuous: bool = True,
        max_requests_per_cohort: int | None = None,
    ) -> None:
        self.state = state
        self.lane = lane
        self.driver = driver
        self.batch_wait_s = max(0.0, float(batch_wait_s))
        self.auto_schedule = bool(auto_schedule)
        self.owner_finalize = owner_finalize
        self.cohort_slots = int(getattr(lane.geometry, "cohort_slots", 8) or 8)
        self.length_spread_tokens = max(0, int(length_spread_tokens))
        self.length_spread_ratio = max(0.0, float(length_spread_ratio))
        # How many queued requests may ride along with one cohort as refill.
        # Bounded on purpose: a job accepted into the refill queue leaves
        # `_pending`, so it can no longer be picked up by a DIFFERENT cohort. An
        # unbounded queue would let one long-running cohort claim every waiting
        # request and hold them behind its own slowest row. One refill per slot
        # is the conservative default; raise it when slots turn over fast
        # relative to how often cohorts seal.
        self.refill_depth = (
            int(refill_depth) if refill_depth is not None else self.cohort_slots
        )
        # Bounded admission queue. Generous by default -- eight cohorts' worth
        # of waiting work -- but finite, because the alternative is accepting
        # requests until memory runs out and then failing in a way nobody can
        # attribute. A caller told "busy, retry" can back off; a caller queued
        # behind an hour of work cannot.
        self.max_queue_depth = (
            int(max_queue_depth)
            if max_queue_depth is not None
            else self.cohort_slots * 8
        )
        # Fraction of the device's recommended working set one cohort may
        # reserve. Passed to the driver to bound GROWTH, and used here to bound
        # what a cohort may SEAL with -- which growth cannot protect, because a
        # cohort that is already too wide never grows.
        self.memory_headroom = max(0.0, float(memory_headroom))
        # T-210 prefix reuse. A conversation's second turn resends everything
        # the first turn contained, and this lane otherwise re-reads all of it.
        # Prefix reuse, through the SAME SessionBank the solo path uses, so a
        # server running both lanes warms one cache rather than two. NOTE the
        # budget: when the server's bank is shared, `prefix_cache_bytes` acts as
        # an ON/OFF switch and the SERVER's configured limit governs size --
        # this lane must not resize a bank the solo path is also using.
        #
        # OFF BY DEFAULT, and deliberately: it holds gigabytes, and the soak
        # that would catch a leak in it has not run. An operator turns it on;
        # nobody gets it by surprise.
        # PREFER THE SERVER'S EXISTING BANK. Constructing a private one here is
        # what this code did until 2026-08-24, while the whole justification for
        # moving to SessionBank was "one cache instead of two, entries shared
        # with the solo path rather than duplicated". That claim was false as
        # wired: the solo path uses `state.sessions.bank`, this lane built its
        # own, and the result was two banks, two budgets, twice the memory, and
        # no cross-lane reuse -- the exact duplication the migration was for.
        #
        # `state` was in scope the entire time.
        self.prefix_bank = None
        if int(prefix_cache_bytes) > 0:
            shared = getattr(getattr(state, "sessions", None), "bank", None)
            if shared is not None:
                self.prefix_bank = shared
            else:
                # No server bank to share -- a bare lane, or a test harness.
                # Build one so the feature still works standalone.
                from mtplx.session_bank import SessionBank

                self.prefix_bank = SessionBank(max_bytes=int(prefix_cache_bytes))
        # Continuous batching (item 4). With this on, the driver holds a LIVE
        # handle on the pending queue and pulls from it at every cycle
        # boundary, so a request that arrives one millisecond after a cohort
        # sealed joins that cohort instead of waiting for it to drain.
        self.continuous = bool(continuous)
        # How many requests one cohort will serve before it winds down. Not a
        # correctness bound -- the driver would happily run forever under load
        # -- but a bound on how much per-request bookkeeping one cohort
        # accumulates, and a natural point at which the length-compatibility
        # window is recomputed from current traffic rather than from whoever
        # happened to seal it.
        self.max_requests_per_cohort = (
            int(max_requests_per_cohort)
            if max_requests_per_cohort is not None
            else self.cohort_slots * 8
        )
        self._rejected_total = 0
        # Wall-clock bound on a cohort. Off by default: a lane that starts
        # cutting requests short because an operator never chose a timeout
        # would be worse than one that runs long visibly. The stall indicator
        # in snapshot() is what makes a long run visible; this is what bounds
        # it once an operator decides what "too long" means for their traffic.
        self.cohort_deadline_s = (
            float(cohort_deadline_s)
            if cohort_deadline_s is not None and float(cohort_deadline_s) > 0
            else None
        )
        # Live cohort state. None between cohorts.
        self._live: dict[str, Any] | None = None
        # Bounded recent windows. A distribution shows drift that a running
        # mean hides, and a bound keeps a long-lived server from accumulating
        # history forever.
        self._recent_cohort_s: deque[float] = deque(maxlen=64)
        self._recent_queue_wait_s: deque[float] = deque(maxlen=256)
        # A cohort committing nothing for longer than this is reported
        # UNHEALTHY. Generous: a long prompt's prefill legitimately produces no
        # tokens for a while, and a false stall alarm at 3am is worse than none.
        self.stall_threshold_s = 120.0
        # A cohort that has committed NO tokens yet is prefilling, not stalled.
        # Prefill of eight rows at several thousand prompt tokens each can run
        # well past the 120s progress threshold while being perfectly healthy,
        # so the not-yet-started case gets its own, much larger budget. Observed
        # in a soak smoke run: cohorts sat at tokens_committed=0 with the age
        # climbing, and the ordering below would have called them STALLED.
        self.start_threshold_s = 600.0
        self._last_cohort_seed = 0
        # Last cohort's KV reservation, surfaced so an operator watching a live
        # server sees it. It is in the driver's run meta too, but a number only
        # a benchmark receipt carries is a number nobody checks at 3am.
        self._last_kv: dict[str, Any] = {}
        self._condition = Condition()
        self._pending: list[DenseMTPBatchJob] = []
        self._active: list[DenseMTPBatchJob] = []
        self._refill: list[DenseMTPBatchJob] = []
        self._pump_scheduled = False
        self._shutdown = False
        self._last_error: str | None = None
        self._last_real_width = 0
        self._batch_histogram: Counter[int] = Counter()
        self._cycles = 0
        self._accepted_drafts = 0
        self._drafted = 0
        self._solo_runs = 0
        self._pad_tokens_total = 0
        # Continuous-batching receipts. `refill_admitted_total` is the one that
        # matters: it counts requests that took a slot another request had
        # already finished with, which is the single fact distinguishing item 4
        # working from item 4 merely existing.
        self._cohort_failures = 0
        self._refill_admitted_total = 0
        self._refill_requeued_total = 0
        self._cohort_resizes = 0
        self._rows_peak = 0
        self._rows_blocked_by_memory = 0
        self._unservable_total = 0
        self._unservable_this_seal: list[DenseMTPBatchJob] = []
        self._last_error_context: dict[str, Any] = {}
        self._requests_served_total = 0
        # T-210 reuse, aggregated across cohorts. The BANK reports what it
        # holds; only the server can say whether holding it is paying off,
        # because that is a ratio against requests it served.
        self._prefix_restores_total = 0
        self._prefix_tokens_skipped_total = 0
        self._prefix_restore_failures_total = 0
        self._max_requests_in_one_cohort = 0
        self._last_owner_finalize: dict[str, Any] = {}

    # ---- introspection ---------------------------------------------------

    @staticmethod
    def _percentiles(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        ordered = sorted(values)

        def pick(fraction: float) -> float:
            index = min(len(ordered) - 1, int(fraction * len(ordered)))
            return round(ordered[index], 4)

        return {
            "n": len(ordered),
            "p50": pick(0.50),
            "p90": pick(0.90),
            "p99": pick(0.99),
            "max": round(ordered[-1], 4),
        }

    def _live_view(self) -> dict[str, Any]:
        """The 3am answer: is a cohort stalled, or merely slow?

        Duration alone cannot tell them apart -- a five-minute cohort is fine if
        its requests are long. Whether TOKENS ARE STILL ARRIVING can, so
        seconds-since-progress is the field that matters and the verdict is
        derived here rather than left to a reader doing arithmetic at 3am.
        """

        live = self._live
        if live is None:
            return {"state": "idle"}
        now = time.perf_counter()
        since_progress = now - float(live["last_progress_s"])
        age = now - float(live["started_s"])
        committed = int(live["tokens_committed"])
        return {
            "state": "running",
            "age_s": round(age, 3),
            "seconds_since_progress": round(since_progress, 3),
            "tokens_committed": committed,
            "tokens_per_s": round(committed / age, 2) if age > 0 else 0.0,
            "width": live["width"],
            "requests": live["requests"],
            "request_ids": list(live["request_ids"]),
            # ORDER MATTERS. Checking STALLED first made the "starting" branch
            # unreachable once age passed the threshold, so a long but healthy
            # prefill reported STALLED -- and the soak's STOP condition reads
            # this field, so it would have aborted a good run and been recorded
            # as a defect in the lane.
            #
            # Two distinct questions: a row that HAS produced tokens and then
            # stopped is stalled at stall_threshold_s; a cohort that has not
            # produced any yet is starting, until start_threshold_s makes even
            # prefill implausible.
            "health": (
                "starting"
                if committed == 0 and age <= self.start_threshold_s
                else "STALLED"
                if (committed == 0 and age > self.start_threshold_s)
                or (committed > 0 and since_progress > self.stall_threshold_s)
                else "healthy"
            ),
            "start_threshold_s": self.start_threshold_s,
            "stall_threshold_s": self.stall_threshold_s,
            # WHY this cohort is not wider, when it is not -- and `None` when
            # nothing is holding it back. Beside a non-empty queue and a width
            # below capacity, `None` is the zombie-slot defect and every other
            # value is a normal reason:
            #
            #   at_width_cap        every row is busy, or the ceiling is reached
            #   cohort_request_cap  WINDING DOWN: served its request budget
            #   no_compatible_work  nothing waiting can share this cohort
            #   queue_empty         nothing waiting at all
            #
            # `cohort_request_cap` is the one that made this necessary. A
            # winding-down cohort drains its rows one at a time while work
            # queues behind it, which is exactly the zombie-slot signature and
            # is entirely normal -- and at the soak's prompt mix it lasts tens
            # of seconds, comfortably longer than the two consecutive samples
            # the abort needs.
            "growth_blocked": live.get("growth_blocked"),
            # The driver's own view of the cohort: current width, how many
            # pulled requests are still waiting for a row, and how many rows the
            # memory budget refused. Refreshed at least once a second while
            # decoding, and `driver_age_s` says how old it is -- because a
            # monitor that cannot tell stale from current will eventually act on
            # stale, which is exactly what nearly aborted an overnight run.
            "driver": dict(live.get("driver") or {}),
            "driver_age_s": (
                round(time.time() - float((live.get("driver") or {}).get("at")), 3)
                if (live.get("driver") or {}).get("at")
                else None
            ),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            accept_rate = (
                self._accepted_drafts / self._drafted if self._drafted else 0.0
            )
            return {
                # Total backlog, not just the unassigned part. `pending_new`
                # and `pending_refill` split it for anyone who needs the
                # distinction.
                "pending": len(self._pending) + len(self._refill),
                "pending_new": len(self._pending),
                "pending_refill": len(self._refill),
                # Requests this cohort has TAKEN ON: the set it sealed with
                # plus everything it has pulled since. Not the number of rows
                # decoding right now -- a cohort at four slots legitimately
                # reports eight here once four of its requests have finished
                # and been replaced.
                #
                # Reporting only `_active` (the sealed set) was worse: it made
                # a cohort that grew from two rows to eight read as two, which
                # beside a non-empty queue IS the soak's zombie-slot signature,
                # produced by a healthy cohort.
                #
                # **Neither number is the zombie-slot signal.** That is
                # `live_cohort.driver.live_rows` against `max_rows`, which only
                # the driver knows and which it now reports at cohort start as
                # well as at every admission boundary. Use that; this field is
                # for "how much work has this cohort absorbed".
                "active": len(self._active) + len(self._refill),
                "active_sealed": len(self._active),
                "active_joined": len(self._refill),
                "pump_scheduled": self._pump_scheduled,
                "cohort_slots": self.cohort_slots,
                "last_real_width": self._last_real_width,
                "last_route_id": str(getattr(self.lane, "route_id", "")),
                "last_error": self._last_error,
                "batch_histogram": {
                    str(width): count
                    for width, count in sorted(self._batch_histogram.items())
                },
                "target_verify_cycles": self._cycles,
                "accepted_draft_tokens": self._accepted_drafts,
                "drafted_tokens": self._drafted,
                "draft_accept_rate": accept_rate,
                "solo_runs": self._solo_runs,
                "left_pad_tokens_total": self._pad_tokens_total,
                "cohort_failures": self._cohort_failures,
                "rejected_total": self._rejected_total,
                "live_cohort": self._live_view(),
                "cohort_duration_s": self._percentiles(list(self._recent_cohort_s)),
                "queue_wait_s": self._percentiles(list(self._recent_queue_wait_s)),
                "max_queue_depth": self.max_queue_depth,
                "cohort_deadline_s": self.cohort_deadline_s,
                **{f"last_{k}": v for k, v in self._last_kv.items()},
                "refill_admitted_total": self._refill_admitted_total,
                "refill_requeued_total": self._refill_requeued_total,
                "requests_served_total": self._requests_served_total,
                "max_requests_in_one_cohort": self._max_requests_in_one_cohort,
                "continuous_batching_observed": (
                    self._max_requests_in_one_cohort > self._last_real_width
                    and self._refill_admitted_total > 0
                ),
                "continuous": self.continuous,
                "max_requests_per_cohort": self.max_requests_per_cohort,
                # Rebuilds of a cohort's row axis, and the widest any cohort
                # actually ran. Under continuous batching `rows_peak` exceeding
                # the sealed width IS demand-following, and a `cohort_resizes`
                # of zero on a busy server says the queue is never being pulled
                # from -- which is what "the feature is present but not
                # working" looks like from the outside.
                "cohort_resizes_total": self._cohort_resizes,
                "rows_peak": self._rows_peak,
                # A lane running narrow because it is out of GPU headroom looks
                # exactly like a lane nobody is using. This is the difference.
                "rows_blocked_by_memory": self._rows_blocked_by_memory,
                # Requests refused because ONE of them could not fit in memory
                # even alone. Distinct from `rejected_total`, which is the queue
                # bound: this one says the machine is too small for the request,
                # not too busy for it.
                "unservable_total": self._unservable_total,
                # T-210. `hit_rate` is the number that says whether this is
                # earning its memory: high on chat traffic is the point, high on
                # independent requests would be suspicious, and zero on a busy
                # server means it is holding memory for nothing.
                "prefix_cache": (
                    # `to_dict`, not `stats`: the bank reports itself in its
                    # own vocabulary and the snapshot passes it through rather
                    # than inventing a parallel one that can drift.
                    # `is not None`, NOT truthiness: SessionBank defines
                    # __len__, so an EMPTY bank is falsy. Testing it as a
                    # boolean reports "no prefix cache configured" during
                    # exactly the window an operator checks -- right after
                    # turning it on, before anything is stored.
                    self._prefix_cache_snapshot()
                ),
                "kv_bytes_per_token_per_row": self._kv_bytes_per_token_per_row(),
                "last_error_context": self._last_error_context,
                "ragged_prompts": True,
                "last_owner_finalize": dict(self._last_owner_finalize),
            }

    def _prefix_cache_snapshot(self) -> dict[str, Any] | None:
        """What the bank holds, plus whether holding it is worth the memory.

        `is not None`, NOT truthiness: SessionBank defines __len__, so an EMPTY
        bank is falsy and a boolean test reports "no prefix cache configured"
        during exactly the window an operator checks -- right after turning it
        on, before anything has been stored.
        """

        if self.prefix_bank is None:
            return None
        out = dict(self.prefix_bank.to_dict())
        served = int(self._requests_served_total)
        out["restores"] = int(self._prefix_restores_total)
        out["restore_failures"] = int(self._prefix_restore_failures_total)
        out["prompt_tokens_skipped"] = int(self._prefix_tokens_skipped_total)
        # The number that decides whether this feature stays on. High on chat
        # traffic is the point; high on independent requests would be
        # suspicious; zero on a busy server means it is holding gigabytes for
        # nothing, which is worse than not having it.
        out["hit_rate"] = (
            round(self._prefix_restores_total / served, 4) if served else 0.0
        )
        return out

    # ---- submission and sealing -----------------------------------------

    def submit(self, job: DenseMTPBatchJob) -> Future:
        schedule = False
        with self._condition:
            if self._shutdown:
                job.future.set_exception(
                    RuntimeError("dense mtp_batch service is shut down")
                )
                return job.future
            # Jobs already committed to a running cohort's refill queue are
            # still backlog: they have not been served, and leaving them out
            # measured the bound against a number smaller than the true queue.
            if len(self._pending) + len(self._refill) >= self.max_queue_depth:
                # Raised, not delivered through the future: a rejected request
                # should fail immediately rather than wait on a cohort it was
                # never admitted to.
                self._rejected_total += 1
                raise DenseMTPBatchQueueFull(
                    f"dense mtp_batch queue is full "
                    f"({len(self._pending) + len(self._refill)}"
                    f"/{self.max_queue_depth} waiting); "
                    "retry shortly"
                )
            if not job.finalize_ownership.accept_owner():
                job.cancel_event.set()
                job.future.set_exception(self._cancelled_exception(job))
                return job.future
            job.finalize_owner_accepted = True
            self._pending.append(job)
            if self.auto_schedule and not self._pump_scheduled:
                self._pump_scheduled = True
                schedule = True
            self._condition.notify_all()
        if schedule:
            self._schedule_pump()
        return job.future

    def _schedule_pump(self) -> None:
        scheduler = getattr(self.state, "model_scheduler", None)
        if scheduler is None or not hasattr(scheduler, "submit_foreground"):
            exc = RuntimeError(
                "dense mtp_batch service requires the model-owner scheduler"
            )
            self._fail_pending(exc)
            return
        scheduler.submit_foreground(self._pump, batch_key="dense_mtp_batch.pump")

    def _cancelled_exception(self, job: DenseMTPBatchJob) -> BaseException:
        return job.cancel_error(job)

    def _drain_cancelled_locked(self) -> list[DenseMTPBatchJob]:
        keep: list[DenseMTPBatchJob] = []
        cancelled: list[DenseMTPBatchJob] = []
        for job in self._pending:
            if job.cancel_requested() or job.future.cancelled():
                cancelled.append(job)
            else:
                keep.append(job)
        self._pending = keep
        return cancelled

    def _length_compatible(self, longest: int, candidate: int) -> bool:
        """True when ``candidate`` may share a cohort with a prompt of ``longest``.

        Unlimited by default: after item 2 there is no padding cost to bound.
        When an operator sets a bound, it is the LARGER of an absolute token
        allowance and a fraction of the longest prompt, so short cohorts are not
        held hostage by the ratio and long ones are not held hostage by the
        absolute number.
        """

        if self.length_spread_tokens <= 0 and self.length_spread_ratio <= 0.0:
            return True
        spread = int(longest) - int(candidate)
        if spread <= 0:
            return True
        allowance = max(
            self.length_spread_tokens,
            int(math.ceil(self.length_spread_ratio * int(longest))),
        )
        return spread <= allowance

    def _compatible_pending_locked(self, base: DenseMTPBatchJob) -> list[DenseMTPBatchJob]:
        """Pick the jobs that may share a cohort with ``base``.

        Two passes, because length compatibility is relative to the cohort's
        longest prompt and that is not known until the members are chosen: take
        every key-compatible job in arrival order, then drop the ones the
        winning length would over-pad.
        """

        candidates = [
            job
            for job in self._pending
            if job.compatibility_key == base.compatibility_key
            and not job.cancel_requested()
        ][: self.cohort_slots]
        if not candidates:
            return []
        selected: list[DenseMTPBatchJob] = []
        longest = 0
        for job in candidates:
            trial = max(longest, len(job.prompt_ids))
            if any(
                not self._length_compatible(trial, len(other.prompt_ids))
                for other in (*selected, job)
            ):
                continue
            selected.append(job)
            longest = trial
        return selected

    def _refill_candidates_locked(
        self, cohort: list[DenseMTPBatchJob]
    ) -> list[DenseMTPBatchJob]:
        """Pending jobs eligible to join ``cohort`` when a slot frees.

        Same compatibility key, which binds the route and the stop set. The
        sampler is deliberately NOT required to match: the driver rebuilds its
        per-row sampling vectors for an admitted row, so a joiner keeps its own
        temperature rather than inheriting the previous occupant's.

        The one exception mirrors the driver's own rule. An all-greedy cohort
        runs the dedicated greedy path, which consumes no randomness and has no
        sampling machinery to update, so a candidate wanting temperature > 0
        cannot join one. Filtered here so the driver's fail-loud check is a
        backstop rather than something operators meet.
        """

        if not cohort:
            return []
        key = cohort[0].compatibility_key
        cohort_samples = any(
            _sampler_triple(job.sampler)[0] > 0.0 for job in cohort
        )
        cohort_penalised = any(
            any(_sampler_penalties(job.sampler)) for job in cohort
        )
        # The operator's length-spread bound has to apply here too. It is a
        # fairness policy -- keep a very long prompt from occupying a slot an
        # operator wanted kept for short requests -- and a bound that cohort
        # selection honours while refill ignores would be silently bypassed by
        # any queue, which is worse than not having it.
        longest = max(len(job.prompt_ids) for job in cohort)
        eligible: list[DenseMTPBatchJob] = []
        for job in self._pending:
            if job.compatibility_key != key or job.cancel_requested():
                continue
            if _sampler_triple(job.sampler)[0] > 0.0 and not cohort_samples:
                continue
            # Same shape of rule for penalties. The driver decides once, at
            # cohort start, whether to build the penalty machinery at all; a
            # joiner wanting a penalty cannot be served by a cohort that has
            # none, and would otherwise be served UNPENALISED and silent.
            if any(_sampler_penalties(job.sampler)) and not cohort_penalised:
                continue
            span = max(longest, len(job.prompt_ids))
            if not self._length_compatible(span, len(job.prompt_ids)):
                continue
            if not self._length_compatible(span, longest):
                continue
            eligible.append(job)
            if len(eligible) >= self.refill_depth:
                break
        return eligible

    def _kv_budget_bytes(self) -> float:
        """What one cohort may reserve, or 0.0 when it cannot be determined."""

        if self.memory_headroom <= 0.0:
            return 0.0
        try:
            import mlx.core as mx

            info = getattr(mx, "device_info", None) or mx.metal.device_info
            return float(self.memory_headroom) * float(
                info()["max_recommended_working_set_size"]
            )
        except Exception:
            return 0.0

    def _kv_bytes_per_token_per_row(self) -> float:
        """Measured from the LAST cohort rather than assumed.

        Self-calibrating on purpose. The number depends on layer count, KV head
        count, head dimension and dtype, and a constant baked in here would be
        wrong for the next model somebody points this lane at -- and wrong in
        the direction of refusing work the machine could have served.
        """

        return float(self._last_kv.get("kv_bytes_per_token_per_slot") or 0.0)

    def _fit_cohort_to_memory(
        self, selected: list[DenseMTPBatchJob]
    ) -> tuple[list[DenseMTPBatchJob], list[DenseMTPBatchJob]]:
        """Split ``selected`` into (runnable, unservable).

        Rows that merely do not fit ALONGSIDE the others are dropped from the
        cohort and left pending -- they are served by the next one. Only a
        request whose own reservation cannot fit on an empty machine is
        unservable, and it is told so.
        """

        budget = self._kv_budget_bytes()
        per_token = self._kv_bytes_per_token_per_row()
        if budget <= 0.0 or per_token <= 0.0 or not selected:
            # No measurement yet (the first cohort after startup) or the check
            # is switched off. Deliberately permissive: guessing here and
            # guessing high would refuse work this machine could serve.
            return selected, []

        def cost(job: DenseMTPBatchJob) -> float:
            return per_token * (len(job.prompt_ids) + int(job.max_tokens))

        runnable: list[DenseMTPBatchJob] = []
        unservable: list[DenseMTPBatchJob] = []
        spent = 0.0
        for job in selected:
            own = cost(job)
            if own > budget:
                unservable.append(job)
                continue
            if spent + own > budget:
                # Fits alone, does not fit here. Not this caller's fault and
                # not their problem: leave it pending.
                continue
            spent += own
            runnable.append(job)
        return runnable, unservable

    def _reject_unservable(self, jobs: list[DenseMTPBatchJob]) -> None:
        budget = self._kv_budget_bytes()
        per_token = self._kv_bytes_per_token_per_row()
        affordable = int(budget / per_token) if per_token > 0 else 0
        for job in jobs:
            asked = len(job.prompt_ids) + int(job.max_tokens)
            job.finish_finalize_ownership(finalized=False)
            if not job.future.done():
                job.future.set_exception(
                    DenseMTPBatchQueueFull(
                        "this request cannot be served: prompt + max_tokens is "
                        f"{asked} tokens, and one row's working memory on this "
                        f"machine tops out near {affordable} tokens "
                        f"({per_token / 1024:.0f} KiB per token per row). "
                        "Lower max_tokens, or serve it on a machine with more "
                        "memory. Refused here rather than allowed to take its "
                        "cohort-mates down with an out-of-memory."
                    )
                )

    def _seal(self, *, wait: bool) -> list[DenseMTPBatchJob]:
        cancelled: list[DenseMTPBatchJob] = []
        selected: list[DenseMTPBatchJob] = []
        with self._condition:
            if self._shutdown:
                return []
            cancelled.extend(self._drain_cancelled_locked())
            if self._pending:
                base = self._pending[0]
                deadline = time.perf_counter() + (self.batch_wait_s if wait else 0.0)
                selected = self._compatible_pending_locked(base)
                while wait and len(selected) < self.cohort_slots:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    self._condition.wait(timeout=remaining)
                    cancelled.extend(self._drain_cancelled_locked())
                    if not self._pending:
                        selected = []
                        break
                    base = self._pending[0]
                    selected = self._compatible_pending_locked(base)
                # One oversized request must not take its cohort-mates down.
                # Found by the soak's fault-injection arm: a caller asked for
                # 260,943 tokens, which is inside the context window so nothing
                # clamped it, and the reservation killed a two-row cohort in
                # which the other request had asked for nothing unreasonable.
                #
                # BEFORE the jobs are claimed, not after. The first version ran
                # this after `mark_admitted`, so a row dropped merely for not
                # fitting ALONGSIDE the others had already left `_pending` and
                # was never served -- stranded, which is the precise failure
                # this fix exists to avoid and which its own test caught.
                selected, unservable = self._fit_cohort_to_memory(selected)
                claimed_ids = {id(job) for job in (*selected, *unservable)}
                self._pending = [
                    job for job in self._pending if id(job) not in claimed_ids
                ]
                now = time.perf_counter()
                admitted: list[DenseMTPBatchJob] = []
                for job in selected:
                    if job.finalize_ownership.mark_admitted():
                        job.admitted_s = now
                        admitted.append(job)
                    else:
                        cancelled.append(job)
                selected = admitted
                # NOT cancellations: a cancelled job asked to stop, this one was
                # refused. Resolved outside the lock, below, with a message that
                # names the number it asked for and the number that would fit.
                self._unservable_this_seal = unservable
                # Anything still pending and compatible rides along as the
                # refill queue, so a slot that frees mid-run is reused instead
                # of idling until the whole cohort drains.
                #
                # Under continuous batching this pre-move is exactly the
                # behaviour being removed: it decides, at seal time, which
                # waiting requests are allowed to join, and every request that
                # arrives afterwards is locked out for the life of the cohort.
                # The driver pulls from `_pending` directly instead.
                refill = (
                    []
                    if self.continuous
                    else self._refill_candidates_locked(selected)
                )
                refill_ids = {id(job) for job in refill}
                self._pending = [
                    job for job in self._pending if id(job) not in refill_ids
                ]
                now_refill = time.perf_counter()
                admitted_refill: list[DenseMTPBatchJob] = []
                for job in refill:
                    if job.finalize_ownership.mark_admitted():
                        job.admitted_s = now_refill
                        admitted_refill.append(job)
                    else:
                        cancelled.append(job)
                self._refill = admitted_refill
                self._active = list(selected) + list(admitted_refill)
                self._last_real_width = len(selected)
                if selected:
                    self._batch_histogram[len(selected)] += 1
        if cancelled:
            for job in cancelled:
                # Pending jobs never allocated request-owned MLX state.
                job.finish_finalize_ownership(finalized=False)
                if not job.future.done():
                    job.future.set_exception(self._cancelled_exception(job))
        unservable = getattr(self, "_unservable_this_seal", None)
        if unservable:
            self._unservable_this_seal = []
            with self._condition:
                self._unservable_total += len(unservable)
            self._reject_unservable(unservable)
        return selected

    # ---- execution -------------------------------------------------------

    def pump_once(self) -> bool:
        jobs = self._seal(wait=False)
        if not jobs:
            return False
        self._run_sealed(jobs)
        return True

    def _pump(self) -> None:
        try:
            while True:
                jobs = self._seal(wait=True)
                if not jobs:
                    return
                self._run_sealed(jobs)
        finally:
            schedule = False
            with self._condition:
                self._pump_scheduled = False
                if self._pending and not self._shutdown:
                    self._pump_scheduled = True
                    schedule = True
                self._condition.notify_all()
            if schedule:
                self._schedule_pump()

    def _run_sealed(self, jobs: list[DenseMTPBatchJob]) -> None:
        # Read WITHOUT clearing. `self._refill` used to be emptied here, which
        # made the queue-depth fix almost inert: for the whole duration of the
        # cohort -- exactly when backlog matters -- these jobs existed only in
        # this local, invisible to snapshot() and to the backpressure bound. A
        # mutation audit caught it, because reverting the fix changed nothing
        # observable. The `finally` below still clears it, so the lifetime is
        # unchanged; only the visibility is.
        refill = list(self._refill)
        try:
            # The solo fast path is correct only when nothing can join. Under
            # continuous batching something always can, and taking it anyway is
            # what produced the convoy this task exists to remove: measured on
            # the 4B at 7f37ec4, eight simultaneous requests sealed as ONE solo
            # run plus a cohort of seven, and those seven waited 7.9 seconds --
            # a whole solo generation -- before starting. A one-row continuous
            # cohort is slower than the tuned solo loop for a request that
            # really is alone; it is enormously faster for one that is not, and
            # the service cannot tell the two apart at seal time.
            if len(jobs) == 1 and not refill and not self.continuous:
                self._run_solo(jobs[0])
            else:
                self._run_cohort(jobs, refill)
        except BaseException as exc:
            with self._condition:
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._cohort_failures += 1
            # EVERY job in the run, joiners included. Iterating `jobs` alone
            # left refill joiners with no exception, no result and no ownership
            # finalisation, so their futures never resolved and their callers
            # hung until timeout. A cohort-mate that gets an error can retry; a
            # caller that gets silence cannot even tell something went wrong.
            #
            # RE-READ `self._refill` here rather than using the `refill` local
            # captured before the run. Under continuous batching that local is
            # empty and the membership is not known until the run is over,
            # because joiners are pulled DURING it. Using the stale local
            # reintroduced this exact hang for every pulled joiner, and the
            # test above caught it within minutes of the change landing.
            with self._condition:
                in_run = [*jobs, *self._refill]
            for job in in_run:
                if not job.future.done():
                    job.future.set_exception(exc)
                job.finish_finalize_ownership(finalized=False)
        finally:
            live = self._live
            if live is not None:
                self._recent_cohort_s.append(
                    max(0.0, time.perf_counter() - live["started_s"])
                )
            self._live = None
            with self._condition:
                self._active = []
                self._refill = []
                self._condition.notify_all()

    def _run_solo(self, job: DenseMTPBatchJob) -> None:
        """A cohort of one runs the ordinary solo MTP path.

        Same call the A3B lane makes, for the same reason: a one-row cohort
        through the batch driver would be slower than the tuned solo loop, so
        batching must never cost a lone request latency.
        """

        try:
            if job.cancel_requested():
                raise self._cancelled_exception(job)
            if job.solo_runner is None:
                raise RuntimeError("dense mtp_batch solo request has no solo runner")
            with self._condition:
                self._solo_runs += 1
            result = dict(job.solo_runner(job))
        except BaseException:
            finalized = False
            try:
                self._finalize_on_owner([job])
                finalized = True
            finally:
                job.finish_finalize_ownership(finalized=finalized)
            raise
        result["_dense_mtp_batch_solo"] = True
        # A solo request decoded at width 1, and must say so. Without this a
        # caller only learns its width when it happened to be batched, so a
        # single-request baseline has nothing to compare a batched run against
        # -- which is the exact comparison section 1 of the contract tells
        # callers to make.
        # UNCONDITIONAL, not conditional on a stats dict already existing. The
        # first version guarded on `isinstance(result.get("stats"), dict)` and
        # so did nothing at all when the solo runner returned no stats -- the
        # field was absent exactly when there was least reason to expect it,
        # which is the failure mode a caller cannot distinguish from "width
        # unknown". The guarantee is that every response reports its width; a
        # guarantee that holds only when some other key happens to be present
        # is not one.
        solo_stats = result.get("stats")
        if not isinstance(solo_stats, dict):
            solo_stats = {}
            result["stats"] = solo_stats
        solo_stats.setdefault("dense_mtp_batch_cohort_width", 1)
        job.finish_finalize_ownership(finalized=True)
        if not job.future.done():
            job.future.set_result(result)

    def _run_cohort(
        self,
        jobs: list[DenseMTPBatchJob],
        refill: list[DenseMTPBatchJob] | None = None,
    ) -> None:
        started = time.perf_counter()
        refill = list(refill or [])
        # MUTABLE on purpose: under continuous batching `_pull` appends to this
        # as requests join, and the driver's request indices are positions in
        # it. Everything a caller gets back is keyed on that index.
        all_jobs = [*jobs, *refill]
        real_width = len(jobs)
        lane = self.lane
        for job in all_jobs:
            job.emit_prefill(
                {
                    "phase": "started",
                    "tokens_total": len(job.prompt_ids),
                    "scheduler_lane": "dense_mtp_batch",
                    "request_id": job.request_id,
                }
            )

        # Item 2: true lengths, no padding. ``ragged_prompts`` groups the rows
        # by length and pins per-row offsets, so a short prompt pays its own
        # prefill and carries its own KV, and its output is what it would have
        # been alone.
        prompts = [list(job.prompt_ids) for job in jobs]
        for job in jobs:
            job.pad_tokens = 0

        row_caps = [int(job.max_tokens) for job in jobs]
        # Item 3: per-row sampling. Greedy and sampling callers coexist.
        triples = [_sampler_triple(job.sampler) for job in jobs]
        row_temperature = [t for t, _p, _k in triples]
        row_top_p = [p for _t, p, _k in triples]
        row_top_k = [k for _t, _p, k in triples]
        stop_ids = set(jobs[0].stop_token_ids)
        cohort_seed = int(jobs[0].seed)

        def _is_cancelled(request: int) -> bool:
            # Eviction probe. A caller that disconnects mid-stream should give
            # its slot back at the next cycle boundary rather than at the end
            # of its token budget, or the lane loses capacity to rows nobody is
            # listening to.
            if request >= len(all_jobs):
                return False
            job = all_jobs[request]
            return job.cancel_requested() or job.callback_error is not None

        now = time.perf_counter()
        self._live = {
            "started_s": now,
            "last_progress_s": now,
            "width": len(jobs),
            "requests": len(all_jobs),
            "tokens_committed": 0,
            "request_ids": [job.request_id for job in all_jobs][:16],
            # Why this cohort is not wider, when it is not. `None` means
            # nothing is holding it back -- which, beside a non-empty queue and
            # a width below capacity, is the zombie-slot defect and nothing
            # else. Every other value names a normal reason.
            "growth_blocked": None,
            "driver": {},
        }
        for job in all_jobs:
            self._recent_queue_wait_s.append(
                max(0.0, (job.admitted_s or job.created_s) - job.created_s)
            )

        decode_started = {"at": None}

        def _on_commit(request: int, token: int) -> None:
            # The driver reports a REQUEST index, and requests are ordered
            # initial-cohort-then-queue, which is exactly `all_jobs`. So a
            # joiner's tokens reach its caller through the same path as
            # everyone else's and joiners stream like any other request.
            if decode_started["at"] is None:
                decode_started["at"] = time.perf_counter()
                for item in all_jobs:
                    item.mark_decode_started()
            if request < len(all_jobs):
                all_jobs[request].emit_token(token)
            # Unlocked on purpose: two scalar writes on the commit path. A torn
            # read of a float cannot happen under CPython, and taking the
            # service lock per token would put contention in the hot path to
            # make a diagnostic marginally fresher.
            live = self._live
            if live is not None:
                live["tokens_committed"] += 1
                live["last_progress_s"] = time.perf_counter()

        def _payload(job: DenseMTPBatchJob) -> dict[str, Any]:
            triple = _sampler_triple(job.sampler)
            penalties = _sampler_penalties(job.sampler)
            return {
                "prompt": list(job.prompt_ids),
                "max_new_tokens": int(job.max_tokens),
                "temperature": triple[0],
                "top_p": triple[1],
                "top_k": triple[2],
                # A joiner brings its own seed. Without it the row would keep
                # decoding from wherever its predecessor's stream had got to --
                # the same cross-request coupling, sourced from a request that
                # has already finished.
                "seed": int(job.seed),
                "presence_penalty": penalties[0],
                "frequency_penalty": penalties[1],
                # Which conversation this row belongs to. The bank uses it to
                # bound how much ONE conversation may hold: every turn stores a
                # fresh copy of its whole prompt, so a long chat left in the
                # global pool evicts everyone else's work before its own.
                "session_id": job.session_id,
            }

        refill_payload = [_payload(job) for job in refill]

        # --- the live queue --------------------------------------------------
        # Called by the driver on the model-owner thread at every cycle
        # boundary, with the number of rows the cohort could take right now.
        # Same thread as `_on_commit`, so appending to `all_jobs` here needs no
        # lock of its own; the service lock guards `_pending` only.
        def _on_stats(payload: dict[str, Any]) -> None:
            # Called on the model-owner thread at every admission boundary. Two
            # scalar writes, no lock: this is a diagnostic, and taking the
            # service lock here would put contention on the hot path to make it
            # marginally fresher.
            live = self._live
            if live is not None:
                live["driver"] = payload

        cohort_key = jobs[0].compatibility_key
        cohort_samples = any(_sampler_triple(job.sampler)[0] > 0.0 for job in jobs)
        cohort_penalised = any(any(_sampler_penalties(job.sampler)) for job in jobs)
        longest_prompt = [max(len(job.prompt_ids) for job in jobs)]

        def _note_growth(reason: str | None) -> None:
            live = self._live
            if live is not None:
                live["growth_blocked"] = reason

        def _pull(capacity: int) -> list[dict[str, Any]]:
            capacity = int(capacity)
            if capacity <= 0:
                # The driver has no room -- it is at its width ceiling, or every
                # row is busy. Not a refusal by this queue.
                _note_growth("at_width_cap")
                return []
            room = self.max_requests_per_cohort - len(all_jobs)
            if room <= 0:
                # WINDING DOWN. The cohort has served its request budget and
                # will not take more; the rows drain and a fresh cohort forms.
                # This is the state that looks exactly like a zombie slot from
                # outside, and it is entirely normal.
                _note_growth("cohort_request_cap")
                return []
            capacity = min(capacity, room)
            taken: list[DenseMTPBatchJob] = []
            with self._condition:
                if self._shutdown:
                    return []
                for job in list(self._pending):
                    if len(taken) >= capacity:
                        break
                    if job.compatibility_key != cohort_key or job.cancel_requested():
                        continue
                    # The driver rebuilds a joining row's sampling vectors, so
                    # the sampler need not match -- with two exceptions that
                    # mirror the driver's own one-time decisions. An all-greedy
                    # cohort runs the dedicated greedy path, which consumes no
                    # randomness and has no sampling machinery to update; and
                    # the penalty buffers are allocated once at cohort start.
                    # A candidate needing either would be served WRONG and
                    # silent, so it waits for the next cohort instead.
                    # The driver ACQUIRES randomness and penalty machinery for
                    # a joiner that needs it (`_upgrade_for`), so neither is a
                    # reason to hold a request back any more. Filtering on them
                    # here was measured costing 27-second queue waits under an
                    # ordinary mixed load: 65% of requests greedy meant most
                    # cohorts sealed all-greedy, and every sampling request
                    # then waited out a cohort that could serve eight times its
                    # width before winding down.
                    #
                    # One case the driver genuinely cannot acquire: a compiled
                    # draft chain has no sampling path. The lane installs
                    # draft_core='eager' only, so this is a guard against a
                    # future revision rather than a live condition.
                    if (
                        _sampler_triple(job.sampler)[0] > 0.0
                        and str(getattr(lane, "draft_core", "eager")) != "eager"
                    ):
                        continue
                    # The operator's length-spread bound applies here too. It
                    # is a fairness policy -- keep one very long prompt from
                    # occupying a row an operator wanted kept for short
                    # requests -- and a bound that cohort selection honours
                    # while admission ignores is silently bypassed by any queue.
                    span = max(longest_prompt[0], len(job.prompt_ids))
                    if not self._length_compatible(span, len(job.prompt_ids)):
                        continue
                    if not self._length_compatible(span, longest_prompt[0]):
                        continue
                    if not job.finalize_ownership.mark_admitted():
                        continue
                    taken.append(job)
                if taken:
                    taken_ids = {id(job) for job in taken}
                    self._pending = [
                        job for job in self._pending if id(job) not in taken_ids
                    ]
                    self._refill.extend(taken)
                    self._condition.notify_all()
            if not taken:
                _note_growth(
                    "no_compatible_work" if self._pending else "queue_empty"
                )
                return []
            _note_growth(None)
            now_admitted = time.perf_counter()
            payloads: list[dict[str, Any]] = []
            for job in taken:
                job.admitted_s = now_admitted
                self._recent_queue_wait_s.append(
                    max(0.0, now_admitted - job.created_s)
                )
                longest_prompt[0] = max(longest_prompt[0], len(job.prompt_ids))
                job.emit_prefill(
                    {
                        "phase": "started",
                        "tokens_total": len(job.prompt_ids),
                        "scheduler_lane": "dense_mtp_batch",
                        "request_id": job.request_id,
                    }
                )
                all_jobs.append(job)
                payloads.append(_payload(job))
            live = self._live
            if live is not None:
                live["requests"] = len(all_jobs)
                live["joined"] = live.get("joined", 0) + len(taken)
            return payloads
        # The driver call belongs INSIDE the try whose finally finalises
        # ownership. It used to sit outside, so a raise here propagated past
        # the cleanup entirely: no MLX finalisation, no futures resolved.
        try:
            with batch_generic_cache_scope():
                result = self.driver(
                    lane.runtime,
                    prompts,
                    max_new_tokens=max(row_caps),
                    max_new_tokens_per_row=row_caps,
                    depth=int(lane.geometry.depth),
                    stop_token_ids=stop_ids,
                    capture_backend=str(lane.capture_backend),
                    head_history=str(lane.head_history),
                    history_window=int(lane.history_window),
                    prefill_chunk=int(lane.prefill_chunk),
                    loop_mode=str(lane.loop_mode),
                    draft_core=str(lane.draft_core),
                    pad_id=int(getattr(lane, "pad_id", 0)),
                    ragged_prompts=True,
                    # `jobs`, not `all_jobs`: `prompts` is built from `jobs`
                    # alone, and relying on the sealed jobs happening to come
                    # first in `all_jobs` would misalign the moment that stops
                    # being true.
                    session_ids=[job.session_id for job in jobs],
                    temperature=row_temperature,
                    top_k=row_top_k,
                    top_p=row_top_p,
                    sampling_seed=cohort_seed,
                    row_sampling_seeds=[int(job.seed) for job in jobs],
                    presence_penalty=[
                        _sampler_penalties(job.sampler)[0] for job in jobs
                    ],
                    frequency_penalty=[
                        _sampler_penalties(job.sampler)[1] for job in jobs
                    ],
                    deadline_s=self.cohort_deadline_s,
                    on_commit=_on_commit,
                    refill_queue=refill_payload or None,
                    is_cancelled=_is_cancelled,
                    pull_queued=_pull if self.continuous else None,
                    max_cohort_rows=self.cohort_slots,
                    memory_headroom=self.memory_headroom,
                    on_stats=_on_stats,
                    session_bank=self.prefix_bank,
                )
        except BaseException as exc:
            # An out-of-memory here is the one failure an operator can actually
            # act on, and the raw Metal message names nothing they control. Say
            # what the cohort was doing when it ran out, so the answer
            # ("lower --decode-batch-max", or cap the context) is derivable
            # from the log rather than from a reproduction.
            if "Insufficient Memory" in str(exc) or "out of memory" in str(exc).lower():
                longest = max((len(job.prompt_ids) for job in all_jobs), default=0)
                self._last_error_context = {
                    "sealed_width": len(jobs),
                    "requests_in_cohort": len(all_jobs),
                    "longest_prompt_tokens": longest,
                    "max_tokens_requested": max(
                        (int(job.max_tokens) for job in all_jobs), default=0
                    ),
                    "cohort_slots": self.cohort_slots,
                    "advice": (
                        "the cohort ran out of GPU working set. Lower "
                        "--decode-batch-max, or cap the context callers may "
                        "send; KV is per row and scales with the LONGEST "
                        "concurrent prompt, not the average one"
                    ),
                }
            # Finalise before re-raising so the model-owner thread's MLX state
            # is cleaned up and no job is left holding finalize ownership. The
            # outer handler resolves the futures.
            finalized = False
            try:
                self._finalize_on_owner(all_jobs)
                finalized = True
            finally:
                for job in all_jobs:
                    job.finish_finalize_ownership(finalized=finalized)
            raise
        self._last_cohort_seed = cohort_seed
        meta = getattr(result, "meta", None) or {}
        # Width receipts. `rows_peak` is how wide the cohort actually ran, which
        # under continuous batching is NOT the sealed width and is the single
        # number that shows demand-following working.
        with self._condition:
            self._cohort_resizes += int(meta.get("cohort_resizes") or 0)
            self._rows_peak = max(
                self._rows_peak, int(meta.get("rows_peak") or 0)
            )
            self._rows_blocked_by_memory += int(
                meta.get("rows_blocked_by_memory") or 0
            )
            self._prefix_restores_total += int(meta.get("prefix_restores") or 0)
            self._prefix_tokens_skipped_total += int(
                meta.get("prefix_prompt_tokens_skipped") or 0
            )
            self._prefix_restore_failures_total += int(
                meta.get("prefix_restore_failures") or 0
            )
        self._last_kv = {
            key: meta[key]
            for key in (
                "kv_reserved_bytes",
                "kv_capacity_tokens_per_slot",
                "kv_bytes_per_token_per_slot",
                "kv_used_bytes_estimate",
            )
            if key in meta
        }

        with self._condition:
            self._cycles += int(result.cycles)
            self._accepted_drafts += int(result.accepted_draft_tokens)
            self._drafted += int(sum(result.drafted_by_depth))
            self._pad_tokens_total += sum(job.pad_tokens for job in jobs)

        # Results are one per REQUEST, in submission order: initial cohort
        # first, then the refill queue. A joiner that never got a slot comes
        # back "not_admitted" and returns to the pending queue rather than
        # failing, because it has not been served and its caller is waiting.
        streams = list(result.streams)
        if len(streams) != len(all_jobs):
            raise RuntimeError(
                f"driver returned {len(streams)} results for {len(all_jobs)} "
                "requests; refusing to guess the mapping"
            )
        requeue = [
            job
            for job, stream in zip(all_jobs, streams, strict=True)
            if str(stream.finish_reason) == "not_admitted"
        ]
        requeue_ids = {id(job) for job in requeue}

        successful: list[tuple[DenseMTPBatchJob, Any]] = []
        try:
            for _row_idx, (job, stream) in enumerate(
                zip(all_jobs, streams, strict=True)
            ):
                # The driver reports prefix reuse per ROW; a caller needs it per
                # REQUEST. `all_jobs` and `streams` are zipped strictly, so the
                # position in this loop IS the cohort row, and stashing it here
                # is what lets the completion below tell the truth about how
                # much of this prompt was actually re-read.
                job.cohort_row = int(_row_idx)
                if id(job) in requeue_ids:
                    continue
                if job.callback_error is not None:
                    if not job.future.done():
                        job.future.set_exception(job.callback_error)
                    continue
                if job.cancel_requested():
                    if not job.future.done():
                        job.future.set_exception(self._cancelled_exception(job))
                    continue
                successful.append((job, stream))
        finally:
            finalized = False
            try:
                self._finalize_on_owner(all_jobs)
                finalized = True
            finally:
                for job in all_jobs:
                    job.finish_finalize_ownership(finalized=finalized)

        served = len(all_jobs) - len(requeue)
        joined = len(all_jobs) - len(jobs)
        with self._condition:
            self._refill_admitted_total += joined - len(requeue)
            self._refill_requeued_total += len(requeue)
            self._requests_served_total += served
            self._max_requests_in_one_cohort = max(
                self._max_requests_in_one_cohort, served
            )

        if requeue:
            # Front of the queue: these callers have already waited a whole
            # cohort without being served.
            with self._condition:
                self._pending = [*requeue, *self._pending]
                self._condition.notify_all()

        for job, stream in successful:
            self._complete_cohort_job(
                job,
                stream=stream,
                result=result,
                real_width=real_width,
                cohort_started_s=started,
            )

    def _finalize_on_owner(self, jobs: list[DenseMTPBatchJob]) -> dict[str, Any]:
        if self.owner_finalize is None:
            return {}
        try:
            receipt = dict(self.owner_finalize(jobs) or {})
        except Exception as exc:
            error = RuntimeError(
                f"dense mtp_batch owner finalize failed: {type(exc).__name__}: {exc}"
            )
            receipt = {"error": str(error)}
            self._poison_owner_finalize(error, receipt)
            raise error from exc
        cleanup = receipt.get("mlx_cache_cleanup")
        if isinstance(cleanup, dict) and cleanup.get("cleared") is False:
            reason = str(cleanup.get("reason") or "cleanup_not_cleared")
            error = RuntimeError(f"dense mtp_batch owner finalize failed: {reason}")
            self._poison_owner_finalize(error, receipt)
            raise error
        with self._condition:
            self._last_owner_finalize = receipt
        return receipt

    def _poison_owner_finalize(
        self, error: RuntimeError, receipt: dict[str, Any]
    ) -> None:
        with self._condition:
            pending = list(self._pending)
            self._pending.clear()
            self._last_owner_finalize = receipt
            self._last_error = f"{type(error).__name__}: {error}"
            self._shutdown = True
            self._condition.notify_all()
        for job in pending:
            job.finish_finalize_ownership(finalized=False)
            if not job.future.done():
                job.future.set_exception(error)

    def _complete_cohort_job(
        self,
        job: DenseMTPBatchJob,
        *,
        stream: Any,
        result: DenseBatchResult,
        real_width: int,
        cohort_started_s: float,
    ) -> None:
        if job.future.done():
            return
        completed_s = time.perf_counter()
        request_elapsed_s = max(0.0, completed_s - job.created_s)
        decode_started_s = job.decode_started_s or cohort_started_s
        decode_elapsed_s = max(0.0, completed_s - decode_started_s)
        prefill_elapsed_s = max(0.0, decode_started_s - cohort_started_s)
        generation_elapsed_s = max(0.0, completed_s - cohort_started_s)
        completion_tokens = len(job.tokens)
        decode_tok_s = (
            completion_tokens / decode_elapsed_s if decode_elapsed_s > 0 else 0.0
        )
        end_to_end_tok_s = (
            completion_tokens / request_elapsed_s if request_elapsed_s > 0 else 0.0
        )
        depth = int(result.depth)
        stats = {
            "mode": "mtp",
            "generation_mode": "mtp",
            "generated_tokens": completion_tokens,
            "elapsed_s": generation_elapsed_s,
            "decode_elapsed_s": decode_elapsed_s,
            "request_elapsed_s": request_elapsed_s,
            "prompt_eval_time_s": prefill_elapsed_s,
            "prefill_wall_time_s": prefill_elapsed_s,
            "decode_tok_s": decode_tok_s,
            "tok_s": decode_tok_s,
            "end_to_end_tok_s": end_to_end_tok_s,
            "mtp_depth": depth,
            "requested_mtp_depth": depth,
            "speculative_depth": depth,
            "requested_speculative_depth": depth,
            "verify_calls": int(result.cycles),
            "target_verify_cycles": int(result.cycles),
            "scheduler_lane": "dense_mtp_batch",
            "scheduler_mode": "mtp_batch",
            "scheduler_policy": f"dense_mtp_batch_width_{int(real_width)}",
            "request_id": job.request_id,
            "active_batch_size": real_width,
            "dense_mtp_batch_real_width": real_width,
            "dense_mtp_batch_cohort_slots": self.cohort_slots,
            "dense_mtp_batch_route_id": str(getattr(self.lane, "route_id", "")),
            # Item 2 removed padding entirely, so this is now always 0. It is
            # kept because it is the cleanest before-and-after signal that
            # ragged prompts are actually in effect: a non-zero value here means
            # something reintroduced padding.
            "dense_mtp_batch_left_pad_tokens": int(job.pad_tokens),
            "dense_mtp_batch_ragged_prompts": True,
            # The seed the cohort actually decoded under. A per-request seed is
            # not honoured inside a cohort (see the module docstring), so this
            # says what was used rather than letting the caller assume its own.
            "dense_mtp_batch_cohort_seed": int(
                getattr(self, "_last_cohort_seed", job.seed)
            ),
            # The width this request decoded at -- which under continuous
            # admission is not ONE number, and reporting it as if it were was
            # wrong. `batch_size` is the width the cohort SEALED at, so a
            # request that ran beside five others in a cohort that opened with
            # one was told "width 1". Measured in a hammer run: 140 of 156
            # responses claimed width 1 while the cohort's peak was 8.
            #
            # `_peak` is the widest the cohort ever ran and is the number a
            # caller comparing two runs wants. `_varied` is the honest answer to
            # "does a single width even apply to my request": when it is true,
            # the advice in contract section 1 to pin the width is not
            # actionable by the caller at all, and needs continuous admission
            # switched off rather than a flag set.
            "dense_mtp_batch_cohort_width": int(
                (getattr(result, "meta", None) or {}).get("rows_peak")
                or getattr(result, "batch_size", 0)
                or 0
            ),
            "dense_mtp_batch_cohort_width_sealed": int(
                getattr(result, "batch_size", 0) or 0
            ),
            "dense_mtp_batch_cohort_width_peak": int(
                (getattr(result, "meta", None) or {}).get("rows_peak")
                or getattr(result, "batch_size", 0)
                or 0
            ),
            "dense_mtp_batch_cohort_width_varied": bool(
                (getattr(result, "meta", None) or {}).get("cohort_resizes")
            ),
            "dense_mtp_batch_prompt_tokens_true": len(job.prompt_ids),
            "dense_mtp_batch_prompt_tokens_padded": len(job.prompt_ids)
            + int(job.pad_tokens),
            "dense_mtp_batch_cohort_accepted_drafts": int(
                result.accepted_draft_tokens
            ),
            "dense_mtp_batch_cohort_aggregate_tok_s": float(
                result.aggregate_decode_tokps
            ),
            "mtp_disabled_reason": None,
            "queue_wait_s": max(
                0.0, (job.admitted_s or job.created_s) - job.created_s
            ),
            "request_started_s": job.created_s,
            "server_seed": job.seed,
        }
        job.request_observability.update(
            {
                "scheduler_policy": f"dense_mtp_batch_width_{int(real_width)}",
                "dense_mtp_batch_real_width": real_width,
                "dense_mtp_batch_route_id": str(
                    getattr(self.lane, "route_id", "")
                ),
                "dense_mtp_batch_left_pad_tokens": int(job.pad_tokens),
            }
        )
        # How much of THIS prompt came from the prefix cache rather than being
        # re-read. Hardcoded to zero until now, so a row that reused 630 of its
        # 755 prompt tokens still reported `cached_tokens: 0` on the wire, and
        # every external instrument agreed the cache did nothing -- the usage
        # block, /metrics, and a multi-turn benchmark all read 0% while the
        # driver logs showed hits. A working optimisation nobody can observe
        # gets deleted by the next reader as dead weight.
        #
        # It goes in `stats` specifically: that is the dict the OpenAI usage
        # block is built from (`generated["stats"]["cached_tokens"]`), and
        # setting it only on the prefill progress payload leaves the wire
        # exactly as silent as before.
        _covered = int(
            (
                (getattr(result, "meta", None) or {}).get("prefix_covered_by_row")
                or {}
            ).get(getattr(job, "cohort_row", -1), 0)
            or 0
        )
        _covered = max(0, min(_covered, len(job.prompt_ids)))
        stats["cached_tokens"] = _covered
        stats["dense_mtp_batch_prefix_reused_tokens"] = _covered
        stats.update(job.request_observability)
        completion_prefill = {
            "phase": "completed",
            "tokens_total": len(job.prompt_ids),
            "tokens_done": len(job.prompt_ids),
            "cached_tokens": _covered,
            "new_prefill_tokens": max(0, len(job.prompt_ids) - _covered),
            "elapsed_s": prefill_elapsed_s,
            "prompt_eval_time_s": prefill_elapsed_s,
            "prefill_tok_s": (
                len(job.prompt_ids) / prefill_elapsed_s
                if prefill_elapsed_s > 0.0
                else None
            ),
            "cache_hit": _covered > 0,
            "scheduler_lane": "dense_mtp_batch",
            "request_id": job.request_id,
        }
        job.future.set_result(
            {
                "request_id": job.request_id,
                "tokens": list(job.tokens),
                "stats": stats,
                "prompt_tokens": len(job.prompt_ids),
                "completion_tokens": completion_tokens,
                "elapsed_s": generation_elapsed_s,
                "request_elapsed_s": request_elapsed_s,
                "tok_s": decode_tok_s,
                "end_to_end_tok_s": end_to_end_tok_s,
                "_final_state": None,
                "_token_times": list(job.token_times),
                "_generation_limits": dict(job.generation_limits),
                "_mtp_batch_defer_mlx_finalize": True,
                "_mtp_batch_decode_on_request": True,
                "_mtp_batch_stop_token_ids": sorted(job.stop_token_ids),
                "_mtp_batch_prefill_callback": job.prefill_callback,
                "_mtp_batch_prefill_completion": completion_prefill,
                "finish_reason": str(stream.finish_reason),
            }
        )

    # ---- teardown --------------------------------------------------------

    def _fail_pending(self, exc: BaseException) -> None:
        with self._condition:
            pending = list(self._pending)
            self._pending.clear()
            self._pump_scheduled = False
            self._last_error = f"{type(exc).__name__}: {exc}"
        for job in pending:
            job.finish_finalize_ownership(finalized=False)
            if not job.future.done():
                job.future.set_exception(exc)

    def shutdown(self, *, timeout_s: float = 30.0) -> None:
        with self._condition:
            self._shutdown = True
            pending = list(self._pending)
            active = list(self._active)
            self._pending.clear()
            for job in [*pending, *active]:
                job.cancel_event.set()
            self._condition.notify_all()
        for job in pending:
            job.finish_finalize_ownership(finalized=False)
            if not job.future.done():
                job.future.set_exception(
                    RuntimeError("dense mtp_batch service is shut down")
                )
        deadline = time.perf_counter() + max(0.0, float(timeout_s))
        with self._condition:
            while self._active:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    raise RuntimeError(
                        "dense mtp_batch owner did not finalize active requests "
                        "before shutdown"
                    )
                self._condition.wait(timeout=remaining)
