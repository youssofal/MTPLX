"""Dense batched-MTP cohort decode (depth-K) for hybrid GDN+attention targets.

WHY THIS EXISTS
---------------
Dense models (model_type ``qwen3_5``, e.g. Qwen3.8-27B) serve concurrent
requests through a serialized solo-MTP queue or N separate processes; both
re-stream the full quantized weights per stream per verify cycle, so aggregate
decode is bandwidth-walled (~80-89 tok/s on an M3 Ultra at 24k context
regardless of N). This driver runs B streams as ONE ``[B, K+1]`` verify forward
per cycle so the weight read is amortized across the cohort, the dense
counterpart of the A3B ``mtp_batch`` lane, built on the SAME model-agnostic
substrate that lane shipped:

* ``RaggedBatchKVCache``, per-row KV logical lengths (``ragged_kv_cache.py``),
* ``OwnedRecurrentStateCache``, batch-major GDN state (``cache_state.py``),
* ``commit_captured_rows``, per-row ragged commit: per-row KV offset rewrite
  plus per-row selection of captured per-step GDN states
  (``gdn_capture.py``), fed by a states-materializing capture backend
  (``stock`` or ``linear_gdn_from_conv_stream``, NOT the B=1 ``tape``
  backend, which cannot express per-row accept lengths).

SEMANTICS (greedy, fixed cohort)
--------------------------------
Per cycle, per row: commit ``x0`` (the row's pending greedy token) plus its
accepted draft prefix ``d_1..d_k``, identical to the solo MTP loop's cadence
(k accepted + 1 per cycle).

THE CORRECTNESS CONTRACT, and what it is NOT
---------------------------------------------
**What holds: per-row forward independence AT FIXED GEOMETRY.** A row's
committed tokens do not depend on which other prompts share its cohort. Two
cohorts of the same width and shape, differing only in the other rows'
contents, produce byte-identical output for the row they have in common.
CPU-tested against a deterministic fake in ``tests/test_dense_mtp_batch.py``,
and measured on real weights (Qwen3.8-27B, 4 rows, 512 tokens, arm B of
``t204_shape_vs_content.py``: MATCH).

**What does NOT hold: byte-identity across DIFFERENT geometries.** A row
decoded in a cohort is *not* byte-identical to that row decoded alone, and a
row in a width-4 cohort is not byte-identical to the same row in a width-8
one. This is the ordinary non-associativity of floating-point reduction across
matmul shapes: the mathematics is the same, the summation order is not, and a
greedy argmax flips wherever the top two logits are close. Measured on real
weights at IDENTICAL prompt lengths and identical KV capacity, varying only the
row count: all four rows diverged from their solo runs, at tokens 25, 15, 60
and 55 of 512.

**This is not specific to ragged prompts** -- it is equally true of the
uniform-length cohorts this driver was originally benchmarked on -- and it is
the same property every batched LLM server has. Callers who need reproducible
output must pin the batch geometry, not merely the seed.

An earlier version of this docstring claimed byte-identity with a solo run as
"the correctness contract" and cited the bench runner as gating it. Both were
wrong: the claim is false, and that gate pins ``cohort_slots``, so it compares
two runs at IDENTICAL geometry and therefore tests the fixed-geometry property
above rather than the stronger one it was cited for.

One host sync per cycle (the ``[K+2, B]`` decision bundle); drafting is K
sequential ``[B, 1]`` MTP-head calls with a fresh head cache per cycle
(``mtp_position_mode: local``); the verify is ONE ``forward_ar_capture`` whose
captures carry per-step GDN states for the per-row commit.

SCOPE: fixed cohort (membership is sealed for the run; no mid-flight admission
or eviction), fail-loud on any commit refusal, no silent fallback to
autoregressive decode. Greedy and exact speculative sampling are both
supported; mixed prompt lengths are supported through ``ragged_prompts=True``
(per-length-group prefill with pinned per-row offsets).
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from .batched_decode import left_pad_prompts, token_sha
from .cache_state import snapshot_cache, snapshot_untrimmable_cache

__all__ = [
    "DenseBatchStreamResult",
    "DenseBatchResult",
    "generate_dense_mtp_batch",
    "left_pad_prompts",
]


@dataclass
class DenseBatchStreamResult:
    """One REQUEST's result.

    ``index`` is the request ordinal, which equals the prompt's position in
    ``prompts`` for the initial cohort and continues past it for anything
    admitted from ``refill_queue``. ``slot`` is the physical row that served it.
    Without refill the two always coincide, which is why ``slot`` defaults to
    ``index``: a caller written before continuous batching keeps working.
    """

    index: int
    prompt_len: int
    tokens: list[int]
    finish_reason: str
    sha: str
    slot: int = -1

    def __post_init__(self) -> None:
        if self.slot < 0:
            self.slot = self.index


@dataclass
class DenseBatchResult:
    batch_size: int
    depth: int
    streams: list[DenseBatchStreamResult]
    cycles: int
    generated_tokens: int
    accepted_draft_tokens: int
    accepted_by_depth: list[int]
    drafted_by_depth: list[int]
    prefill_s: float
    decode_s: float
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def aggregate_decode_tokps(self) -> float:
        return self.generated_tokens / self.decode_s if self.decode_s > 0 else 0.0

    @property
    def tokens_per_cycle(self) -> float:
        return self.generated_tokens / self.cycles if self.cycles else 0.0

    @property
    def shas(self) -> list[str]:
        return [s.sha for s in self.streams]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def generate_dense_mtp_batch(
    rt: Any,
    prompts: list[list[int]],
    *,
    max_new_tokens: int,
    depth: int = 3,
    stop_token_ids: set[int] | None = None,
    capture_backend: str = "stock",
    cohort_slots: int | None = None,
    pad_id: int = 0,
    head_history: str = "committed",
    history_window: int = 8192,
    history_seed_chunk: int = 2048,
    prefill_chunk: int = 2048,
    loop_mode: str = "pipelined",
    draft_core: str = "eager",
    ragged_attention: bool = False,
    collect_stats: bool = True,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    sampling_seed: int = 0,
    max_new_tokens_per_row: list[int] | None = None,
    on_commit: Any = None,
    ragged_prompts: bool = False,
    refill_queue: list[dict[str, Any]] | None = None,
    is_cancelled: Any = None,
    row_sampling_seeds: list[int] | None = None,
    deadline_s: float | None = None,
    presence_penalty: float | list[float] = 0.0,
    frequency_penalty: float | list[float] = 0.0,
    pull_queued: Any = None,
    max_cohort_rows: int | None = None,
    memory_headroom: float = 0.85,
    on_stats: Any = None,
    session_bank: Any = None,
    session_ids: list[str | None] | None = None,
) -> DenseBatchResult:
    """Depth-``K`` batched MTP decode for a dense hybrid runtime.

    ``temperature``, ``top_k`` and ``top_p`` each accept a scalar applied to
    the whole cohort, or a LIST with one value per prompt so concurrent callers
    with different sampling settings can share a cohort. Mixing greedy and
    sampled rows is allowed: a greedy row is encoded internally as ``top_k=1``
    at temperature 1, which makes the filtered distribution a point mass on the
    argmax, and under that point mass the p/q accept rule reduces to "the draft
    matched", the residual resample reduces to the verify argmax, and the bonus
    sample reduces to the argmax at position K. A cohort in which EVERY row is
    greedy takes the dedicated greedy path instead and stays byte-identical to
    a pre-item-3 run, consuming no randomness. The one behaviour that is not
    bit-identical between the two encodings is tie-breaking: exactly equal top
    logits are broken by ``mx.argmax`` on the greedy path and by the sampler on
    the mixed path.

    ``temperature == 0`` (default) decodes greedily: drafts and verify both
    argmax, acceptance is exact prefix match. A non-zero temperature runs exact
    batched speculative sampling: drafts are sampled from the draft
    distribution ``q`` (after the same temperature/top-k/top-p filtering as
    the verify distribution ``p``), draft token ``i`` is accepted with
    probability ``min(1, p_i(x)/q_i(x))``, a rejection resamples from
    ``normalize(max(p_i - q_i, 0))``, and a fully-accepted block samples the
    bonus token from ``p_{K+1}``, so every committed token is distributed
    exactly as sequential sampling from ``p``. Requires
    ``draft_core='eager'``. ``sampling_seed`` makes runs reproducible; rows
    draw independent randomness per cycle.

    ``prompts`` must share one length unless ``ragged_prompts=True``, which
    admits genuinely mixed lengths: rows are grouped by true length, each group
    is prefilled at that length, and the groups are assembled into one batch
    cache with per-row PINNED offsets. Every row then pays only its own prefill
    and carries only its own KV. Note that this does NOT make a row's output
    identical to that row decoded alone — no batching arrangement does, see the
    correctness contract above — but it does mean a short row is not prefilled
    over pad tokens it never sent. :func:`left_pad_prompts` is the
    older answer and is strictly worse on this trunk: the pad tokens are
    processed by the recurrent GDN layers, so a padded row's recurrent state
    entering its first real token is not the zero state, and no KV offset can
    undo that. Use padding only to reproduce a historical result.
    ``cohort_slots`` pads the cohort with inert dummy rows to a
    fixed shape (the A3B fixed-shape gate discipline): every forward has
    identical ``[cohort_slots, .]`` shapes regardless of the real-stream count,
    so the B=1-equivalence sha gate isolates per-row forward independence.

    ``capture_backend`` must materialize per-step GDN states (``stock``,
    ``linear-gdn``, ``linear-gdn-from-conv``, ``linear-gdn-from-conv-stream``).
    The ``tape`` backend is refused up front: its capture cannot express
    per-row accept lengths and ``commit_captured_rows`` fails closed on it.

    ``max_new_tokens_per_row`` gives each real stream its OWN token cap, for
    servers whose concurrent callers ask for different ``max_tokens``. It must
    have one entry per real prompt and none may exceed ``max_new_tokens``,
    which stays the cohort-wide bound that sizes the cycle guard and the KV
    reservation. ``None`` (the default) applies ``max_new_tokens`` to every
    row, which is the shipped behaviour byte-for-byte.

    ``presence_penalty`` and ``frequency_penalty`` are per row (scalar or a
    list, like ``temperature``) and follow
    ``fast_sampling.apply_penalties_mlx``: ``delta = frequency * count +
    presence * (count > 0)`` subtracted from RAW logits before
    temperature/top-k/top-p, coefficients clamped to [-2, 2], counts covering
    the COMPLETION only. They were previously accepted by the server and
    silently discarded here.

    One documented deviation from solo decoding: counts advance once per cycle,
    so positions drafted within a cycle do not see tokens drafted ahead of them
    in that same cycle. Staleness is bounded by ``depth``. Draft and verify use
    the same counts, so speculative acceptance stays exact against the target
    distribution as penalised -- that target is simply penalised with counts up
    to ``depth`` tokens stale. Removing the lag would mean serialising the draft
    chain, which is the whole speculative gain.

    ``deadline_s`` bounds the run in WALL CLOCK, which is what a serving lane
    actually needs bounded -- the cycle guard below is a runaway backstop, not a
    latency bound. On expiry the loop stops and every unfinished row reports
    ``"deadline"``, kept distinct from ``"cycle_cap"`` and ``"length"`` so an
    operator debugging a timeout does not see it disguised as a caller's own
    token limit.

    ``row_sampling_seeds`` gives each row its own randomness stream, keyed on
    that row's seed, so one caller's seed can no longer steer another's tokens.
    It does NOT make a row reproduce what it would emit solo: a row's logits
    depend on the batch geometry, so identical randomness still gives different
    tokens at different widths. The per-row path costs per-row draws (MLX has no
    batched-key form) and so engages only when two or more DISTINCT seeds appear
    among SAMPLING rows; greedy rows draw nothing, so the usual serving mix
    keeps the single-key path unchanged.

    ``pull_queued`` turns the cohort into a CONTINUOUS one. It is called
    ``pull_queued(capacity)`` at every cycle boundary with the number of rows
    the cohort could take right now, and returns up to that many request
    payloads in the same shape as ``refill_queue`` entries. This is the whole
    difference between continuous batching and the refill list it supersedes:
    ``refill_queue`` is a SNAPSHOT taken when the cohort was sealed, so a
    request arriving one millisecond later waits for the entire cohort to
    drain, whereas a pulled request joins within a cycle of arriving. The
    driver returns when every row is finished AND the queue has nothing left to
    give, so under sustained load one cohort can serve an unbounded number of
    requests; the CALLER decides when to wind a cohort down, by returning an
    empty list. Requests are still reported one stream per request, in the
    order initial-cohort-then-pull-order.

    ``session_bank`` gives a joining request the benefit of work already done
    for a conversation it extends. It is the same
    :class:`mtplx.session_bank.SessionBank` the solo path uses, so a server
    running both lanes warms ONE cache rather than two, and entries written by
    either lane are usable by the other.

    On admission the bank is asked for a stored entry sharing a long head with
    the joiner's prompt. The addressable KV is trimmed to the shared length,
    recurrent state is taken from a stored boundary at or below it, and only the
    remainder is prefilled. The concatenate that adds the row is unchanged,
    which is why this fits at all: admission already builds a standalone cache,
    and a restore is another way of producing one.

    Three things about this lane's use of the bank differ from the solo path's,
    all for the same reason -- recurrent state cannot be rewound:

    * It asks for a recurrent BOUNDARY explicitly, and fails closed if it did
      not get one. The bank otherwise tolerates a small gap, restoring KV to the
      match while recurrent state stays at the stored end. On this trunk a gap
      of four tokens was enough to change the answer.
    * It records boundaries just below a prompt's END, because that is where
      the next turn of a conversation diverges -- at the chat template's
      generation marker, typically a handful of tokens from the end.
    * It CAPS how many boundaries it records. One is 49.1 MB on the 4B, against
      65.1 MB for a full 512-token cache, because GDN state is a fixed-size
      running state.

    Measured on the real 4B, a four-turn conversation: 91% / 91% / 92% of each
    prompt reused, prefill 2.10s -> 0.35s, and the cached answer identical to
    the uncached one token for token.

    ``session_ids`` names the conversation each initial row belongs to, and a
    queued item may carry its own under the same key. It is passed straight to
    the bank, which uses it to bound how much ONE conversation may hold.
    Without it every entry lands in a single global pool, so a long chat storing
    a fresh copy of its whole prompt each turn will evict everyone else's work
    before it evicts its own. Optional, and reuse works without it -- the
    difference is fairness between conversations, not whether reuse happens.

    ``None`` (the default) disables it entirely and every caller is unaffected.

    ``on_stats`` is called with a dict at every admission boundary, on the
    host, so a serving lane can report LIVE what the cohort is doing rather than
    only in the run's final meta. It carries the width, how many pulled requests
    are waiting for a row, and how many rows the memory budget refused. A
    monitor cannot otherwise tell a cohort that is winding down from one that
    has stopped admitting rows it could admit, and those look identical from
    outside while being a normal event and a defect respectively.

    ``memory_headroom`` is the fraction of the device's recommended working
    set that a growing cohort may occupy. Admission stops when adding the next
    row would cross it, so the row waits for capacity instead of taking the
    cohort down with a Metal out-of-memory. Set to ``0`` to disable the check.
    It bounds GROWTH only: a cohort that was SEALED too wide for the machine
    was always able to fail this way, and that is the operator's
    ``--decode-batch-max`` to choose.

    ``max_cohort_rows`` bounds how wide a continuous cohort may grow. Width
    follows demand IN STEPS OF ONE and never pads: three requests run as a
    batch of three. When a row finishes it is removed from the batch axis
    rather than left in place as inert compute, and a joining request is
    prefilled at its OWN length and concatenated on. Mutually exclusive with
    ``cohort_slots``, which means the opposite thing (pad to a fixed shape for
    the parity gate's benefit) and is refused alongside ``pull_queued``.

    One consequence worth stating plainly, because it is a real change and not
    a bug: a caller's row changes batch geometry mid-generation as neighbours
    come and go, and geometry changes floating-point reduction order, so the
    tokens a request receives under continuous batching are not the tokens it
    would have received at a pinned width. Content of the neighbours does not
    affect it; the NUMBER of them does. Measured on real weights, see the
    correctness contract above. Every batched server has this property.

    ``is_cancelled`` is called ``is_cancelled(request)`` on the host at each
    cycle boundary for every live row, and returning True EVICTS that row: it
    stops committing, its request finishes as ``"cancelled"``, and its slot
    becomes available to the refill queue immediately rather than at the end of
    that request's ``max_tokens``. Without it an abandoned request holds a slot
    for its full token budget, which under sustained load with real client
    disconnects silently drains the lane's capacity.

    ``on_commit`` is called ``on_commit(request, token)`` on the host, once per
    token, at the moment that token is committed. ``request`` indexes
    ``prompts`` for the initial cohort and continues past it for anything
    admitted from ``refill_queue``; without a queue it equals the row index — the same cadence
    the A3B serving lane's ``on_token`` uses. Committed means kept: the
    pipelined loop's bounded overshoot on a finished row is filtered out
    before the callback, so a caller can forward tokens to a client without
    re-checking them. Exceptions from the callback propagate and abort the
    cohort, so a caller that must not fail the run should swallow its own.

    ``head_history`` selects the MTP-head draft conditioning (the solo loop's
    ``mtp_history_policy``): ``"committed"`` (default, the product default)
    keeps ONE persistent head cache across cycles, seeded from the last
    ``history_window`` prompt positions (hidden at position i paired with
    token i+1, the solo seeding recipe) and, per cycle, rewound to the
    committed base and re-appended from the verify's trunk hiddens with
    per-row committed lengths (the same ragged offset arithmetic as the trunk
    KV). ``"cycle"`` uses a fresh empty head cache every cycle (cheaper,
    measurably worse acceptance at long context). Rope positions are
    head-cache-local throughout, a constant shift versus the solo seeding
    base, which is relationally identical under rope.
    """
    import json
    import os
    import mlx.core as mx

    from .attention_context import attention_phase
    from .batched_decode import to_foldin_cache
    from .gdn_capture import commit_captured_rows, resolve_gdn_capture_backend
    from .ragged_kv_cache import RaggedBatchKVCache

    depth = int(depth)
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if not prompts:
        raise ValueError("prompts must be non-empty")
    if not getattr(rt, "mtp_enabled", False):
        raise RuntimeError("generate_dense_mtp_batch requires an MTP-enabled runtime")
    if pull_queued is not None and cohort_slots:
        # They mean opposite things. ``cohort_slots`` pads to a fixed shape so
        # the parity gate can hold geometry constant; continuous batching moves
        # geometry deliberately. Accepting both would quietly serve one of them.
        raise ValueError(
            "cohort_slots pads to a fixed shape and pull_queued moves width "
            "with demand; they cannot both be set"
        )
    resolved_backend = resolve_gdn_capture_backend(capture_backend)
    if resolved_backend in {"linear_gdn_from_conv_tape", "linear_gdn_final"}:
        raise ValueError(
            f"capture backend {capture_backend!r} does not materialize per-step "
            "states; per-row commit needs 'stock', 'linear-gdn', "
            "'linear-gdn-from-conv', or 'linear-gdn-from-conv-stream'"
        )

    head_history = str(head_history).strip().lower()
    if head_history not in {"cycle", "committed"}:
        raise ValueError("head_history must be 'cycle' or 'committed'")
    loop_mode = str(loop_mode).strip().lower()
    if loop_mode not in {"pipelined", "serial"}:
        raise ValueError("loop_mode must be 'pipelined' or 'serial'")
    draft_core = str(draft_core).strip().lower()
    if draft_core not in {"eager", "compiled"}:
        raise ValueError("draft_core must be 'eager' or 'compiled'")
    if draft_core == "compiled" and head_history != "committed":
        raise ValueError("draft_core='compiled' requires head_history='committed'")
    # --- per-row sampling parameters (item 3) --------------------------------
    # Each of temperature / top_k / top_p accepts either a scalar (applied to
    # every row, the shipped behaviour) or one value per REAL prompt, so a
    # server can put callers with different sampling settings in one cohort.
    def _per_row(value: Any, name: str, cast, floor=None) -> list[Any]:
        if isinstance(value, (list, tuple)):
            values = [cast(v) for v in value]
            if len(values) != len(prompts):
                raise ValueError(
                    f"{name} must be a scalar or one value per prompt "
                    f"({len(values)} values for {len(prompts)} prompts)"
                )
        else:
            values = [cast(value)] * len(prompts)
        if floor is not None and any(v < floor for v in values):
            raise ValueError(f"every {name} must be >= {floor}")
        return values

    row_temperature = _per_row(temperature, "temperature", float, floor=0.0)
    row_top_k = _per_row(top_k, "top_k", int)
    row_top_p = _per_row(top_p, "top_p", float)
    # Greedy is the exact special case of sampling at top_k=1: the filtered
    # distribution is a point mass on the argmax, the p/q accept rule reduces to
    # "the draft matched", the residual resample reduces to the verify argmax,
    # and the bonus sample reduces to the argmax at position K. That identity is
    # what lets one cohort mix greedy callers with sampling callers instead of
    # having to split them. A cohort where EVERY row is greedy still takes the
    # dedicated greedy path below, so it stays byte-identical to today and
    # consumes no randomness at all.
    sampling = any(t > 0 for t in row_temperature)
    if refill_queue:
        # A queued request that wants sampling cannot join an all-greedy
        # cohort: that cohort runs the dedicated greedy path, which consumes no
        # randomness and has no per-row sampling machinery to update. Caught
        # here, before any work, rather than failing a cohort mid-decode.
        hot = [
            index
            for index, item in enumerate(refill_queue)
            if float(item.get("temperature", 0.0) or 0.0) > 0.0
        ]
        if hot and not sampling:
            # A FROZEN refill list is known in full before the run starts, so
            # the cohort can simply be built on the sampling path from the
            # outset -- no need to refuse it, and no need to upgrade later.
            sampling = True
        if hot and draft_core != "eager":
            raise ValueError("temperature > 0 requires draft_core='eager'")
    if sampling and draft_core != "eager":
        raise ValueError("temperature > 0 requires draft_core='eager'")
    if head_history == "committed" and not hasattr(rt, "update_mtp_cache"):
        raise RuntimeError(
            "head_history='committed' requires a runtime with update_mtp_cache"
        )

    n_real = len(prompts)
    if max_new_tokens_per_row is None:
        row_caps = [int(max_new_tokens)] * n_real
    else:
        row_caps = [int(v) for v in max_new_tokens_per_row]
        if len(row_caps) != n_real:
            raise ValueError(
                "max_new_tokens_per_row must have one entry per prompt "
                f"({len(row_caps)} caps for {n_real} prompts)"
            )
        if any(cap < 1 for cap in row_caps):
            raise ValueError("every max_new_tokens_per_row entry must be >= 1")
        if max(row_caps) > int(max_new_tokens):
            raise ValueError(
                "max_new_tokens_per_row entries must not exceed max_new_tokens "
                f"({max(row_caps)} > {int(max_new_tokens)})"
            )
    prompt_lens = [len(p) for p in prompts]
    prompt_len = max(prompt_lens)
    if min(prompt_lens) < 1:
        raise ValueError("each prompt needs at least one token")
    if not ragged_prompts and len(set(prompt_lens)) > 1:
        raise ValueError(
            "all prompts must share a length unless ragged_prompts=True; "
            "left_pad_prompts() equalizes a ragged batch, but padding is a "
            "worse answer than ragged_prompts=True on this trunk (the pads "
            "run through the recurrent GDN layers and no offset undoes that)"
        )

    slots: list[list[int]] = [[int(t) for t in p] for p in prompts]
    if cohort_slots is not None:
        cohort_slots = int(cohort_slots)
        if cohort_slots < n_real:
            raise ValueError(
                f"cohort_slots ({cohort_slots}) must be >= the real prompt "
                f"count ({n_real})"
            )
        dummy = [int(pad_id)] * (1 if ragged_prompts else prompt_len)
        slots.extend([list(dummy) for _ in range(cohort_slots - n_real)])
    batch = len(slots)
    verified = depth + 1  # positions written per verify forward
    stop = {int(t) for t in (stop_token_ids or set())}
    hidden_variant = getattr(getattr(rt, "contract", None), "hidden_variant", None)

    started_all = time.perf_counter()

    if ragged_attention:
        from .attention_split import configure_ragged_2pass_attention

        n_cfg = configure_ragged_2pass_attention(rt.model, enabled=True)
        _require(n_cfg > 0, "ragged_attention: no full-attention layers found")

    # --- prefill: per-LENGTH-GROUP chunked forwards on the stock scalar lane. -
    # Chunking within a group bounds the transient: one [B, 24k] forward
    # materializes q x k attention scores when the fused SDPA path is
    # unavailable for large-q prefill (observed: a 218 GB Metal allocation at
    # B=8 x 24k, over the buffer cap). ``logits_keep=1`` keeps prefill logits at
    # one position (a full [B, 24k, vocab] tensor is tens of GB). The hidden
    # sequence is kept ONLY for the trailing history window (the head-history
    # seed); earlier chunks' hiddens are dropped as we go.
    #
    # RAGGED PROMPTS (item 2). Rows are grouped by TRUE length and each group is
    # prefilled on its own, then the groups are assembled into one batch cache
    # with per-row pinned offsets. Why grouping rather than padding one batch:
    #
    # * Left-padding is not merely wasteful, it is WRONG for this trunk. The
    #   pads are processed by the GDN layers, whose state is recurrent, so a
    #   padded row's recurrent state entering its real first token is not the
    #   zero state. No KV offset can undo that, because the contamination is in
    #   the recurrent state and not in the KV. Rope positions are shifted by the
    #   pad width too. Right-padding fixes rope and (being causal) fixes
    #   attention, but moves the same recurrent problem to the tail.
    # * A group prefilled at its own true length sees exactly the tokens its
    #   caller sent, with no pad prefix folded into the recurrent state. That
    #   is a semantic property, not a bitwise one: it does NOT make the row's
    #   output identical to a solo run, because batch geometry changes float
    #   reduction order regardless. Padding corrupts what the row conditions
    #   on; grouping does not.
    # * Each row now pays only its OWN prefill and carries only its OWN KV.
    #
    # A cohort whose prompts already share a length forms exactly ONE group and
    # takes the identical path it took before this change, so uniform cohorts
    # are unaffected byte for byte.
    prefill_chunk = max(1, int(prefill_chunk))
    memlog = os.environ.get("MTPLX_DENSE_BATCH_MEMLOG", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # Row-axis rebuild trace. Off by default and worth having on a switch: the
    # failure mode of a resize bug is a shape error several frames away from
    # the resize that caused it, and the row set at the boundary is the one
    # fact that names it immediately.
    resize_debug = os.environ.get("MTPLX_DENSE_BATCH_RESIZE_DEBUG", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # Cycle-duration outlier trace. Seconds; 0 disables. A cycle longer than
    # this records one line naming what the cohort was doing at the time.
    #
    # Deliberately NOT phase timing. `MTPLX_DENSE_BATCH_PHASE_TIMING` inserts an
    # `mx.eval` per phase, so leaving it on changes the thing being measured --
    # which is why the 110-144 second pauses observed in an earlier session were
    # never caught in the act. This costs one `perf_counter()` and a comparison
    # per cycle, adds no device sync, and can therefore be left on for a soak.
    try:
        cycle_warn_s = float(
            os.environ.get("MTPLX_DENSE_BATCH_CYCLE_WARN_S", "0") or 0
        )
    except ValueError:
        cycle_warn_s = 0.0
    cycle_trace_path = os.environ.get("MTPLX_DENSE_BATCH_CYCLE_TRACE", "")
    slow_cycles: list[dict[str, Any]] = []
    started = time.perf_counter()

    def _env_int(name: str, default: int) -> int:
        """Read an integer environment variable, falling back on anything odd."""

        try:
            raw = os.environ.get(name)
            return default if raw is None or raw == "" else int(raw)
        except Exception:
            return default

    def _dbg_prefix(msg: str) -> None:
        """Say what the prefix lookup decided, under the same debug switch.

        `_debug_exc` prints only when a path THROWS. A lookup that simply
        returns nothing is silent and indistinguishable from one that was never
        called, which is exactly the state that made a zero reuse rate
        un-diagnosable from the outside.
        """

        if os.environ.get("MTPLX_DEBUG_PREFIX_DIVERGENCE"):
            print(f"[mtplx] {msg}", flush=True)

    def _debug_exc(where: str) -> None:
        """Print why a fail-closed cache path bailed, when debugging is on.

        Every cache path here swallows exceptions on purpose: a restore or a
        store is an optimisation and must never be the reason a request fails.
        The cost of that discipline is that a zero counter looks identical
        whether the feature is working, mis-wired, or throwing on every call.
        This makes the difference askable without changing the behaviour, under
        the same environment variable `session_bank.py` already reads.
        """

        if not os.environ.get("MTPLX_DEBUG_PREFIX_DIVERGENCE"):
            return
        import traceback

        print(f"[mtplx] dense {where} failed:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()

    def _hist_is_written(hist: Any) -> bool:
        """Does this head cache carry real buffers, or is it an empty shell?

        The distinction matters because merging an empty one fails SILENTLY.
        """

        if not hist:
            return False
        for item in hist:
            if isinstance(item, RaggedBatchKVCache) and item.keys is None:
                return False
        return True

    def _slice_row(cache_list: list[Any], row: int) -> list[Any] | None:
        """One row's caches, as standalone objects sharing nothing with the batch.

        Every array is produced by a gather, which allocates, so the result
        survives the batch it came from being resized or freed. That is the
        whole point: the row is about to be dropped.
        """

        from .cache_state import OwnedRecurrentStateCache

        idx = mx.array([int(row)], dtype=mx.int32)
        out: list[Any] = []
        try:
            for entry in cache_list:
                if isinstance(entry, RaggedBatchKVCache):
                    if entry.keys is None:
                        return None
                    clone = RaggedBatchKVCache(
                        step=int(entry.step),
                        keys=entry.keys[idx],
                        values=None if entry.values is None else entry.values[idx],
                        offsets=entry.offsets[idx],
                    )
                    clone._capacity_bound = entry._capacity_bound
                    out.append(clone)
                elif isinstance(entry, OwnedRecurrentStateCache):
                    out.append(
                        OwnedRecurrentStateCache(
                            size=len(entry.state),
                            initial=[
                                None if leaf is None else leaf[idx]
                                for leaf in entry.state
                            ],
                            left_padding=entry.left_padding,
                            lengths=entry.lengths,
                        )
                    )
                else:
                    return None
            mx.eval(*[e.keys for e in out if isinstance(e, RaggedBatchKVCache)])
        except Exception:
            # Extraction is an optimisation. It must never be the reason a
            # request fails, so anything unexpected simply means no entry.
            return None
        return out

    def _request_prompt(request: int) -> list[int]:
        if request < n_real:
            return [int(t) for t in slots[request]]
        return [int(t) for t in queued[request - n_real]["prompt"]]

    def _prefill_from_bank(tokens: list[int]) -> dict[str, Any] | None:
        """Build a row's standalone cache from the SessionBank.

        The bank does the hard parts: longest-prefix lookup, trimming the KV to
        the matched length, and picking the recurrent boundary at or below it.
        What is left here is what this lane adds -- convert the returned scalar
        cache to the ragged lane, reserve, and prefill the suffix.
        """

        if session_bank is None:
            _dbg_prefix("prefill_from_bank: NO BANK")
            return None
        _dbg_prefix(f"prefill_from_bank: called, prompt={len(tokens)}")
        mtp_factory = (
            rt.make_mtp_cache
            if head_history == "committed" and hasattr(rt, "make_mtp_cache")
            else None
        )
        cache_g = hist_g = None
        covered = 0
        entry = None

        # 1. Exact prefix. Cheapest path and the only one that needs no replay,
        #    but it hits only when a stored entry is a STRICT prefix of this
        #    prompt -- which for chat traffic is the rare case, not the common
        #    one, because every prompt ends in its own template suffix.
        try:
            restored = session_bank.restore(
                rt,
                tokens,
                mode="clone",
                mtp_history_policy=head_history,
                cache_factory=rt.make_cache,
                mtp_cache_factory=mtp_factory,
            )
        except Exception:
            _debug_exc("bank-restore")
            restored = None
        _dbg_prefix(
            f"  path1 exact-restore: {'HIT' if restored is not None else 'miss'}"
        )
        if restored is not None:
            covered = int(getattr(restored.entry, "prefix_len", 0) or 0)
            cache_g = restored.cache
            hist_g = restored.mtp_history_cache
            _dbg_prefix(f"  path1 covered={covered}")

        # 2. Divergence-tolerant. This is the path that actually carries chat
        #    and agent traffic: the stored turn and the new prompt share a long
        #    head and then diverge, so the bank rewinds to the last block-
        #    aligned boundary at or below the shared length and this lane
        #    prefills the rest. Turn one's prompt never needs to be a prefix of
        #    turn two's -- only long enough to have crossed a boundary before
        #    they part. Same two-call shape the solo path uses, so a hit here
        #    and a hit there mean the same thing.
        if cache_g is None:
            try:
                candidates = session_bank.near_prefix_candidates(
                    tokens,
                    max_token_gap=bank_max_gap,
                    min_matched_tokens=bank_min_match,
                    block_size=bank_block,
                    block_min_matched_tokens=bank_block_min_match,
                    allow_block_prefix=True,
                    model_path=str(rt.model_path),
                    mtp_enabled=bool(getattr(rt, "mtp_enabled", False)),
                    mtp_history_policy=head_history,
                )
            except Exception:
                _debug_exc("bank-near-prefix")
                candidates = []
            _dbg_prefix(
                f"near-prefix lookup: prompt={len(tokens)} "
                f"candidates={len(candidates or [])} gap={bank_max_gap} "
                f"min_match={bank_min_match} block={bank_block}"
            )
            for entry, matched in candidates or []:  # noqa: B020 - kept after the loop
                matched = int(matched)
                if matched < 2 or matched >= len(tokens):
                    _dbg_prefix(
                        f"  candidate dropped: matched={matched} "
                        f"prompt={len(tokens)} (needs 2 <= matched < prompt)"
                    )
                    continue

                # Ask for the recurrent boundary EXPLICITLY rather than for the
                # match point.
                #
                # Asked for the match point, the bank restores there whenever
                # the leftover gap is small, trimming the addressable KV while
                # the recurrent state stays where it was stored. Those few
                # tokens cannot be rewound, so the model keeps conditioning on
                # them. Naming the boundary removes the judgement call.
                if getattr(entry, "has_recurrent", False):
                    # Ask BELOW the bank's own tolerance window, not merely at
                    # or below the match.
                    #
                    # The bank only performs a true boundary restore when the
                    # requested point is far enough from the stored end;
                    # closer than that it keeps the long-shipped tolerance and
                    # leaves recurrent state alone. A boundary sitting inside
                    # that window therefore gets a NON-boundary restore, which
                    # the guard below then rejects -- correctly, but the whole
                    # restore is lost. Measured: tail guards at 8 tokens landed
                    # exactly on the limit and took reuse from 76% to zero.
                    ceiling = min(matched, int(entry.prefix_len) - _tiny_gap - 1)
                    boundary = entry.recurrent_boundary_at_or_below(ceiling)
                    _dbg_prefix(
                        f"  candidate: matched={matched} entry_len={entry.prefix_len} "
                        f"ceiling={ceiling} boundaries="
                        f"{[b[0] if isinstance(b, tuple) else b for b in (getattr(entry, 'gdn_boundaries', None) or [])]} "
                        f"-> boundary={boundary[0] if boundary else None}"
                    )
                    if boundary is None:
                        continue
                    matched = int(boundary[0])
                    if matched < 2:
                        continue
                served: dict[str, Any] = {}
                try:
                    # `clone` only, never the zero-copy `reference` lease the
                    # solo path prefers: this row's cache is about to be
                    # converted in place by `to_foldin_cache` and then written
                    # by decode, and aliasing the bank's live buffers would
                    # corrupt the stored entry for every later reader.
                    got = session_bank.restore_entry_prefix_cache(
                        rt,
                        entry,
                        matched,
                        mode="clone",
                        cache_factory=rt.make_cache,
                        mtp_cache_factory=mtp_factory,
                        served_out=served,
                    )
                except Exception:
                    _debug_exc("bank-prefix-restore")
                    got = None
                if got is None:
                    continue
                cache_g, hist_g = got[0], got[1]

                # How many tokens the returned cache ACTUALLY holds, which is
                # not always the `matched` that was asked for, and the two
                # answers differ by different amounts:
                #
                # * The bank may rewind further than requested. On a recurrent
                #   trunk a sub-prefix restore has to land where recurrent
                #   state was captured, not merely where the KV can trim, so it
                #   drops to the nearest stored boundary at or below `matched`.
                # * A NON-boundary restore deliberately leaves the cache one
                #   token short, reserving a slot for the last matched token to
                #   be re-forwarded. A boundary restore does not.
                #
                # Reading `matched` and ignoring both is how this lane produced
                # a restore that was 88% faster and quietly wrong: the model
                # never saw token `matched - 1`, and the output diverged a few
                # tokens later with nothing reporting an error.
                point = int(served.get("restore_point", matched))
                boundary_used = bool(served.get("boundary_used"))

                # Fail closed if the recurrent state did NOT come from a stored
                # boundary. Asking for one is not the same as getting one, and
                # the difference is invisible in the output: the answer is
                # merely wrong, a few tokens in, with every counter reporting
                # success. Checked rather than assumed because the assumption
                # is exactly what failed here twice.
                if (
                    getattr(entry, "has_recurrent", False)
                    and not boundary_used
                    and point < int(entry.prefix_len)
                ):
                    prefix_restore_failures[0] += 1
                    cache_g = hist_g = None
                    continue

                covered = point if boundary_used else max(0, point - 1)

                # The restored state must correspond to THIS prompt's first
                # `covered` tokens. Checked directly against the entry's own
                # token list rather than trusted, because everything upstream
                # that could get it wrong -- which candidate was picked, how far
                # the boundary rewound, an off-by-one in either -- fails the
                # same silent way: a confident answer conditioned on somebody
                # else's text. Entries share long prefixes by design (every
                # conversation on a server carries the same system prompt), so
                # a near-match against the WRONG conversation is the expected
                # shape of this failure, not an exotic one.
                stored = getattr(entry, "token_ids", None) or ()
                if (
                    covered > len(stored)
                    or [int(t) for t in stored[:covered]] != tokens[:covered]
                ):
                    if os.environ.get("MTPLX_DEBUG_PREFIX_DIVERGENCE"):
                        print(
                            f"[mtplx] prefix-mismatch: covered={covered} "
                            f"entry_len={len(stored)} matched={matched} "
                            f"boundary_used={boundary_used}",
                            file=sys.stderr,
                            flush=True,
                        )
                    prefix_restore_failures[0] += 1
                    cache_g = hist_g = None
                    continue
                break

        if cache_g is None:
            return None
        suffix = [int(t) for t in tokens[covered:]]
        if covered <= 0 or not suffix:
            # Nothing left to run a forward over. The row needs one position to
            # produce its first logits, and a full-length match wants
            # truncate-and-replay, which is a different mechanism.
            return None
        if head_history == "committed" and not _hist_is_written(hist_g):
            return None

        # Where to pause the suffix prefill and photograph the recurrent state:
        # every absolute multiple of the block size still ahead of us.
        #
        # This is what makes reuse GROW with a conversation instead of stalling.
        # A restored row that stores nothing new leaves the bank knowing only
        # the boundaries of this conversation's first cold prompt, so turn ten
        # would still rewind to turn one's last chunk and replay everything
        # since. Each restored turn extends the ladder, so the next turn starts
        # from a rung near where it diverges.
        segment_ends = _boundary_ends(covered, len(tokens))

        new_boundaries: list[Any] = []
        try:
            # Forward on the STOCK cache and convert to the ragged lane
            # afterwards, the same order `_prefill_group` uses. The row-slicing
            # a boundary snapshot needs only works before the conversion.
            hidden_parts: list[Any] = []
            logits = None
            with attention_phase("prefill"):
                _start = covered
                for _end in segment_ends:
                    _seg = tokens[_start:_end]
                    if not _seg:
                        continue
                    logits, _h = rt.forward_ar(
                        mx.array([_seg]),
                        cache=cache_g,
                        return_hidden=True,
                        logits_keep=1,
                    )
                    mx.eval(logits, _h)
                    hidden_parts.append(_h)
                    _start = _end
                    if _end < len(tokens):
                        _row0 = _slice_stock_row(cache_g, 0)
                        if _row0 is not None:
                            new_boundaries.append(
                                (_end, snapshot_untrimmable_cache(_row0), None)
                            )
            if logits is None or not hidden_parts:
                return None
            hidden_c = (
                hidden_parts[0]
                if len(hidden_parts) == 1
                else mx.concatenate(hidden_parts, axis=1)
            )

            # Extend the draft head's history over the replayed suffix. The
            # restored history stops at the restore point; leaving it there
            # would not corrupt the answer -- a rejected draft is simply
            # corrected -- but it would make every draft after a restore worse,
            # which is a speed cache making speed worse.
            if head_history == "committed" and hist_g is not None:
                _tail = int(hidden_c.shape[1])
                _keep = min(_tail - 1, max(0, int(history_window)))
                if _keep > 0:
                    _seed_hidden = hidden_c[:, _tail - 1 - _keep : _tail - 1, :]
                    _seed_tokens = mx.array([tokens[len(tokens) - _keep :]])
                    _settled = rt.update_mtp_cache(
                        _seed_hidden, _seed_tokens, mtp_cache=hist_g
                    )
                    if _settled is not None:
                        mx.eval(_settled)

            # Bank the completed prompt BEFORE the ragged conversion, carrying
            # the donor's boundaries forward -- they describe token prefixes, so
            # they stay true for this longer prompt verbatim.
            # Carry the donor's boundaries forward, but KEEP ONLY THE HIGHEST
            # FEW, and that cap is load-bearing rather than tidiness.
            #
            # These are shared references, so passing them costs no copy -- but
            # it does keep the referent ALIVE. The donor's list already contains
            # its own donor's, so an uncapped chain means every snapshot ever
            # taken for a conversation stays reachable from its newest entry.
            # The bank's byte accounting does not see them, so eviction never
            # frees them either: the bank reports a flat 1.7 GB while real
            # memory climbs by one snapshot per turn.
            #
            # Measured, 2026-08-24: with the cache ON, MLX active memory grew
            # 4.19 -> 8.58 GB over 1360 requests while the bank's own figure sat
            # flat; with the cache OFF over comparable traffic, MLX peak held at
            # 4.298 GB and RSS DECLINED. Same lane, same load, cache the only
            # difference.
            #
            # Highest positions, not lowest: reuse wants the boundary nearest
            # below where the next turn diverges, and that point moves forward
            # as the conversation grows. The low ones are the least useful AND
            # the ones a long chain accumulates most of.
            _inherited = [
                _b for _b in (getattr(entry, "gdn_boundaries", None) or [])
                if int(_b[0]) <= covered
            ]
            _carry = sorted(
                _inherited + new_boundaries, key=lambda _b: int(_b[0])
            )[-bank_max_boundaries:]
            _bank_put_row(
                tokens,
                0,
                cache_g,
                hist_g,
                logits,
                hidden_c[:, -1:, :],
                boundaries=_carry,
                session_id=_session_for(tokens),
            )

            to_foldin_cache(cache_g, 1)
            if hist_g is not None:
                to_foldin_cache(hist_g, 1)
        except Exception:
            # Fail closed. A restore is an optimisation and must never be why a
            # request fails; the caller falls back to a normal prefill.
            _debug_exc("bank-restore-forward")
            prefix_restore_failures[0] += 1
            return None
        prefix_restores[0] += 1
        prefix_tokens_skipped[0] += covered
        return {
            "covered": int(covered),
            "rows": [0],
            "cache": cache_g,
            "mtp_hist": hist_g,
            "logits": logits[:, -1, :],
            "hidden": hidden_c[:, -1:, :],
            "keep": int(keep_hidden) if "keep_hidden" in dir() else 0,
            "length": len(tokens),
        }

    def _reuse_prefix(tokens: list[int]) -> dict[str, Any] | None:
        """Produce a row's standalone cache from banked work, or nothing."""

        if session_bank is None:
            return None
        return _prefill_from_bank(tokens)

    prefix_covered_by_row: dict[int, int] = {}
    prefix_restores = [0]
    prefix_tokens_skipped = [0]
    bank_entries_written = [0]
    # Restores that were attempted and could not complete. Non-zero means stored
    # entries are not usable by this runtime -- the lane still serves every
    # request correctly, but the cache is costing memory and returning nothing,
    # which is exactly the state that must not be silent.
    prefix_restore_failures = [0]
    # Read the SAME environment variables the solo path reads, so an operator
    # tunes reuse once for the server rather than once per lane. The defaults
    # are the solo path's defaults for the same reason.
    #
    # `bank_block_min_match` is the knob that decides whether ordinary chat
    # benefits at all: a conversation must share at least this many leading
    # tokens with a stored turn before a block restore is allowed. At the
    # default 512 a short first turn stores nothing reusable; lowering it
    # widens reuse and costs more stored boundaries.
    # ZERO, deliberately, and NOT the solo path's default of 8.
    #
    # The bank tolerates a small gap between the match point and the end of the
    # stored entry, restoring the addressable KV to the match while leaving the
    # recurrent state where it was stored. On a pure-attention model that is a
    # harmless tokenizer-drift allowance. On this trunk it is not: recurrent
    # state cannot be rewound, so those few tokens stay folded in and the model
    # conditions on text the prompt does not contain.
    #
    # Measured, 2026-08-24, real 4B, one two-turn conversation: a gap of FOUR
    # tokens left cached and uncached answers agreeing for 15 of 24 tokens and
    # then diverging. No error, no warning -- a fast wrong answer. Forcing the
    # gap to zero pushes every restore onto a stored recurrent boundary, which
    # is exact by construction.
    #
    # The cost is real and worth naming: reuse drops to the last checkpoint at
    # or below the match instead of the match itself, so more tokens get
    # replayed. That is the correct trade -- the whole value of a cache is that
    # its answer is the same answer.
    bank_max_gap = max(0, _env_int("MTPLX_DENSE_NEAR_PREFIX_MAX_TOKEN_GAP", 0))
    bank_min_match = max(1, _env_int("MTPLX_SESSION_NEAR_PREFIX_MIN_MATCH_TOKENS", 64))
    bank_block = max(1, _env_int("MTPLX_SESSION_PREFIX_BLOCK_SIZE", 256))
    # A recurrent boundary is NOT cheap, which is the opposite of what this
    # code assumed until it was measured. On the real 4B one snapshot is
    # 49.1 MB -- 75% of a full 512-token cache -- because GDN state is a
    # fixed-size running state whose size does not depend on how many tokens
    # went into it. On the 27B, with twice the layers and wider heads, expect
    # several times that. So the count is CAPPED rather than left to scale with
    # prompt length, and the cap is low.
    #
    # The cap is on how many are CAPTURED per prefill, which is what costs
    # memory. A stored entry may LIST more than this: boundaries inherited from
    # a shorter entry are shared references to snapshots that already exist, so
    # they cost nothing additional and discarding them would throw away reuse
    # for no gain. The bound that actually holds memory down is the bank's byte
    # budget, and that IS enforced -- measured at 2.00 GB held against a 2.00 GB
    # budget over 40 turns of four concurrent conversations.
    #
    # Stated carefully because an earlier version of this comment said "capped
    # per entry", a stress test checked exactly that, and it failed.
    bank_max_boundaries = max(1, _env_int("MTPLX_DENSE_PREFIX_MAX_BOUNDARIES", 4))
    bank_ladder_stride = max(
        bank_block, _env_int("MTPLX_DENSE_PREFIX_LADDER_STRIDE", 1024)
    )
    # Distances below a prompt's end to record an extra boundary at. See
    # `_boundary_ends` for why these carry the chat case almost single-handedly.
    try:
        from .session_bank import _near_prefix_tiny_gap_limit

        _tiny_gap = int(_near_prefix_tiny_gap_limit())
    except Exception:
        # Matches the bank's shipped default. Read rather than copied above so
        # the two cannot drift apart silently; this is only the fallback.
        _tiny_gap = 8
    bank_tail_guards = tuple(
        sorted(
            {
                # Each must exceed `_tiny_gap`, or the bank answers with a
                # non-boundary restore that this lane cannot accept.
                max(_tiny_gap + 1, _env_int("MTPLX_DENSE_PREFIX_TAIL_GUARD_NEAR", 16)),
                max(_tiny_gap + 1, _env_int("MTPLX_DENSE_PREFIX_TAIL_GUARD_MID", 64)),
            }
        )
    )
    bank_block_min_match = max(
        bank_block, _env_int("MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS", 512)
    )
    #: Recurrent-only snapshots captured during the CURRENT group's prefill,
    #: keyed by row index within that group. Lives at driver scope because the
    #: helpers below close over it; cleared per group rather than rebound, so
    #: those closures keep seeing the same object.
    _bank_boundaries: dict[int, list[Any]] = {}

    #: Conversation id per row, looked up by the row's prompt rather than
    #: threaded through every call path -- a row reaches the two store sites by
    #: different routes (sealed at the start, or pulled from the queue later)
    #: and both already hold the token list. Two conversations sending an
    #: identical prompt collide here, which is harmless: the id is used only for
    #: the bank's per-conversation accounting, and either answer is correct.
    _session_by_prompt: dict[tuple[int, ...], str] = {}
    for _i, _p in enumerate(prompts):
        _sid = None
        if session_ids is not None and _i < len(session_ids):
            _sid = session_ids[_i]
        if _sid:
            _session_by_prompt.setdefault(tuple(int(t) for t in _p), str(_sid))
    for _item in (refill_queue or []):
        _sid = _item.get("session_id")
        if _sid:
            _session_by_prompt.setdefault(
                tuple(int(t) for t in _item.get("prompt") or []), str(_sid)
            )

    def _session_for(tokens: Any) -> str | None:
        return _session_by_prompt.get(tuple(int(t) for t in tokens))

    def _boundary_ends(begin: int, total: int) -> list[int]:
        """Positions to pause a prefill at so recurrent state can be recorded.

        Two families, for two different reasons.

        **Block multiples** give a coarse ladder that lets any later prompt
        rewind to somewhere reasonable.

        **Tail guards** sit a short distance below the prompt's END, and they
        are the ones that matter for chat. Consecutive turns of a conversation
        do not diverge in the middle -- they diverge a handful of tokens before
        the end, where the chat template's generation marker begins and the next
        turn instead continues with the assistant's reply. Measured on the real
        4B: turn one's 626-token prompt shares 622 tokens with turn two's, so
        the divergence is FOUR tokens from the end. A ladder built only from
        256-multiples has to rewind to 512 there and replay everything after it;
        a boundary at 618 replays almost nothing.

        Several guards rather than one because the distance depends on the
        template -- a thinking block makes the suffix longer -- and the bank
        picks the highest usable one on its own. They are recurrent-only
        snapshots, so a spare that never gets used is cheap.
        """

        # Tail guards FIRST, because they are worth far more per megabyte.
        # Measured on the real 4B: guards alone hold reuse at ~91% across a
        # growing conversation; the block ladder alone decayed 76% -> 66% over
        # the same four turns while costing more as prompts get longer.
        chosen: list[int] = []
        for guard in bank_tail_guards:
            at = total - int(guard)
            if begin < at < total:
                chosen.append(at)

        # Then a SPARSE ladder, for prompts that diverge early rather than at
        # the end -- a shared document with a different question in the middle.
        # Spaced by `bank_ladder_stride`, not by the block size, and capped.
        pos = begin
        while len(chosen) < bank_max_boundaries:
            nxt = ((pos // bank_ladder_stride) + 1) * bank_ladder_stride
            if nxt >= total:
                break
            chosen.append(nxt)
            pos = nxt

        return sorted(set(chosen))[:bank_max_boundaries] + [total]

    def _bank_boundary(rows: list[list[int]], end: int, cache_g: Any) -> None:
        """Record a recurrent-only snapshot per row at `end` tokens.

        Recurrent-only on purpose, and it is why the bank's entries are small:
        the KV at this position is already a prefix of the KV at the end of the
        prompt, so it does not need storing twice. Only the recurrent state has
        to be captured here, because it cannot be rewound later.
        """

        if session_bank is None or end < 1:
            return
        for local in range(len(rows)):
            sliced = _slice_stock_row(cache_g, local)
            if sliced is None:
                return
            try:
                _bank_boundaries.setdefault(local, []).append(
                    (int(end), snapshot_untrimmable_cache(sliced), None)
                )
            except Exception:
                # Boundary capture is an accelerator for FUTURE restores and
                # must never break the prefill running right now. Same rule the
                # solo path's `_capture_gdn_boundary` follows.
                _debug_exc("bank-boundary")
                return

    def _bank_put_row(
        row_tokens: list[int],
        local: int,
        cache_g: Any,
        hist_g: Any,
        logits: Any,
        hidden: Any,
        boundaries: list[Any] | None = None,
        session_id: str | None = None,
    ) -> None:
        """Store one row's prompt cache in the bank, with its boundaries.

        Both caches or neither. Under the `committed` policy an entry without
        the draft head's history is unrestorable by design, so storing one
        would only burn memory on something the bank will always refuse.
        """

        if session_bank is None:
            return
        sliced = _slice_stock_row(cache_g, local)
        if sliced is None:
            return
        hist_snapshot = None
        if hist_g is not None:
            hist_row = _slice_stock_row(hist_g, local)
            if hist_row is None:
                return
            hist_snapshot = snapshot_cache(hist_row)
        if head_history == "committed" and hist_snapshot is None:
            return
        try:
            session_bank.put(
                runtime=rt,
                token_ids=[int(t) for t in row_tokens],
                cache=sliced,
                logits=None if logits is None else logits[local : local + 1],
                hidden=None if hidden is None else hidden[local : local + 1],
                mtp_history_policy=head_history,
                session_id=session_id,
                mtp_history_snapshot=hist_snapshot,
                # Trunk and draft-head state were captured at the same instant,
                # which is what the bank asserts this pair of numbers means.
                mtp_snapshot_epoch=0 if hist_snapshot is not None else None,
                gdn_boundaries=list(
                    _bank_boundaries.get(local) or []
                    if boundaries is None
                    else boundaries
                ),
            )
            bank_entries_written[0] += 1
        except Exception:
            # Storing is an optimisation for later requests. It must never be
            # why this one fails -- but a silent fail-closed with no way to ask
            # WHY is how an hour gets spent guessing at a zero counter, so the
            # reason is available behind the same debug flag the bank itself
            # uses.
            _debug_exc("bank-put")
            return

    def _slice_stock_row(cache_list: list[Any], row: int) -> list[Any] | None:
        """One row of a STOCK (pre-foldin) cache, as standalone copies.

        Separate from `_slice_row`, which handles the ragged lane. Checkpoints
        are taken mid-prefill, before `to_foldin_cache` has run, so the entries
        are still whatever the runtime's `make_cache` produced.

        `copy.copy` keeps each entry's own class and attributes and only the
        array references are replaced, so this does not need to know the type --
        which matters, because the trunk's cache classes are mlx-lm's and not
        ours.
        """

        import copy as _copy

        out: list[Any] = []
        try:
            for entry in cache_list:
                clone = _copy.copy(entry)
                keys = getattr(entry, "keys", None)
                if keys is not None:
                    values = getattr(entry, "values", None)
                    clone.keys = keys[row : row + 1]
                    clone.values = None if values is None else values[row : row + 1]
                    out.append(clone)
                    continue
                state = getattr(entry, "state", None)
                if isinstance(state, (list, tuple)) and state:
                    leaves = [
                        None if leaf is None else leaf[row : row + 1] for leaf in state
                    ]
                    if any(
                        leaf is not None and not hasattr(leaf, "shape")
                        for leaf in leaves
                    ):
                        return None
                    clone.state = leaves
                    out.append(clone)
                    continue
                return None
            mx.eval(
                *[
                    a
                    for e in out
                    for a in ([getattr(e, "keys", None)] + list(getattr(e, "state", []) or []))
                    if a is not None and hasattr(a, "shape")
                ]
            )
        except Exception:
            # A checkpoint is an optimisation and must never be why a request
            # fails. Anything unexpected simply means no checkpoint.
            return None
        return out

    def _prefill_group(
        row_ids: list[int], rows: list[list[int]] | None = None
    ) -> dict[str, Any]:
        """Prefill one group of equal-length rows; return its cache and tails.

        ``rows`` overrides the token lists, so a request pulled off the live
        queue mid-cohort is prefilled by this same function, at its OWN length,
        and then concatenated onto the running batch. That is what makes the
        padding trap structurally impossible on the admission path: there is no
        shared length for a joiner to be padded to.
        """

        rows = [slots[r] for r in row_ids] if rows is None else rows
        _bank_boundaries.clear()
        g_batch = len(rows)
        g_len = len(rows[0])
        keep = (
            min(g_len - 1, max(0, int(history_window)))
            if head_history == "committed"
            else 0
        )
        cache_g = rt.make_cache()
        hidden_tail: list[Any] = []
        tail_positions = 0
        logits = None
        hidden_last_g = None
        with attention_phase("prefill"):
            # Chunk ends, plus the boundary positions when a bank is in use.
            # Splitting a chunk costs one extra forward over the same tokens;
            # not splitting costs every later turn of the conversation.
            _chunk_ends = list(range(prefill_chunk, g_len, prefill_chunk)) + [g_len]
            #: Positions a snapshot is actually WANTED at. The prefill pauses at
            #: the union of these and the ordinary chunk ends, but only these
            #: get photographed -- an earlier version snapshotted at every pause,
            #: which silently ignored the cap entirely: with `prefill_chunk` at
            #: its 2048 default, a 32k prompt would capture sixteen snapshots at
            #: ~200 MB each on the 27B, about 3 GB for ONE entry, which is the
            #: exact failure the cap exists to prevent. Found by a stress case
            #: that set the cap to 1 and got three.
            _want_boundary: set[int] = set()
            if session_bank is not None:
                _want_boundary = set(_boundary_ends(0, g_len))
                _chunk_ends = sorted(set(_chunk_ends) | _want_boundary)
            _start_at = 0
            for end in _chunk_ends:
                start = _start_at
                _start_at = end
                if end <= start:
                    continue
                if memlog:
                    print(
                        f"[memlog] prefill chunk @{start} len={g_len} "
                        f"rows={g_batch} "
                        f"active={mx.get_active_memory()/2**30:.1f}GB "
                        f"peak={mx.get_peak_memory()/2**30:.1f}GB "
                        f"cache={mx.get_cache_memory()/2**30:.1f}GB",
                        flush=True,
                    )
                ids = mx.array([row[start:end] for row in rows])
                logits, hidden_c = rt.forward_ar(
                    ids, cache=cache_g, return_hidden=True, logits_keep=1
                )
                mx.eval(logits, hidden_c)
                hidden_last_g = hidden_c[:, -1:, :]
                if session_bank is not None and end < g_len and end in _want_boundary:
                    # Only at WANTED positions, not at every pause. Each of
                    # these is a fixed-size recurrent snapshot -- 49 MB on the
                    # 4B -- so the count has to be bounded by the cap rather
                    # than by how the prefill happened to be chunked.
                    _bank_boundary(rows, end, cache_g)
                if keep > 0:
                    hidden_tail.append(hidden_c)
                    tail_positions += int(hidden_c.shape[1])
                    while (
                        len(hidden_tail) > 1
                        and tail_positions - int(hidden_tail[0].shape[1]) >= keep + 1
                    ):
                        tail_positions -= int(hidden_tail.pop(0).shape[1])
        _require(
            logits is not None
            and int(logits.shape[0]) == g_batch
            and int(hidden_last_g.shape[0]) == g_batch,
            f"prefill collapsed the batch dim for B={g_batch}",
        )
        hidden = mx.concatenate(hidden_tail, axis=1) if hidden_tail else None

        mtp_hist_g = None
        if head_history == "committed":
            mtp_hist_g = rt.make_mtp_cache()
            if keep > 0 and hidden is not None and int(hidden.shape[1]) >= keep + 1:
                tail_len = int(hidden.shape[1])  # covers [L-tail_len, L)
                seed_hidden = hidden[:, tail_len - 1 - keep : tail_len - 1, :]
                seed_tokens = mx.array(
                    [row[g_len - keep :] for row in rows]
                )  # [Bg, keep]: token i+1 paired with hidden at i
                chunk = max(1, int(history_seed_chunk))
                settled = None
                for start in range(0, keep, chunk):
                    end = min(keep, start + chunk)
                    settled = rt.update_mtp_cache(
                        seed_hidden[:, start:end, :],
                        seed_tokens[:, start:end],
                        mtp_cache=mtp_hist_g,
                    )
                if settled is not None:
                    mx.eval(settled)
        # Store each row in the bank BEFORE the ragged conversion -- BOTH
        # caches. The bank is built for scalar caches, which is exactly what a
        # single row of a pre-foldin cache is, so no translation layer is
        # needed and the entries are interchangeable with the solo path's.
        #
        # The trunk state and the draft head's history are stored together on
        # purpose. Under the `committed` policy the bank REFUSES to restore an
        # entry that has one without the other, and it is right to: handing
        # back trunk state while the draft head keeps stale history would not
        # fail loudly, it would quietly draft against the wrong context. That
        # refusal is why an earlier version of this stored 352 MB and served
        # zero restores.
        if session_bank is not None:
            for _local, _row in enumerate(rows):
                _bank_put_row(
                    _row, _local, cache_g, mtp_hist_g, logits, hidden_last_g,
                    session_id=_session_for(_row),
                )

        # Convert to the ragged lane HERE, per group: from_scalar_cache pins
        # every row of this group at the group's own length, which is exactly
        # the per-row pinned offset the assembled cache needs.
        if mtp_hist_g is not None:
            to_foldin_cache(mtp_hist_g, g_batch)
        to_foldin_cache(cache_g, g_batch)
        return {
            "rows": row_ids,
            "cache": cache_g,
            "mtp_hist": mtp_hist_g,
            "logits": logits[:, -1, :],
            "hidden": hidden_last_g,
            "keep": keep,
            "length": g_len,
        }

    def _merge_rows(chunks: list[Any], perm_: Any) -> Any:
        merged = chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=0)
        return merged if perm_ is None else merged[perm_]

    def _permute_cache(cache_: list[Any], perm_: Any) -> None:
        """Reorder every row-major buffer into the caller's prompt order."""

        from .cache_state import OwnedRecurrentStateCache

        for layer_idx, entry in enumerate(cache_):
            if isinstance(entry, RaggedBatchKVCache):
                entry.filter(perm_)
            elif isinstance(entry, OwnedRecurrentStateCache):
                cache_[layer_idx] = OwnedRecurrentStateCache(
                    size=len(entry.state),
                    initial=[
                        None if leaf is None else leaf[perm_] for leaf in entry.state
                    ],
                    left_padding=entry.left_padding,
                    lengths=entry.lengths,
                )

    def _merge_caches(caches: list[Any], perm_: Any, *, capacity: int) -> list[Any]:
        """Concatenate per-group caches along the batch axis, in group order."""

        from .cache_state import OwnedRecurrentStateCache

        base = caches[0]
        for layer_idx, entry in enumerate(base):
            others = [c[layer_idx] for c in caches[1:]]
            if isinstance(entry, RaggedBatchKVCache):
                for other in others:
                    # ``extend`` aligns physical capacities by zero-padding the
                    # shorter buffer and keeps each row's own logical offset,
                    # which is precisely the ragged assembly this needs.
                    entry.extend(other)
                # from_scalar_cache seeded the host capacity bound from ONE
                # group's length. After the merge the true bound is the longest
                # group, and understating it would under-grow the buffer.
                entry._capacity_bound = int(capacity)
            elif isinstance(entry, OwnedRecurrentStateCache) and others:
                leaves = list(entry.state)
                merged: list[Any] = []
                for leaf_idx, leaf in enumerate(leaves):
                    group_leaves = [leaf] + [
                        list(other.state)[leaf_idx] for other in others
                    ]
                    if any(item is None for item in group_leaves):
                        merged.append(None)
                    else:
                        merged.append(mx.concatenate(group_leaves, axis=0))
                # A FRESH owned cache rather than assigning through the setter:
                # the owned buffers are sized for one group and this batch is
                # wider, so reusing them would depend on the reallocation
                # behaviour of a class documented as fixed-shape.
                base[layer_idx] = OwnedRecurrentStateCache(
                    size=len(merged),
                    initial=merged,
                    left_padding=entry.left_padding,
                    lengths=entry.lengths,
                )
        if perm_ is not None:
            _permute_cache(base, perm_)
        return base

    # T-210: the INITIAL cohort consults the store too, not only joiners.
    #
    # Leaving it out was a real gap rather than a tidy scope line: under
    # continuous batching most requests join, so most benefited, but the request
    # that SEALS a cohort never did -- and after an idle period that is the
    # first request every time, which is exactly when a person is waiting.
    #
    # Each row is tried on its own, because a hit changes only how that row's
    # standalone cache is produced. Rows that miss fall back to the ordinary
    # per-length-group prefill, which is what makes this safe to try first: the
    # worst case is the behaviour that existed before.
    parts: list[dict[str, Any]] = []
    assembled_rows: list[int] = []
    restored_rows: set[int] = set()
    if session_bank is not None:
        for row_idx, row in enumerate(slots):
            if row_idx >= n_real:
                continue  # a cohort_slots padding row has no conversation
            part = _reuse_prefix([int(t) for t in row])
            if part is None:
                continue
            parts.append(part)
            assembled_rows.append(row_idx)
            restored_rows.add(row_idx)
            # Per ROW, not per cohort. The cohort totals below cannot be
            # attributed back to a caller, and a response that says
            # "cached_tokens: 0" while the row reused 630 of its 755 tokens is
            # how a working optimisation gets reported as a broken one.
            prefix_covered_by_row[int(row_idx)] = int(part.get("covered") or 0)

    length_groups: dict[int, list[int]] = {}
    for row_idx, row in enumerate(slots):
        if row_idx in restored_rows:
            continue
        length_groups.setdefault(len(row), []).append(row_idx)
    group_order = sorted(length_groups)
    parts.extend(_prefill_group(length_groups[length]) for length in group_order)

    assembled_rows = assembled_rows + [
        r for length in group_order for r in length_groups[length]
    ]
    if assembled_rows == list(range(batch)):
        perm = None
    else:
        position = {row: i for i, row in enumerate(assembled_rows)}
        perm = mx.array([position[b] for b in range(batch)], dtype=mx.int32)

    # Every row, restored or freshly prefilled -- the capacity bound is the
    # longest prompt in the cohort, and a restored row's prompt is still one of
    # them. Sizing this off `group_order` alone would under-grow the buffer
    # whenever the longest row was the one that came from the store.
    max_prompt_len = max(len(slots[r]) for r in assembled_rows)
    cache = _merge_caches(
        [part["cache"] for part in parts], perm, capacity=max_prompt_len
    )
    logits_last = _merge_rows([part["logits"] for part in parts], perm)  # [B, V]
    hidden_last = _merge_rows([part["hidden"] for part in parts], perm)  # [B, 1, H]
    keep_hidden = max(part["keep"] for part in parts)
    prefill_s = time.perf_counter() - started

    # --- persistent head-history cache (committed policy) ---------------------
    mtp_hist = None
    # Bound even on the ``cycle`` policy, where there is no head cache to size:
    # the resize path reads it unconditionally and an unbound local would raise
    # only on the first admission, which is the worst place to find out.
    head_cap = 0
    hist_ragged: list[RaggedBatchKVCache] = []
    if head_history == "committed":
        mtp_hist = _merge_caches(
            [part["mtp_hist"] for part in parts], perm, capacity=keep_hidden
        )
        hist_ragged = [e for e in mtp_hist if isinstance(e, RaggedBatchKVCache)]
        # Freeze the head cache at its true bound: seed + one cycle's write
        # window per cycle (keeps <= K+1) + slack. Removes growth concats from
        # the hot path and gives the compiled draft chain static shapes. The
        # seed term is the LONGEST group's, since rows share one buffer.
        head_cap = keep_hidden + (max_new_tokens + 4) * (depth + 1) + 16
        for rc in hist_ragged:
            rc.step = 64
            rc.freeze_capacity(head_cap)

    ragged = [e for e in cache if isinstance(e, RaggedBatchKVCache)]

    # T-210: store each row's PROMPT-ONLY cache, right after prefill.
    #
    # This is the entry that actually hits, and finding out why is the most
    # useful thing the real-model run produced. Storing only at row FINISH keys
    # on prompt + generated, and measured against a real chat template that key
    # is not a prefix of the next turn's prompt: with thinking disabled the
    # template appends an empty `<think></think>` block to the GENERATION
    # prompt, and renders the same assistant turn without one when it is later
    # history. The sequences diverge inside turn one's own prompt -- 17 tokens
    # in, on the 4B -- so the stored key can never match.
    #
    # Truncating to the common prefix is not available as a repair: KV is
    # addressable and could be trimmed, but the GDN recurrent state cannot be
    # rewound. That is the same architectural fact behind the padding trap.
    #
    # A prompt-only key needs no truncation and hits the case agent traffic is
    # actually made of: a large stable context -- a system prompt, a file, a
    # retrieved document -- resent with a different question after it. The
    # entry is a strict prefix of the next prompt by construction.
    # --- per-REQUEST bookkeeping ---------------------------------------------
    # A slot serves one request at a time and, under refill, several over its
    # lifetime. Everything a caller gets back is indexed by REQUEST; everything
    # the decode loop touches is indexed by SLOT. Conflating the two is how a
    # continuous-batching bug ends up handing one caller another's tokens, so
    # they are separate names here even though they coincide without refill.
    queued = [dict(item) for item in (refill_queue or [])]
    requests: list[dict[str, Any]] = [
        {
            "prompt_len": prompt_lens[b],
            "cap": row_caps[b],
        }
        for b in range(n_real)
    ]
    for item in queued:
        requests.append(
            {
                "prompt_len": len(item["prompt"]),
                "cap": int(item.get("max_new_tokens", max_new_tokens)),
            }
        )
    req_tokens: list[list[int]] = [[] for _ in requests]
    req_finish: list[str | None] = [None] * len(requests)
    # slot -> request index currently occupying it; None once vacated.
    slot_request: list[int | None] = [
        b if b < n_real else None for b in range(batch)
    ]
    # request -> the slot it was admitted into, -1 if it never got one. Kept
    # separately from slot_request because that only ever shows CURRENT
    # occupancy: once a slot is refilled, its previous occupant disappears from
    # it, so counting admissions off the final state undercounts every request
    # that has already finished and been displaced.
    req_slot: list[int] = [b if b < n_real else -1 for b in range(len(requests))]
    next_queued = 0

    # --- continuous-batching state -------------------------------------------
    # ``continuous`` is the switch that keeps every pre-item-4 caller on exactly
    # the path it had. A plain ``generate_dense_mtp_batch(prompts, ...)`` with
    # no queue never resizes its row axis and is byte-for-byte the shipped
    # driver; the resize machinery arms only when a queue is actually in play.
    continuous = pull_queued is not None or bool(queued)
    # How wide this cohort may grow. Width follows demand in steps of one up to
    # this bound and never pads to it -- three requests run as a batch of three.
    max_rows = max(
        batch,
        int(max_cohort_rows)
        if max_cohort_rows is not None and int(max_cohort_rows) > 0
        else batch,
    )
    rows_peak = [batch]
    resizes = [0]
    # Rows a memory budget kept out. A serving lane that quietly runs narrow
    # because it is out of headroom looks identical to one nobody is using,
    # and the difference is the whole reason an operator would change a
    # setting, so it is counted and reported.
    memory_blocked = [0]

    done = [False] * batch
    for b in range(n_real, batch):
        done[b] = True  # dummy rows are inert
    accepted_by_depth = [0] * depth
    drafted_by_depth = [0] * depth
    accepted_total = 0
    evicted_total = 0
    deadline_hit = [False]

    def _commit_row(b: int, toks: list[int]) -> None:
        request = slot_request[b] if b < len(slot_request) else None
        if request is None or done[b]:
            return
        for tok in toks:
            if done[b]:
                return
            req_tokens[request].append(int(tok))
            if on_commit is not None:
                # The REQUEST index, not the slot. A slot serves several
                # requests under refill, so a slot cannot name the caller.
                # Without a refill queue the two are equal, so this is
                # behaviour-preserving for every pre-item-4 caller.
                on_commit(request, int(tok))
            if int(tok) in stop:
                done[b] = True
                req_finish[request] = "stop"
            elif len(req_tokens[request]) >= requests[request]["cap"]:
                done[b] = True
                req_finish[request] = "length"

    def _evict_cancelled() -> int:
        """Mark abandoned rows done so their slots can be reused.

        Host-side bookkeeping only: no cache state is touched, because the
        admission pass zeroes a joining row's state anyway. Marking ``done``
        mid-pipeline is safe -- an in-flight cycle still computes the row, and
        ``_commit_row`` drops its output because the row is done.
        """

        if is_cancelled is None:
            return 0
        evicted = 0
        for slot in range(batch):
            request = slot_request[slot]
            if request is None or done[slot]:
                continue
            try:
                abandoned = bool(is_cancelled(request))
            except Exception:
                # A cancellation probe that throws must not take the cohort
                # down with it; treat it as "still wanted" and carry on.
                abandoned = False
            if abandoned:
                done[slot] = True
                req_finish[request] = "cancelled"
                evicted += 1
        return evicted

    # ------------------------------------------------------------------ #
    # CONTINUOUS BATCHING (item 4).
    #
    # The cohort's ROW SET is rebuilt at admission boundaries: rows whose
    # request has finished are filtered off the batch axis, and joining
    # requests are prefilled at their own length and concatenated on. Width is
    # therefore always exactly the number of live rows and moves in steps of
    # one.
    #
    # Why rebuild rather than reuse a vacated slot in place. The in-place path
    # this replaces prefilled every joiner of one boundary at a single shared
    # ``prompt_len`` and set no pad mask, which is the padding trap: a pad
    # token entering a GDN layer is folded into a recurrent state that no
    # offset rewinds, and it fails SILENTLY -- the model loads, runs, and
    # returns fluent text conditioned on tokens the caller never sent. It was
    # harmless only while admission barely worked. Prefilling each joiner on
    # its own and concatenating along the batch axis removes the trap by
    # construction, and removes with it the whole-cache snapshot and masked
    # restore that the in-place path needed on every admission to protect the
    # rows that were NOT joining -- which was the expensive half of admission.
    # ------------------------------------------------------------------ #

    def _live_rows() -> int:
        return sum(1 for b in range(batch) if not done[b])

    def _kv_now() -> tuple[int, int, float]:
        """(reserved bytes, capacity tokens, bytes per token per row)."""

        reserved = 0
        capacity = 0
        for entry in ragged:
            keys = getattr(entry, "keys", None)
            if keys is None:
                continue
            values = getattr(entry, "values", None)
            reserved += int(keys.nbytes) + int(
                0 if values is None else values.nbytes
            )
            capacity = max(capacity, int(keys.shape[2]))
        if not reserved or not capacity or not batch:
            return reserved, capacity, 0.0
        return reserved, capacity, reserved / (capacity * batch)

    def _memory_admits(candidates: list[dict[str, Any]]) -> int:
        """How many of ``candidates`` fit under the working-set budget.

        Deliberately conservative and deliberately cheap: one host-side
        arithmetic estimate per admission boundary, no device sync. It can be
        wrong in either direction -- MLX's allocator, the prefill intermediates
        and the model weights are all outside this estimate -- so it is a
        BUDGET, not a guarantee, and the fail-loud path below still exists.

        What it does capture is the part that scales with what we admit: each
        new row's own KV over its whole lifetime, and the transient of the
        concatenate that adds it, which allocates the new batch-axis buffer
        while the old one is still live. That transient is the size of the
        current KV and it is the single largest thing growth does.
        """

        if not candidates or memory_headroom <= 0.0:
            return len(candidates)
        try:
            info = getattr(mx, "device_info", None) or mx.metal.device_info
            budget = float(memory_headroom) * float(
                info()["max_recommended_working_set_size"]
            )
            used = float(mx.get_active_memory())
        except Exception:
            # No device introspection (CPU fake, or a future MLX that moved
            # it). Admit everything rather than refuse everything: this guard
            # exists to avoid a crash, not to become one.
            return len(candidates)
        reserved, capacity, per_token = _kv_now()
        if per_token <= 0.0:
            if ragged and not batch:
                # The cohort is EMPTY -- every row finished and was removed --
                # so there is no KV to measure and no way to estimate. Admit
                # exactly one and re-measure at the next boundary. Admitting
                # none would deadlock the lane (it can never acquire the KV it
                # needs to justify admitting anyone); admitting all is the
                # unbounded behaviour this guard exists to remove.
                return 1
            # No ragged KV lane at all. Nothing to protect, nothing to
            # estimate; a guard with no subject must not become a throttle.
            return len(candidates)
        # The concatenate transient, paid once for the whole boundary.
        free = budget - used - float(reserved)
        admitted = 0
        for item in candidates:
            cost = per_token * (
                len(item["prompt"]) + int(item.get("max_new_tokens", max_new_tokens))
            )
            if free - cost < 0:
                break
            free -= cost
            admitted += 1
        if admitted < len(candidates):
            memory_blocked[0] += len(candidates) - admitted
        return admitted

    def _pad_rows_survive() -> bool:
        """Should ``cohort_slots`` padding rows be left in place?

        Yes while there is nothing to put in them: the fixed-shape parity gate
        pins ``cohort_slots`` precisely so every forward keeps identical shapes,
        and shrinking the padding away would defeat it. No the moment a joiner
        exists -- which is the ``_free_slots`` defect this replaces. The old
        version required ``slot_request[b] is not None``, so a row that had
        never held a request was permanently unusable and a joiner queued
        behind a busy row while an idle one sat next to it.
        """

        return bool(cohort_slots) and next_queued >= len(queued)

    def _free_slots() -> list[int]:
        """Rows that are finished and could be recycled or removed.

        Includes rows that never held a request. Retained as the introspection
        answer to "how much of this cohort is idle"; admission itself no longer
        hunts for a slot, it resizes the row axis.
        """

        pad_survives = _pad_rows_survive()
        return [
            b
            for b in range(batch)
            if done[b] and not (pad_survives and slot_request[b] is None)
        ]

    def _pull_from_queue(capacity: int) -> int:
        """Take up to ``capacity`` new requests from the LIVE queue."""

        nonlocal max_cycles
        if pull_queued is None or capacity <= 0:
            return 0
        try:
            items = [dict(item) for item in (pull_queued(int(capacity)) or [])]
        except Exception:
            # A queue that raises must not take the cohort down with it. The
            # rows already decoding belong to callers who are still waiting on
            # them, and a scheduler bug is not their problem.
            items = []
        for item in items:
            # Register the conversation BEFORE the row can be stored. Items
            # arriving here are the common case under load -- continuous
            # batching pulls most requests -- so missing them would leave
            # per-conversation budgeting working only for the handful that
            # happened to seal the cohort.
            _sid = item.get("session_id")
            if _sid:
                _session_by_prompt.setdefault(
                    tuple(int(t) for t in item.get("prompt") or []), str(_sid)
                )
            queued.append(item)
            requests.append(
                {
                    "prompt_len": len(item["prompt"]),
                    "cap": int(item.get("max_new_tokens", max_new_tokens)),
                }
            )
            req_tokens.append([])
            req_finish.append(None)
            req_slot.append(-1)
        if items:
            # The cycle guard is a runaway backstop, not a latency bound, and
            # admitting more work legitimately needs more cycles. Wall clock is
            # what bounds latency here; see ``deadline_s``.
            max_cycles = (max_new_tokens + 4) * (1 + len(queued))
        return len(items)

    def _resize_due() -> bool:
        """Is the row set out of date? Host-only and cheap; called per cycle."""

        if not continuous:
            return False
        if next_queued < len(queued) and _live_rows() < max_rows:
            return True
        # A finished row costs a full row of compute on every cycle until it
        # leaves, so it earns a boundary on its own -- unless removing it would
        # empty the cohort, in which case the loop is about to end anyway.
        return bool(_free_slots()) and _live_rows() > 0

    def _row_sampling_from(item: dict[str, Any]) -> tuple[float, int, float, bool]:
        return _encode_row_sampling(
            float(item.get("temperature", 0.0) or 0.0),
            int(item.get("top_k", 0) or 0),
            float(item.get("top_p", 1.0) or 1.0),
        )

    def _upgrade_for(items: list[dict[str, Any]]) -> None:
        """Acquire whatever the joining requests need, before they are admitted.

        Found by hammering at T4: an all-greedy cohort locked out every
        sampling request for its whole life, and under an ordinary mixed
        serving load that produced 27-second queue waits beside a cohort
        running two rows with four requests pending.

        Both upgrades are ONE-WAY. A cohort that has acquired randomness keeps
        it: dropping back to the greedy fast path mid-run would change how
        exactly-equal top logits break for every row still decoding, which is
        the one difference between the two encodings that is not bit-identical.
        Acquiring is safe in the other direction because a greedy row is
        encoded as top_k=1 at temperature 1 -- a point mass on the argmax --
        so it decodes correctly on the sampling path, it just pays for draws it
        did not previously need.
        """

        nonlocal sampling, any_penalty
        if not sampling and any(
            float(item.get("temperature", 0.0) or 0.0) > 0.0 for item in items
        ):
            if draft_core != "eager":
                # Cannot acquire this one: the compiled draft chain has no
                # sampling path. Leave the cohort greedy and let the service's
                # own filter keep the joiner out, which it does.
                return
            sampling = True
        if not any_penalty and any(
            float(item.get("presence_penalty", 0.0) or 0.0)
            or float(item.get("frequency_penalty", 0.0) or 0.0)
            for item in items
        ):
            # `_counts` is allocated lazily on first penalised call, so there
            # is nothing to build here. Every existing row's counts start at
            # zero, which is correct: they have not been penalised, and
            # penalties count the COMPLETION only.
            any_penalty = True

    def _resize_cohort(ll: Any, hl: Any, x0: Any) -> tuple[Any, Any, Any]:
        """Drop finished rows, admit queued ones, in ONE rebuild of the axis.

        Caller must have drained every pending commit and have no cycle in
        flight, or a stale commit from a row that has already left lands in
        another request's token list.
        """

        nonlocal batch, cache, mtp_hist, ragged, hist_ragged
        nonlocal keep_hidden, head_cap, next_queued
        nonlocal temp_dev, topk_dev, topp_dev, greedy_dev
        nonlocal presence_dev, frequency_dev
        nonlocal any_true_top_k, any_greedy_row, any_top_k, any_top_p

        room = max(0, max_rows - _live_rows())
        waiting = len(queued) - next_queued
        if room > waiting:
            waiting += _pull_from_queue(room - waiting)
        take = max(0, min(room, waiting))
        if take:
            take = _memory_admits(
                [queued[next_queued + i] for i in range(take)]
            )
        joiners = [n_real + next_queued + i for i in range(take)]

        pad_survives = bool(cohort_slots) and not joiners
        keep = [
            b
            for b in range(batch)
            if (not done[b]) or (pad_survives and slot_request[b] is None)
        ]
        if resize_debug:
            print(
                f"[resize] batch={batch} keep={keep} joiners={joiners} "
                f"done={done} slot_request={slot_request} "
                f"ll={ll.shape} hl={hl.shape} x0={x0.shape}",
                flush=True,
            )
        if len(keep) == batch and not joiners:
            return ll, hl, x0
        # BEFORE anything else: acquire whatever the joiners need. The host
        # state built below reads `any_penalty`, and the first-token draw at
        # the end reads `sampling`, so an upgrade that happened afterwards
        # would arrive one token too late for the joiner that asked for it.
        _upgrade_for([queued[request - n_real] for request in joiners])
        next_queued += take
        resizes[0] += 1

        # --- shrink: filter the row axis down to the survivors --------------
        if keep and len(keep) != batch:
            idx = mx.array(keep, dtype=mx.int32)
            _permute_cache(cache, idx)
            if mtp_hist is not None:
                _permute_cache(mtp_hist, idx)
            ll = ll[idx]
            hl = hl[idx]
            x0 = x0[idx]
            if _counts[0] is not None:
                _counts[0] = _counts[0][idx]
        elif not keep:
            # EVERY row finished on the same cycle, which is the common case
            # for a cohort of equal-length requests and was the first thing
            # this path got wrong. The caches are replaced wholesale below --
            # the joiners' own caches become the cohort -- so nothing is
            # filtered here, but the host-side row vectors still have to be
            # emptied. Leaving them at the old width made the joiner's first
            # token get drawn against a three-row `x0` in a one-row cohort.
            ll = ll[:0]
            hl = hl[:0]
            x0 = x0[:0]
            if _counts[0] is not None:
                _counts[0] = _counts[0][:0]
        if len(keep) != batch:
            for host in (
                _eff_temp,
                _eff_k,
                _eff_p,
                _is_greedy,
                _eff_presence,
                _eff_frequency,
                _row_seed,
                _row_stream,
            ):
                host[:] = [host[b] for b in keep]
            done[:] = [done[b] for b in keep]
            slot_request[:] = [slot_request[b] for b in keep]
            batch = len(keep)

        # --- grow: one prefill per DISTINCT joiner length -------------------
        if joiners:
            by_length: dict[int, list[int]] = {}
            for request in joiners:
                item = queued[request - n_real]
                by_length.setdefault(len(item["prompt"]), []).append(request)
            parts = []
            order: list[int] = []
            for length in sorted(by_length):
                group = by_length[length]
                rows_tokens = [
                    [int(t) for t in queued[r - n_real]["prompt"]] for r in group
                ]
                # T-210: try the prefix store one row at a time. A hit changes
                # only how this row's standalone cache is produced -- everything
                # downstream, including the concatenate, is identical.
                restored_any = False
                if session_bank is not None:
                    remaining_rows: list[int] = []
                    remaining_tokens: list[list[int]] = []
                    for request, tokens in zip(group, rows_tokens):
                        part = _reuse_prefix(tokens)
                        if part is None:
                            remaining_rows.append(request)
                            remaining_tokens.append(tokens)
                            continue
                        parts.append(part)
                        order.append(request)
                        restored_any = True
                    group, rows_tokens = remaining_rows, remaining_tokens
                    if not group:
                        continue
                elif restored_any:  # pragma: no cover - defensive
                    pass
                part = _prefill_group(list(range(len(group))), rows=rows_tokens)
                parts.append(part)
                order.extend(group)

            # Host state for the joining rows, appended in the SAME order the
            # caches are concatenated in, which is group order and not arrival
            # order -- the two differ whenever a boundary admits two lengths.
            for request in order:
                item = queued[request - n_real]
                eff_t, eff_k, eff_p, eff_greedy = _row_sampling_from(item)
                _eff_temp.append(eff_t)
                _eff_k.append(eff_k)
                _eff_p.append(eff_p)
                _is_greedy.append(eff_greedy)
                if any_penalty and (
                    float(item.get("presence_penalty", 0.0) or 0.0)
                    or float(item.get("frequency_penalty", 0.0) or 0.0)
                ):
                    _eff_presence.append(
                        _row_penalty(item.get("presence_penalty", 0.0) or 0.0, 0)
                    )
                    _eff_frequency.append(
                        _row_penalty(item.get("frequency_penalty", 0.0) or 0.0, 0)
                    )
                elif (
                    float(item.get("presence_penalty", 0.0) or 0.0)
                    or float(item.get("frequency_penalty", 0.0) or 0.0)
                ):
                    # Unreachable now: `_upgrade_for` above raises
                    # `any_penalty` for exactly this case, so a penalised
                    # joiner always finds the machinery it needs. Kept as a
                    # fail-loud backstop rather than deleted, because the
                    # alternative to reaching it is serving the request
                    # UNPENALISED and silent, and this branch is how that
                    # becomes a crash instead of a wrong answer.
                    raise RuntimeError(
                        "dense mtp_batch: a joining request asks for a "
                        "presence/frequency penalty but this cohort has no "
                        "penalty machinery and could not acquire it"
                    )
                else:
                    _eff_presence.append(0.0)
                    _eff_frequency.append(0.0)
                # A joiner brings its OWN randomness. Without this it would
                # continue from wherever some earlier request's stream had got
                # to -- cross-request coupling sourced from a request that has
                # already finished, which is harder to see and no less wrong.
                seed_value = item.get("seed")
                _row_seed.append(
                    int(seed_value) if seed_value is not None else int(sampling_seed)
                )
                _row_stream.append(_fresh_row_key(_row_seed[-1]))
                slot_request.append(request)
                done.append(False)

            join_from = batch
            batch = batch + len(order)

            capacity = max(
                [int(part["length"]) for part in parts]
                + [
                    int(getattr(entry, "_capacity_bound", 0) or 0)
                    for entry in ragged
                ]
            )
            caches = [part["cache"] for part in parts]
            cache = (
                _merge_caches([cache, *caches], None, capacity=capacity)
                if keep
                else _merge_caches(caches, None, capacity=capacity)
            )
            ragged = [e for e in cache if isinstance(e, RaggedBatchKVCache)]

            if mtp_hist is not None:
                keep_hidden = max(
                    int(keep_hidden), max(int(part["keep"]) for part in parts)
                )
                head_cap = max(
                    int(head_cap),
                    keep_hidden + (max_new_tokens + 4) * (depth + 1) + 16,
                )
                hists = [part["mtp_hist"] for part in parts]
                mtp_hist = (
                    _merge_caches([mtp_hist, *hists], None, capacity=keep_hidden)
                    if keep
                    else _merge_caches(hists, None, capacity=keep_hidden)
                )
                hist_ragged = [
                    e for e in mtp_hist if isinstance(e, RaggedBatchKVCache)
                ]
                for rc in hist_ragged:
                    rc.step = 64
                    rc.freeze_capacity(head_cap)

            join_logits = _merge_rows([part["logits"] for part in parts], None)
            join_hidden = _merge_rows([part["hidden"] for part in parts], None)
            ll = mx.concatenate([ll, join_logits], axis=0) if keep else join_logits
            hl = mx.concatenate([hl, join_hidden], axis=0) if keep else join_hidden
            if _counts[0] is not None:
                _counts[0] = mx.concatenate(
                    [
                        _counts[0],
                        mx.zeros(
                            (len(order), int(_counts[0].shape[1])), dtype=mx.float32
                        ),
                    ],
                    axis=0,
                )
        else:
            join_from = batch

        # --- rebuild the per-row device vectors for the NEW width -----------
        temp_dev = mx.array(_eff_temp, dtype=mx.float32)
        topk_dev = mx.array(_eff_k, dtype=mx.int32)
        topp_dev = mx.array(_eff_p, dtype=mx.float32)
        greedy_dev = mx.array(_is_greedy, dtype=mx.bool_)
        presence_dev = mx.array(_eff_presence, dtype=mx.float32)
        frequency_dev = mx.array(_eff_frequency, dtype=mx.float32)
        any_true_top_k = any(
            k > 0 for k, is_greedy in zip(_eff_k, _is_greedy) if not is_greedy
        )
        any_greedy_row = any(_is_greedy)
        any_top_k = any_true_top_k or any_greedy_row
        any_top_p = any(v < 1.0 for v in _eff_p)

        for row, request in enumerate(slot_request):
            if request is not None:
                req_slot[request] = row
        rows_peak[0] = max(rows_peak[0], batch)
        _emit_stats()

        # ORDER MATTERS. The sampling vectors above must be current BEFORE the
        # first token is drawn, or a joiner's very first token is sampled with
        # some other row's temperature -- the one token nobody thinks to look
        # at. This block used to sit before the vectors were rebuilt and a
        # mutation audit is what caught it.
        if join_from < batch:
            fresh = (
                _categorical(_filtered_logits(ll), _next_keys(1)[0])
                if sampling
                else mx.argmax(_penalised(ll), axis=-1)
            )
            is_new = mx.array(
                [row >= join_from for row in range(batch)], dtype=mx.bool_
            )
            x0 = mx.where(
                is_new,
                fresh,
                mx.concatenate(
                    [x0, mx.zeros((batch - join_from,), dtype=x0.dtype)], axis=0
                ),
            )
        if resize_debug:
            print(
                f"[resize] -> batch={batch} join_from={join_from} "
                f"ll={ll.shape} hl={hl.shape} x0={x0.shape}",
                flush=True,
            )
        return ll, hl, x0

    def _work_remaining() -> bool:
        if next_queued < len(queued):
            return True
        if not all(done):
            return True
        # A continuous cohort asks the queue one last time before giving up:
        # without this it would end on the exact cycle its last row finished
        # and a request that arrived during that cycle would pay a whole cohort
        # teardown and restart for the sake of a millisecond.
        return _pull_from_queue(max_rows) > 0

    # Worst case a row advances ``verified`` positions per cycle and every row
    # commits >= 1 real token per cycle, so the cycle guard is max_new + slack.
    # With refill the run serves several requests through each slot, so the
    # guard scales with how many requests can queue behind one slot.
    max_cycles = max_new_tokens + 2
    if queued:
        # Deliberately multiplicative, and deliberately NOT divided by the slot
        # count. Slots are filled as they free, so if every other slot holds a
        # max-length request and one turns over quickly, that ONE slot serves
        # the whole queue in sequence. A bound that assumed even spreading
        # would truncate that run and return short completions, which is a
        # correctness bug traded for a tidier guard. Wall clock is what bounds
        # latency here; see ``deadline_s``.
        max_cycles = (max_new_tokens + 4) * (1 + len(queued))

    run_deadline = (
        None
        if deadline_s is None or float(deadline_s) <= 0.0
        else started_all + float(deadline_s)
    )

    def _past_deadline() -> bool:
        return run_deadline is not None and time.perf_counter() >= run_deadline

    # ------------------------------------------------------------------ #
    # Decode loop.  All per-cycle cache commits (trunk KV offsets, GDN state
    # selection, head-history rewind/re-append) are DEVICE-SIDE expressions of
    # the accept vector ``k_arr``, no host value feeds the cache state, so
    # the pipelined mode can BUILD cycle N+1's graph before cycle N's single
    # host sync, keeping the GPU fed while the host runs python bookkeeping
    # and graph construction (the A1 overhead lever).  Serial mode syncs each
    # cycle before building the next (A/B baseline); both commit the identical
    # greedy sequence.
    # ------------------------------------------------------------------ #
    def _validate_capture_structure(captures: dict) -> None:
        if captures.get("__final_only__"):
            raise RuntimeError(
                "capture backend returned final-only captures; per-row commit "
                "needs per-step states"
            )
        for layer_idx, entry in enumerate(cache):
            capture = captures.get(layer_idx)
            if capture is not None:
                if "tape" in capture:
                    raise RuntimeError("tape captures cannot commit per-row")
                if "conv_states" not in capture or "states" not in capture:
                    raise RuntimeError("capture lacks per-step conv/gdn states")
                if int(capture.get("capture_start", 0)) != 0:
                    raise RuntimeError(
                        "device-side per-row commit requires capture_start == 0 "
                        "(use 'stock' or 'linear-gdn-from-conv-stream')"
                    )
            elif isinstance(entry, RaggedBatchKVCache):
                continue
            elif entry is not None and hasattr(entry, "state"):
                raise RuntimeError(
                    f"recurrent cache entry {layer_idx} has no capture; refusing "
                    "a partial per-row commit"
                )

    def _take_row_step(value: Any, k_arr: Any) -> Any:
        sel = k_arr.reshape((batch, 1) + (1,) * (int(value.ndim) - 2))
        sel = mx.broadcast_to(sel, (batch, 1) + tuple(value.shape[2:])).astype(mx.int32)
        return mx.contiguous(mx.take_along_axis(value, sel, axis=1)[:, 0])

    def _commit_rows_device(captures: dict, k_arr: Any) -> None:
        """commit_captured_rows semantics with a DEVICE keeps vector."""
        keeps_dev = (k_arr + 1).astype(mx.int32)
        for layer_idx, entry in enumerate(cache):
            capture = captures.get(layer_idx)
            if capture is not None:
                entry[0] = _take_row_step(capture["conv_states"], k_arr)
                entry[1] = _take_row_step(capture["states"], k_arr)
            elif isinstance(entry, RaggedBatchKVCache):
                entry.offsets = (
                    entry.offsets - int(verified) + keeps_dev
                ).astype(mx.int32)

    compiled_chain = None
    if draft_core == "compiled" and mtp_hist is not None and len(hist_ragged) == 1:
        _hist_entry = hist_ragged[0]

        def _chain_fn(ll: Any, hl: Any, keys: Any, values: Any, offsets: Any):
            tmp = RaggedBatchKVCache(
                batch_size=batch,
                step=_hist_entry.step,
                keys=keys,
                values=values,
                offsets=offsets,
            )
            tmp._frozen_capacity = _hist_entry._frozen_capacity
            tmp_list = list(mtp_hist)
            tmp_list[mtp_hist.index(_hist_entry)] = tmp
            x0 = mx.argmax(ll, axis=-1)
            h, t = hl, x0
            drafts = []
            for j in range(depth):
                dl, dh = rt.draft_mtp(
                    h,
                    mx.expand_dims(t, axis=1),
                    mtp_cache=tmp_list,
                    return_hidden=True,
                    mtp_depth=j + 1,
                )
                t = mx.argmax(dl[:, -1, :], axis=-1)
                h = dh[:, -1:, :]
                drafts.append(t)
            return mx.stack([x0, *drafts], axis=1), tmp.keys, tmp.values

        compiled_chain = mx.compile(_chain_fn)
    if draft_core == "compiled" and compiled_chain is None:
        raise RuntimeError(
            "draft_core='compiled' requires a committed head cache with exactly "
            f"one ragged KV entry (found {len(hist_ragged)})"
        )

    # --- exact speculative sampling (temperature > 0) ---------------------
    # One filtering pipeline shared by draft (q) and verify (p) so the
    # accept ratio compares like with like: scale by 1/T, then top-k, then
    # top-p, all on device. Randomness: a fixed base key split per submit
    # (deterministic under sampling_seed; pipelined and serial loops draw
    # the same streams in the same order).
    # Per-row sampling vectors, resolved once. Disabled filters are encoded as
    # values that make the arithmetic a no-op (top_k 0, top_p 1.0) so the hot
    # path has no per-row branches, and a greedy row inside a sampling cohort is
    # encoded as top_k=1 at temperature 1.0, which is exactly argmax.
    _eff_temp: list[float] = []
    _eff_k: list[int] = []
    _eff_p: list[float] = []
    _is_greedy: list[bool] = []
    for _row in range(batch):
        if _row >= n_real:
            _eff_temp.append(1.0)
            _eff_k.append(0)
            _eff_p.append(1.0)
            _is_greedy.append(False)
            continue
        _t = float(row_temperature[_row])
        if _t <= 0.0:
            _eff_temp.append(1.0)
            _eff_k.append(1)
            _eff_p.append(1.0)
            _is_greedy.append(True)
            continue
        _k = int(row_top_k[_row])
        _p = float(row_top_p[_row])
        _eff_temp.append(_t)
        _eff_k.append(_k if _k > 0 else 0)
        _eff_p.append(_p if 0.0 < _p < 1.0 else 1.0)
        _is_greedy.append(False)
    def _encode_row_sampling(
        temperature_value: float, k_value: int, p_value: float
    ) -> tuple[float, int, float, bool]:
        """One row's (temp, top_k, top_p, is_greedy) in the driver's encoding.

        Greedy is expressed as top_k=1 at temperature 1, which is the exact
        point-mass encoding used for the initial cohort. Disabled filters are
        encoded as arithmetic no-ops rather than branches.
        """

        if temperature_value <= 0.0:
            return 1.0, 1, 1.0, True
        return (
            float(temperature_value),
            int(k_value) if int(k_value) > 0 else 0,
            float(p_value) if 0.0 < float(p_value) < 1.0 else 1.0,
            False,
        )

    temp_dev = mx.array(_eff_temp, dtype=mx.float32)
    topk_dev = mx.array(_eff_k, dtype=mx.int32)
    topp_dev = mx.array(_eff_p, dtype=mx.float32)
    greedy_dev = mx.array(_is_greedy, dtype=mx.bool_)
    # A greedy row is encoded as top_k=1, which would otherwise switch the
    # top-k stage on for the WHOLE cohort and cost a full vocab sort every
    # cycle even though no caller asked for top-k. Distinguish the two cases:
    # a genuine caller-requested k needs the sort, whereas "every top-k row is
    # really a greedy row" needs only the row maximum, which is a reduction
    # rather than a sort. Serving is mostly greedy, so this is the common path.
    any_true_top_k = any(
        k > 0 for k, is_greedy in zip(_eff_k, _is_greedy) if not is_greedy
    )
    any_greedy_row = any(_is_greedy)
    any_top_k = any_true_top_k or any_greedy_row
    any_top_p = any(v < 1.0 for v in _eff_p)

    # Per-row seeds. A row with no supplied seed falls back to the run seed, so
    # a partially-seeded cohort behaves like today for the unseeded rows.
    _row_seed: list[int] = [
        int(row_sampling_seeds[r])
        if row_sampling_seeds is not None and r < len(row_sampling_seeds)
        else int(sampling_seed)
        for r in range(batch)
    ]

    def _distinct_sampling_seeds() -> bool:
        """True when per-row streams would actually change anything.

        Only SAMPLING rows matter: a greedy row draws no randomness, so its seed
        is irrelevant and must not drag the cohort onto the slower path. One
        distinct seed means the shared key is already correct and free.
        """

        seeds = {
            _row_seed[r]
            for r in range(n_real)
            if not _is_greedy[r]
        }
        return len(seeds) > 1

    per_row_random = row_sampling_seeds is not None and _distinct_sampling_seeds()

    def _fresh_row_key(seed_value: int) -> Any:
        # Split once so adjacent seeds (0, 1, 2, ...) start decorrelated rather
        # than trusting raw key values to be far apart.
        return mx.random.split(
            mx.random.key(int(seed_value) & 0x7FFFFFFF), num=2
        )[1]

    _row_stream: list[Any] = [_fresh_row_key(_row_seed[r]) for r in range(batch)]

    # --- penalties -------------------------------------------------------
    _PENALTY_MIN, _PENALTY_MAX = -2.0, 2.0

    def _row_penalty(value: float | list[float], row: int) -> float:
        raw = value[row] if isinstance(value, (list, tuple)) else value
        try:
            out = float(raw)
        except (TypeError, ValueError):
            out = 0.0
        return min(max(out, _PENALTY_MIN), _PENALTY_MAX)

    _eff_presence = [
        _row_penalty(presence_penalty, r) if r < n_real else 0.0
        for r in range(batch)
    ]
    _eff_frequency = [
        _row_penalty(frequency_penalty, r) if r < n_real else 0.0
        for r in range(batch)
    ]
    any_penalty = any(
        p != 0.0 for p in (*_eff_presence, *_eff_frequency)
    )
    presence_dev = mx.array(_eff_presence, dtype=mx.float32)
    frequency_dev = mx.array(_eff_frequency, dtype=mx.float32)
    # Dense [B, V] rather than the solo path's sparse scatter: sparse is
    # O(unique seen) for ONE row, but per-row sparse sets cannot be applied to
    # a [B, V] logit block without a loop over rows every cycle. Allocated only
    # when a penalty is actually set.
    _counts: list[Any] = [None]

    def _penalised(logits: Any) -> Any:
        """Raw logits with the OpenAI additive penalty applied, per row.

        Applied BEFORE temperature and the filters, matching
        fast_sampling.apply_penalties_mlx, so a penalty changes which tokens
        survive top-k/top-p rather than only re-weighting the survivors.
        """

        if not any_penalty:
            return logits
        if _counts[0] is None:
            _counts[0] = mx.zeros(
                (batch, int(logits.shape[-1])), dtype=mx.float32
            )
        counts = _counts[0]
        delta = _row_broadcast(frequency_dev, 2) * counts + _row_broadcast(
            presence_dev, 2
        ) * (counts > 0).astype(mx.float32)
        if logits.ndim == 3:
            delta = delta[:, None, :]
        return logits.astype(mx.float32) - delta

    def _count_committed(tokens_2d: Any, accepted: Any) -> None:
        """Advance per-row counts by the tokens committed this cycle.

        Device-side on purpose. Host-side counts would lag by a whole cycle in
        pipelined mode, where cycle N+1's graph is built before cycle N's host
        sync -- and a penalty computed from counts that are a cycle behind is a
        different sampler, not a slightly delayed one.
        """

        if not any_penalty:
            return
        width = int(tokens_2d.shape[1])
        positions = mx.arange(width, dtype=mx.int32).reshape(1, width)
        valid = (positions <= accepted.reshape(batch, 1)).astype(mx.float32)
        rows = mx.repeat(mx.arange(batch, dtype=mx.int32), width)
        _counts[0] = _counts[0].at[rows, tokens_2d.reshape(-1)].add(
            valid.reshape(-1)
        )

    _submit_seq = [0]

    def _next_keys(n: int) -> list[Any]:
        # Fresh key per submit, derived arithmetically from (seed, submit
        # index), MLX 0.32 has no fold_in. Splitting that key gives the
        # per-purpose subkeys for the cycle.
        if per_row_random:
            # One split per row per cycle (not per row per draw site): each row
            # carries its stream forward in slot 0 and spends the rest.
            sites: list[list[Any]] = [[] for _ in range(n)]
            for row in range(batch):
                parts = mx.random.split(_row_stream[row], num=n + 1)
                _row_stream[row] = parts[0]
                for site in range(n):
                    sites[site].append(parts[site + 1])
            _submit_seq[0] += 1
            return sites
        submit_key = mx.random.key(
            (int(sampling_seed) << 20) ^ (0x9E3779B1 * (_submit_seq[0] + 1) & 0xFFFFF)
        )
        _submit_seq[0] += 1
        keys = mx.random.split(submit_key, num=n)
        return [keys[i] for i in range(n)]

    def _row_broadcast(values: Any, ndim: int) -> Any:
        """Shape a per-row [B] vector to broadcast against [B, ..., V] logits."""

        return values.reshape((batch,) + (1,) * (ndim - 1))

    def _filtered_logits(logits: Any) -> Any:
        """[.., V] raw logits -> filtered/scaled logits (-inf outside the
        kept set), suitable for softmax or categorical.

        Every filter stage is per row. The cost is the same kernels as the
        cohort-wide version: the parameters simply arrive as [B] vectors
        broadcast along the vocab axis rather than as Python floats, so nothing
        here is a per-row loop.
        """

        out = _penalised(logits).astype(mx.float32) / _row_broadcast(
            temp_dev, logits.ndim
        )
        neg = mx.array(-1e30, dtype=mx.float32)
        vocab = int(out.shape[-1])
        if any_true_top_k:
            # Per-row k as an INDEX into the ascending sort: the k-th largest
            # value sits at position vocab - k. A row with k disabled indexes
            # position 0, whose value is the row minimum, and `out < min` keeps
            # everything — the disabled case falls out of the arithmetic
            # instead of needing a branch. A greedy row carries k=1, which
            # indexes the maximum, which is exactly argmax.
            k_eff = mx.where(
                (topk_dev <= 0) | (topk_dev >= vocab),
                mx.array(vocab, dtype=mx.int32),
                topk_dev,
            )
            sorted_asc = mx.sort(out, axis=-1)
            idx = mx.broadcast_to(
                _row_broadcast((vocab - k_eff).astype(mx.int32), out.ndim),
                out.shape[:-1] + (1,),
            ).astype(mx.int32)
            kth = mx.take_along_axis(sorted_asc, idx, axis=-1)
            out = mx.where(out < kth, neg, out)
        elif any_greedy_row:
            # No caller asked for top-k; the only rows needing this stage are
            # greedy ones, whose threshold is just their own maximum. Two O(V)
            # reductions instead of an O(V log V) sort over the full vocabulary,
            # every cycle. Non-greedy rows take the row minimum and keep
            # everything, the same no-op the sort path gives them.
            row_max = mx.max(out, axis=-1, keepdims=True)
            row_min = mx.min(out, axis=-1, keepdims=True)
            kth = mx.where(_row_broadcast(greedy_dev, out.ndim), row_max, row_min)
            out = mx.where(out < kth, neg, out)
        if any_top_p:
            sorted_desc = mx.sort(out, axis=-1)[..., ::-1]
            probs_sorted = mx.softmax(sorted_desc, axis=-1)
            cum = mx.cumsum(probs_sorted, axis=-1)
            # Keep the smallest prefix with cumulative mass >= top_p (always
            # keep the argmax). Threshold = value of the last kept entry. A row
            # with top_p disabled carries 1.0, and `cum - probs < 1.0` holds
            # everywhere, so again no branch.
            keep_sorted = (cum - probs_sorted) < _row_broadcast(
                topp_dev, out.ndim
            )
            thresh = mx.min(
                mx.where(keep_sorted, sorted_desc, mx.full(sorted_desc.shape, 1e30)),
                axis=-1,
                keepdims=True,
            )
            out = mx.where(out < thresh, neg, out)
        return out

    def _categorical(filtered: Any, key: Any) -> Any:
        if isinstance(key, list):
            # Row-sliced draws rather than Gumbel-max: same MLX kernel the
            # shared path uses, so the sampling distribution is identical and
            # only the key varies. B launches instead of one; paid only when
            # distinct seeds are actually present.
            return mx.concatenate(
                [
                    mx.random.categorical(
                        filtered[row : row + 1], axis=-1, key=key[row]
                    )
                    for row in range(batch)
                ],
                axis=0,
            )
        return mx.random.categorical(filtered, axis=-1, key=key)

    def _accept_uniform(key: Any, width: int) -> Any:
        """[B, width] acceptance uniforms, per row when seeds differ."""

        if isinstance(key, list):
            return mx.concatenate(
                [
                    mx.random.uniform(shape=(1, width), key=key[row])
                    for row in range(batch)
                ],
                axis=0,
            )
        return mx.random.uniform(shape=(batch, width), key=key)

    structure_checked = False
    phase_timing = str(os.environ.get("MTPLX_DENSE_BATCH_PHASE_TIMING", "")).strip().lower() in {"1", "true", "yes", "on"}
    phase_s: dict[str, float] = {"draft": 0.0, "verify": 0.0, "decide_gather": 0.0, "commit": 0.0, "head_append": 0.0}

    def _phase_mark(key: str, started: float, *arrays: Any) -> float:
        if not phase_timing:
            return started
        mx.eval(*[a for a in arrays if a is not None])
        now = time.perf_counter()
        phase_s[key] += now - started
        return now

    def _submit(ll: Any, hl: Any, x0_ids: Any) -> dict[str, Any]:
        """Build one cycle's full device graph (no host sync). ``x0_ids`` is
        this cycle's first committed token (previous cycle's next_x0)."""
        nonlocal structure_checked
        _pt = time.perf_counter() if phase_timing else 0.0
        cycle_keys = _next_keys(depth + 2) if sampling else None
        q_filtered: list[Any] = []  # per-depth filtered draft logits [B, V]
        if mtp_hist is not None:
            dcache = mtp_hist
            cycle_base = [rc.offsets for rc in hist_ragged]  # rebind-safe refs
        else:
            dcache = rt.make_mtp_cache()
            cycle_base = []
        draft_ids: list[Any] = []
        if compiled_chain is not None:
            base0 = cycle_base[0] if cycle_base else hist_ragged[0].offsets
            inp_c, k2, v2 = compiled_chain(
                ll, hl, hist_ragged[0].keys, hist_ragged[0].values, base0
            )
            hist_ragged[0].keys = k2
            hist_ragged[0].values = v2
            # offsets stay at the cycle base: the committed re-append below
            # overwrites the drafts' [base, base+K) slots and sets keeps.
            draft_ids = [inp_c[:, j + 1] for j in range(depth)]
            x0_ids = inp_c[:, 0]
            _pt = _phase_mark("draft", _pt, inp_c)
            inp = inp_c
            return _submit_post_draft(inp, x0_ids, draft_ids, cycle_base, _pt)
        h, t = hl, x0_ids
        for depth_idx in range(depth):
            for rc in hist_ragged:
                rc.reserve(1)
            # mtp_depth selects the per-depth adapter weights (mtp_adapter_depth)
            # exactly as the solo chained-draft cores do, omitting it drafts
            # depth 2+ without their adapters and costs acceptance.
            d_logits, d_hidden = rt.draft_mtp(
                h,
                mx.expand_dims(t, axis=1),
                mtp_cache=dcache,
                return_hidden=True,
                mtp_depth=depth_idx + 1,
            )
            if sampling:
                fq = _filtered_logits(d_logits[:, -1, :])  # [B, V]
                q_filtered.append(fq)
                t = _categorical(fq, cycle_keys[depth_idx])  # [B]
            else:
                t = mx.argmax(_penalised(d_logits[:, -1, :]), axis=-1)  # [B]
            h = d_hidden[:, -1:, :]
            draft_ids.append(t)
        _pt = _phase_mark("draft", _pt, *draft_ids)

        inp = mx.stack([x0_ids, *draft_ids], axis=1)  # [B, K+1]
        return _submit_post_draft(
            inp, x0_ids, draft_ids, cycle_base, _pt,
            q_filtered=q_filtered, cycle_keys=cycle_keys,
        )

    def _submit_post_draft(
        inp: Any, x0_ids: Any, draft_ids: list[Any], cycle_base: list[Any], _pt: float,
        *, q_filtered: list[Any] | None = None, cycle_keys: list[Any] | None = None,
    ) -> dict[str, Any]:
        nonlocal structure_checked
        for rc in ragged:
            rc.reserve(verified)
        with attention_phase("decode_verify"):
            v_logits, v_hidden, captures = rt.forward_ar_capture(
                inp,
                cache=cache,
                return_hidden=True,
                hidden_variant=hidden_variant,
                capture_backend=resolved_backend,
            )
        _require(
            int(v_logits.shape[0]) == batch
            and int(v_logits.shape[1]) == verified
            and int(v_hidden.shape[0]) == batch,
            f"verify collapsed shape: logits {tuple(v_logits.shape)} "
            f"hidden {tuple(v_hidden.shape)} for [B={batch}, rows={verified}]",
        )
        if not structure_checked:
            _validate_capture_structure(captures)
            structure_checked = True
        _pt = _phase_mark("verify", _pt, v_logits, v_hidden)

        # Device decision: accepted draft prefix length k per row.
        drafts_arr = mx.stack(draft_ids, axis=1)  # [B, K]
        if sampling:
            fp = _filtered_logits(v_logits)          # [B, K+1, V]
            p_probs = mx.softmax(fp, axis=-1)        # [B, K+1, V]
            q_probs = mx.softmax(mx.stack(q_filtered, axis=1), axis=-1)  # [B, K, V]
            d_idx = drafts_arr[:, :, None].astype(mx.int32)
            p_at_d = mx.take_along_axis(p_probs[:, :-1, :], d_idx, axis=-1)[:, :, 0]
            q_at_d = mx.take_along_axis(q_probs, d_idx, axis=-1)[:, :, 0]
            u = _accept_uniform(cycle_keys[depth], depth)
            # u < min(1, p/q)  <=>  u * q < p (q(d) > 0: d was sampled from q)
            match = (u * q_at_d < p_at_d).astype(mx.int32)
        else:
            posterior = mx.argmax(_penalised(v_logits), axis=-1)  # [B, K+1]
            match = (drafts_arr == posterior[:, :-1]).astype(mx.int32)
        k_arr = mx.sum(mx.cumprod(match, axis=1), axis=1).astype(mx.int32)
        # inp is [B, K+1] = x0 followed by the drafts; k_arr is how many drafts
        # were accepted, so positions 0..k_arr are exactly this cycle's
        # committed tokens.
        _count_committed(inp, k_arr)

        # Next-cycle logits/hidden at each row's accept position (device).
        sel = k_arr.reshape(batch, 1)
        gather_l = mx.take_along_axis(
            v_logits,
            mx.broadcast_to(
                sel[:, :, None], (batch, 1, int(v_logits.shape[2]))
            ).astype(mx.int32),
            axis=1,
        )
        gather_h = mx.take_along_axis(
            v_hidden,
            mx.broadcast_to(
                sel[:, :, None], (batch, 1, int(v_hidden.shape[2]))
            ).astype(mx.int32),
            axis=1,
        )

        # Next cycle's first token. Greedy: argmax at the accept position
        # (identical to the pre-sampling code, which argmaxed next_ll at the
        # top of the following cycle). Sampling: exact residual resample on
        # rejection, bonus sample from p at position K on full acceptance.
        if sampling:
            sel3 = mx.broadcast_to(
                k_arr.reshape(batch, 1, 1), (batch, 1, int(p_probs.shape[2]))
            ).astype(mx.int32)
            p_row = mx.take_along_axis(p_probs, sel3, axis=1)[:, 0, :]  # [B, V]
            q_sel = mx.minimum(k_arr, depth - 1).reshape(batch, 1, 1)
            q_row = mx.take_along_axis(
                q_probs,
                mx.broadcast_to(q_sel, (batch, 1, int(q_probs.shape[2]))).astype(mx.int32),
                axis=1,
            )[:, 0, :]
            rejected = (k_arr < depth).reshape(batch, 1).astype(mx.float32)
            resid = mx.maximum(p_row - rejected * q_row, 0.0)
            resid = resid / mx.maximum(
                mx.sum(resid, axis=-1, keepdims=True), mx.array(1e-20, dtype=mx.float32)
            )
            next_x0 = _categorical(mx.log(resid + 1e-30), cycle_keys[depth + 1])
        else:
            next_x0 = mx.argmax(_penalised(gather_l[:, 0, :]), axis=-1)

        _pt = _phase_mark("decide_gather", _pt, k_arr, gather_l, gather_h)

        # Device-side per-row commits: trunk KV + GDN state, then the head
        # history (rewind to base, re-append from trunk hiddens, keeps).
        _commit_rows_device(captures, k_arr)
        if phase_timing:
            _pt = _phase_mark("commit", _pt, *[e[1] for e in cache if hasattr(e, "__getitem__") and not isinstance(e, RaggedBatchKVCache)][:4])
        if mtp_hist is not None:
            for rc, base in zip(hist_ragged, cycle_base):
                rc.offsets = base
                rc.reserve(verified)
            rt.update_mtp_cache(v_hidden, inp, mtp_cache=mtp_hist)
            keeps_dev = (k_arr + 1).astype(mx.int32)
            for rc, base in zip(hist_ragged, cycle_base):
                rc.offsets = (base + keeps_dev).astype(mx.int32)
            _pt = _phase_mark("head_append", _pt, *[rc.offsets for rc in hist_ragged][:2])

        bundle = mx.concatenate([inp.T.astype(mx.int32), k_arr[None, :]], axis=0)
        return {
            "bundle": bundle,
            "next_ll": gather_l[:, 0, :],
            "next_hl": gather_h,
            "next_x0": next_x0,
        }

    def _read(sub: dict[str, Any]) -> tuple[list[list[int]], list[int]]:
        mx.eval(sub["bundle"])  # THE one blocking sync per cycle
        rows = sub["bundle"].tolist()
        inp_host = [[int(rows[j][b]) for j in range(verified)] for b in range(batch)]
        k_host = [int(rows[verified][b]) for b in range(batch)]
        return inp_host, k_host

    def _note_slow_cycle(started_at: float, admitted: bool) -> None:
        """One line per unusually long cycle, naming the cohort's state.

        Called after the host sync, so the elapsed time covers the whole cycle
        including whatever the GPU was doing. Records rather than diagnoses: the
        point is to turn "the cohort paused" into a fact with a width, a queue
        depth and an admission flag attached to it.
        """

        if cycle_warn_s <= 0.0:
            return
        elapsed = time.perf_counter() - started_at
        if elapsed < cycle_warn_s:
            return
        row = {
            "at": time.time(),
            "cycle": int(cycles),
            "elapsed_s": round(elapsed, 3),
            "rows": int(batch),
            "live_rows": int(_live_rows()),
            "admission_boundary": bool(admitted),
            "queued_unadmitted": int(len(queued) - next_queued),
            "rows_blocked_by_memory": int(memory_blocked[0]),
            "resizes": int(resizes[0]),
            "committed_total": int(sum(len(t) for t in req_tokens)),
        }
        slow_cycles.append(row)
        message = "[cycle-warn] " + " ".join(f"{k}={v}" for k, v in row.items())
        print(message, flush=True)
        if cycle_trace_path:
            try:
                with open(cycle_trace_path, "a") as handle:
                    handle.write(json.dumps(row) + "\n")
            except OSError:
                # A trace that cannot be written must not take the cohort down.
                pass

    def _drain(inp_host: list[list[int]], k_host: list[int]) -> None:
        nonlocal accepted_total
        # ``batch`` and not ``n_real``: under continuous batching the row axis
        # is rebuilt as requests come and go, so the live rows are the whole
        # current width and ``n_real`` names only how wide the cohort STARTED.
        for b in range(batch):
            if done[b]:
                continue
            k_b = k_host[b]
            accepted_total += k_b
            for j in range(depth):
                drafted_by_depth[j] += 1
                if j < k_b:
                    accepted_by_depth[j] += 1
            _commit_row(b, inp_host[b][: 1 + k_b])

    _stats_emitted = [0.0]

    def _emit_stats(force: bool = True) -> None:
        """Report the cohort's live state to a monitor.

        ``force=False`` emits at most once a second, which is what makes it safe
        to call from the decode loop.

        **Timestamped, because freshness is the fact a monitor actually needs.**
        The first version emitted only at cohort start and at admission
        boundaries. A cohort that seals narrow and does not resize -- precisely
        the state worth watching -- then froze its payload at the start values,
        and a reader had no way to tell minutes-old numbers from current ones.
        The soak's zombie-slot condition reads these fields and very nearly
        aborted an overnight run on a snapshot of the past.
        """

        if on_stats is None:
            return
        now = time.time()
        if not force and now - _stats_emitted[0] < 1.0:
            return
        _stats_emitted[0] = now
        try:
            on_stats(
                {
                    "at": now,
                    "cycle": int(cycles),
                    "rows": int(batch),
                    "live_rows": int(_live_rows()),
                    "max_rows": int(max_rows),
                    "queued_unadmitted": int(len(queued) - next_queued),
                    "rows_blocked_by_memory": int(memory_blocked[0]),
                    "resizes": int(resizes[0]),
                }
            )
        except Exception:
            # A monitor that throws must not take the cohort down with it.
            pass

    # Emit ONCE before the first cycle. Without this the payload is empty until
    # the first admission boundary, and a monitor reading it during that window
    # sees nothing and has to fall back to a worse signal -- which is exactly
    # when a cohort is at its narrowest and most likely to be misread.
    _emit_stats()

    cycles = 0
    started_decode = time.perf_counter()
    if sampling:
        x0_next = _categorical(_filtered_logits(logits_last), _next_keys(1)[0])
    else:
        x0_next = mx.argmax(logits_last, axis=-1)
    if loop_mode == "serial":
        while _work_remaining() and cycles < max_cycles:
            if _past_deadline():
                deadline_hit[0] = True
                break
            evicted_total += _evict_cancelled()
            # Ask the live queue BEFORE deciding whether a boundary is due.
            # Without this probe the serial loop only ever noticed new work
            # when some row happened to finish, so a cohort could not widen
            # while every row was still busy -- which is the frozen refill list
            # wearing a different hat, and it cost a test failure to find.
            if continuous:
                _pull_from_queue(
                    max_rows - _live_rows() - (len(queued) - next_queued)
                )
            if _resize_due():
                # Top of the loop: the previous iteration's read synced, and
                # its commits are already drained, so done flags are current
                # and no stale commit can leak into a newly admitted request.
                logits_last, hidden_last, x0_next = _resize_cohort(
                    logits_last, hidden_last, x0_next
                )
            if not batch or all(done):
                if not _work_remaining():
                    break
                continue
            _cycle_started = time.perf_counter()
            sub = _submit(logits_last, hidden_last, x0_next)
            _drain(*_read(sub))
            logits_last, hidden_last = sub["next_ll"], sub["next_hl"]
            x0_next = sub["next_x0"]
            cycles += 1
            _note_slow_cycle(_cycle_started, admitted=False)
            _emit_stats(force=False)
    else:
        # PIPELINED: cycle N+1 is built and kicked BEFORE cycle N's sync, so
        # the GPU executes N+1 while the host drains N's bookkeeping.  The
        # done-flags lag one cycle, so the loop can overshoot by <= 1 cycle
        # (its results are simply never read; bounded garbage on done rows).
        sub = _submit(logits_last, hidden_last, x0_next)
        mx.async_eval(sub["bundle"], sub["next_ll"], sub["next_hl"], sub["next_x0"])
        pending: tuple[list[list[int]], list[int]] | None = None
        while True:
            # Pause the pipeline only on cycles where an admission is actually
            # due. `done` lags one cycle here, so a freed slot may be noticed a
            # cycle late; that delays a joiner and cannot mis-admit, because a
            # slot reading free has already been drained.
            if _past_deadline():
                deadline_hit[0] = True
                break
            evicted_total += _evict_cancelled()
            # Ask the live queue BEFORE deciding whether to pause. The probe is
            # a host-side lock acquisition per cycle and nothing more, and
            # skipping it would mean a joiner waits for some OTHER row to
            # finish before anyone notices it arrived -- which is the seal this
            # whole task exists to remove, reintroduced one layer down.
            if continuous:
                _pull_from_queue(
                    max_rows - _live_rows() - (len(queued) - next_queued)
                )
            want_admit = _resize_due()
            nsub = None
            if not want_admit and cycles + 1 < max_cycles and not all(done):
                nsub = _submit(sub["next_ll"], sub["next_hl"], sub["next_x0"])
                mx.async_eval(nsub["bundle"], nsub["next_ll"], nsub["next_hl"], nsub["next_x0"])
            _cycle_started = time.perf_counter()
            if pending is not None:
                _drain(*pending)
            pending = _read(sub)
            cycles += 1
            _note_slow_cycle(_cycle_started, admitted=want_admit)
            _emit_stats(force=False)
            if want_admit:
                # Nothing is in flight (nsub was deliberately not submitted).
                # Drain this cycle's commits BEFORE rebuilding the row axis, or
                # a stale commit from a row that has already left lands in
                # another request's tokens.
                _drain(*pending)
                pending = None
                logits_last, hidden_last, x0_next = _resize_cohort(
                    sub["next_ll"], sub["next_hl"], sub["next_x0"]
                )
                if not batch or not _work_remaining() or cycles >= max_cycles:
                    break
                sub = _submit(logits_last, hidden_last, x0_next)
                mx.async_eval(
                    sub["bundle"], sub["next_ll"], sub["next_hl"], sub["next_x0"]
                )
                continue
            if memlog and cycles % 20 == 1:
                print(
                    f"[memlog] cycle {cycles} "
                    f"active={mx.get_active_memory()/2**30:.1f}GB "
                    f"peak={mx.get_peak_memory()/2**30:.1f}GB "
                    f"cache={mx.get_cache_memory()/2**30:.1f}GB",
                    flush=True,
                )
            if nsub is None:
                break
            _drain(*pending)
            pending = None
            if not _work_remaining() or cycles >= max_cycles:
                break
            sub = nsub
        if pending is not None:
            _drain(*pending)

    def _kv_reservation() -> dict[str, Any]:
        """Physical KV reserved by this cohort, as bytes and as tokens/slot.

        Per-slot capacity is monotone: it grows to the longest prompt a slot has
        served and never shrinks, so a short request that reuses that slot is
        charged its predecessor's high-water mark. Reported because the number
        is large enough to matter -- 64 KiB per token per slot on this trunk --
        and because "reserved" and "used" diverging is the signal that a cohort
        is holding memory for requests that have already finished.
        """

        reserved = 0
        capacity = 0
        for entry in ragged:
            keys = getattr(entry, "keys", None)
            values = getattr(entry, "values", None)
            if keys is None:
                continue
            reserved += int(keys.nbytes) + int(
                0 if values is None else values.nbytes
            )
            capacity = max(capacity, int(keys.shape[2]))
        if not reserved:
            return {}
        used_tokens = sum(
            requests[i]["prompt_len"] + len(req_tokens[i])
            for i in range(len(requests))
        )
        per_token_per_row = (
            reserved / (capacity * batch) if capacity and batch else 0.0
        )
        return {
            "kv_reserved_bytes": reserved,
            "kv_capacity_tokens_per_slot": capacity,
            "kv_bytes_per_token_per_slot": round(per_token_per_row, 1),
            # What the reservation would have been if capacity tracked actual
            # use. The gap IS the cost of the monotone bound.
            "kv_used_bytes_estimate": int(used_tokens * per_token_per_row),
        }

    decode_s = time.perf_counter() - started_decode

    # A request that never got a slot is reported as such rather than as an
    # empty completion: "not admitted" and "produced nothing" are different
    # outcomes and a caller must be able to tell them apart.
    admitted = {index for index, slot in enumerate(req_slot) if slot >= 0}
    for index in range(len(requests)):
        if req_finish[index] is not None:
            continue
        if index not in admitted:
            # Queued but never given a slot: the run hit its guard first.
            # Distinct from "admitted and produced nothing".
            req_finish[index] = (
                "deadline" if deadline_hit[0] else "not_admitted"
            )
        elif len(req_tokens[index]) >= requests[index]["cap"]:
            # A row that reached its own cap finished for its own reason, even
            # if the deadline fired on the same cycle.
            req_finish[index] = "length"
        elif deadline_hit[0]:
            req_finish[index] = "deadline"
        else:
            req_finish[index] = "cycle_cap"

    streams = [
        DenseBatchStreamResult(
            index=index,
            prompt_len=requests[index]["prompt_len"],
            tokens=req_tokens[index],
            finish_reason=str(req_finish[index]),
            sha=token_sha(req_tokens[index]),
            slot=req_slot[index],
        )
        for index in range(len(requests))
    ]
    generated = sum(len(t) for t in req_tokens)
    meta: dict[str, Any] = {}
    if collect_stats:
        meta = {
            "elapsed_s": time.perf_counter() - started_all,
            "lane": "dense_mtp_batch_ragged_kv+postconv_capture",
            "capture_backend": resolved_backend,
            "cohort_slots": None if cohort_slots is None else int(cohort_slots),
            "real_streams": n_real,
            "syncs_per_cycle": 1,
            "verified_per_cycle": verified,
            "head_history": head_history,
            "history_window": int(history_window),
            "prefill_chunk": int(prefill_chunk),
            "loop_mode": loop_mode,
            "draft_core": draft_core,
            "sampling": bool(sampling),
            # Recorded because both of these can now be ACQUIRED mid-cohort by
            # a joining request, so "did this cohort end up sampling / applying
            # penalties" is no longer answerable from the caller's arguments.
            "penalties": bool(any_penalty),
            "row_temperature": [float(t) for t in row_temperature],
            "row_top_k": [int(k) for k in row_top_k],
            "row_top_p": [float(v) for v in row_top_p],
            "ragged_prompts": bool(ragged_prompts),
            "requests_total": len(requests),
            "requests_queued": len(queued),
            "requests_admitted": len(admitted),
            "requests_evicted": evicted_total,
            # The receipts that distinguish continuous batching WORKING from
            # continuous batching merely being present. `cohort_resizes` counts
            # rebuilds of the row axis; `rows_peak` is the widest the cohort
            # ever ran; `rows_final` is how narrow it had shrunk by the end.
            "continuous": bool(continuous),
            "max_cohort_rows": int(max_rows),
            "cohort_resizes": int(resizes[0]),
            "rows_blocked_by_memory": int(memory_blocked[0]),
            "prefix_restores": int(prefix_restores[0]),
            "prefix_prompt_tokens_skipped": int(prefix_tokens_skipped[0]),
            "prefix_covered_by_row": dict(prefix_covered_by_row),
            "prefix_entries_stored": int(bank_entries_written[0]),
            "prefix_restore_failures": int(prefix_restore_failures[0]),
            # Cycles that took longer than MTPLX_DENSE_BATCH_CYCLE_WARN_S, with
            # what the cohort was doing at the time. Empty unless the threshold
            # is set. Bounded to the last 64 so a long run cannot accumulate
            # them without limit.
            "slow_cycles": list(slow_cycles[-64:]),
            "rows_peak": int(rows_peak[0]),
            "rows_final": int(batch),
            "per_row_random": per_row_random,
            "deadline_hit": deadline_hit[0],
            **_kv_reservation(),
            "prompt_lens": list(prompt_lens),
            "ragged_attention": bool(ragged_attention),
            "phase_s": dict(phase_s) if phase_timing else None,
        }
    return DenseBatchResult(
        batch_size=n_real,
        depth=depth,
        streams=streams,
        cycles=cycles,
        generated_tokens=generated,
        accepted_draft_tokens=accepted_total,
        accepted_by_depth=accepted_by_depth,
        drafted_by_depth=drafted_by_depth,
        prefill_s=prefill_s,
        decode_s=decode_s,
        meta=meta,
    )
