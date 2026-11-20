"""Shared serving prefill guard construction and prompt-scoring admission."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Mapping, Sequence


def prompt_scoring_forward_widths(
    runtime: Any, prompt_tokens: int, requested: int | None
) -> list[int | None]:
    """Scoring's target widths, independent of generation for generic models."""
    from mtplx.generation import PROMPT_SCORING_CHUNK_SIZE

    if getattr(runtime, "backend_id", None) != "gemma4_assistant":
        return [PROMPT_SCORING_CHUNK_SIZE]
    from mtplx.backends.gemma4_assistant import (
        GEMMA4_MIN_PREFILL_CHUNK,
        gemma4_prefill_chunk_tokens,
    )

    default = gemma4_prefill_chunk_tokens(prompt_tokens)
    if default is None:
        return [None]
    first = max(GEMMA4_MIN_PREFILL_CHUNK, int(requested)) if requested else default
    return [first, default] if default < first else [first]


def prompt_scoring_growth(
    state: Any, *, prompt_tokens: int, width: int | None
) -> dict[str, Any]:
    """A fresh target cache and bounded logits; no draft, publish, repage or decode.

    Keep the serving policy's calibrated forward scratch. Charge a float32
    logits buffer beside it, or three such buffers for scoring's cast,
    reduction and selection if that is larger. The two phases share memory.
    """
    from mtplx.generation import PROMPT_SCORING_CHUNK_SIZE, _sustained_prefill_layout
    from mtplx.server import openai as srv

    runtime = state.runtime
    args = srv._runtime_text_args(runtime)
    geometry = srv._admission_geometry(state, prefill_width=width)
    plan = getattr(state, "memory_plan", None)
    draft_bytes = int(getattr(plan, "mtp_history_bytes_per_token", 0) or 0)
    if srv._runtime_has_qsa_indexer(runtime):
        dim = int(getattr(args, "indexer_head_dim", 128) or 128)
        ratio = max(1, int(getattr(args, "indexer_compress_ratio", 4) or 4))
        shape = srv._attention_shape(runtime)
        if shape is not None:
            draft_bytes += dim * 2 + dim * 2 // ratio + dim * 4 // ratio
            draft_bytes += 2 * shape[1] * shape[2] * 2
    # The memory plan includes the separate MTP head; scoring never builds it.
    draft_bytes = min(geometry.aux_bytes_per_token, draft_bytes)
    geometry = replace(
        geometry,
        live_bytes_per_token=geometry.live_bytes_per_token - draft_bytes,
        paged_bytes_per_token=geometry.paged_bytes_per_token - draft_bytes,
        aux_bytes_per_token=geometry.aux_bytes_per_token - draft_bytes,
    )
    layout = _sustained_prefill_layout()
    if getattr(runtime, "backend_id", None) == "gemma4_assistant":
        from mtplx.backends.gemma4_assistant import gemma4_resident_kv_bytes_per_token

        layout = "contiguous_dense_decode"
        if width is not None and args is not None:
            # No full-prompt sliding KV is retained for the assistant.
            geometry = replace(
                geometry, live_bytes_per_token=gemma4_resident_kv_bytes_per_token(args)
            )
    elif layout == "contiguous_then_repage":
        layout = "contiguous_dense_decode"  # The scorer never calls repage.
    rows = prompt_tokens if width is None else min(prompt_tokens, width)
    forward_scratch, source = srv._admission_scratch_bytes(
        state, rows=rows, prompt_tokens=prompt_tokens, geometry=geometry
    )
    logits_rows = min(rows, PROMPT_SCORING_CHUNK_SIZE)
    logits_bytes = logits_rows * int(getattr(args, "vocab_size", 0) or 0) * 4
    scratch = max(forward_scratch + logits_bytes, 3 * logits_bytes)
    model = srv._admission_growth(
        geometry, prompt_tokens=prompt_tokens, reused_tokens=0,
        restore_copies_prefix=False, layout=layout, source_layout=None,
        output_tokens=0, publish=False, scratch_bytes=scratch,
        context_transient_bytes_per_token=srv._admission_context_transient_per_token(
            geometry, rows=rows, scratch_source=source
        ),
        prefill_chunk_tokens=width,
    )
    # A cache paged from the start can allocate a quantized working copy
    # during its forwards. Retain that charge, but never a decode reservation.
    model["prefill_end_bytes"] += model["quant_working_bytes"]
    model.update(
        workload="prompt_scoring", repage_bytes=0, decode_start_bytes=0,
        growth_bytes=model["prefill_end_bytes"],
        scratch_source=source, scratch_rows=rows, prefill_chunk_tokens=width,
        logits_rows=logits_rows, logits_bytes=logits_bytes,
        chunk_bytes=srv._admission_chunk_bytes(geometry, rows, scratch),
    )
    return model


_WIDE_CHUNK_ADMISSION_REFUSED_REASON = (
    "the prefill admission priced the wider chunk with the rest of the "
    "request after reclaiming what it could and it did not fit under the "
    "memory line; the request's prefill_admission_shed receipt carries the "
    "arithmetic"
)


def settle_wide_prefill_chunk(
    runtime: Any,
    *,
    prompt_tokens: int,
    rungs: Sequence[int],
    pricing: Mapping[str, Any],
    receipt: dict[str, Any] | None = None,
) -> int | None:
    """The family's wide prefill chunk this request runs, or None for the
    profile's own plan.

    The prefill admission prices the wide ``rungs`` with every other width
    after anything it reclaims and settles on the widest that fits: when it
    priced them (``pricing["growth"]``), its choice stands. The choice used
    to be made before the admission by a second bill against live memory, so
    a cold 123K prompt on 2026-09-29 ran at 2,048 rows although 4,096 fitted
    once the admission had freed memory. When the admission priced nothing
    (switched off, no Metal limit, or its own guard failed), the live-memory
    gate decides as it always did.
    """

    from mtplx.generation import qwen4_wide_prefill_chunk_tokens

    settled = pricing.get("growth")
    if not isinstance(settled, Mapping) or "prefill_chunk_tokens" not in settled:
        gate_receipt: dict[str, Any] = {}
        granted_width = qwen4_wide_prefill_chunk_tokens(
            runtime, prompt_tokens=prompt_tokens, receipt=gate_receipt
        )
        if receipt is not None and gate_receipt:
            receipt.update(gate_receipt, decided_by="live_memory_gate")
        return granted_width
    width = settled.get("prefill_chunk_tokens")
    candidates = [int(rung) for rung in rungs]
    granted = width is not None and int(width) in candidates
    if receipt is not None:
        receipt.update(
            wide_chunk_tokens=max(candidates),
            candidate_chunk_tokens=candidates,
            granted=granted,
            granted_chunk_tokens=int(width) if granted else 0,
            growth_bytes=int(settled.get("growth_bytes") or 0),
            decided_by="prefill_admission",
        )
    if not granted:
        from mtplx.demotions import note

        note("qwen4_wide_prefill_chunk_refused", _WIDE_CHUNK_ADMISSION_REFUSED_REASON)
        return None
    return int(width)


def make_prefill_system_guard(
    state: Any,
    *,
    prompt_tokens: int,
    chunk_tokens: int | None,
    priced: Mapping[str, Any] | None,
    prompt_scoring: bool = False,
    own_session_shed: Any = None,
):
    from mtplx.server import openai as srv

    after_forward: dict[str, Any] = {}
    try:
        if prompt_scoring and priced is None:
            priced = prompt_scoring_growth(
                state, prompt_tokens=prompt_tokens, width=chunk_tokens
            )
        reserve = srv._prefill_chunk_reserve_bytes(
            state, prompt_tokens=prompt_tokens, chunk_tokens=chunk_tokens, priced=priced
        )
        after_forward = srv._prefill_after_forward_plan(
            state, prompt_tokens=prompt_tokens, chunk_tokens=chunk_tokens, priced=priced
        )
        if prompt_scoring:
            # Only completed scoring progress proves the head is finished.
            after_forward = {"after_prefill_reserve_bytes": 0, "forward_rows_bytes": None}
    except Exception as exc:  # noqa: BLE001
        # Keep generation's conservative reservation and visible degraded
        # health if pricing fails. The live guard still checks every chunk.
        from mtplx.memory_plan import RUNTIME_TRANSIENTS_BYTES

        reserve = int(RUNTIME_TRANSIENTS_BYTES)
        srv._note_guard_health(state, where="prefill_chunk_reserve", error=exc)
        event = {
            "action": "prefill_chunk_reserve_error",
            "error": repr(exc),
            "guard_degraded": True,
        }
        srv._record_guard_event(state, event)
        try:
            print("[mtplx] memory guard " + json.dumps(event), flush=True)
        except Exception:
            pass
    else:
        srv._note_guard_health(state, where="prefill_chunk_reserve", error=None)
    priced_growth = priced.get("growth_bytes") if isinstance(priced, Mapping) else None
    return srv._PrefillSystemGuard(
        state,
        chunk_reserve_bytes=reserve,
        own_session_shed=own_session_shed,
        priced_growth_bytes=(
            int(priced_growth) if isinstance(priced_growth, (int, float)) else None
        ),
        **after_forward,
    )


def score_prompt_with_memory_policy(
    state: Any,
    prompt_ids: list[int],
    *,
    top_k: int,
    request_observability: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score under the serving lock with generation's prefill safety policy."""
    from mtplx.generation import prefill_chunk_size_override
    from mtplx.server import openai as srv

    width = prompt_scoring_forward_widths(
        state.runtime, len(prompt_ids), getattr(state.args, "prefill_chunk_tokens", None)
    )[0]
    pricing: dict[str, Any] = {}
    admission = srv._prefill_admission_shed(
        state,
        prompt_ids=prompt_ids,
        session_bank=None,
        session_id=None,
        max_new_tokens=0,
        mtp_depth=0,
        prefill_chunk_tokens=width,
        pricing=pricing,
        prompt_scoring=True,
    )
    if admission is not None:
        if request_observability is not None:
            request_observability["prefill_admission_shed"] = admission
        if admission.get("refused"):
            raise srv._prefill_admission_refusal(state, admission)
        if admission.get("prefill_chunk_tokens") is not None:
            width = int(admission["prefill_chunk_tokens"])
    guard = make_prefill_system_guard(
        state, prompt_tokens=len(prompt_ids), chunk_tokens=width,
        priced=pricing.get("growth"),
        prompt_scoring=True,
    )

    def abort_check() -> bool:
        return srv._pressure_abort_requested(state) or guard()

    try:
        with prefill_chunk_size_override(width):
            scored = srv.score_prompt_logprobs(
                state.runtime, prompt_ids, top_k=top_k,
                abort_check=abort_check, prefill_callback=guard.note_prefill_progress,
            )
        scored["prefill_chunk_tokens"] = width
        return scored
    except srv.PostcommitAbort as abort:
        # The 507 below chains this abort; its traceback holds the scoring
        # frames and their logits (generation's own abort site does the same).
        abort.__traceback__ = None
        if guard.tripped is not None:
            if request_observability is not None:
                request_observability["prefill_system_abort"] = dict(guard.tripped)
            raise srv._prefill_system_abort_exception(state, guard.tripped)
        if srv._pressure_abort_requested(state):
            raise srv._allocation_failure_http_exception(
                state, RuntimeError("sustained critical memory pressure during prompt scoring")
            )
        raise
