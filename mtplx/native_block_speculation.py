"""Generic physical-block greedy speculation for fixed-block proposal backends.

The backend owns only model-specific proposal/cache operations.  This module
owns the target protocol: first-token gating, primary-inclusive block
verification, target-cache rollback, callbacks, and common generation statistics.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol

import mlx.core as mx
import numpy as np

from .attention_context import attention_phase
from .sampling import SamplerConfig


class NativeBlockSpeculativeBackend(Protocol):
    """Construction-installed operations for one native block proposer."""

    backend_id: str
    supported_depths: tuple[int, ...]
    minimum_proposal_target_position: int
    prefill_chunk_size: int

    def make_cache(self, rt: Any) -> Any: ...

    def prefill(self, rt: Any, hidden: mx.array, cache: Any) -> None: ...

    def prefill_chunk(
        self,
        rt: Any,
        hidden: mx.array,
        cache: Any,
        *,
        start_pos: int,
    ) -> None: ...

    def propose(
        self,
        rt: Any,
        hidden: mx.array,
        token_id: int,
        primary_token_id: int,
        cache: Any,
        *,
        start_pos: int,
        width: int,
    ) -> mx.array: ...

    def commit(
        self,
        rt: Any,
        hidden: mx.array,
        cache: Any,
        *,
        start_pos: int,
    ) -> None: ...

    def cache_roots(self, cache: Any) -> list[mx.array]: ...

    def bind_target_forward(self, rt: Any) -> Callable[..., Any]: ...

    def snapshot(self, cache: Any) -> Any: ...

    def restore(self, cache: Any, snapshot: Any) -> None: ...

    def rollback_target(self, target_cache: Any, rejected_rows: int) -> None: ...


def _validate_request(
    backend: NativeBlockSpeculativeBackend,
    sampler: SamplerConfig,
    speculative_depth: int,
    *,
    draft_sampler: SamplerConfig | None,
    constraint: Any | None,
    vision_splice: Any | None,
    adaptive_policy: Any | None,
    adaptive_width_policy: Any | None,
) -> None:
    """Validate the fixed greedy lane once, before prompt execution."""
    if (
        float(sampler.temperature) > 0.0
        or float(sampler.presence_penalty) != 0.0
        or float(sampler.frequency_penalty) != 0.0
    ):
        raise ValueError(
            f"{backend.backend_id} currently requires greedy target sampling"
        )
    if draft_sampler is not None and (
        float(draft_sampler.temperature) > 0.0
        or float(draft_sampler.presence_penalty) != 0.0
        or float(draft_sampler.frequency_penalty) != 0.0
    ):
        raise ValueError(
            f"{backend.backend_id} currently requires greedy draft sampling"
        )
    if int(speculative_depth) not in backend.supported_depths:
        supported = ", ".join(str(value) for value in backend.supported_depths)
        raise ValueError(
            f"{backend.backend_id} measured proposal width must be one of: {supported}"
        )
    if constraint is not None:
        raise ValueError(
            f"{backend.backend_id} does not yet support constrained decoding"
        )
    if vision_splice is not None:
        raise ValueError(f"{backend.backend_id} does not support vision input")
    if adaptive_policy is not None or adaptive_width_policy is not None:
        raise ValueError(
            f"{backend.backend_id} width is fixed for each benchmark request"
        )


def generate_native_block_speculative(
    rt: Any,
    backend: NativeBlockSpeculativeBackend,
    prompt_ids: list[int],
    *,
    abort_check: Callable[[], bool] | None,
    max_tokens: int,
    sampler: SamplerConfig,
    speculative_depth: int,
    seed: int,
    stop_token_ids: set[int] | None,
    draft_sampler: SamplerConfig | None,
    token_callback: Callable[[list[int]], None] | None,
    prefill_callback: Callable[[dict[str, Any]], None] | None,
    constraint: Any | None,
    vision_splice: Any | None,
    adaptive_policy: Any | None,
    adaptive_width_policy: Any | None,
):
    """Run a construction-installed native block proposer against the target.

    Depth two means two genuinely future DSpark drafts.  The already sampled
    target primary plus those drafts are verified in one physical target call.
    Only the accepted prefix remains in the target and proposal caches.
    """
    from .generation import (
        GenerationOutput,
        GenerationStats,
        _attach_runtime_diagnostics,
        _decode,
        _default_stop_tokens,
        _eval,
        _finish_reason_from_tokens,
        _final_logits_prefill_enabled,
        _generation_rate_fields,
        _is_stop,
        _iter_prefill_chunk_spans,
        _make_target_prefill_cache,
        _mean_accept_probability_by_depth,
        _runtime_counter_snapshot,
        _strip_terminal_stop,
    )

    _validate_request(
        backend,
        sampler,
        speculative_depth,
        draft_sampler=draft_sampler,
        constraint=constraint,
        vision_splice=vision_splice,
        adaptive_policy=adaptive_policy,
        adaptive_width_policy=adaptive_width_policy,
    )
    # Bind the construction-certified target callable once.  The loop invokes
    # this exact callable for both M1 tail rows and physical verifier blocks.
    target_forward = backend.bind_target_forward(rt)
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")

    counter_start = _runtime_counter_snapshot(rt)
    stop_token_ids = (
        _default_stop_tokens(rt.tokenizer) if stop_token_ids is None else stop_token_ids
    )
    del seed
    started_all = time.perf_counter()
    prefill_started = started_all
    tokens: list[int] = []
    accepted_drafts = 0
    drafted_tokens = 0
    rejected_drafts = 0
    # With hot event dictionaries disabled, this is the aggregate count of
    # main-loop proposal/verification cycles.  It is the denominator consumed
    # by acceptance reporting, not a physical target-forward count.  It includes
    # terminal/tail cycles but excludes the one-time position seed.
    verify_calls = 0
    prompt_target_time = 0.0
    prompt_proposal_time = 0.0
    prompt_eval_time = 0.0
    accepted_by_depth = [0] * int(speculative_depth)
    drafted_by_depth = [0] * int(speculative_depth)
    target_position = len(prompt_ids)
    last_target_token = int(prompt_ids[-1])

    def finish() -> GenerationOutput:
        elapsed = max(0.0, time.perf_counter() - started_all)
        stats = GenerationStats(
            mode="mtpk",
            generated_tokens=len(tokens),
            elapsed_s=elapsed,
            **_generation_rate_fields(
                generated_tokens=len(tokens),
                elapsed_s=elapsed,
                prompt_eval_time_s=prompt_eval_time,
            ),
            accepted_drafts=accepted_drafts,
            rejected_drafts=rejected_drafts,
            drafted_tokens=drafted_tokens,
            verify_time_s=0.0,
            verify_forward_time_s=0.0,
            verify_eval_time_s=0.0,
            verify_joint_eval_time_s=0.0,
            draft_time_s=0.0,
            target_forward_time_s=0.0,
            prompt_eval_time_s=prompt_eval_time,
            prompt_tps=(
                len(prompt_ids) / prompt_eval_time if prompt_eval_time > 0 else 0.0
            ),
            prompt_target_prefill_time_s=prompt_target_time,
            prompt_mtp_history_time_s=prompt_proposal_time,
            prompt_target_prefill_tok_s=(
                len(prompt_ids) / prompt_target_time if prompt_target_time > 0 else 0.0
            ),
            accept_time_s=0.0,
            rollback_time_s=0.0,
            repair_time_s=0.0,
            commit_time_s=0.0,
            peak_memory_bytes=mx.get_peak_memory(),
            speculative_depth=int(speculative_depth),
            requested_speculative_depth=int(speculative_depth),
            accepted_by_depth=accepted_by_depth,
            drafted_by_depth=drafted_by_depth,
            mean_accept_probability_by_depth=_mean_accept_probability_by_depth(
                [float(value) for value in accepted_by_depth], drafted_by_depth
            ),
            verify_calls=verify_calls,
            events=[],
        )
        _attach_runtime_diagnostics(stats, rt, counter_start)
        return GenerationOutput(
            tokens=tokens,
            text=_decode(rt.tokenizer, _strip_terminal_stop(tokens, stop_token_ids)),
            stats=stats,
            finish_reason=_finish_reason_from_tokens(
                tokens, stop_token_ids=stop_token_ids, max_tokens=max_tokens
            ),
        )

    if prefill_callback is not None:
        try:
            prefill_callback(
                {
                    "phase": "started",
                    "tokens_done": 0,
                    "tokens_total": len(prompt_ids),
                    "cached_tokens": 0,
                    "new_prefill_tokens": len(prompt_ids),
                    "elapsed_s": 0.0,
                    "started_s": prefill_started,
                }
            )
        except Exception:
            pass

    target_cache = _make_target_prefill_cache(rt)
    proposal_cache = backend.make_cache(rt)
    logits = None
    current_hidden = None
    prompt_length = len(prompt_ids)
    body_length = prompt_length - 1
    final_logits_only = _final_logits_prefill_enabled()
    # Match AR's cache-only body geometry exactly, then run its mandatory
    # one-token logits/hidden tail.  DSpark retains its separately measured
    # 128-row ring geometry below.
    target_spans = [
        *_iter_prefill_chunk_spans(body_length),
        (body_length, prompt_length),
    ]
    proposal_chunk_size = int(backend.prefill_chunk_size)
    proposal_start = 0
    proposal_remainder = None

    def prefill_proposal_chunk(hidden: mx.array) -> bool:
        nonlocal prompt_eval_time, prompt_proposal_time, proposal_start
        if abort_check is not None and abort_check():
            return False
        proposal_chunk_started = time.perf_counter()
        backend.prefill_chunk(
            rt,
            hidden,
            proposal_cache,
            start_pos=proposal_start,
        )
        proposal_roots = backend.cache_roots(proposal_cache)
        if proposal_roots:
            _eval(*proposal_roots)
        prompt_proposal_time += max(0.0, time.perf_counter() - proposal_chunk_started)
        prompt_eval_time = prompt_target_time + prompt_proposal_time
        proposal_start += int(hidden.shape[1])
        return True

    for chunk_start, chunk_end in target_spans:
        if abort_check is not None and abort_check():
            return finish()
        final_chunk = chunk_end == prompt_length
        target_chunk_started = time.perf_counter()
        with attention_phase("prefill"):
            chunk_logits, chunk_hidden = target_forward(
                mx.array([prompt_ids[chunk_start:chunk_end]]),
                cache=target_cache,
                return_hidden=True,
                emit_logits=final_chunk or not final_logits_only,
                logits_keep=1 if final_chunk and final_logits_only else None,
            )
        if chunk_logits is not None:
            _eval(chunk_logits, chunk_hidden)
        else:
            _eval(chunk_hidden)
        prompt_target_time += max(0.0, time.perf_counter() - target_chunk_started)
        prompt_eval_time = prompt_target_time + prompt_proposal_time
        if final_chunk:
            logits = chunk_logits[:, -1, :]
            current_hidden = chunk_hidden[:, -1:]

        # Stream every complete proposal block now.  Only the cross-target-span
        # remainder is retained, so long prompts never accumulate a full hidden
        # sequence in the scheduler.
        hidden_offset = 0
        hidden_width = int(chunk_hidden.shape[1])
        if proposal_remainder is not None:
            needed = proposal_chunk_size - int(proposal_remainder.shape[1])
            taken = min(needed, hidden_width)
            proposal_remainder = mx.concatenate(
                [proposal_remainder, chunk_hidden[:, :taken]], axis=1
            )
            hidden_offset = taken
            if int(proposal_remainder.shape[1]) == proposal_chunk_size:
                if not prefill_proposal_chunk(proposal_remainder):
                    return finish()
                proposal_remainder = None
        while hidden_offset + proposal_chunk_size <= hidden_width:
            proposal_chunk = chunk_hidden[
                :, hidden_offset : hidden_offset + proposal_chunk_size
            ]
            if not prefill_proposal_chunk(proposal_chunk):
                return finish()
            hidden_offset += proposal_chunk_size
        if hidden_offset < hidden_width:
            proposal_remainder = chunk_hidden[:, hidden_offset:]
        if final_chunk and proposal_remainder is not None:
            if not prefill_proposal_chunk(proposal_remainder):
                return finish()
            proposal_remainder = None
    if prefill_callback is not None:
        try:
            prefill_elapsed = max(0.0, time.perf_counter() - prefill_started)
            compute_tok_s = (
                len(prompt_ids) / prompt_eval_time if prompt_eval_time > 0 else None
            )
            wall_tok_s = (
                len(prompt_ids) / prefill_elapsed if prefill_elapsed > 0 else None
            )
            prefill_callback(
                {
                    "phase": "completed",
                    "tokens_total": len(prompt_ids),
                    "new_prefill_tokens": len(prompt_ids),
                    "cached_tokens": 0,
                    "elapsed_s": prefill_elapsed,
                    "prompt_eval_time_s": prompt_eval_time,
                    "prefill_tok_s": (
                        compute_tok_s if compute_tok_s is not None else wall_tok_s
                    ),
                    "prefill_compute_tok_s": compute_tok_s,
                    "prefill_wall_tok_s": wall_tok_s,
                    "cache_hit": False,
                }
            )
        except Exception:
            pass

    def emit(block: list[int]) -> None:
        if token_callback is None:
            return
        visible = [token for token in block if not _is_stop(token, stop_token_ids)]
        if visible:
            token_callback(visible)

    # Some native proposers overload absolute position zero as their prefill
    # signal.  Prime such a backend once with ordinary target M1 so every
    # later proposal sees the real position of its carried hidden.  This is a
    # construction-selected initial phase, outside the measured fixed-K loop.
    minimum_proposal_position = int(
        getattr(backend, "minimum_proposal_target_position", 1)
    )
    seed_terminal = False
    while (
        target_position < minimum_proposal_position
        and len(tokens) < max_tokens
        and not seed_terminal
    ):
        if abort_check is not None and abort_check():
            break
        current_top = int(mx.argmax(logits[0], axis=-1).item())
        with attention_phase("ar_decode"):
            seed_logits, seed_hidden = target_forward(
                mx.array([[current_top]]), cache=target_cache, return_hidden=True
            )
        _eval(seed_logits, seed_hidden)
        current_hidden = seed_hidden[:, -1:]
        backend.commit(rt, current_hidden, proposal_cache, start_pos=target_position)
        roots = backend.cache_roots(proposal_cache)
        if roots:
            _eval(*roots)
        logits = seed_logits[:, -1, :]
        last_target_token = current_top
        tokens.append(current_top)
        emit([current_top])
        target_position += 1
        seed_terminal = _is_stop(current_top, stop_token_ids)

    while len(tokens) < max_tokens and not seed_terminal:
        if abort_check is not None and abort_check():
            break
        # The primary is sampled from carried target logits and is therefore
        # authoritative, never a DSpark acceptance gate.  DSpark proposes K
        # future rows, then target verification executes [primary, drafts] in
        # one physical block and rolls back its rejected suffix.
        current_top = int(mx.argmax(logits[0], axis=-1).item())
        remaining = max_tokens - len(tokens)
        width = min(int(speculative_depth) + 1, remaining)
        if _is_stop(current_top, stop_token_ids):
            width = 1
        proposal_snapshot = None
        future_tokens: list[int] = []
        if width > 1:
            proposal_snapshot = backend.snapshot(proposal_cache)
            future = backend.propose(
                rt,
                current_hidden,
                last_target_token,
                current_top,
                proposal_cache,
                start_pos=target_position,
                width=width - 1,
            )
            _eval(future)
            future_tokens = [int(token) for token in np.asarray(future)[0]]
            for index, token in enumerate(future_tokens):
                drafted_by_depth[index] += 1
                if _is_stop(token, stop_token_ids):
                    future_tokens = future_tokens[: index + 1]
                    width = index + 2
                    break
            drafted_tokens += len(future_tokens)
        proposal_tokens = [current_top, *future_tokens]
        proposed = mx.array([proposal_tokens], dtype=mx.int32)
        phase = "decode_verify" if width > 1 else "ar_decode"
        with attention_phase(phase):
            verify_logits, verify_hidden = target_forward(
                proposed,
                cache=target_cache,
                return_hidden=True,
            )
        _eval(verify_logits, verify_hidden)
        verify_calls += 1

        accepted = 1
        for index in range(1, width):
            previous_top = int(mx.argmax(verify_logits[0, index - 1], axis=-1).item())
            if proposal_tokens[index] != previous_top:
                break
            accepted += 1
            if _is_stop(proposal_tokens[index], stop_token_ids):
                break

        rejected_suffix = width - accepted
        if rejected_suffix:
            backend.rollback_target(target_cache, rejected_suffix)

        accepted_drafts += max(0, accepted - 1)
        rejected_drafts += rejected_suffix
        for index in range(1, accepted):
            accepted_by_depth[index - 1] += 1

        # Proposal cache state is backend-owned.  Discard its speculative state
        # and install only the target-verified causal prefix in one commit.
        if proposal_snapshot is not None:
            backend.restore(proposal_cache, proposal_snapshot)
        backend.commit(
            rt,
            verify_hidden[:, :accepted],
            proposal_cache,
            start_pos=target_position,
        )
        roots = backend.cache_roots(proposal_cache)
        if roots:
            _eval(*roots)
        logits = verify_logits[:, accepted - 1, :]
        current_hidden = verify_hidden[:, accepted - 1 : accepted]

        committed = proposal_tokens[:accepted]
        tokens.extend(committed)
        emit(committed)
        last_target_token = committed[-1]
        target_position += accepted
        if _is_stop(committed[-1], stop_token_ids):
            break

    return finish()
