"""Target-sampled prefix verification. Only committed tokens advance either model's history."""

from collections import Counter
import hashlib
import os
import secrets
from time import perf_counter
import mlx.core as mx
import numpy as np

from mtplx.generation import (
    ClientControlledDraw,
    GenerationFinalState,
    GenerationOutput,
    GenerationStats,
    _commit_ar_prompt_state,
    _default_stop_tokens,
    _eval as settle_owner_work,
    _finish_reason_from_tokens,
    _prefill_chunk_size,
    _sample_from_logits,
    first_token_logprobs,
    restore_or_prefill_prompt_state,
)
from mtplx.route_tape import RouteTape, counter_deltas
from mtplx.sampling import client_controls
from .copy_draft import PromptCopy, SessionPromptIndexes
from .customer_controls import ProposalSampler, RepetitionOutput
from .fast import draft as fast_draft
from .fast import engine_site
from .fast.decoder import fast_enabled, stage_spec
from .fast.step import wide_rows_exact
from .fast.state import MAX_PANEL
from .fast.routes import count
from .fast import round as fast_round
from .fast import round_program
from .protocol import completion_text
from .sampling import draw_native, is_native_sampler
from mtplx.token_ids import token_id_bytes


def served_proposals(device_ok, draft_temperature, proposals, dspark_tokens):
    """The proposal method, temperature and sampled count of the route that served them."""
    if not device_ok:
        return {"proposal_route": "host_" + ("sampled" if proposals.sampled else "argmax")}
    sampled = draft_temperature > 0
    return {"proposal_route": "device_" + ("sampled" if sampled else "argmax"),
            "method": "sampled" if sampled else "argmax", "temperature": draft_temperature,
            "sampled_tokens": dspark_tokens if sampled else 0}


def decision_route(sampler, *, constraint=None, draft_sampler=None, verify_strategy=None):
    """``(True, None)`` when the GPU decides the rounds and draws the first token, else
    ``(False, reason)`` for the host decision (RC6's draws), which every request can take.

    Client controls (logit_bias, min_p, repetition_penalty) ride the device round: they are
    applied to the rows it decides from (``ClientControlledDraw.rows``), so the ratio rule and
    the drafter's sampled proposals serve those requests too."""
    if not fast_enabled():
        # MTPLX_INKLING_FAST=off restores RC6's whole path: its forward and its host decision.
        return False, "fast_path_off"
    if verify_strategy == "target_prefix":
        return False, "target_prefix"
    return fast_round.supported(sampler, constraint=constraint, draft_sampler=draft_sampler)


def checked_device_draw(row_logits, *, position, seed, temperature, controls=None):
    """The GPU's draw of one token from a logits row, refused like every host draw when the row
    has NaN, or no finite maximum (+inf, or every entry -inf): one read gives the token and the
    row's status, and a bad row raises the existing numerical error before the id is counted,
    streamed, embedded or banked (review 1, finding 3). ``controls`` (the request's
    ``ClientControlledDraw``) shapes the row first; the error's census reads the model's row."""
    row = row_logits.reshape(1, -1)
    law = controls.rows(row) if controls is not None else row
    drawn = fast_round.draw_checked(law, position=position, seed=seed, temperature=temperature)
    return fast_round.read_draw(drawn, rows=row, settle=settle_owner_work)[0], True


def generate(
    rt,
    prompt_ids,
    *,
    max_tokens,
    sampler,
    depth,
    seed,
    stop_token_ids,
    token_callback,
    prefill_callback,
    session_bank,
    session_id,
    session_restore_mode,
    session_template_hash,
    session_draft_head_identity,
    session_policy_fingerprint,
    capture_final_state,
    commit_prompt_state_to_bank,
    vision_splice,
    constraint,
    abort_check,
    first_token_logprobs_top_k,
    store_prefix_snapshot,
    draft_sampler=None,
    repetition_stop=False,
    verify_strategy=None,
):
    if rt.drafter is None:
        raise RuntimeError("DSpark requires the Inkling draft weights")
    # depth None (no explicit depth from the operator or the client) = the per-round choice
    # by measured yield; an explicit depth is kept exactly.
    if depth is not None and not 1 <= depth < rt.drafter.width:
        raise ValueError(f"DSpark depth must be between 1 and {rt.drafter.width - 1}")
    start = perf_counter()
    prompt = restore_or_prefill_prompt_state(
        rt,
        prompt_ids,
        mtp_history_policy="cycle",
        session_bank=session_bank,
        session_id=session_id,
        restore_mode=session_restore_mode,
        template_hash=session_template_hash,
        draft_head_identity=session_draft_head_identity,
        policy_fingerprint=session_policy_fingerprint,
        prefill_callback=prefill_callback,
        capture_hidden=session_bank is not None,
        store_prefix_snapshot=store_prefix_snapshot,
        vision_splice=vision_splice,
        abort_check=abort_check,
    )
    prompt_commit = {}
    if (
        commit_prompt_state_to_bank
        and session_bank is not None
        and session_id
        and prompt.suffix_tokens
        and not prompt.prefill_store_snapshot.get("stored")
    ):
        prompt_commit = _commit_ar_prompt_state(
            rt,
            prompt_ids,
            prompt,
            session_bank=session_bank,
            session_id=session_id,
            template_hash=session_template_hash,
            draft_head_identity=session_draft_head_identity,
            policy_fingerprint=session_policy_fingerprint,
            vision_splice=vision_splice,
        )
    cache, logits, hidden = prompt.trunk_cache, prompt.logits, prompt.hidden
    rng = np.random.default_rng(seed)
    # logit_bias / repetition_penalty / min_p on every row the request draws from: the device
    # round's rows before its decision (ClientControlledDraw.rows) and the host draw's alike;
    # every emitted token is observed for the repetition penalty, whichever route drew it.
    controls = ClientControlledDraw(sampler, prompt_ids) if client_controls(sampler) else None
    native_draw = (os.environ.get("MTPLX_INKLING_NATIVE_DRAW", "candidate") == "candidate"
                   and is_native_sampler(sampler) and controls is None)
    # The round is decided on the GPU by the product's ratio law (class B) unless the request
    # asks for the seed-match rule (verify strategy target_prefix: RC6's host FP64 draw) or
    # its sampling law needs the host (grammar, penalties, top-p/top-k): one small read a round.
    device_ok, device_decline = decision_route(sampler, constraint=constraint, draft_sampler=draft_sampler,
                                               verify_strategy=verify_strategy)
    count("decision.device" if device_ok else f"decision.host.{device_decline}")
    if device_ok and controls is not None:
        count("decision.device.client_controls")
    # The one-program round has no input for a client's logit transform yet: those rounds run
    # from the pieces below, the controls applied to the verified rows (counted per round).
    program_decline = "client_controls" if controls is not None else None
    # MTPLX_INKLING_ROUND_PROGRAM=0 keeps every round on the replayed step and the pieces around
    # it (a diagnostic rollback; round.decline.* counts every round the program does not serve).
    round_program_on = os.environ.get("MTPLX_INKLING_ROUND_PROGRAM", "on").strip().lower() not in (
        "0", "off", "false", "no")
    round_seed = int(seed) if seed is not None else secrets.randbits(64)
    target_temperature = float(sampler.temperature or 0.0)
    draft_temperature = float(draft_sampler.temperature) if draft_sampler is not None else target_temperature
    device_rounds = device_fallback_rounds = host_rounds = 0
    # No configured depth: the proposal count of each round is chosen before its draws from
    # the yield of earlier rounds and this Mac's round costs (timed at load; RC6's table when
    # the fast path is off), so every round stays exact.
    depth_control = (fast_draft.DepthByYield(cost_ms=getattr(rt, "round_cost_ms", None) or fast_draft.RC6_COST_MS_2K,
                                             max_depth=rt.drafter.width - 1)
                     if depth is None else None)
    stop = (
        _default_stop_tokens(rt.tokenizer) if stop_token_ids is None else stop_token_ids
    )
    tokens, counts = [], Counter()
    proposals = ProposalSampler(draft_sampler, seed)
    repetition = (
        RepetitionOutput(token_callback, stop) if repetition_stop and constraint is None else None
    )
    copy = PromptCopy(
        # The fast step verifies up to 32 rows, every row equal to the one-row step, so a long
        # copied run is verified in one round; RC6's path, and a pack whose stage A or MLP read
        # still takes the reference product on some visit, keep the drafter's 8-row block.
        prompt_ids, session_id=session_id,
        verify_rows=(MAX_PANEL if fast_enabled() and getattr(rt, "fast", None) is not None
                     and wide_rows_exact(rt.model, stage_spec()) else rt.drafter.width),
        image_pad=getattr(vision_splice, "image_pad_token_id", None),
        indexes=SessionPromptIndexes.for_runtime(rt) if session_id else None,
    )
    accepted = drafted = rounds = rejects = 0
    device_proposal_rounds = 0
    draft_s = verify_s = 0.0
    first_logprobs = None

    tape = RouteTape()
    eval_rows, commit = settle_owner_work, rt.commit
    if tape.enabled:
        tape.emit("route", "header", None, {
            "verify_core": "InklingRuntime.verify", "commit_core": "InklingRuntime.commit",
            "speculative_depth": depth if depth is not None else "auto", "max_tokens": int(max_tokens),
            "prompt_tokens": len(prompt_ids), "cached_tokens": prompt.cached_tokens,
            "new_prefill_tokens": prompt.suffix_tokens, "cache_source": prompt.cache_source,
            "mtp_history_policy": "cycle", "cache_class": type(cache[0]).__name__ if cache else None,
            "target_draw": ("device_ratio" if device_ok else ("draw_native" if native_draw else "_sample_from_logits")),
            "device_round_decline": device_decline,
            "draft_clock": "host_submission", "verify_clock": "includes_pending_draft",
            "vision_splice": vision_splice is not None, "constraint_active": constraint is not None,
            "sampler": {"temperature": float(sampler.temperature), "top_p": float(sampler.top_p),
                        "top_k": int(sampler.top_k)},
        })
        verify_submitted = scores_ready = 0.0
        previous_declines = {}

        def eval_rows(rows):
            nonlocal verify_submitted, scores_ready
            verify_submitted = perf_counter()
            settle_owner_work(rows)
            scores_ready = perf_counter()

        def commit(cache, features, tapes, kept, total):
            commit_start = perf_counter()
            rt.commit(cache, features, tapes, kept, total)
            committed = perf_counter()
            # A stop or cancellation can discard unexamined rows without rejecting a draft.
            offered, matched = total - 1, kept - 1
            rejected = kept if min(len(emitted), offered) > matched else None
            slots = sum(len(tape_visit) for tape_visit in (tapes or ()))
            declines = counter_deltas(previous_declines, copy.declines)
            previous_declines.update(copy.declines)
            tape.emit("route", "round", rounds - 1, {
                "complete": True, "verify_route": "inkling_block_verify",
                "verify_core": "InklingRuntime.verify", "commit_core": "InklingRuntime.commit",
                "proposal_source": "prompt_copy" if copied is not None else "dspark",
                "draft_core": "PromptCopy.propose" if copied is not None else "Drafter.propose",
                "verify_width": total, "accepted_depths": matched, "rejected_at_depth": rejected,
                "drafts": [
                    {"depth": i, "accepted": (i <= matched) if i <= len(emitted) else None,
                     "accept_probability": None}
                    for i in range(1, total)
                ],
                "bonus_token": len(emitted) > offered, "target_draws": len(emitted),
                "commit_route": "capture_commit" if kept < total else "verify_retained",
                "committed_rows": kept, "kv_tail_rows_trimmed": total - kept,
                "conv_tape_slots": slots, "conv_slots_restored": slots if kept < total else 0,
                "draft_history": "Drafter.observe",
                # How the verify step was served: replayed natively, recording, checking, the
                # MLX graph or declined, and why (per-round attribution for the Pulse rows).
                "engine": engine_site.last_route(),
                # Whether the whole round ran as one native program (lane L7), and why not.
                "round": ({"route": "declined", "reason": program_decline, "proposals": total - 1}
                          if program_decline and copied is None and device_ok else round_program.last_route()),
                "fallback_deltas": {"prompt_copy_decline": declines} if declines else {},
                # Host seconds. Verify includes the existing logits synchronization.
                # Commit ends on return from the existing cache update and draft observation.
                "timing_s": {"proposal": proposal_s, "verify_forward": verify_submitted - verify_started,
                             "verify_logits_eval": scores_ready - verify_submitted,
                             "target_draw_accept": commit_start - scores_ready,
                             "commit": committed - commit_start},
            })

    def draw(row):
        if constraint is not None:
            row = constraint.mask_logits_row(row)
        if native_draw:
            token = draw_native(row, rng)
        elif controls is not None:
            token = controls.draw(row, rng, token_counts=counts)
        else:
            token, _ = _sample_from_logits(row, sampler, rng, token_counts=counts)
        if constraint is not None:
            constraint.advance(token)
        counts[token] += 1
        return token

    def stream(batch):
        if repetition is not None:
            repetition.stream(tokens)
        elif token_callback:
            visible = [token for token in batch if token not in stop]
            if visible:
                token_callback(visible)

    def device_draw(row_logits, position):
        token, valid = checked_device_draw(row_logits, position=position, seed=round_seed,
                                           temperature=target_temperature, controls=controls)
        counts[token] += 1
        if controls is not None:
            controls.observe(token)
        return token

    decode_start = perf_counter()
    if max_tokens and not (abort_check and abort_check()):
        row = logits.reshape(-1, logits.shape[-1])[-1]
        tokens.append(device_draw(row, len(prompt_ids)) if device_ok else draw(row))
        if first_token_logprobs_top_k is not None:
            first_logprobs = first_token_logprobs(
                row, token_id=tokens[0], top_k=first_token_logprobs_top_k
            )
        stream(tokens)
    pending = bool(tokens)
    while (
        pending
        and (repetition is None or repetition.result is None)
        and len(tokens) < max_tokens
        and tokens[-1] not in stop
        and not (constraint and constraint.stopped)
    ):
        if abort_check and abort_check():
            break
        anchor = tokens[-1]
        round_started = perf_counter()
        remaining = max_tokens - len(tokens)
        copied = copy.propose(tokens, remaining, constraint)
        selected_depth = depth_control.choose(remaining) if depth_control is not None else depth
        if copied is None and selected_depth == 0:
            result = rt.forward_ar(mx.array([[anchor]]), cache=cache, logits_keep=1,
                                   return_hidden=capture_final_state)
            if capture_final_state:
                logits, hidden = result
            else:
                logits = result
            # Keep the draft history current even on a long run of single-token steps.
            settle_owner_work(logits, [entry.state for entry in cache[rt.trunk_entries:]])
            pending = False
            if abort_check and abort_check():
                break
            last_row = logits.reshape(-1, logits.shape[-1])[-1]
            token = (device_draw(last_row, len(prompt_ids) + len(tokens)) if device_ok
                     else draw(last_row))
            tokens.append(token)
            pending = True
            stream([token])
            cost = perf_counter() - round_started
            copy.observe(0, 0, len(tokens), cost)
            if depth_control is not None:
                depth_control.observe(0, 0, [token])
            continue
        position = len(prompt_ids) + len(tokens)
        law = served = None
        if copied is None and device_ok and round_program_on and program_decline:
            count(f"round.decline.{program_decline}")
        elif copied is None and device_ok and round_program_on:
            # The whole round (proposals, verify and decision) as one native program where one
            # is live (lane L7); otherwise the same round from the pieces below, same law.
            served = round_program.serve(rt, anchor, cache, min(selected_depth, remaining),
                                         position=position, seed=round_seed, temperature=target_temperature,
                                         draft_temperature=draft_temperature)
        if served is not None:
            block = proposed = served.block
            device_proposal_rounds += 1
        elif copied is None and device_ok:
            # Proposals drawn on the GPU from the drafter's soft-capped law at the draft
            # temperature; the law comes back for the acceptance test.
            proposed, law = fast_draft.propose(
                rt.drafter, rt.model, anchor, cache[rt.trunk_entries :], min(selected_depth, remaining),
                position=position, seed=round_seed, temperature=draft_temperature,
            )
            block = mx.concatenate((mx.array([anchor], mx.int32), proposed))
            mx.async_eval(block)
            device_proposal_rounds += 1
        elif copied is None:
            proposed = rt.drafter.propose(
                rt.model, anchor, cache[rt.trunk_entries :], min(selected_depth, remaining),
                **proposals.options(counts),
            )
            block = mx.concatenate((mx.array([anchor]), proposed.astype(mx.int32)))
            # Verification can build its graph while the GPU finishes the proposals.
            mx.async_eval(block)
            device_proposal_rounds += 1
        else:
            block = [anchor, *copied]
            proposed = mx.array(copied, mx.int32)
        drafted += len(block) - 1
        proposal_s = perf_counter() - round_started
        draft_s += proposal_s
        verify_started = perf_counter()
        if served is not None:
            rows, features, tapes = served.rows, served.features, served.pending
        else:
            rows, features, tapes = rt.verify(block, cache)
        verified_rows = rows
        decision = None
        width = len(block) if isinstance(block, list) else int(block.shape[0])
        if device_ok and width > fast_round.MAX_ROWS:
            # A copy round wider than the device decision's 8 rows: the host decides it
            # (exact under the seed-match rule; counted), the step stays fast.
            count("decision.host.wide_copy")
        elif device_ok:
            packet = (served.packet if served is not None else
                      fast_round.decide(controls.rows(rows, proposed) if controls is not None else rows,
                                        proposed, position=position, seed=round_seed,
                                        temperature=target_temperature, draft=law))
            # The round's only read; the owner's eval keeps the stall watchdog ticking.
            decision = fast_round.read_packet(packet, width, settle=eval_rows)
            if decision.status:
                decision = None  # non-finite rows: the host draw below raises the product's error
                device_fallback_rounds += 1
            else:
                device_rounds += 1
        if decision is None:
            host_rounds += 1
            eval_rows(rows)
            guesses = block.tolist()[1:] if copied is None else copied
        else:
            guesses = decision.tokens[: decision.accepted]
        rounds += 1
        emitted, kept = [], 1
        pending = False
        if decision is not None:
            offered = width - 1
            for pos, token in enumerate(decision.tokens):
                if len(tokens) + len(emitted) == max_tokens or (abort_check and abort_check()):
                    break
                counts[token] += 1
                if controls is not None:
                    controls.observe(token)
                emitted.append(token)
                repeated = repetition is not None and repetition.observe(tokens, emitted) is not None
                if pos < decision.accepted:
                    accepted += 1
                    kept += 1
                else:
                    rejects += int(pos < offered)
                    pending = True
                    break
                if repeated or token in stop:
                    break
            rows = None
        # Emit the target draw on both branches. A match only lets us read the next verified row.
        for pos, row in enumerate(rows if rows is not None else ()):
            if len(tokens) + len(emitted) == max_tokens or (
                abort_check and abort_check()
            ):
                break
            token = draw(row)
            emitted.append(token)
            repeated = repetition is not None and repetition.observe(tokens, emitted) is not None
            if pos < len(guesses) and guesses[pos] == token:
                accepted += 1
                kept += 1
            else:
                rejects += int(pos < len(guesses))
                pending = True
                break
            if repeated or token in stop or (constraint and constraint.stopped):
                break
        commit(cache, features, tapes, kept, len(block))
        if rows is not None:
            logits = rows[kept - 1][None, None]
        else:
            logits = verified_rows[kept - 1][None, None]
        if capture_final_state and not pending:
            hidden = mx.concatenate([h[:, kept - 1 : kept] for h in features], axis=-1)
        tokens.extend(emitted)
        copy.observe(len(copied) if copied is not None else 0, kept - 1, len(tokens), perf_counter()-round_started)
        if depth_control is not None:
            if copied is None:
                depth_control.observe(len(block) - 1, kept - 1, emitted)
            else:
                depth_control.observe(0, 0, emitted)  # copy rounds only move the channel
        stream(emitted)
        verify_s += perf_counter() - verify_started
    if repetition is not None:
        repetition.stream(tokens, flush=True)
    control_stats = repetition.stats() if repetition is not None else {}
    repeated = bool(control_stats)
    if constraint is not None and constraint.stopped and tokens and tokens[-1] not in stop:
        control_stats["finish_stop_origin"] = "grammar_terminal"
    finish = (
        "stop"
        if repeated or (constraint is not None and constraint.stopped)
        else _finish_reason_from_tokens(
            tokens, stop_token_ids=stop, max_tokens=max_tokens
        )
    )
    final = None
    if capture_final_state and tokens and not repeated:
        if pending:
            logits, hidden = rt.forward_ar(
                mx.array([[tokens[-1]]]), cache=cache, logits_keep=1, return_hidden=True
            )
        settle_owner_work(logits, hidden)
        final = GenerationFinalState(
            cache,
            logits,
            hidden,
            None,
            tuple(tokens),
            True,
            finish,
            prompt_gdn_boundaries=tuple(prompt.gdn_boundaries or ()),
        )
    decode_s, elapsed = perf_counter() - decode_start, perf_counter() - start
    stats = GenerationStats(
        mode="mtp",
        generated_tokens=len(tokens),
        elapsed_s=elapsed,
        tok_s=len(tokens) / max(elapsed, 1e-9),
        decode_elapsed_s=decode_s,
        decode_tok_s=len(tokens) / max(decode_s, 1e-9),
        end_to_end_tok_s=len(tokens) / max(elapsed, 1e-9),
        runtime_mtp_enabled=True,
        draft_head_installed=True,
        drafted_tokens=drafted,
        accepted_drafts=accepted,
        rejected_drafts=rejects,
        draft_time_s=draft_s,
        verify_time_s=verify_s,
        verify_calls=rounds,
        speculative_depth=depth if depth is not None else 0,
        prompt_eval_time_s=prompt.prompt_eval_time_s,
        prompt_tps=prompt.suffix_tokens / max(prompt.prompt_eval_time_s, 1e-9),
        cached_tokens=prompt.cached_tokens,
        new_prefill_tokens=prompt.suffix_tokens,
        prefill_chunk_size=_prefill_chunk_size(),
        session_cache_hit=prompt.cache_hit,
        cache_source=prompt.cache_source,
        ssd_cache_hit=prompt.ssd_cache_hit,
        ssd_cached_tokens=prompt.ssd_cached_tokens,
        ssd_restore_s=prompt.ssd_restore_s,
        ssd_suffix_tokens=prompt.suffix_tokens if prompt.ssd_cache_hit else 0,
        cache_restore_time_s=prompt.cache_restore_time_s,
        cache_miss_reason=prompt.cache_miss_reason,
        session_restore_mode=prompt.restore_mode,
        session_prompt_prefix_bank_commit=prompt_commit,
        session_prefill_store=prompt.prefill_store_snapshot,
        session_restore_served=prompt.restore_served,
        inkling_target_draw=("device_ratio" if device_ok else ("native_full_vocabulary" if native_draw else "stock")),
        inkling_output_ids_sha256=(
            hashlib.sha256(token_id_bytes(tokens)).hexdigest()
            if os.environ.get("MTPLX_INKLING_RECEIPT_TOKEN_HASH") == "1" else None
        ),
        inkling_draft_sampling={
            **proposals.stats(drafted - copy.drafted),
            # What served the drafter's proposals: on the GPU at the draft temperature when the
            # device round decides, else the host proposal sampler (argmax unless one is set).
            **served_proposals(device_ok, draft_temperature, proposals, drafted - copy.drafted),
            "device_proposal_rounds": device_proposal_rounds,
            "host_copy_rounds": rounds - device_proposal_rounds,
            "depth_choices": dict(Counter(depth_control.choices)) if depth_control is not None else None,
            # Who decided each round: the GPU (one small read) or the host, and why.
            "device_round": {"rounds": device_rounds, "host_rounds": host_rounds,
                             "host_fallback_rounds": device_fallback_rounds,
                             "decline": device_decline, "rule": "ratio" if device_ok else "seed_match",
                             "client_controls": list(controls.names) if controls is not None else [],
                             "proposals": "sampled" if device_ok and draft_temperature > 0 else "argmax",
                             "draft_temperature": draft_temperature if device_ok else None},
            "draft_clock": "host_submission",
            "verify_clock": "includes_pending_draft",
        },
        constraint_active=constraint is not None,
        constraint_completed=getattr(constraint, "completed", None),
        constraint_masked_steps=getattr(constraint, "masked_steps", 0),
        constraint_mask_time_s=getattr(constraint, "mask_time_s", 0.0),
        **control_stats,
        **copy.finish(len(tokens)),
    )
    return GenerationOutput(
        tokens,
        completion_text(rt.tokenizer, [t for t in tokens if t not in stop]),
        stats,
        final,
        finish,
        first_logprobs,
    )
