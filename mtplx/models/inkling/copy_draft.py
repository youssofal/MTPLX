"""Prompt-copy proposals for Inkling's existing target-sampled verification loop."""

from collections import Counter, OrderedDict
from threading import Lock
from time import perf_counter

from mtplx import context_copy, copy_gate


class SessionPromptIndexes:
    """Hold at most two completed turns' public NgramIndexes per loaded runtime."""

    def __init__(self):
        self.kept = OrderedDict()
        self.lock = Lock()

    @classmethod
    def for_runtime(cls, runtime):
        indexes = getattr(runtime, "_inkling_prompt_indexes", None)
        if indexes is None:
            indexes = vars(runtime).setdefault("_inkling_prompt_indexes", cls())
        return indexes

    def take(self, session_id, prompt, ng_min, ng_max):
        key = (session_id, ng_min, ng_max)
        # Check out the whole entry. Concurrent requests never mutate the same index.
        with self.lock:
            entry = self.kept.pop(key, None)
        index, reused = None, 0
        if entry is not None:
            covered, existing = entry
            if len(covered) <= len(prompt) and prompt[: len(covered)] == covered:
                index, reused = existing, len(covered)
        if index is None:
            index = context_copy.NgramIndex(ng_min, ng_max)
        index.sync(prompt)
        return index, reused

    def give_back(self, session_id, prompt, index):
        key = (session_id, index.ng_min, index.ng_max)
        with self.lock:
            self.kept[key] = (prompt, index)
            self.kept.move_to_end(key)
            if len(self.kept) > 2:
                self.kept.popitem(last=False)


class PromptCopy:
    def __init__(
        self, prompt_ids, *, session_id, verify_rows, image_pad=None, indexes=None
    ):
        self.prompt = prompt_ids
        self.session_id = session_id
        self.image_pad = image_pad
        # The verify path's width: RC6's target and rollback path covers an anchor plus seven
        # proposals; the fast step verifies up to 32 rows with every row equal to the one-row
        # step (l0/receipts/wide-probe-2.json), so the loop passes its own width.
        self.max_proposals = verify_rows - 1
        self.ng_min = context_copy.context_copy_ng_min()
        self.ng_max = context_copy.context_copy_ng_max()
        self.min_ext = context_copy.context_copy_min_ext()
        self.block_cap = context_copy.context_copy_block_k()
        self.probation_cap = min(
            self.block_cap, context_copy.context_copy_probation_k()
        )
        self.disabled_reason = None
        if not context_copy.context_copy_enabled():
            self.disabled_reason = "disabled_by_setting"
        elif not context_copy.context_copy_batched_enabled():
            self.disabled_reason = "batched_copy_disabled"
        elif self.max_proposals < 1:
            self.disabled_reason = "no_verified_proposal_rows"
        elif len(prompt_ids) <= self.ng_min:
            self.disabled_reason = "no_prompt_continuation"
        # The public fuzzy matcher has a secondary corpus. Rebuild both together in that mode.
        self.indexes = (
            indexes
            if session_id
            and not (context_copy.ramp_enabled() and context_copy.ramp_fuzzy_enabled())
            else None
        )
        self.index = self.covered = None
        self.reused = self.probes = self.rounds = self.drafted = 0
        self.accepted = self.accepted_blocks = 0
        self.ema, self.seen, self.suspend_until, self.backoff = 0.5, 0, 0, 64
        self.suspensions = 0
        self.lookup_build_s = self.copy_round_s = self.ordinary_round_s = 0.0
        self.declines, self.limits, self.verify_rows = Counter(), Counter(), Counter()
        # The cost gate (mtplx/copy_gate.py, MTPLX_CONTEXT_COPY_GATE, off by default): it prices
        # this request's ordinary and copy rounds from their own times and vetoes or shortens a
        # block whose expected tokens a second lose. Class B: which rounds copy follows the clock.
        self.gate = copy_gate.for_request() if self.disabled_reason is None else None
        self.verify_calls = 0

    def decline(self, reason):
        self.declines[reason] += 1
        return None

    def propose(self, tokens, remaining, constraint=None):
        if remaining <= 0:
            return self.decline("requested_output_budget")
        if self.disabled_reason:
            return self.decline(self.disabled_reason)
        if len(tokens) < self.suspend_until:
            return self.decline("acceptance_backoff")
        if self.gate is not None:
            self.gate.cycle(len(tokens), self.verify_calls)
        if self.index is None:
            # Build after streaming the first target token. The corpus is the prompt only.
            start = perf_counter()
            if self.indexes is None:
                self.index = context_copy.NgramIndex(self.ng_min, self.ng_max)
                self.index.sync(self.prompt)
            else:
                self.covered = tuple(self.prompt)
                self.index, self.reused = self.indexes.take(
                    self.session_id, self.covered, self.ng_min, self.ng_max
                )
            self.lookup_build_s = perf_counter() - start
        self.probes += 1
        pos, ext = self.index.find(
            context_copy.JoinedTokens(self.prompt, tokens), max_pos=len(self.prompt)
        )
        if pos is None:
            return self.decline("no_prompt_match")
        if ext < self.min_ext:
            return self.decline("insufficient_match_extension")
        cap = (
            self.block_cap if self.seen >= 2 and self.ema >= 0.5 else self.probation_cap
        )
        length = context_copy.block_for_ext(ext, cap)
        if self.gate is not None:
            planned = self.gate.plan(length)
            if planned <= 0:
                return self.decline("cost_gate")
            if planned < length:
                self.limits["cost_gate"] += 1
            length = planned
        for reason, bound in (
            ("verify_geometry", self.max_proposals),
            ("requested_output_budget", remaining),
            ("end_of_prompt", len(self.prompt) - pos),
        ):
            if bound < length:
                self.limits[reason] += 1
                length = bound
        guesses = [int(t) for t in self.prompt[pos : pos + length]]
        if self.image_pad is not None and self.image_pad in guesses:
            guesses = guesses[: guesses.index(self.image_pad)]
            if not guesses:
                return self.decline("image_embedding_position")
            self.limits["image_embedding_position"] += 1
        if constraint is not None:
            validate = getattr(constraint, "validate_prefix", None)
            if not callable(validate):
                return self.decline("constraint_without_prefix_validator")
            legal = validate(guesses)
            if legal == 0:
                return self.decline("grammar_rejected_prefix")
            if legal < len(guesses):
                guesses = guesses[:legal]
                self.limits["grammar_prefix"] += 1
        return guesses

    def observe(self, offered, landed, emitted_tokens, wall_s):
        # Host wall covers selection, proposal, target verification and commit submission.
        # The stream callback and any later lazy-device work are outside both timers.
        self.verify_calls += 1
        if offered and self.gate is not None:
            self.gate.observe_copy(offered, landed)
        if not offered:
            self.ordinary_round_s += wall_s
            return
        self.copy_round_s += wall_s
        self.rounds += 1
        self.verify_rows[offered + 1] += 1
        self.drafted += offered
        self.accepted += landed
        self.accepted_blocks += int(landed > 0)
        # Public 2.12.3 acceptance probation and exponential retry. Timing is telemetry.
        share = landed / offered
        self.ema = 0.7 * self.ema + 0.3 * share
        self.seen += 1
        if share >= 0.5:
            self.backoff = 64
        if self.seen >= 3 and self.ema < 0.35:
            self.suspend_until = emitted_tokens + self.backoff
            self.backoff = min(self.backoff * 2, 4096)
            self.ema, self.seen = 0.5, 0
            self.suspensions += 1

    def _gate_costs(self):
        # The gate's measured prices, beside the lane's own timers (seconds; none when off).
        stats = self.gate.stats() if self.gate is not None else {}
        return {f"gate_{key[:-3]}_s": stats[key] / 1e3 for key in ("ordinary_ms", "copy_fixed_ms", "copy_row_ms")
                if key in stats}

    def finish(self, emitted_tokens):
        if self.index is not None and self.indexes is not None:
            self.indexes.give_back(self.session_id, self.covered, self.index)
        return dict(
            context_copy_active=self.disabled_reason is None,
            context_copy_disabled_reason=self.disabled_reason,
            context_copy_probes=self.probes,
            context_copy_rounds=self.rounds,
            context_copy_drafted_tokens=self.drafted,
            context_copy_accepted_tokens=self.accepted,
            context_copy_accepted_blocks=self.accepted_blocks,
            context_copy_index_reused_tokens=self.reused,
            context_copy_suspensions=self.suspensions,
            context_copy_suspended=emitted_tokens < self.suspend_until,
            context_copy_backoff_tokens=self.backoff if self.index is not None else 0,
            context_copy_declines=dict(self.declines),
            context_copy_limits=dict(self.limits),
            context_copy_verify_rows={
                str(width): count for width, count in self.verify_rows.items()
            },
            context_copy_timing_s={
                "index_build_or_extend": self.lookup_build_s,
                "copy_rounds": self.copy_round_s,
                "ordinary_rounds": self.ordinary_round_s,
                **self._gate_costs(),
            },
        )
