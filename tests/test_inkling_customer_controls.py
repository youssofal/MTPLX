"""Product dispatch, stock stop policy and independent Markov proposal sampling."""

from collections import Counter
import os
from threading import Lock
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from mtplx import generation as api
from mtplx.models.inkling import generation as inkling
from mtplx.models.inkling.draft import Drafter
from mtplx.sampling import SamplerConfig
from mtplx.server import openai as server


GREEDY = SamplerConfig(temperature=0, top_p=1, top_k=0)
NATIVE = SamplerConfig(temperature=1, top_p=1, top_k=0, presence_penalty=.2, frequency_penalty=.12)
DRAFT = SamplerConfig(temperature=.8, top_p=.9, top_k=5)


class CausalRows:
    """Exact lookup rows plus an append/trim cache. Runs without model weights."""

    backend_id = "inkling2_loop"
    mtp_enabled = True
    trunk_entries = 1
    tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda _: 100, decode=lambda ids: str(ids))

    def __init__(self, next_id, vocab):
        self.next_id, self.vocab = next_id, vocab
        self.model = self
        self.drafter = SimpleNamespace(width=8, propose=self.propose)
        self.guesses = []

    def scores(self, token):
        return mx.array([1.2 if i == self.next_id(token) else -.06 * i for i in range(self.vocab)])

    def propose(self, target, anchor, cache, count, *, select_token=None):
        guesses = []
        for _ in range(count):
            anchor = select_token(self.scores(anchor)) if select_token else self.next_id(anchor)
            guesses.append(anchor)
        self.guesses.extend(guesses)
        return mx.array(guesses)

    def verify(self, block, cache):
        cache[0].extend(block)
        return mx.stack([self.scores(t) for t in block]), [mx.array(block)[None, :, None]], None

    def commit(self, cache, features, tapes, kept, total):
        if kept < total:
            del cache[0][kept - total:]

    def forward_ar(self, ids, *, cache, **_):
        token = int(ids.item())
        cache[0].append(token)
        return self.scores(token)[None, None], ids[..., None]

    def prefill(self, rt, ids, **_):
        return api.PromptState([list(ids)], self.scores(ids[-1])[None, None], None, None, tuple(ids), 0)


class InklingCustomerControls(TestCase):
    def setUp(self):
        self.enterContext(mx.stream(mx.cpu))
        self.enterContext(patch.dict(os.environ, {
            "MTPLX_CONTEXT_COPY": "0",
            "MTPLX_REPETITION_STREAM_HOLDBACK": "strict",
            "MTPLX_REPETITION_STOP_MIN_TOKENS": "10",
            "MTPLX_REPETITION_STOP_MIN_REPEATED_TOKENS": "8",
            "MTPLX_REPETITION_STOP_MAX_BLOCK_TOKENS": "1",
            "MTPLX_REPETITION_STOP_MAX_CYCLE_TOKENS": "96",
            "MTPLX_REPETITION_STOP_MIN_CYCLE_COPIES": "3",
            "MTPLX_REPETITION_STOP_MIN_CYCLE_SPAN_TOKENS": "90",
        }))
        self.rt = CausalRows(lambda t: max(1, t - 1), 8)
        self.stream = []

    def decode(self, *, depth=7, limit=26, target=GREEDY, seed=37, **controls):
        with patch.object(inkling, "restore_or_prefill_prompt_state", self.rt.prefill):
            return api.generate_mtpk(
                self.rt, [3], max_tokens=limit, sampler=target, speculative_depth=depth,
                seed=seed, token_callback=self.stream.extend, capture_final_state=True, **controls,
            )

    def test_short_repetition_trim_and_bank_state(self):
        out = self.decode(repetition_stop=True)
        self.assertEqual((out.finish_reason, out.tokens, self.stream), ("stop", [2], [2]))
        self.assertIsNone(out.final_state)
        stats = out.stats
        self.assertEqual((stats.finish_stop_origin, stats.repetition_stop_reason),
                         ("repetition_stop", "exact_repeated_token_suffix"))
        self.assertEqual((stats.repetition_stop_raw_tokens, stats.repetition_stop_trimmed_tokens,
                          stats.repetition_stop_block_tokens, stats.repetition_stop_repeats), (10, 9, 1, 9))

    def test_the_request_stats_carry_the_prompt_width_in_force(self):
        lengths = []  # the model's own width, which it picks from the prompt's length
        self.rt.prompt_chunk_rows = lambda prompt_tokens=None: lengths.append(prompt_tokens) or 512
        self.assertEqual(self.decode().stats.prefill_chunk_size, 512)
        self.assertEqual(lengths, [1])  # decode() sends a one-token prompt
        del self.rt.prompt_chunk_rows
        with api.prefill_chunk_size_override(1024):  # a width the request carries
            self.assertEqual(self.decode().stats.prefill_chunk_size, 1024)

    def test_buffered_tool_call_keeps_the_owner_alive_after_each_verify(self):
        for trace in (False, True):
            with self.subTest(route_tape=trace):
                self.rt = CausalRows(lambda token: (int(token) + 1) % 8, 8)
                elapsed = 0.0
                probe = server._OwnerStallProbe(deadline_s=300, clock=lambda: elapsed)
                actual_commit = self.rt.commit
                polls = []

                def commit(*args):
                    nonlocal elapsed
                    actual_commit(*args)
                    # No client-visible delta while the tool arguments are being assembled.
                    elapsed += 301
                    polls.append(probe.observe())

                tape = SimpleNamespace(enabled=trace, emit=lambda *args: None)
                with patch.object(inkling, "RouteTape", return_value=tape), patch.object(self.rt, "commit", commit):
                    answer = self.decode(depth=3, limit=16, target=SamplerConfig(temperature=1, top_p=1, top_k=0))
                self.assertEqual(polls, [None] * answer.stats.verify_calls)
                self.assertGreater(len(polls), 1)
                probe.rearm()
                elapsed += 301
                self.assertEqual(probe.observe(), 301)

    def test_long_cycle_stops_without_trimming(self):
        self.rt = CausalRows(lambda t: (t + 1) % 37, 37)
        out = self.decode(repetition_stop=True, limit=160)
        self.assertEqual(out.stats.repetition_stop_reason, "long_cycle")
        self.assertEqual(out.stats.repetition_stop_block_tokens, 37)
        self.assertEqual(out.stats.repetition_stop_trimmed_tokens, 0)
        self.assertEqual(out.finish_reason, "stop")
        self.assertLess(len(out.tokens), 160)
        self.assertEqual(out.tokens, self.stream)
        self.assertIsNone(out.final_state)

    def test_default_keeps_requested_reasoning_budget(self):
        out = self.decode(limit=1031)
        self.assertEqual((out.finish_reason, len(out.tokens)), ("length", 1031))
        self.assertFalse(out.stats.repetition_stop_triggered)
        self.assertEqual(out.tokens, self.stream)
        self.assertEqual(out.final_state.final_trunk_cache, [[3, *out.tokens]])

    def test_constraints_take_precedence_over_explicit_stop(self):
        class Grammar:
            stopped = completed = False
            masked_steps = mask_time_s = 0

            def mask_logits_row(self, row):
                self.masked_steps += 1
                return mx.where(mx.arange(row.size) == 1, row, -float("inf"))

            def advance(self, token):
                self.stopped = self.completed = self.masked_steps == 20

        out = self.decode(repetition_stop=True, constraint=Grammar(), draft_sampler=DRAFT)
        self.assertEqual((out.tokens, self.stream, out.finish_reason), ([1] * 20, [1] * 20, "stop"))
        self.assertTrue(out.stats.constraint_completed)
        self.assertEqual((out.stats.finish_stop_origin, out.stats.constraint_masked_steps), ("grammar_terminal", 20))
        self.assertFalse(out.stats.repetition_stop_triggered)
        self.assertEqual(out.final_state.generated_token_ids, tuple(out.tokens))

    def test_eos_and_in_round_cancellation(self):
        out = self.decode(repetition_stop=True, stop_token_ids={1})
        self.assertEqual((out.tokens, self.stream, out.finish_reason), ([2, 1], [2], "stop"))
        self.stream.clear()
        checks = iter([False, False, False, True])
        out = self.decode(repetition_stop=True, abort_check=lambda: next(checks))
        self.assertEqual((out.tokens, self.stream, out.finish_reason), ([2, 1], [2, 1], "stop"))
        self.assertEqual(out.final_state.final_trunk_cache, [[3, 2, 1]])
        self.assertFalse(out.stats.repetition_stop_triggered)

    def test_separate_proposal_draws_match_target_oracle(self):
        for target in (NATIVE, GREEDY):
            for depth, seed in ((1, 3), (4, 21), (7, 37)):
                self.rt = CausalRows(lambda t: (t + 1) % 8, 8)
                expected, counts, rng = [], Counter(), np.random.default_rng(seed)
                for _ in range(34):
                    row = self.rt.scores(expected[-1] if expected else 3)
                    token, _ = api._sample_from_logits(row, target, rng, token_counts=counts)
                    expected.append(token)
                    counts[token] += 1
                greedy = self.decode(depth=depth, limit=34, target=target, seed=seed)
                first_guesses = self.rt.guesses.copy()
                self.rt.guesses.clear()
                alternate = self.decode(depth=depth, limit=34, target=target, seed=seed, draft_sampler=DRAFT)
                self.assertEqual(greedy.tokens, expected)
                self.assertEqual(alternate.tokens, expected)
                self.assertEqual(alternate.final_state.final_trunk_cache, [[3, *expected]])
                self.assertNotEqual(first_guesses, self.rt.guesses)
                sampling = alternate.stats.inkling_draft_sampling
                self.assertEqual((sampling["temperature"], sampling["top_p"], sampling["top_k"]), (.8, .9, 5))
                self.assertEqual(sampling["sampled_tokens"], len(self.rt.guesses))
                self.assertTrue(alternate.stats.rejected_drafts)

    def test_depth_one_front_door_forwards_both_controls(self):
        with patch.object(inkling, "restore_or_prefill_prompt_state", self.rt.prefill):
            out = api.generate_mtp1(self.rt, [3], max_tokens=30, sampler=GREEDY,
                                    draft_sampler=DRAFT, repetition_stop=True)
        self.assertEqual(out.tokens, [2])
        self.assertGreater(out.stats.inkling_draft_sampling["sampled_tokens"], 0)

    def test_markov_state_uses_the_sampled_previous_token(self):
        # Three vocabulary rows. The next score's sign encodes the preceding id.
        target = SimpleNamespace(embed_tokens=lambda ids: mx.ones((1, 1, 1)), unembed=mx.zeros((3, 1)))
        head = SimpleNamespace(
            width=4, vocab=3, mask=mx.ones((1, 1, 1)), output_norm=mx.ones((1,)),
            blocks=[lambda h, residual, cache: (h, mx.zeros_like(h))],
            cfg={"rms_norm_eps": 1e-6, "markov_rank": 1},
            markov={"rnn_proj.weight": mx.array([[0,0,0,0], [0,0,0,0], [0,1,0,0]]),
                    "rnn_proj.bias": mx.zeros((3,)), "markov_w1.weight": mx.array([[0.], [2.], [-2.]])},
            fold=lambda _: mx.array([[0.,0.], [1.,0.], [-1.,0.]]),
        )
        cache = [None, [mx.ones((1,1,1))]]
        self.assertEqual(Drafter.propose(head, target, 1, cache, 3).tolist(), [1, 1, 1])
        peaks, choices = [], iter([2, 1, 2])
        def select(row):
            peaks.append(int(mx.argmax(row).item()))
            return next(choices)
        self.assertEqual(Drafter.propose(head, target, 1, cache, 3, select_token=select).tolist(), [2, 1, 2])
        self.assertEqual(peaks, [1, 2, 1])

    def test_launch_and_live_draft_ownership(self):
        state = SimpleNamespace(runtime=self.rt, lock=Lock(), args=SimpleNamespace(model="inkling2_loop"),
                                draft_sampler=DRAFT, draft_sampler_pinned=False)
        def resolve(request=None):
            metrics = {}
            effective = server._resolve_draft_sampler_for_request(
                state, request_draft_sampler=request, target_temperature=0,
                target_sampler=GREEDY, request_observability=metrics,
            )
            return effective, metrics["draft_sampler_policy_source"], metrics["draft_sampler_resolved_temperature"]
        self.assertEqual(resolve(), (None, "inkling_argmax", 0.))
        self.assertEqual(resolve(DRAFT), (DRAFT, "request_explicit", .8))
        state.draft_sampler_pinned = server._launch_draft_sampler_pinned(SimpleNamespace(draft_sampler_source="explicit"), DRAFT)
        self.assertEqual(resolve(), (DRAFT, "launch_pinned", .8))
        server._mtplx_apply_settings_payload(state, {"draft_temperature": .7, "draft_top_p": .8, "draft_top_k": 4})
        server._mtplx_apply_settings_payload(state, {"temperature": 0., "top_p": 1., "top_k": 0})
        self.assertEqual(resolve(), (SamplerConfig(.7, .8, 4), "launch_pinned", .7))
        state.runtime = SimpleNamespace(backend_id="qwen3_5")
        self.assertEqual(resolve()[1:], ("launch_pinned+greedy_coupled", 0.))
