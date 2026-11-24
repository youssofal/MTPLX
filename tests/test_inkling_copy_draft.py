"""Exercise public prompt-copy policy and the Inkling target/rollback handoff on CPU."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mtplx.models.inkling.copy_draft import PromptCopy, SessionPromptIndexes


@pytest.fixture(autouse=True)
def policy(monkeypatch):
    for key in (
        "MTPLX_CONTEXT_COPY",
        "MTPLX_CONTEXT_COPY_BATCHED",
        "MTPLX_CONTEXT_COPY_MINEXT",
        "MTPLX_RAMP_ENABLED",
        "MTPLX_INKLING_RECEIPT_TOKEN_HASH",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MTPLX_CONTEXT_COPY_NGMIN", "2")
    monkeypatch.setenv("MTPLX_CONTEXT_COPY_NGMAX", "6")


def test_reuse_is_scoped_and_edits_rebuild():
    text = list(range(16)) + list(range(4))
    pool = SessionPromptIndexes()

    def visit(ids, session):
        lane = PromptCopy(ids, session_id=session, verify_rows=8, indexes=pool)
        lane.propose([4], 20)
        return lane.finish(1)["context_copy_index_reused_tokens"]

    assert visit(text, "alpha") == 0
    assert visit(text, "beta") == 0
    assert visit(text + [4, 5], "alpha") == len(text)
    assert visit([99, *text[1:]], "alpha") == 0
    assert visit(text, "gamma") == 0
    assert len(pool.kept) == 2
    a, _ = pool.take("gamma", tuple(text), 2, 6)
    b, n = pool.take("gamma", tuple(text), 2, 6)
    assert a is not b and n == 0


def test_bounds_and_nonmutating_grammar():
    p = PromptCopy(list(range(24)), session_id=None, verify_rows=8, image_pad=8)
    assert p.propose([1, 2, 3], 25) == [4, 5, 6, 7]
    assert p.propose([6, 7], 25) is None
    assert p.declines == {"image_embedding_position": 1}
    rule = Mock(validate_prefix=Mock(return_value=2))
    assert p.propose([1, 2, 3], 3, rule) == [4, 5]
    rule.validate_prefix.assert_called_once_with([4, 5, 6])
    assert not rule.advance.called
    rule.validate_prefix.return_value = 0
    assert p.propose([1, 2, 3], 3, rule) is None
    assert p.declines["grammar_rejected_prefix"] == 1
    assert p.limits["verify_geometry"] > 0 and p.limits["requested_output_budget"] > 0
    assert p.propose([21, 22], 100) == [23]
    assert p.propose([99, 98], 100) is None
    assert p.propose([1, 2], 0) is None


def test_public_backoff_and_independent_round_costs(monkeypatch):
    lane = PromptCopy(list(range(30)), session_id=None, verify_rows=8)
    for i in range(3):
        offered = lane.propose([1, 2], 40)
        assert len(offered) == 7
        lane.observe(len(offered), 0, i + 2, 0.02)
    assert lane.propose([1, 2], 40) is None
    lane.observe(0, 0, 5, 0.01)
    stats = lane.finish(5)
    assert (
        stats["context_copy_rounds"] == 3 and stats["context_copy_drafted_tokens"] == 21
    )
    assert (
        stats["context_copy_suspensions"] == 1
        and stats["context_copy_backoff_tokens"] == 128
    )
    assert stats["context_copy_declines"] == {"acceptance_backoff": 1} and stats[
        "context_copy_verify_rows"
    ] == {"8": 3}
    assert stats["context_copy_timing_s"]["copy_rounds"] == pytest.approx(0.06)
    assert stats["context_copy_timing_s"]["ordinary_rounds"] == pytest.approx(0.01)
    assert lane.propose([0] * 70 + [1, 2], 10)
    lane.observe(7, 7, 78, 0.02)
    assert lane.finish(78)["context_copy_backoff_tokens"] == 64
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")
    off = PromptCopy([0, 1, 2, 3], session_id=None, verify_rows=8)
    assert off.propose([1, 2], 9) is None
    assert off.finish(2)["context_copy_declines"] == {"disabled_by_setting": 1}


def test_target_sampling_and_commit_match_the_same_seed_without_copy(monkeypatch):
    import mlx.core as mx
    from mtplx.generation import PromptState
    from mtplx.models.inkling import generation
    from mtplx.sampling import SamplerConfig

    class Runtime:
        trunk_entries = 1

        def __init__(self):
            self.verified = []
            self.drafter = SimpleNamespace(
                width=8,
                propose=lambda model, anchor, cache, count: mx.array([0] * count),
            )
            self.model = None
            self.tokenizer = SimpleNamespace(
                convert_tokens_to_ids=lambda s: 100, decode=lambda ids: str(ids)
            )

        def scores(self, token):
            return mx.array(
                [
                    6.0
                    if n == (token + 1) % 17
                    else (1.0 if n == (token + 2) % 17 else -8.0)
                    for n in range(17)
                ]
            )

        def verify(self, block, cache):
            cache[0].extend(block)
            features = [mx.array([[block]]).transpose(0, 2, 1)]
            return mx.stack([self.scores(t) for t in block]), features, None

        def commit(self, cache, features, tapes, kept, total):
            self.verified.append((kept, total))
            if kept < total:
                del cache[0][-(total - kept) :]
            cache[1].extend(cache[0][-kept:])

        def forward_ar(self, ids, cache, **kwargs):
            t = ids.item()
            cache[0].append(t)
            cache[1].append(t)
            return self.scores(t)[None, None], mx.array([[[t]]])

    class Grammar:
        def __init__(self):
            self.committed = []

        def mask_logits_row(self, row):
            return row

        def validate_prefix(self, ids):
            return min(len(ids), 5 - len(self.committed))

        def advance(self, t):
            self.committed.append(t)

        @property
        def stopped(self):
            return len(self.committed) == 5

    # The earlier code span agrees on three guesses, then predicts 12 where the target prefers 8.
    prompt = [0, 1, 2, 3, 4, 5, 6, 7, 12, 10, 11, 12, 13, 0, 1, 2, 3]

    def prefill(rt, ids, **kwargs):
        return PromptState(
            [list(ids), list(ids)],
            rt.scores(ids[-1])[None, None],
            mx.zeros((1, 1, 1)),
            None,
            tuple(ids),
            0.0,
            suffix_tokens=len(ids),
        )

    monkeypatch.setattr(generation, "restore_or_prefill_prompt_state", prefill)
    monkeypatch.setenv("MTPLX_INKLING_RECEIPT_TOKEN_HASH", "1")

    def run(enabled, temperature, budget, stop, grammar, abort=False):
        monkeypatch.setenv("MTPLX_CONTEXT_COPY", str(int(enabled)))
        rt, stream = Runtime(), []
        rule = Grammar() if grammar else None
        requested = SamplerConfig(
            temperature=temperature,
            top_p=1.0,
            top_k=0,
            presence_penalty=0.2,
            frequency_penalty=0.11,
        )
        out = generation.generate(
            rt,
            prompt,
            max_tokens=budget,
            sampler=requested,
            depth=3,
            seed=37,
            stop_token_ids=stop,
            token_callback=stream.extend,
            prefill_callback=None,
            session_bank=None,
            session_id="s",
            session_restore_mode="clone",
            session_template_hash=None,
            session_draft_head_identity=None,
            session_policy_fingerprint=None,
            capture_final_state=True,
            commit_prompt_state_to_bank=False,
            vision_splice=None,
            constraint=rule,
            abort_check=(lambda: len(stream) >= 1) if abort else None,
            first_token_logprobs_top_k=None,
            store_prefix_snapshot=False,
        )
        if out.tokens:
            assert out.final_state.final_trunk_cache == [
                prompt + out.tokens,
                prompt + out.tokens,
            ]
            assert stream == [t for t in out.tokens if t not in stop]
        if rule:
            assert rule.committed == out.tokens
        return out, rt.verified

    with mx.stream(mx.cpu):
        for temperature, budget, stop, grammar, abort in (
            (0.0, 20, set(), False, False),
            (1.0, 20, set(), False, False),
            (1.0, 13, {6}, False, False),
            (0.0, 20, set(), True, False),
            (0.0, 2, set(), False, False),
            (1.0, 20, set(), False, True),
            (1.0, 0, set(), False, False),
        ):
            a, _ = run(False, temperature, budget, stop, grammar, abort)
            b, verified = run(True, temperature, budget, stop, grammar, abort)
            assert a.tokens == b.tokens and a.finish_reason == b.finish_reason
            assert (
                a.stats.inkling_output_ids_sha256 == b.stats.inkling_output_ids_sha256
            )
            assert a.stats.context_copy_rounds == 0
            if budget >= 2 and not abort:
                assert (
                    b.stats.context_copy_rounds > 0
                    and b.stats.context_copy_accepted_tokens > 0
                )
                assert (
                    b.stats.context_copy_rounds
                    + sum(b.stats.context_copy_declines.values())
                    == b.stats.verify_calls
                )
                assert max(width for _, width in verified) <= 8
            if budget == 20 and not grammar and not abort and temperature == 0:
                assert (4, 8) in verified


def test_public_receipts_expose_copy_declines_and_distinguish_active_kv_from_ssd_snapshots():
    from mtplx.server.openai import (
        _health_inkling_cache_payload,
        _metrics_envelope,
        _public_mtplx_stats,
    )

    stats = {
        "context_copy_rounds": 2,
        "context_copy_declines": {"no_prompt_match": 3},
        "context_copy_limits": {"verify_geometry": 2},
        "context_copy_verify_rows": {"8": 2},
        "context_copy_timing_s": {
            "index_build_or_extend": 0.001,
            "copy_rounds": 0.04,
            "ordinary_rounds": 0.03,
        },
        "context_copy_index_reused_tokens": 41,
        "context_copy_suspended": True,
        "context_copy_backoff_tokens": 128,
        "inkling_output_ids_sha256": "a" * 64,
        "ssd_cached_tokens": 41,
        "ssd_cache_hit": True,
        "ssd_restore_s": 0.07,
        "ssd_suffix_tokens": 0,
        "cache_source": "ssd",
    }
    request = _metrics_envelope(
        stats=stats,
        prompt_tokens=41,
        completion_tokens=5,
        request_elapsed_s=0.15,
        token_times=[],
        request_started_s=0,
        lock_wait_time_s=0,
        session_id="a",
        session_cache_hit=True,
        cache_miss_reason=None,
        session_restore_mode="ssd_clone",
        mtp_depth=7,
        generation_limits={},
    )
    public = _public_mtplx_stats({"stats": stats})
    for key in stats:
        assert request[key] == stats[key] and public[key] == stats[key]
    state = SimpleNamespace(
        runtime=SimpleNamespace(backend_id="inkling2_loop"),
        session_bank_cold_tier=object(),
    )
    capabilities = _health_inkling_cache_payload(state)["inkling_cache"]
    assert capabilities["ssd_session_snapshots_enabled"] is True
    assert capabilities["active_ssd_kv"] is capabilities["active_paged_kv"] is False
    assert capabilities["active_kv_quantization"] == "off"
    state.session_bank_cold_tier = None
    assert not _health_inkling_cache_payload(state)["inkling_cache"][
        "ssd_session_snapshots_enabled"
    ]


def test_the_fast_steps_width_verifies_longer_copied_runs():
    """RC6's verify path covers an anchor plus seven proposals; the fast step verifies up to
    32 rows with every row equal to the one-row step, so a proven copy run is offered whole
    (up to the copy lane's own block cap) instead of in pieces of seven."""
    prompt = list(range(200))
    narrow = PromptCopy(prompt, session_id=None, verify_rows=8)
    wide = PromptCopy(prompt, session_id=None, verify_rows=32)
    for lane in (narrow, wide):
        for _ in range(3):  # three fully accepted rounds lift the probation cap
            lane.observe(8, 8, 50, 0.0)
    history = list(range(40))  # the prompt's own continuation: a long match
    assert len(narrow.propose(history, 100)) == 7 and narrow.limits["verify_geometry"] == 1
    assert len(wide.propose(history, 100)) > 7 and not wide.limits["verify_geometry"]


def _rewrite_prompt():
    # A long run with no internal repeats, so any suffix of it matches one place.
    return list(range(1000, 1400))


def test_the_cost_gate_is_off_by_default_and_leaves_the_lane_alone(monkeypatch):
    monkeypatch.delenv("MTPLX_CONTEXT_COPY_GATE", raising=False)
    lane = PromptCopy(_rewrite_prompt(), session_id=None, verify_rows=32)
    assert lane.gate is None
    # Probation cap 8 snaps verify_rows 9 -> 8 (7 proposals + anchor) to stay on warmed tiles.
    assert lane.propose([1000, 1001], 100) == list(range(1002, 1009))
    stats = lane.finish(2)
    assert "cost_gate" not in stats["context_copy_declines"]
    assert not any(k.startswith("gate_") for k in stats["context_copy_timing_s"])


def _drive(lane, prompt, rounds, landed_of, cost_s):
    """Drive the lane as the generation loop does: ordinary rounds of 3 tokens while nothing
    matches, then copy rounds whose first `landed_of(offered)` tokens land; returns the copy
    rounds run and the gate's vetoes."""
    tokens = [7, 8]  # not in the prompt: the first rounds are ordinary
    copies = 0
    for i in range(rounds):
        offered = lane.propose(tokens, 10_000)
        if offered is None:
            lane.observe(0, 0, len(tokens) + 3, 0.05)
            tokens = tokens + [7, 8, 9] if i < 3 else tokens + [prompt[10 * i % 300], prompt[10 * i % 300 + 1]]
            continue
        landed = landed_of(len(offered))
        copies += 1
        lane.observe(len(offered), landed, len(tokens) + landed + 1, cost_s(len(offered) + 1))
        at = prompt.index(offered[0])
        tokens = tokens + offered[:landed] + [7] + [prompt[(at + 50) % 300], prompt[(at + 51) % 300]]
    return copies, lane.declines["cost_gate"]


def test_the_cost_gate_vetoes_copy_rounds_that_lose_and_keeps_those_that_pay(monkeypatch):
    """Q1's session: probation blocks landing 5 of 8 at about 18 ms a verify row (162 ms a round)
    against 50 ms ordinary rounds of 3 tokens: the lane's own rules keep them (share 0.6), the
    gate vetoes them. Blocks that land whole at 4 ms a row keep running."""
    monkeypatch.setenv("MTPLX_CONTEXT_COPY_GATE", "1")
    prompt = _rewrite_prompt()
    monkeypatch.setenv("MTPLX_CONTEXT_COPY_GATE_COSTS", "50,0,18")
    losing = PromptCopy(prompt, session_id=None, verify_rows=32)
    copies, vetoed = _drive(losing, prompt, 40, lambda n: min(5, n), lambda rows: 0.018 * rows)
    assert vetoed > 0 and copies < 40 - 3 - vetoed + 1, (copies, vetoed, losing.declines)
    monkeypatch.setenv("MTPLX_CONTEXT_COPY_GATE_COSTS", "50,0,4")
    paying = PromptCopy(prompt, session_id=None, verify_rows=32)
    copies, vetoed = _drive(paying, prompt, 40, lambda n: n, lambda rows: 0.004 * rows)
    assert vetoed == 0 and copies > 20, (copies, paying.declines)
    stats = paying.finish(100)
    assert stats["context_copy_timing_s"]["gate_ordinary_s"] == pytest.approx(0.05)
    assert stats["context_copy_timing_s"]["gate_copy_row_s"] == pytest.approx(0.004)


def test_mixed_edit_boundary_matches_avoid_lockout_and_recover_width():
    """In a rewrite with edit boundaries (e.g. 5..8 tokens landing out of a 23-token block)
    interleaved with verbatim runs, profitable boundary hits (landed >= 4) must not trigger
    a blind 64-token acceptance_backoff lockout, must step width down smoothly across 8-row
    tiles when short boundaries repeat, and must jump back to full width once a block saturates."""
    prompt = _rewrite_prompt()
    lane = PromptCopy(prompt, session_id=None, verify_rows=32)
    history = prompt[:16]  # ext=4 (full suffix match)

    # Two saturated probation rounds unlock full width (23 proposals -> verify_rows=24).
    for step in range(2):
        offered = lane.propose(history, 200)
        assert len(offered) == 7
        lane.observe(len(offered), len(offered), 10 + 8 * step, 0.048)

    wide_offer = lane.propose(history, 200)
    assert len(wide_offer) == 23

    # First edit boundary on a wide block: 6/23 landed (raw_share=0.26 < 0.35, but 6 verbatim tokens!).
    # Must NOT suspend and steps smoothly from 24 -> 16 (15 proposals).
    lane.observe(len(wide_offer), 6, 30, 0.095)
    assert lane.suspensions == 0
    mid_offer = lane.propose(history, 200)
    assert mid_offer is not None and len(mid_offer) == 15

    # Second consecutive boundary hit: 5/15 landed (still >= 4 verbatim tokens).
    # Must NOT suspend and steps smoothly from 16 -> 8 (7 proposals).
    lane.observe(len(mid_offer), 5, 36, 0.072)
    assert lane.suspensions == 0
    narrow_offer = lane.propose(history, 200)
    assert narrow_offer is not None and len(narrow_offer) == 7

    # Repeated 4..6-token boundary hits stay active at width 8 with zero lockouts.
    for i, landed in enumerate((4, 5, 6, 4, 5)):
        offer = lane.propose(history, 200)
        assert offer is not None and len(offer) == 7
        lane.observe(len(offer), landed, 44 + 7 * i, 0.048)
    assert lane.suspensions == 0
    assert "acceptance_backoff" not in lane.declines

    # As soon as unchanged code resumes and a probation block saturates (7/7),
    # full width (23 proposals) recovers immediately on the next round.
    sat_offer = lane.propose(history, 200)
    assert len(sat_offer) == 7
    lane.observe(len(sat_offer), len(sat_offer), 90, 0.048)
    recovered = lane.propose(history, 200)
    assert recovered is not None and len(recovered) == 23
    assert lane.finish(120)["context_copy_suspensions"] == 0


def test_dead_matches_from_full_width_still_drop_and_suspend():
    """Genuine dead/spurious matches (landed <= 1) must still drop immediately back to
    probation width and trigger exponential suspension backoff once EMA falls below 0.35."""
    prompt = _rewrite_prompt()
    lane = PromptCopy(prompt, session_id=None, verify_rows=32)
    history = prompt[:16]

    # Unlock full width first.
    for step in range(2):
        offer = lane.propose(history, 200)
        lane.observe(len(offer), len(offer), 10 + 8 * step, 0.048)
    assert len(lane.propose(history, 200)) == 23

    # First dead hit (landed=0) on a wide block drops width immediately back to probation (7).
    lane.observe(23, 0, 30, 0.095)
    after_miss = lane.propose(history, 200)
    assert after_miss is not None and len(after_miss) == 7

    # Continued dead/spurious hits (landed=0, 1) push EMA below 0.35 and suspend.
    emitted = 32
    for landed in (0, 1):
        offer = lane.propose(history, 200)
        assert offer is not None and len(offer) == 7
        lane.observe(len(offer), landed, emitted, 0.048)
        emitted += landed + 1

    assert lane.suspensions == 1
    assert lane.propose(history, 200) is None
    assert lane.declines["acceptance_backoff"] == 1
    assert lane.backoff == 128


def test_dense_partial_edits_stay_at_probation_width_without_oscillating_or_locking_out():
    """Repeated 5-token partial matches beat K3 at width 8 but would lose at width 24.
    The lane must keep them active at width 8 without bouncing up to 24 or suspending."""
    prompt = _rewrite_prompt()
    lane = PromptCopy(prompt, session_id=None, verify_rows=32)
    history = prompt[:16]
    widths = []
    for i in range(12):
        offer = lane.propose(history, 200)
        assert offer is not None
        widths.append(len(offer) + 1)
        lane.observe(len(offer), 5, 10 + 6 * i, 0.048)
    assert widths == [8] * 12
    assert lane.suspensions == 0
    assert "acceptance_backoff" not in lane.declines
