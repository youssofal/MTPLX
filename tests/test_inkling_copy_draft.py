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
