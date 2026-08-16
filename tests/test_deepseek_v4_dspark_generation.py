import inspect
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

from mtplx import generation as generation_module  # noqa: E402
from mtplx import native_block_speculation as native_speculation  # noqa: E402
from mtplx.deepseek_v4_dspark_generation import (  # noqa: E402
    DeepseekV4DSparkBackend,
)
from mtplx.mtp_patch import MTPContract  # noqa: E402
from mtplx.models.deepseek_v4 import DeepseekV4DSparkCache  # noqa: E402
from mtplx.runtime import MTPLXRuntime  # noqa: E402
from mtplx.sampling import SamplerConfig  # noqa: E402


def generate_mtpk(rt, prompt_ids, **kwargs):
    """Exercise the native scheduler without depending on outer dispatch."""
    return native_speculation.generate_native_block_speculative(
        rt,
        rt.block_speculative_backend,
        prompt_ids,
        abort_check=kwargs.pop("abort_check", None),
        max_tokens=kwargs.pop("max_tokens"),
        sampler=kwargs.pop("sampler"),
        speculative_depth=kwargs.pop("speculative_depth"),
        seed=kwargs.pop("seed", 0),
        stop_token_ids=kwargs.pop("stop_token_ids", None),
        draft_sampler=kwargs.pop("draft_sampler", None),
        token_callback=kwargs.pop("token_callback", None),
        prefill_callback=kwargs.pop("prefill_callback", None),
        constraint=kwargs.pop("constraint", None),
        vision_splice=kwargs.pop("vision_splice", None),
        adaptive_policy=kwargs.pop("adaptive_policy", None),
        adaptive_width_policy=kwargs.pop("adaptive_width_policy", None),
    )


class _Tokenizer:
    eos_token_id = None
    pad_token_id = None

    def decode(self, tokens):
        return " ".join(str(int(token)) for token in tokens)


class _TargetCache:
    def __init__(self):
        self.offset = 0
        self.trimmed = []

    def trim(self, n):
        self.offset -= int(n)
        self.trimmed.append(int(n))
        return int(n)


class _ProposalCache:
    def __init__(self):
        self.ring = None
        self.prefill_length = 0


class _FakeDSpark:
    def __init__(self):
        self.owner = None
        self.stages = (object(), object(), object())

    def make_cache(self):
        cache = [_ProposalCache(), _ProposalCache(), _ProposalCache()]
        self.owner.proposal_cache = cache
        return cache

    def prefill(self, hidden, cache):
        for entry in cache:
            entry.ring = hidden[...]
            entry.prefill_length = int(hidden.shape[1])
        self.owner.prefill_hidden = np.asarray(hidden).copy()
        self.owner.proposal_prefill_spans.append((0, np.asarray(hidden).copy()))

    def forward(
        self,
        hidden,
        token_ids,
        embed_tokens,
        lm_head,
        cache,
        *,
        start_pos,
        greedy,
        ids_only_width,
        forced_first_token_ids=None,
    ):
        del embed_tokens, lm_head, greedy
        for entry in cache:
            entry.ring = mx.full((1, 1, 1), 999.0)
        ids, _logits, _confidence = self.owner.draft_deepseek_v4_dspark(
            hidden, token_ids, cache, start_pos=start_pos
        )
        if forced_first_token_ids is not None:
            forced = int(np.asarray(forced_first_token_ids)[0])
            ids = mx.concatenate([ids[:, :1], mx.array([[forced]]), ids[:, 2:]], axis=1)
            self.owner.forced_primary_ids.append(forced)
        self.owner.proposal_inputs.append(
            (int(np.asarray(token_ids)[0]), int(start_pos), int(ids_only_width))
        )
        self.owner.proposal_widths.append(int(ids_only_width))
        return ids[:, : 1 + int(ids_only_width)]

    def commit_main(self, hidden, cache, *, start_pos):
        self.owner.proposal_prefill_spans.append(
            (int(start_pos), np.asarray(hidden).copy())
        )
        self.owner.commit_ring_history.append(
            [np.asarray(entry.ring).copy() for entry in cache]
        )
        for entry in cache:
            entry.ring = hidden[...]
            entry.prefill_length = int(start_pos) + int(hidden.shape[1])
        self.owner.commits.append((int(start_pos), np.asarray(hidden).copy()))


class _DSparkModel:
    def __init__(self):
        self._dspark = _FakeDSpark()
        self.model = type("_Body", (), {"embed_tokens": staticmethod(lambda x: x)})()
        self.lm_head = lambda x: x
        self.owner = None

    def __call__(self, *args, **kwargs):
        return self.owner.target_forward(*args, **kwargs)


class _DSparkRuntime(MTPLXRuntime):
    def __init__(self, *, supported_depths=(2,)):
        model = _DSparkModel()
        super().__init__(
            model=model,
            tokenizer=_Tokenizer(),
            model_path=Path("."),
            mtp_enabled=True,
            contract=MTPContract(),
        )
        self.deepseek_v4_dspark_enabled = True
        model._dspark.owner = self
        model.owner = self
        self.block_speculative_backend = DeepseekV4DSparkBackend.bind(
            model, supported_depths=supported_depths
        )
        self.target_cache = _TargetCache()
        self.prefill_hidden = None
        self.proposal_cache = None
        self.draft_widths = []
        self.proposal_widths = []
        self.commits = []
        self.commit_ring_history = []
        self.proposal_inputs = []
        self.proposal_prefill_spans = []
        self.forced_primary_ids = []
        self.target_forward_widths = []
        self.target_forward_inputs = []
        self.target_forward_options = []
        self.runtime_forward_calls = 0

    def make_cache(self):
        return [self.target_cache]

    def target_forward(
        self,
        input_ids,
        cache=None,
        return_hidden=False,
        hidden_variant=None,
        emit_logits=True,
        logits_keep=None,
        input_embeddings=None,
    ):
        del hidden_variant, input_embeddings
        ids = np.asarray(input_ids, dtype=np.int32)
        self.target_forward_widths.append(int(ids.shape[1]))
        self.target_forward_inputs.append(ids.copy())
        self.target_forward_options.append(
            {
                "return_hidden": bool(return_hidden),
                "emit_logits": bool(emit_logits),
                "logits_keep": logits_keep,
            }
        )
        cache[0].offset += ids.shape[1]
        vocab = 1024
        logits = np.full((1, ids.shape[1], vocab), -1000.0, dtype=np.float32)
        for row, token in enumerate(ids[0]):
            logits[0, row, int(token) + 1] = 1000.0
        if logits_keep is not None:
            logits = logits[:, -int(logits_keep) :]
        hidden = np.repeat(ids[..., None].astype(np.float32), 3, axis=-1)
        result = mx.array(logits) if emit_logits else None
        return (result, mx.array(hidden)) if return_hidden else result

    def forward_ar(self, *args, **kwargs):
        self.runtime_forward_calls += 1
        self._count("forward_ar_hidden_calls")
        return self.target_forward(*args, **kwargs)

    def make_deepseek_v4_dspark_cache(self):
        return [object(), object(), object()]

    def prefill_deepseek_v4_dspark(self, hidden, cache):
        del cache
        self.prefill_hidden = np.asarray(hidden).copy()

    def draft_deepseek_v4_dspark(self, hidden, token_ids, cache, *, start_pos):
        del hidden, cache, start_pos
        token = int(np.asarray(token_ids)[0])
        # Primary plus two exact future drafts, followed by a deliberate miss.
        # This exercises K2 proposal acceptance against serial target rows.
        ids = mx.array([[token, token + 1, token + 2, token + 3, 999, 999]])
        return ids, mx.zeros((1, 5, 1024)), mx.ones((1, 5))

    def draft_deepseek_v4_dspark_ids(
        self, hidden, token_ids, cache, *, start_pos, width
    ):
        self.proposal_widths.append(int(width))
        ids, _logits, _confidence = self.draft_deepseek_v4_dspark(
            hidden, token_ids, cache, start_pos=start_pos
        )
        return ids[:, : 1 + int(width)]

    def commit_deepseek_v4_dspark(self, hidden, cache, *, start_pos):
        del cache
        self.commits.append((int(start_pos), np.asarray(hidden).copy()))


def test_dspark_fixed_k2_preserves_the_greedy_target_stream():
    rt = _DSparkRuntime()
    callback = []
    out = generate_mtpk(
        rt,
        [9, 10],
        max_tokens=8,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
        speculative_depth=2,
        stop_token_ids=set(),
        token_callback=lambda block: callback.extend(block),
    )
    assert out.tokens == list(range(11, 19))
    assert callback == out.tokens
    assert out.stats.speculative_depth == 2
    assert out.stats.runtime_mtp_enabled is True
    assert rt.prefill_hidden.tolist() == [[[9.0, 9.0, 9.0], [10.0, 10.0, 10.0]]]
    assert all(hidden.shape[1] <= 3 for _, hidden in rt.commits)
    assert all(float(hidden.max()) < 1000.0 for _, hidden in rt.commits)
    # The tail token is accepted from the final verified block, so every emitted
    # token is already represented in the target cache.
    assert rt.target_cache.offset == 2 + len(out.tokens)
    assert rt.proposal_widths
    assert all(width <= 3 for width in rt.proposal_widths)
    assert out.stats.events == []


class _WidthDivergentTargetRuntime(_DSparkRuntime):
    """Model the receipt-proven case where batched target rows are not serial M1."""

    def __init__(self):
        super().__init__()
        self.batched_decode_calls = 0

    def target_forward(self, input_ids, cache=None, **kwargs):
        offset_before = cache[0].offset
        logits, hidden = super().target_forward(input_ids, cache=cache, **kwargs)
        width = int(np.asarray(input_ids).shape[1])
        if offset_before > 0 and width > 1:
            self.batched_decode_calls += 1
            poisoned_logits = np.full(logits.shape, -1000.0, dtype=np.float32)
            poisoned_logits[..., 777] = 1000.0
            logits = mx.array(poisoned_logits)
            hidden = hidden + 1000.0
        return logits, hidden


def test_dspark_target_verification_uses_physical_m3_when_width_diverges():
    rt = _WidthDivergentTargetRuntime()

    out = generate_mtpk(
        rt,
        [9, 10],
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
    )

    assert out.tokens != [11, 12, 13, 14]
    assert rt.batched_decode_calls > 0
    assert 3 in rt.target_forward_widths
    assert rt.target_cache.trimmed


def test_dspark_uses_one_sanctioned_scheduler_evaluation_boundary(monkeypatch):
    events = []

    class _OrderingRuntime(_DSparkRuntime):
        future = None

        def target_forward(self, input_ids, cache=None, **kwargs):
            if cache[0].offset >= 2 and int(np.asarray(input_ids).shape[1]) > 1:
                events.append("target_block_graph")
            return super().target_forward(input_ids, cache=cache, **kwargs)

    rt = _OrderingRuntime()
    original_propose = DeepseekV4DSparkBackend.propose

    def tracked_propose(self, *args, **kwargs):
        events.append("proposal_graph")
        rt.future = original_propose(self, *args, **kwargs)
        return rt.future

    original_asarray = native_speculation.np.asarray

    def tracked_asarray(value, *args, **kwargs):
        if value is rt.future:
            events.append("proposal_materialized")
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(DeepseekV4DSparkBackend, "propose", tracked_propose)
    monkeypatch.setattr(native_speculation.np, "asarray", tracked_asarray)

    out = generate_mtpk(
        rt,
        [9, 10],
        max_tokens=3,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
    )

    assert out.tokens == [11, 12, 13]
    assert events[:3] == [
        "proposal_graph",
        "proposal_materialized",
        "target_block_graph",
    ]


def test_dspark_uses_generic_backend_even_without_legacy_family_flag():
    rt = _DSparkRuntime()
    rt.deepseek_v4_dspark_enabled = False
    out = generate_mtpk(
        rt,
        [9, 10],
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
    )
    assert out.tokens == [11, 12, 13, 14]


def test_dspark_depth_two_verifies_primary_and_two_drafts_in_one_m3_call():
    rt = _DSparkRuntime()

    out = generate_mtpk(
        rt,
        [9, 10],
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
    )

    assert out.tokens == [11, 12, 13, 14]
    # Chunked prefill owns two M1 calls, followed by one physical M3 and the M1 tail.
    assert rt.target_forward_widths == [1, 1, 3, 1]
    assert rt.proposal_widths == [3]
    assert out.stats.accepted_drafts == 2
    assert out.stats.drafted_tokens == 2
    assert out.stats.accepted_by_depth == [1, 1]
    assert out.stats.repair_time_s == 0.0


def test_dspark_construction_selected_depth_three_verifies_one_physical_m4():
    rt = _DSparkRuntime(supported_depths=(3,))

    out = generate_mtpk(
        rt,
        [9, 10],
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=3,
        stop_token_ids=set(),
    )

    assert out.tokens == [11, 12, 13, 14]
    assert rt.target_forward_widths == [1, 1, 4, 1]
    assert rt.proposal_widths == [4]
    assert out.stats.accepted_drafts == 2
    assert out.stats.accepted_by_depth == [1, 1, 0]


def test_dspark_proposal_restore_precedes_the_full_accepted_prefix_commit():
    """The proposal's poisoned rings never reach the accepted-prefix commit."""
    rt = _DSparkRuntime()

    out = generate_mtpk(
        rt,
        [9, 10],
        max_tokens=3,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
    )

    assert out.stats.accepted_drafts == 2
    assert rt.target_cache.trimmed == []
    assert len(rt.commit_ring_history) == 1
    for ring in rt.commit_ring_history[0]:
        np.testing.assert_array_equal(ring, rt.prefill_hidden)
        assert float(np.max(ring)) != 999.0


def test_dspark_single_token_prompt_uses_one_explicit_m1_seed_before_fixed_m3():
    rt = _DSparkRuntime()

    out = generate_mtpk(
        rt,
        [10],
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
    )

    assert out.tokens == [11, 12, 13, 14]
    assert rt.target_forward_widths == [1, 1, 3]
    assert rt.proposal_inputs == [(11, 1, 3)]
    assert out.stats.verify_calls == 1
    assert out.stats.accepted_drafts == 2


def test_dspark_backend_uses_construction_bound_model_operations():
    rt = _DSparkRuntime()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("enabled backend called a validating runtime wrapper")

    rt.make_deepseek_v4_dspark_cache = forbidden
    rt.prefill_deepseek_v4_dspark = forbidden
    rt.draft_deepseek_v4_dspark_ids = forbidden
    rt.commit_deepseek_v4_dspark = forbidden

    out = generate_mtpk(
        rt,
        [9, 10],
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
    )

    assert out.tokens == [11, 12, 13, 14]


def test_dspark_backend_binds_the_target_callable_before_hot_decode():
    rt = _DSparkRuntime()
    before = dict(rt.diagnostic_counters)
    target_forward = rt.block_speculative_backend.bind_target_forward(rt)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("hot decode re-looked up the runtime target route")

    rt.forward_ar = forbidden
    logits, hidden = target_forward(
        mx.array([[10]]), cache=rt.make_cache(), return_hidden=True
    )

    assert logits.shape[1] == 1
    assert hidden.shape[1] == 1
    assert rt.target_forward_widths == [1]
    assert rt.runtime_forward_calls == 0
    assert rt.diagnostic_counters == before


class _SecondProposalMissRuntime(_DSparkRuntime):
    def draft_deepseek_v4_dspark(self, hidden, token_ids, cache, *, start_pos):
        del hidden, cache, start_pos
        token = int(np.asarray(token_ids)[0])
        ids = mx.array([[token, token + 1, 999, 999, 999, 999]])
        return ids, mx.zeros((1, 5, 1024)), mx.ones((1, 5))


def test_dspark_rejected_drafts_are_trimmed_from_the_target_cache():
    rt = _SecondProposalMissRuntime()

    out = generate_mtpk(
        rt,
        [9, 10],
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
    )

    assert out.tokens == [11, 12, 13, 14]
    assert rt.target_forward_widths == [1, 1, 3, 3, 2, 1]
    assert rt.target_cache.trimmed[:2] == [2, 2]
    assert out.stats.rejected_drafts >= 2


def test_dspark_proposal_restore_precedes_the_primary_only_commit():
    """A rejected suffix restores proposal rings before committing target state."""
    rt = _SecondProposalMissRuntime()

    out = generate_mtpk(
        rt,
        [9, 10],
        max_tokens=3,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
    )

    assert out.stats.accepted_drafts == 0
    assert rt.target_cache.trimmed == [2, 1]
    assert len(rt.commit_ring_history) >= 1
    for ring in rt.commit_ring_history[0]:
        np.testing.assert_array_equal(ring, rt.prefill_hidden)
        assert float(np.max(ring)) != 999.0


class _AcceptOneThenExactRuntime(_DSparkRuntime):
    def draft_deepseek_v4_dspark(self, hidden, token_ids, cache, *, start_pos):
        del hidden, cache, start_pos
        token = int(np.asarray(token_ids)[0])
        if token == 10:
            ids = mx.array([[token, token + 1, 999, 999, 999, 999]])
        else:
            ids = mx.array([[token, token + 1, token + 2, token + 3, 999, 999]])
        return ids, mx.zeros((1, 5, 1024)), mx.ones((1, 5))


def test_dspark_accept_one_resumes_at_the_target_correction_boundary():
    rt = _AcceptOneThenExactRuntime()

    out = generate_mtpk(
        rt,
        [9, 10],
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
    )

    assert out.tokens == [11, 12, 13, 14]
    assert rt.target_forward_widths == [1, 1, 3, 3]
    # The generic engine owns the next target position; DSpark RoPE/ring setup
    # owns the carried hidden's position, exactly one row earlier.
    assert rt.proposal_inputs == [(10, 1, 3), (11, 2, 3)]
    assert out.stats.accepted_drafts == 2
    assert out.stats.rejected_drafts == 2


class _AcceptTwoThenExactRuntime(_DSparkRuntime):
    def draft_deepseek_v4_dspark(self, hidden, token_ids, cache, *, start_pos):
        del hidden, cache, start_pos
        token = int(np.asarray(token_ids)[0])
        if token == 10:
            ids = mx.array([[token, token + 1, token + 2, 999, 999, 999]])
        else:
            ids = mx.array([[token, token + 1, token + 2, token + 3, 999, 999]])
        return ids, mx.zeros((1, 5, 1024)), mx.ones((1, 5))


def test_dspark_accept_two_resumes_at_the_target_correction_boundary():
    rt = _AcceptTwoThenExactRuntime()

    out = generate_mtpk(
        rt,
        [9, 10],
        max_tokens=5,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
    )

    assert out.tokens == [11, 12, 13, 14, 15]
    assert rt.target_forward_widths == [1, 1, 3, 3]
    assert rt.proposal_inputs == [(10, 1, 3), (12, 3, 3)]
    assert out.stats.accepted_drafts == 3
    assert out.stats.rejected_drafts == 1


class _FirstMissRuntime(_DSparkRuntime):
    def draft_deepseek_v4_dspark(self, hidden, token_ids, cache, *, start_pos):
        del hidden, token_ids, cache, start_pos
        ids = mx.array([[0, 999, 999, 999, 999, 999]])
        return ids, mx.zeros((1, 5, 1024)), mx.ones((1, 5))


def test_dspark_wrong_internal_primary_is_replaced_before_physical_target_block():
    rt = _FirstMissRuntime()
    out = generate_mtpk(
        rt,
        [9, 10],
        max_tokens=3,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
    )
    assert out.tokens == [11, 12, 13]
    assert rt.forced_primary_ids == [11, 12]
    assert rt.target_forward_widths == [1, 1, 3, 2, 1]
    assert out.stats.verify_calls == 3


def test_dspark_rejects_non_greedy_sampling_before_prefill():
    rt = _DSparkRuntime()
    with pytest.raises(ValueError, match="greedy"):
        generate_mtpk(
            rt,
            [9, 10],
            max_tokens=4,
            sampler=SamplerConfig(temperature=0.7),
            speculative_depth=2,
            stop_token_ids=set(),
        )
    assert rt.target_cache.offset == 0


@pytest.mark.parametrize("depth", [0, 1, 3, 4, 6])
def test_dspark_rejects_unbenchmarked_widths_before_prefill(depth):
    rt = _DSparkRuntime()
    with pytest.raises(ValueError, match="width"):
        generate_mtpk(
            rt,
            [9, 10],
            max_tokens=4,
            sampler=SamplerConfig(temperature=0.0),
            speculative_depth=depth,
            stop_token_ids=set(),
        )
    assert rt.target_cache.offset == 0


@pytest.mark.parametrize(
    "draft_sampler",
    [
        SamplerConfig(temperature=0.7),
        SamplerConfig(temperature=0.0, presence_penalty=1.0),
        SamplerConfig(temperature=0.0, frequency_penalty=1.0),
    ],
)
def test_dspark_rejects_unsupported_draft_sampling_before_prefill(draft_sampler):
    rt = _DSparkRuntime()
    with pytest.raises(ValueError, match="greedy"):
        generate_mtpk(
            rt,
            [9, 10],
            max_tokens=4,
            sampler=SamplerConfig(temperature=0.0),
            draft_sampler=draft_sampler,
            speculative_depth=2,
            stop_token_ids=set(),
        )
    assert rt.target_forward_widths == []
    assert rt.prefill_hidden is None


def test_dspark_stop_primary_commits_one_target_row_without_drafting():
    rt = _DSparkRuntime()
    callback = []
    out = generate_mtpk(
        rt,
        [9, 10],
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids={11},
        token_callback=lambda block: callback.extend(block),
    )
    assert out.tokens == [11]
    assert out.finish_reason == "stop"
    assert callback == []
    assert rt.target_forward_widths == [1, 1, 1]
    assert rt.proposal_widths == []
    assert rt.target_cache.offset == 3


def test_dspark_accepted_stop_commits_only_the_terminal_prefix():
    rt = _DSparkRuntime()
    callback = []
    out = generate_mtpk(
        rt,
        [9, 10],
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids={12},
        token_callback=lambda block: callback.extend(block),
    )
    assert out.tokens == [11, 12]
    assert out.finish_reason == "stop"
    assert callback == [11]
    assert rt.target_forward_widths == [1, 1, 2]
    assert rt.target_cache.trimmed == []
    assert out.stats.accepted_drafts == 1
    assert out.stats.drafted_tokens == 1


class _RejectedStopRuntime(_DSparkRuntime):
    def draft_deepseek_v4_dspark(self, hidden, token_ids, cache, *, start_pos):
        del hidden, cache, start_pos
        token = int(np.asarray(token_ids)[0])
        if token == 10:
            ids = mx.array([[token, token + 1, 99, 999, 999, 999]])
        else:
            ids = mx.array([[token, token + 1, token + 2, token + 3, 999, 999]])
        return ids, mx.zeros((1, 5, 1024)), mx.ones((1, 5))


def test_dspark_rejected_stop_is_trimmed_and_never_emitted():
    rt = _RejectedStopRuntime()
    out = generate_mtpk(
        rt,
        [9, 10],
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids={99},
    )
    assert out.tokens == [11, 12, 13, 14]
    assert 99 not in out.tokens
    assert rt.target_forward_widths == [1, 1, 2, 3]
    assert rt.target_cache.trimmed == [1]
    assert out.stats.rejected_drafts == 1
    assert out.stats.accepted_drafts == 2


@pytest.mark.parametrize(
    ("max_tokens", "expected_tokens", "expected_widths", "proposal_widths"),
    [
        (1, [11], [1, 1, 1], []),
        (2, [11, 12], [1, 1, 2], [2]),
    ],
)
def test_dspark_generation_tail_uses_only_the_remaining_target_rows(
    max_tokens, expected_tokens, expected_widths, proposal_widths
):
    rt = _DSparkRuntime()
    out = generate_mtpk(
        rt,
        [9, 10],
        max_tokens=max_tokens,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
    )
    assert out.tokens == expected_tokens
    assert rt.target_forward_widths == expected_widths
    assert rt.proposal_widths == proposal_widths
    assert rt.target_cache.offset == 2 + max_tokens


def test_dspark_abort_before_target_prefill_does_no_model_work():
    rt = _DSparkRuntime()
    out = generate_mtpk(
        rt,
        [9, 10],
        abort_check=lambda: True,
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
    )
    assert out.tokens == []
    assert rt.target_forward_widths == []
    assert rt.prefill_hidden is None


def test_dspark_abort_before_proposal_prefill_does_no_proposal_work():
    rt = _DSparkRuntime()
    decisions = iter([False, False, True])
    out = generate_mtpk(
        rt,
        [9, 10],
        abort_check=lambda: next(decisions),
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
    )
    assert out.tokens == []
    assert rt.target_forward_widths == [1, 1]
    assert rt.prefill_hidden is None


def test_dspark_prefill_callback_uses_standard_compute_and_wall_schema():
    rt = _DSparkRuntime()
    callbacks = []
    generate_mtpk(
        rt,
        [9, 10],
        max_tokens=1,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
        prefill_callback=callbacks.append,
    )
    assert [event["phase"] for event in callbacks] == ["started", "completed"]
    assert set(callbacks[0]) == {
        "phase",
        "tokens_done",
        "tokens_total",
        "cached_tokens",
        "new_prefill_tokens",
        "elapsed_s",
        "started_s",
    }
    assert set(callbacks[1]) == {
        "phase",
        "tokens_total",
        "cached_tokens",
        "new_prefill_tokens",
        "elapsed_s",
        "prompt_eval_time_s",
        "prefill_tok_s",
        "prefill_compute_tok_s",
        "prefill_wall_tok_s",
        "cache_hit",
    }
    assert callbacks[1]["prefill_compute_tok_s"] is not None
    assert callbacks[1]["prefill_wall_tok_s"] is not None


@pytest.mark.parametrize("raise_on_call", [1, 2])
def test_dspark_prefill_callback_failure_does_not_abort_generation(raise_on_call):
    rt = _DSparkRuntime()
    calls = 0

    def failing_callback(_event):
        nonlocal calls
        calls += 1
        if calls == raise_on_call:
            raise RuntimeError("dashboard callback failed")

    out = generate_mtpk(
        rt,
        [9, 10],
        max_tokens=1,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
        prefill_callback=failing_callback,
    )
    assert out.tokens == [11]


def test_dspark_rejection_after_required_seed_trims_target_suffix():
    rt = _SecondProposalMissRuntime()
    out = generate_mtpk(
        rt,
        [10],
        max_tokens=4,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
    )
    assert out.tokens == [11, 12, 13, 14]
    assert rt.target_forward_widths[:3] == [1, 1, 3]
    assert rt.proposal_inputs[0] == (11, 1, 3)
    assert rt.target_cache.trimmed
    assert out.stats.rejected_drafts >= 2


def test_dspark_does_not_collect_per_cycle_timing_or_event_dicts(monkeypatch):
    class _Clock:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            return float(self.calls)

    clock = _Clock()
    monkeypatch.setattr(native_speculation.time, "perf_counter", clock)

    short = generate_mtpk(
        _DSparkRuntime(),
        [9, 10],
        max_tokens=1,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
    )
    short_calls = clock.calls
    clock.calls = 0
    long = generate_mtpk(
        _DSparkRuntime(),
        [9, 10],
        max_tokens=10,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
    )
    assert clock.calls == short_calls
    assert short.stats.events == []
    assert long.stats.events == []
    for stats in (short.stats, long.stats):
        assert stats.verify_time_s == 0.0
        assert stats.verify_forward_time_s == 0.0
        assert stats.verify_eval_time_s == 0.0
        assert stats.verify_joint_eval_time_s == 0.0
        assert stats.draft_time_s == 0.0
        assert stats.target_forward_time_s == 0.0
        assert stats.accept_time_s == 0.0
        assert stats.rollback_time_s == 0.0
        assert stats.commit_time_s == 0.0


def test_dspark_prefill_component_timing_accumulates_exact_chunk_boundaries(
    monkeypatch,
):
    class _Clock:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            value = float(self.calls)
            self.calls += 1
            return value

    clock = _Clock()
    monkeypatch.setattr(native_speculation.time, "perf_counter", clock)
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "128")
    out = generate_mtpk(
        _DSparkRuntime(),
        list(range(300)),
        max_tokens=0,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
    )
    assert out.stats.prompt_target_prefill_time_s == 4.0
    assert out.stats.prompt_mtp_history_time_s == 3.0
    assert out.stats.prompt_eval_time_s == 7.0
    assert out.stats.prompt_eval_time_s == (
        out.stats.prompt_target_prefill_time_s + out.stats.prompt_mtp_history_time_s
    )
    assert out.stats.prompt_target_prefill_tok_s == 75.0


def test_dspark_rollback_parameter_names_rejected_rows():
    parameters = inspect.signature(DeepseekV4DSparkBackend.rollback_target).parameters
    assert "rejected_rows" in parameters
    assert "verified_rows" not in parameters


@pytest.mark.parametrize("prompt_length", [9, 129, 256, 300, 385])
def test_dspark_prefill_matches_ar_spans_and_streams_exact_dspark_chunks(
    monkeypatch, prompt_length
):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "128")
    rt = _DSparkRuntime()
    prompt = list(range(prompt_length))
    out = generate_mtpk(
        rt,
        prompt,
        max_tokens=0,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
    )
    assert out.tokens == []
    ar_spans = generation_module._iter_prefill_chunk_spans(prompt_length - 1)
    assert rt.target_forward_widths == [
        *(end - start for start, end in ar_spans),
        1,
    ]
    target_ids = np.concatenate(rt.target_forward_inputs, axis=1)
    np.testing.assert_array_equal(target_ids, np.array([prompt]))
    assert rt.target_forward_options == [
        {"return_hidden": True, "emit_logits": False, "logits_keep": None}
        for _ in ar_spans
    ] + [
        {"return_hidden": True, "emit_logits": True, "logits_keep": 1},
    ]
    assert [
        (start_pos, int(hidden.shape[1]))
        for start_pos, hidden in rt.proposal_prefill_spans
    ] == [
        (start, min(128, prompt_length - start))
        for start in range(0, prompt_length, 128)
    ]
    proposal_hidden = np.concatenate(
        [hidden for _, hidden in rt.proposal_prefill_spans], axis=1
    )
    np.testing.assert_array_equal(
        proposal_hidden,
        np.repeat(np.asarray(prompt, dtype=np.float32)[None, :, None], 3, axis=2),
    )
    assert all(entry.prefill_length == prompt_length for entry in rt.proposal_cache)
    assert rt.target_cache.offset == prompt_length


def test_dspark_target_spans_follow_unchunked_ar_body(monkeypatch):
    monkeypatch.delenv("MTPLX_SUSTAINED_PREFILL", raising=False)
    rt = _DSparkRuntime()
    generate_mtpk(
        rt,
        list(range(300)),
        max_tokens=0,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
    )
    assert generation_module._iter_prefill_chunk_spans(299) == [(0, 299)]
    assert rt.target_forward_widths == [299, 1]
    assert [
        (start_pos, int(hidden.shape[1]))
        for start_pos, hidden in rt.proposal_prefill_spans
    ] == [(0, 128), (128, 128), (256, 44)]


def test_dspark_nine_token_prefill_matches_ar_body_then_final_contract(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "128")
    rt = _DSparkRuntime()
    generate_mtpk(
        rt,
        list(range(9)),
        max_tokens=0,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
    )
    assert rt.target_forward_widths == [8, 1]
    assert [rows[0].tolist() for rows in rt.target_forward_inputs] == [
        list(range(8)),
        [8],
    ]
    assert rt.target_forward_options == [
        {"return_hidden": True, "emit_logits": False, "logits_keep": None},
        {"return_hidden": True, "emit_logits": True, "logits_keep": 1},
    ]
    np.testing.assert_array_equal(
        rt.prefill_hidden,
        np.repeat(np.arange(9, dtype=np.float32)[None, :, None], 3, axis=2),
    )
    assert rt.commits == []


def test_dspark_long_prefill_preserves_decode_position_arithmetic(monkeypatch):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "128")
    rt = _DSparkRuntime()
    out = generate_mtpk(
        rt,
        list(range(300)),
        max_tokens=3,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
    )
    assert out.tokens == [300, 301, 302]
    assert rt.target_forward_widths == [128, 128, 43, 1, 3]
    assert rt.proposal_inputs == [(299, 299, 3)]
    assert rt.commits[-1][0] == 300
    assert rt.target_cache.offset == 303


@pytest.mark.parametrize(
    ("decisions", "expected_target_widths", "expected_proposal_spans"),
    [
        ([True], [], []),
        ([False, True], [128], []),
        ([False, False, True], [128], [(0, 128)]),
        ([False, False, False, True], [128, 128], [(0, 128)]),
        (
            [False, False, False, False, True],
            [128, 128],
            [(0, 128), (128, 128)],
        ),
        (
            [False, False, False, False, False, True],
            [128, 128, 43],
            [(0, 128), (128, 128)],
        ),
        (
            [False, False, False, False, False, False, True],
            [128, 128, 43, 1],
            [(0, 128), (128, 128)],
        ),
    ],
)
def test_dspark_long_prefill_abort_checks_each_target_and_proposal_chunk(
    monkeypatch, decisions, expected_target_widths, expected_proposal_spans
):
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "128")
    rt = _DSparkRuntime()
    decisions = iter(decisions)
    out = generate_mtpk(
        rt,
        list(range(300)),
        abort_check=lambda: next(decisions),
        max_tokens=0,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
    )
    assert out.tokens == []
    assert rt.target_forward_widths == expected_target_widths
    assert [
        (start_pos, int(hidden.shape[1]))
        for start_pos, hidden in rt.proposal_prefill_spans
    ] == expected_proposal_spans


@pytest.mark.parametrize("prompt_length", [9, 129, 256, 300, 385])
def test_dspark_chunked_prefill_ring_matches_one_shot_across_wraps(prompt_length):
    values = mx.arange(prompt_length, dtype=mx.float32).reshape(1, prompt_length, 1)
    one_shot = DeepseekV4DSparkCache(window_size=128, head_dim=1)
    one_shot.prefill(values)

    chunked = DeepseekV4DSparkCache(window_size=128, head_dim=1)
    chunked.prefill(values[:, :128])
    for start_pos in range(128, prompt_length, 128):
        chunked.commit_main(start_pos, values[:, start_pos : start_pos + 128])

    np.testing.assert_array_equal(np.asarray(chunked.ring), np.asarray(one_shot.ring))


def test_dspark_verify_calls_counts_physical_target_cycles():
    rt = _DSparkRuntime()
    out = generate_mtpk(
        rt,
        [9, 10],
        max_tokens=3,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=2,
        stop_token_ids=set(),
    )
    assert rt.target_forward_widths == [1, 1, 3]
    assert out.stats.accepted_drafts == 2
    # One K2 proposal/verification cycle owns one physical target-M3 forward.
    assert out.stats.verify_calls == 1
