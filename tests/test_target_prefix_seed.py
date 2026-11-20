"""A request's seed fixes the answer of the target-prefix lane.

Every token that lane commits is a target id sampled on the device before
acceptance (a kept draft equals it; a correction and the bonus are it). Those
draws took MLX's process-wide generator, which MLX seeds from the clock, so
the request's seed reached only the host generator, which on this lane draws
nothing that is committed: one seed gave a different answer in every server
process and on every request of one server. Seen as the 35B-A3B pair of the
16 November window q4w2 (0 of 4 answers equal, greedy equal); shipped in
2.12.3, whose 35B default lane is this one. The draws now take their key from
the request's seeded generator, as the other lanes' host draws do.

Class B: each draw is still one categorical draw from the same top-k/top-p
support, so the answer's distribution is unchanged; which tokens a seed gives
changed. Greedy draws nothing and is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mtplx.fast_sampling import device_sample_key, sample_token_ids_from_mlx_logits
from mtplx.sampling import SamplerConfig

FAMILY = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
NO_TOP_K = SamplerConfig(temperature=0.6, top_p=1.0, top_k=0)
GREEDY = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)

gpu = pytest.mark.skipif(not mx.metal.is_available(), reason="bfloat16 expert gathers need the GPU")


def _rows(count: int = 64, vocab: int = 256) -> mx.array:
    # Flat logits: every one of the top-k ids is likely, so a draw that
    # ignores its key cannot repeat another by chance.
    return mx.random.normal((count, vocab), key=mx.random.key(3)) * 0.1


def _ids(array: mx.array) -> list[int]:
    return [int(token) for token in np.asarray(array).reshape(-1)]


def test_the_key_comes_from_the_seeded_generator():
    first = device_sample_key(np.random.default_rng(17), FAMILY)
    again = device_sample_key(np.random.default_rng(17), FAMILY)
    other = device_sample_key(np.random.default_rng(18), FAMILY)
    assert np.array_equal(np.asarray(first), np.asarray(again))
    assert not np.array_equal(np.asarray(first), np.asarray(other))
    # A greedy sampler draws nothing, so it takes nothing from the generator.
    rng = np.random.default_rng(17)
    assert device_sample_key(rng, GREEDY) is None
    assert rng.integers(0, 2**31 - 1) == np.random.default_rng(17).integers(0, 2**31 - 1)


@pytest.mark.parametrize("sampler", [FAMILY, NO_TOP_K], ids=["top_k_top_p", "full_vocab"])
def test_a_keyed_draw_ignores_the_process_generator(sampler):
    rows = _rows()
    key = mx.random.key(1234)
    mx.random.seed(1)
    first = _ids(sample_token_ids_from_mlx_logits(rows, sampler, key=key))
    mx.random.seed(2)
    second = _ids(sample_token_ids_from_mlx_logits(rows, sampler, key=key))
    assert first == second
    # A draw without a key still follows the process generator (tools and
    # probes call it so), so reseeding it moves the draw.
    mx.random.seed(1)
    keyless = _ids(sample_token_ids_from_mlx_logits(rows, sampler))
    mx.random.seed(2)
    assert keyless != _ids(sample_token_ids_from_mlx_logits(rows, sampler))


def test_greedy_is_the_argmax_with_or_without_a_key():
    rows = _rows()
    expected = _ids(mx.argmax(rows, axis=-1))
    assert _ids(sample_token_ids_from_mlx_logits(rows, GREEDY)) == expected
    assert _ids(sample_token_ids_from_mlx_logits(rows, GREEDY, key=mx.random.key(5))) == expected


class _Tokenizer:
    def decode(self, ids):
        return " ".join(str(int(i)) for i in ids)


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    pytest.importorskip("mlx_lm.models.qwen3_5_moe")
    from mtplx.mtp_patch import MTPContract
    from mtplx.runtime import MTPLXRuntime
    from tests.a3b_tiny_synth import tiny_model_with_draft_head

    model = tiny_model_with_draft_head(tmp_path_factory.mktemp("target-prefix-seed"))
    return MTPLXRuntime(
        model=model,
        tokenizer=_Tokenizer(),
        model_path=Path("models/tiny-a3b-target-prefix-seed"),
        mtp_enabled=True,
        contract=MTPContract(),
    )


@pytest.fixture(autouse=True)
def _plain_lane(monkeypatch):
    # Prompt-lookup drafting would change the drafts, not the draws; it is
    # held off so a difference can only come from the draws.
    monkeypatch.setenv("MTPLX_CONTEXT_COPY", "0")


def _answer(rt, *, seed: int, sampler=FAMILY, depth: int = 1):
    from mtplx import generation
    from tests.a3b_tiny_synth import prompt

    return generation.generate_mtpk(
        rt,
        prompt(40),
        max_tokens=48,
        sampler=sampler,
        draft_sampler=sampler,
        speculative_depth=depth,
        seed=seed,
        mtp_cache_policy="persistent",
        mtp_history_policy="committed",
        verify_strategy="target_prefix",
        stop_token_ids=set(),
    )


@gpu
@pytest.mark.parametrize("depth", [1, 2])
def test_one_seed_gives_one_answer_on_the_target_prefix_lane(runtime, depth):
    first = _answer(runtime, seed=17, depth=depth)
    # Whatever the process generator did in between (another request, a
    # kernel self-check reseeding it), the seed alone decides the answer.
    mx.random.seed(99)
    mx.random.categorical(mx.zeros((4, 8)))
    again = _answer(runtime, seed=17, depth=depth)
    assert len(first.tokens) == 48
    assert again.tokens == first.tokens
    # And the seed does decide it.
    assert _answer(runtime, seed=18, depth=depth).tokens != first.tokens


@gpu
def test_greedy_on_the_target_prefix_lane_is_unchanged_by_the_seed(runtime):
    first = _answer(runtime, seed=17, sampler=GREEDY)
    assert _answer(runtime, seed=18, sampler=GREEDY).tokens == first.tokens
