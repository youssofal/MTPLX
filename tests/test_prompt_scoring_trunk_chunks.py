"""Prompt scoring with the trunk in prefill-sized forwards.

A tiny hybrid Qwen3.5-MoE (GatedDeltaNet and full-attention layers, eight
experts, random weights) with the draft head injected the product way. The
legacy route (256-row forwards that emit their own logits) is the reference;
the new route runs the trunk without logits and applies the lm_head per
256-row slice of its hidden rows.
"""

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mtplx import generation
from mtplx.generation import score_prompt_logprobs
from mtplx.mtp_patch import MTPContract, inject_mtp_support
from mtplx.runtime import MTPLXRuntime

VOCAB = 1000  # not a multiple of the top-K prefilter block
BOUNDARY_LENGTHS = [1, 255, 256, 257, 2047, 2048, 2049, 8192]


def _text_args():
    from mlx_lm.models.qwen3_5 import TextModelArgs

    return TextModelArgs(
        model_type="qwen3_5_moe",
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        vocab_size=VOCAB,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        tie_word_embeddings=False,
        full_attention_interval=2,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        max_position_embeddings=16384,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
        },
    )


def _hybrid_moe_runtime(tmp_path, dtype) -> MTPLXRuntime:
    from mlx_lm.models.qwen3_5 import DecoderLayer, TextModel

    args = _text_args()
    mx.random.seed(3)
    model = TextModel(args)
    # A peaked head, so rows look like a language model's distributions.
    model.lm_head.weight = model.lm_head.weight * 8.0
    model.set_dtype(dtype)
    mx.eval(model.parameters())
    donor = DecoderLayer(args, layer_idx=args.full_attention_interval - 1)
    hidden = args.hidden_size
    tensors = {
        "mtp.fc.weight": mx.random.normal((hidden, hidden * 2)) * 0.02,
        "mtp.norm.weight": mx.ones((hidden,)),
        "mtp.pre_fc_norm_hidden.weight": mx.ones((hidden,)),
        "mtp.pre_fc_norm_embedding.weight": mx.ones((hidden,)),
    }
    for path, value in tree_flatten(donor.parameters()):
        tensors[f"mtp.layers.0.{path}"] = value
    mx.save_safetensors(str(tmp_path / "mtp.safetensors"), tensors)
    config = {
        "model_type": "qwen3_5_moe",
        "mtp_num_hidden_layers": 1,
        "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
        "tie_word_embeddings": False,
        **{
            key: getattr(args, key)
            for key in (
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "vocab_size",
                "linear_num_value_heads",
                "linear_num_key_heads",
                "linear_key_head_dim",
                "linear_value_head_dim",
                "linear_conv_kernel_dim",
                "full_attention_interval",
                "num_experts",
                "num_experts_per_tok",
                "moe_intermediate_size",
                "shared_expert_intermediate_size",
            )
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert inject_mtp_support(model, tmp_path, config) is True
    return MTPLXRuntime(
        model=model,
        tokenizer=None,
        model_path=tmp_path,
        mtp_enabled=True,
        contract=MTPContract(),
    )


@pytest.fixture(scope="module", params=[mx.float32, mx.bfloat16], ids=["f32", "bf16"])
def runtime(request, tmp_path_factory):
    return _hybrid_moe_runtime(tmp_path_factory.mktemp("hybrid"), request.param)


def _legacy_scores(rt, prompt_ids, monkeypatch, top_k=20):
    with monkeypatch.context() as patch:
        patch.setattr(generation, "_post_norm_logits_head", lambda _rt: None)
        return score_prompt_logprobs(rt, prompt_ids, top_k=top_k)


def _prompt(length: int) -> list[int]:
    return np.random.default_rng(length).integers(0, VOCAB, length).tolist()


def test_logits_from_post_norm_is_the_forwards_own_lm_head(runtime):
    tokens = mx.array([_prompt(40)])

    logits, hidden = runtime.forward_ar(
        tokens,
        cache=runtime.make_cache(),
        return_hidden=True,
        hidden_variant="post_norm",
    )
    head = generation._post_norm_logits_head(runtime)

    assert head is not None
    assert mx.array_equal(head(hidden), logits).item()


@pytest.mark.parametrize("length", BOUNDARY_LENGTHS)
def test_head_split_at_256_rows_is_bitwise_the_legacy_route(
    runtime, length, monkeypatch
):
    prompt_ids = _prompt(length)

    legacy = _legacy_scores(runtime, prompt_ids, monkeypatch)
    split = score_prompt_logprobs(
        runtime, prompt_ids, top_k=20, trunk_chunk_size=256
    )

    assert split["positions"] == legacy["positions"]
    assert split["token_logprobs"] == legacy["token_logprobs"]


@pytest.mark.parametrize("length", BOUNDARY_LENGTHS)
def test_prefill_sized_trunk_matches_the_legacy_route(runtime, length, monkeypatch):
    """Chunk layout is a rounding-class change at most: 1e-4 nats in float32,
    one bf16 step of a logit near 32 (0.25) in bfloat16. Top-1 ids must agree
    wherever the two best candidates are further apart than that."""

    prompt_ids = _prompt(length)
    atol = 1e-4 if runtime.model.lm_head.weight.dtype == mx.float32 else 0.25

    legacy = _legacy_scores(runtime, prompt_ids, monkeypatch)
    wide = score_prompt_logprobs(runtime, prompt_ids, top_k=20, trunk_chunk_size=2048)

    assert len(wide["positions"]) == len(legacy["positions"]) == length - 1
    assert len(wide["token_logprobs"]) == length - 1
    np.testing.assert_allclose(
        wide["token_logprobs"], legacy["token_logprobs"], rtol=0, atol=atol
    )
    for old_row, new_row in zip(legacy["positions"], wide["positions"]):
        old_vals = [value for _token, value in old_row]
        np.testing.assert_allclose(
            [value for _token, value in new_row], old_vals, rtol=0, atol=atol
        )
        if old_vals[0] - old_vals[1] > atol:
            assert new_row[0][0] == old_row[0][0]


def test_top_k_zero_still_scores_each_position_with_one_entry(runtime):
    scored = score_prompt_logprobs(
        runtime, _prompt(300), top_k=0, trunk_chunk_size=2048
    )

    assert len(scored["positions"]) == 299
    assert all(len(entries) == 1 for entries in scored["positions"])


def test_top_k_cap_returns_128_sorted_entries(runtime):
    scored = score_prompt_logprobs(
        runtime, _prompt(300), top_k=128, trunk_chunk_size=2048
    )

    for entries in scored["positions"]:
        values = [value for _token, value in entries]
        assert len(entries) == 128
        assert values == sorted(values, reverse=True)


def test_trunk_chunk_stays_256_without_the_invariant_lane(monkeypatch):
    from mtplx import batch_invariant_prefill

    monkeypatch.delenv("MTPLX_PROMPT_SCORE_TRUNK_CHUNK", raising=False)
    monkeypatch.setitem(batch_invariant_prefill._STATE, "installed", False)
    with generation.prefill_chunk_size_override(1536):
        assert generation._prompt_score_trunk_chunk_size() == 256


def test_trunk_chunk_follows_prefill_unless_overridden(monkeypatch):
    from mtplx import batch_invariant_prefill

    monkeypatch.delenv("MTPLX_PROMPT_SCORE_TRUNK_CHUNK", raising=False)
    monkeypatch.setitem(batch_invariant_prefill._STATE, "installed", True)
    with generation.prefill_chunk_size_override(1536):
        assert generation._prompt_score_trunk_chunk_size() == 1536

    monkeypatch.setenv("MTPLX_PROMPT_SCORE_TRUNK_CHUNK", "256")
    with generation.prefill_chunk_size_override(1536):
        assert generation._prompt_score_trunk_chunk_size() == 256


def test_runtime_without_a_separate_head_keeps_256_row_forwards():
    class _Model:
        def __init__(self):
            self.widths: list[int] = []

        def make_cache(self):
            return []

        def __call__(self, inputs, cache=None, return_hidden=False, **_kwargs):
            self.widths.append(int(inputs.shape[1]))
            logits = mx.zeros((1, inputs.shape[1], 8))
            return (logits, logits) if return_hidden else logits

    model = _Model()
    rt = MTPLXRuntime(
        model=model,
        tokenizer=None,
        model_path=None,
        mtp_enabled=True,
        contract=MTPContract(),
    )

    score_prompt_logprobs(
        rt, [i % 8 for i in range(600)], top_k=2, trunk_chunk_size=2048
    )

    assert model.widths == [256, 256, 88]


def test_lm_head_slices_run_in_the_prefill_phase(monkeypatch):
    from mtplx.attention_context import current_attention_phase

    phases: list[str] = []

    class _Model:
        def make_cache(self):
            return []

        def __call__(self, inputs, cache=None, return_hidden=False, **_kwargs):
            hidden = mx.zeros((1, inputs.shape[1], 4))
            return (None, hidden) if return_hidden else hidden

        def logits_from_post_norm(self, rows):
            phases.append(current_attention_phase())
            return mx.zeros((1, rows.shape[1], 8))

    rt = MTPLXRuntime(
        model=_Model(),
        tokenizer=None,
        model_path=None,
        mtp_enabled=True,
        contract=MTPContract(),
    )

    score_prompt_logprobs(rt, [i % 8 for i in range(300)], top_k=2, trunk_chunk_size=2048)

    assert phases == ["prefill", "prefill"]
