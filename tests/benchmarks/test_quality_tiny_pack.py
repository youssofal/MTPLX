"""The pack pipeline end to end on a four-layer random model of the 35B-A3B's
own architecture (about a megabyte): the one-load measurement writes the
reference scores, the floor and a row per group in the formats the recipe and
the divergence comparison read; forge's converter applies a recipe's per-group
bits and clipped rounding exactly as the recipe says."""

from __future__ import annotations

import json

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mtplx.benchmarks.quality.recipe import choices_from_rows
from mtplx.benchmarks.quality.scoring import compare
from mtplx.benchmarks.quality.sensitivity import measure

TEXT = {
    "model_type": "qwen3_5_moe_text", "hidden_size": 64, "num_hidden_layers": 4, "num_attention_heads": 2,
    "num_key_value_heads": 1, "head_dim": 32, "linear_num_value_heads": 2, "linear_num_key_heads": 1,
    "linear_key_head_dim": 32, "linear_value_head_dim": 32, "linear_conv_kernel_dim": 4,
    "full_attention_interval": 4, "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 64, "vocab_size": 256, "intermediate_size": 64, "tie_word_embeddings": False,
    "rms_norm_eps": 1e-6, "max_position_embeddings": 4096,
    "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25,
                        "rope_theta": 10000000, "rope_type": "default"},
}


@pytest.fixture()
def tiny_source(tmp_path):
    from mlx_lm.models.qwen3_5_moe import Model, ModelArgs

    mx.random.seed(11)
    config = {"model_type": "qwen3_5_moe", "text_config": TEXT}
    model = Model(ModelArgs.from_dict(config))
    weights = dict(tree_flatten(model.parameters()))
    mx.eval(weights)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(json.dumps(config))
    # The converter saves a tokenizer beside the weights: a word-level one over the tiny vocabulary.
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel(vocab={f"t{i}": i for i in range(256)}, unk_token="t0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}))
    return tmp_path


def test_one_load_writes_reference_scores_floor_and_group_rows(tiny_source):
    mx.random.seed(5)
    corpus = [{"id": f"e{i}", "token_ids": mx.random.randint(0, 256, (90,)).tolist(), "spans": [[20, 50], [60, 90]]}
              for i in range(2)]
    rows, scores = [], []
    measure(tiny_source, corpus, positions=40, context=None, bits=[4], expert_bits=[], band=4, chunk=32,
            emit=rows.append, write_score=scores.append, text_of=str, top_k=8, rounding="clip",
            witness=("output head",), extra=(("output head", 8), ("routers", 4)))

    by_group = {(r["group"], r.get("bits"), r.get("rounding")): r for r in rows}
    assert rows[0]["group"] == "reference" and rows[0]["scored_positions"] == 2 * 60
    floor = by_group[("floor (no weight changed)", None, None)]
    assert floor["top1_agreement"] == 1.0 and floor["kl_mean"] < 1e-3
    groups = {r["group"] for r in rows if "bits" in r}
    assert groups == {"routed experts, layers 0-3", "shared experts", "linear attention", "full attention",
                      "routers", "embeddings", "output head"}
    assert ("output head", 4, "plain") in by_group and ("output head", 4, "clip") in by_group
    assert all(r["kl_mean"] > floor["kl_mean"] for r in rows if r.get("bits") == 4)
    # An extra width is measured once, at the pack's rounding; one already planned is not read twice.
    assert by_group[("output head", 8, "clip")]["kl_mean"] < by_group[("output head", 4, "clip")]["kl_mean"]
    assert sum(r["group"] == "routers" for r in rows) == 1
    with pytest.raises(ValueError):
        measure(tiny_source, corpus, positions=40, context=None, bits=[4], expert_bits=[], band=4, chunk=32,
                emit=rows.append, write_score=scores.append, text_of=str, extra=(("router", 8),))
    # The recipe reads the rows of its rounding; the reference scores compare like a server's.
    assert len(choices_from_rows(rows, "clip")) == 7 and len(choices_from_rows(rows, "plain")) == 1
    assert [len(s["positions"]) for s in scores] == [60, 60]
    assert compare(scores, scores)[0].mean_kl == 0


def test_forge_converts_a_recipe_with_its_bits_and_clipped_rounding(tiny_source, tmp_path):
    from mlx_lm.utils import load_model

    from mtplx.commands import forge_mixed_convert
    from mtplx.quant_rounding import clip_for_rounding

    recipe = {"body_bits": 8, "body_group_size": 64, "body_mode": "affine", "rounding": "clip", "module_overrides": [
        {"suffix": "switch_mlp.down_proj", "layers": [0, 1], "bits": 4, "group_size": 64},
        {"suffix": "lm_head", "bits": 4, "group_size": 64},
        {"suffix": "mlp.gate", "bits": 8, "group_size": 64},
    ]}
    dest = tmp_path / "pack"
    assert forge_mixed_convert.main(["--source", str(tiny_source), "--destination", str(dest),
                                     "--recipe-json", json.dumps(recipe)]) == 0
    quantization = json.loads((dest / "config.json").read_text())["quantization"]
    assert quantization["bits"] == 8
    assert quantization["language_model.model.layers.1.mlp.switch_mlp.down_proj"]["bits"] == 4
    assert "language_model.model.layers.2.mlp.switch_mlp.down_proj" not in quantization  # the body's 8 bits

    # What the pack stores is the clipped rounding of the source weight, bit for bit.
    stored = mx.load(str(next(dest.glob("model*.safetensors"))))
    source = mx.load(str(tiny_source / "model.safetensors"))["language_model.lm_head.weight"]
    packed, scales, biases = mx.quantize(clip_for_rounding(source, bits=4, group_size=64), group_size=64, bits=4)
    assert mx.array_equal(stored["language_model.lm_head.weight"], packed).item()
    assert mx.array_equal(stored["language_model.lm_head.scales"], scales).item()

    # And the pack loads and answers close to the source.
    original, _ = load_model(tiny_source)
    converted, _ = load_model(dest)
    tokens = mx.array([[3, 17, 42, 99, 5, 7]])
    a = original(tokens).astype(mx.float32)
    b = converted(tokens).astype(mx.float32)
    p, q = (x - mx.logsumexp(x, axis=-1, keepdims=True) for x in (a, b))
    assert float(mx.sum(mx.exp(p) * (p - q), axis=-1).mean().item()) < 0.05
