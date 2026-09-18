"""Unit coverage for the Forge qwen4_exp (Qwen3.8-Flash-Next) convert lane.

Pure-arithmetic pieces only: the sidecar header, the MTP key layout and the
per-module quant rules. The streaming conversions need the 335 GB source.
"""

from __future__ import annotations

import pytest

from mtplx.commands.forge_qwen4_exp import (
    NGRAM_PRODUCTION_LAYOUT,
    Qwen4ForgeError,
    is_qwen4_exp_source,
    mtp_layout_keys,
    mtp_module_rule,
    ngram_sidecar_layout,
    recipe_params,
)

# Qwen3.8-Flash-Next: 128 shards x 2,500,012 rows x 160 = the official pack's row count.
ROWS = 128 * 2_500_012
DIM = 160


def test_is_qwen4_exp_source_reads_either_level() -> None:
    assert is_qwen4_exp_source({"model_type": "qwen4_exp"})
    assert is_qwen4_exp_source({"model_type": "", "text_config": {"model_type": "qwen4_exp"}})
    assert not is_qwen4_exp_source({"model_type": "qwen3_next"})


def test_recipe_defaults_match_official_pack() -> None:
    params = recipe_params({"body_bits": 4, "body_group_size": 32, "body_mode": "affine"})
    assert params["body_bits"] == 4 and params["body_group"] == 32
    assert (params["ngram_bits"], params["ngram_group"]) == NGRAM_PRODUCTION_LAYOUT
    # The head follows the trunk unless pinned: measured identical greedy drafts
    # at bf16 and 4-bit, bf16 only costs bandwidth.
    assert params["mtp_bits"] == 4
    assert recipe_params({"body_bits": 4, "qwen4_mtp_bits": 0})["mtp_bits"] == 0


def test_recipe_refuses_non_affine_and_bf16_trunk() -> None:
    with pytest.raises(Qwen4ForgeError):
        recipe_params({"body_bits": 4, "body_mode": "mxfp4"})
    with pytest.raises(Qwen4ForgeError):
        recipe_params({"body_bits": 0})


def test_ngram_layout_matches_attach_sidecar_contract() -> None:
    header = ngram_sidecar_layout(ROWS, DIM, bits=4, group=32)
    assert header["__metadata__"] == {
        "ngram_bits": "4", "ngram_group_size": "32", "rows": str(ROWS), "dim": "160",
    }
    assert header["ngram.weight"]["dtype"] == "U32" and header["ngram.weight"]["shape"] == [ROWS, 20]
    assert header["ngram.scales"]["shape"] == [ROWS, 5] and header["ngram.biases"]["shape"] == [ROWS, 5]
    w, s, b = (header[k]["data_offsets"] for k in ("ngram.weight", "ngram.scales", "ngram.biases"))
    assert w[0] == 0 and w[1] == s[0] and s[1] == b[0]
    assert b[1] == ROWS * (20 * 4 + 5 * 2 + 5 * 2)  # 32,000,153,600 bytes: the 29.8 GiB official table


def test_ngram_layout_other_widths_and_raw() -> None:
    assert ngram_sidecar_layout(ROWS, DIM, bits=3, group=32)["ngram.weight"]["shape"] == [ROWS, 15]
    raw = ngram_sidecar_layout(ROWS, DIM, bits=0, group=32)
    assert set(raw) == {"__metadata__", "ngram.weight"} and raw["ngram.weight"]["dtype"] == "BF16"
    with pytest.raises(Qwen4ForgeError):
        ngram_sidecar_layout(ROWS, DIM, bits=4, group=64)  # 160 % 64


def test_mtp_layout_splits_packed_experts_like_sanitize() -> None:
    keys = mtp_layout_keys([
        "mtp.fc_hidden.weight",
        "mtp.layers.0.mlp.experts.gate_up_proj",
        "mtp.layers.0.mlp.experts.down_proj",
        "mtp.layers.0.self_attn.q_norm.weight",
    ])
    assert keys == [
        "fc_hidden.weight",
        "layers.0.mlp.switch_mlp.gate_proj.weight",
        "layers.0.mlp.switch_mlp.up_proj.weight",
        "layers.0.mlp.switch_mlp.down_proj.weight",
        "layers.0.self_attn.q_norm.weight",
    ]


def test_mtp_rules_mirror_trunk_recipe() -> None:
    rule = lambda k, **kw: mtp_module_rule(k, mtp_bits=4, mtp_group=32, **kw)
    assert rule("layers.0.mlp.switch_mlp.gate_proj.weight") == (4, 32)
    assert rule("layers.0.mlp.gate.weight") == (8, 64)
    assert rule("layers.0.mlp.shared_expert.down_proj.weight") == (8, 64)
    assert rule("layers.0.self_attn.indexer.index_qk_proj.weight") == (8, 64)
    assert rule("layers.0.self_attn.q_proj.weight") == (4, 32)
    assert rule("layers.0.self_attn.q_proj.weight", qsa_8bit=True) == (8, 64)
    assert rule("layers.0.attn_hyper_connection.input_mix_weight_down.weight") is None
    assert rule("fc_hidden.weight") is None
    assert mtp_module_rule("layers.0.mlp.switch_mlp.up_proj.weight", mtp_bits=0, mtp_group=32) is None
