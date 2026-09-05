"""MTP heads are extracted from all three layouts found in the wild.

Upstream ``is_mtp_key`` matched only ``mtp.`` and ``language_model.mtp.``.
Checkpoints that store the head as appended decoder layers (GLM-4 MoE,
DeepSeek-V3.2) or in their own namespace (MiMo) yielded no keys, so forge
wrote no sidecar -- while ``_inspection_has_mtp_weight_evidence`` reported a
head was present.  Key counts below are the real numbers from each published
checkpoint's ``model.safetensors.index.json``.
"""

from __future__ import annotations

from mtplx.artifacts import (
    appended_mtp_layer_range,
    is_appended_layer_mtp_key,
    is_mtp_layers_namespace_key,
    uses_appended_layer_mtp,
)

GLM_47_FLASH = {"num_hidden_layers": 47, "num_nextn_predict_layers": 1}
DEEPSEEK_V32 = {"num_hidden_layers": 61, "num_nextn_predict_layers": 1}
MIMO_7B = {"num_hidden_layers": 36, "num_nextn_predict_layers": 1}
NO_MTP = {"num_hidden_layers": 32}


def test_glm_appended_layer_is_the_layer_after_the_trunk():
    assert list(appended_mtp_layer_range(GLM_47_FLASH)) == [47]
    assert is_appended_layer_mtp_key("model.layers.47.shared_head.head.weight", GLM_47_FLASH)


def test_trunk_layers_are_not_mistaken_for_the_head():
    assert not is_appended_layer_mtp_key("model.layers.46.self_attn.q_proj.weight", GLM_47_FLASH)
    assert not is_appended_layer_mtp_key("model.layers.4.mlp.gate_proj.weight", GLM_47_FLASH)


def test_deepseek_v32_appends_after_sixty_one_layers():
    assert list(appended_mtp_layer_range(DEEPSEEK_V32)) == [61]
    assert is_appended_layer_mtp_key("model.layers.61.self_attn.q_a_proj.weight", DEEPSEEK_V32)


def test_mimo_keeps_the_head_in_its_own_namespace():
    assert is_mtp_layers_namespace_key("model.mtp_layers.0.token_layernorm.weight", MIMO_7B)
    assert is_mtp_layers_namespace_key("model.mtp_layers.0.input_proj.weight", MIMO_7B)


def test_the_two_layouts_do_not_claim_each_others_keys():
    # MiMo's trunk has 36 layers, so model.layers.36.* would be the appended
    # form; its head is not stored there and must not be matched by it.
    assert not is_appended_layer_mtp_key("model.mtp_layers.0.input_proj.weight", MIMO_7B)
    assert not is_mtp_layers_namespace_key("model.layers.47.embed_tokens.weight", GLM_47_FLASH)


def test_a_config_without_an_mtp_head_matches_nothing():
    assert not uses_appended_layer_mtp(NO_MTP)
    assert list(appended_mtp_layer_range(NO_MTP)) == []
    assert not is_appended_layer_mtp_key("model.layers.32.mlp.up_proj.weight", NO_MTP)
    assert not is_mtp_layers_namespace_key("model.mtp_layers.0.input_proj.weight", NO_MTP)
