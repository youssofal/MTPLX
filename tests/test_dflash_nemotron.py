from __future__ import annotations

from types import SimpleNamespace

from mtplx.backends.descriptors import descriptor_for_backend_id
from mtplx.backends.dflash import generate_dflash
from mtplx.models.dflash import DFlashConfig, normalize_dflash_weights
from mtplx.server import openai


def test_nvidia_dflash_config_maps_to_runtime_contract():
    cfg = DFlashConfig.from_dict(
        {
            "hidden_size": 2688,
            "num_hidden_layers": 6,
            "intermediate_size": 6144,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "max_position_embeddings": 1048576,
            "rms_norm_eps": 1e-6,
            "mask_token_id": 990,
            "target_layer_ids": [1, 5, 19, 29, 41, 51],
            "has_embed_tokens": True,
            "rope_parameters": {
                "factor": 128.0,
                "original_max_position_embeddings": 8192,
                "rope_theta": 10000,
                "rope_type": "yarn",
            },
            "dflash_config": {
                "causal": False,
                "target_layer_ids": [1, 5, 19, 29, 41, 51],
            },
        }
    )

    assert cfg.block_size == 8
    assert cfg.target_layers == [1, 5, 19, 29, 41, 51]
    assert cfg.target_layer_offset == 0
    assert cfg.has_embed_tokens is True
    assert cfg.causal is False
    assert cfg.rope_theta == 10000
    assert cfg.rope_scaling == {
        "factor": 128.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "yarn",
    }


def test_nvidia_dflash_weight_names_map_to_existing_module_tree():
    normalized = normalize_dflash_weights(
        {
            "hidden_norm.weight": object(),
            "layers.0.input_layernorm.weight": object(),
            "layers.0.post_attention_layernorm.weight": object(),
            "layers.0.self_attn.q_proj.weight": object(),
        }
    )

    assert set(normalized) == {
        "enc_norm.weight",
        "layers.0.attn_norm.weight",
        "layers.0.ffn_norm.weight",
        "layers.0.self_attn.q_proj.weight",
    }


def test_dflash_server_descriptor_uses_external_drafter_without_mtp_head():
    descriptor = descriptor_for_backend_id("dflash")

    assert descriptor.backend_id == "dflash"
    assert descriptor.uses_external_assistant is True
    assert descriptor.uses_draft_lm_head is False
    assert descriptor.draft_semantics.request_field == "speculative_depth"


def test_generate_dflash_keeps_installed_block_and_reports_verification_stats():
    class FakeRuntime:
        config = SimpleNamespace(block_size=8)

        def generate(self, prompt, *, max_tokens, stop_token_ids, token_callback):
            assert prompt == [1, 2]
            assert max_tokens == 4
            assert stop_token_ids == set()
            assert self.config.block_size == 8
            return {
                "text": "done",
                "tokens": [3, 4, 5, 6],
                "rounds": 2,
                "accepted": 5,
                "drafted": 14,
                "rejected": 9,
                "mean_accept": 2.5,
                "tokens_per_target_step": 3.5,
            }

    runtime = FakeRuntime()
    output = generate_dflash(
        runtime,
        [1, 2],
        max_tokens=4,
        speculative_depth=3,
    )

    assert runtime.config.block_size == 8
    assert output.stats.accepted_drafts == 5
    assert output.stats.drafted_tokens == 14
    assert output.stats.rejected_drafts == 9
    assert output.stats.verify_calls == 2
    assert output.stats.speculative_depth == 7
    assert output.stats.requested_speculative_depth == 7


def test_dflash_skips_unsupported_retokenized_session_postcommit():
    state = SimpleNamespace(
        backend_descriptor=descriptor_for_backend_id("dflash")
    )

    skipped = openai._skipped_idle_postcommit_snapshot(
        state=state,
        unsafe_reason="missing_generation_final_state",
        prompt_prefix_len=12,
    )

    assert skipped == {
        "stored": False,
        "mode": "skipped",
        "reason": "dflash_retokenized_postcommit_unsupported",
        "unsafe_reason": "missing_generation_final_state",
        "assistant_tool_calls": 0,
        "prompt_prefix_len": 12,
    }


def test_dflash_request_uses_backend_block_default_not_native_mtp_depth():
    state = SimpleNamespace(
        args=SimpleNamespace(depth=3),
        backend_descriptor=descriptor_for_backend_id("dflash"),
    )

    block_size = openai._request_depth_for_generation(
        state,
        openai.ChatCompletionRequest(),
        generation_mode="mtp",
    )

    assert block_size == 8
