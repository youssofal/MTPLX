from __future__ import annotations

import os
from types import SimpleNamespace

from scripts.run_context_degradation_diagnostics import (
    _local_online_hidden_config,
    _local_online_hidden_server_args,
    _local_profile_env,
    _server_python_env,
)


def test_dirty_detach_profile_sets_gdn_conv_cadence_and_cache_limit():
    env, info = _local_profile_env(
        "detach_gdn_conv_every_4_cache_limit_1gb",
        {},
    )

    assert info["profile_type"] == "dirty_detach_probe"
    assert info["detach_components"] == ["gdn", "conv"]
    assert info["detach_every"] == 4
    assert env["MTPLX_DETACH_COMPONENTS"] == "gdn,conv"
    assert env["MTPLX_DETACH_GDN_EVERY"] == "4"
    assert env["MTPLX_DETACH_CONV_EVERY"] == "4"
    assert env["MTPLX_DETACH_MODE"] == "selected_slice_contiguous_eval"
    assert env["MTPLX_MLX_CACHE_LIMIT"] == "1gb"


def test_sustained_profile_sets_v015_memory_env_without_decode_state_flags():
    env, info = _local_profile_env("sustained", {})

    assert info["profile_type"] == "sustained"
    assert env["MTPLX_SUSTAINED_PREFILL"] == "1"
    assert env["MTPLX_PREFILL_CHUNK_SIZE"] == "auto"
    assert env["MTPLX_PREFILL_CHUNK_SIZE_DENSE"] == "2048"
    assert env["MTPLX_PREFILL_CHUNK_SIZE_REPAGE"] == "2048"
    assert env["MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS"] == "0"
    assert env["MTPLX_DYNAMIC_PAGED_KV"] == "1"
    assert env["MTPLX_VLLM_METAL_PAGED_ATTN_IMPL"] == "mlx_vector_paged"
    assert "MTPLX_TRUNK_CACHE_MATERIALIZE_EVERY" not in env
    assert "MTPLX_EVAL_STATE_ROOTS_ON_COMMIT" not in env


def test_server_python_env_prepends_repo_root_without_dropping_existing_path(tmp_path):
    existing = os.pathsep.join(["/already/there", "/elsewhere"])

    env = _server_python_env({"PYTHONPATH": existing}, tmp_path)

    parts = env["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == str(tmp_path.resolve())
    assert "/already/there" in parts
    assert "/elsewhere" in parts


def test_cache_limit_profile_sets_only_allocator_limit_probe():
    env, info = _local_profile_env("cache_limit_512mb", {})

    assert info["profile_type"] == "cache_limit_probe"
    assert env["MTPLX_MLX_CACHE_LIMIT"] == "512mb"
    assert "MTPLX_DETACH_COMPONENTS" not in env


def test_state_rebase_profile_sets_rebase_interval_and_cache_limit():
    env, info = _local_profile_env(
        "state_rebase_every_2048_cache_limit_1gb",
        {},
    )

    assert info["profile_type"] == "state_rebase_probe"
    assert info["state_rebase_every"] == 2048
    assert env["MTPLX_STATE_REBASE_EVERY"] == "2048"
    assert env["MTPLX_MLX_CACHE_LIMIT"] == "1gb"
    assert "MTPLX_DETACH_COMPONENTS" not in env


def test_dirty_detach_profile_sets_attention_eval_cadence():
    env, info = _local_profile_env(
        "detach_attn_every_64_mode_eval_only_cache_limit_4gb",
        {},
    )

    assert info["profile_type"] == "dirty_detach_probe"
    assert info["detach_components"] == ["attn"]
    assert env["MTPLX_DETACH_COMPONENTS"] == "attn"
    assert env["MTPLX_DETACH_ATTN_EVERY"] == "64"
    assert env["MTPLX_DETACH_MODE"] == "eval_only"
    assert env["MTPLX_MLX_CACHE_LIMIT"] == "4gb"


def test_capture_commit_detach_profile_sets_boundary_cadence():
    env, info = _local_profile_env(
        "capture_commit_detach_gdn_conv_every_16_cache_limit_4gb",
        {},
    )

    assert info["profile_type"] == "capture_commit_detach_probe"
    assert info["capture_commit_detach_components"] == ["gdn", "conv"]
    assert info["capture_commit_detach_every"] == 16
    assert env["MTPLX_CAPTURE_COMMIT_DETACH_COMPONENTS"] == "gdn,conv"
    assert env["MTPLX_CAPTURE_COMMIT_DETACH_GDN_EVERY"] == "16"
    assert env["MTPLX_CAPTURE_COMMIT_DETACH_CONV_EVERY"] == "16"
    assert env["MTPLX_CAPTURE_COMMIT_DETACH_MODE"] == "selected_slice_contiguous_eval"
    assert env["MTPLX_MLX_CACHE_LIMIT"] == "4gb"
    assert "MTPLX_DETACH_COMPONENTS" not in env


def test_owned_attention_tail_profile_sets_tail_owner_mode():
    env, info = _local_profile_env(
        "owned_attn_tail_mode_contiguous_eval_cache_limit_4gb",
        {},
    )

    assert info["profile_type"] == "owned_attn_tail_probe"
    assert info["owned_attn_kv"] == "tail"
    assert info["owned_attn_kv_mode"] == "contiguous_eval"
    assert env["MTPLX_OWNED_ATTN_KV"] == "tail"
    assert env["MTPLX_OWNED_ATTN_KV_MODE"] == "contiguous_eval"
    assert env["MTPLX_MLX_CACHE_LIMIT"] == "4gb"


def test_owned_recurrent_state_profile_sets_persistent_owner_mode():
    env, info = _local_profile_env(
        "owned_recurrent_state_mode_persistent_eval_cache_limit_1gb",
        {},
    )

    assert info["profile_type"] == "owned_recurrent_state_probe"
    assert info["owned_recurrent_state"] is True
    assert info["owned_recurrent_state_mode"] == "persistent_eval"
    assert env["MTPLX_OWNED_RECURRENT_STATE"] == "1"
    assert env["MTPLX_OWNED_RECURRENT_STATE_MODE"] == "persistent_eval"
    assert env["MTPLX_MLX_CACHE_LIMIT"] == "1gb"


def test_owned_attention_block_profile_sets_block_owner_mode():
    env, info = _local_profile_env(
        "owned_attn_block_size_1024_mode_contiguous_eval_cache_limit_4gb",
        {},
    )

    assert info["profile_type"] == "owned_attn_block_probe"
    assert info["owned_attn_kv"] == "block"
    assert info["owned_attn_kv_block_size"] == 1024
    assert info["owned_attn_kv_mode"] == "contiguous_eval"
    assert env["MTPLX_OWNED_ATTN_KV"] == "block"
    assert env["MTPLX_OWNED_ATTN_KV_BLOCK_SIZE"] == "1024"
    assert env["MTPLX_OWNED_ATTN_KV_MODE"] == "contiguous_eval"
    assert env["MTPLX_MLX_CACHE_LIMIT"] == "4gb"


def test_live_output_detach_profile_sets_output_leaf_owner_mode():
    env, info = _local_profile_env(
        "detach_live_outputs_mode_contiguous_eval_cache_limit_1gb",
        {},
    )

    assert info["profile_type"] == "live_output_detach_probe"
    assert info["live_output_detach"] is True
    assert info["live_output_detach_mode"] == "contiguous_eval"
    assert env["MTPLX_DETACH_LIVE_OUTPUTS"] == "1"
    assert env["MTPLX_DETACH_LIVE_OUTPUTS_MODE"] == "contiguous_eval"
    assert env["MTPLX_MLX_CACHE_LIMIT"] == "1gb"


def test_metal_copy_leaf_mode_is_accepted_for_detach_profiles():
    env, info = _local_profile_env(
        "detach_gdn_conv_every_4_mode_metal_copy_leaf_cache_limit_1gb",
        {},
    )

    assert info["detach_mode"] == "metal_copy_leaf"
    assert env["MTPLX_DETACH_MODE"] == "metal_copy_leaf"


def test_split_full_attention_profile_sets_query_chunking_env():
    env, info = _local_profile_env(
        "split_full_attn_chunk_1_threshold_1024_cache_limit_4gb",
        {},
    )

    assert info["profile_type"] == "split_full_attn_probe"
    assert info["split_full_attn_chunk_size"] == 1
    assert info["split_full_attn_threshold"] == 1024
    assert env["MTPLX_SPLIT_FULL_ATTN"] == "1"
    assert env["MTPLX_SPLIT_FULL_ATTN_CHUNK_SIZE"] == "1"
    assert env["MTPLX_SPLIT_FULL_ATTN_THRESHOLD"] == "1024"
    assert env["MTPLX_MLX_CACHE_LIMIT"] == "4gb"


def test_sdpa_2pass_profile_sets_exact_attention_kernel_env():
    env, info = _local_profile_env(
        "sdpa_2pass_threshold_1024_max_q_16_cache_limit_1gb",
        {},
    )

    assert info["profile_type"] == "sdpa_2pass_probe"
    assert info["sdpa_2pass"] is True
    assert info["sdpa_2pass_threshold"] == 1024
    assert info["sdpa_2pass_max_q"] == 16
    assert env["MTPLX_SDPA_2PASS"] == "1"
    assert env["MTPLX_SDPA_2PASS_THRESHOLD"] == "1024"
    assert env["MTPLX_SDPA_2PASS_MAX_Q"] == "16"
    assert env["MTPLX_MLX_CACHE_LIMIT"] == "1gb"


def test_blockwise_attention_profile_sets_block_owned_kv_and_hook_env():
    env, info = _local_profile_env(
        "blockwise_attn_block_size_1024_threshold_1024_mode_contiguous_eval_cache_limit_1gb",
        {},
    )

    assert info["profile_type"] == "blockwise_attn_probe"
    assert info["blockwise_attn_threshold"] == 1024
    assert info["owned_attn_kv"] == "block"
    assert info["owned_attn_kv_block_size"] == 1024
    assert info["owned_attn_kv_mode"] == "contiguous_eval"
    assert env["MTPLX_OWNED_ATTN_KV"] == "block"
    assert env["MTPLX_OWNED_ATTN_KV_BLOCK_SIZE"] == "1024"
    assert env["MTPLX_OWNED_ATTN_KV_MODE"] == "contiguous_eval"
    assert env["MTPLX_BLOCKWISE_ATTN"] == "1"
    assert env["MTPLX_BLOCKWISE_ATTN_THRESHOLD"] == "1024"
    assert env["MTPLX_MLX_CACHE_LIMIT"] == "1gb"


def test_vllm_metal_paged_attention_profile_sets_physical_page_env():
    env, info = _local_profile_env(
        "vllm_metal_paged_attn_block_16_blocks_1024_cache_limit_4gb",
        {},
    )

    assert info["profile_type"] == "vllm_metal_paged_attn_probe"
    assert info["vllm_metal_paged_attn"] is True
    assert info["vllm_metal_paged_block_size"] == 16
    assert info["vllm_metal_paged_num_blocks"] == 1024
    assert env["MTPLX_VLLM_METAL_PAGED_ATTN"] == "1"
    assert env["MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE"] == "16"
    assert env["MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS"] == "1024"
    assert env["MTPLX_MLX_CACHE_LIMIT"] == "4gb"


def test_vllm_metal_partitioned_paged_attention_profile_sets_long_context_env():
    env, info = _local_profile_env(
        "vllm_metal_paged_attn_partitioned_block_16_blocks_1024_partition_threshold_4096",
        {},
    )

    assert info["profile_type"] == "vllm_metal_paged_attn_probe"
    assert info["vllm_metal_paged_attn"] is True
    assert info["vllm_metal_paged_partitioned_attn"] is True
    assert info["vllm_metal_paged_partition_threshold"] == 4096
    assert env["MTPLX_VLLM_METAL_PAGED_ATTN"] == "1"
    assert env["MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN"] == "1"
    assert env["MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD"] == "4096"


def test_vllm_metal_paged_attention_profile_sets_sliding_window_discriminator():
    env, info = _local_profile_env(
        "vllm_metal_paged_attn_partitioned_block_16_blocks_1024_partition_threshold_2048_impl_mlx_vector_paged_window_2048",
        {},
    )

    assert info["profile_type"] == "vllm_metal_paged_attn_probe"
    assert info["vllm_metal_paged_attn_impl"] == "mlx_vector_paged"
    assert info["vllm_metal_paged_sliding_window"] == 2048
    assert env["MTPLX_VLLM_METAL_PAGED_ATTN"] == "1"
    assert env["MTPLX_VLLM_METAL_PAGED_SLIDING_WINDOW"] == "2048"


def test_local_online_hidden_forwarding_is_disabled_by_default():
    args = SimpleNamespace(
        online_hidden_corrector_alpha=0.0,
        online_hidden_corrector_decay=0.8,
        online_hidden_corrector_warmup=1,
        online_hidden_corrector_max_feed_depth=None,
        online_hidden_corrector_key="global",
    )

    assert _local_online_hidden_config(args)["alpha"] == 0.0
    assert _local_online_hidden_server_args(args) == []


def test_local_online_hidden_forwarding_emits_server_args():
    args = SimpleNamespace(
        online_hidden_corrector_alpha=0.15,
        online_hidden_corrector_decay=0.7,
        online_hidden_corrector_warmup=2,
        online_hidden_corrector_max_feed_depth=2,
        online_hidden_corrector_key="token",
    )

    assert _local_online_hidden_config(args) == {
        "alpha": 0.15,
        "decay": 0.7,
        "warmup": 2,
        "max_feed_depth": 2,
        "key": "token",
    }
    assert _local_online_hidden_server_args(args) == [
        "--online-hidden-corrector-alpha",
        "0.15",
        "--online-hidden-corrector-decay",
        "0.7",
        "--online-hidden-corrector-warmup",
        "2",
        "--online-hidden-corrector-key",
        "token",
        "--online-hidden-corrector-max-feed-depth",
        "2",
    ]


def test_vllm_metal_paged_attention_profile_can_use_exact_gather_impl():
    env, info = _local_profile_env(
        "vllm_metal_paged_attn_partitioned_block_16_blocks_1024_partition_threshold_2048_impl_fast_sdpa_gather",
        {},
    )

    assert info["profile_type"] == "vllm_metal_paged_attn_probe"
    assert info["vllm_metal_paged_attn_impl"] == "fast_sdpa_gather"
    assert env["MTPLX_VLLM_METAL_PAGED_ATTN"] == "1"
    assert env["MTPLX_VLLM_METAL_PAGED_ATTN_IMPL"] == "fast_sdpa_gather"


def test_vllm_metal_paged_attention_profile_can_use_fp32_paged_impl():
    env, info = _local_profile_env(
        "vllm_metal_paged_attn_partitioned_block_16_blocks_1024_partition_threshold_2048_impl_fp32_paged",
        {},
    )

    assert info["profile_type"] == "vllm_metal_paged_attn_probe"
    assert info["vllm_metal_paged_attn_impl"] == "fp32_paged"
    assert env["MTPLX_VLLM_METAL_PAGED_ATTN"] == "1"
    assert env["MTPLX_VLLM_METAL_PAGED_ATTN_IMPL"] == "fp32_paged"


def test_vllm_metal_paged_attention_profile_can_use_mlx_vector_paged_impl():
    env, info = _local_profile_env(
        "vllm_metal_paged_attn_partitioned_block_16_blocks_1024_partition_threshold_2048_impl_mlx_vector_paged",
        {},
    )

    assert info["profile_type"] == "vllm_metal_paged_attn_probe"
    assert info["vllm_metal_paged_attn_impl"] == "mlx_vector_paged"
    assert env["MTPLX_VLLM_METAL_PAGED_ATTN"] == "1"
    assert env["MTPLX_VLLM_METAL_PAGED_ATTN_IMPL"] == "mlx_vector_paged"


def test_vllm_metal_paged_attention_profile_can_enable_turboquant():
    env, info = _local_profile_env(
        "vllm_metal_paged_attn_partitioned_block_16_blocks_1024_partition_threshold_2048_impl_mlx_vector_paged_turboquant_k_q8_0_v_q3_0",
        {},
    )

    assert info["profile_type"] == "vllm_metal_paged_attn_probe"
    assert info["vllm_metal_paged_turboquant"] is True
    assert info["vllm_metal_paged_turboquant_k_quant"] == "q8_0"
    assert info["vllm_metal_paged_turboquant_v_quant"] == "q3_0"
    assert env["MTPLX_VLLM_METAL_PAGED_ATTN"] == "1"
    assert env["MTPLX_VLLM_METAL_PAGED_TURBOQUANT"] == "1"
    assert env["MTPLX_VLLM_METAL_PAGED_TURBOQUANT_K_QUANT"] == "q8_0"
    assert env["MTPLX_VLLM_METAL_PAGED_TURBOQUANT_V_QUANT"] == "q3_0"


def test_vllm_metal_paged_attention_profile_can_enable_q8_q4_turboquant():
    env, info = _local_profile_env(
        "vllm_metal_paged_attn_partitioned_block_16_blocks_1024_partition_threshold_2048_impl_mlx_vector_paged_turboquant_k_q8_0_v_q4_0",
        {},
    )

    assert info["vllm_metal_paged_turboquant"] is True
    assert info["vllm_metal_paged_turboquant_k_quant"] == "q8_0"
    assert info["vllm_metal_paged_turboquant_v_quant"] == "q4_0"
    assert env["MTPLX_VLLM_METAL_PAGED_TURBOQUANT_K_QUANT"] == "q8_0"
    assert env["MTPLX_VLLM_METAL_PAGED_TURBOQUANT_V_QUANT"] == "q4_0"


def test_vllm_metal_paged_attention_profile_can_exact_gather_last_layers():
    env, info = _local_profile_env(
        "vllm_metal_paged_attn_partitioned_block_16_blocks_1024_partition_threshold_2048_exact_gather_last_4",
        {},
    )

    assert info["profile_type"] == "vllm_metal_paged_attn_probe"
    assert info["vllm_metal_paged_exact_gather_last_n"] == 4
    assert env["MTPLX_VLLM_METAL_PAGED_ATTN"] == "1"
    assert env["MTPLX_VLLM_METAL_PAGED_ATTN_EXACT_GATHER_LAST_N"] == "4"


def test_vllm_metal_paged_attention_profile_can_exact_gather_indices():
    env, info = _local_profile_env(
        "vllm_metal_paged_attn_partitioned_block_16_blocks_1024_partition_threshold_2048_exact_gather_indices_12_15",
        {},
    )

    assert info["profile_type"] == "vllm_metal_paged_attn_probe"
    assert info["vllm_metal_paged_exact_gather_indices"] == [12, 15]
    assert env["MTPLX_VLLM_METAL_PAGED_ATTN"] == "1"
    assert env["MTPLX_VLLM_METAL_PAGED_ATTN_EXACT_GATHER_INDICES"] == "12,15"


def test_vllm_metal_paged_attention_profile_can_require_snapshots():
    env, info = _local_profile_env(
        "vllm_metal_paged_attn_partitioned_block_16_blocks_1024_partition_threshold_2048_snapshot",
        {},
    )

    assert info["profile_type"] == "vllm_metal_paged_attn_probe"
    assert info["snapshot_required"] is True
    assert env["MTPLX_VLLM_METAL_PAGED_ATTN"] == "1"
    assert env["MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN"] == "1"
    assert "MTPLX_SKIP_VERIFY_SNAPSHOT" not in env


def test_vllm_metal_paged_attention_profile_can_stack_trunk_materialize():
    env, info = _local_profile_env(
        "vllm_metal_paged_attn_partitioned_block_16_blocks_1024_partition_threshold_2048_trunk_materialize_512_cache_limit_4gb",
        {},
    )

    assert info["profile_type"] == "vllm_metal_paged_attn_probe"
    assert info["vllm_metal_paged_attn"] is True
    assert info["vllm_metal_paged_partitioned_attn"] is True
    assert info["vllm_metal_paged_partition_threshold"] == 2048
    assert info["trunk_cache_materialize_every"] == 512
    assert env["MTPLX_VLLM_METAL_PAGED_ATTN"] == "1"
    assert env["MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN"] == "1"
    assert env["MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD"] == "2048"
    assert env["MTPLX_TRUNK_CACHE_MATERIALIZE_EVERY"] == "512"
    assert env["MTPLX_MLX_CACHE_LIMIT"] == "4gb"


def test_vllm_metal_paged_attention_profile_can_stack_native_mlp():
    env, info = _local_profile_env(
        "vllm_metal_paged_attn_partitioned_block_16_blocks_1024_partition_threshold_2048_native_mlp_after_1024",
        {},
    )

    assert info["profile_type"] == "vllm_metal_paged_attn_probe"
    assert info["vllm_metal_paged_attn"] is True
    assert info["vllm_metal_paged_partitioned_attn"] is True
    assert info["native_mlp_rowwise"] is True
    assert info["native_mlp_context_threshold"] == 1024
    assert env["MTPLX_VLLM_METAL_PAGED_ATTN"] == "1"
    assert env["MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN"] == "1"
    assert env["MTPLX_NATIVE_MLP_ROWWISE"] == "1"
    assert env["MTPLX_NATIVE_MLP_CONTEXT_THRESHOLD"] == "1024"


def test_vllm_metal_paged_attention_profile_can_stack_mlp_call_variant():
    env, info = _local_profile_env(
        "vllm_metal_paged_attn_partitioned_block_16_blocks_1024_partition_threshold_2048_mlp_variant_compiled_shapeless_after_2048",
        {},
    )

    assert info["profile_type"] == "vllm_metal_paged_attn_probe"
    assert info["vllm_metal_paged_attn"] is True
    assert info["vllm_metal_paged_partitioned_attn"] is True
    assert info["mlp_call_variant"] == "compiled_shapeless"
    assert info["mlp_call_variant_context_threshold"] == 2048
    assert env["MTPLX_MLP_CALL_VARIANT"] == "compiled_shapeless"
    assert env["MTPLX_NATIVE_MLP_CONTEXT_THRESHOLD"] == "2048"


def test_vllm_metal_paged_attention_profile_can_stack_native_gdn_tail():
    env, info = _local_profile_env(
        "vllm_metal_paged_attn_partitioned_block_16_blocks_1024_partition_threshold_2048_impl_mlx_vector_paged_native_gdn_tail_sg4",
        {},
    )

    assert info["profile_type"] == "vllm_metal_paged_attn_probe"
    assert info["vllm_metal_paged_attn"] is True
    assert info["vllm_metal_paged_attn_impl"] == "mlx_vector_paged"
    assert info["native_gdn_tail"] is True
    assert info["native_gdn_tail_simdgroups"] == 4
    assert env["MTPLX_NATIVE_GDN_TAIL"] == "1"
    assert env["MTPLX_NATIVE_GDN_TAIL_SIMDGROUPS"] == "4"


def test_vllm_metal_paged_attention_profile_can_stack_live_output_detach():
    env, info = _local_profile_env(
        "vllm_metal_paged_attn_partitioned_block_16_blocks_1024_partition_threshold_2048_detach_live_outputs_mode_contiguous_eval_cache_limit_4gb",
        {},
    )

    assert info["profile_type"] == "vllm_metal_paged_attn_probe"
    assert info["vllm_metal_paged_attn"] is True
    assert info["vllm_metal_paged_partitioned_attn"] is True
    assert info["vllm_metal_paged_partition_threshold"] == 2048
    assert info["live_output_detach"] is True
    assert info["live_output_detach_mode"] == "contiguous_eval"
    assert env["MTPLX_VLLM_METAL_PAGED_ATTN"] == "1"
    assert env["MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN"] == "1"
    assert env["MTPLX_DETACH_LIVE_OUTPUTS"] == "1"
    assert env["MTPLX_DETACH_LIVE_OUTPUTS_MODE"] == "contiguous_eval"
    assert env["MTPLX_MLX_CACHE_LIMIT"] == "4gb"


def test_vllm_metal_paged_attention_profile_can_page_mtp_layer():
    env, info = _local_profile_env(
        "vllm_metal_paged_attn_partitioned_block_16_blocks_1024_partition_threshold_2048_mtp_paged",
        {},
    )

    assert info["profile_type"] == "vllm_metal_paged_attn_probe"
    assert info["vllm_metal_paged_attn"] is True
    assert info["vllm_metal_paged_mtp_attn"] is True
    assert env["MTPLX_VLLM_METAL_PAGED_ATTN"] == "1"
    assert env["MTPLX_VLLM_METAL_PAGED_MTP_ATTN"] == "1"
    assert env["MTPLX_VLLM_METAL_PAGED_MTP_BLOCK_SIZE"] == "16"
    assert env["MTPLX_VLLM_METAL_PAGED_MTP_NUM_BLOCKS"] == "1024"
