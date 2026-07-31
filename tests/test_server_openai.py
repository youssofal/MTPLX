from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import time
from threading import Lock
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.backends.gemma4_assistant import _gemma4_draft_position
from mtplx.profiles import DEFAULT_HF_MODEL_ID, get_profile
from mtplx.server import openai
from mtplx.server.openai import _RateLimiter, create_app, parse_args


def test_server_parser_default_model_is_public_hf_default():
    args = parse_args(["--warmup-tokens", "0"])

    assert args.model == DEFAULT_HF_MODEL_ID


def test_server_parser_accepts_native_app_launch_id():
    args = parse_args(["--warmup-tokens", "0", "--app-launch-id", "native-123"])

    assert args.app_launch_id == "native-123"


def test_server_parser_resolves_api_key_file_before_env(monkeypatch, tmp_path):
    api_key_file = tmp_path / "api-key"
    api_key_file.write_text("file-secret\n", encoding="utf-8")
    monkeypatch.setenv("MTPLX_API_KEY", "env-secret")
    monkeypatch.setenv("MTPLX_AUTH", "legacy-secret")
    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", raising=False)
    monkeypatch.delenv("MTPLX_PAGED_KV_QUANT", raising=False)

    args = parse_args(
        [
            "--warmup-tokens",
            "0",
            "--api-key-file",
            str(api_key_file),
            "--paged-kv-quantization",
            "q4",
        ]
    )

    assert args.api_key == "file-secret"
    assert args.api_key_source == "file"
    assert args.paged_kv_quantization == "q4"
    assert "MTPLX_VLLM_METAL_PAGED_KV_QUANT" not in os.environ
    assert "MTPLX_PAGED_KV_QUANT" not in os.environ


def test_server_parser_prefers_api_key_env_over_legacy(monkeypatch):
    monkeypatch.setenv("MTPLX_API_KEY", "new-secret")
    monkeypatch.setenv("MTPLX_AUTH", "legacy-secret")

    args = parse_args(["--warmup-tokens", "0"])

    assert args.api_key == "new-secret"
    assert args.api_key_source == "env:MTPLX_API_KEY"


def test_server_parser_accepts_step_adapter_quant_flags():
    args = parse_args(
        [
            "--model",
            "models/Step-3.7-Flash-MTPLX-step3p5",
            "--warmup-tokens",
            "0",
            "--verify-strategy",
            "trim_commit",
            "--verify-core",
            "stock",
            "--mtp-adapter",
            "outputs/adapters/c4-mtp-adapter-20260603-134243-r4.npz",
            "--mtp-quant-bits",
            "4",
            "--mtp-quant-group-size",
            "64",
            "--mtp-quant-mode",
            "affine",
            "--reasoning-parser",
            "step3p5",
            "--reasoning-effort",
            "medium",
        ]
    )

    assert args.verify_strategy == "trim_commit"
    assert args.verify_core == "stock"
    assert args.mtp_adapter == "outputs/adapters/c4-mtp-adapter-20260603-134243-r4.npz"
    assert args.mtp_quant_bits == 4
    assert args.mtp_quant_group_size == 64
    assert args.mtp_quant_mode == "affine"
    assert args.reasoning_parser == "step3p5"
    assert args.reasoning_effort == "medium"


def test_server_backend_defaults_apply_laguna_response_cap_and_codec():
    args = parse_args(
        [
            "--backend-id",
            "laguna_ar",
            "--no-load-mtp",
            "--generation-mode",
            "ar",
            "--warmup-tokens",
            "0",
        ]
    )
    args.max_response_tokens = None
    args.reasoning_parser = "qwen3"

    openai._apply_backend_server_defaults(args, explicit_flags=set())

    assert args.max_response_tokens == 32_768
    assert args.reasoning_parser == "poolside_v1"
    assert args.tool_prompt_mode == "native"
    assert args.chat_template_profile == "tokenizer"

    with pytest.raises(ValueError, match="requires.*tokenizer chat template"):
        parse_args(
            [
                "--backend-id",
                "laguna_ar",
                "--chat-template-profile",
                "local_qwen36",
                "--warmup-tokens",
                "0",
            ]
        )

    with pytest.raises(ValueError, match="requires.*tokenizer chat template"):
        parse_args(
            [
                "--backend-id",
                "laguna_ar",
                "--chat-template-path",
                "/tmp/qwen.jinja",
                "--warmup-tokens",
                "0",
            ]
        )

    inferred = parse_args(
        [
            "--model",
            "models/Step-3.7-Flash-MTPLX-step3p5",
            "--backend-id",
            "step3p5_mtp",
            "--warmup-tokens",
            "0",
        ]
    )

    assert inferred.reasoning_parser == "step3p5"
    assert inferred.reasoning_effort == "low"


def test_snapshot_required_verifiers_disable_skip_snapshot_override():
    args = SimpleNamespace(
        generation_mode="mtp",
        verify_strategy="trim_commit",
    )

    overrides = openai._server_runtime_env_overrides(
        args,
        {"MTPLX_SKIP_VERIFY_SNAPSHOT": "1"},
    )

    assert overrides["MTPLX_SKIP_VERIFY_SNAPSHOT"] == "0"


def test_capture_commit_keeps_fast_snapshot_skip_override():
    args = SimpleNamespace(
        generation_mode="mtp",
        verify_strategy="capture_commit",
    )

    overrides = openai._server_runtime_env_overrides(
        args,
        {"MTPLX_SKIP_VERIFY_SNAPSHOT": "1"},
    )

    assert overrides["MTPLX_SKIP_VERIFY_SNAPSHOT"] == "1"


def test_server_parser_accepts_tool_prompt_and_template_profile():
    args = parse_args(
        [
            "--warmup-tokens",
            "0",
            "--tool-prompt-mode",
            "native",
            "--chat-template-profile",
            "froggeric_v19",
        ]
    )

    assert args.tool_prompt_mode == "native"
    assert args.chat_template_profile == "froggeric_v19"


def _write_gemma4_pair_bundle(path: Path) -> Path:
    target = path / "target"
    assistant = path / "assistant"
    target.mkdir()
    assistant.mkdir()
    (path / "mtplx_pair.json").write_text(
        json.dumps(
            {
                "layout": {"target": "target", "assistant": "assistant"},
                "benchmark": {
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 64,
                    "best_block_size": 6,
                },
            }
        ),
        encoding="utf-8",
    )
    (target / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4",
                "architectures": ["Gemma4ForConditionalGeneration"],
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 5376,
                    "num_hidden_layers": 60,
                    "hidden_size_per_layer_input": 0,
                    "enable_moe_block": False,
                    "vocab_size": 262144,
                },
            }
        ),
        encoding="utf-8",
    )
    (target / "generation_config.json").write_text(
        json.dumps({"temperature": 1.0, "top_p": 0.95, "top_k": 64}),
        encoding="utf-8",
    )
    (assistant / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4_assistant",
                "architectures": ["Gemma4AssistantForCausalLM"],
                "backbone_hidden_size": 5376,
                "use_ordered_embeddings": False,
                "text_config": {
                    "hidden_size": 1024,
                    "num_hidden_layers": 4,
                    "num_kv_shared_layers": 4,
                    "layer_types": [
                        "sliding_attention",
                        "sliding_attention",
                        "sliding_attention",
                        "full_attention",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_server_parser_applies_gemma4_pair_defaults(tmp_path):
    bundle = _write_gemma4_pair_bundle(tmp_path)

    args = parse_args(["--model", str(bundle), "--warmup-tokens", "0"])

    assert args.backend_id == "gemma4_assistant"
    assert args.model_id == "mtplx-gemma4-31b-assistant-mtp"
    assert args.temperature == 1.0
    assert args.top_p == 0.95
    assert args.top_k == 64
    assert args.draft_top_p == 0.95
    assert args.draft_top_k == 64
    # kvcache-v2 2026-07-03: Gemma4 default draft depth capped at 2 —
    # long-form acceptance collapses by depth, making 5-6 EV-negative
    # (+26%/+8% decode in A/Bs; MTPLX_GEMMA4_DEFAULT_DEPTH overrides).
    assert args.depth == 2
    assert args.draft_block_size == 2
    assert args.reasoning_parser == "gemma4"
    assert args.chat_template_profile == "tokenizer"
    assert args.reasoning == "auto"
    assert args.enable_thinking is True


def test_gemma4_request_uses_draft_block_before_depth(tmp_path):
    bundle = _write_gemma4_pair_bundle(tmp_path)
    args = parse_args(["--model", str(bundle), "--warmup-tokens", "0"])
    state = SimpleNamespace(
        args=args,
        backend_descriptor=openai.descriptor_for_backend_id("gemma4_assistant"),
    )
    request = openai.ChatCompletionRequest(draft_block_size=7, depth=2)

    depth = openai._request_depth_for_generation(
        state,
        request,
        generation_mode="mtp",
    )

    assert depth == 7


def test_gemma4_assistant_draft_position_tracks_primary_token_source():
    assert _gemma4_draft_position(0) == 0
    assert _gemma4_draft_position(1) == 0
    assert _gemma4_draft_position(42) == 41


def test_runtime_mode_label_distinguishes_sustained_max_and_burst():
    assert (
        openai._health_runtime_mode_label("sustained", "mtp", fan_boost_active=False)
        == "Sustained MTP"
    )
    assert (
        openai._health_runtime_mode_label("sustained", "mtp", fan_boost_active=True)
        == "Sustained Max MTP"
    )
    assert (
        openai._health_runtime_mode_label(
            "performance-cold", "mtp", fan_boost_active=True
        )
        == "Burst MTP"
    )
    assert (
        openai._health_runtime_mode_label("sustained", "ar", fan_boost_active=False)
        == "Sustained AR"
    )


def test_startup_urls_distinguish_wildcard_bind_from_local_url():
    args = SimpleNamespace(host="0.0.0.0", port=8000)

    assert openai._startup_bind_label(args) == "0.0.0.0:8000 (all interfaces)"
    assert openai._startup_server_url(args) == "http://127.0.0.1:8000"
    assert openai._startup_openai_base_url(args) == "http://127.0.0.1:8000/v1"


def test_laguna_fused_startup_line_carries_the_engagement_receipt():
    runtime = SimpleNamespace(
        laguna_fused_report=[{"path": "fused_gate_up", "layers_converted": 47}]
    )

    line = openai._laguna_fused_startup_line(runtime)

    assert line is not None
    assert "[laguna-fused]" in line
    assert json.loads(line.split("[laguna-fused] ", 1)[1]) == runtime.laguna_fused_report


def test_laguna_fused_startup_line_stays_silent_without_a_report():
    assert openai._laguna_fused_startup_line(SimpleNamespace()) is None
    assert (
        openai._laguna_fused_startup_line(SimpleNamespace(laguna_fused_report=[]))
        is None
    )


def test_chat_request_accepts_ai_sdk_camel_sampler_aliases():
    request = openai.ChatCompletionRequest.model_validate(
        {
            "model": "mtplx-test",
            "messages": [{"role": "user", "content": "hi"}],
            "topP": 0.9,
            "topK": 20,
        }
    )

    assert request.top_p == 0.9
    assert request.top_k == 20


def test_dynamic_paged_kv_reservation_caps_oversized_response_budget(monkeypatch):
    monkeypatch.delenv("MTPLX_DYNAMIC_PAGED_KV_MAX_INITIAL_NEW_TOKENS", raising=False)

    reservation = openai._dynamic_paged_kv_reservation(
        prompt_tokens=181,
        max_new_tokens=65536,
        mtp_depth=3,
    )

    assert reservation["env"]["MTPLX_DYNAMIC_PAGED_KV_TOKENS"] == str(181 + 16384 + 3)
    assert reservation["requested_new_tokens"] == 65536
    assert reservation["reserved_new_tokens"] == 16384
    assert reservation["initial_new_token_cap"] == 16384
    assert reservation["reservation_capped"] is True


def test_dynamic_paged_kv_reservation_can_disable_initial_cap(monkeypatch):
    monkeypatch.setenv("MTPLX_DYNAMIC_PAGED_KV_MAX_INITIAL_NEW_TOKENS", "off")

    reservation = openai._dynamic_paged_kv_reservation(
        prompt_tokens=10,
        max_new_tokens=65536,
        mtp_depth=3,
    )

    assert reservation["env"]["MTPLX_DYNAMIC_PAGED_KV_TOKENS"] == str(10 + 65536 + 3)
    assert reservation["reserved_new_tokens"] == 65536
    assert reservation["initial_new_token_cap"] is None
    assert reservation["reservation_capped"] is False


def test_metrics_envelope_includes_acceptance_denominators():
    envelope = openai._metrics_envelope(
        stats={
            "prompt_eval_time_s": 2.0,
            "decode_elapsed_s": 1.0,
            "verify_calls": 10,
            "verify_forward_time_s": 1.25,
            "verify_eval_time_s": 0.75,
            "verify_logits_eval_time_s": 0.25,
            "verify_hidden_eval_time_s": 0.35,
            "verify_target_distribution_time_s": 0.15,
            "target_distribution_materialized_rows": 20,
            "target_distribution_materialized_windows": 10,
            "target_distribution_share": 0.075,
            "lazy_bonus_verify_calls": 8,
            "lazy_bonus_commit_time_s": 0.4,
            "accepted_by_depth": [9, 7, 4],
            "drafted_by_depth": [10, 10, 8],
            "mean_accept_probability_by_depth": [0.91, 0.73, 0.52],
            "mtp_history_policy": "last_window",
            "mtp_history_window_tokens": 8192,
            "mtp_history_position_base": 11835,
            "clear_cache_every": 256,
            "clear_cache_events": 3,
            "clear_cache_time_s": 0.42,
            "trunk_cache_materialize_every": 512,
            "trunk_cache_materialize_events": 1,
            "trunk_cache_materialize_time_s": 0.08,
            "state_rebase_every": 2048,
            "state_rebase_events": 2,
            "state_rebase_time_s": 0.11,
            "live_output_detach_enabled": True,
            "live_output_detach_mode": "contiguous_eval",
            "live_output_detach_events": 4,
            "live_output_detach_time_s": 0.07,
        },
        prompt_tokens=100,
        completion_tokens=12,
        request_elapsed_s=3.0,
        token_times=[10.0, 10.5, 11.0],
        request_started_s=9.5,
        lock_wait_time_s=0.0,
        session_id="s1",
        session_cache_hit=False,
        cache_miss_reason="new_session",
        session_restore_mode="cold",
        mtp_depth=3,
        generation_limits={},
    )

    assert envelope["verify_calls"] == 10
    assert envelope["accepted_by_depth"] == [9, 7, 4]
    assert envelope["prefill_tok_s"] == 50.0
    assert envelope["prefill_compute_tok_s"] == 50.0
    assert envelope["verify_forward_time_s"] == 1.25
    assert envelope["verify_eval_time_s"] == 0.75
    assert envelope["verify_logits_eval_time_s"] == 0.25
    assert envelope["verify_hidden_eval_time_s"] == 0.35
    assert envelope["verify_target_distribution_time_s"] == 0.15
    assert envelope["target_distribution_materialized_rows"] == 20
    assert envelope["target_distribution_materialized_windows"] == 10
    assert envelope["target_distribution_share"] == 0.075
    assert envelope["lazy_bonus_verify_calls"] == 8
    assert envelope["lazy_bonus_commit_time_s"] == 0.4
    assert envelope["drafted_by_depth"] == [10, 10, 8]
    assert envelope["mean_accept_probability_by_depth"] == [0.91, 0.73, 0.52]
    assert envelope["mtp_history_policy"] == "last_window"
    assert envelope["mtp_history_window_tokens"] == 8192
    assert envelope["mtp_history_position_base"] == 11835
    assert envelope["clear_cache_every"] == 256
    assert envelope["clear_cache_events"] == 3
    assert envelope["clear_cache_time_s"] == 0.42
    assert envelope["trunk_cache_materialize_every"] == 512
    assert envelope["trunk_cache_materialize_events"] == 1
    assert envelope["trunk_cache_materialize_time_s"] == 0.08
    assert envelope["state_rebase_every"] == 2048
    assert envelope["state_rebase_events"] == 2
    assert envelope["state_rebase_time_s"] == 0.11
    assert envelope["live_output_detach_enabled"] is True
    assert envelope["live_output_detach_mode"] == "contiguous_eval"
    assert envelope["live_output_detach_events"] == 4
    assert envelope["live_output_detach_time_s"] == 0.07


def test_metrics_envelope_prefill_tps_excludes_cached_tokens():
    envelope = openai._metrics_envelope(
        stats={
            "prompt_eval_time_s": 4.0,
            "cached_tokens": 8000,
            "new_prefill_tokens": 2000,
        },
        prompt_tokens=10000,
        completion_tokens=64,
        request_elapsed_s=6.0,
        token_times=[12.0, 12.1],
        request_started_s=10.0,
        lock_wait_time_s=0.0,
        session_id="warm-session",
        session_cache_hit=True,
        cache_miss_reason=None,
        session_restore_mode="longest_prefix",
        mtp_depth=3,
        generation_limits={},
    )

    assert envelope["cached_tokens"] == 8000
    assert envelope["new_prefill_tokens"] == 2000
    assert envelope["prefill_tok_s"] == 500.0
    assert envelope["prefill_compute_tok_s"] == 500.0


def test_metrics_envelope_excludes_ssd_restore_from_decode_timing():
    envelope = openai._metrics_envelope(
        stats={
            "elapsed_s": 8.0,
            "generated_tokens": 64,
            "prompt_eval_time_s": 0.0,
            "cache_restore_time_s": 0.25,
            "cached_tokens": 100410,
            "new_prefill_tokens": 0,
            "cache_source": "ssd",
            "ssd_cache_hit": True,
            "ssd_restore_s": 3.0,
        },
        prompt_tokens=100410,
        completion_tokens=64,
        request_elapsed_s=8.2,
        token_times=[13.0, 13.1, 13.2],
        request_started_s=9.5,
        lock_wait_time_s=0.0,
        session_id="warm-session",
        session_cache_hit=True,
        cache_miss_reason=None,
        session_restore_mode="ssd_clone",
        mtp_depth=3,
        generation_limits={},
    )

    assert envelope["cache_restore_time_s"] == 3.0
    assert envelope["decode_elapsed_s"] == 5.0
    assert envelope["decode_tok_s"] == 12.8
    assert envelope["display_decode_tok_s"] == 12.8


def test_ar_batch_keeps_tool_history_turns_in_fair_lane():
    assert (
        openai._ar_batch_history_bypass_reason(
            {
                "request_message_count": 4,
                "request_message_roles": ["system", "user", "assistant", "tool"],
                "request_tool_count": 2,
                "request_client_hint": "opencode",
            }
        )
        is None
    )
    assert (
        openai._ar_batch_history_bypass_reason(
            {
                "request_message_count": 3,
                "request_message_roles": ["system", "user", "user"],
                "request_tool_count": 2,
                "request_client_hint": "opencode",
            }
        )
        is None
    )
    assert (
        openai._ar_batch_history_bypass_reason(
            {
                "request_message_count": 2,
                "request_message_roles": ["system", "user"],
                "request_tool_count": 2,
                "request_client_hint": "opencode",
            }
        )
        is None
    )


def test_ar_batch_admits_generic_openai_clients():
    """Generic API clients ride the concurrency-adaptive lane.

    A lone request still keeps solo MTP (the in-flight check in
    ``_ar_batch_mtp_fallback_reason`` only diverts under real concurrency),
    but anonymous clients are no longer force-serialized behind the queue.
    """

    assert (
        openai._ar_batch_history_bypass_reason(
            {
                "request_message_count": 2,
                "request_message_roles": ["system", "user"],
                "request_tool_count": 0,
                "request_client_label": "openai",
            }
        )
        is None
    )


def test_ar_batch_strips_nonmergeable_history_caches():
    service = object.__new__(openai._BatchedARGenerationService)
    cached = SimpleNamespace(
        prompt_ids=[1, 2, 3],
        insert_cache=[object()],
        insert_all_tokens=[1, 2],
        insert_prompt_ids=[3],
        cached_tokens=2,
        session_cache_hit=True,
        cache_source="ssd",
        ssd_cache_hit=True,
        ssd_cached_tokens=2,
        ssd_restore_s=0.1,
        ssd_suffix_tokens=1,
        effective_restore_mode="ssd_clone",
        cache_miss_reason=None,
        request_observability={},
    )
    fresh = SimpleNamespace(
        prompt_ids=[4, 5],
        insert_cache=None,
        request_observability={},
    )

    selected, requeued = service._split_unmergeable_history_batch([fresh, cached])

    assert selected == [fresh, cached]
    assert requeued == []
    assert cached.insert_cache is None
    assert cached.insert_prompt_ids == [1, 2, 3]
    assert cached.cached_tokens == 0
    assert cached.session_cache_hit is False
    assert cached.cache_source == "none"
    assert cached.ssd_cache_hit is False
    assert cached.cache_miss_reason == "ar_batch_nonmergeable_history_cache"
    assert cached.request_observability["ar_batch_bypass_reason"] == (
        "nonmergeable_history_cache"
    )
    assert cached.request_observability["ar_batch_cache_restore_skipped"] == (
        "nonmergeable_history_cache"
    )
    assert "ar_batch_history_cache_requeued" not in fresh.request_observability


def test_ar_batch_detects_nonmergeable_history_cache_entries():
    class Mergeable:
        def merge(self, _entries):
            return self

    assert openai._BatchedARGenerationService._cache_supports_batch_history_merge(
        [Mergeable()]
    )
    assert not openai._BatchedARGenerationService._cache_supports_batch_history_merge(
        [object()]
    )


def test_long_prompt_commits_prompt_prefix_when_ssd_cache_is_enabled():
    state = SimpleNamespace(
        session_bank_cold_tier=SimpleNamespace(enabled=True, min_prefix_tokens=512)
    )

    assert openai._commit_prompt_prefix_for_request(
        state,
        prompt_ids=list(range(512)),
        tools_active=False,
    )
    assert not openai._commit_prompt_prefix_for_request(
        state,
        prompt_ids=list(range(511)),
        tools_active=False,
    )
    assert openai._commit_prompt_prefix_for_request(
        SimpleNamespace(session_bank_cold_tier=None),
        prompt_ids=[1],
        tools_active=True,
    )


def test_implicit_anonymous_sessions_do_not_keep_live_refs(monkeypatch):
    monkeypatch.delenv(
        "MTPLX_SESSIONBANK_LIVE_REFS_FOR_IMPLICIT_SESSIONS",
        raising=False,
    )

    assert (
        openai._session_keep_live_refs_for_request(
            session_source="implicit_hash", session_id="anon-bench"
        )
        is False
    )


def test_anonymous_coding_agent_tool_sessions_keep_live_refs(monkeypatch):
    monkeypatch.delenv(
        "MTPLX_SESSIONBANK_LIVE_REFS_FOR_IMPLICIT_SESSIONS",
        raising=False,
    )

    assert (
        openai._session_keep_live_refs_for_request(
            session_source="new",
            session_id="anon-opencode",
            tool_names=["bash", "read", "write"],
        )
        is True
    )


def test_anonymous_non_tool_benchmark_sessions_stay_cold(monkeypatch):
    monkeypatch.delenv(
        "MTPLX_SESSIONBANK_LIVE_REFS_FOR_IMPLICIT_SESSIONS",
        raising=False,
    )

    assert (
        openai._session_keep_live_refs_for_request(
            session_source="new",
            session_id="anon-aime",
            tool_names=[],
        )
        is False
    )


def test_explicit_sessions_keep_live_refs(monkeypatch):
    monkeypatch.delenv(
        "MTPLX_SESSIONBANK_LIVE_REFS_FOR_IMPLICIT_SESSIONS",
        raising=False,
    )

    assert (
        openai._session_keep_live_refs_for_request(
            session_source="metadata.chat_id", session_id="chat-123"
        )
        is True
    )


def test_implicit_session_live_refs_have_diagnostic_override(monkeypatch):
    monkeypatch.setenv("MTPLX_SESSIONBANK_LIVE_REFS_FOR_IMPLICIT_SESSIONS", "1")

    assert (
        openai._session_keep_live_refs_for_request(
            session_source="implicit_hash", session_id="anon-bench"
        )
        is True
    )


def test_opencode_tool_history_live_frontier_is_app_opt_in(monkeypatch):
    monkeypatch.delenv("MTPLX_OPENCODE_TOOL_HISTORY_LIVE_FRONTIER", raising=False)
    assert openai._opencode_tool_history_live_frontier_enabled() is False

    monkeypatch.setenv("MTPLX_OPENCODE_TOOL_HISTORY_LIVE_FRONTIER", "1")
    assert openai._opencode_tool_history_live_frontier_enabled() is True

    monkeypatch.setenv("MTPLX_OPENCODE_TOOL_HISTORY_LIVE_FRONTIER", "0")
    assert openai._opencode_tool_history_live_frontier_enabled() is False


def test_live_frontier_miss_reason_reports_unknown_tool_id():
    messages = [
        openai.ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call_known",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        ),
        openai.ChatMessage(
            role="tool",
            tool_call_id="call_other",
            content="result",
        ),
    ]

    assert (
        openai._live_frontier_miss_reason_for_request(
            messages=messages,
            cache_miss_reason="prefix_divergence_at_token",
            session_source="metadata.chat_id",
            session_keep_live_ref=True,
        )
        == "miss_unknown_tool_id"
    )


def test_live_frontier_miss_reason_maps_agent_cache_causes():
    messages = [
        openai.ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call_read",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        ),
        openai.ChatMessage(
            role="tool",
            tool_call_id="call_read",
            content="result",
        ),
    ]

    assert (
        openai._live_frontier_miss_reason_for_request(
            messages=messages,
            cache_miss_reason="template_mismatch",
            session_source="metadata.chat_id",
            session_keep_live_ref=True,
        )
        == "miss_template_changed"
    )
    assert (
        openai._live_frontier_miss_reason_for_request(
            messages=messages,
            cache_miss_reason="no_snapshot_coverage",
            session_source="metadata.chat_id",
            session_keep_live_ref=True,
        )
        == "miss_live_frontier_consumed_or_missing"
    )
    assert (
        openai._live_frontier_miss_reason_for_request(
            messages=messages,
            cache_miss_reason=None,
            session_source="metadata.chat_id",
            session_keep_live_ref=False,
        )
        == "miss_live_frontier_not_armed"
    )


def test_live_frontier_envelope_fields_reports_miss_reason_on_result_turn():
    # Regression for #100/#99: 1.0.3 raised a TypeError while assembling
    # exactly this envelope on every agent tool-result turn whose live
    # frontier missed, killing Pi/Hermes/OpenCode sessions.
    observability = {
        "live_frontier_result_turn": True,
        "live_frontier_assistant_tool_call_count": 2,
        "live_frontier_tool_result_count": 1,
        "live_frontier_unknown_tool_result_count": 0,
        "request_session_source": "metadata.chat_id",
    }

    fields = openai._live_frontier_envelope_fields(
        request_observability=observability,
        session_cache_hit=False,
        session_restore_mode=None,
        cache_miss_reason="new_session",
        session_keep_live_ref=True,
    )

    assert fields == {
        "live_frontier_hit": False,
        "live_frontier_restore_mode": None,
        "live_frontier_miss_reason": "miss_wrong_session_or_no_prior_frontier",
    }


def test_live_frontier_envelope_fields_hit_carries_no_miss_reason():
    fields = openai._live_frontier_envelope_fields(
        request_observability={
            "live_frontier_result_turn": True,
            "live_frontier_assistant_tool_call_count": 1,
            "live_frontier_tool_result_count": 1,
            "live_frontier_unknown_tool_result_count": 0,
        },
        session_cache_hit=True,
        session_restore_mode="reference_lease",
        cache_miss_reason=None,
        session_keep_live_ref=True,
    )

    assert fields == {
        "live_frontier_hit": True,
        "live_frontier_restore_mode": "reference_lease",
        "live_frontier_miss_reason": None,
    }


def test_live_frontier_envelope_fields_empty_for_non_result_turns():
    assert (
        openai._live_frontier_envelope_fields(
            request_observability={},
            session_cache_hit=False,
            session_restore_mode=None,
            cache_miss_reason=None,
            session_keep_live_ref=False,
        )
        == {}
    )


def test_vision_splice_kwargs_always_match_callee_signatures():
    # Guard for the bug class behind #100: a ``vision_splice=`` kwarg
    # mechanically threaded into a call whose target never declared it.
    # Every call passing vision_splice must target a function that
    # declares the parameter (or **kwargs), resolved by real signatures.
    import ast

    import mtplx.generation
    import mtplx.runtime
    import mtplx.vision.splice

    sources = [
        Path(openai.__file__),
        Path(mtplx.generation.__file__),
        Path(mtplx.runtime.__file__),
        Path(mtplx.vision.splice.__file__),
    ]

    defs: dict[str, tuple[bool, bool]] = {}
    calls: list[tuple[str, int, ast.expr]] = []
    for source in sources:
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = [
                    arg.arg
                    for arg in (
                        node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                    )
                ]
                defs[node.name] = (
                    "vision_splice" in params,
                    node.args.kwarg is not None,
                )
            elif isinstance(node, ast.Call) and any(
                keyword.arg == "vision_splice" for keyword in node.keywords
            ):
                calls.append((source.name, node.lineno, node.func))

    assert len(calls) >= 10, "vision_splice call sites disappeared; audit is stale"

    problems = []
    for filename, lineno, func in calls:
        if not isinstance(func, ast.Name):
            problems.append(f"{filename}:{lineno} passes vision_splice to a non-plain callee")
            continue
        declared, has_kwargs = defs.get(func.id, (False, False))
        if not (declared or has_kwargs):
            problems.append(
                f"{filename}:{lineno} passes vision_splice to {func.id}() which does not accept it"
            )

    assert not problems, "\n".join(problems)


class FakeExecutor:
    def submit(self, fn, *args, **kwargs):
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # pragma: no cover - surfaced by caller
            future.set_exception(exc)
        return future

    def shutdown(self, **_kwargs):
        return None


class StreamingTokenizer:
    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, **_kwargs
    ):
        assert tokenize is True
        text = "\n".join(
            f"{message['role']}:{message.get('content') or ''}" for message in messages
        )
        if add_generation_prompt:
            text = f"{text}\nassistant:" if text else "assistant:"
        return [ord(char) for char in text]

    def encode(self, text):
        return [ord(char) for char in str(text)]

    def decode(self, tokens, **_kwargs):
        return "".join(chr(int(token)) for token in tokens)


class RecordingBank:
    def __init__(self):
        self.puts: list[dict] = []

    def put(self, **kwargs):
        self.puts.append(kwargs)
        return SimpleNamespace(
            prefix_len=len(kwargs["token_ids"]),
            nbytes=123,
            token_hash="test-token-hash",
        )


class ForegroundState:
    def __init__(self) -> None:
        self.lock = Lock()
        self.foreground_active = 0
        self.last_request_started_at = 0.0

    def begin_foreground(self) -> None:
        self.foreground_active += 1

    def end_foreground(self) -> None:
        self.foreground_active = max(0, self.foreground_active - 1)

    def has_foreground(self) -> bool:
        return self.foreground_active > 0

    def foreground_count(self) -> int:
        return self.foreground_active


def _fake_state(*, api_key: str | None = None, rate_limit: int = 0):
    from mtplx.server.dashboard_state import DashboardState

    argv = ["--warmup-tokens", "0", "--rate-limit", str(rate_limit)]
    if api_key:
        argv.extend(["--api-key", api_key])
    args = parse_args(argv)
    return SimpleNamespace(
        args=args,
        model_id="mtplx-test-model",
        lock=Lock(),
        runtime=SimpleNamespace(
            model_path=Path("models/example"),
            mtp_enabled=True,
            tokenizer=SimpleNamespace(
                decode=lambda tokens, **_kwargs: "".join(
                    chr(int(token)) for token in tokens
                ),
            ),
        ),
        profile=get_profile(args.profile),
        context_window=4096,
        load_time_s=0.25,
        draft_lm_head={"installed": False, "reason": "test"},
        draft_head_identity="test-head",
        template_hash="test-template",
        main_system_prompt_hash=None,
        fast_path_env_status={},
        profile_env_status={},
        mlx_cache_limit_status={"configured": False},
        metal_memory_caps={"applied": False, "reason": "test"},
        mlx_runtime_status={"ok": True, "stock_pypi_layout": True},
        warmup_status={"enabled": False, "ran": False, "tokens": 0},
        last_metrics=[{"tok_s": 12.5, "accept_rate": 0.75}],
        rate_limiter=_RateLimiter(rate_limit),
        sessions=SimpleNamespace(
            list_sessions=lambda: {"sessions": [], "count": 0, "session_bank": {}},
            clear_session=lambda session_id: {"cleared": session_id},
            clear_all=lambda: {"cleared": True},
        ),
        generation_executor=FakeExecutor(),
        # Dashboard primitives mirror what ServerState.__init__ allocates.
        dashboard=DashboardState(),
    )


def _fake_streaming_session_state():
    state = _fake_state()
    foreground = ForegroundState()
    state.lock = foreground.lock
    state.begin_foreground = foreground.begin_foreground
    state.end_foreground = foreground.end_foreground
    state.has_foreground = foreground.has_foreground
    state.foreground_count = foreground.foreground_count
    state.runtime.tokenizer = StreamingTokenizer()
    state.sessions = openai.EngineSessionManager(bank=RecordingBank())
    state.generation_executor = ThreadPoolExecutor(max_workers=1)
    state.postcommit_executor = FakeExecutor()
    state.args.stats_footer = False
    return state


def _fake_final_state(tokens):
    return SimpleNamespace(
        final_trunk_cache=["cache"],
        final_logits="logits",
        final_hidden="hidden",
        final_committed_mtp_cache=None,
        generated_token_ids=tuple(tokens),
        safe_to_commit=True,
        finish_reason="stop",
    )


def test_mtplx_settings_endpoint_controls_server_reasoning():
    state = _fake_state(api_key="mtplx-local")
    client = TestClient(create_app(state))

    initial = client.get(
        "/v1/mtplx/settings", headers={"Authorization": "Bearer mtplx-local"}
    )
    assert initial.status_code == 200
    assert initial.json()["reasoning"] == "auto"
    assert initial.json()["metal_memory_caps"] == {"applied": False, "reason": "test"}

    off = client.post(
        "/v1/mtplx/settings",
        json={"reasoning": "off"},
        headers={"Authorization": "Bearer mtplx-local"},
    )
    assert off.status_code == 200
    assert off.json()["reasoning"] == "off"
    assert off.json()["enable_thinking"] is False
    assert state.args.enable_thinking is False

    on = client.post(
        "/v1/mtplx/settings",
        json={"reasoning": "on"},
        headers={"Authorization": "Bearer mtplx-local"},
    )
    assert on.status_code == 200
    assert on.json()["reasoning"] == "on"
    assert on.json()["enable_thinking"] is True
    assert state.args.enable_thinking is True

    effort = client.post(
        "/v1/mtplx/settings",
        json={"reasoning_effort": "high"},
        headers={"Authorization": "Bearer mtplx-local"},
    )
    assert effort.status_code == 200
    assert effort.json()["reasoning_effort"] == "high"
    assert state.args.reasoning_effort == "high"


def test_mtplx_settings_endpoint_ignores_read_only_descriptor_echoes():
    state = _fake_state(api_key="mtplx-local")
    client = TestClient(create_app(state))

    response = client.post(
        "/v1/mtplx/settings",
        json={
            "reasoning": "off",
            "enable_thinking": False,
            "reasoning_effort": "low",
            "model_controls": {"model_family": "qwen3_6"},
            "draft_control": {"maximum": 3},
            "reasoning_policy": {"parser": "qwen3"},
            "kv_quant_policy": {"supported": False},
            "context_window_policy": {"maximum": 262144},
            "sampling_defaults": {"temperature": 0.6},
        },
        headers={"Authorization": "Bearer mtplx-local"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["reasoning"] == "off"
    assert payload["enable_thinking"] is False
    assert payload["reasoning_effort"] == "low"
    assert payload["applied"] == {
        "reasoning": "off",
        "enable_thinking": False,
        "reasoning_effort": "low",
    }


def test_mtplx_settings_endpoint_updates_draft_sampler_without_depth_change():
    state = _fake_state()
    client = TestClient(create_app(state))

    response = client.post(
        "/v1/mtplx/settings",
        json={"draft_temperature": 0.62, "draft_top_p": 0.9, "draft_top_k": 12},
    )

    assert response.status_code == 200
    assert response.json()["depth"] == 3
    assert response.json()["draft_temperature"] == 0.62
    assert response.json()["draft_top_p"] == 0.9
    assert response.json()["draft_top_k"] == 12
    assert state.args.depth == 3
    assert state.draft_sampler == openai.SamplerConfig(
        temperature=0.62,
        top_p=0.9,
        top_k=12,
    )


def test_mtplx_settings_endpoint_mirrors_visible_sampler_into_draft_sampler():
    state = _fake_state()
    state.args.draft_temperature = 0.6
    state.args.draft_top_p = 1.0
    state.args.draft_top_k = 20
    state.draft_sampler = openai.SamplerConfig(
        temperature=0.6,
        top_p=1.0,
        top_k=20,
    )
    client = TestClient(create_app(state))

    response = client.post(
        "/v1/mtplx/settings",
        json={"temperature": 0.72, "top_p": 0.95, "top_k": 35},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["temperature"] == 0.72
    assert body["top_p"] == 0.95
    assert body["top_k"] == 35
    assert body["draft_temperature"] == 0.72
    assert body["draft_top_p"] == 0.95
    assert body["draft_top_k"] == 35
    assert body["applied"] == {"temperature": 0.72, "top_p": 0.95, "top_k": 35}
    assert state.args.draft_temperature == 0.72
    assert state.args.draft_top_p == 0.95
    assert state.args.draft_top_k == 35
    assert state.draft_sampler == openai.SamplerConfig(
        temperature=0.72,
        top_p=0.95,
        top_k=35,
    )


def test_mtplx_settings_endpoint_respects_explicit_draft_sampler_override():
    state = _fake_state()
    state.args.draft_temperature = 0.6
    state.args.draft_top_p = 1.0
    state.args.draft_top_k = 20
    state.draft_sampler = openai.SamplerConfig(
        temperature=0.6,
        top_p=1.0,
        top_k=20,
    )
    client = TestClient(create_app(state))

    response = client.post(
        "/v1/mtplx/settings",
        json={"top_p": 0.95, "draft_top_p": 0.8},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["top_p"] == 0.95
    assert body["draft_top_p"] == 0.8
    assert state.args.top_p == 0.95
    assert state.args.draft_top_p == 0.8
    assert state.draft_sampler.top_p == 0.8


def test_app_capabilities_advertise_cache_and_kv_quant_controls():
    state = _fake_state()
    client = TestClient(create_app(state))

    response = client.get("/v1/mtplx/app/capabilities")

    assert response.status_code == 200
    body = response.json()
    restart_required = set(body["restart_required_settings"])
    assert "ram_session_cache_policy" in restart_required
    assert "ram_session_block_prefix_restore" in restart_required
    assert "ram_session_cache_max_entries" in restart_required
    assert "ram_session_cache_max_size" in restart_required
    assert "ram_session_cache_per_session_max_size" in restart_required
    assert "paged_kv_quantization" in restart_required
    assert body["features"]["ram_session_cache_controls"] is True
    assert body["features"]["paged_kv_quantization"] is True


def test_settings_report_effective_cache_and_kv_quant_controls(monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", "0")
    monkeypatch.setenv("MTPLX_SESSION_BANK_MAX_ENTRIES", "1")
    monkeypatch.setenv("MTPLX_SESSION_BANK_MAX_BYTES", "1G")
    monkeypatch.setenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", "1G")
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", "q8")
    state = _fake_state()
    client = TestClient(create_app(state))

    response = client.get("/v1/mtplx/settings")

    assert response.status_code == 200
    body = response.json()
    assert body["api_key_required"] is False
    assert body["api_key_source"] == "none"
    assert body["ram_session_cache_policy"] == "minimal"
    assert body["ram_session_block_prefix_restore"] is False
    assert body["ram_session_cache_max_entries"] == 1
    assert body["ram_session_cache_max_size"] == "1G"
    assert body["ram_session_cache_per_session_max_size"] == "1G"
    assert body["paged_kv_quantization"] == "q8"
    assert "paged_kv_quantization" in body["restart_required_settings"]
    controls = body["model_controls"]
    assert controls["schema_version"] == 1
    assert controls["model_family"] == "qwen3_6"
    assert controls["backend_id"] == "qwen3_next"
    assert controls["draft_control"]["minimum"] == 1
    assert controls["draft_control"]["maximum"] == 3
    assert controls["tune"]["supported"] is True
    assert controls["reasoning"]["parser"] == "qwen3"


def test_settings_emit_gemma_block_controls_and_tune_policy():
    state = _fake_state()
    state.backend_descriptor = openai.descriptor_for_backend_id("gemma4_assistant")
    state.model_id = "Youssofal/Gemma4-MTPLX-Optimized-Speed"
    state.args.model = "Youssofal/Gemma4-MTPLX-Optimized-Speed"
    client = TestClient(create_app(state))

    response = client.get("/v1/mtplx/settings")

    assert response.status_code == 200
    controls = response.json()["model_controls"]
    assert controls["model_family"] == "gemma4"
    assert controls["backend_id"] == "gemma4_assistant"
    assert controls["draft_control"]["display_label"] == "Draft block"
    assert controls["draft_control"]["minimum"] == 2
    assert controls["draft_control"]["maximum"] == 8
    assert controls["draft_control"]["value_labels"] == [
        "Block 2",
        "Block 3",
        "Block 4",
        "Block 5",
        "Block 6",
        "Block 7",
        "Block 8",
    ]
    assert controls["sampling"]["temperature"] == 1.0
    assert controls["sampling"]["top_k"] == 64
    assert controls["reasoning"]["parser"] == "gemma4"
    assert controls["tune"]["supported"] is True
    assert controls["tune"]["control_field"] == "draft_block_size"
    assert controls["tune"]["candidates"] == [
        "AR",
        "Block 2",
        "Block 3",
        "Block 4",
        "Block 5",
        "Block 6",
        "Block 7",
        "Block 8",
    ]
    assert controls["kv_quant"]["supported"] is False
    assert controls["kv_quant"]["disabled_reason"] == "KV quantization is not supported for Gemma."
    assert controls["context_window"]["maximum"] == 262144
    assert response.json()["context_window_policy"]["maximum"] == 262144


def test_step_descriptor_is_experimental_and_not_qwen_tune():
    controls = openai.model_controls_for_descriptor(
        openai.descriptor_for_backend_id("step3p5_mtp"),
        model_ref="stepfun-ai/Step-3.7-MTPLX",
    )

    assert controls["model_family"] == "step"
    assert controls["backend_id"] == "step3p5_mtp"
    assert controls["support_level"] == "experimental_contract_gated"
    assert controls["draft_control"]["default"] == 1
    assert controls["draft_control"]["minimum"] == 1
    assert controls["draft_control"]["maximum"] == 3
    assert controls["reasoning"]["supported"] is True
    assert controls["reasoning"]["parser"] == "step3p5"
    assert controls["reasoning"]["default"] == "auto"
    assert controls["reasoning"]["effort_levels"] == ["low", "medium", "high"]
    assert controls["reasoning"]["default_effort"] == "low"
    assert controls["tune"]["supported"] is False
    assert controls["kv_quant"]["supported"] is False
    assert controls["kv_quant"]["disabled_reason"] == "KV quantization is not supported for Step."


def test_step_backend_chat_policy_injects_language_anchor():
    state = _fake_state()
    state.backend_descriptor = openai.descriptor_for_backend_id("step3p5_mtp")

    messages, changed = openai._with_backend_chat_policy(
        state,
        [openai.ChatMessage(role="user", content="hi")],
    )

    assert changed is True
    assert [message.role for message in messages] == ["system", "user"]
    assert "MTPLX Step language policy:" in messages[0].content
    assert "Use English by default" in messages[0].content
    assert "Never answer in Chinese for English or ambiguous input." in messages[0].content
    assert messages[1].content == "hi"


def test_step_backend_chat_policy_preserves_existing_system_prompt():
    state = _fake_state()
    state.backend_descriptor = openai.descriptor_for_backend_id("step3p5_mtp")

    messages, changed = openai._with_backend_chat_policy(
        state,
        [
            openai.ChatMessage(role="system", content="Client policy"),
            openai.ChatMessage(role="user", content="hi"),
        ],
    )

    assert changed is True
    assert [message.role for message in messages] == ["system", "user"]
    assert messages[0].content.startswith("Client policy\n\nMTPLX Step language policy:")
    assert messages[1].content == "hi"


def test_qwen_backend_chat_policy_does_not_mutate_messages():
    state = _fake_state()
    original = [openai.ChatMessage(role="user", content="hi")]

    messages, changed = openai._with_backend_chat_policy(state, original)

    assert changed is False
    assert messages is original


def test_settings_reject_cache_and_kv_quant_mutations_as_restart_required():
    state = _fake_state()
    client = TestClient(create_app(state))

    response = client.post(
        "/v1/mtplx/settings",
        json={
            "paged_kv_quantization": "q8",
            "ram_session_cache_policy": "bounded",
            "ram_session_cache_max_entries": 2,
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    message = error["message"]
    assert "paged_kv_quantization" in message
    assert "ram_session_cache_max_entries" in message
    assert "ram_session_cache_policy" in message
    # Structured detail rides along so CLI/app render it without
    # parsing a repr out of the message (QA-105).
    assert error["detail"]["error"] == "restart_required"
    assert set(error["detail"]["keys"]) == {
        "paged_kv_quantization",
        "ram_session_cache_max_entries",
        "ram_session_cache_policy",
    }


def test_server_console_controls_reasoning_and_mtp_defaults():
    state = _fake_state()

    assert "Reasoning: off" in openai._server_console_handle_command(
        state, "/reasoning off"
    )
    assert state.args.reasoning == "off"
    assert state.args.enable_thinking is False

    assert "Reasoning: auto" in openai._server_console_handle_command(
        state, "/reasoning auto"
    )
    assert state.args.reasoning == "auto"
    assert state.args.enable_thinking is True

    assert "MTP: off" in openai._server_console_handle_command(state, "/mtp off")
    assert state.args.generation_mode == "ar"

    assert "MTP: on" in openai._server_console_handle_command(state, "/mtp on")
    assert state.args.generation_mode == "mtp"


def test_openai_server_health_metrics_and_models_fake_state():
    client = TestClient(create_app(_fake_state()))

    root_head = client.head("/")
    assert root_head.status_code == 200

    root = client.get("/")
    assert root.status_code == 200
    assert "text/html" in root.headers["content-type"]
    # Brand and chat scaffold
    assert "<title>MTPLX</title>" in root.text
    assert "MTPLX" in root.text
    assert 'id="messages"' in root.text
    assert 'id="prompt"' in root.text
    assert "Message MTPLX" in root.text
    assert "/v1/chat/completions" in root.text
    assert "reasoning_content" in root.text
    # Inference settings sidebar — all sliders now
    assert 'id="ctl-temp"' in root.text
    assert 'id="ctl-top-p"' in root.text
    assert 'id="ctl-top-k" type="range"' in root.text
    assert 'id="ctl-mtp" type="checkbox"' in root.text
    assert 'id="ctl-depth" type="range"' in root.text
    assert 'id="ctl-max-tokens" type="range"' in root.text
    assert 'id="ctl-system"' in root.text
    assert 'id="reset-defaults"' in root.text
    # New layout: avatar circles + reasoning-as-its-own-block + turn-* classes
    assert "turn turn-assistant" in root.text
    assert 'class="avatar"' in root.text
    assert "reasoning-block" in root.text
    # Auto-scroll, stop, new-chat, persistence
    assert 'id="jump-pill"' not in root.text
    assert 'id="messages-bottom"' in root.text
    assert "ResizeObserver" in root.text
    assert "scrollIntoView" in root.text
    assert "forceAutoScroll" in root.text
    assert "SCROLL_PIN_THRESHOLD = 160" in root.text
    assert 'id="new-chat-btn"' in root.text
    assert "AbortController" in root.text
    # Browser chat must hydrate generation controls from the app-owned daemon.
    # localStorage is allowed to keep only the custom system prompt so stale
    # browser settings cannot override app-owned reasoning, sampler, or MTP mode.
    assert "mtplx.chat.settings.v5:" in root.text
    assert 'const LEGACY_SETTINGS_KEY = "mtplx.chat.settings.v4"' in root.text
    assert "fetchDaemonSettings" in root.text
    assert 'fetch("/v1/mtplx/settings", {cache: "no-store"})' in root.text
    assert 'fetch("/v1/mtplx/settings", {' in root.text
    assert 'method: "POST"' in root.text
    assert "daemonSettingsPayload" in root.text
    assert "refreshDaemonSettings" in root.text
    assert "window.setInterval(() => refreshDaemonSettings(), 1500)" in root.text
    assert "JSON.stringify({system:" in root.text
    assert 'const rawMode = payload.generation_mode == null ? "" : String(payload.generation_mode);' in root.text
    assert "Settings mirror the running MTPLX app." in root.text
    # Auto-detect of context length must be hooked up so the slider isn't
    # capped at a stale 32k for a 256k-context model.
    assert "discoverServerLimits" in root.text
    assert "/health" in root.text
    # Transport watchdog plus heartbeat handling keep long active generations
    # from looking like crashed servers.
    assert "armStallWatchdog" in root.text
    assert "mtplx_progress.heartbeat" in root.text
    assert "Still working" in root.text
    assert "stream connection went quiet" in root.text
    assert "server has crashed" not in root.text
    # Markdown via marked.js
    assert "marked.min.js" in root.text
    # Live tps element
    assert 'id="live-stats"' in root.text
    assert "tok/s" in root.text
    assert '"mtp_enabled": true' in root.text
    assert 'generation_mode: settingsNow.mtp_enabled ? "mtp" : "ar"' in root.text
    assert '"depth": 3' in root.text
    assert 'id="ctl-depth" type="range" min="1" max="3" step="1" value="3"' in root.text
    # Updated max-tokens default and cap.
    assert 'value="8192"' in root.text
    assert 'min="256" max="32768"' in root.text

    v1 = client.get("/v1")
    assert v1.status_code == 200
    assert v1.json()["openwebui"]["base_url"].endswith("/v1")
    assert "Paste this URL into Open WebUI" in v1.json()["message"]

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model"] == "mtplx-test-model"
    assert health.json()["profile"]["model_id"] == "mtplx-test-model"
    assert health.json()["profile"]["profile_default_model_id"] != "mtplx-test-model"
    assert health.json()["profile"]["sampler"] == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    }
    assert health.json()["api_key_required"] is False
    assert health.json()["api_key_source"] == "none"
    assert health.json()["paged_kv_quantization"] in {"off", "q4", "q8"}
    assert health.json()["warmup"]["ran"] is False
    assert health.json()["startup"]["pid"] > 0
    assert health.json()["startup"]["warmup"]["ran"] is False
    assert health.json()["startup"]["api_key_source"] == "none"
    assert health.json()["startup"]["model_controls"]["model_family"] == "qwen3_6"
    assert health.json()["startup"]["model_controls"]["draft_control"]["maximum"] == 3
    assert health.json()["startup"]["tool_prompt_mode"] == "hybrid"
    assert health.json()["startup"]["tool_contract_active"] is True
    from mtplx.server.openai import _MTPLX_TOOL_CONTRACT_POLICY_VERSION

    assert (
        health.json()["startup"]["tool_contract_policy_version"]
        == _MTPLX_TOOL_CONTRACT_POLICY_VERSION
    )
    assert health.json()["thermal"]["max_requested"] is False
    assert health.json()["foreground_active"] == 0
    assert health.json()["active_requests"] == 0
    assert health.json()["last_request_started_at"] == 0.0
    assert health.json()["opencode_short_context_depth2_tokens"] is None
    assert health.json()["opencode_short_context_depth_policy"] == {
        "active": False,
        "reason": "disabled_depth_preservation",
        "default_depth": 3,
    }

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["latest"]["tok_s"] == 12.5

    models = client.get("/v1/models")
    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "mtplx-test-model"


def test_openai_server_health_profile_sampler_reports_active_override():
    state = _fake_state()
    state.args.temperature = 1.0
    state.args.top_p = 0.95
    state.args.top_k = 64
    client = TestClient(create_app(state))

    health = client.get("/health")

    assert health.status_code == 200
    profile = health.json()["profile"]
    assert profile["sampler"] == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
    }
    assert profile["profile_default_sampler"] == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    }


def test_chat_completion_response_reports_served_model_when_request_model_is_stale(
    monkeypatch,
):
    state = _fake_state()
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai, "_run_generation", lambda *_args, **_kwargs: _fake_generation("OK")
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "opencode"},
        json={
            "model": "gemma4-mtplx-optimized-speed",
            "messages": [{"role": "user", "content": "Reply OK only."}],
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "mtplx-test-model"
    stats = payload["mtplx_stats"]
    assert stats["request_model"] == "gemma4-mtplx-optimized-speed"
    assert stats["served_model_id"] == "mtplx-test-model"
    assert stats["request_model_matches_served_model"] is False


def test_chat_stream_response_reports_served_model_when_request_model_is_stale(
    monkeypatch,
):
    state = _fake_state()
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(openai, "_run_generation", _fake_streaming_generation("OK"))

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "opencode"},
        json={
            "model": "gemma4-mtplx-optimized-speed",
            "messages": [{"role": "user", "content": "Reply OK only."}],
            "stream": True,
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    assert payloads
    assert {payload["model"] for payload in payloads} == {"mtplx-test-model"}
    final = [payload for payload in payloads if payload["choices"][0]["finish_reason"]]
    stats = final[-1]["mtplx_stats"]
    assert stats["request_model"] == "gemma4-mtplx-optimized-speed"
    assert stats["served_model_id"] == "mtplx-test-model"
    assert stats["request_model_matches_served_model"] is False


def test_browser_auth_bootstrap_unlocks_api_key_chat_ui():
    client = TestClient(create_app(_fake_state(api_key="test-key")))

    blocked = client.get("/", follow_redirects=False)
    assert blocked.status_code == 401

    bad_bootstrap = client.get(
        f"{openai._BROWSER_AUTH_PATH}?{openai._BROWSER_AUTH_QUERY_PARAM}=wrong",
        follow_redirects=False,
    )
    assert bad_bootstrap.status_code == 401

    bootstrap = client.get(
        f"{openai._BROWSER_AUTH_PATH}?"
        f"{openai._BROWSER_AUTH_QUERY_PARAM}=test-key&next=/",
        follow_redirects=False,
    )
    assert bootstrap.status_code == 303
    assert bootstrap.headers["location"] == "/"
    cookie_header = bootstrap.headers["set-cookie"]
    assert f"{openai._BROWSER_AUTH_COOKIE}=test-key" in cookie_header
    assert "HttpOnly" in cookie_header
    assert "SameSite=lax" in cookie_header

    root = client.get("/")
    assert root.status_code == 200
    assert 'id="messages"' in root.text
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["api_key_required"] is True


def test_startup_browser_urls_use_auth_bootstrap_when_api_key_is_required():
    args = parse_args(
        ["--warmup-tokens", "0", "--port", "8020", "--api-key", "key with space"]
    )

    assert openai._startup_chat_url(args) == "http://127.0.0.1:8020/"
    chat_url = openai._startup_browser_chat_url(args)
    dashboard_url = openai._startup_browser_dashboard_url(args)

    assert chat_url.startswith("http://127.0.0.1:8020/mtplx/browser-auth?")
    assert openai._startup_printable_chat_url(args) == chat_url
    assert "mtplx_api_key=key+with+space" in chat_url
    assert "next=%2F" in chat_url
    assert "next=%2Fdashboard%2F" in dashboard_url


def test_openai_server_auth_and_rate_limit_fake_state():
    client = TestClient(create_app(_fake_state(api_key="test-key", rate_limit=1)))

    assert client.get("/v1/models").status_code == 401
    assert (
        client.get(
            "/v1/models", headers={"Authorization": "Bearer test-key"}
        ).status_code
        == 200
    )

    limited = client.get("/v1/models", headers={"Authorization": "Bearer test-key"})
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) > 0


def test_thermal_fan_mode_can_require_actual_ramp(monkeypatch):
    from mtplx import thermal

    state = _fake_state()
    client = TestClient(create_app(state))
    calls: dict[str, object] = {}

    def fake_set(profile: str, **kwargs):
        calls["profile"] = profile
        calls["kwargs"] = kwargs
        return {"ok": True, "after": {"ok": True}, "message": "ramped"}

    monkeypatch.setattr(thermal, "set_thermal_profile_verified", fake_set)
    monkeypatch.setattr(thermal, "fan_summary", lambda: {"ok": True})

    response = client.post(
        "/v1/mtplx/thermal/fan_mode",
        json={"mode": "max", "require_actual_ramp": True, "timeout_s": 7.5},
    )

    assert response.status_code == 200
    assert response.json()["verified"] is True
    assert calls["profile"] == "performance"
    assert calls["kwargs"] == {
        "require_actual_ramp": True,
        "actual_ramp_timeout_s": 7.5,
    }
    assert response.json()["current_mode"] == "max"


def test_thermal_fan_mode_accepts_smart_default_and_legacy_auto(monkeypatch):
    from mtplx import thermal

    state = _fake_state()
    client = TestClient(create_app(state))
    restore_calls: list[str] = []

    monkeypatch.setattr(
        thermal,
        "restore_thermal_profile_verified",
        lambda **_kwargs: restore_calls.append("auto") or {"ok": True},
    )
    monkeypatch.setattr(thermal, "fan_summary", lambda: {"ok": True})

    for requested, expected in (
        ("smart", "smart"),
        ("default", "default"),
        ("auto", "default"),
    ):
        response = client.post(
            "/v1/mtplx/thermal/fan_mode",
            json={"mode": requested},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["verified"] is True
        assert body["current_mode"] == expected
        assert state.fan_mode == expected
        assert state.args.fan_mode == expected

    assert restore_calls == ["auto", "auto"]


def test_health_reports_smart_fan_boost_state():
    state = _fake_state()
    state.fan_mode = "smart"
    state.args.fan_mode = "smart"
    state.smart_fans = SimpleNamespace(
        status=lambda: {
            "active": True,
            "active_count": 1,
            "active_requests": ["request-1"],
            "commanded_max": True,
            "last_transition_at": 123.0,
            "last_error": None,
            "last_result": {"ok": True},
        }
    )
    client = TestClient(create_app(state))

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["fan_mode"] == "smart"
    assert body["fan_boost_active"] is True
    assert body["smart_fan_active_count"] == 1
    assert body["thermal"]["max_requested"] is True
    assert body["thermal"]["smart"]["active_count"] == 1


def test_anthropic_messages_rejects_empty_request_before_generation():
    client = TestClient(create_app(_fake_state()))

    response = client.post(
        "/v1/messages",
        json={
            "model": "mtplx-test-model",
            "max_tokens": 8,
            "stream": True,
            "messages": [],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "messages must not be empty"


def test_chat_ui_uses_server_depth_default():
    state = _fake_state()
    state.args.depth = 2
    client = TestClient(create_app(state))

    root = client.get("/")

    assert root.status_code == 200
    assert '"depth": 2' in root.text
    assert '"mtp_enabled": true' in root.text
    assert 'id="ctl-depth" type="range" min="1" max="3" step="1" value="2"' in root.text


def test_chat_generation_mode_request_override_routes_ar(monkeypatch):
    captured: dict[str, object] = {}
    state = _fake_state()
    client = TestClient(create_app(state))

    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])

    def fake_run_generation(_state, prompt_ids, **kwargs):
        captured["prompt_ids"] = prompt_ids
        captured["generation_mode"] = kwargs["generation_mode"]
        captured["depth"] = kwargs["depth"]
        return {
            "text": "ok",
            "tokens": [4],
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 1,
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 1,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-mtplx-cache-mode": "bypass",
            "x-mtplx-allow-client-controls": "1",
        },
        json={
            "messages": [{"role": "user", "content": "Say READY"}],
            "max_tokens": 4,
            "generation_mode": "ar",
            "depth": 3,
        },
    )

    assert response.status_code == 200
    assert captured["generation_mode"] == "ar"
    assert captured["depth"] == 0
    assert response.json()["mtplx_stats"]["generation_mode"] == "ar"
    assert response.json()["mtplx_stats"]["mtp_depth"] == 0


def test_generation_truth_stats_distinguish_stock_ar_from_target_ar():
    target_state = _fake_state()
    target_state.args.load_mtp = True
    target_state.runtime.mtp_enabled = True
    target_state.draft_lm_head = {"draft_only": {"bits": 4}}

    stock_state = _fake_state()
    stock_state.args.load_mtp = False
    stock_state.runtime.mtp_enabled = False

    target = openai._generation_truth_stats(target_state, "ar")
    assert target["benchmark_mode"] == "mtplx_mtp_loaded_target_ar"
    assert target["draft_head_installed"] is True
    stock = openai._generation_truth_stats(stock_state, "ar")
    assert stock["benchmark_mode"] == "mtplx_stock_ar_unloaded"
    assert stock["load_mtp"] is False
    assert stock["runtime_mtp_enabled"] is False


def test_chat_generation_mode_request_override_routes_mtp_depth(monkeypatch):
    captured: dict[str, object] = {}
    client = TestClient(create_app(_fake_state()))

    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        captured["generation_mode"] = kwargs["generation_mode"]
        captured["depth"] = kwargs["depth"]
        return {
            "text": "ok",
            "tokens": [4],
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 1,
            },
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-mtplx-cache-mode": "bypass",
            "x-mtplx-allow-client-controls": "1",
        },
        json={
            "messages": [{"role": "user", "content": "Say READY"}],
            "max_tokens": 4,
            "generation_mode": "mtp",
            "depth": 1,
        },
    )

    assert response.status_code == 200
    assert captured == {"generation_mode": "mtp", "depth": 1}
    assert response.json()["mtplx_stats"]["generation_mode"] == "mtp"
    assert response.json()["mtplx_stats"]["mtp_depth"] == 1


def test_chat_request_controls_are_server_owned_without_override(monkeypatch):
    captured: dict[str, object] = {}
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    client = TestClient(create_app(state))

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        captured.update(kwargs)
        return {
            "text": "ok",
            "tokens": [4],
            "stats": {
                **kwargs["request_observability"],
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 1,
            },
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Say READY"}],
            "max_tokens": 4,
            "generation_mode": "ar",
            "depth": 1,
            "temperature": 0.01,
            "top_p": 0.2,
            "top_k": 1,
            "enable_thinking": False,
        },
    )

    assert response.status_code == 200
    assert captured["generation_mode"] == "mtp"
    assert captured["depth"] == 3
    assert captured["temperature"] == 0.6
    assert captured["top_p"] == 0.95
    assert captured["top_k"] == 20
    _messages, template_kwargs = state.runtime.tokenizer.calls[0]
    assert template_kwargs["enable_thinking"] is True
    stats = captured["request_observability"]
    assert stats["mtplx_control_owner"] == "server"
    assert stats["client_controls_allowed"] is False
    assert stats["client_control_fields_ignored"] == [
        "temperature",
        "top_p",
        "top_k",
        "enable_thinking",
        "generation_mode",
        "draft_control",
    ]
    assert stats["client_sampler_fields_ignored"] == [
        "temperature",
        "top_p",
        "top_k",
    ]
    public_stats = response.json()["mtplx_stats"]
    assert public_stats["mtplx_control_owner"] == "server"
    assert public_stats["client_controls_allowed"] is False
    assert public_stats["client_control_fields_ignored"] == [
        "temperature",
        "top_p",
        "top_k",
        "enable_thinking",
        "generation_mode",
        "draft_control",
    ]
    assert public_stats["client_sampler_fields_ignored"] == [
        "temperature",
        "top_p",
        "top_k",
    ]


def test_opencode_chitchat_history_reaches_model_with_tools_kept(monkeypatch):
    captured: dict[str, object] = {}
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    client = TestClient(create_app(state))
    opencode_system_prompt = "You are OpenCode.\n" + ("coding agent policy\n" * 600)

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        captured.update(kwargs)
        return {
            "text": "I'm good, thanks.",
            "tokens": [4],
            "stats": {
                **kwargs["request_observability"],
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 4,
            },
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-mtplx-cache-mode": "bypass",
            "x-mtplx-client": "opencode",
        },
        json={
            "messages": [
                {"role": "system", "content": opencode_system_prompt},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hi! How can I help you today?"},
                {"role": "user", "content": "How are you?"},
            ],
            "tools": [_tool_schema()],
            "max_tokens": 16,
            "stream": False,
        },
    )

    assert response.status_code == 200
    stats = captured["request_observability"]
    assert stats["request_client_label"] == "opencode"
    assert stats["client_controls_allowed"] is False
    assert stats["transcript_canonicalized"] is True
    assert stats["opencode_simple_chat_contract_active"] is False
    assert stats["opencode_prompt_contract_profile"] == "opencode_agent"
    # Simple chitchat keeps the client's tools: dropping schemas for
    # greetings was reversed as a forbidden band-aid (2026-06-09).
    assert stats["request_filtered_tool_names"] == ["session_status"]
    assert stats["request_hidden_tool_names"] == []
    assert stats["request_tools_hidden_by_bridge"] is False
    assert stats["tool_contract_policy_version"] == "compact_tool_contract:schema_free:v1"
    assert stats["tool_contract_active"] is True
    assert stats["no_tools_contract_active"] is False
    assert stats["transcript_replaced_client_system_messages"] == 0
    assert stats["transcript_replaced_client_system_chars"] == 0
    rendered_messages, template_kwargs = state.runtime.tokenizer.calls[0]
    rendered_text = json.dumps(rendered_messages)
    # Compact mode carries tools via the schema-free contract text rather
    # than template tool schemas.
    assert "tools" not in template_kwargs
    assert "You are OpenCode." in rendered_text
    assert "coding agent policy" in rendered_text
    assert "MTPLX tool contract:" in rendered_text
    assert "session_status()" in rendered_text
    assert "How are you?" in rendered_text


def test_opencode_initial_coding_request_uses_compact_mtplx_agent_prompt(monkeypatch):
    captured: dict[str, object] = {}
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    client = TestClient(create_app(state))
    opencode_system_prompt = "You are OpenCode.\n" + ("coding agent policy\n" * 600)

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        captured.update(kwargs)
        return {
            "text": "I'll inspect the project.",
            "tokens": [4],
            "stats": {
                **kwargs["request_observability"],
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 4,
            },
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-mtplx-cache-mode": "bypass",
            "x-mtplx-client": "opencode",
        },
        json={
            "messages": [
                {"role": "system", "content": opencode_system_prompt},
                {"role": "user", "content": "Inspect this project and read package files."},
            ],
            "tools": [_tool_schema()],
            "max_tokens": 16,
            "stream": False,
        },
    )

    assert response.status_code == 200
    stats = captured["request_observability"]
    assert stats["opencode_simple_chat_contract_active"] is False
    assert stats["opencode_prompt_contract_profile"] == "opencode_agent"
    assert stats["transcript_replaced_client_system_messages"] == 0
    assert stats["transcript_replaced_client_system_chars"] == 0
    assert stats["request_effective_message_chars"][0] > 1_000
    rendered_messages, template_kwargs = state.runtime.tokenizer.calls[0]
    rendered_text = str(rendered_messages[0]["content"])
    assert "tools" not in template_kwargs
    assert "You are OpenCode." in rendered_text
    assert "coding agent policy" in rendered_text
    assert "MTPLX tool contract:" in rendered_text


def test_chat_long_context_depth_cap_resolves_runtime_depth(monkeypatch):
    captured: dict[str, object] = {}
    state = _fake_state()
    client = TestClient(create_app(state))

    monkeypatch.setenv("MTPLX_LONG_CONTEXT_MTP_DEPTH_POLICY", "auto")
    monkeypatch.setenv("MTPLX_LONG_CONTEXT_MTP_DEPTH_THRESHOLD", "12000")
    monkeypatch.setenv("MTPLX_LONG_CONTEXT_MTP_DEPTH", "2")
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1] * 12506)

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        captured.update(kwargs)
        return {
            "text": "ok",
            "tokens": [4],
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["resolved_mtp_depth"],
                "requested_mtp_depth": kwargs["depth"],
                "completion_tokens": 1,
            },
            "prompt_tokens": 12506,
            "completion_tokens": 1,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "pi"},
        json={
            "messages": [{"role": "user", "content": "Continue after tools"}],
            "max_tokens": 4,
        },
    )

    assert response.status_code == 200
    assert captured["generation_mode"] == "mtp"
    assert captured["depth"] == 3
    assert captured["resolved_mtp_depth"] == 2
    assert "depth=2" in captured["session_policy_fingerprint"]
    observability = captured["request_observability"]
    assert observability["request_depth"] == 3
    assert observability["request_effective_mtp_depth"] == 2
    assert observability["long_context_mtp_depth_policy"]["active"] is True
    assert (
        observability["long_context_mtp_depth_policy"]["reason"]
        == "long_context_depth_cap"
    )


def test_opencode_short_context_preserves_depth3(monkeypatch):
    captured: dict[str, object] = {}
    client = TestClient(create_app(_fake_state()))

    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1] * 5000)

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        captured["depth"] = kwargs["depth"]
        return {
            "text": "ok",
            "tokens": [4],
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 1,
            },
            "prompt_tokens": 5000,
            "completion_tokens": 1,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "opencode"},
        json={
            "messages": [{"role": "user", "content": "Say READY"}],
            "max_tokens": 4,
        },
    )

    assert response.status_code == 200
    stats = response.json()["mtplx_stats"]
    assert captured["depth"] == 3
    assert stats["mtp_depth"] == 3


def test_opencode_short_context_depth_policy_respects_explicit_depth(monkeypatch):
    captured: dict[str, object] = {}
    client = TestClient(create_app(_fake_state()))

    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1] * 5000)

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        captured["depth"] = kwargs["depth"]
        return {
            "text": "ok",
            "tokens": [4],
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 1,
            },
            "prompt_tokens": 5000,
            "completion_tokens": 1,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "opencode"},
        json={
            "messages": [{"role": "user", "content": "Say READY"}],
            "max_tokens": 4,
            "depth": 3,
        },
    )

    assert response.status_code == 200
    stats = response.json()["mtplx_stats"]
    assert captured["depth"] == 3
    assert stats["mtp_depth"] == 3


def test_opencode_short_context_depth_policy_keeps_depth3_above_threshold(monkeypatch):
    captured: dict[str, object] = {}
    client = TestClient(create_app(_fake_state()))

    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1] * 8000)

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        captured["depth"] = kwargs["depth"]
        return {
            "text": "ok",
            "tokens": [4],
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 1,
            },
            "prompt_tokens": 8000,
            "completion_tokens": 1,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "opencode"},
        json={
            "messages": [{"role": "user", "content": "Say READY"}],
            "max_tokens": 4,
        },
    )

    assert response.status_code == 200
    stats = response.json()["mtplx_stats"]
    assert captured["depth"] == 3
    assert stats["mtp_depth"] == 3


def test_opencode_short_context_depth_policy_reports_decision():
    request = openai.ChatCompletionRequest(
        messages=[{"role": "user", "content": "Say READY"}],
    )

    depth, policy = openai._opencode_short_context_depth_policy(
        request,
        headers={"x-mtplx-client": "opencode"},
        metadata={},
        generation_mode="mtp",
        request_depth=3,
        prompt_tokens=5000,
    )

    assert depth == 3
    assert policy == {
        "active": False,
        "client": "opencode",
        "effective_depth": 3,
        "explicit_depth": False,
        "prompt_tokens": 5000,
        "reason": "disabled_depth_preservation",
        "requested_depth": 3,
        "threshold": None,
    }


def test_streaming_session_uses_generation_final_postcommit_without_retokenized_tail(
    monkeypatch,
):
    state = _fake_streaming_session_state()
    captured: dict[str, object] = {"batch_keys": []}

    class CapturingScheduler:
        def __init__(self) -> None:
            self.current_batch_key: str | None = None

        def is_owner_thread(self) -> bool:
            return False

        def submit_foreground(self, fn, *args, batch_key=None, **kwargs):
            future: Future = Future()
            previous = self.current_batch_key
            self.current_batch_key = batch_key
            captured["batch_keys"].append(batch_key)
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as exc:  # pragma: no cover - surfaced by caller
                future.set_exception(exc)
            finally:
                self.current_batch_key = previous
            return future

        def shutdown(self, **_kwargs):
            return None

    scheduler = CapturingScheduler()
    state.model_scheduler = scheduler

    def fail_retokenized(*_args, **_kwargs):
        raise AssertionError(
            "streaming fast path must not retokenize/prefill postcommit"
        )

    def fake_store_generation_final(*_args, **_kwargs):
        assert scheduler.current_batch_key == "postcommit.stream.final:stream-session"
        return {
            "stored": True,
            "mode": "generation_final_exact",
            "reason": "compatible",
            "prefix_len": 3,
            "nbytes": 123,
        }

    def fake_run_generation(_state, prompt_ids, **kwargs):
        streaming_response = kwargs.get("streaming_response")
        is_streaming = (
            kwargs.get("token_callback") is not None
            if streaming_response is None
            else bool(streaming_response)
        )
        expected_batch_key = (
            "chat.stream" if is_streaming else "chat.nonstream"
        )
        assert scheduler.current_batch_key == expected_batch_key
        captured.setdefault(
            "commit_final_state_to_bank",
            kwargs.get("commit_final_state_to_bank"),
        )
        token_callback = kwargs.get("token_callback")
        tokens = [ord("O"), ord("K")]
        if token_callback is not None:
            token_callback(tokens[:1])
            token_callback(tokens[1:])
        return {
            "text": "OK",
            "tokens": tokens,
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 2,
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 2,
            "finish_reason": "stop",
            "_final_state": _fake_final_state(tokens),
        }

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fail_retokenized)
    monkeypatch.setattr(
        openai, "_store_generation_final_history_snapshot", fake_store_generation_final
    )
    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    with TestClient(create_app(state)) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"x-mtplx-session-id": "stream-session"},
            json={
                "messages": [{"role": "user", "content": "Say OK"}],
                "enable_thinking": False,
                "stream": True,
                "max_tokens": 4,
            },
        ) as response:
            response_status = response.status_code
            response_text = response.read().decode()
        second = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-session-id": "stream-session"},
            json={
                "messages": [
                    {"role": "user", "content": "Say OK"},
                    {"role": "assistant", "content": "OK"},
                    {"role": "user", "content": "Again"},
                ],
                "enable_thinking": False,
                "max_tokens": 4,
            },
        )

    assert response_status == 200
    assert "data: [DONE]" in response_text
    assert '"content": "OK"' in response_text or (
        '"content": "O"' in response_text and '"content": "K"' in response_text
    )
    assert '"mode": "generation_final_exact"' in response_text
    assert captured["commit_final_state_to_bank"] is False
    assert captured["batch_keys"] == [
        "chat.stream",
        "postcommit.stream.final:stream-session",
        "chat.nonstream",
    ]
    assert second.status_code == 200
    assert "already in flight" not in second.text


def test_streaming_unsafe_postcommit_releases_without_blocking_second_request(
    monkeypatch,
):
    state = _fake_streaming_session_state()
    scheduled: list[dict] = []

    def fake_store_generation_final(*_args, **_kwargs):
        return {
            "stored": False,
            "mode": "unsafe",
            "reason": "retokenized_history_mismatch",
        }

    def fake_schedule(*_args, **kwargs):
        scheduled.append(kwargs)
        return {
            "stored": False,
            "mode": "async_pending",
            "reason": kwargs["unsafe_reason"],
        }

    def fake_run_generation(_state, prompt_ids, **kwargs):
        token_callback = kwargs.get("token_callback")
        tokens = [ord("O"), ord("K")]
        if token_callback is not None:
            token_callback(tokens)
        return {
            "text": "OK",
            "tokens": tokens,
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 2,
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 2,
            "finish_reason": "stop",
            "_final_state": _fake_final_state(tokens),
        }

    monkeypatch.setattr(
        openai, "_store_generation_final_history_snapshot", fake_store_generation_final
    )
    monkeypatch.setattr(openai, "_schedule_idle_postcommit_snapshot", fake_schedule)
    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    with TestClient(create_app(state)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-session-id": "unsafe-session"},
            json={
                "messages": [{"role": "user", "content": "Say OK"}],
                "enable_thinking": False,
                "stream": True,
                "max_tokens": 4,
            },
        )
        second = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-session-id": "unsafe-session"},
            json={
                "messages": [{"role": "user", "content": "Say OK again"}],
                "enable_thinking": False,
                "stream": True,
                "max_tokens": 4,
            },
        )

    assert response.status_code == 200
    assert '"mode": "async_pending"' in response.text
    assert '"reason": "retokenized_history_mismatch"' in response.text
    assert '"session_prompt_prefix_commit"' in response.text
    assert '"postcommit_prompt_prefix"' in response.text
    metrics_with_frontier = [
        metric
        for metric in state.last_metrics
        if metric.get("session_prompt_prefix_commit")
    ]
    assert metrics_with_frontier
    assert metrics_with_frontier[-1]["session_prompt_prefix_commit"][
        "boundary_kind"
    ] == "postcommit_prompt_prefix"
    assert metrics_with_frontier[-1]["session_postcommit_snapshot"] == {
        "stored": False,
        "mode": "async_pending",
        "reason": "retokenized_history_mismatch",
    }
    assert scheduled
    assert second.status_code == 200
    assert "already in flight" not in second.text


def test_streaming_stop_boundary_mismatch_schedules_idle_retokenized_postcommit(
    monkeypatch,
):
    state = _fake_streaming_session_state()
    scheduled: list[dict] = []

    def fake_store_generation_final(*_args, **_kwargs):
        return {
            "stored": False,
            "mode": "unsafe",
            "reason": "stop_token_boundary_mismatch",
        }

    def fake_schedule(*_args, **kwargs):
        scheduled.append(kwargs)
        return {
            "stored": False,
            "mode": "async_pending",
            "reason": kwargs["unsafe_reason"],
        }

    def fake_run_generation(_state, prompt_ids, **kwargs):
        token_callback = kwargs.get("token_callback")
        tokens = [ord("O"), ord("K")]
        if token_callback is not None:
            token_callback(tokens)
        return {
            "text": "OK",
            "tokens": tokens,
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 2,
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 2,
            "finish_reason": "stop",
            "_final_state": _fake_final_state(tokens),
        }

    monkeypatch.setattr(
        openai, "_store_generation_final_history_snapshot", fake_store_generation_final
    )
    monkeypatch.setattr(openai, "_schedule_idle_postcommit_snapshot", fake_schedule)
    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    with TestClient(create_app(state)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-session-id": "stop-boundary-session"},
            json={
                "messages": [{"role": "user", "content": "Say OK"}],
                "enable_thinking": False,
                "stream": True,
                "max_tokens": 4,
            },
        )

    assert response.status_code == 200
    assert '"mode": "async_pending"' in response.text
    assert '"reason": "stop_token_boundary_mismatch"' in response.text
    assert '"postcommit_prompt_prefix"' in response.text
    assert scheduled


def test_nonstream_unsafe_mtp_schedules_async_postcommit_in_default_mode(
    monkeypatch,
):
    state = _fake_streaming_session_state()
    scheduled: list[dict] = []

    def fake_store_generation_final(*_args, **_kwargs):
        return {
            "stored": False,
            "mode": "unsafe",
            "reason": "retokenized_history_mismatch",
        }

    def fail_retokenized(*_args, **_kwargs):
        raise AssertionError("non-stream MTP must not retokenize inline by default")

    def fake_schedule(*_args, **kwargs):
        scheduled.append(kwargs)
        return {
            "stored": False,
            "mode": "async_pending",
            "reason": kwargs["unsafe_reason"],
        }

    def fake_run_generation(_state, prompt_ids, **kwargs):
        tokens = [ord("O"), ord("K")]
        return {
            "text": "OK",
            "tokens": tokens,
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 2,
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 2,
            "finish_reason": "stop",
            "_final_state": None,
        }

    monkeypatch.setattr(
        openai, "_store_generation_final_history_snapshot", fake_store_generation_final
    )
    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fail_retokenized)
    monkeypatch.setattr(openai, "_schedule_idle_postcommit_snapshot", fake_schedule)
    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    with TestClient(create_app(state)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-session-id": "nonstream-unsafe-session"},
            json={
                "messages": [{"role": "user", "content": "Say OK"}],
                "enable_thinking": False,
                "stream": False,
                "max_tokens": 4,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["mtplx_stats"]["session_postcommit_snapshot"] == {
        "stored": False,
        "mode": "async_pending",
        "reason": "missing_generation_final_state",
    }
    assert state.last_metrics[-1]["session_postcommit_snapshot"] == {
        "stored": False,
        "mode": "async_pending",
        "reason": "missing_generation_final_state",
    }
    assert scheduled
    assert scheduled[0]["unsafe_reason"] == "missing_generation_final_state"


def test_streaming_ar_schedules_async_postcommit_in_default_mode(monkeypatch):
    state = _fake_streaming_session_state()
    scheduled: list[dict] = []

    def fail_retokenized(*_args, **_kwargs):
        raise AssertionError("AR streaming must not retokenize inline by default")

    def fake_schedule(*_args, **kwargs):
        scheduled.append(kwargs)
        return {
            "stored": False,
            "mode": "async_pending",
            "reason": kwargs["unsafe_reason"],
        }

    def fake_run_generation(_state, prompt_ids, **kwargs):
        token_callback = kwargs.get("token_callback")
        tokens = [ord("A"), ord("R")]
        if token_callback is not None:
            token_callback(tokens)
        return {
            "text": "AR",
            "tokens": tokens,
            "stats": {
                "generation_mode": "ar",
                "mtp_depth": 0,
                "completion_tokens": 2,
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 2,
            "finish_reason": "stop",
            "_final_state": None,
        }

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fail_retokenized)
    monkeypatch.setattr(openai, "_schedule_idle_postcommit_snapshot", fake_schedule)
    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    with TestClient(create_app(state)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "x-mtplx-session-id": "ar-session",
                "x-mtplx-allow-client-controls": "1",
            },
            json={
                "messages": [{"role": "user", "content": "Say AR"}],
                "enable_thinking": False,
                "generation_mode": "ar",
                "stream": True,
                "max_tokens": 4,
            },
        )

    assert response.status_code == 200
    assert scheduled
    assert scheduled[0]["unsafe_reason"] == "missing_generation_final_state"
    assert '"mode": "async_pending"' in response.text
    assert '"reason": "missing_generation_final_state"' in response.text
    assert '"generation_mode": "ar"' in response.text


def test_streaming_ar_honors_explicit_inline_postcommit_mode(monkeypatch):
    state = _fake_streaming_session_state()
    state.args.session_postcommit_mode = "inline"
    retokenized_calls: list[dict] = []

    def fake_retokenized(*_args, **kwargs):
        retokenized_calls.append(kwargs)
        return {
            "stored": True,
            "mode": "retokenized_history",
            "prefix_len": 32,
            "nbytes": 99,
        }

    def fake_run_generation(_state, prompt_ids, **kwargs):
        token_callback = kwargs.get("token_callback")
        tokens = [ord("A"), ord("R")]
        if token_callback is not None:
            token_callback(tokens)
        return {
            "text": "AR",
            "tokens": tokens,
            "stats": {
                "generation_mode": "ar",
                "mtp_depth": 0,
                "completion_tokens": 2,
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 2,
            "finish_reason": "stop",
            "_final_state": None,
        }

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fake_retokenized)
    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    with TestClient(create_app(state)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "x-mtplx-session-id": "ar-inline-session",
                "x-mtplx-allow-client-controls": "1",
            },
            json={
                "messages": [{"role": "user", "content": "Say AR"}],
                "enable_thinking": False,
                "generation_mode": "ar",
                "stream": True,
                "max_tokens": 4,
            },
        )

    assert response.status_code == 200
    assert retokenized_calls
    assert '"mode": "retokenized_history"' in response.text
    assert '"generation_mode": "ar"' in response.text


def test_invalid_generation_mode_returns_400():
    client = TestClient(create_app(_fake_state()))

    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-mtplx-cache-mode": "bypass",
            "x-mtplx-allow-client-controls": "1",
        },
        json={
            "messages": [{"role": "user", "content": "Say READY"}],
            "generation_mode": "off",
        },
    )

    assert response.status_code == 400
    assert (
        response.json()["error"]["message"] == "generation_mode must be 'mtp' or 'ar'"
    )
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_completion_request_controls_are_server_owned_without_override(monkeypatch):
    captured: dict[str, object] = {}
    client = TestClient(create_app(_fake_state()))

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        captured.update(kwargs)
        return {
            "text": "ok",
            "tokens": [4],
            "stats": {
                **kwargs["request_observability"],
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": 1,
            },
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/completions",
        json={
            "prompt": [1, 2, 3],
            "max_tokens": 4,
            "generation_mode": "ar",
            "depth": 1,
            "temperature": 0.01,
            "top_p": 0.2,
            "top_k": 1,
        },
    )

    assert response.status_code == 200
    assert captured["generation_mode"] == "mtp"
    assert captured["depth"] == 3
    assert captured["temperature"] is None
    assert captured["top_p"] is None
    assert captured["top_k"] is None
    stats = captured["request_observability"]
    assert stats["mtplx_control_owner"] == "server"
    assert stats["client_controls_allowed"] is False
    assert stats["client_control_fields_ignored"] == [
        "temperature",
        "top_p",
        "top_k",
        "generation_mode",
        "draft_control",
    ]


def test_chat_accepts_max_completion_tokens_alias_and_benign_extras(monkeypatch):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    client = TestClient(create_app(state))
    seen_max_tokens: list[int | None] = []
    seen_observability: list[dict[str, object]] = []

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        seen_max_tokens.append(kwargs.get("max_tokens"))
        seen_observability.append(dict(kwargs.get("request_observability") or {}))
        return _fake_generation("ok")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "user-agent": "AndroidStudio"},
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 7,
            "stream_options": {"include_usage": True},
            "response_format": {"type": "text"},
            "parallel_tool_calls": False,
            "metadata": {"client": "android-studio"},
            "user": "local-user",
        },
    )

    assert response.status_code == 200
    assert seen_max_tokens == [7]


def test_android_studio_issue58_replay_fixture_is_accepted(monkeypatch):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    client = TestClient(create_app(state))
    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures" / "android_studio_issue58_chat.json"
        ).read_text(encoding="utf-8")
    )
    fixture["stream"] = False
    seen_max_tokens: list[int | None] = []

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        seen_max_tokens.append(kwargs.get("max_tokens"))
        return _fake_generation("ok")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "user-agent": "AndroidStudio"},
        json=fixture,
    )

    assert response.status_code == 200
    assert seen_max_tokens == [64]


class CaptureTokenizer:
    def __init__(self):
        self.calls: list[tuple[list[dict[str, object]], dict[str, object]]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [1, 2, 3]

    def encode(self, text):
        return [ord(char) for char in str(text)]

    def decode(self, tokens, **_kwargs):
        return "".join(chr(int(token)) for token in tokens)


class StepTemplateIgnoringThinkingTokenizer(CaptureTokenizer):
    """Step-like template fixture that always opens <think> on generation."""

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        rendered = "<｜begin▁of▁sentence｜>"
        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort:
            rendered += f"<|im_start|>system\nReasoning: {reasoning_effort}\n\n<|im_end|>\n"
        for message in messages:
            role = str(message.get("role") or "user")
            content = str(message.get("content") or "")
            rendered += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        if kwargs.get("add_generation_prompt", True):
            rendered += "<|im_start|>assistant\n<think>\n"
        if kwargs.get("tokenize", True):
            return [ord(char) for char in rendered]
        return rendered


class ToolSchemaRejectingTokenizer(CaptureTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if "tools" in kwargs:
            raise RuntimeError("rich tool schemas are unsupported")
        return [1, 2, 3]


class QwenToolHistoryBoundaryTokenizer:
    """Tiny Qwen-like tokenizer that exposes the OpenCode cache-boundary bug.

    It deliberately merges ``\n\n</think>`` when a full rendered transcript is
    tokenized at once. Segmenting after ``<think>\n`` preserves the same token
    boundary produced by the previous generation prompt.
    """

    merged = 900_001

    def apply_chat_template(self, messages, **kwargs):
        rendered = self._render(
            messages, add_generation_prompt=bool(kwargs.get("add_generation_prompt"))
        )
        if kwargs.get("tokenize"):
            return self.encode(rendered, add_special_tokens=False)
        return rendered

    def encode(self, text, **_kwargs):
        text = str(text)
        needle = "\n\n</think>"
        ids: list[int] = []
        i = 0
        while i < len(text):
            if text.startswith(needle, i):
                ids.append(self.merged)
                i += len(needle)
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def decode(self, tokens, **_kwargs):
        parts: list[str] = []
        for token in tokens:
            parts.append(
                "\n\n</think>" if int(token) == self.merged else chr(int(token))
            )
        return "".join(parts)

    def _render(self, messages, *, add_generation_prompt: bool) -> str:
        chunks: list[str] = []
        for message in messages:
            role = message["role"]
            content = str(message.get("content") or "")
            if role == "system":
                chunks.append(f"<|im_start|>system\n{content}<|im_end|>\n")
            elif role == "user":
                chunks.append(f"<|im_start|>user\n{content}<|im_end|>\n")
            elif role == "assistant":
                chunks.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
                if content:
                    chunks.append(content + "\n\n")
                for tool_call in message.get("tool_calls") or []:
                    function = tool_call["function"]
                    chunks.append(f"<tool_call>\n<function={function['name']}>\n")
                    for key, value in function.get("arguments", {}).items():
                        chunks.append(f"<parameter={key}>\n{value}\n</parameter>\n")
                    chunks.append("</function>\n</tool_call>")
                chunks.append("<|im_end|>\n")
            elif role == "tool":
                chunks.append(
                    f"<|im_start|>user\n<tool_response>\n{content}\n"
                    "</tool_response><|im_end|>\n"
                )
        if add_generation_prompt:
            chunks.append("<|im_start|>assistant\n<think>\n")
        return "".join(chunks)


def _tool_schema():
    return {
        "type": "function",
        "function": {
            "name": "session_status",
            "description": "Show the current agent session status.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }


def _named_tool_schema(name: str):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }


def _add_tool_schema():
    return {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    }


def _task_tool_schema():
    return {
        "type": "function",
        "function": {
            "name": "Task",
            "description": "Launch a subagent task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                "required": ["description", "prompt"],
                "additionalProperties": False,
            },
        },
    }


def _write_tool_schema():
    return {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "content": {"type": "string"},
                    "createDirs": {"type": "boolean"},
                },
                "required": ["filePath", "content"],
                "additionalProperties": False,
            },
        },
    }


def _bash_tool_schema():
    return {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "description": {"type": "string"},
                    "timeout": {"type": "number"},
                },
                "required": ["command", "description"],
                "additionalProperties": False,
            },
        },
    }


def _todowrite_tool_schema():
    return {
        "type": "function",
        "function": {
            "name": "todowrite",
            "description": "Update the task plan.",
            "parameters": {
                "type": "object",
                "properties": {"todos": {"type": "array"}},
                "required": ["todos"],
                "additionalProperties": False,
            },
        },
    }


def _question_tool_schema():
    return {
        "type": "function",
        "function": {
            "name": "question",
            "description": "Ask the user one or more questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {"type": "array"},
                },
                "required": ["questions"],
                "additionalProperties": False,
            },
        },
    }


def _fake_generation(text: str):
    return {
        "text": text,
        "tokens": [4],
        "stats": {
            "generation_mode": "ar",
            "mtp_depth": 0,
            "completion_tokens": 1,
        },
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "finish_reason": "stop",
    }


def _fake_streaming_generation(text: str, *, finish_reason: str | None = "stop"):
    tokens = [ord(char) for char in text]

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        token_callback = kwargs.get("token_callback")
        if token_callback is not None:
            for token in tokens:
                token_callback([token])
        return {
            "text": text,
            "tokens": tokens,
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": len(tokens),
            },
            "prompt_tokens": 3,
            "completion_tokens": len(tokens),
            "finish_reason": finish_reason,
        }

    return fake_run_generation


def _stream_payloads(response_text: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in response_text.splitlines()
        if line.startswith("data: {")
    ]


def _anthropic_events(response_text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for frame in response_text.split("\n\n"):
        if not frame.strip():
            continue
        event = None
        data = None
        for line in frame.splitlines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))
        if event and data is not None:
            events.append((event, data))
    return events


def test_chat_tools_are_passed_to_qwen_template_and_inherit_default_thinking(
    monkeypatch,
):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stats_footer = False
    state.args.reasoning_parser = "gemma4"
    client = TestClient(create_app(state))
    seen: dict[str, object] = {}

    def fake_run_generation(*_args, **kwargs):
        seen["request_observability"] = dict(kwargs["request_observability"])
        return _fake_generation("ok")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Use the tool."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    messages, kwargs = state.runtime.tokenizer.calls[0]
    rendered = "\n".join(str(message.get("content") or "") for message in messages)
    assert "MTPLX tool contract:" in rendered
    assert "emit one declared <tool_call> now" not in rendered
    assert kwargs["tools"] == [_tool_schema()]
    assert kwargs["enable_thinking"] is True
    stats = seen["request_observability"]
    assert stats["request_reasoning_mode"] == "auto"
    assert stats["request_enable_thinking"] is True
    assert stats["request_enable_thinking_override"] is False
    assert stats["request_reasoning_parser"] == "qwen3"


def test_chat_tools_hide_task_when_latest_user_disallows_subagents(monkeypatch):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(
        openai, "_run_generation", lambda *_args, **_kwargs: _fake_generation("ok")
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [
                {"role": "user", "content": "Make the change. No subagents."}
            ],
            "tools": [_tool_schema(), _task_tool_schema(), _todowrite_tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    _messages, kwargs = state.runtime.tokenizer.calls[0]
    tool_names = [
        tool["function"]["name"]
        for tool in kwargs["tools"]
        if isinstance(tool.get("function"), dict)
    ]
    assert tool_names == ["session_status"]


def test_chat_tools_hide_task_by_default_for_direct_project_work(monkeypatch):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(
        openai, "_run_generation", lambda *_args, **_kwargs: _fake_generation("ok")
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Read the current files and make the requested edit.",
                }
            ],
            "tools": [_tool_schema(), _task_tool_schema(), _todowrite_tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    _messages, kwargs = state.runtime.tokenizer.calls[0]
    tool_names = [
        tool["function"]["name"]
        for tool in kwargs["tools"]
        if isinstance(tool.get("function"), dict)
    ]
    assert tool_names == ["session_status"]


def test_chat_tools_report_filtered_task_names_for_direct_project_work(monkeypatch):
    seen: dict[str, object] = {}
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stats_footer = True
    client = TestClient(create_app(state))

    def fake_run_generation(*_args, **kwargs):
        seen["request_observability"] = dict(kwargs["request_observability"])
        return _fake_generation("ok")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Read the current files and make the requested edit.",
                }
            ],
            "tools": [_tool_schema(), _task_tool_schema(), _todowrite_tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    stats = seen["request_observability"]
    assert stats["request_tool_names"] == ["session_status", "Task", "todowrite"]
    assert stats["request_filtered_tool_names"] == ["session_status"]
    assert stats["request_filtered_tool_count"] == 1
    assert stats["request_hidden_tool_names"] == ["Task", "todowrite"]
    assert stats["request_tools_hidden_by_bridge"] is True


def test_chat_tools_report_no_edit_mutating_tools_hidden(monkeypatch):
    """Generic clients (no coding-agent hint) keep the content-heuristic
    lockdown. OpenCode clients are exempt — see the pass-through test below:
    they curate the toolset per agent mode themselves, and bridge-side hiding
    both broke banked-prefix bytes and starved build mode of write/edit
    (2026-07-04)."""
    seen: dict[str, object] = {}
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stats_footer = True
    client = TestClient(create_app(state))

    def fake_run_generation(*_args, **kwargs):
        seen["request_observability"] = dict(kwargs["request_observability"])
        return _fake_generation("ok")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Do not edit files. Run pwd, then read package.json and "
                        "answer with the scripts."
                    ),
                }
            ],
            "tools": [
                _bash_tool_schema(),
                _write_tool_schema(),
                _named_tool_schema("read"),
                _named_tool_schema("edit"),
                _todowrite_tool_schema(),
            ],
            "tool_choice": "auto",
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    stats = seen["request_observability"]
    assert stats["request_tool_names"] == ["bash", "write", "read", "edit", "todowrite"]
    assert stats["request_filtered_tool_names"] == ["bash", "read"]
    assert stats["request_hidden_tool_names"] == ["write", "edit", "todowrite"]
    assert stats["request_tools_hidden_by_bridge"] is True


def test_chat_tools_opencode_client_toolset_passes_through(monkeypatch):
    seen: dict[str, object] = {}
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stats_footer = True
    client = TestClient(create_app(state))

    def fake_run_generation(*_args, **kwargs):
        seen["request_observability"] = dict(kwargs["request_observability"])
        return _fake_generation("ok")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "opencode"},
        json={
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Do not edit files. Run pwd, then read package.json and "
                        "answer with the scripts."
                    ),
                }
            ],
            "tools": [
                _bash_tool_schema(),
                _write_tool_schema(),
                _named_tool_schema("read"),
                _named_tool_schema("edit"),
                _todowrite_tool_schema(),
            ],
            "tool_choice": "auto",
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    stats = seen["request_observability"]
    assert stats["request_filtered_tool_names"] == [
        "bash",
        "write",
        "read",
        "edit",
        "todowrite",
    ]
    assert stats["request_hidden_tool_names"] == []
    assert stats["request_tools_hidden_by_bridge"] is False
    assert stats["tool_prompt_mode"] == "compact"


def test_chat_tools_keep_task_when_latest_user_explicitly_requests_subagent(monkeypatch):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(
        openai, "_run_generation", lambda *_args, **_kwargs: _fake_generation("ok")
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Use a subagent to explore the project before editing.",
                }
            ],
            "tools": [_tool_schema(), _task_tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    _messages, kwargs = state.runtime.tokenizer.calls[0]
    tool_names = [
        tool["function"]["name"]
        for tool in kwargs["tools"]
        if isinstance(tool.get("function"), dict)
    ]
    assert tool_names == ["session_status", "Task"]


def test_chat_tools_keep_todowrite_when_latest_user_explicitly_requests_plan(monkeypatch):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(
        openai, "_run_generation", lambda *_args, **_kwargs: _fake_generation("ok")
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Write a plan and track todos before editing.",
                }
            ],
            "tools": [_tool_schema(), _task_tool_schema(), _todowrite_tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    _messages, kwargs = state.runtime.tokenizer.calls[0]
    tool_names = [
        tool["function"]["name"]
        for tool in kwargs["tools"]
        if isinstance(tool.get("function"), dict)
    ]
    assert tool_names == ["session_status", "todowrite"]


def test_tool_stream_suppresses_orphan_control_closing_tags():
    translator = openai._ToolAwareContentStreamTranslator(
        tools=[_tool_schema()],
        argument_chunk_chars=8,
    )

    deltas = []
    for chunk in ["\n</parameter>\n", "</function>\n", "Visible answer"]:
        deltas.extend(translator.feed("content", chunk))
    deltas.extend(translator.finish())

    content = "".join(delta.get("content", "") for delta in deltas)
    assert "</parameter>" not in content
    assert "</function>" not in content
    assert content.strip() == "Visible answer"
    assert translator.suppressed_tool_markup is True


def test_visible_malformed_tool_content_drops_orphan_tool_tags():
    visible = openai._visible_malformed_tool_content(
        "\n</parameter>\n</function>\nkeep this\n</tool_call>",
        tokenizer=None,
    )

    assert "</parameter>" not in visible
    assert "</function>" not in visible
    assert "</tool_call>" not in visible
    assert visible.strip() == "keep this"


def test_tool_fed_degenerate_completion_detects_bare_orphan_tool_tail():
    text = "parameter=limit>\n180\n</parameter>\n</function>\n</tool_call>"

    assert (
        openai._tool_fed_degenerate_completion_reason(text)
        == "orphan_tool_control_markup"
    )
    stripped = openai._strip_orphan_tool_control_markup(text)
    assert "parameter=limit>" not in stripped
    assert "</parameter>" not in stripped


def test_visible_malformed_tool_content_drops_orphan_reasoning_tags():
    visible = openai._visible_malformed_tool_content(
        "This is visible.\n</thinking>\n<reasoning>still visible</reasoning>",
        tokenizer=None,
    )

    assert "</thinking>" not in visible
    assert "<reasoning>" not in visible
    assert "</reasoning>" not in visible
    assert visible.strip() == "This is visible.\n\nstill visible"


def test_visible_malformed_tool_content_drops_tool_exec_blocks():
    visible = openai._visible_malformed_tool_content(
        (
            "Let me search.\n"
            "<toolExec>\n"
            '<invoke name="search_codebase">\n'
            '<parameter name="query">type CreateChatCompletionResponse</parameter>\n'
            "</invoke>\n"
            "</toolExec>\n"
            "Done."
        ),
        tokenizer=None,
    )

    assert "<toolExec>" not in visible
    assert "<invoke" not in visible
    assert "CreateChatCompletionResponse" not in visible
    assert visible.strip() == "Let me search.\n\nDone."


def test_tool_contract_includes_exact_schema_keys_for_opencode_write(monkeypatch):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(
        openai, "_run_generation", lambda *_args, **_kwargs: _fake_generation("ok")
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Write a file."}],
            "tools": [_write_tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    messages, kwargs = state.runtime.tokenizer.calls[0]
    rendered = "\n".join(str(message.get("content") or "") for message in messages)
    assert "MTPLX tool contract:" in rendered
    assert "exact argument keys/case" in rendered
    assert "Never print file contents" in rendered
    assert "only inside the declared write/edit tool call arguments" in rendered
    assert "emit one declared <tool_call> now" in rendered
    assert "implementation payloads in the declared tool call arguments" in rendered
    assert "let me fix this" in rendered
    assert kwargs["tools"] == [_write_tool_schema()]


def test_opencode_title_request_fast_path_does_not_enter_model(monkeypatch):
    state = _fake_state()
    client = TestClient(create_app(state))

    def fail_generation(*_args, **_kwargs):
        raise AssertionError("OpenCode title request should not enter generation")

    monkeypatch.setattr(openai, "_run_generation", fail_generation)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mtplx-test-model",
            "stream": True,
            "max_tokens": 32000,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a title generator. You output ONLY a thread title. "
                        "Generate a brief title. Never use tools."
                    ),
                },
                {"role": "user", "content": "Generate a title for this conversation:"},
                {"role": "user", "content": "hi"},
            ],
        },
    )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    assert any(
        choice["delta"].get("content") == "Greeting"
        for payload in payloads
        for choice in payload["choices"]
    )
    assert state.last_metrics[-1]["opencode_title_fast_path"] is True
    assert state.last_metrics[-1]["request_max_tokens"] == 32000


def test_stream_cancellation_metric_records_prompt_and_policy_context():
    state = _fake_streaming_session_state()
    state.requests_cancelled = 0

    openai._record_stream_cancellation_metric(
        state,
        response_id="chatcmpl_cancel",
        session_id="ses_cancel",
        prompt_tokens=31798,
        streamed_completion_tokens=0,
        stream_started_s=time.perf_counter() - 2.0,
        reason="client_disconnected",
        request_observability={
            "tool_prompt_mode": "native",
            "tool_contract_active": False,
            "chat_template_profile": "local_qwen36",
        },
        client_disconnected=True,
    )

    latest = state.last_metrics[-1]
    assert latest["request_cancelled"] is True
    assert latest["cancellation_reason"] == "client_disconnected"
    assert latest["prompt_tokens"] == 31798
    assert latest["streamed_completion_tokens"] == 0
    assert latest["decode_tok_s"] == 0.0
    assert latest["request_tok_s"] == 0.0
    assert latest["partial_decode_tok_s"] == 0.0
    assert latest["tool_prompt_mode"] == "native"
    assert latest["tool_contract_active"] is False
    assert state.requests_cancelled == 1


def test_stream_cancellation_metric_keeps_partial_throughput():
    state = _fake_streaming_session_state()

    openai._record_stream_cancellation_metric(
        state,
        response_id="chatcmpl_cancel_partial",
        session_id="ses_cancel",
        prompt_tokens=28276,
        streamed_completion_tokens=120,
        stream_started_s=time.perf_counter() - 4.0,
        reason="stream_cancelled",
        request_observability={},
        client_disconnected=False,
    )

    latest = state.last_metrics[-1]
    assert latest["completion_tokens"] == 120
    assert latest["generated_tokens"] == 120
    assert latest["streamed_completion_tokens"] == 120
    assert latest["request_cancelled"] is True
    assert latest["decode_tok_s"] > 0.0
    assert latest["request_tok_s"] > 0.0
    assert latest["server_tok_s"] > 0.0
    assert latest["partial_decode_tok_s"] == latest["decode_tok_s"]
    assert latest["partial_request_tok_s"] == latest["request_tok_s"]


def test_tool_requests_enable_prompt_prefix_bank_commit(monkeypatch):
    state = _fake_streaming_session_state()
    state.draft_sampler = None
    state.requests_completed = 0
    captured: dict[str, object] = {}

    def fake_generate_mtpk(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            tokens=[],
            text="",
            stats=SimpleNamespace(
                to_dict=lambda: {
                    "prompt_eval_time_s": 0.0,
                    "generated_tokens": 0,
                    "elapsed_s": 0.0,
                    "tok_s": 0.0,
                }
            ),
            final_state=None,
        )

    monkeypatch.setattr(openai, "generate_mtpk", fake_generate_mtpk)

    openai._run_generation(
        state,
        [1, 2, 3],
        max_tokens=1,
        temperature=None,
        top_p=None,
        top_k=None,
        seed=None,
        generation_mode="mtp",
        depth=3,
        session_id="sess-tool",
        session_bank=state.sessions.bank,
        session_template_hash=state.template_hash,
        session_draft_head_identity=state.draft_head_identity,
        session_policy_fingerprint="policy",
        commit_prompt_prefix_to_bank=True,
    )

    assert captured["commit_prompt_state_to_bank"] is True
    assert captured["commit_prompt_state_keep_live_ref"] is False


def test_run_generation_depth1_clamps_expected_value_policy(monkeypatch):
    state = _fake_streaming_session_state()
    state.draft_sampler = None
    state.requests_completed = 0
    state.args.adaptive_policy = "expected_value"
    state.args.adaptive_ev_base_depth = 2
    captured: dict[str, object] = {}

    def fake_generate_mtpk(*_args, **kwargs):
        captured["adaptive_policy"] = kwargs["adaptive_policy"]
        return SimpleNamespace(
            tokens=[ord("O")],
            text="O",
            stats=SimpleNamespace(
                to_dict=lambda: {
                    "prompt_eval_time_s": 0.0,
                    "generated_tokens": 1,
                    "elapsed_s": 0.1,
                    "tok_s": 10.0,
                }
            ),
            final_state=None,
        )

    monkeypatch.setattr(openai, "generate_mtpk", fake_generate_mtpk)

    openai._run_generation(
        state,
        [1, 2, 3],
        max_tokens=1,
        temperature=None,
        top_p=None,
        top_k=None,
        seed=None,
        generation_mode="mtp",
        depth=1,
        resolved_mtp_depth=1,
    )

    policy = captured["adaptive_policy"]
    assert policy.max_depth == 1
    assert policy.min_depth == 1
    assert policy.base_depth == 1


def test_run_generation_uses_effective_depth_but_preserves_requested_depth(monkeypatch):
    state = _fake_streaming_session_state()
    state.draft_sampler = None
    state.requests_completed = 0
    captured: dict[str, object] = {}

    def fake_generate_mtpk(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            tokens=[ord("O")],
            text="O",
            stats=SimpleNamespace(
                to_dict=lambda: {
                    "prompt_eval_time_s": 0.0,
                    "generated_tokens": 1,
                    "elapsed_s": 0.1,
                    "tok_s": 10.0,
                    "speculative_depth": kwargs["speculative_depth"],
                    "requested_speculative_depth": kwargs["speculative_depth"],
                }
            ),
            final_state=None,
        )

    monkeypatch.setattr(openai, "generate_mtpk", fake_generate_mtpk)

    generated = openai._run_generation(
        state,
        [1, 2, 3],
        max_tokens=1,
        temperature=None,
        top_p=None,
        top_k=None,
        seed=None,
        generation_mode="mtp",
        depth=3,
        resolved_mtp_depth=2,
    )

    assert captured["speculative_depth"] == 2
    assert generated["stats"]["mtp_depth"] == 2
    assert generated["stats"]["requested_mtp_depth"] == 3
    assert generated["stats"]["requested_speculative_depth"] == 3


def test_run_generation_can_store_final_state_without_live_cache_ref(monkeypatch):
    state = _fake_streaming_session_state()
    state.draft_sampler = None
    state.requests_completed = 0

    def fake_generate_mtpk(*_args, **_kwargs):
        tokens = [ord("O"), ord("K")]
        return SimpleNamespace(
            tokens=tokens,
            text="OK",
            stats=SimpleNamespace(
                to_dict=lambda: {
                    "prompt_eval_time_s": 0.0,
                    "generated_tokens": 2,
                    "elapsed_s": 0.1,
                    "tok_s": 20.0,
                }
            ),
            final_state=_fake_final_state(tokens),
        )

    monkeypatch.setattr(openai, "generate_mtpk", fake_generate_mtpk)

    openai._run_generation(
        state,
        [1, 2, 3],
        max_tokens=65536,
        temperature=None,
        top_p=None,
        top_k=None,
        seed=42,
        generation_mode="mtp",
        depth=3,
        session_id="anon-bench",
        session_bank=state.sessions.bank,
        session_template_hash=state.template_hash,
        session_draft_head_identity=state.draft_head_identity,
        session_policy_fingerprint="policy",
        session_keep_live_ref=False,
    )

    assert state.sessions.bank.puts
    assert state.sessions.bank.puts[-1]["keep_live_ref"] is False


def test_tool_template_schema_failure_retries_with_compact_contract(monkeypatch):
    state = _fake_state()
    state.runtime.tokenizer = ToolSchemaRejectingTokenizer()
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(
        openai, "_run_generation", lambda *_args, **_kwargs: _fake_generation("ok")
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Write a file."}],
            "tools": [_write_tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    first_messages, first_kwargs = state.runtime.tokenizer.calls[0]
    second_messages, second_kwargs = state.runtime.tokenizer.calls[-1]
    assert "tools" in first_kwargs
    assert "tools" not in second_kwargs
    assert first_messages == second_messages
    assert "MTPLX tool contract:" in second_messages[0]["content"]
    assert "emit one declared <tool_call> now" in second_messages[0]["content"]


def test_chat_tools_honor_explicit_disable_thinking_with_client_opt_in(monkeypatch):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    client = TestClient(create_app(state))
    seen: dict[str, object] = {}

    def fake_run_generation(*_args, **kwargs):
        seen["request_observability"] = dict(kwargs["request_observability"])
        return _fake_generation("ok")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-mtplx-cache-mode": "bypass",
            "x-mtplx-allow-client-controls": "1",
        },
        json={
            "messages": [{"role": "user", "content": "Use the tool."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "enable_thinking": False,
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    _messages, kwargs = state.runtime.tokenizer.calls[0]
    assert kwargs["enable_thinking"] is False
    stats = seen["request_observability"]
    assert stats["request_reasoning_mode"] == "off"
    assert stats["request_enable_thinking"] is False
    assert stats["request_enable_thinking_override"] is True
    assert stats["mtplx_control_owner"] == "client"
    assert stats["client_controls_allowed"] is True


def test_streaming_tool_request_emits_raw_reasoning_by_default(monkeypatch):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation("<think>raw tool reasoning</think>\nanswer"),
    )

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Use the tool."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 32,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    payloads = _stream_payloads(body)
    reasoning = "".join(
        choice.get("delta", {}).get("reasoning_content", "")
        for payload in payloads
        for choice in payload.get("choices", [])
    )
    content = "".join(
        choice.get("delta", {}).get("content", "")
        for payload in payloads
        for choice in payload.get("choices", [])
    )
    assert reasoning == "raw tool reasoning"
    assert content.strip() == "answer"


def test_opencode_simple_chitchat_streams_reasoning_when_app_reasoning_on(monkeypatch):
    state = _fake_streaming_session_state()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    text = "<think>internal greeting plan</think>\nHello! How can I help you today?"
    tokens = [ord(char) for char in text]

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        token_callback = kwargs.get("token_callback")
        if token_callback is not None:
            for token in tokens:
                token_callback([token])
        return {
            "text": text,
            "tokens": tokens,
            "stats": {
                **(kwargs.get("request_observability") or {}),
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": len(tokens),
            },
            "prompt_tokens": 3,
            "completion_tokens": len(tokens),
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "opencode"},
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 64,
            "enable_thinking": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    payloads = _stream_payloads(body)
    reasoning = "".join(
        choice.get("delta", {}).get("reasoning_content", "")
        for payload in payloads
        for choice in payload.get("choices", [])
    )
    content = "".join(
        choice.get("delta", {}).get("content", "")
        for payload in payloads
        for choice in payload.get("choices", [])
    )
    final = [payload for payload in payloads if payload["choices"][0]["finish_reason"]]

    assert reasoning == "internal greeting plan"
    assert content.strip() == "Hello! How can I help you today?"
    assert "visible_reasoning_policy" not in final[-1]["mtplx_stats"]
    # Chitchat keeps client tools (band-aid removal, 2026-06-09).
    assert final[-1]["mtplx_stats"]["request_tools_hidden_by_bridge"] is False
    assert final[-1]["mtplx_stats"]["opencode_prompt_contract_profile"] == "opencode_agent"


def test_opencode_simple_chitchat_does_not_retry_or_cook_a_reply(monkeypatch):
    state = _fake_streaming_session_state()
    state.args.reasoning = "off"
    state.args.enable_thinking = False
    state.args.stream_interval = 1
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    texts = [
        "The",
        "</think>\n\nHello! How can I help you today?",
    ]
    calls: list[str] = []

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        text = texts[len(calls)]
        calls.append(text)
        tokens = [ord(char) for char in text]
        token_callback = kwargs.get("token_callback")
        if token_callback is not None:
            for token in tokens:
                token_callback([token])
        return {
            "text": text,
            "tokens": tokens,
            "stats": {
                **(kwargs.get("request_observability") or {}),
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": len(tokens),
                "decode_tok_s": 42.0,
            },
            "prompt_tokens": 3,
            "completion_tokens": len(tokens),
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "opencode"},
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 64,
            "enable_thinking": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    payloads = _stream_payloads(body)
    reasoning = "".join(
        choice.get("delta", {}).get("reasoning_content", "")
        for payload in payloads
        for choice in payload.get("choices", [])
    )
    content = "".join(
        choice.get("delta", {}).get("content", "")
        for payload in payloads
        for choice in payload.get("choices", [])
    )
    final = [payload for payload in payloads if payload["choices"][0]["finish_reason"]]

    assert calls == ["The"]
    assert reasoning == ""
    assert content.strip() == "The"
    assert "simple_chitchat_retry_attempted" not in final[-1]["mtplx_stats"]
    assert final[-1]["mtplx_stats"]["request_reasoning_mode"] == "off"
    # Chitchat keeps client tools (band-aid removal, 2026-06-09).
    assert final[-1]["mtplx_stats"]["request_tools_hidden_by_bridge"] is False
    assert final[-1]["mtplx_stats"]["opencode_prompt_contract_profile"] == "opencode_agent"


def test_step_chat_request_encodes_language_policy_without_replacing_user_turn(monkeypatch):
    captured: dict[str, object] = {}
    state = _fake_streaming_session_state()
    state.backend_descriptor = openai.descriptor_for_backend_id("step3p5_mtp")
    state.args.reasoning_parser = "step3p5"
    state.args.stats_footer = False
    client = TestClient(create_app(state))

    def fake_run_generation(_state, prompt_ids, **kwargs):
        captured["prompt_text"] = state.runtime.tokenizer.decode(prompt_ids)
        captured["request_observability"] = dict(kwargs["request_observability"])
        return _fake_generation("hello")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    prompt_text = captured["prompt_text"]
    assert "system:MTPLX Step language policy:" in prompt_text
    assert "Use English by default" in prompt_text
    assert "Never answer in Chinese for English or ambiguous input." in prompt_text
    assert "user:hi" in prompt_text
    assert captured["request_observability"]["backend_chat_policy_active"] is True


def test_step_reasoning_off_closes_template_think_prompt_for_managed_clients(
    monkeypatch,
):
    captured: dict[str, object] = {}
    state = _fake_streaming_session_state()
    state.backend_descriptor = openai.descriptor_for_backend_id("step3p5_mtp")
    state.args.reasoning = "off"
    state.args.enable_thinking = False
    state.args.reasoning_parser = "step3p5"
    state.args.reasoning_effort = "low"
    state.args.stats_footer = False
    state.runtime.tokenizer = StepTemplateIgnoringThinkingTokenizer()
    client = TestClient(create_app(state))

    def fake_run_generation(_state, prompt_ids, **kwargs):
        captured["prompt_text"] = state.runtime.tokenizer.decode(prompt_ids)
        captured["request_observability"] = dict(kwargs["request_observability"])
        return _fake_generation("hello")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-mtplx-cache-mode": "bypass",
            "x-mtplx-client": "opencode",
            "x-mtplx-allow-client-controls": "1",
        },
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "max_tokens": 16,
            "enable_thinking": True,
            "reasoning_effort": "high",
        },
    )

    assert response.status_code == 200
    assert captured["prompt_text"].endswith(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert not captured["prompt_text"].endswith("<think>\n")
    assert "Reasoning: high" not in captured["prompt_text"]
    stats = captured["request_observability"]
    assert stats["mtplx_control_owner"] == "server"
    assert stats["client_controls_allowed"] is False
    assert stats["request_enable_thinking"] is False
    assert stats["request_reasoning_mode"] == "off"
    assert stats["request_enable_thinking_override"] is False
    assert stats["client_control_fields_ignored"] == [
        "enable_thinking",
        "reasoning_effort",
    ]
    assert stats["disabled_thinking_prompt_closed"] is True


def test_step_reasoning_off_strips_orphan_thinks_close_nonstream(monkeypatch):
    state = _fake_streaming_session_state()
    state.backend_descriptor = openai.descriptor_for_backend_id("step3p5_mtp")
    state.args.reasoning = "off"
    state.args.enable_thinking = False
    state.args.reasoning_parser = "step3p5"
    state.args.stats_footer = False
    client = TestClient(create_app(state))

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        generated = _fake_generation(
            "I'm doing well, thank you for asking. </thinks> "
            "I'm doing well, thank you for asking."
        )
        generated["stats"].update(kwargs["request_observability"])
        return generated

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Hi, how are you?"}],
            "stream": False,
            "max_tokens": 32,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    message = payload["choices"][0]["message"]
    assert message["content"] == "I'm doing well, thank you for asking."
    assert message.get("reasoning_content") in (None, "")
    assert "</think" not in message["content"]
    assert "</thinks" not in message["content"]
    stats = payload["mtplx_stats"]
    assert stats["request_reasoning_mode"] == "off"
    assert stats["request_enable_thinking"] is False
    assert stats["visible_reasoning_stripped"] is True


def test_step_reasoning_off_strips_orphan_thinks_close_stream(monkeypatch):
    state = _fake_streaming_session_state()
    state.backend_descriptor = openai.descriptor_for_backend_id("step3p5_mtp")
    state.args.reasoning = "off"
    state.args.enable_thinking = False
    state.args.reasoning_parser = "step3p5"
    state.args.stream_interval = 1
    state.args.stats_footer = False
    client = TestClient(create_app(state))

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        text = (
            "I'm doing well, thank you for asking. </thinks> "
            "I'm doing well, thank you for asking."
        )
        token_callback = kwargs.get("token_callback")
        tokens = [ord(char) for char in text]
        if token_callback is not None:
            for token in tokens:
                token_callback([token])
        return {
            "text": text,
            "tokens": tokens,
            "stats": {
                **kwargs["request_observability"],
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": len(tokens),
            },
            "prompt_tokens": 3,
            "completion_tokens": len(tokens),
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Hi, how are you?"}],
            "stream": True,
            "max_tokens": 32,
        },
    )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    content = "".join(
        payload["choices"][0].get("delta", {}).get("content", "")
        for payload in payloads
    )
    reasoning = "".join(
        payload["choices"][0].get("delta", {}).get("reasoning_content", "")
        for payload in payloads
    )

    assert content.strip() == "I'm doing well, thank you for asking."
    assert reasoning == ""
    assert "</think" not in content
    assert "</thinks" not in content
    final = [payload for payload in payloads if payload["choices"][0]["finish_reason"]]
    assert final[-1]["mtplx_stats"]["request_reasoning_mode"] == "off"
    assert final[-1]["mtplx_stats"]["request_enable_thinking"] is False


def test_settings_reasoning_mode_is_used_by_next_qwen_request(monkeypatch):
    captured: dict[str, object] = {}
    state = _fake_state(api_key="mtplx-local")
    state.runtime.tokenizer = CaptureTokenizer()
    client = TestClient(create_app(state))

    settings = client.post(
        "/v1/mtplx/settings",
        json={"reasoning": "off"},
        headers={"Authorization": "Bearer mtplx-local"},
    )
    assert settings.status_code == 200

    def fake_run_generation(*_args, **kwargs):
        captured["request_observability"] = dict(kwargs["request_observability"])
        return _fake_generation("ok")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer mtplx-local",
            "x-mtplx-cache-mode": "bypass",
        },
        json={
            "messages": [{"role": "user", "content": "Do not think."}],
            "stream": False,
            "max_tokens": 16,
            "enable_thinking": True,
        },
    )

    assert response.status_code == 200
    _messages, kwargs = state.runtime.tokenizer.calls[0]
    assert kwargs["enable_thinking"] is False
    assert captured["request_observability"]["request_reasoning_mode"] == "off"
    assert captured["request_observability"]["request_enable_thinking"] is False
    assert captured["request_observability"]["client_control_fields_ignored"] == [
        "enable_thinking"
    ]


def test_pi_tool_result_empty_template_sentinel_retries_final_answer(monkeypatch):
    state = _fake_streaming_session_state()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    texts = [
        "<|third_empty|>",
        "</think>\n\nImplemented the best-score overlay and verified routes.",
    ]
    calls: list[str] = []

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        text = texts[len(calls)]
        calls.append(text)
        tokens = [ord(char) for char in text]
        token_callback = kwargs.get("token_callback")
        if token_callback is not None:
            for token in tokens:
                token_callback([token])
        return {
            "text": text,
            "tokens": tokens,
            "stats": {
                **(kwargs.get("request_observability") or {}),
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": len(tokens),
                "decode_tok_s": 21.0,
            },
            "prompt_tokens": len(_prompt_ids),
            "completion_tokens": len(tokens),
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "pi"},
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Improve this project after reading the files.",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_read",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": "{\"filePath\":\"package.json\"}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_read",
                    "content": "{\"scripts\":{\"dev\":\"vite\"}}",
                },
            ],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 128,
            "enable_thinking": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    payloads = _stream_payloads(body)
    content = "".join(
        choice.get("delta", {}).get("content", "")
        for payload in payloads
        for choice in payload.get("choices", [])
    )
    final = [payload for payload in payloads if payload["choices"][0]["finish_reason"]]

    assert calls == texts
    assert "<|third_empty|>" not in body
    assert content.strip() == (
        "Implemented the best-score overlay and verified routes."
    )
    assert final[-1]["choices"][0]["finish_reason"] == "stop"
    assert final[-1]["mtplx_stats"]["tool_fed_empty_retry_attempted"] is True
    assert final[-1]["mtplx_stats"]["tool_fed_empty_retry_succeeded"] is True
    assert final[-1]["mtplx_stats"]["tool_fed_empty_retry_reason"] == (
        "empty_tool_fed_completion"
    )


def test_pi_tool_result_orphan_tool_tail_retries_without_stream_leak(monkeypatch):
    state = _fake_streaming_session_state()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    texts = [
        "parameter=limit>\n180\n</parameter>\n</function>\n</tool_call>",
        "</think>\n\nImplemented the HUD cleanup and verified npm build.",
    ]
    calls: list[str] = []

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        text = texts[len(calls)]
        calls.append(text)
        tokens = [ord(char) for char in text]
        token_callback = kwargs.get("token_callback")
        if token_callback is not None:
            for token in tokens:
                token_callback([token])
        return {
            "text": text,
            "tokens": tokens,
            "stats": {
                **(kwargs.get("request_observability") or {}),
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": len(tokens),
                "decode_tok_s": 22.0,
            },
            "prompt_tokens": len(_prompt_ids),
            "completion_tokens": len(tokens),
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "pi"},
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Improve this project after reading the files.",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_read",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": "{\"filePath\":\"src/Game.ts\"}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_read",
                    "content": "{\"content\":\"export const score = 0\"}",
                },
            ],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 128,
            "enable_thinking": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    payloads = _stream_payloads(body)
    content = "".join(
        choice.get("delta", {}).get("content", "")
        for payload in payloads
        for choice in payload.get("choices", [])
    )
    reasoning = "".join(
        choice.get("delta", {}).get("reasoning_content", "")
        for payload in payloads
        for choice in payload.get("choices", [])
    )
    final = [payload for payload in payloads if payload["choices"][0]["finish_reason"]]

    assert calls == texts
    assert "parameter=limit>" not in body
    assert "</parameter>" not in body
    assert "\n180\n" not in body
    assert reasoning == ""
    assert content.strip() == "Implemented the HUD cleanup and verified npm build."
    assert final[-1]["choices"][0]["finish_reason"] == "stop"
    assert final[-1]["mtplx_stats"]["tool_fed_empty_retry_attempted"] is True
    assert final[-1]["mtplx_stats"]["tool_fed_empty_retry_succeeded"] is True
    assert final[-1]["mtplx_stats"]["tool_fed_empty_retry_reason"] == (
        "orphan_tool_control_markup"
    )
    assert final[-1]["mtplx_stats"]["raw_tool_markup_suppressed"] is True


def test_pi_tool_result_reasoning_only_final_turn_repairs_without_visible_leak(
    monkeypatch,
):
    state = _fake_streaming_session_state()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    texts = [
        "The user wants a concise launch-readiness summary. Let me compose it now.",
        "Launch-readiness summary:\n- Strongest part: focused game loop.\n- Checks: npm build passed.",
    ]
    calls: list[str] = []
    prompts: list[str] = []

    def fake_run_generation(_state, prompt_ids, **kwargs):
        text = texts[len(calls)]
        calls.append(text)
        prompts.append(state.runtime.tokenizer.decode(prompt_ids))
        tokens = [ord(char) for char in text]
        token_callback = kwargs.get("token_callback")
        if token_callback is not None:
            for token in tokens:
                token_callback([token])
        return {
            "text": text,
            "tokens": tokens,
            "stats": {
                **(kwargs.get("request_observability") or {}),
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": len(tokens),
                "decode_tok_s": 24.0,
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(tokens),
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "pi"},
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Inspect this project deeply and summarize launch readiness.",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_read",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": "{\"filePath\":\"src/Game.ts\"}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_read",
                    "content": "{\"content\":\"export class Game {}\"}",
                },
            ],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 128,
            "enable_thinking": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    payloads = _stream_payloads(body)
    reasoning = "".join(
        choice.get("delta", {}).get("reasoning_content", "")
        for payload in payloads
        for choice in payload.get("choices", [])
    )
    content = "".join(
        choice.get("delta", {}).get("content", "")
        for payload in payloads
        for choice in payload.get("choices", [])
    )
    final = [payload for payload in payloads if payload["choices"][0]["finish_reason"]]

    assert calls == texts
    assert "The user wants a concise launch-readiness summary" in reasoning
    assert "The user wants a concise launch-readiness summary" not in content
    assert content.strip() == texts[1]
    assert "</think>" in prompts[1]
    assert final[-1]["choices"][0]["finish_reason"] == "stop"
    assert final[-1]["mtplx_stats"]["reasoning_completion_repair_attempted"] is True
    assert final[-1]["mtplx_stats"]["reasoning_completion_repair_succeeded"] is True
    assert final[-1]["mtplx_stats"]["reasoning_completion_repair_reason"] == (
        "tool_fed_reasoning_only_completion"
    )


def test_streaming_tool_call_finishes_without_waiting_for_model_eos(monkeypatch):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation(
            "<tool_call>\n"
            "<function=session_status>\n"
            "</function>\n"
            "</tool_call>"
            " trailing text that should not stream"
        ),
    )

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Check status."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 256,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    payloads = _stream_payloads(body)
    deltas = [
        choice.get("delta", {})
        for payload in payloads
        for choice in payload.get("choices", [])
    ]
    assert any(delta.get("tool_calls") for delta in deltas)
    assert "trailing text" not in body
    final = [payload for payload in payloads if payload["choices"][0]["finish_reason"]]
    assert final[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert final[-1]["mtplx_stats"]["early_tool_cancel_used"] is True
    assert final[-1]["mtplx_stats"]["tool_parser_source"] == "streaming_translator"


def test_streaming_tool_call_canonicalizes_shell_alias_to_bash(monkeypatch):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation(
            "<tool_call>\n"
            "<function=Shell>\n"
            "<parameter=command>\npwd\n</parameter>\n"
            "<parameter=description>\nPrint working directory\n</parameter>\n"
            "</function>\n"
            "</tool_call>"
        ),
    )

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "opencode"},
        json={
            "messages": [{"role": "user", "content": "Run pwd."}],
            "tools": [_bash_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 256,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    payloads = _stream_payloads(body)
    deltas = [
        choice.get("delta", {})
        for payload in payloads
        for choice in payload.get("choices", [])
    ]
    names = [
        item.get("function", {}).get("name")
        for delta in deltas
        for item in delta.get("tool_calls", [])
        if item.get("function", {}).get("name")
    ]
    assert names == ["bash"]
    assert "Shell" not in names
    final = [payload for payload in payloads if payload["choices"][0]["finish_reason"]]
    assert final[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_streaming_unclosed_tool_call_errors_instead_of_hidden_runaway(monkeypatch):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "STREAM_HIDDEN_TOOL_GUARD_TOKENS", 4)
    monkeypatch.setattr(openai, "STREAM_HIDDEN_TOOL_GUARD_S", 0.0)
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation("<tool_call>\n<function=session_status>\n" + "x" * 32),
    )

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Check status."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 256,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "malformed tool_call: unterminated stream" in body
    assert "data: [DONE]" in body


def test_streaming_long_content_first_write_survives_hidden_tool_guard(monkeypatch):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "STREAM_HIDDEN_TOOL_GUARD_TOKENS", 80)
    monkeypatch.setattr(openai, "STREAM_HIDDEN_TOOL_GUARD_S", 0.0)
    long_content = "\n".join(["<!DOCTYPE html>", "<html>", "<body>", "x" * 80, "</body>", "</html>"] * 24)
    text = (
        "<tool_call>\n<function=write>\n"
        f"<parameter=content>\n{long_content}\n</parameter>\n"
        "<parameter=filePath>\n/Users/youssof/Documents/Flappy Bird 3D optimization/index.html\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    monkeypatch.setattr(openai, "_run_generation", _fake_streaming_generation(text))

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Write index.html."}],
            "tools": [_write_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 4096,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "malformed tool_call: unterminated stream" not in body
    assert "<tool_call>" not in body
    payloads = _stream_payloads(body)
    deltas = [
        choice.get("delta", {})
        for payload in payloads
        for choice in payload.get("choices", [])
    ]
    assert any(delta.get("tool_calls") for delta in deltas)
    arguments = "".join(
        item.get("function", {}).get("arguments", "")
        for delta in deltas
        for item in delta.get("tool_calls", [])
    )
    parsed = json.loads(arguments)
    assert parsed["content"] == long_content
    assert parsed["filePath"].endswith("Flappy Bird 3D optimization/index.html")
    final = [
        payload
        for payload in payloads
        if payload.get("choices") and payload["choices"][0].get("finish_reason")
    ]
    assert final[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert final[-1]["mtplx_stats"]["tool_parse_success"] is True


def test_chat_stream_recovers_reasoning_only_completion_without_repair(monkeypatch):
    state = _fake_streaming_session_state()
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    calls: list[list[int]] = []

    def fake_run_generation(_state, prompt_ids, **kwargs):
        calls.append(list(prompt_ids))
        text = "The user is greeting me. I should answer briefly."
        tokens = [ord(char) for char in text]
        token_callback = kwargs.get("token_callback")
        if token_callback is not None:
            for token in tokens:
                token_callback([token])
        return {
            "text": text,
            "tokens": tokens,
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": len(tokens),
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(tokens),
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Hey, how are you?"}],
            "stream": True,
            "max_tokens": 256,
            "enable_thinking": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    payloads = _stream_payloads(body)
    reasoning = "".join(
        choice.get("delta", {}).get("reasoning_content", "")
        for payload in payloads
        for choice in payload.get("choices", [])
    )
    content = "".join(
        choice.get("delta", {}).get("content", "")
        for payload in payloads
        for choice in payload.get("choices", [])
    )
    final = [payload for payload in payloads if payload["choices"][0]["finish_reason"]]
    assert len(calls) == 1
    assert "I should answer briefly" in reasoning
    assert "I should answer briefly" in content
    assert final[-1]["choices"][0]["finish_reason"] == "stop"
    assert final[-1]["mtplx_stats"]["hidden_generation_repair_used"] is False


def test_chat_template_preserves_assistant_tool_calls_and_tool_results():
    tokenizer = CaptureTokenizer()

    openai._encode_messages(
        tokenizer,
        [
            openai.ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    {
                        "id": "call_test",
                        "type": "function",
                        "function": {
                            "name": "session_status",
                            "arguments": "{}",
                        },
                    }
                ],
            ),
            openai.ChatMessage(
                role="tool",
                tool_call_id="call_test",
                content='{"status":"ok"}',
            ),
        ],
        enable_thinking=False,
        add_generation_prompt=False,
    )

    messages, _kwargs = tokenizer.calls[0]
    assert messages[0]["role"] == "assistant"
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {}
    assert messages[1]["role"] == "tool"
    assert messages[1]["tool_call_id"] == "call_test"


def test_gemma4_encoder_renders_assistant_tool_call_before_tool_result():
    class GemmaCaptureTokenizer:
        bos_token = "<bos>"
        model_specific_special_tokens = {
            "think_token": "<|think|>",
            "soc_token": "<|channel>",
            "eoc_token": "<channel|>",
        }

        def __init__(self):
            self.text = ""

        def encode(self, text, **_kwargs):
            self.text = str(text)
            return [ord(char) for char in self.text]

    tokenizer = GemmaCaptureTokenizer()
    tool_call = {
        "id": "call_bash",
        "type": "function",
        "function": {
            "name": "bash",
            "arguments": json.dumps(
                {"command": "ls", "description": "List files"}
            ),
        },
    }

    openai._encode_messages(
        tokenizer,
        [
            openai.ChatMessage(role="user", content="Use ls once."),
            openai.ChatMessage(role="assistant", content="", tool_calls=[tool_call]),
            openai.ChatMessage(
                role="tool",
                tool_call_id="call_bash",
                content="package.json\nsrc",
            ),
        ],
        enable_thinking=True,
        add_generation_prompt=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run a shell command.",
                    "parameters": {
                        "type": "object",
                        "required": ["command"],
                        "properties": {"command": {"type": "string"}},
                    },
                },
            }
        ],
    )

    rendered = tokenizer.text
    tool_call_at = rendered.index("<function=bash>")
    tool_result_at = rendered.index("<|turn>tool_response")
    assert tool_call_at < tool_result_at
    assert "<parameter=command>\nls\n</parameter>" in rendered
    assert "<parameter=description>\nList files\n</parameter>" in rendered
    assert "package.json\nsrc" in rendered


def test_agent_transcript_canonicalization_preserves_tool_history_text():
    tool_call = {
        "id": "call_write",
        "type": "function",
        "function": {"name": "write", "arguments": "{}"},
    }
    polluted = [
        openai.ChatMessage(role="user", content="continue"),
        openai.ChatMessage(
            role="assistant",
            content="Let me continue:\nWrite the Sky, Game, and utils files",
            tool_calls=[tool_call],
        ),
        openai.ChatMessage(role="tool", tool_call_id="call_write", content="ok"),
    ]

    canonical, stats = openai._canonicalize_agent_transcript(
        polluted,
        tools_active=True,
    )

    assert canonical[1].content == "Let me continue:\nWrite the Sky, Game, and utils files"
    assert canonical[1].tool_calls == [tool_call]
    assert stats.stripped_tool_preamble_messages == 0
    assert stats.stripped_tool_preamble_chars == 0
    assert stats.to_metrics()["transcript_canonicalized"] is False


def test_agent_transcript_canonicalization_strips_opencode_tool_preamble_text():
    tool_call = {
        "id": "call_read",
        "type": "function",
        "function": {"name": "read", "arguments": '{"filePath":"src/game/Game.ts"}'},
    }

    canonical, stats = openai._canonicalize_agent_transcript(
        [
            openai.ChatMessage(role="user", content="continue"),
            openai.ChatMessage(
                role="assistant",
                content="Let me inspect the game loop before answering.",
                tool_calls=[tool_call],
            ),
            openai.ChatMessage(role="tool", tool_call_id="call_read", content="ok"),
        ],
        tools_active=True,
        strip_tool_call_preamble_text=True,
    )

    assert canonical[1].content == ""
    assert canonical[1].tool_calls == [tool_call]
    assert stats.stripped_tool_preamble_messages == 1
    assert stats.stripped_tool_preamble_chars == len(
        "Let me inspect the game loop before answering."
    )
    assert stats.to_metrics()["transcript_canonicalized"] is True


def test_agent_transcript_canonicalization_strips_inspection_tool_preamble_text():
    tool_call = {
        "id": "call_read",
        "type": "function",
        "function": {"name": "read", "arguments": '{"filePath":"index.html"}'},
    }

    canonical, stats = openai._canonicalize_agent_transcript(
        [
            openai.ChatMessage(role="user", content="evaluate this project"),
            openai.ChatMessage(
                role="assistant",
                content="Let me read the remaining critical sections.",
                tool_calls=[tool_call],
            ),
            openai.ChatMessage(role="tool", tool_call_id="call_read", content="ok"),
        ],
        tools_active=True,
    )

    assert canonical[1].content == ""
    assert canonical[1].tool_calls == [tool_call]
    assert stats.stripped_tool_preamble_messages == 1
    assert stats.stripped_tool_preamble_chars == len(
        "Let me read the remaining critical sections."
    )
    assert stats.to_metrics()["transcript_canonicalized"] is True


def test_agent_transcript_canonicalization_compacts_digested_large_tool_results():
    tool_call = {
        "id": "call_read",
        "type": "function",
        "function": {
            "name": "read",
            "arguments": '{"filePath":"src/game/HUD.ts"}',
        },
    }
    large_output = (
        "<path>src/game/HUD.ts</path>\n"
        "<content>\n"
        "1: export class HUD {\n"
        + ("2: filler line that should not live forever\n" * 220)
        + "223: setPowerBar(power: number, visible: boolean) {\n"
        "226: }\n"
        "</content>"
    )

    canonical, stats = openai._canonicalize_agent_transcript(
        [
            openai.ChatMessage(role="user", content="find the power meter"),
            openai.ChatMessage(role="assistant", content="", tool_calls=[tool_call]),
            openai.ChatMessage(
                role="tool",
                tool_call_id="call_read",
                content=large_output,
            ),
            openai.ChatMessage(
                role="assistant",
                content="The power meter setter is HUD.ts lines 223-226.",
            ),
            openai.ChatMessage(role="user", content="repeat the line range"),
        ],
        tools_active=True,
    )

    compacted = str(canonical[2].content)
    assert compacted.startswith("<mtplx_compacted_tool_output")
    assert "original_chars=" in compacted
    assert "Call the tool again with a narrower range" in compacted
    assert "1: export class HUD" in compacted
    assert "226: }" in compacted
    assert len(compacted) < 1_000
    assert len(compacted) < len(large_output)
    assert canonical[3].content == "The power meter setter is HUD.ts lines 223-226."
    assert stats.compacted_tool_result_messages == 1
    assert stats.compacted_tool_result_chars == len(large_output) - len(compacted)
    metrics = stats.to_metrics()
    assert metrics["transcript_canonicalized"] is True
    assert metrics["transcript_canonical_message_chars"] < metrics["transcript_raw_message_chars"]


def test_agent_transcript_canonicalization_keeps_followup_tool_digests_small():
    messages = [
        openai.ChatMessage(
            role="system",
            content="OpenCode system prompt\n" + ("tool instructions\n" * 550),
        ),
        openai.ChatMessage(role="user", content="find the power meter"),
        openai.ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call_grep",
                    "type": "function",
                    "function": {"name": "grep", "arguments": "{}"},
                }
            ],
        ),
        openai.ChatMessage(
            role="tool",
            tool_call_id="call_grep",
            content="Found 28 matches\n" + ("dist bundle match\n" * 1_600),
        ),
        openai.ChatMessage(
            role="tool",
            tool_call_id="call_grep_2",
            content="Found 5 matches\n" + ("more grep output\n" * 650),
        ),
        openai.ChatMessage(
            role="tool",
            tool_call_id="call_read",
            content=(
                "<path>src/game/HUD.ts</path>\n"
                + ("read output\n" * 900)
                + "223: setPowerBar(power: number, visible: boolean)\n226: }\n"
            ),
        ),
        openai.ChatMessage(
            role="assistant",
            content="File: src/game/HUD.ts, lines 84-112 and 223-226.",
        ),
        openai.ChatMessage(role="user", content="repeat the line ranges"),
    ]

    canonical, stats = openai._canonicalize_agent_transcript(
        messages,
        tools_active=True,
    )

    metrics = stats.to_metrics()
    assert stats.compacted_tool_result_messages == 3
    assert metrics["transcript_raw_message_chars"] > 45_000
    assert metrics["transcript_canonical_message_chars"] < 12_000
    assert all(
        len(str(message.content)) < 1_000
        for message in canonical
        if str(message.role).lower() == "tool"
    )


def test_agent_transcript_canonicalization_compacts_tool_loop_history_before_latest_assistant():
    first_call = {
        "id": "call_grep",
        "type": "function",
        "function": {"name": "grep", "arguments": "{}"},
    }
    second_call = {
        "id": "call_read",
        "type": "function",
        "function": {"name": "read", "arguments": "{}"},
    }
    large_grep = "Found many matches\n" + ("node_modules hit\n" * 1_000)
    current_output = "current small read output"

    canonical, stats = openai._canonicalize_agent_transcript(
        [
            openai.ChatMessage(role="user", content="find collision logic"),
            openai.ChatMessage(role="assistant", content="", tool_calls=[first_call]),
            openai.ChatMessage(
                role="tool",
                tool_call_id="call_grep",
                content=large_grep,
            ),
            openai.ChatMessage(role="assistant", content="", tool_calls=[second_call]),
            openai.ChatMessage(
                role="tool",
                tool_call_id="call_read",
                content=current_output,
            ),
        ],
        tools_active=True,
    )

    compacted = str(canonical[2].content)
    assert compacted.startswith("<mtplx_compacted_tool_output")
    assert "later assistant step already digested it" in compacted
    assert canonical[4].content == current_output
    assert stats.compacted_tool_result_messages == 1
    assert stats.compacted_active_read_messages == 0
    assert stats.to_metrics()["transcript_canonicalized"] is True


def test_agent_transcript_canonicalization_keeps_current_small_non_read_tool_result_open():
    tool_call = {
        "id": "call_read",
        "type": "function",
        "function": {
            "name": "read",
            "arguments": '{"filePath":"src/game/HUD.ts"}',
        },
    }
    output = "current tool output\n" + ("line\n" * 20)

    canonical, stats = openai._canonicalize_agent_transcript(
        [
            openai.ChatMessage(role="user", content="find the power meter"),
            openai.ChatMessage(role="assistant", content="", tool_calls=[tool_call]),
            openai.ChatMessage(
                role="tool",
                tool_call_id="call_read",
                content=output,
            ),
        ],
        tools_active=True,
    )

    assert canonical[2].content == output
    assert stats.compacted_tool_result_messages == 0
    assert stats.compacted_active_read_messages == 0
    assert stats.to_metrics()["transcript_canonicalized"] is False


def test_agent_transcript_canonicalization_compacts_current_large_glob_output():
    tool_call = {
        "id": "call_glob",
        "type": "function",
        "function": {
            "name": "glob",
            "arguments": '{"pattern":"**/*.ts"}',
        },
    }
    lines = [
        "/Users/youssof/Documents/bow masters 3d/src/game/Arrow.ts",
        "/Users/youssof/Documents/bow masters 3d/src/game/ObstacleManager.ts",
    ]
    lines.extend(
        f"/Users/youssof/Documents/bow masters 3d/node_modules/pkg_{index}/index.d.ts"
        for index in range(900)
    )
    large_output = "\n".join(lines)

    canonical, stats = openai._canonicalize_agent_transcript(
        [
            openai.ChatMessage(role="user", content="find arrow collision handling"),
            openai.ChatMessage(role="assistant", content="", tool_calls=[tool_call]),
            openai.ChatMessage(
                role="tool",
                tool_call_id="call_glob",
                content=large_output,
            ),
        ],
        tools_active=True,
    )

    compacted = str(canonical[2].content)
    assert compacted.startswith("<mtplx_compacted_active_tool_output")
    assert 'anchor="' in compacted
    assert "Large current tool output abbreviated" in compacted
    assert "src/game/Arrow.ts" in compacted
    assert "src/game/ObstacleManager.ts" in compacted
    assert "read_hint_count=2" in compacted
    assert "<next_read_hints>" in compacted
    assert 'filePath="/Users/youssof/Documents/bow masters 3d/src/game/Arrow.ts"' in compacted
    assert "Avoid broad list/glob/grep repeats" in compacted
    assert len(compacted) < 6_000
    assert len(compacted) < len(large_output)
    assert stats.compacted_tool_result_messages == 0
    assert stats.compacted_active_tool_result_messages == 1
    assert stats.compacted_active_read_messages == 0
    metrics = stats.to_metrics()
    assert metrics["transcript_compacted_active_tool_result_messages"] == 1
    assert metrics["transcript_compacted_active_tool_result_read_hints"] == 2
    assert metrics["transcript_canonicalized"] is True


def test_agent_transcript_canonicalization_adds_read_ranges_for_build_output():
    tool_call = {
        "id": "call_bash",
        "type": "function",
        "function": {
            "name": "bash",
            "arguments": '{"command":"npm run typecheck"}',
        },
    }
    large_output = "\n".join(
        [
            "> bowmasters@1.0.0 typecheck",
            "src/game/ObstacleManager.ts:240:13 - error TS2322: Type 'string' is not assignable to type 'number'.",
            "src/game/HUD.ts(88,5): warning TS6133: 'power' is declared but never read.",
            'File "/Users/youssof/Documents/bow masters 3d/scripts/check.py", line 42, in <module>',
            "Traceback (most recent call last):",
        ]
        + [
            f"node_modules/pkg_{index}/index.d.ts: noisy dependency line"
            for index in range(350)
        ]
        + [
            "FAILED typecheck",
            "src/game/ObstacleManager.ts:253:9 - error TS2554: Expected 2 arguments, but got 1.",
        ]
    )

    canonical, stats = openai._canonicalize_agent_transcript(
        [
            openai.ChatMessage(role="user", content="fix the typecheck failures"),
            openai.ChatMessage(role="assistant", content="", tool_calls=[tool_call]),
            openai.ChatMessage(
                role="tool",
                tool_call_id="call_bash",
                content=large_output,
            ),
        ],
        tools_active=True,
    )

    compacted = str(canonical[2].content)
    assert compacted.startswith("<mtplx_compacted_active_tool_output")
    assert "read_hint_count=3" in compacted
    assert "<next_read_hints>" in compacted
    assert 'filePath="src/game/ObstacleManager.ts" start="220" end="273"' in compacted
    assert 'filePath="src/game/HUD.ts" start="68" end="108"' in compacted
    assert 'filePath="/Users/youssof/Documents/bow masters 3d/scripts/check.py" start="22" end="62"' in compacted
    assert "Do not rerun the broad tool command unchanged" in compacted
    assert "error TS2322" in compacted
    assert "error TS2554" in compacted
    assert stats.compacted_active_tool_result_messages == 1
    assert stats.compacted_active_tool_result_chars == len(large_output) - len(compacted)
    assert stats.to_metrics()["transcript_compacted_active_tool_result_read_hints"] == 3


def test_agent_transcript_canonicalization_compacts_current_large_read_outputs():
    tool_call = {
        "id": "call_read",
        "type": "function",
        "function": {
            "name": "read",
            "arguments": '{"filePath":"src/game/ObstacleManager.ts"}',
        },
    }
    body_lines = [
        "1: import * as THREE from 'three';",
        "2: import { Arrow } from './Arrow';",
    ]
    for line_no in range(3, 380):
        if line_no == 240:
            body_lines.append("240:   public checkArrowCollisions(arrow: Arrow): void {")
        elif line_no == 241:
            body_lines.append("241:     if (arrow.isEmbedded()) return;")
        elif line_no == 248:
            body_lines.append("248:       const dist = arrowPos.distanceTo(obstacle.position);")
        elif line_no == 252:
            body_lines.append("252:       if (dist < hitRadius) {")
        elif line_no == 253:
            body_lines.append("253:         arrow.embedInTerrain(obstacle.position.y - 0.5);")
        elif line_no == 320:
            body_lines.append("320:   window.addEventListener('touchstart', flap);")
        elif line_no == 321:
            body_lines.append("321:   scene.remove(pipe);")
        elif line_no == 266:
            body_lines.append("266:   }")
        else:
            body_lines.append(
                f"{line_no}:     filler line {line_no} with enough body text to "
                "represent a real full-file OpenCode read payload;"
            )
    large_read_output = (
        "<path>src/game/ObstacleManager.ts</path>\n"
        "<type>file</type>\n"
        "<content>\n"
        + "\n".join(body_lines)
        + "\n</content>"
    )

    canonical, stats = openai._canonicalize_agent_transcript(
        [
            openai.ChatMessage(role="user", content="find arrow collision handling"),
            openai.ChatMessage(role="assistant", content="", tool_calls=[tool_call]),
            openai.ChatMessage(
                role="tool",
                tool_call_id="call_read",
                content=large_read_output,
            ),
        ],
        tools_active=True,
    )

    compacted = str(canonical[2].content)
    assert compacted.startswith("<mtplx_compacted_active_read_output")
    assert 'anchor="' in compacted
    assert 'line_range="1-379"' in compacted
    assert 'next_read_hint_count="' in compacted
    assert "<next_read_hints>" in compacted
    assert 'read filePath="src/game/ObstacleManager.ts"' in compacted
    assert "offset=" in compacted
    assert "Large current read abbreviated" in compacted
    assert "For review or evaluation, answer from this excerpt" in compacted
    assert "240:   public checkArrowCollisions(arrow: Arrow): void {" in compacted
    assert "253:         arrow.embedInTerrain(obstacle.position.y - 0.5);" in compacted
    assert "320:   window.addEventListener('touchstart', flap);" in compacted
    assert "321:   scene.remove(pipe);" in compacted
    assert len(compacted) < len(large_read_output)
    assert stats.compacted_tool_result_messages == 0
    assert stats.compacted_active_read_messages == 1
    assert stats.compacted_active_read_chars == len(large_read_output) - len(compacted)
    metrics = stats.to_metrics()
    assert metrics["transcript_canonicalized"] is True
    assert metrics["transcript_compacted_active_read_messages"] == 1


def test_agent_transcript_canonicalization_uses_inspection_digest_for_review_reads():
    tool_call = {
        "id": "call_read",
        "type": "function",
        "function": {
            "name": "read",
            "arguments": '{"filePath":"index.html","offset":91,"limit":130}',
        },
    }
    body_lines = []
    for line_no in range(91, 221):
        if line_no == 92:
            body_lines.append(
                '92: let best = parseInt(localStorage.getItem("fb3d_best") || "0");'
            )
        elif line_no == 108:
            body_lines.append("108: function setDifficulty(diff) {")
        elif line_no == 146:
            body_lines.append("146: const renderer = new THREE.WebGLRenderer();")
        elif line_no == 184:
            body_lines.append("184: function checkCollision() {")
        elif line_no == 205:
            body_lines.append("205: requestAnimationFrame(animate);")
        else:
            body_lines.append(
                f"{line_no}:   game implementation line {line_no} "
                "with enough source text to reproduce the OpenCode read shape;"
            )
    read_output = (
        "<path>index.html</path>\n"
        "<type>file</type>\n"
        "<content>\n"
        + "\n".join(body_lines)
        + "\n</content>"
    )

    canonical, stats = openai._canonicalize_agent_transcript(
        [
            openai.ChatMessage(
                role="user",
                content=(
                    "Evaluate the quality of this project. Inspect the relevant "
                    "files with tools and give a concise improvement plan."
                ),
            ),
            openai.ChatMessage(role="assistant", content="", tool_calls=[tool_call]),
            openai.ChatMessage(
                role="tool",
                tool_call_id="call_read",
                content=read_output,
            ),
        ],
        tools_active=True,
    )

    compacted = str(canonical[2].content)
    assert compacted.startswith("<mtplx_read_inspection_digest")
    assert 'anchor="' in compacted
    assert 'line_range="91-220"' in compacted
    assert 'full_file_expansion_needed="false"' in compacted
    assert "read-only review/evaluation evidence digest" in compacted
    assert "do not request adjacent ranges or the full file" in compacted
    assert "do not copy the numbered evidence lines into reasoning" in compacted.lower()
    assert "line 92: let best = parseInt" in compacted
    assert "line 108: function setDifficulty(diff) {" in compacted
    assert "line 184: function checkCollision() {" in compacted
    assert "line 97:   game implementation line 97" not in compacted
    assert "omitted" not in compacted.lower()
    assert len(compacted) < 2_600
    assert len(compacted) < len(read_output)
    assert stats.compacted_active_read_messages == 1
    assert stats.compacted_active_read_inspection_messages == 1
    metrics = stats.to_metrics()
    assert metrics["transcript_compacted_active_read_inspection_messages"] == 1
    assert metrics["transcript_canonical_message_chars"] < 3_000


def test_agent_transcript_canonicalization_spreads_full_file_inspection_anchors():
    tool_call = {
        "id": "call_read",
        "type": "function",
        "function": {
            "name": "read",
            "arguments": '{"filePath":"index.html"}',
        },
    }
    important_lines = {
        5: '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
        6: "<title>Flappy Bird 3D</title>",
        92: 'let best = parseInt(localStorage.getItem("fb3d_best") || "0");',
        119: "const renderer = new THREE.WebGLRenderer({ antialias: true });",
        120: "renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));",
        133: "const ground = new THREE.Mesh(new THREE.BoxGeometry(100, 1, 100));",
        135: "ground.receiveShadow = true;",
        136: "scene.add(ground);",
        137: "const player = new THREE.Mesh(new THREE.SphereGeometry(0.35));",
        169: "const pipeMaterial = new THREE.MeshStandardMaterial({ color });",
        170: "const obstacle = new THREE.Mesh(new THREE.BoxGeometry(1, 4, 1));",
        184: "function checkCollision(pipe) {",
        185: "const relZ = Math.abs(birdZ - pipe.position.z);",
        186: "const pipeHalfZ = 0.9;",
        187: "if (relZ < pipeHalfZ) {",
        188: "if (Math.abs(birdY - pipe.userData.gapCenter) > pipe.userData.halfGap + 0.05) {",
        189: "return true;",
        199: "if (player.position.distanceTo(pipe.position) < hitRadius) {",
        200: "  endGame();",
        211: "function setDifficulty(diff) {",
        212: "  localStorage.setItem('fb3d_diff', diff);",
        222: 'window.addEventListener("keydown", (e) => {',
        223: 'if (e.code === "Space" || e.code === "ArrowUp") {',
        224: "e.preventDefault();",
        225: "flap();",
        226: "}",
        227: "window.addEventListener('visibilitychange', pause);",
        233: "renderer.shadowMap.enabled = true;",
        443: "pipes.forEach(p => scene.remove(p));",
        447: "particles.forEach(p => scene.remove(p));",
        448: "particles.length = 0;",
        499: 'window.addEventListener("touchstart", (e) => {',
        500: "e.preventDefault();",
        501: "flap();",
        502: "}, { passive: false });",
        505: 'window.addEventListener("resize", () => {',
        506: "camera.aspect = window.innerWidth / window.innerHeight;",
        507: "camera.updateProjectionMatrix();",
        513: "requestAnimationFrame(animate);",
        514: "const dt = Math.min(clock.getDelta(), 0.05);",
        553: "for (let i = pipes.length - 1; i >= 0; i--) {",
        554: "if (relZ > pipeRemoveZ) {",
        555: "scene.remove(pipe);",
        556: "pipes.splice(i, 1);",
    }
    body_lines = []
    for line_no in range(1, 636):
        if line_no in important_lines:
            body_lines.append(f"{line_no}: {important_lines[line_no]}")
        else:
            body_lines.append(
                f"{line_no}:   game implementation line {line_no} with enough "
                "source text to reproduce a full OpenCode project read;"
            )
    read_output = (
        "<path>index.html</path>\n"
        "<type>file</type>\n"
        "<content>\n"
        + "\n".join(body_lines)
        + "\n</content>"
    )

    canonical, stats = openai._canonicalize_agent_transcript(
        [
            openai.ChatMessage(
                role="user",
                content=(
                    "Evaluate the quality of this project. Inspect the relevant "
                    "files with tools and give a concise improvement plan."
                ),
            ),
            openai.ChatMessage(role="assistant", content="", tool_calls=[tool_call]),
            openai.ChatMessage(
                role="tool",
                tool_call_id="call_read",
                content=read_output,
            ),
        ],
        tools_active=True,
    )

    compacted = str(canonical[2].content)
    assert compacted.startswith("<mtplx_read_inspection_digest")
    assert 'line 5: <meta name="viewport" content="width=device-width, initial-scale=1.0">' in compacted
    assert "line 6: <title>Flappy Bird 3D</title>" in compacted
    assert "line 185: const relZ = Math.abs(birdZ - pipe.position.z);" in compacted
    assert "line 188: if (Math.abs(birdY - pipe.userData.gapCenter) > pipe.userData.halfGap + 0.05) {" in compacted
    assert "line 189: return true;" in compacted
    assert "line 443: pipes.forEach(p => scene.remove(p));" in compacted
    assert "line 447: particles.forEach(p => scene.remove(p));" in compacted
    assert "line 448: particles.length = 0;" in compacted
    assert "line 120: renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));" in compacted
    assert 'line 222: window.addEventListener("keydown", (e) => {' in compacted
    assert 'line 223: if (e.code === "Space" || e.code === "ArrowUp") {' in compacted
    assert "line 224: e.preventDefault();" in compacted
    assert "line 225: flap();" in compacted
    assert 'line 499: window.addEventListener("touchstart", (e) => {' in compacted
    assert "line 500: e.preventDefault();" in compacted
    assert "line 501: flap();" in compacted
    assert "line 502: }, { passive: false });" in compacted
    assert 'line 505: window.addEventListener("resize", () => {' in compacted
    assert "line 506: camera.aspect = window.innerWidth / window.innerHeight;" in compacted
    assert "line 507: camera.updateProjectionMatrix();" in compacted
    assert "line 514: const dt = Math.min(clock.getDelta(), 0.05);" in compacted
    assert "line 553: for (let i = pipes.length - 1; i >= 0; i--) {" in compacted
    assert "line 554: if (relZ > pipeRemoveZ) {" in compacted
    assert "line 555: scene.remove(pipe);" in compacted
    assert "line 556: pipes.splice(i, 1);" in compacted
    assert len(compacted) < len(read_output)
    assert stats.compacted_active_read_messages == 1
    assert stats.compacted_active_read_inspection_messages == 1


def test_agent_transcript_canonicalization_compacts_plain_read_tool_output():
    tool_call = {
        "id": "call_read",
        "type": "function",
        "function": {
            "name": "read",
            "arguments": '{"filePath":"index.html"}',
        },
    }
    important_lines = {
        78: '<script type="importmap">',
        81: '"three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js"',
        92: 'let best = parseInt(localStorage.getItem("fb3d_best") || "0");',
        108: "function setDifficulty(diff) {",
        122: "const pipeRemoveZ = 15;",
        133: "const renderer = new THREE.WebGLRenderer({ antialias: true });",
        170: "const skyMat = new THREE.ShaderMaterial({",
        413: "function checkCollision(pipe) {",
        443: "pipes.forEach(p => scene.remove(p));",
        499: 'window.addEventListener("touchstart", (e) => {',
        506: "camera.aspect = window.innerWidth / window.innerHeight;",
        507: "camera.updateProjectionMatrix();",
        513: "requestAnimationFrame(animate);",
        514: "const dt = Math.min(clock.getDelta(), 0.05);",
        555: "scene.remove(pipe);",
    }
    plain_read_output = "\n".join(
        important_lines.get(
            line_no,
            f"game implementation line {line_no} with enough source text for Pi;",
        )
        for line_no in range(1, 636)
    )

    canonical, stats = openai._canonicalize_agent_transcript(
        [
            openai.ChatMessage(
                role="user",
                content=(
                    "Evaluate the quality of this project. Inspect the relevant "
                    "files with tools and give a concise improvement plan."
                ),
            ),
            openai.ChatMessage(role="assistant", content="", tool_calls=[tool_call]),
            openai.ChatMessage(
                role="tool",
                tool_call_id="call_read",
                content=plain_read_output,
            ),
        ],
        tools_active=True,
    )

    compacted = str(canonical[2].content)
    assert compacted.startswith("<mtplx_read_inspection_digest")
    assert "<path>index.html</path>" in compacted
    assert "line 81: \"three\": \"https://cdn.jsdelivr.net/npm/three@0.160.0" in compacted
    assert "line 499: window.addEventListener" in compacted
    assert "line 514: const dt = Math.min(clock.getDelta(), 0.05);" in compacted
    assert len(compacted) < len(plain_read_output)
    assert stats.compacted_active_read_messages == 1
    assert stats.compacted_active_read_inspection_messages == 1


def test_agent_transcript_canonicalization_collapses_repeated_inspection_reads():
    first_call = {
        "id": "call_read_1",
        "type": "function",
        "function": {
            "name": "read",
            "arguments": '{"filePath":"index.html"}',
        },
    }
    second_call = {
        "id": "call_read_2",
        "type": "function",
        "function": {
            "name": "read",
            "arguments": '{"filePath":"index.html","offset":250,"limit":90}',
        },
    }
    full_lines = []
    for line_no in range(1, 361):
        if line_no == 146:
            full_lines.append("146: const renderer = new THREE.WebGLRenderer();")
        elif line_no == 184:
            full_lines.append("184: function checkCollision() {")
        elif line_no == 320:
            full_lines.append("320: pipes.forEach(p => scene.remove(p));")
        else:
            full_lines.append(
                f"{line_no}:   game implementation line {line_no} with review evidence;"
            )
    full_read = (
        "<path>index.html</path>\n"
        "<type>file</type>\n"
        "<content>\n"
        + "\n".join(full_lines)
        + "\n</content>"
    )
    repeated_read = (
        "<path>index.html</path>\n"
        "<type>file</type>\n"
        "<content>\n"
        + "\n".join(full_lines[249:340])
        + "\n</content>"
    )

    canonical, stats = openai._canonicalize_agent_transcript(
        [
            openai.ChatMessage(
                role="user",
                content="Evaluate the quality of this project and give a concise plan.",
            ),
            openai.ChatMessage(role="assistant", content="", tool_calls=[first_call]),
            openai.ChatMessage(
                role="tool",
                tool_call_id="call_read_1",
                content=full_read,
            ),
            openai.ChatMessage(role="assistant", content="", tool_calls=[second_call]),
            openai.ChatMessage(
                role="tool",
                tool_call_id="call_read_2",
                content=repeated_read,
            ),
        ],
        tools_active=True,
    )

    first_digest = str(canonical[2].content)
    repeated_digest = str(canonical[4].content)
    assert first_digest.startswith("<mtplx_read_inspection_digest")
    assert repeated_digest.startswith("<mtplx_repeated_read_inspection_digest")
    assert 'anchor="' in repeated_digest
    assert 'line_range="250-340"' in repeated_digest
    assert "Use the earlier digest" in repeated_digest
    assert "Do not call read again" in repeated_digest
    assert "line 320: pipes.forEach" not in repeated_digest
    assert len(repeated_digest) < 900
    assert stats.compacted_active_read_inspection_messages == 2
    assert stats.compacted_repeated_read_inspection_messages == 1
    metrics = stats.to_metrics()
    assert metrics["transcript_compacted_repeated_read_inspection_messages"] == 1
    assert metrics["transcript_canonical_message_chars"] < 5_000


def test_agent_transcript_canonicalization_budgets_multi_file_inspection_reads():
    messages = [
        openai.ChatMessage(
            role="user",
            content=(
                "Do not edit files. Inspect the game loop, player, AI, HUD, "
                "camera, and build config, then summarize the launch risks."
            ),
        )
    ]
    for index, path in enumerate(
        [
            "src/game/Game.ts",
            "src/game/Player.ts",
            "src/game/AI.ts",
            "src/game/HUD.ts",
            "src/game/CameraController.ts",
            "vite.config.ts",
        ],
        start=1,
    ):
        call_id = f"call_read_{index}"
        messages.append(
            openai.ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": json.dumps({"filePath": path}),
                        },
                    }
                ],
            )
        )
        body_lines = []
        for line_no in range(1, 141):
            if line_no == 12:
                body_lines.append(f"{line_no}: export class Component{index} {{")
            elif line_no == 48:
                body_lines.append(f"{line_no}:   update(delta: number): void {{")
            elif line_no == 91:
                body_lines.append(f"{line_no}:   requestAnimationFrame(this.tick);")
            else:
                body_lines.append(
                    f"{line_no}:   source line {line_no} for {path} with enough "
                    "text to simulate a real OpenCode full-file read payload;"
                )
        messages.append(
            openai.ChatMessage(
                role="tool",
                tool_call_id=call_id,
                content=(
                    f"<path>{path}</path>\n"
                    "<type>file</type>\n"
                    "<content>\n"
                    + "\n".join(body_lines)
                    + "\n</content>"
                ),
            )
        )

    canonical, stats = openai._canonicalize_agent_transcript(
        messages,
        tools_active=True,
    )

    digests = [
        str(message.content)
        for message in canonical
        if str(message.role).lower() == "tool"
    ]
    assert len(digests) == 6
    assert all(digest.startswith("<mtplx_read_inspection_digest") for digest in digests)
    evidence_counts = [
        int(re.search(r'evidence_lines="(\d+)"', digest).group(1))
        for digest in digests
    ]
    assert all(3 <= count <= 16 for count in evidence_counts)
    assert all("requestAnimationFrame" in digest for digest in digests)
    assert stats.compacted_active_read_inspection_messages == 6
    metrics = stats.to_metrics()
    assert metrics["transcript_inspection_read_budget_candidate_messages"] == 6
    assert metrics["transcript_inspection_read_budget_max_lines_per_file"] == 16
    assert metrics["transcript_canonical_message_chars"] < 18_000


def test_agent_transcript_canonicalization_compacts_truncated_read_continuation_hints():
    tool_call = {
        "id": "call_read",
        "type": "function",
        "function": {
            "name": "read",
            "arguments": '{"filePath":"index.html","offset":436,"limit":80}',
        },
    }
    body_lines = []
    for line_no in range(436, 516):
        if line_no == 443:
            body_lines.append("443:   pipes.forEach(p => scene.remove(p));")
        elif line_no == 492:
            body_lines.append('492: window.addEventListener("keydown", (e) => {')
        elif line_no == 499:
            body_lines.append('499: window.addEventListener("touchstart", (e) => {')
        elif line_no == 513:
            body_lines.append("513:   requestAnimationFrame(animate);")
        else:
            body_lines.append(f"{line_no}:   game loop line {line_no};")
    truncated_read_output = (
        "<path>index.html</path>\n"
        "<type>file</type>\n"
        "<content>\n"
        + "\n".join(body_lines)
        + "\n\n(Showing lines 436-515 of 635. Use offset=516 to continue.)\n"
        "</content>"
    )
    assert len(truncated_read_output) < openai._ACTIVE_READ_COMPACT_THRESHOLD_CHARS

    canonical, stats = openai._canonicalize_agent_transcript(
        [
            openai.ChatMessage(role="user", content="evaluate this project"),
            openai.ChatMessage(role="assistant", content="", tool_calls=[tool_call]),
            openai.ChatMessage(
                role="tool",
                tool_call_id="call_read",
                content=truncated_read_output,
            ),
        ],
        tools_active=True,
    )

    compacted = str(canonical[2].content)
    assert compacted.startswith("<mtplx_read_inspection_digest")
    assert 'continuation_hint_removed="true"' in compacted
    assert "Use offset=" not in compacted
    assert "do not request adjacent ranges or the full file" in compacted
    assert "line 443:   pipes.forEach(p => scene.remove(p));" in compacted
    assert 'line 499: window.addEventListener("touchstart", (e) => {' in compacted
    assert stats.compacted_active_read_messages == 1
    assert stats.compacted_active_read_chars == len(truncated_read_output) - len(
        compacted
    )


def test_agent_transcript_canonicalization_drops_verbatim_source_dump_assistant_history():
    source_dump = "\n".join(
        f"{line_no}: const copiedLine{line_no} = {line_no};"
        for line_no in range(91, 170)
    )

    canonical, stats = openai._canonicalize_agent_transcript(
        [
            openai.ChatMessage(role="user", content="evaluate this project"),
            openai.ChatMessage(role="assistant", content=source_dump),
            openai.ChatMessage(role="user", content="continue"),
        ],
        tools_active=True,
    )

    assert [message.role for message in canonical] == ["user"]
    assert "const copiedLine" not in str(canonical[0].content)
    assert stats.skipped_verbatim_tool_output_assistant_messages == 1
    assert stats.skipped_verbatim_tool_output_assistant_chars == len(source_dump)
    metrics = stats.to_metrics()
    assert metrics["transcript_skipped_verbatim_tool_output_assistant_messages"] == 1
    assert metrics["transcript_canonicalized"] is True


def test_agent_transcript_canonicalization_strips_inspection_tool_call_preambles():
    tool_call = {
        "id": "call_read",
        "type": "function",
        "function": {
            "name": "read",
            "arguments": '{"filePath":"src/game/Player.ts"}',
        },
    }

    canonical, stats = openai._canonicalize_agent_transcript(
        [
            openai.ChatMessage(
                role="user",
                content="Do not edit files. Inspect controls and game feel.",
            ),
            openai.ChatMessage(
                role="assistant",
                content="I'll inspect the player controls and then compare the game loop.",
                tool_calls=[tool_call],
            ),
            openai.ChatMessage(
                role="tool",
                tool_call_id="call_read",
                content="export class Player {}",
            ),
        ],
        tools_active=True,
    )

    assert canonical[1].role == "assistant"
    assert canonical[1].tool_calls == [tool_call]
    assert canonical[1].content == ""
    assert stats.stripped_tool_preamble_messages == 1
    assert stats.stripped_tool_preamble_chars > 0
    assert stats.to_metrics()["transcript_canonicalized"] is True


def test_agent_transcript_canonicalization_skips_repeated_assistant_text():
    repeated = (
        "Let me continue:\nWrite the Sky, Game, and utils files\n"
        "Run the typecheck\nRun the dev server\n"
    ) * 8

    canonical, stats = openai._canonicalize_agent_transcript(
        [
            openai.ChatMessage(role="user", content="continue"),
            openai.ChatMessage(role="assistant", content=repeated),
            openai.ChatMessage(role="user", content="status?"),
        ],
        tools_active=True,
    )

    assert [message.role for message in canonical] == ["user", "user"]
    assert stats.skipped_repeated_assistant_messages == 1


def test_agent_transcript_canonicalization_skips_stalled_tool_preamble():
    first_tool = {
        "id": "call_bash",
        "type": "function",
        "function": {"name": "bash", "arguments": '{"command":"tsc"}'},
    }
    second_tool = {
        "id": "call_todo",
        "type": "function",
        "function": {"name": "todowrite", "arguments": '{"todos":[]}'},
    }
    duplicate = (
        "Almost there - just strict TypeScript checks. "
        "Let me fix all remaining errors:"
    )

    canonical, stats = openai._canonicalize_agent_transcript(
        [
            openai.ChatMessage(role="user", content="What is status?"),
            openai.ChatMessage(
                role="assistant",
                content="Let me check what's left to fix and get this running.",
                tool_calls=[first_tool],
            ),
            openai.ChatMessage(
                role="tool",
                tool_call_id="call_bash",
                content="src/game/AI.ts: strict TypeScript errors",
            ),
            openai.ChatMessage(
                role="assistant",
                content=duplicate,
                tool_calls=[second_tool],
            ),
            openai.ChatMessage(
                role="tool",
                tool_call_id="call_todo",
                content="todos updated",
            ),
            openai.ChatMessage(role="assistant", content=duplicate),
            openai.ChatMessage(role="user", content="ok"),
        ],
        tools_active=True,
    )

    assert [message.role for message in canonical] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "user",
    ]
    assert canonical[1].content == "Let me check what's left to fix and get this running."
    assert canonical[3].content == duplicate
    assert stats.stripped_tool_preamble_messages == 0
    assert stats.skipped_repeated_assistant_messages == 0
    assert stats.skipped_stalled_agent_preamble_messages == 1


def test_agent_transcript_canonicalization_drops_aborted_assistant_placeholder():
    canonical, stats = openai._canonicalize_agent_transcript(
        [
            openai.ChatMessage(role="user", content="respond"),
            openai.ChatMessage(
                role="assistant",
                content="",
                error="MessageAbortedError",
            ),
            openai.ChatMessage(role="user", content="continue"),
        ],
        tools_active=True,
    )

    assert [message.role for message in canonical] == ["user"]
    assert canonical[0].content == "respond\n\ncontinue"
    assert stats.skipped_aborted_assistant_messages == 1
    assert stats.merged_consecutive_user_messages == 1


def test_agent_transcript_canonicalization_drops_duplicate_user_after_abort():
    canonical, stats = openai._canonicalize_agent_transcript(
        [
            openai.ChatMessage(role="user", content="Hi, how are you?"),
            openai.ChatMessage(
                role="assistant",
                content="",
                error="MessageAbortedError",
            ),
            openai.ChatMessage(role="user", content="Hi, how are you?"),
        ],
        tools_active=True,
    )

    assert [message.role for message in canonical] == ["user"]
    assert canonical[0].content == "Hi, how are you?"
    assert stats.skipped_aborted_assistant_messages == 1
    assert stats.dropped_duplicate_user_messages == 1
    assert stats.merged_consecutive_user_messages == 0
    assert stats.to_metrics()["transcript_canonicalized"] is True


def test_agent_transcript_canonicalization_drops_orphan_chitchat_assistant():
    canonical, stats = openai._canonicalize_agent_transcript(
        [
            openai.ChatMessage(role="user", content="hey"),
            openai.ChatMessage(
                role="assistant",
                content="Hey! What can I help you with?",
            ),
            openai.ChatMessage(role="user", content="Hi, how are you?"),
        ],
        tools_active=True,
    )

    assert [message.role for message in canonical] == ["user"]
    assert canonical[0].content == "Hi, how are you?"
    assert stats.skipped_orphan_chitchat_assistant_messages == 1
    assert stats.dropped_duplicate_user_messages == 1
    assert stats.merged_consecutive_user_messages == 0
    assert stats.to_metrics()["transcript_canonicalized"] is True


def test_agent_transcript_canonicalization_drops_old_history_for_simple_chitchat():
    tool_call = {
        "id": "call_read",
        "type": "function",
        "function": {
            "name": "read",
            "arguments": '{"filePath":"src/game/CameraController.ts"}',
        },
    }
    canonical, stats = openai._canonicalize_agent_transcript(
        [
            openai.ChatMessage(role="system", content="OpenCode system prompt"),
            openai.ChatMessage(role="user", content="evaluate codebase in depth"),
            openai.ChatMessage(role="assistant", content="", tool_calls=[tool_call]),
            openai.ChatMessage(
                role="tool",
                tool_call_id="call_read",
                content="large file output\n" * 400,
            ),
            openai.ChatMessage(
                role="assistant",
                content="Camera controls live in CameraController.ts.",
            ),
            openai.ChatMessage(role="user", content="hi"),
        ],
        tools_active=False,
    )

    assert [message.role for message in canonical] == ["system", "user"]
    assert canonical[-1].content == "hi"
    assert stats.dropped_simple_chitchat_history_messages == 4
    metrics = stats.to_metrics()
    assert metrics["transcript_canonicalized"] is True
    assert metrics["transcript_canonical_message_chars"] < 64
    assert metrics["transcript_dropped_simple_chitchat_history_chars"] > 1_000


def test_agent_transcript_canonicalization_collapses_repeated_user_text_part():
    canonical, stats = openai._canonicalize_agent_transcript(
        [
            openai.ChatMessage(
                role="user",
                content="Hi, how are you?Hi, how are you?",
            ),
        ],
        tools_active=True,
    )

    assert [message.role for message in canonical] == ["user"]
    assert canonical[0].content == "Hi, how are you?"
    assert stats.collapsed_repeated_user_messages == 1
    assert stats.to_metrics()["transcript_canonicalized"] is True


def test_agent_transcript_canonicalization_collapses_short_repeated_chitchat():
    canonical, stats = openai._canonicalize_agent_transcript(
        [
            openai.ChatMessage(role="user", content="hihi"),
        ],
        tools_active=True,
    )

    assert [message.role for message in canonical] == ["user"]
    assert canonical[0].content == "hi"
    assert stats.collapsed_repeated_user_messages == 1
    assert stats.to_metrics()["transcript_canonicalized"] is True


def test_agent_transcript_canonicalization_marks_repeated_shell_timeouts():
    tool_call = {
        "id": "call_tsc_1",
        "type": "function",
        "function": {
            "name": "bash",
            "arguments": json.dumps(
                {"command": "npx tsc --noEmit 2>&1 | head -10", "timeout": 30000}
            ),
        },
    }
    repeated_call = {
        **tool_call,
        "id": "call_tsc_2",
    }
    timeout_output = (
        "(no output)\n\n<shell_metadata>\n"
        "shell tool terminated command after exceeding timeout 30000 ms\n"
        "</shell_metadata>"
    )

    canonical, stats = openai._canonicalize_agent_transcript(
        [
            openai.ChatMessage(role="user", content="fix type errors"),
            openai.ChatMessage(role="assistant", content="", tool_calls=[tool_call]),
            openai.ChatMessage(
                role="tool",
                tool_call_id="call_tsc_1",
                content=timeout_output,
            ),
            openai.ChatMessage(role="assistant", content="", tool_calls=[repeated_call]),
            openai.ChatMessage(
                role="tool",
                tool_call_id="call_tsc_2",
                content=timeout_output,
            ),
        ],
        tools_active=True,
    )

    assert canonical[-1].content.startswith("<mtplx_repeated_timeout_tool_output")
    assert "Do not call the same command again unchanged" in canonical[-1].content
    assert "npx tsc --noEmit" in canonical[-1].content
    assert stats.compacted_repeated_timeout_tool_messages == 1


def test_tool_contract_stabilizes_tool_schema_with_agent_tail_guardrail():
    messages = [{"role": "user", "content": "status?"}]

    with_contract = openai._with_mtplx_tool_contract(
        messages,
        tools=[_bash_tool_schema(), _tool_schema()],
    )

    assert with_contract[0]["role"] == "system"
    assert "MTPLX tool contract:" in with_contract[0]["content"]
    assert "Read a file in ONE call" in with_contract[0]["content"]
    assert "Never print file contents" in with_contract[0]["content"]
    assert "only inside the declared write/edit tool call arguments" in with_contract[0]["content"]
    assert "MTPLX coding-agent tool protocol reminder:" in with_contract[0]["content"]
    assert "emit one declared <tool_call> now" in with_contract[0]["content"]
    assert "implementation payloads in the declared tool call arguments" in with_contract[0]["content"]
    assert "let me fix this" in with_contract[0]["content"]
    assert [message["role"] for message in with_contract] == ["system", "user"]


def test_native_tool_prompt_mode_keeps_template_tools_and_adds_agent_tail():
    tokenizer = CaptureTokenizer()
    observability: dict[str, object] = {}

    openai._encode_messages(
        tokenizer,
        [openai.ChatMessage(role="user", content="Read package.json")],
        enable_thinking=True,
        add_generation_prompt=True,
        tools=[_bash_tool_schema(), _tool_schema()],
        tool_prompt_mode="native",
        template_observability=observability,
    )

    messages, kwargs = tokenizer.calls[-1]
    rendered_content = "\n".join(str(message.get("content") or "") for message in messages)
    assert kwargs["tools"] == [_bash_tool_schema(), _tool_schema()]
    assert "MTPLX tool contract:" not in rendered_content
    assert "MTPLX coding-agent tool protocol reminder:" in rendered_content
    assert "emit one declared <tool_call> now" in rendered_content
    assert "MTPLX tool-result continuation:" not in rendered_content
    assert observability["native_agent_tail_contract_active"] is True


def test_native_tool_prompt_mode_suppresses_agent_tail_for_chitchat():
    tokenizer = CaptureTokenizer()
    observability: dict[str, object] = {}

    openai._encode_messages(
        tokenizer,
        [openai.ChatMessage(role="user", content="hi how are you")],
        enable_thinking=True,
        add_generation_prompt=True,
        tools=[_bash_tool_schema(), _tool_schema()],
        tool_prompt_mode="native",
        template_observability=observability,
    )

    messages, kwargs = tokenizer.calls[-1]
    rendered_content = "\n".join(str(message.get("content") or "") for message in messages)
    assert kwargs["tools"] == [_bash_tool_schema(), _tool_schema()]
    assert "MTPLX tool contract:" not in rendered_content
    assert "MTPLX coding-agent tool protocol reminder:" not in rendered_content
    assert observability["native_agent_tail_contract_active"] is False


def test_native_tool_prompt_mode_uses_continuation_hint_after_tool_result():
    tokenizer = CaptureTokenizer()
    observability: dict[str, object] = {}

    openai._encode_messages(
        tokenizer,
        [
            openai.ChatMessage(role="user", content="Inspect package.json"),
            openai.ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "call_read",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": '{"filePath":"package.json"}',
                        },
                    }
                ],
            ),
            openai.ChatMessage(
                role="tool",
                tool_call_id="call_read",
                content='{"scripts":{"dev":"vite"}}',
            ),
        ],
        enable_thinking=True,
        add_generation_prompt=True,
        tools=[_bash_tool_schema(), _tool_schema()],
        tool_prompt_mode="native",
        template_observability=observability,
    )

    messages, kwargs = tokenizer.calls[-1]
    rendered_content = "\n".join(str(message.get("content") or "") for message in messages)
    assert kwargs["tools"] == [_bash_tool_schema(), _tool_schema()]
    assert messages[-2]["role"] == "tool"
    assert "MTPLX tool-result continuation:" not in messages[-2]["content"]
    assert messages[-1]["role"] == "user"
    assert "MTPLX tool contract:" not in rendered_content
    assert "MTPLX coding-agent tool protocol reminder:" not in rendered_content
    assert "MTPLX tool-result continuation:" not in messages[-1]["content"]
    assert "Continue the active coding task" in messages[-1]["content"]
    assert "not part of the tool output" in rendered_content
    assert "empty response" in rendered_content
    assert "answer now in normal assistant text" in rendered_content
    assert "read-only inspection" in rendered_content
    assert "extra candidate files only to increase confidence" in rendered_content
    assert "skip recap tables and duplicate summaries" in rendered_content
    assert "tools in a specific order" in rendered_content
    assert "bare checklist number" in rendered_content
    assert observability["native_agent_tail_contract_active"] is False


def test_hybrid_tool_prompt_mode_keeps_legacy_contract_for_rollback():
    tokenizer = CaptureTokenizer()

    openai._encode_messages(
        tokenizer,
        [openai.ChatMessage(role="user", content="Read package.json")],
        enable_thinking=True,
        add_generation_prompt=True,
        tools=[_bash_tool_schema(), _tool_schema()],
        tool_prompt_mode="hybrid",
    )

    messages, kwargs = tokenizer.calls[-1]
    rendered_content = "\n".join(str(message.get("content") or "") for message in messages)
    assert kwargs["tools"] == [_bash_tool_schema(), _tool_schema()]
    assert "MTPLX tool contract:" in rendered_content
    assert "MTPLX coding-agent tool protocol reminder:" in rendered_content
    assert "<function=" in rendered_content
    assert '{"name":' not in rendered_content


def test_compact_tool_prompt_mode_omits_native_template_tools():
    tokenizer = CaptureTokenizer()

    openai._encode_messages(
        tokenizer,
        [openai.ChatMessage(role="user", content="Read package.json")],
        enable_thinking=True,
        add_generation_prompt=True,
        tools=[_bash_tool_schema(), _named_tool_schema("read")],
        tool_prompt_mode="compact",
    )

    messages, kwargs = tokenizer.calls[-1]
    rendered_content = "\n".join(str(message.get("content") or "") for message in messages)
    assert "tools" not in kwargs
    assert "MTPLX tool contract:" in rendered_content
    assert "MTPLX coding-agent tool protocol reminder:" in rendered_content
    assert "<function=" in rendered_content
    assert "bash(command:string, description:string, timeout?:number)" in rendered_content
    assert "read()" in rendered_content


def test_compact_tool_prompt_mode_still_validates_real_tool_schema():
    tools = [_bash_tool_schema(), _named_tool_schema("read")]
    policy = openai._tool_prompt_policy_version_for_request(
        tools_active=True,
        tool_prompt_mode="compact",
        no_tools_contract_active=False,
    )

    assert policy == "compact_tool_contract:schema_free:v1"
    assert openai._tool_contract_active_for_mode(
        tools_active=True,
        tool_prompt_mode="compact",
    )
    assert openai._template_tools_for_prompt_mode(
        tools,
        tool_prompt_mode="compact",
    ) is None
    assert [tool["function"]["name"] for tool in tools] == ["bash", "read"]


def test_froggeric_template_profile_applies_from_vendored_file():
    tokenizer = SimpleNamespace(chat_template="official")
    args = SimpleNamespace(chat_template_profile="froggeric_v19", chat_template_path=None)

    report = openai._apply_chat_template_profile(tokenizer, args)

    assert report["profile"] == "froggeric_v19"
    assert report["source"] == "file"
    assert report["applied"] is True
    assert "Agentic Loop Cure" in tokenizer.chat_template
    assert "<function=example_function_name>" in tokenizer.chat_template


def test_tool_contract_suppresses_agent_tail_for_simple_chitchat():
    with_contract = openai._with_mtplx_tool_contract(
        [{"role": "user", "content": "hi how are you"}],
        tools=[_bash_tool_schema(), _tool_schema()],
    )

    assert "MTPLX tool contract:" in with_contract[0]["content"]
    assert "MTPLX coding-agent tool protocol reminder:" not in with_contract[0]["content"]


def test_filter_tool_specs_preserves_tools_for_simple_chitchat():
    tools = [_bash_tool_schema(), _tool_schema()]

    filtered = openai._filter_tool_specs_for_request(
        tools,
        [openai.ChatMessage(role="user", content="Hi")],
    )

    assert filtered == tools


def test_filter_tool_specs_keeps_only_file_tools_for_static_read_only_review():
    tools = [
        _bash_tool_schema(),
        _write_tool_schema(),
        _named_tool_schema("read"),
        _named_tool_schema("glob"),
        _named_tool_schema("grep"),
        _named_tool_schema("webfetch"),
        _question_tool_schema(),
        _task_tool_schema(),
        _todowrite_tool_schema(),
    ]

    filtered = openai._filter_tool_specs_for_request(
        tools,
        [
            openai.ChatMessage(
                role="user",
                content=(
                    "Evaluate the quality of this project. Inspect the relevant "
                    "files with tools, identify strengths and weaknesses, and "
                    "give a concise improvement plan."
                ),
            )
        ],
    )

    assert [
        tool["function"]["name"]
        for tool in filtered
        if isinstance(tool.get("function"), dict)
    ] == ["bash", "read", "glob", "grep"]


def test_filter_tool_specs_preserves_static_review_tools_after_read_budget(monkeypatch):
    monkeypatch.setenv("MTPLX_READ_ONLY_INSPECTION_FORCE_ANSWER_AFTER_TOOLS", "3")
    tools = [
        _bash_tool_schema(),
        _named_tool_schema("read"),
        _named_tool_schema("glob"),
        _named_tool_schema("grep"),
    ]

    filtered = openai._filter_tool_specs_for_request(
        tools,
        [
            openai.ChatMessage(
                role="user",
                content="Do not edit files. Inspect the project and identify risks.",
            ),
            openai.ChatMessage(role="assistant", content="", tool_calls=[]),
            openai.ChatMessage(role="tool", content="package.json"),
            openai.ChatMessage(role="tool", content="src/main.ts"),
            openai.ChatMessage(role="tool", content="src/game/Game.ts"),
        ],
    )

    assert [
        tool["function"]["name"]
        for tool in filtered
        if isinstance(tool.get("function"), dict)
    ] == ["bash", "read", "glob", "grep"]


def test_static_read_only_inspection_allows_build_config_mentions():
    messages = [
        openai.ChatMessage(
            role="user",
            content=(
                "Do not edit files. Inspect the project deeply. Read the game "
                "loop, player controls, build config, package scripts, and "
                "render-loop risks. Return exactly 3 findings."
            ),
        )
    ]

    assert openai._request_is_static_read_only_inspection(messages) is True


def test_read_only_force_answer_contract_allows_requested_lists():
    contract = openai._mtplx_read_only_force_answer_contract_text()
    user_instruction = openai._mtplx_read_only_force_answer_user_instruction_text()

    assert "tools are intentionally closed" in contract
    assert "exactly that number of items" in contract
    assert "filePath/startLine/endingLine markup" in contract
    assert "planning preambles" in contract
    assert "Markdown lists are allowed" in contract
    assert "No markdown lists" not in contract
    assert "MTPLX read-only final answer instruction:" in user_instruction
    assert "glob> path>" in user_instruction
    assert "<mtplx_final_answer>" in user_instruction
    assert "Begin the assistant content" in user_instruction
    assert "The only valid next assistant turn is the final" in user_instruction


def test_filter_tool_specs_passes_through_when_client_manages_tools():
    """OpenCode curates its toolset per agent mode (plan/build) and enforces
    permissions client-side; the bridge must not content-filter it. Hiding
    write/edit on the build turn (triggered by plan-mode reminder wording in
    history) made the model re-plan files it could not create until it looped,
    and every filter flip rewrote the tool digest bytes and broke banked-prefix
    reuse (2026-07-04 chess-project repro)."""
    tools = [
        _bash_tool_schema(),
        _write_tool_schema(),
        _named_tool_schema("edit"),
        _named_tool_schema("read"),
        _named_tool_schema("glob"),
        _named_tool_schema("grep"),
        _task_tool_schema(),
        _todowrite_tool_schema(),
    ]
    plan_reminder = (
        "1. no\n2. yes\n"
        "<system-reminder>\n# Plan Mode - System Reminder\n\n"
        "CRITICAL: Plan mode ACTIVE - you are in READ-ONLY phase. STRICTLY "
        "FORBIDDEN: ANY file edits, modifications, or system changes. You "
        "MUST NOT make any edits.\n</system-reminder>"
    )

    filtered = openai._filter_tool_specs_for_request(
        tools,
        [openai.ChatMessage(role="user", content=plan_reminder)],
        client_manages_tools=True,
    )

    assert filtered == tools


def test_filter_tool_specs_client_managed_still_honors_explicit_no_tools():
    tools = [_bash_tool_schema(), _named_tool_schema("read")]

    filtered = openai._filter_tool_specs_for_request(
        tools,
        [
            openai.ChatMessage(
                role="user",
                content="Do not use any tools. Answer from memory only.",
            )
        ],
        client_manages_tools=True,
    )

    assert filtered == []


def test_build_mode_reminder_is_not_read_only_for_generic_clients():
    """OpenCode's build-mode switch reminder says 'no longer in read-only
    mode... permitted to make file changes'. The unanchored read-only match
    treated that grant as a read-only instruction and kept write/edit hidden
    exactly on the execute-the-plan turn."""
    build_reminder = (
        "execute plan with care and precision in line with my goals.\n"
        "<system-reminder>\nYour operational mode has changed from plan to build.\n"
        "You are no longer in read-only mode.\n"
        "You are permitted to make file changes, run shell commands, and "
        "utilize your arsenal of tools as needed.\n</system-reminder>"
    )
    messages = [openai.ChatMessage(role="user", content=build_reminder)]

    assert openai._request_disallows_file_mutation(messages) is False

    tools = [
        _bash_tool_schema(),
        _write_tool_schema(),
        _named_tool_schema("edit"),
        _named_tool_schema("read"),
    ]
    filtered = openai._filter_tool_specs_for_request(tools, messages)
    names = [
        tool["function"]["name"]
        for tool in filtered
        if isinstance(tool.get("function"), dict)
    ]
    assert "write" in names
    assert "edit" in names


def test_active_read_only_phase_still_hides_mutating_tools_for_generic_clients():
    messages = [
        openai.ChatMessage(
            role="user",
            content="Plan mode ACTIVE - you are in READ-ONLY phase. Do NOT edit files.",
        )
    ]

    assert openai._request_disallows_file_mutation(messages) is True


def test_filter_tool_specs_keeps_upgrade_recommendations_read_only():
    tools = [
        _bash_tool_schema(),
        _write_tool_schema(),
        _named_tool_schema("read"),
        _named_tool_schema("glob"),
        _named_tool_schema("grep"),
        _named_tool_schema("webfetch"),
        _question_tool_schema(),
        _task_tool_schema(),
        _todowrite_tool_schema(),
    ]

    filtered = openai._filter_tool_specs_for_request(
        tools,
        [
            openai.ChatMessage(
                role="user",
                content="Now, what upgrade do you think I should do?",
            )
        ],
    )

    assert [
        tool["function"]["name"]
        for tool in filtered
        if isinstance(tool.get("function"), dict)
    ] == ["read", "glob", "grep"]


def test_filter_tool_specs_keeps_mutating_tools_for_direct_upgrade_request():
    tools = [
        _bash_tool_schema(),
        _write_tool_schema(),
        _named_tool_schema("read"),
        _named_tool_schema("glob"),
        _named_tool_schema("grep"),
    ]

    filtered = openai._filter_tool_specs_for_request(
        tools,
        [
            openai.ChatMessage(
                role="user",
                content="Upgrade this package to the latest version.",
            )
        ],
    )

    assert filtered == tools


def test_filter_tool_specs_keeps_bash_when_static_review_requests_tests():
    tools = [_bash_tool_schema(), _named_tool_schema("read"), _named_tool_schema("glob")]

    filtered = openai._filter_tool_specs_for_request(
        tools,
        [
            openai.ChatMessage(
                role="user",
                content=(
                    "Evaluate the project, inspect the files, and run the tests "
                    "so the diagnosis is grounded."
                ),
            )
        ],
    )

    assert filtered == tools


def test_filter_tool_specs_hides_file_mutation_tools_when_user_says_no_edits():
    tools = [
        _bash_tool_schema(),
        _write_tool_schema(),
        _named_tool_schema("edit"),
        _named_tool_schema("multi_edit"),
        _named_tool_schema("patch"),
        _named_tool_schema("read"),
        _named_tool_schema("glob"),
        _named_tool_schema("grep"),
        _todowrite_tool_schema(),
    ]

    filtered = openai._filter_tool_specs_for_request(
        tools,
        [
            openai.ChatMessage(
                role="user",
                content=(
                    "Do not edit files. Use shell/bash once to run pwd, then read "
                    "package.json and answer with the scripts."
                ),
            )
        ],
    )

    assert [
        tool["function"]["name"]
        for tool in filtered
        if isinstance(tool.get("function"), dict)
    ] == ["bash", "read"]


def test_filter_tool_specs_keeps_shallow_inventory_in_lean_read_only_lane():
    tools = [
        _bash_tool_schema(),
        _write_tool_schema(),
        _named_tool_schema("edit"),
        _named_tool_schema("read"),
        _named_tool_schema("glob"),
        _named_tool_schema("grep"),
        _todowrite_tool_schema(),
    ]

    filtered = openai._filter_tool_specs_for_request(
        tools,
        [
            openai.ChatMessage(
                role="user",
                content=(
                    "Do not edit files. Use tools to run pwd, list top-level "
                    "files, inspect package.json if it exists, run the safest "
                    "available syntax check, and finish with marker=qa."
                ),
            )
        ],
    )

    assert [
        tool["function"]["name"]
        for tool in filtered
        if isinstance(tool.get("function"), dict)
    ] == ["bash", "read"]


def test_filter_tool_specs_keeps_discovery_tools_for_broad_no_edit_review():
    tools = [
        _bash_tool_schema(),
        _write_tool_schema(),
        _named_tool_schema("edit"),
        _named_tool_schema("read"),
        _named_tool_schema("glob"),
        _named_tool_schema("grep"),
        _named_tool_schema("webfetch"),
        _question_tool_schema(),
        _named_tool_schema("skill"),
        _todowrite_tool_schema(),
    ]

    filtered = openai._filter_tool_specs_for_request(
        tools,
        [
            openai.ChatMessage(
                role="user",
                content=(
                    "Do not edit files. Search the relevant files, run the tests, "
                    "read what matters, then answer with risks."
                ),
            )
        ],
    )

    assert [
        tool["function"]["name"]
        for tool in filtered
        if isinstance(tool.get("function"), dict)
    ] == ["bash", "read", "glob", "grep"]


def test_filter_tool_specs_keeps_web_and_question_when_requested():
    tools = [
        _bash_tool_schema(),
        _write_tool_schema(),
        _named_tool_schema("read"),
        _named_tool_schema("glob"),
        _named_tool_schema("grep"),
        _named_tool_schema("webfetch"),
        _question_tool_schema(),
        _named_tool_schema("skill"),
        _todowrite_tool_schema(),
    ]

    filtered = openai._filter_tool_specs_for_request(
        tools,
        [
            openai.ChatMessage(
                role="user",
                content=(
                    "Do not edit files. Read package.json, check the latest docs "
                    "online with webfetch, ask me a question if the upgrade path "
                    "is ambiguous, and use a skill if one applies."
                ),
            )
        ],
    )

    assert [
        tool["function"]["name"]
        for tool in filtered
        if isinstance(tool.get("function"), dict)
    ] == ["bash", "read", "glob", "grep", "webfetch", "question", "skill"]


def test_opencode_agent_tool_client_uses_compact_prompt_mode(monkeypatch):
    seen: dict[str, object] = {}
    state = _fake_state()
    foreground = ForegroundState()
    state.lock = foreground.lock
    state.has_foreground = foreground.has_foreground
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stats_footer = False
    state.args.tool_prompt_mode = "native"
    client = TestClient(create_app(state))

    def fake_run_generation(*_args, **kwargs):
        seen["request_observability"] = dict(kwargs["request_observability"])
        seen["session_policy_fingerprint"] = kwargs["session_policy_fingerprint"]
        return _fake_generation("Done")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "opencode"},
        json={
            "messages": [
                {"role": "system", "content": "You are a coding agent."},
                {
                    "role": "user",
                    "content": "Do not edit files. Read package.json and summarize it.",
                },
            ],
            "tools": [_bash_tool_schema(), _named_tool_schema("read"), _tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 32,
        },
    )

    assert response.status_code == 200
    messages, kwargs = state.runtime.tokenizer.calls[0]
    rendered = "\n".join(str(message.get("content") or "") for message in messages)
    stats = seen["request_observability"]
    assert "tools" not in kwargs
    # OpenCode toolsets pass through unfiltered (client curates per agent
    # mode); the compact prompt-mode repair is what this test pins.
    assert stats["request_filtered_tool_names"] == [
        "bash",
        "read",
        "session_status",
    ]
    assert "MTPLX tool contract:" in rendered
    assert "read()" in rendered
    assert stats["tool_prompt_mode"] == "compact"
    assert stats["tool_prompt_mode_launch"] == "native"
    assert stats["tool_prompt_mode_client"] == "opencode"
    assert stats["tool_prompt_mode_source"] == "client:opencode"
    assert stats["tool_prompt_mode_client_repaired"] is True
    assert stats["tool_contract_active"] is True
    assert stats["tool_contract_policy_version"] == "compact_tool_contract:schema_free:v1"
    assert "tool_prompt_mode=compact" in seen["session_policy_fingerprint"]


def test_laguna_opencode_keeps_poolside_native_tool_protocol(monkeypatch):
    from mtplx.backends.descriptors import LAGUNA_AR_DESCRIPTOR

    seen: dict[str, object] = {}
    state = _fake_state()
    foreground = ForegroundState()
    state.lock = foreground.lock
    state.has_foreground = foreground.has_foreground
    state.runtime.tokenizer = CaptureTokenizer()
    state.backend_descriptor = LAGUNA_AR_DESCRIPTOR
    state.args.backend_id = "laguna_ar"
    state.args.reasoning_parser = "poolside_v1"
    state.args.tool_prompt_mode = "native"
    state.args.stats_footer = False
    client = TestClient(create_app(state))

    def fake_run_generation(*_args, **kwargs):
        seen["request_observability"] = dict(kwargs["request_observability"])
        return _fake_generation("Done")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "opencode"},
        json={
            "messages": [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "Read package.json."},
            ],
            "tools": [_named_tool_schema("read")],
            "tool_choice": "auto",
            "max_tokens": 32,
        },
    )

    assert response.status_code == 200
    messages, kwargs = state.runtime.tokenizer.calls[0]
    rendered = "\n".join(str(message.get("content") or "") for message in messages)
    stats = seen["request_observability"]
    assert "tools" in kwargs
    assert "Qwen native XML" not in rendered
    assert "<function=" not in rendered
    assert "<parameter=" not in rendered
    assert stats["tool_prompt_mode"] == "native"
    assert stats["tool_prompt_mode_source"] == "backend:laguna_ar"
    assert stats["tool_prompt_mode_client_repaired"] is False


@pytest.mark.parametrize("client_hint", ["pi", "hermes"])
def test_agent_tool_clients_repair_native_launch_mode_to_hybrid(monkeypatch, client_hint):
    seen: dict[str, object] = {}
    state = _fake_state()
    foreground = ForegroundState()
    state.lock = foreground.lock
    state.has_foreground = foreground.has_foreground
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stats_footer = False
    state.args.tool_prompt_mode = "native"
    client = TestClient(create_app(state))

    def fake_run_generation(*_args, **kwargs):
        seen["request_observability"] = dict(kwargs["request_observability"])
        seen["session_policy_fingerprint"] = kwargs["session_policy_fingerprint"]
        return _fake_generation("Done")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": client_hint},
        json={
            "messages": [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "Read package.json and summarize it."},
            ],
            "tools": [_bash_tool_schema(), _tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 32,
        },
    )

    assert response.status_code == 200
    messages, kwargs = state.runtime.tokenizer.calls[0]
    rendered = "\n".join(str(message.get("content") or "") for message in messages)
    stats = seen["request_observability"]
    assert [tool["function"]["name"] for tool in kwargs["tools"]] == [
        "session_status"
    ]
    assert "MTPLX tool contract:" in rendered
    assert stats["tool_prompt_mode"] == "hybrid"
    assert stats["tool_prompt_mode_launch"] == "native"
    assert stats["tool_prompt_mode_client"] == client_hint
    assert stats["tool_prompt_mode_source"] == f"client:{client_hint}"
    assert stats["tool_prompt_mode_client_repaired"] is True
    assert stats["tool_contract_active"] is True
    assert (
        stats["tool_contract_policy_version"].startswith("soft_schema_contract:")
        or stats["tool_contract_policy_version"] == "compact_tool_contract:schema_free:v1"
    )
    assert "tool_prompt_mode=hybrid" in seen["session_policy_fingerprint"]


def test_launch_client_env_labels_headerless_hermes(monkeypatch):
    monkeypatch.setenv("MTPLX_CLIENT", "hermes")

    assert openai._request_client_hint_from_headers({}, {}) == "hermes"
    assert (
        openai._request_client_hint_from_headers(
            {"x-mtplx-client": "opencode"},
            {},
        )
        == "opencode"
    )


def test_opencode_chitchat_preserves_agent_tools_without_direct_reply_contract(
    monkeypatch,
):
    seen: dict[str, object] = {}
    state = _fake_state()
    foreground = ForegroundState()
    state.lock = foreground.lock
    state.has_foreground = foreground.has_foreground
    state.runtime.tokenizer = CaptureTokenizer()
    state.draft_sampler = openai.SamplerConfig(temperature=0.7, top_p=0.95, top_k=20)
    state.args.stats_footer = False
    client = TestClient(create_app(state))

    def fake_run_generation(*_args, **kwargs):
        seen["request_observability"] = dict(kwargs["request_observability"])
        seen["draft_sampler"] = kwargs.get("draft_sampler")
        return _fake_generation("Hi! How can I help?")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "opencode"},
        json={
            "messages": [
                {"role": "system", "content": "You are OpenCode."},
                {"role": "user", "content": "hi"},
            ],
            "tools": [_bash_tool_schema(), _tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 32,
        },
    )

    assert response.status_code == 200
    messages, kwargs = state.runtime.tokenizer.calls[0]
    rendered = "\n".join(str(message.get("content") or "") for message in messages)
    stats = seen["request_observability"]
    assert "tools" not in kwargs
    assert "MTPLX direct reply turn:" not in rendered
    assert "Start with the final user-facing answer" not in rendered
    assert "MTPLX tool contract:" in rendered
    assert "You are OpenCode." in rendered
    assert "bash(command:string" in rendered
    assert "session_status()" in rendered
    assert stats["request_tools_hidden_by_bridge"] is False
    assert stats["request_filtered_tool_names"] == ["bash", "session_status"]
    assert stats["request_hidden_tool_names"] == []
    assert stats["no_tools_contract_active"] is False
    assert (
        stats["tool_contract_policy_version"].startswith("soft_schema_contract:")
        or stats["tool_contract_policy_version"] == "compact_tool_contract:schema_free:v1"
    )
    assert stats["tool_contract_active"] is True
    assert stats["opencode_prompt_contract_profile"] == "opencode_agent"
    assert stats["transcript_replaced_client_system_messages"] == 0
    assert stats["sampler_policy"] == "opencode_default_sampler"
    assert stats["effective_temperature"] == 0.6
    assert stats["effective_top_p"] == 0.95
    assert stats["effective_top_k"] == 20
    assert stats["draft_sampler_policy"] == "launch_default"
    assert stats["draft_sampler_policy_temperature"] == 0.7
    assert seen["draft_sampler"] == openai.SamplerConfig(
        temperature=0.7,
        top_p=0.95,
        top_k=20,
    )


def test_chat_tools_add_no_tool_contract_when_non_chitchat_disables_tools(monkeypatch):
    seen: dict[str, object] = {}
    state = _fake_state()
    foreground = ForegroundState()
    state.lock = foreground.lock
    state.has_foreground = foreground.has_foreground
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stats_footer = False
    client = TestClient(create_app(state))

    def fake_run_generation(*_args, **kwargs):
        seen["request_observability"] = dict(kwargs["request_observability"])
        return _fake_generation("The project looks idle.")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "opencode"},
        json={
            "messages": [
                {"role": "system", "content": "You are OpenCode."},
                {"role": "user", "content": "Summarize the project status without tools."},
            ],
            "tools": [_bash_tool_schema(), _tool_schema()],
            "tool_choice": "none",
            "max_tokens": 32,
        },
    )

    assert response.status_code == 200
    messages, kwargs = state.runtime.tokenizer.calls[0]
    rendered = "\n".join(str(message.get("content") or "") for message in messages)
    stats = seen["request_observability"]
    assert "tools" not in kwargs
    assert "MTPLX direct reply turn:" in rendered
    assert "Start with the final user-facing answer" in rendered
    assert stats["no_tools_contract_active"] is True
    assert stats["tool_contract_policy_version"] == "no_tool_direct_reply:v1"


def test_final_round_after_tools_gets_post_tool_answer_contract(monkeypatch):
    """A tool loop's closing round (tool results in the current turn,
    tool_choice=none) must NOT get the terse direct-reply contract —
    its no-lists/no-analysis clauses clipped searched chat answers to a
    few sentences (2026-07-03). It gets the post-tool full-answer
    contract instead."""
    seen: dict[str, object] = {}
    state = _fake_state()
    foreground = ForegroundState()
    state.lock = foreground.lock
    state.has_foreground = foreground.has_foreground
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stats_footer = False
    client = TestClient(create_app(state))

    def fake_run_generation(*_args, **kwargs):
        seen["request_observability"] = dict(kwargs["request_observability"])
        return _fake_generation("Here is the detailed comparison...")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "What is better, X or Y? Explain in detail.",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": "{\"query\": \"X vs Y\"}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "{\"results\": [{\"title\": \"X vs Y\"}]}",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                }
            ],
            "tool_choice": "none",
            "max_tokens": 32,
        },
    )

    assert response.status_code == 200
    messages, kwargs = state.runtime.tokenizer.calls[0]
    rendered = "\n".join(str(message.get("content") or "") for message in messages)
    stats = seen["request_observability"]
    assert "tools" not in kwargs
    assert "MTPLX post-tool answer turn:" in rendered
    assert "Match the depth the user asked for" in rendered
    # The model must be anchored to today and told fresher tool results
    # outrank its training data.
    assert "Today's date is" in rendered
    assert "trust the tool results" in rendered
    # The clipping clauses must be gone from this round.
    assert "MTPLX direct reply turn:" not in rendered
    assert "one short friendly sentence" not in rendered
    assert "No markdown lists" not in rendered
    assert stats["post_tool_answer_contract_active"] is True
    assert stats["no_tools_contract_active"] is False
    assert stats["tool_contract_policy_version"] == "post_tool_full_answer:dated:v2"


def test_chat_tools_add_no_tool_contract_for_explicit_no_tools_text(monkeypatch):
    seen: dict[str, object] = {}
    state = _fake_state()
    foreground = ForegroundState()
    state.lock = foreground.lock
    state.has_foreground = foreground.has_foreground
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stats_footer = False
    client = TestClient(create_app(state))

    def fake_run_generation(*_args, **kwargs):
        seen["request_observability"] = dict(kwargs["request_observability"])
        return _fake_generation("marker=frontier-final-ok")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "opencode"},
        json={
            "messages": [
                {"role": "system", "content": "You are OpenCode."},
                {
                    "role": "user",
                    "content": "Do not use tools. Reply exact: marker=frontier-final-ok",
                },
            ],
            "tools": [_bash_tool_schema(), _tool_schema(), _named_tool_schema("read")],
            "tool_choice": "auto",
            "max_tokens": 32,
        },
    )

    assert response.status_code == 200
    messages, kwargs = state.runtime.tokenizer.calls[0]
    rendered = "\n".join(str(message.get("content") or "") for message in messages)
    stats = seen["request_observability"]
    assert "tools" not in kwargs
    assert "MTPLX direct reply turn:" in rendered
    assert stats["request_filtered_tool_names"] == []
    assert stats["request_hidden_tool_names"] == ["bash", "session_status", "read"]
    assert stats["no_tools_contract_active"] is True
    assert stats["tool_contract_policy_version"] == "no_tool_direct_reply:v1"


def test_chat_tools_add_read_only_force_answer_contract_after_read_budget(monkeypatch):
    monkeypatch.setenv("MTPLX_READ_ONLY_INSPECTION_FORCE_ANSWER_AFTER_TOOLS", "2")
    seen: dict[str, object] = {}
    state = _fake_state()
    foreground = ForegroundState()
    state.lock = foreground.lock
    state.has_foreground = foreground.has_foreground
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stats_footer = False
    client = TestClient(create_app(state))

    def fake_run_generation(*_args, **kwargs):
        seen["request_observability"] = dict(kwargs["request_observability"])
        return _fake_generation(
            "1. First supported risk.\n2. Second supported risk.\nmarker=done"
        )

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "opencode"},
        json={
            "messages": [
                {"role": "system", "content": "You are OpenCode."},
                {
                    "role": "user",
                    "content": (
                        "Do not edit files. Inspect the project and return exactly "
                        "2 risks with marker=done."
                    ),
                },
                {"role": "tool", "content": "package.json evidence"},
                {"role": "tool", "content": "src/game.ts evidence"},
            ],
            "tools": [
                _bash_tool_schema(),
                _named_tool_schema("read"),
                _named_tool_schema("glob"),
                _named_tool_schema("grep"),
            ],
            "tool_choice": "auto",
            "max_tokens": 64,
        },
    )

    assert response.status_code == 200
    messages, kwargs = state.runtime.tokenizer.calls[0]
    rendered = "\n".join(str(message.get("content") or "") for message in messages)
    stats = seen["request_observability"]
    # PREFIX STABILITY (2026-07-04): the force-answer turn must render the
    # SAME toolset through the SAME template mode as every prior round of the
    # loop — the old hybrid/schema switch rewrote the system prompt and forced
    # a fully cold re-prefill of the largest prompt in the session. The
    # conditioning lives in the appended user contract message only.
    assert "tools" not in kwargs or kwargs["tools"] is None or [
        tool["function"]["name"] for tool in kwargs["tools"]
    ] == ["bash", "read", "glob", "grep"]
    assert "MTPLX read-only answer turn:" in rendered
    assert "MTPLX read-only final answer instruction:" in rendered
    assert "MTPLX direct reply turn:" not in rendered
    assert "No markdown lists" not in rendered
    assert "exactly that number of items" in rendered
    assert messages[0]["role"] == "system"
    assert "MTPLX read-only answer turn:" not in str(messages[0]["content"])
    assert "MTPLX read-only final answer instruction:" not in str(
        messages[0]["content"]
    )
    assert messages[-1]["role"] == "user"
    assert "MTPLX read-only answer turn:" in str(messages[-1]["content"])
    assert "MTPLX read-only final answer instruction:" in str(messages[-1]["content"])
    assert stats["request_read_only_inspection_force_answer"] is True
    assert stats["request_filtered_tool_names"] == ["bash", "read", "glob", "grep"]
    assert stats["no_tools_contract_active"] is False
    assert stats["read_only_force_answer_contract_active"] is True
    assert (
        stats["request_session_restore_policy"]
        == "stable_without_transient_force_answer"
    )
    assert stats["request_session_restore_policy_matches_postcommit"] is True
    # Byte-identity guard: the force-answer request must NOT switch the tool
    # contract lane (compact for opencode) — a different contract version
    # means different system prompt bytes and a broken KV prefix.
    assert stats["tool_contract_policy_version"].startswith("compact_tool_contract:")


def test_explicit_single_tool_then_answer_forces_final_after_tool_result(monkeypatch):
    seen: dict[str, object] = {}
    state = _fake_state()
    foreground = ForegroundState()
    state.lock = foreground.lock
    state.has_foreground = foreground.has_foreground
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stats_footer = False
    state.args.reasoning_parser = "gemma4"
    state.backend_descriptor = openai.descriptor_for_backend_id("gemma4_assistant")
    client = TestClient(create_app(state))

    def fake_run_generation(*_args, **kwargs):
        seen["request_observability"] = dict(kwargs["request_observability"])
        return _fake_generation(
            "opencode_project=/tmp/example; check=package.json"
        )

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "opencode"},
        json={
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Use ls exactly once, then answer exactly: "
                        "opencode_project=<directory>; check=<one filename you saw>."
                    ),
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_ls",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps({"command": "ls"}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_ls",
                    "content": "package.json\nsrc",
                },
            ],
            "tools": [_bash_tool_schema(), _named_tool_schema("read")],
            "tool_choice": "auto",
            "max_tokens": 64,
        },
    )

    assert response.status_code == 200
    messages, kwargs = state.runtime.tokenizer.calls[0]
    rendered = "\n".join(str(message.get("content") or "") for message in messages)
    stats = seen["request_observability"]
    assert "tools" not in kwargs
    assert "MTPLX read-only answer turn:" in rendered
    assert messages[-1]["role"] == "user"
    assert stats["request_read_only_inspection_force_answer"] is True
    assert stats["request_enable_thinking"] is False
    assert stats["request_filtered_tool_names"] == []
    assert stats["tool_contract_policy_version"] == "read_only_force_answer:v1"


def test_opencode_explicit_single_tool_stream_emits_only_first_tool(monkeypatch):
    state = _fake_state()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    generated = (
        "<tool_call>\n<function=bash>\n"
        "<parameter=command>\nls\n</parameter>\n"
        "<parameter=description>\nList files\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=bash>\n"
        "<parameter=command>\npwd\n</parameter>\n"
        "<parameter=description>\nPrint cwd\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation(generated),
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "opencode"},
        json={
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Use ls exactly once, then answer exactly: "
                        "opencode_project=<directory>; check=<one filename you saw>."
                    ),
                }
            ],
            "tools": [_bash_tool_schema(), _named_tool_schema("read")],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 128,
        },
    )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    tool_deltas = [
        item
        for payload in payloads
        for item in payload["choices"][0]["delta"].get("tool_calls", [])
    ]
    arguments = "".join(
        item.get("function", {}).get("arguments", "") for item in tool_deltas
    )
    final = [payload for payload in payloads if payload["choices"][0]["finish_reason"]]
    assert '"command":"ls"' in arguments
    assert "pwd" not in arguments
    assert final[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert final[-1]["mtplx_stats"]["tool_calls_emitted"] == 1
    assert final[-1]["mtplx_stats"]["early_tool_cancel_used"] is True


def test_pi_tool_history_adds_convergence_contract_after_budget(monkeypatch):
    monkeypatch.setenv("MTPLX_PI_CONVERGENCE_AFTER_TOOLS", "2")
    seen: dict[str, object] = {}
    state = _fake_state()
    foreground = ForegroundState()
    state.lock = foreground.lock
    state.has_foreground = foreground.has_foreground
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stats_footer = False
    client = TestClient(create_app(state))

    def fake_run_generation(*_args, **kwargs):
        seen["request_observability"] = dict(kwargs["request_observability"])
        return _fake_generation("marker=pi-converged")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "pi"},
        json={
            "messages": [
                {"role": "system", "content": "You are Pi."},
                {
                    "role": "user",
                    "content": (
                        "Inspect this project, implement the safest useful "
                        "change, and run the relevant check."
                    ),
                },
                {"role": "tool", "content": "package.json evidence"},
                {"role": "tool", "content": "src/game.ts evidence"},
            ],
            "tools": [
                _bash_tool_schema(),
                _named_tool_schema("read"),
                _named_tool_schema("edit"),
                _named_tool_schema("write"),
                _named_tool_schema("grep"),
                _named_tool_schema("find"),
                _named_tool_schema("ls"),
            ],
            "tool_choice": "auto",
            "max_tokens": 64,
        },
    )

    assert response.status_code == 200
    messages, kwargs = state.runtime.tokenizer.calls[0]
    rendered = "\n".join(str(message.get("content") or "") for message in messages)
    stats = seen["request_observability"]
    assert "tools" in kwargs
    assert "MTPLX Pi convergence turn:" in rendered
    assert "MTPLX Pi convergence instruction:" in rendered
    assert "A single targeted read" in rendered
    assert "only one narrow line-range refresh" in rendered
    assert messages[-1]["role"] == "user"
    # PREFIX STABILITY (2026-07-04): the convergence contract activates
    # MID-SESSION (tool-count budget), so it must never rewrite earlier
    # transcript bytes — the system message stays untouched and the whole
    # contract travels in the appended user message. The old system-append
    # invalidated every banked KV prefix at the transition round.
    assert messages[0]["role"] == "system"
    assert "MTPLX Pi convergence turn:" not in str(messages[0]["content"])
    assert "MTPLX Pi convergence turn:" in str(messages[-1]["content"])
    assert "MTPLX Pi convergence instruction:" in str(messages[-1]["content"])
    assert stats["request_pi_convergence_contract"] is True
    assert stats["request_pi_convergence_tool_result_count"] == 2
    assert stats["request_pi_convergence_after_tools"] == 2
    assert stats["pi_convergence_contract_active"] is True
    assert (
        stats["request_session_restore_policy"]
        == "stable_without_transient_pi_convergence"
    )
    assert stats["request_session_restore_policy_matches_postcommit"] is True
    assert stats["request_filtered_tool_names"] == [
        "bash",
        "read",
        "edit",
        "write",
        "grep",
        "find",
        "ls",
    ]
    assert stats["tool_contract_policy_version"].endswith("+pi_convergence:v1")


def test_filter_tool_specs_preserves_tools_for_loose_simple_chitchat():
    tools = [_bash_tool_schema(), _tool_schema()]

    filtered = openai._filter_tool_specs_for_request(
        tools,
        [openai.ChatMessage(role="user", content="hey how you")],
    )

    assert filtered == tools


def test_filter_tool_specs_preserves_tools_for_concatenated_simple_chitchat():
    tools = [_bash_tool_schema(), _tool_schema()]

    for content in ("hihi", "Hi, how are you?Hi, how are you?"):
        filtered = openai._filter_tool_specs_for_request(
            tools,
            [openai.ChatMessage(role="user", content=content)],
        )
        assert filtered == tools


def test_filter_tool_specs_drops_tools_when_user_disallows_tools():
    tools = [_bash_tool_schema(), _tool_schema(), _named_tool_schema("read")]

    filtered = openai._filter_tool_specs_for_request(
        tools,
        [
            openai.ChatMessage(
                role="user",
                content="Do not use tools. Reply exact: marker=frontier-final-ok",
            )
        ],
    )

    assert filtered == []


def test_filter_tool_specs_honors_no_tools_exception():
    tools = [_bash_tool_schema(), _tool_schema(), _named_tool_schema("read")]

    filtered = openai._filter_tool_specs_for_request(
        tools,
        [
            openai.ChatMessage(
                role="user",
                content="Do not use tools except read package.json.",
            )
        ],
    )

    assert filtered == tools


def test_filter_tool_specs_keeps_for_explicit_tool_choice():
    tools = [_bash_tool_schema(), _tool_schema()]

    filtered = openai._filter_tool_specs_for_request(
        tools,
        [openai.ChatMessage(role="user", content="Hi")],
        tool_choice={"type": "function", "function": {"name": "bash"}},
    )

    assert filtered == tools


def test_tool_contract_keeps_system_first_for_qwen_template():
    with_contract = openai._with_mtplx_tool_contract(
        [
            {"role": "system", "content": "You are OpenCode."},
            {"role": "user", "content": "status?"},
        ],
        tools=[_bash_tool_schema(), _tool_schema()],
    )

    assert [message["role"] for message in with_contract] == ["system", "user"]
    assert with_contract[0]["content"].startswith("You are OpenCode.")
    assert "MTPLX tool contract:" in with_contract[0]["content"]
    assert "MTPLX coding-agent tool protocol reminder:" in with_contract[0]["content"]


def test_tool_contract_honors_forced_function_choice():
    with_contract = openai._with_mtplx_tool_contract(
        [{"role": "user", "content": "Solve it."}],
        tools=[_named_tool_schema("submit_answer")],
        tool_choice={"type": "function", "function": {"name": "submit_answer"}},
    )

    assert "requires the `submit_answer` tool call" in with_contract[0]["content"]
    assert "instead of a normal text answer" in with_contract[0]["content"]


def test_tool_contract_adds_post_tool_continuation_hint_only_after_tool_result():
    tools = [_bash_tool_schema(), _tool_schema()]

    initial = openai._with_mtplx_tool_contract(
        [
            {"role": "system", "content": "You are OpenCode."},
            {"role": "user", "content": "Read package.json"},
        ],
        tools=tools,
    )
    resumed = openai._with_mtplx_tool_contract(
        [
            {"role": "system", "content": "You are OpenCode."},
            {"role": "user", "content": "Read package.json"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_glob",
                        "type": "function",
                        "function": {"name": "glob", "arguments": {"pattern": "**/*"}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_glob",
                "content": "package.json\nnode_modules/vite/package.json",
            },
        ],
        tools=tools,
    )

    assert "MTPLX tool-result continuation:" not in initial[0]["content"]
    assert "MTPLX tool-result continuation:" not in initial[-1]["content"]
    assert resumed[-2]["role"] == "tool"
    assert "MTPLX tool-result continuation:" not in resumed[-2]["content"]
    assert resumed[-1]["role"] == "user"
    assert "MTPLX tool-result continuation:" not in resumed[-1]["content"]
    assert "<mtplx_tool_result_continuation>" not in resumed[-1]["content"]
    assert "</mtplx_tool_result_continuation>" not in resumed[-1]["content"]
    assert "Continue the active coding task" in resumed[-1]["content"]
    assert "not part of the tool output" in resumed[-1]["content"]
    assert "answer now in normal assistant text" in resumed[-1]["content"]
    assert "read-only inspection" in resumed[-1]["content"]
    assert "one of several files" in resumed[-1]["content"]
    assert "commands, files, checks" in resumed[-1]["content"]
    assert "bare checklist number" in resumed[-1]["content"]
    assert "empty response" in resumed[-1]["content"]
    assert "MTPLX coding-agent tool protocol reminder:" in resumed[0]["content"]


def test_internal_continuation_marker_is_stripped_from_visible_text():
    leaked = (
        "Let me read the relevant portions.\n"
        "</mtplx_tool_result_continuation>\n"
        "MTPLX tool-result continuation: Internal MTPLX continuation note."
    )

    assert (
        openai._strip_mtplx_internal_continuation_markers(leaked).strip()
        == "Let me read the relevant portions."
    )


def test_read_only_force_answer_marker_is_stripped_from_visible_text():
    leaked = (
        "MTPLX read-only final answer instruction: This read-only inspection "
        "is now closed to more tools.\n"
        "1. No mobile input.\nmarker=fa2"
    )

    assert (
        openai._strip_mtplx_internal_continuation_markers(leaked).strip()
        == "1. No mobile input.\nmarker=fa2"
    )


def test_read_only_force_answer_stream_marker_is_stripped_from_visible_text():
    generated = (
        "Let me rehearse this first.\n"
        "<mtplx_final_answer>\n"
        "QUALITY: the highest-risk issue is lifecycle cleanup.\n"
        "</mtplx_final_answer>\n</think>"
    )

    visible, stripped_chars = openai._read_only_force_answer_visible_text(generated)

    assert stripped_chars > 0
    assert visible == "QUALITY: the highest-risk issue is lifecycle cleanup."
    assert "<mtplx_final_answer>" not in visible
    assert "</mtplx_final_answer>" not in visible
    assert "</think>" not in visible
    assert "rehearse" not in visible


def test_stalled_agent_tail_detects_tool_promise_without_tool_call():
    stalled = (
        "Good, the file was rewritten.\n\n"
        "Actually, I should verify the Arrow.ts calls to make sure they're "
        "compatible with the new signatures. Let me read the relevant portions."
    )
    final = (
        "I checked the particle system and the highest-value fix is complete: "
        "the project now uses a pooled particle system with shared geometry."
    )

    assert openai._looks_like_stalled_agent_tool_promise(stalled) is True
    assert openai._looks_like_stalled_agent_tool_promise(final) is False


def test_read_only_force_answer_failure_detects_toolish_drafts():
    toolish = (
        "Let me read the critical sections first.\n\n"
        "filePath> src/game/Game.ts\nstartingLine> 1\nendingLine> 80"
    )
    final = (
        "1. The camera update is the highest-risk path because it owns target "
        "tracking.\n2. HUD state is duplicated across two files.\nmarker=review"
    )

    assert openai._looks_like_read_only_force_answer_failure(toolish) is True
    assert openai._looks_like_read_only_force_answer_failure(final) is False


def test_read_only_force_answer_visible_text_strips_rehearsal_before_final():
    generated = (
        "The user asked for exactly three findings. Let me walk through the "
        "evidence first.\n\n"
        "1. Game loop: requestAnimationFrame is present.\n"
        "2. Player controls: keyboard-only.\n"
        "3. HUD: restart exists.\n"
        "4. Audio: no files found.\n"
        "5. Physics: no engine found.\n\n"
        "1. No mobile/touch input — keyboard-only controls.\n\n"
        "2. Restart UX is broken — gameOverCallback fires only once.\n\n"
        "3. No physics engine — manual collision is fragile.\n\n"
        "marker=fa5"
    )

    visible, stripped_chars = openai._read_only_force_answer_visible_text(generated)

    assert stripped_chars > 0
    assert visible.startswith("1. No mobile/touch input")
    assert "4. Audio" not in visible
    assert "marker=fa5" in visible


def test_merge_final_metrics_keeps_read_only_buffer_observability():
    state = _fake_state()

    openai._merge_final_bridge_stats_into_latest_metrics(
        state,
        {
            "read_only_force_answer_buffered_stream": True,
            "read_only_force_answer_visible_prefix_stripped_chars": 512,
            "read_only_force_answer_visible_tokens": 231,
        },
    )

    latest = state.last_metrics[-1]
    assert latest["read_only_force_answer_buffered_stream"] is True
    assert latest["read_only_force_answer_visible_prefix_stripped_chars"] == 512
    assert latest["read_only_force_answer_visible_tokens"] == 231


def test_read_only_force_answer_stream_starts_after_internal_marker(monkeypatch):
    monkeypatch.setenv("MTPLX_READ_ONLY_INSPECTION_FORCE_ANSWER_AFTER_TOOLS", "2")
    state = _fake_state()
    foreground = ForegroundState()
    state.lock = foreground.lock
    state.has_foreground = foreground.has_foreground
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    client = TestClient(create_app(state))

    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation(
            "Let me rehearse the findings first.\n"
            "<mtplx_final_answer>\n"
            "QUALITY1321: final answer from gathered evidence."
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "opencode"},
        json={
            "messages": [
                {"role": "system", "content": "You are OpenCode."},
                {
                    "role": "user",
                    "content": (
                        "Do not edit files. Inspect the project and return one "
                        "release risk."
                    ),
                },
                {"role": "tool", "content": "package.json evidence"},
                {"role": "tool", "content": "src/game.ts evidence"},
            ],
            "tools": [
                _bash_tool_schema(),
                _named_tool_schema("read"),
                _named_tool_schema("glob"),
                _named_tool_schema("grep"),
            ],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 128,
        },
    )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    content = "".join(
        payload["choices"][0]["delta"].get("content", "")
        for payload in payloads
    )
    final = [payload for payload in payloads if payload["choices"][0]["finish_reason"]]
    assert "QUALITY1321: final answer from gathered evidence." in content
    assert "rehearse" not in content
    assert "<mtplx_final_answer>" not in content
    stats = final[-1]["mtplx_stats"]
    assert stats["finish_reason"] == "stop"
    assert stats["read_only_force_answer_buffered_stream"] is True
    assert stats["read_only_force_answer_marker_stream_started"] is True
    assert stats["read_only_force_answer_visible_tokens"] > 0


def test_read_only_force_answer_stream_fallback_emits_without_marker(monkeypatch):
    monkeypatch.setenv("MTPLX_READ_ONLY_INSPECTION_FORCE_ANSWER_AFTER_TOOLS", "2")
    state = _fake_state()
    foreground = ForegroundState()
    state.lock = foreground.lock
    state.has_foreground = foreground.has_foreground
    state.runtime.tokenizer = CaptureTokenizer()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    client = TestClient(create_app(state))

    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation(
            "Let me rehearse the findings first.\n"
            "SPEEDQA0606: final answer from gathered evidence.",
            finish_reason=None,
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass", "x-mtplx-client": "opencode"},
        json={
            "messages": [
                {"role": "system", "content": "You are OpenCode."},
                {
                    "role": "user",
                    "content": (
                        "Do not edit files. Inspect the project and return one "
                        "release risk."
                    ),
                },
                {"role": "tool", "content": "package.json evidence"},
                {"role": "tool", "content": "src/game.ts evidence"},
            ],
            "tools": [
                _bash_tool_schema(),
                _named_tool_schema("read"),
                _named_tool_schema("glob"),
                _named_tool_schema("grep"),
            ],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 128,
        },
    )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    content = "".join(
        payload["choices"][0]["delta"].get("content", "")
        for payload in payloads
    )
    final = [payload for payload in payloads if payload["choices"][0]["finish_reason"]]
    assert final[-1]["choices"][0]["finish_reason"] == "stop"
    assert "SPEEDQA0606: final answer from gathered evidence." in content
    assert "rehearse" not in content
    stats = final[-1]["mtplx_stats"]
    assert stats["finish_reason"] == "stop"
    assert stats["read_only_force_answer_buffered_stream"] is True
    assert stats["read_only_force_answer_marker_stream_started"] is True
    assert stats["read_only_force_answer_stream_marker_stripped_chars"] == 0
    assert stats["read_only_force_answer_visible_prefix_stripped_chars"] > 0


def test_read_only_force_answer_stream_postcommit_uses_client_history(monkeypatch):
    monkeypatch.setenv("MTPLX_READ_ONLY_INSPECTION_FORCE_ANSWER_AFTER_TOOLS", "2")
    state = _fake_streaming_session_state()
    state.args.stream_interval = 1
    captured: dict[str, object] = {}
    client = TestClient(create_app(state))

    def fake_generation_final(*_args, **kwargs):
        captured["generation_final_messages"] = list(kwargs["messages"])
        captured["generation_final_tool_prompt_mode"] = kwargs["tool_prompt_mode"]
        captured["generation_final_tool_specs"] = list(kwargs["tool_specs"] or [])
        captured["generation_final_policy_fingerprint"] = kwargs["policy_fingerprint"]
        return {
            "stored": False,
            "mode": "unsafe",
            "reason": "stop_token_boundary_mismatch",
        }

    def fake_schedule(_state, **kwargs):
        captured["scheduled_messages"] = list(kwargs["messages"])
        captured["scheduled_tool_prompt_mode"] = kwargs["tool_prompt_mode"]
        captured["scheduled_tool_specs"] = list(kwargs["tool_specs"] or [])
        captured["scheduled_policy_fingerprint"] = kwargs["policy_fingerprint"]
        return {
            "stored": False,
            "mode": "async_pending",
            "reason": kwargs["unsafe_reason"],
        }

    generated_text = (
        "Let me rehearse the findings first.\n"
        "SPEEDQA0606: final answer from gathered evidence."
    )
    generated_tokens = [ord(char) for char in generated_text]

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        captured["request_observability"] = dict(kwargs["request_observability"])
        captured["generation_session_policy_fingerprint"] = kwargs[
            "session_policy_fingerprint"
        ]
        token_callback = kwargs.get("token_callback")
        if token_callback is not None:
            for token in generated_tokens:
                token_callback([token])
        return {
            "text": generated_text,
            "tokens": generated_tokens,
            "stats": {
                **kwargs["request_observability"],
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": len(generated_tokens),
            },
            "prompt_tokens": 3,
            "completion_tokens": len(generated_tokens),
            "finish_reason": None,
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)
    monkeypatch.setattr(
        openai,
        "_store_generation_final_history_snapshot",
        fake_generation_final,
    )
    monkeypatch.setattr(openai, "_schedule_idle_postcommit_snapshot", fake_schedule)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-client": "opencode", "x-mtplx-session-id": "speedqa0606"},
        json={
            "messages": [
                {"role": "system", "content": "You are OpenCode."},
                {
                    "role": "user",
                    "content": (
                        "Do not edit files. Inspect the project and return one "
                        "release risk."
                    ),
                },
                {"role": "tool", "content": "package.json evidence"},
                {"role": "tool", "content": "src/game.ts evidence"},
            ],
            "tools": [
                _bash_tool_schema(),
                _named_tool_schema("read"),
                _named_tool_schema("glob"),
                _named_tool_schema("grep"),
            ],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 128,
        },
    )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    final = [payload for payload in payloads if payload["choices"][0]["finish_reason"]]
    stats = final[-1]["mtplx_stats"]
    generation_final_text = "\n".join(
        str(message.content or "") for message in captured["generation_final_messages"]
    )
    scheduled_text = "\n".join(
        str(message.content or "") for message in captured["scheduled_messages"]
    )
    assert "MTPLX read-only final answer instruction" not in generation_final_text
    assert "MTPLX read-only final answer instruction" not in scheduled_text
    assert "Inspect the project and return one release risk" in scheduled_text
    assert [
        tool["function"]["name"] for tool in captured["generation_final_tool_specs"]
    ] == ["bash", "read", "glob", "grep"]
    assert [tool["function"]["name"] for tool in captured["scheduled_tool_specs"]] == [
        "bash",
        "read",
        "glob",
        "grep",
    ]
    assert (
        "read_only_force_answer_contract=0"
        in captured["generation_final_policy_fingerprint"]
    )
    assert (
        "tool_prompt_mode=compact"
        in captured["generation_final_policy_fingerprint"]
    )
    assert (
        "tool_contract=compact_tool_contract:schema_free:v1"
        in captured["generation_final_policy_fingerprint"]
    )
    assert "read_only_force_answer:v1" not in captured[
        "generation_final_policy_fingerprint"
    ]
    assert (
        "read_only_force_answer_contract=0"
        in captured["generation_session_policy_fingerprint"]
    )
    assert "read_only_force_answer:v1" not in captured[
        "generation_session_policy_fingerprint"
    ]
    assert (
        "read_only_force_answer_contract=0"
        in captured["scheduled_policy_fingerprint"]
    )
    assert "tool_prompt_mode=compact" in captured["scheduled_policy_fingerprint"]
    assert (
        "tool_contract=compact_tool_contract:schema_free:v1"
        in captured["scheduled_policy_fingerprint"]
    )
    assert "read_only_force_answer:v1" not in captured["scheduled_policy_fingerprint"]
    assert captured["generation_final_tool_prompt_mode"] == "compact"
    assert captured["scheduled_tool_prompt_mode"] == "compact"
    assert captured["request_observability"]["request_commit_prompt_prefix"] is False
    assert stats["session_prompt_prefix_commit"] == {
        "committed": False,
        "reason": "transient_generation_contract",
        "prefix_len": 0,
        "boundary_kind": "postcommit_prompt_prefix",
    }


def test_opencode_chitchat_sampler_uses_launched_defaults():
    observability = {"request_client_hint": "opencode"}

    sampler = openai._opencode_default_sampler_override(
        messages=[
            openai.ChatMessage(role="system", content="You are OpenCode."),
            openai.ChatMessage(role="user", content="Hi, how are you"),
        ],
        tools_active=True,
        request_temperature=None,
        request_top_p=None,
        request_top_k=None,
        request_observability=observability,
        default_temperature=0.6,
        default_top_p=0.95,
        default_top_k=20,
    )

    assert sampler == openai.SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
    assert observability["sampler_policy"] == "opencode_default_sampler"
    assert observability["sampler_policy_request_temperature"] is None
    assert observability["sampler_policy_temperature"] == 0.6
    assert observability["sampler_policy_top_p"] == 0.95
    assert observability["sampler_policy_top_k"] == 20


def test_opencode_chitchat_sampler_normalizes_client_defaults():
    observability = {"request_client_hint": "opencode"}

    sampler = openai._opencode_default_sampler_override(
        messages=[
            openai.ChatMessage(role="system", content="You are OpenCode."),
            openai.ChatMessage(role="user", content="Hi, how are you"),
        ],
        tools_active=True,
        request_temperature=0.55,
        request_top_p=1.0,
        request_top_k=None,
        request_observability=observability,
        default_temperature=0.6,
        default_top_p=0.95,
        default_top_k=20,
    )

    assert sampler == openai.SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
    assert observability["sampler_policy"] == "opencode_default_sampler"
    assert observability["sampler_policy_request_temperature"] == 0.55


def test_opencode_chitchat_sampler_uses_launched_top_p():
    observability = {"request_client_hint": "opencode"}

    sampler = openai._opencode_default_sampler_override(
        messages=[
            openai.ChatMessage(role="system", content="You are OpenCode."),
            openai.ChatMessage(role="user", content="How are you?"),
        ],
        tools_active=False,
        request_temperature=0.55,
        request_top_p=1.0,
        request_top_k=None,
        request_observability=observability,
        default_temperature=0.6,
        default_top_p=1.0,
        default_top_k=20,
    )

    assert sampler == openai.SamplerConfig(temperature=0.6, top_p=1.0, top_k=20)
    assert observability["sampler_policy"] == "opencode_default_sampler"
    assert observability["sampler_policy_top_p"] == 1.0


def test_opencode_chitchat_sampler_accepts_app_owned_top_k():
    observability = {"request_client_hint": "opencode"}

    sampler = openai._opencode_default_sampler_override(
        messages=[
            openai.ChatMessage(role="system", content="You are OpenCode."),
            openai.ChatMessage(role="user", content="How are you?"),
        ],
        tools_active=False,
        request_temperature=0.55,
        request_top_p=1.0,
        request_top_k=20,
        request_observability=observability,
        default_temperature=0.6,
        default_top_p=1.0,
        default_top_k=20,
    )

    assert sampler == openai.SamplerConfig(temperature=0.6, top_p=1.0, top_k=20)
    assert observability["sampler_policy"] == "opencode_default_sampler"


def test_opencode_chitchat_sampler_runs_with_tools_active():
    observability = {"request_client_hint": "opencode"}

    sampler = openai._opencode_default_sampler_override(
        messages=[
            openai.ChatMessage(role="system", content="You are OpenCode."),
            openai.ChatMessage(role="user", content="hey"),
        ],
        tools_active=True,
        request_temperature=0.55,
        request_top_p=1.0,
        request_top_k=None,
        request_observability=observability,
        default_temperature=0.6,
        default_top_p=0.95,
        default_top_k=20,
    )

    assert sampler == openai.SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
    assert observability["sampler_policy"] == "opencode_default_sampler"


def test_opencode_default_sampler_uses_launched_defaults_for_agent_turns():
    observability = {"request_client_hint": "opencode"}

    sampler = openai._opencode_default_sampler_override(
        messages=[
            openai.ChatMessage(role="system", content="You are OpenCode."),
            openai.ChatMessage(role="user", content="Read package.json"),
        ],
        tools_active=True,
        request_temperature=0.55,
        request_top_p=1.0,
        request_top_k=None,
        request_observability=observability,
        default_temperature=0.6,
        default_top_p=0.95,
        default_top_k=20,
    )

    assert sampler == openai.SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
    assert observability["sampler_policy"] == "opencode_default_sampler"
    assert observability["sampler_policy_request_top_p"] == 1.0
    assert observability["sampler_policy_temperature"] == 0.6
    assert observability["sampler_policy_top_p"] == 0.95


def test_opencode_agent_sampler_keeps_app_owned_top_p_one():
    observability = {"request_client_hint": "opencode"}

    sampler = openai._opencode_default_sampler_override(
        messages=[
            openai.ChatMessage(role="system", content="You are OpenCode."),
            openai.ChatMessage(role="user", content="Read package.json"),
        ],
        tools_active=True,
        request_temperature=0.55,
        request_top_p=1.0,
        request_top_k=None,
        request_observability=observability,
        default_temperature=0.6,
        default_top_p=1.0,
        default_top_k=20,
    )

    assert sampler == openai.SamplerConfig(temperature=0.6, top_p=1.0, top_k=20)
    assert observability["sampler_policy"] == "opencode_default_sampler"
    assert observability["sampler_policy_top_p"] == 1.0


def test_opencode_default_sampler_does_not_touch_explicit_tool_sampler():
    observability = {"request_client_hint": "opencode"}

    sampler = openai._opencode_default_sampler_override(
        messages=[
            openai.ChatMessage(role="system", content="You are OpenCode."),
            openai.ChatMessage(role="user", content="Read package.json"),
        ],
        tools_active=True,
        request_temperature=None,
        request_top_p=0.95,
        request_top_k=20,
        request_observability=observability,
        default_temperature=0.6,
        default_top_p=0.95,
        default_top_k=20,
    )

    assert sampler is None
    assert "sampler_policy" not in observability


def test_tool_result_history_encoding_keeps_generation_prompt_prefix():
    tokenizer = QwenToolHistoryBoundaryTokenizer()
    tools = [_tool_schema()]
    initial_messages = [
        openai.ChatMessage(role="system", content="You are opencode."),
        openai.ChatMessage(role="user", content="write hello.py"),
    ]
    tool_call = {
        "id": "call_write",
        "type": "function",
        "function": {
            "name": "session_status",
            "arguments": "{}",
        },
    }
    resumed_messages = [
        *initial_messages,
        openai.ChatMessage(role="assistant", content="", tool_calls=[tool_call]),
        openai.ChatMessage(
            role="tool",
            tool_call_id="call_write",
            content='{"status":"ok"}',
        ),
    ]

    first_prompt = openai._encode_messages(
        tokenizer,
        initial_messages,
        enable_thinking=True,
        tools=tools,
    )
    resumed_prompt = openai._encode_messages(
        tokenizer,
        resumed_messages,
        enable_thinking=True,
        tools=tools,
    )

    assert resumed_prompt[: len(first_prompt)] == first_prompt
    assert tokenizer.merged not in resumed_prompt[: len(first_prompt)]
    rendered = tokenizer.decode(resumed_prompt)
    assert "<function=session_status>" in rendered
    assert "<tool_response>" in rendered


def test_postcommit_tool_history_encoding_matches_next_tool_turn_prefix():
    tokenizer = QwenToolHistoryBoundaryTokenizer()
    tools = [_tool_schema()]
    initial_messages = [
        openai.ChatMessage(role="system", content="You are opencode."),
        openai.ChatMessage(role="user", content="write hello.py"),
    ]
    tool_call = {
        "id": "call_write",
        "type": "function",
        "function": {
            "name": "session_status",
            "arguments": "{}",
        },
    }
    history_messages = [
        *initial_messages,
        openai.ChatMessage(role="assistant", content="", tool_calls=[tool_call]),
    ]
    resumed_messages = [
        *history_messages,
        openai.ChatMessage(
            role="tool",
            tool_call_id="call_write",
            content='{"status":"ok"}',
        ),
    ]

    postcommit_prefix = openai._postcommit_next_turn_prefix_ids(
        tokenizer,
        history_messages,
        enable_thinking=True,
        strip_assistant_reasoning_history=False,
        tools=tools,
        assistant_tool_calls=[tool_call],
    )
    resumed_prompt = openai._encode_messages(
        tokenizer,
        resumed_messages,
        enable_thinking=True,
        tools=tools,
    )

    assert postcommit_prefix is not None
    assert resumed_prompt[: len(postcommit_prefix)] == postcommit_prefix
    assert tokenizer.merged not in postcommit_prefix


def test_postcommit_plain_final_answer_preserves_prior_tool_history_boundaries():
    tokenizer = QwenToolHistoryBoundaryTokenizer()
    tools = [_tool_schema()]
    tool_call = {
        "id": "call_write",
        "type": "function",
        "function": {
            "name": "session_status",
            "arguments": "{}",
        },
    }
    messages_for_generation = [
        openai.ChatMessage(role="system", content="You are opencode."),
        openai.ChatMessage(role="user", content="find the power meter"),
        openai.ChatMessage(role="assistant", content="", tool_calls=[tool_call]),
        openai.ChatMessage(
            role="tool",
            tool_call_id="call_write",
            content='{"status":"HUD.ts:84"}',
        ),
    ]

    generation_prompt = openai._encode_messages(
        tokenizer,
        messages_for_generation,
        enable_thinking=True,
        tools=tools,
    )
    postcommit_prefix = openai._postcommit_next_turn_prefix_ids(
        tokenizer,
        [
            *messages_for_generation,
            openai.ChatMessage(
                role="assistant",
                content="The power meter is implemented in HUD.ts.",
            ),
        ],
        enable_thinking=True,
        strip_assistant_reasoning_history=False,
        tools=tools,
        assistant_tool_calls=None,
    )

    assert postcommit_prefix is not None
    assert postcommit_prefix[: len(generation_prompt)] == generation_prompt
    assert tokenizer.merged not in postcommit_prefix[: len(generation_prompt)]

    next_turn_prompt = openai._encode_messages(
        tokenizer,
        [
            *messages_for_generation,
            openai.ChatMessage(
                role="assistant",
                content="The power meter is implemented in HUD.ts.",
            ),
            openai.ChatMessage(role="user", content="repeat the line range"),
        ],
        enable_thinking=True,
        tools=tools,
    )

    assert next_turn_prompt[: len(postcommit_prefix)] == postcommit_prefix


def test_postcommit_recanonicalizes_raw_active_read_as_next_turn_history():
    tokenizer = QwenToolHistoryBoundaryTokenizer()
    tools = [_tool_schema()]
    tool_call = {
        "id": "call_read",
        "type": "function",
        "function": {
            "name": "session_status",
            "arguments": {},
        },
    }
    body_lines = [
        "1: import * as THREE from 'three';",
        "2: import { Arrow } from './Arrow';",
    ]
    for line_no in range(3, 380):
        if line_no == 240:
            body_lines.append("240:   public checkArrowCollisions(arrow: Arrow): void {")
        elif line_no == 252:
            body_lines.append("252:       if (dist < hitRadius) {")
        elif line_no == 253:
            body_lines.append("253:         arrow.embedInTerrain(obstacle.position.y - 0.5);")
        else:
            body_lines.append(
                f"{line_no}:     filler line {line_no} with enough body text to "
                "represent a real full-file OpenCode read payload;"
            )
    large_read_output = (
        "<path>src/game/ObstacleManager.ts</path>\n"
        "<type>file</type>\n"
        "<content>\n"
        + "\n".join(body_lines)
        + "\n</content>"
    )
    raw_messages = [
        openai.ChatMessage(role="system", content="You are opencode."),
        openai.ChatMessage(role="user", content="find arrow collision handling"),
        openai.ChatMessage(role="assistant", content="", tool_calls=[tool_call]),
        openai.ChatMessage(
            role="tool",
            tool_call_id="call_read",
            content=large_read_output,
        ),
    ]
    assistant_answer = "Obstacle collision is in ObstacleManager.ts lines 240-253."

    current_canonical, current_stats = openai._canonicalize_agent_transcript(
        raw_messages,
        tools_active=True,
    )
    assert current_stats.compacted_active_read_messages == 1
    assert current_stats.compacted_tool_result_messages == 0
    assert "<mtplx_compacted_active_read_output" in str(current_canonical[-1].content)

    next_canonical, next_stats = openai._canonicalize_agent_transcript(
        [
            *raw_messages,
            openai.ChatMessage(role="assistant", content=assistant_answer),
            openai.ChatMessage(role="user", content="repeat the line range"),
        ],
        tools_active=True,
    )
    assert next_stats.compacted_tool_result_messages == 1
    assert next_stats.compacted_active_read_messages == 0

    state = SimpleNamespace(
        runtime=SimpleNamespace(tokenizer=tokenizer),
        args=parse_args(["--warmup-tokens", "0"]),
    )
    postcommit_prefix, postcommit_vision_splice = openai._history_ids_for_postcommit(
        state,
        messages=raw_messages,
        assistant_content=assistant_answer,
        assistant_tool_calls=None,
        thinking_enabled=True,
        tool_specs=tools,
    )
    next_turn_prompt = openai._encode_messages(
        tokenizer,
        next_canonical,
        enable_thinking=True,
        tools=tools,
    )

    assert postcommit_vision_splice is None
    assert next_turn_prompt[: len(postcommit_prefix)] == postcommit_prefix
    rendered_prefix = tokenizer.decode(postcommit_prefix)
    assert "<mtplx_compacted_tool_output" in rendered_prefix
    assert "<mtplx_compacted_active_read_output" not in rendered_prefix


def test_postcommit_read_only_final_matches_next_turn_history_boundary():
    tokenizer = QwenToolHistoryBoundaryTokenizer()
    tools = [_named_tool_schema("read")]
    tool_call = {
        "id": "call_read_game",
        "type": "function",
        "function": {
            "name": "read",
            "arguments": {"filePath": "src/game/Game.ts"},
        },
    }
    body_lines = [
        "1: import { Player } from './Player';",
        "2: import { HUD } from './HUD';",
    ]
    for line_no in range(3, 380):
        if line_no == 120:
            body_lines.append("120:   public update(delta: number): void {")
        elif line_no == 164:
            body_lines.append("164:     this.hud.render(this.score, this.health);")
        else:
            body_lines.append(
                f"{line_no}:     review evidence line {line_no} with enough "
                "body text to represent a real OpenCode read payload;"
            )
    large_read_output = (
        "<path>src/game/Game.ts</path>\n"
        "<type>file</type>\n"
        "<content>\n"
        + "\n".join(body_lines)
        + "\n</content>"
    )
    raw_messages = [
        openai.ChatMessage(role="system", content="You are OpenCode."),
        openai.ChatMessage(
            role="user",
            content=(
                "Do not edit files. Inspect the project and evaluate quality. "
                "Return SPEEDQA0606."
            ),
        ),
        openai.ChatMessage(role="assistant", content="", tool_calls=[tool_call]),
        openai.ChatMessage(
            role="tool",
            tool_call_id="call_read_game",
            content=large_read_output,
        ),
    ]
    assistant_answer = (
        "SPEEDQA0606: Game.ts owns the core update loop and HUD writes; "
        "the highest release risk is that this file mixes simulation, scoring, "
        "and rendering state."
    )
    state = SimpleNamespace(
        runtime=SimpleNamespace(tokenizer=tokenizer),
        args=parse_args(["--warmup-tokens", "0"]),
    )

    postcommit_prefix, postcommit_vision_splice = openai._history_ids_for_postcommit(
        state,
        messages=raw_messages,
        assistant_content=assistant_answer,
        assistant_tool_calls=None,
        thinking_enabled=True,
        tool_specs=tools,
    )
    assert postcommit_vision_splice is None
    next_canonical, next_stats = openai._canonicalize_agent_transcript(
        [
            *raw_messages,
            openai.ChatMessage(role="assistant", content=assistant_answer),
            openai.ChatMessage(role="user", content="Now give me the exact line risk."),
        ],
        tools_active=True,
    )
    next_turn_prompt = openai._encode_messages(
        tokenizer,
        next_canonical,
        enable_thinking=True,
        tools=tools,
    )

    # Prefix-stability fix (2026-07-03): the historical read now classifies by
    # its OWN segment's user request (inspection) instead of the latest user
    # text, so both the postcommit and the next turn render it as the same
    # inspection digest — the alignment invariant below is what matters.
    assert next_stats.compacted_tool_result_messages == 0
    assert next_stats.compacted_active_read_inspection_messages >= 1
    assert next_turn_prompt[: len(postcommit_prefix)] == postcommit_prefix
    rendered_prefix = tokenizer.decode(postcommit_prefix)
    assert "MTPLX read-only final answer instruction" not in rendered_prefix
    assert "<mtplx_read_inspection_digest" in rendered_prefix
    assert "<mtplx_compacted_active_read_output" not in rendered_prefix


def test_postcommit_opencode_replay_strips_persisted_tool_preamble():
    tokenizer = QwenToolHistoryBoundaryTokenizer()
    tools = [_named_tool_schema("read")]
    tool_call = {
        "id": "call_read_game",
        "type": "function",
        "function": {
            "name": "read",
            "arguments": {"filePath": "src/game/Game.ts"},
        },
    }
    large_read_output = (
        "<path>src/game/Game.ts</path>\n"
        "<type>file</type>\n"
        "<content>\n"
        + "\n".join(
            f"{line_no}: release-readiness evidence line {line_no}"
            for line_no in range(1, 160)
        )
        + "\n</content>"
    )
    live_messages = [
        openai.ChatMessage(role="system", content="You are OpenCode."),
        openai.ChatMessage(
            role="user",
            content="Evaluate this project like a release candidate.",
        ),
        openai.ChatMessage(role="assistant", content="", tool_calls=[tool_call]),
        openai.ChatMessage(
            role="tool",
            tool_call_id="call_read_game",
            content=large_read_output,
        ),
    ]
    replay_messages = [
        openai.ChatMessage(role="system", content="You are OpenCode."),
        openai.ChatMessage(
            role="user",
            content="Evaluate this project like a release candidate.",
        ),
        openai.ChatMessage(
            role="assistant",
            content="Let me inspect Game.ts before I summarize the release risk.",
            tool_calls=[tool_call],
        ),
        openai.ChatMessage(
            role="tool",
            tool_call_id="call_read_game",
            content=large_read_output,
        ),
    ]
    assistant_answer = "The top launch risk is missing Game teardown."
    state = SimpleNamespace(
        runtime=SimpleNamespace(tokenizer=tokenizer),
        args=parse_args(["--warmup-tokens", "0"]),
    )

    postcommit_prefix, postcommit_vision_splice = openai._history_ids_for_postcommit(
        state,
        messages=live_messages,
        assistant_content=assistant_answer,
        assistant_tool_calls=None,
        thinking_enabled=True,
        tool_specs=tools,
        tool_prompt_mode="compact",
        strip_tool_call_preamble_text=True,
    )
    assert postcommit_vision_splice is None
    next_canonical, stats = openai._canonicalize_agent_transcript(
        [
            *replay_messages,
            openai.ChatMessage(role="assistant", content=assistant_answer),
            openai.ChatMessage(role="user", content="Which fix should ship first?"),
        ],
        tools_active=True,
        strip_tool_call_preamble_text=True,
    )
    next_turn_prompt = openai._encode_messages(
        tokenizer,
        next_canonical,
        enable_thinking=True,
        tools=tools,
        tool_prompt_mode="compact",
    )

    assert stats.stripped_tool_preamble_messages == 1
    assert next_turn_prompt[: len(postcommit_prefix)] == postcommit_prefix
    assert (
        "Let me inspect Game.ts before I summarize the release risk."
        not in tokenizer.decode(postcommit_prefix)
    )


def test_chat_tool_xml_returns_openai_tool_calls_nonstream(monkeypatch):
    client = TestClient(create_app(_fake_state()))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        lambda *_args, **_kwargs: _fake_generation(
            "<tool_call>\n<function=session_status>\n</function>\n</tool_call>"
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Status."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"][0]["function"] == {
        "name": "session_status",
        "arguments": "{}",
    }
    assert "<tool_call>" not in json.dumps(payload)


def test_chat_tool_json_returns_openai_tool_calls_nonstream(monkeypatch):
    client = TestClient(create_app(_fake_state()))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        lambda *_args, **_kwargs: _fake_generation(
            '<tool_call>{"name":"session_status","arguments":{}}</tool_call>'
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Status."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_poolside_tool_call_payload_maps_arg_pairs_to_openai_call():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    calls = openai._parse_generated_tool_calls(
        "<tool_call>get_weather"
        "<arg_key>city</arg_key>"
        "<arg_value>Paris</arg_value>"
        "</tool_call>",
        tools=tools,
    )

    assert calls is not None
    assert calls[0]["function"] == {
        "name": "get_weather",
        "arguments": '{"city":"Paris"}',
    }


def test_poolside_reasoning_parser_keeps_target_default_thinking_enabled():
    from mtplx.backends.descriptors import LAGUNA_AR_DESCRIPTOR

    state = SimpleNamespace(
        args=SimpleNamespace(
            reasoning_parser="poolside_v1",
            enable_thinking=True,
        ),
        backend_descriptor=LAGUNA_AR_DESCRIPTOR,
    )
    request = SimpleNamespace(enable_thinking=None)

    assert openai._thinking_enabled_for_request(state, request) is True


def test_anthropic_messages_returns_tool_use_nonstream(monkeypatch):
    client = TestClient(create_app(_fake_state()))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        lambda *_args, **_kwargs: _fake_generation(
            "<tool_call>\n"
            "<function=Bash>\n"
            "<parameter=command>\n"
            "./test.sh\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        ),
    )

    response = client.post(
        "/v1/messages",
        headers={
            "x-mtplx-cache-mode": "bypass",
            "x-mtplx-allow-client-controls": "1",
        },
        json={
            "model": "mtplx-test-model",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Run ./test.sh"}],
            "tools": [
                {
                    "name": "Bash",
                    "description": "Run a shell command.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                }
            ],
            "tool_choice": {"type": "auto"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["stop_reason"] == "tool_use"
    assert payload["content"][0]["type"] == "tool_use"
    assert payload["content"][0]["name"] == "Bash"
    assert payload["content"][0]["input"] == {"command": "./test.sh"}
    assert "<tool_call" not in response.text


def test_anthropic_messages_streams_tool_use_without_text_leak(monkeypatch):
    client = TestClient(create_app(_fake_state()))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation(
            "<tool_call>\n"
            "<function=Bash>\n"
            "<parameter=command>\n"
            "./test.sh\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        ),
    )

    response = client.post(
        "/v1/messages",
        headers={
            "x-mtplx-cache-mode": "bypass",
            "x-mtplx-allow-client-controls": "1",
        },
        json={
            "model": "mtplx-test-model",
            "max_tokens": 16,
            "stream": True,
            "messages": [{"role": "user", "content": "Run ./test.sh"}],
            "tools": [
                {
                    "name": "Bash",
                    "description": "Run a shell command.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                }
            ],
            "tool_choice": {"type": "auto"},
        },
    )

    assert response.status_code == 200
    events = _anthropic_events(response.text)
    event_names = [event for event, _payload in events]
    assert "content_block_start" in event_names
    assert any(
        payload.get("content_block", {}).get("type") == "tool_use"
        and payload["content_block"]["name"] == "Bash"
        for event, payload in events
        if event == "content_block_start"
    )
    input_json = "".join(
        payload.get("delta", {}).get("partial_json", "")
        for event, payload in events
        if event == "content_block_delta"
        and payload.get("delta", {}).get("type") == "input_json_delta"
    )
    assert "./test.sh" in input_json
    assert any(
        event == "message_delta" and payload["delta"]["stop_reason"] == "tool_use"
        for event, payload in events
    )
    assert "<tool_call" not in response.text
    assert '"type": "text"' not in response.text


def test_anthropic_messages_suppresses_unclosed_tool_markup(monkeypatch):
    state = _fake_state()
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation(
            "<tool_call>\n"
            "<function=Bash>\n"
            "<parameter=command>\n"
            "pwd\n"
            "</parameter>\n"
            "<parameter=description>\n"
            "Print working directory\n"
            "</parameter>"
        ),
    )

    response = client.post(
        "/v1/messages",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "model": "mtplx-test-model",
            "max_tokens": 64,
            "stream": True,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "Run pwd"}],
            "tools": [
                {
                    "name": "Bash",
                    "description": "Run a shell command.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["command", "description"],
                    },
                }
            ],
            "tool_choice": {"type": "auto"},
        },
    )

    assert response.status_code == 200
    events = _anthropic_events(response.text)
    assert not any(
        payload.get("content_block", {}).get("type") == "tool_use"
        for event, payload in events
        if event == "content_block_start"
    )
    text = "".join(
        payload.get("delta", {}).get("text", "")
        for event, payload in events
        if event == "content_block_delta"
        and payload.get("delta", {}).get("type") == "text_delta"
    )
    assert "<tool_call>" not in text
    assert "pwd" not in text
    assert "Print working directory" not in text
    assert text == ""


def test_anthropic_messages_streams_namespaced_tool_use(monkeypatch):
    client = TestClient(create_app(_fake_state()))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation(
            "Checking."
            "<foo-bar:tool_call>"
            '<invoke name="Bash">'
            '<parameter name="command">"pwd"</parameter>'
            '<parameter name="description">"Print working directory"</parameter>'
            "</invoke>"
            "</foo-bar:tool_call>"
        ),
    )

    response = client.post(
        "/v1/messages",
        headers={
            "x-mtplx-cache-mode": "bypass",
            "x-mtplx-allow-client-controls": "1",
        },
        json={
            "model": "mtplx-test-model",
            "max_tokens": 64,
            "stream": True,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "Run pwd"}],
            "tools": [
                {
                    "name": "Bash",
                    "description": "Run a shell command.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["command", "description"],
                    },
                }
            ],
            "tool_choice": {"type": "auto"},
        },
    )

    assert response.status_code == 200
    events = _anthropic_events(response.text)
    streamed_text = "".join(
        payload["delta"]["text"]
        for event, payload in events
        if event == "content_block_delta"
        and payload.get("delta", {}).get("type") == "text_delta"
    )
    assert streamed_text == "Checking."
    assert any(
        payload.get("content_block", {}).get("type") == "tool_use"
        and payload["content_block"]["name"] == "Bash"
        for event, payload in events
        if event == "content_block_start"
    )
    assert "<foo-bar:tool_call>" not in response.text
    assert "<invoke" not in response.text


def test_anthropic_count_tokens_uses_converted_tools():
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    client = TestClient(create_app(state))

    response = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "mtplx-test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "name": "Bash",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json() == {"input_tokens": 3}
    _messages, kwargs = state.runtime.tokenizer.calls[-1]
    assert kwargs["tools"][0]["function"]["name"] == "Bash"


def test_chat_stream_tool_calls_emit_delta_tool_calls(monkeypatch):
    client = TestClient(create_app(_fake_state()))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation(
            "<tool_call>\n<function=session_status>\n</function>\n</tool_call>"
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Status."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    assert '"tool_calls"' in response.text
    assert '"finish_reason": "tool_calls"' in response.text
    assert "<tool_call>" not in response.text


def test_chat_stream_consecutive_qwen_xml_tool_calls_emit_ordered_deltas(monkeypatch):
    client = TestClient(create_app(_fake_state()))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    generated = (
        "<tool_call>\n<function=write>\n<parameter=filePath>\n"
        "one.py\n</parameter>\n<parameter=content>\nprint(1)\n</parameter>\n"
        "</function>\n</tool_call>\n\n"
        "<tool_call>\n<function=write>\n<parameter=filePath>\n"
        "two.py\n</parameter>\n<parameter=content>\nprint(2)\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=write>\n<parameter=filePath>\n"
        "three.py\n</parameter>\n<parameter=content>\nprint(3)\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    monkeypatch.setattr(
        openai, "_run_generation", _fake_streaming_generation(generated)
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Write files."}],
            "tools": [_write_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 256,
        },
    )

    assert response.status_code == 200
    assert "<tool_call" not in response.text
    assert "_call>" not in response.text
    assert "<function=" not in response.text
    payloads = _stream_payloads(response.text)
    tool_deltas = [
        item
        for payload in payloads
        for item in payload["choices"][0]["delta"].get("tool_calls", [])
    ]
    name_deltas = [
        item for item in tool_deltas if item.get("function", {}).get("name") == "write"
    ]
    assert [item["index"] for item in name_deltas] == [0, 1, 2]
    assert any(
        payload["choices"][0].get("finish_reason") == "tool_calls"
        for payload in payloads
    )


def test_chat_stream_qwen_xml_shell_arguments_are_typed(monkeypatch):
    client = TestClient(create_app(_fake_state()))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    generated = (
        "<tool_call>\n<function=bash>\n"
        "<parameter=command>\nnpm run build\n</parameter>\n"
        "<parameter=description>\nBuild the project\n</parameter>\n"
        "<parameter=timeout>\n60000\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    monkeypatch.setattr(
        openai, "_run_generation", _fake_streaming_generation(generated)
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Build."}],
            "tools": [_bash_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 256,
        },
    )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    args_text = "".join(
        item.get("function", {}).get("arguments", "")
        for payload in payloads
        for item in payload["choices"][0]["delta"].get("tool_calls", [])
    )
    args = json.loads(args_text)
    assert args == {
        "command": "npm run build",
        "description": "Build the project",
        "timeout": 60000,
    }
    assert isinstance(args["timeout"], int)
    assert '"timeout":"60000"' not in args_text
    assert any(
        payload["choices"][0].get("finish_reason") == "tool_calls"
        for payload in payloads
    )


def test_chat_stream_qwen_xml_questions_arguments_are_arrays(monkeypatch):
    client = TestClient(create_app(_fake_state()))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    questions = (
        '[{"header":"Scope","id":"scope","question":"Pick one",'
        '"options":[{"label":"A","description":"first"},'
        '{"label":"B","description":"second"}]}]'
    )
    generated = (
        "<tool_call>\n<function=question>\n"
        f"<parameter=questions>\n{questions}\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    monkeypatch.setattr(
        openai, "_run_generation", _fake_streaming_generation(generated)
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Ask."}],
            "tools": [_question_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 256,
        },
    )

    assert response.status_code == 200
    args_text = "".join(
        item.get("function", {}).get("arguments", "")
        for payload in _stream_payloads(response.text)
        for item in payload["choices"][0]["delta"].get("tool_calls", [])
    )
    args = json.loads(args_text)
    assert isinstance(args["questions"], list)
    assert args["questions"][0]["options"][1]["description"] == "second"


def test_chat_stream_tool_call_preamble_is_stored_for_postcommit(monkeypatch):
    state = _fake_streaming_session_state()
    state.args.stream_interval = 1
    captured_generation_final: list[dict] = []
    scheduled: list[dict] = []

    def fake_store_generation_final(*_args, **kwargs):
        captured_generation_final.append(kwargs)
        return {
            "stored": False,
            "mode": "unsafe",
            "reason": "tool_call_history_rewrite",
        }

    def fake_schedule(*_args, **kwargs):
        scheduled.append(kwargs)
        return {
            "stored": False,
            "mode": "async_pending",
            "reason": kwargs["unsafe_reason"],
        }

    monkeypatch.setattr(
        openai, "_store_generation_final_history_snapshot", fake_store_generation_final
    )
    monkeypatch.setattr(openai, "_schedule_idle_postcommit_snapshot", fake_schedule)
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation(
            "Let me research first.\n\n"
            "<tool_call>\n<function=session_status>\n</function>\n</tool_call>"
        ),
    )

    with TestClient(create_app(state)) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={
                    "x-mtplx-session-id": "stream-tool-preamble",
                    "x-mtplx-allow-client-controls": "1",
                },
                json={
                "messages": [{"role": "user", "content": "Status."}],
                "tools": [_tool_schema()],
                "tool_choice": "auto",
                "stream": True,
                "max_tokens": 64,
                "enable_thinking": False,
            },
        )

    assert response.status_code == 200
    assert '"tool_calls"' in response.text
    assert '"finish_reason": "tool_calls"' in response.text
    streamed_content = "".join(
        payload["choices"][0]["delta"]["content"]
        for payload in _stream_payloads(response.text)
        if payload["choices"][0]["delta"].get("content")
    )
    assert streamed_content == "Let me research first.\n\n"
    assert "<tool_call>" not in response.text

    assert captured_generation_final
    assert scheduled
    for call in (captured_generation_final[0], scheduled[0]):
        assert call["assistant_content"] == "Let me research first."
        assert "<tool_call>" not in call["assistant_content"]
        assert call["assistant_tool_calls"][0]["function"] == {
            "name": "session_status",
            "arguments": "{}",
        }


def test_chat_stream_hermes_suppresses_tool_call_preamble(monkeypatch):
    state = _fake_streaming_session_state()
    state.args.stream_interval = 1
    state.args.enable_thinking = False
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation(
            "search\n\nMac Studio M4 Max\n\n"
            "<tool_call>\n<function=session_status>\n</function>\n</tool_call>"
        ),
    )

    with TestClient(create_app(state)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-client": "hermes"},
            json={
                "messages": [{"role": "user", "content": "Status."}],
                "tools": [_tool_schema()],
                "tool_choice": "auto",
                "stream": True,
                "max_tokens": 64,
                "enable_thinking": False,
            },
        )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    assert any(
        payload["choices"][0]["delta"].get("tool_calls") for payload in payloads
    )
    assert not any(
        payload["choices"][0]["delta"].get("content") for payload in payloads
    )
    assert any(
        payload["choices"][0].get("finish_reason") == "tool_calls"
        for payload in payloads
    )


def test_chat_stream_hermes_defers_content_until_native_tool_extraction(monkeypatch):
    state = _fake_streaming_session_state()
    state.args.stream_interval = 1
    state.args.enable_thinking = False
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation("search\n\nMac Studio M4 Max\n\n"),
    )
    monkeypatch.setattr(
        openai,
        "omlx_extract_tool_calls_with_thinking",
        lambda *_args, **_kwargs: SimpleNamespace(
            cleaned_text="",
            cleaned_thinking="",
            tool_calls=[
                {
                    "id": "call_fact_store",
                    "type": "function",
                    "function": {
                        "name": "session_status",
                        "arguments": "{}",
                    },
                }
            ],
            parser_source="native",
            status="parsed",
            malformed_reason=None,
            raw_tool_markup_suppressed=True,
        ),
    )

    with TestClient(create_app(state)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-client": "hermes"},
            json={
                "messages": [{"role": "user", "content": "Status."}],
                "tools": [_tool_schema()],
                "tool_choice": "auto",
                "stream": True,
                "max_tokens": 64,
                "enable_thinking": False,
            },
        )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    assert any(
        payload["choices"][0]["delta"].get("tool_calls") for payload in payloads
    )
    assert not any(
        payload["choices"][0]["delta"].get("content") for payload in payloads
    )
    assert any(
        payload["choices"][0].get("finish_reason") == "tool_calls"
        for payload in payloads
    )


def test_chat_stream_tool_call_postcommit_strips_reasoning_content(monkeypatch):
    state = _fake_streaming_session_state()
    state.args.stream_interval = 1
    captured_generation_final: list[dict] = []
    scheduled: list[dict] = []

    def fake_store_generation_final(*_args, **kwargs):
        captured_generation_final.append(kwargs)
        return {
            "stored": False,
            "mode": "unsafe",
            "reason": "tool_call_history_rewrite",
        }

    def fake_schedule(*_args, **kwargs):
        scheduled.append(kwargs)
        return {
            "stored": False,
            "mode": "async_pending",
            "reason": kwargs["unsafe_reason"],
        }

    monkeypatch.setattr(
        openai, "_store_generation_final_history_snapshot", fake_store_generation_final
    )
    monkeypatch.setattr(openai, "_schedule_idle_postcommit_snapshot", fake_schedule)
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation(
            "<think>raw private plan</think>\n"
            "Let me check.\n"
            "<tool_call>\n<function=session_status>\n</function>\n</tool_call>"
        ),
    )

    with TestClient(create_app(state)) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"x-mtplx-session-id": "stream-tool-reasoning"},
            json={
                "messages": [{"role": "user", "content": "Status."}],
                "tools": [_tool_schema()],
                "tool_choice": "auto",
                "stream": True,
                "max_tokens": 64,
            },
        )

    assert response.status_code == 200
    reasoning = "".join(
        payload["choices"][0]["delta"].get("reasoning_content", "")
        for payload in _stream_payloads(response.text)
        for _choice in [payload["choices"][0]]
    )
    assert reasoning == "raw private plan"
    assert captured_generation_final
    assert scheduled
    for call in (captured_generation_final[0], scheduled[0]):
        assert call["assistant_content"] == "Let me check."
        assert "raw private plan" not in call["assistant_content"]
        assert "<think>" not in call["assistant_content"]


def test_chat_stream_tools_plain_content_stays_incremental(monkeypatch):
    state = _fake_state()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation("Count: 1, 2, 3.\n"),
    )

    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-mtplx-cache-mode": "bypass",
            "x-mtplx-allow-client-controls": "1",
        },
        json={
            "messages": [{"role": "user", "content": "Count."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 16,
            "enable_thinking": False,
        },
    )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    content_deltas = [
        payload["choices"][0]["delta"]["content"]
        for payload in payloads
        if payload["choices"][0]["delta"].get("content")
    ]
    assert len(content_deltas) > 1
    assert "".join(content_deltas) == "Count: 1, 2, 3.\n"
    assert not any(
        payload["choices"][0]["delta"].get("tool_calls") for payload in payloads
    )
    assert any(
        payload["choices"][0].get("finish_reason") == "stop" for payload in payloads
    )


def test_chat_stream_tool_call_arguments_are_incremental(monkeypatch):
    state = _fake_state()
    state.args.stream_interval = 1
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation(
            '<tool_call>{"name":"add","arguments":{"a":25,"b":17}}</tool_call>'
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Use add."}],
            "tools": [_add_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    tool_deltas = [
        payload["choices"][0]["delta"]["tool_calls"][0]
        for payload in payloads
        if payload["choices"][0]["delta"].get("tool_calls")
    ]
    assert tool_deltas[0]["id"].startswith("call_")
    assert tool_deltas[0]["type"] == "function"
    assert tool_deltas[0]["function"] == {"name": "add", "arguments": ""}

    argument_fragments = [
        delta["function"]["arguments"]
        for delta in tool_deltas
        if delta.get("function", {}).get("arguments") is not None
    ]
    assert "".join(argument_fragments) == '{"a":25,"b":17}'
    assert len([fragment for fragment in argument_fragments if fragment]) > 1
    assert '{"a":25,"b":17}' not in argument_fragments


def test_chat_stream_completed_tool_call_suppresses_trailing_text(monkeypatch):
    state = _fake_state()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    tool_text = "<tool_call>\n<function=session_status>\n</function>\n</tool_call>"
    extra_text = "\nSHOULD_STREAM"
    cancel_seen = {"value": False}

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        token_callback = kwargs["token_callback"]
        token_callback([ord(char) for char in tool_text])
        deadline = time.time() + 0.05
        cancel_event = kwargs["cancel_event"]
        while not cancel_event.is_set() and time.time() < deadline:
            time.sleep(0.01)
        cancel_seen["value"] = cancel_event.is_set()
        if not cancel_event.is_set():
            for token in [ord(char) for char in extra_text]:
                token_callback([token])
        text = tool_text + extra_text
        return {
            "text": text,
            "tokens": [ord(char) for char in text],
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": len(text),
            },
            "prompt_tokens": 3,
            "completion_tokens": len(text),
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Status."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 256,
        },
    )

    assert response.status_code == 200
    assert cancel_seen["value"] is False
    assert "SHOULD_STREAM" not in response.text
    payloads = _stream_payloads(response.text)
    assert any(
        payload["choices"][0].get("finish_reason") == "tool_calls"
        for payload in payloads
    )
    final = [payload for payload in payloads if payload["choices"][0]["finish_reason"]]
    assert final[-1]["mtplx_stats"]["early_tool_cancel_used"] is True
    assert final[-1]["mtplx_stats"]["tool_parse_status"] == "success"
    assert "data: [DONE]" in response.text


def test_chat_stream_broken_stdout_pipe_does_not_surface_to_client(monkeypatch):
    state = _fake_state()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(openai, "_run_generation", _fake_streaming_generation("ok"))
    monkeypatch.setattr(openai, "_STDOUT_LOGGING_BROKEN", False)

    def broken_print(*_args, **_kwargs):
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(openai.builtins, "print", broken_print)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Say ok."}],
            "stream": True,
            "max_tokens": 16,
            "enable_thinking": False,
        },
    )

    assert response.status_code == 200
    assert "Broken pipe" not in response.text
    payloads = _stream_payloads(response.text)
    content = "".join(
        payload["choices"][0]["delta"].get("content", "") for payload in payloads
    )
    assert content == "ok"
    assert any(
        payload["choices"][0].get("finish_reason") == "stop" for payload in payloads
    )
    assert "data: [DONE]" in response.text


def test_chat_stream_emits_heartbeat_during_alive_silence(monkeypatch):
    state = _fake_state()
    state.args.stats_footer = False
    state.generation_executor = ThreadPoolExecutor(max_workers=1)
    client = TestClient(create_app(state))
    tokens = [ord("o"), ord("k"), ord("\n")]

    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(openai, "STREAM_HEARTBEAT_INTERVAL_S", 0.0)
    monkeypatch.setattr(openai, "STREAM_SILENCE_WARN_S", 0.01)
    monkeypatch.setattr(openai, "STREAM_SILENCE_WARN_INTERVAL_S", 60.0)

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        # Leave enough wall time for the ASGI stream loop to observe an empty
        # token queue even on slower GitHub macOS runners.
        time.sleep(1.25)
        kwargs["token_callback"](tokens)
        return {
            "text": "ok\n",
            "tokens": tokens,
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": len(tokens),
            },
            "prompt_tokens": 3,
            "completion_tokens": len(tokens),
            "finish_reason": "stop",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    try:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "x-mtplx-cache-mode": "bypass",
                "x-mtplx-allow-client-controls": "1",
            },
            json={
                "messages": [{"role": "user", "content": "Say ok."}],
                "stream": True,
                "max_tokens": 16,
                "enable_thinking": False,
            },
        )
    finally:
        state.generation_executor.shutdown(wait=True)

    assert response.status_code == 200
    payloads = []
    for event in response.text.split("\n\n"):
        if not event.startswith("data: {"):
            continue
        payloads.append(json.loads(event.removeprefix("data: ")))

    heartbeats = [
        payload
        for payload in payloads
        if payload.get("mtplx_progress", {}).get("heartbeat") is True
    ]
    content = "".join(
        payload["choices"][0].get("delta", {}).get("content", "")
        for payload in payloads
    )
    final_chunks = [
        payload
        for payload in payloads
        if payload["choices"][0].get("finish_reason") == "stop"
    ]

    assert heartbeats
    assert heartbeats[0]["mtplx_progress"]["phase"] == "generating"
    assert heartbeats[0]["mtplx_progress"]["completion_tokens"] == 0
    assert content == "ok\n"
    assert final_chunks
    assert "data: [DONE]" in response.text


def test_chat_tools_malformed_tool_call_falls_back_to_content(monkeypatch):
    state = _fake_state()
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        lambda *_args, **_kwargs: _fake_generation("<tool_call>not json</tool_call>"),
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Status."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == ""
    assert "tool_calls" not in choice["message"]
    stats = response.json()["mtplx_stats"]
    assert stats["tool_parse_fallback"] is True
    assert stats["tool_parse_fallback_kind"] == "malformed_tool_call"
    assert state.tool_parse_counters["malformed_tool_call"] == 1
    assert state.tool_parse_counters["tool_parse_fallback"] == 1


def test_chat_tools_malformed_tool_call_drops_punctuation_only_visible_fallback(
    monkeypatch,
):
    state = _fake_state()
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai,
        "_run_generation",
        lambda *_args, **_kwargs: _fake_generation(
            "...<tool_call>not json</tool_call>"
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Status."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == ""
    stats = response.json()["mtplx_stats"]
    assert stats["tool_parse_fallback"] is True
    assert stats["tool_parse_fallback_kind"] == "malformed_tool_call"
    assert stats["raw_tool_markup_suppressed"] is True


def test_chat_tools_unknown_generated_tool_passes_through(monkeypatch):
    state = _fake_state()
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    text = (
        "<tool_call>\n<function=Agent>\n"
        "<parameter=description>\nList files\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    monkeypatch.setattr(
        openai,
        "_run_generation",
        lambda *_args, **_kwargs: _fake_generation(text),
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Status."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "max_tokens": 16,
        },
    )

    # A name outside the request's tools list is still delivered as a real tool call:
    # the OpenAI surface leaves the rejection to the client, which answers the model and
    # lets it self-correct. Degrading the turn to prose instead makes agent clients count
    # a consecutive-mistake strike (three fails the task), so this must NOT fall back.
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    calls = choice["message"]["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["Agent"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"description": "List files"}
    stats = response.json()["mtplx_stats"]
    assert stats["tool_parse_status"] == "parsed"
    assert stats["tool_calls_emitted"] == 1
    assert stats["raw_tool_markup_suppressed"] is True
    assert "tool_parse_fallback_kind" not in stats


def test_chat_stream_unknown_generated_tool_passes_through(monkeypatch):
    state = _fake_state()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    text = (
        "<tool_call>\n<function=task>\n"
        "<parameter=description>\nList files\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    monkeypatch.setattr(openai, "_run_generation", _fake_streaming_generation(text))

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Status."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    streamed_content = "".join(
        payload["choices"][0]["delta"].get("content", "") for payload in payloads
    )
    # Streaming carries the same contract as the buffered path: an unknown name is
    # delivered as incremental tool_calls deltas, never degraded to prose (see the
    # non-stream sibling for why the fallback would cost the client a mistake strike).
    assert streamed_content == ""
    deltas = [
        payload["choices"][0]["delta"]["tool_calls"]
        for payload in payloads
        if payload["choices"][0]["delta"].get("tool_calls")
    ]
    assert deltas, "unknown tool must still stream as tool_calls"
    assert deltas[0][0]["function"]["name"] == "task"
    streamed_args = "".join(
        chunk[0]["function"].get("arguments", "") for chunk in deltas
    )
    assert json.loads(streamed_args) == {"description": "List files"}
    final = [payload for payload in payloads if payload["choices"][0]["finish_reason"]]
    assert final[-1]["choices"][0]["finish_reason"] == "tool_calls"
    stats = final[-1]["mtplx_stats"]
    assert stats["tool_calls_emitted"] == 1
    assert stats["raw_tool_markup_suppressed"] is True
    assert "tool_parse_fallback_kind" not in stats
    assert "data: [DONE]" in response.text


def test_chat_stream_unclosed_tool_call_falls_back_and_finishes(monkeypatch):
    state = _fake_state()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    text = "<tool_call>\n<function=session_status>\n"
    monkeypatch.setattr(openai, "_run_generation", _fake_streaming_generation(text))

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Status."}],
            "tools": [_tool_schema()],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    streamed_content = "".join(
        payload["choices"][0]["delta"].get("content", "") for payload in payloads
    )
    assert streamed_content == ""
    assert not any(
        payload["choices"][0]["delta"].get("tool_calls") for payload in payloads
    )
    final = [payload for payload in payloads if payload["choices"][0]["finish_reason"]]
    assert final[-1]["choices"][0]["finish_reason"] == "stop"
    assert final[-1]["mtplx_stats"]["tool_parse_fallback_kind"] == "unclosed_tool_call"
    assert "data: [DONE]" in response.text


def test_chat_stream_missing_required_tool_argument_still_emits_model_tool_call(
    monkeypatch,
):
    state = _fake_state()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    text = "<tool_call>\n<function=run_shell>\n</function>\n</tool_call>"
    monkeypatch.setattr(openai, "_run_generation", _fake_streaming_generation(text))

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "Run a command."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "run_shell",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    streamed_content = "".join(
        payload["choices"][0]["delta"].get("content", "") for payload in payloads
    )
    assert streamed_content == ""
    assert any(payload["choices"][0]["delta"].get("tool_calls") for payload in payloads)
    final = [payload for payload in payloads if payload["choices"][0]["finish_reason"]]
    assert final[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert final[-1]["mtplx_stats"]["tool_parse_status"] == "parsed"
    assert final[-1]["mtplx_stats"]["tool_calls_emitted"] == 1


def test_server_state_emits_startup_progress(monkeypatch, capsys):
    monkeypatch.setattr(openai, "apply_profile_env", lambda _profile, **_kwargs: None)
    monkeypatch.setattr(openai, "profile_env_status", lambda _profile, **_kwargs: {})
    monkeypatch.setattr(openai, "_fast_path_env_status", lambda: {})
    monkeypatch.setattr(openai, "_mlx_runtime_status", lambda: {"ok": True})
    monkeypatch.setattr(
        openai, "_configure_mlx_cache_limit", lambda _args: {"configured": False}
    )
    monkeypatch.setattr(
        openai,
        "load",
        lambda model, mtp, contract, **_kwargs: SimpleNamespace(
            model_path=Path(model),
            mtp_enabled=mtp,
            tokenizer=SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        openai, "_install_draft_lm_head", lambda *_args, **_kwargs: {"installed": True}
    )
    monkeypatch.setattr(openai, "_draft_head_identity", lambda _runtime: "draft-head")
    monkeypatch.setattr(openai, "_template_hash", lambda _tokenizer: "template")
    monkeypatch.setattr(
        openai, "_resolve_context_window", lambda _tokenizer, _model: 32768
    )
    monkeypatch.setattr(openai, "EngineSessionManager", lambda **_kwargs: SimpleNamespace())

    args = parse_args(["--model", "models/example", "--warmup-tokens", "0"])
    state = openai.ServerState(args)

    captured = capsys.readouterr().out
    assert "[4/6] Preparing Sustained MTP runtime" in captured
    assert "[5/6] Loading model weights: models/example" in captured
    assert "This is the long step" in captured
    assert "Model load in progress (this may take a minute)" in captured
    assert "[5/6] Model loaded" in captured
    assert "[6/6] Warmup skipped" in captured
    assert state.context_window == 32768


def test_server_state_applies_clear_cache_every_after_profile(monkeypatch):
    captured: dict[str, dict[str, str]] = {}

    def capture_apply_profile_env(_profile, **kwargs):
        captured["apply"] = dict(kwargs.get("runtime_env_overrides") or {})

    def capture_profile_env_status(_profile, **kwargs):
        captured["status"] = dict(kwargs.get("runtime_env_overrides") or {})
        return {}

    monkeypatch.setattr(openai, "apply_profile_env", capture_apply_profile_env)
    monkeypatch.setattr(openai, "profile_env_status", capture_profile_env_status)
    monkeypatch.setattr(openai, "_fast_path_env_status", lambda: {})
    monkeypatch.setattr(openai, "_mlx_runtime_status", lambda: {"ok": True})
    monkeypatch.setattr(
        openai,
        "_configure_mlx_cache_limit",
        lambda _args: {"configured": False},
    )
    monkeypatch.setattr(
        openai,
        "load",
        lambda model, mtp, contract, **_kwargs: SimpleNamespace(
            model_path=Path(model),
            mtp_enabled=mtp,
            tokenizer=SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        openai,
        "_install_draft_lm_head",
        lambda *_args, **_kwargs: {"installed": True},
    )
    monkeypatch.setattr(openai, "_draft_head_identity", lambda _runtime: "draft-head")
    monkeypatch.setattr(openai, "_template_hash", lambda _tokenizer: "template")
    monkeypatch.setattr(
        openai,
        "_resolve_context_window",
        lambda _tokenizer, _model: 32768,
    )
    monkeypatch.setattr(openai, "EngineSessionManager", lambda **_kwargs: SimpleNamespace())

    args = parse_args(
        [
            "--model",
            "models/example",
            "--warmup-tokens",
            "0",
            "--clear-cache-every",
            "512",
        ]
    )
    state = openai.ServerState(args)

    assert state.runtime_env_overrides["MTPLX_CLEAR_CACHE_EVERY"] == "512"
    assert captured["apply"]["MTPLX_CLEAR_CACHE_EVERY"] == "512"
    assert captured["status"]["MTPLX_CLEAR_CACHE_EVERY"] == "512"


def test_server_state_reports_model_load_failure(monkeypatch, capsys):
    monkeypatch.setattr(openai, "apply_profile_env", lambda _profile, **_kwargs: None)
    monkeypatch.setattr(openai, "profile_env_status", lambda _profile, **_kwargs: {})
    monkeypatch.setattr(openai, "_fast_path_env_status", lambda: {})
    monkeypatch.setattr(openai, "_mlx_runtime_status", lambda: {"ok": True})
    monkeypatch.setattr(
        openai, "_configure_mlx_cache_limit", lambda _args: {"configured": False}
    )

    def fail_load(model, mtp, contract, **_kwargs):
        assert model == "models/example"
        assert mtp is True
        assert contract is not None
        raise RuntimeError("boom")

    monkeypatch.setattr(openai, "load", fail_load)

    args = parse_args(["--model", "models/example", "--warmup-tokens", "0"])
    with pytest.raises(RuntimeError, match="boom"):
        openai.ServerState(args)

    captured = capsys.readouterr().out
    assert "[5/6] Model load failed" in captured
    assert "RuntimeError: boom" in captured


def test_server_state_passes_step_adapter_quant_contract_to_load(monkeypatch):
    captured = {}
    monkeypatch.setattr(openai, "apply_profile_env", lambda _profile, **_kwargs: None)
    monkeypatch.setattr(openai, "profile_env_status", lambda _profile, **_kwargs: {})
    monkeypatch.setattr(openai, "_fast_path_env_status", lambda: {})
    monkeypatch.setattr(openai, "_mlx_runtime_status", lambda: {"ok": True})
    monkeypatch.setattr(
        openai,
        "_configure_mlx_cache_limit",
        lambda _args: {"configured": False},
    )

    def stop_after_load(model, mtp, contract, **kwargs):
        captured["model"] = model
        captured["mtp"] = mtp
        captured["contract"] = contract
        captured["kwargs"] = kwargs
        raise RuntimeError("stop after load")

    monkeypatch.setattr(openai, "load", stop_after_load)

    args = parse_args(
        [
            "--model",
            "models/Step-3.7-Flash-MTPLX-step3p5",
            "--warmup-tokens",
            "0",
            "--mtp-adapter",
            "outputs/adapters/c4-mtp-adapter-20260603-134243-r4.npz",
            "--mtp-quant-bits",
            "4",
            "--mtp-quant-group-size",
            "64",
            "--mtp-quant-mode",
            "affine",
            "--merge-mtp-adapter",
        ]
    )

    with pytest.raises(RuntimeError, match="stop after load"):
        openai.ServerState(args)

    assert captured["model"] == "models/Step-3.7-Flash-MTPLX-step3p5"
    assert captured["mtp"] is True
    assert captured["contract"].mtp_quant_bits == 4
    assert captured["contract"].mtp_quant_group_size == 64
    assert captured["contract"].mtp_quant_mode == "affine"
    assert captured["kwargs"]["mtp_adapter"] == "outputs/adapters/c4-mtp-adapter-20260603-134243-r4.npz"
    assert captured["kwargs"]["merge_mtp_adapter"] is True


def test_normalize_stop_sequences_accepts_string_list_and_caps_at_four():
    assert openai._normalize_stop_sequences(None) == []
    assert openai._normalize_stop_sequences("END") == ["END"]
    assert openai._normalize_stop_sequences(["a", "", "b", None, "a"]) == ["a", "b"]
    assert openai._normalize_stop_sequences(["1", "2", "3", "4", "5", "6"]) == [
        "1",
        "2",
        "3",
        "4",
    ]
    assert openai._normalize_stop_sequences({"bogus": True}) == []
    assert openai._normalize_stop_sequences(123) == []


def test_stop_sequence_stream_monitor_holds_back_partial_matches():
    monitor = openai._StopSequenceStreamMonitor(["END"])
    assert monitor.feed("hello E") == "hello "
    assert monitor.feed("N") == ""
    assert monitor.feed("D tail") == ""
    assert monitor.stopped is True
    assert monitor.matched_stop == "END"
    assert monitor.feed("more") == ""
    assert monitor.flush() == ""
    assert monitor.emitted_text == "hello "

    released = openai._StopSequenceStreamMonitor(["END"])
    assert released.feed("hello E") == "hello "
    assert released.feed("xtra") == "Extra"
    assert released.stopped is False

    flushed = openai._StopSequenceStreamMonitor(["END"])
    assert flushed.feed("abc E") == "abc "
    assert flushed.flush() == "E"
    assert flushed.emitted_text == "abc E"


def test_chat_stream_stop_sequence_trims_and_cancels_generation(monkeypatch):
    state = _fake_streaming_session_state()
    state.args.stream_interval = 1
    client = TestClient(create_app(state))
    cancel_seen: dict[str, bool] = {}

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        token_callback = kwargs["token_callback"]
        cancel_event = kwargs["cancel_event"]
        token_callback([ord(char) for char in "Hello "])
        token_callback([ord(char) for char in "STOP\n"])
        assert cancel_event.wait(timeout=10), "stop match must cancel generation"
        cancel_seen["cancelled"] = True
        token_callback([ord(char) for char in "after"])
        raise AssertionError("cancelled token callback must raise")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-mtplx-cache-mode": "bypass",
            "x-mtplx-allow-client-controls": "1",
        },
        json={
            "messages": [{"role": "user", "content": "Say hello"}],
            "enable_thinking": False,
            "stream": True,
            "max_tokens": 32,
            "stop": ["STOP"],
        },
    )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    content = "".join(
        payload["choices"][0].get("delta", {}).get("content", "")
        for payload in payloads
    )
    assert content == "Hello "
    final = [
        payload
        for payload in payloads
        if payload["choices"][0].get("finish_reason")
    ]
    assert final[-1]["choices"][0]["finish_reason"] == "stop"
    assert final[-1]["mtplx_stats"]["stop_sequence_hit"] is True
    assert final[-1]["mtplx_stats"]["stop_sequence_matched"] == "STOP"
    assert cancel_seen.get("cancelled") is True
    assert "data: [DONE]" in response.text


def test_chat_stream_stop_sequence_handles_generation_done_race(monkeypatch):
    state = _fake_streaming_session_state()
    state.args.stream_interval = 1
    client = TestClient(create_app(state))

    def fake_run_generation(_state, prompt_ids, **kwargs):
        token_callback = kwargs["token_callback"]
        text = "Hello STOP world\n"
        tokens = [ord(char) for char in text]
        token_callback(tokens)
        return {
            "text": text,
            "tokens": tokens,
            "stats": {
                **kwargs["request_observability"],
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": len(tokens),
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(tokens),
            "finish_reason": "length",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-mtplx-cache-mode": "bypass",
            "x-mtplx-allow-client-controls": "1",
        },
        json={
            "messages": [{"role": "user", "content": "Say hello"}],
            "enable_thinking": False,
            "stream": True,
            "max_tokens": 32,
            "stop": "STOP",
        },
    )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    content = "".join(
        payload["choices"][0].get("delta", {}).get("content", "")
        for payload in payloads
    )
    assert content == "Hello "
    final = [
        payload
        for payload in payloads
        if payload["choices"][0].get("finish_reason")
    ]
    assert final[-1]["choices"][0]["finish_reason"] == "stop"
    assert final[-1]["mtplx_stats"]["stop_sequence_hit"] is True


def test_chat_nonstream_stop_sequence_aborts_generation_early(monkeypatch):
    state = _fake_streaming_session_state()
    client = TestClient(create_app(state))

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        token_callback = kwargs["token_callback"]
        token_callback([ord(char) for char in "Hello "])
        token_callback([ord(char) for char in "STOP\n"])
        raise AssertionError("stop match must abort generation via callback")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-mtplx-cache-mode": "bypass",
            "x-mtplx-allow-client-controls": "1",
        },
        json={
            "messages": [{"role": "user", "content": "Say hello"}],
            "enable_thinking": False,
            "max_tokens": 32,
            "stop": ["STOP"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "Hello "
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == len("Hello ") + len("STOP\n")
    assert body["mtplx_stats"]["stop_sequence_hit"] is True
    assert body["mtplx_stats"]["stop_sequence_matched"] == "STOP"


def test_chat_nonstream_stop_sequence_post_trim_fallback(monkeypatch):
    state = _fake_streaming_session_state()
    client = TestClient(create_app(state))

    def fake_run_generation(_state, prompt_ids, **kwargs):
        token_callback = kwargs.get("token_callback")
        text = "abcSTOPdef"
        tokens = [ord(char) for char in text]
        if token_callback is not None:
            # No whitespace boundary: the incremental decoder holds the whole
            # word back, so only the post-trim fallback can catch the match.
            token_callback(tokens)
        return {
            "text": text,
            "tokens": tokens,
            "stats": {
                **kwargs["request_observability"],
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": len(tokens),
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(tokens),
            "finish_reason": "length",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-mtplx-cache-mode": "bypass",
            "x-mtplx-allow-client-controls": "1",
        },
        json={
            "messages": [{"role": "user", "content": "Say hello"}],
            "enable_thinking": False,
            "max_tokens": 32,
            "stop": ["STOP"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "abc"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["mtplx_stats"]["stop_sequence_matched"] == "STOP"


def test_anthropic_stop_sequences_trim_nonstream(monkeypatch):
    state = _fake_streaming_session_state()
    client = TestClient(create_app(state))

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        token_callback = kwargs["token_callback"]
        token_callback([ord(char) for char in "Hello "])
        token_callback([ord(char) for char in "STOP\n"])
        raise AssertionError("stop match must abort generation via callback")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/messages",
        headers={
            "x-mtplx-cache-mode": "bypass",
            "x-mtplx-allow-client-controls": "1",
        },
        json={
            "model": "mtplx-test-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Say hello"}],
            "stop_sequences": ["STOP"],
            "thinking": {"type": "disabled"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    text_blocks = [
        block for block in body["content"] if block.get("type") == "text"
    ]
    assert text_blocks
    assert text_blocks[0]["text"] == "Hello "
    # A stop_sequences match must surface per the Anthropic wire contract
    # (QA-117): stop_reason="stop_sequence" plus the matched string —
    # previously this flattened to end_turn / null.
    assert body["stop_reason"] == "stop_sequence"
    assert body["stop_sequence"] == "STOP"


def test_completions_stream_is_incremental_with_terminal_finish_reason(monkeypatch):
    state = _fake_streaming_session_state()
    client = TestClient(create_app(state))
    first_batch = "alpha bravo charlie delta echo foxtrot\n"
    second_batch = "golf hotel india juliet kilo lima\n"

    def fake_run_generation(_state, prompt_ids, **kwargs):
        token_callback = kwargs["token_callback"]
        token_callback([ord(char) for char in first_batch])
        token_callback([ord(char) for char in second_batch])
        text = first_batch + second_batch
        tokens = [ord(char) for char in text]
        return {
            "text": text,
            "tokens": tokens,
            "stats": {
                **kwargs["request_observability"],
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": len(tokens),
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(tokens),
            "finish_reason": "length",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/completions",
        json={"prompt": "count", "max_tokens": 8, "stream": True},
    )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    texts = [
        payload["choices"][0]["text"]
        for payload in payloads
        if payload["choices"][0]["text"]
    ]
    # Chunks must follow generation batches, not post-hoc 24-char slices of
    # the final text (the old pseudo-stream behavior).
    assert texts == [first_batch, second_batch]
    final = [
        payload
        for payload in payloads
        if payload["choices"][0].get("finish_reason")
    ]
    assert final[-1]["choices"][0]["finish_reason"] == "length"
    assert final[-1]["usage"]["completion_tokens"] == len(
        first_batch + second_batch
    )
    assert "data: [DONE]" in response.text


def test_completions_stream_honors_stop_sequence(monkeypatch):
    state = _fake_streaming_session_state()
    client = TestClient(create_app(state))
    cancel_seen: dict[str, bool] = {}

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        token_callback = kwargs["token_callback"]
        cancel_event = kwargs["cancel_event"]
        token_callback([ord(char) for char in "Hello "])
        token_callback([ord(char) for char in "STOP\n"])
        assert cancel_event.wait(timeout=10), "stop match must cancel generation"
        cancel_seen["cancelled"] = True
        token_callback([ord(char) for char in "after"])
        raise AssertionError("cancelled token callback must raise")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/completions",
        json={
            "prompt": "say hello",
            "max_tokens": 32,
            "stream": True,
            "stop": ["STOP"],
        },
    )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    texts = [
        payload["choices"][0]["text"]
        for payload in payloads
        if payload["choices"][0]["text"]
    ]
    assert texts == ["Hello "]
    final = [
        payload
        for payload in payloads
        if payload["choices"][0].get("finish_reason")
    ]
    assert final[-1]["choices"][0]["finish_reason"] == "stop"
    assert final[-1]["mtplx_stats"]["stop_sequence_hit"] is True
    assert cancel_seen.get("cancelled") is True
    assert "data: [DONE]" in response.text


def test_completions_nonstream_trims_stop_and_reports_real_finish_reason(
    monkeypatch,
):
    state = _fake_streaming_session_state()
    client = TestClient(create_app(state))

    def fake_run_generation(_state, prompt_ids, **kwargs):
        token_callback = kwargs.get("token_callback")
        text = "abcSTOPdef"
        tokens = [ord(char) for char in text]
        if token_callback is not None:
            token_callback(tokens)
        return {
            "text": text,
            "tokens": tokens,
            "stats": {
                **kwargs["request_observability"],
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": len(tokens),
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(tokens),
            "finish_reason": "length",
        }

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    stopped = client.post(
        "/v1/completions",
        json={"prompt": "say abc", "max_tokens": 32, "stop": "STOP"},
    )
    unstopped = client.post(
        "/v1/completions",
        json={"prompt": "say abc", "max_tokens": 32},
    )

    assert stopped.status_code == 200
    stopped_body = stopped.json()
    assert stopped_body["choices"][0]["text"] == "abc"
    assert stopped_body["choices"][0]["finish_reason"] == "stop"
    assert stopped_body["mtplx_stats"]["stop_sequence_matched"] == "STOP"
    assert unstopped.status_code == 200
    unstopped_body = unstopped.json()
    assert unstopped_body["choices"][0]["text"] == "abcSTOPdef"
    assert unstopped_body["choices"][0]["finish_reason"] == "length"


def test_completions_nonstream_stop_aborts_generation_early(monkeypatch):
    state = _fake_streaming_session_state()
    client = TestClient(create_app(state))

    def fake_run_generation(_state, _prompt_ids, **kwargs):
        token_callback = kwargs["token_callback"]
        token_callback([ord(char) for char in "Hello "])
        token_callback([ord(char) for char in "STOP\n"])
        raise AssertionError("stop match must abort generation via callback")

    monkeypatch.setattr(openai, "_run_generation", fake_run_generation)

    response = client.post(
        "/v1/completions",
        json={"prompt": "say hello", "max_tokens": 32, "stop": ["STOP"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["text"] == "Hello "
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == len("Hello ") + len("STOP\n")
    assert body["mtplx_stats"]["stop_sequence_hit"] is True


def test_anthropic_messages_bare_tools_first_request_completes(monkeypatch):
    # Issue #86's minimal reproduction shape: a fresh server's FIRST
    # /v1/messages request with a tiny prompt, a bare tools array, and a small
    # output budget must stream to message_stop rather than hanging. This
    # guards the request path (parse -> render -> schedule -> stream) for the
    # exact payload; the live deadlock itself is contained separately by the
    # stream stall watchdog (MTPLX_STREAM_STALL_DEADLINE_S).
    client = TestClient(create_app(_fake_state()))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    monkeypatch.setattr(
        openai, "_run_generation", _fake_streaming_generation("On it.")
    )

    response = client.post(
        "/v1/messages",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "model": "mtplx-test-model",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "name": "echo",
                    "description": "Echo a string back.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "message_start" in response.text
    assert "message_stop" in response.text
    assert "On it." in response.text
    assert '"type": "error"' not in response.text


class _ThinkMarkerTokenizer:
    """encode() mirroring Qwen: think markers are single dedicated ids."""

    _VOCAB = {"<think>": [151667], "</think>": [151668]}

    def encode(self, text, add_special_tokens=False):
        if text in self._VOCAB:
            return list(self._VOCAB[text])
        return [11, 22]


def _guard_request_state(argv):
    return SimpleNamespace(
        args=parse_args(["--warmup-tokens", "0", *argv]),
        runtime=SimpleNamespace(tokenizer=_ThinkMarkerTokenizer()),
    )


_AGENT_LANE_OBS = {
    "request_enable_thinking": True,
    "request_tool_count": 4,
    "request_reasoning_effort": "high",
}


def test_thinking_guard_default_is_off(monkeypatch):
    monkeypatch.delenv("MTPLX_THINKING_BUDGET", raising=False)
    config = openai._thinking_guard_config_for_request(
        _guard_request_state([]),
        prompt_ids=[1, 2, 3],
        request_observability=dict(_AGENT_LANE_OBS),
    )
    assert config is None


def test_thinking_guard_auto_optin_maps_effort(monkeypatch):
    monkeypatch.delenv("MTPLX_THINKING_BUDGET", raising=False)
    config = openai._thinking_guard_config_for_request(
        _guard_request_state(["--agent-thinking-budget", "auto"]),
        prompt_ids=[1, 2, 3],
        request_observability=dict(_AGENT_LANE_OBS),
    )
    assert config is not None and config.enabled
    assert config.budget_tokens == 6144


def test_thinking_guard_integer_optin_sets_budget(monkeypatch):
    monkeypatch.delenv("MTPLX_THINKING_BUDGET", raising=False)
    config = openai._thinking_guard_config_for_request(
        _guard_request_state(["--agent-thinking-budget", "1536"]),
        prompt_ids=[1, 2, 3],
        request_observability=dict(_AGENT_LANE_OBS),
    )
    assert config is not None and config.enabled
    assert config.budget_tokens == 1536


def test_thinking_guard_env_optin_beats_default_off(monkeypatch):
    monkeypatch.setenv("MTPLX_THINKING_BUDGET", "2048")
    config = openai._thinking_guard_config_for_request(
        _guard_request_state([]),
        prompt_ids=[1, 2, 3],
        request_observability=dict(_AGENT_LANE_OBS),
    )
    assert config is not None and config.enabled
    assert config.budget_tokens == 2048


def test_thinking_guard_never_touches_plain_chat_even_opted_in(monkeypatch):
    monkeypatch.setenv("MTPLX_THINKING_BUDGET", "2048")
    state = _guard_request_state(["--agent-thinking-budget", "auto"])
    no_tools = openai._thinking_guard_config_for_request(
        state,
        prompt_ids=[1, 2, 3],
        request_observability={
            "request_enable_thinking": True,
            "request_tool_count": 0,
        },
    )
    no_thinking = openai._thinking_guard_config_for_request(
        state,
        prompt_ids=[1, 2, 3],
        request_observability={
            "request_enable_thinking": False,
            "request_tool_count": 4,
        },
    )
    assert no_tools is None and no_thinking is None
# --- parallel_tool_calls wiring ---------------------------------------------


def _two_call_extraction(*_args, **_kwargs):
    def _call(name):
        return {
            "id": f"call_{name}",
            "type": "function",
            "function": {"name": name, "arguments": "{\"q\": 1}"},
        }

    return SimpleNamespace(
        cleaned_text="",
        cleaned_thinking="",
        tool_calls=[_call("session_status"), _call("session_status")],
        parser_source="native",
        status="parsed",
        malformed_reason=None,
        raw_tool_markup_suppressed=True,
    )


def test_parallel_tool_calls_false_truncates_nonstream_tool_calls(monkeypatch):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    monkeypatch.setattr(openai, "_run_generation", lambda *a, **k: _fake_generation("x"))
    monkeypatch.setattr(
        openai, "omlx_extract_tool_calls_with_thinking", _two_call_extraction
    )
    client = TestClient(create_app(state))
    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "status twice"}],
            "tools": [_tool_schema()],
            "parallel_tool_calls": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["choices"][0]["message"]["tool_calls"]) == 1
    assert body["mtplx_stats"]["tool_calls_emitted"] == 1


def test_parallel_tool_calls_unset_keeps_all_nonstream_tool_calls(monkeypatch):
    state = _fake_state()
    state.runtime.tokenizer = CaptureTokenizer()
    monkeypatch.setattr(openai, "_run_generation", lambda *a, **k: _fake_generation("x"))
    monkeypatch.setattr(
        openai, "omlx_extract_tool_calls_with_thinking", _two_call_extraction
    )
    client = TestClient(create_app(state))
    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "status twice"}],
            "tools": [_tool_schema()],
        },
    )
    assert response.status_code == 200
    assert len(response.json()["choices"][0]["message"]["tool_calls"]) == 2


def test_single_tool_call_stream_policy_declared_field_wins():
    policy = openai._single_tool_call_stream_policy
    # Declared field is authoritative in both directions.
    assert policy(parallel_tool_calls=False, client_hint="", explicit_single_tool=False)
    assert not policy(
        parallel_tool_calls=True, client_hint="pi", explicit_single_tool=True
    )
    # Unset falls back to the legacy client-hint heuristics.
    assert policy(parallel_tool_calls=None, client_hint="pi", explicit_single_tool=False)
    assert policy(
        parallel_tool_calls=None, client_hint="opencode", explicit_single_tool=True
    )
    assert not policy(
        parallel_tool_calls=None, client_hint="opencode", explicit_single_tool=False
    )
    assert not policy(parallel_tool_calls=None, client_hint="", explicit_single_tool=False)


def test_request_parallel_tool_calls_only_honors_booleans():
    assert openai._request_parallel_tool_calls(SimpleNamespace(parallel_tool_calls=True)) is True
    assert openai._request_parallel_tool_calls(SimpleNamespace(parallel_tool_calls=False)) is False
    assert openai._request_parallel_tool_calls(SimpleNamespace(parallel_tool_calls="yes")) is None
    assert openai._request_parallel_tool_calls(SimpleNamespace()) is None


def test_parallel_tool_calls_false_streaming_android_studio_shape(monkeypatch):
    # The real client shape from tests/fixtures/android_studio_issue58_chat.json:
    # no client-identifying header, stream:true, stream_options include_usage,
    # and an explicit parallel_tool_calls:false that must drive single-tool
    # truncation on its own — no client-name sniffing involved.
    state = _fake_state()
    state.args.stream_interval = 1
    state.args.stats_footer = False
    client = TestClient(create_app(state))
    monkeypatch.setattr(openai, "_encode_messages", lambda *_args, **_kwargs: [1, 2, 3])
    generated = (
        "<tool_call>\n<function=bash>\n"
        "<parameter=command>\nls\n</parameter>\n"
        "<parameter=description>\nList files\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=bash>\n"
        "<parameter=command>\npwd\n</parameter>\n"
        "<parameter=description>\nPrint cwd\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    monkeypatch.setattr(
        openai,
        "_run_generation",
        _fake_streaming_generation(generated),
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"x-mtplx-cache-mode": "bypass"},
        json={
            "messages": [{"role": "user", "content": "List files, then print cwd."}],
            "tools": [_bash_tool_schema()],
            "stream": True,
            "stream_options": {"include_usage": True},
            "parallel_tool_calls": False,
            "max_tokens": 128,
        },
    )

    assert response.status_code == 200
    payloads = _stream_payloads(response.text)
    tool_deltas = [
        item
        for payload in payloads
        if payload.get("choices")
        for item in payload["choices"][0]["delta"].get("tool_calls", [])
    ]
    arguments = "".join(
        item.get("function", {}).get("arguments", "") for item in tool_deltas
    )
    final = [
        payload
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"]
    ]
    assert '"command":"ls"' in arguments
    assert "pwd" not in arguments
    assert final[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert final[-1]["mtplx_stats"]["tool_calls_emitted"] == 1


def test_run_generation_passes_draft_core_from_serve_args(monkeypatch):
    state = _fake_streaming_session_state()
    state.draft_sampler = None
    state.requests_completed = 0
    state.args.draft_core = "device"
    captured: dict[str, object] = {}

    def fake_generate_mtpk(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            tokens=[],
            text="",
            stats=SimpleNamespace(
                to_dict=lambda: {
                    "prompt_eval_time_s": 0.0,
                    "generated_tokens": 0,
                    "elapsed_s": 0.0,
                    "tok_s": 0.0,
                }
            ),
            final_state=None,
        )

    monkeypatch.setattr(openai, "generate_mtpk", fake_generate_mtpk)

    openai._run_generation(
        state,
        [1, 2, 3],
        max_tokens=1,
        temperature=None,
        top_p=None,
        top_k=None,
        seed=None,
        generation_mode="mtp",
        depth=1,
        session_id="sess-draft-core",
        session_bank=state.sessions.bank,
        session_template_hash=state.template_hash,
        session_draft_head_identity=state.draft_head_identity,
        session_policy_fingerprint="policy",
    )
    assert captured["draft_core"] == "device"

    # serve args without the attribute (older callers) fall back to stock
    del state.args.draft_core
    captured.clear()
    openai._run_generation(
        state,
        [1, 2, 3],
        max_tokens=1,
        temperature=None,
        top_p=None,
        top_k=None,
        seed=None,
        generation_mode="mtp",
        depth=1,
        session_id="sess-draft-core-2",
        session_bank=state.sessions.bank,
        session_template_hash=state.template_hash,
        session_draft_head_identity=state.draft_head_identity,
        session_policy_fingerprint="policy",
    )
    assert captured["draft_core"] == "stock"
def _postcommit_route_state(*, mtp_enabled: bool):
    """A minimal ServerState stub for exercising the postcommit snapshot
    routing decision (mtp vs ar) without real generation."""

    def _boom_mtp_cache():
        # LagunaARRuntime.make_mtp_cache raises this; reaching it means the
        # postcommit took the MTP-only history branch on an AR-only runtime.
        raise RuntimeError("MTP is not enabled for this runtime")

    class _RouteBank:
        per_session_max_bytes = 0

        def __init__(self) -> None:
            self.puts: list[dict] = []

        def longest_prefix(self, _token_ids):
            return None

        def put(self, **kwargs):
            self.puts.append(kwargs)
            return SimpleNamespace(
                prefix_len=len(kwargs["token_ids"]),
                nbytes=1,
                token_hash="route-hash",
            )

    foreground = ForegroundState()
    return SimpleNamespace(
        runtime=SimpleNamespace(
            mtp_enabled=mtp_enabled,
            make_mtp_cache=_boom_mtp_cache,
            tokenizer=SimpleNamespace(),
        ),
        sessions=SimpleNamespace(bank=_RouteBank()),
        lock=foreground.lock,
        begin_foreground=foreground.begin_foreground,
        end_foreground=foreground.end_foreground,
        template_hash="tpl",
        draft_head_identity="head",
    )


def _make_route_restore(captured: dict):
    from mtplx.generation import _mtp_history_uses_committed_cache

    def _fake_restore(rt, prompt_ids, *, mtp_history_policy, **_kwargs):
        captured["restore_policy"] = mtp_history_policy
        if _mtp_history_uses_committed_cache(mtp_history_policy):
            # The real committed-history prefill builds an MTP history cache;
            # on an AR runtime that call is exactly what raised in production.
            rt.make_mtp_cache()
        return SimpleNamespace(
            trunk_cache=["cache"],
            logits="logits",
            hidden="hidden",
            committed_mtp_cache=None,
            gdn_boundaries=[],
            mtp_history_policy=mtp_history_policy,
            cache_hit=False,
            cached_tokens=0,
            suffix_tokens=len(prompt_ids),
            cache_source="none",
            ssd_cache_hit=False,
            ssd_cached_tokens=0,
            ssd_restore_s=0.0,
            cache_miss_reason=None,
        )

    return _fake_restore


def test_ar_postcommit_snapshot_routes_through_ar_path(monkeypatch):
    """A laguna_ar/AR-mode postcommit must re-prefill history through the AR
    (cycle) path — never the committed MTP-history branch that calls
    make_mtp_cache() and raises 'MTP is not enabled for this runtime'."""

    state = _postcommit_route_state(mtp_enabled=False)
    monkeypatch.setattr(
        openai,
        "_history_ids_for_postcommit",
        lambda *_a, **_k: ([1, 2, 3, 4], None),
    )
    captured: dict = {}
    monkeypatch.setattr(
        openai, "restore_or_prefill_prompt_state", _make_route_restore(captured)
    )

    result = openai._store_retokenized_history_snapshot(
        state,
        session_id="sess-1",
        messages=[],
        assistant_content="hello",
        thinking_enabled=False,
        policy_fingerprint="pf",
    )

    # Routed through the AR path (never the MTP-only committed branch).
    assert captured["restore_policy"] == "cycle"
    assert result["stored"] is True
    # The banked entry's policy metadata matches what the next AR turn looks up.
    assert state.sessions.bank.puts
    assert state.sessions.bank.puts[0]["mtp_history_policy"] == "cycle"


def test_mtp_postcommit_snapshot_keeps_committed_policy(monkeypatch):
    """MTP-capable runtimes are unchanged: the postcommit still banks under the
    committed MTP-history policy and exercises the MTP cache."""

    state = _postcommit_route_state(mtp_enabled=True)
    made = {"mtp_cache": 0}

    def _ok_mtp_cache():
        made["mtp_cache"] += 1
        return SimpleNamespace()

    state.runtime.make_mtp_cache = _ok_mtp_cache
    monkeypatch.setattr(
        openai,
        "_history_ids_for_postcommit",
        lambda *_a, **_k: ([1, 2, 3, 4], None),
    )
    captured: dict = {}
    monkeypatch.setattr(
        openai, "restore_or_prefill_prompt_state", _make_route_restore(captured)
    )

    result = openai._store_retokenized_history_snapshot(
        state,
        session_id="sess-1",
        messages=[],
        assistant_content="hello",
        thinking_enabled=False,
        policy_fingerprint="pf",
    )

    assert captured["restore_policy"] == "committed"
    assert made["mtp_cache"] == 1
    assert result["stored"] is True
    assert state.sessions.bank.puts[0]["mtp_history_policy"] == "committed"
