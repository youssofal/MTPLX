from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/qwen38_dflash2_matrix.py"


def _module():
    spec = importlib.util.spec_from_file_location("qwen38_dflash2_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scenarios_match_the_final_campaign_contract() -> None:
    matrix = _module()

    vanity = matrix.scenario("vanity", 100)
    assert vanity.conditioner_tokens == 0
    assert vanity.max_tokens == 1_024
    assert vanity.temperature == 0.0
    assert vanity.reasoning_effort is None
    assert matrix.order_for_context("vanity", 100) == ("dflash2", "dflash2")

    low = matrix.scenario("low", 16_384)
    assert low.conditioner_tokens == 1_024
    assert low.max_tokens == 1_024
    assert low.reasoning_effort == "low"
    assert matrix.order_for_context("low", 16_384) == ("dflash2", "dflash2")
    assert matrix.order_for_context("low", 131_072) == ("dflash2",)

    xhigh = matrix.scenario("xhigh", 1_024)
    assert xhigh.conditioner_tokens == 1_024
    assert xhigh.max_tokens == 1_024
    assert xhigh.reasoning_effort == "xhigh"
    assert matrix.order_for_context("xhigh", 1_024) == ("dflash2", "dflash2")


def test_xhigh_dflash_rejects_every_context_except_1k() -> None:
    matrix = _module()

    try:
        matrix.scenario("xhigh", 16_384)
    except ValueError as exc:
        assert "DFlash2 xhigh is restricted to 1024" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("DFlash2 xhigh accepted an unrequested context")


def test_child_command_uses_the_frozen_prompt_and_conditioner_contract(
    tmp_path: Path,
) -> None:
    matrix = _module()
    scenario = matrix.scenario("low", 1_024)
    command = matrix.child_command(
        python=tmp_path / "python",
        source_root=tmp_path / "pr335",
        source_commit=matrix.PR335_SOURCE_COMMIT,
        model=tmp_path / "target",
        draft=tmp_path / "draft",
        prompt_file=tmp_path / "prompt.jsonl",
        context_file=tmp_path / "context.py",
        lock=tmp_path / "gpu.lock",
        output=tmp_path / "receipt.json",
        scenario=scenario,
    )

    assert command[0] == str((tmp_path / "python").absolute())
    assert "--dflash2-adaptive" in command
    assert command[command.index("--draft-block-size") + 1] == "8"
    assert command[command.index("--conditioner-tokens") + 1] == "1024"
    assert command[command.index("--conditioner-mode") + 1] == "same_prompt"
    assert command[command.index("--reasoning-effort") + 1] == "low"
    assert command[command.index("--max-tokens") + 1] == "1024"


def test_aggregate_rejects_mismatched_or_fallback_receipts() -> None:
    matrix = _module()
    scenario = matrix.scenario("vanity", 100)

    def receipt(token_hash: str) -> dict:
        return {
            "kind": "qwen38_dflash2_frozen_matrix_arm",
            "engine": "pr_dflash2",
            "source_commit": matrix.PR335_SOURCE_COMMIT,
            "harness_commit": "a" * 40,
            "mlx_version": "0.32.2",
            "mlx_metal_version": "0.32.2",
            "gpu_lock_scope": str(Path("/tmp/mtplx-gpu-exclusive.lock").resolve()),
            "model": str(Path("/models/target").resolve()),
            "draft": str(Path("/models/draft").resolve()),
            "stack": {
                "profile": "turbo",
                "dflash_block_size": 8,
                "native_mtp_loaded": False,
                "feature_receipt": {
                    "adaptive_policy": {"active": True},
                    "context_route": {
                        "effective_adaptive": True,
                        "row21_active": True,
                        "row24_decode_active": True,
                        "row24_prefill_active": True,
                        "row48_decode_active": True,
                        "row48_prefill_active": True,
                        "row50_active": True,
                    },
                    "dflash_gqa_widths": {"active": True},
                    "dflash_m6_barrier_free_kp1": {"active": True},
                    "dflash_m8_nax_island": {
                        "active": True,
                        "validated_projections": 32,
                    },
                    "r21_qk_rms_rope": {"active_modules": 16},
                    "r24_eval_ladder": {
                        "active": 1,
                        "decode_active": 1,
                        "prefill_active": 1,
                    },
                    "r24_qk_length_limit": {"active_modules": 16},
                    "r26_prefill_ladder_3": {"active": 1},
                    "r48_boundary_fused": {
                        "active_modules": 64,
                        "decode_active": 1,
                        "prefill_active": 1,
                    },
                    "r50_wired_residency": {"active": True, "installed": True},
                    "r53_command_buffers": {"active": True, "installed": True},
                },
            },
            "workload": {
                "workload": "vanity",
                "prompt_kind": "is_palindrome",
                "prompt_format": "qwen_chat_template_non_thinking",
                "enable_thinking": False,
                "reasoning_effort": None,
                "prompt_tokens": 100,
                "prompt_token_sha256": matrix.PROMPT_TOKEN_SHA256["vanity"][100],
                "prompt_artifact_sha256": matrix.PROMPT_ARTIFACT_SHA256["vanity"],
                "context_artifact_sha256": matrix.PYTHON_CONTEXT_SHA256,
                "output_limit": 1_024,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "seed": 42,
                "conditioner_prompt_tokens": 100,
                "conditioner_output_tokens": 0,
                "conditioner_mode": "same_prompt",
                "conditioner_reuses_timed_prompt": True,
                "cold_prefill": True,
                "prefix_cache_used": False,
                "requested_adaptive": True,
                "requested_draft_m": 8,
            },
            "arm": {
                "engine": "pr_dflash2",
                "prompt_tokens": 100,
                "generated_tokens": 102,
                "finish_reason": "stop",
                "prefill_tps": 800.0,
                "decode_tps": 25.0,
                "wall_s": 5.0,
                "peak_memory_gib": 20.0,
                "token_sha256": token_hash,
                "prefix_cache_used": False,
                "cached_tokens": 0,
                "session_cache_hit": False,
                "session_restore_mode": "cold",
                "requested_adaptive": True,
                "effective_adaptive": True,
                "effective_widths": [5, 6, 7, 8],
                "fallback_ar": False,
            },
        }

    rows = [receipt("a" * 64), receipt("a" * 64)]
    combined = matrix.aggregate(
        scenario=scenario,
        order=("dflash2", "dflash2"),
        receipts=rows,
        expected_harness_commit="a" * 40,
        lock=Path("/tmp/mtplx-gpu-exclusive.lock"),
        model=Path("/models/target"),
        draft=Path("/models/draft"),
    )
    assert combined["invariant_errors"] == []
    assert combined["summary"]["arms"] == 2
    assert combined["summary"]["decode_tok_s_mean"] == 25.0

    rows[1]["arm"]["token_sha256"] = "b" * 64
    rows[1]["arm"]["fallback_ar"] = True
    rejected = matrix.aggregate(
        scenario=scenario,
        order=("dflash2", "dflash2"),
        receipts=rows,
        expected_harness_commit="a" * 40,
        lock=Path("/tmp/mtplx-gpu-exclusive.lock"),
        model=Path("/models/target"),
        draft=Path("/models/draft"),
    )
    assert "paired DFlash2 tokens are not deterministic" in rejected["invariant_errors"]
    assert any("fell back to autoregressive" in error for error in rejected["invariant_errors"])
