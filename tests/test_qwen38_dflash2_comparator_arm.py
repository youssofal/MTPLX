from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).parents[1] / "scripts/qwen38_dflash2_comparator_arm.py"


def _module():
    spec = importlib.util.spec_from_file_location("qwen38_dflash2_comparator_arm", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_workload_mapping_preserves_current_matrix_prompt_contract() -> None:
    arm = _module()

    assert arm._workload(SimpleNamespace(prompt_kind="is_palindrome", reasoning_effort=None)) == "vanity"
    assert arm._workload(SimpleNamespace(prompt_kind="coding", reasoning_effort="low")) == "low"
    assert arm._workload(SimpleNamespace(prompt_kind="coding", reasoning_effort="xhigh")) == "xhigh"


def test_zero_token_conditioner_skips_dflash_generation() -> None:
    arm = _module()
    called = []

    result = arm._generate_or_skip(
        lambda *_args: called.append(True), object(), [1, 2, 3],
        SimpleNamespace(max_tokens=0),
    )

    assert result is None
    assert called == []


def test_dflash_receipt_requires_the_complete_pr335_optimized_stack() -> None:
    arm = _module()
    receipt = {
        "engine": "pr_dflash2",
        "source_commit": arm.PR335_SOURCE_COMMIT,
        "mlx_version": "0.32.2",
        "mlx_metal_version": "0.32.2",
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
                "dflash_gqa_widths": {"active": True, "eligible_modules": 16},
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
        "arm": {"fallback_ar": False},
    }

    assert arm._optimized_stack_errors(receipt) == []

    del receipt["stack"]["feature_receipt"]["r53_command_buffers"]
    assert "DFlash2 optimized feature r53_command_buffers is inactive" in (
        arm._optimized_stack_errors(receipt)
    )
    receipt["stack"]["feature_receipt"]["r53_command_buffers"] = {
        "active": True,
        "installed": True,
    }
    receipt["arm"]["fallback_ar"] = True
    assert "DFlash2 fell back to autoregressive decode" in (
        arm._optimized_stack_errors(receipt)
    )
