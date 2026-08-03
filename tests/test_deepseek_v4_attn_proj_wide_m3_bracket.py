"""Fail-closed contracts for the full-workload attention-projection M3 bracket."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from test_deepseek_v4_adaptive_width_bracket import _common as adaptive_common


ROOT = Path(__file__).parents[1]


def _bench():
    path = ROOT / "scripts" / "deepseek_v4_mtpk_bench.py"
    spec = importlib.util.spec_from_file_location("dsv4_attn_proj_bench", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def _event(width: int) -> dict:
    margins = {1: [0.1], 2: [0.5, 0.5], 3: [0.5, 10.5]}[width]
    return {
        "depth": 3,
        "drafts": [{}] * width,
        "gated_stop_depth": width if width < 3 else None,
        "adaptive_width_policy": {
            "kind": "deepseek_v4_preregistered_max_k3",
            "eligible_full_k3": True,
            "d1_margin_threshold": 0.25,
            "d2_margin_threshold": 10.0,
            "decision_margins": margins,
            "selected_draft_depth": width,
            "target_rows": width + 1,
        },
    }


def _arm(label: str, *, selected: bool, tps: float = 41.0, flip: int | None = None):
    tokens = list(range(256))
    if flip is not None:
        tokens[17] = flip
    events = [_event(1)] * 3 + [_event(2)] * 81 + [_event(3)] * 9
    stats = {
        "events": events,
        "generated_tokens": 256,
        "accepted_by_depth": [85, 54, 1],
        "drafted_by_depth": [93, 90, 9],
        "accepted_drafts": 140,
        "rejected_drafts": 44,
        "drafted_tokens": 192,
        "skipped_drafts": 0,
        "bonus_tokens": 48,
        "correction_tokens": 0,
        "verify_calls": 93,
        "mtp_forward_calls": 192,
        "make_mtp_cache_calls": 1,
        "update_mtp_cache_calls": 89,
        "mtp_history_append_calls": 89,
        "forward_ar_hidden_calls": 97,
        "forward_ar_plain_calls": 0,
    }
    return {
        "label": label,
        "error": None,
        "generated_tokens": 256,
        "finish_reason": "length",
        "tokens": tokens,
        "token_sha256": _sha(tokens),
        "decode_tokens_per_second": tps,
        "stats_full": stats,
        "attn_proj_wide_m3_binding": {
            "selected": selected,
            "projections": 43,
            "original_stock_modules": 0 if selected else 43,
            "candidate_modules": 43 if selected else 0,
        },
    }


def _route():
    return {
        "route": "target_verify_m3_original_q4_attention_projections",
        "logical_input_shape": [1, 3, 1024],
        "body_wq_b_prepared": 43,
        "body_indexer_wq_b_prepared": 0,
        "body_indexer_wq_b_stock": 21,
        "total_q4_projections_prepared": 43,
        "main_geometry": {"k": 1024, "n": 32768, "layers": 43},
        "indexer_geometry_stock": {"k": 1024, "n": 8192, "layers": 21},
        "indexer_activation_threshold_rows": 512,
        "canonical_max_compressed_rows": 146,
        "quantization": "affine_q4_g64",
        "activation_dtype": "bfloat16",
        "mtp_attention_dense_stock": 1,
        "o_lora_stock": 86,
        "small_attention_projections_stock": True,
        "mla_sdpa_cache_stock": True,
        "other_target_widths_stock": [2, 4],
        "ar_prefill_repair_mtp_stock": True,
        "kernel_selfcheck_exact": True,
        "both_arms_preinstalled": True,
        "arm_selection": "between_generations",
        "in_generation_module_rewrites": False,
    }


def _receipt(bench, *, arms=None, common=None):
    common = adaptive_common(bench) if common is None else common
    common["launch_mtplx_env"] = dict(bench._ATTN_PROJ_WIDE_M3_STAGE4_ENV)
    common["deepseek_v4_attn_proj_wide_m3"] = _route()
    common.setdefault(
        "diagnostic_profiler_evidence",
        dict(bench._ATTN_PROJ_WIDE_M3_PROFILER_EVIDENCE),
    )
    arms = arms or [
        _arm("CURRENT-PRIMER", selected=False, tps=40.0),
        _arm("CURRENT-C0", selected=False, tps=41.0),
        _arm("ATTN-PROJ-M3-B", selected=True, tps=42.0),
        _arm("CURRENT-C1", selected=False, tps=41.2),
    ]
    return bench._attn_proj_wide_m3_bracket_receipt(
        common=common,
        arms=arms,
        process_pid=7,
        model_object_id=9,
        policy_receipt=dict(bench._ADAPTIVE_WIDTH_POLICY_RECEIPT),
        route_report=_route(),
    )


def test_valid_one_load_receipt_proves_3483_m3_projection_opportunities():
    bench = _bench()
    receipt = _receipt(bench)
    assert receipt["status"] == 0
    assert receipt["single_process_bracket"]["model_load_count"] == 1
    assert receipt["single_process_bracket"]["execution_order"] == [
        "CURRENT-PRIMER",
        "CURRENT-C0",
        "ATTN-PROJ-M3-B",
        "CURRENT-C1",
    ]
    assert receipt["policy_engagement"]["ATTN-PROJ-M3-B"][
        "eligible_target_m3_projection_calls"
    ] == 3483
    assert receipt["token_quality"]["mode"] == "exact"


def test_near_tie_flip_is_reported_under_authorized_bf16_policy():
    bench = _bench()
    arms = [
        _arm("CURRENT-PRIMER", selected=False),
        _arm("CURRENT-C0", selected=False),
        _arm("ATTN-PROJ-M3-B", selected=True, flip=99999),
        _arm("CURRENT-C1", selected=False),
    ]
    receipt = _receipt(bench, arms=arms)
    assert receipt["status"] == 0
    assert receipt["token_quality"]["mode"] == "bf16_near_tie_reported"
    assert receipt["token_quality"]["human_eval"] == "deferred_by_authorized_policy"


def test_receipt_fails_closed_on_binding_control_or_profiler_drift():
    bench = _bench()
    arms = [
        _arm("CURRENT-PRIMER", selected=False),
        _arm("CURRENT-C0", selected=False),
        _arm("ATTN-PROJ-M3-B", selected=True),
        _arm("CURRENT-C1", selected=False),
    ]
    arms[2]["attn_proj_wide_m3_binding"]["candidate_modules"] = 42
    assert _receipt(bench, arms=arms)["status"] == 1
    arms[2]["attn_proj_wide_m3_binding"]["candidate_modules"] = 43
    arms[3]["tokens"][0] = 99
    assert _receipt(bench, arms=arms)["status"] == 1
    arms[3]["tokens"][0] = 0
    common = adaptive_common(bench)
    common["diagnostic_profiler_evidence"] = {"receipt_sha256": "bad"}
    assert _receipt(bench, arms=arms, common=common)["status"] == 1


def test_guarded_wrapper_requires_primary_success_and_postflight(tmp_path):
    path = ROOT / "scripts" / "deepseek_v4_attn_proj_wide_m3_guarded.py"
    spec = importlib.util.spec_from_file_location("dsv4_attn_proj_guard", path)
    wrapper = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(wrapper)
    tag = "attn-proj-test"
    (tmp_path / f"{tag}.json").write_text(
        json.dumps({
            "status": 0,
            "receipt_role": "attn_proj_wide_m3_performance_bracket",
        })
    )
    postflight = {
        name: {"ok": True}
        for name in ("lock_free", "wired_limit_mb", "quality_models", "quality_ready_chat")
    }
    assert wrapper.run(
        tag,
        bench_dir=tmp_path,
        run_command=lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
        postflight_collector=lambda: postflight,
    ) == 0
