"""Fail-closed gates for the canonical adaptive-width performance bracket."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]
BENCH_PATH = ROOT / "scripts" / "deepseek_v4_mtpk_bench.py"


def _module():
    spec = importlib.util.spec_from_file_location("dsv4_adaptive_width_bench", BENCH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _token_sha(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def _control(label, tps=40.0):
    tokens = list(range(256))
    tokens[221] = 14042
    stats = {
        "events": [
            {"depth": 3, "drafts": [{}, {}, {}], "gated_stop_depth": None},
            {"depth": 3, "drafts": [{}, {}, {}], "gated_stop_depth": None},
        ],
        "generated_tokens": 256,
        "accepted_by_depth": [2, 1, 1],
        "drafted_by_depth": [2, 2, 2],
        "accepted_drafts": 4,
        "rejected_drafts": 1,
        "drafted_tokens": 6,
        "skipped_drafts": 0,
        "bonus_tokens": 1,
        "correction_tokens": 1,
        "verify_calls": 2,
        "mtp_forward_calls": 6,
        "make_mtp_cache_calls": 2,
        "update_mtp_cache_calls": 2,
        "mtp_history_append_calls": 2,
        "forward_ar_hidden_calls": 3,
        "forward_ar_plain_calls": 0,
    }
    return {
        "label": label,
        "error": None,
        "generated_tokens": 256,
        "finish_reason": "length",
        "tokens": tokens,
        "token_sha256": _token_sha(tokens),
        "decode_tokens_per_second": tps,
        "stats_full": stats,
    }


def _candidate(tps=42.0, *, tokens=None):
    if tokens is None:
        tokens = list(range(256))
        tokens[221] = 14042
    else:
        tokens = list(tokens)
    events = []
    for width, margins in ((1, [0.1]), (2, [0.5, 0.5]), (3, [0.5, 10.5])):
        events.append(
            {
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
        )
    stats = {
        "events": events,
        "generated_tokens": 256,
        "accepted_by_depth": [2, 1, 1],
        "drafted_by_depth": [3, 2, 1],
        "accepted_drafts": 4,
        "rejected_drafts": 1,
        "drafted_tokens": 6,
        "skipped_drafts": 0,
        "bonus_tokens": 1,
        "correction_tokens": 1,
        "verify_calls": 3,
        "mtp_forward_calls": 6,
        "make_mtp_cache_calls": 3,
        "update_mtp_cache_calls": 3,
        "mtp_history_append_calls": 3,
        "forward_ar_hidden_calls": 4,
        "forward_ar_plain_calls": 0,
    }
    return {
        "label": "ADAPTIVE-B",
        "error": None,
        "generated_tokens": 256,
        "finish_reason": "length",
        "tokens": tokens,
        "token_sha256": _token_sha(tokens),
        "decode_tokens_per_second": tps,
        "stats_full": stats,
    }


def _policy_receipt():
    return {
        "kind": "deepseek_v4_preregistered_max_k3",
        "immutable": True,
        "d1_margin_threshold": 0.25,
        "d2_margin_threshold": 10.0,
        "max_speculative_depth": 3,
        "target_routes": {"K1": "M2", "K2": "M3", "K3": "M4"},
        "target_rows": [2, 3, 4],
    }


def _moe_tail_report():
    return {
        "route": "decode_verify_m4",
        "body_layers_installed": 43,
        "mtp_layers_stock": 1,
        "verify_rows": 4,
        "repair_rows": 1,
        "topk": 6,
        "hidden_size": 4096,
        "kernel_selfcheck_exact": True,
    }


def _o_lora_report():
    return {
        "mode": "gather_qmm",
        "module_count": 44,
        "trunk_module_count": 43,
        "mtp_module_count": 1,
        "body_direct": 43,
        "mtp_stock": 1,
        "body_all_mode_matches": True,
        "route_plan_matches": True,
        "callable_census": {
            "body_route_objects": 43,
            "body_route_kind": "gather_qmm_m4_wide_direct",
            "body_callable_class": "_DirectGatherOLoraWideM4",
            "mtp_route_objects": 1,
            "mtp_route_kind": "dense_bf16_stock_direct",
            "mtp_callable_class": "_DirectDenseMTPOLora",
            "total_route_objects": 44,
            "unique_route_objects": 44,
            "mtp_distinct_type": True,
        },
    }


def _common(bench):
    return {
        "source_commit": "a" * 40,
        # The artifact identity—not the developer's checkout path—is canonical.
        "model_path": "/models/DeepSeek-V4-Flash-2bit-DQ-mtp",
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43,
        "num_nextn_predict_layers": 1,
        "prompt": {"sha256": bench._CANONICAL_PROMPT_SHA256, "tokens": 328},
        "prompt_tokens": 328,
        "max_tokens": 256,
        "depths": [3],
        "verify_strategy": "capture_commit",
        "verify_core": "stock",
        "mtp_history_policy": "committed",
        "sampling": {"greedy": True, "temperature": 0.0, "stop_token_ids": []},
        "fp32_activations": False,
        "mlx_identity": dict(bench._OFFICIAL_MLX_IDENTITY),
        "artifact_identity": dict(bench._ADAPTIVE_WIDTH_ARTIFACT_IDENTITY),
        "loaded_runtime_identity": dict(bench._ADAPTIVE_WIDTH_LOADED_IDENTITY),
        "launch_mtplx_env": dict(bench._ADAPTIVE_WIDTH_STAGE4_ENV),
        "deepseek_v4_moe_tail": _moe_tail_report(),
        "deepseek_v4_o_lora": _o_lora_report(),
        "guard_window": {"verified": True},
    }


def _receipt(bench, *, arms=None, common=None):
    return bench._adaptive_width_bracket_receipt(
        common=_common(bench) if common is None else common,
        arms=arms
        or [
            _control("K3-PRIMER", 39.0),
            _control("K3-C0", 40.0),
            _candidate(42.0),
            _control("K3-C1", 40.5),
        ],
        process_pid=7,
        model_object_id=9,
        policy_receipt=_policy_receipt(),
    )


def test_bracket_constants_pin_the_canonical_full_workload():
    bench = _module()
    assert bench._ADAPTIVE_WIDTH_BRACKET_ARMS == (
        ("K3-PRIMER", False),
        ("K3-C0", False),
        ("ADAPTIVE-B", True),
        ("K3-C1", False),
    )
    assert bench._ADAPTIVE_WIDTH_STAGE4_ENV == {
        "MTPLX_COMPILED_VERIFY": "off",
        "MTPLX_DSV4_ATTN": "fused",
        "MTPLX_DSV4_FP32_ACTIVATIONS": "0",
        "MTPLX_DSV4_HC_COMPILE": "1",
        "MTPLX_DSV4_MOE_TAIL": "1",
        "MTPLX_DSV4_O_LORA": "gather_qmm",
        "MTPLX_DSV4_SINKHORN_KERNEL": "1",
    }


def test_valid_bracket_derives_width_histogram_quality_and_promotion():
    bench = _module()
    receipt = _receipt(bench)

    assert receipt["status"] == 0
    assert receipt["performance_eligible"] is True
    assert receipt["single_process_bracket"]["model_load_count"] == 1
    assert receipt["single_process_bracket"]["execution_order"] == [
        "K3-PRIMER", "K3-C0", "ADAPTIVE-B", "K3-C1"
    ]
    assert receipt["policy_engagement"]["event_derived_width_histogram"] == {
        "K1_M2": 1,
        "K2_M3": 1,
        "K3_M4": 1,
    }
    assert receipt["policy_engagement"]["policy_thresholds"] == {
        "d1_margin_threshold": 0.25,
        "d2_margin_threshold": 10.0,
    }
    assert receipt["token_quality"]["accepted"] is True
    assert receipt["token_quality"]["mode"] == "exact"
    assert receipt["performance"]["candidate_tps"] == 42.0
    assert receipt["performance"]["reported_below_40_tps"] is False
    assert receipt["performance"]["promotion_pass"] is True


def test_below_40_is_reported_without_becoming_a_receipt_failure():
    bench = _module()
    arms = [
        _control("K3-PRIMER", 38.0),
        _control("K3-C0", 37.0),
        _candidate(39.0),
        _control("K3-C1", 37.1),
    ]
    receipt = _receipt(bench, arms=arms)
    assert receipt["status"] == 0
    assert receipt["performance"]["reported_below_40_tps"] is True


def test_only_the_approved_bf16_first_cause_can_justify_a_propagated_tail():
    bench = _module()
    tokens = list(range(256))
    tokens[221] = 12258
    tokens[222:] = [90000 + index for index in range(34)]
    arms = [
        _control("K3-PRIMER"),
        _control("K3-C0"),
        _candidate(tokens=tokens),
        _control("K3-C1"),
    ]
    receipt = _receipt(bench, arms=arms)
    quality = receipt["token_quality"]
    assert receipt["status"] == 0
    assert quality["mode"] == "approved_bf16_top2_cause"
    assert quality["approved_cause"] == {
        "continuation_index": 221,
        "absolute_position": 549,
        "control_token_id": 14042,
        "candidate_token_id": 12258,
        "control_target_gap": 0.25,
        "candidate_target_gap": 0.0,
    }
    assert quality["propagated_tail"]["documented"] is True


@pytest.mark.parametrize(
    "mutation",
    ("env", "mlx", "artifact", "runtime", "order", "policy", "policy_d2", "events", "counter", "control", "quality"),
)
def test_bracket_fails_closed_on_identity_engagement_counter_control_or_quality(mutation):
    bench = _module()
    common = _common(bench)
    arms = [
        _control("K3-PRIMER"),
        _control("K3-C0"),
        _candidate(),
        _control("K3-C1"),
    ]
    policy = _policy_receipt()
    if mutation == "env":
        common["launch_mtplx_env"]["MTPLX_CONTEXT_COPY"] = "0"
    elif mutation == "mlx":
        common["mlx_identity"]["core_sha256"] = "0" * 64
    elif mutation == "artifact":
        common["artifact_identity"]["config_sha256"] = "0" * 64
    elif mutation == "runtime":
        common["loaded_runtime_identity"]["mtp_blocks_bound"] = 0
    elif mutation == "order":
        arms[1], arms[2] = arms[2], arms[1]
    elif mutation == "policy":
        policy["d1_margin_threshold"] = 0.5
    elif mutation == "policy_d2":
        policy["d2_margin_threshold"] = 1.0
    elif mutation == "events":
        arms[2]["stats_full"]["events"][0].pop("adaptive_width_policy")
    elif mutation == "counter":
        arms[2]["stats_full"]["drafted_tokens"] = True
    elif mutation == "control":
        arms[1]["stats_full"]["events"][0]["adaptive_width_policy"] = {}
    else:
        arms[2]["tokens"][17] = 99999
    receipt = bench._adaptive_width_bracket_receipt(
        common=common,
        arms=arms,
        process_pid=7,
        model_object_id=9,
        policy_receipt=policy,
    )
    assert receipt["status"] != 0
    assert receipt["validation_errors"]


@pytest.mark.parametrize("route", ("moe_tail", "o_lora"))
def test_bracket_fails_closed_on_installed_route_receipts(route):
    bench = _module()
    common = _common(bench)
    if route == "moe_tail":
        common["deepseek_v4_moe_tail"]["body_layers_installed"] = 42
    else:
        common["deepseek_v4_o_lora"]["callable_census"]["mtp_route_kind"] = "stock"

    receipt = _receipt(bench, common=common)

    assert receipt["status"] != 0
    assert any(
        route.replace("_", "-") in error.lower()
        for error in receipt["validation_errors"]
    )


def test_guard_child_has_exact_selector_cleanup_and_canonical_command():
    child = (ROOT / "scripts" / "deepseek_v4_adaptive_width_arms.sh").read_text()
    assert "for entry in ${(f)\"$(env)\"}" in child
    assert "unset MTPLX_CONTEXT_COPY MTPLX_CONTEXT_COPY_TARGET_PREFIX" in child
    assert "--adaptive-width-bracket" in child
    assert "--max-tokens 256 --depths 3" in child
    assert "--verify-strategy capture_commit --verify-core stock" in child
    assert "--mtp-history-policy committed" in child


def test_published_adaptive_bracket_has_no_developer_absolute_paths():
    # Hygiene gate for every published DSV4 bench lane, not just adaptive
    # width: the one lane this test originally skipped (attention island) is
    # exactly the one that shipped a developer home directory (PR #223
    # review).
    paths = (
        ROOT / "scripts" / "deepseek_v4_adaptive_width_arms.sh",
        ROOT / "scripts" / "deepseek_v4_adaptive_width_guarded.py",
        ROOT / "scripts" / "deepseek_v4_attention_island_arms.sh",
        ROOT / "scripts" / "deepseek_v4_attention_island_guarded.py",
        ROOT / "scripts" / "deepseek_v4_attn_proj_wide_m3_arms.sh",
        ROOT / "scripts" / "deepseek_v4_attn_proj_wide_m3_guarded.py",
        BENCH_PATH,
    )
    developer_home = f"/{'Users'}/{'davidtai'}"
    offenders = [str(path) for path in paths if developer_home in path.read_text()]
    assert not offenders, offenders


def _wrapper_module():
    path = ROOT / "scripts" / "deepseek_v4_adaptive_width_guarded.py"
    spec = importlib.util.spec_from_file_location("dsv4_adaptive_width_wrapper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _postflight(ok=True):
    return {
        key: {"ok": ok}
        for key in (
            "lock_free",
            "wired_limit_mb",
            "quality_models",
            "quality_ready_chat",
        )
    }


def test_postflight_wrapper_requires_child_receipt_and_all_restoration_probes(tmp_path):
    wrapper = _wrapper_module()
    tag = "adaptive-width-test"
    primary = {"status": 0, "receipt_role": "adaptive_width_performance_bracket"}
    (tmp_path / f"{tag}.json").write_text(json.dumps(primary))
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["env"] = kwargs["env"]
        return SimpleNamespace(returncode=0)

    status = wrapper.run(
        tag,
        run_command=fake_run,
        postflight_collector=lambda: _postflight(True),
        bench_dir=tmp_path,
    )

    assert status == 0
    assert seen["env"]["MTPLX_DSV4_ADAPTIVE_WIDTH_POSTFLIGHT_WRAPPER"] == "1"
    receipt = json.loads((tmp_path / f"{tag}-postflight.json").read_text())
    assert receipt["postflight_ok"] is True
    assert receipt["primary_receipt"]["status"] == 0
    assert receipt["status"] == 0


@pytest.mark.parametrize("failure", ("child", "missing", "malformed", "probe"))
def test_postflight_wrapper_always_writes_and_fails_closed(tmp_path, failure):
    wrapper = _wrapper_module()
    tag = f"adaptive-width-{failure}"
    if failure != "missing":
        primary = (
            {"status": 1, "receipt_role": "adaptive_width_performance_bracket"}
            if failure == "malformed"
            else {"status": 0, "receipt_role": "adaptive_width_performance_bracket"}
        )
        (tmp_path / f"{tag}.json").write_text(json.dumps(primary))
    probes = _postflight(failure != "probe")
    status = wrapper.run(
        tag,
        run_command=lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1 if failure == "child" else 0
        ),
        postflight_collector=lambda: probes,
        bench_dir=tmp_path,
    )
    assert status == 1
    receipt = json.loads((tmp_path / f"{tag}-postflight.json").read_text())
    assert receipt["status"] == 1


@pytest.mark.parametrize(
    "tag",
    ("../../escaped", "", ".", "..", "slash/name", "bad tag"),
)
def test_invalid_tag_skips_child_but_collects_postflight_and_writes_safe_receipt(
    tmp_path, tag
):
    wrapper = _wrapper_module()
    collected = []

    def collect():
        collected.append(True)
        return _postflight(True)

    status = wrapper.run(
        tag,
        run_command=lambda *_a, **_k: pytest.fail("invalid tag started child"),
        postflight_collector=collect,
        bench_dir=tmp_path,
    )

    digest = hashlib.sha256(tag.encode("utf-8", errors="surrogatepass")).hexdigest()
    receipts = list(
        tmp_path.glob(
            f"adaptive-width-invalid-tag-{digest}-pid-*-postflight.json"
        )
    )
    assert status == 1
    assert collected == [True]
    assert len(receipts) == 1
    assert receipts[0].parent.resolve() == tmp_path.resolve()
    assert list(tmp_path.iterdir()) == receipts
    pid_text = receipts[0].name.removeprefix(
        f"adaptive-width-invalid-tag-{digest}-pid-"
    ).removesuffix("-postflight.json")
    assert pid_text.isdecimal()
    receipt = json.loads(receipts[0].read_text())
    assert receipt["tag_valid"] is False
    assert receipt["tag_sha256"] == digest
    assert receipt["guarded_child_started"] is False
    assert receipt["guarded_child_exit_code"] is None
    assert receipt["postflight_ok"] is True
    assert set(receipt["postflight"]) == set(wrapper.REQUIRED_PROBES)
    assert receipt["status"] == 1


def test_invalid_tag_path_strictly_rejects_malformed_restoration_probe(tmp_path):
    wrapper = _wrapper_module()
    probes = _postflight(True)
    probes["wired_limit_mb"] = {"ok": 1}

    status = wrapper.run(
        "invalid/tag",
        run_command=lambda *_a, **_k: pytest.fail("invalid tag started child"),
        postflight_collector=lambda: probes,
        bench_dir=tmp_path,
    )

    receipt_path = next(tmp_path.glob("adaptive-width-invalid-tag-*-postflight.json"))
    receipt = json.loads(receipt_path.read_text())
    assert status == 1
    assert receipt["postflight_ok"] is False
    assert any(
        "wired_limit_mb is missing or malformed" in error
        for error in receipt["validation_errors"]
    )
