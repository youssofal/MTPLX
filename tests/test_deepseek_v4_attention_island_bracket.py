"""Fail-closed gates for the canonical attention-island GPU bracket."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from mtplx import deepseek_v4_attention_island as AI


ROOT = Path(__file__).parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _bench():
    return _load(
        ROOT / "scripts" / "deepseek_v4_mtpk_bench.py",
        "dsv4_attention_island_bench",
    )


def test_contract_pins_candidate_primer_current_controls_and_authoritative_mix():
    bench = _bench()
    assert bench._ATTENTION_ISLAND_BRACKET_ARMS == (
        ("ATTENTION-ISLAND-PRIMER", True),
        ("CURRENT-C0", False),
        ("ATTENTION-ISLAND-B", True),
        ("CURRENT-C1", False),
    )
    assert bench._ATTENTION_ISLAND_CONTROL_HISTOGRAM == {
        "K1_M2": 6,
        "K2_M3": 76,
        "K3_M4": 10,
    }
    assert bench._ATTENTION_ISLAND_STAGE4_ENV == {
        **bench._ATTN_PROJ_WIDE_M3_STAGE4_ENV,
        "MTPLX_DSV4_ATTENTION_ISLAND": "1",
    }


def test_tape_warmth_requires_all_nine_width_layout_signatures():
    bench = _bench()
    engagement = {
        "event_derived_width_histogram": {
            "K1_M2": 6,
            "K2_M3": 76,
            "K3_M4": 10,
        }
    }
    signatures = bench._attention_island_signatures(engagement)
    assert len(signatures) == 9
    assert signatures == sorted(
        f"M{width}:{layout}"
        for width in (2, 3, 4)
        for layout in bench._ATTENTION_ISLAND_LAYOUTS
    )
    engagement["event_derived_width_histogram"]["K1_M2"] = 0
    assert len(bench._attention_island_signatures(engagement)) == 6


def _paired_evidence():
    return {
        "path": "bench/deepseek-v4/hc-olora-51b0f105-20260802T161346Z-quality.json",
        "sha256": "e8a3c1ed71aa9ac7024a457865c180c3aadafbf654f56066c750cf63e4a4bed2",
        "quality_verdict": "ACCEPTED_SINGLE_IDENTICAL_BF16_TOP2_FLIP",
        "continuation_index": 221,
        "absolute_target_position": 549,
        "control_token_id": 14042,
        "candidate_token_id": 12258,
        "control_gap": 0.25,
        "candidate_gap": 0.0,
    }


def _paired_payload():
    gaps = {
        "cached_gap_to_gather_selected": 0.25,
        "gather_gap_to_cached_selected": 0.0,
    }
    return {
        "status": "COMPLETE",
        "quality_gate_pass": True,
        "quality_verdict": "ACCEPTED_SINGLE_IDENTICAL_BF16_TOP2_FLIP",
        "errors": [],
        "strict_validation_errors": [],
        "quality_acceptance": {
            "policy": "exact_or_single_identical_bf16_top2_flip",
            "accepted_mode": "single_identical_bf16_top2_flip",
            "single_flip": {
                "continuation_index": 221,
                "absolute_target_position": 549,
                "cached_selected_id": 14042,
                "gather_selected_id": 12258,
                "AR": gaps,
                "K3_TARGET_ROWS": gaps,
            },
        },
        "execution_contract": {
            "teacher_forced": True,
            "production_hot_path_instrumentation": False,
            "model_objects": 1,
            "model_load_count": 1,
            "memory_safe_sequential_evaluation": True,
            "ar_rows": 256,
            "k3_target_rows": 256,
            "k3_physical_m": 4,
        },
    }


def test_paired_near_tie_loader_authenticates_digest_and_exact_contract(tmp_path):
    bench = _bench()
    encoded = json.dumps(_paired_payload(), sort_keys=True).encode()
    path = tmp_path / bench._ATTENTION_ISLAND_PAIRED_QUALITY_FILENAME
    path.write_bytes(encoded)
    evidence = bench._load_attention_island_paired_near_tie_evidence(
        tmp_path,
        expected_sha256=hashlib.sha256(encoded).hexdigest(),
    )
    assert evidence == _paired_evidence()

    payload = _paired_payload()
    payload["quality_acceptance"]["single_flip"]["K3_TARGET_ROWS"] = {
        "cached_gap_to_gather_selected": 1.0,
        "gather_gap_to_cached_selected": 0.0,
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    path.write_bytes(encoded)
    try:
        bench._load_attention_island_paired_near_tie_evidence(
            tmp_path,
            expected_sha256=hashlib.sha256(encoded).hexdigest(),
        )
    except ValueError as error:
        assert "k3_gap" in str(error)
    else:
        raise AssertionError("altered paired K3 gap was accepted")


def test_arbitrary_divergence_is_not_mislabeled_as_near_tie():
    bench = _bench()
    control = list(range(256))
    candidate = list(control)
    candidate[17] = 99999
    quality = bench._attention_island_token_quality(
        control, candidate, _paired_evidence()
    )
    assert quality["accepted"] is False
    assert quality["mode"] == "unapproved_divergence"
    assert quality["first_divergence"]["continuation_index"] == 17


def test_source_bound_paired_near_tie_is_allowed_with_propagated_tail():
    bench = _bench()
    control = list(range(256))
    candidate = list(control)
    control[221] = 14042
    candidate[221] = 12258
    candidate[222:] = [99999] * 34
    quality = bench._attention_island_token_quality(
        control, candidate, _paired_evidence()
    )
    assert quality["accepted"] is True
    assert quality["mode"] == "source_bound_paired_bf16_near_tie"
    assert quality["paired_evidence"] == _paired_evidence()
    assert quality["propagated_tail"]["divergent_tokens"] == 34


def _run_fake_bracket(bench, monkeypatch, tmp_path, *, compile_in_b=False):
    import mtplx.deepseek_v4_adaptive_width as adaptive

    model = SimpleNamespace(_target_hc_hidden_route=None)
    selector = AI._AttentionIslandArmSelector(model, object(), object())
    model._mtplx_dsv4_attention_island_selector = selector
    report = {
        "installed": True,
        "widths": [2, 3, 4],
        "body_layers": 43,
        "bound_layer_routes": 129,
        "shared_tapes": 9,
        "expected_shared_tapes": 9,
        "attention": "eager_exact_logical_cache",
        "weight_binding": "explicit_array_inputs",
        "runtime_fallback": False,
        "hot_environment_reads": False,
        "hot_counters": False,
    }
    rt = SimpleNamespace(
        model=model,
        deepseek_v4_attention_island_report=report,
    )
    fake_policy = object()
    monkeypatch.setattr(
        adaptive,
        "install_deepseek_v4_adaptive_width_policy",
        lambda *_args, **_kwargs: fake_policy,
    )
    monkeypatch.setattr(bench, "_installed_policy_receipt", lambda _policy: {})
    monkeypatch.setattr(bench, "_adaptive_width_common_errors", lambda _common: [])
    monkeypatch.setattr(bench, "_validate_behavior_stats", lambda *_args: [])
    monkeypatch.setattr(bench, "_reset_benchmark_state", lambda _rt: None)
    monkeypatch.setattr(AI, "_TAPES", {f"tape-{index}": object() for index in range(9)})

    def run_arm(**kwargs):
        label = kwargs["label"]
        if compile_in_b and label == "ATTENTION-ISLAND-B":
            AI._TAPES["unexpected-tape"] = object()
        tokens = list(range(256))
        return {
            "label": label,
            "error": None,
            "generated_tokens": 256,
            "finish_reason": "length",
            "tokens": tokens,
            "decode_tokens_per_second": {
                "ATTENTION-ISLAND-PRIMER": 39.0,
                "CURRENT-C0": 41.0,
                "ATTENTION-ISLAND-B": 42.0,
                "CURRENT-C1": 41.1,
            }[label],
            "text": "READY",
            "stats_full": {},
        }

    monkeypatch.setattr(bench, "_run_arm", run_arm)
    histogram = {"K1_M2": 6, "K2_M3": 76, "K3_M4": 10}
    monkeypatch.setattr(
        bench,
        "_adaptive_width_engagement",
        lambda _arm: ({"event_derived_width_histogram": dict(histogram)}, []),
    )
    args = SimpleNamespace(
        verify_strategy="capture_commit",
        verify_core="stock",
        mtp_history_policy="committed",
        max_tokens=256,
        prompt_file="prompt.txt",
    )
    common = {
        "launch_mtplx_env": dict(bench._ATTENTION_ISLAND_STAGE4_ENV),
        "paired_near_tie_evidence": _paired_evidence(),
    }
    out = tmp_path / "attention-island"
    status = bench._run_attention_island_bracket(
        rt=rt,
        prompt_ids=list(range(328)),
        args=args,
        common_receipt=common,
        out_stem=out,
    )
    return status, json.loads(out.with_suffix(".json").read_text())


def test_fake_full_bracket_accepts_warmed_nine_tapes(monkeypatch, tmp_path):
    bench = _bench()
    status, receipt = _run_fake_bracket(bench, monkeypatch, tmp_path)
    assert status == 0
    assert receipt["compiled_tape_warmth"]["unprimed"] == []
    assert receipt["compiled_tape_warmth"]["python_tapes_before_b"] == 9
    assert receipt["compiled_tape_warmth"]["python_tapes_after_b"] == 9
    assert receipt["performance"]["promotion_pass"] is True


def test_fake_full_bracket_fails_if_b_enters_new_tape_class(monkeypatch, tmp_path):
    bench = _bench()
    status, receipt = _run_fake_bracket(
        bench, monkeypatch, tmp_path, compile_in_b=True
    )
    assert status == 1
    assert any("new Python tape compilation class" in error for error in receipt["validation_errors"])


def test_arms_script_requires_clean_commit_and_exact_workload():
    source = (ROOT / "scripts" / "deepseek_v4_attention_island_arms.sh").read_text()
    wrapper = _load(
        ROOT / "scripts" / "deepseek_v4_attention_island_guarded.py",
        "dsv4_attention_island_guard_source",
    )
    assert "TO_BE_FILLED" not in source
    assert 'status --porcelain' in source
    assert 'CHILD_OBSERVED_SOURCE_COMMIT=$(git -C "$WORKTREE" rev-parse HEAD)' in source
    assert "--attention-island-bracket --expected-source-commit" in source
    assert "--max-tokens 256 --depths 3" in source
    assert "--verify-strategy capture_commit --verify-core stock" in source
    assert "--mtp-history-policy committed" in source
    assert "MTPLX_DSV4_ATTENTION_ISLAND=1" in source
    assert "MTPLX_DSV4_ATTN_PROJ_WIDE_M3=1" in source
    # Deployment locations come from the operator (portable pair contract).
    assert "MTPLX_DSV4_BENCH_DIR:?" in source
    assert "MTPLX_DSV4_MODEL_PATH:?" in source
    assert "MTPLX_DSV4_PROMPT_FILE:?" in source
    assert "iogpu.wired_limit_mb=114688" not in source
    assert "EXPECTED_WIRED_LIMIT_MB=${MTPLX_DSV4_EXPECTED_WIRED_LIMIT_MB:-" in source
    # Source identity rides the exact-commit + clean-worktree gates; a frozen
    # per-file SHA manifest broke on every legitimate commit (review edit).
    assert "SOURCE_MANIFEST" not in source
    assert '/bin/chmod 600 "$CHILD_STATUS_TMP"' in source
    assert '/bin/chmod 600 -- "$CHILD_STATUS_TMP"' not in source
    assert '/bin/mv -f -- "$CHILD_STATUS_TMP" "$CHILD_STATUS"' in source
    commit = "a" * 40
    assert wrapper._command("attention-island-test", commit, commit)[-3:] == [
        "attention-island-test",
        commit,
        commit,
    ]


def _guarded():
    return _load(
        ROOT / "scripts" / "deepseek_v4_attention_island_guarded.py",
        "dsv4_attention_island_guard",
    )


def _guard_probes(*, ok=True):
    wrapper = _guarded()
    return {
        "lock_free": {
            "ok": ok,
            "requested_path": "/tmp/mtplx-gpu-exclusive.lock",
            "acquired_nonblocking": ok,
            "held_through_probes": ok,
            "released_after_probes": True,
        },
        "wired_limit_mb": {"ok": ok, "value": 114688},
        "quality_models": {
            "ok": ok,
            "models": ["mtplx-qwen36-27b-optimized-quality"],
        },
        "quality_ready_chat": {
            "ok": ok,
            "content": "READY",
            "finish_reason": "stop",
        },
        "quality_plist": {
            "ok": ok,
            "path": str(wrapper.QUALITY_PLIST),
            "sha256": wrapper.QUALITY_PLIST_SHA256,
            "size": wrapper.QUALITY_PLIST_SIZE,
            "plutil_valid": ok,
        },
    }


def _write_primary_and_status(tmp_path, wrapper, tag, commit, *, exit_code=0):
    attestation = {
        "expected": commit,
        "observed": commit,
        "match": True,
        "clean": True,
    }
    (tmp_path / f"{tag}.json").write_text(
        json.dumps(
            {
                "status": int(bool(exit_code)),
                "receipt_role": "attention_island_performance_bracket",
                "source_commit_attestation": attestation,
            }
        )
    )
    sidecar = tmp_path / f"{tag}-child-status.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "attention_island_child_status",
                "tag": tag,
                "expected_source_commit": commit,
                "observed_source_commit": commit,
                "benchmark_exit_code": exit_code,
            }
        )
    )
    sidecar.chmod(0o600)


def test_guarded_wrapper_requires_primary_success_and_restoration(tmp_path):
    wrapper = _guarded()
    tag = "attention-island-test"
    commit = "f" * 40

    def run_command(command, **_kwargs):
        assert command == wrapper._command(tag, commit, commit)
        _write_primary_and_status(tmp_path, wrapper, tag, commit)
        return SimpleNamespace(returncode=0)

    assert wrapper.run(
        tag,
        expected_commit=commit,
        source_commit_reader=lambda: commit,
        source_clean_reader=lambda: True,
        bench_dir=tmp_path,
        run_command=run_command,
        preflight_collector=lambda: _guard_probes(),
        postflight_collector=lambda: _guard_probes(),
    ) == 0
    persisted = json.loads((tmp_path / f"{tag}-postflight.json").read_text())
    assert persisted["kind"] == "deepseek_v4_attention_island_guarded_postflight"
    assert persisted["source_commit_attestation"] == {
        "expected": commit,
        "observed": commit,
        "match": True,
        "clean": True,
    }
    assert persisted["benchmark_child_exit_code"] == 0
    assert persisted["guarded_runner_exit_code"] == 0
    assert persisted["quality_plist_unchanged"] is True


def test_guard_refuses_wrong_sha_before_stopping_service(tmp_path):
    wrapper = _guarded()
    started = []
    status = wrapper.run(
        "attention-island-wrong-sha",
        expected_commit="a" * 40,
        source_commit_reader=lambda: "b" * 40,
        source_clean_reader=lambda: True,
        bench_dir=tmp_path,
        run_command=lambda *_args, **_kwargs: started.append(True),
        preflight_collector=lambda: _guard_probes(),
        postflight_collector=lambda: _guard_probes(),
    )
    assert status == 1
    assert started == []
    receipt = json.loads(
        (tmp_path / "attention-island-wrong-sha-postflight.json").read_text()
    )
    assert receipt["guarded_child_started"] is False
    assert receipt["source_commit_attestation"]["match"] is False


def test_guard_refuses_dirty_same_head_before_stopping_service(tmp_path):
    wrapper = _guarded()
    commit = "d" * 40
    started = []
    status = wrapper.run(
        "attention-island-dirty",
        expected_commit=commit,
        source_commit_reader=lambda: commit,
        source_clean_reader=lambda: False,
        bench_dir=tmp_path,
        run_command=lambda *_args, **_kwargs: started.append(True),
        preflight_collector=lambda: _guard_probes(),
        postflight_collector=lambda: _guard_probes(),
    )
    assert status == 1
    assert started == []
    receipt = json.loads(
        (tmp_path / "attention-island-dirty-postflight.json").read_text()
    )
    assert receipt["source_commit_attestation"]["clean"] is False


def test_guard_preserves_benchmark_exit_when_restoration_also_fails(tmp_path):
    wrapper = _guarded()
    commit = "c" * 40
    tag = "attention-island-child7-restore1"

    def run_command(_command, **_kwargs):
        _write_primary_and_status(tmp_path, wrapper, tag, commit, exit_code=7)
        return SimpleNamespace(returncode=1)

    assert wrapper.run(
        tag,
        expected_commit=commit,
        source_commit_reader=lambda: commit,
        source_clean_reader=lambda: True,
        bench_dir=tmp_path,
        run_command=run_command,
        preflight_collector=lambda: _guard_probes(),
        postflight_collector=lambda: _guard_probes(ok=False),
    ) == 1
    receipt = json.loads((tmp_path / f"{tag}-postflight.json").read_text())
    assert receipt["benchmark_child_exit_code"] == 7
    assert receipt["guarded_runner_exit_code"] == 1
    assert receipt["primary_receipt"]["status"] == 1
    assert any("benchmark child exited 7" in error for error in receipt["validation_errors"])
    assert any(
        "guarded lifecycle returned 1 after benchmark returned 7" in error
        for error in receipt["validation_errors"]
    )


def test_guard_deletes_stale_child_status_before_launch(tmp_path):
    wrapper = _guarded()
    commit = "e" * 40
    tag = "attention-island-stale"
    stale = tmp_path / f"{tag}-child-status.json"
    stale.write_text("{}")
    stale.chmod(0o600)
    assert wrapper.run(
        tag,
        expected_commit=commit,
        source_commit_reader=lambda: commit,
        source_clean_reader=lambda: True,
        bench_dir=tmp_path,
        run_command=lambda *_args, **_kwargs: SimpleNamespace(returncode=1),
        preflight_collector=lambda: _guard_probes(),
        postflight_collector=lambda: _guard_probes(),
    ) == 1
    receipt = json.loads((tmp_path / f"{tag}-postflight.json").read_text())
    assert receipt["benchmark_child_exit_code"] is None
    assert receipt["benchmark_child_status"] is None
    assert "FileNotFoundError" in receipt["benchmark_child_status_error"]


def test_macos_child_status_chmod_form_sets_mode_0600(tmp_path):
    sidecar = tmp_path / "child-status.json"
    sidecar.write_text("{}")
    completed = subprocess.run(
        ["/bin/chmod", "600", str(sidecar)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert sidecar.stat().st_mode & 0o777 == 0o600
