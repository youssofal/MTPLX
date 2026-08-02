"""Integrity gates for the DeepSeek-V4 MoE-tail K3 E2E bracket."""

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


_ROOT = Path(__file__).parents[1]
_VALIDATOR = _ROOT / "scripts" / "deepseek_v4_validate_moe_tail_k3_bracket.py"
_GUARD = _ROOT / "scripts" / "deepseek_v4_guard_window.py"
_BENCHMARK = _ROOT / "scripts" / "deepseek_v4_mtpk_bench.py"
_ARMS = _ROOT / "scripts" / "deepseek_v4_moe_tail_arms.sh"
_POSTFLIGHT = _ROOT / "scripts" / "deepseek_v4_moe_tail_guarded_bracket.py"

_spec = importlib.util.spec_from_file_location("dsv4_moe_tail_bracket", _VALIDATOR)
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))
_bench_spec = importlib.util.spec_from_file_location("dsv4_moe_tail_bench", _BENCHMARK)
H = importlib.util.module_from_spec(_bench_spec)
_bench_spec.loader.exec_module(H)

_postflight_spec = importlib.util.spec_from_file_location(
    "dsv4_moe_tail_postflight", _POSTFLIGHT
)
P = importlib.util.module_from_spec(_postflight_spec)
_postflight_spec.loader.exec_module(P)

_guard_spec = importlib.util.spec_from_file_location("dsv4_guard_window", _GUARD)
G = importlib.util.module_from_spec(_guard_spec)
_guard_spec.loader.exec_module(G)

if not G.LAGUNA_BENCH.is_file():
    pytest.skip(
        "machine-local DeepSeek-V4 bench-harness gates: repository guard "
        f"verifier not present at {G.LAGUNA_BENCH} "
        "(set MTPLX_DSV4_GUARD_VERIFIER to point at it)",
        allow_module_level=True,
    )


def _stage4_env(_enabled: bool) -> dict[str, str]:
    return {
        "MTPLX_COMPILED_VERIFY": "off",
        "MTPLX_DSV4_ATTN": "fused",
        "MTPLX_DSV4_FP32_ACTIVATIONS": "0",
        "MTPLX_DSV4_HC_COMPILE": "1",
        # One process constructs and self-checks the candidate once; controls
        # are proven stock by their bound-callable receipts, not an env reload.
        "MTPLX_DSV4_MOE_TAIL": "1",
        "MTPLX_DSV4_O_LORA": "cached",
        "MTPLX_DSV4_SINKHORN_KERNEL": "1",
    }


def _guard_window(child_pid: int = 200) -> dict:
    requested_lock = "/tmp/mtplx-gpu-exclusive.lock"
    resolved_lock = str(Path(requested_lock).resolve())
    attestation = {
        "schema_version": 1,
        "guard_pid": 100,
        "child_pid": child_pid,
        "issued_monotonic_ns": 1_000_000,
        "expires_monotonic_ns": 61_000_000,
        "lock_path": resolved_lock,
        "lock_device": 1,
        "lock_inode": 2,
        "nonce_sha256": "c" * 64,
    }
    encoded_attestation = json.dumps(
        attestation, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    document = {
        "schema_version": 1,
        "kind": "mtplx_verified_guard_window",
        "verified": True,
        "verified_monotonic_ns": 2_000_000,
        "window_id": hashlib.sha256(encoded_attestation).hexdigest(),
        "attestation": attestation,
        "lock_identity": {
            "requested_path": requested_lock,
            "resolved_path": resolved_lock,
            "device": 1,
            "inode": 2,
        },
    }
    encoded_document = json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {
        **document,
        "receipt_path": "/tmp/mtplx-dsv4-guard-window-test/window.json",
        "receipt_sha256": hashlib.sha256(encoded_document).hexdigest(),
        "consumer_verification": {
            "consumer_pid": 300,
            "ancestry": [300, child_pid, 100],
            "child_pid_index": 1,
            "guard_pid_index": 2,
            "lock_held": True,
            "observed_lock_device": 1,
            "observed_lock_inode": 2,
        },
    }


def _install_report() -> dict:
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


def _receipt(
    tps: float,
    *,
    candidate: bool = False,
    role: str = "measurement",
    tokens: list[int] | None = None,
    guard_window: dict | None = None,
    ar_tps: float = 29.0,
    bracket_index: int = 1,
) -> dict:
    tokens = list(range(256)) if tokens is None else tokens
    stats = {
        "accepted_by_depth": [60, 40, 20],
        "drafted_by_depth": [80, 60, 40],
        "accepted_drafts": 120,
        "rejected_drafts": 40,
        "drafted_tokens": 180,
        "skipped_drafts": 0,
        "bonus_tokens": 20,
        "correction_tokens": 0,
        "verify_calls": 80,
        "mtp_forward_calls": 180,
        "make_mtp_cache_calls": 80,
        "update_mtp_cache_calls": 80,
        "mtp_history_append_calls": 80,
        "forward_ar_hidden_calls": 161,
        "forward_ar_plain_calls": 0,
    }
    ar_stats = {
        key: ([] if key in {"accepted_by_depth", "drafted_by_depth"} else 0)
        for key in stats
    }
    ar_stats.update(
        {
            "generated_tokens": 256,
            "forward_ar_plain_calls": 257,
        }
    )
    resolved_index = 0 if role == "discarded_control_primer" else 2 if candidate else bracket_index
    return {
        "status": 0,
        "source_commit": "8" * 40,
        "model_path": "/Users/davidtai/models/DeepSeek-V4-Flash-2bit-DQ-mtp",
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43,
        "num_nextn_predict_layers": 1,
        "host": {"mlx_version": "0.31.2"},
        "mlx_identity": {
            "version": "0.31.2",
            "core_sha256": "d7bd29fc20b4a08318d21161c3dfb340889cc9454c5e554ad749eb0127cfa2d6",
            "lib_sha256": "2ee6fbd32ff22e22e1301ebe3c3bece95584104ff9cbc900513d41a095211bbd",
        },
        "artifact_identity": {
            "config_sha256": "c8ff87fd5ee5c9587d0c937e9bfd3193e1a1621141aa367848a9610b3291fa6f",
            "index_sha256": "c84d2b369f5d5023d0f2d183fc36a935a3981751414996243b65f069983e43d8",
            "model_type": "deepseek_v4",
            "num_hidden_layers": 43,
            "num_nextn_predict_layers": 1,
            "body_q2_routed_projections": 129,
            "body_q2_manifest_tensors": 387,
            "mtp_manifest_tensors": 35,
            "index_weight_count": 2645,
        },
        "loaded_runtime_identity": {
            "runtime_mtp_enabled": True,
            "body_layers_loaded": 43,
            "mtp_blocks_bound": 1,
            "body_q2_routed_projections": 129,
            "body_q2_weight_dtype": "uint32",
            "mtp_mxfp4_routed_projections": 3,
            "mtp_routed_weight_dtype": "uint32",
        },
        "prompt_file": (
            "/Users/davidtai/projects/OpenSourceWTF/bench/deepseek-v4/"
            "smoke-2bitdq-20260731-prompt2.txt"
        ),
        "prompt": {
            "path": (
                "/Users/davidtai/projects/OpenSourceWTF/bench/deepseek-v4/"
                "smoke-2bitdq-20260731-prompt2.txt"
            ),
            "sha256": "ee94397faa812c91d5f1a0ee17c5bb6ca6032883653591dd33d4cfddb737ac33",
            "tokens": 328,
        },
        "prompt_tokens": 328,
        "max_tokens": 256,
        "depths": [3],
        "verify_strategy": "capture_commit",
        "verify_core": "stock",
        "mtp_history_policy": "committed",
        "receipt_role": role,
        "performance_eligible": role == "measurement",
        "launch_mtplx_env": _stage4_env(candidate),
        "guard_window": _guard_window() if guard_window is None else guard_window,
        "deepseek_v4_moe_tail": _install_report() if candidate else None,
        "single_process_bracket": {
            "bracket_id": "b" * 64,
            "process_pid": 300,
            "model_object_id": 12345,
            "model_load_count": 1,
            "execution_order": ["primer", "C0", "candidate", "C1"],
            "arm_index": resolved_index,
        },
        "route_binding": {
            "ar": "stock",
            "k3": "candidate" if candidate else "stock",
            "post": "stock",
        },
        "route_census": {
            "ar": {
                "body_candidate": 0,
                "body_stock": 43,
                "body_other": 0,
                "mtp_stock": 1,
                "mtp_other": 0,
            },
            "k3": {
                "body_candidate": 43 if candidate else 0,
                "body_stock": 0 if candidate else 43,
                "body_other": 0,
                "mtp_stock": 1,
                "mtp_other": 0,
            },
            "post": {
                "body_candidate": 0,
                "body_stock": 43,
                "body_other": 0,
                "mtp_stock": 1,
                "mtp_other": 0,
            },
        },
        "fp32_activations": False,
        "require_exact": False,
        "spec_equals_ar_enforced": False,
        "arms": [
            {
                "speculative_depth": 3,
                "generated_tokens": 256,
                "tokens": tokens,
                "peak_gib": 97.0,
                "decode_tokens_per_second": tps,
                "stats": stats,
            },
            {
                "speculative_depth": None,
                "generated_tokens": 256,
                "tokens": list(range(1000, 1256)),
                "peak_gib": 97.0,
                "decode_tokens_per_second": ar_tps,
                "stats": ar_stats,
            },
        ],
    }


def test_shell_is_hermetic_and_orders_primer_c0_candidate_c1_validator():
    source = _ARMS.read_text()
    issue = source.index('deepseek_v4_guard_window.py" issue')
    bracket = source.index("--moe-tail-bracket")
    validator = source.index('if "$VENV" -u "$VALIDATOR"')
    assert issue < source.index("shasum -a 256") < bracket < validator
    assert source.count('"$VENV" -u "$WORKTREE/scripts/deepseek_v4_mtpk_bench.py"') == 1
    assert "run_arm" not in source
    assert '"$name" == MTPLX_*' in source
    assert "HF_HUB_OFFLINE=1" in source and "PYTHONNOUSERSITE=1" in source
    assert "--max-tokens 256 --depths 3" in source
    assert "--verify-strategy capture_commit --verify-core stock" in source
    assert "--mtp-history-policy committed" in source
    assert "discarded_control_primer" in source
    assert "if \"$VENV\" -u \"$VALIDATOR\"" in source
    assert "--require-live-guard" in source
    assert "receipts preserved" in source


def test_shell_captures_benchmark_failure_and_still_invokes_validator():
    source = _ARMS.read_text()
    benchmark = source.index(
        '"$VENV" -u "$WORKTREE/scripts/deepseek_v4_mtpk_bench.py"'
    )
    captured = source.index("benchmark_rc=$?", benchmark)
    validator = source.index('if "$VENV" -u "$VALIDATOR"', captured)
    assert source.rindex("set +e", 0, benchmark) < benchmark < captured < validator
    assert source.index("set -e", captured) < validator
    assert '--benchmark-exit-code "$benchmark_rc"' in source


def test_shell_help_documents_exact_guard_restore_and_real_ready_stop_check():
    source = _ARMS.read_text()
    assert "/Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py" in source
    assert "/Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist" in source
    assert "--lock-timeout-seconds 3600 --child-timeout-seconds 3600" in source
    assert "launchctl bootstrap gui/501" in source
    assert "mtplx-qwen36-27b-optimized-quality" in source
    assert "/v1/models" in source
    assert "/v1/chat/completions" in source
    assert "Say READY" in source
    assert 'finish_reason == "stop"' in source


def test_outer_guarded_wrapper_runs_postflight_after_child_failure(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(
        P.subprocess,
        "run",
        lambda command, check=False, **_kwargs: calls.append(command)
        or SimpleNamespace(returncode=17),
    )
    monkeypatch.setattr(
        P,
        "collect_postflight",
        lambda: {
            "lock_free": {"ok": True},
            "wired_limit_mb": {"ok": True, "value": 114688},
            "quality_models": {"ok": True},
            "quality_ready_chat": {"ok": True},
        },
    )
    monkeypatch.setattr(P, "_write_receipt", lambda path, receipt: calls.append(receipt))

    assert P.run("test-tag", bench_dir=tmp_path) == 17
    assert calls[0][-3:] == ["/bin/zsh", str(P.ARMS), "test-tag"]
    receipt = calls[1]
    assert receipt["child_exit_code"] == 17
    assert receipt["postflight_ok"] is True
    assert receipt["exit_code"] == 17


def test_outer_guarded_wrapper_fails_successful_child_when_postflight_fails(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        P.subprocess,
        "run",
        lambda _command, check=False, **_kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        P,
        "collect_postflight",
        lambda: {
            "lock_free": {"ok": False, "error": "still locked"},
            "wired_limit_mb": {"ok": True, "value": 114688},
            "quality_models": {"ok": True},
            "quality_ready_chat": {"ok": True},
        },
    )
    receipts = []
    monkeypatch.setattr(P, "_write_receipt", lambda _path, receipt: receipts.append(receipt))

    assert P.run("test-tag", bench_dir=tmp_path) == 1
    assert receipts[0]["child_exit_code"] == 0
    assert receipts[0]["postflight_ok"] is False
    assert receipts[0]["exit_code"] == 1


def test_outer_guarded_wrapper_source_keeps_run_guarded_as_service_owner():
    source = _POSTFLIGHT.read_text()
    assert "/Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py" in source
    assert "--lock-timeout-seconds" in source
    assert "fcntl.LOCK_NB" in source
    assert "iogpu.wired_limit_mb" in source
    assert "/v1/models" in source
    assert "/v1/chat/completions" in source
    assert "finish_reason" in source
    assert "MTPLX_DSV4_MOE_TAIL_POSTFLIGHT_WRAPPER" in source
    assert "launchctl" not in source
    assert "bootstrap" not in source


def test_postflight_holds_one_canonical_lock_across_every_probe(monkeypatch, tmp_path):
    lock_path = tmp_path / "gpu.lock"
    lock_path.touch()
    monkeypatch.setattr(P, "LOCK_PATH", lock_path)
    probe_names = []

    def guarded_probe(name, result):
        def probe():
            with lock_path.open("rb") as competitor:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(competitor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            probe_names.append(name)
            return result

        return probe

    monkeypatch.setattr(
        P,
        "_check_wired_limit",
        guarded_probe("wired", {"ok": True, "value": 114688}),
    )
    monkeypatch.setattr(
        P,
        "_check_quality_models",
        guarded_probe("models", {"ok": True, "models": [P.QUALITY_MODEL]}),
    )
    monkeypatch.setattr(
        P,
        "_check_quality_ready_chat",
        guarded_probe(
            "chat", {"ok": True, "content": "READY", "finish_reason": "stop"}
        ),
    )

    result = P.collect_postflight()

    assert probe_names == ["wired", "models", "chat"]
    assert result["lock_free"]["acquired_nonblocking"] is True
    assert result["lock_free"]["held_through_probes"] is True
    assert result["lock_free"]["released_after_probes"] is True
    identity = result["lock_free"]["identity"]
    observed = lock_path.stat()
    assert identity["device"] == observed.st_dev
    assert identity["inode"] == observed.st_ino
    with lock_path.open("rb") as after:
        fcntl.flock(after.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(after.fileno(), fcntl.LOCK_UN)


def test_postflight_contention_skips_every_service_probe(monkeypatch, tmp_path):
    lock_path = tmp_path / "gpu.lock"
    lock_path.touch()
    monkeypatch.setattr(P, "LOCK_PATH", lock_path)
    calls = []

    def should_not_run():
        calls.append("probe")
        return {"ok": True}

    monkeypatch.setattr(P, "_check_wired_limit", should_not_run)
    monkeypatch.setattr(P, "_check_quality_models", should_not_run)
    monkeypatch.setattr(P, "_check_quality_ready_chat", should_not_run)
    with lock_path.open("rb") as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = P.collect_postflight()
        fcntl.flock(owner.fileno(), fcntl.LOCK_UN)

    assert calls == []
    assert result["lock_free"]["ok"] is False
    assert result["lock_free"]["acquired_nonblocking"] is False
    for name in ("wired_limit_mb", "quality_models", "quality_ready_chat"):
        assert result[name]["ok"] is False
        assert result[name]["skipped"] is True
        assert result[name]["error_type"] == "LockNotHeld"


def test_ready_chat_null_content_is_a_structured_failure(monkeypatch):
    monkeypatch.setattr(
        P,
        "_request_json",
        lambda *_args, **_kwargs: {
            "choices": [
                {"message": {"content": None}, "finish_reason": "stop"}
            ]
        },
    )

    result = P._check_quality_ready_chat()

    assert result["ok"] is False
    assert "content" in result["error"]


def test_unexpected_probe_error_is_structured_and_receipted(monkeypatch, tmp_path):
    lock_path = tmp_path / "gpu.lock"
    lock_path.touch()
    monkeypatch.setattr(P, "LOCK_PATH", lock_path)
    monkeypatch.setattr(
        P.subprocess,
        "run",
        lambda _command, check=False, **_kwargs: SimpleNamespace(returncode=0),
    )

    def unexpected():
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(P, "_check_wired_limit", unexpected)
    monkeypatch.setattr(P, "_check_quality_models", lambda: {"ok": True})
    monkeypatch.setattr(P, "_check_quality_ready_chat", lambda: {"ok": True})

    assert P.run("probe-error", bench_dir=tmp_path) == 1
    receipt = json.loads((tmp_path / "probe-error-postflight.json").read_text())
    assert receipt["postflight_ok"] is False
    assert receipt["postflight"]["wired_limit_mb"]["ok"] is False
    assert "probe exploded" in receipt["postflight"]["wired_limit_mb"]["error"]
    assert receipt["postflight"]["lock_free"]["released_after_probes"] is True


def test_unexpected_collector_error_still_persists_receipt(monkeypatch, tmp_path):
    monkeypatch.setattr(
        P.subprocess,
        "run",
        lambda _command, check=False, **_kwargs: SimpleNamespace(returncode=0),
    )

    def unexpected():
        raise RuntimeError("collector exploded")

    monkeypatch.setattr(P, "collect_postflight", unexpected)

    assert P.run("collector-error", bench_dir=tmp_path) == 1
    receipt = json.loads((tmp_path / "collector-error-postflight.json").read_text())
    assert receipt["postflight_ok"] is False
    assert receipt["postflight"]["collector"]["ok"] is False
    assert "collector exploded" in receipt["postflight"]["collector"]["error"]


def test_single_process_source_is_clean_before_mlx_and_loads_once():
    source = _BENCHMARK.read_text()
    main = source[source.index("def main()") :]
    assert main.index("source_commit = _require_clean_source") < main.index(
        "import mlx.core as mx"
    )
    assert "single_process_bracket" in source
    assert source.count("mtplx_runtime.load(model_path, mtp=True)") == 1


def test_benchmark_verifies_guard_before_mlx_and_records_tail_installation():
    source = _BENCHMARK.read_text()
    assert source.index("load_verified_guard_window()") < source.index(
        "import mlx.core as mx"
    )
    assert "_deepseek_v4_moe_tail_install_report" in source
    assert "loaded_runtime_identity" in source
    assert "mlx_identity" in source
    assert "receipt_role" in source


def test_benchmark_install_report_proves_43_body_routes_and_stock_mtp():
    class Route:
        pass

    def stock(*_args):
        return None

    backend = SimpleNamespace(
        _InstalledMoETailRoute=Route,
        _stock_moe_tail_combine=stock,
        _MOE_TAIL=True,
        _MOE_TAIL_SELF_CHECKED=True,
        _MOE_TAIL_KERNEL=object(),
    )
    model = SimpleNamespace(
        layers=[SimpleNamespace(ffn=SimpleNamespace(_tail_combine=Route()))] * 43,
        mtp_blocks=[SimpleNamespace(ffn=SimpleNamespace(_tail_combine=stock))],
    )
    assert H._deepseek_v4_moe_tail_install_report(
        SimpleNamespace(model=model), backend
    ) == _install_report()

    backend._MOE_TAIL = False
    for layer in model.layers:
        layer.ffn._tail_combine = stock
    assert H._deepseek_v4_moe_tail_install_report(
        SimpleNamespace(model=model), backend
    ) is None


def test_route_binding_restores_stock_after_candidate():
    class Route:
        pass

    def stock(*_args):
        return None

    routes = [Route() for _ in range(43)]
    backend = SimpleNamespace(
        _InstalledMoETailRoute=Route,
        _stock_moe_tail_combine=stock,
        _MOE_TAIL_SELF_CHECKED=True,
        _MOE_TAIL_KERNEL=object(),
    )
    model = SimpleNamespace(
        layers=[SimpleNamespace(ffn=SimpleNamespace(_tail_combine=route)) for route in routes],
        mtp_blocks=[SimpleNamespace(ffn=SimpleNamespace(_tail_combine=stock))],
    )
    runtime = SimpleNamespace(model=model)
    captured = H._capture_moe_tail_routes(runtime, backend)
    H._bind_moe_tail_routes(runtime, backend, captured, candidate=False)
    assert all(layer.ffn._tail_combine is stock for layer in model.layers)
    H._bind_moe_tail_routes(runtime, backend, captured, candidate=True)
    assert [layer.ffn._tail_combine for layer in model.layers] == routes
    H._bind_moe_tail_routes(runtime, backend, captured, candidate=False)
    assert all(layer.ffn._tail_combine is stock for layer in model.layers)
    assert model.mtp_blocks[0].ffn._tail_combine is stock


def test_route_census_proves_candidate_only_on_body_k3():
    class Route:
        pass

    def stock(*_args):
        return None

    routes = tuple(Route() for _ in range(43))
    backend = SimpleNamespace(_stock_moe_tail_combine=stock)
    model = SimpleNamespace(
        layers=[SimpleNamespace(ffn=SimpleNamespace(_tail_combine=route)) for route in routes],
        mtp_blocks=[SimpleNamespace(ffn=SimpleNamespace(_tail_combine=stock))],
    )
    runtime = SimpleNamespace(model=model)
    assert H._moe_tail_route_census(runtime, backend, routes) == {
        "body_candidate": 43,
        "body_stock": 0,
        "body_other": 0,
        "mtp_stock": 1,
        "mtp_other": 0,
    }
    H._bind_moe_tail_routes(runtime, backend, routes, candidate=False)
    assert H._moe_tail_route_census(runtime, backend, routes) == {
        "body_candidate": 0,
        "body_stock": 43,
        "body_other": 0,
        "mtp_stock": 1,
        "mtp_other": 0,
    }


def test_generation_state_reset_clears_counters_and_metal_cache(monkeypatch):
    calls = []
    fake_mx = SimpleNamespace(
        synchronize=lambda: calls.append("synchronize"),
        clear_cache=lambda: calls.append("clear_cache"),
        reset_peak_memory=lambda: calls.append("reset_peak"),
    )
    monkeypatch.setattr(H, "mx", fake_mx, raising=False)
    monkeypatch.setattr(H.gc, "collect", lambda: calls.append("gc"))
    runtime = SimpleNamespace(diagnostic_counters={"verify_calls": 99})
    H._reset_benchmark_state(runtime)
    assert runtime.diagnostic_counters == {}
    assert calls == ["synchronize", "gc", "clear_cache", "reset_peak"]


def test_single_process_runner_binds_stock_ar_candidate_k3_then_restores(
    tmp_path: Path, monkeypatch
):
    class Route:
        pass

    def stock(*_args):
        return None

    routes = tuple(Route() for _ in range(43))
    backend = SimpleNamespace(_stock_moe_tail_combine=stock)
    model = SimpleNamespace(
        layers=[SimpleNamespace(ffn=SimpleNamespace(_tail_combine=route)) for route in routes],
        mtp_blocks=[SimpleNamespace(ffn=SimpleNamespace(_tail_combine=stock))],
    )
    runtime = SimpleNamespace(model=model, diagnostic_counters={})
    observed = []

    def fake_run_arm(**kwargs):
        bound = tuple(layer.ffn._tail_combine for layer in model.layers)
        observed.append(
            (
                kwargs["label"],
                kwargs["depth"],
                "candidate" if bound == routes else "stock",
            )
        )
        return {
            "label": kwargs["label"],
            "speculative_depth": kwargs["depth"],
            "generated_tokens": 256,
            "tokens": list(range(256)),
            "text": "ok",
            "decode_tokens_per_second": 30.0,
            "error": None,
        }

    monkeypatch.setattr(H, "_run_arm", fake_run_arm)
    monkeypatch.setattr(H, "_reset_benchmark_state", lambda _runtime: None)
    args = SimpleNamespace(
        max_tokens=256,
        verify_strategy="capture_commit",
        verify_core="stock",
        mtp_history_policy="committed",
        require_exact=False,
        prompt_file="prompt.txt",
    )
    out = tmp_path / "one-load"
    assert H._run_single_process_moe_tail_bracket(
        rt=runtime,
        backend=backend,
        routes=routes,
        prompt_ids=[1] * 328,
        args=args,
        common_receipt={"guard_window": {"window_id": "f" * 64}},
        out_stem=out,
    ) == 0
    assert observed == [
        ("primer AR stock", None, "stock"),
        ("primer K=3 stock", 3, "stock"),
        ("C0 AR stock", None, "stock"),
        ("C0 K=3 stock", 3, "stock"),
        ("candidate AR stock", None, "stock"),
        ("candidate K=3 candidate", 3, "candidate"),
        ("C1 AR stock", None, "stock"),
        ("C1 K=3 stock", 3, "stock"),
    ]
    identities = []
    for suffix in ("primer", "before", "candidate", "after"):
        receipt = json.loads(Path(f"{out}-{suffix}.json").read_text())
        identities.append(
            (
                receipt["single_process_bracket"]["process_pid"],
                receipt["single_process_bracket"]["model_object_id"],
            )
        )
    assert len(set(identities)) == 1
    assert all(layer.ffn._tail_combine is stock for layer in model.layers)


@pytest.mark.parametrize("gate", (None, {"enforced": True, "pass": False}))
def test_single_process_runner_fails_false_or_missing_enforced_exactness(
    tmp_path: Path, monkeypatch, gate: dict | None
):
    class Route:
        pass

    def stock(*_args):
        return None

    routes = tuple(Route() for _ in range(43))
    backend = SimpleNamespace(_stock_moe_tail_combine=stock)
    model = SimpleNamespace(
        layers=[SimpleNamespace(ffn=SimpleNamespace(_tail_combine=route)) for route in routes],
        mtp_blocks=[SimpleNamespace(ffn=SimpleNamespace(_tail_combine=stock))],
    )
    runtime = SimpleNamespace(model=model, diagnostic_counters={})

    def fake_run_arm(**kwargs):
        arm = {
            "label": kwargs["label"],
            "speculative_depth": kwargs["depth"],
            "generated_tokens": 256,
            "tokens": list(range(256)),
            "text": "ok",
            "decode_tokens_per_second": 30.0,
            "error": None,
        }
        if kwargs["depth"] == 3 and gate is not None:
            arm["spec_equals_ar"] = gate
        return arm

    monkeypatch.setattr(H, "_run_arm", fake_run_arm)
    monkeypatch.setattr(H, "_reset_benchmark_state", lambda _runtime: None)
    args = SimpleNamespace(
        max_tokens=256,
        verify_strategy="capture_commit",
        verify_core="stock",
        mtp_history_policy="committed",
        require_exact=True,
        prompt_file="prompt.txt",
    )
    out = tmp_path / "exact-failure"
    assert H._run_single_process_moe_tail_bracket(
        rt=runtime,
        backend=backend,
        routes=routes,
        prompt_ids=[1] * 328,
        args=args,
        common_receipt={
            "guard_window": {"window_id": "f" * 64},
            "spec_equals_ar_enforced": True,
        },
        out_stem=out,
    ) == 1
    for suffix in ("primer", "before", "candidate", "after"):
        receipt = json.loads(Path(f"{out}-{suffix}.json").read_text())
        assert receipt["status"] == 1


def test_direct_benchmark_refuses_before_importing_mlx(tmp_path: Path):
    fake_package = tmp_path / "mlx"
    fake_package.mkdir()
    marker = tmp_path / "mlx-imported"
    (fake_package / "__init__.py").write_text("")
    (fake_package / "core.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n"
    )
    environment = {**os.environ, "PYTHONPATH": str(tmp_path)}
    for key in tuple(environment):
        if key.startswith("MTPLX_GUARD_ATTEST_") or key.startswith(
            "MTPLX_DSV4_GUARD_WINDOW_"
        ):
            del environment[key]
    completed = subprocess.run(
        [sys.executable, str(_BENCHMARK), "--tiny"],
        capture_output=True,
        text=True,
        timeout=10,
        env=environment,
        check=False,
    )
    assert completed.returncode != 0
    assert "verified guard window environment is absent or malformed" in completed.stderr
    assert not marker.exists()


def test_guard_attestation_survives_real_zsh_four_grandchild_hops(tmp_path: Path):
    lock_path = tmp_path / "mlx.lock"
    lock_path.write_bytes(b"")
    lock_descriptor = os.open(lock_path, os.O_RDONLY)
    fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock_stat = os.fstat(lock_descriptor)
    read_fd, write_fd = os.pipe()
    nonce = "a" * 64
    output = tmp_path / "windows.jsonl"
    command = (
        'issued=$("$1" -u "$2" issue --expected-lock "$3") || exit $?; '
        "export MTPLX_DSV4_GUARD_WINDOW_PATH=${issued%%$'\\t'*}; "
        "export MTPLX_DSV4_GUARD_WINDOW_SHA256=${issued#*$'\\t'}; "
        '"$1" -u "$2" verify >> "$4" || exit $?; '
        '"$1" -u "$2" verify >> "$4" || exit $?; '
        '"$1" -u "$2" verify >> "$4" || exit $?; '
        '"$1" -u "$2" verify >> "$4"'
    )
    environment = {
        **os.environ,
        "MTPLX_GUARD_ATTEST_FD": str(read_fd),
        "MTPLX_GUARD_ATTEST_NONCE": nonce,
    }
    process = subprocess.Popen(
        (
            "/bin/zsh",
            "-c",
            command,
            "zsh",
            sys.executable,
            str(_GUARD),
            str(lock_path),
            str(output),
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        pass_fds=(read_fd,),
    )
    issued = time.monotonic_ns()
    payload = {
        "schema_version": 1,
        "nonce": nonce,
        "guard_pid": os.getpid(),
        "child_pid": process.pid,
        "issued_monotonic_ns": issued,
        "expires_monotonic_ns": issued + 60_000_000_000,
        "lock_path": str(lock_path.resolve()),
        "lock_device": lock_stat.st_dev,
        "lock_inode": lock_stat.st_ino,
    }
    os.close(read_fd)
    os.write(write_fd, json.dumps(payload).encode())
    os.close(write_fd)
    _stdout, stderr = process.communicate(timeout=15)
    fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
    os.close(lock_descriptor)
    assert process.returncode == 0, stderr
    windows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(windows) == 4
    for window in windows:
        assert window["window_id"] == windows[0]["window_id"]
        assert window["attestation"] == windows[0]["attestation"]
        assert window["lock_identity"] == windows[0]["lock_identity"]
        consumer = window["consumer_verification"]
        assert consumer["ancestry"][0] == consumer["consumer_pid"]
        assert consumer["lock_held"] is True
        assert consumer["observed_lock_device"] == lock_stat.st_dev
        assert consumer["observed_lock_inode"] == lock_stat.st_ino
    receipt_path = Path(windows[0]["receipt_path"])
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o400
    assert hashlib.sha256(receipt_path.read_bytes()).hexdigest() == windows[0][
        "receipt_sha256"
    ]
    receipt_path.unlink()
    receipt_path.parent.rmdir()


def test_live_tuple_ancestry_matches_json_roundtrip_validation(monkeypatch, tmp_path):
    expected = _guard_window()
    document = {key: expected[key] for key in V._WINDOW_KEYS}
    encoded = json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    receipt_path = tmp_path / "window.json"
    environment = {
        G.WINDOW_PATH_ENV: str(receipt_path),
        G.WINDOW_SHA256_ENV: hashlib.sha256(encoded).hexdigest(),
    }
    repository = SimpleNamespace(
        _current_process_ancestry=lambda: (300, 200, 100),
        _lock_is_held_by_other_process=lambda *_args: True,
    )
    monkeypatch.setattr(G, "_assert_mlx_not_imported", lambda: None)
    monkeypatch.setattr(G, "_read_private_receipt", lambda _path: encoded)
    monkeypatch.setattr(G, "_load_repository_guard", lambda: repository)
    monkeypatch.setattr(
        G,
        "_checked_attestation",
        lambda _attestation, _path: document["lock_identity"],
    )
    monkeypatch.setattr(G.os, "getpid", lambda: 300)

    live = G.load_verified_guard_window(environment=environment)
    serialized = json.loads(json.dumps(live))

    assert isinstance(live["consumer_verification"]["ancestry"], list)
    assert live == serialized
    assert V._guard_errors(live, "validator_live") == []
    assert V._guard_errors(serialized, "serialized") == []
    receipts = (
        _receipt(
            1_000_000.0,
            role="discarded_control_primer",
            guard_window=serialized,
        ),
        _receipt(30.0, guard_window=serialized),
        _receipt(32.0, candidate=True, guard_window=serialized),
        _receipt(30.4, bracket_index=3, guard_window=serialized),
    )
    live_result = V.validate_moe_tail_k3_bracket(
        *receipts, peak_ceiling_gib=108.0, live_guard_window=live
    )
    serialized_result = V.validate_moe_tail_k3_bracket(
        *receipts, peak_ceiling_gib=108.0, live_guard_window=serialized
    )
    assert live_result == serialized_result
    assert live_result["status"] == "PASS"
    assert live_result["errors"] == []


def test_validator_treats_tmp_and_private_tmp_as_one_lock_realpath():
    window = _guard_window()
    window["attestation"]["lock_path"] = "/tmp/mtplx-gpu-exclusive.lock"
    encoded_attestation = json.dumps(
        window["attestation"],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    window["window_id"] = hashlib.sha256(encoded_attestation).hexdigest()
    document = {key: window[key] for key in V._WINDOW_KEYS}
    encoded_document = json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    window["receipt_sha256"] = hashlib.sha256(encoded_document).hexdigest()
    assert V._guard_errors(window, "test") == []


def test_validator_passes_only_clear_gain_beyond_post_primer_control_drift():
    primer = _receipt(1_000_000.0, role="discarded_control_primer")
    before = _receipt(30.0)
    candidate = _receipt(32.0, candidate=True)
    after = _receipt(30.4, bracket_index=3)
    result = V.validate_moe_tail_k3_bracket(
        primer,
        before,
        candidate,
        after,
        peak_ceiling_gib=108.0,
        live_guard_window=_guard_window(),
    )
    assert result["status"] == "PASS"
    assert result["integrity_pass"] is True
    assert result["tokens"]["all_equal"] is True
    assert result["counters"]["all_equal"] is True
    assert result["primer"]["performance_data_used"] is False
    assert result["guard_window"]["validator_live_recheck"] is True
    assert result["control"]["candidate_delta_fraction"] > result["control"][
        "drift_fraction"
    ]


def test_validator_preserves_a_correct_but_slower_candidate_as_loss():
    result = V.validate_moe_tail_k3_bracket(
        _receipt(1000.0, role="discarded_control_primer"),
        _receipt(30.0),
        _receipt(29.0, candidate=True),
        _receipt(30.2, bracket_index=3),
        peak_ceiling_gib=108.0,
    )
    assert result["status"] == "LOSS"
    assert result["integrity_pass"] is True
    assert result["performance_pass"] is False


def test_validator_rejects_ar_negative_control_regression_beyond_ar_drift():
    result = V.validate_moe_tail_k3_bracket(
        _receipt(1000.0, role="discarded_control_primer", ar_tps=1000.0),
        _receipt(30.0, ar_tps=29.0),
        _receipt(32.0, candidate=True, ar_tps=20.0),
        _receipt(30.2, ar_tps=29.2, bracket_index=3),
        peak_ceiling_gib=108.0,
    )
    assert result["status"] == "LOSS"
    assert result["integrity_pass"] is True
    assert result["ar_negative_control"]["pass"] is False
    assert result["k3_performance"]["pass"] is True


@pytest.mark.parametrize("mutation", ("tokens", "counters"))
def test_validator_requires_exact_ar_negative_control_data(mutation: str):
    primer = _receipt(1000.0, role="discarded_control_primer")
    before = _receipt(30.0)
    candidate = _receipt(32.0, candidate=True)
    after = _receipt(30.2, bracket_index=3)
    ar = next(arm for arm in candidate["arms"] if arm["speculative_depth"] is None)
    if mutation == "tokens":
        ar["tokens"][7] = 9999
    else:
        ar["stats"]["forward_ar_plain_calls"] += 1
    result = V.validate_moe_tail_k3_bracket(
        primer, before, candidate, after, peak_ceiling_gib=108.0
    )
    assert result["status"] == "INVALID_BRACKET"
    assert any(error.startswith("AR ") for error in result["errors"])


def test_validator_requires_one_process_one_model_and_exact_order():
    primer = _receipt(1000.0, role="discarded_control_primer")
    before = _receipt(30.0)
    candidate = _receipt(32.0, candidate=True)
    after = _receipt(30.2, bracket_index=3)
    candidate["single_process_bracket"]["process_pid"] = 999
    candidate["single_process_bracket"]["model_object_id"] = 999
    result = V.validate_moe_tail_k3_bracket(
        primer, before, candidate, after, peak_ceiling_gib=108.0
    )
    assert result["status"] == "INVALID_BRACKET"
    assert any("single process" in error for error in result["errors"])


def test_validator_rejects_unproven_guard_ancestry_or_inode():
    primer = _receipt(1000.0, role="discarded_control_primer")
    before = _receipt(30.0)
    candidate = _receipt(32.0, candidate=True)
    after = _receipt(30.2, bracket_index=3)
    candidate["guard_window"]["consumer_verification"]["ancestry"] = [300, 200]
    candidate["guard_window"]["consumer_verification"]["observed_lock_inode"] = 3
    result = V.validate_moe_tail_k3_bracket(
        primer, before, candidate, after, peak_ceiling_gib=108.0
    )
    assert result["status"] == "INVALID_BRACKET"
    assert any("ancestry" in error or "inode" in error for error in result["errors"])


def test_validator_cli_writes_loss_receipt_before_returning_nonzero(tmp_path: Path):
    inputs = {
        "primer": _receipt(1000.0, role="discarded_control_primer"),
        "before": _receipt(30.0),
        "candidate": _receipt(29.0, candidate=True),
        "after": _receipt(30.2, bracket_index=3),
    }
    paths = {}
    for name, receipt in inputs.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(receipt))
    verdict = tmp_path / "validation.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(_VALIDATOR),
            "--primer",
            str(paths["primer"]),
            "--before",
            str(paths["before"]),
            "--candidate",
            str(paths["candidate"]),
            "--after",
            str(paths["after"]),
            "--out",
            str(verdict),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert verdict.is_file()
    assert json.loads(verdict.read_text())["status"] == "LOSS"


def test_validator_cli_writes_invalid_receipt_when_benchmark_aborts(tmp_path: Path):
    verdict = tmp_path / "validation.json"
    missing = [tmp_path / f"{name}.json" for name in ("primer", "before", "candidate", "after")]
    completed = subprocess.run(
        [
            sys.executable,
            str(_VALIDATOR),
            "--primer",
            str(missing[0]),
            "--before",
            str(missing[1]),
            "--candidate",
            str(missing[2]),
            "--after",
            str(missing[3]),
            "--benchmark-exit-code",
            "7",
            "--out",
            str(verdict),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    result = json.loads(verdict.read_text())
    assert result["status"] == "INVALID_BRACKET"
    assert result["integrity_pass"] is False
    assert result["performance_pass"] is False
    assert result["benchmark_exit_code"] == 7
    assert any("benchmark aborted with exit code 7" in error for error in result["errors"])
    assert any("missing benchmark receipts" in error for error in result["errors"])


def test_validator_runs_full_validation_before_invalidating_nonzero_benchmark(
    tmp_path: Path,
):
    inputs = {
        "primer": _receipt(1000.0, role="discarded_control_primer"),
        "before": _receipt(30.0),
        "candidate": _receipt(32.0, candidate=True),
        "after": _receipt(30.2, bracket_index=3),
    }
    paths = {}
    for name, receipt in inputs.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(receipt))
    verdict = tmp_path / "validation.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(_VALIDATOR),
            "--primer",
            str(paths["primer"]),
            "--before",
            str(paths["before"]),
            "--candidate",
            str(paths["candidate"]),
            "--after",
            str(paths["after"]),
            "--benchmark-exit-code",
            "7",
            "--out",
            str(verdict),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    result = json.loads(verdict.read_text())
    assert result["status"] == "INVALID_BRACKET"
    assert result["benchmark_exit_code"] == 7
    assert result["k3_performance"]["pass"] is True
    assert any("benchmark aborted with exit code 7" in error for error in result["errors"])


@pytest.mark.parametrize("gate", (None, {"enforced": True, "pass": False}))
def test_validator_rejects_false_or_missing_enforced_exactness(gate: dict | None):
    primer = _receipt(1000.0, role="discarded_control_primer")
    before = _receipt(30.0)
    candidate = _receipt(32.0, candidate=True)
    after = _receipt(30.2, bracket_index=3)
    for receipt in (primer, before, candidate, after):
        receipt["require_exact"] = True
        receipt["spec_equals_ar_enforced"] = True
        k3 = next(
            arm for arm in receipt["arms"] if arm["speculative_depth"] == 3
        )
        if gate is not None:
            k3["spec_equals_ar"] = gate
    result = V.validate_moe_tail_k3_bracket(
        primer, before, candidate, after, peak_ceiling_gib=108.0
    )
    assert result["status"] == "INVALID_BRACKET"
    assert any("enforced exactness gate" in error for error in result["errors"])


def test_validator_derives_enforcement_from_require_exact_receipt():
    primer = _receipt(1000.0, role="discarded_control_primer")
    before = _receipt(30.0)
    candidate = _receipt(32.0, candidate=True)
    after = _receipt(30.2, bracket_index=3)
    for receipt in (primer, before, candidate, after):
        receipt["require_exact"] = True
    result = V.validate_moe_tail_k3_bracket(
        primer, before, candidate, after, peak_ceiling_gib=108.0
    )
    assert result["status"] == "INVALID_BRACKET"
    assert any("exactness enforcement receipt is inconsistent" in error for error in result["errors"])


def test_validator_rejects_incorrect_non_hot_route_census():
    primer = _receipt(1000.0, role="discarded_control_primer")
    before = _receipt(30.0)
    candidate = _receipt(32.0, candidate=True)
    after = _receipt(30.2, bracket_index=3)
    candidate["route_census"]["k3"]["body_candidate"] = 42
    candidate["route_census"]["k3"]["body_other"] = 1
    result = V.validate_moe_tail_k3_bracket(
        primer, before, candidate, after, peak_ceiling_gib=108.0
    )
    assert result["status"] == "INVALID_BRACKET"
    assert any("callable route census" in error for error in result["errors"])


@pytest.mark.parametrize("mutation", ("tokens", "counters", "peak", "guard"))
def test_validator_rejects_integrity_mismatch(mutation: str):
    primer = _receipt(1000.0, role="discarded_control_primer")
    before = _receipt(30.0)
    candidate = _receipt(32.0, candidate=True)
    after = _receipt(30.2, bracket_index=3)
    if mutation == "tokens":
        candidate["arms"][0]["tokens"][4] = 999
    elif mutation == "counters":
        candidate["arms"][0]["stats"]["verify_calls"] += 1
    elif mutation == "peak":
        candidate["arms"][0]["peak_gib"] = 109.0
    else:
        candidate["guard_window"] = _guard_window(child_pid=201)
    result = V.validate_moe_tail_k3_bracket(
        primer, before, candidate, after, peak_ceiling_gib=108.0
    )
    assert result["status"] == "INVALID_BRACKET"
    assert result["integrity_pass"] is False


@pytest.mark.parametrize(
    "mutation",
    ("model", "config", "index", "prompt", "mlx", "topology", "quant", "env"),
)
def test_validator_rejects_noncanonical_identity(mutation: str):
    primer = _receipt(1000.0, role="discarded_control_primer")
    before = _receipt(30.0)
    candidate = _receipt(32.0, candidate=True)
    after = _receipt(30.2, bracket_index=3)
    target = candidate
    if mutation == "model":
        target["model_path"] += "-wrong"
    elif mutation == "config":
        target["artifact_identity"]["config_sha256"] = "0" * 64
    elif mutation == "index":
        target["artifact_identity"]["index_sha256"] = "0" * 64
    elif mutation == "prompt":
        target["prompt"]["sha256"] = "0" * 64
    elif mutation == "mlx":
        target["mlx_identity"]["version"] = "0.32.0"
    elif mutation == "topology":
        target["artifact_identity"]["num_nextn_predict_layers"] = 0
    elif mutation == "quant":
        target["loaded_runtime_identity"]["body_q2_routed_projections"] = 0
    else:
        target["launch_mtplx_env"]["SURPRISE"] = "1"
    result = V.validate_moe_tail_k3_bracket(
        primer, before, candidate, after, peak_ceiling_gib=108.0
    )
    assert result["status"] == "INVALID_BRACKET"


def test_validator_requires_candidate_report_and_stock_controls():
    primer = _receipt(1000.0, role="discarded_control_primer")
    before = _receipt(30.0)
    candidate = _receipt(32.0, candidate=True)
    after = _receipt(30.2, bracket_index=3)
    candidate["deepseek_v4_moe_tail"] = None
    before["deepseek_v4_moe_tail"] = _install_report()
    result = V.validate_moe_tail_k3_bracket(
        primer, before, candidate, after, peak_ceiling_gib=108.0
    )
    assert result["status"] == "INVALID_BRACKET"
    assert any("installation" in error or "stock" in error for error in result["errors"])
