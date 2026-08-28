from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/qwen38_fixed_k3_xhigh_gate.py"


def _module():
    spec = importlib.util.spec_from_file_location("qwen38_fixed_k3_xhigh_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_specs_pair_v292_with_current_fixed_routes(tmp_path: Path) -> None:
    gate = _module()

    specs = gate.build_lane_specs(
        baseline_root=tmp_path / "baseline",
        baseline_commit=gate.matrix.V292_COMMIT,
        candidate_root=tmp_path / "candidate",
        candidate_commit="c" * 40,
        routes=("control", "r08_device_draft"),
    )

    assert tuple(specs) == (
        "v2.9.2-mlx0322",
        "candidate-control",
        "candidate-r08-device-draft",
    )
    assert gate.paired_order(specs) == (
        "v2.9.2-mlx0322",
        "candidate-control",
        "candidate-r08-device-draft",
        "candidate-r08-device-draft",
        "candidate-control",
        "v2.9.2-mlx0322",
    )


@pytest.mark.parametrize("route", ("r11_position_ema", "r28_q4_mtp_block"))
def test_specs_reject_adaptive_and_q4_routes(route: str, tmp_path: Path) -> None:
    gate = _module()

    with pytest.raises(ValueError, match="fixed BF16"):
        gate.build_lane_specs(
            baseline_root=tmp_path / "baseline",
            baseline_commit=gate.matrix.V292_COMMIT,
            candidate_root=tmp_path / "candidate",
            candidate_commit="c" * 40,
            routes=(route,),
        )


def test_diagnostic_arm_command_uses_exact_16k_xhigh_contract(tmp_path: Path) -> None:
    gate = _module()
    lane = gate.matrix.LaneSpec(
        lane_id="candidate-r08-device-draft",
        source_root=tmp_path / "candidate",
        source_commit="c" * 40,
        route_id="r08_device_draft",
    )

    command = gate.arm_command(
        lane=lane,
        output=tmp_path / "arm.json",
        model=tmp_path / "model",
        prompt_file=tmp_path / "prompt.jsonl",
        context_file=tmp_path / "context.py",
        row17_artifact=tmp_path / "row17.safetensors",
        python=tmp_path / "python",
        lock=tmp_path / "gpu.lock",
    )
    joined = " ".join(map(str, command))

    assert "--workload xhigh" in joined
    assert "--prompt-tokens 16384" in joined
    assert "--max-tokens 1024" in joined
    assert "--row17-artifact" in command
    assert str((tmp_path / "row17.safetensors").resolve()) in command
    assert "--warmup-tokens 1024" in joined
    assert "--force-exact-output" in command
    assert "--allow-fixed-diagnostic-route" in command


def test_gate_requires_explicit_frozen_context_artifact(tmp_path: Path) -> None:
    gate = _module()
    parser = gate._parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--baseline-root", str(tmp_path / "baseline"),
                "--routes", "control",
                "--model", str(tmp_path / "model"),
                "--row17-artifact", str(tmp_path / "row17.safetensors"),
                "--output-root", str(tmp_path / "output"),
            ]
        )


def test_aggregate_rejects_per_lane_token_nondeterminism(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gate = _module()
    specs = gate.build_lane_specs(
        baseline_root=tmp_path / "baseline",
        baseline_commit=gate.matrix.V292_COMMIT,
        candidate_root=tmp_path / "candidate",
        candidate_commit="c" * 40,
        routes=("control",),
    )
    order = gate.paired_order(specs)
    receipts = []
    for index, lane_id in enumerate(order):
        receipts.append(
            {
                "lane_id": lane_id,
                "wall_s": 10.0,
                "prefill_tok_s": 800.0,
                "decode_tok_s": 40.0,
                "peak_memory_gib": 24.0,
                "verify_calls": 100,
                "bonus_tokens": 50,
                "correction_tokens": 50,
                "draft_time_s": 2.0,
                "token_hash": f"hash-{index}",
                "prompt_token_sha256": "prompt",
                "prompt_artifact_sha256": "artifact",
                "context_artifact_sha256": "context",
                "model_artifact_hashes": {"config.json": "a" * 64},
                "row17_artifact_sha256": "row17",
            }
        )
    monkeypatch.setattr(gate.matrix, "receipt_errors", lambda *args, **kwargs: [])

    combined = gate.aggregate(order=order, receipts=receipts, specs=specs)

    assert any("token nondeterminism" in error for error in combined["invariant_errors"])
