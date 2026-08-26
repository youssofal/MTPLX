from types import SimpleNamespace

import pytest

from mtplx import cli
from mtplx.benchmarks.runners import dflash2_depth_sweep as runner


MODEL = "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"
QUALITY_MODEL = "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality"
DRAFT = "z-lab/Qwen3.8-27B-DFlash2"
RESOLVED = "/cache/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed"
QUALITY_RESOLVED = "/cache/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Quality"
DRAFT_REVISION = "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
PINNED_DRAFT = f"/cache/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/{DRAFT_REVISION}"


def _verified_inspection(runtime_contract):
    return {
        "passes_primary_gate": True,
        "model_type": "qwen3_5_text",
        "architecture": "Qwen3_5ForConditionalGeneration",
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        "quantization": {
            "bits": 4,
            "group_size": 32,
            "language_model.lm_head": {
                "bits": 8,
                "group_size": 64,
                "mode": "affine",
            },
        },
        "mtp": {"passes_tensor_gate": True, "sidecar_format": "bf16"},
        "compatibility": {
            "tier": "verified",
            "arch_id": "qwen3-next-mtp",
            "support_level": "verified-native",
            "runtime_contract": runtime_contract,
        },
    }


def _verified_quality_inspection(runtime_contract):
    inspection = _verified_inspection(runtime_contract)
    inspection["quantization"] = {
        "bits": 8,
        "group_size": 64,
        "mode": "affine",
    }
    inspection["mtp"] = {
        "passes_tensor_gate": True,
        "sidecar_format": "prequantized-mlx-affine",
    }
    return inspection


def test_quality_runtime_contract_is_accepted_as_a_distinct_verified_target():
    runtime_contract = {
        "recommended_profile": "turbo",
        "mtp_depth_max": 3,
        "mtp_contract": {
            "mtp_quant_group_size": 64,
            "mtp_quant_mode": "affine",
        },
        "verified_on": {"model": "Qwen3.8-27B-MTPLX-Optimized-Quality"},
    }

    assert (
        runner._validated_runtime_contract(
            _verified_quality_inspection(runtime_contract),
            model_id=QUALITY_MODEL,
        )
        is runtime_contract
    )


def test_run_cli_sweep_routes_quality_through_its_own_contract(monkeypatch):
    class QualityReachedDraftResolution(RuntimeError):
        pass

    runtime_contract = {
        "recommended_profile": "turbo",
        "mtp_depth_max": 3,
        "mtp_contract": {
            "mtp_quant_group_size": 64,
            "mtp_quant_mode": "affine",
        },
        "verified_on": {"model": "Qwen3.8-27B-MTPLX-Optimized-Quality"},
    }
    monkeypatch.setattr(runner, "_resolve_model_path", lambda _model: QUALITY_RESOLVED)
    monkeypatch.setattr(
        runner,
        "_inspect_model",
        lambda _model: SimpleNamespace(
            to_dict=lambda: _verified_quality_inspection(runtime_contract)
        ),
    )
    monkeypatch.setattr(
        runner,
        "_resolve_draft_snapshot",
        lambda *_args: (_ for _ in ()).throw(QualityReachedDraftResolution()),
    )

    with pytest.raises(QualityReachedDraftResolution):
        runner.run_cli_sweep(
            SimpleNamespace(
                model=QUALITY_MODEL,
                draft_model=DRAFT,
                widths="1,8",
                repetitions=3,
            ),
            token_count=1024,
        )


def test_dflash_defaults_are_qwen38_greedy():
    args = cli.build_parser().parse_args(["dflash-mlx-baseline"])

    assert args.model == MODEL
    assert args.draft_model == DRAFT
    assert args.temperature == 0.0
    assert args.top_p == 1.0
    assert args.top_k == 0
    assert args.max_tokens == 1024
    assert args.block_size == 8


def test_depth_sweep_parser_is_closed_to_qwen38_workload():
    args = cli.build_parser().parse_args(
        ["dflash2-depth-sweep", "--widths", "1,2,8", "--output", "result.json"]
    )

    assert args.draft_model == "z-lab/Qwen3.8-27B-DFlash2"
    assert args.profile == "turbo"
    assert args.widths == "1,2,8"
    assert args.repetitions == 3
    assert args.prompt_tokens == 1024
    assert args.max_tokens == 1024


def test_depth_sweep_parser_rejects_non_contract_workload():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["dflash2-depth-sweep", "--max-tokens", "32", "--output", "x.json"]
        )


@pytest.mark.parametrize(("selection", "expected"), [(None, 2), ({"best": 8}, 0)])
def test_depth_sweep_command_exit_tracks_parity_selection(
    monkeypatch, tmp_path, selection, expected
):
    args = cli.build_parser().parse_args(
        ["dflash2-depth-sweep", "--output", str(tmp_path / "result.json")]
    )
    written = []
    monkeypatch.setattr(
        runner,
        "run_cli_sweep",
        lambda _args, *, token_count: {
            "selection": selection,
            "workload": {"generated_tokens": token_count},
        },
    )
    monkeypatch.setattr(
        runner,
        "write_depth_sweep_result",
        lambda path, receipt: written.append((path, receipt)),
        raising=False,
    )

    assert args.func(args) == expected
    assert written[0][0] == args.output
    assert written[0][1]["selection"] == selection


def test_run_cli_sweep_applies_turbo_contract_before_one_target_load(monkeypatch):
    calls = []
    runtime_contract = {
        "recommended_profile": "turbo",
        "mtp_depth_max": 3,
        "mtp_contract": {
            "mtp_quant_group_size": 64,
            "mtp_quant_mode": "affine",
        },
        "verified_on": {"model": "Qwen3.8-27B-MTPLX-Optimized-Speed"},
        "runtime_env_overrides": {"MTPLX_QWEN_COMBINE_TAIL": 1},
    }
    inspection = SimpleNamespace(to_dict=lambda: _verified_inspection(runtime_contract))
    bundle = SimpleNamespace(
        runtime=object(),
        tokenizer=object(),
        checkpoint_block_size=8,
        target_layer_ids=(5, 19, 33, 47, 61),
        draft_meta={
            "resolved_model_ref": PINNED_DRAFT,
            "config": {
                "dflash_config": {
                    "block_size": 8,
                    "target_layer_ids": [5, 19, 33, 47, 61],
                }
            },
            "draft_quant": {"weight_bits": 4, "group_size": 64, "act_bits": 16},
        },
    )
    prompt = SimpleNamespace(token_ids=(4, 5, 6))

    monkeypatch.setattr(
        runner,
        "_resolve_model_path",
        lambda model: calls.append(("resolve", model))
        or RESOLVED,
    )
    monkeypatch.setattr(
        runner,
        "_inspect_model",
        lambda model: calls.append(("inspect", model)) or inspection,
        raising=False,
    )
    monkeypatch.setattr(
        runner,
        "_runtime_env_overrides_from_contract",
        lambda contract: calls.append(("contract", contract)) or {"resolved": "1"},
        raising=False,
    )
    monkeypatch.setattr(
        runner,
        "_apply_profile_env",
        lambda profile, **kwargs: calls.append(("profile", profile, kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        runner,
        "_profile_env_overridden",
        lambda: calls.append(("profile-overrides",)) or (),
        raising=False,
    )
    monkeypatch.setattr(
        runner,
        "_load_mtplx_dflash2_bundle",
        lambda model, draft: calls.append(("load", model, draft)) or bundle,
        raising=False,
    )
    monkeypatch.setattr(
        runner,
        "_resolve_draft_snapshot",
        lambda repo, revision: calls.append(("draft-snapshot", repo, revision))
        or PINNED_DRAFT,
        raising=False,
    )
    monkeypatch.setattr(
        runner,
        "_install_draft_lm_head",
        lambda runtime, **kwargs: calls.append(("head", runtime, kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        runner,
        "_build_exact_python_prompt_ids",
        lambda tokenizer, **kwargs: calls.append(("prompt", tokenizer, kwargs)) or prompt,
        raising=False,
    )
    monkeypatch.setattr(
        runner,
        "run_dflash2_depth_sweep",
        lambda **kwargs: calls.append(("sweep", kwargs)) or {"selection": {"best": 8}},
    )
    args = SimpleNamespace(
        model=MODEL,
        draft_model=DRAFT,
        widths="1,8",
        repetitions=3,
    )

    receipt = runner.run_cli_sweep(args, token_count=1024)
    assert receipt["selection"] == {"best": 8}
    assert receipt["model"] == {
        "requested": MODEL,
        "resolved": RESOLVED,
        "draft": {
            "requested": DRAFT,
            "revision": DRAFT_REVISION,
            "resolved": PINNED_DRAFT,
            "quant": {"weight_bits": 4, "group_size": 64, "act_bits": 16},
        },
        "profile": "turbo",
        "mtp_depth": 3,
        "draft_lm_head": {"bits": 4, "group_size": 64, "mode": "affine"},
    }
    assert calls[:9] == [
        ("resolve", MODEL),
        ("inspect", RESOLVED),
        ("draft-snapshot", DRAFT, DRAFT_REVISION),
        ("contract", runtime_contract),
        ("profile", "turbo", {"runtime_env_overrides": {"resolved": "1"}}),
        ("profile-overrides",),
        ("load", RESOLVED, PINNED_DRAFT),
        ("head", bundle.runtime, {"bits": 4, "group_size": 64, "mode": "affine"}),
        ("prompt", bundle.tokenizer, {"token_count": 1024}),
    ]
    assert calls[9] == (
        "sweep",
        {
            "bundle": bundle,
            "prompt_ids": prompt.token_ids,
            "widths": (1, 8),
            "repetitions": 3,
            "max_tokens": 1024,
        },
    )


def test_loaded_draft_must_match_pinned_snapshot_and_quantization():
    bundle = SimpleNamespace(
        checkpoint_block_size=8,
        target_layer_ids=(5, 19, 33, 47, 61),
        draft_meta={
            "resolved_model_ref": "/cache/snapshots/different",
            "config": {
                "dflash_config": {
                    "block_size": 8,
                    "target_layer_ids": [5, 19, 33, 47, 61],
                }
            },
            "draft_quant": {"weight_bits": 4, "group_size": 64, "act_bits": 16},
        },
    )

    with pytest.raises(ValueError, match="pinned snapshot"):
        runner._validated_draft_meta(bundle, PINNED_DRAFT)


@pytest.mark.parametrize(
    ("model", "draft", "message"),
    [
        ("some-other-model", DRAFT, "Optimized Speed"),
        (MODEL, "some-other-draft", "Qwen3.8 DFlash2"),
    ],
)
def test_run_cli_sweep_rejects_substitute_models(model, draft, message):
    args = SimpleNamespace(
        model=model,
        draft_model=draft,
        widths="1,8",
        repetitions=3,
    )
    with pytest.raises(ValueError, match=message):
        runner.run_cli_sweep(args, token_count=1024)


@pytest.mark.parametrize(
    ("widths", "repetitions", "message"),
    [("1,9", 3, "between 1 and 8"), ("1,8", 0, "repetitions")],
)
def test_run_cli_sweep_rejects_invalid_sweep_before_resolution(
    monkeypatch, widths, repetitions, message
):
    monkeypatch.setattr(
        runner,
        "_resolve_model_path",
        lambda _model: pytest.fail("invalid sweep reached model resolution"),
    )
    args = SimpleNamespace(
        model=MODEL,
        draft_model=DRAFT,
        widths=widths,
        repetitions=repetitions,
    )

    with pytest.raises(ValueError, match=message):
        runner.run_cli_sweep(args, token_count=1024)


def test_run_cli_sweep_rejects_operator_profile_contamination(monkeypatch):
    runtime_contract = {
        "recommended_profile": "turbo",
        "mtp_depth_max": 3,
        "mtp_contract": {
            "mtp_quant_group_size": 64,
            "mtp_quant_mode": "affine",
        },
        "verified_on": {"model": "Qwen3.8-27B-MTPLX-Optimized-Speed"},
    }
    monkeypatch.setattr(runner, "_resolve_model_path", lambda _model: RESOLVED)
    monkeypatch.setattr(
        runner,
        "_inspect_model",
        lambda _model: SimpleNamespace(
            to_dict=lambda: _verified_inspection(runtime_contract)
        ),
    )
    monkeypatch.setattr(runner, "_runtime_env_overrides_from_contract", lambda _c: {})
    monkeypatch.setattr(
        runner,
        "_resolve_draft_snapshot",
        lambda _repo, _revision: PINNED_DRAFT,
    )
    monkeypatch.setattr(runner, "_apply_profile_env", lambda *_a, **_k: None)
    monkeypatch.setattr(
        runner,
        "_profile_env_overridden",
        lambda: ({"var": "MTPLX_COMPILED_VERIFY", "actual_value": "0"},),
        raising=False,
    )
    args = SimpleNamespace(
        model=MODEL,
        draft_model=DRAFT,
        widths="1,8",
        repetitions=3,
    )

    with pytest.raises(ValueError, match="operator environment"):
        runner.run_cli_sweep(args, token_count=1024)


def test_in_place_dflash_channel_no_longer_imports_obsolete_module():
    import inspect

    from mtplx.benchmarks.runners.competitor_baselines import (
        run_dflash_mlx_baseline,
    )

    source = inspect.getsource(run_dflash_mlx_baseline)
    assert "dflash.model_mlx" not in source
    assert "run_cli_sweep" in source
