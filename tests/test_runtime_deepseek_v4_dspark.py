from types import SimpleNamespace

import pytest

from mtplx import runtime
from mtplx.deepseek_v4_mia_engine import MiaDeepseekV4EnginePlan


def _exact_mia_config() -> dict:
    return {
        "model_type": "deepseek_v4",
        "hybrid_tr3_tail": {"format": "exl3-trellis"},
        "num_nextn_predict_layers": 1,
        "dspark_block_size": 5,
        "dspark_markov_rank": 256,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [40, 41, 42],
    }


def _sealed_plan() -> MiaDeepseekV4EnginePlan:
    return MiaDeepseekV4EnginePlan(
        context_capacity_tokens=384_000,
        target_physical_capacity_tokens=384_005,
        max_batch_tokens=8_224,
        max_sequences=1,
        page_geometry=(),
        workspace_geometry=(),
        indexer_workspace=None,
        indexer_rope_table=None,
        mla_workspace=None,
        target_cache_arena=object(),
        prewarm_signatures=(),
        installed_routes=(),
        target_artifact="target",
        draft_artifact="draft",
        artifact_small_file_sha256=(),
        identity="test-plan",
    )


def test_explicit_load_routes_only_the_pinned_mia_loader(
    monkeypatch,
    tmp_path,
) -> None:
    events = []

    class _LoadModel:
        dspark = SimpleNamespace(stages=[object(), object(), object()])
        _mia_engine_plan = _sealed_plan()

    model = _LoadModel()
    monkeypatch.setattr(runtime, "load_config", lambda _path: _exact_mia_config())
    monkeypatch.setattr(runtime, "_load_runtime_metadata", lambda _path: {})
    monkeypatch.setattr(
        "mtplx.deepseek_v4_dspark_artifact.open_verified_dspark_artifact",
        lambda _path: pytest.fail("legacy generic DSpark verifier was reached"),
    )
    monkeypatch.setattr(
        runtime,
        "_load_base_model",
        lambda *_args: events.append("model") or (model, object()),
    )

    loaded = runtime.load(tmp_path, mtp=True, dspark=True)

    assert events == ["model"]
    assert loaded.model is model
    assert loaded.mtp_enabled is False
    assert loaded.backend_id == "deepseek_v4_dspark"
    assert not hasattr(loaded, "deepseek_v4_dspark_runtime")


def test_dspark_rejects_generic_deepseek_before_model_construction(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        runtime,
        "load_config",
        lambda _path: {"model_type": "deepseek_v4"},
    )
    monkeypatch.setattr(
        runtime,
        "_load_base_model",
        lambda *_args: pytest.fail("generic DeepSeek model construction was reached"),
    )

    with pytest.raises(ValueError, match="pinned Mia/Sero"):
        runtime.load(tmp_path, dspark=True)


def test_dspark_rejects_a_loader_result_without_the_sealed_engine_plan(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(runtime, "load_config", lambda _path: _exact_mia_config())
    monkeypatch.setattr(runtime, "_load_runtime_metadata", lambda _path: {})
    monkeypatch.setattr(
        "mtplx.deepseek_v4_dspark_artifact.open_verified_dspark_artifact",
        lambda _path: SimpleNamespace(),
    )
    monkeypatch.setattr(
        runtime,
        "_load_base_model",
        lambda *_args: (SimpleNamespace(), object()),
    )

    with pytest.raises(RuntimeError, match="sealed Mia engine plan"):
        runtime.load(tmp_path, dspark=True)


def test_server_parser_accepts_explicit_fixed_dspark_route() -> None:
    from mtplx.server.openai import parse_args

    args = parse_args(
        [
            "--model",
            "/tmp/dspark-model",
            "--generation-mode",
            "dspark",
            "--depth",
            "5",
            "--temperature",
            "0",
            "--top-p",
            "1",
            "--top-k",
            "0",
            "--default-presence-penalty",
            "0",
            "--default-frequency-penalty",
            "0",
            "--scheduler-mode",
            "serial",
            "--warmup-tokens",
            "0",
        ]
    )

    assert args.generation_mode == "dspark"
    assert args.depth == 5
    assert args.temperature == 0.0
    assert args.top_p == 1.0
    assert args.top_k == 0
    assert args.default_presence_penalty == 0.0
    assert args.default_frequency_penalty == 0.0
    assert args.scheduler_mode == "serial"


@pytest.mark.parametrize(
    ("sampler_flag", "value"),
    [
        ("--temperature", "0.6"),
        ("--top-p", "0.95"),
        ("--top-k", "20"),
        ("--default-presence-penalty", "0.5"),
        ("--default-frequency-penalty", "-0.5"),
    ],
)
def test_server_parser_rejects_non_greedy_dspark_launch_before_model_load(
    sampler_flag: str,
    value: str,
) -> None:
    from mtplx.server.openai import parse_args

    with pytest.raises(
        ValueError,
        match=(
            r"DeepSeek V4 DSpark requires.*temperature 0.*top-p 1.*top-k 0"
            r".*presence-penalty 0.*frequency-penalty 0"
        ),
    ):
        parse_args(
            [
                "--model",
                "/tmp/dspark-model",
                "--generation-mode",
                "dspark",
                sampler_flag,
                value,
                "--warmup-tokens",
                "0",
            ]
        )


@pytest.mark.parametrize(
    "scheduler_mode",
    [
        "cooperative",
        "ar_batch",
        "mtp_batch",
        "mtp_cohort_experimental",
    ],
)
def test_server_parser_rejects_nonserial_dspark_scheduler_before_model_load(
    scheduler_mode: str,
) -> None:
    from mtplx.server.openai import parse_args

    with pytest.raises(
        ValueError,
        match=r"DeepSeek V4 DSpark requires --scheduler-mode serial",
    ):
        parse_args(
            [
                "--model",
                "/tmp/dspark-model",
                "--generation-mode",
                "dspark",
                "--scheduler-mode",
                scheduler_mode,
                "--warmup-tokens",
                "0",
            ]
        )


def test_server_parser_selects_dspark_backend_defaults_at_construction() -> None:
    from mtplx.server.openai import parse_args

    args = parse_args(
        [
            "--model",
            "/tmp/dspark-model",
            "--generation-mode",
            "dspark",
            "--warmup-tokens",
            "0",
        ]
    )

    assert args.backend_id == "deepseek_v4_dspark"
    assert args.model_id == "deepseek-v4-dspark-dflash2"
    assert args.depth == 5
    assert args.temperature == 0.0
    assert args.top_p == 1.0
    assert args.top_k == 0
    assert args.chat_template_profile == "tokenizer"
    assert args.reasoning_parser == "none"
    assert args.reasoning == "off"
    assert args.enable_thinking is False


@pytest.mark.parametrize("mode", ["mtp", "ar"])
def test_explicit_sealed_backend_rejects_a_non_dspark_mode(mode: str) -> None:
    from mtplx.server.openai import parse_args

    with pytest.raises(ValueError, match="deepseek_v4_dspark.*generation-mode dspark"):
        parse_args(
            [
                "--model",
                "/tmp/dspark-model",
                "--backend-id",
                "deepseek_v4_dspark",
                "--generation-mode",
                mode,
            ]
        )


def test_stock_ar_normalizes_before_dspark_backend_selection() -> None:
    from mtplx.server.openai import parse_args

    args = parse_args(
        [
            "--model",
            "/tmp/dspark-model",
            "--generation-mode",
            "dspark",
            "--stock-ar",
        ]
    )

    assert args.generation_mode == "ar"
    assert args.load_mtp is False
    assert args.backend_id != "deepseek_v4_dspark"


def test_dspark_descriptor_and_health_modes_report_the_installed_lane() -> None:
    from mtplx.backends.descriptors import (
        context_window_policy_for_model,
        descriptor_for_backend_id,
    )
    from mtplx.server.openai import _available_generation_modes

    descriptor = descriptor_for_backend_id("deepseek_v4_dspark")
    assert descriptor.model_family == "deepseek"
    assert descriptor.artifact_layout == "split_mia_tp1_target_plus_k64_draft"
    assert descriptor.uses_draft_lm_head is False
    assert descriptor.draft_semantics.minimum == 5
    assert descriptor.draft_semantics.maximum == 5
    assert descriptor.required_chat_template_profile == "tokenizer"
    assert descriptor.reasoning_codec.supported is False
    assert descriptor.context_window_policy.default == 384_000
    assert descriptor.context_window_policy.maximum == 384_000
    assert all("affine int4" not in note for note in descriptor.notes)
    exposed_policy = context_window_policy_for_model(
        descriptor=descriptor,
        inspection={"model_context_window": 1_048_576},
    )
    assert exposed_policy.maximum == 384_000
    assert exposed_policy.source == "sealed_mia_engine_plan"

    state = SimpleNamespace(
        args=SimpleNamespace(generation_mode="dspark"),
        runtime=SimpleNamespace(mtp_enabled=False),
        deepseek_v4_dflash2_bundle=object(),
    )
    assert _available_generation_modes(state) == ["dspark"]


def test_server_dspark_route_requires_the_bound_dflash2_bundle() -> None:
    from fastapi import HTTPException
    from mtplx.server.openai import _request_generation_mode_for_generation

    request = SimpleNamespace(generation_mode=None, model_extra=None)
    state = SimpleNamespace(
        args=SimpleNamespace(generation_mode="dspark"),
        runtime=SimpleNamespace(mtp_enabled=False),
        deepseek_v4_dflash2_bundle=object(),
    )

    assert _request_generation_mode_for_generation(state, request) == "dspark"
    state.deepseek_v4_dflash2_bundle = None
    with pytest.raises(HTTPException, match="DFlash2-qualified"):
        _request_generation_mode_for_generation(state, request)


def test_dspark_context_capacity_is_bound_to_the_installed_engine_plan() -> None:
    from mtplx.backends.descriptors import descriptor_for_backend_id
    from mtplx.server.openai import _installed_backend_context_capacity

    backend = descriptor_for_backend_id("deepseek_v4_dspark")
    runtime_owner = SimpleNamespace(
        model=SimpleNamespace(_mia_engine_plan=_sealed_plan())
    )

    assert (
        _installed_backend_context_capacity(
            backend,
            runtime_owner,
            configured_model_max=1_048_576,
        )
        == 384_000
    )


def test_descriptor_from_runtime_rejects_sealed_backend_from_args_alone() -> None:
    from mtplx.backends.descriptors import descriptor_from_runtime

    generic_runtime = SimpleNamespace(backend_id=None, model=SimpleNamespace())
    args = SimpleNamespace(backend_id="deepseek_v4_dspark")

    with pytest.raises(RuntimeError, match="sealed DSpark backend"):
        descriptor_from_runtime(generic_runtime, args)


def test_descriptor_from_runtime_requires_the_installed_sealed_plan() -> None:
    from mtplx.backends.descriptors import descriptor_from_runtime

    with pytest.raises(RuntimeError, match="sealed Mia engine plan"):
        descriptor_from_runtime(
            SimpleNamespace(
                backend_id="deepseek_v4_dspark",
                model=SimpleNamespace(),
            )
        )

    descriptor = descriptor_from_runtime(
        SimpleNamespace(
            backend_id="deepseek_v4_dspark",
            model=SimpleNamespace(_mia_engine_plan=_sealed_plan()),
        )
    )
    assert descriptor.backend_id == "deepseek_v4_dspark"


def test_dspark_explicit_context_over_sealed_capacity_is_rejected_at_startup() -> None:
    from mtplx.backends.descriptors import descriptor_for_backend_id
    from mtplx.server.openai import _validate_requested_backend_context_window

    backend = descriptor_for_backend_id("deepseek_v4_dspark")

    with pytest.raises(ValueError, match="384,000"):
        _validate_requested_backend_context_window(backend, 384_001)
