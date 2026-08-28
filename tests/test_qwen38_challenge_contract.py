from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx import qwen38_challenge
from mtplx.generation import _qwen38_prefill_target_route
from mtplx.mtp_patch import MTPContract
from mtplx.qwen38_challenge import (
    DEFAULT_QWEN38_CACHE_ROUTE,
    QWEN38_FINAL_ROUTE,
    QWEN38_LOW_ADAPTIVE_ROUTE,
    QWEN38_LOW_FIXED_ROUTE,
    QWEN38_Q8_LINEAR_ATTN_LAYERS,
    QWEN38_XHIGH_ADAPTIVE_ROUTE,
    QWEN38_XHIGH_FIXED_ROUTE,
    Qwen38ContractError,
    Qwen38PerformanceProfileConfig,
    Qwen38RouteBindings,
    build_qwen38_route,
    configure_qwen38_row50_wired_residency,
    install_qwen38_control_route,
    install_qwen38_performance_profiles,
    install_qwen38_route,
    is_qwen38_27b_candidate,
    policy_fingerprint_with_qwen38_route,
    qwen38_final_route,
    qwen38_measured_performance_profile_configs,
    qwen38_route_receipt,
    select_qwen38_performance_profile,
    validate_qwen38_27b_contract,
)
from mtplx.runtime import MTPLXRuntime
from mtplx.server.openai import _qwen38_challenge_route_payload
from mtplx.session_bank import CacheMissReason, SessionBank

MODEL_PATH = Path("models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed")


def test_qwen38_challenge_module_has_no_rejected_qmv_or_source_proposal_imports() -> None:
    source = Path(qwen38_challenge.__file__).read_text(encoding="utf-8")
    assert "qwen38_qmv" not in source
    assert "qwen38_source_proposal" not in source
    assert "row70_qmv" not in source


def test_qwen38_runtime_has_no_automatic_source_proposal_route() -> None:
    import mtplx.runtime as runtime

    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "_mtplx_qwen38_pending_source_route" not in source
    assert 'pop("source_proposal"' not in source


def _config() -> dict:
    quantization: dict[str, object] = {
        "bits": 4,
        "group_size": 32,
        "mode": "affine",
        "language_model.model.embed_tokens": {
            "bits": 8,
            "group_size": 64,
            "mode": "affine",
        },
        "language_model.lm_head": {
            "bits": 8,
            "group_size": 64,
            "mode": "affine",
        },
    }
    for layer in QWEN38_Q8_LINEAR_ATTN_LAYERS:
        quantization[f"language_model.model.layers.{layer}.linear_attn.out_proj"] = {
            "bits": 8,
            "group_size": 64,
            "mode": "affine",
        }
    for layer in range(56, 64):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            quantization[
                f"language_model.model.layers.{layer}.mlp.{projection}"
            ] = {"bits": 8, "group_size": 64, "mode": "affine"}
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
        "mtplx_runtime": {"base_trunk": "/models/Qwen--Qwen3.8-27B"},
        "quantization": quantization,
        "text_config": {
            "model_type": "qwen3_5_text",
            "dtype": "bfloat16",
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 248320,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "full_attention_interval": 4,
            "mtp_num_hidden_layers": 1,
        },
    }


def _callable(*args, **kwargs):
    return args, kwargs


def _bindings() -> Qwen38RouteBindings:
    return Qwen38RouteBindings(mtp_cache_append=_callable)


def _route_runtime():
    def stock(*args, **kwargs):
        return args, kwargs

    def kv_only(*args, **kwargs):
        return args, kwargs

    def stock_prepare(*args, **kwargs):
        return args, kwargs

    def dual_prepare(*args, **kwargs):
        return args, kwargs

    text = SimpleNamespace(
        _mtplx_prepare_mtp_inputs_stock=stock_prepare,
        _mtplx_prepare_mtp_inputs_dual=dual_prepare,
    )
    runtime = MTPLXRuntime(
        model=SimpleNamespace(
            mtp_update_cache=stock,
            mtp_update_cache_kv_only_history=kv_only,
            language_model=text,
        ),
        tokenizer=SimpleNamespace(),
        model_path=MODEL_PATH,
        mtp_enabled=True,
        contract=MTPContract(),
    )
    return runtime, stock, kv_only, stock_prepare, dual_prepare


def _target_phase_runtime():
    def stock(*args, **kwargs):
        return ("stock", args, kwargs)

    def row24(*args, **kwargs):
        return ("row24", args, kwargs)

    def row26(*args, **kwargs):
        return ("row26", args, kwargs)

    text = SimpleNamespace(
        _mtplx_forward_layers_stock=stock,
        _mtplx_forward_layers_row24=row24,
        _mtplx_forward_layers_row26=row26,
        _mtplx_prepare_mtp_inputs_stock=_callable,
    )
    runtime = MTPLXRuntime(
        model=SimpleNamespace(
            mtp_update_cache=_callable,
            language_model=text,
        ),
        tokenizer=SimpleNamespace(),
        model_path=MODEL_PATH,
        mtp_enabled=True,
        contract=MTPContract(),
    )
    return runtime, text, stock, row26


def test_exact_qwen38_27b_control_contract_is_accepted() -> None:
    contract = validate_qwen38_27b_contract(_config(), MODEL_PATH)

    assert contract.hidden_size == 5120
    assert contract.vocab_size == 248320
    assert contract.trunk_bits == 4
    assert contract.trunk_group_size == 32
    assert contract.packing == "mlx_affine_u32_le"


def test_final_route_is_the_explicit_unchanged_control() -> None:
    assert DEFAULT_QWEN38_CACHE_ROUTE == "control"
    assert dict(QWEN38_FINAL_ROUTE) == {
        "cache_route": "control",
        "dual_norm": False,
    }
    assert qwen38_final_route() == dict(QWEN38_FINAL_ROUTE)


def test_ordinary_qwen38_runtime_route_installs_no_historical_candidate(
    monkeypatch,
) -> None:
    runtime, stock, _, stock_prepare, _ = _route_runtime()
    final_route = install_qwen38_route(
        runtime,
        _config(),
        MODEL_PATH,
        **qwen38_final_route(),
    )

    assert final_route.route_id == "control"
    assert final_route.history_route_id == "stock_history"
    assert final_route.bindings.mtp_cache_append is stock
    assert final_route.kernel_ids == ()
    assert runtime.qwen38_feature_receipt == {}
    assert runtime.model.language_model._mtplx_prepare_mtp_inputs is stock_prepare

    direct_runtime, direct_stock, _, direct_stock_prepare, _ = _route_runtime()
    direct_route = install_qwen38_route(direct_runtime, _config(), MODEL_PATH)
    assert direct_route.route_id == "control"
    assert direct_route.bindings.mtp_cache_append is direct_stock
    assert direct_route.kernel_ids == ()
    assert direct_runtime.qwen38_feature_receipt == {}
    assert (
        direct_runtime.model.language_model._mtplx_prepare_mtp_inputs
        is direct_stock_prepare
    )

    monkeypatch.setattr(
        "mtplx.qwen38_challenge._validate_qwen38_dual_norm_install",
        lambda text, *, q8_embedding: None,
    )
    reset_runtime, reset_stock, _, reset_stock_prepare, _ = _route_runtime()
    candidate_route = install_qwen38_route(
        reset_runtime,
        _config(),
        MODEL_PATH,
        cache_route="kv_only_history",
        dual_norm=True,
    )
    assert candidate_route.route_id == "kv_only_history+dual_norm"

    reset_route = install_qwen38_route(
        reset_runtime,
        _config(),
        MODEL_PATH,
        **qwen38_final_route(),
    )
    assert reset_route.route_id == "control"
    assert reset_route.history_route_id == "stock_history"
    assert reset_route.bindings.mtp_cache_append is reset_stock
    assert reset_route.kernel_ids == ()
    assert reset_runtime.qwen38_feature_receipt == {}
    assert (
        reset_runtime.model.language_model._mtplx_prepare_mtp_inputs
        is reset_stock_prepare
    )


@pytest.mark.parametrize(
    "sibling",
    [
        "Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed",
        "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality",
        "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16",
    ],
)
def test_unmeasured_qwen38_siblings_stay_outside_the_route(sibling: str) -> None:
    assert not is_qwen38_27b_candidate(_config(), Path(sibling))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("family", "Qwen3.6-27B", "Qwen 3.8 27B identity"),
        ("hidden_size", 4096, "hidden_size"),
        ("dtype", "float16", "dtype"),
        ("bits", 3, "trunk quantization"),
        ("group_size", 64, "trunk quantization"),
        ("mode", "symmetric", "trunk quantization"),
        ("packing", "q4_k", "packing"),
    ],
)
def test_contract_misses_fail_loudly(field: str, value: object, message: str) -> None:
    config = _config()
    path = MODEL_PATH
    packing = "mlx_affine_u32_le"
    if field == "family":
        path = Path(f"models/{value}")
        config["mtplx_runtime"]["base_trunk"] = f"/models/{value}"
    elif field in config["text_config"]:
        config["text_config"][field] = value
    elif field in {"bits", "group_size", "mode"}:
        config["quantization"][field] = value
    else:
        packing = str(value)

    with pytest.raises(Qwen38ContractError, match=message):
        validate_qwen38_27b_contract(config, path, packing=packing)


def test_route_is_immutable_and_fingerprint_covers_kernel_and_policy() -> None:
    base = build_qwen38_route(
        _config(), MODEL_PATH, bindings=_bindings(), route_id="control"
    )

    with pytest.raises(FrozenInstanceError):
        base.route_id = "changed"  # type: ignore[misc]
    assert replace(base, kernel_ids=("kernel-b",)).fingerprint != base.fingerprint
    assert replace(base, policy_id="policy-b").fingerprint != base.fingerprint


def test_route_fingerprint_prevents_cross_route_session_restore() -> None:
    control = build_qwen38_route(
        _config(), MODEL_PATH, bindings=_bindings(), route_id="control"
    )
    candidate = replace(control, route_id="kv_only_history", kernel_ids=("kv-v1",))
    runtime = SimpleNamespace(model_path=MODEL_PATH, mtp_enabled=True)
    bank = SessionBank(max_entries=2, max_bytes=4096, per_session_max_bytes=4096)
    assert bank.put(
        runtime=runtime,
        token_ids=[1, 2, 3],
        cache=[],
        logits=None,
        hidden=None,
        policy_fingerprint=policy_fingerprint_with_qwen38_route("base", control),
        nbytes_override=64,
    )

    assert bank.restore(
        runtime,
        [1, 2, 3],
        policy_fingerprint=policy_fingerprint_with_qwen38_route("base", candidate),
        cache_factory=list,
    ) is None
    assert bank.last_miss_reason == CacheMissReason.POLICY_MISMATCH.value


def test_kv_only_history_route_binds_the_target_shaped_append() -> None:
    calls: list[str] = []

    def stock(*args, **kwargs):
        calls.append("stock")
        return "stock"

    def kv_only(*args, **kwargs):
        calls.append("kv-only")
        return "kv-only"

    runtime = MTPLXRuntime(
        model=SimpleNamespace(
            mtp_update_cache=stock,
            mtp_update_cache_kv_only_history=kv_only,
        ),
        tokenizer=SimpleNamespace(),
        model_path=MODEL_PATH,
        mtp_enabled=True,
        contract=MTPContract(),
    )
    route = install_qwen38_route(
        runtime,
        _config(),
        MODEL_PATH,
        cache_route="kv_only_history",
    )

    assert route is runtime.qwen38_route
    assert route.route_id == "kv_only_history"
    assert route.bindings.mtp_cache_append is kv_only
    assert route.min_context_tokens == 16_384
    short_route = runtime.bind_mtp_history_append_route(1_024)
    long_route = runtime.bind_mtp_history_append_route(16_384)
    short = runtime.update_mtp_cache(
        object(),
        SimpleNamespace(shape=(1, 1024)),
        mtp_cache=[SimpleNamespace(offset=0)],
        history_route=short_route,
    )
    long = runtime.update_mtp_cache(
        object(),
        SimpleNamespace(shape=(1, 16384)),
        mtp_cache=[SimpleNamespace(offset=0)],
        history_route=long_route,
    )
    continued_long = runtime.update_mtp_cache(
        object(),
        SimpleNamespace(shape=(1, 1)),
        mtp_cache=[SimpleNamespace(offset=16384)],
        history_route=long_route,
    )
    windowed_long = runtime.update_mtp_cache(
        object(),
        SimpleNamespace(shape=(1, 8192)),
        mtp_cache=[SimpleNamespace(offset=0)],
        history_route=long_route,
    )

    assert (short, long, continued_long, windowed_long) == (
        "stock",
        "kv-only",
        "kv-only",
        "kv-only",
    )
    assert calls == ["stock", "kv-only", "kv-only", "kv-only"]
    assert short_route.receipt == {
        "route_id": "stock_history",
        "prompt_tokens": 1024,
        "row20_engaged": False,
        "reason": "below_min_context",
    }
    assert long_route.receipt == {
        "route_id": "kv_only_history",
        "prompt_tokens": 16384,
        "row20_engaged": True,
        "reason": "min_context_satisfied",
    }
    assert route.kernel_ids == ("qwen38_mtp_kv_only_history_ge16384_v1",)
    assert route.selfcheck_passed is False
    assert route.selfcheck_status == "unchecked"


def test_non_control_route_defaults_to_unchecked_receipt() -> None:
    route = build_qwen38_route(
        _config(), MODEL_PATH, bindings=_bindings(), route_id="candidate"
    )

    receipt = qwen38_route_receipt(route)

    assert receipt is not None
    assert receipt["selfcheck"] == {"passed": False, "status": "unchecked"}


def test_non_row20_candidate_binds_stock_history_at_long_context() -> None:
    runtime = MTPLXRuntime(
        model=SimpleNamespace(mtp_update_cache=_callable),
        tokenizer=SimpleNamespace(),
        model_path=MODEL_PATH,
        mtp_enabled=True,
        contract=MTPContract(),
    )
    runtime.qwen38_route = build_qwen38_route(
        _config(),
        MODEL_PATH,
        bindings=_bindings(),
        route_id="r10_compact_vocab",
    )

    history_route = runtime.bind_mtp_history_append_route(16_384)

    assert runtime.qwen38_route.history_route_id == "stock_history"
    assert history_route.append is runtime.model.mtp_update_cache
    assert history_route.receipt == {
        "route_id": "stock_history",
        "prompt_tokens": 16_384,
        "row20_engaged": False,
        "reason": "route_has_no_row20",
    }


def test_control_route_and_receipt_are_stable() -> None:
    runtime = MTPLXRuntime(
        model=SimpleNamespace(mtp_update_cache=_callable),
        tokenizer=SimpleNamespace(),
        model_path=MODEL_PATH,
        mtp_enabled=True,
        contract=MTPContract(),
    )
    route = install_qwen38_control_route(runtime, _config(), MODEL_PATH)

    assert route.route_id == "control"
    assert qwen38_route_receipt(route) == {
        "route_id": "control",
        "fingerprint": route.fingerprint,
        "contract_id": route.contract.contract_id,
        "kernel_ids": [],
        "history_route_id": "stock_history",
        "min_context_tokens": 0,
        "policy_id": "current_mtplx",
        "selfcheck": {"passed": True, "status": "control"},
    }
    assert _qwen38_challenge_route_payload(runtime) == {
        **qwen38_route_receipt(route),
        "feature_receipt": {},
    }


def test_row10_candidate_route_names_compact_proposal_only_head(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtplx.qwen38_challenge.configure_qwen38_row10_compact_head",
        lambda runtime, *, active: {"installed": True, "active": active},
    )
    runtime = MTPLXRuntime(
        model=SimpleNamespace(mtp_update_cache=_callable),
        tokenizer=SimpleNamespace(),
        model_path=MODEL_PATH,
        mtp_enabled=True,
        contract=MTPContract(),
    )

    route = install_qwen38_route(
        runtime,
        _config(),
        MODEL_PATH,
        cache_route="control",
        row10_compact_vocab=True,
    )

    assert route.route_id == "r10_compact_vocab"
    assert route.kernel_ids == ("qwen38_row10_compact_q4_g64_vocab_v1",)
    assert runtime.qwen38_feature_receipt["r10_compact_vocab"]["active"] is True


def test_row17_route_installs_the_q4_group64_mtp_block(monkeypatch) -> None:
    artifact = Path("/artifacts/row17/model.safetensors")
    monkeypatch.setattr(
        "mtplx.qwen38_challenge.configure_qwen38_mtp_block",
        lambda runtime, *, variant, artifact_path: {
            "installed": variant is not None,
            "active": variant is not None,
            "variant": variant,
            "artifact_path": str(artifact_path) if artifact_path else None,
            "bits": 4 if variant else None,
            "group_size": 64 if variant else None,
        },
        raising=False,
    )
    runtime = MTPLXRuntime(
        model=SimpleNamespace(mtp_update_cache=_callable),
        tokenizer=SimpleNamespace(),
        model_path=MODEL_PATH,
        mtp_enabled=True,
        contract=MTPContract(),
    )

    route = install_qwen38_route(
        runtime,
        _config(),
        MODEL_PATH,
        cache_route="control",
        mtp_block_variant="r17",
        mtp_block_artifact_path=artifact,
    )

    assert route.route_id == "r17_q4_mtp_block"
    assert route.kernel_ids == ("qwen38_row17_q4_g64_mtp_block_v1",)
    assert runtime.qwen38_feature_receipt["r17_q4_mtp_block"] == {
        "installed": True,
        "active": True,
        "variant": "r17",
        "artifact_path": str(artifact),
        "bits": 4,
        "group_size": 64,
    }


def test_row36_route_installs_q4_block_with_bf16_qkv_islands(monkeypatch) -> None:
    artifact = Path("/artifacts/row36/model.safetensors")
    monkeypatch.setattr(
        "mtplx.qwen38_challenge.configure_qwen38_mtp_block",
        lambda runtime, *, variant, artifact_path: {
            "installed": variant is not None,
            "active": variant is not None,
            "variant": variant,
            "artifact_path": str(artifact_path) if artifact_path else None,
            "bits": 4 if variant else None,
            "group_size": 64 if variant else None,
            "precision_q_rows": 1_024 if variant == "r36" else 0,
            "precision_k_rows": 1_024 if variant == "r36" else 0,
            "precision_v_rows": 1_024 if variant == "r36" else 0,
        },
        raising=False,
    )
    runtime = MTPLXRuntime(
        model=SimpleNamespace(mtp_update_cache=_callable),
        tokenizer=SimpleNamespace(),
        model_path=MODEL_PATH,
        mtp_enabled=True,
        contract=MTPContract(),
    )

    route = install_qwen38_route(
        runtime,
        _config(),
        MODEL_PATH,
        cache_route="control",
        mtp_block_variant="r36",
        mtp_block_artifact_path=artifact,
    )

    assert route.route_id == "r17_q4_mtp_block+r36_qkv_islands"
    assert route.kernel_ids == ("qwen38_row36_q4_g64_bf16_qkv_islands_v1",)
    report = runtime.qwen38_feature_receipt["r36_qkv_islands"]
    assert report["variant"] == "r36"
    assert report["precision_q_rows"] == 1_024
    assert report["precision_k_rows"] == 1_024
    assert report["precision_v_rows"] == 1_024


def test_row18_route_names_input_independent_gdn_decay_memo(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtplx.qwen38_challenge.configure_qwen38_row18_gdn_decay_memo",
        lambda model, *, active: {
            "configured_modules": 48,
            "active_modules": 48 if active else 0,
        },
        raising=False,
    )
    runtime = MTPLXRuntime(
        model=SimpleNamespace(mtp_update_cache=_callable),
        tokenizer=SimpleNamespace(),
        model_path=MODEL_PATH,
        mtp_enabled=True,
        contract=MTPContract(),
    )

    route = install_qwen38_route(
        runtime,
        _config(),
        MODEL_PATH,
        cache_route="control",
        row18_gdn_decay_memo=True,
    )

    assert route.route_id == "r18_gdn_decay_memo"
    assert route.kernel_ids == ("qwen38_row18_gdn_neg_exp_a_log_memo_v1",)
    assert runtime.qwen38_feature_receipt["r18_gdn_decay_memo"] == {
        "configured_modules": 48,
        "active_modules": 48,
    }


def test_enabled_row18_rejects_partial_gdn_module_coverage(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtplx.qwen38_challenge.configure_qwen38_row18_gdn_decay_memo",
        lambda model, *, active: {
            "configured_modules": 47,
            "active_modules": 47 if active else 0,
        },
    )
    runtime = MTPLXRuntime(
        model=SimpleNamespace(mtp_update_cache=_callable),
        tokenizer=SimpleNamespace(),
        model_path=MODEL_PATH,
        mtp_enabled=True,
        contract=MTPContract(),
    )

    with pytest.raises(
        Qwen38ContractError,
        match=r"row 18.*configured=47.*active=47.*expected=48",
    ):
        install_qwen38_route(
            runtime,
            _config(),
            MODEL_PATH,
            cache_route="control",
            row18_gdn_decay_memo=True,
        )


def test_enabled_row21_rejects_partial_attention_module_coverage(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtplx.qwen38_challenge.configure_qwen38_row21_qk_rms_rope",
        lambda model, *, active: {
            "eligible_modules": 15,
            "active_modules": 15 if active else 0,
        },
    )
    runtime = MTPLXRuntime(
        model=SimpleNamespace(mtp_update_cache=_callable),
        tokenizer=SimpleNamespace(),
        model_path=MODEL_PATH,
        mtp_enabled=True,
        contract=MTPContract(),
    )

    with pytest.raises(
        Qwen38ContractError,
        match=r"row 21.*eligible=15.*active=15.*expected=16",
    ):
        install_qwen38_route(
            runtime,
            _config(),
            MODEL_PATH,
            cache_route="control",
            row21_qk_rms_rope=True,
        )


def test_enabled_dual_norm_route_fails_when_prebound_callable_is_missing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "mtplx.qwen38_challenge._validate_qwen38_dual_norm_install",
        lambda text, *, q8_embedding: None,
    )
    runtime = MTPLXRuntime(
        model=SimpleNamespace(mtp_update_cache=_callable),
        tokenizer=SimpleNamespace(),
        model_path=MODEL_PATH,
        mtp_enabled=True,
        contract=MTPContract(),
    )

    with pytest.raises(Qwen38ContractError, match="MTP input route"):
        install_qwen38_route(
            runtime,
            _config(),
            MODEL_PATH,
            cache_route="control",
            dual_norm=True,
        )


def test_explicit_row20_row61_candidate_remains_constructible(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtplx.qwen38_challenge._validate_qwen38_dual_norm_install",
        lambda text, *, q8_embedding: None,
    )

    runtime, _, kv_only, _, dual_prepare = _route_runtime()
    text = runtime.model.language_model

    route = install_qwen38_route(
        runtime,
        _config(),
        MODEL_PATH,
        cache_route="kv_only_history",
        dual_norm=True,
    )

    assert route.route_id == "kv_only_history+dual_norm"
    assert route.bindings.mtp_cache_append is kv_only
    assert route.kernel_ids == (
        "qwen38_mtp_kv_only_history_ge16384_v1",
        "qwen38_dual_rms_norm_concat_bf16_v1",
    )
    assert text._mtplx_prepare_mtp_inputs is dual_prepare
    assert runtime.qwen38_feature_receipt["r20_kv_only_history"]["installed"]
    assert runtime.qwen38_feature_receipt["dual_norm"] == {"active": 1}


def test_enabled_target_ladder_fails_when_prebound_callable_is_missing() -> None:
    runtime = MTPLXRuntime(
        model=SimpleNamespace(mtp_update_cache=_callable),
        tokenizer=SimpleNamespace(),
        model_path=MODEL_PATH,
        mtp_enabled=True,
        contract=MTPContract(),
    )

    with pytest.raises(Qwen38ContractError, match="target layer route"):
        install_qwen38_route(
            runtime,
            _config(),
            MODEL_PATH,
            cache_route="control",
            row24_eval_ladder=True,
        )


def test_row26_without_row21_installs_prefill_only_and_stock_decode() -> None:
    runtime, text, stock, row26 = _target_phase_runtime()

    route = install_qwen38_route(
        runtime,
        _config(),
        MODEL_PATH,
        cache_route="control",
        row24_eval_ladder=True,
        row26_prefill_ladder_3=True,
    )

    assert route.route_id == "r24_eval_ladder+r26_prefill_ladder_3"
    assert text._mtplx_forward_layers is stock
    assert text._mtplx_qwen38_prefill_forward_layers is row26
    assert runtime.qwen38_feature_receipt["r26_prefill_ladder_3"] == {
        "active": 1,
        "phase_scope": "prefill",
        "decode_route": "stock",
    }

    with _qwen38_prefill_target_route(runtime):
        assert text._mtplx_forward_layers is row26
    assert text._mtplx_forward_layers is stock


def test_performance_profiles_are_prebound_and_selected_at_request_boundary(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def stock_target(*args, **kwargs):
        return ("stock-target", args, kwargs)

    def low_target(*args, **kwargs):
        return ("low-target", args, kwargs)

    def xhigh_target(*args, **kwargs):
        return ("xhigh-target", args, kwargs)

    stock_mtp = object()
    low_mtp = object()
    xhigh_mtp = object()
    text = SimpleNamespace(
        mtp=stock_mtp,
        _mtplx_forward_layers=stock_target,
        _mtplx_qwen38_prefill_forward_layers=None,
        _mtplx_prepare_mtp_inputs=_callable,
        _mtplx_draft_lm_head=object(),
    )
    runtime = MTPLXRuntime(
        model=SimpleNamespace(
            mtp=stock_mtp,
            mtp_update_cache=_callable,
            language_model=text,
        ),
        tokenizer=SimpleNamespace(),
        model_path=MODEL_PATH,
        mtp_enabled=True,
        contract=MTPContract(),
    )

    targets = {
        "stock": (stock_target, stock_mtp),
        "low": (low_target, low_mtp),
        "xhigh": (xhigh_target, xhigh_mtp),
    }

    def fake_install(loaded_runtime, config, model_path, **options):
        del config, model_path
        profile = str(options.pop("test_profile"))
        calls.append(profile)
        target, mtp_block = targets[profile]
        loaded_text = loaded_runtime.model.language_model
        loaded_text._mtplx_forward_layers = target
        loaded_text._mtplx_qwen38_prefill_forward_layers = None
        loaded_text.mtp = mtp_block
        loaded_runtime.model.mtp = mtp_block
        route = build_qwen38_route(
            _config(),
            MODEL_PATH,
            bindings=_bindings(),
            route_id=f"{profile}-installed",
            kernel_ids=(f"{profile}-kernel",) if profile != "stock" else (),
        )
        loaded_runtime.qwen38_route = route
        loaded_runtime.qwen38_feature_receipt = (
            {} if profile == "stock" else {f"{profile}-feature": {"active": 1}}
        )
        return route

    monkeypatch.setattr(qwen38_challenge, "install_qwen38_route", fake_install)

    install_qwen38_performance_profiles(
        runtime,
        _config(),
        MODEL_PATH,
        stock=Qwen38PerformanceProfileConfig(
            requested_route_id="control",
            install_options={"test_profile": "stock"},
            draft_core="stock",
        ),
        low=Qwen38PerformanceProfileConfig(
            requested_route_id="low-requested",
            install_options={"test_profile": "low"},
            draft_core="device",
        ),
        xhigh=Qwen38PerformanceProfileConfig(
            requested_route_id="xhigh-requested",
            install_options={"test_profile": "xhigh"},
            draft_core="stock",
        ),
    )

    assert calls == ["stock", "low", "xhigh"]

    low = select_qwen38_performance_profile(runtime, "low")
    assert calls == ["stock", "low", "xhigh"]
    assert low.profile_id == "low"
    assert low.requested_route_id == "low-requested"
    assert low.route.route_id == "low-installed"
    assert low.draft_core == "device"
    assert text._mtplx_forward_layers is low_target
    assert text.mtp is low_mtp
    assert runtime.model.mtp is low_mtp
    assert runtime.qwen38_feature_receipt == {"low-feature": {"active": 1}}

    xhigh = select_qwen38_performance_profile(runtime, "xhigh")
    assert xhigh.profile_id == "xhigh"
    assert xhigh.requested_route_id == "xhigh-requested"
    assert xhigh.route.route_id == "xhigh-installed"
    assert xhigh.draft_core == "stock"
    assert text._mtplx_forward_layers is xhigh_target
    assert text.mtp is xhigh_mtp

    stock = select_qwen38_performance_profile(runtime, "high")
    assert stock.profile_id == "stock"
    assert stock.requested_route_id == "control"
    assert stock.route.route_id == "stock-installed"
    assert text._mtplx_forward_layers is stock_target
    assert text.mtp is stock_mtp

    receipt = qwen38_route_receipt(runtime.qwen38_route)
    assert receipt["performance_profile"] == "stock"
    assert receipt["requested_route_id"] == "control"
    assert receipt["installed_route_id"] == "stock-installed"
    assert receipt["draft_core"] == "stock"
    assert receipt["mtp_block_identity"] == "bf16"


def test_measured_fixed_profiles_use_the_qualified_bf16_stacks() -> None:
    profiles = qwen38_measured_performance_profile_configs(
        adaptive_policy="none",
        q4_mtp_block=None,
    )

    assert profiles["stock"] == Qwen38PerformanceProfileConfig(
        requested_route_id="control",
        install_options={"cache_route": "control"},
        draft_core="stock",
        installed_route_id="control",
    )
    low = profiles["low"]
    assert low.requested_route_id == QWEN38_LOW_FIXED_ROUTE
    assert low.draft_core == "device"
    assert low.row53_command_buffers is True
    assert low.installed_route_id == qwen38_challenge.QWEN38_LOW_BF16_INSTALLED_ROUTE
    assert low.kernel_ids == qwen38_challenge.QWEN38_LOW_BF16_KERNEL_IDS
    assert low.feature_keys == qwen38_challenge.QWEN38_LOW_BF16_FEATURE_KEYS
    assert low.install_options == {
        "cache_route": "kv_only_history",
        "row10_compact_vocab": True,
        "row21_qk_rms_rope": True,
        "row24_eval_ladder": True,
        "row26_prefill_ladder_3": True,
    }
    xhigh = profiles["xhigh"]
    assert xhigh.requested_route_id == QWEN38_XHIGH_FIXED_ROUTE
    assert xhigh.draft_core == "stock"
    assert xhigh.row53_command_buffers is True
    assert xhigh.installed_route_id == (
        qwen38_challenge.QWEN38_XHIGH_BF16_INSTALLED_ROUTE
    )
    assert xhigh.kernel_ids == qwen38_challenge.QWEN38_XHIGH_BF16_KERNEL_IDS
    assert xhigh.feature_keys == qwen38_challenge.QWEN38_XHIGH_BF16_FEATURE_KEYS
    assert xhigh.install_options == {
        "cache_route": "kv_only_history",
        "row24_eval_ladder": True,
        "row26_prefill_ladder_3": True,
        "row50_wired_residency": True,
    }


def test_measured_adaptive_profiles_select_q4_only_for_low(tmp_path: Path) -> None:
    artifact = tmp_path / "qwen38-r17-q4.safetensors"
    artifact.write_bytes(b"row17")

    profiles = qwen38_measured_performance_profile_configs(
        adaptive_policy="position_ema",
        q4_mtp_block=artifact,
    )

    low = profiles["low"]
    assert low.requested_route_id == QWEN38_LOW_ADAPTIVE_ROUTE
    assert low.draft_core == "device"
    assert low.install_options["mtp_block_variant"] == "r17"
    assert low.install_options["mtp_block_artifact_path"] == artifact.resolve()
    assert low.installed_route_id == qwen38_challenge.QWEN38_LOW_Q4_INSTALLED_ROUTE
    assert low.kernel_ids == (
        "qwen38_row17_q4_g64_mtp_block_v1",
        *qwen38_challenge.QWEN38_LOW_BF16_KERNEL_IDS,
    )
    xhigh = profiles["xhigh"]
    assert xhigh.requested_route_id == QWEN38_XHIGH_ADAPTIVE_ROUTE
    assert xhigh.draft_core == "stock"
    assert "mtp_block_variant" not in xhigh.install_options
    assert "mtp_block_artifact_path" not in xhigh.install_options


def test_measured_adaptive_profiles_require_the_q4_artifact(tmp_path: Path) -> None:
    with pytest.raises(Qwen38ContractError, match="Q4 MTP block artifact"):
        qwen38_measured_performance_profile_configs(
            adaptive_policy="position_ema",
            q4_mtp_block=tmp_path / "missing.safetensors",
        )


def test_measured_profiles_reject_an_installed_route_that_drops_the_stack(
    monkeypatch,
) -> None:
    runtime, *_ = _route_runtime()
    profiles = qwen38_measured_performance_profile_configs(
        adaptive_policy="none",
        q4_mtp_block=None,
    )
    monkeypatch.setattr(
        qwen38_challenge,
        "install_qwen38_route",
        lambda loaded_runtime, config, model_path, **options: build_qwen38_route(
            config,
            model_path,
            bindings=_bindings(),
            route_id="control",
        ),
    )

    with pytest.raises(Qwen38ContractError, match="installed route mismatch"):
        install_qwen38_performance_profiles(
            runtime,
            _config(),
            MODEL_PATH,
            stock=profiles["stock"],
            low=profiles["low"],
            xhigh=profiles["xhigh"],
            environment={
                "MLX_MAX_MB_PER_BUFFER": "512",
                "MLX_MAX_OPS_PER_BUFFER": "50",
            },
        )


def test_profile_install_validates_row53_once_at_construction(monkeypatch) -> None:
    runtime, *_ = _route_runtime()
    profile = Qwen38PerformanceProfileConfig(
        requested_route_id=QWEN38_XHIGH_FIXED_ROUTE,
        install_options={"cache_route": "control"},
        row53_command_buffers=True,
    )
    monkeypatch.setattr(
        qwen38_challenge,
        "install_qwen38_route",
        lambda loaded_runtime, config, model_path, **options: build_qwen38_route(
            config,
            model_path,
            bindings=_bindings(),
            route_id="control",
        ),
    )

    with pytest.raises(Qwen38ContractError, match="command-buffer contract"):
        install_qwen38_performance_profiles(
            runtime,
            _config(),
            MODEL_PATH,
            stock=Qwen38PerformanceProfileConfig("control", {}),
            low=profile,
            xhigh=profile,
            environment={
                "MLX_MAX_MB_PER_BUFFER": "128",
                "MLX_MAX_OPS_PER_BUFFER": "50",
            },
        )

    installed = install_qwen38_performance_profiles(
        runtime,
        _config(),
        MODEL_PATH,
        stock=Qwen38PerformanceProfileConfig("control", {}),
        low=profile,
        xhigh=profile,
        environment={
            "MLX_MAX_MB_PER_BUFFER": "512",
            "MLX_MAX_OPS_PER_BUFFER": "50",
        },
    )
    assert installed["low"].feature_receipt["r53_command_buffers"] == {
        "installed": True,
        "active": True,
        "max_mb_per_buffer": 512,
        "max_ops_per_buffer": 50,
        "process_latched": True,
    }


def test_profile_selection_does_not_leak_xhigh_wired_residency_into_low(
    monkeypatch,
) -> None:
    runtime, *_ = _route_runtime()
    wired_calls: list[int] = []

    def install_route(loaded_runtime, config, model_path, **options):
        if options.get("row50_wired_residency"):
            loaded_runtime._qwen38_row50_wired_state = {
                "installed": True,
                "baseline_limit_bytes": 17,
                "target_limit_bytes": 29,
            }
            loaded_runtime._qwen38_row50_set_wired_limit = wired_calls.append
        loaded_runtime.qwen38_feature_receipt = {}
        return build_qwen38_route(
            config,
            model_path,
            bindings=_bindings(),
            route_id=("xhigh-installed" if options.get("row50_wired_residency") else "installed"),
        )

    monkeypatch.setattr(qwen38_challenge, "install_qwen38_route", install_route)
    install_qwen38_performance_profiles(
        runtime,
        _config(),
        MODEL_PATH,
        stock=Qwen38PerformanceProfileConfig("control", {}),
        low=Qwen38PerformanceProfileConfig("low", {}),
        xhigh=Qwen38PerformanceProfileConfig(
            "xhigh",
            {"row50_wired_residency": True},
        ),
    )

    # Construction leaves the explicit stock profile selected.
    assert wired_calls == [17]
    select_qwen38_performance_profile(runtime, "xhigh")
    assert wired_calls[-1] == 29
    select_qwen38_performance_profile(runtime, "low")
    assert wired_calls[-1] == 17


def test_performance_profile_install_rejects_missing_bound_route(monkeypatch) -> None:
    runtime, *_ = _route_runtime()
    monkeypatch.setattr(
        qwen38_challenge,
        "install_qwen38_route",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(Qwen38ContractError, match="stock performance profile"):
        install_qwen38_performance_profiles(
            runtime,
            _config(),
            MODEL_PATH,
            stock=Qwen38PerformanceProfileConfig("control", {}),
            low=Qwen38PerformanceProfileConfig("low", {}),
            xhigh=Qwen38PerformanceProfileConfig("xhigh", {}),
        )


def test_row50_wired_residency_restores_the_control_limit() -> None:
    calls: list[tuple[str, int | None]] = []

    class FakeMX:
        def device_info(self):
            return {
                "memory_size": 128 * 2**30,
                "max_recommended_working_set_size": 100 * 2**30,
            }

        def clear_cache(self):
            calls.append(("clear", None))

        def get_active_memory(self):
            return 25 * 2**30

        def set_wired_limit(self, value):
            calls.append(("wired", int(value)))
            return 0

    runtime = SimpleNamespace()
    candidate = configure_qwen38_row50_wired_residency(
        runtime, active=True, mx_module=FakeMX()
    )
    control = configure_qwen38_row50_wired_residency(
        runtime, active=False, mx_module=FakeMX()
    )

    assert candidate["target_limit_bytes"] == 25 * 2**30 + 64 * 2**20
    assert control["restored_limit_bytes"] == 0
    assert calls == [
        ("clear", None),
        ("wired", 25 * 2**30 + 64 * 2**20),
        ("wired", 0),
    ]
