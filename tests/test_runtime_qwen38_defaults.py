"""Hermetic runtime-load coverage for the Qwen 3.8 default route."""

from __future__ import annotations

from types import SimpleNamespace

from mtplx import qwen38_challenge, runtime
from mtplx.qwen38_challenge import (
    Qwen38ModelContract,
    Qwen38RouteBindings,
    Qwen38RouteSpec,
    qwen38_route_receipt,
)


def test_runtime_load_passes_only_control_qwen38_defaults(monkeypatch, tmp_path):
    """The production load flow forwards the quarantined construction route."""

    model_path = tmp_path / "Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed"
    config = {
        "model_type": "qwen3_5",
        "text_config": {"mtp_num_hidden_layers": 1},
    }

    def stock_append(*args, **kwargs):
        return args, kwargs

    model = SimpleNamespace(mtp_update_cache=stock_append)
    tokenizer = SimpleNamespace()

    import mtplx.a3b_compiled_target_prefix as target_prefix
    import mtplx.a3b_whole_moe as whole_moe
    import mtplx.attention_split as attention_split
    import mtplx.gdn_capture as gdn_capture
    import mtplx.gemma4_pair as gemma4_pair
    import mtplx.kernel_selfcheck as kernel_selfcheck
    import mtplx.moe_packed_projections as packed_projections
    import mtplx.native_mlp as native_mlp
    import mtplx.nax_verify as nax_verify
    import mtplx.qwen_row_owned_router as row_owned
    from mtplx.kernels import gdn_blocked_prefill

    monkeypatch.setattr(gemma4_pair, "resolve_gemma4_pair_paths", lambda _path: None)
    monkeypatch.setattr(runtime, "load_config", lambda _path: config)
    monkeypatch.setattr(runtime, "_load_base_model", lambda *_args: (model, tokenizer))
    monkeypatch.setattr(runtime, "_load_runtime_metadata", lambda _path: {})
    monkeypatch.setattr(runtime, "inject_mtp_support", lambda *_args: True)
    monkeypatch.setattr(runtime, "validate_mtp_support", lambda _model: True)
    monkeypatch.setattr(
        runtime,
        "_install_architectures_declared_module_alias",
        lambda _config: None,
    )
    monkeypatch.setattr(attention_split, "configure_split_full_attention", lambda *_: None)
    monkeypatch.setattr(native_mlp, "configure_native_mlp", lambda *_: None)
    monkeypatch.setattr(packed_projections, "moe_pack_gate_up_enabled", lambda: False)
    monkeypatch.setattr(nax_verify, "nax_env_enabled", lambda: False)
    monkeypatch.setattr(gdn_blocked_prefill, "blocked_prefill_env_enabled", lambda: False)
    monkeypatch.setattr(
        whole_moe,
        "prepare_a3b_whole_moe",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        row_owned,
        "prepare_qwen_row_owned_routers",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        gdn_capture,
        "prepare_a3b_gdn_postconv",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(kernel_selfcheck, "maybe_run_model_selfcheck", lambda *_: None)
    monkeypatch.setattr(
        target_prefix,
        "prepare_a3b_compiled_target_prefix",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.delenv("MTPLX_PROJ_QUANT", raising=False)
    monkeypatch.delenv("MTPLX_PROJ_REQUANT", raising=False)

    calls = []
    control_contract = Qwen38ModelContract(
        contract_id="qwen38-runtime-default-test",
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=64,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        vocab_size=248320,
        dtype="bfloat16",
        trunk_bits=4,
        trunk_group_size=32,
        trunk_mode="affine",
        packing="mlx_affine_u32_le",
    )

    def spy_install(loaded_runtime, received_config, received_path, **options):
        calls.append(
            {
                "runtime": loaded_runtime,
                "config": received_config,
                "path": received_path,
                "options": options,
            }
        )
        route = Qwen38RouteSpec(
            route_id="control",
            contract=control_contract,
            bindings=Qwen38RouteBindings(mtp_cache_append=stock_append),
            history_route_id="stock_history",
            kernel_ids=(),
            selfcheck_status="control",
            selfcheck_passed=True,
        )
        loaded_runtime.qwen38_route = route
        loaded_runtime.qwen38_feature_receipt = {}
        return route

    monkeypatch.setattr(qwen38_challenge, "install_qwen38_route", spy_install)

    loaded = runtime.load(model_path, mtp=True)

    assert len(calls) == 1
    assert calls[0] == {
        "runtime": loaded,
        "config": config,
        "path": model_path,
        "options": {"cache_route": "control", "dual_norm": False},
    }
    assert loaded.model is model
    assert loaded.qwen38_route.route_id == "control"
    assert loaded.qwen38_route.history_route_id == "stock_history"
    assert loaded.qwen38_route.kernel_ids == ()
    assert loaded.qwen38_feature_receipt == {}
    assert qwen38_route_receipt(loaded.qwen38_route) == {
        "route_id": "control",
        "fingerprint": loaded.qwen38_route.fingerprint,
        "contract_id": "qwen38-runtime-default-test",
        "kernel_ids": [],
        "history_route_id": "stock_history",
        "min_context_tokens": 0,
        "policy_id": "current_mtplx",
        "selfcheck": {"passed": True, "status": "control"},
    }
