"""Runtime construction coverage for the canonical DeepSeek-V4 o-LoRA route."""

from __future__ import annotations

from mtplx import runtime
from mtplx.models import deepseek_v4 as D
from tests.test_deepseek_v4_o_lora import (
    _FakeCachedBodyRoute,
    _FakeDenseMTPRoute,
    _FakeGatherBodyRoute,
    _FakeGatherWideM4BodyRoute,
    _canonical_route_model,
    _patch_canonical_route_types,
)


def test_runtime_load_installs_canonical_mixed_o_lora_route(monkeypatch, tmp_path):
    """The real runtime load flow prebinds Q4 body gathers plus dense MTP stock."""

    _patch_canonical_route_types(monkeypatch)
    monkeypatch.setattr(D, "_DirectCachedOLora", _FakeCachedBodyRoute)
    monkeypatch.setattr(D, "_DirectGatherOLora", _FakeGatherBodyRoute)
    monkeypatch.setattr(D, "_DirectGatherOLoraWideM4", _FakeGatherWideM4BodyRoute)
    monkeypatch.setattr(D, "_DirectDenseMTPOLora", _FakeDenseMTPRoute)
    monkeypatch.setattr(D, "_validate_gather_qmm_wide_m4_body_routes", lambda _body: None)
    monkeypatch.setenv("MTPLX_DSV4_O_LORA", "gather_qmm")
    model = _canonical_route_model()
    config = {"model_type": "deepseek_v4", "num_nextn_predict_layers": 1}
    monkeypatch.setattr(runtime, "load_config", lambda _path: config)
    monkeypatch.setattr(runtime, "_load_base_model", lambda *_args: (model, object()))
    monkeypatch.setattr(runtime, "_load_runtime_metadata", lambda _path: {})
    monkeypatch.setattr(runtime, "mtp_weights_present_on_disk", lambda *_args: False)
    monkeypatch.setattr(runtime, "validate_mtp_support", lambda _model: True)

    monkeypatch.setattr(D, "configure_deepseek_v4_moe_tail", lambda *_args: None)
    monkeypatch.setattr(D, "is_deepseek_v4_mtp_config", lambda _config: True)
    monkeypatch.setattr(D, "inject_deepseek_v4_mtp_support", lambda *_args: True)

    import mtplx.a3b_compiled_target_prefix as target_prefix
    import mtplx.a3b_whole_moe as whole_moe
    import mtplx.attention_split as attention_split
    import mtplx.gdn_capture as gdn_capture
    import mtplx.kernel_selfcheck as kernel_selfcheck
    import mtplx.native_mlp as native_mlp
    import mtplx.nax_verify as nax_verify
    import mtplx.qwen_row_owned_router as row_owned

    monkeypatch.setattr(attention_split, "configure_split_full_attention", lambda *_: None)
    monkeypatch.setattr(native_mlp, "configure_native_mlp", lambda *_: None)
    monkeypatch.setattr(nax_verify, "nax_env_enabled", lambda: False)
    monkeypatch.setattr(whole_moe, "prepare_a3b_whole_moe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(row_owned, "prepare_qwen_row_owned_routers", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(gdn_capture, "prepare_a3b_gdn_postconv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(kernel_selfcheck, "maybe_run_model_selfcheck", lambda *_: None)
    monkeypatch.setattr(target_prefix, "prepare_a3b_compiled_target_prefix", lambda *_args, **_kwargs: None)

    loaded = runtime.load(tmp_path, mtp=True)

    assert loaded.model is model
    assert loaded.deepseek_v4_o_lora_report["body_direct"] == 43
    assert loaded.deepseek_v4_o_lora_report["mtp_stock"] == 1
    assert all(
        isinstance(box.attn._o_lora_impl, _FakeGatherWideM4BodyRoute)
        for box in model.layers
    )
    assert isinstance(model.mtp_blocks[0].attn._o_lora_impl, _FakeDenseMTPRoute)
    assert not any(
        isinstance(box.attn._o_lora_impl, _FakeCachedBodyRoute) for box in model.layers
    )


def test_runtime_load_preserves_ar_only_degrade_without_canonical_mixed_route(
    monkeypatch, tmp_path
):
    model = _canonical_route_model(mtp_count=0)
    config = {"model_type": "deepseek_v4", "num_nextn_predict_layers": 1}
    calls = []
    monkeypatch.setenv("MTPLX_DSV4_O_LORA", "gather_qmm")
    monkeypatch.setattr(runtime, "load_config", lambda _path: config)
    monkeypatch.setattr(runtime, "_load_base_model", lambda *_args: (model, object()))
    monkeypatch.setattr(runtime, "_load_runtime_metadata", lambda _path: {})
    monkeypatch.setattr(runtime, "mtp_weights_present_on_disk", lambda *_args: False)

    monkeypatch.setattr(D, "configure_deepseek_v4_moe_tail", lambda *_args: None)
    monkeypatch.setattr(D, "is_deepseek_v4_mtp_config", lambda _config: True)
    monkeypatch.setattr(D, "inject_deepseek_v4_mtp_support", lambda *_args: False)
    monkeypatch.setattr(
        D,
        "install_deepseek_v4_o_lora_routes",
        lambda _model, **kwargs: calls.append(kwargs)
        or {"mode": kwargs.get("mode"), "canonical_mixed_route": False},
    )

    import mtplx.a3b_compiled_target_prefix as target_prefix
    import mtplx.a3b_whole_moe as whole_moe
    import mtplx.attention_split as attention_split
    import mtplx.gdn_capture as gdn_capture
    import mtplx.kernel_selfcheck as kernel_selfcheck
    import mtplx.native_mlp as native_mlp
    import mtplx.nax_verify as nax_verify
    import mtplx.qwen_row_owned_router as row_owned

    monkeypatch.setattr(attention_split, "configure_split_full_attention", lambda *_: None)
    monkeypatch.setattr(native_mlp, "configure_native_mlp", lambda *_: None)
    monkeypatch.setattr(nax_verify, "nax_env_enabled", lambda: False)
    monkeypatch.setattr(whole_moe, "prepare_a3b_whole_moe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(row_owned, "prepare_qwen_row_owned_routers", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(gdn_capture, "prepare_a3b_gdn_postconv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(kernel_selfcheck, "maybe_run_model_selfcheck", lambda *_: None)
    monkeypatch.setattr(target_prefix, "prepare_a3b_compiled_target_prefix", lambda *_args, **_kwargs: None)

    loaded = runtime.load(tmp_path, mtp=True)

    assert loaded.mtp_enabled is False
    assert calls == [{"mode": "cached", "canonical_mixed_route": False}]
    assert loaded.deepseek_v4_o_lora_report == {
        "mode": "cached",
        "canonical_mixed_route": False,
    }
