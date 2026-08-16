"""Runtime construction coverage for the pinned DeepSeek-V4-0731 backend."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from mtplx import runtime
from mtplx.deepseek_v4_dspark_generation import DeepseekV4DSparkBackend
from mtplx.models import deepseek_v4 as D


_DSPARK_CONFIG = {
    "model_type": "deepseek_v4",
    "dspark_block_size": 5,
    "dspark_noise_token_id": 128799,
    "dspark_target_layer_ids": [40, 41, 42],
    "dspark_markov_rank": 256,
}


class _DSpark:
    def __init__(self) -> None:
        self.stages = tuple(SimpleNamespace(attn=_RouteAttention()) for _ in range(3))

    def make_cache(self):
        return []

    def prefill(self, *_args):
        return None

    def forward(self, *_args, **_kwargs):
        return None

    def commit_main(self, *_args, **_kwargs):
        return None


class _Model:
    def __init__(self, *, with_dspark: bool = True) -> None:
        self._dspark = _DSpark() if with_dspark else None
        self.mtp = [] if self._dspark is None else self._dspark.stages
        self.model = SimpleNamespace(embed_tokens=lambda value: value)
        self.lm_head = lambda value: value
        self.layers = [SimpleNamespace(attn=_RouteAttention()) for _ in range(43)]

    @property
    def mtp_blocks(self):
        return list(self.mtp)

    def __call__(self, value, **_kwargs):
        return value


class _RouteAttention:
    def __init__(self) -> None:
        self.installed_modes = []
        self.o_lora_mode = "cached"
        self._o_lora_impl = object()

    def install_o_lora_route(self, mode):
        self.installed_modes.append(str(mode))
        self.o_lora_mode = str(mode)
        self._o_lora_impl = object()
        return {"mode": str(mode), "direct": str(mode) == "gather_qmm"}


class _AdversarialRestoreAttention:
    def __init__(
        self,
        label,
        events,
        *,
        fail_mode_restore=False,
        fail_implementation_restore=False,
    ) -> None:
        self.label = label
        self.events = events
        self.fail_mode_restore = fail_mode_restore
        self.fail_implementation_restore = fail_implementation_restore
        self._mode = "cached"
        self.original_implementation = object()
        self._implementation = self.original_implementation

    @property
    def o_lora_mode(self):
        return self._mode

    @o_lora_mode.setter
    def o_lora_mode(self, value):
        restoring = value == "cached"
        self.events.append((self.label, "mode", "restore" if restoring else "install"))
        if restoring and self.fail_mode_restore:
            raise RuntimeError(f"{self.label} mode restore failed")
        self._mode = value

    @property
    def _o_lora_impl(self):
        return self._implementation

    @_o_lora_impl.setter
    def _o_lora_impl(self, value):
        restoring = value is self.original_implementation
        self.events.append(
            (self.label, "implementation", "restore" if restoring else "install")
        )
        if restoring and self.fail_implementation_restore:
            raise RuntimeError(f"{self.label} implementation restore failed")
        self._implementation = value


def _patch_load_dependencies(
    monkeypatch,
    tmp_path,
    model,
    o_lora_calls,
    *,
    mtp=True,
    real_o_lora_installer=False,
    deepseek_v4_0731_k2=False,
):
    monkeypatch.setattr(runtime, "load_config", lambda _path: dict(_DSPARK_CONFIG))
    monkeypatch.setattr(runtime, "_load_base_model", lambda *_args: (model, object()))
    monkeypatch.setattr(runtime, "_load_runtime_metadata", lambda _path: {})
    monkeypatch.setattr(runtime, "mtp_weights_present_on_disk", lambda *_args: True)
    monkeypatch.setattr(
        runtime,
        "validate_mtp_support",
        lambda _model: (_ for _ in ()).throw(
            AssertionError("DSpark must not use legacy MTP validation")
        ),
    )

    monkeypatch.setattr(D, "configure_deepseek_v4_moe_tail", lambda *_args: None)
    monkeypatch.setattr(D, "_o_lora_mode_from_env", lambda: "gather_qmm")
    if not real_o_lora_installer:
        monkeypatch.setattr(
            D,
            "install_deepseek_v4_o_lora_routes",
            lambda _model, **kwargs: o_lora_calls.append(kwargs) or dict(kwargs),
        )

    import mtplx.a3b_compiled_target_prefix as target_prefix
    import mtplx.a3b_whole_moe as whole_moe
    import mtplx.attention_split as attention_split
    import mtplx.gdn_capture as gdn_capture
    import mtplx.kernel_selfcheck as kernel_selfcheck
    import mtplx.native_mlp as native_mlp
    import mtplx.nax_verify as nax_verify
    import mtplx.qwen_row_owned_router as row_owned

    monkeypatch.setattr(
        attention_split, "configure_split_full_attention", lambda *_: None
    )
    monkeypatch.setattr(native_mlp, "configure_native_mlp", lambda *_: None)
    monkeypatch.setattr(nax_verify, "nax_env_enabled", lambda: False)
    monkeypatch.setattr(
        whole_moe, "prepare_a3b_whole_moe", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        row_owned, "prepare_qwen_row_owned_routers", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        gdn_capture, "prepare_a3b_gdn_postconv", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(kernel_selfcheck, "maybe_run_model_selfcheck", lambda *_: None)
    monkeypatch.setattr(
        target_prefix,
        "prepare_a3b_compiled_target_prefix",
        lambda *_args, **_kwargs: None,
    )
    return runtime.load(
        tmp_path,
        mtp=mtp,
        deepseek_v4_0731_k2=deepseek_v4_0731_k2,
    )


def test_runtime_load_publishes_the_construction_bound_dspark_backend(
    monkeypatch, tmp_path
):
    model = _Model()
    o_lora_calls = []
    monkeypatch.setenv("MTPLX_DECODE_TRACE_JSONL", str(tmp_path / "trace.jsonl"))

    loaded = _patch_load_dependencies(monkeypatch, tmp_path, model, o_lora_calls)

    assert loaded.mtp_enabled is True
    assert loaded.deepseek_v4_dspark_enabled is True
    assert isinstance(loaded.block_speculative_backend, DeepseekV4DSparkBackend)
    assert loaded.block_speculative_backend.dspark is model._dspark
    assert loaded.block_speculative_decode_trace_requested is True
    assert o_lora_calls == [{"mode": "gather_qmm", "canonical_mixed_route": False}]


def test_runtime_dspark_gather_uses_the_real_per_module_o_lora_installer(
    monkeypatch, tmp_path
):
    model = _Model()

    loaded = _patch_load_dependencies(
        monkeypatch,
        tmp_path,
        model,
        [],
        real_o_lora_installer=True,
    )

    assert loaded.deepseek_v4_o_lora_report["module_count"] == 46
    assert loaded.deepseek_v4_o_lora_report["trunk_module_count"] == 43
    assert loaded.deepseek_v4_o_lora_report["mtp_module_count"] == 3
    assert all(layer.attn.installed_modes == ["gather_qmm"] for layer in model.layers)
    assert all(
        stage.attn.installed_modes == ["gather_qmm"] for stage in model._dspark.stages
    )


def test_runtime_load_fails_before_publication_when_dspark_owner_is_missing(
    monkeypatch, tmp_path
):
    model = _Model(with_dspark=False)
    o_lora_calls = []

    with pytest.raises(ValueError, match="DSpark backend cannot bind"):
        _patch_load_dependencies(monkeypatch, tmp_path, model, o_lora_calls)

    assert o_lora_calls == []


def test_runtime_load_does_not_publish_dspark_when_mtp_is_disabled(
    monkeypatch, tmp_path
):
    model = _Model()
    o_lora_calls = []

    loaded = _patch_load_dependencies(
        monkeypatch, tmp_path, model, o_lora_calls, mtp=False
    )

    assert loaded.mtp_enabled is False
    assert loaded.deepseek_v4_dspark_enabled is False
    assert loaded.block_speculative_backend is None
    assert o_lora_calls == [{"mode": "cached", "canonical_mixed_route": False}]


def test_k2_option_rejects_nonexact_artifact_before_model_construction(
    monkeypatch, tmp_path
):
    constructed = []
    monkeypatch.setattr(
        runtime,
        "load_config",
        lambda _path: {"model_type": "deepseek_v4", **_DSPARK_CONFIG},
    )
    monkeypatch.setattr(
        runtime,
        "_load_base_model",
        lambda *_args: constructed.append(True) or (_Model(), object()),
    )

    with pytest.raises(ValueError, match="full DSpark contract failed"):
        runtime.load(tmp_path, deepseek_v4_0731_k2=True)

    assert constructed == []


def test_k2_option_requires_mtp_before_model_construction(monkeypatch, tmp_path):
    constructed = []
    monkeypatch.setattr(
        runtime,
        "_load_base_model",
        lambda *_args: constructed.append(True) or (_Model(), object()),
    )

    with pytest.raises(ValueError, match="requires mtp=True"):
        runtime.load(tmp_path, mtp=False, deepseek_v4_0731_k2=True)

    assert constructed == []


def test_k2_sinkhorn_selector_is_scoped_to_model_construction(monkeypatch):
    events = []
    monkeypatch.setattr(D, "_SINKHORN_KERNEL", False)
    monkeypatch.setattr(D.mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(D.mx, "default_device", lambda: D.mx.gpu)
    monkeypatch.setattr(
        D,
        "_sinkhorn_metal_kernel",
        lambda hc, iters, eps: events.append((hc, iters, eps)),
    )

    outside, _ = D._install_sinkhorn_normaliser(4, 20, 1e-6)
    with D.deepseek_v4_0731_k2_construction():
        inside, _ = D._install_sinkhorn_normaliser(4, 20, 1e-6)
    restored, _ = D._install_sinkhorn_normaliser(4, 20, 1e-6)

    assert (outside, inside, restored) == (False, True, False)
    assert events == [(4, 20, 1e-6)]


class _Prepared:
    def __init__(self, label, events, *, fail_restore=False):
        self.label = label
        self.events = events
        self.fail_restore = fail_restore
        self.receipt = {"candidate": label}

    def publish(self):
        self.events.append(f"{self.label}.publish")

    def restore(self):
        self.events.append(f"{self.label}.restore")
        if self.fail_restore:
            raise RuntimeError(f"{self.label} restore failed")


def _patch_k2_preparers(monkeypatch, tmp_path, events, *, fail_restore=None):
    import mtplx.deepseek_v4_0731_dspark_ffn as dspark_ffn
    import mtplx.deepseek_v4_0731_full_install as full_install
    import mtplx.deepseek_v4_0731_m3_wob as wob
    import mtplx.deepseek_v4_0731_m3_wqb_qnorm_rope as wqb

    def prepare_wqb_qhead_m3(_layers, *, exact_selfcheck):
        events.append(("wqb.prepare", callable(exact_selfcheck)))

    def prepare_wob_m3(_layers, *, exact_selfcheck):
        events.append(("wob.prepare", callable(exact_selfcheck)))

    monkeypatch.setattr(wqb, "prepare_wqb_qhead_m3", prepare_wqb_qhead_m3)
    monkeypatch.setattr(wob, "prepare_wob_m3", prepare_wob_m3)

    def prepare_target(model, config, path, **kwargs):
        events.append(
            (
                "target.prepare",
                kwargs["prepare_wqb_qhead"].__name__,
                kwargs["prepare_wob"].__name__,
            )
        )
        kwargs["prepare_wqb_qhead"](
            (),
            exact_selfcheck=lambda *_args: True,
        )
        kwargs["prepare_wob"](
            (),
            exact_selfcheck=lambda *_args: True,
        )
        return _Prepared(
            "target",
            events,
            fail_restore=fail_restore == "target",
        )

    monkeypatch.setattr(
        full_install,
        "validate_full_0731_dspark_artifact",
        lambda path, config: events.append("artifact.validate") or object(),
    )
    monkeypatch.setattr(
        full_install,
        "prepare_full_0731_dspark_compiled_tail_q2_pair",
        prepare_target,
    )
    monkeypatch.setattr(
        dspark_ffn,
        "prepare_dspark_q3_packed_gate_up_m5",
        lambda model: (
            events.append("ffn.prepare")
            or _Prepared("ffn", events, fail_restore=fail_restore == "ffn")
        ),
    )

    @contextmanager
    def selected_sinkhorn():
        events.append("sinkhorn.enter")
        try:
            yield
        finally:
            events.append("sinkhorn.exit")

    monkeypatch.setattr(D, "deepseek_v4_0731_k2_construction", selected_sinkhorn)


def test_k2_option_publishes_one_construction_transaction(monkeypatch, tmp_path):
    events = []
    model = _Model()
    _patch_k2_preparers(monkeypatch, tmp_path, events)
    monkeypatch.setattr(
        D,
        "install_deepseek_v4_o_lora_routes",
        lambda _model, **kwargs: events.append(("o_lora", kwargs)) or dict(kwargs),
    )
    monkeypatch.setattr(D, "_o_lora_mode_from_env", lambda: "cached")

    loaded = _patch_load_dependencies(
        monkeypatch,
        tmp_path,
        model,
        [],
        real_o_lora_installer=True,
        deepseek_v4_0731_k2=True,
    )

    assert events == [
        "artifact.validate",
        "sinkhorn.enter",
        "sinkhorn.exit",
        (
            "o_lora",
            {"mode": "gather_qmm", "canonical_mixed_route": False},
        ),
        ("target.prepare", "prepare_wqb_qhead_m3", "prepare_wob_m3"),
        ("wqb.prepare", True),
        ("wob.prepare", True),
        "ffn.prepare",
        "target.publish",
        "ffn.publish",
    ]
    assert loaded.deepseek_v4_0731_k2_receipt == {
        "target_protocol": "primary_plus_two_drafts_physical_m3",
        "selected_depth": 2,
        "exact_vs_serial_greedy": False,
        "target": {"candidate": "target"},
        "dspark_ffn": {"candidate": "ffn"},
    }
    assert loaded.block_speculative_backend.dspark is model._dspark


def test_k2_option_restores_both_staged_stacks_when_backend_binding_fails(
    monkeypatch, tmp_path
):
    events = []
    model = _Model()
    _patch_k2_preparers(monkeypatch, tmp_path, events)
    monkeypatch.setattr(
        DeepseekV4DSparkBackend,
        "bind",
        lambda _model, **_kwargs: (_ for _ in ()).throw(RuntimeError("bind failed")),
    )

    with pytest.raises(RuntimeError, match="bind failed"):
        _patch_load_dependencies(
            monkeypatch,
            tmp_path,
            model,
            [],
            deepseek_v4_0731_k2=True,
        )

    assert events[-4:] == [
        "target.publish",
        "ffn.publish",
        "ffn.restore",
        "target.restore",
    ]


def test_k2_option_restores_every_o_lora_owner_when_preparation_fails(
    monkeypatch, tmp_path
):
    events = []
    model = _Model()
    _patch_k2_preparers(monkeypatch, tmp_path, events)
    attentions = tuple(layer.attn for layer in model.layers) + tuple(
        stage.attn for stage in model._dspark.stages
    )
    originals = tuple(
        (attention.o_lora_mode, attention._o_lora_impl) for attention in attentions
    )

    def install_gather(_model, **kwargs):
        assert kwargs == {"mode": "gather_qmm", "canonical_mixed_route": False}
        for attention in attentions:
            attention.o_lora_mode = "gather_qmm"
            attention._o_lora_impl = object()
        return {"mode": "gather_qmm"}

    monkeypatch.setattr(D, "install_deepseek_v4_o_lora_routes", install_gather)
    import mtplx.deepseek_v4_0731_full_install as full_install

    monkeypatch.setattr(
        full_install,
        "prepare_full_0731_dspark_compiled_tail_q2_pair",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("target prepare failed")
        ),
    )

    with pytest.raises(RuntimeError, match="target prepare failed"):
        _patch_load_dependencies(
            monkeypatch,
            tmp_path,
            model,
            [],
            real_o_lora_installer=True,
            deepseek_v4_0731_k2=True,
        )

    assert (
        tuple(
            (attention.o_lora_mode, attention._o_lora_impl) for attention in attentions
        )
        == originals
    )


def test_k2_rollback_attempts_target_and_o_lora_after_ffn_restore_failure(
    monkeypatch, tmp_path
):
    events = []
    model = _Model()
    attentions = tuple(layer.attn for layer in model.layers) + tuple(
        stage.attn for stage in model._dspark.stages
    )
    originals = tuple(
        (attention.o_lora_mode, attention._o_lora_impl) for attention in attentions
    )
    _patch_k2_preparers(monkeypatch, tmp_path, events, fail_restore="ffn")
    monkeypatch.setattr(
        DeepseekV4DSparkBackend,
        "bind",
        lambda _model, **_kwargs: (_ for _ in ()).throw(RuntimeError("bind failed")),
    )

    with pytest.raises(ExceptionGroup) as caught:
        _patch_load_dependencies(
            monkeypatch,
            tmp_path,
            model,
            [],
            deepseek_v4_0731_k2=True,
        )

    assert events[-4:] == [
        "target.publish",
        "ffn.publish",
        "ffn.restore",
        "target.restore",
    ]
    assert [str(error) for error in caught.value.exceptions] == [
        "bind failed",
        "ffn restore failed",
    ]
    assert (
        tuple(
            (attention.o_lora_mode, attention._o_lora_impl) for attention in attentions
        )
        == originals
    )


def test_k2_rollback_attempts_both_properties_and_all_o_lora_owners(
    monkeypatch, tmp_path
):
    events = []
    model = _Model()
    first = _AdversarialRestoreAttention(
        "first",
        events,
        fail_mode_restore=True,
        fail_implementation_restore=True,
    )
    last = _AdversarialRestoreAttention("last", events)
    model.layers[0].attn = first
    model._dspark.stages[-1].attn = last
    _patch_k2_preparers(monkeypatch, tmp_path, events)

    attentions = tuple(layer.attn for layer in model.layers) + tuple(
        stage.attn for stage in model._dspark.stages
    )

    def install_gather(_model, **kwargs):
        assert kwargs == {"mode": "gather_qmm", "canonical_mixed_route": False}
        for attention in attentions:
            attention.o_lora_mode = "gather_qmm"
            attention._o_lora_impl = object()
        return {"mode": "gather_qmm"}

    monkeypatch.setattr(D, "install_deepseek_v4_o_lora_routes", install_gather)
    monkeypatch.setattr(
        DeepseekV4DSparkBackend,
        "bind",
        lambda _model, **_kwargs: (_ for _ in ()).throw(RuntimeError("bind failed")),
    )

    with pytest.raises(ExceptionGroup) as caught:
        _patch_load_dependencies(
            monkeypatch,
            tmp_path,
            model,
            [],
            real_o_lora_installer=True,
            deepseek_v4_0731_k2=True,
        )

    assert [str(error) for error in caught.value.exceptions] == [
        "bind failed",
        "first mode restore failed",
        "first implementation restore failed",
    ]
    assert ("first", "mode", "restore") in events
    assert ("first", "implementation", "restore") in events
    assert ("last", "mode", "restore") in events
    assert ("last", "implementation", "restore") in events
    assert last.o_lora_mode == "cached"
    assert last._o_lora_impl is last.original_implementation
