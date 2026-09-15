import mlx.core as mx
import pytest
from mtplx.models.qwen4_exp import ModelArgs, Model


def _make_dummy_model(*, with_ple: bool = False):
    ple = {
        "ple_layer_ids": [1],
        "ple_embed_dim": 256,
        "ngram_sidecar": True,
    } if with_ple else {}
    args = ModelArgs(
        model_type="qwen4_exp",
        text_config={
            "hidden_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "vocab_size": 1000,
            "layer_types": ["linear_attention", "linear_attention"],
            **ple,
        },
    )
    return Model(args)


def test_qwen4_exp_compile_flag_support(monkeypatch):
    """Verify that MTPLX_QWEN4EXP_COMPILE flag enables the compiled GDN decode path."""
    monkeypatch.setenv("MTPLX_QWEN4EXP_COMPILE", "1")
    monkeypatch.delenv("MTPLX_COMPILED_GDN", raising=False)
    model = _make_dummy_model()
    text_model = model.language_model.model
    assert text_model._gdn_compiled_env is True


def test_qwen4_exp_compile_kill_switch_precedence(monkeypatch):
    """Verify that an explicit 0 on either flag takes precedence as a kill switch."""
    # When MTPLX_COMPILED_GDN=0, compilation must be disabled even if MTPLX_QWEN4EXP_COMPILE=1
    monkeypatch.setenv("MTPLX_COMPILED_GDN", "0")
    monkeypatch.setenv("MTPLX_QWEN4EXP_COMPILE", "1")
    model = _make_dummy_model()
    text_model = model.language_model.model
    assert text_model._gdn_compiled_env is False
    assert text_model._gdn_compile_explicit_off is True

    # When MTPLX_QWEN4EXP_COMPILE=0, compilation must be disabled even if MTPLX_COMPILED_GDN=1
    monkeypatch.setenv("MTPLX_COMPILED_GDN", "1")
    monkeypatch.setenv("MTPLX_QWEN4EXP_COMPILE", "0")
    model2 = _make_dummy_model()
    text_model2 = model2.language_model.model
    assert text_model2._gdn_compiled_env is False
    assert text_model2._gdn_compile_explicit_off is True


def test_qwen4_exp_compile_kill_switch_overrides_pipeline_lane(monkeypatch):
    """Verify that explicit compile-off overrides set_ar_pipeline_mode."""
    monkeypatch.setenv("MTPLX_COMPILED_GDN", "0")
    model = _make_dummy_model()
    model.set_ar_pipeline_mode(True)
    text_model = model.language_model.model
    assert text_model._gdn_compiled_lane is False


def test_streamed_pipeline_route_arms_flushes_and_disarms_without_resident_table():
    model = _make_dummy_model(with_ple=True)
    routes = []

    class Route:
        pending = False

        def set_active(self, value):
            routes.append(("active", bool(value)))

        def flush(self):
            routes.append(("flush",))

        def discard(self):
            routes.append(("discard",))

        @staticmethod
        def make_token():
            return object()

    bound = 0
    bindings = []
    for layer in model.layers:
        if "ple" in layer:
            route = Route()
            layer.ple.ple_embedding._streamed_ar_ple = route
            bindings.append((layer.ple.ple_embedding, route))
            bound += 1
    assert bound > 0

    assert model.set_ar_pipeline_mode(True) is True
    assert model.make_ar_pipeline_token() is not None
    # The enabled hot path uses activation-bound callables. It does not walk
    # module metadata again for every decoded token.
    for embedding, _route in bindings:
        embedding._streamed_ar_active = None
    model.flush_ar_pipeline_ple()
    model.discard_ar_pipeline_ple()
    for embedding, route in bindings:
        embedding._streamed_ar_active = route
    assert all(
        layer.ple.ple_embedding._stage_disabled
        for layer in model.layers
        if "ple" in layer
    )
    assert all(
        not layer.ple.ple_embedding.ngram_embedding.prefer_lazy
        for layer in model.layers
        if "ple" in layer
    )
    assert model.set_ar_pipeline_mode(False) is True
    assert routes == [("active", True), ("flush",), ("discard",), ("active", False)]
