"""Source and construction contracts for the DeepSeek-V4 MoE tail lane.

The candidate deliberately leaves ``SwitchGLU`` alone.  The only replacement is
the stock tail::

    (routed * weights[..., None].astype(routed.dtype)).sum(axis=-2) + shared

for the shipped body geometry: BF16, top-k six, hidden 4096.  The direct Metal
test belongs in a guarded GPU window; these CPU-safe tests pin the invariants
that decide whether such a route may be installed at all.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402
from mtplx.attention_context import (  # noqa: E402
    attention_phase,
    current_attention_phase,
)

@pytest.fixture(autouse=True)
def _cpu_default_device():
    # These contract tests are CPU-deterministic by design, but the device pin
    # must stay test-scoped: pytest imports every test module before running
    # any, so a module-level set_default_device(mx.cpu) here leaked CPU into
    # the whole process and flipped the engine's Metal bit-exactness suites
    # onto CPU fallbacks (48 failures in a full run, none in isolation).
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL = os.path.join(_HERE, "..", "mtplx", "models", "deepseek_v4.py")
_spec = importlib.util.spec_from_file_location("dsv4_moe_tail_undertest", _MODEL)
D = importlib.util.module_from_spec(_spec)
sys.modules["dsv4_moe_tail_undertest"] = D
_spec.loader.exec_module(D)

_GATE_PATH = Path(_HERE).parent / "scripts" / "deepseek_v4_moe_tail_gate.py"
_gate_spec = importlib.util.spec_from_file_location(
    "dsv4_moe_tail_gate_undertest", _GATE_PATH
)
G = importlib.util.module_from_spec(_gate_spec)
sys.modules["dsv4_moe_tail_gate_undertest"] = G
_gate_spec.loader.exec_module(G)


def _args(**over):
    kwargs = dict(
        hidden_size=4096,
        moe_intermediate_size=2048,
        n_routed_experts=256,
        n_shared_experts=1,
        num_experts_per_tok=6,
        num_hash_layers=3,
        num_nextn_predict_layers=1,
        compress_ratios=[0] * 44,
    )
    kwargs.update(over)
    return D.ModelArgs(**kwargs)


def _artifact_contract(*, body_bits=2, mtp=True):
    quantization = {"group_size": 64, "bits": 4, "mode": "affine"}
    weight_map = {}
    for layer in range(43):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            stem = f"model.layers.{layer}.ffn.switch_mlp.{proj}"
            quantization[stem] = {
                "group_size": (
                    32 if proj == "gate_proj" and layer < 42 else 64
                ),
                "bits": body_bits,
                "mode": "affine",
            }
            for suffix in ("weight", "scales", "biases"):
                weight_map[f"{stem}.{suffix}"] = "model.safetensors"
    if mtp:
        weight_map["mtp.0.h_proj.weight"] = "mtp.safetensors"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            quantization[f"mtp.0.ffn.switch_mlp.{proj}"] = {
                "group_size": 32,
                "bits": 4,
                "mode": "mxfp4",
            }
    config = {
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
        "num_hidden_layers": 43,
        "num_nextn_predict_layers": 1 if mtp else 0,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "compress_ratios": [0] * (44 if mtp else 43),
        "quantization": quantization,
    }
    return config, {"metadata": {"total_size": 1}, "weight_map": weight_map}


def _fake_tensor(shape, dtype):
    return SimpleNamespace(shape=tuple(shape), dtype=dtype)


def _fake_quantized_projection(
    *,
    bits,
    group_size,
    mode,
    weight_shape,
    scales_shape,
    biases=True,
    scales_dtype=mx.bfloat16,
):
    return SimpleNamespace(
        bits=bits,
        group_size=group_size,
        mode=mode,
        weight=_fake_tensor(weight_shape, mx.uint32),
        scales=_fake_tensor(scales_shape, scales_dtype),
        biases=_fake_tensor(scales_shape, mx.bfloat16) if biases else None,
    )


def _fake_body_switch(layer_id):
    gate_group_size = 32 if layer_id < 42 else 64
    gate_scale_groups = 128 if layer_id < 42 else 64
    return SimpleNamespace(
        gate_proj=_fake_quantized_projection(
            bits=2,
            group_size=gate_group_size,
            mode="affine",
            weight_shape=(256, 2048, 256),
            scales_shape=(256, 2048, gate_scale_groups),
        ),
        up_proj=_fake_quantized_projection(
            bits=2,
            group_size=64,
            mode="affine",
            weight_shape=(256, 2048, 256),
            scales_shape=(256, 2048, 64),
        ),
        down_proj=_fake_quantized_projection(
            bits=2,
            group_size=64,
            mode="affine",
            weight_shape=(256, 4096, 128),
            scales_shape=(256, 4096, 32),
        ),
    )


def _fake_body_shared():
    return SimpleNamespace(
        gate_proj=_fake_quantized_projection(
            bits=4,
            group_size=64,
            mode="affine",
            weight_shape=(2048, 512),
            scales_shape=(2048, 64),
        ),
        up_proj=_fake_quantized_projection(
            bits=4,
            group_size=64,
            mode="affine",
            weight_shape=(2048, 512),
            scales_shape=(2048, 64),
        ),
        down_proj=_fake_quantized_projection(
            bits=4,
            group_size=64,
            mode="affine",
            weight_shape=(4096, 256),
            scales_shape=(4096, 32),
        ),
    )


def _fake_gate(layer_id):
    gate = SimpleNamespace(
        dim=4096,
        topk=6,
        n_routed=256,
        hash=layer_id < 3,
        weight=_fake_tensor((256, 4096), mx.bfloat16),
    )
    if gate.hash:
        gate.tid2eid = _fake_tensor((129280, 6), mx.int64)
    else:
        gate.e_score_correction_bias = _fake_tensor((256,), mx.float32)
    return gate


def _fake_mtp_switch():
    return SimpleNamespace(
        gate_proj=_fake_quantized_projection(
            bits=4,
            group_size=32,
            mode="mxfp4",
            weight_shape=(256, 2048, 512),
            scales_shape=(256, 2048, 128),
            biases=False,
            scales_dtype=mx.uint8,
        ),
        up_proj=_fake_quantized_projection(
            bits=4,
            group_size=32,
            mode="mxfp4",
            weight_shape=(256, 2048, 512),
            scales_shape=(256, 2048, 128),
            biases=False,
            scales_dtype=mx.uint8,
        ),
        down_proj=_fake_quantized_projection(
            bits=4,
            group_size=32,
            mode="mxfp4",
            weight_shape=(256, 4096, 256),
            scales_shape=(256, 4096, 64),
            biases=False,
            scales_dtype=mx.uint8,
        ),
    )


def _fake_dense_shared():
    return SimpleNamespace(
        gate_proj=SimpleNamespace(weight=_fake_tensor((2048, 4096), mx.bfloat16)),
        up_proj=SimpleNamespace(weight=_fake_tensor((2048, 4096), mx.bfloat16)),
        down_proj=SimpleNamespace(weight=_fake_tensor((4096, 2048), mx.bfloat16)),
    )


def _loaded_tail_model(*, body_layers=43):
    args = _args(
        num_hidden_layers=43,
        num_nextn_predict_layers=1,
        n_shared_experts=1,
        vocab_size=129280,
    )
    layers = []
    for layer_id in range(body_layers):
        layers.append(
            SimpleNamespace(
                ffn=SimpleNamespace(
                    gate=_fake_gate(layer_id),
                    switch_mlp=_fake_body_switch(layer_id),
                    shared_experts=_fake_body_shared(),
                    _tail_combine=D._stock_moe_tail_combine,
                )
            )
        )
    mtp = SimpleNamespace(
        ffn=SimpleNamespace(
            gate=_fake_gate(43),
            switch_mlp=_fake_mtp_switch(),
            shared_experts=_fake_dense_shared(),
            _tail_combine=object(),
        )
    )
    model = SimpleNamespace(
        model_type="deepseek_v4",
        args=args,
        layers=layers,
        mtp_blocks=[mtp],
    )
    config, _index = _artifact_contract()
    config.update(
        {
            "num_hash_layers": 3,
            "n_shared_experts": 1,
            "vocab_size": 129280,
        }
    )
    return model, config


def test_tail_default_is_off_and_stock_expression_remains_visible():
    """The opt-in cannot affect the ordinary model construction path."""
    assert D._MOE_TAIL is False
    source = open(_MODEL, encoding="utf-8").read()
    assert "(routed * weights[..., None].astype(routed.dtype)).sum(axis=-2)" in source


def test_tail_geometry_validation_accepts_only_shipped_body_contract():
    """Top-k, hidden width, and BF16-store arm are installation invariants."""
    D._validate_moe_tail_config(_args())
    with pytest.raises(ValueError, match="top-k=6"):
        D._validate_moe_tail_config(_args(num_experts_per_tok=4))
    with pytest.raises(ValueError, match="hidden_size=4096"):
        D._validate_moe_tail_config(_args(hidden_size=2048))


def test_tail_constructor_always_prebinds_stock_before_weights_load(monkeypatch):
    monkeypatch.setattr(D, "_MOE_TAIL", True)
    monkeypatch.setattr(D, "MoEGate", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(D, "SwitchGLU", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(
        D, "DeepseekV4MLP", lambda *_args, **_kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(
        D,
        "_install_moe_tail_combine",
        lambda _args: pytest.fail("candidate installation ran before weight loading"),
    )
    args = _args(
        hidden_size=32,
        moe_intermediate_size=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
    )
    moe = D.DeepseekV4MoE(args, layer_id=0)
    assert moe._tail_combine is D._stock_moe_tail_combine


def test_post_load_installer_validates_then_binds_body_candidate_and_mtp_stock(
    monkeypatch,
):
    model, config = _loaded_tail_model()
    candidate = object()
    monkeypatch.setattr(D, "_MOE_TAIL", True)
    monkeypatch.setattr(D, "_install_moe_tail_combine", lambda _args: candidate)
    report = D.configure_deepseek_v4_moe_tail(model, config)
    assert report["body_layers_installed"] == 43
    assert report["body_q2_routed_projections"] == 129
    assert all(layer.ffn._tail_combine is candidate for layer in model.layers)
    assert model.mtp_blocks[0].ffn._tail_combine is D._stock_moe_tail_combine


@pytest.mark.parametrize("body_layers", (42, 44))
def test_post_load_installer_rejects_wrong_body_layer_count(monkeypatch, body_layers):
    model, config = _loaded_tail_model(body_layers=body_layers)
    monkeypatch.setattr(D, "_MOE_TAIL", True)
    monkeypatch.setattr(
        D,
        "_install_moe_tail_combine",
        lambda _args: pytest.fail("kernel built before topology validation"),
    )
    with pytest.raises(ValueError, match="exactly 43 body layers"):
        D.configure_deepseek_v4_moe_tail(model, config)


@pytest.mark.parametrize("mtp_blocks", (0, 2))
def test_post_load_installer_rejects_wrong_mtp_block_count(monkeypatch, mtp_blocks):
    model, config = _loaded_tail_model()
    model.mtp_blocks = model.mtp_blocks * mtp_blocks
    monkeypatch.setattr(D, "_MOE_TAIL", True)
    monkeypatch.setattr(
        D,
        "_install_moe_tail_combine",
        lambda _args: pytest.fail("kernel built before MTP topology validation"),
    )
    with pytest.raises(ValueError, match="exactly one MTP block"):
        D.configure_deepseek_v4_moe_tail(model, config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("num_hash_layers", 2, "num_hash_layers=3"),
        ("num_hash_layers", 4, "num_hash_layers=3"),
        ("moe_intermediate_size", 1536, "moe_intermediate_size=2048"),
        ("n_shared_experts", 2, "n_shared_experts=1"),
    ),
)
def test_post_load_installer_rejects_wrong_shape_config(
    monkeypatch, field, value, message
):
    model, config = _loaded_tail_model()
    setattr(model.args, field, value)
    config[field] = value
    monkeypatch.setattr(D, "_MOE_TAIL", True)
    with pytest.raises(ValueError, match=message):
        D.configure_deepseek_v4_moe_tail(model, config)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    (
        ("bits", 4, "bits=2"),
        ("group_size", 16, "group_size=32"),
        ("mode", "mxfp4", "mode=affine"),
    ),
)
def test_post_load_installer_rejects_non_q2_affine_body_storage(
    monkeypatch, attribute, value, message
):
    model, config = _loaded_tail_model()
    setattr(model.layers[3].ffn.switch_mlp.gate_proj, attribute, value)
    monkeypatch.setattr(D, "_MOE_TAIL", True)
    with pytest.raises(ValueError, match=message):
        D.configure_deepseek_v4_moe_tail(model, config)


def test_post_load_installer_requires_layer42_gate_group64_exception(monkeypatch):
    model, config = _loaded_tail_model()
    projection = model.layers[42].ffn.switch_mlp.gate_proj
    assert projection.group_size == 64
    monkeypatch.setattr(D, "_MOE_TAIL", True)
    projection.group_size = 32
    projection.scales.shape = (256, 2048, 128)
    projection.biases.shape = (256, 2048, 128)
    with pytest.raises(ValueError, match="group_size=64"):
        D.configure_deepseek_v4_moe_tail(model, config)


def test_post_load_installer_rejects_config_storage_map_drift(monkeypatch):
    model, config = _loaded_tail_model()
    stem = "model.layers.3.ffn.switch_mlp.gate_proj"
    config["quantization"][stem]["bits"] = 4
    monkeypatch.setattr(D, "_MOE_TAIL", True)
    with pytest.raises(ValueError, match="config quantization"):
        D.configure_deepseek_v4_moe_tail(model, config)


def test_post_load_installer_rejects_non_uint32_or_wrong_packed_geometry(monkeypatch):
    model, config = _loaded_tail_model()
    projection = model.layers[3].ffn.switch_mlp.gate_proj
    monkeypatch.setattr(D, "_MOE_TAIL", True)
    projection.weight.dtype = mx.bfloat16
    with pytest.raises(ValueError, match="uint32"):
        D.configure_deepseek_v4_moe_tail(model, config)
    projection.weight.dtype = mx.uint32
    projection.weight.shape = (256, 2048, 255)
    with pytest.raises(ValueError, match="packed weight shape"):
        D.configure_deepseek_v4_moe_tail(model, config)
    projection.weight.shape = (256, 2048, 256)
    projection.scales.shape = (256, 2048, 127)
    with pytest.raises(ValueError, match="scale/bias shape"):
        D.configure_deepseek_v4_moe_tail(model, config)


@pytest.mark.parametrize("attribute", ("scales", "biases"))
def test_post_load_installer_rejects_missing_q2_affine_storage(monkeypatch, attribute):
    model, config = _loaded_tail_model()
    setattr(model.layers[3].ffn.switch_mlp.gate_proj, attribute, None)
    monkeypatch.setattr(D, "_MOE_TAIL", True)
    with pytest.raises(ValueError, match="scale/bias shape"):
        D.configure_deepseek_v4_moe_tail(model, config)


def test_post_load_installer_rejects_wrong_shared_or_mtp_dense_geometry(monkeypatch):
    model, config = _loaded_tail_model()
    monkeypatch.setattr(D, "_MOE_TAIL", True)
    model.layers[3].ffn.shared_experts.gate_proj.weight.shape = (2047, 512)
    with pytest.raises(ValueError, match="body shared"):
        D.configure_deepseek_v4_moe_tail(model, config)
    model.layers[3].ffn.shared_experts.gate_proj.weight.shape = (2048, 512)
    model.mtp_blocks[0].ffn.shared_experts.down_proj.weight.shape = (4095, 2048)
    with pytest.raises(ValueError, match="MTP dense shared"):
        D.configure_deepseek_v4_moe_tail(model, config)


def test_runtime_configures_tail_after_load_and_requant_before_mtp_publish():
    runtime_source = (Path(_HERE).parent / "mtplx" / "runtime.py").read_text()
    load = runtime_source.index("model, tokenizer = _load_base_model(path, config)")
    requant = runtime_source.index("if proj_quant or proj_requant:", load)
    configure = runtime_source.index("configure_deepseek_v4_moe_tail", requant)
    publish = runtime_source.index("if mtp and not dspark:", configure)
    assert load < requant < configure < publish


def test_tail_rejects_fp32_activation_arm_at_installation():
    """The kernel has BF16 arithmetic by contract, never a hot-path dtype test."""
    saved = D._FP32_ACTIVATIONS
    try:
        D._FP32_ACTIVATIONS = True
        with pytest.raises(ValueError, match="BF16 activation storage"):
            D._validate_moe_tail_config(_args())
    finally:
        D._FP32_ACTIVATIONS = saved


def test_tail_kernel_uses_one_output_owner_and_real_metal_exact_selfcheck():
    """Association is not assumed: the constructed GPU route has to prove it."""
    source = D._MOE_TAIL_METAL_SOURCE
    assert "uint i = thread_position_in_grid.x" in source
    assert "for (uint route = 0; route < TOPK; ++route)" in source
    assert "T product" in source
    assert "T(mixed + shared" in source
    implementation = open(_MODEL, encoding="utf-8").read()
    assert "_verify_moe_tail_exact(kernel)" in implementation
    assert "for rows in (1, 4):" in implementation
    assert "current_attention_phase()" in implementation
    assert "_stock_moe_tail_combine" in implementation


def test_tail_dispatch_binds_the_metal_scalar_template():
    """The source's ``T`` type must be specialized at every dispatch."""
    captured = {}

    def fake_kernel(**kwargs):
        captured.update(kwargs)
        return (mx.zeros((1, 4096), dtype=mx.bfloat16),)

    routed = mx.zeros((1, 6, 4096), dtype=mx.bfloat16)
    weights = mx.zeros((1, 6), dtype=mx.bfloat16)
    shared = mx.zeros((1, 4096), dtype=mx.bfloat16)
    D._moe_tail_apply(fake_kernel, routed, weights, shared)
    assert captured["template"] == [("T", routed.dtype)]


@pytest.mark.parametrize("rows", [1, 4])
def test_prefill_tiny_shapes_remain_stock(monkeypatch, rows):
    """Flattened M alone cannot turn a tiny prefill into decode/verify."""
    monkeypatch.setattr(D, "_moe_tail_apply", lambda *_: mx.array([-99.0]))
    route = D._InstalledMoETailRoute(kernel=object())
    routed = mx.zeros((rows, 6, 8), dtype=mx.bfloat16)
    weights = mx.ones((rows, 6), dtype=mx.bfloat16)
    shared = mx.ones((rows, 8), dtype=mx.bfloat16)
    with attention_phase("prefill"):
        got = route(routed, weights, shared)
    assert tuple(got.shape) == (rows, 8)
    assert bool(mx.all(got == 1))


def test_decode_verify_m4_uses_custom(monkeypatch):
    sentinel = mx.array([-99.0])
    monkeypatch.setattr(D, "_moe_tail_apply", lambda *_: sentinel)
    route = D._InstalledMoETailRoute(kernel=object())
    with attention_phase("decode_verify"):
        got = route(
            mx.zeros((4, 6, 8)), mx.zeros((4, 6)), mx.zeros((4, 8))
        )
    assert got is sentinel


def test_k3_verify_m4_is_custom_but_decode_verify_m1_repair_is_stock(monkeypatch):
    """The generation phase is shared; K3's M=4, not the phase alone, selects."""
    custom_rows = []

    def custom(_kernel, routed, _weights, _shared):
        custom_rows.append(int(routed.shape[0]))
        return mx.full((routed.shape[0], routed.shape[-1]), -99.0)

    monkeypatch.setattr(D, "_moe_tail_apply", custom)
    route = D._InstalledMoETailRoute(kernel=object())
    with attention_phase("decode_verify"):
        verify = route(
            mx.zeros((4, 6, 8), dtype=mx.bfloat16),
            mx.zeros((4, 6), dtype=mx.bfloat16),
            mx.zeros((4, 8), dtype=mx.bfloat16),
        )
        repair = route(
            mx.zeros((1, 6, 8), dtype=mx.bfloat16),
            mx.zeros((1, 6), dtype=mx.bfloat16),
            mx.ones((1, 8), dtype=mx.bfloat16),
        )
    assert custom_rows == [4]
    assert bool(mx.all(verify == -99))
    assert bool(mx.all(repair == 1))


def test_real_mtpk_engine_routes_k3_m4_custom_and_rejection_repair_m1_stock(
    monkeypatch,
):
    """Exercise the production MTP loop, not a hand-written phase simulation."""
    from mtplx.generation import generate_mtpk
    from mtplx.mtp_patch import MTPContract
    from mtplx.runtime import MTPLXRuntime
    from mtplx.sampling import SamplerConfig

    args = D.ModelArgs(
        vocab_size=8,
        hidden_size=32,
        num_hidden_layers=1,
        num_hash_layers=0,
        num_attention_heads=4,
        head_dim=16,
        qk_rope_head_dim=8,
        q_lora_rank=16,
        o_lora_rank=8,
        o_groups=2,
        moe_intermediate_size=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        index_n_heads=4,
        index_head_dim=16,
        index_topk=4,
        compress_ratios=[0, 0],
        sliding_window=16,
        num_nextn_predict_layers=1,
    )
    model = D.Model(args)
    route = D._InstalledMoETailRoute(kernel=object())
    model.layers[0].ffn._tail_combine = route
    custom_rows = []
    stock_rows = []
    real_stock = D._stock_moe_tail_combine

    def custom(_kernel, routed, weights, shared):
        custom_rows.append(int(routed.shape[0]))
        return real_stock(routed, weights, shared)

    def stock(routed, weights, shared):
        if current_attention_phase() == "decode_verify":
            stock_rows.append(int(routed.shape[0]))
        return real_stock(routed, weights, shared)

    monkeypatch.setattr(D, "_moe_tail_apply", custom)
    monkeypatch.setattr(D, "_stock_moe_tail_combine", stock)

    tokenizer = SimpleNamespace(
        eos_token_id=None,
        eos_token_ids=set(),
        decode=lambda tokens: " ".join(str(token) for token in tokens),
    )
    runtime = MTPLXRuntime(
        model=model,
        tokenizer=tokenizer,
        model_path=Path("."),
        mtp_enabled=True,
        contract=MTPContract(),
    )
    real_draft = runtime.draft_mtp

    def rejecting_draft(*args, **kwargs):
        result = real_draft(*args, **kwargs)
        logits, hidden = result if isinstance(result, tuple) else (result, None)
        forced = mx.zeros_like(logits)
        forced[..., 1] = 1
        return (forced, hidden) if isinstance(result, tuple) else forced

    monkeypatch.setattr(runtime, "draft_mtp", rejecting_draft)
    out = generate_mtpk(
        runtime,
        [1, 2, 3, 4],
        max_tokens=5,
        sampler=SamplerConfig(temperature=0.0),
        speculative_depth=3,
        mtp_history_policy="committed",
        stop_token_ids=set(),
        verify_strategy="batched",
    )
    stats = out.stats.to_dict()
    assert stats["requested_speculative_depth"] == 3
    assert stats["rejected_drafts"] > 0
    assert 4 in custom_rows, "K3 target verify must execute flattened M=K+1=4"
    assert 1 in stock_rows, "a rejected K3 cycle must repair through stock M1"
    assert 1 not in custom_rows


@pytest.mark.parametrize("phase", ["ar_decode", "unknown"])
def test_m1_stays_stock_outside_verify_route(monkeypatch, phase):
    sentinel = mx.array([-99.0])
    monkeypatch.setattr(D, "_moe_tail_apply", lambda *_: sentinel)
    route = D._InstalledMoETailRoute(kernel=object())
    routed = mx.zeros((1, 6, 8), dtype=mx.bfloat16)
    weights = mx.zeros((1, 6), dtype=mx.bfloat16)
    shared = mx.ones((1, 8), dtype=mx.bfloat16)
    with attention_phase(phase):
        got = route(routed, weights, shared)
    assert tuple(got.shape) == (1, 8)
    assert bool(mx.all(got == 1))


def test_tail_is_not_a_cpu_silent_fallback_when_explicitly_enabled():
    """An enabled Metal lane must fail before generation on an unsupported device."""
    with pytest.raises(RuntimeError, match="GPU"):
        D._install_moe_tail_combine(_args())


def test_guarded_tail_gate_is_one_load_and_synchronizes_each_sample():
    """Its timings are diagnostics only; the later full TPS bracket is the verdict."""
    source = _GATE_PATH.read_text(encoding="utf-8")
    assert "mtplx_runtime.load(model_path, mtp=True)" in source
    assert "_load_base_model" not in source
    reject = source.index("if D._MOE_TAIL:")
    load = source.index("mtplx_runtime.load(model_path, mtp=True)")
    assert reject < load, "reject an enabled candidate before the heavyweight load"
    assert '_REQUIRED_MLX_VERSION = "0.31.2"' in source
    assert "d7bd29fc20b4a08318d21161c3dfb340889cc9454c5e554ad749eb0127cfa2d6" in source
    assert "ee94397faa812c91d5f1a0ee17c5bb6ca6032883653591dd33d4cfddb737ac33" in source
    assert "smoke-2bitdq-20260731-prompt2.txt" in source
    assert "_REQUIRED_PROMPT_TOKENS = 328" in source
    assert "hashlib.sha256" in source
    assert "mx.eval(out)" in source
    assert "mx.synchronize()" in source
    assert "exact_parity" in source
    assert "promotion" not in source.lower()


def test_legacy_gate_verifies_guard_before_first_mlx_import_and_records_it():
    source = _GATE_PATH.read_text(encoding="utf-8")
    main = source[source.index("def main()") :]
    assert "import mlx.core as mx" not in source[: source.index("def main()")]
    assert main.index("load_verified_guard_window()") < main.index(
        "import mlx.core as mx"
    )
    assert '"guard_window": guard_window' in main


def test_legacy_gate_refuses_unguarded_without_importing_mlx(tmp_path):
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
        [
            sys.executable,
            str(_GATE_PATH),
            "--prompt-file",
            "missing.txt",
            "--out",
            str(tmp_path / "receipt.json"),
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert completed.returncode != 0
    assert "verified guard window environment is absent or malformed" in completed.stderr
    assert not marker.exists()


def test_gate_rejects_4bit_body_routed_experts():
    config, index = _artifact_contract(body_bits=4)
    with pytest.raises(ValueError, match="Q2 affine routed expert"):
        G._validate_model_contract(config, index)


def test_gate_rejects_non_mtp_artifact():
    config, index = _artifact_contract(mtp=False)
    with pytest.raises(ValueError, match=r"43\+1 MTP topology"):
        G._validate_model_contract(config, index)


def test_gate_pins_merged_2bit_dq_mtp_manifests_and_loaded_storage():
    assert G._REQUIRED_MODEL_CONFIG_SHA256 == (
        "c8ff87fd5ee5c9587d0c937e9bfd3193e1a1621141aa367848a9610b3291fa6f"
    )
    assert G._REQUIRED_MODEL_INDEX_SHA256 == (
        "c84d2b369f5d5023d0f2d183fc36a935a3981751414996243b65f069983e43d8"
    )

    config, _index = _artifact_contract()

    def projection(bits, group_size, mode, *, biases=True):
        return SimpleNamespace(
            bits=bits,
            group_size=group_size,
            mode=mode,
            weight=mx.zeros((1,), dtype=mx.uint32),
            scales=mx.zeros((1,), dtype=mx.float16),
            biases=mx.zeros((1,), dtype=mx.float16) if biases else None,
        )

    body_switch = SimpleNamespace(
        gate_proj=projection(2, 32, "affine"),
        up_proj=projection(2, 64, "affine"),
        down_proj=projection(2, 64, "affine"),
    )
    body_switch_last = SimpleNamespace(
        gate_proj=projection(2, 64, "affine"),
        up_proj=projection(2, 64, "affine"),
        down_proj=projection(2, 64, "affine"),
    )
    mtp_switch = SimpleNamespace(
        gate_proj=projection(4, 32, "mxfp4", biases=False),
        up_proj=projection(4, 32, "mxfp4", biases=False),
        down_proj=projection(4, 32, "mxfp4", biases=False),
    )
    model = SimpleNamespace(
        model_type="deepseek_v4",
        layers=(
            [SimpleNamespace(ffn=SimpleNamespace(switch_mlp=body_switch))] * 42
            + [SimpleNamespace(ffn=SimpleNamespace(switch_mlp=body_switch_last))]
        ),
        mtp_blocks=[SimpleNamespace(ffn=SimpleNamespace(switch_mlp=mtp_switch))],
    )
    runtime = SimpleNamespace(model=model, mtp_enabled=True)
    identity = G._validate_loaded_runtime(runtime, config)
    assert identity["body_q2_routed_projections"] == 129
    assert identity["mtp_blocks_bound"] == 1
    assert identity["mtp_mxfp4_routed_projections"] == 3
