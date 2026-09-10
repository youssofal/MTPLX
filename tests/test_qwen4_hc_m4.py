"""MTPLX_QWEN4_HC_M4 — wiring, eligibility, and the eager chain's op contract.

Two things are proved here, both entirely on the CPU stream with tiny tensors
(no Metal, no kernel dispatch, no model):

1. THE OP CONTRACT the Metal kernel encodes. ``mtplx/kernels/qwen4_m4_hyper_read``
   reproduces the eager chain op by op, and every rounding boundary it copies
   is an assumption about how MLX types and rounds a specific expression. If
   MLX ever changes one of those (``2.0 * bf16`` promoting to fp32, ``mx.mean``
   accumulating in fp32, ``nn.silu`` growing its own kernel), the kernel goes
   silently wrong. These tests fail instead.

2. THE GATE. Flag off: nothing changes and nothing is imported. Flag on: rows
   1 keeps the old path, rows 2..8 either take the kernel or RAISE with the
   offending field named. There is no silent fallback — that is the failure
   mode that left MTPLX_FUSED_HC_V3 armed but structurally inert at M=4.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

import mtplx.kernels.qwen4_m4_hyper_read as hcm4
import mtplx.models.qwen4_exp as qwen4_exp
import mtplx.runtime_options as runtime_options

HCD = hcm4.HCD
LOWRANK = hcm4.R_LOWRANK
HC = hcm4.HC


@pytest.fixture(autouse=True)
def _cpu_stream():
    """Confine every op in this module to the CPU stream."""

    with mx.stream(mx.cpu):
        yield


@pytest.fixture(autouse=True)
def _flag_off(monkeypatch):
    """Default every test to the shipped state, whatever the session env is."""

    monkeypatch.setattr(runtime_options, "_QWEN4_HC_M4", False)


@pytest.fixture
def armed(monkeypatch):
    monkeypatch.setattr(runtime_options, "_QWEN4_HC_M4", True)
    return True


def _bits(value: mx.array) -> mx.array:
    widths = {mx.bfloat16: mx.uint16, mx.float16: mx.uint16, mx.float32: mx.uint32}
    return value.view(widths[value.dtype])


def _same_bits(a: mx.array, b: mx.array) -> bool:
    return bool(mx.all(_bits(a) == _bits(b)).item())


# --------------------------------------------------------------------------
# 1. the eager chain's op contract, as encoded in the Metal source
# --------------------------------------------------------------------------


def test_divide_by_hc_count_stays_bf16():
    """``mix / self.hc_count`` must not widen: the kernel rounds it to bf16."""

    a = (mx.random.normal((64,)) * 3).astype(mx.bfloat16)
    assert (a / 4).dtype == mx.bfloat16
    assert _same_bits(a / 4, (a.astype(mx.float32) * 0.25).astype(mx.bfloat16))


def test_inject_scale_stays_bf16():
    """``2.0 * mx.sigmoid(...)`` — a Python float is a weak scalar in MLX, so
    the inject value the kernel writes is bf16, not fp32."""

    a = (mx.random.normal((64,)) * 3).astype(mx.bfloat16)
    s = mx.sigmoid(a)
    assert (2.0 * s).dtype == mx.bfloat16


def test_silu_is_x_times_sigmoid_x():
    """The kernel writes ``(T)(sigmoid(t0) * t0)``; nn.silu must be that."""

    a = (mx.random.normal((256,)) * 3).astype(mx.bfloat16)
    assert nn.silu(a).dtype == mx.bfloat16
    assert _same_bits(nn.silu(a), a * mx.sigmoid(a))


def test_hc_mean_is_bf16_sum_times_quarter():
    """``mx.mean(mix * grouped, axis=-2)`` == ``mx.sum(...) * 0.25``, and that
    sum accumulates IN bf16 — which is why the kernel's hc reduction rounds at
    every add instead of accumulating in fp32."""

    a = (mx.random.normal((512, HC, 4)) * 3).astype(mx.bfloat16)
    m = mx.mean(a, axis=-2)
    s = mx.sum(a, axis=-2)
    assert m.dtype == mx.bfloat16 and s.dtype == mx.bfloat16
    assert _same_bits(m, (s.astype(mx.float32) * 0.25).astype(mx.bfloat16))

    seq = a[:, 0, :]
    for g in range(1, HC):
        seq = seq + a[:, g, :]
    assert _same_bits(s, seq), (
        "mx.sum over a length-4 bf16 axis is no longer a sequential bf16 "
        "accumulation; qwen4_m4_hyper_read's hc reduction must follow it"
    )


def test_hc_mean_is_not_fp32_accumulated():
    """Guard the *reason* the test above is not cosmetic: an fp32 accumulation
    of the same four terms is a genuinely different answer."""

    a = (mx.random.normal((4096, HC)) * 3).astype(mx.bfloat16)
    s = mx.sum(a, axis=-1)
    f32 = a.astype(mx.float32).sum(axis=-1).astype(mx.bfloat16)
    assert not _same_bits(s, f32)


def test_grouped_rms_norm_decomposition():
    """``GroupedRMSNorm`` == per-group rms_norm rounded to bf16, then a bf16
    multiply by the full-width weight — the two-step the kernel's K0 copies."""

    dims, group = 32, 8
    norm = qwen4_exp.GroupedRMSNorm(dims, group, eps=1e-6)
    norm.weight = (mx.random.normal((dims,)) * 0.5 + 1.0).astype(mx.bfloat16)
    x = (mx.random.normal((3, dims)) * 2).astype(mx.bfloat16)

    got = norm(x)
    grouped = x.reshape(3, -1, group)
    step1 = mx.fast.rms_norm(grouped, None, 1e-6).reshape(3, dims)
    assert step1.dtype == mx.bfloat16
    assert _same_bits(got, step1 * norm.weight)


# --------------------------------------------------------------------------
# 2. check_shapes — returns rows, or raises with the field named
# --------------------------------------------------------------------------


def _family_weights(dtype=mx.bfloat16, *, down_rows=LOWRANK):
    return {
        "gamma": mx.zeros((HCD,), dtype=dtype),
        "wd": mx.zeros((down_rows, HCD), dtype=dtype),
        "wu": mx.zeros((HCD, LOWRANK), dtype=dtype),
        "wi": mx.zeros((HC, HCD), dtype=dtype),
    }


@pytest.mark.parametrize("rows", [2, 3, 4, 5, 8])
def test_check_shapes_accepts_verify_widths(rows):
    w = _family_weights()
    x = mx.zeros((rows, HCD), dtype=mx.bfloat16)
    assert hcm4.check_shapes(x, w["gamma"], w["wd"], w["wu"], w["wi"]) == rows


@pytest.mark.parametrize("rows", [1, 9, 16])
def test_check_shapes_rejects_other_widths(rows):
    w = _family_weights()
    x = mx.zeros((rows, HCD), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="wired for 2..8 rows"):
        hcm4.check_shapes(x, w["gamma"], w["wd"], w["wu"], w["wi"])


def test_check_shapes_rejects_wrong_hidden():
    w = _family_weights()
    with pytest.raises(ValueError, match=r"must be \[\.\.\., 10240\]"):
        hcm4.check_shapes(
            mx.zeros((4, 4096), dtype=mx.bfloat16),
            w["gamma"], w["wd"], w["wu"], w["wi"],
        )


def test_check_shapes_names_the_bad_weight():
    w = _family_weights(down_rows=256)
    x = mx.zeros((4, HCD), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="input_mix_weight_down.weight"):
        hcm4.check_shapes(x, w["gamma"], w["wd"], w["wu"], w["wi"])


def test_check_shapes_rejects_dtype_mismatch():
    w = _family_weights()
    w["wu"] = mx.zeros((HCD, LOWRANK), dtype=mx.float16)
    x = mx.zeros((4, HCD), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="input_mix_weight_up.weight dtype"):
        hcm4.check_shapes(x, w["gamma"], w["wd"], w["wu"], w["wi"])


def test_check_shapes_takes_the_module_shape_unreshaped():
    """GatedResidual is called with [B, S, 10240]; R is the leading product,
    and validating must not require materialising a reshape node."""

    w = _family_weights()
    x = mx.zeros((1, 4, HCD), dtype=mx.bfloat16)
    assert hcm4.check_shapes(x, w["gamma"], w["wd"], w["wu"], w["wi"]) == 4
    x = mx.zeros((2, 3, HCD), dtype=mx.bfloat16)
    assert hcm4.check_shapes(x, w["gamma"], w["wd"], w["wu"], w["wi"]) == 6


def test_driver_requires_a_two_d_view():
    w = _family_weights()
    with pytest.raises(ValueError, match="wants a 2-D"):
        hcm4.fused_hc_read_m4(
            mx.zeros((1, 4, HCD), dtype=mx.bfloat16),
            w["gamma"], w["wd"], w["wu"], w["wi"],
        )


def test_check_shapes_allows_missing_inject():
    """The trunk mixer is built with ``use_combine=False``."""

    w = _family_weights()
    x = mx.zeros((4, HCD), dtype=mx.bfloat16)
    assert hcm4.check_shapes(x, w["gamma"], w["wd"], w["wu"], None) == 4


def test_weight_bytes_matches_the_census_figure():
    assert hcm4.weight_bytes_per_read(2) == 13_209_600
    assert hcm4.weight_bytes_per_read(2, has_inject=False) == 13_127_680


# --------------------------------------------------------------------------
# 3. the gate in GatedResidual
# --------------------------------------------------------------------------


def _family_module(use_combine=True, dtype=mx.bfloat16):
    mod = qwen4_exp.GatedResidual(qwen4_exp.TextArgs(), use_combine=use_combine)
    w = _family_weights(dtype)
    mod.hc_norm.weight = w["gamma"]
    mod.input_mix_weight_down.weight = w["wd"]
    mod.input_mix_weight_up.weight = w["wu"]
    if use_combine:
        mod.block_inject_weight.weight = w["wi"]
    return mod


def _small_module(use_combine=True):
    args = qwen4_exp.TextArgs(hidden_size=8, hc_lowrank=4, rms_norm_eps=1e-6)
    mod = qwen4_exp.GatedResidual(args, use_combine=use_combine)
    hcd = args.hc_count * args.hidden_size
    mod.hc_norm.weight = (mx.random.normal((hcd,)) * 0.3 + 1.0).astype(mx.bfloat16)
    mod.input_mix_weight_down.weight = (
        mx.random.normal((args.hc_lowrank, hcd)) * 0.1
    ).astype(mx.bfloat16)
    mod.input_mix_weight_up.weight = (
        mx.random.normal((hcd, args.hc_lowrank)) * 0.1
    ).astype(mx.bfloat16)
    if use_combine:
        mod.block_inject_weight.weight = (
            mx.random.normal((args.hc_count, hcd)) * 0.1
        ).astype(mx.bfloat16)
    return mod, args


def test_flag_off_never_applies():
    mod = _family_module()
    for rows in (1, 2, 4, 8, 32):
        x = mx.zeros((1, rows, HCD), dtype=mx.bfloat16)
        assert mod._hc_m4_applies(x) is False


def test_flag_off_leaves_the_eager_chain_bit_identical():
    """The full eager expression, transcribed, on a tiny config."""

    mod, args = _small_module()
    hcd = args.hc_count * args.hidden_size
    x = (mx.random.normal((1, 4, hcd)) * 2).astype(mx.bfloat16)

    mixed, passthrough, inject = mod(x)

    normed = mod.hc_norm(x)
    mix = nn.silu(mod.input_mix_weight_down(normed) / args.hc_count)
    mix = mx.sigmoid(mod.input_mix_weight_up(mix))
    mix = mix.reshape(*mix.shape[:-1], args.hc_count, args.hidden_size)
    grouped = normed.reshape(*normed.shape[:-1], args.hc_count, args.hidden_size)
    want_mixed = mx.mean(mix * grouped, axis=-2)
    want_inject = 2.0 * mx.sigmoid(mod.block_inject_weight(normed) / args.hc_count)

    assert _same_bits(mixed, want_mixed)
    assert _same_bits(inject, want_inject)
    assert passthrough is x


def test_flag_off_tolerates_off_family_geometry():
    """Construction must not raise for a non-Flash-Next config when the flag
    is unset — this module class is shared."""

    qwen4_exp.GatedResidual(qwen4_exp.TextArgs(hidden_size=8, hc_lowrank=4))


@pytest.mark.parametrize("rows", [2, 3, 4, 8])
def test_armed_applies_at_verify_widths(armed, rows):
    mod = _family_module()
    assert mod._hc_m4_applies(mx.zeros((1, rows, HCD), dtype=mx.bfloat16)) is True


def test_armed_applies_to_the_noncombine_mixer(armed):
    mod = _family_module(use_combine=False)
    assert mod._hc_m4_applies(mx.zeros((1, 4, HCD), dtype=mx.bfloat16)) is True


@pytest.mark.parametrize("rows", [1])
def test_armed_leaves_row_one_alone(armed, rows):
    """rows == 1 is the draft path's business (v3 / eager); this gate must not
    take it, and must not raise on it either."""

    mod = _family_module()
    assert mod._hc_m4_applies(mx.zeros((1, rows, HCD), dtype=mx.bfloat16)) is False


def test_armed_leaves_prefill_widths_alone(armed):
    mod = _family_module()
    assert mod._hc_m4_applies(mx.zeros((1, 64, HCD), dtype=mx.bfloat16)) is False


def test_armed_off_family_config_raises_at_construction(armed):
    with pytest.raises(RuntimeError, match="not the Flash-Next family shape"):
        qwen4_exp.GatedResidual(qwen4_exp.TextArgs(hidden_size=8, hc_lowrank=4))
    with pytest.raises(RuntimeError, match="hc_count=2"):
        qwen4_exp.GatedResidual(qwen4_exp.TextArgs(hc_count=2))


def test_armed_quantized_mix_weights_raise(armed):
    mod = _family_module()
    mod.input_mix_weight_down.scales = mx.zeros((LOWRANK, HCD // 64), mx.bfloat16)
    with pytest.raises(RuntimeError, match="input_mix_weight_down is quantized"):
        mod._hc_m4_applies(mx.zeros((1, 4, HCD), dtype=mx.bfloat16))


def test_armed_dtype_mismatch_raises(armed):
    """fp32 mix weights against a bf16 hyper state: raise, do not fall back."""

    mod = _family_module(dtype=mx.float32)
    with pytest.raises(ValueError, match="dtype"):
        mod._hc_m4_applies(mx.zeros((1, 4, HCD), dtype=mx.bfloat16))


def test_armed_wrong_down_rows_raise(armed):
    mod = _family_module()
    mod.input_mix_weight_down.weight = mx.zeros((256, HCD), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="input_mix_weight_down.weight"):
        mod._hc_m4_applies(mx.zeros((1, 4, HCD), dtype=mx.bfloat16))


def test_env_flag_is_read_at_use_not_frozen_at_import(monkeypatch):
    """The reader must resolve the environment at USE, not freeze at import.

    The server's fixed-M4 auto-arm stamps MTPLX_QWEN4_HC_M4 AFTER
    runtime_options is imported; an import-time freeze left the served lane
    OFF (the battery's 2026-09-07 arming failure). A stamp applied after the
    module exists must be seen. ``_QWEN4_HC_M4`` is left None (unforced) so
    the reader consults the environment.
    """

    monkeypatch.setattr(runtime_options, "_QWEN4_HC_M4", None)
    monkeypatch.delenv("MTPLX_QWEN4_HC_M4", raising=False)
    monkeypatch.delenv("MTPLX_FABLE_HC_M4", raising=False)
    assert runtime_options.qwen4_hc_m4_enabled() is False
    monkeypatch.setenv("MTPLX_QWEN4_HC_M4", "1")
    assert runtime_options.qwen4_hc_m4_enabled() is True
    monkeypatch.delenv("MTPLX_QWEN4_HC_M4", raising=False)
    assert runtime_options.qwen4_hc_m4_enabled() is False
    # A test/A-B may still force a value by setting the module global.
    monkeypatch.setattr(runtime_options, "_QWEN4_HC_M4", True)
    monkeypatch.delenv("MTPLX_QWEN4_HC_M4", raising=False)
    assert runtime_options.qwen4_hc_m4_enabled() is True


def test_env_flag_defaults_off():
    from mtplx.runtime_options import env_bool

    assert env_bool("MTPLX_QWEN4_HC_M4", default=False, env={}) is False


def test_env_flag_rejects_a_bad_spelling():
    from mtplx.runtime_options import env_bool

    with pytest.raises(ValueError):
        env_bool("MTPLX_QWEN4_HC_M4", default=False, env={"MTPLX_QWEN4_HC_M4": "yep"})


# --------------------------------------------------------------------------
# 4. the driver's own guards (no dispatch: these all raise before the kernel)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"norm_threads": 100}, "norm_threads"),
        ({"down_threads": 2048}, "down_threads"),
        ({"out_per_tg": 0}, "out_per_tg"),
        ({"d_per_block": 999}, "d_per_block"),
    ],
)
def test_driver_rejects_bad_tuning(kwargs, match):
    w = _family_weights()
    x = mx.zeros((4, HCD), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match=match):
        hcm4.fused_hc_read_m4(
            x, w["gamma"], w["wd"], w["wu"], w["wi"], **kwargs
        )


def test_dispatch_budget_is_three():
    assert hcm4.DISPATCHES_PER_READ == 3
    assert hcm4.EAGER_DISPATCHES_PER_READ == 11


# --------------------------------------------------------------------------
# 5. the PACK contract moves to install time
# --------------------------------------------------------------------------
#
# Every term below is a property of the loaded weights, so it is the same
# answer for every request the process will ever serve. Checking it when the
# weights land means a mis-armed flag stops the server coming up with a
# precise reason, instead of turning the first request that happens to reach
# verify width into an HTTP 500.


def _real_tree(num_layers: int = 2, *, with_mtp: bool = True):
    """The REAL Qwen3.8 module tree, built from the model's own config.

    Same classes, same attribute names, same containers as a served pack --
    ``Model.language_model.model.layers[i].{attn,mlp}_hyper_connection``, the
    text model's own ``hyper_connection_mixer``, and the MTP sub-tree
    published on ``language_model`` by ``inject_qwen4_exp_mtp_support``.  Only
    the layer count and the vocabulary are cut, and neither is part of the
    MTPLX_QWEN4_HC_M4 contract; the hyper-connection geometry the kernel
    hardcodes (hc_count 4, hidden 2560, lowrank 320) is the shipped one.

    Cheap despite the real shapes: MLX arrays are lazy, and nothing here
    evaluates one.
    """

    config = {
        "model_type": "qwen4_exp",
        "num_hidden_layers": num_layers,
        # full_attention first: Qwen4ExpMTP picks the first non-linear layer
        # as its own, exactly as the shipped config's layer_types allow.
        "layer_types": (["full_attention", "linear_attention"] * num_layers)[
            :num_layers
        ],
        "vocab_size": 1024,
        "tie_word_embeddings": False,
    }
    model = qwen4_exp.Model(qwen4_exp.ModelArgs.from_dict(config))
    if with_mtp:
        # `Qwen4ExpMTP` is published on language_model, NOT on Model --
        # registering it on both trees would double-count its parameters.
        model.language_model.mtp = qwen4_exp.Qwen4ExpMTP(model.language_model.args)
    return model


def test_pack_validation_is_a_no_op_when_the_flag_is_off():
    report = qwen4_exp.install_hc_m4_pack_validation(_real_tree())
    assert report == {"armed": False, "validated": 0}


def test_pack_validation_finds_every_module_in_the_real_tree(armed):
    """The 2026-09-02 dead-server regression, pinned.

    The first cut walked ``dir(layer)`` and found NOTHING on a served pack,
    so an armed flag raised "no GatedResidual ... to replace" and killed the
    load for both the HumanEval screen and the ABBA lane -- on a model where
    the kernel had been installing and running bit-exact all day.
    """

    model = _real_tree(num_layers=2)
    report = qwen4_exp.install_hc_m4_pack_validation(model)
    assert report["armed"] is True
    # 2 per decoder layer + the text model's mixer + the MTP's own layer pair
    # and mixer.
    assert report["validated"] == 8


def test_pack_validation_reaches_all_three_places_the_model_keeps_one(armed):
    """Held directly, inside a list, and behind a published sub-tree."""

    model = _real_tree(num_layers=2)
    paths = [path for path, _ in qwen4_exp._named_gated_residuals(model)]
    assert paths == [
        "language_model.model.hyper_connection_mixer",
        "language_model.model.layers.0.attn_hyper_connection",
        "language_model.model.layers.0.mlp_hyper_connection",
        "language_model.model.layers.1.attn_hyper_connection",
        "language_model.model.layers.1.mlp_hyper_connection",
        "language_model.mtp.hyper_connection_mixer",
        "language_model.mtp.layers.0.attn_hyper_connection",
        "language_model.mtp.layers.0.mlp_hyper_connection",
    ]


def test_dir_cannot_see_an_mlx_modules_children(armed):
    """Why the first cut found nothing -- pinned so it cannot come back.

    An ``nn.Module``'s children live in its dict and are served through
    ``__getattr__``, so they are absent from ``dir()``.  Discovery must go
    through the model's own traversal.
    """

    layer = _real_tree(num_layers=1).language_model.model.layers[0]
    assert isinstance(layer.attn_hyper_connection, qwen4_exp.GatedResidual)
    seen_by_dir = [
        name
        for name in dir(layer)
        if isinstance(getattr(layer, name, None), qwen4_exp.GatedResidual)
    ]
    assert seen_by_dir == []
    seen_by_named_modules = [
        name
        for name, module in layer.named_modules()
        if isinstance(module, qwen4_exp.GatedResidual)
    ]
    assert sorted(seen_by_named_modules) == [
        "attn_hyper_connection",
        "mlp_hyper_connection",
    ]


def test_pack_validation_names_the_quantized_layer(armed):
    model = _real_tree(num_layers=2)
    victim = model.language_model.model.layers[1].attn_hyper_connection
    victim.input_mix_weight_down.scales = mx.zeros((LOWRANK, HCD // 64), mx.bfloat16)
    with pytest.raises(
        RuntimeError, match=r"layers\.1\.attn_hyper_connection.*is quantized"
    ):
        qwen4_exp.install_hc_m4_pack_validation(model)


def test_pack_validation_names_the_wrong_weight_shape(armed):
    model = _real_tree(num_layers=1)
    victim = model.language_model.model.hyper_connection_mixer
    victim.input_mix_weight_down.weight = mx.zeros((256, HCD), dtype=mx.bfloat16)
    with pytest.raises(RuntimeError, match="input_mix_weight_down.weight"):
        qwen4_exp.install_hc_m4_pack_validation(model)


def test_pack_validation_refuses_a_model_with_nothing_to_replace(armed):
    """An armed flag that can never do anything is a deployment error."""

    import mlx.nn as nn

    class _NoHyperConnections(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(8, 8)

    with pytest.raises(RuntimeError, match="no GatedResidual"):
        qwen4_exp.install_hc_m4_pack_validation(_NoHyperConnections())


def test_pack_validation_does_not_check_the_activation_dtype(armed):
    """That term needs an activation; `_hc_m4_applies` still owns it.

    It is process-invariant too (one pack, one activation dtype), so it cannot
    single out one request either -- but it cannot be answered without a
    forward, so it stays where the forward is.
    """

    model = _real_tree(num_layers=1, with_mtp=False)
    assert qwen4_exp.install_hc_m4_pack_validation(model)["validated"] == 3
    with pytest.raises(ValueError, match="dtype"):
        _family_module(dtype=mx.float32)._hc_m4_applies(
            mx.zeros((1, 4, HCD), dtype=mx.bfloat16)
        )


def test_the_load_sequence_publishes_the_mtp_before_the_validation(armed):
    """Replays runtime.load's module wiring in the order runtime.py uses.

    The order is read out of `runtime.py` itself rather than restated, so the
    test fails if the hook is ever moved above the MTP injection or the qwen4
    installs -- which would put it back to validating a tree the model has not
    finished building.
    """

    import inspect

    from mtplx import runtime

    source = inspect.getsource(runtime.load)
    steps = (
        "inject_qwen4_exp_mtp_support(",  # publishes language_model.mtp
        "install_qwen4_fixed_verify_route(",
        "install_qwen4_m4_stage3(",
        "install_hc_m4_pack_validation(runtime.model)",
    )
    positions = [source.index(step) for step in steps]
    assert positions == sorted(positions), "runtime.load's qwen4 order moved"

    # Now the same order, for real: a model whose MTP is not yet published
    # exposes fewer modules than one whose MTP is.
    model = _real_tree(num_layers=2, with_mtp=False)
    before = qwen4_exp.install_hc_m4_pack_validation(model)["validated"]
    model.language_model.mtp = qwen4_exp.Qwen4ExpMTP(model.language_model.args)
    after = qwen4_exp.install_hc_m4_pack_validation(model)["validated"]
    assert (before, after) == (5, 8)


def test_the_runtime_validates_the_pack_at_install(armed):
    """The install hook is wired into the runtime's qwen4 section."""

    import inspect

    from mtplx import runtime

    source = inspect.getsource(runtime)
    assert "install_hc_m4_pack_validation(runtime.model)" in source
